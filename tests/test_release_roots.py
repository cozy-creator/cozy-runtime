"""A root named by its published release: the machine installs, resolves and prepares it.

Real: the Worker, SubmitMachineExecution, the release-root preparation, model resolution
against an HTTP catalog, the serving model placement, the durable journal and a TensorFS
store holding the model. Fakes (test_same_release_placements.Machine): the uv install, the
describe and census derives, device memory; and the host facts naming four H100s.
"""

from __future__ import annotations

import base64
import json
import tempfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal import canonical, hostfacts, storage_admission
from cozy_runtime.internal.worker import machine_model_defaults, machine_release_roots
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_diffusers_upload_configs import PROFILE, _registry, _repository
from test_same_release_placements import LANE, LOCKED, OWNER, PACKAGE, RELEASE, Machine
from upload_standins import HuggingFace, Supervisor

SLOT = "touch.models.model"


class Refused(Exception):
    def __init__(self, code: Any, detail: str, trailers: tuple[tuple[str, str], ...]) -> None:
        super().__init__(detail)
        self.code, self.trailers = code, dict(trailers)
        self.absent = [value for key, value in trailers if key == "cozy-absent-release"]


class Context:
    def __init__(self) -> None:
        self.trailers: tuple[tuple[str, str], ...] = ()

    def abort(self, code: Any, detail: str) -> None:
        raise Refused(code, detail, self.trailers)

    def set_trailing_metadata(self, metadata: tuple[tuple[str, str], ...]) -> None:
        self.trailers = metadata


class Catalog:
    """The owner's binding and one lane resolution, counted. A resolution without a length
    serves the checkpoint's manifest bytes, as `served`."""

    def __init__(
        self,
        manifest: str,
        length: int | None,
        ladder: list[dict[str, Any]],
        served: bytes = b"",
        together: threading.Barrier | None = None,
    ) -> None:
        self.reads: list[str] = []
        self.served = served
        catalog = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def do_GET(self) -> None:
                catalog.reads.append(self.path)
                url = urlsplit(self.path)
                if url.path == "/v1/packages/cozytest/tenant/bindings":
                    body: Any = {
                        "bindings": [
                            {"slot": SLOT, "model": PACKAGE, "release": RELEASE, "ladder": ladder}
                        ]
                    }
                elif url.path == "/v1/models/resolve":
                    query = parse_qs(url.query)
                    assert query == {"ref": [f"{PACKAGE}@{RELEASE}"], "lane": [LANE]}, query
                    if together is not None:
                        together.wait(30)  # each lane is read beside the others
                    body = {
                        "model": PACKAGE,
                        "release": RELEASE,
                        "lane": LANE,
                        "manifest_id": manifest,
                        **({} if length is None else {"manifest_length": length}),
                    }
                elif url.path == f"/v1/models/{PACKAGE}/checkpoints/{manifest}":
                    body = catalog.served
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-root.", dir="/tmp") as root:
        machine = Machine(Path(root), monkeypatch, "published")
        try:
            yield machine
        finally:
            machine.worker.shutdown()


CALLEE = "cozytest/callee"


def facts(machine: Machine, package: str = PACKAGE) -> pb.DeferredInstallation:
    return pb.DeferredInstallation(
        key=f"{package}@{RELEASE}",
        package=package,
        release=RELEASE,
        preparation=pb.PreparePackageSetRequest(
            download_delegation=canonical.write(
                {
                    "format": "cozy.worker.v1.DownloadDelegation/1",
                    "packages": [{"package": package, "release": RELEASE}],
                    "models": [],
                }
            ),
            application="tenant_workflow:app",
            locked_requirements=LOCKED,
            package_interface=machine.interface,
        ),
    )


def submission(
    machine: Machine, request: str, origin: str, *installs: Any
) -> pb.MachineExecutionSubmit:
    return pb.MachineExecutionSubmit(
        claim=machine.claim,
        submission_id=request,
        offer=pb.AttemptOffer(request_id=request),
        payload_canonical_bytes=canonical_json.encode({"seed": 3}),
        expected_execution_workspace_id=machine.executions.workspace_id,
        release_root=pb.ReleaseRoot(
            package=PACKAGE,
            release=RELEASE,
            entrypoint="touch",
            installations=installs,
            catalog_origin=origin,
            # A published callee the root may call: installed on its first selection.
            callees=[pb.ReleaseEdge(caller=f"{PACKAGE}@{RELEASE}", callee=f"{CALLEE}@{RELEASE}")],
        ),
    )


def submit(
    machine: Machine, request: pb.MachineExecutionSubmit
) -> tuple[pb.MachineExecutionReceipt, list[str]]:
    """Replay the identical submission until the machine answers a receipt."""
    progress: list[str] = []
    while True:
        try:
            return machine.rpc.SubmitMachineExecution(request, Context()), progress
        except Refused as refused:
            if refused.trailers.get("cozy-error-code") != "release_root_preparing":
                raise
            progress.append(str(refused))


def test_a_release_root_is_installed_resolved_and_prepared_by_the_machine(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    machine = published
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda *_: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    length = len(
        tensorfs.Store.open(str(machine.store_root)).manifest(machine.manifest)["manifest"]
    )
    catalog = Catalog(
        machine.manifest,
        length,
        [
            {"gpu": "B200", "lane": "fp4"},
            {"gpu": "H100", "gpus": 2, "lane": "two"},  # the slot declares no 2-GPU group
            {"gpu": "H100", "lane": LANE},
            {"gpu": "*", "lane": "cpu"},
        ],
    )
    installs: list[pb.PreparePackageSetRequest] = []
    prepare = machine.worker.prepare_package_set

    def counted(request: pb.PreparePackageSetRequest) -> Any:
        installs.append(request)
        return prepare(request)

    monkeypatch.setattr(machine.worker, "prepare_package_set", counted)
    try:
        # A machine holding no installation names the release whose facts it needs.
        with pytest.raises(Refused) as absent:
            machine.rpc.SubmitMachineExecution(
                submission(machine, "first", catalog.origin), Context()
            )
        assert absent.value.trailers["cozy-error-code"] == "release_root_installations_absent"
        assert absent.value.absent == [f"{CALLEE}@{RELEASE}", f"{PACKAGE}@{RELEASE}"]
        assert catalog.reads == []

        receipt, progress = submit(
            machine,
            submission(machine, "first", catalog.origin, facts(machine, CALLEE), facts(machine)),
        )
        assert receipt.request_id == "first" and len(installs) == 1
        assert progress and progress[0].startswith("installing ")
        # The owner's binding chose the H100 rung the slot can form; one lane was resolved.
        assert [urlsplit(path).path for path in catalog.reads] == [
            "/v1/packages/cozytest/tenant/bindings",
            "/v1/models/resolve",
        ]
        state = machine.executions.status(OWNER, "first")
        assert state.state == "queued"
        # The callee is captured deferred: installed only if the root selects it.
        capture = machine.executions.capture(OWNER, "first")
        assert [row["key"] for row in capture["deferred_installations"]] == [f"{CALLEE}@{RELEASE}"]
        assert {row.get("callee_deferred_key") for row in capture["bindings"]} == {
            None,
            f"{CALLEE}@{RELEASE}",
        }
        assert capture["catalog_origin"] == catalog.origin and "model_defaults" not in capture
        resolved = [
            canonical_json.decode(event.body)
            for event in machine.executions.events(OWNER, "first").events
            if event.kind == "resolved"
        ]
        assert resolved == [
            {
                "installation_id": state_installation(machine, "first"),
                "models": [
                    {
                        "gpus": 0,
                        "lane": LANE,
                        "manifest": {"digest": machine.manifest, "length": length},
                        "parameter": "model",
                        "release": RELEASE,
                        "repository": PACKAGE,
                    }
                ],
                "package": PACKAGE,
                "release": RELEASE,
            }
        ]
        preparation = machine.executions.preparation(OWNER, "first")
        desired = pb.DesiredWorkerState.FromString(base64.b64decode(preparation["state"]))
        assert desired.placement_set.execution_gpus == 0

        # The identical warm root: no facts (the callee's are remembered), no install, one call.
        catalog.reads.clear()
        receipt = machine.rpc.SubmitMachineExecution(
            submission(machine, "second", catalog.origin), Context()
        )
        assert receipt.request_id == "second" and len(installs) == 1
        assert len(catalog.reads) == 2
        replay = machine.rpc.SubmitMachineExecution(
            submission(machine, "second", catalog.origin), Context()
        )
        assert replay == receipt
        # A checkpoint named by digest alone that this machine holds costs no catalog read.
        catalog.reads.clear()
        pinned = submission(machine, "pinned", catalog.origin)
        pinned.release_root.models.add(
            parameter="model",
            repository=PACKAGE,
            manifest=pb.Ref(digest=documents.raw(machine.manifest)),
        )
        assert machine.rpc.SubmitMachineExecution(pinned, Context()).request_id == "pinned"
        assert catalog.reads == []
        assert [
            canonical_json.decode(event.body)["models"][0]["manifest"]
            for event in machine.executions.events(OWNER, "pinned").events
            if event.kind == "resolved"
        ] == [{"digest": machine.manifest, "length": length}]
        # An observer holds one read open until the next event, instead of polling.
        head = machine.executions.events(OWNER, "second").head_sequence
        query = pb.MachineExecutionEventsQuery(
            execution=pb.MachineExecutionQuery(
                claim=machine.claim,
                request_id="second",
                expected_execution_workspace_id=machine.executions.workspace_id,
            ),
            after=head,
            wait=True,
        )
        answered: list[pb.MachineExecutionEventPage] = []
        reader = threading.Thread(
            target=lambda: answered.append(machine.rpc.ListMachineExecutionEvents(query, Context()))
        )
        reader.start()
        reader.join(0.5)
        assert reader.is_alive() and not answered
        machine.executions.record(OWNER, "second", "observed", {"note": 1})
        reader.join(30)
        assert [event.kind for event in answered[0].events] == ["observed"]
        machine.tick()
        events = [
            (event.kind, canonical_json.decode(event.body))
            for event in machine.executions.events(OWNER, "first").events
        ]
        if machine.executions.status(OWNER, "first").state == "failed":
            body = documents.read(
                machine.executions.collect(OWNER, "first").outcome_canonical_bytes,
                pb.AttemptOutcomeBody,
            )
            raise AssertionError(body.get("safe_message"), body.get("cause"))
        assert machine.executions.status(OWNER, "first").state in ("queued", "running"), events
    finally:
        catalog.close()


def test_an_unstated_manifest_length_is_measured_from_the_bytes_its_digest_names(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A release+lane resolution may omit `manifest_length` (th-233): the machine reads the
    checkpoint's manifest, verifies it by digest and runs; bytes the digest does not name
    refuse the root."""
    machine = published
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda *_: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    raw = tensorfs.Store.open(str(machine.store_root)).manifest(machine.manifest)["manifest"]
    catalog = Catalog(machine.manifest, None, [{"gpu": "H100", "lane": LANE}], raw)
    try:
        receipt, _ = submit(
            machine,
            submission(machine, "first", catalog.origin, facts(machine, CALLEE), facts(machine)),
        )
        assert receipt.request_id == "first"
        assert [urlsplit(path).path for path in catalog.reads] == [
            "/v1/packages/cozytest/tenant/bindings",
            "/v1/models/resolve",
            f"/v1/models/{PACKAGE}/checkpoints/{machine.manifest}",
        ]
        (models,) = [
            canonical_json.decode(event.body)["models"]
            for event in machine.executions.events(OWNER, "first").events
            if event.kind == "resolved"
        ]
        assert models[0]["manifest"] == {"digest": machine.manifest, "length": len(raw)}
        machine.tick()
        assert machine.executions.status(OWNER, "first").state in ("queued", "running")

        catalog.served = raw + b" "
        with pytest.raises(Refused, match="manifest bytes do not match"):
            submit(machine, submission(machine, "forged", catalog.origin))
    finally:
        catalog.close()


def state_installation(machine: Machine, request: str) -> str:
    offer = machine.executions.offer(OWNER, request)
    spec = documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
    return str(spec["installation_id"])


def test_an_open_child_slot_is_resolved_by_the_machine_at_the_capture_origin() -> None:
    manifest = "sha256:" + "22" * 32
    catalog = Catalog(manifest, 321, [{"gpu": "*", "lane": LANE}])
    try:
        worker = cast(
            Worker,
            SimpleNamespace(
                options=SimpleNamespace(accelerator_backend="none"),
                host_facts=lambda: hostfacts.measure("none"),
                executions=SimpleNamespace(
                    capture_root=lambda *_: "root",
                    capture=lambda *_: {"catalog_origin": catalog.origin, "model_defaults": []},
                ),
            ),
        )
        call = cast(Call, SimpleNamespace(parent_request="root"))

        def child(package: str) -> Target:
            model = {
                "path": SLOT,
                "default_ladder": [{"gpu": "*", "lane": f"tenant@{RELEASE}/{LANE}"}],
            }
            return Target(
                "installation",
                "touch",
                {"models": [model]},
                {"placement": {"package": {"package": package, "release": RELEASE}}},
                None,
                "",
            )

        expected = {
            "parameter": "model",
            "public_origin": catalog.origin,
            "gpu": "*",
            "gpus": 0,
            "repository": PACKAGE,
            "manifest": {"digest": manifest, "length": 321},
            "release": RELEASE,
            "lane": LANE,
        }
        # The owner's binding of the callee's slot.
        assert machine_model_defaults.select(worker, "owner", call, child(PACKAGE), {}) == [
            expected
        ]
        # No binding on the catalog: the callee's authored default, its org its owner's.
        catalog.reads.clear()
        assert machine_model_defaults.select(
            worker, "owner", call, child("cozytest/other"), {}
        ) == [expected]
        assert [urlsplit(path).path for path in catalog.reads] == [
            "/v1/packages/cozytest/other/bindings",
            "/v1/models/resolve",
        ]
        # A supplied Model is never resolved.
        catalog.reads.clear()
        assert (
            machine_model_defaults.select(worker, "owner", call, child(PACKAGE), {"model": {}})
            == []
        )
        assert catalog.reads == []
    finally:
        catalog.close()


def test_a_provider_source_choice_is_made_by_the_machine_once(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--model model=hf://…@<commit>`: the machine resolves the headers, selects the profile,
    lands only its members and converts them; an identical root touches no provider."""
    machine = published
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda *_: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    origin = HuggingFace(_repository(modular=False))
    source = "hf://example/diffusers@" + "c" * 40
    calls = machine.worker.source_calls
    assert calls is not None
    supervisor = Supervisor(machine.store_root, 1 << 30, 0)
    with origin.running() as hf, supervisor.running() as channel:
        monkeypatch.setattr(storage_admission, "_channel", channel)
        monkeypatch.setattr(calls, "endpoints", {"huggingface": hf})
        monkeypatch.setattr(calls, "native_registry", _registry())

        def sourced(request: str, *installs: Any) -> pb.MachineExecutionSubmit:
            submitted = submission(machine, request, "", *installs)
            submitted.release_root.models.append(pb.ModelChoice(parameter="model", source=source))
            return submitted

        receipt, _ = submit(machine, sourced("first", facts(machine, CALLEE), facts(machine)))
        assert receipt.request_id == "first"
        (resolved,) = [
            canonical_json.decode(event.body)["models"]
            for event in machine.executions.events(OWNER, "first").events
            if event.kind == "resolved"
        ]
        (model,) = resolved
        assert model["source"] == source and model["profiles"] == [PROFILE]
        assert model["resolved"].startswith("hf://example/diffusers@" + "c" * 40)
        assert model["repository"].startswith("local/source-")
        store = tensorfs.Store.open(str(machine.store_root))
        store.verify_checkpoint_source(
            model["repository"], model["manifest"]["digest"], model["manifest"]["length"]
        )
        # The profile's carriers and the pipeline's configs were read; the tokenizer never.
        assert {
            "encoder/diffusion_pytorch_model.safetensors",
            "decoder/diffusion_pytorch_model.safetensors",
        } <= origin.touched
        assert not any(member.startswith("tokenizer/") for member in origin.touched)
        cold = origin.requests

        receipt, _ = submit(machine, sourced("second"))
        assert receipt.request_id == "second"
        assert origin.requests == cold, "a warm identical root read the provider again"
        (again,) = [
            canonical_json.decode(event.body)["models"]
            for event in machine.executions.events(OWNER, "second").events
            if event.kind == "resolved"
        ]
        assert again == resolved


def test_a_published_job_is_a_release_root_the_machine_mints(published: Machine) -> None:
    """A job by its release: the machine takes its declaration, mints the job invocation and
    its directive, and names the owner's publication grant; nothing is prepared per call."""
    machine = published
    root = pb.ReleaseRoot(
        package=PACKAGE,
        release=RELEASE,
        entrypoint="workflow",
        job=True,
        publication_grant="cozytest/_job-jobbed",
        installations=[facts(machine)],
    )
    request = pb.MachineExecutionSubmit(
        claim=machine.claim,
        submission_id="jobbed",
        offer=pb.AttemptOffer(request_id="jobbed"),
        payload_canonical_bytes=canonical_json.encode({"shots": 2}),
        expected_execution_workspace_id=machine.executions.workspace_id,
        release_root=root,
    )
    receipt, _ = submit(machine, request)
    assert receipt.request_id == "jobbed"
    assert machine.executions.status(OWNER, "jobbed").state == "queued"
    offer = machine.executions.offer(OWNER, "jobbed")
    spec = documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
    assert spec["job"]["job_descriptor_id"].startswith("sha256:")
    assert spec["job"]["publication_contract"]["grant_id"] == "cozytest/_job-jobbed"
    assert "serving" not in spec and not offer.placement_id
    desired = pb.DesiredWorkerState.FromString(
        base64.b64decode(machine.executions.preparation(OWNER, "jobbed")["state"])
    )
    # No declared accelerator and no torch in its lock: a CPU orchestration job.
    assert desired.job.orchestration and not desired.job.device_count
    assert desired.job.job_descriptor_id == spec["job"]["job_descriptor_id"]


def test_a_roots_catalog_slots_resolve_beside_each_other(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The package's bindings are read once; its slots' lanes are then read at once."""
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda *_: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    ladder = [{"gpu": "*", "lane": LANE}]
    catalog = Catalog(published.manifest, 1, ladder, together=threading.Barrier(2))
    root = pb.ReleaseRoot(
        package=PACKAGE,
        release=RELEASE,
        entrypoint="touch",
        catalog_origin=catalog.origin,
        models=[
            pb.ModelChoice(parameter=parameter, repository=PACKAGE, release=RELEASE, lane=LANE)
            for parameter in ("model", "second")
        ],
    )
    declaration = {"models": [{"path": SLOT}, {"path": "touch.models.second"}]}
    try:
        rows = machine_release_roots.ReleaseRoots(published.worker)._resolve(
            root, declaration, {}, lambda *_: None
        )
    finally:
        catalog.close()
    assert [row["parameter"] for row in rows] == ["model", "second"]
    assert [urlsplit(path).path for path in catalog.reads] == [
        "/v1/packages/cozytest/tenant/bindings",
        "/v1/models/resolve",
        "/v1/models/resolve",
    ]
