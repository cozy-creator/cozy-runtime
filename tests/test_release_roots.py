"""A root named by its published release: the machine installs, resolves and prepares it.

Real: the Worker, SubmitMachineExecution, the release-root preparation, install facts, callees
and model resolution read at an HTTP catalog (the machine's own Hub), the serving model
placement, the durable journal and a TensorFS store holding the model. Fakes
(test_same_release_placements.Machine): the uv install, the describe and census derives,
device memory; and the host facts naming four H100s.
"""

from __future__ import annotations

import base64
import dataclasses
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
from cozy_runtime.internal.worker import (
    machine_model_defaults,
    machine_release_roots,
    package_prepare,
)
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_diffusers_upload_configs import PROFILE, _registry, _repository
from test_same_release_placements import (
    LANE,
    LOCKED,
    OWNER,
    PACKAGE,
    RELEASE,
    Context,
    Machine,
    Refused,
)
from upload_standins import HuggingFace, Supervisor

SLOT = "touch.models.model"


CALLEE = "cozytest/callee"


class Catalog:
    """The machine's own Hub, counted: published releases (their interface and lock), the
    owner's binding and one lane resolution. A resolution without a length serves the
    checkpoint's manifest bytes, as `served`."""

    def __init__(
        self,
        manifest: str,
        length: int | None,
        ladder: list[dict[str, Any]],
        served: bytes = b"",
        together: threading.Barrier | None = None,
        interface: bytes = b"",
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
                release = catalog.releases.get(url.path)
                if release is not None:
                    body: Any = release
                elif url.path == "/v1/packages/cozytest/tenant/bindings":
                    body = {
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
        # The tenant's lock names one published callee on its org's index; the callee's none.
        wheel = f"sha256:{'b' * 64}/callee-{RELEASE}-py3-none-any.whl"
        callee = f"{self.origin}/v1/index/cozytest/files/{wheel}"
        locks = {
            PACKAGE: LOCKED + f"callee @ {callee} --hash=sha256:{'b' * 64}\n".encode(),
            CALLEE: b"callee==1.0.0 --hash=sha256:" + b"b" * 64 + b"\n",
        }
        self.releases: dict[str, Any] = {}
        for package, lock in locks.items():
            path = f"/v1/packages/{package}"
            self.releases[path] = {"releases": [{"release": "0.9.0"}, {"release": RELEASE}]}
            self.releases[f"{path}/releases/{RELEASE}"] = {
                "release": {"release": RELEASE},
                "package_interface": json.loads(interface or b"{}"),
            }
            self.releases[f"{path}/releases/{RELEASE}/locked-requirements"] = lock
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def serve(self, machine: Machine) -> None:
        """Make this catalog the machine's own Hub, as its grant names it."""
        authority = PublicationAuthority(self.origin, "worker", "token")
        worker = machine.worker
        worker.options = dataclasses.replace(worker.options, publication_authority=authority)

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


def submission(machine: Machine, request: str) -> pb.MachineExecutionSubmit:
    """What a controller sends: the release and its function, nothing it looked up."""
    return pb.MachineExecutionSubmit(
        claim=machine.claim,
        submission_id=request,
        offer=pb.AttemptOffer(request_id=request),
        payload_canonical_bytes=canonical_json.encode({"seed": 3}),
        expected_execution_workspace_id=machine.executions.workspace_id,
        release_root=pb.ReleaseRoot(package=PACKAGE, release=RELEASE, entrypoint="touch"),
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
        interface=machine.interface,
    )
    catalog.serve(machine)
    installs: list[pb.PreparePackageSetRequest] = []
    prepare = machine.worker.prepare_package_set

    def counted(request: pb.PreparePackageSetRequest) -> Any:
        installs.append(request)
        return prepare(request)

    monkeypatch.setattr(machine.worker, "prepare_package_set", counted)
    try:
        # A machine holding no installation reads the release, and the callee its lock
        # names, at its own Hub; the controller sent only the release and function.
        receipt, progress = submit(machine, submission(machine, "first"))
        assert receipt.request_id == "first" and len(installs) == 1
        assert installs[0].locked_requirements.startswith(LOCKED)
        assert installs[0].package_interface == machine.interface
        assert progress and progress[0].startswith("installing ")
        # The owner's binding chose the H100 rung the slot can form; one lane was resolved.
        tenant, callee = (
            f"/v1/packages/{PACKAGE}/releases/{RELEASE}",
            f"/v1/packages/{CALLEE}/releases/{RELEASE}",
        )
        assert [urlsplit(path).path for path in catalog.reads] == [
            tenant,
            tenant + "/locked-requirements",
            callee,
            callee + "/locked-requirements",
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
        assert "catalog_origin" not in capture and "model_defaults" not in capture
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

        # The identical warm root: no release read (each is immutable), no install, one call.
        catalog.reads.clear()
        receipt = machine.rpc.SubmitMachineExecution(submission(machine, "second"), Context())
        assert receipt.request_id == "second" and len(installs) == 1
        assert len(catalog.reads) == 2
        replay = machine.rpc.SubmitMachineExecution(submission(machine, "second"), Context())
        assert replay == receipt
        # A checkpoint named by digest alone that this machine holds costs no catalog read.
        catalog.reads.clear()
        pinned = submission(machine, "pinned")
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
    catalog = Catalog(
        machine.manifest, None, [{"gpu": "H100", "lane": LANE}], raw, interface=machine.interface
    )
    catalog.serve(machine)
    try:
        receipt, _ = submit(machine, submission(machine, "first"))
        assert receipt.request_id == "first"
        assert [urlsplit(path).path for path in catalog.reads][4:] == [
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
            submit(machine, submission(machine, "forged"))
    finally:
        catalog.close()


def state_installation(machine: Machine, request: str) -> str:
    offer = machine.executions.offer(OWNER, request)
    spec = documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
    return str(spec["installation_id"])


def test_an_open_child_slot_is_resolved_by_the_machine_at_its_own_hub() -> None:
    manifest = "sha256:" + "22" * 32
    catalog = Catalog(manifest, 321, [{"gpu": "*", "lane": LANE}])
    try:
        worker = cast(
            Worker,
            SimpleNamespace(
                options=SimpleNamespace(
                    accelerator_backend="none",
                    publication_authority=PublicationAuthority(catalog.origin, "worker", "token"),
                ),
                host_facts=lambda: hostfacts.measure("none"),
                executions=SimpleNamespace(
                    capture_root=lambda *_: "root",
                    capture=lambda *_: {"model_defaults": []},
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
    catalog = Catalog(machine.manifest, 1, [], interface=machine.interface)
    catalog.serve(machine)
    with origin.running() as hf, supervisor.running() as channel:
        monkeypatch.setattr(storage_admission, "_channel", channel)
        monkeypatch.setattr(calls, "endpoints", {"huggingface": hf})
        monkeypatch.setattr(calls, "native_registry", _registry())

        def sourced(request: str) -> pb.MachineExecutionSubmit:
            submitted = submission(machine, request)
            submitted.release_root.models.append(pb.ModelChoice(parameter="model", source=source))
            return submitted

        receipt, _ = submit(machine, sourced("first"))
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
    catalog.close()


def test_a_published_job_is_a_release_root_the_machine_mints(published: Machine) -> None:
    """A job by its release: the machine takes its declaration, mints the job invocation and
    its directive, and names the owner's publication grant; nothing is prepared per call."""
    machine = published
    catalog = Catalog(machine.manifest, 1, [], interface=machine.interface)
    catalog.serve(machine)
    root = pb.ReleaseRoot(
        package=PACKAGE,
        release=RELEASE,
        entrypoint="workflow",
        job=True,
        publication_grant="cozytest/_job-jobbed",
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
    catalog.close()
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
    catalog.serve(published)
    root = pb.ReleaseRoot(
        package=PACKAGE,
        release=RELEASE,
        entrypoint="touch",
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


def test_the_machine_describes_a_release_from_its_own_hub(published: Machine) -> None:
    """A controller asks the machine, never the Hub, for a release's interface: the newest
    release when it names none, read once; an installed release answers from its install."""
    machine = published
    catalog = Catalog(machine.manifest, 1, [], interface=machine.interface)
    catalog.serve(machine)

    def describe(release: str = "") -> pb.DescribedRelease:
        query = pb.MachineExecutionWorkspaceQuery(
            claim=machine.claim, describe=pb.PackageSelection(package=PACKAGE, release=release)
        )
        workspace = machine.rpc.GetMachineExecutionWorkspace(query, Context())
        described: pb.DescribedRelease = workspace.described_release
        return described

    try:
        described = describe()
        assert (described.package, described.release) == (PACKAGE, RELEASE)
        assert described.package_interface == machine.interface
        assert describe(RELEASE) == described
        tenant = f"/v1/packages/{PACKAGE}"
        assert [urlsplit(path).path for path in catalog.reads] == [
            tenant,
            f"{tenant}/releases/{RELEASE}",
            f"{tenant}/releases/{RELEASE}/locked-requirements",
        ]
        with pytest.raises(Refused, match="not a published release"):
            describe("9.9.9")
        # Without a query, the workspace describes nothing and reads nothing.
        catalog.reads.clear()
        plain = machine.rpc.GetMachineExecutionWorkspace(
            pb.MachineExecutionWorkspaceQuery(claim=machine.claim), Context()
        )
        assert not plain.HasField("described_release") and catalog.reads == []
    finally:
        catalog.close()


def test_a_factless_preparation_is_completed_from_the_machines_own_hub(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preparation naming only a release (an install or a deployment) takes its lock and
    interface from the machine's own Hub; the controller looked nothing up."""
    machine = published
    catalog = Catalog(machine.manifest, 1, [], interface=machine.interface)
    catalog.serve(machine)
    seen: list[pb.PreparePackageSetRequest] = []
    install = package_prepare.prepare_package_set

    def recorded(request: pb.PreparePackageSetRequest, **seams: Any) -> Any:
        seen.append(request)
        return install(request, **seams)

    monkeypatch.setattr(package_prepare, "prepare_package_set", recorded)
    delegation = {
        "format": "cozy.worker.v1.DownloadDelegation/1",
        "models": [],
        "packages": [{"package": PACKAGE, "release": RELEASE}],
    }
    try:
        machine.worker.prepare_package_set(
            pb.PreparePackageSetRequest(
                download_delegation=canonical.write(delegation),
                install_root=str(machine.install_root),
            )
        )
    finally:
        catalog.close()
    (sent,) = seen
    assert sent.locked_requirements.startswith(LOCKED)
    assert sent.package_interface == machine.interface
    assert sent.application == "tenant_workflow:app"
