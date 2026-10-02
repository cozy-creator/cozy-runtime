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
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal import (
    canonical,
    hostfacts,
    package_environment,
    package_installation,
    storage_admission,
)
from cozy_runtime.internal.worker import (
    machine_model_defaults,
    machine_model_resolve,
    machine_release_roots,
    package_prepare,
)
from cozy_runtime.internal.worker.machine_child_target import Target
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.internal.worker.workspace_rpc import WorkspaceRPC
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_diffusers_upload_configs import PROFILE, _registry, _repository
from test_model_runtime_closure import _ASSET, _HEADER, _ref
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
        # Private: the Models are the owner's unpublished checkpoints, read only with the
        # machine's worker capability (absent: 404, as if absent; another: 401).
        self.private = False
        self.grant = ("worker", "token")  # the machine's registration at this Hub
        self.presented: list[tuple[str, str, str]] = []
        self.closures: list[str] = []  # each checkpoint TensorFS asked this Hub to name
        catalog = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def do_POST(self) -> None:
                asked = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert self.path == "/v1/tensorfs/closure", self.path  # nothing is presigned
                catalog.closures.append(asked["ref"])
                raw = json.dumps(
                    {
                        "complete": True,
                        "lane": "",
                        "release": "",
                        "model": asked["ref"].partition("@")[0],
                        "manifest": {"sha256": manifest.removeprefix("sha256:"), "length": length},
                        "objects": sorted(
                            (_ref(_HEADER), _ref(_ASSET)), key=lambda row: row["sha256"]
                        ),
                        "presign_max_digests": 10,
                        "scope": "runtime",
                        "server_time_unix": int(time.time()),
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                catalog.reads.append(self.path)
                url = urlsplit(self.path)
                presented = (
                    self.headers.get("X-Cozy-Worker-ID", ""),
                    self.headers.get("X-Cozy-Worker-Token", ""),
                )
                if url.path.startswith("/v1/models/"):
                    catalog.presented.append((url.path, *presented))
                    if catalog.private and presented != catalog.grant:
                        error = b'{"error":{"code":"worker.ended","message":"worker ended"}}'
                        self.send_response(401 if presented[0] else 404)
                        self.send_header("Content-Length", str(len(error)))
                        self.end_headers()
                        self.wfile.write(error)
                        return
                release = catalog.releases.get(url.path)
                if release is not None:
                    body: Any = release
                elif url.path == "/v1/packages/cozytest/tenant/bindings":
                    body = {
                        "bindings": [
                            {"slot": SLOT, "model": PACKAGE, "release": RELEASE, "ladder": ladder}
                        ]
                    }
                elif url.path == f"/v1/models/{PACKAGE}":
                    # Its newest release holds full precision beside a smaller lane.
                    lanes = [{"lane": "fp8", "bytes": 1}, {"lane": LANE, "bytes": 2}]
                    body = {
                        "releases": [
                            {"release": "0.9.0", "lanes": [{"lane": "fp8", "bytes": 1}]},
                            {"release": RELEASE, "lanes": lanes},
                        ]
                    }
                elif url.path == "/v1/models/resolve":
                    query = parse_qs(url.query)
                    # The Hub folds names: any spelling of the model resolves to its own.
                    query["ref"] = [ref.lower() for ref in query["ref"]]
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
                    missing = {"code": "route.not_found", "message": "no route GET " + url.path}
                    raw = json.dumps({"error": missing}).encode()
                    self.send_response(404)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
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

    def serve(self, machine: Machine, token: str = "token") -> None:
        """Make this catalog the machine's own Hub, as its grant names it."""
        authority = PublicationAuthority(self.origin, "worker", token)
        worker = machine.worker
        worker.options = dataclasses.replace(worker.options, publication_authority=authority)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class _Held(dict[str, threading.Event]):
    """Every root `Machine` dispatches is held on its grant, as it holds a job."""

    def __contains__(self, _request: object) -> bool:
        return True

    def __missing__(self, request: str) -> threading.Event:
        return self.setdefault(request, threading.Event())


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-root.", dir="/tmp") as root:
        machine = Machine(Path(root), monkeypatch, "published")
        # This venv holds no Runtime and no torch to boot a model's executor from.
        machine.running = _Held()
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
            f"/v1/packages/{CALLEE}",
            callee,
            callee + "/locked-requirements",
            "/v1/packages/cozytest/tenant/bindings",
            "/v1/models/resolve",
        ]
        state = machine.executions.status(OWNER, "first")
        assert state.state in ("queued", "running")
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
        # Each admission step's wall time rides with the run, for `cozy run show`.
        admission = resolved[0].pop("admission_ms")
        assert admission.keys() == {
            "closure",
            "install",
            "capture",
            "resolve",
            "prepare",
            "builtins",
            "submit",
            "total",
        }
        assert admission["total"] >= admission["install"] + admission["prepare"]
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

        # The identical warm root reads nothing: its release, the owner's binding and the
        # lane were read once for this installation. No install, one call.
        catalog.reads.clear()
        receipt = machine.rpc.SubmitMachineExecution(submission(machine, "second"), Context())
        assert receipt.request_id == "second" and len(installs) == 1
        assert catalog.reads == []
        replay = machine.rpc.SubmitMachineExecution(submission(machine, "second"), Context())
        assert replay == receipt

        # The owner rebinds: the controller that did names the package, and the next root
        # reads the binding and its lane once more. An explicit choice reads no binding.
        class Forget(WorkspaceRPC):
            workspace_service = machine.worker.workspace_service

        def forget() -> None:
            call = pb.ForgetPackageCall(claim=machine.claim, package=PACKAGE)
            Forget().WorkspaceForgetPackage(call, Context())

        forget()
        catalog.reads.clear()
        submit(machine, submission(machine, "rebound"))
        assert [urlsplit(path).path for path in catalog.reads] == [
            "/v1/packages/cozytest/tenant/bindings",
            "/v1/models/resolve",
        ]
        forget()
        catalog.reads.clear()
        chosen = submission(machine, "chosen")
        chosen.release_root.models.add(
            parameter="model", repository=PACKAGE.title(), release=RELEASE, lane=LANE
        )
        submit(machine, chosen)
        assert [urlsplit(path).path for path in catalog.reads] == ["/v1/models/resolve"]
        # Run 1587: a repository named with no lane, for a slot that ranks none, runs its
        # newest release's full-precision lane, and says which it picked.
        forget()
        catalog.reads.clear()
        bare = submission(machine, "bare")
        bare.release_root.models.add(parameter="model", repository=PACKAGE)
        submit(machine, bare)
        assert [urlsplit(path).path for path in catalog.reads] == [
            f"/v1/models/{PACKAGE}",
            "/v1/models/resolve",
        ]
        picked = [
            canonical_json.decode(event.body)["models"][0]
            for event in machine.executions.events(OWNER, "bare").events
            if event.kind == "resolved"
        ]
        assert [(row["release"], row["lane"]) for row in picked] == [(RELEASE, LANE)]
        assert [
            canonical_json.decode(event.body)["models"][0]["repository"]
            for event in machine.executions.events(OWNER, "chosen").events
            if event.kind == "resolved"
        ] == [PACKAGE]
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
        # An observer holds one read open until the next event, instead of polling. Each
        # call placed on its own GPU imported ahead there; those processes (which cannot dial
        # back here) say so on their calls first.
        bound = time.monotonic() + 120  # a hang bound on a loaded box, not a budget
        while machine.worker.prespawns.slots:
            assert time.monotonic() < bound
            time.sleep(0.01)
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
        assert machine.executions.status(OWNER, "first").state == "running", events
    finally:
        catalog.close()


def test_a_run_reads_the_hub_it_came_from(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A machine registered at two Hubs reads a run's release, callees and Models at the Hub
    the run names, with its registration there; its default Hub reads nothing, and what it
    installed for one Hub is not the other's. A Hub it is not registered at is refused."""
    machine = published
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda *_: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    length = len(
        tensorfs.Store.open(str(machine.store_root)).manifest(machine.manifest)["manifest"]
    )
    ladder = [{"gpu": "H100", "lane": LANE}]
    home = Catalog(machine.manifest, length, ladder, interface=machine.interface)
    other = Catalog(machine.manifest, length, ladder, interface=machine.interface)
    other.private, other.grant = True, ("other-worker", "other-token")
    home.serve(machine)
    worker = machine.worker
    worker.options = dataclasses.replace(
        worker.options, hubs=(PublicationAuthority(other.origin, *other.grant),)
    )

    def paths(catalog: Catalog) -> list[str]:
        return [urlsplit(path).path for path in catalog.reads]

    try:
        there = submission(machine, "there")
        there.release_root.hub = other.origin + "/"
        assert submit(machine, there)[0].request_id == "there"
        assert home.reads == [] and "/v1/models/resolve" in paths(other)
        assert {(who, token) for _, who, token in other.presented} == {other.grant}
        assert machine.executions.preparation(OWNER, "there")["hub"] == other.origin
        # Warm at that Hub: nothing is read anywhere.
        other.reads.clear()
        again = submission(machine, "again")
        again.release_root.hub = other.origin
        assert submit(machine, again)[0].request_id == "again"
        assert other.reads == [] and home.reads == []
        query = pb.MachineExecutionWorkspaceQuery(
            claim=machine.claim, describe=pb.PackageSelection(package=PACKAGE, hub=other.origin)
        )
        described = machine.rpc.GetMachineExecutionWorkspace(query, Context()).described_release
        assert described.release == RELEASE and home.reads == []

        # The same release from the default Hub is that Hub's: read and installed there.
        assert submit(machine, submission(machine, "home"))[0].request_id == "home"
        assert f"/v1/packages/{PACKAGE}/releases/{RELEASE}" in paths(home)
        assert "hub" not in machine.executions.preparation(OWNER, "home")

        elsewhere = submission(machine, "elsewhere")
        elsewhere.release_root.hub = "https://elsewhere.example"
        with pytest.raises(Refused, match="no execution access for") as refused:
            submit(machine, elsewhere)
        assert refused.value.trailers.get("cozy-error-code") == "execution_submission_refused"
        assert machine.executions.submission_absent(
            OWNER, "elsewhere", machine.executions.workspace_id
        )

        # Expired delegated access is likewise visible and proven unaccepted. Its
        # underlying typed error cannot replace the submission disposition trailer.
        expired_origin = "https://expired.example"
        worker.options = dataclasses.replace(
            worker.options,
            hubs=(
                PublicationAuthority(expired_origin, access_token="expired-access", expires_at=1),
            ),
        )
        expired = submission(machine, "expired")
        expired.release_root.hub = expired_origin
        with pytest.raises(Refused, match="execution access has expired") as refused:
            submit(machine, expired)
        assert refused.value.trailers.get("cozy-error-code") == "execution_submission_refused"
        assert machine.executions.submission_absent(
            OWNER, "expired", machine.executions.workspace_id
        )
        worker.options = dataclasses.replace(
            worker.options,
            hubs=(
                PublicationAuthority(
                    expired_origin, access_token="another-account", expires_at=int(time.time()) + 60
                ),
            ),
        )
        changed = submission(machine, "changed-account")
        changed.release_root.hub = expired_origin
        with pytest.raises(Refused, match="another account") as refused:
            submit(machine, changed)
        assert refused.value.trailers.get("cozy-error-code") == "execution_submission_refused"
        assert machine.executions.submission_absent(
            OWNER, "changed-account", machine.executions.workspace_id
        )
    finally:
        home.close()
        other.close()


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
        assert [urlsplit(path).path for path in catalog.reads][5:] == [
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

        # Measured afresh (a rebinding), bytes the digest does not name refuse the root.
        catalog.served = raw + b" "
        machine.worker.resolutions.forget(PACKAGE)
        with pytest.raises(Refused, match="manifest bytes do not match"):
            submit(machine, submission(machine, "forged"))
    finally:
        catalog.close()


def state_installation(machine: Machine, request: str) -> str:
    offer = machine.executions.offer(OWNER, request)
    spec = documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
    return str(spec["installation_id"])


def test_a_placement_built_against_another_runtime_is_prepared_again(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held placement whose environment chose its SDK against another machine Runtime (the
    machine was updated since) is not reused as it is: the root prepares it again, and its
    installation follows this Runtime (`package_installation.refresh`)."""
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
        machine.manifest, length, [{"gpu": "H100", "lane": LANE}], interface=machine.interface
    )
    catalog.serve(machine)
    installs: list[pb.PreparePackageSetRequest] = []
    prepare = machine.worker.prepare_package_set

    def counted(request: pb.PreparePackageSetRequest) -> Any:
        installs.append(request)
        return prepare(request)

    monkeypatch.setattr(machine.worker, "prepare_package_set", counted)
    try:
        submit(machine, submission(machine, "before"))
        submit(machine, submission(machine, "warm"))
        assert len(installs) == 1, "a current placement is reused"
        # The installation keeps its lock and was built against another machine Runtime.
        directory = machine.install_root / "installations" / state_installation(machine, "before")
        (directory / "requirements.txt").write_bytes(LOCKED)
        record = json.loads((directory / "installation.json").read_text())
        record["machine"] = {"cozy-runtime": "0.0.1"}
        (directory / "installation.json").write_text(json.dumps(record))
        submit(machine, submission(machine, "after-update"))
        assert len(installs) == 2, "a placement built against another Runtime was reused"
    finally:
        catalog.close()


def _owned(machine: Machine) -> package_installation.InstalledEnvironment:
    """The release as a machine owns a published installation: its generation inside its
    directory, its lock retained, built against this machine's SDK."""
    identifier = package_installation.PUBLISHED_PREFIX + "owned"
    directory = machine.install_root / "installations" / identifier
    directory.mkdir(parents=True)
    generation = directory / "venv"
    generation.symlink_to(machine.installed.generation, target_is_directory=True)
    (directory / "requirements.txt").write_bytes(LOCKED)
    site = machine.installed.site_packages.relative_to(machine.installed.generation)
    record = {
        "package": PACKAGE,
        "release": RELEASE,
        "python": str(generation / "bin" / "python"),
        "generation": str(generation),
        "site_packages": str(generation / site),
        "machine": package_installation.machine_sdk(),
    }
    (directory / "installation.json").write_text(json.dumps(record))
    return package_installation.open_installation(machine.install_root, identifier)


def test_a_restarted_runtime_holds_the_releases_it_prepared(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published root prepared once outlives a Runtime restart: the first run after it
    reads nothing at the Hub, installs nothing and checks no model again. What is kept
    belongs to the installation's incarnation: a rebuilt installation is prepared anew."""
    machine = published
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda *_: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    owned = _owned(machine)
    installs: list[str] = []

    def install(request: pb.PreparePackageSetRequest, **seams: Any) -> Any:
        installs.append(request.hub)
        models = documents.read(request.download_delegation, pb.DownloadDelegation)["models"]
        return package_prepare._prepare_published(
            package_name=PACKAGE,
            release=RELEASE,
            locked=package_environment.read_locked_requirements(LOCKED),
            models=package_prepare.selections(models),
            artifact_cache=seams["artifact_cache"],
            tensorfs_root=seams["tensorfs_root"],
            install_root=seams["install_root"],
            python=owned.python,
            interface=lambda _installed, _distribution: machine.interface,
            verified=seams["verified"],
            job_plan_root=seams["job_plan_root"],
            base=None,
            installed_environment=owned,
            materialized=seams["materialized"],
        )

    monkeypatch.setattr(package_prepare, "prepare_package_set", install)
    checks: list[str] = []
    check = package_prepare.prepare_model_placement

    def checked(**arguments: Any) -> Any:
        checks.append(str(arguments["installed"].installation_id))
        return check(**arguments)

    monkeypatch.setattr(package_prepare, "prepare_model_placement", checked)
    length = len(
        tensorfs.Store.open(str(machine.store_root)).manifest(machine.manifest)["manifest"]
    )
    catalog = Catalog(
        machine.manifest, length, [{"gpu": "H100", "lane": LANE}], interface=machine.interface
    )
    # A release with no published callee: every read is its own.
    catalog.releases[f"/v1/packages/{PACKAGE}/releases/{RELEASE}/locked-requirements"] = LOCKED

    def run(name: str) -> tuple[list[str], list[str], dict[str, float]]:
        catalog.serve(machine)
        catalog.reads.clear()
        receipt, progress = submit(machine, submission(machine, name))
        assert receipt.request_id == name
        (resolved,) = [
            canonical_json.decode(event.body)
            for event in machine.executions.events(OWNER, name).events
            if event.kind == "resolved"
        ]
        assert resolved["installation_id"] == owned.installation_id
        return [urlsplit(path).path for path in catalog.reads], progress, resolved["admission_ms"]

    try:
        reads, progress, admission = run("cold")
        assert reads[:2] == [
            f"/v1/packages/{PACKAGE}/releases/{RELEASE}",
            f"/v1/packages/{PACKAGE}/releases/{RELEASE}/locked-requirements",
        ]
        assert progress[0].startswith("installing ") and "install" in admission
        assert (installs, checks) == ([""], [owned.installation_id])

        _, _, warm_admission = run("warm")
        assert "install" not in warm_admission and "install" in admission
        assert (installs, checks) == ([""], [owned.installation_id])

        kept = package_installation.kept_placements(machine.install_root, owned)
        assert kept is not None
        (kept / "malformed.json").write_bytes(b"{")
        (kept / "obsolete.json").write_bytes(
            b'{"format":"cozy.worker.v1.PlacementSet/1","placements":[{"package_interface":{}}]}'
        )
        machine.worker.shutdown()
        machine.boot()
        reads, progress, admission = run("restarted")
        assert (reads, progress, installs, len(checks)) == ([], [], [""], 1)
        assert "install" not in admission

        # The installation was rebuilt (its record replaced): what was kept is not its own.
        record = owned.generation.parent / "installation.json"
        os.utime(record, ns=(time.time_ns(), time.time_ns() + 1_000_000))
        machine.worker.shutdown()
        machine.boot()
        _, progress, _ = run("rebuilt")
        assert progress[0].startswith("installing ") and len(installs) == 2
    finally:
        catalog.close()


def test_a_newer_callee_release_serves_the_callee_a_lock_names(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tenant's lock names callee 1.0.0; the catalog publishes 1.1.0, which serves its
    calls. A lock is a floor, so a callee fix needs no relock of its callers."""
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
        machine.manifest, length, [{"gpu": "H100", "lane": LANE}], interface=machine.interface
    )
    path = f"/v1/packages/{CALLEE}"
    catalog.releases[path]["releases"].append({"release": "1.1.0"})
    catalog.releases[f"{path}/releases/1.1.0"] = {
        **catalog.releases[f"{path}/releases/{RELEASE}"],
        "release": {"release": "1.1.0"},
    }
    catalog.releases[f"{path}/releases/1.1.0/locked-requirements"] = (
        b"callee==1.1.0 --hash=sha256:" + b"c" * 64 + b"\n"
    )
    catalog.serve(machine)
    try:
        receipt, _ = submit(machine, submission(machine, "newer"))
        assert receipt.request_id == "newer"
        capture = machine.executions.capture(OWNER, "newer")
        assert [row["key"] for row in capture["deferred_installations"]] == [f"{CALLEE}@1.1.0"]
        assert f"{path}/releases/{RELEASE}" not in [urlsplit(row).path for row in catalog.reads]
    finally:
        catalog.close()


def test_a_wheel_a_release_carries_is_installed_not_called(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tenant's lock also names a third-party wheel on its org's index (the build of a
    pinned commit). The Hub publishes no package by that name, so it is a dependency: the
    closure holds the one real callee, and the catalog is asked about the wheel once."""
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
        machine.manifest, length, [{"gpu": "H100", "lane": LANE}], interface=machine.interface
    )
    wheel = f"sha256:{'d' * 64}/diffusers-0.41.0.dev0-py3-none-any.whl"
    carried = f"{catalog.origin}/v1/index/cozytest/files/{wheel}"
    lock = f"/v1/packages/{PACKAGE}/releases/{RELEASE}/locked-requirements"
    catalog.releases[lock] += f"diffusers @ {carried} --hash=sha256:{'d' * 64}\n".encode()
    catalog.serve(machine)
    try:
        for request in ("carried", "again"):
            receipt, _ = submit(machine, submission(machine, request))
            assert receipt.request_id == request
            capture = machine.executions.capture(OWNER, request)
            assert [row["key"] for row in capture["deferred_installations"]] == [
                f"{CALLEE}@{RELEASE}"
            ]
        asked = [urlsplit(row).path for row in catalog.reads]
        assert asked.count("/v1/packages/cozytest/diffusers") == 1
    finally:
        catalog.close()


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
                    hubs=(),
                ),
                host_facts=lambda: hostfacts.measure("none"),
                executions=SimpleNamespace(
                    capture_root=lambda *_: "root",
                    capture=lambda *_: {"model_defaults": []},
                    prepared=lambda *_: SimpleNamespace(hub="", account="cozytest"),
                ),
                lanes=SimpleNamespace(entries=()),
                resolutions=machine_model_resolve.Resolutions(),
            ),
        )
        call = cast(Call, SimpleNamespace(parent_request="root"))

        def child(package: str) -> Target:
            model = {
                "path": SLOT,
                "default_ladder": [{"gpu": "*", "lane": f"tenant@{RELEASE}/{LANE}"}],
            }
            return Target(
                "installation-" + package.replace("/", "-"),
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
        # The same installation's slot again reads nothing.
        catalog.reads.clear()
        assert machine_model_defaults.select(worker, "owner", call, child(PACKAGE), {}) == [
            expected
        ]
        assert catalog.reads == []
        # An unpublished callee has no org and no binding: its default names the run's account.
        catalog.reads.clear()
        assert machine_model_defaults.select(worker, "owner", call, child("local/other"), {}) == [
            expected
        ]
        assert [urlsplit(path).path for path in catalog.reads] == ["/v1/models/resolve"]
        # A supplied Model is never resolved.
        catalog.reads.clear()
        assert (
            machine_model_defaults.select(worker, "owner", call, child(PACKAGE), {"model": {}})
            == []
        )
        assert catalog.reads == []
    finally:
        catalog.close()


def test_an_unpublished_roots_org_relative_default_names_its_owner() -> None:
    """Unpublished code (local/) has no org: its org-relative default names the owner the
    root carries, and without one it refuses rather than guess."""
    manifest = "sha256:" + "33" * 32
    catalog = Catalog(manifest, 7, [])
    try:
        declaration = {
            "path": SLOT,
            "default_ladder": [{"gpu": "*", "lane": f"tenant@{RELEASE}/{LANE}"}],
        }
        resolved = machine_model_resolve.slot(
            machine_model_resolve.Catalog(catalog.origin),
            "local/tenant",
            declaration,
            "model",
            None,
            lambda _rung: True,
            "cozytest",
        )
        assert (resolved["repository"], resolved["manifest"]) == (
            PACKAGE,
            {"digest": manifest, "length": 7},
        )
        assert [urlsplit(path).path for path in catalog.reads] == ["/v1/models/resolve"]
        with pytest.raises(WorkspaceRefusal, match="with no owner"):
            machine_model_resolve.slot(
                machine_model_resolve.Catalog(catalog.origin),
                "local/tenant",
                declaration,
                "model",
                None,
                lambda _rung: True,
            )
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
    secret = "hf_cozy_private_preparation_credential_must_stay_in_memory_292"
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
            submitted.source_credentials.add(
                provider=pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE, credential="bearer " + secret
            )
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
        # The profile's carriers, the pipeline's configs and its tokenizer were read: the
        # choice lands the same self-contained checkpoint an upload does, tokenizer files as
        # assets. Nothing outside the pipeline was.
        repository = _repository(modular=False)
        assert origin.touched == set(repository) - {"README.md"}
        assert any(secret in value for value in origin.authorization), (
            "the real native source child did not present the transient provider credential"
        )
        header = store.manifest(model["manifest"]["digest"])["header"]
        assert header is not None
        assert sorted(tensorfs.parse_header(header)["assets"]) == [
            "tokenizer/tokenizer_config.json",
            "tokenizer/vocab.json",
        ]
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
    assert machine.worker.workspace is not None
    # Include journal/WAL, source facts and plans, native logs and retained preparation.
    roots = (machine.worker.workspace.directory, machine.worker.config.cozy_home)
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and not path.is_symlink():
                assert secret.encode() not in path.read_bytes(), (
                    f"provider credential persisted in {path.relative_to(root)}"
                )


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
    # Its unit dispatches it at once; accepted is all this asserts.
    assert machine.executions.status(OWNER, "jobbed").state in ("queued", "running")
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
    """Explicit choices read no binding; their lanes are read at once."""
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
    # A lane-pinned override of a second slot (H3's `turbo_lora`) is exactly that checkpoint.
    assert [(r["repository"], r["release"], r["lane"], r["manifest"]) for r in rows] == [
        (PACKAGE, RELEASE, LANE, {"digest": published.manifest, "length": 1})
    ] * 2
    assert [urlsplit(path).path for path in catalog.reads] == [
        "/v1/models/resolve",
        "/v1/models/resolve",
    ]


def test_the_machine_describes_a_release_from_its_own_hub(published: Machine) -> None:
    """A controller asks the machine, never the Hub, for a release's interface: the newest
    release when it names none, read once until the package is forgotten; an installed
    release answers from its install."""
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
        catalog.reads.clear()
        assert describe() == described and catalog.reads == []
        machine.worker.resolutions.forget(PACKAGE)
        assert describe() == described
        assert [urlsplit(path).path for path in catalog.reads] == [tenant]
        with pytest.raises(Refused, match="not a published release"):
            describe("9.9.9")
        # Without a query, the workspace describes nothing and reads nothing.
        catalog.reads.clear()
        plain = machine.rpc.GetMachineExecutionWorkspace(
            pb.MachineExecutionWorkspaceQuery(claim=machine.claim), Context()
        )
        assert not plain.HasField("described_release") and catalog.reads == []
        # It resolves an unpublished root's slots under the root's owner, and says so.
        assert plain.release_root_owner
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


def test_the_owners_unpublished_checkpoint_resolves_with_the_worker_capability(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The machine's own Hub serves its owner's unpublished Model only to its worker
    capability: the lane resolution and the digest's manifest bytes both present it, and an
    ended grant is refused typed, never retried anonymously."""
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
    catalog.private = True
    catalog.serve(machine)
    try:
        # The lane resolves at the Hub, and its unstated length is measured from the
        # digest's manifest bytes: both are the owner's, read with the capability.
        receipt, _ = submit(machine, submission(machine, "by-lane"))
        assert receipt.request_id == "by-lane"
        assert [path for path, *_ in catalog.presented] == [
            "/v1/models/resolve",
            f"/v1/models/{PACKAGE}/checkpoints/{machine.manifest}",
        ]
        assert all((who, token) == ("worker", "token") for _, who, token in catalog.presented)

        # Resolved afresh (a rebinding), an ended grant is refused.
        catalog.presented.clear()
        catalog.serve(machine, token="ended")
        machine.worker.resolutions.forget(PACKAGE)
        with pytest.raises(Refused, match=r"catalog answered 401 worker\.ended"):
            submit(machine, submission(machine, "ended"))
        assert catalog.presented == [("/v1/models/resolve", "worker", "ended")], catalog.presented
    finally:
        catalog.close()


def test_a_pinned_checkpoint_held_without_its_repository_is_taken_at_the_runs_hub(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The machine that uploaded a checkpoint holds its manifest and bytes but no custody of
    the repository it published into. A run pinning that checkpoint reads no catalog and
    takes custody at the run's Hub with one closure read: no download step, no byte moved."""
    machine = published
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda *_: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    store = tensorfs.Store.open(str(machine.store_root))
    length = len(store.manifest(machine.manifest)["manifest"])
    catalog = Catalog(machine.manifest, length, [], interface=machine.interface)
    catalog.serve(machine)
    uploaded = "cozytest/uploaded"
    with pytest.raises(tensorfs.errors.Refusal):
        store.verify_checkpoint_source(uploaded, machine.manifest, length)
    request = submission(machine, "pinned")
    request.release_root.models.append(
        pb.ModelChoice(
            parameter="model",
            repository=uploaded,
            manifest=pb.Ref(digest=documents.raw(machine.manifest)),
        )
    )
    try:
        assert submit(machine, request)[0].request_id == "pinned"
        assert catalog.closures == [uploaded + "@" + machine.manifest]
        assert not any("/v1/models/" in path for path in catalog.reads), catalog.reads
        store.verify_checkpoint_source(uploaded, machine.manifest, length)
    finally:
        catalog.close()


def test_a_root_names_an_unpublished_installation_the_machine_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same root message names a captured installation instead of a release: the machine
    resolves its slots and mints the offer from the installation it holds, installs nothing
    for the root itself, and refuses typed for an installation it does not hold."""
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda *_: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    with tempfile.TemporaryDirectory(prefix="cz-root.", dir="/tmp") as directory:
        machine = Machine(Path(directory), monkeypatch, "private")
        try:
            captured = machine.worker.prepare_local_package(
                pb.PrepareLocalPackageRequest(
                    operation_id="sync",
                    package=pb.DevelopmentPackage(
                        package=PACKAGE,
                        release=RELEASE,
                        installation_id=machine.installed.installation_id,
                    ),
                    files=[pb.LocalPackageFile(filename="tenant-1.0.0-py3-none-any.whl")],
                    install_root=str(machine.install_root),
                )
            ).installed_package
            raw = tensorfs.Store.open(str(machine.store_root)).manifest(machine.manifest)[
                "manifest"
            ]
            catalog = Catalog(
                machine.manifest,
                None,
                [{"gpu": "H100", "lane": LANE}],
                raw,
                interface=machine.interface,
            )
            catalog.serve(machine)
            try:
                root = submission(machine, "captured")
                root.release_root.ClearField("release")
                root.release_root.installation_id = captured.installation_id
                receipt, _ = submit(machine, root)
                assert receipt.request_id == "captured"
                (resolved,) = [
                    canonical_json.decode(event.body)
                    for event in machine.executions.events(OWNER, "captured").events
                    if event.kind == "resolved"
                ]
                assert resolved["installation_id"] == captured.installation_id
                assert resolved["release"] == RELEASE
                assert f"/v1/packages/{PACKAGE}/releases/{RELEASE}" not in {
                    urlsplit(path).path for path in catalog.reads
                }

                absent = submission(machine, "absent")
                absent.release_root.ClearField("release")
                absent.release_root.installation_id = "0" * 24
                with pytest.raises(Refused) as refused:
                    machine.rpc.SubmitMachineExecution(absent, Context())
                assert (
                    refused.value.trailers.get("cozy-error-code")
                    == "release_root_installation_absent"
                )
            finally:
                catalog.close()
        finally:
            machine.worker.shutdown()


def test_a_captured_roots_model_is_read_once_and_kept_across_restarts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A captured root naming its Model's repository without a lane reads that release's lanes
    and the lane's checkpoint once. A warm run, and the first run of a restarted Runtime, read
    nothing: the reads outlive the process. A controller that changes the model names it to
    `forget` (in any case), and the next run reads it once more."""
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda *_: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    with tempfile.TemporaryDirectory(prefix="cz-root.", dir="/tmp") as directory:
        machine = Machine(Path(directory), monkeypatch, "private")
        machine.running = _Held()
        raw = tensorfs.Store.open(str(machine.store_root)).manifest(machine.manifest)["manifest"]
        catalog = Catalog(
            machine.manifest, len(raw), [{"gpu": "H100", "lane": LANE}], interface=machine.interface
        )

        def run(name: str) -> list[str]:
            # What the controller sends first whenever the machine answers the root absent.
            captured = machine.worker.prepare_local_package(
                pb.PrepareLocalPackageRequest(
                    operation_id="sync",
                    package=pb.DevelopmentPackage(
                        package=PACKAGE,
                        release=RELEASE,
                        installation_id=machine.installed.installation_id,
                    ),
                    files=[pb.LocalPackageFile(filename="tenant-1.0.0-py3-none-any.whl")],
                    install_root=str(machine.install_root),
                )
            ).installed_package
            catalog.serve(machine)
            root = submission(machine, name)
            root.release_root.ClearField("release")
            root.release_root.installation_id = captured.installation_id
            root.release_root.models.add(parameter="model", repository=PACKAGE)
            catalog.reads.clear()
            receipt, _ = submit(machine, root)
            assert receipt.request_id == name
            return [urlsplit(path).path for path in catalog.reads]

        read = [f"/v1/models/{PACKAGE}", "/v1/models/resolve"]
        try:
            assert run("cold") == read
            assert run("warm") == []
            machine.worker.shutdown()
            machine.boot()
            assert run("restarted") == []
            machine.worker.resolutions.forget("other/model")
            assert run("unrelated") == []
            machine.worker.resolutions.forget(PACKAGE.upper())
            assert run("changed") == read
            machine.worker.shutdown()
            machine.boot()
            assert run("changed-restarted") == []
        finally:
            catalog.close()
            machine.worker.shutdown()


def test_a_roots_download_progress_reaches_its_submitter(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a root's Models download, each answer to its submitter says how many bytes have
    landed of how many, so the run shows its download moving before acceptance."""
    machine = published
    catalog = Catalog(machine.manifest, 1, [], interface=machine.interface)
    catalog.serve(machine)
    seen = threading.Event()

    def materialize(*_: Any, progress: Any = None, **__: Any) -> None:
        progress(1 << 30, 4 << 30)
        seen.wait(30)

    monkeypatch.setattr(machine_model_defaults, "materialize", materialize)
    request = pb.MachineExecutionSubmit(
        claim=machine.claim,
        submission_id="downloading",
        offer=pb.AttemptOffer(request_id="downloading"),
        payload_canonical_bytes=canonical_json.encode({"shots": 2}),
        expected_execution_workspace_id=machine.executions.workspace_id,
        release_root=pb.ReleaseRoot(
            package=PACKAGE,
            release=RELEASE,
            entrypoint="workflow",
            job=True,
            publication_grant="cozytest/_job-downloading",
        ),
    )
    answers: list[tuple[str, str]] = []
    try:
        while True:
            try:
                receipt = machine.rpc.SubmitMachineExecution(request, Context())
                break
            except Refused as refused:
                assert refused.trailers.get("cozy-error-code") == "release_root_preparing"
                if "cozy-progress-bytes" in refused.trailers:
                    answers.append((str(refused), refused.trailers["cozy-progress-bytes"]))
                    seen.set()
    finally:
        catalog.close()
    assert receipt.request_id == "downloading"
    assert ("downloading model weights", f"{1 << 30} {4 << 30}") in answers, answers


def test_the_compatibility_check_overlaps_the_weight_download(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model compatibility check needs the checkpoint heads and the installation, never
    the weight bytes: it runs beside the download, so preparing takes about the longer of the
    two, not their sum (run 1510). A check that fails stops the download and is the answer."""
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
    spans: dict[str, tuple[float, float]] = {}
    refuse = threading.Event()
    check = package_prepare.prepare_model_placement

    def slow_check(**kwargs: Any) -> Any:
        began = time.monotonic()
        time.sleep(1.5)  # the package environment's placement check
        if refuse.is_set():
            spans["check"] = (began, time.monotonic())
            raise package_prepare.PreparationRefusal("model_placement_incompatible", "no fit")
        answer = check(**kwargs)
        spans["check"] = (began, time.monotonic())
        return answer

    def slow_download(
        worker: Any,
        rows: Any,
        *,
        cancellation: Any = None,
        progress: Any = None,
        downloading: Any = None,
        waiting: Any = None,
    ) -> None:
        began = time.monotonic()
        for step in range(15):  # 1.5 s of weights, reported as they land
            if cancellation is not None and cancellation.cancelled:
                spans["download"] = (began, time.monotonic())
                raise WorkspaceRefusal("captured Model default materialization refused: CANCELLED")
            if progress is not None:
                progress(step, 15)
            time.sleep(0.1)
        spans["download"] = (began, time.monotonic())

    monkeypatch.setattr(package_prepare, "prepare_model_placement", slow_check)
    monkeypatch.setattr(machine_model_defaults, "materialize", slow_download)
    try:
        # First, while no placement is prepared: a check that refuses stops the download.
        refuse.set()
        with pytest.raises(Refused, match="model_placement_incompatible"):
            submit(machine, submission(machine, "incompatible"))
        assert spans["download"][1] < spans["check"][1] + 0.5, "the download ran past the refusal"
        spans.clear()
        refuse.clear()
        receipt, _ = submit(machine, submission(machine, "overlapped"))
        assert receipt.request_id == "overlapped"
        (check_began, check_ended), (download_began, download_ended) = (
            spans["check"],
            spans["download"],
        )
        # Together they take about the longer one: 1.5 s each, never back to back.
        window = max(check_ended, download_ended) - min(check_began, download_began)
        assert window < 2.4, f"the check and the download took {window:.2f} s: one after the other"
    finally:
        catalog.close()


def test_warm_serving_keeps_ordered_lora_views_and_reuses_the_same_preparation(
    published: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real shared serving preparation, native LoRA writer and source custody.

    The existing routing fixture supplies device facts and the construction census;
    its accepted roots are held without executing. This is no GPU qualification.
    """
    from cozy_runtime.internal import census_cache
    from test_lora_composition import Value, model

    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda *_: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    machine = published
    store = tensorfs.Store.open(str(machine.store_root))
    base = model(
        store,
        machine.root,
        "reuse-base",
        "transformer",
        {"proj.weight": Value((2, 3), (1, 2, 3, 4, 5, 6))},
        b'{"input":3,"output":2}',
    )
    first = model(
        store,
        machine.root,
        "reuse-first",
        "adapter",
        {
            "proj.lora_A.weight": Value((1, 3), (1, 0, 2)),
            "proj.lora_B.weight": Value((2, 1), (3, 4)),
        },
    )
    second = model(
        store,
        machine.root,
        "reuse-second",
        "adapter",
        {
            "proj.lora_A.weight": Value((2, 3), (1, 1, 0, 0, 1, 2), "f16"),
            "proj.lora_B.weight": Value((2, 2), (1, 2, 3, 4), "f16"),
        },
    )
    monkeypatch.setattr(
        package_prepare,
        "_census",
        lambda *_: (
            lambda wanted: {
                path: census_cache.Census(("transformer",), (), True, "") for path in wanted
            }
        ),
    )
    for name, artifact in (("reuse-base", base), ("reuse-first", first), ("reuse-second", second)):
        store.replace_local(
            None,
            name,
            artifact.tensorfs_receipt_digest,
            artifact.manifest.digest,
            artifact.manifest.length,
        )
    length = base.manifest.length
    catalog = Catalog(base.manifest.digest, length, [], interface=machine.interface)
    catalog.serve(machine)
    preparations = 0

    def counted[**P, R](function: Callable[P, R]) -> Callable[P, R]:
        def observe(*args: P.args, **kwargs: P.kwargs) -> R:
            nonlocal preparations
            preparations += 1
            return function(*args, **kwargs)

        return observe

    monkeypatch.setattr(
        package_prepare, "prepare_model_placement", counted(package_prepare.prepare_model_placement)
    )
    views: dict[str, str] = {}
    try:
        for name, stack, expected_preparations in (
            ("baseline", (), 1),
            ("stack", ((first, "0.5"), (second, "-0.25")), 2),
            ("repeat", ((first, "0.5"), (second, "-0.25")), 2),
            ("reverse", ((second, "-0.25"), (first, "0.5")), 3),
            ("zero", ((first, "0"), (second, "-0.25")), 4),
            ("baseline-after", (), 4),
        ):
            request = submission(machine, "lora-" + name)
            choice = request.release_root.models.add(
                parameter="model",
                repository="local/reuse-base",
                manifest=pb.Ref(digest=documents.raw(base.manifest.digest), length=length),
            )
            for adapter, scale in stack:
                choice.adapters.add(
                    component="transformer",
                    model="local/reuse-first" if adapter == first else "local/reuse-second",
                    manifest=adapter.manifest.digest,
                    source_component="adapter",
                    scale=scale,
                )
            receipt, progress = submit(machine, request)
            # Creator records each new answer as a `machine preparation` stage of the run.
            recorded = [line for i, line in enumerate(progress) if line not in progress[:i][-1:]]
            assert [line for line in recorded if line.startswith("adapter_")] == [
                f"adapter_{index}: local/reuse-{'first' if artifact == first else 'second'} "
                f"{artifact.manifest.digest} scale {scale} on transformer"
                for index, (artifact, scale) in enumerate(stack)
            ], recorded
            prepared = machine.executions.preparation(OWNER, receipt.request_id)
            placement = next(iter(prepared["installations"].values()))["placement"]
            models = {row["id"]: row for row in placement["models"]}
            slot = next(entry for entry in placement["entrypoints"] if entry["name"] == "touch")[
                "slots"
            ][0]
            assert models[slot["reference_model_id"]]["manifest"]["digest"] == base.manifest.digest
            adapters = slot.get("adapters", [])
            assert [
                (models[row["model_id"]]["manifest"]["digest"], row["scale"]) for row in adapters
            ] == [(artifact.manifest.digest, scale) for artifact, scale in stack]
            views[name] = models[slot["components"][0]["model_id"]]["manifest"]["digest"]
            assert (views[name] != base.manifest.digest) == bool(stack)
            assert preparations == expected_preparations, f"{name}: unexpected preparation count"
        # The composition Runtime 0.18.100's canary candidate (55e5ce82) served for this stack.
        assert views["stack"] == (
            "sha256:14b8c0fd2191dc429831dc3e8a36348b7ff8aab213c89dccda47b7ff87cee397"
        )
        assert views["repeat"] == views["stack"]
        assert views["reverse"] != views["stack"] and views["zero"] != views["stack"]
        assert views["baseline-after"] == views["baseline"]
        assert preparations == 4, "baseline or adapted warm preparation was not reused"
    finally:
        catalog.close()
