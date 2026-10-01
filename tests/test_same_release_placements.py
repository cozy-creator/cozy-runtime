"""One release, two prepared placements: concurrent roots never displace each other.

Runs 1273/1274: a workflow job's preparation (every slot) replaced the serving function's
preparation (its own slot) of the same release, so the serving root was refused with
"execution inference does not name the captured root" while the job held the GPUs.

Real: the Worker, its published and private preparation registration, SubmitMachineExecution's
capture verification, the durable journal, each execution's unit and GPU grants, and a TensorFS
store holding the model. Fakes: the uv install (a retained uv venv carries the package), the
describe and census derives, device memory, and the job's run (it holds its GPUs until
`finish`).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import pytest
import tensorfs

import signed_claims
from cozy_runtime import canonical_json
from cozy_runtime.internal import (
    accel,
    canonical,
    census_cache,
    child_env,
    package_environment,
    package_installation,
    package_interface,
)
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.worker import activity, execution_unit, package_prepare
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.machine_execution_rpc import MachineExecutionRPC
from cozy_runtime.internal.worker.session import Worker, WorkerOptions, read_placement_set
from cozy_runtime.internal.worker.supervisor import Unit
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_gpu_scheduler import VIRTUAL, driverless
from test_job_slot_paths import _ASSET, _HEADER, _ref

GiB = 1 << 30
OWNER = "owner"
PACKAGE = "cozytest/tenant"
RELEASE = "1.0.0"
LANE = "bf16"
TENANT = Path(__file__).resolve().parent.parent / "corpus" / "tenant"
SOURCE = (
    (TENANT / "tiny_tenant.py").read_text()
    + """

class WorkflowRequest(msgspec.Struct, forbid_unknown_fields=True):
    shots: int = 1


class WorkflowResult(msgspec.Struct):
    shots: int


@app.job
def workflow(ctx: Context, payload: WorkflowRequest) -> WorkflowResult:
    return WorkflowResult(shots=payload.shots)
"""
)
LOCKED = b"--index-url https://pypi.org/simple\ntenant==1.0.0 --hash=sha256:" + b"a" * 64 + b"\n"


class Refused(Exception):
    def __init__(self, code: object, detail: str, trailers: tuple[tuple[str, str], ...]) -> None:
        super().__init__(detail)
        self.code, self.trailers = code, dict(trailers)
        self.absent = [value for key, value in trailers if key == "cozy-absent-release"]


class Context:
    def __init__(self) -> None:
        self.trailers: tuple[tuple[str, str], ...] = ()

    def abort(self, code: object, detail: str) -> NoReturn:
        raise Refused(code, detail, self.trailers)

    def is_active(self) -> bool:
        return True

    def set_trailing_metadata(self, metadata: tuple[tuple[str, str], ...]) -> None:
        self.trailers = metadata

    def add_callback(self, _callback: Callable[[], None]) -> bool:
        return True


class Machine:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch, origin: str) -> None:
        self.origin = origin
        self.install_root = root / "installs"
        self.store_root, self.manifest = _store(root)
        self.installed = installed = _install(root, self.install_root)
        self.installed_package: pb.InstalledPackage | None = None
        project = root / "project"
        project.mkdir()
        (project / "tenant_workflow.py").write_text(SOURCE)
        (project / "package.toml").write_text(
            (TENANT / "package.toml").read_text().replace("tiny_tenant:app", "tenant_workflow:app")
        )
        self.interface = package_interface.canonical_bytes(
            package_interface.build(discover(project))
        )

        def install(request: pb.PreparePackageSetRequest, **seams: Any) -> Any:
            models = documents.read(request.download_delegation, pb.DownloadDelegation)["models"]
            return package_prepare._prepare_published(
                package_name=PACKAGE,
                release=RELEASE,
                locked=package_environment.read_locked_requirements(LOCKED),
                models=package_prepare.selections(models),
                artifact_cache=seams["artifact_cache"],
                tensorfs_root=seams["tensorfs_root"],
                install_root=seams["install_root"],
                python=installed.python,
                interface=lambda _installed, _distribution: self.interface,
                verified=seams["verified"],
                job_plan_root=seams["job_plan_root"],
                base=None,
                installed_environment=installed,
                materialized=seams["materialized"],
            )

        def census(*_: Any) -> Any:
            return lambda wanted: {
                path: census_cache.Census(("text_encoder", "unet", "vae"), (), True, "")
                for path in wanted
            }

        monkeypatch.setattr(package_prepare, "prepare_package_set", install)
        #: job request -> set when its Python returns
        self.running: dict[str, threading.Event] = {}
        dispatch = Worker.dispatch_machine

        def dispatch_job(worker: Worker, unit: Unit, offered: pb.AttemptOffer, *args: Any) -> Any:
            """A job runs its Python on its grant until `finish` (this venv carries no
            Runtime to boot a job executor from); everything else dispatches for real."""
            if offered.request_id not in self.running:
                return dispatch(worker, unit, offered, *args)
            self.executions.dispatched(OWNER, offered.request_id, offered.attempt_ordinal)
            unit.wait_for(self.running[offered.request_id].is_set)
            return None

        monkeypatch.setattr(Worker, "dispatch_machine", dispatch_job)
        monkeypatch.setattr(package_prepare, "_census", census)
        driverless(monkeypatch)
        monkeypatch.setattr(
            accel,
            "device_memory",
            lambda entry, kind: accel.DeviceMemory("measured", 80 * GiB, 80 * GiB),
        )
        self.root, self.monkeypatch = root, monkeypatch
        self.boot()

    def boot(self) -> None:
        """This machine's Runtime over its roots: a second call is a restarted Runtime."""
        root, monkeypatch = self.root, self.monkeypatch
        base = {
            k: v for k, v in os.environ.items() if not child_env.erased(k) and k != "PYTHONPATH"
        }
        self.worker = Worker(
            replace(
                RuntimeConfig(
                    cozy_home=root / "home",
                    credentials=Credentials(),
                    child_base_env=tuple(sorted(base.items())),
                ),
                record_owner_public_key=signed_claims.PUBLIC_KEY,
            ),
            WorkerOptions(
                **signed_claims.IDENTITY,
                root=root / "worker",
                tensorfs_root=self.store_root,
                install_root=self.install_root,
                artifact_cache=root / "artifacts",
                devices=VIRTUAL,
            ),
            InMemoryControlHost(),
        )
        self.claim = signed_claims.claim(OWNER)
        describe = self.worker._describe_installed

        def described(installed: Any, distribution: str, **options: Any) -> bytes:
            if distribution == "tenant":
                return self.interface
            return describe(installed, distribution, **options)

        monkeypatch.setattr(self.worker, "_describe_installed", described)
        assert self.worker.accept_claim(self.claim, lambda frame: None)[0] >= 1
        assert self.worker.executions is not None
        self.executions = self.worker.executions
        self.rpc = MachineExecutionRPC(self.worker)

    def prepare(self, *slots: str) -> pb.DesiredPlacementSet:
        """The release prepared for exactly these model slots, as Creator asks per request."""
        rows = [
            {
                "lane": LANE,
                "manifest": self.manifest,
                "model": PACKAGE,
                "package": PACKAGE,
                "release": RELEASE,
                "slot": slot,
            }
            for slot in slots
        ]
        if self.origin == "private":
            if self.installed_package is None:
                self.installed_package = self.worker.prepare_local_package(
                    pb.PrepareLocalPackageRequest(
                        operation_id="sync",
                        package=pb.DevelopmentPackage(
                            package=PACKAGE,
                            release=RELEASE,
                            installation_id=self.installed.installation_id,
                        ),
                        files=[pb.LocalPackageFile(filename="tenant-1.0.0-py3-none-any.whl")],
                        install_root=str(self.install_root),
                    )
                ).installed_package
            return self.worker.prepare_unpublished_placement(
                pb.PreparePrivatePlacementRequest(
                    operation_id="sync",
                    installation_id=self.installed.installation_id,
                    download_delegation=canonical.write(
                        {
                            "format": "cozy.worker.v1.DownloadDelegation/1",
                            "packages": [],
                            "models": rows,
                        }
                    ),
                )
            ).placement_set
        result = self.worker.prepare_package_set(
            pb.PreparePackageSetRequest(
                download_delegation=canonical.write(
                    {
                        "format": "cozy.worker.v1.DownloadDelegation/1",
                        "packages": [{"package": PACKAGE, "release": RELEASE}],
                        "models": rows,
                    }
                ),
                install_root=str(self.install_root),
                application="tenant_workflow:app",
                locked_requirements=LOCKED,
                package_interface=self.interface,
            )
        )
        self.installed_package = result.installed_package
        return result.placement_set

    def submit(
        self,
        request: str,
        spec: dict[str, Any],
        state: pb.DesiredWorkerState,
        placement: str = "",
        *,
        hold: bool = True,
    ) -> None:
        assert self.installed_package is not None
        installation = self.installed_package.installation_id
        capture, capture_digest = documents.identity(
            pb.MachineExecutionCapture(
                root_installation_id=installation,
                installed_packages=[self.installed_package],
            )
        )
        payload = canonical_json.encode({})
        digest = documents.spell(documents.digest_of(payload))
        raw, spec_digest = documents.identity(
            pb.InvocationSpec(
                installation_id=installation,
                payload_digest=digest,
                inputs=[
                    pb.InputBinding(
                        input_id="payload",
                        digest=digest,
                        length=len(payload),
                        kind_mime="application/json",
                    )
                ],
                deadline_unix_ms=int(time.time() * 1000) + 600_000,
                **spec,
            )
        )
        state.revision, state.wire_minor, state.posture = 1, WIRE_MINOR, pb.POSTURE_ACCEPTING
        if "job" in spec and hold:
            self.running[request] = threading.Event()
        self.rpc.SubmitMachineExecution(
            pb.MachineExecutionSubmit(
                claim=self.claim,
                submission_id=request,
                capture_digest=capture_digest,
                capture_canonical_bytes=capture,
                offer=pb.AttemptOffer(
                    request_id=request,
                    attempt_ordinal=1,
                    placement_id=placement,
                    invocation_spec_canonical_bytes=raw,
                    invocation_spec_digest=spec_digest,
                    grant=pb.DeliveryGrant(
                        invocation_spec_digest=spec_digest,
                        inputs=[
                            pb.InputAccess(
                                input_id="payload",
                                url="data:application/json;base64,"
                                + base64.b64encode(payload).decode(),
                            )
                        ],
                    ),
                ),
                prepared_state=state,
                payload_canonical_bytes=payload,
                expected_execution_workspace_id=self.executions.workspace_id,
            ),
            Context(),
        )

    def finish(self, request: str) -> None:
        """What the job's executor records when its Python returns."""
        row = self.executions.row(OWNER, request)
        assert row is not None and row.dispatched
        selected = row.attempt_offer()
        body, digest = documents.identity(
            pb.AttemptOutcomeBody(
                request_id=request,
                attempt_ordinal=selected.attempt_ordinal,
                invocation_spec_digest=documents.spell(selected.invocation_spec_digest),
                status=pb.OUTCOME_STATUS_SUCCEEDED,
                cause=pb.OutcomeCause(
                    code=pb.CAUSE_CODE_UNSPECIFIED, origin=pb.CAUSE_ORIGIN_WORKER
                ),
                safe_message="finished",
                result=pb.ResultEnvelope(inline_result=canonical_json.encode(None)),
            )
        )
        terminal = pb.AttemptOutcome(
            request_id=request,
            attempt_ordinal=selected.attempt_ordinal,
            invocation_spec_digest=selected.invocation_spec_digest,
            outcome_id=f"out-{request}",
            outcome_digest=digest,
            outcome_canonical_bytes=body,
        )
        self.executions.workspace.outcome(OWNER, terminal)
        self.executions.reconcile(OWNER, request)
        self.running[request].set()
        self.worker.supervisor.poke(execution_unit.key(request))

    def tick(self) -> None:
        """Every unit has done all it can."""
        bound = time.monotonic() + 300  # a hang bound on a loaded box, not a budget
        while not self.worker.supervisor.quiet_now():
            assert time.monotonic() < bound, self.worker.supervisor.keys("")
            time.sleep(0.01)

    def journal(self, root: str) -> list[tuple[int, str, dict[str, Any]]]:
        return [
            (event.sequence, event.kind, canonical_json.decode(event.body))
            for event in self.executions.events(OWNER, root).events
            if event.kind.startswith("gpu.")
        ]


def _store(root: Path) -> tuple[Path, str]:
    """A TensorFS store holding one released model every selection resolves."""
    store_root = root / "store"
    store = tensorfs.Store.init(str(store_root))
    for name, raw in (("header.cbor", _HEADER), ("vocab.txt", _ASSET)):
        (root / name).write_bytes(raw)
        ref = _ref(raw)
        store.put_file(str(root / name), "sha256:" + ref["sha256"], ref["length"])
    raw = json.dumps(
        {"entries": [{"blob": _ref(_HEADER), "kind": "cozytensors", "path": "model.cozytensors"}]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    manifest = "sha256:" + hashlib.sha256(raw).hexdigest()
    store.put_manifest(raw, manifest, len(raw))
    operation = store.begin_operation("same-release", *PACKAGE.split("/"))
    operation.hold_manifest(manifest, len(raw))
    operation.commit_release(None, RELEASE, LANE, manifest, len(raw))
    return store_root, manifest


def _install(root: Path, install_root: Path) -> package_installation.InstalledEnvironment:
    """The release installed in a real uv venv, retained as a worker installation."""
    generation = root / "generation"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(generation)], check=True, capture_output=True
    )
    (site_packages,) = generation.glob("lib/python*/site-packages")
    metadata = site_packages / "tenant-1.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.3\nName: tenant\nVersion: 1.0.0\n")
    (site_packages / "tenant_workflow.py").write_text(SOURCE)
    return package_installation.retain_environment(
        install_root, generation / "bin/python", package=PACKAGE, release=RELEASE
    )


@pytest.fixture(params=["published", "private"])
def machine(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[Machine]:
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-rel.", dir="/tmp") as root:
        made = Machine(Path(root), monkeypatch, request.param)
        try:
            yield made
        finally:
            made.worker.shutdown()


def test_a_serving_root_waits_behind_a_job_of_its_own_release(machine: Machine) -> None:
    # An earlier serving run prepared `touch`; the workflow then prepared every slot.
    serving = machine.prepare("touch.models.model")
    workflow = machine.prepare("touch.models.model", "retouch.models.model")
    (served,) = read_placement_set(serving)
    (every,) = read_placement_set(workflow)
    installation = served.installation_id
    assert (every.placement_id, every.installation_id) == (served.placement_id, installation)
    assert every.bindings_digest != served.bindings_digest

    interface = package_interface.read_bytes(machine.interface)
    descriptor = package_interface.job_descriptor_id(interface, "workflow")
    machine.submit(
        "A",
        {"job": pb.JobInvocationSpec(installation_id=installation, job_descriptor_id=descriptor)},
        pb.DesiredWorkerState(
            job=pb.JobDirective(installation_id=installation, job_descriptor_id=descriptor)
        ),
    )
    machine.tick()
    holders = machine.worker.stages.view()["holders"]
    assert holders == {"0": "A#1", "1": "A#1", "2": "A#1", "3": "A#1"}

    # The serving root reuses its own earlier preparation, as a pod host answers an
    # identical preparation from its retained result without reaching Runtime.
    touch = next(row for row in served.entrypoints if row.name == "touch")
    machine.submit(
        "B",
        {
            "serving": pb.ServingInvocationSpec(
                entrypoint_binding_digest=documents.spell(touch.entrypoint_binding_digest),
                attempt_binding_id=documents.spell(touch.entrypoint_binding_digest),
                bindings_digest=documents.spell(served.bindings_digest),
            )
        },
        pb.DesiredWorkerState(placement_set=serving),
        placement=served.placement_id,
    )
    held = machine.executions.preparation(OWNER, "B")["installations"][installation]
    assert held["placement"]["bindings_digest"] == documents.spell(served.bindings_digest)
    machine.tick()
    waits = [body for _, kind, body in machine.journal("B") if kind == "gpu.wait"]
    assert waits == [{"key": "B#1", "width": 1, "ordinals": [], "blocked_by": ["A"]}]

    machine.finish("A")
    machine.tick()
    released = next(seq for seq, kind, _ in machine.journal("A") if kind == "gpu.release")
    grant = next(body for _, kind, body in machine.journal("B") if kind == "gpu.grant")
    assert grant["key"] == "B#1" and len(grant["ordinals"]) == 1
    at = {e.sequence: e.at_ms for e in machine.executions.events(OWNER, "A").events}
    granted_at = next(
        e.at_ms for e in machine.executions.events(OWNER, "B").events if e.kind == "gpu.grant"
    )
    assert granted_at >= at[released]


def test_a_job_whose_executor_never_boots_ends_failed(machine: Machine) -> None:
    """This venv carries no Runtime, so the job's executor exits before it dials back: that
    is the execution's FAILED terminal, never a wait, and the rental is free at once."""
    machine.prepare("touch.models.model", "retouch.models.model")
    assert machine.installed_package is not None
    installation = machine.installed_package.installation_id
    interface = package_interface.read_bytes(machine.interface)
    descriptor = package_interface.job_descriptor_id(interface, "workflow")
    machine.submit(
        "A",
        {"job": pb.JobInvocationSpec(installation_id=installation, job_descriptor_id=descriptor)},
        pb.DesiredWorkerState(
            job=pb.JobDirective(installation_id=installation, job_descriptor_id=descriptor)
        ),
        hold=False,
    )
    machine.tick()
    assert machine.executions.status(OWNER, "A").state == "failed"
    body = documents.parse(
        machine.executions.outcome(OWNER, "A").outcome_canonical_bytes, pb.AttemptOutcomeBody
    )
    assert "job_executor_absent" in body.safe_message
    assert not activity.active_work(machine.worker)
