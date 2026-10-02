"""GPU scheduling: the policy module, then the real Worker's execution units over a durable
journal.

Worker tests use a real `Worker` with four device entries, a real claim, real `Executions`
and child-call journal rows, and each execution's own supervised unit. A root is held as a
running attempt the test owns (its CPU parent "in Python"); a child's unit takes its grant
and activates its replica, which, naming no installed package, refuses that child typed.
What is asserted is the scheduler decision as the durable journal records it, once every
unit has settled (`tick`: no unit has anything left to do). The only fake is the hardware
memory seam, `accel.device_memory`.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest

import signed_claims
from conftest import gpu_hidden
from cozy_runtime import canonical_json
from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import (
    accel,
    budget_cell,
    child_env,
    package_interface,
    proctree,
    weight_policy,
)
from cozy_runtime.internal.call_intent import canonical_intent
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.readiness import RuntimeGPU
from cozy_runtime.internal.worker import lanes, machine_model_defaults, memory
from cozy_runtime.internal.worker.attempts import AttemptRecord, AttemptSlot
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.machine_execution_rpc import MachineExecutionRPC
from cozy_runtime.internal.worker.plan import DeclaredBinding
from cozy_runtime.internal.worker.plan import ModelBinding as WorkerSlot
from cozy_runtime.internal.worker.session import HostedPlacement, Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import TERMINAL
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR

GiB = 1 << 30
OWNER = "owner"
#: Four cards no host has: a test worker's executors are sealed where no GPU is.
VIRTUAL = "64,65,66,67"
H3 = [{"sequence_parallel": {"degrees": [2, 4, 7, 8]}}] * 2
MARCO = Path(__file__).resolve().parent.parent / "examples" / "marco-polo"


# ------------------------------------------------------------------------------ policy


def test_default_ladder_picks_the_widest_rung_so_h3_is_one_four_gpu_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from cozy_runtime.internal import hostfacts

    checkpoint = {
        "repository": "proof/h3",
        "manifest": {"digest": "sha256:" + "a" * 64, "length": 1},
    }
    ladder = [
        {"gpu": "H100", "gpus": 2, **checkpoint},
        {"gpu": "H100", "gpus": 4, **checkpoint},
        {"gpu": "H200", "gpus": 1, **checkpoint},
    ]
    rows = [
        {
            "callee_installation_id": "h3",
            "entrypoint": "ref2va_turbo",
            "parameter": name,
            "public_origin": "https://hub.example",
            "rungs": ladder,
        }
        for name in ("base_model", "turbo_lora")
    ]
    worker = SimpleNamespace(
        options=SimpleNamespace(accelerator_backend="cuda"),
        host_facts=lambda: hostfacts.measure("cuda"),
        executions=SimpleNamespace(
            capture_root=lambda *_: "root", capture=lambda *_: {"model_defaults": rows}
        ),
        lanes=SimpleNamespace(entries=("0", "1", "2", "3")),
    )
    target = SimpleNamespace(
        installation_id="h3",
        callee="",
        prepared_installation={},
        entrypoint="ref2va_turbo",
        declaration={
            "models": [{"path": "ref2va_turbo.models." + n} for n in ("base_model", "turbo_lora")]
        },
    )
    monkeypatch.setattr(
        hostfacts,
        "measure",
        lambda _: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    call = SimpleNamespace(parent_request="parent")
    selected = machine_model_defaults.select(worker, OWNER, call, target, {})  # type: ignore[arg-type]
    assert [row["gpus"] for row in selected] == [4, 4]
    worker.lanes = SimpleNamespace(entries=("0", "1"))
    selected = machine_model_defaults.select(worker, OWNER, call, target, {})  # type: ignore[arg-type]
    assert [row["gpus"] for row in selected] == [2, 2]


def test_a_group_and_singletons_bind_overlapping_lanes() -> None:
    """Lanes overlap: a group over 0-3 and the device lanes under it. Which one runs on a
    device when is the stage scheduler's (`test_stage_scheduler`)."""
    lane_set = lanes.LaneSet.from_envelope("0,1,2,3", worker_pid=os.getpid())
    ok = accel.DeviceMemory("measured", 80 * GiB, 80 * GiB)
    measured = dict.fromkeys(range(4), ok)
    group = lane_set.bind("h3", (0, 1, 2, 3), model_bearing=True, measured=measured)
    qwen = [lane_set.bind(f"q{o}", (o,), model_bearing=True, measured=measured) for o in range(3)]
    assert group.lane_id == "lane-0+1+2+3" and [q.lane_id for q in qwen] == [
        "lane-0",
        "lane-1",
        "lane-2",
    ]
    assert all(lane_set.lane_of(p) is not None for p in ("h3", "q0", "q1", "q2"))

    with pytest.raises(lanes.LaneRefusal) as refused:
        lane_set.bind(
            "wide",
            (0, 1, 2, 3),
            model_bearing=True,
            measured={**measured, 3: accel.DeviceMemory("unreadable")},
        )
    assert refused.value.code == "device_group_infeasible"  # never a smaller group
    helper = lane_set.bind("helper", (), model_bearing=False, measured={})
    assert helper.ordinals == () and helper.devices == "" and lane_set.lane_of("helper") is helper
    lane_set.release("helper")
    assert lane_set.lane_of("helper") is None
    lane_set.release("h3")
    assert "lane-0+1+2+3" not in lane_set.by_id
    assert [lane_set.lane_of(f"q{o}") for o in range(3)] == qwen


def test_mixed_degree_declarations_are_refused_when_the_interface_is_built() -> None:
    body = package_interface.build(discover(MARCO))
    entry = next(row for row in body["entrypoints"] if row["name"] == "marco")
    slot = {"path": "marco.models.base", "class": "Base", "component_use": {}}
    with pytest.raises(ConformanceError) as refused:
        package_interface.callable_doc(
            name="marco",
            kind="entrypoint",
            payload_type=dict,
            result_type=dict,
            models=[{**slot, "sequence_parallel": {"degrees": [2, 4]}}, slot],
            assets=None,
            publishes=False,
            weights_outputs=(),
        )
    assert refused.value.code == "sequence_parallel_declaration_conflict"
    assert entry["name"] == "marco"


# ----------------------------------------------------------------------- the worker


def _interface(kind: str) -> str:
    body = copy.deepcopy(_BASE)
    entry = next(row for row in body["entrypoints"] if row["name"] == "marco")
    slot = {"path": "marco.models.model", "class": kind, "component_use": {}}
    entry.pop("models", None)
    if kind == "h3":
        entry["models"] = [{**slot, "sequence_parallel": {"degrees": [2, 4]}}]
    elif kind == "qwen":
        entry["models"] = [slot]
    return base64.b64encode(package_interface.canonical_bytes(body)).decode()


_BASE = package_interface.build(discover(MARCO))


class Machine:
    """One worker, its durable journal and the calls a test submits through it."""

    def __init__(
        self, root: Path, devices: str, boot: str, gpus: tuple[RuntimeGPU, ...] = ()
    ) -> None:
        base = {
            k: v for k, v in os.environ.items() if not child_env.erased(k) and k != "PYTHONPATH"
        }
        config = RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            child_base_env=tuple(sorted(base.items())),
        )
        self.worker = Worker(
            replace(config, record_owner_public_key=signed_claims.PUBLIC_KEY),
            WorkerOptions(
                worker_id=signed_claims.WORKER_ID,
                worker_tls_certificate_digest=signed_claims.TLS_DIGEST,
                root=root / "worker",
                tensorfs_root=root / "store",
                devices=devices,
                gpus=gpus,
                worker_boot_id=boot,
                accelerator_backend="none",
            ),
            InMemoryControlHost(),
        )
        claim = signed_claims.claim(OWNER, boot=boot)
        assert self.worker.accept_claim(claim, lambda frame: None)[0] >= 1
        self.claim = claim
        assert self.worker.executions is not None and self.worker.machine_calls is not None
        self.executions = self.worker.executions
        self.calls = self.worker.machine_calls
        self.children: dict[str, int] = {}
        #: activations a test holds; teardown lets every one go before the worker stops
        self.holds: list[threading.Event] = []

    def close(self) -> None:
        for held in self.holds:
            held.set()
        self.worker.shutdown()

    def submit(
        self,
        request: str,
        spec: pb.InvocationSpec,
        preparation: dict[str, Any] | None,
        placement: str = "",
        admit: bool = False,
        held: bool = False,
    ) -> None:
        raw, digest = documents.identity(spec)
        offer = pb.AttemptOffer(
            request_id=request,
            attempt_ordinal=1,
            placement_id=placement,
            invocation_spec_canonical_bytes=raw,
            invocation_spec_digest=digest,
            grant=pb.DeliveryGrant(invocation_spec_digest=digest),
        )
        prepared = canonical_json.encode(preparation) if preparation else b""
        if admit:  # the worker's root intake, as SubmitMachineExecution reaches it
            self.worker.submit_execution(
                self.claim,
                request,
                hashlib.sha256(request.encode()).digest(),
                offer,
                expected_execution_workspace_id=self.executions.workspace_id,
                preparation=prepared,
            )
            return
        self.executions.submit(
            OWNER,
            request,
            hashlib.sha256(request.encode()).digest(),
            offer,
            expected_execution_workspace_id=self.executions.workspace_id,
            worker_boot=self.worker.fence.worker_boot_id,
            preparation=prepared,
            worker_id=self.worker.options.worker_id,
        )
        if not held:
            self.worker.execution(OWNER, request)

    def root(self, name: str) -> None:
        """A CPU orchestration root, already running its Python."""
        self.submit(
            name,
            pb.InvocationSpec(
                job=pb.JobInvocationSpec(
                    installation_id="inst-root", job_descriptor_id="sha256:" + "12" * 32
                )
            ),
            None,
            held=True,
        )
        selected = self.executions.offer(OWNER, name)
        self.executions.dispatched(OWNER, name, 1)
        attempt = AttemptRecord(name, 1, selected.invocation_spec_digest, {}, state="running")
        self.worker.engine.history[attempt.key()] = attempt
        self.worker.engine.live[name] = attempt
        self.worker.execution(OWNER, name)

    def device_job(self, name: str) -> None:
        state = pb.DesiredWorkerState(
            wire_minor=WIRE_MINOR,
            posture=pb.POSTURE_ACCEPTING,
            job=pb.JobDirective(
                installation_id="inst-job", job_descriptor_id="sha256:" + "34" * 32
            ),
        )
        self.submit(
            name,
            pb.InvocationSpec(
                job=pb.JobInvocationSpec(
                    installation_id="inst-job", job_descriptor_id="sha256:" + "34" * 32
                )
            ),
            {"installations": {}, "state": base64.b64encode(state.SerializeToString()).decode()},
        )

    def child(self, parent: str, kind: str) -> str:
        """A serving child call of `kind`: h3 (degrees 2,4), qwen (a Model) or helper (none)."""
        index = self.children[parent] = self.children.get(parent, -1) + 1
        offer = self.executions.offer(OWNER, parent)
        request = pb.ChildCallRequest(
            parent_request_id=parent,
            parent_attempt_ordinal=offer.attempt_ordinal,
            parent_invocation_spec_digest=offer.invocation_spec_digest,
            call_index=index,
            module="library",
            export=kind,
            request_canonical_bytes=canonical_json.encode({}),
        )
        request.intent_digest = hashlib.sha256(canonical_intent(request)).digest()
        call = self.calls.journal.accept(OWNER, request)
        self.serving(call.child_request, kind)
        return call.child_request

    def serving(self, request: str, kind: str, gpus: int = 0, admit: bool = False) -> None:
        """A serving execution of `kind` against its prepared template, as Creator submits
        a rented root: `gpus` is the exact group width it names, zero for none."""
        digest = "sha256:" + hashlib.sha256(kind.encode()).hexdigest()
        template = {
            "placement_id": "tmpl-" + kind,
            "installation_id": "inst-" + kind,
            "package_interface": _interface(kind),
            "bindings_digest": digest,
            "entrypoints": [{"name": "marco", "entrypoint_binding_digest": digest}],
            "development": {
                "package": "local-" + kind,
                "release": "1.0.0",
                "installation_id": "inst-" + kind,
            },
        }
        raw = canonical_json.encode(
            {"format": "cozy.worker.v1.PlacementSet/1", "placements": [template]}
        )
        state = pb.DesiredWorkerState(
            wire_minor=WIRE_MINOR,
            posture=pb.POSTURE_ACCEPTING,
            placement_set=pb.DesiredPlacementSet(
                placement_set_digest=hashlib.sha256(raw).digest(),
                placement_set_canonical_bytes=raw,
                execution_gpus=gpus,
            ),
        )
        self.submit(
            request,
            pb.InvocationSpec(
                installation_id="inst-" + kind,
                serving=pb.ServingInvocationSpec(
                    entrypoint_binding_digest=digest,
                    attempt_binding_id=digest,
                    bindings_digest=digest,
                ),
            ),
            {"installations": {}, "state": base64.b64encode(state.SerializeToString()).decode()},
            placement="tmpl-" + kind,
            admit=admit,
        )

    def finish(self, request: str, status: pb.OutcomeStatus = pb.OUTCOME_STATUS_SUCCEEDED) -> None:
        """Record the attempt's outcome; a child whose replica could not activate here
        (these templates name no installed package) was already refused typed."""
        if self.executions.status(OWNER, request).state in TERMINAL:
            return
        row = self.executions.row(OWNER, request)
        assert row is not None
        selected = row.attempt_offer()
        body, digest = documents.identity(
            pb.AttemptOutcomeBody(
                request_id=request,
                attempt_ordinal=selected.attempt_ordinal,
                invocation_spec_digest=documents.spell(selected.invocation_spec_digest),
                status=status,
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
        held = self.worker.engine.live.pop(request, None)
        if held is not None:
            held.state = "closed"
        self.worker.execution(OWNER, request)
        self.worker._capacity_changed()

    def control(self, request: str, action: Literal["pause", "cancel"]) -> None:
        state = self.executions.status(OWNER, request)
        self.worker.control_execution(
            self.claim, request, f"{action}-{request}", state.generation, action
        )

    def warm(self) -> dict[str, tuple[int, int]]:
        """Every replica's executor, started as an activated one would be."""
        for hosted in list(self.worker.hosted.values()):
            if hosted.supervision.current is None:
                self.start(hosted)
        return self.executors()

    def start(self, hosted: HostedPlacement) -> None:
        """Activate one replica as a started executor would. These templates name no
        installed package, so acquisition would refuse them; this binds it to its grant
        through the worker's own `_assign_lane`, starts a real executor process sealed to
        that lane, and makes it dispatchable."""
        kind = hosted.placement.installation_id.removeprefix("inst-")
        degrees = (2, 4) if kind == "h3" else ()
        binding = DeclaredBinding(
            entrypoint_binding_digest="sha256:" + hashlib.sha256(kind.encode()).hexdigest(),
            entrypoint="marco",
            model_class="M",
            model_binding_path="marco.models.model",
            model_parameter_name="model",
            release="r",
            logical_weight_bytes=GiB,
            models=(
                WorkerSlot(
                    *("M", "marco.models.model", "model", "", "", ""),
                    components=("unet",),
                    snapshots={},
                    sequence_parallel_degrees=degrees,
                ),
            ),
        )
        hosted.latched = ""
        hosted.bindings = {binding.entrypoint_binding_digest: binding}
        hosted.failed_bindings = {}
        with self.worker.control_lock:
            assert self.worker._assign_lane(hosted.placement, hosted.bindings, hosted.supervision)
        lane = self.worker._lane_of(hosted.placement.placement_id)
        hosted.supervision.spawn(imposed=self.worker.imposed(lane))
        hosted.placement.materialization = pb.MaterializationState.MATERIALIZATION_STATE_STAGED
        hosted.placement.serving = pb.ServingState.SERVING_STATE_DISPATCHABLE
        self.worker._settle()

    def hold_activations(self, monkeypatch: pytest.MonkeyPatch) -> threading.Event:
        """Replica activations wait (their weights still downloading) until the returned
        event is set: calls granted meanwhile hold their GPUs together."""
        downloaded = threading.Event()
        self.holds.append(downloaded)
        activate = Worker._activate_replica

        def downloading(worker: Worker, *args: Any) -> None:
            downloaded.wait()
            activate(worker, *args)

        monkeypatch.setattr(Worker, "_activate_replica", downloading)
        return downloaded

    def await_granted(self, *requests: str) -> None:
        bound = time.monotonic() + 120
        while not all(self.granted(request) for request in requests):
            assert time.monotonic() < bound
            time.sleep(0.01)

    def activate_warm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replicas activate as started executors (`start`) instead of acquiring a package."""
        monkeypatch.setattr(
            Worker, "_activate_hosted", lambda worker, hosted, *_: self.start(hosted)
        )

    def executors(self) -> dict[str, tuple[int, int]]:
        live = {p: h.supervision.current for p, h in self.worker.hosted.items()}
        return {p: (e.epoch, e.pid) for p, e in live.items() if e is not None and e.alive()}

    def tick(self) -> dict[str, Any]:
        """Every unit has done all it can: the scheduler's view, settled."""
        bound = time.monotonic() + 120  # a hang bound on a loaded box, not a budget
        while not self.worker.supervisor.quiet_now():
            assert time.monotonic() < bound, self.worker.supervisor.keys("")
            time.sleep(0.01)
        return self.worker.stages.view()

    def journal(self, root: str) -> list[tuple[int, str, dict[str, Any]]]:
        """The root's gpu.* and stage.* observations, in journal order."""
        page = self.executions.events(OWNER, root)
        return [
            (event.sequence, event.kind, canonical_json.decode(event.body))
            for event in page.events
            if event.kind.startswith(("gpu.", "stage."))
        ]

    def granted(self, request: str) -> list[int]:
        """The GPUs a call holds its turn on, or was prepared on ahead of its first stage
        turn (an executor that takes stage turns)."""
        return next(
            (
                list(body["ordinals"])
                for _seq, root_kind, body in self._all()
                if root_kind in ("gpu.grant", "stage.prepare") and body["key"] == request + "#1"
            ),
            [],
        )

    def _all(self) -> list[tuple[int, str, dict[str, Any]]]:
        rows = []
        for root in ("A", "B", "C", "D", "J"):
            if self.executions.owns(OWNER, root):
                rows.extend(self.journal(root))
        return rows


def driverless(monkeypatch: pytest.MonkeyPatch) -> None:
    """This process asks no GPU driver, and opens no NVIDIA device node: NVML is absent and
    the driver's process tables list none of these executors, which never touch a device."""
    monkeypatch.setattr(accel, "_nvml_library", lambda: None)
    monkeypatch.setattr(
        accel,
        "process_memories",
        lambda pids, kind: {pid: accel.ProcessMemory("absent") for pid in pids},
    )
    monkeypatch.setattr(
        accel,
        "process_memories_by_device",
        lambda pids, kind, devices: {
            device: {pid: accel.ProcessMemory("absent") for pid in pids} for device in devices
        },
    )


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[Machine]:
    # Virtual cards no host has, unless the run opted in to its real ones.
    real = "real_gpu" in request.node.keywords
    devices = "0,1,2,3" if real else VIRTUAL
    if not real:
        driverless(monkeypatch)
    #: by GPU number in the envelope (0-3), as the tests name them
    unreadable: set[str] = set()
    free: dict[str, int] = {}

    def device_memory(entry: str, kind: str) -> accel.DeviceMemory:
        gpu = str(devices.split(",").index(entry))
        if gpu in unreadable:
            return accel.DeviceMemory("unreadable")
        return accel.DeviceMemory("measured", free.get(gpu, 80 * GiB), 80 * GiB)

    monkeypatch.setattr(accel, "device_memory", device_memory)
    # Short: an executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-gpu.", dir="/tmp") as root:
        made = Machine(Path(root), devices, "boot-one")
        made.unreadable = unreadable  # type: ignore[attr-defined]
        made.free = free  # type: ignore[attr-defined]
        try:
            yield made
        finally:
            made.close()


def test_grants_name_each_gpu_as_nvidia_smi_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker launched on cards 64 and 66 schedules its envelope's ordinals 0 and 1. Its
    grant and release name each GPU by nvidia-smi's number and UUID in `gpus`, beside the
    `ordinals` older readers still take."""
    measured = accel.DeviceMemory("measured", 80 * GiB, 80 * GiB)
    monkeypatch.setattr(accel, "device_memory", lambda entry, kind: measured)
    driverless(monkeypatch)
    inventory = tuple(
        RuntimeGPU(
            device_index=index,
            device_name="NVIDIA H100 80GB HBM3",
            device_uuid=f"GPU-{index}",
            driver_version="580.0",
            memory_bytes=80 * GiB,
            pci_bus_id=f"00000000:{index:02x}:00.0",
        )
        for index in (64, 66)
    )
    with tempfile.TemporaryDirectory(prefix="cz-gpu.", dir="/tmp") as root:
        machine = Machine(Path(root), "64,66", "boot-one", gpus=inventory)
        try:
            machine.root("A")
            call = machine.child("A", "h3")
            machine.tick()
            machine.finish(call)
            machine.tick()
            events = {
                kind: body
                for _, kind, body in machine.journal("A")
                if kind in ("gpu.grant", "gpu.release")
            }
        finally:
            machine.close()
    named = [{"gpu": 64, "uuid": "GPU-64"}, {"gpu": 66, "uuid": "GPU-66"}]
    assert events["gpu.grant"] == {"key": call + "#1", "ordinals": [0, 1], "gpus": named}
    assert events["gpu.release"]["ordinals"] == [0, 1]
    assert events["gpu.release"]["gpus"] == named


def test_an_older_root_holds_all_four_gpus_through_its_gap_and_b_starts_after(
    machine: Machine,
) -> None:
    machine.root("A")
    machine.root("B")
    a1 = machine.child("A", "h3")
    machine.await_granted(a1)
    b1 = machine.child("B", "h3")
    view = machine.tick()
    assert machine.granted(a1) == [0, 1, 2, 3]
    assert view["waiting"] == {b1 + "#1": ["A"]}
    machine.finish(a1)
    view = machine.tick()  # A is in Python between its children: B still waits
    assert view["leases"] == {"A": [0, 1, 2, 3]} and b1 + "#1" in view["waiting"]
    a2 = machine.child("A", "h3")
    machine.tick()
    assert machine.granted(a2) == [0, 1, 2, 3] and machine.granted(b1) == []
    machine.finish(a2)
    machine.finish("A")
    machine.tick()
    assert machine.granted(b1) == [0, 1, 2, 3]
    a_events = machine.journal("A")
    released = next(
        seq for seq, kind, body in a_events if kind == "gpu.lease" and not body["ordinals"]
    )
    b_grant = next(seq for seq, kind, body in machine.journal("B") if kind == "gpu.grant")
    b_wait = [body for _, kind, body in machine.journal("B") if kind == "gpu.wait"]
    assert b_wait == [
        {
            "key": b1 + "#1",
            "width": 4,
            "ordinals": [0, 1, 2, 3],
            "blocked_by": ["A"],
            "function": "h3",
        }
    ]
    # B's grant is written after A's release; both journals share one workspace clock order
    a_release_at = next(
        e.at_ms for e in machine.executions.events(OWNER, "A").events if e.sequence == released
    )
    b_grant_at = next(
        e.at_ms for e in machine.executions.events(OWNER, "B").events if e.sequence == b_grant
    )
    assert b_grant_at >= a_release_at


def test_three_reference_calls_fan_out_on_three_distinct_gpus(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded = machine.hold_activations(monkeypatch)
    machine.root("C")
    references = [machine.child("C", "qwen") for _ in range(3)]
    machine.await_granted(*references)
    assert sorted(tuple(machine.granted(r)) for r in references) == [(0,), (1,), (2,)]
    downloaded.set()
    view = machine.tick()
    assert view["leases"] == {"C": [0, 1, 2]} and not view["waiting"]
    # the shot that follows the references takes one 4-rank group on the same root
    for reference in references:
        machine.finish(reference)
    shot = machine.child("C", "h3")
    machine.tick()
    assert machine.granted(shot) == [0, 1, 2, 3]


def test_a_fifth_reference_waits_on_its_own_runs_calls_and_names_itself(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded = machine.hold_activations(monkeypatch)
    machine.root("C")
    references = [machine.child("C", "qwen") for _ in range(4)]
    machine.await_granted(*references)
    references.append(machine.child("C", "qwen"))
    own = "Waiting for GPU (needs 1, 4 in use by this run's other calls)"

    def stages() -> list[str]:
        return [
            canonical_json.decode(event.body)["payload"]["stage"]
            for event in machine.executions.events(OWNER, "C").events
            if event.kind == "progress"
        ]

    bound = time.monotonic() + 120  # a hang bound on a loaded box, not a budget
    while own not in stages():
        assert time.monotonic() < bound, stages()
        time.sleep(0.01)
    waits = [body for _, kind, body in machine.journal("C") if kind == "gpu.wait"]
    fifth = references[4] + "#1"
    assert waits == [
        {"key": fifth, "width": 1, "ordinals": [], "blocked_by": ["C"], "function": "qwen"}
    ]
    downloaded.set()
    machine.tick()
    # Each call's executor start is its own timed phase on the parent's record, never folded
    # into "Checking model inputs" (run 1516: 12.6 s of start and load read as the check).
    started = [
        fields
        for event in machine.executions.events(OWNER, "C").events
        if event.kind == "log"
        and (fields := canonical_json.decode(event.body)["payload"].get("fields", {})).get("phase")
        == "Starting model executor"
    ]
    assert sorted(row["child_request"] for row in started) == sorted(references)
    assert all(row["completed"] and row["elapsed_ms"] >= 0 for row in started), started
    assert "Starting model executor" in stages()


def test_weightless_calls_never_wait_and_a_device_job_waits_for_the_holder(
    machine: Machine,
) -> None:
    machine.root("A")
    a1 = machine.child("A", "h3")
    machine.tick()
    machine.finish(a1)
    machine.root("B")
    helper = machine.child("B", "helper")
    machine.device_job("J")
    view = machine.tick()
    assert helper + "#1" not in view["waiting"]
    assert machine.worker.gpu_status(helper).phase == "none"
    assert view["waiting"] == {"J#1": ["A"]}  # D1: no device job in A's gap
    machine.finish("A")
    machine.tick()
    assert machine.granted("J") == [0, 1, 2, 3]


def test_cancel_while_waiting_takes_no_grant_and_the_next_root_starts(machine: Machine) -> None:
    machine.root("A")
    machine.root("B")
    machine.root("C")
    a1 = machine.child("A", "h3")
    machine.await_granted(a1)
    b1 = machine.child("B", "h3")
    c1 = machine.child("C", "qwen")
    machine.tick()
    assert b1 + "#1" in machine.worker.stages.view()["waiting"]
    machine.control(b1, "cancel")
    machine.tick()
    assert b1 + "#1" not in machine.worker.stages.view()["waiting"] and machine.granted(b1) == []
    machine.control("A", "cancel")
    machine.finish(a1, pb.OUTCOME_STATUS_CANCELED)
    machine.finish("A", pb.OUTCOME_STATUS_CANCELED)
    machine.tick()
    assert machine.granted(c1) == [0]


def test_a_submitted_exact_width_is_the_group_and_leaves_the_rest_free(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit model GPU count reaches Runtime as `execution_gpus`: 2 of 4 GPUs is a
    2-GPU group, never the widest declared degree, and a count no slot declares or the
    machine cannot form is refused typed."""
    downloaded = machine.hold_activations(monkeypatch)
    machine.serving("A", "h3", gpus=2)
    # Establish the first holder before checking which cards remain available.
    # Independent preparation units may otherwise register B's demand first.
    machine.await_granted("A")
    machine.serving("B", "h3", gpus=2)
    machine.serving("C", "h3", gpus=3)
    machine.serving("D", "h3", gpus=8)
    machine.await_granted("A", "B")
    assert machine.granted("A") == [0, 1] and machine.granted("B") == [2, 3]
    view = machine.worker.stages.view()
    assert view["holders"] == {"0": "A#1", "1": "A#1", "2": "B#1", "3": "B#1"}
    assert not view["waiting"]
    downloaded.set()
    machine.tick()
    for refused, gpus in (("C", 3), ("D", 8)):
        assert machine.executions.status(OWNER, refused).state == "failed"
        outcome = machine.executions.collect(OWNER, refused).outcome_canonical_bytes
        message = documents.read(outcome, pb.AttemptOutcomeBody)["safe_message"]
        assert message.startswith("gpu_count_unavailable") and f"exactly {gpus} GPUs" in message


def test_work_this_machine_can_never_run_is_refused_at_admission(machine: Machine) -> None:
    """Never journaled, so it never queues and never holds the rental; runnable work is
    accepted through the same intake."""
    machine.serving("A", "h3", gpus=2, admit=True)
    assert machine.executions.owns(OWNER, "A")
    for request, gpus in (("C", 3), ("D", 8)):
        with pytest.raises(WorkspaceRefusal, match=f"gpu_count_unavailable: exactly {gpus} GPUs"):
            machine.serving(request, "h3", gpus=gpus, admit=True)
        assert not machine.executions.owns(OWNER, request)
    machine.unreadable.update({"0", "1", "2", "3"})  # type: ignore[attr-defined]
    with pytest.raises(WorkspaceRefusal, match="no_capacity: needs 1 GPUs; this machine reads 0"):
        machine.serving("E", "qwen", admit=True)
    assert not machine.executions.owns(OWNER, "E")


def test_an_unreadable_gpu_leaves_the_inventory_and_the_group_narrows(machine: Machine) -> None:
    machine.unreadable.add("3")  # type: ignore[attr-defined]
    machine.root("A")
    a1 = machine.child("A", "h3")
    machine.tick()
    assert machine.granted(a1) == [0, 1]  # 2 of the 3 readable GPUs; never GPU 3
    machine.root("B")
    b1 = machine.child("B", "qwen")
    machine.tick()
    assert machine.granted(b1) == [2]


def test_a_gpu_call_from_a_gpu_holder_is_refused_typed(machine: Machine) -> None:
    machine.root("A")
    parent = AttemptRecord("A", 1, b"", {}, lane_id="lane-0", state="running")
    machine.worker.engine.history[("A", 1)] = parent
    machine.worker.engine.live["A"] = parent
    child = machine.child("A", "qwen")
    machine.tick()
    assert machine.executions.status(OWNER, child).state == "failed"
    body = documents.read(
        machine.executions.collect(OWNER, child).outcome_canonical_bytes, pb.AttemptOutcomeBody
    )
    assert "nested_gpu_call" in body["safe_message"]


def test_sibling_replicas_activate_together_and_a_failure_refuses_only_its_call(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded = machine.hold_activations(monkeypatch)
    machine.root("C")
    references = [machine.child("C", "qwen") for _ in range(3)]
    machine.await_granted(*references)
    downloaded.set()
    machine.tick()
    # D7: all three are granted and activate beside each other, on their own units, without
    # a desired-state revision.
    assert sorted(tuple(machine.granted(r)) for r in references) == [(0,), (1,), (2,)]
    assert machine.worker.accepted.accepted_desired_state_revision == 0
    # These templates name no installed package: each activation fails typed, on its own,
    # and refuses only its own call; the root keeps its lease for its next call.
    for reference in references:
        body = documents.read(
            machine.executions.collect(OWNER, reference).outcome_canonical_bytes,
            pb.AttemptOutcomeBody,
        )
        assert "environment_materialization_input_missing" in body["safe_message"]
    view = machine.tick()
    assert view["leases"] == {"C": [0, 1, 2]} and not machine.worker.hosted


def test_machine_execution_state_reports_the_gpu_phase(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T11: `MachineExecutionState.gpu` as Creator reads it, at each transition."""
    rpc = MachineExecutionRPC(machine.worker)

    def gpu(request: str) -> pb.MachineExecutionGpu:
        return rpc._state(machine.executions.status(OWNER, request)).gpu

    downloaded = threading.Event()
    activate = Worker._activate_replica

    def downloading(self: Worker, *args: Any) -> None:
        downloaded.wait()  # the replica's weights arrive when the test says so
        activate(self, *args)

    monkeypatch.setattr(Worker, "_activate_replica", downloading)
    machine.root("A")
    machine.root("B")
    assert gpu("A").phase == "none"
    a1 = machine.child("A", "h3")
    machine.await_granted(a1)
    b1 = machine.child("B", "h3")
    bound = time.monotonic() + 120
    while gpu("A").phase != "granted" or gpu("B").phase != "waiting":
        assert time.monotonic() < bound
        time.sleep(0.01)
    assert (gpu("A").phase, list(gpu("A").ordinals)) == ("granted", [0, 1, 2, 3])
    assert gpu(a1).phase == "granted" and gpu(a1).width == 4
    waiting = gpu("B")
    assert (waiting.phase, waiting.width, list(waiting.blocked_by)) == ("waiting", 4, ["A"])
    assert gpu(b1) == waiting
    entered = AttemptRecord(a1, 1, b"", {}, lane_id="lane-0+1+2+3", state="running")
    machine.worker.engine.live[a1] = entered
    assert gpu("A").phase == "executing"
    del machine.worker.engine.live[a1]
    downloaded.set()  # its replica (no installed package) refuses it: the grant goes
    machine.tick()
    assert (gpu("A").phase, list(gpu("A").ordinals)) == ("holding", [0, 1, 2, 3])
    downloaded.clear()
    machine.finish("A")
    machine.await_granted(b1)
    assert gpu("B").phase == "granted"
    downloaded.set()
    machine.tick()


needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


@needs_executor
def test_consecutive_roots_of_different_widths_reuse_every_warm_executor(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """References at width 1, then a 4-rank shot, twice: the group and the singletons stay
    resident beside each other, and the second root spawns and retires nothing."""
    machine.activate_warm(monkeypatch)
    downloaded = machine.hold_activations(monkeypatch)
    machine.root("A")
    references = [machine.child("A", "qwen") for _ in range(3)]
    machine.await_granted(*references)
    downloaded.set()
    machine.tick()
    qwen = machine.warm()
    assert len(qwen) == 3
    for reference in references:
        machine.finish(reference)
    shot = machine.child("A", "h3")
    machine.tick()
    assert machine.granted(shot) == [0, 1, 2, 3]
    warm = machine.warm()
    assert len(warm) == 4 and {p: warm[p] for p in qwen} == qwen, "the shot retired nothing"
    lanes_of = {p: machine.worker._lane_of(p).lane_id for p in warm}
    degrees = {
        b.sequence_parallel_degrees()
        for h in machine.worker.hosted.values()
        for b in h.bindings.values()
    }
    assert degrees == {(), (2, 4)}
    assert sorted(lanes_of.values()) == ["lane-0", "lane-0+1+2+3", "lane-1", "lane-2"]
    epoch = machine.worker.admission_epoch
    machine.finish(shot)
    machine.finish("A")
    machine.root("B")
    references = [machine.child("B", "qwen") for _ in range(3)]
    machine.tick()
    # Each call is granted a card its warm replica already holds (warm-first).
    assert all(tuple(machine.granted(r)) in {(0,), (1,), (2,)} for r in references)
    for reference in references:
        machine.finish(reference)
    shot = machine.child("B", "h3")
    machine.tick()
    assert machine.granted(shot) == [0, 1, 2, 3]
    assert not machine.worker._activating, "every grant found its replica started"
    assert machine.executors() == warm, "no executor was spawned, replaced or retired"
    assert {p: machine.worker._lane_of(p).lane_id for p in warm} == lanes_of
    assert machine.worker.admission_epoch == epoch


@needs_executor
def test_two_long_forms_fanned_out_on_one_machine_both_run_their_group_calls(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs 1572 and 1573, on real executors: two roots' references fan out interleaved over
    the four GPUs, then each root asks for all four. Both group calls run, both roots end, and
    no wait is named behind a run that is named behind it, or behind its own run."""
    machine.activate_warm(monkeypatch)
    lock = threading.Lock()
    gates: dict[str, threading.Event] = {}

    def gate(request: str) -> threading.Event:
        with lock:
            if request not in gates:
                gates[request] = threading.Event()
                machine.holds.append(gates[request])
            return gates[request]

    activate = Worker._activate_replica

    def downloading(worker: Worker, *args: Any) -> None:
        gate(args[4]).wait()  # this call's weights arrive when the test says so
        activate(worker, *args)

    monkeypatch.setattr(Worker, "_activate_replica", downloading)

    def until(ready: Callable[[], bool]) -> None:
        bound = time.monotonic() + 120  # a hang bound on a loaded box, not a budget
        while not ready():
            assert time.monotonic() < bound
            time.sleep(0.01)

    def ended(*requests: str) -> bool:
        return all(machine.executions.status(OWNER, r).state in TERMINAL for r in requests)

    def waits(root: str) -> list[tuple[int, list[str]]]:
        at = {e.sequence: e.at_ms for e in machine.executions.events(OWNER, root).events}
        return [
            (at[s], body["blocked_by"])
            for s, kind, body in machine.journal(root)
            if kind == "gpu.wait"
        ]

    def said(root: str) -> list[str]:
        return [
            stage
            for event in machine.executions.events(OWNER, root).events
            if event.kind == "progress"
            and (stage := canonical_json.decode(event.body)["payload"]["stage"]).startswith(
                "Waiting for GPU"
            )
        ]

    machine.root("A")
    machine.root("B")
    references: dict[str, list[str]] = {"A": [], "B": []}
    for root in "ABAB":
        call = machine.child(root, "qwen")
        machine.await_granted(call)
        references[root].append(call)
    assert [machine.granted(r) for r in references["A"]] == [[0], [2]]
    assert [machine.granted(r) for r in references["B"]] == [[1], [3]]
    for call in references["A"]:
        gate(call).set()
    until(lambda: ended(*references["A"]))
    # A's segment asks for all four while B's references are still on GPUs 1 and 3.
    shot_a = machine.child("A", "h3")
    until(lambda: [b for _, b in waits("A")] == [["B"]])
    shot_b = machine.child("B", "h3")
    until(lambda: [b for _, b in waits("B")] == [["B"]])  # its own calls: A waits on them
    for call in references["B"]:
        gate(call).set()
    machine.await_granted(shot_a)
    assert machine.granted(shot_a) == [0, 1, 2, 3]
    until(lambda: [b for _, b in waits("B")] == [["B"], ["A"]])
    gate(shot_a).set()
    until(lambda: ended(shot_a))
    machine.finish("A")
    machine.await_granted(shot_b)
    assert machine.granted(shot_b) == [0, 1, 2, 3]
    gate(shot_b).set()
    until(lambda: ended(shot_b))
    machine.finish("B")
    machine.tick()

    assert {machine.executions.status(OWNER, r).state for r in "AB"} == {"succeeded"}
    for root in "AB":
        calls = [body for _, kind, body in machine.journal(root) if kind == "gpu.release"]
        assert len(calls) == 3 and all(body["cause"] == "exited" for body in calls), calls
    # Each wait names the GPUs it is placed on, as nvidia-smi numbers them.
    assert said("A") == ["Waiting for GPU (GPUs 64-67, behind B)"]
    assert said("B") == [
        "Waiting for GPU (GPUs 64-67, 2 in use by this run's other calls)",
        "Waiting for GPU (GPUs 64-67, behind A)",
    ]
    # B is named behind A only once A no longer waits on B: A's grant ends its wait.
    a_granted = next(
        e.at_ms
        for e in machine.executions.events(OWNER, "A").events
        if e.kind == "gpu.grant" and canonical_json.decode(e.body)["key"] == shot_a + "#1"
    )
    assert waits("B")[1][0] >= a_granted


def _warm_shot(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> tuple[str, dict[str, str]]:
    """Darkness 1513's shape: idle Qwen replicas on lanes 0-2 beside a warm 4-rank H3 group,
    each holding weights. Returns the shot and the placement on each lane."""
    machine.activate_warm(monkeypatch)
    machine.root("A")
    references = [machine.child("A", "qwen") for _ in range(3)]
    machine.tick()
    machine.warm()
    for reference in references:
        machine.finish(reference)
    shot = machine.child("A", "h3")
    machine.tick()
    machine.warm()
    worker = machine.worker
    for placement_id in worker.hosted:
        lane = worker._lane_of(placement_id)
        lane.row(placement_id).ledger.observe_construction(
            {
                "resident": {"text_encoder": 10 * GiB, "unet": 10 * GiB},
                "declared_scopes": {"encode": ["text_encoder"], "denoise": ["unet"]},
                "allocator_bytes": 20 * GiB,
                "reserved_bytes": 20 * GiB,
            }
        )
        lane.touch(placement_id)
    return shot, {worker._lane_of(p).lane_id: p for p in worker.hosted}


@needs_executor
def test_memory_pressure_vacates_idle_co_tenants_on_every_rank_device_and_kills_none(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Darkness 1513: the group's attempt freed only its lead card, and rank 3 refused
    `device_shortfall` beside an idle Qwen on card 3. A group short of room takes idle
    co-tenants off the rank cards it is short on, LRU-first. The shot's shape was never
    measured, so it cannot say what it needs: every rank's idle neighbour gives room, for a
    plane executor as for an older one. Their processes stay, and the caller is never its own
    victim."""
    shot, by_lane = _warm_shot(machine, monkeypatch)
    worker = machine.worker
    before = machine.executors()
    h3, q0, q1, q2 = (by_lane[k] for k in ("lane-0+1+2+3", "lane-0", "lane-1", "lane-2"))
    worker._lane_of(q2).touch(q2)  # q2 is now the most recently used

    def vacated() -> list[str]:
        steps = [e.step for e in worker.activity]
        return [s.split("'")[1] for s in steps if " vacated " in s or " cut to 0 B " in s]

    # The shot's shape was never measured, so it has every card of its group: every
    # rank's idle neighbour goes, least recently used first.
    machine.free["0"] = GiB // 2  # type: ignore[attr-defined]
    hosted = worker.hosted[h3]
    group = worker._lane_of(h3)
    row = group.row(h3)
    digest = next(iter(hosted.bindings))
    row.ledger.observe_device_free(GiB // 2, 80 * GiB)
    slot = AttemptSlot(
        supervision=hosted.supervision,
        ledger=row.ledger,
        chooser=row.chooser,
        bindings=hosted.bindings,
        failed_bindings={},
        lane_id=group.lane_id,
    )
    attempt = AttemptRecord(
        shot, 2, b"", {"serving": {"entrypoint_binding_digest": digest}}, placement_id=h3
    )
    with worker.stages.hold(group.ordinals, "the shot's turn"):
        worker._arbitrate_residency(attempt, slot)
        worker.memory.end(h3)
    assert hosted.supervision.current is not None
    first = [q0, q1, q2]
    assert vacated() == first
    # Qwen on card 1 needs room: the idle H3 group is its only co-tenant there
    machine.free.pop("0")  # type: ignore[attr-defined]
    machine.free["1"] = GiB  # type: ignore[attr-defined]
    worker.memory.make_room(worker._lane_of(q1), q1, None, "a reference")
    assert vacated() == [*first, h3]
    assert machine.executors() == before, "vacated, not killed: every process is the same"
    assert all(
        h.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
        for h in worker.hosted.values()
    )
    assert machine.granted(shot) == [0, 1, 2, 3]


@needs_executor
def test_a_waiting_tenant_is_short_only_while_its_gpu_lacks_room_for_it(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a call past its last stage is answered by. The tenant wanting its GPU next has
    room for its 20 GiB beside what is mapped: the weights stay hot. It lacks room: they are
    unmapped before the turn passes. What cannot be told is short."""
    _shot, by_lane = _warm_shot(machine, monkeypatch)
    worker = machine.worker
    q0 = by_lane["lane-0"]
    gpus = worker._lane_of(q0).ordinals
    need = memory.StageNeed(entrypoint="generate")
    assert worker.memory.short("machine-unknown", need, gpus)
    current = worker.hosted[q0].supervision.current
    assert current is not None
    if not memory.plane_capable(current):
        assert worker.memory.short(q0, need, gpus), "an older executor is never told to stay"
        return
    machine.free["0"] = 30 * GiB  # type: ignore[attr-defined]
    assert not worker.memory.short(q0, need, gpus)
    machine.free["0"] = GiB  # type: ignore[attr-defined]
    assert worker.memory.short(q0, need, gpus)


@needs_executor
def test_a_tenants_budget_leaves_what_an_executor_waiting_to_start_wants(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While an executor waits to start on a GPU, a call that takes its first turn there is
    budgeted around the room that executor wants, so the room it waits for is not taken. A
    stage whose activations were never measured gets no grant: it measures them first."""
    _shot, by_lane = _warm_shot(machine, monkeypatch)
    worker = machine.worker
    q0 = by_lane["lane-0"]
    lane = worker._lane_of(q0)
    current = worker.hosted[q0].supervision.current
    assert current is not None
    if not memory.plane_capable(current):
        return
    unmeasured = worker.memory.start(q0, memory.StageNeed(entrypoint="generate"), lane.ordinals)
    worker.memory.end(q0)
    assert isinstance(unmeasured, memory.Budget) and unmeasured.plane_bytes == -1
    need = memory.StageNeed(entrypoint="generate", activation_bytes=0)
    machine.free["0"] = 30 * GiB  # type: ignore[attr-defined]

    def budget() -> dict[int, int]:
        granted = worker.memory.start(q0, need, lane.ordinals)
        worker.memory.end(q0)
        assert isinstance(granted, memory.Budget)
        return granted.vram

    assert budget() == {0: 30 * GiB - weight_policy.MARGIN}
    worker.memory.wanted["machine-starting"] = {0: GiB}
    assert budget() == {0: 29 * GiB - weight_policy.MARGIN}
    worker.memory.wanted[q0] = {0: 5 * GiB}
    assert budget() == {0: 29 * GiB - weight_policy.MARGIN}, "its own want is not held against it"


@needs_executor
def test_idle_contexts_are_reclaimed_least_recently_used_first_only_where_room_is_short(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every construction that ever ran leaves a process, and its device context, on its GPU
    (8 of them took ~19 GB of one H100 and a later load did not fit). An idle executor stays
    warm while there is room. When a measured need is still short after every idle weight
    byte was cut, idle processes end, least recently used first, until it fits: only on the
    short GPUs, never the asker's own, and each restarts at its next grant."""
    shot, by_lane = _warm_shot(machine, monkeypatch)
    worker = machine.worker
    h3 = by_lane["lane-0+1+2+3"]
    q0, q1, q2 = (by_lane[k] for k in ("lane-0", "lane-1", "lane-2"))
    group = worker._lane_of(h3)
    before = machine.executors()
    for hosted in worker.hosted.values():  # each was started on its GPUs: it holds a context
        assert hosted.supervision.current is not None
        hosted.supervision.current.started = {"ok": True}
    context = 2 * GiB
    reclaim = worker.memory.reclaim

    def ended(tenant: str, executor: Any, why: str) -> bool:
        done = reclaim(tenant, executor, why)
        gpu = str(worker._lane_of(tenant).ordinals[0])
        machine.free[gpu] += context  # type: ignore[attr-defined]  # what the driver reads next
        return done

    monkeypatch.setattr(worker.memory, "reclaim", ended)
    machine.free.update({"0": GiB, "1": GiB})  # type: ignore[attr-defined]
    need = {0: 3 * GiB, 1: 3 * GiB, 2: 3 * GiB}

    worker.memory.make_room(group, h3, need, "a load")
    assert machine.executors() == before, "cuts alone end no process"
    worker.memory.make_room(group, h3, need, "a load", contexts=True)
    assert set(before) - set(machine.executors()) == {q0, q1}, "GPU 2 had room: q2 stays warm"
    assert h3 in machine.executors() and q2 in machine.executors()
    steps = [e.step for e in worker.activity if "holds only its device context" in e.step]
    assert [s.split("'")[1] for s in steps] == [q0, q1], "least recently used first"
    assert machine.granted(shot) == [0, 1, 2, 3]


@needs_executor
def test_a_busy_tenant_outside_its_turn_gives_room_through_its_budget_cell(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An executor that runs out of device memory mid-call asks for room (`DeviceRoom`). A
    tenant busy outside its turn (a load, an attempt's tail) gives it at its next block
    boundary through its budget cell; a tenant in its turn is computing and keeps its room."""
    shot, by_lane = _warm_shot(machine, monkeypatch)
    worker = machine.worker
    h3, q0 = by_lane["lane-0+1+2+3"], by_lane["lane-0"]
    busy = worker.hosted[q0].supervision.current
    assert busy is not None
    cell, fd = budget_cell.Cell.create()
    executor_side = budget_cell.Cell(fd)
    os.close(fd)
    busy.cell = cell
    assert busy._calls.acquire(blocking=False)  # a call holds its seam: it is busy
    applied: list[int] = []

    def block_boundaries() -> None:  # the executor's side: one ask, applied at a boundary
        while (wanted := executor_side.poll()) is None:
            time.sleep(0.005)
        applied.append(wanted + 512 * (1 << 20))  # never below its open stages' floor
        executor_side.acknowledge(applied[-1])

    group = worker._lane_of(h3)
    machine.free["0"] = GiB  # type: ignore[attr-defined]
    worker.memory.active.add(q0)
    worker.memory.make_room(group, h3, {0: 4 * GiB}, "an op of the shot")
    assert not applied, "a tenant in its turn is computing: it keeps its room"
    worker.memory.active.discard(q0)
    boundary = threading.Thread(target=block_boundaries, daemon=True)
    boundary.start()
    worker.memory.make_room(group, h3, {0: 4 * GiB}, "an op of the shot")
    boundary.join(30)
    busy._calls.release()
    assert applied == [512 * (1 << 20)]
    steps = [e.step for e in worker.activity]
    assert [s for s in steps if f"{q0!r}" in s and "cut mid-call to 536870912 B" in s], steps
    assert machine.granted(shot) == [0, 1, 2, 3]


_RANK = """
import signal, sys, torch
from cozy_runtime.internal import proctree
if sys.argv[1]:
    proctree.join_executor_cgroup(proctree.CgroupScope(sys.argv[1], int(sys.argv[2])))
signal.signal(signal.SIGTERM, lambda *_: print("ignored SIGTERM", flush=True))
held = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
print("held", flush=True)
sys.stdin.read()
"""


def _rank(executor: Any) -> subprocess.Popen[str]:
    """A follower rank in `executor`'s scope, as the worker's launcher starts one: the
    worker's own child, holding device memory, ignoring SIGTERM (darkness: every executor
    rode out SIGTERM for 15 s)."""
    scope = executor.scope
    cgroup = isinstance(scope, proctree.CgroupScope)
    env = dict(os.environ)  # the run's own device visibility, never widened here
    if not cgroup:
        env[child_env.EXECUTOR_SCOPE_ENV] = scope.token
    rank = subprocess.Popen(
        [sys.executable, "-c", _RANK]
        + ([scope.relative_path, str(scope.inode)] if cgroup else ["", ""]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert rank.stdout is not None
    assert rank.stdout.readline().strip() == "held"
    rank.send_signal(signal.SIGTERM)
    assert rank.stdout.readline().strip() == "ignored SIGTERM"
    return rank


def _card_rows(*pids: int) -> dict[int, str]:
    rows = accel.process_memories_by_device(set(pids), accel.host_backend_family(), ("0",))
    return {pid: rows["0"][pid].state for pid in pids}


@needs_executor
@pytest.mark.real_gpu
def test_a_broken_group_is_reaped_before_its_cards_are_granted_again(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Darkness 1514: the 1513 group failed `group_broken` and was invalidated, but its
    rank 0 lived on holding 62 GB of card 0 with its ranks unreaped zombies, and the next
    Qwen call there OOMed. The lane's post-attempt pass, still holding the cards the
    failed attempt ran on, now kills the whole generation (SIGKILL: it ignores SIGTERM),
    reaps every process and has the driver agree before the cards go back; only the
    successor waits for the next grant."""
    # Asked only when the run opted in (`--real-gpu`), never at import: importing this module
    # reaches no driver. A run that hides its GPUs forbids the card even when opted in.
    if (
        gpu_hidden()
        or importlib.util.find_spec("torch") is None
        or _card_rows(os.getpid())[os.getpid()] == "unreadable"
    ):
        pytest.skip("a rank holding card 0 needs torch and a visible, driver-readable card")
    shot, by_lane = _warm_shot(machine, monkeypatch)
    worker = machine.worker
    h3 = by_lane["lane-0+1+2+3"]
    hosted, group = worker.hosted[h3], worker._lane_of(h3)
    broken = hosted.supervision.current
    assert broken is not None
    rank = _rank(broken)
    assert _card_rows(rank.pid)[rank.pid] == "present"
    others = {p: e for p, e in machine.executors().items() if p != h3}
    # The failed handler's reply poisons the generation; run_attempt then rebuilds its
    # lane inside the whole turn it holds for the attempt.
    hosted.supervision.invalidate(broken, "failed/group_broken")
    with worker.stages.hold(group.ordinals, "the failed attempt's turn"):
        worker._rebuild_lane(group, h3)
        for pid in (broken.pid, rank.pid):
            assert not Path(f"/proc/{pid}").exists(), f"pid {pid} is alive or unreaped"
        assert _card_rows(rank.pid)[rank.pid] == "absent"
        assert hosted.supervision.current is None
        assert group.row(h3).resident_bytes() == 0
    assert worker._dead_replicas[h3] is broken
    assert {p: e for p, e in machine.executors().items() if p != h3} == others
    machine.finish(shot, pb.OUTCOME_STATUS_FAILED)
    machine.finish("A")
    # The next shot revives the replica on a fresh generation, on its grant.
    monkeypatch.setattr(
        Worker,
        "_prepare_hosted_executor",
        lambda worker, hosted, fresh=True: machine.start(hosted),
    )
    machine.root("B")
    again = machine.child("B", "h3")
    machine.tick()
    assert machine.granted(again) == [0, 1, 2, 3]
    epoch, pid = machine.executors()[h3]
    assert epoch > broken.epoch and pid != broken.pid, (epoch, pid)
    assert h3 not in worker._dead_replicas
