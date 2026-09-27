"""GPU scheduling: the policy module, then the real Worker drive loop over a durable journal.

Worker tests use a real `Worker` with four device entries, a real claim, real `Executions`
and child-call journal rows, and real `drive_executions` ticks. Executors are never
started (the convergence thread is not running), so a granted placement stays pending;
what is asserted is the scheduler decision as the durable journal records it. The only
fake is the hardware memory seam, `accel.device_memory`.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import accel, child_env, package_interface
from cozy_runtime.internal.call_intent import canonical_intent
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.worker import lanes, machine_model_defaults
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.gpu_scheduler import (
    Demand,
    GpuScheduler,
    model_degrees,
    width_for,
)
from cozy_runtime.internal.worker.machine_execution_rpc import MachineExecutionRPC
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.internal.worker.workspace_executions import TERMINAL
from cozy_runtime.protocol import WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb

GiB = 1 << 30
OWNER = "owner"
H3 = [{"sequence_parallel": {"degrees": [2, 4, 7, 8]}}] * 2
MARCO = Path(__file__).resolve().parent.parent / "examples" / "marco-polo"


# ------------------------------------------------------------------------------ policy


def test_width_is_the_largest_declared_degree_the_machine_forms() -> None:
    # base + LoRA both (2,4,7,8): never an undeclared width; the rest of the cards stay free
    assert [width_for(model_degrees(H3), n) for n in range(1, 10)] == [1, 2, 2, 4, 4, 4, 7, 8, 8]
    assert width_for(model_degrees([{}]), 4) == 1  # no declaration is its declared truth
    assert width_for(model_degrees([{"sequence_parallel": {"degrees": "2"}}]), 4) == 1


def test_default_ladder_picks_the_widest_rung_so_h3_is_one_four_gpu_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from cozy_runtime.internal import hostfacts

    ladder = [{"gpu": "H100", "gpus": 2}, {"gpu": "H100", "gpus": 4}, {"gpu": "H200", "gpus": 1}]
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
        executions=SimpleNamespace(
            capture_root=lambda *_: "root", capture=lambda *_: {"model_defaults": rows}
        ),
        lanes=SimpleNamespace(entries=("0", "1", "2", "3")),
    )
    target = SimpleNamespace(
        installation_id="h3",
        entrypoint="ref2va_turbo",
        declaration={
            "models": [{"path": "ref2va_turbo.models." + n} for n in ("base_model", "turbo_lora")]
        },
    )
    monkeypatch.setattr(
        machine_model_defaults.hostfacts,
        "measure",
        lambda _: hostfacts.HostFacts(gpu_name="NVIDIA H100 80GB HBM3", gpu_count=4),
    )
    call = SimpleNamespace(parent_request="parent")
    selected = machine_model_defaults.select(worker, OWNER, call, target, {})  # type: ignore[arg-type]
    assert [row["gpus"] for row in selected] == [4, 4]
    assert width_for(model_degrees(H3), len(worker.lanes.entries)) == 4
    worker.lanes = SimpleNamespace(entries=("0", "1"))
    selected = machine_model_defaults.select(worker, OWNER, call, target, {})  # type: ignore[arg-type]
    assert [row["gpus"] for row in selected] == [2, 2]


def test_a_root_keeps_its_lease_through_the_gap_and_the_head_waiter_reserves() -> None:
    gpus = GpuScheduler(4)
    roots = {"A": 1, "B": 2}
    assert gpus.sync(roots, [Demand("a1#1", "A", 4)]) == {"a1#1": (0, 1, 2, 3)}
    assert gpus.sync(roots, [Demand("b1#1", "B", 1)]) == {}  # a1 ended: A's gap
    assert gpus.view()["leases"] == {"A": [0, 1, 2, 3]}
    assert gpus.view()["waiting"] == {"b1#1": ["A"]}
    grants = gpus.sync(roots, [Demand("a2#1", "A", 4), Demand("b1#1", "B", 1)])
    assert grants == {"a2#1": (0, 1, 2, 3)}  # A's next child reuses its own ordinals
    assert gpus.sync({"B": 2}, [Demand("b1#1", "B", 1)]) == {"b1#1": (0,)}  # A stopped
    # Nothing moves a younger holder: an older root waits for it and reserves what frees.
    gpus = GpuScheduler(4)
    assert gpus.sync({"Y": 5}, [Demand("y#1", "Y", 2)]) == {"y#1": (0, 1)}
    live = [Demand("y#1", "Y", 2), Demand("o#1", "O", 4), Demand("z#1", "Z", 1)]
    assert gpus.sync({"O": 1, "Y": 5, "Z": 9}, live) == {"y#1": (0, 1)}
    assert gpus.view()["leases"] == {"Y": [0, 1], "O": [2, 3]}  # Z cannot take a reserved GPU
    assert gpus.sync({"O": 1, "Z": 9}, live[1:]) == {"o#1": (0, 1, 2, 3)}


def test_fifo_within_a_root_warm_first_and_release_before_the_next_tick() -> None:
    gpus = GpuScheduler(4)
    roots = {"A": 1}
    demands = [Demand("h3#1", "A", 4), Demand("qwen#1", "A", 1)]
    gpus.sync(roots, [Demand("ref#1", "A", 1)])
    assert gpus.sync(roots, [Demand("ref#1", "A", 1), *demands]) == {"ref#1": (0,)}
    gpus.release("ref#1")
    assert gpus.sync(roots, [Demand("ref#1", "A", 1), *demands])["h3#1"] == (0, 1, 2, 3)
    # warm: the placement already resident on ordinal 2 is granted there
    gpus = GpuScheduler(4)
    assert gpus.sync({"A": 1}, [Demand("q#1", "A", 1, warm=(2,))]) == {"q#1": (2,)}
    # excluded ordinals (unreadable, or held by a direct offer) are never granted
    assert gpus.sync({"A": 1}, [Demand("q#2", "A", 3, exclude=(2,))]) == {"q#2": (0, 1, 3)}


def test_a_stopped_roots_attempt_keeps_its_devices_until_it_leaves_them() -> None:
    gpus = GpuScheduler(2)
    assert gpus.sync({"A": 1}, [Demand("a#1", "A", 2)]) == {"a#1": (0, 1)}
    # A was cancelled; its attempt is still on the device
    assert gpus.sync({"B": 2}, [Demand("a#1", "A", 2), Demand("b#1", "B", 1)]) == {"a#1": (0, 1)}
    ranks = [{"rank": r, "pid": 70 + r, "ordinal": r, "start_us": 1, "end_us": 2} for r in (0, 1)]
    gpus.release("a#1", ranks=ranks)
    assert gpus.sync({"B": 2}, [Demand("a#1", "A", 2), Demand("b#1", "B", 1)]) == {"b#1": (0,)}
    events = gpus.drain()
    kinds = [kind for _root, kind, _body in events]
    assert kinds.count("gpu.grant") == 2 and "gpu.wait" in kinds
    exit_ = next(body for _root, kind, body in events if kind == "gpu.release")
    assert exit_ == {"key": "a#1", "ordinals": [0, 1], "cause": "exited", "ranks": ranks}


def test_group_lanes_bind_exactly_the_grant_or_refuse_typed() -> None:
    lane_set = lanes.LaneSet.from_envelope("0,1,2,3", worker_pid=os.getpid())
    ok = accel.DeviceMemory("measured", 80 * GiB, 80 * GiB)
    measured = dict.fromkeys(range(4), ok)
    group = lane_set.bind("h3", (0, 1, 2, 3), model_bearing=True, measured=measured)
    assert group.ordinals == (0, 1, 2, 3) and group.lane_id == "lane-0+1+2+3"
    lane_set.release("h3")
    with pytest.raises(lanes.LaneRefusal) as refused:
        lane_set.bind(
            "h3",
            (0, 1, 2, 3),
            model_bearing=True,
            measured={**measured, 3: accel.DeviceMemory("unreadable")},
        )
    assert refused.value.code == "device_group_infeasible"
    lane_set.bind("qwen", (1,), model_bearing=True, measured=measured)
    with pytest.raises(lanes.LaneRefusal) as refused:
        lane_set.bind("h3", (0, 1, 2, 3), model_bearing=True, measured=measured)
    assert refused.value.code == "device_pin_infeasible"  # never a smaller group
    helper = lane_set.bind("helper", (), model_bearing=False, measured={})
    assert helper.ordinals == () and helper.devices == "" and lane_set.lane_of("helper") is helper
    assert lane_set.release("helper") and lane_set.lane_of("helper") is None


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

    def __init__(self, root: Path, devices: str, boot: str) -> None:
        base = {
            k: v
            for k, v in os.environ.items()
            if not child_env.erased(k) and k not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
        }
        config = RuntimeConfig(
            cozy_home=root / "home",
            credentials=Credentials(),
            child_base_env=tuple(sorted(base.items())),
        )
        self.worker = Worker(
            replace(config, bootstrap_credential="test-bootstrap"),
            WorkerOptions(
                root=root / "worker",
                tensorfs_root=root / "store",
                devices=devices,
                worker_boot_id=boot,
            ),
            InMemoryControlHost(),
        )
        claim = pb.Claim(
            record_owner_id=OWNER,
            record_owner_epoch=1,
            wire_minor=WIRE_MINOR,
            proof=b"test-bootstrap",
        )
        assert self.worker.accept_claim(claim, lambda frame: None)[0] >= 1
        assert self.worker.executions is not None and self.worker.machine_calls is not None
        self.executions = self.worker.executions
        self.calls = self.worker.machine_calls
        self.children: dict[str, int] = {}

    def submit(
        self,
        request: str,
        spec: pb.InvocationSpec,
        preparation: dict[str, Any] | None,
        placement: str = "",
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
        self.executions.submit(
            OWNER,
            request,
            hashlib.sha256(request.encode()).digest(),
            offer,
            expected_execution_workspace_id=self.executions.workspace_id,
            worker_boot=self.worker.fence.worker_boot_id,
            preparation=canonical_json.encode(preparation) if preparation else b"",
        )

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
        )
        selected = self.executions.offer(OWNER, name)
        accepted = pb.AttemptAccepted(
            request_id=name,
            attempt_ordinal=1,
            invocation_spec_digest=selected.invocation_spec_digest,
        )
        self.executions.workspace.mark_running(OWNER, accepted)
        self.executions.observe(OWNER, pb.WorkerFrame(attempt_accepted=accepted))

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
            ),
        )
        self.submit(
            call.child_request,
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
        )
        return call.child_request

    def finish(self, request: str, status: pb.OutcomeStatus = pb.OUTCOME_STATUS_SUCCEEDED) -> None:
        """Record the attempt's outcome; a child whose replica could not activate here
        (these templates name no installed package) was already refused typed."""
        if self.executions.status(OWNER, request).state in TERMINAL:
            return
        selected = self.executions.offer(OWNER, request)
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
        self.executions.observe(OWNER, pb.WorkerFrame(attempt_outcome=terminal))

    def tick(self) -> dict[str, Any]:
        self.worker.drive_executions()
        for thread in list(self.worker._activating.values()):
            thread.join()  # replica activation runs off the tick; settle it deterministically
        return self.worker.gpu.view()

    def journal(self, root: str) -> list[tuple[int, str, dict[str, Any]]]:
        """The root's gpu.* observations, in journal order."""
        page = self.executions.events(OWNER, root)
        return [
            (event.sequence, event.kind, canonical_json.decode(event.body))
            for event in page.events
            if event.kind.startswith("gpu.")
        ]

    def granted(self, request: str) -> list[int]:
        return next(
            (
                list(body["ordinals"])
                for _seq, root_kind, body in self._all()
                if root_kind == "gpu.grant" and body["key"] == request + "#1"
            ),
            [],
        )

    def _all(self) -> list[tuple[int, str, dict[str, Any]]]:
        rows = []
        for root in ("A", "B", "C", "J"):
            if self.executions.owns(OWNER, root):
                rows.extend(self.journal(root))
        return rows


@pytest.fixture
def machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    unreadable: set[str] = set()

    def device_memory(entry: str, kind: str) -> accel.DeviceMemory:
        if entry in unreadable:
            return accel.DeviceMemory("unreadable")
        return accel.DeviceMemory("measured", 80 * GiB, 80 * GiB)

    monkeypatch.setattr(accel, "device_memory", device_memory)
    made = Machine(tmp_path, "0,1,2,3", "boot-one")
    made.unreadable = unreadable  # type: ignore[attr-defined]
    try:
        yield made
    finally:
        made.worker.shutdown()


def test_an_older_root_holds_all_four_gpus_through_its_gap_and_b_starts_after(
    machine: Machine,
) -> None:
    machine.root("A")
    machine.root("B")
    a1 = machine.child("A", "h3")
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
    assert b_wait == [{"key": b1 + "#1", "width": 4, "blocked_by": ["A"]}]  # once, not per tick
    # B's grant is written after A's release; both journals share one workspace clock order
    a_release_at = next(
        e.at_ms for e in machine.executions.events(OWNER, "A").events if e.sequence == released
    )
    b_grant_at = next(
        e.at_ms for e in machine.executions.events(OWNER, "B").events if e.sequence == b_grant
    )
    assert b_grant_at >= a_release_at


def test_three_reference_calls_fan_out_on_three_distinct_gpus(machine: Machine) -> None:
    machine.root("C")
    references = [machine.child("C", "qwen") for _ in range(3)]
    view = machine.tick()
    assert sorted(tuple(machine.granted(r)) for r in references) == [(0,), (1,), (2,)]
    assert view["leases"] == {"C": [0, 1, 2]} and not view["waiting"]
    # the shot that follows the references takes one 4-rank group on the same root
    for reference in references:
        machine.finish(reference)
    shot = machine.child("C", "h3")
    machine.tick()
    assert machine.granted(shot) == [0, 1, 2, 3]


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
    b1 = machine.child("B", "h3")
    c1 = machine.child("C", "qwen")
    machine.tick()
    state = machine.executions.status(OWNER, b1)
    machine.executions.control(OWNER, b1, "cancel-b1", state.generation, "cancel")
    machine.tick()
    assert b1 + "#1" not in machine.worker.gpu.view()["waiting"] and machine.granted(b1) == []
    state = machine.executions.status(OWNER, "A")
    machine.executions.control(OWNER, "A", "cancel-a", state.generation, "cancel")
    machine.finish(a1, pb.OUTCOME_STATUS_CANCELED)
    machine.tick()
    assert machine.granted(c1) == [0]


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


def test_a_restarted_worker_orders_by_durable_root_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        accel,
        "device_memory",
        lambda entry, kind: accel.DeviceMemory("measured", 80 * GiB, 80 * GiB),
    )
    first = Machine(tmp_path, "0,1,2,3", "boot-guarded")
    try:
        first.root("A")
        first.root("B")
        b1 = first.child("B", "h3")
        first.tick()
        assert first.granted(b1) == [0, 1, 2, 3]
    finally:
        first.worker.shutdown()
    second = Machine(tmp_path, "0,1,2,3", "boot-guarded")
    try:
        second.children = {"A": -1, "B": 0}
        a1 = second.child("A", "h3")
        view = second.tick()
        assert list(view["grants"]) == [a1 + "#1"]  # A was accepted first; nothing replayed
        assert view["waiting"] == {b1 + "#1": ["A"]}
    finally:
        second.worker.shutdown()


def test_sibling_replicas_activate_together_and_a_failure_refuses_only_its_call(
    machine: Machine,
) -> None:
    machine.root("C")
    references = [machine.child("C", "qwen") for _ in range(3)]
    machine.tick()
    replicas = [p for p in machine.worker.hosted if p.startswith("machine-")]
    # D7: all three bind in one tick, beside each other, without a desired-state revision
    assert sorted(machine.worker.hosted[p].placement.device_pin for p in replicas) == [
        (0,),
        (1,),
        (2,),
    ]
    assert machine.worker.accepted.accepted_desired_state_revision == 0
    # These templates name no installed package: each activation fails typed, on its own,
    # and refuses only its own call; the root keeps its lease for its next call.
    machine.tick()
    for reference in references:
        body = documents.read(
            machine.executions.collect(OWNER, reference).outcome_canonical_bytes,
            pb.AttemptOutcomeBody,
        )
        assert "environment_materialization_input_missing" in body["safe_message"]
    view = machine.tick()
    assert view["leases"] == {"C": [0, 1, 2]} and not machine.worker.hosted


def test_machine_execution_state_reports_the_gpu_phase(machine: Machine) -> None:
    """T11: `MachineExecutionState.gpu` as Creator reads it, at each transition."""
    rpc = MachineExecutionRPC(machine.worker)

    def gpu(request: str) -> pb.MachineExecutionGpu:
        return rpc._state(machine.executions.status(OWNER, request)).gpu

    machine.root("A")
    machine.root("B")
    assert gpu("A").phase == "none"
    a1 = machine.child("A", "h3")
    b1 = machine.child("B", "h3")
    machine.worker.drive_executions()  # granted, its replica not yet on the device
    assert (gpu("A").phase, list(gpu("A").ordinals)) == ("granted", [0, 1, 2, 3])
    assert gpu(a1).phase == "granted" and gpu(a1).width == 4
    waiting = gpu("B")
    assert (waiting.phase, waiting.width, list(waiting.blocked_by)) == ("waiting", 4, ["A"])
    assert gpu(b1) == waiting
    entered = AttemptRecord(a1, 1, b"", {}, lane_id="lane-0+1+2+3", state="running")
    machine.worker.engine.live[a1] = entered
    assert gpu("A").phase == "executing"
    del machine.worker.engine.live[a1]
    machine.finish(a1)
    machine.tick()
    assert (gpu("A").phase, list(gpu("A").ordinals)) == ("holding", [0, 1, 2, 3])
    machine.finish("A")
    machine.tick()
    assert gpu("B").phase == "granted"
