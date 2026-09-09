"""Group lanes (cr-068): one executor sealed to K devices, K ranks under one cgroup.

Every arm below drives the REAL code path — the real `LaneSet`, the real worker, a real
executor process sealed by the real trampoline, real follower executor processes spawned by
rank 0 over a real `socketpair` seam and a real torch.distributed rendezvous. Nothing is
patched.

WHAT IS REAL AND WHAT IS OWED. This box has ONE card. Proven here on the RTX 4070: the
fuse/split bookkeeping and every typed pin refusal; the per-device ledger rows and the MIN
ceiling; the chooser refusing the ladder under a group; a group is ONE seat and the wire
(`lanes[]`, `device_lane_id`, `device_pins`; proto-024) says so; the author's opt-in
reaching the PackageInterface and its fixed-point scan; an executor sealed to `0,1` reading both
entries and NVLS-off from its own `/proc`
environment; the seal, degree and construction refusals of a group prepare; the reclaim
proof iterating K devices; a degree-1 prepare byte-identical to a plain one. Proven on GLOO
(torch in this interpreter, CPU process group, real follower executor processes): the
rendezvous on a kernel-chosen port, the barrier, the broadcast plan's divergence refusal, a
follower's death observed through its seam, a follower that never dials back judged by its
own meter. OWED to a two-card box, skipping with the reason: the NCCL group over two real
devices, the mirrored Ulysses attempt, follower-death poisoning mid-attempt, and the
per-device absence proof over two driver-readable cards.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author._signature import ModelBinding
from cozy_runtime.internal import accel, canonical, package_interface, spawn
from cozy_runtime.internal.config import seal_snapshot
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.parallel import cp, wire
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.parallel.plan import GroupPlan, GroupRefusal, RankDivergence
from cozy_runtime.internal.worker import lane_wire, lanes
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    PlanChooser,
    PlanRefusal,
    PreparedModel,
    PreparedRequest,
)
from cozy_runtime.internal.worker.plan import ModelBinding as WorkerSlot
from cozy_runtime.internal.worker.session import (
    HostedPlacement,
    Placement,
    Tenant,
    Worker,
    WorkerOptions,
    executor_prepare_command,
)
from cozy_runtime.protocol import documents as documents_
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import PACKAGE, _config, _workspace
from test_end_to_end import NO_EXECUTOR

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")

GiB = 1 << 30


def _torch_here() -> str:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return f"torch must be importable in the test interpreter for the gloo arms: {exc}"
    return ""


NO_TORCH = _torch_here()
needs_torch = pytest.mark.skipif(bool(NO_TORCH), reason=NO_TORCH or "")


MEASURED = {
    0: accel.DeviceMemory("measured", 6 * GiB, 8 * GiB),
    1: accel.DeviceMemory("measured", 4 * GiB, 8 * GiB),
    2: accel.DeviceMemory("unreadable"),
}


# --------------------------------------------------------------------- bookkeeping


def test_fuse_and_split_are_typed_and_bump_the_revision() -> None:
    """FUSE replaces K empty device lanes with ONE group lane whose seal names all K entries
    and whose id joins the ordinals; SPLIT restores them; each moves `revision` (the
    worker's admission-epoch cue). A group never nests and never steals a device."""
    lanes_ = lanes.LaneSet.from_envelope("0,1,2", worker_pid=os.getpid())
    assert lanes_.revision == 0 and len(lanes_) == 3
    fused = lanes_.fuse((0, 1))
    assert fused.lane_id == "lane-0+1" and fused.ordinals == (0, 1) and fused.devices == "0,1"
    assert fused.group and fused.degree == 2 and fused.entries == ("0", "1")
    assert [lane.lane_id for lane in lanes_.lanes] == ["lane-0+1", "lane-2"]
    assert lanes_.by_id["lane-0+1"] is fused and "lane-0" not in lanes_.by_id
    assert lanes_.holder_of(1) is fused and lanes_.revision == 1
    assert lanes_.envelope.ordinals == (0, 1, 2), "the job's envelope lane is untouched"
    # a group is ONE seat regardless of K (cr-066's per-lane constant, K-independent)
    assert lanes.LANE_SEATS == 1
    with pytest.raises(lanes.LaneRefusal) as nested:
        lanes_.fuse((1, 2))
    assert nested.value.code == "device_pin_infeasible" and "held by group" in nested.value.detail
    with pytest.raises(lanes.LaneRefusal) as outside:
        lanes_.fuse((2, 3))
    assert outside.value.code == "device_pin_infeasible" and "outside" in outside.value.detail
    with pytest.raises(lanes.LaneRefusal):
        lanes_.fuse((1,))  # K >= 2
    fused.placements.add("p")
    with pytest.raises(lanes.LaneRefusal) as held:
        lanes_.split(fused)
    assert "still holds ['p']" in held.value.detail
    fused.placements.discard("p")
    parts = lanes_.split(fused)
    assert [lane.lane_id for lane in parts] == ["lane-0", "lane-1"]
    assert [lane.lane_id for lane in lanes_.lanes] == ["lane-0", "lane-1", "lane-2"]
    assert lanes_.revision == 2 and "lane-0+1" not in lanes_.by_id


def test_a_group_pin_refuses_typed_then_fuses() -> None:
    lanes_ = lanes.LaneSet.from_envelope("0,1,2", worker_pid=os.getpid())
    with pytest.raises(lanes.LaneRefusal) as blind:
        lanes_.assign("t", model_bearing=True, declared_bytes=GiB, measured=MEASURED, pin=(2,))
    assert blind.value.code == "device_pin_infeasible" and "unreadable" in blind.value.detail
    tenant = lanes_.assign("t", model_bearing=True, declared_bytes=GiB, measured=MEASURED, pin=(1,))
    assert tenant.lane_id == "lane-1"
    lanes_.lanes[1].row("t").model_bearing = True

    def refused(pin: tuple[int, ...], **kwargs: Any) -> lanes.LaneRefusal:
        placement_id = kwargs.pop("placement_id", "g")
        revision = lanes_.revision
        with pytest.raises(lanes.LaneRefusal) as caught:
            lanes_.assign(
                placement_id,
                model_bearing=True,
                declared_bytes=kwargs.pop("declared", GiB),
                measured=kwargs.pop("measured", MEASURED),
                pin=pin,
                degrees=kwargs.pop("degrees", (2,)),
            )
        assert lanes_.lane_of(placement_id) is None
        assert lanes_.lane_of("t") is (
            lanes_.lanes[1] if "t" in lanes_.lanes[1].placements else None
        )
        assert lanes_.revision == revision, "a refused pin fused nothing"
        return caught.value

    # 1. envelope and occupancy
    assert refused((0, 3)).code == "device_pin_infeasible"
    occupied = refused((0, 1))
    assert occupied.code == "device_pin_infeasible" and "holds for ['t']" in occupied.detail
    assert refused((1, 0)).code == "device_pin_infeasible"  # sorted unique only
    # 2. the declared degree: the worker never chooses one the author did not declare
    undeclared = refused((0, 2), degrees=())
    assert undeclared.code == "device_group_unsupported"
    assert "declares no sequence-parallel degree" in undeclared.detail
    wrong = refused((0, 2), degrees=(4,))
    assert wrong.code == "device_group_unsupported" and "[4]" in wrong.detail
    # 3. per-device fit: every rank holds the full weights, so EVERY card must hold them
    unreadable = refused((0, 2))
    assert unreadable.code == "device_group_infeasible"
    assert "ordinal 2 ('2') free bytes unreadable" in unreadable.detail
    lanes_.release("t")
    short = refused((0, 1), declared=5 * GiB)
    assert short.code == "device_group_infeasible"
    assert "ordinal 1 ('1') has 4294967296 B measured headroom" in short.detail
    assert "ordinal 0" not in short.detail, "device 0 fits; only the short card is named"
    # a weightless placement shards nothing
    with pytest.raises(lanes.LaneRefusal) as weightless:
        lanes_.assign("w", model_bearing=False, declared_bytes=0, measured={}, pin=(0, 1))
    assert weightless.value.code == "device_group_unsupported"
    # then the fuse: one lane, both devices, one row per device when measured
    group = lanes_.assign(
        "g", model_bearing=True, declared_bytes=GiB, measured=MEASURED, pin=(0, 1), degrees=(2,)
    )
    assert group.lane_id == "lane-0+1" and group.placements == {"g"} and lanes_.revision == 1
    assert lanes_.lane_of("g") is group
    # the same placement asks again: its lane, unchanged
    assert lanes_.assign("g", model_bearing=True, declared_bytes=GiB, measured=MEASURED) is group
    # a later pin over the group's devices refuses; the group leaves and the lanes split
    later = refused((0,), degrees=(), placement_id="h")
    assert later.code == "device_pin_infeasible" and "held by group lane-0+1" in later.detail
    lanes_.release("g")
    assert [lane.lane_id for lane in lanes_.lanes] == ["lane-0", "lane-1", "lane-2"]
    assert lanes_.revision == 2


def test_per_device_rows_and_the_min_ceiling() -> None:
    """A group's ledger holds one `DeviceFacts` row per device and prices against the MIN:
    the lane's reading is the tightest card's, an unreadable card makes the lane
    unreadable, and `authorize` hands every rank the MIN headroom."""
    lanes_ = lanes.LaneSet.from_envelope("0,1,2", worker_pid=os.getpid())
    group = lanes_.fuse((0, 1))
    assert lanes.fold_memory({0: MEASURED[0], 1: MEASURED[1]}) == accel.DeviceMemory(
        "measured", 4 * GiB, 8 * GiB
    )
    assert lanes.fold_memory({0: MEASURED[0], 2: MEASURED[2]}).state == "unreadable"
    assert lanes.fold_memory({}).state == "unreadable"
    rows = group.device_facts(MEASURED)
    assert sorted(rows) == [0, 1] and rows[1].entry == "1" and rows[1].free == 4 * GiB
    ledger = group.row("g").ledger
    ledger.observe_devices(rows)
    assert ledger.device_free == 4 * GiB and ledger.device_total == 8 * GiB
    assert sorted(ledger.rows()["devices"]) == ["0", "1"]
    assert ledger.rows()["devices"]["0"]["free"] == 6 * GiB
    granted = group.authorize("g", lanes.fold_memory({0: MEASURED[0], 1: MEASURED[1]}).free_bytes)
    assert granted == 4 * GiB and ledger.authorized_device_limit_bytes == 4 * GiB
    # the group's chooser knows its degree; a device lane's does not
    assert group.row("g").chooser.degree == 2
    assert lanes_.lanes[1].row("x").chooser.degree == 1
    # a generation change clears the rows with everything else
    ledger.begin_generation(0)
    assert ledger.devices == {} and ledger.device_free == -1
    # the real card: its own reading, folded alone, is itself
    real = accel.device_memory("0", accel.host_backend_family())
    if real.state == "measured":
        assert lanes.fold_memory({0: real}) == real
        one = lanes.LaneSet.from_envelope("0", worker_pid=os.getpid()).lanes[0]
        assert one.memory(accel.host_backend_family()).total_bytes == real.total_bytes
        assert (
            lanes.LaneSet.from_envelope("0,1", worker_pid=os.getpid())
            .fuse((0, 1))
            .memory(accel.host_backend_family())
            .state
            == "unreadable"
        ), "one unreadable card makes the group unreadable"


def _worker_slot(degrees: tuple[int, ...]) -> WorkerSlot:
    return WorkerSlot(
        model_class="M",
        model_binding_path="serve.models.m",
        model_parameter_name="m",
        store="",
        variant="",
        reference_snapshot="",
        components=("unet",),
        snapshots={},
        sequence_parallel_degrees=degrees,
    )


def _binding() -> DeclaredBinding:
    return DeclaredBinding(
        entrypoint_binding_digest="sha256:" + "0" * 64,
        entrypoint="serve",
        model_class="M",
        model_binding_path="serve.models.m",
        model_parameter_name="m",
        release="r",
        model_weight_bytes=GiB,
    )


def _pressed_ledger() -> Ledger:
    ledger = Ledger(worker_pid=os.getpid())
    ledger.begin_generation(1)
    ledger.observe_construction(
        {
            "filled_bytes": GiB,
            "filled": 1,
            "allocator_bytes": GiB,
            "reserved_bytes": GiB,
            "resident": {"unet": GiB},
            "device_free_bytes": 100 << 20,
            "device_total_bytes": 8 * GiB,
        }
    )
    ledger.activations_by_cell["height=512,width=512"] = 512 << 20
    return ledger


def test_the_chooser_refuses_the_ladder_under_a_group() -> None:
    """The same pressed ledger: a single-device chooser DEEPENS to a staged rung (cr-008's
    single-tenant rule); a group's chooser REFUSES `CONSTRAINT_INFEASIBLE`-shaped, priced
    against the MIN device, because every rank runs one plan and no rank adapts."""
    declared = _binding()
    prepared = PreparedRequest.unresolved("serve", {"width": 512, "height": 512})
    model = PreparedModel(delivery_rung="verbatim")
    alone = PlanChooser(_pressed_ledger())
    plan = alone.choose(declared, model, prepared)
    assert plan.placement == "component_staged" and "DEGRADED" in plan.decision_explanation
    grouped = PlanChooser(_pressed_ledger(), degree=2)
    assert grouped.fits(declared, prepared.cell()) == "", "nothing fits; nothing is invented"
    with pytest.raises(PlanRefusal) as refused:
        grouped.choose(declared, model, prepared)
    assert refused.value.code == "group_constraint_infeasible"
    assert "2-device group" in refused.value.detail and "not deepened" in refused.value.detail
    assert refused.value.shortfall == {
        "resource": "vram",
        "scope": "group",
        "needed_bytes": 512 << 20,
        "available_bytes": 100 << 20,
        "evidence_class": "measured",
    }
    assert grouped.walks[-1]["refused_under_group"] == 2


def _placement(placement_id: str, lane_id: str = "") -> Placement:
    return Placement(
        placement_id=placement_id,
        placement_set_digest=b"",
        package={},
        development={},
        environment_digest=b"",
        package_interface=("", 0),
        bindings_digest=b"",
        models=(),
        entrypoints=(),
        environment={},
        document={},
        serving=pb.ServingState.SERVING_STATE_DISPATCHABLE,
        lane_id=lane_id,
    )


def _lane_rows(target: pb.ObservedWorkerState | pb.WorkerSnapshotBody) -> list[dict[str, Any]]:
    return [
        {
            "lane_id": lane.lane_id,
            "device_ordinals": list(lane.device_ordinals),
            "available_attempt_slots": lane.available_attempt_slots,
            "placement_ids": list(lane.placement_ids),
            "resident_placement_ids": list(lane.resident_placement_ids),
        }
        for lane in target.lanes
    ]


def test_a_group_is_one_seat_and_the_wire_says_so() -> None:
    """cr-066's Σ-lanes rule holds verbatim: a 2-device envelope as two lanes advertises
    two seats; fused into one group it advertises ONE. The group's seat is taken by a held
    attempt like any other. The wire (proto-024) carries exactly what the runtime holds:
    `ObservedWorkerState.lanes[]` is the lane documents, the worker-level count is their
    sum, and `PlacementStatus.device_lane_id` names the group."""
    with _workspace() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker", devices="0,1"),
            InMemoryControlHost(),
        )
        try:
            worker.phase = pb.WorkerPhase.WORKER_PHASE_ONLINE
            worker.fence.dispatch_open = True
            worker.fence.wire_minor = lane_wire.LANES_WIRE_MINOR
            worker.accepted.mode = "serving"
            primary = _placement("p0", "lane-0")
            hosted = _placement("p1", "lane-1")
            worker.placement = primary
            worker.hosted["p1"] = HostedPlacement(
                placement=hosted, supervision=worker._new_placement_supervision("p1")
            )
            worker.lanes.lanes[0].placements.add("p0")
            worker.lanes.lanes[1].placements.add("p1")
            assert worker.available_attempt_slots() == 2
            before = worker.observed_state()
            assert [row["lane_id"] for row in worker.lane_documents()] == ["lane-0", "lane-1"]
            assert _lane_rows(before) == worker.lane_documents()
            assert before.available_attempt_slots == sum(
                lane.available_attempt_slots for lane in before.lanes
            )
            assert [s.device_lane_id for s in before.placements] == ["lane-0", "lane-1"]
            # the same envelope as ONE group: p1 leaves, p0 takes both devices
            epoch = worker.admission_epoch
            worker.hosted.pop("p1").supervision.close()
            worker._release_lane("p1")
            worker._release_lane("p0")
            group = worker.lanes.fuse((0, 1))
            group.placements.add("p0")
            primary.lane_id = group.lane_id
            assert worker.available_attempt_slots() == 1
            documents = worker.lane_documents()
            assert documents == [
                {
                    "lane_id": "lane-0+1",
                    "device_ordinals": [0, 1],
                    "available_attempt_slots": 1,
                    "placement_ids": ["p0"],
                    "resident_placement_ids": [],
                }
            ]
            held = AttemptRecord(request_id="r", attempt=1, digest=b"", spec={})
            held.placement_id = "p0"
            held.lane_id = group.lane_id
            held.state = "running"
            worker.engine.history[held.key()] = held
            assert worker.available_attempt_slots() == 0, "one attempt at a time across K"
            taken = worker.observed_state()
            assert _lane_rows(taken) == [dict(documents[0], available_attempt_slots=0)]
            assert taken.available_attempt_slots == 0
            del worker.engine.history[held.key()]
            assert lane_wire.lanes_on_wire(worker.fence.wire_minor)
            observed = worker.observed_state()
            assert _lane_rows(observed) == documents
            assert observed.available_attempt_slots == 1
            (status,) = observed.placements
            assert status.device_lane_id == "lane-0+1"
            assert worker.admission_epoch == epoch, "fusing by hand bumps nothing; assign does"
            # release SPLITS the group and bumps the admission epoch once
            worker._release_lane("p0")
            assert [lane.lane_id for lane in worker.lanes.lanes] == ["lane-0", "lane-1"]
            assert worker.admission_epoch == epoch + 1
        finally:
            for slot in worker.hosted.values():
                slot.supervision.close()
            worker.supervision.close()


def test_a_pin_through_the_worker_fuses_and_bumps_the_admission_epoch() -> None:
    """`_assign_lane` reads the accepted pin and the bindings' declared degrees, fuses on
    success and bumps `admission_epoch` once; a refusal latches the placement typed and
    leaves the lanes exactly as they were."""
    with _workspace() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker", devices="0,1"),
            InMemoryControlHost(),
        )
        try:
            worker.accepted.mode = "serving"
            placement = _placement("g")
            binding = dataclasses.replace(
                _binding(), model_weight_bytes=1 << 20, models=(_worker_slot((2,)),)
            )
            assert binding.sequence_parallel_degrees() == (2,)
            bindings = {binding.entrypoint_binding_digest: binding}
            worker.accepted.device_pins = {"g": (0, 1)}
            epoch = worker.admission_epoch
            # the second card does not exist here: the driver's honest UNREADABLE refuses
            # the group before anything fuses (never authorized zero)
            assert not worker._assign_lane(placement, bindings, worker.supervision)
            fault = worker.engine.faults[-1]
            assert fault.reason == "device_group_infeasible", fault
            assert fault.kind == pb.FaultKind.FAULT_KIND_CONFIG_REFUSED
            assert "ordinal 1 ('1') free bytes unreadable" in fault.detail
            assert placement.lane_id == "" and worker.lanes.revision == 0
            assert worker.admission_epoch == epoch
            # a weightless placement pinned to one device lands on it, no fuse, no bump
            weightless = _placement("w")
            worker.accepted.device_pins = {"w": (1,)}
            light = DeclaredBinding(
                entrypoint_binding_digest="sha256:" + "1" * 64,
                entrypoint="ping",
                model_class="",
                model_binding_path="",
                model_parameter_name="",
                release="r",
            )
            assert worker._assign_lane(
                weightless, {light.entrypoint_binding_digest: light}, worker.supervision
            )
            assert weightless.lane_id == "lane-1" and worker.admission_epoch == epoch
        finally:
            worker.supervision.close()


def test_a_two_lane_snapshot_body_names_each_lane_and_sums_them() -> None:
    """proto-024's runtime-side conformance vector: a two-lane serving worker's
    `WorkerSnapshotBody` carries one `DeviceLane` per lane and a worker-level count that
    is their sum, through the same digest-fenced canonical document the RecordOwner
    acknowledges; a one-lane worker's body is the pre-22 body plus exactly `lanes[]`."""
    with _workspace() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker", devices="0,1"),
            InMemoryControlHost(),
        )
        try:
            worker.phase = pb.WorkerPhase.WORKER_PHASE_ONLINE
            worker.fence.dispatch_open = True
            worker.fence.wire_minor = lane_wire.LANES_WIRE_MINOR
            worker.accepted.mode = "serving"
            primary = _placement("p0", "lane-0")
            hosted = _placement("p1", "lane-1")
            worker.placement = primary
            worker.hosted["p1"] = HostedPlacement(
                placement=hosted, supervision=worker._new_placement_supervision("p1")
            )
            worker.lanes.lanes[0].placements.add("p0")
            worker.lanes.lanes[1].placements.add("p1")
            body = worker.snapshot_body()
            assert _lane_rows(body) == [
                {
                    "lane_id": "lane-0",
                    "device_ordinals": [0],
                    "available_attempt_slots": 1,
                    "placement_ids": ["p0"],
                    "resident_placement_ids": [],
                },
                {
                    "lane_id": "lane-1",
                    "device_ordinals": [1],
                    "available_attempt_slots": 1,
                    "placement_ids": ["p1"],
                    "resident_placement_ids": [],
                },
            ]
            assert body.available_attempt_slots == 2
            canonical_bytes, digest = documents_.identity(body)
            document = documents_.read(canonical_bytes, pb.WorkerSnapshotBody)
            assert document["format"] == "cozy.worker.v1.WorkerSnapshotBody/1"
            assert document["lanes"] == [
                {
                    "available_attempt_slots": 1,
                    "device_ordinals": [0],
                    "lane_id": "lane-0",
                    "placement_ids": ["p0"],
                },
                {
                    "available_attempt_slots": 1,
                    "device_ordinals": [1],
                    "lane_id": "lane-1",
                    "placement_ids": ["p1"],
                },
            ]
            assert document["available_attempt_slots"] == sum(
                row["available_attempt_slots"] for row in document["lanes"]
            )
            assert documents_.identity(body)[1] == digest, "the document is deterministic"
            # admission CLOSED (the barrier not yet passed) is zero on every lane, so the
            # sum stays the worker-level count and no lane shows a seat the worker denies
            worker.fence.dispatch_open = False
            closed = worker.snapshot_body()
            assert closed.admission_state == pb.AdmissionState.ADMISSION_STATE_CLOSED
            assert closed.available_attempt_slots == 0
            assert [lane.available_attempt_slots for lane in closed.lanes] == [0, 0]
        finally:
            for slot in worker.hosted.values():
                slot.supervision.close()
            worker.supervision.close()
    with _workspace() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker", devices="0"),
            InMemoryControlHost(),
        )
        try:
            worker.phase = pb.WorkerPhase.WORKER_PHASE_ONLINE
            worker.fence.dispatch_open = True
            worker.fence.wire_minor = lane_wire.LANES_WIRE_MINOR
            worker.accepted.mode = "serving"
            worker.placement = _placement("p0", "lane-0")
            worker.lanes.lanes[0].placements.add("p0")
            body = worker.snapshot_body()
            assert _lane_rows(body) == [
                {
                    "lane_id": "lane-0",
                    "device_ordinals": [0],
                    "available_attempt_slots": 1,
                    "placement_ids": ["p0"],
                    "resident_placement_ids": [],
                }
            ]
            document = documents_.read(documents_.identity(body)[0], pb.WorkerSnapshotBody)
            older = pb.WorkerSnapshotBody()
            older.CopyFrom(body)
            del older.lanes[:]
            for status in older.placements:
                status.device_lane_id = ""
            before_22 = documents_.read(documents_.identity(older)[0], pb.WorkerSnapshotBody)
            assert {k: v for k, v in document.items() if k not in ("lanes", "placements")} == {
                k: v for k, v in before_22.items() if k not in ("lanes", "placements")
            }
            assert [
                {k: v for k, v in status.items() if k != "device_lane_id"}
                for status in document["placements"]
            ] == before_22["placements"]
            assert document["placements"][0]["device_lane_id"] == "lane-0"
        finally:
            worker.supervision.close()


def test_device_pins_are_read_as_sent() -> None:
    """`DesiredPlacementSet.device_pins` reaches `LaneSet.assign` exactly as the
    RecordOwner sent it; the worker validates, never normalizes, a pin."""
    desired_set = pb.DesiredPlacementSet(
        device_pins=[
            pb.PlacementDevicePin(placement_id="g", device_ordinals=[0, 1]),
            pb.PlacementDevicePin(placement_id="w", device_ordinals=[1]),
            pb.PlacementDevicePin(placement_id="bad", device_ordinals=[1, 0, 1]),
        ]
    )
    assert lane_wire.read_device_pins(desired_set) == {"g": (0, 1), "w": (1,), "bad": (1, 0, 1)}
    assert lane_wire.read_device_pins(pb.DesiredPlacementSet()) == {}
    lane_set = lanes.LaneSet.from_envelope("0,1", worker_pid=os.getpid())
    with pytest.raises(lanes.LaneRefusal) as refused:
        lane_set.assign(
            "bad", model_bearing=True, declared_bytes=1, measured=MEASURED, pin=(1, 0, 1)
        )
    assert refused.value.code == "device_pin_infeasible"
    assert [lane.lane_id for lane in lane_set.lanes] == ["lane-0", "lane-1"]


def test_the_prepare_command_carries_the_degree_only_above_one() -> None:
    """A plain executor's prepare is byte-identical: the degree rides only when > 1."""
    binding = _binding()
    plain = executor_prepare_command(binding, devices="0", authorized_device_limit_bytes=7)
    one = executor_prepare_command(
        binding, devices="0", authorized_device_limit_bytes=7, sequence_parallel_degree=1
    )
    assert canonical.write(plain) == canonical.write(one)
    assert "sequence_parallel_degree" not in plain
    group = executor_prepare_command(
        binding, devices="0,1", authorized_device_limit_bytes=7, sequence_parallel_degree=2
    )
    assert group["sequence_parallel_degree"] == 2 and group["devices"] == "0,1"


# ------------------------------------------------------------------------ the opt-in


def _slot_document(slot: ModelBinding) -> dict[str, object]:
    return package_interface.model_slot(
        slot.path,
        slot.class_key,
        slot.encoded_leaves,
        slot.fusion,
        slot.components,
        slot.sequence_parallel,
    )


def test_the_authors_degrees_reach_the_interface_and_its_fixed_point() -> None:
    """`@sequence_parallel(degrees=...)` on the Model class -> `ModelBinding.sequence_parallel`
    -> the interface slot `sequence_parallel: {degrees}` -> `DeclaredBinding` degrees.
    The decorator refuses malformed degrees at BUILD; the interface's closed scan refuses
    a malformed slot; a class that says nothing writes no key."""
    from cozy_runtime.author import ConformanceError, Model, sequence_parallel
    from cozy_runtime.author._signature import ModelBinding as AuthorSlot

    @sequence_parallel(degrees=(2, 4))
    class Sharded(Model[object]):
        pass

    class Plain(Model[object]):
        pass

    assert Sharded.__sequence_parallel__ == (2, 4) and Plain.__sequence_parallel__ == ()
    for bad in ((), (1,), (4, 2), (2, 2), (True,), (2.0,)):
        with pytest.raises(ConformanceError) as refused:
            sequence_parallel(degrees=bad)  # type: ignore[arg-type]
        assert refused.value.code == "sequence_parallel_degrees"
    with pytest.raises(ConformanceError):
        sequence_parallel(degrees=(2,))(object)  # type: ignore[type-var]
    slot = AuthorSlot(
        path="serve.models.m",
        param="m",
        model_class=Sharded,
        components={},
        sequence_parallel=Sharded.__sequence_parallel__,
    )
    document = _slot_document(slot)
    assert document["sequence_parallel"] == {"degrees": [2, 4]}
    assert "sequence_parallel" not in _slot_document(
        dataclasses.replace(slot, model_class=Plain, sequence_parallel=())
    )
    # the real PackageInterface of the marco-polo example carries no key (nothing declared)
    body = package_interface.build(discover(PACKAGE))
    assert not any(
        "sequence_parallel" in model
        for surface in body.get("entrypoints", [])
        for model in surface.get("models", [])
    )
    # the closed scan over REAL interface bytes: a well-formed slot reads back, a
    # malformed one refuses by path, an unknown key inside the slot refuses
    import copy

    def with_slot(slot: dict[str, Any]) -> dict[str, Any]:
        mutated = copy.deepcopy(body)
        entry = next(row for row in mutated["entrypoints"] if row["name"] == "marco")
        entry["models"] = [
            {
                "path": "marco.models.m",
                "class": "Sharded",
                "component_use": {},
                **slot,
            }
        ]
        return mutated

    for degrees, ok in (([2, 4], True), ([1], False), ([4, 2], False), ([], False), ("2", False)):
        raw = package_interface.canonical_bytes(
            with_slot({"sequence_parallel": {"degrees": degrees}})
        )
        if ok:
            parsed = package_interface.read_bytes(raw)
            assert parsed["entrypoints"][0]["models"][0]["sequence_parallel"] == {
                "degrees": degrees
            }
            continue
        with pytest.raises(package_interface.StalePackageInterface) as malformed:
            package_interface.read_bytes(raw)
        assert "sequence_parallel.degrees" in str(malformed.value), (degrees, malformed.value)
    with pytest.raises(package_interface.StalePackageInterface):
        package_interface.read_bytes(
            package_interface.canonical_bytes(with_slot({"sequence_parallel": {"x": 1}}))
        )
    with pytest.raises(package_interface.StalePackageInterface):
        package_interface.read_bytes(
            package_interface.canonical_bytes(
                with_slot({"sequence_parallel": {"degrees": [2], "x": 1}})
            )
        )
    # the worker-side binding: the degrees every slot declares, intersected
    two = dataclasses.replace(_binding(), models=())
    assert two.sequence_parallel_degrees() == ()
    both = dataclasses.replace(_binding(), models=(_worker_slot((2, 4)), _worker_slot((4, 8))))
    assert both.sequence_parallel_degrees() == (4,)
    assert dataclasses.replace(
        _binding(), models=(_worker_slot((2,)),)
    ).sequence_parallel_degrees() == (2,)


# ------------------------------------------------------------------- the seal (real)


def _weightless_prepare(body: dict[str, Any], interface: Path, **extra: Any) -> dict[str, Any]:
    return {
        "cmd": "prepare",
        "binding": {
            "application": str(body["application"]),
            "package_interface": str(interface),
            "development_project": str(PACKAGE),
            "model_class": "",
        },
        "budgets": {},
        "authorized_device_limit_bytes": 0,
        **extra,
    }


@needs_executor
def test_package_describe_executor_is_replaced_when_its_placement_takes_a_group() -> None:
    with _workspace() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker", devices="0,1"),
            InMemoryControlHost(),
        )
        try:
            single = lanes.DeviceLane("lane-0", (0,), "0", worker_pid=os.getpid())
            described = worker.supervision.spawn(imposed=worker.imposed(single))
            assert not described.ready_bindings and described.prepared_model is None
            assert described.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "0"
            group = worker.lanes.fuse((0, 1))
            tenant = Tenant(
                _placement("candidate"), worker.supervision, {}, {}, {}, group, "candidate"
            )
            prepared = worker._own_executor(tenant, [])
            assert not isinstance(prepared, str), prepared
            assert prepared is not described, (
                "the package-describe process retained its single-GPU seal"
            )
            assert not described.alive()
            assert prepared.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "0,1"
            assert worker.supervision.devices == group.entries
            assert worker._own_executor(tenant, []) is prepared, (
                "unchanged seals should still reuse"
            )
        finally:
            worker.supervision.close()


@needs_executor
def test_an_executor_sealed_to_two_ordinals_and_its_reclaim_over_both() -> None:
    """THE K-DEVICE SEAL, read from the child's own `/proc` environment: a group lane's
    executor carries `CUDA_VISIBLE_DEVICES=0,1` and `NCCL_NVLS_ENABLE=0`; a device lane's
    carries no NVLS row. The degree is checked against the seal before any weight moves;
    a weightless construction refuses to shard; degree 1 is a plain prepare. Retirement
    proves absence for EVERY device of the lane: two rows, one per entry, beside the merged
    proof — and the card that does not exist is UNREADABLE, never absent by omission."""
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            config,
            WorkerOptions(root=root / "worker", devices="0,1"),
            InMemoryControlHost(),
        )
        found = discover(PACKAGE)
        body = package_interface.build(found)
        interface = root / "metadata" / "package-interface.json"
        interface.parent.mkdir()
        interface.write_bytes(package_interface.canonical_bytes(body))
        group = worker.lanes.fuse((0, 1))
        assert worker.imposed(group)["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert worker.imposed(group)["NCCL_NVLS_ENABLE"] == "0"
        # the wide envelope lane a job takes IS a group lane (K = the envelope), so its
        # seal carries the row too; a ONE-device lane's seal is exactly what it was
        assert worker.imposed(worker.lanes.envelope)["NCCL_NVLS_ENABLE"] == "0"
        single = lanes.DeviceLane("lane-0", (0,), "0", worker_pid=os.getpid())
        assert "NCCL_NVLS_ENABLE" not in worker.imposed(single)
        assert worker.imposed(single)["CUDA_VISIBLE_DEVICES"] == "0"
        executor = worker.supervision.spawn(imposed=worker.imposed(group))
        try:
            environ = Path(f"/proc/{executor.pid}/environ").read_bytes().split(b"\0")
            rows = dict(row.split(b"=", 1) for row in environ if b"=" in row)
            assert rows[b"CUDA_VISIBLE_DEVICES"] == b"0,1", rows
            assert rows[b"NCCL_NVLS_ENABLE"] == b"0", rows
            assert executor.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "0,1"
            assert executor.hello["rank"] == 0 and executor.hello["world"] == 1
            assert worker.supervision.devices == ("0", "1")
            command = _weightless_prepare(body, interface, devices="0,1")
            # the seal fences the devices before the degree is even read
            other = executor.call({**command, "devices": "0"}, timeout=60.0)
            assert other["code"] == "env_seal_broken", other
        finally:
            evidence = worker.supervision.retire_current(executor, "seal arm done")
        assert evidence is not None
        assert evidence.members == (executor.pid,)
        assert sorted(evidence.devices) == ["0", "1"], evidence
        assert evidence.devices["0"].state == "absent", evidence
        assert evidence.devices["1"].state == "unreadable", (
            "the second card does not exist here: the driver's honest answer, never absent"
        )
        # degree checks, each on a fresh executor (a refused prepare poisons its epoch)
        for extra, code in (
            ({"sequence_parallel_degree": 3}, "sequence_parallel_degree_mismatch"),
            ({"sequence_parallel_degree": 2}, "sequence_parallel_unsupported"),
        ):
            executor = worker.supervision.spawn(imposed=worker.imposed(group))
            try:
                reply = executor.call(
                    _weightless_prepare(body, interface, devices="0,1", **extra), timeout=60.0
                )
                assert reply["code"] == code, reply
                if code == "sequence_parallel_degree_mismatch":
                    assert "sealed to 2 device(s) ('0,1')" in reply["detail"], reply
            finally:
                worker.supervision.retire_current(executor, "degree arm done")
        worker.supervision.close()


@needs_executor
def test_degree_one_is_a_plain_executor() -> None:
    """A degenerate group on one card is byte-identical in behaviour to a plain executor:
    the same weightless prepare with `sequence_parallel_degree: 1` and without it answers
    the same reply (timing aside), and neither spawns a rank."""
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            config,
            WorkerOptions(root=root / "worker", devices="0"),
            InMemoryControlHost(),
        )
        found = discover(PACKAGE)
        body = package_interface.build(found)
        interface = root / "metadata" / "package-interface.json"
        interface.parent.mkdir()
        interface.write_bytes(package_interface.canonical_bytes(body))
        replies: list[dict[str, Any]] = []
        for extra in ({}, {"sequence_parallel_degree": 1}):
            executor = worker.supervision.spawn(imposed=worker.imposed(worker.lanes.lanes[0]))
            try:
                reply = executor.call(
                    _weightless_prepare(body, interface, devices="0", **extra), timeout=120.0
                )
                assert reply["ok"], reply
                children = subprocess.run(
                    ["pgrep", "-P", str(executor.pid)], capture_output=True, text=True
                ).stdout.split()
                assert children == [], f"a degree-1 executor spawned {children}"
                replies.append(reply)
            finally:
                worker.supervision.retire_current(executor, "degree-1 arm done")
        worker.supervision.close()

    def stable(reply: dict[str, Any]) -> dict[str, Any]:
        facts = {k: v for k, v in reply.get("facts", {}).items() if not k.endswith("_ms")}
        facts.pop("rss_bytes", None)
        return {**{k: v for k, v in reply.items() if k != "facts"}, "facts": facts}

    assert stable(replies[0]) == stable(replies[1])
    assert "sequence_parallel" not in replies[1].get("facts", {})


# ---------------------------------------------------------------------- gloo (torch)


def _launcher(env: dict[str, str]) -> Any:
    """The real launch shape under a chosen environment: the trampoline's `inherit`
    backend (parent check, pdeathsig, no_new_privs, the executor's OOM score) exec'ing
    the executor program with its rank, world, seam fd and leader."""

    def launch(module_argv: list[str], fd: int) -> spawn.Child:
        return spawn.spawn_follower(
            python=sys.executable, module_argv=module_argv, env=env, inherit_fd=fd
        )

    return launch


@needs_torch
def test_a_gloo_group_forms_on_a_kernel_chosen_port_and_agrees_or_refuses(
    tmp_path: Path,
) -> None:
    """THIS process is rank 0. Real follower executor processes are spawned by the real
    launch shape (the executor program, told its rank, world, seam fd and leader), dial
    back over their `socketpair` seam under exactly rank 0's seal, join a TCPStore on a
    port the kernel chose, pass the formation barrier, and refuse `rank_divergence` on a
    broadcast plan whose hardware half is not theirs. gloo stands in for NCCL: the
    rendezvous, the barrier, the plan and the seam are backend-agnostic."""
    import torch

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2"}
    sealed = seal_snapshot(env)
    group = RankGroup(
        degree=3,
        backend="gloo",
        python=sys.executable,
        root=str(tmp_path),
        sample_seconds=0.2,
        launch=_launcher(env),
    )
    try:
        group.spawn()
        assert [f.rank for f in group.followers] == [1, 2]
        assert all(f.process.poll() is None for f in group.followers)
        # the followers are DIRECT CHILDREN of rank 0 (they inherit its cgroup and die
        # with it): pdeathsig was armed before exec
        for follower in group.followers:
            stat = Path(f"/proc/{follower.pid}/stat").read_text().rsplit(") ", 1)[1].split()
            assert int(stat[1]) == os.getpid(), "a follower's parent is rank 0"
        group.dial_back(sealed)
        assert [f.hello["rank"] for f in group.followers] == [1, 2]
        assert all(f.hello["world"] == 3 for f in group.followers)
        assert all(f.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "0,1,2" for f in group.followers)
        assert group.port == 0, "no store exists before formation"
        group.form(torch, {"NCCL_NVLS_ENABLE": "0"})
        assert group.port > 1024, group.port
        assert group.formed
        assert torch.distributed.get_world_size() == 3 and torch.distributed.get_rank() == 0
        # the followers answered `join` only after their own barrier: the group is whole
        command = {
            "cmd": "prepare",
            "devices": "0,1,2",
            "sequence_parallel_degree": 3,
            "authorized_device_limit_bytes": 1,
            "binding": {
                "application": "x:app",
                "package_interface": "/nowhere",
                "development_project": "",
                "model_class": "",
            },
            "budgets": {},
        }
        plan = GroupPlan(
            degree=3,
            devices=("0", "1", "2"),
            plan_digest="sha256:" + "a" * 64,
            authorized_device_limit_bytes=1,
        )
        assert GroupPlan.read(plan.document()) == plan
        # the hardware half disagrees: refused BEFORE torch is imported on the follower
        divergent = dataclasses.replace(plan, devices=("0", "1", "7"))
        replies = group.call({**command, "group": {"plan": divergent.document()}})
        assert [r["code"] for r in replies] == ["rank_divergence"] * 2, replies
        assert "rank 0 broadcast degree 3 over ['0', '1', '7']" in replies[0]["detail"]
        assert replies[0]["rank"] == 1 and replies[1]["rank"] == 2
        # a follower that refused is poisoned: a second command says so
        replies = group.call({"cmd": "probe"})
        assert all(r["poisoned"] == "rank divergence" for r in replies), replies
        # an out-of-vocabulary command on a follower refuses; rank 0's plane is closed
        replies = group.call({"cmd": "invoke"})
        assert [r["code"] for r in replies] == ["follower_command_unsupported"] * 2
    finally:
        group.close()
    assert all(f.process.poll() is not None for f in group.followers)
    assert not torch.distributed.is_initialized()


@needs_torch
def test_the_plan_agrees_field_by_field_or_refuses() -> None:
    """A follower ASSERTS the broadcast plan; the first field that differs is named and
    the rank never adapts."""
    plan = GroupPlan(2, ("0", "1"), "sha256:a", 1024)
    plan.assert_agrees(plan, rank=1)
    for field_name, other in (
        ("plan_digest", dataclasses.replace(plan, plan_digest="sha256:b")),
        (
            "authorized_device_limit_bytes",
            dataclasses.replace(plan, authorized_device_limit_bytes=1),
        ),
        ("degree", dataclasses.replace(plan, degree=4)),
    ):
        with pytest.raises(RankDivergence) as refused:
            plan.assert_agrees(other, rank=1)
        assert refused.value.code == "rank_divergence"
        assert refused.value.field_name == field_name and refused.value.rank == 1
        assert "never adapts" in refused.value.message or "no rank adapts" in refused.value.message


@needs_torch
def test_a_follower_death_breaks_the_group_loudly(tmp_path: Path) -> None:
    """A follower killed -9 is observed through its seam the moment it dies; the group
    latches `group_broken` naming the rank and its exit status, every later command
    refuses, and teardown reaps what is left. Never a park on a collective."""
    import torch

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1"}
    group = RankGroup(
        degree=2,
        backend="gloo",
        python=sys.executable,
        root=str(tmp_path),
        sample_seconds=0.2,
        launch=_launcher(env),
    )
    try:
        group.spawn()
        group.dial_back(seal_snapshot(env))
        group.form(torch, {"NCCL_NVLS_ENABLE": "0"})
        victim = group.followers[0]
        victim.process.send_signal(signal.SIGKILL)
        victim.process.wait()
        with pytest.raises(GroupRefusal) as refused:
            group.call({"cmd": "probe"})
        assert refused.value.code == "group_broken"
        assert "rank 1" in refused.value.message and "-9" in refused.value.message, refused.value
        assert group.broken and "rank 1" in group.broken
        with pytest.raises(GroupRefusal):
            group.check_alive()
    finally:
        group.close()


@needs_torch
def test_a_follower_that_never_dials_back_is_judged_by_its_own_meter(tmp_path: Path) -> None:
    """No clock: a follower that never answers `hello` is condemned only once its own
    progress meter has been flat for longer than the floor (`liveness.Pace`), and the
    refusal names the measurement."""
    env = {**os.environ}

    def sleeper(module_argv: list[str], fd: int) -> spawn.Child:
        del module_argv
        return spawn.spawn_follower(
            python=sys.executable,
            module_argv=["-c", "import time; time.sleep(600)"],
            env=env,
            inherit_fd=fd,
        )

    group = RankGroup(
        degree=2,
        backend="gloo",
        python=sys.executable,
        root=str(tmp_path),
        sample_seconds=0.2,
        launch=sleeper,
    )
    try:
        group.spawn()
        with pytest.raises(GroupRefusal) as refused:
            group.dial_back(seal_snapshot(env))
        assert refused.value.code == "group_broken"
        assert "wedged during 'hello'" in refused.value.message
        assert "no measurable progress for" in refused.value.message, refused.value.message
    finally:
        group.close()


# ------------------------------------------------------------------ the wire and gate


@needs_torch
def test_the_rank_wire_is_closed(tmp_path: Path) -> None:
    """Scalars, lists, tuples, str-keyed dicts, tensors (through the spool) and generators
    cross; generators and callbacks stay on the leader."""
    import torch

    spool = wire.TensorSpool(tmp_path / "ranks")
    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    command = wire.run_command((tensor, 1.5, "x", None, (1, 2)), {"steps": [1, 2]}, spool)
    assert command["cmd"] == "run"
    assert (tmp_path / "ranks" / "tensor-0.raw").stat().st_size == 12
    assert command["args"][0] == {
        "__sp__": "tensor",
        "v": "tensor-0.raw",
        "dtype": "bfloat16",
        "shape": [2, 3],
    }
    args, kwargs = wire.run_call(command, spool, device=torch.device("cpu"))
    assert torch.equal(args[0], tensor) and args[1:] == (1.5, "x", None, (1, 2))
    assert kwargs["steps"] == [1, 2]
    empty = wire.marshal(torch.empty(0, 3), spool)
    assert torch.equal(wire.unmarshal(empty, spool, device="cpu"), torch.empty(0, 3))
    for bad in (lambda: None, {1: 2}, {"__sp__": 1}, object(), torch.Generator()):
        with pytest.raises(wire.UncrossableArgument) as refused:
            wire.marshal(bad, spool)
        assert refused.value.code == "uncrossable_argument"
    with pytest.raises(wire.UncrossableArgument):
        wire.unmarshal(
            {"__sp__": "tensor", "v": "../etc/passwd", "dtype": "float32", "shape": []},
            spool,
            device="cpu",
        )
    spool.clear()
    assert not any((tmp_path / "ranks").iterdir())


@needs_torch
def test_a_sharded_forward_outside_the_gate_refuses() -> None:
    import torch

    component = torch.nn.Identity()
    cp._install_gate_guard(component, "dit")
    with pytest.raises(cp.UngatedShardedForward):
        component(torch.ones(1))
    with cp.gated_call():
        torch.testing.assert_close(component(torch.ones(1)), torch.ones(1))
    assert not cp.in_gated_call()
    cp.refuse_unless_heads_divisible(heads=16, degree=4)
    cp.refuse_unless_heads_divisible(heads=0, degree=1)
    with pytest.raises(cp.ContextParallelUnavailable) as heads:
        cp.refuse_unless_heads_divisible(heads=6, degree=4)
    assert (
        heads.value.code == "context_parallel_unavailable" and "head count 6" in heads.value.message
    )
    # H3's 56 heads divide every declared degree; its packed sequence divides none of them,
    # and `ulysses_anything` is why that is no longer a refusal.
    cp.refuse_unless_heads_divisible(heads=56, degree=8)
    with pytest.raises(cp.ContextParallelUnavailable) as ring:
        cp._GroupMesh(None, 2, 0)["ring"].get_group()
    assert "Ulysses" in ring.value.message
    with pytest.raises(cp.ContextParallelUnavailable) as unsharded:
        cp.install_context_parallel(object(), degree=2, comms=None)
    assert "no declared component carries a `_cp_plan`" in unsharded.value.message
    assert cp.install_context_parallel(object(), degree=1, comms=None) == ()


@needs_executor
def test_a_seal_width_change_is_a_replacement_not_a_reuse() -> None:
    """THE DEFECT: a live executor is sealed ONCE, at spawn, and cannot be re-sealed.

    A placement that starts on a one-device lane and is then assigned the K-device group
    lane (cr-068) used to REUSE the running process: `_own_executor` asked whether it was
    alive, unpoisoned, unreserved, unbound and model-free, and never whether it was sealed
    to the devices it was about to serve. The prepare then reached `executor.py` and was
    refused `env_seal_broken` -- correctly, since an executor serves exactly its lane's
    devices -- but only AFTER the group had formed and 97.5 GiB of weights had been
    fetched. Measured on a 2xH200 rental, 2026-09-08: `this executor is sealed to
    CUDA_VISIBLE_DEVICES='0' and the prepare names '0,1'`.

    `Executor.sealed_devices` records what the process was actually sealed with, so both
    reuse paths can ask the question cr-066 answers: does this process serve THIS lane?
    """
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            config,
            WorkerOptions(root=root / "worker", devices="0,1"),
            InMemoryControlHost(),
        )
        group = worker.lanes.fuse((0, 1))
        single = lanes.DeviceLane("lane-0", (0,), "0", worker_pid=os.getpid())
        executor = worker.supervision.spawn(imposed=worker.imposed(single))
        try:
            # what the child really got, and what we recorded, are the same string
            assert executor.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "0"
            assert executor.sealed_devices == "0"
            # it MATCHES the lane it was spawned for, so an ordinary reuse still reuses
            assert executor.sealed_devices == single.devices
            # and it does NOT match the group lane, which is the whole fix: the reuse
            # branch in `_own_executor` now falls through to `replace` instead of handing
            # a one-card process to a two-card prepare
            assert executor.sealed_devices != group.devices
            assert group.devices == "0,1"
            # the width change also adds the NVLS row a group communicator needs
            # (pgw#929), which a reused one-device process would never have carried
            assert "NCCL_NVLS_ENABLE" not in worker.imposed(single)
            assert worker.imposed(group)["NCCL_NVLS_ENABLE"] == "0"
        finally:
            worker.supervision.close()


@needs_executor
def test_a_group_sealed_executor_records_both_ordinals() -> None:
    """The same record, taken from the other side: spawned FOR the group, it says so."""
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            config,
            WorkerOptions(root=root / "worker", devices="0,1"),
            InMemoryControlHost(),
        )
        group = worker.lanes.fuse((0, 1))
        executor = worker.supervision.spawn(imposed=worker.imposed(group))
        try:
            assert executor.sealed_devices == "0,1" == group.devices
            assert executor.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "0,1"
            # and it is not reusable for a one-card lane either -- the rule is symmetric
            single = lanes.DeviceLane("lane-0", (0,), "0", worker_pid=os.getpid())
            assert executor.sealed_devices != single.devices
        finally:
            worker.supervision.close()
