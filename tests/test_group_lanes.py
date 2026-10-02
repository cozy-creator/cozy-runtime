"""Group placement bookkeeping and sealed executor lifecycle regressions.

Actual sharded component forwards and native residency are exercised separately
in test_parallel_calls.py. These tests do not establish multi-GPU H3 performance.

The gloo cases open the NVIDIA device nodes on a host that has a driver, even with
`CUDA_VISIBLE_DEVICES=""`: `torch.distributed` asks torch for its accelerator, and a CUDA
build of torch answers by counting devices. Beside a GPU benchmark, run this file only while
holding the GPU lock, or where `/dev/nvidia*` is hidden.
"""

from __future__ import annotations

import dataclasses
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest

from cozy_runtime.author._signature import ModelBinding
from cozy_runtime.internal import accel, canonical, executor_commands, package_interface, spawn
from cozy_runtime.internal.config import seal_snapshot
from cozy_runtime.internal.discovery import discover, discover_installed
from cozy_runtime.internal.executor_commands import Start
from cozy_runtime.internal.parallel import cp, wire
from cozy_runtime.internal.parallel.group import RankGroup
from cozy_runtime.internal.parallel.plan import GpuDivergence, GroupPlan, GroupRefusal
from cozy_runtime.internal.worker import lane_wire, lanes
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    PlanChooser,
    PreparedModel,
    PreparedRequest,
)
from cozy_runtime.internal.worker.plan import ModelBinding as WorkerSlot
from cozy_runtime.internal.worker.session import (
    Placement,
    Tenant,
    Worker,
    WorkerOptions,
    executor_load_command,
)
from cozy_runtime.protocol import worker_pb2 as pb
from test_device_lanes import PACKAGE, WEIGHTLESS, _config, _workspace
from test_end_to_end import NO_EXECUTOR
from test_gpu_scheduler import driverless
from test_stage_scheduler import Shell

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


def test_a_group_lane_overlaps_its_device_lanes() -> None:
    """A group lane over K devices joins the device lanes instead of replacing them: its
    seal names all K entries and its id joins the ordinals. Which lane's call runs on a
    device when is the stage scheduler's turn (`test_stage_scheduler`)."""
    lanes_ = lanes.LaneSet.from_envelope("0,1,2", worker_pid=os.getpid())
    group = lanes_.group((0, 1))
    assert group.lane_id == "lane-0+1" and group.ordinals == (0, 1) and group.devices == "0,1"
    assert group.group and group.degree == 2 and group.entries == ("0", "1")
    assert lanes_.group((0, 1)) is group
    assert [lane.lane_id for lane in lanes_.lanes] == ["lane-0", "lane-0+1", "lane-1", "lane-2"]
    assert lanes_.envelope.ordinals == (0, 1, 2), "the job's envelope lane is untouched"
    assert [lane.lane_id for lane in lanes_.sharing(lanes_.by_id["lane-1"])] == [
        "lane-1",
        "lane-0+1",
    ]
    group.placements.add("p")
    lanes_.release("q")
    assert lanes_.by_id["lane-0+1"] is group
    lanes_.release("p")
    assert "lane-0+1" not in lanes_.by_id, "the group left with its last placement"
    assert [lane.lane_id for lane in lanes_.lanes] == ["lane-0", "lane-1", "lane-2"]


def test_a_group_binds_exactly_or_refuses_typed() -> None:
    lanes_ = lanes.LaneSet.from_envelope("0,1,2", worker_pid=os.getpid())
    with pytest.raises(lanes.LaneRefusal) as blind:
        lanes_.bind("t", (2,), model_bearing=True, measured=MEASURED)
    assert blind.value.code == "device_pin_infeasible" and "unreadable" in blind.value.detail
    tenant = lanes_.bind("t", (1,), model_bearing=True, measured=MEASURED)
    assert tenant.lane_id == "lane-1"

    def refused(ordinals: tuple[int, ...], placement_id: str = "g") -> lanes.LaneRefusal:
        with pytest.raises(lanes.LaneRefusal) as caught:
            lanes_.bind(placement_id, ordinals, model_bearing=True, measured=MEASURED)
        assert lanes_.lane_of(placement_id) is None
        assert len(lanes_.lanes) == 3, "a refused bind made no lane"
        return caught.value

    assert refused((0, 3)).code == "device_pin_infeasible"  # outside the envelope
    assert refused((1, 0)).code == "device_pin_infeasible"  # sorted unique only
    unreadable = refused((0, 2))
    assert unreadable.code == "device_group_infeasible"
    assert "ordinal 2 ('2') free bytes unreadable" in unreadable.detail
    with pytest.raises(lanes.LaneRefusal) as weightless:
        lanes_.bind("w", (0, 1), model_bearing=False, measured={})
    assert weightless.value.code == "device_group_unsupported"
    # a tenant on lane-1 does not keep a group off device 1, nor the group a singleton
    group = lanes_.bind("g", (0, 1), model_bearing=True, measured=MEASURED)
    assert group.lane_id == "lane-0+1" and group.placements == {"g"}
    assert lanes_.bind("g", (0, 1), model_bearing=True, measured=MEASURED) is group
    assert lanes_.bind("h", (0,), model_bearing=True, measured=MEASURED).lane_id == "lane-0"
    assert lanes_.lane_of("t") is tenant
    lanes_.release("g")
    assert [lane.lane_id for lane in lanes_.lanes] == ["lane-0", "lane-1", "lane-2"]


def _place(
    envelope: str, pin: tuple[int, ...] | None, degrees: tuple[int, ...]
) -> tuple[lanes.LaneSet, lanes.DeviceLane]:
    """An owner-placed placement, as `_assign_lane` places one: its widest declared degree
    within its pin, on the set the stage scheduler would place a call of it."""
    lanes_ = lanes.LaneSet.from_envelope(envelope, worker_pid=os.getpid())
    measured = {
        ordinal: accel.DeviceMemory("measured", 6 * GiB, 8 * GiB)
        for ordinal in range(len(lanes_.entries))
    }
    candidates = pin or tuple(range(len(lanes_.entries)))
    width = max(set(range(1, len(candidates) + 1)) & {1, *degrees})
    ordinals = Shell(len(lanes_.entries)).scheduler.choose(candidates, width, ("inst", "b"))
    return lanes_, lanes_.bind("p", ordinals, model_bearing=True, measured=measured)


def test_an_owner_pin_runs_at_the_largest_declared_degree_it_forms() -> None:
    """A K-device pin runs at the largest declared degree <= K, the declared truth; the
    devices it leaves out stay free device lanes."""
    lanes_, lane = _place("0,1,2", (0, 1, 2), (2, 4))
    assert lane.ordinals == (0, 1) and lane.placements == {"p"}
    assert [other.lane_id for other in lanes_.lanes] == ["lane-0", "lane-0+1", "lane-1", "lane-2"]
    other = lanes_.bind(
        "q",
        (2,),
        model_bearing=True,
        measured={2: accel.DeviceMemory("measured", 6 * GiB, 8 * GiB)},
    )
    assert other.lane_id == "lane-2", "the left-out device serves another placement"
    lanes_.release("p")
    assert [lane.lane_id for lane in lanes_.lanes] == ["lane-0", "lane-1", "lane-2"]
    lanes_, lane = _place("0,1,2", (0, 1, 2), ())
    assert lane.ordinals == (0,) and not lane.group and len(lanes_.lanes) == 3
    assert _place("0,1,2,3", (0, 1, 2, 3), (2,))[1].ordinals == (0, 1)
    assert _place("0,1", (0, 1), (2,))[1].ordinals == (0, 1)
    assert _place("0,1,2", None, (2, 4))[1].ordinals == (0, 1)


def test_per_device_rows_and_the_min_ceiling() -> None:
    """A group's ledger holds one `DeviceFacts` row per device and prices against the MIN:
    the lane's reading is the tightest card's, an unreadable card makes the lane
    unreadable, and the process ceiling every rank is handed is the MIN capacity."""
    lanes_ = lanes.LaneSet.from_envelope("0,1,2", worker_pid=os.getpid())
    group = lanes_.group((0, 1))
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
    assert lanes.fold_memory({0: MEASURED[0], 1: MEASURED[1]}).total_bytes == 8 * GiB
    # The chooser consumes the already-folded ledger; it needs no second copy of degree.
    assert group.row("g").chooser.ledger is ledger
    # a generation change clears the rows with everything else
    ledger.begin_generation(0)
    assert ledger.devices == {} and ledger.device_free == -1


@pytest.mark.real_gpu
def test_a_real_cards_reading_folds_to_itself() -> None:
    """Card 0's own reading, folded alone, is itself; beside a card no host has, unreadable."""
    real = accel.device_memory("0", accel.host_backend_family())
    if real.state != "measured":
        pytest.skip("no driver-readable device 0 on this host")
    assert lanes.fold_memory({0: real}) == real
    one = lanes.LaneSet.from_envelope("0", worker_pid=os.getpid()).lanes[0]
    assert one.memory(accel.host_backend_family()).total_bytes == real.total_bytes
    assert (
        lanes.LaneSet.from_envelope("0,64", worker_pid=os.getpid())
        .group((0, 1))
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
        logical_weight_bytes=GiB,
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


@pytest.mark.parametrize("measured", [False, True])
def test_a_call_whose_need_does_not_fit_is_staged_with_its_measured_headroom(
    measured: bool,
) -> None:
    """What the memory manager could not make room for runs one declared scope at a time."""
    ledger = _pressed_ledger()
    if not measured:
        ledger.activations_by_cell.clear()
    prepared = PreparedRequest.unresolved("serve", {"width": 512, "height": 512})
    chooser = PlanChooser(ledger)
    model = PreparedModel(delivery_rung="verbatim")
    plan = chooser.choose(_binding(), model, prepared, fits=False)
    assert plan.placement == "component_staged"
    assert plan.headroom_bytes == (512 << 20 if measured else 0)
    assert chooser.choose(_binding(), model, prepared).placement == "all_resident"


def _placement(placement_id: str, lane_id: str = "") -> Placement:
    return Placement(
        pb.Placement(placement_id=placement_id),
        b"",
        serving=pb.ServingState.SERVING_STATE_DISPATCHABLE,
        lane_id=lane_id,
    )


def _lane_rows(target: pb.ObservedWorkerState | pb.WorkerSnapshotBody) -> list[dict[str, Any]]:
    return [
        {
            "lane_id": lane.lane_id,
            "device_ordinals": list(lane.device_ordinals),
            "placement_ids": list(lane.placement_ids),
            "resident_placement_ids": list(lane.resident_placement_ids),
        }
        for lane in target.lanes
    ]


def test_a_pin_through_the_worker_binds_a_group_without_moving_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_assign_lane` reads the accepted pin and the bindings' declared degrees; a refusal
    latches the placement typed and leaves the lanes exactly as they were."""
    driverless(monkeypatch)
    with _workspace() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker", devices="64,65"),
            InMemoryControlHost(),
        )
        try:
            worker.accepted.mode = "serving"
            placement = _placement("g")
            binding = dataclasses.replace(
                _binding(), logical_weight_bytes=1 << 20, models=(_worker_slot((2,)),)
            )
            assert binding.sequence_parallel_degrees() == (2,)
            bindings = {binding.entrypoint_binding_digest: binding}
            worker.accepted.device_pins = {"g": (0, 1)}
            placement.device_pin = (0, 1)
            epoch = worker.admission_epoch
            # the second card does not exist here: the driver's honest UNREADABLE refuses
            # the group before any lane exists (never authorized zero)
            assert not worker._assign_lane(placement, bindings, worker.supervision)
            fault = worker.engine.faults[-1]
            assert fault.reason == "device_group_infeasible", fault
            assert fault.kind == pb.FaultKind.FAULT_KIND_CONFIG_REFUSED
            assert "ordinal 1 ('65') free bytes unreadable" in fault.detail
            assert placement.lane_id == "" and len(worker.lanes.lanes) == 2
            assert worker.admission_epoch == epoch
            # a weightless placement pinned to one device lands on it, no bump
            weightless = _placement("w")
            worker.accepted.device_pins = {"w": (1,)}
            weightless.device_pin = (1,)
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


def test_device_pins_are_read_as_sent() -> None:
    """`DesiredPlacementSet.device_pins` reaches `LaneSet.bind` exactly as the
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
        lane_set.bind("bad", (1, 0, 1), model_bearing=True, measured=MEASURED)
    assert refused.value.code == "device_pin_infeasible"
    assert [lane.lane_id for lane in lane_set.lanes] == ["lane-0", "lane-1"]


def test_the_load_command_carries_the_degree_only_above_one() -> None:
    """A plain executor's load is byte-identical: the degree rides only when > 1."""
    binding = _binding()
    key = binding.construction_key()
    plain = executor_load_command(
        binding, construction=key, devices="0", authorized_device_limit_bytes=7
    )
    one = executor_load_command(
        binding,
        construction=key,
        devices="0",
        authorized_device_limit_bytes=7,
        sequence_parallel_degree=1,
    )
    wire = executor_commands.encode(plain)
    assert canonical.write(wire) == canonical.write(executor_commands.encode(one))
    assert "sequence_parallel_degree" not in wire
    assert wire["cmd"] == "load" and plain.construction == key
    group = executor_load_command(
        binding,
        construction=key,
        devices="0,1",
        authorized_device_limit_bytes=7,
        sequence_parallel_degree=2,
    )
    assert group.sequence_parallel_degree == 2 and group.devices == "0,1"


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
    # malformed one refuses by path, an unknown key inside the slot is ignored
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
    tolerated = package_interface.read_bytes(
        package_interface.canonical_bytes(
            with_slot({"sequence_parallel": {"degrees": [2], "x": 1}})
        )
    )
    assert tolerated["entrypoints"][0]["models"][0]["sequence_parallel"]["degrees"] == [2]
    # the worker-side binding: the degrees every slot declares, intersected
    two = dataclasses.replace(_binding(), models=())
    assert two.sequence_parallel_degrees() == ()
    both = dataclasses.replace(_binding(), models=(_worker_slot((2, 4)), _worker_slot((4, 8))))
    assert both.sequence_parallel_degrees() == (4,)
    assert dataclasses.replace(
        _binding(), models=(_worker_slot((2,)),)
    ).sequence_parallel_degrees() == (2,)


# ------------------------------------------------------------------- the seal (real)


def _weightless_start(body: dict[str, Any], interface: Path, **extra: Any) -> Start:
    return msgspec.convert(
        {
            "application": str(body["application"]),
            "package_interface": str(interface),
            **extra,
        },
        Start,
    )


@needs_executor
def test_package_describe_executor_is_replaced_when_its_placement_takes_a_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driverless(monkeypatch)
    with _workspace() as root:
        worker = Worker(
            _config(root / "home"),
            WorkerOptions(root=root / "worker", devices="64,65"),
            InMemoryControlHost(),
        )
        try:
            single = lanes.DeviceLane("lane-0", (0,), "64", worker_pid=os.getpid())
            described = worker.supervision.spawn(imposed=worker.imposed(single))
            assert not described.ready_bindings and not described.loaded
            assert described.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "64"
            group = worker.lanes.group((0, 1))
            tenant = Tenant(_placement("candidate"), worker.supervision, {}, {}, group, "candidate")
            prepared = worker._own_executor(tenant)
            assert not isinstance(prepared, str), prepared
            assert prepared is not described, (
                "the package-describe process retained its single-GPU seal"
            )
            assert not described.alive()
            assert prepared.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "64,65"
            assert worker.supervision.devices == group.entries
            assert worker._own_executor(tenant) is prepared, "unchanged seals should still reuse"
        finally:
            worker.supervision.close()


@needs_executor
@pytest.mark.parametrize("readable", [True, False], ids=["one-readable-device", "cpu-only"])
def test_an_executor_sealed_to_two_ordinals_and_its_reclaim_over_both(
    monkeypatch: pytest.MonkeyPatch, readable: bool
) -> None:
    """THE K-DEVICE SEAL, read from the child's own `/proc` environment: a group lane's
    executor carries `CUDA_VISIBLE_DEVICES=64,65` and `NCCL_NVLS_ENABLE=0`; a device lane's
    carries no NVLS row. The degree is checked against the seal before any weight moves;
    a weightless package refuses to shard. Retirement proves the actual process is gone
    and retains a driver observation for each sealed device. Explicit driver fixtures
    cover a readable device and a CPU-only host without probing physical GPUs."""
    driverless(monkeypatch)
    expected = {
        "64": accel.ProcessMemory("absent", 0) if readable else accel.ProcessMemory("unreadable"),
        "65": accel.ProcessMemory("unreadable"),
    }
    device_reads: list[tuple[set[int], tuple[str, ...]]] = []

    def process_rows(pids: set[int], kind: str) -> dict[int, accel.ProcessMemory]:
        # The real reclaim path must still establish OS exit; a failed driver
        # read cannot claim a live process's device allocations were released.
        return {pid: accel.ProcessMemory("unreadable") for pid in pids}

    def device_rows(
        pids: set[int], kind: str, devices: tuple[str, ...]
    ) -> dict[str, dict[int, accel.ProcessMemory]]:
        device_reads.append((set(pids), devices))
        return {device: {pid: expected[device] for pid in pids} for device in devices}

    monkeypatch.setattr(accel, "process_memories", process_rows)
    monkeypatch.setattr(accel, "process_memories_by_device", device_rows)
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            config,
            WorkerOptions(root=root / "worker", devices="64,65"),
            InMemoryControlHost(),
        )
        found = discover_installed(WEIGHTLESS)
        body = package_interface.build(found)
        interface = root / "metadata" / "package-interface.json"
        interface.parent.mkdir()
        interface.write_bytes(package_interface.canonical_bytes(body))
        group = worker.lanes.group((0, 1))
        assert worker.imposed(group)["CUDA_VISIBLE_DEVICES"] == "64,65"
        assert worker.imposed(group)["NCCL_NVLS_ENABLE"] == "0"
        # Run 2852: over PCI the group's communicator hung at formation; peer memory is
        # NVLink's only, by the seal.
        assert worker.imposed(group)["NCCL_P2P_LEVEL"] == "NVL"
        # the wide envelope lane a job takes IS a group lane (K = the envelope), so its
        # seal carries the row too; a ONE-device lane's seal is exactly what it was
        assert worker.imposed(worker.lanes.envelope)["NCCL_NVLS_ENABLE"] == "0"
        single = lanes.DeviceLane("lane-0", (0,), "64", worker_pid=os.getpid())
        assert "NCCL_NVLS_ENABLE" not in worker.imposed(single)
        assert "NCCL_P2P_LEVEL" not in worker.imposed(single)
        assert worker.imposed(single)["CUDA_VISIBLE_DEVICES"] == "64"
        executor = worker.supervision.spawn(imposed=worker.imposed(group))
        try:
            environ = Path(f"/proc/{executor.pid}/environ").read_bytes().split(b"\0")
            rows = dict(row.split(b"=", 1) for row in environ if b"=" in row)
            assert rows[b"CUDA_VISIBLE_DEVICES"] == b"64,65", rows
            assert rows[b"NCCL_NVLS_ENABLE"] == b"0", rows
            assert rows[b"NCCL_P2P_LEVEL"] == b"NVL", rows
            assert executor.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "64,65"
            assert executor.hello["rank"] == 0 and executor.hello["world"] == 1
            assert worker.supervision.devices == ("64", "65")
            command = _weightless_start(body, interface, devices="64,65")
            # the seal fences the devices before the degree is even read
            other = executor.call(msgspec.structs.replace(command, devices="64"), timeout=60.0)
            assert other["code"] == "env_seal_broken", other
        finally:
            evidence = worker.supervision.retire_current(executor, "seal arm done")
        assert evidence is not None
        assert not executor.alive()
        assert evidence.members == (executor.pid,)
        assert evidence.device == accel.ProcessMemory("absent", 0), evidence
        assert evidence.devices == expected, evidence
        assert device_reads == [({executor.pid}, ("64", "65"))]
        # degree checks, each on a fresh executor (a refused start poisons its epoch)
        for extra, code in (
            ({"sequence_parallel_degree": 3}, "sequence_parallel_degree_mismatch"),
            ({"sequence_parallel_degree": 2}, "sequence_parallel_unsupported"),
        ):
            executor = worker.supervision.spawn(imposed=worker.imposed(group))
            try:
                reply = executor.call(
                    _weightless_start(body, interface, devices="64,65", **extra), timeout=60.0
                )
                assert reply["code"] == code, reply
                if code == "sequence_parallel_degree_mismatch":
                    assert "sealed to 2 device(s) ('64,65')" in reply["detail"], reply
            finally:
                worker.supervision.retire_current(executor, "degree arm done")
            assert not executor.alive()
        worker.supervision.close()


@needs_executor
def test_degree_one_is_a_plain_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """A degenerate group on one card is byte-identical in behaviour to a plain executor:
    the same weightless start with `sequence_parallel_degree: 1` and without it answers
    the same reply (timing aside), and neither spawns a rank."""
    driverless(monkeypatch)
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            config,
            WorkerOptions(root=root / "worker", devices="64"),
            InMemoryControlHost(),
        )
        found = discover_installed(WEIGHTLESS)
        body = package_interface.build(found)
        interface = root / "metadata" / "package-interface.json"
        interface.parent.mkdir()
        interface.write_bytes(package_interface.canonical_bytes(body))
        replies: list[dict[str, Any]] = []
        for extra in ({}, {"sequence_parallel_degree": 1}):
            executor = worker.supervision.spawn(imposed=worker.imposed(worker.lanes.lanes[0]))
            try:
                reply = executor.call(
                    _weightless_start(body, interface, devices="64", **extra), timeout=120.0
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
        return {
            **{k: v for k, v in reply.items() if k not in ("stages", "rss_bytes")},
            "stages": [name for name, _ms in reply["stages"]],
        }

    assert stable(replies[0]) == stable(replies[1])
    assert replies[1]["follower_pids"] == [] and not replies[1]["torch"]


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
    port the kernel chose, pass the formation barrier, and refuse `gpu_divergence` on a
    broadcast plan whose hardware half is not theirs, naming each GPU as nvidia-smi does:
    sealed to cards 4-6, rank 1 is GPU 5. gloo stands in for NCCL: the rendezvous, the
    barrier, the plan and the seam are backend-agnostic."""
    import torch

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "4,5,6"}
    sealed = seal_snapshot(env)
    group = RankGroup(
        degree=3,
        backend="gloo",
        python=sys.executable,
        root=str(tmp_path),
        sample_seconds=0.2,
        launch=_launcher(env),
        devices=("4", "5", "6"),
    )
    try:
        group.spawn()
        assert [(f.rank, f.gpu) for f in group.followers] == [(1, "GPU 5"), (2, "GPU 6")]
        assert all(f.process.poll() is None for f in group.followers)
        # the followers are DIRECT CHILDREN of rank 0 (they inherit its cgroup and die
        # with it): pdeathsig was armed before exec
        for follower in group.followers:
            stat = Path(f"/proc/{follower.pid}/stat").read_text().rsplit(") ", 1)[1].split()
            assert int(stat[1]) == os.getpid(), "a follower's parent is rank 0"
        # each follower answers as its own rank of this world, under this seal
        group.dial_back(sealed)
        # Probe the actual followers before Torch is loaded: rank identity is a
        # process fact, available even for a weightless executor or failed prepare.
        for follower, probe in zip(group.followers, group.call({"cmd": "probe"}), strict=True):
            (process,) = probe["processes"]
            assert process["rank"] == follower.rank and process["pid"] == follower.pid
            assert process["started_ticks"] > 0
        assert group.port == 0, "no store exists before formation"
        group.form(torch, {"NCCL_NVLS_ENABLE": "0"})
        assert group.port > 1024, group.port
        assert group.formed
        assert torch.distributed.get_world_size() == 3 and torch.distributed.get_rank() == 0
        # the followers answered `join` only after their own barrier: the group is whole
        command = {
            "cmd": "load",
            "construction": "fixture",
            "devices": "4,5,6",
            "sequence_parallel_degree": 3,
            "authorized_device_limit_bytes": 1,
            "binding": {
                "application": "x:app",
                "package_interface": "/nowhere",
                "model_class": "",
            },
            "budgets": {},
        }
        # the hardware half disagrees: refused BEFORE torch is imported on the follower
        divergent = {"degree": 3, "devices": ["0", "1", "7"]}
        replies = group.call({**command, "group": divergent})
        assert [r["code"] for r in replies] == ["gpu_divergence"] * 2, replies
        assert replies[0]["detail"].startswith("GPU 5 is sealed to ['4', '5', '6']"), replies
        assert "GPU 4 broadcast 3 GPUs over ['0', '1', '7']" in replies[1]["detail"], replies
        assert replies[0]["rank"] == 1 and replies[1]["rank"] == 2
        # a follower that refused is poisoned: a second command says so
        replies = group.call({"cmd": "probe"})
        assert all(r["poisoned"] == "GPU divergence" for r in replies), replies
        # an out-of-vocabulary command on a follower refuses; rank 0's plane is closed
        replies = group.call({"cmd": "invoke"})
        assert [r["code"] for r in replies] == ["follower_command_unsupported"] * 2
    finally:
        group.close()
    assert all(f.process.poll() is not None for f in group.followers)
    assert not torch.distributed.is_initialized()


@needs_torch
def test_the_plan_agrees_field_by_field_or_refuses() -> None:
    """A follower ASSERTS the broadcast plan; the first field that differs is named, with
    the GPU that disagrees, and no GPU ever adapts."""
    plan = GroupPlan(2, ("2", "3"), "sha256:a", 1024)
    plan.assert_agrees(plan, rank=1)
    for field_name, other in (
        ("plan_digest", msgspec.structs.replace(plan, plan_digest="sha256:b")),
        (
            "authorized_device_limit_bytes",
            msgspec.structs.replace(plan, authorized_device_limit_bytes=1),
        ),
        ("degree", msgspec.structs.replace(plan, degree=4)),
    ):
        with pytest.raises(GpuDivergence) as refused:
            plan.assert_agrees(other, rank=1)
        assert refused.value.code == "gpu_divergence"
        assert refused.value.field_name == field_name and refused.value.gpu == "GPU 3"
        assert refused.value.message.startswith(f"GPU 3 disagrees on {field_name!r}: GPU 2 decided")
        assert "no GPU adapts locally" in refused.value.message


@needs_torch
def test_a_follower_death_breaks_the_group_loudly(tmp_path: Path) -> None:
    """A follower killed -9 is observed through its seam the moment it dies; the group
    latches `group_broken` naming its GPU and exit status, every later command
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
        devices=("0", "1"),
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
        assert "GPU 1" in refused.value.message and "-9" in refused.value.message, refused.value
        assert group.broken and "GPU 1" in group.broken
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
    arguments = [wire.marshal(a, spool) for a in (tensor, 1.5, "x", None, (1, 2))]
    assert (tmp_path / "ranks" / "tensor-0.raw").stat().st_size == 12
    assert arguments[0] == {
        "__sp__": "tensor",
        "v": "tensor-0.raw",
        "dtype": "bfloat16",
        "shape": [2, 3],
    }
    steps = wire.marshal([1, 2], spool)
    run = wire.RunCommand(
        ("m", "dit"), str(tmp_path), "ranks", (), False, True, {}, None, arguments, {"steps": steps}
    )
    command = msgspec.to_builtins(run)
    assert command["cmd"] == "run"
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
def test_a_seal_width_change_is_a_replacement_not_a_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
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
    driverless(monkeypatch)
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            config,
            WorkerOptions(root=root / "worker", devices="64,65"),
            InMemoryControlHost(),
        )
        group = worker.lanes.group((0, 1))
        single = lanes.DeviceLane("lane-0", (0,), "64", worker_pid=os.getpid())
        executor = worker.supervision.spawn(imposed=worker.imposed(single))
        try:
            # what the child really got, and what we recorded, are the same string
            assert executor.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "64"
            assert executor.sealed_devices == "64"
            # it MATCHES the lane it was spawned for, so an ordinary reuse still reuses
            assert executor.sealed_devices == single.devices
            # and it does NOT match the group lane, which is the whole fix: the reuse
            # branch in `_own_executor` now falls through to `replace` instead of handing
            # a one-card process to a two-card prepare
            assert executor.sealed_devices != group.devices
            assert group.devices == "64,65"
            # the width change also adds the NVLS row a group communicator needs
            # (pgw#929), which a reused one-device process would never have carried
            assert "NCCL_NVLS_ENABLE" not in worker.imposed(single)
            assert worker.imposed(group)["NCCL_NVLS_ENABLE"] == "0"
        finally:
            worker.supervision.close()


@needs_executor
def test_a_group_sealed_executor_records_both_ordinals(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same record, taken from the other side: spawned FOR the group, it says so."""
    driverless(monkeypatch)
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            config,
            WorkerOptions(root=root / "worker", devices="64,65"),
            InMemoryControlHost(),
        )
        group = worker.lanes.group((0, 1))
        executor = worker.supervision.spawn(imposed=worker.imposed(group))
        try:
            assert executor.sealed_devices == "64,65" == group.devices
            assert executor.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "64,65"
            # and it is not reusable for a one-card lane either -- the rule is symmetric
            single = lanes.DeviceLane("lane-0", (0,), "64", worker_pid=os.getpid())
            assert executor.sealed_devices != single.devices
        finally:
            worker.supervision.close()
