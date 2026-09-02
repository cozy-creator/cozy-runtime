"""Device lanes (cr-066): the fence, the seal, the assigned ceiling, the per-lane rows.

Every arm below drives the REAL worker — real `Worker`, real trampoline, real executor
processes dialling back over the real seam — against `examples/marco-polo`, and reads what
the child process itself reported about its sealed environment. Nothing is patched.

WHAT IS REAL AND WHAT IS SIMULATED. This box has one card. A two-entry envelope
(`--devices 0,1`) is therefore a real envelope naming one card that exists and one that does
not: the fence, the per-lane seal and the lane arithmetic are exercised for real (the
worker is torch-free and a weightless executor never opens a CUDA context, so the absent
card is never touched), and the driver's answer for entry `1` is the honest UNREADABLE the
lane logic is built to refuse on. Concurrent attempts on two real cards are the live proof
cr-066 owes on a two-GPU box; they are not claimed here.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import accel, child_env
from cozy_runtime.internal import descriptor as desc
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import (
    LocalRecordOwner,
    LocalRequest,
    SliceOutcome,
    development_source_digest,
    run_slice,
)
from cozy_runtime.internal.worker import lane_wire, lanes
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import (
    HostedPlacement,
    Placement,
    Worker,
    WorkerOptions,
    executor_prepare_command,
)
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "examples" / "marco-polo"

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


def _config(home: Path) -> RuntimeConfig:
    """The frozen config the CLI would read, built directly: `read_config` is once per
    process and the end-to-end suite already spends that read."""
    base = {
        k: v
        for k, v in os.environ.items()
        if not child_env.erased(k) and k not in ("PYTHONPATH", "COZY_BOOTSTRAP_CREDENTIAL")
    }
    return RuntimeConfig(
        cozy_home=home,
        credentials=Credentials(),
        child_base_env=tuple(sorted(base.items())),
    )


@contextlib.contextmanager
def _workspace() -> Iterator[Path]:
    # Short: the executor's control socket lives under it and `sun_path` holds 108 bytes.
    with tempfile.TemporaryDirectory(prefix="cz-lanes.", dir="/tmp") as raw:
        yield Path(raw)


def _slice(
    root: Path, *, devices: str, kind: str, hook: Any = None, owner_hook: Any = None
) -> tuple[SliceOutcome, list[str]]:
    """One marco-polo request through `run_slice`, exactly as `cozy-runtime run`/`job`
    builds it, with the worker's device envelope chosen by the arm."""
    config = _config(root / "home")
    found = discover(PACKAGE)
    body = desc.build(found)
    workspace = root / "run"
    workspace.mkdir(parents=True, exist_ok=True)
    revision = development_source_digest(PACKAGE)
    job_records: list[dict[str, Any]] = []
    descriptor_id = ""
    if kind == "job":
        descriptor_id = desc.job_descriptor_id(body, "marco_job")
        descriptor_path = workspace / "package-descriptor.json"
        descriptor_path.write_bytes(desc.canonical_bytes(body))
        job_records.append(
            {
                "job_descriptor_id": descriptor_id,
                "build_id": revision,
                "application": str(body["application"]),
                "package_descriptor": str(descriptor_path),
                "python": sys.executable,
                "environment_content_digest": revision,
                "job": "marco_job",
                "publishes": False,
                "emits_media": False,
                "gpu_rate_micro_usd_per_hour": 0,
                "cap_micro_usd": 0,
                "reclaim_on_terminal": True,
            }
        )
    request = LocalRequest(
        entrypoint="marco_job" if kind == "job" else "marco",
        payload={"message": "marco"},
        outputs=(),
        request_id=f"lanes-{kind}",
        kind=kind,
        job_descriptor_id=descriptor_id,
    )
    seals: list[str] = []

    def capture(owner: LocalRecordOwner, worker: Worker) -> None:
        # The harness reads the child's OWN report of its sealed environment at the moment
        # the supervision publishes it — the same `hello` the worker digest-checks.
        if owner_hook is not None:
            owner_hook(owner)
        publish = worker.supervision.on_change

        def on_change(executor: Any) -> None:
            if executor is not None:
                seals.append(str(executor.hello["sealed"]["CUDA_VISIBLE_DEVICES"]))
            if publish is not None:
                publish(executor)

        worker.supervision.on_change = on_change
        if hook is not None:
            hook(worker)

    with contextlib.chdir(PACKAGE), contextlib.redirect_stdout(sys.stderr):
        outcome = run_slice(
            config,
            request,
            job_records=job_records,
            workspace=workspace,
            package_revision_digest=revision,
            package_descriptor=body if kind == "entrypoint" else None,
            found=found,
            development_artifact_store=config.cozy_home / "cas",
            options=WorkerOptions(root=workspace / "worker", devices=devices, no_warm=True),
            record_owner_hook=capture,
        )
    return outcome, seals


# --------------------------------------------------------------------------- step 0


@needs_executor
def test_one_lane_serves_sealed_to_its_device() -> None:
    """The worker every deployment produces today: one device, one lane, one seal, served."""
    with _workspace() as root:
        outcome, seals = _slice(root, devices="0", kind="entrypoint")
    assert outcome.result == {"message": "polo"}, (outcome.error, outcome.faults)
    assert seals == ["0"], seals


def _owner_below_lanes(owner: LocalRecordOwner) -> None:
    """Re-author the local owner as one that negotiates the minor BEFORE device lanes
    (proto-024, 22): the queued claim and every desired state it will send say 21. This
    is the older peer's exact bytes, not a switch inside the runtime."""
    older = lane_wire.LANES_WIRE_MINOR - 1
    queued = owner.host.inbound.get_nowait()
    assert queued is not None and queued.WhichOneof("msg") == "claim"
    claim = queued.claim
    owner.host.deliver(
        pb.RecordOwnerFrame(
            claim=pb.Claim(
                record_owner_epoch=claim.record_owner_epoch,
                record_owner_id=claim.record_owner_id,
                wire_minor=older,
            )
        )
    )
    desired_state = owner._desired_state

    def _desired_state() -> pb.DesiredWorkerState:
        directive = desired_state()
        directive.wire_minor = older
        return directive

    owner._desired_state = _desired_state  # type: ignore[method-assign]


@needs_executor
def test_a_two_device_serving_envelope_refuses_an_owner_below_minor_22() -> None:
    """THE FENCE. A serving set from a RecordOwner that negotiated minor 21 on a two-lane
    worker refuses `device_lanes_unsupported` (that owner reads one count over two lanes),
    the fault names the minor that carries `lanes[]`, no executor is ever spawned, and
    the worker stays ONLINE (it answered; it did not fail)."""
    workers: list[Worker] = []
    with _workspace() as root:
        outcome, seals = _slice(
            root,
            devices="0,1",
            kind="entrypoint",
            hook=workers.append,
            owner_hook=_owner_below_lanes,
        )
    assert outcome.result is None
    assert "device_lanes_unsupported" in outcome.error, outcome.error
    assert "negotiated wire minor 21" in outcome.error, outcome.error
    reasons = [fault["reason"] for fault in outcome.faults]
    assert "device_lanes_unsupported" in reasons, reasons
    assert seals == [], f"an executor was spawned behind the fence: {seals}"
    (worker,) = workers
    assert worker.phase == pb.WorkerPhase.WORKER_PHASE_ONLINE
    assert worker.latched == "device_lanes_unsupported"
    assert worker.fence.wire_minor == lane_wire.LANES_WIRE_MINOR - 1
    assert worker.supervision.spawns == 0


@needs_executor
def test_a_two_device_serving_envelope_serves_an_owner_at_minor_22_per_lane() -> None:
    """The fence LIFTED. The same set from an owner at 22 is served: the weightless
    placement lands on lane-0 (fewest placements), sealed to device 0 alone, and every
    observed state the owner read carries both lanes, the placement's `device_lane_id`,
    and a worker-level count that is the SUM over lanes."""
    owners: list[LocalRecordOwner] = []
    with _workspace() as root:
        outcome, seals = _slice(root, devices="0,1", kind="entrypoint", owner_hook=owners.append)
    assert outcome.result == {"message": "polo"}, (outcome.error, outcome.faults)
    assert seals == ["0"], seals
    (owner,) = owners
    served = [r for r in owner.reports if r.lanes and r.placements]
    assert served, "no observed state carried lanes and a placement"
    for report in served:
        assert [lane.lane_id for lane in report.lanes] == ["lane-0", "lane-1"]
        assert [list(lane.device_ordinals) for lane in report.lanes] == [[0], [1]]
        assert report.available_attempt_slots == sum(
            lane.available_attempt_slots for lane in report.lanes
        )
        (status,) = report.placements
        assert status.device_lane_id == "lane-0"
        assert list(report.lanes[0].placement_ids) == [status.placement_id]
        assert list(report.lanes[1].placement_ids) == []
    assert report.applied_wire_minor == lane_wire.LANES_WIRE_MINOR


@needs_executor
def test_a_job_keeps_the_whole_envelope() -> None:
    """A job's executor is sealed to the ENVELOPE lane — every device, one seat."""
    with _workspace() as root:
        outcome, seals = _slice(root, devices="0,1", kind="job")
    assert outcome.result == {"message": "polo"}, (outcome.error, outcome.faults)
    assert seals == ["0,1"], seals


# --------------------------------------------------------------------------- step 1


@needs_executor
def test_an_executor_is_sealed_to_exactly_its_lane() -> None:
    """THE SEAL. On a two-lane worker an executor spawned for lane 1 reports
    `CUDA_VISIBLE_DEVICES=1` from its own environment, and refuses `env_seal_broken` when
    told to prepare for any other lane."""
    with _workspace() as root:
        config = _config(root / "home")
        options = WorkerOptions(root=root / "worker", devices="0,1", no_warm=True)
        worker = Worker(config, options, InMemoryControlHost())
        assert [lane.devices for lane in worker.lanes.lanes] == ["0", "1"]
        assert worker.lanes.envelope.devices == "0,1"
        lane = worker.lanes.lanes[1]
        found = discover(PACKAGE)
        body = desc.build(found)
        descriptor = root / "package-descriptor.json"
        descriptor.write_bytes(desc.canonical_bytes(body))
        executor = worker.supervision.spawn(imposed=worker.imposed(lane))
        try:
            assert executor.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "1"
            weightless = {
                "cmd": "prepare",
                "binding": {
                    "application": str(body["application"]),
                    "package_descriptor": str(descriptor),
                    "development_project": str(PACKAGE),
                    "model_class": "",
                },
                "budgets": {},
                "expect_construction_digest": "",
            }
            refused = executor.call(
                {**weightless, "devices": worker.lanes.lanes[0].devices}, timeout=60.0
            )
            assert refused["code"] == "env_seal_broken", refused
            assert "'1'" in refused["detail"] and "'0'" in refused["detail"], refused
        finally:
            worker.supervision.retire_current(executor, "arm done")
            worker.supervision.close()


# --------------------------------------------------------------------------- step 2


def test_ceilings_are_measured_and_never_double_booked() -> None:
    """THE ASSIGNED CEILING. The driver's own number for the card that exists, UNREADABLE
    for the one that does not, and two executors authorized back to back on one lane can
    never both be handed the same headroom.

    The reading is NVML in-process (xs-007 row 19); `nvidia-smi` is what an operator reads,
    so the two are held to agree: NVML is the byte count and nvidia-smi its MiB rounding,
    so free may differ by under one MiB and total not at all.
    """
    kind = accel.host_backend_family()
    real = accel.device_memory("0", kind)
    if real.state != "measured":
        pytest.skip("no driver-readable device 0 on this host")
    assert 0 < real.free_bytes <= real.total_bytes
    assert accel.device_memory("1", kind).state == "unreadable"
    projected = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    free_mib, total_mib = (int(part) for part in projected.split(","))
    assert abs(real.free_bytes - free_mib * (1 << 20)) < (1 << 20), (real, projected)
    assert real.total_bytes == total_mib * (1 << 20), (real, projected)

    lane = lanes.DeviceLane("lane-0", (0,), "0", worker_pid=os.getpid())
    first = lane.authorize("a", real.free_bytes)
    assert first == real.free_bytes
    second = lane.authorize("b", real.free_bytes)
    assert second == 0, "the second executor was handed headroom the first still holds"
    assert lane.rows["a"].ledger.authorized_device_limit_bytes == first
    lane.settle("a")
    assert lane.authorize("b", real.free_bytes) == real.free_bytes


def test_a_prepare_without_an_assigned_ceiling_is_a_worker_defect() -> None:
    """The executor's self-derived ceiling path is DELETED, not defaulted: the prepare
    command always carries the assignment, and the seal it must match."""
    from cozy_runtime.internal.worker.plan import DeclaredBinding

    binding = DeclaredBinding(
        entrypoint_binding_digest="sha256:" + "0" * 64,
        entrypoint="marco",
        model_class="",
        model_binding_path="",
        model_parameter_name="",
        release="marco-polo/1.0.4",
        application="marco_polo_package:app",
        descriptor_path="/nowhere",
    )
    command = executor_prepare_command(binding, devices="1", authorized_device_limit_bytes=7)
    assert command["devices"] == "1"
    assert command["authorized_device_limit_bytes"] == 7


# --------------------------------------------------------------------------- step 3/4


def test_placements_are_assigned_by_measured_fit_or_refused_typed() -> None:
    """Fit is arithmetic over MEASURED bytes, in three steps (cr-066, then cr-022): a
    model-bearing placement takes a lane with room and no other model-bearing tenant; else
    a lane with measured headroom BESIDE its tenants (co-residency); else a lane whose
    measured TOTAL holds it alone (time-sliced sharing). A lane the driver cannot read is
    never chosen; nothing fitting is a typed refusal that moves no other placement. On a
    WIDE envelope the candidate lane is the placement's PIN (cr-068): the worker validates,
    it never chooses which paid card to idle."""
    lanes_ = lanes.LaneSet.from_envelope("0,1,2", worker_pid=os.getpid())
    assert len(lanes_) == 3 and lanes_.envelope.ordinals == (0, 1, 2) and lanes_.wide
    measured = {
        0: accel.DeviceMemory("measured", 4 << 30, 8 << 30),
        1: accel.DeviceMemory("measured", 16 << 30, 24 << 30),
        2: accel.DeviceMemory("unreadable"),
    }
    big = lanes_.assign(
        "big", model_bearing=True, declared_bytes=8 << 30, measured=measured, pin=(1,)
    )
    assert big.lane_id == "lane-1"
    lanes_.lanes[1].row("big").model_bearing = True
    small = lanes_.assign(
        "small", model_bearing=True, declared_bytes=2 << 30, measured=measured, pin=(0,)
    )
    assert small.lane_id == "lane-0"
    lanes_.lanes[0].row("small").model_bearing = True
    # step 2: lane-0 has a tenant; its measured headroom still covers 1 GiB beside it
    beside = lanes_.assign(
        "beside", model_bearing=True, declared_bytes=1 << 30, measured=measured, pin=(0,)
    )
    assert beside.lane_id == "lane-0"
    lanes_.lanes[0].row("beside").model_bearing = True
    # step 3: lane-0 has no 6 GiB of headroom; its 8 GiB total holds it alone (time-sliced)
    shared = lanes_.assign(
        "shared", model_bearing=True, declared_bytes=6 << 30, measured=measured, pin=(0,)
    )
    assert shared.lane_id == "lane-0"
    lanes_.lanes[0].row("shared").model_bearing = True
    with pytest.raises(lanes.LaneRefusal) as refused:
        lanes_.assign(
            "huge", model_bearing=True, declared_bytes=32 << 30, measured=measured, pin=(1,)
        )
    assert refused.value.code == "device_pin_infeasible"
    assert "B total for 34359738368 B" in refused.value.detail
    with pytest.raises(lanes.LaneRefusal) as refused:
        lanes_.assign(
            "blind", model_bearing=True, declared_bytes=1 << 20, measured=measured, pin=(2,)
        )
    assert refused.value.code == "device_pin_infeasible"
    assert "lane-2 free bytes unreadable" in refused.value.detail
    # NO PIN on a wide envelope: refused typed (ruling 3), nothing moved
    with pytest.raises(lanes.LaneRefusal) as refused:
        lanes_.assign("unpinned", model_bearing=True, declared_bytes=1 << 20, measured=measured)
    assert refused.value.code == "device_group_unsupported"
    assert "carries no device pin on a 3-device envelope" in refused.value.detail
    assert lanes_.lane_of("huge") is None and lanes_.lane_of("unpinned") is None
    assert lanes_.lane_of("big") is lanes_.lanes[1] and lanes_.lane_of("small") is lanes_.lanes[0]
    # weightless placements spread over the lanes with the fewest tenants
    weightless = lanes_.assign("w1", model_bearing=False, declared_bytes=0, measured={})
    assert weightless is lanes_.lanes[2]
    lanes_.release("big")
    assert lanes_.lane_of("big") is None and "big" not in lanes_.lanes[1].rows
    # a ONE-device envelope needs no pin: the worker's one lane is the candidate
    one = lanes.LaneSet.from_envelope("0", worker_pid=os.getpid())
    assert not one.wide
    alone = one.assign(
        "solo", model_bearing=True, declared_bytes=1 << 30, measured={0: measured[0]}
    )
    assert alone is one.lanes[0]


def test_one_lane_worker_seats_and_a_two_lane_worker_sum() -> None:
    """`available_attempt_slots` is the SUM of per-lane seats: one on the one-lane worker,
    unchanged on the wire; a held attempt takes only its own lane's seat."""
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            config,
            WorkerOptions(root=root / "worker", devices="0,1", no_warm=True),
            InMemoryControlHost(),
        )
        try:
            assert worker.available_attempt_slots() == 0  # admission CLOSED before a claim
            worker.phase = pb.WorkerPhase.WORKER_PHASE_ONLINE
            worker.fence.dispatch_open = True
            worker.accepted.mode = "serving"
            primary = Placement(
                placement_id="p0",
                placement_set_digest=b"",
                package={},
                development={},
                environment_digest=b"",
                package_descriptor=("", 0),
                bindings_digest=b"",
                models=(),
                entrypoints=(),
                environment={},
                document={},
                serving=pb.ServingState.SERVING_STATE_DISPATCHABLE,
            )
            hosted = dataclasses.replace(primary, placement_id="p1")
            worker.placement = primary
            worker.hosted["p1"] = HostedPlacement(
                placement=hosted, supervision=worker._new_placement_supervision("p1")
            )
            worker.lanes.lanes[0].placements.add("p0")
            primary.lane_id = "lane-0"
            worker.lanes.lanes[1].placements.add("p1")
            hosted.lane_id = "lane-1"
            assert worker.available_attempt_slots() == 2
            held = AttemptRecord(request_id="r", attempt=1, digest=b"", spec={})
            held.placement_id = "p0"
            held.lane_id = "lane-0"
            held.state = "running"
            worker.engine.history[held.key()] = held
            assert worker._lane_slots(worker.lanes.lanes[0]) == 0
            assert worker._lane_slots(worker.lanes.lanes[1]) == 1
            assert worker.available_attempt_slots() == 1
            # lane B's gate is untouched by lane A rebuilding
            worker._settle()
            assert worker.lanes.lanes[1].settled.is_set()
            worker.lanes.lanes[0].settled.clear()
            other = AttemptRecord(request_id="s", attempt=1, digest=b"", spec={})
            other.lane_id = "lane-1"
            worker.await_settled(other)  # returns at once; lane-0's rebuild is not its wait
        finally:
            for slot in worker.hosted.values():
                slot.supervision.close()
            worker.supervision.close()


def test_the_ledger_row_survives_a_switch_to_another_placement() -> None:
    """Per-executor rows under a per-lane ledger: binding an attempt to placement B wipes
    nothing placement A's row measured (it used to `begin_generation` the one ledger)."""
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            config,
            WorkerOptions(root=root / "worker", devices="0", no_warm=True),
            InMemoryControlHost(),
        )
        try:
            worker.accepted.mode = "serving"
            primary = Placement(
                placement_id="p0",
                placement_set_digest=b"",
                package={},
                development={},
                environment_digest=b"",
                package_descriptor=("", 0),
                bindings_digest=b"",
                models=(),
                entrypoints=(),
                environment={},
                document={},
                lane_id="lane-0",
            )
            worker.placement = primary
            hosted = dataclasses.replace(primary, placement_id="p1")
            worker.hosted["p1"] = HostedPlacement(
                placement=hosted, supervision=worker._new_placement_supervision("p1")
            )
            lane = worker.lanes.lanes[0]
            lane.placements.update({"p0", "p1"})
            row = worker.primary_ledger()
            row.observe_construction({"filled_bytes": 1234, "filled": 3, "allocator_bytes": 99})
            assert row.weights_bytes == 1234
            for placement_id in ("p1", "p0", "p1"):
                attempt = AttemptRecord(request_id=placement_id, attempt=1, digest=b"", spec={})
                attempt.placement_id = placement_id
                slot = worker._bind_attempt_slot(attempt)
                assert slot.lane_id == "lane-0" and slot.devices == "0"
            assert worker.primary_ledger() is row
            assert row.weights_bytes == 1234 and row.device_baseline == 99
            assert set(lane.rows) == {"p0", "p1"}
        finally:
            for slot_ in worker.hosted.values():
                slot_.supervision.close()
            worker.supervision.close()
