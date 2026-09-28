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
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from cozy_runtime.internal import accel, child_env, executor_commands, package_interface
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.internal.executor_commands import Start
from cozy_runtime.internal.worker import lanes
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import (
    HostedPlacement,
    Placement,
    Worker,
    WorkerOptions,
    executor_load_command,
)
from cozy_runtime.protocol import worker_pb2 as pb
from local_owner import claimable
from test_end_to_end import NO_EXECUTOR

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "examples" / "marco-polo"
#: A weightless application the executor imports from its installed environment.
WEIGHTLESS = "cozy_runtime.derive.operations:app"

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")


def _config(home: Path) -> RuntimeConfig:
    """The frozen config the CLI would read, built directly: `read_config` is once per
    process and the end-to-end suite already spends that read."""
    base = {k: v for k, v in os.environ.items() if not child_env.erased(k) and k != "PYTHONPATH"}
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


# --------------------------------------------------------------------------- step 0


# --------------------------------------------------------------------------- step 1


@needs_executor
def test_an_executor_is_sealed_to_exactly_its_lane() -> None:
    """THE SEAL. On a two-lane worker an executor spawned for lane 1 reports
    `CUDA_VISIBLE_DEVICES=1` from its own environment, and refuses `env_seal_broken` when
    told to prepare for any other lane."""
    with _workspace() as root:
        config = _config(root / "home")
        options = WorkerOptions(root=root / "worker", devices="0,1")
        worker = Worker(*claimable(config, options), InMemoryControlHost())
        assert [lane.devices for lane in worker.lanes.lanes] == ["0", "1"]
        assert worker.lanes.envelope.devices == "0,1"
        lane = worker.lanes.lanes[1]
        found = discover_installed(WEIGHTLESS)
        body = package_interface.build(found)
        interface = root / "metadata" / "package-interface.json"
        interface.parent.mkdir()
        interface.write_bytes(package_interface.canonical_bytes(body))
        executor = worker.supervision.spawn(imposed=worker.imposed(lane))
        try:
            assert executor.hello["sealed"]["CUDA_VISIBLE_DEVICES"] == "1"
            refused = executor.call(
                Start(
                    application=str(body["application"]),
                    package_interface=str(interface),
                    devices=worker.lanes.lanes[0].devices,
                ),
                timeout=60.0,
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
    after = accel.device_memory("0", kind)
    free_mib, total_mib = (int(part) for part in projected.split(","))
    # Other processes move free memory between reads; nvidia-smi's must lie between NVML's
    # readings taken either side of it, to its MiB rounding.
    low, high = sorted((real.free_bytes, after.free_bytes))
    assert low - (1 << 20) < free_mib * (1 << 20) < high + (1 << 20), (real, after, projected)
    assert real.total_bytes == total_mib * (1 << 20), (real, projected)

    # THE CEILING IS THE DEVICE: the lock is the reservation and idle co-tenants are
    # evicted for the tenant holding it, so nothing is partitioned in advance.
    with _workspace() as root:
        worker = Worker(
            *claimable(_config(root / "home"), WorkerOptions(root=root / "worker", devices="0")),
            InMemoryControlHost(),
        )
        try:
            lane = worker.lanes.lanes[0]
            assert worker._ceiling(lane, "a") == real.total_bytes
        finally:
            worker.supervision.close()


def test_a_load_without_an_assigned_ceiling_is_a_worker_defect() -> None:
    """The executor's self-derived ceiling path is DELETED, not defaulted: the load
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
        interface_path="/nowhere",
    )
    command = executor_load_command(
        binding, construction="sha256:" + "1" * 64, devices="1", authorized_device_limit_bytes=7
    )
    assert command.devices == "1" and executor_commands.encode(command)["cmd"] == "load"
    assert command.authorized_device_limit_bytes == 7
    # Executors before the typed commands index `binding` and `budgets` in a one-model load,
    # a weightless one included (0.18.70/0.18.71 omitted a default `budgets`).
    wire = executor_commands.encode(command)
    assert {"binding", "budgets"} <= set(wire) and "models" not in wire
    # ...and index the weight budget by its wire key.
    assert wire["budgets"] == {"declared_weight_bytes": 0}


# --------------------------------------------------------------------------- step 3/4


def test_placements_are_assigned_by_measured_fit_or_refused_typed() -> None:
    """Fit is arithmetic over MEASURED bytes, in steps (cr-066, then cr-022): a
    model-bearing placement takes a lane with room and no other model-bearing tenant; else
    a lane with measured headroom BESIDE its tenants (co-residency); else a lane whose
    measured TOTAL holds it alone (time-sliced sharing); else any readable lane (bounded
    preparation). A lane the driver cannot read is never chosen. Choosing authorizes no
    bytes, and binding takes exactly the chosen ordinals."""
    lanes_ = lanes.LaneSet.from_envelope("0,1,2", worker_pid=os.getpid())
    assert len(lanes_.lanes) == 3 and lanes_.envelope.ordinals == (0, 1, 2) and lanes_.wide
    measured = {
        0: accel.DeviceMemory("measured", 4 << 30, 8 << 30),
        1: accel.DeviceMemory("measured", 16 << 30, 24 << 30),
        2: accel.DeviceMemory("unreadable"),
    }

    def place(name: str, declared: int, candidates: tuple[int, ...] = (0, 1, 2)) -> str:
        ordinals = lanes_.choose(
            candidates, 1, model_bearing=True, declared_bytes=declared, measured=measured
        )
        lane = lanes_.bind(name, ordinals, model_bearing=True, measured=measured)
        lane.row(name).model_bearing = True
        return lane.lane_id

    assert place("big", 8 << 30) == "lane-1"  # step 1: the only empty lane with room
    assert place("small", 2 << 30) == "lane-0"  # step 1 again: lane-0 is still empty
    assert place("beside", 1 << 30) == "lane-0"  # step 2: headroom beside a tenant
    assert place("shared", 6 << 30, (0,)) == "lane-0"  # step 3: its total holds it alone
    assert place("huge", 32 << 30) == "lane-1"  # step 4: bounded preparation, fewest tenants
    with pytest.raises(lanes.LaneRefusal) as refused:
        lanes_.choose((2,), 1, model_bearing=True, declared_bytes=1 << 20, measured=measured)
    assert refused.value.code == "device_lane_infeasible"
    with pytest.raises(lanes.LaneRefusal) as refused:
        lanes_.bind("blind", (2,), model_bearing=True, measured=measured)
    assert refused.value.code == "device_pin_infeasible"
    assert "ordinal 2 ('2') free bytes unreadable" in refused.value.detail
    assert lanes_.lane_of("blind") is None
    # weightless placements spread over the lanes with the fewest tenants
    assert lanes_.choose((0, 1, 2), 1, model_bearing=False, declared_bytes=0, measured={}) == (2,)
    lanes_.release("big")
    assert lanes_.lane_of("big") is None and "big" not in lanes_.lanes[1].rows


def test_the_ledger_row_survives_a_switch_to_another_placement() -> None:
    """Per-executor rows under a per-lane ledger: binding an attempt to placement B wipes
    nothing placement A's row measured (it used to `begin_generation` the one ledger)."""
    with _workspace() as root:
        config = _config(root / "home")
        worker = Worker(
            *claimable(config, WorkerOptions(root=root / "worker", devices="0")),
            InMemoryControlHost(),
        )
        try:
            worker.accepted.mode = "serving"
            primary = Placement(
                pb.Placement(placement_id="p0", bindings_digest=b"b" * 32), b"", lane_id="lane-0"
            )
            worker.placement = primary
            hosted = dataclasses.replace(
                primary, document=pb.Placement(placement_id="p1", bindings_digest=b"b" * 32)
            )
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
