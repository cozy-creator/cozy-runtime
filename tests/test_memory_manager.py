"""The Worker's memory policy (`worker/memory.py`) over real objects.

The arithmetic runs on a real LaneSet, real Ledgers and executor replies decoded exactly as
the Worker decodes them; nothing reads a driver. The reclaim legs at the end drive a real
Worker with real executor processes, as `test_gpu_scheduler` does, over its four-device
envelope.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from cozy_runtime.internal import accel, proctree, weight_policy
from cozy_runtime.internal.executor_replies import AttemptReply, decode
from cozy_runtime.internal.worker.attempts import AttemptRecord, AttemptSlot
from cozy_runtime.internal.worker.lanes import LaneSet
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.memory import (
    Budget,
    MemoryManager,
    floor_bytes,
    growth,
    pinned_split,
    reach,
    weights,
)
from cozy_runtime.internal.worker.plan import (
    PLANE_PLACEMENT,
    DeclaredBinding,
    PlanChooser,
    PreparedModel,
    PreparedRequest,
)
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR
from test_gpu_scheduler import VIRTUAL, Machine, driverless

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")

GiB = 1 << 30
TOTAL = 80 * GiB
SHAPE = ("segment", "frames=362")


def test_growth_is_a_shapes_own_else_the_least_larger_measured_one() -> None:
    bank = {("s", "frames=242,width=512"): 30, ("s", "frames=362,width=768"): 50}
    assert growth(("s", "frames=242,width=512"), bank) == 30
    assert growth(("s", "frames=121,width=512"), bank) == 30
    assert growth(("s", "frames=300,width=512"), bank) == 50
    assert growth(("s", "frames=500,width=512"), bank) is None, "nothing measured covers it"
    assert growth(("s", "frames=121"), bank) is None, "other axes describe another shape"
    assert growth(("t", "frames=121,width=512"), bank) is None, "another entrypoint"


def test_an_attempt_banks_what_its_executor_reports_and_a_device_failure_forgets_it() -> None:
    lanes = LaneSet.from_envelope("0", worker_pid=os.getpid())
    lane = lanes.lanes[0]
    row = lane.row("p")
    row.emptied.add(0)
    manager = MemoryManager(lanes, lambda *_: None, lambda *_: False)
    planed = decode(
        {
            "ok": True,
            "plane": {"activation_peak_bytes": 3 * GiB, "committed_bytes": 5 * GiB, "late": 2},
            "metrics": {"activation_peak_bytes": 9 * GiB},
        },
        AttemptReply,
    )
    manager.settle(lane, "p", SHAPE, planed, ok=True, forget=False)
    assert row.activation[SHAPE] == 3 * GiB, "the plane's own measurement, not the metric"
    assert row.ledger.plane is not None and row.ledger.plane.committed_bytes == 5 * GiB
    assert not row.emptied, "a call that ran holds its devices again"
    older = decode(
        {"ok": True, "metrics": {"activation_peak_bytes": GiB, "activation_peaks": {"d": 4 * GiB}}},
        AttemptReply,
    )
    row.ledger.observe_attempt(older.metrics, cell=SHAPE[1], succeeded=True)
    manager.settle(lane, "p", SHAPE, older, ok=True, forget=False)
    assert row.activation[SHAPE] == 4 * GiB, "an older executor's per-scope peak, the most seen"
    manager.settle(lane, "p", SHAPE, older, ok=False, forget=False)
    assert row.activation[SHAPE] == 4 * GiB, "a failure measures nothing"
    manager.settle(lane, "p", SHAPE, older, ok=False, forget=True)
    assert SHAPE not in row.activation, "a device failure makes the shape unmeasured again"


def test_activations_come_first_and_torch_cache_is_reused_not_counted_twice() -> None:
    usable = reach(6 * GiB, 2 * GiB, GiB, 3 * GiB)
    assert usable == 9 * GiB, "free, what the plane maps, and the cache the growth reuses"
    # 6 free + 2 mapped, less the 2 GiB the growth needs beyond torch's 1 GiB cache
    assert weight_policy.plane_budget(usable, 3 * GiB) == 6 * GiB - weight_policy.MARGIN
    assert reach(6 * GiB, 2 * GiB, 5 * GiB, 3 * GiB) == 11 * GiB, "torch keeps the rest"
    assert reach(6 * GiB, -1, 0, 0) == 6 * GiB, "an unreadable plane maps nothing"
    assert Budget(vram={0: 5 * GiB, 1: 4 * GiB}).plane_bytes == 4 * GiB, "the tightest GPU"
    assert Budget().plane_bytes == -1, "the executor derives its own"


def test_the_refusal_floor_is_window_one_over_a_stage_or_its_largest_component() -> None:
    ledger = Ledger(worker_pid=os.getpid())
    ledger.begin_generation(1)
    ledger.observe_construction(
        {
            "filled_bytes": 7 * GiB,
            "layouts": {
                "text_encoder": {"common": GiB // 8, "blocks": [GiB // 8] * 15},
                "unet": {"common": GiB // 4, "blocks": [GiB // 4, GiB // 2, GiB // 4]},
                "vae": {"common": GiB // 2},
            },
        }
    )
    assert weights(ledger) == 15 * GiB // 4 and weights(ledger, ("unet",)) == 5 * GiB // 4
    layouts = ledger.layouts
    assert floor_bytes(layouts, ("unet",)) == 3 * GiB // 4
    assert floor_bytes(layouts, ("text_encoder", "unet")) == GiB
    assert floor_bytes(layouts, ()) == 3 * GiB // 4, "components in turn"
    ledger.begin_generation(2)
    ledger.observe_construction({"filled_bytes": 7 * GiB})
    assert weights(ledger) == 7 * GiB, "an older executor's filled bytes"


def test_the_pinned_tier_takes_half_the_host_most_recently_used_first() -> None:
    three = [("a", 8 * GiB, 0), ("b", 8 * GiB, 0), ("c", 8 * GiB, 0)]
    assert pinned_split(20 * GiB, three) == {"a": 8 * GiB, "b": 2 * GiB, "c": 0}
    # What the tier pins counts toward the half; the idle tenant's share shrinks, and the
    # one about to run rises by at most half of what is free now.
    split = pinned_split(4 * GiB, [("a", 8 * GiB, 0), ("b", 8 * GiB, 8 * GiB)])
    assert split == {"a": 2 * GiB, "b": 4 * GiB}
    assert pinned_split(-1, [("a", GiB, 0)]) == {"a": 0}
    # The machine's pinned total is one sum: memory pinned outside these tenants (a tier the
    # Worker keeps for an executor that is gone) counts in it, and leaves them less.
    two = [("a", 8 * GiB, 5 * GiB), ("b", 8 * GiB, 0)]
    assert pinned_split(3 * GiB, two) == {"a": 4 * GiB, "b": 0}
    assert pinned_split(3 * GiB, two, held=9 * GiB) == {"a": 6 * GiB, "b": 0}
    assert pinned_split(0, two, held=9 * GiB) == {"a": 4 * GiB + GiB // 2, "b": 0}


def test_a_cgroups_page_cache_and_shared_memory_are_read_in_either_version(
    tmp_path: Path,
) -> None:
    """A rented pod is cgroup v1 (its subtree's counters under `total_`), a desktop v2."""
    (tmp_path / "memory.stat").write_text(
        "cache 900\nshmem 7\ntotal_inactive_file 300\ntotal_active_file 200\ntotal_shmem 100\n"
    )
    v1 = {"inactive_file": 300, "active_file": 200, "shmem": 100}
    assert proctree._cgroup_stat(tmp_path) == v1
    (tmp_path / "memory.stat").write_text("anon 5\nfile 900\nshmem 100\ninactive_file 300\n")
    assert proctree._cgroup_stat(tmp_path) == {"inactive_file": 300, "active_file": 0, "shmem": 100}


def test_a_plane_document_is_telemetry_and_a_new_generation_forgets_it() -> None:
    ledger = Ledger(worker_pid=os.getpid())
    ledger.begin_generation(1)
    ledger.observe_plane(
        {
            "committed_bytes": 3 * GiB,
            "resident": {"unet": 3 * GiB},
            "h2d_gbps": "fast",
            "from_a_newer_executor": 1,
        }
    )
    assert ledger.plane is not None
    assert (ledger.plane.committed_bytes, ledger.plane.h2d_gbps) == (3 * GiB, 0.0)
    ledger.observe_plane(None)
    assert ledger.plane is not None and ledger.plane.budget_bytes == -1, "unreadable, not 0"
    ledger.device_reserved, ledger.device_allocated = 3 * GiB, GiB
    assert ledger.slack == 2 * GiB
    ledger.begin_generation(2)
    assert ledger.slack == 0 and ledger.plane is None


def test_a_plane_tenant_plans_weight_plane_whatever_room_it_was_given() -> None:
    ledger = Ledger(worker_pid=os.getpid())
    ledger.begin_generation(1)
    ledger.observe_construction({"resident": {"unet": GiB}, "parked": ["vae"]})
    binding = DeclaredBinding(
        entrypoint_binding_digest="sha256:" + "0" * 64,
        entrypoint="serve",
        model_class="M",
        model_binding_path="serve.models.m",
        model_parameter_name="m",
        release="r",
        logical_weight_bytes=GiB,
    )
    chooser, model = PlanChooser(ledger), PreparedModel(delivery_rung="verbatim")
    prepared = PreparedRequest.unresolved("serve", {"width": 512})
    assert chooser.choose(binding, model, prepared, fits=False).placement == "component_staged"
    ledger.observe_construction({"layouts": {"unet": {"common": GiB // 2, "blocks": [GiB]}}})
    ledger.observe_plane({"resident": {"unet": GiB // 2}})
    plan = chooser.choose(binding, model, prepared, fits=False)
    assert (plan.placement, plan.resident_bytes) == (PLANE_PLACEMENT, GiB // 2)
    assert plan.model_weight_bytes == 3 * GiB // 2, "the plane's weight sets, whole"


def test_the_view_reads_no_driver() -> None:
    lanes = LaneSet.from_envelope("0,1", worker_pid=os.getpid())
    lanes.lanes[1].row("p").model_bearing = True
    lanes.lanes[1].touch("p")
    view = MemoryManager(lanes, lambda *_: None, lambda *_: False).view()
    assert view.devices == {}, "no admission has read a device yet"
    assert [(t.tenant, t.gpus, t.active, t.plane) for t in view.tenants] == [
        ("p", (1,), False, None)
    ]


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    measured = accel.DeviceMemory("measured", TOTAL, TOTAL)
    driverless(monkeypatch)
    monkeypatch.setattr(accel, "device_memory", lambda entry, kind: measured)
    with tempfile.TemporaryDirectory(prefix="cz-mem.", dir="/tmp") as root:
        made = Machine(Path(root), VIRTUAL, "boot-one")
        try:
            yield made
        finally:
            made.close()


def long_form(machine: Machine, monkeypatch: pytest.MonkeyPatch) -> tuple[str, list[str]]:
    """1513's shape: a warm Qwen executor on every GPU, then H3 as one 4-rank group beside
    them. Returns the group's placement and the Qwen placements, lane order."""
    machine.activate_warm(monkeypatch)
    machine.root("A")
    references = [machine.child("A", "qwen") for _ in range(4)]
    machine.tick()
    machine.warm()
    for reference in references:
        machine.finish(reference)
    shot = machine.child("A", "h3")
    machine.tick()
    machine.warm()
    worker = machine.worker
    lanes = {worker._lane_of(p).lane_id: p for p in worker.hosted}
    assert machine.granted(shot) == [0, 1, 2, 3]
    return lanes["lane-0+1+2+3"], [lanes[f"lane-{o}"] for o in range(4)]


@needs_executor
def test_a_dispatched_attempt_that_faults_leaves_nothing_on_the_card(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception after dispatch poisons the executor it may have left mid-command; its
    process is reclaimed at once, not whenever its placement is next granted."""
    h3, _qwen = long_form(machine, monkeypatch)
    worker = machine.worker
    hosted, group = worker.hosted[h3], worker._lane_of(h3)
    faulted = hosted.supervision.current
    assert faulted is not None
    row = group.row(h3)
    attempt = AttemptRecord("shot", 1, b"", {}, placement_id=h3)
    attempt.executor, attempt.executions, attempt.lane_id = faulted, 1, group.lane_id
    attempt.slot = AttemptSlot(
        supervision=hosted.supervision,
        ledger=row.ledger,
        chooser=row.chooser,
        bindings=hosted.bindings,
        failed_bindings={},
        lane_id=group.lane_id,
    )
    attempt.state = "closed"
    worker._attempt_faulted(attempt, RuntimeError("the post phase broke"))
    assert hosted.supervision.current is None
    assert not os.path.exists(f"/proc/{faulted.pid}"), "killed and reaped"


@needs_executor
def test_an_executor_that_exits_after_its_withdrawal_is_reaped(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    h3, _qwen = long_form(machine, monkeypatch)
    worker = machine.worker
    hosted = worker.hosted[h3]
    executor = hosted.supervision.current
    assert executor is not None
    hosted.placement.serving = pb.ServingState.SERVING_STATE_OFFLINE
    os.kill(executor.pid, signal.SIGKILL)
    bound = time.monotonic() + 60  # a hang bound on a loaded box, not a budget
    while hosted.supervision.current is not None or os.path.exists(f"/proc/{executor.pid}"):
        assert time.monotonic() < bound, [e.step for e in worker.activity][-5:]
        time.sleep(0.05)


_EXITS_IN_SCOPE = """
import sys
from cozy_runtime.internal import proctree
proctree.join_executor_cgroup(proctree.CgroupScope(sys.argv[1], int(sys.argv[2])))
"""


@needs_executor
def test_a_follower_that_exited_first_is_reaped_with_its_group(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a cgroup-scoped host, a rank that exited before its leader was killed is a zombie
    `cgroup.procs` does not list; the reclaim reaps it from its cgroup, not just the listed."""
    h3, _qwen = long_form(machine, monkeypatch)
    hosted = machine.worker.hosted[h3]
    executor = hosted.supervision.current
    assert executor is not None
    scope = executor.scope
    if not isinstance(scope, proctree.CgroupScope):
        pytest.skip("this host scopes executors by uid, whose census already sees zombies")
    rank = subprocess.Popen(
        [sys.executable, "-c", _EXITS_IN_SCOPE, scope.relative_path, str(scope.inode)]
    )
    bound = time.monotonic() + 60  # a hang bound on a loaded box, not a budget
    while proctree.process_state(proctree.process_identity(rank.pid)) != "Z":
        assert time.monotonic() < bound
        time.sleep(0.01)
    hosted.supervision.retire_current(executor, "the group broke")
    assert not os.path.exists(f"/proc/{rank.pid}"), "the exited rank was reaped"
