"""The machine's memory manager over the real Worker (darkness runs 1513 and 1514).

A real `Worker` with four devices, real execution units and real executor processes, as
`test_gpu_scheduler` drives them. The only fake is the driver: each device's free bytes are
its total less what every live tenant holds there (`LaneRow.held`), so an eviction or a
reclaim moves them exactly as NVML would.
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

from cozy_runtime.internal import accel, proctree
from cozy_runtime.internal.worker.attempts import AttemptRecord, AttemptSlot
from cozy_runtime.internal.worker.memory import Watch
from cozy_runtime.protocol import worker_pb2 as pb
from test_end_to_end import NO_EXECUTOR
from test_gpu_scheduler import Machine

needs_executor = pytest.mark.skipif(bool(NO_EXECUTOR), reason=NO_EXECUTOR or "")

GiB = 1 << 30
TOTAL = 80 * GiB
SHAPE = ("segment", "frames=362")


def held_on(machine: Machine, ordinal: int) -> int:
    """What the live tenants of `ordinal` hold, as the driver would count it."""
    return sum(
        row.held.get(ordinal, 0)
        for lane in machine.worker.lanes.lanes
        if ordinal in lane.ordinals
        for row in lane.rows.values()
        if row.supervision is not None and row.supervision.current is not None
    )


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> Iterator[Machine]:
    box: list[Machine] = []

    def device_memory(entry: str, kind: str) -> accel.DeviceMemory:
        used = held_on(box[0], int(entry)) if box else 0
        return accel.DeviceMemory("measured", TOTAL - used, TOTAL)

    monkeypatch.setattr(accel, "device_memory", device_memory)
    with tempfile.TemporaryDirectory(prefix="cz-mem.", dir="/tmp") as root:
        made = Machine(Path(root), "0,1,2,3", "boot-one")
        box.append(made)
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
    qwen = [lanes[f"lane-{o}"] for o in range(4)]
    h3 = lanes["lane-0+1+2+3"]
    for placement in qwen:
        lane = worker._lane_of(placement)
        lane.row(placement).held = {lane.ordinals[0]: 33 * GiB}
    group = worker._lane_of(h3)
    group.row(h3).held = dict.fromkeys(range(4), 20 * GiB)
    assert machine.granted(shot) == [0, 1, 2, 3]
    return h3, qwen


def vacated(machine: Machine) -> list[str]:
    return [e.step.split("'")[1] for e in machine.worker.activity if " vacated for " in e.step]


@needs_executor
def test_a_group_evicts_the_idle_tenant_of_every_device_it_needs(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1513: the group's measured need does not fit beside the idle Qwen on ANY of its four
    cards, so all four go, and only they: every process stays, warm."""
    h3, qwen = long_form(machine, monkeypatch)
    worker = machine.worker
    before = machine.executors()
    group = worker._lane_of(h3)
    group.row(h3).peak[SHAPE] = dict.fromkeys(range(4), 61 * GiB)
    assert worker.memory.admit(group, h3, SHAPE, "segment 2")
    assert sorted(vacated(machine)) == sorted(qwen)
    assert all(accel.device_memory(str(o), "cuda").free_bytes >= 41 * GiB for o in range(4))
    assert machine.executors() == before, "vacated, not killed"
    assert group.row(h3).held == dict.fromkeys(range(4), 20 * GiB), "the requester kept its own"


@needs_executor
def test_a_need_that_fits_moves_nothing_and_lru_order_is_kept(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything that fits stays hot. A shape never measured has the whole of its devices:
    idle tenants leave least recently used first, and the requester is never one."""
    h3, qwen = long_form(machine, monkeypatch)
    worker = machine.worker
    group = worker._lane_of(h3)
    group.row(h3).peak[SHAPE] = dict.fromkeys(range(4), 40 * GiB)
    assert worker.memory.admit(group, h3, SHAPE, "fits")
    assert vacated(machine) == []
    for placement in (qwen[2], qwen[0], qwen[3], qwen[1]):
        worker._lane_of(placement).touch(placement)
    assert worker.memory.admit(group, h3, ("segment", "frames=500"), "unmeasured")
    assert vacated(machine) == [qwen[2], qwen[0], qwen[3], qwen[1]]
    assert h3 not in vacated(machine)


@needs_executor
def test_a_dead_groups_memory_is_reclaimed_before_its_devices_are_admitted_again(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1514: a broken group keeps nothing. The next tenant on its card kills and reaps it
    before running, so its measured need finds the room instead of an OOM."""
    h3, qwen = long_form(machine, monkeypatch)
    worker = machine.worker
    group = worker._lane_of(h3)
    dead = worker.hosted[h3].supervision.current
    assert dead is not None and dead.alive()
    group.row(h3).held = dict.fromkeys(range(4), 62 * GiB)
    dead.poisoned = "failed/group_broken"
    lane = worker._lane_of(qwen[0])
    lane.row(qwen[0]).peak[SHAPE] = {0: 40 * GiB}
    assert worker.memory.admit(lane, qwen[0], SHAPE, "reference")
    assert worker.hosted[h3].supervision.current is None
    assert not dead.alive()
    assert not os.path.exists(f"/proc/{dead.pid}"), "killed AND reaped"
    assert vacated(machine) == [], "the dead tenant was ended, nobody idle was evicted"
    ended = [e.step for e in worker.activity if f"{h3!r}: reclaimed executor epoch" in e.step]
    assert len(ended) == 1, [e.step for e in worker.activity]


@needs_executor
def test_true_overload_stages_and_crashes_nothing(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A need no eviction can meet evicts every idle tenant, says so in numbers, and
    returns: the attempt runs staged (the plan's rule), nothing is killed."""
    h3, qwen = long_form(machine, monkeypatch)
    worker = machine.worker
    before = machine.executors()
    group = worker._lane_of(h3)
    group.row(h3).peak[SHAPE] = dict.fromkeys(range(4), 200 * GiB)
    assert not worker.memory.admit(group, h3, SHAPE, "segment 3")
    assert sorted(vacated(machine)) == sorted(qwen)
    assert machine.executors() == before
    short = [e.step for e in worker.activity if "no idle tenant is left" in e.step]
    assert short and "needs 193273528320 B free" in short[-1], short


@needs_executor
def test_a_calls_peak_is_what_the_driver_saw_and_a_device_failure_forgets_it(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    h3, _qwen = long_form(machine, monkeypatch)
    worker = machine.worker
    group = worker._lane_of(h3)
    row = group.row(h3)
    free = TOTAL - 53 * GiB
    seen = Watch(
        before=dict.fromkeys(range(4), free),
        low={0: free - 30 * GiB, 1: free - 41 * GiB, 2: free - 30 * GiB, 3: free - 30 * GiB},
        after=dict.fromkeys(range(4), free - GiB),
    )
    worker.memory.settle(group, h3, SHAPE, seen, ok=True, forget=False)
    assert row.peak[SHAPE] == {0: 50 * GiB, 1: 61 * GiB, 2: 50 * GiB, 3: 50 * GiB}
    assert row.held == dict.fromkeys(range(4), 21 * GiB), "what the call left stays counted"
    assert worker.memory.need(group, h3, SHAPE) == {
        0: 29 * GiB,
        1: 40 * GiB,
        2: 29 * GiB,
        3: 29 * GiB,
    }
    worker.memory.settle(group, h3, SHAPE, seen, ok=False, forget=True)
    assert SHAPE not in row.peak and worker.memory.need(group, h3, SHAPE) is None


def test_an_unmeasured_shape_is_bounded_only_by_a_larger_measured_one() -> None:
    from cozy_runtime.internal.worker.memory import bound

    peaks = {("s", "frames=242,width=512"): {0: 30}, ("s", "frames=362,width=768"): {0: 50}}
    assert bound(("s", "frames=121,width=512"), peaks) == {0: 30}
    assert bound(("s", "frames=300,width=512"), peaks) == {0: 50}
    assert bound(("s", "frames=500,width=512"), peaks) is None, "nothing measured covers it"
    assert bound(("s", "frames=121"), peaks) is None, "other axes describe another shape"
    assert bound(("t", "frames=121,width=512"), peaks) is None, "another entrypoint"


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


@needs_executor
def test_a_group_that_can_name_its_ranks_gives_back_only_the_cards_a_call_needs(
    machine: Machine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reference on card 3 needs H3's bytes there, not on cards 0-2: an executor that
    says it can vacate ranks gives back GPU 3 alone, and restores it before it next runs."""
    h3, qwen = long_form(machine, monkeypatch)
    worker = machine.worker
    group = worker._lane_of(h3)
    executor = worker.hosted[h3].supervision.current
    assert executor is not None
    executor.hello["memory"] = ["vacate_ranks"]
    lane = worker._lane_of(qwen[3])
    lane.row(qwen[3]).peak[SHAPE] = {3: 70 * GiB}
    assert worker.memory.admit(lane, qwen[3], SHAPE, "a reference")
    steps = [e.step for e in worker.activity if " vacated " in e.step]
    assert len(steps) == 1 and f"{h3!r}" in steps[0] and "vacated GPU 3 for " in steps[0], steps
    assert group.row(h3).held == dict.fromkeys(range(3), 20 * GiB)
    assert group.row(h3).emptied == {3}


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
