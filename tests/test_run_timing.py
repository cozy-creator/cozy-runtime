"""Real private-loop waits and retained aggregate measurements stay distinct from wall time."""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from pathlib import Path

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author._activity import Activity, EventLoop
from cozy_runtime.internal.worker.run_timing import Timing
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_calls import request, running
from test_machine_execution import complete, offer


@pytest.mark.parametrize("threaded", [False, True])
def test_private_loop_excludes_waits_but_preserves_parallel_author_work(threaded: bool) -> None:
    observed: list[tuple[float, bool, bool]] = []
    activity = Activity(
        lambda _seq, active, known: observed.append((time.monotonic(), active, known))
    )
    loop = EventLoop(activity)
    busy = threading.Event()

    def work() -> None:
        busy.set()
        time.sleep(0.15)  # synchronous author work, in the executor or its managed pool

    async def invoke() -> None:
        await asyncio.sleep(0.2)  # actual event-loop wait for queued children
        if threaded:
            await asyncio.gather(asyncio.to_thread(work), asyncio.sleep(0.2))
        else:
            waiter = asyncio.create_task(asyncio.sleep(0.2))
            work()  # one coroutine runs while another remains blocked
            await waiter

    activity.change(1)
    try:
        loop.run_until_complete(invoke())
    finally:
        loop.close()
        activity.change(-1)
    assert busy.is_set()
    assert all(known for _, _, known in observed)
    measured = sum(b[0] - a[0] for a, b in itertools.pairwise(observed) if a[1])
    assert 0.14 <= measured < 0.3
    assert observed[-1][0] - observed[0][0] >= 0.4


def test_untracked_thread_makes_cooperative_measurement_unknown() -> None:
    observed: list[bool] = []
    activity = Activity(lambda _seq, _active, known: observed.append(known))
    loop = EventLoop(activity)
    release = threading.Event()
    thread = threading.Thread(target=release.wait)
    activity.change(1)
    thread.start()
    try:
        loop.run_until_complete(asyncio.sleep(0.01))
    finally:
        release.set()
        thread.join()
        loop.close()
        activity.change(-1)
    assert observed[-1] is False


def test_run_timing_unions_nested_work_and_keeps_one_cumulative_snapshot(tmp_path: Path) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    executions.submit(
        "owner",
        "parent",
        b"p" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    child = Calls(executions.workspace).accept("owner", request(running(executions))).child_request
    clock = [0.0]
    timing = Timing(executions, monotonic=lambda: clock[0])

    def active(name: str, sequence: int, state: bool) -> None:
        timing.observe("owner", name, 1, {"sequence": sequence, "active": state, "known": True})

    timing.begin("owner", "parent", 1)
    active("parent", 1, True)
    clock[0] = 2
    active("parent", 2, False)
    clock[0] = 102  # GPU queue while every executor is blocked
    timing.begin("owner", child, 1)
    active(child, 1, True)
    clock[0] = 104
    active("parent", 3, True)  # parent CPU overlaps child; the union counts once
    clock[0] = 107
    active(child, 2, False)
    timing.end("owner", child, 1)
    clock[0] = 110
    for _ in range(300):
        timing.snapshot("owner", "parent", 1)
    active("parent", 4, False)
    timing.end("owner", "parent", 1)
    with executions.workspace.locked() as db:
        events = db.execute("SELECT body FROM execution_events WHERE kind='run.timing'").fetchall()
    assert [canonical_json.decode(row[0]) for row in events] == [
        {"attempt": 1, "terminal": True, "execution_ms": 10000}
    ]
    assert timing.runs == timing.members == {}


@pytest.mark.parametrize("problem", ["legacy", "missing", "recovery"])
def test_incomplete_executor_observations_never_become_execution_time(
    tmp_path: Path, problem: str
) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    executions.submit(
        "owner",
        "parent",
        b"p" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    Calls(executions.workspace)
    clock = [0.0]
    timing = Timing(executions, monotonic=lambda: clock[0])
    timing.begin("owner", "parent", 1)
    if problem == "recovery":
        timing = Timing(executions, monotonic=lambda: clock[0])
        timing.begin("owner", "parent", 1)
    if problem != "legacy":
        timing.observe("owner", "parent", 1, {"sequence": 1, "active": True, "known": True})
        clock[0] = 100
        timing.observe("owner", "parent", 1, {"sequence": 3, "active": False, "known": True})
    timing.end("owner", "parent", 1)
    with executions.workspace.locked() as db:
        event = canonical_json.decode(
            db.execute("SELECT body FROM execution_events WHERE kind='run.timing'").fetchone()[0]
        )
    assert event == {"attempt": 1, "terminal": True}


def test_root_ending_with_an_open_child_retires_members_and_reports_unknown(tmp_path: Path) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    executions.submit(
        "owner",
        "parent",
        b"p" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    child = Calls(executions.workspace).accept("owner", request(running(executions))).child_request
    timing = Timing(executions)
    for name in ("parent", child):
        timing.begin("owner", name, 1)
        timing.observe("owner", name, 1, {"sequence": 1, "active": True, "known": True})
    timing.observe("owner", "parent", 1, {"sequence": 2, "active": False, "known": True})
    timing.end("owner", "parent", 1)
    timing.observe("owner", child, 1, {"sequence": 2, "active": False, "known": True})
    timing.end("owner", child, 1)
    assert timing.runs == timing.members == {}
    with executions.workspace.locked() as db:
        event = canonical_json.decode(
            db.execute("SELECT body FROM execution_events WHERE kind='run.timing'").fetchone()[0]
        )
    assert event == {"attempt": 1, "terminal": True}


def test_retry_retains_each_attempt_without_counting_paused_gap(tmp_path: Path) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    executions.submit(
        "owner",
        "parent",
        b"p" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    Calls(executions.workspace)
    clock = [0.0]
    timing = Timing(executions, monotonic=lambda: clock[0])
    for ordinal, start, end in ((1, 0, 2), (2, 1000, 1003)):
        clock[0] = start
        timing.begin("owner", "parent", ordinal)
        timing.observe("owner", "parent", ordinal, {"sequence": 1, "active": True, "known": True})
        clock[0] = end
        timing.observe("owner", "parent", ordinal, {"sequence": 2, "active": False, "known": True})
        timing.end("owner", "parent", ordinal)
        complete(executions, "parent", status=pb.OUTCOME_STATUS_FAILED)
        if ordinal == 1:
            executions.control("owner", "parent", "resume", 1, "resume")
    with executions.workspace.locked() as db:
        events = db.execute(
            "SELECT body FROM execution_events WHERE kind='run.timing' ORDER BY ordinal"
        ).fetchall()
    assert [canonical_json.decode(row[0]) for row in events] == [
        {"attempt": 1, "terminal": True, "execution_ms": 2000},
        {"attempt": 2, "terminal": True, "execution_ms": 3000},
    ]
