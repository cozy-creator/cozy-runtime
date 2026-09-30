"""Real private-loop waits and retained aggregate measurements stay distinct from wall time."""

from __future__ import annotations

import asyncio
import functools
import importlib.metadata
import itertools
import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author import ThreadPoolExecutor as SDKThreadPoolExecutor
from cozy_runtime.author._activity import Activity, EventLoop, observing
from cozy_runtime.internal.worker.run_timing import Timing
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_calls import request, running
from test_machine_execution import complete, offer


def _executor_process[**P](case: Callable[P, None]) -> Callable[P, None]:
    """Give process-lifetime thread observations the same isolation as an executor.

    Other tests run Worker servers and author calls in pytest's process. A server
    started after an earlier author call correctly makes that shared clock unknown.
    Each scenario gets a fresh process; its invocations still share real observer state.
    """

    @functools.wraps(case)
    def run(*args: P.args, **kwargs: P.kwargs) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import json, pathlib, runpy, sys; "
                "sys.path.insert(0, str(pathlib.Path(sys.argv[1]).parent)); "
                "case = runpy.run_path(sys.argv[1])[sys.argv[2]].__wrapped__; "
                "args, kwargs = json.loads(sys.argv[3]); case(*args, **kwargs)",
                str(Path(__file__).resolve()),
                case.__name__,
                json.dumps((args, kwargs)),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    return run


@pytest.mark.parametrize("threaded", [False, True])
@_executor_process
def test_private_loop_excludes_waits_but_preserves_parallel_author_work(threaded: bool) -> None:
    observed: list[tuple[float, bool, bool]] = []
    activity = Activity(
        lambda _seq, active, known, _finished: observed.append((time.monotonic(), active, known))
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


@_executor_process
def test_untracked_thread_makes_cooperative_measurement_unknown() -> None:
    observed: list[bool] = []
    activity = Activity(lambda _seq, _active, known, _finished: observed.append(known))
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


@_executor_process
def test_canceled_pool_submission_cannot_hide_an_untracked_thread() -> None:
    activity = Activity(None)
    loop = EventLoop(activity)
    release = threading.Event()

    async def invoke() -> None:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = loop.run_in_executor(pool, release.wait)
            canceled = loop.run_in_executor(pool, lambda: None)
            canceled.cancel()
            release.set()
            await first
        assert not activity.submitted
        raw_release = threading.Event()
        raw = threading.Thread(target=raw_release.wait)
        raw.start()
        try:
            await asyncio.sleep(0.01)
            assert not activity.known
        finally:
            raw_release.set()
            raw.join()

    activity.change(1)
    try:
        loop.run_until_complete(invoke())
    finally:
        loop.close()
        activity.change(-1)


@_executor_process
def test_sdk_pool_preserves_initializer_cancellation_and_reuse_without_idle_execution() -> None:
    local = threading.local()
    initialized: list[int] = []

    def initialize(value: int) -> None:
        local.value = value
        initialized.append(value)

    with SDKThreadPoolExecutor(max_workers=1, initializer=initialize, initargs=(37,)) as pool:
        assert pool.submit(lambda: local.value).result() == 37  # outside an invocation
        with observing(None) as initial:
            release = threading.Event()
            first = pool.submit(release.wait)
            canceled = pool.submit(lambda: 99)
            assert canceled.cancel()
            release.set()
            assert first.result()
            assert not initial.submitted
        for _ in range(2):
            observed: list[tuple[float, bool, bool, bool]] = []

            def collect(
                _seq: int,
                active: bool,
                known: bool,
                done: bool,
                rows: list[tuple[float, bool, bool, bool]] = observed,
            ) -> None:
                rows.append((time.monotonic(), active, known, done))

            with observing(collect) as activity:
                loop = EventLoop(activity)

                def work() -> int:
                    time.sleep(0.04)
                    return int(local.value)

                async def invoke(loop: EventLoop = loop) -> None:
                    assert await asyncio.wrap_future(pool.submit(work)) == 37
                    await asyncio.sleep(0.12)  # the persistent pool is idle here
                    assert await loop.run_in_executor(pool, work) == 37

                try:
                    loop.run_until_complete(invoke())
                finally:
                    loop.close()
            assert all(known for _, _, known, _ in observed)
            assert observed[-1][1:] == (False, True, True)
            measured = sum(b[0] - a[0] for a, b in itertools.pairwise(observed) if a[1])
            assert 0.07 <= measured < 0.16, (measured, observed)
            assert pool.submit(lambda: local.value).result() == 37
        assert initialized == [37]


@pytest.mark.parametrize("escaped", [False, True])
@_executor_process
def test_joined_helpers_are_covered_by_their_owned_callback_but_escaped_work_is_unknown(
    escaped: bool,
) -> None:
    raw_release = threading.Event()
    raw_threads: list[threading.Thread] = []
    with SDKThreadPoolExecutor(max_workers=1) as pool:
        with observing(None) as activity:
            loop = EventLoop(activity)

            def work() -> None:
                raw = threading.Thread(target=raw_release.wait)
                raw_threads.append(raw)
                raw.start()
                time.sleep(0.04)
                if not escaped:
                    raw_release.set()
                    raw.join()

            async def invoke() -> None:
                await asyncio.wrap_future(pool.submit(work))
                await asyncio.sleep(0.02)

            try:
                loop.run_until_complete(invoke())
            finally:
                raw_release.set()
                for raw in raw_threads:
                    raw.join()
                loop.close()
        assert activity.known is (not escaped)


@_executor_process
def test_an_escaped_thread_keeps_the_next_invocation_unknown() -> None:
    release = threading.Event()
    raw = threading.Thread(target=release.wait)
    try:
        with observing(None) as first:
            raw.start()
        assert not first.known
        with observing(None) as following:
            pass
        assert not following.known
    finally:
        release.set()
        raw.join()


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
    executions.submit(
        "owner",
        child,
        b"c" * 32,
        offer(child),
        expected_execution_workspace_id=executions.workspace_id,
    )
    clock = [0.0]
    timing = Timing(executions, monotonic=lambda: clock[0])

    def active(name: str, sequence: int, state: bool, *, finished: bool = False) -> None:
        timing.observe(
            "owner",
            name,
            1,
            {"sequence": sequence, "active": state, "known": True, "finished": finished},
        )

    timing.begin("owner", "parent", 1)
    active("parent", 1, True)
    clock[0] = 2
    active("parent", 2, False)
    clock[0] = 102  # GPU queue while every executor is blocked
    timing.sample()
    timing.begin("owner", child, 1)
    active(child, 1, True)
    clock[0] = 104
    active("parent", 3, True)  # parent CPU overlaps child; the union counts once
    clock[0] = 107
    active(child, 2, False, finished=True)
    timing.end("owner", child, 1)
    clock[0] = 110
    for _ in range(300):
        timing.sample()
    active("parent", 4, False, finished=True)
    timing.end("owner", "parent", 1)
    with executions.workspace.locked() as db:
        events = db.execute("SELECT body FROM execution_events WHERE kind='run.timing'").fetchall()
    assert [canonical_json.decode(row[0]) for row in events] == [
        {"attempt": 1, "terminal": True, "execution_ms": 10000}
    ]
    assert not timing.runs and not timing.members


@pytest.mark.parametrize("problem", ["legacy", "missing", "recovery", "died_while_blocked"])
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
        timing.observe(
            "owner",
            "parent",
            1,
            {
                "sequence": 2 if problem == "died_while_blocked" else 3,
                "active": False,
                "known": True,
                "finished": problem != "died_while_blocked",
            },
        )
    timing.sample()
    timing.end("owner", "parent", 1)
    with executions.workspace.locked() as db:
        event = canonical_json.decode(
            db.execute("SELECT body FROM execution_events WHERE kind='run.timing'").fetchone()[0]
        )
    assert event == {"attempt": 1, "terminal": True}


@pytest.mark.parametrize("refuse_record", [False, True])
def test_root_ending_with_an_open_child_retires_members_and_reports_unknown(
    tmp_path: Path, refuse_record: bool
) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    executions.submit(
        "owner",
        "parent",
        b"p" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    child = Calls(executions.workspace).accept("owner", request(running(executions))).child_request
    executions.submit(
        "owner",
        child,
        b"c" * 32,
        offer(child),
        expected_execution_workspace_id=executions.workspace_id,
    )
    timing = Timing(executions)
    for name in ("parent", child):
        timing.begin("owner", name, 1)
        timing.observe("owner", name, 1, {"sequence": 1, "active": True, "known": True})
    timing.observe(
        "owner", "parent", 1, {"sequence": 2, "active": False, "known": True, "finished": True}
    )
    if refuse_record:
        with executions.workspace.locked() as db:
            db.execute(
                "CREATE TRIGGER refuse_timing BEFORE INSERT ON execution_events "
                "WHEN NEW.kind='run.timing' BEGIN SELECT RAISE(ABORT,'unavailable'); END"
            )
    timing.end("owner", "parent", 1)
    timing.observe("owner", child, 1, {"sequence": 2, "active": False, "known": True})
    timing.end("owner", child, 1)
    assert not timing.runs and not timing.members
    if refuse_record:
        return
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
        timing.observe(
            "owner",
            "parent",
            ordinal,
            {"sequence": 2, "active": False, "known": True, "finished": True},
        )
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


def test_stale_root_or_descendant_cannot_join_a_retry_clock(tmp_path: Path) -> None:
    executions = Executions(Workspace(tmp_path / "store"))
    executions.submit(
        "owner",
        "parent",
        b"p" * 32,
        offer("parent"),
        expected_execution_workspace_id=executions.workspace_id,
    )
    child = Calls(executions.workspace).accept("owner", request(running(executions))).child_request
    executions.submit(
        "owner",
        child,
        b"c" * 32,
        offer(child),
        expected_execution_workspace_id=executions.workspace_id,
    )
    complete(executions, "parent", status=pb.OUTCOME_STATUS_FAILED)
    executions.control("owner", "parent", "resume", 1, "resume")
    timing = Timing(executions)
    timing.begin("owner", "parent", 1)
    assert not timing.runs and not timing.members
    timing.begin("owner", "parent", 2)
    timing.begin("owner", child, 1)
    assert set(timing.members) == {("owner", "parent", 2)}


@pytest.mark.parametrize(
    ("runtime", "mode"),
    [("", "app"), ("", "script"), ("", "own-loop"), ("0.18.85", "app")],
    ids=["current-sdk", "pep723-script", "script-own-loop", "older-sdk"],
)
def test_real_worker_nested_child_wait_is_not_run_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: str, mode: str
) -> None:
    import test_job_preparation_isolation as fixture
    from test_end_to_end import NO_EXECUTOR
    from test_machine_partial_work import machine

    if NO_EXECUTOR:
        pytest.skip(NO_EXECUTOR)
    if runtime:
        monkeypatch.setattr(fixture, "package", functools.partial(fixture.package, runtime=runtime))
    entered, gate = tmp_path / "entered", tmp_path / "gate"
    pool = (
        "pool = None"
        if runtime
        else (
            "from cozy_runtime.author import ThreadPoolExecutor\n"
            "pool = ThreadPoolExecutor(max_workers=1)"
        )
    )
    source = f"""import asyncio, time, msgspec, importlib.metadata, json, sys
from pathlib import Path
from cozy_runtime.author import App, Context, Telemetry, invocable
from cozy_runtime.author._invoke import Invocation
{pool}
app=App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int
def work(milliseconds):
    start, value = time.monotonic(), 3
    while (time.monotonic() - start) * 1000 < milliseconds:
        value = (value * 17 + 11) % 1000003
    return value
@invocable
async def leaf(ctx: Context, tel: Telemetry, *, n: int) -> Result:
    background = pool.submit(work, 100) if pool is not None else None
    metadata = Path({str(entered)!r}).with_suffix(".writing")
    metadata.write_text(json.dumps({{
        "runtime": importlib.metadata.version("cozy-runtime"),
        "activity": "activity" in Invocation.__dataclass_fields__,
        "loaded": sys.modules[Invocation.__module__].__file__,
        "installed": str(importlib.metadata.distribution("cozy-runtime").locate_file(
            "cozy_runtime/author/_invoke.py")),
    }}))
    metadata.replace({str(entered)!r})
    tel.progress(0.25, stage="Ready")
    while not Path({str(gate)!r}).exists():
        await asyncio.sleep(0.01)
    value = work(150)
    if background is not None:
        value += background.result()
    return Result(value)
app.job(leaf)
@app.job
async def nested(ctx: Context, payload: Request) -> Result:
    return await leaf(n=3)
"""
    if mode != "app":
        source = source[: source.index("app.job(leaf)")].replace("app=App()\n", "")
        source = '# /// script\n# requires-python = ">=3.12"\n# dependencies = []\n# ///\n' + source
        source += (
            "async def main() -> int:\n    value = (await leaf(n=3)).value\n"
            "    await asyncio.to_thread(work, 20)\n    return value\n"
            if mode == "script"
            else "def main() -> int:\n    async def nested():\n"
            "        return (await leaf(n=3)).value\n    return asyncio.run(nested())\n"
        )
        source += (
            "\nfrom cozy_runtime.author import script_app\n"
            'app = script_app("prepare_nested")\napp.job(leaf)\n'
        )
    with machine(monkeypatch, source) as pod:
        pod.start(("leaf",), entrypoint="nested" if mode == "app" else "main")
        pod.submit("timed")
        pod.wait("the nested child to await its gate", entered.exists, 120)
        sdk = json.loads(entered.read_text())
        assert sdk["runtime"] == (runtime or importlib.metadata.version("cozy-runtime")), sdk
        assert Path(sdk["loaded"]).resolve() == Path(sdk["installed"]).resolve(), sdk
        assert sdk["activity"] is (not runtime), sdk
        time.sleep(0.6)  # controlled observation interval, not a production timeout
        gate.touch()
        outcome = pod.outcome("timed")
        assert outcome["status"] == pb.OUTCOME_STATUS_SUCCEEDED, outcome
        rows = pod.rows(
            "SELECT body FROM execution_events WHERE request='timed' AND kind='run.timing'"
        )
        assert len(rows) == 1
        raw = rows[0]["body"]
        assert isinstance(raw, bytes)
        measured = canonical_json.decode(raw)
        if runtime or mode == "own-loop":
            assert measured == {"attempt": 1, "terminal": True}
            return
        assert measured["terminal"] is True and measured["execution_ms"] >= 100, measured
        calls = pod.rows(
            "SELECT body FROM execution_events WHERE request='timed' AND kind='call.phase'"
        )
        phases = []
        for row in calls:
            raw = row["body"]
            assert isinstance(raw, bytes)
            phases.append(canonical_json.decode(raw))
        terminal = next(phase for phase in reversed(phases) if phase["phase"] == "terminal")
        # The existing child phase is its author envelope; the run measurement
        # must exclude the real blocked interval while retaining its CPU work.
        assert terminal["execution_ms"] - measured["execution_ms"] >= 400, (terminal, measured)
        assert not pod.rows("SELECT body FROM execution_events WHERE kind='execution_activity'")


def test_real_worker_reports_execution_between_model_progress_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_end_to_end import NO_EXECUTOR
    from test_machine_partial_work import machine

    if NO_EXECUTOR:
        pytest.skip(NO_EXECUTOR)
    entered, gate = tmp_path / "entered", tmp_path / "gate"
    source = f"""import msgspec
from pathlib import Path
from cozy_runtime.author import App, Context
app = App()
class Request(msgspec.Struct):
    pass
class Result(msgspec.Struct):
    value: int
@app.job
def nested(ctx: Context, payload: Request) -> Result:
    Path({str(entered)!r}).touch()
    value = 7
    while not Path({str(gate)!r}).exists():
        value = (value * 17 + 11) % 1000003
    return Result(value)
"""
    with machine(monkeypatch, source) as pod:
        pod.start(())
        submission = pod.submit("ticks")
        pod.wait("the smooth author work to start", entered.exists, 120)
        query = pb.MachineExecutionQuery(
            claim=pod.claim,
            request_id="ticks",
            expected_execution_workspace_id=submission.expected_execution_workspace_id,
        )
        ticks: list[tuple[int, float]] = []
        cursor = 0
        deadline = time.monotonic() + 12
        try:
            while len(ticks) < 2 and time.monotonic() < deadline:
                page = pod.client.ListMachineExecutionEvents(
                    pb.MachineExecutionEventsQuery(
                        execution=query, after=cursor, limit=128, wait=True
                    ),
                    timeout=max(deadline - time.monotonic(), 0.01),
                )
                cursor = page.next_after
                for event in page.events:
                    if event.kind != "run.timing":
                        continue
                    body = canonical_json.decode(event.body_canonical_bytes)
                    elapsed = body.get("execution_ms")
                    if isinstance(elapsed, (int, float)) and elapsed >= 1000:
                        ticks.append((event.at_ms, float(elapsed)))
                assert pod.state("ticks") == "running"
        finally:
            gate.touch()
        assert len(ticks) >= 2, ticks
        assert 1500 <= ticks[1][0] - ticks[0][0] < 4000, ticks
        assert 1500 <= ticks[1][1] - ticks[0][1] < 4000, ticks
        assert not pod.rows(
            "SELECT body FROM execution_events WHERE request='ticks' AND kind='progress'"
        )
        assert pod.outcome("ticks")["status"] == pb.OUTCOME_STATUS_SUCCEEDED
        assert (
            len(
                pod.rows(
                    "SELECT body FROM execution_events WHERE request='ticks' AND kind='run.timing'"
                )
            )
            == 1
        )
