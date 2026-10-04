"""Real private-loop waits and retained aggregate measurements stay distinct from wall time."""

from __future__ import annotations

import asyncio
import functools
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

from cozy_runtime.author import ThreadPoolExecutor as SDKThreadPoolExecutor
from cozy_runtime.author._activity import Activity, EventLoop, observing


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
