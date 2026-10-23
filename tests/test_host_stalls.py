"""A step that waits on the host names the code it waited in (cr-211): the real sampler thread
over this test's own thread, which runs Python while its "GPU" has nothing queued."""

from __future__ import annotations

import threading
import time

from cozy_runtime.author._stalls import FLOOR_MS, PERIOD, Watch


def _plan() -> int:
    """Host work with no GPU queued: what split's quadratic planner was."""
    return sum(range(10_000))


def _held(watch: Watch, samples: int) -> None:
    """Run `_plan` until the watch has sampled at least `samples` times (a loaded machine
    wakes the sampler late; the count, not the clock, is the measurement)."""
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        _plan()
        with watch.lock:
            if sum(watch.counts.values()) >= samples:
                return


def test_a_host_stall_is_named_by_the_code_it_ran() -> None:
    watch = Watch(threading.get_ident(), idle=lambda: True, alive=lambda: True)
    try:
        _held(watch, int(2 * FLOOR_MS / 1000 / PERIOD))
    finally:
        watch.stop()
    taken = watch.take()
    assert [code for code in taken if code.endswith("._plan")], taken
    assert all(ms >= FLOOR_MS for ms in taken.values())
    assert watch.take() == {}


def test_a_busy_gpu_is_no_stall() -> None:
    watch = Watch(threading.get_ident(), idle=lambda: False, alive=lambda: True)
    try:
        end = time.monotonic() + 15 * PERIOD
        while time.monotonic() < end:
            _plan()
    finally:
        watch.stop()
    assert watch.take() == {}


def test_the_sampler_ends_with_its_attempt() -> None:
    alive = threading.Event()
    alive.set()
    watch = Watch(threading.get_ident(), idle=lambda: True, alive=alive.is_set)
    alive.clear()
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and any(
        t.name == "cozy-host-stalls" and t.is_alive() for t in threading.enumerate()
    ):
        time.sleep(PERIOD)
    watch.stop()
    assert not any(t.name == "cozy-host-stalls" and t.is_alive() for t in threading.enumerate())
