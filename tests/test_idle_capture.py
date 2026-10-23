"""Graphs record off the request path: the executor's idle holder and the graphs' idle queue,
on real threads (the captures themselves are GPU work: tests/test_graphs.py)."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from cozy_runtime import graphs
from cozy_runtime.internal import executor

#: how long a test waits on another thread before calling it stuck
WAIT_S = 10.0


class _Capture:
    """A queued capture with region boundaries: `graphs.IDLE.check` between them, as
    `_dispatch` checks before each region it warms or records."""

    def __init__(self, boundaries: int) -> None:
        self.boundaries = boundaries
        self.runs = 0
        self.done = threading.Event()
        self.started = threading.Event()
        self.go_on = threading.Event()
        self.go_on.set()

    def __call__(self) -> bool:
        self.runs += 1
        for boundary in range(self.boundaries):
            if boundary == 1:
                self.started.set()
                self.go_on.wait(WAIT_S)
            graphs.IDLE.check()
        self.done.set()
        return True


@pytest.fixture(autouse=True)
def alone() -> Iterator[None]:
    """The process's idle queue holds only this test's jobs (an earlier test's stay aside)."""
    with graphs.IDLE.lock:
        earlier = list(graphs.IDLE.jobs)
        graphs.IDLE.jobs.clear()
    yield
    with graphs.IDLE.lock:
        graphs.IDLE.jobs.extend(earlier)


def _serving() -> executor._Order:
    order = executor._Order()
    threading.Thread(
        target=executor._idle, args=(SimpleNamespace(poisoned=""), order), daemon=True
    ).start()
    return order


def _drained() -> None:
    assert not graphs.IDLE.jobs


def test_nothing_records_inside_a_request_and_the_idle_period_after_it_does() -> None:
    _drained()
    order = _serving()
    capture = _Capture(boundaries=3)
    ticket = order.ticket(alone=False)
    with order.turn(ticket):  # a request: its first call queues its capture
        graphs.IDLE.add(capture)
        assert not capture.done.wait(0.2) and capture.runs == 0
    assert capture.done.wait(WAIT_S) and capture.runs == 1
    _drained()


def test_a_capture_yields_at_its_next_boundary_when_a_request_arrives() -> None:
    _drained()
    order = _serving()
    ticket = order.ticket(alone=True)
    with order.turn(ticket):  # an earlier command, answered: an idle period follows
        capture = _Capture(boundaries=4)
        capture.go_on.clear()
        graphs.IDLE.add(capture)
    yields = graphs.IDLE.yields
    assert capture.started.wait(WAIT_S)  # the capture is between two boundaries
    arrived = order.ticket(alone=False)
    capture.go_on.set()
    began = time.perf_counter()
    with order.turn(arrived):  # its turn comes when the capture gives the order back
        waited = time.perf_counter() - began
        assert graphs.IDLE.yields == yields + 1 and not capture.done.is_set()
        assert list(graphs.IDLE.jobs) == [capture]  # first again at the next idle period
    assert waited < WAIT_S and graphs.IDLE.longest_ms > 0
    assert capture.done.wait(WAIT_S) and capture.runs == 2  # retried after the request
    _drained()


def test_a_job_not_ready_waits_for_the_next_period_in_order() -> None:
    idle, ran = graphs._Idle(), []

    def job(name: str, ready: list[bool]) -> object:
        def run() -> bool:
            ran.append(name)
            return ready.pop(0)

        return run

    idle.add(job("measured once", [False, True]))
    idle.add(job("measured", [True]))
    assert idle.run(lambda: False) and ran == ["measured once", "measured"]
    assert len(idle.jobs) == 1 and idle.periods == 1
    assert idle.run(lambda: False) and ran[2:] == ["measured once"] and not idle.jobs


def test_a_device_bound_signature_never_records() -> None:
    """Launch-bound means the device waited on the host inside the region: its device-busy
    time below its host submission. A cold first call (autotune, page-in) never counts."""
    timing = graphs._Timing()
    timing.calls, timing.first_ms, timing.host_ms = 3, 40.0, 40.0 + 2 * 0.5
    submit = timing.submit_ms()
    assert submit == 0.5
    assert graphs._device_bound(submit, busy=3.0).startswith("device-bound")
    assert graphs._device_bound(submit, busy=0.2) == ""
    assert graphs._Timing().submit_ms() is None  # one call: wait for another


def test_a_failing_job_goes_and_the_period_goes_on() -> None:
    """v16: one region's capture raised, which ended every idle period before the UNet's turn.
    A job's failure is counted and said; the next job still runs, and the failed one goes."""
    idle, ran = graphs._Idle(), []

    def failing() -> bool:
        ran.append("failing")
        raise RuntimeError("CachingHostAllocator: use_count")

    def next_one() -> bool:
        ran.append("next")
        return True

    idle.add(failing)
    idle.add(next_one)
    assert idle.run(lambda: False) and ran == ["failing", "next"]
    assert idle.failures == 1 and not idle.jobs


def test_a_device_fault_at_idle_poisons_the_executor() -> None:
    """The verifier's MED on v16's fix: only ordinary failures are swallowed. A sticky or
    device fault during an idle capture ends the period and poisons the executor, as one in a
    request does, and the idle thread stops."""
    torch = pytest.importorskip("torch")
    sticky = torch.AcceleratorError("CUDA error: an illegal memory access was encountered")
    sticky.error_code = 700  # torch sets it from cudaError_t

    def faulting() -> bool:
        raise sticky

    idle = graphs._Idle()
    idle.add(faulting)
    with pytest.raises(type(sticky)):
        idle.run(lambda: False)
    assert idle.failures == 0
    host = SimpleNamespace(poisoned="")
    order = executor._Order()
    thread = threading.Thread(target=executor._idle, args=(host, order), daemon=True)
    graphs.IDLE.add(faulting)
    thread.start()
    ticket = order.ticket(alone=False)
    with order.turn(ticket):  # a command answered: an idle period follows
        pass
    thread.join(WAIT_S)
    assert not thread.is_alive() and host.poisoned.startswith("device fault at idle")
