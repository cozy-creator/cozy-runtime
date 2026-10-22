"""Graphs record off the request path: the executor's idle holder and the graphs' idle queue,
on real threads (the captures themselves are GPU work: tests/test_graphs.py)."""

from __future__ import annotations

import threading
import time

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


def _serving() -> executor._Order:
    order = executor._Order()
    threading.Thread(target=executor._idle, args=(order,), daemon=True).start()
    return order


def _drained() -> None:
    assert not graphs.IDLE.jobs, "an earlier test left idle work queued"


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
