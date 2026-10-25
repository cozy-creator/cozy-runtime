"""A call's clocks: its waits for a turn and its asides are off its stage and step timers, and a
stage ends with its device work. `ThreadPoolExecutor` runs submitted work in its submitter's
context.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as PythonThreadPoolExecutor
from contextvars import ContextVar

_WAITED = threading.local()
#: Records a named span on the running attempt (its stage row), set where the attempt runs.
_ASIDE: ContextVar[Callable[[str, float], None] | None] = ContextVar(
    "cozy_attempt_aside", default=None
)
#: Waits for the device work this process queued; none without a device.
_SETTLE: Callable[[], None] | None = None


@contextlib.contextmanager
def blocked() -> Iterator[None]:
    """A wait for another call's turn is not this call's work."""
    start = time.perf_counter()
    try:
        yield
    finally:
        _WAITED.seconds = getattr(_WAITED, "seconds", 0.0) + time.perf_counter() - start


def work_clock() -> float:
    """`perf_counter` less this thread's waits for a turn: what stage and step timers read."""
    return time.perf_counter() - getattr(_WAITED, "seconds", 0.0)


@contextlib.contextmanager
def asides(record: Callable[[str, float], None]) -> Iterator[None]:
    """Where an attempt runs: `aside` spans inside are recorded through `record(name, ms)`."""
    token = _ASIDE.set(record)
    try:
        yield
    finally:
        _ASIDE.reset(token)


@contextlib.contextmanager
def aside(name: str) -> Iterator[None]:
    """Work the running attempt does outside the stage open around it (a wait for its
    weights): off the stage and step clocks, and in the record as its own stage."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _WAITED.seconds = getattr(_WAITED, "seconds", 0.0) + elapsed
        record = _ASIDE.get()
        if record is not None and elapsed >= 1e-3:  # a wait that was none is not a row
            record(name, elapsed * 1000)


def settles(wait: Callable[[], None]) -> None:
    """This process's device: `wait` returns once the work queued on it is done."""
    global _SETTLE
    _SETTLE = wait


def settle() -> None:
    """A stage ends when its device work does: kernels run after the host queued them, so a
    clock read at the host's exit would hand the stage's tail to the next one."""
    if _SETTLE is not None:
        _SETTLE()


class ThreadPoolExecutor(PythonThreadPoolExecutor):
    """Standard thread pool whose submitted work runs in its submitter's context (its attempt,
    its call), so pools can survive and serve many invocations."""

    def submit[**P, T](self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        return super().submit(contextvars.copy_context().run, fn, *args, **kwargs)
