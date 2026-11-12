"""A call's clocks: its waits for a turn and the Runtime's phases (`_phases`: a weight load, a
compile) are off its stage and step timers, and a stage ends with its device work.
`ThreadPoolExecutor` runs submitted work in its submitter's context.
"""

from __future__ import annotations

import contextlib
import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as PythonThreadPoolExecutor
from contextlib import AbstractContextManager

_WAITED = threading.local()
#: Waits for the device work this process queued; none without a device.
_SETTLE: Callable[[], None] | None = None
#: Bound only after Runtime initializes its device. Called on the submitting thread to
#: capture its actual CUDA stream, including a deliberately selected graph/external stream.
WORKER_CONTEXT: Callable[[], AbstractContextManager[object]] = contextlib.nullcontext


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


def off_clock(seconds: float) -> None:
    """`seconds` of this thread's time went to work outside the stage open around it (a phase,
    `_phases`): off the stage and step clocks."""
    _WAITED.seconds = getattr(_WAITED, "seconds", 0.0) + seconds


def waited() -> float:
    """This thread's seconds off the clocks so far: what a step's device time leaves out."""
    return float(getattr(_WAITED, "seconds", 0.0))


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
        context, device = contextvars.copy_context(), WORKER_CONTEXT()

        def run() -> T:
            with device:
                return fn(*args, **kwargs)

        return super().submit(context.run, run)
