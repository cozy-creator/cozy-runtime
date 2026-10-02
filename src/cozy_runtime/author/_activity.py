"""Observe cooperative executor work, excluding its event loop's blocked waits.

This is elapsed active work, not CPU utilization. Synchronous author work and managed
thread work count; concurrent work counts once. No lifecycle decision uses this clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import selectors
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future
from concurrent.futures import ThreadPoolExecutor as PythonThreadPoolExecutor
from contextvars import ContextVar
from typing import Any

_REGISTRY_LOCK = threading.Lock()
_BASELINE: weakref.WeakSet[threading.Thread] | None = None
_POOL_THREADS: weakref.WeakSet[threading.Thread] = weakref.WeakSet()
_CURRENT: ContextVar[Activity | None] = ContextVar("cozy_executor_activity", default=None)
_WAITED = threading.local()
#: Records a named span on the running attempt (its stage row), set where the attempt runs.
_ASIDE: ContextVar[Callable[[str, float], None] | None] = ContextVar(
    "cozy_attempt_aside", default=None
)
#: Waits for the device work this process queued; none without a device.
_SETTLE: Callable[[], None] | None = None


def _owned[T](call: Callable[[], T]) -> T:
    with _REGISTRY_LOCK:
        _POOL_THREADS.add(threading.current_thread())
    return call()


@contextlib.contextmanager
def observing(emit: Callable[[int, bool, bool, bool], None] | None) -> Iterator[Activity]:
    activity = Activity(emit)
    token = _CURRENT.set(activity)
    activity.change(1)
    try:
        yield activity
    finally:
        activity.change(-1, finished=True)
        _CURRENT.reset(token)


def event_loop() -> asyncio.AbstractEventLoop:
    """The SDK's private loop, preserving unsupported platform/custom policies."""
    activity = _CURRENT.get()
    if activity is not None:
        if type(asyncio.get_event_loop_policy()) is asyncio.DefaultEventLoopPolicy and hasattr(
            selectors, "EpollSelector"
        ):
            try:
                return EventLoop(activity)
            except Exception:
                pass
        activity.change(0, unknown=True)
    return asyncio.new_event_loop()


def observe_await() -> None:
    """An author's own unobserved loop cannot turn child waits into execution."""
    activity = _CURRENT.get()
    if activity is not None:
        try:
            supported = isinstance(asyncio.get_running_loop(), EventLoop)
        except RuntimeError:
            supported = False
        if not supported:
            activity.change(0, unknown=True)


@contextlib.contextmanager
def blocked() -> Iterator[None]:
    """A wait for another call's turn is not this call's work."""
    activity = _CURRENT.get()
    if activity is not None:
        activity.change(-1)
    start = time.perf_counter()
    try:
        yield
    finally:
        _WAITED.seconds = getattr(_WAITED, "seconds", 0.0) + time.perf_counter() - start
        if activity is not None:
            activity.change(1)


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


def author_done() -> None:
    """Freeze an adapter's author clock before framework event-loop shutdown."""
    if (activity := _CURRENT.get()) is not None:
        activity.change(-1, finished=True)


class Activity:
    def __init__(self, emit: Callable[[int, bool, bool, bool], None] | None) -> None:
        global _BASELINE
        with _REGISTRY_LOCK:
            if _BASELINE is None:
                _BASELINE = weakref.WeakSet(threading.enumerate())
        self.emit = emit
        self.lock = threading.RLock()
        self.count = 0
        self.sequence = 0
        self.known = True
        self.active = False
        self.finished = False
        self.submitted: set[object] = set()
        self.unowned: set[threading.Thread] = set()

    def change(self, delta: int, *, unknown: bool = False, finished: bool = False) -> None:
        with self.lock:
            if self.finished:
                return
            self.count += delta
            if self.count == 0 or finished:
                # A joined helper inside an active callback cannot extend the
                # measured union. Only uncovered thread work makes it unknown.
                self.unowned.update(self._outside_threads())
                unknown |= not self.submitted and bool(self.unowned)
            known = (
                self.known
                and not unknown
                and not (finished and (self.count > 0 or self.submitted or self.unowned))
            )
            active = self.count > 0
            if (active, known) == (self.active, self.known) and self.sequence and not finished:
                return
            self.active, self.known = active, known
            self.finished = finished
            self.sequence += 1
            if self.emit is not None:
                # Dropped observations leave a sequence gap; the worker then reports
                # unknown instead of silently claiming a complete measurement.
                with contextlib.suppress(Exception):
                    self.emit(self.sequence, active, known, finished)

    def threaded(self, ticket: object, function: Callable[..., Any], *args: Any) -> Any:
        with _REGISTRY_LOCK:
            _POOL_THREADS.add(threading.current_thread())
        with self.lock:
            self.unowned.discard(threading.current_thread())
            self.submitted.discard(ticket)
            unknown = not self.submitted and bool(self.unowned)
        self.change(1, unknown=unknown)
        token = _CURRENT.set(self)
        try:
            return function(*args)
        finally:
            _CURRENT.reset(token)
            self.change(-1)

    def _outside_threads(self) -> set[threading.Thread]:
        with _REGISTRY_LOCK:
            return set(threading.enumerate()) - set(_BASELINE or ()) - set(_POOL_THREADS)

    def preparing[T](self, function: Callable[[], T]) -> tuple[Callable[[], T], object]:
        ticket = object()
        with self.lock:
            self.submitted.add(ticket)
        return functools.partial(self.threaded, ticket, function), ticket

    def retired(self, ticket: object) -> None:
        with self.lock:
            self.submitted.discard(ticket)
            unknown = not self.submitted and bool(self.unowned)
        self.change(0, unknown=unknown)


class _Selector(selectors.DefaultSelector):
    def __init__(self, activity: Activity) -> None:
        super().__init__()
        self.activity = activity

    def select(self, timeout: float | None = None) -> Any:
        if timeout == 0:
            return super().select(timeout)
        self.activity.change(-1)
        try:
            return super().select(timeout)
        finally:
            self.activity.change(1)


class EventLoop(asyncio.SelectorEventLoop):
    def __init__(self, activity: Activity) -> None:
        self.activity = activity
        super().__init__(_Selector(activity))

    def run_in_executor[*Ts, T](
        self, executor: Executor | None, func: Callable[[*Ts], T], *args: *Ts
    ) -> asyncio.Future[T]:
        if self.activity.finished:
            return super().run_in_executor(executor, func, *args)
        if executor is None or isinstance(executor, PythonThreadPoolExecutor):
            call, ticket = self.activity.preparing(functools.partial(func, *args))
            # The SDK pool captures context too. This submission is already observed
            # here, so its normal submit path must not count the same callback twice.
            token = _CURRENT.set(None)
            try:
                result = super().run_in_executor(executor, call)
            except BaseException:
                self.activity.retired(ticket)
                raise
            finally:
                _CURRENT.reset(token)
            result.add_done_callback(lambda _: self.activity.retired(ticket))
            return result
        else:
            # Process pools do not share this observer. Their work remains valid,
            # but an idle event loop cannot describe their execution duration.
            self.activity.change(0, unknown=True)
        return super().run_in_executor(executor, func, *args)


class ThreadPoolExecutor(PythonThreadPoolExecutor):
    """Standard thread pool whose submitted work participates in run timing.

    Only executing callbacks count; queued futures and idle pool threads do not.
    Attribution is captured per submission, so pools can survive and serve many
    invocations. Outside an invocation this behaves like the standard executor.
    """

    def submit[**P, T](self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Future[T]:
        activity = _CURRENT.get()
        call = functools.partial(fn, *args, **kwargs)
        if activity is None or activity.finished:
            return super().submit(_owned, call)
        observed, ticket = activity.preparing(call)
        try:
            future = super().submit(observed)
        except BaseException:
            activity.retired(ticket)
            raise
        future.add_done_callback(lambda _: activity.retired(ticket))
        return future
