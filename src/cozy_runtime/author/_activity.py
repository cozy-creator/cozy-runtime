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
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any, cast

_INFRASTRUCTURE_THREADS = set(threading.enumerate())


class Activity:
    def __init__(self, emit: Callable[[int, bool, bool], None] | None) -> None:
        self.emit = emit
        self.lock = threading.RLock()
        self.count = 0
        self.sequence = 0
        self.known = True
        self.active = False
        self.baseline = _INFRASTRUCTURE_THREADS
        self.managed: set[threading.Thread] = set()

    def change(self, delta: int, *, unknown: bool = False) -> None:
        with self.lock:
            self.count += delta
            known = self.known and not unknown
            active = self.count > 0
            if (active, known) == (self.active, self.known) and self.sequence:
                return
            self.active, self.known = active, known
            self.sequence += 1
            if self.emit is not None:
                # Dropped observations leave a sequence gap; the worker then reports
                # unknown instead of silently claiming a complete measurement.
                with contextlib.suppress(Exception):
                    self.emit(self.sequence, active, known)

    def threaded(self, function: Callable[..., Any], *args: Any) -> Any:
        with self.lock:
            self.managed.add(threading.current_thread())
        self.change(1)
        try:
            return function(*args)
        finally:
            self.change(-1)

    def untracked_threads(self) -> bool:
        with self.lock:
            return bool(set(threading.enumerate()) - self.baseline - self.managed)


class _Selector(selectors.DefaultSelector):
    def __init__(self, activity: Activity) -> None:
        super().__init__()
        self.activity = activity

    def select(self, timeout: float | None = None) -> Any:
        if timeout == 0:
            return super().select(timeout)
        self.activity.change(-1, unknown=self.activity.untracked_threads())
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
        if executor is None or isinstance(executor, ThreadPoolExecutor):
            func = cast(Callable[[*Ts], T], functools.partial(self.activity.threaded, func))
        else:
            # Process pools do not share this observer. Their work remains valid,
            # but an idle event loop cannot describe their execution duration.
            self.activity.change(0, unknown=True)
        return super().run_in_executor(executor, func, *args)
