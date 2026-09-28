"""Supervised units: one thread per unit of work, contained, woken only by its own events.

A unit's exception is its own: `crashed(exc)` records the failure where the unit's owner
reads it, and every other unit and the worker go on. A crash handler that itself raises
leaves work with no owner, which is the one thing a worker may not survive: that is a worker
integrity failure (`integrity(why)`).

Nothing here polls. A unit blocks in `wait_for` until something it depends on pokes it or
its own next due instant arrives (a deadline), and re-reads its state when woken: a wake is
a hint, so a lost or repeated one cannot wedge it.
"""

from __future__ import annotations

import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path


def fault_text(exc: BaseException) -> str:
    """`Type: message @ file:line function < ...`, innermost frame first."""
    frames = " < ".join(
        f"{Path(frame.filename).name}:{frame.lineno} {frame.name}"
        for frame in reversed(traceback.extract_tb(exc.__traceback__))
    )
    return f"{type(exc).__name__}: {exc}" + (f" @ {frames}" if frames else "")


class Stopping(Exception):
    """The worker is stopping: a unit leaves, its work durable for the next boot."""


class Unit:
    """One supervised unit: a named thread and the event that wakes it."""

    def __init__(self, key: str, stopping: threading.Event) -> None:
        self.key = key
        self.stopping = stopping
        self.woken = threading.Event()
        #: parked in `wait_for` with nothing pending (tests read it through `quiet`)
        self.idle = False

    def poke(self) -> None:
        self.woken.set()

    def wait(self, until: float | None = None) -> None:
        """Block until the next poke, or the monotonic instant `until`."""
        if self.stopping.is_set():
            raise Stopping(self.key)
        self.idle = True
        self.woken.wait(None if until is None else max(0.0, until - time.monotonic()))
        self.idle = False
        self.woken.clear()

    def wait_for(self, ready: Callable[[], bool], until: float | None = None) -> bool:
        """Block until `ready()` or the monotonic instant `until`; re-check on every poke."""
        while not ready():
            if self.stopping.is_set():
                raise Stopping(self.key)
            timeout = None if until is None else until - time.monotonic()
            if timeout is not None and timeout <= 0:
                return False
            self.idle = True
            self.woken.wait(timeout)
            self.idle = False
            self.woken.clear()
        return True


class Supervisor:
    """The registry of live units and their containment."""

    def __init__(
        self,
        integrity: Callable[[str], None],
        note: Callable[[str], None],
    ) -> None:
        self.integrity = integrity
        self.note = note
        self.units: dict[str, Unit] = {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.drained = threading.Condition(self.lock)

    def spawn(
        self,
        key: str,
        body: Callable[[Unit], object],
        crashed: Callable[[BaseException], bool] = lambda exc: False,
    ) -> bool:
        """Start unit `key` unless it is live (then poke it). `crashed(exc)` records the
        failure and answers whether to run `body` again; if it raises, the work has no owner
        left and the worker fails. True when a unit started."""
        with self.lock:
            unit = self.units.get(key)
            if unit is not None:
                unit.poke()
                return False
            if self.stopping.is_set():
                return False
            unit = self.units[key] = Unit(key, self.stopping)
        threading.Thread(
            target=self._run, args=(unit, body, crashed), name=key, daemon=True
        ).start()
        return True

    def _run(
        self, unit: Unit, body: Callable[[Unit], object], crashed: Callable[[BaseException], bool]
    ) -> None:
        try:
            while True:
                try:
                    body(unit)
                except Stopping:
                    return
                except Exception as exc:
                    self.note(f"{unit.key} crashed: {fault_text(exc)}")
                    try:
                        if crashed(exc):
                            continue
                    except Exception as inner:
                        self.integrity(
                            f"{unit.key} has no owner: recording its failure raised "
                            f"{fault_text(inner)} (after {fault_text(exc)})"
                        )
                        return
                with self.lock:
                    # A poke that raced the body's return is work for this unit, not lost.
                    if not unit.woken.is_set() or self.stopping.is_set():
                        del self.units[unit.key]
                        self.drained.notify_all()
                        return
                    unit.woken.clear()
        finally:
            with self.lock:
                if self.units.get(unit.key) is unit:
                    del self.units[unit.key]
                self.drained.notify_all()

    def poke(self, key: str) -> None:
        with self.lock:
            unit = self.units.get(key)
        if unit is not None:
            unit.poke()

    def stop(self) -> None:
        """Every unit leaves at its next wait; no unit starts."""
        self.stopping.set()
        with self.lock:
            units = list(self.units.values())
        for unit in units:
            unit.poke()

    def join(self) -> None:
        """Wait for every unit to leave (after `stop`)."""
        with self.lock:
            self.drained.wait_for(lambda: not self.units)

    def keys(self, prefix: str) -> list[str]:
        with self.lock:
            return [key for key in self.units if key.startswith(prefix)]

    def busy(self, prefix: str = "") -> bool:
        with self.lock:
            return any(key.startswith(prefix) for key in self.units)

    def quiet_now(self) -> bool:
        """Every unit parked with nothing pending: what a test waits for instead of a tick."""
        with self.lock:
            return all(unit.idle and not unit.woken.is_set() for unit in self.units.values())
