"""Supervised units: one thread per unit of work, contained, woken only by its own events.

A unit's exception is its own: `crashed(exc)` records it where the unit's owner reads it.
A crash handler that itself raises leaves work with no owner: a worker `integrity` failure.
Nothing polls: a unit re-reads its state on every wake, so a lost or repeated one is harmless.
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
    def __init__(self, key: str, stopping: threading.Event) -> None:
        self.key = key
        self.stopping = stopping
        self.woken = threading.Event()
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
        """Block until `ready()`, re-checked on every poke, or the monotonic instant `until`."""
        while not ready():
            if self.stopping.is_set():
                raise Stopping(self.key)
            if until is not None and until <= time.monotonic():
                return False
            self.wait(until)
        return True


class Supervisor:
    def __init__(self, integrity: Callable[[str], None], note: Callable[[str], None]) -> None:
        self.integrity, self.note = integrity, note
        self.units: dict[str, Unit] = {}
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.drained = threading.Condition(self.lock)

    def spawn(
        self,
        key: str,
        body: Callable[[Unit], object],
        crashed: Callable[[BaseException], bool] = lambda exc: False,
    ) -> None:
        """Start unit `key`, or poke it when live. `crashed(exc)` records a crash and answers
        whether to run `body` again."""
        with self.lock:
            if (unit := self.units.get(key)) is not None:
                unit.poke()
                return
            if self.stopping.is_set():
                return
            unit = self.units[key] = Unit(key, self.stopping)
        threading.Thread(
            target=self._run, args=(unit, body, crashed), name=key, daemon=True
        ).start()

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
        with self.lock:
            self.drained.wait_for(lambda: not self.units)

    def keys(self, prefix: str) -> list[str]:
        with self.lock:
            return [key for key in self.units if key.startswith(prefix)]

    def busy(self, prefix: str = "") -> bool:
        return bool(self.keys(prefix))

    def quiet_now(self) -> bool:
        """Every unit parked with nothing pending: what a test waits for instead of a tick."""
        with self.lock:
            return all(unit.idle and not unit.woken.is_set() for unit in self.units.values())
