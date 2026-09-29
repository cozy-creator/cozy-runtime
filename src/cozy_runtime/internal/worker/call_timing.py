"""Measured child-call phases, appended to the existing execution journal.

These clocks describe observations; durable execution rows remain lifecycle authority.
A process that did not see an attempt begin reports phases without inventing durations
for its unobserved interval. Finalizing is wall time after author execution returned.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from cozy_runtime.internal import effect_interfaces, source_interfaces
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker.workspace_calls import Calls
from cozy_runtime.internal.worker.workspace_executions import Executions

Phase = Literal["queued", "preparing", "running", "finalizing", "terminal"]
DURATIONS = {"queued": "queued_ms", "preparing": "preparation_ms", "running": "execution_ms"}


@dataclass
class _Clock:
    attempt: int
    complete: bool
    phase: Phase = "queued"
    since: float = 0.0
    at_ms: int = 0
    execution_started_ms: int = 0
    durations: dict[str, float] = field(default_factory=lambda: dict.fromkeys(DURATIONS, 0.0))

    def measurements(self) -> dict[str, Json]:
        if not self.complete:
            return {}
        result: dict[str, Json] = {
            name: round(self.durations[phase] * 1000, 3) for phase, name in DURATIONS.items()
        }
        if self.execution_started_ms:
            result["execution_started_unix_ms"] = self.execution_started_ms
        return result


class Timing:
    def __init__(
        self,
        executions: Executions,
        calls: Calls,
        label: Callable[[str, int, int], str],
        *,
        monotonic: Callable[[], float] = time.monotonic,
        unix_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    ) -> None:
        self.executions, self.calls, self.label = executions, calls, label
        self.monotonic, self.unix_ms = monotonic, unix_ms
        self.lock = threading.Lock()
        self.clocks: dict[tuple[str, str], _Clock] = {}

    def phase(
        self,
        owner: str,
        request: str,
        attempt: int,
        phase: Phase,
        *,
        fresh: bool = False,
        status: str = "",
    ) -> dict[str, Json]:
        try:
            return self._phase(owner, request, attempt, phase, fresh=fresh, status=status)
        except Exception:
            # Observability cannot refuse an accepted call or abandon its valid outcome.
            # Once an interval is unrecorded, later totals for this attempt are unknown.
            with self.lock:
                clock = self.clocks.get((owner, request))
                if clock is not None and clock.attempt == attempt:
                    clock.complete = False
            logging.getLogger(__name__).warning("call phase could not be recorded", exc_info=True)
            return {}

    def _phase(
        self,
        owner: str,
        request: str,
        attempt: int,
        phase: Phase,
        *,
        fresh: bool,
        status: str,
    ) -> dict[str, Json]:
        call = self.calls.child(owner, request)
        if call is None or call.target().module in (
            effect_interfaces.MODULE,
            source_interfaces.MODULE,
        ):
            return {}
        with self.lock:
            key = (owner, request)
            clock = self.clocks.get(key)
            if clock is not None and (
                attempt < clock.attempt or (attempt == clock.attempt and clock.phase == "terminal")
            ):
                return clock.measurements() if attempt == clock.attempt else {}
            if clock is not None and attempt == clock.attempt and phase == clock.phase:
                return clock.measurements()
            now = self.monotonic()
            if clock is None or clock.attempt != attempt:
                clock = _Clock(attempt, fresh, since=now)
                self.clocks[key] = clock
            elif clock.phase in DURATIONS:
                clock.durations[clock.phase] += max(now - clock.since, 0.0)
            clock.phase, clock.since = phase, now
            clock.at_ms = max(self.unix_ms(), clock.at_ms + 1)
            if phase == "running" and not clock.execution_started_ms:
                clock.execution_started_ms = clock.at_ms
            target = call.target()
            document: dict[str, Json] = {
                "request": request,
                "parent": call.parent_request,
                "index": call.call_index,
                "attempt": attempt,
                "module": target.module,
                "export": target.export,
                "label": self.label(call.parent_request, call.parent_ordinal, call.call_index),
                "phase": phase,
                "at_unix_ms": clock.at_ms,
                "called_unix_ms": call.created_ms,
                **clock.measurements(),
            }
            if status:
                document["status"] = status
            root, _ = self.executions.scheduling_root(owner, request)
            self.executions.record(owner, root, "call.phase", document)
            return clock.measurements()
