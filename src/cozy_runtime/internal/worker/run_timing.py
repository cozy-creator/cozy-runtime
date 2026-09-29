"""One run's measured union of cooperative executor work.

Activity messages stay in memory. Existing progress and execution boundaries publish
bounded cumulative snapshots; no selector wakeup creates a retained event of its own.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker.workspace_executions import Executions


@dataclass
class _Participant:
    sequence: int = 0
    active: bool = False


@dataclass
class _Run:
    since: float
    elapsed: float = 0
    complete: bool = True
    participants: dict[tuple[str, int], _Participant] = field(default_factory=dict)

    def advance(self, now: float) -> None:
        if any(part.active for part in self.participants.values()):
            self.elapsed += max(now - self.since, 0)
        self.since = now


class Timing:
    def __init__(
        self, executions: Executions, *, monotonic: Callable[[], float] = time.monotonic
    ) -> None:
        self.executions, self.monotonic = executions, monotonic
        self.lock = threading.RLock()
        self.runs: dict[tuple[str, str, int], _Run] = {}
        self.members: dict[tuple[str, str, int], tuple[str, str, int]] = {}

    def begin(self, owner: str, request: str, attempt: int) -> None:
        with contextlib.suppress(Exception), self.lock:
            root, _ = self.executions.scheduling_root(owner, request)
            row = self.executions.row(owner, root)
            assert row is not None
            key = owner, root, row.ordinal
            run = self.runs.get(key)
            if run is None:
                # A recovered clock has no monotonic observation of its offline time.
                with self.executions.workspace.locked() as db:
                    prior = db.execute(
                        "SELECT 1 FROM execution_events WHERE owner=? AND request=? "
                        "AND ordinal=? AND kind='run.timing' LIMIT 1",
                        key,
                    ).fetchone()
                run = _Run(self.monotonic(), complete=prior is None and root == request)
                self.runs[key] = run
            run.advance(self.monotonic())
            run.participants[request, attempt] = _Participant()
            self.members[owner, request, attempt] = key
            self._snapshot(key, run)

    def observe(self, owner: str, request: str, attempt: int, frame: dict[str, Json]) -> None:
        with contextlib.suppress(Exception), self.lock:
            key = self.members.get((owner, request, attempt))
            if key is None:
                return
            run = self.runs[key]
            part = run.participants[request, attempt]
            sequence = frame.get("sequence")
            if type(sequence) is not int or sequence <= part.sequence:
                return
            run.advance(self.monotonic())
            if sequence != part.sequence + 1 or frame.get("known") is not True:
                run.complete = False
            part.sequence, part.active = sequence, frame.get("active") is True

    def snapshot(self, owner: str, request: str, attempt: int) -> None:
        with contextlib.suppress(Exception), self.lock:
            key = self.members.get((owner, request, attempt))
            if key is not None:
                run = self.runs[key]
                run.advance(self.monotonic())
                self._snapshot(key, run)

    def end(self, owner: str, request: str, attempt: int) -> None:
        with contextlib.suppress(Exception), self.lock:
            key = self.members.pop((owner, request, attempt), None)
            if key is None:
                return
            run = self.runs[key]
            run.advance(self.monotonic())
            part = run.participants.pop((request, attempt))
            # An older executor, missing final observation, or abrupt death cannot
            # supply complete cooperative timing. Never relabel its wall time.
            run.complete &= part.sequence > 0 and not part.active
            terminal = request == key[1]
            if terminal and run.participants:
                run.complete = False
            self._snapshot(key, run, terminal=terminal)
            if terminal:
                del self.runs[key]
                for child, ordinal in run.participants:
                    self.members.pop((owner, child, ordinal), None)

    def _snapshot(self, key: tuple[str, str, int], run: _Run, *, terminal: bool = False) -> None:
        owner, root, attempt = key
        document: dict[str, Json] = {"attempt": attempt, "terminal": terminal}
        if run.complete and all(part.sequence for part in run.participants.values()):
            document["execution_ms"] = round(run.elapsed * 1000, 3)
        self.executions.record(owner, root, "run.timing", document)
