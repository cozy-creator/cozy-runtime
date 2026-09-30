"""One run's measured union of cooperative executor work.

Activity messages stay in memory. Progress, execution boundaries, and the worker's
two-second reporter publish bounded cumulative snapshots.
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
    finished: bool = False


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
            own = row if root == request else self.executions.row(owner, request)
            if row.terminal or own is None or own.ordinal != attempt or own.terminal:
                return
            ancestor = request
            with self.executions.workspace.locked() as db:
                while ancestor != root:
                    parent = db.execute(
                        "SELECT c.parent_request,c.parent_ordinal,e.ordinal "
                        "FROM execution_calls c JOIN executions e ON e.owner=c.owner "
                        "AND e.request=c.parent_request WHERE c.owner=? AND c.child_request=?",
                        (owner, ancestor),
                    ).fetchone()
                    if parent is None or parent[1] != parent[2]:
                        return  # a prior parent attempt's late dispatch is not retry work
                    ancestor = parent[0]
            key = owner, root, row.ordinal
            run = self.runs.get(key)
            if run is None:
                if root != request:
                    return  # an orphaned child cannot reopen its root's clock
                # A recovered clock has no monotonic observation of its offline time.
                with self.executions.workspace.locked() as db:
                    prior = db.execute(
                        "SELECT 1 FROM execution_events WHERE owner=? AND request=? "
                        "AND ordinal=? AND kind='run.timing' LIMIT 1",
                        key,
                    ).fetchone()
                run = _Run(self.monotonic(), complete=prior is None)
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
            if part.finished:
                run.complete = False
                return
            run.advance(self.monotonic())
            if (
                sequence != part.sequence + 1
                or frame.get("known") is not True
                or type(frame.get("active")) is not bool
            ):
                run.complete = False
            part.sequence, part.active = sequence, frame.get("active") is True
            part.finished = frame.get("finished") is True

    def snapshot(self, owner: str, request: str, attempt: int) -> None:
        with contextlib.suppress(Exception), self.lock:
            key = self.members.get((owner, request, attempt))
            if key is not None:
                run = self.runs[key]
                run.advance(self.monotonic())
                self._snapshot(key, run)

    def sample(self) -> None:
        """Refresh each active run's measured total between model progress events."""
        with self.lock:
            now = self.monotonic()
            for key, run in self.runs.items():
                with contextlib.suppress(Exception):
                    run.advance(now)
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
            run.complete &= part.sequence > 0 and not part.active and part.finished
            terminal = request == key[1]
            if terminal and run.participants:
                run.complete = False
            if terminal:
                del self.runs[key]
                for child, ordinal in run.participants:
                    self.members.pop((owner, child, ordinal), None)
            self._snapshot(key, run, terminal=terminal)

    def _snapshot(self, key: tuple[str, str, int], run: _Run, *, terminal: bool = False) -> None:
        owner, root, attempt = key
        row = self.executions.row(owner, root)
        if row is None or row.ordinal != attempt:
            return  # late observations cannot be stamped onto a resumed attempt
        document: dict[str, Json] = {"attempt": attempt, "terminal": terminal}
        if run.complete and all(part.sequence for part in run.participants.values()):
            document["execution_ms"] = round(run.elapsed * 1000, 3)
        self.executions.record(owner, root, "run.timing", document)
