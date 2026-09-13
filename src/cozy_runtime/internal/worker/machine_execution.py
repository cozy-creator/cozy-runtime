"""Durable machine execution metadata, joined to the existing attempt/custody rows.

Transport authenticates the owner; Worker validates preparation and executes offers.
This module stores neither another outcome copy nor native byte/catalog state.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.workspace import Journal, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

SCHEMA7 = """
CREATE TABLE executions (
 owner TEXT NOT NULL, request TEXT NOT NULL, submission TEXT NOT NULL,
 capture BLOB NOT NULL, offer BLOB NOT NULL, accepted_ms INTEGER NOT NULL,
 ordinal INTEGER NOT NULL, generation INTEGER NOT NULL,
 state TEXT NOT NULL, desired TEXT NOT NULL DEFAULT 'run',
 collected INTEGER NOT NULL DEFAULT 0, sequence INTEGER NOT NULL DEFAULT 0,
 compacted_through INTEGER NOT NULL DEFAULT 0,
 max_attempts INTEGER NOT NULL DEFAULT 1, retry_delay_ms INTEGER NOT NULL DEFAULT 0,
 finished_ms INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(owner,request), UNIQUE(owner,submission)
) STRICT;
CREATE TABLE execution_events (
 owner TEXT NOT NULL, request TEXT NOT NULL, sequence INTEGER NOT NULL,
 ordinal INTEGER NOT NULL, at_ms INTEGER NOT NULL, kind TEXT NOT NULL, body BLOB NOT NULL,
 PRIMARY KEY(owner,request,sequence)
) STRICT;
CREATE TABLE execution_commands (
 owner TEXT NOT NULL, request TEXT NOT NULL, command TEXT NOT NULL,
 intent BLOB NOT NULL, response BLOB NOT NULL,
 PRIMARY KEY(owner,request,command)
) STRICT;
"""

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
MAX_PROGRESS_EVENTS = 256
MAX_EVENT_BYTES = 64 << 10
MAX_PAGE = 256
MAX_PAGE_BYTES = 256 << 10
TERMINAL = frozenset({"succeeded", "failed", "paused", "canceled"})


@dataclass(frozen=True)
class Receipt:
    request_id: str
    submission_id: str
    capture_digest: bytes
    invocation_spec_digest: bytes
    accepted_at_ms: int


@dataclass(frozen=True)
class State:
    request_id: str
    attempt_ordinal: int
    generation: int
    state: str
    collected: bool
    sequence: int

    def document(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "attempt_ordinal": self.attempt_ordinal,
            "generation": self.generation,
            "state": self.state,
            "collected": self.collected,
            "sequence": self.sequence,
        }


@dataclass(frozen=True)
class Event:
    sequence: int
    attempt_ordinal: int
    at_ms: int
    kind: str
    body: bytes  # bounded canonical JSON, not arbitrary Python serialization


@dataclass(frozen=True)
class EventPage:
    events: tuple[Event, ...]
    next_after: int
    head_sequence: int
    compacted_through: int


@contextmanager
def transaction(db: Journal) -> Iterator[None]:
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def _id(value: str) -> None:
    if _ID.fullmatch(value) is None:
        raise WorkspaceRefusal("execution identity must be a bounded opaque ID")


def _offer_bytes(offer: pb.AttemptOffer) -> bytes:
    value = pb.AttemptOffer()
    value.CopyFrom(offer)
    # Connection/admission stamps change on reattachment; the invocation does not.
    for name in ("record_owner_epoch", "control_stream_epoch", "worker_boot_id", "admission_epoch"):
        value.ClearField(name)
    return value.SerializeToString(deterministic=True)


class Executions:
    def __init__(self, workspace: Workspace, *, clock_ms: Callable[[], int] | None = None):
        self.workspace = workspace
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)

    @staticmethod
    def _row(db: Journal, owner: str, request: str) -> sqlite3.Row:
        Workspace.owner(owner)
        _id(request)
        row = db.execute(
            "SELECT * FROM executions WHERE owner=? AND request=?", (owner, request)
        ).fetchone()
        if row is None:
            raise WorkspaceRefusal("execution is not held by this owner")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _state(row: sqlite3.Row) -> State:
        return State(
            row["request"],
            row["ordinal"],
            row["generation"],
            row["state"],
            bool(row["collected"]),
            row["sequence"],
        )

    @staticmethod
    def _receipt(row: sqlite3.Row) -> Receipt:
        offer = pb.AttemptOffer.FromString(row["offer"])
        return Receipt(
            row["request"],
            row["submission"],
            row["capture"],
            offer.invocation_spec_digest,
            row["accepted_ms"],
        )

    def _event(self, db: Journal, row: sqlite3.Row, kind: str, body: bytes) -> None:
        if len(body) > MAX_EVENT_BYTES:
            raise WorkspaceRefusal("execution event exceeds its metadata bound")
        canonical_json.decode(body)
        sequence = row["sequence"] + 1
        db.execute(
            "UPDATE executions SET sequence=? WHERE owner=? AND request=?",
            (sequence, row["owner"], row["request"]),
        )
        db.execute(
            "INSERT INTO execution_events VALUES(?,?,?,?,?,?,?)",
            (row["owner"], row["request"], sequence, row["ordinal"], self.clock_ms(), kind, body),
        )
        # Only live progress is lossy. Control, outcomes and authored log records remain.
        if kind == "progress":
            obsolete = db.execute(
                "SELECT sequence FROM execution_events WHERE owner=? AND request=? "
                "AND kind='progress' ORDER BY sequence DESC LIMIT -1 OFFSET ?",
                (row["owner"], row["request"], MAX_PROGRESS_EVENTS),
            ).fetchall()
            if obsolete:
                through = obsolete[0][0]
                db.execute(
                    "DELETE FROM execution_events WHERE owner=? AND request=? "
                    "AND kind='progress' AND sequence<=?",
                    (row["owner"], row["request"], through),
                )
                db.execute(
                    "UPDATE executions SET compacted_through=max(compacted_through,?) "
                    "WHERE owner=? AND request=?",
                    (through, row["owner"], row["request"]),
                )

    def submit(
        self,
        owner: str,
        submission_id: str,
        capture_digest: bytes,
        offer: pb.AttemptOffer,
        *,
        result_schema: bytes = b"",
        worker_boot: str = "",
        memoize: bool = False,
        max_attempts: int = 1,
        retry_delay_ms: int = 0,
    ) -> Receipt:
        """Atomically accept a verified capture/root and its first existing attempt."""
        Workspace.owner(owner)
        _id(submission_id)
        if not 1 <= max_attempts <= 100 or not 0 <= retry_delay_ms <= 86_400_000:
            raise WorkspaceRefusal("execution retry policy exceeds its bound")
        self.workspace.validate_offer(offer)
        if offer.attempt_ordinal != 1 or len(capture_digest) != 32:
            raise WorkspaceRefusal("new execution needs attempt one and an exact capture digest")
        self.workspace.recover(owner)
        raw = _offer_bytes(offer)
        with self.workspace.locked() as db, transaction(db):
            old = db.execute(
                "SELECT * FROM executions WHERE owner=? AND (submission=? OR request=?)",
                (owner, submission_id, offer.request_id),
            ).fetchall()
            if old:
                row = old[0]
                if len(old) != 1 or (
                    row["submission"],
                    row["request"],
                    row["capture"],
                    row["offer"],
                    row["max_attempts"],
                    row["retry_delay_ms"],
                ) != (
                    submission_id,
                    offer.request_id,
                    capture_digest,
                    raw,
                    max_attempts,
                    retry_delay_ms,
                ):
                    raise WorkspaceRefusal("execution submission identity changed")
                return self._receipt(row)
            if db.execute(
                "SELECT 1 FROM attempts WHERE owner=? AND request=?", (owner, offer.request_id)
            ).fetchone():
                raise WorkspaceRefusal("request already belongs to another execution path")
            self.workspace.accept_in(
                db,
                owner,
                offer,
                result_schema=result_schema,
                worker_boot=worker_boot,
                memoize=memoize,
            )
            db.execute(
                "INSERT INTO executions(owner,request,submission,capture,offer,accepted_ms,"
                "ordinal,generation,state,max_attempts,retry_delay_ms) "
                "VALUES(?,?,?,?,?,?,1,1,'queued',?,?)",
                (
                    owner,
                    offer.request_id,
                    submission_id,
                    capture_digest,
                    raw,
                    self.clock_ms(),
                    max_attempts,
                    retry_delay_ms,
                ),
            )
            row = self._row(db, owner, offer.request_id)
            self._event(db, row, "accepted", canonical_json.encode({"generation": 1}))
            return self._receipt(row)

    def owns(self, owner: str, request: str) -> bool:
        with self.workspace.locked() as db:
            return (
                db.execute(
                    "SELECT 1 FROM executions WHERE owner=? AND request=?", (owner, request)
                ).fetchone()
                is not None
            )

    def pending(self, owner: str) -> tuple[str, ...]:
        """Stable admission order; the existing Worker chooses each request's lane."""
        Workspace.owner(owner)
        with self.workspace.locked() as db:
            return tuple(
                row[0]
                for row in db.execute(
                    "SELECT request FROM executions WHERE owner=? "
                    "AND state IN ('queued','running','pausing','canceling') "
                    "ORDER BY accepted_ms,request",
                    (owner,),
                )
            )

    def stop_queued(self, owner: str, request: str, *, deadline: bool = False) -> None:
        """Stop work that has no live executor, through the same durable outcome codec."""
        with self.workspace.locked() as db:
            row = self._row(db, owner, request)
            if row["state"] not in ("queued", "pausing", "canceling"):
                raise WorkspaceRefusal("only undispatched execution may stop without an executor")
            offer = pb.AttemptOffer.FromString(row["offer"])
            ordinal = row["ordinal"]
        body, digest = documents.identity(
            pb.AttemptOutcomeBody(
                request_id=request,
                attempt_ordinal=ordinal,
                invocation_spec_digest=documents.spell(offer.invocation_spec_digest),
                status=pb.OUTCOME_STATUS_CANCELED,
                safe_message="deadline expired before dispatch"
                if deadline
                else "stopped before dispatch",
                cause=pb.OutcomeCause(
                    code=pb.CAUSE_CODE_DEADLINE_EXPIRED
                    if deadline
                    else pb.CAUSE_CODE_CLIENT_CANCEL,
                    origin=pb.CAUSE_ORIGIN_WORKER,
                ),
            )
        )
        self.workspace.outcome(
            owner,
            pb.AttemptOutcome(
                request_id=request,
                attempt_ordinal=ordinal,
                invocation_spec_digest=offer.invocation_spec_digest,
                outcome_id="out-" + digest.hex()[:24],
                outcome_digest=digest,
                outcome_canonical_bytes=body,
            ),
        )
        self.reconcile(owner, request)

    def reconcile(self, owner: str, request: str) -> None:
        """Project an existing durable outcome, including native process-death recovery."""
        self.workspace.recover(owner)
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, request)
            attempt = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, request, row["ordinal"]),
            ).fetchone()
            if attempt is None:
                raise WorkspaceRefusal("execution attempt history is missing")
            if not attempt["outcome"] or row["state"] in TERMINAL:
                return
            body = documents.read(attempt["outcome"], pb.AttemptOutcomeBody)
            state = (
                "paused"
                if row["desired"] == "pause"
                else "canceled"
                if row["desired"] == "cancel" or body["status"] == pb.OUTCOME_STATUS_CANCELED
                else "succeeded"
                if body["status"] == pb.OUTCOME_STATUS_SUCCEEDED
                else "failed"
            )
            db.execute(
                "UPDATE executions SET state=?,finished_ms=? WHERE owner=? AND request=?",
                (state, self.clock_ms(), owner, request),
            )
            self._event(
                db,
                row,
                "outcome",
                canonical_json.encode(
                    {
                        "state": state,
                        "outcome_id": attempt["outcome_id"],
                        "outcome_digest": documents.spell(attempt["outcome_digest"]),
                    }
                ),
            )

    def status(self, owner: str, request: str) -> State:
        self.reconcile(owner, request)
        with self.workspace.locked() as db:
            return self._state(self._row(db, owner, request))

    def retry_due(self, owner: str, *, worker_boot: str = "") -> None:
        """Retry bounded executor failures only; authored errors need an explicit new intent."""
        with self.workspace.locked() as db:
            rows = db.execute(
                "SELECT e.request,e.generation,e.ordinal,e.offer,a.outcome "
                "FROM executions e JOIN attempts a ON a.owner=e.owner AND a.request=e.request "
                "AND a.ordinal=e.ordinal WHERE e.owner=? AND e.state='failed' "
                "AND e.desired='run' AND e.ordinal<e.max_attempts "
                "AND e.finished_ms+e.retry_delay_ms<=?",
                (owner, self.clock_ms()),
            ).fetchall()
        for row in rows:
            outcome = documents.read(row["outcome"], pb.AttemptOutcomeBody)
            if outcome.get("status") != pb.OUTCOME_STATUS_FAILED or outcome.get("cause", {}).get(
                "code"
            ) not in (
                pb.CAUSE_CODE_EXECUTOR_FAULT,
                pb.CAUSE_CODE_EXECUTOR_INVALIDATED,
            ):
                continue
            spec = documents.read(
                pb.AttemptOffer.FromString(row["offer"]).invocation_spec_canonical_bytes,
                pb.InvocationSpec,
            )
            deadline = int(spec.get("deadline_unix_ms", 0))
            if deadline and self.clock_ms() >= deadline:
                continue
            self.control(
                owner,
                row["request"],
                f"retry.{row['ordinal']}",
                row["generation"],
                "resume",
                worker_boot=worker_boot,
                _automatic=True,
            )

    def offer(self, owner: str, request: str) -> pb.AttemptOffer:
        """Return pinned work for Worker; this does not execute or pick a machine."""
        with self.workspace.locked() as db:
            row = self._row(db, owner, request)
            if row["state"] not in ("queued", "running") or row["desired"] != "run":
                raise WorkspaceRefusal("execution is not dispatchable")
            offer = pb.AttemptOffer.FromString(row["offer"])
            offer.attempt_ordinal = row["ordinal"]
            return offer

    def observe(self, owner: str, frame: pb.WorkerFrame) -> None:
        if frame.HasField("attempt_outcome"):
            self.reconcile(owner, frame.attempt_outcome.request_id)
            return
        accepted = frame.attempt_accepted
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, accepted.request_id)
            if row["ordinal"] != accepted.attempt_ordinal or row["state"] != "queued":
                return
            db.execute(
                "UPDATE executions SET state='running' WHERE owner=? AND request=?",
                (owner, accepted.request_id),
            )
            self._event(
                db, row, "running", canonical_json.encode({"generation": row["generation"]})
            )

    def progress(self, owner: str, request: str, ordinal: int, document: dict[str, Any]) -> None:
        raw = canonical_json.encode(document)
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, request)
            if row["ordinal"] == ordinal and row["state"] not in TERMINAL:
                self._event(
                    db, row, "progress" if document.get("type") == "progress" else "log", raw
                )

    def events(self, owner: str, request: str, after: int = 0, limit: int = MAX_PAGE) -> EventPage:
        if not 0 <= after < 1 << 63 or not 1 <= limit <= MAX_PAGE:
            raise WorkspaceRefusal("execution event cursor or page bound is invalid")
        self.reconcile(owner, request)
        with self.workspace.locked() as db:
            row = self._row(db, owner, request)
            if after > row["sequence"]:
                raise WorkspaceRefusal("execution event cursor is ahead of its history")
            rows = db.execute(
                "SELECT * FROM execution_events WHERE owner=? AND request=? "
                "AND sequence>? ORDER BY sequence LIMIT ?",
                (owner, request, after, limit),
            ).fetchall()
            size = 0
            selected = []
            for event in rows:
                size += len(event["body"])
                if size > MAX_PAGE_BYTES:
                    break
                selected.append(event)
            rows = selected
            return EventPage(
                tuple(
                    Event(r["sequence"], r["ordinal"], r["at_ms"], r["kind"], r["body"])
                    for r in rows
                ),
                rows[-1]["sequence"] if rows else after,
                row["sequence"],
                row["compacted_through"],
            )

    def control(
        self,
        owner: str,
        request: str,
        command_id: str,
        expected_generation: int,
        action: Literal["pause", "resume", "cancel"],
        *,
        worker_boot: str = "",
        _automatic: bool = False,
    ) -> State:
        _id(command_id)
        command_id = ("retry:" if _automatic else "user:") + command_id
        if action not in ("pause", "resume", "cancel") or not 0 < expected_generation < 1 << 63:
            raise WorkspaceRefusal("execution command is invalid")
        self.reconcile(owner, request)
        intent = canonical_json.encode({"action": action, "generation": expected_generation})
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, request)
            previous = db.execute(
                "SELECT * FROM execution_commands WHERE owner=? AND request=? AND command=?",
                (owner, request, command_id),
            ).fetchone()
            if previous is not None:
                if previous["intent"] != intent:
                    raise WorkspaceRefusal("execution command identity changed")
                return State(**canonical_json.decode(previous["response"]))
            if row["generation"] != expected_generation:
                raise WorkspaceRefusal("execution command generation is stale")
            state, ordinal, desired = row["state"], row["ordinal"], row["desired"]
            if action != "resume" and state in ("succeeded", "canceled"):
                result = self._state(row)
                db.execute(
                    "INSERT INTO execution_commands VALUES(?,?,?,?,?)",
                    (owner, request, command_id, intent, canonical_json.encode(result.document())),
                )
                return result
            if action == "resume":
                if state not in ("paused", "failed"):
                    raise WorkspaceRefusal("only paused or failed execution can resume")
                attempt = db.execute(
                    "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                    (owner, request, ordinal),
                ).fetchone()
                if attempt is None or not attempt["outcome"]:
                    raise WorkspaceRefusal("previous execution has not durably stopped")
                spec = documents.read(attempt["invocation"], pb.InvocationSpec)
                deadline = int(spec.get("deadline_unix_ms", 0))
                if deadline and self.clock_ms() >= deadline:
                    raise WorkspaceRefusal("execution deadline has expired")
                ordinal += 1
                offer = pb.AttemptOffer.FromString(row["offer"])
                offer.attempt_ordinal = ordinal
                self.workspace.accept_in(
                    db,
                    owner,
                    offer,
                    result_schema=attempt["result_schema"],
                    worker_boot=worker_boot,
                    memoize=bool(attempt["memoize"]),
                )
                state, desired = "queued", "run"
            elif state not in TERMINAL:
                state, desired = (
                    ("pausing", "pause") if action == "pause" else ("canceling", "cancel")
                )
            elif action == "cancel" and state in ("paused", "failed"):
                state, desired = "canceled", "cancel"
            generation = row["generation"] + 1
            db.execute(
                "UPDATE executions SET state=?,desired=?,ordinal=?,generation=?,collected=0 "
                "WHERE owner=? AND request=?",
                (state, desired, ordinal, generation, owner, request),
            )
            self._event(db, self._row(db, owner, request), "control", intent)
            result = self._state(self._row(db, owner, request))
            db.execute(
                "INSERT INTO execution_commands VALUES(?,?,?,?,?)",
                (owner, request, command_id, intent, canonical_json.encode(result.document())),
            )
            return result

    def collect(self, owner: str, request: str, ordinal: int | None = None) -> pb.AttemptOutcome:
        """Return the existing exact terminal; its native grants use existing byte readers."""
        state = self.status(owner, request)
        selected = state.attempt_ordinal if ordinal is None else ordinal
        with self.workspace.locked() as db:
            self._row(db, owner, request)
            row = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, request, selected),
            ).fetchone()
            if row is None or not row["outcome"]:
                raise WorkspaceRefusal("execution has no durable terminal to collect")
            return self.workspace._outcome_frame(row)

    def acknowledge_collection(self, owner: str, ack: pb.AttemptOutcomeAck) -> State:
        self.status(owner, ack.request_id)
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, ack.request_id)
            attempt = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, ack.request_id, ack.attempt_ordinal),
            ).fetchone()
            if (
                attempt is None
                or not attempt["outcome"]
                or attempt["spec"] != ack.invocation_spec_digest
                or attempt["outcome_id"] != ack.outcome_id
                or attempt["outcome_digest"] != ack.outcome_digest
            ):
                raise WorkspaceRefusal("collection acknowledgment changed its exact terminal")
            if ack.attempt_ordinal == row["ordinal"] and not row["collected"]:
                db.execute(
                    "UPDATE executions SET collected=1 WHERE owner=? AND request=?",
                    (owner, ack.request_id),
                )
                self._event(
                    db, row, "collected", canonical_json.encode({"outcome_id": ack.outcome_id})
                )
            # Collection is not native deletion. Explicit existing release APIs govern custody.
            return self._state(self._row(db, owner, ack.request_id))

    def retention_required(self, owner: str) -> bool:
        Workspace.owner(owner)
        with self.workspace.locked() as db:
            return (
                db.execute(
                    "SELECT 1 FROM executions WHERE owner=? AND "
                    "(state NOT IN ('succeeded','failed','canceled') OR collected=0) LIMIT 1",
                    (owner,),
                ).fetchone()
                is not None
            )
