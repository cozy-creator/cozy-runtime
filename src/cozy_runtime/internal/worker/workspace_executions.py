"""Durable machine execution metadata, joined to the existing attempt/custody rows.

Transport authenticates the owner; Worker validates preparation and executes offers.
This module stores neither another outcome copy nor native byte/catalog state.
"""

from __future__ import annotations

import base64
import re
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.workspace import Journal, Row, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

SCHEMA7 = """
ALTER TABLE workspace_state ADD COLUMN execution_workspace_id TEXT NOT NULL DEFAULT '';
UPDATE workspace_state SET execution_workspace_id=lower(hex(randomblob(16)));
CREATE TABLE executions (
 owner TEXT NOT NULL, request TEXT NOT NULL, submission TEXT NOT NULL,
 capture BLOB NOT NULL, offer BLOB NOT NULL, accepted_ms INTEGER NOT NULL,
 ordinal INTEGER NOT NULL, generation INTEGER NOT NULL,
 state TEXT NOT NULL, desired TEXT NOT NULL DEFAULT 'run',
 collected INTEGER NOT NULL DEFAULT 0, sequence INTEGER NOT NULL DEFAULT 0,
 compacted_through INTEGER NOT NULL DEFAULT 0,
 max_attempts INTEGER NOT NULL DEFAULT 1, retry_delay_ms INTEGER NOT NULL DEFAULT 0,
 finished_ms INTEGER NOT NULL DEFAULT 0,
 capture_document BLOB NOT NULL DEFAULT x'', preparation BLOB NOT NULL DEFAULT x'',
 worker_id TEXT NOT NULL DEFAULT '', accepted_boot TEXT NOT NULL DEFAULT '',
 retention_waived INTEGER NOT NULL DEFAULT 0,
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


class StaleExecutionGeneration(WorkspaceRefusal):
    """A rejected command, distinct from a lost reply to an accepted command."""

    code = "execution_generation_stale"


@dataclass(frozen=True)
class Receipt:
    request_id: str
    submission_id: str
    capture_digest: bytes
    invocation_spec_digest: bytes
    accepted_at_ms: int
    worker_id: str
    worker_boot_id: str
    publication_authorization_id: str = ""


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

    @property
    def workspace_id(self) -> str:
        with self.workspace.locked() as db:
            return str(
                db.execute(
                    "SELECT execution_workspace_id FROM workspace_state WHERE singleton=1"
                ).fetchone()[0]
            )

    @staticmethod
    def _row(db: Journal, owner: str, request: str) -> Row:
        Workspace.owner(owner)
        _id(request)
        row = db.execute(
            "SELECT * FROM executions WHERE owner=? AND request=?", (owner, request)
        ).fetchone()
        if row is None:
            raise WorkspaceRefusal("execution is not held by this owner")
        return cast(Row, row)

    @staticmethod
    def _state(row: Row) -> State:
        return State(
            row["request"],
            row["ordinal"],
            row["generation"],
            row["state"],
            bool(row["collected"]),
            row["sequence"],
        )

    @staticmethod
    def _receipt(row: Row) -> Receipt:
        offer = pb.AttemptOffer.FromString(row["offer"])
        return Receipt(
            row["request"],
            row["submission"],
            row["capture"],
            offer.invocation_spec_digest,
            row["accepted_ms"],
            row["worker_id"],
            row["accepted_boot"],
            row["publication_authorization_id"],
        )

    def _event(self, db: Journal, row: Row, kind: str, body: bytes) -> None:
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
        capture_document: bytes = b"",
        preparation: bytes = b"",
        worker_id: str = "",
        publication_authorization_id: str = "",
    ) -> Receipt:
        """Atomically accept a verified capture/root and its first existing attempt."""
        Workspace.owner(owner)
        _id(submission_id)
        if publication_authorization_id:
            try:
                identity = uuid.UUID(publication_authorization_id)
                valid = identity.int != 0 and str(identity) == publication_authorization_id
            except ValueError:
                valid = False
            if not valid:
                raise WorkspaceRefusal("publication authorization must be a canonical nonzero UUID")
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
                    row["capture_document"],
                    row["preparation"],
                    row["publication_authorization_id"],
                ) != (
                    submission_id,
                    offer.request_id,
                    capture_digest,
                    raw,
                    max_attempts,
                    retry_delay_ms,
                    capture_document,
                    preparation,
                    publication_authorization_id,
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
                "ordinal,generation,state,max_attempts,retry_delay_ms,"
                "capture_document,preparation,worker_id,accepted_boot,"
                "publication_authorization_id) "
                "VALUES(?,?,?,?,?,?,1,1,'queued',?,?,?,?,?,?,?)",
                (
                    owner,
                    offer.request_id,
                    submission_id,
                    capture_digest,
                    raw,
                    self.clock_ms(),
                    max_attempts,
                    retry_delay_ms,
                    capture_document,
                    preparation,
                    worker_id,
                    worker_boot,
                    publication_authorization_id,
                ),
            )
            row = self._row(db, owner, offer.request_id)
            self._event(db, row, "accepted", canonical_json.encode({"generation": 1}))
            return self._receipt(row)

    def publication_authorization(self, owner: str, request: str) -> str:
        with self.workspace.locked() as db:
            return str(self._row(db, owner, request)["publication_authorization_id"])

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
                    "AND (state IN ('queued','running','pausing','canceling','retrying') "
                    "OR (desired='cancel' AND retention_waived=0)) "
                    "ORDER BY accepted_ms,request",
                    (owner,),
                )
            )

    def active(self, owner: str) -> bool:
        with self.workspace.locked() as db:
            return (
                db.execute(
                    "SELECT 1 FROM executions WHERE owner=? "
                    "AND state IN ('queued','running','pausing','canceling','retrying') LIMIT 1",
                    (owner,),
                ).fetchone()
                is not None
            )

    def preparation(self, owner: str, request: str) -> dict[str, Any]:
        with self.workspace.locked() as db:
            raw = self._row(db, owner, request)["preparation"]
            return cast(dict[str, Any], canonical_json.decode(raw)) if raw else {}

    def assigned_here(self, owner: str, request: str, ordinal: int, worker_boot: str) -> bool:
        from .workspace_recovery import process_identity

        with self.workspace.locked() as db:
            row = db.execute(
                "SELECT process_owner,worker_boot FROM attempts "
                "WHERE owner=? AND request=? AND ordinal=?",
                (owner, request, ordinal),
            ).fetchone()
            return row is not None and (row["process_owner"], row["worker_boot"]) == (
                process_identity(),
                worker_boot,
            )

    def capture(self, owner: str, request: str) -> dict[str, Any]:
        root = self.capture_root(owner, request)
        with self.workspace.locked() as db:
            raw = self._row(db, owner, root)["capture_document"]
            return documents.read(raw, pb.MachineExecutionCapture)

    def capture_root(self, owner: str, request: str) -> str:
        """Children reference their original capture instead of copying its package inventory."""
        with self.workspace.locked() as db:
            seen: set[str] = set()
            while request not in seen and len(seen) <= 32:
                seen.add(request)
                if self._row(db, owner, request)["capture_document"]:
                    return request
                parent = db.execute(
                    "SELECT parent_request FROM execution_calls WHERE owner=? AND child_request=?",
                    (owner, request),
                ).fetchone()
                if parent is None:
                    break
                request = parent[0]
        raise WorkspaceRefusal("execution has no bounded original capture ancestry")

    def abandon(self, owner: str, request: str) -> bool:
        """Relinquish this stopped root through existing native release operations."""
        from . import workspace_finalize
        from .workspace_calls import Calls

        with self.workspace.locked() as db:
            row = self._row(db, owner, request)
            if (
                row["desired"] != "cancel"
                or row["state"] not in TERMINAL
                or row["retention_waived"]
                or Calls.unsettled_effects_in(db, owner, request)
            ):
                return False
            attempts = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=?", (owner, request)
            ).fetchall()
            if any(not attempt["outcome"] for attempt in attempts):
                return False
            weights = db.execute(
                "SELECT * FROM weights WHERE owner=? AND request=? AND state<>'released'",
                (owner, request),
            ).fetchall()
        for weight in weights:
            if weight["native_digest"]:
                self.workspace.release_result(
                    owner,
                    pb.DerivedResultReleaseRequest(
                        weights_transaction_id=weight["id"],
                        tensorfs_receipt_digest=weight["native_digest"],
                    ),
                )
            else:
                attempt = next(
                    attempt for attempt in attempts if attempt["ordinal"] == weight["ordinal"]
                )
                workspace_finalize.finalize(
                    self.workspace,
                    owner,
                    pb.WeightsFinalizeRequest(
                        owner_authority_scope=owner,
                        request_id=request,
                        invocation_spec_digest=weight["spec"],
                        invocation_spec_canonical_bytes=attempt["invocation"],
                        output_slot=weight["slot"],
                        disposition=pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED,
                    ),
                )
        for attempt in attempts:
            self.workspace.acknowledge(
                owner,
                pb.AttemptOutcomeAck(
                    request_id=request,
                    attempt_ordinal=attempt["ordinal"],
                    invocation_spec_digest=attempt["spec"],
                    outcome_id=attempt["outcome_id"],
                    outcome_digest=attempt["outcome_digest"],
                    retain_work=False,
                ),
            )
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, request)
            if not row["retention_waived"]:
                db.execute(
                    "UPDATE executions SET retention_waived=1 WHERE owner=? AND request=?",
                    (owner, request),
                )
                self._event(
                    db,
                    row,
                    "retention_released",
                    canonical_json.encode({"collected": bool(row["collected"])}),
                )
        return True

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
            if not attempt["outcome"] or row["state"] in TERMINAL or row["state"] == "retrying":
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
            if state == "failed" and self._retryable(row, body):
                state = "retrying"
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

    def _retryable(self, row: Row, body: dict[str, Any]) -> bool:
        cause = body.get("cause", {})
        if (
            row["desired"] != "run"
            or row["ordinal"] >= row["max_attempts"]
            or body.get("status") not in (pb.OUTCOME_STATUS_FAILED, pb.OUTCOME_STATUS_ABANDONED)
            or cause.get("code")
            not in (pb.CAUSE_CODE_EXECUTOR_FAULT, pb.CAUSE_CODE_EXECUTOR_INVALIDATED)
            or cause.get("origin") not in (pb.CAUSE_ORIGIN_WORKER, pb.CAUSE_ORIGIN_RUNTIME)
        ):
            return False
        spec = documents.read(
            pb.AttemptOffer.FromString(row["offer"]).invocation_spec_canonical_bytes,
            pb.InvocationSpec,
        )
        deadline = int(spec.get("deadline_unix_ms", 0))
        return not deadline or self.clock_ms() < deadline

    def retry_due(self, owner: str, *, worker_boot: str = "") -> None:
        """A retryable attempt is not a final execution failure visible to observers."""
        with self.workspace.locked() as db:
            rows = db.execute(
                "SELECT e.*,a.outcome FROM executions e JOIN attempts a "
                "ON a.owner=e.owner AND a.request=e.request AND a.ordinal=e.ordinal "
                "WHERE e.owner=? AND e.state='retrying' AND e.desired='run'",
                (owner,),
            ).fetchall()
        for row in rows:
            outcome = documents.read(row["outcome"], pb.AttemptOutcomeBody)
            if not self._retryable(row, outcome):
                with self.workspace.locked() as db, transaction(db):
                    current = self._row(db, owner, row["request"])
                    if (
                        current["generation"] == row["generation"]
                        and current["state"] == "retrying"
                    ):
                        db.execute(
                            "UPDATE executions SET state='failed' WHERE owner=? AND request=?",
                            (owner, row["request"]),
                        )
                        self._event(
                            db,
                            current,
                            "retry_stopped",
                            canonical_json.encode({"reason": "accepted retry or deadline bound"}),
                        )
                continue
            if row["finished_ms"] + row["retry_delay_ms"] > self.clock_ms():
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
                raise StaleExecutionGeneration("execution command generation is stale")
            state, ordinal, desired = row["state"], row["ordinal"], row["desired"]
            if action == "pause" and state in ("succeeded", "canceled"):
                result = self._state(row)
                db.execute(
                    "INSERT INTO execution_commands VALUES(?,?,?,?,?)",
                    (owner, request, command_id, intent, canonical_json.encode(result.document())),
                )
                return result
            if action == "resume":
                if state not in ("paused", "failed") and not (_automatic and state == "retrying"):
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
            elif action == "cancel":
                state, desired = "succeeded" if state == "succeeded" else "canceled", "cancel"
            generation = row["generation"] + 1
            db.execute(
                "UPDATE executions SET state=?,desired=?,ordinal=?,generation=?,"
                "collected=CASE WHEN ? THEN 0 ELSE collected END "
                "WHERE owner=? AND request=?",
                (state, desired, ordinal, generation, action == "resume", owner, request),
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
        if ordinal is None and state.state not in TERMINAL:
            raise WorkspaceRefusal(
                "execution is not terminal; inspect a specific historical attempt"
            )
        selected = state.attempt_ordinal if ordinal is None else ordinal
        with self.workspace.locked() as db:
            execution = self._row(db, owner, request)
            row = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, request, selected),
            ).fetchone()
            if row is None or not row["outcome"]:
                raise WorkspaceRefusal("execution has no durable terminal to collect")
            outcome = self.workspace._outcome_frame(row)
            verify = not execution["collected"] and not execution["retention_waived"]
        if verify:
            self._verify_outputs(outcome)
        return outcome

    def _verify_outputs(self, outcome: pb.AttemptOutcome) -> None:
        """A journal alone cannot prove an uncollected native result remains present."""
        from cozy_runtime.internal import fill

        body = documents.read(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        if body["status"] != pb.OUTCOME_STATUS_SUCCEEDED:
            return
        manifests = set()
        for reference in body.get("weights_receipts", []):
            receipt = documents.read(
                base64.b64decode(reference["weights_receipt_canonical_bytes"], validate=True),
                pb.WeightsReceipt,
            )
            native = canonical_json.decode(
                base64.b64decode(receipt["tensorfs_receipt_canonical_bytes"], validate=True)
            )
            manifests.add("sha256:" + native["manifest"]["sha256"])
        outputs = list(body.get("output_manifest", {}).get("outputs", []))
        if blob := body.get("result", {}).get("result_blob"):
            outputs.append(blob)
        for output in outputs:
            if native_tree := output.get("native_tree"):
                manifests.add(native_tree["manifest"]["digest"])
        if not manifests:
            return
        store = fill.tensorfs_module().Store.open(str(self.workspace.store_root))
        leases = []
        try:
            for manifest in sorted(manifests):
                leases.append(store.acquire_manifest(manifest))
        finally:
            for lease in leases:
                lease.release()

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
                    "retention_waived=0 AND "
                    "(state NOT IN ('succeeded','canceled') OR collected=0) LIMIT 1",
                    (owner,),
                ).fetchone()
                is not None
            )
