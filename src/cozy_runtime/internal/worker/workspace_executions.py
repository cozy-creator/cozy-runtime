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

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.machine_publication import PublicationRefusal, output_destination
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
#: Accepted and not terminal: work this machine still owes.
OPEN = "('queued','running','pausing','canceling')"


class StaleExecutionGeneration(WorkspaceRefusal):
    """A rejected command, distinct from a lost reply to an accepted command."""

    code = "execution_generation_stale"


class ExecutionChanged(WorkspaceRefusal):
    """The row moved under a writer's feet (a control, a stop, a dispatch): re-read and act
    again. A fence, never a failure of the execution."""


class ExecutionWorkspaceRefusal(WorkspaceRefusal):
    """The expected journal is absent/changed; this is never proof of nonacceptance."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class Receipt:
    request_id: str
    submission_id: str
    capture_digest: bytes
    invocation_spec_digest: bytes
    accepted_at_ms: int
    worker_id: str
    worker_boot_id: str
    execution_workspace_id: str
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


class ExecutionRow(msgspec.Struct, frozen=True):
    """One `executions` row as the supervision logic reads it, decoded once."""

    request: str
    ordinal: int
    generation: int
    state: str
    desired: str
    collected: bool
    retention_waived: bool
    #: the durable acceptance order (rowid): a root's GPU priority
    priority: int
    offer: bytes
    preparation: bytes
    #: its current attempt reached an executor (the dispatch mark)
    dispatched: bool

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    def attempt_offer(self) -> pb.AttemptOffer:
        offer = pb.AttemptOffer.FromString(self.offer)
        offer.attempt_ordinal = self.ordinal
        return offer


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


def _lost_before_dispatch(db: Journal, row: Row, body: pb.AttemptOutcomeBody) -> bool:
    """The Runtime process died before dispatching this attempt (`recover`'s INFRA
    outcome) and no earlier attempt of the execution was lost the same way."""
    if (
        row["desired"] != "run"
        or body.execution_started
        or body.status != pb.OUTCOME_STATUS_ABANDONED
        or body.cause.origin != pb.CAUSE_ORIGIN_INFRA
    ):
        return False
    earlier = db.execute(
        "SELECT outcome FROM attempts WHERE owner=? AND request=? AND ordinal<? AND outcome<>x''",
        (row["owner"], row["request"], row["ordinal"]),
    ).fetchall()
    return not any(
        documents.parse(prior["outcome"], pb.AttemptOutcomeBody).cause.origin
        == pb.CAUSE_ORIGIN_INFRA
        for prior in earlier
    )


def destination_of(offer: pb.AttemptOffer) -> str:
    """The repository a root's weights outputs publish to, as its owner granted it."""
    try:
        return output_destination(output.url for output in offer.grant.outputs)
    except PublicationRefusal as exc:
        raise WorkspaceRefusal(f"output destination is invalid: {exc.code}") from exc


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
        #: told (request, kind) of each non-log event as it is journaled, inside its
        #: transaction: it must not block, take locks or read the journal
        self.changed: Callable[[str, str], None] = lambda request, kind: None

    @property
    def workspace_id(self) -> str:
        with self.workspace.locked() as db:
            return self._workspace_id_in(db)

    @staticmethod
    def _workspace_id_in(db: Journal) -> str:
        return str(
            db.execute(
                "SELECT execution_workspace_id FROM workspace_state WHERE singleton=1"
            ).fetchone()[0]
        )

    @classmethod
    def require_workspace_in(cls, db: Journal, expected: str) -> str:
        if not expected:
            raise ExecutionWorkspaceRefusal(
                "execution_workspace_required", "execution requires its pinned workspace identity"
            )
        actual = cls._workspace_id_in(db)
        if expected != actual:
            raise ExecutionWorkspaceRefusal(
                "execution_workspace_changed", "execution workspace was replaced or does not match"
            )
        return actual

    def require_workspace(self, expected: str) -> None:
        with self.workspace.locked() as db:
            self.require_workspace_in(db, expected)

    def accepted(
        self, owner: str, submission_id: str, request: str, expected: str
    ) -> Receipt | None:
        """The receipt of an execution this submission already created, or None."""
        with self.workspace.locked() as db:
            workspace_id = self.require_workspace_in(db, expected)
            row = db.execute(
                "SELECT * FROM executions WHERE owner=? AND (submission=? OR request=?)",
                (owner, submission_id, request),
            ).fetchone()
            if row is None:
                return None
            if (row["submission"], row["request"]) != (submission_id, request):
                raise WorkspaceRefusal("execution submission identity changed")
            return self._receipt(row, workspace_id)

    def submission_absent(self, owner: str, request: str, expected: str) -> bool:
        """Only the pinned journal can establish that this submission was not accepted."""
        with self.workspace.locked() as db:
            self.require_workspace_in(db, expected)
            return (
                db.execute(
                    "SELECT 1 FROM executions WHERE owner=? AND request=?", (owner, request)
                ).fetchone()
                is None
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
    def _receipt(row: Row, workspace_id: str) -> Receipt:
        offer = pb.AttemptOffer.FromString(row["offer"])
        return Receipt(
            row["request"],
            row["submission"],
            row["capture"],
            offer.invocation_spec_digest,
            row["accepted_ms"],
            row["worker_id"],
            row["accepted_boot"],
            workspace_id,
            row["publication_authorization_id"],
        )

    def _event(
        self, db: Journal, row: Row, kind: str, body: bytes, at_ms: int | None = None
    ) -> int:
        if len(body) > MAX_EVENT_BYTES:
            raise WorkspaceRefusal("execution event exceeds its metadata bound")
        canonical_json.decode(body)
        sequence = int(row["sequence"]) + 1
        db.execute(
            "UPDATE executions SET sequence=? WHERE owner=? AND request=?",
            (sequence, row["owner"], row["request"]),
        )
        at = self.clock_ms() if at_ms is None else at_ms
        db.execute(
            "INSERT INTO execution_events VALUES(?,?,?,?,?,?,?)",
            (row["owner"], row["request"], sequence, row["ordinal"], at, kind, body),
        )
        if kind != "log":
            self.changed(row["request"], kind)
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
        return sequence

    def submit(
        self,
        owner: str,
        submission_id: str,
        capture_digest: bytes,
        offer: pb.AttemptOffer,
        *,
        expected_execution_workspace_id: str,
        result_schema: bytes = b"",
        worker_boot: str = "",
        memoize: bool = False,
        capture_document: bytes = b"",
        preparation: bytes = b"",
        worker_id: str = "",
        publication_authorization_id: str = "",
        owner_memo: bool = False,
    ) -> Receipt:
        """Atomically accept a verified capture/root and its first existing attempt."""
        self.require_workspace(expected_execution_workspace_id)
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
        self.workspace.validate_offer(offer)
        if destination_of(offer) and (
            not publication_authorization_id
            or not documents.parse(
                offer.invocation_spec_canonical_bytes, pb.InvocationSpec
            ).HasField("job")
        ):
            raise WorkspaceRefusal(
                "an output destination needs a job root with publication authority"
            )
        if offer.attempt_ordinal != 1 or len(capture_digest) != 32:
            raise WorkspaceRefusal("new execution needs attempt one and an exact capture digest")
        self.workspace.recover(owner)
        raw = _offer_bytes(offer)
        with self.workspace.locked() as db, transaction(db):
            workspace_id = self.require_workspace_in(db, expected_execution_workspace_id)
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
                    row["capture_document"],
                    row["preparation"],
                    row["publication_authorization_id"],
                ) != (
                    submission_id,
                    offer.request_id,
                    capture_digest,
                    raw,
                    capture_document,
                    preparation,
                    publication_authorization_id,
                ):
                    raise WorkspaceRefusal("execution submission identity changed")
                return self._receipt(row, workspace_id)
            if db.execute(
                "SELECT 1 FROM attempts WHERE owner=? AND request=?", (owner, offer.request_id)
            ).fetchone():
                raise WorkspaceRefusal("request already belongs to another execution path")
            from .machine_checkpoint_inputs import check_submit
            from .workspace_input_trees import verify_adoption

            verify_adoption(db, owner, offer)
            check_submit(db, owner, offer)
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
                "ordinal,generation,state,"
                "capture_document,preparation,worker_id,accepted_boot,"
                "publication_authorization_id,owner_memo) "
                "VALUES(?,?,?,?,?,?,1,1,'queued',?,?,?,?,?,?)",
                (
                    owner,
                    offer.request_id,
                    submission_id,
                    capture_digest,
                    raw,
                    self.clock_ms(),
                    capture_document,
                    preparation,
                    worker_id,
                    worker_boot,
                    publication_authorization_id,
                    owner_memo,
                ),
            )
            row = self._row(db, owner, offer.request_id)
            self._event(db, row, "accepted", canonical_json.encode({"generation": 1}))
            return self._receipt(row, workspace_id)

    def destination(self, owner: str, request: str) -> str:
        with self.workspace.locked() as db:
            return destination_of(
                pb.AttemptOffer.FromString(self._row(db, owner, request)["offer"])
            )

    def publication_authorization(self, owner: str, request: str) -> str:
        with self.workspace.locked() as db:
            seen: set[str] = set()
            claimed: set[str] = set()
            while request not in seen and len(seen) <= 32:
                seen.add(request)
                grant = str(self._row(db, owner, request)["publication_authorization_id"])
                if grant:
                    claimed.add(grant)
                parent = db.execute(
                    "SELECT parent_request FROM execution_calls WHERE owner=? AND child_request=?",
                    (owner, request),
                ).fetchone()
                if parent is None:
                    if claimed and claimed != {grant}:
                        raise WorkspaceRefusal("child publication authority differs from its root")
                    return grant
                request = parent[0]
        raise WorkspaceRefusal("publication authority has no bounded root ancestry")

    def owns(self, owner: str, request: str) -> bool:
        with self.workspace.locked() as db:
            return (
                db.execute(
                    "SELECT 1 FROM executions WHERE owner=? AND request=?", (owner, request)
                ).fetchone()
                is not None
            )

    def row(self, owner: str, request: str) -> ExecutionRow | None:
        with self.workspace.locked() as db:
            row = db.execute(
                "SELECT e.rowid,e.*,a.state AS attempt_state FROM executions e "
                "LEFT JOIN attempts a "
                "ON a.owner=e.owner AND a.request=e.request AND a.ordinal=e.ordinal "
                "WHERE e.owner=? AND e.request=?",
                (owner, request),
            ).fetchone()
        if row is None:
            return None
        return ExecutionRow(
            request=row["request"],
            ordinal=row["ordinal"],
            generation=row["generation"],
            state=row["state"],
            desired=row["desired"],
            collected=bool(row["collected"]),
            retention_waived=bool(row["retention_waived"]),
            priority=row["rowid"],
            offer=row["offer"],
            preparation=row["preparation"],
            dispatched=row["attempt_state"] == "running",
        )

    def owed(self, owner: str, worker_id: str) -> tuple[str, ...]:
        """Every execution this worker still owes work: open, or canceled and still
        retaining custody that its release must end. Boot reconcile gives each a unit."""
        Workspace.owner(owner)
        with self.workspace.locked() as db:
            return tuple(
                row[0]
                for row in db.execute(
                    "SELECT request FROM executions WHERE owner=? AND worker_id=? "
                    f"AND (state IN {OPEN} "
                    "OR (desired='cancel' AND retention_waived=0)) "
                    "ORDER BY rowid",
                    (owner, worker_id),
                )
            )

    def held(self, owner: str, worker_id: str) -> tuple[str, ...]:
        """Terminal executions whose retention a boot releases."""
        Workspace.owner(owner)
        with self.workspace.locked() as db:
            return tuple(
                row["request"]
                for row in db.execute(
                    "SELECT * FROM executions WHERE owner=? AND worker_id=? "
                    f"AND state IN {tuple(sorted(TERMINAL))} AND retention_waived=0 "
                    "ORDER BY rowid",
                    (owner, worker_id),
                )
                if row["desired"] == "cancel" or _released_at_boot(row)
            )

    def scheduling_root(self, owner: str, request: str) -> tuple[str, str]:
        """(root, parent): the original accepted ancestor orders descendants."""
        with self.workspace.locked() as db:
            seen: list[str] = []
            while request not in seen and len(seen) <= 32:
                seen.append(request)
                parent = db.execute(
                    "SELECT parent_request FROM execution_calls WHERE owner=? AND child_request=?",
                    (owner, request),
                ).fetchone()
                if parent is None:
                    self._row(db, owner, request)
                    return request, seen[1] if len(seen) > 1 else ""
                request = parent[0]
        raise WorkspaceRefusal("execution has no bounded scheduling ancestry")

    def record(self, owner: str, request: str, kind: str, document: dict[str, Any]) -> None:
        """Journal one bounded worker observation on an execution, terminal or not."""
        with self.workspace.locked() as db, transaction(db):
            self._event(db, self._row(db, owner, request), kind, canonical_json.encode(document))

    def active(self, owner: str) -> bool:
        with self.workspace.locked() as db:
            return (
                db.execute(
                    f"SELECT 1 FROM executions WHERE owner=? AND state IN {OPEN} LIMIT 1",
                    (owner,),
                ).fetchone()
                is not None
            )

    def retained_installations(self) -> set[str]:
        """Installation ids an uncollected execution may still need to reopen."""
        with self.workspace.locked(allow_frozen=True) as db:
            rows = db.execute(
                "SELECT preparation FROM executions WHERE collected=0 AND preparation<>x''"
            ).fetchall()
        return {
            str(identifier)
            for row in rows
            for identifier in canonical_json.decode(row["preparation"]).get("installations", {})
        }

    def preparation(self, owner: str, request: str) -> dict[str, Any]:
        with self.workspace.locked() as db:
            raw = self._row(db, owner, request)["preparation"]
            return cast(dict[str, Any], canonical_json.decode(raw)) if raw else {}

    def wire_minor(self, owner: str, request: str) -> int:
        """The wire minor this execution was admitted under: its root's submitting Claim.

        Negotiated per operation and retained with the root, so a Runtime or Host restart,
        which forgets every control stream, cannot change what accepted work may do.
        """
        root, _ = self.scheduling_root(owner, request)
        prepared = self.preparation(owner, root)
        if "wire_minor" in prepared:
            return int(prepared["wire_minor"])
        # A root accepted before the minor was retained: its owner's prepared state names it.
        state = pb.DesiredWorkerState.FromString(base64.b64decode(prepared.get("state", "")))
        return state.wire_minor

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

    def release(self, owner: str, request: str, *, boot: bool = False) -> bool:
        """Relinquish what this terminal root retains through the native release operations,
        once no effect of it is in doubt: when it is canceled, and at `boot` also when it is
        collected or failed (a boot ends every hold no one will read)."""
        from . import machine_models, workspace_finalize
        from .workspace_calls import Calls

        with self.workspace.locked() as db:
            row = self._row(db, owner, request)
            if (
                row["state"] not in TERMINAL
                or row["retention_waived"]
                or not (row["desired"] == "cancel" or (boot and _released_at_boot(row)))
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
        machine_models.release(self.workspace, owner, request)
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
        """Stop work that never reached an executor."""
        self._close_undispatched(
            owner,
            request,
            pb.OUTCOME_STATUS_CANCELED,
            pb.CAUSE_CODE_DEADLINE_EXPIRED if deadline else pb.CAUSE_CODE_CLIENT_CANCEL,
            "deadline expired before dispatch" if deadline else "stopped before dispatch",
        )

    def fail(self, owner: str, request: str, why: str) -> None:
        """End an execution no thread here holds FAILED with `why` (printable ASCII): an
        undispatched attempt, or a dispatched one whose holder is gone."""
        self._close_undispatched(
            owner, request, pb.OUTCOME_STATUS_FAILED, pb.CAUSE_CODE_LOCAL_SAFETY, why, lost=True
        )

    def _close_undispatched(
        self,
        owner: str,
        request: str,
        status: pb.OutcomeStatus,
        cause: pb.CauseCode,
        message: str,
        *,
        lost: bool = False,
    ) -> None:
        """Record the current attempt's terminal if, and only if, it was never dispatched
        (or, `lost`, was dispatched and no one holds it): a compare-and-set against the
        dispatch mark, built from the attempt row's own columns. A held attempt ends only
        through its runner (`ExecutionChanged` otherwise)."""
        states = "('accepted','running')" if lost else "('accepted')"
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, request)
            attempt = db.execute(
                "SELECT ordinal,spec,state FROM attempts WHERE owner=? AND request=? "
                f"AND ordinal=? AND state IN {states} AND outcome=x''",
                (owner, request, row["ordinal"]),
            ).fetchone()
            if row["state"] in TERMINAL or attempt is None:
                raise ExecutionChanged("the execution is dispatched or already terminal")
            body, digest = documents.identity(
                pb.AttemptOutcomeBody(
                    request_id=request,
                    attempt_ordinal=attempt["ordinal"],
                    invocation_spec_digest=documents.spell(attempt["spec"]),
                    status=status,
                    safe_message=message[:4096],
                    cause=pb.OutcomeCause(
                        code=cause, origin=pb.CAUSE_ORIGIN_WORKER, detail=message[:1024]
                    ),
                    execution_started=attempt["state"] == "running",
                )
            )
            db.execute(
                "UPDATE attempts SET state='outcome',fenced=1,outcome_id=?,outcome_digest=?,"
                "outcome=? WHERE owner=? AND request=? AND ordinal=?",
                ("out-" + digest.hex()[:24], digest, body, owner, request, attempt["ordinal"]),
            )
        self.reconcile(owner, request)

    def reconcile(self, owner: str, request: str, *, requeue: str = "") -> None:
        """Project the current attempt's durable outcome onto its execution. Every outcome is
        final except the one kind of work the machine never got: an attempt whose Runtime
        process died before dispatching it. Boot reconcile passes its `requeue` worker boot,
        and that attempt runs again as ordinal+1, once per execution."""
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
            body = documents.parse(attempt["outcome"], pb.AttemptOutcomeBody)
            state = (
                "paused"
                if row["desired"] == "pause"
                else "canceled"
                if row["desired"] == "cancel" or body.status == pb.OUTCOME_STATUS_CANCELED
                else "succeeded"
                if body.status == pb.OUTCOME_STATUS_SUCCEEDED
                else "failed"
            )
            if requeue and state == "failed" and _lost_before_dispatch(db, row, body):
                offer = pb.AttemptOffer.FromString(row["offer"])
                offer.attempt_ordinal = row["ordinal"] + 1
                self.workspace.accept_in(
                    db,
                    owner,
                    offer,
                    result_schema=attempt["result_schema"],
                    worker_boot=requeue,
                    memoize=bool(attempt["memoize"]),
                )
                db.execute(
                    "UPDATE executions SET state='queued',ordinal=?,generation=generation+1 "
                    "WHERE owner=? AND request=?",
                    (offer.attempt_ordinal, owner, request),
                )
                self._event(
                    db,
                    self._row(db, owner, request),
                    "requeued",
                    canonical_json.encode({"lost_ordinal": row["ordinal"]}),
                )
                return
            db.execute(
                "UPDATE executions SET state=? WHERE owner=? AND request=?",
                (state, owner, request),
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
        with self.workspace.locked() as db:
            return self._state(self._row(db, owner, request))

    def offer(self, owner: str, request: str) -> pb.AttemptOffer:
        """Return pinned work for Worker; this does not execute or pick a machine."""
        with self.workspace.locked() as db:
            row = self._row(db, owner, request)
            if row["state"] not in ("queued", "running") or row["desired"] != "run":
                raise ExecutionChanged("execution is not dispatchable")
            offer = pb.AttemptOffer.FromString(row["offer"])
            offer.attempt_ordinal = row["ordinal"]
            return offer

    def dispatched(self, owner: str, request: str, ordinal: int) -> None:
        """Execution STARTS: the attempt row moves accepted -> running and the execution
        queued -> running in one transaction, immediately before the executor is handed the
        command. `running` means this and nothing earlier; everything before it is work the
        machine never ran. A terminal recorded first wins (`ExecutionChanged`)."""
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, request)
            moved = db.execute(
                "UPDATE attempts SET state='running' WHERE owner=? AND request=? AND ordinal=? "
                "AND state='accepted' AND fenced=0 AND outcome=x''",
                (owner, request, ordinal),
            ).rowcount
            if not moved or row["ordinal"] != ordinal or row["state"] not in TERMINAL | {"queued"}:
                raise ExecutionChanged("the attempt was stopped before dispatch")
            if row["state"] == "queued":
                db.execute(
                    "UPDATE executions SET state='running' WHERE owner=? AND request=?",
                    (owner, request),
                )
                self._event(
                    db, row, "running", canonical_json.encode({"generation": row["generation"]})
                )

    def progress(self, owner: str, request: str, ordinal: int, document: dict[str, Any]) -> None:
        """Telemetry for a held execution, if any; recorded with the next journal access."""
        raw = canonical_json.encode(document)
        if len(raw) > MAX_EVENT_BYTES:
            raise WorkspaceRefusal("execution event exceeds its metadata bound")
        kind = "progress" if document.get("type") == "progress" else "log"
        at_ms = self.clock_ms()

        def write(db: Journal) -> None:
            row = db.execute(
                "SELECT * FROM executions WHERE owner=? AND request=?", (owner, request)
            ).fetchone()
            if row is not None and row["ordinal"] == ordinal and row["state"] not in TERMINAL:
                self._event(db, row, kind, raw, at_ms)

        self.workspace.defer(write)

    def notice(self, owner: str, request: str, kind: str, document: dict[str, Any]) -> int:
        """A durable effect fact on the execution in any state: its owner acts on it."""
        raw = canonical_json.encode(document)
        with self.workspace.locked() as db, transaction(db):
            return self._event(db, self._row(db, owner, request), kind, raw)

    def owner_memo(self, owner: str, request: str) -> bool:
        with self.workspace.locked() as db:
            return bool(self._row(db, owner, request)["owner_memo"])

    def wait_events(
        self, owner: str, request: str, after: int, limit: int, gone: Callable[[], bool]
    ) -> EventPage:
        """The next page after `after`, waiting for one while the caller is here. A terminal
        execution answers at once: its observer reads what it has and collects."""
        changes = self.workspace.changes()
        while True:
            with changes.condition:
                version = changes.version
            page = self.events(owner, request, after, limit)
            if page.events or gone() or self.status(owner, request).state in TERMINAL:
                return page
            with changes.condition:
                while changes.version == version and not gone():
                    changes.condition.wait()

    def events(self, owner: str, request: str, after: int = 0, limit: int = MAX_PAGE) -> EventPage:
        if not 0 <= after < 1 << 63 or not 1 <= limit <= MAX_PAGE:
            raise WorkspaceRefusal("execution event cursor or page bound is invalid")
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
    ) -> State:
        _id(command_id)
        command_id = "user:" + command_id
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
                answered = canonical_json.decode(previous["response"])
                known = State.__dataclass_fields__
                return State(**{k: v for k, v in answered.items() if k in known})
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
                if state not in ("paused", "failed"):
                    raise WorkspaceRefusal("only paused or failed execution can resume")
                attempt = db.execute(
                    "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                    (owner, request, ordinal),
                ).fetchone()
                if attempt is None or not attempt["outcome"]:
                    raise WorkspaceRefusal("previous execution has not durably stopped")
                deadline = documents.parse(
                    attempt["invocation"], pb.InvocationSpec
                ).deadline_unix_ms
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
        outcome, verify, schema = self._terminal(owner, request, ordinal)
        if verify:
            self._verify_outputs(owner, outcome, schema)
        return outcome

    def outcome(self, owner: str, request: str, ordinal: int | None = None) -> pb.AttemptOutcome:
        """The exact terminal, without the retained-output check collection makes."""
        return self._terminal(owner, request, ordinal)[0]

    def _terminal(
        self, owner: str, request: str, ordinal: int | None
    ) -> tuple[pb.AttemptOutcome, bool, bytes]:
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
            return outcome, verify, row["result_schema"]

    def _verify_outputs(self, owner: str, outcome: pb.AttemptOutcome, schema: bytes) -> None:
        """A journal alone cannot prove an uncollected native result remains present."""
        from cozy_runtime.internal import fill

        body = documents.parse(outcome.outcome_canonical_bytes, pb.AttemptOutcomeBody)
        if body.status != pb.OUTCOME_STATUS_SUCCEEDED:
            return
        if schema and body.result.inline_result:
            from . import machine_models

            records = machine_models.result_records(
                self.workspace,
                owner,
                outcome.request_id,
                canonical_json.decode(body.result.inline_result),
                canonical_json.decode(schema),
            )
            supplied = list(body.result.retained_models)
            expected = [
                pb.RetainedModelResult(
                    result_pointer=pointer,
                    model_artifact_canonical_bytes=artifact,
                    retention=retention,
                )
                for pointer, artifact, retention in records
            ]
            if supplied and supplied != expected:
                raise WorkspaceRefusal("terminal Model custody differs from its typed result")
        manifests = set()
        for reference in body.weights_receipts:
            receipt = documents.parse(reference.weights_receipt_canonical_bytes, pb.WeightsReceipt)
            native = canonical_json.decode(receipt.tensorfs_receipt_canonical_bytes)
            manifests.add("sha256:" + native["manifest"]["sha256"])
        outputs = list(body.output_manifest.outputs)
        if body.result.HasField("result_blob"):
            outputs.append(body.result.result_blob)
        for output in outputs:
            if output.HasField("native_tree"):
                manifests.add(documents.spell(output.native_tree.manifest.digest))
        if not manifests:
            return
        store = fill.store(self.workspace.store_root)
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
            state = self._state(self._row(db, owner, ack.request_id))
        if ack.attempt_ordinal == state.attempt_ordinal and state.state in (
            "succeeded",
            "canceled",
        ):
            from . import machine_models

            machine_models.release(self.workspace, owner, ack.request_id)
        return state

    def retention_required(self, owner: str) -> bool:
        Workspace.owner(owner)
        with self.workspace.locked(allow_frozen=True) as db:
            if (
                db.execute("SELECT state FROM workspace_state WHERE singleton=1").fetchone()[0]
                != "active"
            ):
                return True
            return (
                db.execute(
                    "SELECT 1 FROM executions WHERE owner=? AND "
                    "retention_waived=0 AND "
                    "(state NOT IN ('succeeded','canceled') OR collected=0) LIMIT 1",
                    (owner,),
                ).fetchone()
                is not None
            )


def _released_at_boot(row: Row) -> bool:
    return bool(row["collected"]) or row["state"] in ("canceled", "failed")
