"""One privileged custody journal for every Runtime using a TensorFS workspace.

The file lock serializes whole journal/native transitions across worker processes.
Intent commits precede native mutations; an interrupted mutation is replayed by its
exact identity. This directory is separate from TensorFS's catalog and roots, and
is never placed in an ephemeral worker or author environment.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill
from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.internal.worker import derived_retention
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class WorkspaceRefusal(ValueError):
    """A stable workspace identity or custody invariant refused the operation."""


_authority: ContextVar[Callable[[], None]] = ContextVar("workspace_authority", default=lambda: None)


class Journal(sqlite3.Connection):
    lock_fd: int
    guard: Callable[[], None]

    def native[T](self, operation: Callable[[], T]) -> T:
        """Native owns object/GC locking; no workspace transaction spans its IO."""
        if self.in_transaction:
            raise RuntimeError("native work cannot run inside a workspace SQL transaction")
        fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        try:
            return operation()
        finally:
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
            self.guard()


_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")

_SCHEMA = """
CREATE TABLE workspace_state (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1),
 state TEXT NOT NULL CHECK(state IN ('active','frozen','imported')),
 migration_id TEXT NOT NULL DEFAULT ''
) STRICT;
INSERT INTO workspace_state(singleton,state) VALUES(1,'active');
CREATE TABLE workspace_imports (
 id TEXT PRIMARY KEY, record_count INTEGER NOT NULL, digest BLOB NOT NULL,
 owner TEXT NOT NULL, source_schema INTEGER NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('receiving','imported','active'))
) STRICT;
CREATE TABLE workspace_import_records (
 migration_id TEXT NOT NULL, ordinal INTEGER NOT NULL, kind TEXT NOT NULL, body BLOB NOT NULL,
 verified BLOB NOT NULL DEFAULT x'',
 PRIMARY KEY(migration_id,ordinal)
) STRICT;
CREATE TABLE attempts (
 owner TEXT NOT NULL, request TEXT NOT NULL, ordinal INTEGER NOT NULL,
 spec BLOB NOT NULL, invocation BLOB NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('accepted','running','outcome','released')),
 outcome_id TEXT NOT NULL DEFAULT '', outcome_digest BLOB NOT NULL DEFAULT x'',
 outcome BLOB NOT NULL DEFAULT x'', retain INTEGER NOT NULL DEFAULT 0,
 fenced INTEGER NOT NULL DEFAULT 0,
 worker_boot TEXT NOT NULL DEFAULT '', result_schema BLOB NOT NULL DEFAULT x'',
 transport_owner TEXT NOT NULL DEFAULT '',
 PRIMARY KEY(owner,request,ordinal)
) STRICT;
CREATE TABLE weights (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, request TEXT NOT NULL,
 ordinal INTEGER NOT NULL, spec BLOB NOT NULL, slot TEXT NOT NULL,
 declaration_digest BLOB NOT NULL, declaration BLOB NOT NULL,
 epoch INTEGER NOT NULL, state TEXT NOT NULL CHECK(state IN ('intent','receipt','released')),
 ready INTEGER NOT NULL DEFAULT 0, restore_checkpoint BLOB NOT NULL DEFAULT x'',
 checkpoint BLOB NOT NULL DEFAULT x'', receipt BLOB NOT NULL DEFAULT x'',
 receipt_digest BLOB NOT NULL DEFAULT x'', native_digest BLOB NOT NULL DEFAULT x'',
 manifest BLOB NOT NULL DEFAULT x'', manifest_length INTEGER NOT NULL DEFAULT 0,
 objects BLOB NOT NULL DEFAULT x'5b5d',
 UNIQUE(owner,request,spec,slot)
) STRICT;
CREATE TABLE holds (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, transaction_id TEXT NOT NULL,
 native_digest BLOB NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('retaining','held','releasing','released')),
 manifest BLOB NOT NULL DEFAULT x'', manifest_length INTEGER NOT NULL DEFAULT 0
) STRICT;
CREATE TABLE operation_cache (
 sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
 owner TEXT NOT NULL, key BLOB NOT NULL, body BLOB NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('retaining','ready','evicting','disabled')),
 UNIQUE(owner,key)
) STRICT;
CREATE TABLE operation_lookups (
 owner TEXT NOT NULL, consumer TEXT NOT NULL, key BLOB NOT NULL,
 cache_id TEXT NOT NULL, body BLOB NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('retaining','ready','missing','miss','released')),
 PRIMARY KEY(owner,consumer)
) STRICT;
"""


class Workspace:
    def __init__(self, store_root: Path):
        if not store_root.is_absolute() or store_root == Path("/"):
            raise WorkspaceRefusal("workspace requires the configured absolute store root")
        self.store_root = store_root.resolve()
        fill.tensorfs_module().Store.ensure(str(self.store_root))
        self.directory = self.store_root / ".cozy-workspace"
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise WorkspaceRefusal("workspace journal directory is not an owned directory")
        if self.directory.stat().st_uid != os.geteuid():
            raise WorkspaceRefusal("workspace journal belongs to another operating-system owner")
        os.chmod(self.directory, 0o700)
        for directory in (self.store_root, self.directory):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)

    @contextmanager
    def locked(self, *, allow_frozen: bool = False) -> Iterator[Journal]:
        lock = os.open(self.directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        db: Journal | None = None
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            guard = _authority.get()
            guard()
            path = self.directory / "journal.sqlite3"
            if path.is_symlink():
                raise WorkspaceRefusal("workspace journal cannot be a symlink")
            db = sqlite3.connect(path, isolation_level=None, factory=Journal)
            db.lock_fd = lock
            db.guard = guard
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                db.executescript("BEGIN IMMEDIATE;" + _SCHEMA + "PRAGMA user_version=1;COMMIT;")
            elif version != 1:
                raise WorkspaceRefusal("workspace journal schema is not supported")
            if (
                not allow_frozen
                and db.execute("SELECT state FROM workspace_state WHERE singleton=1").fetchone()[0]
                != "active"
            ):
                raise WorkspaceRefusal("workspace authority transfer is not active")
            yield db
        finally:
            if db is not None:
                db.close()
            os.close(lock)

    @contextmanager
    def authorized(self, guard: Callable[[], None]) -> Iterator[None]:
        token = _authority.set(guard)
        try:
            guard()
            yield
        finally:
            _authority.reset(token)

    @staticmethod
    def owner(value: str) -> None:
        if _OWNER.fullmatch(value) is None:
            raise WorkspaceRefusal("workspace owner is not a bounded authenticated identity")

    def accept(
        self,
        owner: str,
        offer: pb.AttemptOffer,
        *,
        result_schema: bytes = b"",
        worker_boot: str = "",
    ) -> pb.AttemptOutcome | None:
        self.owner(owner)
        if (
            _ID.fullmatch(offer.request_id) is None
            or not 0 < offer.attempt_ordinal < 1 << 63
            or len(offer.invocation_spec_digest) != 32
            or hashlib.sha256(offer.invocation_spec_canonical_bytes).digest()
            != offer.invocation_spec_digest
        ):
            raise WorkspaceRefusal("workspace attempt does not bind its exact invocation")
        documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
        with self.locked() as db:
            row = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, offer.request_id, offer.attempt_ordinal),
            ).fetchone()
            if row is not None:
                if (
                    row["spec"] != offer.invocation_spec_digest
                    or row["invocation"] != offer.invocation_spec_canonical_bytes
                ):
                    raise WorkspaceRefusal("workspace attempt was changed or released")
                if row["state"] in ("outcome", "released"):
                    return self._outcome_frame(row)
                return None
            db.execute(
                "INSERT INTO attempts(owner,request,ordinal,spec,invocation,state,"
                "worker_boot,result_schema) VALUES(?,?,?,?,?,'accepted',?,?)",
                (
                    owner,
                    offer.request_id,
                    offer.attempt_ordinal,
                    offer.invocation_spec_digest,
                    offer.invocation_spec_canonical_bytes,
                    worker_boot,
                    result_schema,
                ),
            )

        return None

    @staticmethod
    def _outcome_frame(row: sqlite3.Row) -> pb.AttemptOutcome:
        return pb.AttemptOutcome(
            request_id=row["request"],
            attempt_ordinal=row["ordinal"],
            invocation_spec_digest=row["spec"],
            outcome_id=row["outcome_id"],
            outcome_digest=row["outcome_digest"],
            outcome_canonical_bytes=row["outcome"],
        )

    def contains(self, owner: str, request: str, ordinal: int) -> bool:
        with self.locked() as db:
            return (
                db.execute(
                    "SELECT 1 FROM attempts WHERE (owner=? OR(owner='' AND transport_owner=?)) "
                    "AND request=? AND ordinal=?",
                    (owner, owner, request, ordinal),
                ).fetchone()
                is not None
            )

    def retained_outcomes(self, owner: str) -> list[pb.AttemptOutcome]:
        with self.locked(allow_frozen=True) as db:
            return [
                self._outcome_frame(row)
                for row in db.execute(
                    "SELECT * FROM attempts WHERE (owner=? OR(owner='' AND transport_owner=?)) "
                    "AND state='outcome' "
                    "ORDER BY request,ordinal",
                    (owner, owner),
                )
            ]

    def projected_weights(self, owner: str, worker_boot: str) -> list[dict[str, object]]:
        with self.locked(allow_frozen=True) as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT w.* FROM weights AS w JOIN attempts AS a ON "
                    "a.owner=w.owner AND a.request=w.request AND a.ordinal=w.ordinal "
                    "WHERE w.owner=? AND a.state<>'released' "
                    "AND (a.state='outcome' OR a.worker_boot=?) ORDER BY w.id",
                    (owner, worker_boot),
                )
            ]

    def retain(
        self, owner: str, request: pb.DerivedRetentionRequest, *, release: bool = False
    ) -> pb.DerivedRetentionResult:
        self.owner(owner)
        with self.locked() as db:
            return self._change_hold(db, owner, request, release=release)

    def release_result(
        self, owner: str, request: pb.DerivedResultReleaseRequest
    ) -> pb.DerivedResultReleaseResult:
        self.owner(owner)
        with self.locked() as db:
            row = db.execute(
                "SELECT * FROM weights WHERE id=?", (request.weights_transaction_id,)
            ).fetchone()
            if (
                row is None
                or row["owner"] != owner
                or row["native_digest"] != request.tensorfs_receipt_digest
            ):
                raise WorkspaceRefusal("original result is not owned by this workspace owner")
            # A durable tombstone prevents reopening while native disposal is
            # pending. Native disposal itself is exact and idempotent.
            db.execute(
                "UPDATE weights SET state='released' WHERE id=?", (request.weights_transaction_id,)
            )
            return db.native(
                lambda: derived_retention.release_result(request, tensorfs_root=self.store_root)
            )

    def outcome(self, owner: str, outcome: pb.AttemptOutcome) -> None:
        self.owner(owner)
        raw = outcome.outcome_canonical_bytes
        if hashlib.sha256(raw).digest() != outcome.outcome_digest:
            raise WorkspaceRefusal("outcome digest differs from exact bytes")
        body = documents.read(raw, pb.AttemptOutcomeBody)
        if (
            body["request_id"] != outcome.request_id
            or body["attempt_ordinal"] != outcome.attempt_ordinal
            or body["invocation_spec_digest"] != documents.spell(outcome.invocation_spec_digest)
        ):
            raise WorkspaceRefusal("outcome body differs from its execution identity")
        with self.locked() as db:
            row = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, outcome.request_id, outcome.attempt_ordinal),
            ).fetchone()
            if (
                row is None
                or row["spec"] != outcome.invocation_spec_digest
                or row["outcome"] not in (b"", raw)
                or row["outcome_id"] not in ("", outcome.outcome_id)
                or row["state"] == "released"
            ):
                raise WorkspaceRefusal("outcome has no unchanged accepted execution")
            db.execute(
                "UPDATE attempts SET state='outcome',outcome_id=?,outcome_digest=?,outcome=? "
                "WHERE owner=? AND request=? AND ordinal=?",
                (
                    outcome.outcome_id,
                    outcome.outcome_digest,
                    raw,
                    owner,
                    outcome.request_id,
                    outcome.attempt_ordinal,
                ),
            )

    def acknowledge(self, owner: str, ack: pb.AttemptOutcomeAck) -> None:
        self.owner(owner)
        with self.locked() as db:
            row = db.execute(
                "SELECT * FROM attempts WHERE (owner=? OR(owner='' AND transport_owner=?)) "
                "AND request=? AND ordinal=?",
                (owner, owner, ack.request_id, ack.attempt_ordinal),
            ).fetchone()
            if (
                row is None
                or row["state"] not in ("outcome", "released")
                or row["spec"] != ack.invocation_spec_digest
                or row["outcome_id"] != ack.outcome_id
                or row["outcome_digest"] != ack.outcome_digest
                or (row["state"] == "released" and ack.retain_work)
            ):
                raise WorkspaceRefusal("outcome acknowledgment changed or revoked its subject")
            db.execute(
                "UPDATE attempts SET state=?,retain=? WHERE owner=? AND request=? AND ordinal=?",
                (
                    "outcome" if ack.retain_work else "released",
                    int(ack.retain_work),
                    row["owner"],
                    ack.request_id,
                    ack.attempt_ordinal,
                ),
            )

    def begin_weights(self, owner: str, intent: pb.WeightsIntentFrame) -> dict[str, object]:
        """Assign the epoch once, before native begin; the Host observes this decision."""
        self.owner(owner)
        declaration = intent.tensorfs_declaration_canonical_bytes
        if (
            not declaration
            or len(declaration) > 16 << 20
            or hashlib.sha256(declaration).digest() != intent.tensorfs_declaration_digest
            or _ID.fullmatch(intent.output_slot) is None
            or intent.weights_transaction_id
            != weights_transaction_id(
                owner,
                intent.request_id,
                documents.spell(intent.invocation_spec_digest),
                intent.output_slot,
            )
        ):
            raise WorkspaceRefusal("weights declaration does not bind the assigned transaction")
        with self.locked() as db:
            attempt = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, intent.request_id, intent.attempt_ordinal),
            ).fetchone()
            if (
                attempt is None
                or attempt["spec"] != intent.invocation_spec_digest
                or attempt["state"] not in ("accepted", "running")
                or attempt["fenced"]
            ):
                raise WorkspaceRefusal("weights intent requires its live accepted invocation")
            row = db.execute(
                "SELECT * FROM weights WHERE id=?", (intent.weights_transaction_id,)
            ).fetchone()
            if row is not None:
                if (
                    row["owner"] != owner
                    or row["request"] != intent.request_id
                    or row["spec"] != intent.invocation_spec_digest
                    or row["slot"] != intent.output_slot
                    or row["declaration"] != declaration
                    or row["state"] == "released"
                ):
                    raise WorkspaceRefusal("weights transaction was changed or released")
                if row["ordinal"] != intent.attempt_ordinal:
                    prior = db.execute(
                        "SELECT state,fenced FROM attempts WHERE owner=? AND request=? "
                        "AND ordinal=?",
                        (owner, row["request"], row["ordinal"]),
                    ).fetchone()
                    if (
                        intent.attempt_ordinal <= row["ordinal"]
                        or prior is None
                        or (prior["state"] not in ("outcome", "released") and not prior["fenced"])
                    ):
                        raise WorkspaceRefusal("previous writer attempt has not stopped")
                    db.execute(
                        "UPDATE weights SET ordinal=?,epoch=epoch+CASE WHEN state='intent' "
                        "THEN 1 ELSE 0 END,ready=0,restore_checkpoint=x'' WHERE id=?",
                        (intent.attempt_ordinal, intent.weights_transaction_id),
                    )
            else:
                db.execute(
                    "INSERT INTO weights(id,owner,request,ordinal,spec,slot,declaration_digest,"
                    "declaration,epoch,state) VALUES(?,?,?,?,?,?,?,?,1,'intent')",
                    (
                        intent.weights_transaction_id,
                        owner,
                        intent.request_id,
                        intent.attempt_ordinal,
                        intent.invocation_spec_digest,
                        intent.output_slot,
                        intent.tensorfs_declaration_digest,
                        declaration,
                    ),
                )
            return dict(
                db.execute(
                    "SELECT * FROM weights WHERE id=?", (intent.weights_transaction_id,)
                ).fetchone()
            )

    def record_receipt(self, owner: str, frame: pb.WeightsReceiptFrame) -> None:
        """Bind the already-native-committed receipt before any projection is emitted."""
        self.owner(owner)
        raw = frame.weights_receipt.weights_receipt_canonical_bytes
        digest = hashlib.sha256(raw).digest()
        if not raw or digest != frame.weights_receipt.weights_receipt_digest:
            raise WorkspaceRefusal("receipt bytes differ from their digest")
        receipt = documents.read(raw, pb.WeightsReceipt)
        if (
            receipt["owner_authority_scope"] != owner
            or receipt["request_id"] != frame.request_id
            or receipt["invocation_spec_digest"] != documents.spell(frame.invocation_spec_digest)
            or receipt["output_slot"] != frame.output_slot
            or receipt["weights_transaction_id"] != frame.weights_transaction_id
        ):
            raise WorkspaceRefusal("receipt differs from its admitted execution")
        native = base64.b64decode(receipt["tensorfs_receipt_canonical_bytes"], validate=True)
        native_digest = hashlib.sha256(native).digest()
        if documents.spell(native_digest) != receipt["tensorfs_receipt_digest"]:
            raise WorkspaceRefusal("native receipt digest differs")
        with self.locked() as db:
            row = db.execute(
                "SELECT * FROM weights WHERE id=?", (frame.weights_transaction_id,)
            ).fetchone()
            if (
                row is None
                or row["owner"] != owner
                or row["request"] != frame.request_id
                or row["ordinal"] != frame.attempt_ordinal
                or row["epoch"] != frame.writer_epoch
                or row["slot"] != frame.output_slot
                or row["spec"] != frame.invocation_spec_digest
                or row["declaration_digest"] != frame.tensorfs_declaration_digest
                or row["state"] == "released"
                or row["receipt"] not in (b"", raw)
            ):
                raise WorkspaceRefusal("receipt writer is stale or its previous receipt changed")
            observed = db.native(
                lambda: (
                    fill.tensorfs_module()
                    .Store.open(str(self.store_root))
                    .derived_lookup(frame.weights_transaction_id)
                )
            )
            if (
                observed.get("state") != "committed"
                or observed.get("disposition", {}).get("kind") == "released"
                or canonical_json.encode(observed.get("receipt")) != native
            ):
                raise WorkspaceRefusal("receipt has no exact live native result")
            manifest = observed["receipt"]["manifest"]
            if (
                frame.manifest.digest.hex() != manifest["sha256"]
                or frame.manifest.length != manifest["length"]
            ):
                raise WorkspaceRefusal("receipt manifest differs from native result")
            current = db.execute(
                "SELECT state,epoch FROM weights WHERE id=?", (frame.weights_transaction_id,)
            ).fetchone()
            if current["state"] == "released" or current["epoch"] != frame.writer_epoch:
                raise WorkspaceRefusal("writer was released during native receipt validation")
            db.execute(
                "UPDATE weights SET state='receipt',receipt=?,receipt_digest=?,native_digest=?,"
                "manifest=?,manifest_length=?,objects=? WHERE id=?",
                (
                    raw,
                    digest,
                    native_digest,
                    frame.manifest.digest,
                    frame.manifest.length,
                    canonical_json.encode(
                        [
                            {
                                "object_id": value.object_id,
                                "length": value.length,
                                "source_ref": value.source_ref,
                            }
                            for value in frame.objects
                        ]
                    ),
                    frame.weights_transaction_id,
                ),
            )

    def weights_row(self, owner: str, transaction: str) -> dict[str, object]:
        with self.locked() as db:
            row = db.execute(
                "SELECT * FROM weights WHERE owner=? AND id=?", (owner, transaction)
            ).fetchone()
            if row is None:
                raise WorkspaceRefusal("workspace has no owned weights transaction")
            return dict(row)

    def ready_weights(self, owner: str, request: pb.WeightsIntentReadyRequest) -> dict[str, object]:
        subject = request.weights
        with self.locked() as db:
            row = db.execute(
                "SELECT * FROM weights WHERE owner=? AND id=?",
                (owner, subject.weights_transaction_id),
            ).fetchone()
            if (
                row is None
                or row["request"] != subject.request_id
                or row["ordinal"] != request.attempt_ordinal
                or row["epoch"] != subject.writer_epoch
                or row["slot"] != subject.output_slot
                or row["spec"] != subject.invocation_spec_digest
                or row["declaration_digest"] != subject.tensorfs_declaration_digest
                or row["state"] == "released"
            ):
                raise WorkspaceRefusal("checkpoint readiness does not bind the current writer")
            checkpoint = (
                request.checkpoint.SerializeToString() if request.HasField("checkpoint") else b""
            )
            if row["ready"]:
                if checkpoint != row["restore_checkpoint"]:
                    raise WorkspaceRefusal("writer readiness already captured another checkpoint")
                return dict(row)
            if checkpoint:
                db.native(lambda: self._validate_checkpoint(row, request.checkpoint))
            current = db.execute(
                "SELECT state,epoch FROM weights WHERE id=?", (row["id"],)
            ).fetchone()
            if current["state"] == "released" or current["epoch"] != row["epoch"]:
                raise WorkspaceRefusal("writer changed during checkpoint validation")
            db.execute(
                "UPDATE weights SET ready=1,restore_checkpoint=? WHERE id=?",
                (checkpoint, row["id"]),
            )
            return dict(db.execute("SELECT * FROM weights WHERE id=?", (row["id"],)).fetchone())

    def checkpoint(self, owner: str, frame: pb.WeightsCheckpointFrame) -> None:
        with self.locked() as db:
            row = db.execute(
                "SELECT * FROM weights WHERE owner=? AND id=?",
                (owner, frame.weights_transaction_id),
            ).fetchone()
            if (
                row is None
                or row["request"] != frame.request_id
                or row["ordinal"] != frame.attempt_ordinal
                or row["epoch"] != frame.writer_epoch
                or row["spec"] != frame.invocation_spec_digest
                or row["slot"] != frame.output_slot
                or row["declaration_digest"] != frame.tensorfs_declaration_digest
                or row["state"] != "intent"
            ):
                raise WorkspaceRefusal("checkpoint does not belong to the current native writer")
            if row["checkpoint"]:
                prior = pb.CheckpointRef.FromString(row["checkpoint"])
                if (
                    frame.checkpoint.plan_digest != prior.plan_digest
                    or frame.checkpoint.index < prior.index
                    or (frame.checkpoint.index == prior.index and frame.checkpoint != prior)
                ):
                    raise WorkspaceRefusal("checkpoint changed or regressed its native chain")
            db.native(lambda: self._validate_checkpoint(row, frame.checkpoint))
            current = db.execute(
                "SELECT state,epoch FROM weights WHERE id=?", (row["id"],)
            ).fetchone()
            if current["state"] != "intent" or current["epoch"] != frame.writer_epoch:
                raise WorkspaceRefusal("writer changed during checkpoint recording")
            db.execute(
                "UPDATE weights SET checkpoint=? WHERE id=?",
                (frame.checkpoint.SerializeToString(), row["id"]),
            )

    def _validate_checkpoint(self, row: sqlite3.Row, checkpoint: pb.CheckpointRef) -> None:
        store = fill.tensorfs_module().Store.open(str(self.store_root))
        page = store.checkpoint_page(
            documents.spell(checkpoint.head.digest),
            checkpoint.head.length,
            operation_id=row["request"],
            slot=row["slot"],
            plan_digest=documents.spell(checkpoint.plan_digest),
            limit=1,
        )
        if page["index"] != checkpoint.index or page["bytes"] != checkpoint.bytes:
            raise WorkspaceRefusal("checkpoint counters differ from native closure")
        store.validate_derived_checkpoint(
            row["id"],
            row["declaration"],
            documents.spell(checkpoint.head.digest),
            checkpoint.head.length,
            operation_id=row["request"],
            slot=row["slot"],
            plan_digest=documents.spell(checkpoint.plan_digest),
        )

    @staticmethod
    def status(row: dict[str, object]) -> pb.WeightsTransactionStatus:
        # SQLite rows have closed schema types; these are transport projections,
        # never another writer epoch or custody decision.
        from typing import Any, cast

        value = cast(dict[str, Any], row)
        status = pb.WeightsTransactionStatus(
            weights_transaction_id=value["id"],
            request_id=value["request"],
            attempt_ordinal=value["ordinal"],
            invocation_spec_digest=documents.spell(value["spec"]),
            output_slot=value["slot"],
            writer_epoch=value["epoch"],
            state=pb.WEIGHTS_TRANSACTION_STATE_RECEIPT
            if value["receipt"]
            else pb.WEIGHTS_TRANSACTION_STATE_INTENT,
            tensorfs_declaration_digest=value["declaration_digest"],
            weights_receipt_digest=value["receipt_digest"],
            intent_ready=bool(value["ready"]),
        )
        if value["checkpoint"]:
            status.checkpoint.CopyFrom(pb.CheckpointRef.FromString(value["checkpoint"]))
        return status

    def _change_hold(
        self,
        db: Journal,
        owner: str,
        request: pb.DerivedRetentionRequest,
        *,
        release: bool,
        cancel_lookup: bool = True,
    ) -> pb.DerivedRetentionResult:
        if (
            _DIGEST.fullmatch(request.weights_transaction_id) is None
            or _DIGEST.fullmatch(request.retention_id) is None
            or len(request.tensorfs_receipt_digest) != 32
        ):
            raise WorkspaceRefusal("native retention requires exact bounded identities")
        row = db.execute("SELECT * FROM holds WHERE id=?", (request.retention_id,)).fetchone()
        subject = (owner, request.weights_transaction_id, request.tensorfs_receipt_digest)
        if row is not None:
            if (row["owner"], row["transaction_id"], row["native_digest"]) != subject:
                raise WorkspaceRefusal("retention identity already belongs to another subject")
            if not release and row["state"] in ("releasing", "released"):
                raise WorkspaceRefusal("released retention cannot be recreated")
        else:
            original = db.execute(
                "SELECT 1 FROM weights WHERE owner=? AND id=? "
                "AND native_digest=? AND state='receipt'",
                subject,
            ).fetchone()
            retained = db.execute(
                "SELECT 1 FROM holds WHERE owner=? AND transaction_id=? "
                "AND native_digest=? AND state='held'",
                subject,
            ).fetchone()
            if original is None and retained is None:
                raise WorkspaceRefusal("native result has no receipt owned by this workspace owner")
            db.execute(
                "INSERT INTO holds(id,owner,transaction_id,native_digest,state) VALUES(?,?,?,?,?)",
                (request.retention_id, *subject, "releasing" if release else "retaining"),
            )
        if release:
            db.execute("UPDATE holds SET state='releasing' WHERE id=?", (request.retention_id,))
        if release and cancel_lookup:
            db.execute(
                "UPDATE operation_lookups SET state='released' WHERE owner=? AND EXISTS "
                "(SELECT 1 FROM json_each(CAST(body AS TEXT),'$.holds') AS h "
                "WHERE json_extract(h.value,'$.retention_id')=?)",
                (owner, request.retention_id),
            )
        # The intent above is durable before entering native code. A crash on
        # either side of this call leaves the same idempotent intent to replay.
        result = db.native(
            lambda: derived_retention.change(
                request, tensorfs_root=self.store_root, release=release
            )
        )
        current = db.execute(
            "SELECT state FROM holds WHERE id=?", (request.retention_id,)
        ).fetchone()
        if not release and current["state"] in ("releasing", "released"):
            db.native(
                lambda: derived_retention.change(
                    request, tensorfs_root=self.store_root, release=True
                )
            )
            raise WorkspaceRefusal("retention was released during native acquisition")
        manifest = result.manifest if result.HasField("manifest") else None
        db.execute(
            "UPDATE holds SET state=?,manifest=?,manifest_length=? WHERE id=?",
            (
                "released" if release else "held",
                manifest.digest if manifest else b"",
                manifest.length if manifest else 0,
                request.retention_id,
            ),
        )
        return result
