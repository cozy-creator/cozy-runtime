"""One privileged custody journal for every Runtime using a TensorFS workspace.

The file lock serializes whole journal/native transitions across worker processes.
Intent commits precede native mutations; an interrupted mutation is replayed by its
exact identity. This directory is separate from TensorFS's catalog and roots, and
is never placed in an ephemeral worker or author environment.
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import sqlite3
import threading
import weakref
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypedDict

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill, weights_sink, weights_writer
from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.internal.worker import derived_retention
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


class WeightsRow(TypedDict):
    """One ``weights`` row."""

    id: str
    owner: str
    request: str
    ordinal: int
    spec: bytes
    slot: str
    declaration_digest: bytes
    declaration: bytes
    epoch: int
    state: Literal["intent", "receipt", "released"]
    ready: int
    restore_checkpoint: bytes
    checkpoint: bytes
    receipt: bytes
    receipt_digest: bytes
    native_digest: bytes
    manifest: bytes
    manifest_length: int
    objects: bytes
    adoption_source: str
    adoption_epoch: int
    adoption_checkpoint: bytes
    retired_ms: int


class WeightsTransaction(WeightsRow):
    """A weights row with its writer attempt's fence and state."""

    attempt_fenced: int
    attempt_state: str


class WorkspaceRefusal(ValueError):
    """A stable workspace identity or custody invariant refused the operation."""


class WorkspaceBusy(WorkspaceRefusal):
    """A foreground custody transition takes precedence over optional collection."""


def same_ref(value: Any, digest: str, length: int) -> bool:
    """A ``{digest, length}`` reference; members another version added are ignored."""
    return (
        isinstance(value, dict) and value.get("digest") == digest and value.get("length") == length
    )


def artifact_identity(value: Any) -> tuple[Any, ...] | None:
    """The provenance members artifact readers consume; additive members are ignored."""
    if not isinstance(value, dict) or not isinstance(value.get("manifest"), dict):
        return None
    identity = (
        value.get("producer_request_id"),
        value.get("output_slot"),
        value["manifest"].get("digest"),
        value["manifest"].get("length"),
        value.get("tensorfs_receipt_digest"),
    )
    return None if None in identity else identity


_authority: ContextVar[Callable[[], None]] = ContextVar("workspace_authority", default=lambda: None)


class Changes:
    """Counts committed journal writes, so a reader can wait for the next one."""

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.version = 0

    def bump(self) -> None:
        with self.condition:
            self.version += 1
            self.condition.notify_all()


_changes: dict[Path, Changes] = {}
_changes_lock = threading.Lock()
Row = sqlite3.Row


class Journal(sqlite3.Connection):
    lock_fd: int
    mutex: threading.Lock
    guard: Callable[[], None]
    #: A newer Runtime's journal has been given this Runtime's schema on this connection.
    converged: bool = False

    def one[T](self, into: type[T], sql: str, parameters: Sequence[object] = ()) -> T | None:
        """The row `sql` selects, decoded once into `into`; None when it selects none."""
        row = self.execute(sql, parameters).fetchone()
        return None if row is None else _decode_row(row, into)

    def all[T](self, into: type[T], sql: str, parameters: Sequence[object] = ()) -> list[T]:
        """Every row `sql` selects, each decoded once into `into`."""
        return [_decode_row(row, into) for row in self.execute(sql, parameters).fetchall()]

    def native[T](self, operation: Callable[[], T]) -> T:
        """Native owns object/GC locking; no workspace transaction spans its IO."""
        if self.in_transaction:
            raise RuntimeError("native work cannot run inside a workspace SQL transaction")
        lock_fd, guard = self.lock_fd, self.guard
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        self.mutex.release()
        try:
            return operation()
        finally:
            self.mutex.acquire()
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self.lock_fd, self.guard = lock_fd, guard
            guard()


def _decode_row[T](row: sqlite3.Row, into: type[T]) -> T:
    try:
        return msgspec.convert(dict(zip(row.keys(), row, strict=True)), into, strict=True)
    except msgspec.ValidationError as exc:
        raise WorkspaceRefusal(f"journal row is not a {into.__name__}: {exc}") from exc


class NativeHold(msgspec.Struct, frozen=True, kw_only=True):
    """One ``holds`` row."""

    id: str
    owner: str
    transaction_id: str
    native_digest: bytes
    state: Literal["retaining", "held", "releasing", "released"]
    manifest: bytes = b""
    manifest_length: int = 0
    kind: Literal["derived", "tree"] = "derived"


class HeldTree(NativeHold, frozen=True, kw_only=True):
    """A ``holds`` row joined with its producing byte output's size."""

    content_bytes: int

    def source(self) -> pb.NativeByteTreeRef:
        return pb.NativeByteTreeRef(
            producer_root_id=self.transaction_id,
            receipt_digest=self.native_digest,
            manifest=pb.Ref(digest=self.manifest, length=self.manifest_length),
            content_bytes=self.content_bytes,
        )


_LOG = logging.getLogger(__name__)


class _Handle:
    """This process's one journal connection for one workspace directory.

    The file lock still serializes processes; the mutex serializes this process's threads
    over the shared connection. Telemetry is deferred and committed without a sync of its
    own: every synchronous commit after it makes it durable, in order.
    """

    def __init__(self, directory: Path):
        for path in (directory.parent, directory):
            fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        self.path = directory / "journal.sqlite3"
        self.pid = os.getpid()
        self.mutex = threading.Lock()
        self.db: Journal | None = None
        self.identity: tuple[int, int] = (0, 0)
        self.process = b""
        self.pending: list[Callable[[Journal], None]] = []
        self.pending_lock = threading.Lock()

    def connect(self) -> Journal:
        if self.path.is_symlink():
            raise WorkspaceRefusal("workspace journal cannot be a symlink")
        try:
            stat = os.stat(self.path)
            identity = (stat.st_dev, stat.st_ino)
        except FileNotFoundError:
            identity = (0, 0)
        if self.db is None or identity != self.identity:
            # A replaced journal is a different workspace. The previous connection closes
            # when its last user drops it.
            db = sqlite3.connect(
                self.path, isolation_level=None, factory=Journal, check_same_thread=False
            )
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            _migrate(db)
            # Recovery reads only live attempts. An index is not a schema version: any
            # Runtime maintains it, so no reader of this journal is refused over it.
            db.execute(
                "CREATE INDEX IF NOT EXISTS attempts_live ON attempts(owner,request,ordinal) "
                "WHERE state IN ('accepted','running')"
            )
            _number_runs(db)
            from .machine_public_reads import install

            install(db)
            stat = os.stat(self.path)
            self.db, self.identity = db, (stat.st_dev, stat.st_ino)
        return self.db

    def defer(self, write: Callable[[Journal], None], flush: Callable[[], None]) -> None:
        with self.pending_lock:
            first = not self.pending
            self.pending.append(write)
        if first:
            # One writer per burst: whatever arrives while it waits joins its commit.
            threading.Thread(target=flush, name="workspace-telemetry", daemon=True).start()

    def drain(self, db: Journal) -> None:
        with self.pending_lock:
            writes, self.pending = self.pending, []
        if not writes:
            return
        db.execute("PRAGMA synchronous=NORMAL")
        try:
            db.execute("BEGIN IMMEDIATE")
            for write in writes:
                write(db)
            db.execute("COMMIT")
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            _LOG.warning(
                "workspace telemetry batch of %d was not recorded", len(writes), exc_info=True
            )
        finally:
            db.execute("PRAGMA synchronous=FULL")


_handles: weakref.WeakValueDictionary[Path, _Handle] = weakref.WeakValueDictionary()
_handles_lock = threading.Lock()


def _handle(directory: Path) -> _Handle:
    with _handles_lock:
        handle = _handles.get(directory)
        if handle is None or handle.pid != os.getpid():
            handle = _handles[directory] = _Handle(directory)
        return handle


#: The journal schema this Runtime writes. Every change is additive: a new table, index or
#: column with a default, so a journal any other Runtime wrote converges (`_converge`).
SCHEMA_VERSION = 14


def _schema() -> str:
    from .machine_checkpoint_inputs import SCHEMA11
    from .machine_models import SCHEMA9
    from .workspace_byte_outputs import SCHEMA5
    from .workspace_calls import SCHEMA8
    from .workspace_executions import SCHEMA7
    from .workspace_input_trees import SCHEMA10
    from .workspace_partial import SCHEMA12
    from .workspace_sources import SCHEMA4

    return (
        "".join((_SCHEMA, SCHEMA4, SCHEMA5, SCHEMA7, SCHEMA8, SCHEMA9, SCHEMA10))
        + SCHEMA11
        + SCHEMA12
        # Providers an execution was given a credential for; never a credential value.
        + "ALTER TABLE executions ADD COLUMN source_providers TEXT NOT NULL DEFAULT '';"
        # The record owner answers this execution's memo lookups (MachineExecutionSubmit).
        + "ALTER TABLE executions ADD COLUMN owner_memo INTEGER NOT NULL DEFAULT 0 "
        + "CHECK(owner_memo IN (0,1));"
    )


def _converge(db: Journal) -> list[str]:
    """What gives a journal another Runtime wrote, older or newer, everything this Runtime's
    schema holds: its missing tables, indexes and columns. What a newer Runtime added stays.
    A missing column without a default cannot be added, and refuses."""
    target = sqlite3.connect(":memory:")
    try:
        target.executescript(_schema())
        statements = []
        for kind, name, sql in target.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ):
            held = db.execute("SELECT 1 FROM sqlite_master WHERE type=? AND name=?", (kind, name))
            if held.fetchone() is None:
                statements.append(sql)
                continue
            if kind != "table":
                continue
            columns = {row[1] for row in db.execute(f"PRAGMA table_info({name})")}
            for _, column, type_, required, default, _ in target.execute(
                f"PRAGMA table_info({name})"
            ):
                if column in columns:
                    continue
                if required and default is None:
                    raise WorkspaceRefusal(
                        f"workspace journal table {name} lacks column {column}, which has no "
                        "default; start this machine with an empty workspace"
                    )
                statements.append(
                    f"ALTER TABLE {name} ADD COLUMN {column} {type_}"
                    + (" NOT NULL" if required else "")
                    + ("" if default is None else f" DEFAULT {default}")
                )
        return statements
    finally:
        target.close()


def _migrate(db: Journal) -> None:
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION or (version > SCHEMA_VERSION and db.converged):
        return
    if version == 0:
        db.executescript(
            f"BEGIN IMMEDIATE;{_schema()}PRAGMA user_version={SCHEMA_VERSION}; COMMIT;"
        )
    else:
        db.execute("BEGIN IMMEDIATE")
        try:
            for statement in _converge(db):
                db.execute(statement)
            if db.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
                db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        db.converged = True


#: Run numbers (wire 66): the journal's sequence of root executions in acceptance order, never
#: reused. Like `attempts_live` it is not a schema version, so no Runtime is refused over it: one
#: that predates it ignores it, and the roots such a Runtime accepted are numbered, in
#: acceptance order, when this one next opens the journal.
_RUN_NUMBERS = (
    "CREATE TABLE IF NOT EXISTS execution_numbers ("
    "number INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, request TEXT NOT NULL, "
    "UNIQUE(owner,request)) STRICT"
)
#: Numbers go to roots, the executions no call made; a child call has none.
NUMBER_ROOT = (
    "INSERT INTO execution_numbers(owner,request) SELECT e.owner,e.request FROM executions e "
    "WHERE {where} AND NOT EXISTS (SELECT 1 FROM execution_numbers n "
    "WHERE n.owner=e.owner AND n.request=e.request) AND NOT EXISTS "
    "(SELECT 1 FROM execution_calls c WHERE c.owner=e.owner AND c.child_request=e.request) "
    "ORDER BY e.rowid"
)


def _number_runs(db: Journal) -> None:
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute(_RUN_NUMBERS)
        db.execute(NUMBER_ROOT.format(where="1"))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")

_SCHEMA = """
CREATE TABLE workspace_state (singleton INTEGER PRIMARY KEY CHECK(singleton=1)) STRICT;
INSERT INTO workspace_state(singleton) VALUES(1);
CREATE TABLE attempts (
 owner TEXT NOT NULL, request TEXT NOT NULL, ordinal INTEGER NOT NULL,
 spec BLOB NOT NULL, invocation BLOB NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('accepted','running','outcome','released')),
 outcome_id TEXT NOT NULL DEFAULT '', outcome_digest BLOB NOT NULL DEFAULT x'',
 outcome BLOB NOT NULL DEFAULT x'', retain INTEGER NOT NULL DEFAULT 0,
 fenced INTEGER NOT NULL DEFAULT 0,
 worker_boot TEXT NOT NULL DEFAULT '', result_schema BLOB NOT NULL DEFAULT x'',
 process_owner BLOB NOT NULL DEFAULT x'',
 memoize INTEGER NOT NULL DEFAULT 0 CHECK(memoize IN (0,1)),
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
 adoption_source TEXT NOT NULL DEFAULT '', adoption_epoch INTEGER NOT NULL DEFAULT 0,
 adoption_checkpoint BLOB NOT NULL DEFAULT x'',
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
 state TEXT NOT NULL CHECK(state IN
 ('retaining','ready','missing','miss','released','acknowledged')),
 PRIMARY KEY(owner,consumer)
) STRICT;
"""


class Workspace:
    def __init__(self, store_root: Path):
        if not store_root.is_absolute() or store_root == Path("/"):
            raise WorkspaceRefusal("workspace requires the configured absolute store root")
        self.store_root = store_root.resolve()
        fill.ensure_store(self.store_root)
        self.directory = self.store_root / ".cozy-workspace"
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise WorkspaceRefusal("workspace journal directory is not an owned directory")
        if self.directory.stat().st_uid != os.geteuid():
            raise WorkspaceRefusal("workspace journal belongs to another operating-system owner")
        os.chmod(self.directory, 0o700)
        self._journal = _handle(self.directory)

    @property
    def _handle(self) -> _Handle:
        if self._journal.pid != os.getpid():
            self._journal = _handle(self.directory)
        return self._journal

    @contextmanager
    def locked(self, *, blocking: bool = True) -> Iterator[Journal]:
        handle = self._handle
        if not handle.mutex.acquire(blocking=blocking):
            raise WorkspaceBusy("foreground workspace transition is active")
        lock = -1
        changed = False
        try:
            lock = os.open(self.directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError as exc:
                raise WorkspaceBusy("foreground workspace transition is active") from exc
            guard = _authority.get()
            guard()
            db = handle.connect()
            db.lock_fd, db.mutex, db.guard = lock, handle.mutex, guard
            baseline = db.total_changes
            try:
                _migrate(db)
                handle.drain(db)
                yield db
            finally:
                if db.in_transaction:
                    db.execute("ROLLBACK")
                changed = db.total_changes != baseline
        finally:
            if lock >= 0:
                os.close(lock)
            handle.mutex.release()
            if changed:
                self.changes().bump()

    def defer(self, write: Callable[[Journal], None]) -> None:
        """Record lossy telemetry with the next journal access, never on the caller's path."""
        self._handle.defer(write, self._flush)

    def _flush(self) -> None:
        try:
            with self.locked():
                pass
        except Exception:
            _LOG.warning("workspace telemetry waits for the next journal access", exc_info=True)

    def changes(self) -> Changes:
        """This journal's change counter, shared by every Workspace opened on it."""
        with _changes_lock:
            return _changes.setdefault(self.directory.resolve(), Changes())

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
        memoize: bool = False,
    ) -> pb.AttemptOutcome | None:
        self.owner(owner)
        self.recover(owner)
        self.validate_offer(offer)
        with self.locked() as db:
            return self.accept_in(
                db,
                owner,
                offer,
                result_schema=result_schema,
                worker_boot=worker_boot,
                memoize=memoize,
            )

    @staticmethod
    def validate_offer(offer: pb.AttemptOffer) -> None:
        if (
            _ID.fullmatch(offer.request_id) is None
            or not 0 < offer.attempt_ordinal < 1 << 63
            or len(offer.invocation_spec_digest) != 32
            or hashlib.sha256(offer.invocation_spec_canonical_bytes).digest()
            != offer.invocation_spec_digest
        ):
            raise WorkspaceRefusal("workspace attempt does not bind its exact invocation")
        documents.parse(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)

    @staticmethod
    def accept_in(
        db: Journal,
        owner: str,
        offer: pb.AttemptOffer,
        *,
        result_schema: bytes = b"",
        worker_boot: str = "",
        memoize: bool = False,
    ) -> pb.AttemptOutcome | None:
        """Insert the existing attempt row within a caller's admission transaction."""
        from .workspace_recovery import process_identity

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
                return Workspace._outcome_frame(row)
            return None
        if any(entry.HasField("catalog_model") for entry in offer.grant.inputs):
            spec = documents.parse(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
            if spec.HasField("serving"):
                # A direct control offer cannot bypass the retained input gate
                # used by machine children. Replay above preserves collected rows.
                from .machine_checkpoint_inputs import check_submit

                check_submit(db, owner, offer)
        db.execute(
            "INSERT INTO attempts(owner,request,ordinal,spec,invocation,state,"
            "worker_boot,result_schema,process_owner,memoize) "
            "VALUES(?,?,?,?,?,'accepted',?,?,?,?)",
            (
                owner,
                offer.request_id,
                offer.attempt_ordinal,
                offer.invocation_spec_digest,
                offer.invocation_spec_canonical_bytes,
                worker_boot,
                result_schema,
                process_identity(),
                int(memoize),
            ),
        )

        return None

    def mark_running(self, owner: str, accepted: pb.AttemptAccepted) -> None:
        """Execution STARTS (a compare-and-set against every terminal recorded before it)."""
        with self.locked() as db:
            if not db.execute(
                "UPDATE attempts SET state='running' WHERE owner=? AND request=? "
                "AND ordinal=? AND spec=? AND state='accepted' AND fenced=0 AND outcome=x''",
                (
                    owner,
                    accepted.request_id,
                    accepted.attempt_ordinal,
                    accepted.invocation_spec_digest,
                ),
            ).rowcount:
                raise WorkspaceRefusal("the attempt was stopped before dispatch")

    def recover(self, owner: str) -> None:
        """Recover dead process births without disturbing another live local worker.

        This process's own attempts are never examined: it is alive to ask.
        """
        from . import workspace_recovery

        self.owner(owner)
        handle = self._handle
        if not handle.process:
            handle.process = workspace_recovery.process_identity()
        with self.locked() as db:
            rows = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND state IN ('accepted','running') "
                "AND process_owner NOT IN (x'',?) ORDER BY request,ordinal",
                (owner, handle.process),
            ).fetchall()
            for attempt in rows:
                if not workspace_recovery.process_ended(attempt["process_owner"]):
                    continue
                weights = db.execute(
                    "SELECT * FROM weights WHERE owner=? AND request=? AND ordinal=?",
                    (owner, attempt["request"], attempt["ordinal"]),
                ).fetchall()
                detail = "Previous Runtime execution process ended before recording its outcome."
                for weight in weights:
                    try:
                        receipt = db.native(
                            partial(workspace_recovery.native_result, self, dict(weight))
                        )
                        if receipt is not None:
                            db.native(partial(self.record_receipt, owner, receipt))
                    except Exception as exc:
                        db.guard()
                        # One unrecoverable row ends only its own attempt: the claim
                        # snapshot and every other attempt keep going.
                        cause = f"{type(exc).__name__}: {exc}"
                        detail = (
                            "Previous Runtime execution ended and its native result could not "
                            "be recovered: "
                            + "".join(c if " " <= c <= "~" else "?" for c in cause)[:512]
                        )
                        break
                current = db.execute(
                    "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                    (owner, attempt["request"], attempt["ordinal"]),
                ).fetchone()
                if current is not None and current["state"] in ("accepted", "running"):
                    workspace_recovery.terminal(db, current, detail)

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
                    "SELECT 1 FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                    (owner, request, ordinal),
                ).fetchone()
                is not None
            )

    def retained_outcomes(self, owner: str) -> list[pb.AttemptOutcome]:
        self.recover(owner)
        with self.locked() as db:
            return [
                self._outcome_frame(row)
                for row in db.execute(
                    "SELECT * FROM attempts WHERE owner=? "
                    "AND state='outcome' "
                    "ORDER BY request,ordinal",
                    (owner,),
                )
            ]

    @staticmethod
    def _model_held(
        db: Journal, owner: str, source: pb.DerivedRetentionRequest, manifest: pb.Ref
    ) -> bool:
        row = db.one(
            NativeHold, "SELECT * FROM holds WHERE id=? AND owner=?", (source.retention_id, owner)
        )
        return row is not None and (
            row.state == "held"
            and row.transaction_id == source.weights_transaction_id
            and row.native_digest == source.tensorfs_receipt_digest
            and row.manifest == manifest.digest
            and row.manifest_length == manifest.length
        )

    def model_held(self, owner: str, source: pb.DerivedRetentionRequest, manifest: pb.Ref) -> bool:
        """Whether the exact retained owner obligation still keeps this model's closure."""
        with self.locked() as db:
            return self._model_held(db, owner, source, manifest)

    @contextmanager
    def held_model(
        self, owner: str, source: pb.DerivedRetentionRequest, manifest: pb.Ref
    ) -> Iterator[None]:
        """Lease an exact retained model while preparation reads its native closure."""
        self.owner(owner)
        if len(manifest.digest) != 32 or not manifest.length:
            raise WorkspaceRefusal("native model manifest is incomplete")

        def verify(db: Journal) -> None:
            if not self._model_held(db, owner, source, manifest):
                raise WorkspaceRefusal("native model has no exact retained owner obligation")

        store = fill.store(self.store_root)
        lease = None
        try:
            with self.locked() as db:
                verify(db)
                lease = db.native(
                    lambda: store.acquire_cozytensors(documents.spell(manifest.digest))
                )
                verify(db)
            yield
            with self.locked() as db:
                verify(db)
        finally:
            if lease is not None:
                lease.release()

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
            row = db.one(
                WeightsRow, "SELECT * FROM weights WHERE id=?", (request.weights_transaction_id,)
            )
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
        body = documents.parse(raw, pb.AttemptOutcomeBody)
        if (
            body.request_id != outcome.request_id
            or body.attempt_ordinal != outcome.attempt_ordinal
            or body.invocation_spec_digest != documents.spell(outcome.invocation_spec_digest)
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
                or row["outcome_digest"] not in (b"", outcome.outcome_digest)
            ):
                raise WorkspaceRefusal("outcome has no unchanged accepted execution")
            if row["state"] == "released" or row["outcome"] == raw:
                return
            from .tree_members import PROJECTION_PREFIX, recorded_projection
            from .workspace_byte_outputs import ByteOutput, recorded_native, reference

            output_rows = db.all(
                ByteOutput,
                "SELECT * FROM byte_outputs WHERE owner=? AND request=? AND ordinal=?",
                (row["owner"], outcome.request_id, outcome.attempt_ordinal),
            )
            if body.status == pb.OUTCOME_STATUS_SUCCEEDED:
                native = {
                    item.output_id: item.native_tree
                    for item in body.output_manifest.outputs
                    if item.HasField("native_tree")
                }
                returned_rows = []
                for output in output_rows:
                    if (
                        output.slot not in native
                        and output.slot.startswith(PROJECTION_PREFIX)
                        and recorded_projection(db, output)
                    ):
                        continue
                    if output.slot.startswith("product."):
                        continue  # a published product: the run's output log holds it

                    if output.native_service_id:
                        if not recorded_native(db, output, complete=True):
                            raise WorkspaceRefusal(
                                "terminal source view has no completed native call"
                            )
                    else:
                        returned_rows.append(output)
                if set(native) != {output.slot for output in returned_rows}:
                    raise WorkspaceRefusal(
                        "terminal native byte outputs do not match completed intents"
                    )
                for output in returned_rows:
                    if output.state != "complete" or reference(output) != native[output.slot]:
                        raise WorkspaceRefusal(
                            "terminal native byte output differs from its receipt"
                        )
            db.execute(
                "UPDATE attempts SET state='outcome',outcome_id=?,outcome_digest=?,outcome=? "
                "WHERE owner=? AND request=? AND ordinal=?",
                (
                    outcome.outcome_id,
                    outcome.outcome_digest,
                    raw,
                    row["owner"],
                    outcome.request_id,
                    outcome.attempt_ordinal,
                ),
            )

    def acknowledge(self, owner: str, ack: pb.AttemptOutcomeAck) -> None:
        from . import workspace_byte_outputs, workspace_sources

        self.owner(owner)
        source_releases: list[str] = []
        byte_releases: list[str] = []
        with self.locked() as db:
            row = db.execute(
                "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, ack.request_id, ack.attempt_ordinal),
            ).fetchone()
            if row is None and not ack.retain_work:
                source_releases = [
                    r[0]
                    for r in db.execute(
                        "SELECT service_id FROM native_calls WHERE owner=? AND parent_request=? "
                        "AND release_ack=? AND state='releasing'",
                        (owner, ack.request_id, workspace_sources.ack_identity(ack)),
                    )
                ]
                for service in source_releases:
                    db.native(partial(workspace_sources.release, self, owner, service))
                return
            if (
                row is None
                or row["state"] not in ("outcome", "released")
                or row["spec"] != ack.invocation_spec_digest
                or row["outcome_id"] != ack.outcome_id
                or row["outcome_digest"] != ack.outcome_digest
                or (row["state"] == "released" and ack.retain_work)
            ):
                raise WorkspaceRefusal("outcome acknowledgment changed or revoked its subject")
            execution = db.execute(
                "SELECT collected,desired,state FROM executions WHERE owner=? AND request=?",
                (row["owner"], ack.request_id),
            ).fetchone()
            if (  # a failed execution holds no result: its hold ends at the next boot
                execution is not None
                and not execution["collected"]
                and execution["desired"] != "cancel"
                and execution["state"] != "failed"
                and not ack.retain_work
            ):
                raise WorkspaceRefusal("uncollected execution outcome must remain retained")
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
            if not ack.retain_work:
                source_releases = workspace_sources.release_intent(db, owner, ack)
                db.execute(
                    "UPDATE byte_outputs SET state='releasing' WHERE owner=? AND request=? "
                    "AND ordinal=? AND spec=? AND state<>'released'",
                    (row["owner"], ack.request_id, ack.attempt_ordinal, ack.invocation_spec_digest),
                )
                byte_releases = [
                    output[0]
                    for output in db.execute(
                        "SELECT id FROM byte_outputs WHERE owner=? AND request=? AND ordinal=? "
                        "AND state='releasing'",
                        (row["owner"], ack.request_id, ack.attempt_ordinal),
                    )
                ]
                active = db.execute(
                    "SELECT 1 FROM weights WHERE owner=? AND request=? AND ordinal=? "
                    "AND state<>'released' UNION ALL SELECT 1 FROM byte_outputs WHERE "
                    "owner=? AND request=? AND ordinal=? AND state<>'released' LIMIT 1",
                    (row["owner"], ack.request_id, ack.attempt_ordinal) * 2,
                ).fetchone()
                execution = db.execute(
                    "SELECT 1 FROM executions WHERE owner=? AND request=?",
                    (row["owner"], ack.request_id),
                ).fetchone()
                if execution is not None:
                    pass  # Native release does not delete machine execution history.
                elif active is None:
                    db.execute(
                        "DELETE FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                        (row["owner"], ack.request_id, ack.attempt_ordinal),
                    )
                else:
                    db.execute(
                        "UPDATE attempts SET invocation=x'',outcome=x'',result_schema=x'' "
                        "WHERE owner=? AND request=? AND ordinal=?",
                        (row["owner"], ack.request_id, ack.attempt_ordinal),
                    )
                db.execute(
                    "UPDATE weights SET declaration=x'',checkpoint=x'',restore_checkpoint=x'',"
                    "receipt=x'',objects=x'5b5d' WHERE owner=? AND request=? AND ordinal=? "
                    "AND state='released'",
                    (row["owner"], ack.request_id, ack.attempt_ordinal),
                )

        for service in source_releases:
            workspace_sources.release(self, owner, service)
        for root in byte_releases:
            workspace_byte_outputs.release(self, owner, root)

    def begin_weights(self, owner: str, intent: pb.WeightsIntentFrame) -> WeightsRow:
        """Assign the epoch once, before native begin; the Host observes this decision."""
        self.owner(owner)
        declaration = intent.tensorfs_declaration_canonical_bytes
        if (
            not declaration
            or len(declaration) > weights_writer.declaration_bound()
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
            row = db.one(
                WeightsRow, "SELECT * FROM weights WHERE id=?", (intent.weights_transaction_id,)
            )
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
            from .workspace_partial import adopt

            return adopt(
                self,
                db,
                owner,
                intent.weights_transaction_id,
            )

    def record_receipt(self, owner: str, frame: pb.WeightsReceiptFrame) -> None:
        """Bind the already-native-committed receipt before any projection is emitted."""
        self.owner(owner)
        raw = frame.weights_receipt.weights_receipt_canonical_bytes
        digest = hashlib.sha256(raw).digest()
        if not raw or digest != frame.weights_receipt.weights_receipt_digest:
            raise WorkspaceRefusal("receipt bytes differ from their digest")
        receipt = documents.parse(raw, pb.WeightsReceipt)
        if (
            receipt.owner_authority_scope != owner
            or receipt.request_id != frame.request_id
            or receipt.invocation_spec_digest != documents.spell(frame.invocation_spec_digest)
            or receipt.output_slot != frame.output_slot
            or receipt.weights_transaction_id != frame.weights_transaction_id
        ):
            raise WorkspaceRefusal("receipt differs from its admitted execution")
        native = receipt.tensorfs_receipt_canonical_bytes
        native_digest = hashlib.sha256(native).digest()
        if documents.spell(native_digest) != receipt.tensorfs_receipt_digest:
            raise WorkspaceRefusal("native receipt digest differs")
        with self.locked() as db:
            row = db.one(
                WeightsRow, "SELECT * FROM weights WHERE id=?", (frame.weights_transaction_id,)
            )
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
                lambda: fill.store(self.store_root).derived_lookup(frame.weights_transaction_id)
            )
            if (
                observed.get("state") != "committed"
                or observed.get("disposition", {}).get("kind") == "released"
                or not weights_sink.same_receipt(observed["receipt"], canonical_json.decode(native))
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

    def weights_row(self, owner: str, transaction: str) -> WeightsTransaction:
        with self.locked() as db:
            row = db.one(
                WeightsTransaction,
                "SELECT w.*,a.fenced AS attempt_fenced,a.state AS attempt_state FROM weights w "
                "JOIN attempts a ON a.owner=w.owner AND a.request=w.request "
                "AND a.ordinal=w.ordinal "
                "WHERE w.owner=? AND w.id=?",
                (owner, transaction),
            )
            if row is None:
                raise WorkspaceRefusal("workspace has no owned weights transaction")
            return row

    def ready_weights(self, owner: str, request: pb.WeightsIntentReadyRequest) -> WeightsRow:
        subject = request.weights
        with self.locked() as db:
            row = db.one(
                WeightsRow,
                "SELECT * FROM weights WHERE owner=? AND id=?",
                (owner, subject.weights_transaction_id),
            )
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
            attempt = db.execute(
                "SELECT state,fenced FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, row["request"], row["ordinal"]),
            ).fetchone()
            if attempt is None or attempt["fenced"]:
                raise WorkspaceRefusal("checkpoint writer attempt was fenced")
            checkpoint = (
                request.checkpoint.SerializeToString() if request.HasField("checkpoint") else b""
            )
            if row["ready"]:
                if checkpoint != row["restore_checkpoint"]:
                    raise WorkspaceRefusal("writer readiness already captured another checkpoint")
                return row
            if attempt["state"] not in ("accepted", "running"):
                raise WorkspaceRefusal("checkpoint writer attempt has stopped")
            if checkpoint:
                db.native(lambda: self._validate_checkpoint(row, request.checkpoint))
            current = self._weights(db, row["id"])
            attempt = db.execute(
                "SELECT state,fenced FROM attempts WHERE owner=? AND request=? AND ordinal=?",
                (owner, row["request"], row["ordinal"]),
            ).fetchone()
            if (
                current["state"] == "released"
                or current["epoch"] != row["epoch"]
                or current["ordinal"] != row["ordinal"]
                or attempt is None
                or attempt["fenced"]
            ):
                raise WorkspaceRefusal("writer changed during checkpoint validation")
            if current["ready"]:
                if checkpoint != current["restore_checkpoint"]:
                    raise WorkspaceRefusal("writer readiness already captured another checkpoint")
                return current
            if attempt["state"] not in ("accepted", "running"):
                raise WorkspaceRefusal("writer stopped during checkpoint validation")
            db.execute(
                "UPDATE weights SET ready=1,restore_checkpoint=? WHERE id=?",
                (checkpoint, row["id"]),
            )
            return self._weights(db, row["id"])

    def checkpoint(self, owner: str, frame: pb.WeightsCheckpointFrame) -> None:
        """Record this worker's own native writer's latest checkpoint.

        The frame is what that writer just produced, so it is not re-validated against its
        whole chain (a cost that grows with every part). A checkpoint named from outside,
        to restore from, is validated in full by `ready_weights`.
        """
        with self.locked() as db:
            row = db.one(
                WeightsRow,
                "SELECT * FROM weights WHERE owner=? AND id=?",
                (owner, frame.weights_transaction_id),
            )
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
            self._checkpoint_advances(row["checkpoint"], frame.checkpoint)
            db.execute(
                "UPDATE weights SET checkpoint=? WHERE id=?",
                (frame.checkpoint.SerializeToString(), row["id"]),
            )

    @staticmethod
    def _checkpoint_advances(previous: bytes, checkpoint: pb.CheckpointRef) -> None:
        if previous:
            prior = pb.CheckpointRef.FromString(previous)
            if (
                checkpoint.plan_digest != prior.plan_digest
                or checkpoint.index < prior.index
                or (checkpoint.index == prior.index and checkpoint != prior)
            ):
                raise WorkspaceRefusal("checkpoint changed or regressed its native chain")

    @staticmethod
    def _weights(db: Journal, transaction: str) -> WeightsRow:
        row = db.one(WeightsRow, "SELECT * FROM weights WHERE id=?", (transaction,))
        if row is None:
            raise WorkspaceRefusal("weights transaction row was removed")
        return row

    def _validate_checkpoint(self, row: WeightsRow, checkpoint: pb.CheckpointRef) -> None:
        store = fill.store(self.store_root)
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

    def reserve_hold(
        self,
        db: Journal,
        owner: str,
        request: pb.DerivedRetentionRequest,
        *,
        release: bool = False,
        kind: str = "derived",
    ) -> None:
        self.owner(owner)
        if kind not in ("derived", "tree"):
            raise WorkspaceRefusal("unknown native retention kind")
        if (
            _DIGEST.fullmatch(request.weights_transaction_id) is None
            or _DIGEST.fullmatch(request.retention_id) is None
            or len(request.tensorfs_receipt_digest) != 32
        ):
            raise WorkspaceRefusal("native retention requires exact bounded identities")
        row = db.one(NativeHold, "SELECT * FROM holds WHERE id=?", (request.retention_id,))
        subject = (owner, request.weights_transaction_id, request.tensorfs_receipt_digest)
        if row is not None:
            if (row.owner, row.transaction_id, row.native_digest) != subject or row.kind != kind:
                raise WorkspaceRefusal("retention identity already belongs to another subject")
            if not release and row.state in ("releasing", "released"):
                raise WorkspaceRefusal("released retention cannot be recreated")
        else:
            original = db.execute(
                "SELECT 1 FROM weights WHERE owner=? AND id=? "
                "AND native_digest=? AND state='receipt'",
                subject,
            ).fetchone()
            retained = db.execute(
                "SELECT 1 FROM holds WHERE owner=? AND transaction_id=? "
                "AND native_digest=? AND kind=? AND state='held'",
                (*subject, kind),
            ).fetchone()
            if original is not None and kind != "derived":
                raise WorkspaceRefusal("weights receipt cannot authorize a source-tree hold")
            if original is None:
                from .workspace_native_memo import native_receipt_owned

                original = native_receipt_owned(db, owner, request, kind, release=release)
                if original is None and kind == "tree":
                    original = db.execute(
                        "SELECT 1 FROM byte_outputs WHERE owner=? AND id=? AND native_digest=? "
                        "AND (state='complete' OR (? AND state IN ('releasing','released')))",
                        (*subject, release),
                    ).fetchone()
            if original is None and kind == "tree":
                original = db.execute(
                    "SELECT 1 FROM input_tree_intakes WHERE owner=? AND id=? AND native_digest=? "
                    "AND (state='prepared' OR (? AND state IN ('aborting','released')))",
                    (*subject, release),
                ).fetchone()
            pending_release = False
            if release and original is None and retained is None:
                # Durable memo/lookup intent authorizes cancellation of a root
                # that may not have reached native creation before a crash.
                # It never authorizes retaining bytes or creating a new root.
                for table in ("operation_cache", "operation_lookups"):
                    claim = db.execute(
                        f"SELECT 1 FROM {table}, json_each(CAST(body AS TEXT),'$.holds') AS h "
                        "WHERE owner=? AND json_extract(h.value,'$.retention_id')=? "
                        "AND json_extract(h.value,'$.transaction_id')=? "
                        "AND json_extract(h.value,'$.native_receipt_digest')=? LIMIT 1",
                        (
                            owner,
                            request.retention_id,
                            request.weights_transaction_id,
                            documents.spell(request.tensorfs_receipt_digest),
                        ),
                    ).fetchone()
                    pending_release |= claim is not None
            if original is None and retained is None and not pending_release:
                raise WorkspaceRefusal("native result has no receipt owned by this workspace owner")
            db.execute(
                "INSERT INTO holds(id,owner,transaction_id,native_digest,state,kind) "
                "VALUES(?,?,?,?,?,?)",
                (request.retention_id, *subject, "releasing" if release else "retaining", kind),
            )

    def _change_hold(
        self,
        db: Journal,
        owner: str,
        request: pb.DerivedRetentionRequest,
        *,
        release: bool,
        cancel_lookup: bool = True,
        kind: str = "derived",
    ) -> pb.DerivedRetentionResult:
        self.reserve_hold(db, owner, request, release=release, kind=kind)
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
        from .workspace_native_memo import change_native_hold

        result = change_native_hold(self, db, owner, request, kind, release=release)
        current = db.execute(
            "SELECT state FROM holds WHERE id=?", (request.retention_id,)
        ).fetchone()
        if not release and current["state"] in ("releasing", "released"):
            change_native_hold(self, db, owner, request, kind, release=True)
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
        if release and kind == "tree":
            db.execute(
                "UPDATE input_tree_intakes SET state='released' WHERE intake_hold=? AND owner=?",
                (request.retention_id, owner),
            )
        if release:
            from cozy_runtime.internal.worker import workspace_memo

            try:
                workspace_memo.compact_released(db, owner)
            except Exception:
                # Native release and its tombstone are already durable. Keep
                # the larger lookup body for the next prune/cleanup to compact.
                import logging

                logging.getLogger(__name__).warning("workspace release compaction deferred")
        return result


def _stop_native_transaction(store_root: Path, transaction: str) -> None:
    """Fence an open native writer and abandon it, or dispose a committed result."""
    store = fill.store(store_root)
    observed = store.derived_lookup(transaction)
    if observed.get("state") == "open":
        if observed.get("writer_session_id"):
            store.derived_fence(transaction, observed["writer_session_id"])
        observed = store.derived_abandon(transaction)
    if observed.get("state") == "committed":
        store.derived_dispose(transaction)
