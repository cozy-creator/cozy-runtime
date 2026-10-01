"""The machine tier of memoized Model methods (tracker #301): the Worker's policy over
TensorFS keyed roots.

Entries are the small results of costly encoders, shared by every request and account on
this machine; the key is their only scope, and only the Worker writes them. The tier manages
itself, with no purge command:
- an entry expires `memo.ttl` after it was written (the scheduled `maintain` job);
- beyond `memo.store_bytes`, and under storage pressure, the least recently used go first;
- when the disk is very low a new entry is not written: a miss, never an error.
Dropping an entry releases its root; TensorFS GC deletes the bytes.
"""

from __future__ import annotations

import hashlib
import os
import queue
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypedDict, cast

from cozy_runtime.author._executor_requests import (
    MemoEntry,
    MemoStored,
    StageMemoLookup,
    StageMemoStore,
)
from cozy_runtime.internal import fill, local_storage_admission, storage_admission
from cozy_runtime.internal.config import MemoConfig
from cozy_runtime.internal.executor_commands import MemoSettings
from cozy_runtime.internal.worker import store_gc
from cozy_runtime.internal.worker.workspace_byte_outputs import (
    Blob,
    manifest_members,
    payload_manifest,
)

SPACE = "stage-memo-1"
#: Entries waiting for their TensorFS commit; a full queue drops new writes, counted.
QUEUE = 256


class KeyedRoot(TypedDict):
    key: str
    manifest_digest: str
    manifest_length: int
    bytes: int
    created_unix_ms: int


class _Lease(Protocol):
    def read_into(self, obj: str, length: int, off: int, size: int, into: bytearray) -> None: ...
    def release(self) -> None: ...


class KeyedRoots(Protocol):
    """The TensorFS Store surface this tier uses (keyed roots arrived after 0.3.78)."""

    def put_keyed_root(
        self, space: str, key: str, manifest: bytes, files: list[tuple[str, str]]
    ) -> KeyedRoot: ...
    def keyed_roots(self, space: str) -> list[KeyedRoot]: ...
    def drop_keyed_root(self, space: str, key: str) -> bool: ...
    def manifest(self, digest: str) -> dict[str, bytes]: ...
    def acquire_manifest(self, digest: str) -> _Lease: ...


@dataclass(slots=True)
class Row:
    #: on disk, manifest included
    bytes: int
    #: the entry file alone
    length: int
    created_ms: int
    cost_ms: float = 0.0
    blob: str = ""
    manifest: str = ""
    #: the staged file served until the commit lands
    staged: Path | None = None


@dataclass(slots=True)
class Claim:
    """A key being computed by one attempt; other lookups of it wait for the outcome."""

    attempt: str
    done: threading.Event = field(default_factory=threading.Event)


class MachineMemo:
    def __init__(
        self, store_root: Path, staging: Path, config: MemoConfig, note: Callable[[str, str], None]
    ) -> None:
        self.root = store_root
        self.staging = staging
        self.config = config
        self.note = note
        store = fill.ensure_store(store_root)
        #: Keyed roots shipped in TensorFS 0.3.84. An older TensorFS keeps executors on their
        #: in-process tier: the capability is detected, never required.
        self.available = config.store_bytes > 0 and hasattr(store, "put_keyed_root")
        self.native = cast(KeyedRoots, store)
        self.rows: OrderedDict[str, Row] = OrderedDict()
        self.claims: dict[str, Claim] = {}
        self.disputed: set[tuple[str, str]] = set()
        self.counts: dict[str, int] = {}
        self.pending: queue.Queue[tuple[str, str, str]] = queue.Queue(QUEUE)
        self.lock = threading.Lock()
        if self.available:
            staging.mkdir(parents=True, exist_ok=True)
            for stale in staging.iterdir():
                stale.unlink(missing_ok=True)
            try:
                roots = self.native.keyed_roots(SPACE)
            except Exception as exc:  # an unreadable tier is off, never a failed boot
                note("memo", f"stage memo tier off: {exc}"[:400])
                self.available, roots = False, []
            for root in roots:
                self.rows[root["key"]] = Row(
                    root["bytes"],
                    root["bytes"] - root["manifest_length"],
                    root["created_unix_ms"],
                    manifest=root["manifest_digest"],
                )
            threading.Thread(target=self._commit, daemon=True, name="stage-memo-commit").start()

    def settings(self) -> MemoSettings:
        return MemoSettings(
            process_bytes=self.config.process_bytes,
            entry_bytes=self.config.entry_bytes if self.available else 0,
        )

    def _count(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    # ---------------------------------------------------------------------- the exchange

    def lookup(self, attempt: str, spool: Path, request: StageMemoLookup) -> MemoEntry:
        """A verified copy in the attempt's spool, or a miss that claims the key."""
        key = request.key
        while True:
            with self.lock:
                if (request.stage, request.numerics) in self.disputed:
                    return MemoEntry(ok=True, disputed=True)
                row = self.rows.get(key)
                if row is not None:
                    self.rows.move_to_end(key)
                    break
                claim = self.claims.get(key)
                if claim is None or claim.attempt == attempt:
                    self.claims[key] = Claim(attempt)
                    self._count("misses")
                    return MemoEntry(ok=True)
            # The producer stores, abandons, or its attempt ends: progress, not a clock.
            claim.done.wait()
        try:
            raw = self._read(row)
        except Exception as exc:  # a missing or corrupt entry is a miss, and goes
            self.note("memo.integrity", f"stage memo {key[:16]} unreadable: {exc}"[:400])
            self._drop([key])
            return self.lookup(attempt, spool, request)
        target = spool / f"stage-memo-{key[:16]}-{time.monotonic_ns()}.safetensors"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        with os.fdopen(os.open(target, flags, 0o644), "wb") as stream:  # never through a link
            stream.write(raw)
        self._count("hits")
        return MemoEntry(ok=True, local=str(target), cost_ms=row.cost_ms)

    def store(self, attempt: str, spool: Path, request: StageMemoStore) -> MemoStored:
        try:
            return self._store(spool, request)
        finally:
            self._settle(request.key, attempt)

    def _store(self, spool: Path, request: StageMemoStore) -> MemoStored:
        if not request.local:
            return MemoStored(ok=True, reason=request.reason)
        if not self.available or not 0 < request.length <= self.config.entry_bytes:
            return MemoStored(ok=True, reason="too_large" if self.available else "unavailable")
        try:
            # Very low disk: the entry is not written. Admission evicts nothing for a cache.
            storage_admission.acquire(
                storage_admission.native_write(self.root, request.length, 1), recover=False
            ).close()
        except storage_admission.StorageRefusal:
            self._count("disk_low")
            return MemoStored(ok=True, reason="disk_low")
        raw = _spooled(spool, Path(request.local), request.length)
        if raw is None:
            return MemoStored(ok=True, reason="local")
        digest = hashlib.sha256(raw).hexdigest()
        with self.lock:
            existing = self.rows.get(request.key)
        if existing is not None:
            agree = existing.blob in ("", digest)
            if not agree:
                with self.lock:
                    self.disputed.add((request.stage, request.numerics))
            return MemoStored(ok=True, reason="present", disputed=not agree)
        staged = self.staging / f"{request.key}.safetensors"
        staged.write_bytes(raw)
        with self.lock:
            self.rows[request.key] = Row(
                request.length,
                request.length,
                int(time.time() * 1000),
                request.cost_ms,
                digest,
                staged=staged,
            )
        try:
            self.pending.put_nowait((request.key, request.stage, request.numerics))
        except queue.Full:
            self._count("queue_full")
            self._drop([request.key])
            return MemoStored(ok=True, reason="queue_full")
        self._count("stored")
        return MemoStored(ok=True, stored=True)

    def _settle(self, key: str, attempt: str) -> None:
        with self.lock:
            claim = self.claims.get(key)
            if claim is not None and claim.attempt == attempt:
                del self.claims[key]
                claim.done.set()

    def release(self, attempt: str) -> None:
        """The attempt ended: whatever it was computing, its waiters compute themselves."""
        with self.lock:
            for key in [k for k, c in self.claims.items() if c.attempt == attempt]:
                self.claims.pop(key).done.set()

    def _read(self, row: Row) -> bytes:
        if row.staged is not None:
            return row.staged.read_bytes()
        if not row.blob:
            body = self.native.manifest(row.manifest)["manifest"]
            (member,) = manifest_members(body, row.length)
            row.blob = member.blob.sha256
        raw = bytearray(row.length)
        lease = self.native.acquire_manifest(row.manifest)
        try:
            lease.read_into("sha256:" + row.blob, row.length, 0, row.length, raw)
        finally:
            lease.release()
        if hashlib.sha256(raw).hexdigest() != row.blob:
            raise ValueError("entry bytes differ from their digest")
        return bytes(raw)

    # ---------------------------------------------------------------------- commits

    def _commit(self) -> None:
        while True:
            key, stage, numerics = self.pending.get()
            with self.lock:
                row = self.rows.get(key)
            if row is None or row.staged is None:
                continue
            staged = row.staged
            try:
                root = self.native.put_keyed_root(
                    SPACE,
                    key,
                    payload_manifest(Blob(row.blob, row.length)),
                    [("payload", str(staged))],
                )
            except Exception as exc:
                if getattr(exc, "code", "") == "TRANSACTION_CONFLICT":
                    # Two computations disagree: the first stays, reuse stops for the scope.
                    with self.lock:
                        self.disputed.add((stage, numerics))
                    self.note("memo.disputed", f"stage memo {key[:16]} computed twice, differently")
                else:
                    self.note("memo.commit", f"stage memo {key[:16]} not stored: {exc}"[:400])
                self._drop([key])
                continue
            with self.lock:
                if self.rows.get(key) is row:
                    row.manifest, row.staged = root["manifest_digest"], None
                    row.bytes = root["bytes"]
            staged.unlink(missing_ok=True)

    # ---------------------------------------------------------------------- self-management

    def maintain(self) -> int:
        """The scheduled job: the TTL, then the cap, then storage pressure."""
        expired, trimmed, relieved = self.expire(), self.trim(), self.relieve()
        if expired or trimmed or relieved:
            self.note(
                "memo",
                f"stage memo: {len(self.rows)} entries, {self._bytes()} B; dropped {expired} "
                f"expired, {trimmed} over the cap, {relieved} for disk; counts {self.counts}",
            )
        return expired + trimmed + relieved

    def expire(self, now_ms: int | None = None) -> int:
        """Drop entries written more than `memo.ttl` ago."""
        now = int(time.time() * 1000) if now_ms is None else now_ms
        with self.lock:
            rows = self.rows.items()
            expired = [key for key, row in rows if row.created_ms + self.config.ttl_ms <= now]
        return self._drop(expired)

    def trim(self) -> int:
        """Keep the tier within `memo.store_bytes`, least recently used first."""
        return self.reclaim(self._bytes() - self.config.store_bytes)

    def relieve(self) -> int:
        """Give space back while the disk is under the storage-pressure policy's watermark."""
        try:
            return self.reclaim(local_storage_admission.pressure_target(self.root))
        except OSError:
            return 0

    def reclaim(self, target_bytes: int) -> int:
        """Drop least recently used entries until `target_bytes` are released."""
        victims: list[str] = []
        freed = 0
        with self.lock:
            for key, row in self.rows.items():
                if freed >= target_bytes:
                    break
                victims.append(key)
                freed += row.bytes
        return self._drop(victims)

    def _bytes(self) -> int:
        with self.lock:
            return sum(row.bytes for row in self.rows.values())

    def _drop(self, keys: list[str]) -> int:
        committed = 0
        for key in keys:
            with self.lock:
                row = self.rows.pop(key, None)
            if row is None:
                continue
            if row.staged is not None:
                row.staged.unlink(missing_ok=True)
            else:
                committed += bool(self.native.drop_keyed_root(SPACE, key))
            self._count("dropped")
        if committed:
            store_gc.collect(self.root)
        return len(keys)


def _spooled(spool: Path, local: Path, length: int) -> bytes | None:
    """The executor's entry file, only from its own attempt spool and never through a link."""
    if local.parent != spool:
        return None
    try:
        descriptor = os.open(local, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            return None
        raw = stream.read(length + 1)
    return raw if len(raw) == length else None


def interval(config: MemoConfig) -> float:
    """How often `maintain` runs: an entry outlives its TTL by at most a twenty-fourth of
    it, and the job runs at least hourly."""
    return max(1.0, min(config.ttl_ms / 24_000, 3600.0))


def scheduled(
    stop: threading.Event,
    jobs: list[tuple[float, Callable[[], object]]],
    note: Callable[[str, str], None],
) -> None:
    """The Worker's scheduled maintenance: each job runs every `interval` seconds. A job that
    fails is noted and runs again at its next turn."""
    due = [time.monotonic() + interval for interval, _ in jobs]
    while not stop.wait(max(0.0, min(due) - time.monotonic())):
        now = time.monotonic()
        for index, (interval, job) in enumerate(jobs):
            if due[index] <= now:
                due[index] = now + interval
                try:
                    job()
                except Exception as exc:  # one job's failure is that pass's, not the lane's
                    note("maintenance", f"{getattr(job, '__name__', job)} failed: {exc}"[:400])
