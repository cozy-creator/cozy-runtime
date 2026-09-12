"""Parent-owned native writers over the existing control lane and attempt spool.

The executor can read the Store and write its spool. Only the Runtime parent mutates
TensorFS; declaration/receipt authorization remains the existing host exchange.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import re
import stat
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._weights import MAX_WEIGHTS_SOURCE_READ
from cozy_runtime.internal import fill, storage_admission
from cozy_runtime.internal.weights_sink import source_config
from cozy_runtime.protocol import weights_limits

_TRANSACTION = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CHUNK = 1 << 20


def _part_bytes(part: Mapping[str, Any], dtypes: Mapping[str, int]) -> int:
    return int(math.prod(part["shape"])) * int(dtypes[part["dtype"]])


def _path(spool: Path, transaction: str, kind: str) -> Path:
    if _TRANSACTION.fullmatch(transaction) is None or kind not in {
        "request",
        "input",
        "output",
        "source-config",
    }:
        raise CapabilityError("invalid writer exchange identity", code="weights_writer_identity")
    return spool / f"weights-writer-{transaction[7:]}-{kind}"


def _facts(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@contextlib.contextmanager
def _read(path: Path, length: int, maximum: int) -> Iterator[BinaryIO]:
    if not 0 <= length <= maximum:
        raise CapabilityError(
            "writer bytes exceed the declared bound", code="weights_writer_length"
        )
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != length or before.st_nlink != 1:
            raise CapabilityError(
                "writer input is not an exact regular file", code="weights_writer_file"
            )
        yield source
        if _facts(before) != _facts(os.fstat(source.fileno())):
            raise CapabilityError(
                "writer input changed during consumption", code="weights_writer_file"
            )


def _write(path: Path, source: Any, maximum: int) -> int:
    # An old name is never followed or opened for overwrite, including a planted symlink.
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o644)
    length = 0
    with os.fdopen(fd, "wb") as target:
        while chunk := source.read(min(_CHUNK, maximum - length + 1)):
            if not isinstance(chunk, bytes) or length + len(chunk) > maximum:
                raise CapabilityError(
                    "writer stream exceeds its bound", code="weights_writer_length"
                )
            target.write(chunk)
            length += len(chunk)
    return length


class StoreProxy:
    """Keep native read-only metadata local, move writer operations to the parent."""

    def __init__(self, store: Any, spool: Path, exchange: Callable[..., dict[str, Any]]) -> None:
        self.store, self.spool, self.exchange = store, spool, exchange

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def config(self, manifest: str, name: str) -> bytes:
        path = _path(self.spool, manifest, "source-config")
        result = self.exchange(
            "weights_writer", {"operation": "source_config", "manifest": manifest, "name": name}
        )
        if not result.get("ok"):
            raise CapabilityError(
                str(result.get("detail", "parent config read refused")),
                code=str(result.get("code", "weights_source_refused")),
            )
        try:
            with _read(path, int(result["length"]), MAX_WEIGHTS_SOURCE_READ) as stream:
                return stream.read()
        finally:
            path.unlink(missing_ok=True)

    def begin_derived(
        self,
        transaction: str,
        epoch: int,
        *args: Any,
        work_fingerprint: str,
        checkpoint: tuple[str, int] | None = None,
    ) -> Any:
        import io

        data = json.dumps(
            {"args": args, "work_fingerprint": work_fingerprint, "checkpoint": checkpoint},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        length = _write(
            _path(self.spool, transaction, "request"),
            io.BytesIO(data),
            weights_limits.MAX_WEIGHTS_DECLARATION_BYTES,
        )
        writer = WriterProxy(self, transaction, epoch)
        writer.call("open", length=length)
        return writer


class WriterProxy:
    def __init__(self, store: StoreProxy, transaction: str, epoch: int) -> None:
        self.store, self.transaction, self.epoch = store, transaction, epoch

    def call(self, operation: str, **fields: Any) -> dict[str, Any]:
        result = self.store.exchange(
            "weights_writer",
            {
                "operation": operation,
                "weights_transaction_id": self.transaction,
                "writer_epoch": self.epoch,
                **fields,
            },
        )
        if not result.get("ok"):
            raise CapabilityError(
                str(result.get("detail", "parent writer refused")),
                code=str(result.get("code", "weights_writer_refused")),
            )
        return result

    def source_read_into(
        self, source: str, component: str, key: str, role: str, offset: int, into: memoryview
    ) -> None:
        length = len(into)
        self.call(
            "read",
            source=source,
            component=component,
            key=key,
            role=role,
            offset=offset,
            length=length,
        )
        path = _path(self.store.spool, self.transaction, "output")
        try:
            with _read(path, length, MAX_WEIGHTS_SOURCE_READ) as stream:
                if cast(Any, stream).readinto(into) != length:
                    raise CapabilityError("short source handoff", code="weights_writer_length")
        finally:
            path.unlink(missing_ok=True)

    def _add(self, operation: str, source: Any, **fields: Any) -> None:
        # The parent derives the exact payload bound from the locked native declaration.
        maximum = int(self.call("bound", target_operation=operation, **fields)["length"])
        path = _path(self.store.spool, self.transaction, "input")
        try:
            length = _write(path, source, maximum)
            self.call(operation, length=length, **fields)
        except BaseException:
            # A failed local spool copy must also relinquish its parent hold.
            with contextlib.suppress(Exception):
                self.fence()
            raise
        finally:
            path.unlink(missing_ok=True)

    def add_part(self, component: str, key: str, role: str, source: Any) -> None:
        self._add("part", source, component=component, key=key, role=role)

    def add_config(self, name: str, source: Any) -> None:
        self._add("config", source, name=name)

    def commit(self) -> Any:
        return self._json("commit", weights_limits.MAX_WEIGHTS_RECEIPT_BYTES)

    def _json(self, operation: str, maximum: int) -> Any:
        result = self.call(operation)
        path = _path(self.store.spool, self.transaction, "output")
        try:
            with _read(path, int(result["length"]), maximum) as stream:
                return json.load(stream)
        finally:
            path.unlink(missing_ok=True)

    def completed_parts(self) -> Any:
        return self._json("completed_parts", weights_limits.MAX_WEIGHTS_DECLARATION_BYTES)

    def completed_configs(self) -> Any:
        return self._json("completed_configs", weights_limits.MAX_WEIGHTS_DECLARATION_BYTES)

    def checkpoint(self, operation_id: str, slot: str, *, previous: Any = None) -> Any:
        return self.call("checkpoint", operation_id=operation_id, slot=slot, previous=previous)[
            "checkpoint"
        ]

    def fence(self) -> None:
        self.call("fence")


@dataclass(frozen=True)
class _Binding:
    epoch: int
    declaration_digest: bytes
    slot: str
    checkpoint: tuple[str, int] | None


class WriterBroker:
    """Live attempt-scoped handles only; native TensorFS owns every durable fact."""

    def __init__(
        self,
        store: Callable[[], Any],
        checkpoint: Callable[[Any, str, _Binding, dict[str, Any]], None] | None = None,
        store_root: Path | None = None,
    ) -> None:
        self.store = store
        self.record_checkpoint = checkpoint
        self.store_root = store_root
        self.bindings: dict[tuple[str, int, str], _Binding] = {}
        self.lock = threading.RLock()
        self.writers: dict[tuple[str, int, str], tuple[Any, dict[str, Any]]] = {}
        self.pending: dict[
            tuple[str, int, str], tuple[tuple[Any, ...], storage_admission.Lease | None]
        ] = {}

    @staticmethod
    def key(attempt: Any, transaction: str) -> tuple[str, int, str]:
        return attempt.request_id, attempt.attempt, transaction

    def payload_pending(self, attempt: Any) -> bool:
        """A source callback must not recursively acquire its writer's disk exclusion."""
        with self.lock:
            return any(key[:2] == (attempt.request_id, attempt.attempt) for key in self.pending)

    def authorize(
        self,
        attempt: Any,
        transaction: str,
        epoch: int,
        declaration: bytes,
        slot: str,
        checkpoint: tuple[str, int] | None = None,
    ) -> None:
        with self.lock:
            self.bindings[self.key(attempt, transaction)] = _Binding(
                epoch, declaration, slot, checkpoint
            )

    def bound(self, declaration: dict[str, Any], frame: Mapping[str, Any]) -> int:
        operation = frame.get("target_operation", frame.get("operation"))
        if operation == "config":
            if str(frame.get("name", "")) not in declaration["configs"]:
                raise CapabilityError("undeclared config", code="weights_writer_ungranted")
            # Native Manifest and config admission share DOC_MAX_BYTES; use its
            # published bound rather than narrowing configs to the intent-frame cap.
            return min(
                int(declaration["max_new_bytes"]), int(fill.tensorfs_module().manifest_max_bytes())
            )
        if operation != "part":
            raise CapabilityError("unknown writer payload", code="weights_writer_ungranted")
        part = declaration["components"][str(frame["component"])]["add"][str(frame["key"])][
            "parts"
        ][str(frame["role"])]
        return _part_bytes(part, fill.tensorfs_module().dtypes())

    @staticmethod
    def _payload(frame: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            frame.get("target_operation", frame.get("operation")),
            *(frame.get(name) for name in ("component", "key", "role", "name")),
        )

    @staticmethod
    def _refused(exc: Exception) -> dict[str, Any]:
        return {
            "ok": False,
            "code": str(getattr(exc, "code", "weights_writer_refused")),
            "detail": str(exc)[:1024],
        }

    def _bound(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        key = self.key(attempt, str(frame.get("weights_transaction_id", "")))
        with self.lock:
            if key in self.pending:
                return self._refused(
                    CapabilityError(
                        "writer payload is already pending", code="weights_writer_payload_pending"
                    )
                )
            result = self._handle(attempt, frame)
            if not result.get("ok"):
                return result
            current = self.writers[key]
            pending = (self._payload(frame), None)
            self.pending[key] = pending
        # Never wait for the global disk lock while holding the broker lock: a
        # different writer may need that lock to finish its already-held part.
        lease = None
        try:
            maximum = int(result["length"])
            lease = storage_admission.acquire(
                storage_admission.native_write(self.store_root, maximum),
                storage_admission.Write(attempt.spool, maximum + MAX_WEIGHTS_SOURCE_READ, 3),
            )
            with self.lock:
                if (
                    self.pending.get(key) is not pending
                    or self.writers.get(key) is not current
                    or attempt.canceling
                    or attempt.state != "running"
                ):
                    raise CapabilityError(
                        "writer closed during admission", code="weights_writer_closed"
                    )
                self.pending[key] = pending[0], lease
                lease = None  # The existing writer lifecycle now owns this hold.
            return result
        except Exception as exc:
            with self.lock:
                if self.pending.get(key) is pending:
                    self.pending.pop(key)
            return self._refused(exc)
        finally:
            if lease is not None:
                lease.close()

    def _source_config(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        from cozy_runtime.internal.worker.grants import MODEL_PREFIX

        manifest, name = str(frame.get("manifest", "")), str(frame.get("name", ""))
        try:
            path = _path(attempt.spool, manifest, "source-config")
            allowed = {
                row["digest"]: row["length"]
                for row in attempt.spec.get("inputs", [])
                if str(row.get("input_id", "")).startswith(MODEL_PREFIX)
            }
            length = allowed.get(manifest)
            if type(length) is not int or length <= 0:
                raise CapabilityError(
                    "weights config source is outside the immutable grant",
                    code="weights_source_ungranted",
                )
            with self.lock:
                if self.payload_pending(attempt):
                    raise CapabilityError(
                        "writer payload is pending", code="weights_writer_payload_pending"
                    )
            with (
                storage_admission.admit(
                    storage_admission.native_write(self.store_root),
                    storage_admission.Write(attempt.spool, MAX_WEIGHTS_SOURCE_READ, 1),
                ),
                self.lock,
            ):
                if attempt.canceling or attempt.state != "running":
                    raise CapabilityError(
                        "config read attempt is not running", code="weights_writer_closed"
                    )
                data = source_config(self.store(), manifest, length, name)
                written = _write(path, io.BytesIO(data), MAX_WEIGHTS_SOURCE_READ)
                return {"ok": True, "length": written}
        except Exception as exc:
            return self._refused(exc)

    def handle(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        operation = frame.get("operation")
        if operation == "source_config":
            return self._source_config(attempt, frame)
        if operation == "bound":
            return self._bound(attempt, frame)
        key = self.key(attempt, str(frame.get("weights_transaction_id", "")))
        with self.lock:
            pending = self.pending.get(key)
            borrowed = pending[1] if pending is not None else None
            if operation in {"part", "config"} and (
                pending is None or borrowed is None or pending[0] != self._payload(frame)
            ):
                return self._refused(
                    CapabilityError(
                        "writer payload has no matching admission", code="weights_writer_ungranted"
                    )
                )
            if pending is not None and operation not in {"part", "config", "read", "fence"}:
                return self._refused(
                    CapabilityError(
                        "writer payload is pending", code="weights_writer_payload_pending"
                    )
                )
        lease = borrowed
        try:
            if lease is None:
                if operation == "fence":
                    lease = storage_admission.Lease()  # Releasing work needs no fresh capacity.
                elif operation == "read":
                    length = int(frame.get("length", 0))
                    if not 0 < length <= MAX_WEIGHTS_SOURCE_READ:
                        raise CapabilityError(
                            "source read exceeds bound", code="weights_writer_length"
                        )
                    lease = storage_admission.acquire(
                        storage_admission.Write(attempt.spool, length, 1)
                    )
                else:
                    lease = storage_admission.acquire(
                        storage_admission.native_write(self.store_root),
                        storage_admission.Write(
                            attempt.spool, weights_limits.MAX_WEIGHTS_RECEIPT_BYTES, 2
                        ),
                    )
            with self.lock, lease.scope():
                result = self._handle(attempt, frame)
                if operation in {"part", "config", "fence"} or not result.get("ok"):
                    self.pending.pop(key, None)
                    if borrowed is not None:
                        borrowed.close()
        except Exception as exc:
            return self._refused(exc)
        finally:
            if lease is not None and lease is not borrowed:
                lease.close()
        if operation == "open" and result.get("ok"):
            try:
                self._preflight_writers(attempt)
            except Exception as exc:
                # The native writer retains completed parts, but no package may
                # start expensive work with an already-insufficient disk budget.
                with self.lock:
                    held = self.writers.pop(key, None)
                    if held is not None:
                        held[0].fence()
                return self._refused(exc)
        return result

    def _preflight_writers(self, attempt: Any) -> None:
        """Check remaining declared additions before handing writers to the package.

        This is a preflight, not a long-held reservation: the per-part write/read
        leases still arbitrate concurrent I/O. Native completion owns resume state;
        inherited roles and completed roles consume no new-payload budget here.
        """
        if not storage_admission.pressure_enabled():
            return
        dtypes = fill.tensorfs_module().dtypes()
        payload = largest = parts = 0
        with self.lock:
            for key, (writer, declaration) in self.writers.items():
                if key[:2] != (attempt.request_id, attempt.attempt):
                    continue
                completed = {tuple(row) for row in writer.completed_parts()}
                for component, target in declaration["components"].items():
                    for name, tensor in target["add"].items():
                        for role, part in tensor["parts"].items():
                            if "source" in part or (component, name, role) in completed:
                                continue
                            length = _part_bytes(part, dtypes)
                            payload += length
                            largest = max(largest, length)
                            parts += 1
        # Configs and commit metadata use the existing native document allowance.
        # One added role and its source-read window can coexist in the spool.
        with storage_admission.admit(
            storage_admission.native_write(self.store_root, payload, parts),
            storage_admission.Write(attempt.spool, largest + MAX_WEIGHTS_SOURCE_READ, 3),
        ):
            pass

    def _handle(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        import io

        transaction = str(frame.get("weights_transaction_id", ""))
        key = self.key(attempt, transaction)
        try:
            if attempt.canceling or attempt.state != "running" or attempt.spool is None:
                raise CapabilityError("writer attempt is not running", code="weights_writer_closed")
            binding = self.bindings.get(key)
            if binding is None or binding.epoch != frame.get("writer_epoch"):
                raise CapabilityError(
                    "writer has no current host binding", code="weights_writer_ungranted"
                )
            operation = str(frame.get("operation", ""))
            spool = attempt.spool
            store = self.store()
            if operation == "open":
                if key in self.writers:
                    raise CapabilityError("writer already opened", code="weights_writer_closed")
                with _read(
                    _path(spool, transaction, "request"),
                    int(frame["length"]),
                    weights_limits.MAX_WEIGHTS_DECLARATION_BYTES,
                ) as source:
                    request = json.load(source)
                if not isinstance(request, dict) or set(request) != {
                    "args",
                    "work_fingerprint",
                    "checkpoint",
                }:
                    raise CapabilityError(
                        "invalid writer request", code="weights_writer_declaration"
                    )
                args = request["args"]
                fingerprint = request["work_fingerprint"]
                checkpoint = request["checkpoint"]
                if (
                    not isinstance(args, list)
                    or len(args) != 5
                    or not fingerprint
                    or fingerprint != attempt.weights_work_fingerprint
                    or checkpoint != (list(binding.checkpoint) if binding.checkpoint else None)
                ):
                    raise CapabilityError(
                        "invalid writer declaration", code="weights_writer_declaration"
                    )
                from cozy_runtime.internal.worker.grants import MODEL_PREFIX

                permitted = {
                    row["digest"]: row["length"]
                    for row in attempt.spec.get("inputs", [])
                    if str(row.get("input_id", "")).startswith(MODEL_PREFIX)
                }
                outputs = {
                    # Canonical protobuf documents omit the zero scalar default.
                    row["output_id"]: row.get("max_bytes", 0)
                    for row in attempt.spec.get("outputs", [])
                }
                if type(args[4]) is not int or args[4] < 0 or args[4] != outputs.get(binding.slot):
                    raise CapabilityError(
                        "writer output exceeds invocation grant", code="weights_writer_ungranted"
                    )
                args[0] = {alias: tuple(ref) for alias, ref in args[0].items()}
                if any(
                    len(ref) != 2 or permitted.get(ref[0]) != ref[1] for ref in args[0].values()
                ):
                    raise CapabilityError(
                        "writer source is outside invocation grant", code="weights_writer_ungranted"
                    )
                args[3] = [tuple(row) for row in args[3]]
                declaration = store.derived_declaration(*args, work_fingerprint=fingerprint)
                if hashlib.sha256(declaration).digest() != binding.declaration_digest:
                    raise CapabilityError(
                        "writer declaration changed after intent", code="weights_writer_declaration"
                    )
                writer = store.begin_derived(
                    transaction,
                    binding.epoch,
                    *args,
                    work_fingerprint=fingerprint,
                    checkpoint=binding.checkpoint,
                )
                self.writers[key] = writer, json.loads(declaration)
                return {"ok": True}
            held = self.writers.get(key)
            if held is None:
                raise CapabilityError("writer is not open", code="weights_writer_closed")
            writer, declaration = held
            if operation == "bound":
                return {"ok": True, "length": self.bound(declaration, frame)}
            if operation == "read":
                length = int(frame["length"])
                if not 0 < length <= MAX_WEIGHTS_SOURCE_READ:
                    raise CapabilityError("source read exceeds bound", code="weights_writer_length")
                value = bytearray(length)
                writer.source_read_into(
                    str(frame["source"]),
                    str(frame["component"]),
                    str(frame["key"]),
                    str(frame["role"]),
                    int(frame["offset"]),
                    memoryview(value),
                )
                _write(_path(spool, transaction, "output"), io.BytesIO(value), length)
            elif operation in {"part", "config"}:
                length = int(frame["length"])
                maximum = self.bound(declaration, frame)
                if operation == "part" and length != maximum:
                    raise CapabilityError(
                        "part length differs from declaration", code="weights_writer_length"
                    )
                with _read(_path(spool, transaction, "input"), length, maximum) as source:
                    if operation == "part":
                        writer.add_part(
                            str(frame["component"]), str(frame["key"]), str(frame["role"]), source
                        )
                    else:
                        writer.add_config(str(frame["name"]), source)
            elif operation in {"commit", "completed_parts", "completed_configs"}:
                facts = getattr(writer, operation)()
                if operation == "commit":
                    del self.writers[key]
                raw = json.dumps(facts, separators=(",", ":"), sort_keys=True).encode()
                length = _write(
                    _path(spool, transaction, "output"),
                    io.BytesIO(raw),
                    weights_limits.MAX_WEIGHTS_RECEIPT_BYTES
                    if operation == "commit"
                    else weights_limits.MAX_WEIGHTS_DECLARATION_BYTES,
                )
                return {"ok": True, "length": length}
            elif operation == "checkpoint":
                if (
                    frame.get("operation_id") != attempt.request_id
                    or frame.get("slot") != binding.slot
                ):
                    raise CapabilityError(
                        "checkpoint differs from writer scope", code="weights_writer_ungranted"
                    )
                previous = frame.get("previous")
                facts = writer.checkpoint(
                    attempt.request_id,
                    binding.slot,
                    previous=tuple(previous) if previous is not None else None,
                )
                if facts.get("head") is not None and self.record_checkpoint is not None:
                    self.record_checkpoint(attempt, transaction, binding, facts)
                return {"ok": True, "checkpoint": facts}
            elif operation == "fence":
                writer.fence()
                del self.writers[key]
            else:
                raise CapabilityError("unknown writer operation", code="weights_writer_ungranted")
            return {"ok": True}
        except Exception as exc:
            held = self.writers.pop(key, None)
            if held is not None:
                with contextlib.suppress(Exception):
                    held[0].fence()
            return {
                "ok": False,
                "code": str(getattr(exc, "code", "weights_writer_refused")),
                "detail": str(exc)[:1024],
            }

    def close_attempt(self, attempt: Any) -> None:
        with self.lock:
            for key in list(self.bindings):
                if key[:2] == (attempt.request_id, attempt.attempt):
                    self._close(key)

    def close(self, transaction: str | None = None) -> None:
        with self.lock:
            for key in list(self.bindings):
                if transaction is None or key[2] == transaction:
                    self._close(key)

    def _close(self, key: tuple[str, int, str]) -> None:
        held = self.writers.pop(key, None)
        self.bindings.pop(key, None)
        pending = self.pending.pop(key, None)
        try:
            if held is not None:
                held[0].fence()
        finally:
            if pending is not None and pending[1] is not None:
                pending[1].close()
