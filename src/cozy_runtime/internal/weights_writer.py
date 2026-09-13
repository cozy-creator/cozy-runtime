"""Parent-owned native writers over the existing control lane and attempt spool.

The executor can read the Store and write its spool. Only the Runtime parent mutates
TensorFS; declaration/receipt authorization remains the existing host exchange.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import re
import socket
import stat
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._weights import MAX_WEIGHTS_SOURCE_READ
from cozy_runtime.internal import storage_admission
from cozy_runtime.internal.seam import DescriptorReply
from cozy_runtime.internal.weights_sink import source_config
from cozy_runtime.protocol import weights_limits

_TRANSACTION = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CHUNK = 1 << 20


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
        result = self.exchange(
            "weights_writer",
            {
                "operation": "open",
                "weights_transaction_id": transaction,
                "writer_epoch": epoch,
                "length": length,
            },
            descriptor=True,
        )
        if not result.get("ok"):
            raise CapabilityError(
                str(result.get("detail", "parent writer refused")),
                code=str(result.get("code", "weights_writer_refused")),
            )
        from tensorfs.derived import DerivedTransaction

        return DerivedTransaction(result["descriptor"])


@dataclass(frozen=True)
class _Binding:
    epoch: int
    declaration_digest: bytes
    slot: str
    checkpoint: tuple[str, int] | None


@dataclass
class _LiveWriter:
    writer: Any
    arguments: tuple[Any, ...]
    fingerprint: str
    binding: _Binding
    stop_socket: socket.socket
    writing: bool = False


class WriterBroker:
    """Runtime execution admission around TensorFS-owned native writer channels."""

    def __init__(
        self,
        store: Callable[[], Any],
        checkpoint: Callable[[Any, str, _Binding, dict[str, Any]], None] | None = None,
        store_root: Path | None = None,
        receipt: Callable[[Any, str, _Binding, dict[str, Any]], None] | None = None,
    ) -> None:
        self.store = store
        self.record_checkpoint = checkpoint
        self.record_receipt = receipt
        self.store_root = store_root
        self.bindings: dict[tuple[str, int, str], _Binding] = {}
        self.lock = threading.RLock()
        self.writers: dict[tuple[str, int, str], _LiveWriter] = {}

    @staticmethod
    def key(attempt: Any, transaction: str) -> tuple[str, int, str]:
        return attempt.request_id, attempt.attempt, transaction

    def payload_pending(self, attempt: Any) -> bool:
        with self.lock:
            return any(
                key[:2] == (attempt.request_id, attempt.attempt) and live.writing
                for key, live in self.writers.items()
            )

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

    @staticmethod
    def _refused(exc: Exception) -> dict[str, Any]:
        return {
            "ok": False,
            "code": str(getattr(exc, "code", "weights_writer_refused")),
            "detail": str(exc)[:1024],
        }

    def _current(self, attempt: Any, transaction: str, binding: _Binding) -> None:
        with self.lock:
            if (
                attempt.canceling
                or attempt.state != "running"
                or self.bindings.get(self.key(attempt, transaction)) != binding
            ):
                raise CapabilityError(
                    "writer attempt is no longer current", code="weights_writer_closed"
                )

    def _source_config(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        from cozy_runtime.internal.worker.grants import MODEL_PREFIX

        manifest, name = str(frame.get("manifest", "")), str(frame.get("name", ""))
        path = _path(attempt.spool, manifest, "source-config")
        allowed = {
            row["digest"]: row["length"]
            for row in attempt.spec.get("inputs", [])
            if str(row.get("input_id", "")).startswith(MODEL_PREFIX)
        }
        length = allowed.get(manifest)
        if type(length) is not int or length <= 0:
            raise CapabilityError(
                "config source is outside the invocation", code="weights_source_ungranted"
            )
        if self.payload_pending(attempt):
            raise CapabilityError(
                "writer payload is pending", code="weights_writer_payload_pending"
            )
        with storage_admission.admit(
            storage_admission.native_write(self.store_root),
            storage_admission.Write(attempt.spool, MAX_WEIGHTS_SOURCE_READ, 1),
        ):
            if attempt.canceling or attempt.state != "running":
                raise CapabilityError(
                    "config read attempt is not running", code="weights_writer_closed"
                )
            data = source_config(self.store(), manifest, length, name)
            written = _write(path, io.BytesIO(data), MAX_WEIGHTS_SOURCE_READ)
            return {"ok": True, "length": written}

    def handle(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        try:
            if frame.get("operation") == "source_config":
                return self._source_config(attempt, frame)
            if frame.get("operation") != "open":
                raise CapabilityError(
                    "writer bytes require the granted TensorFS channel",
                    code="weights_writer_ungranted",
                )
            return self._open(attempt, frame)
        except Exception as exc:
            return self._refused(exc)

    def _open(self, attempt: Any, frame: Mapping[str, Any]) -> DescriptorReply:
        from tensorfs.derived import serve_derived

        from cozy_runtime.internal.worker.grants import MODEL_PREFIX

        transaction = str(frame.get("weights_transaction_id", ""))
        key = self.key(attempt, transaction)
        with self.lock:
            binding = self.bindings.get(key)
            if binding is None or binding.epoch != frame.get("writer_epoch") or key in self.writers:
                raise CapabilityError(
                    "writer has no unused current binding", code="weights_writer_ungranted"
                )
        self._current(attempt, transaction, binding)
        checkpoint_recorder, receipt_recorder = self.record_checkpoint, self.record_receipt
        if checkpoint_recorder is None or receipt_recorder is None:
            raise CapabilityError(
                "native writer has no durable execution recorder", code="weights_host_unavailable"
            )
        with _read(
            _path(attempt.spool, transaction, "request"),
            int(frame["length"]),
            weights_limits.MAX_WEIGHTS_DECLARATION_BYTES,
        ) as source:
            request = json.load(source)
        if not isinstance(request, dict) or set(request) != {
            "args",
            "work_fingerprint",
            "checkpoint",
        }:
            raise CapabilityError("invalid writer request", code="weights_writer_declaration")
        args, fingerprint, checkpoint = (
            request["args"],
            request["work_fingerprint"],
            request["checkpoint"],
        )
        if (
            not isinstance(args, list)
            or len(args) != 5
            or not fingerprint
            or fingerprint != attempt.weights_work_fingerprint
            or checkpoint != (list(binding.checkpoint) if binding.checkpoint else None)
        ):
            raise CapabilityError("invalid writer declaration", code="weights_writer_declaration")
        permitted = {
            row["digest"]: row["length"]
            for row in attempt.spec.get("inputs", [])
            if str(row.get("input_id", "")).startswith(MODEL_PREFIX)
        }
        outputs = {
            row["output_id"]: row.get("max_bytes", 0) for row in attempt.spec.get("outputs", [])
        }
        if type(args[4]) is not int or args[4] < 0 or args[4] != outputs.get(binding.slot):
            raise CapabilityError(
                "writer output exceeds invocation grant", code="weights_writer_ungranted"
            )
        args[0] = {alias: tuple(ref) for alias, ref in args[0].items()}
        if any(len(ref) != 2 or permitted.get(ref[0]) != ref[1] for ref in args[0].values()):
            raise CapabilityError(
                "writer source is outside invocation grant", code="weights_writer_ungranted"
            )
        args[3] = [tuple(row) for row in args[3]]
        store = self.store()
        declaration = store.derived_declaration(*args, work_fingerprint=fingerprint)
        if hashlib.sha256(declaration).digest() != binding.declaration_digest:
            raise CapabilityError(
                "writer declaration changed after intent", code="weights_writer_declaration"
            )
        with self._preflight_writers(attempt, transaction, binding, tuple(args), fingerprint):
            self._current(attempt, transaction, binding)
            writer = store.begin_derived(
                transaction,
                binding.epoch,
                *args,
                work_fingerprint=fingerprint,
                checkpoint=binding.checkpoint,
            )
        client, server = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        live = _LiveWriter(writer, tuple(args), fingerprint, binding, server.dup())
        with self.lock:
            self.writers[key] = live

        @contextlib.contextmanager
        def admit(operation: str, native_bound: int) -> Iterator[None]:
            self._current(attempt, transaction, binding)
            lease = (
                storage_admission.Lease()
                if operation in {"fence", "read", "completed_parts", "completed_configs"}
                else storage_admission.acquire(
                    storage_admission.native_write(self.store_root, native_bound)
                )
            )
            try:
                with lease.scope():
                    self._current(attempt, transaction, binding)
                    with self.lock:
                        live.writing = operation in {"part", "config"}
                    yield
            finally:
                with self.lock:
                    live.writing = False
                lease.close()

        def serve() -> None:
            try:
                serve_derived(
                    writer,
                    server.detach(),
                    operation_id=attempt.request_id,
                    slot=binding.slot,
                    checkpoint=binding.checkpoint,
                    check_current=lambda: self._current(attempt, transaction, binding),
                    admit_io=admit,
                    record_checkpoint=lambda facts: checkpoint_recorder(
                        attempt, transaction, binding, facts
                    ),
                    record_receipt=lambda facts: receipt_recorder(
                        attempt, transaction, binding, facts
                    ),
                )
            except Exception:
                # TensorFS sends the typed refusal on this same channel before raising.
                logging.getLogger(__name__).debug(
                    "derived writer channel closed with refusal", exc_info=True
                )
            finally:
                with self.lock:
                    if self.writers.get(key) is live:
                        self.writers.pop(key)
                live.stop_socket.close()

        try:
            threading.Thread(target=serve, name="tensorfs-derived", daemon=True).start()
        except BaseException:
            client.close()
            server.close()
            live.stop_socket.close()
            writer.fence()
            with self.lock:
                self.writers.pop(key, None)
            raise
        return DescriptorReply({"ok": True}, client)

    @contextlib.contextmanager
    def _preflight_writers(
        self,
        attempt: Any,
        transaction: str,
        binding: _Binding,
        arguments: tuple[Any, ...],
        fingerprint: str,
    ) -> Iterator[None]:
        if not storage_admission.pressure_enabled():
            with storage_admission.admit(storage_admission.native_write(self.store_root)):
                yield
            return
        store = self.store()
        estimate = store.derived_write_estimate(
            transaction,
            binding.epoch,
            *arguments,
            work_fingerprint=fingerprint,
            checkpoint=binding.checkpoint,
        )
        payload, parts = int(estimate["payload_bytes"]), int(estimate["parts"])
        with self.lock:
            existing = tuple(self.writers.items())
        for key, live in existing:
            if key[:2] != (attempt.request_id, attempt.attempt):
                continue
            pending = store.derived_write_estimate(
                key[2],
                live.binding.epoch,
                *live.arguments,
                work_fingerprint=live.fingerprint,
            )
            payload += int(pending["payload_bytes"])
            parts += int(pending["parts"])
        with storage_admission.admit(
            storage_admission.native_write(self.store_root, payload, parts)
        ):
            yield

    def close_attempt(self, attempt: Any) -> None:
        with self.lock:
            keys = [
                key for key in self.bindings if key[:2] == (attempt.request_id, attempt.attempt)
            ]
        for key in keys:
            self._close(key)

    def close(self, transaction: str | None = None) -> None:
        with self.lock:
            keys = [key for key in self.bindings if transaction is None or key[2] == transaction]
        for key in keys:
            self._close(key)

    def _close(self, key: tuple[str, int, str]) -> None:
        with self.lock:
            self.bindings.pop(key, None)
            live = self.writers.pop(key, None)
        if live is not None:
            with contextlib.suppress(OSError):
                live.stop_socket.shutdown(socket.SHUT_RDWR)
            live.writer.fence()
