"""Bind admitted executions to worker-owned native TensorFS source and output handles.

Only bounded declarations and receipts use the executor control exchange. TensorFS
owns source/config transport, streamed parts, checkpoints and native writer semantics.
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
from typing import TYPE_CHECKING, Any, BinaryIO

from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.internal import storage_admission
from cozy_runtime.internal.seam import DescriptorReply
from cozy_runtime.protocol import weights_limits

if TYPE_CHECKING:
    from tensorfs.derived import Derivation, DerivedTransaction, SourceCapability

    from cozy_runtime.author._artifacts import ModelArtifact

_TRANSACTION = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CHUNK = 1 << 20


class ExecutionStorage:
    """Execution bindings only: the executor receives TensorFS handles, never a Store."""

    def __init__(
        self, spool: Path, exchange: Callable[..., dict[str, Any]], outputs: Mapping[str, int]
    ) -> None:
        self.spool, self.exchange, self.outputs = spool, exchange, dict(outputs)
        self.opened: dict[str, str] = {}

    def _call(self, operation: str, *, descriptor: bool = False, **fields: Any) -> dict[str, Any]:
        answer = self.exchange(
            "weights_writer", {"operation": operation, **fields}, descriptor=descriptor
        )
        if not answer.get("ok"):
            raise CapabilityError(
                str(answer.get("detail", "execution storage refused")),
                code=str(answer.get("code", "weights_host_refused")),
            )
        return answer

    def source(self, manifest: str) -> SourceCapability:
        from tensorfs.derived import SourceCapability

        answer = self._call("source", manifest=manifest, descriptor=True)
        return SourceCapability.from_fd(manifest, answer["length"], answer["descriptor"])

    def open_output(self, output_slot: str, definition: Derivation) -> DerivedTransaction:
        from tensorfs.derived import DerivedTransaction

        from cozy_runtime.internal.weights_sink import exchange_file

        maximum = self.outputs.get(output_slot)
        if maximum is None:
            raise CapabilityError("output is not declared", code="weights_output_ungranted")
        data = json.dumps(definition.native_arguments(maximum), separators=(",", ":")).encode()
        length = _write(
            exchange_file(self.spool, output_slot, "derivation"),
            io.BytesIO(data),
            weights_limits.MAX_WEIGHTS_DECLARATION_BYTES,
        )
        answer = self._call("output", output_slot=output_slot, length=length, descriptor=True)
        writer = DerivedTransaction.accept(answer["descriptor"])
        self.opened[answer["transaction"]] = output_slot
        return writer

    def adopt_model(self, facts: Mapping[str, Any]) -> ModelArtifact:
        from cozy_runtime import canonical_json
        from cozy_runtime.author._artifacts import ModelArtifact, ObjectRef
        from cozy_runtime.internal.weights_sink import exchange_file

        transaction = str(facts.get("transaction_id", ""))
        slot = self.opened.get(transaction)
        if slot is None:
            raise CapabilityError("receipt has no opened output", code="weights_receipt_mismatch")
        length = _write(
            exchange_file(self.spool, slot, "native-receipt"),
            io.BytesIO(canonical_json.encode(dict(facts))),
            weights_limits.MAX_WEIGHTS_RECEIPT_BYTES,
        )
        answer = self._call("adopt", output_slot=slot, transaction=transaction, length=length)
        return ModelArtifact(
            answer["request_id"],
            slot,
            ObjectRef(answer["manifest"], answer["manifest_length"]),
            answer["receipt_digest"],
        )


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
        bind_output: Callable[[Any, str, bytes], tuple[str, int]] | None = None,
    ) -> None:
        self.store = store
        self.record_checkpoint = checkpoint
        self.record_receipt = receipt
        self.bind_output = bind_output
        self.store_root = store_root
        self.bindings: dict[tuple[str, int, str], _Binding] = {}
        self.lock = threading.RLock()
        self.writers: dict[tuple[str, int, str], _LiveWriter] = {}
        self.completed_channels: dict[tuple[str, int, str], set[socket.socket]] = {}
        self.sources: dict[tuple[str, int], set[socket.socket]] = {}

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

    def handle(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        try:
            if frame.get("operation") == "source":
                return self._source(attempt, frame)
            if frame.get("operation") == "output":
                return self._output(attempt, frame)
            if frame.get("operation") == "adopt":
                return self._adopt(attempt, frame)
            raise CapabilityError(
                "writer bytes require the granted TensorFS channel", code="weights_writer_ungranted"
            )
        except Exception as exc:
            return self._refused(exc)

    def _source(self, attempt: Any, frame: Mapping[str, Any]) -> DescriptorReply:
        from tensorfs.derived import Source, serve_source

        from cozy_runtime.internal.worker.grants import MODEL_PREFIX

        manifest = str(frame.get("manifest", ""))
        allowed = {
            row["digest"]: row["length"]
            for row in attempt.spec.get("inputs", [])
            if str(row.get("input_id", "")).startswith(MODEL_PREFIX)
        }
        length = allowed.get(manifest)
        if type(length) is not int or length <= 0:
            raise CapabilityError(
                "source is outside the invocation", code="weights_source_ungranted"
            )
        key = attempt.request_id, attempt.attempt
        accepted_digest = getattr(attempt, "digest", None)
        client, server = socket.socketpair()
        stop = server.dup()
        with self.lock:
            if len(self.sources.get(key, ())) >= 16:
                client.close()
                server.close()
                stop.close()
                raise CapabilityError("too many open source handles", code="weights_reader_bounds")
            self.sources.setdefault(key, set()).add(stop)

        def current() -> None:
            with self.lock:
                if (
                    attempt.canceling
                    or attempt.state != "running"
                    or (attempt.request_id, attempt.attempt) != key
                    or getattr(attempt, "digest", None) != accepted_digest
                    or stop not in self.sources.get(key, ())
                ):
                    raise CapabilityError(
                        "source attempt is no longer current", code="weights_writer_closed"
                    )
                if self.payload_pending(attempt):
                    raise CapabilityError(
                        "writer payload is pending", code="weights_writer_payload_pending"
                    )

        @contextlib.contextmanager
        def admit() -> Iterator[None]:
            current()  # Refuse nested inspection before waiting on the native disk exclusion.
            with storage_admission.admit(storage_admission.native_write(self.store_root)):
                current()
                yield

        def serve() -> None:
            try:
                serve_source(
                    self.store(),
                    Source(manifest, length),
                    server.detach(),
                    check_current=current,
                    admit_io=admit,
                )
            except Exception:
                logging.getLogger(__name__).debug(
                    "source channel closed with refusal", exc_info=True
                )
            finally:
                with self.lock:
                    held = self.sources.get(key)
                    if held is not None:
                        held.discard(stop)
                        if not held:
                            self.sources.pop(key, None)
                stop.close()

        try:
            current()
            threading.Thread(target=serve, name="tensorfs-source", daemon=True).start()
        except BaseException:
            client.close()
            server.close()
            with self.lock:
                self.sources[key].discard(stop)
                if not self.sources[key]:
                    self.sources.pop(key)
            stop.close()
            raise
        return DescriptorReply({"ok": True, "length": length}, client)

    def _arguments(self, attempt: Any, slot: str, args: Any) -> tuple[Any, ...]:
        from cozy_runtime.internal.worker.grants import MODEL_PREFIX

        if not isinstance(args, list) or len(args) not in (5, 6):
            raise CapabilityError("invalid derivation arguments", code="weights_writer_declaration")
        outputs = {
            row["output_id"]: row.get("max_bytes", 0) for row in attempt.spec.get("outputs", [])
        }
        if type(args[4]) is not int or args[4] < 0 or args[4] != outputs.get(slot):
            raise CapabilityError(
                "output exceeds invocation grant", code="weights_writer_ungranted"
            )
        permitted = {
            row["digest"]: row["length"]
            for row in attempt.spec.get("inputs", [])
            if str(row.get("input_id", "")).startswith(MODEL_PREFIX)
        }
        if not isinstance(args[0], dict) or any(
            not isinstance(ref, (list, tuple))
            or len(ref) != 2
            or not isinstance(ref[0], str)
            or type(ref[1]) is not int
            or permitted.get(ref[0]) != ref[1]
            for ref in args[0].values()
        ):
            raise CapabilityError(
                "source is outside invocation grant", code="weights_source_ungranted"
            )
        args[0] = {alias: tuple(ref) for alias, ref in args[0].items()}
        args[3] = [tuple(row) for row in args[3]]
        return tuple(args)

    def _output(self, attempt: Any, frame: Mapping[str, Any]) -> DescriptorReply:
        from cozy_runtime.internal.weights_sink import exchange_file

        slot = str(frame.get("output_slot", ""))
        if self.bind_output is None:
            raise CapabilityError("output has no execution owner", code="weights_host_unavailable")
        if attempt.canceling or attempt.state != "running":
            raise CapabilityError("output attempt is not running", code="weights_writer_closed")
        if self.payload_pending(attempt):
            raise CapabilityError(
                "writer payload is pending", code="weights_writer_payload_pending"
            )
        with _read(
            exchange_file(attempt.spool, slot, "derivation"),
            int(frame["length"]),
            weights_limits.MAX_WEIGHTS_DECLARATION_BYTES,
        ) as source:
            args = self._arguments(attempt, slot, json.load(source))
        fingerprint = attempt.weights_work_fingerprint
        declaration = self.store().derived_declaration(*args, work_fingerprint=fingerprint)
        transaction, epoch = self.bind_output(attempt, slot, declaration)
        binding = self.bindings[self.key(attempt, transaction)]
        if binding.epoch != epoch:
            raise CapabilityError("output epoch changed", code="weights_writer_closed")
        result = self._open_native(attempt, transaction, binding, args, fingerprint)
        result["transaction"] = transaction
        return result

    def _adopt(self, attempt: Any, frame: Mapping[str, Any]) -> dict[str, Any]:
        from cozy_runtime import canonical_json
        from cozy_runtime.internal.weights_sink import exchange_file

        slot, transaction = str(frame.get("output_slot", "")), str(frame.get("transaction", ""))
        binding = self.bindings.get(self.key(attempt, transaction))
        if binding is None or binding.slot != slot:
            raise CapabilityError("receipt has no bound output", code="weights_receipt_mismatch")
        self._current(attempt, transaction, binding)
        with _read(
            exchange_file(attempt.spool, slot, "native-receipt"),
            int(frame["length"]),
            weights_limits.MAX_WEIGHTS_RECEIPT_BYTES,
        ) as source:
            facts = canonical_json.decode(source.read())
        native = self.store().derived_lookup(transaction)
        if native.get("state") != "committed" or native.get("receipt") != facts:
            raise CapabilityError(
                "receipt differs from native custody", code="weights_receipt_mismatch"
            )
        if self.record_receipt is None:
            raise CapabilityError(
                "receipt has no execution recorder", code="weights_host_unavailable"
            )
        self.record_receipt(attempt, transaction, binding, facts)
        self._current(attempt, transaction, binding)
        return {
            "ok": True,
            "request_id": attempt.request_id,
            "manifest": "sha256:" + facts["manifest"]["sha256"],
            "manifest_length": facts["manifest"]["length"],
            "receipt_digest": "sha256:" + hashlib.sha256(canonical_json.encode(facts)).hexdigest(),
        }

    def _open_native(
        self,
        attempt: Any,
        transaction: str,
        binding: _Binding,
        args: tuple[Any, ...],
        fingerprint: str,
    ) -> DescriptorReply:
        from tensorfs.derived import serve_derived, serve_derived_replay

        from cozy_runtime import canonical_json

        key = self.key(attempt, transaction)
        self._current(attempt, transaction, binding)
        with self.lock:
            if key in self.writers:
                raise CapabilityError("writer is already open", code="weights_writer_closed")
            if len(self.completed_channels.get(key, ())) >= 16:
                raise CapabilityError(
                    "too many pending receipt deliveries", code="weights_writer_closed"
                )
        checkpoint_recorder, receipt_recorder = self.record_checkpoint, self.record_receipt
        if checkpoint_recorder is None or receipt_recorder is None:
            raise CapabilityError(
                "writer has no execution recorder", code="weights_host_unavailable"
            )
        store = self.store()
        existing = store.derived_lookup(transaction)
        replay = existing.get("receipt") if existing.get("state") == "committed" else None
        writer = None
        if replay is not None:
            disposition = existing.get("disposition")
            if not isinstance(disposition, dict) or disposition.get("kind") not in {
                "pending",
                "adopted",
            }:
                raise CapabilityError(
                    "output is no longer retained", code="weights_transaction_closed"
                )
            if (
                replay.get("transaction_id") != transaction
                or hashlib.sha256(canonical_json.encode(replay.get("declaration"))).digest()
                != binding.declaration_digest
            ):
                raise CapabilityError(
                    "native receipt differs from intent", code="weights_receipt_mismatch"
                )
            manifest = replay["manifest"]
            try:
                store.inspect_derived_source(
                    "sha256:" + manifest["sha256"], manifest["length"], list(args[1]), list(args[2])
                )
            except Exception as exc:
                raise CapabilityError(
                    "completed output is not retained and complete",
                    code="weights_receipt_unavailable",
                ) from exc
        else:
            with self._preflight_writers(attempt, transaction, binding, args, fingerprint):
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

        def record_receipt(facts: dict[str, Any]) -> None:
            receipt_recorder(attempt, transaction, binding, facts)
            # TensorFS sends the successful receipt after this callback. Release
            # writer ownership here so immediate replay cannot race the serving
            # thread's later cleanup. An active or unrecorded writer still excludes
            # another opener; a completed output reopens through native custody.
            with self.lock:
                if self.writers.get(key) is live:
                    # Keep terminal delivery interruptible if its peer stops
                    # reading; replay admission and channel lifetime are separate.
                    self.completed_channels.setdefault(key, set()).add(live.stop_socket)
                    self.writers.pop(key)

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
                if replay is not None:
                    serve_derived_replay(
                        replay,
                        server.detach(),
                        check_current=lambda: self._current(attempt, transaction, binding),
                        record_receipt=record_receipt,
                    )
                else:
                    assert writer is not None
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
                        record_receipt=record_receipt,
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
                    channels = self.completed_channels.get(key)
                    if channels is not None:
                        channels.discard(live.stop_socket)
                        if not channels:
                            self.completed_channels.pop(key)
                live.stop_socket.close()

        try:
            threading.Thread(target=serve, name="tensorfs-derived", daemon=True).start()
        except BaseException:
            client.close()
            server.close()
            live.stop_socket.close()
            if writer is not None:
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
        payload, parts = 0, 1
        if storage_admission.pressure_enabled():
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
                if key[:2] != (attempt.request_id, attempt.attempt) or live.writer is None:
                    continue
                pending = store.derived_write_estimate(
                    key[2],
                    live.binding.epoch,
                    *live.arguments,
                    work_fingerprint=live.fingerprint,
                )
                payload += int(pending["payload_bytes"])
                parts += int(pending["parts"])
        lease = storage_admission.acquire(
            storage_admission.native_write(self.store_root, payload, parts)
        )
        try:
            with lease.scope():
                yield
        finally:
            lease.close()

    def close_attempt(self, attempt: Any) -> None:
        with self.lock:
            sources = self.sources.pop((attempt.request_id, attempt.attempt), set())
        for channel in sources:
            with contextlib.suppress(OSError):
                channel.shutdown(socket.SHUT_RDWR)
        with self.lock:
            keys = [
                key for key in self.bindings if key[:2] == (attempt.request_id, attempt.attempt)
            ]
        for key in keys:
            self._close(key)

    def close(self, transaction: str | None = None) -> None:
        if transaction is None:
            with self.lock:
                sources = [channel for channels in self.sources.values() for channel in channels]
                self.sources.clear()
            for channel in sources:
                with contextlib.suppress(OSError):
                    channel.shutdown(socket.SHUT_RDWR)
        with self.lock:
            keys = [key for key in self.bindings if transaction is None or key[2] == transaction]
        for key in keys:
            self._close(key)

    def _close(self, key: tuple[str, int, str]) -> None:
        with self.lock:
            self.bindings.pop(key, None)
            live = self.writers.pop(key, None)
            completed = self.completed_channels.pop(key, set())
        for channel in completed:
            with contextlib.suppress(OSError):
                channel.shutdown(socket.SHUT_RDWR)
        if live is not None:
            with contextlib.suppress(OSError):
                live.stop_socket.shutdown(socket.SHUT_RDWR)
            if live.writer is not None:
                live.writer.fence()
