"""Privileged native weights and ordered workspace observations.

Workspace is the shared durable authority on local and private workers. The
executor exchanges bounded writer operations with its parent; intent, checkpoint
and receipt observations leave on WorkerControl before the final outcome.
RuntimeWeights retains its bounded Read/Upload operations, but its old Host-ACK
Exchange no longer grants writer authority.
"""

from __future__ import annotations

import hashlib
import io
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast
from urllib.parse import urlparse

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal import egress, fill, weights_sink
from cozy_runtime.internal.weights_writer import WriterAttempt, WriterBinding, WriterBroker
from cozy_runtime.internal.worker.workspace import WeightsTransaction, Workspace
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb


class _LeaseReader:
    """A read-only file-like view of ONE object in a live TensorFS lease.

    The uploader streams through this, so an 8 GiB object is never a bytes object: at most one
    chunk is resident. Reads are served in whole chunks regardless of the size asked for, which
    an HTTP body reader accepts and which keeps an 8 GiB PUT to thousands of reads, not millions.
    """

    __slots__ = ("closed", "lease", "length", "object_id", "offset")

    def __init__(self, lease: fill.ReadLease, object_id: str, length: int) -> None:
        self.lease = lease
        self.object_id = object_id
        self.length = length
        self.offset = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("weights reader is closed")
        remaining = self.length - self.offset
        if remaining <= 0:
            return b""
        want = egress.CHUNK if size is None or size < 0 else max(size, egress.CHUNK)
        want = min(want, remaining)
        target = bytearray(want)
        self.lease.read_into(self.object_id, self.length, self.offset, want, target)
        self.offset += want
        return bytes(target)

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        """Rewind for a re-send (`egress.put_from`): the lease is random-access."""
        if whence != 0 or offset != 0:
            raise ValueError("a lease reader rewinds to its start only")
        self.offset = 0
        return 0

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.lease.release()


class WeightsExchangeError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class _CheckpointFacts(msgspec.Struct, frozen=True):
    """The native writer's checkpoint observation, as TensorFS reports it."""

    head: str
    head_length: int
    plan_digest: str
    index: int
    bytes: int


@dataclass(slots=True)
class _Held:
    transaction_id: str
    writer_epoch: int
    receipt: pb.WeightsReceiptRef
    objects: dict[str, pb.WeightsObjectSource]
    manifest_id: str
    manifest_bytes: bytes
    manifest: pb.Ref


#: Typed refusals this exchange can report, mapped from the code that raised them. An egress
#: code with no entry lands on TRANSFER_FAILED, which is what an unclassified failure is.
_UPLOAD_REFUSALS: dict[str, pb.WeightsUploadRefusal] = {
    "unknown_transaction": pb.WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_TRANSACTION,
    "stale_writer": pb.WEIGHTS_UPLOAD_REFUSAL_STALE_WRITER,
    "unknown_object": pb.WEIGHTS_UPLOAD_REFUSAL_UNKNOWN_OBJECT,
    "source_ref_mismatch": pb.WEIGHTS_UPLOAD_REFUSAL_SOURCE_REF_MISMATCH,
    "source_unavailable": pb.WEIGHTS_UPLOAD_REFUSAL_SOURCE_UNAVAILABLE,
    "egress_host_not_allowed": pb.WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED,
    "egress_blocked_address": pb.WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED,
    "egress_scheme": pb.WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED,
    "egress_url": pb.WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED,
    "egress_redirect_refused": pb.WEIGHTS_UPLOAD_REFUSAL_GRANT_REFUSED,
}


class WeightsExchange:
    """One process's native handles and ordered workspace observations."""

    def __init__(
        self,
        *,
        store_root: Path | None,
        stop: threading.Event,
        owner_scope: Callable[[], str],
        stamp: Callable[[pb.WeightsIntentFrame], pb.WeightsIntentFrame] | None = None,
        note: Callable[[str, str], None] | None = None,
        allow_private_egress: bool = False,
        workspace: Workspace | None = None,
    ) -> None:
        #: The current owner envelope qualifies restore and object transfer requests.
        #: The identity default is used only by isolated native transport tests.
        self.stamp = stamp or (lambda message: message)
        self.stop = stop
        self.owner_scope = owner_scope
        self.note = note or (lambda _kind, _step: None)
        self.store_root = store_root
        self.workspace = workspace
        # Local proof only; production leaves this false, exactly as the supervisor's own
        # `allowPrivateWeightsUploads` does. It widens nothing but the address classes an
        # upload may reach, and only for a caller that constructed this object saying so.
        self.allow_private_egress = allow_private_egress
        self.store: fill.Store | None = None
        self._lock = threading.Lock()
        self._bindings: dict[str, tuple[int, bytes]] = {}
        self._held: dict[str, _Held] = {}
        self._abandoned: set[str] = set()
        self.writer_broker = WriterBroker(
            self._store,
            checkpoint=self.record_checkpoint,
            receipt=self.record_native_receipt,
            store_root=store_root,
            bind_output=self.bind_output,
        )

    def bind_output(
        self, attempt: WriterAttempt, output_slot: str, declaration: bytes
    ) -> tuple[str, int]:
        """Assign the execution identity after TensorFS canonicalizes the declaration."""
        transaction = weights_sink.weights_transaction_id(
            self.owner_scope(), attempt.request_id, documents.spell(attempt.digest), output_slot
        )
        epoch = self.intent(
            attempt,
            output_slot=output_slot,
            transaction_id=transaction,
            declaration_digest=hashlib.sha256(declaration).digest(),
            declaration=declaration,
        )
        return transaction, epoch

    def record_native_receipt(
        self,
        attempt: WriterAttempt,
        transaction: str,
        binding: WriterBinding,
        facts: Mapping[str, object],
    ) -> None:
        """Record the native winner before acknowledging the scoped TensorFS commit."""
        native = weights_sink.native_receipt(facts)
        if (
            native.transaction_id != transaction
            or documents.raw(native.declaration_digest) != binding.declaration_digest
        ):
            raise WeightsExchangeError(
                "weights_receipt_identity_mismatch", "native commit differs from intent"
            )
        wrapped = weights_sink.receipt_from_native(
            binding.slot, transaction, facts, replayed=False, request_id=attempt.request_id
        )
        _, canonical_receipt, digest = weights_sink.protocol_receipt(
            wrapped,
            owner_scope=self.owner_scope(),
            request_id=attempt.request_id,
            invocation_spec_digest=documents.spell(attempt.digest),
        )
        self.receipt(
            attempt,
            output_slot=binding.slot,
            transaction_id=transaction,
            receipt_digest=digest,
            canonical_receipt=canonical_receipt,
        )

    def record_checkpoint(
        self,
        attempt: WriterAttempt,
        transaction: str,
        binding: WriterBinding,
        facts: Mapping[str, object],
    ) -> None:
        """Publish the parent native writer's own observation, never child-authored refs."""
        self._require_open(transaction)
        observed = msgspec.convert(facts, _CheckpointFacts)
        checkpoint = pb.CheckpointRef(
            head=pb.Ref(digest=documents.raw(observed.head), length=observed.head_length),
            plan_digest=documents.raw(observed.plan_digest),
            index=observed.index,
            bytes=observed.bytes,
        )
        observed_checkpoint = pb.WeightsCheckpointFrame(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=attempt.digest,
            output_slot=binding.slot,
            weights_transaction_id=transaction,
            writer_epoch=binding.epoch,
            tensorfs_declaration_digest=binding.declaration_digest,
            checkpoint=checkpoint,
        )
        if self.workspace is None:
            raise WeightsExchangeError("workspace_unavailable", "worker has no workspace custody")
        try:
            self.workspace.checkpoint(self.owner_scope(), observed_checkpoint)
        except Exception as exc:
            raise WeightsExchangeError(
                str(getattr(exc, "code", "workspace_refused")),
                "workspace checkpoint recording refused",
            ) from exc

    def _checkpoint_pending(
        self,
        request: pb.WeightsIntentReadyRequest | pb.CheckpointTransferRequest,
        subject: pb.WeightsCheckpointSubject,
        *,
        attempt_ordinal: int | None = None,
        declaration: bytes | None = None,
    ) -> WeightsTransaction:
        current = self.stamp(pb.WeightsIntentFrame())
        if (
            request.record_owner_epoch != current.record_owner_epoch
            or request.worker_boot_id != current.worker_boot_id
            or request.control_stream_epoch != 0
            or self.workspace is None
        ):
            raise WeightsExchangeError(
                "weights_checkpoint_scope", "checkpoint owner is not current"
            )
        try:
            row = self.workspace.weights_row(self.owner_scope(), subject.weights_transaction_id)
        except ValueError as exc:
            raise WeightsExchangeError(
                "weights_checkpoint_scope", "checkpoint has no owned intent"
            ) from exc
        if (
            row["request"] != subject.request_id
            or row["epoch"] != subject.writer_epoch
            or row["spec"] != subject.invocation_spec_digest
            or row["slot"] != subject.output_slot
            or row["declaration_digest"] != subject.tensorfs_declaration_digest
            or row["state"] != "intent"
            or row["ready"]
            or row["attempt_fenced"]
            or row["attempt_state"] not in ("accepted", "running")
            or self.stop.is_set()
            or (attempt_ordinal is not None and row["ordinal"] != attempt_ordinal)
            or (declaration is not None and row["declaration"] != declaration)
        ):
            raise WeightsExchangeError(
                "weights_checkpoint_scope", "checkpoint differs from pending intent"
            )
        return row

    @contextmanager
    def restore_checkpoint(
        self, request: pb.CheckpointTransferRequest
    ) -> Iterator[tuple[str, int]]:
        subject = request.subject.weights
        row = self._checkpoint_pending(request, subject)
        yield subject.weights_transaction_id, subject.writer_epoch
        current = self._checkpoint_pending(request, subject)
        if current["ordinal"] != row["ordinal"]:
            raise WeightsExchangeError(
                "weights_checkpoint_scope", "writer changed during restoration"
            )

    def validate_checkpoint(
        self, request: pb.ValidateWeightsCheckpointRequest
    ) -> pb.ValidateWeightsCheckpointResult:
        intent, subject = request.intent, request.intent.weights
        result = pb.ValidateWeightsCheckpointResult(weights=subject)
        if intent.HasField("checkpoint"):
            result.checkpoint.CopyFrom(intent.checkpoint)
        try:
            if request.ByteSize() > weights_limits.MAX_INLINE_CONTROL_BYTES:
                raise WeightsExchangeError(
                    "weights_checkpoint_scope", "checkpoint validation exceeds its bound"
                )
            # Omitted bytes name the declaration by reference: the subject's digest
            # already binds the journaled declaration, whatever its size.
            declaration = request.tensorfs_declaration_canonical_bytes or None
            row = self._checkpoint_pending(
                intent, subject, attempt_ordinal=intent.attempt_ordinal, declaration=declaration
            )
            if intent.HasField("checkpoint"):
                assert self.workspace is not None
                self.workspace._validate_checkpoint(row, intent.checkpoint)
            self._checkpoint_pending(
                intent, subject, attempt_ordinal=intent.attempt_ordinal, declaration=declaration
            )
            result.valid = True
        except Exception as exc:
            result.safe_code = str(getattr(exc, "code", "weights_checkpoint_invalid"))[:128]
            result.safe_detail = f"checkpoint validation refused ({type(exc).__name__})"
        return result

    def intent(
        self,
        attempt: WriterAttempt,
        *,
        output_slot: str,
        transaction_id: str,
        declaration_digest: bytes,
        declaration: bytes,
    ) -> int:
        """Journal the writer intent and authorize its native writer; returns its epoch."""
        if self.workspace is None:
            raise WeightsExchangeError("workspace_unavailable", "worker has no workspace custody")
        self._require_open(transaction_id)
        intent = pb.WeightsIntentFrame(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=attempt.digest,
            output_slot=output_slot,
            weights_transaction_id=transaction_id,
            tensorfs_declaration_digest=declaration_digest,
            tensorfs_declaration_canonical_bytes=declaration,
            tensorfs_declaration_length=len(declaration),
        )
        try:
            row = self.workspace.begin_weights(self.owner_scope(), intent)
            epoch = int(row["epoch"])
            # The native writer is authorized by Workspace, never by a Host ACK.
            ready = pb.WeightsIntentReadyRequest(
                attempt_ordinal=attempt.attempt,
                weights=pb.WeightsCheckpointSubject(
                    request_id=attempt.request_id,
                    invocation_spec_digest=attempt.digest,
                    output_slot=output_slot,
                    weights_transaction_id=transaction_id,
                    writer_epoch=epoch,
                    tensorfs_declaration_digest=declaration_digest,
                ),
            )
            if row["checkpoint"]:
                ready.checkpoint.CopyFrom(pb.CheckpointRef.FromString(row["checkpoint"]))
            try:
                row = self.workspace.ready_weights(self.owner_scope(), ready)
            except Exception as exc:
                if ready.HasField("checkpoint") and getattr(exc, "code", None) in {
                    "ROOT_ABSENT",
                    "OBJECT_ABSENT",
                }:
                    raise WeightsExchangeError(
                        "weights_checkpoint_unavailable",
                        "accepted execution checkpoint is unavailable",
                    ) from exc
                raise
            with self._lock:
                self._bindings[transaction_id] = (epoch, declaration_digest)
            restored = (
                pb.CheckpointRef.FromString(row["restore_checkpoint"])
                if row["restore_checkpoint"]
                else None
            )
            checkpoint = (
                (documents.spell(restored.head.digest), restored.head.length) if restored else None
            )
            self.writer_broker.authorize(
                attempt, transaction_id, epoch, declaration_digest, output_slot, checkpoint
            )
            if row["receipt"]:
                reference = pb.WeightsReceiptRef(
                    weights_receipt_digest=row["receipt_digest"],
                    weights_receipt_canonical_bytes=row["receipt"],
                )
                held = self._hold(transaction_id, epoch, reference)
                attempt.weights_receipts[output_slot] = held.receipt
            return epoch
        except WeightsExchangeError:
            raise
        except Exception as exc:
            raise WeightsExchangeError(
                str(getattr(exc, "code", "workspace_refused")),
                "workspace writer authorization refused",
            ) from exc

    def receipt(
        self,
        attempt: WriterAttempt,
        *,
        output_slot: str,
        transaction_id: str,
        receipt_digest: bytes,
        canonical_receipt: bytes,
    ) -> None:
        if (
            len(receipt_digest) != 32
            or not canonical_receipt
            or len(canonical_receipt) > weights_limits.MAX_WEIGHTS_RECEIPT_BYTES
            or hashlib.sha256(canonical_receipt).digest() != receipt_digest
        ):
            raise WeightsExchangeError(
                "weights_receipt_invalid", "receipt bytes or digest are invalid"
            )
        reference = pb.WeightsReceiptRef(
            weights_receipt_digest=receipt_digest,
            weights_receipt_canonical_bytes=canonical_receipt,
        )
        receipt = documents.parse(canonical_receipt, pb.WeightsReceipt)
        if (
            receipt.request_id != attempt.request_id
            or receipt.owner_authority_scope != self.owner_scope()
            or receipt.invocation_spec_digest != documents.spell(attempt.digest)
            or receipt.output_slot != output_slot
            or receipt.weights_transaction_id != transaction_id
        ):
            raise WeightsExchangeError(
                "weights_receipt_identity_mismatch", "receipt does not bind this attempt/output"
            )
        held = self._hold(transaction_id, self._writer_epoch(transaction_id), reference)
        observed_receipt = pb.WeightsReceiptFrame(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=attempt.digest,
            output_slot=output_slot,
            weights_transaction_id=transaction_id,
            writer_epoch=held.writer_epoch,
            tensorfs_declaration_digest=self._declaration_digest(transaction_id),
            weights_receipt=reference,
            objects=[held.objects[key] for key in sorted(held.objects)],
            manifest=held.manifest,
        )
        if self.workspace is None:
            raise WeightsExchangeError("workspace_unavailable", "worker has no workspace custody")
        try:
            self.workspace.record_receipt(self.owner_scope(), observed_receipt)
        except Exception as exc:
            self._drop(transaction_id)
            raise WeightsExchangeError(
                str(getattr(exc, "code", "workspace_refused")),
                "workspace receipt recording refused",
            ) from exc
        attempt.weights_receipts[output_slot] = reference

    def close(self) -> None:
        """Forget receipt metadata; active transports release their own read leases."""
        self.writer_broker.close()
        with self._lock:
            self._held = {}

    def _require_open(self, transaction_id: str) -> None:
        with self._lock:
            if transaction_id in self._abandoned:
                raise WeightsExchangeError(
                    "weights_transaction_abandoned", "the Host refused this transaction"
                )

    def abandon(self, transaction_id: str) -> None:
        """Stop late receipt/transfer registration; native disposition owns the roots."""
        self.writer_broker.close(transaction_id)
        with self._lock:
            self._abandoned.add(transaction_id)
            self._held.pop(transaction_id, None)

    def upload(self, request: pb.WeightsUploadRequest) -> pb.WeightsUploadResult:
        """Stream ONE held object to the exact presigned URL the supervisor granted
        (`RuntimeWeights.Upload`). The bytes never enter the host process."""
        result = pb.WeightsUploadResult(
            weights_transaction_id=request.weights_transaction_id,
            writer_epoch=request.writer_epoch,
            object_id=request.object_id,
            operation_id=request.operation_id,
            grant_revision=request.grant_revision,
        )
        try:
            source, length = self._upload_source(request)
        except WeightsExchangeError as exc:
            result.outcome = pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
            result.refusal = _UPLOAD_REFUSALS.get(exc.code, pb.WEIGHTS_UPLOAD_REFUSAL_UNSPECIFIED)
            result.safe_detail = exc.detail[:4096]
            return result
        grant = request.grant
        host = urlparse(grant.url).hostname or ""
        try:
            receipt = egress.put_from(
                grant.url,
                # The grant IS the allowlist: one host, the one the control plane signed.
                egress.EgressPolicy(allowed_hosts=(host,), allow_private=self.allow_private_egress),
                cast(BinaryIO, source),
                expected_length=length,
                media_type="application/octet-stream",
                what=f"weights object {request.object_id}",
                required_headers={header.name: header.value for header in grant.required_headers},
                accept_precondition_failed=True,
            )
        except egress.EgressRefusal as exc:
            result.outcome = pb.WEIGHTS_UPLOAD_OUTCOME_REFUSED
            result.refusal = _UPLOAD_REFUSALS.get(
                exc.code, pb.WEIGHTS_UPLOAD_REFUSAL_TRANSFER_FAILED
            )
            result.safe_detail = exc.detail[:4096]
            return result
        finally:
            source.close()
        result.transferred_bytes = receipt.length
        result.http_status = receipt.http_status
        result.etag = receipt.etag
        if receipt.http_status == egress.HTTP_PRECONDITION_FAILED:
            # The immutable key already holds these bytes. The supervisor's ledger and
            # tensorhub's finalize pass decide what that means; this is only the status.
            result.outcome = pb.WEIGHTS_UPLOAD_OUTCOME_ALREADY_PRESENT
        else:
            result.outcome = pb.WEIGHTS_UPLOAD_OUTCOME_UPLOADED
            result.checksum_sha256 = "sha256:" + receipt.sha256.hex()
        return result

    def _upload_source(
        self, request: pb.WeightsUploadRequest
    ) -> tuple[io.BytesIO | _LeaseReader, int]:
        """Bind the requested object to what this Runtime actually holds, or refuse."""
        with self._lock:
            held = self._held.get(request.weights_transaction_id)
        if held is None:
            raise WeightsExchangeError(
                "unknown_transaction", "weights transaction is not held by this Runtime"
            )
        if request.writer_epoch != held.writer_epoch:
            raise WeightsExchangeError("stale_writer", "writer epoch is stale")
        row = held.objects.get(request.object_id)
        if row is None:
            raise WeightsExchangeError(
                "unknown_object", "object is outside the exact receipt inventory"
            )
        if request.source_ref != row.source_ref:
            raise WeightsExchangeError(
                "source_ref_mismatch", "source reference does not bind this object"
            )
        if request.length != row.length or request.grant.length != row.length:
            raise WeightsExchangeError(
                "source_unavailable", "granted length is not the exact object length"
            )
        if request.grant.object_id != request.object_id:
            raise WeightsExchangeError("source_ref_mismatch", "the grant names a different object")
        if row.source_ref.startswith("tensorfs-manifest:"):
            return io.BytesIO(held.manifest_bytes), row.length
        try:
            lease = self._store().acquire(held.manifest_id, [(row.object_id, row.length)])
        except Exception as exc:
            raise WeightsExchangeError(
                "source_unavailable", f"TensorFS source unavailable: {type(exc).__name__}"
            ) from exc
        return _LeaseReader(lease, row.object_id, row.length), row.length

    @staticmethod
    def _receipt_ref(reference: pb.WeightsReceiptRef) -> bytes:
        raw = reference.weights_receipt_canonical_bytes
        digest = hashlib.sha256(raw).digest()
        if (
            not raw
            or len(raw) > weights_limits.MAX_WEIGHTS_RECEIPT_BYTES
            or digest != reference.weights_receipt_digest
        ):
            raise WeightsExchangeError(
                "weights_receipt_invalid", "receipt ref digest does not match exact bytes"
            )
        documents.parse(raw, pb.WeightsReceipt)
        return digest

    def _hold(
        self,
        transaction_id: str,
        writer_epoch: int,
        reference: pb.WeightsReceiptRef,
    ) -> _Held:
        self._receipt_ref(reference)
        store = self._store()
        lifecycle = weights_sink.lookup(store, transaction_id)
        if lifecycle.state != "committed":
            raise WeightsExchangeError(
                "weights_transaction_uncommitted", "receipt has no committed TensorFS transaction"
            )
        if lifecycle.disposition.kind == "released":
            self.abandon(transaction_id)
            raise WeightsExchangeError(
                "weights_transaction_abandoned", "committed output was already released"
            )
        with self._lock:
            if transaction_id in self._abandoned:
                raise WeightsExchangeError("weights_transaction_abandoned", "output was abandoned")
            existing = self._held.get(transaction_id)
        if existing is not None:
            if existing.writer_epoch != writer_epoch or existing.receipt != reference:
                raise WeightsExchangeError(
                    "weights_receipt_mismatch", "held transaction carries another receipt"
                )
            return existing
        value = documents.parse(reference.weights_receipt_canonical_bytes, pb.WeightsReceipt)
        if value.weights_transaction_id != transaction_id:
            raise WeightsExchangeError(
                "weights_receipt_identity_mismatch", "receipt names another transaction"
            )
        inner = value.tensorfs_receipt_canonical_bytes
        if documents.spell(hashlib.sha256(inner).digest()) != value.tensorfs_receipt_digest:
            raise WeightsExchangeError(
                "weights_tensorfs_receipt_invalid", "TensorFS receipt digest mismatch"
            )
        try:
            manifest = canonical_json.decode_as(inner, weights_sink.NativeReceipt).manifest
        except (ValueError, msgspec.ValidationError) as exc:
            raise WeightsExchangeError(
                "weights_tensorfs_receipt_invalid", f"TensorFS receipt is invalid: {exc}"
            ) from exc
        manifest_id, manifest_length = manifest.digest, manifest.length
        manifest_value = store.manifest(manifest_id)
        manifest_bytes = bytes(manifest_value["manifest"])
        if (
            len(manifest_bytes) != manifest_length
            or hashlib.sha256(manifest_bytes).hexdigest() != manifest_id[7:]
        ):
            raise WeightsExchangeError(
                "weights_manifest_mismatch", "TensorFS manifest bytes do not match its ObjectRef"
            )
        rows: dict[str, pb.WeightsObjectSource] = {
            manifest_id: pb.WeightsObjectSource(
                object_id=manifest_id,
                length=manifest_length,
                source_ref=f"tensorfs-manifest:{manifest_id}",
            )
        }
        for raw in store.walk(manifest_id):
            object_id, length = str(raw["id"]), int(raw["length"])
            previous = rows.get(object_id)
            if previous is not None and previous.length != length:
                raise WeightsExchangeError(
                    "weights_object_conflict", "one object id carries two lengths"
                )
            rows[object_id] = pb.WeightsObjectSource(
                object_id=object_id,
                length=length,
                source_ref=f"tensorfs-object:{object_id}",
            )
        if not rows or len(rows) > weights_limits.MAX_WEIGHTS_OBJECTS:
            raise WeightsExchangeError(
                "weights_inventory_bound", "weights object count is outside protocol bounds"
            )
        encoded = sum(row.ByteSize() for row in rows.values())
        if encoded > weights_limits.MAX_WEIGHTS_INVENTORY_BYTES:
            raise WeightsExchangeError(
                "weights_inventory_bound", "weights object inventory is over its byte cap"
            )
        held = _Held(
            transaction_id=transaction_id,
            writer_epoch=writer_epoch,
            receipt=reference,
            objects=rows,
            manifest_id=manifest_id,
            manifest_bytes=manifest_bytes,
            manifest=pb.Ref(
                digest=bytes.fromhex(manifest_id[7:]),
                length=manifest_length,
            ),
        )
        with self._lock:
            abandoned = transaction_id in self._abandoned
            raced = None if abandoned else self._held.setdefault(transaction_id, held)
        if abandoned:
            raise WeightsExchangeError("weights_transaction_abandoned", "output was abandoned")
        if raced is not held:
            assert raced is not None
            return raced
        return held

    def _drop(self, transaction_id: str) -> None:
        with self._lock:
            self._held.pop(transaction_id, None)

    def _store(self) -> fill.Store:
        if self.store is None:
            if self.store_root is None:
                raise WeightsExchangeError(
                    "tensorfs_store_unavailable", "this worker has no verified TensorFS Store"
                )
            self.store = fill.open_store(self.store_root)
        return self.store

    def _writer_epoch(self, transaction_id: str) -> int:
        with self._lock:
            binding = self._bindings.get(transaction_id)
        if binding is not None:
            return binding[0]
        raise WeightsExchangeError(
            "weights_transaction_unknown", "receipt names no acknowledged intent"
        )

    def _declaration_digest(self, transaction_id: str) -> bytes:
        with self._lock:
            binding = self._bindings.get(transaction_id)
        if binding is None:
            raise WeightsExchangeError(
                "weights_declaration_absent", "transaction has no acknowledged declaration"
            )
        return binding[1]
