"""Privileged native weights and ordered workspace observations.

Workspace is the shared durable authority on local and private workers. The
executor exchanges bounded writer operations with its parent; intent, checkpoint
and receipt observations leave on WorkerControl before the final outcome.
RuntimeWeights retains its bounded Read/Upload operations, but its old Host-ACK
Exchange no longer grants writer authority.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import urlparse

from cozy_runtime.internal import egress, fill
from cozy_runtime.internal.model_reader import ModelReaderBroker
from cozy_runtime.internal.weights_writer import WriterBroker
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb


class _LeaseReader:
    """A read-only file-like view of ONE object in a live TensorFS lease.

    The uploader streams through this, so an 8 GiB object is never a bytes object: at most one
    chunk is resident. Reads are served in whole chunks regardless of the size asked for, which
    an HTTP body reader accepts and which keeps an 8 GiB PUT to thousands of reads, not millions.
    """

    __slots__ = ("closed", "lease", "length", "object_id", "offset")

    def __init__(self, lease: Any, object_id: str, length: int) -> None:
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
        send: Callable[[pb.WorkerFrame], None],
        stop: threading.Event,
        owner_scope: Callable[[], str],
        stamp: Callable[[Any], Any] | None = None,
        note: Callable[[str, str], None] | None = None,
        allow_private_egress: bool = False,
        workspace: Workspace | None = None,
    ) -> None:
        #: Native observations and outcomes share the ordered WorkerControl stream.
        self.send = send
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
        self.store: Any = None
        self._lock = threading.Lock()
        self._bindings: dict[str, tuple[int, bytes]] = {}
        self._held: dict[str, _Held] = {}
        self._abandoned: set[str] = set()
        self.writer_broker = WriterBroker(self._store, checkpoint=self.record_checkpoint)
        self.reader_broker = ModelReaderBroker(self._store)

    def record_checkpoint(
        self, attempt: Any, transaction: str, binding: Any, facts: dict[str, Any]
    ) -> None:
        """Publish the parent native writer's own observation, never child-authored refs."""
        checkpoint = pb.CheckpointRef(
            head=pb.Ref(digest=documents.raw(facts["head"]), length=facts["head_length"]),
            plan_digest=documents.raw(facts["plan_digest"]),
            index=facts["index"],
            bytes=facts["bytes"],
        )
        frame = pb.WorkerFrame(
            weights_checkpoint=pb.WeightsCheckpointFrame(
                request_id=attempt.request_id,
                attempt_ordinal=attempt.attempt,
                invocation_spec_digest=attempt.digest,
                output_slot=binding.slot,
                weights_transaction_id=transaction,
                writer_epoch=binding.epoch,
                tensorfs_declaration_digest=binding.declaration_digest,
                checkpoint=checkpoint,
            )
        )
        if self.workspace is None:
            raise WeightsExchangeError("workspace_unavailable", "worker has no workspace custody")
        try:
            self.workspace.checkpoint(self.owner_scope(), frame.weights_checkpoint)
        except Exception as exc:
            raise WeightsExchangeError(
                str(getattr(exc, "code", "workspace_refused")),
                "workspace checkpoint recording refused",
            ) from exc
        self.send(frame)
        self.send(
            pb.WorkerFrame(
                weights_transaction=self.workspace.status(
                    self.workspace.weights_row(self.owner_scope(), transaction)
                )
            )
        )

    def _checkpoint_pending(
        self,
        request: pb.WeightsIntentReadyRequest | pb.CheckpointTransferRequest,
        subject: pb.WeightsCheckpointSubject,
        *,
        attempt_ordinal: int | None = None,
        declaration: bytes | None = None,
    ) -> dict[str, Any]:
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
            row = self._checkpoint_pending(
                intent,
                subject,
                attempt_ordinal=intent.attempt_ordinal,
                declaration=request.tensorfs_declaration_canonical_bytes,
            )
            if intent.HasField("checkpoint"):
                assert self.workspace is not None
                self.workspace._validate_checkpoint(row, intent.checkpoint)
            self._checkpoint_pending(
                intent,
                subject,
                attempt_ordinal=intent.attempt_ordinal,
                declaration=request.tensorfs_declaration_canonical_bytes,
            )
            result.valid = True
        except Exception as exc:
            result.safe_code = str(getattr(exc, "code", "weights_checkpoint_invalid"))[:128]
            result.safe_detail = f"checkpoint validation refused ({type(exc).__name__})"
        return result

    def intent(
        self,
        attempt: Any,
        *,
        output_slot: str,
        transaction_id: str,
        declaration_digest: bytes,
        declaration: bytes,
        requested_writer_epoch: int,
    ) -> dict[str, Any]:
        if self.workspace is None:
            raise WeightsExchangeError("workspace_unavailable", "worker has no workspace custody")
        intent = pb.WeightsIntentFrame(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=attempt.digest,
            output_slot=output_slot,
            weights_transaction_id=transaction_id,
            tensorfs_declaration_digest=declaration_digest,
            tensorfs_declaration_canonical_bytes=declaration,
        )
        try:
            row = self.workspace.begin_weights(self.owner_scope(), intent)
            epoch = int(row["epoch"])
            intent.requested_writer_epoch = epoch
            # One FIFO carries the projection and its eventual outcome. The
            # native writer is authorized by Workspace, never by a Host ACK.
            self.send(pb.WorkerFrame(weights_intent=intent))
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
                if not ready.HasField("checkpoint") or getattr(exc, "code", None) not in {
                    "ROOT_ABSENT",
                    "OBJECT_ABSENT",
                }:
                    raise
                self.send(pb.WorkerFrame(weights_transaction=self.workspace.status(row)))
                # An owner may restore an acknowledged remote checkpoint. Its
                # Ready RPC commits to the same journal; no network wait owns a lock.
                while not row["ready"]:
                    if attempt.canceling or self.stop.wait(0.05):
                        raise WeightsExchangeError(
                            "weights_canceled", "writer restoration stopped"
                        ) from exc
                    row = self.workspace.weights_row(self.owner_scope(), transaction_id)
            self.send(pb.WorkerFrame(weights_transaction=self.workspace.status(row)))
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
            return {
                "ok": True,
                "weights_transaction_id": transaction_id,
                "writer_epoch": epoch,
                "weights_receipt_digest": documents.spell(row["receipt_digest"])
                if row["receipt"]
                else "",
                "checkpoint": checkpoint,
            }
        except WeightsExchangeError:
            raise
        except Exception as exc:
            raise WeightsExchangeError(
                str(getattr(exc, "code", "workspace_refused")),
                "workspace writer authorization refused",
            ) from exc

    def receipt(
        self,
        attempt: Any,
        *,
        output_slot: str,
        transaction_id: str,
        receipt_digest: bytes,
        canonical_receipt: bytes,
    ) -> dict[str, Any]:
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
        receipt = documents.read(canonical_receipt, pb.WeightsReceipt)
        if (
            receipt.get("request_id") != attempt.request_id
            or receipt.get("owner_authority_scope") != self.owner_scope()
            or receipt.get("invocation_spec_digest") != documents.spell(attempt.digest)
            or receipt.get("output_slot") != output_slot
            or receipt.get("weights_transaction_id") != transaction_id
        ):
            raise WeightsExchangeError(
                "weights_receipt_identity_mismatch", "receipt does not bind this attempt/output"
            )
        held = self._hold(transaction_id, self._writer_epoch(transaction_id), reference)
        frame = pb.WorkerFrame(
            weights_receipt=pb.WeightsReceiptFrame(
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
        )
        if self.workspace is None:
            raise WeightsExchangeError("workspace_unavailable", "worker has no workspace custody")
        try:
            self.workspace.record_receipt(self.owner_scope(), frame.weights_receipt)
        except Exception as exc:
            self._drop(transaction_id)
            raise WeightsExchangeError(
                str(getattr(exc, "code", "workspace_refused")),
                "workspace receipt recording refused",
            ) from exc
        self.send(frame)
        self.send(
            pb.WorkerFrame(
                weights_transaction=self.workspace.status(
                    self.workspace.weights_row(self.owner_scope(), transaction_id)
                )
            )
        )
        attempt.weights_receipts[output_slot] = reference
        return {
            "ok": True,
            "weights_receipt_digest": documents.spell(receipt_digest),
        }

    def close(self) -> None:
        """Forget receipt metadata; active transports release their own read leases."""
        self.writer_broker.close()
        with self._lock:
            self._held = {}

    def abandon(self, transaction_id: str) -> None:
        """Stop late receipt/transfer registration; native disposition owns the roots."""
        self.writer_broker.close(transaction_id)
        with self._lock:
            self._abandoned.add(transaction_id)
            self._held.pop(transaction_id, None)

    def read(self, request: pb.WeightsReadRequest) -> pb.WorkerFrame:
        refusal = pb.WEIGHTS_READ_REFUSAL_UNSPECIFIED
        detail = ""
        data = b""
        with self._lock:
            held = self._held.get(request.weights_transaction_id)
        row = held.objects.get(request.object_id) if held is not None else None
        if held is None:
            refusal, detail = (
                pb.WEIGHTS_READ_REFUSAL_UNKNOWN_TRANSACTION,
                "weights transaction is not held by this Runtime",
            )
        elif request.writer_epoch != held.writer_epoch:
            refusal, detail = (
                pb.WEIGHTS_READ_REFUSAL_STALE_WRITER,
                "writer epoch is stale",
            )
        elif row is None:
            refusal, detail = (
                pb.WEIGHTS_READ_REFUSAL_UNKNOWN_OBJECT,
                "object is outside the exact receipt inventory",
            )
        elif request.source_ref != row.source_ref:
            refusal, detail = (
                pb.WEIGHTS_READ_REFUSAL_SOURCE_REF_MISMATCH,
                "source reference does not bind this object",
            )
        elif (
            request.max_bytes <= 0
            or request.max_bytes > weights_limits.MAX_WEIGHTS_READ_BYTES
            or request.offset >= row.length
            or request.offset + request.max_bytes > row.length
        ):
            refusal, detail = (
                pb.WEIGHTS_READ_REFUSAL_RANGE_INVALID,
                "requested range is outside the exact object length",
            )
        else:
            try:
                if row.source_ref.startswith("tensorfs-manifest:"):
                    data = held.manifest_bytes[request.offset : request.offset + request.max_bytes]
                else:
                    target = bytearray(request.max_bytes)
                    lease = self._store().acquire(held.manifest_id, [(row.object_id, row.length)])
                    try:
                        lease.read_into(
                            row.object_id,
                            row.length,
                            request.offset,
                            request.max_bytes,
                            target,
                        )
                    finally:
                        lease.release()
                    data = bytes(target)
            except Exception as exc:
                refusal, detail = (
                    pb.WEIGHTS_READ_REFUSAL_SOURCE_UNAVAILABLE,
                    f"TensorFS source read refused: {type(exc).__name__}",
                )
        result = pb.WeightsReadResult(
            weights_transaction_id=request.weights_transaction_id,
            writer_epoch=request.writer_epoch,
            object_id=request.object_id,
            source_ref=request.source_ref,
            read_id=request.read_id,
            offset=request.offset,
        )
        if refusal:
            result.outcome = pb.WEIGHTS_READ_OUTCOME_REFUSED
            result.refusal = refusal
            result.safe_detail = detail[:4096]
        else:
            result.outcome = pb.WEIGHTS_READ_OUTCOME_DATA
            result.data = data
            result.data_digest = hashlib.sha256(data).digest()
        return pb.WorkerFrame(weights_read_result=result)

    def upload(self, request: pb.WeightsUploadRequest) -> pb.WorkerFrame:
        """Stream ONE held object to the exact presigned URL the supervisor granted.

        Returns the answer as a `WorkerFrame`; `RuntimeWeights.Upload` unwraps its inner
        `WeightsUploadResult`, and the pre-21 `WorkerControl` lane sends the frame as-is
        (the session restamps the envelope). The bytes never enter the host process.
        """
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
            return pb.WorkerFrame(weights_upload_result=result)
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
            return pb.WorkerFrame(weights_upload_result=result)
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
        return pb.WorkerFrame(weights_upload_result=result)

    def _upload_source(self, request: pb.WeightsUploadRequest) -> tuple[Any, int]:
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
        documents.read(raw, pb.WeightsReceipt)
        return digest

    def _hold(
        self,
        transaction_id: str,
        writer_epoch: int,
        reference: pb.WeightsReceiptRef,
    ) -> _Held:
        self._receipt_ref(reference)
        store = self._store()
        lifecycle = store.derived_lookup(transaction_id)
        if lifecycle.get("state") != "committed":
            raise WeightsExchangeError(
                "weights_transaction_uncommitted", "receipt has no committed TensorFS transaction"
            )
        if lifecycle["disposition"]["kind"] == "released":
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
        value = documents.read(reference.weights_receipt_canonical_bytes, pb.WeightsReceipt)
        if value.get("weights_transaction_id") != transaction_id:
            raise WeightsExchangeError(
                "weights_receipt_identity_mismatch", "receipt names another transaction"
            )
        try:
            inner = base64.b64decode(str(value["tensorfs_receipt_canonical_bytes"]), validate=True)
            if hashlib.sha256(inner).hexdigest() != str(value["tensorfs_receipt_digest"])[7:]:
                raise ValueError("nested digest mismatch")
            facts = json.loads(inner)
            manifest = facts["manifest"]
            manifest_id = "sha256:" + str(manifest["sha256"])
            manifest_length = int(manifest["length"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, binascii.Error) as exc:
            raise WeightsExchangeError(
                "weights_tensorfs_receipt_invalid", f"TensorFS receipt is invalid: {exc}"
            ) from exc
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

    def _store(self) -> Any:
        if self.store is None:
            if self.store_root is None:
                raise WeightsExchangeError(
                    "tensorfs_store_unavailable", "this worker has no verified TensorFS Store"
                )
            self.store = fill.open_store(fill.tensorfs_module(), self.store_root)
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
