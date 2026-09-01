"""Stateless Runtime side of the WeightsSink host exchange.

The supervisor is the durable authority. Runtime keeps only waiters and live TensorFS read
leases and sends intent before the executor opens a writer; the worker and executor share only
fixed files in the already-brokered attempt spool for declaration/receipt bytes too large for
their control seam.

Runtime holds the produced bytes, so Runtime is what streams them. When the supervisor has
bound a transfer to its durable receipt it hands over the presigned PUT it was given, and
`upload` spends it through the one egress boundary (`internal/egress.py`) exactly as
`write_output` already spends a `DeliveryGrant` output URL. The grant names ONE object, ONE
method and ONE short expiry and is pinned to the object's own digest by its signed checksum
header, so holding it authorizes nothing else. The alternative -- posting the object back to
the supervisor 4 MiB at a time so IT could send the same request -- was 2,048 sequential round
trips for an 8 GiB object and bought no authority the grant did not already carry.

`read` stays: it is the authenticated owner read-back lane, a bounded range of one object the
owner explicitly asked for, which is a different thing from moving a whole object.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import urlparse

from cozy_runtime.internal import egress, fill, pull
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb


class _LeaseReader:
    """A read-only file-like view of ONE object in a live TensorFS lease.

    The uploader streams through this, so an 8 GiB object is never a bytes object: at most one
    chunk is resident. Reads are served in whole chunks regardless of the size asked for, which
    an HTTP body reader accepts and which keeps an 8 GiB PUT to thousands of reads, not millions.
    """

    __slots__ = ("lease", "length", "object_id", "offset")

    def __init__(self, lease: Any, object_id: str, length: int) -> None:
        self.lease = lease
        self.object_id = object_id
        self.length = length
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
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


class WeightsExchangeError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(slots=True)
class _Pending:
    frame: pb.WorkerFrame
    event: threading.Event
    ack: pb.WeightsHostAck | None = None


@dataclass(slots=True)
class _Held:
    transaction_id: str
    writer_epoch: int
    receipt: pb.WeightsReceiptRef
    objects: dict[str, pb.WeightsObjectSource]
    manifest_id: str
    manifest_bytes: bytes
    manifest: pb.Ref
    lease: Any


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
    """One Runtime process's memory-only bridge to its durable host."""

    def __init__(
        self,
        *,
        store_root: Path | None,
        send: Callable[[pb.WorkerFrame], None],
        stop: threading.Event,
        owner_scope: Callable[[], str],
        allow_private_egress: bool = False,
    ) -> None:
        self.send = send
        self.stop = stop
        self.owner_scope = owner_scope
        self.store_root = store_root
        # Local proof only; production leaves this false, exactly as the supervisor's own
        # `allowPrivateWeightsUploads` does. It widens nothing but the address classes an
        # upload may reach, and only for a caller that constructed this object saying so.
        self.allow_private_egress = allow_private_egress
        self.store: Any = None
        self._lock = threading.Lock()
        self._pending: dict[tuple[str, int, str, int], _Pending] = {}
        self._bindings: dict[str, tuple[int, bytes]] = {}
        self._held: dict[str, _Held] = {}

    @staticmethod
    def _key(
        request_id: str, attempt: int, output_slot: str, stage: int
    ) -> tuple[str, int, str, int]:
        return request_id, attempt, output_slot, stage

    def intent(
        self,
        attempt: Any,
        *,
        output_slot: str,
        declaration_digest: bytes,
        declaration: bytes,
        requested_writer_epoch: int,
    ) -> dict[str, Any]:
        if self.store_root is None:
            raise WeightsExchangeError(
                "artifact_store_unavailable", "this worker has no verified TensorFS store"
            )
        if (
            len(declaration_digest) != 32
            or not declaration
            or len(declaration) > weights_limits.MAX_WEIGHTS_DECLARATION_BYTES
            or hashlib.sha256(declaration).digest() != declaration_digest
        ):
            raise WeightsExchangeError(
                "weights_declaration_invalid", "declaration bytes or digest are invalid"
            )
        frame = pb.WorkerFrame(
            weights_intent=pb.WeightsIntentFrame(
                request_id=attempt.request_id,
                attempt_ordinal=attempt.attempt,
                invocation_spec_digest=attempt.digest,
                output_slot=output_slot,
                tensorfs_declaration_digest=declaration_digest,
                requested_writer_epoch=requested_writer_epoch,
                tensorfs_declaration_canonical_bytes=declaration,
            )
        )
        ack = self._exchange(
            self._key(
                attempt.request_id,
                attempt.attempt,
                output_slot,
                pb.WEIGHTS_HOST_STAGE_INTENT,
            ),
            frame,
            canceled=lambda: bool(attempt.canceling),
        )
        self._validate_ack(ack, frame.weights_intent, pb.WEIGHTS_HOST_STAGE_INTENT)
        if ack.outcome == pb.WEIGHTS_HOST_OUTCOME_REFUSED:
            raise WeightsExchangeError(
                pb.WeightsHostRefusal.Name(ack.refusal).lower(), "the weights host refused intent"
            )
        with self._lock:
            known = self._bindings.setdefault(
                ack.weights_transaction_id,
                (ack.writer_epoch, declaration_digest),
            )
        if known != (ack.writer_epoch, declaration_digest):
            raise WeightsExchangeError(
                "weights_host_ack_mismatch", "transaction binding changed within one Runtime"
            )
        receipt_digest = b""
        if ack.HasField("weights_receipt"):
            receipt_digest = self._receipt_ref(ack.weights_receipt)
            receipt_value = documents.read(
                ack.weights_receipt.weights_receipt_canonical_bytes,
                pb.WeightsReceipt,
            )
            if (
                receipt_value.get("request_id") != attempt.request_id
                or receipt_value.get("owner_authority_scope") != self.owner_scope()
                or receipt_value.get("invocation_spec_digest") != documents.spell(attempt.digest)
                or receipt_value.get("output_slot") != output_slot
                or receipt_value.get("weights_transaction_id") != ack.weights_transaction_id
            ):
                raise WeightsExchangeError(
                    "weights_receipt_identity_mismatch",
                    "replayed receipt does not bind this attempt/output",
                )
            held = self._hold(
                ack.weights_transaction_id,
                ack.writer_epoch,
                ack.weights_receipt,
            )
            if ack.manifest != held.manifest:
                raise WeightsExchangeError(
                    "weights_manifest_mismatch", "host replayed another produced Manifest"
                )
            attempt.weights_receipts[output_slot] = held.receipt
        return {
            "ok": True,
            "weights_transaction_id": ack.weights_transaction_id,
            "writer_epoch": ack.writer_epoch,
            "weights_receipt_digest": documents.spell(receipt_digest) if receipt_digest else "",
        }

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
        ack = self._exchange(
            self._key(
                attempt.request_id,
                attempt.attempt,
                output_slot,
                pb.WEIGHTS_HOST_STAGE_RECEIPT,
            ),
            frame,
            canceled=lambda: bool(attempt.canceling),
        )
        self._validate_ack(ack, frame.weights_receipt, pb.WEIGHTS_HOST_STAGE_RECEIPT)
        if ack.outcome == pb.WEIGHTS_HOST_OUTCOME_REFUSED:
            self._drop(transaction_id)
            raise WeightsExchangeError(
                pb.WeightsHostRefusal.Name(ack.refusal).lower(),
                "the weights host refused receipt",
            )
        if (
            not ack.HasField("weights_receipt")
            or self._receipt_ref(ack.weights_receipt) != receipt_digest
        ):
            self._drop(transaction_id)
            raise WeightsExchangeError(
                "weights_receipt_mismatch", "receipt ACK did not echo exact receipt bytes"
            )
        if ack.manifest != held.manifest:
            self._drop(transaction_id)
            raise WeightsExchangeError(
                "weights_manifest_mismatch", "receipt ACK did not echo produced Manifest"
            )
        attempt.weights_receipts[output_slot] = reference
        return {
            "ok": True,
            "weights_receipt_digest": documents.spell(receipt_digest),
        }

    def absorb_ack(self, ack: pb.WeightsHostAck) -> None:
        key = self._key(ack.request_id, ack.attempt_ordinal, ack.output_slot, ack.stage)
        with self._lock:
            pending = self._pending.get(key)
            if pending is None or pending.ack is not None:
                raise WeightsExchangeError(
                    "weights_host_ack_unexpected", "no exact host exchange awaits this ACK"
                )
            pending.ack = ack
            pending.event.set()

    def replay_pending(self) -> None:
        with self._lock:
            frames = [pending.frame for pending in self._pending.values() if pending.ack is None]
        for frame in frames:
            self.send(frame)

    def close(self) -> None:
        """Release every live TensorFS read lease; durable facts remain with the host."""
        with self._lock:
            held, self._held = list(self._held.values()), {}
        for transaction in held:
            transaction.lease.release()

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
                    held.lease.read_into(
                        row.object_id,
                        row.length,
                        request.offset,
                        request.max_bytes,
                        target,
                    )
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
        """Stream ONE held object to the exact presigned URL the supervisor granted."""
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
        return _LeaseReader(held.lease, row.object_id, row.length), row.length

    def _exchange(
        self,
        key: tuple[str, int, str, int],
        frame: pb.WorkerFrame,
        *,
        canceled: Callable[[], bool],
    ) -> pb.WeightsHostAck:
        pending = _Pending(frame=frame, event=threading.Event())
        with self._lock:
            existing = self._pending.get(key)
            if existing is not None:
                raise WeightsExchangeError(
                    "weights_host_exchange_conflict", "another exchange holds this output stage"
                )
            self._pending[key] = pending
        try:
            self.send(frame)
            while not pending.event.wait(0.1):
                if canceled() or self.stop.is_set():
                    raise WeightsExchangeError(
                        "weights_host_exchange_canceled", "attempt ended before host ACK"
                    )
            assert pending.ack is not None
            return pending.ack
        finally:
            with self._lock:
                self._pending.pop(key, None)

    @staticmethod
    def _validate_ack(ack: pb.WeightsHostAck, sent: Any, stage: int) -> None:
        if (
            ack.request_id != sent.request_id
            or ack.attempt_ordinal != sent.attempt_ordinal
            or ack.invocation_spec_digest != sent.invocation_spec_digest
            or ack.output_slot != sent.output_slot
            or ack.stage != stage
            or ack.tensorfs_declaration_digest != sent.tensorfs_declaration_digest
            or ack.outcome
            not in (
                pb.WEIGHTS_HOST_OUTCOME_RECORDED,
                pb.WEIGHTS_HOST_OUTCOME_REPLAYED,
                pb.WEIGHTS_HOST_OUTCOME_REFUSED,
            )
        ):
            raise WeightsExchangeError(
                "weights_host_ack_mismatch", "host ACK does not echo the exact exchange"
            )
        if ack.outcome == pb.WEIGHTS_HOST_OUTCOME_REFUSED:
            if (
                not ack.refusal
                or ack.weights_transaction_id
                or ack.writer_epoch
                or ack.HasField("weights_receipt")
                or ack.HasField("manifest")
            ):
                raise WeightsExchangeError(
                    "weights_host_ack_invalid", "refusal carries transaction authority"
                )
        elif ack.refusal or not ack.weights_transaction_id or ack.writer_epoch <= 0:
            raise WeightsExchangeError(
                "weights_host_ack_invalid", "recorded ACK lacks transaction authority"
            )
        if ack.HasField("weights_receipt") != ack.HasField("manifest"):
            raise WeightsExchangeError(
                "weights_host_ack_invalid", "receipt and produced Manifest appear together"
            )
        if stage == pb.WEIGHTS_HOST_STAGE_RECEIPT:
            if (
                ack.weights_transaction_id != sent.weights_transaction_id
                or ack.writer_epoch != sent.writer_epoch
                or not ack.HasField("weights_receipt")
                or ack.manifest != sent.manifest
            ):
                raise WeightsExchangeError(
                    "weights_host_ack_mismatch", "receipt ACK changed transaction identity"
                )
        elif ack.HasField("weights_receipt") and ack.outcome != pb.WEIGHTS_HOST_OUTCOME_REPLAYED:
            raise WeightsExchangeError(
                "weights_host_ack_invalid", "a newly recorded intent cannot carry a receipt"
            )

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
        store = self._store()
        with self._lock:
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
            checkpoint_evidence = base64.b64decode(
                str(value["checkpoint_evidence_canonical_bytes"]), validate=True
            )
            if facts["checkpoint_evidence_canonical_bytes"].encode("utf-8") != checkpoint_evidence:
                raise ValueError("CheckpointEvidence handoff mismatch")
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
        blobs: list[tuple[str, int]] = []
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
            blobs.append((object_id, length))
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
            lease=store.acquire(manifest_id, sorted(set(blobs))),
        )
        with self._lock:
            raced = self._held.setdefault(transaction_id, held)
        if raced is not held:
            held.lease.release()
            return raced
        return held

    def _drop(self, transaction_id: str) -> None:
        with self._lock:
            held = self._held.pop(transaction_id, None)
        if held is not None:
            held.lease.release()

    def _store(self) -> Any:
        if self.store is None:
            if self.store_root is None:
                raise WeightsExchangeError(
                    "artifact_store_unavailable", "this worker has no verified TensorFS store"
                )
            self.store = pull.open_store(fill.tensorfs_module(), self.store_root)
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
