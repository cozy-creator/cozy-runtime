"""Stateless Runtime side of the ArtifactSink host exchange.

The supervisor is the durable authority. Runtime keeps only waiters and live TensorFS read
leases, sends intent before the executor opens a writer, and exposes produced bytes through
bounded ObjectRef/source-ref reads. No URL, credential, destination, or filesystem path enters
Worker Protocol; the worker and executor share only fixed files in the already-brokered attempt
spool for declaration/receipt bytes too large for their control seam.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.internal import fill, pull
from cozy_runtime.protocol import artifact_limits, documents
from cozy_runtime.protocol import worker_pb2 as pb


class ArtifactExchangeError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(slots=True)
class _Pending:
    frame: pb.WorkerFrame
    event: threading.Event
    ack: pb.ArtifactHostAck | None = None


@dataclass(slots=True)
class _Held:
    transaction_id: str
    writer_generation: int
    receipt: pb.ArtifactReceiptRef
    objects: dict[str, pb.ArtifactObjectSource]
    manifest_id: str
    manifest_bytes: bytes
    manifest: pb.Ref
    lease: Any


class ArtifactExchange:
    """One Runtime process's memory-only bridge to its durable host."""

    def __init__(
        self,
        *,
        store_root: Path | None,
        send: Callable[[pb.WorkerFrame], None],
        stop: threading.Event,
        owner_scope: Callable[[], str],
    ) -> None:
        self.send = send
        self.stop = stop
        self.owner_scope = owner_scope
        self.store_root = store_root
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
        requested_writer_generation: int,
    ) -> dict[str, Any]:
        if self.store_root is None:
            raise ArtifactExchangeError(
                "artifact_store_unavailable", "this worker has no verified TensorFS store"
            )
        if (
            len(declaration_digest) != 32
            or not declaration
            or len(declaration) > artifact_limits.MAX_ARTIFACT_DECLARATION_BYTES
            or hashlib.sha256(declaration).digest() != declaration_digest
        ):
            raise ArtifactExchangeError(
                "artifact_declaration_invalid", "declaration bytes or digest are invalid"
            )
        frame = pb.WorkerFrame(
            artifact_intent=pb.ArtifactIntentFrame(
                request_id=attempt.request_id,
                attempt_ordinal=attempt.attempt,
                invocation_spec_digest=attempt.digest,
                output_slot=output_slot,
                tensorfs_declaration_digest=declaration_digest,
                requested_writer_generation=requested_writer_generation,
                tensorfs_declaration_canonical_bytes=declaration,
            )
        )
        ack = self._exchange(
            self._key(
                attempt.request_id,
                attempt.attempt,
                output_slot,
                pb.ARTIFACT_HOST_STAGE_INTENT,
            ),
            frame,
            canceled=lambda: bool(attempt.canceling),
        )
        self._validate_ack(ack, frame.artifact_intent, pb.ARTIFACT_HOST_STAGE_INTENT)
        if ack.outcome == pb.ARTIFACT_HOST_OUTCOME_REFUSED:
            raise ArtifactExchangeError(
                pb.ArtifactHostRefusal.Name(ack.refusal).lower(), "the artifact host refused intent"
            )
        with self._lock:
            known = self._bindings.setdefault(
                ack.artifact_transaction_id,
                (ack.writer_generation, declaration_digest),
            )
        if known != (ack.writer_generation, declaration_digest):
            raise ArtifactExchangeError(
                "artifact_host_ack_mismatch", "transaction binding changed within one Runtime"
            )
        receipt_digest = b""
        if ack.HasField("artifact_receipt"):
            receipt_digest = self._receipt_ref(ack.artifact_receipt)
            receipt_value = documents.read(
                ack.artifact_receipt.artifact_receipt_canonical_bytes,
                pb.ArtifactReceipt,
            )
            if (
                receipt_value.get("request_id") != attempt.request_id
                or receipt_value.get("owner_authority_scope") != self.owner_scope()
                or receipt_value.get("invocation_spec_digest") != documents.spell(attempt.digest)
                or receipt_value.get("output_slot") != output_slot
                or receipt_value.get("artifact_transaction_id") != ack.artifact_transaction_id
            ):
                raise ArtifactExchangeError(
                    "artifact_receipt_identity_mismatch",
                    "replayed receipt does not bind this attempt/output",
                )
            held = self._hold(
                ack.artifact_transaction_id,
                ack.writer_generation,
                ack.artifact_receipt,
            )
            if ack.manifest != held.manifest:
                raise ArtifactExchangeError(
                    "artifact_manifest_mismatch", "host replayed another produced Manifest"
                )
            attempt.artifact_receipts[output_slot] = held.receipt
        return {
            "ok": True,
            "artifact_transaction_id": ack.artifact_transaction_id,
            "writer_generation": ack.writer_generation,
            "artifact_receipt_digest": documents.spell(receipt_digest) if receipt_digest else "",
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
            or len(canonical_receipt) > artifact_limits.MAX_ARTIFACT_RECEIPT_BYTES
            or hashlib.sha256(canonical_receipt).digest() != receipt_digest
        ):
            raise ArtifactExchangeError(
                "artifact_receipt_invalid", "receipt bytes or digest are invalid"
            )
        reference = pb.ArtifactReceiptRef(
            artifact_receipt_digest=receipt_digest,
            artifact_receipt_canonical_bytes=canonical_receipt,
        )
        receipt = documents.read(canonical_receipt, pb.ArtifactReceipt)
        if (
            receipt.get("request_id") != attempt.request_id
            or receipt.get("owner_authority_scope") != self.owner_scope()
            or receipt.get("invocation_spec_digest") != documents.spell(attempt.digest)
            or receipt.get("output_slot") != output_slot
            or receipt.get("artifact_transaction_id") != transaction_id
        ):
            raise ArtifactExchangeError(
                "artifact_receipt_identity_mismatch", "receipt does not bind this attempt/output"
            )
        held = self._hold(transaction_id, self._writer_generation(transaction_id), reference)
        frame = pb.WorkerFrame(
            artifact_receipt=pb.ArtifactReceiptFrame(
                request_id=attempt.request_id,
                attempt_ordinal=attempt.attempt,
                invocation_spec_digest=attempt.digest,
                output_slot=output_slot,
                artifact_transaction_id=transaction_id,
                writer_generation=held.writer_generation,
                tensorfs_declaration_digest=self._declaration_digest(transaction_id),
                artifact_receipt=reference,
                objects=[held.objects[key] for key in sorted(held.objects)],
                manifest=held.manifest,
            )
        )
        ack = self._exchange(
            self._key(
                attempt.request_id,
                attempt.attempt,
                output_slot,
                pb.ARTIFACT_HOST_STAGE_RECEIPT,
            ),
            frame,
            canceled=lambda: bool(attempt.canceling),
        )
        self._validate_ack(ack, frame.artifact_receipt, pb.ARTIFACT_HOST_STAGE_RECEIPT)
        if ack.outcome == pb.ARTIFACT_HOST_OUTCOME_REFUSED:
            self._drop(transaction_id)
            raise ArtifactExchangeError(
                pb.ArtifactHostRefusal.Name(ack.refusal).lower(),
                "the artifact host refused receipt",
            )
        if (
            not ack.HasField("artifact_receipt")
            or self._receipt_ref(ack.artifact_receipt) != receipt_digest
        ):
            self._drop(transaction_id)
            raise ArtifactExchangeError(
                "artifact_receipt_mismatch", "receipt ACK did not echo exact receipt bytes"
            )
        if ack.manifest != held.manifest:
            self._drop(transaction_id)
            raise ArtifactExchangeError(
                "artifact_manifest_mismatch", "receipt ACK did not echo produced Manifest"
            )
        attempt.artifact_receipts[output_slot] = reference
        return {
            "ok": True,
            "artifact_receipt_digest": documents.spell(receipt_digest),
        }

    def absorb_ack(self, ack: pb.ArtifactHostAck) -> None:
        key = self._key(ack.request_id, ack.attempt_ordinal, ack.output_slot, ack.stage)
        with self._lock:
            pending = self._pending.get(key)
            if pending is None or pending.ack is not None:
                raise ArtifactExchangeError(
                    "artifact_host_ack_unexpected", "no exact host exchange awaits this ACK"
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

    def read(self, request: pb.ArtifactReadRequest) -> pb.WorkerFrame:
        refusal = pb.ARTIFACT_READ_REFUSAL_UNSPECIFIED
        detail = ""
        data = b""
        with self._lock:
            held = self._held.get(request.artifact_transaction_id)
        row = held.objects.get(request.object_id) if held is not None else None
        if held is None:
            refusal, detail = (
                pb.ARTIFACT_READ_REFUSAL_UNKNOWN_TRANSACTION,
                "artifact transaction is not held by this Runtime",
            )
        elif request.writer_generation != held.writer_generation:
            refusal, detail = (
                pb.ARTIFACT_READ_REFUSAL_STALE_WRITER,
                "writer generation is stale",
            )
        elif row is None:
            refusal, detail = (
                pb.ARTIFACT_READ_REFUSAL_UNKNOWN_OBJECT,
                "object is outside the exact receipt inventory",
            )
        elif request.source_ref != row.source_ref:
            refusal, detail = (
                pb.ARTIFACT_READ_REFUSAL_SOURCE_REF_MISMATCH,
                "source reference does not bind this object",
            )
        elif (
            request.max_bytes <= 0
            or request.max_bytes > artifact_limits.MAX_ARTIFACT_READ_BYTES
            or request.offset >= row.length
            or request.offset + request.max_bytes > row.length
        ):
            refusal, detail = (
                pb.ARTIFACT_READ_REFUSAL_RANGE_INVALID,
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
                    pb.ARTIFACT_READ_REFUSAL_SOURCE_UNAVAILABLE,
                    f"TensorFS source read refused: {type(exc).__name__}",
                )
        result = pb.ArtifactReadResult(
            artifact_transaction_id=request.artifact_transaction_id,
            writer_generation=request.writer_generation,
            object_id=request.object_id,
            source_ref=request.source_ref,
            read_id=request.read_id,
            offset=request.offset,
        )
        if refusal:
            result.outcome = pb.ARTIFACT_READ_OUTCOME_REFUSED
            result.refusal = refusal
            result.safe_detail = detail[:4096]
        else:
            result.outcome = pb.ARTIFACT_READ_OUTCOME_DATA
            result.data = data
            result.data_digest = hashlib.sha256(data).digest()
        return pb.WorkerFrame(artifact_read_result=result)

    def _exchange(
        self,
        key: tuple[str, int, str, int],
        frame: pb.WorkerFrame,
        *,
        canceled: Callable[[], bool],
    ) -> pb.ArtifactHostAck:
        pending = _Pending(frame=frame, event=threading.Event())
        with self._lock:
            existing = self._pending.get(key)
            if existing is not None:
                raise ArtifactExchangeError(
                    "artifact_host_exchange_conflict", "another exchange holds this output stage"
                )
            self._pending[key] = pending
        try:
            self.send(frame)
            while not pending.event.wait(0.1):
                if canceled() or self.stop.is_set():
                    raise ArtifactExchangeError(
                        "artifact_host_exchange_canceled", "attempt ended before host ACK"
                    )
            assert pending.ack is not None
            return pending.ack
        finally:
            with self._lock:
                self._pending.pop(key, None)

    @staticmethod
    def _validate_ack(ack: pb.ArtifactHostAck, sent: Any, stage: int) -> None:
        if (
            ack.request_id != sent.request_id
            or ack.attempt_ordinal != sent.attempt_ordinal
            or ack.invocation_spec_digest != sent.invocation_spec_digest
            or ack.output_slot != sent.output_slot
            or ack.stage != stage
            or ack.tensorfs_declaration_digest != sent.tensorfs_declaration_digest
            or ack.outcome
            not in (
                pb.ARTIFACT_HOST_OUTCOME_RECORDED,
                pb.ARTIFACT_HOST_OUTCOME_REPLAYED,
                pb.ARTIFACT_HOST_OUTCOME_REFUSED,
            )
        ):
            raise ArtifactExchangeError(
                "artifact_host_ack_mismatch", "host ACK does not echo the exact exchange"
            )
        if ack.outcome == pb.ARTIFACT_HOST_OUTCOME_REFUSED:
            if (
                not ack.refusal
                or ack.artifact_transaction_id
                or ack.writer_generation
                or ack.HasField("artifact_receipt")
                or ack.HasField("manifest")
            ):
                raise ArtifactExchangeError(
                    "artifact_host_ack_invalid", "refusal carries transaction authority"
                )
        elif ack.refusal or not ack.artifact_transaction_id or ack.writer_generation <= 0:
            raise ArtifactExchangeError(
                "artifact_host_ack_invalid", "recorded ACK lacks transaction authority"
            )
        if ack.HasField("artifact_receipt") != ack.HasField("manifest"):
            raise ArtifactExchangeError(
                "artifact_host_ack_invalid", "receipt and produced Manifest appear together"
            )
        if stage == pb.ARTIFACT_HOST_STAGE_RECEIPT:
            if (
                ack.artifact_transaction_id != sent.artifact_transaction_id
                or ack.writer_generation != sent.writer_generation
                or not ack.HasField("artifact_receipt")
                or ack.manifest != sent.manifest
            ):
                raise ArtifactExchangeError(
                    "artifact_host_ack_mismatch", "receipt ACK changed transaction identity"
                )
        elif ack.HasField("artifact_receipt") and ack.outcome != pb.ARTIFACT_HOST_OUTCOME_REPLAYED:
            raise ArtifactExchangeError(
                "artifact_host_ack_invalid", "a newly recorded intent cannot carry a receipt"
            )

    @staticmethod
    def _receipt_ref(reference: pb.ArtifactReceiptRef) -> bytes:
        raw = reference.artifact_receipt_canonical_bytes
        digest = hashlib.sha256(raw).digest()
        if (
            not raw
            or len(raw) > artifact_limits.MAX_ARTIFACT_RECEIPT_BYTES
            or digest != reference.artifact_receipt_digest
        ):
            raise ArtifactExchangeError(
                "artifact_receipt_invalid", "receipt ref digest does not match exact bytes"
            )
        documents.read(raw, pb.ArtifactReceipt)
        return digest

    def _hold(
        self,
        transaction_id: str,
        writer_generation: int,
        reference: pb.ArtifactReceiptRef,
    ) -> _Held:
        store = self._store()
        with self._lock:
            existing = self._held.get(transaction_id)
        if existing is not None:
            if existing.writer_generation != writer_generation or existing.receipt != reference:
                raise ArtifactExchangeError(
                    "artifact_receipt_mismatch", "held transaction carries another receipt"
                )
            return existing
        value = documents.read(reference.artifact_receipt_canonical_bytes, pb.ArtifactReceipt)
        if value.get("artifact_transaction_id") != transaction_id:
            raise ArtifactExchangeError(
                "artifact_receipt_identity_mismatch", "receipt names another transaction"
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
            raise ArtifactExchangeError(
                "artifact_tensorfs_receipt_invalid", f"TensorFS receipt is invalid: {exc}"
            ) from exc
        manifest_value = store.manifest(manifest_id)
        manifest_bytes = bytes(manifest_value["manifest"])
        if (
            len(manifest_bytes) != manifest_length
            or hashlib.sha256(manifest_bytes).hexdigest() != manifest_id[7:]
        ):
            raise ArtifactExchangeError(
                "artifact_manifest_mismatch", "TensorFS manifest bytes do not match its ObjectRef"
            )
        rows: dict[str, pb.ArtifactObjectSource] = {
            manifest_id: pb.ArtifactObjectSource(
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
                raise ArtifactExchangeError(
                    "artifact_object_conflict", "one object id carries two lengths"
                )
            rows[object_id] = pb.ArtifactObjectSource(
                object_id=object_id,
                length=length,
                source_ref=f"tensorfs-object:{object_id}",
            )
            blobs.append((object_id, length))
        if not rows or len(rows) > artifact_limits.MAX_ARTIFACT_OBJECTS:
            raise ArtifactExchangeError(
                "artifact_inventory_bound", "artifact object count is outside protocol bounds"
            )
        encoded = sum(row.ByteSize() for row in rows.values())
        if encoded > artifact_limits.MAX_ARTIFACT_INVENTORY_BYTES:
            raise ArtifactExchangeError(
                "artifact_inventory_bound", "artifact object inventory is over its byte cap"
            )
        held = _Held(
            transaction_id=transaction_id,
            writer_generation=writer_generation,
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
                raise ArtifactExchangeError(
                    "artifact_store_unavailable", "this worker has no verified TensorFS store"
                )
            self.store = pull.open_store(fill.tensorfs_module(), self.store_root)
        return self.store

    def _writer_generation(self, transaction_id: str) -> int:
        with self._lock:
            binding = self._bindings.get(transaction_id)
        if binding is not None:
            return binding[0]
        raise ArtifactExchangeError(
            "artifact_transaction_unknown", "receipt names no acknowledged intent"
        )

    def _declaration_digest(self, transaction_id: str) -> bytes:
        with self._lock:
            binding = self._bindings.get(transaction_id)
        if binding is None:
            raise ArtifactExchangeError(
                "artifact_declaration_absent", "transaction has no acknowledged declaration"
            )
        return binding[1]
