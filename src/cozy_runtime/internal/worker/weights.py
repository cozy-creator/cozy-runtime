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

TWO LANES TO THE HOST, ONE PREFERRED (proto-025, minor 21). The host's own seam is
`RuntimeWeights`, a loopback service this Runtime serves: the supervisor opens `Exchange`
per control session, intents and receipts go out on it as server events, acks come back as
client messages, and `Upload` is unary. The pre-21 form -- the same messages as
direction-annotated `WorkerControl` frames -- stays wired until the retirement minor. Which
lane an exchange takes is an OBSERVED FACT, not a flag: a live `Exchange` stream selects
itself; with none open the frame goes on `WorkerControl`. Every pending exchange remembers
the lane and the exact envelope it last went out under, and is re-issued only when that
lane is gone (the control stream superseded at the snapshot barrier, or a newer `Exchange`
opened). An ack settles the exchange it answers exactly once, by echoing that envelope: an
ack to a send this Runtime has since re-issued cannot bind a writer epoch the host has
already fenced past.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import urlparse

from cozy_runtime.internal import egress, fill
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


#: Where a pending exchange last went out: `("control", control_stream_epoch)` or
#: `("exchange", stream id)`. The lane is live while that stream is; a send on a lane that
#: is gone is what a replay re-issues.
Lane = tuple[str, int]
#: The three-field envelope a send was stamped with; the ack echoes it exactly.
Envelope = tuple[int, int, str]

_EVENT_FIELD = {"weights_intent": "intent", "weights_receipt": "receipt"}


def _envelope(message: Any) -> Envelope:
    return message.record_owner_epoch, message.control_stream_epoch, message.worker_boot_id


@dataclass(slots=True)
class _Pending:
    frame: pb.WorkerFrame
    event: threading.Event
    ack: pb.WeightsHostAck | None = None
    lane: Lane = ("", 0)
    envelope: Envelope = (0, 0, "")


class _ExchangeStream:
    """ONE open `RuntimeWeights.Exchange` call: the events this Runtime pushes to the
    supervisor, and the fact of whether the call is still open."""

    __slots__ = ("ended", "id", "out", "superseded")

    def __init__(self, stream_id: int) -> None:
        self.id = stream_id
        self.out: queue.Queue[pb.WeightsHostEvent | None] = queue.Queue()
        self.ended = threading.Event()
        self.superseded = False

    def live(self) -> bool:
        return not self.ended.is_set()

    def end(self, *, superseded: bool = False) -> None:
        if superseded:
            self.superseded = True
        if not self.ended.is_set():
            self.ended.set()
            self.out.put(None)


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
        stamp: Callable[[Any], Any] | None = None,
        note: Callable[[str, str], None] | None = None,
        allow_private_egress: bool = False,
    ) -> None:
        #: The `WorkerControl` lane: stamps the frame's envelope and queues it on the live
        #: control stream. Used only while no `Exchange` is open.
        self.send = send
        #: The worker's envelope stamp, for events that leave on `Exchange` instead. The
        #: identity default is for a test that exercises only the upload path, which never
        #: stamps; the worker always passes its own fence stamp.
        self.stamp = stamp or (lambda message: message)
        self.stop = stop
        self.owner_scope = owner_scope
        self.note = note or (lambda _kind, _step: None)
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
        self._abandoned: set[str] = set()
        #: The live `RuntimeWeights.Exchange` call, if a supervisor has one open. Its
        #: presence IS the lane choice; nothing else configures it.
        self._stream: _ExchangeStream | None = None
        self._streams_opened = 0

    # ------------------------------------------------------------- the host's lanes

    def exchange_open(self) -> bool:
        """The observed fact that selects the lane: a supervisor holds `Exchange` open."""
        with self._lock:
            return self._stream is not None and self._stream.live()

    def open_exchange_id(self) -> int:
        """The generation of the live `Exchange`, or 0 if none is open. Observability only:
        each accepted supervisor stream is one higher than the last."""
        with self._lock:
            if self._stream is not None and self._stream.live():
                return self._stream.id
            return 0

    def serve_exchange(self, acks: Iterator[pb.WeightsHostAck]) -> Iterator[pb.WeightsHostEvent]:
        """The server side of `RuntimeWeights.Exchange`, for one supervisor session.

        A newer `Exchange` supersedes this one exactly as a newer control stream supersedes
        its predecessor; the superseded call ends and its late acks are refused, so one
        exchange can never bind two writer epochs. A pending exchange stranded when this call
        ends is re-issued at the next snapshot barrier (`replay_pending`) under the fence that
        is current then, never restamped here where the fence may be mid-transition.
        """
        stream = self._open(acks)
        try:
            while True:
                event = stream.out.get()
                if event is None:
                    break
                yield event
        finally:
            self._close(stream)
        if stream.superseded:
            raise WeightsExchangeError(
                "weights_exchange_superseded", "a newer RuntimeWeights.Exchange replaced this one"
            )

    def _open(self, acks: Iterator[pb.WeightsHostAck]) -> _ExchangeStream:
        with self._lock:
            self._streams_opened += 1
            stream = _ExchangeStream(self._streams_opened)
            previous, self._stream = self._stream, stream
            if previous is not None:
                previous.end(superseded=True)
            if self.stop.is_set():
                stream.end()
                return stream
        # The stream's own thread reads acks; the server thread yields events. Both end with
        # the call, and the call ends when either side says so.
        self.note("artifact", f"RuntimeWeights.Exchange {stream.id} open")
        threading.Thread(
            target=self._read_acks,
            args=(stream, acks),
            daemon=True,
            name=f"weights-exchange-{stream.id}",
        ).start()
        return stream

    def _close(self, stream: _ExchangeStream) -> None:
        stream.end()
        with self._lock:
            if self._stream is stream:
                self._stream = None
        self.note("artifact", f"RuntimeWeights.Exchange {stream.id} ended")

    def _read_acks(self, stream: _ExchangeStream, acks: Iterator[pb.WeightsHostAck]) -> None:
        """The stream's client half: every ack the supervisor sends, until it half-closes,
        cancels, or this Runtime supersedes the call. Ending the read ends the call."""
        try:
            for ack in acks:
                if not stream.live():
                    return
                try:
                    self.absorb_ack(ack, lane=("exchange", stream.id))
                except WeightsExchangeError as exc:
                    self.note("artifact", f"dropped host ACK on Exchange: {exc.code}: {exc.detail}")
        except Exception as exc:
            # The supervisor cancelled or its connection went: the call is over either way,
            # and what it carried stays pending for the next lane.
            self.note("artifact", f"Exchange {stream.id} ack reader ended: {type(exc).__name__}")
        finally:
            stream.end()

    def _dispatch(self, pending: _Pending) -> None:
        """Send (or re-send) one exchange on the lane that is open right now. `_lock` held."""
        stream = self._stream
        if stream is not None and stream.live():
            self._send_on_exchange(pending, stream)
        else:
            self._send_on_control(pending)

    def _send_on_exchange(self, pending: _Pending, stream: _ExchangeStream) -> None:
        # A fresh copy per send: `pending.frame` stays the unstamped template so a re-issue
        # under a later fence starts from the same bytes.
        frame = pb.WorkerFrame()
        frame.CopyFrom(pending.frame)
        kind = frame.WhichOneof("msg")
        assert kind is not None
        inner = self.stamp(getattr(frame, kind))
        event = pb.WeightsHostEvent()
        getattr(event, _EVENT_FIELD[kind]).CopyFrom(inner)
        pending.lane = ("exchange", stream.id)
        pending.envelope = _envelope(inner)
        stream.out.put(event)

    def _send_on_control(self, pending: _Pending) -> None:
        # A fresh copy per send: the control sender serializes what it was handed, and a
        # later re-issue must not restamp bytes a superseded sender may still be reading.
        frame = pb.WorkerFrame()
        frame.CopyFrom(pending.frame)
        self.send(frame)  # worker.send stamps the live fence under the claim lock
        kind = frame.WhichOneof("msg")
        assert kind is not None
        inner = getattr(frame, kind)
        pending.lane = ("control", inner.control_stream_epoch)
        pending.envelope = _envelope(inner)

    def _lane_live(self, lane: Lane, control_stream_epoch: int) -> bool:
        if lane[0] == "control":
            return lane[1] == control_stream_epoch
        stream = self._stream
        return (
            lane[0] == "exchange" and stream is not None and stream.live() and stream.id == lane[1]
        )

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
        transaction_id: str,
        declaration_digest: bytes,
        declaration: bytes,
        requested_writer_epoch: int,
    ) -> dict[str, Any]:
        if self.store_root is None:
            raise WeightsExchangeError(
                "tensorfs_store_unavailable", "this worker has no verified TensorFS Store"
            )
        if (
            len(declaration_digest) != 32
            or not transaction_id.startswith("sha256:")
            or len(transaction_id) != 71
            or any(character not in "0123456789abcdef" for character in transaction_id[7:])
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
                weights_transaction_id=transaction_id,
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

    def absorb_ack(self, ack: pb.WeightsHostAck, *, lane: Lane) -> None:
        """One ack settles the one exchange it answers, ONCE.

        `lane` is where the ack arrived. It must be where the exchange last went out, and
        the ack must echo the envelope of that send: after a re-issue the host may have
        fenced a new writer epoch for the same transaction, and an ack to the earlier send
        would bind this Runtime to an epoch the host has already moved past.
        """
        key = self._key(ack.request_id, ack.attempt_ordinal, ack.output_slot, ack.stage)
        with self._lock:
            pending = self._pending.get(key)
            if pending is None or pending.ack is not None:
                raise WeightsExchangeError(
                    "weights_host_ack_unexpected", "no exact host exchange awaits this ACK"
                )
            if pending.lane != lane or pending.envelope != _envelope(ack):
                raise WeightsExchangeError(
                    "weights_host_ack_superseded",
                    "the ACK answers a send this exchange has since re-issued",
                )
            pending.ack = ack
            pending.event.set()

    def replay_pending(self, control_stream_epoch: int) -> None:
        """The snapshot barrier: re-issue what the SUPERSEDED lanes still owe.

        A pending exchange whose lane is live keeps waiting there -- an open `Exchange`
        survives an owner reconnect, and its ack, when it comes, still names the send it
        answers. Only a send on a control stream that is not this one, or on an `Exchange`
        that has ended, is re-issued, on whichever lane is open now.
        """
        with self._lock:
            stranded = [
                pending
                for pending in self._pending.values()
                if pending.ack is None and not self._lane_live(pending.lane, control_stream_epoch)
            ]
            for pending in stranded:
                self._dispatch(pending)

    def close(self) -> None:
        """Release every live TensorFS read lease; durable facts remain with the host."""
        with self._lock:
            held, self._held = list(self._held.values()), {}
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.end()
        for transaction in held:
            transaction.lease.release()

    def abandon(self, transaction_id: str) -> None:
        """Stop late receipt/transfer registration and release this output's read lease."""
        with self._lock:
            self._abandoned.add(transaction_id)
            held = self._held.pop(transaction_id, None)
        if held is not None:
            held.lease.release()

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
            # Send on whichever lane is open right now, stamped with the current fence, and
            # remember both so the one ack that answers this send is the one that settles it.
            self._dispatch(pending)
        try:
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
        elif ack.weights_transaction_id != sent.weights_transaction_id:
            raise WeightsExchangeError(
                "weights_host_ack_mismatch", "host ACK changed Runtime's supplied transaction id"
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
            abandoned = transaction_id in self._abandoned
            raced = None if abandoned else self._held.setdefault(transaction_id, held)
        if abandoned:
            held.lease.release()
            raise WeightsExchangeError("weights_transaction_abandoned", "output was abandoned")
        if raced is not held:
            held.lease.release()
            assert raced is not None
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
