"""The Runtime's `RuntimeWeights` seam and its two lanes, over real loopback gRPC.

proto-025 gives the host's weights lane its own service: the supervisor opens
`RuntimeWeights.Exchange` per control session, the Runtime pushes intents and receipts as
server events, and the host ACKs come back on the same stream. Until the supervisor speaks
it, the same intents ride `WorkerControl` frames. These tests drive the REAL
`WeightsExchange` behind the REAL `_RuntimeWeightsServicer` on a REAL loopback gRPC server,
with a stand-in supervisor that opens the bidi stream — nothing is mocked.

What they hold the Runtime to:

* the lane is chosen by an OBSERVED FACT — a live `Exchange` — not a flag: with the stream
  open an intent leaves on it and NOT on `WorkerControl`; with none open it falls back to
  the frame;
* an intent is NOT settled until the host ACK arrives (the receipt-before-ack invariant on
  the Runtime's side), and it TOLERATES the ACK arriving after a snapshot barrier;
* it settles ONCE, by the exact send the ACK answers: after a reconnect re-issues the
  intent under a new envelope, an ACK echoing the stale one is refused and only the ACK on
  the live lane settles it — the writer epoch a superseded send would have bound never binds.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from concurrent import futures
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import grpc
import pytest
import tensorfs

from cozy_runtime.internal.worker import weights
from cozy_runtime.internal.worker.control import _PreparationServicer, _RuntimeWeightsServicer
from cozy_runtime.protocol import MIN_COMPATIBLE_WIRE_MINOR, WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as pb_grpc

DECLARATION = b"tensorfs-declaration-canonical-bytes"
DECLARATION_DIGEST = sha256(DECLARATION).digest()
SPEC_DIGEST = sha256(b"invocation-spec").digest()
OUTPUT_SLOT = "primary"
TRANSACTION = "sha256:" + "a" * 64
WAIT = 5.0  # a failsafe only: every assertion is on an observed event, never on elapsed time


@dataclass
class Fence:
    """The three envelope fields a send is stamped with. Mutable so a test can bump the
    control-stream epoch exactly as an accepted reconnect does."""

    record_owner_epoch: int = 7
    control_stream_epoch: int = 1
    worker_boot_id: str = "boot-under-test"

    def stamp(self, message: Any) -> Any:
        message.record_owner_epoch = self.record_owner_epoch
        message.control_stream_epoch = self.control_stream_epoch
        message.worker_boot_id = self.worker_boot_id
        return message


@dataclass
class ControlLane:
    """The `WorkerControl` sender the exchange falls back to: it stamps the inner frame with
    the live fence, exactly as `Worker.send` does, and records what left."""

    fence: Fence
    frames: list[pb.WorkerFrame] = field(default_factory=list)
    signal: threading.Event = field(default_factory=threading.Event)

    def send(self, frame: pb.WorkerFrame) -> None:
        kind = frame.WhichOneof("msg")
        assert kind is not None
        self.fence.stamp(getattr(frame, kind))
        self.frames.append(frame)
        self.signal.set()


def _await(predicate: Any, why: str) -> None:
    """Spin on an OBSERVED fact until it holds. `WAIT` is only the failsafe that turns a
    hang into a named failure; the loop returns the instant the fact is true."""
    for _ in range(int(WAIT * 200)):
        if predicate():
            return
        threading.Event().wait(0.005)
    raise AssertionError(why)


def _attempt() -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-1", attempt=3, digest=SPEC_DIGEST, canceling="", weights_receipts={}
    )


def _recorded_ack(
    intent: pb.WeightsIntentFrame, *, writer_epoch: int, envelope: Fence
) -> pb.WeightsHostAck:
    """What a host returns for a freshly recorded intent, echoing the send's envelope."""
    return pb.WeightsHostAck(
        record_owner_epoch=envelope.record_owner_epoch,
        control_stream_epoch=envelope.control_stream_epoch,
        worker_boot_id=envelope.worker_boot_id,
        request_id=intent.request_id,
        attempt_ordinal=intent.attempt_ordinal,
        invocation_spec_digest=intent.invocation_spec_digest,
        output_slot=intent.output_slot,
        weights_transaction_id=TRANSACTION,
        writer_epoch=writer_epoch,
        tensorfs_declaration_digest=intent.tensorfs_declaration_digest,
        stage=pb.WEIGHTS_HOST_STAGE_INTENT,
        outcome=pb.WEIGHTS_HOST_OUTCOME_RECORDED,
    )


class StandInSupervisor:
    """One `RuntimeWeights.Exchange` call the supervisor opens, driven by hand. It reads the
    Runtime's events on the server-stream half and sends host ACKs on the client-stream half,
    when the test says to — that is how an ACK is 'delayed past a barrier': it is simply not
    sent until then."""

    def __init__(self, stub: Any) -> None:
        self._acks: list[pb.WeightsHostAck] = []
        self._ack_ready = threading.Condition()
        self._closed = False
        self.events: list[pb.WeightsHostEvent] = []
        self._event_ready = threading.Condition()
        self.call = stub.Exchange(self._ack_stream())
        self._reader = threading.Thread(target=self._read_events, daemon=True)
        self._reader.start()

    def _ack_stream(self) -> Iterator[pb.WeightsHostAck]:
        while True:
            with self._ack_ready:
                while not self._acks and not self._closed:
                    self._ack_ready.wait()
                if self._acks:
                    yield self._acks.pop(0)
                    continue
                return

    def _read_events(self) -> None:
        try:
            for event in self.call:
                with self._event_ready:
                    self.events.append(event)
                    self._event_ready.notify_all()
        except grpc.RpcError:
            return

    def next_intent(self) -> pb.WeightsIntentFrame:
        with self._event_ready:
            while not self.events:
                assert self._event_ready.wait(WAIT), "no Exchange event arrived"
            event = self.events.pop(0)
        assert event.WhichOneof("event") == "intent", event
        return event.intent

    def event_count(self) -> int:
        with self._event_ready:
            return len(self.events)

    def ack(self, ack: pb.WeightsHostAck) -> None:
        with self._ack_ready:
            self._acks.append(ack)
            self._ack_ready.notify_all()

    def close(self) -> None:
        with self._ack_ready:
            self._closed = True
            self._ack_ready.notify_all()
        self.call.cancel()


@dataclass
class Rig:
    exchange: weights.WeightsExchange
    fence: Fence
    control: ControlLane
    stub: Any
    server: Any
    preparation: Any


@pytest.fixture()
def rig(tmp_path: Path) -> Iterator[Rig]:
    fence = Fence()
    control = ControlLane(fence)
    exchange = weights.WeightsExchange(
        store_root=tmp_path / "store",
        send=control.send,
        stamp=fence.stamp,
        stop=threading.Event(),
        owner_scope=lambda: "owner-scope",
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_RuntimeWeightsServicer_to_server(_RuntimeWeightsServicer(exchange), server)
    pb_grpc.add_RuntimePreparationServicer_to_server(
        _PreparationServicer(
            None, None, None, None, None, weights_checkpoint_validator=exchange.validate_checkpoint
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = pb_grpc.RuntimeWeightsStub(channel)
    try:
        yield Rig(
            exchange=exchange,
            fence=fence,
            control=control,
            stub=stub,
            server=server,
            preparation=pb_grpc.RuntimePreparationStub(channel),
        )
    finally:
        exchange.close()
        channel.close()
        server.stop(grace=0)


def _run_intent(rig: Rig) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def call() -> None:
        try:
            result["value"] = rig.exchange.intent(
                _attempt(),
                transaction_id=TRANSACTION,
                output_slot=OUTPUT_SLOT,
                declaration_digest=DECLARATION_DIGEST,
                declaration=DECLARATION,
                requested_writer_epoch=0,
            )
        except weights.WeightsExchangeError as exc:
            result["error"] = exc

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    return {"thread": thread, "result": result}


def test_open_exchange_selects_the_lane_and_the_ack_settles_after_a_barrier(rig: Rig) -> None:
    """The observed fact — a live Exchange — takes the intent, not WorkerControl. The ack is
    withheld until AFTER a snapshot barrier; the Runtime keeps waiting and settles on it."""
    supervisor = StandInSupervisor(rig.stub)
    # The stream reaches the exchange's server handler: the observed fact is now true.
    _await(rig.exchange.exchange_open, "the Exchange did not register as open")

    pending = _run_intent(rig)
    intent = supervisor.next_intent()
    assert intent.record_owner_epoch == rig.fence.record_owner_epoch
    assert intent.control_stream_epoch == rig.fence.control_stream_epoch
    assert intent.tensorfs_declaration_canonical_bytes == DECLARATION
    assert intent.weights_transaction_id == TRANSACTION
    # The intent went out ON the Exchange and NOT as a WorkerControl frame.
    assert rig.control.frames == []

    # A snapshot barrier for the SAME session: replay must not duplicate an exchange whose
    # lane is still live, and the still-blocked intent must not have settled.
    rig.exchange.replay_pending(rig.fence.control_stream_epoch)
    assert supervisor.event_count() == 0, "the barrier re-issued a live-lane intent"
    assert not pending["result"], "the intent settled before any host ACK"

    supervisor.ack(_recorded_ack(intent, writer_epoch=4, envelope=rig.fence))
    pending["thread"].join(WAIT)
    assert not pending["thread"].is_alive(), "the intent never settled on its host ACK"
    assert pending["result"]["value"]["writer_epoch"] == 4
    assert pending["result"]["value"]["weights_transaction_id"] == TRANSACTION
    supervisor.close()


def test_reconnect_reissues_under_the_new_envelope_and_a_stale_ack_is_refused(rig: Rig) -> None:
    """The intent is sent on one session's Exchange, the owner reconnects (a new
    control-stream epoch and a fresh Exchange), and the barrier re-issues it. An ACK echoing
    the SUPERSEDED send is refused; only the ACK on the live lane settles the intent, once,
    with the writer epoch that send bound."""
    first = StandInSupervisor(rig.stub)
    _await(rig.exchange.exchange_open, "the first Exchange did not register as open")

    pending = _run_intent(rig)
    stale_intent = first.next_intent()
    assert stale_intent.control_stream_epoch == 1

    # Reconnect: a new accepted stream mints the next control-stream epoch, and the
    # supervisor opens a fresh Exchange, which supersedes the first.
    rig.fence.control_stream_epoch = 2
    second = StandInSupervisor(rig.stub)
    _await(lambda: rig.exchange.open_exchange_id() == 2, "the second Exchange did not supersede")

    # The snapshot barrier of the new session re-issues the still-unsettled intent on the
    # new lane, under the new envelope.
    rig.exchange.replay_pending(rig.fence.control_stream_epoch)
    fresh_intent = second.next_intent()
    assert fresh_intent.control_stream_epoch == 2

    # An ACK that answers the superseded send (envelope epoch 1) must not settle it: the
    # exchange has since re-issued and would otherwise bind a stale writer epoch.
    stale_ack = _recorded_ack(fresh_intent, writer_epoch=99, envelope=Fence(control_stream_epoch=1))
    second.ack(stale_ack)
    pending["thread"].join(0.5)
    assert pending["thread"].is_alive(), "a stale-envelope ACK settled the intent"
    assert not pending["result"]

    # The ACK on the live lane, echoing the live envelope, settles it — exactly once.
    second.ack(_recorded_ack(fresh_intent, writer_epoch=5, envelope=rig.fence))
    pending["thread"].join(WAIT)
    assert not pending["thread"].is_alive(), "the live-lane ACK did not settle the intent"
    assert pending["result"]["value"]["writer_epoch"] == 5
    first.close()
    second.close()


def test_without_an_exchange_the_intent_rides_worker_control(rig: Rig) -> None:
    """The pre-PR2 supervisor never opens Exchange: the observed fact is false, so the same
    intent leaves as a `WorkerControl` frame and its host ACK on that lane settles it."""
    assert not rig.exchange.exchange_open()

    pending = _run_intent(rig)
    assert rig.control.signal.wait(WAIT), "no WorkerControl frame carried the intent"
    (frame,) = rig.control.frames
    assert frame.WhichOneof("msg") == "weights_intent"
    intent = frame.weights_intent
    assert intent.tensorfs_declaration_canonical_bytes == DECLARATION

    # The host answers on WorkerControl, exactly as the current supervisor does.
    rig.exchange.absorb_ack(
        _recorded_ack(intent, writer_epoch=2, envelope=rig.fence),
        lane=("control", rig.fence.control_stream_epoch),
    )
    pending["thread"].join(WAIT)
    assert not pending["thread"].is_alive(), "the control-lane ACK did not settle the intent"
    assert pending["result"]["value"]["writer_epoch"] == 2


def test_host_cannot_replace_the_runtime_supplied_transaction_id(rig: Rig) -> None:
    supervisor = StandInSupervisor(rig.stub)
    _await(rig.exchange.exchange_open, "the Exchange did not register as open")
    pending = _run_intent(rig)
    intent = supervisor.next_intent()
    assert intent.weights_transaction_id == TRANSACTION
    answer = _recorded_ack(intent, writer_epoch=1, envelope=rig.fence)
    answer.weights_transaction_id = "sha256:" + "b" * 64
    supervisor.ack(answer)
    pending["thread"].join(WAIT)
    assert not pending["thread"].is_alive()
    assert "value" not in pending["result"]
    assert pending["result"]["error"].code == "weights_host_ack_mismatch"
    supervisor.close()


def test_protocol_probe_is_static_and_does_not_open_a_store_or_change_fence(rig: Rig) -> None:
    before = (
        rig.fence.record_owner_epoch,
        rig.fence.control_stream_epoch,
        rig.fence.worker_boot_id,
    )
    assert rig.exchange.store_root is not None
    assert not rig.exchange.store_root.exists()
    for _ in range(2):
        info = rig.preparation.ProtocolInfo(pb.ProtocolInfoRequest(), timeout=WAIT)
        assert (info.minimum_wire_minor, info.wire_minor) == (MIN_COMPATIBLE_WIRE_MINOR, WIRE_MINOR)
    assert before == (
        rig.fence.record_owner_epoch,
        rig.fence.control_stream_epoch,
        rig.fence.worker_boot_id,
    )
    assert rig.exchange.store_root is not None
    assert not rig.exchange.store_root.exists()
    assert not rig.exchange._pending and not rig.exchange.writer_broker.bindings


def test_checkpoint_refusal_has_no_writer_authority() -> None:
    sent = pb.WeightsCheckpointFrame(
        request_id="request",
        attempt_ordinal=1,
        invocation_spec_digest=SPEC_DIGEST,
        output_slot=OUTPUT_SLOT,
        weights_transaction_id=TRANSACTION,
        writer_epoch=9,
        tensorfs_declaration_digest=DECLARATION_DIGEST,
    )
    refused = pb.WeightsHostAck(
        request_id=sent.request_id,
        attempt_ordinal=sent.attempt_ordinal,
        invocation_spec_digest=sent.invocation_spec_digest,
        output_slot=sent.output_slot,
        tensorfs_declaration_digest=sent.tensorfs_declaration_digest,
        stage=pb.WEIGHTS_HOST_STAGE_CHECKPOINT,
        outcome=pb.WEIGHTS_HOST_OUTCOME_REFUSED,
        refusal=pb.WEIGHTS_HOST_REFUSAL_STALE_WRITER,
    )
    weights.WeightsExchange._validate_ack(refused, sent, pb.WEIGHTS_HOST_STAGE_CHECKPOINT)
    refused.writer_epoch = sent.writer_epoch
    with pytest.raises(weights.WeightsExchangeError) as error:
        weights.WeightsExchange._validate_ack(refused, sent, pb.WEIGHTS_HOST_STAGE_CHECKPOINT)
    assert error.value.code == "weights_host_ack_invalid"


def test_restored_native_work_validates_under_pending_intent_without_importing_writer(
    rig: Rig, tmp_path: Path
) -> None:
    original = tensorfs.Store.init(tmp_path / "original")
    restored = tensorfs.Store.init(rig.exchange.store_root)
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    targets = {
        "model": {
            "drop": [],
            "add": {
                "weight": {
                    "logical_dtype": "f32",
                    "shape": [512],
                    "encoding": plain,
                    "parts": {"value": {"dtype": "f32", "shape": [512]}},
                }
            },
        }
    }
    order = [("model", "weight")]
    fingerprint = "sha256:" + "65" * 32
    declaration = original.derived_declaration(
        {}, targets, {}, order, 2048, work_fingerprint=fingerprint
    )
    writer = original.begin_derived(
        TRANSACTION, 7, {}, targets, {}, order, 2048, work_fingerprint=fingerprint
    )
    writer.add_part("model", "weight", "value", io.BytesIO(b"\x31" * 2048))
    head = writer.checkpoint("req-1", OUTPUT_SLOT)
    writer.fence()
    head_id, head_length = head["head"], head["head_length"]
    plan_digest, head_index, head_bytes = head["plan_digest"], head["index"], head["bytes"]
    assert isinstance(head_id, str) and isinstance(head_length, int)
    assert (
        isinstance(plan_digest, str) and isinstance(head_index, int) and isinstance(head_bytes, int)
    )
    tensorfs.transfer_object(original, restored, head_id, head_length)
    page = original.checkpoint_page(
        head_id,
        head_length,
        operation_id="req-1",
        slot=OUTPUT_SLOT,
        plan_digest=plan_digest,
    )
    assert page["previous"] is None
    for obj in page["objects"]:
        tensorfs.transfer_object(original, restored, obj["object_id"], obj["length"])
    assert restored.derived_lookup(TRANSACTION)["state"] == "absent"

    supervisor = StandInSupervisor(rig.stub)
    _await(rig.exchange.exchange_open, "exchange did not open")
    result: dict[str, Any] = {}

    def intent_call() -> None:
        result["value"] = rig.exchange.intent(
            _attempt(),
            output_slot=OUTPUT_SLOT,
            transaction_id=TRANSACTION,
            declaration_digest=sha256(declaration).digest(),
            declaration=declaration,
            requested_writer_epoch=0,
        )

    pending = threading.Thread(target=intent_call, daemon=True)
    pending.start()
    sent = supervisor.next_intent()
    checkpoint = pb.CheckpointRef(
        head=pb.Ref(digest=bytes.fromhex(head_id[7:]), length=head_length),
        plan_digest=bytes.fromhex(plan_digest[7:]),
        index=head_index,
        bytes=head_bytes,
    )
    ready = pb.WeightsIntentReadyRequest(
        record_owner_epoch=rig.fence.record_owner_epoch,
        worker_boot_id=rig.fence.worker_boot_id,
        attempt_ordinal=sent.attempt_ordinal,
        weights=pb.WeightsCheckpointSubject(
            request_id=sent.request_id,
            invocation_spec_digest=sent.invocation_spec_digest,
            output_slot=sent.output_slot,
            weights_transaction_id=sent.weights_transaction_id,
            writer_epoch=8,
            tensorfs_declaration_digest=sent.tensorfs_declaration_digest,
        ),
        checkpoint=checkpoint,
    )
    request = pb.ValidateWeightsCheckpointRequest(
        intent=ready, tensorfs_declaration_canonical_bytes=declaration
    )
    accepted = rig.preparation.ValidateWeightsCheckpoint(request, timeout=WAIT)
    assert accepted.valid and accepted.checkpoint == checkpoint
    assert pending.is_alive() and result == {}
    assert restored.derived_lookup(TRANSACTION)["state"] == "absent"
    assert not rig.exchange.writer_broker.bindings
    assert supervisor.event_count() == 0  # no periodic intent resend while Ready is delayed
    for mutation in ("counter", "declaration", "scope"):
        wrong = pb.ValidateWeightsCheckpointRequest()
        wrong.CopyFrom(request)
        if mutation == "counter":
            wrong.intent.checkpoint.bytes += 1
        elif mutation == "declaration":
            wrong.tensorfs_declaration_canonical_bytes += b" "
        else:
            wrong.intent.weights.output_slot = "other"
        refused = rig.preparation.ValidateWeightsCheckpoint(wrong, timeout=WAIT)
        assert not refused.valid and refused.safe_code
    ack = _recorded_ack(sent, writer_epoch=8, envelope=rig.fence)
    ack.checkpoint.CopyFrom(checkpoint)
    supervisor.ack(ack)
    pending.join(WAIT)
    assert not pending.is_alive()
    assert result["value"]["checkpoint"] == (head["head"], head["head_length"])
    assert restored.derived_lookup(TRANSACTION)["state"] == "absent"
    supervisor.close()
