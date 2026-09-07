"""Native custody is committed by Runtime before ordered control observations.

The loopback preparation RPC checks exact restored checkpoint identity. There is
no supervisor ACK authority and no fallback execution lane.
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
from typing import Any, cast

import grpc
import pytest
import tensorfs

from cozy_runtime.author import WeightsPart, WeightsTarget, WeightsTensor
from cozy_runtime.author._weights import WeightsCommit
from cozy_runtime.internal.weights_sink import (
    WeightsHostBinding,
    WeightsTransactionHost,
    protocol_receipt,
    weights_transaction_id,
)
from cozy_runtime.internal.worker import weights
from cozy_runtime.internal.worker.control import (
    GrpcControlHost,
    _PreparationServicer,
    _RuntimeWeightsServicer,
)
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import MIN_COMPATIBLE_WIRE_MINOR, WIRE_MINOR, documents
from cozy_runtime.protocol import worker_pb2 as pb
from cozy_runtime.protocol import worker_pb2_grpc as pb_grpc

OWNER = "owner-scope"
OUTPUT_SLOT = "model"
INVOCATION = (
    (Path(__file__).parent / "testdata/worker-protocol/canonical/invocation_spec_job.json")
    .read_bytes()
    .strip()
)
TRANSACTION = weights_transaction_id(
    OWNER, "req-1", documents.spell(sha256(INVOCATION).digest()), OUTPUT_SLOT
)
WAIT = 5.0


@dataclass
class Fence:
    record_owner_epoch: int = 7
    control_stream_epoch: int = 1
    worker_boot_id: str = "boot-under-test"

    def stamp(self, message: Any) -> Any:
        if not hasattr(message, "record_owner_epoch"):
            return message
        message.record_owner_epoch = self.record_owner_epoch
        message.control_stream_epoch = self.control_stream_epoch
        message.worker_boot_id = self.worker_boot_id
        return message


@dataclass
class ControlLane:
    fence: Fence
    workspace: Workspace
    frames: list[pb.WorkerFrame] = field(default_factory=list)

    def send(self, frame: pb.WorkerFrame) -> None:
        kind = frame.WhichOneof("msg")
        assert kind is not None
        self.fence.stamp(getattr(frame, kind))
        if kind == "weights_intent":
            observed = frame.weights_intent
            row = self.workspace.weights_row(OWNER, observed.weights_transaction_id)
            assert row["epoch"] == observed.requested_writer_epoch > 0
        if kind == "weights_receipt":
            observed_receipt = frame.weights_receipt
            row = self.workspace.weights_row(OWNER, observed_receipt.weights_transaction_id)
            assert (
                row["receipt"] == observed_receipt.weights_receipt.weights_receipt_canonical_bytes
            )
        self.frames.append(frame)


@dataclass
class Rig:
    exchange: weights.WeightsExchange
    workspace: Workspace
    store: Any
    fence: Fence
    control: ControlLane
    preparation: Any
    legacy: Any
    invocation: bytes
    declaration: bytes
    targets: Any

    def accept(self, ordinal: int = 1) -> Any:
        digest = sha256(self.invocation).digest()
        self.workspace.accept(
            OWNER,
            pb.AttemptOffer(
                request_id="req-1",
                attempt_ordinal=ordinal,
                invocation_spec_digest=digest,
                invocation_spec_canonical_bytes=self.invocation,
            ),
        )
        return SimpleNamespace(
            request_id="req-1", attempt=ordinal, digest=digest, canceling="", weights_receipts={}
        )

    def begin(self, attempt: Any = None) -> dict[str, Any]:
        attempt = self.accept() if attempt is None else attempt
        return self.workspace.begin_weights(
            OWNER,
            pb.WeightsIntentFrame(
                request_id=attempt.request_id,
                attempt_ordinal=attempt.attempt,
                invocation_spec_digest=attempt.digest,
                output_slot=OUTPUT_SLOT,
                weights_transaction_id=TRANSACTION,
                tensorfs_declaration_digest=sha256(self.declaration).digest(),
                tensorfs_declaration_canonical_bytes=self.declaration,
            ),
        )

    def intent(self, attempt: Any, *, transaction: str = TRANSACTION, requested: int = 0) -> Any:
        return self.exchange.intent(
            attempt,
            transaction_id=transaction,
            output_slot=OUTPUT_SLOT,
            declaration_digest=sha256(self.declaration).digest(),
            declaration=self.declaration,
            requested_writer_epoch=requested,
        )


@pytest.fixture()
def rig(tmp_path: Path) -> Iterator[Rig]:
    store = tensorfs.Store.init(tmp_path / "store")
    workspace = Workspace(Path(store.root))
    fence = Fence()
    control = ControlLane(fence, workspace)
    exchange = weights.WeightsExchange(
        store_root=Path(store.root),
        workspace=workspace,
        send=control.send,
        stamp=fence.stamp,
        stop=threading.Event(),
        owner_scope=lambda: OWNER,
    )
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
    declaration = store.derived_declaration(
        {}, targets, {}, [("model", "weight")], 2048, work_fingerprint="sha256:" + "65" * 32
    )
    invocation = (
        (Path(__file__).parent / "testdata/worker-protocol/canonical/invocation_spec_job.json")
        .read_bytes()
        .strip()
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_RuntimePreparationServicer_to_server(
        _PreparationServicer(
            None, None, None, None, None, weights_checkpoint_validator=exchange.validate_checkpoint
        ),
        server,
    )
    pb_grpc.add_RuntimeWeightsServicer_to_server(_RuntimeWeightsServicer(exchange), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield Rig(
            exchange,
            workspace,
            store,
            fence,
            control,
            pb_grpc.RuntimePreparationStub(channel),
            pb_grpc.RuntimeWeightsStub(channel),
            invocation,
            declaration,
            targets,
        )
    finally:
        exchange.close()
        channel.close()
        server.stop(grace=0).wait()


def terminal(rig: Rig, attempt: Any) -> None:
    raw, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=documents.spell(attempt.digest),
            status=pb.OUTCOME_STATUS_ABANDONED,
            execution_started=True,
        )
    )
    rig.workspace.outcome(
        OWNER,
        pb.AttemptOutcome(
            request_id=attempt.request_id,
            attempt_ordinal=attempt.attempt,
            invocation_spec_digest=attempt.digest,
            outcome_id="out-" + str(attempt.attempt),
            outcome_digest=digest,
            outcome_canonical_bytes=raw,
        ),
    )


def ready(rig: Rig, checkpoint: pb.CheckpointRef | None = None) -> pb.WeightsIntentReadyRequest:
    row = rig.workspace.weights_row(OWNER, TRANSACTION)
    value = pb.WeightsIntentReadyRequest(
        record_owner_epoch=rig.fence.record_owner_epoch,
        worker_boot_id=rig.fence.worker_boot_id,
        attempt_ordinal=row["ordinal"],
        weights=pb.WeightsCheckpointSubject(
            request_id="req-1",
            invocation_spec_digest=row["spec"],
            output_slot=OUTPUT_SLOT,
            weights_transaction_id=TRANSACTION,
            writer_epoch=row["epoch"],
            tensorfs_declaration_digest=row["declaration_digest"],
        ),
    )
    if checkpoint is not None:
        value.checkpoint.CopyFrom(checkpoint)
    return value


def test_intent_uses_durable_epoch_and_ordered_control_without_host_ack(rig: Rig) -> None:
    attempt = rig.accept()
    first = rig.intent(attempt)
    assert first["writer_epoch"] == 1
    assert [frame.WhichOneof("msg") for frame in rig.control.frames] == [
        "weights_intent",
        "weights_transaction",
    ]
    assert rig.control.frames[-1].weights_transaction.intent_ready
    # Stream reconnection cannot allocate or grant a new native writer epoch.
    rig.fence.control_stream_epoch += 1
    second = rig.intent(attempt)
    assert second == first
    assert rig.control.frames[-2].weights_intent.control_stream_epoch == 2
    assert Workspace(Path(rig.store.root)).weights_row(OWNER, TRANSACTION)["epoch"] == 1


def test_new_attempt_fences_prior_epoch_and_cannot_change_transaction(rig: Rig) -> None:
    first = rig.accept()
    rig.intent(first)
    with pytest.raises(weights.WeightsExchangeError):
        rig.intent(first, transaction="sha256:" + "b" * 64)
    terminal(rig, first)
    rig.exchange.writer_broker.close_attempt(first)
    resumed = rig.accept(2)
    assert rig.intent(resumed)["writer_epoch"] == 2
    with pytest.raises(weights.WeightsExchangeError):
        rig.intent(first)
    assert rig.workspace.weights_row(OWNER, TRANSACTION)["epoch"] == 2


def test_native_receipt_is_durable_before_control_observation(rig: Rig) -> None:
    attempt = rig.accept()
    plain = next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")
    commit = WeightsCommit(
        OUTPUT_SLOT,
        {},
        {
            "model": WeightsTarget(
                add={
                    "weight": WeightsTensor(
                        "f32", (512,), plain, {"value": WeightsPart("f32", (512,))}
                    )
                }
            )
        },
        {},
        (("model", "weight"),),
        2048,
    )

    def bind(slot: str, transaction: str, digest: bytes, declaration: bytes) -> WeightsHostBinding:
        result = rig.exchange.intent(
            attempt,
            output_slot=slot,
            transaction_id=transaction,
            declaration_digest=digest,
            declaration=declaration,
            requested_writer_epoch=0,
        )
        return WeightsHostBinding(result["weights_transaction_id"], result["writer_epoch"])

    def record(receipt: Any) -> None:
        _, raw, digest = protocol_receipt(
            receipt,
            owner_scope=OWNER,
            request_id=attempt.request_id,
            invocation_spec_digest=documents.spell(attempt.digest),
        )
        rig.exchange.receipt(
            attempt,
            output_slot=OUTPUT_SLOT,
            transaction_id=receipt.weights_transaction_id,
            receipt_digest=digest,
            canonical_receipt=raw,
        )

    producer = WeightsTransactionHost(
        store=rig.store,
        owner_scope=OWNER,
        request_id=attempt.request_id,
        invocation_spec_digest=documents.spell(attempt.digest),
        work_fingerprint="sha256:" + "65" * 32,
        writer_session_id=1,
        allowed_sources={},
        output_bounds={OUTPUT_SLOT: 2048},
        bind_intent=bind,
        record_receipt=record,
    )
    writer = producer.open(commit)
    writer.add_part("model", "weight", "value", b"\x31" * 2048)
    receipt = writer.commit()
    assert [f.WhichOneof("msg") for f in rig.control.frames] == [
        "weights_intent",
        "weights_transaction",
        "weights_receipt",
        "weights_transaction",
    ]
    row = rig.workspace.weights_row(OWNER, receipt.weights_transaction_id)
    assert row["state"] == "receipt"
    assert rig.store.derived_lookup(receipt.weights_transaction_id)["state"] == "committed"
    observed = rig.control.frames[2].weights_receipt
    with pytest.raises(weights.WeightsExchangeError):
        rig.exchange.receipt(
            attempt,
            output_slot=OUTPUT_SLOT,
            transaction_id=receipt.weights_transaction_id,
            receipt_digest=observed.weights_receipt.weights_receipt_digest,
            canonical_receipt=observed.weights_receipt.weights_receipt_canonical_bytes + b" ",
        )
    assert len(rig.control.frames) == 4


@pytest.mark.parametrize(
    "changed", ["owner", "boot", "lane", "spec", "declaration", "slot", "transaction", "epoch"]
)
def test_checkpoint_restore_is_bound_to_current_workspace_intent(rig: Rig, changed: str) -> None:
    rig.begin()
    request = pb.CheckpointTransferRequest(
        record_owner_epoch=rig.fence.record_owner_epoch,
        worker_boot_id=rig.fence.worker_boot_id,
        subject=pb.CheckpointSubject(weights=ready(rig).weights),
    )
    with rig.exchange.restore_checkpoint(request) as scope:
        assert scope == (TRANSACTION, 1)
    if changed == "owner":
        request.record_owner_epoch += 1
    elif changed == "boot":
        request.worker_boot_id = "other-boot"
    elif changed == "lane":
        request.control_stream_epoch = 1
    elif changed == "spec":
        request.subject.weights.invocation_spec_digest = b"x" * 32
    elif changed == "declaration":
        request.subject.weights.tensorfs_declaration_digest = b"x" * 32
    elif changed == "slot":
        request.subject.weights.output_slot = "other"
    elif changed == "transaction":
        request.subject.weights.weights_transaction_id = "sha256:" + "b" * 64
    else:
        request.subject.weights.writer_epoch += 1
    with pytest.raises(weights.WeightsExchangeError), rig.exchange.restore_checkpoint(request):
        pytest.fail("mismatched restore reached native transfer")


def test_restore_rechecks_workspace_after_io(rig: Rig) -> None:
    attempt = rig.accept()
    rig.begin(attempt)
    request = pb.CheckpointTransferRequest(
        record_owner_epoch=rig.fence.record_owner_epoch,
        worker_boot_id=rig.fence.worker_boot_id,
        subject=pb.CheckpointSubject(weights=ready(rig).weights),
    )
    with pytest.raises(weights.WeightsExchangeError), rig.exchange.restore_checkpoint(request):
        terminal(rig, attempt)
        rig.begin(rig.accept(2))
    assert not rig.exchange.writer_broker.bindings


@pytest.mark.parametrize("cancel_during_validation", [False, True])
def test_restored_native_checkpoint_is_validated_before_writer_adoption(
    rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_during_validation: bool
) -> None:
    rig.begin()
    original = tensorfs.Store.init(tmp_path / "original")
    writer = original.begin_derived(
        TRANSACTION,
        7,
        {},
        rig.targets,
        {},
        [("model", "weight")],
        2048,
        work_fingerprint="sha256:" + "65" * 32,
    )
    writer.add_part("model", "weight", "value", io.BytesIO(b"\x31" * 2048))
    head = cast(dict[str, Any], writer.checkpoint("req-1", OUTPUT_SLOT))
    writer.fence()
    tensorfs.transfer_object(original, rig.store, head["head"], head["head_length"])
    page = original.checkpoint_page(
        head["head"],
        head["head_length"],
        operation_id="req-1",
        slot=OUTPUT_SLOT,
        plan_digest=head["plan_digest"],
    )
    for obj in page["objects"]:
        tensorfs.transfer_object(original, rig.store, obj["object_id"], obj["length"])
    checkpoint = pb.CheckpointRef(
        head=pb.Ref(digest=documents.raw(head["head"]), length=head["head_length"]),
        plan_digest=documents.raw(head["plan_digest"]),
        index=head["index"],
        bytes=head["bytes"],
    )
    request = pb.ValidateWeightsCheckpointRequest(
        intent=ready(rig, checkpoint), tensorfs_declaration_canonical_bytes=rig.declaration
    )
    if cancel_during_validation:
        entered, finish = threading.Event(), threading.Event()
        validate_native = rig.workspace._validate_checkpoint
        answers: list[pb.ValidateWeightsCheckpointResult] = []

        def delayed(row: Any, checkpoint: pb.CheckpointRef) -> None:
            validate_native(row, checkpoint)
            entered.set()
            assert finish.wait(WAIT), "native validation test was not released"

        monkeypatch.setattr(rig.workspace, "_validate_checkpoint", delayed)
        validating = threading.Thread(
            target=lambda: answers.append(
                rig.preparation.ValidateWeightsCheckpoint(request, timeout=WAIT)
            )
        )
        validating.start()
        try:
            assert entered.wait(WAIT), "native checkpoint was not validated"
            # This transition must not be blocked by a global lock held over IO.
            terminal(rig, rig.accept())
        finally:
            finish.set()
            validating.join(WAIT)
        assert not validating.is_alive()
        assert len(answers) == 1 and not answers[0].valid and answers[0].safe_code
        assert rig.store.derived_lookup(TRANSACTION)["state"] == "absent"
        assert not rig.exchange.writer_broker.bindings
        return
    verdict = rig.preparation.ValidateWeightsCheckpoint(request, timeout=WAIT)
    assert verdict.valid and verdict.checkpoint == checkpoint
    assert rig.store.derived_lookup(TRANSACTION)["state"] == "absent"
    assert not rig.exchange.writer_broker.bindings
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
    row = rig.workspace.ready_weights(OWNER, request.intent)
    assert row["ready"]
    # Validation/Ready records the exact checkpoint without importing a native writer.
    assert rig.store.derived_lookup(TRANSACTION)["state"] == "absent"


def test_local_control_host_serves_probe_without_preparation_handlers(tmp_path: Path) -> None:
    bound = threading.Event()
    address = tmp_path / "worker.addr"
    host = GrpcControlHost("127.0.0.1:0", address, on_bound=lambda _: bound.set())
    plane: Any = SimpleNamespace()
    thread = threading.Thread(target=host.serve, args=(plane,), daemon=True)
    thread.start()
    try:
        assert bound.wait(WAIT)
        with grpc.insecure_channel(address.read_text().strip()) as connection:
            api = pb_grpc.RuntimePreparationStub(connection)
            info = api.ProtocolInfo(pb.ProtocolInfoRequest(), timeout=WAIT)
            assert (info.wire_minor, info.minimum_wire_minor) == (
                WIRE_MINOR,
                MIN_COMPATIBLE_WIRE_MINOR,
            )
            with pytest.raises(grpc.RpcError) as error:
                api.ValidateWeightsCheckpoint(pb.ValidateWeightsCheckpointRequest(), timeout=WAIT)
            assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
        assert list(tmp_path.iterdir()) == [address]
    finally:
        host.stop()
        thread.join(WAIT)


def test_retired_exchange_cannot_grant_custody(rig: Rig) -> None:
    with pytest.raises(grpc.RpcError) as error:
        next(rig.legacy.Exchange(iter(()), timeout=WAIT))
    assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
    assert not rig.exchange.writer_broker.bindings
    assert not rig.control.frames
