"""A Host custody refusal fails only its transaction's attempt; the session keeps running."""

from __future__ import annotations

import queue
from hashlib import sha256
from pathlib import Path

import pytest
import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime import canonical_json
from cozy_runtime.author import CapabilityError
from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.weights_writer import ExecutionStorage
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import worker_pb2 as pb
from weights_channel import exchange as channel_exchange

OWNER = "owner-scope"
SLOT = "model"
INVOCATION = (
    (Path(__file__).parent / "testdata/worker-protocol/canonical/invocation_spec_job.json")
    .read_bytes()
    .strip()
)


def _attempt(worker: Worker, root: Path, request: str) -> tuple[AttemptRecord, ExecutionStorage]:
    digest = sha256(INVOCATION).digest()
    assert worker.workspace is not None
    worker.workspace.accept(
        OWNER,
        pb.AttemptOffer(
            request_id=request,
            attempt_ordinal=1,
            invocation_spec_digest=digest,
            invocation_spec_canonical_bytes=INVOCATION,
        ),
    )
    attempt = AttemptRecord(
        request_id=request, attempt=1, digest=digest, spec=canonical_json.decode(INVOCATION)
    )
    attempt.state = "running"
    attempt.spool = root / request
    attempt.spool.mkdir()
    attempt.weights_work_fingerprint = "sha256:" + "65" * 32
    worker.engine.history[attempt.key()] = attempt
    storage = ExecutionStorage(
        attempt.spool,
        channel_exchange(lambda frame: worker.weights.writer_broker.handle(attempt, frame)),
        {SLOT: attempt.spec["outputs"][0]["max_bytes"]},
    )
    return attempt, storage


def _definition() -> Derivation:
    plain = dict(tensorfs.seed_digests())["plain/1"]
    return Derivation(
        {},
        {
            "model": Target(
                add={"weight": Tensor("f32", (512,), plain, {"value": Part("f32", (512,))})}
            )
        },
        {},
        (("model", "weight"),),
    )


def _drain(worker: Worker) -> list[pb.WorkerFrame]:
    frames: list[pb.WorkerFrame] = []
    while True:
        try:
            frame = worker.outbound.get_nowait()
        except queue.Empty:
            return frames
        if frame is not None:
            frames.append(frame)


def test_a_refused_transaction_fails_only_its_attempt(tmp_path: Path) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    worker = Worker(
        RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
        WorkerOptions(root=tmp_path / "worker", tensorfs_root=Path(store.root), worker_id="w"),
        InMemoryControlHost(),
    )
    worker.fence.worker_boot_id, worker.fence.record_owner_id = "boot-1", OWNER
    worker.fence.record_owner_epoch, worker.fence.control_stream_epoch = 7, 2
    worker.fence.record_owner_recorded = worker.fence.dispatch_open = True
    try:
        refused, refused_storage = _attempt(worker, tmp_path, "refused")
        concurrent, concurrent_storage = _attempt(worker, tmp_path, "concurrent")
        refused_writer = refused_storage.open_output(SLOT, _definition())
        refused_writer.add_part("model", "weight", "value", b"\x31" * 2048)
        concurrent_writer = concurrent_storage.open_output(SLOT, _definition())
        assert worker.workspace is not None
        (transaction,) = _transactions(worker, "refused")
        row = worker.workspace.weights_row(OWNER, transaction)
        _drain(worker)

        def refusal(**envelope: object) -> pb.RecordOwnerFrame:
            message = pb.WeightsTransactionRefused(
                record_owner_epoch=7,
                control_stream_epoch=2,
                worker_boot_id="boot-1",
                request_id="refused",
                attempt_ordinal=1,
                invocation_spec_digest=refused.digest,
                output_slot=SLOT,
                weights_transaction_id=transaction,
                writer_epoch=row["epoch"],
                stage=pb.WEIGHTS_HOST_STAGE_RECEIPT,
                refusal=pb.WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT,
                safe_detail="weights receipt conflicts with the Host record; rerun the attempt",
            )
            for name, value in envelope.items():
                setattr(message, name, value)
            return pb.RecordOwnerFrame(weights_transaction_refused=message)

        # Another session's refusal, or one naming another writer epoch, changes nothing.
        worker._dispatch_current_frame(refusal(worker_boot_id="boot-0"))
        worker._dispatch_current_frame(refusal(writer_epoch=row["epoch"] + 1))
        assert not refused.weights_refused
        worker._dispatch_current_frame(refusal())
        worker._dispatch_current_frame(refusal())  # a duplicate is a no-op
        assert "WEIGHTS_HOST_REFUSAL_RECEIPT_CONFLICT" in refused.weights_refused
        assert worker.workspace.weights_row(OWNER, transaction)["state"] == "released"

        # Its writer is stopped and no further frame for it is sent.
        with pytest.raises((CapabilityError, OSError)):
            refused_writer.commit()
        # The concurrent transaction commits, is recorded and observed.
        concurrent_writer.add_part("model", "weight", "value", b"\x32" * 2048)
        concurrent_writer.commit()
        frames = _drain(worker)
        kinds = {(f.WhichOneof("msg"), _subject(f)) for f in frames}
        assert ("weights_receipt", "concurrent") in kinds
        assert not any(subject == "refused" for _, subject in kinds)
        assert concurrent.weights_receipts[SLOT].weights_receipt_canonical_bytes
        assert not concurrent.weights_refused and not worker.stop.is_set()

        # The refused attempt's terminal is weights_custody_refused, whatever it returned.
        assert refused.spool is not None
        outcome = worker.engine._terminal(refused, {"ok": True}, refused.spool, "refused")
        body = canonical_json.decode(outcome.outcome_canonical_bytes)
        assert body["status"] == pb.OUTCOME_STATUS_FAILED
        assert body["safe_message"].startswith("weights_custody_refused: output 'model'")
        assert "rerun the attempt" in body["safe_message"]
    finally:
        worker.shutdown()


def _transactions(worker: Worker, request: str) -> set[str]:
    assert worker.workspace is not None
    with worker.workspace.locked() as db:
        return {
            row["id"] for row in db.execute("SELECT id FROM weights WHERE request=?", (request,))
        }


def _subject(frame: pb.WorkerFrame) -> str:
    kind = frame.WhichOneof("msg")
    inner = getattr(frame, kind) if kind else None
    return str(getattr(inner, "request_id", "")) if inner is not None else ""
