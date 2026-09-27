"""A weights declaration over the Host's 1 MiB inline bound rides to the Host by reference."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import tensorfs
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.internal.config import Credentials, RuntimeConfig
from cozy_runtime.internal.worker.control import InMemoryControlHost
from cozy_runtime.internal.worker.session import Worker, WorkerOptions
from cozy_runtime.protocol import weights_limits
from cozy_runtime.protocol import worker_pb2 as pb
from test_weights_transaction_refusal import OWNER, SLOT, _attempt, _drain, _transactions

BOUND = weights_limits.MAX_WEIGHTS_DECLARATION_BYTES


def _wide(tensors: int) -> Derivation:
    """A checkpoint of many small tensors: its declaration grows ~270 bytes a tensor."""
    plain = dict(tensorfs.seed_digests())["plain/1"]
    add = {
        f"blocks.{i:05d}.attention.projection.weight": Tensor(
            "f32", (1,), plain, {"value": Part("f32", (1,))}
        )
        for i in range(tensors)
    }
    return Derivation({}, {"model": Target(add=add)}, {}, tuple(("model", name) for name in add))


def _worker(tmp_path: Path) -> Worker:
    store = tensorfs.Store.init(tmp_path / "store")
    worker = Worker(
        RuntimeConfig(cozy_home=tmp_path / "home", credentials=Credentials()),
        WorkerOptions(root=tmp_path / "worker", tensorfs_root=Path(store.root), worker_id="w"),
        InMemoryControlHost(),
    )
    worker.fence.worker_boot_id, worker.fence.record_owner_id = "boot-1", OWNER
    worker.fence.record_owner_epoch, worker.fence.control_stream_epoch = 7, 2
    worker.fence.record_owner_recorded = worker.fence.dispatch_open = True
    return worker


def _intent(worker: Worker, request: str, tensors: int) -> tuple[pb.WeightsIntentFrame, bytes]:
    """Open a real native output and return the intent sent to the Host and the declaration
    the workspace journaled."""
    _, storage = _attempt(worker, worker.options.root.parent, request)
    storage.open_output(SLOT, _wide(tensors))
    (transaction,) = _transactions(worker, request)
    assert worker.workspace is not None
    declaration = worker.workspace.weights_row(OWNER, transaction)["declaration"]
    (intent,) = [f.weights_intent for f in _drain(worker) if f.HasField("weights_intent")]
    return intent, declaration


def test_a_declaration_over_the_inline_bound_is_sent_by_reference(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    try:
        intent, declaration = _intent(worker, "large", 5000)
        assert len(declaration) > BOUND  # a real TensorFS declaration, not a stand-in
        assert not intent.tensorfs_declaration_canonical_bytes
        assert intent.tensorfs_declaration_length == len(declaration)
        assert intent.tensorfs_declaration_digest == sha256(declaration).digest()
        assert intent.requested_writer_epoch > 0

        intent, declaration = _intent(worker, "small", 10)
        assert len(declaration) <= BOUND  # an older Host still takes what fits
        assert intent.tensorfs_declaration_canonical_bytes == declaration
        assert intent.tensorfs_declaration_length == len(declaration)
    finally:
        worker.shutdown()


def test_an_older_host_refusing_a_large_declaration_says_why(tmp_path: Path) -> None:
    worker = _worker(tmp_path)
    try:
        intent, declaration = _intent(worker, "large", 5000)
        assert worker.workspace is not None
        row = worker.workspace.weights_row(OWNER, intent.weights_transaction_id)
        # What a Host before by-reference intents answers every declaration over its bound.
        worker._dispatch_current_frame(
            pb.RecordOwnerFrame(
                weights_transaction_refused=pb.WeightsTransactionRefused(
                    record_owner_epoch=7,
                    control_stream_epoch=2,
                    worker_boot_id="boot-1",
                    request_id="large",
                    attempt_ordinal=1,
                    invocation_spec_digest=intent.invocation_spec_digest,
                    output_slot=SLOT,
                    weights_transaction_id=intent.weights_transaction_id,
                    writer_epoch=row["epoch"],
                    stage=pb.WEIGHTS_HOST_STAGE_INTENT,
                    refusal=pb.WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT,
                    safe_detail="the Host refused this weights intent (intent_conflict)",
                )
            )
        )
        attempt = worker.engine.history[("large", 1)]
        assert f"its weights declaration is {len(declaration)} bytes" in attempt.weights_refused
        assert f"this pod Host's {BOUND}-byte inline limit" in attempt.weights_refused
        assert "worker image" in attempt.weights_refused
    finally:
        worker.shutdown()
