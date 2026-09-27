"""Native checkpoint and finalization races preserve the winning durable fact."""

from __future__ import annotations

import hashlib
import io
import threading
from pathlib import Path
from typing import Any

import pytest
import tensorfs

from cozy_runtime.internal.weights_sink import protocol_receipt, weights_transaction_id
from cozy_runtime.internal.worker import workspace_finalize
from cozy_runtime.internal.worker.weights_finalize import FinalizationRefusal
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_custody import produced, retention


@pytest.mark.parametrize("operation", ["checkpoint", "ready"])
def test_delayed_native_validation_cannot_replace_committed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = tensorfs.Store.init(tmp_path / "store")
    first = Workspace(Path(store.root))
    second = Workspace(Path(store.root))
    invocation = documents.canonical_bytes(
        pb.InvocationSpec(
            job=pb.JobInvocationSpec(
                installation_id="local-" + "11" * 16, job_descriptor_id="sha256:" + "12" * 32
            )
        )
    )
    spec = hashlib.sha256(invocation).digest()
    first.accept(
        "owner",
        pb.AttemptOffer(
            request_id="checkpoint-order",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            invocation_spec_canonical_bytes=invocation,
        ),
    )
    transaction = weights_transaction_id(
        "owner", "checkpoint-order", documents.spell(spec), "model"
    )
    plain = next(d for n, d in tensorfs.seed_digests() if n == "plain/1")
    tensor = {
        "logical_dtype": "f32",
        "shape": [1],
        "encoding": plain,
        "parts": {"value": {"dtype": "f32", "shape": [1]}},
    }
    targets = {"model": {"drop": [], "add": {"a": tensor, "b": tensor}}}
    order = [("model", "a"), ("model", "b")]
    fingerprint = "sha256:" + "42" * 32
    declaration = store.derived_declaration({}, targets, {}, order, 8, work_fingerprint=fingerprint)
    declared = hashlib.sha256(declaration).digest()
    row = first.begin_weights(
        "owner",
        pb.WeightsIntentFrame(
            request_id="checkpoint-order",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            output_slot="model",
            weights_transaction_id=transaction,
            tensorfs_declaration_digest=declared,
            tensorfs_declaration_canonical_bytes=declaration,
        ),
    )
    epoch = int(row["epoch"])
    writer = store.begin_derived(
        transaction, epoch, {}, targets, {}, order, 8, work_fingerprint=fingerprint
    )

    def frame(head: Any) -> pb.WeightsCheckpointFrame:
        return pb.WeightsCheckpointFrame(
            request_id="checkpoint-order",
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            output_slot="model",
            weights_transaction_id=transaction,
            writer_epoch=epoch,
            tensorfs_declaration_digest=declared,
            checkpoint=pb.CheckpointRef(
                head=pb.Ref(digest=documents.raw(head["head"]), length=head["head_length"]),
                plan_digest=documents.raw(head["plan_digest"]),
                index=head["index"],
                bytes=head["bytes"],
            ),
        )

    def commit(workspace: Workspace, checkpoint: pb.WeightsCheckpointFrame) -> None:
        if operation == "checkpoint":
            workspace.checkpoint("owner", checkpoint)
        else:
            workspace.ready_weights(
                "owner",
                pb.WeightsIntentReadyRequest(
                    attempt_ordinal=1,
                    weights=pb.WeightsCheckpointSubject(
                        request_id="checkpoint-order",
                        invocation_spec_digest=spec,
                        output_slot="model",
                        weights_transaction_id=transaction,
                        writer_epoch=epoch,
                        tensorfs_declaration_digest=declared,
                    ),
                    checkpoint=checkpoint.checkpoint,
                ),
            )

    entered, finish = threading.Event(), threading.Event()
    errors: list[Exception] = []
    native_validate = first._validate_checkpoint

    def delayed(row: Any, checkpoint: pb.CheckpointRef) -> None:
        native_validate(row, checkpoint)
        entered.set()
        assert finish.wait(5), "checkpoint race was not released"

    monkeypatch.setattr(first, "_validate_checkpoint", delayed)
    try:
        writer.add_part("model", "a", "value", io.BytesIO(b"1234"))
        older = frame(writer.checkpoint("checkpoint-order", "model"))
        writer.add_part("model", "b", "value", io.BytesIO(b"5678"))
        newer = frame(writer.checkpoint("checkpoint-order", "model"))
        assert newer.checkpoint.index > older.checkpoint.index

        def older_call() -> None:
            try:
                commit(first, older)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=older_call)
        thread.start()
        try:
            assert entered.wait(5), "native validation did not run"
            commit(second, newer)
        finally:
            finish.set()
            thread.join(5)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], WorkspaceRefusal)
        stored = first.weights_row("owner", transaction)
        key = "checkpoint" if operation == "checkpoint" else "restore_checkpoint"
        assert pb.CheckpointRef.FromString(stored[key]) == newer.checkpoint
    finally:
        writer.fence()


def test_conflicting_finalization_cannot_release_an_adopted_result(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path)
    with workspace.locked() as db:
        attempt = dict(db.execute("SELECT * FROM attempts").fetchone())
    reference, _, _ = protocol_receipt(
        receipt,
        owner_scope="owner",
        request_id="producer",
        invocation_spec_digest=documents.spell(attempt["spec"]),
    )
    raw, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(attempt["spec"]),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            execution_started=True,
            weights_receipts=[reference],
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id="producer",
            attempt_ordinal=1,
            invocation_spec_digest=attempt["spec"],
            outcome_id="out-producer",
            outcome_digest=digest,
            outcome_canonical_bytes=raw,
        ),
    )
    adopt = pb.WeightsFinalizeRequest(
        owner_authority_scope="owner",
        request_id="producer",
        invocation_spec_digest=attempt["spec"],
        invocation_spec_canonical_bytes=attempt["invocation"],
        output_slot="model",
        weights_receipt_digest=reference.weights_receipt_digest,
        disposition=pb.WEIGHTS_FINALIZE_DISPOSITION_ADOPT,
        scratch_root_id="test/adopted",
    )
    result = workspace_finalize.finalize(workspace, "owner", adopt)
    assert result.outcome == pb.WEIGHTS_FINALIZE_OUTCOME_ADOPTED
    abandon = pb.WeightsFinalizeRequest()
    abandon.CopyFrom(adopt)
    abandon.disposition = pb.WEIGHTS_FINALIZE_DISPOSITION_ABANDON
    abandon.scratch_root_id = ""
    with pytest.raises(FinalizationRefusal):
        workspace_finalize.finalize(workspace, "owner", abandon)
    assert workspace.weights_row("owner", receipt.weights_transaction_id)["state"] == "receipt"
    assert store.derived_lookup(receipt.weights_transaction_id)["disposition"]["kind"] == "adopted"
    held = workspace.retain("owner", retention(receipt, 73))
    assert not held.released and held.manifest.digest
    # The original first-wins request still replays after the refused conflict.
    assert workspace_finalize.finalize(workspace, "owner", adopt) == result
