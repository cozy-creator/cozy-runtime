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


class _Writer:
    """One accepted attempt with a live native writer over two one-element tensors."""

    def __init__(self, tmp_path: Path) -> None:
        self.store = tensorfs.Store.init(tmp_path / "store")
        self.first = Workspace(Path(self.store.root))
        self.second = Workspace(Path(self.store.root))
        invocation = documents.canonical_bytes(
            pb.InvocationSpec(
                job=pb.JobInvocationSpec(
                    installation_id="local-" + "11" * 16, job_descriptor_id="sha256:" + "12" * 32
                )
            )
        )
        self.spec = hashlib.sha256(invocation).digest()
        self.first.accept(
            "owner",
            pb.AttemptOffer(
                request_id="checkpoint-order",
                attempt_ordinal=1,
                invocation_spec_digest=self.spec,
                invocation_spec_canonical_bytes=invocation,
            ),
        )
        self.transaction = weights_transaction_id(
            "owner", "checkpoint-order", documents.spell(self.spec), "model"
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
        declaration = self.store.derived_declaration(
            {}, targets, {}, order, 8, work_fingerprint=fingerprint
        )
        self.declared = hashlib.sha256(declaration).digest()
        row = self.first.begin_weights(
            "owner",
            pb.WeightsIntentFrame(
                request_id="checkpoint-order",
                attempt_ordinal=1,
                invocation_spec_digest=self.spec,
                output_slot="model",
                weights_transaction_id=self.transaction,
                tensorfs_declaration_digest=self.declared,
                tensorfs_declaration_canonical_bytes=declaration,
            ),
        )
        self.epoch = int(row["epoch"])
        self.writer = self.store.begin_derived(
            self.transaction, self.epoch, {}, targets, {}, order, 8, work_fingerprint=fingerprint
        )

    def checkpoint(self, key: str, value: bytes) -> pb.WeightsCheckpointFrame:
        self.writer.add_part("model", key, "value", io.BytesIO(value))
        head: Any = self.writer.checkpoint("checkpoint-order", "model")
        return pb.WeightsCheckpointFrame(
            request_id="checkpoint-order",
            attempt_ordinal=1,
            invocation_spec_digest=self.spec,
            output_slot="model",
            weights_transaction_id=self.transaction,
            writer_epoch=self.epoch,
            tensorfs_declaration_digest=self.declared,
            checkpoint=pb.CheckpointRef(
                head=pb.Ref(digest=documents.raw(head["head"]), length=head["head_length"]),
                plan_digest=documents.raw(head["plan_digest"]),
                index=head["index"],
                bytes=head["bytes"],
            ),
        )

    def ready(self, workspace: Workspace, checkpoint: pb.CheckpointRef) -> None:
        workspace.ready_weights(
            "owner",
            pb.WeightsIntentReadyRequest(
                attempt_ordinal=1,
                weights=pb.WeightsCheckpointSubject(
                    request_id="checkpoint-order",
                    invocation_spec_digest=self.spec,
                    output_slot="model",
                    weights_transaction_id=self.transaction,
                    writer_epoch=self.epoch,
                    tensorfs_declaration_digest=self.declared,
                ),
                checkpoint=checkpoint,
            ),
        )


def test_delayed_native_validation_cannot_replace_committed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _Writer(tmp_path)
    entered, finish = threading.Event(), threading.Event()
    errors: list[Exception] = []
    native_validate = setup.first._validate_checkpoint

    def delayed(row: Any, checkpoint: pb.CheckpointRef) -> None:
        native_validate(row, checkpoint)
        entered.set()
        assert finish.wait(5), "checkpoint race was not released"

    monkeypatch.setattr(setup.first, "_validate_checkpoint", delayed)
    try:
        older = setup.checkpoint("a", b"1234")
        newer = setup.checkpoint("b", b"5678")
        assert newer.checkpoint.index > older.checkpoint.index

        def older_call() -> None:
            try:
                setup.ready(setup.first, older.checkpoint)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=older_call)
        thread.start()
        try:
            assert entered.wait(5), "native validation did not run"
            setup.ready(setup.second, newer.checkpoint)
        finally:
            finish.set()
            thread.join(5)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], WorkspaceRefusal)
        stored = setup.first.weights_row("owner", setup.transaction)
        assert pb.CheckpointRef.FromString(stored["restore_checkpoint"]) == newer.checkpoint
    finally:
        setup.writer.fence()


def test_a_restore_checkpoint_is_validated_and_the_writers_own_is_recorded_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = _Writer(tmp_path)
    try:
        first = setup.checkpoint("a", b"1234")
        second = setup.checkpoint("b", b"5678")
        forged = pb.CheckpointRef()
        forged.CopyFrom(second.checkpoint)
        forged.index += 1
        with pytest.raises(WorkspaceRefusal, match="native closure"):
            setup.ready(setup.first, forged)
        forged.CopyFrom(second.checkpoint)
        forged.head.digest = hashlib.sha256(b"no such checkpoint").digest()
        with pytest.raises(Exception):  # noqa: B017 - TensorFS refuses the unknown head natively
            setup.ready(setup.first, forged)

        validated: list[pb.CheckpointRef] = []
        monkeypatch.setattr(
            setup.second, "_validate_checkpoint", lambda _row, ref: validated.append(ref)
        )
        setup.second.checkpoint("owner", first)
        setup.second.checkpoint("owner", second)
        assert validated == []
        stored = setup.second.weights_row("owner", setup.transaction)
        assert pb.CheckpointRef.FromString(stored["checkpoint"]) == second.checkpoint
        with pytest.raises(WorkspaceRefusal, match="regressed"):
            setup.second.checkpoint("owner", first)
    finally:
        setup.writer.fence()


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
