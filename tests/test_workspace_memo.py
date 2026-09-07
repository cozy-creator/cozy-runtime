from __future__ import annotations

import json
from pathlib import Path

import msgspec
import pytest
import tensorfs

from cozy_runtime.author import WeightsReceipt
from cozy_runtime.internal.worker import workspace_memo
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_custody import produced


def completed(
    workspace: Workspace, receipt: WeightsReceipt, *, result: bytes = b'{"value":7}'
) -> pb.RecordOperationResultCall:
    with workspace.locked() as db:
        row = db.execute(
            "SELECT * FROM weights WHERE id=?", (receipt.weights_transaction_id,)
        ).fetchone()
        spec = row["spec"]
        request_id = row["request"]
        reference = pb.WeightsReceiptRef(
            weights_receipt_digest=row["receipt_digest"],
            weights_receipt_canonical_bytes=row["receipt"],
        )
    raw, digest = documents.identity(
        pb.AttemptOutcomeBody(
            request_id=request_id,
            attempt_ordinal=1,
            invocation_spec_digest=documents.spell(spec),
            status=pb.OUTCOME_STATUS_SUCCEEDED,
            execution_started=True,
            result=pb.ResultEnvelope(inline_result=result),
            weights_receipts=[reference],
        )
    )
    workspace.outcome(
        "owner",
        pb.AttemptOutcome(
            request_id=request_id,
            attempt_ordinal=1,
            invocation_spec_digest=spec,
            outcome_id="outcome",
            outcome_digest=digest,
            outcome_canonical_bytes=raw,
        ),
    )
    return pb.RecordOperationResultCall(
        computation_digest=b"\x71" * 32,
        request_id=request_id,
        attempt_ordinal=1,
        invocation_spec_digest=spec,
        outcome_id="outcome",
        outcome_digest=digest,
    )


def release_original(
    workspace: Workspace, call: pb.RecordOperationResultCall, receipt: WeightsReceipt
) -> None:
    workspace.release_result(
        "owner",
        pb.DerivedResultReleaseRequest(
            weights_transaction_id=receipt.weights_transaction_id,
            tensorfs_receipt_digest=documents.raw(receipt.tensorfs_receipt_digest),
        ),
    )
    workspace.acknowledge(
        "owner",
        pb.AttemptOutcomeAck(
            request_id=call.request_id,
            attempt_ordinal=call.attempt_ordinal,
            invocation_spec_digest=call.invocation_spec_digest,
            outcome_id=call.outcome_id,
            outcome_digest=call.outcome_digest,
            retain_work=False,
        ),
    )


def test_fresh_workspace_lookup_survives_original_cancel_and_prunes_only_unowned_results(
    tmp_path: Path,
) -> None:
    store, workspace, receipt = produced(tmp_path)
    call = completed(workspace, receipt)
    assert workspace_memo.record(workspace, "owner", call).recorded
    release_original(workspace, call, receipt)
    tensorfs.gc(str(store.root))
    other = Workspace(Path(store.root))
    lookup = pb.LookupOperationCall(
        computation_digest=call.computation_digest, consumer_request_id="fresh-script-child"
    )
    hit = workspace_memo.lookup(other, "owner", lookup)
    assert hit.found and len(hit.retentions) == 1
    assert hit.source.request_id == "producer"
    assert hit.source.outcome_digest == call.outcome_digest
    assert workspace_memo.prune(workspace, "owner").removed_entries == 0
    assert workspace_memo.lookup(workspace, "owner", lookup) == hit
    hold = hit.retentions[0]
    other.retain(
        "owner",
        pb.DerivedRetentionRequest(
            weights_transaction_id=hold.weights_transaction_id,
            retention_id=hold.retention_id,
            tensorfs_receipt_digest=hold.tensorfs_receipt_digest,
        ),
        release=True,
    )
    with pytest.raises(WorkspaceRefusal, match="released"):
        workspace_memo.lookup(workspace, "owner", lookup)
    pruned = workspace_memo.prune(workspace, "owner")
    assert pruned.removed_entries == 1 and pruned.reclaimed_bytes > 0
    miss = workspace_memo.lookup(
        workspace,
        "owner",
        pb.LookupOperationCall(
            computation_digest=call.computation_digest,
            consumer_request_id="after-prune",
        ),
    )
    assert not miss.found


def test_operation_identity_and_original_outcome_cannot_be_rewritten(tmp_path: Path) -> None:
    _, workspace, receipt = produced(tmp_path)
    call = completed(workspace, receipt)
    assert workspace_memo.record(workspace, "owner", call).recorded
    with workspace.locked() as db:
        outcome = db.execute("SELECT outcome FROM attempts WHERE request='producer'").fetchone()[0]
    with pytest.raises(WorkspaceRefusal, match="unchanged accepted"):
        workspace.outcome(
            "owner",
            pb.AttemptOutcome(
                request_id="producer",
                attempt_ordinal=1,
                invocation_spec_digest=call.invocation_spec_digest,
                outcome_id="changed-id",
                outcome_digest=call.outcome_digest,
                outcome_canonical_bytes=outcome,
            ),
        )
    foreign = workspace_memo.lookup(
        workspace,
        "another-owner",
        pb.LookupOperationCall(
            computation_digest=call.computation_digest,
            consumer_request_id="new-child",
        ),
    )
    assert not foreign.found
    original = pb.LookupOperationCall(
        computation_digest=call.computation_digest, consumer_request_id="new-child"
    )
    assert workspace_memo.lookup(workspace, "owner", original).found
    original.computation_digest = b"\x72" * 32
    with pytest.raises(WorkspaceRefusal, match="changed"):
        workspace_memo.lookup(workspace, "owner", original)


def test_native_cache_removal_becomes_a_replayable_miss(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path)
    call = completed(workspace, receipt)
    assert workspace_memo.record(workspace, "owner", call).recorded
    release_original(workspace, call, receipt)
    with workspace.locked() as db:
        cache = json.loads(db.execute("SELECT body FROM operation_cache").fetchone()[0])
    held = cache["holds"][0]
    store.release_derived_retention(
        held["transaction_id"], held["native_receipt_digest"], held["retention_id"]
    )
    lookup = pb.LookupOperationCall(
        computation_digest=call.computation_digest, consumer_request_id="missing-native-cache"
    )
    assert not workspace_memo.lookup(workspace, "owner", lookup).found
    assert not workspace_memo.lookup(workspace, "owner", lookup).found


@pytest.mark.parametrize(
    "first_value,second_value",
    [
        (b'{"value":7}', b'{"value":8}'),
        (
            b'{"producer_request_id":"first","value":7}',
            b'{"producer_request_id":"second","value":7}',
        ),
    ],
)
def test_contradictory_results_disable_mapping_without_rewriting_history(
    tmp_path: Path,
    first_value: bytes,
    second_value: bytes,
) -> None:
    _, workspace, first = produced(tmp_path)
    original = completed(workspace, first, result=first_value)
    assert workspace_memo.record(workspace, "owner", original).recorded
    prior = workspace_memo.lookup(
        workspace,
        "owner",
        pb.LookupOperationCall(
            computation_digest=original.computation_digest, consumer_request_id="already-adopted"
        ),
    )
    _, _, second = produced(tmp_path, request_id="other-producer")
    conflicting = completed(workspace, second, result=second_value)
    assert not workspace_memo.record(workspace, "owner", conflicting).recorded
    miss = workspace_memo.lookup(
        workspace,
        "owner",
        pb.LookupOperationCall(
            computation_digest=original.computation_digest, consumer_request_id="after-conflict"
        ),
    )
    assert not miss.found
    assert (
        workspace_memo.lookup(
            workspace,
            "owner",
            pb.LookupOperationCall(
                computation_digest=original.computation_digest,
                consumer_request_id="already-adopted",
            ),
        )
        == prior
    )
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM attempts WHERE state='outcome'").fetchone()[0] == 2
        assert db.execute("SELECT state FROM operation_cache").fetchone()[0] == "disabled"


def test_typed_model_provenance_is_not_a_content_contradiction(tmp_path: Path) -> None:
    schema = b'{"input":"model"}'
    _, workspace, first = produced(tmp_path, result_schema=schema)
    original = completed(workspace, first, result=msgspec.json.encode(first.artifact))
    assert workspace_memo.record(workspace, "owner", original).recorded
    _, _, second = produced(tmp_path, request_id="another-producer", result_schema=schema)
    equivalent = completed(workspace, second, result=msgspec.json.encode(second.artifact))
    assert first.tensorfs_receipt_digest != second.tensorfs_receipt_digest
    assert first.artifact.manifest == second.artifact.manifest
    assert workspace_memo.record(workspace, "owner", equivalent).recorded
    hit = workspace_memo.lookup(
        workspace,
        "owner",
        pb.LookupOperationCall(
            computation_digest=original.computation_digest, consumer_request_id="new-consumer"
        ),
    )
    assert hit.found and hit.source.request_id == "producer"
