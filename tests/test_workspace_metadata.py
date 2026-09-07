"""Optional memo metadata has a budget; owned native results survive compaction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal.worker import workspace_memo as memo
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb
from test_workspace_cache_recovery import native_outputs, scalar


def request(key: bytes, consumer: str) -> pb.LookupOperationCall:
    return pb.LookupOperationCall(computation_digest=key, consumer_request_id=consumer)


def release(workspace: Workspace, held: pb.DerivedRetentionResult) -> None:
    workspace.retain(
        "owner",
        pb.DerivedRetentionRequest(
            weights_transaction_id=held.weights_transaction_id,
            tensorfs_receipt_digest=held.tensorfs_receipt_digest,
            retention_id=held.retention_id,
        ),
        release=True,
    )


def settle_original(
    workspace: Workspace,
    originals: list[pb.DerivedResultReleaseRequest],
    call: pb.RecordOperationResultCall,
) -> None:
    for original in originals:
        workspace.release_result("owner", original)
    workspace.acknowledge(
        "owner",
        pb.AttemptOutcomeAck(
            request_id=call.request_id,
            attempt_ordinal=call.attempt_ordinal,
            invocation_spec_digest=call.invocation_spec_digest,
            outcome_id=call.outcome_id,
            outcome_digest=call.outcome_digest,
        ),
    )


def usage(workspace: Workspace) -> int:
    with workspace.locked() as db:
        return memo.metadata_bytes(db)


def test_scalar_hits_and_initial_misses_do_not_accumulate_recipient_history(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "store")
    key = b"a" * 32
    for index in range(40):
        assert not memo.lookup(workspace, "owner", request(key, f"absent-{index}")).found
    assert scalar(workspace, "source", 7, key)
    before = usage(workspace)
    for index in range(40):
        assert memo.lookup(workspace, "owner", request(key, f"recipient-{index}")).found
    assert usage(workspace) == before
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM operation_lookups").fetchone()[0] == 0
        # Forgetting a scalar reply requires no native custody compensation.
        db.execute("UPDATE operation_cache SET state='evicting'")
    assert not memo.lookup(workspace, "owner", request(key, "recipient-0")).found


def test_delivered_native_result_compacts_its_copy_without_releasing_bytes(tmp_path: Path) -> None:
    store, workspace, originals, call = native_outputs(tmp_path)
    assert memo.record(workspace, "owner", call).recorded
    lookup = request(call.computation_digest, "recipient")
    hit = memo.lookup(workspace, "owner", lookup)
    assert hit.found and len(hit.retentions) == 2
    assert memo.lookup(Workspace(Path(store.root)), "owner", lookup) == hit
    before = usage(workspace)
    memo.delivered(workspace, "owner", lookup.consumer_request_id)
    memo.delivered(workspace, "owner", lookup.consumer_request_id)
    assert usage(workspace) < before
    with workspace.locked() as db:
        row = db.execute("SELECT state,body FROM operation_lookups").fetchone()
        payload = json.loads(row["body"])
        assert row["state"] == "acknowledged"
        assert "outcome_canonical_bytes" not in payload["source"]
        assert len(payload["holds"]) == 2
        assert db.execute("SELECT count(*) FROM holds WHERE state='held'").fetchone()[0] == 4
    with pytest.raises(WorkspaceRefusal, match="delivered"):
        memo.lookup(workspace, "owner", lookup)
    settle_original(workspace, originals, call)
    tensorfs.gc(str(store.root))
    for held in hit.retentions:
        assert store.manifest("sha256:" + held.manifest.digest.hex())["manifest"]
    assert memo.prune(workspace, "owner").removed_entries == 0
    for held in hit.retentions:
        release(workspace, held)
    assert memo.prune(workspace, "owner").removed_entries == 1


def test_release_compacts_only_after_all_output_holds_are_settled(tmp_path: Path) -> None:
    _, workspace, _, call = native_outputs(tmp_path)
    assert memo.record(workspace, "owner", call).recorded
    hit = memo.lookup(workspace, "owner", request(call.computation_digest, "recipient"))
    release(workspace, hit.retentions[0])
    with workspace.locked() as db:
        memo.compact_released(db, "owner")
        row = db.execute("SELECT body FROM operation_lookups").fetchone()
        assert "outcome_canonical_bytes" in json.loads(row[0])["source"]
    release(workspace, hit.retentions[1])
    with workspace.locked() as db:
        memo.compact_released(db, "owner")
        row = db.execute("SELECT body FROM operation_lookups").fetchone()
        assert "outcome_canonical_bytes" not in json.loads(row[0])["source"]
    with pytest.raises(WorkspaceRefusal, match="released"):
        memo.lookup(workspace, "owner", request(call.computation_digest, "recipient"))


def test_metadata_budget_declines_new_native_copies_but_keeps_existing_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, workspace, originals, call = native_outputs(tmp_path)
    assert memo.record(workspace, "owner", call).recorded
    prior = request(call.computation_digest, "accepted")
    hit = memo.lookup(workspace, "owner", prior)
    settle_original(workspace, originals, call)
    before = usage(workspace)
    monkeypatch.setattr(memo, "MAX_METADATA_BYTES", before)
    assert not memo.lookup(workspace, "owner", request(call.computation_digest, "new")).found
    assert usage(workspace) == before
    assert memo.lookup(workspace, "owner", prior) == hit
    # Delivery and required native cleanup are not optional cache admissions.
    memo.delivered(workspace, "owner", prior.consumer_request_id)
    for held in hit.retentions:
        release(workspace, held)
    assert memo.prune(workspace, "owner").removed_entries == 1


def test_metadata_budget_is_shared_across_owners_and_does_not_block_terminal_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace(tmp_path / "store")
    assert scalar(workspace, "first", 7, b"a" * 32)
    monkeypatch.setattr(memo, "MAX_METADATA_BYTES", usage(workspace))
    assert not scalar(workspace, "second", 8, b"b" * 32, owner="another-owner")
    with workspace.locked() as db:
        row = dict(db.execute("SELECT * FROM attempts WHERE owner='another-owner'").fetchone())
        assert db.execute("SELECT count(*) FROM operation_cache").fetchone()[0] == 1
    ack = pb.AttemptOutcomeAck(
        request_id=row["request"],
        attempt_ordinal=row["ordinal"],
        invocation_spec_digest=row["spec"],
        outcome_id=row["outcome_id"],
        outcome_digest=row["outcome_digest"],
    )
    workspace.acknowledge("another-owner", ack)
    workspace.acknowledge("another-owner", ack)
    with workspace.locked() as db:
        assert (
            db.execute("SELECT count(*) FROM attempts WHERE owner='another-owner'").fetchone()[0]
            == 0
        )


def test_budget_always_allows_disabling_a_contradicted_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace(tmp_path / "store")
    key = b"a" * 32
    assert scalar(workspace, "first", 7, key, evidence="x" * 8192)
    before = usage(workspace)
    monkeypatch.setattr(memo, "MAX_METADATA_BYTES", before)
    assert not scalar(workspace, "second", 8, key, evidence="x" * 8192)
    assert usage(workspace) < before
    assert not memo.lookup(workspace, "owner", request(key, "recipient")).found
    with workspace.locked() as db:
        row = db.execute("SELECT state,body FROM operation_cache").fetchone()
        assert row["state"] == "disabled"
        assert len(row["body"]) < 2048


@pytest.mark.parametrize("consumer", ["", "invalid/path", "x" * 257, "invalid space"])
def test_lookup_rejects_invalid_recipient_identity(tmp_path: Path, consumer: str) -> None:
    workspace = Workspace(tmp_path / "store")
    with pytest.raises(WorkspaceRefusal, match="bounded consumer"):
        memo.lookup(workspace, "owner", request(b"a" * 32, consumer))
    assert usage(workspace) == 0
