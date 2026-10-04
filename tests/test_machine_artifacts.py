"""New recipients reuse real native bytes without rewriting their producer."""

from pathlib import Path

import msgspec
import pytest
import tensorfs

from cozy_runtime.internal.worker.machine_artifacts import model_retention
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from test_workspace_custody import produced
from test_workspace_memo import completed, release_original


def test_independent_machine_hold_survives_producer_release(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path)
    artifact = receipt.artifact
    source = model_retention(workspace, "owner", "parent", "result.model", artifact)
    assert source == model_retention(workspace, "owner", "parent", "result.model", artifact)
    first = workspace.retain("owner", source)
    release_original(workspace, completed(workspace, receipt), receipt)
    tensorfs.gc(str(store.root))
    successor = model_retention(workspace, "owner", "later-parent", "input", artifact)
    assert successor.retention_id != source.retention_id
    assert successor.weights_transaction_id == source.weights_transaction_id
    second = workspace.retain("owner", successor)
    assert first.manifest == second.manifest
    workspace.retain("owner", source, release=True)
    tensorfs.gc(str(store.root))
    with workspace.held_model("owner", successor, first.manifest):
        assert store.manifest(artifact.manifest.digest)["manifest"]
    assert artifact.producer_request_id == "producer"


def test_model_hold_rejects_forged_provenance_and_other_owners(tmp_path: Path) -> None:
    _, workspace, receipt = produced(tmp_path)
    artifact = receipt.artifact
    for changed in (
        msgspec.structs.replace(artifact, producer_request_id="not-the-producer"),
        msgspec.structs.replace(artifact, output_slot="other"),
        msgspec.structs.replace(artifact, tensorfs_receipt_digest="sha256:" + "ab" * 32),
    ):
        with pytest.raises(WorkspaceRefusal, match="provenance"):
            model_retention(workspace, "owner", "parent", "result", changed)
    with pytest.raises(WorkspaceRefusal, match="provenance"):
        model_retention(workspace, "different-owner", "parent", "result", artifact)
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM holds").fetchone()[0] == 0
