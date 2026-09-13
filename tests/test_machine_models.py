"""Typed borrowed results retain physical bytes until their recipient collects them."""

import shutil
from pathlib import Path

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import machine_models, workspace_memo
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import ack, complete, offer
from test_workspace_custody import produced
from test_workspace_memo import completed, release_original


def test_borrowed_model_root_collection_retains_bytes_and_rejects_a_copied_journal(
    tmp_path: Path,
) -> None:
    store, workspace, receipt = produced(tmp_path)
    value = msgspec.to_builtins(receipt.artifact)
    executions = Executions(workspace)
    executions.submit("owner", "root", b"c" * 32, offer(), result_schema=b'{"input":"model"}')
    machine_models.retain_result(
        workspace, "owner", "root", "call.0.result", value, {"input": "model"}
    )
    release_original(workspace, completed(workspace, receipt), receipt)
    tensorfs.gc(str(store.root))
    outcome = complete(executions, "root", value)
    assert executions.collect("owner", "root") == outcome

    empty = Workspace(tmp_path / "empty")
    shutil.copyfile(workspace.directory / "journal.sqlite3", empty.directory / "journal.sqlite3")
    with pytest.raises(Exception) as lost:
        Executions(empty).collect("owner", "root")
    assert getattr(lost.value, "code", None) in {
        "ROOT_ABSENT",
        "OBJECT_ABSENT",
        "TRANSACTION_CLOSED",
    }
    assert not executions.status("owner", "root").collected
    assert store.manifest(receipt.artifact.manifest.digest)
    executions.acknowledge_collection("owner", ack(outcome))
    with workspace.locked() as db:
        assert not db.execute("SELECT 1 FROM holds WHERE state<>'released'").fetchone()
    tensorfs.gc(str(store.root))
    assert executions.collect("owner", "root") == outcome  # Collected history is metadata.


def test_schema_walk_never_recognizes_user_json_as_model_provenance(tmp_path: Path) -> None:
    _, workspace, receipt = produced(tmp_path)
    value = msgspec.to_builtins(receipt.artifact)
    assert list(machine_models.artifacts(value, {"opaque": "json"})) == []
    assert list(machine_models.artifacts(value, {"map": {"key": "str", "value": "str"}})) == []
    schema = {
        "fields": [
            {
                "name": "models",
                "type": {"tuple": [{"input": "model"}, {"union": ["null", {"input": "model"}]}]},
            }
        ]
    }
    found = list(machine_models.artifacts({"models": [value, None]}, schema))
    assert len(found) == 1 and found[0][1] == receipt.artifact
    with workspace.locked() as db:
        assert not db.execute("SELECT 1 FROM holds").fetchone()


def test_parent_hold_survives_release_of_temporary_memo_recipient(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path, result_schema=b'{"input":"model"}')
    call = completed(
        workspace, receipt, result=canonical_json.encode(msgspec.to_builtins(receipt.artifact))
    )
    assert workspace_memo.record(workspace, "owner", call).recorded
    hit = workspace_memo.lookup(
        workspace,
        "owner",
        pb.LookupOperationCall(
            computation_digest=call.computation_digest, consumer_request_id="child"
        ),
    )
    assert hit.found
    machine_models.retain_result(
        workspace,
        "owner",
        "parent",
        "result",
        msgspec.to_builtins(receipt.artifact),
        {"input": "model"},
    )
    workspace_memo.delivered(workspace, "owner", "child")
    workspace_memo.release_lookup(workspace, "owner", "child")
    workspace_memo.release_lookup(workspace, "owner", "child")
    release_original(workspace, call, receipt)
    tensorfs.gc(str(store.root))
    assert store.manifest(receipt.artifact.manifest.digest)
    machine_models.release(workspace, "owner", "parent")
    assert workspace_memo.prune(workspace, "owner").removed_entries == 1
