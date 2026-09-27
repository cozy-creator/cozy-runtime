"""Typed borrowed results retain physical bytes until their recipient collects them."""

import shutil
from pathlib import Path

import msgspec
import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import machine_models, workspace_memo
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
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
    executions.submit(
        "owner",
        "root",
        b"c" * 32,
        offer(),
        result_schema=b'{"input":"model"}',
        expected_execution_workspace_id=executions.workspace_id,
    )
    machine_models.retain_result(
        workspace, "owner", "root", "call.0.result", value, {"input": "model"}
    )
    release_original(workspace, completed(workspace, receipt), receipt)
    tensorfs.gc(str(store.root))
    outcome = complete(executions, "root", value)
    assert executions.collect("owner", "root") == outcome

    empty = Workspace(tmp_path / "empty")
    with workspace.locked() as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
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


def test_metadata_file_resolver_requires_actual_parent_received_custody(tmp_path: Path) -> None:
    import hashlib

    from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
    from test_workspace_byte_outputs import producing

    workspace, spec, manifest, path = producing(tmp_path)
    data = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    source = outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    with pytest.raises(WorkspaceRefusal, match="received"):
        machine_models.file_source(workspace, "owner", "parent", digest, len(data))
    grant = machine_models.retain_bytes(workspace, "owner", "parent", "call.0.report", source)
    held = machine_models.file_source(workspace, "owner", "parent", digest, len(data))
    assert held.source == source and held.retention_id == grant.retention_id
    assert machine_models.file_source(workspace, "owner", "parent", digest) == held
    for owner, parent, wrong in (
        ("foreign", "parent", digest),
        ("owner", "different", digest),
        ("owner", "parent", "sha256:" + "de" * 32),
    ):
        with pytest.raises(WorkspaceRefusal, match="received"):
            machine_models.file_source(workspace, owner, parent, wrong, len(data))
    with pytest.raises(WorkspaceRefusal, match="8 MiB"):
        machine_models.file_source(workspace, "owner", "parent", digest, (8 << 20) + 1)
    effect = machine_models.retain_bytes(workspace, "owner", "effect", "report", source)
    machine_models.release(workspace, "owner", "parent")
    tensorfs.gc(str(workspace.store_root))
    with outputs.leased(
        workspace,
        "owner",
        pb.NativeByteRetentionRequest(source=source, retention_id=effect.retention_id),
    ):
        pass
    with pytest.raises(WorkspaceRefusal, match="received"):
        machine_models.file_source(workspace, "owner", "parent", digest, len(data))


def test_terminal_model_records_bind_typed_pointer_artifact_and_native_hold(tmp_path: Path) -> None:
    _, workspace, receipt = produced(tmp_path)
    artifact = msgspec.to_builtins(receipt.artifact)
    schema = {"fields": [{"name": "model", "type": {"input": "model"}}]}
    records = machine_models.result_records(workspace, "owner", "root", {"model": artifact}, schema)
    assert records == machine_models.result_records(
        workspace, "owner", "root", {"model": artifact}, schema
    )
    pointer, raw, retention = records[0]
    assert pointer == "/model" and canonical_json.decode(raw) == artifact
    assert retention.weights_transaction_id == receipt.weights_transaction_id
    with pytest.raises(WorkspaceRefusal, match="32 entries"):
        machine_models.result_records(
            workspace, "owner", "root", [artifact] * 33, {"list": {"input": "model"}}
        )
    envelope = pb.ResultEnvelope(
        inline_result=canonical_json.encode({"model": artifact}),
        retained_models=[
            pb.RetainedModelResult(
                result_pointer=pointer,
                model_artifact_canonical_bytes=raw,
                retention=retention,
            )
        ],
    )
    body = pb.AttemptOutcomeBody(
        request_id="root",
        attempt_ordinal=1,
        invocation_spec_digest="sha256:" + "ab" * 32,
        status=pb.OUTCOME_STATUS_SUCCEEDED,
        result=envelope,
    )
    canonical, digest = documents.identity(body)
    terminal = pb.AttemptOutcome(
        request_id="root", outcome_canonical_bytes=canonical, outcome_digest=digest
    )
    Executions(workspace)._verify_outputs("owner", terminal, canonical_json.encode(schema))
    body.result.retained_models[0].result_pointer = "/other"
    canonical, digest = documents.identity(body)
    terminal.outcome_canonical_bytes, terminal.outcome_digest = canonical, digest
    with pytest.raises(WorkspaceRefusal, match="typed result"):
        Executions(workspace)._verify_outputs("owner", terminal, canonical_json.encode(schema))
