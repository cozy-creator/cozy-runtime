"""Native child input custody survives producer/parent cleanup without widening scope."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal import schema
from cozy_runtime.internal.worker import grants, machine_byte_inputs, machine_models
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.calls import Calls
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_native_child_results import NativeInputs, retained


def test_native_child_inputs_hold_independently_and_forward_after_parent_gc(tmp_path: Path) -> None:
    workspace, reply, data = retained(tmp_path)
    Executions(workspace)
    source = reply.byte_result_grants[0]
    file_digest = "sha256:" + hashlib.sha256(data).hexdigest()
    tree_digest = documents.spell(source.source.manifest.digest)
    spec = pb.InvocationSpec(
        inputs=[
            pb.InputBinding(
                input_id="report",
                digest=file_digest,
                length=len(data),
                kind_mime="application/json",
            ),
            pb.InputBinding(
                input_id="bundle",
                digest=tree_digest,
                length=source.source.manifest.length,
                kind_mime=outputs.TREE_MIME,
            ),
        ]
    )
    _, identity = documents.identity(spec)
    parent = AttemptRecord("parent", 1, identity, documents.body(spec))
    parent.grant = grants.bind(
        parent.spec,
        pb.DeliveryGrant(
            invocation_spec_digest=identity,
            inputs=[
                pb.InputAccess(
                    input_id=name,
                    native_tree=pb.NativeByteRetentionRequest(
                        source=source.source, retention_id=source.retention_id
                    ),
                )
                for name in ("report", "bundle")
            ],
        ),
        identity,
    )
    calls = Calls(lambda *_: pb.ChildCallResult(), workspace=workspace, owner=lambda: "owner")
    request = {"report": file_digest, "bundle": tree_digest}
    bindings, access = machine_byte_inputs.inputs(
        workspace, "owner", parent, "child", schema.render(NativeInputs), request, calls
    )
    assert [row.input_id for row in bindings] == ["report", "bundle"]
    assert len({entry.native_tree.retention_id for entry in access}) == 2
    assert all(entry.native_tree.retention_id != source.retention_id for entry in access)
    for original in reply.byte_result_grants:
        outputs.change_hold(
            workspace,
            "owner",
            pb.NativeByteRetentionRequest(
                source=original.source, retention_id=original.retention_id
            ),
            release=True,
        )
    tensorfs.gc(str(workspace.store_root))
    child_spec = pb.InvocationSpec(inputs=bindings)
    _, child_digest = documents.identity(child_spec)
    child = AttemptRecord("child", 1, child_digest, documents.body(child_spec))
    child.grant = grants.bind(
        child.spec,
        pb.DeliveryGrant(invocation_spec_digest=child_digest, inputs=access),
        child_digest,
    )
    files = grants.hydrate_inputs(
        child.grant,
        grants.Authorizer(),
        spool=tmp_path / "child",
        workspace=workspace,
        owner="owner",
    )
    trees = grants.read_trees(
        child.grant,
        grants.Authorizer(),
        spool=tmp_path / "child",
        workspace=workspace,
        owner="owner",
    )
    assert files["report"].local.read_bytes() == data
    assert (trees[tree_digest][0] / "payload").read_bytes() == data
    _, grandchild_access = machine_byte_inputs.inputs(
        workspace, "owner", child, "grandchild", schema.render(NativeInputs), request, calls
    )
    machine_models.release(workspace, "owner", "child")
    tensorfs.gc(str(workspace.store_root))
    for granted in grandchild_access:
        with outputs.leased(workspace, "owner", granted.native_tree):
            pass
    machine_models.release(workspace, "owner", "grandchild")


@pytest.mark.parametrize("mutation", ["foreign_parent", "owner", "released", "digest", "size"])
def test_native_child_input_refuses_missing_or_changed_authority(
    tmp_path: Path, mutation: str
) -> None:
    workspace, reply, data = retained(tmp_path)
    Executions(workspace)
    source = reply.byte_result_grants[0]
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    parent = AttemptRecord("parent", 1, b"p" * 32, {})
    parent.grant.inputs["report"] = grants.BoundInput(
        "report",
        "",
        documents.raw(digest),
        len(data),
        "application/json",
        0,
        pb.NativeByteRetentionRequest(source=source.source, retention_id=source.retention_id),
    )
    request = {"report": digest}
    if mutation == "foreign_parent":
        parent = AttemptRecord("stranger", 1, b"p" * 32, {})
    elif mutation == "digest":
        request["report"] = "sha256:" + "f0" * 32
    elif mutation == "released":
        hold = parent.grant.inputs["report"].native_tree
        assert hold is not None
        outputs.change_hold(workspace, "owner", hold, release=True)
    declaration = {
        "fields": [
            {
                "name": "report",
                "type": {"asset": "file"},
                "asset_bound": {"max_bytes": 1 if mutation == "size" else 200000},
            }
        ]
    }
    calls = Calls(lambda *_: pb.ChildCallResult(), workspace=workspace, owner=lambda: "owner")
    with pytest.raises(WorkspaceRefusal):
        machine_byte_inputs.inputs(
            workspace,
            "other" if mutation == "owner" else "owner",
            parent,
            "child",
            declaration,
            request,
            calls,
        )
    with workspace.locked() as db:
        assert (
            db.execute(
                "SELECT count(*) FROM execution_model_holds WHERE recipient='child'"
            ).fetchone()[0]
            == 0
        )


def test_native_input_projection_uses_only_typed_leaves_and_preserves_list_order() -> None:
    digest = "sha256:" + "a0" * 32
    shape = {
        "fields": [
            {"name": "batch", "type": {"list": {"asset": "file"}}},
            {"name": "other", "type": "opaque"},
            {"name": "maybe", "type": {"union": ["null", {"input": "tree"}]}},
        ]
    }
    payload = {"batch": [digest, digest], "other": {"asset_ref": digest}, "maybe": None}
    before = canonical_json.encode(payload)
    rows = list(machine_byte_inputs._leaves(payload, shape))
    assert [(row[0], row[1]) for row in rows] == [("batch.0", 0), ("batch.1", 1)]
    assert canonical_json.encode(payload) == before
