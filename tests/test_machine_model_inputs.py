"""Accepted derived root inputs keep native bytes after their source is released."""

from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal.worker import machine_model_inputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import ack, complete, offer
from test_workspace_custody import produced
from test_workspace_memo import completed, release_original


def model_offer(manifest: str, length: int) -> pb.AttemptOffer:
    offered = offer()
    spec = pb.InvocationSpec.FromString(b"")
    documents_doc = documents.read(offered.invocation_spec_canonical_bytes, pb.InvocationSpec)
    spec.job.build_id = documents_doc["job"]["build_id"]
    spec.job.job_descriptor_id = documents_doc["job"]["job_descriptor_id"]
    spec.payload_digest = documents.spell(documents.digest_of(b"{}"))
    spec.inputs.extend(
        [
            pb.InputBinding(
                input_id="model:source",
                digest=manifest,
                length=length,
                kind_mime="application/vnd.cozy.model-manifest",
            ),
            pb.InputBinding(
                input_id="payload",
                digest=spec.payload_digest,
                length=2,
                kind_mime="application/json",
            ),
        ]
    )
    raw, digest = documents.identity(spec)
    offered.invocation_spec_canonical_bytes = raw
    offered.invocation_spec_digest = digest
    offered.grant.CopyFrom(
        pb.DeliveryGrant(
            invocation_spec_digest=digest,
            inputs=[
                pb.InputAccess(input_id="model:source", url="model://" + manifest),
                pb.InputAccess(input_id="payload", url="data:application/json;base64,e30="),
            ],
        )
    )
    return offered


def test_root_input_survives_source_release_gc_and_workspace_restart(tmp_path: Path) -> None:
    store, workspace, receipt = produced(tmp_path)
    selected = model_offer(receipt.artifact.manifest.digest, receipt.artifact.manifest.length)
    machine_model_inputs.retain(workspace, "owner", selected, {"source"})
    executions = Executions(workspace)
    executions.submit("owner", "root-input", b"x" * 32, selected)
    release_original(workspace, completed(workspace, receipt), receipt)
    tensorfs.gc(str(store.root))
    restored = Workspace(Path(store.root))
    machine_model_inputs.retain(restored, "owner", selected, {"source"})
    assert store.manifest(receipt.artifact.manifest.digest)
    with restored.locked() as db:
        rows = db.execute("SELECT * FROM execution_model_holds WHERE recipient='root'").fetchall()
        assert len(rows) == 1 and rows[0]["path"] == "input/source"
    recovered = Executions(restored)
    outcome = complete(recovered, "root", 14)
    recovered.acknowledge_collection("owner", ack(outcome))
    with restored.locked() as db:
        assert not db.execute("SELECT 1 FROM holds WHERE state='held'").fetchone()


def test_root_input_refuses_foreign_unowned_and_wrong_length_before_acceptance(
    tmp_path: Path,
) -> None:
    _, workspace, receipt = produced(tmp_path)
    selected = model_offer(receipt.artifact.manifest.digest, receipt.artifact.manifest.length)
    with pytest.raises(WorkspaceRefusal, match="derived custody"):
        machine_model_inputs.retain(workspace, "foreign", selected, {"source"})
    with pytest.raises(WorkspaceRefusal, match="declaration"):
        machine_model_inputs.retain(workspace, "owner", selected, {"different"})
    wrong = model_offer(receipt.artifact.manifest.digest, receipt.artifact.manifest.length + 1)
    with pytest.raises(WorkspaceRefusal, match="derived custody"):
        machine_model_inputs.retain(workspace, "owner", wrong, {"source"})
    release_original(workspace, completed(workspace, receipt), receipt)
    with pytest.raises(WorkspaceRefusal, match="derived custody"):
        machine_model_inputs.retain(workspace, "owner", selected, {"source"})
    with workspace.locked() as db:
        assert not db.execute("SELECT 1 FROM execution_model_holds").fetchone()
