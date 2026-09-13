"""A valid evaluator report is insufficient without the machine's producing work."""

import hashlib
from pathlib import Path

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ObjectRef
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal.worker import machine_assessment, machine_models, workspace_byte_outputs
from cozy_runtime.internal.worker.machine_publication import PublicationRefusal
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from test_workspace_byte_outputs import producing


def test_received_report_bytes_get_independent_effect_custody(tmp_path: Path) -> None:
    workspace, spec, manifest, path = producing(tmp_path)
    data = path.read_bytes()
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    source = workspace_byte_outputs.commit(
        workspace, "owner", "producer", 1, spec, "report", manifest, [("payload", path)], 200000
    )
    with pytest.raises(WorkspaceRefusal, match="received"):
        machine_assessment.read_file(workspace, "owner", "parent", "effect", "report", digest)
    machine_models.retain_bytes(workspace, "owner", "parent", "report", source)
    observed, producer = machine_assessment.read_file(
        workspace, "owner", "parent", "effect", "report", digest
    )
    assert observed == data and producer == "producer"
    machine_models.release(workspace, "owner", "parent")
    # The effect's own grant remains readable after the caller releases its copy.
    assert machine_models.file_source(workspace, "owner", "effect", digest).source == source


def test_semantically_valid_report_does_not_establish_execution_provenance(tmp_path: Path) -> None:
    from cozy_eval import contract  # type: ignore[import-untyped]
    from cozy_eval.assessment import AssessmentReport  # type: ignore[import-untyped]

    raw = (Path(__file__).parent / "testdata/assessment-provenance-report.json").read_bytes()
    validated = contract.load(raw, expect=AssessmentReport)
    validated.body.contract_check()
    candidate = validated.body.subject.candidate_checkpoint
    checkpoint = CheckpointRef(
        "alice/model", candidate, ObjectRef(candidate, 161), "existing-publication", ""
    )
    workspace = Workspace(tmp_path / "store")
    with pytest.raises(PublicationRefusal, match="binding_mismatch"):
        machine_assessment.verify(
            workspace,
            "owner",
            "invented-producer",
            "effect",
            checkpoint,
            raw,
            canonical_json.encode([]),
        )
    with workspace.locked() as db:
        assert db.execute("SELECT count(*) FROM execution_model_holds").fetchone()[0] == 0


def test_malformed_report_is_refused_by_the_evaluator(tmp_path: Path) -> None:
    candidate = "sha256:" + "11" * 32
    checkpoint = CheckpointRef(
        "alice/model", candidate, ObjectRef(candidate, 161), "publication", ""
    )
    with pytest.raises(PublicationRefusal, match="report_invalid"):
        machine_assessment.verify(
            Workspace(tmp_path / "store"), "owner", "producer", "effect", checkpoint, b"{}", b"[]"
        )
