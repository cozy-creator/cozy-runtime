"""A valid evaluator report is insufficient without the machine's producing work."""

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ObjectRef
from cozy_runtime.author.publication import CheckpointRef
from cozy_runtime.internal.worker import machine_assessment, machine_models, workspace_byte_outputs
from cozy_runtime.internal.worker.machine_publication import PublicationRefusal
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
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


def test_serving_association_uses_only_the_selected_exact_placement() -> None:
    fixture = Path(__file__).parent / "testdata/worker-protocol/canonical/placement_set.json"
    placement = documents.read(fixture.read_bytes(), pb.PlacementSet)["placements"][0]
    entry = next(row for row in placement["entrypoints"] if row["name"] == "upscale")
    artifact = next(row["manifest"] for row in placement["models"] if row["id"] == "upscaler")
    checkpoint = artifact["digest"]
    request = {"upscaler": {"manifest": artifact}, "seed": 1234}
    intent = machine_assessment.Intent("module", "export", request)
    installation = placement["installation_id"]
    spec = pb.InvocationSpec(
        payload_digest=documents.spell(hashlib.sha256(canonical_json.encode(request)).digest()),
        installation_id=installation,
        serving=pb.ServingInvocationSpec(
            bindings_digest=placement["bindings_digest"],
            entrypoint_binding_digest=entry["entrypoint_binding_digest"],
        ),
    )

    def prepared(placement: dict[str, Any]) -> bytes:
        held = {"placement": placement, "installation_id": installation}
        return canonical_json.encode({"installations": {installation: held}})

    preparation = prepared(placement)
    call = canonical_json.encode(
        {"kind": "serving", "installation_id": installation, "entrypoint": "upscale"}
    )
    assert machine_assessment.render_arguments(spec, intent, preparation, call, checkpoint) == {
        "seed": 1234
    }
    # The same retained installation also contains H3. A report cannot substitute that
    # unselected model, change the admitted bindings, or hide a mixed-model slot.
    with pytest.raises(PublicationRefusal, match="binding_mismatch"):
        machine_assessment.render_arguments(
            spec, intent, preparation, call, placement["models"][0]["manifest"]["digest"]
        )
    changed = deepcopy(spec)
    changed.serving.bindings_digest = "sha256:" + "11" * 32
    with pytest.raises(PublicationRefusal, match="binding_mismatch"):
        machine_assessment.render_arguments(changed, intent, preparation, call, checkpoint)
    changed = deepcopy(spec)
    changed.installation_id = "install-other"
    with pytest.raises(PublicationRefusal, match="binding_mismatch"):
        machine_assessment.render_arguments(changed, intent, preparation, call, checkpoint)
    changed_intent = machine_assessment.Intent("module", "export", request | {"seed": 1235})
    with pytest.raises(PublicationRefusal, match="binding_mismatch"):
        machine_assessment.render_arguments(spec, changed_intent, preparation, call, checkpoint)
    changed_placement = deepcopy(placement)
    selected = next(row for row in changed_placement["entrypoints"] if row["name"] == "upscale")
    selected["slots"][0]["components"][0]["model_id"] = "h3"
    mixed = prepared(changed_placement)
    with pytest.raises(PublicationRefusal, match="binding_mismatch"):
        machine_assessment.render_arguments(spec, intent, mixed, call, checkpoint)
