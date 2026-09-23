"""Catalog inputs use native checkpoint roots without inventing a producer or receipt."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import tensorfs

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import grants, machine_checkpoint_inputs, machine_model_inputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import Executions
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from test_machine_execution import ack, complete
from test_machine_model_inputs import model_offer
from test_model_runtime_closure import _snapshot


def catalog(tmp_path: Path) -> tuple[tensorfs.Store, Workspace, pb.AttemptOffer]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root, manifest, length, store = _snapshot(tmp_path, include_asset=True, checkpoint_only=True)
    store.replace_local(None, "catalog", "sha256:" + "21" * 32, manifest, length)
    offer = model_offer(manifest, length)
    offer.grant.inputs[0].catalog_model.repository = "local/catalog"
    return store, Workspace(root), offer


def input_row(workspace: Workspace, recipient: str = "root") -> dict[str, object]:
    with workspace.locked() as db:
        row = db.execute(
            "SELECT * FROM execution_checkpoint_inputs WHERE recipient=?", (recipient,)
        ).fetchone()
        assert row is not None
        return dict(row)


def test_catalog_root_survives_repository_removal_restart_and_collection(tmp_path: Path) -> None:
    store, workspace, offer = catalog(tmp_path)
    machine_model_inputs.retain(workspace, "owner", offer, {"source"})
    original = input_row(workspace)
    assert original["input_id"] == "model:source" and original["state"] == "held"
    native = store.checkpoint_root(str(original["native_owner"]))
    assert native is not None and native["repository"] == "local/catalog" and not native["released"]
    with workspace.locked() as db:
        for table in ("weights", "native_calls", "holds", "execution_model_holds"):
            assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    executions = Executions(workspace)
    receipt = executions.submit(
        "owner",
        "catalog-submit",
        b"x" * 32,
        offer,
        expected_execution_workspace_id=executions.workspace_id,
    )
    store.remove_local(store.repo_get("local", "catalog"), "catalog")
    tensorfs.gc(store.root)
    restored = Workspace(Path(store.root))
    machine_model_inputs.retain(restored, "owner", offer, {"source"})
    assert input_row(restored)["native_owner"] == original["native_owner"]
    assert store.manifest(native["manifest_digest"])
    recovered = Executions(restored)
    outcome = complete(recovered, "root", 14)
    recovered.acknowledge_collection("owner", ack(outcome))
    assert input_row(restored)["state"] == "released"
    released = store.checkpoint_root(str(original["native_owner"]))
    assert released is not None and released["released"]
    # A lost submission reply can be replayed after collection without resurrecting custody.
    machine_model_inputs.retain(restored, "owner", offer, {"source"})
    assert (
        recovered.submit(
            "owner",
            "catalog-submit",
            b"x" * 32,
            offer,
            expected_execution_workspace_id=receipt.execution_workspace_id,
        )
        == receipt
    )
    tensorfs.gc(store.root)
    with pytest.raises(tensorfs.errors.Refusal):
        store.manifest(native["manifest_digest"])


def test_bare_or_unowned_catalog_subject_never_becomes_a_derived_artifact(tmp_path: Path) -> None:
    store, workspace, offer = catalog(tmp_path)
    offer.grant.inputs[0].ClearField("catalog_model")
    with pytest.raises(WorkspaceRefusal, match="derived custody"):
        machine_model_inputs.retain(workspace, "owner", offer, {"source"})
    offer.grant.inputs[0].catalog_model.repository = "other/absent"
    with pytest.raises(WorkspaceRefusal, match="REPOSITORY_ABSENT"):
        machine_model_inputs.retain(workspace, "owner", offer, {"source"})
    offer.grant.inputs[0].catalog_model.repository = "local/catalog"
    store.remove_local(store.repo_get("local", "catalog"), "catalog")
    # Header/asset bytes are still in CAS, but there is no source authority left.
    with pytest.raises(WorkspaceRefusal, match="REPOSITORY_ABSENT"):
        machine_model_inputs.retain(workspace, "owner", offer, {"source"})
    with workspace.locked() as db:
        assert not db.execute("SELECT 1 FROM execution_checkpoint_inputs").fetchone()


def test_catalog_grant_is_closed_to_model_inputs_and_native_repository_names(
    tmp_path: Path,
) -> None:
    _, workspace, original = catalog(tmp_path)
    for source in (
        "",
        "local",
        "local//catalog",
        "../catalog",
        "local/catalog?x",
        "x" * 129 + "/a",
    ):
        offer = pb.AttemptOffer.FromString(original.SerializeToString())
        offer.grant.inputs[0].catalog_model.repository = source
        with pytest.raises(grants.GrantRefusal, match="grant_catalog_access"):
            machine_model_inputs.retain(workspace, "owner", offer, {"source"})
    payload = pb.AttemptOffer.FromString(original.SerializeToString())
    payload.grant.inputs[1].catalog_model.repository = "local/catalog"
    with pytest.raises(grants.GrantRefusal, match="grant_catalog_access"):
        machine_model_inputs.retain(workspace, "owner", payload, {"source"})
    mixed = pb.AttemptOffer.FromString(original.SerializeToString())
    mixed.grant.inputs[0].native_tree.retention_id = "sha256:" + "11" * 32
    with pytest.raises(grants.GrantRefusal, match="grant_catalog_access"):
        machine_model_inputs.retain(workspace, "owner", mixed, {"source"})


def test_all_catalog_inputs_preflight_before_creating_any_recipient(tmp_path: Path) -> None:
    _, workspace, offer = catalog(tmp_path)
    spec = documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
    spec["inputs"].append(
        dict(spec["inputs"][0], input_id="model:z_unknown", digest="sha256:" + "ff" * 32)
    )
    offer.invocation_spec_canonical_bytes = canonical_json.encode(spec)
    offer.invocation_spec_digest = documents.digest_of(offer.invocation_spec_canonical_bytes)
    offer.grant.invocation_spec_digest = offer.invocation_spec_digest
    offer.grant.inputs.add(
        input_id="model:z_unknown",
        url="model://sha256:" + "ff" * 32,
        catalog_model=pb.CatalogModelSource(repository="local/catalog"),
    )
    with pytest.raises(WorkspaceRefusal, match="ROOT_ABSENT"):
        machine_model_inputs.retain(workspace, "owner", offer, {"source", "z_unknown"})
    with workspace.locked() as db:
        assert not db.execute("SELECT 1 FROM execution_checkpoint_inputs").fetchone()
        assert not db.execute("SELECT 1 FROM execution_model_holds").fetchone()


def test_failed_unaccepted_intake_releases_only_its_pin_and_can_retry(tmp_path: Path) -> None:
    store, workspace, offer = catalog(tmp_path)
    other = pb.AttemptOffer.FromString(offer.SerializeToString())
    other.request_id = "other"
    machine_model_inputs.retain(workspace, "owner", other, {"source"})
    other_pin = input_row(workspace, "other")
    with (
        pytest.raises(WorkspaceRefusal, match="later input"),
        machine_checkpoint_inputs.admission(workspace, "owner", offer.request_id),
    ):
        machine_model_inputs.retain(workspace, "owner", offer, {"source"})
        raise WorkspaceRefusal("later input refused before submission")
    previous = input_row(workspace)
    assert previous["state"] == "released"
    assert input_row(workspace, "other") == other_pin
    native_other = store.checkpoint_root(str(other_pin["native_owner"]))
    assert native_other is not None and not native_other["released"]
    machine_model_inputs.retain(workspace, "owner", offer, {"source"})
    current = input_row(workspace)
    assert current["state"] == "held" and current["generation"] == 2
    assert current["native_owner"] != previous["native_owner"]
    native = store.checkpoint_root(str(previous["native_owner"]))
    assert native is not None and native["released"]
    executions = Executions(workspace)
    executions.submit(
        "owner",
        "catalog-retry",
        b"x" * 32,
        offer,
        expected_execution_workspace_id=executions.workspace_id,
    )
    with (
        pytest.raises(WorkspaceRefusal, match="already accepted"),
        machine_checkpoint_inputs.admission(workspace, "owner", offer.request_id),
    ):
        raise WorkspaceRefusal("a later observer error after work was already accepted")
    assert input_row(workspace)["state"] == "held"


@pytest.mark.parametrize("index", range(4))
def test_abort_and_accept_have_one_native_custody_outcome(tmp_path: Path, index: int) -> None:
    store, workspace, offer = catalog(tmp_path / str(index))
    machine_model_inputs.retain(workspace, "owner", offer, {"source"})
    barrier = Barrier(3)

    def submit() -> bool:
        barrier.wait()
        try:
            executions = Executions(workspace)
            executions.submit(
                "owner",
                "race",
                b"x" * 32,
                offer,
                expected_execution_workspace_id=executions.workspace_id,
            )
            return True
        except WorkspaceRefusal:
            return False

    def abort() -> None:
        barrier.wait()
        machine_checkpoint_inputs.release(
            workspace, "owner", offer.request_id, unaccepted_only=True
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = pool.submit(submit)
        released = pool.submit(abort)
        barrier.wait()
        success = accepted.result()
        released.result()
    row = input_row(workspace)
    assert row["state"] == ("held" if success else "released")
    native = store.checkpoint_root(str(row["native_owner"]))
    assert native is not None and native["released"] is not success
