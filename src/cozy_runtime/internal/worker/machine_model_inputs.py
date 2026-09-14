"""Retain exact root Model inputs through independent derived or catalog custody."""

from __future__ import annotations

from typing import Any

import msgspec

from cozy_runtime.author._artifacts import ModelArtifact, ObjectRef
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import grants, machine_checkpoint_inputs, machine_models
from .workspace import Workspace, WorkspaceRefusal


def retain(workspace: Workspace, owner: str, offer: pb.AttemptOffer, declared: set[str]) -> None:
    spec = documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
    if (
        not declared
        and not any(entry.HasField("catalog_model") for entry in offer.grant.inputs)
        and not any(
            item["input_id"].startswith(grants.MODEL_PREFIX) for item in spec.get("inputs", [])
        )
    ):
        return
    selected = grants.model_inputs(grants.bind(spec, offer.grant, offer.invocation_spec_digest))
    if set(selected) != declared:
        raise WorkspaceRefusal("root Model inputs differ from the captured job declaration")
    if machine_checkpoint_inputs.already_accepted(workspace, owner, offer):
        return
    catalog = {
        parameter: entry for parameter, entry in selected.items() if entry.catalog_repository
    }
    # Reject incomplete multi-model declarations before minting any recipient hold.
    sources = [
        (parameter, _owned(workspace, owner, documents.spell(entry.digest), entry.length))
        for parameter, entry in sorted(selected.items())
        if not entry.catalog_repository
    ]
    machine_checkpoint_inputs.preflight(workspace, owner, offer, catalog)
    for entry in catalog.values():
        machine_checkpoint_inputs.retain(workspace, owner, offer, entry)
    for parameter, artifact in sources:
        request, held = machine_models.retain(
            workspace, owner, offer.request_id, "input/" + parameter, artifact
        )
        with workspace.held_model(owner, request, held.manifest):
            pass


def _owned(workspace: Workspace, owner: str, manifest: str, length: int) -> ModelArtifact:
    """A digest grants no bytes; resolve provenance only through this owner's live custody."""
    with workspace.locked() as db:
        row = db.execute(
            "SELECT w.request,w.slot,w.native_digest FROM weights w "
            "WHERE w.owner=? AND w.manifest=? AND w.manifest_length=? "
            "AND (w.state='receipt' OR EXISTS(SELECT 1 FROM holds h WHERE h.owner=w.owner "
            "AND h.transaction_id=w.id AND h.native_digest=w.native_digest "
            "AND h.kind='derived' AND h.state='held')) ORDER BY w.id LIMIT 1",
            (owner, documents.raw(manifest), length),
        ).fetchone()
        if row is not None:
            return ModelArtifact(
                str(row["request"]),
                str(row["slot"]),
                ObjectRef(manifest, length),
                documents.spell(bytes(row["native_digest"])),
            )
        rows = db.execute(
            "SELECT n.result FROM native_calls n JOIN holds h "
            "ON h.owner=n.owner AND h.transaction_id=n.native_owner "
            "WHERE n.owner=? AND n.operation='convert_cozytensors' "
            "AND n.state IN ('complete','releasing','released') "
            "AND h.state='held' AND h.kind='derived' AND h.manifest=? "
            "AND h.manifest_length=? ORDER BY n.native_owner,h.id",
            (owner, documents.raw(manifest), length),
        ).fetchall()
        for native in rows:
            value: Any = msgspec.json.decode(bytes(native["result"]))
            artifact = msgspec.convert(value, type=ModelArtifact, strict=True)
            if artifact.manifest == ObjectRef(manifest, length):
                return artifact
    raise WorkspaceRefusal("root Model input has no exact retained derived custody for this owner")
