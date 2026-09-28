"""Admission for package-private callables, using Runtime-owned child journal facts."""

from __future__ import annotations

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def require_child(
    workspace: Workspace | None,
    owner: str,
    offered: pb.AttemptOffer,
    installation_id: str,
    entrypoint: str,
    *,
    machine_owned: bool,
) -> None:
    """A public request cannot gain child authority by supplying a known request ID."""
    detail = "internal_callable: only managed children of the same installation may invoke this"
    if not machine_owned or workspace is None:
        raise WorkspaceRefusal(detail)
    with workspace.locked() as db:
        row = db.execute(
            "SELECT c.prepared,p.offer AS parent_offer,e.offer AS child_offer "
            "FROM execution_calls c JOIN executions p "
            "ON p.owner=c.owner AND p.request=c.parent_request "
            "JOIN executions e ON e.owner=c.owner AND e.request=c.child_request "
            "WHERE c.owner=? AND c.child_request=? AND c.safe_code=''",
            (owner, offered.request_id),
        ).fetchone()
    if row is None or not row["prepared"]:
        raise WorkspaceRefusal(detail)
    plan = canonical_json.decode(row["prepared"])
    parent = pb.AttemptOffer.FromString(row["parent_offer"])
    child = pb.AttemptOffer.FromString(row["child_offer"])
    parent_spec = documents.parse(parent.invocation_spec_canonical_bytes, pb.InvocationSpec)
    if (
        not installation_id
        or plan.get("installation_id") != installation_id
        or plan.get("entrypoint") != entrypoint
        or parent_spec.job.installation_id != installation_id
        or child.invocation_spec_digest != offered.invocation_spec_digest
        or child.invocation_spec_canonical_bytes != offered.invocation_spec_canonical_bytes
    ):
        raise WorkspaceRefusal(detail)
