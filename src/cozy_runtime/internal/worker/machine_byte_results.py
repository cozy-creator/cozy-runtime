"""Declared native output fields acquire independent parent custody before child ACK."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from cozy_runtime.author._call_results import result_at
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from . import machine_models
from .workspace import Workspace, WorkspaceRefusal


def paths(schema: Any, path: str = "", bound: int = 256 << 20) -> Iterator[tuple[str, int]]:
    if not isinstance(schema, dict):
        return
    if schema.get("asset") or schema.get("input") == "tree":
        if not path:
            raise WorkspaceRefusal("native byte result needs a named result field")
        yield path, bound
    elif "fields" in schema:
        for field in schema["fields"]:
            size = field.get("asset_bound", {}).get("max_bytes", bound)
            yield from paths(
                field["type"],
                path + "." + field["name"] if path else field["name"],
                min(bound, size),
            )
    elif "tuple" in schema:
        for index, member in enumerate(schema["tuple"]):
            yield from paths(member, path + "." + str(index) if path else str(index), bound)
    elif "union" in schema or "list" in schema or "map" in schema:
        # These shapes need output slots only known after executing the program. The
        # existing fixed-slot grant protocol cannot promise their cardinality.
        def has_assets(node: Any) -> bool:
            if isinstance(node, dict):
                return (
                    bool(node.get("asset"))
                    or node.get("input") == "tree"
                    or any(has_assets(v) for v in node.values())
                )
            return isinstance(node, list) and any(has_assets(v) for v in node)

        if has_assets(schema):
            raise WorkspaceRefusal("dynamic native result fields require fixed output slots")


def retain(
    workspace: Workspace,
    owner: str,
    parent: str,
    index: int,
    value: Any,
    schema: Any,
    outputs: Sequence[pb.OutputEntry],
    *,
    capture: bool = False,
) -> None:
    """Hold each typed output's native bytes for the parent. The entries are the decoded
    protobuf, so an empty output (proto3 `content_bytes` 0, omitted from its document) reads
    as 0 rather than a missing key."""
    by_id = {entry.output_id: entry for entry in outputs}
    fields = list(paths(schema))
    if capture:
        fields.append(("runtime.capture", 64 << 20))
    for path, _ in fields:
        entry = by_id.get(path)
        node = result_at(value, path) if path != "runtime.capture" else None
        if (
            entry is None
            or not entry.HasField("native_tree")
            or (
                path != "runtime.capture"
                and (
                    not isinstance(node, dict)
                    or node.get("digest") != documents.spell(entry.digest)
                )
            )
        ):
            raise WorkspaceRefusal("typed child output lacks its exact native result entry")
        if node is not None and node.get("size_bytes") != entry.native_tree.content_bytes:
            raise WorkspaceRefusal("typed child output length differs from native custody")
        machine_models.retain_bytes(
            workspace, owner, parent, f"call.{index}.output/{path}", entry.native_tree
        )


def grants(
    workspace: Workspace,
    owner: str,
    parent: str,
    index: int,
    schema: Any,
    *,
    capture: bool = False,
) -> list[pb.ChildByteResultGrant]:
    result = []
    with workspace.locked() as db:
        fields = list(paths(schema))
        if capture:
            fields.append(("runtime.capture", 64 << 20))
        for path, _ in fields:
            registered = db.execute(
                "SELECT retention FROM execution_model_holds "
                "WHERE owner=? AND recipient=? AND path=?",
                (owner, parent, f"call.{index}.output/{path}"),
            ).fetchone()
            if registered is None:
                raise WorkspaceRefusal("typed child output has no registered parent custody")
            hold = pb.DerivedRetentionRequest.FromString(registered["retention"])
            row = db.execute(
                "SELECT h.*,b.content_bytes FROM holds h JOIN byte_outputs b "
                "ON b.id=h.transaction_id AND b.owner=h.owner "
                "WHERE h.id=? AND h.owner=? AND h.kind='tree' AND h.state='held'",
                (hold.retention_id, owner),
            ).fetchone()
            if row is None:
                raise WorkspaceRefusal("typed child output parent custody is no longer held")
            result.append(
                pb.ChildByteResultGrant(
                    output_id=path,
                    retention_id=hold.retention_id,
                    source=pb.NativeByteTreeRef(
                        producer_root_id=row["transaction_id"],
                        receipt_digest=row["native_digest"],
                        manifest=pb.Ref(digest=row["manifest"], length=row["manifest_length"]),
                        content_bytes=row["content_bytes"],
                    ),
                )
            )
    return result
