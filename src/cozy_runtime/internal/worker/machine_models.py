"""Bind a captured job's Model arguments to exact existing native custody."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.internal.worker import grants
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

SCHEMA9 = """
ALTER TABLE executions ADD COLUMN publication_authorization_id TEXT NOT NULL DEFAULT '';
CREATE TABLE execution_model_holds (
 owner TEXT NOT NULL, recipient TEXT NOT NULL, path TEXT NOT NULL, retention BLOB NOT NULL,
 PRIMARY KEY(owner,recipient,path)
) STRICT;
"""


def retain(
    workspace: Workspace,
    owner: str,
    recipient: str,
    path: str,
    artifact: ModelArtifact,
) -> tuple[pb.DerivedRetentionRequest, pb.DerivedRetentionResult]:
    """Journal semantic ownership before crossing the existing native hold boundary."""
    from .machine_artifacts import model_retention

    request = model_retention(workspace, owner, recipient, path, artifact)
    register(workspace, owner, recipient, path, request)
    return request, workspace.retain(owner, request)


def register(
    workspace: Workspace,
    owner: str,
    recipient: str,
    path: str,
    request: pb.DerivedRetentionRequest,
    *,
    kind: str = "derived",
) -> None:
    raw = request.SerializeToString(deterministic=True)
    with workspace.locked() as db:
        db.execute(
            "INSERT INTO execution_model_holds(owner,recipient,path,retention) VALUES(?,?,?,?) "
            "ON CONFLICT(owner,recipient,path) DO NOTHING",
            (owner, recipient, path, raw),
        )
        row = db.execute(
            "SELECT retention FROM execution_model_holds WHERE owner=? AND recipient=? AND path=?",
            (owner, recipient, path),
        ).fetchone()
        if row["retention"] != raw:
            raise WorkspaceRefusal("execution Model hold changed its exact subject")
        workspace.reserve_hold(db, owner, request, release=False, kind=kind)


def retain_bytes(
    workspace: Workspace, owner: str, recipient: str, path: str, source: pb.NativeByteTreeRef
) -> pb.ChildByteResultGrant:
    from . import workspace_byte_outputs

    identity = documents.spell(
        hashlib.sha256(
            canonical_json.encode(
                ["cozy.machine-byte-hold/1", owner, recipient, path, documents.body(source)]
            )
        ).digest()
    )
    request = pb.DerivedRetentionRequest(
        weights_transaction_id=source.producer_root_id,
        tensorfs_receipt_digest=source.receipt_digest,
        retention_id=identity,
    )
    register(workspace, owner, recipient, path, request, kind="tree")
    workspace_byte_outputs.change_hold(
        workspace,
        owner,
        pb.NativeByteRetentionRequest(source=source, retention_id=identity),
        release=False,
    )
    return pb.ChildByteResultGrant(source=source, retention_id=identity)


def release(workspace: Workspace, owner: str, recipient: str) -> None:
    """Collection or explicit abandonment releases this recipient's own native holds."""
    with workspace.locked() as db:
        rows = db.execute(
            "SELECT retention FROM execution_model_holds WHERE owner=? AND recipient=?",
            (owner, recipient),
        ).fetchall()
    for row in rows:
        request = pb.DerivedRetentionRequest.FromString(row["retention"])
        with workspace.locked() as db:
            held = db.execute(
                "SELECT kind FROM holds WHERE id=?", (request.retention_id,)
            ).fetchone()
            workspace._change_hold(
                db, owner, request, release=True, kind=held["kind"] if held else "derived"
            )
    # Keep the tiny ownership record for idempotent release after a lost reply.


def file_source(
    workspace: Workspace, owner: str, parent: str, digest: str, length: int | None = None
) -> pb.NativeByteRetentionRequest:
    """Resolve a typed file only among native payload trees actually received by this parent."""
    from . import workspace_byte_outputs

    documents.raw(digest)
    if length is not None and (type(length) is not int or not 0 <= length <= 8 << 20):
        raise WorkspaceRefusal("received metadata file exceeds its 8 MiB bound")
    with workspace.locked() as db:
        rows = db.execute(
            "SELECT retention FROM execution_model_holds "
            "WHERE owner=? AND recipient=? ORDER BY path",
            (owner, parent),
        ).fetchall()
        candidates = []
        for row in rows:
            hold = pb.DerivedRetentionRequest.FromString(row["retention"])
            held = db.execute(
                "SELECT h.*,b.content_bytes FROM holds h JOIN byte_outputs b ON b.owner=h.owner "
                "AND b.id=h.transaction_id WHERE h.owner=? AND h.id=? AND h.state='held' "
                "AND h.kind='tree' AND h.native_digest=? AND b.content_bytes<=? "
                "AND (? IS NULL OR b.content_bytes=?)",
                (owner, hold.retention_id, hold.tensorfs_receipt_digest, 8 << 20, length, length),
            ).fetchone()
            if held is not None:
                candidates.append(
                    pb.NativeByteRetentionRequest(
                        retention_id=hold.retention_id,
                        source=pb.NativeByteTreeRef(
                            producer_root_id=held["transaction_id"],
                            receipt_digest=held["native_digest"],
                            manifest=pb.Ref(
                                digest=held["manifest"], length=held["manifest_length"]
                            ),
                            content_bytes=held["content_bytes"],
                        ),
                    )
                )
    for candidate in candidates:
        with workspace_byte_outputs.leased(workspace, owner, candidate) as (_, _, members):
            if members == [
                {
                    "kind": "file",
                    "path": "payload",
                    "blob": {"sha256": digest[7:], "length": candidate.source.content_bytes},
                }
            ]:
                return candidate
    raise WorkspaceRefusal("metadata file has no exact received native payload tree")


def inputs(
    workspace: Workspace,
    owner: str,
    recipient: str,
    entrypoint: str,
    declaration: dict[str, Any],
    arguments: dict[str, Any],
) -> tuple[list[pb.InputBinding], list[pb.InputAccess]]:
    retain_result(workspace, owner, recipient, "input", arguments, declaration["request"])
    if not declaration.get("models"):
        return [], []
    bindings, accesses = [], []
    for model in declaration.get("models", []):
        prefix = f"{entrypoint}.models."
        parameter = model["path"].removeprefix(prefix)
        if not model["path"].startswith(prefix) or not parameter.isidentifier():
            raise WorkspaceRefusal("captured Model parameter has no exact declared name")
        artifact = arguments.get(parameter)
        if not isinstance(artifact, dict):
            raise WorkspaceRefusal("captured Model input requires a retained artifact")
        retention, held = retain(
            workspace,
            owner,
            recipient,
            "input/" + parameter,
            msgspec.convert(artifact, type=ModelArtifact, strict=True),
        )
        manifest = artifact["manifest"]
        if documents.body(held.manifest) != manifest:
            raise WorkspaceRefusal("native Model custody differs from its argument")
        with workspace.held_model(owner, retention, held.manifest):
            bindings.append(
                pb.InputBinding(
                    input_id=grants.MODEL_PREFIX + parameter,
                    digest=manifest["digest"],
                    length=manifest["length"],
                    kind_mime=grants.MODEL_MIME,
                )
            )
            accesses.append(
                pb.InputAccess(
                    input_id=grants.MODEL_PREFIX + parameter,
                    url="model://" + manifest["digest"],
                )
            )
    return bindings, accesses


def _path(parent: str, key: str | int) -> str:
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def artifacts(value: Any, schema: Any, path: str = "") -> Iterator[tuple[str, ModelArtifact]]:
    """Select only schema-declared Models, preserving arbitrary user JSON unchanged."""
    if not isinstance(schema, dict):
        return
    if schema.get("input") == "model":
        yield path, msgspec.convert(value, type=ModelArtifact, strict=True)
    elif "fields" in schema and isinstance(value, dict):
        for field in schema["fields"]:
            if field["name"] in value:
                yield from artifacts(
                    value[field["name"]],
                    field["type"],
                    _path(path, field["name"]),
                )
    elif "list" in schema and isinstance(value, list):
        for index, item in enumerate(value):
            yield from artifacts(item, schema["list"], _path(path, index))
    elif "tuple" in schema and isinstance(value, list):
        for index, (item, member) in enumerate(zip(value, schema["tuple"], strict=True)):
            yield from artifacts(item, member, _path(path, index))
    elif "map" in schema and isinstance(value, dict):
        for key, item in value.items():
            # JSON keys are strings; only the declared value can contain a Model.
            yield from artifacts(item, schema["map"]["value"], _path(path, key))
    elif "union" in schema:
        branches = schema["union"]
        if value is None and "null" in branches:
            return
        selected = [branch for branch in branches if branch != "null"]
        if tag := schema.get("tag_field"):
            selected = [
                branch
                for branch in selected
                if isinstance(branch, dict)
                and isinstance(value, dict)
                and branch.get("tag") == value.get(tag)
            ]
        if len(selected) == 1:
            yield from artifacts(value, selected[0], path)
        elif any(isinstance(branch, dict) for branch in selected):
            raise WorkspaceRefusal("ambiguous structured result cannot grant native Models")


def retain_result(
    workspace: Workspace,
    owner: str,
    recipient: str,
    path: str,
    value: Any,
    schema: Any,
) -> None:
    for field, artifact in artifacts(value, schema):
        hold, observed = retain(workspace, owner, recipient, path + field, artifact)
        if (documents.spell(observed.manifest.digest), observed.manifest.length) != (
            artifact.manifest.digest,
            artifact.manifest.length,
        ):
            raise WorkspaceRefusal("returned Model custody differs from the typed result")
        with workspace.held_model(owner, hold, observed.manifest):
            pass
