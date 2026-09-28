"""Bind a captured job's Model arguments to exact existing native custody."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker import grants, workspace_byte_outputs
from cozy_runtime.internal.worker.workspace import HeldTree, NativeHold, Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_byte_outputs import Blob, FileMember
from cozy_runtime.internal.worker.workspace_executions import transaction
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

SCHEMA9 = """
ALTER TABLE executions ADD COLUMN publication_authorization_id TEXT NOT NULL DEFAULT '';
CREATE TABLE execution_model_holds (
 owner TEXT NOT NULL, recipient TEXT NOT NULL, path TEXT NOT NULL, retention BLOB NOT NULL,
 PRIMARY KEY(owner,recipient,path)
) STRICT;
"""


class ModelHold(msgspec.Struct, frozen=True):
    """One ``execution_model_holds`` row."""

    owner: str
    recipient: str
    path: str
    retention: bytes


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
    # One commit for the path and its reservation: each is a synchronous journal write, and
    # a child binding run 1516's five references and context paid both per input.
    with workspace.locked() as db, transaction(db):
        db.execute(
            "INSERT INTO execution_model_holds(owner,recipient,path,retention) VALUES(?,?,?,?) "
            "ON CONFLICT(owner,recipient,path) DO NOTHING",
            (owner, recipient, path, raw),
        )
        row = db.one(
            ModelHold,
            "SELECT * FROM execution_model_holds WHERE owner=? AND recipient=? AND path=?",
            (owner, recipient, path),
        )
        if row is None or row.retention != raw:
            raise WorkspaceRefusal("execution Model hold changed its exact subject")
        workspace.reserve_hold(db, owner, request, release=False, kind=kind)


def byte_retention_id(owner: str, recipient: str, path: str, source: pb.NativeByteTreeRef) -> str:
    return documents.spell(
        hashlib.sha256(
            canonical_json.encode(
                ["cozy.machine-byte-hold/1", owner, recipient, path, documents.body(source)]
            )
        ).digest()
    )


def retain_bytes(
    workspace: Workspace, owner: str, recipient: str, path: str, source: pb.NativeByteTreeRef
) -> pb.ChildByteResultGrant:
    identity = byte_retention_id(owner, recipient, path, source)
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
        rows = db.all(
            ModelHold,
            "SELECT * FROM execution_model_holds WHERE owner=? AND recipient=?",
            (owner, recipient),
        )
    for row in rows:
        request = pb.DerivedRetentionRequest.FromString(row.retention)
        with workspace.locked() as db:
            held = db.one(NativeHold, "SELECT * FROM holds WHERE id=?", (request.retention_id,))
            workspace._change_hold(
                db, owner, request, release=True, kind=held.kind if held else "derived"
            )
    from . import machine_checkpoint_inputs

    machine_checkpoint_inputs.release(workspace, owner, recipient)
    # Keep the tiny ownership record for idempotent release after a lost reply.


def file_source(
    workspace: Workspace, owner: str, parent: str, digest: str, length: int | None = None
) -> pb.NativeByteRetentionRequest:
    """Resolve a typed file only among native payload trees actually received by this parent."""
    documents.raw(digest)
    if length is not None and (type(length) is not int or not 0 <= length <= 8 << 20):
        raise WorkspaceRefusal("received metadata file exceeds its 8 MiB bound")
    with workspace.locked() as db:
        rows = db.all(
            ModelHold,
            "SELECT * FROM execution_model_holds WHERE owner=? AND recipient=? ORDER BY path",
            (owner, parent),
        )
        candidates = []
        for row in rows:
            hold = pb.DerivedRetentionRequest.FromString(row.retention)
            held = db.one(
                HeldTree,
                "SELECT h.*,b.content_bytes FROM holds h JOIN byte_outputs b ON b.owner=h.owner "
                "AND b.id=h.transaction_id WHERE h.owner=? AND h.id=? AND h.state='held' "
                "AND h.kind='tree' AND h.native_digest=? AND b.content_bytes<=? "
                "AND (? IS NULL OR b.content_bytes=?)",
                (owner, hold.retention_id, hold.tensorfs_receipt_digest, 8 << 20, length, length),
            )
            if held is not None:
                candidates.append(
                    pb.NativeByteRetentionRequest(
                        retention_id=hold.retention_id, source=held.source()
                    )
                )
    for candidate in candidates:
        with workspace_byte_outputs.leased(workspace, owner, candidate) as (_, _, members):
            if members == (
                FileMember("file", "payload", Blob(digest[7:], candidate.source.content_bytes)),
            ):
                return candidate
    raise WorkspaceRefusal("metadata file has no exact received native payload tree")


def inputs(
    workspace: Workspace,
    owner: str,
    recipient: str,
    entrypoint: str,
    declaration: Mapping[str, object],
    arguments: Mapping[str, Json],
) -> tuple[list[pb.InputBinding], list[pb.InputAccess]]:
    declared = msgspec.convert(declaration, package_interface.CallableDoc, strict=True)
    retain_result(workspace, owner, recipient, "input", arguments, declared.request)
    bindings, accesses = [], []
    for model in declared.models:
        prefix = f"{entrypoint}.models."
        parameter = model.path.removeprefix(prefix)
        if not model.path.startswith(prefix) or not parameter.isidentifier():
            raise WorkspaceRefusal("captured Model parameter has no exact declared name")
        supplied = arguments.get(parameter)
        if not isinstance(supplied, dict):
            raise WorkspaceRefusal("captured Model input requires a retained artifact")
        artifact = msgspec.convert(supplied, type=ModelArtifact, strict=True)
        retention, held = retain(workspace, owner, recipient, "input/" + parameter, artifact)
        manifest = artifact.manifest
        if (documents.spell(held.manifest.digest), held.manifest.length) != (
            manifest.digest,
            manifest.length,
        ):
            raise WorkspaceRefusal("native Model custody differs from its argument")
        with workspace.held_model(owner, retention, held.manifest):
            bindings.append(
                pb.InputBinding(
                    input_id=grants.MODEL_PREFIX + parameter,
                    digest=manifest.digest,
                    length=manifest.length,
                    kind_mime=grants.MODEL_MIME,
                )
            )
            accesses.append(
                pb.InputAccess(
                    input_id=grants.MODEL_PREFIX + parameter, url="model://" + manifest.digest
                )
            )
    return bindings, accesses


def _path(parent: str, key: str | int) -> str:
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def artifacts(value: Json, schema: Json, path: str = "") -> Iterator[tuple[str, ModelArtifact]]:
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
    value: Json,
    schema: Json,
) -> None:
    result_records(workspace, owner, recipient, value, schema, prefix=path)


def result_records(
    workspace: Workspace,
    owner: str,
    recipient: str,
    value: Json,
    schema: Json,
    *,
    prefix: str = "result",
) -> list[tuple[str, bytes, pb.DerivedRetentionRequest]]:
    """Freeze exact, bounded Model custody metadata before the terminal is hashed."""
    selected = list(artifacts(value, schema))
    if len(selected) > weights_limits.MAX_RETAINED_MODEL_RESULTS:
        raise WorkspaceRefusal("native Model result inventory exceeds 32 entries")
    encoded = [
        (path, canonical_json.encode(msgspec.to_builtins(artifact)), artifact)
        for path, artifact in selected
    ]
    if any(
        len(path.encode()) > weights_limits.MAX_MODEL_RESULT_POINTER_BYTES
        or len(raw) > weights_limits.MAX_RETAINED_MODEL_ARTIFACT_BYTES
        for path, raw, _ in encoded
    ):
        raise WorkspaceRefusal("native Model result metadata exceeds its inline bounds")
    records = []
    for path, raw, artifact in encoded:
        hold, observed = retain(workspace, owner, recipient, prefix + path, artifact)
        if (documents.spell(observed.manifest.digest), observed.manifest.length) != (
            artifact.manifest.digest,
            artifact.manifest.length,
        ):
            raise WorkspaceRefusal("native result retention differs from its exact artifact")
        with workspace.held_model(owner, hold, observed.manifest):
            pass
        records.append((path, raw, hold))
    return records
