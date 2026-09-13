"""Bind a captured job's Model arguments to exact existing native custody."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import msgspec

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
    return request, workspace.retain(owner, request)


def release(workspace: Workspace, owner: str, recipient: str) -> None:
    """Collection or explicit abandonment releases this recipient's own native holds."""
    with workspace.locked() as db:
        rows = db.execute(
            "SELECT path,retention FROM execution_model_holds WHERE owner=? AND recipient=?",
            (owner, recipient),
        ).fetchall()
    for row in rows:
        workspace.retain(
            owner, pb.DerivedRetentionRequest.FromString(row["retention"]), release=True
        )
    # Keep the tiny ownership record for idempotent release after a lost reply.


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
