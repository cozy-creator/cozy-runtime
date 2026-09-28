"""Post-phase import of quiesced output files into ordinary native trees."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import workspace_byte_outputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_byte_outputs import TREE_MIME, Blob, FileMember
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def roster(root: Path, *, tree: bool, max_bytes: int) -> tuple[bytes, list[tuple[str, Path]]]:
    files: list[tuple[str, Path]] = []
    if root.is_symlink():
        raise WorkspaceRefusal("output spool cannot contain a symbolic link")
    if tree:
        if not root.is_dir():
            raise WorkspaceRefusal("output tree has no spool directory")
        for directory, dirs, names in os.walk(root, followlinks=False):
            base = Path(directory)
            if any((base / name).is_symlink() for name in dirs):
                raise WorkspaceRefusal("output tree contains a symbolic link")
            for name in names:
                path = base / name
                files.append((path.relative_to(root).as_posix(), path))
                if len(files) > 8192:
                    raise WorkspaceRefusal("output tree exceeds its file count bound")
    else:
        files = [("payload", root)]
    entries = []
    total = 0
    for name, path in sorted(files):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as file:
            if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
                raise WorkspaceRefusal("output spool is not a regular file")
            length, digest = 0, hashlib.sha256()
            while block := file.read(min(1 << 20, max_bytes - total + 1)):
                total += len(block)
                length += len(block)
                if total > max_bytes:
                    raise WorkspaceRefusal("output content exceeds its granted byte bound")
                digest.update(block)
        entries.append(FileMember("file", name, Blob(digest.hexdigest(), length)))
    return workspace_byte_outputs.manifest_body(entries), files


def commit(
    workspace: Workspace,
    owner: str,
    request: str,
    ordinal: int,
    spec: bytes,
    output_id: str,
    root: Path,
    *,
    tree: bool,
    max_bytes: int,
    media_type: str,
) -> pb.OutputEntry:
    manifest, files = roster(root, tree=tree, max_bytes=max_bytes)
    source = workspace_byte_outputs.commit(
        workspace,
        owner,
        request,
        ordinal,
        spec,
        output_id,
        manifest,
        files,
        max_bytes,
    )
    if tree:
        digest, length = source.manifest.digest, source.manifest.length
    else:
        blob = workspace_byte_outputs.manifest_members(manifest, max_bytes)[0].blob
        digest, length = bytes.fromhex(blob.sha256), blob.length
    return pb.OutputEntry(
        output_id=output_id,
        digest=digest,
        length=length,
        mime_type=TREE_MIME if tree else media_type,
        native_tree=source,
    )


def reexport(
    workspace: Workspace,
    owner: str,
    request: str,
    ordinal: int,
    spec: bytes,
    output_id: str,
    retained: pb.NativeByteRetentionRequest,
    *,
    tree: bool,
    max_bytes: int,
    media_type: str,
) -> pb.OutputEntry:
    """An explicit return creates ordinary output custody over a live received input."""
    with workspace_byte_outputs.leased(workspace, owner, retained) as (store, _, members):
        body = store.manifest(documents.spell(retained.source.manifest.digest))["manifest"]
        if not tree and (len(members) != 1 or members[0].path != "payload"):
            raise WorkspaceRefusal("returned FileAsset has no single native payload")
    source = workspace_byte_outputs.commit(
        workspace, owner, request, ordinal, spec, output_id, body, [], max_bytes, retained=retained
    )
    if tree:
        digest, length = source.manifest.digest, source.manifest.length
    else:
        blob = members[0].blob
        digest, length = bytes.fromhex(blob.sha256), blob.length
    return pb.OutputEntry(
        output_id=output_id,
        digest=digest,
        length=length,
        mime_type=TREE_MIME if tree else media_type,
        native_tree=source,
    )


def settle_result(value: object, outputs: list[pb.OutputEntry]) -> bytes:
    """Rewrite only declared output leaves using the post phase's verified identities."""
    replacements = {entry.output_id: entry for entry in outputs if entry.HasField("native_tree")}
    found: set[str] = set()

    def replace(node: object, path: str) -> None:
        entry = replacements.get(path)
        if entry is not None:
            if not isinstance(node, dict) or "asset_ref" not in node:
                raise WorkspaceRefusal("native output no longer matches its result field")
            tree = node.get("kind") == "tree"
            if tree != (entry.mime_type == TREE_MIME):
                raise WorkspaceRefusal("native output kind changed during post processing")
            digest = documents.spell(entry.digest)
            node.update(
                asset_ref=digest,
                digest=digest,
                size_bytes=entry.native_tree.content_bytes if tree else entry.length,
            )
            if not tree:
                node["media_type"] = entry.mime_type
            found.add(path)
            return
        if isinstance(node, dict):
            for key, child in node.items():
                replace(child, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                replace(child, f"{path}.{index}" if path else str(index))

    replace(value, "")
    if found != set(replacements) - {"runtime.capture"}:
        raise WorkspaceRefusal("native output result fields are incomplete")
    return canonical_json.encode(value)
