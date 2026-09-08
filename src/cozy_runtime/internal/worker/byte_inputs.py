"""Materialize only independently retained native byte trees into an attempt's spool."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from cozy_runtime.internal.worker import workspace_byte_outputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

TREE_MIME = "application/vnd.cozy.tree-manifest"
MAX_BYTES = 256 << 20


def copy_retained(
    workspace: Workspace,
    owner: str,
    grant: pb.NativeByteRetentionRequest,
    destination: Path,
    *,
    max_bytes: int = MAX_BYTES,
    expected_blob: tuple[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Stream through the native read lease; publish no paths before all bytes verify."""
    if grant.source.content_bytes > max_bytes:
        raise WorkspaceRefusal("native input exceeds its granted byte bound")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".native-", dir=destination.parent))
    try:
        with workspace_byte_outputs.leased(workspace, owner, grant) as (_store, lease, members):
            if expected_blob is not None:
                blob = file_member(grant, members)
                if expected_blob != ("sha256:" + blob["sha256"], blob["length"]):
                    raise WorkspaceRefusal("native FileAsset differs from its input identity")
            for member in members:
                relative = PurePosixPath(member["path"])
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or any(part in ("", ".", "..") for part in relative.parts)
                    or str(relative) != member["path"]
                ):
                    raise WorkspaceRefusal("native input manifest has an invalid member path")
                target = temporary.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                blob = member["blob"]
                length = blob["length"]
                digest = hashlib.sha256()
                offset = 0
                with target.open("xb") as output:
                    while offset < length:
                        amount = min(1 << 20, length - offset)
                        buffer = bytearray(amount)
                        lease.read_into("sha256:" + blob["sha256"], length, offset, amount, buffer)
                        output.write(buffer)
                        digest.update(buffer)
                        offset += amount
                if digest.hexdigest() != blob["sha256"]:
                    raise WorkspaceRefusal("native input member failed byte verification")
                target.chmod(0o444)
            # No mutable checkout or publisher path crosses into the consumer.
            for folder in sorted(temporary.rglob("*"), reverse=True):
                if folder.is_dir():
                    folder.chmod(0o555)
            temporary.chmod(0o555)
            if destination.exists():
                raise WorkspaceRefusal("native input destination already exists")
            os.rename(temporary, destination)
            return members
    finally:
        if temporary.exists():
            for folder in [temporary, *temporary.rglob("*")]:
                if folder.is_dir():
                    folder.chmod(0o755)
            shutil.rmtree(temporary)


def file_member(
    grant: pb.NativeByteRetentionRequest, members: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(members) != 1 or members[0]["path"] != "payload":
        raise WorkspaceRefusal("native FileAsset must retain exactly one payload member")
    blob: dict[str, Any] = members[0]["blob"]
    if blob["length"] != grant.source.content_bytes:
        raise WorkspaceRefusal("native FileAsset size differs from its retained payload")
    return blob


def validate_source(source: pb.NativeByteTreeRef, retention_id: str) -> None:
    if (
        not retention_id
        or len(retention_id.encode()) > 256
        or not source.producer_root_id
        or len(source.producer_root_id.encode()) > 256
        or len(source.receipt_digest) != 32
        or len(source.manifest.digest) != 32
        or not 0 < source.manifest.length <= 1 << 20
    ):
        raise WorkspaceRefusal("native input has an incomplete retained identity")
    # Digest spelling is deliberately independent of the producer's identifier.
    documents.spell(source.manifest.digest)


def remove_checkout(root: Path | None) -> None:
    """Remove only a checkout owned by this attempt, including its readonly folders."""
    if root is None or not root.exists():
        return
    for directory in [root, *root.rglob("*")]:
        if directory.is_dir():
            directory.chmod(0o755)
    shutil.rmtree(root)


def chunks(lease: Any, digest: str, length: int) -> Iterator[bytes]:
    offset = 0
    while offset < length:
        amount = min(1 << 20, length - offset)
        buffer = bytearray(amount)
        lease.read_into(digest, length, offset, amount, buffer)
        yield bytes(buffer)
        offset += amount
