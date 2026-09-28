"""Materialize only independently retained native byte trees into an attempt's spool."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path, PurePosixPath

from cozy_runtime.internal import fill
from cozy_runtime.internal.worker import workspace_byte_outputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_byte_outputs import Blob, FileMember
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

TREE_MIME = workspace_byte_outputs.TREE_MIME
MAX_BYTES = 256 << 20


def copy_retained(
    workspace: Workspace,
    owner: str,
    grant: pb.NativeByteRetentionRequest,
    destination: Path,
    *,
    max_bytes: int = MAX_BYTES,
    expected_blob: tuple[str, int] | None = None,
) -> tuple[FileMember, ...]:
    """Stream through the native read lease; publish no paths before all bytes verify."""
    if grant.source.content_bytes > max_bytes:
        raise WorkspaceRefusal("native input exceeds its granted byte bound")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".native-", dir=destination.parent))
    try:
        with workspace_byte_outputs.leased(workspace, owner, grant) as (_store, lease, members):
            if expected_blob is not None:
                blob = file_member(grant, members)
                if expected_blob != ("sha256:" + blob.sha256, blob.length):
                    raise WorkspaceRefusal("native FileAsset differs from its input identity")
            for member in members:
                relative = PurePosixPath(member.path)
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or any(part in ("", ".", "..") for part in relative.parts)
                    or str(relative) != member.path
                ):
                    raise WorkspaceRefusal("native input manifest has an invalid member path")
                target = temporary.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                blob = member.blob
                length = blob.length
                digest = hashlib.sha256()
                offset = 0
                with target.open("xb") as output:
                    while offset < length:
                        amount = min(1 << 20, length - offset)
                        buffer = bytearray(amount)
                        lease.read_into("sha256:" + blob.sha256, length, offset, amount, buffer)
                        output.write(buffer)
                        digest.update(buffer)
                        offset += amount
                if digest.hexdigest() != blob.sha256:
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
        remove_checkout(temporary)


def file_member(grant: pb.NativeByteRetentionRequest, members: Sequence[FileMember]) -> Blob:
    if len(members) != 1 or members[0].path != "payload":
        raise WorkspaceRefusal("native FileAsset must retain exactly one payload member")
    blob = members[0].blob
    if blob.length != grant.source.content_bytes:
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


def remove_checkout(root: Path | None, *, keep_root: bool = False) -> int:
    """Reclaim only this owned projection; links never lead to their original storage."""
    if root is None:
        return 0
    if root.is_symlink():
        root.unlink()
        return 1
    if not root.exists():
        return 0
    if not root.is_dir():
        root.unlink()
        return 1
    removed = 0
    for directory, dirs, files in os.walk(root, topdown=True, followlinks=False):
        folder = Path(directory)
        folder.chmod(0o700)
        for name in files:
            (folder / name).unlink()
            removed += 1
        for name in dirs:
            child = folder / name
            if child.is_symlink():
                child.unlink()
                removed += 1
    for directory, _dirs, _files in os.walk(root, topdown=False, followlinks=False):
        folder = Path(directory)
        if folder != root or not keep_root:
            folder.rmdir()
    return removed


def chunks(lease: fill.ReadLease, digest: str, length: int) -> Iterator[bytes]:
    offset = 0
    while offset < length:
        amount = min(1 << 20, length - offset)
        buffer = bytearray(amount)
        lease.read_into(digest, length, offset, amount, buffer)
        yield bytes(buffer)
        offset += amount
