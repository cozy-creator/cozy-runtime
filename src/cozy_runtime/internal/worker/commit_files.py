"""Snapshot one parent's own spool file into independently retained native custody."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author._calls import RUNTIME_UPDATE
from cozy_runtime.author._services import MAX_OUTPUT_BYTES
from cozy_runtime.internal import fill, output_budget, source_interfaces
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.source_steps import Commit, Produced
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

REQUEST_FIELDS = frozenset(source_interfaces.CommitFile.__struct_fields__)


class UnsupportedCommitRequest(WorkspaceRefusal):
    """A newer SDK asked this commit for behaviour this Runtime does not implement."""

    code = "commit_request_unsupported"


def unsupported_detail(accepted: bytes) -> str:
    """Name the options a commit request carries that this Runtime does not serve."""
    try:
        args = canonical_json.decode(accepted)
    except ValueError:
        return ""
    unknown = sorted(set(args) - REQUEST_FIELDS) if isinstance(args, dict) else []
    if not unknown:
        return ""
    return f"commit_file does not support {', '.join(unknown)}: {RUNTIME_UPDATE}"[:1024]


def prepare(
    workspace: Workspace,
    owner: str,
    command: pb.NativeSourceCommand,
    computation: bytes,
    spool_root: Path | None,
) -> Commit:
    call = command.parent_call
    with workspace.locked() as db:
        native = outputs.native_call(db, owner, command)
        parent = db.execute(
            "SELECT invocation FROM attempts WHERE owner=? AND request=? AND ordinal=?",
            (owner, call.parent_request_id, call.parent_attempt_ordinal),
        ).fetchone()
        budget = output_budget.intermediate(documents.read(parent["invocation"], pb.InvocationSpec))
        if detail := unsupported_detail(native.accepted):
            # Dropping an unknown option could write a different output; only this call fails.
            raise UnsupportedCommitRequest(detail)
        request = msgspec.json.decode(native.accepted, type=source_interfaces.CommitFile)
        tail = request.slot.removeprefix("file/")
        if (
            spool_root is None
            or not request.slot.startswith("file/")
            or not re.fullmatch(r"[0-9]{4,16}", tail)
            or f"{int(tail):04d}" != tail
            or int(tail) == 0
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", request.digest)
            or not 0 <= request.size_bytes <= MAX_OUTPUT_BYTES
            or not 0 < len(request.media_type.encode()) <= 255
            or any(ord(c) < 32 or ord(c) > 126 for c in request.media_type)
        ):
            raise WorkspaceRefusal("file commit has no bounded own pending file")
        # The worker derives the directory and basename. Neither is author input.
        spool = spool_root / opaque_key(
            "attempt-input-spool", call.parent_request_id, call.parent_attempt_ordinal
        )
        body = outputs.payload_manifest(outputs.Blob(request.digest[7:], request.size_bytes))
    row = outputs.reserve_native(workspace, owner, command, body, MAX_OUTPUT_BYTES)
    with workspace.locked() as db:
        outputs.native_call(db, owner, command)
        total = db.execute(
            "SELECT coalesce(sum(content_bytes),0) FROM byte_outputs "
            "WHERE owner=? AND request=? AND ordinal=? AND slot LIKE 'runtime.commit_file.%'",
            (owner, row.request, row.ordinal),
        ).fetchone()[0]
        if total > budget:
            raise WorkspaceRefusal("file commit exceeds its parent output byte budget")
        db.execute(
            "UPDATE native_calls SET state='executing',computation_digest=? "
            "WHERE owner=? AND service_id=?",
            (computation, owner, command.service_id),
        )
    return Commit(
        request=request,
        view_owner=row.id,
        spool=str(spool),
        basename=f"file-{tail}",
        staging=str(workspace.directory / "byte-staging"),
        manifest=body,
    )


def execute(store: fill.Store, step: Commit) -> Produced:
    """Runs in the killable native process; a writable spool is never retained directly."""
    request, root = step.request, step.view_owner
    existing = store.tree_root(root)
    if existing and existing["complete"] and not existing["released"]:
        produced = existing
    else:
        staging = Path(step.staging)
        staging.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory = os.open(step.spool, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(
                step.basename,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
        finally:
            os.close(directory)
        with os.fdopen(fd, "rb") as source, tempfile.TemporaryDirectory(dir=staging) as temp:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != request.size_bytes:
                raise WorkspaceRefusal("pending file size or kind changed")
            snapshot = Path(temp) / "payload"
            length, digest = 0, hashlib.sha256()
            with snapshot.open("xb") as dest:
                while block := source.read(min(1 << 20, request.size_bytes - length + 1)):
                    length += len(block)
                    if length > request.size_bytes:
                        raise WorkspaceRefusal("pending file exceeded its declared byte bound")
                    dest.write(block)
                    digest.update(block)
                dest.flush()
                os.fsync(dest.fileno())
            after = os.fstat(source.fileno())
            if (
                length != request.size_bytes
                or "sha256:" + digest.hexdigest() != request.digest
                or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            ):
                raise WorkspaceRefusal("pending file changed during its snapshot")
            produced = store.import_tree(root, step.manifest, [("payload", str(snapshot))])
    file = {"asset_ref": request.digest, "kind": "file", "digest": request.digest}
    return Produced(
        result={
            "file": {**file, "size_bytes": request.size_bytes, "media_type": request.media_type}
        },
        native_receipt=produced["receipt"],
    )
