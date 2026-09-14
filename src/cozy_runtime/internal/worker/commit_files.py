"""Snapshot one parent's own spool file into independently retained native custody."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.author._services import MAX_OUTPUT_BYTES
from cozy_runtime.internal import output_budget
from cozy_runtime.internal.pathkey import opaque_key
from cozy_runtime.internal.worker import workspace_byte_outputs as outputs
from cozy_runtime.internal.worker.workspace import Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def prepare(
    workspace: Workspace,
    owner: str,
    command: pb.NativeSourceCommand,
    computation: bytes,
    spool_root: Path | None,
) -> dict[str, Any]:
    call = command.parent_call
    with workspace.locked() as db:
        native = outputs.native_call(db, owner, command)
        parent = db.execute(
            "SELECT invocation FROM attempts WHERE owner=? AND request=? AND ordinal=?",
            (owner, call.parent_request_id, call.parent_attempt_ordinal),
        ).fetchone()
        budget = output_budget.intermediate(documents.read(parent["invocation"], pb.InvocationSpec))
        args = canonical_json.decode(native["accepted"])
        slot = args.get("slot")
        tail = str(slot).removeprefix("file/")
        if (
            spool_root is None
            or set(args) != {"slot", "digest", "size_bytes", "media_type"}
            or not isinstance(slot, str)
            or not slot.startswith("file/")
            or not re.fullmatch(r"[0-9]{4,16}", tail)
            or f"{int(tail):04d}" != tail
            or int(tail) == 0
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", args["digest"])
            or type(args["size_bytes"]) is not int
            or not 0 <= args["size_bytes"] <= MAX_OUTPUT_BYTES
            or not isinstance(args["media_type"], str)
            or not 0 < len(args["media_type"].encode()) <= 255
            or any(ord(c) < 32 or ord(c) > 126 for c in args["media_type"])
        ):
            raise WorkspaceRefusal("file commit has no bounded own pending file")
        # The worker derives the directory and basename. Neither is author input.
        spool = spool_root / opaque_key(
            "attempt-input-spool", call.parent_request_id, call.parent_attempt_ordinal
        )
        body = canonical_json.encode(
            {
                "entries": [
                    {
                        "path": "payload",
                        "kind": "file",
                        "blob": {"sha256": args["digest"][7:], "length": args["size_bytes"]},
                    }
                ]
            }
        )
    row = outputs.reserve_native(workspace, owner, command, body, MAX_OUTPUT_BYTES)
    with workspace.locked() as db:
        outputs.native_call(db, owner, command)
        total = db.execute(
            "SELECT coalesce(sum(content_bytes),0) FROM byte_outputs "
            "WHERE owner=? AND request=? AND ordinal=? AND slot LIKE 'runtime.commit_file.%'",
            (owner, row["request"], row["ordinal"]),
        ).fetchone()[0]
        if total > budget:
            raise WorkspaceRefusal("file commit exceeds its parent output byte budget")
        db.execute(
            "UPDATE native_calls SET state='executing',computation_digest=? "
            "WHERE owner=? AND service_id=?",
            (computation, owner, command.service_id),
        )
    return {
        "file_spool": str(spool),
        "file_basename": f"file-{tail}",
        "file_staging": str(workspace.directory / "byte-staging"),
        "view_owner": row["id"],
        "file_manifest": base64.b64encode(body).decode(),
    }


def execute(store: Any, command: dict[str, Any]) -> dict[str, Any]:
    """Runs in the killable native process; a writable spool is never retained directly."""
    args, root = command["arguments"], command["view_owner"]
    existing = store.tree_root(root)
    if existing and existing["complete"] and not existing["released"]:
        produced = existing
    else:
        staging = Path(command["file_staging"])
        staging.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory = os.open(command["file_spool"], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(
                command["file_basename"],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
        finally:
            os.close(directory)
        with os.fdopen(fd, "rb") as source, tempfile.TemporaryDirectory(dir=staging) as temp:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size != args["size_bytes"]:
                raise WorkspaceRefusal("pending file size or kind changed")
            snapshot = Path(temp) / "payload"
            length, digest = 0, hashlib.sha256()
            with snapshot.open("xb") as dest:
                while block := source.read(min(1 << 20, args["size_bytes"] - length + 1)):
                    length += len(block)
                    if length > args["size_bytes"]:
                        raise WorkspaceRefusal("pending file exceeded its declared byte bound")
                    dest.write(block)
                    digest.update(block)
                dest.flush()
                os.fsync(dest.fileno())
            after = os.fstat(source.fileno())
            if (
                length != args["size_bytes"]
                or "sha256:" + digest.hexdigest() != args["digest"]
                or (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            ):
                raise WorkspaceRefusal("pending file changed during its snapshot")
            produced = store.import_tree(
                root,
                base64.b64decode(command["file_manifest"], validate=True),
                [("payload", str(snapshot))],
            )
    return {
        "result": {
            "file": {
                "asset_ref": args["digest"],
                "kind": "file",
                "digest": args["digest"],
                "size_bytes": args["size_bytes"],
                "media_type": args["media_type"],
            }
        },
        "native_receipt": base64.b64encode(produced["receipt"]).decode(),
    }
