"""Project a granted native Tree member without copying its materialized bytes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.author._assets import file_state
from cozy_runtime.author._media import KIND_MEDIA, SNIFF_BYTES, admits, sniff
from cozy_runtime.internal.worker import byte_inputs, machine_models, workspace_byte_outputs
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .attempts import AttemptRecord
    from .calls import Calls


def project(calls: Calls, attempt: AttemptRecord, frame: Mapping[str, Any]) -> dict[str, Any]:
    """Only existing attempt capabilities can authorize a projection or its custody."""
    from .calls import _Received

    try:
        with calls.lock:
            workspace = calls.workspace
            if workspace is None or attempt.state != "running" or attempt.canceling:
                raise WorkspaceRefusal("Tree member requires an active native attempt")
            digest, path, kind = frame.get("tree"), frame.get("path"), frame.get("asset_kind")
            limit, media_types = frame.get("max_bytes"), frame.get("media_types")
            if (
                not isinstance(digest, str)
                or not isinstance(path, str)
                or not path
                or len(path.encode()) > 4096
                or "\\" in path
                or "\x00" in path
                or PurePosixPath(path).is_absolute()
                or str(PurePosixPath(path)) != path
                or any(part in (".", "..") for part in PurePosixPath(path).parts)
                or kind not in KIND_MEDIA
                or type(limit) is not int
                or not 0 < limit <= byte_inputs.MAX_BYTES
                or not isinstance(media_types, list)
                or len(media_types) > 32
                or any(not isinstance(item, str) for item in media_types)
            ):
                raise WorkspaceRefusal("Tree member requires a canonical path and bounded asset")
            parent = (attempt.request_id, attempt.attempt)
            source: pb.NativeByteRetentionRequest | None = None
            local: Path | None = None
            for received in calls.received.get(parent, {}).values():
                if received.metadata["kind"] == "tree" and received.metadata["digest"] == digest:
                    source = received.source
                    local = Path(received.metadata["local"]) / path
                    break
            if source is None:
                for entry in attempt.grant.inputs.values():
                    if (
                        entry.native_tree is not None
                        and entry.native_tree.source.manifest.digest == documents.raw(digest)
                        and (
                            entry.kind_mime == byte_inputs.TREE_MIME
                            or entry.input_id.startswith("tree:")
                        )
                    ):
                        materialized = attempt.trees.get(digest)
                        if materialized is not None and materialized[1] == digest:
                            source, local = entry.native_tree, materialized[0] / path
                            break
            if source is None or local is None:
                raise WorkspaceRefusal("Tree member has no exact parent capability")
            with workspace_byte_outputs.leased(workspace, calls.owner(), source) as (_, _, members):
                member = next((item for item in members if item["path"] == path), None)
                if member is None:
                    raise WorkspaceRefusal("Tree member path is absent from its native manifest")
                blob = member["blob"]
                if blob["length"] > limit:
                    raise WorkspaceRefusal("Tree member exceeds its declared byte bound")
                # Materialized native trees are immutable and have no symlinks. Check
                # every path component again before exposing an existing local file.
                root = local
                for _ in PurePosixPath(path).parts:
                    if root.is_symlink():
                        raise WorkspaceRefusal("Tree member contains a symbolic link")
                    root = root.parent
                state = file_state(local)
                actual = hashlib.sha256()
                with local.open("rb") as stream:
                    media = sniff(stream.read(SNIFF_BYTES)) or "application/octet-stream"
                    stream.seek(0)
                    while block := stream.read(1 << 20):
                        actual.update(block)
                if (
                    actual.hexdigest() != blob["sha256"]
                    or local.stat().st_size != blob["length"]
                    or file_state(local) != state
                    or not admits(kind, media)
                    or (media_types and media not in media_types)
                ):
                    raise WorkspaceRefusal("Tree member changed its native bytes or media type")
                identity = canonical_json.encode([kind, "sha256:" + blob["sha256"], media]).decode()
                held = calls.received.get(parent, {}).get(identity)
                if held is None:
                    body = canonical_json.encode(
                        {"entries": [{"path": "payload", "kind": "file", "blob": blob}]}
                    )
                    slot = "runtime.tree_member." + hashlib.sha256(body).hexdigest()
                    native = workspace_byte_outputs.commit(
                        workspace,
                        calls.owner(),
                        attempt.request_id,
                        attempt.attempt,
                        attempt.digest,
                        slot,
                        body,
                        [("payload", local)],
                        limit,
                    )
                    retained = machine_models.retain_bytes(
                        workspace, calls.owner(), attempt.request_id, slot, native
                    )
                    metadata = {
                        "kind": kind,
                        "digest": "sha256:" + blob["sha256"],
                        "length": blob["length"],
                        "content_bytes": blob["length"],
                        "media_type": media,
                        "local": str(local),
                    }
                    held = _Received(
                        metadata,
                        pb.NativeByteRetentionRequest(
                            source=retained.source, retention_id=retained.retention_id
                        ),
                    )
                    calls.received.setdefault(parent, {})[identity] = held
                if file_state(local) != state:
                    raise WorkspaceRefusal("Tree member changed during native projection")
                return {
                    "ok": True,
                    "tree": digest,
                    "path": path,
                    "digest": "sha256:" + blob["sha256"],
                    "length": blob["length"],
                    "media_type": media,
                    "local": str(local),
                    "file_state": list(state),
                }
    except (OSError, ValueError) as exc:
        return {"ok": False, "code": "tree_member_refused", "detail": str(exc)[:1024]}
