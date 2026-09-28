"""Readonly native source projections use the common byte-output authority."""

from __future__ import annotations

import hashlib

import msgspec

from cozy_runtime.author._artifacts import SourceArtifact
from cozy_runtime.author.sources import SOURCE_FILES_MAX_BYTES
from cozy_runtime.internal import fill
from cozy_runtime.internal.source_interfaces import SourceFiles
from cozy_runtime.internal.worker import workspace_byte_outputs as byte_outputs
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.source_steps import View
from cozy_runtime.internal.worker.workspace import Journal, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def _access(db: Journal, owner: str, parent: str, artifact: SourceArtifact) -> str:
    producer = workspace_sources.native_owner(owner, artifact.producer_request_id)
    access = db.execute(
        "SELECT a.retention_id FROM source_access a JOIN holds h ON h.id=a.retention_id "
        "WHERE a.owner=? AND a.request=? AND a.producer=? AND a.native_digest=? "
        "AND a.manifest=? AND a.manifest_length=? AND a.state='held' "
        "AND h.owner=a.owner AND h.transaction_id=a.producer AND h.kind='tree' "
        "AND h.state='held' AND h.native_digest=a.native_digest "
        "AND h.manifest=a.manifest AND h.manifest_length=a.manifest_length",
        (
            owner,
            parent,
            producer,
            documents.raw(artifact.tensorfs_receipt_digest),
            documents.raw(artifact.manifest.digest),
            artifact.manifest.length,
        ),
    ).fetchone()
    if access is None:
        raise WorkspaceRefusal("source view lacks exact retained source custody")
    return str(access["retention_id"])


def _source(db: Journal, owner: str, command: pb.NativeSourceCommand) -> tuple[SourceArtifact, str]:
    call = byte_outputs.native_call(db, owner, command)
    if call.operation != "source_files":
        raise WorkspaceRefusal("source projection requires its own native operation")
    artifact = msgspec.json.decode(call.accepted, type=SourceFiles).source
    return artifact, _access(db, owner, call.parent_request, artifact)


def prepare(
    workspace: Workspace, owner: str, command: pb.NativeSourceCommand, computation: bytes
) -> View:
    """Validate source custody and reserve before the killable native process runs."""
    with workspace.locked() as db:
        artifact, access = _source(db, owner, command)
        manifest = artifact.manifest
        if not 0 < manifest.length <= 1 << 20:
            raise WorkspaceRefusal("source view manifest exceeds its metadata bound")
        store = fill.store(workspace.store_root)
        observed = db.native(lambda: store.tree_root(access))
        if (
            not observed
            or not observed["complete"]
            or observed["released"]
            or observed["manifest_digest"] != manifest.digest
            or observed["manifest_length"] != manifest.length
            or observed["receipt_digest"] != artifact.tensorfs_receipt_digest
        ):
            raise WorkspaceRefusal("source view native source receipt changed")
        body = db.native(lambda: store.manifest(manifest.digest)["manifest"])
        if (
            len(body) != manifest.length
            or documents.spell(hashlib.sha256(body).digest()) != manifest.digest
        ):
            raise WorkspaceRefusal("source view manifest differs from accepted source")
        row = db.native(
            lambda: byte_outputs.reserve_native(
                workspace, owner, command, body, SOURCE_FILES_MAX_BYTES
            )
        )
        if _source(db, owner, command) != (artifact, access):
            raise WorkspaceRefusal("source view access changed during native validation")
        db.execute(
            "UPDATE native_calls SET state='executing',computation_digest=? "
            "WHERE owner=? AND service_id=?",
            (computation, owner, command.service_id),
        )
        return View(
            source_owner=access,
            view_owner=row.id,
            manifest=manifest,
            receipt_digest=artifact.tensorfs_receipt_digest,
            view_bytes=row.content_bytes,
        )


def complete(
    workspace: Workspace,
    owner: str,
    command: pb.NativeSourceCommand,
    result: bytes,
    computation: bytes,
    receipt: bytes,
    root_id: str,
) -> tuple[pb.NativeByteTreeRef, int, bytes]:
    with workspace.locked() as db:
        source = _source(db, owner, command)
        fact = db.native(
            lambda: byte_outputs.complete_native(
                workspace, owner, command, result, computation, receipt, root_id
            )
        )
        if _source(db, owner, command) != source:
            raise WorkspaceRefusal("source view access changed during native completion")
        return fact


def replay(
    workspace: Workspace, owner: str, command: pb.NativeSourceCommand
) -> tuple[pb.NativeByteTreeRef, int, bytes]:
    return byte_outputs.replay_native(workspace, owner, command)


def abort_unfinished(workspace: Workspace, owner: str, command: pb.NativeSourceCommand) -> None:
    """Release stopped work; a detached active call remains available for reattachment."""
    with workspace.locked() as db:
        row = db.execute(
            "SELECT state FROM native_calls WHERE owner=? AND service_id=?",
            (owner, command.service_id),
        ).fetchone()
        stopped = row is not None and row["state"] in ("stopped", "releasing", "released")
    if stopped:
        byte_outputs.release_native(workspace, owner, command.service_id)
