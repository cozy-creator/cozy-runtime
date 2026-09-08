"""Completed native source views share the ordinary byte-output authority."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.author.sources import SOURCE_FILES_MAX_BYTES
from cozy_runtime.internal import fill
from cozy_runtime.internal.worker import workspace_byte_outputs as byte_outputs
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.workspace import Journal, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

PREFIX = "runtime.source_files."


def slot(index: int) -> str:
    return f"{PREFIX}{index}"


def _value(row: Mapping[str, Any]) -> dict[str, Any]:
    digest = documents.spell(hashlib.sha256(row["manifest_body"]).digest())
    return {
        "files": {
            "asset_ref": digest,
            "kind": "tree",
            "digest": digest,
            "size_bytes": row["content_bytes"],
        }
    }


def recorded(
    db: Journal, row: Mapping[str, Any], *, complete: bool, released: bool = False
) -> bool:
    """A reserved name alone is never permission to publish a pending author output."""
    suffix = str(row["slot"]).removeprefix(PREFIX)
    if not suffix.isascii() or not suffix.isdecimal() or str(int(suffix)) != suffix:
        return False
    index = int(suffix)
    if not 0 <= index < 1 << 32 or row["slot"] != slot(index):
        return False
    if row["id"] != byte_outputs.identity(
        row["owner"], row["request"], row["ordinal"], row["spec"], row["slot"]
    ):
        return False
    call = db.execute(
        "SELECT * FROM native_calls WHERE owner=? AND parent_request=? AND call_index=? "
        "AND operation='source_files'",
        (row["owner"], row["request"], index),
    ).fetchone()
    if call is None:
        return False
    artifact = canonical_json.decode(call["accepted"])["source"]
    if artifact["manifest"] != {
        "digest": documents.spell(hashlib.sha256(row["manifest_body"]).digest()),
        "length": len(row["manifest_body"]),
    }:
        return False
    if not complete:
        return True
    return (
        row["state"] in (("complete", "releasing", "released") if released else ("complete",))
        and call["state"] in (("complete", "releasing", "released") if released else ("complete",))
        and call["native_receipt"] == row["receipt"]
        and call["result"] == canonical_json.encode(_value(row))
    )


def _call(db: Journal, owner: str, command: pb.NativeSourceCommand) -> Any:
    workspace_sources._parent(db, owner, command.parent_call)
    row = db.execute(
        "SELECT * FROM native_calls WHERE owner=? AND service_id=?", (owner, command.service_id)
    ).fetchone()
    call = command.parent_call
    if (
        row is None
        or row["operation"] != "source_files"
        or row["parent_request"] != call.parent_request_id
        or row["call_index"] != call.call_index
        or row["parent_ordinal"] != call.parent_attempt_ordinal
        or row["parent_spec"] != call.parent_invocation_spec_digest
        or row["intent_digest"] != call.intent_digest
        or row["state"] not in ("accepted", "executing", "complete")
    ):
        raise WorkspaceRefusal("source view has no exact current native call")
    return row


def _access(db: Journal, owner: str, parent: str, artifact: dict[str, Any]) -> str:
    producer = workspace_sources.native_owner(owner, artifact["producer_request_id"])
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
            documents.raw(artifact["tensorfs_receipt_digest"]),
            documents.raw(artifact["manifest"]["digest"]),
            artifact["manifest"]["length"],
        ),
    ).fetchone()
    if access is None:
        raise WorkspaceRefusal("source view lacks exact retained source custody")
    return str(access["retention_id"])


def prepare(
    workspace: Workspace, owner: str, command: pb.NativeSourceCommand, computation: bytes
) -> dict[str, Any]:
    """Record the reserved producer intent before the killable native process runs."""
    call = command.parent_call
    with workspace.locked() as db:
        native = _call(db, owner, command)
        artifact = canonical_json.decode(native["accepted"])["source"]
        access = _access(db, owner, call.parent_request_id, artifact)
        manifest = artifact["manifest"]
        if not 0 < manifest["length"] <= 1 << 20:
            raise WorkspaceRefusal("source view manifest exceeds its metadata bound")
        store = fill.tensorfs_module().Store.open(str(workspace.store_root))
        observed = db.native(lambda: store.tree_root(access))
        if (
            not observed
            or not observed["complete"]
            or observed["released"]
            or observed["manifest_digest"] != manifest["digest"]
            or observed["manifest_length"] != manifest["length"]
            or observed["receipt_digest"] != artifact["tensorfs_receipt_digest"]
        ):
            raise WorkspaceRefusal("source view native source receipt changed")
        body = db.native(lambda: bytes(store.manifest(manifest["digest"])["manifest"]))
        members = byte_outputs.manifest_members(body, SOURCE_FILES_MAX_BYTES)
        if (
            len(body) != manifest["length"]
            or documents.spell(hashlib.sha256(body).digest()) != manifest["digest"]
        ):
            raise WorkspaceRefusal("source view manifest differs from accepted source")
        _call(db, owner, command)
        if _access(db, owner, call.parent_request_id, artifact) != access:
            raise WorkspaceRefusal("source view access changed during native validation")
        output_id = slot(call.call_index)
        rows = db.execute(
            "SELECT * FROM byte_outputs WHERE owner=? AND request=? AND slot=? "
            "AND state NOT IN ('releasing','released')",
            (owner, call.parent_request_id, output_id),
        ).fetchall()
        if len(rows) > 1:
            raise WorkspaceRefusal("source view has conflicting original producers")
        row = rows[0] if rows else None
        root_id = (
            row["id"]
            if row is not None
            else byte_outputs.identity(
                owner,
                call.parent_request_id,
                call.parent_attempt_ordinal,
                call.parent_invocation_spec_digest,
                output_id,
            )
        )
        total = sum(member["blob"]["length"] for member in members)
        if row is None:
            count = db.execute(
                "SELECT count(*) FROM byte_outputs WHERE owner=? AND request=? AND ordinal=?",
                (owner, call.parent_request_id, call.parent_attempt_ordinal),
            ).fetchone()[0]
            if count >= byte_outputs.MAX_RESULTS:
                raise WorkspaceRefusal("source view exceeds the parent byte-output inventory")
            db.execute(
                "INSERT INTO byte_outputs(id,owner,request,ordinal,spec,slot,manifest_body,"
                "content_bytes,state) "
                "VALUES(?,?,?,?,?,?,?,?,'intent')",
                (
                    root_id,
                    owner,
                    call.parent_request_id,
                    call.parent_attempt_ordinal,
                    call.parent_invocation_spec_digest,
                    output_id,
                    body,
                    total,
                ),
            )
        elif (
            row["manifest_body"] != body
            or row["content_bytes"] != total
            or not recorded(db, row, complete=False)
        ):
            raise WorkspaceRefusal("source view changed its reserved producer intent")
        db.execute(
            "UPDATE native_calls SET state='executing',computation_digest=? "
            "WHERE owner=? AND service_id=?",
            (computation, owner, command.service_id),
        )
        return {
            "source_owner": access,
            "view_owner": root_id,
            "view_manifest": manifest,
            "view_bytes": total,
        }


def complete(
    workspace: Workspace,
    owner: str,
    command: pb.NativeSourceCommand,
    result: bytes,
    computation: bytes,
    receipt: bytes,
    root_id: str,
) -> tuple[pb.NativeByteTreeRef, int, bytes]:
    call = command.parent_call
    with workspace.locked() as db:
        native = _call(db, owner, command)
        row = db.execute("SELECT * FROM byte_outputs WHERE id=?", (root_id,)).fetchone()
        if (
            row is None
            or row["owner"] != owner
            or row["request"] != call.parent_request_id
            or row["slot"] != slot(call.call_index)
            or row["state"] not in ("intent", "complete")
            or not recorded(db, row, complete=False)
        ):
            raise WorkspaceRefusal("source view completion lacks its exact producer intent")
        if (
            result != canonical_json.encode(_value(row))
            or len(computation) != 32
            or native["computation_digest"] != computation
            or not 0 < len(receipt) <= 1 << 20
        ):
            raise WorkspaceRefusal("source view completion differs from its bounded result")
        store = fill.tensorfs_module().Store.open(str(workspace.store_root))
        observed = db.native(lambda: store.tree_root(root_id))
        if (
            not observed
            or not observed["complete"]
            or observed["released"]
            or observed["producer"] != root_id
            or observed["receipt"] != receipt
            or observed["manifest_digest"] != _value(row)["files"]["digest"]
            or observed["manifest_length"] != len(row["manifest_body"])
            or observed["receipt_digest"] != documents.spell(hashlib.sha256(receipt).digest())
        ):
            raise WorkspaceRefusal("source view has no matching native byte receipt")
        _call(db, owner, command)
        _access(
            db, owner, call.parent_request_id, canonical_json.decode(native["accepted"])["source"]
        )
        current = db.execute("SELECT * FROM byte_outputs WHERE id=?", (root_id,)).fetchone()
        if current["state"] not in ("intent", "complete") or current["receipt"] not in (
            b"",
            receipt,
        ):
            raise WorkspaceRefusal("source view producer changed during native completion")
        digest = hashlib.sha256(receipt).digest()
        db.execute(
            "UPDATE byte_outputs SET state='complete',receipt=?,native_digest=? WHERE id=?",
            (receipt, digest, root_id),
        )
        db.execute(
            "UPDATE native_calls SET state='complete',result=?,native_receipt=?,"
            "computation_digest=?,native_error='' "
            "WHERE owner=? AND service_id=?",
            (result, receipt, computation, owner, command.service_id),
        )
        return (
            byte_outputs.reference(
                dict(db.execute("SELECT * FROM byte_outputs WHERE id=?", (root_id,)).fetchone())
            ),
            int(row["ordinal"]),
            bytes(row["spec"]),
        )


def replay(
    workspace: Workspace, owner: str, command: pb.NativeSourceCommand
) -> tuple[pb.NativeByteTreeRef, int, bytes]:
    with workspace.locked() as db:
        native = _call(db, owner, command)
        rows = db.execute(
            "SELECT * FROM byte_outputs WHERE owner=? AND request=? AND slot=? AND native_digest=?",
            (
                owner,
                command.parent_call.parent_request_id,
                slot(command.parent_call.call_index),
                hashlib.sha256(native["native_receipt"]).digest(),
            ),
        ).fetchall()
        if len(rows) != 1 or not recorded(db, rows[0], complete=True):
            raise WorkspaceRefusal("completed source view has no exact retained producer")
        result = byte_outputs.reference(dict(rows[0]))
        store = fill.tensorfs_module().Store.open(str(workspace.store_root))
        observed = db.native(lambda: store.tree_root(result.producer_root_id))
        if (
            not observed
            or not observed["complete"]
            or observed["released"]
            or observed["receipt"] != native["native_receipt"]
        ):
            raise WorkspaceRefusal("completed source view native custody was lost")
        _call(db, owner, command)
        return result, int(rows[0]["ordinal"]), bytes(rows[0]["spec"])


def abort_unfinished(workspace: Workspace, owner: str, command: pb.NativeSourceCommand) -> None:
    """The native process has stopped; dispose only this call's unfinished producer."""
    with workspace.locked() as db:
        rows = db.execute(
            "SELECT * FROM byte_outputs WHERE owner=? AND request=? AND slot=? "
            "AND state<>'released'",
            (owner, command.parent_call.parent_request_id, slot(command.parent_call.call_index)),
        ).fetchall()
        ids = []
        for row in rows:
            if recorded(db, row, complete=True):
                continue
            if not recorded(db, row, complete=False):
                raise WorkspaceRefusal("source view cleanup has no exact native producer")
            db.execute("UPDATE byte_outputs SET state='releasing' WHERE id=?", (row["id"],))
            ids.append(row["id"])
    for root_id in ids:
        byte_outputs.release(workspace, owner, root_id)


def release_service(workspace: Workspace, owner: str, native: Mapping[str, Any]) -> None:
    """A terminal native call releases its original producer, including an older attempt."""
    with workspace.locked() as db:
        rows = db.execute(
            "SELECT * FROM byte_outputs WHERE owner=? AND request=? AND slot=? "
            "AND state<>'released'",
            (owner, native["parent_request"], slot(native["call_index"])),
        ).fetchall()
        for row in rows:
            if not recorded(db, row, complete=False):
                raise WorkspaceRefusal("source view release changed its native call")
            db.execute("UPDATE byte_outputs SET state='releasing' WHERE id=?", (row["id"],))
    for row in rows:
        byte_outputs.release(workspace, owner, row["id"])
