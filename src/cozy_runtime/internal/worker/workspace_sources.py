"""Fixed native source-call state within the common physical workspace journal.

Creator owns accepted intent and cancellation. These rows record its accepted native
execution, exact pin and custody; they do not schedule jobs or define a retry queue.
"""

from __future__ import annotations

import hashlib
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill, source_interfaces
from cozy_runtime.internal.worker.workspace import Journal, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

# Integrated by Workspace.locked after the schema3 weights-adoption migration.
SCHEMA4 = """
CREATE TABLE native_calls (
 owner TEXT NOT NULL, parent_request TEXT NOT NULL, call_index INTEGER NOT NULL,
 parent_ordinal INTEGER NOT NULL, parent_spec BLOB NOT NULL, intent_digest BLOB NOT NULL,
 service_id TEXT NOT NULL, operation TEXT NOT NULL,
 accepted BLOB NOT NULL, pinned BLOB NOT NULL DEFAULT x'',
 computation_digest BLOB NOT NULL DEFAULT x'',
 state TEXT NOT NULL CHECK(state IN
 ('accepted','resolved','executing','complete','failed','canceled')),
 result BLOB NOT NULL DEFAULT x'', native_receipt BLOB NOT NULL DEFAULT x'',
 native_owner TEXT NOT NULL,
 native_error TEXT NOT NULL DEFAULT '',
 PRIMARY KEY(owner,parent_request,call_index), UNIQUE(owner,service_id)
) STRICT;
CREATE TABLE source_access (
 owner TEXT NOT NULL, request TEXT NOT NULL, call_index INTEGER NOT NULL,
 producer TEXT NOT NULL, native_digest BLOB NOT NULL,
 manifest BLOB NOT NULL, manifest_length INTEGER NOT NULL,
 retention_id TEXT PRIMARY KEY,
 state TEXT NOT NULL CHECK(state IN ('retaining','held','releasing','released')),
 UNIQUE(owner,request,call_index,producer,native_digest)
) STRICT;
ALTER TABLE holds ADD COLUMN kind TEXT NOT NULL DEFAULT 'derived';
"""


def identity(owner: str, parent: str, index: int) -> str:
    return (
        "source-" + hashlib.sha256(canonical_json.encode([owner, parent, index])).hexdigest()[:48]
    )


def native_owner(owner: str, service: str) -> str:
    return (
        "sha256:"
        + hashlib.sha256(canonical_json.encode(["native-source", owner, service])).hexdigest()
    )


def _parent(db: Journal, owner: str, call: pb.ChildCallRequest) -> None:
    row = db.execute(
        "SELECT spec,state,fenced FROM attempts WHERE owner=? AND request=? AND ordinal=?",
        (owner, call.parent_request_id, call.parent_attempt_ordinal),
    ).fetchone()
    if (
        row is None
        or row["spec"] != call.parent_invocation_spec_digest
        or row["fenced"]
        or row["state"] != "running"
    ):
        raise WorkspaceRefusal("native source call has no current running parent attempt")


def accepted(workspace: Workspace, owner: str, command: pb.NativeSourceCommand) -> dict[str, Any]:
    call = command.parent_call
    if (
        call.call_index >= 32
        or call.interface_digest != documents.raw(source_interfaces.DIGEST)
        or call.module != source_interfaces.MODULE
        or len(call.request_canonical_bytes) > 48 << 10
    ):
        raise WorkspaceRefusal("native source call exceeds its fixed interface")
    expected_operation = {
        "download_huggingface": pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
        "download_civitai": pb.NATIVE_SOURCE_OPERATION_CIVITAI,
        "convert_cozytensors": pb.NATIVE_SOURCE_OPERATION_CONVERT,
    }.get(call.export)
    if expected_operation is None or command.operation != expected_operation:
        raise WorkspaceRefusal("native source operation differs from its fixed interface")
    value = canonical_json.decode(call.request_canonical_bytes)
    intent = canonical_json.encode(
        {
            "interface_digest": source_interfaces.DIGEST,
            "module": call.module,
            "export": call.export,
            "request": value,
        }
    )
    if hashlib.sha256(intent).digest() != call.intent_digest:
        raise WorkspaceRefusal("native source intent digest differs")
    if command.service_id != identity(owner, call.parent_request_id, call.call_index):
        raise WorkspaceRefusal("native source service id differs from parent call")
    with workspace.locked() as db:
        _parent(db, owner, call)
        row = db.execute(
            "SELECT * FROM native_calls WHERE owner=? AND parent_request=? AND call_index=?",
            (owner, call.parent_request_id, call.call_index),
        ).fetchone()
        if row is None:
            db.execute(
                "INSERT INTO native_calls(owner,parent_request,call_index,parent_ordinal,"
                "parent_spec,intent_digest,"
                "service_id,operation,accepted,state,native_owner) "
                "VALUES(?,?,?,?,?,?,?,?,?,'accepted',?)",
                (
                    owner,
                    call.parent_request_id,
                    call.call_index,
                    call.parent_attempt_ordinal,
                    call.parent_invocation_spec_digest,
                    call.intent_digest,
                    command.service_id,
                    call.export,
                    call.request_canonical_bytes,
                    native_owner(owner, command.service_id),
                ),
            )
        elif (
            row["intent_digest"] != call.intent_digest
            or row["accepted"] != call.request_canonical_bytes
        ):
            raise WorkspaceRefusal("parent call already names another native source intent")
        else:
            db.execute(
                "UPDATE native_calls SET parent_ordinal=?,parent_spec=? "
                "WHERE owner=? AND service_id=?",
                (
                    call.parent_attempt_ordinal,
                    call.parent_invocation_spec_digest,
                    owner,
                    command.service_id,
                ),
            )
        return dict(
            db.execute(
                "SELECT * FROM native_calls WHERE owner=? AND service_id=?",
                (owner, command.service_id),
            ).fetchone()
        )


def selection_content(selection: pb.NativeSourceSelection) -> dict[str, Any]:
    """Only the immutable accepted pin; delivery URLs and credentials never persist."""
    return {
        "canonical": selection.canonical,
        "selection_digest": documents.spell(selection.selection_digest),
        "content_manifest": {
            "digest": documents.spell(selection.content_manifest.digest),
            "length": selection.content_manifest.length,
        },
        "members": [
            {
                "member": row.member,
                "digest": documents.spell(row.object.digest),
                "length": row.object.length,
            }
            for row in selection.members
        ],
    }


def record_pin(
    workspace: Workspace,
    owner: str,
    command: pb.NativeSourceCommand,
    selection: pb.NativeSourceSelection,
) -> None:
    content = canonical_json.encode(selection_content(selection))
    if len(content) > 1 << 20 or not 0 < len(selection.members) <= 4096:
        raise WorkspaceRefusal("source selection exceeds its bounded roster")
    with workspace.locked() as db:
        _parent(db, owner, command.parent_call)
        row = db.execute(
            "SELECT * FROM native_calls WHERE owner=? AND service_id=?", (owner, command.service_id)
        ).fetchone()
        if row is None or (row["pinned"] and row["pinned"] != content):
            raise WorkspaceRefusal("native call changed its accepted immutable pin")
        db.execute(
            "UPDATE native_calls SET pinned=?,"
            "state=CASE WHEN state='accepted' THEN 'resolved' ELSE state END "
            "WHERE owner=? AND service_id=?",
            (content, owner, command.service_id),
        )


def record_complete(
    workspace: Workspace,
    owner: str,
    command: pb.NativeSourceCommand,
    result: bytes,
    computation: bytes,
    native_receipt: bytes,
) -> None:
    value = canonical_json.decode(result)
    receipt = canonical_json.decode(native_receipt)
    if len(native_receipt) > 1 << 20:
        raise WorkspaceRefusal("native receipt exceeds its bounded document")
    if len(result) > 48 << 10 or len(computation) != 32:
        raise WorkspaceRefusal("native source completion exceeds its result identity bound")
    with workspace.locked() as db:
        _parent(db, owner, command.parent_call)
        row = db.execute(
            "SELECT * FROM native_calls WHERE owner=? AND service_id=?", (owner, command.service_id)
        ).fetchone()
        if row is None or row["state"] == "canceled":
            raise WorkspaceRefusal("source completion has no active accepted call")
        store = fill.tensorfs_module().Store.open(str(workspace.store_root))
        source = row["operation"] != "convert_cozytensors"

        def validate_native() -> None:
            if source:
                observed = store.tree_root(row["native_owner"])
                if (
                    not observed
                    or not observed["complete"]
                    or observed["released"]
                    or observed["receipt"] != native_receipt
                ):
                    raise WorkspaceRefusal(
                        "source completion lacks its exact retained native receipt"
                    )
            else:
                observed = store.derived_lookup(row["native_owner"])
                if (
                    observed.get("state") != "committed"
                    or canonical_json.encode(observed.get("receipt")) != native_receipt
                ):
                    raise WorkspaceRefusal(
                        "converted model completion lacks its exact native derive receipt"
                    )
            manifest = receipt["manifest"]
            if (
                value.get("producer_request_id") != command.service_id
                or value.get("output_slot") != ("source" if source else "model")
                or value.get("tensorfs_receipt_digest")
                != "sha256:" + hashlib.sha256(native_receipt).hexdigest()
                or value.get("manifest")
                != {"digest": "sha256:" + manifest["sha256"], "length": manifest["length"]}
            ):
                raise WorkspaceRefusal("source result differs from its native receipt")

        db.native(validate_native)
        _parent(db, owner, command.parent_call)
        if row["result"] and (
            row["result"] != result
            or row["computation_digest"] != computation
            or row["native_receipt"] != native_receipt
        ):
            raise WorkspaceRefusal("source completion contradicts its prior result")
        db.execute(
            "UPDATE native_calls SET result=?,computation_digest=?,native_receipt=?,"
            "state='complete',native_error='' "
            "WHERE owner=? AND service_id=?",
            (result, computation, native_receipt, owner, command.service_id),
        )
