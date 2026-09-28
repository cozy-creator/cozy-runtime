"""Fixed native source-call state within the common physical workspace journal.

Creator owns accepted intent and cancellation. These rows record its accepted native
execution, exact pin and custody; they do not schedule jobs or define a retry queue.
"""

from __future__ import annotations

import hashlib
import time
from functools import partial
from typing import Any, Literal

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill, source_interfaces, weights_sink
from cozy_runtime.internal.call_intent import canonical_intent
from cozy_runtime.internal.worker import source_upload
from cozy_runtime.internal.worker.workspace import (
    Journal,
    Workspace,
    WorkspaceRefusal,
    artifact_identity,
    same_ref,
)
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
 ('accepted','resolved','executing','complete','failed','stopped','releasing','released')),
 result BLOB NOT NULL DEFAULT x'', native_receipt BLOB NOT NULL DEFAULT x'',
 native_owner TEXT NOT NULL,
 native_error TEXT NOT NULL DEFAULT '', release_ack BLOB NOT NULL DEFAULT x'',
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


class NativeCall(msgspec.Struct, frozen=True):
    """One ``native_calls`` row."""

    owner: str
    parent_request: str
    call_index: int
    parent_ordinal: int
    parent_spec: bytes
    intent_digest: bytes
    service_id: str
    operation: str
    accepted: bytes
    pinned: bytes
    computation_digest: bytes
    state: Literal[
        "accepted",
        "resolved",
        "executing",
        "complete",
        "failed",
        "stopped",
        "releasing",
        "released",
    ]
    result: bytes
    native_receipt: bytes
    native_owner: str
    native_error: str
    release_ack: bytes
    retired_ms: int


class SourceAccess(msgspec.Struct, frozen=True):
    """One ``source_access`` row."""

    owner: str
    request: str
    call_index: int
    producer: str
    native_digest: bytes
    manifest: bytes
    manifest_length: int
    retention_id: str
    state: Literal["retaining", "held", "releasing", "released"]


class HeldAccess(SourceAccess, frozen=True):
    """A ``source_access`` row joined with its hold's kind; every access has its hold."""

    kind: Literal["derived", "tree"]


def call_row(db: Journal, owner: str, service_id: str) -> NativeCall | None:
    return db.one(
        NativeCall, "SELECT * FROM native_calls WHERE owner=? AND service_id=?", (owner, service_id)
    )


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


def accepted(workspace: Workspace, owner: str, command: pb.NativeSourceCommand) -> NativeCall:
    call = command.parent_call
    if call.module != source_interfaces.MODULE or len(call.request_canonical_bytes) > 48 << 10:
        raise WorkspaceRefusal("native source call exceeds its fixed interface")
    expected_operation = {
        "download_huggingface": pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
        "download_civitai": pb.NATIVE_SOURCE_OPERATION_CIVITAI,
        "convert_cozytensors": pb.NATIVE_SOURCE_OPERATION_CONVERT,
        "source_files": pb.NATIVE_SOURCE_OPERATION_SOURCE_FILES,
        "commit_file": pb.NATIVE_SOURCE_OPERATION_COMMIT_FILE,
        "upload_huggingface": pb.NATIVE_SOURCE_OPERATION_HUGGINGFACE,
        "upload_civitai": pb.NATIVE_SOURCE_OPERATION_CIVITAI,
    }.get(call.export)
    if expected_operation is None or command.operation != expected_operation:
        raise WorkspaceRefusal("native source operation differs from its fixed interface")
    intent = canonical_intent(call)
    if hashlib.sha256(intent).digest() != call.intent_digest:
        raise WorkspaceRefusal("native source intent digest differs")
    if command.service_id != identity(owner, call.parent_request_id, call.call_index):
        raise WorkspaceRefusal("native source service id differs from parent call")
    with workspace.locked() as db:
        _parent(db, owner, call)
        row = db.one(
            NativeCall,
            "SELECT * FROM native_calls WHERE owner=? AND parent_request=? AND call_index=?",
            (owner, call.parent_request_id, call.call_index),
        )
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
            row.intent_digest != call.intent_digest or row.accepted != call.request_canonical_bytes
        ):
            raise WorkspaceRefusal("parent call already names another native source intent")
        else:
            if row.state in ("releasing", "released"):
                raise WorkspaceRefusal("native source call was permanently released")
            if row.state == "failed" and call.parent_attempt_ordinal > row.parent_ordinal:
                db.execute(
                    "UPDATE native_calls SET state=?,native_error='' "
                    "WHERE owner=? AND service_id=?",
                    ("resolved" if row.pinned else "accepted", owner, command.service_id),
                )
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
        row = call_row(db, owner, command.service_id)
        assert row is not None
        return row


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
        row = call_row(db, owner, command.service_id)
        if row is None or (row.pinned and row.pinned != content):
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
        row = call_row(db, owner, command.service_id)
        if row is None or row.state in ("releasing", "released"):
            raise WorkspaceRefusal("source completion has no active accepted call")
        store = fill.store(workspace.store_root)
        source = row.operation != "convert_cozytensors"
        producer = row.native_owner

        def validate_native() -> None:
            if source:
                observed = store.tree_root(producer)
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
                derived = store.derived_lookup(producer)
                if derived.get("state") != "committed" or not weights_sink.same_receipt(
                    derived["receipt"], canonical_json.decode(native_receipt)
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
                or not same_ref(
                    value.get("manifest"), "sha256:" + manifest["sha256"], manifest["length"]
                )
            ):
                raise WorkspaceRefusal("source result differs from its native receipt")

        db.native(validate_native)
        _parent(db, owner, command.parent_call)
        row = call_row(db, owner, command.service_id)
        if (
            row is None
            or row.state in ("releasing", "released")
            or row.intent_digest != command.parent_call.intent_digest
        ):
            raise WorkspaceRefusal("source call changed during native result validation")
        if row.result and (
            artifact_identity(canonical_json.decode(row.result)) != artifact_identity(value)
            or row.computation_digest != computation
            or row.native_receipt != native_receipt
        ):
            raise WorkspaceRefusal("source completion contradicts its prior result")
        db.execute(
            "UPDATE native_calls SET result=?,computation_digest=?,native_receipt=?,"
            "state='complete',native_error='' "
            "WHERE owner=? AND service_id=?",
            (result, computation, native_receipt, owner, command.service_id),
        )


def record_upload(
    workspace: Workspace,
    owner: str,
    command: pb.NativeSourceCommand,
    result: bytes,
    computation: bytes,
) -> None:
    """An upload's result is the Hub checkpoint it acknowledged; it retains no local root."""
    if len(result) > 48 << 10 or len(computation) != 32:
        raise WorkspaceRefusal("upload completion exceeds its result identity bound")
    with workspace.locked() as db:
        _parent(db, owner, command.parent_call)
        row = call_row(db, owner, command.service_id)
        if (
            row is None
            or row.state in ("releasing", "released")
            or row.operation not in source_interfaces.UPLOADS
            or row.intent_digest != command.parent_call.intent_digest
        ):
            raise WorkspaceRefusal("upload completion has no active accepted call")
        if row.result and (row.result != result or row.computation_digest != computation):
            raise WorkspaceRefusal("upload completion contradicts its prior result")
        db.execute(
            "UPDATE native_calls SET result=?,computation_digest=?,state='complete',"
            "native_error='' WHERE owner=? AND service_id=?",
            (result, computation, owner, command.service_id),
        )


def adopt_download_progress(workspace: Workspace, owner: str, service_id: str) -> bool:
    """Independently acquire only a stopped predecessor with the exact native work key."""
    with workspace.locked() as db:
        current = call_row(db, owner, service_id)
        if current is None or current.operation == "convert_cozytensors":
            return False
        candidates = db.all(
            NativeCall,
            "SELECT n.* FROM native_calls n LEFT JOIN attempts a "
            "ON a.owner=n.owner AND a.request=n.parent_request AND a.ordinal=n.parent_ordinal "
            "WHERE n.owner=? AND n.service_id<>? AND n.computation_digest=? AND n.operation=? "
            "AND length(n.result)=0 AND n.state<>'released' AND (n.state IN ('failed','stopped') "
            "OR a.fenced=1 OR a.state IN ('outcome','released')) ORDER BY n.service_id",
            (owner, service_id, current.computation_digest, current.operation),
        )
        store = fill.store(workspace.store_root)
        adopted = ""
        for candidate in candidates:
            try:
                db.native(
                    partial(
                        store.adopt_source_progress,
                        candidate.native_owner,
                        current.native_owner,
                    )
                )
            except Exception as exc:
                if getattr(exc, "code", None) in {
                    "STORE_BUSY",
                    "ROOT_ABSENT",
                    "TRANSACTION_CLOSED",
                    "DURABILITY_UNPROVEN",
                }:
                    continue
                raise
            adopted = candidate.service_id
            break
    if not adopted:
        return False
    # The recipient now retains the progress independently; a retired partial is spent.
    consume_partial(workspace, owner, adopted)
    return True


def conversion_predecessor(workspace: Workspace, owner: str, service_id: str) -> str | None:
    """Discover an exact stopped conversion; native readback/adoption proves the heads."""
    with workspace.locked() as db:
        current = call_row(db, owner, service_id)
        if current is None or current.operation != "convert_cozytensors":
            return None
        candidates = db.all(
            NativeCall,
            "SELECT n.* FROM native_calls n LEFT JOIN attempts a "
            "ON a.owner=n.owner AND a.request=n.parent_request AND a.ordinal=n.parent_ordinal "
            "WHERE n.owner=? AND n.service_id<>? AND n.computation_digest=? "
            "AND n.operation='convert_cozytensors' AND length(n.result)=0 AND n.state<>'released' "
            "AND (n.state IN ('failed','stopped') OR a.fenced=1 "
            "OR a.state IN ('outcome','released')) "
            "ORDER BY n.service_id",
            (owner, service_id, current.computation_digest),
        )
        store = fill.store(workspace.store_root)
        retained = set(db.native(store.model_source_operations))
        for candidate in candidates:
            if candidate.service_id in retained:
                return candidate.service_id
        return None


def stopped(workspace: Workspace, owner: str, command: pb.NativeSourceCommand) -> None:
    """Authenticate a stop against recorded call identity, including after parent exit."""
    call = command.parent_call
    with workspace.locked() as db:
        row = call_row(db, owner, command.service_id)
        if (
            row is None
            or row.parent_request != call.parent_request_id
            or row.call_index != call.call_index
            or row.intent_digest != call.intent_digest
            or row.parent_ordinal != call.parent_attempt_ordinal
            or row.parent_spec != call.parent_invocation_spec_digest
        ):
            raise WorkspaceRefusal("source stop changed its recorded parent call")
        db.execute(
            "UPDATE native_calls SET state='stopped' WHERE owner=? AND service_id=? "
            "AND state IN ('accepted','resolved','executing','failed')",
            (owner, command.service_id),
        )


def ack_identity(ack: pb.AttemptOutcomeAck) -> bytes:
    return canonical_json.encode(
        {
            "request": ack.request_id,
            "ordinal": ack.attempt_ordinal,
            "spec": documents.spell(ack.invocation_spec_digest),
            "outcome_id": ack.outcome_id,
            "outcome_digest": documents.spell(ack.outcome_digest),
        }
    )


def release_intent(db: Journal, owner: str, ack: pb.AttemptOutcomeAck) -> list[str]:
    """Called only after Workspace validates the exact terminal ACK with retain_work=false."""
    rows = db.execute(
        "SELECT service_id FROM native_calls WHERE owner=? AND parent_request=? "
        "AND parent_ordinal<=? AND state<>'released'",
        (owner, ack.request_id, ack.attempt_ordinal),
    ).fetchall()
    db.execute(
        "UPDATE native_calls SET state='releasing',release_ack=? WHERE owner=? "
        "AND parent_request=? AND parent_ordinal<=? AND state<>'released'",
        (ack_identity(ack), owner, ack.request_id, ack.attempt_ordinal),
    )
    return [str(row["service_id"]) for row in rows]


# Incomplete progress of these calls is keyed only by its exact pinned computation, so it
# outlives its released parent as a retired partial (workspace_partial) for a later
# identical call to adopt.
_PARTIAL_OPERATIONS = ("download_huggingface", "download_civitai", "convert_cozytensors")

# An upload's landed source is a donor for a later upload of the same source whether or
# not it finished, so a retired upload is a partial either way.
PARTIALS = (
    "SELECT n.* FROM native_calls n WHERE n.state='stopped' AND n.retired_ms>0 "
    "AND (length(n.result)=0 OR n.operation IN ('upload_huggingface','upload_civitai'))"
)


def live_uploads(db: Journal) -> set[str]:
    """Uploads whose conversion session is still owned by a call that may run again."""
    return {
        str(row[0])
        for row in db.execute(
            "SELECT service_id FROM native_calls WHERE operation IN "
            "('upload_huggingface','upload_civitai') AND state NOT IN ('stopped','released')"
        )
    }


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def _has_progress(store: fill.Store, row: NativeCall) -> bool:
    if row.operation == "convert_cozytensors":
        return row.service_id in store.model_source_operations()
    root = store.tree_root(row.native_owner)
    return root is not None and not root["released"] and not root["complete"]


def _release_native(
    store: fill.Store, owner: str, row: NativeCall, workspace: Workspace, live: set[str]
) -> None:
    if row.native_owner != native_owner(owner, row.service_id):
        return
    if row.operation in source_interfaces.UPLOADS:
        source_upload.release(store, workspace.directory, row.service_id, row.native_owner, live)
        return
    if row.operation == "convert_cozytensors":
        state = store.derived_lookup(row.native_owner)
        if state.get("state") == "committed":
            store.derived_dispose(row.native_owner)
        elif state.get("state") == "open":
            epoch = state.get("writer_session_id")
            if epoch:
                store.derived_fence(row.native_owner, epoch)
            store.derived_abandon(row.native_owner)
        store.release_model_source(row.service_id)
    elif store.tree_root(row.native_owner) is not None:
        store.release_tree_root(row.native_owner)


def release(workspace: Workspace, owner: str, service_id: str) -> None:
    from . import workspace_memo

    with workspace.locked() as db:
        row = call_row(db, owner, service_id)
        if row is None or row.state == "released":
            return
        if row.state != "releasing":
            raise WorkspaceRefusal("native cleanup has no permanent release intent")
    if row.operation not in ("source_files", "commit_file", *source_interfaces.UPLOADS):
        workspace_memo.release_native_result(workspace, owner, service_id)
        workspace_memo.release_native_lookup(workspace, owner, service_id)
    store = fill.store(workspace.store_root)
    final = "released"
    if row.operation in ("commit_file", "source_files"):
        from . import workspace_byte_outputs

        workspace_byte_outputs.release_native(workspace, owner, service_id)
    elif (
        row.operation in _PARTIAL_OPERATIONS
        and not row.result
        and row.computation_digest
        and _has_progress(store, row)
    ) or (
        row.operation in source_interfaces.UPLOADS
        and source_upload.retire(store, workspace.directory, service_id, row.native_owner)
    ):
        final = "stopped"
    else:
        with workspace.locked() as db:
            live = live_uploads(db)
        _release_native(store, owner, row, workspace, live)
    with workspace.locked() as db:
        db.execute(
            "UPDATE native_calls SET state=?,retired_ms=CASE WHEN ?<>'stopped' THEN 0 "
            "WHEN retired_ms>0 THEN retired_ms ELSE ? END WHERE owner=? AND service_id=? "
            "AND state='releasing'",
            (final, final, now_ms(), owner, service_id),
        )


def spend(db: Journal, workspace: Workspace, row: NativeCall) -> None:
    live = live_uploads(db)
    store = fill.store(workspace.store_root)
    db.native(partial(_release_native, store, row.owner, row, workspace, live))
    db.execute(
        "UPDATE native_calls SET state='released' WHERE owner=? AND service_id=? "
        "AND state='stopped'",
        (row.owner, row.service_id),
    )


def consume_partial(workspace: Workspace, owner: str, service_id: str) -> bool:
    """Spend a retired partial that a later call adopted."""
    with workspace.locked() as db:
        found = db.one(
            NativeCall, PARTIALS + " AND n.owner=? AND n.service_id=?", (owner, service_id)
        )
        if found is None:
            return False
        spend(db, workspace, found)
    return True
