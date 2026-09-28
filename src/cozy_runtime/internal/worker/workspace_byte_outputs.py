"""Ordinary output trees use the workspace journal and native tree custody."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill, storage_admission
from cozy_runtime.internal.worker import workspace_sources
from cozy_runtime.internal.worker.workspace import (
    Journal,
    NativeHold,
    Workspace,
    WorkspaceRefusal,
    same_ref,
)
from cozy_runtime.internal.worker.workspace_sources import NativeCall
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

TREE_MIME = "application/vnd.cozy.tree-manifest"
MAX_RESULTS = 32
#: Distinct byte strings one attempt may publish as products (parts included).
MAX_PRODUCTS = 4096


class ByteOutput(msgspec.Struct, frozen=True, kw_only=True):
    """One ``byte_outputs`` row."""

    id: str
    owner: str
    request: str
    ordinal: int
    spec: bytes
    slot: str
    manifest_body: bytes
    content_bytes: int
    state: Literal["intent", "complete", "releasing", "released"] = "intent"
    receipt: bytes = b""
    native_digest: bytes = b""
    native_service_id: str = ""


class Blob(msgspec.Struct, frozen=True):
    sha256: Annotated[str, msgspec.Meta(pattern="^[0-9a-f]{64}$")]
    length: Annotated[int, msgspec.Meta(ge=0)]


class FileMember(msgspec.Struct, frozen=True):
    kind: Literal["file"]
    path: str
    blob: Blob


class TreeManifest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """An ordinary output's file manifest; native import stays its validation authority."""

    entries: Annotated[tuple[FileMember, ...], msgspec.Meta(min_length=1)]


def manifest_body(entries: Sequence[FileMember]) -> bytes:
    return canonical_json.encode(msgspec.to_builtins(TreeManifest(tuple(entries))))


def payload_manifest(blob: Blob) -> bytes:
    """The manifest of one file, stored as `payload`."""
    return manifest_body([FileMember("file", "payload", blob)])


SCHEMA5 = """
CREATE TABLE byte_outputs (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, request TEXT NOT NULL,
 ordinal INTEGER NOT NULL, spec BLOB NOT NULL, slot TEXT NOT NULL,
 manifest_body BLOB NOT NULL, content_bytes INTEGER NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('intent','complete','releasing','released')),
 receipt BLOB NOT NULL DEFAULT x'', native_digest BLOB NOT NULL DEFAULT x'',
 native_service_id TEXT NOT NULL DEFAULT '',
 UNIQUE(owner,request,ordinal,spec,slot)
) STRICT;
CREATE UNIQUE INDEX byte_outputs_native_service ON byte_outputs(owner,native_service_id)
 WHERE native_service_id<>'';
"""


def native_slot(operation: str, index: int) -> str:
    if operation not in ("source_files", "commit_file") or not 0 <= index < 1 << 32:
        raise WorkspaceRefusal("unknown native byte operation")
    return f"runtime.{operation}.{index}"


def native_call(db: Journal, owner: str, command: pb.NativeSourceCommand) -> NativeCall:
    call = command.parent_call
    workspace_sources._parent(db, owner, call)
    row = workspace_sources.call_row(db, owner, command.service_id)
    if (
        row is None
        or row.operation not in ("source_files", "commit_file")
        or row.operation != call.export
        or row.parent_request != call.parent_request_id
        or row.call_index != call.call_index
        or row.parent_ordinal != call.parent_attempt_ordinal
        or row.parent_spec != call.parent_invocation_spec_digest
        or row.intent_digest != call.intent_digest
        or row.state not in ("accepted", "executing", "complete")
    ):
        raise WorkspaceRefusal("native byte output has no exact current native call")
    return row


def native_value(row: ByteOutput, operation: str, accepted: bytes) -> dict[str, Any]:
    digest = documents.spell(hashlib.sha256(row.manifest_body).digest())
    if operation == "source_files":
        return {
            "files": {
                "asset_ref": digest,
                "kind": "tree",
                "digest": digest,
                "size_bytes": row.content_bytes,
            }
        }
    if operation == "commit_file":
        request = canonical_json.decode(accepted)
        return {
            "file": {
                "asset_ref": request["digest"],
                "kind": "file",
                "digest": request["digest"],
                "size_bytes": request["size_bytes"],
                "media_type": request["media_type"],
            }
        }
    raise WorkspaceRefusal("unknown native byte result")


def recorded_native(
    db: Journal, row: ByteOutput, *, complete: bool, released: bool = False
) -> bool:
    service = row.native_service_id
    if not service or row.id != identity(row.owner, row.request, row.ordinal, row.spec, row.slot):
        return False
    call = workspace_sources.call_row(db, row.owner, service)
    if (
        call is None
        or call.operation not in ("source_files", "commit_file")
        or call.parent_request != row.request
        or native_slot(call.operation, call.call_index) != row.slot
    ):
        return False
    args = canonical_json.decode(call.accepted)
    if call.operation == "source_files":
        if not same_ref(
            args["source"]["manifest"],
            documents.spell(hashlib.sha256(row.manifest_body).digest()),
            len(row.manifest_body),
        ):
            return False
    elif not _single_payload(
        row.manifest_body,
        args["digest"].removeprefix("sha256:"),
        args["size_bytes"],
    ):
        return False
    if not complete:
        return True
    states = ("complete", "releasing", "released") if released else ("complete",)
    return (
        row.state in states
        and call.state in states
        and call.native_receipt == row.receipt
        and same_result(call.result, native_value(row, call.operation, call.accepted))
    )


def _single_payload(body: bytes, sha256: str, length: int) -> bool:
    """One ``payload`` file of the accepted blob; additive member fields are ignored."""
    try:
        entries = canonical_json.decode_as(body, TreeManifest).entries
    except ValueError:
        return False
    return entries == (FileMember("file", "payload", Blob(sha256, length)),)


_RESULT_IDENTITY = ("kind", "digest", "size_bytes")


def same_result(stored: bytes, expected: Mapping[str, Any]) -> bool:
    """Equal on every member both versions write, with the byte identity always present."""
    try:
        value = canonical_json.decode(stored)
    except ValueError:
        return False
    if not isinstance(value, dict):
        return False
    for name, want in expected.items():
        got = value.get(name)
        if (
            not isinstance(got, dict)
            or any(key not in got for key in _RESULT_IDENTITY)
            or any(got[key] != want[key] for key in got.keys() & want.keys())
        ):
            return False
    return True


def reserve_native(
    workspace: Workspace,
    owner: str,
    command: pb.NativeSourceCommand,
    manifest_body: bytes,
    max_bytes: int,
) -> ByteOutput:
    """Reserve once by accepted native call; later attempts keep the real producer."""
    members = manifest_members(manifest_body, max_bytes)
    total = sum(member.blob.length for member in members)
    with workspace.locked() as db:
        call = native_call(db, owner, command)
        slot = native_slot(call.operation, call.call_index)
        row = db.one(
            ByteOutput,
            "SELECT * FROM byte_outputs WHERE owner=? AND native_service_id=?",
            (owner, command.service_id),
        )
        if row is None:
            count = db.execute(
                "SELECT count(*) FROM byte_outputs WHERE owner=? AND request=? AND ordinal=? "
                "AND substr(slot,1,8)<>'product.'",
                (owner, call.parent_request, call.parent_ordinal),
            ).fetchone()[0]
            if count >= MAX_RESULTS:
                raise WorkspaceRefusal("native byte output exceeds the parent output inventory")
            row = ByteOutput(
                id=identity(
                    owner, call.parent_request, call.parent_ordinal, call.parent_spec, slot
                ),
                owner=owner,
                request=call.parent_request,
                ordinal=call.parent_ordinal,
                spec=call.parent_spec,
                slot=slot,
                manifest_body=manifest_body,
                content_bytes=total,
                native_service_id=command.service_id,
            )
            if not recorded_native(db, row, complete=False):
                raise WorkspaceRefusal("native byte manifest differs from its accepted call")
            db.execute(
                "INSERT INTO byte_outputs(id,owner,request,ordinal,spec,slot,manifest_body,"
                "content_bytes,state,native_service_id) VALUES(?,?,?,?,?,?,?,?,'intent',?)",
                (
                    row.id,
                    owner,
                    row.request,
                    row.ordinal,
                    row.spec,
                    slot,
                    manifest_body,
                    total,
                    command.service_id,
                ),
            )
        if (
            row.state not in ("intent", "complete")
            or row.manifest_body != manifest_body
            or row.content_bytes != total
            or not recorded_native(db, row, complete=False)
        ):
            raise WorkspaceRefusal("native byte output changed or was permanently released")
        return row


def complete_native(
    workspace: Workspace,
    owner: str,
    command: pb.NativeSourceCommand,
    result: bytes,
    computation: bytes,
    receipt: bytes,
    root_id: str,
) -> tuple[pb.NativeByteTreeRef, int, bytes]:
    """Verify native custody under current authority without rewriting producer history."""
    with workspace.locked() as db:
        call = native_call(db, owner, command)
        row = db.one(
            ByteOutput,
            "SELECT * FROM byte_outputs WHERE owner=? AND native_service_id=? AND id=?",
            (owner, command.service_id, root_id),
        )
        if (
            row is None
            or row.state not in ("intent", "complete")
            or not recorded_native(db, row, complete=False)
            or not same_result(result, native_value(row, call.operation, call.accepted))
            or len(computation) != 32
            or call.computation_digest != computation
            or not 0 < len(receipt) <= 1 << 20
        ):
            raise WorkspaceRefusal("native byte completion differs from its accepted intent")
        store = fill.store(workspace.store_root)
        observed = db.native(lambda: store.tree_root(root_id))
        if (
            not observed
            or not observed["complete"]
            or observed["released"]
            or observed["producer"] != root_id
            or observed["receipt"] != receipt
            or observed["manifest_digest"]
            != documents.spell(hashlib.sha256(row.manifest_body).digest())
            or observed["manifest_length"] != len(row.manifest_body)
            or observed["receipt_digest"] != documents.spell(hashlib.sha256(receipt).digest())
        ):
            raise WorkspaceRefusal("native byte completion has no matching native receipt")
        native_call(db, owner, command)
        current = _row(db, root_id)
        if current.state not in ("intent", "complete") or current.receipt not in (b"", receipt):
            raise WorkspaceRefusal("native byte producer changed during completion")
        db.execute(
            "UPDATE byte_outputs SET state='complete',receipt=?,native_digest=? WHERE id=?",
            (receipt, hashlib.sha256(receipt).digest(), root_id),
        )
        db.execute(
            "UPDATE native_calls SET state='complete',result=?,native_receipt=?,native_error='' "
            "WHERE owner=? AND service_id=?",
            (result, receipt, owner, command.service_id),
        )
        current = _row(db, root_id)
        return reference(current), current.ordinal, current.spec


def replay_native(
    workspace: Workspace, owner: str, command: pb.NativeSourceCommand
) -> tuple[pb.NativeByteTreeRef, int, bytes]:
    with workspace.locked() as db:
        call = native_call(db, owner, command)
        row = db.one(
            ByteOutput,
            "SELECT * FROM byte_outputs WHERE owner=? AND native_service_id=?",
            (owner, command.service_id),
        )
        if row is None or not recorded_native(db, row, complete=True):
            raise WorkspaceRefusal("native byte replay has no completed producer")
        result, computation, receipt, root_id = (
            call.result,
            call.computation_digest,
            call.native_receipt,
            row.id,
        )
    return complete_native(workspace, owner, command, result, computation, receipt, root_id)


def release_native(workspace: Workspace, owner: str, service_id: str) -> None:
    with workspace.locked() as db:
        row = db.one(
            ByteOutput,
            "SELECT * FROM byte_outputs WHERE owner=? AND native_service_id=?",
            (owner, service_id),
        )
        if row is None or row.state == "released":
            return
        if not recorded_native(db, row, complete=False):
            raise WorkspaceRefusal("native byte release has no accepted producer")
        call = workspace_sources.call_row(db, owner, service_id)
        if call is None or call.state not in ("stopped", "releasing", "released"):
            raise WorkspaceRefusal("native byte release has no permanent stop intent")
        db.execute("UPDATE byte_outputs SET state='releasing' WHERE id=?", (row.id,))
        root_id = row.id
    release(workspace, owner, root_id)


def identity(owner: str, request: str, ordinal: int, spec: bytes, slot: str) -> str:
    return documents.spell(
        hashlib.sha256(
            canonical_json.encode(["byte-output", owner, request, ordinal, spec.hex(), slot])
        ).digest()
    )


def reference(row: ByteOutput) -> pb.NativeByteTreeRef:
    if not row.receipt or len(row.native_digest) != 32:
        raise WorkspaceRefusal("ordinary output has no verified native receipt")
    body = row.manifest_body
    return pb.NativeByteTreeRef(
        producer_root_id=row.id,
        receipt_digest=row.native_digest,
        manifest=pb.Ref(digest=hashlib.sha256(body).digest(), length=len(body)),
        content_bytes=row.content_bytes,
    )


def _row(db: Journal, root_id: str) -> ByteOutput:
    row = db.one(ByteOutput, "SELECT * FROM byte_outputs WHERE id=?", (root_id,))
    if row is None:
        raise WorkspaceRefusal("ordinary output row was removed")
    return row


def manifest_members(body: bytes, max_bytes: int) -> tuple[FileMember, ...]:
    """Read capacity facts; native import remains the manifest validation authority."""
    if not 0 < len(body) <= 1 << 20 or max_bytes < 0:
        raise WorkspaceRefusal("ordinary output manifest exceeds its metadata bound")
    try:
        entries = canonical_json.decode_as(body, TreeManifest).entries
    except ValueError as exc:
        raise WorkspaceRefusal(f"ordinary output requires one file manifest: {exc}") from exc
    if sum(entry.blob.length for entry in entries) > max_bytes:
        raise WorkspaceRefusal("ordinary output exceeds its granted content bytes")
    return entries


def commit(
    workspace: Workspace,
    owner: str,
    request: str,
    ordinal: int,
    spec: bytes,
    slot: str,
    manifest_body: bytes,
    files: Sequence[tuple[str, Path]],
    max_bytes: int,
    *,
    retained: pb.NativeByteRetentionRequest | None = None,
) -> pb.NativeByteTreeRef:
    if slot.startswith(("runtime.source_files.", "runtime.commit_file.")):
        raise WorkspaceRefusal("author outputs cannot enter the native byte namespace")
    entries = manifest_members(manifest_body, max_bytes)
    content_bytes = sum(entry.blob.length for entry in entries)
    root_id = identity(owner, request, ordinal, spec, slot)
    workspace.owner(owner)
    with workspace.locked() as db:
        attempt = db.execute(
            "SELECT state,spec,fenced FROM attempts WHERE owner=? AND request=? AND ordinal=?",
            (owner, request, ordinal),
        ).fetchone()
        if (
            attempt is None
            or attempt["state"] != "running"
            or attempt["spec"] != spec
            or attempt["fenced"]
        ):
            raise WorkspaceRefusal("ordinary output has no current accepted execution")
        row = db.one(ByteOutput, "SELECT * FROM byte_outputs WHERE id=?", (root_id,))
        if row is None:
            # A run's published products are one inventory, its returned outputs another.
            product = slot.startswith("product.")
            count = db.execute(
                "SELECT count(*) FROM byte_outputs WHERE owner=? AND request=? AND ordinal=? "
                "AND (substr(slot,1,8)='product.')=?",
                (owner, request, ordinal, product),
            ).fetchone()[0]
            bound = MAX_PRODUCTS if product else MAX_RESULTS
            if count >= bound or not slot or len(slot.encode()) > 1024:
                raise WorkspaceRefusal("ordinary output slot inventory exceeds its bound")
            db.execute(
                "INSERT INTO byte_outputs(id,owner,request,ordinal,spec,slot,manifest_body,"
                "content_bytes,state) VALUES(?,?,?,?,?,?,?,?,'intent')",
                (root_id, owner, request, ordinal, spec, slot, manifest_body, content_bytes),
            )
        elif (
            row.native_service_id
            or row.manifest_body != manifest_body
            or row.state in ("releasing", "released")
        ):
            raise WorkspaceRefusal("ordinary output changed or was permanently released")

        def produce() -> dict[str, Any]:
            if retained is not None:
                # The referenced producer remains the input. This current attempt
                # obtains a distinct ordinary output root over the same native bytes.
                with leased(workspace, owner, retained) as (store, _, _):
                    if (
                        retained.source.manifest.digest != hashlib.sha256(manifest_body).digest()
                        or retained.source.manifest.length != len(manifest_body)
                        or retained.source.content_bytes != content_bytes
                        or files
                    ):
                        raise WorkspaceRefusal("re-export changed its retained input manifest")
                    with storage_admission.admit(
                        storage_admission.native_write(workspace.store_root)
                    ):
                        return dict(
                            store.create_tree_root(
                                root_id,
                                documents.spell(retained.source.manifest.digest),
                                retained.source.manifest.length,
                            )
                        )
            store = fill.store(workspace.store_root)
            # Native holds its writer guard and records each member before import.
            # Separate put_file/create_tree_root calls leave a process-death/GC gap.
            with storage_admission.admit(
                storage_admission.native_write(workspace.store_root, content_bytes, len(files))
            ):
                return dict(
                    store.import_tree(root_id, manifest_body, [(p, str(f)) for p, f in files])
                )

        observed = db.native(produce)
        receipt = bytes(observed.get("receipt", b""))
        if (
            not observed.get("complete")
            or observed.get("released")
            or observed.get("producer") != root_id
            or observed.get("manifest_digest")
            != documents.spell(hashlib.sha256(manifest_body).digest())
            or observed.get("manifest_length") != len(manifest_body)
            or observed.get("receipt_digest") != documents.spell(hashlib.sha256(receipt).digest())
            or not receipt
        ):
            raise WorkspaceRefusal("native ordinary output disagrees with its accepted intent")
        row = _row(db, root_id)
        attempt = db.execute(
            "SELECT state,fenced FROM attempts WHERE owner=? AND request=? AND ordinal=?",
            (owner, request, ordinal),
        ).fetchone()
        if (
            row.state in ("releasing", "released")
            or attempt is None
            or attempt["state"] != "running"
            or attempt["fenced"]
        ):
            db.execute("UPDATE byte_outputs SET state='releasing' WHERE id=?", (root_id,))
            db.native(lambda: release(workspace, owner, root_id))
            raise WorkspaceRefusal("ordinary output execution ended during native import")
        digest = hashlib.sha256(receipt).digest()
        if row.receipt not in (b"", receipt):
            raise WorkspaceRefusal("ordinary output native receipt changed on replay")
        db.execute(
            "UPDATE byte_outputs SET state='complete',receipt=?,native_digest=? WHERE id=?",
            (receipt, digest, root_id),
        )
        return reference(_row(db, root_id))


def owned(db: Journal, owner: str, source: pb.NativeByteTreeRef, *, released: bool = False) -> bool:
    row = db.one(
        ByteOutput,
        "SELECT * FROM byte_outputs WHERE owner=? AND id=?",
        (owner, source.producer_root_id),
    )
    if row is None:
        from .workspace_input_trees import owned as input_owned

        return input_owned(db, owner, source, released=released)
    if row.state not in (("complete", "releasing", "released") if released else ("complete",)):
        return False
    if row.native_service_id and not recorded_native(db, row, complete=True, released=released):
        return False
    return reference(row) == source


def change_hold(
    workspace: Workspace, owner: str, request: pb.NativeByteRetentionRequest, *, release: bool
) -> pb.NativeByteRetentionResult:
    source = request.source
    if (
        len(source.receipt_digest) != 32
        or len(source.manifest.digest) != 32
        or not source.manifest.length
    ):
        raise WorkspaceRefusal("ordinary retention has an incomplete native identity")
    with workspace.locked() as db:
        original = owned(db, owner, source, released=release)
        retained = db.execute(
            "SELECT 1 FROM holds WHERE owner=? AND transaction_id=? AND native_digest=? "
            "AND manifest=? AND manifest_length=? AND kind='tree' AND state='held'",
            (
                owner,
                source.producer_root_id,
                source.receipt_digest,
                source.manifest.digest,
                source.manifest.length,
            ),
        ).fetchone()
        if not original and retained is None:
            raise WorkspaceRefusal("ordinary tree is not retained by this owner")
        store = fill.store(workspace.store_root)
        body = db.native(
            lambda: store.manifest(documents.spell(source.manifest.digest))["manifest"]
        )
        check_manifest(source, body)
        internal = pb.DerivedRetentionRequest(
            weights_transaction_id=source.producer_root_id,
            tensorfs_receipt_digest=source.receipt_digest,
            retention_id=request.retention_id,
        )
        result = workspace._change_hold(db, owner, internal, release=release, kind="tree")
        if result.manifest != source.manifest:
            raise WorkspaceRefusal("ordinary retention changed its manifest")
        return pb.NativeByteRetentionResult(
            source=source, retention_id=result.retention_id, released=result.released
        )


def check_manifest(source: pb.NativeByteTreeRef, body: bytes) -> tuple[FileMember, ...]:
    if (
        len(body) != source.manifest.length
        or hashlib.sha256(body).digest() != source.manifest.digest
    ):
        raise WorkspaceRefusal("ordinary artifact manifest identity changed")
    members = manifest_members(body, source.content_bytes)
    if sum(item.blob.length for item in members) != source.content_bytes:
        raise WorkspaceRefusal("ordinary artifact content byte count changed")
    return members


def _held(db: Journal, owner: str, request: pb.NativeByteRetentionRequest) -> bool:
    source = request.source
    row = db.one(
        NativeHold, "SELECT * FROM holds WHERE id=? AND owner=?", (request.retention_id, owner)
    )
    return row is not None and (
        row.state == "held"
        and row.kind == "tree"
        and row.transaction_id == source.producer_root_id
        and row.native_digest == source.receipt_digest
        and row.manifest == source.manifest.digest
        and row.manifest_length == source.manifest.length
    )


def _retained(
    db: Journal, workspace: Workspace, owner: str, request: pb.NativeByteRetentionRequest
) -> fill.Store:
    if not _held(db, owner, request):
        raise WorkspaceRefusal("ordinary artifact has no exact received retention")
    store = fill.store(workspace.store_root)
    root = db.native(lambda: store.tree_root(request.retention_id))
    if (
        not root
        or not root["complete"]
        or root["released"]
        or root["receipt_digest"] != documents.spell(request.source.receipt_digest)
    ):
        raise WorkspaceRefusal("ordinary artifact recipient root is unavailable")
    return store


def retained(
    workspace: Workspace, owner: str, request: pb.NativeByteRetentionRequest
) -> tuple[fill.Store, Callable[[], bool]]:
    """Check one independently held recipient without leasing its tree.

    Returns the store and a re-check of the hold. The retention keeps the tree from
    collection, so a reader leases only the objects it reads.
    """
    with workspace.locked() as db:
        store = _retained(db, workspace, owner, request)

    def held() -> bool:
        with workspace.locked() as db:
            return _held(db, owner, request)

    return store, held


@contextmanager
def leased(
    workspace: Workspace, owner: str, request: pb.NativeByteRetentionRequest
) -> Iterator[tuple[fill.Store, fill.ReadLease, tuple[FileMember, ...]]]:
    """Acquire the native read lease while checking one independently held recipient."""
    source = request.source
    lease = None
    try:
        with workspace.locked() as db:
            store = _retained(db, workspace, owner, request)
            manifest = documents.spell(source.manifest.digest)
            lease = db.native(lambda: store.acquire_manifest(manifest))
            body = db.native(lambda: store.manifest(manifest)["manifest"])
            members = check_manifest(source, body)
            if not _held(db, owner, request):
                raise WorkspaceRefusal(
                    "ordinary artifact was released while acquiring its read lease"
                )
        yield store, lease, members
    finally:
        if lease is not None:
            lease.release()


def release(workspace: Workspace, owner: str, root_id: str) -> None:
    with workspace.locked() as db:
        row = db.execute(
            "SELECT state FROM byte_outputs WHERE owner=? AND id=?", (owner, root_id)
        ).fetchone()
        if row is None or row["state"] not in ("releasing", "released"):
            raise WorkspaceRefusal("ordinary output release has no durable owner intent")
        try:
            db.native(lambda: fill.store(workspace.store_root).release_tree_root(root_id))
        except Exception as exc:
            # Cancellation can win after intent reservation but before native import.
            # Only this exact absence is already released; integrity errors still refuse.
            if getattr(exc, "code", None) != "ROOT_ABSENT":
                raise
        db.execute(
            "UPDATE byte_outputs SET state='released' WHERE owner=? AND id=?", (owner, root_id)
        )
