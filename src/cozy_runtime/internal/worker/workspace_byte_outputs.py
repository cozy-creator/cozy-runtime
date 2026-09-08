"""Ordinary output trees use the workspace journal and native tree custody."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill
from cozy_runtime.internal.worker.workspace import Journal, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

TREE_MIME = "application/vnd.cozy.tree-manifest"
MAX_RESULTS = 32
SCHEMA5 = """
CREATE TABLE byte_outputs (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, request TEXT NOT NULL,
 ordinal INTEGER NOT NULL, spec BLOB NOT NULL, slot TEXT NOT NULL,
 manifest_body BLOB NOT NULL, content_bytes INTEGER NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('intent','complete','releasing','released')),
 receipt BLOB NOT NULL DEFAULT x'', native_digest BLOB NOT NULL DEFAULT x'',
 UNIQUE(owner,request,ordinal,spec,slot)
) STRICT;
"""


def identity(owner: str, request: str, ordinal: int, spec: bytes, slot: str) -> str:
    return documents.spell(
        hashlib.sha256(
            canonical_json.encode(["byte-output", owner, request, ordinal, spec.hex(), slot])
        ).digest()
    )


def reference(row: Mapping[str, Any]) -> pb.NativeByteTreeRef:
    if not row["receipt"] or len(row["native_digest"]) != 32:
        raise WorkspaceRefusal("ordinary output has no verified native receipt")
    body = row["manifest_body"]
    return pb.NativeByteTreeRef(
        producer_root_id=row["id"],
        receipt_digest=row["native_digest"],
        manifest=pb.Ref(digest=hashlib.sha256(body).digest(), length=len(body)),
        content_bytes=row["content_bytes"],
    )


def manifest_members(body: bytes, max_bytes: int) -> list[dict[str, Any]]:
    """Read capacity facts; native import remains the manifest validation authority."""
    if not 0 < len(body) <= 1 << 20 or max_bytes < 0:
        raise WorkspaceRefusal("ordinary output manifest exceeds its metadata bound")
    value = canonical_json.decode(body)
    if not isinstance(value, dict) or set(value) != {"entries"}:
        raise WorkspaceRefusal("ordinary output requires one file manifest")
    entries = value["entries"]
    if not isinstance(entries, list) or not entries:
        raise WorkspaceRefusal("ordinary output has no manifest members")
    total = 0
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or entry.get("kind") != "file"
            or not isinstance(entry.get("blob"), dict)
            or type(entry["blob"].get("length")) is not int
            or entry["blob"]["length"] < 0
        ):
            raise WorkspaceRefusal("ordinary output contains a non-file member")
        total += entry["blob"]["length"]
        if total > max_bytes:
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
) -> pb.NativeByteTreeRef:
    entries = manifest_members(manifest_body, max_bytes)
    content_bytes = sum(entry["blob"]["length"] for entry in entries)
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
        row = db.execute("SELECT * FROM byte_outputs WHERE id=?", (root_id,)).fetchone()
        if row is None:
            count = db.execute(
                "SELECT count(*) FROM byte_outputs WHERE owner=? AND request=? AND ordinal=?",
                (owner, request, ordinal),
            ).fetchone()[0]
            if count >= MAX_RESULTS or not slot or len(slot.encode()) > 1024:
                raise WorkspaceRefusal("ordinary output slot inventory exceeds its bound")
            db.execute(
                "INSERT INTO byte_outputs(id,owner,request,ordinal,spec,slot,manifest_body,"
                "content_bytes,state) VALUES(?,?,?,?,?,?,?,?,'intent')",
                (root_id, owner, request, ordinal, spec, slot, manifest_body, content_bytes),
            )
        elif row["manifest_body"] != manifest_body or row["state"] in ("releasing", "released"):
            raise WorkspaceRefusal("ordinary output changed or was permanently released")

        def produce() -> dict[str, Any]:
            store = fill.tensorfs_module().Store.open(str(workspace.store_root))
            # Native holds its writer guard and records each member before import.
            # Separate put_file/create_tree_root calls leave a process-death/GC gap.
            return dict(store.import_tree(root_id, manifest_body, [(p, str(f)) for p, f in files]))

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
        row = db.execute("SELECT * FROM byte_outputs WHERE id=?", (root_id,)).fetchone()
        attempt = db.execute(
            "SELECT state,fenced FROM attempts WHERE owner=? AND request=? AND ordinal=?",
            (owner, request, ordinal),
        ).fetchone()
        if (
            row["state"] in ("releasing", "released")
            or attempt is None
            or attempt["state"] != "running"
            or attempt["fenced"]
        ):
            db.execute("UPDATE byte_outputs SET state='releasing' WHERE id=?", (root_id,))
            db.native(lambda: release(workspace, owner, root_id))
            raise WorkspaceRefusal("ordinary output execution ended during native import")
        digest = hashlib.sha256(receipt).digest()
        if row["receipt"] not in (b"", receipt):
            raise WorkspaceRefusal("ordinary output native receipt changed on replay")
        db.execute(
            "UPDATE byte_outputs SET state='complete',receipt=?,native_digest=? WHERE id=?",
            (receipt, digest, root_id),
        )
        return reference(
            dict(db.execute("SELECT * FROM byte_outputs WHERE id=?", (root_id,)).fetchone())
        )


def owned(db: Journal, owner: str, source: pb.NativeByteTreeRef, *, released: bool = False) -> bool:
    row = db.execute(
        "SELECT * FROM byte_outputs WHERE owner=? AND id=?", (owner, source.producer_root_id)
    ).fetchone()
    if row is None or row["state"] not in (
        ("complete", "releasing", "released") if released else ("complete",)
    ):
        return False
    return reference(dict(row)) == source


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
        store = fill.tensorfs_module().Store.open(str(workspace.store_root))
        body = db.native(
            lambda: bytes(store.manifest(documents.spell(source.manifest.digest))["manifest"])
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


def check_manifest(source: pb.NativeByteTreeRef, body: bytes) -> list[dict[str, Any]]:
    if (
        len(body) != source.manifest.length
        or hashlib.sha256(body).digest() != source.manifest.digest
    ):
        raise WorkspaceRefusal("ordinary artifact manifest identity changed")
    members = manifest_members(body, source.content_bytes)
    if sum(item["blob"]["length"] for item in members) != source.content_bytes:
        raise WorkspaceRefusal("ordinary artifact content byte count changed")
    return members


@contextmanager
def leased(
    workspace: Workspace, owner: str, request: pb.NativeByteRetentionRequest
) -> Iterator[tuple[Any, Any, list[dict[str, Any]]]]:
    """Acquire the native read lease while checking one independently held recipient."""
    source = request.source
    lease = None
    try:
        with workspace.locked() as db:

            def held() -> bool:
                row = db.execute(
                    "SELECT * FROM holds WHERE id=? AND owner=?", (request.retention_id, owner)
                ).fetchone()
                return row is not None and (
                    row["state"] == "held"
                    and row["kind"] == "tree"
                    and row["transaction_id"] == source.producer_root_id
                    and row["native_digest"] == source.receipt_digest
                    and row["manifest"] == source.manifest.digest
                    and row["manifest_length"] == source.manifest.length
                )

            if not held():
                raise WorkspaceRefusal("ordinary artifact has no exact received retention")
            store = fill.tensorfs_module().Store.open(str(workspace.store_root))
            root = db.native(lambda: store.tree_root(request.retention_id))
            if (
                not root
                or not root["complete"]
                or root["released"]
                or root["receipt_digest"] != documents.spell(source.receipt_digest)
            ):
                raise WorkspaceRefusal("ordinary artifact recipient root is unavailable")
            manifest = documents.spell(source.manifest.digest)
            lease = db.native(lambda: store.acquire_manifest(manifest))
            body = db.native(lambda: bytes(store.manifest(manifest)["manifest"]))
            members = check_manifest(source, body)
            if not held():
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
        db.native(
            lambda: (
                fill.tensorfs_module()
                .Store.open(str(workspace.store_root))
                .release_tree_root(root_id)
            )
        )
        db.execute(
            "UPDATE byte_outputs SET state='released' WHERE owner=? AND id=?", (owner, root_id)
        )
