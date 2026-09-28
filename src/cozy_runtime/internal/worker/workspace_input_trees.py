"""Bounded input intake: exact client bytes become native custody before root acceptance."""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
from collections.abc import Iterable
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill, storage_admission
from cozy_runtime.protocol import documents, weights_limits
from cozy_runtime.protocol import worker_pb2 as pb

from .workspace import Journal, NativeHold, Workspace, WorkspaceBusy, WorkspaceRefusal
from .workspace_byte_outputs import FileMember, TreeManifest, manifest_body, manifest_members

MAX_CONTENT_BYTES = 256 << 20
SCHEMA10 = """
CREATE TABLE input_tree_intakes (
 id TEXT PRIMARY KEY, owner TEXT NOT NULL, request TEXT NOT NULL, input_id TEXT NOT NULL,
 header BLOB NOT NULL, manifest BLOB NOT NULL, manifest_length INTEGER NOT NULL,
 manifest_body BLOB NOT NULL, content_bytes INTEGER NOT NULL, intake_hold TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'receiving', native_digest BLOB NOT NULL DEFAULT x'',
 receipt BLOB NOT NULL DEFAULT x'', UNIQUE(owner,request,input_id)
) STRICT;
"""


class Intake(msgspec.Struct, frozen=True, kw_only=True):
    """One ``input_tree_intakes`` row."""

    id: str
    owner: str
    request: str
    input_id: str
    header: bytes
    manifest: bytes
    manifest_length: int
    manifest_body: bytes
    content_bytes: int
    intake_hold: str
    state: Literal["receiving", "prepared", "aborting", "released"] = "receiving"
    native_digest: bytes = b""
    receipt: bytes = b""


def _header(header: pb.InputTreeImportHeader) -> tuple[bytes, tuple[FileMember, ...]]:
    body = header.manifest_canonical_bytes
    if (
        re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", header.request_id) is None
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,1023}", header.input_id) is None
        or header.input_id == "payload"
        or len(body) > weights_limits.MAX_INPUT_TREE_MANIFEST_BYTES
        or header.content_bytes > MAX_CONTENT_BYTES
        or header.manifest.length != len(body)
        or hashlib.sha256(body).digest() != header.manifest.digest
    ):
        raise WorkspaceRefusal("input tree header has invalid identity or capacity")
    members = manifest_members(body, header.content_bytes)
    if manifest_body(members) != body:
        raise WorkspaceRefusal("input tree manifest is not canonical")
    names: set[str] = set()
    blobs: dict[str, int] = {}
    total = 0
    previous = ""
    for row in members:
        blob, name = row.blob, row.path
        if (
            not name
            or name == "."
            or "\\" in name
            or "\x00" in name
            or PurePosixPath(name).is_absolute()
            or str(PurePosixPath(name)) != name
            or any(part in (".", "..") for part in PurePosixPath(name).parts)
            or name in names
            or name <= previous
        ):
            raise WorkspaceRefusal("input tree manifest has an unsafe or repeated path")
        if blobs.setdefault(blob.sha256, blob.length) != blob.length:
            raise WorkspaceRefusal("input tree object length changed")
        names.add(name)
        previous = name
        total += blob.length
    for name in names:
        if any(str(parent) in names for parent in PurePosixPath(name).parents):
            raise WorkspaceRefusal("input tree file shadows a member directory")
    if total != header.content_bytes:
        raise WorkspaceRefusal("input tree changed its complete content size")
    copied = pb.InputTreeImportHeader()
    copied.CopyFrom(header)
    copied.ClearField("claim")
    return copied.SerializeToString(deterministic=True), members


def _row(db: Journal, owner: str, identity: str) -> Intake:
    row = db.one(
        Intake, "SELECT * FROM input_tree_intakes WHERE id=? AND owner=?", (identity, owner)
    )
    if row is None:
        raise WorkspaceRefusal("input tree intake is absent")
    return row


def _source(row: Intake) -> pb.NativeByteTreeRef:
    return pb.NativeByteTreeRef(
        producer_root_id=row.id,
        receipt_digest=row.native_digest,
        manifest=pb.Ref(digest=row.manifest, length=row.manifest_length),
        content_bytes=row.content_bytes,
    )


def owned(db: Journal, owner: str, source: pb.NativeByteTreeRef, *, released: bool = False) -> bool:
    row = db.one(
        Intake,
        "SELECT * FROM input_tree_intakes WHERE owner=? AND id=?",
        (owner, source.producer_root_id),
    )
    return (
        row is not None
        and row.state in (("prepared", "aborting", "released") if released else ("prepared",))
        and _source(row) == source
    )


def _observe(workspace: Workspace, db: Journal, row: Intake, observed: dict[str, Any]) -> Intake:
    receipt = bytes(observed.get("receipt", b""))
    if (
        not observed.get("complete")
        or observed.get("producer") != row.id
        or observed.get("manifest_digest") != documents.spell(row.manifest)
        or observed.get("manifest_length") != row.manifest_length
        or not receipt
        or observed.get("receipt_digest") != documents.spell(hashlib.sha256(receipt).digest())
        or row.receipt not in (b"", receipt)
    ):
        raise WorkspaceRefusal("native input tree differs from its accepted intake")
    db.execute(
        "UPDATE input_tree_intakes SET receipt=?,native_digest=? WHERE id=?",
        (receipt, hashlib.sha256(receipt).digest(), row.id),
    )
    return _row(db, row.owner, row.id)


def _recover(workspace: Workspace, db: Journal, row: Intake) -> Intake:
    observed = db.native(lambda: fill.store(workspace.store_root).tree_root(row.id))
    if observed is not None:
        row = _observe(workspace, db, row, dict(observed))
    return row


def _abort(workspace: Workspace, db: Journal, row: Intake) -> pb.NativeByteRetentionResult:
    row = _recover(workspace, db, row)
    source = _source(row) if row.receipt else None
    if source is not None:
        if (
            db.execute(
                "SELECT 1 FROM executions WHERE owner=? AND request=?",
                (row.owner, row.request),
            ).fetchone()
            is None
        ):
            from . import machine_models

            # Admission checks the abort tombstone in its own SQL transaction.
            # Fence this unaccepted root's pending recipient even if its native
            # retain crossed the abort; an already accepted root keeps its hold.
            workspace._change_hold(
                db,
                row.owner,
                pb.DerivedRetentionRequest(
                    weights_transaction_id=row.id,
                    tensorfs_receipt_digest=row.native_digest,
                    retention_id=machine_models.byte_retention_id(
                        row.owner, row.request, "input/" + row.input_id, source
                    ),
                ),
                release=True,
                kind="tree",
            )
        hold = db.execute("SELECT 1 FROM holds WHERE id=?", (row.intake_hold,)).fetchone()
        if hold is not None:
            workspace._change_hold(
                db,
                row.owner,
                pb.DerivedRetentionRequest(
                    weights_transaction_id=row.id,
                    tensorfs_receipt_digest=row.native_digest,
                    retention_id=row.intake_hold,
                ),
                release=True,
                kind="tree",
            )
        db.native(lambda: fill.store(workspace.store_root).release_tree_root(row.id))
    db.execute("UPDATE input_tree_intakes SET state='released' WHERE id=?", (row.id,))
    return pb.NativeByteRetentionResult(source=source, retention_id=row.intake_hold, released=True)


def _finish(
    workspace: Workspace, owner: str, identity: str, files: list[tuple[str, Path]]
) -> pb.NativeByteRetentionResult:
    with workspace.locked() as db:
        row = _recover(workspace, db, _row(db, owner, identity))
        if row.state in ("aborting", "released"):
            return _abort(workspace, db, row)
        if not row.receipt:
            observed = db.native(
                lambda: dict(
                    fill.store(workspace.store_root).import_tree(
                        identity, row.manifest_body, [(name, str(path)) for name, path in files]
                    )
                )
            )
            row = _observe(workspace, db, _row(db, owner, identity), observed)
            if row.state in ("aborting", "released"):
                return _abort(workspace, db, row)
        db.execute("UPDATE input_tree_intakes SET state='prepared' WHERE id=?", (identity,))
        held = db.one(NativeHold, "SELECT * FROM holds WHERE id=?", (row.intake_hold,))
        released = held is not None and held.state in ("releasing", "released")
        if not released:
            workspace._change_hold(
                db,
                owner,
                pb.DerivedRetentionRequest(
                    weights_transaction_id=identity,
                    tensorfs_receipt_digest=row.native_digest,
                    retention_id=row.intake_hold,
                ),
                release=False,
                kind="tree",
            )
        row = _row(db, owner, identity)
        if row.state in ("aborting", "released"):
            return _abort(workspace, db, row)
        db.native(lambda: fill.store(workspace.store_root).release_tree_root(identity))
        row = _row(db, owner, identity)
        if row.state in ("aborting", "released"):
            return _abort(workspace, db, row)
        return pb.NativeByteRetentionResult(
            source=_source(row), retention_id=row.intake_hold, released=released
        )


def receive(
    workspace: Workspace,
    owner: str,
    header: pb.InputTreeImportHeader,
    frames: Iterable[pb.InputTreeImportFrame],
) -> pb.NativeByteRetentionResult:
    raw, members = _header(header)
    identity = canonical_json.digest(
        ["cozy.input-tree/1", owner, header.request_id, header.input_id]
    )
    retention = canonical_json.digest(["cozy.input-tree-recipient/1", identity])
    with workspace.locked() as db:
        existing = db.one(Intake, "SELECT * FROM input_tree_intakes WHERE id=?", (identity,))
        if existing is None:
            if (
                db.execute(
                    "SELECT 1 FROM executions WHERE owner=? AND request=?",
                    (owner, header.request_id),
                ).fetchone()
                is not None
            ):
                raise WorkspaceRefusal("accepted root cannot add another input intake")
            count = db.execute(
                "SELECT count(*) FROM input_tree_intakes WHERE owner=? AND request=?",
                (owner, header.request_id),
            ).fetchone()[0]
            if count >= weights_limits.MAX_CHILD_ARTIFACT_GRANTS:
                raise WorkspaceRefusal("root input intake exceeds its field bound")
            db.execute(
                "INSERT INTO input_tree_intakes(id,owner,request,input_id,header,manifest,"
                "manifest_length,manifest_body,content_bytes,intake_hold) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    identity,
                    owner,
                    header.request_id,
                    header.input_id,
                    raw,
                    header.manifest.digest,
                    len(header.manifest_canonical_bytes),
                    header.manifest_canonical_bytes,
                    header.content_bytes,
                    retention,
                ),
            )
        elif existing.header != raw:
            raise WorkspaceRefusal("input tree replay changed its accepted subject")
    iterator = iter(frames)
    first = next(iterator, None)
    abort = first is not None and first.WhichOneof("body") == "commit" and first.commit.abort
    if abort:
        with workspace.locked() as db:
            db.execute(
                "UPDATE input_tree_intakes SET state='aborting' WHERE id=? AND state!='released'",
                (identity,),
            )
    root = workspace.directory / "input-intakes" / identity.removeprefix("sha256:")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = os.open(root / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WorkspaceBusy("input tree intake is already active") from None
        if abort:
            if next(iterator, None) is not None:
                raise WorkspaceRefusal("input tree abort must be the final frame")
            with workspace.locked() as db:
                return _abort(workspace, db, _row(db, owner, identity))
        with workspace.locked() as db:
            row = _recover(workspace, db, _row(db, owner, identity))
            if row.state in ("aborting", "released"):
                raise WorkspaceRefusal("aborted input tree cannot be reopened")
            complete = bool(row.receipt)
        stage = root / "staging"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(mode=0o700)
        try:
            with storage_admission.admit(
                storage_admission.native_write(
                    workspace.store_root, header.content_bytes * 2, len(members) * 2
                )
            ):
                objects = {row.blob.sha256: row.blob.length for row in members}
                _receive_objects(
                    workspace,
                    owner,
                    identity,
                    chain(() if first is None else (first,), iterator),
                    objects,
                    stage,
                    complete=complete,
                )
                files = [(row.path, stage / row.blob.sha256) for row in members]
                return _finish(workspace, owner, identity, files)
        finally:
            shutil.rmtree(stage)
    finally:
        os.close(lock)


def _receive_objects(
    workspace: Workspace,
    owner: str,
    identity: str,
    frames: Iterable[pb.InputTreeImportFrame],
    objects: dict[str, int],
    stage: Path,
    *,
    complete: bool,
) -> None:
    seen: set[str] = set()
    current, offset, committed = "", 0, False
    output = None
    digest = hashlib.sha256()
    try:
        for frame in frames:
            with workspace.locked() as db:
                if _row(db, owner, identity).state in ("aborting", "released"):
                    raise WorkspaceRefusal("input tree intake was aborted")
            kind = frame.WhichOneof("body")
            if committed or kind not in ("blob", "commit"):
                raise WorkspaceRefusal("input tree stream changed its header or final commit")
            if kind == "commit":
                if frame.commit.abort or output is not None:
                    raise WorkspaceRefusal("input tree commit omitted a complete declared object")
                if not complete and seen != set(objects):
                    _reuse(workspace, owner, identity, set(objects) - seen, stage)
                committed = True
                continue
            blob = frame.blob
            selected = blob.object.digest.hex()
            if (
                len(blob.object.digest) != 32
                or objects.get(selected) != blob.object.length
                or len(blob.data) > weights_limits.MAX_INPUT_TREE_CHUNK_BYTES
            ):
                raise WorkspaceRefusal("input tree stream has an undeclared or oversized object")
            if output is None:
                if selected in seen or blob.offset != 0:
                    raise WorkspaceRefusal("input tree object repeats or omits its first bytes")
                current, offset, digest = selected, 0, hashlib.sha256()
                output = (stage / selected).open("xb")
            if (
                selected != current
                or blob.offset != offset
                or len(blob.data) > objects[selected] - offset
                or (not blob.data and objects[selected] != 0)
            ):
                raise WorkspaceRefusal("input tree object changed its contiguous byte range")
            output.write(blob.data)
            digest.update(blob.data)
            offset += len(blob.data)
            if offset == objects[selected]:
                output.close()
                output = None
                if digest.hexdigest() != selected:
                    raise WorkspaceRefusal("input tree object failed its declared content digest")
                seen.add(selected)
        if not committed:
            raise WorkspaceRefusal("input tree stream ended before its final commit")
    finally:
        if output is not None:
            output.close()


def _reuse(workspace: Workspace, owner: str, identity: str, wanted: set[str], stage: Path) -> None:
    """Stage objects the commit omitted from this owner's earlier inputs holding them.

    Only the same owner's own intakes are searched, so a commit never learns whether another
    owner holds some bytes. An object no earlier input still holds is refused typed, and the
    sender sends its bytes."""
    store = fill.store(workspace.store_root)
    for digest in sorted(wanted):
        with workspace.locked() as db:
            rows = db.all(
                Intake,
                "SELECT * FROM input_tree_intakes WHERE owner=? AND id<>? "
                "AND instr(manifest_body,?)>0 ORDER BY rowid DESC LIMIT 8",
                (owner, identity, digest.encode()),
            )
        staged = stage / digest
        for row in rows:
            member = next(
                (
                    entry
                    for entry in canonical_json.decode_as(row.manifest_body, TreeManifest).entries
                    if entry.blob.sha256 == digest
                ),
                None,
            )
            if member is None:
                continue
            try:
                store.materialize(row.manifest.hex(), member.path, staged)
            except Exception:  # an earlier input released and collected: try another
                staged.unlink(missing_ok=True)
                continue
            with staged.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() == digest:
                    break
            staged.unlink()
        else:
            raise WorkspaceRefusal(f"input_objects_absent: {len(wanted)} object(s) must be sent")


def verify_adoption(db: Journal, owner: str, offer: pb.AttemptOffer) -> None:
    """The input abort and root acceptance have one durable ordering boundary."""
    for access in offer.grant.inputs:
        if not access.HasField("native_tree"):
            continue
        row = db.one(
            Intake,
            "SELECT * FROM input_tree_intakes WHERE owner=? AND id=?",
            (owner, access.native_tree.source.producer_root_id),
        )
        if (
            row is not None
            and row.intake_hold == access.native_tree.retention_id
            and (
                row.state != "prepared"
                or row.request != offer.request_id
                or row.input_id != access.input_id
            )
        ):
            raise WorkspaceRefusal(
                "root input intake was aborted or belongs to another request field"
            )
