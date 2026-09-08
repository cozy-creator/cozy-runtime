"""Adopt exact stopped native work; a partial checkpoint is never a memo result."""

from __future__ import annotations

from typing import Any

from cozy_runtime.internal import fill
from cozy_runtime.internal.worker.workspace import Journal, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb


def adopt(workspace: Workspace, db: Journal, owner: str, transaction: str) -> dict[str, Any]:
    row = db.execute(
        "SELECT * FROM weights WHERE owner=? AND id=?", (owner, transaction)
    ).fetchone()
    if row is None:
        raise WorkspaceRefusal("partial recipient has no accepted writer intent")
    if row["state"] != "intent" or row["checkpoint"] or row["ready"]:
        return dict(row)
    attempt = db.execute(
        "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
        (owner, row["request"], row["ordinal"]),
    ).fetchone()
    if attempt is None or attempt["fenced"] or attempt["state"] not in ("accepted", "running"):
        raise WorkspaceRefusal("partial recipient attempt has stopped")
    if not attempt["memoize"]:
        return dict(row)
    if not row["adoption_source"]:
        candidates = db.execute(
            "SELECT w.* FROM weights w JOIN attempts a ON "
            "a.owner=w.owner AND a.request=w.request AND a.ordinal=w.ordinal "
            "WHERE w.owner=? AND w.id<>? AND w.slot=? AND w.state='intent' "
            "AND w.declaration_digest=? AND w.declaration=? AND w.checkpoint<>x'' "
            "AND a.memoize=1 AND (a.state='outcome' OR a.fenced=1) AND a.state<>'released' "
            "ORDER BY w.rowid DESC LIMIT 32",
            (owner, transaction, row["slot"], row["declaration_digest"], row["declaration"]),
        ).fetchall()
        if not candidates:
            return dict(row)
        source = max(
            candidates, key=lambda item: pb.CheckpointRef.FromString(item["checkpoint"]).bytes
        )
        db.execute(
            "UPDATE weights SET adoption_source=?,adoption_epoch=?,adoption_checkpoint=? "
            "WHERE id=?",
            (source["id"], source["epoch"], source["checkpoint"], transaction),
        )
        row = db.execute("SELECT * FROM weights WHERE id=?", (transaction,)).fetchone()

    def acquire() -> pb.CheckpointRef:
        store = fill.tensorfs_module().Store.open(str(workspace.store_root))
        checkpoint = pb.CheckpointRef.FromString(row["adoption_checkpoint"])
        # The captured epoch cannot fence a newer attempt that started while the
        # workspace lock was released. Native adoption also refuses an active writer.
        store.derived_fence(row["adoption_source"], row["adoption_epoch"])
        facts = store.adopt_derived_checkpoint(
            transaction,
            row["adoption_source"],
            row["declaration"],
            documents.spell(checkpoint.head.digest),
            checkpoint.head.length,
            operation_id=row["request"],
            slot=row["slot"],
        )
        return pb.CheckpointRef(
            head=pb.Ref(digest=documents.raw(facts["head"]), length=facts["head_length"]),
            plan_digest=documents.raw(facts["plan_digest"]),
            index=facts["index"],
            bytes=facts["bytes"],
        )

    checkpoint = db.native(acquire)
    current = db.execute("SELECT * FROM weights WHERE id=?", (transaction,)).fetchone()
    attempt = db.execute(
        "SELECT state,fenced FROM attempts WHERE owner=? AND request=? AND ordinal=?",
        (owner, row["request"], row["ordinal"]),
    ).fetchone()
    if (
        current is None
        or current["state"] != "intent"
        or current["ordinal"] != row["ordinal"]
        or current["epoch"] != row["epoch"]
        or current["adoption_source"] != row["adoption_source"]
        or attempt is None
        or attempt["fenced"]
        or attempt["state"] not in ("accepted", "running")
    ):
        raise WorkspaceRefusal("partial recipient changed during native adoption")
    if current["checkpoint"] and pb.CheckpointRef.FromString(current["checkpoint"]) != checkpoint:
        raise WorkspaceRefusal("partial recipient already accepted different progress")
    db.execute(
        "UPDATE weights SET checkpoint=? WHERE id=?", (checkpoint.SerializeToString(), transaction)
    )
    return dict(db.execute("SELECT * FROM weights WHERE id=?", (transaction,)).fetchone())
