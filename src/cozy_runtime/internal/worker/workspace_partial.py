"""Adopt exact stopped native work; a partial checkpoint is never a memo result.

Stopped work that no owner holds any more is RETIRED, not destroyed: an incomplete source
download or conversion, and a memoized attempt's checkpointed but uncommitted weights. It
is keyed by its exact computation, so a later identical call adopts it and only the rest is
done again. Adoption spends it; storage pressure evicts the oldest first; a rental's end
takes the whole store.
"""

from __future__ import annotations

from functools import partial

from cozy_runtime.internal import fill
from cozy_runtime.internal.worker import store_gc, workspace_sources
from cozy_runtime.internal.worker.workspace import (
    Journal,
    WeightsRow,
    Workspace,
    WorkspaceBusy,
    WorkspaceRefusal,
    _stop_native_transaction,
)
from cozy_runtime.internal.worker.workspace_sources import NativeCall
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

SCHEMA12 = """
ALTER TABLE native_calls ADD COLUMN retired_ms INTEGER NOT NULL DEFAULT 0;
ALTER TABLE weights ADD COLUMN retired_ms INTEGER NOT NULL DEFAULT 0;
"""


def retire_weights(db: Journal, transaction: str) -> None:
    db.execute(
        "UPDATE weights SET retired_ms=? WHERE id=? AND state='intent' AND retired_ms=0",
        (workspace_sources.now_ms(), transaction),
    )


def _spend_weights(db: Journal, workspace: Workspace, transaction: str) -> None:
    db.native(partial(_stop_native_transaction, workspace.store_root, transaction))
    db.execute(
        "UPDATE weights SET state='released',declaration=x'',checkpoint=x'',"
        "restore_checkpoint=x'',receipt=x'',objects=x'5b5d' WHERE id=? AND retired_ms>0",
        (transaction,),
    )


def evict(db: Journal, workspace: Workspace, *, target_bytes: int) -> tuple[int, int, bool]:
    """Release retired partials, oldest first; returns reclaimed bytes, count, store busy."""
    victims = [
        (row["retired_ms"], partial(_spend_weights, db, workspace, row["id"]))
        for row in db.execute(
            "SELECT id,retired_ms FROM weights WHERE state='intent' AND retired_ms>0 "
            "ORDER BY retired_ms LIMIT 256"
        )
    ] + [
        (row.retired_ms, partial(workspace_sources.spend, db, workspace, row))
        for row in db.all(
            NativeCall, workspace_sources.PARTIALS + " ORDER BY n.retired_ms LIMIT 256"
        )
    ]
    reclaimed = removed = 0
    for _, spend in sorted(victims, key=lambda victim: victim[0]):
        if reclaimed >= target_bytes:
            break
        spend()
        removed += 1
        collected = db.native(lambda: store_gc.collect(workspace.store_root))
        reclaimed += collected.reclaimed_bytes
        if collected.store_busy:
            return reclaimed, removed, True
    return reclaimed, removed, False


def reclaim(workspace: Workspace, target_bytes: int) -> int:
    """Evict retired partials, oldest first, for a writer that needs ``target_bytes``."""
    try:
        with workspace.locked(blocking=False) as db:
            reclaimed, _, _ = evict(db, workspace, target_bytes=max(target_bytes, 1))
    except WorkspaceBusy:
        return 0
    return reclaimed


def adopt(workspace: Workspace, db: Journal, owner: str, transaction: str) -> WeightsRow:
    row = db.one(WeightsRow, "SELECT * FROM weights WHERE owner=? AND id=?", (owner, transaction))
    if row is None:
        raise WorkspaceRefusal("partial recipient has no accepted writer intent")
    if row["state"] != "intent" or row["checkpoint"] or row["ready"]:
        return row
    attempt = db.execute(
        "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
        (owner, row["request"], row["ordinal"]),
    ).fetchone()
    if attempt is None or attempt["fenced"] or attempt["state"] not in ("accepted", "running"):
        raise WorkspaceRefusal("partial recipient attempt has stopped")
    if not attempt["memoize"]:
        return row
    if not row["adoption_source"]:
        candidates = db.all(
            WeightsRow,
            "SELECT w.* FROM weights w JOIN attempts a ON "
            "a.owner=w.owner AND a.request=w.request AND a.ordinal=w.ordinal "
            "WHERE w.owner=? AND w.id<>? AND w.slot=? AND w.state='intent' "
            "AND w.declaration_digest=? AND w.declaration=? AND w.checkpoint<>x'' "
            "AND a.memoize=1 AND (w.retired_ms>0 "
            "OR ((a.state='outcome' OR a.fenced=1) AND a.state<>'released')) "
            "ORDER BY w.rowid DESC LIMIT 32",
            (owner, transaction, row["slot"], row["declaration_digest"], row["declaration"]),
        )
        if not candidates:
            return row
        source = max(
            candidates, key=lambda item: pb.CheckpointRef.FromString(item["checkpoint"]).bytes
        )
        db.execute(
            "UPDATE weights SET adoption_source=?,adoption_epoch=?,adoption_checkpoint=? "
            "WHERE id=?",
            (source["id"], source["epoch"], source["checkpoint"], transaction),
        )
        row = Workspace._weights(db, transaction)

    def acquire() -> pb.CheckpointRef:
        store = fill.store(workspace.store_root)
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
    current = db.one(WeightsRow, "SELECT * FROM weights WHERE id=?", (transaction,))
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
    # The recipient now retains the progress independently; a retired donor is spent.
    donor = db.execute(
        "SELECT retired_ms FROM weights WHERE id=?", (row["adoption_source"],)
    ).fetchone()
    if donor is not None and donor["retired_ms"]:
        _spend_weights(db, workspace, row["adoption_source"])
    return Workspace._weights(db, transaction)
