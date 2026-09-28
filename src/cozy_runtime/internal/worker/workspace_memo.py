"""Completed operation mappings over the shared workspace custody journal."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Callable
from typing import TYPE_CHECKING

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.local_storage_admission import POLICY
from cozy_runtime.internal.worker import store_gc, workspace_partial
from cozy_runtime.internal.worker.workspace import (
    _ID,
    Journal,
    NativeHold,
    WorkspaceBusy,
    WorkspaceRefusal,
    same_ref,
)
from cozy_runtime.internal.worker.workspace_byte_outputs import ByteOutput
from cozy_runtime.internal.worker.workspace_byte_outputs import reference as byte_reference
from cozy_runtime.internal.worker.workspace_memo_rows import (
    Hold,
    Memo,
    NativeReceipt,
    Source,
    decode,
    lookup_row,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from .workspace_native_memo import (
    change_native_hold,
    compare_native,
)
from .workspace_native_memo import lookup_native as lookup_native
from .workspace_native_memo import record_native as record_native
from .workspace_native_memo import record_reuse as record_reuse
from .workspace_native_memo import release_native_lookup as release_native_lookup
from .workspace_native_memo import release_native_result as release_native_result
from .workspace_native_memo import retain_native_result as retain_native_result

if TYPE_CHECKING:
    from cozy_runtime.internal.worker.workspace import Workspace

# Enabled results only; compact contradiction identities must not be forgotten.
MAX_ENTRIES = 128
MAX_METADATA_BYTES = 64 << 20
# Logical metadata accounting reserves bounded row overhead and enough space to
# remember a future contradiction without rejecting that safety transition.
_ROW_OVERHEAD = 128
_CONTRADICTION_ALLOWANCE = 4096


def metadata_bytes(db: Journal) -> int:
    """Optional payloads and native hold markers across this physical workspace.

    This is an admission budget, not a limit on the SQLite file or required
    execution history. Required retention, terminal recording and cleanup never
    consult it. Native tensor bytes and required attempt/weights bodies are not
    charged here.
    """
    total = 0
    for table, columns in (
        ("operation_cache", ("id", "owner", "key", "body")),
        ("operation_lookups", ("owner", "consumer", "key", "cache_id", "body")),
        ("holds", ("id", "owner", "transaction_id", "native_digest", "manifest")),
    ):
        charge = "+".join(f"length(CAST({column} AS BLOB))" for column in columns)
        extra = (
            f"+CASE WHEN state<>'disabled' THEN {_CONTRADICTION_ALLOWANCE} ELSE 0 END"
            if table == "operation_cache"
            else ""
        )
        total += db.execute(
            f"SELECT coalesce(sum({_ROW_OVERHEAD}+{charge}{extra}),0) FROM {table}"
        ).fetchone()[0]
    return total


def _row_bytes(*values: str | bytes) -> int:
    return _ROW_OVERHEAD + sum(
        len(value.encode() if isinstance(value, str) else value) for value in values
    )


def _hold_growth(db: Journal, owner: str, holds: tuple[Hold, ...]) -> int:
    return sum(
        _row_bytes(owner, hold.retention_id, hold.transaction_id, bytes(32), bytes(32))
        for hold in holds
        if db.execute("SELECT 1 FROM holds WHERE id=?", (hold.retention_id,)).fetchone() is None
    )


def delivered(workspace: Workspace, owner: str, consumer: str) -> None:
    """Drop a delivered native lookup's outcome copy after the owner stores it.

    The validated ChildCallResult transition owns when this may happen. Native
    holds remain independent until their explicit release; this is metadata
    compaction, not a finalization or an ownership release.
    """
    workspace.owner(owner)
    with workspace.locked() as db:
        lookup = lookup_row(db, owner, consumer)
        if lookup is None or lookup.state in ("acknowledged", "released", "miss"):
            return
        if lookup.state != "ready":
            raise WorkspaceRefusal("lookup delivery requires its completed native ownership")
        if not lookup.memo.holds:
            db.execute(
                "DELETE FROM operation_lookups WHERE owner=? AND consumer=?", (owner, consumer)
            )
            return
        db.execute(
            "UPDATE operation_lookups SET state='acknowledged',body=? WHERE owner=? AND consumer=?",
            (lookup.memo.compact().encode(), owner, consumer),
        )


def release_lookup(workspace: Workspace, owner: str, consumer: str) -> None:
    """Release temporary attempt lookup custody after its recipient owns the result."""
    workspace.owner(owner)
    with workspace.locked() as db:
        lookup = lookup_row(db, owner, consumer)
        if lookup is None:
            return
        lookup.memo.require("attempt")
        db.execute(
            "UPDATE operation_lookups SET state='released' WHERE owner=? AND consumer=?",
            (owner, consumer),
        )
        for hold in lookup.memo.holds:
            workspace._change_hold(db, owner, hold.request(), kind=hold.kind, release=True)
        compact_released(db, owner)


def compact_released(db: Journal, owner: str) -> None:
    """Compact settled lookup bodies only after every planned hold is released."""
    db.execute(
        "DELETE FROM operation_lookups WHERE owner=? "
        "AND coalesce(json_array_length(CAST(body AS TEXT),'$.holds'),0)=0",
        (owner,),
    )
    rows = db.execute(
        "SELECT consumer,body FROM operation_lookups WHERE owner=? "
        "AND state IN ('released','miss') "
        "AND json_type(CAST(body AS TEXT),'$.source.outcome_canonical_bytes') IS NOT NULL",
        (owner,),
    ).fetchall()
    for row in rows:
        memo = decode(row["body"], Memo)
        settled = True
        for hold in memo.holds:
            native = db.one(NativeHold, "SELECT * FROM holds WHERE id=?", (hold.retention_id,))
            if native is None or native.state != "released":
                settled = False
                break
            if (native.owner, native.transaction_id, native.native_digest) != (
                owner,
                hold.transaction_id,
                documents.raw(hold.native_receipt_digest),
            ):
                raise WorkspaceRefusal("released lookup hold changed its exact native subject")
        if settled:
            db.execute(
                "UPDATE operation_lookups SET body=? WHERE owner=? AND consumer=?",
                (memo.compact().encode(), owner, row["consumer"]),
            )


class _Uncacheable(Exception):
    pass


class _Output(msgspec.Struct, frozen=True):
    native_digest: str
    digest: str
    length: int

    def manifest(self) -> Json:
        return {"digest": self.digest, "length": self.length}


def _typed_result(value: Json, schema: Json, producer: str, receipts: dict[str, _Output]) -> Json:
    if not isinstance(schema, dict):
        return value
    if schema.get("input") == "model":
        if not isinstance(value, dict) or value.get("producer_request_id") != producer:
            raise _Uncacheable("borrowed model result requires independent provenance support")
        slot = value.get("output_slot")
        receipt = receipts.get(slot) if isinstance(slot, str) else None
        if (
            receipt is None
            or value.get("tensorfs_receipt_digest") != receipt.native_digest
            or not same_ref(value.get("manifest"), receipt.digest, receipt.length)
        ):
            raise WorkspaceRefusal("typed model result differs from its native receipt")
        return {"manifest": receipt.manifest()}
    if isinstance(branches := schema.get("union"), list):
        if value is None and "null" in branches:
            return None
        selected = [branch for branch in branches if branch != "null"]
        if isinstance(tag := schema.get("tag_field"), str) and tag:
            selected = [
                branch
                for branch in branches
                if isinstance(branch, dict)
                and isinstance(value, dict)
                and branch.get("tag") == value.get(tag)
            ]
        if len(selected) == 1:
            return _typed_result(value, selected[0], producer, receipts)
        # Untagged alternatives must not guess which user fields are provenance.
        if any(isinstance(branch, dict) for branch in selected):
            raise _Uncacheable("ambiguous structured result union")
        return value
    if isinstance(fields := schema.get("fields"), list) and isinstance(value, dict):
        result = dict(value)
        for field in fields:
            if (
                isinstance(field, dict)
                and isinstance(name := field["name"], str)
                and name in result
            ):
                result[name] = _typed_result(result[name], field["type"], producer, receipts)
        return result
    if "list" in schema and isinstance(value, list):
        return [_typed_result(item, schema["list"], producer, receipts) for item in value]
    return value


def _comparison(memo: Memo, schema: Json) -> bytes:
    source = memo.source
    if source.kind == "native":
        return compare_native(memo)
    body = documents.parse(source.outcome_canonical_bytes, pb.AttemptOutcomeBody)
    receipts = {}
    for reference in body.weights_receipts:
        receipt = documents.parse(reference.weights_receipt_canonical_bytes, pb.WeightsReceipt)
        native = decode(receipt.tensorfs_receipt_canonical_bytes, NativeReceipt).ref()
        receipts[receipt.output_slot] = _Output(
            receipt.tensorfs_receipt_digest, native.digest, native.length
        )
    result = canonical_json.decode(body.result.inline_result or b"{}")
    return canonical_json.encode(
        {
            "result": _typed_result(result, schema, source.request_id, receipts),
            "outputs": {slot: receipt.manifest() for slot, receipt in receipts.items()},
        }
    )


def _source(db: Journal, owner: str, call: pb.RecordOperationResultCall) -> Memo:
    row = db.execute(
        "SELECT * FROM attempts WHERE owner=? AND request=? AND ordinal=?",
        (owner, call.request_id, call.attempt_ordinal),
    ).fetchone()
    if (
        row is None
        or row["state"] != "outcome"
        or row["spec"] != call.invocation_spec_digest
        or row["outcome_id"] != call.outcome_id
        or row["outcome_digest"] != call.outcome_digest
    ):
        raise WorkspaceRefusal("memo result has no exact successful owned terminal")
    body = documents.parse(row["outcome"], pb.AttemptOutcomeBody)
    if body.status != pb.OUTCOME_STATUS_SUCCEEDED:
        raise WorkspaceRefusal("failed work cannot enter the memo index")
    weights = db.execute(
        "SELECT * FROM weights WHERE owner=? AND request=? AND ordinal=?",
        (owner, call.request_id, call.attempt_ordinal),
    ).fetchall()
    if len(body.weights_receipts) != len(weights):
        raise WorkspaceRefusal("memo terminal omitted its native output custody")
    holds: list[Hold] = []
    seen = set()
    for reference in body.weights_receipts:
        raw = reference.weights_receipt_canonical_bytes
        transaction = documents.parse(raw, pb.WeightsReceipt).weights_transaction_id
        match = next((weight for weight in weights if weight["id"] == transaction), None)
        if (
            match is None
            or transaction in seen
            or match["state"] != "receipt"
            or match["receipt"] != raw
            or hashlib.sha256(raw).digest() != reference.weights_receipt_digest
        ):
            raise WorkspaceRefusal("memo receipt is not this execution's native output")
        seen.add(transaction)
        holds.append(
            Hold(
                transaction_id=transaction,
                retention_id="",
                native_receipt_digest=documents.spell(match["native_digest"]),
                manifest_digest=documents.spell(match["manifest"]),
                manifest_length=match["manifest_length"],
            )
        )
    outputs = list(body.output_manifest.outputs)
    if body.result.HasField("result_blob"):
        outputs.append(body.result.result_blob)
    for output in outputs:
        if output.HasField("native_tree"):
            native = output.native_tree
            original = db.one(
                ByteOutput,
                "SELECT * FROM byte_outputs WHERE owner=? AND request=? AND ordinal=? "
                "AND slot=? AND state='complete'",
                (owner, call.request_id, call.attempt_ordinal, output.output_id),
            )
            if original is None or byte_reference(original) != native:
                raise _Uncacheable("memo byte output has no exact original native receipt")
            holds.append(
                Hold(
                    kind="tree",
                    transaction_id=native.producer_root_id,
                    retention_id="",
                    native_receipt_digest=documents.spell(native.receipt_digest),
                    manifest_digest=documents.spell(native.manifest.digest),
                    manifest_length=native.manifest.length,
                    content_bytes=native.content_bytes,
                    output_id=output.output_id,
                )
            )
            if len(holds) > 32:
                raise _Uncacheable("memo result exceeds its native output count bound")
            continue
        if not any(
            output.digest == documents.raw(hold.manifest_digest)
            and output.length == hold.manifest_length
            for hold in holds
        ):
            raise _Uncacheable("memo output has no independently retainable native custody")
    return Memo(
        source=Source(
            request_id=call.request_id,
            attempt_ordinal=call.attempt_ordinal,
            invocation_spec_digest=documents.spell(call.invocation_spec_digest),
            outcome_id=call.outcome_id,
            outcome_digest=documents.spell(call.outcome_digest),
            outcome_canonical_bytes=row["outcome"],
        ),
        holds=tuple(holds),
        schema=json.loads(row["result_schema"]) if row["result_schema"] else {},
    )


def _retain(
    workspace: Workspace,
    db: Journal,
    owner: str,
    holds: tuple[Hold, ...],
    *,
    consumer: str = "",
) -> list[pb.DerivedRetentionResult]:
    results = []
    for hold in holds:
        if consumer:
            row = db.execute(
                "SELECT state FROM operation_lookups WHERE owner=? AND consumer=?",
                (owner, consumer),
            ).fetchone()
            if row is None or row["state"] not in ("retaining", "ready"):
                raise WorkspaceRefusal("memo consumer stopped during native acquisition")
        result = workspace._change_hold(db, owner, hold.request(), release=False, kind=hold.kind)
        if (
            documents.spell(result.manifest.digest) != hold.manifest_digest
            or result.manifest.length != hold.manifest_length
        ):
            raise WorkspaceRefusal("memo native result changed its recorded manifest")
        results.append(result)
    return results


def _reserve_holds(workspace: Workspace, db: Journal, owner: str, holds: tuple[Hold, ...]) -> None:
    """Reserve every planned output before a single native call unlocks the journal.

    The caller commits these intents with the cache/lookup record. A crash or
    cancellation between native acquisitions can then release every planned ID,
    including outputs whose original producer has already been disposed.
    """
    if not db.in_transaction:
        raise RuntimeError("memo hold reservations require their journal transaction")
    for hold in holds:
        workspace.reserve_hold(db, owner, hold.request(), release=False, kind=hold.kind)


def _drain(workspace: Workspace, db: Journal, owner: str) -> int:
    rows = db.execute(
        "SELECT id,state,body FROM operation_cache WHERE owner=? "
        "AND state IN ('evicting','disabled')",
        (owner,),
    ).fetchall()
    for row in rows:
        memo = decode(row["body"], Memo)
        for hold in memo.holds:
            workspace._change_hold(db, owner, hold.request(), release=True, kind=hold.kind)
        if row["state"] == "disabled":
            # The negative identity survives capacity pressure, but it does not
            # need either full outcome or the result schema. Prior consumers
            # already own their exact terminal bytes in their lookup records.
            db.execute(
                "UPDATE operation_cache SET body=? WHERE id=?",
                (msgspec.structs.replace(memo.compact(), holds=()).encode(), row["id"]),
            )
        else:
            db.execute("DELETE FROM operation_cache WHERE id=? AND state='evicting'", (row["id"],))
    compact_released(db, owner)
    return sum(row["state"] == "evicting" for row in rows)


def _missing_cleanup(workspace: Workspace, db: Journal, owner: str, consumer: str) -> None:
    lookup = lookup_row(db, owner, consumer)
    if lookup is None or lookup.state == "released":
        raise WorkspaceRefusal("released lookup cannot become a cache miss")
    db.execute(
        "UPDATE operation_lookups SET state='missing' WHERE owner=? AND consumer=?",
        (owner, consumer),
    )
    # Generation fencing keeps an old missing lookup from evicting a later
    # successful execution of this computation.
    db.execute(
        "UPDATE operation_cache SET state='evicting' WHERE owner=? AND id=? AND key=? "
        "AND state<>'disabled'",
        (owner, lookup.cache_id, lookup.key),
    )
    for hold in lookup.memo.holds:
        workspace._change_hold(
            db, owner, hold.request(), release=True, cancel_lookup=False, kind=hold.kind
        )
    _drain(workspace, db, owner)
    completed = db.execute(
        "UPDATE operation_lookups SET state='miss' WHERE owner=? AND consumer=? "
        "AND state='missing'",
        (owner, consumer),
    )
    if completed.rowcount != 1:
        raise WorkspaceRefusal("memo recipient was released during missing-custody cleanup")
    compact_released(db, owner)


def _missing_native_result(
    workspace: Workspace, db: Journal, owner: str, consumer: str, error: Exception
) -> bool:
    code = getattr(error, "code", None)
    if code in {"ROOT_ABSENT", "OBJECT_ABSENT"}:
        return True
    if code != "TRANSACTION_CLOSED":
        return False
    lookup = lookup_row(db, owner, consumer)
    # A completed consumer's native tombstone is authoritative. Only a fresh
    # acquisition can discover that the source cache itself was removed.
    if lookup is None or lookup.state != "retaining":
        return False
    cached = db.execute(
        "SELECT body FROM operation_cache WHERE owner=? AND id=?", (owner, lookup.cache_id)
    ).fetchone()
    if cached is None:
        return False
    for hold in decode(cached["body"], Memo).holds:
        try:
            # Verify the ORIGINAL cache hold, never another consumer ID. This
            # separates an evicted source from a released recipient.
            change_native_hold(workspace, db, owner, hold.request(), hold.kind, release=False)
        except Exception as exc:
            if getattr(exc, "code", None) in {"ROOT_ABSENT", "OBJECT_ABSENT", "TRANSACTION_CLOSED"}:
                return True
            raise
    return False


def record(
    workspace: Workspace, owner: str, call: pb.RecordOperationResultCall
) -> pb.RecordOperationResultResult:
    expected = Source(
        request_id=call.request_id,
        attempt_ordinal=call.attempt_ordinal,
        invocation_spec_digest=documents.spell(call.invocation_spec_digest),
        outcome_id=call.outcome_id,
        outcome_digest=documents.spell(call.outcome_digest),
    )
    return pb.RecordOperationResultResult(
        computation_digest=call.computation_digest,
        recorded=_record(
            workspace, owner, call.computation_digest, expected, lambda db: _source(db, owner, call)
        ),
    )


def _record(
    workspace: Workspace,
    owner: str,
    key: bytes,
    expected: Source,
    load: Callable[[Journal], Memo],
) -> bool:
    workspace.owner(owner)
    if len(key) != 32:
        raise WorkspaceRefusal("memo key is not an exact computation digest")
    with workspace.locked() as db:
        _drain(workspace, db, owner)
        row = db.execute(
            "SELECT id,state,body FROM operation_cache WHERE owner=? AND key=?",
            (owner, key),
        ).fetchone()
        if row is None:
            try:
                memo = load(db)
                _comparison(memo, memo.schema)
            except _Uncacheable:
                return False
            count = db.execute(
                "SELECT count(*) FROM operation_cache WHERE owner=? AND state<>'disabled'", (owner,)
            ).fetchone()[0]
            if count >= MAX_ENTRIES:
                victim = db.execute(
                    "SELECT id FROM operation_cache AS c WHERE owner=? AND state<>'disabled' "
                    "AND NOT EXISTS "
                    "(SELECT 1 FROM operation_lookups AS l WHERE l.cache_id=c.id "
                    "AND l.state='retaining') ORDER BY sequence LIMIT 1",
                    (owner,),
                ).fetchone()
                if victim is None:
                    return False
                db.execute("UPDATE operation_cache SET state='evicting' WHERE id=?", (victim[0],))
                _drain(workspace, db, owner)
            identity = "memo-" + secrets.token_hex(16)
            memo = memo.assigned("cache", owner, identity, key)
            encoded = memo.encode()
            growth = (
                _row_bytes(identity, owner, key, encoded)
                + _CONTRADICTION_ALLOWANCE
                + _hold_growth(db, owner, memo.holds)
            )
            if metadata_bytes(db) + growth > MAX_METADATA_BYTES:
                return False
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO operation_cache(id,owner,key,body,state) "
                    "VALUES(?,?,?,?,'retaining')",
                    (identity, owner, key, encoded),
                )
                _reserve_holds(workspace, db, owner, memo.holds)
                db.commit()
            except BaseException:
                db.rollback()
                raise
        else:
            if row["state"] == "disabled":
                return False
            memo, identity = decode(row["body"], Memo), row["id"]
            # Replays after the original ACK remain valid, but a different
            # producer must still prove its own exact successful terminal.
            if memo.source.producer() != expected.producer():
                try:
                    candidate = load(db)
                    same = _comparison(memo, candidate.schema) == _comparison(
                        candidate, candidate.schema
                    )
                except _Uncacheable:
                    return False
                if not same:
                    disabled = Memo(
                        source=memo.source.identity(),
                        conflicting_source=candidate.source.identity(),
                        holds=memo.holds,
                    )
                    db.execute(
                        "UPDATE operation_cache SET state='disabled',body=? WHERE id=?",
                        (disabled.encode(), identity),
                    )
                    _drain(workspace, db, owner)
                    return False
            elif memo.source.identity() != expected:
                raise WorkspaceRefusal("memo recording replay changed its original terminal")
            db.execute("BEGIN IMMEDIATE")
            try:
                _reserve_holds(workspace, db, owner, memo.holds)
                db.commit()
            except BaseException:
                db.rollback()
                raise
        _retain(workspace, db, owner, memo.holds)
        committed = db.execute(
            "UPDATE operation_cache SET state='ready' WHERE id=? "
            "AND state IN ('retaining','ready')",
            (identity,),
        )
        if committed.rowcount != 1:
            raise WorkspaceRefusal("memo mapping was evicted during native acquisition")
        return True


def lookup(
    workspace: Workspace, owner: str, call: pb.LookupOperationCall
) -> pb.LookupOperationResult:
    answer = pb.LookupOperationResult(
        computation_digest=call.computation_digest, consumer_request_id=call.consumer_request_id
    )
    found = _lookup(workspace, owner, call.computation_digest, call.consumer_request_id, "attempt")
    return _found(answer, *found) if found is not None else answer


def _lookup(
    workspace: Workspace,
    owner: str,
    key: bytes,
    consumer: str,
    kind: str,
) -> tuple[Memo, list[pb.DerivedRetentionResult]] | None:
    workspace.owner(owner)
    if len(key) != 32 or _ID.fullmatch(consumer) is None:
        raise WorkspaceRefusal("memo lookup requires an exact key and bounded consumer identity")
    with workspace.locked() as db:
        _drain(workspace, db, owner)
        lookup = lookup_row(db, owner, consumer)
        if lookup is not None:
            if lookup.key != key or lookup.state == "released":
                raise WorkspaceRefusal("memo recipient was changed or released")
            if lookup.state == "acknowledged":
                raise WorkspaceRefusal("memo result was delivered; use the owner's recorded result")
            if lookup.state == "miss":
                return None
            if lookup.state == "missing":
                _missing_cleanup(workspace, db, owner, consumer)
                return None
            memo = lookup.memo
            memo.require(kind)
            db.execute("BEGIN IMMEDIATE")
            try:
                _reserve_holds(workspace, db, owner, memo.holds)
                db.commit()
            except BaseException:
                db.rollback()
                raise
        else:
            cached = db.execute(
                "SELECT id,body FROM operation_cache WHERE owner=? AND key=? AND state='ready'",
                (owner, key),
            ).fetchone()
            if cached is None:
                return None
            memo = decode(cached["body"], Memo)
            memo.require(kind)
            if not memo.holds:
                # No native effect needs a recipient journal. Losing this reply
                # may re-query the computation key or recompute after eviction;
                # the owner's accepted call history stores any delivered value.
                db.execute(
                    "UPDATE operation_cache SET sequence="
                    "(SELECT coalesce(max(sequence),0)+1 FROM operation_cache) WHERE id=?",
                    (cached["id"],),
                )
                return memo, []
            memo = memo.assigned("consumer", owner, consumer, key)
            encoded = memo.encode()
            growth = _row_bytes(owner, consumer, key, cached["id"], encoded) + _hold_growth(
                db, owner, memo.holds
            )
            if metadata_bytes(db) + growth > MAX_METADATA_BYTES:
                return None
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO operation_lookups(owner,consumer,key,cache_id,body,state) "
                    "VALUES(?,?,?,?,?,'retaining')",
                    (
                        owner,
                        consumer,
                        key,
                        cached["id"],
                        encoded,
                    ),
                )
                db.execute(
                    "UPDATE operation_cache SET sequence="
                    "(SELECT coalesce(max(sequence),0)+1 FROM operation_cache) WHERE id=?",
                    (cached["id"],),
                )
                _reserve_holds(workspace, db, owner, memo.holds)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        try:
            retained = _retain(workspace, db, owner, memo.holds, consumer=consumer)
        except Exception as exc:
            # Missing payloads can be recomputed. Integrity failures, I/O errors
            # and native release tombstones are distinct typed refusals.
            if not _missing_native_result(workspace, db, owner, consumer, exc):
                raise
            _missing_cleanup(workspace, db, owner, consumer)
            return None
        committed = db.execute(
            "UPDATE operation_lookups SET state='ready' WHERE owner=? AND consumer=? "
            "AND state IN ('retaining','ready')",
            (owner, consumer),
        )
        if committed.rowcount != 1:
            raise WorkspaceRefusal("memo consumer stopped before result exposure")
        return memo, retained


def _found(
    answer: pb.LookupOperationResult,
    memo: Memo,
    retained: list[pb.DerivedRetentionResult],
) -> pb.LookupOperationResult:
    source = memo.source
    answer.source.CopyFrom(
        pb.AttemptOutcome(
            request_id=source.request_id,
            attempt_ordinal=source.attempt_ordinal,
            invocation_spec_digest=documents.raw(source.invocation_spec_digest),
            outcome_id=source.outcome_id,
            outcome_digest=documents.raw(source.outcome_digest),
            outcome_canonical_bytes=source.outcome_canonical_bytes,
        )
    )
    holds = {hold.retention_id: hold for hold in memo.holds}
    for result in retained:
        hold = holds[result.retention_id]
        if hold.kind == "tree":
            answer.byte_retentions.append(
                pb.NativeByteRetentionResult(
                    source=pb.NativeByteTreeRef(
                        producer_root_id=result.weights_transaction_id,
                        receipt_digest=result.tensorfs_receipt_digest,
                        manifest=result.manifest,
                        content_bytes=hold.content_bytes,
                    ),
                    retention_id=result.retention_id,
                    released=result.released,
                )
            )
        else:
            answer.retentions.append(result)
    answer.found = True
    return answer


def _mark_unused(db: Journal, owner: str, limit: int, cache_id: str | None = None) -> int:
    """Existing custody rules, with age used only to order eligible victims."""
    rows = [
        (row["id"], row["state"], decode(row["body"], Memo))
        for row in db.execute(
            "SELECT id,state,body FROM operation_cache WHERE owner=? ORDER BY sequence,id",
            (owner,),
        )
    ]
    cache_holds = {hold.retention_id for _, _, memo in rows for hold in memo.holds}
    selected = 0
    for identity, state, memo in rows:
        if state == "disabled" or (cache_id is not None and identity != cache_id):
            continue
        if selected >= limit:
            break
        if not memo.holds:
            continue
        source = memo.source
        if source.kind == "native":
            active = db.execute(
                "SELECT 1 FROM attempts WHERE owner=? AND request=? AND state<>'released'",
                (owner, source.parent_request),
            ).fetchone()
        else:
            active = db.execute(
                "SELECT 1 FROM attempts WHERE owner=? AND request=? "
                "AND ordinal=? AND state<>'released'",
                (owner, source.request_id, source.attempt_ordinal),
            ).fetchone()
        pending = db.execute(
            "SELECT 1 FROM operation_lookups WHERE owner=? AND cache_id=? AND state='retaining'",
            (owner, identity),
        ).fetchone()
        if active is not None or pending is not None:
            continue
        external = False
        for hold in memo.holds:
            roots = db.execute(
                "SELECT id FROM holds WHERE owner=? AND transaction_id=? "
                "AND native_digest=? AND state IN ('retaining','held')",
                (owner, hold.transaction_id, documents.raw(hold.native_receipt_digest)),
            ).fetchall()
            external |= any(root["id"] not in cache_holds for root in roots)
        if not external:
            db.execute("UPDATE operation_cache SET state='evicting' WHERE id=?", (identity,))
            selected += 1
    return selected


def prune(workspace: Workspace, owner: str) -> pb.PruneOperationCacheResult:
    """Drop cache-only native roots; retained computations and live readers win."""
    workspace.owner(owner)
    with workspace.locked() as db:
        removed = _drain(workspace, db, owner)
        _mark_unused(db, owner, MAX_ENTRIES)
        removed += _drain(workspace, db, owner)
        collected = db.native(lambda: store_gc.collect(workspace.store_root))
        return pb.PruneOperationCacheResult(
            removed_entries=removed,
            reclaimed_bytes=collected.reclaimed_bytes,
            store_busy=collected.store_busy,
        )


def reclaim_unused(
    workspace: Workspace, *, target_bytes: int, max_entries: int = int(POLICY["max_memo_victims"])
) -> pb.PruneOperationCacheResult:
    try:
        return _reclaim_unused(workspace, target_bytes=target_bytes, max_entries=max_entries)
    except WorkspaceBusy:
        return pb.PruneOperationCacheResult(store_busy=True)


def _reclaim_unused(
    workspace: Workspace, *, target_bytes: int, max_entries: int
) -> pb.PruneOperationCacheResult:
    """One bounded physical workspace cleanup; no caller may name or release a hold.

    The workspace owns all cached results here. Explicit owners, unfinished producers
    and pending/native recipients retain priority across every native lock crossing.
    """
    if target_bytes <= 0 or not 1 <= max_entries <= MAX_ENTRIES:
        raise ValueError("cache reclamation requires a positive target and bounded batch")
    result = pb.PruneOperationCacheResult()
    with workspace.locked(blocking=False) as db:
        # A previous busy pass may already have dropped its cache mapping. Sweep
        # now-unreferenced bytes before choosing another victim, including when
        # no optional memo roots remain. Live native readers/writers win first.
        collected = db.native(lambda: store_gc.collect(workspace.store_root))
        result.reclaimed_bytes = collected.reclaimed_bytes
        result.store_busy = collected.store_busy
        if result.store_busy or result.reclaimed_bytes >= target_bytes:
            return result
        # Stopped work nobody has adopted is the cheapest to lose: oldest partials go first.
        reclaimed, removed, busy = workspace_partial.evict(
            db, workspace, target_bytes=target_bytes - result.reclaimed_bytes
        )
        result.reclaimed_bytes += reclaimed
        result.removed_entries += removed
        if busy or result.reclaimed_bytes >= target_bytes:
            result.store_busy = result.store_busy or busy
            return result
        candidates = db.execute(
            "SELECT owner,id FROM operation_cache WHERE state<>'disabled' "
            "ORDER BY sequence,id LIMIT ?",
            (MAX_ENTRIES,),
        ).fetchall()
        for row in candidates:
            if result.removed_entries >= max_entries or result.reclaimed_bytes >= target_bytes:
                break
            owner = row["owner"]
            if not _mark_unused(db, owner, 1, row["id"]):
                continue
            result.removed_entries += _drain(workspace, db, owner)
            collected = db.native(lambda: store_gc.collect(workspace.store_root))
            result.reclaimed_bytes += collected.reclaimed_bytes
            if collected.store_busy:
                result.store_busy = True
                break
    return result
