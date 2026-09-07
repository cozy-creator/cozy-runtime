"""Completed operation mappings over the shared workspace custody journal."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from functools import partial
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import derived_retention, store_gc
from cozy_runtime.internal.worker.workspace import Journal, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from cozy_runtime.internal.worker.workspace import Workspace

# Enabled results only; compact contradiction identities must not be forgotten.
MAX_ENTRIES = 128


class _Uncacheable(Exception):
    pass


def _typed_result(
    value: Any, schema: Any, producer: str, receipts: dict[str, dict[str, Any]]
) -> Any:
    if not isinstance(schema, dict):
        return value
    if schema.get("input") == "model":
        if not isinstance(value, dict) or value.get("producer_request_id") != producer:
            raise _Uncacheable("borrowed model result requires independent provenance support")
        slot = value.get("output_slot")
        receipt = receipts.get(slot) if isinstance(slot, str) else None
        if (
            receipt is None
            or value.get("tensorfs_receipt_digest") != receipt["native_digest"]
            or value.get("manifest") != receipt["manifest"]
        ):
            raise WorkspaceRefusal("typed model result differs from its native receipt")
        return {"manifest": receipt["manifest"]}
    if "union" in schema:
        branches = schema["union"]
        if value is None and "null" in branches:
            return None
        selected = [branch for branch in branches if branch != "null"]
        if tag := schema.get("tag_field"):
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
    if "fields" in schema and isinstance(value, dict):
        result = dict(value)
        for field in schema["fields"]:
            if field["name"] in result:
                result[field["name"]] = _typed_result(
                    result[field["name"]], field["type"], producer, receipts
                )
        return result
    if "list" in schema and isinstance(value, list):
        return [_typed_result(item, schema["list"], producer, receipts) for item in value]
    return value


def _comparison(payload: dict[str, Any], schema: Any) -> bytes:
    source = payload["source"]
    body = documents.read(
        base64.b64decode(source["outcome_canonical_bytes"], validate=True), pb.AttemptOutcomeBody
    )
    receipts = {}
    for reference in body.get("weights_receipts", []):
        receipt = documents.read(
            base64.b64decode(reference["weights_receipt_canonical_bytes"], validate=True),
            pb.WeightsReceipt,
        )
        native = canonical_json.decode(
            base64.b64decode(receipt["tensorfs_receipt_canonical_bytes"], validate=True)
        )
        receipts[receipt["output_slot"]] = {
            "native_digest": receipt["tensorfs_receipt_digest"],
            "manifest": {
                "digest": "sha256:" + native["manifest"]["sha256"],
                "length": native["manifest"]["length"],
            },
        }
    raw = base64.b64decode(body.get("result", {}).get("inline_result", ""), validate=True)
    result = canonical_json.decode(raw or b"{}")
    return canonical_json.encode(
        {
            "result": _typed_result(result, schema, source["request_id"], receipts),
            "outputs": {slot: receipt["manifest"] for slot, receipt in receipts.items()},
        }
    )


def _hold_id(kind: str, owner: str, recipient: str, key: bytes, transaction: str) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_json.encode([kind, owner, recipient, key.hex(), transaction])
        ).hexdigest()
    )


def _request(hold: dict[str, Any]) -> pb.DerivedRetentionRequest:
    return pb.DerivedRetentionRequest(
        weights_transaction_id=hold["transaction_id"],
        retention_id=hold["retention_id"],
        tensorfs_receipt_digest=documents.raw(hold["native_receipt_digest"]),
    )


def _source(db: Journal, owner: str, call: pb.RecordOperationResultCall) -> dict[str, Any]:
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
    body = documents.read(row["outcome"], pb.AttemptOutcomeBody)
    if body["status"] != pb.OUTCOME_STATUS_SUCCEEDED:
        raise WorkspaceRefusal("failed work cannot enter the memo index")
    weights = db.execute(
        "SELECT * FROM weights WHERE owner=? AND request=? AND ordinal=?",
        (owner, call.request_id, call.attempt_ordinal),
    ).fetchall()
    references = body.get("weights_receipts", [])
    if len(references) != len(weights):
        raise WorkspaceRefusal("memo terminal omitted its native output custody")
    holds = []
    seen = set()
    for reference in references:
        raw = base64.b64decode(reference["weights_receipt_canonical_bytes"], validate=True)
        receipt = documents.read(raw, pb.WeightsReceipt)
        transaction = receipt["weights_transaction_id"]
        match = next((weight for weight in weights if weight["id"] == transaction), None)
        if (
            match is None
            or transaction in seen
            or match["state"] != "receipt"
            or match["receipt"] != raw
            or documents.spell(hashlib.sha256(raw).digest()) != reference["weights_receipt_digest"]
        ):
            raise WorkspaceRefusal("memo receipt is not this execution's native output")
        seen.add(transaction)
        holds.append(
            {
                "transaction_id": transaction,
                "native_receipt_digest": documents.spell(match["native_digest"]),
                "manifest_digest": documents.spell(match["manifest"]),
                "manifest_length": match["manifest_length"],
            }
        )
    outputs = list(body.get("output_manifest", {}).get("outputs", []))
    if blob := body.get("result", {}).get("result_blob"):
        outputs.append(blob)
    for output in outputs:
        if not any(
            output["digest"] == hold["manifest_digest"]
            and output["length"] == hold["manifest_length"]
            for hold in holds
        ):
            raise _Uncacheable("memo output has no independently retainable native custody")
    return {
        "source": {
            "request_id": call.request_id,
            "attempt_ordinal": call.attempt_ordinal,
            "invocation_spec_digest": documents.spell(call.invocation_spec_digest),
            "outcome_id": call.outcome_id,
            "outcome_digest": documents.spell(call.outcome_digest),
            "outcome_canonical_bytes": base64.b64encode(row["outcome"]).decode(),
        },
        "holds": holds,
        "schema": json.loads(row["result_schema"]) if row["result_schema"] else {},
    }


def _retain(
    workspace: Workspace,
    db: Journal,
    owner: str,
    holds: list[dict[str, Any]],
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
        result = workspace._change_hold(db, owner, _request(hold), release=False)
        if (
            documents.spell(result.manifest.digest) != hold["manifest_digest"]
            or result.manifest.length != hold["manifest_length"]
        ):
            raise WorkspaceRefusal("memo native result changed its recorded manifest")
        results.append(result)
    return results


def _reserve_holds(
    workspace: Workspace, db: Journal, owner: str, holds: list[dict[str, Any]]
) -> None:
    """Reserve every planned output before a single native call unlocks the journal.

    The caller commits these intents with the cache/lookup record. A crash or
    cancellation between native acquisitions can then release every planned ID,
    including outputs whose original producer has already been disposed.
    """
    if not db.in_transaction:
        raise RuntimeError("memo hold reservations require their journal transaction")
    for hold in holds:
        workspace.reserve_hold(db, owner, _request(hold), release=False)


def _drain(workspace: Workspace, db: Journal, owner: str) -> int:
    rows = db.execute(
        "SELECT * FROM operation_cache WHERE owner=? AND state IN ('evicting','disabled')", (owner,)
    ).fetchall()
    for row in rows:
        payload = json.loads(row["body"])
        for hold in payload["holds"]:
            workspace._change_hold(db, owner, _request(hold), release=True)
        if row["state"] == "disabled":
            # The negative identity survives capacity pressure, but it does not
            # need either full outcome or the result schema. Prior consumers
            # already own their exact terminal bytes in their lookup records.
            payload = {
                name: {
                    field: value
                    for field, value in payload[name].items()
                    if field != "outcome_canonical_bytes"
                }
                for name in ("source", "conflicting_source")
            } | {"holds": []}
            db.execute(
                "UPDATE operation_cache SET body=? WHERE id=?",
                (canonical_json.encode(payload), row["id"]),
            )
        else:
            db.execute("DELETE FROM operation_cache WHERE id=? AND state='evicting'", (row["id"],))
    return sum(row["state"] == "evicting" for row in rows)


def _missing_cleanup(workspace: Workspace, db: Journal, owner: str, consumer: str) -> None:
    row = db.execute(
        "SELECT * FROM operation_lookups WHERE owner=? AND consumer=?", (owner, consumer)
    ).fetchone()
    if row is None or row["state"] == "released":
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
        (owner, row["cache_id"], row["key"]),
    )
    for hold in json.loads(row["body"])["holds"]:
        workspace._change_hold(db, owner, _request(hold), release=True, cancel_lookup=False)
    _drain(workspace, db, owner)
    completed = db.execute(
        "UPDATE operation_lookups SET state='miss' WHERE owner=? AND consumer=? "
        "AND state='missing'",
        (owner, consumer),
    )
    if completed.rowcount != 1:
        raise WorkspaceRefusal("memo recipient was released during missing-custody cleanup")


def _missing_native_result(
    workspace: Workspace, db: Journal, owner: str, consumer: str, error: Exception
) -> bool:
    code = getattr(error, "code", None)
    if code in {"ROOT_ABSENT", "OBJECT_ABSENT"}:
        return True
    if code != "TRANSACTION_CLOSED":
        return False
    lookup = db.execute(
        "SELECT * FROM operation_lookups WHERE owner=? AND consumer=?", (owner, consumer)
    ).fetchone()
    # A completed consumer's native tombstone is authoritative. Only a fresh
    # acquisition can discover that the source cache itself was removed.
    if lookup is None or lookup["state"] != "retaining":
        return False
    cached = db.execute(
        "SELECT body FROM operation_cache WHERE owner=? AND id=?", (owner, lookup["cache_id"])
    ).fetchone()
    if cached is None:
        return False
    for hold in json.loads(cached["body"])["holds"]:
        try:
            # Verify the ORIGINAL cache hold, never another consumer ID. This
            # separates an evicted source from a released recipient.
            request = _request(hold)
            db.native(
                partial(
                    derived_retention.change,
                    request,
                    tensorfs_root=workspace.store_root,
                    release=False,
                )
            )
        except Exception as exc:
            if getattr(exc, "code", None) in {"ROOT_ABSENT", "OBJECT_ABSENT", "TRANSACTION_CLOSED"}:
                return True
            raise
    return False


def record(
    workspace: Workspace, owner: str, call: pb.RecordOperationResultCall
) -> pb.RecordOperationResultResult:
    workspace.owner(owner)
    if len(call.computation_digest) != 32:
        raise WorkspaceRefusal("memo key is not an exact computation digest")
    answer = pb.RecordOperationResultResult(computation_digest=call.computation_digest)
    with workspace.locked() as db:
        _drain(workspace, db, owner)
        row = db.execute(
            "SELECT * FROM operation_cache WHERE owner=? AND key=?",
            (owner, call.computation_digest),
        ).fetchone()
        if row is None:
            try:
                payload = _source(db, owner, call)
                _comparison(payload, payload["schema"])
            except _Uncacheable:
                return answer
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
                    return answer
                db.execute("UPDATE operation_cache SET state='evicting' WHERE id=?", (victim[0],))
                _drain(workspace, db, owner)
            identity = "memo-" + secrets.token_hex(16)
            for hold in payload["holds"]:
                hold["retention_id"] = _hold_id(
                    "cache", owner, identity, call.computation_digest, hold["transaction_id"]
                )
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO operation_cache(id,owner,key,body,state) "
                    "VALUES(?,?,?,?,'retaining')",
                    (identity, owner, call.computation_digest, canonical_json.encode(payload)),
                )
                _reserve_holds(workspace, db, owner, payload["holds"])
                db.commit()
            except BaseException:
                db.rollback()
                raise
        else:
            if row["state"] == "disabled":
                return answer
            payload, identity = json.loads(row["body"]), row["id"]
            # Replays after the original ACK remain valid, but a different
            # producer must still prove its own exact successful terminal.
            source = payload["source"]
            if (source["request_id"], source["attempt_ordinal"], source["outcome_id"]) != (
                call.request_id,
                call.attempt_ordinal,
                call.outcome_id,
            ):
                try:
                    candidate = _source(db, owner, call)
                    same = _comparison(payload, candidate["schema"]) == _comparison(
                        candidate, candidate["schema"]
                    )
                except _Uncacheable:
                    return answer
                if not same:
                    payload["conflicting_source"] = candidate["source"]
                    db.execute(
                        "UPDATE operation_cache SET state='disabled',body=? WHERE id=?",
                        (canonical_json.encode(payload), identity),
                    )
                    _drain(workspace, db, owner)
                    return answer
            elif source["invocation_spec_digest"] != documents.spell(
                call.invocation_spec_digest
            ) or source["outcome_digest"] != documents.spell(call.outcome_digest):
                raise WorkspaceRefusal("memo recording replay changed its original terminal")
            db.execute("BEGIN IMMEDIATE")
            try:
                _reserve_holds(workspace, db, owner, payload["holds"])
                db.commit()
            except BaseException:
                db.rollback()
                raise
        _retain(workspace, db, owner, payload["holds"])
        committed = db.execute(
            "UPDATE operation_cache SET state='ready' WHERE id=? "
            "AND state IN ('retaining','ready')",
            (identity,),
        )
        if committed.rowcount != 1:
            raise WorkspaceRefusal("memo mapping was evicted during native acquisition")
        answer.recorded = True
        return answer


def lookup(
    workspace: Workspace, owner: str, call: pb.LookupOperationCall
) -> pb.LookupOperationResult:
    workspace.owner(owner)
    if len(call.computation_digest) != 32 or not call.consumer_request_id:
        raise WorkspaceRefusal("memo lookup requires a key and durable consumer identity")
    answer = pb.LookupOperationResult(
        computation_digest=call.computation_digest, consumer_request_id=call.consumer_request_id
    )
    with workspace.locked() as db:
        _drain(workspace, db, owner)
        row = db.execute(
            "SELECT * FROM operation_lookups WHERE owner=? AND consumer=?",
            (owner, call.consumer_request_id),
        ).fetchone()
        if row is not None:
            if row["key"] != call.computation_digest or row["state"] == "released":
                raise WorkspaceRefusal("memo recipient was changed or released")
            if row["state"] == "miss":
                return answer
            if row["state"] == "missing":
                _missing_cleanup(workspace, db, owner, call.consumer_request_id)
                return answer
            payload = json.loads(row["body"])
            db.execute("BEGIN IMMEDIATE")
            try:
                _reserve_holds(workspace, db, owner, payload["holds"])
                db.commit()
            except BaseException:
                db.rollback()
                raise
        else:
            cached = db.execute(
                "SELECT * FROM operation_cache WHERE owner=? AND key=? AND state='ready'",
                (owner, call.computation_digest),
            ).fetchone()
            if cached is None:
                db.execute(
                    "INSERT INTO operation_lookups(owner,consumer,key,cache_id,body,state) "
                    "VALUES(?,?,?,'',?,'miss')",
                    (owner, call.consumer_request_id, call.computation_digest, b"{}"),
                )
                return answer
            payload = json.loads(cached["body"])
            for hold in payload["holds"]:
                hold["retention_id"] = _hold_id(
                    "consumer",
                    owner,
                    call.consumer_request_id,
                    call.computation_digest,
                    hold["transaction_id"],
                )
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO operation_lookups(owner,consumer,key,cache_id,body,state) "
                    "VALUES(?,?,?,?,?,'retaining')",
                    (
                        owner,
                        call.consumer_request_id,
                        call.computation_digest,
                        cached["id"],
                        canonical_json.encode(payload),
                    ),
                )
                db.execute(
                    "UPDATE operation_cache SET sequence="
                    "(SELECT coalesce(max(sequence),0)+1 FROM operation_cache) WHERE id=?",
                    (cached["id"],),
                )
                _reserve_holds(workspace, db, owner, payload["holds"])
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        try:
            retained = _retain(
                workspace, db, owner, payload["holds"], consumer=call.consumer_request_id
            )
        except Exception as exc:
            # Missing payloads can be recomputed. Integrity failures, I/O errors
            # and native release tombstones are distinct typed refusals.
            if not _missing_native_result(workspace, db, owner, call.consumer_request_id, exc):
                raise
            _missing_cleanup(workspace, db, owner, call.consumer_request_id)
            return answer
        committed = db.execute(
            "UPDATE operation_lookups SET state='ready' WHERE owner=? AND consumer=? "
            "AND state IN ('retaining','ready')",
            (owner, call.consumer_request_id),
        )
        if committed.rowcount != 1:
            raise WorkspaceRefusal("memo consumer stopped before result exposure")
        source = payload["source"]
        answer.source.CopyFrom(
            pb.AttemptOutcome(
                request_id=source["request_id"],
                attempt_ordinal=source["attempt_ordinal"],
                invocation_spec_digest=documents.raw(source["invocation_spec_digest"]),
                outcome_id=source["outcome_id"],
                outcome_digest=documents.raw(source["outcome_digest"]),
                outcome_canonical_bytes=base64.b64decode(
                    source["outcome_canonical_bytes"], validate=True
                ),
            )
        )
        answer.retentions.extend(retained)
        answer.found = True
        return answer


def prune(workspace: Workspace, owner: str) -> pb.PruneOperationCacheResult:
    """Drop cache-only native roots; retained computations and live readers win."""
    workspace.owner(owner)
    with workspace.locked() as db:
        removed = _drain(workspace, db, owner)
        rows = db.execute("SELECT * FROM operation_cache WHERE owner=?", (owner,)).fetchall()
        cache_holds = {
            hold["retention_id"] for row in rows for hold in json.loads(row["body"])["holds"]
        }
        for row in rows:
            payload = json.loads(row["body"])
            if not payload["holds"]:
                continue
            source = payload["source"]
            active = db.execute(
                "SELECT 1 FROM attempts WHERE owner=? AND request=? "
                "AND ordinal=? AND state<>'released'",
                (owner, source["request_id"], source["attempt_ordinal"]),
            ).fetchone()
            pending = db.execute(
                "SELECT 1 FROM operation_lookups WHERE owner=? AND cache_id=? "
                "AND state='retaining'",
                (owner, row["id"]),
            ).fetchone()
            if active is not None or pending is not None:
                continue
            external = False
            for hold in payload["holds"]:
                roots = db.execute(
                    "SELECT id FROM holds WHERE owner=? AND transaction_id=? "
                    "AND native_digest=? AND state IN ('retaining','held')",
                    (owner, hold["transaction_id"], documents.raw(hold["native_receipt_digest"])),
                ).fetchall()
                external |= any(root["id"] not in cache_holds for root in roots)
            if not external:
                db.execute("UPDATE operation_cache SET state='evicting' WHERE id=?", (row["id"],))
        removed += _drain(workspace, db, owner)
        collected = db.native(lambda: store_gc.collect(workspace.store_root))
        return pb.PruneOperationCacheResult(
            removed_entries=removed,
            reclaimed_bytes=collected.reclaimed_bytes,
            store_busy=collected.store_busy,
        )
