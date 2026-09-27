"""Completed operation mappings over the shared workspace custody journal."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.internal.local_storage_admission import POLICY
from cozy_runtime.internal.worker import store_gc, workspace_partial
from cozy_runtime.internal.worker.workspace import (
    _ID,
    Journal,
    WorkspaceBusy,
    WorkspaceRefusal,
    same_ref,
)
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

from .workspace_native_memo import (
    lookup_native as lookup_native,
)
from .workspace_native_memo import (
    record_native as record_native,
)
from .workspace_native_memo import (
    record_reuse as record_reuse,
)
from .workspace_native_memo import (
    release_native_lookup as release_native_lookup,
)
from .workspace_native_memo import (
    release_native_result as release_native_result,
)
from .workspace_native_memo import (
    retain_native_result as retain_native_result,
)

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


def _hold_growth(db: Journal, owner: str, holds: list[dict[str, Any]]) -> int:
    return sum(
        _row_bytes(owner, hold["retention_id"], hold["transaction_id"], bytes(32), bytes(32))
        for hold in holds
        if db.execute("SELECT 1 FROM holds WHERE id=?", (hold["retention_id"],)).fetchone() is None
    )


def _source_identity(source: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in source.items()
        if name not in ("outcome_canonical_bytes", "native_receipt", "result")
        and not (source.get("kind") == "native" and name == "parent_ordinal")
    }


def _compact_payload(payload: dict[str, Any]) -> bytes:
    return canonical_json.encode(
        {"source": _source_identity(payload["source"]), "holds": payload["holds"]}
    )


def delivered(workspace: Workspace, owner: str, consumer: str) -> None:
    """Drop a delivered native lookup's outcome copy after the owner stores it.

    The validated ChildCallResult transition owns when this may happen. Native
    holds remain independent until their explicit release; this is metadata
    compaction, not a finalization or an ownership release.
    """
    workspace.owner(owner)
    with workspace.locked() as db:
        row = db.execute(
            "SELECT * FROM operation_lookups WHERE owner=? AND consumer=?", (owner, consumer)
        ).fetchone()
        if row is None or row["state"] in ("acknowledged", "released", "miss"):
            return
        if row["state"] != "ready":
            raise WorkspaceRefusal("lookup delivery requires its completed native ownership")
        payload = json.loads(row["body"])
        if not payload["holds"]:
            db.execute(
                "DELETE FROM operation_lookups WHERE owner=? AND consumer=?", (owner, consumer)
            )
            return
        db.execute(
            "UPDATE operation_lookups SET state='acknowledged',body=? WHERE owner=? AND consumer=?",
            (_compact_payload(payload), owner, consumer),
        )


def release_lookup(workspace: Workspace, owner: str, consumer: str) -> None:
    """Release temporary attempt lookup custody after its recipient owns the result."""
    workspace.owner(owner)
    with workspace.locked() as db:
        row = db.execute(
            "SELECT body FROM operation_lookups WHERE owner=? AND consumer=?", (owner, consumer)
        ).fetchone()
        if row is None:
            return
        payload = canonical_json.decode(row["body"])
        _require_kind(payload, "attempt")
        db.execute(
            "UPDATE operation_lookups SET state='released' WHERE owner=? AND consumer=?",
            (owner, consumer),
        )
        for hold in payload["holds"]:
            workspace._change_hold(
                db, owner, _request(hold), kind=hold.get("kind", "derived"), release=True
            )
        compact_released(db, owner)


def compact_released(db: Journal, owner: str) -> None:
    """Compact settled lookup bodies only after every planned hold is released."""
    db.execute(
        "DELETE FROM operation_lookups WHERE owner=? "
        "AND coalesce(json_array_length(CAST(body AS TEXT),'$.holds'),0)=0",
        (owner,),
    )
    rows = db.execute(
        "SELECT * FROM operation_lookups WHERE owner=? AND state IN ('released','miss') "
        "AND json_type(CAST(body AS TEXT),'$.source.outcome_canonical_bytes') IS NOT NULL",
        (owner,),
    ).fetchall()
    for row in rows:
        payload = json.loads(row["body"])
        settled = True
        for hold in payload["holds"]:
            native = db.execute(
                "SELECT * FROM holds WHERE id=?", (hold["retention_id"],)
            ).fetchone()
            if native is None or native["state"] != "released":
                settled = False
                break
            if (native["owner"], native["transaction_id"], native["native_digest"]) != (
                owner,
                hold["transaction_id"],
                documents.raw(hold["native_receipt_digest"]),
            ):
                raise WorkspaceRefusal("released lookup hold changed its exact native subject")
        if settled:
            db.execute(
                "UPDATE operation_lookups SET body=? WHERE owner=? AND consumer=?",
                (_compact_payload(payload), owner, row["consumer"]),
            )


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
            or not same_ref(
                value.get("manifest"), receipt["manifest"]["digest"], receipt["manifest"]["length"]
            )
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
    if source.get("kind") == "native":
        from .workspace_native_memo import compare_native

        return compare_native(payload)
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
        if native := output.get("native_tree"):
            from .workspace_byte_outputs import reference as byte_reference

            original = db.execute(
                "SELECT * FROM byte_outputs WHERE owner=? AND request=? AND ordinal=? "
                "AND slot=? AND state='complete'",
                (owner, call.request_id, call.attempt_ordinal, output["output_id"]),
            ).fetchone()
            if original is None or documents.body(byte_reference(dict(original))) != native:
                raise _Uncacheable("memo byte output has no exact original native receipt")
            holds.append(
                {
                    "kind": "tree",
                    "transaction_id": native["producer_root_id"],
                    "native_receipt_digest": native["receipt_digest"],
                    "manifest_digest": native["manifest"]["digest"],
                    "manifest_length": native["manifest"]["length"],
                    "content_bytes": native.get("content_bytes", 0),
                    "output_id": output["output_id"],
                }
            )
            if len(holds) > 32:
                raise _Uncacheable("memo result exceeds its native output count bound")
            continue
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
        result = workspace._change_hold(
            db, owner, _request(hold), release=False, kind=hold.get("kind", "derived")
        )
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
        workspace.reserve_hold(
            db, owner, _request(hold), release=False, kind=hold.get("kind", "derived")
        )


def _drain(workspace: Workspace, db: Journal, owner: str) -> int:
    rows = db.execute(
        "SELECT * FROM operation_cache WHERE owner=? AND state IN ('evicting','disabled')", (owner,)
    ).fetchall()
    for row in rows:
        payload = json.loads(row["body"])
        for hold in payload["holds"]:
            workspace._change_hold(
                db, owner, _request(hold), release=True, kind=hold.get("kind", "derived")
            )
        if row["state"] == "disabled":
            # The negative identity survives capacity pressure, but it does not
            # need either full outcome or the result schema. Prior consumers
            # already own their exact terminal bytes in their lookup records.
            payload = {
                name: _source_identity(payload[name]) for name in ("source", "conflicting_source")
            } | {"holds": []}
            db.execute(
                "UPDATE operation_cache SET body=? WHERE id=?",
                (canonical_json.encode(payload), row["id"]),
            )
        else:
            db.execute("DELETE FROM operation_cache WHERE id=? AND state='evicting'", (row["id"],))
    compact_released(db, owner)
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
        workspace._change_hold(
            db,
            owner,
            _request(hold),
            release=True,
            cancel_lookup=False,
            kind=hold.get("kind", "derived"),
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
            from .workspace_native_memo import change_native_hold

            change_native_hold(
                workspace, db, owner, request, hold.get("kind", "derived"), release=False
            )
        except Exception as exc:
            if getattr(exc, "code", None) in {"ROOT_ABSENT", "OBJECT_ABSENT", "TRANSACTION_CLOSED"}:
                return True
            raise
    return False


def _producer(source: dict[str, Any]) -> tuple[Any, ...]:
    if source.get("kind") == "native":
        return ("native", source["service_id"])
    return ("attempt", source["request_id"], source["attempt_ordinal"], source["outcome_id"])


def record(
    workspace: Workspace, owner: str, call: pb.RecordOperationResultCall
) -> pb.RecordOperationResultResult:
    expected = {
        "request_id": call.request_id,
        "attempt_ordinal": call.attempt_ordinal,
        "invocation_spec_digest": documents.spell(call.invocation_spec_digest),
        "outcome_id": call.outcome_id,
        "outcome_digest": documents.spell(call.outcome_digest),
    }
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
    expected: dict[str, Any],
    load: Callable[[Journal], dict[str, Any]],
) -> bool:
    workspace.owner(owner)
    if len(key) != 32:
        raise WorkspaceRefusal("memo key is not an exact computation digest")
    with workspace.locked() as db:
        _drain(workspace, db, owner)
        row = db.execute(
            "SELECT * FROM operation_cache WHERE owner=? AND key=?",
            (owner, key),
        ).fetchone()
        if row is None:
            try:
                payload = load(db)
                _comparison(payload, payload["schema"])
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
            for hold in payload["holds"]:
                hold["retention_id"] = _hold_id(
                    "cache", owner, identity, key, hold["transaction_id"]
                )
            encoded = canonical_json.encode(payload)
            growth = (
                _row_bytes(identity, owner, key, encoded)
                + _CONTRADICTION_ALLOWANCE
                + _hold_growth(db, owner, payload["holds"])
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
                _reserve_holds(workspace, db, owner, payload["holds"])
                db.commit()
            except BaseException:
                db.rollback()
                raise
        else:
            if row["state"] == "disabled":
                return False
            payload, identity = json.loads(row["body"]), row["id"]
            # Replays after the original ACK remain valid, but a different
            # producer must still prove its own exact successful terminal.
            source = payload["source"]
            if _producer(source) != _producer(expected):
                try:
                    candidate = load(db)
                    same = _comparison(payload, candidate["schema"]) == _comparison(
                        candidate, candidate["schema"]
                    )
                except _Uncacheable:
                    return False
                if not same:
                    payload = {
                        "source": _source_identity(payload["source"]),
                        "conflicting_source": _source_identity(candidate["source"]),
                        "holds": payload["holds"],
                    }
                    db.execute(
                        "UPDATE operation_cache SET state='disabled',body=? WHERE id=?",
                        (canonical_json.encode(payload), identity),
                    )
                    _drain(workspace, db, owner)
                    return False
            elif _source_identity(source) != expected:
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
        return True


def lookup(
    workspace: Workspace, owner: str, call: pb.LookupOperationCall
) -> pb.LookupOperationResult:
    answer = pb.LookupOperationResult(
        computation_digest=call.computation_digest, consumer_request_id=call.consumer_request_id
    )
    found = _lookup(workspace, owner, call.computation_digest, call.consumer_request_id, "attempt")
    return _found(answer, *found) if found is not None else answer


def _require_kind(payload: dict[str, Any], expected: str) -> None:
    if payload["source"].get("kind", "attempt") != expected:
        raise WorkspaceRefusal("memo result has another terminal kind")


def _lookup(
    workspace: Workspace,
    owner: str,
    key: bytes,
    consumer: str,
    kind: str,
) -> tuple[dict[str, Any], list[pb.DerivedRetentionResult]] | None:
    workspace.owner(owner)
    if len(key) != 32 or _ID.fullmatch(consumer) is None:
        raise WorkspaceRefusal("memo lookup requires an exact key and bounded consumer identity")
    with workspace.locked() as db:
        _drain(workspace, db, owner)
        row = db.execute(
            "SELECT * FROM operation_lookups WHERE owner=? AND consumer=?",
            (owner, consumer),
        ).fetchone()
        if row is not None:
            if row["key"] != key or row["state"] == "released":
                raise WorkspaceRefusal("memo recipient was changed or released")
            if row["state"] == "acknowledged":
                raise WorkspaceRefusal("memo result was delivered; use the owner's recorded result")
            if row["state"] == "miss":
                return None
            if row["state"] == "missing":
                _missing_cleanup(workspace, db, owner, consumer)
                return None
            payload = json.loads(row["body"])
            _require_kind(payload, kind)
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
                (owner, key),
            ).fetchone()
            if cached is None:
                return None
            payload = json.loads(cached["body"])
            _require_kind(payload, kind)
            if not payload["holds"]:
                # No native effect needs a recipient journal. Losing this reply
                # may re-query the computation key or recompute after eviction;
                # the owner's accepted call history stores any delivered value.
                db.execute(
                    "UPDATE operation_cache SET sequence="
                    "(SELECT coalesce(max(sequence),0)+1 FROM operation_cache) WHERE id=?",
                    (cached["id"],),
                )
                return payload, []
            for hold in payload["holds"]:
                hold["retention_id"] = _hold_id(
                    "consumer",
                    owner,
                    consumer,
                    key,
                    hold["transaction_id"],
                )
            encoded = canonical_json.encode(payload)
            growth = _row_bytes(owner, consumer, key, cached["id"], encoded) + _hold_growth(
                db, owner, payload["holds"]
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
                _reserve_holds(workspace, db, owner, payload["holds"])
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        try:
            retained = _retain(workspace, db, owner, payload["holds"], consumer=consumer)
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
        return payload, retained


def _found(
    answer: pb.LookupOperationResult,
    payload: dict[str, Any],
    retained: list[pb.DerivedRetentionResult],
) -> pb.LookupOperationResult:
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
    holds = {hold["retention_id"]: hold for hold in payload["holds"]}
    for result in retained:
        hold = holds[result.retention_id]
        if hold.get("kind") == "tree":
            answer.byte_retentions.append(
                pb.NativeByteRetentionResult(
                    source=pb.NativeByteTreeRef(
                        producer_root_id=result.weights_transaction_id,
                        receipt_digest=result.tensorfs_receipt_digest,
                        manifest=result.manifest,
                        content_bytes=hold["content_bytes"],
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
    rows = db.execute(
        "SELECT * FROM operation_cache WHERE owner=? ORDER BY sequence,id", (owner,)
    ).fetchall()
    cache_holds = {
        hold["retention_id"] for row in rows for hold in json.loads(row["body"])["holds"]
    }
    selected = 0
    for row in rows:
        if row["state"] == "disabled" or (cache_id is not None and row["id"] != cache_id):
            continue
        if selected >= limit:
            break
        payload = json.loads(row["body"])
        if not payload["holds"]:
            continue
        source = payload["source"]
        if source.get("kind") == "native":
            active = db.execute(
                "SELECT 1 FROM attempts WHERE owner=? AND request=? AND state<>'released'",
                (owner, source["parent_request"]),
            ).fetchone()
        else:
            active = db.execute(
                "SELECT 1 FROM attempts WHERE owner=? AND request=? "
                "AND ordinal=? AND state<>'released'",
                (owner, source["request_id"], source["attempt_ordinal"]),
            ).fetchone()
        pending = db.execute(
            "SELECT 1 FROM operation_lookups WHERE owner=? AND cache_id=? AND state='retaining'",
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
