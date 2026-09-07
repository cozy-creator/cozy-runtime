"""Exact one-way import of the frozen Host custody journal into the workspace."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from typing import Any, NoReturn, cast

from cozy_runtime import canonical_json
from cozy_runtime.internal import fill
from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.internal.worker.workspace import Journal, Workspace, WorkspaceRefusal
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

MAX_RECORD = 64 << 20
MAX_CHUNK = 48 << 10
KINDS = ("attempt", "weights", "hold", "cache", "lookup")
TABLES = ("attempts", "weights", "holds", "operation_cache", "operation_lookups")
Authorize = Callable[[pb.Claim], str]
_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _refuse(why: str) -> NoReturn:
    raise WorkspaceRefusal("workspace import: " + why)


def _fields(value: Any, fields: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        _refuse("record has unknown or missing fields")
    return cast(dict[str, Any], value)


def _text(value: Any, *, owner: bool = False, empty: bool = False) -> str:
    if not isinstance(value, str) or not (
        (empty and value == "") or (_OWNER if owner else _ID).fullmatch(value)
    ):
        _refuse("identity is not bounded or canonical")
    return value


def _number(value: Any, *, positive: bool = False) -> int:
    if type(value) is not int or not (1 if positive else 0) <= value <= (1 << 53) - 1:
        _refuse("integer is outside the admitted range")
    return value


def _digest(value: Any, *, empty: bool = False) -> bytes:
    if empty and value == "":
        return b""
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        _refuse("digest is not sha256 hex")
    return bytes.fromhex(value[7:])


def _bytes(value: Any) -> bytes:
    if not isinstance(value, str):
        _refuse("byte field is not base64")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise WorkspaceRefusal("workspace import: malformed base64") from exc
    if base64.b64encode(raw).decode() != value:
        _refuse("byte field has a second base64 spelling")
    return raw


def _bound(raw: bytes, digest: bytes, what: str) -> None:
    if not raw or hashlib.sha256(raw).digest() != digest:
        _refuse(what + " bytes differ from their recorded digest")


def _state(value: Any, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        _refuse("unknown custody state")
    return cast(str, value)


def _source(value: Any, *, absent: bool = False) -> dict[str, Any]:
    row = _fields(
        value,
        "request_id attempt_ordinal invocation_spec_digest outcome_id outcome_digest "
        "outcome_canonical_bytes",
    )
    if absent and row == {
        "request_id": "",
        "attempt_ordinal": 0,
        "invocation_spec_digest": "",
        "outcome_id": "",
        "outcome_digest": "",
        "outcome_canonical_bytes": "",
    }:
        return row
    _text(row["request_id"])
    _text(row["outcome_id"])
    _number(row["attempt_ordinal"], positive=True)
    spec = _digest(row["invocation_spec_digest"])
    raw = _bytes(row["outcome_canonical_bytes"])
    _bound(raw, _digest(row["outcome_digest"]), "source outcome")
    body = documents.read(raw, pb.AttemptOutcomeBody)
    if (
        body.get("request_id"),
        body.get("attempt_ordinal"),
        body.get("invocation_spec_digest"),
        body.get("status"),
    ) != (
        row["request_id"],
        row["attempt_ordinal"],
        documents.spell(spec),
        pb.OUTCOME_STATUS_SUCCEEDED,
    ):
        _refuse("memo source is not its exact successful terminal")
    return row


def parse_record(kind: str, raw: bytes, owner: str) -> dict[str, Any]:
    if not raw or len(raw) > MAX_RECORD:
        _refuse("record exceeds its byte bound")
    try:
        row = json.loads(raw)
        if canonical_json.encode(row) != raw:
            _refuse("record is not canonical JSON")
    except (ValueError, TypeError) as exc:
        raise WorkspaceRefusal("workspace import: record is not canonical JSON") from exc
    _text(
        row.get("owner_scope") if isinstance(row, dict) else None,
        owner=True,
        empty=kind == "attempt",
    )
    if row["owner_scope"] not in (owner, "" if kind == "attempt" else owner):
        _refuse("record belongs to another authenticated owner")
    if kind == "attempt":
        _fields(
            row,
            "owner_scope request_id attempt_ordinal invocation_spec_digest "
            "invocation_spec_canonical_bytes state outcome_id outcome_digest "
            "outcome_canonical_bytes retain_work",
        )
        _text(row["request_id"])
        _number(row["attempt_ordinal"], positive=True)
        invocation = _bytes(row["invocation_spec_canonical_bytes"])
        _bound(invocation, _digest(row["invocation_spec_digest"]), "invocation")
        documents.read(invocation, pb.InvocationSpec)
        state = _state(row["state"], ("received", "running", "outcome"))
        if type(row["retain_work"]) is not bool:
            _refuse("retain_work is not boolean")
        if state == "outcome":
            _text(row["outcome_id"])
            outcome = _bytes(row["outcome_canonical_bytes"])
            _bound(outcome, _digest(row["outcome_digest"]), "outcome")
            body = documents.read(outcome, pb.AttemptOutcomeBody)
            if (
                body.get("request_id"),
                body.get("attempt_ordinal"),
                body.get("invocation_spec_digest"),
            ) != (row["request_id"], row["attempt_ordinal"], row["invocation_spec_digest"]):
                _refuse("terminal differs from its accepted attempt")
        elif any(row[key] for key in ("outcome_id", "outcome_digest", "outcome_canonical_bytes")):
            _refuse("running attempt carries an uncommitted terminal")
    elif kind == "weights":
        _fields(
            row,
            "transaction_id owner_scope request_id attempt_ordinal invocation_spec_digest "
            "output_slot declaration_digest declaration_canonical_bytes writer_epoch state "
            "intent_ready restore_checkpoint checkpoint receipt_digest "
            "receipt_canonical_bytes manifest_digest manifest_length objects",
        )
        _digest(row["transaction_id"])
        if row["transaction_id"] != weights_transaction_id(
            owner, row["request_id"], row["invocation_spec_digest"], row["output_slot"]
        ):
            _refuse("weights transaction is not derived from its exact owner invocation")
        _text(row["request_id"])
        _text(row["output_slot"])
        _number(row["attempt_ordinal"], positive=True)
        _number(row["writer_epoch"], positive=True)
        _digest(row["invocation_spec_digest"])
        _bound(
            _bytes(row["declaration_canonical_bytes"]),
            _digest(row["declaration_digest"]),
            "declaration",
        )
        _state(row["state"], ("intent", "receipt"))
        if type(row["intent_ready"]) is not bool:
            _refuse("intent_ready is not boolean")
        for key in ("restore_checkpoint", "checkpoint"):
            data = _bytes(row[key])
            if data:
                checkpoint = pb.CheckpointRef.FromString(data)
                if len(checkpoint.head.digest) != 32 or checkpoint.head.length <= 0:
                    _refuse("checkpoint is not an exact native reference")
        if row["state"] == "receipt":
            outer = _bytes(row["receipt_canonical_bytes"])
            _bound(outer, _digest(row["receipt_digest"]), "weights receipt")
            receipt = documents.read(outer, pb.WeightsReceipt)
            if (
                receipt.get("owner_authority_scope"),
                receipt.get("request_id"),
                receipt.get("invocation_spec_digest"),
                receipt.get("output_slot"),
                receipt.get("weights_transaction_id"),
            ) != (
                owner,
                row["request_id"],
                row["invocation_spec_digest"],
                row["output_slot"],
                row["transaction_id"],
            ):
                _refuse("weights receipt changed its admitted identity")
            _bound(
                _bytes(receipt["tensorfs_receipt_canonical_bytes"]),
                _digest(receipt["tensorfs_receipt_digest"]),
                "native receipt",
            )
            _digest(row["manifest_digest"])
            _number(row["manifest_length"], positive=True)
        elif any(
            row[k]
            for k in (
                "receipt_digest",
                "receipt_canonical_bytes",
                "manifest_digest",
                "manifest_length",
            )
        ):
            _refuse("uncommitted weights carry a final receipt")
        if not isinstance(row["objects"], list):
            _refuse("weights objects are not a bounded inventory")
        seen = set()
        for obj in row["objects"]:
            _fields(obj, "object_id length source_ref")
            _digest(obj["object_id"])
            _number(obj["length"])
            if (
                not isinstance(obj["source_ref"], str)
                or len(obj["source_ref"]) > 512
                or obj["object_id"] in seen
            ):
                _refuse("weights object identity is repeated or unbounded")
            seen.add(obj["object_id"])
    elif kind == "hold":
        _fields(row, "retention_id owner_scope transaction_id native_receipt_digest state")
        _digest(row["retention_id"])
        _digest(row["transaction_id"])
        _digest(row["native_receipt_digest"])
        _state(row["state"], ("retaining", "retained", "releasing", "released"))
    elif kind in ("cache", "lookup"):
        _fields(
            row,
            "cache_id owner_scope computation_digest state source holds "
            + ("sequence" if kind == "cache" else "consumer_request_id"),
        )
        _text(row["cache_id"], empty=kind == "lookup")
        _digest(row["computation_digest"])
        if kind == "cache":
            _number(row["sequence"], positive=True)
            _state(row["state"], ("retaining", "ready", "evicting"))
        else:
            _text(row["consumer_request_id"])
            _state(row["state"], ("retaining", "ready", "missing", "miss", "released"))
        _source(row["source"], absent=kind == "lookup" and row["state"] in ("miss", "released"))
        if not isinstance(row["holds"], list):
            _refuse("memo holds are not an inventory")
        seen = set()
        for hold in row["holds"]:
            _fields(
                hold,
                "transaction_id retention_id native_receipt_digest manifest_digest manifest_length",
            )
            for key in (
                "transaction_id",
                "retention_id",
                "native_receipt_digest",
                "manifest_digest",
            ):
                _digest(hold[key])
            _number(hold["manifest_length"], positive=True)
            if hold["retention_id"] in seen:
                _refuse("memo retention identity is repeated")
            seen.add(hold["retention_id"])
    else:
        _refuse("record kind is unknown")
    return cast(dict[str, Any], row)


def _start(workspace: Workspace, begin: pb.WorkspaceImportBegin, owner: str) -> None:
    workspace.owner(owner)
    digest = _digest(begin.migration_id)
    if begin.source_schema != 19 or not 0 <= begin.record_count < (1 << 53):
        _refuse("source schema or count is unsupported")
    with workspace.locked(allow_frozen=True) as db:
        state = db.execute("SELECT * FROM workspace_state WHERE singleton=1").fetchone()
        old = db.execute(
            "SELECT * FROM workspace_imports WHERE id=?", (begin.migration_id,)
        ).fetchone()
        if old is not None:
            if (
                old["owner"] != owner
                or old["source_schema"] != 19
                or old["record_count"] != begin.record_count
                or old["digest"] != digest
                or state["migration_id"] != begin.migration_id
            ):
                _refuse("retry differs from the frozen migration")
            return
        if (
            state["state"] != "active"
            or state["migration_id"]
            or any(db.execute("SELECT 1 FROM " + table + " LIMIT 1").fetchone() for table in TABLES)
        ):
            _refuse("existing Runtime authority cannot be overwritten")
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute(
                "INSERT INTO "
                "workspace_imports(id,record_count,digest,state,owner,source_schema) "
                "VALUES(?,?,?,'receiving',?,19)",
                (begin.migration_id, begin.record_count, digest, owner),
            )
            db.execute(
                "UPDATE workspace_state SET state='frozen',migration_id=? WHERE singleton=1",
                (begin.migration_id,),
            )
            db.commit()
        except BaseException:
            db.rollback()
            raise


def _stage(
    workspace: Workspace, identity: str, index: int, kind: str, raw: bytes, owner: str
) -> None:
    parse_record(kind, raw, owner)
    with workspace.locked(allow_frozen=True) as db:
        existing = db.execute(
            "SELECT kind,body FROM workspace_import_records WHERE migration_id=? AND ordinal=?",
            (identity, index),
        ).fetchone()
        if existing is not None:
            if existing["kind"] != kind or existing["body"] != raw:
                _refuse("replayed record changed its immutable bytes")
            return
        pending = db.execute(
            "SELECT state FROM workspace_imports WHERE id=?", (identity,)
        ).fetchone()
        if pending is None or pending["state"] != "receiving":
            _refuse("completed import cannot receive new records")
        db.execute(
            "INSERT INTO workspace_import_records(migration_id,ordinal,kind,body) VALUES(?,?,?,?)",
            (identity, index, kind, raw),
        )


def _records(db: Journal, identity: str) -> Iterator[tuple[int, str, bytes]]:
    index = 0
    while True:
        row = db.execute(
            "SELECT kind,body FROM workspace_import_records WHERE migration_id=? AND ordinal=?",
            (identity, index),
        ).fetchone()
        if row is None:
            return
        yield index, row["kind"], row["body"]
        index += 1


def receive(
    workspace: Workspace, frames: Iterable[pb.WorkspaceImportFrame], authorize: Authorize
) -> pb.WorkspaceImportResult:
    stream = iter(frames)
    first = next(stream, None)
    if first is None or first.WhichOneof("frame") != "begin":
        _refuse("stream must start with its authenticated migration")
    begin = first.begin
    owner = authorize(begin.claim)
    _start(workspace, begin, owner)
    index, last_kind = 0, -1
    aggregate = hashlib.sha256()
    pending: pb.WorkspaceImportRecord | None = None
    body = bytearray()
    committed = False
    for frame in stream:
        which = frame.WhichOneof("frame")
        if committed:
            _refuse("trailing frame after commit")
        if which == "record":
            row = frame.record
            if (
                pending is not None
                or row.index != index
                or index >= begin.record_count
                or row.kind not in KINDS
                or not 0 < row.length <= MAX_RECORD
                or len(row.digest) != 32
                or KINDS.index(row.kind) < last_kind
            ):
                _refuse("record ordering, identity or bound changed")
            pending = row
            body = bytearray()
        elif which == "chunk":
            part = frame.chunk
            if (
                pending is None
                or part.index != index
                or part.offset != len(body)
                or not 0 < len(part.data) <= MAX_CHUNK
                or len(body) + len(part.data) > pending.length
            ):
                _refuse("record chunk offset or bound changed")
            body.extend(part.data)
            if len(body) == pending.length:
                raw = bytes(body)
                _bound(raw, pending.digest, "record")
                if authorize(begin.claim) != owner:
                    _refuse("authenticated owner changed during transfer")
                _stage(workspace, begin.migration_id, index, pending.kind, raw, owner)
                aggregate.update(pending.kind.encode() + b"\0" + raw + b"\n")
                last_kind = KINDS.index(pending.kind)
                pending = None
                body.clear()
                index += 1
        elif which == "commit":
            if (
                pending is not None
                or index != begin.record_count
                or frame.commit.record_count != index
                or frame.commit.digest != aggregate.digest()
                or begin.migration_id != documents.spell(aggregate.digest())
            ):
                _refuse("commit differs from the complete frozen export")
            committed = True
        else:
            _refuse("unexpected stream frame")
    if not committed:
        _refuse("stream ended before commit")
    if authorize(begin.claim) != owner:
        _refuse("authenticated owner changed before commit")
    _import(
        workspace, begin.migration_id, owner, begin.record_count, lambda: authorize(begin.claim)
    )
    return pb.WorkspaceImportResult(
        migration_id=begin.migration_id, record_count=index, imported=True
    )


def activate(
    workspace: Workspace, call: pb.WorkspaceActivationCall, authorize: Authorize
) -> pb.WorkspaceImportResult:
    owner = authorize(call.claim)
    _digest(call.migration_id)
    with workspace.locked(allow_frozen=True) as db:
        row = db.execute(
            "SELECT * FROM workspace_imports WHERE id=?", (call.migration_id,)
        ).fetchone()
        state = db.execute("SELECT * FROM workspace_state WHERE singleton=1").fetchone()
        if (
            row is None
            or row["owner"] != owner
            or row["state"] not in ("imported", "active")
            or state["migration_id"] != call.migration_id
        ):
            _refuse("activation has no exact committed import")
        if authorize(call.claim) != owner:
            _refuse("activation owner changed while waiting for workspace authority")
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute("UPDATE workspace_state SET state='active' WHERE singleton=1")
            db.execute(
                "UPDATE workspace_imports SET state='active' WHERE id=?", (call.migration_id,)
            )
            db.commit()
        except BaseException:
            db.rollback()
            raise
        return pb.WorkspaceImportResult(
            migration_id=call.migration_id, record_count=row["record_count"], imported=True
        )


def _native(workspace: Workspace, kind: str, row: dict[str, Any]) -> dict[str, Any]:
    store = fill.tensorfs_module().Store.open(str(workspace.store_root))
    if kind == "weights":
        transaction = row["transaction_id"]
        found = store.derived_lookup(transaction)
        state = found["state"]
        if state == "open" and (writer := found.get("writer_session_id")):
            if writer > row["writer_epoch"]:
                _refuse("native writer advanced beyond frozen Host epoch")
            store.derived_fence(transaction, writer)
        if row["state"] == "intent":
            for name in ("restore_checkpoint", "checkpoint"):
                if data := _bytes(row[name]):
                    checkpoint = pb.CheckpointRef.FromString(data)
                    store.validate_derived_checkpoint(
                        transaction,
                        _bytes(row["declaration_canonical_bytes"]),
                        documents.spell(checkpoint.head.digest),
                        checkpoint.head.length,
                        operation_id=row["request_id"],
                        slot=row["output_slot"],
                        plan_digest=documents.spell(checkpoint.plan_digest),
                    )
            return {
                "released": state == "abandoned"
                or found.get("disposition", {}).get("kind") == "released"
            }
        outer = documents.read(_bytes(row["receipt_canonical_bytes"]), pb.WeightsReceipt)
        native = _bytes(outer["tensorfs_receipt_canonical_bytes"])
        if state != "committed" or canonical_json.encode(found.get("receipt")) != native:
            _refuse("recorded weights have no exact native receipt")
        facts = found["receipt"]
        if canonical_json.encode(facts["declaration"]) != _bytes(
            row["declaration_canonical_bytes"]
        ):
            _refuse("native declaration differs from frozen writer intent")
        manifest = facts["manifest"]
        if (row["manifest_digest"], row["manifest_length"]) != (
            "sha256:" + manifest["sha256"],
            manifest["length"],
        ):
            _refuse("recorded manifest differs from native result")
        released = found.get("disposition", {}).get("kind") == "released"
        if not released:
            store.inspect_derived_source(
                row["manifest_digest"],
                row["manifest_length"],
                sorted(facts["declaration"]["components"]),
                [],
            )
            held = store.acquire_cozytensors(row["manifest_digest"])
            held.release()
            expected = {
                row["manifest_digest"]: (
                    row["manifest_length"],
                    "tensorfs-manifest:" + row["manifest_digest"],
                )
            }
            for obj in store.walk(row["manifest_digest"]):
                expected[obj["id"]] = (obj["length"], "tensorfs-object:" + obj["id"])
            received = {
                obj["object_id"]: (obj["length"], obj["source_ref"]) for obj in row["objects"]
            }
            if expected != received:
                _refuse("weights object inventory differs from native manifest")
        return {"released": released, "native_digest": outer["tensorfs_receipt_digest"]}
    if kind == "hold":
        release = row["state"] in ("releasing", "released")
        action = store.release_derived_retention if release else store.retain_derived_result
        result = action(row["transaction_id"], row["native_receipt_digest"], row["retention_id"])
        if (
            result["transaction_id"],
            result["receipt_digest"],
            result["retention_id"],
            result["released"],
        ) != (row["transaction_id"], row["native_receipt_digest"], row["retention_id"], release):
            _refuse("native hold changed its frozen subject")
        manifest = result.get("manifest")
        if not release and not manifest:
            _refuse("active hold has no native manifest")
        return {"released": release, "manifest": manifest}
    return {}


def _unique_subjects(db: Journal, identity: str) -> None:
    subjects = {
        "attempt": (("owner_scope", "request_id", "attempt_ordinal"),),
        "weights": (
            ("transaction_id",),
            ("owner_scope", "request_id", "invocation_spec_digest", "output_slot"),
        ),
        "hold": (("retention_id",),),
        "cache": (("cache_id",), ("owner_scope", "computation_digest"), ("sequence",)),
        "lookup": (("owner_scope", "consumer_request_id"),),
    }
    for kind, keys in subjects.items():
        for fields in keys:
            grouping = ",".join(
                "json_extract(CAST(body AS TEXT),'$." + field + "')" for field in fields
            )
            duplicate = db.execute(
                "SELECT 1 FROM workspace_import_records WHERE migration_id=? AND kind=? GROUP BY "
                + grouping
                + " HAVING count(*)>1 LIMIT 1",
                (identity, kind),
            ).fetchone()
            if duplicate is not None:
                _refuse("duplicate journal subject before native validation")


def _validate_links(db: Journal, identity: str, owner: str, *, native: bool) -> None:
    # Staging contains canonical records. SQL extraction joins bounded metadata,
    # not an in-memory replica of all migration records.
    def staged(kind: str) -> Iterator[dict[str, Any]]:
        cursor = db.execute(
            "SELECT body FROM workspace_import_records WHERE migration_id=? AND kind=? ORDER "
            "BY ordinal",
            (identity, kind),
        )
        for record in cursor:
            yield json.loads(record[0])

    for row in staged("weights"):
        attempt = db.execute(
            "SELECT body FROM workspace_import_records WHERE migration_id=? AND "
            "kind='attempt' AND json_extract(CAST(body AS TEXT),'$.request_id')=? AND "
            "json_extract(CAST(body AS TEXT),'$.attempt_ordinal')=?",
            (identity, row["request_id"], row["attempt_ordinal"]),
        ).fetchall()
        if len(attempt) != 1:
            _refuse("weights have no unique imported attempt")
        accepted = json.loads(attempt[0][0])
        if (
            accepted["owner_scope"] != owner
            or accepted["invocation_spec_digest"] != row["invocation_spec_digest"]
        ):
            _refuse("weights do not bind their accepted invocation")
    for kind in ("cache", "lookup"):
        for row in staged(kind):
            if not row["source"]["request_id"]:
                if row["holds"]:
                    _refuse("empty lookup source carries native holds")
                continue
            source = row["source"]
            history = db.execute(
                "SELECT body FROM workspace_import_records WHERE migration_id=? AND kind='attempt' "
                "AND json_extract(CAST(body AS TEXT),'$.request_id')=? "
                "AND json_extract(CAST(body AS TEXT),'$.attempt_ordinal')=?",
                (identity, source["request_id"], source["attempt_ordinal"]),
            ).fetchall()
            if history:
                if len(history) != 1:
                    _refuse("memo source has ambiguous imported attempt history")
                accepted = json.loads(history[0][0])
                if (
                    accepted["owner_scope"] != owner
                    or accepted["state"] != "outcome"
                    or any(
                        accepted[field] != source[field]
                        for field in (
                            "invocation_spec_digest",
                            "outcome_id",
                            "outcome_digest",
                            "outcome_canonical_bytes",
                        )
                    )
                ):
                    _refuse("memo source differs from its qualified imported attempt")
            outcome = documents.read(
                _bytes(source["outcome_canonical_bytes"]), pb.AttemptOutcomeBody
            )
            receipts: dict[str, tuple[str, dict[str, Any]]] = {}
            for reference in outcome.get("weights_receipts", []):
                raw = _bytes(reference["weights_receipt_canonical_bytes"])
                _bound(raw, _digest(reference["weights_receipt_digest"]), "memo output")
                receipt = documents.read(raw, pb.WeightsReceipt)
                if (
                    receipt["owner_authority_scope"] != owner
                    or receipt["request_id"] != row["source"]["request_id"]
                    or receipt["invocation_spec_digest"] != row["source"]["invocation_spec_digest"]
                ):
                    _refuse("memo receipt belongs to another computation")
                native_bytes = _bytes(receipt["tensorfs_receipt_canonical_bytes"])
                _bound(
                    native_bytes, _digest(receipt["tensorfs_receipt_digest"]), "memo native receipt"
                )
                facts = json.loads(native_bytes)
                transaction = receipt["weights_transaction_id"]
                if transaction in receipts:
                    _refuse("memo repeats a produced transaction")
                receipts[transaction] = (receipt["tensorfs_receipt_digest"], facts["manifest"])
            outputs = list(outcome.get("output_manifest", {}).get("outputs", []))
            if blob := outcome.get("result", {}).get("result_blob"):
                outputs.append(blob)
            manifests = {(facts[1]["sha256"], facts[1]["length"]) for facts in receipts.values()}
            if any(
                (output["digest"].removeprefix("sha256:"), output["length"]) not in manifests
                for output in outputs
            ):
                _refuse("memo contains an output without native retained custody")
            if len(receipts) != len(row["holds"]):
                _refuse("memo omits native output ownership")
            for hold in row["holds"]:
                bound_receipt = receipts.pop(hold["transaction_id"], None)
                if bound_receipt is None or bound_receipt != (
                    hold["native_receipt_digest"],
                    {"sha256": hold["manifest_digest"][7:], "length": hold["manifest_length"]},
                ):
                    _refuse("memo hold differs from its produced result")
                binding = db.execute(
                    "SELECT body,verified FROM workspace_import_records WHERE migration_id=? "
                    "AND kind='hold' AND json_extract(CAST(body AS TEXT),'$.retention_id')=?",
                    (identity, hold["retention_id"]),
                ).fetchall()
                if len(binding) != 1:
                    _refuse("memo retention lacks a unique journal obligation")
                claim = json.loads(binding[0][0])
                verified = json.loads(binding[0][1]) if native else {}
                if (
                    claim["owner_scope"],
                    claim["transaction_id"],
                    claim["native_receipt_digest"],
                ) != (owner, hold["transaction_id"], hold["native_receipt_digest"]):
                    _refuse("memo retention changed owner or native subject")
                live = row["state"] in ("ready", "retaining")
                if live and claim["state"] not in ("retaining", "retained"):
                    _refuse("live memo points at a released hold obligation")
                if (
                    live
                    and native
                    and (verified["released"] or verified["manifest"] != bound_receipt[1])
                ):
                    _refuse("ready memo has no verified live native hold")


def _insert(db: Journal, kind: str, row: dict[str, Any], verified: dict[str, Any]) -> None:
    if kind == "attempt":
        db.execute(
            "INSERT INTO "
            "attempts(owner,request,ordinal,spec,invocation,state,outcome_id,outcome_digest,outcome,retain,fenced)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["owner_scope"],
                row["request_id"],
                row["attempt_ordinal"],
                _digest(row["invocation_spec_digest"]),
                _bytes(row["invocation_spec_canonical_bytes"]),
                "accepted" if row["state"] == "received" else row["state"],
                row["outcome_id"],
                _digest(row["outcome_digest"], empty=True),
                _bytes(row["outcome_canonical_bytes"]),
                row["retain_work"],
                row["state"] in ("received", "running"),
            ),
        )
    elif kind == "weights":
        db.execute(
            "INSERT INTO "
            "weights(id,owner,request,ordinal,spec,slot,declaration_digest,declaration,epoch,state,ready,restore_checkpoint,checkpoint,receipt,receipt_digest,native_digest,manifest,manifest_length,objects)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                row["transaction_id"],
                row["owner_scope"],
                row["request_id"],
                row["attempt_ordinal"],
                _digest(row["invocation_spec_digest"]),
                row["output_slot"],
                _digest(row["declaration_digest"]),
                _bytes(row["declaration_canonical_bytes"]),
                row["writer_epoch"],
                "released" if verified["released"] else row["state"],
                row["intent_ready"],
                _bytes(row["restore_checkpoint"]),
                _bytes(row["checkpoint"]),
                _bytes(row["receipt_canonical_bytes"]),
                _digest(row["receipt_digest"], empty=True),
                _digest(verified.get("native_digest", ""), empty=True),
                _digest(row["manifest_digest"], empty=True),
                row["manifest_length"],
                canonical_json.encode(row["objects"]),
            ),
        )
    elif kind == "hold":
        manifest = verified.get("manifest")
        db.execute(
            "INSERT INTO "
            "holds(id,owner,transaction_id,native_digest,state,manifest,manifest_length) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                row["retention_id"],
                row["owner_scope"],
                row["transaction_id"],
                _digest(row["native_receipt_digest"]),
                "released" if verified["released"] else "held",
                bytes.fromhex(manifest["sha256"]) if manifest else b"",
                manifest["length"] if manifest else 0,
            ),
        )
    else:
        body = canonical_json.encode({"source": row["source"], "holds": row["holds"]})
        if kind == "cache":
            db.execute(
                "INSERT INTO operation_cache(sequence,id,owner,key,body,state) VALUES(?,?,?,?,?,?)",
                (
                    row["sequence"],
                    row["cache_id"],
                    row["owner_scope"],
                    _digest(row["computation_digest"]),
                    body,
                    row["state"],
                ),
            )
        else:
            db.execute(
                "INSERT INTO operation_lookups(owner,consumer,key,cache_id,body,state) "
                "VALUES(?,?,?,?,?,?)",
                (
                    row["owner_scope"],
                    row["consumer_request_id"],
                    _digest(row["computation_digest"]),
                    row["cache_id"],
                    body,
                    row["state"],
                ),
            )


def _import(
    workspace: Workspace, identity: str, owner: str, count: int, authorize: Callable[[], str]
) -> None:
    with workspace.locked(allow_frozen=True) as db:
        migration = db.execute(
            "SELECT state FROM workspace_imports WHERE id=?", (identity,)
        ).fetchone()
        if migration["state"] in ("imported", "active"):
            return
        total = db.execute(
            "SELECT count(*) FROM workspace_import_records WHERE migration_id=?", (identity,)
        ).fetchone()[0]
        if total != count:
            _refuse("staged record count differs from complete stream")
        _unique_subjects(db, identity)
        _validate_links(db, identity, owner, native=False)
        for index, kind, raw in _records(db, identity):
            row = parse_record(kind, raw, owner)
            verified = db.native(partial(_native, workspace, kind, row))
            phase = db.execute(
                "SELECT state,migration_id FROM workspace_state WHERE singleton=1"
            ).fetchone()
            if phase["migration_id"] != identity or phase["state"] != "frozen":
                _refuse("workspace activation advanced during import validation")
            db.execute(
                "UPDATE workspace_import_records SET verified=? WHERE migration_id=? AND ordinal=?",
                (canonical_json.encode(verified), identity, index),
            )
        _validate_links(db, identity, owner, native=True)
        if authorize() != owner:
            _refuse("owner authority changed during native verification")
        db.execute("BEGIN IMMEDIATE")
        try:
            if any(
                db.execute("SELECT 1 FROM " + table + " LIMIT 1").fetchone() for table in TABLES
            ):
                _refuse("new Runtime state cannot be overwritten by an import")
            for index, kind, raw in _records(db, identity):
                verified = db.execute(
                    "SELECT verified FROM workspace_import_records WHERE migration_id=? AND "
                    "ordinal=?",
                    (identity, index),
                ).fetchone()[0]
                _insert(db, kind, json.loads(raw), json.loads(verified))
            db.execute("UPDATE workspace_imports SET state='imported' WHERE id=?", (identity,))
            db.execute(
                "UPDATE workspace_state SET state='imported' WHERE singleton=1 AND migration_id=?",
                (identity,),
            )
            db.commit()
        except BaseException:
            db.rollback()
            raise
