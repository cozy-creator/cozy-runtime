"""Frozen Python call intents and effect send markers in the existing workspace journal."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Literal, cast

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal.call_intent import canonical_intent
from cozy_runtime.internal.effect_interfaces import MODULE
from cozy_runtime.internal.worker.workspace import Journal, Row, Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import transaction
from cozy_runtime.protocol import worker_pb2 as pb

SCHEMA8 = """
CREATE TABLE execution_calls (
 owner TEXT NOT NULL, parent_request TEXT NOT NULL, call_index INTEGER NOT NULL,
 parent_ordinal INTEGER NOT NULL, parent_generation INTEGER NOT NULL,
 intent_digest BLOB NOT NULL, intent BLOB NOT NULL, child_request TEXT NOT NULL,
 prepared BLOB NOT NULL DEFAULT x'',
 sent INTEGER NOT NULL DEFAULT 0, result BLOB NOT NULL DEFAULT x'',
 safe_code TEXT NOT NULL DEFAULT '', safe_detail TEXT NOT NULL DEFAULT '',
 created_ms INTEGER NOT NULL,
 PRIMARY KEY(owner,parent_request,call_index), UNIQUE(owner,child_request)
) STRICT;
"""


class CallPlan(msgspec.Struct, frozen=True):
    """A job, serving or source call's frozen plan (`execution_calls.prepared`). An effect
    call freezes its own plan there and never reads it as one of these; every field has a
    default so a plan an older worker froze still reads."""

    kind: str = ""
    installation_id: str = ""
    entrypoint: str = ""
    parent_attempt: int = 0
    computation: str = ""
    result_schema: object = None


class CallIntent(msgspec.Struct, frozen=True):
    """What a call names (`execution_calls.intent`, `call_intent.canonical_intent`)."""

    module: str
    export: str


@dataclass(frozen=True)
class Call:
    parent_request: str
    call_index: int
    parent_ordinal: int
    parent_generation: int
    intent_digest: bytes
    intent: bytes
    child_request: str
    prepared: bytes
    sent: bool
    result: bytes
    safe_code: str
    safe_detail: str
    #: when the parent called: the call's start on its root's timeline
    created_ms: int = 0

    def plan(self) -> CallPlan:
        return canonical_json.decode_as(self.prepared, CallPlan)

    def target(self) -> CallIntent:
        return canonical_json.decode_as(self.intent, CallIntent)


class CallFenced(WorkspaceRefusal):
    """This call may not act now; a later parent generation or its settlement decides."""


class Calls:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    @staticmethod
    def _call(row: Row) -> Call:
        fields: dict[str, Any] = {name: row[name] for name in Call.__dataclass_fields__}
        fields["sent"] = bool(fields["sent"])
        return Call(**fields)

    @staticmethod
    def _row(db: Journal, owner: str, parent: str, index: int) -> Row:
        row = db.execute(
            "SELECT * FROM execution_calls WHERE owner=? AND parent_request=? AND call_index=?",
            (owner, parent, index),
        ).fetchone()
        if row is None:
            raise WorkspaceRefusal("call has no accepted intent")
        return cast(Row, row)

    @staticmethod
    def _parent(
        db: Journal, owner: str, parent: str, ordinal: int, generation: int | None = None
    ) -> Row:
        row = db.execute(
            "SELECT e.*,a.spec AS parent_spec,a.state AS attempt_state,a.fenced "
            "FROM executions e JOIN attempts a "
            "ON a.owner=e.owner AND a.request=e.request AND a.ordinal=e.ordinal "
            "WHERE e.owner=? AND e.request=?",
            (owner, parent),
        ).fetchone()
        if (
            row is None
            or row["ordinal"] != ordinal
            or row["state"] != "running"
            or row["desired"] != "run"
            or row["attempt_state"] != "running"
            or row["fenced"]
            or (generation is not None and row["generation"] != generation)
        ):
            raise CallFenced("call parent is no longer the admitted running generation")
        stopped = db.execute(
            "WITH RECURSIVE ancestors(request) AS (VALUES(?) UNION "
            "SELECT c.parent_request FROM execution_calls c JOIN ancestors a "
            "ON c.child_request=a.request WHERE c.owner=?) "
            "SELECT 1 FROM executions e JOIN ancestors a ON e.request=a.request "
            "WHERE e.owner=? AND (e.desired<>'run' OR e.state<>'running') LIMIT 1",
            (parent, owner, owner),
        ).fetchone()
        if stopped:
            raise CallFenced("call ancestor is no longer an admitted running generation")
        return cast(Row, row)

    @staticmethod
    def unsettled_effects_in(db: Journal, owner: str, root_request: str) -> bool:
        """Include descendants: canceling a root cannot waive an uncertain child effect."""
        return (
            db.execute(
                "WITH RECURSIVE descendants(request) AS (VALUES(?) UNION "
                "SELECT c.child_request FROM execution_calls c JOIN descendants d "
                "ON c.parent_request=d.request WHERE c.owner=?) "
                "SELECT 1 FROM execution_calls c JOIN descendants d ON c.parent_request=d.request "
                "WHERE c.owner=? AND c.sent=1 AND c.result=x'' "
                "AND json_extract(CAST(c.intent AS TEXT),'$.module')=? LIMIT 1",
                (root_request, owner, owner, MODULE),
            ).fetchone()
            is not None
        )

    def canceled(self, owner: str, call: Call) -> bool:
        """Whether the call's parent or an ancestor is canceled, so it will never act again."""
        with self.workspace.locked() as db:
            return (
                db.execute(
                    "WITH RECURSIVE ancestors(request) AS (VALUES(?) UNION "
                    "SELECT c.parent_request FROM execution_calls c JOIN ancestors a "
                    "ON c.child_request=a.request WHERE c.owner=?) "
                    "SELECT 1 FROM executions e JOIN ancestors a ON e.request=a.request "
                    "WHERE e.owner=? AND e.desired='cancel' LIMIT 1",
                    (call.parent_request, owner, owner),
                ).fetchone()
                is not None
            )

    def has_unsettled_effects(self, owner: str, root_request: str) -> bool:
        with self.workspace.locked() as db:
            return self.unsettled_effects_in(db, owner, root_request)

    def accept(self, owner: str, request: pb.ChildCallRequest) -> Call:
        self.workspace.owner(owner)
        if not 0 <= request.call_index < 1 << 32 or len(request.request_canonical_bytes) > 48 << 10:
            raise WorkspaceRefusal("call index or argument bytes exceed their bound")
        value = canonical_json.decode(request.request_canonical_bytes)
        if (
            not isinstance(value, dict)
            or canonical_json.encode(value) != request.request_canonical_bytes
        ):
            raise WorkspaceRefusal("call arguments must be one canonical object")
        intent = canonical_intent(request)
        if hashlib.sha256(intent).digest() != request.intent_digest:
            raise WorkspaceRefusal("call intent does not bind its exact arguments and target")
        with self.workspace.locked() as db, transaction(db):
            parent = self._parent(
                db, owner, request.parent_request_id, request.parent_attempt_ordinal
            )
            if parent["parent_spec"] != request.parent_invocation_spec_digest:
                raise WorkspaceRefusal("call names another parent invocation")
            old = db.execute(
                "SELECT * FROM execution_calls WHERE owner=? AND parent_request=? AND call_index=?",
                (owner, request.parent_request_id, request.call_index),
            ).fetchone()
            if old is not None:
                if old["intent_digest"] != request.intent_digest or old["intent"] != intent:
                    raise WorkspaceRefusal("call index already names a different frozen intent")
                db.execute(
                    "UPDATE execution_calls SET parent_ordinal=?,parent_generation=? "
                    "WHERE owner=? AND parent_request=? AND call_index=?",
                    (
                        request.parent_attempt_ordinal,
                        parent["generation"],
                        owner,
                        request.parent_request_id,
                        request.call_index,
                    ),
                )
            else:
                child = (
                    "call-"
                    + hashlib.sha256(
                        canonical_json.encode(
                            [owner, request.parent_request_id, request.call_index]
                        )
                    ).hexdigest()[:40]
                )
                if db.execute(
                    "SELECT 1 FROM executions WHERE owner=? AND request=?", (owner, child)
                ).fetchone():
                    raise WorkspaceRefusal("child identity is already owned by another submission")
                db.execute(
                    "INSERT INTO execution_calls(owner,parent_request,call_index,parent_ordinal,"
                    "parent_generation,intent_digest,intent,child_request,created_ms) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        owner,
                        request.parent_request_id,
                        request.call_index,
                        request.parent_attempt_ordinal,
                        parent["generation"],
                        request.intent_digest,
                        intent,
                        child,
                        time.time_ns() // 1_000_000,
                    ),
                )
            return self._call(self._row(db, owner, request.parent_request_id, request.call_index))

    def get(self, owner: str, parent: str, index: int) -> Call:
        with self.workspace.locked() as db:
            return self._call(self._row(db, owner, parent, index))

    def child(self, owner: str, child: str) -> Call | None:
        """The call a child request answers; None for a root."""
        with self.workspace.locked() as db:
            row = db.execute(
                "SELECT * FROM execution_calls WHERE owner=? AND child_request=?", (owner, child)
            ).fetchone()
        return self._call(row) if row is not None else None

    def children(self, owner: str, parent: str) -> list[Call]:
        with self.workspace.locked() as db:
            return [
                self._call(row)
                for row in db.execute(
                    "SELECT * FROM execution_calls WHERE owner=? AND parent_request=? "
                    "ORDER BY call_index",
                    (owner, parent),
                )
            ]

    def unsettled_effects(self, owner: str) -> list[str]:
        """Sent effect calls whose remote commit is still in doubt: owed a readback."""
        with self.workspace.locked() as db:
            return [
                row[0]
                for row in db.execute(
                    "SELECT child_request FROM execution_calls WHERE owner=? AND sent=1 "
                    "AND result=x'' AND json_extract(CAST(intent AS TEXT),'$.module')=?",
                    (owner, MODULE),
                )
            ]

    def ancestor_stop(self, owner: str, child: str) -> Literal["", "pause", "cancel"]:
        """How a child follows its stopped ancestors: cancel when one was canceled, pause
        when one paused or failed, nothing while every ancestor runs (or succeeded)."""
        with self.workspace.locked() as db:
            rows = db.execute(
                "WITH RECURSIVE ancestors(request) AS ("
                "SELECT parent_request FROM execution_calls WHERE owner=? AND child_request=? "
                "UNION SELECT c.parent_request FROM execution_calls c JOIN ancestors a "
                "ON c.child_request=a.request WHERE c.owner=?) "
                "SELECT e.desired,e.state FROM executions e JOIN ancestors a "
                "ON e.request=a.request WHERE e.owner=?",
                (owner, child, owner, owner),
            ).fetchall()
        stopped = [
            row
            for row in rows
            if row["desired"] != "run" or row["state"] in ("failed", "canceled", "paused")
        ]
        if not stopped:
            return ""
        if any(row["desired"] == "cancel" or row["state"] == "canceled" for row in stopped):
            return "cancel"
        return "pause"

    def check_active(self, owner: str, call: Call) -> None:
        with self.workspace.locked() as db:
            row = self._row(db, owner, call.parent_request, call.call_index)
            self._parent(
                db, owner, call.parent_request, call.parent_ordinal, call.parent_generation
            )
            if row["intent_digest"] != call.intent_digest or row["result"] or row["safe_code"]:
                raise CallFenced("call cannot act after settlement")

    def refuse(self, owner: str, call: Call, code: str, detail: str) -> None:
        """Settle admission failure only before a child or external write is accepted."""
        if not 0 < len(code.encode()) <= 128 or len(detail.encode()) > 1024:
            raise WorkspaceRefusal("call refusal exceeds its bound")
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, call.parent_request, call.call_index)
            if (
                row["intent_digest"] != call.intent_digest
                or row["sent"]
                or row["result"]
                or (row["safe_code"] and (row["safe_code"], row["safe_detail"]) != (code, detail))
                or db.execute(
                    "SELECT 1 FROM executions WHERE owner=? AND request=?",
                    (owner, call.child_request),
                ).fetchone()
            ):
                raise WorkspaceRefusal("accepted work cannot be replaced by an admission refusal")
            db.execute(
                "UPDATE execution_calls SET safe_code=?,safe_detail=? "
                "WHERE owner=? AND parent_request=? AND call_index=?",
                (code, detail, owner, call.parent_request, call.call_index),
            )

    def unresolved(self, owner: str, call: Call, code: str) -> None:
        """Expose an effect problem without claiming its prior send did not land."""
        if not 0 < len(code.encode()) <= 128:
            raise WorkspaceRefusal("effect failure exceeds its bound")
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, call.parent_request, call.call_index)
            if row["intent_digest"] != call.intent_digest or not row["sent"] or row["result"]:
                raise WorkspaceRefusal("effect has no unresolved send")
            db.execute(
                "UPDATE execution_calls SET safe_code=?,safe_detail=? "
                "WHERE owner=? AND parent_request=? AND call_index=?",
                (
                    code,
                    "publication outcome requires reconciliation",
                    owner,
                    call.parent_request,
                    call.call_index,
                ),
            )

    def uncommitted(self, owner: str, call: Call, code: str, detail: str) -> None:
        """Settle a sent effect whose authority durably answered that its commit did not land."""
        if not 0 < len(code.encode()) <= 128 or len(detail.encode()) > 1024:
            raise WorkspaceRefusal("effect failure exceeds its bound")
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, call.parent_request, call.call_index)
            if row["intent_digest"] != call.intent_digest or not row["sent"] or row["result"]:
                raise WorkspaceRefusal("effect has no unresolved send")
            db.execute(
                "UPDATE execution_calls SET sent=0,safe_code=?,safe_detail=? "
                "WHERE owner=? AND parent_request=? AND call_index=?",
                (code, detail, owner, call.parent_request, call.call_index),
            )

    def freeze(self, owner: str, call: Call, prepared: bytes) -> Call:
        """Bind read-dependent metadata before a write, once for this exact call.

        For example, a release's original revision and complete desired lane set
        are frozen here. Retrying may not read a newer baseline and overwrite it.
        Byte inventories remain in TensorFS; only their identities belong here.
        """
        if not 0 < len(prepared) <= 48 << 10:
            raise WorkspaceRefusal("prepared call metadata exceeds its inline bound")
        if canonical_json.encode(canonical_json.decode(prepared)) != prepared:
            raise WorkspaceRefusal("prepared call metadata must be canonical")
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, call.parent_request, call.call_index)
            if row["intent_digest"] != call.intent_digest:
                raise WorkspaceRefusal("prepared call names a different intent")
            if row["prepared"]:
                if row["prepared"] != prepared:
                    raise WorkspaceRefusal("prepared call cannot change its frozen baseline")
                return self._call(row)
            self._parent(
                db, owner, call.parent_request, call.parent_ordinal, call.parent_generation
            )
            if row["sent"] or row["result"] or row["safe_code"]:
                raise WorkspaceRefusal(
                    "prepared call cannot be invented after a send or settlement"
                )
            db.execute(
                "UPDATE execution_calls SET prepared=? "
                "WHERE owner=? AND parent_request=? AND call_index=?",
                (prepared, owner, call.parent_request, call.call_index),
            )
            return self._call(self._row(db, owner, call.parent_request, call.call_index))

    def observe_result(self, owner: str, call: Call, result: bytes) -> None:
        """Settle read-only convergence without inventing a remote-write marker."""
        if not 0 < len(result) <= 48 << 10:
            raise WorkspaceRefusal("call result exceeds its inline bound")
        if canonical_json.encode(canonical_json.decode(result)) != result:
            raise WorkspaceRefusal("call result must be canonical")
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, call.parent_request, call.call_index)
            self._parent(
                db, owner, call.parent_request, call.parent_ordinal, call.parent_generation
            )
            if (
                row["intent_digest"] != call.intent_digest
                or row["safe_code"]
                or row["result"] not in (b"", result)
            ):
                raise WorkspaceRefusal("call observation changed its intent or prior result")
            db.execute(
                "UPDATE execution_calls SET result=? "
                "WHERE owner=? AND parent_request=? AND call_index=?",
                (result, owner, call.parent_request, call.call_index),
            )

    def before_write(self, owner: str, call: Call) -> None:
        """Record that the effect's commit may be sent, before IO, under the parent fence.

        Idempotent staging an effect can always redo (a checkpoint's object pushes) uses
        ``check_active`` instead: only a possible commit makes a row ``sent``.
        """
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, call.parent_request, call.call_index)
            self._parent(
                db, owner, call.parent_request, call.parent_ordinal, call.parent_generation
            )
            if row["intent_digest"] != call.intent_digest or row["result"] or row["safe_code"]:
                raise CallFenced("call cannot issue another write after settlement")
            db.execute(
                "UPDATE execution_calls SET sent=1 "
                "WHERE owner=? AND parent_request=? AND call_index=?",
                (owner, call.parent_request, call.call_index),
            )

    def complete(self, owner: str, call: Call, result: bytes) -> None:
        """Record an exact completed effect even if cancellation raced its committed write."""
        if not 0 < len(result) <= 48 << 10:
            raise WorkspaceRefusal("call result exceeds its inline bound")
        canonical_json.decode(result)
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, call.parent_request, call.call_index)
            if (
                row["intent_digest"] != call.intent_digest
                or not row["sent"]
                or row["result"] not in (b"", result)
            ):
                raise WorkspaceRefusal("call result changed its authorized write or prior result")
            db.execute(
                "UPDATE execution_calls SET result=?,safe_code='',safe_detail='' "
                "WHERE owner=? AND parent_request=? AND call_index=?",
                (result, owner, call.parent_request, call.call_index),
            )
