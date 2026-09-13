"""Frozen Python call intents and effect send markers in the existing workspace journal."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, cast

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker.workspace import Journal, Row, Workspace, WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import transaction
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

SCHEMA8 = """
CREATE TABLE execution_calls (
 owner TEXT NOT NULL, parent_request TEXT NOT NULL, call_index INTEGER NOT NULL,
 parent_ordinal INTEGER NOT NULL, parent_generation INTEGER NOT NULL,
 intent_digest BLOB NOT NULL, intent BLOB NOT NULL, child_request TEXT NOT NULL,
 sent INTEGER NOT NULL DEFAULT 0, result BLOB NOT NULL DEFAULT x'',
 safe_code TEXT NOT NULL DEFAULT '', safe_detail TEXT NOT NULL DEFAULT '',
 created_ms INTEGER NOT NULL,
 PRIMARY KEY(owner,parent_request,call_index), UNIQUE(owner,child_request)
) STRICT;
"""


@dataclass(frozen=True)
class Call:
    parent_request: str
    call_index: int
    parent_ordinal: int
    parent_generation: int
    intent_digest: bytes
    intent: bytes
    child_request: str
    sent: bool
    result: bytes
    safe_code: str
    safe_detail: str


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
            raise WorkspaceRefusal("call parent is no longer the admitted running generation")
        return cast(Row, row)

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
        body: dict[str, Any] = {
            "interface_digest": documents.spell(request.interface_digest),
            "module": request.module,
            "export": request.export,
            "request": value,
        }
        if request.HasField("capture"):
            body["capture"] = documents.body(request.capture)
        intent = canonical_json.encode(body)
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

    def before_write(self, owner: str, call: Call) -> None:
        """Record possible remote mutation before IO, under the current parent fence."""
        with self.workspace.locked() as db, transaction(db):
            row = self._row(db, owner, call.parent_request, call.call_index)
            self._parent(
                db, owner, call.parent_request, call.parent_ordinal, call.parent_generation
            )
            if row["intent_digest"] != call.intent_digest or row["result"] or row["safe_code"]:
                raise WorkspaceRefusal("call cannot issue another write after settlement")
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
                or row["safe_code"]
                or row["result"] not in (b"", result)
            ):
                raise WorkspaceRefusal("call result changed its authorized write or prior result")
            db.execute(
                "UPDATE execution_calls SET result=? "
                "WHERE owner=? AND parent_request=? AND call_index=?",
                (result, owner, call.parent_request, call.call_index),
            )
