"""THE WORKER'S OWN DURABLE ATTEMPT RECORD (th-142 child 2).

`WorkerRecords` used to say "the caller owns restart recovery", and on a pod that caller was
the supervisor: it kept a SQLite ledger of every attempt it forwarded, closed the interrupted
ones at its own boot, and rode the survivors beside the worker's snapshot as a second
document, `HostSnapshotBody/1`. That second document existed only because the record lived in
the wrong process. Nothing it carried was unobservable to the worker that produced it, so the
record moves here and the second document is deleted.

WHAT IS DURABLE, AND WHY EACH ROW IS. Exactly the facts an unacknowledged obligation is made
of, written before the frame that reports them leaves:

  accepted   an attempt this worker ADMITTED. An offer it never accepted owes nothing and is
             not here: acceptance is what makes an attempt a claim.
  outcome    that attempt's terminal — id, digest, and the EXACT canonical bytes, so a replay
             after a restart is byte-identical rather than re-authored.
  closed     the RecordOwner's ack landed. The obligation is over and the rows are dead.
  weights    one weights transaction this worker holds open, in the shape the snapshot
             reports it (`WeightsTransactionStatus`), or its removal.

FORM. One JSON document per line, appended and fsynced, under a root the supervisor creates.
A line is durable when its bytes are all on the disk; a torn tail — the crash that this file
exists for — is not a record and is dropped on the next read. Nothing rewrites a line: the
file is folded, and a fold is what recovery reads.

RECOVERY IS NOT A CLOCK. An attempt with no terminal is not "old"; it is one whose executor is
provably gone, because the process that held it is the one that just died. It gets the
smallest honest terminal — ABANDONED, EXECUTOR_INVALIDATED, INFRA — authored ONCE and appended
before it is used, so a second restart replays the same bytes under the same outcome id.
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

#: The record kinds that outlive this process. Every other kind `WorkerRecords` carries is
#: narration about a run that is over when the process is.
DURABLE = ("accepted", "outcome", "closed", "weights")

FILENAME = "attempts.jsonl"

#: Fold and rewrite once the dead rows outnumber this. The file is bounded by the work in
#: flight, not by the work this worker has ever done.
COMPACT_AFTER = 256

CAUSE = pb.CauseCode
ORIGIN = pb.CauseOrigin
STATUS = pb.OutcomeStatus

INTERRUPTED = (
    "the worker process that held this attempt exited; its execution context is gone and "
    "this ordinal will not run again"
)


class Journal:
    """One append-only file, or nothing at all.

    A worker with no record root — the in-process local door, and every test that does not ask
    for one — gets a journal that holds nothing. That is not a degraded mode: a run whose owner
    is in the same process has no restart to recover from.
    """

    __slots__ = ("dead", "path")

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.dead = 0
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    @classmethod
    def under(cls, root: Path | None) -> Journal:
        return cls(None if root is None else Path(root) / FILENAME)

    def append(self, row: dict[str, Any]) -> None:
        """Durable BEFORE the frame that reports it leaves. No buffering, no batching."""
        if self.path is None:
            return
        line = json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        handle = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(handle, line)
            os.fsync(handle)
        finally:
            os.close(handle)
        if row.get("kind") == "closed":
            self.dead += 1

    def read(self) -> list[dict[str, Any]]:
        """Every whole line, in order. A torn tail never became a fact."""
        if self.path is None or not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_bytes().split(b"\n"):
            if not line:
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue  # the crash boundary: bytes that did not all land are not a record
            if isinstance(row, dict) and row.get("kind") in DURABLE:
                rows.append(row)
        return rows

    def rewrite(self, rows: list[dict[str, Any]]) -> None:
        """Replace the file with the fold, atomically. Only ever fewer facts, never other ones."""
        if self.path is None:
            return
        temporary = self.path.with_suffix(".rewrite")
        body = b"".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
        )
        handle = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(handle, body)
            os.fsync(handle)
        finally:
            os.close(handle)
        os.replace(temporary, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        self.dead = 0


@dataclass(slots=True)
class Recovery:
    """What the previous process left, folded."""

    #: `accepted`/`outcome`/`weights` rows worth keeping, in order — the seed for this
    #: process's `WorkerRecords`, so `attempts()` and `outcome_bytes()` read as they always did.
    rows: list[dict[str, Any]] = field(default_factory=list)
    #: One row per attempt this worker still owes an ack for, each with a terminal.
    attempts: list[dict[str, Any]] = field(default_factory=list)
    #: One row per weights transaction still open, in `WeightsTransactionStatus` shape.
    weights: list[dict[str, Any]] = field(default_factory=list)
    #: The attempts whose terminal THIS boot authored because the last process could not.
    interrupted: list[tuple[str, int]] = field(default_factory=list)


def fold(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The surviving facts, in a stable order: what is still owed and nothing else.

    An acknowledged attempt's three rows collapse to nothing, and a removed weights
    transaction to nothing. This is what bounds the file: it holds the work in flight, never
    the work this worker has ever done.
    """
    accepted: dict[tuple[str, int], dict[str, Any]] = {}
    outcomes: dict[tuple[str, int], dict[str, Any]] = {}
    closed: set[tuple[str, int]] = set()
    weights: dict[str, dict[str, Any]] = {}
    for row in lines:
        body = row.get("body")
        if not isinstance(body, dict):
            continue
        kind = row["kind"]
        if kind == "weights":
            transaction = str(body.get("weights_transaction_id", ""))
            if not transaction:
                continue
            if body.get("removed"):
                weights.pop(transaction, None)
            else:
                weights[transaction] = body
            continue
        key = (str(body.get("request_id", "")), int(body.get("attempt", 0)))
        if not key[0] or key[1] <= 0:
            continue
        if kind == "accepted":
            accepted[key] = body
        elif kind == "outcome":
            outcomes[key] = body
        elif kind == "closed":
            closed.add(key)
    rows: list[dict[str, Any]] = []
    for key in sorted(accepted.keys() | outcomes.keys()):
        if key in closed:
            continue
        if key in accepted:
            rows.append(_row("accepted", accepted[key]))
        if key in outcomes:
            rows.append(_row("outcome", outcomes[key]))
    for transaction in sorted(weights):
        rows.append(_row("weights", weights[transaction]))
    for index, row in enumerate(rows, start=1):
        row["seq"] = index
    return rows


def recover(journal: Journal, *, boot_id: str, instance_id: str) -> Recovery:
    """Fold the file, close what the dead process left open, and hand back what is still owed.

    Order matters and is the whole crash story: the interrupted terminal is APPENDED before it
    is returned, so a worker that dies again between authoring and reporting replays exactly
    these bytes rather than minting a second outcome id for the same ordinal.
    """
    recovery = Recovery()
    lines = journal.read()
    rows = fold(lines)
    admitted: dict[tuple[str, int], dict[str, Any]] = {}
    terminals: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        body = row["body"]
        if row["kind"] == "accepted":
            admitted[(body["request_id"], int(body["attempt"]))] = body
        elif row["kind"] == "outcome":
            terminals[(body["request_id"], int(body["attempt"]))] = body
        else:
            recovery.weights.append(body)
    for key in sorted(admitted.keys() | terminals.keys()):
        terminal = terminals.get(key)
        if terminal is None:
            terminal = _interrupted(key, admitted[key], boot_id=boot_id, instance_id=instance_id)
            if terminal is None:
                continue
            journal.append(_row("outcome", terminal))
            terminals[key] = terminal
            recovery.interrupted.append(key)
        recovery.attempts.append({**admitted.get(key, {}), **terminal})
    if recovery.interrupted:
        rows = fold(lines + [_row("outcome", terminals[key]) for key in recovery.interrupted])
    recovery.rows = rows
    if lines and len(lines) > len(rows):
        journal.rewrite(rows)
    return recovery


def _row(kind: str, body: dict[str, Any]) -> dict[str, Any]:
    return {"seq": 0, "kind": kind, "at_unix_ms": _now(), "body": body}


def _interrupted(
    key: tuple[str, int], admitted: dict[str, Any], *, boot_id: str, instance_id: str
) -> dict[str, Any] | None:
    """The smallest honest terminal for an attempt whose worker died holding it.

    `execution_started` is TRUE for every one of them, and that is a claim this worker can
    make: acceptance binds an executor, so by the time a row is here the work may have run.
    A dead process cannot report otherwise, and the reading that costs a RecordOwner
    something is the one that says it did not.
    """
    spelled = str(admitted.get("invocation_spec_digest", ""))
    if not spelled:
        return None
    try:
        digest = documents.raw(spelled)
    except documents.DocumentError:
        return None
    body = pb.AttemptOutcomeBody(
        request_id=key[0],
        attempt_ordinal=key[1],
        invocation_spec_digest=spelled,
        status=STATUS.OUTCOME_STATUS_ABANDONED,
        safe_message=INTERRUPTED,
        cause=pb.OutcomeCause(
            code=CAUSE.CAUSE_CODE_EXECUTOR_INVALIDATED,
            origin=ORIGIN.CAUSE_ORIGIN_INFRA,
            detail=INTERRUPTED,
        ),
        execution_started=True,
    )
    canonical_bytes, outcome_digest = documents.identity(body)
    return {
        "request_id": key[0],
        "attempt": key[1],
        "worker_boot_id": boot_id,
        "instance_id": instance_id,
        "invocation_spec_digest": documents.spell(digest),
        "placement_id": str(admitted.get("placement_id", "")),
        "outcome_id": f"out-{uuid.uuid4().hex[:24]}",
        "outcome_digest": documents.spell(outcome_digest),
        "outcome_canonical_b64": base64.b64encode(canonical_bytes).decode(),
        "interrupted": True,
    }


def _now() -> int:
    return int(time.time() * 1000)
