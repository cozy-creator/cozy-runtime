"""The facts one Runtime worker holds — the narration in memory, the obligations on disk."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any

from cozy_runtime.internal.worker import durable

KINDS = (
    "session",
    "ownership",
    "placement_set",
    "accepted",
    "outcome",
    "closed",
    "epoch",
    "warm",
    "ledger",
    "triage",
    "job_checkpoint",
    "publication",
    "weights",
)


@dataclass(slots=True)
class Record:
    seq: int
    kind: str
    at_unix_ms: int
    body: dict[str, Any]


@dataclass(slots=True)
class WorkerRecords:
    """This worker's ordered facts, and the durable subset of them (th-142 child 2).

    Most kinds here are narration about a run that ends when the process does. Four are not:
    an attempt this worker accepted, its terminal, its close, and a weights transaction it
    holds open are OBLIGATIONS, and an obligation the worker forgets across a restart is one
    a RecordOwner waits on forever. Those go to `journal` — appended and fsynced HERE, before
    the frame that reports them is built, which is why the write sits at this one seam and not
    at a dozen call sites.

    A journal with no path holds nothing, and that is the whole of the local door's story: a
    run whose owner is in this process has no restart to recover from.
    """

    seq: int = 0
    records: list[Record] = field(default_factory=list)
    journal: durable.Journal = field(default_factory=lambda: durable.Journal(None))

    @property
    def appends(self) -> int:
        return len(self.records)

    def append(self, kind: str, body: dict[str, Any]) -> Record:
        if kind not in KINDS:
            raise ValueError(f"record kind {kind!r} is not one of {KINDS}")
        record = Record(self.seq + 1, kind, int(time.time() * 1000), body)
        if kind in durable.DURABLE:
            self.journal.append(
                {"seq": record.seq, "kind": kind, "at_unix_ms": record.at_unix_ms, "body": body}
            )
            if self.journal.dead >= durable.COMPACT_AFTER:
                # The file holds the work in flight, not the work this worker has ever done.
                # An acknowledged attempt's rows are dead; folding drops them.
                self.journal.rewrite(durable.fold(self.journal.read()))
        self.seq = record.seq
        self.records.append(record)
        return record

    def restore(self, rows: list[dict[str, Any]]) -> None:
        """Seed this process with the previous one's surviving facts, before anything runs."""
        for row in rows:
            record = Record(int(row["seq"]), str(row["kind"]), int(row["at_unix_ms"]), row["body"])
            self.seq = max(self.seq, record.seq)
            self.records.append(record)

    def of_kind(self, kind: str) -> list[Record]:
        return [record for record in self.records if record.kind == kind]

    def outcome_bytes(self, key: tuple[str, int]) -> bytes:
        row = self.attempts().get(key)
        if row is None or "outcome_canonical_b64" not in row:
            return b""
        return base64.b64decode(row["outcome_canonical_b64"])

    def attempts(self) -> dict[tuple[str, int], dict[str, Any]]:
        table: dict[tuple[str, int], dict[str, Any]] = {}
        for record in self.records:
            body = record.body
            if record.kind == "accepted":
                table[(body["request_id"], body["attempt"])] = {"state": "accepted", **body}
            elif record.kind == "outcome":
                key = (body["request_id"], body["attempt"])
                table.setdefault(key, {"state": "unassigned", **body}).update(
                    state="outcome", **body
                )
            elif record.kind == "closed":
                key = (body["request_id"], body["attempt"])
                if key in table:
                    table[key]["state"] = "closed"
        return table
