"""Bounded same-process facts used by one Runtime worker."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any

KINDS = (
    "session",
    "ownership",
    "placement_set",
    "accepted",
    "outcome",
    "closed",
    "epoch",
    "ledger",
    "triage",
    "job_checkpoint",
    "publication",
)


@dataclass(slots=True)
class Record:
    seq: int
    kind: str
    at_unix_ms: int
    body: dict[str, Any]


@dataclass(slots=True)
class WorkerRecords:
    """The current process's ordered facts; the caller owns restart recovery."""

    seq: int = 0
    records: list[Record] = field(default_factory=list)

    @property
    def appends(self) -> int:
        return len(self.records)

    def append(self, kind: str, body: dict[str, Any]) -> Record:
        if kind not in KINDS:
            raise ValueError(f"record kind {kind!r} is not one of {KINDS}")
        record = Record(self.seq + 1, kind, int(time.time() * 1000), body)
        self.seq = record.seq
        self.records.append(record)
        return record

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
