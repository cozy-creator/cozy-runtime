"""Bounded same-process facts used by one Runtime worker."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field

import msgspec

KINDS = (
    "session",
    "ownership",
    "placement_set",
    "accepted",
    "entered",
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
    body: dict[str, object]


@dataclass(slots=True)
class WorkerRecords:
    """The current process's ordered facts; the caller owns restart recovery."""

    seq: int = 0
    records: list[Record] = field(default_factory=list)

    @property
    def appends(self) -> int:
        return len(self.records)

    def append(self, kind: str, body: dict[str, object]) -> Record:
        if kind not in KINDS:
            raise ValueError(f"record kind {kind!r} is not one of {KINDS}")
        record = Record(self.seq + 1, kind, int(time.time() * 1000), body)
        self.seq = record.seq
        self.records.append(record)
        return record

    def of_kind(self, kind: str) -> list[Record]:
        return [record for record in self.records if record.kind == kind]

    def outcome_bytes(self, key: tuple[str, int]) -> bytes:
        """The canonical outcome bytes the attempt's latest `outcome` record carries."""
        for record in reversed(self.of_kind("outcome")):
            outcome = msgspec.convert(record.body, _Outcome)
            if (outcome.request_id, outcome.attempt) == key:
                return base64.b64decode(outcome.outcome_canonical_b64)
        return b""


class _Outcome(msgspec.Struct, frozen=True):
    """What `outcome_bytes` reads of an `outcome` record."""

    request_id: str
    attempt: int
    outcome_canonical_b64: str
