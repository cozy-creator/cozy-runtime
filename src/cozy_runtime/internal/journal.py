"""The durable attempt/terminal journal — the worker's at-most-once evidence.

Journal-before-send is the non-negotiable order (worker-protocol/02 §2): acceptance persists
before AttemptAccepted, the terminal persists before AttemptTerminal, closure records only on
TerminalAck. Everything the supervisor may later CLAIM about an attempt is a line in this
file, and a line is written with `fsync` before the claim leaves the process.

Journal honesty (§4): ENOSPC, corruption or I/O uncertainty means NO acknowledgement, NO
execution, and no further acceptance — at-most-once is never claimed from memory. The
mechanism is `stopped`: the first failed append latches it, and the supervisor's admission
path reads it before anything else.

Line format is `<sha256 hex> <canonical json>`: the digest is over exactly the canonical
bytes that follow, so a flipped byte anywhere in the record is a detected corruption rather
than a plausible different record, and a re-emitted (pretty-printed, reordered) line refuses
under the re-emit law. Recovery reads forward and stops at the first unreadable line — a
torn tail from a crash mid-write is the ONE thing that truncates without refusing, and it is
reported as a torn tail rather than silently dropped.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical

#: Record kinds. Closed on purpose: a new kind is a schema change, not a new string.
KINDS = (
    "session",  # a new session minted (control-parent boot)
    "accepted",  # PREPARED acceptance: spec + plan + model-construction digests
    "terminal",  # the exact canonical TerminalBody bytes, journaled before send
    "closed",  # TerminalAck bound by terminal_id + terminal_digest
    "epoch",  # executor incarnation / readiness epoch movement
    "warm",  # one boot warm-pass outcome
    "ledger",  # one attempt's byte-class reconciliation at close (cr-008a)
    "triage",  # one WorkerTriageBundle write receipt (cr-011): subject, digest, length
    "job_checkpoint",  # one DURABLE job checkpoint identity + receipt (cr-009)
    "publication",  # ONE canonical bundle snapshot per publication (cr-009)
)


class JournalError(Exception):
    """A durability refusal. Its presence means the worker must not claim anything."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(slots=True)
class Record:
    seq: int
    kind: str
    at_unix_ms: int
    body: dict[str, Any]


@dataclass(slots=True)
class Journal:
    """An append-only, fsynced, self-verifying record log."""

    path: Path
    _fd: int = -1
    seq: int = 0
    stopped: str = ""
    torn_tail: bool = False
    appends: int = 0
    fsync_ns: int = 0
    records: list[Record] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = list(self._load())
        self.seq = self.records[-1].seq if self.records else 0
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)

    # ------------------------------------------------------------------ read

    def _load(self) -> Iterator[Record]:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        for index, line in enumerate(raw.split(b"\n")):
            if not line:
                continue
            head, _, rest = line.partition(b" ")
            if len(head) != 64 or not rest:
                self._tear(index, line, "line is not `<digest> <canonical json>`")
                return
            if hashlib.sha256(rest).hexdigest() != head.decode(errors="replace"):
                raise JournalError(
                    "journal_corrupt",
                    f"{self.path}:{index + 1}: record digest does not match its bytes — "
                    "the journal is the at-most-once evidence and cannot be guessed at",
                )
            try:
                value = canonical.parse_canonical(rest)
            except canonical.CanonicalError as exc:
                raise JournalError("journal_corrupt", f"{self.path}:{index + 1}: {exc}") from exc
            yield Record(value["seq"], value["kind"], value["at_unix_ms"], value["body"])

    def _tear(self, index: int, line: bytes, why: str) -> None:
        """A torn tail is a crash mid-append: admissible ONLY as the last line."""
        if index != len(self.path.read_bytes().split(b"\n")) - 2:
            raise JournalError("journal_corrupt", f"{self.path}:{index + 1}: {why}")
        self.torn_tail = True

    # ------------------------------------------------------------------ write

    def append(self, kind: str, body: dict[str, Any]) -> Record:
        """Persist one record and fsync it. A failure LATCHES `stopped` and refuses."""
        if kind not in KINDS:
            raise JournalError("journal_kind", f"{kind!r} is not one of {KINDS}")
        if self.stopped:
            raise JournalError("journal_stopped", self.stopped)
        record = Record(self.seq + 1, kind, int(time.time() * 1000), body)
        data = canonical.write(
            {"seq": record.seq, "kind": kind, "at_unix_ms": record.at_unix_ms, "body": body}
        )
        line = hashlib.sha256(data).hexdigest().encode() + b" " + data + b"\n"
        try:
            written = os.write(self._fd, line)
            if written != len(line):
                raise OSError(f"short write: {written} of {len(line)} B")
            started = time.perf_counter_ns()
            os.fsync(self._fd)
            self.fsync_ns += time.perf_counter_ns() - started
        except OSError as exc:
            self.stopped = (
                f"journal append failed ({exc.__class__.__name__}: {exc}) — the worker "
                "stops accepting and executes nothing: at-most-once is never claimed from "
                "memory"
            )
            raise JournalError("journal_unwritable", self.stopped) from exc
        self.seq = record.seq
        self.appends += 1
        self.records.append(record)
        return record

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    # ------------------------------------------------------------------ views

    def of_kind(self, kind: str) -> list[Record]:
        return [r for r in self.records if r.kind == kind]

    def terminal_bytes(self, key: tuple[str, int]) -> bytes:
        """One attempt's journaled canonical terminal, re-read.

        The DURABLE record is here, so a closed attempt does not have to hold a second copy
        of it in memory for a replay that may never come — and a terminal can carry a 4 MiB
        inline result (codex audit, adopted 2026-08-25). Empty when the journal has no
        terminal for that ordinal, which the caller must treat as "nothing to replay".
        """
        row = self.attempts().get(key)
        if row is None or "terminal_canonical_b64" not in row:
            return b""
        return base64.b64decode(row["terminal_canonical_b64"])

    def attempts(self) -> dict[tuple[str, int], dict[str, Any]]:
        """Replay into the attempt table: acceptance, terminal body, closure.

        `state` is one of `accepted` (accepted-but-incomplete — §2's ABANDONED case),
        `terminal` (journaled, awaiting TerminalAck) or `closed` (compactable).
        """
        table: dict[tuple[str, int], dict[str, Any]] = {}
        for record in self.records:
            body = record.body
            if record.kind == "accepted":
                key = (body["request_id"], body["attempt"])
                table[key] = {"state": "accepted", **body}
            elif record.kind == "terminal":
                key = (body["request_id"], body["attempt"])
                entry = table.get(key)
                if entry is None:
                    # A terminal with no journaled assignment cannot be replayed: the
                    # three-authorization rule (§2) needs the assignment.
                    entry = table[key] = {"state": "unassigned", **body}
                entry.update(state="terminal", **body)
            elif record.kind == "closed":
                key = (body["request_id"], body["attempt"])
                if key in table:
                    table[key]["state"] = "closed"
        return table
