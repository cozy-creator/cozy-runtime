"""The WorkerTriageBundle: ONE bounded document per terminal attempt. The schema FREEZES here.

Debugging a failed attempt on a dead rented pod is v1's dominant triage cost, and the reason
is always the same: the evidence lived in the process. When the process is gone — killed,
OOMed, reclaimed with the pod — there is nothing left to read, so the answer is a rerun and
a guess. This module exists so that a killed executor is explainable from a FILE.

The bundle answers, in one document and without a live process:

    which spec ran, under which plan and which construction (identity)
    what the worker's applied posture was when it ran (the deployment's own state)
    what it measured — the ledger's byte classes, the attested metrics, where time went
    what it confessed — every serve that differed from the plain reading of the request
    what went wrong — typed faults, the terminal's cause and origin
    what it was saying — the capped tail of the bounded observation ring

Four properties are structural rather than aspirational:

**Bounded, everywhere.** `TriageBundleRef.length` caps the wire reference at 1 MiB
(worker-protocol/01), so the assembler MEASURES its own canonical bytes and sheds the oldest
observations until it fits, recording that it did. Raw unbounded logs never ride the durable
control stream, and there is no field here for them.

**Opaque by identity.** The handle is `subject_id`, minted per terminal attempt. Nothing in
the bundle is addressable by path, and the RecordOwner's own opaque attempt key
(`records.AttemptByKey`, cl-001) maps to it through the terminal — so a triage read is a key
lookup on both sides, never a filesystem walk.

**Durable before it is claimed.** The bytes are written and fsynced, then a WRITE RECEIPT is
appended to the control journal, and only then may a terminal reference them. A terminal
naming a bundle that is not on disk would be exactly the fabrication class the runtime
exists to prevent, so the order is not negotiable.

**Observation only.** Nothing here can replace or override a terminal field
(worker-protocol/01, `TriageBundleRef`). A log cannot declare success.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import canonical
from cozy_runtime.internal.journal import Journal

BUNDLE_FORMAT = "cozy.runtime.WorkerTriageBundle/1"
RECEIPT_FORMAT = "cozy.runtime.TriageWriteReceipt/1"

#: `TriageBundleRef.length` is bounded at 1 MiB by the protocol. The assembler trims to it.
MAX_BUNDLE_BYTES = 1 << 20

#: The document's CLOSED top-level key set. A reader may check it exactly; a writer that
#: wants a new section changes this line, which is a schema change and looks like one.
SECTIONS = (
    "format",
    "subject_id",
    "attempt",
    "plan",
    "posture",
    "terminal",
    "faults",
    "measurements",
    "confessions",
    "liveness",
    "events",
    "caps",
)


class TriageError(Exception):
    """The bundle could not be built or committed. Never silently skipped."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def mint_subject() -> str:
    """The opaque handle. Not derived from the request id: a subject that leaks the
    request's name is an address, and the whole point is that there is no address."""
    return f"trb-{uuid.uuid4().hex[:24]}"


@dataclass(frozen=True, slots=True)
class Receipt:
    """Proof the bundle bytes are durable, and the identity the terminal carries."""

    subject_id: str
    bundle_digest: str
    length: int
    journal_seq: int

    def document(self) -> dict[str, Any]:
        return {
            "format": RECEIPT_FORMAT,
            "subject_id": self.subject_id,
            "bundle_digest": self.bundle_digest,
            "length": self.length,
        }


@dataclass(slots=True)
class Assembly:
    """Everything one terminal attempt contributes. Every field has exactly one producer."""

    subject_id: str
    attempt: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    posture: dict[str, Any] = field(default_factory=dict)
    terminal: dict[str, Any] = field(default_factory=dict)
    faults: list[dict[str, Any]] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    confessions: list[dict[str, Any]] = field(default_factory=list)
    liveness: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    caps: dict[str, Any] = field(default_factory=dict)

    def build(self, limit: int = MAX_BUNDLE_BYTES) -> tuple[dict[str, Any], bytes]:
        """Assemble and TRIM to the wire cap, shedding OLDEST observations first.

        The trim measures the real canonical bytes rather than estimating them: the cap is a
        protocol fact and an estimate that is wrong by one byte produces a terminal nobody
        can accept. What is shed is recorded in `caps`, because a bundle that quietly lost
        its tail is worse than one that says how much it lost.
        """
        events = list(self.events)
        shed = 0
        while True:
            document = self._document(events, shed)
            try:
                data = canonical.write(document)
            except canonical.CanonicalError as exc:
                raise TriageError("bundle_uncanonical", str(exc)) from exc
            if len(data) <= limit:
                return document, data
            if not events:
                raise TriageError(
                    "bundle_irreducible",
                    f"the bundle is {len(data)} B with no observations left to shed and the "
                    f"cap is {limit} B: a fixed section is over budget, which is a schema "
                    "defect rather than a flood",
                )
            drop = max(1, len(events) // 8)
            del events[:drop]
            shed += drop

    def _document(self, events: list[dict[str, Any]], shed: int) -> dict[str, Any]:
        caps = dict(self.caps)
        caps["shed_to_fit"] = shed
        caps["max_bundle_bytes"] = MAX_BUNDLE_BYTES
        caps["truncated"] = bool(shed or caps.get("dropped", 0))
        return {
            "format": BUNDLE_FORMAT,
            "subject_id": self.subject_id,
            "attempt": self.attempt,
            "plan": self.plan,
            "posture": self.posture,
            "terminal": self.terminal,
            "faults": self.faults,
            "measurements": self.measurements,
            "confessions": self.confessions,
            "liveness": self.liveness,
            "events": events,
            "caps": caps,
        }


class BundleStore:
    """Bundle bytes on disk plus a journaled write receipt. Survives the executor's death,
    and — because the receipt is in the control journal — the worker's own."""

    def __init__(self, root: Path, journal: Journal) -> None:
        self.root = root
        self.journal = journal
        self.written = 0
        self.bytes = 0

    def commit(self, assembly: Assembly) -> Receipt:
        """Write, fsync, journal the receipt. In THAT order, and never any other."""
        self.root.mkdir(parents=True, exist_ok=True)
        _, data = assembly.build()
        path = self.root / f"{assembly.subject_id}.json"
        tmp = path.with_suffix(".part")
        handle = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            if os.write(handle, data) != len(data):
                raise TriageError("bundle_short_write", f"{path.name} did not write whole")
            os.fsync(handle)
        finally:
            os.close(handle)
        os.replace(tmp, path)
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        # A failed append raises and the caller ships no reference: the bytes stay on disk
        # unreferenced, which is the SAFE half of the failure. A terminal must never name a
        # bundle whose durability was not recorded.
        journaled = self.journal.append(
            "triage",
            {
                "format": RECEIPT_FORMAT,
                "subject_id": assembly.subject_id,
                "bundle_digest": digest,
                "length": len(data),
            },
        )
        self.written += 1
        self.bytes += len(data)
        return Receipt(assembly.subject_id, digest, len(data), journaled.seq)

    # ------------------------------------------------------------------ reading

    @staticmethod
    def receipts(journal_path: Path) -> dict[str, dict[str, Any]]:
        """Every journaled write receipt, by subject — read from the FILE, no live process."""
        found: dict[str, dict[str, Any]] = {}
        if not journal_path.is_file():
            return found
        for line in journal_path.read_bytes().split(b"\n"):
            _, _, rest = line.partition(b" ")
            if not rest:
                continue
            try:
                row = canonical.parse_canonical(rest)
            except canonical.CanonicalError:
                continue
            if isinstance(row, dict) and row.get("kind") == "triage":
                found[str(row["body"]["subject_id"])] = row["body"]
        return found

    @staticmethod
    def read(root: Path, subject_id: str, journal_path: Path | None = None) -> dict[str, Any]:
        """Read ONE bundle by its opaque subject, verified against its journaled receipt.

        This is the whole triage path: a directory, a key, and a digest. No process, no SSH,
        no socket. When the journal is present the digest is CHECKED against it, so a bundle
        edited after the fact is a detected corruption rather than a plausible story.
        """
        path = root / f"{subject_id}.json"
        if not path.is_file():
            raise TriageError("bundle_absent", f"no triage bundle for subject {subject_id!r}")
        data = path.read_bytes()
        document = canonical.parse_canonical(data)
        if not isinstance(document, dict) or document.get("format") != BUNDLE_FORMAT:
            raise TriageError("bundle_malformed", f"{subject_id}: not a {BUNDLE_FORMAT}")
        if journal_path is not None:
            receipt = BundleStore.receipts(journal_path).get(subject_id)
            if receipt is None:
                raise TriageError(
                    "receipt_absent",
                    f"{subject_id}: bytes exist but no write receipt is journaled - the "
                    "bundle was never durably claimed and is not evidence",
                )
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            if digest != receipt["bundle_digest"] or len(data) != receipt["length"]:
                raise TriageError(
                    "bundle_corrupt",
                    f"{subject_id}: {len(data)} B hashing to {digest[:23]} does not match "
                    f"the journaled receipt ({receipt['length']} B, "
                    f"{str(receipt['bundle_digest'])[:23]})",
                )
        return document


def explain(document: dict[str, Any]) -> list[str]:
    """The bundle, as the lines a person reads first. A PROJECTION, never authority."""
    attempt = document.get("attempt", {})
    terminal = document.get("terminal", {})
    measurements = document.get("measurements", {})
    lines = [
        f"attempt   {attempt.get('request_id')}#{attempt.get('attempt')} "
        f"({document.get('subject_id')})",
        f"spec      {str(attempt.get('exec_spec_digest', ''))[:30]}",
        f"outcome   {terminal.get('status')} / {terminal.get('cause_code')} "
        f"from {terminal.get('cause_origin')}",
        f"because   {terminal.get('safe_message', '')[:300]}",
        f"executor  incarnation {attempt.get('executor_incarnation')} "
        f"admission generation {attempt.get('admission_generation')}",
    ]
    for fault in document.get("faults", ()):
        lines.append(f"fault     {fault.get('reason')}: {str(fault.get('detail', ''))[:200]}")
    for row in document.get("confessions", ()):
        lines.append(f"confessed {row.get('name')}: {row.get('value')}")
    for row in document.get("liveness", ()):
        if row.get("verdict"):
            lines.append(
                f"liveness  {row.get('subject')} [{row.get('step')}] {row.get('verdict')} "
                f"at position {row.get('position')} after {row.get('samples')} observations"
            )
    for row in measurements.get("ledger", {}).get("classes", ()):
        if row.get("verdict") in ("leak", "baseline_shift"):
            lines.append(f"ledger    {row['class']}: {row['verdict']} {row.get('delta')} B")
    attribution = measurements.get("attribution", {})
    for name, track in sorted(attribution.get("stages", {}).items()):
        lines.append(f"stage     {name}: {track['total_ms']} ms over {track['count']}")
    for name, track in sorted(attribution.get("steps", {}).items()):
        lines.append(
            f"steps     {name}: {track['count']} x mean {track['mean_ms']} ms "
            f"(min {track['min_ms']}, max {track['max_ms']})"
        )
    caps = document.get("caps", {})
    lines.append(
        f"tail      {len(document.get('events', ()))} kept, {caps.get('dropped', 0)} shed by "
        f"the ring, {caps.get('shed_to_fit', 0)} shed to fit the {caps.get('max_bundle_bytes')} "
        f"B cap, {caps.get('refused', 0)} refused at the emit boundary"
    )
    return lines
