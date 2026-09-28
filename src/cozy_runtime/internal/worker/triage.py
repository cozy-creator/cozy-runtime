"""One bounded triage record per terminal attempt.

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

**Written before it is claimed.** The bytes are written and fsynced before their digest and
length enter the terminal. A terminal naming bytes that are not on disk would be exactly the
fabrication class the runtime exists to prevent, so the order is not negotiable.

**Observation only.** Nothing here can replace or override a terminal field
(worker-protocol/01, `TriageBundleRef`). A log cannot declare success.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import msgspec

from cozy_runtime.internal import canonical
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.worker.records import WorkerRecords

#: `TriageBundleRef.length` is bounded at 1 MiB by the protocol. The assembler trims to it.
MAX_BUNDLE_BYTES = 1 << 20


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
    record_seq: int


class Assembly(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """Everything one terminal attempt contributes. Every field has exactly one producer.

    The fields are the bundle's CLOSED section set: a reader decodes exactly these, and a
    writer that wants a new section adds a field, which is a schema change and looks like
    one. Each section is its producer's own observation document.
    """

    subject_id: str
    attempt: dict[str, Json]
    plan: dict[str, Json]
    posture: dict[str, Json]
    terminal: dict[str, Json]
    faults: list[dict[str, Json]]
    measurements: dict[str, Json]
    confessions: list[dict[str, Json]]
    liveness: list[dict[str, Json]]
    events: list[dict[str, Json]]
    caps: dict[str, Json]
    #: the latest sample of each narrated stream (`EventRing.streams`), never a record
    streams: list[dict[str, Json]] = []

    def build(self, limit: int = MAX_BUNDLE_BYTES) -> bytes:
        """Assemble and TRIM to the wire cap, shedding stream samples and then the OLDEST
        observations first.

        The trim measures the real canonical bytes rather than estimating them: the cap is a
        protocol fact and an estimate that is wrong by one byte produces a terminal nobody
        can accept. What is shed is recorded in `caps`, because a bundle that quietly lost
        its tail is worse than one that says how much it lost.
        """
        streams, events = list(self.streams), list(self.events)
        shed = 0
        while True:
            caps = {
                **self.caps,
                "shed_to_fit": shed,
                "max_bundle_bytes": MAX_BUNDLE_BYTES,
                "truncated": bool(shed or self.caps.get("dropped", 0)),
            }
            document = msgspec.structs.replace(self, streams=streams, events=events, caps=caps)
            try:
                data = canonical.write(msgspec.to_builtins(document))
            except canonical.CanonicalError as exc:
                raise TriageError("bundle_uncanonical", str(exc)) from exc
            if len(data) <= limit:
                return data
            if streams:
                shed += len(streams)
                streams = []
                continue
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


class _Written(msgspec.Struct, frozen=True):
    """A bundle's write receipt as the worker records it."""

    subject_id: str
    bundle_digest: str
    length: int


class BundleStore:
    """Bundle bytes on disk plus a same-process write receipt.

    `owner` is who may READ what this store writes, when that is not the writer: a pod's
    root worker commits under the media plane's subtree and hands each bundle to the
    plane's uid, so the owner's fetch (`GET /v1/triage/{subject}`) opens a 0600 file the
    plane owns. The handoff happens on the staged file, before the rename that makes the
    bundle visible, so a visible bundle is always a readable one.
    """

    def __init__(
        self, root: Path, records: WorkerRecords, *, owner: tuple[int, int] | None = None
    ) -> None:
        self.root = root
        self.records = records
        self.owner = owner
        self.written = 0
        self.bytes = 0

    def commit(self, assembly: Assembly) -> Receipt:
        """Write, fsync, hand over, then record the receipt."""
        self.root.mkdir(parents=True, exist_ok=True)
        data = assembly.build()
        path = self.root / f"{assembly.subject_id}.json"
        tmp = path.with_suffix(".part")
        handle = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            if os.write(handle, data) != len(data):
                raise TriageError("bundle_short_write", f"{path.name} did not write whole")
            os.fsync(handle)
            if self.owner is not None:
                os.fchown(handle, *self.owner)
        finally:
            os.close(handle)
        os.replace(tmp, path)
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        # A failed append raises and the caller ships no reference: the bytes stay on disk
        # unreferenced, which is the SAFE half of the failure. A terminal must never name a
        # bundle whose durability was not recorded.
        recorded = self.records.append(
            "triage",
            {
                "subject_id": assembly.subject_id,
                "bundle_digest": digest,
                "length": len(data),
            },
        )
        self.written += 1
        self.bytes += len(data)
        return Receipt(assembly.subject_id, digest, len(data), recorded.seq)

    # ------------------------------------------------------------------ reading

    @staticmethod
    def read(root: Path, subject_id: str, receipt_source: WorkerRecords | None = None) -> Assembly:
        """Read one bundle by opaque subject, optionally checked against its live receipt.

        This is the whole triage path: a directory, a key, and a digest. No process, no SSH,
        no socket. The terminal digest is the offline verifier; same-process readers may
        additionally supply the worker records.
        """
        path = root / f"{subject_id}.json"
        if not path.is_file():
            raise TriageError("bundle_absent", f"no triage bundle for subject {subject_id!r}")
        data = path.read_bytes()
        try:
            bundle = msgspec.convert(canonical.parse_canonical(data), Assembly)
        except msgspec.ValidationError as exc:
            raise TriageError("bundle_malformed", f"{subject_id}: {exc}") from exc
        if receipt_source is not None:
            written = [
                msgspec.convert(row.body, _Written) for row in receipt_source.of_kind("triage")
            ]
            receipt = next((row for row in written if row.subject_id == subject_id), None)
            if receipt is None:
                raise TriageError(
                    "receipt_absent",
                    f"{subject_id}: bytes exist but no write receipt was recorded",
                )
            digest = "sha256:" + hashlib.sha256(data).hexdigest()
            if digest != receipt.bundle_digest or len(data) != receipt.length:
                raise TriageError(
                    "bundle_corrupt",
                    f"{subject_id}: {len(data)} B hashing to {digest[:23]} does not match "
                    f"the recorded receipt ({receipt.length} B, {receipt.bundle_digest[:23]})",
                )
        return bundle


SUBJECT = re.compile(r"trb-[0-9a-f]{24}")


def retained_bytes(root: Path, subject_id: str, digest: str, length: int) -> bytes:
    """The exact bytes an outcome's `TriageBundleRef` names, checked against it."""
    if SUBJECT.fullmatch(subject_id) is None:
        raise TriageError("bundle_malformed", f"{subject_id!r} is not a triage subject")
    path = root / f"{subject_id}.json"
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_BUNDLE_BYTES + 1)
    except FileNotFoundError as exc:
        raise TriageError("bundle_absent", f"triage bundle {subject_id} is not retained") from exc
    if len(data) != length or "sha256:" + hashlib.sha256(data).hexdigest() != digest:
        raise TriageError(
            "bundle_corrupt", f"{subject_id}: bytes differ from the outcome's reference"
        )
    return data
