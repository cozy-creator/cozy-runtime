"""The ONE observation record, its BOUNDED ring, and O(1) time attribution.

Worker observations are bounded everywhere or they are not observations — an unbounded
event list is a memory leak that only shows up on the longest attempt, which is exactly the
attempt whose triage matters. So there is one record shape, one ring, and one accumulator:

**One record.** `Observation` is what `tel.log/metric/stage` produce, what the runtime's own
confessions and liveness verdicts produce, and what the triage bundle carries. There is no
second event type anywhere; a new kind is a line in `KINDS`, not a new class.

**One ring.** `EventRing` is bounded by COUNT and by BYTES, sheds oldest first, and counts
what it shed. A flood cannot evict the cap itself, and the drop count is on the record — a
bundle that silently lost half its tail would be worse than one that says it did.

**One accumulator, and progress is NOT in the ring.** Progress frames are the lossy live
lane; ringing 4,096 of them would evict every log an operator needs. Their bounded durable
form is `Attribution`: five floats per named stage, updated in O(1) with no allocation, so
per-step timing costs the hot path essentially nothing and still answers "where did the time
go". That is the ROADMAP's step-time instrument.

**The emit boundary is where credentials die.** A field named or valued like a secret is a
typed refusal at `emit`, not a redaction downstream: redaction is a filter someone can
forget to apply, and a refusal is one the author fixes once.

This module is in the AUTHOR package on purpose — the caps must bite where the author code
emits, and the worker imports the same implementation rather than a second one.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

from cozy_runtime.author._errors import CapabilityError

#: The CLOSED observation vocabulary. `progress` is deliberately absent: it is the lossy
#: live lane and never a retained record (its durable form is `Attribution`).
Kind = Literal["log", "stage", "metric", "confession", "boot", "liveness", "fault"]
KINDS: tuple[Kind, ...] = ("log", "stage", "metric", "confession", "boot", "liveness", "fault")

#: A retained value: one scalar, never a structure. A nested payload has no cap.
Scalar = str | int | float | None

#: The caps. Every one is a ceiling the deployment may lower and nothing in code raises.
MAX_EVENTS = 256
MAX_RING_BYTES = 128 * 1024
NAME_CAP = 200
TEXT_CAP = 400
FIELDS_CAP = 8
KEY_CAP = 48
#: Distinct attribution names. A handler that names a fresh stage per step folds into
#: `other` rather than growing a dict for the life of the attempt.
MAX_TRACKS = 32
#: Fixed per-row bookkeeping charged against `MAX_RING_BYTES` beside the row's own text.
ROW_OVERHEAD = 96

#: Names that are credential-shaped WHEREVER they appear as a whole `_.-`-delimited segment.
#: Nothing an inference handler measures is called one of these, so position does not matter.
_SECRET_NAME = re.compile(
    r"(?i)(^|[_.-])(secrets?|passwo?rds?|passwd|api_?keys?|auth|authorization"
    r"|bearer|credentials?|private_?key|session_?key|cookie|access_?key)([_.-]|$)"
)
#: `token` and `signature` are the two credential words that are ALSO ordinary telemetry
#: nouns: `output_tokens`, `tokens_per_second` and `eos_token` are counts, rates and
#: vocabulary — the most obvious fields an inference endpoint emits — and refusing them made
#: the guard a false positive at the one boundary it exists to protect. They are
#: credential-shaped only when the name IS the bare word, or ends in it in the SINGULAR
#: directly after a credential qualifier. A plural, a trailing word, or an unqualified
#: `eos_token` is a description; `access_token` and `api_signature` are still an identity.
_SECRET_NAME_QUALIFIED = re.compile(
    r"(?i)^(token|signature)$"
    r"|(^|[_.-])(access|api|app|bearer|client|csrf|id|oauth|refresh|sas|security"
    r"|session|xsrf)[_.-](token|signature)$"
)
#: Value shapes that carry credentials — signed URLs and the common bearer/key spellings.
_SECRET_VALUE = re.compile(
    r"(?i)(x-amz-(signature|credential|security-token)|[?&](token|sig|signature|key)="
    r"|bearer\s+[\w.\-]{12,}|\b(sk|hf|ghp|gho|xox[baprs])[-_][A-Za-z0-9]{16,})"
)


class ObservationRefusal(CapabilityError):
    """A record the emit boundary will not admit. Typed, and never a silent drop."""


def _text(value: str, cap: int, what: str) -> str:
    if len(value) > cap:
        raise ObservationRefusal(
            f"{what} is {len(value)} characters, over the {cap}-character cap: an "
            "observation is a bounded record, and a record that can grow without limit is "
            "not one",
            code="observation_too_large",
        )
    return value


def _no_secret(where: str, text: str) -> None:
    if _SECRET_NAME.search(where) is not None or _SECRET_NAME_QUALIFIED.search(where) is not None:
        raise ObservationRefusal(
            f"observation field {where!r} is credential-shaped: telemetry is retained, "
            "shipped and rendered, so a secret-named field is refused at the emit boundary "
            "rather than redacted somewhere downstream that can be forgotten",
            code="credential_field",
        )
    if _SECRET_VALUE.search(text) is not None:
        raise ObservationRefusal(
            f"the value of {where!r} carries a signed URL or bearer-shaped credential; "
            "observations never carry authority material",
            code="credential_value",
        )


# --------------------------------------------------------------------------- the record


@dataclass(frozen=True, slots=True)
class Observation:
    """One bounded record. Correlation is stamped by the ring, never by the emitter."""

    kind: Kind
    name: str
    value: Scalar = None
    fields: Mapping[str, Scalar] = field(default_factory=dict)
    at_unix_ms: int = 0
    seq: int = 0

    def held_bytes(self) -> int:
        """This row's memory footprint, by its own text lengths plus fixed overhead."""
        total = ROW_OVERHEAD + len(self.kind) + len(self.name)
        if isinstance(self.value, str):
            total += len(self.value)
        for key, item in self.fields.items():
            total += len(key) + (len(item) if isinstance(item, str) else 8)
        return total

    def document(self, subject: str) -> dict[str, Scalar | dict[str, Scalar]]:
        """The canonical row. EVERY row carries its attempt correlation — a bundle line
        that cannot name its attempt is a line nobody can act on."""
        row: dict[str, Scalar | dict[str, Scalar]] = {
            "seq": self.seq,
            "kind": self.kind,
            "name": self.name,
            "at_unix_ms": self.at_unix_ms,
            "attempt_key": subject,
        }
        if self.value is not None:
            row["value"] = self.value
        if self.fields:
            row["fields"] = dict(self.fields)
        return row


# ----------------------------------------------------------------------------- the ring


@dataclass(slots=True)
class EventRing:
    """A bounded, oldest-first-shedding record of one attempt's observations."""

    #: the opaque attempt correlation every row is stamped with
    subject: str = ""
    max_events: int = MAX_EVENTS
    max_bytes: int = MAX_RING_BYTES
    events: deque[Observation] = field(default_factory=deque)
    held: int = 0
    admitted: int = 0
    dropped: int = 0
    refused: int = 0

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self.events)

    def emit(self, kind: Kind, name: str, value: Scalar = None, **fields: Scalar) -> Observation:
        """Validate, bound, stamp and admit ONE record. Refusals are counted and raised."""
        try:
            observation = self._build(kind, name, value, fields)
        except ObservationRefusal:
            self.refused += 1
            raise
        self.admit(observation)
        return observation

    def _build(
        self, kind: Kind, name: str, value: Scalar, fields: Mapping[str, Scalar]
    ) -> Observation:
        if kind not in KINDS:
            raise ObservationRefusal(
                f"{kind!r} is not one of the closed observation kinds {KINDS}: a new kind is "
                "a schema change, never a new string",
                code="observation_kind",
            )
        _text(name, NAME_CAP, f"observation name {name[:40]!r}")
        _no_secret(name, name if isinstance(value, str) else "")
        if isinstance(value, str):
            _text(value, TEXT_CAP, f"observation value of {name!r}")
            _no_secret(name, value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ObservationRefusal(
                f"observation {name!r} carries {value!r}, which has no canonical spelling",
                code="observation_value",
            )
        if len(fields) > FIELDS_CAP:
            raise ObservationRefusal(
                f"observation {name!r} carries {len(fields)} fields, over the {FIELDS_CAP} cap",
                code="observation_too_large",
            )
        clean: dict[str, Scalar] = {}
        for key, item in fields.items():
            _text(key, KEY_CAP, f"observation field name {key[:40]!r}")
            text = item if isinstance(item, str) else ""
            _no_secret(key, text)
            if text:
                _text(text, TEXT_CAP, f"observation field {key!r}")
            clean[key] = item
        return Observation(kind, name, value, clean, int(time.time() * 1000), 0)

    def admit(self, observation: Observation) -> None:
        """Admit a built record, shedding OLDEST first until both caps hold."""
        stamped = Observation(
            observation.kind,
            observation.name,
            observation.value,
            observation.fields,
            observation.at_unix_ms or int(time.time() * 1000),
            self.admitted + 1,
        )
        self.admitted += 1
        self.events.append(stamped)
        self.held += stamped.held_bytes()
        while self.events and (len(self.events) > self.max_events or self.held > self.max_bytes):
            self.held -= self.events.popleft().held_bytes()
            self.dropped += 1

    def rows(self) -> list[dict[str, Scalar | dict[str, Scalar]]]:
        return [event.document(self.subject) for event in self.events]

    def caps(self) -> dict[str, int]:
        """What the ring HELD and what it SHED — a lost tail is a fact, never a silence."""
        return {
            "kept": len(self.events),
            "admitted": self.admitted,
            "dropped": self.dropped,
            "refused": self.refused,
            "held_bytes": self.held,
            "max_events": self.max_events,
            "max_bytes": self.max_bytes,
        }


# --------------------------------------------------------------------------- attribution


@dataclass(slots=True)
class Track:
    """One named span's timing, as five numbers. Nothing here grows with the step count."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    first_ms: float = 0.0
    last_ms: float = 0.0

    def add(self, ms: float) -> None:
        if self.count == 0:
            self.first_ms = self.min_ms = ms
        elif ms < self.min_ms:
            self.min_ms = ms
        self.count += 1
        self.total_ms += ms
        self.last_ms = ms
        if ms > self.max_ms:
            self.max_ms = ms

    def document(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.total_ms / self.count, 3) if self.count else 0.0,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "first_ms": round(self.first_ms, 3),
            "last_ms": round(self.last_ms, 3),
        }


@dataclass(slots=True)
class Attribution:
    """Where an attempt's time went, in O(1) per span and O(names) in memory.

    `stages` are `tel.stage(name)` brackets — the COMMITTED `stage_ms` fact. `steps` are the
    per-step intervals of `tel.step_callback`, which is what makes inference step time
    attributable without keeping one row per step. Both are observation only: nothing here
    reaches placement, availability or the ComponentUseContract.
    """

    stages: dict[str, Track] = field(default_factory=dict)
    steps: dict[str, Track] = field(default_factory=dict)
    folded: int = 0

    def stage(self, name: str, ms: float) -> None:
        self._track(self.stages, name).add(ms)

    def step(self, name: str, ms: float) -> None:
        self._track(self.steps, name).add(ms)

    def _track(self, book: dict[str, Track], name: str) -> Track:
        track = book.get(name)
        if track is None:
            if len(book) >= MAX_TRACKS:
                self.folded += 1
                name = "other"
                track = book.get(name)
                if track is not None:
                    return track
            track = book[name] = Track()
        return track

    def document(self) -> dict[str, object]:
        return {
            "stages": {name: track.document() for name, track in sorted(self.stages.items())},
            "steps": {name: track.document() for name, track in sorted(self.steps.items())},
            "folded_names": self.folded,
            "method": (
                "perf_counter deltas accumulated in place: count/total/min/max/first/last per "
                "named span, never one row per step"
            ),
        }
