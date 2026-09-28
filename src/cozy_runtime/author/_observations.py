"""The ONE observation record, its BOUNDED ring, and O(1) time attribution.

Worker observations are bounded everywhere or they are not observations — an unbounded
event list is a memory leak that only shows up on the longest attempt, which is exactly the
attempt whose triage matters. So there is one record shape, one ring, and one accumulator:

**One record.** `Observation` is what `tel.log/metric/stage` produce, what the runtime's own
confessions and liveness verdicts produce, and what the triage bundle carries. There is no
second event type anywhere; a new kind is a line in `KINDS`, not a new class.

**One ring.** `EventRing` is bounded by COUNT and by BYTES, sheds oldest first, and counts
what it shed. A flood cannot evict the cap itself, and the drop count is on the record — a
bundle that silently lost half its tail would be worse than one that says it did. A row that
carries a `position` samples a narrated stream (a prefetch's download): it replaces that
stream's previous sample in a lane of its own, so no rate of narration sheds a record.

**One accumulator, and progress is NOT in the ring.** Progress frames are the lossy live
lane; ringing 4,096 of them would evict every log an operator needs. Their bounded durable
form is `Attribution`: five floats per named stage, updated in O(1) with no allocation, so
per-step timing costs the hot path essentially nothing and still answers "where did the time
go". That is the ROADMAP's step-time instrument.

**The emit boundary REDACTS credential-shaped VALUES.** A signed URL or bearer token in an
observation value is replaced with `<redacted>` and counted. It does not refuse: telemetry
must never fail the thing it observes, and a `log()` that raises turns a successful GPU
attempt into a failed one. Names are not inspected at all — the v1 name regex false-positived
on `output_tokens`, `tokens_per_second` and `eos_token`, which are the most ordinary fields
an inference package emits.

**An over-cap VALUE is TRUNCATED and counted, never raised** (cr-122). The same rule, for
the same reason: a value is data, and data whose length the emitter cannot predict must not
be able to fail the attempt that produced it. It did — a contract attestation rendered a
1,744-character line, the refusal escaped device release, and the worker abandoned the
attempt `EXECUTOR_INVALIDATED`, rebuilt and retried the same 30-step generation forever. So
values and field values clip to `TEXT_CAP` with a marker and bump `truncated`; a lost tail
is a fact on `caps()`, exactly like a shed row. An over-cap NAME or KEY keeps a digest of its
tail, so two records never alias; fields past `FIELDS_CAP` go; a numpy or torch scalar
unwraps; a number JSON cannot carry exactly (NaN, beyond 2^53) and any other object become
text. Each is counted, and none raises.

This module is in the AUTHOR package on purpose — the caps must bite where the author code
emits, and the worker imports the same implementation rather than a second one.
"""

from __future__ import annotations

import hashlib
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
Scalar = str | bool | int | float | None

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
#: Per-step points a step track keeps, first come; a longer schedule counts the rest.
MAX_SERIES = 512
#: Narrated streams the ring keeps a latest sample of; the least recent one sheds first.
MAX_STREAMS = 32

#: Value shapes that carry credentials — signed URLs and the common bearer/key spellings.
_SECRET_VALUE = re.compile(
    r"(?i)(x-amz-(signature|credential|security-token)|[?&](token|sig|signature|key)="
    r"|bearer\s+[\w.\-]{12,}|\b(sk|hf|ghp|gho|xox[baprs])[-_][A-Za-z0-9]{16,})"
)


class ObservationRefusal(CapabilityError):
    """A record the emit boundary will not admit. Typed, and never a silent drop."""


#: The widest integer every JSON reader holds exactly (I-JSON).
_EXACT_INT = (1 << 53) - 1


REDACTED = "<redacted>"
#: What a clipped value ends with, so a bounded record says it was bounded.
TRUNCATED = "…"


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

    @property
    def stream(self) -> str:
        """The narrated stream a row carrying a `position` samples, else "". A later
        sample of its stream supersedes it."""
        if "position" not in self.fields:
            return ""
        return f"{self.name} {self.fields.get('stage') or ''}"

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
    """A bounded, oldest-first-shedding record of one attempt's observations, beside the
    latest sample of each narrated stream."""

    #: the opaque attempt correlation every row is stamped with
    subject: str = ""
    max_events: int = MAX_EVENTS
    max_bytes: int = MAX_RING_BYTES
    events: deque[Observation] = field(default_factory=deque)
    streams: dict[str, Observation] = field(default_factory=dict)
    held: int = 0
    admitted: int = 0
    dropped: int = 0
    coalesced: int = 0
    refused: int = 0
    redactions: int = 0
    truncated: int = 0

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
        return self.admit(observation)

    def _build(
        self, kind: Kind, name: str, value: Scalar, fields: Mapping[str, Scalar]
    ) -> Observation:
        if kind not in KINDS:
            raise ObservationRefusal(
                f"{kind!r} is not one of the closed observation kinds {KINDS}: a new kind is "
                "a schema change, never a new string",
                code="observation_kind",
            )
        items = list(fields.items())
        if len(items) > FIELDS_CAP:
            self.truncated += 1
            items = items[:FIELDS_CAP]
        clean = {self._identifier(key, KEY_CAP): self._scalar(item) for key, item in items}
        return Observation(
            kind,
            self._identifier(name, NAME_CAP),
            self._scalar(value),
            clean,
            int(time.time() * 1000),
            0,
        )

    def _identifier(self, value: object, cap: int) -> str:
        """A NAME or KEY; over its cap it keeps a digest of the whole, so none alias."""
        text = str(value)
        if len(text) <= cap:
            return text
        self.truncated += 1
        return text[: cap - 9] + "~" + hashlib.sha256(text.encode()).hexdigest()[:8]

    def _scalar(self, value: object) -> Scalar:
        """Any value as a scalar JSON carries exactly; everything else becomes bounded text."""
        unwrap = getattr(value, "item", None)  # a numpy or torch scalar
        if callable(unwrap) and not isinstance(value, (str, bytes)):
            try:
                value = unwrap()
            except (TypeError, ValueError, RuntimeError):
                value = str(value)
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int) and abs(value) <= _EXACT_INT:
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        return self._clip(self._redact(value if isinstance(value, str) else str(value)))

    def _clip(self, text: str) -> str:
        """Bound one VALUE and count it. Never raises — see the module docstring."""
        if len(text) <= TEXT_CAP:
            return text
        self.truncated += 1
        return text[: TEXT_CAP - len(TRUNCATED)] + TRUNCATED

    def _redact(self, text: str) -> str:
        """Replace credential-shaped VALUE material and count it. Never raises.

        Observability may not fail the thing it observes: a signed URL in a log line is a
        thing to strip, not a reason to lose a completed GPU attempt.
        """
        redacted, count = _SECRET_VALUE.subn(REDACTED, text)
        self.redactions += count
        return redacted

    def admit(self, observation: Observation) -> Observation:
        """Admit a built record, shedding OLDEST first until both caps hold; a stream sample
        replaces its stream's last one instead. The stamped row is returned: its `seq` is
        what the worker de-duplicates the live lane by."""
        stamped = Observation(
            observation.kind,
            observation.name,
            observation.value,
            observation.fields,
            observation.at_unix_ms or int(time.time() * 1000),
            self.admitted + 1,
        )
        self.admitted += 1
        if stamped.stream:
            self.coalesced += self.streams.pop(stamped.stream, None) is not None
            self.streams[stamped.stream] = stamped
            if len(self.streams) > MAX_STREAMS:
                del self.streams[next(iter(self.streams))]
                self.dropped += 1
            return stamped
        self.events.append(stamped)
        self.held += stamped.held_bytes()
        while self.events and (len(self.events) > self.max_events or self.held > self.max_bytes):
            self.held -= self.events.popleft().held_bytes()
            self.dropped += 1
        return stamped

    def rows(self) -> list[dict[str, Scalar | dict[str, Scalar]]]:
        return [event.document(self.subject) for event in self.events]

    def stream_rows(self) -> list[dict[str, Scalar | dict[str, Scalar]]]:
        return [event.document(self.subject) for event in self.streams.values()]

    def caps(self) -> dict[str, int]:
        """What the ring HELD and what it SHED — a lost tail is a fact, never a silence."""
        return {
            "kept": len(self.events),
            "admitted": self.admitted,
            "dropped": self.dropped,
            "streams": len(self.streams),
            "coalesced": self.coalesced,
            "refused": self.refused,
            "truncated": self.truncated,
            "redactions": self.redactions,
            "held_bytes": self.held,
            "max_events": self.max_events,
            "max_bytes": self.max_bytes,
        }


# --------------------------------------------------------------------------- attribution


@dataclass(slots=True)
class Track:
    """One named span's timing and wall bounds. A step track also keeps a bounded series of
    `(end_unix_ms, ms)` points; nothing else grows with the step count."""

    count: int = 0
    total_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    first_ms: float = 0.0
    last_ms: float = 0.0
    started_unix_ms: int = 0
    ended_unix_ms: int = 0
    series: list[tuple[int, float]] | None = None
    series_dropped: int = 0

    def add(self, ms: float) -> None:
        end = time.time() * 1000
        if self.count == 0:
            self.first_ms = self.min_ms = ms
            self.started_unix_ms = int(end - ms)
        elif ms < self.min_ms:
            self.min_ms = ms
        self.count += 1
        self.total_ms += ms
        self.last_ms = ms
        self.ended_unix_ms = int(end)
        if ms > self.max_ms:
            self.max_ms = ms
        if self.series is not None:
            if len(self.series) < MAX_SERIES:
                self.series.append((int(end), round(ms, 3)))
            else:
                self.series_dropped += 1

    def document(self) -> dict[str, object]:
        document: dict[str, object] = {
            "count": self.count,
            "total_ms": round(self.total_ms, 3),
            "mean_ms": round(self.total_ms / self.count, 3) if self.count else 0.0,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "first_ms": round(self.first_ms, 3),
            "last_ms": round(self.last_ms, 3),
            "started_unix_ms": self.started_unix_ms,
            "ended_unix_ms": self.ended_unix_ms,
        }
        if self.series is not None:
            document["series"] = [[end, ms] for end, ms in self.series]
            document["series_dropped"] = self.series_dropped
        return document


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
        self._track(self.steps, name, series=True).add(ms)

    def _track(self, book: dict[str, Track], name: str, *, series: bool = False) -> Track:
        track = book.get(name)
        if track is None:
            if len(book) >= MAX_TRACKS:
                self.folded += 1
                name = "other"
                track = book.get(name)
                if track is not None:
                    return track
            track = book[name] = Track(series=[] if series else None)
        return track

    def document(self) -> dict[str, object]:
        return {
            "stages": {name: track.document() for name, track in sorted(self.stages.items())},
            "steps": {name: track.document() for name, track in sorted(self.steps.items())},
            "folded_names": self.folded,
            "method": (
                "perf_counter deltas accumulated in place: count/total/min/max/first/last and "
                f"wall bounds per named span; step tracks keep up to {MAX_SERIES} "
                "[end_unix_ms, ms] points"
            ),
        }
