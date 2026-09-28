"""Process-local measured plan facts.

Measurements remain keyed by the resolved plan digest so unrelated bytes, builds, devices,
providers and construction contracts cannot share them. They deliberately die with the
process: Runtime executes and measures, but it is not a durable observation store.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from threading import Lock

from cozy_runtime.internal import canonical


@dataclass(frozen=True, slots=True)
class RunFact:
    """ONE run's observation of ONE resolved plan. Written once, never edited.

    Every number came off a real prepare or a real step and says which. `-1` means NOT
    MEASURED and never zero: a zero fill wall would read as instant and win every latency
    comparison it was never entitled to enter.
    """

    plan_digest: str
    run_id: str
    stored_bytes: int = -1
    resident_bytes: int = -1
    scratch_bytes: int = -1
    fill_ms: int = -1
    decode_ms: int = -1
    step_ms: float = -1.0
    """Measured per-step wall of one forward pass. A prepare banks a fill wall and cannot
    bank this: it does not run a forward pass."""
    reference_deviation_rel_l2: float = -1.0
    """Single-step relative L2 against the REFERENCE plan on identical inputs. The reference
    plan's own is 0.0 by construction."""
    evidence: str = ""
    at_ns: int = 0


#: The numeric fields a fold aggregates, and whether "measured" means `>= 0`.
_FOLDED = (
    "stored_bytes",
    "resident_bytes",
    "scratch_bytes",
    "fill_ms",
    "decode_ms",
    "step_ms",
    "reference_deviation_rel_l2",
)


@dataclass(frozen=True, slots=True)
class PlanFact:
    """The FOLD of every run of one plan: a median per field, and an honest sample count.

    MEDIAN, not mean and not latest, and the reason is the population: these are wall
    measurements taken on shared machines, so the distribution has a long right tail from
    neighbours and a mean tracks the worst neighbour rather than the plan. The median of one
    sample is that sample, which is what makes a first prepare usable without pretending it
    is a distribution — `samples` is what a reader checks, and the resolver prints it.
    """

    plan_digest: str
    stored_bytes: int = -1
    resident_bytes: int = -1
    scratch_bytes: int = -1
    fill_ms: int = -1
    decode_ms: int = -1
    step_ms: float = -1.0
    reference_deviation_rel_l2: float = -1.0
    samples: int = 0
    measured: tuple[str, ...] = ()
    """Which fields at least one run actually measured. A field absent from here is -1 and
    means nobody looked, which is a different statement from "it was zero"."""

    @property
    def timed(self) -> bool:
        """A per-step wall exists. The precondition of a LATENCY score, and the reason a
        latency objective used to have nothing to stand on at the layer that refused."""
        return self.step_ms >= 0.0

    @property
    def conformed(self) -> bool:
        return self.reference_deviation_rel_l2 >= 0.0

    @property
    def single_sample(self) -> bool:
        return self.samples <= 1

    def document(self) -> dict[str, object]:
        return {
            "plan_digest": self.plan_digest,
            "stored_bytes": self.stored_bytes,
            "resident_bytes": self.resident_bytes,
            "scratch_bytes": self.scratch_bytes,
            "fill_ms": self.fill_ms,
            "decode_ms": self.decode_ms,
            "step_ms": self.step_ms,
            "reference_deviation_rel_l2": self.reference_deviation_rel_l2,
            "samples": self.samples,
            "measured": list(self.measured),
        }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def fold(plan_digest: str, runs: Iterable[RunFact]) -> PlanFact:
    """Every run of one plan -> one fact. A field nobody measured stays -1."""
    rows = list(runs)
    median = {
        field: _median(seen)
        for field in _FOLDED
        if (seen := [value for row in rows if (value := float(getattr(row, field))) >= 0.0])
    }

    def whole(field: str) -> int:
        return int(median.get(field, -1))

    return PlanFact(
        plan_digest,
        stored_bytes=whole("stored_bytes"),
        resident_bytes=whole("resident_bytes"),
        scratch_bytes=whole("scratch_bytes"),
        fill_ms=whole("fill_ms"),
        decode_ms=whole("decode_ms"),
        step_ms=median.get("step_ms", -1.0),
        reference_deviation_rel_l2=median.get("reference_deviation_rel_l2", -1.0),
        samples=len(rows),
        measured=tuple(sorted(median)),
    )


class PlanFacts:
    """One process's immutable observations, grouped by exact plan digest."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, RunFact]] = {}
        self._lock = Lock()

    def runs(self, plan_digest: str) -> list[RunFact]:
        with self._lock:
            return sorted(
                self._rows.get(plan_digest, {}).values(),
                key=lambda row: (row.at_ns, row.run_id),
            )

    def get(self, plan_digest: str) -> PlanFact:
        return fold(plan_digest, self.runs(plan_digest))

    def digests(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._rows))

    def bank(self, fact: RunFact) -> None:
        """Retain one run for this process. A duplicate run id keeps its first value."""

        with self._lock:
            self._rows.setdefault(fact.plan_digest, {}).setdefault(fact.run_id, fact)

    def prune(self, plan_digest: str, keep: int) -> int:
        """Keep the newest observations for one plan in this process."""

        with self._lock:
            rows = self._rows.get(plan_digest)
            if not rows:
                return 0
            ordered = sorted(rows.values(), key=lambda row: (row.at_ns, row.run_id), reverse=True)
            retained = {row.run_id: row for row in ordered[: max(keep, 0)]}
            removed = len(rows) - len(retained)
            if retained:
                self._rows[plan_digest] = retained
            else:
                self._rows.pop(plan_digest, None)
            return removed


def run_id(plan_digest: str, device_tuple: str, release: str) -> str:
    """A run's own identity: the plan, the machine, the build, the wall clock and the pid.

    The clock, the pid and eight bytes of entropy are what make it a RUN rather than a plan:
    two prepares of one plan on one machine are two observations, and a key that collapsed
    them would turn the sample count back into a merge. The entropy is there because a
    nanosecond clock is not guaranteed to advance between two calls in one process, and two
    runs colliding would silently drop the second's observation.
    """
    preimage = canonical.write(
        {
            "plan": plan_digest,
            "device": device_tuple,
            "release": release,
            # HEX STRINGS, not integers. `canonical.write` refuses an integer outside the
            # interoperable 2**53 range, while a nanosecond clock is about 1.8e18.
            "at_ns_hex": f"{time.time_ns():x}",
            "pid_hex": f"{os.getpid():x}",
            "entropy": os.urandom(8).hex(),
        }
    )
    return hashlib.sha256(b"cozy.runtime.plan-run\0" + preimage).hexdigest()[:32]
