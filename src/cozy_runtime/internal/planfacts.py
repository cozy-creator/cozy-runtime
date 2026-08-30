"""#549.2 — measured envelopes, keyed by RESOLVED-PLAN DIGEST, in private SQLite rows.

**What this replaces, and why it was a real defect.** `DeliveryTable` keyed a measured
envelope by `(variant, rung)`. Neither half of that key binds the thing the numbers describe:

* not the SNAPSHOT BYTES — a variant is a NAME in a binding record, so re-pointing
  `"fp8"` at a different snapshot reused the old artifact's fill wall, resident bytes and
  deviation under the new one's name;
* not the RUNTIME BUILD — a decode rewritten between deploys inherited the old decode's
  timings;
* not the PROVIDER MAP — the whole point of #517e is that one artifact has two routes with
  opposite standings, and a rung LABEL cannot tell them apart in a mixed plan;
* not the CONSTRUCTION CONTRACT or the CONFIG — a different topology under one variant name
  is a different resident footprint.

So a reused label reused timings from unrelated bytes. The key here is the resolved plan's
own digest, which binds all five by construction (`resolution.py`'s `ResolvedModelPlan`
hashes the snapshot map, the per-tensor implementation digests, the runtime identity, the
device identity, the construction facts and the config digest).

**Immutable per-run rows, and the choice is stated.** The old table was a shared JSON
document that every writer loaded, merged and rewrote. Two concurrent prepares could silently
drop one observation. Each run now inserts one row under a unique `(plan_digest, run_id)` key:

* a run inserts ONE row. There is nothing to merge, so there is nothing to lose.
* reading is a query and a fold. Every run's observation survives, which is also
  what makes `samples` an honest count instead of an incremented integer nobody can audit.
* SQLite supplies the concurrency and crash boundary; there is no hand-rolled file lock.

`prune` remains the explicit operator ceiling.

**Old label-keyed files are INVALID and are not migrated.** They live under a different name
(`delivery-*.json`), nothing here reads them, and the numbers inside them cannot be
attributed to a plan after the fact — that is precisely what they failed to record. A first
prepare on the new key banks a fresh fact, which is the same rule #511b applies to capability
records: measurement-minted, and a re-measure re-mints.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cozy_runtime.internal import cache, canonical


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

    def document(self) -> dict[str, Any]:
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
    if not rows:
        return PlanFact(plan_digest)
    values: dict[str, list[float]] = {}
    for row in rows:
        for field in _FOLDED:
            value = float(getattr(row, field))
            if value >= 0.0:
                values.setdefault(field, []).append(value)
    folded: dict[str, Any] = {}
    for field in _FOLDED:
        seen = values.get(field)
        if not seen:
            continue
        median = _median(seen)
        exact = field in ("step_ms", "reference_deviation_rel_l2")
        folded[field] = median if exact else int(median)
    return PlanFact(
        plan_digest,
        samples=len(rows),
        measured=tuple(sorted(values)),
        **folded,
    )


class PlanFacts:
    """The private cache: one immutable SQLite row per run.

    It is a DEVICE-and-BUILD property, like the probe cache, and it lives beside that one
    under `cozy_home` rather than beside a checkpoint. Unlike the probe cache it retains
    every distinct run, because two runs of one plan are two observations and both are true.
    """

    def __init__(self, root: Path | str | None) -> None:
        self.root = Path(root) if root is not None else None

    def runs(self, plan_digest: str) -> list[RunFact]:
        if self.root is None:
            return []
        with cache.connect(self.root) as database:
            rows = database.execute(
                """
                SELECT plan_digest, run_id, stored_bytes, resident_bytes, scratch_bytes,
                       fill_ms, decode_ms, step_ms, reference_deviation_rel_l2, evidence, at_ns
                FROM plan_facts WHERE plan_digest = ? ORDER BY at_ns, run_id
                """,
                (plan_digest,),
            )
            return [RunFact(**dict(row)) for row in rows]

    def get(self, plan_digest: str) -> PlanFact:
        return fold(plan_digest, self.runs(plan_digest))

    def digests(self) -> tuple[str, ...]:
        """Every plan with at least one banked run, for operator/reporting views."""

        if self.root is None:
            return ()
        with cache.connect(self.root) as database:
            rows = database.execute("SELECT DISTINCT plan_digest FROM plan_facts ORDER BY 1")
            return tuple(str(row["plan_digest"]) for row in rows)

    def bank(self, fact: RunFact) -> Path | None:
        """Insert ONE run's observation. No merge and no rewrite.

        A run id collision means the same run wrote twice, which is a caller bug rather than
        a race, so the existing row is left alone.
        """
        if self.root is None:
            return None
        with cache.connect(self.root) as database:
            database.execute(
                """
                INSERT OR IGNORE INTO plan_facts (
                    plan_digest, run_id, stored_bytes, resident_bytes, scratch_bytes, fill_ms,
                    decode_ms, step_ms, reference_deviation_rel_l2, evidence, at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.plan_digest,
                    fact.run_id,
                    fact.stored_bytes,
                    fact.resident_bytes,
                    fact.scratch_bytes,
                    fact.fill_ms,
                    fact.decode_ms,
                    fact.step_ms,
                    fact.reference_deviation_rel_l2,
                    fact.evidence,
                    fact.at_ns,
                ),
            )
            database.commit()
        return self.root / cache.FILENAME

    def prune(self, plan_digest: str, keep: int) -> int:
        """Drop all but the `keep` newest runs of one plan. An operator's ceiling, never
        automatic: a store that silently forgot its own samples would make `samples` a lie."""
        if self.root is None:
            return 0
        with cache.connect(self.root) as database:
            removed = database.execute(
                """
                DELETE FROM plan_facts
                WHERE plan_digest = ? AND run_id IN (
                    SELECT run_id FROM plan_facts
                    WHERE plan_digest = ?
                    ORDER BY at_ns DESC, run_id DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (plan_digest, plan_digest, max(keep, 0)),
            ).rowcount
            database.commit()
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
