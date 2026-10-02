"""The worker's observation surface: boot steps, confessions, and measured progress.

Three things live here because they are one idea — what the worker knows about itself, kept
bounded and kept honest.

**Boot is attributed per step, and every step is typed.** `download | load | derive` are
the closed vocabulary; a step emits
`started` at POSITION ZERO so a wedge before the first unit still renders, and it ends
with an outcome. A compile/derive slice is named as
its own step even though nothing measures it yet: leaving it unnamed is how the cache door's
cost became invisible in v1.

**Progress is observed and never promoted into a timer-driven failure.** An advancing
MONOTONE position or measured process work is progress. An unchanged sample is reported as
silent; no arbitrary count of samples fabricates a WEDGED verdict. Process exit, a typed
operation result, explicit cancellation, or an enclosing caller deadline decides lifecycle.

**A confession is typed and quantified.** Whenever the worker serves something other than
the plain reading of the request — a degraded plan, a substitution, an engaged kernel arm, a
skipped verification — it says so in a `confession` observation carrying the numbers.
Benchmark-override invokes are marked so they can never pollute a measurement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from cozy_runtime.author._observations import EventRing, Observation, Scalar

#: The CLOSED boot-step vocabulary (§3.3). `derive` is named even though no torchcg issue
#: measures it — an unnamed step is a cost nobody can see.
BOOT_STEPS = ("download", "load", "derive")
#: The two phases of one attempt (cr-079): the device phase, and the post phase that
#: encodes and writes its outputs after device release.
ATTEMPT_STEPS = ("invoke", "post")


@dataclass(slots=True)
class Cursor:
    """One monitored subject's MONOTONE position, and how often it failed to move."""

    subject: str
    step: str
    position: int = 0
    #: the position the previous SAMPLE saw, which is the whole of the comparison
    sampled: int = 0
    samples: int = 0
    still: int = 0
    advanced: int = 0
    ended: str = ""

    def document(self) -> dict[str, Scalar]:
        return {
            "subject": self.subject,
            "step": self.step,
            "position": self.position,
            "samples": self.samples,
            "advances": self.advanced,
            "consecutive_still": self.still,
            "verdict": "ended:" + self.ended if self.ended else "active",
        }


class SilenceMonitor:
    """Report advance versus silence over monotone positions. It never decides failure."""

    def __init__(
        self,
        ring: EventRing,
        busy: Callable[[], int] | None = None,
    ) -> None:
        self.ring = ring
        self.cursors: dict[str, Cursor] = {}
        #: CPU nanoseconds the device process has burned, read by the WORKER. A subject
        #: that emits nothing while its process burns CPU is BUSY, not silent: a 20-minute
        #: hash-bound conversion leg has nothing to say and is still working.
        #: Absent (None) means unreadable, and an unreadable meter decides nothing.
        self.busy = busy
        self.cpu: dict[str, int] = {}

    def start(self, subject: str, step: str) -> Cursor:
        """Open a subject and emit `started` AT POSITION 0 — a pre-first-unit wedge must
        render, and it cannot if nothing was ever said about the subject."""
        if step not in BOOT_STEPS and step not in ATTEMPT_STEPS:
            raise ValueError(f"{step!r} is not a typed step {(*BOOT_STEPS, *ATTEMPT_STEPS)}")
        cursor = self.cursors[subject] = Cursor(subject, step)
        self.ring.emit("boot", "started", 0, subject=subject, step=step)
        return cursor

    def advance(self, subject: str, position: int) -> None:
        """Record forward motion. Positions are monotone; a lower one is ignored, never
        treated as regress — an out-of-order lossy frame is not a rewind."""
        cursor = self.cursors.get(subject)
        if cursor is not None and position > cursor.position:
            cursor.position = position

    def sample(self, subject: str) -> str:
        """One observation: `advancing`, `busy`, `silent`, or `closed`."""
        cursor = self.cursors.get(subject)
        if cursor is None or cursor.ended:
            return "closed"
        cursor.samples += 1
        if cursor.position > cursor.sampled:
            cursor.sampled = cursor.position
            cursor.still = 0
            cursor.advanced += 1
            return "advancing"
        cursor.still += 1
        burned = self.busy() if self.busy is not None else -1
        if burned >= 0:
            before = self.cpu.get(subject, -1)
            self.cpu[subject] = burned
            if before >= 0 and burned > before:
                # WORKING, not silent. A process burning CPU is making measurable progress.
                cursor.still = 0
                return "busy"
        return "silent"

    def end(self, subject: str, outcome: str) -> Cursor | None:
        cursor = self.cursors.get(subject)
        if cursor is None:
            return None
        cursor.ended = outcome
        self.ring.emit(
            "boot", "ended", cursor.position, subject=subject, step=cursor.step, outcome=outcome
        )
        return cursor

    def document(self) -> list[dict[str, Scalar]]:
        return [cursor.document() for cursor in self.cursors.values()]


# --------------------------------------------------------------------------- confessions


@dataclass(frozen=True, slots=True)
class Confession:
    """One typed statement that the worker served other than the plain reading (§3.6)."""

    kind: str
    """`degraded` | `substituted` | `kernel_arm` | `verification_skipped` | `benchmark`."""
    subject: str
    quantified: str
    """The WHY with numbers in it — `needed N, had M, short by N-M for component X`."""
    benchmark_override: bool = False

    def emit_into(self, ring: EventRing) -> Observation:
        return ring.emit(
            "confession",
            self.kind,
            self.quantified,
            subject=self.subject,
            benchmark_override=self.benchmark_override,
        )


def degradation(subject: str, needed: int, had: int, component: str) -> Confession:
    """The FnDegraded-style confession: needed N, had M, short by N-M, for component X."""
    return Confession(
        "degraded",
        subject,
        f"needed {needed} B, had {had} B, short by {needed - had} B for component {component!r}",
    )


# --------------------------------------------------------------------------- demand


@dataclass(slots=True)
class DemandLedger:
    """Declared-versus-actual, banked per serve. THE FALSIFIER, and it decides nothing.

    §3.2's demand discipline puts the instrument FIRST and gives it no authority, because
    the v1 failure was an estimator that could refuse. This one lives in the OBSERVATION
    plane on purpose: the `timing-decides-nothing` fence already proves this module cannot
    import the planner, the ledger or the fill plane, and cannot name a residency
    operation — so "the instrument cannot enforce" is a property of the import graph rather
    than a rule someone has to remember.

    A `demand_miss` is counted per LANE — `(entrypoint, placement rung)`, which is the
    launch regime's cell — and never per request, so one pathological request cannot look
    like a broken model. Warm-pass observations are marked and never banked (se#715).
    """

    #: lane -> (serves, misses, worst declared, worst actual)
    lanes: dict[str, list[int]] = field(default_factory=dict)

    def bank(
        self, lane: str, declared: int, actual: int, *, benchmark: bool = False
    ) -> Confession | None:
        """Bank one serve's pair. Returns a CONFESSION when the declaration was short.

        A benchmark-override serve banks NOTHING: a synthetic payload at a cheapened
        schedule is not evidence about demand, and letting one in is how a measurement
        starts describing the harness.
        """
        if benchmark or actual < 0 or declared <= 0:
            return None
        row = self.lanes.setdefault(lane, [0, 0, 0, 0])
        row[0] += 1
        if actual <= declared:
            return None
        row[1] += 1
        if actual - declared > row[3] - row[2]:
            row[2], row[3] = declared, actual
        return Confession(
            "demand_miss",
            lane,
            f"declared {declared} B of headroom and the serve used {actual} B, short by "
            f"{actual - declared} B; {row[1]} miss(es) in {row[0]} serve(s) on this lane",
        )

    def document(self) -> dict[str, object]:
        return {
            "lanes": {
                lane: {
                    "serves": row[0],
                    "misses": row[1],
                    "worst_declared_bytes": row[2],
                    "worst_actual_bytes": row[3],
                }
                for lane, row in sorted(self.lanes.items())
            },
            "authority": "none - this instrument counts and confesses; it selects nothing",
        }


# ------------------------------------------------------------------------- the boot record


@dataclass(slots=True)
class BootRecord:
    """The worker's own boot narrative, bounded, and readable after the process dies."""

    ring: EventRing = field(default_factory=lambda: EventRing(subject="worker"))
    steps: list[dict[str, Scalar]] = field(default_factory=list)

    def step(self, name: str, outcome: str, ms: float, **fields: Scalar) -> None:
        if name not in BOOT_STEPS:
            raise ValueError(f"{name!r} is not one of the typed boot steps {BOOT_STEPS}")
        row: dict[str, Scalar] = {"step": name, "outcome": outcome, "ms": round(ms, 2), **fields}
        self.steps.append(row)
        del self.steps[:-32]
        self.ring.emit("boot", name, outcome, ms=round(ms, 2), **fields)
