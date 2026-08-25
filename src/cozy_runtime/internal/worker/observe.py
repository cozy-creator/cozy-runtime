"""The supervisor's observation surface: boot steps, confessions, and SILENCE-BOUNDED liveness.

Three things live here because they are one idea — what the worker knows about itself, kept
bounded and kept honest.

**Boot is attributed per step, and every step is typed.** `download | load | warm | derive`
are the closed vocabulary; a step emits `started` at POSITION ZERO so a wedge before the
first unit still renders, and it ends with an outcome. A compile/derive slice is named as
its own step even though nothing measures it yet: leaving it unnamed is how the cache door's
cost became invisible in v1.

**Liveness is silence-bounded and there is NO CLOCK in the detection.** Progress means an
advancing MONOTONE position — integral units moved, or an advancing sequence. A wedge is
SILENCE: consecutive observations that find the position where the previous one left it.
`give_up_after` counts observations, not seconds, so a slow-but-advancing transfer is never
mistaken for a wedged one however slow it is, and no boot timeout exists here to be tuned.
The cadence that drives sampling is a scheduling choice; the VERDICT is a comparison of two
positions and nothing else.

**A confession is typed and quantified.** Whenever the worker serves something other than
the plain reading of the request — a degraded plan, a substitution, an engaged kernel arm, a
skipped verification — it says so in a `confession` observation carrying the numbers.
Benchmark-override invokes are marked so they can never pollute a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cozy_runtime.author._observations import EventRing, Observation, Scalar

#: The CLOSED boot-step vocabulary (§3.3). `derive` is named even though no torchcg issue
#: measures it — an unnamed step is a cost nobody can see.
BOOT_STEPS = ("download", "load", "warm", "derive")

#: Consecutive non-advancing observations before a subject is called WEDGED. A count, never
#: a duration: that is what makes the detection clock-free.
GIVE_UP_AFTER = 8


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
    wedged: bool = False
    ended: str = ""

    def document(self) -> dict[str, Scalar]:
        return {
            "subject": self.subject,
            "step": self.step,
            "position": self.position,
            "samples": self.samples,
            "advances": self.advanced,
            "consecutive_still": self.still,
            "verdict": "wedged" if self.wedged else ("ended:" + self.ended if self.ended else ""),
        }


class SilenceMonitor:
    """Advance versus silence over monotone positions. It never reads a clock."""

    def __init__(self, ring: EventRing, give_up_after: int = GIVE_UP_AFTER) -> None:
        self.ring = ring
        self.give_up_after = give_up_after
        self.cursors: dict[str, Cursor] = {}

    def start(self, subject: str, step: str) -> Cursor:
        """Open a subject and emit `started` AT POSITION 0 — a pre-first-unit wedge must
        render, and it cannot if nothing was ever said about the subject."""
        if step not in BOOT_STEPS and step != "invoke":
            raise ValueError(f"{step!r} is not a typed step {(*BOOT_STEPS, 'invoke')}")
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
        """One VERDICT: `advancing`, `silent`, or `wedged`. Two integers, no clock."""
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
        if cursor.still < self.give_up_after:
            return "silent"
        if not cursor.wedged:
            cursor.wedged = True
            self.ring.emit(
                "liveness",
                "wedged",
                cursor.position,
                subject=subject,
                step=cursor.step,
                still=cursor.still,
                detail=(
                    f"{cursor.still} consecutive observations found position "
                    f"{cursor.position} unchanged: this subject is wedged by SILENCE, not by "
                    "an expired clock - there is no boot timeout and there will not be one"
                ),
            )
        return "wedged"

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


# ------------------------------------------------------------------------- the boot record


@dataclass(slots=True)
class BootRecord:
    """The worker's own boot narrative, bounded, and readable after the process dies."""

    ring: EventRing = field(default_factory=lambda: EventRing(subject="worker"))
    steps: list[dict[str, Any]] = field(default_factory=list)

    def step(self, name: str, outcome: str, ms: float, **fields: Scalar) -> None:
        if name not in BOOT_STEPS:
            raise ValueError(f"{name!r} is not one of the typed boot steps {BOOT_STEPS}")
        row: dict[str, Any] = {"step": name, "outcome": outcome, "ms": round(ms, 2), **fields}
        self.steps.append(row)
        del self.steps[:-32]
        self.ring.emit("boot", name, outcome, ms=round(ms, 2), **fields)

    def document(self) -> dict[str, Any]:
        return {"steps": list(self.steps), "events": self.ring.rows(), "caps": self.ring.caps()}
