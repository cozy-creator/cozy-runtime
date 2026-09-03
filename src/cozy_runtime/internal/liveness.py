"""Observed progress, and the ONE rule for calling a subject wedged.

Nothing in this runtime is killed because a number of seconds passed. What ends a process
early is process death, a typed result, a cancel, a caller's own deadline — or a PROVEN
wedge, and this module is where "proven" is defined. The transport's stream ledger (now
tensorfs's, cr-090) already judged a stream this way: silent for far longer than the longest
gap the same pull had actually shown, and never for less than the sampling noise floor.
That rule is lifted here so a PROCESS is judged the same way — an executor answering a
seam command, a native-operator fixture, the whole boot of a local run:

    wedged  =  the meter has not changed for longer than STILL_FACTOR times the longest pause
               this same subject has already shown, AND for longer than the floor the caller
               states.

The floor is the caller's to derive, and it says what the term is for: the sampling noise
floor (`noise_floor`) for a subject with no machinery of its own, or the deepest silence
some inner machinery is itself allowed to show before it acts — an observer outside the
pull ledger must not call the worker wedged before that ledger has had its own chance.

The meter is whatever the caller reads: `proctree.progress_burn` for a process (CPU plus
bytes moved), a sum of boot-cursor positions, the whole observation a reclaim is waiting on.
Any CHANGE is movement — a composite meter falls when a child exits, and a child exiting is
not stillness. An UNREADABLE reading (`None`) decides nothing, which is `progress_burn`'s
own rule restated. A slow subject teaches its pace and is left alone; only one that has
stopped, against its own evidence, is ended.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from cozy_runtime.internal import proctree

#: How often a meter is SAMPLED. Resolution, not a verdict: nothing is killed because this
#: many seconds passed, and the smallest pause the rule can see is one of these.
SAMPLE_SECONDS = 5.0

#: The NOISE FLOOR, in samples, below which stillness is not evidence of anything. A meter
#: that ticks in milliseconds must not condemn a subject that is merely one interval slow.
STILL_SAMPLES = 6

#: How many times a subject's OWN longest observed pause it may be still for before it is
#: wedged. This is the term that carries the verdict and it is MEASURED: a boot that has
#: shown a 40 s pause earns 320 s of patience, one that never paused earns the floor.
STILL_FACTOR = 8.0


def noise_floor(sample_seconds: float | None = None) -> float:
    """The floor for a subject with no inner machinery: `STILL_SAMPLES` intervals."""
    return STILL_SAMPLES * (SAMPLE_SECONDS if sample_seconds is None else sample_seconds)


def burn(pid: int, tid: int = 0) -> int | None:
    """`proctree.progress_burn` as a reading: UNREADABLE is `None`, here and on a platform
    with no `/proc` to read."""
    try:
        value = proctree.progress_burn(pid, tid)
    except proctree.ProcessTreeUnsupported:
        return None
    return None if value < 0 else value


@dataclass(slots=True)
class Pace:
    """One subject's observed pace: its last reading, when it last changed, and the longest
    pause it has shown. It reports; the caller decides with `wedged`."""

    reading: object = None
    moved_at: float = 0.0
    worst_pause: float = 0.0
    samples: int = 0
    changes: int = 0

    def observe(self, reading: object, now: float | None = None) -> str:
        """One sample: `advancing`, `still`, or `unreadable`."""
        at = time.monotonic() if now is None else now
        if reading is None:
            return "unreadable"
        self.samples += 1
        if self.reading is None:
            # The first readable sample opens the pause clock; it is not itself a pause.
            self.reading, self.moved_at = reading, at
            return "advancing"
        if reading == self.reading:
            return "still"
        self.worst_pause = max(self.worst_pause, at - self.moved_at)
        self.reading, self.moved_at = reading, at
        self.changes += 1
        return "advancing"

    def still_for(self, now: float | None = None) -> float:
        if self.reading is None:
            return 0.0
        return (time.monotonic() if now is None else now) - self.moved_at

    def patience(self, floor: float) -> float:
        return max(STILL_FACTOR * self.worst_pause, floor)

    def wedged(self, floor: float, now: float | None = None) -> bool:
        return self.reading is not None and self.still_for(now) > self.patience(floor)

    def verdict(self, floor: float, now: float | None = None) -> str:
        """The measured facts a wedge refusal carries, so the reader can check the rule."""
        return (
            f"no measurable progress for {self.still_for(now):.1f} s; its longest earlier "
            f"pause was {self.worst_pause:.1f} s over {self.changes} advance(s), so it was "
            f"given {self.patience(floor):.1f} s"
        )
