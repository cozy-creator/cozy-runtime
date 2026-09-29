"""Byte progress for one named stage of native work, on the ordinary progress lane.

A stage emits throttled `progress` samples (bytes done/total and the rate since the
previous sample) and one `log` record when it ends, so its elapsed time and average rate
survive the lossy progress lane as durable per-stage timings.
"""

from __future__ import annotations

import time
from collections.abc import Callable

# Display resolution of the lossy lane: the observer polls about once a second.
SAMPLE_SECONDS = 1.0

# A frame of the worker's progress lane.
Emit = Callable[[dict[str, object]], None]
# Where native child work reports: (parent request, parent attempt, call index, frame).
Progress = Callable[[str, int, int, dict[str, object]], None]


class StageProgress:
    def __init__(self, stage: str, emit: Emit, *, clock: Callable[[], float] = time.monotonic):
        self.stage, self.emit, self.clock = stage[:120], emit, clock
        self.started = clock()
        self.started_ms = int(time.time() * 1000)
        self.baseline: int | None = None
        self.position = 0
        self.total = 0
        self.sampled_at = self.started
        self.sampled = 0
        self.complete = self.done = False
        self.ended: float | None = None
        emit({"kind": "progress", "stage": self.stage})

    def update(self, position: int, total: int) -> None:
        position = max(0, min(position, total))
        if self.done or total <= 0 or (self.complete and position == total):
            return
        if self.baseline is None:
            self.baseline = self.sampled = position
        self.position, self.total = position, total
        now = self.clock()
        if now - self.sampled_at < SAMPLE_SECONDS and position < total:
            return
        frame: dict[str, object] = {
            "kind": "progress",
            "stage": self.stage,
            "position": position,
            "total": total,
            "unit": "bytes",
        }
        if now > self.sampled_at and position >= self.sampled:
            frame["rate"] = round((position - self.sampled) / (now - self.sampled_at), 1)
        self.sampled_at, self.sampled = now, position
        self.complete = position == total
        self.emit(frame)

    def elapsed(self) -> float:
        """Seconds the stage has run, or ran once it ended."""
        return (self.clock() if self.ended is None else self.ended) - self.started

    def finish(self, completed: bool) -> None:
        if self.done:
            return
        if completed and self.total:
            self.update(self.total, self.total)
        self.done = True
        self.ended = self.clock()
        elapsed = self.elapsed()
        moved = self.position - (self.baseline or 0)
        fields: dict[str, object] = {
            "phase": self.stage,
            "completed": completed,
            "started_unix_ms": self.started_ms,
            "elapsed_ms": round(elapsed * 1000, 3),
        }
        if self.total:
            fields.update(bytes=self.position, total_bytes=self.total, moved_bytes=moved)
            if elapsed > 0:
                fields["rate_bytes_per_second"] = round(moved / elapsed, 1)
        self.emit(
            {
                "kind": "log",
                "name": self.stage,
                "value": "info" if completed else "warning",
                "at_unix_ms": int(time.time() * 1000),
                "fields": fields,
            }
        )
