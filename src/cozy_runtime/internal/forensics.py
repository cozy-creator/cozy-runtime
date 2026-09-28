"""Fill-plane MEMORY FORENSICS — the boundaries a device-byte gap can be composed at.

The fill plane keeps a ledger of what it BELIEVES it holds and the allocator holds what it
ACTUALLY holds, and until those two numbers are read at the same instants nobody can say
which one is wrong. Measured on the H3 N-ary fill: ledger 91,731,860,032 B live against an
allocator peak of 149,267,319,808 B, and a rollback that returned part. Neither number was
observable at a boundary, so the 53.6 GiB had no composition — only a size.

This is the permanent instrument, not a script. It is OFF unless a DIRECTORY names where
its captures go: `COZY_FILL_FORENSICS_DIR` is a PATH, which is configuration, and the code
branches on having somewhere to write rather than on a boolean that gates behaviour.

What it captures, at every boundary the fill crosses:

  * the allocator's own three numbers (live, reserved, high-water) beside the driver's
    free/total, which is the only pair that sees other processes;
  * the fill plane's ledger rows for all four classes — `vram`, `pinned`, `staging`,
    `decode` — because the refusal path prints ONE of them and a gap composed of the
    other three reads as unexplained;
  * the per-component charge the plane made and the per-component bytes its module tree
    actually holds, summed over DISTINCT storages — a component whose census counts a
    shared storage twice prices itself wrong in the same direction every time.

And, where the allocator supports it, a full `torch.cuda.memory._dump_snapshot()` with
Python allocating stacks at the expensive boundaries (fill end, rollback end, refusal).
That is what turns "53.6 GiB is unaccounted" into a table of call sites: every live block
carries the frame that allocated it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from cozy_runtime.internal import accel

#: How many allocator history entries to retain when stacks are recorded. Bounded because
#: the history is held in device-process memory and an unbounded one competes with the very
#: fill it is measuring.
HISTORY_ENTRIES = 150_000


def distinct_storage_bytes(tensors: Any) -> int:
    """Device bytes a set of tensors actually occupies — DISTINCT storages, counted once.

    `numel() * element_size()` summed over `named_parameters` + `named_buffers` is the
    number the fill plane charges with, and it double-counts every tied weight and every
    view. This is the same walk with the storage identity as the unit, so the two can be
    subtracted and the difference is the census's own error rather than a mystery.
    """
    seen: set[int] = set()
    total = 0
    for tensor in tensors:
        try:
            storage = tensor.untyped_storage()
        except Exception:
            continue
        token = storage.data_ptr()
        if token == 0 or token in seen:
            continue
        seen.add(token)
        total += int(storage.nbytes())
    return total


class FillForensics:
    """One fill's capture. Every `mark` is a row; every `dump` is an allocator snapshot."""

    def __init__(self, torch: Any, kind: str, device: Any, root: Path, tag: str) -> None:
        self.torch = torch
        self.kind = kind
        self.device = device
        self.root = root
        self.tag = tag
        self.rows = root / f"{tag}.jsonl"
        self.started = time.perf_counter()
        self.dumps: list[str] = []
        self.stacks = False
        root.mkdir(parents=True, exist_ok=True)
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python", max_entries=HISTORY_ENTRIES
            )
            self.stacks = True
        except Exception:
            # An allocator without a history recorder still gives every counter below. The
            # capture degrades to numbers-without-call-sites and SAYS so, rather than
            # refusing to measure anything.
            self.stacks = False

    @classmethod
    def open(cls, torch: Any, kind: str, device: Any, tag: str, where: str) -> FillForensics | None:
        """A capture when a directory was configured and the backend can be metered.

        `where` is passed down from the executor's sealed environment snapshot (cr-038):
        this module never reads the process environment itself.
        """
        if not where or kind != "cuda":
            return None
        try:
            return cls(torch, kind, device, Path(where), tag)
        except Exception:
            # Instrumentation NEVER fails a fill. A capture that cannot be opened is a
            # capture that does not happen; the serving path is unchanged either way.
            return None

    # ------------------------------------------------------------------ the boundaries

    def mark(self, boundary: str, **fields: object) -> None:
        """One boundary: the allocator, the driver, the ledger, and the caller's facts."""
        try:
            row: dict[str, object] = {
                "boundary": boundary,
                "t_ms": int((time.perf_counter() - self.started) * 1000),
            }
            row.update(accel.allocation(self.torch, self.kind, self.device))
            row["peak_allocated_bytes"] = accel.peak_allocated(self.torch, self.kind)
            row.update(fields)
            with self.rows.open("a") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        except Exception:
            return

    def dump(self, boundary: str) -> None:
        """The expensive one: every live block with the Python frame that allocated it."""
        if not self.stacks:
            self.mark(f"{boundary}:snapshot_unavailable")
            return
        path = self.root / f"{self.tag}.{boundary}.pickle"
        try:
            self.torch.cuda.memory._dump_snapshot(str(path))
            self.dumps.append(path.name)
            self.mark(f"{boundary}:snapshot", snapshot=path.name)
        except Exception as exc:  # pragma: no cover - allocator/environment
            self.mark(f"{boundary}:snapshot_failed", error=str(exc)[:200])

    def close(self) -> None:
        try:
            self.torch.cuda.memory._record_memory_history(enabled=None)
        except Exception:
            return
