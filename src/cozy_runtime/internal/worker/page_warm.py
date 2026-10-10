"""Cancellable store-read mechanism; the machine memory policy admits the host footprint."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Required, TypedDict

from cozy_runtime.internal import fill


class WarmReport(TypedDict, total=False):
    component: Required[str]
    skipped: str
    bytes: int
    ms: float
    read_bytes: int


class _Superseded(Exception):
    """A changed queue or demand preparation ended this speculative read."""


def warm_component(
    store_root: str,
    manifest: str,
    component: str,
    *,
    reserve: Callable[[int], bool],
    release: Callable[[], None],
    wanted: Callable[[], bool],
) -> WarmReport:
    """Stream one eligible component with bounded slots; every batch checks current demand."""
    checkpoint = fill.Checkpoint(store_root, manifest)
    table = checkpoint.header["components"].get(component) or {}
    traversal = [(component, str(key)) for key in table]
    if not traversal:
        return {"component": component, "skipped": "the checkpoint holds no rows for it"}
    plan = checkpoint.read_plan(traversal, fill.WINDOW_BYTES, (component,))
    need = int(plan.bytes)
    if not need:
        return {"component": component, "bytes": 0, "ms": 0, "read_bytes": 0}
    count = min(len(plan), fill.RING_SLOTS)
    slots_bytes = fill.slot_bytes(fill.WINDOW_BYTES) * count
    if not wanted() or not reserve(need + slots_bytes):
        return {
            "component": component,
            "skipped": f"queue or host admission declined {need + slots_bytes} B of prefetch",
        }

    def delivered(batch: fill.Batch) -> None:
        batch.release()  # No device consumer: this slot may return immediately.
        if not wanted():
            raise _Superseded

    try:
        slots = [bytearray(fill.slot_bytes(fill.WINDOW_BYTES)) for _ in range(count)]
        before = fill.block_read_bytes()
        started = time.perf_counter()
        with checkpoint.acquire() as lease:
            streamed = lease.stream(plan, slots, delivered, readers=fill.READERS)
    except _Superseded:
        return {"component": component, "skipped": "queue changed or demand preparation started"}
    finally:
        release()
    took = (time.perf_counter() - started) * 1000
    after = fill.block_read_bytes()
    return {
        "component": component,
        "bytes": int(streamed["bytes"]),
        "ms": round(took, 1),
        "read_bytes": max(after - before, 0) if min(before, after) >= 0 else -1,
    }


def describe(report: WarmReport) -> str:
    """One line for the worker's residency notes, in the words the restore path uses."""
    moved, took, read = report["bytes"], report["ms"], report["read_bytes"]
    rate = f"{moved / took / 1e6:.1f} GB/s" if took > 0 else "instant"
    if read < 0:
        served = "block-device reads unreadable"
    else:
        cache = "cold" if read > moved // 2 else "warm page cache"
        served = f"{read} B read from the block layer ({cache})"
    return f"page-warmed {report['component']!r}: {moved} B in {took:.0f} ms ({rate}), {served}"
