"""Page-cache warm of a generation's PARKED components (h3a-018).

A parked component is staged on its first declared use, from the store. The residency
ledger on an H100 pod shows that stage at 23-33 GB/s from a warm page cache and at
0.73 GB/s from the block layer -- 1.5 s against 55 s for one 40 GB DiT. The worker reads
each parked component's stored bytes once, right after the prepare that parked it, through
the SAME store stream the fill uses (`ReadLease.stream` over the component's complete read
plan), into host buffers it drops. What it leaves behind is the page cache.

No device byte moves, no lane lock is held, and nothing here decides anything: a component
larger than the host's available memory is skipped BY THAT MEASUREMENT, with the two
numbers, because reading it would only evict what a neighbour just warmed.
"""

from __future__ import annotations

import time
from typing import Required, TypedDict

from cozy_runtime.internal import fill


class WarmReport(TypedDict, total=False):
    component: Required[str]
    skipped: str
    bytes: int
    ms: float
    read_bytes: int


def available_memory_bytes() -> int:
    """`MemAvailable` from /proc/meminfo, or -1 where the kernel does not say."""
    try:
        with open("/proc/meminfo", "rb") as rows:
            for row in rows:
                if row.startswith(b"MemAvailable:"):
                    return int(row.split()[1]) << 10
    except (OSError, ValueError, IndexError):
        return -1
    return -1


def warm_component(store_root: str, manifest: str, component: str) -> WarmReport:
    """Stream one component's stored bytes through the store, for the page cache alone."""
    checkpoint = fill.Checkpoint(store_root, manifest)
    table = checkpoint.header["components"].get(component) or {}
    traversal = [(component, str(key)) for key in table]
    if not traversal:
        return {"component": component, "skipped": "the checkpoint holds no rows for it"}
    plan = checkpoint.read_plan(traversal, fill.WINDOW_BYTES, (component,))
    need = int(plan.bytes)
    available = available_memory_bytes()
    if 0 <= available < need:
        return {
            "component": component,
            "skipped": f"{need} B would not stay cached beside {available} B available",
        }
    slots = [bytearray(fill.slot_bytes(fill.WINDOW_BYTES)) for _ in range(fill.RING_SLOTS)]
    before = fill.block_read_bytes()
    started = time.perf_counter()
    with checkpoint.acquire() as lease:
        streamed = lease.stream(plan, slots, lambda batch: batch.release(), readers=fill.READERS)
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
