"""Cancellable store-read mechanism; the machine memory policy admits the host footprint."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Required, TypedDict

from cozy_runtime.internal import fill


class WarmReport(TypedDict, total=False):
    component: Required[str]
    skipped: str
    bytes: int
    ms: float
    read_bytes: int
    complete: bool
    batches: int


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
    reading: Callable[[], AbstractContextManager[bool]] = lambda: nullcontext(True),
    allowance: Callable[[int], int] | None = None,
) -> WarmReport:
    """Verified-only optional reads, one slot/reader, interrupted at native batch boundaries.

    The gate registration outlives stream return: TensorFS joins its native readers before
    returning or raising. A demand begin ACK can therefore prove the optional I/O drained.
    Delivered bytes bound the logical pass, not exact cache growth or kernel readahead.
    """
    with reading() as admitted:
        if not admitted or not wanted():
            return {"component": component, "skipped": "foreground read or queue changed"}
        store = fill.store(store_root)
        acquire = getattr(store, "acquire_cozytensors_if_verified", None)
        if not callable(acquire):
            return {"component": component, "skipped": "verified-only acquisition unavailable"}
        try:
            lease = acquire(manifest)
        except Exception as exc:
            return {
                "component": component,
                "skipped": f"verified acquisition: {type(exc).__name__}",
            }
        with lease:
            header_bytes = getattr(lease, "header", None)
            if not isinstance(header_bytes, bytes):
                return {"component": component, "skipped": "verified header unavailable"}
            header = fill.tensorfs_module().parse_header(header_bytes)
            table = header["components"].get(component) or {}
            traversal = [(component, str(key)) for key in table]
            if not traversal:
                return {"component": component, "skipped": "the checkpoint holds no rows for it"}
            plan = fill.tensorfs_module().plan(
                header_bytes, traversal, [component], window=fill.WINDOW_BYTES
            )
            need = int(plan.bytes)
            if not need:
                return {
                    "component": component,
                    "bytes": 0,
                    "ms": 0,
                    "read_bytes": 0,
                    "complete": True,
                }
            slots_bytes = fill.slot_bytes(fill.WINDOW_BYTES)
            budget = need if allowance is None else min(need, allowance(slots_bytes))
            if not wanted():
                return {"component": component, "skipped": "queue changed before native read"}
            if budget < min(need, slots_bytes) or not reserve(slots_bytes):
                return {
                    "component": component,
                    "skipped": f"host admission declined {slots_bytes} B of prefetch",
                }
            delivered_bytes = delivered_batches = 0
            stopped = ""

            def delivered(batch: fill.Batch) -> None:
                nonlocal delivered_bytes, delivered_batches
                delivered_bytes += batch.nbytes
                delivered_batches += 1
                batch.release()
                if not wanted():
                    raise _Superseded("queue changed or demand preparation started")
                next_bytes = min(need - delivered_bytes, slots_bytes)
                if next_bytes > 0 and (
                    budget - delivered_bytes < next_bytes
                    or (allowance is not None and allowance(0) < next_bytes)
                ):
                    raise _Superseded("optional unreclaimed headroom exhausted")

            try:
                slots = [bytearray(slots_bytes)]
            finally:
                # Live cgroup usage now includes this materialized buffer.
                release()
            before, started = fill.block_read_bytes(), time.perf_counter()
            try:
                lease.stream(plan, slots, delivered, readers=1)
            except _Superseded as exc:
                stopped = str(exc)
            took, after = (time.perf_counter() - started) * 1000, fill.block_read_bytes()
            report: WarmReport = {
                "component": component,
                "bytes": delivered_bytes,
                "ms": round(took, 1),
                "read_bytes": max(after - before, 0) if min(before, after) >= 0 else -1,
                "complete": not stopped and delivered_bytes == need,
                "batches": delivered_batches,
            }
            if stopped:
                report["skipped"] = stopped
            return report


def describe(report: WarmReport) -> str:
    """Keep measured work even when a read stopped; a denied read invents no measurements."""
    if "bytes" not in report:
        return report.get("skipped", "no page read measured")
    moved, took, read = report["bytes"], report["ms"], report["read_bytes"]
    rate = f"{moved / took / 1e6:.1f} GB/s" if took > 0 else "rate unavailable"
    if read < 0:
        served = "block-device reads unreadable"
    else:
        cache = "cold" if read > moved // 2 else "warm page cache"
        served = f"{read} B read from the block layer ({cache})"
    measured = f"{moved} B in {took:.1f} ms ({rate}), {served}"
    if "batches" in report:
        measured += f", {report['batches']} batches"
    if report.get("complete") is True:
        return f"page-warmed {report['component']!r}: {measured}"
    state = "partial page-warm" if report.get("complete") is False else "page read"
    if reason := report.get("skipped"):
        measured += f"; stopped: {reason}"
    # Worker notes are bounded: keep work and its stop reason ahead of a long component name.
    return f"{state}: {measured}; component={report['component']!r}"
