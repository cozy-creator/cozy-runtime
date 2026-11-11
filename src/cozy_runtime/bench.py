"""Unstable: for the release gate only, outside the stable API; it may change or go in any release.

The Runtime's attention bench (xs-171): built-in calls of the Runtime's own App (`derive.operations`), which every
environment holds, so the release gate measures the Runtime's kernels from inside the Runtime. A gate script calls them
with `call(bench.attention_speed, ...)`; each returns its measurements as one JSON document. `internal/bench/attention.py`
says what each measures."""

from __future__ import annotations

import json

import msgspec

from cozy_runtime.author import Context


class Measured(msgspec.Struct, frozen=True):
    """One call's measurements, as JSON (the release gate's `suites/attention/collect.py` reads them)."""

    detail: str


def _run(work: object) -> Measured:
    from cozy_runtime.internal.bench import attention

    found = attention.off_request(work)  # type: ignore[arg-type]  # a zero-argument callable
    return Measured(json.dumps(found, sort_keys=True))


def attention_inventory(ctx: Context) -> Measured:
    """This card's hardware class, its attention kernels, and the sets of them that can share one process."""
    from cozy_runtime.internal.bench import attention

    return _run(attention.inventory)


def attention_speed(ctx: Context, *, kernels: str = "", points: str = "", seed: int = 171) -> Measured:
    """`kernels` (comma-separated; SDPA joins as the anchor) timed over the grid's points (`points`, comma-separated
    ids; empty is every one), interleaved, three rounds."""
    from cozy_runtime.internal.bench import attention

    names = [n for n in kernels.split(",") if n]
    wanted = {p for p in points.split(",") if p}
    chosen = [p for p in attention.GRID if not wanted or p.id in wanted]
    return _run(lambda: attention.speed(names, chosen, seed, ctx))


def attention_quality(ctx: Context, *, kernel: str, seed: int = 171) -> Measured:
    """`kernel`'s worst-head relative L2 against FP32 attention on every site, with the best answer a kernel could give
    without computing."""
    from cozy_runtime.internal.bench import attention

    return _run(lambda: attention.quality(kernel, seed, ctx))
