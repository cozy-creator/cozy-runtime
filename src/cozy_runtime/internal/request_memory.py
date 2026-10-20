"""Request memory evidence from every executor rank, using physical execution GPU IDs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from cozy_runtime.internal.executor_replies import Copy, Eviction, Memory, Peak, Step

if TYPE_CHECKING:
    from tensorfs.varena.torch import RequestStats


def measured(request: RequestStats | None, devices: Sequence[int]) -> Memory | None:
    if request is None:
        return None

    def physical(gpu: int) -> int:
        if gpu < 0:
            return gpu
        if gpu >= len(devices) or devices[gpu] < 0:
            raise ValueError(f"memory names unmapped CUDA ordinal {gpu}")
        return devices[gpu]

    steps: dict[str, list[int]] = {}
    for step in request.steps:
        steps.setdefault(step.model, []).append(step.paged_in + step.streamed)
    return Memory(
        leftover=sum(n for _, n in request.leftover),
        paged_in=request.copied,
        steps=steps,
        rows=[
            Step(
                model=s.model,
                gpu=physical(s.gpu),
                step=s.step,
                home=s.home,
                resident=s.resident,
                paged_in=s.paged_in,
                streamed=s.streamed,
                stall_ns=s.stall_ns,
                activations=s.activations,
                allocated=s.allocated,
            )
            for s in request.steps
        ],
        evictions=[
            Eviction(
                model=e.model, gpu=physical(e.gpu), rung=e.rung, reason=e.reason, nbytes=e.nbytes
            )
            for e in request.evictions
        ],
        peaks=[
            Peak(model=p.model, shape=p.shape, peak=p.peak, cuts=list(p.cuts))
            for p in request.peaks
        ],
        retries=request.retries,
        leftover_by_gpu=[(physical(gpu), n) for gpu, n in request.leftover],
        plan=[list(step) for step in request.plan],
        copies=[
            Copy(
                kind=c.kind,
                gpu=physical(c.gpu),
                bytes=c.bytes,
                seconds=c.seconds,
                gbps=c.gbps,
                ceiling=c.ceiling,
                pct=c.pct,
                link=c.link,
                busy_s=c.busy_s,
                pin_s=c.pin_s,
                staged=c.staged,
                cached=c.cached,
            )
            for c in request.copies
        ],
    )


def combined(records: Sequence[Memory]) -> Memory | None:
    if not records:
        return None
    plan = records[0].plan
    if any(record.plan != plan for record in records):
        raise ValueError("executor ranks recorded different request plans")
    by_gpu = [row for record in records for row in record.leftover_by_gpu]
    if len({gpu for gpu, _ in by_gpu}) != len(by_gpu):
        raise ValueError("executor ranks recorded duplicate physical GPUs")
    steps: dict[str, list[int]] = {}
    for record in records:
        for model, values in record.steps.items():
            into = steps.setdefault(model, [])
            into.extend([0] * max(len(values) - len(into), 0))
            for index, value in enumerate(values):
                into[index] += value
    return Memory(
        leftover=sum(record.leftover for record in records),
        paged_in=sum(record.paged_in for record in records),
        steps=steps,
        rows=[row for record in records for row in record.rows],
        evictions=[row for record in records for row in record.evictions],
        peaks=[row for record in records for row in record.peaks],
        retries=sum(record.retries for record in records),
        leftover_by_gpu=sorted(by_gpu),
        plan=plan,
        copies=[row for record in records for row in record.copies],
    )
