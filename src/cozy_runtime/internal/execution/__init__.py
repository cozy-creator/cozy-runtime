"""Torch execution under varena's memory (varena-slim §2): exact pure-site cuts that recover
from the allocator's OOM, and the Ulysses agreement. varena makes room; this runs finer.

The executor opens a request (`with mem.request(id, plan=[]) as r, serving(r):`) and installs
the catch on every model it loads (`install`); nothing else here is the executor's to call.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import tensorfs
import torch
from tensorfs.varena import allocator_refusals, allocator_stats
from tensorfs.varena.torch import Request, blocks_of
from torch import nn

from cozy_runtime.internal.execution import agree, catch, cuts
from cozy_runtime.internal.execution.agree import agreed, sizes
from cozy_runtime.internal.execution.catch import collective, piece, serving
from cozy_runtime.internal.execution.cuts import attend, split
from cozy_runtime.internal.executor_replies import Allocator, Ceiling, GpuMemory, Refusal
from cozy_runtime.internal.executor_replies import Stats as Record

__all__ = [
    "Steps",
    "agreed",
    "attend",
    "collective",
    "counters",
    "install",
    "piece",
    "record",
    "serving",
    "sizes",
    "split",
]


def install(module: nn.Module) -> None:
    """Install known pure cuts and name their enclosing blocks, without replay wrappers."""
    cuts.install(module)
    blocks = blocks_of(module)
    for path in blocks:
        catch.wrap(module.get_submodule(path), path)
    for path, child in module.named_children():
        if path not in blocks:
            catch.wrap(child, path)
    catch.wrap(module, "model", agree.passing)


class Steps:
    """The device's step clock (`author._observations.DEVICE`): a CUDA event recorded on the
    current stream at each step, the device time between two read once it passed both."""

    def mark(self) -> object:
        event = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        event.record()
        return event

    def between(self, start: object, end: object) -> float:
        assert isinstance(start, torch.cuda.Event) and isinstance(end, torch.cuda.Event)
        end.synchronize()  # read at the attempt's end: its streams are idle already
        return float(start.elapsed_time(end))


def record(
    request: Request, gpu: int, device: int | None, store: Path, since: Allocator | None
) -> GpuMemory:
    """Physical GPU `gpu`'s row of the reply's `measurements.memory.varena`: TensorFS's fill
    ceilings for CUDA `device` (None: no GPU) and `store`, varena's record of the request, its
    pool's drop, and torch's allocator since its start (`counters`)."""
    now = counters(device)
    delta = None if now is None or since is None else _delta(now, since)
    # The callback's last refusal per pool is the process's: this request's only when the
    # device's refusals grew while it ran.
    refused = () if delta is None or not delta.refused else allocator_refusals(device)
    return GpuMemory(
        gpu=gpu,
        ceiling=_ceiling(device, store),
        stats=msgspec.json.decode(request.stats().json(), type=Record),
        torch=delta,
        refusals=tuple(
            Refusal(pool=pool, why=why, size=size, free=free)
            for _, pool, why, size, free in refused
        ),
        flush_ms=round(request.flush_ms, 3),
        pool_free=tuple(request.pool_free),
    )


def counters(device: int | None) -> Allocator | None:
    """torch's allocator counters on CUDA `device` now, and varena's callback's refusals there
    (None: no GPU)."""
    if device is None:
        return None
    found = torch.cuda.memory_stats(device)
    names = [name for name in Allocator.__struct_fields__ if name != "refused"]
    return Allocator(
        **{name: int(found.get(name, 0)) for name in names},
        refused=allocator_stats(device)[4],
    )


def _delta(now: Allocator, since: Allocator) -> Allocator:
    fields = Allocator.__struct_fields__
    return Allocator(**{f: getattr(now, f) - getattr(since, f) for f in fields})


#: each (device, store)'s measured ceilings, once per executor
_CEILINGS: dict[tuple[int | None, Path], Ceiling] = {}


def _ceiling(device: int | None, store: Path) -> Ceiling | None:
    """TensorFS times the copies once (no GPU has no H2D); None while it cannot (a store with
    no bytes to read yet), measured again at the next record."""
    found = _CEILINGS.get((device, store))
    if found is None:
        try:
            measured = tensorfs.ceiling(device=device, path=store)
        except tensorfs.Error:
            return None
        found = Ceiling(h2d_gbps=measured.h2d_gbps, disk_gbps=measured.disk_gbps)
        _CEILINGS[(device, store)] = found
    return found
