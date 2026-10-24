"""Torch execution under varena's memory (varena-slim §2): exact pure-site cuts sized from
their operands against the request's room, and the Ulysses agreement.

The executor opens a request (`with mem.request(id, plan=[]) as r, serving(r):`) and installs
the catch on every model it loads (`install`); nothing else here is the executor's to call.
"""

from __future__ import annotations

from pathlib import Path

import tensorfs
import torch
from tensorfs.varena import Fills, Stats
from tensorfs.varena.torch import Request, blocks_of
from torch import nn

from cozy_runtime.internal.execution import agree, catch, cuts
from cozy_runtime.internal.execution.agree import agreed, sizes
from cozy_runtime.internal.execution.catch import cut_rows, piece, serving
from cozy_runtime.internal.execution.cuts import attend, split
from cozy_runtime.internal.executor_replies import Ceiling, GpuMemory

__all__ = [
    "Steps",
    "agreed",
    "attend",
    "cut_rows",
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
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def between(self, start: object, end: object) -> float:
        assert isinstance(start, torch.cuda.Event) and isinstance(end, torch.cuda.Event)
        end.synchronize()  # read at the attempt's end: its streams are idle already
        return float(start.elapsed_time(end))


def record(
    request: Request, gpu: int, device: int | None, store: Path, torch_peak: int
) -> GpuMemory:
    """Physical GPU `gpu`'s row of the reply's `measurements.memory.varena`: TensorFS's fill
    ceilings for CUDA `device` (None: no GPU) and `store`, varena's record of the request and its
    arena's torch high-water."""
    return GpuMemory(
        gpu=gpu,
        ceiling=_ceiling(device, store),
        stats=_stats(request.stats()),
        torch_peak=torch_peak,
    )


def _stats(stats: Stats) -> dict[str, object]:
    """varena's record of the request (`tensorfs.varena.Stats`, its one definition), copied into
    the reply attribute by attribute: what make_room took by class (F, O, S, N, D, W, P), the
    most it held at a pass end, its passes' compute bytes and the request-end audit."""
    moved = stats.moved
    return {
        "request": stats.request,
        "moved": {
            "kept": _fills(moved.kept),
            "leased": _fills(moved.leased),
            "store": _fills(moved.store),
            "speculative": moved.speculative,
        },
        "trace": [
            {
                "step": ref.step,
                "set": ref.set,
                "block": ref.block,
                "bytes": ref.bytes,
                "origin": str(ref.origin),
                "speculative": ref.speculative,
            }
            for ref in stats.trace
        ],
        "steps": [{"capacity": s.capacity, "evicted": list(s.evicted)} for s in stats.steps],
        "stall_ns": stats.stall_ns,
        "evicted": list(stats.evicted),
        "peak": stats.peak,
        "compute": stats.compute,
        "host": stats.host,
        "leftover": stats.leftover,
        "make_room": [
            {
                "step": m.step,
                "trigger": m.trigger,
                "need": m.need,
                "freed": m.freed,
                "waited": m.waited,
            }
            for m in stats.make_room
        ],
        "outlived": [{"ptr": o.ptr, "bytes": o.bytes, "stream": o.stream} for o in stats.outlived],
    }


def _fills(fills: Fills) -> dict[str, int]:
    return {"bytes": fills.bytes, "ns": fills.ns}


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
