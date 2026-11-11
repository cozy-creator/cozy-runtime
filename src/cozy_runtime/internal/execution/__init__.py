"""Torch execution with executor-local residency: pure-site cuts sized by native selection from
their operands' shape estimates."""

from __future__ import annotations

from pathlib import Path

import tensorfs
import torch
from torch import nn

from cozy_runtime.internal.execution import catch, cuts
from cozy_runtime.internal.execution.catch import checkpoint as piece
from cozy_runtime.internal.execution.catch import refusals, serving
from cozy_runtime.internal.execution.cuts import attend, split, windows
from cozy_runtime.internal.executor_replies import Ceiling, GpuMemory
from cozy_runtime.internal.residency import Request, blocks_of

__all__ = [
    "Steps",
    "attend",
    "install",
    "piece",
    "record",
    "refusals",
    "serving",
    "split",
    "windows",
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
    catch.wrap(module, "model", catch.passing)


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
    request: Request, gpu: int, device: int | None, store: Path, torch_peak: int | None
) -> GpuMemory:
    """Physical GPU `gpu`'s memory observation. The historical `varena` wire key remains;
    the record identifies stock Torch's process-wide allocator scope explicitly."""
    return GpuMemory(
        gpu=gpu,
        ceiling=_ceiling(device, store),
        stats=request.stats(),  # type: ignore[arg-type]  # a RequestStats is a str-keyed dict on the wire
        torch_peak=torch_peak if device is not None else None,
    )


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
