"""Torch execution with executor-local residency: exact pure-site cuts sized from
their operands against the request's room, and the Ulysses agreement.

The executor opens a request (`with mem.request(id, plan=[]) as r, serving(r):`) and installs
the catch on every model it loads (`install`); nothing else here is the executor's to call.
"""

from __future__ import annotations

import threading
from pathlib import Path

import tensorfs
import torch
from torch import nn

from cozy_runtime.internal.execution import agree, catch, cuts
from cozy_runtime.internal.execution.agree import agreed, sizes
from cozy_runtime.internal.execution.catch import cut_rows, piece, serving
from cozy_runtime.internal.execution.cuts import attend, split
from cozy_runtime.internal.executor_replies import Ceiling, GpuMemory
from cozy_runtime.internal.residency import Request, blocks_of

__all__ = [
    "Steps",
    "agreed",
    "attend",
    "cut_rows",
    "install",
    "measure",
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
    request: Request, gpu: int, device: int | None, store: Path, torch_peak: int | None
) -> GpuMemory:
    """Physical GPU `gpu`'s memory observation. The historical `varena` wire key remains;
    the record identifies stock Torch's process-wide allocator scope explicitly."""
    return GpuMemory(
        gpu=gpu,
        ceiling=_ceiling(device, store),
        stats=request.stats(),
        torch_peak=torch_peak if device is not None else None,
    )



#: each (device, store)'s ceilings: None while being measured, absent until a measure starts
_CEILINGS: dict[tuple[int | None, Path], Ceiling | None] = {}


def measure(device: int | None, store: Path) -> None:
    """TensorFS times the copies once (no GPU has no H2D), off every request: it reads up to
    1 GiB of the store and times pinned H2D, 0.2-0.5 s once a boot, which a request's close
    would add to its reply. The executor starts it when its memory opens, beside its imports."""
    key = (device, store)
    if key not in _CEILINGS:
        _CEILINGS[key] = None
        threading.Thread(target=_measure, args=key, name="cozy-ceiling", daemon=True).start()


def _measure(device: int | None, store: Path) -> None:
    try:
        measured = tensorfs.ceiling(device=device, path=store)
    except tensorfs.Error:  # a store with no bytes to read yet: the next record starts it again
        _CEILINGS.pop((device, store), None)
        return
    _CEILINGS[(device, store)] = Ceiling(h2d_gbps=measured.h2d_gbps, disk_gbps=measured.disk_gbps)


def _ceiling(device: int | None, store: Path) -> Ceiling | None:
    """The measured ceilings; None while they are being measured."""
    measure(device, store)
    return _CEILINGS.get((device, store))
