"""Torch execution under varena's memory (varena-slim §2): the out-of-memory catch, exact cuts,
parking and the Ulysses agreement. varena decides memory; this runs what it answers.

The executor opens a request (`with mem.request(id, plan=[]) as r, serving(r):`) and installs
the catch on every model it loads (`install`); nothing else here is the executor's to call.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import msgspec
import tensorfs
from tensorfs.varena.torch import Stats, blocks_of
from torch import nn

from cozy_runtime.internal.execution import agree, catch, cuts
from cozy_runtime.internal.execution.agree import agreed, sizes
from cozy_runtime.internal.execution.catch import collective, keep, piece, serving
from cozy_runtime.internal.execution.cuts import attend, launch, split
from cozy_runtime.internal.executor_replies import Ceiling, GpuMemory
from cozy_runtime.internal.executor_replies import Stats as Record

__all__ = [
    "agreed",
    "attend",
    "collective",
    "install",
    "keep",
    "launch",
    "piece",
    "record",
    "serving",
    "sizes",
    "split",
]


def install(module: nn.Module, state: Sequence[str] = ()) -> None:
    """The catch on a loaded model: each block varena binds, and what a pass runs outside them
    (the model's forward, which agrees its exchanges' level when sharded, and each direct
    child's, which a method such as `vae.decode` calls), with its FFN and norm cuts. `state`:
    the attribute paths its forward changes (`_cozy_state`), put back before a block runs
    again."""
    kept, blocks = tuple(state), blocks_of(module)
    for path in blocks:
        catch.wrap(module.get_submodule(path), catch.Block(path, module, kept))
    for path, child in module.named_children():
        if path not in blocks:
            catch.wrap(child, catch.Block(path, module, kept))
    catch.wrap(module, catch.Block("model", module, kept), agree.passing)
    cuts.install(module)


def record(stats: Stats, gpu: int, device: int | None, store: Path) -> GpuMemory:
    """Physical GPU `gpu`'s row of the reply's `measurements.memory.varena`: TensorFS's fill
    ceilings for CUDA `device` (None: no GPU) and `store`, and varena's record of the request."""
    return GpuMemory(
        gpu=gpu,
        ceiling=_ceiling(device, store),
        stats=msgspec.json.decode(stats.json(), type=Record),
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
