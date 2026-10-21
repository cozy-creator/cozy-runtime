"""Torch execution under varena's memory (varena-slim §2): the out-of-memory catch, exact cuts,
parking and the Ulysses agreement. varena decides memory; this runs what it answers.

The executor opens a request (`with mem.request(id, plan=[]) as r, serving(r):`) and installs
the catch on every model it loads (`install`); nothing else here is the executor's to call.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from pathlib import Path

import tensorfs
from tensorfs.varena.torch import Stats, blocks_of
from torch import nn

from cozy_runtime.internal.execution import agree, catch, cuts
from cozy_runtime.internal.execution.agree import agreed
from cozy_runtime.internal.execution.catch import collective, serving
from cozy_runtime.internal.execution.cuts import attend, launch, split

__all__ = ["agreed", "attend", "collective", "install", "launch", "record", "serving", "split"]


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


def record(stats: Stats, gpu: int, device: int | None, store: Path) -> dict[str, object]:
    """Physical GPU `gpu`'s row of the reply's `measurements.memory.varena`: TensorFS's fill
    ceilings for CUDA `device` (None: no GPU) and `store`, and varena's record of the request
    as its core writes it."""
    h2d, disk = _ceiling(device, store)
    return {
        "gpu": gpu,
        "ceiling": {"h2d_gbps": h2d, "disk_gbps": disk},
        "stats": json.loads(stats.json()),
    }


@functools.cache
def _ceiling(device: int | None, store: Path) -> tuple[float | None, float]:
    """Measured once per executor (TensorFS times the copies); no GPU has no H2D."""
    found = tensorfs.ceiling(device=device, path=store)
    return found.h2d_gbps, found.disk_gbps
