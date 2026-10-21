"""Torch execution under varena's memory (varena-slim §2): the out-of-memory catch, exact cuts,
parking and the Ulysses agreement. varena decides memory; this runs what it answers.

The executor opens a request (`with mem.request(id, plan=[]) as r, serving(r):`) and installs
the catch on every model it loads (`install`); nothing else here is the executor's to call.
"""

from __future__ import annotations

from collections.abc import Sequence

from tensorfs.varena.torch import blocks_of
from torch import nn

from cozy_runtime.internal.execution import agree, catch, cuts
from cozy_runtime.internal.execution.agree import agreed
from cozy_runtime.internal.execution.catch import collective, serving
from cozy_runtime.internal.execution.cuts import attend, launch, split

__all__ = ["agreed", "attend", "collective", "install", "launch", "serving", "split"]


def install(module: nn.Module, state: Sequence[str] = ()) -> None:
    """The catch on a loaded model: each block varena binds and the model itself (its pass,
    which agrees its exchanges' level when sharded), with its FFN and norm cuts. `state`: the
    attribute paths its forward changes (`_cozy_state`), put back before a block runs again."""
    kept = tuple(state)
    for path in blocks_of(module):
        catch.wrap(module.get_submodule(path), catch.Block(path, module, kept))
    catch.wrap(module, catch.Block("model", module, kept), agree.passing)
    cuts.install(module)
