"""Ulysses' cut level and row counts, agreed once per pass (varena-slim §1, Collectives).

At a sharded model's pass start every GPU of its group sends two int64s in one small async
all-gather: the level this process proposes for the exchanges (the finest its room needed for
the last pass's exchanges, from their shapes: `propose`) and a digest of its input shapes. The
first exchange waits on it and every GPU keeps the most of them; every exchange of the pass
crosses in the same head groups. Mid-pass a GPU never runs an exchange again.

Every GPU's input shapes fix every exchange's rows on every GPU. So an exchange gathers its
rows only in the first pass at a group's shapes (`sizes`), and later passes at the same shapes
read them: each GPU decides alike, on the same gathered digests.
"""

from __future__ import annotations

import threading
import weakref
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn

from cozy_runtime.internal.execution.catch import Args, Kwargs, tensors

#: every exchange of a model is one cut site: the same shapes cross in every block
SITE = "ulysses"


class _Pass(threading.local):
    def __init__(self) -> None:
        #: the pass's gathered (level, shapes digest) per GPU: in flight, or agreed
        self.level: tuple[dist.Work, torch.Tensor] | int | None = None
        #: its model and every GPU's shapes digest (None: no sharded pass), and the exchanges
        #: it ran so far
        self.model: nn.Module | None = None
        self.key: tuple[int, ...] | None = None
        self.exchanges = 0


_pass = _Pass()
#: the level this process proposes at the next pass: the finest its room needed for an exchange
_PROPOSED = [0]
#: per sharded model, the rows each exchange of a pass at the latest shapes gathered, by its
#: position in the pass
_ROWS: weakref.WeakKeyDictionary[nn.Module, tuple[tuple[int, ...], list[list[int]]]] = (
    weakref.WeakKeyDictionary()
)


@contextmanager
def passing(module: nn.Module, args: Args, kwargs: Kwargs) -> Iterator[None]:
    """A pass of `module`: when it is sharded, its group's level and shapes are being gathered
    meanwhile."""
    group = _group(module)
    outer = (_pass.level, _pass.model, _pass.key, _pass.exchanges)
    _pass.level, _pass.model, _pass.key, _pass.exchanges = None, None, None, 0
    if group is not None:
        mine, _PROPOSED[0] = _PROPOSED[0], 0  # this process's room's level, from the last pass
        shapes = zlib.crc32(repr([(t.shape, t.dtype) for t in tensors((args, kwargs))]).encode())
        sent = torch.tensor([mine, shapes], dtype=torch.int64, device=_device())
        got = torch.empty(2 * group.size(), dtype=torch.int64, device=sent.device)
        work = dist.all_gather_into_tensor(got, sent, group=group, async_op=True)
        _pass.level, _pass.model = (work, got), module
    try:
        yield
    finally:
        _pass.level, _pass.model, _pass.key, _pass.exchanges = outer


def agreed() -> int:
    """The level this pass's exchanges cross at (0 outside a sharded pass), kept as this
    process's the first time it is read."""
    found = _pass.level
    if isinstance(found, tuple):
        work, got = found
        work.wait()
        levels, *_ = gathered = got.view(-1, 2).T.tolist()
        found = _pass.level = max(levels)
        _pass.key = tuple(gathered[1])
    return found or 0


def propose(level: int) -> None:
    """An exchange's level as this GPU's room sizes it now: the next pass's proposal."""
    _PROPOSED[0] = max(_PROPOSED[0], level)


def sizes(gather: Callable[[], list[int]]) -> list[int]:
    """Every GPU's rows at this exchange: `gather()` in the first pass at the group's shapes
    (and outside a sharded pass), else what it gathered then."""
    agreed()
    at, key, model = _pass.exchanges, _pass.key, _pass.model
    _pass.exchanges += 1
    if key is None or model is None:
        return gather()
    known = _ROWS.get(model)
    if known is None or known[0] != key:
        known = _ROWS[model] = (key, [])
    if at < len(known[1]):
        return known[1][at]
    rows = gather()
    known[1].append(rows)
    return rows


def _group(module: nn.Module) -> dist.ProcessGroup | None:
    """The Ulysses group of a model Runtime sharded (`parallel.cp`), else None."""
    config = getattr(module, "_parallel_config", None)
    cp = getattr(config, "context_parallel_config", None)
    if cp is None or int(cp.ulysses_degree) <= 1 or cp._ulysses_mesh is None:
        return None
    group: dist.ProcessGroup = cp._ulysses_mesh.get_group()
    return group


def _device() -> torch.device:
    if torch.cuda.is_initialized():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
