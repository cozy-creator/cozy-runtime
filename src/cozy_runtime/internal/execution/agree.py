"""Ulysses' cut level, agreed once per pass (varena-slim §1, Collectives).

At a sharded model's pass start every GPU of its group sends the most of the levels Rust
proposes for the pass's collective sites (`r.levels()`) in one small async all-reduce (max, one
int64 whatever sites a GPU knows, so no GPU can wait on a different collective); the first
exchange waits on it and tells Rust (`r.agree`), and every exchange of the pass crosses in the
same head groups. An OOM on one GPU so becomes the same cut on all of them from the next pass;
mid-pass a GPU recovers only locally (`catch.collective`).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn

from cozy_runtime.internal.execution.catch import CURRENT

#: every exchange of a model is one cut site: the same shapes cross in every block
SITE = "ulysses"


class _Pass(threading.local):
    def __init__(self) -> None:
        #: the pass's level: in flight (the all-reduce and its tensor), agreed, or None
        self.level: tuple[dist.Work, torch.Tensor] | int | None = None


_pass = _Pass()


@contextmanager
def passing(module: nn.Module) -> Iterator[None]:
    """A pass of `module`: when it is sharded, its group's level is being agreed meanwhile."""
    group = _group(module)
    outer = _pass.level
    if group is not None:
        request = CURRENT.get()
        proposed = request.levels() if request is not None else []
        mine = max((level for _, level in proposed), default=0)
        level = torch.tensor([mine], dtype=torch.int64, device=_device())
        work = dist.all_reduce(level, op=dist.ReduceOp.MAX, group=group, async_op=True)
        _pass.level = (work, level)
    else:
        _pass.level = None
    try:
        yield
    finally:
        _pass.level = outer


def agreed() -> int:
    """The level this pass's exchanges cross at (0 outside a sharded pass), told to Rust the
    first time it is read."""
    found = _pass.level
    if isinstance(found, tuple):
        work, level = found
        work.wait()
        found = _pass.level = int(level.item())
        request = CURRENT.get()
        if request is not None:
            request.agree([(SITE, found)])
    return found or 0


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
