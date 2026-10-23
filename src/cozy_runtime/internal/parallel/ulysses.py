"""The torch side of Runtime's Ulysses attention: the process group a parallel configuration
names, and the exchange around a head-local kernel (`attention_ulysses`). Sequence shards cross
to whole sequences per head subset, the kernel runs, and its output crosses back; packed H3
lengths do not divide, so every GPU's rows are gathered once per exchange and every crossing
uses them. The exchange's head groups are agreed once per pass (each GPU proposing what its
room needs); an exchange is never rerun locally once any rank can have entered its collectives. Importing this module serves the torch-free
attention dispatch boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed import _functional_collectives as funcol

from cozy_runtime.internal import attention_ulysses
from cozy_runtime.internal.execution import agree, catch, sizing
from cozy_runtime.internal.parallel import copy_engine

type Compute = Callable[[Tensor, Tensor, Tensor], Tensor]


def group(config: object) -> ProcessGroup:
    """The process group of Runtime's initialized Ulysses configuration; ring is refused."""
    cp = getattr(config, "context_parallel_config", None)
    if cp is None:
        raise ValueError("context parallelism requires an initialized Ulysses configuration")
    if cp.ring_degree != 1:
        raise ValueError("ring attention needs LSE merging; only Ulysses is served")
    mesh = cp._ulysses_mesh
    handle = None if mesh is None else mesh.get_group()
    if handle is None or not dist.is_initialized():
        raise ValueError("the Ulysses configuration is not initialized")
    if dist.get_world_size(handle) != cp.ulysses_degree:
        raise ValueError("the Ulysses group does not match its declared degree")
    return cast(ProcessGroup, handle)


def exchange(
    query: Tensor, key: Tensor, value: Tensor, handle: ProcessGroup, compute: Compute
) -> Tensor:
    """`compute` over whole sequences per head subset, back to this GPU's rows of every head."""
    heads, world = int(query.shape[2]), handle.size()

    def gather() -> list[int]:
        mine = torch.tensor([int(query.shape[1])], device=query.device)
        every = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(every, mine, group=handle)
        return [int(e[0]) for e in every]

    sizes = agree.sizes(gather)
    pad = -heads % world
    if pad:
        query, key, value = (
            torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in (query, key, value)
        )
    padded = heads + pad
    groups = sorted(
        {padded, *(n for n in range(world, padded, world) if padded % n == 0)}, reverse=True
    )
    step = groups[min(agree.agreed(), len(groups) - 1)]
    agree.propose(_level(groups, query, sum(sizes), world))
    engines = copy_engine.for_group(handle)
    if step == padded:
        out = _cross(engines, query, key, value, handle, sizes, compute)
    else:
        out = query.new_empty(query.shape)
        for at in range(0, padded, step):
            part = slice(at, at + step)
            out[:, :, part] = _cross(
                engines,
                query[:, :, part],
                key[:, :, part],
                value[:, :, part],
                handle,
                sizes,
                compute,
            )
    return out[:, :, :heads] if pad else out


def _level(groups: Sequence[int], query: Tensor, rows: int, world: int) -> int:
    """The finest head group this GPU's room needs for the exchange (P4: the gathered buffer
    is the group's every row): q, k and v sent and received, and the output, for `step` heads.
    Proposed for the next pass; mid-pass every GPU keeps the agreed one."""
    request = catch.CURRENT.get()
    if request is None:
        return 0
    room = request.room()
    batch, local, dim, b = (
        int(query.shape[0]),
        int(query.shape[1]),
        int(query.shape[3]),
        query.element_size(),
    )

    def need(step: int) -> int:
        sent = 4 * batch * local * step * dim * b
        gathered = 4 * batch * rows * (step // world) * dim * b
        return sizing.rounded(sent, room.granularity) + sizing.rounded(gathered, room.granularity)

    fits = room.free + room.cold
    return next((n for n, step in enumerate(groups) if need(step) <= fits), len(groups) - 1)


def _cross(
    engines: copy_engine.CopyEngineExchange | None,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    handle: ProcessGroup,
    sizes: Sequence[int],
    compute: Compute,
) -> Tensor:
    """One exchange of whole heads per GPU, over the copy engines when the group formed over
    them, else NCCL; its result is shaped like `query`."""
    if engines is not None:
        result = _copy_engines(engines, query, key, value, sizes, compute)
    else:
        result = _nccl(query, key, value, handle, sizes, compute)
    if result.shape != query.shape or result.dtype != query.dtype:
        raise ValueError("Ulysses attention returned incompatible output")
    return result


def _nccl(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    handle: ProcessGroup,
    sizes: Sequence[int],
    compute: Compute,
) -> Tensor:
    """Diffusers' unequal-length exchange in both directions, at the gathered sizes: GPU p is
    sent its head group of this GPU's rows, and gets back its rows of every head."""
    batch, rows, heads, dim = (int(n) for n in query.shape)
    world = len(sizes)
    local = heads // world

    def out(t: Tensor) -> Tensor:  # (B, rows, H, D) -> (S, B, H/W, D), in flight
        laid = t.reshape(batch, rows, world, local, dim).permute(2, 1, 0, 3, 4)
        return funcol.all_to_all_single(
            laid.reshape(world * rows, batch, local, dim), list(sizes), [rows] * world, handle
        )

    def landed(t: Tensor) -> Tensor:
        return _wait(t).permute(1, 0, 2, 3).contiguous()

    flying = [out(t) for t in (query, key, value)]
    q, k, v = (landed(t) for t in flying)
    del flying
    output = compute(q, k, v)
    del q, k, v  # not live through the output's exchange
    back = funcol.all_to_all_single(
        output.permute(1, 0, 2, 3).contiguous(), [rows] * world, list(sizes), handle
    )
    merged = _wait(back).reshape(world, rows, batch, local, dim)
    return merged.permute(2, 1, 0, 3, 4).reshape(batch, rows, heads, dim)


def _wait(t: Tensor) -> Tensor:
    return cast(Tensor, funcol.wait_tensor(t))


def _copy_engines(
    engines: copy_engine.CopyEngineExchange,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sizes: Sequence[int],
    compute: Compute,
) -> Tensor:
    """Diffusers' two exchanges with Q, K and V in one: GPU s holds `sizes[s]` rows of every
    head and sends each GPU its head group; the output returns each GPU its rows."""
    batch, rows, heads, dim = (int(n) for n in query.shape)
    world = engines.world
    local = heads // world

    def heads_first(tensor: Tensor) -> Tensor:
        # (B, S, W, H/W, D) -> (W, S, B, H/W, D): rows for GPU p are block p
        grouped = tensor.reshape(batch, rows, world, local, dim).permute(2, 1, 0, 3, 4)
        return grouped.reshape(world * rows, batch, local, dim).contiguous()

    sent = [[sizes[s]] * world for s in range(world)]
    q, k, v = (
        part.permute(1, 0, 2, 3).contiguous()
        for part in engines.all_to_all([heads_first(t) for t in (query, key, value)], sent)
    )
    output = compute(q, k, v)
    back = [[sizes[p] for p in range(world)] for _ in range(world)]
    (returned,) = engines.all_to_all([output.permute(1, 0, 2, 3).contiguous()], back)
    merged = cast(Tensor, returned).reshape(world, rows, batch, local, dim).permute(2, 1, 0, 3, 4)
    return merged.reshape(batch, rows, heads, dim).contiguous()


attention_ulysses.serve(group, exchange)
