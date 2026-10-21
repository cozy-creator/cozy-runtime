"""One Ulysses wrapper that makes every Runtime-local attention kernel context-parallel.

After Diffusers' unequal-length all-to-all each rank holds the WHOLE sequence for its head
subset, so a forward-only kernel whose reductions stay within one head computes exactly what
one GPU computes for those heads. `member` registers a `Local` kernel as a Diffusers backend:
without a parallel config it calls the kernel, with Runtime's Ulysses config it exchanges
around it. Ring attention needs LSE merging and is refused. A kernel with a cross-head
reduction (`head_local=False`) is registered without context parallelism.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from diffusers.models.attention_dispatch import AttentionBackendName
    from torch import Tensor
    from torch.distributed import ProcessGroup

    class PartialOptions(TypedDict, total=False):
        """Forwarding adapters may omit options and use the concrete kernel's defaults."""

        attn_mask: Tensor | None
        dropout_p: float
        is_causal: bool
        scale: float | None
        enable_gqa: bool


from cozy_runtime.internal.attention_registry import register
from cozy_runtime.internal.parallel import copy_engine


class Options(TypedDict):
    """The keyword options every `Local.forward` takes; `attn_mask` is a tensor or None."""

    attn_mask: Tensor | None
    dropout_p: float
    is_causal: bool
    scale: float | None
    enable_gqa: bool


@dataclass(frozen=True, slots=True)
class Local:
    """A forward-only kernel over [batch, sequence, heads, dim].

    `forward(query, key, value, *, attn_mask, dropout_p, is_causal, scale, enable_gqa)`
    validates what it serves; `impl(query, key, value, options)` names the concrete
    implementation serving those operands, e.g. the SDPA backend torch dispatches. `rows`: the
    query block a call may be cut at and compute each row as whole (memory's `attend` cut);
    0 where the kernel needs every query row at once (a scale or mean over the sequence).
    """

    forward: Callable[..., Tensor]
    impl: Callable[..., str]
    head_local: bool = True
    rows: int = 0


#: backend -> implementations served in the observed scope on this rank, and the operand
#: signatures already named there (an implementation is named once per signature, not per call)
_SEEN: ContextVar[tuple[dict[str, set[str]], set[object]] | None] = ContextVar(
    "cozy_attention_seen", default=None
)
_MEMBERS: dict[str, AttentionBackendName] = {}
#: the memory's attention cut (`tensorfs.varena.cuts.attend`), installed with the memory
_CUT: Callable[..., Tensor] | None = None
#: what one allocation on a device may take now (`tensorfs.varena.cuts.available`; 0 unknown)
_ROOM: Callable[[object], int] | None = None
#: an exchange's buffers at their most, in GPU shares of the heads crossing: Q, K and V in
#: flight, each in its exchange's layout and its result (6), and the next one being laid out
_BUFFERS = 7


def install(cut: Callable[..., Tensor] | None, room: Callable[[object], int] | None = None) -> None:
    global _CUT, _ROOM
    _CUT, _ROOM = cut, room


@contextmanager
def observing() -> Iterator[dict[str, set[str]]]:
    """Collect, per backend, the implementations that served on this rank in this scope."""
    seen: dict[str, set[str]] = {}
    token = _SEEN.set((seen, set()))
    try:
        yield seen
    finally:
        _SEEN.reset(token)


def member(backend: str, local: Local) -> AttentionBackendName:
    """The Diffusers member for `backend`, registered once per process."""
    found = _MEMBERS.get(backend)
    if found is None:
        found = register(
            backend, _dispatch(backend, local), supports_context_parallel=local.head_local
        )
        _MEMBERS[backend] = found
    return found


def _dispatch(backend: str, local: Local) -> Callable[..., Tensor]:
    def attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        _parallel_config: object = None,
    ) -> Tensor:
        if return_lse:
            raise ValueError(f"{backend} does not return log-sum-exp")
        options = Options(
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        if _parallel_config is None:
            return _call(backend, local, query, key, value, options)
        if not local.head_local:
            raise ValueError(f"{backend} reduces across heads and does not serve Ulysses")
        if attn_mask is not None or is_causal or query.shape[2] != key.shape[2]:
            raise ValueError(
                f"{backend} under Ulysses serves unmasked noncausal attention with equal heads"
            )
        return exchange(
            query,
            key,
            value,
            group(_parallel_config),
            lambda q, k, v: _call(backend, local, q, k, v, options),
        )

    return attention


def _call(
    backend: str, local: Local, query: Tensor, key: Tensor, value: Tensor, options: Options
) -> Tensor:
    scope = _SEEN.get()
    if scope is not None:
        masked, causal = options["attn_mask"] is not None, options["is_causal"]
        signature = (backend, query.shape, key.shape, query.dtype, masked, causal)
        seen, named = scope
        if signature not in named:
            named.add(signature)
            seen.setdefault(backend, set()).add(local.impl(query, key, value, options))
    cut = _CUT
    if cut is None:
        return local.forward(query, key, value, **options)
    return cut(
        lambda q, k, v, mask: local.forward(q, k, v, **{**options, "attn_mask": mask}),
        query,
        key,
        value,
        options["attn_mask"],
        rows=0 if options["is_causal"] else local.rows,
        heads=local.head_local,
    )


def group(config: object) -> ProcessGroup:
    """The process group of Runtime's initialized Ulysses configuration; ring is refused."""
    import torch.distributed as dist
    from diffusers.models._modeling_parallel import ParallelConfig

    if not isinstance(config, ParallelConfig) or config.context_parallel_config is None:
        raise ValueError("context parallelism requires an initialized Ulysses configuration")
    cp = config.context_parallel_config
    if cp.ring_degree != 1:
        raise ValueError("ring attention needs LSE merging; only Ulysses is served")
    mesh = cp._ulysses_mesh
    handle = None if mesh is None else mesh.get_group()
    if handle is None or not dist.is_initialized():
        raise ValueError("the Ulysses configuration is not initialized")
    if dist.get_world_size(handle) != cp.ulysses_degree:
        raise ValueError("the Ulysses group does not match its declared degree")
    return handle


def exchange(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    handle: ProcessGroup,
    compute: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tensor:
    """Sequence shards -> full sequence per head subset -> `compute` -> back, using Diffusers'
    unequal-length exchange in both directions (packed H3 lengths do not divide), or the
    group's copy engines when it formed over them (`copy_engine`). Heads cross all at once when
    their buffers fit the room every GPU of the group has, else in the largest groups that do
    (`_heads`), each group's output back before the next group's Q/K/V cross."""
    import torch
    import torch.distributed as dist

    heads, world = int(query.shape[2]), handle.size()
    room = _ROOM(query.device) if _ROOM is not None else 0
    mine = torch.tensor([int(query.shape[1]), room or 1 << 62], device=query.device)
    every = [torch.empty_like(mine) for _ in range(world)]
    dist.all_gather(every, mine, group=handle)  # every GPU sizes its groups from the same facts
    rows, room = sum(int(e[0]) for e in every), min(int(e[1]) for e in every)
    step = _heads(heads, world, rows * int(query.shape[3]) * query.element_size(), room)
    if step == heads:
        return _exchange(query, key, value, handle, compute)
    out = query.new_empty(query.shape)
    for at in range(0, heads, step):
        part = slice(at, at + step)
        out[:, :, part] = _exchange(
            query[:, :, part], key[:, :, part], value[:, :, part], handle, compute
        )
    return out


def _heads(heads: int, world: int, per_head: int, room: int) -> int:
    """Heads one exchange moves: all of them when their buffers (`_BUFFERS` GPU shares of
    `per_head` bytes a head, every row) fit `room`, else the largest group of whole heads on
    each GPU, dividing them, whose buffers fit beside the output the groups fill, else the
    smallest such group."""
    whole = heads // world * per_head
    groups = [s for s in range(world, heads, world) if heads % s == 0]
    if not groups or _BUFFERS * whole <= room:
        return heads
    fits = (s for s in groups if whole + _BUFFERS * (s // world) * per_head <= room)
    return max(fits, default=groups[0])


def _exchange(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    handle: ProcessGroup,
    compute: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tensor:
    from diffusers.models.attention_dispatch import (
        all_to_all_single_any_o_async,
        all_to_all_single_any_qkv_async,
        ulysses_anything_metadata,
    )

    engines = copy_engine.for_group(handle)
    if engines is not None:
        result = _copy_engine_exchange(engines, query, key, value, handle, compute)
        if result.shape != query.shape or result.dtype != query.dtype:
            raise ValueError("Ulysses attention returned incompatible output")
        return result
    metadata = ulysses_anything_metadata(query)
    waits = [all_to_all_single_any_qkv_async(t, handle, **metadata) for t in (query, key, value)]
    q, k, v = (wait() for wait in waits)
    output = compute(q, k, v)
    del q, k, v  # not live through the output's exchange
    result = all_to_all_single_any_o_async(output, handle, **metadata)()
    if result.shape != query.shape or result.dtype != query.dtype:
        raise ValueError("Ulysses attention returned incompatible output")
    return result


def _copy_engine_exchange(
    engines: copy_engine.CopyEngineExchange,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    handle: ProcessGroup,
    compute: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tensor:
    """Diffusers' two exchanges with Q, K and V in one: GPU s holds `sizes[s]` rows of every
    head and sends each GPU its head group; the output returns each GPU its rows."""
    import torch
    import torch.distributed as dist

    batch, rows, heads, dim = (int(size) for size in query.shape)
    world = engines.world
    local = heads // world
    gathered = torch.empty(world, dtype=torch.int64, device=query.device)
    dist.all_gather_into_tensor(gathered, torch.tensor([rows], device=query.device), group=handle)
    sizes = [int(size) for size in gathered.tolist()]

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
    merged = returned.reshape(world, rows, batch, local, dim).permute(2, 1, 0, 3, 4)
    return merged.reshape(batch, rows, heads, dim).contiguous()


def describe(seen: dict[str, set[str]], names: dict[str, str]) -> str:
    """One implementation line: the bare implementation when one kernel served, else
    `kernel:impl` per kernel."""
    rows = {names.get(backend, backend): "+".join(sorted(impls)) for backend, impls in seen.items()}
    if len(rows) == 1:
        return next(iter(rows.values()))
    return ",".join(f"{name}:{impl}" for name, impl in sorted(rows.items()))


def record(backend: str, implementation: str) -> None:
    """Record a producer that projects before the ordinary Q/K/V dispatch seam."""
    scope = _SEEN.get()
    if scope is not None:
        scope[0].setdefault(backend, set()).add(implementation)
