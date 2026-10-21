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
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict

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


class Execution(Protocol):
    """The execution layer's side (`internal.execution`, installed with the memory): the
    attention cut, the pass's agreed exchange level, and a collective's extent."""

    def attend(
        self,
        call: Callable[..., Tensor],
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
        *,
        rows: int = 0,
        heads: bool = True,
    ) -> Tensor: ...

    def agreed(self) -> int: ...

    def collective(self, site: str) -> AbstractContextManager[None]: ...


_EXECUTION: Execution | None = None
#: the exchanges' one cut site (`execution.agree.SITE`)
SITE = "ulysses"


def install(execution: Execution | None) -> None:
    global _EXECUTION
    _EXECUTION = execution


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
    if _EXECUTION is None:
        return local.forward(query, key, value, **options)
    return _EXECUTION.attend(
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
    """Sequence shards -> full sequence per head subset -> `compute` -> back, rows unequal
    across GPUs (packed H3 lengths do not divide), over NCCL or the group's copy engines when it
    formed over them (`copy_engine`). Heads cross in the groups of the level the pass agreed
    (`Execution.agreed`: all at once at 0, then ever smaller groups of whole heads per GPU),
    each group's output back before the next group's Q/K/V cross."""
    heads, world = int(query.shape[2]), handle.size()
    groups = sorted({heads, *(s for s in range(world, heads, world) if heads % s == 0)})[::-1]
    level = _EXECUTION.agreed() if _EXECUTION is not None else 0
    step = groups[min(level, len(groups) - 1)]
    with nullcontext() if _EXECUTION is None else _EXECUTION.collective(SITE):
        if step == heads:
            return _exchange(query, key, value, handle, compute)
        out = query.new_empty(query.shape)
        for at in range(0, heads, step):
            part = slice(at, at + step)
            out[:, :, part] = _exchange(
                query[:, :, part], key[:, :, part], value[:, :, part], handle, compute
            )
    return out


def _exchange(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    handle: ProcessGroup,
    compute: Callable[[Tensor, Tensor, Tensor], Tensor],
) -> Tensor:
    import torch
    import torch.distributed as dist

    engines = copy_engine.for_group(handle)
    if engines is not None:
        result = _copy_engine_exchange(engines, query, key, value, handle, compute)
        if result.shape != query.shape or result.dtype != query.dtype:
            raise ValueError("Ulysses attention returned incompatible output")
        return result
    batch, rows, heads, dim = (int(n) for n in query.shape)
    world = handle.size()
    local = heads // world
    gathered = torch.empty(world, dtype=torch.int64, device=query.device)
    dist.all_gather_into_tensor(gathered, torch.tensor([rows], device=query.device), group=handle)
    sizes, mine = [int(n) for n in gathered.tolist()], [rows] * world
    # Every buffer the exchanges cross through, before the first: out of memory, this GPU
    # recovers (or fails) before its peers wait on it. Q's send buffer takes the output back
    # and K's receive buffer sends it, so the peak is the exchange's, not more.
    sends = [query.new_empty((world * rows, batch, local, dim)) for _ in range(3)]
    gots = [query.new_empty((sum(sizes), batch, local, dim)) for _ in range(3)]
    for send, tensor in zip(sends, (query, key, value), strict=True):
        grouped = tensor.reshape(batch, rows, world, local, dim).permute(2, 1, 0, 3, 4)
        send.view(world, rows, batch, local, dim).copy_(grouped)
    for got, send in zip(gots, sends, strict=True):
        dist.all_to_all_single(got, send, sizes, mine, group=handle)
    returned, back = sends[0], gots[1]
    del sends
    q, k, v = (got.permute(1, 0, 2, 3).contiguous() for got in gots)
    del gots
    output = compute(q, k, v)
    del q, k, v  # not live through the output's exchange
    back.copy_(output.permute(1, 0, 2, 3))
    del output
    dist.all_to_all_single(returned, back, mine, sizes, group=handle)
    merged = returned.view(world, rows, batch, local, dim).permute(2, 1, 0, 3, 4)
    result = merged.reshape(batch, rows, heads, dim)
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
