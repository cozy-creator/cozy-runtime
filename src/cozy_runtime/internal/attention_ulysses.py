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
from typing import Any, TypedDict

from cozy_runtime.internal.attention_registry import register

# Tensors, process groups, parallel configs and Diffusers' backend members are `Any`: neither
# torch nor diffusers is installed in the check venv.


class Options(TypedDict):
    """The keyword options every `Local.forward` takes; `attn_mask` is a tensor or None."""

    attn_mask: Any
    dropout_p: float
    is_causal: bool
    scale: float | None
    enable_gqa: bool


@dataclass(frozen=True, slots=True)
class Local:
    """A forward-only kernel over [batch, sequence, heads, dim].

    `forward(query, key, value, *, attn_mask, dropout_p, is_causal, scale, enable_gqa)`
    validates what it serves; `impl(query, key, value, options)` names the concrete
    implementation serving those operands, e.g. the SDPA backend torch dispatches.
    """

    forward: Callable[..., Any]
    impl: Callable[..., str]
    head_local: bool = True


#: backend -> implementations served in the observed scope on this rank
_SEEN: ContextVar[dict[str, set[str]] | None] = ContextVar("cozy_attention_seen", default=None)
_MEMBERS: dict[str, Any] = {}


@contextmanager
def observing() -> Iterator[dict[str, set[str]]]:
    """Collect, per backend, the implementations that served on this rank in this scope."""
    seen: dict[str, set[str]] = {}
    token = _SEEN.set(seen)
    try:
        yield seen
    finally:
        _SEEN.reset(token)


def member(backend: str, local: Local) -> Any:
    """The Diffusers member for `backend`, registered once per process."""
    found = _MEMBERS.get(backend)
    if found is None:
        found = register(
            backend, _dispatch(backend, local), supports_context_parallel=local.head_local
        )
        _MEMBERS[backend] = found
    return found


def _dispatch(backend: str, local: Local) -> Callable[..., Any]:
    def attention(
        query: Any,
        key: Any,
        value: Any,
        attn_mask: Any = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        _parallel_config: Any = None,
    ) -> Any:
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


def _call(backend: str, local: Local, query: Any, key: Any, value: Any, options: Options) -> Any:
    import torch

    seen = _SEEN.get()
    if seen is not None and not torch.compiler.is_compiling():
        seen.setdefault(backend, set()).add(local.impl(query, key, value, options))
    return local.forward(query, key, value, **options)


def group(config: Any) -> Any:
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
    query: Any, key: Any, value: Any, handle: Any, compute: Callable[[Any, Any, Any], Any]
) -> Any:
    """Sequence shards -> full sequence per head subset -> `compute` -> back, using Diffusers'
    unequal-length exchange in both directions (packed H3 lengths do not divide)."""
    from diffusers.models.attention_dispatch import (
        all_to_all_single_any_o_async,
        all_to_all_single_any_qkv_async,
        ulysses_anything_metadata,
    )

    metadata = ulysses_anything_metadata(query)
    waits = [all_to_all_single_any_qkv_async(t, handle, **metadata) for t in (query, key, value)]
    q, k, v = (wait() for wait in waits)
    output = compute(q, k, v)
    result = all_to_all_single_any_o_async(output, handle, **metadata)()
    if result.shape != query.shape or result.dtype != query.dtype:
        raise ValueError("Ulysses attention returned incompatible output")
    return result


def describe(seen: dict[str, set[str]], names: dict[str, str]) -> str:
    """One implementation line: the bare implementation when one kernel served, else
    `kernel:impl` per kernel."""
    rows = {names.get(backend, backend): "+".join(sorted(impls)) for backend, impls in seen.items()}
    if len(rows) == 1:
        return next(iter(rows.values()))
    return ",".join(f"{name}:{impl}" for name, impl in sorted(rows.items()))
