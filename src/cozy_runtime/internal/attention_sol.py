"""Upstream BF16 Sol-Attn with explicit document and conditioning semantics.

No sparse kernel is implemented here. Sparse block selection is approximate;
protected query rows are recomputed with the selected dense reference. The
model supplies its live layout, step policy and dense module paths explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from inspect import Parameter, Signature, signature
from types import MethodType
from typing import Any

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout


@dataclass(frozen=True, slots=True)
class Site:
    component: str
    path: str


_SITE: ContextVar[Site | None] = ContextVar("cozy_sol_site", default=None)
_DENSE: Callable[..., Any] | None = None
_DENSE_IDENTITY = ""
_COUNTS: ContextVar[dict[str, int] | None] = ContextVar("cozy_sol_counts", default=None)
_RANK_COUNTS: ContextVar[dict[int, dict[str, int]] | None] = ContextVar(
    "cozy_sol_rank_counts", default=None
)


@contextmanager
def observing(*, ranks: dict[int, dict[str, int]] | None = None) -> Iterator[dict[str, int]]:
    """Count successful calls in this request, restoring any outer observation."""
    counts = dict.fromkeys(("sparse", "dense_step", "dense_path", "dense_prefix"), 0)
    token = _COUNTS.set(counts)
    rank_token = _RANK_COUNTS.set(ranks)
    try:
        yield counts
    finally:
        _COUNTS.reset(token)
        _RANK_COUNTS.reset(rank_token)


def _count(kind: str) -> None:
    counts = _COUNTS.get()
    if counts is not None:
        counts[kind] += 1


def record_rank(rank: int, counts: dict[str, int]) -> None:
    """Retain counts acknowledged by a follower on the existing mirrored-call reply."""
    ranks = _RANK_COUNTS.get()
    if ranks is None:
        return
    if (
        type(rank) is not int
        or rank < 1
        or set(counts) != {"sparse", "dense_step", "dense_path", "dense_prefix"}
        or any(type(value) is not int or value < 0 for value in counts.values())
    ):
        raise ValueError("invalid follower Sol attention counts")
    total = ranks.setdefault(rank, dict.fromkeys(counts, 0))
    for name, value in counts.items():
        total[name] += value


def bind_dense(call: Callable[..., Any], identity: str) -> None:
    """Install the already-resolved dense reference once before construction warmup."""
    if not callable(call) or not identity:
        raise ValueError("Sol requires a resolved dense attention reference and its identity")
    global _DENSE, _DENSE_IDENTITY
    if _DENSE is not None and identity != _DENSE_IDENTITY:
        raise RuntimeError("Sol dense reference is already bound to a different implementation")
    _DENSE, _DENSE_IDENTITY = call, identity


def dense_identity() -> str:
    return _DENSE_IDENTITY


@dataclass(frozen=True, slots=True)
class _InstalledSite:
    site: Site
    forward: Callable[..., Any]
    original: Callable[..., Any]
    bind_original: bool
    had_instance_forward: bool
    prior_instance_forward: Any


def install_site(module: Any, component: str, path: str) -> None:
    """Scope the actual forward, after pre-hooks and before post-hooks.

    Paired hooks cannot identify an invocation whose pre-hook never ran: an
    always-call post-hook can consume an outer recursive call's token. A local
    try/finally around the real forward has exactly one token per entered call.
    """
    site = Site(component, path)
    existing = getattr(module, "_cozy_sol_site", None)
    if existing is not None:
        if existing.site != site:
            raise ValueError("attention module was already assigned another Sol site")
        if module.forward is not existing.forward:
            raise ValueError("attention forward was replaced after installing its Sol site")
        return
    original = module.forward
    bound = isinstance(original, MethodType) and original.__self__ is module
    call = original.__func__ if bound else original
    had_forward = "forward" in vars(module)
    prior = vars(module).get("forward")
    original_signature = signature(original)
    self_name = "_cozy_module"
    while self_name in original_signature.parameters:
        self_name += "_"

    @wraps(call, updated=())
    def scoped_forward(self: Any, /, *args: Any, **kwargs: Any) -> Any:
        # Read the copied module's record rather than closing over a bound method
        # on the original module. deepcopy must execute the copied weights.
        installed = self._cozy_sol_site
        token = _SITE.set(installed.site)
        try:
            if installed.bind_original:
                return installed.original(self, *args, **kwargs)
            return installed.original(*args, **kwargs)
        finally:
            _SITE.reset(token)

    if not bound:
        # A callable object's metadata must not retain its original mutable state.
        # Its explicit signature below still preserves the public call contract.
        del scoped_forward.__wrapped__
    scoped_forward.__dict__["__signature__"] = Signature(
        [Parameter(self_name, Parameter.POSITIONAL_ONLY), *original_signature.parameters.values()],
        return_annotation=original_signature.return_annotation,
    )
    wrapper = MethodType(scoped_forward, module)
    module.forward = wrapper
    try:
        module._cozy_sol_site = _InstalledSite(site, wrapper, call, bound, had_forward, prior)
    except BaseException:
        if had_forward:
            module.forward = prior
        else:
            del module.forward
        raise


def remove_site(module: Any, component: str, path: str) -> None:
    """Restore the previous forward only while our exact wrapper still owns it."""
    existing = getattr(module, "_cozy_sol_site", None)
    if existing is None:
        return
    if existing.site != Site(component, path):
        raise ValueError("attention module is assigned another Sol site")
    if module.forward is not existing.forward:
        raise ValueError("attention forward was replaced after installing its Sol site")
    if existing.had_instance_forward:
        module.forward = existing.prior_instance_forward
    else:
        del module.forward
    del module._cozy_sol_site


def _dense(query: Any, key: Any, value: Any, scale: float | None) -> Any:
    if _DENSE is None:
        raise RuntimeError("Sol dense reference was not resolved before execution")
    output = _DENSE(query=query, key=key, value=value, scale=scale)
    _validate_result(output, query)
    return output


def _validate_result(output: Any, query: Any) -> None:
    import torch

    if (
        not isinstance(output, torch.Tensor)
        or output.shape != query.shape
        or output.dtype != query.dtype
        or output.device != query.device
    ):
        raise ValueError("attention backend returned incompatible output")


def _native(query: Any, key: Any, value: Any, scale: float | None, sink_tokens: int) -> Any:
    """Call the upstream kernel; a full-document sink selects every KV block.

    This does not perform the wrapper's protected-query recomputation. The same
    public upstream call therefore provides an all-keys arithmetic control.
    """
    return import_module("sol_attn").sol_attn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        scale=scale,
        tau=1.0,
        thresh_type="exact",
        kv_splits=1,
        sink_start=0,
        sink_tokens=sink_tokens,
    )


def sol_attention(
    query: Any,
    key: Any,
    value: Any,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    _parallel_config: Any = None,
    return_lse: bool = False,
) -> Any:
    """One BF16 NHD document; preserve its prefix and zero explicit trailing padding.

    Policy v1 uses upstream tau=1, exact threshold statistics and one KV split.
    `exact` names threshold computation, not dense-equivalent attention. Initial
    dense steps and paths are explicit model policy in the active layout.
    """
    import torch

    if torch.compiler.is_compiling():
        raise ValueError("Sol's model layout and site scope require eager execution")
    layout, site = _ACTIVE_LAYOUT.get(), _SITE.get()
    if layout is None or site is None:
        raise ValueError(
            "Sol requires model-owned attention layout and a registered attention site"
        )
    if attn_mask is not None or is_causal or dropout_p != 0 or enable_gqa or return_lse:
        raise ValueError("Sol supports unmasked noncausal inference without dropout, GQA or LSE")
    group = _parallel_group(_parallel_config) if _parallel_config is not None else None
    tensors = (query, key, value)
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in tensors):
        raise ValueError("Sol requires batch/sequence/head/dimension tensors")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Sol requires equal Q/K/V geometry")
    if query.shape[0] != 1 or query.shape[-1] != 128 or not query.shape[2]:
        raise ValueError("Sol requires one document with head dimension 128")
    if any(
        t.dtype != torch.bfloat16 or t.device != query.device or t.requires_grad for t in tensors
    ):
        raise ValueError("Sol requires BF16 inference inputs on one device")
    if query.device.type != "cuda" or torch.cuda.get_device_capability(query.device) != (9, 0):
        raise ValueError("This Sol adapter supports the H100 SM90 route only")
    if scale is not None:
        import math

        if not math.isfinite(scale):
            raise ValueError("attention scale must be finite")
    if group is not None:
        return _execute_parallel(query, key, value, scale, layout, site, group)
    return _execute(query, key, value, scale, layout, site)


def _parallel_group(config: Any) -> Any:
    """Only Runtime's initialized, head-divisible two/four-rank Ulysses route."""
    import torch.distributed as dist
    from diffusers.models._modeling_parallel import ParallelConfig

    if not isinstance(config, ParallelConfig):
        raise ValueError("Sol context parallelism requires an initialized Ulysses configuration")
    cp = config.context_parallel_config
    if cp is None or cp.ring_degree != 1 or cp.ulysses_degree not in (2, 4):
        raise ValueError("Sol supports Ulysses degrees 2/4 only; ring attention is unsupported")
    if cp._ulysses_mesh is None:
        raise ValueError("Sol Ulysses configuration is not initialized")
    group = cp._ulysses_mesh.get_group()
    if (
        group is None
        or not dist.is_initialized()
        or dist.get_world_size(group) != cp.ulysses_degree
    ):
        raise ValueError("Sol Ulysses group does not match its declared degree")
    return group


def _execute_parallel(
    query: Any,
    key: Any,
    value: Any,
    scale: float | None,
    layout: AttentionLayout,
    site: Site,
    group: Any,
) -> Any:
    """Exchange sequence shards for heads, then apply the same global-document policy.

    Use Diffusers 0.40's actual unequal-length Ulysses exchange in both directions.
    No rank-local interpretation of the model's prefix or trailing padding is valid:
    Sol and the dense reference see the original full token order on each head shard.
    """
    import torch.distributed as dist
    from diffusers.models.attention_dispatch import (
        all_to_all_single_any_o_async,
        all_to_all_single_any_qkv_async,
        ulysses_anything_metadata,
    )

    if query.shape[2] % dist.get_world_size(group):
        raise ValueError("Sol Ulysses requires heads divisible by the group degree")
    metadata = ulysses_anything_metadata(query)
    waits = [all_to_all_single_any_qkv_async(t, group, **metadata) for t in (query, key, value)]
    q, k, v = (wait() for wait in waits)
    output = _execute(q, k, v, scale, layout, site)
    result = all_to_all_single_any_o_async(output, group, **metadata)()
    _validate_result(result, query)
    return result


def _execute(
    query: Any, key: Any, value: Any, scale: float | None, layout: AttentionLayout, site: Site
) -> Any:
    """Apply the model's explicit policy after hardware/operand validation."""
    import torch

    dense_path = any(site.path == p or site.path.startswith(p + ".") for p in layout.dense_paths)
    if dense_path and query.shape[1] < layout.live_tokens:
        # A refiner has its own shorter document; its explicit dense policy wins.
        output = _dense(query, key, value, scale)
        _count("dense_path")
        return output
    if layout.live_tokens > query.shape[1]:
        raise ValueError("attention layout exceeds Q/K/V sequence length")
    q, k, v = (t[:, : layout.live_tokens] for t in (query, key, value))
    if dense_path or layout.step < layout.dense_until_step:
        live_output = _dense(q, k, v, scale)
        _count("dense_path" if dense_path else "dense_step")
    else:
        live_output = _native(q, k, v, scale, layout.protected_prefix)
        _validate_result(live_output, q)
        if any(
            live_output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (query, key, value)
        ):
            raise ValueError("Sol returned output sharing input storage")
        _count("sparse")
        if layout.protected_prefix:
            live_output[:, : layout.protected_prefix] = _dense(
                q[:, : layout.protected_prefix], k, v, scale
            )
            _count("dense_prefix")
    _validate_result(live_output, q)
    if layout.live_tokens == query.shape[1]:
        return live_output
    output = torch.zeros_like(query)
    output[:, : layout.live_tokens].copy_(live_output)
    return output
