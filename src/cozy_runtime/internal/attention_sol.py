"""Upstream BF16 Sol-Attn with explicit document and conditioning semantics.

No sparse kernel is implemented here. Sparse block selection is approximate;
protected query rows are recomputed with the selected dense reference. The
model supplies its live layout, step policy and dense module paths explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from importlib import import_module
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


@contextmanager
def observing() -> Iterator[dict[str, int]]:
    """Count successful calls in this request, restoring any outer observation."""
    counts = dict.fromkeys(("sparse", "dense_step", "dense_path", "dense_prefix"), 0)
    token = _COUNTS.set(counts)
    try:
        yield counts
    finally:
        _COUNTS.reset(token)


def _count(kind: str) -> None:
    counts = _COUNTS.get()
    if counts is not None:
        counts[kind] += 1


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


def install_site(module: Any, component: str, path: str) -> None:
    """Attach site metadata without changing a processor or global attention choice."""
    site = Site(component, path)
    existing = getattr(module, "_cozy_sol_site", None)
    if existing is not None:
        if existing[0] != site:
            raise ValueError("attention module was already assigned another Sol site")
        return
    tokens: ContextVar[tuple[Token[Site | None], ...]] = ContextVar(
        "cozy_sol_site_tokens", default=()
    )

    def before(_module: Any, _args: Any) -> None:
        token = _SITE.set(site)
        tokens.set((*tokens.get(), token))

    def after(_module: Any, _args: Any, _output: Any) -> None:
        stack = tokens.get()
        if stack:
            tokens.set(stack[:-1])
            _SITE.reset(stack[-1])

    first = module.register_forward_pre_hook(before)
    try:
        second = module.register_forward_hook(after, always_call=True)
    except BaseException:
        first.remove()
        raise
    module._cozy_sol_site = site, first, second


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
    if _parallel_config is not None:
        raise ValueError("Sol context parallelism is not qualified")
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
    return _execute(query, key, value, scale, layout, site)


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
        sol_attn = import_module("sol_attn").sol_attn

        live_output = sol_attn(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            scale=scale,
            tau=1.0,
            thresh_type="exact",
            kv_splits=1,
            sink_start=0,
            sink_tokens=layout.protected_prefix,
        )
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
