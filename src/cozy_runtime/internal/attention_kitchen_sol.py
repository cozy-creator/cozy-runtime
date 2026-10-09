"""Opt-in Comfy Kitchen Sol kernel under Runtime's existing H3 sparse policy.

This adapter changes the numerical kernel, not the dense-step, dense-path or
protected-prefix policy. It uses current-call statistics and materialized Q/K/V;
ComfyUI's H3 chunked producer and previous-step scale reuse are not enabled.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from cozy_runtime.internal import attention_sol, attention_upstream
from cozy_runtime.internal.attention_ulysses import Local


def sparse(query: Any, key: Any, value: Any, scale: float | None, sink_tokens: int) -> Any:
    """The upstream NHD kernel; the shared policy recomputes exact protected query rows."""
    kitchen = import_module("comfy_kitchen")
    tensors = (x if x.stride(-1) == 1 else x.contiguous() for x in (query, key, value))
    return kitchen.sol_attn(
        *tensors,
        tau=1.0,
        scale=scale,
        sink_blocks=[0, -(-sink_tokens // 64)],
        sink_q=[0, 0],
        token_aug=0,
    )


def forward(
    query: Any,
    key: Any,
    value: Any,
    *,
    attn_mask: Any = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Any:
    return attention_sol._attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        sparse=sparse,
        dense=attention_upstream.kitchen_int8,
    )


KITCHEN_SOL = Local(forward, lambda *_: "comfy_kitchen.sol_attn(int8, tau=1, current-call stats)")


def sage_dense(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    """The shared, explicit dense reference for sparse-arithmetic comparisons."""
    return attention_upstream.SAGE.forward(query, key, value, scale=scale)


def forward_sage(query: Any, key: Any, value: Any, **options: Any) -> Any:
    return attention_sol._attention(query, key, value, **options, sparse=sparse, dense=sage_dense)


KITCHEN_SOL_SAGE = Local(forward_sage, KITCHEN_SOL.impl)
