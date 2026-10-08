"""Explicit FP8 exact QK/PV, FP16 tile accumulators, FP32 online state.

Routing and approximate tails use original BF16 operands with FP32 accumulation.
Every exact carrier is bounded by16. D128 QK sums are bounded by32768; N64 PV
sums by16384. Dense model regions use the same Sage reference as the BF16 control.
This experiment has no fallback to BF16 when compilation or execution fails.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from cozy_runtime.internal import (
    attention_kitchen_sol,
    attention_sol,
    attention_ulysses,
    attention_upstream,
)
from cozy_runtime.internal.attention_ulysses import Local, Options


def sparse(query: Any, key: Any, value: Any, scale: float | None, sink_tokens: int) -> Any:
    import torch

    if query.device.type != "cuda" or torch.cuda.get_device_capability(query.device) != (12, 0):
        raise ValueError("sol-fp8-fp16 is an explicit SM120 experiment")
    from cozy_runtime.internal import attention_fp8_sol_kernel as kernel

    tensors = tuple(t.contiguous() for t in (query, key, value))
    batch, tokens, heads, dim = query.shape
    blocks = -(-tokens // 64)
    scale = dim**-0.5 if scale is None else scale
    if dim != 128 or query.dtype != torch.bfloat16:
        raise ValueError("sol-fp8-fp16 requires BF16 D128 operands")
    codes = [torch.empty_like(t, dtype=torch.float8_e4m3fn) for t in tensors]
    scales = [
        torch.empty(batch, blocks, heads, device=query.device, dtype=torch.float32) for _ in tensors
    ]
    for tensor, code, factor in zip(tensors, codes, scales, strict=True):
        kernel.quantize_blocks[(blocks, heads, batch)](
            tensor, code, factor, tokens, heads, dim, num_warps=4
        )

    # Preserve the BF16 Sol statistics and threshold estimator. No previous-step cache.
    attention_sol._pin_preprocess()
    kc, vc, threshold = import_module("sol_attn.preprocess").prepare(
        *tensors, tau=1.0, scale=scale, thresh_type="exact"
    )
    output = torch.empty_like(query)
    kernel.forward_kernel[(1, blocks, batch * heads)](
        codes[0],
        tensors[0],
        codes[1],
        codes[2],
        *scales,
        kc,
        vc,
        threshold,
        output,
        scale,
        tokens,
        kc.shape[1],
        0,
        -(-sink_tokens // 64),
        HAS_SINK=sink_tokens > 0,
        H=heads,
        D=dim,
        NT=blocks,
        BV=128,
        BLOCK_SIZE=64,
        GROUP_SIZE=32,
        num_warps=4,
        num_stages=1,
    )
    return output


def forward(query: Any, key: Any, value: Any, **options: Any) -> Any:
    return attention_sol._attention(
        query, key, value, **options, sparse=sparse, dense=attention_kitchen_sol.sage_dense
    )


FP8_SOL = Local(
    forward,
    lambda *_: "exact=e4m3/f16_tile,route_tail=bf16/f32,state=f32,amax=16",
)


def fp4_dense(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    return attention_ulysses._call(
        "_cozy_sage3_fp4_global",
        attention_upstream.SAGE3_GLOBAL,
        query,
        key,
        value,
        Options(attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False),
    )


def sage_prefix(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    return attention_ulysses._call(
        "_cozy_sage_int8_fp8",
        attention_upstream.SAGE,
        query,
        key,
        value,
        Options(attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False),
    )


def forward_fp4_dense(query: Any, key: Any, value: Any, **options: Any) -> Any:
    """Materialized QKV; NVFP4 dense policy calls, unchanged FP8 sparse tiles, Sage2 prefix."""
    return attention_sol._attention(
        query, key, value, **options, sparse=sparse, dense=fp4_dense, prefix_dense=sage_prefix
    )


FP8_SOL_FP4_DENSE = Local(
    forward_fp4_dense,
    lambda *_: (
        "exact=e4m3/f16_tile,route_tail=bf16/f32,state=f32,amax=16; "
        "dense=nvfp4_global_head_mean; prefix=sageattention"
    ),
)
