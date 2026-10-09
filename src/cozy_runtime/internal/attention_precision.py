"""Explicit SM120 precision controls; dense calls never invoke Sol routing.

FP8 uses E4M3 operands bounded by16 and FP16 tile MMA, widened into FP32
online-softmax state. BF16 uses Torch's Flash SDPA and upstream BF16 Sol.
These controls are request-only experiments, not automatic preferences.
"""

from __future__ import annotations

import math
from typing import Any

from cozy_runtime.internal import attention_fp8_sol, attention_sol
from cozy_runtime.internal.attention_ulysses import Local


def validate(query: Any, key: Any, value: Any, options: dict[str, Any]) -> None:
    import torch

    if (
        options.get("attn_mask") is not None
        or options.get("is_causal", False)
        or options.get("dropout_p", 0.0) != 0
        or options.get("enable_gqa", False)
    ):
        raise ValueError("precision controls require unmasked noncausal inference without GQA")
    tensors = (query, key, value)
    if any(not isinstance(t, torch.Tensor) or t.ndim != 4 for t in tensors):
        raise ValueError("precision controls require BTHD tensors")
    if (
        key.shape != value.shape
        or query.shape[0] != key.shape[0]
        or query.shape[2:] != key.shape[2:]
    ):
        raise ValueError("precision controls require matching batch, heads and D128")
    if query.shape[-1] != 128 or min(query.shape[:3]) < 1 or key.shape[1] < 1:
        raise ValueError("precision controls require nonempty D128 tensors")
    if any(
        t.dtype != torch.bfloat16 or t.device != query.device or t.requires_grad for t in tensors
    ):
        raise ValueError("precision controls require BF16 inference operands on one device")
    if query.device.type != "cuda" or torch.cuda.get_device_capability(query.device) != (12, 0):
        raise ValueError("precision controls are explicit SM120 experiments")
    scale = options.get("scale")
    if scale is not None and not math.isfinite(scale):
        raise ValueError("attention scale must be finite")


def fp8_dense(query: Any, key: Any, value: Any, **options: Any) -> Any:
    """All Q/KV blocks, including unequal Q/KV lengths; no routing or approximation."""
    import torch

    validate(query, key, value, options)
    from cozy_runtime.internal import attention_fp8_sol_kernel as kernel

    tensors = tuple(t.contiguous() for t in (query, key, value))
    batch, q_tokens, heads, dim = query.shape
    kv_tokens = key.shape[1]
    codes = [torch.empty_like(t, dtype=torch.float8_e4m3fn) for t in tensors]
    scales = [
        torch.empty(batch, -(-t.shape[1] // 64), heads, device=query.device, dtype=torch.float32)
        for t in tensors
    ]
    for tensor, code, factor in zip(tensors, codes, scales, strict=True):
        tokens = tensor.shape[1]
        kernel.quantize_blocks[(-(-tokens // 64), heads, batch)](
            tensor, code, factor, tokens, heads, dim, num_warps=4
        )
    output = torch.empty_like(tensors[0])
    scale = options.get("scale")
    kernel.dense_forward_kernel[(-(-q_tokens // 64), batch * heads)](
        *codes,
        *scales,
        output,
        dim**-0.5 if scale is None else scale,
        q_tokens,
        kv_tokens,
        H=heads,
        D=dim,
        num_warps=4,
        num_stages=1,
    )
    return output


def bf16_dense(query: Any, key: Any, value: Any, **options: Any) -> Any:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    validate(query, key, value, options)
    # Restrict the backend; never silently choose cuDNN, math or a quantized kernel.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=options.get("scale"),
        ).transpose(1, 2)


def fp8_mixed(query: Any, key: Any, value: Any, **options: Any) -> Any:
    return attention_sol._attention(
        query, key, value, **options, sparse=attention_fp8_sol.sparse, dense=fp8_dense
    )


def bf16_mixed(query: Any, key: Any, value: Any, **options: Any) -> Any:
    return attention_sol._attention(query, key, value, **options, dense=bf16_dense)


FP8_DENSE = Local(fp8_dense, lambda *_: "dense=e4m3/f16_tile,state=f32,amax=16,no_router")
FP8_MIXED = Local(
    fp8_mixed,
    lambda *_: "dense_exact=e4m3/f16_tile,route_tail=bf16/f32,state=f32,amax=16",
)
BF16_DENSE = Local(bf16_dense, lambda *_: "torch.flash_attention(bf16/f32),no_router")
BF16_MIXED = Local(
    bf16_mixed, lambda *_: "dense=torch.flash_attention(bf16/f32);sparse=sol(bf16/f32)"
)
NAMES = {
    "fp8-fp16-dense": FP8_DENSE,
    "sol-fp8-fp16-pure": FP8_MIXED,
    "bf16-fp32-dense": BF16_DENSE,
    "sol-bf16-fp32": BF16_MIXED,
}
