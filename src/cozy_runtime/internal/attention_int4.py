"""INT4 Q·Kᵀ attention on Ampere and Ada (`_kernels._attn4`): Kitchen INT8 attention with Q and K
in int4 (each row Hadamard-turned, K less its mean, per-thread scales) and the rest Kitchen's own
INT8 path, so against `kitchen-int8` only Q·Kᵀ's precision differs. Allocated with torch, launched
on its current stream, over Diffusers' [batch, sequence, heads, 128] without a transpose."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cozy_runtime import _kernels
from cozy_runtime.internal import attention_upstream as upstream
from cozy_runtime.internal.attention_ulysses import Local

if TYPE_CHECKING:
    from torch import Tensor

#: Kitchen's key tile: 128 keys past 1024 (its own rule), else 64
LONG_KEYS = 1024


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    import torch

    upstream._options("kitchen-int4qk", dropout_p, scale)
    upstream._cuda_inputs(query, key, value, enable_gqa)
    if attn_mask is not None or is_causal:
        raise ValueError("kitchen-int4qk serves no mask and no causal attention")
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (query, key, value))
    batch, len_q, heads, dim = q.shape
    len_k, heads_k = k.shape[1], k.shape[2]
    cta = 128 if len_k > LONG_KEYS else 64
    blocks = -(-len_k // cta)
    device, f32, i8 = q.device, torch.float32, torch.int8
    out = torch.empty_like(q)
    q4 = torch.empty(batch, heads, len_q, dim // 2, dtype=i8, device=device)
    k4 = torch.empty(batch, heads_k, len_k, dim // 2, dtype=i8, device=device)
    v8 = torch.empty(batch * heads_k * dim, blocks * cta, dtype=i8, device=device)
    q_scale = torch.empty(batch, heads, -(-len_q // 128) * 32, dtype=f32, device=device)
    k_scale = torch.empty(batch, heads_k, blocks * 4, dtype=f32, device=device)
    k_sum = torch.empty(batch, heads_k, dim, dtype=f32, device=device)
    v_scale = torch.empty(batch * heads_k * dim, dtype=f32, device=device)

    def strides(x: Tensor) -> tuple[int, int, int]:
        return (x.stride(0), x.stride(2), x.stride(1))

    code = _kernels._attn4.attention(
        device.index,
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        out.data_ptr(),
        q4.data_ptr(),
        q_scale.data_ptr(),
        k4.data_ptr(),
        k_scale.data_ptr(),
        k_sum.data_ptr(),
        v8.data_ptr(),
        v_scale.data_ptr(),
        batch,
        heads,
        heads_k,
        len_q,
        len_k,
        cta,
        q.dtype == torch.bfloat16,
        dim**-0.5 if scale is None else float(scale),
        strides(q),
        strides(k),
        strides(v),
        strides(out),
        torch.cuda.current_stream(device).cuda_stream,
    )
    if code:
        raise RuntimeError(f"kitchen-int4qk: {_kernels._attn4.error_name(code)}")
    return out


KITCHEN_INT4QK = Local(attention, upstream._named("cozy _attn4 (int4 Q·Kᵀ, Kitchen int8 P·V)"))
