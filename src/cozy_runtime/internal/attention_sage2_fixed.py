"""Opt-in Hopper Sage2 with a fixed K-mean tree, including BF16 Sol's dense prefix.

The selected upstream mode stays INT8 Q/K, FP8 P/V and buffered FP32 accumulation.
Only K's token reduction changes; this is not a promise of unchanged rounding versus
upstream's full-head mean. The same callable serves dense steps and protected query rows.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as TorchDevice

from cozy_runtime.internal import attention_sol, attention_ulysses, attention_upstream
from cozy_runtime.internal.attention_ulysses import Local, Options

DENSE_NAME = "sageattention-fixed-kmean"
SOL_NAME = "sol-attn-sage2-fixed-kmean"
IMPLEMENTATION = "sageattn_qk_int8_pv_fp8_cuda_sm90(per_thread, fp32+fp32, fixed_kmean)"


def _shape(query: Tensor, key: Tensor, value: Tensor) -> None:
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError("Fixed-Kmean Sage2 requires B/N/H/D tensors")
    if (
        query.shape[0] != 1
        or key.shape != value.shape
        or key.shape[0] != 1
        or query.shape[2:] != key.shape[2:]
        or query.shape[-1] != 128
        or min(query.shape[1], key.shape[1], query.shape[2]) <= 0
    ):
        raise ValueError("Fixed-Kmean Sage2 requires B1, equal Q/KV heads, D128 and nonempty rows")


def validate_device(device: TorchDevice) -> None:
    torch = import_module("torch")
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("Fixed-Kmean Sage2 serves Hopper SM90 only")


def forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attn_mask: object = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    import torch

    attention_upstream._options(DENSE_NAME, attn_mask, dropout_p, scale)
    if is_causal or enable_gqa:
        raise ValueError("Fixed-Kmean Sage2 serves noncausal equal-head inference only")
    attention_upstream._cuda_inputs(
        query, key, value, enable_gqa, name=DENSE_NAME, admitted=((9, 0),), dims={128}, bf16=True
    )
    _shape(query, key, value)
    if any(t.stride(-1) != 1 for t in (query, key, value)):
        raise ValueError("Fixed-Kmean Sage2 requires contiguous channels")
    core = import_module("sageattention.core")
    if not core.SM90_ENABLED:
        raise RuntimeError("SageAttention2 SM90 native implementation is absent")
    from cozy_runtime.internal.attention_sage2_fixed_mean import mean

    torch.cuda.set_device(value.device)
    km = mean(key)
    qi, qs, ki, ks = core.per_thread_int8_triton(
        query, key, km, tensor_layout="NHD", BLKQ=64, WARPQ=16, BLKK=128, WARPK=128
    )
    output = torch.empty(query.size(), dtype=query.dtype, device=query.device)
    pad = (-key.shape[1]) % 128
    if pad:
        value = torch.cat(
            [
                value,
                torch.zeros(1, pad, value.shape[2], 128, dtype=value.dtype, device=value.device),
            ],
            dim=1,
        )
    vf, vs, _ = core.per_channel_fp8(value, tensor_layout="NHD", smooth_v=False)
    core.sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
        qi,
        ki,
        vf,
        output,
        qs,
        ks,
        vs,
        0,
        0,
        3,
        scale if scale is not None else 128**-0.5,
        0,
    )
    return attention_upstream._output(output, query)


DENSE = Local(forward, lambda *_: IMPLEMENTATION)


def dense(*, query: Tensor, key: Tensor, value: Tensor, scale: float | None) -> Tensor:
    return attention_ulysses._call(
        "_cozy_sage2_fixed_kmean",
        DENSE,
        query,
        key,
        value,
        Options(attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale, enable_gqa=False),
    )


def sparse_forward(query: Tensor, key: Tensor, value: Tensor, **options: Any) -> Tensor:
    # No replacement sparse math: the existing CuTe kernel and model layout stay authoritative.
    return attention_sol._attention(
        query,
        key,
        value,
        **options,
        dense=dense,
        prefix_dense=dense,
        validate_device=validate_device,
    )


SOL = Local(
    sparse_forward,
    lambda q, *_: f"dense={IMPLEMENTATION}; sparse={attention_sol.backend(q.device)}",
)
