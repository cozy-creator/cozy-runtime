"""Opt-in Hopper Sage2 with a fixed K-mean tree, including BF16 Sol's dense prefix.

The selected upstream mode stays INT8 Q/K, FP8 P/V and buffered FP32 accumulation.
Only K's token reduction changes; this is not a promise of unchanged rounding versus
upstream's full-head mean. The same callable serves dense steps and protected query rows.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from cozy_runtime.internal import attention_sol, attention_upstream
from cozy_runtime.internal.attention_ulysses import Local

DENSE_NAME = "sageattention-fixed-kmean"
SOL_NAME = "sol-attn-sage2-fixed-kmean"
IMPLEMENTATION = "sageattn_qk_int8_pv_fp8_cuda_sm90(per_thread, fp32+fp32, fixed_kmean)"


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

    attention_upstream._options(DENSE_NAME, dropout_p, scale)
    attention_upstream._cuda_inputs(query, key, value, enable_gqa)
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
#: Sol's CuTe sparse rows; its dense rows are its row's declared kernel (`attention.Candidate`).
SOL = Local(
    attention_sol._attention,
    lambda q, *_: f"dense={IMPLEMENTATION}; sparse={attention_sol.backend(q.device)}",
)
