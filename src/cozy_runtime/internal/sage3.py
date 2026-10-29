"""SageAttention3's Python API (thu-ml/SageAttention d1a57a5,
`sageattention3_blackwell/sageattn3/api.py`, Apache-2.0) over the Runtime wheel's `_kernels._sage3`:
upstream's allocations, order and arithmetic, its `fp4attn_cuda` and `fp4quant_cuda` calls made with
raw pointers on PyTorch's current stream, as theirs launch. Its Triton group mean is a Runtime set
(`KERNELS`, built at machine start) launched on the launcher its kernel opened (`serve`); the
kernel carries one constexpr more than upstream's (`BF16`), which only names its dtype's entry.

`attention_upstream` serves this module or upstream's own `sageattn3.api` (the source build), the
same functions either way (`bind_sage3`).
"""

# Copyright (c) 2025 by SageAttention team (the functions below, from upstream's `api.py`).
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention as sdpa

from cozy_runtime import _kernels
from cozy_runtime.author.kernels import Launch, TritonEntry, TritonKernels

GROUP_SIZE = 128


@triton.jit  # type: ignore[untyped-decorator]
def group_mean_kernel(  # type: ignore[no-untyped-def]
    q_ptr,
    q_out_ptr,
    qm_out_ptr,
    B,
    H,
    L,
    D: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_ql,
    stride_qd,
    stride_qmb,
    stride_qmh,
    stride_qml,
    stride_qmd,
    GROUP_SIZE: tl.constexpr,
    BF16: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)

    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)

    q_offsets = (
        pid_b * stride_qb
        + pid_h * stride_qh
        + offsets[:, None] * stride_ql
        + tl.arange(0, D)[None, :] * stride_qd
    )
    q_group = tl.load(q_ptr + q_offsets)

    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE

    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    qm_offset = (
        pid_b * stride_qmb
        + pid_h * stride_qmh
        + pid_group * stride_qml
        + tl.arange(0, D) * stride_qmd
    )
    tl.store(qm_out_ptr + qm_offset, qm_group)


KERNELS = TritonKernels(
    "cozy-sage3-group-mean",
    [
        TritonEntry(
            group_mean_kernel,
            {**dict.fromkeys(("q_ptr", "q_out_ptr", "qm_out_ptr"), f"*{dtype}"),
             **dict.fromkeys(("B", "H", "L"), "i32"),
             **dict.fromkeys(("stride_qb", "stride_qh", "stride_ql", "stride_qd", "stride_qmb",
                              "stride_qmh", "stride_qml", "stride_qmd"), "i32"),
             "D": "constexpr", "GROUP_SIZE": "constexpr", "BF16": "constexpr"},
            {"D": dim, "GROUP_SIZE": GROUP_SIZE, "BF16": int(dtype == "bf16")},
        )
        for dtype in ("bf16", "fp16")
        for dim in (64, 128)
    ],
)  # fmt: skip
#: Device index -> its loaded launcher (`serve`).
_LAUNCH: dict[int, Launch] = {}


def serve(index: int, launch: Launch) -> None:
    _LAUNCH[index] = launch


def triton_group_mean(q: Tensor) -> tuple[Tensor, Tensor]:
    B, H, L, D = q.shape
    num_groups = L // GROUP_SIZE

    q_out = torch.empty_like(q)  # [B, H, L, D]
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype)

    grid = (B, H, num_groups)
    _LAUNCH[q.device.index or 0](
        group_mean_kernel,
        grid,
        q,
        q_out,
        qm,
        B,
        H,
        L,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        qm.stride(0),
        qm.stride(1),
        qm.stride(2),
        qm.stride(3),
        GROUP_SIZE,
        int(q.dtype == torch.bfloat16),
    )
    return q_out, qm


def preprocess_qkv(
    q: Tensor, k: Tensor, v: Tensor, per_block_mean: bool = True
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    def pad_128(x: Tensor) -> Tensor:
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0).contiguous()

    k -= k.mean(dim=-2, keepdim=True)
    q, k, v = map(lambda x: pad_128(x), [q, k, v])
    if per_block_mean:
        q, qm = triton_group_mean(q)
    else:
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
    delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
    return q, k, v, delta_s


def _strides(t: Tensor) -> tuple[int, int, int]:
    """(batch, head, row) of a (B, H, rows, cols) tensor, as upstream's kernels take them."""
    return int(t.stride(0)), int(t.stride(1)), int(t.stride(2))


def _stream(t: Tensor) -> int:
    return int(torch.cuda.current_stream(t.device).cuda_stream)


def _checked(code: int, what: str) -> None:
    if code:
        raise ValueError(f"sageattention3 prebuilt {what}: {_kernels._sage3.error_name(code)}")


def _quant(kind: int, x: Tensor, packed_fp4: Tensor, fp8_scale: Tensor) -> None:
    """`fp4quant_cuda`'s checks, then its kernel at tensor layout 1."""
    if not (x.is_cuda and packed_fp4.is_cuda and fp8_scale.is_cuda):
        raise ValueError("sageattention3 quantization: tensors must be on CUDA")
    if any(t.stride(-1) != 1 for t in (x, packed_fp4, fp8_scale)):
        raise ValueError("sageattention3 quantization: the last dimension must be contiguous")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"sageattention3 quantization: no {x.dtype} input")
    B, H, N, D = x.shape
    code = _kernels._sage3.quant(
        kind, x.device.index or 0, x.data_ptr(), packed_fp4.data_ptr(), fp8_scale.data_ptr(),
        x.dtype == torch.bfloat16, B, H, N, D, _strides(x), _strides(packed_fp4),
        _strides(fp8_scale), _stream(x),
    )  # fmt: skip
    _checked(code, "quantization")


def scale_and_quant_fp4(x: Tensor) -> tuple[Tensor, Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    _quant(0, x, packed_fp4, fp8_scale)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_permute(x: Tensor) -> tuple[Tensor, Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    _quant(1, x, packed_fp4, fp8_scale)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_transpose(x: Tensor) -> tuple[Tensor, Tensor]:
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    _quant(2, x, packed_fp4, fp8_scale)
    return packed_fp4, fp8_scale


def _fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sfq: Tensor,
    sfk: Tensor,
    sfv: Tensor,
    delta_s: Tensor,
    unpadded_k: int,
    softmax_scale: float,
    is_causal: bool,
    per_block_mean: bool,
    is_bf16: bool,
) -> list[Tensor]:
    """`fp4attn_cuda.fwd` (`mha_fwd`): its checks and outputs, then its kernel."""
    if not (q.dtype == k.dtype == v.dtype == torch.uint8):
        raise ValueError("sageattention3: packed Q, K and V must be uint8")
    if not (sfq.dtype == sfk.dtype == sfv.dtype == torch.float8_e4m3fn):
        raise ValueError("sageattention3: scales must be float8_e4m3fn")
    if any(t.stride(-1) != 1 for t in (q, k, v, delta_s)) or not all(
        t.is_contiguous() for t in (q, k, v)
    ):
        raise ValueError("sageattention3: Q, K and V must be contiguous")
    batch, heads, seqlen_q, packed = q.shape
    head_dim, seqlen_k, heads_k = packed * 2, k.size(2), k.size(1)
    if heads != heads_k:
        raise ValueError("sageattention3: no MQA/GQA")
    if tuple(k.shape) != (batch, heads_k, seqlen_k, packed) or tuple(v.shape) != (
        batch, heads_k, head_dim, seqlen_k // 2
    ):  # fmt: skip
        raise ValueError("sageattention3: K or V is not Q's shape")
    dtype = torch.bfloat16 if is_bf16 else torch.float16
    out = torch.empty((batch, heads, seqlen_q, head_dim), device=q.device, dtype=dtype)
    softmax_lse = torch.empty((batch, heads, seqlen_q), device=q.device, dtype=torch.float32)
    semaphore = (
        torch.full((1,), 132, device=q.device, dtype=torch.int32)
        if is_causal
        else torch.empty((1,), device=q.device, dtype=torch.int32)
    )
    if seqlen_k == 0:
        out.zero_()
        softmax_lse.fill_(float("inf"))
        return [out, softmax_lse]
    code = _kernels._sage3.fwd(
        q.device.index or 0, q.data_ptr(), k.data_ptr(), v.data_ptr(), sfq.data_ptr(),
        sfk.data_ptr(), sfv.data_ptr(), delta_s.data_ptr(), out.data_ptr(),
        softmax_lse.data_ptr(), semaphore.data_ptr(), batch, heads, heads_k, seqlen_q, seqlen_k,
        int(unpadded_k), head_dim, _strides(q), _strides(k), _strides(v), _strides(sfq),
        _strides(sfk), _strides(sfv), _strides(delta_s), _strides(out), float(softmax_scale),
        bool(is_causal), bool(per_block_mean), bool(is_bf16), _stream(q),
    )  # fmt: skip
    _checked(code, "attention")
    return [out, softmax_lse]


def blockscaled_fp4_attn(
    qlist: tuple[Tensor, Tensor],
    klist: tuple[Tensor, Tensor],
    vlist: tuple[Tensor, Tensor],
    delta_s: Tensor,
    KL: int,
    is_causal: bool = False,
    per_block_mean: bool = True,
    is_bf16: bool = True,
) -> list[Tensor]:
    softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)
    return _fwd(
        qlist[0], klist[0], vlist[0], qlist[1], klist[1], vlist[1], delta_s, KL,
        softmax_scale, is_causal, per_block_mean, is_bf16,
    )  # fmt: skip


def sageattn3_blackwell(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    is_causal: bool = False,
    per_block_mean: bool = True,
    **kwargs: object,
) -> Tensor:
    if q.size(-1) >= 256:
        return sdpa(q, k, v, is_causal=is_causal)
    QL = q.size(2)
    KL = k.size(2)
    is_bf16 = q.dtype == torch.bfloat16
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    qlist_from_cuda = scale_and_quant_fp4(q)
    klist_from_cuda = scale_and_quant_fp4_permute(k)
    vlist_from_cuda = scale_and_quant_fp4_transpose(v)
    o_fp4 = blockscaled_fp4_attn(
        qlist_from_cuda,
        klist_from_cuda,
        vlist_from_cuda,
        delta_s,
        KL,
        is_causal,
        per_block_mean,
        is_bf16,
    )[0][:, :, :QL, :].contiguous()
    return o_fp4
