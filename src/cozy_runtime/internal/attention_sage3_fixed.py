"""Opt-in Sage3 fixed means/correction; no global policy or upstream monkeypatch.

Arithmetic is the GPU-qualified research candidate from Runtime PR1288/run5228.
Only input addressing expands to the actual H3 packed/strided QKV layouts.
"""

from __future__ import annotations

import functools
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from triton.runtime.jit import JITFunction

CENTER_TILE = 16


@functools.cache
def _kernels() -> tuple[JITFunction, JITFunction, JITFunction]:
    """The shared fixed means (`attention_sage2_fixed_mean`) and this candidate's centering: one
    compile per card, whatever the length (`warm`)."""
    import triton
    import triton.language as tl

    means = import_module("cozy_runtime.internal.attention_sage2_fixed_mean")

    @triton.jit(
        do_not_specialize=["qt", "qh", "kt", "kh", "vt", "vh", "tokens", "padded"],
        do_not_specialize_on_alignment=["qt", "qh", "kt", "kh", "vt", "vh"],
    )  # type: ignore[untyped-decorator]
    def center_and_correct(  # type: ignore[no-untyped-def]
        Q,
        K,
        V,
        QM,
        KM,
        QO,
        KO,
        VO,
        DELTA,
        qt,
        qh,
        kt,
        kh,
        vt,
        vh,
        tokens,
        padded,
        D: tl.constexpr,
        BT: tl.constexpr,
        FIXED_CORRECTION: tl.constexpr,
    ):
        head = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1)
        token = (tile * BT + tl.arange(0, BT)).to(tl.int64)
        channel = tl.arange(0, D)
        source_q = token[:, None] * qt + head * qh + channel[None, :]
        source_k = token[:, None] * kt + head * kh + channel[None, :]
        source_v = token[:, None] * vt + head * vh + channel[None, :]
        live = token[:, None] < tokens
        q = tl.load(Q + source_q, mask=live, other=0).to(tl.float32)
        k = tl.load(K + source_k, mask=live, other=0).to(tl.float32)
        v = tl.load(V + source_v, mask=live, other=0)
        qm = tl.load(QM + head * D + channel).to(tl.float32)
        km = tl.load(KM + head * D + channel).to(tl.float32)
        # The upstream means and centered tensors each round to BF16. K padding is
        # zero after centering; Q padding participates as original zero minus qm.
        qc = (q - qm[None, :]).to(tl.bfloat16)
        kc = tl.where(live, (k - km[None, :]).to(tl.bfloat16), 0.0).to(tl.bfloat16)
        destination = (head * padded + token[:, None]) * D + channel[None, :]
        tl.store(QO + destination, qc, mask=token[:, None] < padded)
        tl.store(KO + destination, kc, mask=token[:, None] < padded)
        tl.store(VO + destination, v, mask=token[:, None] < padded)
        # Fixed 128-channel FP32 products/tree; intentionally not claimed cuBLAS-bit
        # identical. Preserve the old BF16 correction rounding before FP32 storage.
        if FIXED_CORRECTION:
            correction = (
                tl.sum(qm[None, :] * kc.to(tl.float32), axis=1).to(tl.bfloat16).to(tl.float32)
            )
            tl.store(DELTA + head * padded + token, correction, mask=token < padded)

    return means.partial_mean, means.finish_mean, center_and_correct


def warm(index: int) -> None:
    """Build every kernel `fixed_preprocess` launches on GPU `index`, for every length,
    launching nothing (`kernel_ready`)."""
    import torch

    import_module("cozy_runtime.internal.attention_sage2_fixed_mean").warm(index)
    center_and_correct = _kernels()[2]
    bf16, f32 = torch.bfloat16, torch.float32
    with torch.cuda.device(index):
        for fixed in (False, True):
            center_and_correct.warmup(
                *[bf16] * 8, f32, *[1] * 8, 128, CENTER_TILE, fixed,
                grid=(1,), num_warps=4, enable_fp_fusion=False,
            )  # fmt: skip


def fixed_preprocess(
    q: Tensor, k: Tensor, v: Tensor, *, fixed_correction: bool = False
) -> dict[str, Tensor]:
    """B1/BF16/NHD/D128 to owned contiguous HND values and explicit BF16 means."""
    import torch
    import triton

    center_and_correct = _kernels()[2]

    if (
        q.shape != k.shape
        or q.shape != v.shape
        or q.ndim != 4
        or q.shape[0] != 1
        or q.shape[-1] != 128
    ):
        raise ValueError("Fixed preprocessing requires equal B1/NHD/D128 QKV")
    if any(
        x.dtype != torch.bfloat16 or x.device != q.device or x.stride(-1) != 1 for x in (q, k, v)
    ):
        raise ValueError(
            "Fixed preprocessing requires BF16 inputs with contiguous channels on one device"
        )
    if q.device.type != "cuda":
        raise ValueError("The fixed candidate is CUDA only")
    _, tokens, heads, dim = q.shape
    if tokens < 1 or heads < 1:
        raise ValueError("The input geometry must be nonempty")
    padded = triton.cdiv(tokens, 128) * 128
    fixed_mean = import_module("cozy_runtime.internal.attention_sage2_fixed_mean").fixed_mean
    qm, km = (fixed_mean(x, n).view(1, heads, 1, dim) for x, n in ((q, padded), (k, tokens)))
    outputs = [
        torch.empty((1, heads, padded, dim), device=q.device, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    delta = torch.empty(
        (1, heads, 1, padded) if fixed_correction else (1,), device=q.device, dtype=torch.float32
    )
    center_and_correct[(heads, triton.cdiv(padded, CENTER_TILE))](
        q,
        k,
        v,
        qm,
        km,
        *outputs,
        delta,
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        v.stride(1),
        v.stride(2),
        tokens,
        padded,
        dim,
        CENTER_TILE,
        fixed_correction,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if not fixed_correction:
        delta = torch.matmul(qm, outputs[1].transpose(-2, -1)).to(torch.float32).contiguous()
    return {"q": outputs[0], "k": outputs[1], "v": outputs[2], "qm": qm, "km": km, "delta": delta}


def forward(q: Tensor, k: Tensor, v: Tensor, *, fixed_correction: bool = False) -> Tensor:
    """Preserve the public quantizers/attention and short centered-buffer lifetimes."""
    api = import_module("sageattn3.api")
    tokens = q.shape[1]
    prepared = fixed_preprocess(q, k, v, fixed_correction=fixed_correction)
    del prepared["qm"], prepared["km"]
    qlist = api.scale_and_quant_fp4(prepared.pop("q"))
    klist = api.scale_and_quant_fp4_permute(prepared.pop("k"))
    vlist = api.scale_and_quant_fp4_transpose(prepared.pop("v"))
    output = api.blockscaled_fp4_attn(
        qlist, klist, vlist, prepared["delta"], tokens, False, False, True
    )[0]
    return output[:, :, :tokens, :].transpose(1, 2).contiguous()
