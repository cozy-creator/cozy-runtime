"""Fixed arithmetic trees; the head count changes addresses/grid, never reductions."""

from __future__ import annotations

import triton
import triton.language as tl

TOKEN_TILE = 256
CHANNEL_TILE = 32
CENTER_TILE = 16


@triton.jit(do_not_specialize=["stride_t"], do_not_specialize_on_alignment=["stride_t"])
def partial_mean(
    X,
    P,
    stride_t,
    N: tl.constexpr,
    D: tl.constexpr,
    TILES: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    head, channel_tile, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    token = tile * BT + tl.arange(0, BT)
    channel = channel_tile * BC + tl.arange(0, BC)
    values = tl.load(
        X + token[:, None] * stride_t + head * D + channel[None, :],
        mask=(token[:, None] < N) & (channel[None, :] < D),
        other=0,
    ).to(tl.float32)
    sums = tl.sum(values, axis=0)
    tl.store(P + (head * TILES + tile) * D + channel, sums, mask=channel < D)


@triton.jit
def finish_mean(
    P,
    M,
    D: tl.constexpr,
    TILES: tl.constexpr,
    DENOMINATOR: tl.constexpr,
    REDUCE: tl.constexpr,
    BC: tl.constexpr,
):
    head, channel_tile = tl.program_id(0), tl.program_id(1)
    tile = tl.arange(0, REDUCE)
    channel = channel_tile * BC + tl.arange(0, BC)
    partial = tl.load(
        P + (head * TILES + tile[:, None]) * D + channel[None, :],
        mask=(tile[:, None] < TILES) & (channel[None, :] < D),
        other=0,
    )
    mean = tl.sum(partial, axis=0) * (1.0 / DENOMINATOR)
    tl.store(M + head * D + channel, mean.to(tl.bfloat16), mask=channel < D)


@triton.jit(do_not_specialize=["stride_t"], do_not_specialize_on_alignment=["stride_t"])
def center_and_correct(
    Q,
    K,
    V,
    QM,
    KM,
    QO,
    KO,
    VO,
    DELTA,
    stride_t,
    N: tl.constexpr,
    PADDED: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    head, tile = tl.program_id(0), tl.program_id(1)
    token = tile * BT + tl.arange(0, BT)
    channel = tl.arange(0, D)
    source = token[:, None] * stride_t + head * D + channel[None, :]
    live = token[:, None] < N
    q = tl.load(Q + source, mask=live, other=0).to(tl.float32)
    k = tl.load(K + source, mask=live, other=0).to(tl.float32)
    v = tl.load(V + source, mask=live, other=0)
    qm = tl.load(QM + head * D + channel).to(tl.float32)
    km = tl.load(KM + head * D + channel).to(tl.float32)
    # The upstream means and centered tensors each round to BF16. K padding is
    # zero after centering; Q padding participates as original zero minus qm.
    qc = (q - qm[None, :]).to(tl.bfloat16)
    kc = tl.where(live, (k - km[None, :]).to(tl.bfloat16), 0.0).to(tl.bfloat16)
    destination = (head * PADDED + token[:, None]) * D + channel[None, :]
    tl.store(QO + destination, qc, mask=token[:, None] < PADDED)
    tl.store(KO + destination, kc, mask=token[:, None] < PADDED)
    tl.store(VO + destination, v, mask=token[:, None] < PADDED)
    # Fixed 128-channel FP32 products/tree; intentionally not claimed cuBLAS-bit
    # identical. Preserve the old BF16 correction rounding before FP32 storage.
    correction = tl.sum(qm[None, :] * kc.to(tl.float32), axis=1).to(tl.bfloat16).to(tl.float32)
    tl.store(DELTA + head * PADDED + token, correction, mask=token < PADDED)


def fixed_preprocess(q, k, v):
    """B1/BF16/NHD/D128 to owned contiguous HND values and explicit BF16 means."""
    import torch

    if (
        q.shape != k.shape
        or q.shape != v.shape
        or q.ndim != 4
        or q.shape[0] != 1
        or q.shape[-1] != 128
    ):
        raise ValueError("Fixed preprocessing requires equal B1/NHD/D128 QKV")
    if any(
        x.dtype != torch.bfloat16 or x.device != q.device or not x.is_contiguous()
        for x in (q, k, v)
    ):
        raise ValueError("Fixed preprocessing requires fresh contiguous BF16 inputs on one device")
    if q.device.type != "cuda":
        raise ValueError("The fixed candidate is CUDA only")
    _, tokens, heads, dim = q.shape
    if tokens < 1 or heads < 1:
        raise ValueError("The input geometry must be nonempty")
    padded = triton.cdiv(tokens, 128) * 128
    tiles = triton.cdiv(tokens, TOKEN_TILE)
    means = []
    for value, denominator in ((q, padded), (k, tokens)):
        partial = torch.empty((heads, tiles, dim), device=q.device, dtype=torch.float32)
        mean = torch.empty((1, heads, 1, dim), device=q.device, dtype=torch.bfloat16)
        partial_mean[(heads, triton.cdiv(dim, CHANNEL_TILE), tiles)](
            value,
            partial,
            value.stride(1),
            tokens,
            dim,
            tiles,
            TOKEN_TILE,
            CHANNEL_TILE,
            num_warps=8,
            enable_fp_fusion=False,
        )
        finish_mean[(heads, triton.cdiv(dim, CHANNEL_TILE))](
            partial,
            mean,
            dim,
            tiles,
            denominator,
            triton.next_power_of_2(tiles),
            CHANNEL_TILE,
            num_warps=8,
            enable_fp_fusion=False,
        )
        means.append(mean)
        del partial
    qm, km = means
    outputs = [
        torch.empty((1, heads, padded, dim), device=q.device, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    delta = torch.empty((1, heads, 1, padded), device=q.device, dtype=torch.float32)
    center_and_correct[(heads, triton.cdiv(padded, CENTER_TILE))](
        q,
        k,
        v,
        qm,
        km,
        *outputs,
        delta,
        q.stride(1),
        tokens,
        padded,
        dim,
        CENTER_TILE,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return {"q": outputs[0], "k": outputs[1], "v": outputs[2], "qm": qm, "km": km, "delta": delta}
