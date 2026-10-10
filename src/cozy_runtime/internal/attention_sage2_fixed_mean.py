"""Fixed token reduction for Sage2 K means; imported only by the explicit SM90 path."""

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl

TOKEN_TILE, CHANNEL_TILE = 256, 32


@triton.jit(  # type: ignore[untyped-decorator]
    do_not_specialize=["stride_t", "stride_h"],
    do_not_specialize_on_alignment=["stride_t", "stride_h"],
)
def partial_mean(  # type: ignore[no-untyped-def]
    X,
    P,
    stride_t,
    stride_h,
    N: tl.constexpr,
    D: tl.constexpr,
    TILES: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    channel_tile, tile = tl.program_id(1), tl.program_id(2)
    token = (tile * BT + tl.arange(0, BT)).to(tl.int64)
    channel = channel_tile * BC + tl.arange(0, BC)
    values = tl.load(
        X + token[:, None] * stride_t + head * stride_h + channel[None, :],
        mask=(token[:, None] < N) & (channel[None, :] < D),
        other=0,
    ).to(tl.float32)
    sums = tl.sum(values, axis=0)
    tl.store(P + (head * TILES + tile) * D + channel, sums, mask=channel < D)


@triton.jit  # type: ignore[untyped-decorator]
def finish_mean(  # type: ignore[no-untyped-def]
    P,
    M,
    D: tl.constexpr,
    TILES: tl.constexpr,
    DENOMINATOR: tl.constexpr,
    REDUCE: tl.constexpr,
    BC: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    channel_tile = tl.program_id(1)
    tile = tl.arange(0, REDUCE)
    channel = channel_tile * BC + tl.arange(0, BC)
    partial = tl.load(
        P + (head * TILES + tile[:, None]) * D + channel[None, :],
        mask=(tile[:, None] < TILES) & (channel[None, :] < D),
        other=0,
    )
    mean = tl.sum(partial, axis=0) * (1.0 / DENOMINATOR)
    tl.store(M + head * D + channel, mean.to(tl.bfloat16), mask=channel < D)


def mean(k: Any) -> Any:
    import torch

    if k.ndim != 4 or k.shape[0] != 1 or k.shape[-1] != 128:
        raise ValueError("Fixed K mean requires B1/NHD/D128")
    if k.dtype != torch.bfloat16 or k.device.type != "cuda" or k.stride(-1) != 1:
        raise ValueError("Fixed K mean requires CUDA BF16 K with contiguous channels")
    _, tokens, heads, dim = k.shape
    if min(tokens, heads) <= 0:
        raise ValueError("Nonempty K required")
    tiles = triton.cdiv(tokens, TOKEN_TILE)
    partial = torch.empty((heads, tiles, dim), device=k.device, dtype=torch.float32)
    result = torch.empty((1, 1, heads, dim), device=k.device, dtype=torch.bfloat16)
    partial_mean[(heads, triton.cdiv(dim, CHANNEL_TILE), tiles)](
        k,
        partial,
        k.stride(1),
        k.stride(2),
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
        result,
        dim,
        tiles,
        tokens,
        triton.next_power_of_2(tiles),
        CHANNEL_TILE,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return result
