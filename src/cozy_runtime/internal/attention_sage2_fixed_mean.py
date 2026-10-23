"""Fixed token reduction for Sage K means (Sage2's K, Sage3's Q and K): one compile per card,
whatever the length, so `warm` builds every variant off the request (`kernel_ready`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

import triton
import triton.language as tl

TOKEN_TILE, CHANNEL_TILE = 256, 32
#: The largest power-of-two tile count `warm` builds `finish_mean` for: 2048 x 256 tokens.
REDUCE_MAX = 2048


@triton.jit(  # type: ignore[untyped-decorator]
    do_not_specialize=["stride_t", "stride_h", "tokens", "tiles"],
    do_not_specialize_on_alignment=["stride_t", "stride_h"],
)
def partial_mean(  # type: ignore[no-untyped-def]
    X,
    P,
    stride_t,
    stride_h,
    tokens,
    tiles,
    D: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    channel_tile, tile = tl.program_id(1), tl.program_id(2)
    token = (tile * BT + tl.arange(0, BT)).to(tl.int64)
    channel = channel_tile * BC + tl.arange(0, BC)
    values = tl.load(
        X + token[:, None] * stride_t + head * stride_h + channel[None, :],
        mask=(token[:, None] < tokens) & (channel[None, :] < D),
        other=0,
    ).to(tl.float32)
    sums = tl.sum(values, axis=0)
    tl.store(P + (head * tiles + tile) * D + channel, sums, mask=channel < D)


@triton.jit(do_not_specialize=["tiles"])  # type: ignore[untyped-decorator]
def finish_mean(  # type: ignore[no-untyped-def]
    P,
    M,
    tiles,
    inverse,
    D: tl.constexpr,
    REDUCE: tl.constexpr,
    BC: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    channel_tile = tl.program_id(1)
    tile = tl.arange(0, REDUCE)
    channel = channel_tile * BC + tl.arange(0, BC)
    partial = tl.load(
        P + (head * tiles + tile[:, None]) * D + channel[None, :],
        mask=(tile[:, None] < tiles) & (channel[None, :] < D),
        other=0,
    )
    mean = tl.sum(partial, axis=0) * inverse
    tl.store(M + head * D + channel, mean.to(tl.bfloat16), mask=channel < D)


def fixed_mean(x: Tensor, denominator: int) -> Tensor:
    """BF16 [heads, D] sums of B1/NHD `x` over its tokens, in fixed tiles, over `denominator`.
    `1 / denominator` is f32 as a constant fold rounds it."""
    import torch

    _, tokens, heads, dim = x.shape
    tiles = triton.cdiv(tokens, TOKEN_TILE)
    partial = torch.empty((heads, tiles, dim), device=x.device, dtype=torch.float32)
    result = torch.empty((heads, dim), device=x.device, dtype=torch.bfloat16)
    channels = triton.cdiv(dim, CHANNEL_TILE)
    partial_mean[(heads, channels, tiles)](
        x, partial, x.stride(1), x.stride(2), tokens, tiles, dim, TOKEN_TILE, CHANNEL_TILE,
        num_warps=8, enable_fp_fusion=False,
    )  # fmt: skip
    finish_mean[(heads, channels)](
        partial, result, tiles, 1.0 / denominator, dim, triton.next_power_of_2(tiles),
        CHANNEL_TILE, num_warps=8, enable_fp_fusion=False,
    )  # fmt: skip
    return result


def warm(index: int, dim: int = 128) -> None:
    """Build both kernels on GPU `index` for every length up to `REDUCE_MAX` tiles, launching
    nothing."""
    import torch

    with torch.cuda.device(index):
        partial_mean.warmup(
            torch.bfloat16, torch.float32, 1, 1, 1, 1, dim, TOKEN_TILE, CHANNEL_TILE,
            grid=(1,), num_warps=8, enable_fp_fusion=False,
        )  # fmt: skip
        reduce = 1
        while reduce <= REDUCE_MAX:
            finish_mean.warmup(
                torch.float32, torch.bfloat16, 1, 1.0, dim, reduce, CHANNEL_TILE,
                grid=(1,), num_warps=8, enable_fp_fusion=False,
            )  # fmt: skip
            reduce *= 2


def mean(k: Tensor) -> Tensor:
    import torch

    if k.ndim != 4 or k.shape[0] != 1 or k.shape[-1] != 128:
        raise ValueError("Fixed K mean requires B1/NHD/D128")
    if k.dtype != torch.bfloat16 or k.device.type != "cuda" or k.stride(-1) != 1:
        raise ValueError("Fixed K mean requires CUDA BF16 K with contiguous channels")
    _, tokens, heads, dim = k.shape
    if min(tokens, heads) <= 0:
        raise ValueError("Nonempty K required")
    return fixed_mean(k, tokens).view(1, 1, heads, dim)
