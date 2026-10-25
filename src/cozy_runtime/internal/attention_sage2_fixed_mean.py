"""Fixed token reduction for Sage K means (Sage2's K, Sage3's Q and K): one compile per card,
whatever the length (`ENTRIES`), on the machine's compile manager (`triton_kernels`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

import triton
import triton.language as tl

from cozy_runtime.author.kernels import Launch, TritonEntry, TritonKernels

TOKEN_TILE, CHANNEL_TILE = 256, 32
#: The largest power-of-two tile count `finish_mean` is built for: 2048 x 256 tokens.
REDUCE_MAX = 2048
DIM = 128


@triton.jit  # type: ignore[untyped-decorator]
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


@triton.jit  # type: ignore[untyped-decorator]
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


#: Every specialization `fixed_mean` launches: one `partial_mean`, `finish_mean` per
#: power-of-two tile count. Sizes, strides and the divisor are runtime values and the input may
#: be any view (only the outputs are assumed aligned); nothing is fused, so each product and sum
#: rounds as written.
ENTRIES = (
    TritonEntry(
        partial_mean,
        {"X": "*bf16", "P": "*fp32", "stride_t": "i32", "stride_h": "i32", "tokens": "i32",
         "tiles": "i32", "D": "constexpr", "BT": "constexpr", "BC": "constexpr"},
        {"D": DIM, "BT": TOKEN_TILE, "BC": CHANNEL_TILE},
        ("P",),
        num_warps=8,
        fp_fusion=False,
    ),
    *(
        TritonEntry(
            finish_mean,
            {"P": "*fp32", "M": "*bf16", "tiles": "i32", "inverse": "fp32", "D": "constexpr",
             "REDUCE": "constexpr", "BC": "constexpr"},
            {"D": DIM, "REDUCE": 1 << shift, "BC": CHANNEL_TILE},
            ("P", "M"),
            num_warps=8,
            fp_fusion=False,
        )
        for shift in range(REDUCE_MAX.bit_length())
    ),
)  # fmt: skip
KERNELS = TritonKernels("cozy-sage2-fixed-kmean", ENTRIES)
#: Device index -> its loaded launcher (`serve`).
_LAUNCH: dict[int, Launch] = {}


def serve(index: int, launch: Launch) -> None:
    _LAUNCH[index] = launch


def fixed_mean(launch: Launch, x: Tensor, denominator: int) -> Tensor:
    """BF16 [heads, D] sums of B1/NHD `x` over its tokens, in fixed tiles, over `denominator`.
    `1 / denominator` is f32 as a constant fold rounds it."""
    import torch

    _, tokens, heads, dim = x.shape
    tiles = triton.cdiv(tokens, TOKEN_TILE)
    partial = torch.empty((heads, tiles, dim), device=x.device, dtype=torch.float32)
    result = torch.empty((heads, dim), device=x.device, dtype=torch.bfloat16)
    channels = triton.cdiv(dim, CHANNEL_TILE)
    launch(
        partial_mean, (heads, channels, tiles),
        x, partial, x.stride(1), x.stride(2), tokens, tiles, dim, TOKEN_TILE, CHANNEL_TILE,
    )  # fmt: skip
    launch(
        finish_mean, (heads, channels, 1),
        partial, result, tiles, 1.0 / denominator, dim, triton.next_power_of_2(tiles),
        CHANNEL_TILE,
    )  # fmt: skip
    return result


def mean(k: Tensor) -> Tensor:
    """Sage2's fixed K mean, on the launcher its kernel opened (`serve`)."""
    _, tokens, heads, dim = k.shape
    return fixed_mean(_LAUNCH[k.device.index or 0], k, tokens).view(1, 1, heads, dim)
