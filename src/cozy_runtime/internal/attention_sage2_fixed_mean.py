"""Fixed token reduction for Sage K means (Sage2's K, Sage3's Q and K): one compile per card,
whatever the length (`ENTRIES`), on the machine's compile manager (`triton_kernels`). Sage2's set
also merges a sparse kernel's rows with its protected keys' rows by log-sum-exp (`merge`)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

from cozy_runtime.accel.kernels import Launch, TritonEntry, TritonKernels

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


@triton.jit  # type: ignore[untyped-decorator]
def merge_keys(  # type: ignore[no-untyped-def]
    ROWS,
    LSE,
    KEYED,
    RAW,
    Q,
    KM,
    rows,
    heads,
    stride_rows,
    stride_keyed,
    stride_q,
    scale,
    D: tl.constexpr,
    BR: tl.constexpr,
):
    """BR rows of one head: ROWS (log-sum-exp LSE, natural, [rows, heads]) and KEYED, the same
    queries over other keys from Sage2 (its kernel's RAW log2-sum-exp, [heads, rows], over keys
    smoothed by KM: `scale * q . km` restores it), merged in place in ROWS."""
    tile = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    row = (tile * BR + tl.arange(0, BR)).to(tl.int64)
    live = row < rows
    dims = tl.arange(0, D)
    q = tl.load(
        Q + row[:, None] * stride_q + head * D + dims[None, :], mask=live[:, None], other=0.0
    )
    km = tl.load(KM + head * D + dims)
    shift = tl.sum(q.to(tl.float32) * km.to(tl.float32)[None, :], axis=1) * scale
    lse_keyed = tl.load(RAW + head * rows + row, mask=live, other=0.0) / 1.44269504 + shift
    lse_rows = tl.load(LSE + row * heads + head, mask=live, other=0.0)
    top = tl.maximum(lse_rows, lse_keyed)
    w_rows = tl.exp(lse_rows - top)
    w_keyed = tl.exp(lse_keyed - top)
    at_rows = ROWS + row[:, None] * stride_rows + head * D + dims[None, :]
    own = tl.load(at_rows, mask=live[:, None], other=0.0).to(tl.float32)
    keyed = tl.load(
        KEYED + row[:, None] * stride_keyed + head * D + dims[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)
    merged = (own * w_rows[:, None] + keyed * w_keyed[:, None]) / (w_rows + w_keyed)[:, None]
    tl.store(at_rows, merged.to(tl.bfloat16), mask=live[:, None])


MERGE_ROWS = 32
MERGE = TritonEntry(
    merge_keys,
    {"ROWS": "*bf16", "LSE": "*fp32", "KEYED": "*bf16", "RAW": "*fp32", "Q": "*bf16", "KM": "*bf16",
     "rows": "i32", "heads": "i32", "stride_rows": "i32", "stride_keyed": "i32", "stride_q": "i32",
     "scale": "fp32", "D": "constexpr", "BR": "constexpr"},
    {"D": DIM, "BR": MERGE_ROWS},
    ("ROWS", "LSE", "KEYED", "RAW", "Q", "KM"),
    num_warps=4,
)  # fmt: skip
KERNELS = TritonKernels("cozy-sage2-fixed-kmean", (*ENTRIES, MERGE))
#: Device index -> its loaded launcher (`serve`).
_LAUNCH: dict[int, Launch] = {}


def serve(index: int, launch: Launch) -> None:
    _LAUNCH[index] = launch


def fixed_mean(launch: Launch, x: Tensor, denominator: int) -> Tensor:
    """BF16 [heads, D] sums of B1/NHD `x` over its tokens, in fixed tiles, over `denominator`.
    `1 / denominator` is f32 as a constant fold rounds it."""
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


def merge(
    rows: Tensor, lse: Tensor, keyed: Tensor, raw: Tensor, query: Tensor, km: Tensor, scale: float
) -> None:
    """ROWS ([1, n, heads, D], log-sum-exp LSE [1, n, heads] F32) merged in place with KEYED,
    QUERY's Sage2 rows over other keys (`merge_keys`)."""
    _, count, heads, dim = rows.shape
    if (
        dim != DIM
        or lse.shape != (1, count, heads)
        or raw.shape != (1, heads, count)
        or not (lse.is_contiguous() and raw.is_contiguous() and km.is_contiguous())
        or any(
            t.shape != rows.shape or t.stride(2) != dim or t.stride(3) != 1 for t in (keyed, query)
        )
        or rows.stride(2) != dim
        or rows.stride(3) != 1
    ):
        raise ValueError("a Sage2 keys merge needs [1, n, heads, 128] rows and their log-sum-exps")
    _LAUNCH[rows.device.index or 0](
        merge_keys, (triton.cdiv(count, MERGE_ROWS), heads, 1),
        rows, lse, keyed, raw, query, km, count, heads, rows.stride(1), keyed.stride(1),
        query.stride(1), float(scale), dim, MERGE_ROWS,
    )  # fmt: skip
