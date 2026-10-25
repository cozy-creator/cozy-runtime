"""Sol's sparse steps on datacenter Blackwell (SM100, SM103), on FlashAttention-4.

Sol's own route decides which 64-query x 64-key blocks attend exactly (upstream `prepare`:
block-mean keys, a threshold per query block; plus the +-1 diagonal and the sink). FA4's
block-sparse forward runs 256-query x 128-key tiles, so the route is pooled onto them: a tile
attends exactly when any of its 64x64 blocks does, and the protected prefix's tiles attend
every key. Every other 64-key block contributes Sol's approximation (block-mean key and value,
mass times the block's length), merged into FA4's output by log-sum-exp in one kernel.

Pooling only turns approximated blocks exact. On B200 with H3 15 s Q/K/V (111,598 rows, 56
heads) it raises density from 17% to 33% and lowers the error against exact FP32 from
0.09-0.14 (Sol's kernel) to 0.06-0.09; FA4 runs the exact tiles at 1.72 PF in FP8 and 1.2 PF
in BF16, where Sol's own SM100 kernel reaches 0.55 PF. A whole FP8 call takes 109-113 ms on a
B200 and 81-86 ms on a B300, against Sol's 134-146 ms (h3a-093).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import triton
import triton.language as tl

from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author.kernels import Launch, TritonEntry, TritonKernels
from cozy_runtime.internal import attention_sol
from cozy_runtime.internal import attention_upstream as upstream

if TYPE_CHECKING:
    from torch import Tensor

BLOCK = 64
#: FA4's SM100 forward tile: two 128-row query stages share every key/value tile.
TILE_Q, TILE_K = 256, 128
LOG2E = 1.4426950408889634


def sparse(
    query: Tensor, key: Tensor, value: Tensor, scale: float | None, sink_tokens: int, *, fp8: bool
) -> Tensor:
    """One BF16 BTHD document; every row, the protected prefix's included. `fp8` runs the exact
    tiles in e4m3 with Hadamard-rotated Q/K, as the dense rows' FA4 FP8 does."""
    scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    attention_sol._pin_preprocess()
    kc, vc, threshold = import_module("sol_attn.preprocess").prepare(
        query, key, value, scale=scale, tau=1.0, thresh_type="exact"
    )
    return _routed(query, key, value, kc, vc, threshold, scale, sink_tokens, fp8=fp8)


def _routed(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kc: Tensor,
    vc: Tensor,
    threshold: Tensor,
    scale: float,
    sink_tokens: int,
    *,
    fp8: bool,
) -> Tensor:
    """`sparse` past upstream's preprocess: Sol's route pooled onto FA4's tiles, FA4's exact
    tiles, then the approximation merged in (`_LAUNCH`'s kernel)."""
    import torch

    pooled = route(query, kc, threshold, scale, sink_tokens)
    count = pooled.sum(-1, dtype=torch.int32)
    index = torch.argsort(pooled.to(torch.int8), dim=-1, descending=True, stable=True)
    index = index.to(torch.int32)
    tiles = import_module("flash_attn.cute.block_sparsity").BlockSparseTensorsTorch(
        mask_block_cnt=torch.zeros_like(count)[None],
        mask_block_idx=index[None],
        full_block_cnt=count[None],
        full_block_idx=index[None],
        block_size=(TILE_Q, TILE_K),
    )
    if fp8:
        q, k, v, descales = upstream._e4m3(
            upstream._hadamard(query), upstream._hadamard(key), value
        )
    else:
        q, k, v, descales = query, key, value, {}
    out, lse = import_module("flash_attn.cute.interface")._flash_attn_fwd(
        q, k, v, softmax_scale=scale, block_sparse_tensors=tiles, return_lse=True, **descales
    )[:2]
    _merge_approximation(query, kc, vc, pooled, out, lse, scale)
    return out


@request_path.forbid("kernel build: sol fa4")
def warm(index: int, launch: Launch) -> None:
    """With the merge loaded (`launch`, built by the compile manager), build this card's FA4
    block-sparse forwards (FP8 and BF16) at preparation, one small call each on stand-in
    preprocess outputs: FA4's cache keys neither on length nor on head count, so a run's first
    sparse step finds them built. FA4 is CuTe-DSL compiled in this process by upstream, which
    has no object to build elsewhere; upstream's preprocess builds per length (`attention_sol`)."""
    import torch

    _LAUNCH[index] = launch
    device = torch.device("cuda", index)
    with torch.no_grad(), torch.cuda.device(device):
        q, k, v = (
            torch.randn(1, 1024, 2, 128, device=device, dtype=torch.bfloat16) for _ in range(3)
        )
        kc, vc = (torch.randn(1, 1024 // BLOCK, 2, 128, device=device) for _ in range(2))
        threshold = torch.zeros(1, 1024 // BLOCK, 2, device=device)
        for fp8 in (True, False):
            _routed(q, k, v, kc.bfloat16(), vc.bfloat16(), threshold, 128**-0.5, 64, fp8=fp8)
        torch.cuda.synchronize(device)


def route(query: Tensor, kc: Tensor, threshold: Tensor, scale: float, sink_tokens: int) -> Tensor:
    """Sol's exact-block decisions pooled onto FA4's tiles: bool [heads, query tiles, key tiles]."""
    import torch

    _, tokens, heads, dim = query.shape
    blocks = kc.shape[1]
    pad = blocks * BLOCK - tokens
    counts = torch.full((blocks, 1, 1), float(BLOCK), device=query.device)
    counts[-1] = BLOCK - pad
    padded = torch.nn.functional.pad(query[0], (0, 0, 0, 0, 0, pad))
    centroids = padded.view(blocks, BLOCK, heads, dim).sum(1, dtype=torch.float32) / counts
    q_tiles, k_tiles = -(-blocks // 4), -(-blocks // 2)
    exact = torch.zeros(heads, q_tiles * 4, k_tiles * 2, dtype=torch.bool, device=query.device)
    limit = (threshold[0].T / (scale * LOG2E))[:, :, None]  # [heads, blocks, 1], score units
    for h0 in range(0, heads, 8):  # bounds the FP32 score at 8 x blocks^2
        h1 = min(heads, h0 + 8)
        score = torch.einsum("ihd,jhd->hij", centroids[:, h0:h1], kc[0, :, h0:h1].float())
        exact[h0:h1, :blocks, :blocks] = score > limit[h0:h1]
    near = torch.arange(blocks, device=query.device)
    exact[:, :blocks, :blocks] |= ((near[:, None] - near[None, :]).abs() <= 1)[None]
    exact[:, :, : -(-sink_tokens // BLOCK)] = True
    pooled = exact.view(heads, q_tiles, 4, k_tiles, 2).any(4).any(2)
    pooled[:, : -(-sink_tokens // TILE_Q)] = True
    return pooled


def _merge_approximation(
    query: Tensor, kc: Tensor, vc: Tensor, pooled: Tensor, out: Tensor, lse: Tensor, scale: float
) -> None:
    import torch

    _, tokens, heads, dim = query.shape
    blocks = kc.shape[1]
    if lse.shape != (1, heads, tokens) or not (lse.is_contiguous() and out.is_contiguous()):
        raise ValueError("FA4 returned an unexpected output or LSE layout")
    length = (tokens - torch.arange(blocks, device=query.device) * BLOCK).clamp(1, BLOCK).float()
    keys = kc[0].transpose(0, 1).contiguous()  # head-major block means
    values = (vc[0].float() / length[:, None, None]).to(torch.bfloat16).transpose(0, 1).contiguous()
    _LAUNCH[query.device.index or 0](
        _approximate_and_merge,
        (triton.cdiv(tokens, 128), heads, 1),
        query,
        keys,
        values,
        torch.log2(length),
        pooled.view(torch.int8),
        out,
        lse,
        tokens,
        blocks,
        heads,
        pooled.shape[1],
        pooled.shape[2],
        scale * LOG2E,
        dim,
        128,
        128,
    )


@triton.jit  # type: ignore[untyped-decorator]
def _approximate_and_merge(  # type: ignore[no-untyped-def]
    Q,
    KC,
    VM,
    BIAS,
    POOLED,
    OUT,
    LSE,
    tokens,
    blocks,
    heads,
    q_tiles,
    k_tiles,
    scale_log2,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sol's approximate branch for BLOCK_M rows of one head: every 64-key block outside the
    rows' exact tile contributes its block-mean key KC and value VM with mass `length * exp(s)`
    (BIAS = log2 length). Merged in place with FA4's exact output by log-sum-exp (FA4's LSE is
    natural-log)."""
    tile = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    rows = (tile * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    dims = tl.arange(0, D)
    live = rows < tokens
    q = tl.load(
        Q + (rows[:, None] * heads + head) * D + dims[None, :], mask=live[:, None], other=0.0
    )
    pooled_row = POOLED + (head * q_tiles + (tile * BLOCK_M) // 256) * k_tiles
    keys = KC + head * blocks * D
    values = VM + head * blocks * D
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for start in tl.range(0, blocks, BLOCK_N, warp_specialize=True):
        cols = start + tl.arange(0, BLOCK_N)
        valid = cols < blocks
        kc = tl.load(keys + cols[:, None] * D + dims[None, :], mask=valid[:, None], other=0.0)
        keep = tl.load(pooled_row + cols // 2, mask=valid, other=1) == 0
        bias = tl.where(keep, tl.load(BIAS + cols, mask=valid, other=0.0), float("-inf"))
        s = tl.dot(q, tl.trans(kc)) * scale_log2 + bias[None, :]
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        vm = tl.load(values + cols[:, None] * D + dims[None, :], mask=valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vm)
        m_i = m_new
    o_offsets = (rows[:, None] * heads + head) * D + dims[None, :]
    exact_out = tl.load(OUT + o_offsets, mask=live[:, None], other=0.0).to(tl.float32)
    lse_exact = tl.load(LSE + head * tokens + rows, mask=live, other=0.0) * 1.4426950408889634
    lse_approx = tl.where(l_i > 0, m_i + tl.log2(tl.maximum(l_i, 1e-30)), float("-inf"))
    top = tl.maximum(lse_exact, lse_approx)
    w_exact = tl.exp2(lse_exact - top)
    w_approx = tl.where(l_i > 0, tl.exp2(lse_approx - top), 0.0)
    approx = acc / tl.maximum(l_i, 1e-30)[:, None]
    merged = (exact_out * w_exact[:, None] + approx * w_approx[:, None]) / (w_exact + w_approx)[
        :, None
    ]
    tl.store(OUT + o_offsets, merged.to(tl.bfloat16), mask=live[:, None])


#: The merge, every length and head count: they are runtime sizes.
KERNELS = TritonKernels(
    "cozy-sol-fa4-merge",
    [
        TritonEntry(
            _approximate_and_merge,
            {"Q": "*bf16", "KC": "*bf16", "VM": "*bf16", "BIAS": "*fp32", "POOLED": "*i8",
             "OUT": "*bf16", "LSE": "*fp32", "tokens": "i32", "blocks": "i32", "heads": "i32",
             "q_tiles": "i32", "k_tiles": "i32", "scale_log2": "fp32", "D": "constexpr",
             "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
            {"D": 128, "BLOCK_M": 128, "BLOCK_N": 128},
            ("Q", "KC", "VM", "BIAS", "POOLED", "OUT", "LSE"),
            num_warps=8,
            num_stages=2,
        )
    ],
)  # fmt: skip
#: Device index -> the merge's loaded launcher (`warm`).
_LAUNCH: dict[int, Launch] = {}

__all__ = ["KERNELS", "route", "sparse", "warm"]
