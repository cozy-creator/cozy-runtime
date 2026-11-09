"""The grouped expert kernels: every (token, expert) pair of a layer in one launch per projection,
each pair's weight read through a table of the experts' addresses, so routing stays on the device
and a decode or denoise step records as one graph.

The GEMV (a few pairs) computes `BLOCK_N` output rows of one pair per program. The GEMM (many
pairs) runs over the pairs sorted by expert: a program computes a `GEMM_M` x `GEMM_N` tile of one
expert's segment, its expert and rows read from the tiles' spans the device builds
(`Experts._tiles`), so no count reaches the host. The grid is the most tiles any routing of the
pairs needs (each segment's last tile ragged, never padded in memory); a program past the
routing's own tiles exits. `GROUP` row tiles take each weight block in turn while it is in L2."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cozy_runtime.accel.kernels import TritonEntry

try:
    import triton
    import triton.language as tl
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    if TYPE_CHECKING:
        raise
    TRITON_ABSENT = f"triton does not import ({_absent})"
else:
    TRITON_ABSENT = ""

BLOCK_N, BLOCK_K = 32, 128
#: The GEMM's tile, one config for every card: at most 72 KiB of shared memory in three stages
#: per kind, under the 99 KiB of sm86, sm89 and sm120.
GEMM_M, GEMM_N, GROUP = 64, 128, 8
GEMM_K = {"bf16": 64, "int8": 128}

if not TRITON_ABSENT:

    @triton.jit  # type: ignore[untyped-decorator]
    def experts_bf16_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        weights_ptr,
        experts_ptr,
        out_ptr,
        n,
        k,
        div,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """`out[p] = W[experts[p]] @ x[p // div]`: bf16 `[n, k]` weights at `weights[e]`, f32
        accumulation, rounded to bf16 once."""
        p = tl.program_id(0)
        rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        expert = tl.load(experts_ptr + p)
        w = tl.load(weights_ptr + expert).to(tl.pointer_type(tl.bfloat16))
        x = x_ptr + (p // div) * k
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for start in range(0, k, BLOCK_K):
            cols = start + tl.arange(0, BLOCK_K)
            xv = tl.load(x + cols, mask=cols < k, other=0.0).to(tl.float32)
            ok = (rows[:, None] < n) & (cols[None, :] < k)
            wv = tl.load(w + rows[:, None] * k + cols[None, :], mask=ok, other=0.0)
            acc += tl.sum(wv.to(tl.float32) * xv[None, :], axis=1)
        tl.store(out_ptr + p * n + rows, acc.to(tl.bfloat16), mask=rows < n)

    @triton.jit  # type: ignore[untyped-decorator]
    def experts_int8_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        x_scale_ptr,
        weights_ptr,
        scales_ptr,
        experts_ptr,
        out_ptr,
        n,
        k,
        div,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """`int8.convrot` W8A8 as `int8_linear.eager`: int8 rows (rotated, per-row scale) times
        the int8 `[n, k]` weight at `weights[e]`, exact in int32, then the row's scale and the
        weight row's (`scales[e]`, f32), rounded to bf16 once."""
        p = tl.program_id(0)
        rows = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        expert = tl.load(experts_ptr + p)
        w = tl.load(weights_ptr + expert).to(tl.pointer_type(tl.int8))
        x = x_ptr + (p // div) * k
        acc = tl.zeros([BLOCK_N], dtype=tl.int32)
        for start in range(0, k, BLOCK_K):
            cols = start + tl.arange(0, BLOCK_K)
            xv = tl.load(x + cols, mask=cols < k, other=0).to(tl.int32)
            ok = (rows[:, None] < n) & (cols[None, :] < k)
            wv = tl.load(w + rows[:, None] * k + cols[None, :], mask=ok, other=0)
            acc += tl.sum(wv.to(tl.int32) * xv[None, :], axis=1)
        s = tl.load(scales_ptr + expert).to(tl.pointer_type(tl.float32))
        y = acc.to(tl.float32) * tl.load(x_scale_ptr + p // div)
        y = y * tl.load(s + rows, mask=rows < n, other=0.0)
        tl.store(out_ptr + p * n + rows, y.to(tl.bfloat16), mask=rows < n)

    @triton.jit  # type: ignore[untyped-decorator]
    def _tile(  # type: ignore[no-untyped-def]
        order_ptr,
        spans_ptr,
        n,
        div,
        tiles,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        GROUP: tl.constexpr,
        GATHER: tl.constexpr,
    ):
        """This program's tile of the `[3, tiles]` spans (expert, or -1 past the routing's tiles;
        first sorted row; its segment's end): its expert, which of its `BLOCK_M` sorted pairs are
        live, its `BLOCK_N` output columns, and each pair's input and output row. GATHER: input
        `order[r] // div` (a token's row), output r; else input r, output `order[r]` (the pair's
        (token, k) slot)."""
        pid = tl.program_id(0)
        in_group = GROUP * tl.cdiv(n, BLOCK_N)
        first = pid // in_group * GROUP
        size = tl.minimum(tiles - first, GROUP)
        tile = first + pid % in_group % size
        cols = pid % in_group // size * BLOCK_N + tl.arange(0, BLOCK_N)
        expert = tl.load(spans_ptr + tile)
        rows = tl.load(spans_ptr + tiles + tile) + tl.arange(0, BLOCK_M)
        live = (rows < tl.load(spans_ptr + 2 * tiles + tile)) & (expert >= 0)
        pair = tl.load(order_ptr + rows, mask=live, other=0)
        if GATHER:
            source = pair // div
            target = rows
        else:
            source = rows
            target = pair
        return expert, live, cols, source, target

    @triton.jit  # type: ignore[untyped-decorator]
    def grouped_bf16_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        weights_ptr,
        order_ptr,
        spans_ptr,
        out_ptr,
        n,
        k,
        div,
        tiles,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP: tl.constexpr,
        GATHER: tl.constexpr,
    ):
        """`out[target] = W[e] @ x[source]` for one tile (`_tile`): bf16 `[n, k]` weights at
        `weights[e]`, f32 accumulation, rounded to bf16 once."""
        expert, live, cols, source, target = _tile(
            order_ptr, spans_ptr, n, div, tiles, BLOCK_M, BLOCK_N, GROUP, GATHER
        )
        if expert < 0:
            return
        w = tl.multiple_of(tl.load(weights_ptr + expert), 16).to(tl.pointer_type(tl.bfloat16))
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for start in range(0, k, BLOCK_K):
            kk = start + tl.arange(0, BLOCK_K)
            a_ok = live[:, None] & (kk[None, :] < k)
            a = tl.load(x_ptr + source[:, None] * k + kk[None, :], mask=a_ok, other=0.0)
            b_ok = (kk[:, None] < k) & (cols[None, :] < n)
            b = tl.load(w + cols[None, :] * k + kk[:, None], mask=b_ok, other=0.0)
            acc = tl.dot(a, b, acc)
        ok = live[:, None] & (cols[None, :] < n)
        tl.store(out_ptr + target[:, None] * n + cols[None, :], acc.to(tl.bfloat16), mask=ok)

    @triton.jit  # type: ignore[untyped-decorator]
    def grouped_int8_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        x_scale_ptr,
        weights_ptr,
        scales_ptr,
        order_ptr,
        spans_ptr,
        out_ptr,
        n,
        k,
        div,
        tiles,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP: tl.constexpr,
        GATHER: tl.constexpr,
    ):
        """`experts_int8_kernel`'s law for one tile (`_tile`): int8 rows times the int8 `[n, k]`
        weight at `weights[e]`, exact in int32, then the row's scale and the weight row's
        (`scales[e]`, f32), rounded to bf16 once."""
        expert, live, cols, source, target = _tile(
            order_ptr, spans_ptr, n, div, tiles, BLOCK_M, BLOCK_N, GROUP, GATHER
        )
        if expert < 0:
            return
        w = tl.multiple_of(tl.load(weights_ptr + expert), 16).to(tl.pointer_type(tl.int8))
        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int32)
        for start in range(0, k, BLOCK_K):
            kk = start + tl.arange(0, BLOCK_K)
            a_ok = live[:, None] & (kk[None, :] < k)
            a = tl.load(x_ptr + source[:, None] * k + kk[None, :], mask=a_ok, other=0)
            b_ok = (kk[:, None] < k) & (cols[None, :] < n)
            b = tl.load(w + cols[None, :] * k + kk[:, None], mask=b_ok, other=0)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        s = tl.multiple_of(tl.load(scales_ptr + expert), 16).to(tl.pointer_type(tl.float32))
        y = acc.to(tl.float32) * tl.load(x_scale_ptr + source, mask=live, other=0.0)[:, None]
        y = y * tl.load(s + cols, mask=cols < n, other=0.0)[None, :]
        ok = live[:, None] & (cols[None, :] < n)
        tl.store(out_ptr + target[:, None] * n + cols[None, :], y.to(tl.bfloat16), mask=ok)


def _signature(kernel: Any, **types: str) -> dict[str, str]:
    return {
        name: types.get(name)
        or ("constexpr" if name.isupper() else "*i64" if name.endswith("s_ptr") else "i32")
        for name in kernel.arg_names
    }


def entries() -> list[TritonEntry]:
    if TRITON_ABSENT:
        return []
    blocks = {"BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K}
    bf16 = TritonEntry(
        experts_bf16_kernel,
        _signature(experts_bf16_kernel, x_ptr="*bf16", out_ptr="*bf16"),
        blocks,
        num_warps=4,
    )
    int8 = TritonEntry(
        experts_int8_kernel,
        _signature(experts_int8_kernel, x_ptr="*i8", x_scale_ptr="*fp32", out_ptr="*bf16"),
        blocks,
        num_warps=4,
    )
    tile = {"BLOCK_M": GEMM_M, "BLOCK_N": GEMM_N, "GROUP": GROUP}
    grouped = [
        TritonEntry(
            kernel,
            _signature(kernel, out_ptr="*bf16", order_ptr="*i64", **types),
            {**tile, "BLOCK_K": GEMM_K[kind], "GATHER": gather},
            aligned=("x_ptr", "out_ptr", "n", "k"),
            num_warps=4,
            num_stages=3,
        )
        for kind, kernel, types in (
            ("bf16", grouped_bf16_kernel, {"x_ptr": "*bf16"}),
            ("int8", grouped_int8_kernel, {"x_ptr": "*i8", "x_scale_ptr": "*fp32"}),
        )
        for gather in (1, 0)
    ]
    return [bf16, int8, *grouped]
