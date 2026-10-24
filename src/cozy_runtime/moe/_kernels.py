"""The grouped expert GEMV: every (token, expert) pair of a layer in one launch, each pair's weight
read through a table of the experts' addresses, so routing stays on the device and a decode step
records as one graph. A program computes `BLOCK_N` output rows of one pair."""

from __future__ import annotations

from typing import Any

from cozy_runtime.author.kernels import TritonEntry

try:
    import triton
    import triton.language as tl
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    TRITON_ABSENT = f"triton does not import ({_absent})"
else:
    TRITON_ABSENT = ""

BLOCK_N, BLOCK_K = 32, 128

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
    return [bf16, int8]
