"""The engine's routed-MLP kernels (`engine.moe`): the router's top k and the experts' combine, in the
model's declared set (`engine.kernels`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    if TYPE_CHECKING:
        raise
    TRITON_ABSENT = f"triton does not import ({_absent})"  # type: ignore[unreachable]  # absent Triton
else:
    TRITON_ABSENT = ""

if not TRITON_ABSENT:

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def route_kernel(  # type: ignore[no-untyped-def]
        logits_ptr,
        index_ptr,
        weights_ptr,
        experts,
        E_BLOCK: tl.constexpr,
        K: tl.constexpr,
        RENORM: tl.constexpr,
    ):
        """`Qwen3_5MoeTopKRouter` after its linear, one token a program: softmax in f32, the top
        `K` in descending order (a tie to the lowest index), divided by their sum where `RENORM`,
        rounded to the weights' dtype."""
        row = tl.program_id(0).to(tl.int64)
        c = tl.arange(0, E_BLOCK)
        x = tl.load(logits_ptr + row * experts + c, mask=c < experts, other=-float("inf")).to(tl.float32)
        p = libdevice.exp(x - tl.max(x, axis=0))
        p = p / tl.sum(p, axis=0)
        # The picks' sum first (the same picks as below: the selection is deterministic), then each.
        rest = p
        total = tl.max(p, axis=0) * 0.0
        for _ in tl.static_range(K):
            best = tl.max(rest, axis=0)
            total += best
            rest = tl.where(c == tl.min(tl.where(rest == best, c, E_BLOCK), axis=0), -1.0, rest)
        for j in tl.static_range(K):
            best = tl.max(p, axis=0)
            index = tl.min(tl.where(p == best, c, E_BLOCK), axis=0)
            tl.store(index_ptr + row * K + j, index.to(index_ptr.dtype.element_ty))
            if RENORM:
                best = best / total
            tl.store(weights_ptr + row * K + j, best.to(weights_ptr.dtype.element_ty))
            p = tl.where(c == index, -1.0, p)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def combine_kernel(  # type: ignore[no-untyped-def]
        pairs_ptr,
        weights_ptr,
        shared_ptr,
        gate_ptr,
        out_ptr,
        width,
        K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """`Qwen3_5MoeSparseMoeBlock`'s sum, one launch: each pair's output rounded to the model's
        dtype (the weights'), times its weight, rounded, summed in f32 and rounded (torch's sum);
        plus sigmoid(gate) rounded, times the shared expert's output rounded, rounded; the sum
        rounded, written in `out`'s dtype."""
        row = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = c < width
        dtype = weights_ptr.dtype.element_ty
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(K):
            w = tl.load(weights_ptr + row * K + j).to(tl.float32)
            y = tl.load(pairs_ptr + (row * K + j) * width + c, mask=ok, other=0.0).to(dtype).to(tl.float32)
            acc += (y * w).to(dtype).to(tl.float32)
        routed = acc.to(dtype).to(tl.float32)
        g = tl.load(gate_ptr + row).to(dtype).to(tl.float32)
        sigmoid = (1.0 / (1.0 + libdevice.exp(-g))).to(dtype).to(tl.float32)
        s = tl.load(shared_ptr + row * width + c, mask=ok, other=0.0).to(dtype).to(tl.float32)
        total = (routed + (sigmoid * s).to(dtype).to(tl.float32)).to(dtype)
        tl.store(out_ptr + row * width + c, total.to(out_ptr.dtype.element_ty), mask=ok)
