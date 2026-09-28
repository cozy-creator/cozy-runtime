"""Per-(batch, head) e4m3 quantisation for FlashAttention-3's fp8 forward (cr-136).

Amax over (sequence, head dim) per head maps onto 440 rather than e4m3's 448 (e4m3fn has no
inf, so a product past 448 becomes NaN in torch's cast), then the scaled cast. On CUDA that
is two Triton launches (~5 bytes per element): h3a-013 measured the eager form at 5.6 ms per
call against a 42 ms kernel and the fused form at 3.4 ms. Everything else (CPU, no triton, a
head dimension the tile cannot take) runs the eager passes. Every reduction stays within one
head, so Ulysses' head split leaves each head's scale exactly what one GPU computes.
"""

from __future__ import annotations

from typing import Any

# Tensors and triton's kernels are `Any`: neither torch nor triton is installed in the check venv.

#: The largest finite e4m3fn magnitude, and the value a head's amax is mapped onto instead.
E4M3_MAX = 448.0
E4M3_TARGET = 440.0
#: Below this a head's amax is treated as this, so an all-zero head quantises to zeros
#: instead of dividing by zero. Shared by the fused and the eager path so their descales agree.
SCALE_FLOOR = 1e-12
#: Rows of one head a fused program handles per launch; the head dimension is the tile width.
CHUNK = 64


def _triton_kernels() -> Any:
    """The fused per-head quantiser: an amax launch and a cast launch, or None where triton is
    absent or refuses. Cached once per process, like cr-086's rowwise kernel.

    Each program owns `CHUNK` rows of ONE head — a (CHUNK, dim) tile whose rows are `dim`
    contiguous elements `heads * dim` apart, which is what (batch, seq, heads, dim) looks like
    from inside a head. The amax pass folds into a (batch, heads) buffer with `atomic_max`,
    which is exact for non-negative f32 because their bit patterns order like their values.
    Same divide (`div_rn`), same clamp and the same RTNE downcast as cr-086; the one honest
    difference from the eager path is that triton's f32->e4m3 cast double-rounds through f16,
    so an element sitting exactly on a representational tie may pick the other neighbour.
    """
    cached = getattr(_triton_kernels, "_cached", "unset")
    if cached != "unset":
        return cached
    try:
        import triton
        import triton.language as tl

        @triton.jit  # type: ignore[untyped-decorator]
        def _amax(x_ptr, amax_ptr, seq, heads, CHUNK: tl.constexpr, DIM: tl.constexpr):  # type: ignore[no-untyped-def]
            head = tl.program_id(0)
            rows = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
            cols = tl.arange(0, DIM)
            batch = head // heads
            offsets = ((batch * seq + rows)[:, None] * heads + head % heads) * DIM + cols[None, :]
            value = tl.load(x_ptr + offsets, mask=rows[:, None] < seq, other=0.0).to(tl.float32)
            tl.atomic_max(amax_ptr + head, tl.max(tl.abs(value)))

        @triton.jit  # type: ignore[untyped-decorator]
        def _cast(x_ptr, out_ptr, amax_ptr, seq, heads, CHUNK: tl.constexpr, DIM: tl.constexpr):  # type: ignore[no-untyped-def]
            head = tl.program_id(0)
            rows = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
            cols = tl.arange(0, DIM)
            batch = head // heads
            offsets = ((batch * seq + rows)[:, None] * heads + head % heads) * DIM + cols[None, :]
            mask = rows[:, None] < seq
            value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            scale = tl.maximum(tl.load(amax_ptr + head) / 440.0, 1e-12)
            quantized = tl.clamp(tl.div_rn(value, scale), -448.0, 448.0)
            tl.store(
                out_ptr + offsets,
                quantized.to(tl.float8e4nv, fp_downcast_rounding="rtne"),
                mask=mask,
            )

        kernels: Any = (_amax, _cast, triton.cdiv)
    except Exception:
        kernels = None
    _triton_kernels._cached = kernels  # type: ignore[attr-defined]
    return kernels


def quantise(tensor: Any) -> tuple[Any, Any]:
    """(batch, seq, heads, dim) bf16/fp16 -> e4m3 of the same shape and a (batch, heads) fp32
    descale, the form FA3's forward takes (`CHECK_SHAPE(q_descale, batch_size, num_heads_k)`).

    The fused path serves a contiguous CUDA tensor whose head dimension is a power of two the
    tile can be built for; everything else takes `quantise_eager`.
    """
    import torch

    batch, seq, heads, dim = tensor.shape
    kernels = _triton_kernels() if tensor.is_cuda and tensor.is_contiguous() else None
    if kernels is None or dim & (dim - 1) or not (16 <= dim <= 256) or not tensor.numel():
        return quantise_eager(tensor)
    amax_kernel, cast_kernel, cdiv = kernels
    amax = torch.zeros((batch, heads), dtype=torch.float32, device=tensor.device)
    out = torch.empty(tensor.shape, dtype=torch.float8_e4m3fn, device=tensor.device)
    grid = (batch * heads, cdiv(seq, CHUNK))
    amax_kernel[grid](tensor, amax, seq, heads, CHUNK=CHUNK, DIM=dim)
    cast_kernel[grid](tensor, out, amax, seq, heads, CHUNK=CHUNK, DIM=dim)
    return out, (amax / E4M3_TARGET).clamp_(min=SCALE_FLOOR)


def quantise_eager(tensor: Any) -> tuple[Any, Any]:
    """The same arithmetic in torch passes: the executable floor, and what a CPU probe runs."""
    import torch

    amax = tensor.abs().amax(dim=(1, 3)).float()
    descale = (amax / E4M3_TARGET).clamp_(min=SCALE_FLOOR)
    scaled = (tensor.float() / descale[:, None, :, None]).clamp_(-E4M3_MAX, E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn), descale.contiguous()
