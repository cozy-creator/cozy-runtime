"""Private SM120 experiment: scalar FP8 GEMM then bounded FP32 outer rescaling.

The public rowwise recipe, stored bytes and quantization are unchanged. Hardware
outside SM120 and non-BF16 destinations keep the existing native rowwise call.
This is not a claim that the new path wins on arbitrary shapes or full requests.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

MAX_FP32_BYTES = 128 << 20
MAX_ROWS = 4096


def tile_rows(columns: int) -> int:
    """Power-of-two rows within the FP32-output cap; zero means even one row cannot fit."""
    if columns <= 0:
        return 0
    limit = min(MAX_ROWS, MAX_FP32_BYTES // (4 * columns))
    return 1 << (limit.bit_length() - 1) if limit else 0


def legacy(torch: Any, a: Any, b: Any, scale_a: Any, scale_b: Any, dtype: Any) -> Any:
    """The unchanged native rowwise spelling, retained for other arches and reference."""
    return torch._scaled_mm(
        a, b, scale_a=scale_a, scale_b=scale_b, out_dtype=dtype, use_fast_accum=True
    )


def eligible(torch: Any, a: Any, columns: int, dtype: Any) -> bool:
    return bool(
        a.is_cuda
        and dtype == torch.bfloat16
        and a.shape[0] > 0
        and a.shape[1] > 0
        and tile_rows(columns)
        and torch.cuda.get_device_capability(a.device) == (12, 0)
    )


def mm(torch: Any, a: Any, b: Any, scale_a: Any, scale_b: Any, dtype: Any) -> Any:
    if not eligible(torch, a, int(b.shape[1]), dtype):
        return legacy(torch, a, b, scale_a, scale_b, dtype)
    return scalar_rescaled(torch, a, b, scale_a, scale_b)


def scalar_rescaled(torch: Any, a: Any, b: Any, scale_a: Any, scale_b: Any) -> Any:
    """One FP32 output tile at a time; vendor workspace is outside this tensor cap."""
    rows, columns = int(a.shape[0]), int(b.shape[1])
    tile = tile_rows(columns)
    if not tile:
        raise ValueError("one FP32 output row exceeds the explicit scratch bound")
    output = torch.empty((rows, columns), dtype=torch.bfloat16, device=a.device)
    # Request-local scalar, not a CUDA tensor cache that survives the invocation.
    one = torch.ones((), dtype=torch.float32, device=a.device)
    for start in range(0, rows, tile):
        stop = min(start + tile, rows)
        accum = torch._scaled_mm(
            a[start:stop],
            b,
            scale_a=one,
            scale_b=one,
            out_dtype=torch.float32,
            use_fast_accum=False,
            bias=None,
        )
        if accum.dtype != torch.float32 or not accum.is_contiguous():
            raise ValueError("scalar FP8 GEMM must return a contiguous FP32 output")
        rescale(accum, scale_a[start:stop], scale_b, output[start:stop])
        # Quantizers, GEMM and this epilogue launch on the current Torch stream.
        # Release before allocating the next tile, so two FP32 tiles cannot overlap.
        del accum
    return output


@lru_cache(maxsize=1)
def _kernel() -> Any:
    import triton
    import triton.language as tl

    @triton.jit  # type: ignore[untyped-decorator]
    def scale_kernel(  # type: ignore[no-untyped-def]
        accum,
        scale_a,
        scale_b,
        output,
        count,
        columns,
        stride_a,
        stride_b,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = index < count
        value = tl.load(accum + index, valid, other=0).to(tl.float32)
        row = tl.load(scale_a + (index // columns) * stride_a, valid, other=0).to(tl.float32)
        col = tl.load(scale_b + (index % columns) * stride_b, valid, other=0).to(tl.float32)
        value = (value * row) * col
        tl.store(output + index, value.to(tl.bfloat16), valid)

    return scale_kernel


def rescale(accum: Any, scale_a: Any, scale_b: Any, output: Any) -> None:
    count = accum.numel()
    _kernel()[((count + 1023) // 1024,)](
        accum,
        scale_a,
        scale_b,
        output,
        count,
        accum.shape[1],
        scale_a.stride(0),
        scale_b.stride(1),
        BLOCK=1024,
        enable_fp_fusion=False,
    )
