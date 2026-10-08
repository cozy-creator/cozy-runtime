"""Separate exact-order FP32 row/column scaling followed by BF16 storage."""
import triton
import triton.language as tl


@triton.jit
def _rescale(acc, row_scale, column_scale, output, count: tl.constexpr, columns: tl.constexpr,
             BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < count
    value = tl.load(acc + index, valid, other=0).to(tl.float32)
    rows = tl.load(row_scale + index // columns, valid, other=0).to(tl.float32)
    cols = tl.load(column_scale + index % columns, valid, other=0).to(tl.float32)
    # Match the CUTLASS epilogue's sequential multiply order. Disable contraction.
    value = (value * rows) * cols
    tl.store(output + index, value.to(tl.bfloat16), valid)


def rescale(acc, row_scale, column_scale, output):
    _rescale[(triton.cdiv(acc.numel(), 1024),)](
        acc, row_scale, column_scale, output, acc.numel(), acc.shape[1],
        BLOCK=1024, enable_fp_fusion=False,
    )
    return output
