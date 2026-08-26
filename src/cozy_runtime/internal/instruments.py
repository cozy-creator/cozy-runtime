"""The MEASUREMENT INSTRUMENTS: quantizers that build probe subjects and never serve one.

#549.10 moved these two functions out of `encoding.py`. They are not decoders, not
providers, and no serving path calls them — their only callers are `probe.py`, which needs a
real carrier to qualify a provider against, and the live drivers, which need a CONTROL to
subtract. Living beside the providers made them look like a third route; a reader had to get
several paragraphs in to learn that nothing in the fill plane can reach them.

What they measure is #514a's question — what a SECOND quantization costs — and the answer is
the product, which is why both return a record of the cost rather than a bare tensor.

They are not dead code and they are not tests. `probe.py` cannot mint a capability record
without one: `_weight_roles` quantizes a seeded weight by the encoding's OWN equation so the
routes are compared on a real carrier rather than on random bytes no producer would emit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cozy_runtime.internal.encoding.formats import (
    E4M3_MAX,
    MX_BLOCK,
    ROW_SCALE_FLOOR,
    SATURATION_MARGIN,
    MicroScaledDequant,
)


@dataclass(frozen=True, slots=True)
class RowwiseDerivation:
    """What a rowwise quantization produced, and what it COST.

    Returned rather than logged: the error IS the product here, because the derivation is a
    second lossy quantization and a caller that cannot see its cost cannot gate on it.
    """

    data: Any
    """E4M3 payload, the logical shape. `fp8-rowwise/1`'s `data` role."""
    scale: Any
    """f32, one CONTINUOUS scale per output row. `fp8-rowwise/1`'s `scale` role."""
    clamped: int
    """Elements the 448 clamp caught AT ALL, boundary rounding included."""
    saturated: int
    """Elements that overshot 448 by more than rounding can explain — the ones that mean
    the row scale genuinely did not cover its row. `clamped` minus this is benign: the row
    maximum is SUPPOSED to land on 448, three roundings can put it a few ULP over, and
    clamping it back to 448 yields the very value the cast would have produced anyway."""
    max_rescaled: float
    """The largest magnitude offered to the cast, before clamping. The number that lets a
    reader tell 448.00003 (rounding) from 900 (lost range) instead of guessing."""
    zero_rows: int
    """Rows whose amax was 0, held off the divide by `ROW_SCALE_FLOOR`."""


def derive_rowwise(
    torch: Any, data: Any, scale: Any, *, block: int = MX_BLOCK
) -> RowwiseDerivation:
    """`mxfp8/1` roles -> `fp8-rowwise/1` roles. A SECOND LOSSY QUANTIZATION (#514a).

    Hopper's fp8 tensor cores take one scale per row, not one per 32-element block, so
    serving mxfp8 there would mean collapsing the block scales into a continuous row scale
    and RE-ROUNDING the payload. Both halves of that lose information and neither is hidden:

    * the row scale is `rowmax|V| / 448`, so the row's largest magnitude lands exactly on
      E4M3's largest. That is v1's rule for the same quantity, and it makes the derived
      form's dynamic range the ROW's rather than the BLOCK's — which is precisely the
      information mxfp8 had and `fp8-rowwise/1` cannot carry.
    * each block's payload is then rescaled by `mult[r,b] / s[r]` and re-rounded to E4M3,
      ROUND-TO-NEAREST-EVEN, clamped to +/-448 BEFORE the cast because that cast does not
      saturate. A block far below its row's maximum loses mantissa bits here; a block at the
      maximum loses none.

    The ratio is NOT a power of two — `mult` is, `s[r]` is continuous — so the rescale is a
    real multiply and a real rounding, not an exponent shift. That is the whole source of
    the double-quantization penalty, and it is MEASURED against a DIRECT quantization of the
    same weights (`quantize_rowwise` below) rather than asserted.

    **NO SERVING PATH CALLS THIS.** It is an instrument, and #517a is why it is not a route:
    the block→row collapse costs 1.44-1.52x relative error on real weights, and a lane that
    served through it would be paying that for a convenience nobody asked for.

    **Structured blockwise on purpose.** `rowmax|V|` is computed as
    `max_b(blockmax|payload| * mult[r,b])` — in the PAYLOAD domain, never by materializing
    the dequantized weight — so the derivation's working set is the payload plus two grids
    of 1/32 its size.
    """
    rows, cols = int(data.shape[0]), int(data.shape[1])
    blocks = int(scale.shape[-1])
    if cols != blocks * block:
        raise AssertionError(
            f"a ragged mxfp8 tail ({cols} cols, {blocks} blocks) has no rowwise derivation"
        )
    mult = MicroScaledDequant._multiplier(torch, scale, torch.float32)
    payload = data.view(rows, blocks, block).to(torch.float32)
    # The row's true maximum magnitude, exactly, without ever forming the row.
    #
    # E8M0's NaN byte is EXCLUDED from the maximum rather than allowed to propagate into
    # it. A NaN block scale means THAT BLOCK is not a number; letting it set the row scale
    # turns 32 poisoned elements into a poisoned ROW, which is a strictly larger claim than
    # the stored bytes make. The NaN still reaches the output — `ratio` carries it, so the
    # block's own 32 payload elements cast to E4M3's NaN — it just stops there.
    block_peak = payload.abs().amax(dim=-1) * mult
    amax = torch.nan_to_num(block_peak, nan=0.0).amax(dim=-1)
    zero_rows = int((amax == 0).sum())
    row_scale = (amax / E4M3_MAX).clamp(min=ROW_SCALE_FLOOR)
    ratio = mult / row_scale.unsqueeze(-1)
    rescaled = payload * ratio.unsqueeze(-1)
    magnitude = rescaled.abs()
    # nan-safe: a NaN block must not erase the peak a reader uses to tell rounding-boundary
    # clamping from genuinely lost range.
    peak = float(torch.nan_to_num(magnitude, nan=0.0).max())
    clamped = int((magnitude > E4M3_MAX).sum())
    saturated = int((magnitude > E4M3_MAX * SATURATION_MARGIN).sum())
    derived = rescaled.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(rows, cols)
    return RowwiseDerivation(
        derived, row_scale.to(torch.float32), clamped, saturated, peak, zero_rows
    )


def quantize_rowwise(torch: Any, weight: Any) -> RowwiseDerivation:
    """float -> `fp8-rowwise/1` DIRECTLY, in one quantization. The derivation's CONTROL.

    Same rule, same clamp, same floor — the only difference is that this one sees the
    original weight and `derive_rowwise` sees a weight that has already been through mxfp8.
    Subtracting the two is what turns "a second quantization is lossy" from a statement
    into a number.

    `probe.py` also uses it to BUILD its subject: a qualification that measured a decoder
    against random bytes would be measuring nothing a producer emits.
    """
    work = weight.to(torch.float32)
    amax = work.abs().amax(dim=-1)
    zero_rows = int((amax == 0).sum())
    row_scale = (amax / E4M3_MAX).clamp(min=ROW_SCALE_FLOOR)
    rescaled = work / row_scale.unsqueeze(-1)
    magnitude = rescaled.abs()
    return RowwiseDerivation(
        rescaled.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn),
        row_scale.to(torch.float32),
        int((magnitude > E4M3_MAX).sum()),
        int((magnitude > E4M3_MAX * SATURATION_MARGIN).sum()),
        float(magnitude.max()),
        zero_rows,
    )
