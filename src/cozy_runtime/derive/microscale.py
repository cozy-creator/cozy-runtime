"""The numeric core of the microscaling quant kinds: E2M1/E4M3/E8M0 arithmetic in numpy.

Every encoder here is INTEGER-DETERMINISTIC — table lookups, comparisons and float32
arithmetic with no reduction order that a device can reorder — so two runs on two hosts
produce the same bytes. That is not a nicety: #501c banked that fp8 is bit-identical
within a host but not across hosts once a GPU is in the loop, and a quant artifact whose
digest depends on which card produced it cannot be pinned by a recipe.

The element tables are TABLES, not formulas, and they are written to match the border's
reference decoders exactly (`tensorfs crates/tensorfs-core/src/bin/tfs/conform.rs`:
`f32_from_e4m3fn`, `f32_from_e2m1`, `f32_from_e8m0`). Producer and reader agree by
construction rather than by both being "an implementation of the spec".

THE SCALE-SELECTION RULE (#513/#517's defect class, replanted for fp4)
----------------------------------------------------------------------
#517 root-caused 1.02 dB of the mxfp8 gap to a scale rule that was POSITIONAL — the OCP
floor exponent — rather than chosen. Its own largest element clamped in every dense
block. The same trap is one line wide in nvfp4, because the two-level scheme divides by a
per-block E4M3 scale that a round-to-nearest cast can round DOWN, at which point the
block's max lands above E2M1's 6.0 and saturates.

So the rule here is CHOSEN ERROR-MINIMIZING, never positional:

  * the anchor is the smallest E4M3 scale that cannot clip — the first value at or above
    the block's exact requirement, re-checked in the same float32 arithmetic the encoder
    actually uses and bumped where double-to-single narrowing left it short, so the
    no-clip guarantee of that anchor is arithmetic rather than algebraic;
  * the candidate set is the anchor, its neighbour UP, and its rounded-DOWN (clipping)
    neighbour — admissible per the #543d family ruling and selected by measured error;
  * every candidate is quantized THROUGH the stored representation and the smaller
    sum-of-squares wins, ties to the smaller scale. Positional selection is what #517
    convicted; this compares. Saturation is a REPORTED per-tensor stat, never a
    constraint.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from cozy_runtime.author import UnsupportedInput

from .safetensors_io import row_chunks

#: Rows per step for the fp8 encoders: a working set that stays in cache is several times
#: faster than one streamed through memory. Encoders are row-independent, so no byte changes.
CACHE_ELEMENTS = 1 << 18

# --------------------------------------------------------------------- element tables


def _e4m3_magnitudes() -> np.ndarray:
    """Byte 0x00..0x7E in ascending magnitude; 0x7F is NaN and is not representable here.
    exp==0 is the subnormal run man*2^-9; otherwise (1+man/8)*2^(exp-7). Max 448."""
    mags = np.empty(127, dtype=np.float64)
    for b in range(127):
        exp, man = (b >> 3) & 0x0F, b & 0x07
        mags[b] = man * 2.0**-9 if exp == 0 else (1.0 + man / 8.0) * 2.0 ** (exp - 7)
    return mags


#: E4M3 (`f8_e4m3fn`) magnitudes by byte, and E2M1's sixteen-value nibble table.
E4M3_MAG = _e4m3_magnitudes()
E4M3_MAG32 = E4M3_MAG.astype(np.float32)
E4M3_MAX = float(E4M3_MAG[-1])  # 448.0

E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
E2M1_MAG32 = E2M1_MAG.astype(np.float32)
E2M1_MAX = 6.0
#: nibble -> value, sign in bit 3. The border reads element 2k from the LOW nibble.
E2M1_DECODE32 = np.concatenate([E2M1_MAG32, -E2M1_MAG32]).astype(np.float32)


def _nearest_even(mags: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Index of the nearest table magnitude, ties to the EVEN index.

    Even index is even mantissa-LSB for both tables (E2M1: 0, 1.0, 2.0, 4.0 have man 0;
    E4M3: byte parity IS the mantissa LSB), so this is round-to-nearest-even in the
    representation, not merely in the index.
    """
    mids = (mags[:-1] + mags[1:]) / 2.0
    idx = np.searchsorted(mids, x, side="right")
    tie = (idx > 0) & (x == mids[np.maximum(idx - 1, 0)])
    return np.where(tie & (idx % 2 == 1), idx - 1, idx).astype(np.uint8)


def e4m3_encode(values: np.ndarray) -> np.ndarray:
    """float32 -> E4M3 bytes, RNE, SATURATING at +/-448 (the cast itself does not, which
    is why every caller in this family clamps and why the border's own note says so)."""
    v = np.asarray(values, dtype=np.float32)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v, nan=0.0, posinf=E4M3_MAX, neginf=-E4M3_MAX)
    return _e4m3_signed(v, np.minimum(np.abs(v), np.float32(E4M3_MAX)))


def _e4m3_signed(values: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    """E4M3 bytes of finite `values` whose magnitudes are already at most 448."""
    bits = magnitude.view(np.uint32)
    # Normal values discard 20 of float32's 23 mantissa bits. Adding half an
    # ulp minus one, plus the retained LSB, rounds ties to even; carry also
    # rounds across exponent boundaries. Adjust exponent bias from 127 to 7.
    code = bits + np.uint32(0x7FFFF)
    code += (bits >> 20) & np.uint32(1)
    code >>= 20
    code -= np.uint32(120 << 3)
    # E4M3 subnormals have fixed spacing 2**-9. Multiplication is exact here, and adding
    # 2**23 rounds to an integer with the same even-tie rule, left in the low bits.
    small = bits < np.uint32(0x3C800000)
    sub = (magnitude[small] * np.float32(512) + np.float32(2.0**23)).view(np.uint32)
    code[small] = sub - np.uint32(0x4B000000)
    code |= (np.ascontiguousarray(values).view(np.uint32) >> 24) & np.uint32(0x80)
    out: np.ndarray = code.astype(np.uint8)
    return out


#: byte -> float32, sign in bit 7; the NaN codes 0x7F/0xFF read as the largest magnitude.
E4M3_DECODE32 = np.concatenate([E4M3_MAG32[np.minimum(np.arange(128), 126)]] * 2)
E4M3_DECODE32[128:] *= -1


def e4m3_decode(raw: np.ndarray) -> np.ndarray:
    out: np.ndarray = E4M3_DECODE32.take(np.asarray(raw, dtype=np.uint8), mode="clip")
    return out


def e2m1_encode(values: np.ndarray) -> np.ndarray:
    """float32 -> E2M1 nibbles (0..15), RNE, saturating at +/-6."""
    v = np.nan_to_num(
        np.asarray(values, dtype=np.float32), nan=0.0, posinf=E2M1_MAX, neginf=-E2M1_MAX
    )
    mag = np.minimum(np.abs(v).astype(np.float64), E2M1_MAX)
    idx = _nearest_even(E2M1_MAG, mag)
    out: np.ndarray = (idx | np.where(np.signbit(v), 0x08, 0x00).astype(np.uint8)).astype(np.uint8)
    return out


def pack_nibbles(nibbles: np.ndarray) -> np.ndarray:
    """[r, c] nibbles -> [r, c/2] bytes, element 2k in the LOW nibble (the border's
    packing; `nvfp4:nibbles-swapped` is one of its own WRONG conformance arms)."""
    rows, cols = nibbles.shape
    pairs = nibbles.reshape(rows, cols // 2, 2)
    out: np.ndarray = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).astype(np.uint8)
    return out


def unpack_nibbles(packed: np.ndarray, cols: int) -> np.ndarray:
    rows = packed.shape[0]
    out = np.empty((rows, cols), dtype=np.uint8)
    out[:, 0::2] = packed & 0x0F
    out[:, 1::2] = packed >> 4
    return out


# ------------------------------------------------------------------------- the result


class Companion(NamedTuple):
    """One scale sibling: the carrier SUFFIX (appended to the module STEM, #501d), the
    header dtype, the shape, and the stored bytes."""

    suffix: str
    dtype: str
    shape: list[int]
    raw: bytes


class Encoded(NamedTuple):
    payload_dtype: str
    payload_shape: list[int]
    payload: bytes
    companions: list[Companion]
    saturated: int
    """Elements whose magnitude exceeded the payload format's range AFTER scaling. The
    #517 defect class reduced to one integer, carried into the manifest."""
    sse: float
    energy: float
    max_abs_err: float

    def relative_frobenius(self) -> float:
        """||W - dequant(quant(W))||_F / ||W||_F — the per-tensor quality number this
        family banks. Zero energy (an all-zero tensor) is exactly reproduced, so 0.0."""
        return float(np.sqrt(self.sse / self.energy)) if self.energy > 0 else 0.0


class _Accum:
    __slots__ = ("energy", "max_abs", "sat", "sse")

    def __init__(self) -> None:
        self.sse = 0.0
        self.energy = 0.0
        self.max_abs = 0.0
        self.sat = 0

    def add(self, source: np.ndarray, recon: np.ndarray) -> None:
        wide = source.astype(np.float64)
        diff = recon.astype(np.float64)
        diff -= wide
        if diff.size:
            self.max_abs = max(self.max_abs, float(diff.max()), -float(diff.min()))
        self.sse += float(np.sum(np.square(diff, out=diff)))
        self.energy += float(np.sum(np.square(wide, out=wide)))


def _require_2d(key: str, values: np.ndarray, block: int) -> tuple[int, int]:
    if values.ndim != 2:
        raise UnsupportedInput(
            f"{key!r} is rank {values.ndim}: this family encodes 2-D GEMM weights, and a "
            "norm or a bias stays at source precision rather than being reshaped into one"
        )
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if cols % block:
        raise UnsupportedInput(
            f"{key!r} has {cols} columns, not a multiple of the {block}-element block — "
            "the encoding's blocking is a shape fact and padding it would change the "
            "tensor the consumer reads"
        )
    return rows, cols


# ----------------------------------------------------------------------------- nvfp4

NVFP4_BLOCK = 16


def encode_nvfp4(key: str, values: np.ndarray) -> Encoded:
    """The two-level NVFP4 scheme, in the border's `nvfp4-w4a4/1` base-variant layout.

    REFERENCE. The scheme is NVIDIA's NVFP4 as exported by TensorRT Model Optimizer and
    decoded by this platform's own border: `value[i,j] = e2m1(nibble) *
    e4m3(weight_scale[i, j/16]) * weight_scale_2`, with `weight` packed u8 [out, in/2],
    `weight_scale` e4m3 [out, in/16] and `weight_scale_2` an f32 SCALAR. That decode line
    is the contract this producer is written against (registry.rs `nvfp4-w4a4/1`,
    conform.rs `nvfp4_inner`) — a second-hand description of the format is not, which
    matters because two of the border's three conformance arms for this encoding are
    plausible misreadings (`nibbles-swapped`, `global-scale-ignored`).

    The second-level scalar is `amax / (448 * 6)`: it maps the tensor's largest element to
    the largest product the two levels can express, so the block requiring the most range
    lands at E4M3's own maximum and no block requirement is unrepresentable. Per block the
    exact requirement is `block_amax / (6 * s2)` and the scale is chosen by min-SSE
    through the stored representation over the family rule's candidate set — the
    rounded-down (CLIPPING) neighbour included, per #543d (see the module docstring).
    """
    rows, cols = _require_2d(key, values, NVFP4_BLOCK)
    nb = cols // NVFP4_BLOCK
    amax = float(np.max(np.abs(values))) if values.size else 0.0
    if not np.isfinite(amax):
        raise UnsupportedInput(f"{key!r} contains a non-finite weight — refuse, never encode")
    # An all-zero tensor has no scale to derive; 1.0 reproduces it exactly and keeps the
    # scalar a real number rather than a zero that makes every later division undefined.
    s2 = np.float32(amax / (E4M3_MAX * E2M1_MAX)) if amax > 0 else np.float32(1.0)

    payload = np.empty((rows, cols // 2), dtype=np.uint8)
    scales = np.empty((rows, nb), dtype=np.uint8)
    acc = _Accum()
    for sl in row_chunks(rows, cols):
        blk = np.ascontiguousarray(values[sl], dtype=np.float32).reshape(-1, nb, NVFP4_BLOCK)
        bamax = np.max(np.abs(blk), axis=-1)
        need = (bamax / np.float32(E2M1_MAX * s2)).astype(np.float64)
        # Saturation-free by construction: the smallest E4M3 value at or above `need`.
        idx = np.clip(np.searchsorted(E4M3_MAG, need, side="left"), 0, 126).astype(np.int64)
        # ...then by ARITHMETIC: re-check in the same float32 the encoder will use, and
        # bump anything the double-to-single narrowing left one ulp short. Two rounds is
        # a bound, not a hope — one bump moves the scale a full E4M3 step.
        for _ in range(3):
            eff = (E4M3_MAG32[idx] * s2).astype(np.float32)
            short = (bamax > eff * np.float32(E2M1_MAX)) & (idx < 126)
            if not bool(short.any()):
                break
            idx = np.where(short, idx + 1, idx)

        best_sse: np.ndarray | None = None
        best: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # THE FAMILY RULE (#543d, realigning this kind to #517): clipping candidates are
        # ADMISSIBLE. The rounded-DOWN neighbour clips the block's own largest element,
        # and in fp4 that trade WINS 0.33 dB weight-SNR on the banked audit — an E4M3
        # scale step is ~6% while the resolution gain applies to all 16 elements.
        # Iteration order is smallest scale first, so the strict `<` keeps the smaller
        # scale on ties; saturation is a REPORTED stat below, never a constraint.
        for cand in (np.maximum(idx - 1, 0), idx, np.minimum(idx + 1, 126)):
            eff = (E4M3_MAG32[cand] * s2).astype(np.float32)[..., None]
            safe = np.where(eff > 0, eff, np.float32(1.0))
            nib = e2m1_encode(np.where(eff > 0, blk / safe, np.float32(0.0)))
            recon = (E2M1_DECODE32[nib] * eff).astype(np.float32)
            sse = np.sum((recon - blk).astype(np.float64) ** 2, axis=-1)
            if best_sse is None:
                best_sse, best = sse, (cand, nib, recon)
            else:
                assert best is not None
                take = sse < best_sse  # strict: a tie keeps the SMALLER scale
                best_sse = np.where(take, sse, best_sse)
                best = (
                    np.where(take, cand, best[0]),
                    np.where(take[..., None], nib, best[1]),
                    np.where(take[..., None], recon, best[2]),
                )
        assert best is not None
        chosen, nib, recon = best
        limit = (E4M3_MAG32[chosen] * s2).astype(np.float32)[..., None] * np.float32(E2M1_MAX)
        acc.sat += int(np.count_nonzero(np.abs(blk) > limit))
        acc.add(blk.reshape(-1, cols), recon.reshape(-1, cols))
        payload[sl] = pack_nibbles(nib.reshape(-1, cols))
        scales[sl] = chosen.astype(np.uint8)

    return Encoded(
        payload_dtype="U8",
        payload_shape=[rows, cols // 2],
        payload=payload.tobytes(),
        companions=[
            Companion(".weight_scale", "F8_E4M3", [rows, nb], scales.tobytes()),
            Companion(".weight_scale_2", "F32", [], np.float32(s2).tobytes()),
        ],
        saturated=acc.sat,
        sse=acc.sse,
        energy=acc.energy,
        max_abs_err=acc.max_abs,
    )


# ----------------------------------------------------------------------------- mxfp8

MXFP8_BLOCK = 32


def encode_mxfp8(key: str, values: np.ndarray) -> Encoded:
    """OCP MXFP8 in `mxfp8/1`: E4M3 elements, one E8M0 byte per 32-element block.

    The exponent rule is #517's: both admissible exponents are quantized THROUGH the
    stored representation and the smaller squared error wins, ties to the lower. This is
    the ONE implementation of the rule since #549.10 retired the torch duplicate — integer
    arithmetic so the mixed kind does not need a GPU host to emit an mxfp8 tensor.
    """
    rows, cols = _require_2d(key, values, MXFP8_BLOCK)
    nb = cols // MXFP8_BLOCK
    payload = np.empty((rows, cols), dtype=np.uint8)
    scales = np.empty((rows, nb), dtype=np.uint8)
    acc = _Accum()
    for sl in row_chunks(rows, cols, CACHE_ELEMENTS):
        blk = np.ascontiguousarray(values[sl], dtype=np.float32).reshape(-1, nb, MXFP8_BLOCK)
        bamax = _block_amax(blk)
        with np.errstate(divide="ignore"):
            e_lo = np.where(bamax > 0, np.floor(np.log2(bamax.astype(np.float64))) - 8.0, -127.0)
        byte_lo = np.clip(e_lo + 127.0, 0.0, 254.0)
        byte_hi = np.where(bamax > 0, np.clip(byte_lo + 1.0, 0.0, 254.0), byte_lo)
        enc, recon, sse = _mxfp8_candidate(blk, byte_lo)
        enc_hi, recon_hi, sse_hi = _mxfp8_candidate(blk, byte_hi)
        take = sse_hi < sse  # strict: a tie keeps the LOWER exponent
        chosen = np.where(take, byte_hi, byte_lo)
        np.copyto(enc, enc_hi, where=take[..., None])
        np.copyto(recon, recon_hi, where=take[..., None])
        scale = np.power(np.float32(2.0), (chosen - 127.0).astype(np.float32))
        # Only a block whose largest element exceeds the E4M3 range can saturate.
        over = bamax / scale > E4M3_MAX
        if over.any():
            acc.sat += int(np.count_nonzero(np.abs(blk[over]) / scale[over][:, None] > E4M3_MAX))
        acc.add(blk.reshape(-1, cols), recon.reshape(-1, cols))
        payload[sl] = enc.reshape(-1, cols)
        scales[sl] = chosen.astype(np.uint8)

    return Encoded(
        payload_dtype="F8_E4M3",
        payload_shape=[rows, cols],
        payload=payload.tobytes(),
        companions=[Companion(".weight_scale", "U8", [rows, nb], scales.tobytes())],
        saturated=acc.sat,
        sse=acc.sse,
        energy=acc.energy,
        max_abs_err=acc.max_abs,
    )


def _block_amax(blk: np.ndarray) -> np.ndarray:
    """max |x| over the last axis by halving: exact, and far cheaper than a short-axis reduce."""
    m = np.abs(blk)
    while m.shape[-1] > 1:
        m = np.maximum(m[..., : m.shape[-1] // 2], m[..., m.shape[-1] // 2 :])
    return m[..., 0]


def _mxfp8_candidate(blk: np.ndarray, exponent: np.ndarray) -> tuple[np.ndarray, ...]:
    """One candidate exponent per block, quantized THROUGH the stored representation."""
    scale = np.power(np.float32(2.0), (exponent - 127.0).astype(np.float32))[..., None]
    scaled = blk / scale
    np.clip(scaled, -E4M3_MAX, E4M3_MAX, out=scaled)
    enc = _e4m3_signed(scaled, np.abs(scaled))
    recon = E4M3_DECODE32.take(enc, mode="clip")
    recon *= scale
    diff = (recon - blk).astype(np.float64)
    return enc, recon, np.sum(np.square(diff, out=diff), axis=-1)


# ---------------------------------------------------------------------- fp8 rowwise


def encode_fp8_rowwise(key: str, values: np.ndarray) -> Encoded:
    """`fp8-rowwise/1`, rank-1 scale variant: continuous per-output-row f32 scale
    `clamp(rowmax|v|/448, min 1e-12)`, RNE payload, clamp BEFORE the cast. Bit-for-bit
    the runtime's `quantize_rowwise` reference, restated in numpy."""
    rows, cols = _require_2d(key, values, 1)
    payload = np.empty((rows, cols), dtype=np.uint8)
    row_scale = np.empty((rows,), dtype=np.float32)
    acc = _Accum()
    for sl in row_chunks(rows, cols, CACHE_ELEMENTS):
        v = np.ascontiguousarray(values[sl], dtype=np.float32)
        amax = np.maximum(v.max(axis=-1), -v.min(axis=-1))
        scale = np.maximum(amax / np.float32(E4M3_MAX), np.float32(1e-12)).astype(np.float32)
        rescaled = v / scale[:, None]
        # Division is monotone, so only a row whose largest element exceeds 448 saturates.
        over = amax / scale > E4M3_MAX
        if over.any():
            acc.sat += int(np.count_nonzero(np.abs(rescaled[over]) > E4M3_MAX))
            np.clip(rescaled, -E4M3_MAX, E4M3_MAX, out=rescaled)
        enc = _e4m3_signed(rescaled, np.abs(rescaled))
        recon = E4M3_DECODE32.take(enc, mode="clip")
        recon *= scale[:, None]
        acc.add(v, recon)
        payload[sl] = enc
        row_scale[sl] = scale

    return Encoded(
        payload_dtype="F8_E4M3",
        payload_shape=[rows, cols],
        payload=payload.tobytes(),
        companions=[Companion(".weight_scale", "F32", [rows], row_scale.tobytes())],
        saturated=acc.sat,
        sse=acc.sse,
        energy=acc.energy,
        max_abs_err=acc.max_abs,
    )
