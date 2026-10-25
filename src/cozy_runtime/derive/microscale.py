"""Compatibility encoders and independent decoders for derived weight formats.

TensorFS owns the versioned NVFP4 and ConvRot numerical producers and acceptance statistics.
Their adapters preserve bounded row chunks and the existing Encoded/Companion result contract.
FP8/W4A8 producers and independent decode helpers remain here.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import tensorfs
from tensorfs import raw

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


#: E4M3 (`f8_e4m3fn`) magnitudes by byte.
E4M3_MAG = _e4m3_magnitudes()
E4M3_MAG32 = E4M3_MAG.astype(np.float32)
E4M3_MAX = float(E4M3_MAG[-1])  # 448.0


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


def pack_nibbles(nibbles: np.ndarray) -> np.ndarray:
    """[r, c] nibbles -> [r, c/2] bytes, element 2k in the LOW nibble (the border's
    packing; `nvfp4:nibbles-swapped` is one of its own WRONG conformance arms)."""
    rows, cols = nibbles.shape
    pairs = nibbles.reshape(rows, cols // 2, 2)
    out: np.ndarray = (pairs[:, :, 0] | (pairs[:, :, 1] << 4)).astype(np.uint8)
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

    def merge(self, measured: raw.QuantizationStats) -> None:
        self.sse += measured["sse"]
        self.energy += measured["energy"]
        self.max_abs = max(self.max_abs, measured["max_abs_error"])
        self.sat += measured["saturated"]


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


def _measured(
    key: str,
    values: np.ndarray,
    scheme: str,
    strategy: str,
    params: dict[str, object],
    tensor_amax: float | None = None,
) -> tuple[tensorfs.Encoded, raw.QuantizationStats]:
    """One bounded immutable F32 input; only the native encoding refusal changes family."""
    values = np.ascontiguousarray(values, dtype="<f4")
    try:
        return raw.quantize_measured(
            scheme,
            values.tobytes(),
            dtype="F32",
            shape=values.shape,
            strategy=strategy,
            params=params,
            tensor_amax=tensor_amax,
        )
    except tensorfs.DecodeError as error:
        raise UnsupportedInput(f"{key!r}: {error}") from error


# ----------------------------------------------------------------------------- nvfp4

NVFP4_BLOCK = 16


def encode_nvfp4(key: str, values: np.ndarray, amax: float | None = None) -> Encoded:
    """TensorFS's explicit min-SSE NVFP4 producer, exposed through the existing U8 carrier."""
    rows, cols = _require_2d(key, values, NVFP4_BLOCK)
    if not rows:
        # Historical empty results carry only metadata; no tensor arithmetic or native work.
        maximum = 0.0 if amax is None else amax
        if not np.isfinite(maximum):
            raise UnsupportedInput(f"{key!r} contains a non-finite weight — refuse, never encode")
        empty_scale = np.float32(maximum / (E4M3_MAX * 6.0)) if maximum > 0 else np.float32(1.0)
        return Encoded(
            "U8",
            [0, cols // 2],
            b"",
            [
                Companion(".weight_scale", "F8_E4M3", [0, cols // NVFP4_BLOCK], b""),
                Companion(".weight_scale_2", "F32", [], empty_scale.tobytes()),
            ],
            0,
            0.0,
            0.0,
            0.0,
        )
    if not cols:
        raise UnsupportedInput(f"{key!r}: NVFP4 requires nonempty weight columns")
    if amax is None:
        amax = 0.0
        for sl in row_chunks(rows, cols):
            maximum = float(np.max(np.abs(values[sl])))
            if not np.isfinite(maximum):
                raise UnsupportedInput(
                    f"{key!r} contains a non-finite weight — refuse, never encode"
                )
            amax = max(amax, maximum)
    payload: list[bytes] = []
    scales: list[bytes] = []
    second: bytes | None = None
    acc = _Accum()
    for sl in row_chunks(rows, cols):
        encoded, measured = _measured(
            key,
            values[sl],
            "nvfp4",
            "nvfp4-min-sse/1",
            {
                "block_size": NVFP4_BLOCK,
                "global_scale": "multiply",
                "nibble_order": "low_first",
                "scale_layout": "row_major",
            },
            amax,
        )
        payload.append(encoded.parts["weight"].tobytes())
        scales.append(encoded.parts["scale"].tobytes())
        current = encoded.parts["scale_2"].tobytes()
        if second is not None and second != current:
            raise RuntimeError("native NVFP4 chunks changed their shared global scale")
        second = current
        acc.merge(measured)
    assert second is not None
    return Encoded(
        payload_dtype="U8",  # Native F4(R,C) is the same physical U8(R,C/2) byte carrier.
        payload_shape=[rows, cols // 2],
        payload=b"".join(payload),
        companions=[
            Companion(".weight_scale", "F8_E4M3", [rows, cols // NVFP4_BLOCK], b"".join(scales)),
            Companion(".weight_scale_2", "F32", [], second),
        ],
        saturated=acc.sat,
        sse=acc.sse,
        energy=acc.energy,
        max_abs_err=acc.max_abs,
    )


# ---------------------------------------------------------------------- fp8 rowwise


def encode_fp8_rowwise(key: str, values: np.ndarray) -> Encoded:
    """`fp8.rowwise`, rank-1 scale variant: continuous per-output-row f32 scale
    `clamp(rowmax|v|/448, min 1e-12)`, RNE payload, clamp BEFORE the cast."""
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


# ------------------------------------------------------------------- int8 convrot256

#: ConvRot's rotation group: one orthonormal Hadamard rotation per 256 input channels.
CONVROT_GROUP = 256
INT8_MAX = 127.0


def hadamard256(values: np.ndarray) -> np.ndarray:
    """Each 256-group of `[rows, K]` times the regular Hadamard H4⊗H4⊗H4⊗H4, entries ±1 and
    UNNORMALIZED (orthonormal is this / 16). H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
    is symmetric, so H is its own transpose and H·H = 256·I. Additions only: an integer input
    stays an exact integer, which is what makes the decode below exact."""
    rows, cols = values.shape
    work = values.reshape(-1, 4, 4, 4, 4)
    for axis in range(1, 5):
        a, b, c, d = (np.take(work, i, axis=axis) for i in range(4))
        work = np.stack((a + b + c - d, a + b - c + d, a - b + c + d, b + c + d - a), axis=axis)
    return work.reshape(rows, cols)


def decode_int8_convrot256(payload: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """`int8-convrot256/1`'s value law, exactly: `scale[i] * (data[i] · H)[j] / 16` in f64.
    The Hadamard sum of int8 is an exact integer and every factor after it is exact in f64,
    so the only rounding is the caller's one cast into the logical dtype."""
    sums = hadamard256(payload.astype(np.int32)).astype(np.float64)
    return sums * scale.astype(np.float64)[:, None] / 16.0


def encode_int8_convrot256(key: str, values: np.ndarray) -> Encoded:
    """TensorFS's versioned H256/16 rowwise ConvRot producer and measured acceptance facts."""
    rows, cols = _require_2d(key, values, CONVROT_GROUP)
    if not rows:
        return Encoded(
            "I8",
            [0, cols],
            b"",
            [Companion(".weight_scale", "F32", [0], b"")],
            0,
            0.0,
            0.0,
            0.0,
        )
    if not cols:
        raise UnsupportedInput(f"{key!r}: ConvRot requires nonempty weight columns")
    payload: list[bytes] = []
    scales: list[bytes] = []
    acc = _Accum()
    for sl in row_chunks(rows, cols, CACHE_ELEMENTS):
        encoded, measured = _measured(
            key,
            values[sl],
            "int8.convrot",
            "int8-convrot256/1",
            {"rotation": "kron_regular_h4", "rotation_size": CONVROT_GROUP},
        )
        payload.append(encoded.parts["weight"].tobytes())
        scales.append(encoded.parts["scale"].tobytes())
        acc.merge(measured)
    return Encoded(
        payload_dtype="I8",
        payload_shape=[rows, cols],
        payload=b"".join(payload),
        companions=[Companion(".weight_scale", "F32", [rows], b"".join(scales))],
        saturated=acc.sat,
        sse=acc.sse,
        energy=acc.energy,
        max_abs_err=acc.max_abs,
    )


# ------------------------------------------------------------------------------ w4a8

#: comfy-kitchen's frozen Lloyd-Max levels for a group-normalized Gaussian (`_FIXED_LUT`):
#: ConvRot makes every layer's rotated groups Gaussian, so this one table serves (kitchen fits
#: its own only past an excess kurtosis of -0.1, which it notes real models never reach).
W4A8_CODEBOOK = np.array(
    [-0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
     0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671],
    dtype=np.float32,
)  # fmt: skip
W4A8_GROUP = 16


def _nearest_level(levels: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The index of `levels` (ascending) nearest each of `x`, ties to the lower."""
    pos = np.searchsorted(levels, x)
    lo, hi = np.clip(pos - 1, 0, 15), np.clip(pos, 0, 15)
    return np.where(np.abs(x - levels[hi]) < np.abs(x - levels[lo]), hi, lo)


def encode_w4a8(key: str, values: np.ndarray) -> Encoded:
    """`w4a8.convrot` as our producer writes it (design `four-bit-lanes.md` §3-4):
    comfy-kitchen's symmetric codebook quantizer on int8.convrot's rotated rows. Each 16-group's
    scale is refined by two ALS passes over the frozen codebook; the row scale makes the largest
    level 127, the group scale relative to it is stored E4M3; each weight then takes the nearest
    of its group's int8 levels `round(codebook * group_scale)`, the grid the GEMM multiplies."""
    rows, cols = _require_2d(key, values, CONVROT_GROUP)
    groups = cols // W4A8_GROUP
    codes = np.empty((rows, cols // 2), dtype=np.uint8)
    group_scale = np.empty((rows, groups), dtype=np.uint8)
    row_scale = np.empty((rows,), dtype=np.float32)
    book = W4A8_CODEBOOK.astype(np.float64)
    acc = _Accum()
    for sl in row_chunks(rows, cols, CACHE_ELEMENTS):
        v = np.ascontiguousarray(values[sl], dtype=np.float32)
        n = v.shape[0]
        g = (hadamard256(v.astype(np.float64)) / 16.0).reshape(n, groups, W4A8_GROUP)
        gs = np.maximum(np.abs(g).max(axis=-1, keepdims=True), 1e-8)
        idx = _nearest_level(book, g / gs)
        for _ in range(2):
            c = book[idx]
            energy = np.maximum((c * c).sum(-1, keepdims=True), 1e-8)
            gs = np.maximum((g * c).sum(-1, keepdims=True) / energy, 1e-8)
            idx = _nearest_level(book, g / gs)
        amax = np.abs(book[idx] * gs).max(axis=(1, 2))
        scale = np.maximum(amax / INT8_MAX, 1e-8).astype(np.float32)
        rel = e4m3_encode((gs[..., 0] / scale.astype(np.float64)[:, None]).astype(np.float32))
        levels = np.clip(np.rint(W4A8_CODEBOOK * e4m3_decode(rel)[..., None]), -INT8_MAX, INT8_MAX)
        target = g / scale.astype(np.float64)[:, None, None]
        acc.sat += int(np.count_nonzero(np.abs(target) > INT8_MAX + 0.5))
        dist = np.abs(target[..., :, None] - levels[..., None, :].astype(np.float64))
        q = dist.argmin(axis=-1).astype(np.uint8)  # the first of equals: ties to the lower
        q8 = np.take_along_axis(levels, q.astype(np.intp), axis=-1).reshape(n, cols)
        acc.add(v, decode_int8_convrot256(q8, scale))
        codes[sl] = pack_nibbles(q.reshape(n, cols))
        group_scale[sl] = rel
        row_scale[sl] = scale
    return Encoded(
        payload_dtype="I8",
        payload_shape=[rows, cols // 2],
        payload=codes.tobytes(),
        companions=[
            Companion(".weight_s_rel", "F8_E4M3", [rows, groups], group_scale.tobytes()),
            Companion(".weight_s_channel", "F32", [rows], row_scale.tobytes()),
            Companion(".weight_codebook", "F32", [16], W4A8_CODEBOOK.tobytes()),
        ],
        saturated=acc.sat,
        sse=acc.sse,
        energy=acc.energy,
        max_abs_err=acc.max_abs,
    )
