"""Compatibility encoders and independent decoders for derived weight formats.

TensorFS owns the versioned offline numerical producers and acceptance statistics.
Their adapters preserve bounded row chunks and the existing Encoded/Companion result contract.
Only thin producer adapters and independent decode helpers remain here.
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


#: byte -> float32, sign in bit 7; the NaN codes 0x7F/0xFF read as the largest magnitude.
E4M3_DECODE32 = np.concatenate([E4M3_MAG32[np.minimum(np.arange(128), 126)]] * 2)
E4M3_DECODE32[128:] *= -1


def e4m3_decode(raw: np.ndarray) -> np.ndarray:
    out: np.ndarray = E4M3_DECODE32.take(np.asarray(raw, dtype=np.uint8), mode="clip")
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
    """TensorFS's versioned F32 row-scale and E4M3 RNE producer."""
    rows, cols = _require_2d(key, values, 1)
    if not rows:
        return Encoded(
            "F8_E4M3",
            [0, cols],
            b"",
            [Companion(".weight_scale", "F32", [0], b"")],
            0,
            0.0,
            0.0,
            0.0,
        )
    if not cols:
        raise UnsupportedInput(f"{key!r}: FP8 requires nonempty weight columns")
    payload: list[bytes] = []
    scales: list[bytes] = []
    acc = _Accum()
    for sl in row_chunks(rows, cols, CACHE_ELEMENTS):
        encoded, measured = _measured(
            key,
            values[sl],
            "fp8.rowwise",
            "fp8-rowwise-rne/1",
            {},
        )
        payload.append(encoded.parts["weight"].tobytes())
        scales.append(encoded.parts["scale"].tobytes())
        acc.merge(measured)
    return Encoded(
        "F8_E4M3",
        [rows, cols],
        b"".join(payload),
        [Companion(".weight_scale", "F32", [rows], b"".join(scales))],
        acc.sat,
        acc.sse,
        acc.energy,
        acc.max_abs,
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

#: Metadata for the historical zero-row result only; nonempty tensors get their codebook
#: and all fitted values from the native producer.
W4A8_CODEBOOK = np.array(
    [-0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
     0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671],
    dtype=np.float32,
)  # fmt: skip
W4A8_GROUP = 16


def encode_w4a8(key: str, values: np.ndarray) -> Encoded:
    """TensorFS's fixed-codebook F64 ConvRot and two-pass group-scale producer."""
    rows, cols = _require_2d(key, values, CONVROT_GROUP)
    if not rows:
        return Encoded(
            "I8",
            [0, cols // 2],
            b"",
            [
                Companion(".weight_s_rel", "F8_E4M3", [0, cols // W4A8_GROUP], b""),
                Companion(".weight_s_channel", "F32", [0], b""),
                Companion(".weight_codebook", "F32", [16], W4A8_CODEBOOK.tobytes()),
            ],
            0,
            0.0,
            0.0,
            0.0,
        )
    if not cols:
        raise UnsupportedInput(f"{key!r}: W4A8 requires nonempty weight columns")
    payload: list[bytes] = []
    scales: list[bytes] = []
    row_scales: list[bytes] = []
    codebook: bytes | None = None
    acc = _Accum()
    for sl in row_chunks(rows, cols, CACHE_ELEMENTS):
        encoded, measured = _measured(
            key,
            values[sl],
            "w4a8.convrot",
            "w4a8-convrot256-fixed/1",
            {
                "group_size": W4A8_GROUP,
                "rotation": "kron_regular_h4",
                "rotation_size": CONVROT_GROUP,
            },
        )
        payload.append(encoded.parts["weight"].tobytes())
        scales.append(encoded.parts["group_scale"].tobytes())
        row_scales.append(encoded.parts["scale"].tobytes())
        current = encoded.parts["codebook"].tobytes()
        if codebook is not None and current != codebook:
            raise RuntimeError("native W4A8 chunks changed their shared codebook")
        codebook = current
        acc.merge(measured)
    assert codebook is not None
    return Encoded(
        "I8",
        [rows, cols // 2],
        b"".join(payload),
        [
            Companion(".weight_s_rel", "F8_E4M3", [rows, cols // W4A8_GROUP], b"".join(scales)),
            Companion(".weight_s_channel", "F32", [rows], b"".join(row_scales)),
            Companion(".weight_codebook", "F32", [16], codebook),
        ],
        acc.sat,
        acc.sse,
        acc.energy,
        acc.max_abs,
    )
