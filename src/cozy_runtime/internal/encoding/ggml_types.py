"""ggml's block types (cozytensors spec §6, `ggml-common.h`): the one table every reader of a
`ggml.*` tensor sizes it by, and ggml's own type ids for the ones it multiplies."""

from __future__ import annotations

#: Each type: elements and bytes per block.
STRUCTS = {
    "q4_0": (32, 18),
    "q4_1": (32, 20),
    "q5_0": (32, 22),
    "q5_1": (32, 24),
    "q8_0": (32, 34),
    "q2_k": (256, 84),
    "q3_k": (256, 110),
    "q4_k": (256, 144),
    "q5_k": (256, 176),
    "q6_k": (256, 210),
    "mxfp4": (32, 17),
    "nvfp4": (64, 36),
    "q1_0": (128, 18),
    "q2_0": (64, 18),
}

#: The schemes TensorFS writes (`ggml-ref/1`) and `_kernels._ggml` multiplies: ggml's type id.
WRITTEN = {
    "ggml.q8_0": 8,
    "ggml.q5_0": 6,
    "ggml.q5_1": 7,
    "ggml.q4_k": 12,
    "ggml.q5_k": 13,
    "ggml.q6_k": 14,
}


def block(scheme: str) -> int:
    """Elements per block of a `ggml.<type>` scheme."""
    return STRUCTS[scheme.removeprefix("ggml.")][0]


def nbytes(scheme: str, rows: int, columns: int) -> int:
    """A [rows, columns] tensor's bytes in `scheme`."""
    n, size = STRUCTS[scheme.removeprefix("ggml.")]
    return rows * (columns // n) * size
