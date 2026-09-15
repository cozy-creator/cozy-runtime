"""Declared keepdim checkpoint geometry for the standard-weight fixtures."""

from collections.abc import Sequence

from tensorfs.derived import Part, Tensor

from cozy_runtime.internal.encoding.formats import SPEC_ROWWISE_KEEPDIM


def rowwise_spec(shape: Sequence[int], *, dtype: str = "bf16") -> Tensor:
    return Tensor(
        dtype,
        tuple(shape),
        SPEC_ROWWISE_KEEPDIM,
        {"data": Part("f8_e4m3fn", tuple(shape)), "scale": Part("f32", (shape[0], 1))},
    )
