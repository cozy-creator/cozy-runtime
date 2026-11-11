"""Speed, not correctness: everything here is optional, and a package that uses none of it
still serves. Its modules are `kernels`, `fusion`, `stepcache`, `moe`, `lora`, `rows` and
`attention`."""

from cozy_runtime.accel._concurrent import concurrently
from cozy_runtime.accel._spread import spread
from cozy_runtime.author._model import (
    hold_dtype,
    placeable,
    plain_forward,
    sequence_parallel,
)

__all__ = [
    "concurrently",
    "hold_dtype",
    "placeable",
    "plain_forward",
    "sequence_parallel",
    "spread",
]
