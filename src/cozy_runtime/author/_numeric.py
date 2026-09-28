"""Numeric quality gates: a measurement passes only when it is finite and within its bound.

`error > bound` is False for NaN, and `max()` keeps whichever operand came first when the
other is NaN, so both forms let a decoder, a kernel or a conversion that writes NaN pass.
FA3 FP8 once passed an attention probe while producing garbage video. Every gate and every
running worst goes through these two functions instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def within(measured: float, bound: float) -> bool:
    """Pass only a finite measurement no larger than `bound`; NaN and inf always refuse."""
    return math.isfinite(measured) and measured <= bound


def worst(current: float, measured: float) -> float:
    """The larger of two measurements, where NaN is worse than any number."""
    return math.nan if math.isnan(current) or math.isnan(measured) else max(current, measured)


def worst_of(measured: Mapping[str, float]) -> tuple[str, float]:
    """The worst named measurement, where NaN is worse than any number."""
    name = max(measured, key=lambda key: math.inf if math.isnan(measured[key]) else measured[key])
    return name, measured[name]
