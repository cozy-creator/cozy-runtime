"""Media clock conversion accepts PyAV's exact rational representations."""

from __future__ import annotations

from fractions import Fraction

from cozy_runtime.author._decode import _fraction


class _PyAVRational:
    """Shape used by PyAV releases that do not return ``fractions.Fraction``."""

    numerator = 30_000
    denominator = 1_001


def test_fraction_preserves_pyav_rational_exactly() -> None:
    assert _fraction(_PyAVRational(), "video average rate") == Fraction(30_000, 1_001)
