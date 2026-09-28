"""Every numeric quality gate refuses NaN and inf.

`error > bound` is False for NaN and `max()` drops a NaN that is not its first operand, so a
decoder, kernel or conversion writing NaN used to pass. The graders below are the real
ones; numpy's `finfo` stands in for torch's (same fields), which is all they read.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cozy_runtime.author._numeric import within, worst, worst_of
from cozy_runtime.internal import probe
from cozy_runtime.internal.probe import e4m3_value

POISON = (math.nan, math.inf, -math.inf)


def test_helpers_pass_only_finite_values_within_their_bound() -> None:
    assert within(0.5, 1.0) and within(1.0, 1.0) and not within(1.5, 1.0)
    assert not any(within(value, 1.0) for value in POISON)
    assert math.isnan(worst(0.1, math.nan)) and math.isnan(worst(math.nan, 0.1))
    assert worst(0.1, 0.3) == 0.3
    assert worst_of({"first": 0.1, "last": math.nan})[0] == "last"
    assert worst_of({"first": 0.1, "edge": 0.4})[0] == "edge"


SCALE = 0.013671875
EPS16 = float(np.finfo(np.float16).eps)


def _scaled() -> list[float]:
    return [e4m3_value(byte) * SCALE for byte in range(256)]


def test_scaled_fp8_grading_refuses_poison_and_nonzero_zeros() -> None:
    assert probe._grade_scaled(EPS16, _scaled(), SCALE).passed
    normal = 0x38  # e4m3 1.0
    for value in POISON:
        got = _scaled()
        got[normal] = value
        assert not probe._grade_scaled(EPS16, got, SCALE).passed, value
    garbage_zero = _scaled()
    garbage_zero[0x00] = 1e-3
    assert not probe._grade_scaled(EPS16, garbage_zero, SCALE).passed


def test_an_all_nan_decoder_fails_both_decode_checks() -> None:
    everywhere = [math.nan] * 256
    assert not probe._grade_scaled(EPS16, everywhere, SCALE).passed
    sweep = probe._domain_sweep(np, np.float32, everywhere, [(1.0, "s")])
    assert not sweep.passed and sweep.wrong is not None and sweep.wrong["normal"] > 0


def _domain(dtype: type[np.floating], scale: float) -> list[float]:
    info = np.finfo(dtype)
    half_ulp = float(info.tiny) * float(info.eps) / 2.0
    out = []
    for byte in range(256):
        want = e4m3_value(byte) * scale
        if math.isnan(want):
            out.append(want)
        elif abs(want) > float(info.max):
            out.append(math.copysign(math.inf, want))
        elif want != 0.0 and abs(want) < half_ulp:
            out.append(0.0)
        else:
            out.append(want)
    return out


@pytest.mark.parametrize(
    ("dtype", "scale", "band"),
    [(np.float32, 1.0, "normal"), (np.float16, 2.0**-20, "subnormal")],
)
def test_domain_sweep_refuses_poison_in_every_band(
    dtype: type[np.floating], scale: float, band: str
) -> None:
    exact = _domain(dtype, scale)
    assert probe._domain_sweep(np, dtype, exact, [(scale, "s")]).passed
    tiny = float(np.finfo(dtype).tiny)
    index = next(
        i
        for i, want in enumerate(e4m3_value(b) * scale for b in range(256))
        if want > 0 and (want < tiny) == (band == "subnormal") and want == exact[i]
    )
    for value in POISON:
        got = list(exact)
        got[index] = value
        sweep = probe._domain_sweep(np, dtype, got, [(scale, "s")])
        assert not sweep.passed and sweep.wrong is not None and sweep.wrong[band] == 1, value
