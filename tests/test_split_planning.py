"""`split`'s planning (cr-211): the windows and the finest window are what the scan that walked
every span count planned, now arithmetic, and H3's 15 s FFN plans in milliseconds.

`_reference_*` is the replaced planner verbatim: it rebuilt each candidate count's spans, which
is O(tokens²) at the finest cut (H3 1.45.3's FFN at 136k tokens: minutes, the GPU idle).
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from itertools import pairwise

import pytest

from cozy_runtime.author._split import Cut, _finest, _windows


def _reference_spans(length: int, cut: Cut, count: int) -> list[tuple[int, int]]:
    step = max(-(-max(length - cut.overlap, 1) // count), cut.overlap, 1)
    step = -(-step // cut.align) * cut.align
    starts = range(0, max(length - cut.overlap, 1), step)
    return [(a, min(a + step + cut.overlap, length)) for a in starts]


def _reference_longest(spans: Sequence[tuple[int, int]]) -> int:
    return max(b - a for a, b in spans)


def _reference_largest(
    shape: tuple[int, ...], cuts: tuple[Cut, ...], spans: Sequence[Sequence[tuple[int, int]]]
) -> int:
    rest = math.prod(shape) // math.prod(shape[c.dim] for c in cuts)
    return rest * math.prod(_reference_longest(s) for s in spans)


def _reference_windows(
    shape: tuple[int, ...], cuts: tuple[Cut, ...], most: float
) -> list[tuple[tuple[int, int], ...]]:
    lengths = [shape[c.dim] for c in cuts]
    counts = [1] * len(cuts)
    spans = [_reference_spans(n, c, 1) for n, c in zip(lengths, cuts, strict=True)]
    while _reference_largest(shape, cuts, spans) > most:
        for i in sorted(range(len(cuts)), key=lambda i: -_reference_longest(spans[i])):
            more = next(
                (
                    n
                    for n in range(counts[i] + 1, lengths[i] + 1)
                    if _reference_longest(_reference_spans(lengths[i], cuts[i], n))
                    < _reference_longest(spans[i])
                ),
                0,
            )
            if more:
                counts[i], spans[i] = more, _reference_spans(lengths[i], cuts[i], more)
                break
        else:
            break
    windows: list[tuple[tuple[int, int], ...]] = [()]
    for axis in spans:
        windows = [(*w, s) for w in windows for s in axis]
    return windows


def _reference_finest(shape: tuple[int, ...], cuts: tuple[Cut, ...]) -> int:
    return _reference_largest(
        shape, cuts, list(zip(*_reference_windows(shape, cuts, 0), strict=True))
    )


def _case(rng: random.Random) -> tuple[tuple[int, ...], tuple[Cut, ...]]:
    shape = tuple(rng.randint(1, 90) for _ in range(rng.randint(1, 4)))
    dims = rng.sample(range(len(shape)), rng.randint(1, min(2, len(shape))))
    cuts = tuple(
        Cut(
            d - len(shape) if rng.random() < 0.5 else d,
            scale=rng.choice((0.5, 1.0, 2.0)),
            overlap=rng.choice((0, 0, 1, 3, 6, 12)),
            align=rng.choice((1, 1, 2, 4, 8)),
        )
        for d in dims
    )
    return shape, cuts


def test_windows_and_finest_equal_the_scan_they_replace() -> None:
    rng = random.Random(211)
    for _ in range(1500):
        shape, cuts = _case(rng)
        total = math.prod(shape)
        assert _finest(shape, cuts) == _reference_finest(shape, cuts), (shape, cuts)
        for most in {0, 1, total, total + 1, *(rng.randint(1, total) for _ in range(4))}:
            assert _windows(shape, cuts, most) == _reference_windows(shape, cuts, most), (
                shape,
                cuts,
                most,
            )


@pytest.mark.parametrize("tokens", [68_172, 136_344])
def test_h3_15s_ffn_plans_in_milliseconds(tokens: int) -> None:
    """H3's FFN input, [1, tokens, 5376] cut by tokens: its finest window (one token) and a cut
    to a few thousand tokens, as the memory asks on a small card."""
    shape, cuts = (1, tokens, 5376), (Cut(1),)
    started = time.perf_counter()
    assert _finest(shape, cuts) == 5376
    windows = _windows(shape, cuts, 4096 * 5376)
    assert time.perf_counter() - started < 0.5
    assert max(b - a for ((a, b),) in windows) <= 4096
    assert windows[0][0][0] == 0 and windows[-1][0][1] == tokens
    assert all(w[0][1] == v[0][0] for w, v in pairwise(windows))
