"""The checkpoint push window follows observed throughput and keeps its best level."""

from __future__ import annotations

import pytest

from cozy_runtime.internal.worker import machine_publication


@pytest.mark.parametrize(
    "seconds,levels",
    [((1.0, 4.0, 0.25), (32, 16, 16)), ((1.0, 1.0, 1.0), (32, 64, 64))],
)
def test_push_window_follows_observed_throughput_and_keeps_the_best_level(
    monkeypatch: pytest.MonkeyPatch,
    seconds: tuple[float, ...],
    levels: tuple[int, ...],
) -> None:
    """Exercise the real policy with known byte/rate observations, apart from TLS noise."""
    now = 0.0

    class Clock:
        @staticmethod
        def monotonic() -> float:
            return now

    # Control only this policy's clock; the integration above uses real transport
    # and clocks. No production callback or replacement pacing policy is involved.
    monkeypatch.setattr(machine_publication, "time", Clock)
    pace = machine_publication._Pace(16, 64)
    pace.started()
    now += 0.25
    pace.ended(16, None)
    assert (pace.level, pace.count, pace.settled) == (16, 0, False)

    for duration, expected in zip(seconds, levels, strict=True):
        level = pace.level
        for _ in range(level):
            pace.started()
        now += duration
        for _ in range(level):
            pace.ended(level, (1 << 20, duration))
        assert pace.level == expected and pace.running == 0
    assert pace.settled and pace.best[1] == levels[-1]
