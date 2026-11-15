"""Device interval accounting shared by offline traces and the cr-297 diagnostic adapter."""

from __future__ import annotations

import math
from typing import Any


def interval_summary(
    intervals: list[tuple[float, float]], *, window: tuple[float, float] | None = None
) -> dict[str, Any]:
    """Union intervals on one device, across streams; timestamps and bounds are microseconds.

    An explicit window intersects each event before accounting, retaining only positive
    intersections. Its inactivity includes both tails; it has no causal attribution.
    Without bounds, only observed first-to-last span and interior gaps are available.
    """
    for start, end in [*intervals, *([] if window is None else [window])]:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Interval bounds must be finite and end must not precede start")
    ordered = sorted(
        (start, end)
        for start, end in (
            (max(left, window[0]), min(right, window[1])) if window else (left, right)
            for left, right in intervals
        )
        if end > start
    )
    result: dict[str, Any] = {
        "event_count": len(ordered),
        "duration_sum_us": sum(right - left for left, right in ordered),
        "span_us": 0.0,
        "busy_union_us": 0.0,
        "interior_gap_total_us": 0.0,
        "interior_gap_count": 0,
        "interior_gap_p50_us": 0.0,
        "interior_gap_p95_us": 0.0,
        "longest_interior_gaps": [],
    }
    if ordered:
        start, end = ordered[0]
        busy, gaps = 0.0, []
        for left, right in ordered[1:]:
            if left > end:
                busy += end - start
                gaps.append({"start_us": end, "end_us": left, "duration_us": left - end})
                start = left
            end = max(end, right)
        busy += end - start
        span = end - ordered[0][0]
        durations = sorted(row["duration_us"] for row in gaps)
        result.update(
            first_start_us=ordered[0][0],
            last_end_us=end,
            span_us=span,
            busy_union_us=busy,
            interior_gap_total_us=span - busy,
            interior_gap_count=len(gaps),
            interior_gap_p50_us=durations[(len(durations) - 1) // 2] if durations else 0.0,
            interior_gap_p95_us=durations[math.ceil(len(durations) * 0.95) - 1]
            if durations
            else 0.0,
            longest_interior_gaps=sorted(gaps, key=lambda row: -row["duration_us"])[:50],
        )
    if window is not None:
        result.update(
            window_start_us=window[0],
            window_end_us=window[1],
            window_duration_us=window[1] - window[0],
            unclassified_inactivity_us=window[1] - window[0] - result["busy_union_us"],
        )
    return result
