"""Attribute GPU activities to CPU annotation ranges through CUDA correlation IDs.

Use the full ChromeTrace twice; do not load all CPU operators into memory. GPU
execution can outlive its launching CPU annotation, so GPU timestamps alone are
not an attribution rule. Durations here are sums, never elapsed critical paths.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Ranges:
    rows: list[tuple[float, float, str]]

    def __post_init__(self) -> None:
        self.rows.sort(key=lambda row: (row[0], -row[1], row[2]))
        self.starts = [row[0] for row in self.rows]
        self.size = 1 << max(0, (len(self.rows) - 1).bit_length())
        self.ends = [-math.inf] * (2 * self.size)
        for index, (_, end, _) in enumerate(self.rows):
            self.ends[self.size + index] = end
        for index in range(self.size - 1, 0, -1):
            self.ends[index] = max(self.ends[index * 2], self.ends[index * 2 + 1])

    def enclosing(self, timestamp: float) -> tuple[str, ...]:
        last = bisect_right(self.starts, timestamp) - 1
        found = []
        pending = [(1, 0, self.size)]
        while pending:
            index, left, right = pending.pop()
            if left > last or self.ends[index] <= timestamp:
                continue
            if right - left == 1:
                found.append(self.rows[left][2])
                continue
            middle = (left + right) // 2
            pending.append((index * 2 + 1, middle, right))
            pending.append((index * 2, left, middle))
        return tuple(found)


def interval(event: dict[str, Any]) -> tuple[float, float] | None:
    start, duration = event.get("ts"), event.get("dur")
    if event.get("ph") != "X" or not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in (start, duration)
    ):
        return None
    if duration < 0:
        return None
    return float(start), float(start + duration)


def args(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("args", {})
    return value if isinstance(value, dict) else {}


def correlation(event: dict[str, Any]) -> str | None:
    value = args(event).get("correlation")
    return str(value) if isinstance(value, (int, str)) else None


def attribute(
    events: Callable[[], Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    annotations: dict[tuple[str, str], list[tuple[float, float, str]]] = defaultdict(list)
    # CUDA correlation identifiers are normally unique within one process. A
    # multi-process trace with collisions is reported as ambiguous, never guessed.
    launches: dict[str, tuple[str, str, float] | None] = {}
    duplicates: set[str] = set()
    host_processes: set[str] = set()
    for event in events():
        times = interval(event)
        if times is None:
            continue
        category = event.get("cat")
        thread = str(event.get("pid")), str(event.get("tid"))
        if category == "user_annotation":
            annotations[thread].append((*times, str(event.get("name", ""))))
        elif category in {"cuda_runtime", "cuda_driver"}:
            key = correlation(event)
            if key is None:
                continue
            host_processes.add(thread[0])
            if key in launches:
                # Runtime/driver nesting can emit the same ID twice. Requiring
                # one observed API event avoids silently choosing a parent.
                launches[key] = None
                duplicates.add(key)
            else:
                launches[key] = (*thread, times[0])
    ranges = {thread: Ranges(rows) for thread, rows in annotations.items()}
    outcomes: Counter[str] = Counter()
    rows: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}
    invalid_gpu = 0
    for event in events():
        category = str(event.get("cat", ""))
        if category not in {"kernel", "gpu_kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        times = interval(event)
        if times is None:
            invalid_gpu += 1
            continue
        key = correlation(event)
        launch = launches.get(key) if key is not None else None
        path: tuple[str, ...] = ()
        if key is None:
            outcome = "gpu_without_correlation"
        elif key in duplicates:
            outcome = "ambiguous_correlation"
        elif launch is None:
            outcome = "missing_cpu_api"
        else:
            thread_ranges = ranges.get(launch[:2])
            if thread_ranges is not None:
                path = thread_ranges.enclosing(launch[2])
            outcome = "annotated" if path else "api_without_annotation"
        outcomes[outcome] += 1
        device = str(args(event).get("device", f"trace_pid:{event.get('pid')}"))
        group = (category, device, str(event.get("name", "")), path)
        row = rows.setdefault(
            group,
            {
                "category": category,
                "device": device,
                "kernel_or_copy": group[2],
                "annotations": list(path),
                "count": 0,
                "duration_sum_us": 0.0,
                "outcomes": Counter(),
            },
        )
        row["count"] += 1
        row["duration_sum_us"] += times[1] - times[0]
        row["outcomes"][outcome] += 1
    return {
        "method": "GPU correlation -> unique CPU CUDA API -> annotations on its CPU thread",
        "host_processes": sorted(host_processes),
        "annotation_count": sum(len(value.rows) for value in ranges.values()),
        "cpu_api_correlation_count": len(launches),
        "ambiguous_cpu_correlations": len(duplicates),
        "gpu_attribution_counts": dict(outcomes),
        "invalid_gpu_events": invalid_gpu,
        "groups": sorted(rows.values(), key=lambda row: -row["duration_sum_us"]),
        "limitations": [
            "Duration sums include stream overlap; they are not elapsed or critical-path times.",
            "Nested annotation paths are retained as one group, not summed as independent phases.",
            "Uncorrelated events stay in totals; no GPU timestamp-containment fallback is used.",
            "Cross-thread asynchronous work may lack its originating annotation.",
            "No kernels outside this process's capture window are implied to be recorded.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--parser", type=Path, required=True, help="Path to PR1243 trace_summary.py"
    )
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    spec = importlib.util.spec_from_file_location("h3_trace_stream", options.parser)
    if spec is None or spec.loader is None:
        raise ValueError("Trace parser cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = attribute(lambda: module.trace_events(options.trace))
    result["trace"] = str(options.trace)
    options.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}))


if __name__ == "__main__":
    main()
