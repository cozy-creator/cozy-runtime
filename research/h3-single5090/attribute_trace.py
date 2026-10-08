"""Attribute GPU activities to CPU annotation ranges through CUDA correlation IDs.

Use three streaming passes; keep only CPU APIs referenced by GPU activities. GPU
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


def manual_annotations(
    rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> Iterable[dict[str, Any]]:
    """Map externally recorded host ranges into the retained Kineto clock domain."""
    base = metadata.get("baseTimeNanoseconds")
    if not isinstance(base, int):
        raise ValueError("Manual Unix ranges require trace baseTimeNanoseconds")
    owners: dict[tuple[str, int], set[tuple[Any, Any]]] = defaultdict(set)
    for row in rows:
        identity = (row.get("native_tid", row.get("tid")), row.get("python_thread_ident"))
        for key in (
            "tid",
            "native_tid",
            "pthread_tid_int32",
            "pthread_tid_uint32",
            "pthread_tid_chrome",
        ):
            if isinstance(row.get(key), int):
                owners[str(row["pid"]), row[key]].add(identity)
    for row in rows:
        start, end = row["start_unix_ns"], row["end_unix_ns"]
        if not isinstance(start, int) or not isinstance(end, int) or end < start:
            raise ValueError("Invalid manual range timestamps")
        if "pid" not in row or "tid" not in row:
            raise ValueError("Manual ranges must identify their host process and thread")
        aliases = {
            row[key]
            for key in (
                "tid",
                "native_tid",
                "pthread_tid_int32",
                "pthread_tid_uint32",
                "pthread_tid_chrome",
            )
            if isinstance(row.get(key), int)
        }
        for tid in sorted(aliases):
            if len(owners[str(row["pid"]), tid]) != 1:
                raise ValueError("Manual ranges contain colliding thread aliases")
            yield {
                "ph": "X",
                "cat": "user_annotation",
                "name": row["label"],
                "pid": row["pid"],
                "tid": tid,
                "ts": (start - base) / 1000,
                "dur": (end - start) / 1000,
            }


def attribute(
    events: Callable[[], Iterable[dict[str, Any]]],
) -> dict[str, Any]:
    annotations: dict[tuple[str, str], list[tuple[float, float, str]]] = defaultdict(list)
    # CUDA correlation identifiers are normally unique within one process. A
    # multi-process trace with collisions is reported as ambiguous, never guessed.
    launches: dict[str, tuple[str, str, float] | None] = {}
    duplicates: set[str] = set()
    host_processes: set[str] = set()
    referenced: set[str] = set()
    for event in events():
        times = interval(event)
        if times is None:
            continue
        category = event.get("cat")
        thread = str(event.get("pid")), str(event.get("tid"))
        if category == "user_annotation":
            annotations[thread].append((*times, str(event.get("name", ""))))
        elif category in {"kernel", "gpu_kernel", "gpu_memcpy", "gpu_memset"}:
            key = correlation(event)
            if key is not None:
                referenced.add(key)
    for event in events():
        times = interval(event)
        if times is None:
            continue
        thread = str(event.get("pid")), str(event.get("tid"))
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}:
            key = correlation(event)
            if key is None or key not in referenced:
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
        "gpu_referenced_correlation_count": len(referenced),
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
    parser.add_argument(
        "--ranges", type=Path, help="CUDA-only capture summary with manual host ranges"
    )
    options = parser.parse_args()
    spec = importlib.util.spec_from_file_location("h3_trace_stream", options.parser)
    if spec is None or spec.loader is None:
        raise ValueError("Trace parser cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata: dict[str, Any] = {}
    manual = json.loads(options.ranges.read_text())["ranges"] if options.ranges else []

    def events() -> Iterable[dict[str, Any]]:
        yield from module.trace_events(options.trace, metadata)
        if options.ranges:
            yield from manual_annotations(manual, metadata)

    result = attribute(events)
    result["trace"] = str(options.trace)
    result["trace_metadata"] = metadata
    result["manual_ranges_source"] = str(options.ranges) if options.ranges else None
    result["manual_range_count"] = len(manual)
    result["manual_thread_aliases"] = "Native and captured pthread IDs; collisions refuse"
    options.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}))


if __name__ == "__main__":
    main()
