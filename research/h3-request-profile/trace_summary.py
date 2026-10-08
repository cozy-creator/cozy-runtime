"""Summarize native or Comfy Torch ChromeTrace JSON/gzip without summing stream overlap."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO


class JSONStream:
    """Decode one value at a time, so the full raw ChromeTrace need not remain in memory."""

    def __init__(self, source: TextIO) -> None:
        self.source, self.buffer, self.pos, self.eof = source, "", 0, False
        self.decoder = json.JSONDecoder()

    def more(self) -> None:
        chunk = self.source.read(1 << 20)
        self.buffer = self.buffer[self.pos :] + chunk
        self.pos, self.eof = 0, not chunk

    def peek(self) -> str:
        while True:
            while self.pos < len(self.buffer) and self.buffer[self.pos].isspace():
                self.pos += 1
            if self.pos < len(self.buffer):
                return self.buffer[self.pos]
            if self.eof:
                raise ValueError("Unexpected end of ChromeTrace JSON")
            self.more()

    def take(self, expected: str) -> None:
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"Expected {expected!r} in ChromeTrace, found {actual!r}")
        self.pos += 1

    def value(self) -> Any:
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.pos)
                if end == len(self.buffer) and not self.eof:
                    self.more()  # a number at the chunk boundary may not be complete
                    continue
                self.pos = end
                return value
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.more()


def trace_events(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as source:
        reader = JSONStream(source)

        def array() -> Iterator[dict[str, Any]]:
            reader.take("[")
            while reader.peek() != "]":
                event = reader.value()
                if not isinstance(event, dict):
                    raise ValueError("ChromeTrace event must be an object")
                yield event
                if reader.peek() != "]":
                    reader.take(",")
            reader.take("]")

        if reader.peek() == "[":
            yield from array()
            return
        reader.take("{")
        found = False
        while reader.peek() != "}":
            key = reader.value()
            reader.take(":")
            if key == "traceEvents":
                if found:
                    raise ValueError("Duplicate traceEvents field")
                found = True
                yield from array()
            else:
                reader.value()
            if reader.peek() != "}":
                reader.take(",")
        reader.take("}")
        if not found:
            raise ValueError("ChromeTrace has no traceEvents field")


def event_kind(event: dict[str, Any]) -> str:
    category = str(event.get("cat", "")).lower()
    name = str(event.get("name", "")).lower()
    if category in {"kernel", "gpu_kernel"}:
        return "device_kernel"
    if category in {"gpu_memcpy", "gpu_memset"}:
        for patterns, kind in (
            (("htod", "h2d", "host to device"), "device_copy_h2d"),
            (("dtoh", "d2h", "device to host"), "device_copy_d2h"),
            (("dtod", "d2d", "device to device"), "device_copy_d2d"),
        ):
            if any(pattern in name for pattern in patterns):
                return kind
        return "device_memset" if category == "gpu_memset" else "device_copy_other"
    if category in {"cuda_runtime", "cuda_driver"}:
        if "synchronize" in name or "waitevent" in name:
            return "host_synchronization_api"
        if "launch" in name:
            return "host_launch_api"
        if "memcpy" in name or "memset" in name:
            return "host_copy_api"
        return "host_other_cuda_api"
    if category == "cpu_op":
        return "host_operator"
    if category == "user_annotation":
        return "host_annotation"
    return "other"


def interval_summary(intervals: list[tuple[float, float]]) -> dict[str, Any]:
    """Union occupied intervals; interior gaps exclude unobserved leading/trailing time."""
    ordered = sorted(intervals)
    if not ordered:
        return {"event_count": 0, "busy_union_us": 0.0, "interior_gap_total_us": 0.0}
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
    return {
        "event_count": len(ordered),
        "first_start_us": ordered[0][0],
        "last_end_us": end,
        "span_us": span,
        "duration_sum_us": sum(right - left for left, right in ordered),
        "busy_union_us": busy,
        "interior_gap_total_us": span - busy,
        "interior_gap_count": len(gaps),
        "interior_gap_p50_us": durations[(len(durations) - 1) // 2] if durations else 0.0,
        "interior_gap_p95_us": durations[math.ceil(len(durations) * 0.95) - 1]
        if durations
        else 0.0,
        "longest_interior_gaps": sorted(gaps, key=lambda row: -row["duration_us"])[:50],
    }


def summarize(events: Iterator[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    device_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    stream_intervals: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    launches: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    categories: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    labels = []
    invalid, total = 0, 0
    for event in events:
        total += 1
        phases[str(event.get("ph", ""))] += 1
        if event.get("ph") == "M":
            if event.get("name") in {"process_name", "thread_name"}:
                labels.append(event)
            continue
        if event.get("ph") != "X":
            continue
        start, duration = event.get("ts"), event.get("dur")
        if (
            not isinstance(start, (float, int))
            or not isinstance(duration, (float, int))
            or not math.isfinite(start)
            or not math.isfinite(duration)
            or duration < 0
        ):
            invalid += 1
            continue
        kind, name = event_kind(event), str(event.get("name", ""))
        categories[kind] += 1
        args = event.get("args", {})
        if not isinstance(args, dict):
            args = {}
        device = str(args.get("device", f"trace_pid:{event.get('pid')}"))
        interval = (float(start), float(start + duration))
        if kind.startswith("device_"):
            stream = str(args.get("stream", f"trace_tid:{event.get('tid')}"))
            device_intervals[device].append(interval)
            stream_intervals[device, stream].append(interval)
        else:
            device = "host"
        if kind == "host_launch_api":
            launches[str(event.get("pid")), str(event.get("tid"))].append(interval)
        key = kind, device, name
        row = groups.setdefault(
            key,
            {
                "kind": kind,
                "device": device,
                "name": name,
                "count": 0,
                "duration_sum_us": 0.0,
                "duration_max_us": 0.0,
            },
        )
        row["count"] += 1
        row["duration_sum_us"] += duration
        row["duration_max_us"] = max(row["duration_max_us"], duration)
        size = args.get("bytes")
        if kind.startswith("device_copy_") and isinstance(size, int) and size >= 0:
            row["bytes_sum"] = row.get("bytes_sum", 0) + size
            row["events_with_bytes"] = row.get("events_with_bytes", 0) + 1
    for row in groups.values():
        row["duration_mean_us"] = row["duration_sum_us"] / row["count"]
    return {
        "trace_event_count": total,
        "trace_phases": dict(phases),
        "complete_event_categories": dict(categories),
        "invalid_complete_events_skipped": invalid,
        "events_by_name": sorted(groups.values(), key=lambda row: -row["duration_sum_us"]),
        "device_activity": [
            {"device": device, **interval_summary(values)}
            for device, values in sorted(device_intervals.items())
        ],
        "stream_activity": [
            {"device": device, "stream": stream, **interval_summary(values)}
            for (device, stream), values in sorted(stream_intervals.items())
        ],
        "host_launch_activity": [
            {"pid": pid, "tid": tid, **interval_summary(values)}
            for (pid, tid), values in sorted(launches.items())
        ],
        "process_thread_labels": labels,
        "units": "microseconds, as specified for ChromeTrace ts/dur; timestamps unchanged",
        "notes": [
            "Host operator/annotation durations are inclusive and overlap; do not add them.",
            "Device name totals are aggregate kernel/copy duration, not elapsed request time.",
            "Device activity unions all observed kernel/copy streams; streams can overlap.",
            "Gaps cover only first-to-last observed activity, excluding unobserved tails.",
            "A device gap is unoccupied traced time, not proof of CPU launch or paging cause.",
            "Host launch gaps describe API call timing, not GPU execution or queue latency.",
            "Complete X duration events are aggregated; metadata/flows are counted separately.",
            "Missing device/stream IDs fall back to explicitly labeled trace pid/tid.",
            "Profiler overhead and activities outside the recorded process remain unmeasured.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(trace_events(args.trace))
    with args.trace.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    result["source"] = {
        "path": str(args.trace),
        "bytes": args.trace.stat().st_size,
        "sha256": digest,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
