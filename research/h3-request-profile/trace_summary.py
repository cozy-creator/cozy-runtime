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


def trace_events(path: Path, metadata: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
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
                value = reader.value()
                if metadata is not None and key in {
                    "schemaVersion",
                    "baseTimeNanoseconds",
                    "displayTimeUnit",
                    "deviceProperties",
                    "trace_id",
                    "traceName",
                    "host_name",
                }:
                    metadata[key] = value
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
        "units": (
            "ChromeTrace ts/dur microseconds, unchanged; Kineto absolute Unix time is "
            "ts + trace_metadata.baseTimeNanoseconds/1000 when that header is present"
        ),
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


def summarize_file(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    result = summarize(trace_events(path, metadata))
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    result["source"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
    result["trace_metadata"] = metadata
    return result


def summarize_files(paths: list[Path]) -> dict[str, Any]:
    if len(paths) == 1:
        return summarize_file(paths[0])
    segments = [summarize_file(path) for path in paths]
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    devices: dict[str, dict[str, Any]] = {}
    for segment in segments:
        for source in segment["events_by_name"]:
            key = source["kind"], source["device"], source["name"]
            row = rows.setdefault(
                key,
                dict(
                    source,
                    count=0,
                    duration_sum_us=0,
                    duration_max_us=0,
                    bytes_sum=0,
                    events_with_bytes=0,
                ),
            )
            for field in ("count", "duration_sum_us", "bytes_sum", "events_with_bytes"):
                row[field] += source.get(field, 0)
            row["duration_max_us"] = max(row["duration_max_us"], source["duration_max_us"])
            row["duration_mean_us"] = row["duration_sum_us"] / row["count"]
        for source in segment["device_activity"]:
            row = devices.setdefault(
                source["device"],
                {
                    "device": source["device"],
                    "segment_count": 0,
                    "event_count": 0,
                    "recorded_busy_union_us_sum": 0,
                    "recorded_span_us_sum": 0,
                    "within_segment_gap_us_sum": 0,
                },
            )
            row["segment_count"] += 1
            row["event_count"] += source["event_count"]
            row["recorded_busy_union_us_sum"] += source["busy_union_us"]
            row["recorded_span_us_sum"] += source["span_us"]
            row["within_segment_gap_us_sum"] += source["interior_gap_total_us"]
    return {
        "segments": segments,
        "events_by_name": sorted(rows.values(), key=lambda row: -row["duration_sum_us"]),
        "recorded_device_activity": list(devices.values()),
        "inter_segment_gaps_excluded": True,
        "notes": [
            "Each segment retains its own metadata, device/stream unions and interior gaps.",
            "No first-to-last union/gap crosses segment boundaries or profiler-export pauses.",
            "Recorded busy sums assume disjoint profiler segments, not overlapping captures.",
            "The collection's unrecorded pauses are not assigned to model/kernel work.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    paths: list[Path] = []
    for path in args.trace:
        candidates = list(path.glob("trace*.json.gz")) if path.is_dir() else [path]
        if path.is_dir() and not candidates:
            candidates = list(path.glob("trace*.json"))
        paths.extend(
            candidate.resolve()
            for candidate in sorted(candidates)
            if candidate.resolve() != args.out.resolve()
        )
    paths = list(dict.fromkeys(paths))
    if not paths:
        parser.error("No trace files found")
    result = summarize_files(paths)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
