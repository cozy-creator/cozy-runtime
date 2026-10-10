"""Trace arithmetic and actual CPU-profiler format; no inferred GPU speed claims."""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
from pathlib import Path
from types import ModuleType

import pytest


def summarizer() -> ModuleType:
    path = Path(__file__).parents[1] / "research/h3-request-profile/trace_summary.py"
    spec = importlib.util.spec_from_file_location("h3_trace_summary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event(
    name: str, category: str, start: float, duration: float,
    *, device: int = 0, stream: int = 7, **args: object,
) -> dict[str, object]:
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "pid": 20,
        "tid": 30,
        "ts": start,
        "dur": duration,
        "args": {"device": device, "stream": stream, **args},
    }


def test_union_keeps_devices_streams_and_host_launches_separate() -> None:
    module = summarizer()
    events = [
        event("kernel A", "kernel", 10, 10),
        event("kernel B", "kernel", 15, 10, stream=8),
        event("Memcpy HtoD", "gpu_memcpy", 30, 5, bytes=4096),
        event("kernel A", "kernel", 10, 50, device=1),
        event("cudaLaunchKernel", "cuda_runtime", 0, 1),
        event("cudaLaunchKernel", "cuda_runtime", 3, 1),
        event("cudaMemcpyAsync", "cuda_runtime", 5, 2),
        event("cudaStreamSynchronize", "cuda_runtime", 7, 3),
        event("aten::outer", "cpu_op", 0, 50),
        event("aten::inner", "cpu_op", 10, 20),
    ]
    result = module.summarize(iter(events))
    first, second = result["device_activity"]
    assert first["device"] == "0" and second["device"] == "1"
    assert first["busy_union_us"] == 20 and first["duration_sum_us"] == 25
    assert first["span_us"] == 25 and first["interior_gap_total_us"] == 5
    assert second["busy_union_us"] == 50  # not added into GPU0's busy time
    stream = result["stream_activity"][0]
    assert stream["busy_union_us"] == 15 and stream["interior_gap_total_us"] == 10
    launch = result["host_launch_activity"][0]
    assert launch["busy_union_us"] == 2 and launch["interior_gap_total_us"] == 2
    rows = {row["name"]: row for row in result["events_by_name"]}
    assert rows["cudaLaunchKernel"]["count"] == 2
    assert rows["cudaMemcpyAsync"]["kind"] == "host_copy_api"
    assert rows["cudaStreamSynchronize"]["kind"] == "host_synchronization_api"
    assert rows["Memcpy HtoD"]["bytes_sum"] == 4096
    assert rows["Memcpy HtoD"]["kind"] == "device_copy_h2d"


@pytest.mark.parametrize("compressed,wrapped", [(False, True), (True, True), (False, False)])
def test_reads_standard_object_or_array_with_optional_gzip(
    tmp_path: Path, compressed: bool, wrapped: bool,
) -> None:
    module = summarizer()
    events = [event("kernel λ", "kernel", 101, 3)]
    document = {"metadata_before": [1, 2], "traceEvents": events, "displayTimeUnit": "ms"}
    data = json.dumps(document if wrapped else events).encode()
    path = tmp_path / "trace.data"  # magic bytes determine gzip, not the extension
    path.write_bytes(gzip.compress(data) if compressed else data)
    result = list(module.trace_events(path))
    assert result == events
    assert module.summarize(iter(result))["device_activity"][0]["first_start_us"] == 101


def test_streamed_numbers_and_strings_cross_read_boundaries() -> None:
    class TinyReads(io.StringIO):
        def read(self, size: int | None = -1) -> str:
            return super().read(1 if size is None else min(size, 1))

    reader = summarizer().JSONStream(TinyReads(' 123456,"hello λ" '))
    assert reader.value() == 123456
    reader.take(",")
    assert reader.value() == "hello λ"


def test_invalid_durations_and_non_duration_phases_are_visible(tmp_path: Path) -> None:
    module = summarizer()
    bad = event("bad", "kernel", 1, -1)
    result = module.summarize(iter([bad, {"ph": "s", "name": "flow"}]))
    assert result["invalid_complete_events_skipped"] == 1
    assert result["trace_phases"] == {"X": 1, "s": 1}
    assert not result["device_activity"]
    path = tmp_path / "broken.json"
    path.write_text('{"traceEvents":[{"name":')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        list(module.trace_events(path))


def test_actual_cpu_chrome_trace_keeps_host_work_without_inventing_device_work(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    module = summarizer()
    with (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof,
        torch.profiler.record_function("h3.step.0.dense"),
    ):
        torch.ones(8, 8).sin()
    path = tmp_path / "cpu.json"
    prof.export_chrome_trace(str(path))
    result = module.summarize(module.trace_events(path))
    rows = {row["name"]: row for row in result["events_by_name"]}
    assert rows["aten::sin"]["kind"] == "host_operator"
    assert rows["h3.step.0.dense"]["kind"] == "host_annotation"
    assert not result["device_activity"] and not result["host_launch_activity"]


def test_multiple_segments_exclude_export_pauses_and_retain_time_base(tmp_path: Path) -> None:
    module = summarizer()
    paths = []
    for index, start in enumerate((10, 1010)):
        path = tmp_path / f"trace-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "baseTimeNanoseconds": 1000000,
                    "traceEvents": [event("same kernel", "kernel", start, 10)],
                }
            )
        )
        paths.append(path)
    result = module.summarize_files(paths)
    device = result["recorded_device_activity"][0]
    assert device["recorded_busy_union_us_sum"] == 20
    assert device["recorded_span_us_sum"] == 20
    assert device["within_segment_gap_us_sum"] == 0  # 1000us export gap is not model idle
    assert result["events_by_name"][0]["count"] == 2
    assert result["segments"][0]["trace_metadata"]["baseTimeNanoseconds"] == 1000000
