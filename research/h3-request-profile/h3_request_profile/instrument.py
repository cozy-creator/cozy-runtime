"""Bounded profiler around a real H3 request; no alternate kernels or arithmetic."""

from __future__ import annotations

import functools
import gzip
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import msgspec

TRACE_BYTES = 64 << 30


def host_range(label: str) -> dict[str, Any]:
    ident = threading.get_ident()
    native = threading.get_native_id()
    low32 = ident & 0xFFFFFFFF
    signed = low32 if low32 < 0x80000000 else low32 - 0x100000000
    return {
        "label": label,
        "pid": os.getpid(),
        "tid": native,
        "native_tid": native,
        "python_ident": ident,
        "pthread_tid_int32": signed,
        "pthread_tid_uint32": low32,
        "pthread_tid_chrome": abs(signed),
        "start_unix_ns": time.time_ns(),
    }


def close_range(row: dict[str, Any], ranges: list[dict[str, Any]]) -> None:
    row["end_unix_ns"] = time.time_ns()
    ranges.append(row)


@contextmanager
def observe_bakes(ranges: list[dict[str, Any]]) -> Any:
    """Name the existing bake call without tensor hooks or retained tensor arguments."""
    module = import_module("cozy_runtime.internal.lora_bake")
    original = module.bake

    @functools.wraps(original)
    def observed(*args: Any, **kwargs: Any) -> Any:
        row = host_range("h3.weights.lora_bake")
        try:
            return original(*args, **kwargs)
        finally:
            close_range(row, ranges)

    module.bake = observed
    try:
        yield
    finally:
        module.bake = original


class ObservedTelemetry:
    """Retain the original stage service; add only host-clock correlation ranges."""

    def __init__(self, target: Any, ranges: list[dict[str, Any]]) -> None:
        self.target, self.ranges = target, ranges

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    @contextmanager
    def stage(self, name: str, **options: Any) -> Any:
        row = host_range("h3.phase." + name)
        try:
            with self.target.stage(name, **options) as value:
                yield value
        finally:
            close_range(row, self.ranges)


_COUNTERS = (
    "h2d_bytes",
    "disk_copy_bytes",
    "mapped_copy_bytes",
    "bounce_copy_bytes",
    "disk_read_bytes",
    "late",
    "stall_ns",
    "misses",
    "evictions",
    "oom_retries",
    "alloc_retries",
    "alloc_ooms",
    "cache_releases",
    "cache_released_bytes",
    "lora_baked",
)


def counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(after[key]) - int(before[key])
        for key in _COUNTERS
        if key in before and key in after and before[key] >= 0 and after[key] >= before[key]
    }


def category(name: str) -> str:
    lower = name.lower()
    if "memcpy" in lower and any(x in lower for x in ("htod", "host to device", "h2d")):
        return "h2d_copy"
    if "memcpy" in lower and any(x in lower for x in ("dtoh", "device to host", "d2h")):
        return "d2h_copy"
    if "memcpy" in lower:
        return "other_copy"
    if "synchronize" in lower or "waitevent" in lower:
        return "synchronization"
    if "launchkernel" in lower:
        return "kernel_launch"
    if "producer" in lower:
        return "fused_norm_rope_quant_producer"
    if any(x in lower for x in ("rope_norm", "rms_rope", "norm_rope")):
        return "fused_norm_rope"
    if "quant" in lower:
        return "quantization"
    if any(x in lower for x in ("qk_int8", "sol_attn", "flash", "fmha")):
        return "attention"
    if any(x in lower for x in ("gemm", "matmul", "scaled_mm")):
        return "gemm"
    return "unclassified"  # full names remain available; no inferred attribution


def software_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in ("cozy-runtime", "tensorfs", "torch", "triton", "diffusers", "comfy-kitchen"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def model_census(transformer: Any) -> dict[str, Any]:
    """Observe implementation classes without reading tensor storage or adding hooks."""
    classes: dict[str, dict[str, Any]] = {}
    sites = []
    for path, module in transformer.named_modules():
        cls = type(module)
        name = cls.__module__ + "." + cls.__qualname__
        row = classes.setdefault(name, {"count": 0, "examples": []})
        row["count"] += 1
        if len(row["examples"]) < 8:
            row["examples"].append(path)
        if path.endswith((".attn", ".ff", ".ff.net.0", ".ff.net.2")):
            sites.append({"path": path, "class": name})
    return {
        "classes": classes,
        "sites": sites,
        "fusion_report": getattr(transformer, "_cozy_fusion_report", None),
        "scope": "Implementation metadata only; no tensor reads, materialization or hooks",
    }


def retain_trace_file(
    raw: Path, trace_sink: Callable[[Path], Any] | None
) -> tuple[Any, dict[str, int]]:
    """Keep the whole ChromeTrace using bounded copies, including traces larger than1GiB."""
    artifacts = raw.parent / "artifacts"
    artifacts.mkdir()
    zipped = artifacts / "trace.json.gz"
    with (
        raw.open("rb") as source,
        gzip.open(zipped, mode="wb", compresslevel=1) as destination,
    ):
        shutil.copyfileobj(source, destination, length=1 << 20)
    sizes = {"raw_bytes": raw.stat().st_size, "gzip_bytes": zipped.stat().st_size}
    # Production retains through Outputs.save_tree's bounded file copier. Small CPU
    # contract tests may still consume the original byte return shape without a sink.
    trace = trace_sink(artifacts) if trace_sink is not None else zipped.read_bytes()
    return trace, sizes


def retained_handler(name: str) -> Callable[..., Any]:
    """Use the registered implementation, even when managed jobs replace public exports."""
    registration = import_module("h3").app.get(name)
    if registration.kind != "entrypoint" or registration.internal or registration.hidden:
        raise ValueError("Profiling requires a public retained H3 serving handler")
    return cast(Callable[..., Any], registration.fn)


class ObservedOutputs:
    """Delegate the existing sink unchanged; observe its overlapping worker-thread wall."""

    def __init__(self, target: Any, ranges: list[dict[str, Any]] | None = None) -> None:
        self.target = target
        self.sinks: list[dict[str, Any]] = []
        self.ranges = ranges if ranges is not None else []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    def save_video_stream(self, events: Any) -> Any:
        row = host_range("h3.output.save_video_stream")
        began = time.perf_counter()
        source = iter(events)
        source_ms = 0.0
        count = 0
        eof: float | None = None
        succeeded = False

        def observed() -> Any:
            nonlocal source_ms, count, eof
            while True:
                start = time.perf_counter()
                try:
                    event = next(source)
                except StopIteration:
                    eof = time.perf_counter()
                    return
                finally:
                    source_ms += (time.perf_counter() - start) * 1000
                count += 1
                yield event

        try:
            result = self.target.save_video_stream(observed())
            succeeded = True
            return result
        finally:
            ended = time.perf_counter()
            close_range(row, self.ranges)
            total = (ended - began) * 1000
            self.sinks.append(
                {
                    "operation": "Outputs.save_video_stream",
                    "succeeded": succeeded,
                    "wall_ms": total,
                    "source_event_wait_and_materialize_ms": source_ms,
                    "sink_wall_excluding_source_ms": total - source_ms,
                    "after_source_eof_finalize_probe_ms": None
                    if eof is None
                    else (ended - eof) * 1000,
                    "events": count,
                    "overlaps_decode_video": True,
                }
            )


def profile_request(
    model: Any,
    dense_steps: int,
    render: Callable[[Any, Any], Any],
    *,
    outputs: Any,
    telemetry: Any,
    detailed: bool = False,
    full_request_trace: bool = True,
    trace_cpu_ops: bool = False,
) -> tuple[Any, Any, bytes]:
    if outputs._attempt.max_output_bytes < TRACE_BYTES:
        raise ValueError(
            "Full trace requires the declared64GiB output grant before rendering; "
            f"worker granted{outputs._attempt.max_output_bytes} bytes"
        )
    import torch
    from torch.profiler import ProfilerActivity

    if torch.cuda.device_count() != 1:
        raise ValueError("This diagnostic profiles one allocated GPU only")
    transformer = model.pipe.components["fl2va_dit"]
    residency = getattr(model, "_cozy_residency", None)
    if residency is None:
        raise ValueError("H3 profile requires the real Runtime-owned model/weight plane")

    def facts() -> dict[str, Any]:
        return cast(dict[str, Any], msgspec.to_builtins(residency.weights.facts()))

    ranges: list[dict[str, Any]] = []
    observed_outputs = ObservedOutputs(outputs, ranges)
    observed_telemetry = ObservedTelemetry(telemetry, ranges)
    return record_request(
        transformer,
        dense_steps,
        lambda: render(observed_outputs, observed_telemetry),
        facts,
        [ProfilerActivity.CPU, ProfilerActivity.CUDA] if trace_cpu_ops else [ProfilerActivity.CUDA],
        detailed=detailed,
        full_request_trace=full_request_trace,
        attribution=lambda: telemetry._attempt.attribution.document(),
        output_sinks=observed_outputs.sinks,
        trace_sink=outputs.save_tree,
        trace_cpu_ops=trace_cpu_ops,
        ranges=ranges,
        fusion_report=getattr(model, "_cozy_fusion_report", None),
    )


def record_request(
    transformer: Any,
    dense_steps: int,
    render: Callable[[], Any],
    facts: Callable[[], dict[str, Any]],
    activities: list[Any],
    *,
    detailed: bool = False,
    attribution: Callable[[], dict[str, Any]] | None = None,
    output_sinks: list[dict[str, Any]] | None = None,
    full_request_trace: bool = False,
    trace_sink: Callable[[Path], Any] | None = None,
    trace_cpu_ops: bool = True,
    ranges: list[dict[str, Any]] | None = None,
    fusion_report: dict[str, Any] | None = None,
) -> tuple[Any, Any, bytes]:
    """The same bounded recorder can be exercised with real CPU modules off-rental."""
    from torch.profiler import profile, record_function, schedule

    if full_request_trace and detailed:
        raise ValueError("Whole-request tracing requires shape/memory recording off")
    if detailed and not trace_cpu_ops:
        raise ValueError("Detailed operator tracing requires explicit trace_cpu_ops")
    ranges = ranges if ranges is not None else []

    steps: list[dict[str, Any]] = []
    handles: list[Any] = []
    scopes: dict[str, Any] = {}
    step_index = [0]
    before: list[dict[str, Any]] = [{}]
    began = [0.0]
    traces = []
    averages = []
    trace_sizes = []
    census = model_census(transformer)
    if fusion_report is not None:
        census["fusion_report"] = fusion_report
    request_before = facts()
    attribution_before = attribution() if attribution is not None else {}
    profile_began = time.perf_counter()

    def begin_scope(label: str) -> tuple[Any, dict[str, Any]]:
        row = host_range(label)
        marker: Any = record_function(label) if trace_cpu_ops else None
        if marker is not None:
            marker.__enter__()
        return marker, row

    def end_scope(scope: tuple[Any, dict[str, Any]]) -> None:
        marker, row = scope
        if marker is not None:
            marker.__exit__(None, None, None)
        close_range(row, ranges)

    def finished_trace(prof: Any) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-profile-") as temp:
            path = Path(temp) / "trace.json"
            prof.export_chrome_trace(str(path))
            raw_bytes = path.stat().st_size
            if not full_request_trace and raw_bytes > 512 << 20:
                raise ValueError("Two-step profiler trace exceeds512MiB diagnostic bound")
            trace, sizes = retain_trace_file(path, trace_sink)
            traces.append(trace)
            trace_sizes.append(sizes)
        if full_request_trace:
            # The raw trace is authoritative and the streaming offline reader owns
            # aggregation. key_averages constructs Python FunctionEvents for the
            # entire CPU trace, adding another large memory peak after file export.
            return
        for event in prof.key_averages(group_by_input_shape=detailed):
            averages.append(
                {
                    "name": str(event.key),
                    "count": int(event.count),
                    "device_type": str(event.device_type),
                    "self_cpu_us": float(event.self_cpu_time_total),
                    "self_device_us": float(getattr(event, "self_device_time_total", 0.0)),
                    "inclusive_device_us": float(getattr(event, "device_time_total", 0.0)),
                    "input_shapes": event.input_shapes,
                    "category": category(str(event.key)),
                }
            )

    # Full request executes. Trace the last forced-dense and first sparse-eligible calls;
    # a dense-only attention pin can still execute dense kernels in the latter phase.
    # are recorded; exports/teardown and all profiler overhead are diagnostic time.
    with observe_bakes(ranges), profile(
        activities=activities,
        schedule=None
        if full_request_trace
        else schedule(wait=dense_steps - 2, warmup=1, active=2, repeat=1),
        on_trace_ready=None if full_request_trace else finished_trace,
        record_shapes=detailed,
        profile_memory=detailed,
        with_stack=False,
    ) as prof:

        def step_start(*args: Any) -> None:
            before[0], began[0] = facts(), time.perf_counter()
            phase = "dense" if step_index[0] < dense_steps else "sparse_eligible"
            scopes["step"] = begin_scope(f"h3.step.{step_index[0]}.{phase}")

        def step_end(*args: Any) -> None:
            marker = scopes.pop("step", None)
            if marker is not None:
                end_scope(marker)
            after = facts()
            steps.append(
                {
                    "step": step_index[0],
                    "policy_phase": "dense" if step_index[0] < dense_steps else "sparse_eligible",
                    "cpu_enqueue_wall_ms": (time.perf_counter() - began[0]) * 1000,
                    "plane_before": before[0],
                    "plane_after": after,
                    "plane_delta": counter_delta(before[0], after),
                }
            )
            step_index[0] += 1
            prof.step()

        handles += [
            transformer.register_forward_pre_hook(step_start),
            transformer.register_forward_hook(step_end, always_call=True),
        ]
        for path, module in transformer.named_modules():
            detailed_site = path.endswith(
                (
                    ".attn",
                    ".to_q",
                    ".to_k",
                    ".to_v",
                    ".norm_q",
                    ".norm_k",
                    ".to_out.0",
                    ".ff.net.0.proj",
                    ".ff.net.2",
                )
            )
            block = (path.startswith("transformer_blocks.") and path.count(".") == 1) or (
                "refiner_blocks." in path and path.rsplit(".", 1)[-1].isdigit()
            )
            coarse_site = block or path.endswith((".attn", ".ff", ".ff.net.0", ".ff.net.2"))
            if not (coarse_site or (trace_cpu_ops and detailed_site)):
                continue

            def enter(*args: Any, label: str = path) -> None:
                scopes.setdefault(label, []).append(begin_scope("h3.module." + label))

            def leave(*args: Any, label: str = path) -> None:
                stack = scopes.get(label, [])
                if stack:
                    end_scope(stack.pop())

            handles += [
                module.register_forward_pre_hook(enter),
                module.register_forward_hook(leave, always_call=True),
            ]
        try:
            render_began = time.perf_counter()
            result = render()
            render_wall_ms = (time.perf_counter() - render_began) * 1000
            request_after = facts()
            attribution_after = attribution() if attribution is not None else {}
        finally:
            for handle in handles:
                handle.remove()
            for remaining in scopes.values():
                for marker in remaining if isinstance(remaining, list) else [remaining]:
                    end_scope(marker)
    if full_request_trace:
        finished_trace(prof)
    if len(traces) != 1:
        raise ValueError(f"Expected one profiler trace, captured{len(traces)}")
    summary = {
        "diagnostic_only": True,
        "software": software_versions(),
        "model_census": census,
        "bake_ranges": "Original lora_bake.bake calls; tensor-free wrapper restored on exit",
        "record_shapes_and_memory": detailed,
        "not_benchmark_time": True,
        "full_request_trace": full_request_trace,
        "trace_cpu_ops": trace_cpu_ops,
        "projection_hooks": "enabled for explicit CPU tracing"
        if trace_cpu_ops
        else "QKV/out omitted",
        "ranges": ranges,
        "thread_id_note": (
            "Linux native_tid and Python get_ident plus signed/unsigned low32 aliases are "
            "retained. CUDA-only Kineto may use pthread low32 instead of native_tid; "
            "verify the actual runtime-event join rather than assuming either ID."
        ),
        "profiled_step_indices": list(range(len(steps)))
        if full_request_trace
        else [dense_steps - 1, dense_steps],
        "trace_format": "tree/trace.json.gz"
        if trace_sink is not None
        else "pytorch_chrome_json_gzip",
        "trace_sizes": trace_sizes,
        "original_timestamps_preserved": True,
        "trace_collection": "continuous whole handler" if full_request_trace else "two DiT calls",
        "pre_handler_construction_and_warmup_in_trace": False,
        "steps": steps,
        "events": averages,
        "operator_aggregation": "offline from raw trace" if full_request_trace else "key_averages",
        "render_call": {
            "wall_ms": render_wall_ms,
            "profiler_context_wall_ms": (time.perf_counter() - profile_began) * 1000,
            "plane_before": request_before,
            "plane_after": request_after,
            "plane_delta": counter_delta(request_before, request_after),
            "attribution_before": attribution_before,
            "attribution_after": attribution_after,
            "scope": ("Entire H3 handler: prepare, TE, denoise, both VAEs, sink and output checks"),
        },
        "output_sinks": output_sinks or [],
        "inclusive_times_must_not_be_summed": True,
        "kernel_category_is_heuristic": True,
        "notes": (
            "Default ranges cover steps/blocks/attention/FFN and delegated H3 phases. "
            "QKV/output projection hooks and CPU ATen recording require explicit trace_cpu_ops; "
            "default CUDA-only profiling preserves shared-QKV eligibility. "
            "Kernel names/input shapes identify child operations. CUDA memcpy spans "
            "and plane deltas distinguish transfers/eviction from compute. CPU enqueue "
            "walls are not GPU execution times; raw trace preserves stream overlap. "
            "Plane counters are asynchronous boundary snapshots; inspect cumulative "
            "request deltas and correlated CUDA transfers before assigning a step cost. "
            "Detailed shape/memory mode retains tensor references and can perturb paging; "
            "it is off by default and must not be treated as an unperturbed baseline. "
            "Attribution is the existing Runtime document, not separately timed pipeline stages; "
            "stage work clocks exclude loading-weight asides. Sink wall overlaps video decode; "
            "source next() includes waiting, digesting and materializing decoded events. "
            "The post-EOF sink tail includes encoder flush, mux, probe and output registration. "
            "Machine preparation before this handler and final artifact commit/export after it "
            "must be read from ordinary run-show evidence; reused setup is not charged again. "
            "Sparse-eligible labels describe the schedule, not proof of a sparse kernel; "
            "actual kernel identity comes from the trace and native run evidence."
            " Full mode continuously records CUDA activity through TE, all DiT steps, "
            "both VAEs, encode and checks; "
            "record_shapes/profile_memory/with_stack remain off. Pre-handler construction/warm "
            "kernels remain outside this authored trace and retain native setup timing evidence. "
            " Plane deltas start at handler entry, after cold construction; raw before/after "
            "snapshots preserve the cumulative counters without charging earlier requests again."
            " The trace is streamed to gzip and retained by Outputs.save_tree; no full compressed "
            "byte buffer or read_bytes is used by the authored profiling path. CPU event buffers "
            "and the temporary raw ChromeTrace still scale with request length."
        ),
    }
    return result, traces[0], json.dumps(summary, separators=(",", ":"), allow_nan=False).encode()
