"""Bounded profiler around a real H3 request; no alternate kernels or arithmetic."""

from __future__ import annotations

import gzip
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import msgspec

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
    if "memcpy" in lower:
        return "other_copy"
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


class ObservedOutputs:
    """Delegate the existing sink unchanged; observe its overlapping worker-thread wall."""

    def __init__(self, target: Any) -> None:
        self.target = target
        self.sinks: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)

    def save_video_stream(self, events: Any) -> Any:
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
    render: Callable[[Any], Any],
    *,
    outputs: Any,
    telemetry: Any,
    detailed: bool = False,
) -> tuple[Any, bytes, bytes]:
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

    observed_outputs = ObservedOutputs(outputs)
    return record_request(
        transformer,
        dense_steps,
        lambda: render(observed_outputs),
        facts,
        [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        detailed=detailed,
        attribution=lambda: telemetry._attempt.attribution.document(),
        output_sinks=observed_outputs.sinks,
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
) -> tuple[Any, bytes, bytes]:
    """The same bounded recorder can be exercised with real CPU modules off-rental."""
    from torch.profiler import profile, record_function, schedule

    steps: list[dict[str, Any]] = []
    handles: list[Any] = []
    scopes: dict[str, Any] = {}
    step_index = [0]
    before: list[dict[str, Any]] = [{}]
    began = [0.0]
    traces = []
    averages = []
    request_before = facts()
    attribution_before = attribution() if attribution is not None else {}
    profile_began = time.perf_counter()

    def finished_trace(prof: Any) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-profile-") as temp:
            path = Path(temp) / "trace.json"
            prof.export_chrome_trace(str(path))
            if path.stat().st_size > 512 << 20:
                raise ValueError("Two-step profiler trace exceeds512MiB diagnostic bound")
            traces.append(gzip.compress(path.read_bytes(), compresslevel=3))
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
    with profile(
        activities=activities,
        schedule=schedule(wait=dense_steps - 2, warmup=1, active=2, repeat=1),
        on_trace_ready=finished_trace,
        record_shapes=detailed,
        profile_memory=detailed,
        with_stack=False,
    ) as prof:

        def step_start(*args: Any) -> None:
            before[0], began[0] = facts(), time.perf_counter()
            phase = "dense" if step_index[0] < dense_steps else "sparse_eligible"
            marker: Any = record_function(f"h3.step.{step_index[0]}.{phase}")
            marker.__enter__()
            scopes["step"] = marker

        def step_end(*args: Any) -> None:
            marker = scopes.pop("step", None)
            if marker is not None:
                marker.__exit__(None, None, None)
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
            if not path.endswith(
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
            ):
                continue

            def enter(*args: Any, label: str = path) -> None:
                marker: Any = record_function("h3.module." + label)
                marker.__enter__()
                scopes.setdefault(label, []).append(marker)

            def leave(*args: Any, label: str = path) -> None:
                stack = scopes.get(label, [])
                if stack:
                    stack.pop().__exit__(None, None, None)

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
                    marker.__exit__(None, None, None)
    if len(traces) != 1:
        raise ValueError(f"Expected one bounded two-step trace, captured{len(traces)}")
    summary = {
        "diagnostic_only": True,
        "record_shapes_and_memory": detailed,
        "not_benchmark_time": True,
        "profiled_step_indices": [dense_steps - 1, dense_steps],
        "steps": steps,
        "events": averages,
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
            "QKV ranges include original encoded-leaf quantization and LoRA hooks. "
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
        ),
    }
    return result, traces[0], json.dumps(summary, separators=(",", ":"), allow_nan=False).encode()
