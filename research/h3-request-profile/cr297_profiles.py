"""Replacement for b41d433fc's measurement-only graphs/profiles.py; never a runtime hook.

Copy beside profile_intervals.py into that diagnostic build only. Device activity is
unioned per device inside the annotated call, including its completion synchronization.
Aggregate device work can exceed the window. Inactivity does not identify its cause.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from .profile_intervals import interval_summary

if TYPE_CHECKING:
    from types import ModuleType

PROFILES: dict[str, dict[str, object]] = {}
_SEEN: dict[str, int] = {}
_LOCK = threading.Lock()
_WINDOW = "cozy.cr297.profiled-call"


def profiled[T](torch: ModuleType, label: str, run: Callable[[], T], *, at: int = 2) -> T:
    """Profile the `at`th call; report each GPU separately in the profiler's clock."""
    with _LOCK:
        n = _SEEN[label] = _SEEN.get(label, 0) + 1
    if n != at or label in PROFILES:
        return run()
    from torch.profiler import ProfilerActivity, profile, record_function

    devices = range(torch.cuda.device_count())
    for device in devices:
        torch.cuda.synchronize(device)
    began = time.perf_counter()
    with (
        profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof,
        record_function(_WINDOW),
    ):
        result = run()
        for device in devices:
            torch.cuda.synchronize(device)
    record: dict[str, object] = {
        "profiler_context_wall_ms": (time.perf_counter() - began) * 1e3,
    }
    try:
        events = list(prof.events())
        windows = [
            event
            for event in events
            if event.name == _WINDOW and str(event.device_type).endswith("CPU")
        ]
        if len(windows) != 1:
            raise ValueError(f"Expected one measurement annotation, found {len(windows)}")
        window = (float(windows[0].time_range.start), float(windows[0].time_range.end))
        bounds = interval_summary([], window=window)
        intervals: dict[int, list[tuple[float, float]]] = {device: [] for device in devices}
        by_name: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for event in events:
            if not str(event.device_type).endswith("CUDA"):
                continue
            device = int(event.device_index)
            start, end = float(event.time_range.start), float(event.time_range.end)
            intervals.setdefault(device, []).append((start, end))
            duration = max(0.0, min(end, window[1]) - max(start, window[0]))
            if duration:
                by_name[device][event.name] += duration / 1e3
        activity = []
        for device, values in sorted(intervals.items()):
            summary = interval_summary(values, window=window)
            activity.append(
                {
                    "device_index": device,
                    "event_count": summary["event_count"],
                    "busy_union_ms": summary["busy_union_us"] / 1e3,
                    "aggregate_device_work_ms": summary["duration_sum_us"] / 1e3,
                    "observed_span_ms": summary["span_us"] / 1e3,
                    "unclassified_inactivity_ms": summary["unclassified_inactivity_us"] / 1e3,
                    "top_aggregate_device_work_ms": sorted(
                        by_name[device].items(), key=lambda item: -item[1]
                    )[:8],
                }
            )
        record.update(
            measurement_window={
                "source": _WINDOW,
                "clock": "PyTorch FunctionEvent.time_range, microseconds from trace start",
                "start_us": window[0],
                "end_us": window[1],
                "duration_ms": bounds["window_duration_us"] / 1e3,
            },
            device_activity=activity,
        )
    except Exception as exc:  # A diagnostic failure must not discard a completed model call.
        record["error"] = f"{type(exc).__name__}: {exc}"[:240]
    PROFILES[label] = record
    return result


def hook_children(torch: ModuleType, component: str, root: object) -> None:
    """Profile the first warm call of each `decoder`/`encoder` child of a component root (a
    VAE's chunks pass through them, never through the root's region)."""
    for name in ("decoder", "encoder"):
        child = getattr(root, name, None)
        if not isinstance(child, torch.nn.Module) or getattr(child, "_cozy_profiled", False):
            continue
        call = type(child).__call__

        def wrapped(
            self: object,
            *args: object,
            _call: Callable[..., object] = call,
            _label: str = f"{component}.{name}",
            **kwargs: object,
        ) -> object:
            shapes = tuple(tuple(a.shape) for a in args if isinstance(a, torch.Tensor))
            return profiled(torch, f"{_label} {shapes}", lambda: _call(self, *args, **kwargs))

        child.__class__ = type(
            type(child).__name__, (type(child),), {"__call__": wrapped, "_cozy_profiled": True}
        )
