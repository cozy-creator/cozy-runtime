"""Small GPU-only capture capability test; does not time or claim H3 performance."""

import os
import threading
import time
from collections import Counter
from contextlib import contextmanager


def thread_identity():
    ident = threading.get_ident()
    return {
        "pid": os.getpid(),
        "python_thread_ident": ident,
        "native_tid": threading.get_native_id(),
        "pthread_low32": ident & 0xFFFFFFFF,
    }


def capture(path, dimension, repeats):
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    generator = torch.Generator(device="cpu").manual_seed(20261008)
    host = torch.randn((dimension, dimension), generator=generator).pin_memory()
    received = torch.empty_like(host, pin_memory=True)
    a = host.to("cuda")
    b = host.to("cuda")
    for _ in range(3):
        result = a @ b
    torch.cuda.synchronize()
    labels = ["probe/h2d", "probe/matmul", "probe/d2h"]
    capture_thread = thread_identity()
    ranges = []

    @contextmanager
    def span(name):
        identity = thread_identity()
        start_unix_ns, start_perf_ns = time.time_ns(), time.perf_counter_ns()
        with record_function(name):
            yield
        ranges.append(
            {
                "name": name,
                **identity,
                "start_unix_ns": start_unix_ns,
                "end_unix_ns": time.time_ns(),
                "start_perf_ns": start_perf_ns,
                "end_perf_ns": time.perf_counter_ns(),
            }
        )

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(repeats):
            with span(labels[0]):
                a.copy_(host, non_blocking=True)
            with span(labels[1]):
                result = a @ b
            with span(labels[2]):
                received.copy_(result, non_blocking=True)
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    return summarize(
        path,
        labels,
        {
            "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "dimension": dimension,
            "repeats": repeats,
            "capture_thread": capture_thread,
            "manual_ranges": ranges,
            "manual_range_semantics": "Host enqueue intervals; no per-range CUDA synchronization",
        },
    )


def summarize(path, labels, facts):
    import json

    document = json.loads(path.read_text())
    events = document.get("traceEvents", [])
    categories = Counter(event.get("cat", "") for event in events)
    kernels = [event for event in events if event.get("cat") == "kernel"]
    copies = [
        event
        for event in events
        if "memcpy" in event.get("cat", "").lower() or "memcpy" in event.get("name", "").lower()
    ]
    runtime = [event for event in events if event.get("cat") == "cuda_runtime"]
    recorded = [label for label in labels if any(event.get("name") == label for event in events)]
    correlation = Counter(
        key
        for event in events
        for key in event.get("args", {})
        if "correlation" in key.lower() or "external id" in key.lower()
    )
    return {
        **facts,
        "activities": ["CUDA"],
        "trace_bytes": path.stat().st_size,
        "events": len(events),
        "categories": dict(categories),
        "thread_ids_by_category": {
            category: sorted(
                {
                    event["tid"]
                    for event in events
                    if event.get("cat", "") == category and "tid" in event
                },
                key=str,
            )
            for category in categories
        },
        "trace_timebase_fields": {
            key: value
            for key, value in document.items()
            if "time" in key.lower() and isinstance(value, (int, float, str))
        },
        "kernel_events": len(kernels),
        "memcpy_events": len(copies),
        "cuda_runtime_events": len(runtime),
        "requested_labels": labels,
        "recorded_labels": recorded,
        "missing_labels": sorted(set(labels) - set(recorded)),
        "correlation_field_counts": dict(correlation),
        "kernel_names": sorted({event.get("name", "") for event in kernels}),
        "memcpy_names": sorted({event.get("name", "") for event in copies}),
        "scope": "Capability probe only; no H3 benchmark, no claim of full CPU coverage",
    }
