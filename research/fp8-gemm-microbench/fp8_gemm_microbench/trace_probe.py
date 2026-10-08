"""Small GPU-only capture capability test; does not time or claim H3 performance."""

from collections import Counter


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
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        for _ in range(repeats):
            with record_function(labels[0]):
                a.copy_(host, non_blocking=True)
            with record_function(labels[1]):
                result = a @ b
            with record_function(labels[2]):
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
