"""DIAGNOSTIC ONLY (never merged): each invocation's GPU kernel timeline and host Python stacks.

Writes `<tmp>/cozy-profile-<request>.json`: the kernels' busy time over the invocation window,
the top kernels, and the host's sampled stacks (every thread but this sampler, every 2 ms).
"""

from __future__ import annotations

import collections
import json
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def profiled(request_id: str) -> Iterator[None]:
    try:
        import torch
    except ImportError:
        yield
        return
    if not torch.cuda.is_available():
        yield
        return
    samples: collections.Counter[tuple[str, ...]] = collections.Counter()
    leaves: collections.Counter[str] = collections.Counter()
    stop = threading.Event()
    me = threading.get_ident

    def sample() -> None:
        own = me()
        while not stop.wait(0.002):
            for ident, frame in sys._current_frames().items():
                if ident == own:
                    continue
                stack = []
                f = frame
                while f is not None and len(stack) < 40:
                    code = f.f_code
                    stack.append(f"{Path(code.co_filename).name}:{code.co_name}:{f.f_lineno}")
                    f = f.f_back
                if any("threading.py:wait" in s or "selectors.py" in s for s in stack[:2]):
                    continue
                samples[tuple(stack[:12])] += 1
                for s in stack[:40]:
                    leaves[s.rsplit(":", 1)[0]] += 1

    sampler = threading.Thread(target=sample, daemon=True)
    started = time.perf_counter()
    activities = [torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        sampler.start()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            stop.set()
    wall = time.perf_counter() - started
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    spans = sorted((e.time_range.start, e.time_range.end) for e in events)
    busy, end = 0.0, -1.0
    for s, e in spans:
        if s > end:
            busy += e - s
            end = e
        elif e > end:
            busy += e - end
            end = e
    first = spans[0][0] if spans else 0.0
    last = max((e for _, e in spans), default=0.0)
    kernels: collections.Counter[str] = collections.Counter()
    counts: collections.Counter[str] = collections.Counter()
    for e in events:
        kernels[e.name[:100]] += e.time_range.end - e.time_range.start
        counts[e.name[:100]] += 1
    total = sum(samples.values()) or 1
    document: dict[str, Any] = {
        "wall_ms": round(wall * 1e3, 1),
        "gpu_span_ms": round((last - first) / 1e3, 1),
        "gpu_busy_ms": round(busy / 1e3, 1),
        "kernel_count": len(events),
        "top_kernels_ms": [[k, round(v / 1e3, 2), counts[k]] for k, v in kernels.most_common(25)],
        "host_samples": total,
        "host_inclusive_pct": [[k, round(100 * v / total, 1)] for k, v in leaves.most_common(80)],
        "host_stacks": [[list(k), v] for k, v in samples.most_common(25)],
    }
    out = Path(tempfile.gettempdir()) / f"cozy-profile-{request_id}.json"
    out.write_text(json.dumps(document, indent=1))
    print(f"[diag] profile written to {out}", file=sys.stderr)
