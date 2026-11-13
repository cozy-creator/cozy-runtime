"""cr-297 measurement only (never merged): one warm call of each kind profiled for its kernels'
busy time against its wall time, so the record says where a call's time goes and how much of it
is launch overhead (wall less busy). Reported in every component's graphs record (`describe`)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from types import ModuleType

T = TypeVar("T")
#: label -> wall ms, kernel busy ms, kernel count, the top kernels by busy ms
PROFILES: dict[str, dict[str, object]] = {}
_SEEN: dict[str, int] = {}
_LOCK = threading.Lock()


def profiled(torch: ModuleType, label: str, run: Callable[[], T], *, at: int = 2) -> T:
    """`run`, profiled when it is the `at`th call under `label` (the first warm one)."""
    with _LOCK:
        n = _SEEN[label] = _SEEN.get(label, 0) + 1
    if n != at or label in PROFILES:
        return run()
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    began = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = run()
        torch.cuda.synchronize()
    wall = (time.perf_counter() - began) * 1e3
    busy, kernels, top = 0.0, 0, []
    try:
        for event in prof.events():
            if str(getattr(event, "device_type", "")).endswith("CUDA"):
                us = float(getattr(event, "device_time", 0.0) or getattr(event, "self_device_time_total", 0.0) or 0.0)
                busy += us / 1e3
                kernels += 1
        by = {}
        for row in prof.key_averages():
            us = float(getattr(row, "self_device_time_total", 0.0) or 0.0)
            if us:
                by[row.key[:80]] = by.get(row.key[:80], 0.0) + us / 1e3
        top = sorted(by.items(), key=lambda kv: -kv[1])[:8]
    except Exception as exc:  # measurement must never fail a call
        top = [(f"{type(exc).__name__}: {exc}"[:120], 0.0)]
    PROFILES[label] = {
        "wall_ms": round(wall, 2),
        "busy_ms": round(busy, 2),
        "kernels": kernels,
        "top": [(k, round(v, 2)) for k, v in top],
    }
    return result


def hook_children(torch: ModuleType, component: str, root: object) -> None:
    """Profile the first warm call of each `decoder`/`encoder` child of a component root (a
    VAE's chunks pass through them, never through the root's region)."""
    for name in ("decoder", "encoder"):
        child = getattr(root, name, None)
        if not isinstance(child, torch.nn.Module) or getattr(child, "_cozy_profiled", False):
            continue
        call = type(child).__call__

        def wrapped(self: object, *args: object, _call: Callable[..., object] = call,
                    _label: str = f"{component}.{name}", **kwargs: object) -> object:
            shapes = tuple(tuple(a.shape) for a in args if isinstance(a, torch.Tensor))
            return profiled(torch, f"{_label} {shapes}", lambda: _call(self, *args, **kwargs))

        child.__class__ = type(type(child).__name__, (type(child),), {"__call__": wrapped, "_cozy_profiled": True})
