"""Interleaved A/B timing on CUDA.

Arms alternate every repetition (and the order flips every other repetition) so thermal
drift and clock ramp land on both arms equally instead of becoming the result.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from typing import Any

import torch


def _time_once(fn: Callable[[], Any], inner: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(inner):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / inner


def interleaved(
    arms: Sequence[tuple[str, Callable[[], Any]] | tuple[str, Callable[[], Any], Any]],
    reps: int = 21,
    inner: int = 3,
    warmup: int = 5,
) -> dict[str, dict[str, float]]:
    """Each arm is (name, fn) or (name, fn, setup). `setup` runs OUTSIDE the timed region,
    so arms that share one module tree can swap their plan in without paying for it."""
    normalized = [(a[0], a[1], a[2] if len(a) > 2 else None) for a in arms]
    for _, fn, setup in normalized:
        if setup is not None:
            setup()
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {name: [] for name, _, _ in normalized}
    for rep in range(reps):
        order = list(normalized) if rep % 2 == 0 else list(reversed(normalized))
        for name, fn, setup in order:
            if setup is not None:
                setup()
                fn()
                torch.cuda.synchronize()
            samples[name].append(_time_once(fn, inner))

    out: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        values = sorted(values)
        out[name] = {
            "median_ms": statistics.median(values),
            "min_ms": values[0],
            "p25_ms": values[len(values) // 4],
            "p75_ms": values[(3 * len(values)) // 4],
            "n": float(len(values)),
        }
    return out


def report(title: str, results: dict[str, dict[str, float]], baseline: str | None = None) -> None:
    print(f"\n=== {title}")
    names = list(results)
    base = baseline or names[0]
    b = results[base]["median_ms"]
    width = max(len(n) for n in names)
    for name in names:
        r = results[name]
        speed = b / r["median_ms"]
        print(
            f"  {name:<{width}}  median {r['median_ms']:9.4f} ms  "
            f"[p25 {r['p25_ms']:8.4f} p75 {r['p75_ms']:8.4f}]  x{speed:5.3f} vs {base}"
        )


def device_note() -> str:
    p = torch.cuda.get_device_properties(0)
    return (
        f"{p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}, "
        f"torch {torch.__version__}"
    )
