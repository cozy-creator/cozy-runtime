"""Kernel launches and device time per arm — the load-independent half of the evidence.

A shared box makes wall-clock noisy; the number of kernels a forward launches, and the
sum of their device durations, do not move with someone else's job.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).parent))
from bench_model import Arms


def measure(arms: Arms, name: str) -> tuple[int, float, Counter[str]]:
    arms.install(name)
    for _ in range(3):
        arms.call()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        arms.call()
        torch.cuda.synchronize()
    launches = 0
    total_us = 0.0
    names: Counter[str] = Counter()
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        launches += int(event.count)
        total_us += float(event.self_device_time_total)
        names[event.key] += int(event.count)
    return launches, total_us / 1000.0, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--latent-hw", type=int, default=128)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("arms", nargs="+")
    args = ap.parse_args()

    arms = Arms(args.layers, args.latent_hw)
    print(f"num_layers={args.layers}, latent {args.latent_hw}^2, bf16")
    base: tuple[int, float] | None = None
    for name in args.arms:
        launches, ms, names = measure(arms, name)
        if base is None:
            base = (launches, ms)
            print(f"  {name:22s} {launches:5d} kernels   {ms:8.3f} ms device")
        else:
            print(
                f"  {name:22s} {launches:5d} kernels ({launches - base[0]:+d})   "
                f"{ms:8.3f} ms device (x{base[1] / ms:.3f})"
            )
        if args.top:
            for key, count in names.most_common(args.top):
                print(f"        {count:4d}  {key[:90]}")


if __name__ == "__main__":
    main()
