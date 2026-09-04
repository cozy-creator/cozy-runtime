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


def measure(arms: Arms, name: str) -> tuple[int, float, Counter[str], dict[str, float]]:
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
    micros: dict[str, float] = {}
    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        launches += int(event.count)
        total_us += float(event.self_device_time_total)
        names[event.key] += int(event.count)
        micros[event.key] = micros.get(event.key, 0.0) + float(event.self_device_time_total)
    return launches, total_us / 1000.0, names, micros


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
        launches, ms, names, micros = measure(arms, name)
        if base is None:
            base = (launches, ms)
            print(f"  {name:22s} {launches:5d} kernels   {ms:8.3f} ms device")
        else:
            print(
                f"  {name:22s} {launches:5d} kernels ({launches - base[0]:+d})   "
                f"{ms:8.3f} ms device (x{base[1] / ms:.3f})"
            )
        if args.top:
            ranked = sorted(micros.items(), key=lambda kv: -kv[1])[: args.top]
            for key, us in ranked:
                share = 100.0 * us / max(ms * 1000.0, 1e-9)
                print(f"        {us / 1000.0:8.2f} ms {share:5.1f}%  x{names[key]:<4d} {key[:78]}")


if __name__ == "__main__":
    main()
