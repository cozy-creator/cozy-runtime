"""Seed to noise, one way for every model: drawn on the CPU, so a seed makes the same image on any
device, the CPU degraded mode included."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def generator(seed: int) -> torch.Generator:
    """The request's random stream: a CPU generator seeded with `seed`. Diffusers pipelines and
    schedulers take it for any device."""
    return torch.Generator().manual_seed(seed)


def noise(
    shape: Sequence[int],
    seed: int | torch.Generator,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Standard normal noise of `shape` in `dtype`, drawn on the CPU from `seed` (or from a stream
    `generator` made), then moved to `device`."""
    source = generator(seed) if isinstance(seed, int) else seed
    return torch.randn(tuple(shape), generator=source, dtype=dtype).to(device)
