"""Seed to noise, one way for every model, the same on any device, the CPU degraded mode included.

`noise` draws a request's noise on the CPU and moves it. `counter_noise` and `counter_uniform`
compute a step's noise where it is used, from (seed, step, stream) alone (`internal.philox`): no
generator state, no table and no host read, so a `denoise_loop` step draws them, eager or
recorded."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from cozy_runtime.internal import philox


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


def counter_seed(seed: int, device: torch.device | str) -> torch.Tensor:
    """`seed` as the one-element int64 tensor `counter_noise` takes. A keyed `denoise_loop` passes
    it in `inputs`, so its kept program serves every seed; a host int seed is baked into it."""
    return torch.tensor([philox.int64(seed)], dtype=torch.int64, device=device)


def counter_noise(
    shape: Sequence[int],
    seed: int | torch.Tensor,
    step: int | torch.Tensor,
    *,
    stream: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Standard normal noise of `shape`: a function of (`seed`, `step`, `stream`), computed in
    float32 on `device` and cast to `dtype`.

    Inside a `denoise_loop` step, `step` is the step's index `i`, the int64 device tensor a
    recorded program advances on each replay, and `seed` is a `counter_seed` tensor the loop
    takes in `inputs` (or a host int its `key` names). Either may be a host int or a one-element
    int64 tensor; the values are the same. `stream`, 32 bits, keeps one step's draws apart.
    `device` defaults to a tensor argument's device, else the CPU.

    Element n of the flattened result depends on (seed, step, stream, n) alone, bit for bit, in
    any shape holding it, eager or recorded, at any CPU thread count, and on CPU and CUDA alike
    by construction: Philox words in integer ops, then Box-Muller in float32 operations that are
    exact or correctly rounded (frexp, add, multiply, sqrt; log, sine and cosine are Cephes
    polynomials). On CUDA one kernel writes the result and holds nothing else; elsewhere torch
    ops hold about 30 bytes per element at their peak."""
    return philox.draw(shape, seed, step, stream, 0, dtype, device)


def counter_uniform(
    shape: Sequence[int],
    seed: int | torch.Tensor,
    step: int | torch.Tensor,
    *,
    stream: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Uniform noise in [0, 1) of `shape`, as `counter_noise` draws (24 bits per float32, cast to
    `dtype`, where rounding can reach 1). It never shares values with `counter_noise`."""
    return philox.draw(shape, seed, step, stream, 1, dtype, device)
