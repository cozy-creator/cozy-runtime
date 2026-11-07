"""Seed to noise, one way for every model, the same on any device, the CPU degraded mode included.

`noise` draws a request's noise on the CPU and moves it. `counter_noise` and `counter_uniform`
compute a step's noise where it is used, from (seed, step, stream) alone: Philox4x32-10 counters
in int64 tensor ops, then float32 operations that round correctly on every device. No generator
state, no table and no host read, so a `denoise_loop` step draws them, eager or recorded."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

#: Philox4x32 multipliers and key increments (Salmon et al. 2011, Random123)
_M0, _M1, _W0, _W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85
_MASK = 0xFFFFFFFF
#: Cephes logf's polynomial, highest power first
_LOG = (
    7.0376836292e-2,
    -1.1514610310e-1,
    1.1676998740e-1,
    -1.2420140846e-1,
    1.4249322787e-1,
    -1.6668057665e-1,
    2.0000714765e-1,
    -2.4999993993e-1,
    3.3333331174e-1,
)


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
    return torch.tensor([(seed + 2**63) % 2**64 - 2**63], dtype=torch.int64, device=device)


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
    polynomials). It costs about 330 elementwise kernels and 30 bytes per element at its peak."""
    words = _words(shape, seed, step, stream, 0, device)
    radius = (_log((words[:, 0::2] + 1).float() * 2.0**-32) * -2.0).sqrt()
    cos, sin = _turn(words[:, 1::2])
    values = torch.stack((radius * cos, radius * sin), -1).reshape(-1)
    return values[: math.prod(shape)].reshape(tuple(shape)).to(dtype)


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
    words = _words(shape, seed, step, stream, 1, device).reshape(-1)
    return ((words[: math.prod(shape)] >> 8).float() * 2.0**-24).reshape(tuple(shape)).to(dtype)


def _words(
    shape: Sequence[int], seed: Any, step: Any, stream: int, kind: int, device: Any
) -> torch.Tensor:
    """Philox4x32-10 of the counter (block, step, stream, kind) under the key `seed`: the four
    32-bit words (int64) of each block of four elements of `shape`, one row per block."""
    if device is None:
        device = next((v.device for v in (step, seed) if isinstance(v, torch.Tensor)), "cpu")
    blocks = (math.prod(shape) + 3) // 4
    c = [torch.arange(blocks, device=device), step & _MASK, stream & _MASK, kind]
    k0, k1 = seed & _MASK, (seed >> 32) & _MASK
    for r in range(10):
        if r:
            k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
        hi0, lo0 = _mulhilo(c[0], _M0)
        hi1, lo1 = _mulhilo(c[2], _M1)
        c = [hi1 ^ c[1] ^ k0, lo1, hi0 ^ c[3] ^ k1, lo0]
    return torch.stack(c, -1)


def _mulhilo(a: Any, m: int) -> tuple[Any, Any]:
    """The high and low 32-bit words of `a * m`, a 32-bit word by a 32-bit constant, by 16-bit
    halves of `m` so int64 never overflows."""
    low = a * (m & 0xFFFF)
    mid = a * (m >> 16) + (low >> 16)
    return mid >> 16, ((mid & 0xFFFF) << 16) | (low & 0xFFFF)


def _log(x: torch.Tensor) -> torch.Tensor:
    """Natural log of float32 `x` > 0 (Cephes logf)."""
    m, e = torch.frexp(x)
    low = m < math.sqrt(0.5)
    x = torch.where(low, m + m, m) - 1.0
    e = (e - low.int()).float()
    z = x * x
    y = x * _LOG[0] + _LOG[1]
    for coefficient in _LOG[2:]:
        y = y * x + coefficient
    return x + (y * x * z + e * -2.12194440e-4 - z * 0.5) + e * 0.693359375


def _turn(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine of a uniform angle set by the 32-bit word `w`: its top two bits pick a
    quarter turn, the rest an angle in [-pi/4, pi/4) (Cephes sinf and cosf)."""
    x = ((w & 0x3FFFFFFF) - 0x20000000).float() * (math.pi / 2**31)
    z = x * x
    s = ((z * -1.9515295891e-4 + 8.3321608736e-3) * z - 1.6666654611e-1) * z * x + x
    c = ((z * 2.443315711809948e-5 - 1.388731625493765e-3) * z + 4.166664568298827e-2) * z * z
    c = c - z * 0.5 + 1.0
    quarter, half = (w & 2**30) != 0, w >= 2**31
    cos, sin = torch.where(quarter, -s, c), torch.where(quarter, c, s)
    return torch.where(half, -cos, cos), torch.where(half, -sin, sin)
