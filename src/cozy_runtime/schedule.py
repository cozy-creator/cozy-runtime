"""A Diffusers scheduler's steps as device tables, for `author.denoise_loop`.

A Diffusers `step` keeps host state (its step index and sigmas) and may draw noise; a loop's
step may do neither. `scheduled` runs the scheduler's own `scale_model_input` and `step` once
per step on the host beforehand, on probe values, and keeps what each computes: the model's
input as `scale * sample`, the next sample as `a * sample + b * output + c * noise`. The loop's
step reads its row by its index tensor and draws its noise from the request's seed and its index
(`rng.counter_noise`, stream 0), eager or recorded.

    schedule = scheduled(scheduler, latents, seed, eta=1.0)
    def step(i, latents, *tables):
        s = Schedule(*tables)
        s.update(i, latents, model(s.scale(i, latents), s.timestep(i)))
    denoise_loop(step, range(len(schedule)), state=(latents,), inputs=schedule.inputs)

A scheduler whose step is not that one linear update (a multistep solver's history, Heun's two
evaluations per step, clipping or thresholding) is refused by name.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any

import torch

from cozy_runtime import rng

#: Each probe step's (sample, output) pairs: four fixed ones (the update's three coefficients,
#: and a large pair a clamp would change), then the same four on even steps and random ones
#: otherwise, then the converse, so an update that reads an earlier step differs.
_BASIS = ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0), (6.0, -8.0))


@dataclass(frozen=True, slots=True)
class Schedule:
    """`timesteps`: the model's timestep per step, as the scheduler gives it. `scales`: what
    `scale_model_input` multiplies the sample by. `table`: (a, b, c) per step, float32.
    `seed`: the noise's `rng.counter_seed`; None when no step draws."""

    timesteps: torch.Tensor
    scales: torch.Tensor
    table: torch.Tensor
    seed: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.table.shape[0])

    @property
    def inputs(self) -> tuple[torch.Tensor, ...]:
        """Its tensors, for `denoise_loop(inputs=)`; `Schedule(*inputs)` in the step."""
        tables = (self.timesteps, self.scales, self.table)
        return tables if self.seed is None else (*tables, self.seed)

    def timestep(self, i: torch.Tensor) -> torch.Tensor:
        return self.timesteps.index_select(0, i)

    def scale(self, i: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        return (sample.float() * self.scales.index_select(0, i)).to(sample.dtype)

    def update(self, i: torch.Tensor, sample: torch.Tensor, output: torch.Tensor) -> None:
        """Step `i` on `sample`, in place, from the model's `output`, in float32: within a
        rounding of the sample's dtype per step of Diffusers' own, whose scalar products round
        differently on CPU and CUDA."""
        a, b, c = self.table.index_select(0, i)[0].unbind(0)
        nxt = sample.float() * a + output.float() * b
        if self.seed is not None:
            nxt = nxt + rng.counter_noise(sample.shape, self.seed, i, device=sample.device) * c
        sample.copy_(nxt)


def scheduled(
    scheduler: Any, sample: torch.Tensor, seed: int | None = None, **step: Any
) -> Schedule:
    """`scheduler` after its `set_timesteps`, for loops over samples like `sample`, on its
    device. `step`: what the package passes its `step` (DDIM's `eta`). A scheduler that draws
    noise takes the request's `seed` for it. The scheduler itself does not advance."""
    clone = copy.deepcopy(scheduler)
    for name, value in list(vars(clone).items()):
        if isinstance(value, torch.Tensor):
            setattr(clone, name, value.cpu())
    name = type(scheduler).__name__
    draws = torch.Generator().manual_seed(0)
    takes = "generator" in inspect.signature(clone.step).parameters
    scaling = getattr(clone, "scale_model_input", None)
    rows, scales = [], []
    for k, t in enumerate(clone.timesteps):
        x, eps, tested = _probe(k, draws)
        shape = (1, 1, 1, x.numel())  # a latent's (batch, channels, height, width)
        probe = int(torch.randint(1 << 62, (1,), generator=draws))
        options = {**step, "generator": torch.Generator().manual_seed(probe)} if takes else step
        try:
            scaled = x if scaling is None else scaling(x.view(shape), t).flatten()
            out = clone.step(eps.view(shape), t, x.view(shape), return_dict=False, **options)[0]
        except Exception as exc:
            raise ValueError(f"{name} step {k} does not run on a probe: {exc}") from exc
        out = out.float().flatten()
        n = torch.randn(shape, generator=torch.Generator().manual_seed(probe)).flatten()
        c = out[2] / n[2]
        a, b = out[0] - c * n[0], out[1] - c * n[1]
        terms = (a * x, b * eps, c * n)
        off = (out - sum(terms)).abs() > 1e-4 * sum(term.abs() for term in terms) + 1e-6
        scale = scaled[0].float()
        if off[tested].any() or not torch.allclose(scaled.float(), scale * x, rtol=1e-5):
            raise ValueError(
                f"{name} step {k} is not one linear update of the sample, the model's output "
                "and fresh noise (a multistep solver, two evaluations per step, clipping or "
                "thresholding): `scheduled` serves first-order schedulers"
            )
        rows.append(torch.stack((a, b, c)))
        scales.append(scale)
    table = torch.stack(rows)
    drawn = None
    if bool(table[:, 2].ne(0).any()):
        if seed is None:
            raise ValueError(f"{name} draws noise: pass the request's seed to `scheduled`")
        drawn = rng.counter_seed(seed, sample.device)
    return Schedule(
        torch.as_tensor(scheduler.timesteps).to(sample.device),
        torch.stack(scales).to(sample.device),
        table.to(sample.device),
        drawn,
    )


def _probe(k: int, draws: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Step K's probe samples and outputs, and the elements whose update it checks."""
    basis = torch.tensor(_BASIS)
    other = torch.randn(len(_BASIS), 2, generator=draws) * 4
    even = k % 2 == 0
    pairs = torch.cat([basis, *((basis, other) if even else (other, basis))])
    width = len(_BASIS)
    tested = [*range(width), *(range(width, 2 * width) if even else range(2 * width, 3 * width))]
    return pairs[:, 0].contiguous(), pairs[:, 1].contiguous(), tested
