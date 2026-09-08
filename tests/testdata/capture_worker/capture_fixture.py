"""Tiny real ordered model: capture must not change its seeded output or RNG state."""

from __future__ import annotations

import hashlib

import msgspec

from cozy_runtime.author import App, Config, Context, Loader, Model, Telemetry, uses_components

app = App()


class Request(msgspec.Struct, forbid_unknown_fields=True):
    seed: int = 1234
    steps: int = 3


class Result(msgspec.Struct):
    value: float
    components: list[str]
    device: str
    cpu_rng: str
    cuda_rng: str
    next_noise: list[float]


class OrderedPipeline:
    def __init__(self) -> None:
        import torch

        self.components = {
            "zeta": torch.nn.Linear(2, 2, bias=False, dtype=torch.float32),
            "alpha": torch.nn.Linear(2, 2, bias=False, dtype=torch.float32),
        }


def build_pipeline(config: Config) -> OrderedPipeline:
    return OrderedPipeline()


class OrderedModel(Model[OrderedPipeline]):
    pipe: OrderedPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OrderedPipeline, factory=build_pipeline)

    @uses_components("alpha", "zeta")
    def measure(self, seed: int, steps: int, tel: Telemetry) -> Result:
        import torch

        zeta = self.pipe.components["zeta"]
        alpha = self.pipe.components["alpha"]
        torch.manual_seed(seed)
        generator = torch.Generator(device=zeta.weight.device).manual_seed(seed)
        on_step = tel.step_callback(steps, stage="denoise")
        with torch.inference_mode():
            vector = torch.randn((1, 2), generator=generator, device=zeta.weight.device)
            for step in range(steps):
                vector = alpha(zeta(vector))
                on_step(step)
            return Result(
                float(vector.sum().item()),
                list(self.pipe.components),
                vector.device.type,
                hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
                hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                torch.randn((2,), generator=generator, device=zeta.weight.device).tolist(),
            )


@app.entrypoint
def generate(ctx: Context, payload: Request, model: OrderedModel, tel: Telemetry) -> Result:
    ctx.raise_if_cancelled()
    return model.measure(payload.seed, payload.steps, tel)
