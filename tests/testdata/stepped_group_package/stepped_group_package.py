"""A tiny real H3 transformer sampled step by step: each step waits, then one sharded forward."""

from __future__ import annotations

import time
from typing import Any, cast

import msgspec
import torch
from diffusers import MiniMaxH3Transformer3DModel

from cozy_runtime.author import (
    App,
    Config,
    Loader,
    Model,
    Telemetry,
    sequence_parallel,
    uses_components,
)

app = App()

CONFIG = dict(
    num_attention_heads=4,
    attention_head_dim=16,
    hidden_size=16,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=32,
    in_channels=2,
    audio_in_channels=2,
    patch_size=[1, 1, 1],
    text_dim=4,
    freq_dim=8,
    time_embed_hidden_dim=16,
    time_embed_dim=8,
    rope_freq_dim=2,
)


def batch() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(91)
    return {
        "hidden_states": torch.randn(1, 8, 2, generator=generator),
        "audio_hidden_states": torch.randn(1, 8, 2, generator=generator),
        "encoder_hidden_states": torch.randn(1, 8, 4, generator=generator),
        "timestep": torch.tensor([0.0, 0.25, 0.75]),
        "timestep_indices": torch.arange(24) % 3,
        "token_tags": torch.arange(24) % 3,
        "position_ids": torch.arange(72).reshape(24, 3) % 5,
        "video_indices": torch.arange(0, 24, 3),
        "text_indices": torch.arange(1, 24, 3),
        "audio_indices": torch.arange(2, 24, 3),
        "return_dict": False,
    }


class StepPipe:
    def __init__(self, config: Config) -> None:
        mapping = cast(dict[str, Any], config.mapping())
        constructor: Any = MiniMaxH3Transformer3DModel
        dit: Any = constructor(**mapping["dit"]).eval()
        dit.set_attention_backend("native")
        self.components = {"dit": dit}


@sequence_parallel(degrees=(2,))
class Stepped(Model[StepPipe]):
    pipe: StepPipe

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(StepPipe, factory=StepPipe)

    @uses_components("dit")
    def denoise(self, steps: int, seconds: float, on_step: Any) -> float:
        dit = self.pipe.components["dit"]
        device = next(dit.parameters()).device
        values = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch().items()}
        total = 0.0
        for step in range(steps):
            time.sleep(seconds)
            with torch.no_grad():
                total += sum(float(value.sum()) for value in dit(**values))
            on_step(step)
        return total


class Input(msgspec.Struct):
    steps: int = 1
    seconds: float = 0.0


class Output(msgspec.Struct):
    steps: int
    checksum: float


@app.entrypoint
def generate(payload: Input, base: Stepped, tel: Telemetry) -> Output:
    on_step = tel.step_callback(payload.steps, stage="denoise")
    return Output(payload.steps, base.denoise(payload.steps, payload.seconds, on_step))
