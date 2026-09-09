"""Tiny real H3 plus an independent native projection for group preparation tests."""

from __future__ import annotations

from typing import Any

import msgspec
import torch
from diffusers import MiniMaxH3Transformer3DModel

from cozy_runtime.author import App, Config, Loader, Model, sequence_parallel, uses_components

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


def project(module: Any, args: Any, kwargs: Any) -> Any:
    del module
    options = kwargs.get("attention_kwargs") or {}
    projection = options.get("projection")
    if projection is None:
        return None
    value = projection(kwargs["hidden_states"])
    return args, {**kwargs, "hidden_states": value, "attention_kwargs": {}}


class BasePipe:
    def __init__(self, config: Config) -> None:
        dit = MiniMaxH3Transformer3DModel(**config.mapping()["dit"]).eval()
        dit.set_attention_backend("native")
        dit.register_forward_pre_hook(project, with_kwargs=True)
        self.components = {"dit": dit}


class OverlayPipe:
    def __init__(self, config: Config) -> None:
        del config
        self.components = {"projection": torch.nn.Linear(2, 2, bias=False)}


@sequence_parallel(degrees=(2, 4))
class Base(Model[BasePipe]):
    pipe: BasePipe

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(BasePipe, factory=BasePipe)

    def warm(self, ctx: Any) -> None:
        ctx.raise_if_cancelled()
        device = next(self.pipe.components["dit"].parameters()).device
        self.plain(
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch().items()}
        )

    @uses_components("dit")
    def plain(self, values: dict[str, Any]) -> Any:
        with torch.no_grad():
            return self.pipe.components["dit"](**values)

    @uses_components("dit")
    def sample(self, overlay: Overlay, state: Any, callback: Any) -> Any:
        return overlay.sample(self, state, callback)


@sequence_parallel(degrees=(2, 4))
class Overlay(Model[OverlayPipe]):
    pipe: OverlayPipe

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(OverlayPipe, factory=OverlayPipe)

        def active(module: Any, args: Any) -> None:
            del module, args
            assert self._cozy_active == ("sample", ("projection",))

        self.pipe.components["projection"].register_forward_pre_hook(active)

    @uses_components("projection")
    def sample(self, base: Base, state: Any, callback: Any) -> Any:
        with torch.no_grad():
            state.result = base.pipe.components["dit"](
                **state.values,
                attention_kwargs={"projection": self.pipe.components["projection"]},
            )
        callback()
        return state.result


class Input(msgspec.Struct):
    value: int = 1


class Output(msgspec.Struct):
    value: int


@app.entrypoint
def generate(payload: Input, base: Base, overlay: Overlay) -> Output:
    del base, overlay
    return Output(payload.value)
