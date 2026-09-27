"""One Diffusers attention site whose backend each request may select (cr-191)."""

from __future__ import annotations

import msgspec
import torch
from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

from cozy_runtime.author import App, Config, Loader, Model, uses_components

app = App()


def site() -> WanAttention:
    return WanAttention(dim=64, heads=2, dim_head=32, processor=WanAttnProcessor())


class Pipe:
    def __init__(self, config: Config) -> None:
        del config
        self.components = {"dit": site().to(torch.bfloat16)}


class Attention(Model[Pipe]):
    pipe: Pipe

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipe, factory=Pipe)

    @uses_components("dit")
    def attend(self) -> float:
        dit = self.pipe.components["dit"]
        tokens = torch.ones(1, 128, 64, dtype=torch.bfloat16, device=next(dit.parameters()).device)
        with torch.no_grad():
            return float(dit(tokens).float().sum())


class Request(msgspec.Struct):
    value: int = 1


class Result(msgspec.Struct):
    checksum: float


@app.entrypoint
def generate(payload: Request, model: Attention) -> Result:
    return Result(model.attend() * payload.value)
