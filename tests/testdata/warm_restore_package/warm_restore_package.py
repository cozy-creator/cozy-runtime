"""Two real components whose staged warm ends with only the decoder resident."""

from typing import Any

import msgspec
import torch

from cozy_runtime.author import App, Config, Context, Loader, Model, uses_components

app = App()


class Pipeline:
    def __init__(self, config: Config) -> None:
        self.components = {
            name: torch.nn.Linear(16, 16, bias=False) for name in ("encoder", "decoder")
        }


class WarmModel(Model[Pipeline]):
    pipe: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=Pipeline)

    def warm(self, ctx: Context) -> None:
        self.decode(self.encode(1))

    @uses_components("encoder")
    def encode(self, value: int) -> Any:
        component = self.pipe.components["encoder"]
        with torch.no_grad():
            return component(torch.full((1, 16), float(value), device=component.weight.device))

    @uses_components("decoder")
    def decode(self, value: Any) -> float:
        return float(self.pipe.components["decoder"](value).sum())


class Input(msgspec.Struct):
    value: int = 1


class Output(msgspec.Struct):
    checksum: float


@app.entrypoint
def generate(payload: Input, model: WarmModel) -> Output:
    return Output(model.decode(model.encode(payload.value)))
