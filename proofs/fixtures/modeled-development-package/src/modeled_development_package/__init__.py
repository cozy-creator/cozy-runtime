from __future__ import annotations

from typing import Any

import msgspec

from cozy_runtime.author import App, Config, Loader, Model, uses_components


class Request(msgspec.Struct):
    value: int


class Result(msgspec.Struct):
    value: int


class Pipeline:
    def __init__(self, core: Any) -> None:
        self.components = {"core": core}


def build(config: Config) -> Pipeline:
    import torch

    if config.as_int("width") != 4:
        raise ValueError("wrong exact config")
    core = torch.nn.Module()
    core.register_parameter(
        "weight", torch.nn.Parameter(torch.empty(4, dtype=torch.bfloat16), requires_grad=False)
    )
    return Pipeline(core)


class TinyModel(Model[Pipeline]):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=build)

    @uses_components("core")
    def first(self) -> int:
        return int(self.pipe.components["core"].weight[0].item())


app = App()


@app.entrypoint
def render(payload: Request, model: TinyModel) -> Result:
    return Result(payload.value + model.first())
