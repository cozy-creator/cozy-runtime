from __future__ import annotations

import msgspec

from cozy_runtime.author import App, Config, Loader, Model, uses_components


class Request(msgspec.Struct):
    value: int


class Result(msgspec.Struct):
    value: int


class Pipeline:
    def __init__(self, core: object) -> None:
        self.components = {"core": core}


def build(config: Config) -> Pipeline:
    import torch

    if config.as_int("width") != 4:
        raise ValueError("wrong exact config")
    return Pipeline(torch.nn.Module())


class ToyModel(Model[Pipeline]):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=build)

    @uses_components("core")
    def touch(self) -> None:
        return None


class HiddenToyModel(ToyModel):
    pass


class PeerModel(ToyModel):
    pass


app = App()


@app.entrypoint
def metadata(payload: Request) -> Result:
    return Result(payload.value)


@app.entrypoint
def render(payload: Request, model: ToyModel, peer: PeerModel) -> Result:
    model.touch()
    peer.touch()
    return Result(payload.value)


@app.entrypoint(hidden=True)
def hidden_render(payload: Request, model: HiddenToyModel) -> Result:
    model.touch()
    return Result(payload.value)
