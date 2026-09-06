"""Five separately used components, with a whole-model scope as the refusal control."""

from typing import Any

import msgspec

from cozy_runtime.author import App, Config, Context, Loader, Model, uses_components

app = App()


class Request(msgspec.Struct):
    seed: int = 0


class Result(msgspec.Struct):
    checksum: float


class Pipeline:
    def __init__(self, config: Config) -> None:
        import torch

        class Component(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.empty(config.as_int("elements"), dtype=torch.bfloat16),
                    requires_grad=False,
                )

        self.components = {name: Component() for name in ("a", "b", "c", "d", "e")}


def build(config: Config) -> Pipeline:
    return Pipeline(config)


def first(member: Any) -> float:
    return float(member.weight[0])


class StagedModel(Model[Pipeline]):
    pipe: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=build)

    @uses_components("a")
    def a(self) -> float:
        return first(self.pipe.components["a"])

    @uses_components("b")
    def b(self) -> float:
        return first(self.pipe.components["b"])

    @uses_components("c")
    def c(self) -> float:
        return first(self.pipe.components["c"])

    @uses_components("d")
    def d(self) -> float:
        return first(self.pipe.components["d"])

    @uses_components("e")
    def e(self) -> float:
        return first(self.pipe.components["e"])


@app.entrypoint
def touch(ctx: Context, payload: Request, model: StagedModel) -> Result:
    return Result(model.a() + model.b() + model.c() + model.d() + model.e())
