"""Model classes: keywords, sequence parallelism, component use, inheritance."""

from __future__ import annotations

from typing import Any

from cozy_runtime.author import Config, Loader, Model, sequence_parallel, uses_components


class Pipeline:
    """Stands in for a framework pipeline; never constructed at describe."""


def build_pipeline(config: Config) -> Pipeline:
    return Pipeline()


class BaseModel(Model[Pipeline], encoded_leaves="accept"):
    pipe: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=build_pipeline)

    @uses_components("text_encoder")
    def encode(self, text: str) -> Any:
        return None


@sequence_parallel(degrees=(2, 4))
class RenderModel(BaseModel, fusion="accept"):
    @uses_components("dit", "vae")
    def render(self, state: Any, *, on_step: Any) -> Any:
        return None

    def _private(self) -> None:
        return None


class SourceModel(Model[object]):
    def load(self, loader: Any) -> None:
        return None
