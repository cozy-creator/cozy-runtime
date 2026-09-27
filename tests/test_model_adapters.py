from __future__ import annotations

import pytest

from cozy_runtime.author import (
    AdapterCompatibility,
    AdapterRef,
    CapabilityError,
    Model,
    ModelRegistry,
    uses_components,
)
from cozy_runtime.author._loader import Artifact, Config, Loader


class SyntheticModel(Model[object]):
    __adapter_compatibility__ = (
        AdapterCompatibility("lora", "synthetic", ("transformer",), -1.0, 1.0),
    )

    def __init__(self) -> None:
        self.applied: list[tuple[str, float]] = []

    def prepare_adapters(self, overlays: tuple[AdapterRef, ...]) -> None:
        self.validate_adapter_stack(overlays)
        self.applied = [(overlay.ref, overlay.scale) for overlay in overlays]


class PreparedModel(SyntheticModel):
    def load(self, loader: Loader) -> None:
        loader.construct(PreparedPipeline, factory=lambda config: PreparedPipeline())

    @uses_components("transformer")
    def run(self) -> str:
        return "ok"


class PreparedComponent:
    def state_dict(self) -> dict[str, object]:
        return {}


class PreparedPipeline:
    def __init__(self) -> None:
        self.transformer = PreparedComponent()


def adapter(ref: str, weight: float, *, component: str = "transformer") -> AdapterRef:
    return AdapterRef(ref, weight, "lora", component, "lora", "synthetic")


def test_incompatible_adapter_refuses_before_composer() -> None:
    model = SyntheticModel()
    with pytest.raises(CapabilityError, match="adapter_incompatible"):
        model.prepare_adapters((adapter("wrong", 0.5, component="vae"),))


def test_model_without_composer_refuses_typed() -> None:
    model: Model[object] = Model.for_test()
    with pytest.raises(CapabilityError, match="adapter_composition_unsupported"):
        model.prepare_adapters((adapter("a", 0.5),))


def test_registry_constructs_distinct_prepared_variants_once() -> None:
    registry = ModelRegistry(release="test")
    artifact = Artifact(snapshot="sha256:" + "ab" * 32, tensor_schema={}, config=Config({}))
    stack = (adapter("a", 0.5), adapter("b", -0.25))
    first = registry.acquire("serve.models.model", PreparedModel, artifact, adapters=stack)
    second = registry.acquire("serve.models.model.ordered", PreparedModel, artifact, adapters=stack)
    pristine = registry.acquire("serve.models.pristine", PreparedModel, artifact)
    assert first is second
    assert first is not pristine
    assert first.applied == [("a", 0.5), ("b", -0.25)]
    assert pristine.applied == []
