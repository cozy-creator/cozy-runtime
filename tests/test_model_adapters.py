from __future__ import annotations

import pytest

from cozy_runtime.author import AdapterCompatibility, AdapterRef, CapabilityError, Model


class SyntheticModel(Model[object]):
    __adapter_compatibility__ = (
        AdapterCompatibility("lora", "synthetic", ("transformer",), -1.0, 1.0),
    )

    def __init__(self) -> None:
        self.applied: list[tuple[str, float]] = []

    def prepare_adapters(self, overlays: tuple[AdapterRef, ...]) -> None:
        self.validate_adapter_stack(overlays)
        self.applied = [(overlay.ref, overlay.scale) for overlay in overlays]


def adapter(ref: str, weight: float, *, component: str = "transformer") -> AdapterRef:
    return AdapterRef(ref, weight, "lora", component, "lora", "synthetic")


def test_ordered_weighted_overlays_are_prepared_once() -> None:
    model = SyntheticModel()
    stack = (adapter("a", 0.5), adapter("b", -0.25))
    model.prepare_adapters(stack)
    assert model.applied == [("a", 0.5), ("b", -0.25)]


def test_incompatible_adapter_refuses_before_composer() -> None:
    model = SyntheticModel()
    with pytest.raises(CapabilityError, match="adapter_incompatible"):
        model.prepare_adapters((adapter("wrong", 0.5, component="vae"),))


def test_base_model_without_overlay_stays_pristine() -> None:
    model = SyntheticModel()
    model.prepare_adapters(())
    assert model.applied == []


def test_model_without_composer_refuses_typed() -> None:
    model = Model.for_test()
    with pytest.raises(CapabilityError, match="adapter_composition_unsupported"):
        model.prepare_adapters((adapter("a", 0.5),))
