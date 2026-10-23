"""A package whose svdq-nvfp4 recipe calibrates on its model, beside a plain int8 recipe."""

from __future__ import annotations

from typing import Any

from cozy_runtime.author import (
    App,
    Context,
    EncodingRule,
    Loader,
    Model,
    QuantizationPlan,
    Recipe,
    Telemetry,
    uses_components,
)

app = App()


class Tiny(Model[object]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("a plain lane never builds the calibrating model")

    @uses_components("unet")
    def denoise(self, state: Any) -> Any:
        return state


def run_inputs(ctx: Context, tel: Telemetry, model: Tiny, record: Any) -> None:
    """The package's real inputs through its model, while their recorder is open."""
    with record(model.graph.components):
        model.denoise(None)


INT8 = Recipe(
    lane="int8",
    model=Tiny,
    plan=QuantizationPlan(
        encoding="int8.convrot", components=("unet",), include=("unet.a.weight",)
    ),
)
SVDQ_NVFP4 = Recipe(
    lane="svdq-nvfp4",
    model=Tiny,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("unet",),
        include=("unet.a.weight",),
        encodings=(EncodingRule("unet.*", "svdq.nvfp4"),),
    ),
    calibrate=run_inputs,
)
