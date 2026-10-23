"""The recipes: plans are data; `z_only` is a hook, code that writes the lane itself."""

from __future__ import annotations

from typing import Any

from cozy_runtime.author import (
    Context,
    DerivationHook,
    EncodingRule,
    Model,
    ModelArtifact,
    QuantizationPlan,
    Recipe,
    SelectionPlan,
    Telemetry,
    Weights,
)
from cozy_runtime.derive.operations import select_into

from .models import Speaker, Tiny

_LINEARS = ("a", "z")

INT8 = Recipe(
    lane="int8",
    model=Tiny,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=("unet",),
        include=tuple(f"unet.{name}.weight" for name in _LINEARS),
        output_precision="preserve",
    ),
)
#: Tiny again: chosen by its plan only for a checkpoint holding `unet.q.weight`.
INT8_Q = Recipe(
    lane="int8",
    model=Tiny,
    plan=QuantizationPlan(
        encoding="int8.convrot", components=("unet",), include=("unet.q.weight",)
    ),
)
W4A8 = Recipe(
    lane="w4a8",
    model=Tiny,
    plan=QuantizationPlan(
        encoding="int8.convrot",
        components=INT8.plan.components,
        include=(*INT8.plan.include,),
        encodings=(EncodingRule("unet.a.*", "w4a8.convrot"),),
        output_precision="preserve",
    ),
)
SPEAKER_INT8 = Recipe(
    lane="int8",
    model=Speaker,
    plan=QuantizationPlan(encoding="int8.convrot", components=("text",)),
)
BF16 = Recipe(lane="bf16", model=Model, plan=QuantizationPlan())
HEAD = Recipe(lane="head", model=Tiny, plan=SelectionPlan(include=("unet.a.*",)))


def z_only(ctx: Context, tel: Telemetry, source: Weights[Any], output: str) -> ModelArtifact:
    return select_into(ctx, tel, source=source, plan=SelectionPlan(include=("unet.z.*",)), output=output)


Z = Recipe(lane="z", model=Tiny, plan=DerivationHook(z_only))
