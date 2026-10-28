"""`quantize`: the Runtime's one entrypoint for the recipes a package declares.

A package with recipes serves it (`register`). The caller names the lane (and, of several, the
recipe) and the package and release it ran; the run takes the recipe of that lane whose model's
components the source holds, and of several the one whose plan selects from it. The lane's
header records where it came from (`provenance`); no reader needs it to load the lane.

A package with a calibrated recipe (`Recipe(calibrate=)`, an `svdq.nvfp4` plan) serves
`quantize_calibrated` as its `quantize` instead: on a GPU, with the recipe's model built on demand
from the source at 16 bits, which its `calibrate` runs real inputs through while the recorder is
open (`operations.calibrate`).

This module is the declaration and holds no derivation code: an executor installs the Runtime's
side (`install(operations.run_recipe)`), so describing a package never imports the derivation.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from typing import Any, Protocol

from cozy_runtime.author import (
    App,
    ConformanceError,
    Context,
    MemoDistribution,
    Model,
    ModelArtifact,
    Recipe,
    Telemetry,
    Weights,
    invocable,
)
from cozy_runtime.author._recipes import (
    ENTRYPOINT,
    DeclaredRecipe,
    Plan,
    canonical,
    identity,
    outputs,
)


# The code that decides a lane's bytes: its replay key changes with any of it.
@invocable(
    memo_dependencies=(
        "cozy_runtime.derive.recipes",
        "cozy_runtime.derive.operations",
        "cozy_runtime.derive.facade",
        "cozy_runtime.derive.quantization",
        "cozy_runtime.derive.native_quantization",
        "cozy_runtime.derive.microscale",
        "cozy_runtime.derive.safetensors_io",
        "cozy_runtime.weights_store",
        MemoDistribution("tensorfs"),
        MemoDistribution("numpy"),
    )
)
async def quantize(
    ctx: Context,
    *,
    source: Weights[Any],
    lane: str,
    package: str,
    release: str,
    recipe: str = "",
    tel: Telemetry,
) -> ModelArtifact:
    """The package's recipe that writes `lane` (or the one named `recipe`) from `source`; the
    lane's header records it (`provenance`)."""
    return _run(ctx, tel, source, None, lane=lane, package=package, release=release, recipe=recipe)


@invocable(
    memo_dependencies=(
        "cozy_runtime.derive.recipes",
        "cozy_runtime.derive.operations",
        "cozy_runtime.derive.facade",
        "cozy_runtime.derive.quantization",
        "cozy_runtime.derive.native_quantization",
        "cozy_runtime.derive.microscale",
        "cozy_runtime.derive.svdq",
        "cozy_runtime.derive.safetensors_io",
        "cozy_runtime.weights_store",
        MemoDistribution("tensorfs"),
        MemoDistribution("numpy"),
    )
)
async def quantize_calibrated(
    ctx: Context,
    *,
    source: Weights[Any],
    model: Model[Any],
    lane: str,
    package: str,
    release: str,
    recipe: str = "",
    tel: Telemetry,
) -> ModelArtifact:
    """`quantize` for a package with a calibrated recipe: `model` is its model class (`register`
    names it), built on demand from the source at 16 bits for a recipe that calibrates."""
    return _run(ctx, tel, source, model, lane=lane, package=package, release=release, recipe=recipe)


class Runner(Protocol):
    """The Runtime's side of `quantize`, installed by an executor."""

    def __call__(
        self,
        ctx: Context,
        tel: Telemetry,
        source: Weights[Any],
        model: Model[Any] | None,
        *,
        lane: str,
        package: str,
        release: str,
        recipe: str,
    ) -> ModelArtifact: ...


_RUNNER: Runner | None = None


def install(runner: Runner | None) -> None:
    global _RUNNER
    _RUNNER = runner


def _run(
    ctx: Context,
    tel: Telemetry,
    source: Weights[Any],
    model: Model[Any] | None,
    *,
    lane: str,
    package: str,
    release: str,
    recipe: str,
) -> ModelArtifact:
    if _RUNNER is None:
        raise RuntimeError("quantize runs inside a Runtime executor")
    return _RUNNER(
        ctx, tel, source, model, lane=lane, package=package, release=release, recipe=recipe
    )


def register(app: App, declared: Sequence[DeclaredRecipe]) -> None:
    """Serve `declared` from `app` as its `quantize` entrypoint (once): `quantize_calibrated`
    with the calibrated recipes' model class when one calibrates, else `quantize`."""
    if any(registered.name == ENTRYPOINT for registered in app.registrations()):
        return
    lanes = outputs(row.lane for row in declared)
    calibrated = {_recipe(row, importlib.import_module(row.module)).model for row in declared
                  if row.calibrate}  # fmt: skip
    if not calibrated:
        app.entrypoint(quantize, name=ENTRYPOINT, weights=lanes, accelerator=False)
        return
    if len(calibrated) > 1:
        raise ConformanceError("calibrated recipes of two model classes", code="recipe_declaration")
    app.entrypoint(
        quantize_calibrated, name=ENTRYPOINT, weights=lanes, on_demand=("model",),
        model_classes={"model": calibrated.pop()},
    )  # fmt: skip


def declared_plan(row: DeclaredRecipe) -> Plan:
    """The plan bound at the row's module and export, as the interface records it."""
    return declared_recipe(row).plan


def declared_recipe(row: DeclaredRecipe) -> Recipe:
    """The `Recipe` bound at the row's module and export, as the interface records it."""
    return _recipe(row, sys.modules.get(row.module))


def _recipe(row: DeclaredRecipe, module: object) -> Recipe:
    value = getattr(module, row.export, None)
    calibrate = value.calibrate if isinstance(value, Recipe) else None
    if (
        not isinstance(value, Recipe)
        or value.lane != row.lane
        or canonical(value.plan) != row.plan
        or (identity(calibrate) if calibrate is not None else "") != row.calibrate
    ):
        raise ConformanceError(
            f"{row.module}.{row.export} is not the {row.lane} recipe the interface declares",
            code="recipe_declaration",
        )
    return value
