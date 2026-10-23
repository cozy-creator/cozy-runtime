"""`quantize`: the Runtime's one entrypoint for the recipes a package declares.

A package with recipes serves it (`register`). The caller names the lane (and, of several, the
recipe) and the package and release it ran; the run takes the recipe of that lane whose model's
components the source holds, and of several the one whose plan selects from it. The lane's
header records where it came from (`provenance`); no reader needs it to load the lane.

A package with a calibrated recipe (`Recipe(calibrate=)`, an `svdq.nvfp4` plan) serves
`quantize_calibrated` as its `quantize` instead: on a GPU, with the recipe's model built on demand
from the source at 16 bits, which its `calibrate` runs real inputs through while the recorder is
open (`operations.calibrate`).
"""

from __future__ import annotations

import fnmatch
import importlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from cozy_runtime.author import (
    App,
    ConformanceError,
    Context,
    DerivationHook,
    MemoDistribution,
    Model,
    ModelArtifact,
    QuantizationPlan,
    Recipe,
    SelectionPlan,
    Telemetry,
    UnsupportedInput,
    Weights,
    invocable,
)
from cozy_runtime.author._recipes import (
    ENTRYPOINT,
    DeclaredRecipe,
    Plan,
    canonical,
    fingerprint,
    identity,
    outputs,
)
from cozy_runtime.weights_store import Checkpoint

from . import svdq
from .operations import calibrate, quantize_into, select_into
from .quantization import source_rows


# The code that decides a lane's bytes: its replay key changes with any of it.
@invocable(
    memo_dependencies=(
        "cozy_runtime.derive.recipes",
        "cozy_runtime.derive.operations",
        "cozy_runtime.derive.facade",
        "cozy_runtime.derive.quantization",
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
    opened = ctx.tensorfs_source(source)
    row, plan = choose(ctx._recipes, opened, lane=lane, recipe=recipe, package=package)
    form = canonical(plan)
    digest = fingerprint(form, package, release, row.calibrate)
    provenance = {"package": package, "release": release, "recipe": row.name,
                  "fingerprint": digest, "source": opened.id, "plan": form}  # fmt: skip
    if row.calibrate:
        provenance["calibrate"] = row.calibrate
    opener = ctx._tensorfs_output
    if opener is None:
        raise UnsupportedInput("this attempt writes no weights", code="weights_output_ungranted")
    ctx._tensorfs_output = lambda slot, made: opener(slot, replace(made, provenance=provenance))
    tel.log("recipe", recipe=row.name, lane=row.lane, fingerprint=digest)
    try:
        if isinstance(plan, QuantizationPlan):
            recorded = _calibrated(ctx, tel, opened, model, row, plan) if row.calibrate else None
            return quantize_into(
                ctx, tel, source=opened, plan=plan, output=row.lane, calibration=recorded
            )
        if isinstance(plan, SelectionPlan):
            return select_into(ctx, tel, source=opened, plan=plan, output=row.lane)
        return plan.function(ctx, tel, source, row.lane)
    finally:
        ctx._tensorfs_output = opener


def _calibrated(
    ctx: Context,
    tel: Telemetry,
    source: Checkpoint,
    model: Model[Any] | None,
    row: DeclaredRecipe,
    plan: QuantizationPlan,
) -> svdq.Calibration:
    """What `row`'s `calibrate` recorded running real inputs through `model`."""
    if model is None:
        raise UnsupportedInput(
            f"{row.name} calibrates on its model, and this quantize has no model slot",
            code="recipe_declaration",
        )
    recorders: list[svdq.Calibration] = []

    def record(components: Any) -> svdq.Calibration:
        torch = sys.modules["torch"]  # the model's own: a module of it is in `components`
        recorders.append(calibrate(ctx, torch, source=source, components=components, plan=plan))
        return recorders[-1]

    calibration = declared_recipe(row).calibrate
    assert calibration is not None
    calibration(ctx, tel, model, record)
    if not recorders:
        raise UnsupportedInput(
            f"{row.calibrate} recorded nothing: it never opened `record(components)`",
            code="recipe_calibration",
        )
    return recorders[-1]


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


def choose(
    declared: Sequence[DeclaredRecipe],
    source: Checkpoint,
    *,
    lane: str,
    recipe: str,
    package: str,
    plan: Callable[[DeclaredRecipe], Plan] | None = None,
) -> tuple[DeclaredRecipe, Plan]:
    """The one recipe of `lane` (named `recipe`, when given) that reads `source`; each row's
    plan is the one its module binds (`declared_plan`) unless `plan` reads it otherwise."""
    named = recipe.replace("-", "_").lower()
    offered = [r for r in declared if r.lane == lane and (not named or r.name == named)]
    if not offered:
        what = f"{recipe!r} writing {lane}" if recipe else lane
        lanes = ", ".join(sorted({r.lane for r in declared}))
        raise UnsupportedInput(
            f"{package} declares no {what} recipe; its lanes: {lanes}", code="recipe_not_found"
        )
    held = set(source.components)
    reads = [r for r in offered if r.model.components() <= held]
    if not reads:
        wants = "; ".join(
            f"{r.name} reads {r.model.class_name} ({', '.join(sorted(r.model.components()))})"
            for r in offered
        )
        raise UnsupportedInput(
            f"no {lane} recipe of {package} reads this checkpoint ({', '.join(sorted(held))}): "
            f"{wants}",
            code="recipe_source",
        )
    found = [(row, (plan or declared_plan)(row)) for row in reads]
    if len(found) > 1:
        found = [(row, plan) for row, plan in found if fits(source, plan)]
    if len(found) != 1:
        names = ", ".join(row.name for row in reads)
        raise UnsupportedInput(
            f"{len(found)} of {package}'s {lane} recipes ({names}) select from this checkpoint; "
            "name one with recipe=",
            code="recipe_ambiguous",
        )
    return found[0]


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


def fits(source: Checkpoint, plan: Plan) -> bool:
    """Whether every pattern and key of `plan` names a weight of `source`."""
    if isinstance(plan, DerivationHook):
        return True
    within = plan.components if isinstance(plan, QuantizationPlan) else source.components
    names = [f"{c}.{key}" for c, key, _ in source_rows(source) if c in within]
    keys = plan.keys if isinstance(plan, QuantizationPlan) else ()
    return (
        set(within) <= set(source.components)
        and all(any(fnmatch.fnmatchcase(n, rule) for n in names) for rule in plan.include)
        and all(f"{c}.{key}" in names for c in within for key in keys)
    )
