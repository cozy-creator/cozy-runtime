"""`quantize`: the Runtime's one entrypoint for the recipes a package declares.

A package with recipes serves it (`register`). The caller names the lane (and, of several, the
recipe) and the package and release it ran; the run takes the recipe of that lane whose model's
components the source holds, and of several the one whose plan selects from it. The lane's
header records where it came from (`provenance`); no reader needs it to load the lane.
"""

from __future__ import annotations

import fnmatch
import sys
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from cozy_runtime.author import (
    App,
    ConformanceError,
    Context,
    DerivationHook,
    MemoDistribution,
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
    outputs,
)
from cozy_runtime.weights_store import Checkpoint, Derivation, Output

from .operations import quantize_into, select_into
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
    """The package's recipe that writes `lane` (or the one named `recipe`) from `source`."""
    return run(
        ctx, tel, ctx._recipes, source, lane=lane, recipe=recipe, package=package, release=release
    )


def register(app: App, declared: Sequence[DeclaredRecipe]) -> None:
    """Serve `declared` from `app` as its `quantize` entrypoint (once)."""
    if any(registered.name == ENTRYPOINT for registered in app.registrations()):
        return
    lanes = outputs(row.lane for row in declared)
    app.entrypoint(quantize, name=ENTRYPOINT, weights=lanes, accelerator=False)


def run(
    ctx: Context,
    tel: Telemetry,
    declared: Sequence[DeclaredRecipe],
    source: Weights[Any],
    *,
    lane: str,
    recipe: str,
    package: str,
    release: str,
) -> ModelArtifact:
    opened = ctx.tensorfs_source(source)
    row, plan = choose(declared, opened, lane=lane, recipe=recipe, package=package)
    form = canonical(plan)
    digest = fingerprint(form, package, release)
    provenance: dict[str, object] = {
        "package": package,
        "release": release,
        "recipe": row.name,
        "fingerprint": digest,
        "source": opened.id,
        "plan": form,
    }
    opener = ctx._tensorfs_output
    if opener is None:
        raise UnsupportedInput("this attempt writes no weights", code="weights_output_ungranted")

    def recorded(slot: str, derivation: Derivation) -> Output:
        return opener(slot, replace(derivation, provenance=provenance))

    ctx._tensorfs_output = recorded
    tel.log("recipe", recipe=row.name, lane=row.lane, fingerprint=digest)
    try:
        if isinstance(plan, QuantizationPlan):
            return quantize_into(ctx, tel, source=opened, plan=plan, output=row.lane)
        if isinstance(plan, SelectionPlan):
            return select_into(ctx, tel, source=opened, plan=plan, output=row.lane)
        return plan.function(ctx, tel, source, row.lane)
    finally:
        ctx._tensorfs_output = opener


def choose(
    declared: Sequence[DeclaredRecipe],
    source: Checkpoint,
    *,
    lane: str,
    recipe: str,
    package: str,
) -> tuple[DeclaredRecipe, Plan]:
    """The one recipe of `lane` (named `recipe`, when given) that reads `source`."""
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
    found = [(row, declared_plan(row)) for row in reads]
    if len(found) > 1:
        fitting = [(row, plan) for row, plan in found if fits(source, plan)]
        names = ", ".join(row.name for row, _ in fitting or found)
        if not fitting:
            raise UnsupportedInput(
                f"none of {package}'s {lane} recipes ({names}) selects from this checkpoint",
                code="recipe_source",
            )
        if len(fitting) > 1:
            raise UnsupportedInput(
                f"{names} all write {lane} from this checkpoint; name one with recipe=",
                code="recipe_ambiguous",
            )
        found = fitting
    return found[0]


def declared_plan(row: DeclaredRecipe) -> Plan:
    """The plan bound at the row's module and export, as the interface records it."""
    module = sys.modules.get(row.module)
    value = getattr(module, row.export, None)
    if not isinstance(value, Recipe) or value.lane != row.lane or canonical(value.plan) != row.plan:
        raise ConformanceError(
            f"{row.module}.{row.export} is not the {row.lane} recipe the interface declares",
            code="recipe_declaration",
        )
    return value.plan


def fits(source: Checkpoint, plan: Plan) -> bool:
    """Whether every pattern and key of `plan` names a weight of `source`."""
    if isinstance(plan, DerivationHook):
        return True
    names = [f"{component}.{key}" for component, key, _ in source_rows(source)]
    if isinstance(plan, SelectionPlan):
        return all(any(fnmatch.fnmatchcase(n, rule) for n in names) for rule in plan.include)
    if not set(plan.components) <= set(source.components):
        return False
    selected = [n for n in names if n.partition(".")[0] in plan.components]
    return all(
        any(fnmatch.fnmatchcase(n, rule) for n in selected) for rule in plan.include
    ) and all(f"{c}.{key}" in names for c in plan.components for key in plan.keys)
