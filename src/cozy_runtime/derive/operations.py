"""Shared managed derivations; callers supply policy data, never numerical code."""

from __future__ import annotations

import fnmatch
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.author import (
    MAX_WEIGHTS_NEW_BYTES,
    App,
    Context,
    DerivationHook,
    MemoDistribution,
    Model,
    ModelArtifact,
    QuantizationPlan,
    SelectionPlan,
    Telemetry,
    UnsupportedInput,
    Weights,
    WeightsOutput,
    invocable,
)
from cozy_runtime.author._recipes import DeclaredRecipe, Plan, canonical, fingerprint
from cozy_runtime.weights_store import Checkpoint, Derivation

from . import facade, native_quantization, svdq
from .quantization import (
    _ENCODINGS,
    ArtifactQuantizationPlan,
    QuantizationTensor,
    column_block,
    prepare_source_quantization,
    refuse_quantized,
    source_rows,
)
from .recipes import declared_plan, declared_recipe

# This is the platform's representable grant ceiling, not a model-size estimate.
# Writer admission uses actual remaining added-role geometry (cr-151).
MAX_QUANTIZED_BYTES = MAX_WEIGHTS_NEW_BYTES
app = App()


def _selection(source: Checkpoint, plan: QuantizationPlan) -> ArtifactQuantizationPlan:
    encoding = plan.encoding
    if (
        not 1 <= len(plan.components) <= 64
        or len(set(plan.components)) != len(plan.components)
        or any(not name for name in plan.components)
        or len(plan.keys) > 65_536
        or len(set(plan.keys)) != len(plan.keys)
        or any(not key for key in plan.keys)
        or plan.output_precision not in {"normalize-f32-to-bf16", "preserve"}
        or len(set(plan.source_order_digests)) != len(plan.source_order_digests)
        or (plan.keys and plan.include)
        or len(set(plan.include)) != len(plan.include)
        or len({rule.pattern for rule in plan.encodings}) != len(plan.encodings)
        or any(rule.encoding not in _ENCODINGS for rule in plan.encodings)
    ):
        raise UnsupportedInput(
            "quantization plan has invalid or repeated selections", code="quantization_plan"
        )
    digests = plan.source_order_digests + (
        (plan.selected_schema_digest,) if plan.selected_schema_digest is not None else ()
    )
    if any(re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None for digest in digests):
        raise UnsupportedInput(
            "quantization plan requires canonical digests", code="quantization_plan"
        )
    rows = list(source_rows(source))
    by_key = {(component, key): tensor for component, key, tensor in rows}
    if len(by_key) != len(rows):
        raise UnsupportedInput("quantization source repeats a tensor", code="quantization_source")
    refuse_quantized(rows, plan.components)
    if (
        plan.source_order_digests
        and canonical_json.digest([[component, key] for component, key, _ in rows])
        not in plan.source_order_digests
    ):
        raise UnsupportedInput(
            "quantization source construction order differs", code="quantization_source"
        )
    if plan.include:
        matched = [
            (component, key)
            for component, key in by_key
            if component in plan.components
            and any(fnmatch.fnmatchcase(f"{component}.{key}", rule) for rule in plan.include)
        ]
        unused = [
            rule
            for rule in plan.include
            if not any(fnmatch.fnmatchcase(f"{c}.{k}", rule) for c, k in matched)
        ]
        if unused or not matched:
            raise UnsupportedInput(
                f"quantization include patterns match no weight: {unused or plan.include}",
                code="quantization_source",
            )
        pairs = matched
    else:
        pairs = [(component, key) for component in plan.components for key in plan.keys]

    def encoding_of(component: str, key: str) -> str | None:
        name = f"{component}.{key}"
        return next(
            (rule.encoding for rule in plan.encodings if fnmatch.fnmatchcase(name, rule.pattern)),
            None,
        )

    if plan.encodings and not pairs:
        raise UnsupportedInput("per-weight encodings need a selection", code="quantization_plan")
    unused = [
        rule.pattern
        for rule in plan.encodings
        if not any(fnmatch.fnmatchcase(f"{c}.{k}", rule.pattern) for c, k in pairs)
    ]
    if unused:
        raise UnsupportedInput(
            f"quantization encoding patterns match no selected weight: {unused}",
            code="quantization_plan",
        )
    if pairs:
        selected: list[QuantizationTensor] = []
        for component, key in pairs:
            row = by_key.get((component, key))
            chosen = encoding_of(component, key)
            block = column_block(chosen or encoding)
            if row is None or (
                not key.endswith(".weight")
                or not row.readable
                or row.logical_dtype not in {"f16", "bf16", "f32"}
                or len(row.shape) != 2
                or row.shape[0] <= 0
                or row.shape[1] <= 0
                or row.shape[1] % block
            ):
                raise UnsupportedInput(
                    f"quantization source has no supported exact weight {component}.{key}",
                    code="quantization_source",
                )
            selected.append(
                QuantizationTensor(
                    component,
                    key,
                    row.logical_dtype,
                    (row.shape[0], row.shape[1]),
                    encoding=chosen,
                )
            )
        selection = ArtifactQuantizationPlan(list(plan.components), list(by_key), selected)
    else:
        selection = prepare_source_quantization(
            source, components=plan.components, block=column_block(encoding)
        )
        selection.tensors.sort(key=lambda row: plan.components.index(row.component))
    schema = []
    for item in selection.tensors:
        tensor = by_key[item.component, item.key]
        if not tensor.readable:
            raise UnsupportedInput(
                f"quantization source {item.component}.{item.key} is neither plain nor per-row",
                code="quantization_source",
            )
        schema.append([item.component, item.key, tensor.logical_dtype, list(tensor.shape)])
    if (
        plan.selected_schema_digest is not None
        and canonical_json.digest(schema) != plan.selected_schema_digest
    ):
        raise UnsupportedInput(
            "quantization source selected geometry differs", code="quantization_source"
        )
    return selection


@invocable(
    memo_version="quantize/1",
    memo_dependencies=(
        _selection,
        "cozy_runtime.derive.facade",
        "cozy_runtime.derive.quantization",
        "cozy_runtime.derive.native_quantization",
        "cozy_runtime.derive.microscale",
        "cozy_runtime.derive.safetensors_io",
        "cozy_runtime.weights_store",
        MemoDistribution("tensorfs"),
        MemoDistribution("numpy"),
        MemoDistribution("msgspec"),
    ),
)
async def quantize(
    ctx: Context,
    *,
    source: Weights[Any],
    plan: QuantizationPlan,
    tel: Telemetry,
    max_relative_frobenius: float | None = None,
) -> ModelArtifact:
    """Quantize one granted source under a closed caller-authored selection plan."""
    return quantize_into(
        ctx,
        tel,
        source=source,
        plan=plan,
        output="model",
        max_relative_frobenius=max_relative_frobenius,
    )


def _opened(ctx: Context, source: Weights[Any] | Checkpoint) -> Checkpoint:
    return source if isinstance(source, Checkpoint) else ctx.tensorfs_source(source)


def quantize_into(
    ctx: Context,
    tel: Telemetry,
    *,
    source: Weights[Any] | Checkpoint,
    plan: QuantizationPlan,
    output: str,
    max_relative_frobenius: float | None = None,
    calibration: svdq.Calibration | None = None,
    fit_workspace_bytes: int = native_quantization.FIT_WORKSPACE_BYTES,
) -> ModelArtifact:
    """`plan`'s lane, written into the CALLING job's weights output `output`. A plan with
    `svdq.nvfp4` weights takes the `calibration` its recipe recorded (`calibrate`)."""
    ctx.raise_if_cancelled()
    intent = facade.plan(
        plan.components, plan.encoding, max_relative_frobenius=max_relative_frobenius
    )
    opened = _opened(ctx, source)
    result, artifact = facade._quantize(
        opened,
        intent,
        ctx=ctx,
        tel=tel,
        output=output,
        selection=None if plan.encoding is None else _selection(opened, plan),
        preserve_precision=plan.output_precision == "preserve",
        calibration=calibration,
        fit_workspace_bytes=fit_workspace_bytes,
    )
    return facade._artifact_result(result, artifact, tel, output)


def select_into(
    ctx: Context,
    tel: Telemetry,
    *,
    source: Weights[Any] | Checkpoint,
    plan: SelectionPlan,
    output: str,
) -> ModelArtifact:
    """`plan`'s weights of `source`, written into the CALLING job's weights output `output`."""
    ctx.raise_if_cancelled()
    opened = _opened(ctx, source)
    rows = [(c, key) for c, key, _ in source_rows(opened)]

    def kept(component: str, key: str) -> bool:
        return any(fnmatch.fnmatchcase(f"{component}.{key}", rule) for rule in plan.include)

    order = [(c, key) for c, key in rows if kept(c, key)]
    unused = [
        rule
        for rule in plan.include
        if not any(fnmatch.fnmatchcase(f"{c}.{key}", rule) for c, key in order)
    ]
    if not plan.include or len(set(plan.include)) != len(plan.include) or unused:
        raise UnsupportedInput(
            f"selection patterns match no weight: {unused or list(plan.include)}",
            code="selection_plan",
        )
    with ctx.output(output, Derivation(base=None, sources=(opened,), order=order)) as out:
        out.inherit(opened)
        for component, key in order:
            if not out.has(component, key):
                out.copy(component, key, opened)
        artifact = ctx.adopt_model(out)
    tel.log("selection", output=output, kept_keys=len(order), dropped_keys=len(rows) - len(order))
    return artifact


app.entrypoint(
    quantize,
    weights=(WeightsOutput("model", max_new_bytes=MAX_QUANTIZED_BYTES),),
    accelerator=False,
)


def calibrate(
    ctx: Context,
    torch: Any,
    *,
    source: Weights[Any] | Checkpoint,
    components: dict[str, Any],
    plan: QuantizationPlan,
) -> svdq.Calibration:
    """A recorder of the inputs of every linear `plan` encodes as `svdq.nvfp4` (by its rules or
    as the lane's encoding): open it (`with`) while the model whose `components` (name -> module)
    hold those linears runs real inputs, then hand it to `quantize_into`."""
    selection = _selection(_opened(ctx, source), plan)
    keys = [
        f"{t.component}.{t.key}"
        for t in selection.tensors
        if (t.encoding or plan.encoding) == "svdq.nvfp4"
    ]
    if not keys:
        raise UnsupportedInput("the plan encodes no weight as svdq.nvfp4", code="quantization_plan")
    return svdq.Calibration(torch, components, keys)


def run_recipe(
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
    """The Runtime's side of `recipes.quantize`, which an executor installs."""
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
    ctx._tensorfs_output = lambda slot, made: opener(
        slot, replace(made, provenance={**provenance, **(made.provenance or {})})
    )
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
