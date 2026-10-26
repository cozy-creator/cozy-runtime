"""Shared managed derivations; callers supply policy data, never numerical code."""

from __future__ import annotations

import fnmatch
import re
from typing import Any

from cozy_runtime import canonical_json
from cozy_runtime.author import (
    MAX_WEIGHTS_NEW_BYTES,
    App,
    Context,
    MemoDistribution,
    ModelArtifact,
    QuantizationPlan,
    SelectionPlan,
    Telemetry,
    UnsupportedInput,
    Weights,
    WeightsOutput,
    invocable,
)
from cozy_runtime.weights_store import Checkpoint, Derivation

from . import facade, svdq
from .quantization import (
    _ENCODINGS,
    ArtifactQuantizationPlan,
    QuantizationTensor,
    column_block,
    prepare_source_quantization,
    source_rows,
)

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
    fit_workspace_bytes: int = 512 << 20,
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
