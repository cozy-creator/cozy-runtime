"""Shared managed derivations; callers supply policy data, never numerical code."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.author import (
    MAX_WEIGHTS_NEW_BYTES,
    App,
    Context,
    MemoDistribution,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    Weights,
    WeightsOutput,
    invocable,
)

from . import facade

if TYPE_CHECKING:
    from tensorfs.derived import SourceInspection

    from .quantization import ArtifactQuantizationPlan


class QuantizationPlan(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Closed selection and source compatibility policy for generic quantization."""

    components: tuple[str, ...]
    keys: tuple[str, ...] = ()
    selected_schema_digest: str | None = None
    source_order_digests: tuple[str, ...] = ()
    output_precision: Literal["normalize-f32-to-bf16", "preserve"] = "normalize-f32-to-bf16"


# This is the platform's representable grant ceiling, not a model-size estimate.
# Writer admission uses actual remaining added-role geometry (cr-151).
MAX_QUANTIZED_BYTES = MAX_WEIGHTS_NEW_BYTES
app = App()


def _selection(structure: SourceInspection, plan: QuantizationPlan) -> ArtifactQuantizationPlan:
    from .quantization import (
        PLAIN_SPEC,
        ArtifactQuantizationPlan,
        QuantizationTensor,
        prepare_source_quantization,
        source_parts,
        source_rows,
    )

    if (
        not 1 <= len(plan.components) <= 64
        or len(set(plan.components)) != len(plan.components)
        or any(not name for name in plan.components)
        or len(plan.keys) > 65_536
        or len(set(plan.keys)) != len(plan.keys)
        or any(not key for key in plan.keys)
        or plan.output_precision not in {"normalize-f32-to-bf16", "preserve"}
        or len(set(plan.source_order_digests)) != len(plan.source_order_digests)
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
    rows = list(source_rows(structure))
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
    if plan.keys:
        selected: list[QuantizationTensor] = []
        for component in plan.components:
            for key in plan.keys:
                row = by_key.get((component, key))
                if row is None or (
                    not key.endswith(".weight")
                    or row.logical_dtype not in {"f16", "bf16", "f32"}
                    or len(row.shape) != 2
                    or row.shape[0] <= 0
                    or row.shape[1] <= 0
                    or row.shape[1] % 32
                ):
                    raise UnsupportedInput(
                        f"quantization source has no supported exact weight {component}.{key}",
                        code="quantization_source",
                    )
                selected.append(
                    QuantizationTensor(
                        component, key, row.logical_dtype, (row.shape[0], row.shape[1])
                    )
                )
        selection = ArtifactQuantizationPlan(
            list(plan.components), list(structure.configs), list(by_key), selected
        )
    else:
        selection = prepare_source_quantization(structure, components=plan.components)
        selection.tensors.sort(key=lambda row: plan.components.index(row.component))
    schema = []
    for item in selection.tensors:
        tensor = by_key[item.component, item.key]
        parts = source_parts(tensor)
        plain = parts.get("value")
        if (
            tensor.encoding != PLAIN_SPEC
            or len(parts) != 1
            or plain is None
            or plain.dtype != tensor.logical_dtype
            or tuple(plain.shape) != tuple(tensor.shape)
        ):
            raise UnsupportedInput(
                f"quantization source {item.component}.{item.key} is not plain",
                code="quantization_source",
            )
        schema.append(
            [
                item.component,
                item.key,
                tensor.logical_dtype,
                list(tensor.shape),
                [[role, part.dtype, list(part.shape)] for role, part in parts.items()],
            ]
        )
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
        "cozy_runtime.derive.microscale",
        "cozy_runtime.derive.safetensors_io",
        "tensorfs.derived",
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
    encoding: str,
    tel: Telemetry,
    max_relative_frobenius: float | None = None,
) -> ModelArtifact:
    """Quantize one granted source under a closed caller-authored selection plan."""
    ctx.raise_if_cancelled()
    intent = facade.plan(plan.components, encoding, max_relative_frobenius=max_relative_frobenius)
    structure = ctx.tensorfs_source(source).inspect()
    selected = _selection(structure, plan)
    result, receipt = facade._quantize(
        source,
        intent,
        ctx=ctx,
        tel=tel,
        output="model",
        selection=selected,
        source_view=structure,
        preserve_precision=plan.output_precision == "preserve",
    )
    return facade._artifact_result(result, receipt, tel, "model")


app.entrypoint(
    quantize,
    weights=(WeightsOutput("model", max_new_bytes=MAX_QUANTIZED_BYTES),),
    accelerator=False,
)
