"""cr-076: ``plan(components, encoding)`` + ``quantize(source, plan)`` at the dozen-line bar.

One thin layer over the existing derivation math in ``.quantization`` — a per-family
quantize job is its model class, one ``plan(...)``, and its threshold policy. ``quantize``
opens one declared lane transaction, normalizes every plain float tensor to BF16, encodes
the plan's selected GEMM weights, and runs the tier-1 tripwire INSIDE (per-tensor
round-trip against the encoding bound, worst relative Frobenius against the declared
threshold), refusing typed with code ``quantization_tripwire`` on encoding breakage.
Plain non-normalizing derivation stays ``derive_fp8``/``derive_mxfp8``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import msgspec

from cozy_runtime.author import (
    Context,
    Model,
    Telemetry,
    UnsupportedInput,
    WeightsSink,
    WeightsSource,
    WeightsTarget,
)

from .quantization import (
    _ENCODINGS,
    ArtifactQuantizationRequest,
    bf16_targets,
    inherited_configs,
    prepare_bf16,
    prepare_source_quantization,
    quantization_additions,
    quantize_component_into,
    write_bf16,
)


class QuantizePlan(msgspec.Struct, frozen=True):
    """One lane intent: BF16 normalization plus ``encoding`` over ``components``' weights.

    ``encoding=None`` is the pure BF16 lane. ``max_relative_frobenius`` is the caller's
    tier-1 threshold policy over the worst per-tensor round-trip; the encoding's own
    round-trip bound is always enforced underneath it.
    """

    components: tuple[str, ...]
    encoding: str | None
    max_relative_frobenius: float | None


class QuantizeResult(msgspec.Struct):
    """What one committed lane reports — directly returnable from a job body."""

    output: str
    weights_transaction_id: str
    tensorfs_receipt_digest: str
    replayed: bool
    encoded_keys: int
    converted_keys: int
    inherited_keys: int
    source_bytes_read_this_run: int
    new_bytes_written_this_run: int
    saturated_elements_this_run: int
    worst_relative_frobenius: float


def plan(
    components: Sequence[str] = (),
    encoding: str | None = None,
    *,
    max_relative_frobenius: float | None = None,
) -> QuantizePlan:
    """Build one reviewed lane plan; selection resolves against the source in ``quantize``."""
    selected = tuple(components)
    if encoding is None:
        if selected:
            raise UnsupportedInput(
                "a plain BF16 lane selects no components; name an encoding to quantize",
                code="quantization_plan",
            )
        if max_relative_frobenius is not None:
            raise UnsupportedInput(
                "a tripwire threshold needs an encoding lane", code="quantization_plan"
            )
    else:
        if encoding not in _ENCODINGS:
            raise UnsupportedInput(
                f"unsupported quantization encoding {encoding!r}", code="quantization_plan"
            )
        if not selected or len(set(selected)) != len(selected):
            raise UnsupportedInput(
                "quantization components must be non-empty and unique",
                code="quantization_plan",
            )
        if max_relative_frobenius is not None and (
            not math.isfinite(max_relative_frobenius) or max_relative_frobenius < 0
        ):
            raise UnsupportedInput(
                "max_relative_frobenius must be one finite nonnegative value",
                code="quantization_plan",
            )
    return QuantizePlan(selected, encoding, max_relative_frobenius)


def quantize(
    source: Model[Any] | WeightsSource,
    plan: QuantizePlan,
    /,
    *,
    sink: WeightsSink,
    ctx: Context,
    tel: Telemetry,
    output: str = "model",
) -> QuantizeResult:
    """Run one lane end-to-end and commit it, tier-1 tripwire inside.

    ``source`` is the granted source Model (constructed or derive-only) or the
    ``WeightsSink.structure`` view already fetched from it — both reach the same tensors.
    """
    model, structure = _resolve(source, sink)
    bf16 = prepare_bf16(structure)
    if plan.encoding is None:
        artifact = None
        selected: set[tuple[str, str]] = set()
        targets = bf16_targets(bf16, source="source")
    else:
        artifact = prepare_source_quantization(structure, components=plan.components)
        selected = {(tensor.component, tensor.key) for tensor in artifact.tensors}
        targets = bf16_targets(bf16, source="source", excluded=selected)
        for component in artifact.components:
            combined = {
                **targets[component].add,
                **quantization_additions(
                    plan.encoding, artifact, component, target_logical_dtype="bf16"
                ),
            }
            targets[component] = WeightsTarget(
                source="source",
                source_component=component,
                drop=tuple(sorted(combined)),
                add=combined,
            )
    converted = [
        tensor for tensor in bf16.tensors if (tensor.component, tensor.key) not in selected
    ]
    encoded = 0 if artifact is None else len(artifact.tensors)
    inherited = len(bf16.order) - len(converted) - encoded

    with sink.open(
        output,
        sources={"source": model},
        targets=targets,
        configs=inherited_configs(bf16, source="source"),
        order=bf16.order,
    ) as transaction:
        if transaction.replayed:
            receipt = transaction.receipt
            assert receipt is not None
            return QuantizeResult(
                output=output,
                weights_transaction_id=receipt.weights_transaction_id,
                tensorfs_receipt_digest=receipt.tensorfs_receipt_digest,
                replayed=True,
                encoded_keys=encoded,
                converted_keys=len(converted),
                inherited_keys=inherited,
                source_bytes_read_this_run=0,
                new_bytes_written_this_run=0,
                saturated_elements_this_run=0,
                worst_relative_frobenius=0.0,
            )
        read = write_bf16(
            transaction, ctx, tel, plan=bf16, source="source", excluded=selected or None
        )
        written = sum(2 * math.prod(tensor.shape) for tensor in converted)
        saturated = 0
        worst = 0.0
        if artifact is not None and plan.encoding is not None:
            payload = ArtifactQuantizationRequest(
                max_relative_frobenius=plan.max_relative_frobenius
            )
            for component in artifact.components:
                stats = quantize_component_into(
                    transaction,
                    ctx,
                    payload,
                    tel,
                    encoding=plan.encoding,
                    plan=artifact,
                    component=component,
                    source="source",
                    source_component=component,
                    target_component=component,
                )
                read += stats.source_bytes_read
                written += stats.new_bytes_written
                saturated += stats.saturated_elements
                worst = max(worst, stats.worst_relative_frobenius)
        receipt = transaction.commit()
    return QuantizeResult(
        output=output,
        weights_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=receipt.tensorfs_receipt_digest,
        replayed=False,
        encoded_keys=encoded,
        converted_keys=len(converted),
        inherited_keys=inherited,
        source_bytes_read_this_run=read,
        new_bytes_written_this_run=written,
        saturated_elements_this_run=saturated,
        worst_relative_frobenius=worst,
    )


def _resolve(
    source: Model[Any] | WeightsSource, sink: WeightsSink
) -> tuple[Model[Any], WeightsSource]:
    """The granted source Model and its structure view, whichever half was handed in."""
    if isinstance(source, WeightsSource):
        granted = {id(model): model for model in sink.models.values()}
        if len(granted) != 1:
            raise UnsupportedInput(
                "quantize over a pre-fetched structure is unambiguous only for a job with "
                "exactly one granted model input; pass the source Model instead",
                code="quantization_source",
            )
        return next(iter(granted.values())), source
    return source, sink.structure(source)
