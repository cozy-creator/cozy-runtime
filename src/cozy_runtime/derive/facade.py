"""cr-076: ``plan(components, encoding)`` + ``quantize(source, plan)`` at the dozen-line bar.

One thin layer over the existing derivation math in ``.quantization`` — a per-family
ingest producer is its model class, one ``plan(...)``, and its threshold policy.
``quantize`` opens one declared lane transaction, normalizes every plain f32 tensor to
BF16 as a recorded derivation (FP16 is opt-in; ``keep``-listed components stay
at source precision; per-tensor round-trip stats recorded on the cut), encodes the
plan's selected GEMM weights, and runs the tier-1 tripwire INSIDE (per-tensor
round-trip against the encoding bound, worst relative Frobenius against the declared
threshold), refusing typed with code ``quantization_tripwire`` on encoding breakage.
The ``encoding=None`` lane is the canonical ingest lane and its result says so.
The managed operation can preserve source precision through its explicit policy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import msgspec
from tensorfs.derived import Derivation, SourceInspection, Target, derive

from cozy_runtime.author import (
    Context,
    Model,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
)

if TYPE_CHECKING:
    from .quantization import ArtifactQuantizationPlan


class QuantizePlan(msgspec.Struct, frozen=True):
    """One lane intent: BF16 normalization plus ``encoding`` over ``components``' weights.

    ``encoding=None`` is the canonical plain lane (#695): f32 normalizes to bf16
    by default. ``bf16_sources`` may explicitly include f16 for a BF16 execution
    lane; other tensors and ``keep``-listed components inherit at source precision.
    ``max_relative_frobenius`` is the caller's tier-1
    threshold policy over the worst per-tensor round-trip; the encoding's own
    round-trip bound is always enforced underneath it.
    """

    components: tuple[str, ...]
    encoding: str | None
    max_relative_frobenius: float | None
    keep: tuple[str, ...] = ()
    bf16_sources: tuple[str, ...] = ("f32",)


class QuantizeResult(msgspec.Struct):
    """What one committed lane reports — directly returnable from a job body."""

    output: str
    weights_transaction_id: str
    tensorfs_receipt_digest: str
    replayed: bool
    encoded_keys: int
    reused_keys: int
    converted_keys: int
    inherited_keys: int
    source_bytes_read_this_run: int
    new_bytes_written_this_run: int
    saturated_elements_this_run: int
    worst_relative_frobenius: float | None
    #: Selected float tensors held at source precision by the plan's keep list.
    kept_keys: int
    #: Worst per-tensor relative Frobenius of the selected float→bf16 cut.
    normalization_worst_relative_frobenius: float | None
    #: True on the plain lane: the ingest record marks the canonical bf16 lane (#695).
    canonical: bool


def plan(
    components: Sequence[str] = (),
    encoding: str | None = None,
    *,
    keep: Sequence[str] = (),
    bf16_sources: Sequence[str] = ("f32",),
    max_relative_frobenius: float | None = None,
) -> QuantizePlan:
    """Build one reviewed lane plan; selection resolves against the source in ``quantize``.

    ``keep`` is the ingest config's per-component keep list (#695): a named component's
    selected float tensors inherit at source precision instead of normalizing to bf16.
    ``bf16_sources=("f16", "f32")`` also prepares FP16 adapters ahead of inference.
    """
    from .quantization import _ENCODINGS, bf16_source_dtypes

    selected = tuple(components)
    kept = tuple(keep)
    if len(set(kept)) != len(kept) or any(not name for name in kept):
        raise UnsupportedInput(
            "keep must name unique non-empty components", code="quantization_plan"
        )
    overlap = sorted(set(kept) & set(selected))
    if overlap:
        raise UnsupportedInput(
            f"components {overlap} are keep-listed and cannot also be quantized",
            code="quantization_plan",
        )
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
    return QuantizePlan(
        selected, encoding, max_relative_frobenius, kept, bf16_source_dtypes(bf16_sources)
    )


def quantize(
    source: Model[Any] | SourceInspection,
    plan: QuantizePlan,
    /,
    *,
    ctx: Context,
    tel: Telemetry,
    output: str = "model",
) -> QuantizeResult:
    """Run one lane end-to-end and commit it, tier-1 tripwire inside.

    ``source`` is the granted source Model (constructed or derive-only) or the
    native source inspection already fetched from it — both reach the same tensors.
    """
    return _quantize(source, plan, ctx=ctx, tel=tel, output=output)[0]


def quantize_artifact(
    source: Model[Any] | SourceInspection,
    plan: QuantizePlan,
    /,
    *,
    ctx: Context,
    tel: Telemetry,
    output: str = "model",
) -> ModelArtifact:
    """Run the same lane kernel and return its immutable managed-call artifact.

    Measurements and receipt provenance remain on the original execution. The result
    contains no per-attempt counters, so memoized operations can retain it unchanged.
    """
    result, receipt = _quantize(source, plan, ctx=ctx, tel=tel, output=output)
    return _artifact_result(result, receipt, tel, output)


def _artifact_result(
    result: QuantizeResult, artifact: ModelArtifact, tel: Telemetry, output: str
) -> ModelArtifact:
    tel.log(
        "quantization measurement",
        output=output,
        encoded_keys=result.encoded_keys,
        reused_keys=result.reused_keys,
        source_bytes_read_this_run=result.source_bytes_read_this_run,
        new_bytes_written_this_run=result.new_bytes_written_this_run,
        saturated_elements_this_run=result.saturated_elements_this_run,
        worst_relative_frobenius=result.worst_relative_frobenius,
        normalization_worst_relative_frobenius=result.normalization_worst_relative_frobenius,
    )
    return artifact


def _quantize(
    source: Model[Any] | SourceInspection,
    plan: QuantizePlan,
    /,
    *,
    ctx: Context,
    tel: Telemetry,
    output: str,
    selection: ArtifactQuantizationPlan | None = None,
    preserve_precision: bool = False,
    source_view: SourceInspection | None = None,
) -> tuple[QuantizeResult, ModelArtifact]:
    from .quantization import (
        ArtifactQuantizationRequest,
        bf16_targets,
        inherited_configs,
        prepare_bf16,
        prepare_source_quantization,
        quantization_additions,
        quantize_component_into,
        source_rows,
        write_bf16,
    )

    if isinstance(source, SourceInspection):
        source_capability = source.source
        structure = source
    else:
        source_capability = ctx.tensorfs_source(source)
        structure = source_view or source_capability.inspect()
    keep = (
        tuple(dict.fromkeys(component for component, _, _ in source_rows(structure)))
        if preserve_precision
        else plan.keep
    )
    bf16 = prepare_bf16(structure, keep=keep, source_dtypes=plan.bf16_sources)
    canonical = plan.encoding is None
    if plan.encoding is None:
        artifact = None
        selected: set[tuple[str, str]] = set()
        targets = bf16_targets(bf16, source="source")
    else:
        artifact = selection or prepare_source_quantization(structure, components=plan.components)
        selected = {(tensor.component, tensor.key) for tensor in artifact.tensors}
        targets = bf16_targets(bf16, source="source", excluded=selected)
        for component in artifact.components:
            # Encoded tensors follow the same explicit logical-dtype cut as the
            # unselected tensors. The encoding still reads the original values.
            additions = quantization_additions(plan.encoding, artifact, component)
            combined = {
                **targets[component].add,
                **{
                    key: replace(tensor, logical_dtype="bf16")
                    if tensor.logical_dtype in plan.bf16_sources and not preserve_precision
                    else tensor
                    for key, tensor in additions.items()
                },
            }
            targets[component] = Target(
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

    configs = inherited_configs(bf16, source="source")
    manager = derive(
        ctx.output(output), Derivation({"source": source_capability}, targets, configs, bf16.order)
    )
    with manager as transaction:
        if transaction.receipt is not None:
            receipt = transaction.receipt
            result_artifact = ctx.adopt_model(receipt)
            transaction_id = receipt["transaction_id"]
            return QuantizeResult(
                output=output,
                weights_transaction_id=transaction_id,
                tensorfs_receipt_digest=result_artifact.tensorfs_receipt_digest,
                replayed=True,
                encoded_keys=encoded,
                reused_keys=encoded + len(converted),
                converted_keys=len(converted),
                inherited_keys=inherited,
                source_bytes_read_this_run=0,
                new_bytes_written_this_run=0,
                saturated_elements_this_run=0,
                worst_relative_frobenius=None,
                kept_keys=len(bf16.kept),
                normalization_worst_relative_frobenius=None,
                canonical=canonical,
            ), result_artifact
        normalization = write_bf16(
            transaction, ctx, tel, plan=bf16, source="source", excluded=selected or None
        )
        read = normalization.source_bytes_read
        written = normalization.new_bytes_written
        saturated = 0
        worst: float | None = 0.0
        reused = 0
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
                reused += stats.reused_keys
                worst = (
                    max(worst, stats.worst_relative_frobenius)
                    if worst is not None and stats.worst_relative_frobenius is not None
                    else None
                )
        receipt = transaction.commit()
        result_artifact = ctx.adopt_model(receipt)
        transaction_id = receipt["transaction_id"]
    return QuantizeResult(
        output=output,
        weights_transaction_id=transaction_id,
        tensorfs_receipt_digest=result_artifact.tensorfs_receipt_digest,
        replayed=False,
        encoded_keys=encoded,
        reused_keys=reused,
        converted_keys=len(converted),
        inherited_keys=inherited,
        source_bytes_read_this_run=read,
        new_bytes_written_this_run=written,
        saturated_elements_this_run=saturated,
        worst_relative_frobenius=worst,
        kept_keys=len(bf16.kept),
        normalization_worst_relative_frobenius=normalization.worst_relative_frobenius,
        canonical=canonical,
    ), result_artifact
