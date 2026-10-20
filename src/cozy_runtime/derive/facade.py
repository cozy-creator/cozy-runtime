"""cr-076: ``plan(components, encoding)`` + ``quantize(source, plan)``.

``quantize`` opens one declared output over the source, normalizes every plain f32 tensor to
BF16 (FP16 opt-in; ``keep``-listed components stay at source precision; per-tensor round-trip
stats recorded), encodes the plan's selected weights, and runs the tier-1 tripwire inside
(round-trip against the scheme's bound, worst relative Frobenius against the declared
threshold), refusing ``quantization_tripwire``. ``encoding=None`` is the canonical bf16 lane.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import msgspec

from cozy_runtime.author import (
    Context,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    Weights,
)
from cozy_runtime.weights_store import Checkpoint, Derivation

from .quantization import (
    _ENCODINGS,
    ArtifactQuantizationPlan,
    ArtifactQuantizationRequest,
    bf16_source_dtypes,
    column_block,
    prepare_bf16,
    prepare_source_quantization,
    quantize_component_into,
    source_rows,
    write_bf16,
)


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
    """What one committed lane reports — directly returnable from a function body."""

    output: str
    weights_transaction_id: str
    #: the committed checkpoint's header ref
    checkpoint: str
    #: nothing was written this run: the output was committed, or journaled whole, before
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
    source: Weights[Any] | Checkpoint,
    plan: QuantizePlan,
    /,
    *,
    ctx: Context,
    tel: Telemetry,
    output: str = "model",
) -> QuantizeResult:
    """Run one lane end-to-end and commit it, tier-1 tripwire inside. ``source`` is a granted
    ``Weights`` argument or the source already opened from it."""
    return _quantize(source, plan, ctx=ctx, tel=tel, output=output)[0]


def quantize_artifact(
    source: Weights[Any] | Checkpoint,
    plan: QuantizePlan,
    /,
    *,
    ctx: Context,
    tel: Telemetry,
    output: str = "model",
) -> ModelArtifact:
    """The same lane, returning its immutable managed-call artifact (no per-attempt counters,
    so memoized operations can retain it unchanged)."""
    result, artifact = _quantize(source, plan, ctx=ctx, tel=tel, output=output)
    return _artifact_result(result, artifact, tel, output)


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
    source: Weights[Any] | Checkpoint,
    plan: QuantizePlan,
    /,
    *,
    ctx: Context,
    tel: Telemetry,
    output: str,
    selection: ArtifactQuantizationPlan | None = None,
    preserve_precision: bool = False,
) -> tuple[QuantizeResult, ModelArtifact]:
    opened = source if isinstance(source, Checkpoint) else ctx.tensorfs_source(source)
    keep = (
        tuple(dict.fromkeys(component for component, _, _ in source_rows(opened)))
        if preserve_precision
        else plan.keep
    )
    bf16 = prepare_bf16(opened, keep=keep, source_dtypes=plan.bf16_sources)
    artifact = None
    if plan.encoding is not None:
        artifact = selection or prepare_source_quantization(
            opened, components=plan.components, block=column_block(plan.encoding)
        )
    selected = {(t.component, t.key) for t in artifact.tensors} if artifact else set()
    converted = [t for t in bf16.tensors if (t.component, t.key) not in selected]
    encoded = len(selected)
    inherited = len(bf16.order) - len(converted) - encoded

    def logical(tensor: Any) -> str:
        if tensor.logical_dtype in plan.bf16_sources and not preserve_precision:
            return "bf16"
        return str(tensor.logical_dtype)

    with ctx.output(output, Derivation(base=opened, order=bf16.order)) as out:
        normalization = write_bf16(opened, out, ctx, tel, plan=bf16, excluded=selected or None)
        read, written = normalization.source_bytes_read, normalization.new_bytes_written
        reused, saturated = normalization.reused_keys, 0
        worst: float | None = 0.0
        if artifact is not None and plan.encoding is not None:
            request = ArtifactQuantizationRequest(
                max_relative_frobenius=plan.max_relative_frobenius
            )
            for component in artifact.components:
                stats = quantize_component_into(
                    opened, out, ctx, request, tel, encoding=plan.encoding, plan=artifact,
                    component=component, logical_dtype=logical,
                )  # fmt: skip
                read += stats.source_bytes_read
                written += stats.new_bytes_written
                saturated += stats.saturated_elements
                reused += stats.reused_keys
                worst = (
                    max(worst, stats.worst_relative_frobenius)
                    if worst is not None and stats.worst_relative_frobenius is not None
                    else None
                )
        result_artifact = ctx.adopt_model(out)
        transaction = out.transaction
    return QuantizeResult(
        output=output,
        weights_transaction_id=transaction,
        checkpoint=result_artifact.checkpoint,
        replayed=reused == encoded + len(converted),
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
        canonical=plan.encoding is None,
    ), result_artifact
