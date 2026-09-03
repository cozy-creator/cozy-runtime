"""Carrier-free FP8 and MXFP8 model derivation jobs.

The worker receives one exact TensorFS Manifest capability and returns another through
``WeightsSink``. It never receives a filesystem path, store path, border binary, or
publication credential. Publication belongs to Tensorhub's production transaction.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from importlib.resources import files
from typing import Any, NamedTuple, Protocol, cast

import msgspec
import numpy as np

from cozy_runtime.author import (
    Context,
    Model,
    Telemetry,
    UnsupportedInput,
    WeightsConfig,
    WeightsPart,
    WeightsSink,
    WeightsTarget,
    WeightsTensor,
    WeightsTransaction,
)

from . import safetensors_io as st
from .microscale import (
    Encoded,
    encode_fp8_rowwise,
    encode_mxfp8,
)


class _Encoding(NamedTuple):
    """The encoder and its maximum reconstruction error as a multiple of tensor amax."""

    round_trip_bound: float
    encode: Callable[[str, np.ndarray], Encoded]


_ENCODINGS: dict[str, _Encoding] = {
    "mxfp8/1": _Encoding(
        round_trip_bound=2.0**-3,
        encode=encode_mxfp8,
    ),
    "fp8-rowwise/1": _Encoding(
        round_trip_bound=2.0**-3,
        encode=encode_fp8_rowwise,
    ),
}

#: RNE fp32->bf16 keeps every element within amax * 2^-8 (8-bit significand, half-ulp).
#: The normalization's own tier-1 bound — enforced on every cut, stats recorded (#695).
BF16_ROUND_TRIP_BOUND = 2.0**-8


class EncodingConformance(msgspec.Struct):
    """What these kinds PROVE: byte layout, decoder agreement, reproducibility (#549.6).
    Necessary for any publication; sufficient only for an explicitly quality-unapproved
    one."""

    passed: bool
    basis: list[str]


class QuantizationSource(Model[object]):
    """Exact TensorFS Manifest input. The job reads it only through WeightsSink."""

    def load(self, loader: Any) -> None:
        del loader


class QuantizationTensor(msgspec.Struct, forbid_unknown_fields=True):
    """One source tensor selected by the reviewed structural plan."""

    component: str
    key: str
    logical_dtype: str
    shape: tuple[int, int]
    source_role: str = "value"


class ArtifactQuantizationPlan(msgspec.Struct, forbid_unknown_fields=True):
    """Bounded source structure, kept separate from model-sized source bytes."""

    components: list[str]
    configs: list[str]
    order: list[tuple[str, str]]
    tensors: list[QuantizationTensor]


class ArtifactQuantizationRequest(msgspec.Struct, forbid_unknown_fields=True):
    max_relative_frobenius: float | None = None


class ArtifactQuantizationResult(msgspec.Struct):
    artifact_transaction_id: str
    tensorfs_receipt_digest: str
    replayed: bool
    encoded_keys: int
    inherited_keys: int
    source_bytes_read_this_run: int
    new_bytes_written_this_run: int
    saturated_elements_this_run: int
    worst_relative_frobenius: float
    conformance: EncodingConformance


class QuantizationStats(msgspec.Struct):
    encoded_keys: int
    source_bytes_read: int
    new_bytes_written: int
    saturated_elements: int
    worst_relative_frobenius: float


class SourcePart(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dtype(self) -> str: ...

    @property
    def shape(self) -> tuple[int, ...]: ...


class SourceTensor(Protocol):
    @property
    def component(self) -> str: ...

    @property
    def key(self) -> str: ...

    @property
    def logical_dtype(self) -> str: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def parts(self) -> tuple[SourcePart, ...]: ...


class SourceStructure(Protocol):
    @property
    def configs(self) -> tuple[str, ...]: ...

    @property
    def tensors(self) -> tuple[SourceTensor, ...]: ...


class StructureWeights(Protocol):
    def structure(self, source: Model[object]) -> SourceStructure: ...


class Bf16Tensor(msgspec.Struct):
    component: str
    key: str
    source_dtype: str
    shape: tuple[int, ...]
    source_role: str


class Bf16Plan(msgspec.Struct):
    components: list[str]
    configs: list[str]
    order: list[tuple[str, str]]
    tensors: list[Bf16Tensor]
    #: f32 tensors a keep-listed component holds at source precision (#695).
    kept: list[tuple[str, str]]


class Bf16Stats(msgspec.Struct):
    """What one normalization cut recorded — the tier-1 round-trip stats of #695."""

    converted_keys: int
    source_bytes_read: int
    new_bytes_written: int
    worst_relative_frobenius: float


_ARTIFACT_SPECS = {
    # TensorFS 0.3's exact, vector-backed platform specs. Aliases are display;
    # WeightsTensor always carries the immutable spec identity.
    "fp8-rowwise/1": "sha256:c4be0120fb4548306b134f6ee07eb2545a363bc140a005af1ef6543c790cf890",
    "mxfp8/1": "sha256:7e9b1ad8f2e5ddd236a4d4303042d632a96eadef4f25d44eb0fb63124cec7cfd",
}
PLAIN_SPEC = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"
_MAX_ARTIFACT_PLAN_BYTES = 8 << 20
_SOURCE_READ_CHUNK = 32 << 20
_MAX_SOURCE_TENSOR_BYTES = 1 << 30
MAX_OUTPUT_BYTES = 32 << 30


def h3_quantization_plan() -> bytes:
    """Return the package-owned exact H3 pruned-DiT quantization plan."""
    return files(__package__).joinpath("h3-dit-quantization-plan.json").read_bytes()


def derive_mxfp8(
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    source: QuantizationSource,
    artifacts: WeightsSink,
    tel: Telemetry,
    *,
    plan_bytes: bytes | None = None,
) -> ArtifactQuantizationResult:
    """Derive MXFP8 directly from the exact source Manifest."""
    return _artifact_quantize(
        "mxfp8/1",
        _plan_or_source(plan_bytes, source, artifacts),
        ctx,
        payload,
        source,
        artifacts,
        tel,
    )


def derive_fp8(
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    source: QuantizationSource,
    artifacts: WeightsSink,
    tel: Telemetry,
    *,
    plan_bytes: bytes | None = None,
) -> ArtifactQuantizationResult:
    """Derive row-wise FP8 directly from the source, never from the MXFP8 sibling."""
    return _artifact_quantize(
        "fp8-rowwise/1",
        _plan_or_source(plan_bytes, source, artifacts),
        ctx,
        payload,
        source,
        artifacts,
        tel,
    )


def _plan_or_source(
    plan_bytes: bytes | None,
    source: QuantizationSource,
    weights: WeightsSink,
) -> ArtifactQuantizationPlan:
    if plan_bytes is not None:
        return prepare_quantization(plan_bytes)
    structure = cast(StructureWeights, weights).structure(source)
    return prepare_source_quantization(
        structure,
        components=tuple(dict.fromkeys(tensor.component for tensor in structure.tensors)),
    )


def prepare_quantization(plan_bytes: bytes) -> ArtifactQuantizationPlan:
    """Validate one closed structural plan for caller-owned transactions."""
    return _artifact_plan(plan_bytes)


def prepare_bf16(structure: SourceStructure, *, keep: tuple[str, ...] = ()) -> Bf16Plan:
    """Plan the fp32→bf16 ingest normalization from granted structure facts (#695).

    Only plain f32 tensors convert: fp16 carries the same bytes and would lose three
    mantissa bits for nothing, so fp16/bf16 and every already-encoded tensor inherit
    untouched. A component named in ``keep`` is precision-sensitive by the ingest
    config's own declaration — its f32 tensors inherit at source precision and are
    recorded in ``kept``.
    """
    keep_set = set(keep)
    if len(keep_set) != len(keep):
        raise UnsupportedInput("keep list repeats a component")
    available = {tensor.component for tensor in structure.tensors}
    missing = sorted(keep_set - available)
    if missing:
        raise UnsupportedInput(f"keep list names absent components {missing}")
    components: list[str] = []
    tensors: list[Bf16Tensor] = []
    kept: list[tuple[str, str]] = []
    order: list[tuple[str, str]] = []
    for tensor in structure.tensors:
        if tensor.component not in components:
            components.append(tensor.component)
        order.append((tensor.component, tensor.key))
        if tensor.logical_dtype != "f32":
            continue
        if tensor.component in keep_set:
            kept.append((tensor.component, tensor.key))
            continue
        role = _plain_source_role(tensor)
        tensors.append(
            Bf16Tensor(
                tensor.component,
                tensor.key,
                tensor.logical_dtype,
                tensor.shape,
                role,
            )
        )
    if not components or not order:
        raise UnsupportedInput("weights source structure contains no tensors")
    return Bf16Plan(components, list(structure.configs), order, tensors, kept)


def prepare_source_quantization(
    structure: SourceStructure,
    *,
    components: tuple[str, ...],
) -> ArtifactQuantizationPlan:
    """Select plain rank-2 weights that both row-wise encodings can represent."""
    available = {tensor.component for tensor in structure.tensors}
    missing = sorted(set(components) - available)
    if missing:
        raise UnsupportedInput(f"weights source has no components {missing}")
    selected: list[QuantizationTensor] = []
    for tensor in structure.tensors:
        if (
            tensor.component not in components
            or not tensor.key.endswith(".weight")
            or tensor.logical_dtype not in {"f16", "bf16", "f32"}
            or len(tensor.shape) != 2
            or tensor.shape[1] % 32
        ):
            continue
        selected.append(
            QuantizationTensor(
                component=tensor.component,
                key=tensor.key,
                logical_dtype=tensor.logical_dtype,
                shape=(tensor.shape[0], tensor.shape[1]),
                source_role=_plain_source_role(tensor),
            )
        )
    if not selected:
        raise UnsupportedInput("weights source has no block-aligned rank-2 float weights")
    return ArtifactQuantizationPlan(
        components=list(components),
        configs=list(structure.configs),
        order=[(tensor.component, tensor.key) for tensor in structure.tensors],
        tensors=selected,
    )


def bf16_targets(
    plan: Bf16Plan,
    *,
    source: str,
    excluded: set[tuple[str, str]] | None = None,
) -> dict[str, WeightsTarget]:
    """Declare BF16 replacements for the plan's f32 cut; everything else inherits."""
    omitted = excluded or set()
    targets: dict[str, WeightsTarget] = {}
    for component in plan.components:
        additions = {
            tensor.key: WeightsTensor(
                logical_dtype="bf16",
                shape=tensor.shape,
                encoding=PLAIN_SPEC,
                parts={"value": WeightsPart("bf16", tensor.shape)},
            )
            for tensor in plan.tensors
            if tensor.component == component and (tensor.component, tensor.key) not in omitted
        }
        targets[component] = WeightsTarget(
            source=source,
            source_component=component,
            drop=tuple(sorted(additions)),
            add=additions,
        )
    return targets


def write_bf16(
    transaction: WeightsTransaction,
    ctx: Context,
    tel: Telemetry,
    *,
    plan: Bf16Plan,
    source: str,
    excluded: set[tuple[str, str]] | None = None,
) -> Bf16Stats:
    """Stream the plan's f32 cut through one bounded RNE conversion, stats recorded.

    Every converted tensor's round trip is measured against the bf16 representational
    bound (``BF16_ROUND_TRIP_BOUND`` of its amax) and logged per tensor; the worst
    relative Frobenius rides the returned stats — the recorded derivation of #695,
    never a silent cast.
    """
    omitted = excluded or set()
    selected = [tensor for tensor in plan.tensors if (tensor.component, tensor.key) not in omitted]
    read = 0
    written = 0
    worst = 0.0
    for done, tensor in enumerate(selected, 1):
        ctx.raise_if_cancelled()
        values, length = _read_values(
            transaction,
            source=source,
            component=tensor.component,
            key=tensor.key,
            role=tensor.source_role,
            logical_dtype=tensor.source_dtype,
            shape=tensor.shape,
        )
        raw = st.from_f32(values, "BF16")
        error = np.abs(values - st.to_f32(raw, "BF16", list(tensor.shape)))
        bound = float(np.max(np.abs(values))) * BF16_ROUND_TRIP_BOUND
        if float(np.max(error)) > bound:
            raise UnsupportedInput(
                f"{tensor.component}.{tensor.key}: bf16 round-trip error "
                f"{float(np.max(error)):.4g} exceeds the representational bound {bound:.4g}",
                code="quantization_tripwire",
            )
        norm = float(np.linalg.norm(values))
        relative = float(np.linalg.norm(error)) / norm if norm else 0.0
        worst = max(worst, relative)
        transaction.add_part(tensor.component, tensor.key, "value", raw)
        read += length
        written += len(raw)
        tel.log(
            "normalized tensor",
            level="info",
            component=tensor.component,
            key=tensor.key,
            source_dtype=tensor.source_dtype,
            relative_frobenius=relative,
            done=done,
            total=len(selected),
        )
        tel.progress(done / len(selected), stage="bf16")
        del values, raw, error
    return Bf16Stats(
        converted_keys=len(selected),
        source_bytes_read=read,
        new_bytes_written=written,
        worst_relative_frobenius=worst,
    )


def inherited_configs(plan: Bf16Plan, *, source: str) -> dict[str, WeightsConfig]:
    return {name: WeightsConfig(source=source, source_config=name) for name in plan.configs}


def quantization_additions(
    encoding: str,
    plan: ArtifactQuantizationPlan,
    component: str,
    *,
    target_logical_dtype: str | None = None,
) -> dict[str, WeightsTensor]:
    """Declare the encoded tensor replacements for one plan component."""
    if encoding not in _ENCODINGS:
        raise UnsupportedInput(f"unsupported quantization encoding {encoding!r}")
    if component not in plan.components:
        raise UnsupportedInput(f"quantization plan has no component {component!r}")
    return {
        tensor.key: _artifact_tensor(
            encoding,
            tensor,
            target_logical_dtype=target_logical_dtype,
        )
        for tensor in plan.tensors
        if tensor.component == component
    }


def quantize_component_into(
    transaction: WeightsTransaction,
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    tel: Telemetry,
    *,
    encoding: str,
    plan: ArtifactQuantizationPlan,
    component: str,
    source: str,
    source_component: str,
    target_component: str,
) -> QuantizationStats:
    """Write one plan component into an already-open caller-owned transaction."""
    _validate_request(payload)
    encoder = _ENCODINGS.get(encoding)
    if encoder is None:
        raise UnsupportedInput(f"unsupported quantization encoding {encoding!r}")
    if component not in plan.components:
        raise UnsupportedInput(f"quantization plan has no component {component!r}")
    tensors = [tensor for tensor in plan.tensors if tensor.component == component]

    source_bytes = 0
    new_bytes = 0
    saturated = 0
    worst = 0.0
    total = len(tensors)
    for done, tensor in enumerate(tensors, 1):
        ctx.raise_if_cancelled()
        values, read = _source_values(
            transaction, tensor, source=source, source_component=source_component
        )
        encoded = encoder.encode(tensor.key, values)
        bound = float(np.max(np.abs(values))) * encoder.round_trip_bound
        if encoded.max_abs_err > bound:
            raise UnsupportedInput(
                f"{source_component}.{tensor.key}: round-trip error "
                f"{encoded.max_abs_err:.4g} exceeds the {encoding} bound {bound:.4g}",
                code="quantization_tripwire",
            )
        if len(encoded.companions) != 1:
            raise UnsupportedInput(f"{encoding} emitted an unexpected companion set")
        scale = encoded.companions[0]
        expected_scale = (
            ("U8", [tensor.shape[0], tensor.shape[1] // 32])
            if encoding == "mxfp8/1"
            else ("F32", [tensor.shape[0]])
        )
        if (
            encoded.payload_dtype != "F8_E4M3"
            or encoded.payload_shape != list(tensor.shape)
            or (scale.dtype, scale.shape) != expected_scale
        ):
            raise UnsupportedInput(f"{encoding} changed its declared TensorFS geometry")
        transaction.add_part(target_component, tensor.key, "data", encoded.payload)
        transaction.add_part(target_component, tensor.key, "scale", scale.raw)
        written = len(encoded.payload) + len(scale.raw)
        relative = encoded.relative_frobenius()
        worst = max(worst, relative)
        source_bytes += read
        new_bytes += written
        saturated += encoded.saturated
        tel.log(
            "quantized tensor",
            level="info",
            encoding=encoding,
            component=target_component,
            key=tensor.key,
            done=done,
            total=total,
        )
        tel.progress(done / total, stage=f"{encoding}-{target_component}")
        del values, encoded, scale

    if payload.max_relative_frobenius is not None and worst > payload.max_relative_frobenius:
        raise UnsupportedInput(
            f"worst per-tensor relative Frobenius {worst:.6f} exceeds the declared "
            f"tripwire {payload.max_relative_frobenius}",
            code="quantization_tripwire",
        )
    return QuantizationStats(
        encoded_keys=total,
        source_bytes_read=source_bytes,
        new_bytes_written=new_bytes,
        saturated_elements=saturated,
        worst_relative_frobenius=worst,
    )


def _artifact_plan(raw: bytes) -> ArtifactQuantizationPlan:
    try:
        length = len(raw)
        if not 0 < length <= _MAX_ARTIFACT_PLAN_BYTES:
            raise UnsupportedInput(
                f"quantization plan is {length} bytes; expected 1..{_MAX_ARTIFACT_PLAN_BYTES}"
            )
        plan = msgspec.json.decode(raw, type=ArtifactQuantizationPlan)
    except (OSError, msgspec.ValidationError) as exc:
        raise UnsupportedInput(f"quantization plan is not the closed schema: {exc}") from exc
    if not plan.components or len(plan.components) > 64:
        raise UnsupportedInput("quantization plan needs 1..64 target components")
    if len(set(plan.components)) != len(plan.components):
        raise UnsupportedInput("quantization plan repeats a target component")
    if len(set(plan.configs)) != len(plan.configs) or len(plan.configs) > 64:
        raise UnsupportedInput("quantization plan repeats or exceeds 64 copied configs")
    if not plan.order or len(plan.order) > 100_000 or len(set(plan.order)) != len(plan.order):
        raise UnsupportedInput("quantization plan order must contain 1..100000 unique rows")
    if any(component not in plan.components for component, _ in plan.order):
        raise UnsupportedInput("quantization plan order names an undeclared component")
    selected = {(tensor.component, tensor.key) for tensor in plan.tensors}
    if not selected or len(selected) != len(plan.tensors) or len(selected) > 65_536:
        raise UnsupportedInput("quantization plan tensors must contain 1..65536 unique keys")
    if not selected <= set(plan.order):
        raise UnsupportedInput("quantization plan selects a tensor outside its exact order")
    for tensor in plan.tensors:
        rows, cols = tensor.shape
        if (
            tensor.component not in plan.components
            or not tensor.key.endswith(".weight")
            or tensor.logical_dtype not in {"bf16", "f16", "f32"}
            or tensor.source_role != "value"
            or rows <= 0
            or cols <= 0
        ):
            raise UnsupportedInput(
                f"quantization plan tensor {tensor.component}.{tensor.key} is not one "
                "positive rank-2 float weight in the exact value role"
            )
    return plan


def _artifact_tensor(
    encoding: str,
    tensor: QuantizationTensor,
    *,
    target_logical_dtype: str | None = None,
) -> WeightsTensor:
    rows, cols = tensor.shape
    if encoding == "mxfp8/1":
        if cols % 32:
            raise UnsupportedInput(
                f"{tensor.component}.{tensor.key} has {cols} columns, not an MXFP8 block multiple"
            )
        parts = {
            "data": WeightsPart("f8_e4m3fn", (rows, cols)),
            "scale": WeightsPart("u8", (rows, cols // 32)),
        }
    else:
        parts = {
            "data": WeightsPart("f8_e4m3fn", (rows, cols)),
            "scale": WeightsPart("f32", (rows,)),
        }
    return WeightsTensor(
        logical_dtype=target_logical_dtype or tensor.logical_dtype,
        shape=tensor.shape,
        encoding=_ARTIFACT_SPECS[encoding],
        parts=parts,
    )


def _source_values(
    transaction: WeightsTransaction,
    tensor: QuantizationTensor,
    *,
    source: str,
    source_component: str,
) -> tuple[np.ndarray, int]:
    return _read_values(
        transaction,
        source=source,
        component=source_component,
        key=tensor.key,
        role=tensor.source_role,
        logical_dtype=tensor.logical_dtype,
        shape=tensor.shape,
    )


def _read_values(
    transaction: WeightsTransaction,
    *,
    source: str,
    component: str,
    key: str,
    role: str,
    logical_dtype: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    itemsize = {"bf16": 2, "f16": 2, "f32": 4}[logical_dtype]
    length = math.prod(shape) * itemsize
    if length > _MAX_SOURCE_TENSOR_BYTES:
        raise UnsupportedInput(
            f"source {component}.{key} is {length} bytes; the per-tensor "
            f"quantization bound is {_MAX_SOURCE_TENSOR_BYTES}"
        )
    raw = bytearray(length)
    view = memoryview(raw)
    for offset in range(0, length, _SOURCE_READ_CHUNK):
        transaction.source_read_into(
            source,
            component,
            key,
            role,
            offset,
            view[offset : min(offset + _SOURCE_READ_CHUNK, length)],
        )
    values = st.to_f32(raw, logical_dtype.upper(), list(shape))
    if not bool(np.isfinite(values).all()):
        raise UnsupportedInput(f"source {component}.{key} contains non-finite weights")
    return values, length


def _plain_source_role(tensor: SourceTensor) -> str:
    if (
        len(tensor.parts) != 1
        or tensor.parts[0].name != "value"
        or tensor.parts[0].dtype != tensor.logical_dtype
        or tensor.parts[0].shape != tensor.shape
    ):
        raise UnsupportedInput(
            f"source {tensor.component}.{tensor.key} is not one plain value role"
        )
    return "value"


def _validate_request(payload: ArtifactQuantizationRequest) -> None:
    if payload.max_relative_frobenius is not None and (
        not math.isfinite(payload.max_relative_frobenius) or payload.max_relative_frobenius < 0
    ):
        raise UnsupportedInput("max_relative_frobenius must be one finite nonnegative value")


def _artifact_quantize(
    encoding: str,
    plan: ArtifactQuantizationPlan,
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    source: QuantizationSource,
    artifacts: WeightsSink,
    tel: Telemetry,
) -> ArtifactQuantizationResult:
    _validate_request(payload)
    selected: dict[str, dict[str, QuantizationTensor]] = {
        component: {} for component in plan.components
    }
    for tensor in plan.tensors:
        selected[tensor.component][tensor.key] = tensor
    targets = {
        component: WeightsTarget(
            source="source",
            source_component=component,
            drop=tuple(sorted(selected[component])),
            add=quantization_additions(encoding, plan, component),
        )
        for component in plan.components
    }
    configs = {name: WeightsConfig(source="source", source_config=name) for name in plan.configs}
    with artifacts.open(
        "model",
        sources={"source": source},
        targets=targets,
        configs=configs,
        order=plan.order,
    ) as transaction:
        if transaction.replayed:
            receipt = transaction.receipt
            assert receipt is not None
            return ArtifactQuantizationResult(
                artifact_transaction_id=receipt.weights_transaction_id,
                tensorfs_receipt_digest=receipt.tensorfs_receipt_digest,
                replayed=True,
                encoded_keys=len(plan.tensors),
                inherited_keys=len(plan.order) - len(plan.tensors),
                source_bytes_read_this_run=0,
                new_bytes_written_this_run=0,
                saturated_elements_this_run=0,
                worst_relative_frobenius=0.0,
                conformance=EncodingConformance(
                    True,
                    ["TensorFS replayed the previously committed exact transaction"],
                ),
            )

        stats = [
            quantize_component_into(
                transaction,
                ctx,
                payload,
                tel,
                encoding=encoding,
                plan=plan,
                component=component,
                source="source",
                source_component=component,
                target_component=component,
            )
            for component in plan.components
        ]
        receipt = transaction.commit()
    source_bytes = sum(stat.source_bytes_read for stat in stats)
    new_bytes = sum(stat.new_bytes_written for stat in stats)
    saturated = sum(stat.saturated_elements for stat in stats)
    worst = max(stat.worst_relative_frobenius for stat in stats)
    tel.metric("quantization.source_bytes", float(source_bytes), unit="bytes")
    tel.metric("quantization.new_bytes", float(new_bytes), unit="bytes")
    return ArtifactQuantizationResult(
        artifact_transaction_id=receipt.weights_transaction_id,
        tensorfs_receipt_digest=receipt.tensorfs_receipt_digest,
        replayed=False,
        encoded_keys=len(plan.tensors),
        inherited_keys=len(plan.order) - len(plan.tensors),
        source_bytes_read_this_run=source_bytes,
        new_bytes_written_this_run=new_bytes,
        saturated_elements_this_run=saturated,
        worst_relative_frobenius=worst,
        conformance=EncodingConformance(
            True,
            [
                "each selected source role was read through the exact granted Manifest",
                "per-tensor round-trip stayed inside the encoding bound",
                "TensorFS committed the declared derived Manifest transaction",
            ],
        ),
    )
