"""Carrier-free FP8, MXFP8 and INT8 model derivation jobs.

The worker receives one exact TensorFS Manifest capability and returns another through
TensorFS scoped derivation handles. It receives no Store or filesystem access.
Publication belongs to Tensorhub's production transaction.
"""

from __future__ import annotations

import contextlib
import functools
import math
import tempfile
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import IO, Any, NamedTuple

import msgspec
import numpy as np
from tensorfs.derived import Config, DerivedTransaction, Part, SourceInspection, Target, Tensor

from cozy_runtime._cpus import cpus
from cozy_runtime.author import (
    Context,
    Model,
    Telemetry,
    UnsupportedInput,
    _numeric,
)

from . import safetensors_io as st
from .microscale import (
    CONVROT_GROUP,
    NVFP4_BLOCK,
    Encoded,
    encode_fp8_rowwise,
    encode_int8_convrot256,
    encode_mxfp8,
    encode_nvfp4,
)


class _Encoding(NamedTuple):
    """The encoder, its maximum reconstruction error as a multiple of tensor amax, the data
    role's carrier, the column multiple it needs, and its roles: the data role, then each
    companion's (role, dtype, shape) for a `[rows, cols]` tensor. A companion shaped `()` is one
    per tensor (nvfp4's second level), so the tensor's amax is read before any chunk is encoded
    and every chunk is encoded against it; every other companion leads with `rows`."""

    round_trip_bound: float
    encode: Callable[[str, np.ndarray, float], Encoded]
    payload: str
    data_dtype: str
    block: int
    companions: Callable[[int, int], tuple[tuple[str, str, tuple[int, ...]], ...]]
    data_role: str = "data"
    packed: int = 1

    def data_shape(self, rows: int, cols: int) -> tuple[int, int]:
        return rows, cols // self.packed

    def roles(self) -> tuple[str, ...]:
        return (self.data_role, *(role for role, _, _ in self.companions(1, self.block)))

    def tensor_scaled(self) -> bool:
        return any(not shape for _, _, shape in self.companions(1, self.block))


def _row_scale(rows: int, cols: int) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    return (("scale", "f32", (rows,)),)


def _rowwise(
    encode: Callable[[str, np.ndarray], Encoded],
) -> Callable[[str, np.ndarray, float], Encoded]:
    return lambda key, values, _amax: encode(key, values)


_ENCODINGS: dict[str, _Encoding] = {
    "mxfp8/1": _Encoding(
        round_trip_bound=2.0**-3,
        encode=_rowwise(encode_mxfp8),
        payload="F8_E4M3",
        data_dtype="f8_e4m3fn",
        block=32,
        companions=lambda rows, cols: (("scale", "u8", (rows, cols // 32)),),
    ),
    "fp8-rowwise/1": _Encoding(
        round_trip_bound=2.0**-3,
        encode=_rowwise(encode_fp8_rowwise),
        payload="F8_E4M3",
        data_dtype="f8_e4m3fn",
        block=32,
        companions=_row_scale,
    ),
    # Measured worst on Qwen-Image-2.1 and Qwen3-VL weights is far inside this; the bound is
    # a tripwire for a broken encoder, not a quality gate.
    "int8-convrot256/1": _Encoding(
        round_trip_bound=2.0**-4,
        encode=_rowwise(encode_int8_convrot256),
        payload="I8",
        data_dtype="i8",
        block=CONVROT_GROUP,
        companions=_row_scale,
    ),
    # A block's error is at most half its largest E2M1 step (2 of 6) times its scale, and a
    # clipped candidate gives up at most one E4M3 step on the block max: well inside amax / 4.
    "nvfp4-w4a4/1": _Encoding(
        round_trip_bound=2.0**-2,
        encode=encode_nvfp4,
        payload="U8",
        data_dtype="u8",
        block=NVFP4_BLOCK,
        companions=lambda rows, cols: (
            ("weight_scale", "f8_e4m3fn", (rows, cols // NVFP4_BLOCK)),
            ("weight_scale_2", "f32", ()),
        ),
        data_role="weight",
        packed=2,
    ),
}

#: RNE fp32->bf16 keeps every element within amax * 2^-8 (8-bit significand, half-ulp).
#: The normalization's own tier-1 bound — enforced on every cut, stats recorded (#695).
BF16_ROUND_TRIP_BOUND = 2.0**-8


class QuantizationSource(Model[object]):
    """Exact TensorFS Manifest input. The job reads it only through its scoped TensorFS handle."""

    def load(self, loader: Any) -> None:
        del loader


class QuantizationTensor(msgspec.Struct, forbid_unknown_fields=True):
    """One source tensor selected by the reviewed structural plan."""

    component: str
    key: str
    logical_dtype: str
    shape: tuple[int, int]
    source_role: str = "value"
    #: This weight's own encoding, when it is not the lane's.
    encoding: str | None = None


class ArtifactQuantizationPlan(msgspec.Struct, forbid_unknown_fields=True):
    """Bounded source structure, kept separate from model-sized source bytes."""

    components: list[str]
    configs: list[str]
    order: list[tuple[str, str]]
    tensors: list[QuantizationTensor]


class ArtifactQuantizationRequest(msgspec.Struct, forbid_unknown_fields=True):
    max_relative_frobenius: float | None = None


class QuantizationStats(msgspec.Struct):
    encoded_keys: int
    reused_keys: int
    source_bytes_read: int
    new_bytes_written: int
    saturated_elements: int
    worst_relative_frobenius: float | None


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
    worst_relative_frobenius: float | None


_ARTIFACT_SPECS = {
    # TensorFS 0.3's exact, vector-backed platform specs. Aliases are display;
    # Tensor always carries the immutable spec identity.
    "fp8-rowwise/1": "sha256:c4be0120fb4548306b134f6ee07eb2545a363bc140a005af1ef6543c790cf890",
    "mxfp8/1": "sha256:7e9b1ad8f2e5ddd236a4d4303042d632a96eadef4f25d44eb0fb63124cec7cfd",
    "int8-convrot256/1": "sha256:ac25d323baebfdf1fd0ce3fdfd9f26400ce57914fdc442d0ade8cb187aef183f",
    "nvfp4-w4a4/1": "sha256:f2e718c521634876830f9c2647bddfbbccafc744decc88adb345d33d73bf5b1a",
}
PLAIN_SPEC = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"
_MAX_ARTIFACT_PLAN_BYTES = 8 << 20
_SOURCE_READ_CHUNK = 8 << 20
MAX_OUTPUT_BYTES = 32 << 30


def prepare_quantization(plan_bytes: bytes) -> ArtifactQuantizationPlan:
    """Validate one closed structural plan for caller-owned transactions."""
    return _artifact_plan(plan_bytes)


def source_rows(structure: SourceInspection) -> Iterator[tuple[str, str, Tensor]]:
    """Traverse the native header's construction order."""
    for component, tensors in structure.components.items():
        for key, tensor in tensors.items():
            yield component, key, tensor


def source_parts(tensor: Tensor) -> Mapping[str, Part]:
    return tensor.parts


def completed_parts(transaction: DerivedTransaction) -> set[tuple[str, str, str]]:
    return set(transaction.completed_parts())


def bf16_source_dtypes(sources: Sequence[str]) -> tuple[str, ...]:
    """Validate the explicit floating-point source cut; ordering has no meaning."""
    if not sources or len(set(sources)) != len(sources) or set(sources) - {"f16", "f32"}:
        raise UnsupportedInput(
            "bf16_sources must contain unique f16 or f32 source dtypes",
            code="quantization_plan",
        )
    return tuple(sorted(sources))


def prepare_bf16(
    structure: SourceInspection,
    *,
    keep: tuple[str, ...] = (),
    source_dtypes: tuple[str, ...] = ("f32",),
) -> Bf16Plan:
    """Normalize selected plain floats; other source dtypes inherit unchanged."""
    source_dtypes = bf16_source_dtypes(source_dtypes)
    keep_set = set(keep)
    if len(keep_set) != len(keep):
        raise UnsupportedInput("keep list repeats a component")
    rows = list(source_rows(structure))
    components = list(dict.fromkeys(component for component, _, _ in rows))
    missing = sorted(keep_set - set(components))
    if missing:
        raise UnsupportedInput(f"keep list names absent components {missing}")
    tensors: list[Bf16Tensor] = []
    kept: list[tuple[str, str]] = []
    order = [(component, key) for component, key, _ in rows]
    for component, key, tensor in rows:
        if tensor.logical_dtype not in source_dtypes:
            continue
        if component in keep_set:
            kept.append((component, key))
            continue
        tensors.append(
            Bf16Tensor(
                component,
                key,
                tensor.logical_dtype,
                tuple(tensor.shape),
                _plain_source_role(tensor),
            )
        )
    if not components or not order:
        raise UnsupportedInput("weights source structure contains no tensors")
    return Bf16Plan(components, list(structure.configs), order, tensors, kept)


def column_block(encoding: str | None) -> int:
    """The column multiple an encoding's tensors need; 32 serves both fp8 encodings."""
    found = _ENCODINGS.get(encoding or "")
    return found.block if found is not None else 32


def prepare_source_quantization(
    structure: SourceInspection, *, components: tuple[str, ...], block: int = 32
) -> ArtifactQuantizationPlan:
    """Select plain rank-2 weights whose columns are a multiple of the encoding's `block`."""
    rows = list(source_rows(structure))
    missing = sorted(set(components) - {component for component, _, _ in rows})
    if missing:
        raise UnsupportedInput(f"weights source has no components {missing}")
    selected: list[QuantizationTensor] = []
    for component, key, tensor in rows:
        if (
            component not in components
            or not key.endswith(".weight")
            or tensor.logical_dtype not in {"f16", "bf16", "f32"}
            or len(tensor.shape) != 2
            or tensor.shape[1] % block
        ):
            continue
        selected.append(
            QuantizationTensor(
                component=component,
                key=key,
                logical_dtype=tensor.logical_dtype,
                shape=(tensor.shape[0], tensor.shape[1]),
                source_role=_plain_source_role(tensor),
            )
        )
    if not selected:
        raise UnsupportedInput("weights source has no block-aligned rank-2 float weights")
    return ArtifactQuantizationPlan(
        list(components),
        list(structure.configs),
        [(component, key) for component, key, _ in rows],
        selected,
    )


def bf16_targets(
    plan: Bf16Plan,
    *,
    source: str,
    excluded: set[tuple[str, str]] | None = None,
) -> dict[str, Target]:
    """Declare BF16 replacements for the selected float cut; everything else inherits."""
    omitted = excluded or set()
    targets: dict[str, Target] = {}
    for component in plan.components:
        additions = {
            tensor.key: Tensor(
                logical_dtype="bf16",
                shape=tensor.shape,
                encoding=PLAIN_SPEC,
                parts={"value": Part("bf16", tensor.shape)},
            )
            for tensor in plan.tensors
            if tensor.component == component and (tensor.component, tensor.key) not in omitted
        }
        targets[component] = Target(
            source=source,
            source_component=component,
            drop=tuple(sorted(additions)),
            add=additions,
        )
    return targets


def write_bf16(
    transaction: DerivedTransaction,
    ctx: Context,
    tel: Telemetry,
    *,
    plan: Bf16Plan,
    source: str,
    excluded: set[tuple[str, str]] | None = None,
) -> Bf16Stats:
    """Stream the selected float cut through one bounded RNE conversion, stats recorded.

    Every converted tensor's round trip is measured against the bf16 representational
    bound (``BF16_ROUND_TRIP_BOUND`` of its amax) and logged per tensor; the worst
    relative Frobenius rides the returned stats — the recorded derivation of #695,
    never a silent cast.
    """
    omitted = excluded or set()
    selected = [tensor for tensor in plan.tensors if (tensor.component, tensor.key) not in omitted]
    read = 0
    written = 0
    worst: float | None = 0.0
    completed = completed_parts(transaction)
    for done, tensor in enumerate(selected, 1):
        ctx.raise_if_cancelled()
        if (tensor.component, tensor.key, "value") in completed:
            worst = None
            tel.progress(done / len(selected), stage="bf16")
            continue
        amax = max_error = sse = energy = 0.0
        length = size = 0
        with tempfile.TemporaryFile() as spill:
            for values in _read_chunks(
                transaction,
                source=source,
                component=tensor.component,
                key=tensor.key,
                role=tensor.source_role,
                logical_dtype=tensor.source_dtype,
                shape=tensor.shape,
            ):
                raw = st.from_f32(values, "BF16")
                error = np.abs(values - st.to_f32(raw, "BF16", list(values.shape)))
                if values.size:
                    amax = _numeric.worst(amax, float(np.max(np.abs(values))))
                    max_error = _numeric.worst(max_error, float(np.max(error)))
                sse += float(np.sum(error.astype(np.float64) ** 2))
                energy += float(np.sum(values.astype(np.float64) ** 2))
                length += values.size * {"bf16": 2, "f16": 2, "f32": 4}[tensor.source_dtype]
                size += spill.write(raw)
            bound = amax * BF16_ROUND_TRIP_BOUND
            if not _numeric.within(max_error, bound):
                raise UnsupportedInput(
                    f"{tensor.component}.{tensor.key}: bf16 round-trip error "
                    f"{max_error:.4g} exceeds the representational bound {bound:.4g}",
                    code="quantization_tripwire",
                )
            relative = math.sqrt(sse / energy) if energy else 0.0
            worst = _numeric.worst(worst, relative) if worst is not None else None
            spill.seek(0)
            transaction.add_part(tensor.component, tensor.key, "value", spill)
        transaction.checkpoint()
        read += length
        written += size
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
    return Bf16Stats(
        converted_keys=len(selected),
        source_bytes_read=read,
        new_bytes_written=written,
        worst_relative_frobenius=worst,
    )


def inherited_configs(plan: Bf16Plan, *, source: str) -> dict[str, Config]:
    return {name: Config("copy", source=source, source_config=name) for name in plan.configs}


def quantization_additions(
    encoding: str,
    plan: ArtifactQuantizationPlan,
    component: str,
    *,
    target_logical_dtype: str | None = None,
) -> dict[str, Tensor]:
    """Declare the encoded tensor replacements for one plan component."""
    if encoding not in _ENCODINGS:
        raise UnsupportedInput(f"unsupported quantization encoding {encoding!r}")
    if component not in plan.components:
        raise UnsupportedInput(f"quantization plan has no component {component!r}")
    return {
        tensor.key: _artifact_tensor(
            tensor.encoding or encoding,
            tensor,
            target_logical_dtype=target_logical_dtype,
        )
        for tensor in plan.tensors
        if tensor.component == component
    }


def quantize_component_into(
    transaction: DerivedTransaction,
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
    """Write one plan component into an already-open caller-owned transaction.

    Tensors encode ahead on the shared pool; their roles are added and checkpointed here,
    one tensor at a time and in plan order, exactly as a serial encoder would.
    """
    _validate_request(payload)
    if encoding not in _ENCODINGS or any(
        tensor.encoding not in (None, *_ENCODINGS) for tensor in plan.tensors
    ):
        raise UnsupportedInput(f"unsupported quantization encoding {encoding!r}")
    if component not in plan.components:
        raise UnsupportedInput(f"quantization plan has no component {component!r}")
    tensors = [tensor for tensor in plan.tensors if tensor.component == component]

    source_bytes = 0
    new_bytes = 0
    saturated = 0
    worst = 0.0
    total = len(tensors)
    completed = completed_parts(transaction)
    reused = 0

    def encoding_of(tensor: QuantizationTensor) -> str:
        return tensor.encoding or encoding

    def finished(tensor: QuantizationTensor) -> bool:
        roles = _ENCODINGS[encoding_of(tensor)].roles()
        return all((target_component, tensor.key, role) in completed for role in roles)

    def encode(tensor: QuantizationTensor, stop: threading.Event) -> _Encoded:
        return _encode_tensor(
            transaction,
            encoding_of(tensor),
            tensor,
            stop,
            source=source,
            source_component=source_component,
        )

    results = _in_order([tensor for tensor in tensors if not finished(tensor)], encode)
    with contextlib.closing(results):
        for done, tensor in enumerate(tensors, 1):
            ctx.raise_if_cancelled()
            if finished(tensor):
                reused += 1
                tel.progress(done / total, stage=f"{encoding}-{target_component}")
                continue
            result = next(results)
            name = encoding_of(tensor)
            roles = _ENCODINGS[name].roles()
            with result.spill as spill:
                bound = result.amax * _ENCODINGS[name].round_trip_bound
                if not _numeric.within(result.max_error, bound):
                    raise UnsupportedInput(
                        f"{source_component}.{tensor.key}: round-trip error "
                        f"{result.max_error:.4g} exceeds the {name} bound {bound:.4g}",
                        code="quantization_tripwire",
                    )
                if result.rows != tensor.shape[0]:
                    raise UnsupportedInput(f"{name} changed its declared TensorFS geometry")
                relative = math.sqrt(result.sse / result.energy) if result.energy > 0 else 0.0
                if payload.max_relative_frobenius is not None and not _numeric.within(
                    relative, payload.max_relative_frobenius
                ):
                    raise UnsupportedInput(
                        f"{source_component}.{tensor.key}: relative Frobenius {relative:.6f} "
                        f"exceeds the declared tripwire {payload.max_relative_frobenius}",
                        code="quantization_tripwire",
                    )
                # A role becomes resumable only after every numerical acceptance check. A
                # half-written group is recomputed; TensorFS accepts only identical role replay.
                written = 0
                data_bytes = spill.tell()
                spill.seek(0)
                for role, part, part_bytes in (
                    (roles[0], spill, data_bytes),
                    *(
                        (role, raw, len(raw))
                        for role, raw in zip(roles[1:], result.scales, strict=True)
                    ),
                ):
                    transaction.add_part(target_component, tensor.key, role, part)
                    if (target_component, tensor.key, role) not in completed:
                        written += part_bytes
            transaction.checkpoint()
            worst = _numeric.worst(worst, relative)
            source_bytes += result.read
            new_bytes += written
            saturated += result.saturated
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

    return QuantizationStats(
        encoded_keys=total - reused,
        reused_keys=reused,
        source_bytes_read=source_bytes,
        new_bytes_written=new_bytes,
        saturated_elements=saturated,
        worst_relative_frobenius=None if reused else worst,
    )


class _Encoded(NamedTuple):
    """One tensor's payload (spilled to disk), its scales and its acceptance facts."""

    spill: IO[bytes]
    scales: tuple[bytes, ...]
    amax: float
    max_error: float
    sse: float
    energy: float
    saturated: int
    read: int
    rows: int


def _encode_tensor(
    transaction: DerivedTransaction,
    encoding: str,
    tensor: QuantizationTensor,
    stop: threading.Event,
    *,
    source: str,
    source_component: str,
) -> _Encoded:
    encoder = _ENCODINGS[encoding]
    expected = encoder.companions(*tensor.shape)
    read_chunks = functools.partial(
        _read_chunks,
        transaction,
        source=source,
        component=source_component,
        key=tensor.key,
        role=tensor.source_role,
        logical_dtype=tensor.logical_dtype,
        shape=tensor.shape,
    )
    amax = max_error = sse = energy = 0.0
    if encoder.tensor_scaled():
        for values in read_chunks():
            if stop.is_set():
                raise _Stopped
            if values.size:
                amax = _numeric.worst(amax, float(np.max(np.abs(values))))
    read = rows = saturated = 0
    scales: list[list[bytes]] = [[] for _ in expected]
    spill = tempfile.TemporaryFile()
    try:
        for values in read_chunks():
            if stop.is_set():
                raise _Stopped
            encoded = encoder.encode(tensor.key, values, amax)
            count = values.shape[0]
            if (
                encoded.payload_dtype != encoder.payload
                or encoded.payload_shape != list(encoder.data_shape(count, tensor.shape[1]))
                or len(encoded.companions) != len(expected)
            ):
                raise UnsupportedInput(f"{encoding} changed its declared TensorFS geometry")
            for got, (_, dtype, shape), chunks in zip(
                encoded.companions, expected, scales, strict=True
            ):
                row_shaped = bool(shape)
                want = [count, *shape[1:]] if row_shaped else []
                if (got.dtype, got.shape) != (_SAFETENSORS_DTYPES[dtype], want):
                    raise UnsupportedInput(f"{encoding} changed its declared TensorFS geometry")
                if row_shaped or not chunks:
                    chunks.append(got.raw)
                elif got.raw != chunks[0]:
                    raise UnsupportedInput(f"{encoding} encoded one tensor scale two ways")
            if values.size:
                amax = _numeric.worst(amax, float(np.max(np.abs(values))))
            max_error = _numeric.worst(max_error, encoded.max_abs_err)
            sse += encoded.sse
            energy += encoded.energy
            saturated += encoded.saturated
            read += values.size * {"bf16": 2, "f16": 2, "f32": 4}[tensor.logical_dtype]
            rows += count
            spill.write(encoded.payload)
            del values, encoded
        return _Encoded(
            spill,
            tuple(b"".join(chunks) for chunks in scales),
            amax,
            max_error,
            sse,
            energy,
            saturated,
            read,
            rows,
        )
    except BaseException:
        spill.close()
        raise


#: TensorFS part dtype -> the safetensors dtype an encoder's companion declares.
_SAFETENSORS_DTYPES = {"f32": "F32", "u8": "U8", "f8_e4m3fn": "F8_E4M3"}


class _Stopped(Exception):
    """The consumer left; encoding ahead of it is abandoned."""


def _in_order(
    items: Sequence[QuantizationTensor],
    produce: Callable[[QuantizationTensor, threading.Event], _Encoded],
) -> Generator[_Encoded, None, None]:
    """`produce` over `items` on the shared pool, a bounded window ahead, yielded in order.

    Closing the iterator stops the producers and releases every result nobody consumed.
    """
    stop = threading.Event()
    window: deque[Future[_Encoded]] = deque()
    pending = iter(items)
    try:
        for _ in items:
            while len(window) <= cpus() and (item := next(pending, None)) is not None:
                window.append(_pool().submit(produce, item, stop))
            yield window.popleft().result()
    finally:
        stop.set()
        for future in window:
            if not future.cancel():
                with contextlib.suppress(Exception):
                    future.result().spill.close()


@functools.cache
def _pool() -> ThreadPoolExecutor:
    """One process-wide encoder pool: callers that shard their own work share its bound."""
    return ThreadPoolExecutor(cpus(), thread_name_prefix="cozy-quantize")


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
            or tensor.encoding not in (None, *_ENCODINGS)
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
) -> Tensor:
    rows, cols = tensor.shape
    encoder = _ENCODINGS[encoding]
    if cols % encoder.block:
        raise UnsupportedInput(
            f"{tensor.component}.{tensor.key} has {cols} columns, not a multiple of "
            f"{encoding}'s {encoder.block}"
        )
    parts = {encoder.data_role: Part(encoder.data_dtype, encoder.data_shape(rows, cols))}
    parts.update(
        {role: Part(dtype, shape) for role, dtype, shape in encoder.companions(rows, cols)}
    )
    return Tensor(
        logical_dtype=target_logical_dtype or tensor.logical_dtype,
        shape=tensor.shape,
        encoding=_ARTIFACT_SPECS[encoding],
        parts=parts,
    )


def _read_chunks(
    transaction: DerivedTransaction,
    *,
    source: str,
    component: str,
    key: str,
    role: str,
    logical_dtype: str,
    shape: tuple[int, ...],
) -> Iterator[np.ndarray]:
    """Whole rows of float32 values, about `_SOURCE_READ_CHUNK` source bytes at a time.

    Host memory stays one chunk whatever the tensor's size; one row is the smallest unit.
    """
    itemsize = {"bf16": 2, "f16": 2, "f32": 4}[logical_dtype]
    rows, tail = (shape[0], shape[1:]) if shape else (1, ())
    row_bytes = math.prod(tail) * itemsize
    step = max(1, _SOURCE_READ_CHUNK // max(row_bytes, 1))
    for first in range(0, rows, step):
        count = min(step, rows - first)
        raw = bytearray(count * row_bytes)
        view = memoryview(raw)
        base = first * row_bytes
        for offset in range(0, len(raw), _SOURCE_READ_CHUNK):
            transaction.source_read_into(
                source,
                component,
                key,
                role,
                base + offset,
                view[offset : min(offset + _SOURCE_READ_CHUNK, len(raw))],
            )
        chunk_shape = [count, *tail] if shape else []
        values = st.to_f32(raw, logical_dtype.upper(), chunk_shape)
        if not bool(np.isfinite(values).all()):
            raise UnsupportedInput(f"source {component}.{key} contains non-finite weights")
        yield values


def _plain_source_role(tensor: Any) -> str:
    parts = source_parts(tensor)
    value = parts.get("value")
    if (
        len(parts) != 1
        or value is None
        or value.dtype != tensor.logical_dtype
        or tuple(value.shape) != tuple(tensor.shape)
    ):
        raise UnsupportedInput("source tensor is not one plain value role")
    return "value"


def _validate_request(payload: ArtifactQuantizationRequest) -> None:
    if payload.max_relative_frobenius is not None and (
        not math.isfinite(payload.max_relative_frobenius) or payload.max_relative_frobenius < 0
    ):
        raise UnsupportedInput("max_relative_frobenius must be one finite nonnegative value")
