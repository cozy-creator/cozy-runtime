"""Lane derivations: normalize floats to bf16 and encode selected weights, in the store.

A derivation reads one source checkpoint and writes one output (`weights_store.Output`) whose
base is that source: every component and tensor it does not write again carries over, in the
source's construction order. Writes are resumable: a tensor an interrupted attempt wrote
(`has`) is not computed again.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import IO, NamedTuple

import msgspec
import numpy as np

from cozy_runtime._cpus import cpus
from cozy_runtime.author import Context, Telemetry, UnsupportedInput, _numeric
from cozy_runtime.author._loader import TENSORFS_DTYPES, TENSORFS_NAMES
from cozy_runtime.weights_store import Checkpoint, Output, PartData, read_into

from . import safetensors_io as st
from . import svdq
from .microscale import (
    CONVROT_GROUP,
    NVFP4_BLOCK,
    W4A8_GROUP,
    Encoded,
    encode_fp8_rowwise,
    encode_int8_convrot256,
    encode_nvfp4,
    encode_w4a8,
)


class _Encoding(NamedTuple):
    """A scheme's encoder: its round-trip bound (a multiple of tensor amax), the column multiple
    it needs, its params, and its parts after `weight`, each `(role, dtype, shape)` for an
    `[rows, cols]` tensor. A part shaped `()` is one per tensor (nvfp4's second level), so the
    tensor's amax is read before any chunk is encoded and every chunk encodes against it. A
    `whole` part is also one per tensor but fixed (w4a8's codebook): every chunk writes it the
    same. `pack` weights share a stored element (w4a8's codes are `[rows, cols / 2]` bytes)."""

    round_trip_bound: float
    encode: Callable[[str, np.ndarray, float], Encoded]
    weight_dtype: str
    block: int
    params: dict[str, object]
    companions: Callable[[int, int], tuple[tuple[str, str, tuple[int, ...]], ...]]
    whole: frozenset[str] = frozenset()
    pack: int = 1
    #: Encoded whole from a `svdq.Calibration` of the model's own activations (`svdq.fit`).
    calibrated: bool = False

    def tensor_scaled(self) -> bool:
        return any(not shape for _, _, shape in self.companions(1, self.block))


def _row_scale(rows: int, cols: int) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    return (("scale", "F32", (rows,)),)


def _rowwise(
    encode: Callable[[str, np.ndarray], Encoded],
) -> Callable[[str, np.ndarray, float], Encoded]:
    return lambda key, values, _amax: encode(key, values)


_ENCODINGS: dict[str, _Encoding] = {
    "fp8.rowwise": _Encoding(
        2.0**-3,
        _rowwise(encode_fp8_rowwise),
        "F8_E4M3",
        32,
        {},
        _row_scale,
    ),
    # The bound is a tripwire for a broken encoder, not a quality gate.
    "int8.convrot": _Encoding(
        2.0**-4,
        _rowwise(encode_int8_convrot256),
        "I8",
        CONVROT_GROUP,
        {"rotation": "kron_regular_h4", "rotation_size": CONVROT_GROUP},
        _row_scale,
    ),
    # A block's error is at most half its largest E2M1 step (2 of 6) times its scale, and a
    # clipped candidate gives up at most one E4M3 step on the block max: well inside amax / 4.
    "nvfp4": _Encoding(
        2.0**-2,
        encode_nvfp4,
        "F4",
        NVFP4_BLOCK,
        {
            "block_size": NVFP4_BLOCK,
            "global_scale": "multiply",
            "nibble_order": "low_first",
            "scale_layout": "row_major",
        },
        lambda rows, cols: (
            ("scale", "F8_E4M3", (rows, cols // NVFP4_BLOCK)),
            ("scale_2", "F32", ()),
        ),
    ),
    # Four bits after the rotation: a tripwire well above a 16-level grid's worst element.
    "w4a8.convrot": _Encoding(
        2.0**-1,
        _rowwise(encode_w4a8),
        "I8",
        CONVROT_GROUP,
        {"group_size": W4A8_GROUP, "rotation": "kron_regular_h4", "rotation_size": CONVROT_GROUP},
        lambda rows, cols: (
            ("group_scale", "F8_E4M3", (rows, cols // W4A8_GROUP)),
            ("scale", "F32", (rows,)),
            ("codebook", "F32", (16,)),
        ),
        whole=frozenset({"codebook"}),
        pack=2,
    ),
    # SVDQuant (xs-066): the bound holds the nvfp4 residual's round trip, in the smoothed domain.
    "svdq.nvfp4": _Encoding(
        2.0**-2,
        lambda key, values, amax: _uncalibrated(key),
        "F4",
        svdq.K_ALIGN,
        {
            "block_size": NVFP4_BLOCK,
            "global_scale": "multiply",
            "nibble_order": "low_first",
            "scale_layout": "row_major",
        },
        svdq.companions,
        calibrated=True,
    ),
}


def _uncalibrated(key: str) -> Encoded:
    raise UnsupportedInput(
        f"{key}: svdq.nvfp4 is fitted to the model's own activations; pass quantize_into a "
        "calibration (derive.svdq.Calibration)",
        code="quantization_plan",
    )


#: RNE fp32->bf16 keeps every element within amax * 2^-8 (8-bit significand, half-ulp).
BF16_ROUND_TRIP_BOUND = 2.0**-8
_SOURCE_READ_CHUNK = 8 << 20
_WIDTH = {"bf16": 2, "f16": 2, "f32": 4}


class SourceTensor(NamedTuple):
    """A source tensor as a derivation plans on it: its Runtime dtype, shape, and whether it is
    stored plain."""

    logical_dtype: str
    shape: tuple[int, ...]
    plain: bool


def source_rows(source: Checkpoint) -> Iterator[tuple[str, str, SourceTensor]]:
    """The source's tensors in construction order."""
    for component, held in source.components.items():
        for key, info in held.tensors.items():
            dtype = TENSORFS_DTYPES.get(info.dtype, info.dtype)
            yield component, key, SourceTensor(dtype, tuple(info.shape), info.encoding is None)


class QuantizationTensor(msgspec.Struct, forbid_unknown_fields=True):
    """One source tensor selected by the reviewed structural plan."""

    component: str
    key: str
    logical_dtype: str
    shape: tuple[int, int]
    #: This weight's own scheme, when it is not the lane's.
    encoding: str | None = None


class ArtifactQuantizationPlan(msgspec.Struct, forbid_unknown_fields=True):
    """The selected tensors, kept separate from model-sized source bytes."""

    components: list[str]
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


class Bf16Plan(msgspec.Struct):
    components: list[str]
    order: list[tuple[str, str]]
    tensors: list[Bf16Tensor]
    #: float tensors a keep-listed component holds at source precision (#695)
    kept: list[tuple[str, str]]


class Bf16Stats(msgspec.Struct):
    """What one normalization cut recorded: the round-trip stats of #695."""

    converted_keys: int
    reused_keys: int
    source_bytes_read: int
    new_bytes_written: int
    worst_relative_frobenius: float | None


def bf16_source_dtypes(sources: Sequence[str]) -> tuple[str, ...]:
    """Validate the explicit floating-point source cut; ordering has no meaning."""
    if not sources or len(set(sources)) != len(sources) or set(sources) - {"f16", "f32"}:
        raise UnsupportedInput(
            "bf16_sources must contain unique f16 or f32 source dtypes", code="quantization_plan"
        )
    return tuple(sorted(sources))


def prepare_bf16(
    source: Checkpoint, *, keep: tuple[str, ...] = (), source_dtypes: tuple[str, ...] = ("f32",)
) -> Bf16Plan:
    """Normalize selected plain floats; other tensors carry over unchanged."""
    source_dtypes = bf16_source_dtypes(source_dtypes)
    keep_set = set(keep)
    if len(keep_set) != len(keep):
        raise UnsupportedInput("keep list repeats a component")
    rows = list(source_rows(source))
    components = list(dict.fromkeys(component for component, _, _ in rows))
    missing = sorted(keep_set - set(components))
    if missing:
        raise UnsupportedInput(f"keep list names absent components {missing}")
    tensors: list[Bf16Tensor] = []
    kept: list[tuple[str, str]] = []
    for component, key, tensor in rows:
        if not tensor.plain or tensor.logical_dtype not in source_dtypes:
            continue
        if component in keep_set:
            kept.append((component, key))
            continue
        tensors.append(Bf16Tensor(component, key, tensor.logical_dtype, tensor.shape))
    if not rows:
        raise UnsupportedInput("weights source structure contains no tensors")
    return Bf16Plan(components, [(c, k) for c, k, _ in rows], tensors, kept)


def column_block(encoding: str | None) -> int:
    """The column multiple a scheme's tensors need; 32 is fp8.rowwise's."""
    found = _ENCODINGS.get(encoding or "")
    return found.block if found is not None else 32


def prepare_source_quantization(
    source: Checkpoint, *, components: tuple[str, ...], block: int = 32
) -> ArtifactQuantizationPlan:
    """Select plain rank-2 float weights whose columns are a multiple of the scheme's `block`."""
    rows = list(source_rows(source))
    missing = sorted(set(components) - {component for component, _, _ in rows})
    if missing:
        raise UnsupportedInput(f"weights source has no components {missing}")
    selected = [
        QuantizationTensor(component, key, tensor.logical_dtype, (tensor.shape[0], tensor.shape[1]))
        for component, key, tensor in rows
        if component in components
        and key.endswith(".weight")
        and tensor.plain
        and tensor.logical_dtype in _WIDTH
        and len(tensor.shape) == 2
        and tensor.shape[1] % block == 0
    ]
    if not selected:
        raise UnsupportedInput("weights source has no block-aligned rank-2 float weights")
    return ArtifactQuantizationPlan(list(components), [(c, k) for c, k, _ in rows], selected)


def write_bf16(
    source: Checkpoint,
    output: Output,
    ctx: Context,
    tel: Telemetry,
    *,
    plan: Bf16Plan,
    excluded: set[tuple[str, str]] | None = None,
) -> Bf16Stats:
    """Stream the selected float cut through one bounded RNE conversion, stats recorded: each
    tensor's round trip is held to `BF16_ROUND_TRIP_BOUND` of its amax and logged."""
    omitted = excluded or set()
    selected = [t for t in plan.tensors if (t.component, t.key) not in omitted]
    read = written = reused = 0
    worst: float | None = 0.0
    for done, tensor in enumerate(selected, 1):
        ctx.raise_if_cancelled()
        if output.has(tensor.component, tensor.key):
            reused += 1
            worst = None
            tel.progress(done / len(selected), stage="bf16")
            continue
        amax = max_error = sse = energy = 0.0
        with tempfile.TemporaryFile() as spill:
            for values in _read_chunks(
                source, tensor.component, tensor.key, tensor.source_dtype, tensor.shape
            ):
                raw = st.from_f32(values, "BF16")
                error = np.abs(values - st.to_f32(raw, "BF16", list(values.shape)))
                if values.size:
                    amax = _numeric.worst(amax, float(np.max(np.abs(values))))
                    max_error = _numeric.worst(max_error, float(np.max(error)))
                sse += float(np.sum(error.astype(np.float64) ** 2))
                energy += float(np.sum(values.astype(np.float64) ** 2))
                read += values.size * _WIDTH[tensor.source_dtype]
                spill.write(raw)
            bound = amax * BF16_ROUND_TRIP_BOUND
            if not _numeric.within(max_error, bound):
                raise UnsupportedInput(
                    f"{tensor.component}.{tensor.key}: bf16 round-trip error {max_error:.4g} "
                    f"exceeds the representational bound {bound:.4g}",
                    code="quantization_tripwire",
                )
            relative = math.sqrt(sse / energy) if energy else 0.0
            worst = _numeric.worst(worst, relative) if worst is not None else None
            written += spill.tell()
            spill.seek(0)
            output.put(tensor.component, tensor.key, "BF16", tensor.shape, spill)
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
    return Bf16Stats(len(selected), reused, read, written, worst)


def quantize_component_into(
    source: Checkpoint,
    output: Output,
    ctx: Context,
    payload: ArtifactQuantizationRequest,
    tel: Telemetry,
    *,
    encoding: str,
    plan: ArtifactQuantizationPlan,
    component: str,
    logical_dtype: Callable[[QuantizationTensor], str],
    calibration: svdq.Calibration | None = None,
) -> QuantizationStats:
    """Encode one plan component into the output. Tensors encode ahead on the shared pool and
    are written here, one at a time and in plan order, as a serial encoder would."""
    _validate_request(payload)
    if encoding not in _ENCODINGS or any(
        t.encoding not in (None, *_ENCODINGS) for t in plan.tensors
    ):
        raise UnsupportedInput(f"unsupported quantization encoding {encoding!r}")
    if component not in plan.components:
        raise UnsupportedInput(f"quantization plan has no component {component!r}")
    tensors = [t for t in plan.tensors if t.component == component]
    pending = [t for t in tensors if not output.has(t.component, t.key)]
    source_bytes = new_bytes = saturated = 0
    worst = 0.0
    total, reused = len(tensors), len(tensors) - len(pending)

    def scheme(tensor: QuantizationTensor) -> str:
        return tensor.encoding or encoding

    fitted = None
    if any(_ENCODINGS[scheme(t)].calibrated for t in pending):
        # Fitted one at a time on this thread: its GPU allocations are the attempt's, which the
        # memory manager answers by making room (varena serves torch). Each tensor is a step of the
        # run's record, with the host code any slow one waited in (`author._stalls`).
        fitted = tel.step_callback(len(pending), stage=f"fitting {component}")
        stop = threading.Event()
        results: Generator[_Encoded, None, None] = (
            _encode_tensor(source, scheme(t), t, stop, calibration) for t in pending
        )
    else:
        results = _in_order(
            pending, lambda tensor, stop: _encode_tensor(source, scheme(tensor), tensor, stop)
        )
    with contextlib.closing(results):
        for done, tensor in enumerate(pending, 1 + reused):
            ctx.raise_if_cancelled()
            result = next(results)
            name = scheme(tensor)
            encoder = _ENCODINGS[name]
            with result.spill as spill:
                bound = result.amax * encoder.round_trip_bound
                if not _numeric.within(result.max_error, bound):
                    raise UnsupportedInput(
                        f"{component}.{tensor.key}: round-trip error {result.max_error:.4g} "
                        f"exceeds the {name} bound {bound:.4g}",
                        code="quantization_tripwire",
                    )
                relative = math.sqrt(result.sse / result.energy) if result.energy > 0 else 0.0
                if payload.max_relative_frobenius is not None and not _numeric.within(
                    relative, payload.max_relative_frobenius
                ):
                    raise UnsupportedInput(
                        f"{component}.{tensor.key}: relative Frobenius {relative:.6f} exceeds "
                        f"the declared tripwire {payload.max_relative_frobenius}",
                        code="quantization_tripwire",
                    )
                rows, cols = tensor.shape
                spill_bytes = spill.tell()
                spill.seek(0)
                parts: list[PartData] = [
                    ("weight", encoder.weight_dtype, (rows, cols // encoder.pack), spill),
                    *(
                        (role, dtype, shape, raw)
                        for (role, dtype, shape), raw in zip(
                            encoder.companions(rows, cols), result.scales, strict=True
                        )
                    ),
                ]
                output.put_encoded(
                    component, tensor.key, TENSORFS_NAMES[logical_dtype(tensor)], tensor.shape,
                    name, encoder.params, parts,
                )  # fmt: skip
                new_bytes += spill_bytes + sum(len(raw) for raw in result.scales)
            worst = _numeric.worst(worst, relative)
            source_bytes += result.read
            saturated += result.saturated
            tel.log("quantized tensor", level="info", encoding=name, component=component,
                    key=tensor.key, done=done, total=total, **(result.fit or {}))  # fmt: skip
            tel.progress(done / total, stage=f"{encoding}-{component}")
            if fitted is not None:
                fitted(done - 1 - reused)
    return QuantizationStats(
        encoded_keys=len(pending),
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
    #: A calibrated scheme's fit, logged per tensor: its alpha and output errors (`svdq.Fitted`).
    fit: dict[str, float] | None = None


def _encode_tensor(
    source: Checkpoint,
    encoding: str,
    tensor: QuantizationTensor,
    stop: threading.Event,
    calibration: svdq.Calibration | None = None,
) -> _Encoded:
    encoder = _ENCODINGS[encoding]
    expected = encoder.companions(*tensor.shape)
    read_chunks = functools.partial(
        _read_chunks, source, tensor.component, tensor.key, tensor.logical_dtype, tensor.shape
    )
    if encoder.calibrated:
        return _fit_tensor(read_chunks, tensor, calibration)
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
            if len(encoded.companions) != len(expected):
                raise UnsupportedInput(f"{encoding} changed its declared geometry")
            for got, (role, dtype, shape), chunks in zip(
                encoded.companions, expected, scales, strict=True
            ):
                whole = not shape or role in encoder.whole
                want = list(shape) if whole else [count, *shape[1:]]
                if (got.dtype, got.shape) != (dtype, want):
                    raise UnsupportedInput(f"{encoding} changed its declared geometry")
                if not whole or not chunks:
                    chunks.append(got.raw)
                elif got.raw != chunks[0]:
                    raise UnsupportedInput(f"{encoding} encoded one tensor scale two ways")
            if values.size:
                amax = _numeric.worst(amax, float(np.max(np.abs(values))))
            max_error = _numeric.worst(max_error, encoded.max_abs_err)
            sse += encoded.sse
            energy += encoded.energy
            saturated += encoded.saturated
            read += values.size * _WIDTH[tensor.logical_dtype]
            rows += count
            spill.write(encoded.payload)
            del values, encoded
        if rows != tensor.shape[0]:
            raise UnsupportedInput(f"{encoding} changed its declared geometry")
        return _Encoded(
            spill, tuple(b"".join(chunks) for chunks in scales), amax, max_error, sse, energy,
            saturated, read,
        )  # fmt: skip
    except BaseException:
        spill.close()
        raise


def _fit_tensor(
    read_chunks: Callable[[], Iterator[np.ndarray]],
    tensor: QuantizationTensor,
    calibration: svdq.Calibration | None,
) -> _Encoded:
    """A calibrated scheme's whole tensor, fitted to its linear's recorded inputs."""
    if calibration is None:
        _uncalibrated(f"{tensor.component}.{tensor.key}")
    assert calibration is not None
    started = time.perf_counter()
    values = np.concatenate(list(read_chunks()))
    torch = calibration.torch
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    read = time.perf_counter()
    fitted = svdq.fit(
        torch, tensor.key, values, calibration.stats(tensor.component, tensor.key),
        "cuda" if cuda else "cpu",
    )  # fmt: skip
    measured = {
        "read_s": read - started,
        "fit_s": time.perf_counter() - read,
        "encode_s": fitted.encode_s,
        **_host_mb(),
    }
    if cuda:  # what this fit held, then given back: the model's pages need the room more
        measured["peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
        measured["reserved_mb"] = torch.cuda.memory_reserved() / 2**20
        torch.cuda.empty_cache()
    spill = tempfile.TemporaryFile()  # noqa: SIM115 - the caller closes it, as _encode_tensor's
    spill.write(fitted.payload)
    return _Encoded(
        spill,
        fitted.companions,
        fitted.amax,  # the bound holds the residual's round trip, in the smoothed domain
        fitted.max_abs_err,
        fitted.sse,
        fitted.energy,
        fitted.saturated,
        values.size * _WIDTH[tensor.logical_dtype],
        {"alpha": fitted.alpha, "error": fitted.error, "nvfp4_error": fitted.nvfp4_error}
        | {name: round(value, 2) for name, value in measured.items()},
    )


def _host_mb() -> dict[str, float]:
    """The host's memory now, from `/proc` (none where there is no `/proc`): this process's
    resident set, what the kernel says is available, and its page cache."""
    out: dict[str, float] = {}
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        out["rss_mb"] = pages * os.sysconf("SC_PAGE_SIZE") / 2**20
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in ("MemAvailable", "Cached"):
                out[{"MemAvailable": "avail_mb", "Cached": "cache_mb"}[name]] = (
                    int(value.split()[0]) / 1024
                )
    except (OSError, IndexError, ValueError):
        pass
    return out


class _Stopped(Exception):
    """The consumer left; encoding ahead of it is abandoned."""


def _in_order(
    items: Sequence[QuantizationTensor],
    produce: Callable[[QuantizationTensor, threading.Event], _Encoded],
) -> Generator[_Encoded, None, None]:
    """`produce` over `items` on the shared pool, a bounded window ahead, yielded in order.
    Closing the iterator stops the producers and releases every result nobody consumed."""
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


def _read_chunks(
    source: Checkpoint, component: str, key: str, logical_dtype: str, shape: tuple[int, ...]
) -> Iterator[np.ndarray]:
    """Whole rows of float32 values, about `_SOURCE_READ_CHUNK` source bytes at a time."""
    itemsize = _WIDTH[logical_dtype]
    rows, tail = (shape[0], shape[1:]) if shape else (1, ())
    row_bytes = math.prod(tail) * itemsize
    step = max(1, _SOURCE_READ_CHUNK // max(row_bytes, 1))
    for first in range(0, rows, step):
        count = min(step, rows - first)
        raw = bytearray(count * row_bytes)
        read_into(source, component, key, "", first * row_bytes, raw)
        values = st.to_f32(raw, TENSORFS_NAMES[logical_dtype], [count, *tail] if shape else [])
        if not bool(np.isfinite(values).all()):
            raise UnsupportedInput(f"source {component}.{key} contains non-finite weights")
        yield values


def _validate_request(payload: ArtifactQuantizationRequest) -> None:
    if payload.max_relative_frobenius is not None and (
        not math.isfinite(payload.max_relative_frobenius) or payload.max_relative_frobenius < 0
    ):
        raise UnsupportedInput("max_relative_frobenius must be one finite nonnegative value")
