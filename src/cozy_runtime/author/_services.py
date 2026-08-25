"""The typed services. A signature that names one IS the declaration of that capability.

`describe` derives the AuthorCapabilitySet from these parameter types (§1.3), so there is
no capability list to drift from the code: a handler without `Outputs` structurally
cannot save, one without `Secrets` structurally cannot read a secret. The set names powers
granted to AUTHOR code — the runtime still journals, fetches, stages and emits telemetry
under its own policy, and none of that appears in an author-facing schema.

Every service is ATTEMPT-BOUND: it stops working the moment its attempt closes, which is
what makes "escaped handle" a typed failure instead of a review rule.
"""

from __future__ import annotations

import struct
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from cozy_runtime.author._assets import (
    Asset,
    AudioAsset,
    FileAsset,
    ImageAsset,
    VideoAsset,
    digest_bytes,
)
from cozy_runtime.author._context import Context
from cozy_runtime.author._errors import CapabilityError, OutputError

T = TypeVar("T")
S = TypeVar("S")
A = TypeVar("A", bound="Asset")

RowKind = Literal["adjusted", "clamp", "warn"]

#: Default per-attempt output ceiling. The deployment lowers it; nothing raises it in code.
MAX_OUTPUT_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AdjustmentRow:
    """One caller-visible divergence between requested and served (§1.2)."""

    kind: RowKind
    field: str
    requested: object
    applied: object
    reason: str
    source: str = "handler"
    """`handler`, or the runtime-owned origin (`checkpoint:…`, `adapter:…`, `policy`)."""


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    kind: Literal["progress", "log", "stage", "metric"]
    name: str
    value: object = None
    fields: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Attempt:
    """The output/adjustment transaction one attempt owns. Runtime-side, never injected."""

    request_id: str
    spool: Path
    max_output_bytes: int = MAX_OUTPUT_BYTES
    closed: bool = False
    pending: dict[str, Asset] = field(default_factory=dict)
    rows: list[AdjustmentRow] = field(default_factory=list)
    events: list[TelemetryEvent] = field(default_factory=list)
    sink: Callable[[TelemetryEvent], None] | None = None
    """The runtime's LOSSY progress lane (cr-007), installed by the executor.

    Events are always recorded for the terminal envelope; the sink is an additional,
    best-effort forward so a long attempt renders live. It cannot fail an attempt: the
    lossy lane never has authority over a durable fact.
    """
    _counter: int = 0

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)
        if self.sink is not None:
            self.sink(event)

    def next_handle(self, kind: str) -> str:
        self._counter += 1
        return f"attempt:{self.request_id}/{kind}/{self._counter:04d}"

    def check_open(self, what: str) -> None:
        if self.closed:
            raise CapabilityError(
                f"{what} used after attempt {self.request_id} closed: an escaped service "
                "handle is a typed failure, never a silent write",
                code="escaped_handle",
            )


class _Bound:
    """Common attempt binding for every injected service."""

    __slots__ = ("_attempt",)

    def __init__(self, attempt: Attempt) -> None:
        self._attempt = attempt


# --------------------------------------------------------------------------- encode


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """Raw RGB pixels — the SDK's own encoder input, so no imaging library is required."""

    width: int
    height: int
    rgb: bytes

    def __post_init__(self) -> None:
        expect = self.width * self.height * 3
        if len(self.rgb) != expect:
            raise OutputError(
                f"image frame is {len(self.rgb)} bytes, expected {expect} "
                f"for {self.width}x{self.height} RGB",
                code="frame_shape",
            )


@runtime_checkable
class _PilLike(Protocol):
    """Structural view of a PIL image — duck-typed, never imported."""

    mode: str
    size: tuple[int, int]

    def tobytes(self) -> bytes: ...


def as_frame(image: object) -> ImageFrame:
    if isinstance(image, ImageFrame):
        return image
    if isinstance(image, _PilLike):
        if image.mode != "RGB":
            raise OutputError(f"image mode {image.mode!r} is not RGB", code="frame_mode")
        width, height = image.size
        return ImageFrame(width, height, image.tobytes())
    raise OutputError(
        f"cannot encode {type(image).__name__}: pass an ImageFrame or an RGB image",
        code="frame_type",
    )


def encode_png(frame: ImageFrame) -> bytes:
    """A real, dependency-free PNG encoder — the encode the fakes exercise IS this one.

    cr-017 owns the codec plane (webp/mp4/flac, the capability probe, the integrity floor,
    and C2PA signing over the final container). This is the one format that exists today,
    and it is genuinely produced, not stubbed.
    """
    raw = bytearray()
    stride = frame.width * 3
    for y in range(frame.height):
        raw.append(0)
        raw += frame.rgb[y * stride : (y + 1) * stride]

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", frame.width, frame.height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


# --------------------------------------------------------------------------- services


class Outputs(_Bound):
    """Transactional outputs (`out`). Every save REGISTERS a pending handle and validates
    its declared bounds; nothing is durable until the handler returns it (§3.4 step 1)."""

    __slots__ = ()

    def save_image(self, image: object, *, format: str = "png") -> ImageAsset:
        if format != "png":
            raise OutputError(
                f"format {format!r} has no encoder in this release: png is the one that "
                "exists; the codec plane (webp/mp4/flac + integrity floor + C2PA) is cr-017",
                code="encoder_unavailable",
            )
        return self._register(encode_png(as_frame(image)), ImageAsset, "image/png")

    def save_audio(self, waveform: object, *, sample_rate: int, format: str = "flac") -> AudioAsset:
        raise OutputError(
            f"audio encode ({format}, {sample_rate} Hz) is cr-017's codec plane",
            code="encoder_unavailable",
        )

    def save_video(
        self,
        frames: object,
        *,
        fps: float,
        audio: object = None,
        sample_rate: int | None = None,
        format: str = "mp4",
    ) -> VideoAsset:
        raise OutputError(
            f"video encode ({format}, {fps} fps) — including the A/V mux that makes one "
            "container out of frames+audio — is cr-017's codec plane",
            code="encoder_unavailable",
        )

    def save_bytes(self, data: bytes, *, media_type: str) -> FileAsset:
        return self._register(data, FileAsset, media_type)

    def save_file(
        self, path: Path | str, *, media_type: str = "application/octet-stream"
    ) -> FileAsset:
        """Copies immediately: a MUTABLE AUTHOR PATH is never retained (§3.4 step 2)."""
        return self._register(Path(path).read_bytes(), FileAsset, media_type)

    def temporary_file(self, suffix: str = "") -> Path:
        """A bounded attempt-spool target for encoders that need a filesystem path.

        Becomes an output only when a save converts it; otherwise it dies with the spool.
        No general scratch directory exists for endpoints.
        """
        self._attempt.check_open("Outputs.temporary_file")
        tmp = self._attempt.spool / "tmp"
        tmp.mkdir(exist_ok=True)
        return tmp / f"{self._attempt.next_handle('tmp').rsplit('/', 1)[-1]}{suffix}"

    def _register(self, data: bytes, kind: type[A], media_type: str) -> A:
        self._attempt.check_open("Outputs.save")
        if len(data) > self._attempt.max_output_bytes:
            raise OutputError(
                f"output is {len(data)} bytes, over the attempt limit of "
                f"{self._attempt.max_output_bytes}",
                code="output_too_large",
            )
        handle = self._attempt.next_handle(kind.kind)
        blob = self._attempt.spool / f"{kind.kind}-{handle.rsplit('/', 1)[-1]}"
        blob.write_bytes(data)
        asset = kind(
            handle,
            media_type=media_type,
            size_bytes=len(data),
            digest=digest_bytes(data),
            local=blob,
            attempt=self._attempt.request_id,
        )
        self._attempt.pending[handle] = asset
        return asset


class Telemetry(_Bound):
    """User-facing status and operator diagnostics (`tel`). Observability only: it never
    changes availability, placement, or the ComponentUseContract."""

    __slots__ = ("_ctx",)

    def __init__(self, attempt: Attempt, ctx: Context) -> None:
        super().__init__(attempt)
        self._ctx = ctx

    def progress(self, fraction: float, *, stage: str | None = None) -> None:
        self._attempt.check_open("Telemetry.progress")
        self._attempt.emit(
            TelemetryEvent("progress", stage or "", round(min(max(fraction, 0.0), 1.0), 6))
        )

    def step_callback(self, total: int, *, stage: str | None = None) -> Any:
        """One hook doing progress + cancellation — every v1 endpoint hand-rolled this."""
        self._attempt.check_open("Telemetry.step_callback")

        def on_step(step: int) -> None:
            self._ctx.raise_if_cancelled()
            self.progress((step + 1) / total if total else 1.0, stage=stage)

        return on_step

    def log(self, message: str, *, level: str = "info", **fields: object) -> None:
        self._attempt.check_open("Telemetry.log")
        self._attempt.emit(TelemetryEvent("log", message, level, dict(fields)))

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Per-stage timing bracket; `stage_ms` feeds the duration model and benchmarks."""
        self._attempt.check_open("Telemetry.stage")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self._attempt.emit(TelemetryEvent("stage", name, round(elapsed, 3)))

    def metric(self, name: str, value: float, *, unit: str | None = None) -> None:
        self._attempt.check_open("Telemetry.metric")
        self._attempt.emit(TelemetryEvent("metric", name, value, {"unit": unit or ""}))

    def training_metric(self, name: str, value: float, *, step: int) -> None:
        self._attempt.check_open("Telemetry.training_metric")
        self._attempt.emit(TelemetryEvent("metric", name, value, {"step": step}))


class Adjustments(_Bound):
    """Handler-authored rows in the adjustments envelope (`adj`).

    Runtime-owned rows — ModelDefault resolution and deployment policy clamps — reach the
    envelope WITHOUT this service; a handler requests it only when its own model logic
    performs an additional caller-visible adjustment (§1.2).
    """

    __slots__ = ()

    def adjusted(self, field: str, requested: object, applied: object, reason: str) -> None:
        self._row("adjusted", field, requested, applied, reason)

    def clamp(self, field: str, requested: object, applied: object, reason: str) -> None:
        self._row("clamp", field, requested, applied, reason)

    def warn(self, field: str, message: str) -> None:
        self._row("warn", field, None, None, message)

    def _row(
        self, kind: RowKind, field: str, requested: object, applied: object, reason: str
    ) -> None:
        self._attempt.check_open("Adjustments")
        self._attempt.rows.append(AdjustmentRow(kind, field, requested, applied, reason))


@dataclass(frozen=True, slots=True)
class EgressResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class Egress(_Bound):
    """Supervisor-brokered outbound HTTP (`net`) — connector endpoints only (§1.3).

    Injection and capability derivation are this issue's; the BROKER is cr-012's and its
    consumer is se-006. With no broker installed the call refuses typed — the executor
    keeps zero outbound network either way, so there is nothing here to fall back to.
    """

    __slots__ = ("_broker",)

    def __init__(self, attempt: Attempt, broker: Any = None) -> None:
        super().__init__(attempt)
        self._broker = broker

    def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
    ) -> EgressResponse:
        self._attempt.check_open("Egress.fetch")
        if self._broker is None:
            raise CapabilityError(
                f"no egress broker is installed for {method} {url}: the supervisor-side "
                "broker, allowlist and SSRF fail-closed path are cr-012",
                code="egress_broker_unavailable",
            )
        result = self._broker(url, method, dict(headers or {}), body)
        assert isinstance(result, EgressResponse)
        return result


class Settings(Generic[T]):
    """Typed deployment settings. The author declares the strict schema; the VALUES are
    owned by the deployment config plane, validated against it and digest-pinned into the
    execution spec. No ambient environment variable ever carries configuration (§3.5)."""

    __slots__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value = value


class Secrets(Generic[S]):
    """Granted secrets against a strict named schema — a bare unbounded bag is invalid.
    Never raw getenv; injectable only when the capability is declared."""

    __slots__ = ("value",)

    def __init__(self, value: S) -> None:
        self.value = value


class Scratch(_Bound):
    """Job service (§2) — bounded working storage. cr-009 owns the implementation."""

    __slots__ = ()


class Checkpoints(_Bound):
    """Job service (§2) — resumable training checkpoints. cr-009 owns the implementation."""

    __slots__ = ()


class Budget(_Bound):
    """Job service (§2) — the attempt's spend envelope. cr-009 owns the implementation."""

    __slots__ = ()


#: capability name -> service type. The ONE table injection and derivation share, so a
#: capability cannot exist without an injectable service or vice versa.
SERVICES: Mapping[str, type] = {
    "save": Outputs,
    "telemetry": Telemetry,
    "adjust": Adjustments,
    "egress": Egress,
    "settings": Settings,
    "secrets": Secrets,
    "scratch": Scratch,
    "checkpoints": Checkpoints,
    "budget": Budget,
}
