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

import hashlib
import struct
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

from cozy_runtime.author import _codec
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
from cozy_runtime.author._observations import Attribution, EventRing, Observation, Scalar

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
class ProgressFrame:
    """One frame on the LOSSY live lane. Never retained: its durable form is `Attribution`
    plus the monotone `position` the supervisor's silence detector reads."""

    name: str
    fraction: float
    position: int


@dataclass(slots=True)
class Attempt:
    """The output/adjustment transaction one attempt owns. Runtime-side, never injected."""

    request_id: str
    spool: Path
    max_output_bytes: int = MAX_OUTPUT_BYTES
    closed: bool = False
    pending: dict[str, Asset] = field(default_factory=dict)
    rows: list[AdjustmentRow] = field(default_factory=list)
    ring: EventRing = field(default_factory=EventRing)
    """The BOUNDED observation record (cr-011). It replaced an unbounded list: a 4,096-step
    attempt used to grow one row per step, so the longest attempt — the one whose triage
    matters most — was the one that leaked."""
    attribution: Attribution = field(default_factory=Attribution)
    #: MONOTONE progress position, in completed units. The supervisor's liveness detector
    #: reads advance here and nowhere else; it is never a clock.
    position: int = 0
    sink: Callable[[ProgressFrame | Observation], None] | None = None
    """The runtime's LOSSY live lane (cr-007), installed by the executor.

    Best-effort and never authoritative: the lossy lane has no authority over a durable
    fact, and its failure cannot fail an attempt. RETAINED observations ride it too, and
    that is what makes a KILLED executor explainable — a ring that only crosses in the
    reply crosses never when the process that holds it is `kill -9`ed. The supervisor
    dedupes by the executor's own sequence, so nothing is counted twice.
    """
    #: The metric lane's own budget. A training job emits one row per step for tens of
    #: thousands of steps, so the RATE LIMIT is a number here rather than a hope: past it
    #: metrics are DROPPED and COUNTED, never silently, and never at the cost of the log tail.
    metric_budget: int = 4096
    metrics_emitted: int = 0
    metrics_dropped: int = 0
    _counter: int = 0
    _step_mark: float = 0.0

    def __post_init__(self) -> None:
        if not self.ring.subject:
            self.ring.subject = self.request_id

    def emit(self, kind: str, name: str, value: Scalar = None, **fields: Scalar) -> Observation:
        """One RETAINED record: the bounded ring, its emit-boundary checks, then the lane."""
        observation = self.ring.emit(cast("Any", kind), name, value, **fields)
        if self.sink is not None:
            self.sink(observation)
        return observation

    def signal(self, frame: ProgressFrame) -> None:
        """One LOSSY frame. Not retained, so a flood cannot evict the retained tail."""
        self.position = frame.position
        if self.sink is not None:
            self.sink(frame)

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
                f"image format {format!r} has no encoder: png is the image product "
                "(mp4/flac live in save_video/save_audio; webp, the integrity floor and "
                "C2PA are cr-017's remaining half)",
                code="encoder_unavailable",
            )
        return self._register(encode_png(as_frame(image)), ImageAsset, "image/png")

    def save_audio(self, waveform: object, *, sample_rate: int, format: str = "flac") -> AudioAsset:
        self._attempt.check_open("Outputs.save_audio")
        _codec.require_encoder()
        if format != "flac":
            raise OutputError(
                f"audio format {format!r} has no encoder: flac is the audio product and "
                "the codec zoo is a closed door (cr-017)",
                code="encoder_unavailable",
            )
        self._check_rate(sample_rate)
        planar, clipped = _codec.snapshot_audio(waveform, self._attempt.max_output_bytes)
        self._clip_row(clipped)
        return self._register(_codec.encode_flac(planar, sample_rate), AudioAsset, "audio/flac")

    def save_video(
        self,
        frames: object,
        *,
        fps: float,
        audio: object = None,
        sample_rate: int | None = None,
        format: str = "mp4",
    ) -> VideoAsset:
        self._attempt.check_open("Outputs.save_video")
        _codec.require_encoder()
        if format != "mp4":
            raise OutputError(
                f"video format {format!r} has no encoder: the one A/V product is the muxed "
                "mp4 (h264+aac) and the codec zoo is a closed door (cr-017)",
                code="encoder_unavailable",
            )
        if not (isinstance(fps, (int, float)) and 0 < fps < 1000 and fps == fps):
            raise OutputError(f"fps {fps!r} is not a positive frame rate", code="frame_rate")
        video = _codec.snapshot_video(frames, self._attempt.max_output_bytes)
        planar = None
        if audio is not None:
            if sample_rate is None:
                raise OutputError(
                    "save_video(audio=...) needs sample_rate: a waveform without its clock "
                    "is not muxable",
                    code="audio_rate",
                )
            self._check_rate(sample_rate)
            planar, clipped = _codec.snapshot_audio(audio, self._attempt.max_output_bytes)
            self._clip_row(clipped)
        data = _codec.encode_mp4(video, float(fps), planar, sample_rate)
        return self._register(data, VideoAsset, "video/mp4")

    @staticmethod
    def _check_rate(sample_rate: object) -> None:
        if not (isinstance(sample_rate, int) and 0 < sample_rate <= 384_000):
            raise OutputError(
                f"sample_rate {sample_rate!r} is not a positive sample rate", code="audio_rate"
            )

    def _clip_row(self, clipped: int) -> None:
        """The encoder's clamp is runtime policy, so it is VISIBLE as an adjustment row —
        never a silent rewrite of the author's waveform."""
        if clipped:
            self._attempt.rows.append(
                AdjustmentRow(
                    "clamp",
                    "audio",
                    f"{clipped} samples outside [-1, 1]",
                    "clamped to [-1, 1]",
                    "encoder input range is [-1, 1]; excursions are clamped and counted",
                    source="policy",
                )
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
        attempt = self._attempt
        attempt.check_open("Telemetry.progress")
        attempt.signal(
            ProgressFrame(stage or "", round(min(max(fraction, 0.0), 1.0), 6), attempt.position + 1)
        )

    def step_callback(self, total: int, *, stage: str | None = None) -> Any:
        """One hook doing progress + cancellation — every v1 endpoint hand-rolled this.

        It is also the STEP-TIME instrument: the interval between consecutive calls is
        accumulated in place (five floats per stage), so inference step time is attributable
        without one retained row per step. The hot-path cost is one `perf_counter` and a
        handful of float operations.
        """
        attempt = self._attempt
        attempt.check_open("Telemetry.step_callback")
        name = stage or "step"
        # The mark starts HERE, at callback construction, so the FIRST step has an interval
        # too. Starting it at the first call would silently drop step 0 from the count — and
        # on a one-step request that is the whole measurement.
        attempt._step_mark = time.perf_counter()

        def on_step(step: int) -> None:
            self._ctx.raise_if_cancelled()
            now = time.perf_counter()
            mark = attempt._step_mark
            if mark:
                attempt.attribution.step(name, (now - mark) * 1000)
            attempt._step_mark = now
            attempt.signal(ProgressFrame(name, (step + 1) / total if total else 1.0, step + 1))

        return on_step

    def log(self, message: str, *, level: str = "info", **fields: Scalar) -> None:
        self._attempt.check_open("Telemetry.log")
        self._attempt.emit("log", message, level, **fields)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Per-stage timing bracket; `stage_ms` feeds the duration model and benchmarks.

        Structurally separate from a component-use scope: this records a NUMBER and touches
        no residency, no placement and no ComponentUseContract — `scripts/fence.py`'s
        `timing-decides-nothing` fence proves the module cannot reach them.
        """
        self._attempt.check_open("Telemetry.stage")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self._attempt.attribution.stage(name, elapsed)
            self._attempt.emit("stage", name, round(elapsed, 3))

    def metric(self, name: str, value: float, *, unit: str | None = None) -> None:
        self._attempt.check_open("Telemetry.metric")
        if self._admit():
            self._attempt.emit("metric", name, value, unit=unit or "")

    def training_metric(self, name: str, value: float, *, step: int) -> None:
        """The per-step training lane. RATE-LIMITED like `metric`: a 50,000-step job must
        not be able to spend the attempt's whole observation budget on one number."""
        self._attempt.check_open("Telemetry.training_metric")
        if self._admit():
            self._attempt.emit("metric", name, value, step=step)

    def _admit(self) -> bool:
        attempt = self._attempt
        if attempt.metrics_emitted >= attempt.metric_budget:
            attempt.metrics_dropped += 1
            return False
        attempt.metrics_emitted += 1
        return True


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


# ----------------------------------------------------------------------- job services


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    """Proof one durable checkpoint save completed (suffix law: a Receipt, never a Decision).

    `replayed` says the identity was already durable and this call returned the SAME
    receipt — the idempotency fact a resumed job needs, and never a second write.
    """

    receipt_id: str
    operation_key: str
    logical_key: str
    content_digest: str
    length: int
    replayed: bool = False


class CheckpointConflict(CapabilityError):
    """One logical/operation key, two different contents. Never a replacement (§2)."""

    default_code = "checkpoint_conflict"


@dataclass(frozen=True, slots=True)
class CheckpointSave:
    """What the runtime is asked to make durable. Bytes stay in the attempt's own spool."""

    operation_key: str
    logical_key: str
    content_digest: str
    length: int
    path: Path


@dataclass(frozen=True, slots=True)
class BudgetFacts:
    """The cost-governor facts for ONE job attempt, from the TYPED CONFIG CHANNEL (§2).

    Cost facts are not filesystem state and the GPU rate never arrives through environment:
    the deployment config plane owns these values and they are digest-pinned into the
    execution spec like any other setting.
    """

    gpu_rate_usd_per_hour: float = 0.0
    gpu_count: int = 0
    cap_usd: float = 0.0
    started_monotonic: float = 0.0


class BudgetExhausted(CapabilityError):
    """The attempt reached its declared spend cap. A FACT, never a retry command."""

    default_code = "budget_exhausted"


class Scratch(_Bound):
    """Job service (§2) — bounded working storage for ONE run.

    `checkpoint_dir(key=)` is keyed on the RUN, not the attempt: a retried attempt of the
    same run is handed back the SAME tree, which is what makes resume real rather than a
    naming convention. `mktemp()` and `materialize_blob()` are attempt-scoped and die with
    the attempt spool.
    """

    __slots__ = ("_root",)

    def __init__(self, attempt: Attempt, root: Path | None = None) -> None:
        super().__init__(attempt)
        self._root = root

    def _base(self, what: str) -> Path:
        self._attempt.check_open(what)
        if self._root is None:
            raise CapabilityError(
                f"{what}: no scratch is granted to this attempt — Scratch is a JOB service, "
                "and the runtime allocates its tree per run before the body starts",
                code="scratch_unavailable",
            )
        return self._root

    def checkpoint_dir(self, *, key: str) -> Path:
        """The SAME directory for the same (run, key) across attempts. Resumable by
        construction — a retried attempt reads what the killed one left."""
        base = self._base("Scratch.checkpoint_dir")
        if not key or "/" in key or key.startswith("."):
            raise CapabilityError(
                f"scratch key {key!r} is not a bare name: a key is a flat label, never a path",
                code="scratch_key",
            )
        target = base / "checkpoints" / key
        target.mkdir(parents=True, exist_ok=True)
        return target

    def mktemp(self, suffix: str = "") -> Path:
        """An attempt-scoped temporary path. Dies with the attempt; never resumable."""
        base = self._base("Scratch.mktemp")
        target = base / "tmp"
        target.mkdir(parents=True, exist_ok=True)
        return target / f"{self._attempt.next_handle('scratch').rsplit('/', 1)[-1]}{suffix}"

    def materialize_blob(self, data: bytes, *, name: str = "") -> Path:
        """Put bytes on the filesystem for a library that only takes a path."""
        target = self.mktemp() if not name else self._base("Scratch.materialize_blob") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target


class Checkpoints(_Bound):
    """Job service (§2) — DURABLE idempotent saves over a distinct exchange.

    Identity is (run, attempt, operation key, logical key, content digest). Repeating the
    same identity returns the durable receipt; the same logical/operation key with different
    bytes is a CONFLICT, never a replacement. The write is journaled and acknowledged before
    this call reports success, over the durable control exchange — never the lossy stream,
    which has no authority over a durable fact.
    """

    __slots__ = ("_sink",)

    def __init__(
        self, attempt: Attempt, sink: Callable[[CheckpointSave], CheckpointReceipt] | None = None
    ) -> None:
        super().__init__(attempt)
        self._sink = sink

    def save(
        self, logical_key: str, data: bytes | Path, *, operation_key: str = ""
    ) -> CheckpointReceipt:
        """Make one checkpoint durable, and return only once it IS."""
        self._attempt.check_open("Checkpoints.save")
        if self._sink is None:
            raise CapabilityError(
                "no durable checkpoint exchange is installed for this attempt: Checkpoints is "
                "a JOB service and its exchange is the supervisor's, never a local file write",
                code="checkpoints_unavailable",
            )
        if not logical_key:
            raise CapabilityError("a checkpoint needs a logical key", code="checkpoint_key")
        blob = Path(data) if isinstance(data, Path) else None
        if blob is None:
            assert isinstance(data, bytes)
            blob = self._attempt.spool / f"ckpt-{self._attempt.next_handle('ckpt')[-4:]}"
            blob.write_bytes(data)
        raw = blob.read_bytes()
        return self._sink(
            CheckpointSave(
                operation_key=operation_key or logical_key,
                logical_key=logical_key,
                # SHA-256, the one identity spelling the platform uses for content (law 1),
                # so the receipt the coordinator holds is addressable by the same digest.
                content_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
                length=len(raw),
                path=blob,
            )
        )


class Budget(_Bound):
    """Job service (§2) — the attempt's spend envelope, as its own service.

    Cost facts are NOT filesystem state and the GPU rate is not an environment variable:
    everything here comes from the typed config channel. It reports facts and raises when
    the declared cap is reached; it never decides retryability and never prices anything.
    """

    __slots__ = ("_facts",)

    def __init__(self, attempt: Attempt, facts: BudgetFacts | None = None) -> None:
        super().__init__(attempt)
        self._facts = facts

    def _read(self) -> BudgetFacts:
        self._attempt.check_open("Budget")
        if self._facts is None:
            raise CapabilityError(
                "no budget is granted to this attempt: Budget is a JOB service and its facts "
                "arrive on the typed config channel, never from the environment",
                code="budget_unavailable",
            )
        return self._facts

    @property
    def gpu_rate_usd_per_hour(self) -> float:
        return self._read().gpu_rate_usd_per_hour

    @property
    def cap_usd(self) -> float:
        return self._read().cap_usd

    def elapsed_s(self) -> float:
        facts = self._read()
        return max(0.0, time.monotonic() - facts.started_monotonic)

    def spent_usd(self) -> float:
        """GPU-seconds x rate x devices. The one arithmetic, in the one place."""
        facts = self._read()
        return self.elapsed_s() / 3600.0 * facts.gpu_rate_usd_per_hour * max(facts.gpu_count, 1)

    def remaining_usd(self) -> float:
        return self._read().cap_usd - self.spent_usd()

    def raise_if_exhausted(self) -> None:
        """THE spend spelling, mirroring `ctx.raise_if_cancelled()`."""
        facts = self._read()
        if facts.cap_usd and self.spent_usd() >= facts.cap_usd:
            raise BudgetExhausted(
                f"the attempt spent {self.spent_usd():.4f} USD of a {facts.cap_usd:.4f} USD "
                f"cap after {self.elapsed_s():.1f} s on {max(facts.gpu_count, 1)} device(s)"
            )


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

#: The one service set a JOB may request and a serving entrypoint may not (§2). A job runs
#: to completion and owns durable working state; a request handler owns neither.
JOB_ONLY: frozenset[str] = frozenset({"scratch", "checkpoints", "budget"})

#: The mirror: SERVING-ONLY surface. `adjust` writes the caller-visible adjustments envelope,
#: which is a fact about serving one request; a run-to-completion job has no caller to
#: confess to and no envelope to write into.
SERVING_ONLY: frozenset[str] = frozenset({"adjust"})
