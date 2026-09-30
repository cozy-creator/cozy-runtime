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
import math
import os
import shutil
import stat
import threading
import time
from collections.abc import Buffer, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol, TypeVar, cast, runtime_checkable

from cozy_runtime.author import _codec
from cozy_runtime.author._assets import (
    Asset,
    AudioAsset,
    FileAsset,
    ImageAsset,
    Tree,
    VideoAsset,
    digest_bytes,
)
from cozy_runtime.author._capture import _capture_clock
from cozy_runtime.author._context import Context
from cozy_runtime.author._decode import (
    DecodedAudioChunk,
    DecodedAudioFormat,
    DecodedMediaEvent,
    DecodedMediaHeader,
    DecodedVideoFrame,
    MediaDecoder,
)
from cozy_runtime.author._errors import AuthorError, CapabilityError, OutputError
from cozy_runtime.author._executor_requests import Publish, Published, PublishPart
from cozy_runtime.author._model_reader import WeightsReader
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
    """One bounded frame on the LOSSY live lane; never durable lifecycle state."""

    stage: str
    stage_fraction: float | None
    advance: int
    position: int | None = None
    total: int | None = None
    overall_fraction: float | None = None
    step_ms: float = 0.0
    call_request: str | None = None
    call_attempt: int | None = None


@dataclass(frozen=True, slots=True)
class _ProgressScope:
    telemetry: Telemetry
    name: str
    bounds: tuple[float, float] | None


_progress_scope: ContextVar[_ProgressScope | None] = ContextVar("cozy_progress_scope", default=None)


@dataclass(frozen=True, slots=True)
class HostFrame:
    """One output a save REGISTERED and nobody has encoded yet (cr-079).

    The raw host snapshot sits in the attempt spool beside the facts its codec needs; the
    handler returned without a codec running, so the device phase ends at the reply. The
    worker's post thread encodes it, bounds the ENCODED bytes and writes them under the
    grant. `charge` is what the registration counted against the attempt's aggregate
    output ceiling: the raw bytes for a lossless codec (never smaller than its output),
    nothing for a lossy one, whose only pre-encode bound is the snapshot ceiling.
    """

    handle: str
    codec: str
    raw: Path
    raw_bytes: int
    charge: int
    facts: dict[str, Any]

    @property
    def media_type(self) -> str:
        return _codec.FRAME_MEDIA_TYPES[self.codec]

    def row(self) -> dict[str, Any]:
        """The seam row: a spool FILE NAME and facts, never an absolute path."""
        return {
            "handle": self.handle,
            "codec": self.codec,
            "media_type": self.media_type,
            "raw": self.raw.name,
            "raw_bytes": self.raw_bytes,
            "facts": dict(self.facts),
        }


@dataclass(slots=True)
class Attempt:
    """The output/adjustment transaction one attempt owns. Runtime-side, never injected."""

    request_id: str
    spool: Path
    max_output_bytes: int = MAX_OUTPUT_BYTES
    closed: bool = False
    frames: dict[str, HostFrame] = field(default_factory=dict)
    """handle -> the registered host frame awaiting its codec. In the executor these cross
    the seam and the worker's post thread settles them; the asset in `pending` carries
    the media type and no size or digest until then."""
    settle_at_save: bool = False
    """Encode at the save instead (a fake harness: there is no worker to defer to)."""
    entered: bool = False
    """Whether the package handler itself was entered. Pre-entry refusals never poison a
    healthy executor; one structural bit is safer than duplicating refusal-code lists."""
    failed_at_call: bool = False
    """Whether what ended the handler was a package call's own failure, raised at the call
    boundary: a safe point like a cancel, so the process stays reusable (run 1514's parent
    paid a fresh 7.8 s start in run 1516 for a child's OOM)."""
    pending: dict[str, Asset] = field(default_factory=dict)
    committed_files: dict[str, FileAsset] = field(default_factory=dict)
    pending_trees: dict[str, Tree] = field(default_factory=dict)
    rows: list[AdjustmentRow] = field(default_factory=list)
    ignored: tuple[str, ...] = ()
    """The request fields its type does not declare, dropped before it decoded."""
    ring: EventRing = field(default_factory=EventRing)
    """The BOUNDED observation record (cr-011). It replaced an unbounded list: a 4,096-step
    attempt used to grow one row per step, so the longest attempt — the one whose triage
    matters most — was the one that leaked."""
    attribution: Attribution = field(default_factory=Attribution)
    #: MONOTONE progress position, in completed units. The worker's liveness detector
    #: reads advance here and nowhere else; it is never a clock.
    position: int = 0
    sink: Callable[[ProgressFrame | Observation], None] | None = None
    """The runtime's LOSSY live lane (cr-007), installed by the executor.

    Best-effort and never authoritative: the lossy lane has no authority over a durable
    fact, and its failure cannot fail an attempt. RETAINED observations ride it too, and
    that is what makes a KILLED executor explainable — a ring that only crosses in the
    reply crosses never when the process that holds it is `kill -9`ed. The worker
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
    #: Sends one publish to the worker, which journals it on the run's output log. None in a
    #: harness with no worker: `published` then records what would have been shown.
    publish: Callable[[Publish], Published] | None = None
    published: list[Publish] = field(default_factory=list)
    #: Handles delivered as products, or joined into one: the result need not return them.
    shown: set[str] = field(default_factory=set)
    #: A composite asset's parts, by its handle: what `publish` sends instead of one file.
    parts: dict[str, tuple[PublishPart, ...]] = field(default_factory=dict)
    _overall_fraction: float | None = None
    _progress_diagnostics: set[str] = field(default_factory=set)
    #: Overlapped calls (`author.concurrently`) record from their own threads.
    _recording: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not self.ring.subject:
            self.ring.subject = self.request_id

    def emit(self, kind: str, name: str, value: Scalar = None, **fields: Scalar) -> Observation:
        """One RETAINED record: the bounded ring, its emit-boundary checks, then the lane."""
        with self._recording:
            observation = self.ring.emit(cast("Any", kind), name, value, **fields)
            if self.sink is not None:
                self.sink(observation)
        return observation

    def signal(self, frame: ProgressFrame) -> None:
        """One LOSSY frame. Not retained, so a flood cannot evict the retained tail."""
        self.position = frame.advance
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


@dataclass(frozen=True, slots=True)
class SavedAudioFacts:
    """Audio facts re-probed from one committed streaming MP4."""

    codec: str
    profile: str | None
    channels: int
    sample_rate: int
    codec_frame_samples: int
    decoded_samples: int


@dataclass(frozen=True, slots=True)
class SavedVideo:
    """One atomically committed streaming MP4 plus facts re-probed from its bytes."""

    video: VideoAsset
    width: int
    height: int
    frame_count: int
    frame_rate: Fraction
    pixel_aspect_ratio: Fraction
    video_codec: str
    video_profile: str | None
    color_primaries: int
    color_transfer: int
    color_matrix: int
    color_range: int
    audio: SavedAudioFacts | None
    submitted_audio_samples: int


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
    """Encode exact RGB pixels as maximum-compression lossless PNG."""

    return _codec.encode_png(frame.width, frame.height, frame.rgb)


def settle_frame(attempt: Attempt, handle: str) -> Asset:
    """Encode ONE registered frame in place: the blob takes the handle's spool name, the
    asset takes its size and digest, the raw snapshot goes. The worker's post thread does
    the same over the seam row; this is the in-process form a fake harness uses."""
    frame = attempt.frames.pop(handle)
    asset = attempt.pending[handle]
    data = _codec.encode_frame(frame.codec, frame.facts, frame.raw.read_bytes())
    settled = sum(a.size_bytes for a in attempt.pending.values()) + sum(
        f.charge for f in attempt.frames.values()
    )
    if settled + len(data) > attempt.max_output_bytes:
        raise OutputError(
            f"output {handle} encodes to {len(data)} bytes, bringing the attempt to "
            f"{settled + len(data)} bytes over its {attempt.max_output_bytes}-byte limit",
            code="output_too_large",
        )
    blob = attempt.spool / f"{asset.kind}-{handle.rsplit('/', 1)[-1]}"
    blob.write_bytes(data)
    frame.raw.unlink(missing_ok=True)
    asset.size_bytes = len(data)
    asset.digest = digest_bytes(data)
    asset._local = blob
    return asset


# --------------------------------------------------------------------------- services


class Outputs(_Bound):
    """Transactional outputs (``out``). Save registers a bounded pending handle.

    Returning a saved value completes it with the invocation. ``await commit(file)``
    completes an owned file earlier so the running caller can forward it.
    """

    __slots__ = ()

    @property
    def video_audio_frame_samples(self) -> int:
        """The fixed audio codec quantum used by ``save_video_stream``."""

        return _codec.AAC_FRAME_SAMPLES

    def save_image(self, image: object, *, format: str = "png") -> ImageAsset:
        frame = as_frame(image)
        if format not in ("png", "webp"):
            raise OutputError(
                f"image format {format!r} has no encoder: choose png or lossless webp",
                code="encoder_unavailable",
            )
        return self._register_frame(
            ImageAsset,
            format,
            (frame.rgb,),
            {"width": frame.width, "height": frame.height},
            charge=len(frame.rgb),
        )

    def save_audio(self, waveform: object, *, sample_rate: int, format: str = "flac") -> AudioAsset:
        self._attempt.check_open("Outputs.save_audio")
        if format != "flac":
            raise OutputError(
                f"audio format {format!r} has no encoder: flac is the audio product and "
                "the codec zoo is a closed door (cr-017)",
                code="encoder_unavailable",
            )
        self._check_rate(sample_rate)
        planar, clipped = _codec.snapshot_audio(waveform, self._attempt.max_output_bytes)
        self._clip_row(clipped)
        channels, samples = planar.shape
        return self._register_frame(
            AudioAsset,
            "flac",
            (planar.data,),
            {"channels": channels, "samples": samples, "sample_rate": sample_rate},
            charge=len(planar),
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
        self._attempt.check_open("Outputs.save_video")
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
        facts: dict[str, Any] = {
            "shape": list(video.shape),
            "fps": float(fps),
            "video_bytes": len(video),
            "audio_channels": planar.shape[0] if planar is not None else 0,
            "audio_samples": planar.shape[1] if planar is not None else 0,
            "sample_rate": int(sample_rate or 0),
        }
        parts = (video.data,) if planar is None else (video.data, planar.data)
        # h264/aac are lossy: the raw snapshot bounds nothing about the container, so the
        # registration charges nothing and the post phase bounds the encoded bytes.
        return self._register_frame(VideoAsset, "mp4", parts, facts, charge=0)

    def save_video_stream(self, events: Iterator[DecodedMediaEvent]) -> SavedVideo:
        """Consume one canonical event stream into one bounded, probed, atomic MP4."""
        self._attempt.check_open("Outputs.save_video_stream")
        _codec.require_encoder()
        source = iter(events)
        try:
            header = next(source)
        except StopIteration:
            raise OutputError("streaming video has no header", code="media_header") from None
        if not isinstance(header, DecodedMediaHeader) or header.video is None:
            raise OutputError(
                "streaming video must begin with one video header", code="media_header"
            )
        video_format = header.video
        if video_format.nominal_frame_rate is None:
            raise OutputError("streaming video header has no nominal frame rate", code="frame_rate")
        rate = video_format.nominal_frame_rate
        audio_format = header.audio

        def feed(encoder: _codec.StreamingMP4Encoder) -> int:
            audio_samples = 0
            for event in source:
                self._attempt.check_open("Outputs.save_video_stream")
                if isinstance(event, DecodedMediaHeader):
                    raise OutputError(
                        "streaming video contains a second header", code="media_header"
                    )
                if not isinstance(event, DecodedVideoFrame):
                    audio_samples = self._write_audio(encoder, audio_format, event, audio_samples)
                    continue
                if (
                    event.width != video_format.width
                    or event.height != video_format.height
                    or event.time_base != video_format.time_base
                    or event.pixel_aspect_ratio != video_format.pixel_aspect_ratio
                    or event.color_primaries != video_format.color_primaries
                    or event.color_transfer != video_format.color_transfer
                    or event.color_matrix != video_format.color_matrix
                    or event.color_range != video_format.color_range
                ):
                    raise OutputError(
                        "streaming video frame differs from its header", code="frame_shape"
                    )
                expected = Fraction(encoder.video_frames, 1) / rate
                if event.start_time != expected:
                    raise OutputError(
                        f"streaming video frame starts at {event.start_time}, expected {expected}",
                        code="video_discontinuity",
                    )
                if event.duration * event.time_base != 1 / rate:
                    raise OutputError(
                        "streaming video frame duration differs from its output rate",
                        code="video_discontinuity",
                    )
                encoder.write_video(event.rgb)
            return audio_samples

        return self._save_mp4(
            dict(
                width=video_format.width,
                height=video_format.height,
                frame_rate=rate,
                pixel_aspect_ratio=video_format.pixel_aspect_ratio,
                color_primaries=video_format.color_primaries,
                color_transfer=video_format.color_transfer,
                color_matrix=video_format.color_matrix,
                color_range=video_format.color_range,
            ),
            audio_format,
            feed,
        )

    def save_video_concat(
        self,
        videos: Sequence[VideoAsset],
        soundtrack: Callable[[Sequence[int]], Iterator[DecodedMediaEvent]],
    ) -> SavedVideo:
        """Join H.264 videos by packet copy under one newly encoded soundtrack.

        No picture is decoded or encoded. Every video must be one H.264 track with the same
        parameters, starting on a keyframe, on a zero-based fixed frame clock; otherwise it
        refuses `video_copy_incompatible` before `soundtrack` is called. `soundtrack` receives
        each video's frame count and returns an audio-only header and its chunks on the
        zero-based sample clock, encoded as `save_video_stream` encodes audio.
        """
        self._attempt.check_open("Outputs.save_video_concat")
        _codec.require_encoder()
        if not videos:
            raise OutputError("a joined video needs at least one video", code="media_header")
        paths = []
        for video in videos:
            if (
                not isinstance(video, VideoAsset)
                or video._local is None
                or video._attempt != self._attempt.request_id
            ):
                raise CapabilityError(
                    "joined videos must be hydrated inputs of this attempt", code="foreign_asset"
                )
            video._check_file()
            paths.append(video._local)
        copy = _codec.plan_video_copy(tuple(paths))
        source = iter(soundtrack(copy.frames))
        header = next(source, None)
        if (
            not isinstance(header, DecodedMediaHeader)
            or header.video is not None
            or not header.audio
        ):
            raise OutputError(
                "a joined soundtrack must begin with an audio header", code="media_header"
            )
        audio_format = header.audio

        def feed(encoder: _codec.StreamingMP4Encoder) -> int:
            audio_samples = 0
            pending: DecodedMediaEvent | None = next(source, None)

            def before(time: Fraction | None) -> None:
                """Write the soundtrack up to `time`; None writes the rest."""
                nonlocal audio_samples, pending
                while pending is not None and (
                    time is None
                    or not isinstance(pending, DecodedAudioChunk)  # refused when written
                    or pending.start_time < time
                ):
                    audio_samples = self._write_audio(encoder, audio_format, pending, audio_samples)
                    pending = next(source, None)

            encoder.copy_video(before)
            before(None)
            for video in videos:
                video._check_file()
            return audio_samples

        return self._save_mp4(
            dict(
                width=copy.width,
                height=copy.height,
                frame_rate=copy.frame_rate,
                pixel_aspect_ratio=copy.pixel_aspect_ratio,
                color_primaries=copy.color_primaries,
                color_transfer=copy.color_transfer,
                color_matrix=copy.color_matrix,
                color_range=copy.color_range,
            ),
            audio_format,
            feed,
            copy,
        )

    def join_video(self, audio: DecodedAudioFormat | None) -> VideoJoin:
        """A packet-copy join that grows one video at a time, for publishing as it grows.

        Each `append` copies one more H.264 video's packets (no picture is decoded or encoded)
        with its stretch of `audio`, the one soundtrack encoded across the join, and returns
        the video joined so far, publishable with `publish` as a normal indexed MP4 snapshot.
        Each revision copies encoded packets; internal fragments keep append work incremental.
        `last=True` or `finish` also flushes the soundtrack;
        `finish` returns the last revision byte for byte when the last append said `last=True`.
        """
        self._attempt.check_open("Outputs.join_video")
        _codec.require_encoder()
        return VideoJoin(self, audio)

    def _write_audio(
        self,
        encoder: _codec.StreamingMP4Encoder,
        audio_format: DecodedAudioFormat | None,
        event: DecodedMediaEvent,
        written: int,
    ) -> int:
        """One chunk on the declared format and sample clock; the samples now written."""
        planar = self._audio_planar(audio_format, event, written)
        encoder.write_audio(planar)
        return written + planar.shape[1]

    def _audio_planar(
        self, audio_format: DecodedAudioFormat | None, event: DecodedMediaEvent, written: int
    ) -> _codec.Planar:
        """One chunk checked against the declared format and continuing the sample clock."""
        if not isinstance(event, DecodedAudioChunk):
            raise OutputError(
                f"streaming video received {type(event).__name__}, not a media event",
                code="media_event",
            )
        if audio_format is None:
            raise OutputError("streaming video contains undeclared audio", code="audio_absent")
        if (
            event.channels != audio_format.channels
            or event.sample_rate != audio_format.sample_rate
            or event.channel_layout != audio_format.channel_layout
            or event.channel_names != audio_format.channel_names
            or event.time_base != audio_format.time_base
        ):
            raise OutputError("streaming audio chunk differs from its header", code="audio_format")
        expected = Fraction(written, audio_format.sample_rate)
        if event.start_time != expected:
            raise OutputError(
                f"streaming audio starts at {event.start_time}, expected {expected}",
                code="audio_discontinuity",
            )
        planar, clipped = _codec.audio_chunk(event.pcm_f32le, event.sample_count)
        self._clip_row(clipped)
        return planar

    def _save_mp4(
        self,
        video: dict[str, Any],
        audio_format: DecodedAudioFormat | None,
        feed: Callable[[_codec.StreamingMP4Encoder], int],
        copy: _codec.VideoCopy | None = None,
    ) -> SavedVideo:
        """Write, probe and atomically register one MP4; `feed` returns its audio samples."""
        handle = self._attempt.next_handle(VideoAsset.kind)
        tail = handle.rsplit("/", 1)[-1]
        temporary = self._attempt.spool / f".video-{tail}.partial"
        final = self._attempt.spool / f"video-{tail}"
        encoder: _codec.StreamingMP4Encoder | None = None
        finished = False
        registered = False
        try:
            encoder = _codec.StreamingMP4Encoder(
                temporary,
                **video,
                max_bytes=self._attempt.max_output_bytes,
                audio_channels=audio_format.channels if audio_format is not None else 0,
                audio_sample_rate=audio_format.sample_rate if audio_format is not None else 0,
                copy=copy,
            )
            audio_samples = feed(encoder)
            if audio_format is not None and audio_samples == 0:
                raise OutputError(
                    "streaming video declared audio but yielded no samples", code="audio_shape"
                )
            frame_count = encoder.video_frames
            facts = encoder.finish()
            finished = True
            if (
                facts.width != video["width"]
                or facts.height != video["height"]
                or facts.frame_count != frame_count
                or facts.frame_rate != video["frame_rate"]
                or facts.pixel_aspect_ratio != video["pixel_aspect_ratio"]
                or facts.color_primaries != video["color_primaries"]
                or facts.color_transfer != video["color_transfer"]
                or facts.color_matrix != video["color_matrix"]
                or facts.color_range != video["color_range"]
                or facts.video_codec != "h264"
                or (audio_format is None and facts.audio_codec)
                or (audio_format is not None and facts.audio_codec != "aac")
                or (
                    audio_format is not None
                    and (
                        facts.audio_channels != audio_format.channels
                        or facts.audio_sample_rate != audio_format.sample_rate
                        or facts.audio_frame_samples != self.video_audio_frame_samples
                    )
                )
                or facts.audio_decoded_samples != audio_samples
            ):
                raise OutputError(
                    "streaming MP4 probe disagrees with submitted media: "
                    f"submitted frames={frame_count} rate={video['frame_rate']} "
                    f"audio={audio_samples}; probed {facts}",
                    code="output_probe",
                )

            size, digest = self._measure_file(temporary)
            if self._pending_bytes() + size > self._attempt.max_output_bytes:
                raise OutputError(
                    "streaming output would cross the attempt aggregate output limit",
                    code="output_too_large",
                )
            os.link(temporary, final)
            self._fsync_directory(final.parent)
            temporary.unlink()
            asset = VideoAsset(
                handle,
                media_type="video/mp4",
                size_bytes=size,
                digest=digest,
                local=final,
                attempt=self._attempt.request_id,
            )
            self._attempt.pending[handle] = asset
            registered = True
            return SavedVideo(
                video=asset,
                width=facts.width,
                height=facts.height,
                frame_count=facts.frame_count,
                frame_rate=facts.frame_rate,
                pixel_aspect_ratio=facts.pixel_aspect_ratio,
                video_codec=facts.video_codec,
                video_profile=facts.video_profile,
                color_primaries=facts.color_primaries,
                color_transfer=facts.color_transfer,
                color_matrix=facts.color_matrix,
                color_range=facts.color_range,
                audio=(
                    SavedAudioFacts(
                        codec=facts.audio_codec,
                        profile=facts.audio_profile,
                        channels=facts.audio_channels,
                        sample_rate=facts.audio_sample_rate,
                        codec_frame_samples=facts.audio_frame_samples,
                        decoded_samples=facts.audio_decoded_samples,
                    )
                    if audio_format is not None
                    else None
                ),
                submitted_audio_samples=audio_samples,
            )
        except AuthorError:
            raise
        except Exception as exc:
            raise OutputError(
                f"streaming video encode failed: {type(exc).__name__}: {exc}",
                code="output_encode",
            ) from exc
        finally:
            if temporary.exists():
                if encoder is not None and not finished:
                    encoder.abort()
                else:
                    temporary.unlink(missing_ok=True)
            if not registered:
                final.unlink(missing_ok=True)

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

    def publish(self, output: str, asset: Asset, *, label: str = "") -> None:
        """Show `asset` to the run's owner now, as a product of `output`: an asset field of
        this function's declared result. A single field's product is replaced by each publish;
        a `list[...]` field grows by one. Returning the result is the last publish, so the
        returned value keeps what was published (a list extends it). `label` is for people:
        its first 256 characters, printable ASCII (others read "?"). A child call's publishes
        are not shown; its parent publishes what it wants seen."""
        attempt = self._attempt
        attempt.check_open("Outputs.publish")
        if not isinstance(asset, Asset):
            raise OutputError("publish takes one saved or received asset", code="output_kind")
        # The log carries printable ASCII: anything else reads as "?", never a refusal.
        label = "".join(char if 0x20 <= ord(char) <= 0x7E else "?" for char in label[:256])
        if asset.ref in attempt.frames:
            settle_frame(attempt, asset.ref)
        request = Publish(
            output=output,
            label=label,
            asset_ref=asset.ref,
            asset_kind=asset.kind,
            media_type=asset.media_type,
            size_bytes=asset.size_bytes,
            digest=asset.digest,
            parts=attempt.parts.get(asset.ref, ()),
        )
        if attempt.publish is None:
            attempt.published.append(request)
        else:
            answer = attempt.publish(request)
            if not answer.ok:
                raise OutputError(
                    answer.detail or "the worker refused the publish", code=answer.code
                )
        attempt.shown.add(asset.ref)

    def save_bytes(self, data: bytes, *, media_type: str) -> FileAsset:
        return self._register(data, FileAsset, media_type)

    async def commit(self, pending: FileAsset) -> FileAsset:
        """Complete an owned pending file now, for reading or forwarding to another call."""
        from cozy_runtime.author._output_commit import commit

        return await commit(self._attempt, pending)

    def save_tree(self, path: Path | str) -> Tree:
        """Snapshot regular files now; native retention commits after the handler returns."""
        self._attempt.check_open("Outputs.save_tree")
        source = Path(path)
        if source.is_symlink() or not source.is_dir():
            raise OutputError("output tree must be a regular directory", code="output_tree")
        handle = self._attempt.next_handle("tree")
        target = self._attempt.spool / f"tree-{handle.rsplit('/', 1)[-1]}"
        target.mkdir()
        available = self._attempt.max_output_bytes - self._pending_bytes()
        count, total = 0, 0
        try:
            for directory, dirs, files in os.walk(source, followlinks=False):
                current = Path(directory)
                if any((current / name).is_symlink() for name in dirs):
                    raise OutputError("output tree cannot contain symlinks", code="output_tree")
                for name in sorted(files):
                    count += 1
                    if count > 8192:
                        raise OutputError("output tree exceeds its file bound", code="output_tree")
                    original = current / name
                    fd = os.open(original, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, "rb") as reader:
                        if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
                            raise OutputError(
                                "output tree requires regular files", code="output_tree"
                            )
                        dest = target / original.relative_to(source)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with dest.open("xb") as writer:
                            while chunk := reader.read(min(1 << 20, available - total + 1)):
                                total += len(chunk)
                                if total > available:
                                    raise OutputError(
                                        "output tree exceeds its byte grant",
                                        code="output_too_large",
                                    )
                                writer.write(chunk)
            if not count:
                raise OutputError("output tree must contain at least one file", code="output_tree")
            tree = Tree(handle, root=target, attempt=self._attempt.request_id)
            tree.size_bytes = total
            self._attempt.pending_trees[handle] = tree
            return tree
        except Exception:
            shutil.rmtree(target)
            raise

    def save_file(
        self, path: Path | str, *, media_type: str = "application/octet-stream"
    ) -> FileAsset:
        """Copies immediately: a MUTABLE AUTHOR PATH is never retained (§3.4 step 2)."""
        return self._register(Path(path).read_bytes(), FileAsset, media_type)

    def temporary_file(self, suffix: str = "") -> Path:
        """A bounded attempt-spool target for encoders that need a filesystem path.

        Becomes an output only when a save converts it; otherwise it dies with the spool.
        No general scratch directory exists for packages.
        """
        self._attempt.check_open("Outputs.temporary_file")
        tmp = self._attempt.spool / "tmp"
        tmp.mkdir(exist_ok=True)
        return tmp / f"{self._attempt.next_handle('tmp').rsplit('/', 1)[-1]}{suffix}"

    def _register(self, data: bytes, kind: type[A], media_type: str) -> A:
        self._attempt.check_open("Outputs.save")
        if self._pending_bytes() + len(data) > self._attempt.max_output_bytes:
            raise OutputError(
                f"output would bring the attempt to {self._pending_bytes() + len(data)} bytes, "
                f"over its {self._attempt.max_output_bytes}-byte aggregate limit",
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

    def _register_frame(
        self,
        kind: type[A],
        codec: str,
        parts: tuple[Buffer, ...],
        facts: dict[str, Any],
        *,
        charge: int,
    ) -> A:
        """REGISTER a host frame: the raw snapshot lands in the spool, the aggregate ceiling
        is charged now, and the codec runs later — in the worker's post phase, or here
        when the attempt settles at the save. `parts` are written back to back: the spool
        write is the one host copy, never a concatenation before it."""
        self._attempt.check_open("Outputs.save")
        if self._pending_bytes() + charge > self._attempt.max_output_bytes:
            raise OutputError(
                f"output would bring the attempt to {self._pending_bytes() + charge} bytes, "
                f"over its {self._attempt.max_output_bytes}-byte aggregate limit",
                code="output_too_large",
            )
        handle = self._attempt.next_handle(kind.kind)
        path = self._attempt.spool / f"{kind.kind}-{handle.rsplit('/', 1)[-1]}.raw"
        raw_bytes = 0
        with path.open("wb") as spool:
            for part in parts:
                raw_bytes += spool.write(part)
        frame = HostFrame(handle, codec, path, raw_bytes, charge, facts)
        asset = kind(handle, media_type=frame.media_type, attempt=self._attempt.request_id)
        self._attempt.pending[handle] = asset
        self._attempt.frames[handle] = frame
        if self._attempt.settle_at_save:
            settle_frame(self._attempt, handle)
        return asset

    def _pending_bytes(self) -> int:
        return (
            sum(asset.size_bytes for asset in self._attempt.pending.values())
            + sum(tree.size_bytes for tree in self._attempt.pending_trees.values())
            + sum(frame.charge for frame in self._attempt.frames.values())
        )

    @staticmethod
    def _measure_file(path: Path) -> tuple[int, str]:
        digest = hashlib.blake2b(digest_size=16)
        length = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                length += len(chunk)
                digest.update(chunk)
        return length, "blake2b:" + digest.hexdigest()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name != "posix":
            return
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class VideoJoin:
    """`Outputs.join_video`: an indexed copy join published as it grows."""

    def __init__(self, outputs: Outputs, audio: DecodedAudioFormat | None) -> None:
        self._outputs, self._attempt, self._audio = outputs, outputs._attempt, audio
        self._written = 0
        directory = (
            self._attempt.spool / f"join-{self._attempt.next_handle('join').rsplit('/', 1)[-1]}"
        )
        directory.mkdir()
        self._join = _codec.FragmentedJoin(
            directory,
            audio_channels=audio.channels if audio is not None else 0,
            audio_sample_rate=audio.sample_rate if audio is not None else 0,
            max_bytes=self._attempt.max_output_bytes,
        )

    def append(
        self,
        video: VideoAsset,
        soundtrack: Callable[[int], Iterable[DecodedMediaEvent]] = lambda frames: (),
        *,
        last: bool = False,
    ) -> VideoAsset:
        """Join `video` after the others with its stretch of the soundtrack: the audio chunks
        `soundtrack` returns given the video's frame count, continuing the joined sample
        clock. Each returned revision has a duration and seek index. `last` also closes
        the soundtrack; neither publishing nor closing re-encodes the joined media."""
        self._attempt.check_open("VideoJoin.append")
        if (
            not isinstance(video, VideoAsset)
            or video._local is None
            or video._attempt != self._attempt.request_id
        ):
            raise CapabilityError(
                "joined videos must be hydrated inputs of this attempt", code="foreign_asset"
            )
        video._check_file()

        def planars(frames: int) -> Iterator[_codec.Planar]:
            for event in soundtrack(frames):
                planar = self._outputs._audio_planar(self._audio, event, self._written)
                self._written += planar.shape[1]
                yield planar

        self._join.append(video._local, planars, last=last)
        video._check_file()
        # Its bytes live on in the joined video: a saved segment need not be returned.
        self._attempt.shown.add(video.ref)
        return self._revision()

    def finish(self) -> SavedVideo:
        """The joined video, registered as this attempt's output; its parts are the last
        revision's, so publishing it again costs nothing."""
        self._attempt.check_open("VideoJoin.finish")
        if not self._join.parts:
            raise OutputError("a joined video needs at least one video", code="media_header")
        self._join.finish()
        handle = self._attempt.next_handle(VideoAsset.kind)
        final = self._attempt.spool / f"video-{handle.rsplit('/', 1)[-1]}"
        with final.open("xb") as writer:
            for part, _ in self._join.parts:
                with part.open("rb") as reader:
                    shutil.copyfileobj(reader, writer, 1 << 20)
            writer.flush()
            os.fsync(writer.fileno())
        facts = _codec.probe_mp4(final)
        if facts.frame_count != self._join.frames or (
            self._audio is not None and facts.audio_decoded_samples != self._written
        ):
            final.unlink()
            raise OutputError(
                f"joined MP4 probe disagrees: {facts.frame_count} frames of {self._join.frames}, "
                f"{facts.audio_decoded_samples} samples of {self._written}",
                code="output_probe",
            )
        size, digest = self._outputs._measure_file(final)
        asset = VideoAsset(
            handle,
            media_type="video/mp4",
            size_bytes=size,
            digest=digest,
            local=final,
            attempt=self._attempt.request_id,
        )
        self._attempt.pending[handle] = asset
        self._attempt.parts[handle] = self._parts()
        return SavedVideo(
            video=asset,
            width=facts.width,
            height=facts.height,
            frame_count=facts.frame_count,
            frame_rate=facts.frame_rate,
            pixel_aspect_ratio=facts.pixel_aspect_ratio,
            video_codec=facts.video_codec,
            video_profile=facts.video_profile,
            color_primaries=facts.color_primaries,
            color_transfer=facts.color_transfer,
            color_matrix=facts.color_matrix,
            color_range=facts.color_range,
            audio=(
                SavedAudioFacts(
                    codec=facts.audio_codec,
                    profile=facts.audio_profile,
                    channels=facts.audio_channels,
                    sample_rate=facts.audio_sample_rate,
                    codec_frame_samples=facts.audio_frame_samples,
                    decoded_samples=facts.audio_decoded_samples,
                )
                if self._audio is not None
                else None
            ),
            submitted_audio_samples=self._written,
        )

    def _parts(self) -> tuple[PublishPart, ...]:
        return tuple(
            PublishPart(local=f"{part.parent.name}/{part.name}", duration_us=micros)
            for part, micros in self._join.parts
        )

    def _revision(self) -> VideoAsset:
        """Publish one indexed byte identity for desktop and browser playback."""
        path = self._join.indexed_revision()
        handle = self._attempt.next_handle("revision")
        revision = VideoAsset(
            handle,
            media_type="video/mp4",
            size_bytes=path.stat().st_size,
            attempt=self._attempt.request_id,
        )
        self._attempt.parts[handle] = (
            PublishPart(
                local=f"{path.parent.name}/{path.name}",
                duration_us=sum(duration for _, duration in self._join.parts),
            ),
        )
        return revision


class Telemetry(_Bound):
    """User-facing status and operator diagnostics (`tel`). Observability only: it never
    changes availability, placement, or the ComponentUseContract."""

    __slots__ = ("_ctx",)

    def __init__(self, attempt: Attempt, ctx: Context) -> None:
        super().__init__(attempt)
        self._ctx = ctx

    def progress(
        self,
        stage_fraction: float,
        *,
        stage: str | None = None,
        overall_fraction: float | None = None,
    ) -> None:
        attempt = self._attempt
        attempt.check_open("Telemetry.progress")
        self._ctx.raise_if_cancelled()
        name = self._stage_name(stage)
        if name is None:
            return
        fraction = self._fraction(stage_fraction, "stage_fraction")
        overall = self._overall(overall_fraction)
        if fraction is not None or overall is not None:
            self._progress_frame(
                name,
                stage_fraction=fraction,
                overall_fraction=overall,
            )

    def step_callback(
        self,
        total: int,
        *,
        stage: str | None = None,
        overall_range: tuple[float, float] | None = None,
    ) -> Any:
        """Return one counted-work hook carrying stage and optional overall coordinates.

        It is also the STEP-TIME instrument: the interval between consecutive calls is
        accumulated in place (five floats per stage), so inference step time is attributable
        without one retained row per step. The hot-path cost is one `perf_counter` and a
        handful of float operations.
        """
        attempt = self._attempt
        attempt.check_open("Telemetry.step_callback")
        name = self._stage_name(stage or "step")
        bounds = self._overall_range(overall_range)
        valid_total = (
            isinstance(total, int) and not isinstance(total, bool) and 0 < total <= 1_000_000_000
        )
        if not valid_total:
            self._diagnose("total must be an integer in 1..1000000000")
        # The mark starts HERE, at callback construction, so the FIRST step has an interval
        # too. Starting it at the first call would silently drop step 0 from the count — and
        # on a one-step request that is the whole measurement.
        attempt._step_mark = time.perf_counter()
        capture = _capture_clock.get()
        if capture is not None:
            capture.schedule(total)

        def on_step(step: int) -> None:
            self._ctx.raise_if_cancelled()
            now = time.perf_counter()
            mark = attempt._step_mark
            if mark and name is not None:
                attempt.attribution.step(name, (now - mark) * 1000)
            attempt._step_mark = now
            step_ms = (now - mark) * 1000 if mark else 0.0
            if (
                name is None
                or not valid_total
                or not isinstance(step, int)
                or isinstance(step, bool)
                or not 0 <= step < total
            ):
                if valid_total and (not isinstance(step, int) or isinstance(step, bool)):
                    self._diagnose("step must be an integer")
                elif valid_total and isinstance(step, int) and not 0 <= step < total:
                    self._diagnose("step must be within the declared total")
                return
            if capture is not None:
                capture.completed(step)
            raw_fraction = (step + 1) / total
            fraction = round(raw_fraction, 6)
            overall = None
            if bounds is not None:
                overall = self._overall(bounds[0] + raw_fraction * (bounds[1] - bounds[0]))
            self._progress_frame(
                name,
                stage_fraction=fraction,
                position=step + 1,
                total=total,
                overall_fraction=overall,
                step_ms=step_ms,
            )

        return on_step

    def log(self, message: str, *, level: str = "info", **fields: Scalar) -> None:
        self._attempt.check_open("Telemetry.log")
        self._attempt.emit("log", message, level, **fields)

    @contextmanager
    def scope(
        self, name: str, *, overall_range: tuple[float, float] | None = None
    ) -> Iterator[None]:
        """Compose local or managed child progress inside a named work range.

        Ranges describe ordered planned work, not elapsed time. Concurrent operations
        should leave the range unknown. Ordinary timing stages keep their semantics.
        """
        self._attempt.check_open("Telemetry.scope")
        self._ctx.raise_if_cancelled()
        stage = self._stage_name(name)
        bounds = self._overall_range(overall_range)
        if stage is None:
            yield
            return
        parent = _progress_scope.get()
        if parent is not None and parent.telemetry._attempt is self._attempt:
            stage = f"{parent.name} / {stage}"[:120]
            if bounds is not None and parent.bounds is not None:
                start, end = parent.bounds
                bounds = (start + bounds[0] * (end - start), start + bounds[1] * (end - start))
            elif parent.bounds is None:
                bounds = None
        self._progress_frame(
            stage, overall_fraction=self._overall(bounds[0], scoped=False) if bounds else None
        )
        token = _progress_scope.set(_ProgressScope(self, stage, bounds))
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            _progress_scope.reset(token)
            if succeeded and bounds is not None:
                self._progress_frame(stage, overall_fraction=self._overall(bounds[1], scoped=False))

    def _child_progress(self, payload: Mapping[str, Any], scope: _ProgressScope) -> None:
        """Relay measured child coordinates without making child completion parent completion."""
        self._attempt.check_open("Telemetry.child_progress")
        stage = self._stage_name(payload.get("stage"))
        if stage is None:
            return
        fraction = (
            self._fraction(payload["stage_fraction"], "stage_fraction")
            if "stage_fraction" in payload
            else None
        )
        overall = None
        if scope.bounds is not None and "overall_fraction" in payload:
            child = self._fraction(payload["overall_fraction"], "overall_fraction")
            if child is not None:
                start, end = scope.bounds
                overall = self._overall(start + child * (end - start), scoped=False)
        position, total = payload.get("position"), payload.get("total")
        if not (
            type(position) is int
            and type(total) is int
            and 0 <= position <= total <= 1_000_000_000
            and total > 0
        ):
            position = total = None
        step_ms = payload.get("step_ms", 0.0)
        try:
            valid_timing = (
                not isinstance(step_ms, bool)
                and isinstance(step_ms, (int, float))
                and math.isfinite(step_ms)
                and step_ms >= 0
            )
        except OverflowError:
            valid_timing = False
        if not valid_timing:
            step_ms = 0.0
        call_request, call_attempt = payload.get("call_request"), payload.get("call_attempt")
        if not isinstance(call_request, str) or not 0 < len(call_request) <= 255:
            call_request = None
        if type(call_attempt) is not int or call_attempt <= 0:
            call_attempt = None
        self._progress_frame(
            f"{scope.name} / {stage}"[:120],
            stage_fraction=fraction,
            position=position,
            total=total,
            overall_fraction=overall,
            step_ms=step_ms,
            call_request=call_request,
            call_attempt=call_attempt,
            scoped=False,
        )

    @contextmanager
    def stage(
        self, name: str, *, overall_range: tuple[float, float] | None = None
    ) -> Iterator[None]:
        """Per-stage timing bracket; `stage_ms` feeds the duration model and benchmarks.

        Structurally separate from a component-use scope: this records a NUMBER and touches
        no residency, no placement and no ComponentUseContract — `checks/architecture.py`'s
        `timing-decides-nothing` fence proves the module cannot reach them.
        """
        self._attempt.check_open("Telemetry.stage")
        self._ctx.raise_if_cancelled()
        stage = self._stage_name(name)
        bounds = self._overall_range(overall_range)
        if stage is not None:
            self._progress_frame(
                stage,
                overall_fraction=self._overall(bounds[0]) if bounds is not None else None,
            )
        start = time.perf_counter()
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            if stage is not None:
                self._attempt.attribution.stage(stage, elapsed)
                self._attempt.emit("stage", stage, round(elapsed, 3))
                if succeeded and bounds is not None:
                    self._progress_frame(stage, overall_fraction=self._overall(bounds[1]))

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

    def _progress_frame(
        self,
        stage: str | None,
        *,
        stage_fraction: float | None = None,
        position: int | None = None,
        total: int | None = None,
        overall_fraction: float | None = None,
        step_ms: float = 0.0,
        call_request: str | None = None,
        call_attempt: int | None = None,
        scoped: bool = True,
    ) -> None:
        name = self._stage_name(stage)
        if name is None:
            return
        scope = _progress_scope.get() if scoped else None
        if (
            scope is not None
            and scope.telemetry._attempt is self._attempt
            and name != scope.name
            and not name.startswith(scope.name + " / ")
        ):
            name = f"{scope.name} / {name}"[:120]
        attempt = self._attempt
        attempt.signal(
            ProgressFrame(
                stage=name,
                stage_fraction=stage_fraction,
                advance=attempt.position + 1,
                position=position,
                total=total,
                overall_fraction=overall_fraction,
                step_ms=step_ms,
                call_request=call_request,
                call_attempt=call_attempt,
            )
        )

    def _stage_name(self, value: object) -> str | None:
        if not isinstance(value, str) or not value or len(value) > 120 or not value.isprintable():
            self._diagnose("stage must be 1..120 printable characters")
            return None
        return value

    def _fraction(self, value: object, field: str) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._diagnose(f"{field} must be a finite number in 0..1")
            return None
        try:
            number = float(value)
        except (OverflowError, ValueError):
            self._diagnose(f"{field} must be a finite number in 0..1")
            return None
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            self._diagnose(f"{field} must be a finite number in 0..1")
            return None
        return round(number, 6)

    def _overall(self, value: object | None, *, scoped: bool = True) -> float | None:
        if value is None:
            return None
        overall = self._fraction(value, "overall_fraction")
        if overall is None:
            return None
        scope = _progress_scope.get() if scoped else None
        if scope is not None and scope.telemetry._attempt is self._attempt:
            if scope.bounds is None:
                return None
            start, end = scope.bounds
            overall = round(start + overall * (end - start), 6)
        previous = self._attempt._overall_fraction
        if previous is not None and overall < previous:
            self._diagnose("overall_fraction must not move backward")
            return None
        self._attempt._overall_fraction = overall
        return overall

    def _overall_range(self, value: object | None) -> tuple[float, float] | None:
        if value is None:
            return None
        if not isinstance(value, tuple) or len(value) != 2:
            self._diagnose("overall_range must be a (start, end) pair")
            return None
        start = self._fraction(value[0], "overall_range start")
        end = self._fraction(value[1], "overall_range end")
        if start is None or end is None:
            return None
        if end < start:
            self._diagnose("overall_range end must not precede its start")
            return None
        return start, end

    def _diagnose(self, reason: str) -> None:
        attempt = self._attempt
        if reason in attempt._progress_diagnostics:
            return
        attempt._progress_diagnostics.add(reason)
        attempt.emit("log", "progress omitted", "warn", reason=reason)


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
    """Worker-brokered outbound HTTP (`net`) — connector endpoints only (§1.3).

    NOT GRANTABLE YET, and `UNGRANTABLE` is where that is stated: a package declaring
    `net: Egress` refuses at DESCRIBE rather than passing every gate and raising on its
    first real request. The type, the injection and the capability derivation stand so that
    cr-012 lands the broker and the control-plane allowlist together and nothing else has
    to move. `fakes.fake_egress` drives this class directly, which is the only way an
    `Egress` is reachable today; the refusal below is its fail-closed floor.
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
                f"no egress broker is installed for {method} {url}: the worker-side "
                "broker, the control-plane host allowlist and the SSRF fail-closed path "
                "are cr-012, and a package cannot declare `net` until they land",
                code="egress_broker_unavailable",
            )
        result = self._broker(url, method, dict(headers or {}), body)
        assert isinstance(result, EgressResponse)
        return result


class Settings[T]:
    """Typed deployment settings. The author declares the strict schema; the VALUES are
    owned by the deployment config plane, validated against it and digest-pinned into the
    execution spec. No ambient environment variable ever carries configuration (§3.5)."""

    __slots__ = ("value",)

    def __init__(self, value: T) -> None:
        self.value = value


class Secrets[S]:
    """Granted secrets against a strict named schema — a bare unbounded bag is invalid.
    Never raw getenv; injectable only when the capability is declared."""

    __slots__ = ("value",)

    def __init__(self, value: S) -> None:
        self.value = value


# ----------------------------------------------------------------------- job services


@dataclass(frozen=True, slots=True)
class CheckpointDeclaration:
    """One attempt DECLARED a checkpoint, and the declaration is recorded.

    NOT A DURABILITY RECEIPT, and the name is the fix (#553a). The type this replaces was
    called `CheckpointReceipt` and carried a `receipt_id`, which is what a caller reads when
    it decides the bytes survive an executor kill. Nothing stores the bytes: the worker
    keeps a process-local metadata row and the protocol's `JobCheckpointRequest.artifact`
    field goes out empty, so the only true fact in the exchange is that an attempt said it
    had a checkpoint, with this digest, at this length. That fact is what this carries.

    `replayed` says the identity was already recorded and this call added nothing.
    """

    declaration_id: str
    operation_key: str
    logical_key: str
    content_digest: str
    length: int
    replayed: bool = False


class CheckpointConflict(CapabilityError):
    """One logical/operation key, two different contents. Never a replacement (§2)."""

    default_code = "checkpoint_conflict"


class CheckpointSinkUnbuilt(CapabilityError):
    """`Checkpoints.save()` is HARD-DISABLED: checkpoint use of TensorFS derivation is absent.

    The dependency is named because it is the whole answer — #552 adopted ONE ARTIFACT
    TRANSACTION, in which Runtime's TensorFS derivation writes TensorFS objects and returns a
    durable receipt. Model outputs now have that transaction; checkpoints do not yet have
    the separate immutable-operation identity and lifecycle needed to reuse it. `declare()`
    is what remains true in the meantime.
    """

    default_code = "checkpoint_sink_unbuilt"


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
    """Job service (§2) — checkpoint DECLARATIONS. `save()` is hard-disabled (#553a).

    What this service could honestly do and what it CLAIMED to do had come apart. The claim
    was that `save()` made bytes durable and returned only once they were. The mechanism was
    a metadata row: the worker records (run, attempt, operation key, logical key, content
    digest, length) in process memory and the RecordOwner stores a copy, while the
    protocol's `JobCheckpointRequest.artifact` field — the only place bytes could ride — goes
    out empty and no component anywhere copies or stores them. A killed executor took every
    checkpoint with it, and the receipt said otherwise.

    So the identity, the idempotency and the CONFLICT rule are all real and all kept — they
    are properties of the declaration, and repeating one adds nothing while the same keys
    with different bytes stay a conflict, never a replacement. The DURABILITY is the part
    that was not real, and `save()` now refuses `checkpoint_sink_unbuilt` naming the
    checkpoint integration with TensorFS derivation #552. When that integration exists, `save()`
    gets a body again and the acceptance test is the one #553a names: kill executor and
    worker, recover BYTES by receipt, never a process record.
    """

    __slots__ = ("_sink",)

    def __init__(
        self,
        attempt: Attempt,
        sink: Callable[[CheckpointSave], CheckpointDeclaration] | None = None,
    ) -> None:
        super().__init__(attempt)
        self._sink = sink

    def declare(
        self, logical_key: str, data: bytes | Path, *, operation_key: str = ""
    ) -> CheckpointDeclaration:
        """Record that this attempt HAS a checkpoint: its keys, digest and length.

        Everything this returns is true. The bytes stay exactly where the caller put them —
        the attempt spool, or the run's `Scratch.checkpoint_dir(key=)` tree, which is what
        actually survives into the next attempt today.
        """
        self._attempt.check_open("Checkpoints.declare")
        if self._sink is None:
            raise CapabilityError(
                "no durable checkpoint exchange is installed for this attempt: Checkpoints is "
                "a JOB service and its exchange is the worker's, never a local file write",
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
                # so the declaration the RecordOwner holds is addressable by the same digest.
                content_digest="sha256:" + hashlib.sha256(raw).hexdigest(),
                length=len(raw),
                path=blob,
            )
        )

    def save(self, logical_key: str, data: bytes | Path, *, operation_key: str = "") -> NoReturn:
        """HARD-DISABLED. Refuses `checkpoint_sink_unbuilt` — see the class docstring.

        `NoReturn` is the point: there is no value this can hand back, so no caller can be
        holding a receipt for bytes nobody stored. `declare()` is the honest half.
        """
        del data, operation_key
        self._attempt.check_open("Checkpoints.save")
        raise CheckpointSinkUnbuilt(
            f"{logical_key or '<unkeyed>'}: Checkpoints.save() is disabled because the "
            "checkpoint-to-TensorFS derivation lifecycle is unbuilt — model outputs have a durable "
            "transaction, but checkpoint identity/adoption is still absent. Use "
            "Checkpoints.declare() for the fact that IS true (an attempt declared this "
            "checkpoint), and Scratch.checkpoint_dir(key=) for bytes a retried attempt of "
            "the same run can read back."
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
    "media_decode": MediaDecoder,
    "weights_read": WeightsReader,
}

#: The one service set a JOB may request and a serving entrypoint may not (§2). A job runs
#: to completion and owns durable working state; a request handler owns neither.
JOB_ONLY: frozenset[str] = frozenset(
    {"scratch", "checkpoints", "budget", "weights", "weights_read"}
)

#: The mirror: SERVING-ONLY surface. `adjust` writes the caller-visible adjustments envelope,
#: which is a fact about serving one request; a run-to-completion job has no caller to
#: confess to and no envelope to write into.
SERVING_ONLY: frozenset[str] = frozenset({"adjust"})

#: capability name -> why NOTHING can grant it yet. A capability with no grant path refuses
#: at BUILD, naming what has to land, rather than passing describe and deploy and failing on
#: the first real request — which is the worst clock to learn it on.
UNGRANTABLE: Mapping[str, str] = {
    "egress": "no deployment can grant outbound HTTP yet. The broker needs a "
    "CONTROL-PLANE-MINTED host allowlist, and there is none: cr-042 removed "
    "`DeclaredBinding.egress_allowed_hosts` because it never had a writer and its "
    "per-binding shape was unioned across deployments, which decision #451 forbids. "
    "Deriving the allowlist from the URL instead would grant whatever the package asked "
    "for, since a `net:` URL comes from package code — every other destination the worker "
    "opens is control-plane-minted and passes the CLOSED authorization table in "
    "`internal/worker/grants.py`. cr-012 lands the allowlist writer, the grant row and the "
    "broker together; until then this declaration is refused here rather than at the first "
    "request that needs it",
}
