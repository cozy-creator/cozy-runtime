"""The one bounded input-media decoder: verified bytes in, bounded media values out.

There is deliberately no path, URL, fetch, resize, model-normalization, or output-codec
surface here. PyAV is a lazy member of the ``media`` extra; the base author package does
not import it.
"""

from __future__ import annotations

import io
import math
import struct
from array import array
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Literal, Protocol, TypeVar

from PIL.Image import Transpose
from PIL.Image import frombytes as pil_frombytes
from PIL.Image import open as pil_open

from cozy_runtime.author import _color
from cozy_runtime.author._assets import Asset, AudioAsset, Image, ImageAsset, VideoAsset
from cozy_runtime.author._errors import AuthorError, CapabilityError, InvalidRequest, RuntimeFailure
from cozy_runtime.author._markers import ImagePreparation


class _Attempt(Protocol):
    request_id: str

    def check_open(self, what: str) -> None: ...


@dataclass(frozen=True, slots=True)
class DecodeLimits:
    """Runtime codec ceilings. A field's retained-value bound can only lower these.

    ``max_decoded_bytes`` accounts for the retained RGB8/float32 values returned to the
    handler. It is not a claim about peak process RSS: encoded input, codec working memory,
    and transient conversion buffers remain implementation memory.
    """

    max_decoded_bytes: int
    max_pixels_per_frame: int
    max_video_frames: int
    max_audio_channels: int
    max_audio_samples: int

    def __post_init__(self) -> None:
        if (
            min(
                self.max_decoded_bytes,
                self.max_pixels_per_frame,
                self.max_video_frames,
                self.max_audio_channels,
                self.max_audio_samples,
            )
            <= 0
        ):
            raise ValueError("every decode limit must be positive")


# Structural codec ceilings, not package policy. Duration and reference-count limits stay
# at the package; a field's mandatory max_decoded_bytes is normally the tighter bound.
DEFAULT_DECODE_LIMITS = DecodeLimits(
    max_decoded_bytes=2 * 1024**3,
    max_pixels_per_frame=8192 * 8192,
    max_video_frames=3600,
    max_audio_channels=8,
    max_audio_samples=48_000 * 60 * 10,
)

MAX_ACTIVE_MEDIA_STREAMS = 2


class DecoderUnavailable(CapabilityError):
    default_code = "decoder_unavailable"


class MediaDecodeError(InvalidRequest):
    default_code = "media_decode"


class MediaMaterializationError(RuntimeFailure):
    default_code = "media_materialization"


@dataclass(frozen=True, slots=True)
class DecodedImage:
    width: int
    height: int
    rgb: bytes


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """Planar native-clock PCM on the container's presentation timeline.

    ``start_time`` and a containing :class:`DecodedVideo`'s ``start_time`` share that
    timeline, so a package can preserve their exact A/V offset instead of assuming both
    streams begin at zero.
    """

    channels: int
    sample_count: int
    sample_rate: int
    channel_layout: str
    channel_names: tuple[str, ...]
    pcm_f32le: tuple[bytes, ...]
    start_time: Fraction

    @property
    def duration(self) -> Fraction:
        return Fraction(self.sample_count, self.sample_rate)


@dataclass(frozen=True, slots=True)
class DecodedVideo:
    """Display-oriented RGB8 frames, exact presentation clock, and optional soundtrack."""

    width: int
    height: int
    frame_count: int
    frames_rgb: tuple[bytes, ...]
    frame_pts: tuple[int, ...]
    frame_durations: tuple[int, ...]
    time_base: Fraction
    pixel_aspect_ratio: Fraction
    soundtrack: DecodedAudio | None

    @property
    def start_time(self) -> Fraction:
        return self.frame_pts[0] * self.time_base

    @property
    def duration(self) -> Fraction:
        return (self.frame_pts[-1] + self.frame_durations[-1] - self.frame_pts[0]) * self.time_base


Mixed = Image | DecodedVideo | DecodedAudio
DECODED_ASSET_TYPES = {Image: ImageAsset, DecodedVideo: VideoAsset, DecodedAudio: AudioAsset}


@dataclass(frozen=True, slots=True)
class DecodedVideoFrame:
    """One display-oriented RGB8 frame from a single-consumer media stream."""

    width: int
    height: int
    rgb: bytes
    pts: int
    duration: int
    time_base: Fraction
    pixel_aspect_ratio: Fraction
    color_primaries: int
    color_transfer: int
    color_matrix: int
    color_range: int

    @property
    def start_time(self) -> Fraction:
        return self.pts * self.time_base


@dataclass(frozen=True, slots=True)
class DecodedAudioChunk:
    """One planar float32 PCM chunk on the source presentation timeline."""

    channels: int
    sample_count: int
    sample_rate: int
    channel_layout: str
    channel_names: tuple[str, ...]
    pcm_f32le: tuple[bytes, ...]
    pts: int
    time_base: Fraction

    @property
    def start_time(self) -> Fraction:
        return self.pts * self.time_base

    @property
    def duration(self) -> Fraction:
        return Fraction(self.sample_count, self.sample_rate)


@dataclass(frozen=True, slots=True)
class DecodedVideoFormat:
    width: int
    height: int
    time_base: Fraction
    pixel_aspect_ratio: Fraction
    nominal_frame_rate: Fraction | None
    color_primaries: int
    color_transfer: int
    color_matrix: int
    color_range: int


@dataclass(frozen=True, slots=True)
class DecodedAudioFormat:
    channels: int
    sample_rate: int
    channel_layout: str
    channel_names: tuple[str, ...]
    time_base: Fraction


@dataclass(frozen=True, slots=True)
class DecodedMediaHeader:
    """The one actual decoded format declaration, always the first stream event."""

    video: DecodedVideoFormat | None
    audio: DecodedAudioFormat | None


DecodedMediaEvent = DecodedMediaHeader | DecodedVideoFrame | DecodedAudioChunk


@dataclass(slots=True)
class _Budget:
    limit: int
    parent: _Budget | None = None
    used: int = 0
    scope: str = "retained decoded"
    cumulative: bool = True

    def claim(self, size: int, what: str) -> None:
        next_used = self.used + size if self.cumulative else max(self.used, size)
        if size < 0 or next_used > self.limit:
            raise MediaDecodeError(
                f"{self.scope} {what} would cross the {self.limit}-byte value limit",
                code="decoded_bytes_limit",
            )
        if self.parent is not None:
            self.parent.claim(size, "attempt media")
        self.used = next_used


class _AudioAccumulator:
    """The snapshot adapter over the one incremental audio decoder."""

    def __init__(self, av: Any, stream: Any, limits: DecodeLimits, budget: _Budget) -> None:
        self._decoder = _AudioStreamDecoder(av, stream, limits, budget)
        self._channels: list[io.BytesIO] = []

    def append(self, frame: Any) -> None:
        event = self._decoder.decode(frame)
        if not self._channels:
            self._channels = [io.BytesIO() for _ in range(event.channels)]
        for channel, chunk in zip(self._channels, event.pcm_f32le, strict=True):
            channel.write(chunk)

    def finish(self) -> DecodedAudio:
        self._decoder.finish()
        assert self._decoder.start_time is not None
        return DecodedAudio(
            channels=self._decoder.channels,
            sample_count=self._decoder.sample_count,
            sample_rate=self._decoder.sample_rate,
            channel_layout=self._decoder.channel_layout,
            channel_names=self._decoder.channel_names,
            pcm_f32le=tuple(channel.getvalue() for channel in self._channels),
            start_time=self._decoder.start_time,
        )


class _VideoAccumulator:
    def __init__(self, av: Any, stream: Any, limits: DecodeLimits, budget: _Budget) -> None:
        self._decoder = _VideoStreamDecoder(av, stream, limits, budget)
        self._chunks: list[bytes] = []
        self.frame_pts: list[int] = []
        self.frame_durations: list[int] = []

    def append(self, frame: Any) -> None:
        event = self._decoder.decode(frame)
        if event is None:
            return
        self._chunks.append(event.rgb)
        self.frame_pts.append(event.pts)
        self.frame_durations.append(event.duration)

    def finish(self, soundtrack: DecodedAudio | None) -> DecodedVideo:
        event = self._decoder.finish()
        self._chunks.append(event.rgb)
        self.frame_pts.append(event.pts)
        self.frame_durations.append(event.duration)
        assert self._decoder.time_base is not None
        return DecodedVideo(
            width=self._decoder.width,
            height=self._decoder.height,
            frame_count=len(self._chunks),
            frames_rgb=tuple(self._chunks),
            frame_pts=tuple(self.frame_pts),
            frame_durations=tuple(self.frame_durations),
            time_base=self._decoder.time_base,
            pixel_aspect_ratio=self._decoder.pixel_aspect_ratio,
            soundtrack=soundtrack,
        )


class _AudioStreamDecoder:
    """Validate one native-clock audio stream while retaining at most one decoded frame."""

    def __init__(self, av: Any, stream: Any, limits: DecodeLimits, budget: _Budget) -> None:
        self._av = av
        self._limits = limits
        self._budget = budget
        self._resampler: Any | None = None
        self._last_presentation_time: Fraction | None = None
        self._time_base: Fraction | None = None
        self._start_time: Fraction | None = None
        self.channels = 0
        self.sample_rate = 0
        self.channel_layout = ""
        self.channel_names: tuple[str, ...] = ()
        self.sample_count = 0

        context = stream.codec_context
        context.thread_count = 1
        _set_decoder_option(context, "err_detect", "explode")
        _set_decoder_option(context, "max_samples", limits.max_audio_samples)
        header_channels = int(context.channels or 0)
        if header_channels > limits.max_audio_channels:
            raise MediaDecodeError(
                f"audio has {header_channels} channels; limit is {limits.max_audio_channels}",
                code="audio_channels_limit",
            )

    @property
    def start_time(self) -> Fraction | None:
        return self._start_time

    def _lock_format(self, frame: Any) -> None:
        self.channels = len(frame.layout.channels)
        self.sample_rate = int(frame.sample_rate or 0)
        self.channel_layout = str(frame.layout.name)
        self.channel_names = tuple(str(channel.name) for channel in frame.layout.channels)
        if self.channels <= 0 or self.channels > self._limits.max_audio_channels:
            raise MediaDecodeError(
                f"audio has {self.channels} channels; limit is {self._limits.max_audio_channels}",
                code="audio_channels_limit",
            )
        if (
            self.sample_rate <= 0
            or not self.channel_layout
            or len(self.channel_names) != self.channels
        ):
            raise MediaDecodeError(
                "audio frame has no valid native rate/layout", code="audio_format"
            )
        self._resampler = self._av.AudioResampler(
            format="fltp", layout=self.channel_layout, rate=self.sample_rate
        )

    def decode(self, frame: Any) -> DecodedAudioChunk:
        if frame.is_corrupt:
            raise MediaDecodeError("audio decoder marked a frame corrupt", code="media_corrupt")
        if self._resampler is None:
            self._lock_format(frame)
        samples = int(frame.samples or 0)
        channels = len(frame.layout.channels)
        rate = int(frame.sample_rate or 0)
        layout = str(frame.layout.name)
        names = tuple(str(channel.name) for channel in frame.layout.channels)
        if samples <= 0:
            raise MediaDecodeError("audio contains an empty frame", code="media_empty")
        if (
            channels != self.channels
            or rate != self.sample_rate
            or layout != self.channel_layout
            or names != self.channel_names
        ):
            raise MediaDecodeError(
                "audio format changes between decoded frames", code="audio_format_change"
            )

        valid_samples = _valid_audio_samples(frame, samples, self.sample_rate)
        if frame.pts is None or frame.time_base is None:
            raise MediaDecodeError("audio frame has no exact presentation time", code="media_clock")
        time_base = _positive_fraction(frame.time_base, "audio frame time base")
        presentation_time = _fraction(frame.pts, "audio frame PTS") * time_base
        if self._time_base is None:
            self._time_base = time_base
        elif time_base != self._time_base:
            raise MediaDecodeError("audio time base changes between frames", code="media_clock")
        if (
            self._last_presentation_time is not None
            and presentation_time <= self._last_presentation_time
        ):
            raise MediaDecodeError("audio PTS is not strictly monotone", code="media_clock")
        if self._start_time is None:
            self._start_time = presentation_time
        else:
            expected = self._start_time + Fraction(self.sample_count, self.sample_rate)
            if abs(presentation_time - expected) > time_base / 2:
                raise MediaDecodeError(
                    f"audio presentation is discontinuous: expected {expected}, got "
                    f"{presentation_time}",
                    code="audio_discontinuity",
                )
        self._last_presentation_time = presentation_time
        if self.sample_count + valid_samples > self._limits.max_audio_samples:
            raise MediaDecodeError(
                f"audio crosses the {self._limits.max_audio_samples}-sample limit",
                code="audio_samples_limit",
            )
        self._budget.claim(valid_samples * self.channels * 4, "audio PCM")

        assert self._resampler is not None
        converted = self._resampler.resample(frame)
        converted = list(converted) if converted is not None else []
        if sum(int(out.samples) for out in converted) != samples:
            raise MediaDecodeError(
                "native-rate audio conversion changed the sample clock", code="audio_format"
            )
        chunks = [bytearray() for _ in range(self.channels)]
        remaining = valid_samples
        for out in converted:
            if str(out.format.name) != "fltp" or len(out.planes) != self.channels:
                raise MediaDecodeError(
                    "audio conversion returned a non-planar float frame", code="audio_format"
                )
            take = min(remaining, int(out.samples))
            if take:
                for channel, plane in enumerate(out.planes):
                    raw = memoryview(plane)
                    size = take * 4
                    if len(raw) < size:
                        raise MediaDecodeError(
                            "audio plane is shorter than its sample count", code="media_geometry"
                        )
                    chunk = bytes(raw[:size])
                    _require_finite_f32(chunk)
                    chunks[channel] += chunk
                remaining -= take
        if remaining:
            raise MediaDecodeError(
                "audio conversion returned fewer samples than its clock", code="media_truncated"
            )

        self.sample_count += valid_samples
        return DecodedAudioChunk(
            channels=self.channels,
            sample_count=valid_samples,
            sample_rate=self.sample_rate,
            channel_layout=self.channel_layout,
            channel_names=self.channel_names,
            pcm_f32le=tuple(bytes(chunk) for chunk in chunks),
            pts=int(frame.pts),
            time_base=time_base,
        )

    def finish(self) -> None:
        if self.sample_count == 0 or self._resampler is None or self._start_time is None:
            raise MediaDecodeError("audio stream decoded no samples", code="media_empty")
        if self._resampler.resample(None):
            raise MediaDecodeError(
                "native-rate audio conversion left unclocked samples", code="media_clock"
            )


def _stream_audio_format(stream: Any, limits: DecodeLimits) -> DecodedAudioFormat:
    context = stream.codec_context
    channels = int(context.channels or 0)
    sample_rate = int(context.sample_rate or 0)
    time_base = _positive_fraction(stream.time_base, "audio stream time base")
    layout = context.layout
    channel_layout = str(layout.name) if layout is not None else ""
    channel_names = (
        tuple(str(channel.name) for channel in layout.channels) if layout is not None else ()
    )
    if channels <= 0 or channels > limits.max_audio_channels:
        raise MediaDecodeError(
            f"audio has {channels} channels; limit is {limits.max_audio_channels}",
            code="audio_channels_limit",
        )
    if sample_rate <= 0 or not channel_layout or len(channel_names) != channels:
        raise MediaDecodeError("audio header has no valid native rate/layout", code="audio_format")
    return DecodedAudioFormat(
        channels=channels,
        sample_rate=sample_rate,
        channel_layout=channel_layout,
        channel_names=channel_names,
        time_base=time_base,
    )


class _VideoStreamDecoder:
    """Validate one video stream while retaining at most one RGB frame."""

    def __init__(self, av: Any, stream: Any, limits: DecodeLimits, budget: _Budget) -> None:
        self._limits = limits
        self._budget = budget
        self._av = av
        self._reformatter = av.video.reformatter.VideoReformatter()
        self._declared_frames = int(stream.frames or 0)
        self._last_pts: int | None = None
        self._pending: DecodedVideoFrame | None = None
        self._rotation: int | None = None
        self._color_facts: tuple[int, int, int, int] | None = None
        self._width = 0
        self._height = 0
        self._time_base: Fraction | None = None
        self.frame_count = 0
        rate = stream.average_rate
        self._nominal_frame_rate = (
            _positive_fraction(rate, "video average rate") if rate is not None else None
        )
        ratio = stream.sample_aspect_ratio
        self._pixel_aspect_ratio = (
            _positive_fraction(ratio, "video pixel aspect ratio")
            if ratio is not None
            else Fraction(1, 1)
        )

        context = stream.codec_context
        context.thread_count = 1
        _set_decoder_option(context, "err_detect", "explode")
        _set_decoder_option(context, "max_pixels", limits.max_pixels_per_frame)
        width, height = int(context.width or 0), int(context.height or 0)
        if width > 0 and height > 0:
            _check_pixels(width, height, limits)
        if self._declared_frames > limits.max_video_frames:
            raise MediaDecodeError(
                f"video metadata declares {self._declared_frames} frames; limit is "
                f"{limits.max_video_frames}",
                code="video_frames_limit",
            )

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def time_base(self) -> Fraction | None:
        return self._time_base

    @property
    def pixel_aspect_ratio(self) -> Fraction:
        return self._pixel_aspect_ratio

    @property
    def nominal_frame_rate(self) -> Fraction | None:
        return self._nominal_frame_rate

    def decode(self, frame: Any) -> DecodedVideoFrame | None:
        if frame.is_corrupt:
            raise MediaDecodeError("video decoder marked a frame corrupt", code="media_corrupt")
        if self.frame_count >= self._limits.max_video_frames:
            raise MediaDecodeError(
                f"video crosses the {self._limits.max_video_frames}-frame limit",
                code="video_frames_limit",
            )
        _check_pixels(int(frame.width), int(frame.height), self._limits)
        if frame.pts is None or frame.time_base is None:
            raise MediaDecodeError("video frame has no exact PTS/time base", code="media_clock")
        time_base = _positive_fraction(frame.time_base, "video frame time base")
        if self._time_base is None:
            self._time_base = time_base
        elif time_base != self._time_base:
            raise MediaDecodeError("video time base changes between frames", code="media_clock")
        pts = int(frame.pts)
        if self._last_pts is not None and pts <= self._last_pts:
            raise MediaDecodeError("video PTS is not strictly monotone", code="media_clock")
        duration = int(frame.duration or 0)
        if duration <= 0:
            raise MediaDecodeError("video frame has no exact positive duration", code="media_clock")
        rotation = _display_rotation(frame)
        if self._rotation is None:
            self._rotation = rotation
            if rotation % 180:
                self._pixel_aspect_ratio = 1 / self._pixel_aspect_ratio
        elif rotation != self._rotation:
            raise MediaDecodeError("display rotation changes between frames", code="media_rotation")

        color_facts = _color.normalize(
            width=int(frame.width),
            height=int(frame.height),
            primaries=int(frame.color_primaries),
            transfer=int(frame.color_trc),
            matrix=int(frame.colorspace),
            color_range=int(frame.color_range),
        )
        if self._color_facts is None:
            self._color_facts = color_facts
        elif color_facts != self._color_facts:
            raise MediaDecodeError(
                "video color facts change between frames", code="media_color_change"
            )

        self._budget.claim(int(frame.width) * int(frame.height) * 3, "video RGB")
        rgb = _tight_rgb(frame, self._reformatter, av=self._av, color_facts=color_facts)
        rgb, width, height = _rotate_rgb(rgb, int(frame.width), int(frame.height), rotation)
        if self._width and (width != self._width or height != self._height):
            raise MediaDecodeError(
                "video frame geometry changes after display rotation", code="media_geometry"
            )
        self._width, self._height = width, height
        self._last_pts = pts
        self.frame_count += 1
        current = DecodedVideoFrame(
            width=width,
            height=height,
            rgb=rgb,
            pts=pts,
            duration=duration,
            time_base=time_base,
            pixel_aspect_ratio=self._pixel_aspect_ratio,
            color_primaries=color_facts[0],
            color_transfer=color_facts[1],
            color_matrix=color_facts[2],
            color_range=color_facts[3],
        )
        previous = self._pending
        self._pending = current
        if previous is None:
            return None
        derived_duration = current.pts - previous.pts
        if derived_duration <= 0:
            raise MediaDecodeError(
                "video presentation has a gap or overlap", code="video_discontinuity"
            )
        # Coarse clocks (for example WebM milliseconds at 24 fps) round adjacent starts by
        # one source tick. Normalize that one-tick ambiguity; anything larger is a real gap.
        if abs(derived_duration - previous.duration) > 1:
            raise MediaDecodeError(
                "video presentation has a gap or overlap", code="video_discontinuity"
            )
        return replace(previous, duration=derived_duration)

    def finish(self) -> DecodedVideoFrame:
        if self.frame_count == 0 or self._time_base is None or self._pending is None:
            raise MediaDecodeError("video stream decoded no frames", code="media_empty")
        if self._declared_frames and self.frame_count != self._declared_frames:
            raise MediaDecodeError(
                f"video decoded {self.frame_count} of {self._declared_frames} declared frames",
                code="media_truncated",
            )
        return self._pending


AssetT = TypeVar("AssetT", ImageAsset, AudioAsset, VideoAsset)


class DecodedMediaStream(Iterator[DecodedMediaEvent]):
    """One context-managed, single-consumer stream over verified media bytes.

    PyAV, packets and the attempt-spool path remain private. A successful context must be
    consumed to exhaustion so truncation and final decoder state are always checked.
    """

    __slots__ = (
        "_acquire",
        "_asset",
        "_attempt",
        "_av",
        "_budget",
        "_closed",
        "_container",
        "_entered",
        "_events",
        "_exhausted",
        "_extra_containers",
        "_extra_sources",
        "_limits",
        "_mode",
        "_open_source",
        "_release",
        "_source",
    )

    def __init__(
        self,
        attempt: _Attempt,
        asset: AudioAsset | VideoAsset,
        *,
        av: Any,
        budget: _Budget,
        limits: DecodeLimits,
        mode: Literal["audio", "video"],
        open_source: Callable[[], BinaryIO],
        acquire: Callable[[], None],
        release: Callable[[], None],
    ) -> None:
        self._attempt = attempt
        self._asset = asset
        self._av = av
        self._budget = budget
        self._limits = limits
        self._mode = mode
        self._open_source = open_source
        self._acquire = acquire
        self._release = release
        self._container: Any | None = None
        self._source: BinaryIO | None = None
        self._events: Iterator[DecodedMediaEvent] | None = None
        self._extra_containers: list[Any] = []
        self._extra_sources: list[BinaryIO] = []
        self._entered = False
        self._closed = False
        self._exhausted = False

    def __enter__(self) -> DecodedMediaStream:
        if self._entered:
            raise CapabilityError(
                "a decoded media stream may be entered once", code="media_stream_used"
            )
        self._attempt.check_open("MediaDecoder.stream")
        self._asset._check_file()
        self._acquire()
        try:
            self._source = self._open_source()
            self._container = self._av.open(self._source, mode="r")
            self._events = (
                self._video_events(self._container)
                if self._mode == "video"
                else self._audio_events(self._container)
            )
            self._entered = True
            return self
        except MediaDecodeError as exc:
            self._close()
            _attribute(exc, self._asset)
            raise
        except OSError as exc:
            self._close()
            raise MediaMaterializationError(
                f"verified input bytes could not be opened: {_safe_error(exc)}",
                fields=[self._asset._input_id] if self._asset._input_id else (),
            ) from exc
        except Exception as exc:
            self._close()
            raise MediaDecodeError(
                f"media container could not be opened: {_safe_error(exc)}",
                code="media_malformed",
                fields=[self._asset._input_id] if self._asset._input_id else (),
            ) from exc

    def __iter__(self) -> DecodedMediaStream:
        return self

    def __next__(self) -> DecodedMediaEvent:
        if not self._entered or self._events is None:
            raise CapabilityError(
                "a decoded media stream must be used as a context manager",
                code="media_stream_context",
            )
        if self._closed:
            raise StopIteration
        self._attempt.check_open("MediaDecoder.stream")
        try:
            self._asset._check_file()
            event = next(self._events)
            self._asset._check_file()
            return event
        except RuntimeFailure:
            self._close()
            raise
        except StopIteration:
            self._exhausted = True
            self._close()
            self._asset._check_file()
            raise
        except MediaDecodeError as exc:
            self._close()
            _attribute(exc, self._asset)
            raise
        except OSError as exc:
            self._close()
            raise MediaMaterializationError(
                f"verified input bytes could not be read: {_safe_error(exc)}",
                fields=[self._asset._input_id] if self._asset._input_id else (),
            ) from exc
        except Exception as exc:
            self._close()
            raise MediaDecodeError(
                f"media container could not be decoded: {_safe_error(exc)}",
                code="media_malformed",
                fields=[self._asset._input_id] if self._asset._input_id else (),
            ) from exc

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        incomplete = exc_type is None and not self._exhausted
        self._close()
        self._asset._check_file()
        if incomplete:
            error = MediaDecodeError(
                "decoded media stream exited before every event was consumed",
                code="media_stream_incomplete",
            )
            _attribute(error, self._asset)
            raise error

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._container is not None:
                with suppress(Exception):
                    self._container.close()
            for container in self._extra_containers:
                with suppress(Exception):
                    container.close()
        finally:
            try:
                if self._source is not None:
                    with suppress(Exception):
                        self._source.close()
                for source in self._extra_sources:
                    with suppress(Exception):
                        source.close()
            finally:
                self._release()

    def _audio_events(self, container: Any) -> Iterator[DecodedMediaEvent]:
        streams = list(container.streams.audio)
        if len(streams) != 1:
            _required_stream("audio", len(streams))
        audio_format = _stream_audio_format(streams[0], self._limits)
        decoder = _AudioStreamDecoder(self._av, streams[0], self._limits, self._budget)
        yield DecodedMediaHeader(video=None, audio=audio_format)
        for frame in container.decode(streams[0]):
            yield decoder.decode(frame)
        decoder.finish()

    def _video_events(self, container: Any) -> Iterator[DecodedMediaEvent]:
        video_streams = list(container.streams.video)
        audio_streams = list(container.streams.audio)
        if len(video_streams) != 1:
            _required_stream("video", len(video_streams))
        if len(audio_streams) > 1:
            raise MediaDecodeError(
                "video asset has more than one soundtrack stream",
                code="multiple_audio_streams",
            )
        video_stream = video_streams[0]
        video = _VideoStreamDecoder(self._av, video_stream, self._limits, self._budget)

        def decoded_video_events() -> Iterator[DecodedVideoFrame]:
            for frame in container.decode(video_stream):
                event = video.decode(frame)
                if event is not None:
                    yield event
            yield video.finish()

        video_frames = iter(decoded_video_events())
        try:
            next_video: DecodedVideoFrame | None = next(video_frames)
        except StopIteration:
            next_video = None

        audio: _AudioStreamDecoder | None = None
        audio_frames: Iterator[Any] = iter(())
        next_audio: DecodedAudioChunk | None = None
        audio_format: DecodedAudioFormat | None = None
        if audio_streams:
            audio_source = self._open_source()
            audio_container = self._av.open(audio_source, mode="r")
            self._extra_sources.append(audio_source)
            self._extra_containers.append(audio_container)
            second_audio_streams = list(audio_container.streams.audio)
            if len(second_audio_streams) != 1:
                _required_stream("audio", len(second_audio_streams))
            audio_stream = second_audio_streams[0]
            audio_format = _stream_audio_format(audio_stream, self._limits)
            audio = _AudioStreamDecoder(self._av, audio_stream, self._limits, self._budget)
            audio_frames = iter(audio_container.decode(audio_stream))
            try:
                next_audio = audio.decode(next(audio_frames))
            except StopIteration:
                next_audio = None

        if next_video is not None:
            yield DecodedMediaHeader(
                video=DecodedVideoFormat(
                    width=next_video.width,
                    height=next_video.height,
                    time_base=next_video.time_base,
                    pixel_aspect_ratio=next_video.pixel_aspect_ratio,
                    nominal_frame_rate=video.nominal_frame_rate,
                    color_primaries=next_video.color_primaries,
                    color_transfer=next_video.color_transfer,
                    color_matrix=next_video.color_matrix,
                    color_range=next_video.color_range,
                ),
                audio=audio_format,
            )
        while next_video is not None or next_audio is not None:
            if next_audio is None or (
                next_video is not None and next_video.start_time <= next_audio.start_time
            ):
                assert next_video is not None
                yield next_video
                try:
                    next_video = next(video_frames)
                except StopIteration:
                    next_video = None
            else:
                yield next_audio
                assert audio is not None
                try:
                    next_audio = audio.decode(next(audio_frames))
                except StopIteration:
                    next_audio = None
        if audio is not None:
            audio.finish()


class MediaDecoder:
    """Attempt-bound decoder injected into handlers that explicitly declare it."""

    __slots__ = ("_active", "_attempt", "_budget", "_limits", "_open_streams")

    def __init__(
        self,
        attempt: _Attempt,
        *,
        active: Callable[[], bool],
        limits: DecodeLimits = DEFAULT_DECODE_LIMITS,
    ) -> None:
        self._attempt = attempt
        self._active = active
        self._limits = limits
        self._budget = _Budget(limits.max_decoded_bytes)
        self._open_streams = 0

    def value(self, asset: Asset) -> Mixed:
        """The lazy Assets projection, sharing this decoder's checks and budget."""
        if isinstance(asset, ImageAsset):
            image = self.decode_image(asset)
            return pil_frombytes("RGB", (image.width, image.height), image.rgb)
        if isinstance(asset, VideoAsset):
            return self.decode_video(asset)
        if isinstance(asset, AudioAsset):
            return self.decode_audio(asset)
        raise CapabilityError("decoded Assets requires image, video or audio", code="asset_kind")

    def decode_image(self, asset: ImageAsset) -> DecodedImage:
        _av_module, budget, path = self._start(asset, ImageAsset, "decode_image")
        try:
            return decode_image_path(path, self._limits, budget, asset._image_preparation)
        except MediaDecodeError as exc:
            _attribute(exc, asset)
            raise
        except OSError as exc:
            raise MediaMaterializationError(
                f"verified input bytes could not be read: {_safe_error(exc)}",
                fields=[asset._input_id] if asset._input_id else (),
            ) from exc
        except Exception as exc:
            raise MediaDecodeError(
                f"image container could not be decoded: {_safe_error(exc)}",
                code="media_malformed",
                fields=[asset._input_id] if asset._input_id else (),
            ) from exc
        finally:
            asset._check_file()

    def decode_audio(self, asset: AudioAsset) -> DecodedAudio:
        av, budget, path = self._start(asset, AudioAsset, "decode_audio")
        try:
            with av.open(str(path), mode="r") as container:
                streams = list(container.streams.audio)
                if len(streams) != 1:
                    _required_stream("audio", len(streams))
                accumulator = _AudioAccumulator(av, streams[0], self._limits, budget)
                for frame in container.decode(streams[0]):
                    accumulator.append(frame)
                return accumulator.finish()
        except MediaDecodeError as exc:
            _attribute(exc, asset)
            raise
        except OSError as exc:
            raise MediaMaterializationError(
                f"verified input bytes could not be read: {_safe_error(exc)}",
                fields=[asset._input_id] if asset._input_id else (),
            ) from exc
        except Exception as exc:
            raise MediaDecodeError(
                f"audio container could not be decoded: {_safe_error(exc)}",
                code="media_malformed",
                fields=[asset._input_id] if asset._input_id else (),
            ) from exc
        finally:
            asset._check_file()

    def decode_video(self, asset: VideoAsset) -> DecodedVideo:
        av, budget, path = self._start(asset, VideoAsset, "decode_video")
        try:
            with av.open(str(path), mode="r") as container:
                video_streams = list(container.streams.video)
                audio_streams = list(container.streams.audio)
                if len(video_streams) != 1:
                    _required_stream("video", len(video_streams))
                if len(audio_streams) > 1:
                    raise MediaDecodeError(
                        "video asset has more than one soundtrack stream",
                        code="multiple_audio_streams",
                    )
                video = _VideoAccumulator(av, video_streams[0], self._limits, budget)
                audio = (
                    _AudioAccumulator(av, audio_streams[0], self._limits, budget)
                    if audio_streams
                    else None
                )
                selected = (video_streams[0], *audio_streams)
                for packet in container.demux(*selected):
                    for frame in packet.decode():
                        if packet.stream.index == video_streams[0].index:
                            video.append(frame)
                        elif audio is not None:
                            audio.append(frame)
                return video.finish(audio.finish() if audio is not None else None)
        except MediaDecodeError as exc:
            _attribute(exc, asset)
            raise
        except OSError as exc:
            raise MediaMaterializationError(
                f"verified input bytes could not be read: {_safe_error(exc)}",
                fields=[asset._input_id] if asset._input_id else (),
            ) from exc
        except Exception as exc:
            raise MediaDecodeError(
                f"video container could not be decoded: {_safe_error(exc)}",
                code="media_malformed",
                fields=[asset._input_id] if asset._input_id else (),
            ) from exc
        finally:
            asset._check_file()

    def stream_audio(self, asset: AudioAsset | VideoAsset) -> DecodedMediaStream:
        """An audio asset, or a video's one soundtrack without decoding a picture."""
        if isinstance(asset, VideoAsset):
            av, budget, path = self._start(asset, VideoAsset, "stream_audio", retained=False)
        else:
            av, budget, path = self._start(asset, AudioAsset, "stream_audio", retained=False)
        return DecodedMediaStream(
            self._attempt,
            asset,
            av=av,
            budget=budget,
            limits=self._limits,
            mode="audio",
            open_source=lambda: path.open("rb"),
            acquire=self._acquire_stream,
            release=self._release_stream,
        )

    def stream_video(self, asset: VideoAsset) -> DecodedMediaStream:
        av, budget, path = self._start(asset, VideoAsset, "stream_video", retained=False)
        return DecodedMediaStream(
            self._attempt,
            asset,
            av=av,
            budget=budget,
            limits=self._limits,
            mode="video",
            open_source=lambda: path.open("rb"),
            acquire=self._acquire_stream,
            release=self._release_stream,
        )

    def _acquire_stream(self) -> None:
        self._attempt.check_open("MediaDecoder.stream")
        if self._active():
            raise CapabilityError(
                "media streaming is forbidden inside an active component-use scope",
                code="decode_in_component_scope",
            )
        if self._open_streams >= MAX_ACTIVE_MEDIA_STREAMS:
            raise CapabilityError(
                f"one MediaDecoder admits at most {MAX_ACTIVE_MEDIA_STREAMS} active streams",
                code="media_stream_active",
            )
        self._open_streams += 1

    def _release_stream(self) -> None:
        self._open_streams -= 1

    def _start(
        self,
        asset: AssetT,
        expected: type[AssetT],
        operation: str,
        *,
        retained: bool = True,
    ) -> tuple[Any, _Budget, Path]:
        self._attempt.check_open(f"MediaDecoder.{operation}")
        if self._active():
            raise CapabilityError(
                "media decode is forbidden inside an active component-use scope; decode "
                "verified inputs before acquiring model components",
                code="decode_in_component_scope",
            )
        if not isinstance(asset, expected):
            raise CapabilityError(
                f"MediaDecoder.{operation} requires {expected.__name__}, got "
                f"{type(asset).__name__}",
                code="asset_kind",
            )
        if not asset.hydrated:
            raise MediaDecodeError(
                f"asset {asset.ref!r} was not hydrated before decode",
                code="asset_unhydrated",
                fields=[asset._input_id] if asset._input_id else (),
            )
        if asset._attempt != self._attempt.request_id:
            raise CapabilityError(
                f"asset {asset.ref!r} belongs to attempt {asset._attempt!r}",
                code="foreign_asset",
                fields=[asset._input_id] if asset._input_id else (),
            )
        if asset._max_decoded_bytes <= 0:
            raise CapabilityError(
                f"asset field {asset.ref!r} has no decoded-byte ceiling",
                code="decoded_bound_missing",
                fields=[asset._input_id] if asset._input_id else (),
            )
        if asset._read_guard is not None:
            asset._read_guard()
        av = _av()
        budget = _Budget(
            min(asset._max_decoded_bytes, self._limits.max_decoded_bytes),
            parent=self._budget if retained else None,
            scope="retained decoded" if retained else "streamed decoded",
            cumulative=retained,
        )
        asset._check_file()
        assert asset._local is not None
        return av, budget, asset._local


def _av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise DecoderUnavailable("media decode requires cozy-runtime[media] (PyAV)") from exc
    return av


def require_decoder_dependency() -> None:
    """Build-time closure check for a surface that declares ``MediaDecoder``."""
    _av()


# Local preparation can read a larger source than the admitted derivative. These
# are source/working limits, separate from AssetBound and retained RGB budgets.
IMAGE_SOURCE_MAX_BYTES = 256 << 20
IMAGE_WORKING_MAX_BYTES = 2 << 30


def image_target_size(
    width: int, height: int, preparation: ImagePreparation | None
) -> tuple[int, int]:
    if preparation is None:
        return width, height
    scale = 1.0
    if preparation.max_edge is not None:
        scale = min(scale, preparation.max_edge / max(width, height))
    if preparation.max_pixels is not None:
        scale = min(scale, math.sqrt(preparation.max_pixels / (width * height)))
    target_width, target_height = (
        max(1, math.floor(width * scale)),
        max(1, math.floor(height * scale)),
    )
    if preparation.max_pixels is not None and target_width * target_height > preparation.max_pixels:
        if target_width >= target_height:
            target_width = max(1, preparation.max_pixels // target_height)
        else:
            target_height = max(1, preparation.max_pixels // target_width)
    return target_width, target_height


def image_decoded_bytes(path: Path) -> int:
    """Header-only RGB8 size of an encoded image: what a decoded bound promises."""
    with pil_open(path) as header:
        width, height = header.size
    if width <= 0 or height <= 0:
        raise MediaDecodeError("image has invalid frame geometry", code="media_geometry")
    return width * height * 3


def image_source_geometry(
    path: Path, limits: DecodeLimits, preparation: ImagePreparation
) -> tuple[int, int]:
    """Header-only refusal before the codec opens or decodes a full frame."""
    if path.stat().st_size > IMAGE_SOURCE_MAX_BYTES:
        raise MediaDecodeError(
            "image source exceeds preparation byte ceiling", code="image_source_bytes"
        )
    with pil_open(path) as header:
        width, height = header.size
        _check_pixels(width, height, limits)
    target = image_target_size(width, height, preparation)
    # A conservative planned envelope, not a claim that native allocations are
    # individually intercepted. The executor's containment remains authoritative.
    working = width * height * 8 + target[0] * target[1] * 16 + (64 << 20)
    if working > IMAGE_WORKING_MAX_BYTES:
        raise MediaDecodeError(
            "image source exceeds preparation working envelope", code="image_working_bytes"
        )
    return width, height


def decode_image_path(
    path: Path, limits: DecodeLimits, budget: _Budget, preparation: ImagePreparation | None = None
) -> DecodedImage:
    """One exact decoder core for a verified Runtime input or authorized local helper.

    It receives no grant authority. MediaDecoder still performs its attempt checks;
    the neutral CLI helper owns its separately authorized local input and limits.
    """
    source_geometry = (
        image_source_geometry(path, limits, preparation) if preparation is not None else None
    )
    with path.open("rb") as source:
        jpeg = source.read(2) == b"\xff\xd8"
        source.seek(0)
        rotation = _jpeg_rotation(source) if jpeg else None
    av = _av()
    with av.open(str(path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            _required_stream("image", len(streams))
        stream = streams[0]
        _set_decoder_option(stream.codec_context, "err_detect", "explode")
        _set_decoder_option(stream.codec_context, "max_pixels", limits.max_pixels_per_frame)
        stream.codec_context.thread_count = 1
        width = int(stream.codec_context.width or 0)
        height = int(stream.codec_context.height or 0)
        if width > 0 and height > 0:
            _check_pixels(width, height, limits)
            if source_geometry is not None and (width, height) != source_geometry:
                raise MediaDecodeError(
                    "image header and codec geometry disagree", code="media_geometry"
                )
        decoded: DecodedImage | None = None
        for frame in container.decode(stream):
            if (
                source_geometry is not None
                and (int(frame.width), int(frame.height)) != source_geometry
            ):
                raise MediaDecodeError(
                    "image decoded geometry differs from its header", code="media_geometry"
                )
            if decoded is not None:
                raise MediaDecodeError(
                    "image asset contains more than one frame", code="animated_image"
                )
            decoded = _decode_image_frame(
                frame, limits, budget, rotation=rotation, preparation=preparation
            )
        if decoded is None:
            raise MediaDecodeError("image stream decoded no frame", code="media_empty")
        return decoded


def _decode_image_frame(
    frame: Any,
    limits: DecodeLimits,
    budget: _Budget,
    *,
    rotation: int | None,
    preparation: ImagePreparation | None = None,
) -> DecodedImage:
    if frame.is_corrupt:
        raise MediaDecodeError("image decoder marked the frame corrupt", code="media_corrupt")
    _check_pixels(int(frame.width), int(frame.height), limits)
    rotation = _display_rotation(frame) if rotation is None else rotation
    width, height = image_target_size(int(frame.width), int(frame.height), preparation)
    budget.claim(width * height * 3, "image RGB")
    if (width, height) != (int(frame.width), int(frame.height)):
        av = _av()
        interpolation = (
            av.video.reformatter.Interpolation.LANCZOS
            | av.video.reformatter.Interpolation.ACCURATE_RND
            | av.video.reformatter.Interpolation.BITEXACT
        )
        frame = frame.reformat(
            width=width, height=height, format="rgb24", interpolation=interpolation, threads=1
        )
    rgb = _tight_rgb(frame)
    rgb, width, height = _rotate_rgb(rgb, width, height, rotation)
    return DecodedImage(width=width, height=height, rgb=rgb)


def _tight_rgb(
    frame: Any,
    reformatter: Any | None = None,
    *,
    av: Any | None = None,
    color_facts: tuple[int, int, int, int] | None = None,
) -> bytes:
    if reformatter is None or av is None or color_facts is None:
        rgb = frame.reformat(format="rgb24")
    else:
        primaries, transfer, matrix, color_range = color_facts
        try:
            colorspace = _color.colorspace(av, matrix)
            source_range = _color.range_value(av, color_range)
        except ValueError as exc:
            raise MediaDecodeError(str(exc), code="video_color") from exc
        interpolation = (
            av.video.reformatter.Interpolation.BILINEAR
            | av.video.reformatter.Interpolation.ACCURATE_RND
            | av.video.reformatter.Interpolation.BITEXACT
        )
        rgb = reformatter.reformat(
            frame,
            format="rgb24",
            src_colorspace=colorspace,
            dst_colorspace=colorspace,
            src_color_range=source_range,
            dst_color_range=av.video.reformatter.ColorRange.JPEG,
            dst_color_trc=transfer,
            dst_color_primaries=primaries,
            interpolation=interpolation,
            threads=1,
        )
    width, height = int(rgb.width), int(rgb.height)
    if width <= 0 or height <= 0 or len(rgb.planes) != 1:
        raise MediaDecodeError("decoder returned invalid RGB geometry", code="media_geometry")
    plane = rgb.planes[0]
    row_bytes = width * 3
    line_size = int(plane.line_size)
    raw = memoryview(plane)
    if line_size < row_bytes or len(raw) < line_size * height:
        raise MediaDecodeError(
            "RGB plane stride is shorter than its geometry", code="media_geometry"
        )
    if line_size == row_bytes:
        return bytes(raw[: row_bytes * height])
    packed = bytearray(row_bytes * height)
    for row in range(height):
        start = row * line_size
        packed[row * row_bytes : (row + 1) * row_bytes] = raw[start : start + row_bytes]
    return bytes(packed)


def _rotate_rgb(rgb: bytes, width: int, height: int, rotation: int) -> tuple[bytes, int, int]:
    if rotation == 0:
        return rgb, width, height
    transpose = {
        90: Transpose.ROTATE_90,
        -90: Transpose.ROTATE_270,
        180: Transpose.ROTATE_180,
        -180: Transpose.ROTATE_180,
    }[rotation]
    image = pil_frombytes("RGB", (width, height), rgb).transpose(transpose)
    return image.tobytes(), image.width, image.height


def _valid_audio_samples(frame: Any, samples: int, rate: int) -> int:
    duration = frame.duration
    if duration is None or int(duration) <= 0:
        return samples
    time_base = frame.time_base
    if time_base is None:
        return samples
    exact_base = _positive_fraction(time_base, "audio frame time base")
    if exact_base != Fraction(1, rate):
        return samples
    exact = _fraction(duration, "audio frame duration") * exact_base * rate
    # AAC exposes final-frame padding as an exact shorter duration on the native sample
    # clock. Coarser Opus/Vorbis container clocks are not sample-count authority even when
    # their millisecond duration happens to multiply to an integer.
    if exact.denominator == 1 and 0 < exact.numerator <= samples:
        return exact.numerator
    return samples


def _require_finite_f32(data: bytes) -> None:
    # Summed in double, finite f32 samples cannot overflow; one NaN or infinity cannot vanish.
    values = array("f")
    values.frombytes(data)
    if not math.isfinite(sum(values)):
        raise MediaDecodeError("audio PCM contains NaN or infinity", code="audio_format")


def _jpeg_rotation(source: BinaryIO | bytes) -> int:
    """Read bounded JPEG EXIF segments without snapshotting compressed input."""
    stream = io.BytesIO(source) if isinstance(source, bytes) else source
    stream.seek(0, io.SEEK_END)
    size = stream.tell()
    stream.seek(2)
    found: list[int] = []
    while stream.tell() < size:
        if stream.read(1) != b"\xff":
            break
        marker_bytes = stream.read(1)
        while marker_bytes == b"\xff":
            marker_bytes = stream.read(1)
        if not marker_bytes:
            raise MediaDecodeError("JPEG marker is truncated", code="media_malformed")
        marker = marker_bytes[0]
        if marker in (0xD9, 0xDA):
            break
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        length_bytes = stream.read(2)
        if len(length_bytes) != 2:
            raise MediaDecodeError("JPEG segment length is truncated", code="media_malformed")
        length = int.from_bytes(length_bytes, "big") - 2
        if length < 0 or stream.tell() + length > size:
            raise MediaDecodeError("JPEG segment crosses the input", code="media_malformed")
        if marker == 0xE1:
            payload = stream.read(length)
            if payload.startswith(b"Exif\x00\x00"):
                orientation = _tiff_orientation(payload[6:])
                if orientation is not None:
                    found.append(orientation)
        else:
            stream.seek(length, io.SEEK_CUR)
    if len(found) > 1:
        raise MediaDecodeError("JPEG carries multiple EXIF orientations", code="media_rotation")
    orientation = found[0] if found else 1
    rotations = {1: 0, 3: 180, 6: -90, 8: 90}
    if orientation not in rotations:
        raise MediaDecodeError(
            f"JPEG EXIF orientation {orientation} contains a mirror transform",
            code="media_rotation",
        )
    return rotations[orientation]


def _tiff_orientation(tiff: bytes) -> int | None:
    if len(tiff) < 8 or tiff[:2] not in {b"II", b"MM"}:
        raise MediaDecodeError("EXIF TIFF header is malformed", code="media_malformed")
    byteorder: Literal["little", "big"] = "little" if tiff[:2] == b"II" else "big"
    if int.from_bytes(tiff[2:4], byteorder) != 42:
        raise MediaDecodeError("EXIF TIFF magic is invalid", code="media_malformed")
    ifd = int.from_bytes(tiff[4:8], byteorder)
    if ifd + 2 > len(tiff):
        raise MediaDecodeError("EXIF IFD offset crosses the segment", code="media_malformed")
    count = int.from_bytes(tiff[ifd : ifd + 2], byteorder)
    end = ifd + 2 + count * 12
    if end > len(tiff):
        raise MediaDecodeError("EXIF IFD entries are truncated", code="media_malformed")
    values: list[int] = []
    for index in range(count):
        entry = tiff[ifd + 2 + index * 12 : ifd + 14 + index * 12]
        tag = int.from_bytes(entry[:2], byteorder)
        if tag != 0x0112:
            continue
        kind = int.from_bytes(entry[2:4], byteorder)
        items = int.from_bytes(entry[4:8], byteorder)
        if kind != 3 or items != 1:
            raise MediaDecodeError("EXIF orientation has an invalid type", code="media_malformed")
        values.append(int.from_bytes(entry[8:10], byteorder))
    if len(values) > 1:
        raise MediaDecodeError("EXIF IFD repeats the orientation tag", code="media_rotation")
    return values[0] if values else None


_PURE_DISPLAY_MATRICES = {
    (65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824): 0,
    (0, -65536, 0, 65536, 0, 0, 0, 0, 1073741824): 90,
    (0, 65536, 0, -65536, 0, 0, 0, 0, 1073741824): -90,
    (-65536, 0, 0, 0, -65536, 0, 0, 0, 1073741824): 180,
}


def _display_rotation(frame: Any) -> int:
    try:
        matrices = [side for side in frame.side_data if side.type.name == "DISPLAYMATRIX"]
    except ValueError as exc:
        # PyAV 18.1 cannot represent some FFmpeg side-data enums (notably EXIF display
        # transforms). Accepting only frame.rotation would silently discard mirror state.
        raise MediaDecodeError(
            "display transform cannot be inspected without losing orientation",
            code="media_rotation",
        ) from exc
    if not matrices:
        if int(frame.rotation):
            raise MediaDecodeError(
                "frame reports rotation without an inspectable display matrix",
                code="media_rotation",
            )
        return 0
    if len(matrices) != 1:
        raise MediaDecodeError("frame has multiple display matrices", code="media_rotation")
    raw = bytes(matrices[0])
    if len(raw) != 36:
        raise MediaDecodeError("display matrix has an invalid size", code="media_rotation")
    rotation = _PURE_DISPLAY_MATRICES.get(struct.unpack("=9i", raw))
    if rotation is None:
        raise MediaDecodeError(
            "display matrix contains a flip, shear, scale, or non-quarter rotation",
            code="media_rotation",
        )
    reported = int(frame.rotation)
    if reported == -180:
        reported = 180
    if reported != rotation:
        raise MediaDecodeError("display matrix rotation is inconsistent", code="media_rotation")
    return rotation


def _set_decoder_option(context: Any, name: str, value: object) -> None:
    context.options = {**context.options, name: str(value)}


def _check_pixels(width: int, height: int, limits: DecodeLimits) -> None:
    pixels = width * height
    if width <= 0 or height <= 0:
        raise MediaDecodeError("media has invalid frame geometry", code="media_geometry")
    if pixels > limits.max_pixels_per_frame:
        raise MediaDecodeError(
            f"frame has {pixels} pixels; limit is {limits.max_pixels_per_frame}",
            code="pixels_per_frame_limit",
        )


def _required_stream(kind: str, count: int) -> None:
    if count == 0:
        raise MediaDecodeError(f"{kind} asset has no {kind} stream", code=f"{kind}_stream_missing")
    raise MediaDecodeError(
        f"{kind} asset has {count} {kind} streams; exactly one is required",
        code=f"multiple_{kind}_streams",
    )


def _fraction(value: object, what: str) -> Fraction:
    try:
        if isinstance(value, Fraction):
            return value
        if isinstance(value, int | float | str):
            return Fraction(value)
        raise TypeError(type(value).__name__)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise MediaDecodeError(f"{what} is not rational", code="media_clock") from exc


def _positive_fraction(value: object, what: str) -> Fraction:
    exact = _fraction(value, what)
    if exact <= 0:
        raise MediaDecodeError(f"{what} is not positive", code="media_clock")
    return exact


def _safe_error(exc: Exception) -> str:
    text = str(exc).splitlines()[0][:240]
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _attribute(exc: AuthorError, asset: ImageAsset | AudioAsset | VideoAsset) -> None:
    if not exc.fields and asset._input_id:
        exc.fields = (asset._input_id,)
