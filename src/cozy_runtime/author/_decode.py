"""The one bounded input-media reader: verified bytes in, bounded media values out.

Every hydrated `ImageAsset`, `VideoAsset` and `AudioAsset` reads through its attempt's
one `MediaReads`, which owns the attempt's retained-value budget and active-stream count.
There is deliberately no path, URL, fetch, resize-of-video, model-normalization, or
output-codec surface here. PyAV is a lazy member of the ``media`` extra; the base author
package does not import it.
"""

from __future__ import annotations

import io
import math
import os
import struct
import sys
import threading
from array import array
from collections.abc import Callable, Generator, Iterator
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from PIL.Image import DecompressionBombError, Resampling, Transpose
from PIL.Image import Image as PILImage
from PIL.Image import frombytes as pil_frombytes
from PIL.Image import open as pil_open
from PIL.ImageOps import exif_transpose

from cozy_runtime.author import _color
from cozy_runtime.author._errors import AuthorError, CapabilityError, InvalidRequest, RuntimeFailure
from cozy_runtime.author._markers import ImagePreparation, seconds

if TYPE_CHECKING:
    from cozy_runtime.author._assets import Asset, AudioAsset, ImageAsset, VideoAsset


class _Attempt(Protocol):
    request_id: str

    def check_open(self, what: str) -> None: ...

    def components_active(self) -> bool: ...


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


# Structural codec ceilings, not package policy. Frame and sample ceilings count values a
# read keeps; duration policy is the field's declared preparation. `max_decoded_bytes` is only
# what an `Assets` media kind that declares none publishes (its interface stays byte-stable);
# it is never enforced: the attempt's budget is what the machine holds (`host_decode_limits`),
# and only a bound the author declared narrows its field's reads.
DEFAULT_DECODE_LIMITS = DecodeLimits(
    max_decoded_bytes=2 * 1024**3,
    max_pixels_per_frame=8192 * 8192,
    max_video_frames=3600,
    max_audio_channels=8,
    max_audio_samples=48_000 * 60 * 10,
)

MAX_ACTIVE_MEDIA_STREAMS = 2


def host_decode_limits() -> DecodeLimits:
    """The codec ceilings with the retained-media budget set by what this machine holds, its
    memory and swap, never a flat constant: decoded media that fits the machine is served. A
    field's declared `AssetBound(max_decoded_bytes=...)` still narrows its own value."""
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        memory = sys.maxsize  # unknown: nothing is refused on a guess
    return replace(
        DEFAULT_DECODE_LIMITS, max_decoded_bytes=min(memory + _swap_bytes(), sys.maxsize)
    )


def _swap_bytes() -> int:
    """The machine's swap, where pages of retained media can go: 0 where it is not readable."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("SwapTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


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
    """Display-oriented RGB8 frames, exact presentation clock, and optional soundtrack.

    A constant-clock read (`fps=`) repeats a frame by reference: `frames_rgb[i] is
    frames_rgb[j]` when one source frame fills both slots.
    """

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


@dataclass(frozen=True, slots=True)
class ImageProbe:
    """Header facts of an image input. `width`/`height` are what `read()` returns."""

    width: int
    height: int
    source_width: int
    source_height: int


@dataclass(frozen=True, slots=True)
class AudioProbe:
    """Header facts of an audio input. `duration` is what a read can return at most."""

    duration: Fraction
    source_duration: Fraction
    channels: int
    sample_rate: int
    channel_layout: str


@dataclass(frozen=True, slots=True)
class VideoProbe:
    """Header facts of a video input: display-oriented geometry, clock and soundtrack."""

    width: int
    height: int
    duration: Fraction
    source_duration: Fraction
    frame_rate: Fraction | None
    pixel_aspect_ratio: Fraction
    soundtrack: DecodedAudioFormat | None


Probe = ImageProbe | VideoProbe | AudioProbe


@dataclass(slots=True)
class _Budget:
    limit: int
    parent: _Budget | None = None
    used: int = 0
    scope: str = "retained decoded"
    cumulative: bool = True
    lock: threading.Lock | None = None

    def claim(self, size: int, what: str) -> None:
        with self.lock or nullcontext():
            next_used = self.used + size if self.cumulative else max(self.used, size)
            if size < 0 or next_used > self.limit:
                raise MediaDecodeError(
                    f"{self.scope} {what} would cross the {self.limit}-byte value limit",
                    code="decoded_bytes_limit",
                )
            if self.parent is not None:
                self.parent.claim(size, "attempt media")
            self.used = next_used

    def refund(self) -> None:
        """Return what a failed read claimed, so the attempt can read again."""
        if self.parent is not None:
            self.parent.release(self.used)
        self.used = 0

    def release(self, size: int) -> None:
        with self.lock or nullcontext():
            self.used -= size


def _half_up(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


class _AudioStreamDecoder:
    """Validate one native-clock audio stream while retaining at most one decoded frame.

    With `end`, samples at or after `origin + end` are never converted: the frame that
    crosses it is validated whole and cut to the exact sample, and the caller stops reading.
    `origin` defaults to the stream's own first presentation time. The total-sample ceiling
    applies to a retained read; a stream holds one chunk at a time.
    """

    def __init__(
        self,
        av: Any,
        stream: Any,
        limits: DecodeLimits,
        budget: _Budget,
        *,
        end: Fraction | None = None,
        origin: Fraction | None = None,
        retained: bool = True,
    ) -> None:
        self._av = av
        self._limits = limits
        self._budget = budget
        self._end = end
        self._origin = origin
        self._retained = retained
        self._keep: int | None = None
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

    @property
    def done(self) -> bool:
        """Whether every sample before `end` is already read."""
        return self._keep is not None and self.sample_count >= self._keep

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

    def decode(self, frame: Any) -> DecodedAudioChunk | None:
        """One validated chunk, or None when nothing of this frame precedes `end`."""
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
            if self._end is not None:
                origin = presentation_time if self._origin is None else self._origin
                cut = (origin + self._end - presentation_time) * self.sample_rate
                self._keep = max(0, _half_up(cut))
        else:
            expected = self._start_time + Fraction(self.sample_count, self.sample_rate)
            if abs(presentation_time - expected) > time_base / 2:
                raise MediaDecodeError(
                    f"audio presentation is discontinuous: expected {expected}, got "
                    f"{presentation_time}",
                    code="audio_discontinuity",
                )
        self._last_presentation_time = presentation_time
        take = (
            valid_samples
            if self._keep is None
            else min(valid_samples, self._keep - self.sample_count)
        )
        if take <= 0:
            return None
        if self._retained and self.sample_count + take > self._limits.max_audio_samples:
            raise MediaDecodeError(
                f"audio crosses the {self._limits.max_audio_samples}-sample limit",
                code="audio_samples_limit",
            )
        self._budget.claim(take * self.channels * 4, "audio PCM")

        assert self._resampler is not None
        converted = self._resampler.resample(frame)
        converted = list(converted) if converted is not None else []
        if sum(int(out.samples) for out in converted) != samples:
            raise MediaDecodeError(
                "native-rate audio conversion changed the sample clock", code="audio_format"
            )
        chunks = [bytearray() for _ in range(self.channels)]
        remaining = take
        for out in converted:
            if str(out.format.name) != "fltp" or len(out.planes) != self.channels:
                raise MediaDecodeError(
                    "audio conversion returned a non-planar float frame", code="audio_format"
                )
            part = min(remaining, int(out.samples))
            if part:
                for channel, plane in enumerate(out.planes):
                    raw = memoryview(plane)
                    size = part * 4
                    if len(raw) < size:
                        raise MediaDecodeError(
                            "audio plane is shorter than its sample count", code="media_geometry"
                        )
                    chunk = bytes(raw[:size])
                    _require_finite_f32(chunk)
                    chunks[channel] += chunk
                remaining -= part
        if remaining:
            raise MediaDecodeError(
                "audio conversion returned fewer samples than its clock", code="media_truncated"
            )

        self.sample_count += take
        return DecodedAudioChunk(
            channels=self.channels,
            sample_count=take,
            sample_rate=self.sample_rate,
            channel_layout=self.channel_layout,
            channel_names=self.channel_names,
            pcm_f32le=tuple(bytes(chunk) for chunk in chunks),
            pts=int(frame.pts),
            time_base=time_base,
        )

    def finish(self) -> None:
        if self._keep == 0:
            return  # `end` precedes the first sample: an empty selection, not an empty stream
        if self.sample_count == 0 or self._resampler is None or self._start_time is None:
            raise MediaDecodeError("audio stream decoded no samples", code="media_empty")
        if self._resampler.resample(None):
            raise MediaDecodeError(
                "native-rate audio conversion left unclocked samples", code="media_clock"
            )

    def chunks(self, frames: Iterator[Any]) -> Iterator[DecodedAudioChunk]:
        """Decode until EOF or `end`; the demux stops at the cut."""
        for frame in frames:
            chunk = self.decode(frame)
            if chunk is None:
                break
            yield chunk
            if self.done:
                break
        self.finish()


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


@dataclass(slots=True)
class _Picture:
    """One validated, not yet converted picture and its exact presentation interval."""

    clock: _VideoClock
    frame: Any
    pts: int
    duration: int
    color: tuple[int, int, int, int]

    @property
    def start_time(self) -> Fraction:
        assert self.clock.time_base is not None
        return self.pts * self.clock.time_base

    def convert(self, budget: _Budget) -> DecodedVideoFrame:
        return self.clock.convert(self, budget)


class _VideoClock:
    """Validate one video stream in presentation order; pictures convert only when kept.

    Each frame shows until the next one starts, whatever duration the container records
    for it (B-frame and variable-rate streams record decode-order timing); only the last
    frame uses its own. Time running backwards refuses. With `end` (seconds after the first
    frame), the first frame starting at or after it ends the decode: only its start is
    read, it is never validated or converted, and the demux stops there. The last kept
    frame's duration is clipped to `end`. The declared frame count is checked only when
    the decode reached the end of the stream.
    """

    def __init__(self, av: Any, stream: Any, limits: DecodeLimits, end: Fraction | None) -> None:
        self._limits = limits
        self._av = av
        self._end = end
        self._reformatter = av.video.reformatter.VideoReformatter()
        self._declared_frames = int(stream.frames or 0)
        self._last_pts: int | None = None
        self._rotation: int | None = None
        self._color_facts: tuple[int, int, int, int] | None = None
        self._geometry: tuple[int, int] | None = None
        self._seen = 0
        self.origin: int | None = None
        self.time_base: Fraction | None = None
        self.width = 0
        self.height = 0
        self.stopped = False
        rate = stream.average_rate
        self.nominal_frame_rate = _positive_fraction(rate, "video average rate") if rate else None
        ratio = stream.sample_aspect_ratio
        # PyAV 19 represents unspecified ratios as a falsy AVRational rather
        # than None. The unspecified pixel aspect ratio means square pixels.
        self.pixel_aspect_ratio = (
            _positive_fraction(ratio, "video pixel aspect ratio") if ratio else Fraction(1, 1)
        )

        context = stream.codec_context
        context.thread_count = 1
        _set_decoder_option(context, "err_detect", "explode")
        _set_decoder_option(context, "max_pixels", limits.max_pixels_per_frame)
        width, height = int(context.width or 0), int(context.height or 0)
        if width > 0 and height > 0:
            _check_pixels(width, height, limits)

    @property
    def origin_time(self) -> Fraction:
        assert self.origin is not None and self.time_base is not None
        return self.origin * self.time_base

    def _check(self, frame: Any) -> _Picture:
        if frame.is_corrupt:
            raise MediaDecodeError("video decoder marked a frame corrupt", code="media_corrupt")
        width, height = int(frame.width), int(frame.height)
        _check_pixels(width, height, self._limits)
        if frame.pts is None or frame.time_base is None:
            raise MediaDecodeError("video frame has no exact PTS/time base", code="media_clock")
        time_base = _positive_fraction(frame.time_base, "video frame time base")
        if self.time_base is None:
            self.time_base = time_base
        elif time_base != self.time_base:
            raise MediaDecodeError("video time base changes between frames", code="media_clock")
        pts = int(frame.pts)
        if self._last_pts is not None and pts <= self._last_pts:
            raise MediaDecodeError("video PTS is not strictly monotone", code="media_clock")
        duration = int(frame.duration or 0)
        rotation = _display_rotation(frame)
        if self._rotation is None:
            self._rotation = rotation
            if rotation % 180:
                self.pixel_aspect_ratio = 1 / self.pixel_aspect_ratio
        elif rotation != self._rotation:
            raise MediaDecodeError("display rotation changes between frames", code="media_rotation")
        if self._geometry is None:
            self._geometry = (width, height)
            self.width, self.height = (height, width) if rotation % 180 else (width, height)
        elif self._geometry != (width, height):
            raise MediaDecodeError(
                "video frame geometry changes after display rotation", code="media_geometry"
            )
        color_facts = _color.normalize(
            width=width,
            height=height,
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
        if self.origin is None:
            self.origin = pts
        self._last_pts = pts
        self._seen += 1
        return _Picture(self, frame, pts, duration, color_facts)

    def _ends(self, frame: Any, pending: _Picture) -> bool:
        """Whether `frame` starts at or after `end`; then its start ends `pending`."""
        if self._end is None or frame.pts is None or frame.time_base is None:
            return False
        assert self.origin is not None and self.time_base is not None
        if _positive_fraction(frame.time_base, "video frame time base") != self.time_base:
            return False
        pts = int(frame.pts)
        if (pts - self.origin) * self.time_base < self._end:
            return False
        if pts <= pending.pts:
            raise MediaDecodeError("video PTS is not strictly monotone", code="media_clock")
        pending.duration = pts - pending.pts
        return True

    def pictures(self, frames: Iterator[Any]) -> Iterator[_Picture]:
        pending: _Picture | None = None
        for frame in frames:
            if pending is not None and self._ends(frame, pending):
                self.stopped = True
                break
            current = self._check(frame)
            if pending is not None:
                pending.duration = current.pts - pending.pts
                yield pending
            pending = current
        if pending is None:
            raise MediaDecodeError("video stream decoded no frames", code="media_empty")
        if not self.stopped and self._declared_frames and self._seen != self._declared_frames:
            raise MediaDecodeError(
                f"video decoded {self._seen} of {self._declared_frames} declared frames",
                code="media_truncated",
            )
        if pending.duration <= 0:
            raise MediaDecodeError(
                "the last video frame has no exact positive duration", code="media_clock"
            )
        if self._end is not None:
            assert self.origin is not None and self.time_base is not None
            room = self._end / self.time_base - (pending.pts - self.origin)
            if pending.duration > room:
                pending.duration = max(1, _half_up(room))
        yield pending

    def convert(self, picture: _Picture, budget: _Budget) -> DecodedVideoFrame:
        frame = picture.frame
        if frame is None:
            raise MediaDecodeError("video picture was already released", code="media_clock")
        assert self._rotation is not None and self.time_base is not None
        budget.claim(int(frame.width) * int(frame.height) * 3, "video RGB")
        rgb = _tight_rgb(frame, self._reformatter, av=self._av, color_facts=picture.color)
        rgb, width, height = _rotate_rgb(rgb, int(frame.width), int(frame.height), self._rotation)
        picture.frame = None
        primaries, transfer, matrix, color_range = picture.color
        return DecodedVideoFrame(
            width=width,
            height=height,
            rgb=rgb,
            pts=picture.pts,
            duration=picture.duration,
            time_base=self.time_base,
            pixel_aspect_ratio=self.pixel_aspect_ratio,
            color_primaries=primaries,
            color_transfer=transfer,
            color_matrix=matrix,
            color_range=color_range,
        )


_Event = DecodedMediaHeader | _Picture | DecodedAudioChunk


def _one_stream(streams: list[Any], kind: str) -> Any:
    if len(streams) != 1:
        _required_stream(kind, len(streams))
    return streams[0]


def _soundtrack(container: Any) -> Any | None:
    streams = list(container.streams.audio)
    if len(streams) > 1:
        raise MediaDecodeError(
            "video asset has more than one soundtrack stream", code="multiple_audio_streams"
        )
    return streams[0] if streams else None


def _audio_events(
    av: Any,
    container: Any,
    stream: Any,
    limits: DecodeLimits,
    budget: _Budget,
    *,
    end: Fraction | None,
    origin: Fraction | None,
    retained: bool,
) -> Iterator[_Event]:
    audio_format = _stream_audio_format(stream, limits)
    decoder = _AudioStreamDecoder(
        av, stream, limits, budget, end=end, origin=origin, retained=retained
    )
    yield DecodedMediaHeader(video=None, audio=audio_format)
    yield from decoder.chunks(iter(container.decode(stream)))


def _media_events(
    av: Any,
    path: Path,
    kind: str,
    limits: DecodeLimits,
    budget: _Budget,
    *,
    end: Fraction | None,
    retained: bool,
    audio: bool = True,
    pictures: bool = True,
) -> Generator[_Event]:
    """The time-ordered header, pictures and PCM of one input, bounded by `end`.

    The soundtrack reads through its own container, so neither track's demux runs past
    `end`. A soundtrack is cut at the first decoded video frame's time plus `end`, with or
    without its pictures.
    """
    with ExitStack() as stack:
        source = stack.enter_context(path.open("rb"))
        container = stack.enter_context(av.open(source, mode="r"))
        if kind == "audio":
            stream = _one_stream(list(container.streams.audio), "audio")
            yield from _audio_events(
                av, container, stream, limits, budget, end=end, origin=None, retained=retained
            )
            return
        _one_stream(list(container.streams.video), "video")
        soundtrack = _soundtrack(container)
        if not pictures:
            if soundtrack is None:
                _required_stream("audio", 0)
            origin = None if end is None else _first_picture_time(av, path)
            yield from _audio_events(
                av, container, soundtrack, limits, budget, end=end, origin=origin, retained=retained
            )
            return
        video_stream = container.streams.video[0]
        clock = _VideoClock(av, video_stream, limits, end)
        video = clock.pictures(iter(container.decode(video_stream)))
        next_video: _Picture | None = next(video)
        assert next_video is not None and clock.time_base is not None

        decoder: _AudioStreamDecoder | None = None
        sound: Iterator[DecodedAudioChunk] = iter(())
        audio_format: DecodedAudioFormat | None = None
        if audio and soundtrack is not None:
            audio_source = stack.enter_context(path.open("rb"))
            audio_container = stack.enter_context(av.open(audio_source, mode="r"))
            audio_stream = _one_stream(list(audio_container.streams.audio), "audio")
            audio_format = _stream_audio_format(audio_stream, limits)
            decoder = _AudioStreamDecoder(
                av,
                audio_stream,
                limits,
                budget,
                end=end,
                origin=clock.origin_time,
                retained=retained,
            )
            sound = decoder.chunks(iter(audio_container.decode(audio_stream)))
        next_audio = next(sound, None)

        primaries, transfer, matrix, color_range = next_video.color
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(
                width=clock.width,
                height=clock.height,
                time_base=clock.time_base,
                pixel_aspect_ratio=clock.pixel_aspect_ratio,
                nominal_frame_rate=clock.nominal_frame_rate,
                color_primaries=primaries,
                color_transfer=transfer,
                color_matrix=matrix,
                color_range=color_range,
            ),
            audio=audio_format,
        )
        while next_video is not None or next_audio is not None:
            if next_audio is None or (
                next_video is not None and next_video.start_time <= next_audio.start_time
            ):
                assert next_video is not None
                yield next_video
                next_video = next(video, None)
            else:
                yield next_audio
                next_audio = next(sound, None)


def _first_picture_time(av: Any, path: Path) -> Fraction:
    """The presentation time of a video's first decoded frame: the origin `end` counts from."""
    with path.open("rb") as source, av.open(source, mode="r") as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise MediaDecodeError("video frame has no exact PTS/time base", code="media_clock")
            return _fraction(frame.pts, "video frame PTS") * _positive_fraction(
                frame.time_base, "video frame time base"
            )
    raise MediaDecodeError("video stream decoded no frames", code="media_empty")


class DecodedMediaStream(Iterator[DecodedMediaEvent]):
    """One context-managed, single-consumer stream over verified media bytes.

    PyAV, packets and the attempt-spool path remain private. Reading to the end (or to
    the read's `end`) checks truncation and final decoder state; leaving the context early
    only closes it. The budget covers one event at a time, and a stream has no frame or
    sample-count ceiling: it retains nothing.
    """

    __slots__ = (
        "_asset",
        "_av",
        "_budget",
        "_closed",
        "_entered",
        "_events",
        "_header",
        "_open",
        "_reads",
    )

    def __init__(
        self,
        reads: MediaReads,
        asset: Asset,
        *,
        av: Any,
        budget: _Budget,
        events: Callable[[], Generator[_Event]],
    ) -> None:
        self._reads = reads
        self._asset = asset
        self._av = av
        self._budget = budget
        self._open = events
        self._events: Generator[DecodedMediaEvent] | None = None
        self._header: DecodedMediaEvent | None = None
        self._entered = False
        self._closed = False

    def __enter__(self) -> DecodedMediaStream:
        if self._entered:
            raise CapabilityError(
                "a decoded media stream may be entered once", code="media_stream_used"
            )
        self._reads._acquire_stream()
        self._entered = True
        self._events = self._converted(self._open())
        try:
            with _reading(self._asset, self._av, "media"):
                self._header = next(self._events)
        except BaseException:
            self._close()
            raise
        return self

    def _converted(self, events: Generator[_Event]) -> Generator[DecodedMediaEvent]:
        try:
            for event in events:
                yield event.convert(self._budget) if isinstance(event, _Picture) else event
        finally:
            events.close()

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
        if self._header is not None:
            header, self._header = self._header, None
            return header
        self._reads._attempt.check_open(f"{type(self._asset).__name__}.stream")
        try:
            with _reading(self._asset, self._av, "media"):
                event = next(self._events, None)
        except BaseException:
            self._close()
            raise
        if event is None:
            self._close()
            self._asset._check_file()
            raise StopIteration
        return event

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._close()
        self._asset._check_file()

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._events is not None:
                with suppress(Exception):
                    self._events.close()
        finally:
            self._reads._release_stream()


@contextmanager
def _reading(asset: Asset, av: Any, what: str) -> Iterator[None]:
    """One error boundary for every read: decode refusals name the field."""
    fields = [asset._input_id] if asset._input_id else []
    try:
        yield
    except MediaDecodeError as exc:
        _attribute(exc, asset)
        raise
    except AuthorError:
        raise
    except OSError as exc:
        raise MediaMaterializationError(
            f"verified input bytes could not be read: {_safe_error(exc)}", fields=fields
        ) from exc
    except Exception as exc:
        if av is not None and not isinstance(exc, av.FFmpegError):
            raise
        raise MediaDecodeError(
            f"{what} container could not be decoded: {_safe_error(exc)}",
            code="media_malformed",
            fields=fields,
        ) from exc
    finally:
        asset._check_file()


class MediaReads:
    """The attempt's one media reader. Every hydrated handle of the attempt reads here.

    One retained-value budget, the machine's memory, spans every `read()` of the attempt; a
    field's declared bound narrows its own reads. Streams are bounded per event and at most
    `MAX_ACTIVE_MEDIA_STREAMS` are open at once. Reads refuse inside an active component-use
    scope and after the attempt closes.
    """

    __slots__ = ("_attempt", "_budget", "_limits", "_lock", "_open_streams")

    def __init__(self, attempt: _Attempt, *, limits: DecodeLimits | None = None) -> None:
        self._attempt = attempt
        self._limits = limits = limits or host_decode_limits()
        # Overlapped calls (`author.concurrently`) read from their own threads.
        self._lock = threading.Lock()
        self._budget = _Budget(limits.max_decoded_bytes, lock=self._lock)
        self._open_streams = 0

    @property
    def retained_bytes(self) -> int:
        """Bytes every `read()` of this attempt has retained so far."""
        return self._budget.used

    def image(self, asset: ImageAsset) -> PILImage:
        budget, path = self._start(asset, "ImageAsset.read")
        try:
            with _reading(asset, None, "image"):
                prepared = _image(path, self._limits, budget, _image_preparation(asset))
        except BaseException:
            budget.refund()
            raise
        # A fresh RGB image of those pixels: no source metadata (a palette's `transparency`,
        # EXIF, ICC) reaches the package, so a later `convert("RGBA")` stays opaque.
        return pil_frombytes("RGB", prepared.size, prepared.tobytes())

    def video(self, asset: VideoAsset, *, end: object, fps: object, audio: bool) -> DecodedVideo:
        cut = _effective_end(end, asset._end)
        rate = None if fps is None else _rate(fps)
        budget, path = self._start(asset, "VideoAsset.read")
        av = _av()
        events = _media_events(
            av, path, "video", self._limits, budget, end=cut, retained=True, audio=audio
        )
        try:
            with _reading(asset, av, "video"):
                return _collect_video(events, budget, self._limits, rate)
        except BaseException:
            budget.refund()
            raise
        finally:
            events.close()

    def audio(self, asset: AudioAsset, *, end: object) -> DecodedAudio:
        cut = _effective_end(end, asset._end)
        budget, path = self._start(asset, "AudioAsset.read")
        av = _av()
        events = _media_events(av, path, "audio", self._limits, budget, end=cut, retained=True)
        try:
            with _reading(asset, av, "audio"):
                header = next(events)
                assert isinstance(header, DecodedMediaHeader) and header.audio is not None
                sound = _collect_audio(header.audio, events)
                if sound is None:
                    raise MediaDecodeError("audio read selected no samples", code="media_empty")
                return sound
        except BaseException:
            budget.refund()
            raise
        finally:
            events.close()

    def stream(
        self, asset: VideoAsset | AudioAsset, *, end: object, audio: bool, pictures: bool
    ) -> DecodedMediaStream:
        cut = _effective_end(end, asset._end)
        if not (audio or pictures):
            raise ValueError("a stream needs pictures, audio or both")
        budget, path = self._start(asset, f"{type(asset).__name__}.stream", retained=False)
        av = _av()
        return DecodedMediaStream(
            self,
            asset,
            av=av,
            budget=budget,
            events=lambda: _media_events(
                av,
                path,
                asset.kind,
                self._limits,
                budget,
                end=cut,
                retained=False,
                audio=audio,
                pictures=pictures,
            ),
        )

    def probe(self, asset: Asset) -> Probe:
        """Source header facts, cached on the handle. Never decodes a picture or sample."""
        _budget, path = self._start(asset, f"{type(asset).__name__}.probe", decode=False)
        if asset._probed is None:
            if asset.kind == "image":
                with _reading(asset, None, "image"):
                    asset._probed = _probe_image(path, self._limits)
            else:
                av = _av()
                with _reading(asset, av, asset.kind):
                    asset._probed = (
                        _probe_video(av, path, self._limits)
                        if asset.kind == "video"
                        else _probe_audio(av, path, self._limits)
                    )
        probed = asset._probed
        if isinstance(probed, ImageProbe):
            width, height = image_target_size(
                probed.source_width, probed.source_height, _image_preparation(asset)
            )
            return replace(probed, width=width, height=height)
        assert isinstance(probed, (VideoProbe, AudioProbe))
        if asset._end is not None and probed.source_duration > asset._end:
            return replace(probed, duration=asset._end)
        return probed

    def _acquire_stream(self) -> None:
        self._attempt.check_open("VideoAsset.stream")
        if self._attempt.components_active():
            raise CapabilityError(
                "media streaming is forbidden inside an active component-use scope",
                code="decode_in_component_scope",
            )
        with self._lock:
            if self._open_streams >= MAX_ACTIVE_MEDIA_STREAMS:
                raise CapabilityError(
                    f"an attempt admits at most {MAX_ACTIVE_MEDIA_STREAMS} active media streams",
                    code="media_stream_active",
                )
            self._open_streams += 1

    def _release_stream(self) -> None:
        with self._lock:
            self._open_streams -= 1

    def _start(
        self, asset: Asset, operation: str, *, retained: bool = True, decode: bool = True
    ) -> tuple[_Budget, Path]:
        self._attempt.check_open(operation)
        if self._attempt.components_active():
            raise CapabilityError(
                "media decode is forbidden inside an active component-use scope; read "
                "verified inputs before acquiring model components",
                code="decode_in_component_scope",
            )
        fields = [asset._input_id] if asset._input_id else []
        if asset._attempt != self._attempt.request_id:
            raise CapabilityError(
                f"asset {asset.ref!r} belongs to attempt {asset._attempt!r}",
                code="foreign_asset",
                fields=fields,
            )
        if asset._read_guard is not None:
            asset._read_guard()
        declared = asset._max_decoded_bytes  # 0: the field declares none; the machine's applies
        budget = _Budget(
            min(declared, self._limits.max_decoded_bytes)
            if declared > 0
            else self._limits.max_decoded_bytes,
            parent=self._budget if retained else None,
            scope="retained decoded" if retained else "streamed decoded",
            cumulative=retained,
        )
        asset._check_file()
        assert asset._local is not None
        return budget, asset._local


def _image_preparation(asset: Asset) -> ImagePreparation | None:
    preparation = asset._preparation
    return preparation if isinstance(preparation, ImagePreparation) else None


def _effective_end(end: object, prepared: Fraction | None) -> Fraction | None:
    if end is None:
        return prepared
    try:
        requested = seconds(end)
    except ValueError as exc:
        raise ValueError(f"end must be a positive number of seconds: {exc}") from None
    return requested if prepared is None else min(requested, prepared)


def _rate(fps: object) -> Fraction:
    try:
        return seconds(fps)
    except ValueError as exc:
        raise ValueError(f"fps must be a positive frame rate: {exc}") from None


def _collect_audio(
    audio_format: DecodedAudioFormat, events: Iterator[_Event]
) -> DecodedAudio | None:
    channels = [io.BytesIO() for _ in range(audio_format.channels)]
    start: Fraction | None = None
    count = 0
    sample_rate = audio_format.sample_rate
    names = audio_format.channel_names
    layout = audio_format.channel_layout
    for event in events:
        assert isinstance(event, DecodedAudioChunk)
        if start is None:
            start = event.start_time
            sample_rate, names, layout = (
                event.sample_rate,
                event.channel_names,
                event.channel_layout,
            )
            if len(channels) != event.channels:
                channels = [io.BytesIO() for _ in range(event.channels)]
        for channel, chunk in zip(channels, event.pcm_f32le, strict=True):
            channel.write(chunk)
        count += event.sample_count
    if start is None:
        return None
    return DecodedAudio(
        channels=len(channels),
        sample_count=count,
        sample_rate=sample_rate,
        channel_layout=layout,
        channel_names=names,
        pcm_f32le=tuple(channel.getvalue() for channel in channels),
        start_time=start,
    )


def _collect_video(
    events: Iterator[_Event], budget: _Budget, limits: DecodeLimits, fps: Fraction | None
) -> DecodedVideo:
    header = next(events)
    assert isinstance(header, DecodedMediaHeader) and header.video is not None
    frames: list[bytes] = []
    pts: list[int] = []
    durations: list[int] = []
    pcm = [io.BytesIO() for _ in range(header.audio.channels)] if header.audio else []
    audio_start: Fraction | None = None
    samples = 0
    origin: int | None = None
    slots = 0

    def keep(count: int) -> None:
        if len(frames) + count > limits.max_video_frames:
            raise MediaDecodeError(
                f"video crosses the {limits.max_video_frames}-frame limit",
                code="video_frames_limit",
            )

    for event in events:
        if isinstance(event, DecodedAudioChunk):
            if audio_start is None:
                audio_start = event.start_time
            for channel, chunk in zip(pcm, event.pcm_f32le, strict=True):
                channel.write(chunk)
            samples += event.sample_count
            continue
        assert isinstance(event, _Picture)
        if origin is None:
            origin = event.pts
        if fps is None:
            keep(1)
            frames.append(event.convert(budget).rgb)
            pts.append(event.pts)
            durations.append(event.duration)
            continue
        time_base = header.video.time_base
        start = (event.pts - origin) * time_base
        end = start + event.duration * time_base
        count = _half_up(end * fps) - slots
        if count <= 0:
            event.frame = None
            continue
        keep(count)
        rgb = event.convert(budget).rgb
        frames.extend([rgb] * count)
        slots += count

    if not frames:
        raise MediaDecodeError(f"video has no frame on the {fps}-fps clock", code="media_empty")
    assert origin is not None
    soundtrack = None
    if header.audio is not None and audio_start is not None:
        soundtrack = DecodedAudio(
            channels=header.audio.channels,
            sample_count=samples,
            sample_rate=header.audio.sample_rate,
            channel_layout=header.audio.channel_layout,
            channel_names=header.audio.channel_names,
            pcm_f32le=tuple(channel.getvalue() for channel in pcm),
            start_time=audio_start,
        )
    time_base = header.video.time_base
    if fps is not None:
        # The constant clock starts at the first frame; the soundtrack keeps its offset.
        if soundtrack is not None:
            soundtrack = replace(soundtrack, start_time=soundtrack.start_time - origin * time_base)
        time_base = 1 / fps
        pts = list(range(len(frames)))
        durations = [1] * len(frames)
    return DecodedVideo(
        width=header.video.width,
        height=header.video.height,
        frame_count=len(frames),
        frames_rgb=tuple(frames),
        frame_pts=tuple(pts),
        frame_durations=tuple(durations),
        time_base=time_base,
        pixel_aspect_ratio=header.video.pixel_aspect_ratio,
        soundtrack=soundtrack,
    )


def _stream_duration(av: Any, container: Any, stream: Any) -> Fraction:
    """A stream's presentation length from its header, else the container's (Matroska
    records only that), else from a packet scan without decoding (live WebM)."""
    time_base = _positive_fraction(stream.time_base, f"{stream.type} stream time base")
    if stream.duration:
        return int(stream.duration) * time_base
    if container.duration:
        return Fraction(int(container.duration), int(av.time_base))
    first: int | None = None
    last: int | None = None
    for packet in container.demux(stream):
        if packet.pts is None:
            continue
        start = int(packet.pts)
        end = start + int(packet.duration or 0)
        first = start if first is None else min(first, start)
        last = end if last is None else max(last, end)
    if first is None or last is None or last <= first:
        raise MediaDecodeError(
            f"{stream.type} stream has no measurable duration", code="media_clock"
        )
    return (last - first) * time_base


def _probe_video(av: Any, path: Path, limits: DecodeLimits) -> VideoProbe:
    with path.open("rb") as source, av.open(source, mode="r") as container:
        stream = _one_stream(list(container.streams.video), "video")
        soundtrack = _soundtrack(container)
        context = stream.codec_context
        width, height = int(context.width or 0), int(context.height or 0)
        _check_pixels(width, height, limits)
        matrix = (context.coded_side_data or {}).get("display_matrix")
        rotation = 0
        if matrix is not None:
            raw = bytes(matrix)
            found = (
                _PURE_DISPLAY_MATRICES.get(struct.unpack("=9i", raw)) if len(raw) == 36 else None
            )
            if found is None:
                raise MediaDecodeError(
                    "display matrix contains a flip, shear, scale, or non-quarter rotation",
                    code="media_rotation",
                )
            rotation = found
        ratio = stream.sample_aspect_ratio
        aspect = _positive_fraction(ratio, "video pixel aspect ratio") if ratio else Fraction(1)
        if rotation % 180:
            width, height, aspect = height, width, 1 / aspect
        rate = stream.average_rate
        audio_format = _stream_audio_format(soundtrack, limits) if soundtrack is not None else None
        duration = _stream_duration(av, container, stream)
    return VideoProbe(
        width=width,
        height=height,
        duration=duration,
        source_duration=duration,
        frame_rate=_positive_fraction(rate, "video average rate") if rate else None,
        pixel_aspect_ratio=aspect,
        soundtrack=audio_format,
    )


def _probe_audio(av: Any, path: Path, limits: DecodeLimits) -> AudioProbe:
    with path.open("rb") as source, av.open(source, mode="r") as container:
        stream = _one_stream(list(container.streams.audio), "audio")
        audio_format = _stream_audio_format(stream, limits)
        duration = _stream_duration(av, container, stream)
    return AudioProbe(
        duration=duration,
        source_duration=duration,
        channels=audio_format.channels,
        sample_rate=audio_format.sample_rate,
        channel_layout=audio_format.channel_layout,
    )


_SWAPPED_ORIENTATIONS = frozenset({5, 6, 7, 8})


@contextmanager
def _pillow() -> Iterator[None]:
    """Pillow's refusals as typed decode errors; filesystem failures pass through."""
    try:
        yield
    except MediaDecodeError:
        raise
    except DecompressionBombError as exc:
        raise MediaDecodeError(str(exc), code="pixels_per_frame_limit") from exc
    except OSError as exc:
        if exc.errno is not None:
            raise  # Filesystem failures retain the caller's materialization error boundary.
        raise MediaDecodeError(
            f"image could not be decoded: {_safe_error(exc)}", code="media_malformed"
        ) from exc
    except (ValueError, SyntaxError) as exc:
        raise MediaDecodeError(
            f"image could not be decoded: {_safe_error(exc)}", code="media_malformed"
        ) from exc


def _probe_image(path: Path, limits: DecodeLimits) -> ImageProbe:
    if path.stat().st_size > IMAGE_SOURCE_MAX_BYTES:
        raise MediaDecodeError(
            "image source exceeds preparation byte ceiling", code="image_source_bytes"
        )
    with _pillow(), pil_open(path) as header:
        width, height = header.size
        _check_pixels(width, height, limits)
        if getattr(header, "n_frames", 1) != 1:
            raise MediaDecodeError(
                "image asset contains more than one frame", code="animated_image"
            )
        # The base reader parses header metadata only; a PNG `eXIf` after its pixels
        # would need a decode and is not consulted here.
        orientation = PILImage.getexif(header).get(0x0112, 1)
    if orientation in _SWAPPED_ORIENTATIONS:
        width, height = height, width
    return ImageProbe(width=width, height=height, source_width=width, source_height=height)


def _av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise DecoderUnavailable("media decode requires cozy-runtime[media] (PyAV)") from exc
    return av


def require_decoder_dependency() -> None:
    """Build-time closure check for a surface that names a video or audio input."""
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
    path: Path, limits: DecodeLimits, preparation: ImagePreparation | None
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


def _image(
    path: Path, limits: DecodeLimits, budget: _Budget, preparation: ImagePreparation | None
) -> PILImage:
    """Decode a still image with Pillow; CLI preparation and Runtime use this one core.

    Resource ceilings are application policy. Pillow owns image formats, EXIF
    orientation (including mirrors), RGB conversion and resampling.
    """
    with _pillow():
        source_geometry = image_source_geometry(path, limits, preparation)
        with pil_open(path) as image:
            if image.size != source_geometry:
                raise MediaDecodeError("image header geometry changed", code="media_geometry")
            if getattr(image, "n_frames", 1) != 1:
                raise MediaDecodeError(
                    "image asset contains more than one frame", code="animated_image"
                )
            width, height = image_target_size(*source_geometry, preparation)
            budget.claim(width * height * 3, "image RGB")
            # in_place avoids an extra full-size copy for the common upright case.
            exif_transpose(image, in_place=True)
            rgb = image.convert("RGB")
            size = image_target_size(*rgb.size, preparation)
            if rgb.size != size:
                rgb = rgb.resize(size, Resampling.LANCZOS)
            return rgb


def decode_image_path(
    path: Path, limits: DecodeLimits, budget: _Budget, preparation: ImagePreparation | None = None
) -> DecodedImage:
    """The `ImageAsset.read` pixels as packed RGB8, for local preparation."""
    image = _image(path, limits, budget, preparation)
    return DecodedImage(image.width, image.height, image.tobytes())


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
    if not time_base:
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


_PURE_DISPLAY_MATRICES = {
    (65536, 0, 0, 0, 65536, 0, 0, 0, 1073741824): 0,
    (0, -65536, 0, 65536, 0, 0, 0, 0, 1073741824): 90,
    (0, 65536, 0, -65536, 0, 0, 0, 0, 1073741824): -90,
    (-65536, 0, 0, 0, -65536, 0, 0, 0, 1073741824): 180,
}


def _display_rotation(frame: Any) -> int:
    try:
        # Both PyAV 18 and 19 support typed lookup; iteration changed from values
        # to keys in 19 when SideDataContainer adopted the Mapping contract.
        matrix = frame.side_data.get("DISPLAYMATRIX")
    except ValueError as exc:
        # PyAV 18.1 cannot represent some FFmpeg side-data enums (notably EXIF display
        # transforms). Accepting only frame.rotation would silently discard mirror state.
        raise MediaDecodeError(
            "display transform cannot be inspected without losing orientation",
            code="media_rotation",
        ) from exc
    if matrix is None:
        if int(frame.rotation):
            raise MediaDecodeError(
                "frame reports rotation without an inspectable display matrix",
                code="media_rotation",
            )
        return 0
    raw = bytes(matrix)
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
        # PyAV has exposed AVRatios as both ``fractions.Fraction`` and a small
        # rational object with integer numerator/denominator attributes across
        # supported releases.  Preserve that ratio exactly instead of routing
        # it through float conversion.
        numerator = getattr(value, "numerator", None)
        denominator = getattr(value, "denominator", None)
        if isinstance(numerator, Integral) and isinstance(denominator, Integral):
            return Fraction(int(numerator), int(denominator))
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


def _attribute(exc: AuthorError, asset: Asset) -> None:
    if not exc.fields and asset._input_id:
        exc.fields = (asset._input_id,)
