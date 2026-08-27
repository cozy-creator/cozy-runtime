"""The one bounded input-media decoder: verified bytes in, immutable values out.

There is deliberately no path, URL, fetch, resize, model-normalization, or output-codec
surface here. PyAV is a lazy member of the ``media`` extra; the base author package does
not import it.
"""

from __future__ import annotations

import io
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, Protocol, TypeVar

from cozy_runtime.author._assets import AudioAsset, ImageAsset, VideoAsset
from cozy_runtime.author._errors import CapabilityError, InvalidRequest


class _Attempt(Protocol):
    request_id: str

    def check_open(self, what: str) -> None: ...


@dataclass(frozen=True, slots=True)
class DecodeLimits:
    """Runtime codec-safety ceilings. A field's decoded-byte bound can only lower these."""

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


# Structural codec ceilings, not endpoint policy. Duration and reference-count limits stay
# at the endpoint; a field's mandatory max_decoded_bytes is normally the tighter bound.
DEFAULT_DECODE_LIMITS = DecodeLimits(
    max_decoded_bytes=2 * 1024**3,
    max_pixels_per_frame=8192 * 8192,
    max_video_frames=3600,
    max_audio_channels=8,
    max_audio_samples=48_000 * 60 * 10,
)


class DecoderUnavailable(CapabilityError):
    default_code = "decoder_unavailable"


class MediaDecodeError(InvalidRequest):
    default_code = "media_decode"


@dataclass(frozen=True, slots=True)
class DecodedImage:
    width: int
    height: int
    rgb: bytes


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """One little-endian float32 byte string per channel, on the native sample clock."""

    channels: int
    sample_count: int
    sample_rate: int
    channel_layout: str
    channel_names: tuple[str, ...]
    pcm_f32le: tuple[bytes, ...]

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
    average_rate: Fraction
    pixel_aspect_ratio: Fraction
    soundtrack: DecodedAudio | None

    @property
    def duration(self) -> Fraction:
        return (self.frame_pts[-1] + self.frame_durations[-1] - self.frame_pts[0]) * self.time_base


@dataclass(slots=True)
class _Budget:
    limit: int
    parent: _Budget | None = None
    used: int = 0

    def claim(self, size: int, what: str) -> None:
        if size < 0 or self.used + size > self.limit:
            raise MediaDecodeError(
                f"decoded {what} would cross the {self.limit}-byte expansion limit",
                code="decoded_bytes_limit",
            )
        if self.parent is not None:
            self.parent.claim(size, "attempt media")
        self.used += size


class _AudioAccumulator:
    """Validate and retain native-clock audio without a NumPy-sized second authority."""

    def __init__(self, av: Any, stream: Any, limits: DecodeLimits, budget: _Budget) -> None:
        self._av = av
        self._limits = limits
        self._budget = budget
        self._resampler: Any | None = None
        self._channels: list[io.BytesIO] = []
        self._last_time: Fraction | None = None
        self.channels = 0
        self.sample_rate = 0
        self.channel_layout = ""
        self.channel_names: tuple[str, ...] = ()
        self.sample_count = 0

        context = stream.codec_context
        _set_decoder_option(context, "err_detect", "explode")
        _set_decoder_option(context, "max_samples", limits.max_audio_samples)
        header_channels = int(context.channels or 0)
        if header_channels > limits.max_audio_channels:
            raise MediaDecodeError(
                f"audio has {header_channels} channels; limit is {limits.max_audio_channels}",
                code="audio_channels_limit",
            )

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
        self._channels = [io.BytesIO() for _ in range(self.channels)]
        self._resampler = self._av.AudioResampler(
            format="fltp", layout=self.channel_layout, rate=self.sample_rate
        )

    def append(self, frame: Any) -> None:
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

        if frame.pts is None or frame.time_base is None:
            raise MediaDecodeError("audio frame has no exact presentation time", code="media_clock")
        presentation_time = _fraction(frame.pts, "audio frame PTS") * _positive_fraction(
            frame.time_base, "audio frame time base"
        )
        if self._last_time is not None and presentation_time <= self._last_time:
            raise MediaDecodeError("audio PTS is not strictly monotone", code="media_clock")
        self._last_time = presentation_time

        valid_samples = _valid_audio_samples(frame, samples, self.sample_rate)
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
                    self._channels[channel].write(chunk)
                remaining -= take
        if remaining:
            raise MediaDecodeError(
                "audio conversion returned fewer samples than its clock", code="media_truncated"
            )

        self.sample_count += valid_samples

    def finish(self) -> DecodedAudio:
        if self.sample_count == 0 or self._resampler is None:
            raise MediaDecodeError("audio stream decoded no samples", code="media_empty")
        if self._resampler.resample(None):
            raise MediaDecodeError(
                "native-rate audio conversion left unclocked samples", code="media_clock"
            )
        return DecodedAudio(
            channels=self.channels,
            sample_count=self.sample_count,
            sample_rate=self.sample_rate,
            channel_layout=self.channel_layout,
            channel_names=self.channel_names,
            pcm_f32le=tuple(channel.getvalue() for channel in self._channels),
        )


class _VideoAccumulator:
    def __init__(self, stream: Any, limits: DecodeLimits, budget: _Budget) -> None:
        self._limits = limits
        self._budget = budget
        self._chunks: list[bytes] = []
        self._declared_frames = 0
        self._last_pts: int | None = None
        self._rotation: int | None = None
        self.frame_pts: list[int] = []
        self.frame_durations: list[int] = []
        self.width = 0
        self.height = 0
        self.time_base: Fraction | None = None
        self.average_rate = _positive_fraction(stream.average_rate, "video average rate")
        ratio = stream.sample_aspect_ratio
        self.pixel_aspect_ratio = (
            _positive_fraction(ratio, "video pixel aspect ratio")
            if ratio is not None
            else Fraction(1, 1)
        )

        context = stream.codec_context
        _set_decoder_option(context, "err_detect", "explode")
        _set_decoder_option(context, "max_pixels", limits.max_pixels_per_frame)
        width, height = int(context.width or 0), int(context.height or 0)
        if width > 0 and height > 0:
            _check_pixels(width, height, limits)
        declared = int(stream.frames or 0)
        self._declared_frames = declared
        if declared > limits.max_video_frames:
            raise MediaDecodeError(
                f"video metadata declares {declared} frames; limit is {limits.max_video_frames}",
                code="video_frames_limit",
            )
        if declared and width and height and declared * width * height * 3 > budget.limit:
            raise MediaDecodeError(
                "video metadata exceeds the decoded-byte expansion limit",
                code="decoded_bytes_limit",
            )

    def append(self, frame: Any) -> None:
        if frame.is_corrupt:
            raise MediaDecodeError("video decoder marked a frame corrupt", code="media_corrupt")
        if len(self._chunks) >= self._limits.max_video_frames:
            raise MediaDecodeError(
                f"video crosses the {self._limits.max_video_frames}-frame limit",
                code="video_frames_limit",
            )
        _check_pixels(int(frame.width), int(frame.height), self._limits)

        pts, time_base = frame.pts, frame.time_base
        if pts is None or time_base is None:
            raise MediaDecodeError("video frame has no exact PTS/time base", code="media_clock")
        exact_base = _positive_fraction(time_base, "video frame time base")
        if self.time_base is None:
            self.time_base = exact_base
        elif exact_base != self.time_base:
            raise MediaDecodeError("video time base changes between frames", code="media_clock")
        exact_pts = int(pts)
        if self._last_pts is not None and exact_pts <= self._last_pts:
            raise MediaDecodeError("video PTS is not strictly monotone", code="media_clock")
        duration = int(frame.duration or 0)
        if duration <= 0:
            raise MediaDecodeError("video frame has no exact positive duration", code="media_clock")

        rotation = _display_rotation(frame)
        if self._rotation is None:
            self._rotation = rotation
            if rotation % 180:
                self.pixel_aspect_ratio = 1 / self.pixel_aspect_ratio
        elif rotation != self._rotation:
            raise MediaDecodeError("display rotation changes between frames", code="media_rotation")

        self._budget.claim(int(frame.width) * int(frame.height) * 3, "video RGB")
        rgb = _tight_rgb(frame)
        rgb, width, height = _rotate_rgb(rgb, int(frame.width), int(frame.height), rotation)
        if self.width and (width != self.width or height != self.height):
            raise MediaDecodeError(
                "video frame geometry changes after display rotation", code="media_geometry"
            )
        self.width, self.height = width, height
        self._chunks.append(rgb)
        self.frame_pts.append(exact_pts)
        self.frame_durations.append(duration)
        self._last_pts = exact_pts

    def finish(self, soundtrack: DecodedAudio | None) -> DecodedVideo:
        if not self._chunks or self.time_base is None:
            raise MediaDecodeError("video stream decoded no frames", code="media_empty")
        if self._declared_frames and len(self._chunks) != self._declared_frames:
            raise MediaDecodeError(
                f"video decoded {len(self._chunks)} of {self._declared_frames} declared frames",
                code="media_truncated",
            )
        return DecodedVideo(
            width=self.width,
            height=self.height,
            frame_count=len(self._chunks),
            frames_rgb=tuple(self._chunks),
            frame_pts=tuple(self.frame_pts),
            frame_durations=tuple(self.frame_durations),
            time_base=self.time_base,
            average_rate=self.average_rate,
            pixel_aspect_ratio=self.pixel_aspect_ratio,
            soundtrack=soundtrack,
        )


AssetT = TypeVar("AssetT", ImageAsset, AudioAsset, VideoAsset)


class MediaDecoder:
    """Attempt-bound decoder injected into handlers that explicitly declare it."""

    __slots__ = ("_active", "_attempt", "_budget", "_limits")

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

    def decode_image(self, asset: ImageAsset) -> DecodedImage:
        av, budget, data = self._start(asset, ImageAsset, "decode_image")
        try:
            jpeg_rotation = _jpeg_rotation(data) if data.startswith(b"\xff\xd8") else None
            with av.open(io.BytesIO(data), mode="r") as container:
                streams = list(container.streams.video)
                if len(streams) != 1:
                    _required_stream("image", len(streams))
                stream = streams[0]
                _set_decoder_option(stream.codec_context, "err_detect", "explode")
                _set_decoder_option(
                    stream.codec_context, "max_pixels", self._limits.max_pixels_per_frame
                )
                width = int(stream.codec_context.width or 0)
                height = int(stream.codec_context.height or 0)
                if width > 0 and height > 0:
                    _check_pixels(width, height, self._limits)
                decoded: DecodedImage | None = None
                for frame in container.decode(stream):
                    if decoded is not None:
                        raise MediaDecodeError(
                            "image asset contains more than one frame", code="animated_image"
                        )
                    decoded = _decode_image_frame(
                        frame, self._limits, budget, rotation=jpeg_rotation
                    )
                if decoded is None:
                    raise MediaDecodeError("image stream decoded no frame", code="media_empty")
                return decoded
        except MediaDecodeError as exc:
            _attribute(exc, asset)
            raise
        except Exception as exc:
            raise MediaDecodeError(
                f"image container could not be decoded: {_safe_error(exc)}",
                code="media_malformed",
                fields=[asset._input_id] if asset._input_id else (),
            ) from exc

    def decode_audio(self, asset: AudioAsset) -> DecodedAudio:
        av, budget, data = self._start(asset, AudioAsset, "decode_audio")
        try:
            with av.open(io.BytesIO(data), mode="r") as container:
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
        except Exception as exc:
            raise MediaDecodeError(
                f"audio container could not be decoded: {_safe_error(exc)}",
                code="media_malformed",
                fields=[asset._input_id] if asset._input_id else (),
            ) from exc

    def decode_video(self, asset: VideoAsset) -> DecodedVideo:
        av, budget, data = self._start(asset, VideoAsset, "decode_video")
        try:
            with av.open(io.BytesIO(data), mode="r") as container:
                video_streams = list(container.streams.video)
                audio_streams = list(container.streams.audio)
                if len(video_streams) != 1:
                    _required_stream("video", len(video_streams))
                if len(audio_streams) > 1:
                    raise MediaDecodeError(
                        "video asset has more than one soundtrack stream",
                        code="multiple_audio_streams",
                    )
                video = _VideoAccumulator(video_streams[0], self._limits, budget)
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
        except Exception as exc:
            raise MediaDecodeError(
                f"video container could not be decoded: {_safe_error(exc)}",
                code="media_malformed",
                fields=[asset._input_id] if asset._input_id else (),
            ) from exc

    def _start(
        self, asset: AssetT, expected: type[AssetT], operation: str
    ) -> tuple[Any, _Budget, bytes]:
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
        av = _av()
        budget = _Budget(
            min(asset._max_decoded_bytes, self._limits.max_decoded_bytes),
            parent=self._budget,
        )
        return av, budget, asset.read_bytes()


def _av() -> Any:
    try:
        import av
    except ImportError as exc:
        raise DecoderUnavailable("media decode requires cozy-runtime[media] (PyAV)") from exc
    return av


def _decode_image_frame(
    frame: Any,
    limits: DecodeLimits,
    budget: _Budget,
    *,
    rotation: int | None,
) -> DecodedImage:
    if frame.is_corrupt:
        raise MediaDecodeError("image decoder marked the frame corrupt", code="media_corrupt")
    _check_pixels(int(frame.width), int(frame.height), limits)
    rotation = _display_rotation(frame) if rotation is None else rotation
    budget.claim(int(frame.width) * int(frame.height) * 3, "image RGB")
    rgb = _tight_rgb(frame)
    rgb, width, height = _rotate_rgb(rgb, int(frame.width), int(frame.height), rotation)
    return DecodedImage(width=width, height=height, rgb=rgb)


def _tight_rgb(frame: Any) -> bytes:
    rgb = frame.reformat(format="rgb24")
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
    out_width, out_height = (height, width) if rotation % 180 else (width, height)
    out = bytearray(len(rgb))
    for y in range(height):
        for x in range(width):
            if rotation == 90:
                dx, dy = y, width - 1 - x
            elif rotation == -90:
                dx, dy = height - 1 - y, x
            else:
                dx, dy = width - 1 - x, height - 1 - y
            source = (y * width + x) * 3
            target = (dy * out_width + dx) * 3
            out[target : target + 3] = rgb[source : source + 3]
    return bytes(out), out_width, out_height


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
    if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<f", data)):
        raise MediaDecodeError("audio PCM contains NaN or infinity", code="audio_format")


def _jpeg_rotation(data: bytes) -> int:
    """Read JPEG's EXIF orientation without an imaging library.

    PyAV 18.1 exposes its normalized rotation but cannot expose the raw EXIF side-data enum;
    rotation alone conflates pure rotations with mirrored transforms. The tiny TIFF reader
    below reads only IFD0's orientation tag and refuses the mirror cases explicitly.
    """
    offset = 2
    found: list[int] = []
    while offset < len(data):
        if data[offset] != 0xFF:
            break
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            raise MediaDecodeError("JPEG marker is truncated", code="media_malformed")
        marker = data[offset]
        offset += 1
        if marker == 0xD9:
            break
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA:
            break
        if offset + 2 > len(data):
            raise MediaDecodeError("JPEG segment length is truncated", code="media_malformed")
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            raise MediaDecodeError("JPEG segment crosses the input", code="media_malformed")
        payload = data[offset + 2 : offset + length]
        offset += length
        if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            orientation = _tiff_orientation(payload[6:])
            if orientation is not None:
                found.append(orientation)
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


def _attribute(exc: MediaDecodeError, asset: ImageAsset | AudioAsset | VideoAsset) -> None:
    if not exc.fields and asset._input_id:
        exc.fields = (asset._input_id,)
