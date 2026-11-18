"""cr-017's codec plane, the H3-first slice: FLAC audio and the muxed H.264+AAC mp4.

Encoders come from the `media` extra (PyAV — FFmpeg bundled as an abi3 wheel, which is
what the wheel-only packaging rule admits; a system ffmpeg binary never was). The base
package stays free of it: `av` is imported inside a function, and an absent wheel is the
same typed `encoder_unavailable` refusal this release always had.

Determinism is a pinned property, not a hope: every encoder runs with fixed settings, a
fixed thread count and `fflags +bitexact` on the container, so one (input, wheel version,
settings, x264 threads) tuple produces one byte sequence, held across machines by the
locked wheel. x264's thread count is `x264_threads()`: the CPUs this process may use, under
`X264_THREAD_CEILING`. Frame-threaded x264 restricts motion search to what earlier threads
have finished, so the bitstream — and therefore the DECODED pixels of a lossy codec — is a
function of the thread count. The ceiling is what makes that a plan fact rather than
drift: every qualified pod has at least that many vCPUs and so encodes identically; a
smaller host (a laptop, a CI runner) encodes with fewer threads and a different digest.
Any test that pins a container digest is pinned to a thread count and must be updated
deliberately with it. The PRE-ENCODE tensors remain the cross-platform bit-grade
artifacts (H3 banks those digests itself); the container digest is not an output identity.

Tensor input is duck-typed — a torch tensor is drained through `.detach().cpu().numpy()`
and an ndarray through the buffer protocol — so this module imports neither torch nor
numpy. The snapshot is a zero-copy view of the drained host array; the ONE host copy is
the registration's spool write, which completes before the save returns, so the output
bytes are source-independent and mutating the author's tensor afterwards changes nothing.
"""

from __future__ import annotations

import io
import math
import os
import queue
import struct
import threading
from array import array
from collections.abc import Buffer, Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import accumulate, groupby, pairwise
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, BinaryIO, TypedDict, cast

from PIL import Image

import cozy_runtime.author._color as _color
from cozy_runtime._cpus import cpus
from cozy_runtime.author._errors import OutputError

if TYPE_CHECKING:
    import numpy as np
    from av.audio.codeccontext import AudioCodecContext
    from av.audio.frame import AudioFrame
    from av.packet import Packet
    from av.stream import Stream
    from av.video.frame import VideoFrame
    from torch import Tensor

    from cozy_runtime.canonical_json import JsonValue

#: What frames and waveforms save from: a torch tensor (on any device) or a numpy array.
type Samples = Tensor | np.ndarray[tuple[int, ...], np.dtype[np.generic[object]]]

#: Raw-snapshot ceiling as a multiple of the attempt's output ceiling. H3's largest launch
#: product (124 frames of 1344x768 RGB) is ~1.45x the 256 MiB output cap; 16x admits an
#: order of magnitude of headroom while refusing a runaway [T] dimension typed, before the
#: snapshot copy — not the OOM killer — is what stops it.
SNAPSHOT_FACTOR = 16

#: Pinned x264 settings. crf 17 turned low-contrast areas into flat rectangles and lost grain
#: (0.94x of the source; crf 14 keeps it, for 40% more bytes and the same CPU). veryfast keeps
#: the CPU tail small on pods; medium cost 5x the CPU and looked no better. Changing either
#: changes every banked container digest, so a change here is deliberate, never drift.
_X264 = {"crf": "14", "preset": "veryfast"}

#: The most x264 frame threads any host gets. It is the smallest vCPU count of a qualified
#: H3 pod class (h3a-017), so every pod encodes with the same count and the same bytes; it
#: also bounds x264's per-thread frame contexts on a host whose CPU count is large. Raising
#: it changes every pod's container digest — deliberate, never drift.
X264_THREAD_CEILING = 8

#: Pinned AAC bitrate — set explicitly so a wheel-default change cannot move the output.
_AAC_BIT_RATE = 192_000

_FLAC_CHUNK = 4096
AAC_FRAME_SAMPLES = 1024

# Delayed MP4 metadata must not turn a missing initial track into an unbounded
# native muxer queue. Payload bytes and packet count bound data and per-packet metadata.
_MP4_PREAMBLE_BYTES = 8 << 20
_MP4_PREAMBLE_PACKETS = 1024

#: Converted frames waiting for the streaming encoder's x264 thread (~1.5 MB each at H3's
#: 1344x768): enough to absorb a decode chunk while the caller converts the next one.
_ENCODE_QUEUE_FRAMES = 16


def x264_threads() -> int:
    """The x264 thread count on THIS host: the CPUs this process may use, under the ceiling."""
    return min(X264_THREAD_CEILING, cpus())


@dataclass(frozen=True, slots=True)
class Planar:
    """An immutable CPU snapshot of a tensor-like: a raw byte buffer (bytes, or a zero-copy
    view of a host array the registration writes out) plus the shape it claims."""

    data: Buffer
    shape: tuple[int, ...]

    def view(self) -> memoryview:
        return memoryview(self.data).cast("B")

    def __len__(self) -> int:
        return self.view().nbytes


def _av() -> ModuleType:
    try:
        import av
    except ImportError:
        raise OutputError(
            "audio/video encode needs the `media` extra (PyAV): install "
            "cozy-runtime[media]. Without it the codec plane does not exist in this venv",
            code="encoder_unavailable",
        ) from None
    return av


def require_encoder() -> None:
    """The capability confession comes FIRST: on an encoder-less venv every save_audio/
    save_video refusal is `encoder_unavailable`, never an input error that implies the
    capability exists."""
    _av()


def _image(width: int, height: int, pixels: bytes, mode: str) -> Image.Image:
    return Image.frombytes(mode, (width, height), pixels)


def encode_png(width: int, height: int, pixels: bytes, mode: str = "RGB") -> bytes:
    """Encode exact RGB or RGBA pixels losslessly at zlib's fastest level.

    The worker encodes a registered frame at the attempt's tail, before its outcome. On
    1344x768 and 1024x1024 frames, level 9 with `optimize` took 1.1-2.0 s here and 2.9 s
    on an H100 pod to save 10-16% over level 1's ~0.12 s: a second inference tail.
    """

    encoded = io.BytesIO()
    _image(width, height, pixels, mode).save(encoded, format="PNG", compress_level=1)
    body = encoded.getvalue()
    if body[:8] != b"\x89PNG\r\n\x1a\n":
        raise OutputError("the lossless PNG encoder returned a malformed container", code="encode")
    return body


def encode_webp(width: int, height: int, pixels: bytes, mode: str = "RGB") -> bytes:
    """Encode exact RGB or RGBA pixels as lossless WebP at its fastest effort.

    Lossless and exact at every setting: effort trades only bytes for time. On three
    1024px SDXL outputs, method 5 at quality 100 took 1.15 s for 1.07 MB; method 0 at
    quality 50 took 0.12 s for 1.20 MB. Packaging is on every image run's critical path.
    """

    encoded = io.BytesIO()
    _image(width, height, pixels, mode).save(
        encoded, format="WEBP", lossless=True, method=0, quality=50, exact=True
    )
    body = encoded.getvalue()
    if len(body) < 12 or body[:4] != b"RIFF" or body[8:12] != b"WEBP":
        raise OutputError("the lossless WebP encoder returned a malformed container", code="encode")
    return body


def _drain(obj: object, what: str) -> tuple[object, str]:
    """Duck-typed to an ndarray-like on the CPU; returns (ndarray-like, dtype name)."""
    if hasattr(obj, "detach"):
        obj = obj.detach()
    if hasattr(obj, "cpu"):
        obj = obj.cpu()
    if hasattr(obj, "contiguous"):
        obj = obj.contiguous()
    if hasattr(obj, "numpy"):
        obj = obj.numpy()
    if not (hasattr(obj, "tobytes") and hasattr(obj, "shape") and hasattr(obj, "dtype")):
        raise OutputError(
            f"cannot encode {type(obj).__name__} as {what}: pass a tensor or ndarray "
            "(anything with shape/dtype/tobytes, or a torch tensor)",
            code="frame_type",
        )
    return obj, str(obj.dtype)


def snapshot_video(frames: object, max_output_bytes: int) -> Planar:
    """[T, H, W, 3] uint8, cut to even H and W (yuv420p), snapshot-bounded. Refusals are typed
    at registration, never at encode time."""
    arr, dtype = _drain(frames, "video frames")
    if "uint8" not in dtype:
        raise OutputError(f"video frames dtype {dtype} is not uint8", code="frame_dtype")
    shape = tuple(int(d) for d in arr.shape)  # type: ignore[attr-defined]
    if len(shape) != 4 or shape[3] != 3 or shape[0] < 1:
        raise OutputError(
            f"video frames shape {shape} is not [T, H, W, 3] with T >= 1", code="frame_shape"
        )
    t, h, w, _ = shape
    if h % 2 or w % 2:
        h, w = h - h % 2, w - w % 2
        arr = arr[:, :h, :w]  # type: ignore[index]
        shape = (t, h, w, 3)
    limit = SNAPSHOT_FACTOR * max_output_bytes
    if t * h * w * 3 > limit:
        raise OutputError(
            f"video snapshot is {t * h * w * 3} bytes, over the {limit} snapshot bound "
            f"({SNAPSHOT_FACTOR}x the attempt output ceiling)",
            code="output_too_large",
        )
    view = memoryview(arr)  # type: ignore[arg-type]
    if not view.c_contiguous:
        view = memoryview(arr.copy())  # type: ignore[attr-defined]
    data = view.cast("B")
    if data.nbytes != t * h * w * 3:
        raise OutputError(
            f"video frames yielded {data.nbytes} bytes for shape {shape}", code="frame_shape"
        )
    return Planar(data, shape)


def even_rgb(rgb: bytes, width: int, height: int) -> bytes:
    """An RGB8 frame without its last row or column where a side is odd, as H.264 needs."""
    if not width % 2:
        return rgb if not height % 2 else rgb[: width * (height - 1) * 3]
    image = Image.frombuffer("RGB", (width, height), rgb, "raw", "RGB", 0, 1)
    return image.crop((0, 0, width - 1, height - height % 2)).tobytes()


def snapshot_audio(waveform: object, max_output_bytes: int) -> tuple[Planar, int]:
    """[S], [1, S] or [2, S] float32; non-finite refuses; excursions past [-1, 1] are
    clamped and counted (the caller records the adjustment row). Returns (planar, clipped):
    planar.shape is (channels, samples) and planar.data is channel-major clamped float32."""
    arr, dtype = _drain(waveform, "audio waveform")
    if "float32" not in dtype:
        raise OutputError(
            f"audio dtype {dtype} is not float32 (convert before saving)", code="audio_dtype"
        )
    shape = tuple(int(d) for d in arr.shape)  # type: ignore[attr-defined]
    if len(shape) == 1:
        channels, samples = 1, shape[0]
    elif len(shape) == 2 and shape[0] in (1, 2):
        channels, samples = shape[0], shape[1]
    else:
        raise OutputError(f"audio shape {shape} is not [S], [1, S] or [2, S]", code="audio_shape")
    if samples < 1:
        raise OutputError("audio waveform is empty", code="audio_shape")
    limit = SNAPSHOT_FACTOR * max_output_bytes
    if channels * samples * 4 > limit:
        raise OutputError(
            f"audio snapshot is {channels * samples * 4} bytes, over the {limit} snapshot "
            f"bound ({SNAPSHOT_FACTOR}x the attempt output ceiling)",
            code="output_too_large",
        )
    floats = array("f")
    floats.frombytes(bytes(arr.tobytes()))  # type: ignore[attr-defined]
    clipped = 0
    for i, value in enumerate(floats):
        if not math.isfinite(value):
            raise OutputError(
                f"audio sample {i} is not finite ({value!r}) — a NaN/inf waveform is a "
                "corrupt product, never something to clamp quiet",
                code="audio_nonfinite",
            )
        if value > 1.0:
            floats[i] = 1.0
            clipped += 1
        elif value < -1.0:
            floats[i] = -1.0
            clipped += 1
    return Planar(floats.tobytes(), (channels, samples)), clipped


def audio_chunk(pcm_f32le: tuple[bytes, ...], sample_count: int) -> tuple[Planar, int]:
    """Validate and clamp one already-planar decoded chunk without joining a timeline."""
    channels = len(pcm_f32le)
    if channels not in (1, 2) or sample_count <= 0:
        raise OutputError(
            f"streaming audio shape ({channels}, {sample_count}) is not mono/stereo PCM",
            code="audio_shape",
        )
    if any(len(channel) != sample_count * 4 for channel in pcm_f32le):
        raise OutputError(
            "streaming audio plane length disagrees with its sample count", code="audio_shape"
        )
    floats = array("f")
    floats.frombytes(b"".join(pcm_f32le))
    # The whole chunk in C first: finite (see `_require_finite_f32`) and inside [-1, 1].
    if math.isfinite(sum(floats)) and max(floats) <= 1.0 and min(floats) >= -1.0:
        return Planar(floats.tobytes(), (channels, sample_count)), 0
    clipped = 0
    for index, value in enumerate(floats):
        if not math.isfinite(value):
            raise OutputError(
                f"audio sample {index} is not finite ({value!r})",
                code="audio_nonfinite",
            )
        if value > 1.0:
            floats[index] = 1.0
            clipped += 1
        elif value < -1.0:
            floats[index] = -1.0
            clipped += 1
    return Planar(floats.tobytes(), (channels, sample_count)), clipped


def _audio_frames(
    av: ModuleType,
    planar: Planar,
    sample_rate: int,
    fmt: str,
    chunk: int,
    *,
    start_pts: int = 0,
) -> Iterator[AudioFrame]:
    """Yield AudioFrames of `chunk` samples in `fmt` (packed), pts in sample units."""
    channels, samples = planar.shape
    layout = "mono" if channels == 1 else "stereo"
    if fmt == "s16":
        floats = array("f")
        floats.frombytes(planar.data)
        pcm = array("h", (round(x * 32767.0) for x in floats))
        if channels == 2:
            inter = array("h", bytes(len(pcm) * 2))
            inter[0::2] = pcm[:samples]
            inter[1::2] = pcm[samples:]
            pcm = inter
        raw, unit = memoryview(pcm.tobytes()), 2 * channels
    else:  # flt
        if channels == 2:
            floats = array("f")
            floats.frombytes(planar.data)
            inter_f = array("f", bytes(len(floats) * 4))
            inter_f[0::2] = floats[:samples]
            inter_f[1::2] = floats[samples:]
            raw, unit = memoryview(inter_f.tobytes()), 4 * channels
        else:
            raw, unit = planar.view(), 4 * channels

    def gen() -> Iterator[AudioFrame]:
        pts = start_pts
        for off in range(0, samples, chunk):
            n = min(chunk, samples - off)
            frame = av.AudioFrame(format=fmt, layout=layout, samples=n)
            frame.sample_rate = sample_rate
            frame.pts = pts
            frame.time_base = Fraction(1, sample_rate)
            frame.planes[0].update(raw[off * unit : (off + n) * unit])
            pts += n
            yield frame

    return gen()


def encode_flac(planar: Planar, sample_rate: int) -> bytes:
    av = _av()
    buf = io.BytesIO()
    with av.open(buf, "w", format="flac", options={"fflags": "+bitexact"}) as container:
        stream = container.add_stream("flac", rate=sample_rate)
        codec = stream.codec_context
        codec.thread_count = 1
        codec.layout = "mono" if planar.shape[0] == 1 else "stereo"
        codec.format = "s16"
        for frame in _audio_frames(av, planar, sample_rate, "s16", _FLAC_CHUNK):
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return buf.getvalue()


def keyframes(interval: object) -> int | None:
    """`keyframe_interval` as the encoder takes it: None (its default) or frames between keyframes,
    1 for every frame a keyframe, so any frame shows at once when scrubbed."""
    if interval is None:
        return None
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise OutputError(
            f"keyframe_interval {interval!r} is not a whole number of frames, 1 or more",
            code="keyframe_interval",
        )
    return interval


def encode_mp4(
    video: Planar,
    fps: float,
    audio: Planar | None,
    sample_rate: int | None,
    keyframe_interval: int | None = None,
) -> bytes:
    av = _av()
    t, h, w, _ = video.shape
    rate = Fraction(fps).limit_denominator(65535)
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4", options={"fflags": "+bitexact"}) as container:
        vstream = container.add_stream("libx264", rate=rate)
        vstream.width = w
        vstream.height = h
        vstream.pix_fmt = "yuv420p"
        vcodec = vstream.codec_context
        vcodec.thread_count = x264_threads()
        if keyframe_interval is not None:
            vcodec.gop_size = keyframe_interval
        vcodec.options = dict(_X264)

        astream = None
        if audio is not None:
            assert sample_rate is not None  # validated at the save_* boundary
            astream = container.add_stream("aac", rate=sample_rate)
            acodec = astream.codec_context
            acodec.thread_count = 1
            acodec.layout = "mono" if audio.shape[0] == 1 else "stereo"
            acodec.format = "fltp"
            acodec.bit_rate = _AAC_BIT_RATE

        stride = h * w * 3
        pixels = video.view()
        for index in range(t):
            frame = av.VideoFrame(w, h, "rgb24")
            plane = frame.planes[0]
            raw = pixels[index * stride : (index + 1) * stride]
            if plane.line_size == w * 3:
                plane.update(raw)
            else:
                padded = bytearray()
                pad = b"\x00" * (plane.line_size - w * 3)
                for y in range(h):
                    padded += raw[y * w * 3 : (y + 1) * w * 3]
                    padded += pad
                plane.update(bytes(padded))
            frame = frame.reformat(format="yuv420p")
            frame.pts = index
            frame.time_base = 1 / rate
            container.mux(vstream.encode(frame))
        container.mux(vstream.encode())

        if audio is not None and astream is not None and sample_rate is not None:
            for aframe in _audio_frames(av, audio, sample_rate, "flt", AAC_FRAME_SAMPLES):
                container.mux(astream.encode(aframe))
            container.mux(astream.encode())
    return buf.getvalue()


#: The codecs a REGISTERED host frame is encoded with (cr-079), by the format a save
#: names. A save registers the raw snapshot and returns; the codec runs in the worker's
#: post phase — or at the save, in a fake harness that has no worker.
FRAME_MEDIA_TYPES: dict[str, str] = {
    "png": "image/png",
    "webp": "image/webp",
    "flac": "audio/flac",
    "mp4": "video/mp4",
}


class _FrameFacts(TypedDict, total=False):
    """What a registered frame's codec reads from its facts (`_services` writes them)."""

    width: int
    height: int
    mode: str
    channels: int
    samples: int
    sample_rate: int
    video_bytes: int
    shape: list[int]
    audio_channels: int
    audio_samples: int
    fps: float
    keyframe_interval: int
    max_bytes: int


def encode_frame(codec: str, frame_facts: Mapping[str, JsonValue], raw: bytes) -> bytes:
    """Encode ONE registered host frame from its raw snapshot and the facts its codec needs.

    One function under both callers — the worker's post thread and a fake's settle-at-save
    — so the bytes an author's test sees are the bytes the worker writes. Every shape here
    was validated at registration. A result field's encoded-byte bound is checked here,
    before either caller can commit the blob to its output spool.
    """
    facts = cast("_FrameFacts", frame_facts)  # registration validated these shapes
    if codec in ("png", "webp"):
        encode = encode_png if codec == "png" else encode_webp
        encoded = encode(int(facts["width"]), int(facts["height"]), raw, str(facts["mode"]))
    elif codec == "flac":
        planar = Planar(raw, (int(facts["channels"]), int(facts["samples"])))
        encoded = encode_flac(planar, int(facts["sample_rate"]))
    elif codec == "mp4":
        split = int(facts["video_bytes"])
        view = memoryview(raw)
        video = Planar(view[:split], tuple(int(d) for d in facts["shape"]))
        audio = None
        sample_rate = None
        if int(facts.get("audio_channels", 0)):
            audio = Planar(
                view[split:], (int(facts["audio_channels"]), int(facts["audio_samples"]))
            )
            sample_rate = int(facts["sample_rate"])
        keys = facts.get("keyframe_interval")
        interval = None if keys is None else int(keys)
        encoded = encode_mp4(video, float(facts["fps"]), audio, sample_rate, interval)
    else:
        raise OutputError(f"{codec!r} is not a registered-frame codec", code="encoder_unavailable")
    maximum = facts.get("max_bytes")
    if maximum is not None and len(encoded) > maximum:
        raise OutputError(
            f"encoded output is {len(encoded)} B, over its declared {maximum}-byte bound",
            code="result_bound",
        )
    return encoded


@dataclass(frozen=True, slots=True)
class EncodedVideoFacts:
    width: int
    height: int
    frame_count: int
    frame_rate: Fraction
    pixel_aspect_ratio: Fraction
    video_codec: str
    video_profile: str | None
    audio_codec: str
    audio_profile: str | None
    audio_channels: int
    audio_sample_rate: int
    audio_frame_samples: int
    audio_decoded_samples: int
    color_primaries: int
    color_transfer: int
    color_matrix: int
    color_range: int


class _BoundedWriter:
    """Seekable AVIO wrapper that refuses before a write crosses the output grant."""

    def __init__(self, raw: BinaryIO, cap: int) -> None:
        self.raw = raw
        self.cap = cap

    def write(self, data: bytes) -> int:
        end = self.raw.tell() + len(data)
        if end > self.cap:
            raise OutputError(
                f"streaming output would cross the {self.cap}-byte attempt limit",
                code="output_too_large",
            )
        written = self.raw.write(data)
        return int(written)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return int(self.raw.seek(offset, whence))

    def tell(self) -> int:
        return int(self.raw.tell())

    def truncate(self, size: int | None = None) -> int:
        target = self.raw.tell() if size is None else size
        if target > self.cap:
            raise OutputError(
                f"streaming output would cross the {self.cap}-byte attempt limit",
                code="output_too_large",
            )
        return int(self.raw.truncate(target))

    def flush(self) -> None:
        self.raw.flush()

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


def _refuse_copy(why: str) -> OutputError:
    return OutputError(
        f"videos cannot be joined by packet copy: {why}", code="video_copy_incompatible"
    )


#: One H.264 track's copy parameters: extradata, width, height, frame rate, pixel aspect ratio,
#: time base and the four colour tags.
type _CopyFacts = tuple[bytes, int, int, Fraction, Fraction, Fraction, int, int, int, int]

#: One copied picture: its bytes, its decode and presentation times on its track's clock, and
#: whether it is a keyframe.
type _Coded = tuple[bytes, int, int, bool]


def _copy_packets(path: Path) -> tuple[_CopyFacts, list[_Coded]]:
    """One H.264 track's copy parameters and pictures, or `video_copy_incompatible` unless it
    begins on a keyframe on a zero-based fixed frame clock."""
    with _av().open(str(path), "r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1 or streams[0].codec_context.name != "h264":
            raise _refuse_copy(f"{path.name} is not exactly one H.264 track")
        stream = streams[0]
        context = stream.codec_context
        facts = (
            bytes(context.extradata or b""),
            int(context.width or 0),
            int(context.height or 0),
            Fraction(stream.average_rate or 0),
            Fraction(stream.sample_aspect_ratio or 1),
            Fraction(stream.time_base or 0),
            int(context.color_primaries),
            int(context.color_trc),
            int(context.colorspace),
            int(context.color_range),
        )
        if not facts[3] or not facts[5] or (1 / (facts[3] * facts[5])).denominator != 1:
            raise _refuse_copy("the frame rate is not a whole number of time-base ticks")
        step = int(1 / (facts[3] * facts[5]))
        pictures: list[_Coded] = []
        prior: int | None = None
        for packet in container.demux(stream):
            if packet.size == 0:
                continue
            if not pictures and not packet.is_keyframe:
                raise _refuse_copy(f"{path.name} does not begin on a keyframe")
            if (
                packet.dts is None
                or packet.pts is None
                or packet.dts > packet.pts
                or (prior is not None and packet.dts <= prior)
            ):
                raise _refuse_copy(f"{path.name} has no monotone decode clock")
            prior = packet.dts
            pictures.append((bytes(packet), int(packet.dts), int(packet.pts), packet.is_keyframe))
        times = sorted(pts for _, _, pts, _ in pictures)
        if times != list(range(0, step * len(times), step)) or not times:
            raise _refuse_copy(f"{path.name} is not on a zero-based fixed frame clock")
        return facts, pictures


def copy_frames(paths: Iterable[Path]) -> tuple[Fraction, tuple[int, ...]]:
    """The H.264 tracks' frame rate and each one's frame count, or `video_copy_incompatible`
    unless they share one set of H.264 parameters and one zero-based frame clock, and each
    begins on a keyframe."""
    shared: _CopyFacts | None = None
    frames: list[int] = []
    for path in paths:
        facts, pictures = _copy_packets(path)
        if shared is None:
            shared = facts
        elif facts != shared:
            raise _refuse_copy(f"{path.name} differs in codec parameters or clock")
        frames.append(len(pictures))
    if shared is None:
        raise OutputError("a joined video needs at least one video", code="media_header")
    return shared[3], tuple(frames)


class StreamingMP4Encoder:
    """The bounded H.264/AAC writer used by the author stream sinks.

    The caller converts each RGB frame (swscale on the x264 thread count, bitwise
    the single-threaded output) and encodes AAC, while one encode thread runs x264 and every
    mux in submission order: the bytes are the serial writer's. H3's decode over four ranks
    delivers ~80 fps; the serial writer kept ~33 (runs 1435-1441).
    """

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        frame_rate: Fraction,
        pixel_aspect_ratio: Fraction,
        color_primaries: int,
        color_transfer: int,
        color_matrix: int,
        color_range: int,
        max_bytes: int,
        audio_channels: int = 0,
        audio_sample_rate: int = 0,
        keyframe_interval: int | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise OutputError(
                f"streaming video geometry {width}x{height} is not positive and even",
                code="frame_shape",
            )
        if frame_rate <= 0 or frame_rate >= 1000:
            raise OutputError(f"frame rate {frame_rate} is not supported", code="frame_rate")
        if pixel_aspect_ratio <= 0:
            raise OutputError(
                f"pixel aspect ratio {pixel_aspect_ratio} is not positive", code="frame_shape"
            )
        if audio_channels not in (0, 1, 2):
            raise OutputError(
                f"streaming audio has {audio_channels} channels; mono or stereo is required",
                code="audio_shape",
            )
        if bool(audio_channels) != bool(audio_sample_rate):
            raise OutputError(
                "streaming audio channels and sample rate must be declared together",
                code="audio_rate",
            )

        self.path = path
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.audio_channels = audio_channels
        self.audio_sample_rate = audio_sample_rate
        self.color_primaries = color_primaries
        self.color_transfer = color_transfer
        self.color_matrix = color_matrix
        self.color_range = color_range
        self.video_frames = 0
        self.audio_samples = 0
        self._max_bytes = max_bytes
        self._encoded_bytes = 0
        self._preamble_bytes = 0
        self._preamble_packets = 0
        self._preamble_written = not bool(audio_channels)
        self._streams_started: set[int] = set()
        self._jobs: queue.Queue[Callable[[], None] | None] = queue.Queue(_ENCODE_QUEUE_FRAMES)
        self._failure: BaseException | None = None
        self._stopping = False
        self._worker: threading.Thread | None = None
        self._av = _av()
        try:
            self._colorspace = _color.colorspace(self._av, color_matrix)
            self._color_range = _color.range_value(self._av, color_range)
        except ValueError as exc:
            raise OutputError(str(exc), code="video_color") from exc
        self._reformatter = self._av.video.reformatter.VideoReformatter()
        self._raw = path.open("x+b")
        self._bounded = _BoundedWriter(self._raw, max_bytes)
        try:
            self._container = self._av.open(
                self._bounded,
                "w",
                format="mp4",
                options={
                    "fflags": "+bitexact",
                    "frag_duration": "1000000",
                    # Delay the initial metadata until timestamps are known so the
                    # edit list can discard AAC encoder priming without moving video.
                    # Keep recoverable fragments while encoding, then have libavformat
                    # write a normal indexed MP4 at close. Empty movie headers alone
                    # make desktop players misreport duration and seek into delta frames.
                    # Every frame a keyframe: a fragment a second, not one per frame.
                    "movflags": "+hybrid_fragmented+empty_moov+default_base_moof+skip_sidx"
                    + ("" if keyframe_interval == 1 else "+frag_keyframe")
                    + ("+delay_moov" if audio_channels else ""),
                    "use_editlist": "1" if audio_channels else "0",
                },
            )
            self._video = self._container.add_stream("libx264", rate=frame_rate)
            self._video.width = width
            self._video.height = height
            self._video.pix_fmt = "yuv420p"
            self._video.sample_aspect_ratio = pixel_aspect_ratio
            vcodec = self._video.codec_context
            vcodec.thread_count = x264_threads()
            vcodec.gop_size = keyframe_interval or max(1, round(float(frame_rate)))
            vcodec.max_b_frames = 0
            vcodec.color_primaries = color_primaries
            vcodec.color_trc = color_transfer
            vcodec.colorspace = color_matrix
            vcodec.color_range = color_range
            vcodec.options = dict(_X264)
            self._audio = None
            if audio_channels:
                self._audio = self._container.add_stream("aac", rate=audio_sample_rate)
                acodec = self._audio.codec_context
                acodec.thread_count = 1
                acodec.layout = "mono" if audio_channels == 1 else "stereo"
                acodec.format = "fltp"
                acodec.bit_rate = _AAC_BIT_RATE
            # Open both encoders and write the header here, as the first encode and mux would,
            # so the encode thread (x264, mux) and the caller (AAC) never share them.
            for stream in (self._video, self._audio):
                if stream is not None:
                    stream.codec_context.open(strict=False)
            self._container.start_encoding()
        except Exception:
            self._raw.close()
            path.unlink(missing_ok=True)
            raise
        self._worker = threading.Thread(target=self._drain, name="mp4-encode", daemon=True)
        self._worker.start()

    def write_video(self, rgb: bytes) -> None:
        expected = self.width * self.height * 3
        if len(rgb) != expected:
            raise OutputError(
                f"streaming RGB frame is {len(rgb)} bytes, expected {expected}",
                code="frame_shape",
            )
        frame = self._av.VideoFrame(self.width, self.height, "rgb24")
        frame.color_primaries = self.color_primaries
        frame.color_trc = self.color_transfer
        frame.colorspace = self.color_matrix
        frame.color_range = 2  # immutable RGB events are full-range.
        plane = frame.planes[0]
        if plane.line_size == self.width * 3:
            plane.update(rgb)
        else:
            padded = bytearray()
            pad = b"\x00" * (plane.line_size - self.width * 3)
            stride = self.width * 3
            for row in range(self.height):
                padded += rgb[row * stride : (row + 1) * stride] + pad
            plane.update(bytes(padded))
        interpolation = (
            self._av.video.reformatter.Interpolation.BILINEAR
            | self._av.video.reformatter.Interpolation.ACCURATE_RND
            | self._av.video.reformatter.Interpolation.BITEXACT
        )
        frame = self._reformatter.reformat(
            frame,
            format="yuv420p",
            src_colorspace=self._colorspace,
            dst_colorspace=self._colorspace,
            src_color_range=self._av.video.reformatter.ColorRange.JPEG,
            dst_color_range=self._color_range,
            dst_color_trc=self.color_transfer,
            dst_color_primaries=self.color_primaries,
            interpolation=interpolation,
            threads=x264_threads(),
        )
        frame.pts = self.video_frames
        frame.time_base = Fraction(self.frame_rate.denominator, self.frame_rate.numerator)
        self._submit(lambda: self._encode_video(frame))
        self.video_frames += 1

    def _encode_video(self, frame: VideoFrame) -> None:
        for packet in self._video.encode(frame):
            packet.duration = 1
            self._mux(packet)

    def write_audio(self, planar: Planar) -> None:
        if self._audio is None:
            raise OutputError("streaming video was declared without audio", code="audio_absent")
        if planar.shape[0] != self.audio_channels:
            raise OutputError(
                f"streaming audio has {planar.shape[0]} channels, expected {self.audio_channels}",
                code="audio_shape",
            )
        packets = [
            packet
            for frame in _audio_frames(
                self._av,
                planar,
                self.audio_sample_rate,
                "flt",
                AAC_FRAME_SAMPLES,
                # Keep the author's sample clock. The MP4 edit list accounts for the
                # encoder's negative initial timestamp; it is not an extra PCM frame.
                start_pts=self.audio_samples,
            )
            for packet in self._audio.encode(frame)
        ]
        self._submit(lambda: self._mux_all(packets))
        self.audio_samples += planar.shape[1]

    def finish(self) -> EncodedVideoFacts:
        if self.video_frames == 0:
            raise OutputError("streaming video contains no frames", code="frame_shape")
        audio = self._audio.encode() if self._audio is not None else []
        self._submit(lambda: self._flush(audio))
        self._join()
        self._container.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        return probe_mp4(self.path)

    def _flush(self, audio: list[Packet[Stream]]) -> None:
        for packet in self._video.encode():
            packet.duration = 1
            self._mux(packet)
        self._mux_all(audio)

    def _mux_all(self, packets: list[Packet[Stream]]) -> None:
        for packet in packets:
            self._mux(packet)

    def _submit(self, job: Callable[[], None]) -> None:
        """Run `job` on the encode thread, after every job before it."""
        if self._failure is not None:
            raise self._failure
        if self._worker is None:
            job()
        else:
            self._jobs.put(job)

    def _drain(self) -> None:
        while (job := self._jobs.get()) is not None:
            if self._failure is None and not self._stopping:
                try:
                    job()
                except BaseException as exc:
                    self._failure = exc

    def _join(self) -> None:
        """Wait for every submitted job; the first failure is the writer's failure."""
        if self._worker is not None:
            self._jobs.put(None)
            self._worker.join()
            self._worker = None
        if self._failure is not None:
            raise self._failure

    def _mux(self, packet: Packet[Stream]) -> None:
        # Encoded payload is a lower bound on container bytes. Charge it before
        # handing it to a muxer that can buffer instead of calling the file writer.
        self._encoded_bytes += int(packet.size)
        if self._encoded_bytes > self._max_bytes:
            raise OutputError(
                f"encoded streaming payload would cross the {self._max_bytes}-byte attempt limit",
                code="output_too_large",
            )
        if not self._preamble_written:
            self._preamble_bytes += int(packet.size)
            self._preamble_packets += 1
            if (
                self._preamble_bytes > _MP4_PREAMBLE_BYTES
                or self._preamble_packets > _MP4_PREAMBLE_PACKETS
            ):
                raise OutputError(
                    "streaming audio/video startup exceeds its 8 MiB or 1024-packet buffer; "
                    "interleave both tracks earlier",
                    code="media_interleave",
                )
        self._container.mux(packet)
        self._streams_started.add(int(packet.stream.index))
        if (
            not self._preamble_written
            and self._audio is not None
            and {self._video.index, self._audio.index} <= self._streams_started
            and self._raw.tell() > 0
        ):
            self._preamble_written = True

    def abort(self) -> None:
        self._stopping = True
        with suppress(BaseException):
            self._join()
        with suppress(Exception):
            self._container.close()
        self._raw.close()
        self.path.unlink(missing_ok=True)


def _boxes(data: bytes | bytearray, start: int = 0, end: int | None = None) -> Iterator[_Box]:
    """ISO-BMFF boxes between `start` and `end`: (type, offset, header length, size)."""
    end = len(data) if end is None else end
    at = start
    while at < end:
        if end - at < 8:
            raise OutputError("MP4 box is truncated", code="output_probe")
        size, kind = struct.unpack_from(">I4s", data, at)
        head = 8
        if size == 1:
            size, head = struct.unpack_from(">Q", data, at + 8)[0], 16
        elif size == 0:
            size = end - at
        if size < head or at + size > end:
            raise OutputError("MP4 box overruns its parent", code="output_probe")
        yield kind.decode("latin-1"), at, head, size
        at += size


type _Box = tuple[str, int, int, int]


def _box(kind: str, *body: bytes) -> bytes:
    """One ISO-BMFF box; past 4 GiB its size is 64-bit."""
    length = 8 + sum(len(part) for part in body)
    if length >= 1 << 32:
        return struct.pack(">I4sQ", 1, kind.encode("latin-1"), length + 8) + b"".join(body)
    return struct.pack(">I4s", length, kind.encode("latin-1")) + b"".join(body)


def _full(kind: str, version: int, *body: bytes) -> bytes:
    """A full box (version, no flags) of `body`."""
    return _box(kind, struct.pack(">I", version << 24), *body)


def _runs(values: Iterable[int]) -> list[tuple[int, int]]:
    """(count, value) for each run of equal values."""
    runs: list[list[int]] = []
    for value in values:
        if runs and runs[-1][1] == value:
            runs[-1][0] += 1
        else:
            runs.append([1, value])
    return [(count, value) for count, value in runs]


def _child(data: bytes, box: _Box, kind: str) -> _Box:
    _, at, head, size = box
    for found in _boxes(data, at + head, at + size):
        if found[0] == kind:
            return found
    raise OutputError(f"libavformat wrote no {kind!r} box", code="output_probe")


def _raw(data: bytes, box: _Box) -> bytes:
    return data[box[1] : box[1] + box[3]]


def _payload(data: bytes, box: _Box) -> bytes:
    return data[box[1] + box[2] : box[1] + box[3]]


#: A sample entry's fields before its own boxes (ISO 14496-12: visual 78 bytes, audio 28).
_ENTRY_FIELDS = {"avc1": 78, "mp4a": 28}


def _described(data: bytes, stsd: _Box) -> bytes:
    """A sample description without its bitrate boxes: a template's `btrt` counts its own
    one or two samples, not the joined video's."""
    _, at, head, size = stsd
    entries = []
    for entry in _boxes(data, at + head + 8, at + size):
        kind, start, entry_head, _ = entry
        fields = _ENTRY_FIELDS.get(kind)
        body = _payload(data, entry)
        # An audio entry past version 0 is QuickTime's, with more fields: kept whole.
        if fields is None or (kind == "mp4a" and body[8:10] != b"\0\0"):
            entries.append(_raw(data, entry))
            continue
        inner = start + entry_head + fields
        kept = [_raw(data, box) for box in _boxes(data, inner, start + entry[3]) if box[0] != "btrt"]
        entries.append(_box(kind, body[:fields], *kept))
    return _box("stsd", data[at + head : at + head + 8], *entries)


@dataclass(frozen=True, slots=True)
class _TrackTemplate:
    """What libavformat writes of a track that its samples never change."""

    #: `tkhd`'s fields after its duration: layer, group, volume, matrix, width, height.
    shape: bytes
    flags: int
    track: int
    language: bytes
    hdlr: bytes
    media_header: bytes
    dinf: bytes
    stsd: bytes
    sgpd: bytes

    @classmethod
    def read(cls, data: bytes, trak: _Box) -> _TrackTemplate:
        tkhd = _payload(data, _child(data, trak, "tkhd"))
        version, flags = tkhd[0], int.from_bytes(tkhd[1:4])
        track, shape = (tkhd[20:24], tkhd[36:]) if version else (tkhd[12:16], tkhd[24:])
        mdia = _child(data, trak, "mdia")
        minf = _child(data, mdia, "minf")
        stbl = _child(data, minf, "stbl")
        header = next(box for box in _boxes(data, minf[1] + minf[2], minf[1] + minf[3]) if box[0] in ("vmhd", "smhd"))
        sgpd = next((box for box in _boxes(data, stbl[1] + stbl[2], stbl[1] + stbl[3]) if box[0] == "sgpd"), None)
        return cls(
            shape=shape,
            flags=flags,
            track=int.from_bytes(track),
            language=_payload(data, _child(data, mdia, "mdhd"))[-4:],
            hdlr=_raw(data, _child(data, mdia, "hdlr")),
            media_header=_raw(data, header),
            dinf=_raw(data, _child(data, minf, "dinf")),
            stsd=_described(data, _child(data, stbl, "stsd")),
            sgpd=_raw(data, sgpd) if sgpd is not None else b"",
        )


@dataclass(slots=True)
class _Track:
    """One joined track's samples so far, in decode order, and the chunks they lie in."""

    template: _TrackTemplate
    timescale: int
    #: Decode times on the track's clock, from 0, and presentation minus decode for each.
    times: array[int] = field(default_factory=lambda: array("q"))
    offsets: array[int] = field(default_factory=lambda: array("q"))
    sizes: array[int] = field(default_factory=lambda: array("q"))
    #: Keyframes' 1-based sample numbers; for audio, every sample is one.
    sync: array[int] | None = None
    #: (file offset, samples) of each chunk, in file order.
    chunks: list[tuple[int, int]] = field(default_factory=list)
    #: The decode time past its last sample, where its presentation starts, and how long it is.
    end: int = 0
    start: int = 0
    shown: int = 0

    def add(self, time: int, offset: int, size: int, key: bool) -> None:
        if self.times and time <= self.times[-1]:
            raise _refuse_copy("its decode clock would run backwards across the join")
        self.times.append(time)
        self.offsets.append(offset)
        self.sizes.append(size)
        if key and self.sync is not None:
            self.sync.append(len(self.times))

    def trak(self, movie: int) -> bytes:
        """This track's `trak` as its samples stand, its edit on the `movie` clock."""
        t = self.template
        shown = self.shown * movie // self.timescale
        wide = max(shown, self.end, self.start) >= 1 << 31
        tkhd = _box(
            "tkhd",
            struct.pack(">I", int(wide) << 24 | t.flags),
            struct.pack(">QQIIQ" if wide else ">IIIII", 0, 0, t.track, 0, shown),
            t.shape,
        )
        edit = struct.pack(">QqHH" if wide else ">IiHH", shown, self.start, 1, 0)
        mdhd = _full("mdhd", int(wide), struct.pack(">QQIQ" if wide else ">IIII", 0, 0, self.timescale, self.end), t.language)
        times = list(self.times)
        durations = [later - earlier for earlier, later in pairwise([*times, self.end])]
        stts = _runs(durations)
        tables = [t.stsd, _full("stts", 0, struct.pack(f">I{2 * len(stts)}I", len(stts), *(v for run in stts for v in run)))]
        if self.sync is not None and len(self.sync) != len(times):
            tables.append(_full("stss", 0, struct.pack(f">I{len(self.sync)}I", len(self.sync), *self.sync)))
        if any(self.offsets):
            ctts = _runs(self.offsets)
            signed = min(self.offsets) < 0
            tables.append(_full("ctts", int(signed), struct.pack(">I", len(ctts)),
                                b"".join(struct.pack(">Ii" if signed else ">II", count, value) for count, value in ctts)))
        stsc = _runs(count for _, count in self.chunks)
        firsts = list(accumulate([0, *(runs for runs, _ in stsc)]))[:-1]
        tables.append(_full("stsc", 0, struct.pack(">I", len(stsc)),
                            b"".join(struct.pack(">III", first + 1, each, 1) for first, (_, each) in zip(firsts, stsc, strict=True))))
        uniform = len(set(self.sizes)) == 1
        tables.append(_full("stsz", 0, struct.pack(">II", self.sizes[0] if uniform else 0, len(times)),
                            b"" if uniform else struct.pack(f">{len(times)}I", *self.sizes)))
        offsets = [offset for offset, _ in self.chunks]
        far = bool(offsets) and offsets[-1] >= 1 << 32
        tables.append(_full("co64" if far else "stco", 0, struct.pack(f">I{len(offsets)}{'QI'[not far]}", len(offsets), *offsets)))
        if t.sgpd:
            tables += [t.sgpd, _full("sbgp", 0, b"roll", struct.pack(">III", 1, len(times), 1))]
        minf = _box("minf", t.media_header, t.dinf, _box("stbl", *tables))
        return _box("trak", tkhd, _box("edts", _full("elst", int(wide), struct.pack(">I", 1), edit)),
                    _box("mdia", mdhd, t.hdlr, minf))


#: How much of a joined video one chunk holds, a track at a time: a player streaming it reads a
#: second of both tracks together.
_CHUNK_SECONDS = 1


class GrowingMP4:
    """H.264 videos joined by packet copy under one AAC soundtrack, as an MP4 that grows (cl-1210).

    The file is `ftyp`, then one `mdat` per appended video, written once (its pictures, and the
    sound encoded over its stretch, interleaved a second at a time), then an index of everything
    so far (`moov`), written again for each revision. Every revision is an ordinary indexed MP4,
    so players read its duration and seek, and the next keeps all its parts but the index: a
    reader holding one reads only the new `mdat` and `moov`. The AAC encoder persists across
    appends, so the soundtrack is one encode. Finishing flushes it into the last `mdat`, or one
    of its own: the final is the last revision. The sample descriptions are libavformat's own,
    from a template it writes of the first video and the soundtrack.
    """

    def __init__(
        self, directory: Path, *, audio_channels: int, audio_sample_rate: int, max_bytes: int
    ) -> None:
        if audio_channels not in (0, 1, 2) or bool(audio_channels) != bool(audio_sample_rate):
            raise OutputError(
                "a joined soundtrack is mono or stereo with its rate", code="audio_shape"
            )
        self.directory, self.max_bytes = directory, max_bytes
        self.audio_channels, self.audio_sample_rate = audio_channels, audio_sample_rate
        #: (file, media microseconds) per `mdat` part, the first beginning with `ftyp`.
        self.parts: list[tuple[Path, int]] = []
        self.frames = 0
        self.samples = 0
        self.flushed = False
        self._size = 0
        self._facts: _CopyFacts | None = None
        self._ftyp = b""
        self._video: _Track | None = None
        self._sound: _Track | None = None
        #: Video ticks a frame, and track ticks a video tick.
        self._step = self._scale = 1
        #: How far the first video's decode clock and the soundtrack start before zero (frame
        #: reordering; AAC priming): their edit lists start there.
        self._delay = 0
        self._priming: int | None = None
        self._index: tuple[tuple[int, bool], Path] | None = None
        self._av = _av()
        self._aac: AudioCodecContext | None = None
        if audio_channels:
            aac = self._av.CodecContext.create("aac", "w")
            aac.sample_rate = audio_sample_rate
            aac.layout = "mono" if audio_channels == 1 else "stereo"
            aac.format = "fltp"
            aac.bit_rate = _AAC_BIT_RATE
            aac.thread_count = 1
            aac.time_base = Fraction(1, audio_sample_rate)
            aac.open()
            self._aac = aac

    def append(
        self, video: Path, audio: Callable[[int], Iterable[Planar]], *, last: bool = False
    ) -> None:
        """Join one more video and its stretch of the soundtrack, which `audio` returns given
        the video's frame count; `last` closes the soundtrack."""
        if self.flushed:
            raise OutputError("the joined video is already finished", code="media_join")
        facts, pictures = _copy_packets(video)
        if self._facts is None:
            self._start(video, facts, pictures)
        elif facts != self._facts:
            raise _refuse_copy(f"{video.name} differs in codec parameters or clock")
        packets = []
        for planar in audio(len(pictures)):
            packets += self._encode(planar)
        if last:
            packets += self._flush()
        self._write(pictures, packets)

    def revision(self) -> list[tuple[Path, int]]:
        """The joined video so far as its parts: each `mdat` as written, then its index."""
        if not self.parts:
            raise OutputError("a joined video needs at least one video", code="media_header")
        state = (len(self.parts), self.flushed)
        if self._index is None or self._index[0] != state:
            index = self._moov()
            if self._size + len(index) > self.max_bytes:
                raise OutputError(
                    f"the joined video would cross the {self.max_bytes}-byte limit",
                    code="output_too_large",
                )
            path = self.directory / f"join-index-{len(self.parts):04d}{'-final' if self.flushed else ''}"
            _write_part(path, index)
            self._index = state, path
        return [*self.parts, (self._index[1], 0)]

    def finish(self) -> list[tuple[Path, int]]:
        """Close the soundtrack, its last packets in an `mdat` of their own unless the last
        append took them, and return the final parts."""
        if not self.parts:
            raise OutputError("a joined video needs at least one video", code="media_header")
        if not self.flushed:
            packets = self._flush()
            if packets:
                self._write([], packets)
        return self.revision()

    def _start(self, video: Path, facts: _CopyFacts, pictures: list[_Coded]) -> None:
        self._facts = facts
        rate, base = facts[3], facts[5]
        self._step, self._scale = int(1 / (rate * base)), base.numerator
        self._delay = max(0, -pictures[0][1])
        data = self._template(video)
        moov = next(box for box in _boxes(data) if box[0] == "moov")
        traks = {
            _payload(data, _child(data, _child(data, trak, "mdia"), "hdlr"))[8:12]: trak
            for trak in _boxes(data, moov[1] + moov[2], moov[1] + moov[3])
            if trak[0] == "trak"
        }
        self._ftyp = _raw(data, next(box for box in _boxes(data) if box[0] == "ftyp"))
        self._video = _Track(_TrackTemplate.read(data, traks[b"vide"]), base.denominator, sync=array("q"))
        self._video.start = self._delay * self._scale
        if self._aac is not None:
            self._sound = _Track(_TrackTemplate.read(data, traks[b"soun"]), self.audio_sample_rate)

    def _template(self, video: Path) -> bytes:
        """libavformat's MP4 of the first video's first picture and a moment of the soundtrack's
        silence, encoded alike: what it writes of these tracks, to copy."""
        path = self.directory / ".join-template.mp4"
        try:
            container = self._av.open(str(path), "w", format="mp4", options={"fflags": "+bitexact"})
            try:
                with self._av.open(str(video), "r") as source:
                    track = container.add_stream_from_template(source.streams.video[0])
                    sound = None
                    if self._aac is not None:
                        sound = container.add_stream("aac", rate=self.audio_sample_rate)
                        sound.codec_context.layout = self._aac.layout
                        sound.codec_context.format = "fltp"
                        sound.codec_context.bit_rate = _AAC_BIT_RATE
                        sound.codec_context.thread_count = 1
                    container.start_encoding()
                    packet = next(packet for packet in source.demux(source.streams.video[0]) if packet.size)
                    packet.stream = track
                    container.mux(packet)
                    if sound is not None:
                        quiet = Planar(bytes(4 * self.audio_channels * AAC_FRAME_SAMPLES), (self.audio_channels, AAC_FRAME_SAMPLES))
                        for frame in _audio_frames(self._av, quiet, self.audio_sample_rate, "flt", AAC_FRAME_SAMPLES):
                            container.mux(sound.encode(frame))
                        container.mux(sound.encode(None))
            finally:
                container.close()
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)

    def _encode(self, planar: Planar) -> list[Packet[Stream]]:
        if self._aac is None or planar.shape[0] != self.audio_channels:
            raise OutputError(
                "joined audio does not match the declared soundtrack", code="audio_shape"
            )
        frames = _audio_frames(
            self._av,
            planar,
            self.audio_sample_rate,
            "flt",
            AAC_FRAME_SAMPLES,
            start_pts=self.samples,
        )
        self.samples += planar.shape[1]
        return [packet for frame in frames for packet in self._aac.encode(frame)]

    def _flush(self) -> list[Packet[Stream]]:
        self.flushed = True
        return list(self._aac.encode()) if self._aac is not None else []

    def _write(self, pictures: list[_Coded], sound: list[Packet[Stream]]) -> None:
        """One `mdat` of `pictures` and `sound` after those written, a second of a track a chunk."""
        video = self._video
        assert video is not None
        held: list[tuple[int, int, int, bytes]] = []
        base = self.frames * self._step + self._delay
        for index, (coded, dts, pts, key) in enumerate(pictures):
            time = (base + dts) * self._scale
            video.add(time, (pts - dts) * self._scale, len(coded), key)
            seconds = (time - video.start) // (video.timescale * _CHUNK_SECONDS)
            held.append((max(0, seconds), 0, index, coded))
        if pictures:
            video.end = video.times[-1] + self._step * self._scale
        self.frames += len(pictures)
        video.shown = self.frames * self._step * self._scale
        if sound:
            track = self._sound
            assert track is not None
            if self._priming is None:
                self._priming = max(0, -int(sound[0].pts or 0))
                track.start = self._priming
            for index, packet in enumerate(sound):
                time = int(packet.pts or 0) + self._priming
                track.add(time, 0, int(packet.size), True)
                track.end = time + int(packet.duration or AAC_FRAME_SAMPLES)
                seconds = (time - track.start) // (track.timescale * _CHUNK_SECONDS)
                held.append((max(0, seconds), 1, index, bytes(packet)))
            track.shown = min(track.end - track.start, self.samples)
        held.sort(key=lambda sample: sample[:3])
        payload = sum(len(sample[3]) for sample in held)
        head = 8 if payload + 8 < 1 << 32 else 16
        lead = self._ftyp if not self.parts else b""
        at = self._size + len(lead) + head
        if self._size + len(lead) + head + payload > self.max_bytes:
            raise OutputError(
                f"the joined video would cross the {self.max_bytes}-byte limit",
                code="output_too_large",
            )
        for (_, kind), samples in groupby(held, key=lambda sample: sample[:2]):
            run = list(samples)
            track = video if kind == 0 else self._sound
            assert track is not None
            track.chunks.append((at, len(run)))
            at += sum(len(sample[3]) for sample in run)
        header = struct.pack(">I4s", 8 + payload, b"mdat") if head == 8 else struct.pack(">I4sQ", 1, b"mdat", 16 + payload)
        path = self.directory / f"join-part-{len(self.parts):04d}"
        _write_part(path, lead, header, *(sample[3] for sample in held))
        self._size = at
        micros = round(Fraction(len(pictures)) / self._facts[3] * 1_000_000) if self._facts else 0
        self.parts.append((path, micros))

    def _moov(self) -> bytes:
        tracks = [track for track in (self._video, self._sound) if track is not None and track.times]
        movie = math.lcm(*(track.timescale for track in tracks))
        movie = movie if movie < 1 << 31 else tracks[0].timescale
        duration = max(track.shown * movie // track.timescale for track in tracks)
        wide = duration >= 1 << 32
        mvhd = _full(
            "mvhd",
            int(wide),
            struct.pack(">QQIQ" if wide else ">IIII", 0, 0, movie, duration),
            struct.pack(">IH10x", 0x10000, 0x100),
            _MATRIX,
            bytes(24),
            struct.pack(">I", 3),
        )
        return _box("moov", mvhd, *(track.trak(movie) for track in tracks))


_MATRIX = struct.pack(">9I", 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000)


def _write_part(path: Path, *pieces: bytes) -> None:
    """A part, durable before it's named in a revision."""
    with path.open("xb") as writer:
        for piece in pieces:
            writer.write(piece)
        writer.flush()
        os.fsync(writer.fileno())


def probe_mp4(path: Path) -> EncodedVideoFacts:
    """Read the final bytes' container facts: stream headers and packet clocks.

    No picture or PCM is decoded: this process encoded those packets, and a second decode
    would only prove the codec agrees with itself.
    """
    av = _av()
    with av.open(str(path), "r") as container:
        videos = list(container.streams.video)
        audios = list(container.streams.audio)
        if len(videos) != 1 or len(audios) > 1:
            raise OutputError(
                "encoded MP4 does not contain exactly one video and at most one audio track",
                code="output_probe",
            )
        video = videos[0]
        width = int(video.codec_context.width or 0)
        height = int(video.codec_context.height or 0)
        times: list[int] = []
        audio_ticks = 0
        for packet in container.demux(video, *audios):
            if packet.size == 0:
                continue
            if packet.pts is None:
                raise OutputError("encoded packet has no exact clock", code="output_probe")
            if packet.stream.index == video.index:
                times.append(int(packet.pts))
            else:
                # Priming before zero is not audio; the edit list removes it on playback.
                start = int(packet.pts)
                audio_ticks += max(0, start + int(packet.duration or 0)) - max(0, start)
        times.sort()
        steps = {later - earlier for earlier, later in pairwise(times)}
        if len(steps) > 1 or min(steps, default=1) <= 0:
            raise OutputError(
                f"encoded video frame clock is discontinuous: steps={sorted(steps)[:4]}",
                code="output_probe",
            )
        frame_rate = (
            1 / (steps.pop() * Fraction(video.time_base))
            if steps
            else Fraction(video.average_rate or 0)
        )
        context = video.codec_context
        color_facts = (
            int(context.color_primaries),
            int(context.color_trc),
            int(context.colorspace),
            int(context.color_range),
        )
        video_frames = len(times)
        if video_frames == 0 or width <= 0 or height <= 0 or frame_rate <= 0:
            raise OutputError("encoded MP4 has no valid video clock", code="output_probe")
        audio = audios[0] if audios else None
        pixel_aspect_ratio = Fraction(video.sample_aspect_ratio or 1)
        if pixel_aspect_ratio <= 0:
            raise OutputError("encoded MP4 has no valid pixel aspect ratio", code="output_probe")
        if audio is not None and int(audio.codec_context.frame_size or 0) <= 0:
            raise OutputError("encoded MP4 has no audio codec-frame size", code="output_probe")
        return EncodedVideoFacts(
            width=width,
            height=height,
            frame_count=video_frames,
            frame_rate=frame_rate,
            pixel_aspect_ratio=pixel_aspect_ratio,
            video_codec=str(video.codec_context.name),
            video_profile=(
                str(video.codec_context.profile)
                if video.codec_context.profile is not None
                else None
            ),
            audio_codec=str(audio.codec_context.name) if audio is not None else "",
            audio_profile=(
                str(audio.codec_context.profile)
                if audio is not None and audio.codec_context.profile is not None
                else None
            ),
            audio_channels=int(audio.codec_context.channels or 0) if audio is not None else 0,
            audio_sample_rate=int(audio.codec_context.sample_rate or 0) if audio is not None else 0,
            audio_frame_samples=int(audio.codec_context.frame_size or 0)
            if audio is not None
            else 0,
            audio_decoded_samples=round(
                audio_ticks * Fraction(audio.time_base) * int(audio.codec_context.sample_rate)
            )
            if audio is not None
            else 0,
            color_primaries=color_facts[0],
            color_transfer=color_facts[1],
            color_matrix=color_facts[2],
            color_range=color_facts[3],
        )
