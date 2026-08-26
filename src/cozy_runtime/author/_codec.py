"""cr-017's codec plane, the H3-first slice: FLAC audio and the muxed H.264+AAC mp4.

Encoders come from the `media` extra (PyAV — FFmpeg bundled as an abi3 wheel, which is
what the wheel-only packaging rule admits; a system ffmpeg binary never was). The base
package stays free of it: `av` is imported inside a function, and an absent wheel is the
same typed `encoder_unavailable` refusal this release always had.

Determinism is a pinned property, not a hope: every encoder runs with `thread_count=1`,
fixed settings, and `fflags +bitexact` on the container, so one (input, wheel version,
settings) triple produces one byte sequence — proven by the repeat-digest arms in
`scripts/media-live.py`, and held across machines by the locked wheel. The PRE-ENCODE
tensors remain the cross-platform bit-grade artifacts (H3 banks those digests itself);
the container digest is bit-grade per pinned wheel.

Tensor input is duck-typed — a torch tensor is drained through `.detach().cpu().numpy()`
and an ndarray through `.tobytes()` — so this module imports neither torch nor numpy and
the snapshot it takes IS the §3.4 immutable CPU copy: by the time a save returns, the
output bytes are source-independent and mutating the author's tensor changes nothing.
"""

from __future__ import annotations

import io
import math
from array import array
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from cozy_runtime.author._errors import OutputError

#: Raw-snapshot ceiling as a multiple of the attempt's output ceiling. H3's largest launch
#: product (124 frames of 1344x768 RGB) is ~1.45x the 256 MiB output cap; 16x admits an
#: order of magnitude of headroom while refusing a runaway [T] dimension typed, before the
#: snapshot copy — not the OOM killer — is what stops it.
SNAPSHOT_FACTOR = 16

#: Pinned x264 settings. crf 17 is visually-lossless-grade for a generation product;
#: veryfast keeps the CPU tail small on pods. Changing either changes every banked
#: container digest, so a change here is deliberate, never drift.
_X264 = {"crf": "17", "preset": "veryfast"}

#: Pinned AAC bitrate — set explicitly so a wheel-default change cannot move the output.
_AAC_BIT_RATE = 192_000

_FLAC_CHUNK = 4096
_AAC_CHUNK = 1024


@dataclass(frozen=True, slots=True)
class Planar:
    """An immutable CPU snapshot of a tensor-like: raw bytes plus the shape they claim."""

    data: bytes
    shape: tuple[int, ...]


def _av() -> Any:
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
    """[T, H, W, 3] uint8, H and W even (yuv420p), snapshot-bounded. Refusals are typed
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
        raise OutputError(
            f"video frames are {w}x{h}: H.264 yuv420p needs even dimensions — the geometry "
            "fit/restore plane (cr-017's remaining half) owns fitting, not the encoder",
            code="frame_shape",
        )
    limit = SNAPSHOT_FACTOR * max_output_bytes
    if t * h * w * 3 > limit:
        raise OutputError(
            f"video snapshot is {t * h * w * 3} bytes, over the {limit} snapshot bound "
            f"({SNAPSHOT_FACTOR}x the attempt output ceiling)",
            code="output_too_large",
        )
    data = bytes(arr.tobytes())  # type: ignore[attr-defined]
    if len(data) != t * h * w * 3:
        raise OutputError(
            f"video frames yielded {len(data)} bytes for shape {shape}", code="frame_shape"
        )
    return Planar(data, shape)


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


def _audio_frames(av: Any, planar: Planar, sample_rate: int, fmt: str, chunk: int) -> Any:
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
        raw, unit = pcm.tobytes(), 2 * channels
    else:  # flt
        if channels == 2:
            floats = array("f")
            floats.frombytes(planar.data)
            inter_f = array("f", bytes(len(floats) * 4))
            inter_f[0::2] = floats[:samples]
            inter_f[1::2] = floats[samples:]
            raw, unit = inter_f.tobytes(), 4 * channels
        else:
            raw, unit = planar.data, 4 * channels

    def gen() -> Any:
        pts = 0
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


def encode_mp4(video: Planar, fps: float, audio: Planar | None, sample_rate: int | None) -> bytes:
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
        vcodec.thread_count = 1
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
        for index in range(t):
            frame = av.VideoFrame(w, h, "rgb24")
            plane = frame.planes[0]
            raw = video.data[index * stride : (index + 1) * stride]
            if plane.line_size == w * 3:
                plane.update(raw)
            else:
                padded = bytearray()
                pad = b"\x00" * (plane.line_size - w * 3)
                for y in range(h):
                    padded += raw[y * w * 3 : (y + 1) * w * 3] + pad
                plane.update(bytes(padded))
            frame = frame.reformat(format="yuv420p")
            frame.pts = index
            frame.time_base = 1 / rate
            container.mux(vstream.encode(frame))
        container.mux(vstream.encode())

        if audio is not None and astream is not None and sample_rate is not None:
            for aframe in _audio_frames(av, audio, sample_rate, "flt", _AAC_CHUNK):
                container.mux(astream.encode(aframe))
            container.mux(astream.encode())
    return buf.getvalue()
