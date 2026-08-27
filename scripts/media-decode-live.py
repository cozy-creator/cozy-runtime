#!/usr/bin/env python3
"""The bounded input-media decoder over real PyAV containers and the real injector.

No mocks and no model: deterministic codec fixtures prove stride removal, AAC padding,
coarse Opus clocks, one-pass video+soundtrack demux, display transforms, lifetime, and
expansion bounds.

    uv run --extra media python scripts/media-decode-live.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from array import array
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any, cast

import msgspec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cozy_runtime.author import (  # noqa: E402
    App,
    AssetBound,
    AudioAsset,
    DecodedAudioChunk,
    DecodedMediaHeader,
    DecodedVideoFormat,
    DecodedVideoFrame,
    ImageAsset,
    ImageFrame,
    Invocation,
    MediaDecodeError,
    MediaDecoder,
    VideoAsset,
    _color,
    attempt,
)
from cozy_runtime.author._assets import GrantedInput, bind, digest_bytes  # noqa: E402
from cozy_runtime.author._decode import (  # noqa: E402
    DEFAULT_DECODE_LIMITS,
    DecodeLimits,
)
from cozy_runtime.author._describe import describe  # noqa: E402
from cozy_runtime.author._errors import AuthorError, ConformanceError  # noqa: E402
from cozy_runtime.author._services import Attempt, Outputs  # noqa: E402
from cozy_runtime.internal.descriptor import _services  # noqa: E402
from cozy_runtime.internal.schema import render  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    PASS += int(ok)
    FAIL += int(not ok)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))


def expects(name: str, code: str, fn: Any, *, field: str | None = None) -> None:
    try:
        fn()
    except AuthorError as exc:
        check(
            name,
            exc.code == code and (field is None or exc.fields == (field,)),
            f"code={exc.code} fields={exc.fields}",
        )
        return
    check(name, False, "no refusal")


class NdLike:
    def __init__(self, data: bytes, shape: tuple[int, ...], dtype: str) -> None:
        self.data = data
        self.shape = shape
        self.dtype = dtype

    def tobytes(self) -> bytes:
        return self.data


def hydrate(asset: ImageAsset | AudioAsset | VideoAsset, *, input_id: str, cap: int) -> None:
    assert asset._local is not None
    bind(
        asset,
        local=asset._local,
        attempt="decode-live",
        media_type=asset.media_type,
        digest=asset.digest,
        length=asset.size_bytes,
        max_decoded_bytes=cap,
        input_id=input_id,
    )


def packed_rgb(frame: Any, seed: int = 0) -> bytes:
    row = bytes((seed + value) % 256 for value in range(frame.width * 3))
    return b"".join(row for _ in range(frame.height))


def fill_rgb(frame: Any, data: bytes) -> None:
    plane = frame.planes[0]
    row_bytes = frame.width * 3
    plane.update(
        b"".join(
            data[row * row_bytes : (row + 1) * row_bytes] + b"\0" * (plane.line_size - row_bytes)
            for row in range(frame.height)
        )
    )


def webm(av: Any, path: Path) -> tuple[int, int]:
    samples = 15_984
    with av.open(str(path), "w", format="webm") as container:
        video = container.add_stream("libvpx-vp9", rate=24)
        video.width, video.height, video.pix_fmt = 32, 16, "yuv420p"
        audio = container.add_stream("libopus", rate=48_000)
        audio.layout = "mono"
        for index in range(8):
            frame = av.VideoFrame(32, 16, "rgb24")
            fill_rgb(frame, packed_rgb(frame, index))
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        sound = av.AudioFrame(format="flt", layout="mono", samples=samples)
        sound.sample_rate, sound.pts, sound.time_base = 48_000, 0, Fraction(1, 48_000)
        pcm = array(
            "f", (0.2 * math.sin(2 * math.pi * 440 * index / 48_000) for index in range(samples))
        )
        sound.planes[0].update(pcm.tobytes())
        for packet in audio.encode(sound):
            container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode():
                container.mux(packet)
    with av.open(str(path), "r") as probe:
        decoded_samples = sum(int(frame.samples) for frame in probe.decode(audio=0))
    return 8, decoded_samples


def discontinuous_webm(av: Any, path: Path, *, second_pts: int) -> None:
    """A reachable VP9+Opus input with caller-controlled audio discontinuity."""
    with av.open(str(path), "w", format="webm") as container:
        video = container.add_stream("libvpx-vp9", rate=24)
        video.width, video.height, video.pix_fmt = 32, 16, "yuv420p"
        audio = container.add_stream("libopus", rate=48_000)
        audio.layout = "stereo"
        for index in range(4):
            picture = av.VideoFrame(32, 16, "rgb24")
            fill_rgb(picture, packed_rgb(picture, index))
            picture.pts, picture.time_base = index, Fraction(1, 24)
            for packet in video.encode(picture):
                container.mux(packet)
        for pts in (0, second_pts):
            frame = av.AudioFrame(format="fltp", layout="stereo", samples=960)
            frame.sample_rate, frame.pts, frame.time_base = 48_000, pts, Fraction(1, 48_000)
            for plane in frame.planes:
                plane.update(bytes(960 * 4))
            for packet in audio.encode(frame):
                container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode():
                container.mux(packet)


def offset_webm(av: Any, path: Path) -> None:
    """A soundtrack beginning 100 ms after its video's zero clock."""
    with av.open(str(path), "w", format="webm") as container:
        video = container.add_stream("libvpx-vp9", rate=24)
        video.width, video.height, video.pix_fmt = 32, 16, "yuv420p"
        audio = container.add_stream("libopus", rate=48_000)
        audio.layout = "mono"
        for index in range(8):
            picture = av.VideoFrame(32, 16, "rgb24")
            fill_rgb(picture, packed_rgb(picture, index))
            picture.pts, picture.time_base = index, Fraction(1, 24)
            for packet in video.encode(picture):
                container.mux(packet)
        for pts in (4800, 5760, 6720):
            frame = av.AudioFrame(format="fltp", layout="mono", samples=960)
            frame.sample_rate, frame.pts, frame.time_base = 48_000, pts, Fraction(1, 48_000)
            frame.planes[0].update(bytes(960 * 4))
            for packet in audio.encode(frame):
                container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode():
                container.mux(packet)


def rotated_mp4(av: Any, path: Path, *, hflip: bool) -> None:
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 4, 2, "yuv420p"
        stream.set_display_rotation(90, hflip=hflip)
        frame = av.VideoFrame(4, 2, "rgb24")
        fill_rgb(frame, packed_rgb(frame))
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def unspecified_hd_mp4(av: Any, path: Path) -> bytes:
    """BT.709 YUV bytes with cleared tags: Runtime must normalize HD before RGB decode."""
    width, height = 1280, 720
    rgb = bytes(
        channel
        for y in range(height)
        for x in range(width)
        for channel in (
            x * 255 // (width - 1),
            y * 255 // (height - 1),
            (x + y) * 255 // (width + height - 2),
        )
    )
    source = av.VideoFrame(width, height, "rgb24")
    fill_rgb(source, rgb)
    reformatter = av.video.reformatter.VideoReformatter()
    yuv = reformatter.reformat(
        source,
        format="yuv420p",
        src_colorspace="ITU709",
        dst_colorspace="ITU709",
        src_color_range="JPEG",
        dst_color_range="MPEG",
        dst_color_trc=1,
        dst_color_primaries=1,
        threads=1,
    )
    yuv.color_primaries = 2
    yuv.color_trc = 2
    yuv.colorspace = 2
    yuv.color_range = 0
    with av.open(str(path), "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.codec_context.thread_count = 1
        stream.codec_context.max_b_frames = 0
        stream.codec_context.options = {"crf": "0", "preset": "ultrafast"}
        yuv.pts, yuv.time_base = 0, Fraction(1, 24)
        for packet in stream.encode(yuv):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return rgb


def jpeg(av: Any, *, orientation: int) -> bytes:
    frame = av.VideoFrame(4, 2, "rgb24")
    fill_rgb(frame, packed_rgb(frame))
    codec = av.CodecContext.create("mjpeg", "w")
    codec.width, codec.height, codec.pix_fmt = 4, 2, "yuvj420p"
    body = b"".join(bytes(packet) for packet in (*codec.encode(frame), *codec.encode(None)))
    tiff = (
        b"II*\x00"
        + (8).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (0x0112).to_bytes(2, "little")
        + (3).to_bytes(2, "little")
        + (1).to_bytes(4, "little")
        + orientation.to_bytes(2, "little")
        + b"\x00\x00"
        + b"\x00\x00\x00\x00"
    )
    payload = b"Exif\x00\x00" + tiff
    return body[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + body[2:]


def undecorated_rgb(av: Any, path: Path) -> bytes:
    """The stored pixel plane without applying its presentation transform."""
    with av.open(str(path), "r") as container:
        source = next(container.decode(video=0))
        primaries, transfer, matrix, color_range = _color.normalize(
            width=int(source.width),
            height=int(source.height),
            primaries=int(source.color_primaries),
            transfer=int(source.color_trc),
            matrix=int(source.colorspace),
            color_range=int(source.color_range),
        )
        frame = av.video.reformatter.VideoReformatter().reformat(
            source,
            format="rgb24",
            src_colorspace=_color.colorspace(av, matrix),
            dst_colorspace=_color.colorspace(av, matrix),
            src_color_range=_color.range_value(av, color_range),
            dst_color_range=av.video.reformatter.ColorRange.JPEG,
            dst_color_trc=transfer,
            dst_color_primaries=primaries,
            interpolation=(
                av.video.reformatter.Interpolation.BILINEAR
                | av.video.reformatter.Interpolation.ACCURATE_RND
                | av.video.reformatter.Interpolation.BITEXACT
            ),
            threads=1,
        )
        plane = frame.planes[0]
        row_bytes = int(frame.width) * 3
        raw = memoryview(plane)
        return b"".join(
            bytes(raw[row * plane.line_size : row * plane.line_size + row_bytes])
            for row in range(int(frame.height))
        )


def undecorated_image_rgb(av: Any, path: Path) -> bytes:
    """Image decoder reference; still images have no video color-normalization contract."""
    with av.open(str(path), "r") as container:
        frame = next(container.decode(video=0)).reformat(format="rgb24")
        plane = frame.planes[0]
        row_bytes = int(frame.width) * 3
        raw = memoryview(plane)
        return b"".join(
            bytes(raw[row * plane.line_size : row * plane.line_size + row_bytes])
            for row in range(int(frame.height))
        )


def reordered_pixels(rgb: bytes, order: tuple[int, ...]) -> bytes:
    pixels = [rgb[offset : offset + 3] for offset in range(0, len(rgb), 3)]
    return b"".join(pixels[index] for index in order)


def input_asset(path: Path, *, ref: str) -> VideoAsset:
    data = path.read_bytes()
    asset = VideoAsset(ref)
    bind(
        asset,
        local=path,
        attempt="decode-live",
        media_type="video/mp4" if path.suffix == ".mp4" else "video/webm",
        digest=digest_bytes(data),
        length=len(data),
        max_decoded_bytes=8 << 20,
        input_id=ref,
    )
    return asset


def input_image(path: Path, data: bytes, *, ref: str) -> ImageAsset:
    path.write_bytes(data)
    asset = ImageAsset(ref)
    bind(
        asset,
        local=path,
        attempt="decode-live",
        media_type="image/jpeg",
        digest=digest_bytes(data),
        length=len(data),
        max_decoded_bytes=1 << 20,
        input_id=ref,
    )
    return asset


app = App()


class DecodeInput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=1 << 20, max_decoded_bytes=1 << 20),
    ]


class DecodeResult(msgspec.Struct):
    width: int
    height: int
    length: int


@app.entrypoint
def decode_through_injector(payload: DecodeInput, decoder: MediaDecoder) -> DecodeResult:
    image = decoder.decode_image(payload.image)
    return DecodeResult(image.width, image.height, len(image.rgb))


class ImageGroup(msgspec.Struct):
    images: Annotated[
        list[ImageAsset],
        AssetBound(max_bytes=1 << 20, max_decoded_bytes=1 << 20),
    ]


class PictureReference(msgspec.Struct, tag="image"):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=1 << 20, max_decoded_bytes=1 << 20),
    ]


class ClipReference(msgspec.Struct, tag="video"):
    video: Annotated[
        VideoAsset,
        AssetBound(max_bytes=1 << 20, max_decoded_bytes=8 << 20),
    ]


class NestedDecodeInput(msgspec.Struct):
    groups: list[ImageGroup]
    references: list[PictureReference | ClipReference]


class ReferencesOnlyInput(msgspec.Struct):
    references: list[PictureReference | ClipReference]


class UnboundedClipReference(msgspec.Struct, tag="unbounded_video"):
    video: VideoAsset


class IncompleteReferencesInput(msgspec.Struct):
    references: list[PictureReference | UnboundedClipReference]


@app.entrypoint
def decode_nested(payload: NestedDecodeInput, decoder: MediaDecoder) -> DecodeResult:
    decoded = [decoder.decode_image(image) for group in payload.groups for image in group.images]
    decoded.extend(
        decoder.decode_image(reference.image)
        for reference in payload.references
        if isinstance(reference, PictureReference)
    )
    return DecodeResult(decoded[0].width, decoded[0].height, sum(len(item.rgb) for item in decoded))


def decode_references_only(payload: ReferencesOnlyInput, decoder: MediaDecoder) -> DecodeResult:
    reference = payload.references[0]
    if not isinstance(reference, PictureReference):
        raise AssertionError("fixture supplies an image reference")
    image = decoder.decode_image(reference.image)
    return DecodeResult(image.width, image.height, len(image.rgb))


def decode_incomplete_references(
    payload: IncompleteReferencesInput, decoder: MediaDecoder
) -> DecodeResult:
    del payload, decoder
    return DecodeResult(0, 0, 0)


class TooWideInput(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(max_decoded_bytes=DEFAULT_DECODE_LIMITS.max_decoded_bytes + 1),
    ]


def decode_too_wide(payload: TooWideInput, decoder: MediaDecoder) -> DecodeResult:
    image = decoder.decode_image(payload.image)
    return DecodeResult(image.width, image.height, len(image.rgb))


def main() -> int:
    import av

    root = Path(tempfile.mkdtemp(prefix="media-decode-live-"))
    record = Attempt("decode-live", root)
    out = Outputs(record)
    decoder = MediaDecoder(record, active=lambda: False)

    print("section: real service injection + padded RGB")
    pixels = bytes(range(18))
    image = out.save_image(ImageFrame(3, 2, pixels))
    assert image._local is not None
    grant = GrantedInput("image", image._local, image.media_type, image.digest, image.size_bytes)
    envelope, outcome, _ = attempt(
        app.get("decode_through_injector"),
        {"image": "input:image"},
        Invocation(
            request_id="decode-live",
            spool=root / "invoke",
            deadline=10**30,
            assets={"image": grant},
        ),
    )
    check(
        "signature declares and receives the decoder",
        envelope is not None and envelope.result == DecodeResult(3, 2, 18),
        f"terminal={outcome.terminal} code={outcome.code}",
    )
    nested_grants = {
        "groups.0.images.0": GrantedInput(
            "groups.0.images.0", image._local, image.media_type, image.digest, image.size_bytes
        ),
        "references.0.image": GrantedInput(
            "references.0.image", image._local, image.media_type, image.digest, image.size_bytes
        ),
    }
    nested_envelope, nested_outcome, _ = attempt(
        app.get("decode_nested"),
        {
            "groups": [{"images": ["input:group"]}],
            "references": [{"type": "image", "image": "input:reference"}],
        },
        Invocation(
            request_id="decode-live",
            spool=root / "nested",
            deadline=10**30,
            assets=nested_grants,
        ),
    )
    check(
        "nested lists and concrete tagged-union fields retain their own bounds",
        nested_envelope is not None
        and nested_envelope.result == DecodeResult(3, 2, 36)
        and nested_outcome.code == "ok",
        f"terminal={nested_outcome.terminal} code={nested_outcome.code} "
        f"fields={nested_outcome.fields}",
    )
    references_only = App()
    references_only.entrypoint(decode_references_only)
    reference_envelope, reference_outcome, _ = attempt(
        references_only.get("decode_references_only"),
        {"references": [{"type": "image", "image": "input:reference"}]},
        Invocation(
            request_id="decode-live",
            spool=root / "reference-only",
            deadline=10**30,
            assets={"references.0.image": nested_grants["references.0.image"]},
        ),
    )
    check(
        "a references-only tagged union describes and hydrates its concrete leaf",
        reference_envelope is not None
        and reference_envelope.result == DecodeResult(3, 2, 18)
        and reference_outcome.code == "ok",
        f"code={reference_outcome.code} fields={reference_outcome.fields}",
    )
    incomplete = App()
    incomplete.entrypoint(decode_incomplete_references)
    try:
        describe(incomplete)
    except ConformanceError as exc:
        check(
            "an unbounded tagged-union branch refuses at its exact leaf",
            exc.code == "asset_decode_unbounded"
            and exc.fields == ("references[]{UnboundedClipReference}.video",),
            f"code={exc.code} fields={exc.fields}",
        )
    else:
        check("an unbounded tagged-union branch refuses at its exact leaf", False)
    bad_grant = GrantedInput(
        "image", image._local, "application/octet-stream", image.digest, image.size_bytes
    )
    _bad_envelope, bad_outcome, _ = attempt(
        app.get("decode_through_injector"),
        {"image": "input:image"},
        Invocation(
            request_id="decode-live",
            spool=root / "bad-media-type",
            deadline=10**30,
            assets={"image": bad_grant},
        ),
    )
    check(
        "hydration refusals identify the concrete field",
        bad_outcome.code == "asset_media_type" and bad_outcome.fields == ("image",),
        f"code={bad_outcome.code} fields={bad_outcome.fields}",
    )
    schema = render(DecodeInput)
    assert isinstance(schema, dict)
    fields = schema.get("fields")
    assert isinstance(fields, list) and isinstance(fields[0], dict)
    published = fields[0].get("asset_bound")
    check(
        "descriptor publishes every effective image decode ceiling",
        isinstance(published, dict)
        and published.get("max_decoded_bytes") == 1 << 20
        and published.get("max_pixels_per_frame") == DEFAULT_DECODE_LIMITS.max_pixels_per_frame,
    )
    media_service = _services()["media_decode"]
    assert isinstance(media_service, dict)
    aggregate_limits = media_service.get("limits")
    check(
        "descriptor publishes the attempt aggregate decoded-value ceiling",
        isinstance(aggregate_limits, dict)
        and aggregate_limits.get("max_attempt_decoded_bytes")
        == DEFAULT_DECODE_LIMITS.max_decoded_bytes
        and aggregate_limits.get("max_active_streams") == 2,
    )
    hydrate(image, input_id="image", cap=1 << 20)
    decoded_image = decoder.decode_image(image)
    check(
        "RGB row padding is stripped exactly",
        decoded_image.rgb == pixels and (decoded_image.width, decoded_image.height) == (3, 2),
    )

    print("section: exact PCM + video soundtrack")
    pcm = array("f", (index / 6000 for index in range(3000)))
    audio = out.save_audio(NdLike(pcm.tobytes(), (3000,), "float32"), sample_rate=48_000)
    hydrate(audio, input_id="audio", cap=1 << 20)
    decoded_audio = decoder.decode_audio(audio)
    check(
        "standalone native PCM clock",
        decoded_audio.channels == 1
        and decoded_audio.sample_count == 3000
        and decoded_audio.sample_rate == 48_000
        and len(decoded_audio.pcm_f32le) == 1
        and len(decoded_audio.pcm_f32le[0]) == 12_000,
        f"start={decoded_audio.start_time}",
    )
    frames = bytes((index * 7) % 256 for index in range(3 * 2 * 4 * 3))
    video = out.save_video(
        NdLike(frames, (3, 2, 4, 3), "uint8"),
        fps=24,
        audio=NdLike(pcm.tobytes(), (3000,), "float32"),
        sample_rate=48_000,
    )
    hydrate(video, input_id="video", cap=8 << 20)
    decoded_video = decoder.decode_video(video)
    check(
        "video and AAC soundtrack decode in one pass",
        decoded_video.frame_count == 3
        and len(decoded_video.frames_rgb) == 3
        and decoded_video.soundtrack is not None
        and decoded_video.soundtrack.sample_count == 3000
        and decoded_video.soundtrack.start_time == 0,
    )
    assert decoded_video.soundtrack is not None

    print("section: single-consumer incremental media")
    streamed_frames: list[DecodedVideoFrame] = []
    streamed_samples = 0
    streamed_header: DecodedMediaHeader | None = None
    with decoder.stream_video(video) as stream:
        for event in stream:
            if isinstance(event, DecodedMediaHeader):
                streamed_header = event
            elif isinstance(event, DecodedVideoFrame):
                streamed_frames.append(event)
            elif isinstance(event, DecodedAudioChunk):
                streamed_samples += event.sample_count
    check(
        "stream_video yields exact RGB clocks and soundtrack samples without a media object",
        len(streamed_frames) == decoded_video.frame_count
        and streamed_header is not None
        and streamed_header.video is not None
        and (streamed_header.video.width, streamed_header.video.height)
        == (decoded_video.width, decoded_video.height)
        and streamed_header.audio is not None
        and streamed_header.audio.sample_rate == decoded_video.soundtrack.sample_rate
        and [frame.rgb for frame in streamed_frames] == list(decoded_video.frames_rgb)
        and [frame.pts for frame in streamed_frames] == list(decoded_video.frame_pts)
        and streamed_samples == decoded_video.soundtrack.sample_count,
        f"frames={len(streamed_frames)} samples={streamed_samples}",
    )

    streamed_audio_samples = 0
    standalone_header: DecodedMediaHeader | None = None
    with decoder.stream_audio(audio) as stream:
        for event in stream:
            if isinstance(event, DecodedMediaHeader):
                standalone_header = event
            else:
                assert isinstance(event, DecodedAudioChunk)
                streamed_audio_samples += event.sample_count
    check(
        "stream_audio preserves the standalone native sample clock",
        streamed_audio_samples == decoded_audio.sample_count
        and standalone_header is not None
        and standalone_header.video is None
        and standalone_header.audio is not None
        and standalone_header.audio.sample_rate == decoded_audio.sample_rate,
        f"samples={streamed_audio_samples}",
    )

    def abandon_stream() -> None:
        with decoder.stream_video(video) as stream:
            next(stream)

    expects(
        "a successful context cannot abandon undecoded input",
        "media_stream_incomplete",
        abandon_stream,
        field="video",
    )

    third_refused = False
    with decoder.stream_video(video) as first, decoder.stream_audio(audio) as second_stream:
        try:
            with decoder.stream_video(video):
                pass
        except AuthorError as exc:
            third_refused = exc.code == "media_stream_active"
        for _event in first:
            pass
        for _event in second_stream:
            pass
    check("one decoder admits exactly two active streams", third_refused)

    with decoder.stream_video(video) as source:
        saved = out.save_video_stream(source)
    hydrate(saved.video, input_id="streamed-output", cap=8 << 20)
    streamed_roundtrip = decoder.decode_video(saved.video)
    check(
        "stream decode feeds one fragmented atomic H.264/AAC output",
        saved.frame_count == decoded_video.frame_count
        and streamed_roundtrip.frame_count == decoded_video.frame_count
        and streamed_roundtrip.soundtrack is not None,
        f"frames={saved.frame_count} audio={saved.decoded_audio_samples}",
    )
    with decoder.stream_video(video) as source:
        repeated = out.save_video_stream(source)
    check(
        "the same canonical stream reproduces exact MP4 bytes",
        repeated.video.digest == saved.video.digest,
        f"{saved.video.digest} vs {repeated.video.digest}",
    )

    with decoder.stream_video(video) as source:
        canonical_events = list(source)
    pending_before = set(record.pending)
    expects(
        "a second stream header aborts without publishing a partial asset",
        "media_header",
        lambda: out.save_video_stream(iter((canonical_events[0], canonical_events[0]))),
    )
    check(
        "failed streaming output leaves no pending handle or partial file",
        set(record.pending) == pending_before and not list(root.glob(".video-*.partial")),
    )

    wrong_duration = list(canonical_events)
    for index, event in enumerate(wrong_duration):
        if isinstance(event, DecodedVideoFrame):
            wrong_duration[index] = DecodedVideoFrame(
                width=event.width,
                height=event.height,
                rgb=event.rgb,
                pts=event.pts,
                duration=event.duration + 1,
                time_base=event.time_base,
                pixel_aspect_ratio=event.pixel_aspect_ratio,
                color_primaries=event.color_primaries,
                color_transfer=event.color_transfer,
                color_matrix=event.color_matrix,
                color_range=event.color_range,
            )
            break
    expects(
        "a frame duration contradicting the output rate refuses before commit",
        "video_discontinuity",
        lambda: out.save_video_stream(iter(wrong_duration)),
    )

    def corrupt_late() -> Any:
        for index, event in enumerate(canonical_events):
            if index == len(canonical_events) - 1:
                raise MediaDecodeError(
                    "later shot bytes are corrupt",
                    code="media_corrupt",
                    fields=("shots.7.video",),
                )
            yield event

    expects(
        "late-shot corruption preserves its field and commits no partial output",
        "media_corrupt",
        lambda: out.save_video_stream(iter(corrupt_late())),
        field="shots.7.video",
    )
    check(
        "late corruption leaves no new pending or partial streaming output",
        set(record.pending) == pending_before and not list(root.glob(".video-*.partial")),
    )

    capped_record = Attempt("stream-cap", root / "stream-cap", max_output_bytes=128)
    capped_record.spool.mkdir()
    capped_out = Outputs(capped_record)
    expects(
        "the output cap fires during streaming container production",
        "output_too_large",
        lambda: capped_out.save_video_stream(iter(canonical_events)),
    )
    check(
        "a cap refusal leaves the bounded output spool empty",
        not capped_record.pending and not list(capped_record.spool.iterdir()),
    )

    silent_source = out.save_video(NdLike(frames, (3, 2, 4, 3), "uint8"), fps=24)
    hydrate(silent_source, input_id="silent-source", cap=8 << 20)
    with decoder.stream_video(silent_source) as source:
        silent_saved = out.save_video_stream(source)
    check(
        "a silent stream emits no invented audio track",
        silent_saved.submitted_audio_samples == 0 and silent_saved.decoded_audio_samples == 0,
    )

    color_width, color_height = 128, 96
    color_rgb = bytes(
        channel
        for y in range(color_height)
        for x in range(color_width)
        for channel in (x * 255 // 127, y * 255 // 95, (x + y) * 255 // 222)
    )

    def color_events(*, primaries: int, transfer: int, matrix: int) -> Any:
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(
                width=color_width,
                height=color_height,
                time_base=Fraction(1, 24),
                pixel_aspect_ratio=Fraction(1, 1),
                nominal_frame_rate=Fraction(24, 1),
                color_primaries=primaries,
                color_transfer=transfer,
                color_matrix=matrix,
                color_range=1,
            ),
            audio=None,
        )
        yield DecodedVideoFrame(
            width=color_width,
            height=color_height,
            rgb=color_rgb,
            pts=0,
            duration=1,
            time_base=Fraction(1, 24),
            pixel_aspect_ratio=Fraction(1, 1),
            color_primaries=primaries,
            color_transfer=transfer,
            color_matrix=matrix,
            color_range=1,
        )

    bt709 = out.save_video_stream(color_events(primaries=1, transfer=1, matrix=1))
    itu601 = out.save_video_stream(color_events(primaries=6, transfer=6, matrix=5))
    assert bt709.video._local is not None and itu601.video._local is not None
    bt709_rgb = undecorated_rgb(av, bt709.video._local)
    itu601_rgb = undecorated_rgb(av, itu601.video._local)
    bt709_error = max(abs(left - right) for left, right in zip(color_rgb, bt709_rgb, strict=True))
    itu601_error = max(abs(left - right) for left, right in zip(color_rgb, itu601_rgb, strict=True))
    with av.open(str(bt709.video._local)) as color_probe:
        bt709_matrix = int(color_probe.streams.video[0].codec_context.colorspace)
    with av.open(str(itu601.video._local)) as color_probe:
        itu601_matrix = int(color_probe.streams.video[0].codec_context.colorspace)
    check(
        "BT.709 and ITU601 use their declared RGB-to-YUV matrices and retain their tags",
        bt709_matrix == 1
        and itu601_matrix == 5
        and bt709_error <= 20
        and itu601_error <= 20
        and bt709.video.digest != itu601.video.digest,
        f"matrix={bt709_matrix}/{itu601_matrix} max_error={bt709_error}/{itu601_error}",
    )

    unspecified_hd_path = root / "unspecified-hd.mp4"
    expected_hd = unspecified_hd_mp4(av, unspecified_hd_path)
    unspecified_hd = input_asset(unspecified_hd_path, ref="unspecified-hd")
    decoded_hd = decoder.decode_video(unspecified_hd)
    hd_error = max(
        abs(left - right) for left, right in zip(expected_hd, decoded_hd.frames_rgb[0], strict=True)
    )
    with decoder.stream_video(unspecified_hd) as source:
        hd_header = next(source)
        for _event in source:
            pass
    assert isinstance(hd_header, DecodedMediaHeader) and hd_header.video is not None
    check(
        "unspecified HD tags normalize to the BT.709 matrix before RGB conversion",
        hd_header.video.color_primaries == 1
        and hd_header.video.color_transfer == 1
        and hd_header.video.color_matrix == 1
        and hd_header.video.color_range == 1
        and hd_error <= 20,
        f"facts={hd_header.video} max_error={hd_error}",
    )

    class NoSnapshotVideo(VideoAsset):
        def read_bytes(self) -> bytes:
            raise AssertionError("encoded video was snapshotted")

    direct_asset = NoSnapshotVideo("direct-video")
    assert video._local is not None
    bind(
        direct_asset,
        local=video._local,
        attempt="decode-live",
        media_type=video.media_type,
        digest=video.digest,
        length=video.size_bytes,
        max_decoded_bytes=8 << 20,
        input_id="direct-video",
    )
    direct = decoder.decode_video(direct_asset)
    check(
        "snapshot decode also opens the verified spool object directly",
        direct.frame_count == decoded_video.frame_count,
    )

    missing_stream = VideoAsset("missing-stream")
    bind(
        missing_stream,
        local=root / "missing-stream.mp4",
        attempt="decode-live",
        media_type="video/mp4",
        digest=digest_bytes(b"missing"),
        length=7,
        max_decoded_bytes=8 << 20,
        input_id="missing-stream",
    )

    def open_missing_stream() -> None:
        with decoder.stream_video(missing_stream) as stream:
            next(stream)

    expects(
        "lost streamed spool bytes retain Runtime ownership",
        "media_materialization",
        open_missing_stream,
        field="missing-stream",
    )

    malformed_path = root / "malformed-retained.mp4"
    malformed_path.write_bytes(b"not an mp4")
    malformed_stream = VideoAsset("malformed-retained")
    bind(
        malformed_stream,
        local=malformed_path,
        attempt="decode-live",
        media_type="video/mp4",
        digest=digest_bytes(malformed_path.read_bytes()),
        length=malformed_path.stat().st_size,
        max_decoded_bytes=8 << 20,
        input_id="malformed-retained",
    )
    before_fds = len(list(Path("/proc/self/fd").iterdir()))
    failed_streams = []
    for _ in range(40):
        failed = decoder.stream_video(malformed_stream)
        failed_streams.append(failed)
        try:
            with failed:
                pass
        except AuthorError as exc:
            assert exc.code == "media_malformed"
    after_fds = len(list(Path("/proc/self/fd").iterdir()))
    check(
        "retaining failed stream handles leaks no file descriptors",
        after_fds == before_fds and len(failed_streams) == 40,
        f"fds {before_fds}->{after_fds}",
    )

    print("section: coarse Opus container clock")
    webm_path = root / "coarse.webm"
    expected_frames, expected_samples = webm(av, webm_path)
    decoded_webm = decoder.decode_video(input_asset(webm_path, ref="webm"))
    check(
        "Opus pre-skip/coarse duration does not invent truncation",
        decoded_webm.frame_count == expected_frames
        and decoded_webm.soundtrack is not None
        and decoded_webm.soundtrack.sample_count == expected_samples,
        f"samples={decoded_webm.soundtrack.sample_count if decoded_webm.soundtrack else 0}",
    )
    gap_path = root / "gap.webm"
    discontinuous_webm(av, gap_path, second_pts=1920)
    expects(
        "a real Opus presentation gap refuses instead of compacting PCM",
        "audio_discontinuity",
        lambda: decoder.decode_video(input_asset(gap_path, ref="gap")),
        field="gap",
    )
    overlap_path = root / "overlap.webm"
    discontinuous_webm(av, overlap_path, second_pts=480)
    expects(
        "a real Opus presentation overlap refuses instead of compacting PCM",
        "audio_discontinuity",
        lambda: decoder.decode_video(input_asset(overlap_path, ref="overlap")),
        field="overlap",
    )
    offset_path = root / "offset.webm"
    offset_webm(av, offset_path)
    offset_video = decoder.decode_video(input_asset(offset_path, ref="offset"))
    check(
        "a nonzero soundtrack offset remains on the shared container timeline",
        offset_video.start_time == 0
        and offset_video.soundtrack is not None
        and offset_video.soundtrack.start_time == Fraction(101, 1000),
        f"video={offset_video.start_time} "
        f"audio={offset_video.soundtrack.start_time if offset_video.soundtrack else None}",
    )

    print("section: display transforms")
    pure, flipped = root / "pure.mp4", root / "flipped.mp4"
    rotated_mp4(av, pure, hflip=False)
    rotated_mp4(av, flipped, hflip=True)
    rotated = decoder.decode_video(input_asset(pure, ref="pure-rotation"))
    check(
        "pure quarter rotation applies the exact pixel order",
        (rotated.width, rotated.height) == (2, 4)
        and rotated.frames_rgb[0]
        == reordered_pixels(undecorated_rgb(av, pure), (3, 7, 2, 6, 1, 5, 0, 4)),
    )
    expects(
        "a mirrored display matrix refuses instead of losing the flip",
        "media_rotation",
        lambda: decoder.decode_video(input_asset(flipped, ref="flipped")),
    )
    jpeg_rotated = decoder.decode_image(
        input_image(root / "rotated.jpg", jpeg(av, orientation=6), ref="jpeg-rotation")
    )
    check(
        "JPEG EXIF pure rotation applies the exact pixel order without Pillow",
        (jpeg_rotated.width, jpeg_rotated.height) == (2, 4)
        and jpeg_rotated.rgb
        == reordered_pixels(
            undecorated_image_rgb(av, root / "rotated.jpg"), (4, 0, 5, 1, 6, 2, 7, 3)
        ),
    )
    expects(
        "JPEG EXIF mirror refuses instead of sharing the rotation-only path",
        "media_rotation",
        lambda: decoder.decode_image(
            input_image(root / "mirrored.jpg", jpeg(av, orientation=2), ref="jpeg-mirror")
        ),
    )

    print("section: retained-value and lifetime refusals")
    tiny = ImageAsset("tiny-cap")
    assert image._local is not None
    bind(
        tiny,
        local=image._local,
        attempt="decode-live",
        media_type=image.media_type,
        digest=image.digest,
        length=image.size_bytes,
        max_decoded_bytes=17,
        input_id="tiny",
    )
    expects(
        "retained decoded-value cap",
        "decoded_bytes_limit",
        lambda: decoder.decode_image(tiny),
        field="tiny",
    )
    expects(
        "active component scope",
        "decode_in_component_scope",
        lambda: MediaDecoder(record, active=lambda: True).decode_image(image),
    )
    missing = ImageAsset("missing-bound")
    bind(
        missing,
        local=image._local,
        attempt="decode-live",
        media_type=image.media_type,
        digest=image.digest,
        length=image.size_bytes,
        input_id="missing",
    )
    expects(
        "missing field decoded-value bound",
        "decoded_bound_missing",
        lambda: decoder.decode_image(missing),
    )
    foreign = ImageAsset("foreign")
    bind(
        foreign,
        local=image._local,
        attempt="other",
        media_type=image.media_type,
        digest=image.digest,
        length=image.size_bytes,
        max_decoded_bytes=1 << 20,
        input_id="foreign",
    )
    expects("foreign attempt", "foreign_asset", lambda: decoder.decode_image(foreign))
    closed = Attempt("decode-live", root / "closed", closed=True)
    expects(
        "escaped decoder handle",
        "escaped_handle",
        lambda: MediaDecoder(closed, active=lambda: False).decode_image(image),
    )
    limits = DecodeLimits(1 << 20, 5, 10, 2, 48_000)
    expects(
        "structural pixel ceiling",
        "pixels_per_frame_limit",
        lambda: MediaDecoder(record, active=lambda: False, limits=limits).decode_image(image),
    )
    aggregate = MediaDecoder(
        record,
        active=lambda: False,
        limits=DecodeLimits(30, 1000, 10, 2, 48_000),
    )
    aggregate.decode_image(image)
    second = ImageAsset("second")
    bind(
        second,
        local=image._local,
        attempt="decode-live",
        media_type=image.media_type,
        digest=image.digest,
        length=image.size_bytes,
        max_decoded_bytes=1 << 20,
        input_id="second",
    )
    expects(
        "attempt aggregate retained-value cap",
        "decoded_bytes_limit",
        lambda: aggregate.decode_image(second),
        field="second",
    )

    print("section: malformed bytes + absent media extra")
    corrupt_path = root / "corrupt.png"
    corrupt_path.write_bytes(image._local.read_bytes()[:20])
    corrupt = ImageAsset("corrupt")
    bind(
        corrupt,
        local=corrupt_path,
        attempt="decode-live",
        media_type="image/png",
        digest=digest_bytes(corrupt_path.read_bytes()),
        length=corrupt_path.stat().st_size,
        max_decoded_bytes=1 << 20,
        input_id="corrupt",
    )
    expects(
        "truncated container",
        "media_malformed",
        lambda: decoder.decode_image(corrupt),
        field="corrupt",
    )
    unavailable_path = root / "removed.png"
    unavailable = ImageAsset("removed")
    bind(
        unavailable,
        local=unavailable_path,
        attempt="decode-live",
        media_type="image/png",
        digest=image.digest,
        length=image.size_bytes,
        max_decoded_bytes=1 << 20,
        input_id="removed",
    )
    expects(
        "missing verified spool bytes are a Runtime-owned field failure",
        "media_materialization",
        lambda: decoder.decode_image(unavailable),
        field="removed",
    )
    _missing_envelope, missing_outcome, _ = attempt(
        app.get("decode_through_injector"),
        {"image": "input:image"},
        Invocation(
            request_id="decode-live",
            spool=root / "missing-materialization",
            deadline=10**30,
            assets={
                "image": GrantedInput(
                    "image",
                    unavailable_path,
                    "image/png",
                    image.digest,
                    image.size_bytes,
                )
            },
        ),
    )
    check(
        "lost admitted bytes report Runtime ownership",
        missing_outcome.terminal == "failed"
        and missing_outcome.origin == "runtime"
        and missing_outcome.code == "media_materialization"
        and missing_outcome.fields == ("image",),
        f"terminal={missing_outcome.terminal} origin={missing_outcome.origin} "
        f"code={missing_outcome.code} fields={missing_outcome.fields}",
    )
    too_wide = App()
    too_wide.entrypoint(decode_too_wide)
    try:
        describe(too_wide)
    except ConformanceError as exc:
        check(
            "a field cannot advertise above Runtime's effective decoded ceiling",
            exc.code == "asset_decode_limit" and exc.fields == ("image",),
            f"code={exc.code} fields={exc.fields}",
        )
    else:
        check("a field cannot advertise above Runtime's effective decoded ceiling", False)
    held = sys.modules.get("av")
    sys.modules["av"] = cast("Any", None)
    try:
        missing_dependency = App()
        missing_dependency.entrypoint(decode_through_injector)
        try:
            describe(missing_dependency)
        except ConformanceError as exc:
            check(
                "a decoder-declaring release cannot build without PyAV",
                exc.code == "media_decoder_dependency",
                f"code={exc.code}",
            )
        else:
            check("a decoder-declaring release cannot build without PyAV", False)
        expects("absent PyAV", "decoder_unavailable", lambda: decoder.decode_image(image))
    finally:
        if held is None:
            sys.modules.pop("av", None)
        else:
            sys.modules["av"] = held

    print(f"\nmedia decode: {PASS} ok, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
