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
    ImageAsset,
    ImageFrame,
    Invocation,
    MediaDecoder,
    VideoAsset,
    attempt,
)
from cozy_runtime.author._assets import GrantedInput, bind, digest_bytes  # noqa: E402
from cozy_runtime.author._decode import (  # noqa: E402
    DecodeLimits,
    DecoderUnavailable,
    MediaDecodeError,
)
from cozy_runtime.author._errors import CapabilityError  # noqa: E402
from cozy_runtime.author._services import Attempt, Outputs  # noqa: E402

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
    except (CapabilityError, MediaDecodeError, DecoderUnavailable) as exc:
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
        and decoded_video.soundtrack.sample_count == 3000,
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

    print("section: display transforms")
    pure, flipped = root / "pure.mp4", root / "flipped.mp4"
    rotated_mp4(av, pure, hflip=False)
    rotated_mp4(av, flipped, hflip=True)
    rotated = decoder.decode_video(input_asset(pure, ref="pure-rotation"))
    check("pure quarter rotation is applied", (rotated.width, rotated.height) == (2, 4))
    expects(
        "a mirrored display matrix refuses instead of losing the flip",
        "media_rotation",
        lambda: decoder.decode_video(input_asset(flipped, ref="flipped")),
    )
    jpeg_rotated = decoder.decode_image(
        input_image(root / "rotated.jpg", jpeg(av, orientation=6), ref="jpeg-rotation")
    )
    check(
        "JPEG EXIF pure rotation is applied without Pillow",
        (jpeg_rotated.width, jpeg_rotated.height) == (2, 4),
    )
    expects(
        "JPEG EXIF mirror refuses instead of sharing the rotation-only path",
        "media_rotation",
        lambda: decoder.decode_image(
            input_image(root / "mirrored.jpg", jpeg(av, orientation=2), ref="jpeg-mirror")
        ),
    )

    print("section: expansion and lifetime refusals")
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
        "decoded expansion cap",
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
        "missing field expansion bound",
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
        "attempt aggregate expansion cap",
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
    held = sys.modules.get("av")
    sys.modules["av"] = cast("Any", None)
    try:
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
