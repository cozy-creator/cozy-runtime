"""Media inputs are handles read by spec, driven with real files through the real kernel.

Every source here is encoded at test time with PyAV: a 2-minute 1080p reference at 2 fps
with an AAC soundtrack, a 125-second clip past the 3600-frame ceiling, constant- and
variable-frame-rate clips, and a live WebM whose container records no duration.
"""

from __future__ import annotations

import math
import sys
import time
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any

import av
import msgspec
import numpy as np
import pytest
from PIL import Image, ImageOps

from cozy_runtime.author import (
    AdjustmentRow,
    App,
    AssetBound,
    AssetLimits,
    Assets,
    AudioAsset,
    AudioPreparation,
    CapabilityError,
    ConformanceError,
    DecodedAudio,
    DecodedAudioChunk,
    DecodedMediaHeader,
    DecodedVideo,
    DecodedVideoFrame,
    ImageAsset,
    ImagePreparation,
    InvalidRequest,
    Invocation,
    MediaDecodeError,
    VideoAsset,
    VideoPreparation,
    attempt,
    describe,
    fakes,
)
from cozy_runtime.author._assets import GrantedInput
from cozy_runtime.author._decode import DEFAULT_DECODE_LIMITS
from cozy_runtime.author._media import KIND_MEDIA
from cozy_runtime.internal import (
    executor_replies,
    interface_wheel,
    package_interface,
    schema,
    static_interface,
)
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.executor import adjustment_row
from media_sources import mp4, steady

HD = 1920 * 1080 * 3


@pytest.fixture(scope="module")
def media(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("media")
    long_starts, long_end = steady(2, 240)
    many_starts, many_end = steady(30, 3750)
    clip_starts, clip_end = steady(30, 60)
    vfr = [Fraction(0)]
    for gap in [Fraction(1, 30), Fraction(1, 15), Fraction(1, 60), Fraction(1, 24)] * 10:
        vfr.append(vfr[-1] + gap)
    paths = {
        "reference": mp4(
            root / "reference.mp4",
            size=(1920, 1080),
            starts=long_starts,
            end=long_end,
            audio_seconds=120,
        ),
        "many": mp4(root / "many.mp4", size=(64, 36), starts=many_starts, end=many_end),
        "clip": mp4(
            root / "clip.mp4",
            size=(64, 48),
            starts=clip_starts,
            end=clip_end,
            audio_seconds=2,
            sample_rate=48000,
        ),
        "vfr": mp4(root / "vfr.mp4", size=(64, 48), starts=vfr[:-1], end=vfr[-1]),
    }
    live = root / "live.webm"
    with av.open(str(live), "w", format="webm", options={"live": "1"}) as out:
        stream = out.add_stream("libvpx", rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(90):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), index * 2, np.uint8), format="rgb24"
            )
            frame.pts = index
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    paths["live"] = live
    photo = root / "photo.jpg"
    exif = Image.Exif()
    exif[274] = 6  # displayed rotated a quarter turn: 120x80 stored, 80x120 shown
    pixels = np.random.default_rng(7).integers(0, 256, (80, 120, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(photo, format="JPEG", exif=exif, quality=95)
    paths["photo"] = photo
    wide = root / "wide.png"
    Image.new("RGB", (200, 100), (10, 200, 30)).save(wide)
    paths["wide"] = wide
    voice = fakes.fake_outputs(spool=root / "voice").save_audio(
        (0.1 * np.sin(np.arange(24000, dtype=np.float32) / 9)).astype(np.float32),
        sample_rate=24000,
    )
    assert voice._local is not None
    paths["voice"] = voice._local
    return paths


def _run(
    app: App, name: str, tmp_path: Path, wire: dict[str, Any], grants: dict[str, GrantedInput]
) -> tuple[Any, Any, Any]:
    return attempt(
        app.get(name),
        wire,
        Invocation(name, tmp_path / name, time.monotonic() + 120, assets=grants),
    )


# ---------------------------------------------------------------- truncation at hydration

TRUNCATE = VideoPreparation(min_seconds=2, max_seconds=15, excess="truncate")


class Reference(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(max_bytes=1 << 30, max_decoded_bytes=1 << 30), TRUNCATE]


class Seen(msgspec.Struct):
    frames: int
    duration: str
    probed: list[str]
    samples: int
    retained: list[int]
    slots: int
    distinct: int


references = App()


@references.entrypoint(accelerator=False)
def first_fifteen(payload: Reference) -> Seen:
    video = payload.video
    probe = video.probe()
    media = video._media
    assert media is not None
    natural = video.read()
    assert natural.soundtrack is not None
    after_natural = media.retained_bytes
    seen = Seen(
        natural.frame_count,
        str(natural.duration),
        [str(probe.duration), str(probe.source_duration), f"{probe.width}x{probe.height}"],
        natural.soundtrack.sample_count,
        [after_natural],
        0,
        0,
    )
    del natural
    mapped = video.read(fps=24, audio=False)
    seen.retained.append(media.retained_bytes - after_natural)
    seen.slots = mapped.frame_count
    seen.distinct = len({id(frame) for frame in mapped.frames_rgb})
    assert mapped.duration == 15 and mapped.soundtrack is None
    return seen


def test_two_minute_1080p_reference_reads_its_first_fifteen_seconds(
    tmp_path: Path, media: dict[str, Path]
) -> None:
    grant = fakes.granted_input(media["reference"], input_id="video")
    result, outcome, record = _run(
        references, "first_fifteen", tmp_path, {"video": grant.digest}, {"video": grant}
    )
    assert outcome.terminal == "succeeded", outcome
    seen = result.result
    # 30 kept frames at 2 fps, the last one ending exactly at 15 s; the 225 after it are
    # never converted, and the soundtrack ends on the sample at 15 s.
    assert (seen.frames, seen.duration, seen.samples) == (30, "15", 15 * 16000)
    assert seen.probed == ["15", "120", "1920x1080"]
    assert seen.retained == [30 * HD + 15 * 16000 * 4, 30 * HD]
    # The 24-fps clock repeats each kept frame by reference: 360 slots, 30 retained frames.
    assert (seen.slots, seen.distinct) == (360, 30)
    assert result.adjustments == (
        AdjustmentRow("adjusted", "video", 120.0, 15, "video truncated to 15 s", source="policy"),
    )
    warnings = [o for o in record.ring if o.kind == "log" and o.value == "warn"]
    assert [o.name for o in warnings] == ["input video truncated"]
    assert adjustment_row(result.adjustments[0]) == {
        "field": "video",
        "kind": "adjusted",
        "requested": "120.0",
        "applied": "15",
        "reason": "video truncated to 15 s",
        "source": "policy",
    }


@pytest.mark.parametrize(
    "source,preparation,code",
    [
        ("many", VideoPreparation(max_seconds=15), "media_too_long"),
        (
            "clip",
            VideoPreparation(min_seconds=3, max_seconds=15, excess="truncate"),
            "media_too_short",
        ),
    ],
    ids=["above-maximum-refuses", "below-minimum-refuses"],
)
def test_duration_policy_refuses_naming_the_field_before_the_handler(
    tmp_path: Path, media: dict[str, Path], source: str, preparation: VideoPreparation, code: str
) -> None:
    field = Annotated[VideoAsset, AssetBound(max_decoded_bytes=1 << 30), preparation]
    strict_request = msgspec.defstruct("Strict", [("video", field)])

    def strict(payload: Any) -> Done:
        raise AssertionError("a refused input never reaches the handler")

    strict.__annotations__ = {"payload": strict_request, "return": Done}
    app = App()
    app.entrypoint(strict, accelerator=False)
    grant = fakes.granted_input(media[source], input_id="video")
    result, outcome, record = _run(
        app, "strict", tmp_path, {"video": grant.digest}, {"video": grant}
    )
    assert result is None and not record.entered
    assert (outcome.terminal, outcome.code, outcome.fields) == ("refused", code, ("video",))


def test_audio_preparation_refuses_a_short_voice(media: dict[str, Path]) -> None:
    with pytest.raises(InvalidRequest) as caught:
        fakes.file_input(AudioAsset, media["voice"], preparation=AudioPreparation(min_seconds=2))
    assert caught.value.code == "media_too_short" and caught.value.fields == ("input",)


# ---------------------------------------------------------------- end-bounded decode


def test_a_source_past_the_frame_ceiling_reads_and_streams_its_first_seconds(
    media: dict[str, Path],
) -> None:
    record = fakes.fake_attempt("ceiling")
    video = fakes.file_input(VideoAsset, media["many"], attempt=record)
    assert video.probe().source_duration == 125  # 3750 declared frames
    first = video.read(end=15)
    assert first.frame_count == 450 and first.duration == 15
    assert record.media.retained_bytes == 450 * 64 * 36 * 3
    with video.stream(end=15) as stream:
        frames = [event for event in stream if isinstance(event, DecodedVideoFrame)]
    assert len(frames) == 450 and frames[-1].start_time == Fraction(449, 30)
    with pytest.raises(MediaDecodeError) as caught:
        video.read()  # every frame kept: 3750 crosses the 3600-frame ceiling
    assert caught.value.code == "video_frames_limit"
    with video.stream() as stream:  # a stream retains one frame: no frame ceiling
        assert sum(isinstance(event, DecodedVideoFrame) for event in stream) == 3750


def test_the_soundtrack_is_cut_on_the_exact_sample_and_the_last_frame_clipped(
    media: dict[str, Path],
) -> None:
    record = fakes.fake_attempt("cut")
    video = fakes.file_input(VideoAsset, media["clip"], attempt=record)
    end = Fraction(7, 4)  # mid-frame at 30 fps and mid-AAC-frame at 48 kHz
    full = video.read()
    cut = video.read(end=end)
    assert full.soundtrack is not None and cut.soundtrack is not None
    assert cut.frame_count == 53 and cut.duration == end
    assert cut.frames_rgb == full.frames_rgb[:53]
    assert cut.frame_durations[-1] * cut.time_base == end - Fraction(52, 30)
    kept = 84000
    assert cut.soundtrack.start_time == full.soundtrack.start_time
    assert cut.soundtrack.sample_count == kept and cut.soundtrack.duration == end
    assert cut.soundtrack.pcm_f32le[0] == full.soundtrack.pcm_f32le[0][: kept * 4]

    with video.stream(end=end) as stream:
        events = list(stream)
    assert isinstance(events[0], DecodedMediaHeader)
    pictures = [e for e in events if isinstance(e, DecodedVideoFrame)]
    chunks = [e for e in events if isinstance(e, DecodedAudioChunk)]
    assert len(pictures) == 53 and pictures[-1].duration == cut.frame_durations[-1]
    assert sum(chunk.sample_count for chunk in chunks) == kept
    with video.stream(end=end, pictures=False) as stream:
        soundtrack = [e for e in stream if isinstance(e, DecodedAudioChunk)]
    assert b"".join(c.pcm_f32le[0] for c in soundtrack) == cut.soundtrack.pcm_f32le[0]


def _h3_slots(video: DecodedVideo, fps: Fraction) -> list[int]:
    """minimax-h3's `_video_at_24fps` selection, at any rate: each frame shows until the
    next starts, and each boundary rounds to the nearest whole slot."""
    starts, origin = video.frame_pts, video.frame_pts[0]
    boundaries = [Fraction(value - origin) * video.time_base for value in starts]
    boundaries.append(Fraction(starts[-1] + video.frame_durations[-1] - origin) * video.time_base)
    slots = [math.floor(boundary * fps + Fraction(1, 2)) for boundary in boundaries]
    return [
        index for index in range(video.frame_count) for _ in range(slots[index + 1] - slots[index])
    ]


@pytest.mark.parametrize("fps", [Fraction(24), Fraction(24000, 1001)], ids=["24", "23.976"])
@pytest.mark.parametrize("source", ["clip", "vfr"], ids=["30fps", "variable"])
def test_a_constant_clock_read_selects_the_frames_h3_selects(
    media: dict[str, Path], source: str, fps: Fraction
) -> None:
    record = fakes.fake_attempt("clock")
    video = fakes.file_input(VideoAsset, media[source], attempt=record)
    full = video.read(audio=False)
    if source == "vfr":
        assert len(set(full.frame_durations)) == 4
    before = record.media.retained_bytes
    mapped = video.read(fps=fps, audio=False)
    selected = _h3_slots(full, fps)
    assert mapped.frame_count == len(selected) and mapped.time_base == 1 / fps
    assert mapped.frame_pts == tuple(range(len(selected)))
    assert set(mapped.frame_durations) == {1}
    assert list(mapped.frames_rgb) == [full.frames_rgb[index] for index in selected]
    assert (mapped.width, mapped.height) == (full.width, full.height)  # never resized
    distinct = len(set(selected))
    assert record.media.retained_bytes - before == distinct * full.width * full.height * 3


def test_a_live_webm_without_a_recorded_duration_is_measured_from_packets(
    media: dict[str, Path],
) -> None:
    with av.open(str(media["live"])) as container:
        assert container.streams.video[0].duration is None
    record = fakes.fake_attempt("live")
    video = fakes.file_input(
        VideoAsset,
        media["live"],
        attempt=record,
        preparation=VideoPreparation(max_seconds=2, excess="truncate"),
    )
    probe = video.probe()
    assert probe.source_duration == 3 and probe.duration == 2
    read = video.read()
    assert read.duration == 2 and read.frame_count == 60
    assert [row.field for row in record.rows] == ["input"]


def test_a_read_needs_no_declared_decode_bound(media: dict[str, Path]) -> None:
    """A field that declares none reads within what the machine holds (owner, 2026-10-17)."""
    video = fakes.file_input(VideoAsset, media["clip"], max_decoded_bytes=0)
    assert video.probe().source_duration == 2
    assert video.read().duration == 2


def test_reads_refuse_inside_a_component_scope(media: dict[str, Path]) -> None:
    record = fakes.fake_attempt("scope")
    video = fakes.file_input(VideoAsset, media["clip"], attempt=record)
    video.probe()
    record.components_active = lambda: True
    for read in (video.read, video.probe, lambda: video.stream().__enter__()):
        with pytest.raises(CapabilityError) as caught:
            read()
        assert caught.value.code == "decode_in_component_scope"
    record.components_active = lambda: False
    record.closed = True
    with pytest.raises(CapabilityError, match="escaped_handle"):
        video.probe()


# ---------------------------------------------------------------- collections and images

Mixed = Annotated[
    Assets[ImageAsset | VideoAsset | AudioAsset],
    AssetLimits(images=1, videos=1, audio=1),
    ImagePreparation(max_edge=64),
    VideoPreparation(max_seconds=1, excess="truncate"),
    AudioPreparation(min_seconds=0.5),
]


class Empty(msgspec.Struct):
    pass


class MixedSeen(msgspec.Struct):
    image: list[int]
    video: list[str]
    audio: list[str]


collections = App()


@collections.entrypoint(accelerator=False)
def mix(payload: Empty, assets: Mixed) -> MixedSeen:
    image, video, audio = assets["photo"], assets["clip"], assets["voice"]
    assert isinstance(image, ImageAsset) and isinstance(video, VideoAsset)
    assert isinstance(audio, AudioAsset)
    picture, clip, voice = image.read(), video.read(), audio.read()
    assert isinstance(clip, DecodedVideo) and isinstance(voice, DecodedAudio)
    return MixedSeen(
        [*picture.size, image.probe().source_width],
        [str(clip.duration), str(video.probe().source_duration), str(clip.frame_count)],
        [str(voice.duration), str(voice.sample_rate)],
    )


def test_an_assets_collection_prepares_each_kind(tmp_path: Path, media: dict[str, Path]) -> None:
    grants = {
        f"assets.{position}.asset": fakes.granted_input(
            media[name], input_id=f"assets.{position}.asset", order=position
        )
        for position, name in enumerate(("wide", "clip", "voice"))
    }
    wire = {
        "assets": [
            {"asset": grants["assets.0.asset"].digest, "label": "photo"},
            {"asset": grants["assets.1.asset"].digest, "label": "clip"},
            {"asset": grants["assets.2.asset"].digest, "label": "voice"},
        ]
    }
    result, outcome, _ = _run(collections, "mix", tmp_path, wire, grants)
    assert outcome.terminal == "succeeded", outcome
    assert result.result.image == [64, 32, 200]
    assert result.result.video == ["1", "2", "30"]
    assert result.result.audio == ["1", "24000"]
    assert [(row.field, row.kind, row.source) for row in result.adjustments] == [
        ("assets.1.asset", "adjusted", "policy")
    ]
    assert "media_decode" in result.capabilities


class Photo(msgspec.Struct):
    image: Annotated[
        ImageAsset, AssetBound(max_decoded_bytes=1 << 20), ImagePreparation(max_edge=48)
    ]


class PhotoSeen(msgspec.Struct):
    size: list[int]
    probed: list[int]
    pixels: str
    retained: int


photos = App()


@photos.entrypoint(accelerator=False)
def photo(payload: Photo) -> PhotoSeen:
    image = payload.image.read()
    probe = payload.image.probe()
    media = payload.image._media
    assert media is not None
    return PhotoSeen(
        list(image.size),
        [probe.width, probe.height, probe.source_width, probe.source_height],
        image.tobytes().hex(),
        media.retained_bytes,
    )


def test_a_single_image_field_reads_prepared_pillow_pixels(
    tmp_path: Path, media: dict[str, Path]
) -> None:
    grant = fakes.granted_input(media["photo"], input_id="image")
    result, outcome, _ = _run(photos, "photo", tmp_path, {"image": grant.digest}, {"image": grant})
    assert outcome.terminal == "succeeded", outcome
    seen = result.result
    with Image.open(media["photo"]) as source:
        oracle = ImageOps.exif_transpose(source).convert("RGB")
    expected = oracle.resize((32, 48), Image.Resampling.LANCZOS)
    assert seen.size == [32, 48] and seen.probed == [32, 48, 80, 120]
    assert bytes.fromhex(seen.pixels) == expected.tobytes()
    assert seen.retained == 32 * 48 * 3  # charged after preparation, not at source size


# ---------------------------------------------------------------- interface and reply

SOURCE = """from typing import Annotated
import msgspec
from cozy_runtime.author import (
    App, AssetBound, AssetLimits, Assets, Context, ImageAsset, ImagePreparation, VideoAsset,
    VideoPreparation, invocable,
)
References = Annotated[
    Assets[ImageAsset | VideoAsset],
    AssetLimits(images=2, videos=1),
    ImagePreparation(max_edge=1024),
    VideoPreparation(max_seconds=15, excess="truncate"),
]
class Shot(msgspec.Struct):
    clip: Annotated[
        VideoAsset,
        AssetBound(max_bytes=1 << 30, max_decoded_bytes=4 << 30),
        VideoPreparation(min_seconds=2, max_seconds=15, excess="truncate"),
    ]
class Done(msgspec.Struct):
    ok: bool
@invocable
async def render(ctx: Context, *, shot: Shot, refs: References) -> Done:
    raise RuntimeError("source only")
app = App()
app.entrypoint(render, accelerator=False)
"""


def test_the_interface_carries_each_preparation_profile(tmp_path: Path) -> None:
    sys.modules.pop("prepared_media", None)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="prepared-media"\nversion="0.1.0"\n')
    (tmp_path / "package.toml").write_text('[application]\nobject="prepared_media:app"\n')
    (tmp_path / "prepared_media.py").write_text(SOURCE)
    interface = package_interface.build(discover(tmp_path))
    raw = package_interface.canonical_bytes(interface)
    assert raw == package_interface.canonical_bytes(static_interface.build(tmp_path))
    entry = interface["entrypoints"][0]
    assert entry["assets"] == {
        "parameter": "refs",
        "kinds": [
            {
                "kind": "image",
                "media_types": list(KIND_MEDIA["image"]),
                "max_count": 2,
                "max_decoded_bytes": DEFAULT_DECODE_LIMITS.max_decoded_bytes,
                "prepare": {"profile": "image-fit/1", "max_edge": 1024},
            },
            {
                "kind": "video",
                "media_types": list(KIND_MEDIA["video"]),
                "max_count": 1,
                "max_decoded_bytes": DEFAULT_DECODE_LIMITS.max_decoded_bytes,
                "prepare": {"profile": "video-trim/1", "excess": "truncate", "max_seconds": 15},
            },
        ],
    }
    shot = next(f for f in entry["request"]["fields"] if f["name"] == "shot")
    assert shot["type"]["fields"][0] == {
        "name": "clip",
        "type": {"asset": "video"},
        "asset_bound": {
            "max_bytes": 1 << 30,
            "prepare": {
                "profile": "video-trim/1",
                "excess": "truncate",
                "min_seconds": 2,
                "max_seconds": 15,
            },
        },
    }
    assert "media_decode" in entry["invocable"]["capabilities"]
    package_interface.read_bytes(raw)

    module = ModuleType("generated_prepared_media")
    sys.modules[module.__name__] = module
    code = interface_wheel.generate(raw)["prepared_media/__init__.py"]
    assert (
        b"VideoPreparation(**{'excess': 'truncate', 'min_seconds': 2, 'max_seconds': 15})" in code
    )
    assert b"ImagePreparation(**{'max_edge': 1024})" in code
    exec(compile(code, "generated_prepared_media.py", "exec"), module.__dict__)
    binding = module.__dict__["__cozy_bindings__"]["render"]
    payload = next(f for f in msgspec.structs.fields(binding.request) if f.name == "payload")
    assert schema.render(payload.type) == entry["request"]

    def refused(prepare: object, kind: int = 1) -> None:
        document = msgspec.json.decode(raw)
        document["entrypoints"][0]["assets"]["kinds"][kind]["prepare"] = prepare
        with pytest.raises(ConformanceError):
            package_interface.read_bytes(msgspec.json.encode(document))

    refused({"profile": "video-trim/1", "max_seconds": 15, "excess": "clip"})
    refused({"profile": "video-trim/1", "max_seconds": 15, "excess": "refuse", "start": 1})
    refused({"profile": "video-trim/1", "min_seconds": 9, "max_seconds": 3, "excess": "refuse"})
    refused({"profile": "image-fit/1", "max_edge": 64})  # an image profile on videos
    document = msgspec.json.decode(raw)
    document["entrypoints"][0]["assets"]["kinds"][1]["prepare"] = {"profile": "video-fit/2"}
    package_interface.read_bytes(msgspec.json.encode(document))  # a newer writer's profile

    def field_refused(prepare: object) -> None:
        document = msgspec.json.decode(raw)
        fields = document["entrypoints"][0]["request"]["fields"]
        shot = next(f for f in fields if f["name"] == "shot")
        shot["type"]["fields"][0]["asset_bound"]["prepare"] = prepare
        with pytest.raises(ConformanceError):
            package_interface.read_bytes(msgspec.json.encode(document))

    field_refused({"profile": "image-fit/1", "max_edge": 64})  # an image profile on a video
    field_refused({"profile": "video-trim/1", "max_seconds": 15, "excess": "refuse", "start": 1})


class Misplaced(msgspec.Struct):
    image: Annotated[
        ImageAsset, AssetBound(max_decoded_bytes=1 << 20), VideoPreparation(max_seconds=1)
    ]


class Done(msgspec.Struct):
    ok: bool


def test_misplaced_preparations_refuse_at_describe() -> None:
    app = App()

    @app.entrypoint(accelerator=False)
    def misplaced(payload: Misplaced) -> Done:
        raise AssertionError("describe only")

    with pytest.raises(ConformanceError, match="preparation_target"):
        describe(app)
    for bad in (
        lambda: VideoPreparation(),
        lambda: VideoPreparation(min_seconds=3, max_seconds=2),
        lambda: VideoPreparation(min_seconds=1, excess="truncate"),
        lambda: AudioPreparation(max_seconds=0),
        lambda: AudioPreparation(max_seconds=1, excess="clip"),  # type: ignore[arg-type]
    ):
        with pytest.raises(ConformanceError):
            bad()


def test_executor_reply_rows_keep_kind_and_source() -> None:
    rows = [
        AdjustmentRow("adjusted", "video", 120.0, 15, "video truncated to 15 s", source="policy"),
        AdjustmentRow("warn", "seed", None, None, "seed ignored"),
    ]
    wire = [adjustment_row(row) for row in rows]
    assert wire[1] == {
        "field": "seed",
        "kind": "warn",
        "requested": "",
        "applied": "",
        "reason": "seed ignored",
        "source": "handler",
    }
    frame = {"ok": True, "adjustments": wire}
    attempted = executor_replies.decode(frame, executor_replies.AttemptReply)
    prepared = executor_replies.decode(frame, executor_replies.PreparedReply)
    for adjustments in (attempted.adjustments, prepared.adjustments):
        assert [(a.kind, a.source, a.requested) for a in adjustments] == [
            ("adjusted", "policy", "120.0"),
            ("warn", "handler", ""),
        ]
