"""How a media handle reads real files: clocks, origins, budgets, ceilings and refusals."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image, ImageOps

from cozy_runtime.author import (
    AudioAsset,
    DecodedAudioChunk,
    DecodedVideo,
    DecodedVideoFrame,
    ImageAsset,
    InvalidRequest,
    MediaDecodeError,
    VideoAsset,
    fakes,
)
from cozy_runtime.author._decode import (
    DEFAULT_DECODE_LIMITS,
    MediaReads,
    _Budget,
    decode_image_path,
)
from cozy_runtime.author._services import Attempt
from media_sources import joined, mp4, steady, webm

PIXELS = 64 * 48 * 3


def _h3_selection(video: DecodedVideo, fps: Fraction) -> list[int]:
    starts, origin = video.frame_pts, video.frame_pts[0]
    boundaries = [Fraction(value - origin) * video.time_base for value in starts]
    boundaries.append(Fraction(starts[-1] + video.frame_durations[-1] - origin) * video.time_base)
    slots = [math.floor(boundary * fps + Fraction(1, 2)) for boundary in boundaries]
    return [
        index for index in range(video.frame_count) for _ in range(slots[index + 1] - slots[index])
    ]


def _limited(name: str, **limits: int) -> Attempt:
    record = fakes.fake_attempt(name)
    record.media = MediaReads(record, limits=replace(DEFAULT_DECODE_LIMITS, **limits))
    return record


def test_variable_rate_b_frames_show_each_frame_until_the_next_starts(tmp_path: Path) -> None:
    starts = [Fraction(0)]
    for gap in [
        Fraction(1, 30),
        Fraction(1, 30),
        Fraction(1, 24),
        Fraction(1, 60),
        Fraction(1, 30),
    ] * 12:
        starts.append(starts[-1] + gap)
    path = mp4(
        tmp_path / "phone.mp4", size=(64, 48), starts=starts[:-1], end=starts[-1], b_frames=3
    )
    with av.open(str(path)) as container:
        decoded = list(container.decode(video=0))
    recorded = [(int(f.pts or 0), int(f.duration or 0)) for f in decoded]
    assert sum(b - a != d for (a, d), (b, _) in pairwise(recorded)) > 0

    video = fakes.file_input(VideoAsset, path)
    full = video.read(audio=False)
    assert full.frame_count == 60
    assert [(pts - full.frame_pts[0]) * full.time_base for pts in full.frame_pts] == starts[:-1]
    assert list(full.frame_durations[:-1]) == [b - a for a, b in pairwise(full.frame_pts)]
    mapped = video.read(fps=24, audio=False)
    assert list(mapped.frames_rgb) == [
        full.frames_rgb[i] for i in _h3_selection(full, Fraction(24))
    ]
    with video.stream(audio=False) as stream:
        frames = [event for event in stream if isinstance(event, DecodedVideoFrame)]
    assert [(f.pts, f.duration) for f in frames] == list(
        zip(full.frame_pts, full.frame_durations, strict=True)
    )
    assert [f.rgb for f in frames] == list(full.frames_rgb)


def test_the_frame_at_end_is_never_validated(tmp_path: Path) -> None:
    first = webm(tmp_path / "a.webm", size=(64, 48), frames=30)
    second = webm(tmp_path / "b.webm", size=(32, 24), frames=30, shade=100)
    video = fakes.file_input(VideoAsset, joined(tmp_path / "joined.webm", first, second))
    with pytest.raises(MediaDecodeError) as caught:
        video.read()  # the 31st frame changes size
    assert caught.value.code == "media_geometry"
    cut = video.read(end=1)
    assert cut.duration == 1 and (cut.frame_count, cut.width, cut.height) == (30, 64, 48)
    assert video.read(end=1, fps=24).frame_count == 24
    with video.stream(end=1) as stream:
        assert sum(isinstance(event, DecodedVideoFrame) for event in stream) == 30


def test_a_refused_read_returns_its_bytes_to_the_attempt(tmp_path: Path) -> None:
    starts, end = steady(30, 60)
    path = mp4(tmp_path / "clip.mp4", size=(64, 48), starts=starts, end=end)
    record = _limited("refund", max_decoded_bytes=50 * PIXELS)
    video = fakes.file_input(VideoAsset, path, attempt=record, max_decoded_bytes=50 * PIXELS)
    with pytest.raises(MediaDecodeError) as caught:
        video.read()  # 60 frames cross the attempt's 50
    assert caught.value.code == "decoded_bytes_limit"
    assert record.media.retained_bytes == 0
    assert video.read(end=1).frame_count == 30
    assert record.media.retained_bytes == 30 * PIXELS


def test_concurrent_reads_and_streams_share_one_exact_budget(tmp_path: Path) -> None:
    starts, end = steady(30, 60)
    path = mp4(tmp_path / "clip.mp4", size=(64, 48), starts=starts, end=end)
    record = fakes.fake_attempt("threads")
    video = fakes.file_input(VideoAsset, path, attempt=record)

    def read(_: int) -> int:
        return video.read(end=Fraction(1, 2), audio=False).frame_count

    def stream(_: int) -> int:
        with video.stream(end=Fraction(1, 2)) as events:
            return sum(isinstance(event, DecodedVideoFrame) for event in events)

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(read, range(8))) == [15] * 8
        assert list(pool.map(stream, range(6))) == [15] * 6
    assert record.media.retained_bytes == 8 * 15 * PIXELS
    assert record.media._open_streams == 0


def test_matroska_probe_reads_the_container_duration_without_a_packet_scan(
    tmp_path: Path,
) -> None:
    path = webm(tmp_path / "clip.webm", size=(64, 48), frames=90)
    with av.open(str(path)) as container:
        assert container.streams.video[0].duration is None and container.duration
        recorded = Fraction(int(container.duration), int(av.time_base))
    head = tmp_path / "head.webm"
    head.write_bytes(path.read_bytes()[: path.stat().st_size // 3])
    # Only the header survives the cut: a packet scan would measure about one second.
    assert fakes.file_input(VideoAsset, head).probe().source_duration == recorded == 3


@pytest.mark.parametrize(
    "video_start", [Fraction(0), Fraction(1, 2)], ids=["together", "late-video"]
)
def test_end_counts_from_the_first_frame_with_or_without_pictures(
    tmp_path: Path, video_start: Fraction
) -> None:
    starts, _ = steady(30, 90)
    path = mp4(
        tmp_path / "av.mp4",
        size=(64, 48),
        starts=[video_start + start for start in starts],
        end=video_start + 3,
        audio_seconds=4,
    )
    video = fakes.file_input(VideoAsset, path)
    end = Fraction(1)
    cut = video.read(end=end)
    assert cut.soundtrack is not None and cut.start_time == video_start
    with video.stream(end=end, pictures=False) as stream:
        alone = [e for e in stream if isinstance(e, DecodedAudioChunk)]
    with video.stream(end=end) as stream:
        beside = [e for e in stream if isinstance(e, DecodedAudioChunk)]
    kept = int((video_start + end - cut.soundtrack.start_time) * 16000)
    assert cut.soundtrack.sample_count == kept
    for chunks in (alone, beside):
        assert b"".join(c.pcm_f32le[0] for c in chunks) == cut.soundtrack.pcm_f32le[0]


def test_streams_keep_no_frame_or_sample_ceiling(tmp_path: Path) -> None:
    starts, end = steady(30, 60)
    path = mp4(tmp_path / "clip.mp4", size=(64, 48), starts=starts, end=end, audio_seconds=2)
    voice = fakes.fake_outputs(spool=tmp_path / "voice").save_audio(
        (0.1 * np.sin(np.arange(24000, dtype=np.float32) / 9)).astype(np.float32),
        sample_rate=24000,
    )
    record = _limited("ceilings", max_video_frames=30, max_audio_samples=16000)
    video = fakes.file_input(VideoAsset, path, attempt=record)
    audio = fakes.fake_input(voice, attempt=record)
    for read, code in (
        (lambda: video.read(audio=False), "video_frames_limit"),
        (audio.read, "audio_samples_limit"),
    ):
        with pytest.raises(MediaDecodeError) as caught:
            read()  # retained reads keep their ceilings
        assert caught.value.code == code
    with video.stream() as stream:
        events = list(stream)
    assert sum(isinstance(e, DecodedVideoFrame) for e in events) == 60
    assert sum(e.sample_count for e in events if isinstance(e, DecodedAudioChunk)) == 32000
    with audio.stream() as stream:
        assert sum(e.sample_count for e in stream if isinstance(e, DecodedAudioChunk)) == 24000
    assert isinstance(audio, AudioAsset)


def test_a_malformed_image_is_a_decode_refusal_from_probe_and_read(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4)
    image = fakes.file_input(ImageAsset, path)
    assert image.media_type == "image/png"
    for read in (image.probe, image.read):
        with pytest.raises(InvalidRequest) as caught:
            read()
        assert caught.value.code == "media_malformed" and caught.value.fields == ("input",)


def _as_0_26_5(path: Path) -> Image.Image:
    """0.26.5's `MediaDecoder.value`: the decoded RGB bytes rebuilt as a fresh PIL image."""
    decoded = decode_image_path(path, DEFAULT_DECODE_LIMITS, _Budget(1 << 30))
    return Image.frombytes("RGB", (decoded.width, decoded.height), decoded.rgb)


def test_an_image_read_carries_no_source_metadata(tmp_path: Path) -> None:
    path = tmp_path / "palette.png"
    indexes = (np.arange(120 * 120).reshape(120, 120) % 7).astype(np.uint8)
    source = Image.fromarray(indexes, mode="P")
    colours = [(i * 30, 255 - i * 30, (i * 70) % 256) for i in range(7)]
    source.putpalette([value for colour in colours for value in colour])
    source.save(path, transparency=3)
    with Image.open(path) as opened:
        naive = np.asarray(opened.convert("RGB").convert("RGBA"))[..., 3]
    assert (naive == 0).sum() > 0  # Pillow carries `transparency` through convert("RGB")

    image = fakes.file_input(ImageAsset, path).read()
    expected = _as_0_26_5(path)
    assert (image.mode, image.size, image.info) == (expected.mode, expected.size, expected.info)
    assert image.info == {} and image.tobytes() == expected.tobytes()
    assert (np.asarray(image.convert("RGBA"))[..., 3] == 255).all()


def test_an_exif_rotated_jpeg_reads_upright(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6  # stored 120x80, displayed a quarter turn
    pixels = np.random.default_rng(3).integers(0, 256, (80, 120, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="JPEG", exif=exif, quality=95)
    image = fakes.file_input(ImageAsset, path).read()
    with Image.open(path) as opened:
        upright = ImageOps.exif_transpose(opened).convert("RGB")
    assert image.size == (80, 120) and image.info == {}
    assert image.tobytes() == upright.tobytes() == _as_0_26_5(path).tobytes()
