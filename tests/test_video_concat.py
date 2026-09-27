"""Joining encoded videos copies their H.264 packets and encodes only the soundtrack."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cozy_runtime.author import (
    DecodedAudioChunk,
    DecodedAudioFormat,
    DecodedMediaHeader,
    DecodedVideoFormat,
    DecodedVideoFrame,
    OutputError,
    VideoAsset,
    fakes,
)
from cozy_runtime.author._decode import DecodedMediaEvent

av = pytest.importorskip("av", reason="the media extra is absent from this venv")

RATE = 32000
AUDIO = DecodedAudioFormat(
    channels=2,
    sample_rate=RATE,
    channel_layout="stereo",
    channel_names=("FL", "FR"),
    time_base=Fraction(1, RATE),
)


def _segment(tmp_path: Path, name: str, frames: int, width: int = 64) -> VideoAsset:
    out = fakes.fake_outputs(fakes.fake_attempt(name, spool=tmp_path / name))
    tone = (0.1 * np.sin(np.arange(frames * RATE // 24) * 0.05)).astype("<f4")

    def events() -> Iterator[DecodedMediaEvent]:
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(
                width=width,
                height=48,
                time_base=Fraction(1, 24),
                pixel_aspect_ratio=Fraction(1),
                nominal_frame_rate=Fraction(24),
                color_primaries=1,
                color_transfer=1,
                color_matrix=1,
                color_range=1,
            ),
            audio=AUDIO,
        )
        per = RATE // 24
        for index in range(frames):
            rgb = np.full((48, width, 3), (index * 7 + len(name) * 40) % 256, np.uint8)
            rgb[:, index % width] = 255
            yield DecodedVideoFrame(
                width=width,
                height=48,
                rgb=rgb.tobytes(),
                pts=index,
                duration=1,
                time_base=Fraction(1, 24),
                pixel_aspect_ratio=Fraction(1),
                color_primaries=1,
                color_transfer=1,
                color_matrix=1,
                color_range=1,
            )
            pcm = tone[index * per : (index + 1) * per].tobytes()
            yield DecodedAudioChunk(
                channels=2,
                sample_count=per,
                sample_rate=RATE,
                channel_layout="stereo",
                channel_names=("FL", "FR"),
                pcm_f32le=(pcm, pcm),
                pts=index * per,
                time_base=Fraction(1, RATE),
            )

    return out.save_video_stream(events()).video


def _packets(path: Path) -> list[bytes]:
    with av.open(str(path)) as container:
        return [bytes(p) for p in container.demux(container.streams.video[0]) if p.size]


def test_join_copies_packets_under_an_exact_soundtrack(tmp_path: Path) -> None:
    sources = [_segment(tmp_path, name, frames) for name, frames in (("a", 30), ("bb", 45))]
    parent = fakes.fake_attempt("parent", spool=tmp_path / "parent")
    videos = [
        fakes.fake_input(v, attempt="parent", input_id=f"v{i}", max_decoded_bytes=8 << 20)
        for i, v in enumerate(sources)
    ]
    decoder = fakes.fake_media_decoder(parent)
    counted: list[Sequence[int]] = []

    def soundtrack(frames: Sequence[int]) -> Iterator[DecodedMediaEvent]:
        counted.append(frames)
        yield DecodedMediaHeader(video=None, audio=AUDIO)
        written = 0
        for video in videos:  # each video's soundtrack, read without decoding a picture
            with decoder.stream_audio(video) as stream:
                for event in stream:
                    if isinstance(event, DecodedAudioChunk):
                        yield DecodedAudioChunk(
                            channels=2,
                            sample_count=event.sample_count,
                            sample_rate=RATE,
                            channel_layout="stereo",
                            channel_names=("FL", "FR"),
                            pcm_f32le=event.pcm_f32le,
                            pts=written,
                            time_base=Fraction(1, RATE),
                        )
                        written += event.sample_count

    saved = fakes.fake_outputs(parent).save_video_concat(videos, soundtrack)
    assert counted == [(30, 45)]
    assert saved.frame_count == 75 and saved.frame_rate == 24
    assert saved.audio is not None and saved.audio.decoded_samples == 75 * (RATE // 24)
    assert saved.submitted_audio_samples == 75 * (RATE // 24)
    assert saved.video._local is not None and all(v._local for v in sources)
    joined = _packets(saved.video._local)
    assert joined == [p for v in sources for p in _packets(v._local)]  # type: ignore[arg-type]
    reader = fakes.fake_attempt("reader", spool=tmp_path / "reader")
    decoded = fakes.fake_media_decoder(reader).decode_video(
        fakes.fake_input(saved.video, attempt="reader", max_decoded_bytes=8 << 20)
    )
    assert decoded.frame_count == 75
    assert all(v * decoded.time_base == Fraction(i, 24) for i, v in enumerate(decoded.frame_pts))


def test_different_tracks_refuse_before_the_soundtrack_is_read(tmp_path: Path) -> None:
    sources = [_segment(tmp_path, "a", 24), _segment(tmp_path, "wide", 24, width=96)]
    parent = fakes.fake_attempt("parent", spool=tmp_path / "parent")
    videos = [
        fakes.fake_input(v, attempt="parent", input_id=f"v{i}", max_decoded_bytes=8 << 20)
        for i, v in enumerate(sources)
    ]

    def soundtrack(frames: Sequence[int]) -> Iterator[DecodedMediaEvent]:
        raise AssertionError("an incompatible join consumed its soundtrack")

    with pytest.raises(OutputError) as refused:
        fakes.fake_outputs(parent).save_video_concat(videos, soundtrack)
    assert refused.value.code == "video_copy_incompatible"
    assert not list(parent.spool.glob("*video*"))
