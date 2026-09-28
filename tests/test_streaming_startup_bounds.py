"""Real muxer startup is bounded before a delayed initial track can emit a header."""

from __future__ import annotations

import threading
from collections.abc import Iterator
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
    fakes,
)
from cozy_runtime.author._decode import DecodedMediaEvent

pytest.importorskip("av", reason="the media extra is absent from this venv")


@pytest.mark.parametrize("case", ["output-cap", "video-packets", "video-bytes", "audio-packets"])
def test_missing_initial_track_refuses_before_unbounded_buffering(
    tmp_path: Path, case: str
) -> None:
    cap = 4096 if case == "output-cap" else 64 << 20
    dimension = 512 if case == "video-bytes" else 32
    count = 200 if case == "video-bytes" else 2000
    attempt = fakes.fake_attempt("bounded-startup", spool=tmp_path, max_output_bytes=cap)
    out = fakes.fake_outputs(attempt)
    emitted = 0
    random = np.random.default_rng(11)

    def video(index: int) -> DecodedVideoFrame:
        rgb = (
            random.integers(0, 256, (dimension, dimension, 3), dtype=np.uint8).tobytes()
            if case == "video-bytes"
            else bytes([index % 255]) * (dimension * dimension * 3)
        )
        return DecodedVideoFrame(
            width=dimension,
            height=dimension,
            rgb=rgb,
            pts=index,
            duration=1,
            time_base=Fraction(1, 24),
            pixel_aspect_ratio=Fraction(1),
            color_primaries=1,
            color_transfer=1,
            color_matrix=1,
            color_range=1,
        )

    def audio(index: int) -> DecodedAudioChunk:
        return DecodedAudioChunk(
            channels=1,
            sample_count=2048,
            sample_rate=32000,
            channel_layout="mono",
            channel_names=("FC",),
            pcm_f32le=(bytes(2048 * 4),),
            pts=index * 2048,
            time_base=Fraction(1, 32000),
        )

    def events() -> Iterator[DecodedMediaEvent]:
        nonlocal emitted
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(
                width=dimension,
                height=dimension,
                time_base=Fraction(1, 24),
                pixel_aspect_ratio=Fraction(1),
                nominal_frame_rate=Fraction(24),
                color_primaries=1,
                color_transfer=1,
                color_matrix=1,
                color_range=1,
            ),
            audio=DecodedAudioFormat(
                channels=1,
                sample_rate=32000,
                channel_layout="mono",
                channel_names=("FC",),
                time_base=Fraction(1, 32000),
            ),
        )
        for index in range(count):
            emitted += 1
            yield audio(index) if case == "audio-packets" else video(index)
        yield video(0) if case == "audio-packets" else audio(0)

    with pytest.raises(OutputError) as caught:
        out.save_video_stream(events())
    assert caught.value.code == ("output_too_large" if case == "output-cap" else "media_interleave")
    assert emitted < count
    assert not attempt.pending
    assert not list(tmp_path.glob(".video-*")) and not list(tmp_path.glob("video-*"))


@pytest.mark.parametrize("fail_at", [0, 30])
def test_a_stream_failing_under_the_encode_thread_leaves_no_file_or_thread(
    tmp_path: Path, fail_at: int
) -> None:
    """The source fails while x264 frames are still queued on the encode thread (H3's
    decode abandoning its stream): the author's error surfaces, the writer joins its
    thread and unlinks the partial file."""
    attempt = fakes.fake_attempt("abandoned", spool=tmp_path)
    out = fakes.fake_outputs(attempt)
    random = np.random.default_rng(3)

    def events() -> Iterator[DecodedMediaEvent]:
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(
                width=256,
                height=144,
                time_base=Fraction(1, 24),
                pixel_aspect_ratio=Fraction(1),
                nominal_frame_rate=Fraction(24),
                color_primaries=1,
                color_transfer=1,
                color_matrix=1,
                color_range=1,
            ),
            audio=None,
        )
        for index in range(48):
            if index == fail_at:
                raise OutputError("the decode abandoned the video stream", code="output_integrity")
            yield DecodedVideoFrame(
                width=256,
                height=144,
                rgb=random.integers(0, 256, (144, 256, 3), dtype=np.uint8).tobytes(),
                pts=index,
                duration=1,
                time_base=Fraction(1, 24),
                pixel_aspect_ratio=Fraction(1),
                color_primaries=1,
                color_transfer=1,
                color_matrix=1,
                color_range=1,
            )

    with pytest.raises(OutputError) as caught:
        out.save_video_stream(events())
    assert caught.value.code == "output_integrity"
    assert not attempt.pending
    assert not list(tmp_path.glob(".video-*")) and not list(tmp_path.glob("video-*"))
    assert not [t for t in threading.enumerate() if t.name == "mp4-encode"]
