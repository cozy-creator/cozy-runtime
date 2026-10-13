"""Streamed media failures name their true origin: early exits, author errors, bad bytes."""

from __future__ import annotations

import traceback
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cozy_runtime.author import (
    DecodedAudioFormat,
    DecodedMediaEvent,
    DecodedMediaHeader,
    DecodedVideoFormat,
    DecodedVideoFrame,
    MediaDecodeError,
    OutputError,
    VideoAsset,
    classify,
    fakes,
)
from cozy_runtime.author._services import Attempt

pytest.importorskip("av", reason="the media extra is absent from this venv")

FACTS = dict(
    time_base=Fraction(1, 24),
    pixel_aspect_ratio=Fraction(1),
    color_primaries=1,
    color_transfer=1,
    color_matrix=1,
    color_range=1,
)


def _source(tmp_path: Path) -> tuple[Attempt, VideoAsset]:
    attempt = fakes.fake_attempt("source", spool=tmp_path / "spool")
    pixels = np.random.default_rng(5).integers(0, 256, (24, 32, 48, 3), dtype=np.uint8)
    return attempt, fakes.fake_outputs(attempt).save_video(pixels, fps=24)


def test_leaving_a_stream_after_its_first_frame_saves_that_frame(tmp_path: Path) -> None:
    attempt, source = _source(tmp_path)
    out = fakes.fake_outputs(attempt)
    video = fakes.fake_input(source, attempt=attempt.request_id, max_decoded_bytes=8 << 20)
    decoder = fakes.fake_media_decoder(attempt)

    def first_frame(events: Iterator[DecodedMediaEvent]) -> Iterator[DecodedMediaEvent]:
        for event in events:
            yield event
            if isinstance(event, DecodedVideoFrame):
                break

    with decoder.stream_video(video) as stream:
        saved = out.save_video_stream(first_frame(stream))
    assert (saved.frame_count, saved.frame_rate, saved.video_codec) == (1, 24, "h264")
    reinput = fakes.fake_input(saved.video, attempt=attempt.request_id, max_decoded_bytes=8 << 20)
    decoded = decoder.decode_video(reinput)
    assert (decoded.width, decoded.height, decoded.frame_count) == (48, 32, 1)
    assert decoded.duration == Fraction(1, 24)


@pytest.mark.parametrize(
    "error",
    [ValueError("model diverged"), RuntimeError("CUDA error: out of memory")],
    ids=["author-error", "device-oom"],
)
def test_an_author_generator_error_surfaces_unchanged(tmp_path: Path, error: Exception) -> None:
    attempt = fakes.fake_attempt("author", spool=tmp_path)

    def events() -> Iterator[DecodedMediaEvent]:
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(width=64, height=64, nominal_frame_rate=Fraction(24), **FACTS),
            audio=None,
        )
        for index in range(48):
            if index == 30:
                raise error
            yield DecodedVideoFrame(
                width=64,
                height=64,
                rgb=bytes([index]) * 64 * 64 * 3,
                pts=index,
                duration=1,
                **FACTS,
            )

    with pytest.raises(type(error)) as caught:
        fakes.fake_outputs(attempt).save_video_stream(events())
    assert caught.value is error
    assert traceback.extract_tb(caught.value.__traceback__)[-1].name == "events"
    assert classify(caught.value).code == (
        "unhandled_exception" if isinstance(error, ValueError) else "device_out_of_memory"
    )
    assert not attempt.pending
    assert not list(tmp_path.glob(".video-*")) and not list(tmp_path.glob("video-*"))


def test_an_encoder_failure_is_output_encode(tmp_path: Path) -> None:
    rate = 12345  # AAC has no such sampling frequency; FFmpeg's encoder refuses to open.
    events = [
        DecodedMediaHeader(
            video=DecodedVideoFormat(width=32, height=32, nominal_frame_rate=Fraction(24), **FACTS),
            audio=DecodedAudioFormat(
                channels=1,
                sample_rate=rate,
                channel_layout="mono",
                channel_names=("FC",),
                time_base=Fraction(1, rate),
            ),
        )
    ]
    with pytest.raises(OutputError) as caught:
        fakes.fake_outputs(spool=tmp_path).save_video_stream(iter(events))
    assert caught.value.code == "output_encode"


@pytest.mark.parametrize("damage", ["truncated", "noisy-samples"])
def test_corrupt_video_bytes_are_media_malformed(tmp_path: Path, damage: str) -> None:
    attempt, source = _source(tmp_path)
    assert source._local is not None
    data = bytearray(source._local.read_bytes())
    if damage == "truncated":
        del data[len(data) // 2 :]
    else:
        start = data.find(b"mdat") + 64
        data[start : start + 2000] = np.random.default_rng(3).bytes(2000)
    path = tmp_path / f"{damage}.mp4"
    path.write_bytes(data)
    video = fakes.fake_input(
        VideoAsset(str(path), local=path, size_bytes=len(data)),
        attempt=attempt.request_id,
        max_decoded_bytes=8 << 20,
    )
    decoder = fakes.fake_media_decoder(attempt)
    with pytest.raises(MediaDecodeError) as caught, decoder.stream_video(video) as stream:
        list(stream)
    assert caught.value.code == "media_malformed" and caught.value.fields == ("input",)
