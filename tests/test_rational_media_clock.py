"""Real codec clocks normalize exactly across supported PyAV interfaces."""

from __future__ import annotations

import hashlib
import time
from fractions import Fraction
from pathlib import Path
from typing import Annotated

import msgspec
import numpy as np
import pytest

from cozy_runtime.author import App, AssetBound, Invocation, MediaDecoder, VideoAsset, attempt
from cozy_runtime.author._assets import GrantedInput, file_state
from cozy_runtime.author._decode import MediaDecodeError, _positive_fraction

av = pytest.importorskip("av")


class Request(msgspec.Struct):
    clip: Annotated[VideoAsset, AssetBound(max_bytes=1 << 20, max_decoded_bytes=1 << 20)]


class Summary(msgspec.Struct):
    frames: int
    video_clock: tuple[int, int]
    audio_rate: int
    audio_samples: int


app = App()


@app.entrypoint
def inspect_clip(payload: Request, decoder: MediaDecoder) -> Summary:
    decoded = decoder.decode_video(payload.clip)
    assert isinstance(decoded.time_base, Fraction)
    assert decoded.soundtrack is not None
    assert isinstance(decoded.soundtrack.start_time, Fraction)
    return Summary(
        decoded.frame_count,
        (decoded.time_base.numerator, decoded.time_base.denominator),
        decoded.soundtrack.sample_rate,
        decoded.soundtrack.sample_count,
    )


@pytest.mark.parametrize("declared_aspect", [False, True])
def test_actual_encoded_video_and_audio_clock_reaches_author_decoder(
    tmp_path: Path, declared_aspect: bool
) -> None:
    source = tmp_path / "clock.mp4"
    with av.open(str(source), "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=24)
        video.width = video.height = 16
        video.pix_fmt = "yuv420p"
        video.codec_context.thread_count = 1
        video.codec_context.color_primaries = 1
        video.codec_context.color_trc = 1
        video.codec_context.colorspace = 1
        video.codec_context.color_range = 1
        if declared_aspect:
            video.codec_context.sample_aspect_ratio = Fraction(1, 1)
        audio = container.add_stream("aac", rate=32000)
        audio.codec_context.thread_count = 1
        audio.codec_context.layout = "mono"
        for index in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), index * 40, dtype=np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        wave = av.AudioFrame.from_ndarray(
            np.zeros((1, 4000), np.float32), format="fltp", layout="mono"
        )
        wave.sample_rate, wave.pts, wave.time_base = 32000, 0, Fraction(1, 32000)
        for packet in audio.encode(wave):
            container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode(None):
                container.mux(packet)
    with av.open(str(source)) as container:
        clock = container.streams.audio[0].time_base
        assert _positive_fraction(clock, "actual audio clock") == Fraction(1, 32000)
    raw = source.read_bytes()
    result, outcome, _ = attempt(
        app.get("inspect_clip"),
        {"clip": "verified-clock"},
        Invocation(
            "clock-consumer",
            tmp_path / "result",
            time.monotonic() + 30,
            assets={
                "clip": GrantedInput(
                    input_id="clip",
                    local=source,
                    media_type="video/mp4",
                    digest="sha256:" + hashlib.sha256(raw).hexdigest(),
                    length=len(raw),
                    file_state=file_state(source),
                )
            },
        ),
    )
    assert outcome.terminal == "succeeded", outcome
    assert result is not None
    assert result.result.frames == 3
    assert result.result.audio_rate == 32000 and 0 < result.result.audio_samples <= 5000


@pytest.mark.parametrize("value", [0, -1, Fraction(-1, 32000), "bad-clock", object(), float("nan")])
def test_malformed_or_nonpositive_clock_still_refuses(value: object) -> None:
    with pytest.raises(MediaDecodeError) as failure:
        _positive_fraction(value, "clock")
    assert failure.value.code == "media_clock"


@pytest.mark.parametrize("numerator,denominator", [(0, 1), (-1, 32000), (1, 0), (-1, 0), (0, 0)])
def test_native_undefined_or_nonpositive_rational_still_refuses(
    numerator: int, denominator: int
) -> None:
    rational = getattr(av, "AVRational", None)
    if rational is None:
        pytest.skip("this provider exposes finite Fraction clocks only")
    with pytest.raises(MediaDecodeError) as failure:
        _positive_fraction(rational(numerator, denominator), "native clock")
    assert failure.value.code == "media_clock"
