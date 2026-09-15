"""Real streaming output and decoding preserve authored PCM and video clocks."""

from __future__ import annotations

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
    fakes,
)
from cozy_runtime.author._decode import DecodedMediaEvent

pytest.importorskip("av", reason="the media extra is absent from this venv")


@pytest.mark.parametrize("rate", [32000, 48000])
@pytest.mark.parametrize(
    "frames,duration",
    [(120, "5"), (360, "15"), (124, "5.175"), (362, "15.075")],
    ids=["5s", "15s", "h3-5s", "h3-15s"],
)
@pytest.mark.parametrize("late_audio", [False, True], ids=["interleaved", "audio-last"])
def test_streaming_media_preserves_pcm_clock(
    tmp_path: Path,
    rate: int,
    frames: int,
    duration: str,
    late_audio: bool,
) -> None:
    samples = int(Fraction(duration) * rate)
    waveform = np.random.default_rng(7).uniform(-0.125, 0.125, samples).astype("<f4")
    attempt = fakes.fake_attempt("encode", spool=tmp_path / "encode")
    output = fakes.fake_outputs(attempt)

    def audio(start: int, stop: int) -> DecodedAudioChunk:
        return DecodedAudioChunk(
            channels=1,
            sample_count=stop - start,
            sample_rate=rate,
            channel_layout="mono",
            channel_names=("FC",),
            pcm_f32le=(waveform[start:stop].tobytes(),),
            pts=start,
            time_base=Fraction(1, rate),
        )

    def events() -> Iterator[DecodedMediaEvent]:
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(
                width=32,
                height=32,
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
                sample_rate=rate,
                channel_layout="mono",
                channel_names=("FC",),
                time_base=Fraction(1, rate),
            ),
        )
        submitted = 0
        for index in range(frames):
            yield DecodedVideoFrame(
                width=32,
                height=32,
                rgb=bytes([index % 255]) * (32 * 32 * 3),
                pts=index,
                duration=1,
                time_base=Fraction(1, 24),
                pixel_aspect_ratio=Fraction(1),
                color_primaries=1,
                color_transfer=1,
                color_matrix=1,
                color_range=1,
            )
            if not late_audio:
                end = min(samples, (index + 1) * rate // 24)
                if end > submitted:
                    yield audio(submitted, end)
                    submitted = end
                if index == 72:
                    # Header delay must still publish fragments before the stream ends.
                    assert next(attempt.spool.glob(".video-*.partial")).stat().st_size > 0
        if submitted < samples:
            yield audio(submitted, samples)

    saved = output.save_video_stream(events())
    assert saved.frame_count == frames and saved.frame_rate == 24
    assert saved.submitted_audio_samples == samples
    assert saved.audio is not None
    assert saved.audio.decoded_samples == samples and saved.audio.sample_rate == rate
    assert saved.audio.codec_frame_samples == 1024

    reader = fakes.fake_attempt("decode", spool=tmp_path / "decode")
    asset = fakes.fake_input(saved.video, attempt=reader.request_id, max_decoded_bytes=8 << 20)
    decoded = fakes.fake_media_decoder(reader).decode_video(asset)
    assert decoded.frame_count == frames and decoded.start_time == 0
    assert decoded.duration == Fraction(frames, 24)
    assert all(
        value * decoded.time_base == Fraction(i, 24) for i, value in enumerate(decoded.frame_pts)
    )
    sound = decoded.soundtrack
    assert sound is not None and sound.start_time == 0
    assert sound.sample_count == samples and sound.sample_rate == rate
    observed = np.frombuffer(sound.pcm_f32le[0], dtype="<f4")
    # Cutting the last codec frame would fix a count while leaving leading priming.
    # Broadband sample correlation also proves the submitted content starts on time.
    exact = float(np.corrcoef(waveform[2048:-2048], observed[2048:-2048])[0, 1])
    shifted = float(np.corrcoef(waveform[2048:-3072], observed[3072:-2048])[0, 1])
    assert exact > 0.75 and abs(shifted) < 0.1
