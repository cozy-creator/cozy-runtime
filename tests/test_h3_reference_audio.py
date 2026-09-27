"""H3 reference audio reaches the official pipeline at the audio VAE rate, without torchaudio.

The official setup step resamples any other rate with torchaudio, which has no release for
the Torch H3 serves: run 1182 failed with that ImportError for a 24 kHz voice reference.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.modular_pipelines.minimax_h3.before_encoder import (  # noqa: E402
    MiniMaxH3Ref2VASetupStep,
)

from cozy_runtime.author._decode import DecodedAudio, DecodedVideo  # noqa: E402
from cozy_runtime.models.minimax_h3.official import OfficialH3Pipeline  # noqa: E402

VAE_RATE = 32000
# The only pipeline state the reference conversions read.
PIPE = SimpleNamespace(sample_rate=VAE_RATE)
RATES = [16000, 22050, 24000, 32000, 44100, 48000]


def _decoded(rate: int, seconds: int, channels: int = 1, hz: float = 440.0) -> DecodedAudio:
    samples = rate * seconds
    wave = (0.25 * np.sin(2 * np.pi * hz * np.arange(samples) / rate)).astype("<f4")
    return DecodedAudio(
        channels=channels,
        sample_count=samples,
        sample_rate=rate,
        channel_layout="mono" if channels == 1 else "stereo",
        channel_names=("FC",) if channels == 1 else ("FL", "FR"),
        pcm_f32le=tuple(wave.tobytes() for _ in range(channels)),
        start_time=Fraction(0),
    )


def _normalized(audio: object, rate: int, seconds: float) -> object:
    # Diffusers' own normalization, exactly as the ref2va setup step calls it.
    return MiniMaxH3Ref2VASetupStep._normalize_audio_condition(
        audio, rate, VAE_RATE, max_duration=seconds
    )


def _peak_hz(waveform: object) -> float:
    steady = waveform[0, VAE_RATE // 4 : -VAE_RATE // 4].double()  # type: ignore[index]
    spectrum = torch.fft.rfft(steady * torch.hann_window(len(steady), dtype=torch.float64))
    return float(spectrum.abs().argmax()) * VAE_RATE / len(steady)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("channels", [1, 2])
def test_audio_reference_needs_no_torchaudio(
    monkeypatch: pytest.MonkeyPatch, rate: int, channels: int
) -> None:
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    audio = _decoded(rate, 3, channels)
    reference = OfficialH3Pipeline.audio_reference(PIPE, audio)  # type: ignore[arg-type]
    assert reference.sample_rate == VAE_RATE
    assert tuple(reference.audio.shape) == (channels, 3 * VAE_RATE)
    waveform = _normalized(reference.audio, reference.sample_rate, 5.0)
    assert tuple(waveform.shape) == (2, 3 * VAE_RATE)  # type: ignore[attr-defined]
    assert abs(_peak_hz(waveform) - 440.0) < 1.0
    rms = float(waveform[0, 8000:-8000].pow(2).mean().sqrt())  # type: ignore[index]
    assert abs(rms - 0.25 / math.sqrt(2)) < 0.002


@pytest.mark.parametrize("rate", [24000, 44100, 48000])
def test_video_soundtrack_needs_no_torchaudio(monkeypatch: pytest.MonkeyPatch, rate: int) -> None:
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    frames = 48
    video = DecodedVideo(
        width=16,
        height=16,
        frame_count=frames,
        frames_rgb=tuple(bytes(16 * 16 * 3) for _ in range(frames)),
        frame_pts=tuple(range(frames)),
        frame_durations=(1,) * frames,
        time_base=Fraction(1, 24),
        pixel_aspect_ratio=Fraction(1),
        soundtrack=_decoded(rate, 2),
    )
    reference = OfficialH3Pipeline.video_reference(PIPE, video)  # type: ignore[arg-type]
    assert reference.sample_rate == VAE_RATE
    assert tuple(reference.audio.shape) == (1, 2 * VAE_RATE)
    waveform = _normalized(reference.audio, reference.sample_rate, 5.0)
    assert abs(_peak_hz(waveform) - 440.0) < 1.0
