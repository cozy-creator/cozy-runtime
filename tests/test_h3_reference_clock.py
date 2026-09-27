"""H3 video references play like a player: gaps hold a frame, soundtracks snap to a sample."""

from __future__ import annotations

from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from cozy_runtime.author import ConformanceError  # noqa: E402
from cozy_runtime.author._decode import DecodedAudio, DecodedVideo  # noqa: E402
from cozy_runtime.models.minimax_h3.official import (  # noqa: E402
    _aligned_soundtrack,
    _video_at_24fps,
)


def _video(pts: tuple[int, ...], durations: tuple[int, ...], **extra: object) -> DecodedVideo:
    return DecodedVideo(
        width=2,
        height=2,
        frame_count=len(pts),
        frames_rgb=tuple(bytes([index]) * 12 for index in range(len(pts))),
        frame_pts=pts,
        frame_durations=durations,
        time_base=Fraction(1, 24),
        pixel_aspect_ratio=Fraction(1),
        soundtrack=extra.get("soundtrack"),  # type: ignore[arg-type]
    )


def _shown(video: DecodedVideo) -> list[int]:
    return [int(frame[0, 0, 0]) for frame in _video_at_24fps(video)]


def test_a_timestamp_gap_holds_the_previous_frame() -> None:
    assert _shown(_video((0, 1, 4), (1, 1, 1))) == [0, 1, 1, 1, 2]


def test_an_overlap_ends_the_earlier_frame_when_the_next_starts() -> None:
    assert _shown(_video((0, 2, 3), (3, 1, 1))) == [0, 0, 1, 2]


def test_presentation_time_running_backwards_still_refuses() -> None:
    with pytest.raises(ConformanceError, match="runs backwards"):
        _video_at_24fps(_video((0, 2, 1), (1, 1, 1)))


def test_a_soundtrack_off_the_sample_grid_snaps_to_the_nearest_sample() -> None:
    rate = 48000
    samples = rate
    pcm = torch.arange(samples, dtype=torch.float32).numpy().astype("<f4").tobytes()
    audio = DecodedAudio(
        channels=1,
        sample_count=samples,
        sample_rate=rate,
        channel_layout="mono",
        channel_names=("FC",),
        pcm_f32le=(pcm,),
        start_time=Fraction(1, 3 * rate),  # a third of a sample after the video starts
    )
    video = _video(tuple(range(24)), (1,) * 24, soundtrack=audio)
    aligned = _aligned_soundtrack(video)
    assert aligned is not None
    assert tuple(aligned.shape) == (1, rate)
    assert float(aligned[0, 0]) == 0.0 and float(aligned[0, -1]) == rate - 1
