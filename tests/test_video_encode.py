"""The mp4 save path, driven end to end through the real `Outputs` (h3a-017).

`save_video` registers a zero-copy view of the author's host array and the spool write is
the one host copy; x264 runs with the host-derived, ceiling-bounded thread count. The
container is deterministic per (input, wheel, thread count) and is NOT an output
identity — the pre-encode pixels are. Every arm here goes through `fakes.fake_outputs`,
which settles at the save: the same `snapshot_video` -> `_register_frame` ->
`encode_frame` -> `encode_mp4` -> `probe_mp4` chain the worker's post thread runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from cozy_runtime.author import _codec, fakes
from cozy_runtime.author._services import MAX_OUTPUT_BYTES

pytest.importorskip("av", reason="the media extra is absent from this venv")


def _clip(frames: int = 12, height: int = 96, width: int = 128) -> np.ndarray:
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    clip = np.empty((frames, height, width, 3), np.uint8)
    texture = rng.integers(0, 16, (height, width, 3), dtype=np.int16)
    for t in range(frames):
        base = np.stack(
            [
                128 + 100 * np.sin(xx / 17 + t / 3),
                128 + 100 * np.cos(yy / 11 - t / 3),
                128 + 100 * np.sin((xx + yy) / 23 + t / 2),
            ],
            -1,
        ).astype(np.int16)
        clip[t] = np.clip(base + texture - 8, 0, 255)
    return clip


def _blob(asset: object) -> Path:
    local = getattr(asset, "_local", None)
    assert isinstance(local, Path), asset
    return local


def _decoded_planes(path: Path) -> list[bytes]:
    import av

    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        return [f.to_ndarray(format="yuv420p").tobytes() for f in container.decode(video=0)]


def test_thread_count_is_the_host_cpu_count_under_the_ceiling() -> None:
    threads = _codec.x264_threads()
    assert 1 <= threads <= _codec.X264_THREAD_CEILING
    assert threads == min(_codec.X264_THREAD_CEILING, os.cpu_count() or 1)


def test_the_snapshot_is_a_view_and_the_spool_write_is_the_one_copy() -> None:
    clip = _clip()
    planar = _codec.snapshot_video(clip, MAX_OUTPUT_BYTES)
    assert len(planar) == clip.nbytes
    # Zero-copy: a write to the array is visible through the snapshot view.
    clip[0, 0, 0, 0] ^= 0xFF
    assert planar.view()[0] == clip[0, 0, 0, 0]
    clip[0, 0, 0, 0] ^= 0xFF

    outs = fakes.fake_outputs()
    asset = outs.save_video(clip, fps=24.0)
    before = _blob(asset).read_bytes()
    # Source-independent once the save returned: mutating the array changes nothing.
    clip[:] = 0
    assert _blob(asset).read_bytes() == before
    assert asset.size_bytes == len(before) and asset.digest


def test_the_registered_raw_bytes_are_the_source_pixels_then_audio() -> None:
    clip = _clip()
    waveform = (0.5 * np.sin(np.arange(8000, dtype=np.float32) / 9)).astype(np.float32)
    spool = Path(fakes.fake_attempt().spool)
    attempt = fakes.fake_attempt("raw-arm", spool=spool)
    attempt.settle_at_save = False  # keep the raw so the arm can read it
    outs = fakes.fake_outputs(attempt)
    outs.save_video(clip, fps=24.0, audio=waveform, sample_rate=8000)
    (frame,) = attempt.frames.values()
    raw = frame.raw.read_bytes()
    assert frame.raw_bytes == len(raw) == clip.nbytes + waveform.nbytes
    assert raw[: clip.nbytes] == clip.tobytes()
    assert raw[clip.nbytes :] == waveform.tobytes()
    assert frame.facts["video_bytes"] == clip.nbytes


def test_the_container_is_deterministic_and_probes_to_its_facts() -> None:
    clip = _clip()
    waveform = (0.25 * np.sin(np.arange(8000, dtype=np.float32) / 7)).astype(np.float32)
    first = fakes.fake_outputs().save_video(clip, fps=24.0, audio=waveform, sample_rate=8000)
    second = fakes.fake_outputs().save_video(clip, fps=24.0, audio=waveform, sample_rate=8000)
    assert _blob(first).read_bytes() == _blob(second).read_bytes()
    facts = _codec.probe_mp4(_blob(first))
    assert (facts.width, facts.height, facts.frame_count) == (128, 96, 12)
    assert facts.frame_rate == 24 and facts.video_codec == "h264" and facts.audio_codec == "aac"
    assert len(_decoded_planes(_blob(first))) == 12


def test_a_torch_free_ndarray_that_is_not_contiguous_still_snapshots_exactly() -> None:
    clip = _clip()
    flipped = clip[:, ::-1]  # a view with a negative stride: copied, never mis-read
    planar = _codec.snapshot_video(flipped, MAX_OUTPUT_BYTES)
    assert planar.view().tobytes() == np.ascontiguousarray(flipped).tobytes()
