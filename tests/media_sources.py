"""Real media encoded at test time with PyAV, for the media-handle tests."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

TICK = Fraction(1, 90000)


def steady(fps: int, frames: int) -> tuple[list[Fraction], Fraction]:
    return [Fraction(index, fps) for index in range(frames)], Fraction(frames, fps)


def mp4(
    path: Path,
    *,
    size: tuple[int, int],
    starts: Sequence[Fraction],
    end: Fraction,
    audio_seconds: int = 0,
    sample_rate: int = 16000,
    b_frames: int = 0,
) -> Path:
    """H.264 frames shown at `starts` until `end`, each a distinct flat luma, plus AAC from 0.

    With `b_frames`, the container records decode-order sample durations, so a frame's own
    duration differs from the gap to the next frame on a variable clock.
    """
    width, height = size
    durations = {
        int(start / TICK): int((stop - start) / TICK)
        for start, stop in zip(starts, [*starts[1:], end], strict=True)
    }
    with av.open(str(path), "w", format="mp4") as out:
        video = out.add_stream("libx264", rate=None)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.time_base = video.codec_context.time_base = TICK
        video.options = {"preset": "ultrafast", "crf": "40", "bf": str(b_frames)}
        sound = out.add_stream("aac", rate=sample_rate, layout="mono") if audio_seconds else None

        def mux(packets: Sequence[Any]) -> None:
            for packet in packets:
                packet.duration = durations[int(packet.pts or 0)]
                out.mux(packet)

        for index, start in enumerate(starts):
            frame = av.VideoFrame(width, height, "yuv420p")
            luma = frame.planes[0].line_size * height
            chroma = frame.planes[1].line_size * (height // 2)
            frame.planes[0].update(bytes([(index * 37) % 256]) * luma)
            frame.planes[1].update(bytes([128]) * chroma)
            frame.planes[2].update(bytes([128]) * chroma)
            frame.pts, frame.time_base = int(start / TICK), TICK
            mux(video.encode(frame))
        mux(video.encode())
        if sound is not None:
            total = audio_seconds * sample_rate
            clock = np.arange(total, dtype=np.float32)
            wave = (0.2 * np.sin(2 * np.pi * 440 * clock / sample_rate)).astype(np.float32)
            for offset in range(0, total, 1024):
                chunk = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(wave[None, offset : offset + 1024]),
                    format="fltp",
                    layout="mono",
                )
                chunk.sample_rate, chunk.pts = sample_rate, offset
                chunk.time_base = Fraction(1, sample_rate)
                for packet in sound.encode(chunk):
                    out.mux(packet)
            for packet in sound.encode():
                out.mux(packet)
    return path


def webm(
    path: Path,
    *,
    size: tuple[int, int],
    frames: int,
    shade: int = 0,
    options: dict[str, str] | None = None,
) -> Path:
    """VP8 at 30 fps on WebM's millisecond clock."""
    with av.open(str(path), "w", format="webm", options=options or {}) as out:
        stream = out.add_stream("libvpx", rate=30)
        stream.width, stream.height, stream.pix_fmt = size[0], size[1], "yuv420p"
        for index in range(frames):
            pixels = np.full((size[1], size[0], 3), (shade + index * 2) % 256, np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def joined(path: Path, first: Path, second: Path) -> Path:
    """One VP8 track: `first`'s packets, then `second`'s (another size) right after them."""
    with (
        av.open(str(first)) as a,
        av.open(str(second)) as b,
        av.open(str(path), "w", format="webm") as out,
    ):
        track = out.add_stream_from_template(a.streams.video[0])
        offset = 0
        for packet in a.demux(a.streams.video[0]):
            if packet.dts is None:
                continue
            offset = max(offset, int(packet.pts or 0) + int(packet.duration or 0))
            packet.stream = track
            out.mux(packet)
        for packet in b.demux(b.streams.video[0]):
            if packet.dts is None:
                continue
            packet.pts, packet.dts = int(packet.pts or 0) + offset, int(packet.dts) + offset
            packet.stream = track
            out.mux(packet)
    return path
