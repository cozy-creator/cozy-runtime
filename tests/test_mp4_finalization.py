"""Every published revision exposes duration and seeks to real reference frames."""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from cozy_runtime.author import DecodedAudioChunk, OutputError, VideoAsset, _codec, fakes
from cozy_runtime.author._decode import DecodedMediaEvent
from test_video_concat import AUDIO, _packets, _segment

av = pytest.importorskip("av")


def _seek_frames(path: Path) -> None:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        expected = {
            frame.pts: frame.to_ndarray(format="yuv420p").tobytes()
            for frame in container.decode(stream)
        }
    assert expected
    times = sorted(expected)
    # Cold opens and backwards/non-keyframe seeks must decode the same pictures.
    for index in (len(times) - 2, 13, len(times) // 2 + 7, 1):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            target = times[index]
            container.seek(target, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts >= target:
                    assert frame.pts == target
                    assert frame.to_ndarray(format="yuv420p").tobytes() == expected[target]
                    break
            else:
                pytest.fail(f"seek to {target} did not produce a frame")


def _indexed(path: Path, frames: int) -> None:
    data = path.read_bytes()
    kinds = [kind for kind, *_ in _codec._boxes(data)]
    assert "moov" in kinds and "moof" not in kinds
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        assert video.frames == frames
        assert video.duration * video.time_base == Fraction(frames, 24)
    _seek_frames(path)
    if tool := shutil.which("gst-discoverer-1.0"):
        probe = subprocess.run([tool, str(path)], text=True, capture_output=True, check=True)
        duration = re.search(r"Duration: (\d+):(\d+):(\d+)\.(\d+)", probe.stdout)
        assert duration is not None, probe.stdout
        hours, minutes, seconds, nanos = duration.groups()
        measured = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + Fraction("0." + nanos)
        _, start, head, size = next(box for box in _codec._boxes(data) if box[0] == "moov")
        _, at, header, _ = next(
            box for box in _codec._boxes(data, start + head, start + size) if box[0] == "mvhd"
        )
        offset = at + header
        timescale = struct.unpack_from(">I", data, offset + (20 if data[offset] else 12))[0]
        assert abs(measured - Fraction(frames, 24)) <= Fraction(1, timescale)


def _timeline(path: Path) -> list[tuple[str, bytes, Fraction, Fraction]]:
    with av.open(str(path)) as container:
        return sorted(
            [
                (p.stream.type, bytes(p), p.pts * p.time_base, p.dts * p.time_base)
                for p in container.demux()
                if p.size
            ],
            key=lambda row: (row[0], row[3]),
        )


def _audio(path: Path) -> list[tuple[int, Fraction, int, bytes]]:
    with av.open(str(path)) as container:
        return [
            (f.pts, f.time_base, f.samples, f.to_ndarray().tobytes())
            for f in container.decode(audio=0)
        ]


@pytest.mark.parametrize("last", [False, True])
@pytest.mark.parametrize("with_audio", [False, True])
def test_every_join_revision_is_indexed_without_changing_packets_or_previous_revisions(
    tmp_path: Path, last: bool, with_audio: bool
) -> None:
    sources = [_segment(tmp_path, name, 48) for name in ("a", "bb", "ccc")]
    parent = fakes.fake_attempt("parent", spool=tmp_path / "parent")
    outputs = fakes.fake_outputs(parent)
    decoder = fakes.fake_media_decoder(parent)
    join = outputs.join_video(AUDIO if with_audio else None)
    written = 0
    revisions: list[tuple[Path, bytes]] = []

    def soundtrack(video: VideoAsset, frames: int) -> Iterator[DecodedMediaEvent]:
        nonlocal written
        if with_audio:
            with decoder.stream_audio(video) as stream:
                for event in stream:
                    if isinstance(event, DecodedAudioChunk):
                        yield DecodedAudioChunk(
                            channels=event.channels,
                            sample_count=event.sample_count,
                            sample_rate=event.sample_rate,
                            channel_layout=event.channel_layout,
                            channel_names=event.channel_names,
                            pcm_f32le=event.pcm_f32le,
                            pts=written,
                            time_base=event.time_base,
                        )
                        written += event.sample_count

    for i, source in enumerate(sources):
        video = fakes.fake_input(
            source, attempt="parent", input_id=f"v{i}", max_decoded_bytes=8 << 20
        )
        revision = join.append(video, partial(soundtrack, video), last=last and i == 2)
        parts = [parent.spool / part.local for part in parent.parts[revision.ref]]
        assert len(parts) == 1
        preview = parts[0]
        _indexed(preview, (i + 1) * 48)
        revisions.append((preview, preview.read_bytes()))
        internal = tmp_path / f"internal-{i}.mp4"
        internal.write_bytes(b"".join(path.read_bytes() for path, _ in join._join.parts))
        assert _timeline(preview) == _timeline(internal)
        if with_audio:
            decoded = _audio(preview)
            assert decoded == _audio(internal)  # AAC priming and sample clocks stay intact.
            assert decoded[0][0] == 0
            count = sum(samples for _, _, samples, _ in decoded)
            if not (last and i == 2):
                assert count <= written  # The unfinished AAC encoder retains its tail.
    saved = join.finish()
    final = saved.video._local
    assert final is not None
    _indexed(final, 144)
    assert _packets(final) == [
        packet
        for source in sources
        for packet in _packets(source._local)  # type: ignore[arg-type]
    ]
    assert all(path.read_bytes() == original for path, original in revisions)
    if last:
        assert b"".join(path.read_bytes() for path in parts) == final.read_bytes()
    if with_audio:
        assert saved.audio is not None and saved.audio.decoded_samples == written
        # Raw AAC decoders may expose a padded final frame. The MP4 duration and
        # the SDK's media reader must trim that padding to the authored samples.
        extra = sum(samples for _, _, samples, _ in _audio(final)) - written
        assert 0 <= extra < saved.audio.codec_frame_samples
        hydrated = fakes.fake_input(
            saved.video, attempt="parent", input_id="finished", max_decoded_bytes=8 << 20
        )
        with decoder.stream_audio(hydrated) as stream:
            actual_samples = sum(e.sample_count for e in stream if isinstance(e, DecodedAudioChunk))
            assert actual_samples == written
    again = join.finish()
    assert again.video._local is not None and again.video._local.read_bytes() == final.read_bytes()
    assert again.submitted_audio_samples == saved.submitted_audio_samples
    assert not list((parent.spool / parts[0].parent.name).glob(".join-*"))


def test_stream_encoder_finishes_with_duration_and_sample_index(tmp_path: Path) -> None:
    source = _segment(tmp_path, "stream", 73)
    assert source._local is not None
    _indexed(source._local, 73)


def test_finalization_copies_audio_and_video_packets_and_keeps_source_on_failure(
    tmp_path: Path,
) -> None:
    source = _segment(tmp_path, "source", 73)._local
    assert source is not None
    original = source.read_bytes()
    destination = tmp_path / "final.mp4"
    _codec.finalize_mp4(source, destination, max_bytes=1 << 20)

    def packets(path: Path) -> list[tuple[str, bytes, Fraction]]:
        with av.open(str(path)) as container:
            return sorted(
                [
                    (packet.stream.type, bytes(packet), packet.pts * packet.time_base)
                    for packet in container.demux()
                    if packet.size
                ],
                key=lambda row: (row[0], row[2]),
            )

    assert packets(destination) == packets(source)
    failed = tmp_path / "failed.mp4"
    with pytest.raises(OutputError, match="attempt limit"):
        _codec.finalize_mp4(source, failed, max_bytes=100)
    assert not failed.exists() and source.read_bytes() == original
    with pytest.raises(FileExistsError):
        _codec.finalize_mp4(source, destination, max_bytes=1 << 20)
    assert packets(destination) == packets(source)


def test_reordered_video_keeps_its_presentation_clock_when_joined(tmp_path: Path) -> None:
    source = tmp_path / "reordered.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.max_b_frames = 2
        stream.codec_context.gop_size = 24
        for i in range(48):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), i * 5, np.uint8), format="rgb24"
            )
            frame.pts, frame.time_base = i, Fraction(1, 24)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    with av.open(str(source)) as container:
        packets = [p for p in container.demux(video=0) if p.size]
        assert any(p.pts != p.dts for p in packets)
    directory = tmp_path / "join"
    directory.mkdir()
    join = _codec.FragmentedJoin(
        directory, audio_channels=0, audio_sample_rate=0, max_bytes=1 << 20
    )
    for _ in range(3):
        join.append(source, lambda _: ())
    preview = tmp_path / "preview.mp4"
    preview.write_bytes(b"".join(part.read_bytes() for part, _ in join.parts))
    join.finish()
    final = join.parts[0][0]
    _indexed(final, 144)

    def clocks(path: Path) -> list[tuple[Fraction, Fraction]]:
        with av.open(str(path)) as container:
            return [
                (p.pts * p.time_base, p.dts * p.time_base)
                for p in container.demux(video=0)
                if p.size
            ]

    assert clocks(final) == clocks(preview)
    assert min(pts for pts, _ in clocks(final)) == 0

    # A valid input may also start after zero. Keep that exact edit-list gap;
    # the default millisecond movie clock used to round it by four video ticks.
    shifted = tmp_path / "shifted.mp4"
    with (
        av.open(str(source)) as incoming,
        av.open(str(shifted), "w", options={"movie_timescale": "12288"}) as outgoing,
    ):
        track = outgoing.add_stream_from_template(incoming.streams.video[0])
        for packet in incoming.demux(video=0):
            if packet.size:
                packet.pts += 1024
                packet.dts += 1024
                packet.stream = track
                outgoing.mux(packet)
    indexed = tmp_path / "shifted-indexed.mp4"
    _codec.finalize_mp4(shifted, indexed, max_bytes=1 << 20)
    assert clocks(indexed) == clocks(shifted)
