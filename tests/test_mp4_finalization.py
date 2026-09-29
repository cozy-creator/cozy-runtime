"""Completed downloads expose duration and seek to real reference frames."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from functools import partial
from pathlib import Path

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
    kinds = [kind for kind, *_ in _codec._boxes(path.read_bytes())]
    assert "moov" in kinds and "moof" not in kinds
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        assert video.frames == frames
        assert video.duration * video.time_base == Fraction(frames, 24)
    _seek_frames(path)


@pytest.mark.parametrize("last", [False, True])
@pytest.mark.parametrize("with_audio", [False, True])
def test_progressive_join_finishes_indexed_without_changing_live_parts(
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
        if i < 2:
            revisions.extend((path, path.read_bytes()) for path in parts)
            preview = tmp_path / f"preview-{i}.mp4"
            preview.write_bytes(b"".join(path.read_bytes() for path in parts))
            with av.open(str(preview)) as container:
                assert len(list(container.decode(video=0))) == (i + 1) * 48
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
