"""Real PyAV clocks, side data, and audio survive the Runtime media round trip."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from cozy_runtime.author import VideoAsset, fakes
from cozy_runtime.author._decode import MediaDecodeError

av = pytest.importorskip("av", reason="the media extra is absent from this venv")


@pytest.mark.parametrize("mirror", [False, True], ids=["quarter-turn", "mirror-refused"])
def test_real_rotated_video_and_audio_round_trip(tmp_path: Path, mirror: bool) -> None:
    frames, rate = 12, 32000
    pixels = np.zeros((frames, 32, 48, 3), dtype=np.uint8)
    pixels[:, :, :24, 0] = 255
    waveform = (0.125 * np.sin(np.arange(rate // 2, dtype=np.float32) / 7)).astype("<f4")
    source = fakes.fake_outputs().save_video(pixels, fps=24, audio=waveform, sample_rate=rate)
    assert source._local is not None
    oriented = tmp_path / "oriented.mp4"
    with av.open(str(source._local)) as reader, av.open(str(oriented), "w") as writer:
        streams = {
            stream.index: writer.add_stream_from_template(stream) for stream in reader.streams
        }
        streams[0].set_display_rotation(90, hflip=mirror)
        for packet in reader.demux():
            if packet.dts is not None:
                packet.stream = streams[packet.stream.index]
                writer.mux(packet)

    attempt = fakes.fake_attempt("roundtrip", spool=tmp_path / "roundtrip")
    asset = fakes.fake_input(
        VideoAsset(str(oriented), local=oriented, size_bytes=oriented.stat().st_size),
        attempt=attempt.request_id,
        max_decoded_bytes=8 << 20,
    )
    decoder = fakes.fake_media_decoder(attempt)
    if mirror:
        with pytest.raises(MediaDecodeError) as error, decoder.stream_video(asset) as stream:
            fakes.fake_outputs(attempt).save_video_stream(iter(stream))
        assert error.value.code == "media_rotation"
        return

    original = decoder.decode_video(asset)
    with decoder.stream_video(asset) as stream:
        saved = fakes.fake_outputs(attempt).save_video_stream(iter(stream))
    reinput = fakes.fake_input(saved.video, attempt=attempt.request_id, max_decoded_bytes=8 << 20)
    decoded = decoder.decode_video(reinput)
    assert (decoded.width, decoded.height, decoded.frame_count) == (32, 48, frames)
    assert decoded.start_time == original.start_time == 0
    assert decoded.duration == original.duration == Fraction(frames, 24)
    assert original.soundtrack is not None and decoded.soundtrack is not None
    assert decoded.soundtrack.sample_count == original.soundtrack.sample_count == rate // 2
    assert decoded.soundtrack.start_time == original.soundtrack.start_time == 0
    # The encoded frames retain the display orientation, without leaving a rotation tag
    # to apply for a second time. Lossy codecs need pixel/audio similarity, not byte equality.
    before = np.frombuffer(original.frames_rgb[0], dtype=np.uint8).astype(np.int16)
    after = np.frombuffer(decoded.frames_rgb[0], dtype=np.uint8).astype(np.int16)
    assert float(np.abs(before - after).mean()) < 3
    observed = np.frombuffer(decoded.soundtrack.pcm_f32le[0], dtype="<f4")
    assert float(np.corrcoef(waveform[2048:-2048], observed[2048:-2048])[0, 1]) > 0.95
