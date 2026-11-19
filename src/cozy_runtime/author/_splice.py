"""`Outputs.splice_video`'s codec work (se-177): a window of an H.264/AAC MP4 made again.

Pictures: the source's packets are copied as they are up to the last point before the window that
no reordered frame crosses, and again from the first such IDR after it. Between those, the source's
own frames (decoded) and the window's are encoded by x264 at the source's profile, level, pixel
format, aspect, colour and frame clock, without B-frames. When x264 writes the source's own SPS and
PPS byte for byte (a take this Runtime encoded), the run uses them and the avcC stays the source's,
so the output still joins its sibling shots by packet copy. Otherwise the run's SPS and PPS take an
id the source does not use and join the source's in the avcC, so each copied packet decodes against
its own parameter sets and each IDR switches between them, as H.264 allows. A source without
B-frames re-encodes none of its frames before the window; after it, those up to the next IDR.

Sound: an AAC frame overlaps each neighbour by half, so a copied frame beside an encoded one must
hold only kept sound. The encoded run starts two frames before the window and ends two after it
(43 ms at 48 kHz), fed two more frames of the source on each side, so its frames at the seams are
encoded from the same sound the source's were. A run that starts the track keeps its encoder's
priming, as the source kept its own. When the window changes the length, the copied sound after it
moves by whole AAC frames, the nearest to its picture's move: within 512 samples.
"""

from __future__ import annotations

import math
import os
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from fractions import Fraction
from heapq import merge
from itertools import accumulate, pairwise
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import cozy_runtime.author._codec as _codec
import cozy_runtime.author._color as _color
from cozy_runtime.author._decode import _matrix_rotation, _rotate_rgb, frame_clock
from cozy_runtime.author._errors import OutputError

if TYPE_CHECKING:
    from av.audio.stream import AudioStream
    from av.container import InputContainer
    from av.packet import Packet
    from av.stream import Stream
    from av.video.codeccontext import VideoCodecContext
    from av.video.frame import VideoFrame
    from av.video.stream import VideoStream

type _Muxed = Packet[VideoStream] | Packet[AudioStream]

_FRAME = _codec.AAC_FRAME_SAMPLES
#: Sound encoded again on each side of a seam: the frame overlapping it, and one more.
_MARGIN = 2 * _FRAME
#: x264's name for each H.264 profile_idc it writes.
_PROFILES = {66: "baseline", 77: "main", 100: "high", 110: "high10", 122: "high422", 244: "high444"}


def _frame_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _refuse(why: str) -> OutputError:
    return OutputError(f"the source cannot be spliced: {why}", code="video_copy_incompatible")


def _ue(data: bytes, bit: int) -> tuple[int, int]:
    """One Exp-Golomb value at `bit`, and the bit after it."""
    zeros = 0
    while not data[bit >> 3] >> (7 - bit % 8) & 1:
        zeros += 1
        bit += 1
    value = 1
    for _ in range(zeros):
        bit += 1
        value = value << 1 | data[bit >> 3] >> (7 - bit % 8) & 1
    return value - 1, bit + 1


def _set_id(nal: bytes) -> int:
    """An SPS's or PPS's own id (H.264 7.3.2.1, 7.3.2.2)."""
    rbsp = nal[1:16].replace(b"\x00\x00\x03", b"\x00\x00")
    try:
        return _ue(rbsp, 24 if nal[0] & 31 == 7 else 0)[0]
    except IndexError:
        raise _refuse("an H.264 parameter set is truncated") from None


def _annexb(data: bytes) -> list[bytes]:
    """The NAL units of a start-code stream."""
    return [nal.rstrip(b"\x00") for nal in data.split(b"\x00\x00\x01")[1:]]


def _nal_types(sample: bytes, size: int) -> Iterator[tuple[int, int]]:
    """Each NAL unit's type and offset in a length-prefixed sample."""
    at = 0
    while at + size < len(sample):
        length = int.from_bytes(sample[at : at + size])
        yield sample[at + size] & 31, at + size
        at += size + length


@dataclass(frozen=True, slots=True)
class _AvcC:
    """An avcC record (ISO/IEC 14496-15 5.3.3.1): its head, parameter sets and trailing fields."""

    head: bytes
    sps: tuple[bytes, ...]
    pps: tuple[bytes, ...]
    tail: bytes

    @classmethod
    def read(cls, data: bytes) -> _AvcC:
        def sets(at: int, count: int) -> tuple[tuple[bytes, ...], int]:
            found = []
            for _ in range(count):
                size = int.from_bytes(data[at : at + 2])
                found.append(data[at + 2 : at + 2 + size])
                at += 2 + size
            return tuple(found), at

        try:
            if data[0] != 1:
                raise IndexError
            sps, at = sets(6, data[5] & 31)
            pps, end = sets(at + 1, data[at])
        except IndexError:
            raise _refuse("its H.264 parameter sets are not an avcC record") from None
        if end > len(data) or not sps or not all(sps + pps):
            raise _refuse("its H.264 parameter sets are not an avcC record")
        return cls(data[:5], sps, pps, data[end:])

    @property
    def length(self) -> int:
        """The bytes of each NAL unit's length prefix."""
        return (self.head[4] & 3) + 1

    def joined(self, sps: bytes, pps: bytes) -> bytes:
        """This record with one more SPS and PPS, its compatibility and level covering both."""
        if len(self.sps) >= 31 or len(self.pps) >= 255:
            raise _refuse("its avcC has no room for another parameter set")
        head = (*self.head[:2], self.head[2] & sps[2], max(self.head[3], sps[3]), self.head[4])
        body = bytearray(head)
        body.append(0xE0 | len(self.sps) + 1)
        for nal in (*self.sps, sps):
            body += len(nal).to_bytes(2) + nal
        body.append(len(self.pps) + 1)
        for nal in (*self.pps, pps):
            body += len(nal).to_bytes(2) + nal
        return bytes(body + self.tail)


@dataclass(frozen=True, slots=True)
class _Pictures:
    """The source's H.264 track, read without decoding: its packets in decode order."""

    pts: tuple[int, ...]
    dts: tuple[int, ...]
    #: Per decode position 0..N: whether every frame before it shows before every frame after it.
    clean: tuple[bool, ...]
    #: The clean positions the source can be decoded or copied from: an IDR, or the end.
    cuts: tuple[int, ...]
    first: int
    step: int
    time_base: Fraction
    avcc: _AvcC
    #: Every SPS and PPS id the track uses, in its avcC or its packets.
    ids: frozenset[int]
    #: Whether its packets carry parameter sets of their own.
    inband: bool
    width: int
    height: int
    pix_fmt: str
    #: Its pixel aspect ratio; 0 where it states none.
    sar: Fraction
    #: Primaries, transfer, matrix and range as the track tags them.
    tags: tuple[int, int, int, int]
    #: The quarter turn its display matrix shows it at (`_decode`'s reading).
    turn: int

    def frame(self, pts: int) -> int:
        return (pts - self.first) // self.step

    def time(self, frame: int) -> Fraction:
        return (self.first + frame * self.step) * self.time_base


def _pictures(av: ModuleType, path: Path) -> _Pictures:
    with av.open(str(path), "r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1 or streams[0].codec_context.name != "h264":
            raise _refuse("it is not exactly one H.264 track")
        stream = streams[0]
        context = stream.codec_context
        avcc = _AvcC.read(bytes(context.extradata or b""))
        ids = {_set_id(nal) for nal in (*avcc.sps, *avcc.pps)}
        inband = False
        pts: list[int] = []
        dts: list[int] = []
        idr: list[bool] = []
        for packet in container.demux(stream):
            if not packet.size:
                continue
            if packet.pts is None or packet.dts is None:
                raise _refuse("a picture has no clock")
            if packet.is_discard:
                raise _refuse("an edit list hides some of its frames")
            data = bytes(packet)
            kinds = set()
            for kind, at in _nal_types(data, avcc.length):
                kinds.add(kind)
                if kind in (7, 8):
                    inband = True
                    ids.add(_set_id(data[at : at + 16]))
            pts.append(int(packet.pts))
            dts.append(int(packet.dts))
            idr.append(5 in kinds)
        time_base = Fraction(stream.time_base or 0)
        step = frame_clock(stream, pts).step
        matrix = context.coded_side_data.get("display_matrix")
        tags = (
            int(context.color_primaries),
            int(context.color_trc),
            int(context.colorspace),
            int(context.color_range),
        )
        width, height, pix_fmt = int(context.width), int(context.height), str(context.pix_fmt)
        sar = Fraction(context.sample_aspect_ratio or 0)
    if not pts or step is None:
        raise _refuse("its frames are not on a fixed clock")
    if any(later <= earlier for earlier, later in pairwise(dts)) or any(
        p < d for p, d in zip(pts, dts, strict=True)
    ):
        raise _refuse("its decode clock is not monotone")
    before = list(accumulate(pts, max))
    after = list(accumulate(reversed(pts), min))[::-1]
    count = len(pts)
    clean = tuple(i in (0, count) or before[i - 1] < after[i] for i in range(count + 1))
    cuts = tuple(i for i in range(count + 1) if clean[i] and (i == count or idr[i]))
    if cuts[0] != 0:
        raise _refuse("it does not begin on an IDR")
    return _Pictures(
        tuple(pts),
        tuple(dts),
        clean,
        cuts,
        min(pts),
        step,
        time_base,
        avcc,
        frozenset(ids),
        inband,
        width,
        height,
        pix_fmt,
        sar,
        tags,
        _matrix_rotation(bytes(matrix))[0] if matrix is not None else 0,
    )


@dataclass(frozen=True, slots=True)
class _Sound:
    """The source's AAC-LC track, read without decoding: each frame's first sample, contiguous."""

    starts: tuple[int, ...]
    end: int
    rate: int
    channels: int


def _sound(av: ModuleType, path: Path) -> _Sound | None:
    with av.open(str(path), "r") as container:
        streams = list(container.streams.audio)
        if not streams:
            return None
        if len(streams) > 1:
            raise _refuse("it has more than one sound track")
        stream = streams[0]
        context = stream.codec_context
        rate, channels = int(context.sample_rate or 0), int(context.channels or 0)
        if context.name != "aac" or context.profile != "LC" or context.frame_size != _FRAME:
            raise _refuse("its sound is not AAC-LC")
        if channels not in (1, 2):
            raise _refuse("its sound is not mono or stereo")
        if Fraction(stream.time_base or 0) != Fraction(1, rate):
            raise _refuse("its sound is not on its sample clock")
        starts: list[int] = []
        end = 0
        for packet in container.demux(stream):
            if not packet.size:
                continue
            if packet.pts is None or (starts and packet.pts != starts[-1] + _FRAME):
                raise _refuse("its sound frames are not contiguous")
            starts.append(int(packet.pts))
            end = int(packet.pts) + int(packet.duration or _FRAME)
    if not starts:
        return None
    return _Sound(tuple(starts), end, rate, channels)


def _decoded(
    av: ModuleType, path: Path, pictures: _Pictures, lo: int, hi: int
) -> Iterator[VideoFrame]:
    """The source's frames `lo` to before `hi`, decoded from the cut before `lo`."""
    if lo >= hi:
        return
    begin = pictures.cuts[bisect_right(pictures.cuts, lo) - 1]
    stop = next(i for i in range(hi, len(pictures.clean)) if pictures.clean[i])
    with av.open(str(path), "r") as container:
        stream = container.streams.video[0]

        def kept(frames: Iterable[VideoFrame]) -> Iterator[VideoFrame]:
            for frame in frames:
                if frame.pts is not None and lo <= pictures.frame(int(frame.pts)) < hi:
                    yield frame

        index = 0
        for packet in container.demux(stream):
            if not packet.size:
                continue
            if index >= stop:
                break
            if index >= begin:
                yield from kept(stream.decode(packet))
            index += 1
        yield from kept(stream.decode(None))


def _source_sound(av: ModuleType, path: Path, sound: _Sound, lo: int, hi: int) -> list[bytearray]:
    """The source's decoded sound from sample `lo` to before `hi`, planar float32, silent where it
    has none. Decoding starts two frames early: a decoder's first frame lacks its overlap."""
    planes = [bytearray(max(0, hi - lo) * 4) for _ in range(sound.channels)]
    if hi <= lo:
        return planes
    with av.open(str(path), "r") as container:
        stream = container.streams.audio[0]
        for packet in container.demux(stream):
            if not packet.size or int(packet.pts or 0) < lo - _MARGIN:
                continue
            if int(packet.pts or 0) >= hi:
                break
            for frame in stream.decode(packet):
                at, count = int(frame.pts or 0), frame.samples
                begin, end = max(at, lo), min(at + count, hi)
                if begin >= end:
                    continue
                for plane, decoded in zip(planes, frame.planes, strict=True):
                    plane[(begin - lo) * 4 : (end - lo) * 4] = bytes(decoded)[
                        (begin - at) * 4 : (end - at) * 4
                    ]
    return planes


def _x264(
    av: ModuleType, pictures: _Pictures, sps_id: int
) -> tuple[VideoCodecContext, bytes, bytes]:
    """An open x264 encoder for the source's track, its SPS and PPS of id `sps_id`, and those."""
    profile = pictures.avcc.head[1]
    if profile not in _PROFILES:
        raise _refuse(f"x264 writes no H.264 profile {profile}")
    rate = 1 / (pictures.step * pictures.time_base)
    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height = pictures.width, pictures.height
    encoder.pix_fmt = pictures.pix_fmt
    encoder.sample_aspect_ratio = pictures.sar
    encoder.time_base = 1 / rate
    encoder.framerate = rate
    encoder.thread_count = _codec.x264_threads()
    encoder.gop_size = max(1, round(rate))
    encoder.max_b_frames = 0
    (
        encoder.color_primaries,
        encoder.color_trc,
        encoder.colorspace,
        encoder.color_range,
    ) = pictures.tags
    encoder.flags |= av.codec.context.Flags.global_header
    encoder.options = {
        **_codec._X264,
        "profile": _PROFILES[profile],
        "level": str(pictures.avcc.head[3]),
        "x264-params": f"sps-id={sps_id}",
    }
    encoder.open()
    sets = _annexb(bytes(encoder.extradata or b""))
    sps = [nal for nal in sets if nal[0] & 31 == 7]
    pps = [nal for nal in sets if nal[0] & 31 == 8]
    if len(sps) != 1 or len(pps) != 1 or sps[0][1] != profile:
        raise _refuse(f"x264 cannot encode its H.264 profile {profile}")
    return encoder, sps[0], pps[0]


class _Run:
    """The pictures between the copied ones, encoded by x264 as the source's track was encoded:
    same profile, level, pixel format, aspect, colour and clock, no B-frames, under the source's
    parameter sets when x264 writes them byte for byte, else under its own of a free id.
    Its packets wait in a scratch file in decode order, length-prefixed as the source's are."""

    def __init__(self, av: ModuleType, pictures: _Pictures, scratch: Path, max_bytes: int) -> None:
        avcc = pictures.avcc
        encoder, sps, pps = _x264(av, pictures, _set_id(avcc.sps[0]))
        #: The track's avcC with the run's parameter sets added; None when they are the source's.
        self.extradata: bytes | None = None
        if pictures.inband or sps not in avcc.sps or pps not in avcc.pps:
            free = next((i for i in range(32) if i not in pictures.ids), None)
            if free is None:
                raise _refuse("it uses every parameter-set id")
            encoder, sps, pps = _x264(av, pictures, free)
            self.extradata = avcc.joined(sps, pps)
        self.frames = 0
        #: Each packet's offset and size in the scratch file, and whether it is an IDR.
        self.packets: list[tuple[int, int, bool]] = []
        self._encoder = encoder
        self._none = av.video.frame.PictureType.NONE
        self._length = pictures.avcc.length
        self._max_bytes = max_bytes
        self._file = scratch.open("x+b")

    def write(self, frame: VideoFrame) -> None:
        frame.pts = self.frames
        frame.time_base = self._encoder.time_base
        # x264 would follow a decoded frame's own type.
        frame.pict_type = self._none
        self._keep(self._encoder.encode(frame))
        self.frames += 1

    def finish(self) -> None:
        self._keep(self._encoder.encode(None))
        self._file.flush()

    def read(self, index: int) -> bytes:
        offset, size, _ = self.packets[index]
        return os.pread(self._file.fileno(), size, offset)

    def close(self) -> None:
        self._file.close()

    def _keep(self, packets: Iterable[Packet[Stream]]) -> None:
        for packet in packets:
            try:
                body = b"".join(
                    len(nal).to_bytes(self._length) + nal
                    for nal in _annexb(bytes(packet))
                    if nal[0] & 31 not in (7, 8, 9)
                )
            except OverflowError:
                raise _refuse(f"its {self._length}-byte NAL lengths cannot hold x264's") from None
            offset = self._file.tell()
            if offset + len(body) > self._max_bytes:
                raise OutputError(
                    f"the spliced video would cross the {self._max_bytes}-byte limit",
                    code="output_too_large",
                )
            self._file.write(body)
            self.packets.append((offset, len(body), packet.is_keyframe))


@dataclass(frozen=True, slots=True)
class Spliced:
    """A written splice: its probe and the sound samples it holds from time zero."""

    facts: _codec.EncodedVideoFacts
    samples: int


class VideoSplice:
    """`source` with its frames `start` to before `end` replaced by the window written to it:
    `write_video` and `write_audio` in the window's order, then `finish`."""

    def __init__(
        self,
        source: Path,
        destination: Path,
        start: int,
        end: int | None,
        *,
        scratch: Path,
        max_bytes: int,
    ) -> None:
        self._av = av = _codec._av()
        self._source, self._destination = source, destination
        self._scratch, self._max_bytes = scratch, max_bytes
        self.pictures = pictures = _pictures(av, source)
        self.sound = _sound(av, source)
        count = len(pictures.pts)
        end = count if end is None else end
        if not (_frame_index(start) and _frame_index(end) and 0 <= start <= end <= count):
            raise OutputError(
                f"frames {start} to {end} are not within the source's {count}", code="splice_range"
            )
        self.start, self.end = start, end
        #: Decode positions copied: before `head`, and from `tail`.
        self.head = max(i for i in range(start + 1) if pictures.clean[i])
        self.tail = pictures.cuts[bisect_left(pictures.cuts, end)]
        #: The source's display-oriented size, and the window's when it differs and is scaled.
        turned = pictures.turn in (90, -90)
        self.size = (
            (pictures.height, pictures.width) if turned else (pictures.width, pictures.height)
        )
        self.scaled: tuple[int, int] | None = None
        self.window = 0
        self._run: _Run | None = None
        self._reformatter = av.video.reformatter.VideoReformatter()
        primaries, transfer, matrix, color_range = _color.normalize(
            width=pictures.width,
            height=pictures.height,
            primaries=pictures.tags[0],
            transfer=pictures.tags[1],
            matrix=pictures.tags[2],
            color_range=2 if pictures.pix_fmt.startswith("yuvj") else pictures.tags[3],
        )
        try:
            self._colorspace = _color.colorspace(av, matrix)
            self._color_range = _color.range_value(av, color_range)
        except ValueError as exc:
            raise _refuse(str(exc)) from exc
        self._color_tags = (primaries, transfer, matrix)
        self._frame_time: Fraction | None = None
        self._sound_time: Fraction | None = None
        self._sound_rate = 0
        self._sound_parts: list[_codec.Planar] = []
        try:
            for frame in _decoded(av, source, pictures, self.head, start):
                self._encode(frame)
        except BaseException:
            self.close()
            raise

    def write_video(self, rgb: bytes, width: int, height: int, time: Fraction) -> None:
        """One display-oriented RGB8 frame of the window, shown at `time` on its own clock."""
        if (width, height) != self.size:
            self.scaled = (width, height)
        rgb, width, height = _rotate_rgb(rgb, width, height, -self.pictures.turn)
        if self._frame_time is None:
            self._frame_time = time
        self._encode(
            _codec.yuv_frame(
                self._av,
                self._reformatter,
                rgb,
                width,
                height,
                size=(self.pictures.width, self.pictures.height),
                pix_fmt=self.pictures.pix_fmt,
                tags=self._color_tags,
                colorspace=self._colorspace,
                color_range=self._color_range,
            )
        )
        self.window += 1

    def write_audio(self, planar: _codec.Planar, rate: int, time: Fraction) -> None:
        """The window's next stretch of sound; the first starts at `time` on the window's clock."""
        if self._sound_time is None:
            self._sound_time = time
        self._sound_rate = rate
        self._sound_parts.append(planar)

    def finish(self) -> Spliced:
        av = self._av
        try:
            for frame in _decoded(av, self._source, self.pictures, self.end, self.tail):
                self._encode(frame)
            expected = (self.start - self.head) + self.window + (self.tail - self.end)
            if self._run is not None:
                self._run.finish()
                if len(self._run.packets) != expected:
                    raise OutputError(
                        f"x264 wrote {len(self._run.packets)} pictures of {expected}",
                        code="output_probe",
                    )
            elif expected:
                raise OutputError("the source's seam frames did not decode", code="output_probe")
            samples = self._mux()
        finally:
            self.close()
        facts = _codec.probe_mp4(self._destination)
        frames = len(self.pictures.pts) - (self.end - self.start) + self.window
        if facts.frame_count != frames or facts.audio_decoded_samples != samples:
            raise OutputError(
                f"spliced MP4 probe disagrees: {facts.frame_count} frames of {frames}, "
                f"{facts.audio_decoded_samples} samples of {samples}",
                code="output_probe",
            )
        return Spliced(facts, samples)

    def close(self) -> None:
        """Drop the run's scratch file; the output stays for the caller to keep or remove."""
        if self._run is not None:
            self._run.close()
        self._scratch.unlink(missing_ok=True)

    def _encode(self, frame: VideoFrame) -> None:
        if self._run is None:
            self._run = _Run(self._av, self.pictures, self._scratch, self._max_bytes)
        self._run.write(frame)

    def _mux(self) -> int:
        """Write the output: copied packets, the run's, copied again. The sound samples it holds."""
        av = self._av
        raw = self._destination.open("x+b")
        with raw:
            with (
                av.open(str(self._source), "r") as pictures,
                av.open(str(self._source), "r") as sound,
            ):
                # Edit-list gaps use the movie clock (`_codec.finalize_mp4`).
                timescale = math.lcm(
                    *(Fraction(s.time_base or 1).denominator for s in pictures.streams)
                )
                out = av.open(
                    _codec._BoundedWriter(raw, self._max_bytes),
                    "w",
                    format="mp4",
                    options={
                        "fflags": "+bitexact",
                        "use_editlist": "1",
                        "movie_timescale": str(min(timescale, (1 << 31) - 1)),
                    },
                )
                try:
                    track = out.add_stream_from_template(pictures.streams.video[0])
                    if self._run is not None and self._run.extradata is not None:
                        track.codec_context.extradata = self._run.extradata
                    tracks: list[Iterator[_Muxed]] = [self._pictures_out(pictures, track)]
                    if self.sound is not None:
                        audio = out.add_stream_from_template(sound.streams.audio[0])
                        tracks.append(self._sound_out(sound, audio, self.sound))
                    written = 0
                    for packet in merge(*tracks, key=_decode_time):
                        if packet.stream.type == "audio":
                            start, length = int(packet.pts or 0), int(packet.duration or 0)
                            written += max(0, start + length) - max(0, start)
                        out.mux(packet)
                except BaseException:
                    # Closing a failed AVIO can raise again; keep the actual refusal.
                    with suppress(BaseException):
                        out.close()
                    raise
                out.close()
            raw.flush()
            os.fsync(raw.fileno())
        return written

    def _pictures_out(
        self, container: InputContainer, track: VideoStream
    ) -> Iterator[Packet[VideoStream]]:
        """The picture packets in decode order, on one decode clock a frame per step."""
        pictures = self.pictures
        shift = (self.window - (self.end - self.start)) * pictures.step
        position = 0
        # Early enough for every picture to decode by its time: a join of tracks that reorder
        # differently (one with B-frames, one without) starts before its first packet's.
        copied = (
            t - i * pictures.step
            for i, t in enumerate(pictures.pts)
            if i < self.head or i >= self.tail
        )
        origin = min(pictures.dts[0], pictures.first, *copied)

        def stamped(packet: Packet[VideoStream], pts: int) -> Packet[VideoStream]:
            nonlocal position
            packet.pts, packet.dts = pts, origin + position * pictures.step
            if packet.pts < packet.dts:
                raise _refuse("its pictures reorder further than its decode clock allows")
            packet.stream = track
            position += 1
            return packet

        def run() -> Iterator[Packet[VideoStream]]:
            assert self._run is not None
            for index, (_, _, key) in enumerate(self._run.packets):
                packet = self._av.Packet(self._run.read(index))
                packet.time_base = pictures.time_base
                packet.duration = pictures.step
                packet.is_keyframe = key
                yield stamped(packet, pictures.first + (self.head + index) * pictures.step)

        source = (packet for packet in container.demux(container.streams.video[0]) if packet.size)
        for index, packet in enumerate(source):
            if index == self.head and self._run is not None:
                yield from run()
            if index < self.head:
                yield stamped(packet, int(packet.pts or 0))
            elif index >= self.tail:
                yield stamped(packet, int(packet.pts or 0) + shift)
        if self.head == len(pictures.pts) and self._run is not None:
            yield from run()

    def _sound_out(
        self, container: InputContainer, track: AudioStream, sound: _Sound
    ) -> Iterator[Packet[AudioStream]]:
        """The sound packets: the source's, an encoded run around the window, the source's moved."""
        pictures = self.pictures
        at = [round(pictures.time(frame) * sound.rate) for frame in (self.start, self.end)]
        window_end = round(pictures.time(self.start + self.window) * sound.rate)
        shift = window_end - at[1]
        starts = sound.starts
        head = max(0, bisect_right(starts, at[0] - _MARGIN) - 1)
        tail = bisect_left(starts, at[1] + _MARGIN)
        if tail < len(starts):
            shift = _FRAME * round(Fraction(shift, _FRAME))
        begin = starts[0] + _FRAME if head == 0 else max(starts[0] + _FRAME, starts[head] - _MARGIN)
        keep = starts[0] if head == 0 else starts[head]
        until = starts[tail] + shift if tail < len(starts) else None
        stop = until + _MARGIN if until is not None else max(window_end, sound.end + shift)
        pcm = self._run_sound(sound, begin, stop, at[0], window_end, shift)
        encoded = [
            packet
            for packet in _encode_sound(self._av, sound, pcm, begin)
            if int(packet.pts or 0) >= keep and (until is None or int(packet.pts or 0) < until)
        ]
        clock = [int(packet.pts or 0) for packet in encoded]
        if clock != list(range(keep, keep + _FRAME * len(clock), _FRAME)) or (
            until is not None and keep + _FRAME * len(clock) != until
        ):
            raise OutputError("the encoded sound is off its frame grid", code="output_probe")
        source = (packet for packet in container.demux(container.streams.audio[0]) if packet.size)
        for index, packet in enumerate(source):
            if index == head:
                for made in encoded:
                    made.stream = track
                    yield made
            if index < head:
                packet.stream = track
                yield packet
            elif index >= tail:
                packet.pts = int(packet.pts or 0) + shift
                packet.dts = packet.pts
                packet.stream = track
                yield packet

    def _run_sound(
        self, sound: _Sound, begin: int, stop: int, start: int, window_end: int, shift: int
    ) -> list[bytearray]:
        """The encoded run's sound from `begin` to `stop` on the output's sample clock: the source's
        before `start`, the window's until `window_end`, then the source's moved by `shift`."""
        av = self._av
        window = self._window_sound(sound, window_end - start)
        before = _source_sound(av, self._source, sound, begin, min(start, stop))
        after = _source_sound(av, self._source, sound, max(window_end, begin) - shift, stop - shift)
        middle = [
            plane[max(0, begin - start) * 4 : (min(window_end, stop) - start) * 4]
            for plane in window
        ]
        return [a + b + c for a, b, c in zip(before, middle, after, strict=True)]

    def _window_sound(self, sound: _Sound, length: int) -> list[bytearray]:
        """The window's sound at the source's rate and channels, placed against its first frame
        and cut or padded to `length` samples."""
        planes = [bytearray(length * 4) for _ in range(sound.channels)]
        if not self._sound_parts or self._frame_time is None or not length:
            return planes
        assert self._sound_time is not None
        av = self._av
        rate = self._sound_rate
        layout = "mono" if sound.channels == 1 else "stereo"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=sound.rate)
        frames = []
        played = 0
        for planar in self._sound_parts:
            for frame in _codec._audio_frames(av, planar, rate, "flt", _FRAME, start_pts=played):
                frames += resampler.resample(frame)
            played += planar.shape[1]
        frames += resampler.resample(None)
        at = round((self._sound_time - self._frame_time) * sound.rate)
        for frame in frames:
            count = frame.samples
            begin, end = max(at, 0), min(at + count, length)
            if begin < end:
                for plane, made in zip(planes, frame.planes, strict=True):
                    plane[begin * 4 : end * 4] = bytes(made)[(begin - at) * 4 : (end - at) * 4]
            at += count
        return planes


def _decode_time(packet: _Muxed) -> Fraction:
    return int(packet.dts or 0) * packet.time_base


def _encode_sound(
    av: ModuleType, sound: _Sound, planes: list[bytearray], begin: int
) -> list[Packet[AudioStream]]:
    """`planes` as AAC frames, the first fed sample at `begin` on the output's sample clock."""
    aac = av.CodecContext.create("aac", "w")
    aac.sample_rate = sound.rate
    aac.layout = "mono" if sound.channels == 1 else "stereo"
    aac.format = "fltp"
    aac.bit_rate = _codec._AAC_BIT_RATE
    aac.thread_count = 1
    aac.time_base = Fraction(1, sound.rate)
    aac.open()
    count = len(planes[0]) // 4
    packets: list[Packet[AudioStream]] = []
    if count:
        planar = _codec.Planar(b"".join(planes), (sound.channels, count))
        for frame in _codec._audio_frames(av, planar, sound.rate, "flt", _FRAME, start_pts=begin):
            packets += aac.encode(frame)
    packets += aac.encode(None)
    return packets
