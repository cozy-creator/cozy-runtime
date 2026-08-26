#!/usr/bin/env python
"""cr-017's codec plane, live: FLAC + muxed mp4 arms over the REAL author services.

No worker, no GPU, no weights: these arms exercise `Outputs.save_audio`/`save_video`
exactly as an endpoint calls them — through an `Attempt` and its spool — with synthetic
deterministic fixtures. The tensor duck-typing means the same code path a torch tensor
takes after `.cpu().numpy()` is the one the ndarray-like fixtures take here; the
device-side D2H half of the source-independence law is se-002's pod arm.

    nice -n 19 uv run python scripts/media-live.py
"""

from __future__ import annotations

import io
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from array import array
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cozy_runtime.author._assets import digest_bytes  # noqa: E402
from cozy_runtime.author._errors import CapabilityError, OutputError  # noqa: E402
from cozy_runtime.author._services import Attempt, Outputs  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "ok " if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    PASS += ok
    FAIL += not ok


class NdLike:
    """The minimal ndarray-like the duck contract admits (shape/dtype/tobytes)."""

    def __init__(self, data: bytes, shape: tuple[int, ...], dtype: str) -> None:
        self._data = data
        self.shape = shape
        self.dtype = dtype

    def tobytes(self) -> bytes:
        return bytes(self._data)


def _tri(v: int) -> int:
    """Continuous triangle wave over 0..255 — no 255->0 cliffs, so the decode-back bound
    measures the CODEC (crf 17 + 4:2:0 chroma), not fixture discontinuities."""
    return abs((v % 510) - 255)


def video_fixture(t: int, h: int, w: int) -> NdLike:
    """Deterministic smooth gradient with per-frame motion: every frame differs, and the
    signal is continuous so the crf-17 tolerance arm has an honest subject."""
    frames = bytearray()
    for f in range(t):
        for y in range(h):
            row = bytearray()
            for x in range(w):
                row += bytes((_tri(x * 2 + f * 5), _tri(y * 3 + f * 7), _tri(x + y)))
            frames += row
    return NdLike(bytes(frames), (t, h, w, 3), "uint8")


def tone(seconds: float, rate: int, amp: float = 0.5, channels: int = 1) -> NdLike:
    n = int(seconds * rate)
    mono = array("f", (amp * math.sin(2 * math.pi * 440 * i / rate) for i in range(n)))
    if channels == 1:
        return NdLike(mono.tobytes(), (n,), "float32")
    return NdLike(mono.tobytes() * channels, (channels, n), "float32")


def fresh(tmp: Path, cap: int = 256 * 1024 * 1024) -> tuple[Attempt, Outputs]:
    spool = Path(tempfile.mkdtemp(dir=tmp, prefix="attempt-"))
    attempt = Attempt("media-live", spool, max_output_bytes=cap)
    return attempt, Outputs(attempt)


def expects(name: str, code: str, fn: object) -> None:
    try:
        fn()  # type: ignore[operator]
    except (OutputError, CapabilityError) as err:
        check(name, err.code == code, f"code={err.code}")
        return
    check(name, False, "no refusal raised")


def main() -> int:
    import av

    tmp = Path(tempfile.mkdtemp(prefix="media-live-"))
    t0 = time.perf_counter()
    fps, rate = 24.0, 44100
    frames = video_fixture(16, 96, 128)
    audio = tone(16 / 24, rate)

    print("section: product path (mux, streams, geometry)")
    attempt, out = fresh(tmp)
    asset = out.save_video(frames, fps=fps, audio=audio, sample_rate=rate)
    data = (attempt.spool / f"video-{asset.ref.rsplit('/', 1)[-1]}").read_bytes()
    check("registered pending", asset.ref in attempt.pending and asset.media_type == "video/mp4")
    check(
        "spool blob is the asset",
        len(data) == asset.size_bytes and digest_bytes(data) == asset.digest,
        f"{len(data)} B",
    )
    probe: Any = av.open(io.BytesIO(data))
    vstreams = probe.streams.video
    astreams = probe.streams.audio
    decoded = [f.reformat(format="rgb24") for f in probe.decode(video=0)]
    check("two streams muxed", len(vstreams) == 1 and len(astreams) == 1)
    check(
        "video geometry + rate",
        vstreams[0].width == 128
        and vstreams[0].height == 96
        and float(vstreams[0].average_rate) == fps
        and len(decoded) == 16,
        f"{vstreams[0].width}x{vstreams[0].height}@{float(vstreams[0].average_rate)} "
        f"x{len(decoded)}",
    )
    check(
        "audio clock survives the mux",
        astreams[0].sample_rate == rate and astreams[0].codec_context.codec.name == "aac",
    )
    dur = float(vstreams[0].duration * vstreams[0].time_base)
    check("duration is frames/fps", abs(dur - 16 / 24) < 1e-6, f"{dur:.4f} s")

    worst = 0
    src = frames.tobytes()
    stride = 96 * 128 * 3
    for i, frame in enumerate(decoded):
        plane = frame.planes[0]
        raw = bytes(plane)
        width3 = 128 * 3
        rows = (
            raw
            if plane.line_size == width3
            else b"".join(
                raw[y * plane.line_size : y * plane.line_size + width3] for y in range(96)
            )
        )
        worst = max(
            worst,
            max(abs(a - b) for a, b in zip(rows, src[i * stride : (i + 1) * stride], strict=True)),
        )
    check("decode-back tolerance (crf 17)", worst <= 24, f"max|d|={worst}")

    print("section: mute + flac")
    attempt, out = fresh(tmp)
    muted = out.save_video(frames, fps=fps, audio=None)
    mdata = (attempt.spool / f"video-{muted.ref.rsplit('/', 1)[-1]}").read_bytes()
    mprobe: Any = av.open(io.BytesIO(mdata))
    check(
        "mute omits the track",
        len(mprobe.streams.video) == 1 and len(mprobe.streams.audio) == 0,
    )

    attempt, out = fresh(tmp)
    flac = out.save_audio(audio, sample_rate=rate)
    fdata = (attempt.spool / f"audio-{flac.ref.rsplit('/', 1)[-1]}").read_bytes()
    got = array("h")
    fprobe: Any = av.open(io.BytesIO(fdata))
    for aframe in fprobe.decode(audio=0):
        got.frombytes(bytes(aframe.planes[0])[: aframe.samples * 2])
    floats = array("f")
    floats.frombytes(audio.tobytes())
    want = array("h", (round(max(-1.0, min(1.0, x)) * 32767.0) for x in floats))
    check(
        "flac roundtrip bit-exact vs s16 quantization",
        got.tolist() == want.tolist() and flac.media_type == "audio/flac",
        f"{len(got)} samples",
    )

    print("section: determinism (pinned settings, bitexact)")
    digests = set()
    for _ in range(3):
        attempt, out = fresh(tmp)
        digests.add(out.save_video(frames, fps=fps, audio=audio, sample_rate=rate).digest)
    child = subprocess.run(
        [sys.executable, __file__, "--digest-only"],
        capture_output=True,
        text=True,
        env=dict(os.environ),
        check=False,
    )
    digests.add(child.stdout.strip().splitlines()[-1] if child.returncode == 0 else "child-failed")
    check(
        "3 in-process + 1 fresh-process digests identical",
        len(digests) == 1,
        next(iter(digests))[:16],
    )

    print("section: clamp visibility")
    attempt, out = fresh(tmp)
    loud = NdLike(array("f", [0.0, 1.5, -2.0, 0.25, 1.0, -1.0]).tobytes(), (6,), "float32")
    out.save_audio(loud, sample_rate=8000)
    clamps = [r for r in attempt.rows if r.kind == "clamp" and r.source == "policy"]
    check(
        "clamp is an adjustment row, never silent",
        len(clamps) == 1 and "2 samples" in str(clamps[0].requested),
        str(clamps[0].requested) if clamps else "no row",
    )

    print("section: typed refusals (all pre-encode, spool untouched)")
    attempt, out = fresh(tmp)
    expects(
        "odd width refuses",
        "frame_shape",
        lambda: out.save_video(video_fixture(2, 96, 127), fps=24.0),
    )
    expects(
        "wrong dtype refuses",
        "frame_dtype",
        lambda: out.save_video(NdLike(b"\0" * 12, (1, 2, 2, 3), "float32"), fps=24.0),
    )
    expects(
        "non-tensor refuses",
        "frame_type",
        lambda: out.save_video(object(), fps=24.0),
    )
    expects(
        "audio without its clock refuses",
        "audio_rate",
        lambda: out.save_video(frames, fps=24.0, audio=audio),
    )
    expects(
        "codec zoo closed (video)",
        "encoder_unavailable",
        lambda: out.save_video(frames, fps=24.0, format="webm"),
    )
    expects(
        "codec zoo closed (audio)",
        "encoder_unavailable",
        lambda: out.save_audio(audio, sample_rate=rate, format="ogg"),
    )
    expects(
        "NaN waveform refuses",
        "audio_nonfinite",
        lambda: out.save_audio(
            NdLike(array("f", [0.1, math.nan, 0.2]).tobytes(), (3,), "float32"),
            sample_rate=rate,
        ),
    )
    expects(
        "float64 audio refuses",
        "audio_dtype",
        lambda: out.save_audio(NdLike(b"\0" * 24, (3,), "float64"), sample_rate=rate),
    )
    expects(
        "zero sample rate refuses",
        "audio_rate",
        lambda: out.save_audio(audio, sample_rate=0),
    )
    check("no spool blob from any refusal", not list(attempt.spool.iterdir()), "spool empty")

    print("section: transactional bounds")
    attempt, out = fresh(tmp, cap=4096)
    expects(
        "over the output cap refuses at register",
        "output_too_large",
        lambda: out.save_video(frames, fps=24.0),
    )
    check(
        "refused output leaves nothing",
        not attempt.pending and not list(attempt.spool.iterdir()),
    )
    attempt, out = fresh(tmp)
    attempt.closed = True
    expects(
        "escaped handle refuses BEFORE the encode burns",
        "escaped_handle",
        lambda: out.save_video(frames, fps=24.0),
    )

    print("section: source-independence (CPU half of the red arm)")
    attempt, out = fresh(tmp)
    mutable = bytearray(video_fixture(4, 32, 32).tobytes())
    tensor = NdLike(bytes(mutable), (4, 32, 32, 3), "uint8")
    asset = out.save_video(tensor, fps=24.0)
    before = asset.digest
    mutable[: 3 * 1024] = b"\xff" * (3 * 1024)
    after_bytes = (attempt.spool / f"video-{asset.ref.rsplit('/', 1)[-1]}").read_bytes()
    check(
        "mutating the source cannot change the output",
        digest_bytes(after_bytes) == before,
    )

    print("section: encoder-absent red arm (simulated missing wheel)")
    shim = tmp / "shim"
    shim.mkdir(exist_ok=True)
    (shim / "av.py").write_text('raise ImportError("simulated absent media extra")\n')
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{shim}{os.pathsep}{ROOT / 'src'}"
    probe_absent = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import tempfile\n"
            "from cozy_runtime.author._services import Attempt, Outputs\n"
            "from cozy_runtime.author._errors import OutputError\n"
            "a = Attempt('probe', Path(tempfile.mkdtemp()))\n"
            "try:\n"
            "    Outputs(a).save_audio(None, sample_rate=1)\n"
            "except OutputError as e:\n"
            "    print('code=' + e.code)\n",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    check(
        "absent wheel is the typed refusal",
        "code=encoder_unavailable" in probe_absent.stdout,
        probe_absent.stdout.strip() or probe_absent.stderr.strip()[-120:],
    )

    print("section: H3-shaped benchmark (scaled)")
    bench = video_fixture(16, 768, 1344)
    baud = tone(16 / 24, rate)
    r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    b0 = time.perf_counter()
    attempt, out = fresh(tmp)
    basset = out.save_video(bench, fps=24.0, audio=baud, sample_rate=rate)
    wall = time.perf_counter() - b0
    r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    per_frame = wall / 16 * 1000
    check(
        "H3-frame-size encode benchmark",
        basset.size_bytes > 0,
        f"16 frames 1344x768: {wall:.2f} s ({per_frame:.0f} ms/frame, "
        f"H3's 124 frames ~{per_frame * 124 / 1000:.1f} s), "
        f"output {basset.size_bytes // 1024} KiB, peak RSS +{(r1 - r0) // 1024} MiB",
    )

    total = time.perf_counter() - t0
    print(f"\nmedia: {PASS} ok, {FAIL} failed, {total:.1f} s total")
    return 1 if FAIL else 0


def digest_only() -> int:
    """One fresh-process encode of the determinism fixture; prints the digest."""
    tmp = Path(tempfile.mkdtemp(prefix="media-digest-"))
    _, out = fresh(tmp)
    asset = out.save_video(
        video_fixture(16, 96, 128), fps=24.0, audio=tone(16 / 24, 44100), sample_rate=44100
    )
    print(asset.digest)
    return 0


if __name__ == "__main__":
    sys.exit(digest_only() if "--digest-only" in sys.argv else main())
