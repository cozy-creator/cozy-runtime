#!/usr/bin/env python3
"""cr-029: real 2-vs-8-shot streaming encode and memory proof.

The child path creates one real H.264 fixture, repeatedly decodes it through
``MediaDecoder.stream_video``, rebases immutable events, and commits one fragmented MP4
through ``Outputs.save_video_stream``. The parent compares fresh processes so peak RSS is
not inherited from an earlier arm. ``--snapshot`` is the poison control: it deliberately
retains every decoded RGB event before encoding and must move memory with shot count.
``--containment`` additionally composes PR #10's ProcessTreeScope reclaim with worker
partial cleanup; run it only where the complete same-UID process census is readable.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cozy_runtime.author import (  # noqa: E402
    DecodedMediaEvent,
    DecodedMediaHeader,
    DecodedVideoFormat,
    DecodedVideoFrame,
    MediaDecoder,
)
from cozy_runtime.author._assets import VideoAsset, bind  # noqa: E402
from cozy_runtime.author._services import Attempt, Outputs  # noqa: E402
from cozy_runtime.internal import proctree, spawn  # noqa: E402
from cozy_runtime.internal.child_env import EXECUTOR_SCOPE_ENV  # noqa: E402
from cozy_runtime.internal.worker import grants  # noqa: E402

WIDTH = 512
HEIGHT = 288
FRAMES = 24
RATE = Fraction(24, 1)
STREAMING_MARGIN_KIB = 48 << 10
SNAPSHOT_RED_KIB = 48 << 10


class NdLike:
    def __init__(self, data: bytes, shape: tuple[int, ...], dtype: str) -> None:
        self.data = data
        self.shape = shape
        self.dtype = dtype

    def tobytes(self) -> bytes:
        return self.data


def fixture(out: Outputs) -> VideoAsset:
    frame = bytes((index * 17) % 256 for index in range(WIDTH * HEIGHT * 3))
    asset = out.save_video(
        NdLike(frame * FRAMES, (FRAMES, HEIGHT, WIDTH, 3), "uint8"),
        fps=float(RATE),
    )
    assert asset._local is not None
    bind(
        asset,
        local=asset._local,
        attempt=out._attempt.request_id,
        media_type=asset.media_type,
        digest=asset.digest,
        length=asset.size_bytes,
        max_decoded_bytes=1 << 20,
        input_id="shot",
    )
    return asset


def composed(
    decoder: MediaDecoder, asset: VideoAsset, shots: int, debug_rss: bool = False
) -> Iterator[DecodedMediaEvent]:
    output_index = 0
    for shot in range(shots):
        with decoder.stream_video(asset) as stream:
            source_header = next(stream)
            assert isinstance(source_header, DecodedMediaHeader)
            assert source_header.video is not None and source_header.audio is None
            if shot == 0:
                yield DecodedMediaHeader(
                    video=DecodedVideoFormat(
                        width=source_header.video.width,
                        height=source_header.video.height,
                        time_base=1 / RATE,
                        pixel_aspect_ratio=source_header.video.pixel_aspect_ratio,
                        nominal_frame_rate=RATE,
                        color_primaries=source_header.video.color_primaries,
                        color_transfer=source_header.video.color_transfer,
                        color_matrix=source_header.video.color_matrix,
                        color_range=source_header.video.color_range,
                    ),
                    audio=None,
                )
            source_index = 0
            for event in stream:
                assert isinstance(event, DecodedVideoFrame)
                if shot and source_index == 0:
                    source_index += 1
                    continue
                yield DecodedVideoFrame(
                    width=event.width,
                    height=event.height,
                    rgb=event.rgb,
                    pts=output_index,
                    duration=1,
                    time_base=1 / RATE,
                    pixel_aspect_ratio=event.pixel_aspect_ratio,
                    color_primaries=event.color_primaries,
                    color_transfer=event.color_transfer,
                    color_matrix=event.color_matrix,
                    color_range=event.color_range,
                )
                output_index += 1
                source_index += 1
        if debug_rss:
            print(
                f"shot {shot + 1}: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} KiB",
                file=sys.stderr,
            )


def child(shots: int, snapshot: bool, debug_rss: bool = False) -> int:
    with tempfile.TemporaryDirectory(prefix="cozy-media-stream-") as directory:
        root = Path(directory)
        attempt = Attempt("media-stream", root, max_output_bytes=256 << 20)
        out = Outputs(attempt)
        decoder = MediaDecoder(attempt, active=lambda: False)
        source = fixture(out)
        events: Any = composed(decoder, source, shots, debug_rss)
        if snapshot:
            events = list(events)
        saved = out.save_video_stream(iter(events))
        expected_frames = shots * FRAMES - (shots - 1)
        if saved.frame_count != expected_frames:
            raise RuntimeError(f"frames {saved.frame_count}, expected {expected_frames}")
        print(
            json.dumps(
                {
                    "shots": shots,
                    "snapshot": snapshot,
                    "frames": saved.frame_count,
                    "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "bytes": saved.video.size_bytes,
                    "digest": saved.video.digest,
                },
                sort_keys=True,
            )
        )
    return 0


def kill_child(root: Path) -> int:
    attempt = Attempt("media-stream-kill", root, max_output_bytes=256 << 20)
    out = Outputs(attempt)
    rgb = bytes((index * 19) % 256 for index in range(WIDTH * HEIGHT * 3))

    def events() -> Iterator[DecodedMediaEvent]:
        yield DecodedMediaHeader(
            video=DecodedVideoFormat(
                width=WIDTH,
                height=HEIGHT,
                time_base=1 / RATE,
                pixel_aspect_ratio=Fraction(1, 1),
                nominal_frame_rate=RATE,
                color_primaries=6,
                color_transfer=6,
                color_matrix=5,
                color_range=1,
            ),
            audio=None,
        )
        for index in range(3600):
            yield DecodedVideoFrame(
                width=WIDTH,
                height=HEIGHT,
                rgb=rgb,
                pts=index,
                duration=1,
                time_base=1 / RATE,
                pixel_aspect_ratio=Fraction(1, 1),
                color_primaries=6,
                color_transfer=6,
                color_matrix=5,
                color_range=1,
            )
            time.sleep(0.005)

    out.save_video_stream(events())
    return 0


def killed_midmux() -> None:
    with tempfile.TemporaryDirectory(prefix="cozy-media-kill-") as directory:
        root = Path(directory)
        proctree.arm_subreaper()
        scope = proctree.ProcessTreeScope(secrets.token_hex(32))
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        process = spawn.spawn_executor(
            python=sys.executable,
            module_argv=[__file__, "--kill-child", str(root)],
            base_env=tuple(sorted(environment.items())),
            imposed={EXECUTOR_SCOPE_ENV: scope.token},
            cwd=str(root),
            scope=scope,
        )
        leader = proctree.process_identity(process.pid)
        proctree.wait_process_stopped(leader, seconds=10)
        proctree.signal_process(leader, signal.SIGCONT)
        deadline = time.monotonic() + 10
        partials: list[Path] = []
        while time.monotonic() < deadline and proctree.same_process(leader):
            partials = list(root.glob(".video-*.partial"))
            if partials and partials[0].stat().st_size > 0:
                break
            time.sleep(0.02)
        if not partials:
            with contextlib.suppress(BaseException):
                proctree.signal_process(leader, signal.SIGKILL)
            raise RuntimeError("streaming child created no partial before its deadline")
        processes = proctree.freeze_process_tree(
            scope, None, set(), worker_pid=os.getpid(), seconds=2.0
        )
        proctree.kill_process_tree(processes)
        for owned in processes:
            wait_deadline = time.monotonic() + 5
            while proctree.process_state(owned) is not None:
                proctree.reap_child(owned)
                if time.monotonic() >= wait_deadline:
                    raise RuntimeError(f"contained media child survived kill: {owned}")
                time.sleep(0.01)
        removed = grants.abort_outputs(root)
        if removed < 1 or list(root.rglob("*")):
            raise RuntimeError("worker abort did not remove the killed encoder partial")
        print(
            "killed mid-mux: ProcessTreeScope reclaimed the child, then worker abort "
            "removed its partial"
        )


def run_child(shots: int, snapshot: bool = False, debug_rss: bool = False) -> dict[str, Any]:
    command = [sys.executable, __file__, "--child", str(shots)]
    if snapshot:
        command.append("--snapshot")
    if debug_rss:
        command.append("--debug-rss")
    done = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    return cast(dict[str, Any], json.loads(done.stdout.splitlines()[-1]))


def parent(*, containment: bool, debug_rss: bool = False) -> int:
    if containment:
        killed_midmux()
    two = run_child(2, debug_rss=debug_rss)
    eight = run_child(8, debug_rss=debug_rss)
    poison = run_child(8, snapshot=True, debug_rss=debug_rss)
    streaming_growth = int(eight["maxrss_kib"]) - int(two["maxrss_kib"])
    poison_growth = int(poison["maxrss_kib"]) - int(eight["maxrss_kib"])
    print(json.dumps({"two": two, "eight": eight, "poison": poison}, indent=2, sort_keys=True))
    print(
        f"streaming 2->8 peak delta {streaming_growth / 1024:.1f} MiB; "
        f"snapshot poison adds {poison_growth / 1024:.1f} MiB"
    )
    if streaming_growth >= STREAMING_MARGIN_KIB:
        raise SystemExit("streaming RSS grew with accumulated shot payload")
    if poison_growth <= SNAPSHOT_RED_KIB:
        raise SystemExit("snapshot poison did not make the memory gate go red")
    print("media stream: 2-shot and 8-shot memory is codec/event bounded; poison went red")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=int)
    parser.add_argument("--containment", action="store_true")
    parser.add_argument("--kill-child", type=Path)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--debug-rss", action="store_true", help="print per-shot peak RSS")
    args = parser.parse_args()
    if args.kill_child is not None:
        return kill_child(args.kill_child)
    if args.child:
        return child(args.child, args.snapshot, args.debug_rss)
    return parent(containment=args.containment, debug_rss=args.debug_rss)


if __name__ == "__main__":
    raise SystemExit(main())
