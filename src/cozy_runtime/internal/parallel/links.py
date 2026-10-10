"""The measured links between a group's GPUs, and the route its Ulysses exchange takes.

A pair of GPUs can be joined by NVLink, by PCIe through a switch or through the CPU's root
complex, or not at all for peer traffic. What a topology table reports says little about
speed (H100 NVL pairs: NVLink 261 GB/s, peer copies across the host bridge 28 GB/s, staged
through host memory 11 GB/s, NCCL's peer-memory kernels across that bridge 3.3 GB/s), so
nothing here is inferred: once per machine boot a child process copies a buffer over every
ordered pair, directly (copy engines, verified) and staged through pinned host memory, and
the group forms over the result. The child owns those device contexts, not the executor.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

#: Bytes each pair copies per measurement: large enough that link setup and launch latency
#: do not decide it, small enough that a whole group measures in about a second.
_BYTES = 64 << 20
_CHUNK = 8 << 20
_REPEATS = 3


@dataclass(frozen=True, slots=True)
class Pair:
    """One ordered pair of group positions: whether `src` reaches `dst`'s memory directly, and
    how fast a verified direct copy and a pinned-host staged copy moved the same bytes."""

    src: int
    dst: int
    peer: bool
    direct_gbps: float
    staged_gbps: float
    verified: bool
    detail: str = ""

    @property
    def direct(self) -> bool:
        return self.peer and self.verified and self.direct_gbps > self.staged_gbps


@dataclass(frozen=True, slots=True)
class Links:
    """Every ordered pair of a group's GPUs, and the NCCL setting the group forms under."""

    pairs: tuple[Pair, ...] = ()
    reason: str = ""

    @property
    def all_direct(self) -> bool:
        return bool(self.pairs) and all(pair.direct for pair in self.pairs)

    def nccl(self) -> dict[str, str]:
        """NCCL's peer memory on every pair that measured direct-and-faster, else only over
        NVLink (where NCCL's own kernels are fastest) with host memory between the rest."""
        return {"NCCL_P2P_LEVEL": "SYS" if self.all_direct else "NVL"}

    def why(self, number: Mapping[int, int] | None = None) -> str:
        """Why the group takes its route; GPUs by `number` (position -> nvidia-smi number)."""
        if self.reason:
            return self.reason
        slow = [pair for pair in self.pairs if not pair.direct]
        if not slow:
            return "every GPU pair copies directly faster than through host memory"
        pair = slow[0]
        cause = (
            "has no peer access"
            if not pair.peer
            else pair.detail or "did not copy correctly"
            if not pair.verified
            else f"copies directly at {pair.direct_gbps:.1f} GB/s, through host memory at "
            f"{pair.staged_gbps:.1f} GB/s"
        )
        number = number or {}
        return (
            f"GPU {number.get(pair.src, pair.src)} to GPU {number.get(pair.dst, pair.dst)} {cause}"
        )

    def document(self) -> dict[str, Any]:
        return {"pairs": [asdict(pair) for pair in self.pairs], "reason": self.why()}

    @classmethod
    def read(cls, document: Mapping[str, Any]) -> Links:
        return cls(tuple(Pair(**row) for row in document.get("pairs", ())), "")


def _gbps(nbytes: int, seconds: float) -> float:
    return round(nbytes / seconds / 1e9, 2) if seconds > 0 else 0.0


def _measure(torch: Any, count: int, emit: Any) -> None:
    """In the probe process: every ordered pair, staged first (it bounds how long a direct
    copy may take before it can no longer win), then direct, each checked by a byte sum."""
    devices = [torch.device("cuda", index) for index in range(count)]
    sources = []
    for device in devices:
        generator = torch.Generator(device=device).manual_seed(device.index + 1)
        sources.append(
            torch.randint(0, 256, (_BYTES,), dtype=torch.uint8, device=device, generator=generator)
        )
    sums = [int(source.sum(dtype=torch.int64)) for source in sources]
    targets = [torch.zeros(_BYTES, dtype=torch.uint8, device=device) for device in devices]
    pinned = [torch.empty(_CHUNK, dtype=torch.uint8, pin_memory=True) for _ in range(2)]

    def staged(src: int, dst: int) -> None:
        out_stream, in_stream = torch.cuda.Stream(devices[src]), torch.cuda.Stream(devices[dst])
        landed = [torch.cuda.Event() for _ in pinned]
        drained = [torch.cuda.Event() for _ in pinned]
        for event in drained:
            event.record(in_stream)
        for index, start in enumerate(range(0, _BYTES, _CHUNK)):
            slot = index % len(pinned)
            out_stream.wait_event(drained[slot])
            with torch.cuda.stream(out_stream):
                pinned[slot].copy_(sources[src][start : start + _CHUNK], non_blocking=True)
                landed[slot].record(out_stream)
            in_stream.wait_event(landed[slot])
            with torch.cuda.stream(in_stream):
                targets[dst][start : start + _CHUNK].copy_(pinned[slot], non_blocking=True)
                drained[slot].record(in_stream)
        in_stream.synchronize()

    def timed(copy: Any, *devices_used: int) -> float:
        began = time.perf_counter()
        for _ in range(_REPEATS):
            copy()
        for index in devices_used:
            torch.cuda.synchronize(index)
        return (time.perf_counter() - began) / _REPEATS

    for src in range(count):
        for dst in range(count):
            if src == dst:
                continue
            staged(src, dst)
            staged_s = timed(partial(staged, src, dst), src, dst)
            peer = bool(torch.cuda.can_device_access_peer(src, dst))
            direct_s, verified, detail = 0.0, False, ""
            if peer:
                targets[dst].zero_()
                torch.cuda.synchronize(dst)
                done = torch.cuda.Event()
                began = time.perf_counter()
                with torch.cuda.device(src):
                    targets[dst].copy_(sources[src], non_blocking=True)
                    done.record()
                # A direct copy slower than the staged one has lost; one that has not finished
                # in many times that long is not waited for (a broken peer path can wedge).
                while not done.query():
                    if time.perf_counter() - began > 50 * staged_s + 1.0:
                        emit(
                            Pair(
                                src,
                                dst,
                                peer,
                                0.0,
                                _gbps(_BYTES, staged_s),
                                False,
                                "its direct copy did not finish",
                            )
                        )
                        os._exit(0)
                    time.sleep(0.0005)
                verified = int(targets[dst].sum(dtype=torch.int64)) == sums[src]
                if verified:
                    direct_s = timed(partial(targets[dst].copy_, sources[src], True), src, dst)
                else:
                    detail = "its direct copy landed different bytes"
            emit(
                Pair(
                    src,
                    dst,
                    peer,
                    _gbps(_BYTES, direct_s),
                    _gbps(_BYTES, staged_s),
                    verified,
                    detail,
                )
            )


def _key(devices: str) -> str:
    """This boot, driver and device list: the links do not change while all three hold."""
    parts = [devices]
    for path in ("/proc/sys/kernel/random/boot_id", "/proc/driver/nvidia/version"):
        try:
            parts.append(Path(path).read_text().strip())
        except OSError:
            parts.append("")
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:32]


def measured(directory: Path | None, devices: str, count: int) -> Links:
    """The group's links: kept for this boot under `directory`, else measured now in a child
    process. A probe that dies or wedges leaves its pairs unmeasured, and the group takes
    the conservative route, saying why."""
    if count < 2:
        return Links()
    path = None if directory is None else directory / f"{_key(devices)}-{count}.json"
    lock = None
    try:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = open(path.with_suffix(".lock"), "a+")  # noqa: SIM115
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                return Links.read(json.loads(path.read_text()))
            marker = path.with_suffix(".measuring")
            if marker.exists():
                return Links(reason="the link probe of this boot did not finish")
            marker.write_text(str(os.getpid()))
        links = _probe(devices, count)
        if path is not None and links.pairs:
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(links.document()))
            temporary.replace(path)
            path.with_suffix(".measuring").unlink(missing_ok=True)
        return links
    except OSError as exc:
        return Links(reason=f"the links could not be measured: {exc}")
    finally:
        if lock is not None:
            lock.close()


def _probe(devices: str, count: int) -> Links:
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": devices}
    child = subprocess.run(
        [sys.executable, "-I", "-m", "cozy_runtime.internal.parallel.links", str(count)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    pairs = []
    for line in child.stdout.splitlines():
        if line.startswith("{"):
            pairs.append(Pair(**json.loads(line)))
    if len(pairs) != count * (count - 1) or child.returncode != 0:
        tail = (child.stderr or "").strip().splitlines()[-1:] or [f"exit {child.returncode}"]
        broken = next((pair for pair in pairs if not pair.verified and pair.peer), None)
        reason = (
            f"GPU {broken.src} to GPU {broken.dst}: {broken.detail}"
            if broken
            else f"the link probe ended early: {tail[0][:200]}"
        )
        return Links(tuple(pairs) if broken else (), reason)
    return Links(tuple(pairs))


@dataclass(slots=True)
class Transport:
    """What one group formed over, as `cozy run show` reports it per call."""

    degree: int
    gpus: list[int]
    route: str
    nccl: dict[str, str]
    links: Links
    calibration: dict[str, float] = field(default_factory=dict)

    def document(self) -> dict[str, Any]:
        """GPUs by their nvidia-smi numbers; each pair's route is direct or through host memory."""
        number = dict(enumerate(self.gpus))
        return {
            "degree": self.degree,
            "gpus": self.gpus,
            "route": self.route,
            "nccl": dict(self.nccl),
            "reason": self.links.why(number),
            "pairs": [
                {
                    "a": number.get(pair.src, pair.src),
                    "b": number.get(pair.dst, pair.dst),
                    "peer": pair.peer,
                    "direct_gbps": pair.direct_gbps,
                    "staged_gbps": pair.staged_gbps,
                    "route": "direct" if pair.direct else "host",
                }
                for pair in self.links.pairs
            ],
            "calibration": dict(self.calibration),
        }


def main(argv: Sequence[str]) -> int:
    import torch

    count = int(argv[0])

    def emit(pair: Pair) -> None:
        print(json.dumps(asdict(pair)), flush=True)

    _measure(torch, count, emit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
