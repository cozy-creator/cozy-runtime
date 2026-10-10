"""The measured links between a group's GPUs, and the route its Ulysses exchange takes.

A pair of GPUs can be joined by NVLink, by PCIe through a switch or through the CPU's root
complex, or not at all for peer traffic. What a topology table reports says little about
speed (H100 NVL pairs: NVLink 261 GB/s, peer copies across the host bridge 28 GB/s, staged
through host memory 11 GB/s, NCCL's peer-memory kernels across that bridge 3.3 GB/s), so
nothing here is inferred: once per machine boot a child process copies a buffer over every
ordered pair, directly (copy engines, verified) and staged through pinned host memory, and
the group forms over the result. The child owns those device contexts, not the executor.

A copy within one process proves little about the paths between processes: on one A5000
host every pair copied directly, yet NCCL's peer path between follower processes failed
(`ncclProxyClientGetFd`). So when every pair is direct, the probe then forms a group of its
own processes, as an executor forms one, and runs NCCL over peer memory and a copy-engine
exchange through symmetric memory; a group that errs or stops making progress keeps the
conservative route.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import selectors
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from cozy_runtime.internal import spawn
from cozy_runtime.internal.config import inherited_environment
from cozy_runtime.internal.parallel import copy_engine, group

#: Bytes each pair copies per measurement: large enough that link setup and launch latency
#: do not decide it, small enough that a whole group measures in about a second.
_BYTES = 64 << 20
_CHUNK = 8 << 20
_REPEATS = 3
_FOLLOWER = ("-m", "cozy_runtime.internal.parallel.links", "--follower")


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
    """Every ordered pair of a group's GPUs, what a group of processes over them managed, and
    the NCCL setting the group forms under."""

    pairs: tuple[Pair, ...] = ()
    #: the process probe: `nccl_peer`, `copy_engines` (bools), `detail` and timings
    processes: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def all_direct(self) -> bool:
        return bool(self.pairs) and all(pair.direct for pair in self.pairs)

    @property
    def all_peers(self) -> bool:
        """Every pair reaches the other's memory and its direct copy lands the right bytes."""
        return bool(self.pairs) and all(pair.peer and pair.verified for pair in self.pairs)

    @property
    def peer_route(self) -> bool:
        return self.all_direct and bool(self.processes.get("nccl_peer"))

    @property
    def copy_engines(self) -> bool:
        """Copy engines may carry the exchange where every pair copies correctly and a group of
        processes mapped each other's memory; the calibration then times them against NCCL."""
        return self.all_peers and bool(self.processes.get("copy_engines"))

    def nccl(self) -> dict[str, str]:
        """NCCL's peer memory where every pair measured direct-and-faster and a group of
        processes formed over it, else only over NVLink, with host memory between the rest."""
        return {"NCCL_P2P_LEVEL": "SYS" if self.peer_route else "NVL"}

    def why(self, number: Mapping[int, int] | None = None) -> str:
        """Why the group takes its route; GPUs by `number` (position -> nvidia-smi number)."""
        if self.reason:
            return self.reason
        slow = [pair for pair in self.pairs if not pair.direct]
        if slow and self.copy_engines:
            pair = slow[0]
            return (
                f"GPU {(number or {}).get(pair.src, pair.src)} to GPU "
                f"{(number or {}).get(pair.dst, pair.dst)} copies directly at "
                f"{pair.direct_gbps:.1f} GB/s, through host memory at {pair.staged_gbps:.1f}: "
                "NCCL keeps peer memory on NVLink; copy engines are timed against it"
            )
        if not slow:
            if not self.peer_route:
                return f"peer memory between processes failed: {self.processes.get('detail', '')}"
            if not self.copy_engines:
                return f"copy engines between processes failed: {self.processes.get('detail', '')}"
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
        return {
            "pairs": [asdict(pair) for pair in self.pairs],
            "processes": dict(self.processes),
            "reason": self.why(),
        }

    @classmethod
    def read(cls, document: Mapping[str, Any]) -> Links:
        pairs = tuple(Pair(**row) for row in document.get("pairs", ()))
        return cls(pairs, dict(document.get("processes") or {}), "")


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


#: What the probe measures; a probe that measures more is not answered by an older one's file.
_PROBE = "links/2"


def _key(devices: str) -> str:
    """This boot, driver, device list and probe: the links do not change while all hold."""
    parts = [_PROBE, devices]
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
    """Run the probe child, reading its lines as they come. It may wedge only in the process
    group (inside NCCL); that phase gets four times what the child took to start and measure
    its first pair (torch import, CUDA, a copy), and is ended as a whole if it says nothing."""
    environment = {**inherited_environment(), "CUDA_VISIBLE_DEVICES": devices}
    began = time.monotonic()
    child = subprocess.Popen(
        [sys.executable, "-I", "-m", "cozy_runtime.internal.parallel.links", str(count)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert child.stdout is not None
    pairs: list[Pair] = []
    processes: dict[str, Any] = {}
    window: float | None = None
    watch = selectors.DefaultSelector()
    watch.register(child.stdout, selectors.EVENT_READ)
    while child.poll() is None or watch.select(0):
        ready = watch.select(None if window is None else max(window - time.monotonic(), 0))
        if not ready:
            os.killpg(child.pid, signal.SIGKILL)
            processes = {
                "nccl_peer": False,
                "copy_engines": False,
                "detail": "a group of probe processes stopped making progress in NCCL",
            }
            break
        line = child.stdout.readline()
        if not line:
            break
        row = json.loads(line) if line.startswith("{") else {}
        if "group" in row:
            processes = row["group"]
        elif row:
            pairs.append(Pair(**row))
            if len(pairs) == 1:
                started = time.monotonic() - began
            if len(pairs) == count * (count - 1):
                window = time.monotonic() + 4 * started
    child.wait()
    stderr = child.stderr.read() if child.stderr else ""
    if len(pairs) != count * (count - 1):
        tail = stderr.strip().splitlines()[-1:] or [f"exit {child.returncode}"]
        broken = next((pair for pair in pairs if not pair.verified and pair.peer), None)
        reason = (
            f"GPU {broken.src} to GPU {broken.dst}: {broken.detail}"
            if broken
            else f"the link probe ended early: {tail[0][:200]}"
        )
        return Links(tuple(pairs) if broken else (), {}, reason)
    return Links(tuple(pairs), processes)


def _group(torch: Any, rank: int, world: int, port: int, level: str) -> dict[str, Any]:
    """One process of the probe's own group: NCCL over peer memory, then the copy engines
    through symmetric memory, each checked against the other's bytes."""
    from torch.distributed import _symmetric_memory as symm

    dist = torch.distributed
    group.route({"NCCL_P2P_LEVEL": level})
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    store = dist.TCPStore("127.0.0.1", port, world, is_master=rank == 0)
    dist.init_process_group("nccl", store=store, rank=rank, world_size=world, device_id=device)
    report: dict[str, Any] = {"nccl_peer": False, "copy_engines": False, "detail": ""}
    try:
        rows = (_CHUNK // 2 // 1024) // world * world
        send = torch.full((rows, 1024), float(rank + 1), dtype=torch.bfloat16, device=device)
        received = torch.empty_like(send)
        began = time.perf_counter()
        dist.all_to_all_single(received, send)
        torch.cuda.synchronize()
        expected = torch.arange(1, world + 1, dtype=torch.bfloat16, device=device)
        expected = expected.repeat_interleave(rows // world)[:, None].expand(rows, 1024)
        report["nccl_peer"] = bool(torch.equal(received, expected))
        report["nccl_ms"] = round((time.perf_counter() - began) * 1000, 1)
        exchange = copy_engine.CopyEngineExchange(torch, symm, dist.group.WORLD, rank, world)
        matrix = [[rows // world] * world for _ in range(world)]
        report["copy_engines"] = bool(torch.equal(exchange.all_to_all([send], matrix)[0], expected))
    except Exception as exc:  # its first line is why the group keeps the conservative route
        report["detail"] = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:300]}"
    agreed = torch.tensor([report["nccl_peer"], report["copy_engines"]], dtype=torch.int32).to(
        device
    )
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    report["nccl_peer"], report["copy_engines"] = (bool(value) for value in agreed.tolist())
    dist.destroy_process_group()
    return report


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

    if argv[0] == "--follower":
        rank, world, port = (int(value) for value in argv[1:4])
        _group(torch, rank, world, port, argv[4])
        return 0
    count = int(argv[0])
    pairs: list[Pair] = []

    def emit(pair: Pair) -> None:
        pairs.append(pair)
        print(json.dumps(asdict(pair)), flush=True)

    _measure(torch, count, emit)
    if pairs and all(pair.peer and pair.verified for pair in pairs):
        # NCCL forms at the level the group would take; the copy engines are tried either way.
        level = "SYS" if all(pair.direct for pair in pairs) else "NVL"
        # Followers launch as an executor's do (its trampoline, its seal), as children.
        port = _free_port()
        seam, unused = os.pipe()
        followers = [
            spawn.spawn_follower(
                python=sys.executable,
                module_argv=[*_FOLLOWER, str(rank), str(count), str(port), level],
                env=inherited_environment(),
                inherit_fd=unused,
            )
            for rank in range(1, count)
        ]
        os.close(seam)
        print(json.dumps({"group": _group(torch, 0, count, port, level)}), flush=True)
        for follower in followers:
            follower.wait()
    return 0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
