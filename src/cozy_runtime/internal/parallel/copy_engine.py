"""Ulysses' all-to-all on the GPUs' copy engines, peer to peer or through pinned host memory.

NCCL moves an exchange with kernels on the SMs, which across a CPU root complex are an order of
magnitude slower than the copy engines (H100 NVL, 4 GPUs: 3.3 GB/s against 20.8 GB/s per GPU).
Without peer access its shared-memory transport has those kernels write and read host memory
across PCIe (2x RTX 4090, cl-816: an H3 step took 15 s at degree 2 against 9 s at degree 1,
moving 27 GB a GPU each way). So a group times NCCL against the copy-engine exchanges its links
allow as it forms (`calibrate`) and keeps the fastest; the result is the same bytes either way.

- `CopyEngineExchange` (every pair peer-accessible): each GPU maps one receive buffer of every
  other (torch symmetric memory). A round is a barrier, one copy-engine copy per peer into that
  peer's buffer (each on its own stream, so the NVLink and PCIe engines run at once), a second
  barrier, and the local copies out.
- `HostStagedExchange` (any pair not direct): each GPU's inbox is pinned host memory every GPU of
  the group maps, made from varena's host number (TensorFS `Pages.staging`: host RAM is varena's,
  REVIEW #3). A round writes a GPU's rows for each peer into that peer's inbox (D2H), a barrier,
  and reads its own inbox back (H2D), in two halves, so one round's reads overlap the next round's
  writes on the other engine.

Mapping happens when the group forms, never inside a call. Barriers are NCCL collectives of the
group, so a GPU that fails between them ends its peers' wait when the group interrupts
(`RankGroup.interrupt`) instead of leaving them spinning (run 13108). `start` runs an exchange on
the exchange's own streams, ordered after the caller's work so far, and returns the event that
ends it: Ulysses overlaps it with the attention of the head group before (`ulysses.exchange`).
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from cozy_runtime.graphs import streams
from cozy_runtime.internal.config import short_socket_directory

#: The exchange the group formed with, by process group; absent: NCCL.
_BY_GROUP: dict[int, Exchange] = {}
#: Each GPU's receive buffer: an H3 step's Q/K/V crossing (~0.8 GB per GPU at 97k rows) takes
#: four rounds, each long enough that its two barriers cost a few percent.
BUFFER_BYTES = 256 << 20
#: Each GPU's host inbox, both halves: a region per source GPU in each.
INBOX_BYTES = 128 << 20
#: Exchange streams outrank compute, so a barrier's one block is never queued behind a wave.
_PRIORITY = -1


def for_group(handle: Any) -> Exchange | None:
    return _BY_GROUP.get(id(handle))


def install(exchange: Exchange | None, handle: Any) -> None:
    _drop(keep=exchange)
    if exchange is not None:
        _BY_GROUP[id(handle)] = exchange


def _drop(keep: Exchange | None = None) -> None:
    for old in _BY_GROUP.values():
        if old is not keep:
            old.close()
    _BY_GROUP.clear()


class Exchange:
    """All-to-all over dimension 0 for a list of tensors at once, by an agreed send matrix:
    `sends[t]` is (rows, ...) contiguous; this GPU sends `matrix[rank][p]` rows of each to GPU p,
    in p order, and receives `matrix[s][rank]` from each s, stacked in s order. Every GPU passes
    the same matrix and tensors of the same row shapes, so every GPU takes the same rounds."""

    name = ""
    torch: Any
    rank: int
    world: int
    device: Any

    def start(
        self,
        sends: Sequence[Any],
        matrix: Sequence[Sequence[int]],
        out: Sequence[Any],
        chunk: int = 0,
    ) -> Any:
        """The crossing on this exchange's streams after the caller's work so far; its end event.
        `chunk` caps each round's rows per pair (0: as many as a buffer holds)."""
        raise NotImplementedError

    def close(self) -> None:
        return None

    def all_to_all(
        self, sends: Sequence[Any], matrix: Sequence[Sequence[int]], chunk: int = 0
    ) -> list[Any]:
        out = [t.new_empty((sum(row[self.rank] for row in matrix), *t.shape[1:])) for t in sends]
        self.all_to_all_into(sends, matrix, out, chunk)
        return out

    def all_to_all_into(
        self,
        sends: Sequence[Any],
        matrix: Sequence[Sequence[int]],
        out: Sequence[Any],
        chunk: int = 0,
    ) -> None:
        """The same crossing into actual preallocated receivers, on the current stream's order."""
        done = self.start(sends, matrix, out, chunk)
        self.torch.cuda.current_stream(self.device).wait_event(done)

    def _check(
        self, sends: Sequence[Any], matrix: Sequence[Sequence[int]], out: Sequence[Any]
    ) -> list[int]:
        if len(out) != len(sends):
            raise ValueError("exchange receiver count differs from its sends")
        for tensor, target in zip(sends, out, strict=True):
            if (
                not tensor.is_contiguous()
                or not target.is_contiguous()
                or target.shape != (sum(row[self.rank] for row in matrix), *tensor.shape[1:])
                or target.dtype != tensor.dtype
                or target.device != tensor.device
            ):
                raise ValueError("exchange receiver differs from its declared layout")
        return [tensor.element_size() * math.prod(tensor.shape[1:]) for tensor in sends]


class CopyEngineExchange(Exchange):
    """Peer to peer through symmetric memory. Allocates its buffer here (which may fail on one
    GPU: the caller agrees before `map`)."""

    name = "copy-engine"

    def __init__(self, torch: Any, symm: Any, pg: Any, rank: int, world: int) -> None:
        self.torch, self.symm, self.pg = torch, symm, pg
        self.rank, self.world = rank, world
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.streams = [streams.stream(torch, self.device) for _ in range(world)]
        self.side = streams.stream(torch, self.device, priority=_PRIORITY)
        self.buffer = symm.empty(BUFFER_BYTES, dtype=torch.uint8, device=self.device)
        self.flag = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.handle: Any = None
        self.peers: list[Any] = []

    def map(self) -> None:
        """Collective, once per formation: every GPU maps every other's buffer."""
        with short_socket_directory():
            self.handle = self.symm.rendezvous(self.buffer, self.pg.group_name)
        shape = (self.buffer.numel(),)
        self.peers = [self.handle.get_buffer(p, shape, self.torch.uint8) for p in range(self.world)]

    def _barrier(self) -> None:
        """Every GPU's stream reached here: an abortable NCCL collective, never a spin."""
        self.torch.distributed.all_reduce(self.flag, group=self.pg)

    def start(
        self,
        sends: Sequence[Any],
        matrix: Sequence[Sequence[int]],
        out: Sequence[Any],
        chunk: int = 0,
    ) -> Any:
        torch = self.torch
        widths = self._check(sends, matrix, out)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        self.side.wait_event(ready)
        with torch.cuda.stream(self.side):
            self._rounds(sends, matrix, out, widths, chunk)
            done = torch.cuda.Event()
            done.record(self.side)
        return done

    def _rounds(
        self,
        sends: Sequence[Any],
        matrix: Sequence[Sequence[int]],
        out: Sequence[Any],
        widths: Sequence[int],
        chunk: int,
    ) -> None:
        torch, world, rank = self.torch, self.world, self.rank
        chunk = min(chunk or self.buffer.numel(), self.buffer.numel() // (world * sum(widths)))
        if chunk < 1:
            raise ValueError(f"one row of {sum(widths)} B per GPU exceeds the exchange buffer")
        firsts = [sum(matrix[s][rank] for s in range(q)) for q in range(world)]
        current = torch.cuda.current_stream(self.device)
        for start in range(0, max(max(row) for row in matrix), chunk):
            counts = [[min(max(n - start, 0), chunk) for n in row] for row in matrix]
            self._barrier()
            issued = torch.cuda.Event()
            issued.record(current)
            # The receive buffers are scratch: their copies bypass Python dispatch, so the
            # memory's catch never takes a block that writes them for one that writes its inputs.
            with torch._C._DisableTorchDispatch():
                self._issue(sends, matrix, counts, start, widths, issued, current)
            self._barrier()
            base = 0
            for tensor, width, into in zip(sends, widths, out, strict=True):
                for s in range(world):
                    rows, first = counts[s][rank], firsts[s] + start
                    region = self.buffer[base : base + rows * width].view(tensor.dtype)
                    into[first : first + rows].copy_(region.view(rows, *tensor.shape[1:]))
                    base += rows * width

    def _issue(
        self,
        sends: Sequence[Any],
        matrix: Sequence[Sequence[int]],
        counts: Sequence[Sequence[int]],
        start: int,
        widths: Sequence[int],
        issued: Any,
        current: Any,
    ) -> None:
        """One copy-engine copy per tensor per peer for this round, each peer on its own stream."""
        torch, world, rank = self.torch, self.world, self.rank
        for step in range(world):
            peer = (rank + step) % world
            rows = counts[rank][peer]
            sent = sum(matrix[rank][:peer]) + start
            offset = sum(counts[s][peer] for s in range(rank))
            landing = sum(counts[s][peer] for s in range(world))
            stream = self.streams[peer]
            stream.wait_event(issued)
            with torch.cuda.stream(stream):
                base = 0
                for tensor, width in zip(sends, widths, strict=True):
                    if rows:
                        source = tensor[sent : sent + rows].reshape(-1).view(torch.uint8)
                        begin = base + offset * width
                        self.peers[peer][begin : begin + rows * width].copy_(
                            source, non_blocking=True
                        )
                    base += landing * width
            current.wait_stream(stream)


class HostStagedExchange(Exchange):
    """Through pinned host memory every GPU of the group maps. Each GPU's inbox comes from its
    host tier (`pages`): varena's host number holds it until the exchange closes, and no other
    executor maps or counts it twice (TensorFS `Staging`). Its inbox holds two halves of one region
    per source GPU. A round on the write stream waits for its half to be free, copies this GPU's
    rows for each peer into that peer's region for it; the read stream then barriers (every GPU's
    writes of the round landed, and its reads of the round before ended), marks the other half
    free, and copies its own inbox's regions out."""

    name = "host-staged"

    def __init__(self, torch: Any, pg: Any, rank: int, world: int, pages: Any) -> None:
        self.torch, self.pg, self.rank, self.world = torch, pg, rank, world
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.slot = INBOX_BYTES // (2 * world) // 4096 * 4096
        self.size = 2 * world * self.slot
        self.writes = streams.stream(torch, self.device, priority=_PRIORITY)
        self.reads = streams.stream(torch, self.device, priority=_PRIORITY)
        self.flag = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.half = 0
        self.inboxes: list[Any] = []
        self.peers: list[Any] = []
        self.own: Any = pages.staging(self.size) if pages is not None else None
        if self.own is None:
            raise MemoryError("varena's host number has no room for this GPU's exchange inbox")

    def map(self, store: Any, agree: Callable[[bool], bool]) -> bool:
        """Collective, once per formation: every GPU maps every inbox for its own; each unlinks
        its inbox's name once all agreed. False on every GPU alike when any could not."""
        from tensorfs.varena import _runtime as varena

        store.set(f"cozy.staged/{self.rank}", self.own.name)
        names = [
            bytes(name).decode()
            for name in store.multi_get([f"cozy.staged/{r}" for r in range(self.world)])
        ]
        ok = True
        try:
            for rank, name in enumerate(names):
                staging = self.own
                if rank != self.rank:
                    staging = varena.Staging.open(self.device.index, name, self.size)
                    self.peers.append(staging)
                self.inboxes.append(_host(self.torch, staging))
        except Exception:
            ok = False
        agreed = agree(ok)
        self.own.unlink()
        if not agreed:
            self.close()
            return False
        self.free = [self.torch.cuda.Event(), self.torch.cuda.Event()]
        for event in self.free:
            event.record(self.reads)
        return True

    def close(self) -> None:
        if self.inboxes:
            self.writes.synchronize()
            self.reads.synchronize()
        self.inboxes = []
        for staging in self.peers:
            staging.close()
        self.peers = []
        if self.own is not None:
            self.own.close()
            self.own = None

    def _barrier(self) -> None:
        self.torch.distributed.all_reduce(self.flag, group=self.pg)

    def _region(self, inbox: Any, half: int, source: int) -> Any:
        at = (half * self.world + source) * self.slot
        return inbox[at : at + self.slot]

    def start(
        self,
        sends: Sequence[Any],
        matrix: Sequence[Sequence[int]],
        out: Sequence[Any],
        chunk: int = 0,
    ) -> Any:
        torch, world, rank = self.torch, self.world, self.rank
        widths = self._check(sends, matrix, out)
        chunk = min(chunk or self.slot, self.slot // sum(widths))
        if chunk < 1:
            raise ValueError(f"one row of {sum(widths)} B per GPU exceeds the exchange inbox")
        firsts = [sum(matrix[s][rank] for s in range(q)) for q in range(world)]
        sent = [sum(matrix[rank][:p]) for p in range(world)]
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        self.writes.wait_event(ready)
        self.reads.wait_event(ready)
        with torch.cuda.stream(self.reads):
            rows = matrix[rank][rank]
            for tensor, into in zip(sends, out, strict=True):
                if rows:
                    into[firsts[rank] : firsts[rank] + rows].copy_(
                        tensor[sent[rank] : sent[rank] + rows]
                    )
        for start in range(0, max(max(row) for row in matrix), chunk):
            counts = [[min(max(n - start, 0), chunk) for n in row] for row in matrix]
            half, self.half = self.half, self.half ^ 1
            self.writes.wait_event(self.free[half])
            # The inboxes are scratch: their writes bypass Python dispatch (`CopyEngineExchange`).
            with torch.cuda.stream(self.writes), torch._C._DisableTorchDispatch():
                for peer in range(world):
                    if peer != rank:
                        region, base = self._region(self.inboxes[peer], half, rank), 0
                        n = counts[rank][peer]
                        for tensor, width in zip(sends, widths, strict=True):
                            if n:
                                source = tensor[sent[peer] + start : sent[peer] + start + n]
                                region[base : base + n * width].copy_(
                                    source.reshape(-1).view(torch.uint8), non_blocking=True
                                )
                            base += n * width
                written = torch.cuda.Event()
                written.record(self.writes)
            self.reads.wait_event(written)
            with torch.cuda.stream(self.reads):
                self._barrier()
                self.free[half ^ 1].record(self.reads)
                for source in range(world):
                    if source != rank:
                        region, base = self._region(self.inboxes[rank], half, source), 0
                        n, first = counts[source][rank], firsts[source] + start
                        for width, into in zip(widths, out, strict=True):
                            if n:
                                into[first : first + n].reshape(-1).view(torch.uint8).copy_(
                                    region[base : base + n * width], non_blocking=True
                                )
                            base += n * width
        done = torch.cuda.Event()
        done.record(self.reads)
        return done


def _host(torch: Any, staging: Any) -> Any:
    """`staging`'s bytes as a CPU uint8 tensor, valid while it is open."""
    return torch.frombuffer(
        (ctypes.c_uint8 * staging.bytes).from_address(staging.ptr), dtype=torch.uint8
    )


#: Bytes each GPU sends in the calibration: the size class of one Ulysses exchange.
_CALIBRATION_BYTES = 64 << 20


def calibrate(
    torch: Any,
    pg: Any,
    rank: int,
    world: int,
    candidates: Sequence[str] = ("copy-engine",),
    pages: Any = None,
) -> tuple[Exchange | None, dict[str, float]]:
    """Collective, on every GPU of a just-formed group: the same all-to-all through NCCL and
    through each candidate exchange, timed and checked against NCCL's bytes; every GPU keeps the
    fastest alike. A candidate any GPU cannot map is dropped on all of them. `pages`: this GPU's
    host tier, which a host-staged inbox comes from."""
    from torch.distributed.distributed_c10d import _get_process_group_store

    dist = torch.distributed

    def everyone(value: float, op: Any) -> float:
        agreed = torch.tensor([value], dtype=torch.float64, device="cuda")
        dist.all_reduce(agreed, op=op, group=pg)
        return float(agreed.item())

    def agree(ok: bool) -> bool:
        return everyone(1.0 if ok else 0.0, dist.ReduceOp.MIN) >= 1.0

    # A formation again (after a call ran short of memory) frees the last one's buffers first.
    _drop()
    rows = _CALIBRATION_BYTES // 2 // 1024 // world * world
    # Two rounds, the second short: every formation checks the rounds against NCCL's bytes.
    half = rows // world // 2 + 1
    send = torch.randn(rows, 1024, dtype=torch.bfloat16, device="cuda")
    received = torch.empty_like(send)
    matrix = [[rows // world] * world for _ in range(world)]
    sent = send.numel() * send.element_size() * (world - 1) / world

    def timed(run: Any) -> float:
        run()
        torch.cuda.synchronize()
        dist.barrier(group=pg)
        began, ended = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        began.record()
        for _ in range(3):
            run()
        ended.record()
        ended.synchronize()
        seconds = everyone(max(began.elapsed_time(ended) / 3000, 1e-9), dist.ReduceOp.MAX)
        return float(round(sent / seconds / 1e9, 2))

    timings: dict[str, float] = {
        "nccl_gbps": timed(lambda: dist.all_to_all_single(received, send, group=pg))
    }
    best: Exchange | None = None
    for name in candidates:
        label = name.replace("-", "_")
        exchange = _open(name, torch, pg, rank, world, agree, _get_process_group_store(pg), pages)
        timings[f"{label}_mapped"] = 0.0 if exchange is None else 1.0
        if exchange is None:
            continue
        try:
            timings[f"{label}_gbps"] = timed(partial(exchange.all_to_all, [send], matrix, half))
            exact = torch.equal(received, exchange.all_to_all([send], matrix, half)[0])
        except Exception:  # the whole group then drops it, every GPU alike
            exact = False
            timings[f"{label}_gbps"] = 0.0
        timings[f"{label}_exact"] = 1.0 if agree(exact) else 0.0
        floor = (
            timings["nccl_gbps"] if best is None else timings[f"{best.name.replace('-', '_')}_gbps"]
        )
        if timings[f"{label}_exact"] < 1.0 or timings[f"{label}_gbps"] <= floor:
            exchange.close()
            continue
        if best is not None:
            best.close()
        best = exchange
    torch.cuda.synchronize()
    return best, timings


def _open(
    name: str,
    torch: Any,
    pg: Any,
    rank: int,
    world: int,
    agree: Callable[[bool], bool],
    store: Any,
    pages: Any,
) -> Exchange | None:
    """`name`'s exchange mapped on every GPU of the group, or None on every one alike."""
    made: Exchange | None = None
    try:
        if name == "copy-engine":
            from torch.distributed import _symmetric_memory as symm

            made = CopyEngineExchange(torch, symm, pg, rank, world)
        elif name == "host-staged":
            made = HostStagedExchange(torch, pg, rank, world, pages)
    except Exception:  # an unsupported driver, no host room, no shared memory: NCCL stays
        made = None
    if not agree(made is not None):
        if made is not None:
            made.close()
        return None
    if isinstance(made, CopyEngineExchange):
        made.map()
        return made
    assert isinstance(made, HostStagedExchange)
    prefixed = torch.distributed.PrefixStore(f"cozy.exchange/{pg.group_name}/", store)
    return made if made.map(prefixed, agree) else None
