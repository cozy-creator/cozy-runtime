"""Ulysses' all-to-all on the GPUs' copy engines, through symmetric memory.

Every GPU of a group maps one receive buffer of every other (torch symmetric memory: CUDA
IPC/cuMem handles exchanged once over the group's store), when the group forms and never inside
a call. An exchange is a round per buffer's worth of rows: a barrier, one copy-engine copy per
peer into that peer's buffer (each on its own stream, so the NVLink and PCIe engines run at
once), a second barrier, and the local copies out. Both barriers are NCCL collectives of the
group, so a GPU that fails between them ends its peers' wait when the group interrupts
(`RankGroup.interrupt`) instead of leaving them spinning (run 13108: a buffer grown inside a
call left three GPUs in a rendezvous nothing could end). NCCL moves the same bytes with kernels
on the SMs, which across a CPU root complex are an order of magnitude slower than the copy
engines (H100 NVL, 4 GPUs: 3.3 GB/s against 20.8 GB/s per GPU). The group uses this only where
every pair measured direct and the calibration at formation measured it faster than NCCL; the
result is the same bytes either way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from cozy_runtime.internal import streams
from cozy_runtime.internal.config import short_socket_directory

#: The exchange the group formed with, by process group; absent: NCCL.
_BY_GROUP: dict[int, CopyEngineExchange] = {}
#: Each GPU's receive buffer: an H3 step's Q/K/V crossing (~0.8 GB per GPU at 97k rows) takes
#: four rounds, each long enough that its two barriers cost a few percent.
BUFFER_BYTES = 256 << 20


def for_group(handle: Any) -> CopyEngineExchange | None:
    return _BY_GROUP.get(id(handle))


def install(exchange: CopyEngineExchange | None, handle: Any) -> None:
    _BY_GROUP.clear()
    if exchange is not None:
        _BY_GROUP[id(handle)] = exchange


class CopyEngineExchange:
    """All-to-all over dimension 0 for a list of tensors at once, by an agreed send matrix.
    Allocates its buffer here (which may fail on one GPU: the caller agrees before `map`)."""

    def __init__(self, torch: Any, symm: Any, pg: Any, rank: int, world: int) -> None:
        self.torch, self.symm, self.pg = torch, symm, pg
        self.rank, self.world = rank, world
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.streams = [streams.stream(torch, self.device) for _ in range(world)]
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

    def all_to_all(
        self, sends: Sequence[Any], matrix: Sequence[Sequence[int]], chunk: int = 0
    ) -> list[Any]:
        """`sends[t]` is (rows, ...) contiguous; this GPU sends `matrix[self.rank][p]` rows of
        each to GPU p, in p order, and receives `matrix[s][self.rank]` from each s, stacked in
        s order. Every GPU passes the same matrix and tensors of the same row shapes, so every
        GPU takes the same rounds: each moves up to `chunk` rows of every pair's block (0: as
        many as the buffer holds; the calibration asks for fewer, to cross in rounds)."""
        torch, world, rank = self.torch, self.world, self.rank
        widths = [tensor.element_size() * math.prod(tensor.shape[1:]) for tensor in sends]
        chunk = min(chunk or self.buffer.numel(), self.buffer.numel() // (world * sum(widths)))
        if chunk < 1:
            raise ValueError(f"one row of {sum(widths)} B per GPU exceeds the exchange buffer")
        # where each sender's rows land in the result
        firsts = [sum(matrix[s][rank] for s in range(q)) for q in range(world)]
        out = [t.new_empty((sum(row[rank] for row in matrix), *t.shape[1:])) for t in sends]
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
        return out

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


#: Bytes each GPU sends in the calibration: the size class of one Ulysses exchange.
_CALIBRATION_BYTES = 64 << 20


def calibrate(
    torch: Any, pg: Any, rank: int, world: int
) -> tuple[CopyEngineExchange | None, dict[str, float]]:
    """Collective, on every GPU of a just-formed group: the same all-to-all through NCCL and
    through the copy engines, timed and compared; every GPU drops the slower one alike. A
    GPU that cannot map symmetric memory makes the whole group keep NCCL."""
    from torch.distributed import _symmetric_memory as symm

    dist = torch.distributed

    def everyone(value: float, op: Any) -> float:
        agreed = torch.tensor([value], dtype=torch.float64, device="cuda")
        dist.all_reduce(agreed, op=op, group=pg)
        return float(agreed.item())

    # A formation again (after a call ran short of memory) frees the last one's buffer first.
    _BY_GROUP.clear()
    rows = _CALIBRATION_BYTES // 2 // 1024 // world * world
    # Two rounds, the second short: every formation checks the rounds against NCCL's bytes.
    half = rows // world // 2 + 1
    send = torch.randn(rows, 1024, dtype=torch.bfloat16, device="cuda")
    received = torch.empty_like(send)
    matrix = [[rows // world] * world for _ in range(world)]
    exchange: CopyEngineExchange | None = None
    try:
        exchange = CopyEngineExchange(torch, symm, pg, rank, world)
    except Exception:  # an unsupported driver or container: the group stays on NCCL
        exchange = None
    if everyone(0.0 if exchange is None else 1.0, dist.ReduceOp.MIN) < 1.0:
        return None, {"copy_engine_mapped": 0.0}
    assert exchange is not None
    exchange.map()
    sent = send.numel() * send.element_size() * (world - 1) / world
    timings: dict[str, float] = {}

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

    timings["nccl_gbps"] = timed(lambda: dist.all_to_all_single(received, send, group=pg))
    try:
        timings["copy_engine_gbps"] = timed(lambda: exchange.all_to_all([send], matrix, half))
        exact = torch.equal(received, exchange.all_to_all([send], matrix, half)[0])
    except Exception:  # the whole group then keeps NCCL, every GPU alike
        exact = False
        timings["copy_engine_gbps"] = 0.0
    timings["copy_engine_exact"] = everyone(1.0 if exact else 0.0, dist.ReduceOp.MIN)
    if timings["copy_engine_exact"] < 1.0 or timings["copy_engine_gbps"] <= timings["nccl_gbps"]:
        torch.cuda.synchronize()
        return None, timings
    return exchange, timings
