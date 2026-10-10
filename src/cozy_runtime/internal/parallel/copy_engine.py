"""Ulysses' all-to-all on the GPUs' copy engines, through symmetric memory.

Every GPU of a group maps one receive buffer of every other (torch symmetric memory: CUDA
IPC/cuMem handles exchanged once over the group's store). An exchange is a barrier, one
copy-engine copy per peer into that peer's buffer (each on its own stream, so the NVLink and
PCIe engines run at once), a second barrier, and one local copy out. NCCL moves the same
bytes with kernels on the SMs, which across a CPU root complex are an order of magnitude
slower than the copy engines (H100 NVL, 4 GPUs: 3.3 GB/s against 20.8 GB/s per GPU).
The group uses this only where every pair measured direct and the calibration at formation
measured it faster than NCCL; the result is the same bytes either way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from cozy_runtime.internal.config import short_socket_directory

#: The exchange the group formed with, by process group; absent: NCCL.
_BY_GROUP: dict[int, CopyEngineExchange] = {}


def for_group(handle: Any) -> CopyEngineExchange | None:
    return _BY_GROUP.get(id(handle))


def install(exchange: CopyEngineExchange | None, handle: Any) -> None:
    _BY_GROUP.clear()
    if exchange is not None:
        _BY_GROUP[id(handle)] = exchange


class CopyEngineExchange:
    """All-to-all over dimension 0 for a list of tensors at once, by an agreed send matrix."""

    def __init__(self, torch: Any, symm: Any, pg: Any, rank: int, world: int) -> None:
        self.torch, self.symm, self.pg = torch, symm, pg
        self.rank, self.world = rank, world
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.streams = [torch.cuda.Stream(self.device) for _ in range(world)]
        self.buffer: Any = None
        self.handle: Any = None
        self.peers: list[Any] = []

    def _room(self, nbytes: int) -> None:
        """Every GPU's buffer holds the largest receive of the group (each computes the same
        size from the same matrix, so all of them grow together, collectively)."""
        if self.buffer is not None and self.buffer.numel() >= nbytes:
            return
        torch = self.torch
        size = nbytes if self.buffer is None else max(nbytes, 2 * self.buffer.numel())
        if self.buffer is not None:
            torch.cuda.current_stream(self.device).synchronize()
        self.peers, self.handle, self.buffer = [], None, None
        self.buffer = self.symm.empty(size, dtype=torch.uint8, device=self.device)
        with short_socket_directory():
            self.handle = self.symm.rendezvous(self.buffer, self.pg.group_name)
        self.peers = [self.handle.get_buffer(p, (size,), torch.uint8) for p in range(self.world)]

    def all_to_all(self, sends: Sequence[Any], matrix: Sequence[Sequence[int]]) -> list[Any]:
        """`sends[t]` is (rows, ...) contiguous; this GPU sends `matrix[self.rank][p]` rows of
        each to GPU p, in p order, and receives `matrix[s][self.rank]` from each s, stacked in
        s order. Every GPU passes the same matrix and tensors of the same row shapes."""
        torch, world, rank = self.torch, self.world, self.rank
        widths = [tensor.element_size() * math.prod(tensor.shape[1:]) for tensor in sends]
        received = [sum(row[p] for row in matrix) for p in range(world)]
        self._room(max(rows * sum(widths) for rows in received))
        current = torch.cuda.current_stream(self.device)
        self.handle.barrier(channel=0)
        issued = torch.cuda.Event()
        issued.record(current)
        # The receive buffers are scratch: a group GPU journals every block's writes into
        # storage it did not make (`weights.Journal`), and cloning these whole buffers per
        # block cost 70 s a denoise step (H100 NVL, H3). Their copies bypass Python dispatch.
        with torch._C._DisableTorchDispatch():
            self._issue(sends, matrix, widths, received, issued, current)
        self.handle.barrier(channel=1)
        out, base = [], 0
        for tensor, width in zip(sends, widths, strict=True):
            region = self.buffer[base : base + received[rank] * width]
            out.append(region.view(tensor.dtype).reshape(-1, *tensor.shape[1:]).clone())
            base += received[rank] * width
        return out

    def _issue(
        self,
        sends: Sequence[Any],
        matrix: Sequence[Sequence[int]],
        widths: Sequence[int],
        received: Sequence[int],
        issued: Any,
        current: Any,
    ) -> None:
        """One copy-engine copy per tensor per peer, each peer on its own stream."""
        torch, world, rank = self.torch, self.world, self.rank
        for step in range(world):
            peer = (rank + step) % world
            rows = matrix[rank][peer]
            sent = sum(matrix[rank][:peer])
            offset = sum(matrix[s][peer] for s in range(rank))
            stream = self.streams[peer]
            stream.wait_event(issued)
            with torch.cuda.stream(stream):
                base = 0
                for tensor, width in zip(sends, widths, strict=True):
                    if rows:
                        source = tensor[sent : sent + rows].reshape(-1).view(torch.uint8)
                        start = base + offset * width
                        self.peers[peer][start : start + rows * width].copy_(
                            source, non_blocking=True
                        )
                    base += received[peer] * width
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

    rows = _CALIBRATION_BYTES // 2 // 1024 // world * world
    send = torch.randn(rows, 1024, dtype=torch.bfloat16, device="cuda")
    received = torch.empty_like(send)
    matrix = [[rows // world] * world for _ in range(world)]
    exchange = CopyEngineExchange(torch, symm, pg, rank, world)
    try:
        exchange.buffer = symm.empty(rows * 1024 * 2, dtype=torch.uint8, device=exchange.device)
        mapped = 1.0
    except Exception:  # an unsupported driver or container: the group stays on NCCL
        mapped = 0.0
    if everyone(mapped, dist.ReduceOp.MIN) < 1.0:
        return None, {"copy_engine_mapped": 0.0}
    exchange.buffer = None
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
        timings["copy_engine_gbps"] = timed(lambda: exchange.all_to_all([send], matrix))
        exact = torch.equal(received, exchange.all_to_all([send], matrix)[0])
    except Exception:  # the whole group then keeps NCCL, every GPU alike
        exact = False
        timings["copy_engine_gbps"] = 0.0
    timings["copy_engine_exact"] = everyone(1.0 if exact else 0.0, dist.ReduceOp.MIN)
    if timings["copy_engine_exact"] < 1.0 or timings["copy_engine_gbps"] <= timings["nccl_gbps"]:
        torch.cuda.synchronize()
        return None, timings
    return exchange, timings
