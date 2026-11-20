"""Ulysses' all-to-all on the GPUs' copy engines, peer to peer or through pinned host memory.

NCCL moves an exchange with kernels on the SMs, which across a CPU root complex are an order of
magnitude slower than the copy engines (H100 NVL, 4 GPUs: 3.3 GB/s against 20.8 GB/s per GPU).
Without peer access its shared-memory transport has those kernels write and read host memory
across PCIe (2x RTX 4090, cl-816: an H3 step took 15 s at degree 2 against 9 s at degree 1,
moving 27 GB a GPU each way). So the links of a group name its exchange (`links.Links.exchange`):
peer memory on every pair, or host memory where a pair has none. `form` maps it as the group forms
and checks one crossing against the bytes NCCL lands; the result is the same bytes either way.

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

Each tensor of a crossing has its own send matrix (a cross-attention's queries and keys have
different rows), so what one GPU sends another is a payload of byte segments, one a tensor
(`_plan`), and a round moves a window of every pair's payload. A GPU's own rows never leave it.
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

from cozy_runtime.graphs import streams
from cozy_runtime.internal.config import short_socket_directory
from cozy_runtime.internal.parallel.group import gather

if TYPE_CHECKING:
    from tensorfs.varena._runtime import Pages, Staging
    from torch import Tensor
    from torch import device as Device
    from torch._C._distributed_c10d import ProcessGroup, Store, _SymmetricMemory
    from torch.cuda import Event, Stream

#: GPU s sends `matrix[s][d]` rows of a tensor to GPU d.
type Matrix = Sequence[Sequence[int]]

#: The exchange the group formed with, by process group; absent: NCCL.
_BY_GROUP: dict[int, Exchange] = {}
#: Each GPU's receive buffer: an H3 step's Q/K/V crossing (~0.8 GB per GPU at 97k rows) takes
#: four rounds, each long enough that its two barriers cost a few percent.
BUFFER_BYTES = 256 << 20
#: Each GPU's host inbox, both halves: a region per source GPU in each.
INBOX_BYTES = 128 << 20
#: Exchange streams outrank compute, so a barrier's one block is never queued behind a wave.
_PRIORITY = -1


def for_group(handle: object) -> Exchange | None:
    return _BY_GROUP.get(id(handle))


def install(exchange: Exchange | None, handle: object) -> None:
    _drop(keep=exchange)
    if exchange is not None:
        _BY_GROUP[id(handle)] = exchange


def _drop(keep: Exchange | None = None) -> None:
    for old in _BY_GROUP.values():
        if old is not keep:
            old.close()
    _BY_GROUP.clear()


@dataclass(frozen=True, slots=True)
class _Segment:
    """One tensor's rows a GPU sends another: bytes `at` of the sender's tensor, landing at bytes
    `into` of the receiver's, `begins` bytes into the pair's payload."""

    tensor: int
    at: int
    into: int
    bytes: int
    begins: int


def _plan(
    widths: Sequence[int], matrices: Sequence[Matrix], world: int
) -> list[list[list[_Segment]]]:
    """`plan[s][d]`: GPU s's payload for GPU d, its segments in tensor order. `widths[t]`: bytes a
    row of tensor t. Every GPU computes the same plan from the same matrices."""
    plan: list[list[list[_Segment]]] = [[[] for _ in range(world)] for _ in range(world)]
    for s in range(world):
        for d in range(world):
            begins = 0
            for t, (width, matrix) in enumerate(zip(widths, matrices, strict=True)):
                size = matrix[s][d] * width
                if size:
                    at = sum(matrix[s][:d]) * width
                    into = sum(matrix[q][d] for q in range(s)) * width
                    plan[s][d].append(_Segment(t, at, into, size, begins))
                begins += size
    return plan


def _window(
    segments: Sequence[_Segment], start: int, size: int
) -> Iterator[tuple[_Segment, int, int, int]]:
    """The parts of a payload's `segments` in its bytes [start, start + size): each segment, the
    offset into it, the bytes, and where they sit in the window."""
    for segment in segments:
        low = max(segment.begins, start)
        high = min(segment.begins + segment.bytes, start + size)
        if low < high:
            yield segment, low - segment.begins, high - low, low - start


def _longest(plan: Sequence[Sequence[Sequence[_Segment]]]) -> int:
    """The largest payload between two GPUs: every GPU takes the rounds it needs."""
    world = len(plan)
    return max(
        (sum(s.bytes for s in plan[a][b]) for a in range(world) for b in range(world) if a != b),
        default=0,
    )


def _bytes(tensors: Sequence[Tensor], torch: ModuleType) -> list[Tensor]:
    return [tensor.reshape(-1).view(torch.uint8) for tensor in tensors]


class Exchange:
    """All-to-all over dimension 0 for a list of tensors at once, each by its agreed send matrix:
    `sends[t]` is (rows, ...) contiguous; this GPU sends `matrices[t][rank][p]` rows of it to GPU
    p, in p order, and receives `matrices[t][s][rank]` from each s, stacked in s order. Every GPU
    passes the same matrices and tensors of the same row shapes, so every GPU takes the same
    rounds."""

    name = ""
    torch: ModuleType
    rank: int
    world: int
    device: Device

    def start(
        self,
        sends: Sequence[Tensor],
        matrices: Sequence[Matrix],
        out: Sequence[Tensor],
        chunk: int = 0,
    ) -> Event:
        """The crossing on this exchange's streams after the caller's work so far; its end event.
        `chunk` caps each round's bytes per pair (0: as many as a buffer holds)."""
        raise NotImplementedError

    def close(self) -> None:
        return None

    def all_to_all(
        self, sends: Sequence[Tensor], matrices: Sequence[Matrix], chunk: int = 0
    ) -> list[Tensor]:
        out = [
            t.new_empty((sum(row[self.rank] for row in m), *t.shape[1:]))
            for t, m in zip(sends, matrices, strict=True)
        ]
        self.all_to_all_into(sends, matrices, out, chunk)
        return out

    def all_to_all_into(
        self,
        sends: Sequence[Tensor],
        matrices: Sequence[Matrix],
        out: Sequence[Tensor],
        chunk: int = 0,
    ) -> None:
        """The same crossing into actual preallocated receivers, on the current stream's order."""
        done = self.start(sends, matrices, out, chunk)
        self.torch.cuda.current_stream(self.device).wait_event(done)

    def _check(
        self, sends: Sequence[Tensor], matrices: Sequence[Matrix], out: Sequence[Tensor]
    ) -> list[int]:
        if not len(out) == len(sends) == len(matrices):
            raise ValueError("exchange receivers or matrices differ in count from its sends")
        for tensor, target, matrix in zip(sends, out, matrices, strict=True):
            if (
                not tensor.is_contiguous()
                or not target.is_contiguous()
                or tensor.shape[0] != sum(matrix[self.rank])
                or target.shape != (sum(row[self.rank] for row in matrix), *tensor.shape[1:])
                or target.dtype != tensor.dtype
                or target.device != tensor.device
            ):
                raise ValueError("exchange receiver differs from its declared layout")
        return [tensor.element_size() * math.prod(tensor.shape[1:]) for tensor in sends]

    def _own(
        self,
        plan: Sequence[Sequence[Sequence[_Segment]]],
        into: Sequence[Tensor],
        sent: Sequence[Tensor],
    ) -> None:
        """This GPU's rows for itself, on the current stream."""
        for s in plan[self.rank][self.rank]:
            into[s.tensor][s.into : s.into + s.bytes].copy_(sent[s.tensor][s.at : s.at + s.bytes])


class CopyEngineExchange(Exchange):
    """Peer to peer through symmetric memory. Allocates its buffer here (which may fail on one
    GPU: the caller agrees before `map`)."""

    name = "copy-engine"

    def __init__(
        self, torch: ModuleType, symm: ModuleType, pg: ProcessGroup, rank: int, world: int
    ) -> None:
        self.torch, self.symm, self.pg = torch, symm, pg
        self.rank, self.world = rank, world
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.streams = [streams.stream(torch, self.device) for _ in range(world)]
        self.side = streams.stream(torch, self.device, priority=_PRIORITY)
        # Straight from symmetric memory's own allocator: `symm.empty` allocates inside a MemPool
        # of the current allocator, which varena's (pluggable) refuses (`createOrIncrefPool`), so
        # inside an executor the exchange never mapped and NCCL carried it (cr-268).
        empty = symm._SymmetricMemory.empty_strided_p2p
        self.buffer: Tensor = empty((BUFFER_BYTES,), (1,), torch.uint8, self.device)
        self.flag: Tensor = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.handle: _SymmetricMemory | None = None
        self.peers: list[Tensor] = []

    def map(self) -> None:
        """Collective, once per formation: every GPU maps every other's buffer."""
        with short_socket_directory():
            handle: _SymmetricMemory = self.symm.rendezvous(self.buffer, self.pg.group_name)
            self.handle = handle
        shape = (self.buffer.numel(),)
        self.peers = [handle.get_buffer(p, shape, self.torch.uint8) for p in range(self.world)]

    def _barrier(self) -> None:
        """Every GPU's stream reached here: an abortable NCCL collective, never a spin."""
        self.torch.distributed.all_reduce(self.flag, group=self.pg)

    def start(
        self,
        sends: Sequence[Tensor],
        matrices: Sequence[Matrix],
        out: Sequence[Tensor],
        chunk: int = 0,
    ) -> Event:
        torch = self.torch
        widths = self._check(sends, matrices, out)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        self.side.wait_event(ready)
        with torch.cuda.stream(self.side):
            self._rounds(sends, matrices, out, widths, chunk)
            done: Event = torch.cuda.Event()
            done.record(self.side)
        return done

    def _rounds(
        self,
        sends: Sequence[Tensor],
        matrices: Sequence[Matrix],
        out: Sequence[Tensor],
        widths: Sequence[int],
        chunk: int,
    ) -> None:
        """Each GPU's buffer holds a region per source GPU; a round moves a window of every pair's
        payload into its region there, between two barriers, and each GPU copies its own out."""
        torch, world, rank = self.torch, self.world, self.rank
        region = self.buffer.numel() // world // 4096 * 4096
        size = min(chunk or region, region)
        plan = _plan(widths, matrices, world)
        sent, into = _bytes(sends, torch), _bytes(out, torch)
        current = torch.cuda.current_stream(self.device)
        self._own(plan, into, sent)
        for start in range(0, _longest(plan), size):
            self._barrier()
            issued = torch.cuda.Event()
            issued.record(current)
            # The receive buffers are scratch: their copies bypass Python dispatch, so the
            # memory's catch never takes a block that writes them for one that writes its inputs.
            with torch._C._DisableTorchDispatch():
                self._issue(plan, sent, start, size, region, issued, current)
            self._barrier()
            for source in range(world):
                if source == rank:
                    continue
                base = source * region
                for s, offset, n, at in _window(plan[source][rank], start, size):
                    begin = s.into + offset
                    into[s.tensor][begin : begin + n].copy_(self.buffer[base + at : base + at + n])

    def _issue(
        self,
        plan: Sequence[Sequence[Sequence[_Segment]]],
        sent: Sequence[Tensor],
        start: int,
        size: int,
        region: int,
        issued: Event,
        current: Stream,
    ) -> None:
        """This round's window of each peer's payload into this GPU's region of that peer's
        buffer, each peer on its own stream."""
        torch, world, rank = self.torch, self.world, self.rank
        base = rank * region
        for step in range(1, world):
            peer = (rank + step) % world
            stream = self.streams[peer]
            stream.wait_event(issued)
            with torch.cuda.stream(stream):
                for s, offset, n, at in _window(plan[rank][peer], start, size):
                    begin = s.at + offset
                    self.peers[peer][base + at : base + at + n].copy_(
                        sent[s.tensor][begin : begin + n], non_blocking=True
                    )
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

    def __init__(
        self, torch: ModuleType, pg: ProcessGroup, rank: int, world: int, pages: Pages | None
    ) -> None:
        self.torch, self.pg, self.rank, self.world = torch, pg, rank, world
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.slot = INBOX_BYTES // (2 * world) // 4096 * 4096
        self.size = 2 * world * self.slot
        self.writes = streams.stream(torch, self.device, priority=_PRIORITY)
        self.reads = streams.stream(torch, self.device, priority=_PRIORITY)
        self.flag = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.half = 0
        self.inboxes: list[Tensor] = []
        self.peers: list[Staging] = []
        own = pages.staging(self.size) if pages is not None else None
        if own is None:
            raise MemoryError("varena's host number has no room for this GPU's exchange inbox")
        #: its own inbox until `close`, which `map` names to the group
        self.own: Staging | None = own

    def map(self, store: Store, agree: Callable[[bool], bool]) -> bool:
        """Collective, once per formation: every GPU maps every inbox for its own; each unlinks
        its inbox's name once all agreed. False on every GPU alike when any could not."""
        from tensorfs.varena import _runtime as varena

        own = self.own
        assert own is not None  # mapped once per formation, before `close`
        store.set(f"cozy.staged/{self.rank}", own.name)
        keys = [f"cozy.staged/{r}" for r in range(self.world)]
        names = [bytes(name).decode() for name in gather(store, keys, "the exchange's inboxes")]
        ok = True
        try:
            for rank, name in enumerate(names):
                staging = own
                if rank != self.rank:
                    staging = varena.Staging.open(self.device.index, name, self.size)
                    self.peers.append(staging)
                self.inboxes.append(_host(self.torch, staging))
        except Exception:
            ok = False
        agreed = agree(ok)
        own.unlink()
        if not agreed:
            self.close()
            return False
        self.free: list[Event] = [self.torch.cuda.Event(), self.torch.cuda.Event()]
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

    def _region(self, inbox: Tensor, half: int, source: int) -> Tensor:
        at = (half * self.world + source) * self.slot
        return inbox[at : at + self.slot]

    def start(
        self,
        sends: Sequence[Tensor],
        matrices: Sequence[Matrix],
        out: Sequence[Tensor],
        chunk: int = 0,
    ) -> Event:
        torch, world, rank = self.torch, self.world, self.rank
        widths = self._check(sends, matrices, out)
        plan = _plan(widths, matrices, world)
        longest = _longest(plan)
        size = min(chunk or self.slot, self.slot)
        sent, into = _bytes(sends, torch), _bytes(out, torch)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.device))
        self.writes.wait_event(ready)
        self.reads.wait_event(ready)
        with torch.cuda.stream(self.reads):
            self._own(plan, into, sent)
        for start in range(0, longest, size):
            half, self.half = self.half, self.half ^ 1
            self.writes.wait_event(self.free[half])
            # The inboxes are scratch: their writes bypass Python dispatch (`CopyEngineExchange`).
            with torch.cuda.stream(self.writes), torch._C._DisableTorchDispatch():
                for peer in range(world):
                    if peer == rank:
                        continue
                    region = self._region(self.inboxes[peer], half, rank)
                    for s, offset, n, at in _window(plan[rank][peer], start, size):
                        begin = s.at + offset
                        piece = sent[s.tensor][begin : begin + n]
                        region[at : at + n].copy_(piece, non_blocking=True)
                written = torch.cuda.Event()
                written.record(self.writes)
            self.reads.wait_event(written)
            with torch.cuda.stream(self.reads):
                self._barrier()
                self.free[half ^ 1].record(self.reads)
                for source in range(world):
                    if source == rank:
                        continue
                    region = self._region(self.inboxes[rank], half, source)
                    for s, offset, n, at in _window(plan[source][rank], start, size):
                        begin = s.into + offset
                        landing = into[s.tensor][begin : begin + n]
                        landing.copy_(region[at : at + n], non_blocking=True)
        done: Event = torch.cuda.Event()
        done.record(self.reads)
        return done


def _host(torch: ModuleType, staging: Staging) -> Tensor:
    """`staging`'s bytes as a CPU uint8 tensor, valid while it is open."""
    host: Tensor = torch.frombuffer(
        (ctypes.c_uint8 * staging.bytes).from_address(staging.ptr), dtype=torch.uint8
    )
    return host


#: Bytes each GPU sends in the check a formation runs, in two rounds.
_CHECK_BYTES = 8 << 20


def form(
    torch: ModuleType,
    pg: ProcessGroup,
    rank: int,
    world: int,
    name: str,
    pages: Pages | None = None,
) -> tuple[Exchange | None, dict[str, bool | str]]:
    """Collective, on every GPU of a just-formed group: `name` (`links.Links.exchange`) mapped,
    and one all-to-all through it checked against the bytes NCCL lands. An exchange any GPU cannot
    map, or whose bytes differ on any, goes on all of them alike and NCCL carries the group; the
    check says why (`failed`: this GPU's exception, or what another GPU's agreement said).
    `pages`: the host tier of this GPU, which a host-staged inbox comes from."""
    from torch.distributed.distributed_c10d import _get_process_group_store

    dist = torch.distributed

    def agree(ok: bool) -> bool:
        agreed = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=pg)
        return bool(agreed.item())

    # A formation again (after a call ran short of memory) frees the buffers of the last first.
    _drop()
    exchange, failed = _open(
        name, torch, pg, rank, world, agree, _get_process_group_store(pg), pages
    )
    if exchange is None:
        return None, {"mapped": False, "exact": False, **({"failed": failed} if failed else {})}
    rows = _CHECK_BYTES // 2 // 1024 // world * world
    send = torch.randn(rows, 1024, dtype=torch.bfloat16, device="cuda")
    received = torch.empty_like(send)
    dist.all_to_all_single(received, send, group=pg)
    matrix = [[rows // world] * world for _ in range(world)]
    try:
        # Two rounds, the second short.
        row = send[0].numel() * send.element_size()
        landed = exchange.all_to_all([send], [matrix], (rows // world // 2 + 1) * row)[0]
        exact = bool(torch.equal(received, landed))
        failed = "" if exact else "its bytes differ from what NCCL lands"
    except Exception as exc:  # the whole group then drops it, every GPU alike
        exact, failed = False, _said(exc)
    agreed = agree(exact)
    if not agreed:
        exchange.close()
    torch.cuda.synchronize()
    check: dict[str, bool | str] = {"mapped": True, "exact": agreed}
    if not agreed:
        check["failed"] = failed or "another GPU of the group failed its check"
    return (exchange if agreed else None), check


def _open(
    name: str,
    torch: ModuleType,
    pg: ProcessGroup,
    rank: int,
    world: int,
    agree: Callable[[bool], bool],
    store: Store,
    pages: Pages | None,
) -> tuple[Exchange | None, str]:
    """`name`'s exchange mapped on every GPU of the group, or None on every one alike, with why
    (empty when no exchange was asked for)."""
    made: Exchange | None = None
    failed = ""
    try:
        if name == "copy-engine":
            from torch.distributed import _symmetric_memory as symm

            made = CopyEngineExchange(torch, symm, pg, rank, world)
        elif name == "host-staged":
            made = HostStagedExchange(torch, pg, rank, world, pages)
    except Exception as exc:  # an unsupported driver, no host room, no shared memory: NCCL stays
        made, failed = None, _said(exc)
    if not agree(made is not None):
        if made is not None:
            made.close()
        asked = name in ("copy-engine", "host-staged")
        return None, failed or ("another GPU of the group could not map it" if asked else "")
    if isinstance(made, CopyEngineExchange):
        made.map()
        return made, ""
    assert isinstance(made, HostStagedExchange)
    prefixed = torch.distributed.PrefixStore(f"cozy.exchange/{pg.group_name}/", store)
    if made.map(prefixed, agree):
        return made, ""
    return None, "an inbox did not map on every GPU of the group"


def _said(exc: BaseException) -> str:
    """An exception as a check keeps it: its type and the first line of its message."""
    return f"{type(exc).__name__}: {str(exc).partition(chr(10))[0]}"[:300]
