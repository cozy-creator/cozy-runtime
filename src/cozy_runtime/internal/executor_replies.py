"""One request's memory, as its invoke reply states it: `measurements.memory.varena`."""

from __future__ import annotations

from typing import Literal

import msgspec


class Ceiling(msgspec.Struct, frozen=True, kw_only=True):
    """TensorFS's measured ceilings for a GPU's fills: host to device (None: no GPU), and
    disk."""

    h2d_gbps: float | None
    disk_gbps: float


class Fills(msgspec.Struct, frozen=True, kw_only=True):
    """Bytes filled from one source and the time the fills took."""

    bytes: int
    ns: int


class Moved(msgspec.Struct, frozen=True, kw_only=True):
    """Bytes moved by source (kept pages imported, the RAM tier, the store); `speculative` of
    them were prefetched; `saved`, filled pages copied into the RAM tier afterwards."""

    kept: Fills
    tier: Fills
    store: Fills
    speculative: int
    saved: Fills = Fills(bytes=0, ns=0)


class Ref(msgspec.Struct, frozen=True, kw_only=True):
    """One bind: the page reference trace the gates compute Belady MIN from."""

    step: int
    set: str
    block: int
    bytes: int
    origin: Literal["Hit", "Kept", "Tier", "Store"]
    speculative: bool


class Step(msgspec.Struct, frozen=True, kw_only=True):
    """One step: the capacity it ran against and what make_room took in it, by class."""

    capacity: int
    evicted: tuple[int, ...]


class MakeRoom(msgspec.Struct, frozen=True, kw_only=True):
    """One make_room that had to take bytes: the plan step, what asked (the allocator hook, a
    bind, a range's preparation), the free bytes asked and taken, and the release events it
    waited on first."""

    step: int
    trigger: str
    need: int
    freed: int
    waited: int


class Unregisters(msgspec.Struct, frozen=True, kw_only=True):
    """RAM-tier host ranges unlocked while it ran (each stalls the device until idle)."""

    calls: int
    bytes: int
    max_ns: int


class Refusal(msgspec.Struct, frozen=True, kw_only=True):
    """varena's allocator hook's last refusal on a pool during the request: why, the bytes
    refused, and the row's free bytes then."""

    pool: tuple[int, int]
    why: str
    size: int
    free: int


class Stats(msgspec.Struct, frozen=True, kw_only=True):
    """varena's record of one request (varena-slim §5), as its core writes it: what make_room
    took (`evicted`, by class F, O, S, N, D, W, P), the most it held at a pass end (pages and
    segments) and its passes' compute bytes."""

    request: str
    moved: Moved
    trace: tuple[Ref, ...]
    steps: tuple[Step, ...]
    stall_ns: int
    evicted: tuple[int, ...]
    peak: int
    compute: int
    host: int | None
    tier: int
    leftover: int
    keep_pending: int
    unregisters: Unregisters
    make_room: tuple[MakeRoom, ...]


class Allocator(msgspec.Struct, frozen=True, kw_only=True):
    """torch's caching allocator over a request on one GPU (`torch.cuda.memory_stats` deltas,
    the device's: a request beside it on that GPU counts too): allocations retried after it
    freed its cache, cudaMalloc and cudaFree calls, and the all-stream syncs those frees took;
    and the allocations varena's callback refused on the device."""

    num_alloc_retries: int
    num_device_alloc: int
    num_device_free: int
    num_sync_all_streams: int
    refused: int = 0


class GpuMemory(msgspec.Struct, frozen=True, kw_only=True):
    """One GPU's part of a request: its physical number, the fill ceilings there (None: not
    measured yet), varena's record of the request, and torch's allocator over it (None: no
    GPU)."""

    gpu: int
    ceiling: Ceiling | None
    stats: Stats
    torch: Allocator | None
    #: varena's allocator hook's refusals during the request, one per pool (its last)
    refusals: tuple[Refusal, ...]
    #: the request's pool's drop at its end (torch syncs its events), and per pass that made
    #: room its pool's free blocks then: (model, key, bytes, the largest)
    flush_ms: float
    pool_free: tuple[tuple[str, str, int, int], ...]


class RankMemory(msgspec.Struct, frozen=True, kw_only=True):
    ok: bool
    rank: int
    request_id: str
    memory: GpuMemory | None = None
    code: str = ""
    detail: str = ""
