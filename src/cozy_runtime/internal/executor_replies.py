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
    """One step: the capacity it ran against, its working set, its evictions by class."""

    capacity: int
    w: int
    evicted: tuple[int, ...]


class Decision(msgspec.Struct, frozen=True, kw_only=True):
    """One non-trivial decision a request's demand made: what asked (`trigger`: pass, bind,
    oom, prepare, workspace, outside), what it did (`action`: lower, cut, cut_outside, park,
    exhausted, evict_d, evict_w), the bytes asked (exhausted: short) and freed, a cut's level."""

    step: int
    trigger: str
    action: str
    site: str | None
    need: int
    freed: int
    level: int | None


class Unregisters(msgspec.Struct, frozen=True, kw_only=True):
    """RAM-tier host ranges unlocked while it ran (each stalls the device until idle)."""

    calls: int
    bytes: int
    max_ns: int


class Refusal(msgspec.Struct, frozen=True, kw_only=True):
    """varena's allocator callback's last refusal on a pool during the request: why
    (`no_reclaim`: over the limit, the stock gave nothing), the bytes refused, the pool's live
    bytes plus outside it, and the limit it was admitted to."""

    pool: tuple[int, int]
    why: str
    size: int
    held: int
    limit: int


class Stats(msgspec.Struct, frozen=True, kw_only=True):
    """varena's record of one request (varena-slim §5), as its core writes it. `evicted` is by
    class (F, O, S, N, D, W, P)."""

    request: str
    moved: Moved
    trace: tuple[Ref, ...]
    steps: tuple[Step, ...]
    stall_ns: int
    evicted: tuple[int, ...]
    peak: int
    leftover: int
    #: the policy's own counters, absent from a core with one make_room loop (varena-slim step 1)
    evicted_lowered: int = 0
    speculative_evicted_d: int = 0
    retries: int = 0
    cuts: int = 0
    parks: int = 0
    parked: int = 0
    lowered: bool = False
    room: int = 0
    host: int | None = None
    tier: int = 0
    keep_pending: int = 0
    unregisters: Unregisters = Unregisters(calls=0, bytes=0, max_ns=0)
    decisions: tuple[Decision, ...] = ()


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
    #: varena's allocator callback's refusals during the request, one per pool (its last)
    refusals: tuple[Refusal, ...] = ()


class RankMemory(msgspec.Struct, frozen=True, kw_only=True):
    ok: bool
    rank: int
    request_id: str
    memory: GpuMemory | None = None
    code: str = ""
    detail: str = ""
