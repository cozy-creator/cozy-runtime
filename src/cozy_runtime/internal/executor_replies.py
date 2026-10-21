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
    them were prefetched."""

    kept: Fills
    tier: Fills
    store: Fills
    speculative: int


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


class Stats(msgspec.Struct, frozen=True, kw_only=True):
    """varena's record of one request (varena-slim §5), as its core writes it. `evicted` is by
    class (F, O, S, N, D, W, P)."""

    request: str
    moved: Moved
    trace: tuple[Ref, ...]
    steps: tuple[Step, ...]
    stall_ns: int
    evicted: tuple[int, ...]
    evicted_lowered: int
    speculative_evicted_d: int
    retries: int
    cuts: int
    parks: int
    parked: int
    lowered: bool
    peak: int
    leftover: int


class GpuMemory(msgspec.Struct, frozen=True, kw_only=True):
    """One GPU's part of a request: its physical number, the fill ceilings there (None: not
    measured yet), and varena's record of the request."""

    gpu: int
    ceiling: Ceiling | None
    stats: Stats


class RankMemory(msgspec.Struct, frozen=True, kw_only=True):
    ok: bool
    rank: int
    request_id: str
    memory: GpuMemory | None = None
    code: str = ""
    detail: str = ""
