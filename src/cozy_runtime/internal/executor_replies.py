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
    """Bytes moved by source (kept pages imported, leased chunks adopted, the store through the
    page cache); `speculative` of them were prefetched."""

    kept: Fills
    leased: Fills
    store: Fills
    speculative: int


class Ref(msgspec.Struct, frozen=True, kw_only=True):
    """One bind: the page reference trace the gates compute Belady MIN from."""

    step: int
    set: str
    block: int
    bytes: int
    origin: Literal["Hit", "Kept", "Store"]
    speculative: bool


class Step(msgspec.Struct, frozen=True, kw_only=True):
    """One step: the free bytes at its start and what make_room took in it, by class."""

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
    leftover: int
    make_room: tuple[MakeRoom, ...]
    #: the request-end audit: its arena's torch blocks still live outside any region at its
    #: end, (address, bytes, stream) each; tests fail on any
    outlived: tuple[tuple[int, int, int], ...]


class GpuMemory(msgspec.Struct, frozen=True, kw_only=True):
    """One GPU's part of a request: its physical number, the fill ceilings there (None: not
    measured yet), varena's record of the request, and its arena's torch high-water (bytes:
    torch's own statistics don't see varena's allocator)."""

    gpu: int
    ceiling: Ceiling | None
    stats: Stats
    torch_peak: int


class RankMemory(msgspec.Struct, frozen=True, kw_only=True):
    ok: bool
    rank: int
    request_id: str
    memory: GpuMemory | None = None
    code: str = ""
    detail: str = ""
