"""One request's memory, as its invoke reply states it: `measurements.memory.varena`."""

from __future__ import annotations

import msgspec


class Ceiling(msgspec.Struct, frozen=True, kw_only=True):
    """TensorFS's measured ceilings for a GPU's fills: host to device (None: no GPU), and
    disk."""

    h2d_gbps: float | None = None
    disk_gbps: float = 0.0


class GpuMemory(msgspec.Struct, frozen=True, kw_only=True):
    """One GPU's part of a request: its physical number, the fill ceilings there (None: not
    measured yet), varena's record of the request (`tensorfs.varena.Stats`, its one definition,
    copied attribute by attribute: `execution.record`), and its arena's torch high-water (bytes:
    torch's own statistics don't see varena's allocator). A follower's crosses the group's seam:
    every field but the GPU has a default, and unknown ones are ignored."""

    gpu: int
    ceiling: Ceiling | None = None
    stats: dict[str, object] = msgspec.field(default_factory=dict)
    torch_peak: int = 0


class RankMemory(msgspec.Struct, frozen=True, kw_only=True):
    ok: bool
    rank: int
    request_id: str
    memory: GpuMemory | None = None
    code: str = ""
    detail: str = ""
