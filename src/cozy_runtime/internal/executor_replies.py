"""One request's memory, as its invoke reply states it: `measurements.memory.varena`."""

from __future__ import annotations

import msgspec


class Ceiling(msgspec.Struct, frozen=True, kw_only=True):
    """TensorFS's measured ceilings for a GPU's fills: host to device (None: no GPU), and
    disk."""

    h2d_gbps: float | None = None
    disk_gbps: float = 0.0


class GpuMemory(msgspec.Struct, frozen=True, kw_only=True):
    """One physical GPU's request observation. `stats` labels its allocator scope: stock Torch
    counters are process-wide and can include overlapping attempts. `torch_peak` is the
    absolute allocator high-water, not bytes exclusively owned by this request. The historic
    `memory.varena` wire key remains so existing record readers retain these observations."""

    gpu: int
    ceiling: Ceiling | None = None
    stats: dict[str, object] = msgspec.field(default_factory=dict)
    torch_peak: int | None = None


class RankMemory(msgspec.Struct, frozen=True, kw_only=True):
    ok: bool
    rank: int
    request_id: str
    memory: GpuMemory | None = None
    code: str = ""
    detail: str = ""
