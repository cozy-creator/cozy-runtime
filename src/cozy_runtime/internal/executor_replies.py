"""One request's memory, as its invoke reply states it: `measurements.memory.varena`."""

from __future__ import annotations

from typing import Any

import msgspec


class Ceiling(msgspec.Struct, frozen=True, kw_only=True):
    """TensorFS's measured ceilings for a GPU's fills: host to device, and disk."""

    h2d_gbps: float
    disk_gbps: float


class GpuMemory(msgspec.Struct, frozen=True, kw_only=True):
    """One GPU's part of a request: its physical number, the fill ceilings there, and varena's
    record of the request (its `Stats`, as varena's core writes it)."""

    gpu: int
    ceiling: Ceiling
    stats: dict[str, Any]


class RankMemory(msgspec.Struct, frozen=True, kw_only=True):
    ok: bool
    rank: int
    request_id: str
    memory: GpuMemory | None = None
    code: str = ""
    detail: str = ""
