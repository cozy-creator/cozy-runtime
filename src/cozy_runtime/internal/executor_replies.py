"""One request's memory, as its invoke reply states it (memory-varena §5.8, xs-046 B4's wire)."""

from __future__ import annotations

import msgspec


class Step(msgspec.Struct, frozen=True, kw_only=True):
    """One pass of a model: bytes at home and resident, copied home and streamed through slots,
    the GPU-timed prefetch stall, and torch's allocated peak above the pass's start."""

    model: str
    gpu: int
    step: int
    home: int
    resident: int
    paged_in: int
    streamed: int
    stall_ns: int
    activations: int


class Eviction(msgspec.Struct, frozen=True, kw_only=True):
    model: str
    gpu: int
    rung: int
    reason: str
    nbytes: int


class Peak(msgspec.Struct, frozen=True, kw_only=True):
    """A model's measured activation peak at one shape key, and the cuts in force there (B3)."""

    model: str
    shape: str
    peak: int
    cuts: list[str] = []


class Memory(msgspec.Struct, frozen=True, kw_only=True):
    """The invoke reply's `memory`, which `cozy run show` states as `measurements.memory.stats`
    and the merge gate reads: torch bytes package state still held at the request's end
    (`leftover`), weight bytes copied to the device while it ran (`paged_in`, home or through
    slots), and each model's bytes copied per step (`steps`). Then the detail: every step's
    row, the evictions it made, the peaks per shape key, and `leftover` by GPU."""

    leftover: int = 0
    paged_in: int = 0
    steps: dict[str, list[int]] = {}
    rows: list[Step] = []
    evictions: list[Eviction] = []
    peaks: list[Peak] = []
    leftover_by_gpu: list[tuple[int, int]] = []
