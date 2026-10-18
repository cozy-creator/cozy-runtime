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
    """The invoke reply's `memory`: each model's steps, the evictions the request made, the
    peaks per shape key, and torch bytes package state still held at its end, by GPU."""

    steps: list[Step] = []
    evictions: list[Eviction] = []
    peaks: list[Peak] = []
    leftover: list[tuple[int, int]] = []
