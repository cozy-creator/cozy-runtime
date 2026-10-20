"""One request's memory, as its invoke reply states it (memory-varena §5.8, xs-046 B4's wire)."""

from __future__ import annotations

import msgspec


class Step(msgspec.Struct, frozen=True, kw_only=True):
    """One pass of a model: bytes at home and resident, copied home and streamed through slots,
    the GPU-timed prefetch stall, torch's allocated peak above the pass's start, and torch's
    allocated bytes at that start (a leak shows as their growth)."""

    model: str
    gpu: int
    step: int
    home: int
    resident: int
    paged_in: int
    streamed: int
    stall_ns: int
    activations: int
    allocated: int = 0


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


class Copy(msgspec.Struct, frozen=True, kw_only=True):
    """A run of weight copies since the last request (`tensorfs.varena.torch.Copy`): to a GPU
    (`h2d`: a restore, refill or fill) or a host layout read from disk (`disk`), in GB/s and %
    of its measured ceiling (None while unmeasured), with the PCIe link, GPU copy and page-lock
    seconds."""

    kind: str
    gpu: int
    bytes: int
    seconds: float
    gbps: float
    ceiling: float
    pct: float | None
    link: str = ""
    busy_s: float = 0.0
    pin_s: float = 0.0
    staged: int = 0
    cached: int = 0


class Memory(msgspec.Struct, frozen=True, kw_only=True):
    """The invoke reply's `memory`, which `cozy run show` states as `measurements.memory.stats`
    and the merge gate reads: torch bytes package state still held at the request's end
    (`leftover`), weight bytes copied to the device while it ran (`paged_in`, home or through
    slots), and each model's bytes copied per step (`steps`). Then the detail: every step's
    row, the evictions it made, the peaks and cuts per shape key, its out-of-memory retries,
    `leftover` by GPU, and the copy runs against their ceilings."""

    leftover: int = 0
    paged_in: int = 0
    steps: dict[str, list[int]] = {}
    rows: list[Step] = []
    evictions: list[Eviction] = []
    peaks: list[Peak] = []
    retries: int = 0
    leftover_by_gpu: list[tuple[int, int]] = []
    #: the declared plan (`Context.plan`), each step's components
    plan: list[list[str]] = []
    copies: list[Copy] = []


class RankMemory(msgspec.Struct, frozen=True, kw_only=True):
    ok: bool
    rank: int
    request_id: str
    memory: Memory | None = None
    code: str = ""
    detail: str = ""
