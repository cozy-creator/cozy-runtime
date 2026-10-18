"""The executor's memory facts, as each reply and probe states them (`tensorfs.varena.torch`)."""

from __future__ import annotations

import msgspec


class Step(msgspec.Struct, frozen=True, kw_only=True):
    """One pass of a model: bytes at home and resident, copied home and streamed through slots,
    the GPU-timed prefetch stall, and torch's allocated peak above the pass's start."""

    model: str
    step: int
    home: int
    resident: int
    paged_in: int
    streamed: int
    stall_ns: int
    activations: int


class Eviction(msgspec.Struct, frozen=True, kw_only=True):
    model: str
    rung: int
    reason: str
    nbytes: int


class Memory(msgspec.Struct, frozen=True, kw_only=True):
    """One request's memory (memory-varena §5.8): each model's steps, the evictions it made,
    and torch bytes package state still held at its end."""

    steps: list[Step] = []
    evictions: list[Eviction] = []
    leftover_bytes: int = 0


class PlaneFacts(msgspec.Struct, frozen=True, kw_only=True):
    """The executor's memory as exact counters, cumulative since it started. Bytes are this
    GPU's; -1 is unreadable, never zero."""

    #: weight bytes resident, and by model
    committed_bytes: int = -1
    resident: dict[str, int] = {}
    #: CUDA context and runtime workspaces outside torch and the weights, measured at load
    context_bytes: int = -1
    #: torch's reserved bytes, and the process's device bytes (context + reserved + weights)
    reserved_bytes: int = -1
    process_bytes: int = -1
    #: torch's peak above its bytes at the attempt's start
    activation_peak_bytes: int = -1
    #: weight bytes copied to the device, home or through slots, and the prefetch stall
    h2d_bytes: int = 0
    stall_ns: int = 0
    evictions: int = 0
    lora_baked: int = 0
    #: torch's allocator: times it dropped its whole cache to place one allocation, and its
    #: out-of-memory errors
    alloc_retries: int = 0
    alloc_ooms: int = 0
    memory: Memory | None = None
