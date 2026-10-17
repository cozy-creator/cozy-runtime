"""The executor's weight-plane facts, as each reply and probe states them."""

from __future__ import annotations

import msgspec


class Streamed(msgspec.Struct, frozen=True, kw_only=True):
    """One component's streamed tail at its last stage: regions, resident prefix, window."""

    blocks: int = 0
    resident_blocks: int = 0
    window: int = 0


class PlaneFacts(msgspec.Struct, frozen=True, kw_only=True):
    """The executor's weight plane (`weight_plane/1`), as exact counters. Bytes are per device
    of this rank; `-1` is unreadable, never zero."""

    budget_bytes: int = -1
    committed_bytes: int = -1
    leased_bytes: int = -1
    pinned_budget_bytes: int = -1
    pinned_bytes: int = -1
    #: CUDA context and runtime workspaces outside torch and the plane, measured at load
    context_bytes: int = -1
    #: torch's reserved bytes, and the process's device bytes (context + reserved + mapped)
    reserved_bytes: int = -1
    process_bytes: int = -1
    #: the owner's process cap in force (`process_cap/1`); -1: none
    cap_bytes: int = -1
    #: torch's peak above its bytes at the attempt's start: activations and derived weights
    activation_peak_bytes: int = -1
    resident: dict[str, int] = {}
    streamed: dict[str, Streamed] = {}
    h2d_bytes: int = 0
    h2d_gbps: float = 0.0
    #: of those, copied to the device straight from disk (the pinned tier did not hold them)
    #: and straight from the page cache (it did not either)
    disk_copy_bytes: int = 0
    mapped_copy_bytes: int = 0
    #: of those, host memory this process could not page-lock, through the staging buffers;
    #: the rest of the host bytes were one DMA each. Each path's own rate (bytes / ns)
    bounce_copy_bytes: int = 0
    host_gbps: float = 0.0
    bounce_gbps: float = 0.0
    disk_gbps: float = 0.0
    #: page locks the driver refused (a sealed layout counts once)
    register_failed: int = 0
    #: per stage method, its last plan's inputs, bytes: driver free, torch reserved and
    #: allocated, the plane's committed and the bound weights' decoded copies, the stage's
    #: activation growth and the budget that came out (`Weights.driver_budget`)
    plan_inputs: dict[str, dict[str, int]] = {}
    #: GPU regions mapped from the machine's custody (`weights.attach/1`): outside
    #: `committed_bytes`, counted once by the machine
    shared_bytes: int = 0
    #: bytes this executor process read from storage since it started (`/proc/self/io`;
    #: TensorFS 0.3.92): a gate compares a delta across a return. -1: not counted
    disk_read_bytes: int = -1
    #: RAM of the machine's sealed layouts this executor maps (theirs, never its own)
    sealed_bytes: int = 0
    #: LoRA'd weights baked into device bytes since the process started (a streamed copy counts
    #: at every acquire), and the milliseconds its resident bakes took
    lora_baked: int = 0
    lora_bake_ms: float = 0.0
    #: LoRA'd weights whose bytes arrived baked from the machine's custody: no bake here
    lora_attached: int = 0
    late: int = 0
    stall_ns: int = 0
    misses: int = 0
    evictions: int = 0
    oom_retries: int = 0
    #: torch's allocator short of room: times it dropped its whole cache to place one
    #: allocation, and out-of-memory errors it raised; then this process's own cache releases
    #: that returned something, their bytes, and both by the function that asked. A stage
    #: with room shows none of the first two.
    alloc_retries: int = 0
    alloc_ooms: int = 0
    cache_releases: int = 0
    cache_released_bytes: int = 0
    cache_released_by: dict[str, tuple[int, int]] = {}
    #: components given back to make torch room since the process began, and their bytes
    paged_out: int = 0
    paged_out_bytes: int = 0
    #: times this process's own tensors moved to host memory for an op since it began, and
    #: their bytes (`spill`)
    spilled: int = 0
    spilled_bytes: int = 0
    #: What memory made the call do, e.g. {"bands": "12 ops, 805306368 B of results in host memory"}
    modes: dict[str, str] = {}
