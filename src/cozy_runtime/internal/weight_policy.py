"""Weight-plane memory policy: pure arithmetic over measured facts (weight-plane.md §4-5).

The plane moves bytes and never decides; this module decides and never moves bytes. It takes
no torch, no plane and no I/O, so the executor (per stage) and the Worker (per tenant) run the
same rules. Activations come first: the plane gets what the driver has free after them.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

#: Slots in flight when nothing is measured yet: the block computing and the next one arriving.
DEFAULT_WINDOW = 2
MAX_WINDOW = 8
#: Allocator fragmentation between the plane's budget and torch's activations. Small because a
#: torch OOM at a block boundary lowers the budget and retries (§4, activations first).
MARGIN = 64 << 20


@dataclass(frozen=True, slots=True)
class Layout:
    """One weight set as the plane holds it: `common` and each no-split block, in bytes, in
    address order. A component without declared blocks is `common` alone."""

    common: int
    blocks: tuple[int, ...] = ()

    @property
    def total(self) -> int:
        return self.common + sum(self.blocks)

    @property
    def largest(self) -> int:
        return max(self.blocks, default=0)


@dataclass(frozen=True, slots=True)
class Residence:
    """One component's placement for one stage: which blocks stay resident and the streamed
    tail's window. Block numbers are address-order indexes into `Layout.blocks`."""

    resident: tuple[int, ...]
    streamed: tuple[int, ...]
    window: int

    def vram_bytes(self, layout: Layout) -> int:
        """What this placement holds on the device: common, the prefix and the slot ring."""
        slots = self.window * max((layout.blocks[i] for i in self.streamed), default=0)
        return layout.common + sum(layout.blocks[i] for i in self.resident) + slots


class BelowFloor(Exception):
    """The budget cannot hold even the non-streamable minimum. Carries both numbers."""

    def __init__(self, needed: int, available: int, detail: str) -> None:
        super().__init__(f"{detail}: needs {needed} B, {available} B available")
        self.needed = needed
        self.available = available


def plane_budget(free: int, activation: int, context: int = 0, margin: int = MARGIN) -> int:
    """`free - activation growth - context`: the device bytes the plane may map."""
    return max(free - activation - context - margin, 0)


def floor(layout: Layout, window: int, activation: int = 0) -> int:
    """The least a stage can run in: common, W slots of the largest block, activations. A
    component without blocks cannot stream, so all of it."""
    if not layout.blocks:
        return layout.total + activation
    return layout.common + max(window, 1) * layout.largest + activation


def window(block_bytes: int, link_bps: float, compute_s: float) -> int:
    """Slots so the copy of block i+W-1 finishes while blocks i..i+W-2 compute."""
    if block_bytes <= 0 or link_bps <= 0 or compute_s <= 0:
        return DEFAULT_WINDOW
    ahead = math.ceil(block_bytes / link_bps / compute_s)
    return min(max(ahead + 1, DEFAULT_WINDOW), MAX_WINDOW)


def residence(
    layout: Layout,
    budget: int,
    *,
    order: Sequence[int] = (),
    window_size: int = DEFAULT_WINDOW,
    cyclic: bool = True,
) -> Residence:
    """The largest resident prefix that fits `budget` beside the slot ring.

    `order` is the blocks' execution order once observed (address order before). A cyclic stage
    (a denoise loop) keeps a fixed prefix in that order: the tail's refill for the next step
    then overlaps the prefix's compute, and nothing resident is ever evicted by the loop itself.
    A run-once stage gains nothing from a partial prefix within the request, so it is resident
    whole when it fits and streamed whole otherwise. Raises `BelowFloor` when even the floor
    does not fit.
    """
    count = len(layout.blocks)
    walk = tuple(order) if sorted(order) == list(range(count)) else tuple(range(count))
    if layout.total <= budget:
        return Residence(resident=walk, streamed=(), window=0)
    need = floor(layout, window_size)
    if not layout.blocks or need > budget:
        raise BelowFloor(need, budget, "the stage's weights do not fit the plane budget")
    if not cyclic:
        return Residence(resident=(), streamed=walk, window=window_size)
    room = budget - layout.common - window_size * layout.largest
    kept = 0
    for index in walk:
        if layout.blocks[index] > room:
            break
        room -= layout.blocks[index]
        kept += 1
    return Residence(resident=walk[:kept], streamed=walk[kept:], window=window_size)


def stage(
    layouts: Mapping[str, Layout],
    budget: int,
    *,
    orders: Mapping[str, Sequence[int]] | None = None,
    windows: Mapping[str, int] | None = None,
    cyclic: Collection[str] = (),
) -> dict[str, Residence]:
    """Every declared component's residence for one stage, inside one plane budget.

    Each component first gets its floor; what is left goes to cyclic components' prefixes
    first (they repeat, so every resident byte saves a copy per step), then to run-once ones,
    which are resident only when whole. Components outside the stage are not counted: they
    are evictable, lower priority, and the plane unmaps them as this stage's wants arrive.
    """
    orders, windows = orders or {}, windows or {}
    width = {name: windows.get(name, DEFAULT_WINDOW) for name in layouts}
    need = {name: floor(layout, width[name]) for name, layout in layouts.items()}
    if sum(need.values()) > budget:
        # Serial streaming (one slot, no overlap) is slower, never refused.
        width = dict.fromkeys(layouts, 1)
        need = {name: floor(layout, 1) for name, layout in layouts.items()}
    if sum(need.values()) > budget:
        raise BelowFloor(
            sum(need.values()),
            budget,
            "the stage's common weights and one streaming slot do not fit the plane budget",
        )
    spare = budget - sum(need.values())
    placed: dict[str, Residence] = {}
    for name in sorted(layouts, key=lambda n: (n not in cyclic, -layouts[n].total, n)):
        layout = layouts[name]
        chosen = residence(
            layout,
            need[name] + spare,
            order=orders.get(name, ()),
            window_size=width[name],
            cyclic=name in cyclic,
        )
        spare -= max(chosen.vram_bytes(layout) - need[name], 0)
        placed[name] = chosen
    return placed


def stall_s(streamed_bytes: int, link_bps: float, overlap_s: float) -> float:
    """Predicted stall per step: the copy time the compute cannot hide."""
    if link_bps <= 0:
        return math.inf if streamed_bytes else 0.0
    return max(0.0, streamed_bytes / link_bps - overlap_s)


def decode_mode(free: int, evictable: int, untiled_bytes: int) -> Literal["untiled", "tiled"]:
    """Untiled when its activations fit what is free plus what the plane may unmap."""
    return "untiled" if untiled_bytes <= free + evictable else "tiled"


def batch_fits(free: int, evictable: int, extra_bytes: int) -> bool:
    """Whether a batched call (e.g. cond+uncond as one batch) fits, counting evictable weights
    as free. Decided before the stage and recorded; never switched mid-request."""
    return extra_bytes <= free + evictable
