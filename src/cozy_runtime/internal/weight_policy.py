"""Weight-plane memory policy: pure arithmetic over measured facts (weight-plane.md §4-5).

The plane moves bytes and never decides; this module decides and never moves bytes. It takes
no torch, no plane and no I/O, so the executor (per stage) and the Worker (per tenant) run the
same rules. Activations come first: the plane gets what the driver has free after them.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

#: Streamed positions in flight when nothing is measured yet: the block computing and the next
#: one arriving.
DEFAULT_WINDOW = 2
MAX_WINDOW = 6
#: Allocator fragmentation between the plane's budget and torch's activations. Small because a
#: torch OOM at a block boundary lowers the budget and retries (§4, activations first).
MARGIN = 64 << 20
#: Room a GPU library is given when its call is refused: a first launch takes memory for
#: itself (cuBLAS 36 MiB for a bf16 GEMM, 6 MiB more for fp16; cuDNN 2 MiB for a convolution;
#: measured on an L4 with CUDA 13).
LIBRARY_ROOM = 64 << 20
#: The event model's fixed costs: one copy's issue, and one block's acquire/release hook.
COPY_OVERHEAD_S = 20e-6
HOOK_S = 15e-6


@dataclass(frozen=True, slots=True)
class Layout:
    """One weight set as the plane holds it: `common` and each no-split block, in bytes, in
    address order. A component without declared blocks is `common` alone."""

    common: int
    blocks: tuple[int, ...] = ()
    #: (common, largest region) at the sub-block grain, when that is finer than `blocks`
    fine: tuple[int, int] | None = None

    @property
    def total(self) -> int:
        return self.common + sum(self.blocks)

    @property
    def largest(self) -> int:
        return max(self.blocks, default=0)


@dataclass(frozen=True, slots=True)
class Residence:
    """One component's placement for one stage: the blocks that stay resident and the ones
    streamed `window` ahead through a `ring` of device bytes (both address-order indexes
    into `Layout.blocks`)."""

    resident: tuple[int, ...]
    streamed: tuple[int, ...]
    window: int
    ring: int = 0

    def vram_bytes(self, layout: Layout) -> int:
        """What this placement holds on the device: common, the resident set and the ring."""
        return layout.common + sum(layout.blocks[i] for i in self.resident) + self.ring


class BelowFloor(Exception):
    """The budget cannot hold even the non-streamable minimum. Carries both numbers, and the
    component that did not fit at its current grain."""

    def __init__(self, needed: int, available: int, detail: str, component: str = "") -> None:
        super().__init__(f"{detail}: needs {needed} B, {available} B available")
        self.needed = needed
        self.available = available
        self.component = component


def plane_budget(free: int, activation: int, context: int = 0, margin: int = MARGIN) -> int:
    """`free - activation growth - context`: the device bytes the plane may map."""
    return max(free - activation - context - margin, 0)


def pinned_total(available: int, pinned: int) -> int:
    """The most the pinned host tier may hold across the machine: half of what the host has
    for it (free before its tightest limit, plus what is pinned already). Pinned memory is
    unreclaimable; the other half stays for processes and the page cache, the tier's own
    fallback. Everyone who pins (each executor, the Worker's kept tiers) shares this one sum."""
    return (max(available, 0) + max(pinned, 0)) // 2


def floor(layout: Layout, activation: int = 0) -> int:
    """The least a stage can run in: common, a ring of the largest region, activations, at
    the finest grain the component offers. A component without blocks cannot stream, so all
    of it."""
    coarse = layout.common + layout.largest if layout.blocks else layout.total
    if layout.fine is not None:
        coarse = min(coarse, sum(layout.fine))
    return coarse + activation


def ring_bytes(sizes: Sequence[int], window: int) -> int:
    """The ring that streams `sizes` (execution order, cyclic) `window` ahead without waiting
    on anything but the release `window` positions back: room for any `window` consecutive
    regions plus one wrap's waste, and never more than the plane's `window` x largest. Sizes
    are spans (2 MiB multiples), which hold the plane's 4 KiB-aligned ring placements."""
    if not sizes:
        return 0
    largest = max(sizes)
    if window <= 1:
        return largest
    cycle = [sizes[i % len(sizes)] for i in range(len(sizes) + window - 1)]
    run = most = sum(cycle[:window])
    for i in range(window, len(cycle)):
        run += cycle[i] - cycle[i - window]
        most = max(most, run)
    return min(window * largest, most + largest)


@dataclass(frozen=True, slots=True)
class Costs:
    """A stage's measured streaming costs: GPU seconds per use of each block (address index),
    from the plane's per-region timing, and the achieved host-to-device rate. Empty or zero is
    unmeasured, never a constant stand-in."""

    compute: Mapping[int, float] = field(default_factory=dict)
    link_bps: float = 0.0


def simulate(
    sizes: Sequence[int],
    compute: Sequence[float],
    streamed: Sequence[bool],
    window: int,
    link_bps: float,
    passes: int = 4,
) -> float:
    """Seconds per steady pass of one stage over its blocks in execution order (C2's model).

    One copy stream, `window` positions in flight, issued in order (a `ring_bytes` ring never
    adds a wait): streamed use k starts copying at the later of the previous copy's end and
    the release of the use `window` before it, and takes
    `COPY_OVERHEAD_S + bytes / link`. A block starts after the previous one and, when streamed,
    after its copy; it holds the GPU for its compute plus `HOOK_S`. Copies run ahead across pass
    boundaries. The mean of passes 2..`passes` (the only pass when `passes` is 1).
    """
    released: list[float] = []
    copied = clock = 0.0
    starts = []
    for _ in range(passes):
        starts.append(clock)
        for size, seconds, stream in zip(sizes, compute, streamed, strict=True):
            if stream:
                use = len(released)
                begin = max(copied, released[use - window] if use >= window else 0.0)
                copied = begin + COPY_OVERHEAD_S + size / link_bps
                clock = max(clock, copied)
            clock += seconds + HOOK_S
            if stream:
                released.append(clock)
    starts.append(clock)
    spans = [b - a for a, b in pairwise(starts)]
    steady = spans[1:] or spans
    return sum(steady) / len(steady)


def residence(
    layout: Layout,
    budget: int,
    *,
    order: Sequence[int] = (),
    costs: Costs | None = None,
    cyclic: bool = True,
) -> Residence:
    """The fastest placement of one component that fits `budget`.

    `order` is the blocks' execution order once observed (address order before). For a cyclic
    stage (a denoise loop) every window up to `MAX_WINDOW` is tried with three resident sets --
    streamed blocks spread evenly by compute, a streamed tail, and the most compute per byte
    streamed first -- and C2's event model (`simulate`) picks the fastest. Spread wins wherever
    it matters: a streamed tail idles the copy stream during a resident prefix (SDXL, +17-37%).
    Unmeasured costs place a spread set at `DEFAULT_WINDOW`. A run-once stage is resident whole
    when it fits and streamed whole otherwise. Raises `BelowFloor` below common + the largest
    block.
    """
    count = len(layout.blocks)
    walk = list(order) if sorted(order) == list(range(count)) else list(range(count))
    if layout.total <= budget:
        return Residence(resident=tuple(walk), streamed=(), window=0)
    need = layout.common + layout.largest if layout.blocks else layout.total
    if need > budget:
        raise BelowFloor(need, budget, "the component does not fit the plane budget at its grain")
    sizes = [layout.blocks[i] for i in walk]
    costs = costs or Costs()
    known = [costs.compute[i] for i in walk if i in costs.compute]
    measured = costs.link_bps > 0 and bool(known)
    # A block not yet timed (it never ran under a cursor) takes the mean of those that were.
    typical = sum(known) / len(known) if known else 1.0
    compute = [costs.compute.get(i, typical) for i in walk] if measured else [1.0] * count
    if not cyclic:
        width = DEFAULT_WINDOW if layout.common + ring_bytes(sizes, DEFAULT_WINDOW) <= budget else 1
        return Residence((), tuple(walk), width, ring_bytes(sizes, width))
    best: tuple[float, int, int, Residence] | None = None
    widths = range(1, MAX_WINDOW + 1) if measured else (DEFAULT_WINDOW, 1)
    picks = (_spread(compute), _tail(count), _dense(sizes, compute))
    for width in widths:
        for pick in picks:
            streamed = _fit(layout, budget, width, sizes, pick)
            if streamed is None:
                continue
            placed = Residence(
                resident=tuple(i for p, i in enumerate(walk) if p not in streamed),
                streamed=tuple(i for p, i in enumerate(walk) if p in streamed),
                window=width,
                ring=ring_bytes([sizes[p] for p in sorted(streamed)], width),
            )
            flags = [p in streamed for p in range(count)]
            seconds = simulate(sizes, compute, flags, width, costs.link_bps) if measured else 0.0
            rank = (seconds, sum(sizes[p] for p in streamed), width, placed)
            if best is None or rank[:3] < best[:3]:
                best = rank
        if best is not None and not measured:
            break
    assert best is not None  # the window-1 floor fits, so streaming everything does
    return best[3]


def _fit(
    layout: Layout,
    budget: int,
    width: int,
    sizes: Sequence[int],
    pick: Callable[[int], list[int]],
) -> frozenset[int] | None:
    """The fewest streamed positions `pick` chooses (bisected) so common, the rest and the
    ring streaming them `width` ahead fit `budget`; None when none do."""
    total = sum(sizes)

    def fits(count: int) -> frozenset[int] | None:
        chosen = pick(count)
        ring = ring_bytes([sizes[p] for p in sorted(chosen)], width)
        held = layout.common + total - sum(sizes[p] for p in chosen) + ring
        return frozenset(chosen) if held <= budget else None

    low, high = 1, len(sizes)
    found = fits(high)
    if found is None:
        return None
    while low < high:
        middle = (low + high) // 2
        if (chosen := fits(middle)) is not None:
            found, high = chosen, middle
        else:
            low = middle + 1
    return found


def _spread(compute: Sequence[float]) -> Callable[[int], list[int]]:
    """`count` positions spaced evenly by cumulative compute."""
    total, ends, running = sum(compute), [], 0.0
    for seconds in compute:
        running += seconds
        ends.append(running - seconds / 2)

    def pick(count: int) -> list[int]:
        chosen: list[int] = []
        taken: set[int] = set()
        for j in range(count):
            target = (j + 0.5) * total / count
            position = min(
                (p for p in range(len(ends)) if p not in taken), key=lambda p: abs(ends[p] - target)
            )
            chosen.append(position)
            taken.add(position)
        return chosen

    return pick


def _tail(count_all: int) -> Callable[[int], list[int]]:
    return lambda count: list(range(count_all - count, count_all))


def _dense(sizes: Sequence[int], compute: Sequence[float]) -> Callable[[int], list[int]]:
    """The positions whose compute best hides their own copy, first."""
    ranked = sorted(range(len(sizes)), key=lambda p: -compute[p] / max(sizes[p], 1))
    return lambda count: ranked[:count]


def stage(
    layouts: Mapping[str, Layout],
    budget: int,
    *,
    orders: Mapping[str, Sequence[int]] | None = None,
    costs: Mapping[str, Costs] | None = None,
    cyclic: Collection[str] = (),
) -> dict[str, Residence]:
    """Every declared component's residence for one stage, inside one plane budget.

    Each component first gets its floor; what is left goes to cyclic components first
    (they repeat, so every resident byte saves a copy per pass), then to run-once ones, which
    are resident only when whole. Components outside the stage are not counted: they are
    evictable, lower priority, and the plane unmaps them as this stage's wants arrive.
    """
    orders, costs = orders or {}, costs or {}
    need = {name: floor(layout) for name, layout in layouts.items()}
    if sum(need.values()) > budget:
        raise BelowFloor(
            sum(need.values()),
            budget,
            "the stage's common weights and its largest regions do not fit the plane budget",
        )
    spare = budget - sum(need.values())
    placed: dict[str, Residence] = {}
    for name in sorted(layouts, key=lambda n: (n not in cyclic, -layouts[n].total, n)):
        layout = layouts[name]
        try:
            chosen = residence(
                layout,
                need[name] + spare,
                order=orders.get(name, ()),
                costs=costs.get(name),
                cyclic=name in cyclic,
            )
        except BelowFloor as exc:
            raise BelowFloor(
                exc.needed, exc.available, "a component does not fit at its grain", name
            ) from None
        spare -= max(chosen.vram_bytes(layout) - need[name], 0)
        placed[name] = chosen
    return placed


def decode_mode(free: int, evictable: int, untiled_bytes: int) -> Literal["untiled", "tiled"]:
    """Untiled when its activations fit what is free plus what the plane may unmap."""
    return "untiled" if untiled_bytes <= free + evictable else "tiled"


def batch_fits(free: int, evictable: int, extra_bytes: int) -> bool:
    """Whether a batched call (e.g. cond+uncond as one batch) fits, counting evictable weights
    as free. Decided before the stage and recorded; never switched mid-request."""
    return extra_bytes <= free + evictable
