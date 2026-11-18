"""Which rung a step takes, judged before it runs from what its first step measured (cr-336).

A step key's first step runs eagerly with nothing leased and is measured. `judge` then says whether
its graph is admitted: the weights the step reads, the graph's pool, what is live, one more eager
step (the check that runs beside the pool) and what the step writes of its rows' K/V and state
must fit the card together. Otherwise the key stays eager and its weights stream by the plan.
Either way it says how much of the call's declared K/V the card may hold; the rest lives off the
card (`engine.kv`). That is a bound, not a reservation: K/V pages are mapped as rows grow, and
weights give way to them then. A card that can't hold one eager step is `Short`, with what one
step needs. Every number is measured or declared: no reserve, no threshold.

`judge` is a function of its arguments. Its caller keeps the answer per key for the executor's
life and asks again when the room grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class StepKey:
    """What a step's graph depends on. No context length and no window: lengths are device data."""

    kind: Literal["decode", "mixed"]
    rows: int
    tokens: int
    depth: int
    kernels: str
    lane: str
    layout_generation: int


@dataclass(frozen=True)
class StepMeasure:
    """A key's first step, run eagerly: its working peak above what was live when it began, what
    was live then (working memory; a K/V Cache's pages are not among it), the bytes of the
    weights a graph of it would lease, and the largest single unit among them (a streamed step
    holds one unit at a time)."""

    peak_above_entry: int
    live: int
    leased_weights: int
    largest_unit: int


@dataclass(frozen=True)
class CardFacts:
    """The card as varena reports it now (`vt.room`): `capacity`, the card or its cap less what
    other programs hold, and the weight bytes other models of this process hold for use now."""

    capacity: int
    pinned_elsewhere: int


@dataclass(frozen=True)
class KvNeed:
    """A call's K/V and state as its manager declared them (`engine.kv`): the exact upper bound
    of the call, and of that, what a step writes (running rows' tail blocks and state slots),
    which never leaves the card."""

    declared: int
    written: int


class PoolMemory(Protocol):
    """Each graph's capture pool as this machine measured it (`execution.rungs`)."""

    def pool(self, key: StepKey) -> int:
        """The pool `key`'s capture measured here, 0 when none has."""
        ...

    def guess(self, peak: int) -> int:
        """A first pool for a capture whose eager step peaked at `peak` (`vt.pool_bytes`)."""
        ...


@dataclass(frozen=True)
class Graph:
    """The key's step is captured and replayed, in a pool of `pool_bytes`; the card holds
    `kv_on_card` bytes of the call's K/V."""

    pool_bytes: int
    kv_on_card: int


@dataclass(frozen=True)
class Eager:
    """The key's step runs as written, its weights streamed by the plan; the card holds
    `kv_on_card` bytes of the call's K/V. `reason` carries the numbers."""

    reason: str
    kv_on_card: int


@dataclass(frozen=True)
class Short:
    """The card can't hold one eager step: `floor` is what one needs (what is live, the step's
    peak, the K/V it writes and the largest weight unit it reads) and `room` what the card has."""

    floor: int
    room: int


def judge(
    key: StepKey, measure: StepMeasure, card: CardFacts, pools: PoolMemory, kv: KvNeed
) -> Graph | Eager | Short:
    """`key`'s rung, and the bytes of the call's K/V the card may hold.

    Graph: leased weights + pool + live + one eager step + the K/V the step writes fit the card.
    K/V takes what that leaves, up to the call's declared need. A graph with part of its K/V off
    the card is still chosen over an eager step: eager would put those K/V bytes on the card by
    taking as many weight bytes off it, to be read again every token.

    Eager: nothing is leased. One step needs what is live, its measured peak, the K/V it writes
    and the largest weight unit it reads. K/V takes what that leaves: it is read whole every step,
    and a weight that streams is read again whether or not K/V sits beside it."""
    room = card.capacity - card.pinned_elsewhere
    step, live = measure.peak_above_entry, measure.live
    pool = pools.pool(key) or pools.guess(step)
    fixed = measure.leased_weights + pool + live + step
    if fixed + kv.written <= room:
        return Graph(pool, min(kv.declared, room - fixed))
    floor = live + step + kv.written + measure.largest_unit
    if floor > room:
        return Short(floor, room)
    reason = (
        f"graph: {_gib(measure.leased_weights)} GiB weights + {_gib(pool)} pool + {_gib(live)} "
        f"live + {_gib(step)} step + {_gib(kv.written)} written K/V > {_gib(room)} GiB: eager"
    )
    return Eager(reason, min(kv.declared, room - floor + kv.written))


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.1f}"
