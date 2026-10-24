"""A pure cut site's level from its op's live shapes and executor-local available memory.

Arithmetic on the operands, never an OOM probe, so a first try fits: `need(level)` is the bytes
live at a piece's peak, the whole output included, each term rounded up to the allocation
granularity. Level L cuts a site's units into 2^L pieces (`pieces`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

#: a piece's bytes for its unit count (tokens, rows, images, heads or query rows), each term
#: in whole allocation granules of the given size
type Per = Callable[[int, int], int]
#: a level's bytes (2^level pieces), in whole granules of the given size
type Need = Callable[[int, int], int]


class Room(Protocol):
    """Local free bytes and eligible cached weights outside/inside the current scope's plan.
    Active weights and graph-retained tensors are excluded. The pure-cut caller reclaims any
    selected cached bytes before executing; the allocator itself owns no eviction policy."""

    @property
    def free(self) -> int: ...

    @property
    def cold(self) -> int: ...

    @property
    def warm(self) -> int: ...

    @property
    def granularity(self) -> int: ...

    @property
    def floor(self) -> int: ...


def pieces(units: int, level: int) -> int:
    """Units a piece holds at `level` (2^level pieces, the last one short)."""
    return -(-units // (1 << level))


def levels(units: int) -> int:
    """The finest level: one unit a piece."""
    return max(units - 1, 0).bit_length()


def levelled(units: int, per: Per) -> Need:
    """`per`'s piece bytes at each level of `units` cut into 2^level pieces."""
    return lambda level, granularity: per(pieces(units, level), granularity)


def choose(finest: int, need: Need, room: Room) -> tuple[int, str]:
    """The coarsest level of 0..`finest` that fits, and its tier: within free and cold bytes
    ("cold": no weight of the step evicted), else within everything revocable less the streaming
    floor ("warm"), else the finest ("floor"), where only the physical floor refuses."""
    for tier, room_bytes in (
        ("cold", room.free + room.cold),
        ("warm", room.free + room.cold + room.warm - room.floor),
    ):
        level = next(
            (n for n in range(finest + 1) if need(n, room.granularity) <= room_bytes), None
        )
        if level is not None:
            return level, tier
    return finest, "floor"


def descend[T](
    finest: int,
    need: Need,
    room: Callable[[], Room],
    run: Callable[[int], T],
    refused: Callable[[BaseException], bool],
) -> tuple[T, int, int, str]:
    """`run(level)` at the level `choose` gives against the room now; a refusal (`refused`) at a
    level above the finest re-reads the room and runs at least one level finer. Each step is a
    smaller need, never the same size again, so at most `levels` steps; the finest's refusal is
    the floor and raises. Returns the result, its level, the steps taken and the room's tier."""
    floor, steps = 0, 0
    while True:
        chosen, tier = choose(finest, need, room())
        level = max(chosen, floor)
        try:
            return run(level), level, steps, tier
        except Exception as exc:
            if level >= finest or not refused(exc):
                raise
            floor, steps = level + 1, steps + 1


def agreed(proposals: Iterable[int]) -> int:
    """A group's level: the finest any GPU needs, so every GPU crosses its collectives alike."""
    return max(proposals, default=0)


def rounded(nbytes: int, granularity: int) -> int:
    """`nbytes` in whole allocation granules."""
    return -(-nbytes // granularity) * granularity if nbytes > 0 else 0


def peak(granularity: int, out: int, *phases: Sequence[int]) -> int:
    """The output held throughout plus the largest phase of terms live together, each term
    rounded to the granularity."""
    terms = [sum(rounded(t, granularity) for t in phase) for phase in phases]
    return rounded(out, granularity) + max(terms, default=0)


@dataclass(frozen=True, slots=True)
class Ffn:
    """A token-wise feed-forward: `din` in, a projection to `projected` features (`2 * inner`
    when gated), `temps` live `n x inner` activation temporaries (1 fused, 2-3 for composite
    GELU-tanh or an unfused SiLU-mul), a down projection to `dout`. `quantized`: the encoding
    leaf quantizes each linear's input rows (int8/fp8: a byte an element and an fp32 scale a
    row)."""

    din: int
    dout: int
    inner: int
    projected: int
    temps: int
    itemsize: int
    quantized: bool

    def need(self, tokens: int) -> Per:
        b = self.itemsize

        def piece(n: int, granularity: int) -> int:
            def quant(width: int) -> int:
                return n * width + 4 * n if self.quantized else 0

            return peak(
                granularity,
                tokens * self.dout * b,
                (quant(self.din), n * self.projected * b),
                (n * self.projected * b, self.temps * n * self.inner * b),
                (n * self.inner * b, quant(self.inner), n * self.dout * b),
            )

        return piece


def norm(rows: int, row: int, itemsize: int, intermediate: int) -> Per:
    """A norm over whole rows: the output held throughout, a piece's output and its op's own
    `intermediate` bytes an element (0 fused; a composite fp32 rms_norm holds an upcast)."""
    return lambda n, granularity: peak(
        granularity, rows * row * itemsize, (n * row * itemsize, n * row * intermediate)
    )


def conv(out: int, per_unit_in: int, per_unit_out: int, halo: int, workspace: int) -> Per:
    """A conv in bands: the output held throughout; a band's padded input copy (`per_unit_in`
    a unit plus its `halo` rows, 0 when the band is a view), its output, and cuDNN's workspace
    bound for its algorithm."""
    return lambda n, granularity: peak(
        granularity,
        out,
        (n * per_unit_in + halo if per_unit_in else 0, n * per_unit_out, workspace),
    )


@dataclass(frozen=True, slots=True)
class Attention:
    """Attention over [batch, rows, heads, dim] queries and `keys` keys of `kv_heads`: the
    backend's scratch (`math`: the scores and their softmax; `sage`: quantized q and k), and the
    whole K/V layout copies made inside the cut (`kv_copies` of each; 0 when hoisted)."""

    batch: int
    rows: int
    heads: int
    kv_heads: int
    keys: int
    dim: int
    itemsize: int
    backend: str
    kv_copies: int

    def need(self, heads: int, rows: int, granularity: int) -> int:
        """A piece of `heads` query heads by `rows` query rows."""
        b, kv = self.itemsize, self.batch * self.keys * self.kv_heads * self.dim
        scratch = {
            "math": self.batch * heads * rows * self.keys * (4 + b),
            "sage": self.batch * (rows * heads + self.keys * self.kv_heads) * (self.dim + 4),
        }.get(self.backend, 0)
        return peak(
            granularity,
            self.batch * self.rows * self.heads * self.dim * b,
            (
                self.batch * rows * heads * self.dim * b,
                self.batch * heads * rows * 4,
                scratch,
                *[kv * b] * (2 * self.kv_copies),
            ),
        )


def fitted(small: tuple[int, int], large: tuple[int, int]) -> Callable[[int], int]:
    """A `split` op's bytes for a window of w elements, from two measured windows (elements,
    peak bytes less the output): `a*w + c*w^2` through both, so an op quadratic in its window
    (attention inside it) scales as it does; never below the line through the larger."""
    (w1, p1), (w2, p2) = small, large
    assert 0 < w1 < w2, (small, large)
    c = max((p2 * w1 - p1 * w2) / (w1 * w2 * (w2 - w1)), 0.0)
    a = (p2 - c * w2 * w2) / w2
    return lambda w: math.ceil(a * w + c * w * w)
