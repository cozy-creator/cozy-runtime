"""A pure cut site's level from its op's live shapes, against the room the agent reports
(design/varena-leases.md §8 "No learning loop"; status/s4-leases-runtime-plan.md §5).

Arithmetic on the operands, never an OOM probe, so a first try fits: `need(level)` is the bytes
live at a piece's peak, the whole output included, each term rounded up to the allocation
granularity. Level L cuts a site's units into 2^L pieces (`pieces`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

#: a piece's bytes for its unit count (tokens, rows, images, heads or query rows)
type Need = Callable[[int], int]


@dataclass(frozen=True, slots=True)
class Room:
    """What this process may hold now (`r.room(step)`): its float and the manager's free (net of
    the driver headroom), revocable bytes outside the step's plan sets (`cold`), its own inside
    them (`warm`), and the streaming floor the warm tier leaves: SLOTS times the largest block it
    would evict (`floor`), so those blocks can bind again within the step."""

    free: int
    cold: int
    warm: int
    floor: int


def pieces(units: int, level: int) -> int:
    """Units a piece holds at `level` (2^level pieces, the last one short)."""
    return -(-units // (1 << level))


def levels(units: int) -> int:
    """The finest level: one unit a piece."""
    return max(units - 1, 0).bit_length()


def choose(units: int, need: Need, room: Room) -> tuple[int, str]:
    """The coarsest level whose piece fits, and its tier: within free and cold bytes ("cold": no
    weight of the step evicted), else within everything revocable less the streaming floor
    ("warm"), else the finest ("floor"), where only the physical floor refuses."""
    finest = levels(units)
    for tier, room_bytes in (
        ("cold", room.free + room.cold),
        ("warm", room.free + room.cold + room.warm - room.floor),
    ):
        level = next((n for n in range(finest + 1) if need(pieces(units, n)) <= room_bytes), None)
        if level is not None:
            return level, tier
    return finest, "floor"


def descend[T](
    units: int,
    need: Need,
    room: Callable[[], Room],
    run: Callable[[int], T],
    refused: Callable[[BaseException], bool],
) -> tuple[T, int, int]:
    """`run(level)` at the level `choose` gives against the room now; a refusal (`refused`) at a
    level above the finest re-reads the room and runs at least one level finer. Each step is a
    smaller need, never the same size again, so at most `levels` steps; the finest's refusal is
    the floor and raises. Returns the result, its level and the steps taken."""
    finest, floor, steps = levels(units), 0, 0
    while True:
        level = max(choose(units, need, room())[0], floor)
        try:
            return run(level), level, steps
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

    def need(self, tokens: int, granularity: int) -> Need:
        b = self.itemsize

        def piece(n: int) -> int:
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


def norm(rows: int, row: int, itemsize: int, intermediate: int, granularity: int) -> Need:
    """A norm over whole rows: the output held throughout, a piece's output and its op's own
    `intermediate` bytes an element (0 fused; a composite fp32 rms_norm holds an upcast)."""
    return lambda n: peak(
        granularity, rows * row * itemsize, (n * row * itemsize, n * row * intermediate)
    )


def conv(
    out: int, per_unit_in: int, per_unit_out: int, halo: int, workspace: int, granularity: int
) -> Need:
    """A conv in bands: the output held throughout; a band's padded input copy (`per_unit_in`
    a unit plus its `halo` rows, 0 when the band is a view), its output, and cuDNN's workspace
    bound for its algorithm."""
    return lambda n: peak(
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
