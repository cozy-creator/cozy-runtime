"""A pure cut's allocation estimate from its op's live shapes.

`need(level)` lists what a piece's peak holds, the whole output included. It is an estimate:
native selection chooses from it, allocator reclaim covers a short one, and a measured peak
replaces it after a completed phase. Level L cuts a site's units into 2^L pieces (`pieces`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

#: a piece's allocations for its unit count (tokens, rows, images, heads or query rows)
type Per = Callable[[int], tuple[int, ...]]
#: a level's allocations (2^level pieces)
type Need = Callable[[int], tuple[int, ...]]


def pieces(units: int, level: int) -> int:
    """Units a piece holds at `level` (2^level pieces, the last one short)."""
    return -(-units // (1 << level))


def levels(units: int) -> int:
    """The finest level: one unit a piece."""
    return max(units - 1, 0).bit_length()


def levelled(units: int, per: Per) -> Need:
    """`per`'s piece allocations at each level of `units` cut into 2^level pieces."""
    return lambda level: per(pieces(units, level))


def peak(out: int, *phases: Sequence[int]) -> tuple[int, ...]:
    """The output held throughout, and the largest phase of terms live together."""
    return (out, *max(phases, key=sum, default=()))


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

        def piece(n: int) -> tuple[int, ...]:
            def quant(width: int) -> tuple[int, ...]:
                return (n * width, 4 * n) if self.quantized else ()

            up = n * self.projected * b
            return peak(
                tokens * self.dout * b,
                (*quant(self.din), up),
                (up, *[n * self.inner * b] * self.temps),
                (n * self.inner * b, *quant(self.inner), n * self.dout * b),
            )

        return piece


def norm(rows: int, row: int, itemsize: int, intermediate: int) -> Per:
    """A norm over whole rows: the output held throughout, a piece's output and its op's own
    `intermediate` bytes an element (0 fused; a composite fp32 rms_norm holds an upcast)."""
    return lambda n: peak(rows * row * itemsize, (n * row * itemsize, n * row * intermediate))


@dataclass(frozen=True, slots=True)
class Attention:
    """Attention over [batch, rows, heads, dim] queries and `keys` keys of `kv_heads`, with one
    K/V layout copy made inside the kernel."""

    batch: int
    rows: int
    heads: int
    kv_heads: int
    keys: int
    dim: int
    itemsize: int

    def need(self, heads: int, rows: int) -> tuple[int, ...]:
        """A piece of `heads` query heads by `rows` query rows."""
        b, group = self.itemsize, max(self.heads // self.kv_heads, 1)
        kv = self.batch * self.keys * max(heads // group, 1) * self.dim * b
        return peak(
            self.batch * self.rows * self.heads * self.dim * b,
            (self.batch * rows * heads * self.dim * b, self.batch * heads * rows * 4, kv, kv),
        )
