"""The decode cores (cozytensors/1 spec §6, building blocks) on GPU and CPU: affine, table and
block structs, each one generic Triton kernel and one eager torch reference that agree bit for
bit with the spec's f32 arithmetic and with TensorFS's Rust decoder. Outside the `encoding`
package, so it imports alone (Triton's interpreter cannot import the package's GEMM kernels).

A scheme is an `Adapter` (`encoding/adapters.py`) that maps its parts and params onto one core
call:
- `affine`: `b`-bit codes unpacked from words (`Packing`), then `(q - z) x s`, `q x s + c` or
  `q x s` with the scale and term on a `Grid`;
- `table`: `T[q] x s`, optionally then `x scale_2` or with `s / scale_2`;
- `blocks`: fixed-size structs, each decoded by a scheme's `@triton.jit` function (on GPU) or
  its torch function (eager).

Every kernel rounds its f32 result once into the destination (`round_L`: nearest even, NaN
canonical) with integer bit operations, so the GPU and Triton's interpreter round alike, and
launches with `enable_fp_fusion=False`: a fused multiply-add would round once where the spec
rounds twice.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as Device

try:
    import triton
    import triton.language as tl
except ImportError:  # a host without Triton decodes eagerly
    triton = None

#: The kernels run here (CUDA, or Triton's interpreter on CPU).
HAS_TRITON = triton is not None


FILLS = ("word", "dense", "planar")
GRIDS = ("tensor", "row", "col", "blocks")
LAYOUTS = ("row_major", "transposed", "cublas_128x4")
#: Destination dtypes (torch names) by the kernels' `OUT` code.
OUTS = ("float32", "bfloat16", "float16")
BLOCK = 1024


@dataclass(frozen=True, slots=True)
class Packing:
    """A packed part: the `[R, C]` view's codes, stored as `[R, C]` or transposed `[C, R]`,
    packed along `axis` of that stored array into little-endian words of `word` bits."""

    bits: int
    word: int
    transposed: bool = False
    axis: int = 1
    fill: str = "word"
    #: `low_first`, `high_first`, or slot -> code (AWQ: (0, 2, 4, 6, 1, 3, 5, 7))
    order: str | tuple[int, ...] = "low_first"
    signed: bool = False
    #: an unsigned code is `raw - offset`
    offset: int = 0

    @property
    def per(self) -> int:
        return self.word // self.bits

    def words(self, n: int) -> int:
        """Words in a lane of `n` codes."""
        if self.fill == "word":
            return -(-n // self.per)
        if self.fill == "dense":
            return -(-n * self.bits // self.word)
        if n % self.per:
            raise ValueError(f"{n} codes do not fill planes of {self.per}")
        return n // self.per

    def slots(self) -> tuple[int, ...]:
        """The slot of each code index within a word."""
        per = self.per
        if self.order == "low_first":
            return tuple(range(per))
        if self.fill == "dense":
            raise ValueError("a dense bit stream is low first")
        if self.order == "high_first":
            return tuple(range(per - 1, -1, -1))
        order = self.order if isinstance(self.order, tuple) else ()
        if sorted(order) != list(range(per)):
            raise ValueError(f"slots {self.order} are not a permutation of {per}")
        return tuple(order.index(e) for e in range(per))

    def stored(self, rows: int, cols: int) -> tuple[int, int, int]:
        """(words per lane, lanes, codes per lane) of the stored array."""
        a, b = (cols, rows) if self.transposed else (rows, cols)
        n, along = (a, b) if self.axis == 0 else (b, a)
        return self.words(n), along, n


@dataclass(frozen=True, slots=True)
class Grid:
    """Where element (i, j) finds its scale: one for the tensor, one per row or column, or row
    blocks of `size` (`W = ceil(C / size)` to a row, block `j // size` or a `g_idx`) stored
    `row_major` `[R, W]`, `transposed` `[W, R]` or `cublas_128x4`."""

    kind: str = "tensor"
    size: int = 1
    layout: str = "row_major"


TENSOR, ROW, COL = Grid("tensor"), Grid("row"), Grid("col")


def blocks_of(size: int, layout: str = "row_major") -> Grid:
    return Grid("blocks", size, layout)


class CodeOutsideTable(ValueError):
    """A code with no entry in its table: the spec refuses the decode."""


class Cores:
    """The three cores bound to torch and to a path: the Triton kernels (`kernel=True`: CUDA
    tensors, or CPU tensors under Triton's interpreter) or the eager torch reference."""

    def __init__(self, torch: ModuleType, kernel: bool) -> None:
        if kernel and not HAS_TRITON:
            raise RuntimeError("Triton does not import here; decode with kernel=False")
        self.torch, self.kernel = torch, kernel

    def affine(
        self,
        words: Tensor,
        packing: Packing,
        rows: int,
        cols: int,
        scale: Tensor,
        grid: Grid,
        out: Tensor,
        *,
        zero: Tensor | None = None,
        bias: Tensor | None = None,
        g_idx: Tensor | None = None,
        post: Tensor | None = None,
    ) -> None:
        """`out` (`R*C` elements, f32/bf16/f16) = the affine decode of the packed codes, then
        `x post` (one element, or one per column) when given."""
        if zero is not None and bias is not None:
            raise ValueError("an affine decode has a zero or a bias, not both")
        torch = self.torch
        scale = scale.reshape(-1).to(torch.float32)
        stored = zero if zero is not None else bias
        term = None if stored is None else stored.reshape(-1).to(torch.float32)
        if not self.kernel:
            q = unpack(torch, words, packing, rows, cols).to(torch.float32)
            at = _grid_index(torch, grid, rows, cols, g_idx, scale.device)
            s = scale[at]
            if term is None:
                v = q * s
            elif zero is not None:
                v = (q - term[at]) * s
            else:
                v = q * s + term[at]
            round_into(torch, _post(torch, v, post, cols), out)
            return
        nw, _, _ = packing.stored(rows, cols)
        slots = torch.tensor(
            packing.slots() if packing.fill != "dense" else (0,),
            dtype=torch.int32,
            device=out.device,
        )
        _affine_kernel[_launch(rows * cols)](
            _bits(torch, out),
            _words(torch, words, packing.word),
            slots,
            scale,
            scale if term is None else term,
            _g_idx(torch, g_idx, scale),
            rows * cols,
            rows,
            cols,
            nw,
            packing.offset,
            grid.size,
            *_packing_codes(packing),
            *_grid_codes(grid, g_idx),
            *_post_args(torch, post, cols, scale),
            TERM=0 if term is None else (1 if zero is not None else 2),
            OUT=_out_code(torch, out),
            BLOCK=BLOCK,
            enable_fp_fusion=False,
        )

    def table(
        self,
        words: Tensor,
        packing: Packing,
        rows: int,
        cols: int,
        values: Tensor,
        scale: Tensor,
        grid: Grid,
        out: Tensor,
        *,
        scale_2: Tensor | None = None,
        divide: bool = False,
        g_idx: Tensor | None = None,
        post: Tensor | None = None,
    ) -> None:
        """`out` = `T[q] x s`, then `x scale_2`; or `T[q] x (s / scale_2)` with `divide`; then
        `x post` (one element, or one per column) when given."""
        torch = self.torch
        values = values.reshape(-1).to(torch.float32)
        scale = scale.reshape(-1).to(torch.float32)
        second = None if scale_2 is None else scale_2.reshape(1).to(torch.float32)
        if not self.kernel:
            q = unpack(torch, words, packing, rows, cols)
            if q.numel() and (int(q.min()) < 0 or int(q.max()) >= values.numel()):
                raise CodeOutsideTable(f"a code is outside a table of {values.numel()}")
            s = scale[_grid_index(torch, grid, rows, cols, g_idx, scale.device)]
            if second is not None and divide:
                s = s / second
            v = values[q] * s
            if second is not None and not divide:
                v = v * second
            round_into(torch, _post(torch, v, post, cols), out)
            return
        nw, _, _ = packing.stored(rows, cols)
        slots = torch.tensor(
            packing.slots() if packing.fill != "dense" else (0,),
            dtype=torch.int32,
            device=out.device,
        )
        error = torch.zeros(1, dtype=torch.int32, device=out.device)
        _table_kernel[_launch(rows * cols)](
            _bits(torch, out),
            _words(torch, words, packing.word),
            slots,
            values,
            values.numel(),
            scale,
            scale if second is None else second,
            _g_idx(torch, g_idx, scale),
            error,
            rows * cols,
            rows,
            cols,
            nw,
            packing.offset,
            grid.size,
            *_packing_codes(packing),
            *_grid_codes(grid, g_idx),
            *_post_args(torch, post, cols, scale),
            SCALE_2=0 if second is None else (2 if divide else 1),
            OUT=_out_code(torch, out),
            BLOCK=BLOCK,
            enable_fp_fusion=False,
        )
        if int(error.item()):
            raise CodeOutsideTable(f"a code is outside a table of {values.numel()}")

    def blocks(
        self,
        data: Tensor,
        n: int,
        size: int,
        out: Tensor,
        *,
        kernel_fn: object = None,
        eager_fn: Callable[[ModuleType, Tensor], Tensor] | None = None,
        post: Tensor | None = None,
    ) -> None:
        """`out` from `U8` structs of `size` bytes holding `n` elements each, in row order.
        `kernel_fn(base, element, mask)` is a `@triton.jit` function returning one element's
        f32 value from its struct's bytes at `base`; `eager_fn(torch, structs)` maps every
        struct (`[count, size]` u8) to its `[count, n]` f32 values. `post`: a factor of one element
        multiplying every value last."""
        torch = self.torch
        total = out.numel()
        if not self.kernel:
            assert eager_fn is not None, "an eager decode needs its struct function"
            structs = data.reshape(-1).view(torch.uint8).reshape(-1, size)
            round_into(torch, _post(torch, eager_fn(torch, structs).reshape(-1), post, total), out)
            return
        _blocks_kernel[_launch(total)](
            _bits(torch, out),
            data.reshape(-1).view(torch.uint8),
            total,
            *_post_args(torch, post, total, data),
            N=n,
            S=size,
            DECODE=kernel_fn,
            OUT=_out_code(torch, out),
            BLOCK=BLOCK,
            enable_fp_fusion=False,
        )


# ------------------------------------------------------------------- the eager reference


def unpack(torch: ModuleType, words: Tensor, p: Packing, rows: int, cols: int) -> Tensor:
    """The view's codes, int64 [R, C], on the words' device: spec §6's bit packing, one lane per
    row of words. Also how a scheme reads its small packed parts (zeros)."""
    nw, along, n = p.stored(rows, cols)
    raw: Tensor = _words(torch, words, p.word).reshape(-1).to(torch.int64) & ((1 << p.word) - 1)
    if raw.numel() != nw * along:
        raise ValueError(f"{raw.numel()} words are not {nw}x{along}")
    lanes = raw.reshape(nw, along).t() if p.axis == 0 else raw.reshape(along, nw)
    t: Tensor = torch.arange(n, device=raw.device)
    b, w = p.bits, p.word
    if p.fill == "dense":
        k, off = (t * b) // w, (t * b) % w
        spill = off + b > w
        lo = lanes[:, k] >> off
        hi: Tensor = torch.where(spill, lanes[:, (k + 1).clamp(max=nw - 1)] << (w - off), 0)
        codes = (lo | hi) & ((1 << b) - 1)
    else:
        k, e = (t // p.per, t % p.per) if p.fill == "word" else (t % nw, t // nw)
        slot: Tensor = torch.tensor(p.slots(), device=raw.device)[e]
        codes = (lanes[:, k] >> (slot * b)) & ((1 << b) - 1)
    codes = codes - ((codes >> (b - 1)) << b) if p.signed else codes - p.offset
    stored = codes.t() if p.axis == 0 else codes
    return stored.t() if p.transposed else stored


def _grid_index(
    torch: ModuleType, g: Grid, rows: int, cols: int, g_idx: Tensor | None, device: Device
) -> Tensor:
    i: Tensor = torch.arange(rows, device=device).reshape(rows, 1)
    j: Tensor = torch.arange(cols, device=device).reshape(1, cols)
    if g.kind == "tensor":
        zero: Tensor = torch.zeros(rows, cols, dtype=torch.int64, device=device)
        return zero
    if g.kind == "row":
        return i.expand(rows, cols)
    if g.kind == "col":
        return j.expand(rows, cols)
    w = -(-cols // g.size)
    b = (
        j // g.size
        if g_idx is None
        else g_idx.reshape(1, cols).to(device=device, dtype=torch.int64)
    )
    if g.layout == "row_major":
        return i * w + b
    if g.layout == "transposed":
        return b * rows + i
    return ((i // 128) * -(-w // 4) + b // 4) * 512 + (i % 32) * 16 + (i % 128 // 32) * 4 + b % 4


def round_into(torch: ModuleType, v: Tensor, out: Tensor) -> None:
    """`round_L` into `out` by the kernels' own bit operations: nearest even, NaN canonical
    (torch's own casts keep NaN payloads)."""
    v = v.reshape(-1).to(torch.float32)
    nan = v.isnan()
    bits = v.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    if out.dtype == torch.float32:
        rounded = torch.where(nan, 0x7FC00000, bits)
    elif out.dtype == torch.bfloat16:
        rounded = torch.where(nan, 0x7FC0, (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)
    else:
        half = v.to(torch.float16).view(torch.int16).to(torch.int64) & 0xFFFF
        rounded = torch.where(nan, 0x7E00, half)
    wide = out.element_size() == 4
    signed = rounded - (rounded >> (31 if wide else 15) << (32 if wide else 16))
    _bits(torch, out).copy_(signed.to(torch.int32 if wide else torch.int16))


# ----------------------------------------------------------------------- the kernels


def _launch(total: int) -> tuple[int]:
    return (max(1, -(-total // BLOCK)),)


def _out_code(torch: ModuleType, out: Tensor) -> int:
    return OUTS.index(str(out.dtype).removeprefix("torch."))


def _bits(torch: ModuleType, out: Tensor) -> Tensor:
    """The destination as integers of its width: the kernels store rounded bits."""
    return out.view(torch.int32 if out.element_size() == 4 else torch.int16).reshape(-1)


def _words(torch: ModuleType, words: Tensor, width: int) -> Tensor:
    view = {8: torch.uint8, 16: torch.int16, 32: torch.int32}[width]
    return words.reshape(-1).view(torch.uint8).view(view)


def _g_idx(torch: ModuleType, g_idx: Tensor | None, placeholder: Tensor) -> Tensor:
    return placeholder if g_idx is None else g_idx.reshape(-1).to(torch.int32)


def _post(torch: ModuleType, v: Tensor, post: Tensor | None, cols: int) -> Tensor:
    """`v x post`: one factor, or one per column of the `[R, C]` view."""
    if post is None:
        return v
    factor = post.reshape(-1).to(device=v.device, dtype=torch.float32)
    return (v.reshape(-1, cols) * (factor if factor.numel() == cols else factor[0])).reshape(-1)


def _post_args(
    torch: ModuleType, post: Tensor | None, cols: int, placeholder: Tensor
) -> tuple[Tensor, int]:
    """(post pointer, POST code): 0 none, 1 one factor, 2 one per column."""
    if post is None:
        return placeholder, 0
    factor = post.reshape(-1).to(torch.float32)
    return factor, 2 if factor.numel() == cols and cols != 1 else 1


def _packing_codes(p: Packing) -> tuple[int, ...]:
    order = {"low_first": 0, "high_first": 1}.get(p.order, 2) if isinstance(p.order, str) else 2
    return (p.bits, p.word, int(p.transposed), p.axis, FILLS.index(p.fill), order, int(p.signed))


def _grid_codes(g: Grid, g_idx: Tensor | None) -> tuple[int, int, int]:
    return GRIDS.index(g.kind), LAYOUTS.index(g.layout), int(g_idx is not None)


if triton is not None:

    @triton.jit  # type: ignore[untyped-decorator]
    def _word(words, at, mask, WORD: tl.constexpr):  # type: ignore[no-untyped-def]
        w = tl.load(words + at, mask=mask, other=0)
        if WORD == 32:
            u = w.to(tl.uint32, bitcast=True)
        elif WORD == 16:
            u = w.to(tl.uint16, bitcast=True).to(tl.uint32)
        else:
            u = w.to(tl.uint32)
        return u

    @triton.jit  # type: ignore[untyped-decorator]
    def _code(  # type: ignore[no-untyped-def]
        words,
        slots,
        i,
        j,
        rows,
        cols,
        nw,
        offset,
        valid,
        BITS: tl.constexpr,
        WORD: tl.constexpr,
        TRANSPOSED: tl.constexpr,
        AXIS: tl.constexpr,
        FILL: tl.constexpr,
        ORDER: tl.constexpr,
        SIGNED: tl.constexpr,
    ):
        """Code q[i, j] of the view (spec §6 bit packing)."""
        if TRANSPOSED:
            x = j
            y = i
            nb = rows
        else:
            x = i
            y = j
            nb = cols
        if AXIS == 0:
            t = x
            lane = y
        else:
            t = y
            lane = x
        PER: tl.constexpr = WORD // BITS
        MASK: tl.constexpr = (1 << BITS) - 1
        if FILL == 1:
            bit = t * BITS
            k = bit // WORD
            off = (bit % WORD).to(tl.uint32)
            if AXIS == 0:
                at = k * nb + lane
                nxt = at + nb
            else:
                at = lane * nw + k
                nxt = at + 1
            spill = valid & (off + BITS > WORD)
            lo = _word(words, at, valid, WORD) >> off
            hi = _word(words, nxt, spill, WORD) << tl.where(spill, WORD - off, 0)
            raw = (lo | hi) & MASK
        else:
            if FILL == 0:
                k = t // PER
                e = t % PER
            else:
                k = t % nw
                e = t // nw
            if ORDER == 0:
                slot = e
            elif ORDER == 1:
                slot = PER - 1 - e
            else:
                slot = tl.load(slots + e, mask=valid, other=0)
            if AXIS == 0:  # noqa: SIM108 - a constexpr branch, not a select
                at = k * nb + lane
            else:
                at = lane * nw + k
            raw = (_word(words, at, valid, WORD) >> (slot * BITS).to(tl.uint32)) & MASK
        q = raw.to(tl.int32)
        if SIGNED:  # noqa: SIM108 - a constexpr branch, not a select
            q = q - ((q >> (BITS - 1)) << BITS)
        else:
            q = q - offset
        return q

    @triton.jit  # type: ignore[untyped-decorator]
    def _grid(  # type: ignore[no-untyped-def]
        i,
        j,
        rows,
        cols,
        size,
        g_idx,
        valid,
        GRID: tl.constexpr,
        LAYOUT: tl.constexpr,
        HAS_G_IDX: tl.constexpr,
    ):
        """The index of s(i, j) in its stored part (spec §6 scale grids)."""
        if GRID == 0:
            g = i * 0
        elif GRID == 1:
            g = i
        elif GRID == 2:
            g = j
        else:
            w = (cols + size - 1) // size
            if HAS_G_IDX:  # noqa: SIM108 - a constexpr branch: no load without a g_idx
                b = tl.load(g_idx + j, mask=valid, other=0).to(tl.int64)
            else:
                b = j // size
            if LAYOUT == 0:
                g = i * w + b
            elif LAYOUT == 1:
                g = b * rows + i
            else:
                tiles = (w + 3) // 4
                g = (
                    ((i // 128) * tiles + b // 4) * 512
                    + (i % 32) * 16
                    + ((i % 128) // 32) * 4
                    + b % 4
                )
        return g

    @triton.jit  # type: ignore[untyped-decorator]
    def _by_post(post, j, v, valid, POST: tl.constexpr):  # type: ignore[no-untyped-def]
        """`v x post`: one factor (1), one per column (2), or none (0)."""
        if POST == 1:
            v = v * tl.load(post)
        elif POST == 2:
            v = v * tl.load(post + j, mask=valid, other=1.0)
        return v

    @triton.jit  # type: ignore[untyped-decorator]
    def _store(out, e, v, valid, OUT: tl.constexpr):  # type: ignore[no-untyped-def]
        """round_L into the destination's bits: nearest even, NaN canonical."""
        nan = v != v
        if OUT == 0:
            bits = tl.where(nan, 0x7FC00000, v.to(tl.int32, bitcast=True))
        elif OUT == 1:
            b = v.to(tl.uint32, bitcast=True)
            r = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
            bits = tl.where(nan, 0x7FC0, r).to(tl.uint16).to(tl.int16, bitcast=True)
        else:
            bits = tl.where(nan, 0x7E00, v.to(tl.float16).to(tl.int16, bitcast=True))
        tl.store(out + e, bits, mask=valid)

    @triton.jit  # type: ignore[untyped-decorator]
    def _affine_kernel(  # type: ignore[no-untyped-def]
        out,
        words,
        slots,
        scale,
        term,
        g_idx,
        total,
        rows,
        cols,
        nw,
        offset,
        size,
        BITS: tl.constexpr,
        WORD: tl.constexpr,
        TRANSPOSED: tl.constexpr,
        AXIS: tl.constexpr,
        FILL: tl.constexpr,
        ORDER: tl.constexpr,
        SIGNED: tl.constexpr,
        GRID: tl.constexpr,
        LAYOUT: tl.constexpr,
        HAS_G_IDX: tl.constexpr,
        post,
        POST: tl.constexpr,
        TERM: tl.constexpr,
        OUT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = e < total
        i = e // cols
        j = e % cols
        q = _code(
            words,
            slots,
            i,
            j,
            rows,
            cols,
            nw,
            offset,
            valid,
            BITS,
            WORD,
            TRANSPOSED,
            AXIS,
            FILL,
            ORDER,
            SIGNED,
        )
        g = _grid(i, j, rows, cols, size, g_idx, valid, GRID, LAYOUT, HAS_G_IDX)
        s = tl.load(scale + g, mask=valid, other=0.0)
        v = q.to(tl.float32)
        if TERM == 1:
            v = (v - tl.load(term + g, mask=valid, other=0.0)) * s
        elif TERM == 2:
            v = v * s + tl.load(term + g, mask=valid, other=0.0)
        else:
            v = v * s
        v = _by_post(post, j, v, valid, POST)
        _store(out, e, v, valid, OUT)

    @triton.jit  # type: ignore[untyped-decorator]
    def _table_kernel(  # type: ignore[no-untyped-def]
        out,
        words,
        slots,
        values,
        n_values,
        scale,
        scale_2,
        g_idx,
        error,
        total,
        rows,
        cols,
        nw,
        offset,
        size,
        BITS: tl.constexpr,
        WORD: tl.constexpr,
        TRANSPOSED: tl.constexpr,
        AXIS: tl.constexpr,
        FILL: tl.constexpr,
        ORDER: tl.constexpr,
        SIGNED: tl.constexpr,
        GRID: tl.constexpr,
        LAYOUT: tl.constexpr,
        HAS_G_IDX: tl.constexpr,
        post,
        POST: tl.constexpr,
        SCALE_2: tl.constexpr,
        OUT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = e < total
        i = e // cols
        j = e % cols
        q = _code(
            words,
            slots,
            i,
            j,
            rows,
            cols,
            nw,
            offset,
            valid,
            BITS,
            WORD,
            TRANSPOSED,
            AXIS,
            FILL,
            ORDER,
            SIGNED,
        )
        bad = valid & ((q < 0) | (q >= n_values))
        tl.store(error + q * 0, 1, mask=bad)
        t = tl.load(values + q, mask=valid & ~bad, other=0.0)
        s = tl.load(
            scale + _grid(i, j, rows, cols, size, g_idx, valid, GRID, LAYOUT, HAS_G_IDX),
            mask=valid,
            other=0.0,
        )
        if SCALE_2 == 2:
            s = tl.math.div_rn(s, tl.load(scale_2))
        v = t * s
        if SCALE_2 == 1:
            v = v * tl.load(scale_2)
        v = _by_post(post, j, v, valid, POST)
        _store(out, e, v, valid, OUT)

    @triton.jit  # type: ignore[untyped-decorator]
    def _blocks_kernel(  # type: ignore[no-untyped-def]
        out,
        data,
        total,
        post,
        POST: tl.constexpr,
        N: tl.constexpr,
        S: tl.constexpr,
        DECODE: tl.constexpr,
        OUT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        e = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = e < total
        v = DECODE(data + (e // N) * S, e % N, valid)
        v = _by_post(post, e, v, valid, POST)
        _store(out, e, v, valid, OUT)

    @triton.jit
    def u8_at(base, at, mask):  # type: ignore[no-untyped-def]
        """A struct's byte at `at`, as int32."""
        return tl.load(base + at, mask=mask, other=0).to(tl.int32)

    @triton.jit
    def i8_at(base, at, mask):  # type: ignore[no-untyped-def]
        return tl.load(base + at, mask=mask, other=0).to(tl.int8, bitcast=True).to(tl.int32)

    @triton.jit
    def f16_at(base, at, mask):  # type: ignore[no-untyped-def]
        """A struct's little-endian F16 at byte `at`, as f32 (exact)."""
        lo = tl.load(base + at, mask=mask, other=0).to(tl.uint16)
        hi = tl.load(base + at + 1, mask=mask, other=0).to(tl.uint16)
        return (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
