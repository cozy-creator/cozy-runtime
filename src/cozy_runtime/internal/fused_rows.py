"""Rows a model's blocks compute value by value, each in one pass (`execution.fused` says whose):

    norm:     out = rms(x) [* weight] [* scale] [+ shift]
    rotation: out = x * cos + turn(x) * sin

The norm is over `[..., width]` rows in bf16 or fp16, `rms(x) = x * rsqrt(mean(x^2) + eps)` in
float32, each step rounded to the rows' dtype where torch rounds it (`law`). Eager, that is an
fp32 upcast, its square, a mean, an fp32 product and a downcast, then a pass for each of the scale
and the shift: about 30 bytes moved a value for the norm alone, 4 here. `scale` and `shift` are
tables in the rows' dtype that broadcast over `x`'s leading axes (`table`): one row for every
token, one a batch, one a token, read where they are, at any row stride.

The rotation turns each pair of a head's values by its (cos, sin) entry, in float32, rounded once
to `x`'s dtype (`turn_law`): pairs `(j, j + head/2)` (split) or `(2i, 2i + 1)` (interleaved).
Its fp32 tables are read where they lie (`Turn`: a table entry a row of `x`'s heads, by up to
three row axes, any strides); eager makes fp32 copies of `x`, its products and a transposed copy
back.

One Triton set (`KERNELS`, the Runtime's: shipped prebuilt, built at machine start otherwise). A
norm row is read in `BLOCK`-wide tiles, twice, the second read from the cache, so one entry serves
any width from one `BLOCK`. The set serves a card once its probe is within rounding of the laws
there (`bound`, `turn_bound`; the two differ in the order of the mean's sum and of fp32 sums).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from cozy_runtime.accel.kernels import Launch, TritonEntry, TritonKernels, jit, read
from cozy_runtime.internal import kernel_ready, verify

try:
    import triton
    import triton.language as tl
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    TRITON_ABSENT = f"triton does not import ({_absent})"
    if TYPE_CHECKING:
        raise
else:
    TRITON_ABSENT = ""

#: A norm row's tile, and the narrowest norm row the kernel takes (one tile).
BLOCK = 1024
_WARPS = 4
#: The rows' dtypes: each one's pointer type and `ROUND` (1 bf16, 2 fp16).
_DTYPES = {torch.bfloat16: ("*bf16", 1), torch.float16: ("*fp16", 2)}
#: The heads a rotation entry takes, and the values one of its programs turns.
HEADS = (64, 128)
_TURNED = 2048
#: A table over the rows: (rows an entry, entries, elements between entries).
type Table = tuple[int, int, int]

if not TRITON_ABSENT:

    @triton.jit
    def _rb(x):  # type: ignore[no-untyped-def]
        # fp32 rounded to bf16, nearest even, in the integer domain so no pass folds the round trip
        bits = x.to(tl.int32, bitcast=True)
        bits = bits + 0x7FFF + ((bits >> 16) & 1)
        return (bits & -65536).to(tl.float32, bitcast=True)

    @triton.jit
    def _rh(x):  # type: ignore[no-untyped-def]
        return tl.inline_asm_elementwise(
            "{ .reg .b16 h; cvt.rn.f16.f32 h, $1; cvt.f32.f16 $0, h; }",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _round(x, ROUND: tl.constexpr):  # type: ignore[no-untyped-def]
        if ROUND == 1:
            x = _rb(x)
        elif ROUND == 2:
            x = _rh(x)
        return x

    @jit
    def norm_rows_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        out_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        n_cols,
        n_tiles,
        first,
        scale_per,
        scale_count,
        scale_stride,
        shift_per,
        shift_count,
        shift_stride,
        eps,
        WEIGHT: tl.constexpr,
        MODULATE: tl.constexpr,
        ROUND: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One row a program, in BLOCK-wide tiles: its squares, then its writes. MODULATE 1: the
        # scale; 2: the scale and the shift. Row r of the launch is row `first + r` of the call,
        # and reads table entry `(row // per) % count`, `stride` elements from the one before.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        base = row * n_cols
        squares = tl.zeros([BLOCK], dtype=tl.float32)
        for seen in range(0, n_tiles):
            cell = seen * BLOCK + cols
            value = tl.load(x_ptr + base + cell, mask=cell < n_cols, other=0.0).to(tl.float32)
            squares += value * value
        inv = tl.math.rsqrt(tl.sum(squares, axis=0) / n_cols + eps)
        # an unused table is launched with one entry for every row: its row is 0
        scale_row = ((row + first) // scale_per) % scale_count * scale_stride
        shift_row = ((row + first) // shift_per) % shift_count * shift_stride
        for tile in range(0, n_tiles):
            at = tile * BLOCK + cols
            mask = at < n_cols
            x = tl.load(x_ptr + base + at, mask=mask, other=0.0).to(tl.float32)
            out = _round(x * inv, ROUND)
            if WEIGHT:
                weight = tl.load(weight_ptr + at, mask=mask, other=0.0).to(tl.float32)
                out = _round(out * weight, ROUND)
            if MODULATE > 0:
                scale = tl.load(scale_ptr + scale_row + at, mask=mask, other=0.0).to(tl.float32)
                out = _round(out * scale, ROUND)
            if MODULATE == 2:
                shift = tl.load(shift_ptr + shift_row + at, mask=mask, other=0.0).to(tl.float32)
                out = _round(out + shift, ROUND)
            tl.store(out_ptr + base + at, out.to(out_ptr.dtype.element_ty), mask=mask)

    @jit
    def turn_rows_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        out_ptr,
        cos_ptr,
        sin_ptr,
        n_rows,
        first,
        size_1,
        size_2,
        stride_0,
        stride_1,
        stride_2,
        stride_col,
        split,
        wide,
        HEAD: tl.constexpr,
        ROWS: tl.constexpr,
        ROUND: tl.constexpr,
    ):
        # ROWS rows of one head's HEAD values a program, each rounded once to ROUND's dtype. Row
        # n of the call (`first` on) is (n // (size_1 * size_2), n // size_2 % size_1, n % size_2)
        # over x's row axes, and its tables' entry is those times their strides. split: value j
        # pairs with j +- HEAD / 2, else 2i with 2i + 1; wide: the tables hold a column a value,
        # else one a pair.
        rows = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)[:, None]
        cols = tl.arange(0, HEAD)[None, :]
        live = rows < n_rows
        at = rows + first
        entry = (at // (size_1 * size_2)) * stride_0 + (at // size_2 % size_1) * stride_1
        entry += (at % size_2) * stride_2
        half = HEAD // 2
        low = tl.where(split != 0, cols < half, cols % 2 == 0)
        partner = tl.where(split != 0, tl.where(low, cols + half, cols - half), cols ^ 1)
        column = tl.where(wide != 0, cols, tl.where(split != 0, cols % half, cols // 2))
        x = tl.load(x_ptr + rows * HEAD + cols, mask=live, other=0.0).to(tl.float32)
        paired = tl.load(x_ptr + rows * HEAD + partner, mask=live, other=0.0).to(tl.float32)
        paired = tl.where(low, -paired, paired)
        table = entry + column * stride_col
        cos = tl.load(cos_ptr + table, mask=live, other=1.0)
        sin = tl.load(sin_ptr + table, mask=live, other=0.0)
        out = _round(x * cos + paired * sin, ROUND)
        tl.store(out_ptr + rows * HEAD + cols, out.to(out_ptr.dtype.element_ty), mask=live)


def _entries() -> tuple[TritonEntry, ...]:
    if TRITON_ABSENT:
        return ()
    found: list[TritonEntry] = []
    pointers = ("x_ptr", "out_ptr", "weight_ptr", "scale_ptr", "shift_ptr")
    for (pointer, rounding), weight, modulate in itertools.product(
        _DTYPES.values(), (0, 1), (0, 1, 2)
    ):
        sizes = {"n_cols": "i32", "n_tiles": "i32", "eps": "fp32"}
        types = {**dict.fromkeys(pointers, pointer), **sizes}
        signature = {
            name: "constexpr" if name.isupper() else types.get(name, "i64")
            for name in norm_rows_kernel.arg_names
        }
        constexprs = {"WEIGHT": weight, "MODULATE": modulate, "ROUND": rounding, "BLOCK": BLOCK}
        # The rows are addressed at `row * n_cols`: every launch's are 16-byte aligned (`takes`).
        found.append(
            TritonEntry(
                norm_rows_kernel, signature, constexprs, ("x_ptr", "out_ptr", "n_cols"), _WARPS
            )
        )
    for (pointer, rounding), head in itertools.product(_DTYPES.values(), HEADS):
        types = {"x_ptr": pointer, "out_ptr": pointer, "cos_ptr": "*fp32", "sin_ptr": "*fp32"}
        types |= {"split": "i32", "wide": "i32"}
        signature = {
            name: "constexpr" if name.isupper() else types.get(name, "i64")
            for name in turn_rows_kernel.arg_names
        }
        constexprs = {"HEAD": head, "ROWS": _TURNED // head, "ROUND": rounding}
        # x's rows are addressed at `row * HEAD`, from an aligned start (`turns_take`).
        found.append(TritonEntry(turn_rows_kernel, signature, constexprs, ("x_ptr", "out_ptr")))
    return tuple(found)


#: Built with the Runtime's own sets (`machine_kernels.triton_sets`): a request never compiles.
KERNELS = TritonKernels("cozy-fused-rows", _entries())


# --------------------------------------------------------------------------- norms


def fits(x: Tensor) -> bool:
    """Whether rows of `x`'s kind may be the norm kernel's: bf16 or fp16, over two axes or more,
    a tile wide or more (`width`)."""
    return x.dim() >= 2 and x.dtype in _DTYPES and width(int(x.shape[-1]))


def width(n: int) -> bool:
    """Whether the norm kernel takes rows `n` wide."""
    return n >= BLOCK and n % 16 == 0


def takes(x: Tensor) -> bool:
    """Whether `x`'s rows are the norm kernel's: `fits`, on a card, contiguous, each row 16-byte
    aligned."""
    return fits(x) and x.is_cuda and x.is_contiguous() and x.data_ptr() % 16 == 0


def table(x: Tensor, value: Tensor) -> Table | None:
    """How `value` broadcasts over `x`'s rows, where the kernel can read it there: its last axis
    is the rows' width at unit stride, and its other axes are 1 or `x`'s over one run of them, at
    one stride (a table for every row, one a batch, one a token; a slice of a wider table). None
    for any other shape, and for one that would widen `x`."""
    lead, own = x.dim() - 1, value.dim() - 1
    if own < 0 or own > lead or value.shape[-1] != x.shape[-1] or value.stride(-1) != 1:
        return None
    shape = (1,) * (lead - own) + tuple(int(n) for n in value.shape[:-1])
    strides = (0,) * (lead - own) + tuple(int(n) for n in value.stride()[:-1])
    rows = [int(n) for n in x.shape[:-1]]
    held = [i for i, n in enumerate(shape) if n != 1]
    if not held:
        return max(math.prod(rows), 1), 1, 0
    start, end = held[0], held[-1] + 1
    if any(shape[i] != rows[i] for i in range(start, end)):
        return None
    step = expected = strides[end - 1]
    for i in range(end - 1, start - 1, -1):
        if shape[i] != 1 and strides[i] != expected:
            return None
        expected *= shape[i]
    return max(math.prod(rows[end:]), 1), math.prod(rows[start:end]), step


def rows_of(value: Tensor, read_as: Table, first: int, count: int) -> Tensor:
    """`value`'s entries for rows `[first, first + count)` of the call, `[count, width]`, as the
    kernel reads them (`read_as`, its `table`)."""
    per, entries, stride = read_as
    width = int(value.shape[-1])
    at = torch.arange(first, first + count, device=value.device) // per % entries * stride
    cells = at[:, None] + torch.arange(width, device=value.device)[None, :]
    return _storage(value)[cells + value.storage_offset()]


def _storage(value: Tensor) -> Tensor:
    """All of `value`'s storage, one element after another."""
    return value.as_strided((value.untyped_storage().nbytes() // value.element_size(),), (1,), 0)


def launched(
    launch: Launch,
    x: Tensor,
    eps: float,
    weight: Tensor | None = None,
    scale: tuple[Tensor, Table] | None = None,
    shift: tuple[Tensor, Table] | None = None,
    first: int = 0,
) -> Tensor:
    """`law` of `x` (`takes(x)`) in one launch: a fresh tensor of `x`'s shape. `scale` and
    `shift` come with their `table` over the whole call's rows, of which `x` holds those from
    `first`; a `shift` comes with a `scale`."""
    width = int(x.shape[-1])
    rows = x.numel() // width
    out = torch.empty_like(x)
    if not rows:
        return out
    unused: tuple[Tensor, Table] = (x, (1, 1, 0))
    (scales, (scale_per, scale_count, scale_stride)) = scale or unused
    (shifts, (shift_per, shift_count, shift_stride)) = shift or unused
    launch(
        norm_rows_kernel,
        (rows,),
        x,
        out,
        x if weight is None else read(weight),
        scales,
        shifts,
        width,
        triton.cdiv(width, BLOCK),
        first,
        scale_per,
        scale_count,
        scale_stride,
        shift_per,
        shift_count,
        shift_stride,
        float(eps),
        int(weight is not None),
        (scale is not None) + (shift is not None),
        _DTYPES[x.dtype][1],
        BLOCK,
    )
    return out


def law(
    x: Tensor,
    eps: float,
    weight: Tensor | None = None,
    scale: Tensor | None = None,
    shift: Tensor | None = None,
) -> Tensor:
    """The norm's rows in torch ops: what the kernel computes, step for step, but for the order
    its mean sums in."""
    wide = x.float()
    out = (wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)
    for factor in (weight, scale):
        out = out if factor is None else out * factor
    return out if shift is None else out + shift


def bound(
    x: Tensor,
    eps: float,
    weight: Tensor | None = None,
    scale: Tensor | None = None,
    shift: Tensor | None = None,
) -> Tensor:
    """How far two computations of `law(x, ...)` may lie apart, a value at a time, from rounding
    alone. Each step that rounds to the rows' dtype moves its value by at most half a step there,
    at most half `finfo.eps` of its size, and a later factor carries that; two computations differ
    by twice that. The mean's sum, in either order over `width` positive squares, is within
    `width` fp32 epsilons of the other's (rsqrt halves it), and that carries to the end too."""
    info, single = torch.finfo(x.dtype), torch.finfo(torch.float32).eps
    value = x.float()
    value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + eps)
    carried, rounded = value.abs(), 1
    for factor in (weight, scale):
        if factor is not None:
            wide = factor.float()
            value = value * wide
            carried = carried * wide.abs() + value.abs()
            rounded += 1
    drift = value.abs() * (single * int(x.shape[-1]))
    if shift is not None:
        value = value + shift.float()
        carried = carried + value.abs()
        rounded += 1
    return carried * info.eps + drift + rounded * info.tiny * info.eps


def within(got: Tensor, want: Tensor, allowed: Tensor) -> float:
    """The largest difference between `got` and `want` as a share of what rounding allows there:
    1 or less where they are the same computation. NaN where either is not finite."""
    if got.shape != want.shape or got.dtype != want.dtype:
        raise ValueError(f"{tuple(got.shape)} {got.dtype} against {tuple(want.shape)} {want.dtype}")
    if not got.numel():
        return 0.0
    return float(((got.float() - want.float()).abs() / allowed).max())


# --------------------------------------------------------------------------- rotations


@dataclass(frozen=True, slots=True)
class Turn:
    """How the rotation kernel reads a call: `x` as rows of one `head`'s values over three row
    axes (`sizes`, outermost first), its tables' entry for a row at `strides` along them and
    `column` between columns; `split` pairs value j with j + head / 2 (else 2i with 2i + 1), and
    `wide` tables hold a column for every value (else one a pair)."""

    head: int
    split: bool
    wide: bool
    sizes: tuple[int, int, int]
    strides: tuple[int, int, int]
    column: int


def turns_take(x: Tensor) -> bool:
    """Whether `x` may be a rotation kernel's input: bf16 or fp16 on a card, contiguous, aligned,
    over two to four axes."""
    return (
        2 <= x.dim() <= 4
        and x.is_cuda
        and x.dtype in _DTYPES
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
    )


def tables_take(x: Tensor, cos: Tensor, sin: Tensor) -> bool:
    """Whether `(cos, sin)` may be a rotation kernel's tables over `x`: fp32 beside `x`, laid out
    alike."""
    return (
        cos.dtype == sin.dtype == torch.float32
        and cos.device == sin.device == x.device
        and cos.dim() >= 1
        and cos.shape == sin.shape
        and cos.stride() == sin.stride()
    )


def turns(x: Tensor, cos: Tensor) -> Iterator[Turn]:
    """Every way the kernel can read tables laid out as `cos` against `x`: a head of `HEADS` that
    divides `x`'s last axis (its row axes then `x`'s others and the heads, at most three); the
    tables' last axis one head's columns, or every head's in turn; each of their other axes as
    long as one row axis, a different one each, or 1 (read for every row); both pairings. Which
    one, if any, a function computes, its first call says (`execution.fused`)."""
    width, lead = int(x.shape[-1]), [int(n) for n in x.shape[:-1]]
    last, column = int(cos.shape[-1]), int(cos.stride(-1))
    axes = [(int(n), int(s)) for n, s in zip(cos.shape[:-1], cos.stride()[:-1], strict=True)]
    held = [(n, s) for n, s in axes if n != 1]
    for head in HEADS:
        if width % head:
            continue
        heads = width // head
        rows = lead + ([heads] if heads > 1 else [])
        if len(rows) > 3:
            continue
        rows = [1] * (3 - len(rows)) + rows
        for wide in (True, False):
            per = head if wide else head // 2
            spans = heads > 1 and last == per * heads
            if last != per and not spans:
                continue
            fixed: dict[int, int] = {2: per * column} if spans else {}
            free = [i for i in range(3) if i not in fixed]
            for chosen in itertools.permutations(free, len(held)):
                if any(rows[i] != n for i, (n, _) in zip(chosen, held, strict=True)):
                    continue
                strides = [0, 0, 0]
                for i, (_, s) in zip(chosen, held, strict=True):
                    strides[i] = s
                for i, s in fixed.items():
                    strides[i] = s
                sizes = (rows[0], rows[1], rows[2])
                read_at = (strides[0], strides[1], strides[2])
                for split in (True, False):
                    yield Turn(head, split, wide, sizes, read_at, column)


def turned(
    launch: Launch, x: Tensor, cos: Tensor, sin: Tensor, turn: Turn, first: int = 0
) -> Tensor:
    """`turn_law` of `x` in one launch: a fresh tensor of `x`'s shape; `x` holds the call's rows
    of `turn.head` values from `first`."""
    out = torch.empty_like(x)
    rows = x.numel() // turn.head
    if rows:
        (_, size_1, size_2), (stride_0, stride_1, stride_2) = turn.sizes, turn.strides
        launch(
            turn_rows_kernel,
            (triton.cdiv(rows, _TURNED // turn.head),),
            x,
            out,
            cos,
            sin,
            rows,
            first,
            size_1,
            size_2,
            stride_0,
            stride_1,
            stride_2,
            turn.column,
            int(turn.split),
            int(turn.wide),
            turn.head,
            _TURNED // turn.head,
            _DTYPES[x.dtype][1],
        )
    return out


def _columns(value: Tensor, turn: Turn, first: int, count: int) -> Tensor:
    """A table's entry for each of rows `[first, first + count)` of the call, one value a column
    of the head, `[count, head]`, as `turn` reads it."""
    (_, size_1, size_2), (stride_0, stride_1, stride_2) = turn.sizes, turn.strides
    at = torch.arange(first, first + count, device=value.device)
    entry = at // (size_1 * size_2) * stride_0 + at // size_2 % size_1 * stride_1
    entry = entry + at % size_2 * stride_2
    cols = torch.arange(turn.head, device=value.device)
    half = turn.head // 2
    column = cols if turn.wide else cols % half if turn.split else cols // 2
    cells = entry[:, None] + (column * turn.column)[None, :]
    return _storage(value)[cells + value.storage_offset()]


def _paired(values: Tensor, turn: Turn) -> Tensor:
    """Each value's partner in its pair, negated where the rotation subtracts it."""
    if turn.split:
        low, high = values.chunk(2, -1)
        return torch.cat([-high, low], -1)
    pairs = values.unflatten(-1, (-1, 2))
    return torch.stack([-pairs[..., 1], pairs[..., 0]], -1).flatten(-2)


def turn_law(x: Tensor, cos: Tensor, sin: Tensor, turn: Turn, first: int = 0) -> Tensor:
    """The rotation of `x`'s rows (the call's from `first`) in torch ops, as `turn` reads its
    tables: in fp32, rounded once to `x`'s dtype."""
    rows = x.reshape(-1, turn.head).float()
    count = int(rows.shape[0])
    c, s = (_columns(t, turn, first, count) for t in (cos, sin))
    return (rows * c + _paired(rows, turn) * s).to(x.dtype).view(x.shape)


def turn_bound(
    x: Tensor, cos: Tensor, sin: Tensor, turn: Turn, want: Tensor, first: int = 0
) -> Tensor:
    """How far two computations of a rotation whose result is `want` may lie apart, a value at a
    time, from rounding alone: each rounds its two fp32 products and their sum (at most an fp32
    epsilon of the products' sizes), then the sum once to `x`'s dtype (at most half a step of
    `want`'s size there)."""
    info, single = torch.finfo(x.dtype), torch.finfo(torch.float32).eps
    rows = x.reshape(-1, turn.head).float()
    count = int(rows.shape[0])
    c, s = (_columns(t, turn, first, count).abs() for t in (cos, sin))
    products = rows.abs() * c + _paired(rows, turn).abs() * s
    out = want.reshape(-1, turn.head).float().abs()
    return (out * info.eps + products * (2 * single) + info.tiny * info.eps).view(x.shape)


# --------------------------------------------------------------------------- serving


def serve(device: torch.device) -> Launch | None:
    """This card's kernels once they serve (`kernel_ready.serve`), else None: the modules' own
    forwards serve."""
    if TRITON_ABSENT or not kernel_ready.TRITON:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    return kernel_ready.serve(_name(device), _opener(device), capturing)


def _name(device: torch.device) -> str:
    return f"{KERNELS.name} {device}"


def _opener(device: torch.device) -> kernel_ready.Opener[Launch]:
    return partial(kernel_ready.TRITON[0], KERNELS, partial(probe, device), device=device)


def prime(index: int) -> None:
    """Opens cuda:`index`'s kernels at device init (`kernel_ready.prime`), beside the weight fill,
    so a model's first norm finds them."""
    if not TRITON_ABSENT and kernel_ready.TRITON:
        device = torch.device("cuda", index)
        kernel_ready.prime(_name(device), _opener(device))


def _case(
    generator: torch.Generator, dtype: torch.dtype, shape: Sequence[int], device: torch.device
) -> Tensor:
    return torch.randn(*shape, generator=generator, device=device).to(dtype)


def probe(device: torch.device, launch: Launch) -> verify.Verdict:
    """Both kernels against their laws on this card, within rounding (`bound`, `turn_bound`)."""
    generator = torch.Generator(device=device).manual_seed(20261119)
    with torch.cuda.device(device):
        norms = _probe_norms(device, launch, generator)
        if isinstance(norms, verify.Verdict):
            return norms
        turns_seen = _probe_turns(device, launch, generator)
        if isinstance(turns_seen, verify.Verdict):
            return turns_seen
    worst, cases = max(norms[0], turns_seen[0]), norms[1] + turns_seen[1]
    return verify.Verdict(True, f"within {worst:.3g} of rounding's allowance ({cases} cases)")


def _probe_norms(
    device: torch.device, launch: Launch, generator: torch.Generator
) -> tuple[float, int] | verify.Verdict:
    """Norms: both dtypes; a row of one tile, of several and of a ragged last one; with and
    without a weight; no table, a scale, and a scale and a shift, as one row, one a batch, one a
    token from a wider table's slices, and one a token shared by the batch; rows from `first` on.
    The worst share of rounding's allowance and the cases, or the first failure."""
    worst, cases = 0.0, 0
    for dtype, width, weighted in itertools.product(
        _DTYPES, (BLOCK, 4 * BLOCK, 2 * BLOCK + 48), (False, True)
    ):
        batch, tokens = 3, 37
        make = partial(_case, generator, dtype, device=device)
        x = make((batch, tokens, width)).mul_(3.0)
        weight = make((width,)) if weighted else None
        wider = make((batch, tokens, 6, width))
        chains: tuple[tuple[Tensor | None, Tensor | None], ...] = (
            (None, None),
            (make((width,)), None),
            (make((batch, 1, width)), make((1, 1, width))),
            (wider.unbind(2)[1], wider.unbind(2)[4]),
            (make((1, tokens, width)), make((batch, tokens, width))),
        )
        for (scale, shift), first in itertools.product(chains, (0, 5 * tokens // 2)):
            pairs: list[tuple[Tensor, Table] | None] = []
            for value in (scale, shift):
                read_as = None if value is None else table(x, value)
                if value is not None and read_as is None:
                    shape = tuple(value.shape)
                    return verify.Verdict(False, f"a {shape} table over {tuple(x.shape)}")
                pairs.append(None if value is None or read_as is None else (value, read_as))
            rows = x.view(-1, width)[first:]
            got = launched(launch, rows, 1e-6, weight, pairs[0], pairs[1], first)
            want = law(x, 1e-6, weight, scale, shift).view(-1, width)[first:]
            off = within(got, want, bound(x, 1e-6, weight, scale, shift).view(-1, width)[first:])
            worst, cases = max(worst, off), cases + 1
            if not off <= 1:
                case = f"norm {dtype} width {width} weight {weighted} first {first}"
                return verify.Verdict(False, f"{case}: {off:.3g} of rounding's allowance")
    return worst, cases


def _probe_turns(
    device: torch.device, launch: Launch, generator: torch.Generator
) -> tuple[float, int] | verify.Verdict:
    """Rotations: both dtypes and heads; every layout `turns` finds for per-head tables read
    through a transpose, for tables over every head in turn, and for one table row a token shared
    by the batch and the heads; rows from `first` on. As `_probe_norms` says it."""
    worst, cases = 0.0, 0
    for dtype, head in itertools.product(_DTYPES, HEADS):
        make = partial(_case, generator, dtype, device=device)
        angles = partial(_case, generator, torch.float32, device=device)
        batch, tokens, heads = 2, 37, 3
        per_head = angles((batch, tokens, heads, head // 2)).swapaxes(1, 2)
        layouts = (
            (make((batch, tokens, heads * head)), per_head),
            (make((batch, tokens, heads * head)), angles((batch, tokens, heads * head))),
            (make((batch, heads, tokens, head)), angles((tokens, head))),
        )
        for x, theta in layouts:
            cos, sin = theta.cos(), theta.sin()
            found = list(turns(x, cos))
            if not found:
                shape = tuple(cos.shape)
                return verify.Verdict(False, f"no layout of {shape} over {tuple(x.shape)}")
            for turn, first in itertools.product(found, (0, 3 * tokens // 2)):
                rows = x.reshape(-1, turn.head)[first:]
                got = turned(launch, rows, cos, sin, turn, first)
                want = turn_law(rows, cos, sin, turn, first)
                off = within(got, want, turn_bound(rows, cos, sin, turn, want, first))
                worst, cases = max(worst, off), cases + 1
                if not off <= 1:
                    case = f"rotation {dtype} {turn} first {first}"
                    return verify.Verdict(False, f"{case}: {off:.3g} of rounding's allowance")
    return worst, cases
