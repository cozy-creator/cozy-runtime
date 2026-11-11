"""Ulysses' int8 wire (`ulysses.exchange`): each row of a head's Q, K, V or output, less a shift
every GPU shares (K's mean), turned by the orthonormal Hadamard matrix of its order and scaled to
its own absmax, crosses as int8 with that scale in its dtype; back, the same turn undoes it.

On CUDA one Triton set (`KERNELS`, a Runtime set: built at machine start) does each direction in
one pass over any strides, so a head group's rows pack straight into their send rows and unpack
straight into the attention kernel's layout. The turn is a product with the ±1 matrix (exact in
bf16, summed in fp32), then the order's -1/2 power. It serves once its probe loses no more than
`verify.SLACK` times `law`'s error on this card (`kernel_ready.serve`). Until then, and on a CPU,
`law` computes the same in torch ops (fp32), and either side decodes the other's rows.
"""

from __future__ import annotations

import itertools
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from cozy_runtime.accel.kernels import Launch, TritonEntry, TritonKernels
from cozy_runtime.internal import kernel_ready, verify

try:
    import triton
    import triton.language as tl
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    if TYPE_CHECKING:
        raise
    TRITON_ABSENT = f"triton does not import ({_absent})"
else:
    TRITON_ABSENT = ""

#: Rows a program turns: 32 at 4 warps keeps the turn's operands in registers (sm89: no spills)
_ROWS = 32
#: Orders served, and each dtype's name in a Triton signature
_ORDERS = (64, 128)
_TYPES = {torch.bfloat16: "bf16", torch.float16: "fp16"}
#: The rows' dtype, by the kernels' `T` constexpr: it keys their entries
_CODES = {torch.bfloat16: 0, torch.float16: 1}
#: ±1 Sylvester matrices by device and order
_SIGNS: dict[tuple[torch.device, int], Tensor] = {}


def signs(order: int, device: torch.device) -> Tensor:
    """The Sylvester-Hadamard matrix of `order` (a power of two), ±1 in bf16: symmetric, and its
    square is `order` times the identity."""
    key = (device, order)
    found = _SIGNS.get(key)
    if found is None:
        turn = torch.ones(1, 1)
        while turn.shape[0] < order:
            turn = torch.cat([torch.cat([turn, turn], 1), torch.cat([turn, -turn], 1)])
        found = _SIGNS[key] = turn.to(device, torch.bfloat16)
    return found


def turned(rows: Tensor) -> Tensor:
    """`rows` (any float, last dimension a power of two) turned, in fp32: a shift `narrow` takes."""
    order = rows.shape[-1]
    return (rows.float() @ signs(order, rows.device).float()) * order**-0.5


def law(x: Tensor, shift: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """`x`'s rows less the turned `shift`, turned, as int8 scaled to each row's absmax (rounded
    half away from zero), and those scales in `x`'s dtype."""
    t = turned(x)
    if shift is not None:
        t -= shift
    tiny = torch.finfo(x.dtype).tiny
    scale = (t.abs().amax(-1, keepdim=True) * (1.0 / 127)).clamp_(min=tiny).to(x.dtype)
    q = (t / scale.float()).clamp_(-127, 127)
    return (q.sign() * (q.abs() + 0.5).floor()).to(torch.int8), scale


def law_back(values: Tensor, scale: Tensor) -> Tensor:
    """`law`'s rows back, in `scale`'s dtype."""
    order = values.shape[-1]
    summed = values.float() @ signs(order, values.device).float()
    return (summed * (scale.float() * order**-0.5)).to(scale.dtype)


def narrow(
    launch: Launch | None, x: Tensor, out: Tensor, scale: Tensor, shift: Tensor | None = None
) -> None:
    """`law(x, shift)` into `out` (int8) and `scale` (`x`'s dtype, last dimension 1): the same
    leading shape, any strides, `shift` broadcast to it; on `launch` where it serves rows whole."""
    if launch is None or not _whole(x, out, *(() if shift is None else (shift,))):
        values, factors = law(x, shift)
        out.copy_(values)
        scale.copy_(factors)
        return
    lead = x.shape[:-1]
    held = (x, out, scale.expand(*lead, 1)) + (() if shift is None else (shift.expand_as(x),))
    sizes, strides = _lead(lead), [_strides(t) for t in held]
    grid = (triton.cdiv(x[..., 0].numel(), _ROWS),)
    with _on(x):
        launch(
            _narrow_kernel, grid, x, out, scale, shift if shift is not None else scale,
            signs(x.shape[-1], x.device), *sizes, *strides[0], *strides[1], *strides[2],
            *(strides[3] if shift is not None else (0, 0, 0, 0)), torch.finfo(x.dtype).tiny,
            x.shape[-1] ** -0.5, x.shape[-1], int(shift is not None), _CODES[x.dtype], _ROWS,
        )  # fmt: skip


def widen(launch: Launch | None, values: Tensor, scale: Tensor, out: Tensor) -> None:
    """`law_back(values, scale)` into `out`: the same leading shape, any strides."""
    if launch is None or not _whole(values, out):
        out.copy_(law_back(values, scale))
        return
    lead = values.shape[:-1]
    strides = [_strides(t) for t in (values, scale.expand(*lead, 1), out)]
    grid = (triton.cdiv(values[..., 0].numel(), _ROWS),)
    with _on(values):
        launch(
            _widen_kernel, grid, values, scale, out, signs(values.shape[-1], values.device),
            *_lead(lead), *strides[0], *strides[1], *strides[2], values.shape[-1] ** -0.5,
            values.shape[-1], _CODES[out.dtype], _ROWS,
        )  # fmt: skip


def _on(t: Tensor) -> AbstractContextManager[object]:
    """`t`'s GPU current (a CPU tensor runs the kernels interpreted: `TRITON_INTERPRET=1`)."""
    return torch.cuda.device(t.device) if t.is_cuda else nullcontext()


def _lead(lead: torch.Size) -> tuple[int, ...]:
    return (1,) * (4 - len(lead)) + tuple(lead)


def _whole(*rows: Tensor) -> bool:
    """Each row contiguous and 16-aligned (its start and every leading stride), at most 4 leading
    dimensions and 2**31 rows: what the kernels read and write, a row's elements 16 B at a time."""
    return rows[0][..., 0].numel() < 1 << 31 and all(
        t.stride(-1) == 1
        and t.dim() <= 5
        and t.data_ptr() % 16 == 0
        and all(stride % 16 == 0 for stride in t.stride()[:-1])
        for t in rows
    )


def _strides(t: Tensor) -> tuple[int, ...]:
    lead = t.stride()[:-1]
    return (0,) * (4 - len(lead)) + tuple(lead)


def launcher(x: Tensor) -> Launch | None:
    """This card's kernels for rows like `x` once they serve (`kernel_ready.serve`), else None:
    `law` serves."""
    if (
        x.device.type != "cuda"
        or x.dtype not in _TYPES
        or x.shape[-1] not in _ORDERS
        or TRITON_ABSENT
        or not kernel_ready.TRITON
    ):
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    return kernel_ready.serve(_name(x.device), _opener(x.device), capturing)


if not TRITON_ABSENT:

    @triton.jit  # type: ignore[untyped-decorator]
    def _index(r, n1, n2, n3):  # type: ignore[no-untyped-def]
        """Row `r`'s four indices: int32 division, once a row (64-bit division is a long
        software routine on the card)."""
        i3 = r % n3
        t = r // n3
        i2 = t % n2
        t = t // n2
        return t // n1, t % n1, i2, i3

    @triton.jit  # type: ignore[untyped-decorator]
    def _at(i0, i1, i2, i3, s0, s1, s2, s3):  # type: ignore[no-untyped-def]
        return (
            i0.to(tl.int64) * s0
            + i1.to(tl.int64) * s1
            + i2.to(tl.int64) * s2
            + i3.to(tl.int64) * s3
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def _narrow_kernel(  # type: ignore[no-untyped-def]
        x, y, s, h, sign, n0, n1, n2, n3,
        x0, x1, x2, x3, y0, y1, y2, y3, s0, s1, s2, s3, h0, h1, h2, h3,
        tiny, c, D: tl.constexpr, SHIFT: tl.constexpr, T: tl.constexpr, ROWS: tl.constexpr,
    ):  # fmt: skip
        """`law` for `ROWS` rows; `c` is the order's -1/2 power."""
        r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        whole = r < n0 * n1 * n2 * n3
        live = whole[:, None]
        i0, i1, i2, i3 = _index(r, n1, n2, n3)
        cols = tl.arange(0, D)[None, :]
        at_x = tl.multiple_of(_at(i0, i1, i2, i3, x0, x1, x2, x3), 16)
        rows = tl.load(x + at_x[:, None] + cols, mask=live, other=0.0)
        turn = tl.load(sign + tl.arange(0, D)[:, None] * D + cols).to(x.dtype.element_ty)
        t = tl.dot(rows, turn, out_dtype=tl.float32) * c
        if SHIFT:
            at_h = tl.multiple_of(_at(i0, i1, i2, i3, h0, h1, h2, h3), 16)
            t -= tl.load(h + at_h[:, None] + cols, mask=live, other=0.0)
        scale = tl.maximum(tl.max(tl.abs(t) * (1.0 / 127.0), axis=1), tiny).to(s.dtype.element_ty)
        q = tl.minimum(tl.maximum(t / scale.to(tl.float32)[:, None], -127.0), 127.0)
        q = tl.where(q < 0, -1.0, 1.0) * tl.floor(tl.abs(q) + 0.5)
        at = tl.multiple_of(_at(i0, i1, i2, i3, y0, y1, y2, y3), 16)[:, None] + cols
        tl.store(y + at, q.to(tl.int8), mask=live)
        tl.store(s + _at(i0, i1, i2, i3, s0, s1, s2, s3), scale, mask=whole)

    @triton.jit  # type: ignore[untyped-decorator]
    def _widen_kernel(  # type: ignore[no-untyped-def]
        q, s, out, sign, n0, n1, n2, n3,
        q0, q1, q2, q3, s0, s1, s2, s3, o0, o1, o2, o3, c,
        D: tl.constexpr, T: tl.constexpr, ROWS: tl.constexpr,
    ):  # fmt: skip
        """`law_back` for `ROWS` rows; `c` is the order's -1/2 power."""
        r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        whole = r < n0 * n1 * n2 * n3
        live = whole[:, None]
        i0, i1, i2, i3 = _index(r, n1, n2, n3)
        cols = tl.arange(0, D)[None, :]
        at_q = tl.multiple_of(_at(i0, i1, i2, i3, q0, q1, q2, q3), 16)
        values = tl.load(q + at_q[:, None] + cols, mask=live, other=0)
        turn = tl.load(sign + tl.arange(0, D)[:, None] * D + cols).to(out.dtype.element_ty)
        summed = tl.dot(values.to(out.dtype.element_ty), turn, out_dtype=tl.float32)
        scale = tl.load(s + _at(i0, i1, i2, i3, s0, s1, s2, s3), mask=whole, other=0.0)
        rows = summed * (scale.to(tl.float32) * c)[:, None]
        at = tl.multiple_of(_at(i0, i1, i2, i3, o0, o1, o2, o3), 16)[:, None] + cols
        tl.store(out + at, rows.to(out.dtype.element_ty), mask=live)


def _entries() -> tuple[TritonEntry, ...]:
    if TRITON_ABSENT:
        return ()
    strides = {f"{t}{i}": "i64" for t in "xysh" for i in range(4)}
    sizes = {f"n{i}": "i32" for i in range(4)}
    narrowing = tuple(
        TritonEntry(
            _narrow_kernel,
            {"x": f"*{name}", "y": "*i8", "s": f"*{name}", "h": "*fp32" if shift else f"*{name}",
             "sign": "*bf16", **sizes, **strides, "tiny": "fp32", "c": "fp32", "D": "constexpr",
             "SHIFT": "constexpr", "T": "constexpr", "ROWS": "constexpr"},
            {"D": order, "SHIFT": shift, "T": _CODES[dtype], "ROWS": _ROWS},
            aligned=("x", "y", "h", "sign") if shift else ("x", "y", "sign"),
            num_warps=4,
            num_stages=1,
        )
        for (dtype, name), order, shift in itertools.product(_TYPES.items(), _ORDERS, (0, 1))
    )  # fmt: skip
    strides = {f"{t}{i}": "i64" for t in "qso" for i in range(4)}
    widening = tuple(
        TritonEntry(
            _widen_kernel,
            {"q": "*i8", "s": f"*{name}", "out": f"*{name}", "sign": "*bf16", **sizes, **strides,
             "c": "fp32", "D": "constexpr", "T": "constexpr", "ROWS": "constexpr"},
            {"D": order, "T": _CODES[dtype], "ROWS": _ROWS},
            aligned=("q", "out", "sign"),
            num_warps=4,
            num_stages=1,
        )
        for (dtype, name), order in itertools.product(_TYPES.items(), _ORDERS)
    )  # fmt: skip
    return narrowing + widening


#: Built with the Runtime's own sets (`machine_kernels.triton_sets`): a request never compiles.
KERNELS = TritonKernels("cozy-ulysses-turn", _entries())


def _name(device: torch.device) -> str:
    return f"{KERNELS.name} {device}"


def _opener(device: torch.device) -> kernel_ready.Opener[Launch]:
    return partial(kernel_ready.TRITON[0], KERNELS, partial(probe, device), device=device)


def prime(index: int) -> None:
    """Opens cuda:`index`'s kernels at device init (`kernel_ready.prime`), beside the weight fill,
    so a group's first exchange serves them."""
    if not TRITON_ABSENT and kernel_ready.TRITON:
        device = torch.device("cuda", index)
        kernel_ready.prime(_name(device), _opener(device))


def _probe_rows(dtype: torch.dtype, order: int, device: torch.device, seed: int) -> Tensor:
    """[3, 70, 5, order] rows with what the wire must carry: a K-like mean on every row, a
    V-like channel twenty times the rest, one row of zeros."""
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = torch.randn(3, 70, 5, order, generator=generator, device=device)
    rows += 4 * torch.randn(1, 1, 5, order, generator=generator, device=device)
    rows[..., 3] *= 20
    rows[1, 7] = 0
    return rows.to(dtype)


def probe(device: torch.device, launch: Launch) -> verify.Verdict:
    """Both kernels against `law` on this card, each dtype and order, rows read from a strided
    view and written transposed, with and without a shift: the round trip's loss against the
    unquantized rows within `verify.SLACK` of the law's; and `law`'s own rows widened within
    1e-5 of `law_back`."""
    losses: dict[str, tuple[float, float]] = {}
    for dtype, order in itertools.product(_TYPES, _ORDERS):
        rows = _probe_rows(dtype, order, device, 20261110 + order)
        wide = torch.zeros(3, 70, 9, order, dtype=dtype, device=device)
        wide[:, :, 2:7] = rows
        x = wide[:, :, 2:7]
        mean = rows.float().mean((0, 1))
        for shift in (None, turned(mean)):
            values = torch.empty(5, 70, 3, order, dtype=torch.int8, device=device)
            scale = torch.empty(5, 70, 3, 1, dtype=dtype, device=device)
            narrow(launch, x, values.permute(2, 1, 0, 3), scale.permute(2, 1, 0, 3), shift)
            back = torch.empty_like(rows)
            widen(launch, values.permute(2, 1, 0, 3), scale.permute(2, 1, 0, 3), back)
            exact = rows.float() - (0 if shift is None else mean)
            want = law(x, shift)
            case = f"{_TYPES[dtype]} D={order}" + (" shifted" if shift is not None else "")
            losses[case] = (
                verify.error(torch, back, exact),
                verify.error(torch, law_back(*want), exact),
            )
            plain = torch.empty_like(rows).permute(1, 0, 2, 3)
            widen(launch, want[0].permute(1, 0, 2, 3), want[1].permute(1, 0, 2, 3), plain)
            drift = verify.error(torch, plain.permute(1, 0, 2, 3), law_back(*want))
            if drift > 1e-5:
                return verify.Verdict(False, f"{case}: widen drifts {drift:.2e} from law_back")
    return verify.against_law(losses)
