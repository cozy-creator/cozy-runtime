"""Exact cuts (varena-slim §2): each computes what the whole computes, results on the device, at
the level Rust holds for its site (`r.cut`). Level L is 2^L pieces, down to the finest.

Op bands (the layer's, `catch.RULES`): conv, constant pad, nearest upsampling and group norm in
row bands with halos and two-pass statistics, and the elementwise SiLU, GELU, add and mul.
Call cuts: Diffusers' FFNs by token, its channel RMS norms by rows, attention by head groups
then query rows, a package's `split` by windows.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch._ops import OpOverload
from torch.utils._python_dispatch import _disable_current_modes

from cozy_runtime.internal.execution import catch
from cozy_runtime.internal.execution.catch import Args, Kwargs, Scope

aten = torch.ops.aten
type Band = Callable[[int, int], Tensor]
type Attention = Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor]
#: diffusers' token-wise feed-forwards and the activation projections inside them
_FFN = frozenset({"FeedForward", "GELU", "GEGLU", "SwiGLU", "ApproximateGELU", "LinearActivation"})


def levels(units: int) -> int:
    """The finest level of `units` pieces: one unit a piece."""
    return max(units - 1, 0).bit_length()


def pieces(units: int, level: int) -> int:
    """Units a piece holds at `level` (2^level pieces, the last one short)."""
    return -(-units // (1 << level))


# ---------------------------------------------------------------------------- op bands


@dataclass(frozen=True, slots=True)
class Plan:
    """An op as rows [a, b) of its result `out` (meta) along `axis`: `band(a, b)` computes them,
    starting on multiples of `align`; `gather(rows)` reads what every band needs first;
    `finish` makes the op's own result from the filled tensor."""

    x: Tensor
    out: Tensor
    axis: int
    band: Band
    align: int = 1
    gather: Callable[[int], None] | None = None
    finish: Callable[[Tensor], object] | None = None

    @property
    def finest(self) -> int:
        return levels(-(-int(self.out.shape[self.axis]) // self.align))

    def rows(self, level: int) -> int:
        return pieces(-(-int(self.out.shape[self.axis]) // self.align), level) * self.align

    def __call__(self, own: Scope) -> object:
        operands = (self.x,)
        out = catch.settle(lambda: _empty(self.out, self.x.device), None, operands)
        if self.gather is not None:
            gather = self.gather
            catch.settle(lambda: gather(self.rows(own.level)), own, operands, first=True)
        height, at = int(self.out.shape[self.axis]), 0
        while at < height:
            write = functools.partial(self._write, out, at, own)
            at = catch.settle(write, own, operands, first=True)
        return out if self.finish is None else self.finish(out)

    def _write(self, out: Tensor, at: int, own: Scope) -> int:
        stop = min(at + self.rows(own.level), int(out.shape[self.axis]))
        out.narrow(self.axis, at, stop - at).copy_(self.band(at, stop))
        return stop


def _plan(made: list[Tensor], x: Tensor, axis: int, band: Band, **more: object) -> Plan | None:
    """When the op makes a tensor of two rows or more."""
    if not made or made[0].shape[axis] < 2:
        return None
    return Plan(x, made[0], axis, band, **more)  # type: ignore[arg-type]


def _arguments(func: OpOverload, args: Args, kwargs: Kwargs) -> dict[str, object]:
    named: dict[str, object] = {}
    for i, a in enumerate(func._schema.arguments):
        if i < len(args) or a.name in kwargs or a.has_default_value():
            default = a.default_value if a.has_default_value() else None
            named[a.name] = args[i] if i < len(args) else kwargs.get(a.name, default)
    return named


def _meta(func: OpOverload, args: Args, kwargs: Kwargs) -> list[Tensor]:
    """The op's results as meta tensors (empty: not new tensors, or no meta kernel)."""

    def meta(v: object) -> object:
        if isinstance(v, Tensor):
            return torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device="meta")
        return type(v)(map(meta, v)) if isinstance(v, list | tuple) else v

    try:
        with _disable_current_modes():
            made = func(*map(meta, args), **{k: meta(v) for k, v in kwargs.items()})
    except Exception:
        return []
    return [m for m in (made if isinstance(made, tuple) else (made,)) if isinstance(m, Tensor)]


def _empty(like: Tensor, device: torch.device) -> Tensor:
    return torch.empty_strided(like.shape, like.stride(), dtype=like.dtype, device=device)


def _rows(x: Tensor, axis: int, first: int, last: int, value: float = 0.0) -> Tensor:
    """Rows [first, last) of `x` along `axis`, `value` past its edges."""
    size = int(x.shape[axis])
    start, before = min(max(first, 0), size), min(max(-first, 0), last - first)
    length = max(min(last, size) - start, 0)
    edge = [0, 0] * (x.dim() - 1 - axis) + [before, last - first - before - length]
    rows = x.narrow(axis, start, length)
    return aten.constant_pad_nd.default(rows, edge, value) if any(edge) else rows


def _axis(x: Tensor) -> int:
    """The rows bands cut: a 1D signal's length, else the second-to-last dim."""
    return x.dim() - 2 if x.dim() > 3 else 2


def _each(value: object, n: int, default: int) -> list[int]:
    values = list(value) if isinstance(value, list | tuple) else [value or default]
    return [int(cast(int, v)) for v in values] * (n if len(values) == 1 else 1)


def _pointwise(func: OpOverload, args: Args, kwargs: Kwargs) -> Plan | None:
    """SiLU, GELU, add, mul: any rows of the result read the same rows of each operand that
    spans them (a broadcast operand whole), so bands along the result's longest axis."""
    if func not in (aten.silu.default, aten.gelu.default, aten.add.Tensor, aten.mul.Tensor):
        return None
    made = _meta(func, args, kwargs)
    if len(made) != 1 or made[0].dim() == 0:
        return None
    out = made[0]
    axis = max(range(out.dim()), key=lambda i: out.shape[i])
    height = int(out.shape[axis])

    def rows(v: object, lo: int, hi: int) -> object:
        if not isinstance(v, Tensor):
            return v
        at = axis - (out.dim() - v.dim())  # broadcasting aligns trailing dims
        return v.narrow(at, lo, hi - lo) if at >= 0 and v.shape[at] == height else v

    def band(lo: int, hi: int) -> Tensor:
        return func(
            *(rows(a, lo, hi) for a in args), **{k: rows(v, lo, hi) for k, v in kwargs.items()}
        )

    x = next(t for t in catch.tensors(args) if isinstance(t, Tensor))
    return _plan(made, x, axis, band)


def _conv(func: OpOverload, args: Args, kwargs: Kwargs) -> Plan | None:
    """Output rows [a, b) read input rows [a*s - p, (b-1)*s - p + d*(k-1) + 1), zeros past the
    edges."""
    a = _arguments(func, args, kwargs)
    x, w, bias, groups = a.get("input"), a.get("weight"), a.get("bias"), a.get("groups") or 1
    if not isinstance(x, Tensor) or not isinstance(w, Tensor) or not x.dim() == w.dim() > 2:
        return None
    if a.get("transposed") or isinstance(a.get("padding"), str):
        return None
    n, axis = w.dim() - 2, _axis(x)
    stride, padding = _each(a.get("stride"), n, 1), _each(a.get("padding"), n, 0)
    dilation, j = _each(a.get("dilation"), n, 1), axis - 2
    s, d, k = stride[j], dilation[j], int(w.shape[axis])

    def band(lo: int, hi: int) -> Tensor:
        rows = _rows(x, axis, lo * s - padding[j], (hi - 1) * s - padding[j] + d * (k - 1) + 1)
        pads = [0 if i == j else p for i, p in enumerate(padding)]
        conv = aten.convolution.default
        return conv(rows, w, bias, stride, pads, dilation, False, [0] * n, groups)

    return _plan(_meta(func, args, kwargs), x, axis, band)


def _pad(func: OpOverload, args: Args, kwargs: Kwargs) -> Plan | None:
    """A constant padding: output rows [a, b) are input rows [a - top, b - top), the value past
    its edges."""
    a = _arguments(func, args, kwargs)
    x, pad = a.get("self"), list(a.get("pad") or ())  # type: ignore[call-overload]
    value = float(a.get("value") or 0)  # type: ignore[arg-type]
    if not isinstance(x, Tensor) or x.dim() < 2 or min(pad, default=0) < 0:
        return None
    if a.get("mode", "constant") != "constant":
        return None
    axis, rest = x.dim() - 2, [*pad, 0, 0, 0, 0][: max(len(pad), 4)]
    others = [*rest[:2], 0, 0, *rest[4:]]

    def band(lo: int, hi: int) -> Tensor:
        rows = _rows(x, axis, lo - rest[2], hi - rest[2], value)
        return aten.constant_pad_nd.default(rows, others, value)

    return _plan(_meta(aten.constant_pad_nd.default, (x, pad, value), {}), x, axis, band)


def _upsample(func: OpOverload, args: Args, kwargs: Kwargs) -> Plan | None:
    """Nearest upsampling by a whole factor `r`: output rows [a, b), on multiples of `r`, read
    input rows [a/r, b/r)."""
    x, made = args[0] if args else None, _meta(func, args, kwargs)
    if not isinstance(x, Tensor) or x.dim() < 3 or not made:
        return None
    axis, sizes = _axis(x), [int(n) for n in made[0].shape[2:]]
    r = sizes[axis - 2] // max(int(x.shape[axis]), 1)
    if r < 2 or r * x.shape[axis] != sizes[axis - 2]:
        return None

    def band(lo: int, hi: int) -> Tensor:
        cut = [hi - lo if i == axis - 2 else n for i, n in enumerate(sizes)]
        return func.overloadpacket.vec(x.narrow(axis, lo // r, (hi - lo) // r), cut, None)

    return _plan(made, x, axis, band, align=r)


def _group_norm(func: OpOverload, args: Args, kwargs: Kwargs) -> Plan | None:
    """Per image and group, mean and variance over every band in float32 or wider, merged as one
    pass over all of it would be (up to rounding); then each band normalized."""
    a = _arguments(func, args, kwargs)
    x, weight, bias = a.get("input"), a.get("weight"), a.get("bias")
    groups = int(cast(int, a.get("num_groups", a.get("group")) or 1))
    eps = float(a.get("eps") or 1e-5)  # type: ignore[arg-type]
    if not isinstance(x, Tensor) or x.dim() < 3 or x.shape[1] % groups:
        return None
    axis, made, batch = _axis(x), _meta(func, args, kwargs), int(x.shape[0])
    per, height, total = int(x.shape[1]) // groups, int(x.shape[axis]), x.numel() // batch // groups
    wide, shape = torch.promote_types(x.dtype, torch.float32), [batch, -1] + [1] * (x.dim() - 2)
    held: list[Tensor] = []  # scale, shift, mean, rstd

    def gather(rows: int) -> None:
        parts = []
        for lo in range(0, height, rows):
            flat = x.narrow(axis, lo, min(rows, height - lo)).to(wide).reshape(batch, groups, -1)
            parts.append((*torch.var_mean(flat, dim=-1, correction=0), flat.shape[-1] / total))
        mean = sum(m * n for _, m, n in parts)
        rstd = torch.rsqrt(sum((v + (m - mean) ** 2) * n for v, m, n in parts) + eps)
        scale = rstd.repeat_interleave(per, dim=1)
        scale = scale * weight.to(wide) if isinstance(weight, Tensor) else scale
        shift = -mean.repeat_interleave(per, dim=1) * scale
        shift = shift + bias.to(wide) if isinstance(bias, Tensor) else shift
        held[:] = [scale.reshape(shape), shift.reshape(shape), mean, rstd]

    def band(lo: int, hi: int) -> Tensor:
        return torch.addcmul(held[1], x.narrow(axis, lo, hi - lo), held[0])

    def finish(out: Tensor) -> object:
        stats = (held[2].to(made[1].dtype), held[3].to(made[2].dtype)) if len(made) > 2 else ()
        return (out, *stats) if stats else out

    return _plan(made, x, axis, band, gather=gather, finish=finish)


_UPSAMPLES = [f"{p}{d}d" for p in ("upsample_nearest", "_upsample_nearest_exact") for d in "123"]
_SPEC = {
    _pointwise: "silu gelu add mul",
    _conv: "convolution conv1d conv2d conv3d",
    _pad: "constant_pad_nd pad",
    _upsample: " ".join(_UPSAMPLES),
    _group_norm: "group_norm native_group_norm",
}
catch.RULES.update({getattr(aten, n): r for r, ns in _SPEC.items() for n in ns.split()})


# --------------------------------------------------------------------------- call cuts


def install(module: nn.Module) -> int:
    """Call cuts under `module`: token pieces for Diffusers' FFNs and their activation
    projections (GELU, GEGLU, SwiGLU and subclasses; a block that calls its FFN's parts still
    cuts at the projection, whose [tokens, inner] intermediate is the widest), and row bands
    for its channel RMS norms. Each site is the module's path. Returns how many it cut."""
    count = 0
    for path, sub in module.named_modules():
        kinds = [c.__name__ for c in type(sub).__mro__ if c.__module__.startswith("diffusers.")]
        rms = _rms_contract(sub, kinds)
        if rms is not None and "_cozy_cut" not in vars(sub):
            sub.forward = _rms(f"{path}:rows", *rms, sub.forward)
            vars(sub)["_cozy_cut"] = True
            count += 1
        if any(k in _FFN for k in kinds) and "_cozy_cut" not in vars(sub):
            sub.forward = _ffn(f"{path}:tokens", sub.forward)
            vars(sub)["_cozy_cut"] = True
            count += 1
    return count


def _ffn(site: str, forward: Callable[..., object]) -> Callable[..., object]:
    whole: list[bool] = []  # a piece returned no tensor: it runs whole from then on

    def cut(x: object, *args: object, **kwargs: object) -> object:
        if args or kwargs or whole or not isinstance(x, Tensor) or x.dim() < 2:
            return forward(x, *args, **kwargs)
        tokens = x.numel() // max(x.shape[-1], 1)
        if tokens < 2:
            return forward(x)
        try:
            run = functools.partial(_tokens, forward, x, tokens)
            return catch.cut(site, levels(tokens), run, (x,))
        except _Whole:
            whole.append(True)
        return forward(x)

    return cut


class _Whole(Exception):
    """A cut's pieces do not join (a call returned no tensor): it runs whole."""


def _tokens(forward: Callable[..., object], x: Tensor, tokens: int, level: int) -> object:
    if not level:
        return forward(x)
    size, flat, out = pieces(tokens, level), x.reshape(tokens, x.shape[-1]), None
    for at in range(0, tokens, size):
        part = forward(flat[at : at + size])
        if not isinstance(part, Tensor):
            raise _Whole
        out = part.new_empty((tokens, *part.shape[1:])) if out is None else out
        out[at : at + part.shape[0]].copy_(part)
        del part
    assert out is not None
    return out.view(*x.shape[:-1], *out.shape[1:])


def _rms_contract(module: nn.Module, kinds: Sequence[str]) -> tuple[bool, int, int] | None:
    """Diffusers' channel RMS norm: channel-only affine terms, no spatial reduction."""
    kind = next((name for name in kinds if name.endswith("RMS_norm")), None)
    gamma, bias = getattr(module, "gamma", None), getattr(module, "bias", None)
    first, scale = getattr(module, "channel_first", None), getattr(module, "scale", None)
    if kind is None or type(first) is not bool or not isinstance(gamma, Tensor):
        return None
    shape = tuple(gamma.shape)
    if not shape or (any(n != 1 for n in shape[1:]) if first else len(shape) != 1):
        return None
    channels = int(shape[0])
    if channels < 1 or not isinstance(scale, float) or scale != channels**0.5:
        return None
    if isinstance(bias, Tensor):
        if bias.shape != gamma.shape:
            return None
    elif bias != 0.0:
        return None
    return first, channels, gamma.dim()


def _rms(
    site: str, first: bool, channels: int, affine_dims: int, forward: Callable[..., object]
) -> Callable[..., object]:
    def cut(x: object, *args: object, **kwargs: object) -> object:
        if (args or kwargs or not isinstance(x, Tensor) or x.layout is not torch.strided
                or not x.is_contiguous() or torch.is_grad_enabled() or x.dim() < 3):
            return forward(x, *args, **kwargs)
        normalized = 1 if first else x.dim() - 1
        if x.shape[normalized] != channels or (first and x.dim() != affine_dims + 1):
            return forward(x)
        axis = max((i for i in range(1, x.dim()) if i != normalized), key=lambda i: x.shape[i])
        height = int(x.shape[axis])
        if height < 2:
            return forward(x)
        try:
            run = functools.partial(_rms_bands, forward, x, axis)
            return catch.cut(site, levels(height), run, (x,))
        except _Whole:
            return forward(x)

    return cut


def _rms_bands(forward: Callable[..., object], x: Tensor, axis: int, level: int) -> object:
    height = int(x.shape[axis])
    if not level:
        return forward(x)
    rows, out = pieces(height, level), None
    for at in range(0, height, rows):
        length = min(rows, height - at)
        part = forward(x.narrow(axis, at, length))
        expected = (*x.shape[:axis], length, *x.shape[axis + 1 :])
        if not isinstance(part, Tensor) or part.shape != expected:
            raise _Whole
        out = part.new_empty(x.shape) if out is None else out
        if part.dtype != out.dtype or part.device != out.device:
            raise _Whole
        out.narrow(axis, at, length).copy_(part)
        del part
    return out


def attend(
    call: Attention,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    *,
    rows: int = 0,
    heads: bool = True,
) -> Tensor:
    """`call(q, k, v, mask)` over [batch, sequence, heads, dim], in pieces at its site's level,
    each computed as the whole computes it: query heads with the K/V heads they read (whole K/V
    groups, then heads sharing one), when `heads` (the kernel reduces within a head); then query
    rows on multiples of `rows` (the kernel's query block; 0: never), each beside every key."""
    site = catch.site("attention")
    if site is None:
        return call(q, k, v, mask)
    count, length = int(q.shape[2]), int(q.shape[1])
    group = max(count // int(k.shape[2]), 1)
    if mask is not None and mask.dim() != 4:
        rows, heads = 0, False
    cuts = [(h, length) for h in (_head_units(count, group) if heads else [count])]
    while 0 < rows < cuts[-1][1]:
        cuts.append((cuts[-1][0], max(rows, cuts[-1][1] // 2 // rows * rows)))

    def piece(level: int) -> Tensor:
        return _attend(call, q, k, v, mask, group, *cuts[level])

    return catch.cut(site, len(cuts) - 1, piece, (q, k, v))


@functools.cache
def _head_units(count: int, group: int) -> list[int]:
    """Query heads per piece that keep each piece on whole K/V groups or inside one."""
    wholes = {group * m for m in range(1, count // group + 1)}
    return sorted(wholes | {d for d in range(1, group + 1) if group % d == 0}, reverse=True)


def _attend(
    call: Attention, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, *cut: int
) -> Tensor:
    """Query heads [h, h + hp) with the K/V heads they read, by query rows [s, s + rp)."""
    group, hp, rp = cut
    batch, length, count = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    if hp >= count and rp >= length:
        return call(q, k, v, mask)
    whole = mask.expand(-1, count, length, -1) if mask is not None else None
    out: Tensor | None = None
    for h in range(0, count, hp):
        n, lo, hi = min(hp, count - h), h // group, (min(h + hp, count) - 1) // group + 1
        for s in range(0, length, rp):
            part = whole[:, h : h + n, s : s + rp] if whole is not None else None
            piece = call(q[:, s : s + rp, h : h + n], k[:, :, lo:hi], v[:, :, lo:hi], part)
            out = piece.new_empty((batch, length, count, *piece.shape[3:])) if out is None else out
            out[:, s : s + rp, h : h + n].copy_(piece)
            del piece
    assert out is not None
    return out


def split[T](name: str, key: str, total: int, finest: int, run: Callable[[int], T]) -> T:
    """A package's `split` (`cozy_runtime.author.split`): `run(most)` runs it in windows of at
    most `most` of its `total` elements; `finest` is the least a window can hold."""

    def at(level: int) -> T:
        return run(max(pieces(total, level), finest))

    return catch.cut(f"split:{key}", levels(-(-total // max(finest, 1))), at, ())


def launch[T](operands: Sequence[object], run: Callable[[], T]) -> T:
    """A launch no dispatch sees (a Triton kernel's): out of memory it did not run, so it runs
    again as an op does (12586: a glue kernel on a full GPU, after a collective)."""
    return catch.attempt(run, operands)
