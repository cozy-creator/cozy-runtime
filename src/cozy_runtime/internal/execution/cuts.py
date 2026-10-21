"""Exact cuts (varena-slim §2): each computes what the whole computes, results on the device, at
the level Rust holds for its site (`r.cut`). Level L is 2^L pieces, down to the finest.

Op bands (the layer's, `catch.RULES`): conv only, by images, else by output channels (whole
groups), each band written by the conv's `out` form into its slice of the result, so its
workspace shrinks with the band (measured on a GPU: tests/test_band_live_bytes.py). Nearest
upsampling and group norm keep no working memory beyond their results: bands save nothing
there, and Rust cuts the call around them (a package's `split`). SiLU and GELU run in place when
their input is dead (`catch`), else Rust parks.
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


#: What a band writes: the op on [lo, hi) of its result's band axis, straight into its slice of
#: the op's preallocated result, through the op's `out` form.
type Write = Callable[[int, int, Tensor], None]


@dataclass(frozen=True, slots=True)
class Plan:
    """An op in bands of its result's outermost axis of more than one (a batch, else the
    channels: a contiguous slice), `align` of them a unit. `write` runs the op on its operands'
    matching part straight into its slice of the preallocated result: no band holds a result of
    its own."""

    x: Tensor
    made: Tensor  # the op's result, meta
    axis: int
    write: Write
    align: int = 1

    @property
    def units(self) -> int:
        return -(-int(self.made.shape[self.axis]) // self.align)

    @property
    def finest(self) -> int:
        return levels(self.units)

    def __call__(self, own: Scope) -> Tensor:
        out = catch.settle(functools.partial(_empty, self.made, self.x.device), None, self.x)
        operands = (self.x, out)  # never parked while a band writes them
        at = 0
        while at < self.units:
            band = functools.partial(self._band, out, at, own)
            at = catch.settle(band, own, operands, first=True)
        return out

    def _band(self, out: Tensor, at: int, own: Scope) -> int:
        stop = min(at + pieces(self.units, own.level), self.units)
        self.write(at * self.align, stop * self.align, out)
        return stop


def _plan(made: list[Tensor], x: Tensor, write: Write, align: int = 1) -> Plan | None:
    """Over the result's outermost axis of more than one unit, when it is contiguous."""
    if len(made) != 1 or not made[0].is_contiguous():
        return None
    axis = next((i for i, n in enumerate(made[0].shape[:2]) if n > 1), None)
    if axis is None or made[0].shape[axis] < 2 * align:
        return None
    return Plan(x, made[0], axis, write, align)


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


def _conv(func: OpOverload, args: Args, kwargs: Kwargs) -> Plan | None:
    """By images, else by output channels (whole groups of them, each reading its groups' input
    channels): what the whole conv computes for them."""
    a = _arguments(func, args, kwargs)
    x, w, bias = a.get("input"), a.get("weight"), a.get("bias")
    groups = int(cast(int, a.get("groups") or 1))
    if not isinstance(x, Tensor) or not isinstance(w, Tensor) or a.get("transposed"):
        return None
    out, made = cast(OpOverload, aten.convolution.out), _meta(func, args, kwargs)
    rest = [a[k] for k in ("stride", "padding", "dilation", "transposed", "output_padding")]
    per_out, per_in = int(w.shape[0]) // groups, int(x.shape[1]) // groups
    by_image = bool(made) and made[0].shape[0] > 1

    def write(lo: int, hi: int, result: Tensor) -> None:
        if by_image:
            out(x[lo:hi], w, bias, *rest, groups, out=result[lo:hi])
            return
        part = bias[lo:hi] if isinstance(bias, Tensor) else None
        g0, g1 = lo // per_out, hi // per_out
        inputs = x if groups == 1 else x[:, g0 * per_in : g1 * per_in]
        out(inputs, w[lo:hi], part, *rest, 1 if groups == 1 else g1 - g0, out=result[:, lo:hi])

    return _plan(made, x, write, 1 if by_image or groups == 1 else per_out)


catch.RULES[aten.convolution] = _conv


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
        catch.piece()
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
        catch.piece()
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
    each computed as the whole computes it. Level L is 2^L pieces of whole K/V groups (their
    query heads with them) when `heads` (the kernel reduces within a head); past one group a
    piece, the rows halve: query rows on multiples of `rows` (the kernel's query block; 0:
    never), each beside every key."""
    site = catch.site("attention")
    if site is None:
        return call(q, k, v, mask)
    count, length, kv = int(q.shape[2]), int(q.shape[1]), int(k.shape[2])
    group = max(count // kv, 1)
    if mask is not None and mask.dim() != 4:
        rows, heads = 0, False
    by_heads = levels(kv) if heads else 0
    blocks = -(-length // rows) if rows else 1

    def piece(level: int) -> Tensor:
        across = min(level, by_heads)
        hp = pieces(kv, across) * group if heads else count
        rp = pieces(blocks, level - across) * rows if rows else length
        return _attend(call, q, k, v, mask, group, hp, rp)

    return catch.cut(site, by_heads + levels(blocks), piece, (q, k, v))


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
            catch.piece()
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
