"""Cuts preserve reduction and quantization domains, with floating-point rounding tolerance.

Purity alone is insufficient: tensor-global activation scales and RNG state forbid cuts.
Changing GEMM row geometry may change rounding; bitwise equivalence is not promised.
The results stay on the device, at the level each call's operands need against the request's
room (`catch`, `sizing`). Level L is 2^L pieces, down to the finest.

Call cuts: Diffusers' FFNs by token, its channel RMS norms by rows, attention by head groups
then query rows, a package's `split` by windows.
"""

from __future__ import annotations

import functools
import math
import sys
import threading
from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch import Tensor, nn

import cozy_runtime.internal.execution.catch as catch
import cozy_runtime.internal.execution.sizing as sizing
from cozy_runtime.internal.encoding.leaves import NVFP4
from cozy_runtime.internal.execution.sizing import levelled, levels, pieces

type Band = Callable[[int, int], Tensor]
type Attention = Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor]
#: diffusers' token-wise feed-forwards and the activation projections inside them
_FFN = frozenset({"FeedForward", "GELU", "GEGLU", "SwiGLU", "ApproximateGELU", "LinearActivation"})


# --------------------------------------------------------------------------- call cuts


def install(module: nn.Module) -> int:
    """Call cuts under `module`: token pieces for Diffusers' FFNs and their activation
    projections (GELU, GEGLU, SwiGLU and subclasses; a block that calls its FFN's parts still
    cuts at the projection, whose [tokens, inner] intermediate is the widest), row bands for its
    channel RMS norms, and image or output-channel bands for its 2-D and 3-D convs (at the
    conv itself, so a subclass's forward around it, a causal pad, is untouched). Each site is
    the module's path. Returns how many it cut."""
    count = 0
    for path, sub in module.named_modules():
        if (
            isinstance(sub, nn.Conv2d | nn.Conv3d)
            and "_cozy_cut" not in vars(sub)
            and "_conv_forward" not in vars(sub)
            and type(sub)._conv_forward in (nn.Conv2d._conv_forward, nn.Conv3d._conv_forward)
        ):
            sub._conv_forward = _conv_bands(f"{path}:bands", sub, _conv_of(sub))
            vars(sub)["_cozy_cut"] = True
            count += 1
        # torch's RMS norm reduces each row alone: token pieces are the whole (v7 H3 13826 ran
        # out in a DiT block's 546 MiB rms_norm, a site with no cut).
        if type(sub) is nn.RMSNorm and "forward" not in vars(sub) and "_cozy_cut" not in vars(sub):
            norm: nn.Module = sub
            norm.forward = catch.transparent(_norm(f"{path}:tokens", sub, _forward(sub)))
            vars(sub)["_cozy_cut"] = True
            count += 1
            continue
        # A custom subclass/instance forward is package code, even when its base is pure.
        if "forward" in vars(sub) or not type(sub).forward.__module__.startswith("diffusers."):
            continue
        kinds = [c.__name__ for c in type(sub).__mro__ if c.__module__.startswith("diffusers.")]
        rms = _rms_contract(sub, kinds)
        if rms is not None and "_cozy_cut" not in vars(sub):
            sub.forward = catch.transparent(_rms(f"{path}:rows", sub, *rms, _forward(sub)))
            vars(sub)["_cozy_cut"] = True
            count += 1
        if any(k in _FFN for k in kinds) and "_cozy_cut" not in vars(sub):
            sub.forward = catch.transparent(_ffn(f"{path}:tokens", sub, _forward(sub)))
            vars(sub)["_cozy_cut"] = True
            count += 1
    return count


type Conv = Callable[[Tensor, Tensor, Tensor | None], Tensor]


def _conv_of(module: nn.Module) -> Conv:
    """`module`'s class `_conv_forward` as it is at each call."""
    return lambda x, w, b: type(module)._conv_forward(module, x, w, b)


def _conv_bands(site: str, module: nn.Module, conv: Conv) -> Conv:
    """A conv at the level the process keeps for `site`: whole at 0, else in bands of images, else
    (one image, ungrouped) of output channels, each band copied into its slice of the result.
    Measured on a GPU (tests/test_band_live_bytes.py): the band's workspace is what shrinks."""

    def cut(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        batched = x.dim() == weight.dim()
        by_images = batched and x.shape[0] > 1
        # the conv's own groups apply to every call: a channel band of a grouped one is not
        # its conv; an encoded leaf does not slice
        whole_groups = module.groups == 1 and type(weight) in (Tensor, nn.Parameter)
        units = x.shape[0] if by_images else weight.shape[0] if batched and whole_groups else 1
        if (
            units < 2
            or torch.is_grad_enabled()
            or type(module)._conv_forward not in (nn.Conv2d._conv_forward, nn.Conv3d._conv_forward)
        ):
            return conv(x, weight, bias)
        run = functools.partial(_bands, conv, x, weight, bias, by_images, units)
        need = levelled(units, _conv_need(module, x, units, by_images))
        return catch.cut(site, levels(units), need, run)

    return cut


def _conv_need(module: nn.Module, x: Tensor, units: int, by_images: bool) -> sizing.Per:
    """A band's bytes: the whole output, held throughout; the band's output; and, where the conv
    pads by copying (a padding mode other than zeros), the band's padded input (an image band's
    images, or the whole input again for a channel band). cuDNN's workspace comes from the
    agent's float (its bound is the GPU round's measure)."""
    dims, b = len(module.kernel_size), x.element_size()
    spatial = [int(n) for n in x.shape[-dims:]]
    if module.padding == "same":
        sizes = spatial
    else:
        pad = (0,) * dims if module.padding == "valid" else module.padding
        sizes = [
            (n + 2 * p - d * (k - 1) - 1) // s + 1
            for n, p, d, k, s in zip(
                spatial, pad, module.dilation, module.kernel_size, module.stride, strict=True
            )
        ]
    lead = math.prod(int(n) for n in x.shape[: -dims - 1])
    out = lead * module.out_channels * math.prod(sizes) * b
    copied = module.padding_mode != "zeros"
    whole_in = x.numel() * b

    def piece(n: int, granularity: int) -> int:
        copy = (n * whole_in // units if by_images else whole_in) if copied else 0
        return sizing.peak(granularity, out, (copy, n * out // units))

    return piece


def _bands(
    conv: Conv,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    by_images: bool,
    units: int,
    level: int,
) -> Tensor:
    if not level:
        return conv(x, weight, bias)
    size, out, axis = pieces(units, level), None, 0 if by_images else 1
    for at in range(0, units, size):
        catch.piece()
        if by_images:
            part = conv(x[at : at + size], weight, bias)
        else:
            band = slice(at, at + size)
            part = conv(x, weight[band], bias[band] if bias is not None else None)
        if out is None:
            shape = list(part.shape)
            shape[axis] = units
            out = part.new_empty(shape)
        out.narrow(axis, at, part.shape[axis]).copy_(part)
        del part
    assert out is not None
    return out


def _forward(module: nn.Module) -> Callable[..., object]:
    """A forward set on the instance as it is now, else its class's at each call."""
    return vars(module).get("forward") or catch.current(module)


def _ffn(site: str, module: nn.Module, forward: Callable[..., object]) -> Callable[..., object]:
    whole: list[bool] = []  # a piece returned no tensor: it runs whole from then on

    def cut(x: object, *args: object, **kwargs: object) -> object:
        if (
            args
            or kwargs
            or whole
            or not isinstance(x, Tensor)
            or x.dim() < 2
            or module.training
            or torch.is_grad_enabled()
            or not type(module).forward.__module__.startswith("diffusers.")
            or not _token_local(module)
        ):
            return forward(x, *args, **kwargs)
        tokens = x.numel() // max(x.shape[-1], 1)
        if tokens < 2:
            return forward(x)
        shape = _ffn_shape(module, x)
        if shape is None:
            whole.append(True)
            print(
                f"cozy-runtime: {site} has no feed-forward layout it reads: whole", file=sys.stderr
            )
            return forward(x)
        try:
            run = functools.partial(_tokens, forward, x, tokens)
            return catch.cut(site, levels(tokens), levelled(tokens, shape.need(tokens)), run)
        except _Whole:
            whole.append(True)
        return forward(x)

    return cut


#: activations computing in one kernel (one `n x inner` temporary); the others hold two
_FUSED = frozenset({"GELU"})
_GATED = frozenset({"GEGLU", "SwiGLU"})


def _ffn_shape(module: nn.Module, x: Tensor) -> sizing.Ffn | None:
    """A Diffusers feed-forward's widths, read from the linears it has: its `net`'s activation
    projection (`proj`) and down projection, an activation projection alone (whose output is its
    inner width), or up projections from the input's width beside one down projection (Z-Image's
    and Bria FIBO's gated `w1`/`w3` and `w2`); the activation's temporaries; and whether an
    encoding leaf quantizes the linears' input rows. None when it has none of these."""
    din, net = int(x.shape[-1]), getattr(module, "net", None)
    act = net[0] if isinstance(net, nn.ModuleList) else module
    kinds = {c.__name__ for c in type(act).__mro__}
    projection = _widths(getattr(act, "proj", None))
    if projection is not None:
        projected = projection[1]
        inner = projected // 2 if kinds & _GATED else projected
        nets = [w for m in net if (w := _widths(m))] if isinstance(net, nn.ModuleList) else []
        dout = nets[-1][1] if nets else inner
        temps = 1 if kinds & _FUSED and not kinds & _GATED else 2
    else:
        widths = [w for m in module.children() if (w := _widths(m))]
        ups, down = [o for i, o in widths if i == din], [w for w in widths if w[0] != din]
        if not ups or len(down) != 1:
            return None
        projected, (inner, dout), temps = sum(ups), down[0], 2
    return sizing.Ffn(
        din=din,
        dout=dout,
        inner=inner,
        projected=projected,
        temps=temps,
        itemsize=x.element_size(),
        quantized=any(getattr(c, "provider", None) is not None for c in module.modules()),
    )


def _widths(module: object) -> tuple[int, int] | None:
    """A projection's (in, out) widths as it states them: a Linear, or an encoding leaf in its
    place; None for anything else."""
    i, o = (getattr(module, k, None) for k in ("in_features", "out_features"))
    return (i, o) if isinstance(i, int) and isinstance(o, int) else None


def _token_local(module: nn.Module) -> bool:
    """NVFP4 uses a tensor-global activation scale and changes backend at small M.

    Keep the complete call even if a later kernel build would enable that arm. No chunk
    may silently select a different quantizer/backend. Other approved row quantizers keep
    their full channel/block reduction within each token.
    """
    return not any(getattr(child, "provider", None) == NVFP4 for child in module.modules())


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


def _norm(site: str, module: nn.Module, forward: Callable[..., object]) -> Callable[..., object]:
    """A norm over its trailing `normalized_shape` in pieces of whole rows (tokens)."""
    dims = len(cast("tuple[int, ...]", module.normalized_shape))

    def cut(x: object, *args: object, **kwargs: object) -> object:
        if (
            args
            or kwargs
            or not isinstance(x, Tensor)
            or torch.is_grad_enabled()
            or x.dim() <= dims
        ):
            return forward(x, *args, **kwargs)
        row = math.prod(int(n) for n in x.shape[-dims:])
        tokens = x.numel() // max(row, 1)
        if tokens < 2:
            return forward(x)
        flat = x.reshape(tokens, *x.shape[-dims:])
        run = functools.partial(_rows, forward, flat, tokens)
        # torch's rms_norm holds an fp32 upcast of its rows where it is not fused
        need = levelled(tokens, sizing.norm(tokens, row, x.element_size(), 4))
        out = catch.cut(site, levels(tokens), need, run)
        return cast(Tensor, out).view(x.shape)

    return cut


def _rows(forward: Callable[..., object], flat: Tensor, tokens: int, level: int) -> object:
    if not level:
        return forward(flat)
    size, out = pieces(tokens, level), None
    for at in range(0, tokens, size):
        catch.piece()
        part = cast(Tensor, forward(flat[at : at + size]))
        out = part.new_empty((tokens, *part.shape[1:])) if out is None else out
        out[at : at + part.shape[0]].copy_(part)
        del part
    return out


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
    site: str,
    module: nn.Module,
    first: bool,
    channels: int,
    affine_dims: int,
    forward: Callable[..., object],
) -> Callable[..., object]:
    def cut(x: object, *args: object, **kwargs: object) -> object:
        if (
            args
            or kwargs
            or not isinstance(x, Tensor)
            or x.layout is not torch.strided
            or not x.is_contiguous()
            or torch.is_grad_enabled()
            or x.dim() < 3
            or not type(module).forward.__module__.startswith("diffusers.")
        ):
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
            # Diffusers' channel RMS normalizes, then scales: two fp32-wide temporaries a value
            need = levelled(height, sizing.norm(height, x.numel() // height, x.element_size(), 8))
            return catch.cut(site, levels(height), need, run)
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
    scratch: str = "flash",
) -> Tensor:
    """`call(q, k, v, mask)` over [batch, sequence, heads, dim], in pieces at the level its
    operands need against the request's room, each computed as the whole computes it. Level L is
    2^L pieces of whole K/V groups (their query heads with them) when `heads` (the kernel reduces
    within a head); past one group a piece, the rows halve: query rows on multiples of `rows`
    (the kernel's query block; 0: never), each beside every key. `scratch` is the backend's
    working memory (`sizing.Attention`): flash-like, sage's quantized q/k, or math's scores."""
    site = catch.site("attention")
    if site is None:
        return call(q, k, v, mask)
    count, length, kv = int(q.shape[2]), int(q.shape[1]), int(k.shape[2])
    group = max(count // kv, 1)
    if mask is not None and mask.dim() != 4:
        rows, heads = 0, False
    by_heads = levels(kv) if heads else 0
    blocks = -(-length // rows) if rows else 1

    def cut(level: int) -> tuple[int, int]:
        across = min(level, by_heads)
        hp = pieces(kv, across) * group if heads else count
        rp = pieces(blocks, level - across) * rows if rows else length
        return hp, rp

    shape = sizing.Attention(
        int(q.shape[0]),
        length,
        count,
        kv,
        int(k.shape[1]),
        int(q.shape[3]),
        q.element_size(),
        scratch,
        kv_copies=1,
    )

    def need(level: int, granularity: int) -> int:
        return shape.need(*cut(level), granularity)

    def piece(level: int) -> Tensor:
        return _attend(call, q, k, v, mask, group, *cut(level))

    return catch.cut(site, by_heads + levels(blocks), need, piece)


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


#: per split key, its op's measured bytes at its first two windows: (window elements, the
#: section's peak less what was live at its start and the output). A measurement on the live
#: tensors, never an OOM; fitted once both are in (`sizing.fitted`)
_MEASURED: dict[str, list[tuple[int, int]]] = {}
#: split keys whose larger measuring window the device refused: the second point is a smaller one
_REFUSED: set[str] = set()
_MEASURED_LOCK = threading.Lock()


def split[T](
    name: str, key: str, total: int, finest: int, out_bytes: int, run: Callable[[int], T]
) -> T:
    """A package's `split` (`cozy_runtime.author.split`): `run(most)` runs it in windows of at
    most `most` of its `total` elements; `finest` is the least a window can hold. Its author
    must preserve global statistics/quantizers and declare any overlap approximation; merely
    having no side effects does not make differently sized windows numerically equivalent.
    Its op's bytes are measured at its first two successful window sizes, then fitted: an
    op quadratic in its window scales as one. The first measurement groups at most 128
    minimum windows; allocation refusals use the ordinary descent to smaller windows.
    Like every split call, the declared operation must be safe to retry without side effects."""
    deepest = levels(-(-total // max(finest, 1)))

    def window(level: int) -> int:
        return max(pieces(total, level), finest)

    with _MEASURED_LOCK:
        points = sorted(_MEASURED.get(key, ()))
        refused = key in _REFUSED
    request = catch.CURRENT.get()
    measured: tuple[int, int] | None = None

    def at(level: int) -> T:
        nonlocal measured
        if len(points) == 2 or request is None or deepest == 0:
            return run(window(level))
        request.reset_peak()
        start = request.reset_peak()  # Restart after each refused attempt, not before descent.
        result = run(window(level))
        grew = max(request.reset_peak() - start - out_bytes, 0)
        measured = (window(level), grew)
        return result

    if len(points) == 2:
        fit = sizing.fitted(points[0], points[1])

        def need(level: int, granularity: int) -> int:
            return sizing.peak(granularity, out_bytes, (fit(window(level)),))

        return catch.cut(f"split:{key}", deepest, need, at)
    # A whole long sequence at one token per launch is expensive even when it fits.
    # Start bounded, then measure the next larger window; where the device refused it (a short
    # axis: H3's 56 heads start at 28 a window), the next smaller. OOM still descends.
    if not points:
        target = min(deepest, max(deepest - 7, 1))
    elif refused:
        target = next((lv for lv in range(deepest + 1) if window(lv) < points[0][0]), deepest)
    else:
        target = next((lv for lv in range(deepest, -1, -1) if window(lv) > points[-1][0]), 0)

    def measuring(level: int, granularity: int) -> int:
        return 0 if level >= target else sys.maxsize  # a coarser window is not yet measured

    if request is None or deepest == 0:
        return catch.cut(f"split:{key}", deepest, measuring, at)
    result = catch.cut(f"split:{key}", deepest, measuring, at)
    assert measured is not None
    with _MEASURED_LOCK:
        known = _MEASURED.setdefault(key, [])
        if len(known) < 2 and all(measured[0] != w for w, _ in known):
            known.append(measured)
            known.sort()
        elif len(known) == 1:
            _REFUSED.add(key)  # the larger window descended to the known one
    return result
