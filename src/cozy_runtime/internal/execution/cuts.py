"""Cuts preserve reduction and quantization domains, with floating-point rounding tolerance.

Purity alone is insufficient: tensor-global activation scales and RNG state forbid cuts.
Changing GEMM row geometry may change rounding; bitwise equivalence is not promised.
The results stay on the device, at the level native selection takes from each call's operands
(`catch`, `sizing`). Level L is 2^L pieces, down to the finest.

Call cuts: Diffusers' FFNs, and any module shaped as one (`_ffn_like`), by token, a Runtime-fused
block's gated MLP by token (`gated`), its channel RMS norms by rows, what Diffusers' attention
processors run before dispatch (their q/k RMS norms by rows, their RoPE by sequence rows: `rope`),
attention by head groups then query rows, a Transformers vision tower by pictures, a package's
`split` by windows.
"""

from __future__ import annotations

import functools
import inspect
import math
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, cast

import torch
from torch import Tensor, nn

import cozy_runtime.internal.execution.catch as catch
import cozy_runtime.internal.execution.pinned as pinned
import cozy_runtime.internal.execution.sizing as sizing
from cozy_runtime.author._errors import is_capacity, unwind
from cozy_runtime.internal.encoding.leaves import NVFP4
from cozy_runtime.internal.execution.sizing import levelled, levels, pieces
from cozy_runtime.internal.residency import Request

type Band = Callable[[int, int], Tensor]


class Forward(Protocol):
    """A module's forward as a cut calls it: its input first, then whatever else it takes."""

    def __call__(self, x: object, *args: object, **kwargs: object) -> object: ...


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
    the module's path. A Transformers gated MLP cuts by tokens as Diffusers' FFNs do, and a
    Transformers vision tower that reads packed patches and each picture's
    grid (`forward(hidden_states, grid_thw)`: Qwen2.5-VL, Qwen3-VL, Qwen3.5) runs its pictures in
    groups. Diffusers' RMS norms (each attention processor's q/k norm) cut by rows, and the RoPE
    functions Diffusers' processors call (`rope`). A module of linears and activations alone, up
    projections from its input's width and one down projection (`_ffn_like`), is a feed-forward
    whoever wrote it, and cuts by tokens. Returns how many it cut."""
    count = _ropes()
    for path, sub in module.named_modules():
        if _gated_mlp(sub) and "_cozy_cut" not in vars(sub):
            sub.forward = catch.transparent(_ffn(f"{path}:tokens", sub, _forward(sub)))
            vars(sub)["_cozy_cut"] = True
            count += 1
            continue
        if _tower(sub) and "_cozy_cut" not in vars(sub):
            sub.forward = catch.transparent(_pictures(f"{path}:pictures", sub, _forward(sub)))
            vars(sub)["_cozy_cut"] = True
            count += 1
            continue
        if (
            isinstance(sub, nn.Conv2d | nn.Conv3d)
            and "_cozy_cut" not in vars(sub)
            and "_conv_forward" not in vars(sub)
            and type(sub)._conv_forward in (nn.Conv2d._conv_forward, nn.Conv3d._conv_forward)
        ):
            # The instance's own conv shadows its class's, with positional parameters.
            sub._conv_forward = _conv_bands(f"{path}:bands", sub, _conv_of(sub))  # type: ignore[method-assign,assignment]
            vars(sub)["_cozy_cut"] = True
            count += 1
        # torch's RMS norm reduces each row alone: token pieces are the whole (v7 H3 13826 ran
        # out in a DiT block's 546 MiB rms_norm, a site with no cut).
        if (
            (type(sub) is nn.RMSNorm or _diffusers_rms(sub))
            and "forward" not in vars(sub)
            and "_cozy_cut" not in vars(sub)
        ):
            norm: nn.Module = sub
            norm.forward = catch.transparent(_norm(f"{path}:tokens", sub, _forward(sub)))
            vars(sub)["_cozy_cut"] = True
            count += 1
            continue
        if _ffn_like(sub) and "_cozy_cut" not in vars(sub):
            library = type(sub).forward.__module__.startswith(_LIBRARIES)
            sub.forward = catch.transparent(_ffn(f"{path}:tokens", sub, _forward(sub), library))
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


type Convolution = nn.Conv2d | nn.Conv3d


def _conv_of(module: Convolution) -> Conv:
    """`module`'s class `_conv_forward` as it is at each call."""
    # Each conv class's `_conv_forward` takes its own instance, which `module` is.
    return lambda x, w, b: type(module)._conv_forward(module, x, w, b)  # type: ignore[arg-type]


def _conv_bands(site: str, module: Convolution, conv: Conv) -> Conv:
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


def _conv_need(module: Convolution, x: Tensor, units: int, by_images: bool) -> sizing.Per:
    """A band's bytes: the whole output, held throughout; the band's output; and, where the conv
    pads by copying (a padding mode other than zeros), the band's padded input (an image band's
    images, or the whole input again for a channel band). cuDNN's workspace is left to allocator
    reclaim and the measured sample."""
    dims, b = len(module.kernel_size), x.element_size()
    spatial = [int(n) for n in x.shape[-dims:]]
    if module.padding == "same":
        sizes = spatial
    else:
        # Neither "same" nor "valid": torch keeps a padding tuple.
        pad = (0,) * dims if module.padding == "valid" else cast("tuple[int, ...]", module.padding)
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

    def piece(n: int) -> tuple[int, ...]:
        copy = (n * whole_in // units if by_images else whole_in) if copied else 0
        return sizing.peak(out, (copy, n * out // units))

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
        catch.checkpoint()
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


def _forward(module: nn.Module) -> Forward:
    """A forward set on the instance as it is now, else its class's at each call."""
    return vars(module).get("forward") or catch.current(module)


def _ffn(site: str, module: nn.Module, forward: Forward, library: bool = True) -> Forward:
    """A feed-forward by tokens. `library`: its forward is Diffusers' or Transformers', which a
    refusal inside a piece may run again finer (`catch.cut`'s `pure`); a package's runs once."""
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
            or (library and not type(module).forward.__module__.startswith(_LIBRARIES))
            or not _token_local(module)
        ):
            return forward(x, *args, **kwargs)
        tokens = x.numel() // max(x.shape[-1], 1)
        if tokens < 2:
            return forward(x)
        shape = (_ffn_shape(module, x) if library else None) or _layout_shape(module, x)
        if shape is None:
            if library:
                whole.append(True)
                print(
                    f"cozy-runtime: {site} has no feed-forward layout it reads: whole",
                    file=sys.stderr,
                )
            return forward(x)
        try:
            run = functools.partial(_tokens, forward, x, tokens, rank=not library)
            need = levelled(tokens, shape.need(tokens))
            return catch.cut(site, levels(tokens), need, run, pure=library)
        except _Whole:
            whole.append(True)
        return forward(x)

    return cut


#: the libraries whose modules' forwards the Runtime reads as written
_LIBRARIES = ("diffusers.", "transformers.")
#: modules a feed-forward holds beside its linears: activations and no-ops (torch's, and the
#: libraries' own activation modules), which act on each element alone
_ELEMENTWISE = (
    nn.GELU, nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU, nn.Mish, nn.Tanh, nn.Sigmoid,
    nn.Hardswish, nn.Softplus, nn.Identity, nn.Dropout,
)  # fmt: skip
_ACTIVATION_MODULES = ("diffusers.models.activations", "transformers.activations")
#: what a forward that mixes tokens names: reductions over any axis, attention and matmuls
#: across rows, axis moves and convolutions. A feed-forward's forward names none of them.
_MIXING = frozenset({
    "mean", "sum", "prod", "var", "std", "norm", "amax", "amin", "max", "min", "argmax", "argmin",
    "softmax", "log_softmax", "logsumexp", "cumsum", "cumprod", "sort", "topk", "roll", "flip",
    "scaled_dot_product_attention", "matmul", "bmm", "baddbmm", "einsum", "mm", "transpose",
    "permute", "movedim", "swapaxes", "mT", "T", "conv1d", "conv2d", "conv3d", "avg_pool1d",
    "avg_pool2d", "max_pool1d", "max_pool2d", "interpolate", "group_norm", "batch_norm",
})  # fmt: skip


def _ffn_like(module: nn.Module) -> bool:
    """A feed-forward by its shape, whoever wrote it: its own forward over one input, linears and
    elementwise activations alone below it (`_layout`), and a forward that names nothing that mixes
    tokens (`_MIXING`). RelayVSR's Wan FFN (`net.0.proj`, tanh GELU, `net.2`) is one."""
    forward = type(module).forward
    if _widths(module) is not None or forward is nn.Module.forward:
        return False
    if forward.__module__.startswith("cozy_runtime."):
        return False
    if isinstance(module, nn.ModuleList) or _layout(module) is None:
        return False
    return _screened(type(module).forward) and (
        "forward" not in vars(module) or _screened(vars(module)["forward"])
    )


def _screened(forward: object) -> bool:
    """Whether `forward`'s code, and the code nested in it, names nothing in `_MIXING`."""
    while hasattr(forward, "__wrapped__"):
        forward = forward.__wrapped__
    code = getattr(forward, "__code__", None)
    if not inspect.iscode(code):
        return False
    stack = [code]
    while stack:
        at = stack.pop()
        if _MIXING.intersection(at.co_names):
            return False
        stack.extend(c for c in at.co_consts if inspect.iscode(c))
    return True


def _layout(module: nn.Module, width: int | None = None) -> tuple[int, int, tuple[int, int]] | None:
    """A feed-forward's widths from the linears under `module`: (din, projected, (inner, dout)),
    up projections from din (`width` where given; `projected` their outputs together) and one down
    projection from `inner`, which is `projected` or half of it (gated). None where anything else
    holds weights or computes: any module but a linear, an elementwise activation, a container,
    or a holder of those without parameters or a forward of its own (RelayVSR's `GELU`)."""
    linears: list[tuple[int, int]] = []
    stack = [module]
    while stack:
        sub = stack.pop()
        widths = _widths(sub) if sub is not module else None
        if widths is not None:
            linears.append(widths)
            continue
        if (
            next(sub.parameters(recurse=False), None) is not None
            or next(sub.buffers(recurse=False), None) is not None
        ):
            return None
        if sub is module:
            stack.extend(sub.children())
            continue
        elementwise = isinstance(sub, _ELEMENTWISE) or type(sub).__module__ in _ACTIVATION_MODULES
        if not (
            elementwise
            or isinstance(sub, nn.ModuleList | nn.Sequential)
            or type(sub).forward is nn.Module.forward
        ):
            return None
        stack.extend(sub.children())
    for din in sorted({i for i, _ in linears} if width is None else {width}):
        ups = [o for i, o in linears if i == din]
        down = [w for w in linears if w[0] != din]
        if ups and len(down) == 1 and 2 * down[0][0] in (2 * sum(ups), sum(ups)):
            return din, sum(ups), down[0]
    return None


def _layout_shape(module: nn.Module, x: Tensor) -> sizing.Ffn | None:
    """A feed-forward found by its shape (`_ffn_like`), sized for `x`; None when `x` is not its
    input's width. Its activation's temporaries are unknown: two, as an unfused one holds."""
    layout = _layout(module, int(x.shape[-1]))
    if layout is None:
        return None
    din, projected, (inner, dout) = layout
    return sizing.Ffn(
        din=din,
        dout=dout,
        inner=inner,
        projected=projected,
        temps=2,
        itemsize=x.element_size(),
        quantized=any(getattr(c, "provider", None) is not None for c in module.modules()),
    )


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


def gated(x: Tensor, linears: Sequence[nn.Module], forward: Callable[[Tensor], Tensor]) -> Tensor:
    """`forward(x)`: a Runtime-fused block's gated MLP over `x`'s token rows, `linears` its gate
    [and up] projections, then its down one. In token pieces at the level native selection takes
    under the current block, as Diffusers' FFNs (qwen-image-2's ten-reference edit, 65k tokens:
    [tokens, 12288] gate and up refused at 8 GiB, runs 46964, 47037). Whole outside a named block,
    under autograd, with an NVFP4 linear (`_token_local`) or one whose widths it can't read."""
    site = catch.site("mlp")
    widths = [_widths(m) for m in linears]
    tokens = x.numel() // max(int(x.shape[-1]), 1)
    if (
        site is None
        or tokens < 2
        or torch.is_grad_enabled()
        or any(w is None for w in widths)
        or any(getattr(m, "provider", None) == NVFP4 for m in linears)
    ):
        return forward(x)
    *ups, (inner, dout) = cast("list[tuple[int, int]]", widths)
    shape = sizing.Ffn(
        din=int(x.shape[-1]),
        dout=dout,
        inner=inner,
        projected=sum(o for _, o in ups),
        temps=1,
        itemsize=x.element_size(),
        quantized=any(getattr(m, "provider", None) is not None for m in linears),
    )
    run = functools.partial(_tokens, cast("Forward", forward), x, tokens)
    return cast(Tensor, catch.cut(site, levels(tokens), levelled(tokens, shape.need(tokens)), run))


class _Whole(Exception):
    """A cut's pieces do not join (a call returned no tensor): it runs whole."""


#: a vision tower's working bytes per patch, in widths of its hidden state: 16.5 measured on
#: Qwen3.5-9B's tower (4.6 GiB live for 129,600 patches of width 1152, bf16, run 46568), and
#: the rotary table it was refused growing
_TOWER_WIDTHS = 24


def _gated_mlp(module: nn.Module) -> bool:
    """A Transformers gated MLP (`down(act(gate(x)) * up(x))`, every Llama-, Qwen- and
    Mistral-style text encoder and LM): token by token, its [tokens, inner] intermediates the
    widest a long prompt's prefill holds."""
    if "forward" in vars(module) or not type(module).forward.__module__.startswith("transformers."):
        return False
    return {"gate_proj", "up_proj", "down_proj"} <= {name for name, _ in module.named_children()}


def _tower(module: nn.Module) -> bool:
    """A Transformers vision tower whose pictures attend only to their own patches: its forward
    reads packed patches, then each picture's (t, h, w) grid."""
    forward = type(module).forward
    if "forward" in vars(module) or not forward.__module__.startswith("transformers."):
        return False
    try:
        names = list(inspect.signature(forward).parameters)[1:3]
    except (TypeError, ValueError):
        return False
    return names == ["hidden_states", "grid_thw"]


def _pictures(site: str, module: nn.Module, forward: Forward) -> Forward:
    """A vision tower's pictures in groups: each attends only to its own patches (`cu_seqlens`),
    so a group's output rows are the whole call's rows for those pictures. Patch rows, the
    merged rows and any per-layer features join in picture order."""
    whole: list[bool] = []

    def cut(*given: object, **kwargs: object) -> object:
        args = list(given)
        patches = args.pop(0) if args else kwargs.pop("hidden_states", None)
        grid = args.pop(0) if args else kwargs.pop("grid_thw", None)
        if (
            args
            or whole
            or not isinstance(patches, Tensor)
            or not isinstance(grid, Tensor)
            or grid.dim() != 2
            or grid.shape[0] < 2
            or module.training
            or torch.is_grad_enabled()
        ):
            return forward(patches, grid, *args, **kwargs)
        rows = [int(n) for n in grid.prod(-1).tolist()]
        if sum(rows) != patches.shape[0]:
            return forward(patches, grid, **kwargs)
        config = getattr(module, "config", None)
        width = int(getattr(config, "hidden_size", patches.shape[-1]))
        each = _TOWER_WIDTHS * width * patches.element_size()
        out = sum(rows) * width * patches.element_size()

        def need(level: int) -> tuple[int, ...]:
            size = pieces(len(rows), level)
            return (out, max(sum(rows[at : at + size]) for at in range(0, len(rows), size)) * each)

        def run(level: int) -> object:
            if not level:
                return forward(patches, grid, **kwargs)
            size, parts, at = pieces(len(rows), level), [], 0
            for first in range(0, len(rows), size):
                catch.checkpoint()
                count = sum(rows[first : first + size])
                parts.append(
                    forward(patches[at : at + count], grid[first : first + size], **kwargs)
                )
                at += count
            return _join(parts)

        try:
            return catch.cut(site, levels(len(rows)), need, run)
        except _Whole:
            whole.append(True)
            print(f"cozy-runtime: {site}'s pieces do not join: whole", file=sys.stderr)
        return forward(patches, grid, **kwargs)

    return cast("Forward", cut)  # it reads the tower's arguments by position or by name


def _join(parts: list[object]) -> object:
    """Pieces' outputs as the whole call's: tensors in order along their rows, a model output's
    or a sequence's fields each so; None stays None. Anything else does not join (`_Whole`)."""
    first = parts[0]
    if isinstance(first, Tensor):
        return torch.cat(cast("list[Tensor]", parts), 0)
    if first is None:
        return None
    if isinstance(first, dict):  # Transformers' ModelOutput is an ordered dict of its fields
        fields = {key: _join([cast("dict[str, object]", p)[key] for p in parts]) for key in first}
        return type(first)(**fields)
    if isinstance(first, list | tuple):
        joined = [_join([cast("Sequence[object]", p)[i] for p in parts]) for i in range(len(first))]
        return type(first)(joined) if isinstance(first, list) else tuple(joined)
    raise _Whole


def _tokens(forward: Forward, x: Tensor, tokens: int, level: int, rank: bool = False) -> object:
    """`forward` over `x`'s token rows in 2^level pieces, each [n, width]; with `rank`, [1, …, n,
    width] as `x`'s rank is (a package's forward may read its input's dims)."""
    if not level:
        return forward(x)
    size, flat, out = pieces(tokens, level), x.reshape(tokens, x.shape[-1]), None
    lead = (1,) * (x.dim() - 2) if rank else ()
    for at in range(0, tokens, size):
        catch.checkpoint()
        rows = flat[at : at + size]
        part = forward(rows.view(*lead, *rows.shape))
        if not isinstance(part, Tensor) or part.shape[: len(lead)] != lead:
            raise _Whole
        part = part.view(-1, *part.shape[len(lead) + 1 :]) if lead else part
        if part.shape[0] != rows.shape[0]:
            raise _Whole
        out = part.new_empty((tokens, *part.shape[1:])) if out is None else out
        out[at : at + part.shape[0]].copy_(part)
        del part
    assert out is not None
    return out.view(*x.shape[:-1], *out.shape[1:])


def _diffusers_rms(module: nn.Module) -> bool:
    """Diffusers' `RMSNorm` as written: every row of its last axis alone (each attention
    processor's q/k norm, over [batch, tokens, heads, head_dim])."""
    kind = type(module)
    return kind.__name__ == "RMSNorm" and kind.__module__ == "diffusers.models.normalization"


def _norm(site: str, module: nn.Module, forward: Forward) -> Forward:
    """A norm over its trailing `normalized_shape` (torch's), or its last axis (Diffusers' RMSNorm),
    in pieces of whole rows (tokens). Diffusers' holds an fp32 upcast and its square, then an fp32
    product: two fp32 temporaries a value."""
    torch_norm = isinstance(module, nn.RMSNorm)
    dims = len(cast("tuple[int, ...]", module.normalized_shape)) if torch_norm else 1
    intermediate = 4 if torch_norm else 8

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
        need = levelled(tokens, sizing.norm(tokens, row, x.element_size(), intermediate))
        out = catch.cut(site, levels(tokens), need, run)
        return cast(Tensor, out).view(x.shape)

    return cut


def _rows(forward: Forward, flat: Tensor, tokens: int, level: int) -> object:
    if not level:
        return forward(flat)
    size, out = pieces(tokens, level), None
    for at in range(0, tokens, size):
        catch.checkpoint()
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
    forward: Forward,
) -> Forward:
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


def _rms_bands(forward: Forward, x: Tensor, axis: int, level: int) -> object:
    height = int(x.shape[axis])
    if not level:
        return forward(x)
    rows, out = pieces(height, level), None
    for at in range(0, height, rows):
        length = min(rows, height - at)
        catch.checkpoint()
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


class Rows(Protocol):
    """Where a RoPE call's sequence rows are: the axis of its input, or None (it runs whole)."""

    def __call__(
        self, x: Tensor, freqs: object, /, *args: object, **kwargs: object
    ) -> int | None: ...


class Rotation(Protocol):
    """A RoPE function: its input and tables first, then whatever else it takes."""

    def __call__(self, x: Tensor, freqs: object, /, *args: object, **kwargs: object) -> Tensor: ...


def _real_rows(x: Tensor, freqs: object, /, *args: object, **kwargs: object) -> int | None:
    """`diffusers.models.embeddings.apply_rotary_emb`: (cos, sin) tables, [S, D], over the axis
    it is told (`sequence_dim`, 2 by default). Its complex form (Lumina's) runs whole."""
    real = args[0] if args else kwargs.get("use_real", True)
    axis = args[2] if len(args) > 2 else kwargs.get("sequence_dim", 2)
    return axis if real is True and isinstance(axis, int) and axis in (1, 2) else None


def _complex_rows(x: Tensor, freqs: object, /, *args: object, **kwargs: object) -> int | None:
    """Qwen-Image's `apply_rotary_emb_qwen`: a complex table, [S, D/2], over [B, S, H, D]. Its
    real form lines its tables up with the heads, and runs whole."""
    real = args[0] if args else kwargs.get("use_real", True)
    return 1 if real is False else None


#: The RoPE functions Diffusers' attention processors call before dispatch, by the module that
#: defines each, its name there, and where a call's sequence rows are (`Rows`).
_QWEN = "diffusers.models.transformers.transformer_qwenimage"
_ROPES: tuple[tuple[str, str, Rows], ...] = (
    ("diffusers.models.embeddings", "apply_rotary_emb", _real_rows),
    (_QWEN, "apply_rotary_emb_qwen", _complex_rows),
    (f"{_QWEN}21", "apply_rotary_emb_qwen", _complex_rows),
)
#: a RoPE function's original, on its cut (`rope`)
_ORIGINAL = "_cozy_rope"
#: a piece's temporaries a rotated value: its fp32 upcast and the upcast rotation, a product of
#: each with its table, and the rotation itself
_ROTATED = 14


def _ropes() -> int:
    """Cut every `_ROPES` function this process has loaded: each where it is defined, and every
    loaded Diffusers module's name bound to it (a processor's module imported it by name), so a
    processor finds the cut whichever way it reaches it. Returns how many it cut now."""
    cut: dict[int, Rotation] = {}
    for name, attribute, rows in _ROPES:
        found = sys.modules.get(name)
        original = vars(found).get(attribute) if found is not None else None
        if not callable(original) or _ORIGINAL in vars(original):
            continue
        cut[id(original)] = rope(cast("Rotation", original), rows)
    if cut:
        for name, loaded in list(sys.modules.items()):
            if name.startswith("diffusers.") and loaded is not None:
                bound = vars(loaded)
                for key, value in list(bound.items()):
                    if id(value) in cut and callable(value):
                        bound[key] = cut[id(value)]
    return len(cut)


def rope(original: Rotation, rows: Rows) -> Rotation:
    """`original`, a RoPE function a Diffusers attention processor calls on its queries and keys
    before dispatch, in pieces of their sequence rows (`rows` says where they are; None: whole), at
    the level native selection takes under the current block. It rotates each value alone with
    its row's table entry, so a piece is the whole call's rows, byte for byte: PAN2's fp32
    temporaries were 0.8 GiB a call (cr-320). Whole outside a named block, under autograd, and for
    an input that is not contiguous [B, ·, ·, D] or tables that are not one row a token."""

    def cut(x: Tensor, freqs: object, /, *args: object, **kwargs: object) -> Tensor:
        axis = (
            rows(x, freqs, *args, **kwargs)
            if isinstance(x, Tensor) and x.dim() == 4 and x.is_contiguous()
            else None
        )
        site = None if axis is None or torch.is_grad_enabled() else catch.site("rope")
        if site is None or axis is None:
            return original(x, freqs, *args, **kwargs)
        length = int(x.shape[axis])
        tables = list(freqs) if isinstance(freqs, list | tuple) else [freqs]
        if length < 2 or not all(
            isinstance(t, Tensor) and t.dim() >= 1 and t.shape[0] == length for t in tables
        ):
            return original(x, freqs, *args, **kwargs)
        need = levelled(
            length, sizing.norm(length, x.numel() // length, x.element_size(), _ROTATED)
        )
        run = functools.partial(_rotated, original, x, freqs, axis, args, kwargs)
        return catch.cut(site, levels(length), need, run)

    functools.update_wrapper(cut, original)
    vars(cut)[_ORIGINAL] = original
    return cut


def _rotated(
    original: Rotation,
    x: Tensor,
    freqs: object,
    axis: int,
    args: Sequence[object],
    kwargs: dict[str, object],
    level: int,
) -> Tensor:
    if not level:
        return original(x, freqs, *args, **kwargs)
    length = int(x.shape[axis])
    size, out = pieces(length, level), None
    for at in range(0, length, size):
        catch.checkpoint()
        n = min(size, length - at)
        tables = (
            type(freqs)(cast(Tensor, t).narrow(0, at, n) for t in freqs)
            if isinstance(freqs, list | tuple)
            else cast(Tensor, freqs).narrow(0, at, n)
        )
        part = original(x.narrow(axis, at, n), tables, *args, **kwargs)
        if out is None:
            out = part.new_empty((*x.shape[:axis], length, *part.shape[axis + 1 :]))
        out.narrow(axis, at, n).copy_(part)
        del part
    assert out is not None
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
    """`call(q, k, v, mask)` over [batch, sequence, heads, dim], in pieces at the level native
    selection takes, each computed as the whole computes it. Level L is 2^L pieces of whole K/V
    groups (their query heads with them) when `heads` (the kernel reduces within a head); past one
    group a piece, the rows halve: query rows on multiples of `rows` (the kernel's query block; 0:
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
    )

    def need(level: int) -> tuple[int, ...]:
        return shape.need(*cut(level))

    def piece(level: int) -> Tensor:
        return _attend(call, q, k, v, mask, group, *cut(level))

    return catch.cut(site, by_heads + levels(blocks), need, piece)


def _attend(
    call: Attention, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, *cut: int
) -> Tensor:
    """Query heads [h, h + hp) with the K/V heads they read, by query rows [s, s + rp)."""
    group, hp, rp = cut
    length, count = int(q.shape[1]), int(q.shape[2])
    if hp >= count and rp >= length:
        return call(q, k, v, mask)
    whole = mask.expand(-1, count, length, -1) if mask is not None else None
    out = q.new_empty(q.shape)
    for h in range(0, count, hp):
        n, lo, hi = min(hp, count - h), h // group, (min(h + hp, count) - 1) // group + 1
        for s in range(0, length, rp):
            catch.checkpoint()
            part = whole[:, h : h + n, s : s + rp] if whole is not None else None
            piece = call(q[:, s : s + rp, h : h + n], k[:, :, lo:hi], v[:, :, lo:hi], part)
            out[:, s : s + rp, h : h + n].copy_(piece)
            del piece
    return out


def split[T](
    name: str,
    key: str,
    total: int,
    finest: int,
    out_bytes: int,
    item_bytes: float,
    holds: int,
    run: Callable[[int, bool], T],
    pure: bool = False,
    hostable: bool = False,
    exact: str = "",
) -> T:
    """A package's `split` (`cozy_runtime.author.split`): `run(most, host)` runs it in windows of
    at most `most` of its `total` elements; `finest` is the least a window can hold. Its author
    must preserve global statistics/quantizers and declare any overlap approximation; merely
    having no side effects does not make differently sized windows numerically equivalent.
    Package code runs once; `pure` says `run` is a Runtime leaf's, which runs again finer only
    when varena answers a refusal in it so (`catch.retry`), keeping the windows it finished
    (`author._split._Progress`). `out_bytes` is the whole output held beside the windows,
    `item_bytes` a window's output per element, `holds` how many of that its op's peak holds: its
    author's estimate, which a completed pass's measured sample replaces.

    Measure, then commit (cr-305, design `varena-activation-plan`): the whole is measured at its
    first allocation; from then on its measured bytes are the levels' need and it is allocated
    before any window runs (`ahead`), so a card short of it refuses before work is done. Where
    `hostable` and no level fits with the whole on the card, the whole goes to pinned host memory
    and the levels need only their windows. A host decision is made per use (var-030): its request
    keeps the largest room this input shape's whole was refused in (`Request.host_rooms`), and a
    call whose room is larger asks the card again; a refusal judged on the package's estimate
    holds for its call only. `exact` names this call's input shape, which the whole's measure and
    that room are kept under; `key` leaves out the cut extents, so the levels learned serve every
    length."""
    deepest = levels(-(-total // max(finest, 1)))
    site = f"split:{key}"
    request = catch.CURRENT.get()
    if request is None or not request.memory.cuda or _capturing():  # a capture allocates no host
        hostable = False
    exact = exact or key
    measured = _WHOLES.get(exact)
    whole_bytes = (  # a whole of windows the op returns on the host takes none of the card
        out_bytes if measured is None else _nbytes(measured[0], measured[1]) * _on_card(exact)
    )

    def window(level: int) -> int:
        return max(pieces(total, level), finest)

    def need(level: int) -> tuple[int, ...]:
        return (whole_bytes, *[int(window(level) * item_bytes)] * holds)

    def windows_only(level: int) -> tuple[int, ...]:
        return (*[int(window(level) * item_bytes)] * holds,)

    hsite = f"split:{exact}"  # a host decision is this shape's: another may fit the card
    room, on_host = 0, False
    if hostable and request is not None:
        # Per use, from measured room (var-030): on host while the room is no larger than one this
        # shape's whole was refused in. A refused selection has evicted every weight by then, so
        # the card is asked again only once something no eviction could move has gone.
        try:
            room = _STARTED[hsite] = catch.usable(request)
            on_host = room <= request.host_rooms.get(hsite, -1)
        except catch.varena.Error:  # the card's room can't be read now (another thread records)
            hostable = False
    if not on_host:
        try:
            return catch.cut(
                site, deepest, need, lambda level: run(window(level), False), pure=pure
            )
        except catch.Unplaced as exc:  # refused before a window ran (or the op is pure)
            if not hostable or request is None:
                raise
            unwind(exc)
            if measured is not None:  # judged on the package's estimate: this call only
                request.host_rooms[hsite] = max(room, request.host_rooms.get(hsite, -1))
            _hosted(request, hsite, name, _ESTIMATED if measured is None else _MEASURED)
    return catch.cut(
        f"{site}:host", deepest, windows_only, lambda level: run(window(level), True), pure=pure
    )


def whole(part: Tensor, shape: Sequence[int], key: str, host: bool, zeros: bool = False) -> Tensor:
    """The whole output a package's `split` assembles its windows in, `part` its first (zeroed
    with `zeros`: a blended whole's accumulator): pinned host memory with `host`; else the
    device's, or pinned host memory when the device refuses it beside the windows (a Runtime
    retry: no window runs again). A whole of windows the op returns on the host is plain host
    memory: there is no card rung to choose."""
    _WHOLES[key] = (tuple(int(n) for n in shape), part.dtype, part.device)
    if not part.is_cuda:
        return part.new_zeros(shape) if zeros else part.new_empty(shape)
    if host:
        return _pinned(shape, part.dtype, part.device, zeros)
    try:
        return part.new_zeros(shape) if zeros else part.new_empty(shape)
    except BaseException as exc:
        request = catch.CURRENT.get()
        if request is None or not part.is_cuda or not is_capacity(exc) or _capturing():
            raise
        room = catch.room(part.device.index or 0)  # beside the refused bytes, for the record
        try:
            out = _pinned(shape, part.dtype, part.device, zeros)
        except Exception as failed:
            exc.add_note(f"its whole output in host memory failed too: {failed}")
            raise exc from None
        unwind(exc)
        request.retries += 1
        hsite = f"split:{key}"  # the room its call began with: what the next call compares
        request.host_rooms[hsite] = max(_STARTED.get(hsite, -1), request.host_rooms.get(hsite, -1))
        _hosted(request, hsite, key, room)
        return out


class _Short(catch.Unplaced):
    """A split's measured whole refused on the card before any of its windows ran."""


def ahead(key: str, host: bool, zeros: bool) -> Tensor | None:
    """The whole output allocated before a split's first window, where an earlier call measured
    it: on host with `host`, else on the card. A card that refuses it raises `_Short`, before any
    window ran, and the split takes the host rung (a Runtime retry). None while it is unmeasured."""
    known = _WHOLES.get(key)
    request = catch.CURRENT.get()
    if known is None or request is None or _capturing():
        return None
    shape, dtype, device = known
    if device.type != "cuda":
        return (
            torch.zeros(shape, dtype=dtype, device=device)
            if zeros
            else torch.empty(shape, dtype=dtype, device=device)
        )
    if host:
        return _pinned(shape, dtype, device, zeros)
    try:
        if zeros:
            return torch.zeros(shape, dtype=dtype, device=device)
        return torch.empty(shape, dtype=dtype, device=device)
    except BaseException as exc:
        if not is_capacity(exc):
            raise
        room = catch.room(device.index or 0)
        unwind(exc)
        request.retries += 1
        raise _Short(
            f"{key}: no room for its whole ({_nbytes(shape, dtype)} bytes) beside {room}"
        ) from None


def landed(out: Tensor) -> Tensor:
    """`out`, a whole kept in host memory, on the card where it now fits: one copy at the link's
    speed, so what computes on it in place (`split(out=)`) runs on device memory (var-030). `out`
    itself where it is no host whole, does not fit, or is refused; a refused size is asked again
    only in a larger room, as a split's host decision is (`Request.host_rooms`)."""
    request = catch.CURRENT.get()
    if request is None or _capturing() or not pinned.held(out) or not out.is_contiguous():
        return out
    nbytes = out.numel() * out.element_size()
    site, room = f"land:{nbytes}", catch.usable(request)
    if nbytes > room or room <= request.host_rooms.get(site, -1):
        return out
    try:
        on_card = pinned.read(out)
    except BaseException as exc:
        if not is_capacity(exc):
            raise
        unwind(exc)
        request.retries += 1
        request.host_rooms[site] = room
        return out
    pinned.count("landed", 1)
    return on_card


def staged(window: Tensor) -> Tensor:
    """A window of a whole kept on host, on the card for an op that writes it in place (`write`
    copies it back); `window` itself where it is no host window, or the card refuses the copy
    (it is then written over the link, as before)."""
    if not pinned.held(window):
        return window
    try:
        return pinned.read(window)
    except BaseException as exc:
        request = catch.CURRENT.get()
        if request is None or not is_capacity(exc):
            raise
        unwind(exc)
        request.retries += 1
        return window


def moved(out: Tensor) -> Tensor:
    """A split's partly written whole on host: the windows it finished come with it."""
    if pinned.held(out) or not out.is_cuda:
        return out
    host = pinned.empty(out.shape, out.dtype, out.device)
    pinned.write(host, out)
    return host


def _pinned(shape: Sequence[int], dtype: torch.dtype, device: torch.device, zeros: bool) -> Tensor:
    out = pinned.empty(shape, dtype, device)
    return out.zero_() if zeros else out  # a kernel fills it, in the stream's order


def narrowed(whole: Tensor, dtype: torch.dtype) -> Tensor:
    """A blended whole cast to its windows' `dtype`, where it is: a host whole stays on host."""
    if not pinned.held(whole):
        return whole.to(dtype)
    out = pinned.empty(whole.shape, dtype, whole.device)
    out.copy_(whole)  # a kernel over both: no host round trip
    return out


def write(dst: Tensor, src: Tensor) -> None:
    """A split's window into its whole output (`pinned.write`)."""
    pinned.write(dst, src)


def read[X](window: X) -> X:
    """A split's input window on the device, where it is a window of a host whole."""
    return cast("X", pinned.read(window)) if isinstance(window, Tensor) else window


#: each split call shape's whole output, its shape, dtype and device, as its allocation measured it
_WHOLES: dict[str, tuple[tuple[int, ...], torch.dtype, torch.device]] = {}
#: the room (`catch.usable`) each hostable split shape's running call began with
_STARTED: dict[str, int] = {}
#: `Request.hosted`'s reason for a whole sent to host without a refusal in the run: selection
#: judged the package's estimate (for that call only) or the measured whole (for the request)
_ESTIMATED, _MEASURED = "estimated", "measured"


def _nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return math.prod(shape) * dtype.itemsize


def _on_card(key: str) -> bool:
    """Whether `key`'s measured whole is a card's: only those take the card's room."""
    known = _WHOLES.get(key)
    return known is not None and known[2].type == "cuda"


def _hosted(request: Request, site: str, name: str, why: str) -> None:
    request.hosted[site] = why
    named = (_ESTIMATED, _MEASURED)
    print(
        f"cozy-runtime: {name}: no room for its whole output beside its windows; it is kept in "
        f"host memory ({why if why in named else f'refused beside {why}'})",
        file=sys.stderr,
    )


def _capturing() -> bool:
    return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()


def windows(
    name: str, key: str, needs: Sequence[int]
) -> tuple[int, Callable[[], AbstractContextManager[None]]]:
    """A package's `windows` (`cozy_runtime.author.windows`): which of its layouts, fewest
    windows first, a loop keeps (`needs`: each one's whole bytes), chosen once before it, and the
    context its every window's work runs in (`catch.Fixed.piece`)."""
    chosen = catch.fixed(f"windows:{key}", len(needs) - 1, lambda n: (needs[n],))
    return chosen.level, chosen.piece
