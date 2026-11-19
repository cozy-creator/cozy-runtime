"""Row strips of a conv network's blocks: a block runs whole where it fits, else a strip of output
rows at a time, each strip reading its rows and a halo either side and keeping only its own.

A block is a module whose forward takes Diffusers' causal cache (`feat_cache`, `feat_idx`: Wan's,
Qwen-Image's and their kin's residual blocks, resamples and up blocks), or a conv subclass whose
forward wraps its conv (a causal pad). Its halo and scale come from its module tree: each conv's
reach along a row on its wider side (its kernel's span less the padding before it, or that
padding; summed, a bound for convs in sequence), each nearest upsampler's factor. A conv with a
stride or a wrapping pad, a norm whose statistics span rows, an interpolating upsampler, a
transposed conv, a pool or a pixel shuffle makes it no block.

The first call of a shape measures the block's peak on two whole crops of its input (`_Call.fit`):
each level's need, whole or a strip, comes from that, so a block whose whole fits runs whole. A
block inside a block that runs whole, its measured peak admitted, ran whole in that measure: it
starts whole and goes finer only where it is refused (`catch.cut`'s `covered`). A level that cuts
first probes the shape: on a crop, whole and in two strips must give the same rows, each element
within a few units of its own last place, so a halo short by one row fails. A block that reads
across rows (an attention) fails it and runs whole from then on. A strip reads
its rows of each cache entry and writes its own rows of the entry's next value; the entries go to
host memory once a block cuts (they are saved inputs, read back a strip at a time). Nothing is
committed until every strip is done, so a refused strip runs again finer. A block that keeps
tensors between calls, on itself or a module under it, as an attribute or in a list or dict (a
causal conv holding its last frames) runs whole: strips see only the caches handed to them.
PAN2's 5 s decode at a 6 GiB cap ran out in a causal pad (`up_blocks.3.resnets.0.conv1`, 1.36 GB
beside 3.2 GiB, pod-session 44426689158).
"""

from __future__ import annotations

import functools
import inspect
import math
import sys
from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol, cast

import tensorfs.varena.torch as vt
import torch
from torch import Tensor, nn

from cozy_runtime.author._errors import is_capacity, unwind
from cozy_runtime.internal import residency
from cozy_runtime.internal.execution import catch, pinned, sizing

_ROW_NORMS = (nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)
#: modules whose reach along rows the tree does not give: a block holding one is no block
_UNBOUNDED = (
    nn.Conv1d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.PixelShuffle,
    nn.PixelUnshuffle,
    nn.Fold,
    nn.Unfold,
)
#: what every module keeps in itself: its parameters, buffers, children, hooks and flags
_TORCH = frozenset(vars(nn.Module()))


class _Cache(Protocol):
    """Diffusers' causal cache as a block reads it: entries by index, set in place."""

    def __getitem__(self, index: int, /) -> object: ...
    def __setitem__(self, index: int, value: object, /) -> None: ...
    def __len__(self) -> int: ...


@dataclass(frozen=True, slots=True)
class Block:
    """A block's reach in input rows and columns either side, and its output rows per input row."""

    rows: int
    columns: int
    scale: int
    signature: inspect.Signature = field(compare=False)


def install(module: nn.Module) -> int:
    """Row strips on every block under `module` (`Block`). Returns how many."""
    count = 0
    for path, sub in module.named_modules():
        if "forward" in vars(sub) or getattr(type(sub), "_cozy_fused", None):
            continue
        block = _block(sub)
        if block is None:
            continue
        forward = catch.current(sub)
        sub.forward = catch.transparent(_strips(f"{path}:strips", sub, block, forward))
        count += 1
    return count


def _block(module: nn.Module) -> Block | None:
    kind = type(module)
    try:
        signature = inspect.signature(kind.forward)
    except (TypeError, ValueError):
        return None
    cached = "feat_cache" in signature.parameters
    padded = isinstance(module, nn.Conv2d | nn.Conv3d) and kind.forward not in (
        nn.Conv2d.forward,
        nn.Conv3d.forward,
    )
    if not (cached or padded):
        return None
    rows = columns = 0
    scale = 1
    for sub in module.modules():
        if isinstance(sub, _ROW_NORMS + _UNBOUNDED):
            return None
        if isinstance(sub, nn.Conv2d | nn.Conv3d):
            if (
                sub.stride[-1] != 1
                or sub.stride[-2] != 1
                or sub.padding == "valid"
                or sub.padding_mode == "circular"  # a strip's edge would wrap to its own far edge
            ):
                return None
            reach = _reach(sub)
            rows, columns = rows + reach[0], columns + reach[1]
        elif isinstance(sub, nn.Upsample):
            if sub.mode not in ("nearest", "nearest-exact"):
                return None  # an interpolating one reads its neighbours
            factor = sub.scale_factor
            first = factor[-2] if isinstance(factor, tuple) else factor
            if first is None or float(first) != int(first) or int(first) < 1:
                return None
            scale *= int(first)
        elif isinstance(
            sub, nn.ZeroPad2d | nn.ConstantPad2d | nn.ReflectionPad2d | nn.ReplicationPad2d
        ):
            return None  # an asymmetric pad ahead of a strided conv: the encoder's downsample
    return Block(rows, columns, scale, signature)


def _reach(conv: nn.Conv2d | nn.Conv3d) -> tuple[int, int]:
    """The input rows and columns an output reads past its own on its wider side: the kernel's
    span less the padding before it, or that padding, whichever is more (an even kernel or a
    one-sided pad reads more on one side). Padding added outside the conv counts as none, the
    widest it can be."""

    def side(axis: int) -> int:
        span = conv.dilation[axis] * (conv.kernel_size[axis] - 1)
        padding = conv.padding
        before = span // 2 if isinstance(padding, str) else int(padding[axis])
        return max(before, span - before)

    return side(-2), side(-1)


class _Rows:
    """A strip's view of a causal cache: each entry read at its rows (and columns, in a probe),
    on the strip's device; each entry written kept for the strip's caller."""

    __slots__ = ("columns", "device", "height", "hi", "lo", "parent", "width", "written")

    def __init__(
        self,
        parent: _Cache,
        height: int,
        lo: int,
        hi: int,
        device: torch.device,
        width: int = 0,
        columns: tuple[int, int] | None = None,
    ) -> None:
        self.parent, self.height, self.lo, self.hi, self.device = parent, height, lo, hi, device
        self.width, self.columns = width, columns
        self.written: dict[int, object] = {}

    def __getitem__(self, index: int) -> object:
        entry = self.parent[index]
        if not isinstance(entry, Tensor):
            return entry
        f = _factor(int(entry.shape[-2]), self.height)
        entry = entry.narrow(-2, self.lo * f, (self.hi - self.lo) * f)
        if self.columns is not None:
            g = _factor(int(entry.shape[-1]), self.width)
            a, b = self.columns
            entry = entry.narrow(-1, a * g, (b - a) * g)
        if pinned.held(entry):
            return pinned.read(entry)
        return entry if entry.device == self.device else entry.to(self.device)

    def __setitem__(self, index: int, value: object) -> None:
        self.written[index] = value

    def __len__(self) -> int:
        return len(self.parent)


class _Mismatch(Exception):
    """The block is not what its tree says: it runs whole."""


def _factor(size: int, height: int) -> int:
    if height <= 0 or size % height:
        raise _Mismatch(f"an entry of {size} rows over {height}")
    return size // height


#: a block's call by its shape without its rows
type _Key = tuple[tuple[int, ...], torch.dtype]
#: a block runs whole, its measured peak admitted: each block inside it ran whole in that measure
_MEASURED: ContextVar[bool] = ContextVar("strips_measured", default=False)


@dataclass(frozen=True, slots=True)
class _Fit:
    """A block's peak over a call: fixed bytes (a conv's workspace), and bytes per cell of its
    rows and columns, each with its halo either side."""

    fixed: int
    cell: float
    rows: int
    columns: int

    def of(self, rows: int, width: int) -> int:
        return self.fixed + int(self.cell * (rows + 2 * self.rows) * (width + 2 * self.columns))


@dataclass(slots=True)
class _Seen:
    """What a block's calls found, kept on the block itself, so no other model's block of the same
    path and shape reads it: each shape's probe verdict and measured peak (None where it could not
    be measured), and whether the block keeps state between calls."""

    verdicts: dict[_Key, bool] = field(default_factory=dict)
    fits: dict[_Key, _Fit | None] = field(default_factory=dict)
    stateful: bool = False


def _seen(module: nn.Module) -> _Seen:
    seen = vars(module).get("_cozy_strips")
    if not isinstance(seen, _Seen):
        seen = vars(module)["_cozy_strips"] = _Seen()
    return seen


def _state(module: nn.Module) -> dict[str, Tensor]:
    """The tensors `module` keeps as its own attributes: not its parameters or buffers, nor the
    Runtime's `_cozy_*` marks."""
    return {
        k: v for k, v in vars(module).items() if isinstance(v, Tensor) and not k.startswith("_cozy")
    }


type _Held = dict[tuple[int, str], tuple[nn.Module, object, object]]


def _held(module: nn.Module) -> _Held:
    """What `module` and every module under it keep as plain attributes that are or hold tensors
    (not parameters, buffers or children, nor the Runtime's `_cozy_*` marks): each one's value
    and, for a list or dict, a copy of its items now."""
    out: _Held = {}
    for m in module.modules():
        for k, v in vars(m).items():
            if k in _TORCH or k.startswith("_cozy"):
                continue
            if isinstance(v, list | tuple):
                if any(isinstance(item, Tensor) for item in v):
                    out[(id(m), k)] = (m, v, list(v))
            elif isinstance(v, dict):
                if any(isinstance(item, Tensor) for item in v.values()):
                    out[(id(m), k)] = (m, v, dict(v))
            elif isinstance(v, Tensor):
                out[(id(m), k)] = (m, v, None)
    return out


def _same(a: object, b: object) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(a[k] is b[k] for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(x is y for x, y in zip(a, b, strict=True))
    return a is b


def _kept(before: _Held, module: nn.Module) -> list[str]:
    """The attributes a call of `module` set, rebound or changed in place under it since
    `before` (its own or any module's below it), each put back as it was."""
    after, changed = _held(module), []
    for key in before.keys() | after.keys():
        then, now = before.get(key), after.get(key)
        if then is not None and now is not None and then[1] is now[1] and _same(then[2], now[2]):
            continue
        held = then if then is not None else now
        assert held is not None
        m, name = held[0], key[1]
        changed.append(name)
        if then is None:
            del vars(m)[name]
            continue
        vars(m)[name] = then[1]
        if isinstance(then[1], list) and isinstance(then[2], list):
            then[1][:] = then[2]
        elif isinstance(then[1], dict) and isinstance(then[2], dict):
            then[1].clear()
            then[1].update(then[2])
    return changed


def _keeps(module: nn.Module, names: Iterable[str]) -> str:
    _seen(module).stateful = True
    return f"it keeps {', '.join(sorted(set(names)))} between calls"


def _records(module: nn.Module, values: Iterable[object]) -> bool:
    """Whether autograd records this call: grad mode on, and an input or a parameter needs grad."""
    return torch.is_grad_enabled() and (
        any(isinstance(v, Tensor) and v.requires_grad for v in values)
        or any(p.requires_grad for p in module.parameters())
    )


def _peak[T](run: Callable[[], T], device: torch.device) -> tuple[T, int | None]:
    """`run()`, and the working bytes it held at its peak above what was live as it began, measured
    as nested in whatever measures around it (`residency.peak_within`)."""
    request = catch.CURRENT.get()
    if request is None or not request.memory.cuda or device.type != "cuda":
        return run(), None
    index = device.index or 0
    before = int(vt.stats(index)["allocated_bytes"])
    out, peak = residency.peak_within(index, run)
    return out, max(peak - before, 0)


def _strips(site: str, module: nn.Module, block: Block, forward: catch.Forward) -> catch.Forward:
    def cut(*args: object, **kwargs: object) -> object:
        try:
            bound = block.signature.bind_partial(module, *args, **kwargs)
        except TypeError:
            return forward(*args, **kwargs)
        names = list(bound.arguments)
        x = bound.arguments.get(names[1]) if len(names) > 1 else None
        cached = bound.arguments.get("feat_cache") is not None
        if (
            not isinstance(x, Tensor)
            or (cached and not isinstance(bound.arguments.get("feat_idx"), list))
            or x.dim() < 4
            or _records(module, bound.arguments.values())
            or module.training
            or getattr(type(module), "_cozy_fused", None)
            or _seen(module).stateful
        ):
            return forward(*args, **kwargs)
        if held := _state(module):
            print(f"cozy-runtime: {site} runs whole: {_keeps(module, held)}", file=sys.stderr)
            return forward(*args, **kwargs)
        height = int(x.shape[-2])
        reach = max(block.rows, 1)
        if height < 2 * block.rows + 2:
            return forward(*args, **kwargs)
        seen = _seen(module)
        key = ((*(int(n) for n in x.shape[:-2]), int(x.shape[-1])), x.dtype)
        if seen.verdicts.get(key) is False:
            return forward(*args, **kwargs)
        call = _Call(site, module, forward, block, bound, names[1])
        width = int(x.shape[-1])
        if key not in seen.fits and not catch._capturing():
            try:
                seen.fits[key] = call.fit()
            except _Mismatch as exc:
                seen.verdicts[key] = False
                print(f"cozy-runtime: {site} runs whole: {exc}", file=sys.stderr)
                return forward(*args, **kwargs)
        fit = seen.fits.get(key)

        def need(level: int) -> tuple[int, ...]:
            if fit is None:
                return (0,)  # not measured (no native memory, or inside a capture): whole
            # a cut level's whole output goes to host where the card refuses it (`_whole`)
            rows = min(height, sizing.pieces(height, level) + 2 * block.rows) if level else height
            return (fit.of(rows, width),)

        def run(level: int) -> object:
            if level and key not in seen.verdicts and not catch._capturing():
                seen.verdicts[key] = call.probe()
            measured = _MEASURED.set(fit is not None and not level)
            try:
                if not level or not seen.verdicts.get(key):
                    return call.strips([(0, height)], host=False)
                n = sizing.pieces(height, level)
                spans = [(a, min(height, a + n)) for a in range(0, height, n)]
                return call.strips(spans, host=True)
            finally:
                _MEASURED.reset(measured)

        finest = sizing.levels(-(-height // reach))
        try:
            return catch.cut(site, finest, need, run, covered=_MEASURED.get())
        except _Mismatch as exc:
            seen.verdicts[key] = False
            print(f"cozy-runtime: {site} runs whole: {exc}", file=sys.stderr)
            return forward(*args, **kwargs)

    return cut


class _Call:
    """One call of a block: its input, its other arguments, its cache and the cache's index."""

    def __init__(
        self,
        site: str,
        module: nn.Module,
        forward: catch.Forward,
        block: Block,
        bound: inspect.BoundArguments,
        first: str,
    ) -> None:
        self.site, self.module, self.forward = site, module, forward
        self.block, self.first = block, first
        self.bound: dict[str, object] = dict(bound.arguments)
        self.signature = bound.signature
        self.x = cast("Tensor", self.bound[first])
        cache = self.bound.get("feat_cache")
        self.cache = cast("_Cache", cache) if cache is not None else None
        index = self.bound.get("feat_idx")
        self.index = cast("list[int]", index) if self.cache is not None else None

    def _run(
        self,
        lo: int,
        hi: int,
        columns: tuple[int, int] | None = None,
        cache: _Cache | None = None,
    ) -> tuple[Tensor, _Rows | None, int]:
        """The block on rows [lo, hi) (and `columns`) of its input: its output, what it wrote to
        its cache, and the cache index it ended on."""
        x = self.x
        height, width = int(x.shape[-2]), int(x.shape[-1])

        def sliced(t: Tensor) -> Tensor:
            t = t.narrow(-2, lo, hi - lo)
            t = t if columns is None else t.narrow(-1, columns[0], columns[1] - columns[0])
            return pinned.read(t)  # a window of a host whole comes to the device

        arguments: dict[str, object] = {}
        for name, value in self.bound.items():
            if (
                isinstance(value, Tensor)
                and value.dim() == x.dim()
                and value.shape[-2:] == x.shape[-2:]
            ):
                value = sliced(value)
            arguments[name] = value
        rows: _Rows | None = None
        start = self.index[0] if self.index is not None else 0
        index = [start]
        if self.cache is not None:
            rows = _Rows(
                cache if cache is not None else self.cache, height, lo, hi, x.device, width, columns
            )
            arguments["feat_cache"] = rows
            arguments["feat_idx"] = index
        call = inspect.BoundArguments(self.signature, OrderedDict(arguments))
        before = _held(self.module)
        y = self.forward(*call.args[1:], **call.kwargs)
        if kept := _kept(before, self.module):  # undone: the call runs again whole as it began
            raise _Mismatch(_keeps(self.module, kept))
        if not isinstance(y, Tensor):
            raise _Mismatch(f"returned {type(y).__name__}")
        span = (hi - lo) * self.block.scale
        if int(y.shape[-2]) != span:
            raise _Mismatch(f"{hi - lo} rows made {int(y.shape[-2])}, not {span}")
        end = index[0] if rows is not None else start
        return y, rows, end

    def strips(self, spans: list[tuple[int, int]], *, host: bool) -> Tensor:
        """The block over `spans` of output rows (input rows, before its scale), each with its
        halo; its cache's next entries assembled from each strip's own rows, on host when `host`.
        Committed once every strip is done."""
        x, scale, reach = self.x, self.block.scale, self.block.rows
        height = int(x.shape[-2])
        out: Tensor | None = None
        made: dict[int, object] = {}
        end: int | None = None
        on_host = self.site if host else None
        for a, b in spans:
            catch.checkpoint()
            lo, hi = max(0, a - reach), min(height, b + reach)
            y, rows, last = self._run(lo, hi)
            if end is not None and last != end:
                raise _Mismatch(f"strips ended on cache entries {end} and {last}")
            end = last
            kept = y.narrow(-2, (a - lo) * scale, (b - a) * scale)
            if len(spans) == 1:
                out = y
            else:
                if out is None:
                    out = _whole(self.site, y, (*y.shape[:-2], height * scale, y.shape[-1]))
                pinned.write(out.narrow(-2, a * scale, (b - a) * scale), kept)
            del y, kept
            if rows is not None:
                for index, value in rows.written.items():
                    made[index] = _assemble(
                        made.get(index), value, height, a, b, lo, hi, on_host, len(spans)
                    )
        assert out is not None
        if self.cache is not None and self.index is not None and end is not None:
            for index in sorted(made):
                self.cache[index] = made[index]
            self.index[0] = end
        return out

    def _crop(self, tall: int, wide: int) -> tuple[_Call, _Rows | None]:
        """The block's call on a centred crop of `tall` rows and `wide` columns of its input, and
        the crop's view of its cache."""
        x = self.x
        height, width = int(x.shape[-2]), int(x.shape[-1])
        top, left = (height - tall) // 2, (width - wide) // 2
        crop = _Call(
            self.site,
            self.module,
            self.forward,
            self.block,
            self.signature.bind_partial(**self.bound),
            self.first,
        )
        for name, value in self.bound.items():
            if (
                isinstance(value, Tensor)
                and value.dim() == x.dim()
                and value.shape[-2:] == x.shape[-2:]
            ):
                crop.bound[name] = value.narrow(-2, top, tall).narrow(-1, left, wide)
        crop.x = cast("Tensor", crop.bound[self.first])
        cache = (
            None
            if self.cache is None
            else _Rows(self.cache, height, top, top + tall, x.device, width, (left, left + wide))
        )
        return crop, cache

    def _crop_size(self) -> tuple[int, int]:
        """The probe's crop: rows and columns, four halos and four more each."""
        x, block = self.x, self.block
        tall = min(int(x.shape[-2]), 4 * block.rows + 4)
        return tall, min(int(x.shape[-1]), 4 * block.columns + 4)

    def fit(self) -> _Fit | None:
        """The block's peak over its call, from whole crops at two widths: fixed bytes and bytes a
        cell. None where it cannot be measured, or the card refused a crop."""
        block, width = self.block, int(self.x.shape[-1])
        tall, wide = self._crop_size()
        broad = min(width, 2 * wide + 2 * block.columns)
        peaks: list[tuple[int, int]] = []
        for w in dict.fromkeys((wide, broad)):
            crop, cache = self._crop(tall, w)
            try:
                _, peak = _peak(functools.partial(crop._run, 0, tall, cache=cache), self.x.device)
            except BaseException as exc:
                if not is_capacity(exc):
                    raise
                unwind(exc)
                return None
            if peak is None:
                return None
            peaks.append(((tall + 2 * block.rows) * (w + 2 * block.columns), peak))
        (c1, p1), (c2, p2) = peaks[0], peaks[-1]
        if c2 > c1 and p2 > p1:
            cell = (p2 - p1) / (c2 - c1)
            fixed = max(int(p1 - cell * c1), 0)
        else:
            cell, fixed = p1 / c1, 0
        return _Fit(fixed, cell, block.rows, block.columns)

    def probe(self) -> bool:
        """Whether the block's rows are its strips': on a crop of its input, whole and in two
        strips, the same output and the same cache writes, each element within a few units of
        its own last place (kernels round a different shape differently), its size counted with
        the output's mean so an element near zero is not judged on its rounding alone."""
        x, block, site = self.x, self.block, self.site
        tall, wide = self._crop_size()
        crop, cache = self._crop(tall, wide)
        try:
            whole, whole_rows, _ = crop._run(0, tall, cache=cache)
            half = tall // 2
            parts: list[Tensor] = []
            writes: dict[int, object] = {}
            for a, b in ((0, half), (half, tall)):
                lo, hi = max(0, a - block.rows), min(tall, b + block.rows)
                y, rows, _ = crop._run(lo, hi, cache=cache)
                parts.append(y.narrow(-2, (a - lo) * block.scale, (b - a) * block.scale))
                if rows is not None:
                    for index, value in rows.written.items():
                        writes[index] = _assemble(
                            writes.get(index), value, tall, a, b, lo, hi, None, 2
                        )
        except _Mismatch as exc:
            print(f"cozy-runtime: {site} runs whole: {exc}", file=sys.stderr)
            return False
        stripped = torch.cat(parts, dim=-2)
        pairs = [(whole, stripped)]
        if whole_rows is not None:
            if set(whole_rows.written) != set(writes):
                print(
                    f"cozy-runtime: {site} runs whole: strips write other cache entries",
                    file=sys.stderr,
                )
                return False
            for i, v in whole_rows.written.items():
                w = writes[i]
                if isinstance(v, Tensor) and isinstance(w, Tensor):
                    pairs.append((v, w))
                elif isinstance(v, Tensor) or isinstance(w, Tensor) or v != w:
                    print(
                        f"cozy-runtime: {site} runs whole: strips write another entry",
                        file=sys.stderr,
                    )
                    return False
        worst = 0.0
        for expected, got in pairs:
            if expected.shape != got.shape:
                print(
                    f"cozy-runtime: {site} runs whole: strips make another shape", file=sys.stderr
                )
                return False
            if torch.equal(expected, got):
                continue
            # each element against its own size and the output's typical one: a row read without
            # its neighbour (a halo short by one) is off by that neighbour's share of it
            size = expected.float().abs()
            gap = (expected.float() - got.float()).abs() / (size + size.mean())
            worst = max(worst, float(gap.max().nan_to_num(float("inf"))))
        eps = torch.finfo(x.dtype).eps if x.dtype.is_floating_point else 0.0
        if x.dtype == torch.float32 and torch.backends.cudnn.allow_tf32:
            eps = 2.0**-10  # a float32 conv computes in TF32's 10 bits
        bound = 8 * eps
        verdict = worst <= bound
        how = "exact" if worst == 0 else f"within {worst:.2g} of each element's size"
        print(
            f"cozy-runtime: {site} {'cuts by rows' if verdict else 'runs whole'}: strips of a "
            f"{tall}x{wide} crop are {how} (halo {block.rows}, scale {block.scale})",
            file=sys.stderr,
        )
        return verdict


def _assemble(
    whole: object,
    value: object,
    height: int,
    a: int,
    b: int,
    lo: int,
    hi: int,
    host: str | None,
    count: int,
) -> object:
    """A cache entry's next value with a strip's own rows [a, b) of `value` (rows [lo, hi)): in
    host memory for the cut block `host` (selection cut it for want of room, and its entries are
    saved inputs, read back a strip at a time), else on the device, as inside a capture."""
    if not isinstance(value, Tensor):
        return value
    if count == 1:
        return value
    f = _factor(int(value.shape[-2]), hi - lo)
    if not isinstance(whole, Tensor):
        shape = (*value.shape[:-2], height * f, value.shape[-1])
        whole = (
            _hosted(host, "its cache entries (cut by rows)", shape, value.dtype, value.device)
            if host is not None and value.is_cuda and not catch._capturing()
            else torch.empty(shape, dtype=value.dtype, device=value.device)
        )
    pinned.write(whole.narrow(-2, a * f, (b - a) * f), value.narrow(-2, (a - lo) * f, (b - a) * f))
    return whole


def _whole(site: str, part: Tensor, shape: tuple[int, ...]) -> Tensor:
    """A cut block's whole output, in `part`'s layout (the next block reads its rows back a strip
    at a time)."""
    return _placed(site, "its whole output", shape, part, _format(part))


def _placed(
    site: str, what: str, shape: tuple[int, ...], like: Tensor, layout: torch.memory_format
) -> Tensor:
    """`what` of a cut block: on the card where its room holds it (the card less what is live and
    the weights in use, as a nested cut reads it) and the allocation is not refused, else in
    pinned host memory the card addresses, named in the request (a refusal answered so counts
    as a Runtime retry). Inside a capture it stays on the card, as the capture's own do."""
    request = catch.CURRENT.get()
    hostable = request is not None and like.is_cuda and not catch._capturing()
    device = like.device.index or 0
    if hostable:
        room = vt.room(device)
        usable = int(room["capacity"]) - int(room["working_live"]) - int(room["pinned"])
        if math.prod(shape) * like.element_size() > usable:
            why = f"{what} (more than the room beside {catch.room(device)})"
            return _hosted(site, why, shape, like.dtype, like.device)
    try:
        return torch.empty(shape, dtype=like.dtype, device=like.device, memory_format=layout)
    except BaseException as exc:
        if not hostable or not is_capacity(exc):
            raise
        assert request is not None
        why = f"{what} (refused beside {catch.room(device)})"
        try:
            out = _hosted(site, why, shape, like.dtype, like.device)
        except Exception as failed:
            exc.add_note(f"{what} in host memory failed too: {failed}")
            raise exc from None
        unwind(exc)
        request.retries += 1  # the refusal is answered: no OOM a package swallowed (`contract`)
        return out


def _hosted(
    site: str, why: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Pinned host memory for `site`, named in its request (`Request.hosted`), which frees the
    pinned buffers at its end."""
    out = pinned.empty(shape, dtype, device)
    request = catch.CURRENT.get()
    if request is not None and site not in request.hosted:
        request.hosted[site] = why
        print(f"cozy-runtime: {site} keeps {why} in host memory", file=sys.stderr)
    return out


def _format(t: Tensor) -> torch.memory_format:
    if (
        t.dim() == 5
        and t.is_contiguous(memory_format=torch.channels_last_3d)
        and not t.is_contiguous()
    ):
        return torch.channels_last_3d
    if (
        t.dim() == 4
        and t.is_contiguous(memory_format=torch.channels_last)
        and not t.is_contiguous()
    ):
        return torch.channels_last
    return torch.contiguous_format
