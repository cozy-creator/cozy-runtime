"""Row strips of a conv network's blocks: a block runs whole where it fits, else a strip of output
rows at a time, each strip reading its rows and a halo either side and keeping only its own.

A block is a module whose forward takes Diffusers' causal cache (`feat_cache`, `feat_idx`: Wan's,
Qwen-Image's and their kin's residual blocks, resamples and up blocks), or a conv subclass whose
forward wraps its conv (a causal pad). Its halo and scale come from its module tree: each conv's
reach along a row (`dilation * (kernel - 1) // 2`, summed: a bound for convs in sequence), each
upsampler's factor. A conv with a stride, or a norm whose statistics span rows, is no block.

The first call of a shape that would cut runs a probe: the block on a small crop of its input,
whole and in two strips, must give the same rows. A block that reads across rows (an attention)
fails it and runs whole from then on. A strip reads its rows of each cache entry and writes its
own rows of the entry's next value; the entries go to host memory once a block cuts (they are
saved inputs, read back a strip at a time). Nothing is committed until every strip is done, so a
refused strip runs again finer. PAN2's 5 s decode at a 6 GiB cap ran out in a causal pad
(`up_blocks.3.resnets.0.conv1`, 1.36 GB beside 3.2 GiB, pod-session 44426689158).
"""

from __future__ import annotations

import inspect
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from cozy_runtime.author._errors import is_capacity, unwind
from cozy_runtime.internal.execution import catch, pinned, sizing

#: a strip's working set over its input rows: a causal conv's input, its norm, the activation, the
#: cache clone, the cache `cat` and the pad, each the strip's rows (an estimate: a measured peak
#: replaces it)
_HOLDS = 6
_ROW_NORMS = (nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)


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
        if isinstance(sub, _ROW_NORMS):
            return None
        if isinstance(sub, nn.Conv2d | nn.Conv3d):
            if sub.stride[-1] != 1 or sub.stride[-2] != 1 or sub.padding == "valid":
                return None
            rows += sub.dilation[-2] * (sub.kernel_size[-2] - 1) // 2
            columns += sub.dilation[-1] * (sub.kernel_size[-1] - 1) // 2
        elif isinstance(sub, nn.Upsample):
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


#: a block's probe verdict by (site, shape without its rows)
_VERDICTS: dict[tuple[str, tuple[int, ...], torch.dtype], bool] = {}


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
            or torch.is_grad_enabled()
            or module.training
            or getattr(type(module), "_cozy_fused", None)
        ):
            return forward(*args, **kwargs)
        height = int(x.shape[-2])
        reach = max(block.rows, 1)
        if height < 2 * block.rows + 2:
            return forward(*args, **kwargs)
        key = (site, (*(int(n) for n in x.shape[:-2]), int(x.shape[-1])), x.dtype)
        if _VERDICTS.get(key) is False:
            return forward(*args, **kwargs)
        call = _Call(site, module, forward, block, bound, names[1])
        row = x.narrow(-2, 0, 1).numel() * x.element_size()
        out_row = row * block.scale * block.scale

        def need(level: int) -> tuple[int, ...]:
            if not level:
                return sizing.peak(height * out_row, (height * row * _HOLDS * block.scale**2,))
            # a cut level's whole output goes to host where the card refuses it (`_whole`)
            span = min(height, sizing.pieces(height, level) + 2 * block.rows)
            return (span * row * _HOLDS * block.scale**2,)

        def run(level: int) -> object:
            if level and key not in _VERDICTS and not catch._capturing():
                _VERDICTS[key] = call.probe()
            if not level or not _VERDICTS.get(key):
                return call.strips([(0, height)], host=False)
            n = sizing.pieces(height, level)
            return call.strips([(a, min(height, a + n)) for a in range(0, height, n)], host=True)

        finest = sizing.levels(-(-height // reach))
        try:
            return catch.cut(site, finest, need, run)
        except _Mismatch as exc:
            _VERDICTS[key] = False
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
        y = self.forward(*call.args[1:], **call.kwargs)
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

    def probe(self) -> bool:
        """Whether the block's rows are its strips': on a crop of its input, whole and in two
        strips, the same output and the same cache writes. Kernels may round a different shape
        differently, so a difference within a few units of the output's last place passes."""
        x, block, site = self.x, self.block, self.site
        height, width = int(x.shape[-2]), int(x.shape[-1])
        tall = min(height, 4 * block.rows + 4)
        wide = min(width, 4 * block.columns + 4)
        top, left = (height - tall) // 2, (width - wide) // 2
        crop = _Call(
            site,
            self.module,
            self.forward,
            block,
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
            scale = float(expected.abs().max()) or 1.0
            gap = float((expected.float() - got.float()).abs().max()) / scale
            worst = max(worst, gap)
        eps = torch.finfo(x.dtype).eps if x.dtype.is_floating_point else 0.0
        if x.dtype == torch.float32 and torch.backends.cudnn.allow_tf32:
            eps = 2.0**-10  # a float32 conv computes in TF32's 10 bits
        bound = 8 * eps
        verdict = worst <= bound
        how = "exact" if worst == 0 else f"within {worst:.2g} of its scale"
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
    host memory for the cut block `host`, else on the device."""
    if not isinstance(value, Tensor):
        return value
    if count == 1:
        return value
    f = _factor(int(value.shape[-2]), hi - lo)
    if not isinstance(whole, Tensor):
        shape = (*value.shape[:-2], height * f, value.shape[-1])
        whole = (
            _hosted(host, "its cache entries (cut by rows)", shape, value.dtype, value.device)
            if host is not None and value.is_cuda
            else torch.empty(shape, dtype=value.dtype, device=value.device)
        )
    pinned.write(whole.narrow(-2, a * f, (b - a) * f), value.narrow(-2, (a - lo) * f, (b - a) * f))
    return whole


def _whole(site: str, part: Tensor, shape: tuple[int, ...]) -> Tensor:
    """A cut block's whole output: on the card in `part`'s layout, else in pinned host memory the
    card addresses, where the card refuses it beside the strips (a Runtime retry; the next block
    reads its rows back a strip at a time)."""
    try:
        return torch.empty(shape, dtype=part.dtype, device=part.device, memory_format=_format(part))
    except BaseException as exc:
        request = catch.CURRENT.get()
        if request is None or not part.is_cuda or not is_capacity(exc) or catch._capturing():
            raise
        room = catch.room(part.device.index or 0)  # beside the refused bytes, for the record
        why = f"its whole output (refused beside {room})"
        try:
            out = _hosted(site, why, shape, part.dtype, part.device)
        except Exception as failed:
            exc.add_note(f"its whole output in host memory failed too: {failed}")
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
