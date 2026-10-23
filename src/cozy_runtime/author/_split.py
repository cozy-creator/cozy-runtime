"""`split`: the package says how an op cuts (`Cut`); the memory says how finely (varena's
`cuts.split`, installed with it): whole, else in the windows that fit, remembered per op."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Cut:
    """A dimension of input and output `fn` may be cut along. The output extent is `scale` times
    the input's; pieces start on multiples of `align` and neighbours share `overlap` elements,
    cross-faded in the output (none: pieces join exactly)."""

    dim: int
    scale: float = 1.0
    overlap: int = 0
    align: int = 1


class Pieces(Protocol):
    """The memory's side of `split`: `run(most)` in windows of at most `most` of `total`
    elements, the least `finest`, retried finer while out of memory."""

    def __call__(
        self, name: str, key: str, device: Any, total: int, finest: int, run: Callable[[int], Any]
    ) -> Any: ...


_RUNTIME: Pieces | None = None


def install(runtime: Pieces | None) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def split(fn: Callable[[Any], Any], x: Any, *cuts: Cut) -> Any:
    """`fn(x)`, whole or in the fewest pieces along `cuts` that fit the device; `fn` maps each
    window of `x` to that window of its result. Without a managing memory it runs whole."""
    runtime = _RUNTIME
    if runtime is None or not cuts:
        return fn(x)
    shape = tuple(int(n) for n in x.shape)
    return runtime(
        _name(fn),
        _key(fn, shape, cuts, x.dtype),
        x.device,
        math.prod(shape),
        _finest(shape, cuts),
        lambda most: _run(fn, x, cuts, _windows(shape, cuts, most)),
    )


type Window = tuple[tuple[int, int], ...]


def _key(
    fn: Callable[[Any], Any], shape: tuple[int, ...], cuts: tuple[Cut, ...], dtype: Any
) -> str:
    """Bytes per element depend on the op and its uncut extents."""
    cut = {c.dim % len(shape) for c in cuts}
    rest = tuple(n if i not in cut else -1 for i, n in enumerate(shape))
    return f"{_name(fn)}{rest}:{dtype}"


def _name(fn: Callable[[Any], Any]) -> str:
    return str(getattr(fn, "__qualname__", type(fn).__qualname__))


def _step(length: int, cut: Cut, count: int) -> int:
    """The aligned distance between neighbouring starts of about `count` spans of `length`."""
    step = max(-(-max(length - cut.overlap, 1) // count), cut.overlap, 1)
    return -(-step // cut.align) * cut.align


def _spans(length: int, cut: Cut, count: int) -> list[tuple[int, int]]:
    """About `count` aligned spans covering `length`, neighbours sharing `overlap`."""
    step = _step(length, cut, count)
    starts = range(0, max(length - cut.overlap, 1), step)
    return [(a, min(a + step + cut.overlap, length)) for a in starts]


def _longest(length: int, cut: Cut, count: int) -> int:
    """The longest of `count` spans: the first, as every later one is no longer."""
    return min(_step(length, cut, count) + cut.overlap, length)


def _finer(length: int, cut: Cut, count: int) -> int:
    """The fewest spans above `count` whose longest is shorter, 0 when none is. A span count's
    step only falls as the count grows, so it is the least count whose step reaches the
    largest aligned step below the current longest less the overlap: arithmetic, never a scan
    (a scan rebuilding each count's spans planned 136k tokens for minutes, cr-211)."""
    target = (_longest(length, cut, count) - cut.overlap - 1) // cut.align * cut.align
    if target < _step(length, cut, length):
        return 0
    more = max(count + 1, -(-max(length - cut.overlap, 1) // target))
    return more if more <= length else 0


def _windows(shape: tuple[int, ...], cuts: tuple[Cut, ...], most: float) -> list[Window]:
    """The fewest windows of at most `most` elements, longest axis cut first."""
    lengths = [shape[c.dim] for c in cuts]
    counts = [1] * len(cuts)
    longest = [_longest(n, c, 1) for n, c in zip(lengths, cuts, strict=True)]
    while _largest(shape, cuts, longest) > most:
        for i in sorted(range(len(cuts)), key=lambda i: -longest[i]):
            more = _finer(lengths[i], cuts[i], counts[i])
            if more:
                counts[i], longest[i] = more, _longest(lengths[i], cuts[i], more)
                break
        else:
            break
    windows: list[Window] = [()]
    for n, c, k in zip(lengths, cuts, counts, strict=True):
        windows = [(*w, s) for w in windows for s in _spans(n, c, k)]
    return windows


def _finest(shape: tuple[int, ...], cuts: tuple[Cut, ...]) -> int:
    """Elements in the largest window of the finest cut: each axis at as many spans as it has
    elements, which no greater count shortens."""
    return _largest(shape, cuts, [_longest(shape[c.dim], c, shape[c.dim]) for c in cuts])


def _largest(shape: tuple[int, ...], cuts: tuple[Cut, ...], longest: Sequence[int]) -> int:
    """Elements in the largest window, each cut axis at its longest span."""
    rest = math.prod(shape) // math.prod(shape[c.dim] for c in cuts)
    return rest * math.prod(longest)


def _run(fn: Callable[[Any], Any], x: Any, cuts: tuple[Cut, ...], windows: list[Window]) -> Any:
    if len(windows) == 1:
        return fn(x)
    ndim = x.dim()
    dims = [c.dim % ndim for c in cuts]
    blend = any(c.overlap for c in cuts)
    out: Any = None
    total: Any = None
    sizes = [math.ceil(x.shape[d] * c.scale - 1e-9) for d, c in zip(dims, cuts, strict=True)]
    for window in windows:
        index = [slice(None)] * ndim
        for d, (a, b) in zip(dims, window, strict=True):
            index[d] = slice(a, b)
        part = fn(x[tuple(index)])
        place = [slice(None)] * part.dim()
        for d, c, (a, _), size in zip(dims, cuts, window, sizes, strict=True):
            start = round(a * c.scale)
            place[d] = slice(start, min(start + int(part.shape[d]), size))
        if out is None:
            shape = list(part.shape)
            for d, size in zip(dims, sizes, strict=True):
                shape[d] = size
            wide = part if part.dtype.itemsize >= 4 else part.float()
            out = wide.new_zeros(shape) if blend else part.new_empty(shape)
            if blend:
                total = out.new_zeros([size if d in dims else 1 for d, size in enumerate(shape)])
        trim = tuple(slice(0, s.stop - s.start) if s.start is not None else s for s in place)
        if not blend:
            out[tuple(place)] = part[trim]
            continue
        weight = out.new_ones([1] * part.dim())
        for d, c, (a, b) in zip(dims, cuts, window, strict=True):
            length = place[d].stop - place[d].start
            ramp = _ramp(length, round(c.overlap * c.scale), a > 0, b < x.shape[d])
            shape = [1] * part.dim()
            shape[d] = length
            weight = weight * out.new_tensor(ramp).view(shape)
        out[tuple(place)] += part[trim].to(out.dtype) * weight
        total[tuple(s if i in dims else slice(None) for i, s in enumerate(place))] += weight
    if blend:
        return out.div_(total).to(part.dtype)
    return out


def _ramp(length: int, overlap: int, start: bool, end: bool) -> list[float]:
    """1 inside, a cosine ramp over a shared `overlap` at either end."""
    weight = [1.0] * length
    span = max(0, min(overlap, length - 1))
    if span:
        rise = [0.5 - 0.5 * math.cos(math.pi * i / max(span - 1, 1)) for i in range(span)]
        if start:
            weight[:span] = rise
        if end:
            weight[length - span :] = rise[::-1]
    return weight
