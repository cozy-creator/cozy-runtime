"""`split`: the package says how an op cuts (`Cut`); the memory says how finely (the execution
layer's `split`, installed with it): whole, else in the windows native selection gives. A
layout that changes the result (a blended cut) comes from the shape alone (`most`)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
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
    """The memory's side of `split`: `split` runs `run(most)` in windows of at most `most` of
    `total` elements, the least `finest`, once; `piece` marks each window's start."""

    def split(
        self,
        name: str,
        key: str,
        total: int,
        finest: int,
        out_bytes: int,
        item_bytes: float,
        holds: int,
        run: Callable[[int], Any],
        pure: bool = False,
    ) -> Any: ...

    def piece(self) -> None: ...

    def windows(
        self, name: str, key: str, needs: Sequence[int]
    ) -> tuple[int, Callable[[], AbstractContextManager[None]]]: ...


_RUNTIME: Pieces | None = None


def install(runtime: Pieces | None) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def split(
    fn: Callable[..., Any],
    x: Any,
    *cuts: Cut,
    most: int | None = None,
    out: Any = None,
    holds: int = 5,
    host: bool = False,
) -> Any:
    """`fn(x)`, whole or in the fewest pieces along `cuts` that fit the device; `fn` maps each
    window of `x` to that window of its result. Without a managing memory it runs whole.

    With `out`, `fn(window, out_window)` writes each window's result into the same window of
    `out`, which `split` returns: nothing whole is allocated beside it, so an op may write its
    input's rows in place (H3's block tail into its attention output). Its cuts keep their
    extent: no scale and no overlap.

    With `most`, the windows are the fewest of at most `most` elements, on every device and
    request, managed or not: the layout for an `fn` whose result depends on it, as a blended
    cut's (an `overlap`) does. Without it such a layout follows free memory.

    With `host`, `fn` returns its windows in host memory and the result is assembled there: the
    whole output is no device need.

    `holds` is `fn`'s device peak beside the whole output, in windows of its output: the estimate
    the device's choice starts from before it has measured a pass."""
    runtime = _RUNTIME
    shape = tuple(int(n) for n in x.shape)
    # Writing windows in place (`out`) can consume its input, so that never runs twice.
    pure = out is None and getattr(fn, "PURE_WINDOWS", False)
    if out is not None and any(c.scale != 1 or c.overlap for c in cuts):
        raise ValueError("split(out=) writes windows in place: its cuts take no scale or overlap")
    whole = 0 if out is not None or host else _bytes(shape, cuts, x)
    if most is not None and cuts:
        windows = _windows(shape, cuts, most)
        if runtime is None:
            return _run(fn, x, cuts, windows, lambda: None, out)
        # The largest window these cuts made, not `most`: a small input cuts smaller.
        largest = max(math.prod(b - a for a, b in w) for w in windows) * (
            math.prod(shape) // math.prod(shape[c.dim] for c in cuts)
        )
        return runtime.split(
            _name(fn),
            _key(fn, shape, cuts, x.dtype),
            largest,
            largest,
            whole,
            _item_bytes(cuts, x),
            holds,
            lambda _: _run(fn, x, cuts, windows, runtime.piece, out),
            pure,
        )
    if runtime is None or not cuts:
        if out is None:
            return fn(x)
        fn(x, out)
        return out
    return runtime.split(
        _name(fn),
        _key(fn, shape, cuts, x.dtype),
        math.prod(shape),
        _finest(shape, cuts),
        whole,
        _item_bytes(cuts, x),
        holds,
        lambda most: _run(fn, x, cuts, _windows(shape, cuts, most), runtime.piece, out),
        pure,
    )


@dataclass(frozen=True)
class Windows:
    """A window layout a loop keeps throughout (`windows`): `spans`, the `(start, stop)` of each
    window along its cut. Run each window's work inside `piece()`, so the layout's memory is
    measured and the next call chooses from what it held."""

    spans: tuple[tuple[int, int], ...]
    _piece: Callable[[], AbstractContextManager[None]] = nullcontext

    def piece(self) -> AbstractContextManager[None]:
        return self._piece()

    def __len__(self) -> int:
        return len(self.spans)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self.spans)


def windows(
    name: str,
    shape: Sequence[int],
    cut: Cut,
    *,
    per_element_bytes: float,
    most: int | None = None,
    equal: bool = False,
) -> Windows:
    """The fewest windows along `cut` of a tensor of `shape` that fit the device, chosen once
    for a whole loop, as `split` chooses a call's: a layout the loop's tables and labels are
    built for before it starts. `per_element_bytes` is what a window's work holds at its peak, in
    bytes per element of the window (every axis whole but the cut's); the device's choice starts
    from it and then from what earlier calls measured. Starts fall on `cut.align`, neighbours share
    `cut.overlap`, and with `equal` every window has one length (the fewest equal windows at
    least as many as chosen). With `most`, the fewest of at most `most` elements, from the shape
    alone, on every device. Without a managing memory, one window unless `most` says more."""
    shape = tuple(int(n) for n in shape)
    length = shape[cut.dim]
    rest = math.prod(shape) // max(length, 1)
    layouts: list[list[tuple[int, int]]] = []
    count = 1  # level L asks for about 2^L windows, as `split` does
    while count <= max(length, 1):
        spans = _equal(length, cut, count) if equal else _spans(length, cut, count)
        if spans is None or (layouts and _span(spans) >= _span(layouts[-1])):
            break  # no shorter window: finer only adds launches
        layouts.append(spans)
        count = max(2 * count, len(spans) + 1)
    if most is not None:
        return Windows(tuple(_fewest(length, cut, equal, most // max(rest, 1))))
    runtime = _RUNTIME
    if runtime is None or len(layouts) == 1:
        return Windows(tuple(layouts[0]))
    needs = [int(_span(w) * rest * per_element_bytes) for w in layouts]
    key = f"{name}{shape}:{cut}:{per_element_bytes}:{equal}"
    at, piece = runtime.windows(name, key, needs)
    return Windows(tuple(layouts[at]), piece)


def _fewest(length: int, cut: Cut, equal: bool, longest: int) -> list[tuple[int, int]]:
    """The fewest windows no longer than `longest`, else the finest there are."""
    count, spans = 1, [(0, length)]
    while _span(spans) > longest:
        more = (len(spans) + 1) if equal else _finer(length, cut, count)
        if not more:
            break
        found = _equal(length, cut, more) if equal else _spans(length, cut, more)
        if found is None:
            break
        count, spans = more, found
    return spans


def _span(spans: Sequence[tuple[int, int]]) -> int:
    """The longest window's length."""
    return max(b - a for a, b in spans)


def _equal(length: int, cut: Cut, count: int) -> list[tuple[int, int]] | None:
    """The fewest windows, at least `count`, all of one length, covering `length` with starts
    spread evenly on `align` and neighbours sharing at least `overlap`; None when there are none.
    One length `L` keeps `L ≡ length (mod align)`, so the last window ends at `length`."""
    if count <= 1 or length <= cut.overlap:
        return [(0, length)]
    for k in range(count, length + 1):
        need = -(-(length + (k - 1) * cut.overlap) // k)
        size = need + (length - need) % cut.align
        if size >= length:
            continue
        steps = (length - size) // cut.align
        starts = [cut.align * round(i * steps / (k - 1)) for i in range(k)]
        if all(b - a <= size - cut.overlap for a, b in itertools.pairwise(starts)):
            return [(a, a + size) for a in starts]
    return None


type Window = tuple[tuple[int, int], ...]


def _key(
    fn: Callable[[Any], Any], shape: tuple[int, ...], cuts: tuple[Cut, ...], dtype: Any
) -> str:
    """Bytes per element depend on the op and its uncut extents."""
    cut = {c.dim % len(shape) for c in cuts}
    rest = tuple(n if i not in cut else -1 for i, n in enumerate(shape))
    return f"{getattr(fn, '__module__', '')}.{_name(fn)}{rest}:{dtype}"


def _item_bytes(cuts: tuple[Cut, ...], x: Any) -> float:
    """Output bytes per input element."""
    return float(math.prod(c.scale for c in cuts) * x.element_size())


def _bytes(shape: tuple[int, ...], cuts: tuple[Cut, ...], x: Any) -> int:
    """The whole output's bytes, which `split` allocates beside its windows."""
    return int(math.prod(shape) * _item_bytes(cuts, x))


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


def _run(
    fn: Callable[..., Any],
    x: Any,
    cuts: tuple[Cut, ...],
    windows: list[Window],
    piece: Callable[[], None],
    into: Any = None,
) -> Any:
    ndim = x.dim()
    dims = [c.dim % ndim for c in cuts]
    if into is not None:  # each window writes its own rows of `into` (`split(out=)`)
        for window in windows:
            index = [slice(None)] * ndim
            for d, (a, b) in zip(dims, window, strict=True):
                index[d] = slice(a, b)
            piece()
            fn(x[tuple(index)], into[tuple(index)])
        return into
    if len(windows) == 1:
        return fn(x)
    blend = any(c.overlap for c in cuts)
    out: Any = None
    total: Any = None
    sizes = [math.ceil(x.shape[d] * c.scale - 1e-9) for d, c in zip(dims, cuts, strict=True)]
    for window in windows:
        index = [slice(None)] * ndim
        for d, (a, b) in zip(dims, window, strict=True):
            index[d] = slice(a, b)
        piece()
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
