"""`split`: the package says how an op cuts (`Cut`); the memory says how finely (the execution
layer's `split`, installed with it): whole, else in the windows native selection gives. A
layout that changes the result (a blended cut) comes from the shape alone (`most`)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast, overload

if TYPE_CHECKING:
    import torch
    from torch import Tensor


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
    """The memory's side of `split`: `split` runs `run(most, host)` in windows of at most `most`
    of `total` elements, the least `finest`, once, its whole output in host memory with `host`
    (only where `hostable`); `piece` marks each window's start. `whole` is the output windows are
    assembled in (`ahead`: allocated before the first window, where measured), `write` puts a
    window there, `read` brings an input's window to the device and `moved` a partly written
    whole to host."""

    def split[T](
        self,
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
    ) -> T: ...

    def piece(self) -> None: ...

    def whole(
        self, part: Tensor, shape: Sequence[int], key: str, host: bool, zeros: bool = False
    ) -> Tensor: ...

    def ahead(self, key: str, host: bool, zeros: bool) -> Tensor | None: ...

    def moved(self, out: Tensor) -> Tensor: ...

    def narrowed(self, whole: Tensor, dtype: torch.dtype) -> Tensor: ...

    def write(self, dst: Tensor, src: Tensor) -> None: ...

    def read[X](self, window: X) -> X: ...

    def windows(
        self, name: str, key: str, needs: Sequence[int]
    ) -> tuple[int, Callable[[], AbstractContextManager[None]]]: ...


class Sliceable[W](Protocol):
    """What `split` cuts: a tensor, or an object standing for one whose windows are `W`
    (`x[index]`). Run whole, `fn` is handed `x` itself, which stands for its whole window."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def element_size(self) -> int: ...

    def dim(self) -> int: ...

    def __getitem__(self, index: tuple[slice[int | None], ...], /) -> W: ...


_RUNTIME: Pieces | None = None


def install(runtime: Pieces | None) -> None:
    global _RUNTIME
    _RUNTIME = runtime


@overload
def split[W](
    fn: Callable[[W], Tensor],
    x: Sliceable[W],
    *cuts: Cut,
    most: int | None = None,
    out: None = None,
    holds: int = 5,
    host: bool = False,
) -> Tensor: ...


@overload
def split[W](
    fn: Callable[[W, Tensor], object],
    x: Sliceable[W],
    *cuts: Cut,
    most: int | None = None,
    out: Tensor,
    holds: int = 5,
    host: bool = False,
) -> Tensor: ...


def split[W](
    fn: Callable[[W], Tensor] | Callable[[W, Tensor], object],
    x: Sliceable[W],
    *cuts: Cut,
    most: int | None = None,
    out: Tensor | None = None,
    holds: int = 5,
    host: bool = False,
) -> Tensor:
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
    the device's choice starts from before it has measured a pass.

    Where the device has no room for the whole output beside the windows, the memory keeps it in
    host memory the device addresses (a CUDA tensor all the same), and a `split` reading it
    brings each window back: slower, never out of memory. A blended cut's float32 whole goes
    there too, each window's weighted part added in place."""
    runtime = _RUNTIME
    shape = tuple(int(n) for n in x.shape)
    # Writing windows in place (`out`) can consume its input, so that never runs twice.
    pure = out is None and getattr(fn, "PURE_WINDOWS", False)
    if out is not None and any(c.scale != 1 or c.overlap for c in cuts):
        raise ValueError("split(out=) writes windows in place: its cuts take no scale or overlap")
    # A blended whole accumulates in float32 (`_run`).
    widen = max(4 // x.element_size(), 1) if any(c.overlap for c in cuts) else 1
    whole = 0 if out is not None or host else _bytes(shape, cuts, x) * widen
    hostable = bool(whole)
    key = _key(fn, shape, cuts, x.dtype)
    exact = f"{key}|{shape}"  # what the memory measures this call's whole under: its shape too
    progress = _Progress()  # what a retry of this call keeps: the windows it finished
    if most is not None and cuts:
        windows = _windows(shape, cuts, most)
        if runtime is None:
            return _run(fn, x, cuts, windows, None, key, out)
        # The largest window these cuts made, not `most`: a small input cuts smaller.
        largest = max(math.prod(b - a for a, b in w) for w in windows) * (
            math.prod(shape) // math.prod(shape[c.dim] for c in cuts)
        )
        return runtime.split(
            _name(fn),
            key,
            largest,
            largest,
            whole,
            _item_bytes(cuts, x),
            holds,
            lambda _, on_host: _run(fn, x, cuts, windows, runtime, exact, out, on_host, progress),
            pure,
            hostable,
            exact,
        )
    if runtime is None or not cuts:
        entire = cast("W", x)  # run whole, `fn` takes `x` as its own window (`Sliceable`)
        if out is None:
            return cast("Callable[[W], Tensor]", fn)(entire)  # no `out`: the first overload
        cast("Callable[[W, Tensor], object]", fn)(entire, out)  # `out`: the second overload
        return out
    return runtime.split(
        _name(fn),
        key,
        math.prod(shape),
        _finest(shape, cuts),
        whole,
        _item_bytes(cuts, x),
        holds,
        lambda most, on_host: _run(
            fn, x, cuts, _windows(shape, cuts, most), runtime, exact, out, on_host, progress
        ),
        pure,
        hostable,
        exact,
    )


@dataclass(frozen=True)
class Windows:
    """A window layout a loop keeps throughout (`windows`): `spans`, the `(start, stop)` of each
    window along its cut. Run each window's work inside `piece()`: its sites cut inside it, and
    a window refused memory once they can cut no finer makes the next call take more windows."""

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
    built for before it starts. `per_element_bytes` is what a window's work holds whole at its
    peak, in bytes per element of the window (every axis whole but the cut's): the attention, FFN
    and other sites inside it cut themselves to what is left. Starts fall on `cut.align`,
    neighbours share `cut.overlap`, and with `equal` every window has one length (the fewest
    equal windows at least as many as chosen). With `most`, the fewest of at most `most`
    elements, from the shape alone, on every device. Without a managing memory, one window
    unless `most` says more."""
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


def _key(fn: object, shape: tuple[int, ...], cuts: tuple[Cut, ...], dtype: torch.dtype) -> str:
    """Bytes per element depend on the op and its uncut extents."""
    cut = {c.dim % len(shape) for c in cuts}
    rest = tuple(n if i not in cut else -1 for i, n in enumerate(shape))
    return f"{getattr(fn, '__module__', '')}.{_name(fn)}{rest}:{dtype}"


def _item_bytes[W](cuts: tuple[Cut, ...], x: Sliceable[W]) -> float:
    """Output bytes per input element."""
    return float(math.prod(c.scale for c in cuts) * x.element_size())


def _bytes[W](shape: tuple[int, ...], cuts: tuple[Cut, ...], x: Sliceable[W]) -> int:
    """The whole output's bytes, which `split` allocates beside its windows."""
    return int(math.prod(shape) * _item_bytes(cuts, x))


def _name(fn: object) -> str:
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


@dataclass(slots=True)
class _Progress:
    """What one split call finished, kept across the memory's retries of it (cr-305): its whole
    output and blend weights so far, the windows written into them, and their dtype. A retry runs
    only the windows no finished one contains."""

    out: Tensor | None = None
    total: Tensor | None = None
    done: list[Window] = field(default_factory=list)
    dtype: torch.dtype | None = None
    #: a blended whole's float32 overlaps (`_Bands`); None once it fell back to a float32 whole
    bands: _Bands | None = None

    def finished(self, window: Window) -> bool:
        return any(
            all(a0 <= a and b <= b0 for (a, b), (a0, b0) in zip(window, d, strict=True))
            for d in self.done
        )


def _run[W](
    fn: Callable[[W], Tensor] | Callable[[W, Tensor], object],
    x: Sliceable[W],
    cuts: tuple[Cut, ...],
    windows: list[Window],
    runtime: Pieces | None,
    key: str,
    into: Tensor | None = None,
    host: bool = False,
    progress: _Progress | None = None,
) -> Tensor:
    """`fn` over `windows` of `x`, assembled whole (`into` where given), in host memory with
    `host`, skipping the windows `progress` says a refused try finished."""
    ndim = x.dim()
    dims = [c.dim % ndim for c in cuts]
    progress = progress if progress is not None else _Progress()

    def at(index: list[slice[int | None]]) -> W:
        window = x[tuple(index)]
        return window if runtime is None else runtime.read(window)

    if into is not None:  # each window writes its own rows of `into` (`split(out=)`)
        fill = cast("Callable[[W, Tensor], object]", fn)  # `out`: the second overload
        for window in windows:
            index = [slice(None)] * ndim
            for d, (a, b) in zip(dims, window, strict=True):
                index[d] = slice(a, b)
            if runtime is not None:
                runtime.piece()
            fill(at(index), into[tuple(index)])
        return into
    fn = cast("Callable[[W], Tensor]", fn)  # no `out`: the first overload
    if len(windows) == 1 and not progress.done:
        # run whole, `fn` takes `x` as its own window (`Sliceable`)
        return fn(cast("W", x if runtime is None else runtime.read(x)))
    blend = any(c.overlap for c in cuts)
    sizes = [math.ceil(x.shape[d] * c.scale - 1e-9) for d, c in zip(dims, cuts, strict=True)]
    if blend and progress.bands is None and not progress.done:
        progress.bands = _Bands(tuple(int(n) for n in x.shape), cuts, dims, windows, sizes)
    if runtime is not None:
        if progress.out is not None and host:
            progress.out = runtime.moved(progress.out)  # its finished windows come with it
        elif progress.out is None:
            # measured: before any window runs; a float32 whole only once it fell back to one
            progress.out = runtime.ahead(key, host, blend and progress.bands is None)
    if blend and progress.out is not None and progress.total is None:
        made = progress.out.shape
        progress.total = progress.out.new_zeros(
            [made[d] if d in dims else 1 for d in range(len(made))], dtype=_wide(progress.out.dtype)
        )

    def write(dst: Tensor, src: Tensor) -> None:
        if runtime is None:
            dst.copy_(src)
        else:
            runtime.write(dst, src)

    def band(shape: Sequence[int], kinds: str, like: Tensor) -> Tensor:
        """A float32 band of a blended whole, placed as the whole is (card, else host)."""
        probe = like.new_empty((0,), dtype=_wide(like.dtype))
        if runtime is None:
            return probe.new_zeros(shape)
        return runtime.whole(probe, shape, f"{key}|{kinds}", host, zeros=True)

    for window in windows:
        if progress.finished(window):
            continue
        index = [slice(None)] * ndim
        for d, (a, b) in zip(dims, window, strict=True):
            index[d] = slice(a, b)
        if runtime is not None:
            runtime.piece()
        part = fn(at(index))
        place = [slice(None)] * part.dim()
        for d, c, (a, _), size in zip(dims, cuts, window, sizes, strict=True):
            start = round(a * c.scale)
            place[d] = slice(start, min(start + int(part.shape[d]), size))
        if progress.out is None:
            shape = list(part.shape)
            for d, size in zip(dims, sizes, strict=True):
                shape[d] = size
            if blend:
                # The whole is in the windows' dtype; only a fallback's is float32 (`_Bands`).
                progress.out = (
                    part.new_empty(shape)
                    if runtime is None
                    else runtime.whole(part, shape, key, host)
                )
                progress.total = progress.out.new_zeros(
                    [size if d in dims else 1 for d, size in enumerate(shape)],
                    dtype=_wide(part.dtype),
                )
            else:
                progress.out = (
                    part.new_empty(shape)
                    if runtime is None
                    else runtime.whole(part, shape, key, host)
                )
        out = progress.out
        progress.dtype = part.dtype
        trim = tuple(slice(0, s.stop - s.start) if s.start is not None else s for s in place)
        if not blend:
            if runtime is None:
                out[tuple(place)] = part[trim]
            else:
                runtime.write(out[tuple(place)], part[trim])
            progress.done.append(window)
            continue
        held = cast("Tensor", progress.total)  # blending: made with `out`
        weight = held.new_ones([1] * part.dim())
        for d, c, (a, b) in zip(dims, cuts, window, strict=True):
            length = place[d].stop - place[d].start
            ramp = _ramp(length, round(c.overlap * c.scale), a > 0, b < x.shape[d])
            shape = [1] * part.dim()
            shape[d] = length
            weight = weight * held.new_tensor(ramp).view(shape)
        bands = progress.bands
        if bands is not None and not bands.add(out, part[trim], window, place, weight, write, band):
            # A window off the layout `_Bands` read from the windows: the float32 whole from here.
            progress.out = out = bands.widened(out, progress.done, write, band)
            progress.bands = None
        if progress.bands is None:
            out[tuple(place)] += part[trim].to(out.dtype) * weight
        held[tuple(s if i in dims else slice(None) for i, s in enumerate(place))] += weight
        progress.done.append(window)
    whole = cast("Tensor", progress.out)  # made at the latest at the first window
    if blend:
        total = cast("Tensor", progress.total)
        if progress.bands is not None:
            progress.bands.finish(whole, total, write)
            return whole
        whole.div_(total)
        dtype = cast("torch.dtype", progress.dtype)  # at least one window ran
        return whole.to(dtype) if runtime is None else runtime.narrowed(whole, dtype)
    return whole


def _wide(dtype: torch.dtype) -> torch.dtype:
    """What a blended whole accumulates in: float32, or the windows' own dtype if wider."""
    import torch

    return dtype if dtype.itemsize >= 4 else torch.float32


#: (start, stop, blended, offset): a run of output positions along one cut, whether one window
#: covers them at weight 1 (not blended) or not, and where it starts among positions of its kind
type _Run = tuple[int, int, bool, int]
type _Band = Callable[[Sequence[int], str, Tensor], Tensor]
type _Write = Callable[[Tensor, Tensor], None]


class _Bands:
    """A blended split's whole in its windows' dtype, float32 only where windows overlap.

    One float32 whole adds each window's `value * weight` and divides by the weights' sum at the
    end. An element one window covers at weight 1 ends as that window's value ((0 + v * 1) / 1,
    cast back), so it is written as its window comes. The others accumulate in float32 bands, one
    per class (inside each cut's overlaps or not), and divide at the end: the same operations in
    the same order as the float32 whole. SeedVR2's 2160p decode kept 36 GB of float32 and its
    18 GB narrowed copy on host for an 18 GB result (var-030 section 14).

    The layout is read from the windows before they run, each output span predicted as `split`
    places it; a window whose output lands elsewhere turns the call into the float32 whole it
    would have been (`widened`)."""

    def __init__(
        self,
        shape: tuple[int, ...],
        cuts: tuple[Cut, ...],
        dims: list[int],
        windows: list[Window],
        sizes: list[int],
    ) -> None:
        self.dims = dims
        self.places: list[dict[tuple[int, int], tuple[int, int]]] = []
        self.runs: list[list[_Run]] = []
        self.counts: list[tuple[int, int]] = []
        self.bands: dict[tuple[bool, ...], Tensor] = {}
        for d, c, size, spans in zip(dims, cuts, sizes, zip(*windows, strict=True), strict=True):
            cover, ones = [0] * size, [True] * size
            places: dict[tuple[int, int], tuple[int, int]] = {}
            for a, b in sorted(set(spans)):
                start = round(a * c.scale)
                stop = min(start + round((b - a) * c.scale), size)
                places[(a, b)] = (start, stop)
                ramp = _ramp(stop - start, round(c.overlap * c.scale), a > 0, b < shape[d])
                for at, weight in zip(range(start, stop), ramp, strict=True):
                    cover[at] += 1
                    ones[at] = ones[at] and weight == 1.0
            blended = [cover[at] != 1 or not ones[at] for at in range(size)]
            runs: list[_Run] = []
            counts = [0, 0]
            at = 0
            while at < size:
                end = at
                while end < size and blended[end] == blended[at]:
                    end += 1
                runs.append((at, end, blended[at], counts[blended[at]]))
                counts[blended[at]] += end - at
                at = end
            self.places.append(places)
            self.runs.append(runs)
            self.counts.append((counts[0], counts[1]))

    def _pieces(self, cut: int, start: int, stop: int) -> list[_Run]:
        """The runs of `cut` inside [start, stop), each clipped, its offset moved with its start."""
        return [
            (max(a, start), min(b, stop), kind, offset + max(a, start) - a)
            for a, b, kind, offset in self.runs[cut]
            if a < stop and start < b
        ]

    def _band(self, kinds: tuple[bool, ...], like: Tensor, make: _Band) -> Tensor:
        held = self.bands.get(kinds)
        if held is None:
            shape = list(like.shape)
            for cut, (d, kind) in enumerate(zip(self.dims, kinds, strict=True)):
                shape[d] = self.counts[cut][kind]
            held = self.bands[kinds] = make(shape, "".join("b" if k else "i" for k in kinds), like)
        return held

    def add(
        self,
        out: Tensor,
        part: Tensor,
        window: Window,
        place: list[slice[int | None]],
        weight: Tensor,
        write: _Write,
        make: _Band,
    ) -> bool:
        """`part` (trimmed to `place`) into `out` and the bands; False, with nothing written, when
        its output span is not the one the layout read."""
        spans = [
            (int(cast("int", place[d].start)), int(cast("int", place[d].stop))) for d in self.dims
        ]
        if any(
            self.places[cut].get(w) != s
            for cut, (w, s) in enumerate(zip(window, spans, strict=True))
        ):
            return False
        pieces = [self._pieces(cut, *span) for cut, span in enumerate(spans)]
        for combo in itertools.product(*pieces):
            whole, local, inner = (
                list(place),
                [slice(None)] * part.dim(),
                [slice(None)] * part.dim(),
            )
            for d, (lo, hi, _, offset), (start, _) in zip(self.dims, combo, spans, strict=True):
                whole[d], local[d], inner[d] = (
                    slice(lo, hi),
                    slice(lo - start, hi - start),
                    slice(offset, offset + hi - lo),
                )
            kinds = tuple(kind for _, _, kind, _ in combo)
            if not any(kinds):
                write(out[tuple(whole)], part[tuple(local)])
                continue
            share = weight[tuple(local)]
            self._band(kinds, out, make)[tuple(inner)] += part[tuple(local)].to(share.dtype) * share
        return True

    def _places(
        self, kinds: tuple[bool, ...], ndim: int
    ) -> Iterator[tuple[tuple[slice[int | None], ...], tuple[slice[int | None], ...]]]:
        """Each block of a class: its index in the whole and in its band."""
        runs = [[r for r in self.runs[cut] if r[2] == kind] for cut, kind in enumerate(kinds)]
        for combo in itertools.product(*runs):
            whole, inner = [slice(None)] * ndim, [slice(None)] * ndim
            for d, (lo, hi, _, offset) in zip(self.dims, combo, strict=True):
                whole[d], inner[d] = slice(lo, hi), slice(offset, offset + hi - lo)
            yield tuple(whole), tuple(inner)

    def finish(self, out: Tensor, total: Tensor, write: _Write) -> None:
        """Each band's sums over their weights' sums, cast to the whole's dtype, into the whole."""
        for kinds, held in self.bands.items():
            for whole, inner in self._places(kinds, out.dim()):
                weights = tuple(s if i in self.dims else slice(None) for i, s in enumerate(whole))
                write(out[whole], (held[inner] / total[weights]).to(out.dtype))
        self.bands.clear()

    def widened(self, out: Tensor, done: list[Window], write: _Write, make: _Band) -> Tensor:
        """The float32 whole the finished windows would have made: their single-window elements
        and the bands."""
        wide = make(list(out.shape), "", out)
        for window in done:
            spans = [self.places[cut][w] for cut, w in enumerate(window)]
            for combo in itertools.product(*(self._pieces(cut, *s) for cut, s in enumerate(spans))):
                if any(kind for _, _, kind, _ in combo):
                    continue
                index = [slice(None)] * out.dim()
                for d, (lo, hi, _, _) in zip(self.dims, combo, strict=True):
                    index[d] = slice(lo, hi)
                write(wide[tuple(index)], out[tuple(index)].to(wide.dtype))
        for kinds, held in self.bands.items():
            for whole, inner in self._places(kinds, out.dim()):
                write(wide[whole], held[inner])
        self.bands.clear()
        return wide


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
