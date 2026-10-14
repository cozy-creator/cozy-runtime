"""An op a package lets run in pieces: the package says how it splits (`Cut`), the Runtime
says how finely, from the bytes the op measured per input element and what the device has
free for it (idle weights unmap first). Whole when it fits: a larger card runs fewer pieces
of the same request, and none of the package's code names a size."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from cozy_runtime.author._errors import is_device_oom

#: The refusal an op ends with when the device cannot give it the bytes it needs at once.
EXCEEDS = "operation_exceeds_device"


@dataclass(frozen=True, slots=True)
class Cut:
    """A dimension `fn` may be cut along. `dim` indexes its input and its output alike; the
    output's is `scale` times the input's (8 for a latent decode, 1/8 for its encode). Pieces
    start on multiples of `align` input elements and neighbours share `overlap` of them, which
    the output cross-fades; with none the pieces join exactly (a batch, a frame axis that
    carries no state)."""

    dim: int
    scale: float = 1.0
    overlap: int = 0
    align: int = 1


class Pieces(Protocol):
    """The Runtime's side of `split`, one per executor process."""

    def room(self, key: str) -> tuple[float | None, int]:
        """The peak bytes per input element `key` measured (None before its first run), and
        the device bytes one piece may take now."""
        ...

    def piece(self, key: str, elements: int, need: int) -> AbstractContextManager[None]:
        """One piece of `elements` input elements: room for `need` bytes first, its peak
        measured and kept for `key`."""
        ...

    def ran(self, name: str, pieces: int) -> None:
        """The op `name` ran in `pieces` pieces: the run's record of how tight memory was."""
        ...


_RUNTIME: Pieces | None = None


def install(runtime: Pieces | None) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def split(fn: Callable[[Any], Any], x: Any, *cuts: Cut) -> Any:
    """`fn(x)`, with `fn` a tensor-to-tensor op whose result on a window of `x` along `cuts`
    is that window of its whole result (up to `overlap`). The Runtime runs it whole when the
    device can hold it, else in the fewest pieces that fit; a piece that still runs out of
    memory runs again at half the size. Without a Runtime that manages device memory (a CPU
    executor, a test) it runs whole."""
    runtime = _RUNTIME
    if runtime is None or not cuts:
        return fn(x)
    shape = tuple(int(n) for n in x.shape)
    key = _key(fn, shape, cuts, x.dtype)
    ratio, room = runtime.room(key)
    windows = _windows(shape, cuts, room / ratio if ratio else math.inf)
    while True:
        try:
            out = _run(fn, x, cuts, windows, runtime, key, ratio or 0.0)
            runtime.ran(_name(fn), len(windows))
            return out
        except Exception as exc:
            if not _short(exc) or len(_halved(shape, cuts, windows)) <= len(windows):
                raise
        # outside `except`, so the failed piece's frames are gone
        windows = _halved(shape, cuts, windows)


type Window = tuple[tuple[int, int], ...]


def _key(
    fn: Callable[[Any], Any], shape: tuple[int, ...], cuts: tuple[Cut, ...], dtype: Any
) -> str:
    """Bytes per element follow the op and every extent it is not cut along."""
    cut = {c.dim % len(shape) for c in cuts}
    rest = tuple(n if i not in cut else -1 for i, n in enumerate(shape))
    return f"{_name(fn)}{rest}:{dtype}"


def _name(fn: Callable[[Any], Any]) -> str:
    return str(getattr(fn, "__qualname__", type(fn).__qualname__))


def _short(exc: BaseException) -> bool:
    return is_device_oom(exc) or getattr(exc, "code", None) == EXCEEDS


def _spans(length: int, cut: Cut, count: int) -> list[tuple[int, int]]:
    """About `count` spans covering `length`: aligned starts, `overlap` shared, none shorter
    than an overlap past it."""
    step = max(-(-max(length - cut.overlap, 1) // count), cut.overlap, 1)
    step = -(-step // cut.align) * cut.align
    starts = range(0, max(length - cut.overlap, 1), step)
    return [(a, min(a + step + cut.overlap, length)) for a in starts]


def _windows(shape: tuple[int, ...], cuts: tuple[Cut, ...], most: float) -> list[Window]:
    """The fewest windows with at most `most` elements each where the cuts allow, the longest
    pieces cut first."""
    lengths = [shape[c.dim] for c in cuts]
    counts = [1] * len(cuts)
    spans = [_spans(n, c, 1) for n, c in zip(lengths, cuts, strict=True)]
    while _largest(shape, cuts, spans) > most:
        for i in sorted(range(len(cuts)), key=lambda i: -_longest(spans[i])):
            more = next(
                (
                    n
                    for n in range(counts[i] + 1, lengths[i] + 1)
                    if _longest(_spans(lengths[i], cuts[i], n)) < _longest(spans[i])
                ),
                0,
            )
            if more:
                counts[i], spans[i] = more, _spans(lengths[i], cuts[i], more)
                break
        else:
            break
    windows: list[Window] = [()]
    for axis in spans:
        windows = [(*w, s) for w in windows for s in axis]
    return windows


def _halved(shape: tuple[int, ...], cuts: tuple[Cut, ...], windows: list[Window]) -> list[Window]:
    return _windows(shape, cuts, _largest(shape, cuts, list(zip(*windows, strict=True))) // 2)


def _longest(spans: Sequence[tuple[int, int]]) -> int:
    return max(b - a for a, b in spans)


def _largest(
    shape: tuple[int, ...], cuts: tuple[Cut, ...], spans: Sequence[Sequence[tuple[int, int]]]
) -> int:
    """Elements in the largest window, from each cut's spans."""
    rest = math.prod(shape) // math.prod(shape[c.dim] for c in cuts)
    return rest * math.prod(_longest(s) for s in spans)


def _run(
    fn: Callable[[Any], Any],
    x: Any,
    cuts: tuple[Cut, ...],
    windows: list[Window],
    runtime: Pieces,
    key: str,
    ratio: float,
) -> Any:
    if len(windows) == 1:
        with runtime.piece(key, x.numel(), int(ratio * x.numel())):
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
        piece = x[tuple(index)]
        with runtime.piece(key, piece.numel(), int(ratio * piece.numel())):
            part = fn(piece)
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
    """1 inside, a cosine rise over the `overlap` a neighbour shares at either end."""
    weight = [1.0] * length
    span = max(0, min(overlap, length - 1))
    if span:
        rise = [0.5 - 0.5 * math.cos(math.pi * i / max(span - 1, 1)) for i in range(span)]
        if start:
            weight[:span] = rise
        if end:
            weight[length - span :] = rise[::-1]
    return weight
