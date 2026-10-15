"""Large maps computed in bands of rows, exactly: no tiles, no blending.

An op's output rows need a known span of each input's rows: a convolution's reach (kernel,
stride, dilation; zeros only past the input's edges), a nearest upsampling's source rows, a
padding's shifted rows, the same rows for a pointwise op, a cast, a copy, a view that keeps the
rows or a norm over channels. A group norm gathers its statistics over every band first, in
float32. Ops that mix rows (attention, matmuls) are never cut. Nothing here knows a model: the
rows an op needs follow from the op and its arguments.

A row-local op whose result is a large map is not computed when called: its result (`Rows`)
computes any band of its rows on demand, from its inputs' rows. What reads it in rows (a
convolution, a group norm, another row-local op) runs band by band, so a chain of them runs on
band-sized temporaries a cache holds, never on full-size ones. Any other op reads it whole,
computed then. A write into what a pending result reads computes that result first; none
outlives the call.

An op runs in bands, too, when an operand lives in host memory (a spill, or an earlier band's
result: each band reads its rows over the bus once, where a kernel would read them again and
again), or when it ran out of device memory whole. A result goes on the device when it fits
there with idle weights unmapped and the least recently used tensors in host memory, else into
host memory. A convolution right after the zero padding it reads skips the taps that meet only
that padding: a lone frame through a causal 3D convolution is a 2D one.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from cozy_runtime.author._errors import is_device_oom
from cozy_runtime.internal import spill

if TYPE_CHECKING:
    from torch import Tensor

    from cozy_runtime.internal.weights import Weights

#: A row-local op's result of at least this many bytes, four dims or more, is computed by its
#: reader (`Rows`).
LAZY = 32 << 20
#: Elements of a result's band when a chain computes into it: band-sized temporaries a cache holds.
BAND = 1 << 22
#: Source reads per row past which a chain's pending inputs are computed first.
READS = 4

type Rule = Callable[[Any, tuple[Any, ...], dict[str, Any]], "Plan | None"]


@dataclass(slots=True)
class Plan:
    """An op as rows of its result (`out`, a meta tensor) along `axis`: `band(a, b)` computes
    rows [a, b), which start on multiples of `align`; one row takes `per_row` device bytes
    while it computes. `gather(rows)` reads what every band needs first (a norm's
    statistics); `finish` makes the op's own result from the filled tensor. `sources`: the
    tensors its bands read."""

    out: Any
    axis: int
    band: Callable[[int, int], Any]
    per_row: int
    align: int = 1
    gather: Callable[[int], None] | None = None
    finish: Callable[[Any], Any] | None = None
    sources: list[Any] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Zeros:
    """The last op's result `padded`: `source` with `pads` zeros (before, after) per dim."""

    padded: Any
    source: Any
    pads: tuple[tuple[int, int], ...]


class Bands:
    """One executor's banded ops (`Weights.bands`), run from its recovery mode."""

    def __init__(self, weights: Weights) -> None:
        self.weights = weights
        torch = self.torch = weights.torch
        self.aten = aten = torch.ops.aten
        self.pointwise, self.seeded = torch.Tag.pointwise, torch.Tag.nondeterministic_seeded
        self.Rows = _rows_type(torch)
        #: pending results by id, computed when the call ends or a write reaches what they read
        self.live: dict[int, weakref.ref[Any]] = {}
        self.last: _Zeros | None = None
        #: two streams bands alternate on, made before any weight maps: on a full card the
        #: driver cannot make one (run 4297: `cuStreamCreateWithPriority`)
        self.streams: tuple[Any, ...] = ()
        if weights.device.type == "cuda":
            self.streams = (torch.cuda.Stream(weights.device), torch.cuda.Stream(weights.device))
        #: this call's banded ops and the bytes of their results put in host memory; ops
        #: rewritten since the process began (`Effects.banded`)
        self.ops = self.hosted = self.rewrites = 0
        self.pads = {aten.pad, aten.constant_pad_nd}
        self.convs = {aten.conv1d, aten.conv2d, aten.conv3d, aten.convolution}
        ups = (
            aten.upsample_nearest1d,
            aten.upsample_nearest2d,
            aten.upsample_nearest3d,
            aten._upsample_nearest_exact1d,
            aten._upsample_nearest_exact2d,
            aten._upsample_nearest_exact3d,
        )
        rows = (
            aten.to,
            aten._to_copy,
            aten.type_as,
            aten.clone,
            aten.contiguous,
            aten.linalg_vector_norm,
            aten.repeat_interleave,
            aten.cat,
        )
        views = (
            aten.view,
            aten._unsafe_view,
            aten.reshape,
            aten.expand,
            aten.expand_as,
            aten.squeeze,
            aten.unsqueeze,
            aten.permute,
            aten.select,
            aten.slice,
            aten.alias,
            aten.detach,
        )
        self.rules: dict[Any, Rule] = {
            **dict.fromkeys(self.convs, self._conv),
            **dict.fromkeys(self.pads, self._pad),
            **dict.fromkeys(ups, self._upsample),
            **dict.fromkeys((aten.group_norm, aten.native_group_norm), self._norm),
            **dict.fromkeys(rows, self._rows),
            **dict.fromkeys(views, self._view),
        }
        self.lazy = {self._rows, self._pad, self._upsample, self._view}

    def run(
        self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], short: bool = False
    ) -> Any:
        """`func(*args, **kwargs)` as rows, or NotImplemented: the op runs as it is. `short`:
        it just ran out of device memory whole."""
        last, self.last = self.last, None
        if func._schema.is_mutable:
            self._before_write(func, args, kwargs)
            return NotImplemented
        packet = func.overloadpacket
        if packet == self.aten.dropout and not _arg(func, args, kwargs, "train"):
            return args[0]  # as the op does: its input, still pending if it is
        rule = self.rules.get(packet)
        if rule is None and self.pointwise in func.tags:
            rule = self._rows
        if rule is None or self.seeded in func.tags:
            return NotImplemented
        tensors = _tensors((args, kwargs), self.torch.Tensor)
        pending = any(isinstance(t, self.Rows) and t.value is None for t in tensors)
        hosted = bool(spill.HELD) and any(map(spill.hosted, tensors))
        if rule == self._view and not pending:
            return NotImplemented  # a view of a computed tensor is a view
        if pending and rule == self._rows and self._same(func, args, kwargs):
            return args[0]  # a cast to its own dtype: its operand, still pending
        lazy = rule in self.lazy
        if not (short or hosted or pending):
            big = max(map(_bytes, tensors), default=0) >= LAZY // 8
            if (lazy or rule == self._norm) and big:
                plan = rule(func, args, kwargs)
                if plan is not None and plan.out.dim() >= 4 and _bytes(plan.out) >= LAZY:
                    return self._defer(func, plan, args, kwargs)
            if packet in self.pads:
                return self._remember(func, args, kwargs)
            plan = None
            if packet in self.convs and last is not None and args[0] is last.padded:
                plan = self._conv(func, args, kwargs, last)  # a convolution skipping taps
            if plan is None:
                return NotImplemented
            self.rewrites += 1
            return plan.band(0, int(plan.out.shape[plan.axis]))
        plan = None
        if packet in self.convs and last is not None and args[0] is last.padded:
            plan = self._conv(func, args, kwargs, last)
        plan = plan or rule(func, args, kwargs)
        if plan is None:
            return NotImplemented
        if lazy or (plan.gather is not None and plan.out.dim() >= 4):
            return self._defer(func, plan, args, kwargs)
        self.rewrites += 1
        out = self._banded(plan, (*args, *kwargs.values()))
        return out if plan.finish is None else plan.finish(out)

    def _same(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        """Whether the op returns its first operand itself (a cast to its own dtype)."""
        first = self.torch.empty_strided(
            args[0].shape, args[0].stride(), dtype=args[0].dtype, device="meta"
        )
        try:
            return func(first, *args[1:], **kwargs) is first
        except Exception:
            return False

    def settle(self) -> None:
        """Every pending result computed: what reads it next may be anything."""
        for rows in self._pending():
            rows.whole()

    # -- pending results

    def _defer(self, func: Any, plan: Plan, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """The op's result pending (`Rows`); a group norm's statistics gathered first. A chain
        whose rows would read its sources too often is computed up to here first."""
        self.rewrites += 1
        operands = _tensors((args, kwargs), self.torch.Tensor)
        if sum(getattr(t, "reads", 1) for t in operands if _spans(t, plan.out, plan.axis)) > READS:
            for t in operands:
                if isinstance(t, self.Rows):
                    t.whole()
        if plan.gather is not None:
            self._gather(plan, _band_rows(plan.out, plan.axis, plan.align))
        rows = self.Rows(self, plan, operands)
        if func.overloadpacket in self.pads:
            self._zeros(func, rows, args, kwargs)
        return rows if plan.finish is None else plan.finish(rows)

    def _before_write(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """A write into a tensor a pending result reads: that result is computed first."""
        if not self.live:
            return
        written = {_address(t) for t in _written(func, args, kwargs)} - {-1}
        for rows in self._pending():
            if rows.sources & written:
                rows.whole()

    def _pending(self) -> list[Any]:
        return [rows for ref in list(self.live.values()) if (rows := ref()) is not None]

    # -- placing and running bands

    def _banded(self, plan: Plan, operands: Any) -> Any:
        """The op in bands, its result on the device when it fits there: idle weights unmap
        and the tensors used least recently move to host memory first (`spill`); else in
        host memory. A result a pending chain computes into goes in cache-sized bands."""
        torch, weights = self.torch, self.weights
        meta, axis = plan.out, plan.axis
        height = int(meta.shape[axis])
        nbytes = _span(meta) * meta.element_size()
        free, movable = weights.activation_room()
        room, least = free + movable, nbytes + 2 * plan.per_row * plan.align
        if least > room and weights.spill(least - room, operands):
            free, movable = weights.activation_room()
            room = free + movable
        on_device = least <= room or not self._hostable(nbytes)
        # two bands in flight (`_on`)
        rows = (room - (nbytes if on_device else 0)) // max(2 * plan.per_row, 1)
        if any(isinstance(t, self.Rows) and t.value is None for t in plan.sources):
            rows = min(rows, _band_rows(meta, axis, plan.align))
        rows = max(plan.align, min(height, rows) // plan.align * plan.align)
        out = None
        if on_device:
            weights.free_for(nbytes + 2 * rows * plan.per_row)
            try:
                out = torch.empty_strided(
                    meta.shape, meta.stride(), dtype=meta.dtype, device=weights.device
                )
            except Exception as exc:
                if not is_device_oom(exc) or not self._hostable(nbytes):
                    raise
        if "bands" not in weights.modes:
            self.ops = self.hosted = 0
        self.ops += 1
        if out is None:
            out = spill.empty(torch, weights.device, meta)
            self.hosted += nbytes
        hosted = f", {self.hosted} B of results in host memory" if self.hosted else ""
        weights.modes["bands"] = f"{self.ops} ops{hosted}"
        rows = self._gather(plan, rows)
        # Bands alternate between two streams: one's rows cross the bus while the other's
        # compute. Each band's tensors live and die on its own stream.
        main: Any = torch.cuda.current_stream(weights.device) if self.streams else None
        for stream in self.streams:
            stream.wait_stream(main)
        a, turn = 0, 0
        try:
            while a < height:
                b = min(a + rows, height)
                try:
                    with self._on(turn):
                        out.narrow(axis, a, b - a).copy_(plan.band(a, b))
                except Exception as exc:
                    if not is_device_oom(exc) or rows <= plan.align:
                        raise
                    rows = max(plan.align, rows // 2 // plan.align * plan.align)
                    continue
                a, turn = b, turn + 1
        finally:
            for stream in self.streams:
                main.wait_stream(stream)
        return out

    def _gather(self, plan: Plan, rows: int) -> int:
        """A norm's statistics, at a band height memory allows; the height that did."""
        while plan.gather is not None:
            try:
                plan.gather(rows)
                plan.gather = None
            except Exception as exc:
                if not is_device_oom(exc) or rows <= plan.align:
                    raise
                rows = max(plan.align, rows // 2 // plan.align * plan.align)
        return rows

    def _on(self, turn: int) -> AbstractContextManager[Any]:
        if not self.streams:
            return nullcontext()
        return cast("AbstractContextManager[Any]", self.torch.cuda.stream(self.streams[turn % 2]))

    def _hostable(self, nbytes: int) -> bool:
        """Whether a result of `nbytes` may live in host memory (never a group's)."""
        if self.weights.grouped:
            return False
        lockable = self.weights.lockable(nbytes)
        return lockable < 0 or nbytes <= lockable

    def _take(self, x: Any, dim: int, lo: int, hi: int, fill: float = 0.0) -> Any:
        """Rows [lo, hi) of `x` along `dim` on the device, `fill` past its edges: a pending
        result's computed now; a host block's copied once; a device tensor's a view unless
        padded."""
        n = int(x.shape[dim])
        start, stop = min(max(lo, 0), n), max(min(hi, n), 0)
        if isinstance(x, self.Rows) and x.value is None:
            if dim != x.dim() - 2:
                x = x.whole()  # its rows are another dim's
        if isinstance(x, self.Rows) and x.value is None:
            rows = x.rows(start, stop)
        else:
            x = x.value if isinstance(x, self.Rows) else x
            rows = x.narrow(dim, start, max(stop - start, 0))
            if spill.hosted(x):
                rows = rows.clone(memory_format=self.torch.contiguous_format)
        if start - lo or hi - stop:
            pad = [0, 0] * (x.dim() - 1 - dim) + [start - lo, hi - stop]
            rows = self.aten.constant_pad_nd(rows, pad, fill)
        return rows

    def _narrowed(self, value: Any, out: Any, axis: int, a: int, b: int) -> Any:
        """`value` as the band [a, b) of `out` reads it: a tensor spanning `out`'s rows
        (aligned from the right, as broadcasting does) narrowed; any other whole."""
        if isinstance(value, (list, tuple)):
            return type(value)(self._narrowed(v, out, axis, a, b) for v in value)
        if not isinstance(value, self.torch.Tensor):
            return value
        if not _spans(value, out, axis):
            return value.whole() if isinstance(value, self.Rows) else value
        return self._take(value, value.dim() - (out.dim() - axis), a, b)

    def _meta(
        self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], views: bool = False
    ) -> Any:
        """The op's result as a meta tensor; None when it is not one new tensor (a view, with
        `views`, is one)."""
        made: list[Any] = []

        def meta(value: Any) -> Any:
            if isinstance(value, (list, tuple)):
                return type(value)(map(meta, value))
            if isinstance(value, self.torch.Tensor):
                made.append(
                    self.torch.empty_strided(
                        value.shape, value.stride(), dtype=value.dtype, device="meta"
                    )
                )
                return made[-1]
            return value

        try:
            out = func(*meta(args), **{k: meta(v) for k, v in kwargs.items()})
        except Exception:  # no meta kernel, or arguments it refuses: the op runs as it is
            return None
        if not isinstance(out, self.torch.Tensor) or any(out is m for m in made):
            return None  # no one tensor, or an operand itself
        if out._is_view() and not views:
            return None  # an alias of an operand is never cut
        return out

    # -- rules: an op's output rows from its inputs' rows

    def _rows(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Plan | None:
        """A pointwise op, a cast or copy on one device, a norm, repeat or join along dims
        before the last two: each output row reads the same row of every operand that spans
        the rows (aligned from the right, as broadcasting does)."""
        packet = func.overloadpacket
        if packet == self.aten.to and func._overloadname != "dtype":
            return None
        if packet == self.aten._to_copy:
            device = _arg(func, args, kwargs, "device")
            if device is not None and self.torch.device(device) != args[0].device:
                return None
        dims = _arg(func, args, kwargs, "dim")
        if dims is not None:
            rank = _tensors(args[:1], self.torch.Tensor)[0].dim()
            dims = [dims] if isinstance(dims, int) else list(dims)
            if not dims or any(d % rank >= rank - 2 for d in dims):
                return None
        out = self._meta(func, args, kwargs)
        if out is None or out.dim() < 2:
            return None
        axis = out.dim() - 2
        height = int(out.shape[axis])
        if height < 2:
            return None
        sources = [t for t in _tensors((args, kwargs), self.torch.Tensor) if _spans(t, out, axis)]
        per_row = (_bytes(out) + sum(map(_bytes, sources))) // height

        def band(a: int, b: int) -> Any:
            return func(
                *self._narrowed(args, out, axis, a, b),
                **{k: self._narrowed(v, out, axis, a, b) for k, v in kwargs.items()},
            )

        return Plan(out, axis, band, per_row, sources=sources)

    def _view(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Plan | None:
        """A view of a pending result that keeps its rows (the same extent and stride as the
        second-to-last dim on both sides): each band is the view of the result's band."""
        x = args[0]
        out = self._meta(func, args, kwargs, views=True)
        if out is None or out.dim() < 2 or x.dim() < 2:
            return None
        if out.shape[-2] != x.shape[-2] or out.stride()[-2] != x.stride()[-2]:
            return None
        axis, packet = out.dim() - 2, func.overloadpacket
        sized = packet in (self.aten.view, self.aten._unsafe_view, self.aten.reshape)

        def band(a: int, b: int) -> Any:
            rows = x.rows(a, b)
            if sized or packet in (self.aten.expand, self.aten.expand_as):
                shape = list(out.shape)
                shape[axis] = b - a
                return rows.reshape(shape) if sized else rows.expand(shape)
            return func(rows, *args[1:], **kwargs)

        rows = Plan(out, axis, band, _bytes(out) // max(int(out.shape[axis]), 1), sources=[x])
        rows.finish = lambda pending: pending.based(x, func, args, kwargs)
        return rows

    def _upsample(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Plan | None:
        """Nearest (or nearest-exact) upsampling by a whole factor `r` along the rows: output
        rows [a, b) on multiples of `r` read input rows [a/r, b/r)."""
        x = args[0]
        out = self._meta(func, args, kwargs)
        if out is None or x.dim() < 3:
            return None
        axis = x.dim() - 2 if x.dim() > 3 else 2
        height, source = int(out.shape[axis]), int(x.shape[axis])
        ratio = height // max(source, 1)
        if ratio < 2 or ratio * source != height:
            return None
        names = [argument.name for argument in func._schema.arguments]

        def band(a: int, b: int) -> Any:
            values = dict(zip(names, args, strict=False)) | kwargs
            if values.get("output_size") is not None:
                size = list(values["output_size"])
                size[axis - 2] = b - a
                values["output_size"] = size
            values[names[0]] = self._take(x, axis, a // ratio, -(-b // ratio))
            return func(**values)

        per_row = (_bytes(out) + _bytes(x)) // height
        return Plan(out, axis, band, per_row, align=ratio, sources=[x])

    def _pad(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Plan | None:
        """A constant padding: output rows [a, b) are input rows [a - top, b - top), the
        value past its edges; every other dim padded as asked."""
        x, pads, value = self._padding(func, args, kwargs)
        if pads is None or min(pads, default=0) < 0 or x.dim() < 2:
            return None
        out = self._meta(self.aten.constant_pad_nd, (x, pads, value), {})
        if out is None:
            return None
        axis = x.dim() - 2
        top = pads[2] if len(pads) >= 4 else 0
        rest = [0 if i in (2, 3) else p for i, p in enumerate(pads)]

        def band(a: int, b: int) -> Any:
            rows = self._take(x, axis, a - top, b - top, value)
            return self.aten.constant_pad_nd(rows, rest, value)

        per_row = (_bytes(out) + _bytes(x)) // max(int(out.shape[axis]), 1)
        return Plan(out, axis, band, per_row, sources=[x])

    def _padding(
        self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, list[int] | None, float]:
        x, pads = args[0], [int(p) for p in _arg(func, args, kwargs, "pad")]
        if func.overloadpacket == self.aten.pad and _arg(func, args, kwargs, "mode") != "constant":
            return x, None, 0.0
        return x, pads, float(_arg(func, args, kwargs, "value") or 0.0)

    def _remember(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """A padding with room runs as it is; the op after it may read past it."""
        out = func(*args, **kwargs)
        self._zeros(func, out, args, kwargs)
        return out

    def _zeros(self, func: Any, out: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        x, pads, value = self._padding(func, args, kwargs)
        if pads is None or value != 0 or min(pads, default=0) < 0:
            return
        pairs = [(0, 0)] * x.dim()
        for i in range(len(pads) // 2):
            pairs[x.dim() - 1 - i] = (pads[2 * i], pads[2 * i + 1])
        self.last = _Zeros(out, x, tuple(pairs))

    def _conv(
        self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], zeros: _Zeros | None = None
    ) -> Plan | None:
        """A convolution's output rows [a, b) read input rows [a*s - p, (b-1)*s - p + d*(k-1)
        + 1), zeros past the edges. Right after a zero padding (`zeros`) it reads the
        padding's source, the padding folded into its own and the taps that meet only zeros
        dropped; None when no tap or dim drops."""
        spec = _conv_spec(func, args, kwargs)
        if spec is None:
            return None
        x, w, bias, stride, padding, dilation, groups = spec
        n = w.dim() - 2
        out = self._meta(func, args, kwargs)
        if out is None or x.dim() != w.dim() or n < 1:
            return None
        front, back, dropped = list(padding), list(padding), False
        if zeros is not None:
            folded = _fold(zeros, w, dilation, padding, out)
            if folded is None:
                return None
            x, w, front, back, dropped = folded
        j = n - 2 if n >= 2 else 0
        axis, s, d, k = 2 + j, stride[j], dilation[j], int(w.shape[2 + j])
        # a dim of one element, one tap and no padding drops out (3D -> 2D)
        drop = [
            2 + i
            for i in range(n)
            if i != j and x.shape[2 + i] == w.shape[2 + i] == 1 and front[i] == back[i] == 0
        ]
        if zeros is not None and not (dropped or drop):
            return None  # the padded tensor exists already: reading its source gains nothing
        conv = self.aten.convolution

        def band(a: int, b: int) -> Any:
            lo, hi = a * s - front[j], (b - 1) * s - front[j] + d * (k - 1) + 1
            edge = max(-lo, 0)
            if edge == max(hi - int(x.shape[axis]), 0):  # as many zeros both sides: its own
                rows = self._take(x, axis, lo + edge, hi - edge)
            else:
                rows, edge = self._take(x, axis, lo, hi), 0
            pad = [edge if i == j else front[i] for i in range(n)]
            weight = w
            if drop:
                rows, weight = rows.squeeze(drop), w.squeeze(drop)
            kept = [i for i in range(n) if 2 + i not in drop]
            y = conv(
                rows,
                weight,
                bias,
                [stride[i] for i in kept],
                [pad[i] for i in kept],
                [dilation[i] for i in kept],
                False,
                [0] * len(kept),
                groups,
            )
            for dim in drop:
                y = y.unsqueeze(dim)
            return y

        per_row = (2 * _bytes(out) + _bytes(x) * s) // max(int(out.shape[axis]), 1)
        return Plan(out, axis, band, per_row, sources=[x])

    def _norm(self, func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Plan | None:
        """A group norm: per image and group, mean and variance over every band in float32
        (merged as one pass over all of it would be, up to rounding), then each band
        normalized."""
        torch = self.torch
        x = args[0]
        if func.overloadpacket == self.aten.group_norm:
            groups = int(_arg(func, args, kwargs, "num_groups"))
            weight, bias = _arg(func, args, kwargs, "weight"), _arg(func, args, kwargs, "bias")
        else:
            weight, bias, groups = args[1], args[2], int(args[6])
        eps = float(_arg(func, args, kwargs, "eps"))
        if x.dim() < 3 or x.shape[1] % groups:
            return None
        axis = x.dim() - 2 if x.dim() > 3 else 2
        batch, channels, height = int(x.shape[0]), int(x.shape[1]), int(x.shape[axis])
        out = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="meta")
        shape = [batch, channels] + [1] * (x.dim() - 2)
        affine: list[Any] = []
        moments: list[Any] = []

        def gather(rows: int) -> None:
            parts = []
            for a in range(0, height, rows):
                flat = self._take(x, axis, a, min(a + rows, height)).float()
                flat = flat.reshape(batch, groups, -1)
                parts.append((*torch.var_mean(flat, dim=-1, correction=0), flat.shape[-1]))
            total = sum(count for _, _, count in parts)
            mean = sum(m * (count / total) for _, m, count in parts)
            var = sum((v + (m - mean) ** 2) * (count / total) for v, m, count in parts)
            rstd = (var + eps).rsqrt()
            scale = rstd.repeat_interleave(channels // groups, dim=1)
            if weight is not None:
                scale = scale * weight.float()
            shift = -mean.repeat_interleave(channels // groups, dim=1) * scale
            if bias is not None:
                shift = shift + bias.float()
            affine[:] = [scale.reshape(shape), shift.reshape(shape)]
            moments[:] = [mean, rstd]

        def band(a: int, b: int) -> Any:
            rows = self._take(x, axis, a, b)
            scale, shift = affine
            return torch.addcmul(shift, rows, scale, out=torch.empty_like(rows))

        def finish(result: Any) -> Any:
            if func.overloadpacket == self.aten.group_norm:
                return result
            mean, rstd = moments
            return result, mean.to(x.dtype), rstd.to(x.dtype)

        per_row = (2 * _bytes(x) + 4 * x.numel()) // max(height, 1)
        return Plan(out, axis, band, per_row, gather=gather, finish=finish, sources=[x])


def _rows_type(torch: Any) -> type:
    """`Rows` over this torch: a wrapper tensor with no storage of its own."""

    class Rows(torch.Tensor):  # type: ignore[misc,name-defined]
        """A pending result: `rows(a, b)` computes its rows [a, b); `whole()` all of them,
        once. A view of one keeps its base (`based`), so writes through it reach the base.
        Code that reads its storage (a kernel handed its pointer) gets it computed."""

        @property
        def pending(self) -> bool:
            return self.value is None

        @staticmethod
        def __new__(cls, owner: Bands, plan: Plan, operands: list[Any]) -> Any:
            meta = plan.out
            pending = torch.Tensor._make_wrapper_subclass(
                cls,
                meta.shape,
                strides=meta.stride(),
                dtype=meta.dtype,
                device=owner.weights.device,
            )
            pending.owner, pending.plan, pending.value, pending.base = owner, plan, None, None
            pending.reads = sum(getattr(t, "reads", 1) for t in plan.sources)
            pending.sources = set()
            for t in operands:
                pending.sources |= getattr(t, "sources", None) or {_address(t)}
            key = id(pending)
            owner.live[key] = weakref.ref(pending, lambda _: owner.live.pop(key, None))
            return pending

        @classmethod
        def __torch_dispatch__(
            cls, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None
        ) -> Any:
            # An op the bands did not take: its pending operands are computed first.
            def whole(value: Any) -> Any:
                return value.whole() if isinstance(value, cls) else value

            tree_map = torch.utils._pytree.tree_map
            return func(*tree_map(whole, args), **tree_map(whole, kwargs or {}))

        def rows(self, a: int, b: int) -> Any:
            if self.value is not None:
                return self.owner._take(self.value, self.dim() - 2, a, b)
            plan = self.plan
            lo, hi = a // plan.align * plan.align, -(-b // plan.align) * plan.align
            hi = min(hi, int(self.shape[plan.axis]))
            band = plan.band(lo, hi)
            return band if (lo, hi) == (a, b) else band.narrow(plan.axis, a - lo, b - a)

        def based(self, base: Any, func: Any, args: tuple[Any, ...], kwargs: Any) -> Any:
            self.base = (base, func, args, kwargs)
            return self

        def data_ptr(self) -> int:
            return int(self.whole().data_ptr())

        def untyped_storage(self) -> Any:
            return self.whole().untyped_storage()

        @property
        def __cuda_array_interface__(self) -> Any:  # type: ignore[override]
            return self.whole().__cuda_array_interface__

        def whole(self) -> Any:
            if self.value is None:
                with torch.utils._python_dispatch._disable_current_modes():
                    if self.base is not None:  # a view: the view of its base, computed
                        base, func, args, kwargs = self.base
                        self.value = func(base.whole(), *args[1:], **kwargs)
                    else:
                        owner, plan = self.owner, self.plan
                        self.value = owner._banded(plan, plan.sources)
                self.plan = self.base = None
                self.reads, self.sources = 1, {_address(self.value)}
                self.owner.live.pop(id(self), None)
            return self.value

    return Rows


def _band_rows(out: Any, axis: int, align: int) -> int:
    """Rows of `out` in a band of about `BAND` elements: a cache holds a chain's temporaries."""
    per_row = max(int(out.numel()) // max(int(out.shape[axis]), 1), 1)
    return max(align, BAND // per_row // align * align)


def _arg(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], name: str) -> Any:
    """Argument `name` of `func` as called, else its schema default."""
    for index, argument in enumerate(func._schema.arguments):
        if argument.name == name:
            if index < len(args):
                return args[index]
            if name in kwargs:
                return kwargs[name]
            return argument.default_value if argument.has_default_value() else None
    return kwargs.get(name)


def _written(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    """The tensors an op writes into (`Tensor(a!)` in its schema)."""
    return [
        args[i] if i < len(args) else kwargs.get(argument.name)
        for i, argument in enumerate(func._schema.arguments)
        if argument.alias_info is not None and argument.alias_info.is_write
    ]


def _address(value: Any) -> int:
    """The address of a tensor's storage; -1 for one with none, or pending."""
    if getattr(value, "pending", False):
        return -1
    try:
        return int(value.untyped_storage().data_ptr())
    except (AttributeError, RuntimeError):
        return -1


def _conv_spec(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...] | None:
    """(input, weight, bias, stride, padding, dilation, groups), each per spatial dim; None for
    a transposed convolution or a padding given by name."""
    w = args[1]
    n = w.dim() - 2
    padding = _arg(func, args, kwargs, "padding")
    if _arg(func, args, kwargs, "transposed") or isinstance(padding, str):
        return None

    def each(value: Any, default: int) -> list[int]:
        value = default if value is None else value
        values = [int(v) for v in value] if isinstance(value, (list, tuple)) else [int(value)]
        return values * n if len(values) == 1 else values

    return (
        args[0],
        w,
        _arg(func, args, kwargs, "bias"),
        each(_arg(func, args, kwargs, "stride"), 1),
        each(padding, 0),
        each(_arg(func, args, kwargs, "dilation"), 1),
        int(_arg(func, args, kwargs, "groups") or 1),
    )


def _fold(
    zeros: _Zeros, w: Any, dilation: list[int], padding: list[int], out: Any
) -> tuple[Any, Any, list[int], list[int], bool] | None:
    """The convolution of `zeros.padded` as one of its source: the padding moved into the
    convolution's own, and along a dim with one output the taps that meet only zeros
    dropped (and whether any did). None when the padding cannot all move."""
    x, pads = zeros.source, zeros.pads
    n = w.dim() - 2
    if x.dim() != w.dim() or any(pair != (0, 0) for pair in pads[:2]):
        return None
    front = [pads[2 + i][0] + padding[i] for i in range(n)]
    back = [pads[2 + i][1] + padding[i] for i in range(n)]
    dropped = False
    for i in range(n):
        if front[i] == back[i] or out.shape[2 + i] != 1:
            continue
        k, d, size = int(w.shape[2 + i]), dilation[i], int(x.shape[2 + i])
        taps = [t for t in range(k) if 0 <= t * d - front[i] < size]
        # the taps left must read the whole source along this dim, one element each
        if len(taps) != size or (size > 1 and d != 1) or taps[0] * d != front[i]:
            return None
        w = w.narrow(2 + i, taps[0], len(taps))
        front[i] = back[i] = 0
        dropped = True
    if front != back:
        return None
    return x, w, front, back, dropped


def _spans(value: Any, out: Any, axis: int) -> bool:
    dim = value.dim() - (out.dim() - axis)
    return dim >= 0 and bool(value.shape[dim] == out.shape[axis])


def _tensors(value: Any, kind: type) -> list[Tensor]:
    found: list[Tensor] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, kind):
            found.append(item)
    return found


def _span(t: Any) -> int:
    """Elements from a tensor's first to its last, past its strides."""
    if not t.numel():
        return 0
    return 1 + sum(int(n - 1) * int(s) for n, s in zip(t.shape, t.stride(), strict=True))


def _bytes(t: Any) -> int:
    return int(t.numel()) * int(t.element_size())
