"""A block's rows in one kernel a call where its own code is written that way (`fused_rows`),
for any model: nothing is declared and no class or function is named.

Norms, and the scale and shift a block applies after one:

    normed = norm(x)                       # an RMS norm over the last axis
    normed = normed * (1 + scale) + shift  # the block's own lines, as written

A module the row cut takes for a norm (`cuts.norm_like`) that states a float `eps` and holds at
most one vector (its weight) may be such a norm (`_norm`). Its first call runs its
own forward and the block's ops as written, which serve that call; the kernel's rows of a piece of
the same input, scale and shift are compared with them on the card, with no wait, under each law
its weight may follow (as given, or zero-centred: `* (1 + weight)`), and the kernel serves from
the first call after the comparison lands, under the law where every value compared was within
rounding's allowance (`fused_rows.bound`). A norm whose forward is one kernel already (it makes no
temporary the size of its output), or whose rows are no law's, is left as written, for good, with
one line.

A verdict is its law's, not its instance's: the class's code (its file's digest) and configuration,
the rows' dtype and width, the card and the Runtime build (`_key`). It is kept in the machine's
kernel store beside the compiled kernels, so another instance, or a later process, recalls it and
runs no comparison: a norm recalled as served still runs its first call as written, which shows
that its block reads its rows through torch, and serves the kernel from the next.

A proven norm returns its rows not yet computed (`Rows`). The block's own `* table` then
`+ table` join them where the table broadcasts over the rows (`fused_rows.table`), and the add
computes them: one kernel reads `x` and writes the result. Anything else that reads them first
(a projection, a shape other than a scale's) computes them as they stand, the norm alone or with
its scale, and runs as written on the result. On its proving call a norm's rows hold the eager
values, so a reader that bypasses torch still reads the right ones; such a reader never computes
them, and a norm whose proving rows are gone uncomputed is left as written at its next call.

Rotations: a function an attention processor calls on its queries and keys with one `(cos, sin)`
pair of fp32 tables (`cuts.rows` hands each call here). Its first call of each shape runs as
written and serves; the kernel's rotation of a piece of the same input, in each layout the tables
could be read in (`fused_rows.turns`), is compared with it, and the first within rounding's
allowance serves that shape from then on, recalled by later processes as a norm's verdict is. A
call that left its input as it was proves nothing and is tried again at the next.

The kernels make no temporary, so they need only their output and are never cut; whatever does
not take them (another dtype, a narrow or unaligned row, autograd, a kernel not ready) runs the
forward or function as it was, under the cut it had.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import queue
import sys
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, cast

import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode

import cozy_runtime.internal.execution.catch as catch
import cozy_runtime.internal.execution.cuts as cuts
from cozy_runtime.accel.kernels import Launch
from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author._model import Building, Yielded
from cozy_runtime.internal import fused_rows, kernel_cache, verify

#: a module's site, on the module (None once it is left as written)
_SITE = "_cozy_fused_rows"
#: what a block may ask of rows not yet computed without computing them
_PROPERTIES = frozenset({"shape", "dtype", "device", "ndim", "is_cuda", "layout", "requires_grad"})
_METHODS = frozenset({
    "dim", "ndimension", "size", "numel", "nelement", "element_size", "get_device",
    "is_floating_point", "is_complex", "is_contiguous", "is_inference", "__len__",
})  # fmt: skip
_SCALES = frozenset({"mul", "__mul__", "__rmul__"})
_SHIFTS = frozenset({"add", "__add__", "__radd__"})
#: what a rotation's arguments beside its input and tables may be (part of its call's form)
_SCALARS = (bool, int, float, str, type(None))
#: each card's launcher, once its kernels serve
_LAUNCHES: dict[int, Launch] = {}
_SAID: set[str] = set()


class _Call(Protocol):
    """A function or forward, any arguments."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


def _say(text: str) -> None:
    if text not in _SAID:
        _SAID.add(text)
        print(f"cozy-runtime: {text}", file=sys.stderr)


def _launch(device: torch.device) -> Launch | None:
    """`device`'s kernels once they serve (`fused_rows.serve`), else None."""
    index = device.index or 0
    launch = _LAUNCHES.get(index)
    if launch is None:
        launch = fused_rows.serve(device)
        if launch is not None:
            _LAUNCHES[index] = launch
    return launch


@dataclass(slots=True)
class Check:
    """One law's first-call checks (`taken`): the sites the kernel serves after theirs, those left
    as written and the first reason, the worst difference seen as a share of rounding's allowance,
    the values compared and the checks' time on the call's path."""

    served: int = 0
    left: int = 0
    why: str = ""
    worst: float = 0.0
    values: int = 0
    ms: float = 0.0
    recalled: int = 0


#: each (kind, law)'s checks since the executor last took them for a run's record
_CHECKS: dict[tuple[str, str], Check] = {}


def _checked(
    kind: str, law: str, ms: float, off: float, values: int, why: str = "", recalled: bool = False
) -> None:
    check = _CHECKS.setdefault((kind, law), Check())
    if why:
        check.left += 1
        check.why = check.why or why
    else:
        check.served += 1
        check.worst = max(check.worst, off)
    check.values += values
    check.ms += ms
    check.recalled += int(recalled)


def _since(began: float) -> float:
    return (time.perf_counter() - began) * 1e3


def taken() -> list[tuple[str, str, Check]]:
    """The first-call checks since the last take, for the attempt's record (the executor's
    `kernel.fused_rows`): a check is said in the run whose call ran it, and one still on the card
    when its run ends in the run it lands in."""
    for site in list(_WAITING):
        site.landed()
    rows = [(kind, law, check) for (kind, law), check in sorted(_CHECKS.items())]
    _CHECKS.clear()
    return rows


#: each verdict this process found or recalled, by its key's digest: {"why": ""} where the kernel
#: serves it ({"turn": ...} for a rotation, its layout), else {"why": <why it runs as written>}
_VERDICTS: dict[str, dict[str, object]] = {}
#: the norms whose comparison is still on the card
_WAITING: weakref.WeakSet[_Site] = weakref.WeakSet()


def _jsonable(value: object) -> object:
    """`value` as a key's JSON: dtypes, floats and shapes said as strings and lists."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list | torch.Size):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, bool | int | str):
        return value
    return repr(value)


@functools.cache
def _source(of: object) -> str:
    """The sha256 of the file that defines `of` (a class or a function); "" where it has none."""
    try:
        path = inspect.getsourcefile(cast("type[object]", of))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else ""
    except (OSError, TypeError):
        return ""


def _key(kind: str, of: object, device: torch.device, **inputs: object) -> kernel_cache.Key | None:
    """A verdict's key: what decides it (`of`'s code by its file, the Runtime build, the card and
    `inputs`); None where `of` has no file to read."""
    source = _source(of)
    if not source:
        return None
    named = f"{getattr(of, '__module__', '')}.{getattr(of, '__qualname__', '')}"
    card = torch.cuda.get_device_capability(device)
    document = {"runtime": verify.RUNTIME, "kind": kind, "of": named, "source": source}
    shaped = cast("Mapping[str, object]", _jsonable({**document, "card": card, **inputs}))
    return kernel_cache.Key.of("fused-rows-verdict", shaped)


def _recalled(key: kernel_cache.Key | None) -> dict[str, object] | None:
    """`key`'s verdict as this process or the machine's store holds it (`_VERDICTS`); None where
    none was found yet."""
    if key is None:
        return None
    if key.digest in _VERDICTS:
        return _VERDICTS[key.digest]
    store = kernel_cache.configured()
    try:
        data = None if store is None else store.get(key)
        found = None if data is None else json.loads(data)
    except (OSError, ValueError, kernel_cache.KernelCacheRefusal):
        found = None
    if not isinstance(found, dict):
        return None
    _VERDICTS[key.digest] = verdict = cast("dict[str, object]", found)
    return verdict


def _remember(key: kernel_cache.Key | None, verdict: dict[str, object], off: float) -> None:
    """`key`'s verdict for this process at once, and for the machine's store off the call's path
    (`_keep`): a verdict not kept before the process ends is found again by the next one."""
    if key is None or key.digest in _VERDICTS:
        return
    _VERDICTS[key.digest] = verdict
    store = kernel_cache.configured()
    if store is None:
        return
    if not _KEEPING:
        _KEEPING.append(threading.Thread(target=_keep, name="fused-rows verdicts", daemon=True))
        _KEEPING[0].start()
    _KEPT.put((store, key, json.dumps(verdict).encode(), repr(off)))


#: verdicts on their way to the machine's store, and the thread that writes them
type _Kept = tuple[kernel_cache.Store, kernel_cache.Key, bytes, str]
_KEPT: queue.SimpleQueue[_Kept] = queue.SimpleQueue()
_KEEPING: list[threading.Thread] = []


def _keep() -> None:
    while True:
        store, key, data, off = _KEPT.get()
        try:
            if store.get(key) is None:
                store.put(key, data, {"worst": off})
        except (OSError, ValueError, kernel_cache.KernelCacheRefusal):
            pass


def _piece(rows: int, width: int, itemsize: int) -> int:
    """How many of a call's `rows` a first-call check compares, so all it holds stays under what
    runs unasked (`catch._SMALL`): a value's kernel and eager rows (`itemsize` each), up to two
    tables' entries with their int64 index, and the allowance's float32 temporaries (the value,
    what it carried, a factor and the terms of their sums: seven at most)."""
    held = 2 * itemsize + 2 * (itemsize + 8) + 7 * 4
    return min(rows, max(1, catch._SMALL // (held * width)))


def _failed(exc: Exception) -> str:
    """A kernel failure this path answers by running as written, said in one line; memory and the
    device are not its to answer, and raise."""
    if isinstance(exc, Building | Yielded) or fault(exc) is not Fault.OTHER:
        raise exc
    return f"its kernel failed: {type(exc).__name__}: {exc}"[:300]


# --------------------------------------------------------------------------- norms


class _Made(TorchDispatchMode):
    """The bytes of every fresh tensor the ops under it return."""

    made = 0

    def __torch_dispatch__(  # type: ignore[no-untyped-def]  # torch leaves it untyped
        self, func, types, args=(), kwargs=None
    ):
        out = func(*args, **(kwargs or {}))
        outs = out if isinstance(out, tuple | list) else (out,)
        for value, slot in zip(outs, func._schema.returns, strict=False):
            if isinstance(value, Tensor) and slot.alias_info is None:
                self.made += value.numel() * value.element_size()
        return out


@dataclass(slots=True)
class _Waiting:
    """A norm's comparison on the card: done once `event` is; `host` then holds its share of
    rounding's allowance under each of its site's `laws`, in order. `ms` the call's time launching
    it, `values` the values compared."""

    event: torch.cuda.Event
    host: Tensor
    key: kernel_cache.Key | None
    ms: float
    values: int


@dataclass(slots=True)
class _Pending:
    """A norm's rows of `x` not yet computed, with the `scale` and `shift` its block applied
    since. On the proving call `eager` holds the same rows as the module and the block's ops made
    them. `value` once computed."""

    site: _Site
    launch: Launch
    x: Tensor
    scale: Tensor | None = None
    shift: Tensor | None = None
    eager: Tensor | None = None
    value: Tensor | None = None

    def like(self) -> Tensor:
        """A tensor of the rows' shape, dtype and device."""
        return self.x if self.value is None else self.value

    def scaled(self, table: object) -> Tensor | None:
        """`rows * table`, still not computed; None where the kernel does not read `table`."""
        if self.value is not None or self.scale is not None or not _reads(self.x, table):
            return None
        scale = cast("Tensor", table)
        eager = None if self.eager is None else self.eager * scale
        return _rows(replace(self, scale=scale, eager=eager))

    def shifted(self, table: object) -> Tensor | None:
        """`rows * scale + table`, computed; None without a scale or where the kernel does not
        read `table`."""
        if self.value is not None or self.scale is None or not _reads(self.x, table):
            return None
        shift = cast("Tensor", table)
        eager = None if self.eager is None else self.eager + shift
        return replace(self, shift=shift, eager=eager).realize()

    def realize(self) -> Tensor:
        if self.value is None:
            self.value = self.site.rows(self)
            self.eager = None
        return self.value


def _reads(x: Tensor, table: object) -> bool:
    return (
        type(table) in (Tensor, nn.Parameter)
        and cast("Tensor", table).dtype == x.dtype
        and cast("Tensor", table).device == x.device
        and fused_rows.table(x, cast("Tensor", table)) is not None
    )


class Rows(Tensor):
    """A proven norm's rows before they are computed: a block's `* scale` and `+ shift` join
    them (`_Pending`), their shape and dtype are their input's, and any other read computes
    them once and runs on the result."""

    _cozy_pending: _Pending

    @classmethod
    def __torch_function__(  # type: ignore[no-untyped-def]  # torch leaves it untyped
        cls, func, types, args=(), kwargs=None
    ):
        name = getattr(func, "__name__", "")
        if not kwargs and args and type(args[0]) is Rows:
            pending = args[0]._cozy_pending
            owner = getattr(func, "__self__", None)
            if name == "__get__" and getattr(owner, "__name__", "") in _PROPERTIES:
                return func(pending.like())
            if name in _METHODS:
                return func(pending.like(), *args[1:])
        if not kwargs and len(args) == 2 and (name in _SCALES or name in _SHIFTS):
            rows, table = args if type(args[0]) is Rows else args[::-1]
            if type(rows) is Rows and type(table) is not Rows:
                pending = rows._cozy_pending
                joined = pending.scaled(table) if name in _SCALES else pending.shifted(table)
                if joined is not None:
                    return joined
        return func(*(_plain(v) for v in args), **{k: _plain(v) for k, v in (kwargs or {}).items()})


def _rows(pending: _Pending) -> Tensor:
    """`pending` as a tensor: over its eager rows on the proving call, else over nothing."""
    anchor = pending.eager if pending.eager is not None else pending.x.new_empty(0)
    made = anchor.as_subclass(Rows)
    made._cozy_pending = pending
    return made


def _plain(value: object) -> object:
    """`value` with every `Rows` in it computed (a list's or a tuple's too: `torch.cat`'s)."""
    if type(value) is Rows:
        return value._cozy_pending.realize()
    if type(value) in (tuple, list):
        items = [_plain(v) for v in cast("Sequence[object]", value)]
        return items if type(value) is list else tuple(items)
    return value


@dataclass(eq=False)
class _Site:
    """One norm module: its forward as it stood (`inner`: its cut, or its class's), the instance
    forward this one replaced, its epsilon and its weight's name. `asked`: the rows a proving call
    handed out, until they are computed; `waiting`: its comparison, until it lands."""

    module: nn.Module
    inner: catch.Forward
    own: catch.Forward | None
    eps: float
    weight_name: str | None
    #: where it is: its root's class and its path under it, part of its verdict's key
    path: str = ""
    proven: bool = False
    asked: weakref.ref[Tensor] | None = None
    left: bool = False
    waiting: _Waiting | None = None
    #: its first call served by a verdict found before: no comparison
    recall: bool = False
    #: its weight is zero-centred (`fused_rows.law`'s `centred`), as its comparison found
    centred: bool = False

    def weight(self) -> Tensor | None:
        return None if self.weight_name is None else getattr(self.module, self.weight_name)

    def fits(self, x: Tensor) -> bool:
        """Whether rows of `x`'s kind may be the kernel's at all (`fused_rows.fits`), its weight's
        kind too."""
        weight = self.weight()
        if not fused_rows.fits(x) or weight is None:
            return fused_rows.fits(x)
        same = weight.shape == x.shape[-1:] and weight.dtype == x.dtype
        return same and weight.device == x.device

    def takes(self, x: Tensor) -> bool:
        """Whether this call's rows are the kernel's now."""
        return not (
            not fused_rows.takes(x)
            or self.module.training
            or torch.is_grad_enabled()
            or torch.compiler.is_compiling()
            or torch.is_autocast_enabled(x.device.type)
        )

    def __call__(self, *args: object, **kwargs: object) -> object:
        x = args[0] if len(args) == 1 and not kwargs else None
        if self.left or not isinstance(x, Tensor) or type(x) is not Tensor:
            return self.inner(*args, **kwargs)
        if not self.fits(x):
            self.leave()  # a narrow or fp32 norm: no call of its takes the kernel
            return self.inner(x)
        if not self.takes(x):
            return self.inner(x)
        launch = _launch(x.device)
        if launch is None:
            return self.inner(x)
        if self.waiting is not None and not self.landed():
            return self.inner(x)  # its comparison is still on the card: as written until it lands
        if self.proven:
            return _rows(_Pending(self, launch, x))
        if vars(self.module).get(_SITE) is not self:
            return self.inner(x)  # its comparison just landed outside the law: left as written
        if self.asked is not None and self.asked() is None:
            self.left_checked(0.0, "its rows were read where they cannot wait", key=self.key(x))
            return self.inner(x)
        if torch.cuda.is_current_stream_capturing():
            return self.inner(x)  # a capture cannot hold the proof's comparison
        key = self.key(x)
        recalled = _recalled(key)
        if recalled is not None:
            why = str(recalled.get("why", ""))
            if why:
                self.leave(why)
                _checked("norm", _name(self.module), 0.0, 0.0, 0, why, recalled=True)
                return self.inner(x)
            # served before: this call runs as written, so its block's reads show they go
            # through torch (`asked`), and the kernel serves from the next
            whole = self.inner(x)
            if type(whole) is not Tensor or whole.shape != x.shape or whole.dtype != x.dtype:
                return whole
            self.recall, self.centred = True, recalled.get("centred") is True
            made = _rows(_Pending(self, launch, x, eager=whole))
            self.asked = weakref.ref(made)
            return made
        counted = _Made()  # type: ignore[no-untyped-call]  # TorchDispatchMode's __init__
        with counted:
            whole = self.inner(x)
        if not (
            type(whole) is Tensor
            and whole.shape == x.shape
            and whole.dtype == x.dtype
            and whole.device == x.device
        ):
            self.left_checked(0.0, "its result is not its input's rows", key=key)
            return whole
        if counted.made < 2 * whole.numel() * whole.element_size():
            # one kernel already: no temporary the size of its output (said once, in the record)
            self.leave()
            _checked("norm", _name(self.module), 0.0, 0.0, 0, "one kernel already")
            _remember(key, {"why": "one kernel already"}, 0.0)
            return whole
        made = _rows(_Pending(self, launch, x, eager=whole))
        self.asked = weakref.ref(made)
        return made

    def key(self, x: Tensor) -> kernel_cache.Key | None:
        """This norm's verdict's key: its class's code, configuration and place, its weight's
        kind, and the rows' dtype and width (`_key`)."""
        weight = self.weight()
        held = None if weight is None else (int(weight.shape[0]), weight.dtype)
        return _key(
            "norm", type(self.module), x.device, path=self.path, eps=self.eps, weight=held,
            config=cuts._config(self.module), dtype=x.dtype, width=int(x.shape[-1]),
        )  # fmt: skip

    def rows(self, pending: _Pending) -> Tensor:
        """The rows `pending` stands for. On the proving call, the eager ones, the kernel's
        compared with them."""
        x, scale, shift = pending.x, pending.scale, pending.shift
        if pending.eager is not None:
            if not self.proven and not self.left and self.waiting is None:
                self.asked = None
                if self.recall:
                    self.proven = True
                    _checked("norm", _name(self.module), 0.0, 0.0, 0, recalled=True)
                else:
                    self.compare(pending, pending.eager)
            return pending.eager
        if not self.left:
            try:
                tables = _tables(x, scale, shift)
                weight, centred = self.weight(), self.centred
                return fused_rows.launched(
                    pending.launch, x, self.eps, weight, *tables, centred=centred
                )
            except Exception as exc:
                self.leave(_failed(exc))
        out = cast("Tensor", self.inner(x))
        out = out if scale is None else out * scale
        return out if shift is None else out + shift

    def laws(self) -> tuple[bool, ...]:
        """The laws its rows may follow, each `fused_rows.law`'s `centred`: a weight as given, or
        zero-centred (`* (1 + weight)`); without a weight, the one."""
        return (False,) if self.weight_name is None else (False, True)

    def compare(self, pending: _Pending, want: Tensor) -> None:
        """Prove the kernel on this call: its rows of the middle of `pending.x`, a piece small
        enough to run unasked, within rounding's allowance of `want`'s under one of its `laws`.
        `want` came from the input as it was when the norm was called, so an input changed since
        fails here too. The comparison runs on the card and nothing waits for it: its share of
        each law's allowance is copied back behind it, and the norm's next call (or the run's end)
        takes it (`landed`)."""
        began = time.perf_counter()
        x, width = pending.x, int(pending.x.shape[-1])
        key = self.key(x)
        if want.shape != x.shape or want.dtype != x.dtype:
            why = "its block's scale or shift changes its rows' shape or dtype"
            self.left_checked(_since(began), why, key=key)
            return
        rows = x.numel() // width
        count = _piece(rows, width, x.element_size())
        at = (rows - count) // 2
        try:
            piece, weight = x.view(rows, width)[at : at + count], self.weight()
            expected = want.reshape(rows, width)[at : at + count]
            tables = _tables(x, pending.scale, pending.shift)
            scale, shift = (t if t is None else fused_rows.rows_of(*t, at, count) for t in tables)
            shares: list[Tensor] = []
            for centred in self.laws():  # one after another: one law's temporaries at a time
                got = fused_rows.launched(
                    pending.launch, piece, self.eps, weight, *tables, at, centred=centred
                )
                allowed = fused_rows.bound(piece, self.eps, weight, scale, shift, centred=centred)
                shares.append(fused_rows.share(got, expected, allowed))
                del got, allowed
            share = torch.stack(shares)
            host = torch.empty(share.shape, dtype=torch.float32, pin_memory=True)
            host.copy_(share, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        except Exception as exc:
            self.left_checked(_since(began), _failed(exc))
            return
        self.waiting = _Waiting(event, host, key, _since(began), count * width)
        _WAITING.add(self)

    def landed(self) -> bool:
        """Whether this norm's comparison has landed (`compare`), taking its verdict if so: the
        kernel serves from here on, or the norm is left as written; the verdict is kept."""
        waiting = self.waiting
        if waiting is None:
            return True
        if not waiting.event.query():
            return False
        self.waiting = None
        _WAITING.discard(self)
        shares = [float(v) for v in waiting.host.tolist()]
        found = next((i for i, off in enumerate(shares) if off <= 1), None)
        if found is not None:
            off, self.centred, self.proven = shares[found], self.laws()[found], True
            _checked("norm", _name(self.module), waiting.ms, off, waiting.values)
            _remember(waiting.key, {"why": "", "centred": self.centred}, off)
            law = " of (1 + weight)" if self.centred else ""
            _say(f"{_name(self.module)}: one kernel a call{law}, each proven on its first call")
        else:
            lie = f"{shares[0]:.3g}" + "".join(f" (zero-centred {v:.3g})" for v in shares[1:])
            why = f"its rows lie {lie} times rounding's allowance from the kernel's"
            self.left_checked(waiting.ms, why, waiting.values, waiting.key)
        return True

    def left_checked(
        self, ms: float, why: str, values: int = 0, key: kernel_cache.Key | None = None
    ) -> None:
        """Its check refused it: left as written, said in the record and once in a line, and
        kept as its verdict where `key` names it (a refusal of the call, not the law, is not)."""
        self.leave(why)
        _checked("norm", _name(self.module), ms, 0.0, values, why)
        _remember(key, {"why": why}, 0.0)

    def leave(self, why: str = "") -> None:
        """This module runs as written from here on: its forward goes back where it is still this
        one, and this one passes every call to it where something has wrapped it since."""
        if self.left:
            return
        self.left = True
        module = self.module
        vars(module)[_SITE] = None
        if vars(module).get("forward") is self:
            if self.own is None:
                del vars(module)["forward"]
            else:
                module.forward = self.own
        if why:
            _say(f"{_name(module)}: left as written ({why})")


def _tables(
    x: Tensor, scale: Tensor | None, shift: Tensor | None
) -> tuple[tuple[Tensor, fused_rows.Table] | None, tuple[Tensor, fused_rows.Table] | None]:
    """Each table with how the kernel reads it (`_reads` held when the block applied it)."""
    found = [
        None if t is None else (t, cast("fused_rows.Table", fused_rows.table(x, t)))
        for t in (scale, shift)
    ]
    return found[0], found[1]


def _name(module: nn.Module) -> str:
    return f"{type(module).__module__}.{type(module).__qualname__}"


def _norm(module: nn.Module) -> tuple[float, str | None] | None:
    """`module`'s epsilon and its weight's name where the kernel may serve it: a module the row
    cut takes for a norm (`cuts.norm_like`, whose tensors are vectors) that states a float `eps`
    and holds at most one (the weight), as wide as the kernel takes (`fused_rows.width`). Its
    first call says whether it is one (`_Site`)."""
    eps = getattr(module, "eps", None)
    if type(eps) is not float or cuts.norm_like(module) is None:
        return None
    own = [*module.named_parameters(recurse=False), *module.named_buffers(recurse=False)]
    if len(own) > 1 or (own and not fused_rows.width(int(own[0][1].shape[0]))):
        return None
    return eps, own[0][0] if own else None


def install(module: nn.Module) -> int:
    """Each norm-like leaf under `module` (`_norm`) proves the kernel on its first call and
    serves it after; its forward as it stands now (its cut) stays under it. Returns how many."""
    if fused_rows.TRITON_ABSENT:
        return 0
    count, root = 0, f"{type(module).__module__}.{type(module).__qualname__}"
    for path, sub in module.named_modules():
        found = None if _SITE in vars(sub) else _norm(sub)
        if found is None:
            continue
        own = cast("catch.Forward | None", vars(sub).get("forward"))
        site = _Site(sub, own or catch.current(sub), own, *found, path=f"{root}:{path}")
        vars(sub)[_SITE] = site
        sub.forward = catch.transparent(site)
        count += 1
    return count


# --------------------------------------------------------------------------- rotations

#: each rotation's call form's verdict: the layout its kernel serves, or None (it runs as written)
_TURNS: dict[tuple[object, ...], fused_rows.Turn | None] = {}


def rotation(
    original: _Call,
    eager: _Call,
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> object:
    """A call of `original`, a function an attention processor calls on its queries or keys:
    the rotation kernel's result where this shape of call proved it (`_turned`) or a verdict kept
    before says so (`_recalled`), else `eager(*args, **kwargs)`, its cut, which serves the
    proving call."""
    call = _turning(args, kwargs)
    if call is None:
        return eager(*args, **kwargs)
    x, cos, sin, form = call
    local, law = (id(original), *form), _function(original)
    launch = _launch(x.device)
    if launch is None:
        return eager(*args, **kwargs)
    key = None if local in _TURNS else _key("rotation", original, x.device, form=form)
    recalled = _recalled(key)
    if recalled is not None:
        _TURNS[local] = _turn(recalled)
        why = "" if _TURNS[local] is not None else str(recalled.get("why", "")) or "no layout"
        _checked("rotation", law, 0.0, 0.0, 0, why, recalled=True)
    known, turn = local in _TURNS, _TURNS.get(local)
    if known and turn is None:
        return eager(*args, **kwargs)
    if turn is not None:
        try:
            return fused_rows.turned(launch, x, cos, sin, turn)
        except Exception as exc:
            _TURNS[local] = None
            _say(f"{law}: left as written ({_failed(exc)})")
            return eager(*args, **kwargs)
    if torch.cuda.is_current_stream_capturing():
        return eager(*args, **kwargs)  # a capture cannot hold the proof's comparison
    want = eager(*args, **kwargs)
    began = time.perf_counter()
    if type(want) is not Tensor or want.shape != x.shape or want.dtype != x.dtype:
        why = "its result is not its input's shape and dtype"
        _TURNS[local] = None
        _checked("rotation", law, _since(began), 0.0, 0, why)
        _remember(key, {"why": why}, 0.0)
        return want
    try:
        proven, off, values = _turned(launch, x, cos, sin, want)
    except Exception as exc:
        _TURNS[local] = None
        why = _failed(exc)
        _checked("rotation", law, _since(began), 0.0, 0, why)
        _say(f"{law}: left as written ({why})")
        return want
    if proven is not False:
        _TURNS[local] = proven
        if proven is None:
            why = "no layout of its tables matches"
            _checked("rotation", law, _since(began), 0.0, values, why)
            _remember(key, {"why": why}, 0.0)
            _say(f"{law}: left as written ({why})")
        else:
            _checked("rotation", law, _since(began), off, values)
            _remember(key, {"turn": _jsonable(asdict(proven))}, off)
            _say(f"{law}: one kernel a call, proven on each shape's first call")
    return want


def _turn(verdict: Mapping[str, object]) -> fused_rows.Turn | None:
    """The layout a kept rotation verdict names, or None (it runs as written)."""
    held = verdict.get("turn")
    if not isinstance(held, dict):
        return None
    try:
        sizes, strides = (tuple(int(n) for n in held[k]) for k in ("sizes", "strides"))
        return fused_rows.Turn(
            int(held["head"]), bool(held["split"]), bool(held["wide"]),
            (sizes[0], sizes[1], sizes[2]), (strides[0], strides[1], strides[2]),
            int(held["column"]),
        )  # fmt: skip
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _turning(
    args: Sequence[object], kwargs: Mapping[str, object]
) -> tuple[Tensor, Tensor, Tensor, tuple[object, ...]] | None:
    """A call the rotation kernel may take: its input first, one `(cos, sin)` pair of tables,
    and scalars beside them; with its form (shapes, strides, dtypes and scalars) as its key."""
    x = args[0] if args else None
    if (
        not isinstance(x, Tensor)
        or type(x) is not Tensor
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not fused_rows.turns_take(x)
    ):
        return None
    pair: tuple[Tensor, Tensor] | None = None
    form: list[object] = [tuple(x.shape), x.dtype, tuple(kwargs)]
    for value in (*args[1:], *kwargs.values()):
        if isinstance(value, _SCALARS):
            form.append(value)
            continue
        items = cast("Sequence[object]", value) if type(value) in (tuple, list) else ()
        if pair is not None or len(items) != 2 or any(type(t) is not Tensor for t in items):
            return None
        cos, sin = cast("tuple[Tensor, Tensor]", tuple(items))
        if not fused_rows.tables_take(x, cos, sin):
            return None
        pair = cos, sin
        form.append((tuple(cos.shape), cos.stride()))
    if pair is None:
        return None
    return x, pair[0], pair[1], tuple(form)


def _turned(
    launch: Launch, x: Tensor, cos: Tensor, sin: Tensor, want: Tensor
) -> tuple[fused_rows.Turn | Literal[False] | None, float, int]:
    """The first layout (`fused_rows.turns`) whose rotation of the middle of `x`, a piece small
    enough to run unasked (`_piece`), is within rounding's allowance of `want`'s, with its
    difference as a share of that allowance and the values compared; None for none; False where
    the call left that piece as it was (no layout can be told from another there)."""
    info, compared = torch.finfo(x.dtype), 0
    for turn in fused_rows.turns(x, cos):
        rows = x.numel() // turn.head
        count = _piece(rows, turn.head, x.element_size())
        at = (rows - count) // 2
        piece = x.reshape(rows, turn.head)[at : at + count]
        expected = want.reshape(rows, turn.head)[at : at + count]
        unturned = expected.float().abs() * info.eps + info.tiny * info.eps
        if fused_rows.within(piece, expected, unturned) <= 1:
            return False, 0.0, 0
        got = fused_rows.turned(launch, piece, cos, sin, turn, at)
        allowed = fused_rows.turn_bound(piece, cos, sin, turn, expected, at)
        off, compared = fused_rows.within(got, expected, allowed), compared + count * turn.head
        if off <= 1:
            return turn, off, compared
    return None, 0.0, compared


def _function(original: _Call) -> str:
    return f"{original.__module__}.{getattr(original, '__qualname__', '')}"
