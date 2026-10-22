"""The out-of-memory catch (varena-slim §1): Runtime runs torch, varena decides memory.

An op out of device memory asks the request (`r.oom(need, site)`) and does what Rust answers:
Retry runs it again, Cut(level) runs the innermost cut around it at that level (else the op in
its own bands), Park moves idle tensors to page-locked host memory, Exhausted raises. Python
keeps no rung order, room arithmetic, retry count or size memo: progress is Rust's to judge.

The catch wraps blocks, calling each one's class forward as it is at the call (a class swap
after install, such as HiDiffusion's at a request's first step, runs). A block runs bare unless
its first run wrote a tensor that existed before it, or it once ran short; bare and out of
memory, it gets back its generators, its declared state and the attributes its first run showed
it changes (step counters, a shared dict's items), and runs again under the per-op layer, which
catches each op at itself (a dispatch mode on every op costs CPU, so only those blocks). A
block that joins a collective is never run again on one GPU alone: it runs bare in a pass whose
room was measured (`r.measured`), and out of memory there it fails typed and runs under the
layer at those shapes from then on. Ops outside every block (a package's code between passes)
run under the request's layer, which no block's bare run sees.
"""

from __future__ import annotations

import ctypes
import functools
import gc
import mmap
import threading
import traceback
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol

import torch
from tensorfs.varena.torch import Next
from torch import Tensor, nn
from torch._ops import OpOverload
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)

from cozy_runtime.author._errors import Fault, fault

type Args = Sequence[object]
type Kwargs = Mapping[str, object]

#: ops other GPUs wait on: a block that runs one never runs again on this GPU alone
_COLLECTIVES = frozenset({"c10d", "_c10d_functional", "_c10d_functional_autograd"})
_PLAIN = (int, float, bool, str, bytes, type(None))
_MISSING = object()
#: set on an OOM a writing op raised: it may have changed its destination, so nothing replays it
_NO_REPLAY = "_cozy_no_replay"


class Request(Protocol):
    """varena's request handle, as the execution layer uses it."""

    def oom(
        self,
        need: int | None,
        site: str | None,
        collective: bool = False,
        finest: int = 0,
        shape: str = "",
    ) -> Next: ...
    def cut(self, site: str, shape: str = "") -> int: ...
    def measured(self) -> bool: ...
    def levels(self) -> list[tuple[str, int]]: ...
    def agree(self, levels: list[tuple[str, int]]) -> None: ...
    def parked(self, nbytes: int) -> None: ...
    def site(
        self, name: str, collective: bool = False, finest: int = 0
    ) -> AbstractContextManager[None]: ...


#: the request the running attempt's ops answer to (`serving`)
CURRENT: ContextVar[Request | None] = ContextVar("cozy_execution_request", default=None)


class Finer(Exception):
    """An op inside a cut ran out and Rust cut that cut finer: it runs again at `level`."""

    def __init__(self, scope: Scope, level: int) -> None:
        super().__init__(f"{scope.site}: cut to level {level}")
        self.scope, self.level = scope, level


@dataclass(eq=False, slots=True)
class Scope:
    """A cut in force: its site, its level and the finest level it has, the operands it reads
    (never parked while it runs), and how many ops of each kind its current piece ran (the
    sites of the ops inside it: the same in every piece, whatever the level). Across a
    `collective` one no cut goes finer mid-pass (its peers run the same pass): Rust answers it
    locally."""

    site: str
    finest: int
    level: int = 0
    collective: bool = False
    operands: object = ()
    ordinals: dict[str, int] = field(default_factory=dict)
    #: its operands' shape key: Rust keeps a level per (site, shape)
    shape: str = ""


@dataclass(eq=False, slots=True)
class _Run:
    """A block running: how many ops of each kind it ran under the layer (their cut sites)."""

    block: Block
    ordinals: dict[str, int] = field(default_factory=dict)

    def site(self, kind: str) -> str:
        n = self.ordinals[kind] = self.ordinals.get(kind, 0) + 1
        return f"{self.block.name}/{kind}#{n}"


class _Local(threading.local):
    def __init__(self) -> None:
        self.layer: Layer | None = None
        self.scopes: list[Scope] = []
        self.run: _Run | None = None
        #: the bytes the allocator last refused on this thread (its OOM observer)
        self.need: int | None = None


_local = _Local()
_observed: set[int] = set()


@contextmanager
def serving(request: Request) -> Iterator[None]:
    """The attempt's ops answer to `request` while it runs: a model's in its blocks, and the
    rest (a package's code between passes) under the request's layer."""
    if torch.cuda.is_initialized():
        observe(torch.cuda.current_device())
    token = CURRENT.set(request)
    try:
        with _layer():
            yield
    finally:
        CURRENT.reset(token)
        reclaim()


def observe(device: int) -> None:
    """Torch's allocator tells this thread the bytes each refused allocation asked for."""
    if device not in _observed:
        _observed.add(device)
        with torch.cuda.device(device):
            torch._C._cuda_attach_out_of_memory_observer(_refused)


def _refused(device: int, nbytes: int, allocated: int, free: int) -> None:
    _local.need = int(nbytes)


def site(kind: str) -> str | None:
    """The cut site of a call of `kind` in the running block (None outside one): numbered within
    the innermost cut's piece when one is running, so a level Rust keeps for it lands on the same
    call in every pass at any level of the cuts around it."""
    if _local.run is None:
        return None
    if not _local.scopes:
        return _local.run.site(kind)
    scope = _local.scopes[-1]
    n = scope.ordinals[kind] = scope.ordinals.get(kind, 0) + 1
    return f"{scope.site}/{kind}#{n}"


def piece() -> None:
    """A cut's next piece starts: the ops inside it are numbered from the start again."""
    if _local.scopes:
        _local.scopes[-1].ordinals.clear()


def shape_key(operands: object) -> str:
    """The shapes of the tensors in `operands`, as Rust keys a site's level by them."""
    return "|".join("x".join(map(str, t.shape)) for t in tensors(operands))


def short(exc: BaseException) -> bool:
    """An allocation failure the catch may answer: not one a writing op raised, nor in a
    capture (a graph cannot run an op twice)."""
    return fault(exc) is Fault.OOM and not getattr(exc, _NO_REPLAY, False) and not _capturing()


def _need() -> int | None:
    found, _local.need = _local.need, None
    return found


def _clear(exc: BaseException) -> None:
    """The failed attempt's frames go, and with them its tensors."""
    seen: set[int] = set()
    at: BaseException | None = exc
    while at is not None and id(at) not in seen:
        seen.add(id(at))
        if at.__traceback__ is not None:
            traceback.clear_frames(at.__traceback__)
        at = at.__cause__ or at.__context__


def _capturing() -> bool:
    return bool(torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing())


# ------------------------------------------------------------------------------ answers


def answer(
    request: Request, nbytes: int | None, own: Scope | None, operands: object, first: bool = False
) -> None:
    """Ask `request` about an OOM and do what it answers. Retry returns (run it again); Park
    moves idle tensors to the host, then returns. Cut goes to the innermost cut that can go
    finer (enclosing cuts first, then `own`; `own` first when `first`): `own`'s new level is
    set and returns, an enclosing cut's raises `Finer`. Exhausted raises varena's error."""
    order = (own, *reversed(_local.scopes)) if first else (*reversed(_local.scopes), own)
    scope = next(
        (s for s in order if s is not None and (s.collective or s.level < s.finest)), None
    )
    collective = scope is not None and scope.collective
    if scope is None:
        found = request.oom(nbytes, None)
    else:
        found = request.oom(nbytes, scope.site, collective, scope.finest, scope.shape)
    match found:
        case Next.Retry():
            return
        case Next.Cut(level=level) if scope is not None and not collective:
            if scope is not own:
                raise Finer(scope, level)
            scope.level = level
        case Next.Park(budget=budget):
            used = [operands, *(s.operands for s in _local.scopes)]
            request.parked(park(_device_of(operands), nbytes, budget, used))
        case Next.Exhausted(error=error):
            raise error
        case _:
            raise RuntimeError(f"varena answered {found!r} for {scope}")


def settle[T](run: Callable[[], T], own: Scope | None, operands: object, first: bool = False) -> T:
    """`run()` until it runs, each OOM answered (`answer`); `run` reads `own.level`."""
    request = CURRENT.get()
    while True:
        _local.need = None  # an OOM torch answered itself before this leaves no stale size
        try:
            return run()
        except Exception as exc:
            if request is None or not short(exc):
                raise
            nbytes = _need()
            _clear(exc)
        answer(request, nbytes, own, operands, first)


def cut[T](
    site: str,
    finest: int,
    run: Callable[[int], T],
    operands: object,
    shape: str | None = None,
    measured: bool = True,
) -> T:
    """`run(level)` for a call that cuts itself (FFN tokens, attention pieces, a package's
    split, a norm's or a conv's bands): at the level Rust holds for `site` at its `shape` (its
    operands' shapes when None), finer when it or an op inside it ran out and Rust cut it. Its
    peak is measured as the site's (`r.site`) when `measured`: a conv's, many and small within
    its pass, is not (Rust keeps the level it cut one to)."""
    request = CURRENT.get()
    if request is None:
        return run(0)
    key = shape_key(operands) if shape is None else shape
    scope = Scope(site, finest, request.cut(site, key), operands=operands, shape=key)
    _local.scopes.append(scope)

    def whole() -> T:
        scope.ordinals.clear()
        return run(scope.level)

    try:
        with request.site(site, False, finest) if measured else nullcontext():
            while True:
                try:
                    return settle(whole, scope, operands, first=True)
                except Finer as finer:
                    if finer.scope is not scope:
                        raise
                    scope.level = finer.level
                    _clear(finer)
    finally:
        _local.scopes.remove(scope)


@contextmanager
def collective(site: str, finest: int) -> Iterator[None]:
    """A collective's extent, `finest` its finest level: an OOM inside asks Rust as
    `collective` (local answers only, the finer level kept for the next pass's agreement), and
    no cut outside it goes finer. Its peak is measured as the site's."""
    request = CURRENT.get()
    scope = Scope(site, finest, 0, collective=True)
    _local.scopes.append(scope)
    for journal in _local.layer.journals if _local.layer is not None else ():
        journal.joined = True  # its block never runs again on this GPU alone
    try:
        with nullcontext() if request is None else request.site(site, True, finest):
            yield
    finally:
        _local.scopes.remove(scope)


def attempt[T](run: Callable[[], T], operands: object) -> T:
    """`run()` where no dispatch sees it (a Triton launch): out of memory it did not run, so it
    runs again as an op does. A block whose first run launches it writes what its operands are,
    as far as the layer can tell: one it did not make keeps the block from running bare."""
    for journal in _local.layer.journals if _local.layer is not None else ():
        journal.wrote |= any(p not in journal.made for p in _storages(operands))
    return settle(run, None, operands)


# ---------------------------------------------------------------------------------- park


#: storages a cache or a hold owns (KV slots, a graph's static inputs), by data address: park
#: never moves them while their tensor lives
_KEPT: dict[int, int] = {}
#: cudaHostRegisterPortable | cudaHostRegisterMapped: every device addresses it over the bus
_MAPPED = 0x1 | 0x2


def keep(*held: Tensor) -> None:
    """Tensors a cache or a hold owns: park never moves them while they live."""
    for tensor in held:
        at = _storage(tensor)
        if at:
            _KEPT[at] = _KEPT.get(at, 0) + 1
            weakref.finalize(tensor, _unkeep, at)


def _unkeep(at: int) -> None:
    if _KEPT.get(at, 0) <= 1:
        _KEPT.pop(at, None)
    else:
        _KEPT[at] -= 1


def park(device: torch.device | None, nbytes: int | None, budget: int, used: object) -> int:
    """Tensors this process holds in torch's own pool on `device` move to page-locked host
    memory in place, largest first, until `nbytes` moved (as many as fit when unsaid) within
    `budget`, the host headroom varena measured. Never one the failing op or a cut around it
    reads (`used`), one a cache or a hold owns (`keep`), nor one in a graph's pool. Each stays a
    CUDA tensor read over the bus, so whatever holds it is unchanged. Returns the bytes moved."""
    if device is None or device.type != "cuda":
        return 0
    reclaim()
    torch.cuda.synchronize(device)
    segments = [
        (s["address"], s["address"] + s["total_size"])
        for s in torch.cuda.memory_snapshot()
        if s["device"] == device.index and tuple(s["segment_pool_id"]) == (0, 0)
    ]
    read = set(map(_storage, tensors(used, depth=6)))
    found: dict[int, torch.UntypedStorage] = {}
    for value in gc.get_objects():
        if type(value) not in (Tensor, nn.Parameter) or value.device != device:
            continue
        at = _storage(value)
        if at and at not in read and at not in _KEPT and any(lo <= at < hi for lo, hi in segments):
            found.setdefault(at, value.untyped_storage())
    moved, goal = 0, budget if nbytes is None else min(nbytes, budget)
    for storage in sorted(found.values(), key=lambda s: -s.nbytes()):
        if moved >= goal:
            break
        if moved + storage.nbytes() > budget:
            continue
        host = torch.as_tensor(_Parked(storage.nbytes(), device), device=device)
        host.copy_(torch.empty(0, dtype=torch.uint8, device=device).set_(storage))
        storage._swap_data_ptr_(host.untyped_storage())  # its device bytes go with `host`
        moved += storage.nbytes()
    return moved


#: parked host blocks whose tensors are gone, until the device passes their last reader
_GONE: list[tuple[torch.cuda.Event, _Parked]] = []


class _Parked:
    """A parked tensor's bytes: anonymous host memory (never torch's host cache) page-locked
    and mapped, which the GPU addresses over the bus, kept by its tensor. When the tensor goes
    they wait for an event the device passes, then are unmapped."""

    def __init__(self, nbytes: int, device: torch.device) -> None:
        self.memory = mmap.mmap(-1, nbytes)
        self.view = ctypes.c_char.from_buffer(self.memory)
        self.address, self.device = ctypes.addressof(self.view), device
        _cuda(torch.cuda.cudart().cudaHostRegister(self.address, nbytes, _MAPPED))
        data = (self.address, False)
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": data,
            "version": 3,
        }

    def __del__(self) -> None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        _GONE.append((event, self))

    def free(self) -> None:
        _cuda(torch.cuda.cudart().cudaHostUnregister(self.address))
        del self.view
        self.memory.close()


def _cuda(status: object) -> None:
    if int(getattr(status, "value", status)):  # type: ignore[call-overload]
        raise RuntimeError(f"CUDA host registration failed: {status}")


def reclaim() -> None:
    """Parked blocks the device has passed go back."""
    waiting = []
    for event, parked in _GONE:
        if event.query():
            parked.free()
        else:
            waiting.append((event, parked))
    _GONE[:] = waiting


# -------------------------------------------------------------------------------- blocks


@dataclass(eq=False, slots=True)
class Block:
    """A block for the catch: its site name, the model root its declared `state` paths are
    under, and what its first run showed (`bare` None: not run yet): whether it may run bare,
    whether it joins a collective, the generators it draws from, the attributes it rebinds and the
    lists and dicts it changes in place, and the input shapes at which it ran out bare in a
    collective."""

    name: str
    root: nn.Module
    state: tuple[str, ...] = ()
    bare: bool | None = None
    joined: bool = False
    generators: tuple[torch.Generator, ...] = ()
    changes: tuple[tuple[object, str], ...] = ()
    contents: tuple[list[object] | dict[object, object], ...] = ()
    short: set[tuple[object, ...]] = field(default_factory=set)


def wrap(
    module: nn.Module,
    block: Block,
    around: Callable[[nn.Module, Args, Kwargs], AbstractContextManager[None]] | None = None,
) -> None:
    """`module`'s forward runs through the catch as `block`, inside `around(module, args,
    kwargs)`: a forward set on the instance as it is now, else its class's at each call."""
    forward = vars(module).get("forward") or current(module)

    @functools.wraps(module.forward)
    def caught(*args: object, **kwargs: object) -> object:
        with nullcontext() if around is None else around(module, args, kwargs):
            request = CURRENT.get()
            if request is None:
                return forward(*args, **kwargs)
            return run(block, module, forward, args, kwargs, request)

    module.forward = caught


def run[T](
    block: Block,
    module: nn.Module,
    forward: Callable[..., T],
    args: Args,
    kwargs: Kwargs,
    request: Request,
) -> T:
    """A block's forward: observed under the layer on its first run, then bare when that showed
    it may run again, else under the layer."""
    outer, _local.run = _local.run, _Run(block)
    try:
        if block.joined:
            shapes = _shapes((args, kwargs))
            if request.measured() and shapes not in block.short:
                return _collective(block, shapes, forward, args, kwargs)
        elif block.bare:
            return _bare(block, forward, args, kwargs)
        with _layer() as layer:
            if block.bare is not None:
                return forward(*args, **kwargs)
            journal = _Journal()
            before, short_before = _attributes(module), layer.short
            layer.journals.append(journal)
            try:
                out = forward(*args, **kwargs)
            finally:
                layer.journals.remove(journal)
        block.bare = not journal.wrote and layer.short == short_before
        block.joined, block.generators = journal.joined, tuple(journal.generators.values())
        block.changes, block.contents = _changed(module, before, _attributes(module))
        return out
    finally:
        _local.run = outer


def _bare[T](block: Block, forward: Callable[..., T], args: Args, kwargs: Kwargs) -> T:
    states = [(g, g.get_state()) for g in block.generators]
    put = _kept(block.root, block.state, block)
    try:
        with _unlayered():
            return forward(*args, **kwargs)
    except Exception as exc:
        if not short(exc):
            raise
        _clear(exc)
    for generator, state in states:
        generator.set_state(state)
    put()
    block.bare = False  # it ran short once: under the layer from now on
    _local.run = _Run(block)
    with _layer():
        return forward(*args, **kwargs)


def _collective[T](
    block: Block, shapes: tuple[object, ...], forward: Callable[..., T], args: Args, kwargs: Kwargs
) -> T:
    """A collective block in a measured pass: bare, its peers waiting on it in its collectives,
    so it never runs again here. Out of memory (a misstated room) the request fails typed, and
    at these shapes the block runs under the layer from now on."""
    try:
        with _unlayered():
            return forward(*args, **kwargs)
    except Exception as exc:
        if short(exc):
            block.short.add(shapes)
        raise


@contextmanager
def _unlayered() -> Iterator[None]:
    """No layer while a bare block runs: the request's (or an outer block's) is set aside."""
    layer = _local.layer
    pop = layer is not None and _get_current_dispatch_mode() is layer  # else it runs under it
    try:
        with _pop_mode_temporarily() if pop else nullcontext():
            _local.layer = None if pop else layer
            yield
    finally:
        _local.layer = layer


def _shapes(value: object) -> tuple[object, ...]:
    return tuple((tuple(t.shape), t.dtype) for t in tensors(value))


@contextmanager
def _layer() -> Iterator[Layer]:
    """This thread's layer, pushed if there is none."""
    if _local.layer is not None:
        yield _local.layer
        return
    layer = _local.layer = Layer()
    try:
        with layer:
            yield layer
    finally:
        _local.layer = None


def current(module: nn.Module) -> Callable[..., object]:
    """`module`'s class forward as it is at each call."""
    return lambda *args, **kwargs: type(module).forward(module, *args, **kwargs)


def _kept(root: nn.Module, paths: Sequence[str], block: Block | None = None) -> Callable[[], None]:
    """Declared state and what `block`'s run changes (`_changed`) as they are now, and how to
    put them back (lists and dicts by their items)."""
    owners: list[tuple[object, str]] = []
    for path in paths:
        where, _, attr = path.rpartition(".")
        owners.append((functools.reduce(getattr, where.split("."), root) if where else root, attr))
    taken: list[tuple[object, str, object, object]] = []
    for owner, attr in (*owners, *(block.changes if block else ())):
        value = getattr(owner, attr, _MISSING)
        taken.append((owner, attr, value, value.copy() if isinstance(value, list | dict) else 0))
    filled = [(held, held.copy()) for held in (block.contents if block else ())]

    def put() -> None:
        for container, saved in filled:
            if isinstance(container, dict) and isinstance(saved, dict):
                container.clear()
                container.update(saved)
            elif isinstance(container, list) and isinstance(saved, list):
                container[:] = saved
        for owner, attr, value, items in taken:
            if value is _MISSING:
                vars(owner).pop(attr, None)
                continue
            if isinstance(value, dict) and isinstance(items, dict):
                value.clear()
                value.update(items)
            elif isinstance(value, list) and isinstance(items, list):
                value[:] = items
            setattr(owner, attr, value)

    return put


def _changed(
    module: nn.Module,
    before: Mapping[tuple[int, str], object],
    after: Mapping[tuple[int, str], object],
) -> tuple[tuple[tuple[object, str], ...], tuple[list[object] | dict[object, object], ...]]:
    """What `_attributes` saw change between `before` and `after`: the attributes rebound, and
    the lists and dicts whose items changed in place, each once however many attributes hold it
    (HiDiffusion's one `info` dict on every module)."""
    owners = {id(m): m for m in module.modules()}
    rebound: list[tuple[object, str]] = []
    contents: dict[int, list[object] | dict[object, object]] = {}
    for key in before.keys() | after.keys():
        was = before.get(key, _MISSING)
        if key[0] not in owners or was == after.get(key, _MISSING):
            continue
        owner = owners[key[0]]
        value = getattr(owner, key[1], _MISSING)
        if isinstance(value, list | dict) and isinstance(was, tuple) and was[0] == id(value):
            contents[id(value)] = value
        else:
            rebound.append((owner, key[1]))
    return tuple(rebound), tuple(contents.values())


def _attributes(module: nn.Module) -> dict[tuple[int, str], object]:
    """A module tree's plain attributes, tensors held (by identity), public lists and dicts
    (their items by identity): what a forward may change that no op shows."""
    found: dict[tuple[int, str], object] = {}
    for m in module.modules():
        for name, value in vars(m).items():
            if isinstance(value, _PLAIN):
                found[(id(m), name)] = (type(value), value)
            elif isinstance(value, Tensor):
                found[(id(m), name)] = id(value)
            elif not name.startswith("_") and isinstance(value, list | dict):
                items = value.values() if isinstance(value, dict) else value
                found[(id(m), name)] = (id(value), tuple(map(id, items)))
    return found


# ---------------------------------------------------------------------------- the layer


class Banded(Protocol):
    """An op as bands (`cuts.Plan`): its finest level, and itself at `own.level` (finer while a
    band runs out and Rust cuts `own`)."""

    @property
    def finest(self) -> int: ...

    def __call__(self, own: Scope) -> Tensor: ...


#: what an op that runs out may become: its result in bands, else None
type Rule = Callable[[OpOverload, Args, Kwargs], Banded | None]
#: by op packet (`cuts`)
RULES: dict[object, Rule] = {}


@functools.cache
def _trait(func: OpOverload) -> tuple[bool, bool, bool, Rule | None]:
    """Whether the op writes an argument, draws from a generator and makes fresh results, and
    its band rule."""
    schema = func._schema
    writes = any(a.alias_info is not None and a.alias_info.is_write for a in schema.arguments)
    seeded = torch.Tag.nondeterministic_seeded in func.tags
    fresh = all(r.alias_info is None for r in schema.returns)
    return writes, seeded, fresh, RULES.get(func.overloadpacket)


@dataclass(eq=False, slots=True)
class _Journal:
    """A block's first run: the storages its ops made, whether one wrote into any other (a
    storage alive then is no freed address its ops made again), whether it joined a
    collective, and the generators its seeded ops drew from."""

    made: set[int] = field(default_factory=set)
    wrote: bool = False
    joined: bool = False
    generators: dict[int, torch.Generator] = field(default_factory=dict)

    def before(self, func: OpOverload, args: Args, kwargs: Kwargs) -> None:
        writes, seeded, _, _ = _trait(func)
        self.joined |= func.namespace in _COLLECTIVES
        if writes and not self.wrote:
            schema = func._schema.arguments
            written = [
                args[i] if i < len(args) else kwargs.get(a.name)
                for i, a in enumerate(schema)
                if a.alias_info is not None and a.alias_info.is_write
            ]
            self.wrote = any(p not in self.made for p in _storages(written))
        if seeded:
            self.generators.update((id(g), g) for g in _generators(func, args, kwargs))

    def after(self, func: OpOverload, out: object) -> None:
        if _trait(func)[2]:
            self.made.update(_storages(out))


class Layer(TorchDispatchMode):  # type: ignore[misc]
    """Every op, caught at itself."""

    def __init__(self) -> None:
        super().__init__()
        self.journals: list[_Journal] = []
        #: OOMs it caught: a block that ran short stays under the layer
        self.short = 0

    @classmethod
    def _should_skip_dynamo(cls) -> bool:
        return False  # nothing compiles here: no Dynamo wrapper around every op

    def __torch_dispatch__(
        self, func: OpOverload, types: object, args: Args = (), kwargs: Kwargs | None = None
    ) -> object:
        kwargs = kwargs or {}
        for journal in self.journals:
            journal.before(func, args, kwargs)
        out = self._run(func, args, kwargs)
        for journal in self.journals:
            journal.after(func, out)
        return out

    def _run(self, func: OpOverload, args: Args, kwargs: Kwargs) -> object:
        writes, draws, _, rule = _trait(func)
        request = CURRENT.get()
        if writes or func.namespace in _COLLECTIVES or request is None:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                # A failed resize/out/in-place op may have changed its destination, and a
                # collective its peers: nothing around it runs it again.
                if fault(exc) is Fault.OOM:
                    setattr(exc, _NO_REPLAY, True)
                raise
        own: Scope | None = None
        named = site(func.overloadpacket.__name__) if rule is not None else None
        if named is not None:
            key = shape_key((args, kwargs))
            own = Scope(named, 0, request.cut(named, key), shape=key)
        bands: list[Banded | None] = []  # the op's bands, made once, when first needed
        states = [(g, g.get_state()) for g in _generators(func, args, kwargs)] if draws else []

        def again() -> object:
            if own is not None and own.level:
                if not bands:
                    bands.append(rule(func, args, kwargs))  # type: ignore[misc]
                if bands[0] is not None:
                    return bands[0](own)
            for generator, state in states:
                generator.set_state(state)
            return func(*args, **kwargs)

        try:
            return again()
        except Exception as exc:
            if not short(exc):
                raise
            nbytes = _need()
            _clear(exc)
        self.short += 1
        if own is not None:
            if not bands:
                bands.append(rule(func, args, kwargs))  # type: ignore[misc]
            own.finest = 0 if bands[0] is None else bands[0].finest
        answer(request, nbytes, own, (args, kwargs))
        return settle(again, own, (args, kwargs))


def tensors(value: object, depth: int = 3) -> list[Tensor]:
    """The tensors in `value`, `depth` levels of sequences and mappings deep."""
    if isinstance(value, Tensor):
        return [value]
    if depth and isinstance(value, list | tuple | Mapping):
        items = value.values() if isinstance(value, Mapping) else value
        return [t for item in items for t in tensors(item, depth - 1)]
    return []


def _storages(value: object, depth: int = 3) -> set[int]:
    """The data addresses of the storages of the tensors in `value` (`tensors`)."""
    return {_storage(t) for t in tensors(value, depth)}


def _storage(t: Tensor) -> int:
    try:
        return int(t.untyped_storage().data_ptr())
    except RuntimeError:  # a wrapper with no storage (a weight placeholder)
        return 0


def _device_of(operands: object) -> torch.device | None:
    return next((t.device for t in tensors(operands)), None)


def _generators(func: OpOverload, args: Args, kwargs: Kwargs) -> list[torch.Generator]:
    """The generators a seeded op draws from: its own, else its device's default."""
    own = [v for v in (*args, *kwargs.values()) if isinstance(v, torch.Generator)]
    if own:
        return own
    named = kwargs.get("device")
    device = torch.device(named) if isinstance(named, str | torch.device) else None
    device = device or next((t.device for t in tensors((args, kwargs))), torch.device("cpu"))
    if device.type != "cuda":
        return [torch.default_generator]
    index = device.index if device.index is not None else torch.cuda.current_device()
    return [torch.cuda.default_generators[index]]
