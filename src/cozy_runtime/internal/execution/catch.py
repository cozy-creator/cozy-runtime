"""The out-of-memory catch (varena-slim §1): Runtime runs torch, varena decides memory.

An op out of device memory asks the request (`r.oom(need, site)`) and does what Rust answers:
Retry runs it again, Cut(level) runs the innermost cut around it at that level (else the op in
its own bands), Park moves idle tensors to page-locked host memory, Exhausted raises. Python
keeps no rung order, room arithmetic, retry count or size memo: progress is Rust's to judge.

The catch wraps blocks. A block runs bare unless its first run wrote a tensor that existed
before it or changed its attributes, or it once ran short; bare and out of memory, it gets its
generators and declared state back and runs again under the per-op layer, which catches each op
at itself (a dispatch mode on every op costs CPU, so only those blocks). A block that joins a
collective is never run again on one GPU alone: it runs bare in a pass whose room was measured
(`r.measured`), and out of memory there it fails typed and runs under the layer at those shapes
from then on. Ops outside every block (a package's code between passes) run under the request's
layer, which no block's bare run sees.
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
        self, need: int | None, site: str | None, collective: bool = False, finest: int = 0
    ) -> Next: ...
    def cut(self, site: str) -> int: ...
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


@dataclass(eq=False, slots=True)
class _Run:
    """A block running: how many ops of each kind it ran under the layer (their cut sites) and
    how many ops in all (each op's index: the same in every run of the block)."""

    block: Block
    ordinals: dict[str, int] = field(default_factory=dict)
    ops: int = 0

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
        found = request.oom(nbytes, scope.site, collective, scope.finest)
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


def cut[T](site: str, finest: int, run: Callable[[int], T], operands: object) -> T:
    """`run(level)` for a call that cuts itself (FFN tokens, attention pieces, a package's
    split, a norm's bands): at the level Rust holds for `site`, finer when it or an op inside it
    ran out and Rust cut it. Its peak is measured as the site's (`r.site`)."""
    request = CURRENT.get()
    if request is None:
        return run(0)
    scope = Scope(site, finest, request.cut(site), operands=operands)
    _local.scopes.append(scope)

    def whole() -> T:
        scope.ordinals.clear()
        return run(scope.level)

    try:
        with request.site(site, False, finest):
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
        journal.wrote |= any(p not in journal.made for p in map(_storage, tensors(operands)))
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
    whether it joins a collective, the generators it draws from, the ops whose first input is
    dead after them, and the input shapes at which it ran out bare in a collective."""

    name: str
    root: nn.Module
    state: tuple[str, ...] = ()
    bare: bool | None = None
    joined: bool = False
    generators: tuple[torch.Generator, ...] = ()
    dead: frozenset[int] = frozenset()
    short: set[tuple[object, ...]] = field(default_factory=set)


def wrap(
    module: nn.Module,
    block: Block,
    around: Callable[[nn.Module, Args, Kwargs], AbstractContextManager[None]] | None = None,
) -> None:
    """`module`'s forward runs through the catch as `block`, inside `around(module, args,
    kwargs)`."""
    forward = module.forward

    @functools.wraps(forward)
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
            journal = _Journal(_local.run)
            before, short_before = _attributes(module), layer.short
            layer.journals.append(journal)
            try:
                out = forward(*args, **kwargs)
            finally:
                layer.journals.remove(journal)
        same = before == _attributes(module)
        block.bare = not journal.wrote and same and layer.short == short_before
        block.joined, block.generators = journal.joined, tuple(journal.generators.values())
        block.dead = journal.dead((out, args, kwargs, _held(module)))
        return out
    finally:
        _local.run = outer


def _bare[T](block: Block, forward: Callable[..., T], args: Args, kwargs: Kwargs) -> T:
    states = [(g, g.get_state()) for g in block.generators]
    put = _kept(block.root, block.state)
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


def _held(module: nn.Module) -> list[object]:
    """The tensors a module tree holds as attributes after a run, lists and dicts included."""
    return [v for m in module.modules() for v in vars(m).values()]


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


def _kept(root: nn.Module, paths: Sequence[str]) -> Callable[[], None]:
    """Declared state as it is now, and how to put it back (lists and dicts by their items)."""
    taken: list[tuple[object, str, object, object]] = []
    for path in paths:
        where, _, attr = path.rpartition(".")
        owner = functools.reduce(getattr, where.split("."), root) if where else root
        value = getattr(owner, attr, _MISSING)
        taken.append((owner, attr, value, value.copy() if isinstance(value, list | dict) else 0))

    def put() -> None:
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


#: elementwise ops with an in-place form: run in place when their input is dead after them
_IN_PLACE = {torch.ops.aten.silu.default: torch.ops.aten.silu_.default}
_IN_PLACE[torch.ops.aten.gelu.default] = torch.ops.aten.gelu_.default


@dataclass(eq=False, slots=True)
class _Journal:
    """A block's first run (`run`): the storages its ops made, whether one wrote into any other,
    whether it joined a collective, the generators its seeded ops drew from, and each storage's
    last read, for the in-place ops whose input is dead after them."""

    run: _Run
    made: set[int] = field(default_factory=set)
    wrote: bool = False
    joined: bool = False
    generators: dict[int, torch.Generator] = field(default_factory=dict)
    #: ops it saw, the last op that read each storage, and its own in-place candidates: the
    #: op it saw as which, its index in its run, and the storage of its input
    seen: int = 0
    last: dict[int, int] = field(default_factory=dict)
    candidates: list[tuple[int, int, int]] = field(default_factory=list)

    def before(self, func: OpOverload, args: Args, kwargs: Kwargs, index: int) -> None:
        writes, seeded, _, _ = _trait(func)
        self.joined |= func.namespace in _COLLECTIVES
        if writes and not self.wrote:
            schema = func._schema.arguments
            written = [
                args[i] if i < len(args) else kwargs.get(a.name)
                for i, a in enumerate(schema)
                if a.alias_info is not None and a.alias_info.is_write
            ]
            self.wrote = any(p not in self.made for p in map(_storage, tensors(written)))
        if seeded:
            self.generators.update((id(g), g) for g in _generators(func, args, kwargs))
        self.seen += 1
        self.last.update((p, self.seen) for p in map(_storage, tensors((args, kwargs))))
        if func in _IN_PLACE and _local.run is self.run and args:
            self.candidates.append((self.seen, index, _storage(tensors(args[0])[0])))

    def after(self, func: OpOverload, out: object) -> None:
        if _trait(func)[2]:
            self.made.update(map(_storage, tensors(out)))

    def dead(self, escaped: object) -> frozenset[int]:
        """The in-place candidates whose input the block made and nothing read after them, nor
        holds once it returns (its result, its arguments, its modules' attributes)."""
        kept = set(map(_storage, tensors(escaped, depth=4)))
        return frozenset(
            index
            for seen, index, p in self.candidates
            if p and p in self.made and self.last.get(p) == seen and p not in kept
        )


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
        run = _local.run
        index = -1 if run is None else run.ops
        if run is not None:
            run.ops += 1
        for journal in self.journals:
            journal.before(func, args, kwargs, index)
        out = self._run(func, args, kwargs, index)
        for journal in self.journals:
            journal.after(func, out)
        return out

    def _run(self, func: OpOverload, args: Args, kwargs: Kwargs, index: int) -> object:
        writes, draws, _, rule = _trait(func)
        request = CURRENT.get()
        run = _local.run
        if (
            run is not None
            and func in _IN_PLACE
            and index in run.block.dead
            and not _local.scopes  # no cut around it runs again and reads its input
            and not torch.is_grad_enabled()
        ):
            in_place = _IN_PLACE[func]
            in_place(args[0], *args[1:], **kwargs)
            return args[0]
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
            own = Scope(named, 0, request.cut(named))
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
