"""Pure-site cuts and parking over Varena's allocation recovery.

The allocator makes room without rerunning Python. Only explicitly pure cut sites may repeat
with smaller pieces after exhaustion. Model blocks and package code always execute once;
their exceptions escape to the request boundary. No dispatch mode intercepts arbitrary ops.
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

from cozy_runtime.author._errors import Fault, fault

type Args = Sequence[object]
type Kwargs = Mapping[str, object]


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


class _Local(threading.local):
    def __init__(self) -> None:
        self.scopes: list[Scope] = []
        self.named: str | None = None
        self.ordinals: dict[str, int] = {}
        #: the bytes the allocator last refused on this thread (its OOM observer)
        self.need: int | None = None


_local = _Local()
_observed: set[int] = set()


@contextmanager
def serving(request: Request) -> Iterator[None]:
    """The attempt's pure cut sites answer to `request`; package code is never retried."""
    if torch.cuda.is_initialized():
        observe(torch.cuda.current_device())
    token = CURRENT.set(request)
    try:
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
    """A known cut's stable name under the current model block."""
    if _local.scopes:
        scope = _local.scopes[-1]
        n = scope.ordinals[kind] = scope.ordinals.get(kind, 0) + 1
        return f"{scope.site}/{kind}#{n}"
    if _local.named is None:
        return None
    n = _local.ordinals[kind] = _local.ordinals.get(kind, 0) + 1
    return f"{_local.named}/{kind}#{n}"


def piece() -> None:
    """A cut's next piece starts: the ops inside it are numbered from the start again."""
    if _local.scopes:
        _local.scopes[-1].ordinals.clear()


def shape_key(operands: object) -> str:
    """The shapes of the tensors in `operands`, as Rust keys a site's level by them."""
    return "|".join("x".join(map(str, t.shape)) for t in tensors(operands))


def short(exc: BaseException) -> bool:
    """A pure site's allocation failure outside capture."""
    return fault(exc) is Fault.OOM and not _capturing()


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
    scope = next((s for s in order if s is not None and (s.collective or s.level < s.finest)), None)
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
    if request is None or (own is None and not _local.scopes):
        return run()
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
    if _capturing():
        return run(scope.level)  # prepared variant; no metrics or recovery inside capture
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
    try:
        with nullcontext() if request is None else request.site(site, True, finest):
            yield
    finally:
        _local.scopes.remove(scope)


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


def wrap(
    module: nn.Module,
    name: str,
    around: Callable[[nn.Module, Args, Kwargs], AbstractContextManager[None]] | None = None,
) -> None:
    """Name a block's pure sites and enter pass agreement; invoke its forward exactly once."""
    forward = vars(module).get("forward") or current(module)

    @functools.wraps(module.forward)
    def named(*args: object, **kwargs: object) -> object:
        outer = _local.named, _local.ordinals
        _local.named, _local.ordinals = name, {}
        try:
            with nullcontext() if around is None else around(module, args, kwargs):
                return forward(*args, **kwargs)
        finally:
            _local.named, _local.ordinals = outer

    module.forward = named
    vars(module)["_cozy_execution_forward"] = named


def current(module: nn.Module) -> Callable[..., object]:
    """`module`'s class forward as it is at each call."""
    return lambda *args, **kwargs: type(module).forward(module, *args, **kwargs)


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
