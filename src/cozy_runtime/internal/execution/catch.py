"""Pure-site cuts over the allocator's own recovery (varena-slim step 1).

The allocator hook makes room (varena's make_room) without rerunning Python; when it cannot,
torch raises its OOM. Only an explicitly pure cut site runs again, one level finer, and the
process keeps each (site, shape)'s level (its memo) for every later call. A collective site's
finer level waits for the next pass, whose GPUs agree it (`agree`). Anywhere else the OOM ends
the request, typed (device_out_of_memory). Model blocks and package code always execute once.
"""

from __future__ import annotations

import functools
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from cozy_runtime.author._errors import Fault, fault

type Args = Sequence[object]
type Kwargs = Mapping[str, object]

#: the attempt being served (`serving`): its pure cut sites recover; outside one nothing does
CURRENT: ContextVar[object | None] = ContextVar("cozy_execution_request", default=None)
#: each (site, shape)'s cut level in this process, raised by an OOM at it, never lowered
_LEVELS: dict[tuple[str, str], int] = {}
_LEVELS_LOCK = threading.Lock()


def level(site: str, shape: str = "") -> int:
    return _LEVELS.get((site, shape), 0)


def levels() -> tuple[tuple[tuple[str, str], int], ...]:
    """Every site's level now: a graph recorded under them replays only under them."""
    with _LEVELS_LOCK:
        return tuple(sorted(_LEVELS.items()))


def set_level(site: str, shape: str, to: int) -> None:
    with _LEVELS_LOCK:
        _LEVELS[(site, shape)] = max(_LEVELS.get((site, shape), 0), to)


class Finer(Exception):
    """An op inside a cut ran out and that cut went finer: it runs again at `level`."""

    def __init__(self, scope: Scope, level: int) -> None:
        super().__init__(f"{scope.site}: cut to level {level}")
        self.scope, self.level = scope, level


@dataclass(eq=False, slots=True)
class Scope:
    """A cut in force: its site, its level and the finest level it has, and how many ops of each
    kind its current piece ran (the sites of the ops inside it: the same in every piece,
    whatever the level). Across a `collective` one no cut goes finer mid-pass (its peers run the
    same pass)."""

    site: str
    finest: int
    level: int = 0
    collective: bool = False
    ordinals: dict[str, int] = field(default_factory=dict)
    #: its operands' shape key: the memo keeps a level per (site, shape)
    shape: str = ""


class _Local(threading.local):
    def __init__(self) -> None:
        self.scopes: list[Scope] = []
        self.named: str | None = None
        self.ordinals: dict[str, int] = {}


_local = _Local()


@contextmanager
def serving(request: object) -> Iterator[None]:
    """The attempt's pure cut sites recover while it runs; package code is never retried."""
    token = CURRENT.set(request)
    try:
        yield
    finally:
        CURRENT.reset(token)


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
    """The shapes of the tensors in `operands`, as the memo keys a site's level by them."""
    return "|".join("x".join(map(str, t.shape)) for t in tensors(operands))


def short(exc: BaseException) -> bool:
    """A pure site's allocation failure outside capture."""
    return fault(exc) is Fault.OOM and not _capturing()


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


def settle[T](run: Callable[[], T], own: Scope | None, first: bool = False) -> T:
    """`run()` until it runs; `run` reads `own.level`. An OOM goes to the innermost cut that can
    go finer (enclosing cuts first, then `own`; `own` first when `first`): `own` runs again at
    its new level, an enclosing one is raised `Finer`. With none it fails, typed."""
    if CURRENT.get() is None or (own is None and not _local.scopes):
        return run()
    while True:
        try:
            return run()
        except Exception as exc:
            if not short(exc):
                raise
            order = (own, *reversed(_local.scopes)) if first else (*reversed(_local.scopes), own)
            scope = next(
                (s for s in order if s is not None and (s.collective or s.level < s.finest)), None
            )
            if scope is None or scope.collective:
                raise
            _clear(exc)
            set_level(scope.site, scope.shape, scope.level + 1)
            if scope is not own:
                raise Finer(scope, scope.level + 1) from None
            scope.level += 1


def cut[T](
    site: str, finest: int, run: Callable[[int], T], operands: object, shape: str | None = None
) -> T:
    """`run(level)` for a call that cuts itself (FFN tokens, attention pieces, a package's
    split, a norm's or a conv's bands): at the level the memo holds for `site` at its `shape`
    (its operands' shapes when None), finer when it or an op inside it ran out."""
    if CURRENT.get() is None:
        return run(0)
    key = shape_key(operands) if shape is None else shape
    scope = Scope(site, finest, level(site, key), shape=key)
    if _capturing():
        return run(scope.level)  # prepared variant; no recovery inside capture
    _local.scopes.append(scope)

    def whole() -> T:
        scope.ordinals.clear()
        return run(scope.level)

    try:
        while True:
            try:
                return settle(whole, scope, first=True)
            except Finer as finer:
                if finer.scope is not scope:
                    raise
                scope.level = finer.level
                _clear(finer)
    finally:
        _local.scopes.remove(scope)


@contextmanager
def collective(site: str, finest: int) -> Iterator[None]:
    """A collective's extent, `finest` its finest level; no cut outside it goes finer. An OOM
    inside (the allocator already made what room it could) is never run again here, since peers
    may be waiting in its collectives: it fails the request, and this process proposes one level
    finer for the next pass's agreement (`agree`)."""
    scope = Scope(site, finest, 0, collective=True)
    _local.scopes.append(scope)
    try:
        yield
    except Exception as exc:
        if CURRENT.get() is not None and short(exc):
            set_level(site, "", min(level(site) + 1, finest))
        raise
    finally:
        _local.scopes.remove(scope)


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

    module.forward = transparent(named)
    vars(module)["_cozy_execution_forward"] = named


def transparent[F: Callable[..., object]](forward: F) -> F:
    """`forward` stands in for its module's own (the class's at each call) with its results:
    marked as TensorFS's wrappers were, which a package's shape check allows (H3's `_plain`)."""
    vars(forward)["_varena"] = True
    return forward


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
