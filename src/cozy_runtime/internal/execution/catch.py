"""Pure-site cuts sized from their operands against executor-local residency (`sizing`).

A pure cut site estimates its level from live shapes (`need`) and local room (`r.room()`).
An unexpected capacity refusal tries bounded reclamation before the full recovery fallback,
retrying only this pure level after memory was released; otherwise it descends. Failure at the
finest level ends the request, typed (device_out_of_memory). Model blocks and package code run once.
"""

from __future__ import annotations

import functools
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field

import msgspec
import torch
from torch import Tensor, nn

from cozy_runtime.author._context import CURRENT as ATTEMPT
from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.internal.execution import sizing
from cozy_runtime.internal.residency import Request

type Args = Sequence[object]
type Kwargs = Mapping[str, object]

#: the request being served (`serving`): its pure cut sites size from its local room
CURRENT: ContextVar[Request | None] = ContextVar("cozy_execution_request", default=None)


class Cut(msgspec.Struct, frozen=True, kw_only=True):
    """A site's dispatch: the level its operands' need fit, that need (bytes, ungranulated),
    the room's tier it fit in, and the levels it descended after refusals."""

    site: str
    level: int
    need: int
    tier: str
    descent: int
    #: Same-level pure retries after observed local reclamation.
    retries: int = 0


class Cuts(msgspec.Struct, kw_only=True):
    """A request's cut sites (`measurements.execution.cuts`): each site's first dispatch and
    every one that descended, and the refusals at the floor."""

    sites: list[Cut] = msgspec.field(default_factory=list)
    oom_floor: int = 0


#: per request (by identity), its cuts so far
_ROWS: dict[int, Cuts] = {}
_ROWS_LOCK = threading.Lock()


@dataclass(eq=False, slots=True)
class Scope:
    """A cut in force: its site, and how many ops of each kind its current piece ran (the sites
    of the ops inside it: the same in every piece, whatever the level)."""

    site: str
    ordinals: dict[str, int] = field(default_factory=dict)


class _Local(threading.local):
    def __init__(self) -> None:
        self.scopes: list[Scope] = []
        self.named: str | None = None
        self.ordinals: dict[str, int] = {}


_local = _Local()


@contextmanager
def serving(request: Request) -> Iterator[None]:
    """The attempt's pure cut sites size from `request`'s room while it runs."""
    token = CURRENT.set(request)
    try:
        yield
    finally:
        CURRENT.reset(token)


def cut_rows(request: Request) -> Cuts:
    """`request`'s cuts, taken at its end (`measurements.execution.cuts`)."""
    with _ROWS_LOCK:
        return _ROWS.pop(id(request), Cuts())


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
    checkpoint()
    if _local.scopes:
        _local.scopes[-1].ordinals.clear()


def checkpoint() -> None:
    """Stop at an owned eager boundary; group cancellation stays with mirrored calls.

    One rank must not leave a collective while another enters it. A CUDA capture also keeps
    its prepared interval intact; its caller checks cancellation before the next launch.
    """
    context = ATTEMPT.get()
    if context is None or _capturing():
        return
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        return
    context.raise_if_cancelled()


def short(exc: BaseException) -> bool:
    """A pure site's allocation refusal outside capture."""
    return fault(exc) is Fault.OOM and not _capturing()


def _clear(exc: BaseException) -> None:
    """The refused attempt's frames go, and with them its tensors."""
    seen: set[int] = set()
    at: BaseException | None = exc
    while at is not None and id(at) not in seen:
        seen.add(id(at))
        if at.__traceback__ is not None:
            traceback.clear_frames(at.__traceback__)
        at = at.__cause__ or at.__context__


def _capturing() -> bool:
    return bool(torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing())


def _refused(exc: BaseException) -> bool:
    if not short(exc):
        return False
    _clear(exc)
    return True


def cut[T](site: str, finest: int, need: sizing.Need, run: Callable[[int], T]) -> T:
    """`run(level)` for a call that cuts itself into 2^level pieces down to `finest` (FFN tokens,
    attention pieces, a package's split, a norm's or a conv's bands), at the coarsest level
    whose `need` fits the request's room; a refusal descends. Outside a request it runs whole;
    inside a capture at the level the room gives now (its pool was prepared for it)."""
    checkpoint()
    request = CURRENT.get()
    if request is None:
        return run(0)
    scope = Scope(site)
    _local.scopes.append(scope)
    available = request.room()
    retries = 0

    def room() -> sizing.Room:
        nonlocal available
        available = request.room()
        return available

    def whole(level: int) -> T:
        nonlocal retries
        checkpoint()
        scope.ordinals.clear()
        # Stock Torch cannot evict live cached weights inside an allocation. A declared pure
        # site's predicted cold/warm bytes must be reclaimed before running that site.
        if not _capturing() and (wanted := need(level, available.granularity)) > available.free:
            request.reclaim(wanted)
        try:
            return run(level)
        except Exception as exc:
            if not short(exc):
                raise
            _clear(exc)
            # A failed allocation can reveal pressure the shape estimate cannot predict.
            # This also runs at the finest level, which sizing.descend cannot make smaller.
            # First release a bounded amount, now that failed-frame pins are gone. An
            # underestimated workspace still gets the existing full-reclaim last chance.
            fresh = request.room()
            target = max(need(level, fresh.granularity), fresh.free + fresh.granularity)
            failure = exc
            for wanted in (target, 0):
                if request.reclaim(wanted) <= 0:
                    continue
                retries += 1
                checkpoint()
                scope.ordinals.clear()
                try:
                    return run(level)
                except Exception as retry_error:
                    if not short(retry_error):
                        raise
                    _clear(retry_error)
                    failure = retry_error
            raise failure from None

    try:
        if _capturing():
            return whole(sizing.choose(finest, need, available)[0])
        try:
            result, level, steps, tier = sizing.descend(finest, need, room, whole, _refused)
        except Exception as exc:
            if short(exc):
                with _ROWS_LOCK:
                    _ROWS.setdefault(id(request), Cuts()).oom_floor += 1
            raise
        row = Cut(
            site=site, level=level, need=need(level, 1), tier=tier, descent=steps, retries=retries
        )
        with _ROWS_LOCK:
            rows = _ROWS.setdefault(id(request), Cuts())
            if steps or retries or all(r.site != site for r in rows.sites):
                rows.sites.append(row)
        return result
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
            checkpoint()
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
