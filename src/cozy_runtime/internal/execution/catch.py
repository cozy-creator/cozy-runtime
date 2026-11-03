"""Execution naming, cancellation boundaries and native uncommitted-scope guards."""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field

import msgspec
import torch
from torch import Tensor, nn

from cozy_runtime.author._context import CURRENT as ATTEMPT
from cozy_runtime.author._model import TRANSPARENT
from cozy_runtime.internal.residency import Request

type Args = Sequence[object]
type Kwargs = Mapping[str, object]

#: the request being served; native providers read its actual owners and phase
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
    every one that descended, each site's eager calls at levels 0, 1, ..., and the refusals at
    the floor."""

    sites: list[Cut] = msgspec.field(default_factory=list)
    levels: dict[str, list[int]] = msgspec.field(default_factory=dict)
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
    """Bind the current request without installing a Python allocator policy."""
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


def uncommitted_scope() -> bool:
    """An uncommitted enclosing scope cannot authorize a native operation."""
    return bool(_local.scopes)


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


def _capturing() -> bool:
    return bool(torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing())


def wrap(
    module: nn.Module,
    name: str,
    around: Callable[[nn.Module, Args, Kwargs], AbstractContextManager[None]] | None = None,
) -> None:
    """Name a block's execution sites and invoke its forward exactly once."""
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
    """`forward` stands in for its module's own (the class's at each call) with its results,
    which `author.plain_forward` reads as plain."""
    vars(forward)[TRANSPARENT] = True
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
