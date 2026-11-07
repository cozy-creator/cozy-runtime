"""Execution naming, cancellation boundaries and the general split.

A cut registers its levels' shape estimates (`sizing`) once as a varena site; varena chooses each
op's level, makes its room and observes it, all before a piece runs, and runs a small op or one
inside another's leaf unobserved. A level nothing can make room for is refused there (`Final`),
naming the next step. An allocation refused inside the leaf is varena's to answer: a Runtime-owned
pure op runs again (Retry) at the level it gives, anything else's refusal stands (Final). Nothing
here decides that; package code never runs twice.
"""

from __future__ import annotations

import functools
import sys
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

import tensorfs.varena.torch as vt
import torch
from tensorfs import varena
from torch import Tensor, nn

from cozy_runtime.author._context import CURRENT as ATTEMPT
from cozy_runtime.author._errors import DoesNotFit
from cozy_runtime.author._model import TRANSPARENT
from cozy_runtime.internal.execution import phase, sizing
from cozy_runtime.internal.residency import Request

type Args = Sequence[object]
type Kwargs = Mapping[str, object]

#: the request being served: a site's weights and the phase it runs in
CURRENT: ContextVar[Request | None] = ContextVar("cozy_execution_request", default=None)
#: each cut's varena site, by its name and operands' extent
SITES: dict[tuple[str, int, tuple[int, ...]], vt.Site] = {}


class _Local(threading.local):
    def __init__(self) -> None:
        self.named: str | None = None
        self.ordinals: dict[str, int] = {}


_local = _Local()


@contextmanager
def serving(request: Request) -> Iterator[None]:
    """Bind the current request for the sites it runs."""
    token = CURRENT.set(request)
    try:
        yield
    finally:
        CURRENT.reset(token)


def site(kind: str) -> str | None:
    """A known cut's stable name under the current model block."""
    if _local.named is None:
        return None
    n = _local.ordinals[kind] = _local.ordinals.get(kind, 0) + 1
    return f"{_local.named}/{kind}#{n}"


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


def owners(request: Request) -> tuple[Any, ...]:
    """What a site's samples live and die with: the running models, or a componentless request."""
    with request.memory.lock:
        models = tuple(m for m in request.memory.models.values() if m.name in request.current)
    return models or (request,)


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.1f} GiB"


def refused(name: str, exc: BaseException) -> DoesNotFit:
    """varena's final answer for `name` (`varena.Final`), as one line naming the next step."""
    answer = getattr(exc, "refusal", {})
    level = "its finest split" if answer.get("finest") else f"split level {answer.get('level', 0)}"
    held = (
        f"{_gib(answer.get('live_bytes', 0))} live and "
        f"{_gib(answer.get('pinned_bytes', 0))} of weights in use"
    )
    if answer.get("next") == "again":
        then = "varena learned what it needs: run it again and it splits finer"
    else:
        then = "run it smaller, or rent a larger card: `cozy run … --on new`"
    return DoesNotFit(
        f"{name}: {level} needs {_gib(answer.get('need_bytes', 0))} beside {held}, and this card "
        f"has {_gib(answer.get('capacity_bytes', 0))}: {then}"
    )


def _site(device: int, name: str, finest: int, need: sizing.Need, pure: bool) -> vt.Site:
    """`name`'s varena site for these operands: registered at its first call, keyed by its whole
    level's bytes; varena keeps the levels worth having."""
    whole = tuple(need(0))
    found = SITES.get((name, finest, whole))
    if found is None:
        menu = [need(n) for n in range(finest + 1)]
        found = SITES[(name, finest, whole)] = vt.Site(
            device, f"{name}:{hash(tuple(menu)):x}", menu, pure=pure
        )
    return found


def menu_site(device: int, name: str, menu: Sequence[Sequence[int]]) -> vt.Site:
    """`name`'s varena site for an explicit menu (each level's allocations, largest first): an
    op no varena answer may run again (a group's collective)."""
    return _site(device, name, len(menu) - 1, lambda level: tuple(menu[level]), False)


def note(request: Request, key: str, name: str, leaf: vt.Leaf) -> None:
    """A site's calls per level and what each was last chosen against, for the run record."""
    record = request.working_splits.setdefault(key, {"site": name, "levels": {}, "chosen": {}})
    level = str(leaf.level)
    record["levels"][level] = record["levels"].get(level, 0) + 1
    if (selection := leaf.selection) is not None:
        record["chosen"][level] = {
            "full": selection["allocation_backing_bytes"],
            "budget": selection["budget_bytes"],
            "capacity": selection["capacity_bytes"],
            "live": selection["live_bytes"],
            "weights": selection["phase_weight_bytes"],
            "required": selection["required_bytes"],
            "available": selection["available_bytes"],
            "target": selection["eligible"][0] if selection["eligible"] else None,
        }


def _device() -> int | None:
    """The device a cut runs on: the request's native one, a capture's, else none (whole)."""
    request = CURRENT.get()
    if request is not None and request.memory.cuda:
        return request.memory.device.index or 0
    return int(torch.cuda.current_device()) if _capturing() else None


def cut[T](
    name: str, finest: int, need: sizing.Need, run: Callable[[int], T], *, pure: bool = True
) -> T:
    """`run(level)` for a call that cuts itself into 2^level pieces down to `finest` (FFN tokens,
    attention pieces, a norm's or a conv's bands, a package's split), at the level varena gives
    it. Without native memory it runs whole."""
    checkpoint()
    device = _device()
    if device is None:
        return run(0)
    site = _site(device, name, finest, need, pure)
    stream = int(torch.cuda.current_stream(device).cuda_stream)
    request = CURRENT.get()
    if request is not None:
        request.memory.remember_working_phase(site.key, owners(request))
    try:
        leaf = site.enter(stream=stream, owner=phase.completion_owner(device, stream))
    except varena.Exhausted as exc:  # varena's Final: nothing has run
        raise refused(name, exc) from exc
    while True:
        if leaf.token is not None:
            phase.choice()
            if request is not None:
                note(request, site.key, name, leaf)
        try:
            result = run(leaf.level)
        except BaseException as exc:
            _unwind(exc)  # the attempt's frames and tensors go before varena counts room
            try:
                again = leaf.exit(submitted=False)
            except varena.Exhausted as final:
                if request is not None:  # the run record names it: a package's `holds` was short
                    level = getattr(final, "refusal", {}).get("level")
                    request.working_splits[site.key]["refused"] = {"level": level, "pure": pure}
                raise refused(name, final) from exc
            if again is None:
                raise
            print(f"cozy-runtime: {name} refused at level {leaf.level}: varena runs it again "
                  f"at level {again.level}", file=sys.stderr)
            if request is not None:
                request.finer_splits += 1
            leaf = again
            continue
        leaf.exit(submitted=True)
        return result


def refusals(device: int) -> Callable[[], tuple[int, Request | None]]:
    """varena's count of allocations it refused on `device`, with the request whose Retries it
    excludes, for the package contract's scopes."""
    return lambda: (vt.ooms(device), CURRENT.get())


def _unwind(exc: BaseException) -> None:
    seen: set[int] = set()
    at: BaseException | None = exc
    while at is not None and id(at) not in seen:
        seen.add(id(at))
        if at.__traceback__ is not None:
            traceback.clear_frames(at.__traceback__)
        at = at.__cause__ or at.__context__


class Fixed:
    """A cut's level chosen once, before a loop runs its pieces (`author.windows`): `piece()`
    around each piece's work observes it at that level, so it is measured for the next call. A
    refusal inside a piece is Final: the layout is the package's loop's."""

    def __init__(self, level: int, site: vt.Site | None = None, name: str = "") -> None:
        self.level, self.site, self.name = level, site, name

    @contextmanager
    def piece(self) -> Iterator[None]:
        checkpoint()
        if self.site is None:
            yield
            return
        device = self.site.device
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        try:
            leaf = self.site.enter(
                stream=stream, owner=phase.completion_owner(device, stream), level=self.level
            )
        except varena.Exhausted as exc:
            raise refused(self.name, exc) from exc
        try:
            yield
        except BaseException as exc:
            _unwind(exc)
            try:
                leaf.exit(submitted=False)
            except varena.Exhausted as final:
                raise refused(self.name, final) from exc
            raise
        leaf.exit(submitted=True)


def fixed(name: str, finest: int, need: sizing.Need) -> Fixed:
    """The level of a cut into 2^level pieces down to `finest` that a loop keeps throughout,
    chosen now by varena, its room made; `DoesNotFit` where none can have it. Without native
    memory, whole."""
    checkpoint()
    device = _device()
    if device is None:
        return Fixed(0)
    site = _site(device, name, finest, need, False)
    request = CURRENT.get()
    if request is not None:
        request.memory.remember_working_phase(site.key, owners(request))
    try:
        leaf = site.choose(stream=int(torch.cuda.current_stream(device).cuda_stream))
    except varena.Exhausted as exc:
        raise refused(name, exc) from exc
    if request is not None:
        note(request, site.key, name, leaf)
    return Fixed(leaf.level, site, name)


def _tensors(value: object, depth: int = 3) -> list[Tensor]:
    """The tensors in `value`, `depth` levels of sequences and mappings deep."""
    if isinstance(value, Tensor):
        return [value]
    if depth and isinstance(value, list | tuple | Mapping):
        items = value.values() if isinstance(value, Mapping) else value
        return [t for item in items for t in _tensors(item, depth - 1)]
    return []


@contextmanager
def passing(module: nn.Module, args: Args, kwargs: Kwargs) -> Iterator[None]:
    """A component's forward as one observed execution, keyed by its operands' shapes: native
    keeps its working peak for the next pass's prefetch, and its sites' measured peaks replace
    their estimates. An enclosing loop's observation owns the device instead."""
    request = CURRENT.get()
    if request is None or request.idle or not request.memory.cuda or _capturing():
        yield
        return
    device = request.memory.device.index or 0
    shapes = [(tuple(t.shape), str(t.dtype)) for t in _tensors((args, kwargs))]
    key = phase.execution_key(f"{type(module).__qualname__}:{shapes}", "runtime.pass/1")
    assert key is not None
    request.memory.remember_working_phase(key, owners(request))
    with phase.observing(device, "eager", request.working_phases, key=key) as observation:
        yield
        if observation is not None:
            observation.completion_stream.synchronize()
            observation.complete()


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
