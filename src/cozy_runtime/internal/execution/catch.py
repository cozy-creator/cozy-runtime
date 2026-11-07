"""Execution naming, cancellation boundaries and the general split.

A site lists its levels' shape estimates (`sizing`); native selection chooses one, makes its room
and observes it, all before a piece runs. A level nothing can make room for is refused there
(`DoesNotFit`), naming the next step. Every operation runs once: nothing here catches an
allocation failure, which is the memory manager's to prevent.
"""

from __future__ import annotations

import functools
import threading
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
#: Below this a whole op runs unasked: the allocator's reclaim makes its room, and three native
#: calls cost more than a cut could save (an SDXL UNet's hundred small convs, in every eager step).
_SMALL = 64 << 20
#: Native leaves do not nest and this process has one device: a site that finds this held, in
#: another site's leaf on any thread, runs inside that leaf's observation.
LEAF = threading.Lock()
#: each key's last native choice, for a capture (which may not select) to record the same pieces
LEVELS: dict[str, int] = {}


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


def useful(sizes: Sequence[Sequence[int]]) -> list[Sequence[int]]:
    """The levels worth listing: down to the first within 1/64 of the finest's bytes (the whole
    output is the need by then) or at 1/64 of the whole's. Finer only adds launches."""
    least, whole = sum(sizes[-1]), sum(sizes[0])
    floor = max(least + least // 64, whole // 64)
    return list(sizes[: next(n for n, row in enumerate(sizes) if sum(row) <= floor) + 1])


def owners(request: Request) -> tuple[Any, ...]:
    """What a site's samples live and die with: the running models, or a componentless request."""
    with request.memory.lock:
        models = tuple(m for m in request.memory.models.values() if m.name in request.current)
    return models or (request,)


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.1f} GiB"


def refuse(device: int, name: str, level: str, need: int, exc: BaseException) -> DoesNotFit:
    """Native selection cannot make room for `name` at `level` (`need` bytes) with everything
    evictable gone, before any piece ran."""
    room = vt.room(device)
    return DoesNotFit(
        f"{name}: {level} needs {_gib(need)} beside {_gib(room['working_live'])} live and "
        f"{_gib(room['pinned'])} of weights in use, and this card has {_gib(room['capacity'])} "
        f"({exc}): run it smaller, or rent a larger card: `cozy run … --on new`"
    )


def select(
    device: int, key: str, name: str, menu: Sequence[tuple[str, Sequence[int]]], stream: int
) -> dict[str, Any]:
    """The level native selection takes, its room made; `DoesNotFit` where none can have it."""
    try:
        return vt.split_select(device, key, menu, stream=stream)
    except varena.Exhausted as exc:  # a decision, not a failure: nothing has run
        raise refuse(device, name, "its finest split", sum(menu[-1][1]), exc) from exc


def begin(
    device: int,
    key: str,
    name: str,
    menu: Sequence[tuple[str, Sequence[int]]],
    level: int,
    selection: dict[str, Any],
    stream: int,
) -> int:
    """Open `level`'s native observation; `DoesNotFit` where its room is gone since selection
    (another program took it, and the core's wait for it ran out)."""
    phase.choice()
    try:
        return phase.begin_split(device, key, str(level), selection, stream=stream)
    except varena.Exhausted as exc:  # before the op runs
        raise refuse(device, name, f"split level {level}", sum(menu[level][1]), exc) from exc


def note(
    request: Request, key: str, name: str, level: int, selection: dict[str, Any] | None = None
) -> None:
    """A site's calls per level and what each was last chosen against, for the run record."""
    record = request.working_splits.setdefault(key, {"site": name, "levels": {}, "chosen": {}})
    record["levels"][str(level)] = record["levels"].get(str(level), 0) + 1
    if selection is not None:
        record["chosen"][str(level)] = {
            "full": selection["allocation_backing_bytes"],
            "budget": selection["budget_bytes"],
            "capacity": selection["capacity_bytes"],
            "live": selection["live_bytes"],
            "weights": selection["phase_weight_bytes"],
            "required": selection["required_bytes"],
            "available": selection["available_bytes"],
            "target": selection["eligible"][0] if selection["eligible"] else None,
        }


def cut[T](name: str, finest: int, need: sizing.Need, run: Callable[[int], T]) -> T:
    """`run(level)` for a call that cuts itself into 2^level pieces down to `finest` (FFN tokens,
    attention pieces, a norm's or a conv's bands, a package's split), once. Without native memory,
    or small, it runs whole; inside a capture or another site's leaf, at this key's last level."""
    checkpoint()
    request = CURRENT.get()
    native = request is not None and request.memory.cuda
    if not (native or _capturing()) or sum(need(0)) < _SMALL:
        return run(0)
    menu = tuple(
        (str(level), row) for level, row in enumerate(useful([need(n) for n in range(finest + 1)]))
    )
    key = f"{name}:{hash(menu):x}"
    if not native or not LEAF.acquire(blocking=False):
        return run(LEVELS.get(key, 0))
    assert request is not None
    try:
        device = request.memory.device.index or 0
        if not vt.split_context(device)["eligible"]:
            return run(LEVELS.get(key, 0))
        request.memory.remember_working_phase(key, owners(request))
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        selection = select(device, key, name, menu, stream)
        level = int(selection["choice_id"])
        token = begin(device, key, name, menu, level, selection, stream)
        LEVELS[key] = level
        note(request, key, name, level, selection)
        submitted = False
        try:
            result = run(level)
            submitted = True
            return result
        finally:
            vt.split_end(device, token, submitted=submitted)
    finally:
        LEAF.release()


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
