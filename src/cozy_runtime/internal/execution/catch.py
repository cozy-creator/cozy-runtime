"""Execution naming, cancellation boundaries and the general split.

A pure site lists its levels' shape estimates (`sizing`); native selection chooses one and
observes it. An ordinary operation runs once. A refusal inside a Runtime-owned pure op selects
a finer level before anything leaves it; a package's `split` never runs again.
"""

from __future__ import annotations

import functools
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any

import tensorfs.varena.torch as vt
import torch
from torch import nn

from cozy_runtime.author._context import CURRENT as ATTEMPT
from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author._model import TRANSPARENT
from cozy_runtime.internal.execution import phase, sizing
from cozy_runtime.internal.residency import Request

type Args = Sequence[object]
type Kwargs = Mapping[str, object]

#: the request being served: a site's weights and the phase it runs in
CURRENT: ContextVar[Request | None] = ContextVar("cozy_execution_request", default=None)
#: Below this a whole op runs unasked: allocator reclaim covers it, and three native calls cost
#: more than it could save (an SDXL UNet's hundred small convs, in every eager step).
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


def refused(exc: BaseException) -> bool:
    """An allocation refusal outside capture; the refused attempt's frames and tensors go."""
    if fault(exc) is not Fault.OOM or _capturing():
        return False
    seen: set[int] = set()
    at: BaseException | None = exc
    while at is not None and id(at) not in seen:
        seen.add(id(at))
        if at.__traceback__ is not None:
            traceback.clear_frames(at.__traceback__)
        at = at.__cause__ or at.__context__
    return True


def _finer[T](run: Callable[[int], T], level: int, finest: int, pure: bool) -> T:
    """`run(level)` outside a native leaf; a pure op's refusal goes one level finer."""
    while True:
        try:
            return run(level)
        except Exception as exc:
            if not pure or level >= finest or not refused(exc):
                raise
        level += 1


def preferred(menu: Sequence[tuple[str, Sequence[int]]]) -> str:
    """The choice to take room for when nothing fits: the coarsest within 1/64 of the finest's
    bytes. Past it the whole output is the need, and more pieces only add launches."""
    least = sum(menu[-1][1])
    return next(id_ for id_, sizes in menu if sum(sizes) <= least + least // 64)


def owners(request: Request) -> tuple[Any, ...]:
    """What a site's samples live and die with: the running models, or a componentless request."""
    with request.memory.lock:
        models = tuple(m for m in request.memory.models.values() if m.name in request.current)
    return models or (request,)


def select(
    device: int, key: str, menu: Sequence[tuple[str, Sequence[int]]], stream: int, floor: int = 0
) -> tuple[int, dict[str, Any] | None]:
    """The level native selection takes, no coarser than `floor`. An estimate that nothing fits
    is not a refusal: the finest runs, and the allocator decides."""
    try:
        selection = vt.split_select(device, key, menu, preferred=preferred(menu), stream=stream)
    except Exception as exc:
        if fault(exc) is not Fault.OOM:
            raise
        return len(menu) - 1, None
    return max(int(selection["choice_id"]), floor), selection


def begin(
    device: int, key: str, level: int, selection: dict[str, Any] | None, stream: int
) -> int | None:
    """Open `level`'s native observation, making its room; None where the estimate is refused."""
    phase.choice((key, level))
    if selection is None:
        return None
    try:
        return phase.begin_split(device, key, str(level), selection, stream=stream)
    except Exception as exc:
        if fault(exc) is not Fault.OOM:
            raise
        return None


def note(request: Request, key: str, name: str, level: int) -> dict[str, Any]:
    """A site's calls per level, for the run record."""
    record = request.working_splits.setdefault(key, {"site": name, "levels": {}, "finer": 0})
    record["levels"][level] = record["levels"].get(level, 0) + 1
    return record


def cut[T](
    name: str, finest: int, need: sizing.Need, run: Callable[[int], T], *, pure: bool = True
) -> T:
    """`run(level)` for a call that cuts itself into 2^level pieces down to `finest` (FFN tokens,
    attention pieces, a norm's or a conv's bands, a package's split). Without native memory, or
    small, it runs whole; inside a capture or another site's leaf, at this key's last level."""
    checkpoint()
    request = CURRENT.get()
    if request is None or not request.memory.cuda or sum(need(0)) < _SMALL:
        return _finer(run, 0, finest, pure)
    menu = tuple((str(level), need(level)) for level in range(finest + 1))
    key = f"{name}:{hash(menu):x}"
    if not LEAF.acquire(blocking=False):
        return _finer(run, LEVELS.get(key, 0), finest, pure)
    try:
        device = request.memory.device.index or 0
        if not vt.split_context(device)["eligible"]:
            return _finer(run, LEVELS.get(key, 0), finest, pure)
        request.memory.remember_working_phase(key, owners(request))
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        floor = 0
        while True:
            level, selection = select(device, key, menu, stream, floor)
            token = begin(device, key, level, selection, stream)
            LEVELS[key] = level
            record = note(request, key, name, level)
            submitted = False
            try:
                result = run(level)
                submitted = True
                return result
            except Exception as exc:
                if not pure or level >= finest or not refused(exc):
                    raise
                record["finer"] += 1
                floor = level + 1
            finally:
                if token is not None:
                    vt.split_end(device, token, submitted=submitted)
    finally:
        LEAF.release()


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
