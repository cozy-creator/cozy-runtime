"""Execution naming, cancellation boundaries and the general split.

A site lists its levels' shape estimates (`sizing`); native selection chooses one, makes its room
and observes it, all before a piece runs. A level nothing can make room for is refused there
(`DoesNotFit`), naming the next step. An allocation refused inside the leaf is varena's to answer:
it learns the level's need and says whether a Runtime-owned pure op runs again finer (Retry) or the
refusal stands (Final). Nothing here decides that.

The driver's own allocation (a kernel's code at its first launch, local memory, a library's handle)
fails a launch, not an allocation of ours, so no reclaim answered it: a named forward (a block, a
component) that it failed runs again once varena has given the driver room (`rerun`), where its
operands show it wrote none of them.
"""

from __future__ import annotations

import enum
import functools
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import TYPE_CHECKING, Protocol, cast

import tensorfs.varena.torch as vt
import torch
from tensorfs import varena
from torch import Tensor, nn

from cozy_runtime.author._context import CURRENT as ATTEMPT
from cozy_runtime.author._errors import RERAN, ROOM, DoesNotFit, from_driver, is_capacity
from cozy_runtime.author._model import TRANSPARENT
from cozy_runtime.internal.execution import phase, sizing
from cozy_runtime.internal.residency import PASS_SHAPE, Model, Request, Rerun

if TYPE_CHECKING:
    from tensorfs.varena._facts import SplitSelection

#: varena's split choice (`split_select`, `split_refused`).
type Selection = SplitSelection

type Args = Sequence[object]
type Kwargs = Mapping[str, object]


class Forward(Protocol):
    """A module's forward: any arguments, as torch types `nn.Module.forward`."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


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
#: A forward the driver's own allocation failed runs again at most this often, and waits at most
#: this long all told for memory other programs took (varena's `OUTSIDE_WAIT`), once nothing of
#: ours is left to give.
_RERUNS = 16
_RERUN_WAIT_S = 300.0
#: set on a failure a named forward gave up running again: its enclosing ones do not try
_FINAL = "cozy_rerun_final"


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
    """Stop at an owned eager boundary. In a group, the GPU that stops ends the others' calls
    (`spmd`). A CUDA capture keeps its prepared interval intact; its caller checks cancellation
    before the next launch.
    """
    context = ATTEMPT.get()
    if context is None or _capturing():
        return
    context.raise_if_cancelled()


def _capturing() -> bool:
    initialized = torch.cuda.is_initialized()
    return bool(initialized and torch.cuda.is_current_stream_capturing())


def useful(sizes: Sequence[Sequence[int]]) -> list[Sequence[int]]:
    """The levels worth listing: down to the first within 1/64 of the finest's bytes (the whole
    output is the need by then) or at 1/64 of the whole's. Finer only adds launches."""
    least, whole = sum(sizes[-1]), sum(sizes[0])
    floor = max(least + least // 64, whole // 64)
    return list(sizes[: next(n for n, row in enumerate(sizes) if sum(row) <= floor) + 1])


def owners(request: Request) -> tuple[Model | Request, ...]:
    """What a site's samples live and die with: the running models, or a componentless request."""
    with request.memory.lock:
        models = tuple(m for m in request.memory.models.values() if m.name in request.current)
    return models or (request,)


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.1f} GiB"


def _room_line(failure: BaseException) -> str:
    """The card's room now, for a device OOM's line (`_errors.ROOM`): read where the failure is
    classified, its frames still held."""
    try:
        if not torch.cuda.is_initialized():
            return "a card this process has not started"
        device = torch.cuda.current_device()
        room = vt.room(device)
        offered = (
            f"{_gib(room['driver_reserve'])} kept free for the driver"
            if from_driver(failure)
            else "every evictable byte offered"
        )
    except Exception as exc:  # the line only describes the failure; it never replaces it
        return f"a room that could not be read ({type(exc).__name__})"
    return (
        f"{_gib(room['working_live'])} live and {_gib(room['pinned'])} of weights in use, "
        f"{_gib(room['weights_resident'])} resident, of {_gib(room['capacity'])} on GPU {device}, "
        f"{offered}"
    )


if not ROOM:
    ROOM.append(_room_line)


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
) -> Selection:
    """The level native selection takes, its room made; `DoesNotFit` where none can have it."""
    try:
        selection: Selection = vt.split_select(device, key, menu, stream=stream)
        return selection
    except varena.Exhausted as exc:  # a decision, not a failure: nothing has run
        raise refuse(device, name, "its finest split", sum(menu[-1][1]), exc) from exc


def begin(
    device: int,
    key: str,
    name: str,
    menu: Sequence[tuple[str, Sequence[int]]],
    level: int,
    selection: Selection,
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
    request: Request, key: str, name: str, level: int, selection: Selection | None = None
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


def cut[T](
    name: str, finest: int, need: sizing.Need, run: Callable[[int], T], *, pure: bool = True
) -> T:
    """`run(level)` for a call that cuts itself into 2^level pieces down to `finest` (FFN tokens,
    attention pieces, a norm's or a conv's bands, a package's split). Without native memory, or
    small, it runs whole; inside a capture or another site's leaf, at this key's last level."""
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
        while True:
            level = int(selection["choice_id"])
            token = begin(device, key, name, menu, level, selection, stream)
            LEVELS[key] = level
            note(request, key, name, level, selection)
            try:
                result = run(level)
            except BaseException as exc:
                again = retry(device, key, name, menu, level, token, stream, exc, pure)
                if again is None:
                    if is_capacity(exc):  # the run record names it: a package's `holds` was short
                        request.working_splits[key]["refused"] = {"level": level, "pure": pure}
                    raise
                selection = again
                continue
            vt.split_end(device, token, submitted=True)
            return result
    finally:
        LEAF.release()


def retry(
    device: int,
    key: str,
    name: str,
    menu: Sequence[tuple[str, Sequence[int]]],
    level: int,
    token: int,
    stream: int,
    exc: BaseException,
    pure: bool,
) -> Selection | None:
    """Close the failed leaf. For a pure op whose leaf varena refused memory, varena's answer: the
    finer level to run again at, its room made (`stats.retries`; a short estimate), or None,
    and the failure stands. The refused attempt's frames and tensors go before it counts room."""
    closed = vt.split_end(device, token, submitted=False)
    if not (pure and closed.get("refused_bytes") and is_capacity(exc)):
        return None
    _unwind(exc)
    try:
        again: Selection | None = vt.split_refused(device, key, str(level), stream=stream)
    except varena.Exhausted as refused:
        raise refuse(device, name, "every finer split", sum(menu[-1][1]), refused) from exc
    if again is not None:
        print(f"cozy-runtime: {name} refused at level {level}: a short estimate", file=sys.stderr)
        if (request := CURRENT.get()) is not None:
            request.retries += 1
    return again


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
    around each piece's work observes it, so the level is measured for the next call."""

    def __init__(
        self,
        level: int,
        key: str | None = None,
        name: str = "",
        menu: Sequence[tuple[str, Sequence[int]]] = (),
        device: int = 0,
    ) -> None:
        self.level, self.key, self.name, self.menu, self.device = level, key, name, menu, device

    @contextmanager
    def piece(self) -> Iterator[None]:
        checkpoint()
        if self.key is None or _capturing() or not LEAF.acquire(blocking=False):
            yield
            return
        try:
            stream = int(torch.cuda.current_stream(self.device).cuda_stream)
            # The plan generation now; the level stays the loop's.
            selection = select(self.device, self.key, self.name, self.menu, stream)
            token = begin(
                self.device, self.key, self.name, self.menu, self.level, selection, stream
            )
            submitted = False
            try:
                yield
                submitted = True
            finally:
                vt.split_end(self.device, token, submitted=submitted)
        finally:
            LEAF.release()


def fixed(name: str, finest: int, need: sizing.Need) -> Fixed:
    """The level of a cut into 2^level pieces down to `finest` that a loop keeps throughout,
    chosen now by native selection, its room made; `DoesNotFit` where none can have it. Without
    native memory, or small, whole; inside a capture or another site's leaf, the key's last."""
    checkpoint()
    request = CURRENT.get()
    native = request is not None and request.memory.cuda
    if not (native or _capturing()) or sum(need(0)) < _SMALL:
        return Fixed(0)
    menu = tuple(
        (str(level), row) for level, row in enumerate(useful([need(n) for n in range(finest + 1)]))
    )
    key = f"{name}:{hash(menu):x}"
    if not native or not LEAF.acquire(blocking=False):
        return Fixed(LEVELS.get(key, 0))
    assert request is not None
    try:
        device = request.memory.device.index or 0
        if not vt.split_context(device)["eligible"]:
            return Fixed(LEVELS.get(key, 0))
        request.memory.remember_working_phase(key, owners(request))
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        selection = select(device, key, name, menu, stream)
        level = LEVELS[key] = int(selection["choice_id"])
        note(request, key, name, level, selection)
        return Fixed(level, key, name, menu, device)
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
    # Its passes' shape where its blocks mark them: a prompt's prefill is not its decode (cr-286).
    shape = PASS_SHAPE.set(hash(key) & ((1 << 64) - 1) or 1)
    try:
        with phase.observing(
            device,
            "eager",
            request.working_phases,
            key=key,
        ) as observation:
            yield
            if observation is not None:
                # An eager observation holds the stream it ran on (`phase.observing`).
                cast("torch.cuda.Stream", observation.completion_stream).synchronize()
                observation.complete()
    finally:
        PASS_SHAPE.reset(shape)


def _alone() -> bool:
    """No other process runs this forward beside this one: a group's ranks meet in collectives,
    so one running a forward again alone would wait for the others forever."""
    return not (torch.distributed.is_available() and torch.distributed.is_initialized()) or (
        torch.distributed.get_world_size() == 1
    )


#: operands a forward cannot change: anything else (a list, a dict, an object) might change unseen
_PLAIN = (type(None), bool, int, float, str, bytes, slice, enum.Enum, torch.dtype, torch.device)


def _untouched(args: Args, kwargs: Kwargs) -> Callable[[], bool] | None:
    """A check that the forward has written none of its operands since, by their version counters;
    None where it could not tell: an inference tensor keeps no version, and a list, a dict or an
    object can change unseen. A forward that wrote one cannot run again."""
    kept: list[tuple[Tensor, int]] = []

    def plain(value: object, depth: int) -> bool:
        if isinstance(value, Tensor):
            if value.is_inference():
                return False
            kept.append((value, value._version))
            return True
        if isinstance(value, tuple):
            return depth > 0 and all(plain(item, depth - 1) for item in value)
        return isinstance(value, _PLAIN)

    if not (plain(tuple(args), 3) and all(plain(v, 2) for v in kwargs.values())):
        return None
    return lambda: all(t._version == version for t, version in kept)


def _note(exc: BaseException, key: str, value: object) -> None:
    try:
        setattr(exc, key, value)
    except AttributeError:  # an exception type with no instance attributes
        pass


def rerun(
    request: Request, name: str, exc: BaseException, untouched: bool, attempt: int, waited: float
) -> float | None:
    """Whether the named forward `exc` ended runs again, and after how long a pause. A failure of
    the driver's own allocation (`from_driver`): its frames and tensors go, varena grows the reserve
    its maps leave the driver and gives up what it can (`vt.driver_short`), and it runs again; with
    nothing of ours left to give, after a pause, while memory other programs took may come back.
    None: the failure stands. A forward that wrote an operand in place leaves it to the forward
    around it, as one whose operands it could not check; past the last lever the failure is final
    for every enclosing one."""
    if getattr(exc, _FINAL, False) or not from_driver(exc) or not untouched:
        return None
    _unwind(exc)  # the failed attempt's tensors go before varena counts the room
    try:
        short = vt.driver_short(request.memory.device.index or 0)
    except Exception as refused:  # the failure stands, and says why nothing could run again
        why = f"{type(refused).__name__}: {refused}"
        _note(exc, RERAN, f"varena could not give the driver room ({why})")
        return None
    line = str(exc).partition("\n")[0][:200]
    gave = bool(short["released"] or short["grew"])
    if attempt >= _RERUNS or not (gave or (short["others"] and waited < _RERUN_WAIT_S)):
        tried = f"ran again {attempt} times, nothing more to give"
        _note(exc, RERAN, tried if attempt else "nothing to give")
        _note(exc, _FINAL, True)
        return None
    pause = 0.0 if gave else min(0.25 * 2**attempt, 8.0, _RERUN_WAIT_S - waited)
    if pause:
        print(
            f"cozy-runtime: {name}: {line}: waiting {pause:.2f} s for the "
            f"{_gib(short['others'])} other programs took",
            file=sys.stderr,
        )
        end = time.monotonic() + pause
        while (left := end - time.monotonic()) > 0:
            checkpoint()
            time.sleep(min(left, 0.25))
    request.retries += 1
    request.reruns.append(
        Rerun(
            site=name,
            error=line,
            reserve=short["reserve"],
            released=short["released"],
            free=short["free"],
            paused_ms=int(pause * 1000),
        )
    )
    print(
        f"cozy-runtime: {name}: {line}: varena leaves the driver {_gib(short['reserve'])} and gave "
        f"up {_gib(short['released'])} for it: running it again",
        file=sys.stderr,
    )
    return pause


def wrap(
    module: nn.Module,
    name: str,
    around: Callable[[nn.Module, Args, Kwargs], AbstractContextManager[None]] | None = None,
) -> None:
    """Name a block's execution sites and invoke its forward: once, or again after a failure of the
    driver's own allocation (`rerun`)."""
    forward: Forward = vars(module).get("forward") or current(module)

    def named(*args: object, **kwargs: object) -> object:
        outer = _local.named, _local.ordinals
        attempt, waited = 0, 0.0
        try:
            while True:
                _local.named, _local.ordinals = name, {}
                request = CURRENT.get()
                rerunnable = (
                    request is not None
                    and request.memory.cuda
                    and not request.idle
                    and not _capturing()
                    and _alone()
                )
                untouched = _untouched(args, kwargs) if rerunnable else None
                try:
                    checkpoint()
                    with nullcontext() if around is None else around(module, args, kwargs):
                        return forward(*args, **kwargs)
                except BaseException as exc:
                    if request is None or untouched is None:
                        raise
                    pause = rerun(request, name, exc, untouched(), attempt, waited)
                    if pause is None:
                        raise
                    attempt, waited = attempt + 1, waited + pause
        finally:
            _local.named, _local.ordinals = outer

    functools.update_wrapper(named, module.forward)  # as `@functools.wraps` would
    module.forward = transparent(named)
    vars(module)["_cozy_execution_forward"] = named


def transparent[F](forward: F) -> F:
    """`forward` stands in for its module's own (the class's at each call) with its results,
    which `author.plain_forward` reads as plain."""
    vars(forward)[TRANSPARENT] = True
    return forward


def current(module: nn.Module) -> Forward:
    """`module`'s class forward as it is at each call."""
    return lambda *args, **kwargs: type(module).forward(module, *args, **kwargs)
