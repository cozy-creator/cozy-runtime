"""Execution naming, cancellation boundaries and the general split.

A site lists its levels' shape estimates (`sizing`); native selection chooses one, makes its room
and observes it, all before a piece runs. A level nothing can make room for is refused there
(`DoesNotFit`), naming the next step. An allocation refused inside the leaf is varena's to answer:
it learns the level's need and says whether a Runtime-owned pure op runs again finer (Retry) or the
refusal stands (Final). Nothing here decides that; package code never runs twice.
"""

from __future__ import annotations

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
from tensorfs.varena import _runtime as native
from torch import Tensor, nn

from cozy_runtime.author._context import CURRENT as ATTEMPT
from cozy_runtime.author._errors import ROOM, DoesNotFit, is_capacity, outside_books
from cozy_runtime.author._model import TRANSPARENT
from cozy_runtime.internal.execution import phase, sizing
from cozy_runtime.internal.residency import PASS_SHAPE, Model, Request

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


def small(need: sizing.Need) -> bool:
    """A whole op this small runs unasked, whatever the room (`_SMALL`)."""
    return sum(need(0)) < _SMALL


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
    """The levels worth listing: every one down to the site's finest, except past the first
    within 1/64 of the finest's bytes, where a whole output is the need and finer only adds
    launches. A site whose need is its windows lists them down to one."""
    least = sum(sizes[-1])
    last = next(n for n, row in enumerate(sizes) if sum(row) <= least + least // 64)
    return list(sizes[: last + 1])


def owners(request: Request) -> tuple[Model | Request, ...]:
    """What a site's samples live and die with: the running models, or a componentless request."""
    with request.memory.lock:
        models = tuple(m for m in request.memory.models.values() if m.name in request.current)
    return models or (request,)


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.1f} GiB"


def _holders(device: int, refused: Mapping[str, int] | None = None) -> str:
    """What holds `device`'s memory: live tensors, weights in use, graph pools and caches, as the
    refused allocation found them (`refused`, `vt.refusal`), else now, with the working pages
    mapped beside the live tensors (free but in pages a live tensor shares, or freed and awaiting
    their stream)."""
    beside = ""
    if refused is None:
        room, stats = vt.room(device), vt.stats(device)
        refused = {
            "working_live": room["working_live"],
            "pinned": room["pinned"],
            "weights_resident": room["weights_resident"],
            "capacity": room["capacity"],
            "pools": stats["pools_bytes"],
            "caches": stats["cache_mapped_bytes"],
        }
        mapped = max(stats["mapped_bytes"] - room["working_live"], 0)
        pending = _gib(stats["pending_bytes"])
        beside = f" and {_gib(mapped)} mapped beside them ({pending} awaiting their stream)"
    pools, caches = refused["pools"], refused["caches"]
    held = (
        f"{_gib(pools)} of graph pools, {_gib(caches)} of caches"
        if pools or caches
        else "no graph pools or caches"
    )
    return (
        f"{_gib(refused['working_live'])} of live tensors{beside}, {_gib(refused['pinned'])} of "
        f"weights in use ({_gib(refused['weights_resident'])} resident), {held}, of "
        f"{_gib(refused['capacity'])} on GPU {device}"
    )


def room(device: int) -> str:
    """`device`'s room now, as a device OOM's line says it."""
    return _holders(device)


def _room_line() -> str:
    """The card's room for a device OOM's line (`_errors.ROOM`): as varena found it when it refused
    the allocation, its pins still held (cr-319: read here, after the frames unwound, Qwen3.6's
    45.2 GiB of pinned weights read "0.0 GiB in use"); "every evictable byte offered" only when
    every lever of the reclaim order was tried first."""
    try:
        if not torch.cuda.is_initialized():
            return "a card this process has not started"
        device = torch.cuda.current_device()
        refused = cast("Mapping[str, int] | None", vt.refusal(device))
        holders = _holders(device, refused)
    except Exception as exc:  # the line only describes the failure; it never replaces it
        return f"a room that could not be read ({type(exc).__name__})"
    if refused is None:
        return f"{holders} now"
    if not refused["offered"]:
        return f"{holders}, its graph pool short"
    if limit := refused.get("host_limit", 0):  # varena's last resort was refused host memory too
        return (
            f"{holders}, every evictable byte offered, and host memory too: "
            f"{_gib(refused['host_free'])} free under its {_gib(limit)} limit for "
            f"{_gib(refused['host_wanted'])} more"
        )
    return f"{holders}, every evictable byte offered"


if not ROOM:
    ROOM.append(_room_line)


class Unplaced(DoesNotFit):
    """No level of a cut can be given room before it runs (or, for a pure op, runs again): its
    caller may run it another way."""


def refuse(device: int, name: str, level: str, need: int, exc: BaseException) -> Unplaced:
    """Native selection cannot make room for `name` at `level` (`need` bytes) with everything
    evictable gone, before any piece ran."""
    return Unplaced(
        f"{name}: {level} needs {_gib(need)} beside {_holders(device)} ({exc}): run it smaller, "
        "or rent a larger card: `cozy run … --on new`"
    )


def fallback(on: bool) -> AbstractContextManager[object]:
    """Allocations inside with a cheaper way out than varena's last resort, pinned host memory the
    card reads over the link: a finer level of a pure op, a split's whole on host, a library's own
    smaller choice, a graph that goes. A refusal then comes back here, the cheaper step runs, and
    varena counts it answered. An enclosing one holds inside."""
    scope = getattr(vt, "fallback", None)  # a TensorFS with the last resort
    return scope(on) if scope is not None else nullcontext()


def _attempt[T](run: Callable[[int], T], level: int, finer: bool, pure: bool, device: int) -> T:
    """`run(level)`, its cheaper ways out first. Package code runs once. A pure op with a finer level
    is refused for it (`retry`, `nested`). One at its last level first leaves a refusal to the
    libraries it calls (cuDNN's next plan, cr-324), and runs again for varena's last resort only if
    one escapes: their own choice costs less than a workspace over the link. One whose library or
    the driver refused its own allocation, outside varena's books, runs once more if varena freed
    room for the driver from what is idle (var-040: a library's first use on a full card)."""
    if not pure:
        return run(level)
    try:
        return _cheaper(run, level, finer)
    except BaseException as exc:
        if not (outside_books(exc) and _driver_room(device)):
            raise
        _unwind(exc)
    print(f"cozy-runtime: a library refused its own allocation at {level}: again", file=sys.stderr)
    if (request := CURRENT.get()) is not None:
        request.retries += 1
    return _cheaper(run, level, finer)


def _driver_room(device: int) -> bool:
    """Room for the driver freed from what is idle; False if none came free or varena can't say."""
    try:
        return native.make_driver_room(device)
    except Exception:
        return False


def _cheaper[T](run: Callable[[int], T], level: int, finer: bool) -> T:
    if finer or getattr(vt, "fallback", None) is None:
        with fallback(finer):
            return run(level)
    try:
        with fallback(True):
            return run(level)
    except BaseException as exc:
        if not is_capacity(exc) or outside_books(exc):
            raise
        _unwind(exc)
    return run(level)


def select(
    device: int,
    key: str,
    name: str,
    menu: Sequence[tuple[str, Sequence[int]]],
    stream: int,
    behind: bool = False,
) -> Selection:
    """The level native selection takes, its room made; `DoesNotFit` where none can have it and a
    cheaper step stands `behind` the refusal, else the level nearest to fitting (varena's floor)."""
    try:
        with fallback(behind):
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
    behind: bool = False,
) -> int:
    """Open `level`'s native observation; `DoesNotFit` where its room is gone since selection
    (another program took it, and the core's wait for it ran out) and a cheaper step stands
    `behind` the refusal, else it opens short (varena's floor)."""
    phase.choice()
    try:
        with fallback(behind):
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


@contextmanager
def timed(request: Request, key: str, level: int, device: int) -> Iterator[None]:
    """A call's span, for its site's `time` in the run record (`Request.split_spans`): two events on
    its stream, read once the GPU passed them, and the calling thread's nanoseconds. A refused try
    counts at its level too: its work was spent."""
    stream = torch.cuda.current_stream(device)
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(stream)
    began = time.perf_counter_ns()
    try:
        yield
    finally:
        stop.record(stream)
        request.split_spans.append((key, str(level), start, stop, time.perf_counter_ns() - began))
        request.split_times()


def cut[T](
    name: str,
    finest: int,
    need: sizing.Need,
    run: Callable[[int], T],
    *,
    pure: bool = True,
    behind: bool = False,
) -> T:
    """`run(level)` for a call that cuts itself into 2^level pieces down to `finest` (FFN tokens,
    attention pieces, a norm's or a conv's bands, a package's split). Without native memory, or
    small, it runs whole; inside a capture, at this key's last level; inside another site's leaf,
    at the level the card's free room takes now (`nested`). `behind`: a cheaper step stands behind
    a level refused before it runs (a split's whole on host), so selection refuses for it."""
    checkpoint()
    request = CURRENT.get()
    native = request is not None and request.memory.cuda
    if not (native or _capturing()) or small(need):
        return run(0)
    menu = tuple(
        (str(level), row) for level, row in enumerate(useful([need(n) for n in range(finest + 1)]))
    )
    key = f"{name}:{hash(menu):x}"
    if not native or _capturing():
        return run(LEVELS.get(key, 0))
    if not LEAF.acquire(blocking=False):
        assert request is not None
        return nested(request, key, name, menu, run, pure)
    assert request is not None
    try:
        device = request.memory.device.index or 0
        if not vt.split_context(device)["eligible"]:
            return run(LEVELS.get(key, 0))
        request.memory.remember_working_phase(key, owners(request))
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        selection = select(device, key, name, menu, stream, behind)
        while True:
            level = int(selection["choice_id"])
            token = begin(device, key, name, menu, level, selection, stream, behind)
            LEVELS[key] = level
            note(request, key, name, level, selection)
            try:
                with timed(request, key, level, device):
                    result = _attempt(run, level, level + 1 < len(menu), pure, device)
            except BaseException as exc:
                again = retry(device, key, name, menu, level, token, stream, exc, pure, behind)
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


#: A nested site's level a refusal moved it to, by key: its next call starts no coarser.
_NESTED: dict[str, int] = {}


def nested[T](
    request: Request,
    key: str,
    name: str,
    menu: Sequence[tuple[str, Sequence[int]]],
    run: Callable[[int], T],
    pure: bool,
) -> T:
    """A cut inside another site's leaf (an FFN or attention in a split's window), where native
    selection can't open a second leaf: the coarsest level whose need the card's room holds now,
    no coarser than one refused before. The room is the card less what is live and the weights in
    use: what is unmapped, cached or held by idle weights, which varena reclaims for it. A pure op
    refused there runs again a level finer, the windows of the leaf around it kept (cr-305: H3's
    upscale ran its MLP whole inside a window and ran out). The last level's refusal stands."""
    level = max(_fitting(request, menu), _NESTED.get(key, 0))
    while True:
        LEVELS[key] = level
        note(request, key, name, level)
        device = request.memory.device.index or 0
        try:
            with timed(request, key, level, device):
                return _attempt(run, level, level + 1 < len(menu), pure, device)
        except BaseException as exc:
            if not (pure and is_capacity(exc)) or level + 1 >= len(menu):
                raise
            _unwind(exc)
        request.retries += 1
        print(
            f"cozy-runtime: {name} refused at level {level} inside another cut: level {level + 1}",
            file=sys.stderr,
        )
        level = _NESTED[key] = level + 1


def _fitting(request: Request, menu: Sequence[tuple[str, Sequence[int]]]) -> int:
    """The coarsest level of `menu` whose need the card's room holds now, else the finest: the card
    less what is live and the weights in use (what is unmapped, cached or held by idle weights,
    varena reclaims for it)."""
    room = vt.room(request.memory.device.index or 0)
    usable = int(room["capacity"]) - int(room["working_live"]) - int(room["pinned"])
    return next((at for at, (_, row) in enumerate(menu) if sum(row) <= usable), len(menu) - 1)


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
    behind: bool = False,
) -> Selection | None:
    """Close the failed leaf. For a pure op whose leaf varena refused memory, varena's answer: the
    finer level to run again at, its room made (`stats.retries`; a short estimate), or None,
    and the failure stands. The refused attempt's frames and tensors go before it counts room."""
    closed = vt.split_end(device, token, submitted=False)
    if not (pure and closed.get("refused_bytes") and is_capacity(exc)):
        return None
    _unwind(exc)
    try:
        with fallback(behind):
            again: Selection | None = vt.split_refused(device, key, str(level), stream=stream)
    except varena.Exhausted as refused:
        raise refuse(device, name, "every finer split", sum(menu[-1][1]), refused) from exc
    if again is not None:
        print(f"cozy-runtime: {name} refused at level {level}: a short estimate", file=sys.stderr)
        if (request := CURRENT.get()) is not None:
            request.retries += 1
    return again


def refusals(device: int) -> Callable[[], tuple[int, int, Request | None]]:
    """varena's count of allocations it refused on `device`, of them those a cheaper step answered
    (inside `fallback`: `vt.answered`; on a TensorFS without that count, the Runtime's retries), and
    the request, for the package contract's scopes."""
    answered = getattr(vt, "answered", None)  # a TensorFS that counts them

    def read() -> tuple[int, int, Request | None]:
        request = CURRENT.get()
        if answered is not None:
            return vt.ooms(device), answered(device), request
        return vt.ooms(device), request.retries if request is not None else 0, request

    return read


def _unwind(exc: BaseException) -> None:
    seen: set[int] = set()
    at: BaseException | None = exc
    while at is not None and id(at) not in seen:
        seen.add(id(at))
        if at.__traceback__ is not None:
            traceback.clear_frames(at.__traceback__)
        at = at.__cause__ or at.__context__


#: A loop's layout a piece's allocation was refused at, by key: the next choice is finer.
_REFUSED: dict[str, int] = {}


class Fixed:
    """A cut's level chosen once, before a loop runs its pieces (`author.windows`). A piece is
    not a leaf: its sites (attention, FFN, norms) choose and retry inside it, as the window's
    need assumes. A piece refused memory once its sites can cut no finer fails its call, and
    the key's next choice is finer (a loop's layout cannot change under it)."""

    def __init__(self, level: int, key: str | None = None, name: str = "", levels: int = 1) -> None:
        self.level, self.key, self.name, self.levels = level, key, name, levels

    @contextmanager
    def piece(self) -> Iterator[None]:
        checkpoint()
        try:
            yield
        except BaseException as exc:
            short = isinstance(exc, DoesNotFit) or is_capacity(exc)
            if self.key is not None and short and self.level + 1 < self.levels:
                _REFUSED[self.key] = self.level + 1
                if (request := CURRENT.get()) is not None and self.key in request.working_splits:
                    request.working_splits[self.key]["refused"] = {
                        "level": self.level,
                        "pure": False,
                    }
                print(
                    f"cozy-runtime: {self.name} refused at level {self.level}: the next call "
                    f"takes level {self.level + 1}",
                    file=sys.stderr,
                )
            raise


def fixed(name: str, finest: int, need: sizing.Need) -> Fixed:
    """The level of a cut into 2^level pieces down to `finest` that a loop keeps throughout,
    chosen now by native selection, its room made, no coarser than one a piece was refused at;
    `DoesNotFit` where none can have it. Without native memory, or small, whole; inside a
    capture, the key's last; inside another site's leaf (a loop in a split's window), as a nested
    cut chooses: the coarsest layout the card's room holds now, no coarser than one refused."""
    checkpoint()
    request = CURRENT.get()
    native = request is not None and request.memory.cuda
    if not (native or _capturing()) or small(need):
        return Fixed(0)
    menu = tuple(
        (str(level), row) for level, row in enumerate(useful([need(n) for n in range(finest + 1)]))
    )
    key = f"{name}:{hash(menu):x}"
    if not native or _capturing():
        return Fixed(LEVELS.get(key, 0))
    assert request is not None
    if not LEAF.acquire(blocking=False):
        level = LEVELS[key] = max(_fitting(request, menu), _REFUSED.get(key, 0))
        note(request, key, name, level)
        return Fixed(level, key, name, len(menu))
    try:
        device = request.memory.device.index or 0
        if not vt.split_context(device)["eligible"]:
            return Fixed(LEVELS.get(key, 0))
        request.memory.remember_working_phase(key, owners(request))
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        selection = select(device, key, name, menu, stream)
        level = LEVELS[key] = max(int(selection["choice_id"]), _REFUSED.get(key, 0))
        note(request, key, name, level, selection)
        return Fixed(level, key, name, len(menu))
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
            began = time.perf_counter()
            yield
            submitted = (time.perf_counter() - began) * 1e3
            if observation is not None:
                # An eager observation holds the stream it ran on (`phase.observing`).
                cast("torch.cuda.Stream", observation.completion_stream).synchronize()
                observation.complete()
        if observation is not None and observation.result is not None:
            from cozy_runtime import graphs  # residency imports graphs: loaded by now

            graphs.passed_at(submitted, int(observation.result["peak_bytes"]))
    finally:
        PASS_SHAPE.reset(shape)


def wrap(
    module: nn.Module,
    name: str,
    around: Callable[[nn.Module, Args, Kwargs], AbstractContextManager[None]] | None = None,
) -> None:
    """Name a block's execution sites and invoke its forward exactly once."""
    forward: Forward = vars(module).get("forward") or current(module)

    def named(*args: object, **kwargs: object) -> object:
        outer = _local.named, _local.ordinals
        _local.named, _local.ordinals = name, {}
        try:
            checkpoint()
            with nullcontext() if around is None else around(module, args, kwargs):
                return forward(*args, **kwargs)
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
