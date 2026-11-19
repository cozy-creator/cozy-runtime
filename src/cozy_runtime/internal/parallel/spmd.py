"""Every GPU of a group runs the attempt's handler, as torchrun runs a script on every rank.

The same request, seeds and random streams make every GPU compute the same per-step state with
the same code; a `_cp_plan` component splits and gathers inside its own call (Diffusers' hooks).
What one GPU alone decides or holds crosses here, always at the same point of every GPU's
program: a store meeting (`Ranks.meet`) that also says which GPUs are not at the same call, then
one broadcast per tensor from the GPU that holds it.

A stage (`uses_components`) none of whose components is sharded runs on one GPU (`place`): its
components' home, else GPU 0. Its result and the random streams' state after it cross to the
others, which fetch, hold and compute nothing for it; while they wait they serve its `spread`
calls. A stage with a sharded component runs on every GPU, at the degree chosen at its first
sharded call (`_Choice`); at one GPU, GPU 0 runs its sharded calls and their outputs cross.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import itertools
import json
import mmap
import os
import random
import secrets
import sys
import threading
import time
import types
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, cast

import msgspec

from cozy_runtime.author import _group, _model
from cozy_runtime.internal.canonical import Json
from cozy_runtime.internal.parallel import wire
from cozy_runtime.internal.parallel.group import ENDED, gather
from cozy_runtime.internal.parallel.plan import GroupRefusal

if TYPE_CHECKING:
    from types import ModuleType

    import torch
    from torch import Tensor
    from torch.distributed import ProcessGroup

#: One GPU's row at a meeting: what it is at, and what it states there.
type Row = dict[str, Json]


class UngatedShardedForward(GroupRefusal):
    """A sharded or hosted component called outside a group attempt."""

    default_code = "ungated_sharded_forward"


#: A crossing's payload as its owner sends it: its structure, its tensors, its shared mapping.
type _Made = tuple[Json, list[torch.Tensor], _Shared | None]


def _made(value: object, what: str) -> _Made:
    tensors: list[torch.Tensor] = []
    structure = wire.marshal(value, tensors)
    return structure, tensors, _Shared.make(tensors, what)


class Ranks:
    """This GPU's part of one group attempt: who it is among the group, and the meetings and
    crossings that keep every GPU on one program."""

    def __init__(
        self,
        *,
        rank: int,
        world: int,
        pg: ProcessGroup,
        device: torch.device,
        tag: str,
        gpu: Callable[[int], str],
        stage: Callable[[str], None] | None = None,
        call: tuple[str, str] = ("", ""),
        route: str = "",
    ) -> None:
        import torch.distributed as dist
        from torch._C._distributed_c10d import PrefixStore
        from torch.distributed.distributed_c10d import _get_process_group_store

        self.rank, self.world, self.pg, self.device, self.gpu = rank, world, pg, device, gpu
        #: the generation and the function this attempt runs: a stage's degree key (`degrees`)
        self.call = call
        #: the route the group's exchange takes (`links.Transport.route`; GPU 0's, which chooses)
        self.route = route
        #: each sharded stage's calls so far in this attempt: a record row a call (`_degree`)
        self.calls: dict[str, int] = {}
        #: tells the machine a long meeting's wait as the call's stage (GPU 0's; None elsewhere)
        self.stage = stage
        self.store = PrefixStore(f"cozy.spmd/{tag}/", _get_process_group_store(pg))
        self.sequence = 0
        self.lock = threading.Lock()
        self.nccl = dist.get_backend(pg) == "nccl"

    def meet(self, value: Mapping[str, Json], what: str) -> list[Row]:
        """Every GPU's `value` at this point of the program. GPUs at different points raise,
        naming each; a GPU whose handler already ended is named at once (`gather`). A GPU still
        computing toward it (a hosted call's home, a spread round's) is waited for as long as it
        lives; a long wait is told as "waiting for GPU 1: <what>"."""
        with self.lock:
            self.sequence += 1
            n = self.sequence
            keys = [f"{n}/{rank}" for rank in range(self.world)]
            self.store.set(keys[self.rank], json.dumps({"what": what, **value}))
            ended = [f"end/{rank}" for rank in range(self.world)]
            waiting = self._waiting(what)
            rows: list[Row] = [
                json.loads(raw) for raw in gather(self.store, keys, what, ended, waiting)
            ]
            if n > 1:
                self.store.delete_key(f"{n - 1}/{self.rank}")
        if len({row["what"] for row in rows}) > 1:
            raise GroupRefusal(
                "the GPUs are not at one call: "
                + ", ".join(f"{self.gpu(rank)} at {row['what']}" for rank, row in enumerate(rows)),
                code="group_diverged",
            )
        return rows

    def _waiting(self, what: str) -> Callable[[list[int]], None] | None:
        stage = self.stage
        if stage is None:
            return None

        def waiting(absent: list[int]) -> None:
            stage(f"waiting for {', '.join(self.gpu(rank) for rank in absent)}: {what}")

        return waiting

    def end(self, ok: bool) -> None:
        """This GPU's handler returned or raised: a GPU still waiting at a meeting for it learns
        so at once."""
        self.store.set(f"end/{self.rank}", "ok" if ok else "failed")

    def clear(self, meetings: Mapping[int, int]) -> None:
        """GPU 0, once every GPU answered: the attempt's last keys go."""
        keys = [f"end/{rank}" for rank in range(self.world)]
        keys += [f"{n}/{rank}" for rank, n in meetings.items() if n]
        for key in keys:
            with contextlib.suppress(Exception):
                self.store.delete_key(key)

    def all(self, flag: bool, what: str) -> bool:
        return all(row["flag"] for row in self.meet({"flag": bool(flag)}, what))

    def told(self, value: Mapping[str, Json] | None, what: str) -> Row:
        """GPU 0's `value` on every GPU."""
        told = self.meet({"told": dict(value or {}) if self.rank == 0 else None}, what)[0]["told"]
        if not isinstance(told, dict):
            raise GroupRefusal(f"{what}: GPU 0 told no value", code="group_diverged")
        return told

    def cross(self, value: object, owner: int, what: str, made: _Made | None = None) -> object:
        """`value` as GPU `owner` holds it, on every GPU: its structure through the store, each
        device tensor by one broadcast from `owner` onto this GPU, and its host tensors (and
        arrays) through one shared mapping every other GPU reads in place (`_Shared`; `made`:
        the owner's, made already). The mapping's name and pages go however the crossing ends."""
        tensors: list[torch.Tensor] = []
        shared: _Shared | None = None
        if self.rank == owner:
            structure, tensors, shared = made or _made(value, what)
        else:
            structure = None
        try:
            name = shared.name if shared is not None else ""
            rows = self.meet({"structure": structure, "shared": name}, what)
            structure, name = rows[owner]["structure"], str(rows[owner].get("shared") or "")
            specs = wire.specs(structure)
            views = _Shared.views(name, specs, what) if name and self.rank != owner else {}
            received = [
                (tensors[spec.v] if self.rank == owner else views[spec.v])
                if name and spec.host
                else self._broadcast(tensors[spec.v] if self.rank == owner else None, spec, owner)
                for spec in specs
            ]
            if name:
                self.meet({}, f"{what}: mapped")  # every GPU holds the mapping: its name goes
        finally:
            if shared is not None:
                shared.done()
        return wire.unmarshal(structure, received)

    def spread[*Ts, R](
        self, method: Callable[[*Ts], R], arguments: Iterable[tuple[*Ts]]
    ) -> Iterator[R]:
        """Call n runs on GPU n mod K; a round's K results cross from their GPUs to every GPU,
        and come out in call order."""
        name = getattr(method, "__qualname__", getattr(method, "__name__", "spread"))
        calls, at = iter(arguments), 0
        while batch := list(itertools.islice(calls, self.world)):
            mine = method(*batch[self.rank]) if self.rank < len(batch) else None
            sent = wire.one_tensor(mine) if self.rank < len(batch) else None
            tensors: list[torch.Tensor] = []
            head = wire.marshal(sent, tensors)
            rows = self.meet({"head": head}, f"spread {name} {at}-{at + len(batch) - 1}")
            for owner in range(len(batch)):
                spec = msgspec.convert(rows[owner]["head"], wire._Tensor)
                received = self._broadcast(sent if owner == self.rank else None, spec, owner)
                # every call returns one tensor (`author.spread`): GPU `owner`'s is that tensor
                yield cast(R, mine if owner == self.rank else received)
            at += len(batch)
            if len(batch) < self.world:
                return

    def _broadcast(
        self, tensor: torch.Tensor | None, spec: wire._Tensor, owner: int
    ) -> torch.Tensor:
        import torch
        import torch.distributed as dist
        from torch.utils._python_dispatch import _disable_current_modes

        dtype = getattr(torch, spec.dtype)
        wire_device = self.device if self.nccl else torch.device("cpu")
        with _disable_current_modes():
            if tensor is not None:
                buffer = tensor.detach().resolve_conj().resolve_neg().contiguous()
                buffer = buffer.to(wire_device)
            else:
                buffer = torch.empty(spec.shape, dtype=dtype, device=wire_device)
            if buffer.numel():
                dist.broadcast(buffer.reshape(-1).view(torch.uint8), src=owner, group=self.pg)
        if tensor is not None:
            return tensor
        return buffer.cpu() if spec.host else buffer.to(self.device)


def formed() -> None:
    """A group formed on this GPU: what an earlier group decided per process goes, so a GPU
    that came back meets one that kept nothing either (`_WRITES`), and files of processes that
    ended are swept from /dev/shm (a crossing's mapping, a board)."""
    _WRITES.clear()
    sweep()


def sweep() -> None:
    """Remove `cozy-cross-<pid>-*` and `cozy-board-<pid>-*` files whose process has ended."""
    import glob

    for path in glob.glob("/dev/shm/cozy-cross-*") + glob.glob("/dev/shm/cozy-board-*"):
        try:
            pid = int(os.path.basename(path).split("-")[2])
        except (IndexError, ValueError):
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            with contextlib.suppress(OSError):
                os.unlink(path)
        except PermissionError:  # alive, another user's
            pass


def current() -> Ranks | None:
    group = _group.CURRENT[0]
    return group if isinstance(group, Ranks) else None


@contextlib.contextmanager
def attempt(ranks: Ranks, entropy: int) -> Iterator[Ranks]:
    """`ranks` serves this process's attempt; every GPU's global random streams start from the
    attempt's entropy, so code drawing from them draws the same numbers everywhere."""
    seed(entropy)
    _group.CURRENT[0] = ranks
    try:
        yield ranks
    finally:
        _group.CURRENT[0] = None


def seed(entropy: int) -> None:
    import torch

    random.seed(entropy)
    torch.manual_seed(entropy & ((1 << 63) - 1))
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(entropy & 0xFFFFFFFF)


def install_call_check(component: torch.nn.Module, name: str) -> None:
    """Before each call of a sharded root, every GPU meets at it: a GPU at another call, or one
    whose handler already returned, is named instead of hanging a collective. Arguments may
    differ: a GPU's own rows of a table the plan splits are its own. There each GPU states the
    kernels it has landed, and every GPU serves those landed on all of them from this call on
    (`kernel_ready.landed`): a sparse kernel built during the attempt serves its next steps.
    A sharded stage's first call chooses its degree (`_Choice`), so the check wraps the call
    itself: at degree 1 the call's own CP hooks are off before torch reads its forward."""
    if getattr(component, "_cozy_spmd_check", None) is not None:
        return
    object.__setattr__(component, "_cozy_spmd_check", name)
    # A method of the component, so a copy of it checks the copy.
    object.__setattr__(component, "_call_impl", types.MethodType(_checked_call, component))


def _checked_call(module: torch.nn.Module, *args: object, **kwargs: object) -> object:
    from cozy_runtime.internal import kernel_ready

    name = str(module._cozy_spmd_check)
    choice = getattr(_CHOICE, "stage", None)
    if choice is not None:
        choice.called(module, name, args, kwargs)
        if choice.hosted:
            return choice.hosted_call(module, args, kwargs)
    if not getattr(_ALONE, "unsharded", False):  # a stage at degree 1 runs on its own
        ranks = current()
        if ranks is None:
            raise UngatedShardedForward(
                f"sharded component {name!r} was called outside a group attempt: every GPU "
                "of its group joins its collectives, and none is running one"
            )
        rows = ranks.meet({"landed": kernel_ready.landed()}, f"{name} call")
        states = [set(cast("list[str]", row.get("landed") or [])) for row in rows]
        common = set.intersection(*states)
        if common:
            kernel_ready.boundary(common)
    return type(module)._call_impl(module, *args, **kwargs)


_hosting = threading.local()


def host_component(component: torch.nn.Module, *, name: str, home: int, rank: int) -> None:
    """Every module call into `component`, its root or any submodule, runs on GPU `home`, which
    alone holds its bytes; its result crosses to every GPU (`Ranks.cross`)."""
    for path, module in component.named_modules():
        what = f"{name}.{path}" if path else name
        object.__setattr__(module, "_call_impl", _hosted(module._call_impl, what, home, rank))


def _hosted[**P](call: Callable[P, object], what: str, home: int, rank: int) -> Callable[P, object]:
    def hosted(*args: P.args, **kwargs: P.kwargs) -> object:
        if getattr(_hosting, "inside", False):
            return call(*args, **kwargs)
        ranks = current()
        if ranks is None:
            if rank == home:
                return call(*args, **kwargs)
            raise UngatedShardedForward(
                f"{what} lives on GPU {home}'s process and was called outside a group attempt"
            )
        if ranks.rank != home:
            return ranks.cross(None, home, f"hosted {what}")
        _hosting.inside = True
        try:
            result = call(*args, **kwargs)
        finally:
            _hosting.inside = False
        return ranks.cross(result, home, f"hosted {what}")

    return hosted


class Placement:
    """This GPU's `author.Placement`: a scope whose components all live on other GPUs stages
    nothing here (their calls run on their home GPUs, `host_component`)."""

    def __init__(self, hosted: Mapping[str, int], rank: int) -> None:
        self.hosted, self.rank = dict(hosted), rank

    def remote(self, components: tuple[str, ...]) -> bool:
        return bool(components) and all(
            self.hosted.get(name, self.rank) != self.rank for name in components
        )


def signature(value: object, depth: int = 0) -> str:
    """`value`'s shape: tensor shapes and dtypes, files' kinds, containers' structure, and
    other values' types (numbers and text may differ between GPUs; their places may not)."""
    import torch

    if depth > 12:
        return "…"
    if isinstance(value, torch.Tensor):
        return f"{str(value.dtype).removeprefix('torch.')}{list(value.shape)}"
    if isinstance(value, msgspec.Struct):
        return _record(value, value.__struct_fields__, depth)
    kind = getattr(value, "kind", None)
    if isinstance(kind, str) and hasattr(value, "media_type"):
        return f"file:{kind}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(signature(v, depth + 1) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{signature(v, depth + 1)}" for k, v in value.items()) + "}"
    if dataclasses.is_dataclass(value):
        return _record(value, tuple(f.name for f in dataclasses.fields(value)), depth)
    return type(value).__name__


def _record(value: object, fields: tuple[str, ...], depth: int) -> str:
    inner = ",".join(f"{f}={signature(getattr(value, f), depth + 1)}" for f in fields)
    return f"{type(value).__name__}({inner})"


class Stages:
    """Where a group's model runs each stage: the components its GPUs shard, and the GPU each
    homed component lives on (`host_component`); every other component is GPU 0's."""

    def __init__(
        self, sharded: Iterable[str], homes: Mapping[str, int], roots: Mapping[str, object]
    ) -> None:
        self.sharded = frozenset(sharded)
        self.homes = dict(homes)
        #: each root by identity, so a `spread` call names its component to the other GPUs
        self.roots = dict(roots)

    def home(self, components: tuple[str, ...]) -> int | None:
        """The one GPU a stage of `components` runs on; None: every GPU (a sharded one among them,
        or components homed apart)."""
        if self.sharded.intersection(components):
            return None
        homes = {self.homes.get(name, 0) for name in components}
        return homes.pop() if len(homes) == 1 else None

    def name(self, component: object) -> str | None:
        return next((name for name, root in self.roots.items() if root is component), None)


#: Per process: the stage running alone on this GPU, if any (`place`).
_ALONE = threading.local()
#: Stages that wrote state beside their result once (`_left`): every GPU of the group runs them,
#: as every GPU learned it from the same crossing.
_WRITES: set[str] = set()


def place(
    model: object,
    method: str,
    components: tuple[str, ...],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    run: Callable[[], object],
) -> object:
    """`run` (the stage `model.method`) where its group runs it (`author._model.PLACE`). A stage
    runs on every GPU where a component is sharded, its components live apart, it is called
    inside another stage (whose body may reach a sharded component: H3's turbo samplers call a
    LoRA model's stages around the base DiT), an argument cannot cross (a pipeline's state, a
    generator, a callable: what the stage does to it on the other GPUs is unknown), or it once
    wrote state beside its result (`_writes`)."""
    ranks = current()
    stages = getattr(model, "_cozy_group_stages", None)
    if ranks is None or not isinstance(stages, Stages) or getattr(_ALONE, "stage", None):
        return run()
    key = f"{type(model).__qualname__}.{method}"
    home = stages.home(components)
    given: list[torch.Tensor] = []
    if _model.open_stage():
        return run()
    if home is None and stages.sharded.intersection(components) and ranks.call[0]:
        return _degree(ranks, stages, key, run)
    if home is None or key in _WRITES:
        return run()
    if not _crossable((args, kwargs), given):
        return run()
    what = f"stage {method}"
    if ranks.rank == home:
        return _run_alone(
            ranks, stages, home, what, run, given, (model, key, components), (args, kwargs)
        )
    return _receive(ranks, stages, home, what, run, given, key)


def _crossable(value: object, tensors: list[torch.Tensor]) -> bool:
    try:
        wire.marshal(value, tensors)
    except wire.UncrossableArgument:
        return False
    return True


@contextlib.contextmanager
def _alone(
    ranks: Ranks, stages: Stages, home: int, what: str, *, served: bool = True
) -> Iterator[None]:
    """This GPU runs a stage by itself: nothing inside meets the others (hosted calls run here,
    the memo decides alone), except, where they wait for it (`served`), its `spread` calls,
    which go to them (`_Alone`)."""
    before = _group.CURRENT[0]
    _ALONE.stage = what
    _group.CURRENT[0] = _Alone(ranks, stages, home, what, served)
    try:
        yield
    finally:
        _group.CURRENT[0] = before
        _ALONE.stage = None


def _run_alone(
    ranks: Ranks,
    stages: Stages,
    home: int,
    what: str,
    run: Callable[[], object],
    given: list[torch.Tensor],
    stage: tuple[object, str, tuple[str, ...]],
    arguments: tuple[tuple[object, ...], dict[str, object]],
) -> object:
    """The stage here, alone; then its result, the streams' state, and every argument tensor it
    may have written in place to the others: its version moved, or it keeps none (an inference
    tensor), so the others never read a stale argument. A stage that left other state behind
    (`_left`: in the model, its components or its arguments' structure) is run by the others
    too, and by every GPU from then on; so is one whose result cannot cross. One whose host
    tensors find no shared mapping now runs on the others too, this once."""
    import torch

    model, key, components = stage
    versions = [None if tensor.is_inference() else tensor._version for tensor in given]
    before = _state(torch, model, stages, components)
    before["arguments"] = _arguments(arguments)
    try:
        with _alone(ranks, stages, home, what):
            result = run()
    except BaseException as exc:
        with contextlib.suppress(Exception):
            ranks.cross({"op": "failed", "why": f"{type(exc).__name__}: {exc}"[:600]}, home, what)
        raise
    states = _streams(torch)
    after = _state(torch, model, stages, components)
    after["arguments"] = _arguments(arguments)
    left = _left(before, after)
    if left:
        # The others run it as this GPU did, from the streams' same state, and so does every GPU
        # next time.
        _WRITES.add(key)
        _EVERYWHERE[what] = left
        ranks.cross({"op": "everywhere", "left": left, "keep": True}, home, what)
        return result
    tensors: list[torch.Tensor] = []
    if not _crossable(result, tensors):
        _WRITES.add(key)
        _EVERYWHERE[what] = ""
        ranks.cross({"op": "everywhere", "left": "", "keep": True}, home, what)
        return result
    written = {
        str(i): t
        for i, (t, v) in enumerate(zip(given, versions, strict=True))
        if v is None or t._version != v
    }
    payload = {"op": "done", "result": result, "streams": states, "written": written}
    made = _made(payload, what)
    bulky = any(t.device.type == "cpu" and t.numel() for t in (*tensors, *written.values()))
    if bulky and made[2] is None:
        # No room for a shared mapping now: host tensors would broadcast through every card.
        _EVERYWHERE[what] = "no shared mapping for its host tensors"
        ranks.cross({"op": "everywhere", "left": "", "keep": False}, home, what)
        return result
    ranks.cross(payload, home, what, made)
    return result


def _arguments(arguments: tuple[tuple[object, ...], dict[str, object]]) -> object:
    """A stage's arguments as they cross, their tensors by identity: an entry a stage adds to
    a dict, list or struct argument, or a tensor it puts in one, shows."""
    tensors: list[torch.Tensor] = []
    structure = wire.marshal(arguments, tensors)
    return (json.dumps(structure, sort_keys=True), tuple(map(id, tensors)))


def _receive(
    ranks: Ranks,
    stages: Stages,
    home: int,
    what: str,
    run: Callable[[], object],
    given: list[torch.Tensor],
    key: str,
) -> object:
    import torch

    began = time.perf_counter()
    while True:
        told = cast("dict[str, object]", ranks.cross(None, home, what))
        op = told.get("op")
        if op == "spread":
            _serve_spread(ranks, stages, told)
            continue
        _row(what)["waited_ms"] += round((time.perf_counter() - began) * 1e3, 1)
        if op == "done":
            _set_streams(torch, cast("dict[str, object]", told["streams"]))
            with torch.inference_mode():
                for index, tensor in cast("dict[str, torch.Tensor]", told["written"]).items():
                    given[int(index)].copy_(tensor)
            return told["result"]
        if op == "everywhere":
            if told.get("keep"):  # as GPU 0 keeps it
                _WRITES.add(key)
            with _alone(ranks, stages, home, what, served=False):
                return run()
        why = f"{what}: {ranks.gpu(home)} failed: {told.get('why', '')}"
        raise GroupRefusal(why[:900], code=ENDED)


def _serve_spread(ranks: Ranks, stages: Stages, told: Mapping[str, object]) -> None:
    """A spread call of a stage another GPU runs alone: this GPU's share (`Ranks.spread`)."""
    root = stages.roots.get(str(told["component"]))
    method = getattr(root, str(told["method"]), None)
    if method is None:
        missing = f"spread: no {told['component']}.{told['method']} on this GPU"
        raise GroupRefusal(missing, code=ENDED)
    calls = [tuple(args) for args in cast("list[list[object]]", told["arguments"])]
    with _alone(ranks, stages, 0, "spread", served=False):  # the call alone, its rounds with all
        for _ in ranks.spread(method, calls):
            pass


def local() -> bool:
    """Whether this process's calls reach no other GPU of its group now: a stage running alone
    whose `spread` calls stay here (`_alone(served=False)`), as one GPU's hosted sharded call."""
    group = _group.CURRENT[0]
    return isinstance(group, _Alone) and not group.served


class _Alone:
    """The group as a stage running alone on one GPU sees it (`_group.CURRENT`): one GPU, but its
    `spread` calls go to the GPUs waiting for the stage's result."""

    def __init__(self, ranks: Ranks, stages: Stages, home: int, what: str, served: bool) -> None:
        self.ranks, self.stages, self.home, self.what = ranks, stages, home, what
        self.rank, self.world, self.served = ranks.rank, ranks.world, served

    def all(self, flag: bool, what: str) -> bool:
        return flag

    def spread[*Ts, R](
        self, method: Callable[[*Ts], R], arguments: Iterable[tuple[*Ts]]
    ) -> Iterator[R]:
        name = self.stages.name(getattr(method, "__self__", None))
        calls = list(arguments)
        # Here, one after another: no GPU waits for it, or its arguments cannot cross to them.
        if name is None or not self.served or not _crossable(calls, []):
            return (method(*call) for call in calls)
        told: dict[str, object] = {
            "op": "spread",
            "component": name,
            "method": getattr(method, "__name__", ""),
            "arguments": [list(cast("tuple[object, ...]", call)) for call in calls],
        }
        self.ranks.cross(told, self.home, self.what)
        return self.ranks.spread(method, calls)


#: CUDA's, then the host's default generator states, Python's and NumPy's.
def _streams(torch: ModuleType) -> dict[str, object]:
    version, internal, gauss = random.getstate()
    states: dict[str, object] = {
        "torch": torch.random.get_rng_state(),
        "python": [version, list(internal), gauss],
    }
    if torch.cuda.is_initialized():
        states["cuda"] = torch.cuda.get_rng_state()
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        kind, keys, pos, gauss, cached = numpy.random.get_state()
        states["numpy"] = [kind, keys.tolist(), int(pos), int(gauss), float(cached)]
    return states


def _set_streams(torch: ModuleType, states: Mapping[str, object]) -> None:
    torch.random.set_rng_state(cast("Tensor", states["torch"]).cpu())
    if "cuda" in states:
        torch.cuda.set_rng_state(cast("Tensor", states["cuda"]).cpu())
    version, internal, gauss = cast("list[object]", states["python"])
    random.setstate((version, tuple(cast("list[int]", internal)), gauss))
    numpy = sys.modules.get("numpy")
    if numpy is not None and "numpy" in states:
        kind, keys, pos, has, cached = cast("list[object]", states["numpy"])
        numpy.random.set_state((kind, numpy.asarray(keys, dtype=numpy.uint32), pos, has, cached))


_PLAIN = (str, bytes, int, float, bool, type(None))


def _shallow(torch: ModuleType, value: object) -> object:
    """A value as `_state` compares it: plain values and their tuples by value, a container by
    its items' identities (an entry added to a cache dict shows), a tensor by identity and
    version (an inference tensor, which keeps none, by its storage), a generator by its state,
    anything else by identity."""
    if isinstance(value, _PLAIN):
        return (type(value), value)
    if isinstance(value, tuple) and all(isinstance(item, _PLAIN) for item in value):
        return value
    if isinstance(value, dict):
        return ("dict", id(value), tuple((repr(k), id(v)) for k, v in value.items()))
    if isinstance(value, (list, set)):
        return (type(value).__name__, id(value), tuple(sorted(map(id, value))))
    if isinstance(value, torch.Tensor):
        mark = value.data_ptr() if value.is_inference() else value._version
        return ("tensor", id(value), mark)
    if isinstance(value, torch.Generator):
        return ("generator", id(value), bytes(value.get_state().numpy()))
    return ("id", id(value))


def _state(
    torch: ModuleType, model: object, stages: Stages, components: tuple[str, ...]
) -> dict[str, object]:
    """What a stage could leave behind beside its result: the model's own attributes, every
    object they reach that is not a module (a tokenizer's `padding_side`, a cache inside a
    helper, each once), its components' attributes, and their buffers' versions. Runtime state
    (`_cozy…`) aside."""
    found: dict[str, object] = {}
    seen: set[int] = {id(model)}
    #: objects still to read, each with its path (a stack: no recursion limit on deep state)
    todo: list[tuple[str, object]] = []

    def attributes(prefix: str, owner: object, skip: tuple[str, ...] = ()) -> None:
        for name, value in list(getattr(owner, "__dict__", {}).items()):
            if not name.startswith("_cozy") and name not in skip:
                found[prefix + name] = _shallow(torch, value)
                todo.append((f"{prefix}{name}.", value))

    attributes("", model)
    while todo:
        prefix, value = todo.pop()
        if isinstance(value, (torch.nn.Module, torch.Tensor, torch.Generator, type, *_PLAIN)):
            continue
        if id(value) in seen:
            continue
        seen.add(id(value))
        items: Iterable[tuple[str, object]] = ()
        if isinstance(value, dict):
            items = ((f"[{k!r}]", v) for k, v in list(value.items()))
        elif isinstance(value, (list, tuple)):
            items = ((f"[{i}]", v) for i, v in enumerate(value))
        elif hasattr(value, "__dict__"):
            attributes(prefix, value)
        for at, item in items:  # a container's plain items are its own entry's
            if not isinstance(item, _PLAIN):
                found[prefix + at] = _shallow(torch, item)
                todo.append((f"{prefix}{at}.", item))
    for name in components:
        root = stages.roots.get(name)
        if isinstance(root, torch.nn.Module):
            attributes(f"{name}.", root, ("_parameters", "_buffers", "_modules"))
            for path, buffer in root.named_buffers():
                found[f"{name}.{path}"] = None if buffer.is_inference() else buffer._version
    return found


def _left(before: Mapping[str, object], after: Mapping[str, object]) -> str:
    """The state a stage left changed, named, or empty."""
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    return ", ".join(changed)[:300]


#: By crossing, this attempt: the host bytes that went through a shared mapping, the ms the GPU
#: holding them took to share them and each other GPU to map them, and the ms a GPU waited for a
#: stage another GPU ran alone (`place`).
_CROSSED: dict[str, dict[str, float]] = {}
#: Stages every GPU ran after one ran alone, and why (`_left`): what they left beside their result.
_EVERYWHERE: dict[str, str] = {}


def _row(what: str) -> dict[str, float]:
    fields = ("bytes", "share_ms", "map_ms", "waited_ms")
    return _CROSSED.setdefault(what, dict.fromkeys(fields, 0.0))


def taken() -> dict[str, dict[str, float | str]]:
    """This GPU's crossings since the last take (`cozy run show`: `transport.stages`), each
    stage that fell back to every GPU, with what it left (`everywhere`), and each sharded stage's
    degree key, its record's name (`key`)."""
    found: dict[str, dict[str, float | str]] = {
        name: {k: v for k, v in row.items() if v} for name, row in _CROSSED.items()
    }
    for name, why in _EVERYWHERE.items():
        found.setdefault(name, {})["everywhere"] = why or "its result cannot cross"
    for name, digest in _CHOSEN.items():
        found.setdefault(name, {})["key"] = digest
    for name, why in _SKIPPED.items():
        found.setdefault(name, {})["skipped_degree"] = why
    _CROSSED.clear()
    _EVERYWHERE.clear()
    _CHOSEN.clear()
    _SKIPPED.clear()
    return found


def _layout(specs: Iterable[wire._Tensor]) -> dict[int, tuple[int, int]]:
    """Each host tensor's (offset, bytes) in a crossing's mapping, from its envelope alone, so the
    GPU that writes it and those that read it agree; 64-byte aligned."""
    import torch

    at, found = 0, {}
    for spec in specs:
        if spec.host:
            size = getattr(torch, spec.dtype).itemsize
            for n in spec.shape:
                size *= n
            found[spec.v] = (at, size)
            at += -(-size // 64) * 64
    return found


def _admit(nbytes: int) -> str:
    """The machine's host admission for `nbytes` of plain host memory (TensorFS's, W5): empty
    when admitted; else why not, in one line. Admitting reserves nothing: the pages count by the
    kernel's charge from `posix_fallocate`, which follows at once, until their last view goes."""
    try:
        from tensorfs import _ext
    except ImportError:
        return ""
    admit = getattr(_ext, "_varena_host_admit", None)  # absent before TensorFS carries W5
    refused = admit(nbytes) if admit is not None else None
    return str(refused or "")


class _Shared:
    """A crossing's host tensors in one shared mapping (`/dev/shm`): admitted by the machine's
    host admission (`_admit`), written once by the GPU that holds them, mapped copy-on-write by
    each other GPU, which reads it in place. Its name goes once every GPU has mapped it, and its
    pages (counted by the kernel's charge from `posix_fallocate`) when their last view does."""

    def __init__(self, name: str, page: mmap.mmap) -> None:
        self.name, self.page = name, page

    @classmethod
    def make(cls, tensors: list[torch.Tensor], what: str) -> _Shared | None:
        import torch

        host = {v: t for v, t in enumerate(tensors) if t.device.type == "cpu" and t.numel()}
        if not host:
            return None
        began = time.perf_counter()
        specs = [
            wire._Tensor(v, str(t.dtype).removeprefix("torch."), list(t.shape), True)
            for v, t in host.items()
        ]
        layout = _layout(specs)
        size = max(at + n for at, n in layout.values())
        if _admit(size):
            return None  # no room on the host now: every tensor broadcasts
        name = f"/dev/shm/cozy-cross-{os.getpid()}-{secrets.token_hex(8)}"
        try:
            fd = os.open(name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except OSError:
            return None  # no shared memory here: every tensor broadcasts
        try:
            os.posix_fallocate(fd, 0, size)  # a short /dev/shm refuses here, never with a SIGBUS
            page = mmap.mmap(fd, size)
        except OSError:
            os.unlink(name)
            return None
        finally:
            os.close(fd)
        for v, tensor in host.items():
            at, n = layout[v]
            into = torch.frombuffer(page, dtype=torch.uint8, count=n, offset=at)
            into.copy_(tensor.detach().contiguous().reshape(-1).view(torch.uint8))
            del into
        row = _row(what)
        row["bytes"] += size
        row["share_ms"] += round((time.perf_counter() - began) * 1e3, 1)
        return cls(name, page)

    @staticmethod
    def views(name: str, specs: list[wire._Tensor], what: str) -> dict[int, torch.Tensor]:
        """Each host tensor of the crossing as a view of its mapping, copy-on-write: a GPU that
        writes one writes its own pages."""
        import torch

        began = time.perf_counter()
        layout = _layout(specs)
        fd = os.open(name, os.O_RDWR)
        try:
            size = os.fstat(fd).st_size
            page = mmap.mmap(fd, size, flags=mmap.MAP_PRIVATE)
        finally:
            os.close(fd)
        found: dict[int, torch.Tensor] = {}
        for spec in specs:
            if not spec.host:
                continue
            dtype = getattr(torch, spec.dtype)
            at, n = layout[spec.v]
            raw = (
                torch.frombuffer(page, dtype=torch.uint8, count=n, offset=at)
                if n
                else torch.empty(0, dtype=torch.uint8)
            )
            found[spec.v] = raw.view(dtype).view(spec.shape)
        row = _row(what)
        row["bytes"] += size
        row["map_ms"] += round((time.perf_counter() - began) * 1e3, 1)
        return found

    def done(self) -> None:
        with contextlib.suppress(OSError):
            os.unlink(self.name)
        with contextlib.suppress(BufferError):
            self.page.close()


def _shape_class(
    module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
) -> str:
    """A sharded call's shape class: the state it computes on (its forward's first argument: a
    DiT's latent rows, the video) by its exact shape, and every other tensor with each dimension
    raised to the next power of two: the conditioning, whose length a prompt sets (H3 packs its
    text rows into the sequence the DiT attends over), so a video's shape chooses and a prompt's
    length within a power of two does not. No other value is any class of its own: a seed, a
    step's timestep, a scale."""
    try:
        bound = dict(
            inspect.signature(type(module).forward).bind(module, *args, **kwargs).arguments
        )
    except (TypeError, ValueError):
        bound = {"self": module, **{str(i): a for i, a in enumerate(args)}, **kwargs}
    names = list(bound)[1:]
    return ",".join(f"{n}={_shape(bound[n], exact=i == 0)}" for i, n in enumerate(names))


def _shape(value: object, *, exact: bool) -> str:
    import torch

    if isinstance(value, torch.Tensor):
        dims = (
            list(value.shape)
            if exact
            else [1 << (n - 1).bit_length() if n else 0 for n in value.shape]
        )
        return f"{str(value.dtype).removeprefix('torch.')}{dims}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_shape(item, exact=exact) for item in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items())
        return "{" + ",".join(f"{k}:{_shape(v, exact=exact)}" for k, v in items) + "}"
    return type(value).__name__


def _runtime() -> str:
    from cozy_runtime import __version__

    try:
        from cozy_runtime import _build_provenance

        return f"{__version__}+{getattr(_build_provenance, 'COMMIT', '')}"
    except ImportError:
        return __version__


#: Per thread: the sharded stage running, whose first sharded call chooses its degree (`_Choice`).
_CHOICE = threading.local()
#: By stage row, this attempt: its degree key's digest (`degrees.Key.digest`), its record's name.
_CHOSEN: dict[str, str] = {}
#: By stage row, this attempt: why its group's degree was not tried (`degrees.Decision.skipped`).
_SKIPPED: dict[str, str] = {}
#: Sharded stages whose sharded call's output once could not cross: at one GPU every GPU runs them.
_UNHOSTED: set[str] = set()


def _degree(ranks: Ranks, stages: Stages, stage: str, run: Callable[[], object]) -> object:
    """`run`, a sharded stage, at the degree its first sharded call chooses (`_Choice`)."""
    # A stage called again in one attempt has its own key (its shape class) and its own row.
    method = stage.rpartition(".")[2]
    ranks.calls[method] = nth = ranks.calls.get(method, 0) + 1
    choice = _Choice(
        ranks, stages, stage, f"stage {method}" if nth == 1 else f"stage {method} {nth}"
    )
    before, _CHOICE.stage = getattr(_CHOICE, "stage", None), choice
    try:
        with choice.scope:
            result = run()
            choice.ended()
    finally:
        _CHOICE.stage = before
    choice.finish()
    return result


class _Choice:
    """A sharded stage's degree: its group's, or one GPU, by its key's records (`degrees.choose`,
    GPU 0's word for every GPU), chosen at the stage's first sharded call, where every GPU meets
    anyway. The key's shape class is that call's (`_shape_class`): a request's video chooses, and
    its seed, its words or its prompt's length within a power of two do not.

    At one GPU, one GPU runs each sharded call unsharded and its output crosses (`hosted_call`):
    GPU 0, unless its stated room cannot hold the stage's need and another's can (`_home`). The
    others run the rest of the stage's program on it, so its random streams, generators and
    callbacks stay theirs, and compute, fetch and hold none of the call. A stage whose output once
    could not cross runs unsharded on every GPU where it fits there (`_fits`), else at the group's.

    Measuring, the GPU running them times each sharded call (steps 2 on: step 1 carries fills and
    compiles); a
    step in which a GPU read weights from disk or saw a kernel build or land makes the record cold
    (`_cold`). Trying the group's degree, its second step is compared with one GPU's steady step at
    the third call, and a slower one moves the rest of the stage to one GPU (`_trial`)."""

    def __init__(self, ranks: Ranks, stages: Stages, stage: str, what: str) -> None:
        self.ranks, self.stages, self.stage, self.what = ranks, stages, stage, what
        self.key: object = None
        self.degree = ranks.world
        self.measure = False
        #: one GPU's steady step the group's degree is tried against (`degrees.Decision`)
        self.against_ms: float | None = None
        #: at one GPU: the GPU `home` runs each sharded call and its output crosses
        self.hosted = False
        self.home = 0
        #: what the stage runs inside from its first sharded call to its end, at degree 1
        self.scope = contextlib.ExitStack()
        #: the timing GPU's (`_timer`), measuring: an event at each sharded call's start, then the
        #: stage's end
        self.events: list[torch.cuda.Event] = []
        #: the timing GPU's, measuring one GPU: the weight bytes filled onto it at each event
        self.filled: list[int] = []
        self.calls = 0
        #: this GPU's `_cold` at the steady steps' start and at the stage's end
        self.cold: list[tuple[int, int, bool, int]] = []

    def called(
        self,
        module: torch.nn.Module,
        name: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        self.calls += 1
        if self.calls == 1:
            # GPU 0's key is every GPU's (`_choose`)
            self._choose(
                f"{name}({_shape_class(module, args, kwargs)})" if self.ranks.rank == 0 else ""
            )
        elif self.calls == 2 and self.measure:
            self.cold.append(_cold(self.stages))
        elif self.calls == 3 and self.measure and self.against_ms is not None:
            self._trial()
        self._mark()

    def _choose(self, shape: str) -> None:
        import torch

        from cozy_runtime.internal.parallel import degrees

        ranks = self.ranks
        told: dict[str, Json] = {}
        if ranks.rank == 0:
            card = torch.cuda.get_device_name(ranks.device) if torch.cuda.is_available() else "cpu"
            key = degrees.Key(*ranks.call, self.stage, shape, card, ranks.world, _runtime())
            # Nothing is measured on a CPU: its group's degree throughout.
            decision = (
                degrees.choose(key, ranks.route)
                if torch.cuda.is_available()
                else degrees.Decision(ranks.world)
            )
            told = {
                "key": cast("dict[str, Json]", key.document()),
                "degree": decision.degree,
                "measure": decision.measure,
                "against_ms": decision.against_ms,
                "skipped": decision.skipped,
            }
        told = dict(ranks.told(told, f"degree {self.stage}"))
        key = degrees.Key.of(cast("dict[str, Json]", told["key"]))
        self.key, self.degree, self.measure = (
            key,
            int(cast("int", told["degree"])),
            bool(told["measure"]),
        )
        against = told.get("against_ms")
        self.against_ms = None if against is None else float(cast("float", against))
        _CHOSEN[self.what] = key.digest()
        if told.get("skipped"):
            _SKIPPED[self.what] = str(told["skipped"])
        if self.degree == 1 < ranks.world:
            self._one()

    def _one(self) -> None:
        """The rest of the stage at one GPU."""
        from cozy_runtime.internal.parallel import cp

        ranks = self.ranks
        roots = [self.stages.roots[name] for name in sorted(self.stages.sharded)]
        if self.stage in _UNHOSTED and not ranks.all(
            _fits(self.stages, self.key), f"degree 1 fits {self.stage}"
        ):
            self.degree, self.measure = ranks.world, False
            _row(self.what)["refused_degree_1"] = 1.0
            return
        self.scope.enter_context(cp.unsharded(cast("list[torch.nn.Module]", roots)))
        if self.stage in _UNHOSTED:
            self._everywhere()
        else:
            self.hosted, self.home = True, self._home(roots)
            _row(self.what)["hosted"] = 1.0
            _row(self.what)["home"] = float(self.home)

    def _home(self, roots: list[object]) -> int:
        """The GPU that runs the stage's sharded calls at one GPU: GPU 0 where its stated room (the
        card less what other processes hold, `vt.room`) holds the stage's need, its sharded weights
        and one GPU's recorded working peak; else the GPU with the most room, the lowest on a tie.
        Another executor may hold much of a card (a machine serving one-GPU and group requests)."""
        import tensorfs.varena.torch as vt
        import torch

        from cozy_runtime.internal.parallel import degrees

        need = None
        if self.ranks.rank == 0:
            row = degrees.known(cast("degrees.Key", self.key)).get(1, {})
            need = int(cast("int", row.get("peak_bytes", 0))) + sum(
                p.numel() * p.element_size()
                for root in roots
                for p in cast("torch.nn.Module", root).parameters()
            )
        room = int(vt.room(torch.cuda.current_device())["capacity"])
        rows = self.ranks.meet({"room": room, "need": need}, f"home {self.stage}")
        rooms = [int(cast("int", r["room"])) for r in rows]
        if rooms[0] >= int(cast("int", rows[0]["need"])):
            return 0
        return max(range(len(rooms)), key=lambda r: (rooms[r], -r))

    def _timer(self) -> int:
        """The GPU whose stream runs the stage's sharded calls: the home at one GPU, else GPU 0."""
        return self.home if self.hosted else 0

    def _everywhere(self) -> None:
        """Every GPU runs the rest of the stage unsharded on its own, nothing crossing."""
        self.hosted = False
        self.scope.enter_context(_alone(self.ranks, self.stages, 0, self.what, served=False))
        self.scope.enter_context(_unsharded())

    def hosted_call(
        self, module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> object:
        """One sharded call at one GPU: its home runs it unsharded, and its output crosses."""
        ranks, home, what = self.ranks, self.home, f"{self.what} hosted"
        if ranks.rank != home:
            told = cast("dict[str, object]", ranks.cross(None, home, what))
            if told.get("op") == "done":
                return told["out"]
            if told.get("op") != "everywhere":
                why = f"{what}: {ranks.gpu(home)} failed: {told.get('why', '')}"
                raise GroupRefusal(why[:900], code=ENDED)
            self._unhosted()
            return type(module)._call_impl(module, *args, **kwargs)
        try:
            with _alone(ranks, self.stages, home, what, served=False), _unsharded():
                out = type(module)._call_impl(module, *args, **kwargs)
        except BaseException as exc:
            with contextlib.suppress(Exception):
                why = f"{type(exc).__name__}: {exc}"[:600]
                ranks.cross({"op": "failed", "why": why}, home, what)
            raise
        if not _crossable(out, []):
            ranks.cross({"op": "everywhere"}, home, what)
            self._unhosted()
            return out
        ranks.cross({"op": "done", "out": out}, home, what)
        return out

    def _unhosted(self) -> None:
        _UNHOSTED.add(self.stage)
        _EVERYWHERE[self.what] = "its sharded call's output cannot cross"
        self._everywhere()

    def _trial(self) -> None:
        """The third sharded call trying the group's degree: a second step slower than one GPU's
        steady step moves the rest of the stage to one GPU, and that step is the group's record."""
        from cozy_runtime.internal import attention_ulysses
        from cozy_runtime.internal.parallel import degrees

        step = None
        if self.ranks.rank == 0:
            self._mark()  # the second step's end
            self.events[2].synchronize()
            step = round(self.events[1].elapsed_time(self.events[2]), 2)
            self.events.pop()  # `called` marks this call's start
        now = _cold(self.stages)
        warm = len(self.cold) == 1 and self.cold[0] == now and not now[2]
        rows = self.ranks.meet({"peak": _peak(), "warm": warm, "step": step}, f"trial {self.stage}")
        second = float(cast("float", rows[0]["step"]))
        _row(self.what)["trial_ms"] = second
        if second <= cast("float", self.against_ms):
            return
        kept: dict[str, object] = {
            "steady_ms": second,
            "trial": True,
            "peak_bytes": max(int(cast("int", r["peak"])) for r in rows),
            "kernels": attention_ulysses.served(),
        }
        if not all(r["warm"] for r in rows):
            kept["cold"] = True
        if self.ranks.rank == 0:
            degrees.record(cast("degrees.Key", self.key), self.degree, kept)
        self.degree, self.measure = 1, False
        self._one()

    def _mark(self) -> None:
        if self.measure and self.ranks.rank == self._timer():
            import torch

            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.append(event)
            if self.degree == 1:
                self.filled.append(_filled())

    def ended(self) -> None:
        """The stage's end, inside it."""
        if self.measure:
            self._mark()
            self.cold.append(_cold(self.stages))

    def _steady_ms(self) -> float | None:
        """Mean ms of steps 2 on (step 1 carries fills and compiles); None under two steps."""
        events = self.events
        if len(events) < 3:
            return None
        events[-1].synchronize()
        spans = [a.elapsed_time(b) for a, b in itertools.pairwise(events[1:])]
        return round(sum(spans) / len(spans), 2)

    def finish(self) -> None:
        """The stage's row, and, measuring, its steady step recorded, cold or not (`degrees`)."""
        from cozy_runtime.internal import attention_ulysses
        from cozy_runtime.internal.parallel import degrees

        if not isinstance(self.key, degrees.Key):
            return  # it made no sharded call
        row = _row(self.what)
        row["degree"] = float(self.degree)
        if not self.measure:
            return
        timer = self._timer()
        mine = self.ranks.rank == timer
        steady = self._steady_ms() if mine else None
        # the weight bytes filled onto the timing GPU over its last sharded call (`degrees.choose`)
        paged = self.filled[-1] - self.filled[-2] if mine and len(self.filled) > 1 else None
        warm = len(self.cold) == 2 and self.cold[0] == self.cold[1] and not self.cold[1][2]
        told = {"peak": _peak(), "warm": warm, "steady": steady, "paged": paged, "kernels": None}
        if mine:
            told["kernels"] = cast("dict[str, Json]", attention_ulysses.served())
        rows = self.ranks.meet(told, f"peak {self.stage}")
        if rows[timer]["steady"] is None:
            return
        row["steady_ms"] = steady = float(cast("float", rows[timer]["steady"]))
        # At one GPU only its home computes the stage's sharded calls.
        timed = [rows[timer]] if self.hosted else rows
        kept: dict[str, object] = {
            "steady_ms": steady,
            "peak_bytes": max(int(cast("int", r["peak"])) for r in timed),
            "kernels": rows[timer]["kernels"],
        }
        if not all(r["warm"] for r in timed):
            row["cold"] = 1.0
            kept["cold"] = True
        if self.degree == 1 and rows[timer]["paged"] is not None:
            kept["paged_bytes"] = row["paged_bytes"] = float(cast("float", rows[timer]["paged"]))
        if self.ranks.rank == 0:
            degrees.record(self.key, self.degree, kept)


@contextlib.contextmanager
def _unsharded() -> Iterator[None]:
    _ALONE.unsharded = True
    try:
        yield
    finally:
        _ALONE.unsharded = False


def _cold(stages: Stages) -> tuple[int, int, bool, int]:
    """What makes a step not steady on this GPU, so far in its request: the weight bytes read
    from disk (the store, a derived page; a page kept in host memory is a tight card's steady
    paging), the kernels' changes (`LANDED`), whether a kernel is still building or landed and
    not serving yet, and the graphs its sharded components tried to record."""
    from cozy_runtime import graphs
    from cozy_runtime.author._model import LANDED
    from cozy_runtime.internal import kernel_ready
    from cozy_runtime.internal.execution import catch

    disk = 0
    request = catch.CURRENT.get()
    if request is not None and request.memory.pages is not None and not request.idle:
        request.record_fills()
        disk = sum(request.moved.get(origin, (0,))[0] for origin in ("store", "derived"))
    building = any(at in ("opening", "landed") for at in kernel_ready.pending().values())
    roots = (stages.roots[name] for name in sorted(stages.sharded))
    captured = sum(graphs.capture_attempts(cast("torch.nn.Module", root)) for root in roots)
    return disk, LANDED[0], building, captured


def _filled() -> int:
    """The weight bytes filled onto this GPU so far in its request, from anywhere but its own
    resident pages."""
    from cozy_runtime.internal.execution import catch

    request = catch.CURRENT.get()
    if request is None or request.memory.pages is None or request.idle:
        return 0
    request.record_fills()
    return sum(moved[0] for origin, moved in request.moved.items() if origin != "resident")


def _peak() -> int:
    """This GPU's working high-water in the request so far (`accel.allocator_peak`): at least the
    stage's, so a degree-1 need taken from it errs large."""
    from cozy_runtime.internal import accel
    from cozy_runtime.internal.execution import catch

    request = catch.CURRENT.get()
    return accel.allocator_peak(request.memory, request.id) if request is not None else 0


def _fits(stages: Stages, key: object) -> bool:
    """Whether a stage of `key` fits unsharded on this GPU: its sharded components' weights and
    the group's degree times the working peak a GPU took at that degree (attention and activations
    scale with a GPU's share of the sequence) within the card's capacity. Nothing measured: no."""
    import tensorfs.varena.torch as vt
    import torch

    from cozy_runtime.internal.parallel import degrees

    assert isinstance(key, degrees.Key)
    row = degrees.known(key).get(key.world, {})
    need = int(cast("int", row.get("peak_bytes", 0)))
    if not need or not torch.cuda.is_available():
        return False
    weights = sum(
        p.numel() * p.element_size()
        for name in stages.sharded
        for p in cast("torch.nn.Module", stages.roots[name]).parameters()
    )
    capacity = int(vt.room(torch.cuda.current_device())["capacity"])
    return weights + key.world * need <= capacity
