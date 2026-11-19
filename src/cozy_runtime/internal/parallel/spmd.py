"""Every GPU of a group runs the attempt's handler, as torchrun runs a script on every rank.

The same request, seeds and random streams make every GPU compute the same per-step state with
the same code; a `_cp_plan` component splits and gathers inside its own call (Diffusers' hooks).
What one GPU alone decides or holds crosses here, always at the same point of every GPU's
program: a store meeting (`Ranks.meet`) that also says which GPUs are not at the same call, then
one broadcast per tensor from the GPU that holds it.

A stage (`uses_components`) none of whose components is sharded runs on one GPU (`place`): its
components' home, else GPU 0. Its result and the random streams' state after it cross to the
others, which fetch, hold and compute nothing for it; while they wait they serve its `spread`
calls. A stage with a sharded component runs on every GPU.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import json
import mmap
import os
import random
import secrets
import sys
import threading
import time
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
    from torch.distributed import ProcessGroup
    from torch.utils.hooks import RemovableHandle

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
    ) -> None:
        import torch.distributed as dist
        from torch._C._distributed_c10d import PrefixStore
        from torch.distributed.distributed_c10d import _get_process_group_store

        self.rank, self.world, self.pg, self.device, self.gpu = rank, world, pg, device, gpu
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
    (`kernel_ready.landed`): a sparse kernel built during the attempt serves its next steps."""
    if getattr(component, "_cozy_spmd_check", None) is not None:
        return

    def check(module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        from cozy_runtime.internal import kernel_ready

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

    handle: RemovableHandle = component.register_forward_pre_hook(check, with_kwargs=True)
    with contextlib.suppress(Exception):
        object.__setattr__(component, "_cozy_spmd_check", handle)


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
    if home is None or key in _WRITES or _model.open_stage():
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
        told = {
            "op": "spread",
            "component": name,
            "method": getattr(method, "__name__", ""),
            "arguments": [list(call) for call in calls],
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
    torch.random.set_rng_state(cast("torch.Tensor", states["torch"]).cpu())
    if "cuda" in states:
        torch.cuda.set_rng_state(cast("torch.Tensor", states["cuda"]).cpu())
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
    """This GPU's crossings since the last take (`cozy run show`: `transport.stages`), and each
    stage that fell back to every GPU, with what it left (`everywhere`)."""
    found: dict[str, dict[str, float | str]] = {
        name: {k: v for k, v in row.items() if v} for name, row in _CROSSED.items()
    }
    for name, why in _EVERYWHERE.items():
        found.setdefault(name, {})["everywhere"] = why or "its result cannot cross"
    _CROSSED.clear()
    _EVERYWHERE.clear()
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
