"""Executor-local weight residency over ordinary Torch allocations.

TensorFS fills Torch-owned destinations. Runtime declares immutable weight groups and their
fill transforms; hooks bind those groups while used. Inactive groups may be discarded at a
preparation boundary. No allocator hook, VMM mapping, shared page or package replay lives here.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import tensorfs
import torch
from tensorfs.raw import parts as stored_parts
from torch import Tensor, nn
from torch.utils._pytree import tree_leaves, tree_map

__all__ = [
    "Binding",
    "Declared",
    "Hold",
    "Level",
    "Memory",
    "Model",
    "Region",
    "Request",
    "ResidencyError",
    "Room",
    "StaleBinding",
    "Weights",
    "blocks_of",
]

progress = 0
_RESERVE = 512 << 20
_DTYPES = {
    name: kind
    for name, kind in (
        pair.split("=")
        for pair in [
            "F64=float64",
            "F32=float32",
            "F16=float16",
            "BF16=bfloat16",
            "F8_E4M3=float8_e4m3fn",
            "F8_E5M2=float8_e5m2",
            "F8_E8M0=float8_e8m0fnu",
            "I64=int64",
            "I32=int32",
            "I16=int16",
            "I8=int8",
            "U8=uint8",
            "BOOL=bool",
            "F4=uint8",
        ]
    )
}
_REQUEST: ContextVar[Request | None] = ContextVar("cozy_residency_request", default=None)
_MEMORIES: weakref.WeakValueDictionary[str, Memory] = weakref.WeakValueDictionary()
_OPENING = threading.Lock()
# A failed graph completion must retain the actual storage owners until executor exit.
_UNPROVEN: list[Binding] = []


class ResidencyError(RuntimeError):
    code = "RESIDENCY"
    sticky = False
    cuda_code = None


class StaleBinding(ResidencyError):
    """A cached graph must retire before these weights can change storage or fill."""

    code = "STALE_BINDING"


class Level(Enum):
    RAM = "ram"
    GPU = "gpu"


@dataclass(frozen=True, slots=True)
class Room:
    free: int
    cold: int
    warm: int
    granularity: int = 512
    floor: int = 0


@dataclass(frozen=True, slots=True)
class Weights:
    ck: tensorfs.Checkpoint
    component: str
    release: str = ""
    shard: str = "0/1"
    keys: Collection[str] | None = None
    bound: Mapping[str, str] | None = None


class Filling(Protocol):
    @property
    def identity(self) -> str | None: ...

    def __call__(self, block: str, parts: Mapping[str, Mapping[str, Tensor]], /) -> None: ...


@dataclass(frozen=True, slots=True)
class Declared:
    weights: Weights
    blocks: Sequence[str] | None = None
    fill: Filling | None = None
    bind: Callable[[str, Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None


class Placeholder(Tensor):  # type: ignore[misc]  # Torch is optional in the checking environment.
    """Metadata between binds; a real read obtains ordinary owning Torch storage."""

    @staticmethod
    def __new__(cls: Any, shape: Sequence[int], dtype: Any, device: Any) -> Any:
        return Tensor._make_wrapper_subclass(
            cls, tuple(shape), dtype=dtype, device=device, storage_size=0
        )

    def __repr__(self) -> str:
        return f"UnboundWeight({tuple(self.shape)}, {self.dtype}, {self.device})"

    def _varena_read(self) -> Tensor:
        raise ResidencyError("this weight has no declared source")

    @classmethod
    def __torch_dispatch__(
        cls: Any, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        aten, kwargs = torch.ops.aten, kwargs or {}
        if func in (aten.detach.default, aten.alias.default):
            return args[0]
        if func is aten._to_copy.default:
            source = args[0]
            dtype = kwargs.get("dtype") or source.dtype
            device = kwargs.get("device") or source.device
            made = cls(source.shape, dtype, device)
            read = source._varena_read
            made._varena_read = lambda: read().to(device=device, dtype=dtype)
            return made
        return func(*tree_map(_readable, args), **tree_map(_readable, kwargs))


def _readable(value: Any) -> Any:
    return value._varena_read() if isinstance(value, Placeholder) else value


def blocks_of(module: nn.Module) -> list[str]:
    declared = vars(module).get("_cozy_weights")
    if isinstance(declared, Declared) and declared.blocks is not None:
        return list(declared.blocks)
    names = set(getattr(module, "_no_split_modules", None) or ())
    found = [
        path
        for path, sub in module.named_modules()
        if path and any(kind.__name__ in names for kind in type(sub).__mro__)
    ]
    return [a for a in found if not any(b.startswith(a + ".") for b in found)]


@dataclass(frozen=True, slots=True)
class _Site:
    table: dict[str, Any]
    name: str
    parameter: bool


@dataclass(frozen=True, slots=True)
class _Part:
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    role: str


@dataclass(eq=False, slots=True)
class _Block:
    model: Model
    path: str
    index: int
    parts: dict[str, dict[str, _Part]] = field(default_factory=dict)
    data: dict[str, dict[str, Tensor]] | None = None
    assigned: list[tuple[_Site, Tensor]] | None = None
    saved: list[tuple[_Site, Any]] = field(default_factory=list)
    count: int = 0
    retained: int = 0
    loading: bool = False
    generation: int = 0
    last: int = 0
    ready: Any = None

    @property
    def nbytes(self) -> int:
        return sum(part.nbytes for parts in self.parts.values() for part in parts.values())

    def tensors(self) -> Iterator[Tensor]:
        if self.data is not None:
            yield from (tensor for parts in self.data.values() for tensor in parts.values())
        if self.assigned is not None:
            yield from (tensor for _, tensor in self.assigned)


class Model:
    def __init__(self, memory: Memory, module: nn.Module, declared: Declared) -> None:
        self.memory, self.module, self.source = memory, module, declared.weights
        self.component = declared.weights.component
        self.name = f"{self.source.release}/{self.source.ck.id}/{self.component}"
        self.fill, self.bind = declared.fill, declared.bind
        self.identity = self.wanted_identity()
        self.generation = 0
        self.local = threading.local()
        self.users = 0
        self.thread: int | None = None
        self.retiring = False
        self.retire_graphs: Callable[[], None] | None = None
        self.sites = _sites(module)
        self.handles: list[Any] = []
        self.holds: list[Hold] = []
        self.blocks: dict[str, _Block] = {}
        self.common: _Block | None = None
        self.closed = False

    def wanted_identity(self) -> str | None:
        return "stored" if self.fill is None else self.fill.identity

    def _claim(self) -> None:
        # Runtime's component scopes normally serialize these uses. Refuse an undeclared
        # concurrent caller instead of swapping one module's slots beneath its forward.
        with self.memory.lock:
            me = threading.get_ident()
            if self.closed or self.retiring:
                raise ResidencyError("the model is closed or retiring its cached graphs")
            if self.users and self.thread != me:
                raise ResidencyError("concurrent model use requires separate executor storage")
            self.users += 1
            self.thread = me

    def _release(self) -> None:
        with self.memory.lock:
            self.users -= 1
            if not self.users:
                self.thread = None

    def groups(self) -> list[_Block]:
        return [b for b in (self.common, *self.blocks.values()) if b is not None]

    @contextmanager
    def passing(self, key: str = "") -> Iterator[Model]:
        self.memory._open(self, key)
        try:
            yield self
        finally:
            self.memory._close(self)

    def retain(self, paths: Collection[str] = ("",)) -> Binding:
        self._claim()
        try:
            self.memory._declare(self)
            selected = [
                b
                for b in self.groups()
                if not b.path
                or any(
                    not p or b.path == p or b.path.startswith(p + ".") or p.startswith(b.path + ".")
                    for p in paths
                )
            ]
            return Binding(self, selected)
        finally:
            self._release()

    def hold(
        self, shed: Callable[[], None] | None = None, *, free_only: bool = False
    ) -> Hold | None:
        self._claim()
        try:
            self.memory._declare(self)
            groups = self.groups()
            missing = sum(b.nbytes for b in groups if b.data is None)
            if free_only and missing > self.memory.room().free:
                if self.memory.cuda:
                    torch.cuda.empty_cache()
                if missing > self.memory.room().free:
                    self._release()
                    return None
            entered: list[_Block] = []
            try:
                for block in groups:
                    self.memory._enter(block, reclaim=not free_only)
                    entered.append(block)
            except torch.OutOfMemoryError:
                for block in reversed(entered):
                    self.memory._exit(block)
                if free_only:
                    self._release()
                    return None
                raise
            except BaseException:
                for block in reversed(entered):
                    self.memory._exit(block)
                raise
            hold = Hold(self, entered, shed, free_only)
            self.holds.append(hold)
            return hold
        except BaseException:
            self._release()
            raise

    def close(self) -> None:
        self.memory._retire(self)
        with self.memory.lock:
            if self.closed:
                return
            if self.users or any(b.count or b.retained for b in self.groups()):
                raise ResidencyError("retire graph bindings and active model uses before unload")
            self.closed = True
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            for block in self.groups():
                self.memory._discard(block)
            self.memory.models.pop(self.module, None)


class Hold:
    def __init__(
        self,
        model: Model,
        groups: list[_Block],
        shed: Callable[[], None] | None,
        reclaimable: bool,
    ) -> None:
        self.model, self.groups, self.on_shed = model, groups, shed
        self.reclaimable = reclaimable
        self.active = True

    def release(self) -> None:
        with self.model.memory.lock:
            if not self.active:
                return
            self.active = False
            groups, self.groups = self.groups, []
            if self in self.model.holds:
                self.model.holds.remove(self)
        try:
            for block in reversed(groups):
                self.model.memory._exit(block)
        finally:
            self.model._release()

    def shed(self) -> None:
        self.release()
        if self.on_shed is not None:
            self.on_shed()

    def __enter__(self) -> Hold:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class Binding:
    """A cached graph's owning storage, closed after its readers retire.

    An undeclared fill may prepare once for an ephemeral direct read. It never reports a
    valid/resident graph binding and cannot be replayed or prepared a second time.
    """

    def __init__(self, model: Model, groups: list[_Block]) -> None:
        self.model, self.groups = model, groups
        self.fill = model.wanted_identity()
        self.generation = model.generation
        self.identity = hashlib.sha256(
            repr((model.name, self.fill, self.generation, [b.index for b in groups])).encode()
        ).hexdigest()
        self.closed = False
        self.completion_lost = False
        self.used = False
        self.held: list[tuple[_Block, int]] = []
        self.lock = threading.RLock()
        self.last: Any = None

    def valid(self) -> bool:
        # A caller retires this graph before _declare changes the model's cached fill.
        return (
            not self.closed
            and not self.completion_lost
            and not self.model.closed
            and self.fill is not None
            and self.fill == self.model.wanted_identity()
            and self.generation == self.model.generation
            and all(
                b.data is not None and b.generation == generation for b, generation in self.held
            )
        )

    def resident(self) -> bool:
        return self.valid() and all(b.data is not None for b in self.groups)

    @contextmanager
    def prepare(self, stream: Any = None) -> Iterator[Binding]:
        memory = self.model.memory
        if memory.capturing():
            raise ResidencyError("prepare graph weights before capture")
        stream = memory.stream() if stream is None else stream
        with self.lock:
            self.model._claim()
            try:
                ephemeral = (
                    self.fill is None
                    and not self.used
                    and not self.closed
                    and not self.completion_lost
                    and self.model.wanted_identity() is None
                    and self.generation == self.model.generation
                )
                if not self.valid() and not ephemeral:
                    raise StaleBinding("graph weight storage or fill changed")
                request = _REQUEST.get()
                if request is not None and request.idle and not self.resident():
                    raise StaleBinding("idle graph preparation cannot page weights in")
                self.used = True
                with torch.cuda.stream(stream) if memory.cuda else nullcontext():
                    if self.last is not None:
                        stream.wait_event(self.last)
                    entered: list[_Block] = []
                    retained: list[tuple[_Block, int]] = []
                    published = False
                    try:
                        if not self.held:
                            # Claim the complete dependency set before any preparation. Neither
                            # another preparation nor a nested fill can evict its later blocks.
                            with memory.lock:
                                for block in self.groups:
                                    block.retained += 1
                                    retained.append((block, block.generation))
                        for block in self.groups:
                            memory._enter(block)
                            entered.append(block)
                        if not self.held:
                            self.held = [(block, block.generation) for block, _ in retained]
                            published = True
                        yield self
                    except BaseException:
                        if retained and not published:
                            with memory.lock:
                                for block, _ in retained:
                                    block.retained -= 1
                        raise
                    finally:
                        try:
                            if memory.cuda and entered:
                                self._finish(stream)
                        finally:
                            for block in reversed(entered):
                                memory._exit(block)
            finally:
                self.model._release()

    @contextmanager
    def replay_use(self, stream: Any = None) -> Iterator[Binding]:
        """Use already-captured storage without rebinding every module or touching its weights.

        The graph reads the retained allocations directly. Its complete launch/output-copy
        interval is serialized here, and close joins the final use before releasing storage.
        """
        memory = self.model.memory
        stream = memory.stream() if stream is None else stream
        with self.lock:
            self.model._claim()
            try:
                if not self.valid() or len(self.held) != len(self.groups):
                    raise StaleBinding("graph weight storage is not retained")
                if self.last is not None:
                    stream.wait_event(self.last)
                try:
                    yield self
                finally:
                    if memory.cuda:
                        self._finish(stream)
            finally:
                self.model._release()

    def _unproven(self, cause: BaseException) -> None:
        if not self.completion_lost:
            self.completion_lost = True
            _UNPROVEN.append(self)
        error = torch.AcceleratorError(f"graph completion is unproven; retire executor: {cause}")
        error.error_code = 999
        raise error from cause

    def _finish(self, stream: Any) -> None:
        try:
            event = torch.cuda.Event()
            event.record(stream)
            self.last = event
        except BaseException:
            try:
                torch.cuda.synchronize(self.model.memory.device)
            except BaseException as incomplete:
                self._unproven(incomplete)
            self.last = None
            raise

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            if self.completion_lost:
                self._unproven(ResidencyError("the previous completion failed"))
            if self.last is not None:
                try:
                    self.last.synchronize()
                except BaseException as incomplete:
                    self._unproven(incomplete)
            self.closed = True
            with self.model.memory.lock:
                for block, _ in self.held:
                    block.retained -= 1
                self.held.clear()


class Region:
    """An explicit owner's disposable cache; it never intercepts or remaps Torch storage."""

    def __init__(self, memory: Memory, owner: str, release: Callable[[], None] | None) -> None:
        self.memory, self.owner, self.release = memory, owner, release
        self.lock = threading.RLock()
        self.pins = 0
        self.fresh = False
        self.closed = False
        self.reclaiming = False
        self.last = time.monotonic_ns()

    @contextmanager
    def pin(self) -> Iterator[bool]:
        with self.lock:
            if self.closed or self.reclaiming:
                raise ResidencyError(f"cache {self.owner!r} is closed or being reclaimed")
            self.pins += 1
            fresh, self.fresh = self.fresh, False
        try:
            yield fresh
        finally:
            with self.lock:
                self.pins -= 1
                self.last = time.monotonic_ns()

    allocating = pin

    def discard(self) -> bool:
        if not self.lock.acquire(blocking=False):
            return False
        try:
            if self.closed or self.pins or self.reclaiming or self.release is None:
                return False
            self.reclaiming = True
            # No Memory metadata lock is held. The owner releases its ordinary tensors and
            # joins any exceptional consumers or cached graph outputs before dropping them.
            self.release()
            self.fresh = True
            return True
        finally:
            self.reclaiming = False
            self.lock.release()

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            if self.pins:
                raise ResidencyError(f"cache {self.owner!r} is still in use")
            self.closed = True
            release, self.release = self.release, None
            if release is not None and not self.reclaiming:
                release()


class Memory:
    """The residency of one executor device; Torch remains its allocator."""

    def __init__(self, device: Any, budget_bytes: int | None = None) -> None:
        self.device = torch.device(device)
        self.cuda: bool = self.device.type == "cuda"
        self.budget_bytes = budget_bytes
        self._budget_applied = False
        self.lock = threading.RLock()
        self.models: dict[nn.Module, Model] = {}
        self.regions: weakref.WeakSet[Region] = weakref.WeakSet()
        self.requests: set[Request] = set()
        self.last: dict[str, Any] | None = None

    @classmethod
    def open(cls, gpu: Any = 0, *, budget_bytes: int | None = None) -> Memory:
        device = torch.device("cuda", gpu) if isinstance(gpu, int) else torch.device(gpu)
        if device.type == "cuda":
            device = torch.device("cuda", device.index or 0)
        if budget_bytes is not None and budget_bytes < 0:
            raise ResidencyError("a local allocator budget cannot be negative")
        with _OPENING:
            found = _MEMORIES.get(str(device))
            if found is not None:
                if found.budget_bytes != budget_bytes:
                    raise ResidencyError("an open device cannot silently change allocator budgets")
                return found
            memory = cls(device, budget_bytes)
            _MEMORIES[str(device)] = memory
            return memory

    def apply_budget(self) -> None:
        """Apply an explicitly assigned LOCAL Torch allowance after device initialization."""
        if self._budget_applied or not self.cuda:
            return
        if self.budget_bytes is not None:
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(
                min(self.budget_bytes / total, 1.0), self.device
            )
        self._budget_applied = True

    def stream(self) -> Any:
        return torch.cuda.current_stream(self.device) if self.cuda else None

    def capturing(self) -> bool:
        return bool(
            self.cuda and torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()
        )

    def _figures(self) -> tuple[int, int, int]:
        if self.cuda and torch.cuda.is_initialized():
            return (
                int(torch.cuda.memory_allocated(self.device)),
                int(torch.cuda.memory_reserved(self.device)),
                int(torch.cuda.max_memory_allocated(self.device)),
            )
        if self.device.type == "mps":
            used = int(torch.mps.current_allocated_memory())
            return used, int(torch.mps.driver_allocated_memory()), used
        return 0, 0, 0

    def room(self, current: Collection[str] = ()) -> Room:
        self.apply_budget()
        if self.cuda:
            free, _ = torch.cuda.mem_get_info(self.device)
            allocated, _, _ = self._figures()
            # reserved - allocated includes graph-private storage that ordinary eager
            # allocation cannot necessarily reuse. Driver free is a conservative bound.
            free = max(int(free) - _RESERVE, 0)
            if self.budget_bytes is not None:
                free = min(free, max(self.budget_bytes - allocated, 0))
        else:
            from cozy_runtime.internal.accel import host_memory

            free = host_memory().free_bytes
        with self.lock:
            cold = warm = 0
            for model in self.models.values():
                for block in model.groups():
                    if block.data is not None and not (
                        block.count
                        or block.retained
                        or block.loading
                        or (model.users and model.thread != threading.get_ident())
                    ):
                        if model.name in current:
                            warm += block.nbytes
                        else:
                            cold += block.nbytes
        return Room(int(free), cold, warm)

    def load(self, module: nn.Module, level: Level | str = Level.GPU) -> Model:
        level = Level(level) if isinstance(level, str) else level
        model = self.models.get(module)
        if model is None:
            declared = vars(module).get("_cozy_weights")
            if not isinstance(declared, Declared):
                raise ResidencyError("the model has no _cozy_weights declaration")
            model = self._register(module, declared)
        if level is Level.GPU:
            hold = model.hold()
            if hold is not None:
                hold.release()
        else:
            self._retire(model)
            with self.lock:
                if model.users or any(b.count or b.retained for b in model.groups()):
                    raise ResidencyError("retire graphs and model uses before idle offload")
                for block in model.groups():
                    self._discard(block)
        return model

    def _register(self, module: nn.Module, declared: Declared) -> Model:
        model = Model(self, module, declared)
        w, paths = declared.weights, blocks_of(module)
        tensors = w.ck.components[w.component].tensors
        by_path: dict[str, dict[str, dict[str, _Part]]] = {}
        for key in tensors:
            if w.keys is not None and key not in w.keys:
                continue
            bound = w.bound.get(key, key) if w.bound else key
            path = _block_of(bound, paths)
            # A tied tensor used by distinct modules stays common, so either forward sees
            # one live parameter. The declared encoded bind routes must name their targets.
            group = model.sites.get(bound)
            if group is not None:
                aliases = [name for name, sites in model.sites.items() if sites is group]
                if any(_block_of(name, paths) != path for name in aliases):
                    path = ""
            parts = {}
            for role, part in stored_parts(w.ck, w.component, key).items():
                parts[role or "value"] = _Part(part.dtype, tuple(part.shape), part.length, role)
            by_path.setdefault(path, {})[key] = parts
        for index, path in enumerate(p for p in ("", *paths) if p in by_path):
            block = _Block(model, path, index, by_path[path])
            if path:
                model.blocks[path] = block
            else:
                model.common = block
        _placeholders(model)
        _hooks(model)
        self.models[module] = model
        return model

    def _declare(self, model: Model) -> None:
        identity = model.wanted_identity()
        if identity != model.identity or identity is None:
            if any(b.retained for b in model.groups()):
                raise StaleBinding("retire old graph bindings before changing a weight fill")
            for hold in list(model.holds):
                hold.shed()
            if any(b.count for b in model.groups()):
                raise ResidencyError("a weight fill cannot change during an active use")
            for block in model.groups():
                self._discard(block)
            model.identity = identity
            model.generation += 1

    def _discard(self, block: _Block) -> int:
        with self.lock:
            if (
                block.count
                or block.retained
                or block.loading
                or block.data is None
                or (block.model.users and block.model.thread != threading.get_ident())
            ):
                return 0
            block.data = block.assigned = block.ready = None
            block.generation += 1
            return block.nbytes

    def _retire(self, model: Model) -> bool:
        with self.lock:
            if model.users or model.retiring or model.closed or model.retire_graphs is None:
                return False
            model.retiring = True
        try:
            # Cached graph ownership is distinct from an active model use. Runtime drains
            # graph readers and closes bindings here, with no residency metadata lock held.
            model.retire_graphs()
            return True
        finally:
            with self.lock:
                model.retiring = False

    def reclaim(self, need: int = 0, *, current: Collection[str] = ()) -> int:
        """Discard idle local owners at a preparation boundary; never inside an allocator."""
        if self.capturing():
            return 0
        before_allocated, before_reserved, _ = self._figures()
        before_free = int(torch.cuda.mem_get_info(self.device)[0]) if self.cuda else 0
        with self.lock:
            blocks = sorted(
                (b for m in self.models.values() for b in m.groups()),
                key=lambda b: (b.model.name in current, b.last),
            )
        for block in blocks:
            if need and self.room(current).free >= need:
                break
            self._discard(block)
        if not need or self.room(current).free < need:
            for model in sorted(list(self.models.values()), key=lambda m: m.name in current):
                if need and self.room(current).free >= need:
                    break
                if self._retire(model):
                    for block in model.groups():
                        self._discard(block)
        # A callback may discard graph owners or decoded caches. It runs outside metadata
        # locks, and a pinned/locked region is skipped rather than waited on.
        for region in sorted(list(self.regions), key=lambda r: r.last):
            if need and self.room(current).free >= need:
                break
            region.discard()
        if self.cuda:
            torch.cuda.empty_cache()
        allocated, reserved, _ = self._figures()
        free = int(torch.cuda.mem_get_info(self.device)[0]) if self.cuda else 0
        return max(before_allocated - allocated, before_reserved - reserved, free - before_free, 0)

    def trim(self) -> int:
        """Idle offload: release eligible owners and return actual reserved-byte reduction."""
        if self.capturing():
            raise ResidencyError("idle offload cannot run during graph capture")
        before = self._figures()[1]
        for model in list(self.models.values()):
            self._retire(model)
        self.reclaim()
        return max(before - self._figures()[1], 0)

    def region(
        self,
        owner: str,
        kind: str,
        sets: object = (),
        spill: bool = False,
        *,
        release: Callable[[], None] | None = None,
    ) -> Region:
        del kind, sets, spill  # Cache semantics belong to its explicit owner, not an allocator.
        region = Region(self, owner, release)
        self.regions.add(region)
        return region

    def _empty(self, nbytes: int, reclaim: bool) -> Tensor:
        self.apply_budget()
        try:
            return torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        except torch.OutOfMemoryError:
            if not reclaim or self.capturing() or not self.reclaim():
                raise
            # Only the failed allocation is retried. No read, fill transform or forward ran.
            return torch.empty(nbytes, dtype=torch.uint8, device=self.device)

    @torch.inference_mode(False)  # type: ignore[untyped-decorator]
    def _ensure(self, block: _Block, reclaim: bool) -> None:
        if block.data is not None:
            if block.ready is not None:
                self.stream().wait_event(block.ready)
            return
        if self.capturing():
            raise ResidencyError("all graph weights must be resident before capture")
        with self.lock:
            block.loading = True
        data: dict[str, dict[str, Tensor]] = {}
        read_bytes = read_ns = 0
        try:
            source = block.model.source
            tensors = source.ck.components[source.component].tensors
            for key, parts in block.parts.items():
                views = data[key] = {}
                for role, part in parts.items():
                    raw = self._empty(part.nbytes, reclaim)
                    began = time.perf_counter_ns()
                    try:
                        tensors.read_into(key, raw, role=part.role or None)
                    except tensorfs.Error:
                        # TensorFS retains borrowed destinations on failed completion. Surface
                        # a sticky device failure as Torch's typed error before this executor
                        # can be reused; ordinary I/O errors keep their original classification.
                        if self.cuda:
                            torch.cuda.synchronize(self.device)
                        raise
                    read_ns += time.perf_counter_ns() - began
                    read_bytes += part.nbytes
                    dtype = getattr(torch, _DTYPES[part.dtype])
                    shape = part.shape
                    if part.dtype == "F4":
                        if not shape or shape[-1] % 2:
                            raise ResidencyError("packed F4 weights require an even last axis")
                        shape = (*shape[:-1], shape[-1] // 2)
                    view = raw.view(dtype)
                    if view.numel() != math.prod(shape):
                        raise ResidencyError(f"stored geometry does not match {key}#{role}")
                    views[role] = view.view(shape)
            if block.model.fill is not None:
                block.model.fill(block.path, data)
            ready = None
            if self.cuda:
                ready = torch.cuda.Event()
                ready.record(self.stream())
            with self.lock:
                block.data, block.ready = data, ready
                block.generation += 1
        finally:
            request = _REQUEST.get()
            if request is not None and read_bytes:
                request.read(block, read_ns, read_bytes)
            with self.lock:
                block.loading = False

    def _enter(self, block: _Block, reclaim: bool = True) -> None:
        global progress
        block.model._claim()
        with self.lock:
            block.count += 1
            nested = block.count > 1
        try:
            if not nested:
                request = _REQUEST.get()
                self._ensure(block, reclaim and not (request is not None and request.idle))
                if block.assigned is None:
                    block.assigned = _assign(block)
                for site, value in block.assigned:
                    block.saved.append((site, site.table[site.name]))
                    site.table[site.name] = value
            self._used(block)
        except BaseException:
            self._exit(block)
            raise
        progress += 1

    def _used(self, block: _Block) -> None:
        if self.cuda:
            stream = self.stream()
            for tensor in block.tensors():
                tensor.record_stream(stream)

    def _exit(self, block: _Block) -> None:
        try:
            self._used(block)
        finally:
            with self.lock:
                block.count -= 1
                if block.count == 0:
                    for site, previous in reversed(block.saved):
                        site.table[site.name] = previous
                    block.saved.clear()
                    block.last = time.monotonic_ns()
            block.model._release()

    def _read(self, block: _Block, site: _Site) -> Tensor:
        model = block.model
        model._claim()
        try:
            if not getattr(model.local, "depth", 0):
                self._declare(model)
            self._enter(block)
            try:
                value = site.table[site.name]
                if isinstance(value, Placeholder):
                    raise ResidencyError("the declared binder did not supply this weight")
                return value
            finally:
                self._exit(block)
        finally:
            model._release()

    def _open(self, model: Model, key: str) -> None:
        model._claim()
        depth = getattr(model.local, "depth", 0)
        model.local.depth = depth + 1
        if depth:
            return
        request = _REQUEST.get()
        try:
            self._declare(model)
            model.local.request = request
            if request is not None:
                request.current.add(model.name)
            if model.common is not None:
                self._enter(model.common)
        except BaseException:
            model.local.depth = 0
            if request is not None:
                request.current.discard(model.name)
            model._release()
            raise

    def _close(self, model: Model) -> None:
        try:
            model.local.depth -= 1
            if model.local.depth == 0:
                try:
                    if model.common is not None:
                        self._exit(model.common)
                finally:
                    request = getattr(model.local, "request", None)
                    if request is not None:
                        request.current.discard(model.name)
                        request.sample()
        finally:
            model._release()

    @contextmanager
    def request(
        self, id: str, plan: Sequence[object] = (), *, idle: bool = False
    ) -> Iterator[Request]:
        request = Request(self, id, idle)
        request.plan(plan)
        with self.lock:
            if self.requests:
                request.overlap = True
                for other in self.requests:
                    other.overlap = True
            self.requests.add(request)
        token = _REQUEST.set(request)
        try:
            yield request
        finally:
            request.sample()
            request.ended = request.stats()
            with self.lock:
                self.requests.discard(request)
                self.last = request.ended
            _REQUEST.reset(token)

    def stats(self) -> dict[str, Any] | None:
        return self.last

    def torch_stats(self) -> dict[str, Any]:
        allocated, reserved, peak = self._figures()
        return {
            "allocated": allocated,
            "mapped": reserved,
            "peak": peak,
            "requests": {r.id: (allocated, max(r.peak, peak)) for r in self.requests},
            "allocator_scope": "process",
        }


class Request:
    """Request context; Torch observations are process-wide, including overlapping calls."""

    def __init__(self, memory: Memory, id: str, idle: bool = False) -> None:
        self.memory, self.id, self.idle = memory, id, idle
        self.current: set[str] = set()
        self.steps: list[set[str]] = []
        self.peak = 0
        self.overlap = False
        self.moved = self.read_ns = 0
        self.trace: list[dict[str, object]] = []
        self.ended: dict[str, Any] | None = None
        self.sample()

    def sample(self) -> None:
        self.peak = max(self.peak, self.memory._figures()[2])

    def plan(self, steps: Sequence[object]) -> None:
        def name(value: object) -> str:
            if isinstance(value, Model):
                return value.name
            if isinstance(value, nn.Module):
                return self.memory.models[value].name
            return str(value)

        self.steps = [
            {name(v) for v in step}
            if isinstance(step, Collection) and not isinstance(step, (str, nn.Module))
            else {name(step)}
            for step in steps
        ]

    def room(self, step: int | None = None) -> Room:
        current = self.current if step is None or step >= len(self.steps) else self.steps[step]
        return self.memory.room(current)

    def reclaim(self, need: int = 0) -> int:
        """A known pure cut may shed the step's speculative whole-model hold.

        Its currently executing block/common hooks retain their own uses. Explicit holds and
        graph-owned storage stay pinned. This never retries a package or changes graph backing.
        """
        if self.idle or self.memory.capturing():
            return 0
        with self.memory.lock:
            holds = [
                hold
                for model in self.memory.models.values()
                if model.name in self.current and model.thread == threading.get_ident()
                for hold in model.holds
                if hold.reclaimable
            ]
        for hold in holds:
            hold.shed()
        return self.memory.reclaim(need, current=self.current)

    @contextmanager
    def on_thread(self) -> Iterator[Request]:
        token = _REQUEST.set(self)
        try:
            yield self
        finally:
            _REQUEST.reset(token)

    @contextmanager
    def yielding(self) -> Iterator[None]:
        # A child wait does not prove this entire invocation is idle or release live tensors.
        yield

    def reset_peak(self) -> int:
        self.sample()
        _, _, peak = self.memory._figures()
        if self.memory.cuda and len(self.memory.requests) == 1:
            torch.cuda.reset_peak_memory_stats(self.memory.device)
        return peak

    def read(self, block: _Block, ns: int, nbytes: int) -> None:
        self.moved += nbytes
        self.read_ns += ns
        self.trace.append(
            {
                "step": len(self.trace),
                "set": block.model.name,
                "block": block.index,
                "bytes": nbytes,
                "origin": "store",
                "speculative": False,
            }
        )
        self.sample()

    def stats(self) -> dict[str, Any]:
        if self.ended is not None:
            return self.ended
        self.sample()
        allocated, reserved, _ = self.memory._figures()
        return {
            "request": self.id,
            "allocator_scope": "process",
            "overlapping": self.overlap,
            "peak_window": "process allocator high-water, carried across section resets",
            "sets": {
                m.name: sum(b.nbytes for b in m.groups()) for m in self.memory.models.values()
            },
            "moved": {"store": {"bytes": self.moved, "ns": self.read_ns}},
            "trace": list(self.trace),
            "peak": self.peak,
            "allocated": allocated,
            "reserved": reserved,
            "host": None,
            "steps": [],
            "make_room": [],
        }


def _block_of(name: str, paths: Sequence[str]) -> str:
    return max((p for p in paths if name == p or name.startswith(p + ".")), key=len, default="")


def _sites(module: nn.Module) -> dict[str, list[_Site]]:
    groups: dict[int, list[_Site]] = {}
    names: dict[str, int] = {}
    for path, sub in module.named_modules(remove_duplicate=False):
        for table, parameter in ((sub._parameters, True), (sub._buffers, False)):
            for attr, tensor in table.items():
                if tensor is None:
                    continue
                names[f"{path}.{attr}" if path else attr] = id(tensor)
                sites = groups.setdefault(id(tensor), [])
                if not any(s.table is table and s.name == attr for s in sites):
                    sites.append(_Site(table, attr, parameter))
    return {name: groups[key] for name, key in names.items()}


def _placeholders(model: Model) -> None:
    done: set[int] = set()
    for name, sites in model.sites.items():
        meta = sites[0].table[sites[0].name]
        if id(meta) in done or meta.device.type != "meta":
            continue
        done.add(id(meta))
        block = next((b for b in model.groups() if name in b.parts), None)
        if block is None:
            path = _block_of(name, list(model.blocks))
            block = model.blocks.get(path) or model.common
        if block is None:
            raise ResidencyError(f"{model.name}: no stored group for {name}")
        held = Placeholder(meta.shape, meta.dtype, model.memory.device)
        if sites[0].parameter:
            held = nn.Parameter(held, requires_grad=False)

        def read(b: _Block = block, site: _Site = sites[0]) -> Tensor:
            return model.memory._read(b, site)

        held._varena_read = read
        for site in sites:
            site.table[site.name] = held


def _assign(block: _Block) -> list[tuple[_Site, Tensor]]:
    model = block.model
    if block.data is None:
        raise ResidencyError("weight assignment before a complete fill")
    assigned: dict[tuple[int, str], tuple[_Site, Tensor]] = {}
    for key, parts in block.data.items():
        if model.bind is not None:
            values = model.bind(key, parts)
        elif set(parts) == {"value"}:
            name = model.source.bound.get(key, key) if model.source.bound else key
            values = {name: parts["value"]}
        else:
            raise ResidencyError(f"{model.name}: encoded {key} needs a declared binder")
        for name, value in values.items():
            sites = model.sites.get(name, ())
            if not sites:
                continue
            old = sites[0].table[sites[0].name]
            tensor = value if old is None or old.dtype == value.dtype else value.to(old.dtype)
            if any(site.parameter for site in sites):
                tensor = nn.Parameter(tensor, requires_grad=False)
            for site in sites:
                # All names originally tied to one object receive the same Parameter object,
                # not merely distinct wrappers sharing its storage.
                assigned[(id(site.table), site.name)] = site, tensor
    return list(assigned.values())


def _hooks(model: Model) -> None:
    memory = model.memory
    for entry, sub in (("", model.module), *model.module.named_children()):

        def opened(args: tuple[object, ...], kwargs: dict[str, object], name: str = entry) -> None:
            key = (
                name
                + "|"
                + "x".join(
                    str(d)
                    for value in tree_leaves((args, kwargs))
                    if isinstance(value, Tensor)
                    for d in value.shape
                )
            )
            memory._open(model, key)

        model.handles.extend(_paired(sub, opened, lambda: memory._close(model)))
    for block in model.blocks.values():
        sub = model.module.get_submodule(block.path)

        def entered(*_: object, b: _Block = block) -> None:
            memory._enter(b)

        def exited(b: _Block = block) -> None:
            memory._exit(b)

        model.handles.extend(_paired(sub, entered, exited))


def _paired(module: nn.Module, enter: Callable[..., None], leave: Callable[[], None]) -> list[Any]:
    owed = threading.local()

    def pre(_: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        enter(args, kwargs)
        owed.count = getattr(owed, "count", 0) + 1

    def post(*_: object) -> None:
        if getattr(owed, "count", 0):
            owed.count -= 1
            leave()

    return [
        module.register_forward_pre_hook(pre, with_kwargs=True),
        module.register_forward_hook(post, always_call=True),
    ]
