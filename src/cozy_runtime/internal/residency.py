"""Executor-local weight residency with ordinary Torch views.

CUDA weights use TensorFS's process-local Rust reservations. A view owns its physical pin
through the Torch storage lifetime, including escaped aliases. Runtime drops idle views while
the native manager keeps eligible backing cached. CPU and MPS use Torch-owned storage.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import sys
import threading
import time
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import pairwise
from typing import Any, Protocol

import tensorfs
import torch
from tensorfs import varena
from tensorfs.raw import parts as stored_parts
from torch import Tensor, nn
from torch.utils._pytree import tree_leaves, tree_map

from cozy_runtime import graphs
from cozy_runtime.author._context import CURRENT
from cozy_runtime.author._errors import CapabilityError, Fault, fault
from cozy_runtime.internal import proctree
from cozy_runtime.internal.accel import (
    ResidencyError,
    StaleBinding,
    allocation,
    host_memory,
    release_cached,
)
from cozy_runtime.internal.hostfacts import physical_memory

__all__ = [
    "Binding",
    "BindingLayout",
    "Declared",
    "Hold",
    "Level",
    "Memory",
    "Model",
    "QueuedPrefetch",
    "Region",
    "Request",
    "Room",
    "TensorLayout",
    "Weights",
    "blocks_of",
    "paging",
    "split",
    "unpaged",
]

progress = 0
_RESERVE = 512 << 20


def _torch_bytes(device: torch.device) -> tuple[int, int, int]:
    """Torch's allocated, reserved and peak allocated CUDA bytes in one allocator read.
    `torch.cuda.memory_allocated` and its siblings each flatten the allocator's whole statistics
    dict (0.1 ms); a cut site read them three to five times, 5 s of each SDXL request at 3 GB."""
    stats = torch.cuda.memory_stats_as_nested_dict(device)
    allocated, reserved = stats["allocated_bytes"]["all"], stats["reserved_bytes"]["all"]
    return int(allocated["current"]), int(reserved["current"]), int(allocated["peak"])


#: Streamed (non-preferred) blocks resident at once: the one the GPU runs, the one entering, and
#: the next one being read. The read takes the oldest completed block's room (Varena evicts it on
#: its last-consumer events, never waiting), so it overlaps compute instead of following it.
_STREAMED = 3
#: Other overlays' pages a block keeps parked: base and a LoRA alternate with one.
_PARKED = 2
_HOST_STAGING = 64 << 20
#: Host RAM the baked-page cache leaves free (`Memory._keep_baked`).
_BAKED_HEADROOM = 8 << 30
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
# During authored stage preparation, undeclared model reads must not page in sibling blocks.
_CAPTURE_BLOCKS: ContextVar[frozenset[int] | None] = ContextVar("cozy_capture_blocks", default=None)
#: A native loop recording its pass in stages (`internal.loop`): block hooks mark each cut.
CUTTER: ContextVar[Any] = ContextVar("cozy_loop_cutter", default=None)
#: A native loop's eager call (`internal.loop`): the models and blocks it ran.
ENTERED: ContextVar[set[Any] | None] = ContextVar("cozy_loop_entered", default=None)
_MEMORIES: weakref.WeakValueDictionary[str, Memory] = weakref.WeakValueDictionary()
_OPENING = threading.Lock()
# A failed graph completion must retain the actual storage owners until executor exit.
_UNPROVEN: list[Binding] = []


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
    #: Torch's reserved, unallocated bytes: a cut's level and its pre-reclaim count them (its
    #: activations reuse them first); a reclaim returns them before weights.
    cached: int = 0


@dataclass(frozen=True, slots=True)
class Weights:
    ck: tensorfs.Checkpoint
    component: str
    release: str = ""
    shard: str = "0/1"
    keys: Collection[str] | None = None
    bound: Mapping[str, str] | None = None


class Filling(Protocol):
    """A model's fill. `identity` names what every page's fill shares (None: never kept);
    `overlay` names what a fill of one block would add to it now (a LoRA a package arms per
    forward). A block's page serves only a use wanting the overlay it was filled with."""

    @property
    def identity(self) -> str | None: ...

    def overlay(self, block: str, keys: Collection[str], /) -> Hashable: ...

    def __call__(self, block: str, parts: Mapping[str, Mapping[str, Tensor]], /) -> bool:
        """Fill `block`'s parts in place; whether their bytes now differ from the stored ones."""
        ...


@dataclass(frozen=True, slots=True)
class TensorLayout:
    """One final buffer's exact geometry; source names an eligible whole raw-role alias."""

    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...] = ()
    source: str | None = None

    def elements(self) -> int:
        strides = self.strides or _strides(self.shape)
        if (
            len(self.shape) != len(strides)
            or any(n < 0 for n in self.shape)
            or any(n < 0 for n in strides)
        ):
            raise ResidencyError("invalid final buffer geometry")
        extent = 1
        for stride, length in sorted(
            (s, n) for n, s in zip(self.shape, strides, strict=True) if n > 1
        ):
            if stride < extent:
                raise ResidencyError("final buffer axes overlap")
            extent = stride * length
        return (
            0
            if any(n == 0 for n in self.shape)
            else 1 + sum((n - 1) * s for n, s in zip(self.shape, strides, strict=True))
        )


class BindingLayout(Protocol):
    """An explicitly immutable binder. Its cold writer is not a native-prefetch callback.
    `encoded`: it binds encoded parts (the ordinary weight routes), so a stage may load them."""

    encoded: bool

    @property
    def identity(self) -> str: ...

    def layout(self, key: str, /) -> Mapping[str, TensorLayout] | None: ...

    def write(
        self, key: str, parts: Mapping[str, Tensor], destinations: Mapping[str, Tensor], /
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class Declared:
    weights: Weights
    blocks: Sequence[str] | None = None
    #: The `paging` rule that found `blocks` (`+split` when `split` made units smaller).
    rule: str = ""
    fill: Filling | None = None
    bind: Callable[[str, Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None
    finals: BindingLayout | None = None


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
    return paging(module)[1]


def paging(module: nn.Module) -> tuple[str, list[str]]:
    """A component's paging blocks and the rule that found them, so no package has to name them:
    its declaration (`declared`), else its root's `_no_split_modules` (`root`), else those of the
    models inside it (`submodules`: a wrapper around a Transformers or Diffusers model takes that
    model's units), else its largest list of repeated same-class children (`repeated`: a model's
    layers). MOSS-Image's 9B text encoder and 5B DiT each named none and were one block."""
    declared = vars(module).get("_cozy_weights")
    if isinstance(declared, Declared) and declared.blocks is not None:
        return declared.rule or "declared", list(declared.blocks)
    rule, names = "root", set(getattr(module, "_no_split_modules", None) or ())
    if not names:
        rule = "submodules"
        names = {n for sub in module.modules() for n in getattr(sub, "_no_split_modules", None) or ()}
    found = [
        path
        for path, sub in module.named_modules()
        if path and any(kind.__name__ in names for kind in type(sub).__mro__)
    ]
    found = [a for a in found if not any(b.startswith(a + ".") for b in found)]
    if found:
        return rule, found
    best, most = [], 0
    for path, sub in module.named_modules():
        children = list(sub.named_children()) if isinstance(sub, nn.ModuleList | nn.Sequential) else []
        if len(children) < 2 or len({type(child) for _, child in children}) > 1:
            continue
        size = sum(t.numel() * t.element_size() for t in sub.parameters())
        if size > most:
            best, most = [f"{path}.{name}" if path else name for name, _ in children], size
    return ("repeated", best) if best else ("none", [])


#: A paging unit larger than this is a warning: a quarter of an 8 GB card.
_LARGEST_UNIT = 2 << 30
#: A paging unit holding more than this splits (`split`), the same on every card.
_SPLIT_UNIT = 1 << 30


def _unit(name: str, blocks: Collection[str]) -> str:
    """The block holding tensor `name` (its longest dotted prefix that is a block), "" for common."""
    while name:
        name = name.rpartition(".")[0]
        if name in blocks:
            return name
    return ""


def split(module: nn.Module, paths: Sequence[str], sizes: Mapping[str, int]) -> list[str]:
    """`paths` with each unit over `_SPLIT_UNIT` split where child modules allow: a block gives way to
    its weighted children (its own tensors join the common weights), and the common weights give their
    largest children that hold no block, one at a time, until under. A new block over the bound splits
    the same way, down to single layers; one tensor over it stays whole. `sizes` are the stored bytes
    by module-relative tensor name. Hunyuan-Image-3 kept 10 GB of attention and dense layers in common
    beside its expert blocks."""
    modules = dict(module.named_modules())
    blocks = set(paths)

    def held() -> dict[str, int]:
        units: dict[str, int] = {}
        for name, n in sizes.items():
            unit = _unit(name, blocks)
            units[unit] = units.get(unit, 0) + n
        return units

    def free(path: str) -> list[tuple[int, str]]:
        """`path`'s weighted child modules that hold no block, through children that hold one."""
        found, stack = [], [path]
        while stack:
            parent = stack.pop()
            for name, _ in modules[parent].named_children():
                child = f"{parent}.{name}" if parent else name
                if any(b == child or b.startswith(child + ".") for b in blocks):
                    if child not in blocks:
                        stack.append(child)
                elif size := sum(n for t, n in sizes.items() if t.startswith(child + ".")):
                    found.append((size, child))
        return sorted(found, reverse=True)

    units = held()
    queue = [unit for unit, n in units.items() if n > _SPLIT_UNIT]
    while queue:
        unit = queue.pop()
        if held().get(unit, 0) <= _SPLIT_UNIT:
            continue
        if unit:
            children = free(unit)
            if not children:
                continue
            blocks.discard(unit)
            blocks.update(child for _, child in children)
            queue += ["", *(child for size, child in children if size > _SPLIT_UNIT)]
            continue
        for size, child in free(""):
            if held().get("", 0) <= _SPLIT_UNIT:
                break
            blocks.add(child)
            if size > _SPLIT_UNIT:
                queue.append(child)
    order = {path: i for i, path in enumerate(modules)}
    return sorted(blocks, key=order.__getitem__)


def unpaged(module: nn.Module, rule: str, groups: Mapping[str, int]) -> str | None:
    """Why this component cannot page on a small card, or None: a unit over `_LARGEST_UNIT` (the
    whole component, when it has no blocks). `groups` are the bytes by block path ("" common, its
    embedding tables excluded: one row per token is read, and they are not what pages)."""
    path, size = max(groups.items(), key=lambda item: item[1], default=("", 0))
    if size <= _LARGEST_UNIT:
        return None
    unit = f"block {path}" if path else "its common weights"
    return (f"{type(module).__name__} pages by rule '{rule}' and {unit} is {size / (1 << 30):.2f} GiB: "
            f"a card with less free memory cannot run it; name its paging units (`_no_split_modules`)")


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
    strides: tuple[int, ...] = ()


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
    region: Any = None
    preferred: bool = False
    preference_set: bool = False
    native_storage: int | None = None
    #: the Varena Uses behind the raw and final views; `_used` declares their consumer streams
    native_use: Any = None
    final_native_use: Any = None
    aliases: dict[str, dict[str, str]] | None = None
    raw_views: dict[tuple[int, str], tuple[int, _Part]] = field(default_factory=dict)
    finals: dict[str, dict[str, _Part]] = field(default_factory=dict)
    final_region: Any = None
    final_use: Any = None
    final_data: dict[str, dict[str, Tensor]] | None = None
    final_storage: int | None = None
    final_fills: int = 0
    final_hits: int = 0
    next: _Block | None = None
    #: the fill overlay (`Filling.overlay`) its page was written with
    overlay: Hashable = None
    #: native pages of other overlays, parked for reuse (`_switch_overlay`): (raw, final) regions
    parked: dict[Hashable, tuple[Any, Any]] = field(default_factory=dict)

    def paged(self) -> bool:
        """Whether a page a fill wrote may still serve: bound data or a native region."""
        return self.data is not None or self.region is not None or self.final_region is not None

    @property
    def nbytes(self) -> int:
        return sum(part.nbytes for parts in self.parts.values() for part in parts.values())

    @property
    def storage_bytes(self) -> int:
        # Every part is independently aligned, just as its former Torch allocation was.
        return sum(
            (part.nbytes + 255) // 256 * 256 for p in self.parts.values() for part in p.values()
        )

    @property
    def allocation_bytes(self) -> int:
        if self.model.memory.native is None:
            return self.nbytes
        granularity = self.model.memory.granularity
        return sum(
            int(region.allocated_bytes)
            if region is not None
            else (size + granularity - 1) // granularity * granularity
            for region, size in (
                (self.region, self.storage_bytes),
                (self.final_region, self.final_bytes),
            )
        )

    @property
    def final_bytes(self) -> int:
        return sum(
            (p.nbytes + 255) // 256 * 256 for row in self.finals.values() for p in row.values()
        )

    def cached(self) -> bool:
        if self.final_bytes and (self.final_region is None or not self.final_region.resident):
            return False
        raw_needed = self.aliases is None or any(self.aliases.values()) or not self.finals
        return not raw_needed or (self.region is not None and bool(self.region.resident))

    def final_only(self) -> bool:
        return bool(self.finals) and self.aliases is not None and not any(self.aliases.values())

    def tensors(self) -> Iterator[Tensor]:
        if self.data is not None:
            yield from (tensor for parts in self.data.values() for tensor in parts.values())
        if self.assigned is not None:
            yield from (tensor for _, tensor in self.assigned)


@dataclass(frozen=True, slots=True)
class QueuedPrefetch:
    """One exact prepared-record snapshot, never permission to invoke that request."""

    request_id: str
    models: tuple[Model, ...]
    valid: Callable[[], bool]


@dataclass(slots=True)
class _Prefetch:
    block: _Block
    region: Any
    read: Any
    identity: str
    request: Request | None
    queued: QueuedPrefetch | None = None
    #: the fill overlay its page holds, and whether it fills the final (cast) region
    overlay: Hashable = None
    final: bool = False


class _HostPage:
    """A page's bytes in host RAM, registered with CUDA (pinned) so its refill is one DMA, and
    unregistered when the page goes, after its last copy (`event`) completed. Registered once
    filled: registering touched memory runs at about twice the rate of untouched memory."""

    __slots__ = ("event", "tensor")

    def __init__(self, tensor: Tensor) -> None:
        self.tensor = tensor
        self.event: Any = None
        code = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), tensor.numel(), 0)
        if int(code) != 0:
            raise RuntimeError(f"cudaHostRegister returned {code}")

    def release(self) -> None:
        if self.event is not None:
            self.event.synchronize()
        torch.cuda.cudart().cudaHostUnregister(self.tensor.data_ptr())


class _HostRead:
    """A prefetch from a host page: one DMA on the copy stream. Its fill commits only once the
    copy completed, so nothing that pins the region ready can read it early."""

    def __init__(self, use: Any, event: Any, nbytes: int) -> None:
        self.use, self.event, self.nbytes, self.canceled = use, event, nbytes, False

    def start(self) -> None:
        pass

    def poll(self) -> dict[str, bool]:
        return {"done": bool(self.event.query())}

    def cancel(self) -> None:
        self.canceled = True

    def wait(self) -> dict[str, Any]:
        self.event.synchronize()
        use, self.use = self.use, None
        if use is not None and not self.canceled:
            use.commit()
        nbytes = 0 if self.canceled else self.nbytes
        return {"canceled": self.canceled, "bytes": nbytes, "host": True}


class Model:
    def __init__(self, memory: Memory, module: nn.Module, declared: Declared) -> None:
        self.memory, self.module, self.source = memory, module, declared.weights
        self.component = declared.weights.component
        #: The component by its checkpoint's content, however its package was installed.
        self.content = f"{self.source.ck.id}/{self.component}"
        self.name = f"{self.source.release}/{self.content}"
        self.fill, self.bind, self.finals = declared.fill, declared.bind, declared.finals
        self.declared_finals = declared.finals
        self.binding_identity = None if self.finals is None else self.finals.identity
        self.layout_identity = ""
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
        # Once a block streams, later blocks cannot grow the resident prefix again.
        self.prefix: int | None = None
        #: the block entered last, whose successor in execution order `_enter` records (`next`)
        self.entered: _Block | None = None

    def wanted_identity(self) -> str | None:
        fill = "stored" if self.fill is None else self.fill.identity
        if fill is None:
            return fill
        layout = None if self.finals is None else self.finals.identity
        return (
            repr((fill, layout, self.layout_identity)) if layout or self.layout_identity else fill
        )

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
    def passing(self, key: str = "", *, executing: bool = False) -> Iterator[Model]:
        self.memory._open(self, key, executing=executing)
        try:
            yield self
        finally:
            self.memory._close(self, executing=executing)

    @contextmanager
    def lending(self) -> Iterator[None]:
        """This thread's open pass lends its common block to a native stage preparation, which
        needs every input idle, and takes it back after. A failed take-back stays lent."""
        if (
            not getattr(self.local, "depth", 0)
            or self.common is None
            or getattr(self.local, "lent", False)
        ):
            yield
            return
        self.memory._exit(self.common)
        self.local.lent = True
        try:
            yield
        finally:
            self.memory._enter(self.common)
            self.local.lent = False

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
            if free_only and self.memory.native is not None:
                # Native backing is already a cache. A speculative whole-model Torch
                # hold would pin idle blocks and defeat allocation-time native recovery.
                return None
            groups = self.groups()
            missing = sum(b.allocation_bytes for b in groups if b.data is None and not b.cached())
            if self.memory.device.type == "mps" and missing:
                missing += min(
                    max(
                        (
                            part.nbytes
                            for b in groups
                            if b.data is None
                            for p in b.parts.values()
                            for part in p.values()
                        ),
                        default=0,
                    ),
                    _HOST_STAGING,
                )
            if free_only and missing > self.memory.room().free:
                release_cached(torch, self.memory.device.type)
                if missing > self.memory.room().free:
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
                    return None
                raise
            except BaseException:
                for block in reversed(entered):
                    self.memory._exit(block)
                raise
            hold = Hold(self, entered, shed, free_only)
            self.holds.append(hold)
            return hold
        finally:
            self._release()

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
                self.memory._discard(block, forget=True)
            self.memory._forget_baked(self)
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
        for block in reversed(groups):
            self.model.memory._exit(block)

    def shed(self) -> None:
        self.release()
        if self.on_shed is not None:
            self.on_shed()

    def __enter__(self) -> Hold:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _state(region: Any) -> str:
    """A block region's residency, as a refusal names it."""
    return "none" if region is None else "resident" if region.resident else "cold"


class Binding:
    """A cached graph's owning storage, closed after its readers retire.

    An undeclared fill may prepare once for an ephemeral direct read. It never reports a
    valid/resident graph binding and cannot be replayed or prepared a second time.
    """

    def __init__(self, model: Model, groups: list[_Block]) -> None:
        self.model, self.groups = model, groups
        self.fill = model.wanted_identity()
        self.generation = model.generation
        self.targets = [
            (site, tuple(value.shape), value.dtype)
            for sites in model.sites.values()
            for site in sites
            if (value := site.table[site.name]) is not None
        ]
        self.identity = hashlib.sha256(
            repr(
                (
                    model.name,
                    self.fill,
                    self.generation,
                    [b.index for b in groups],
                    [(shape, str(dtype)) for _, shape, dtype in self.targets],
                )
            ).encode()
        ).hexdigest()
        self.closed = False
        self.completion_lost = False
        self.used = False
        self.held: list[tuple[_Block, int]] = []
        self.lock = threading.RLock()
        self.last: Any = None
        self._load_slot: Any = None
        self._stage_source: Any = None
        self._stage_regions: tuple[tuple[_Block, Any], ...] = ()
        self._load_data: dict[_Block, dict[str, dict[str, Tensor]]] | None = None
        self._load_storage: dict[_Block, int] = {}
        self._load_source: Weights | None = None
        self._load_target_strides: list[tuple[_Site, tuple[int, ...]]] = []

    def valid(self) -> bool:
        return not self._invalid()

    def _invalid(self) -> str:
        """The first reason this binding cannot serve a graph, or empty when it can."""
        # A caller retires this graph before _declare changes the model's cached fill.
        model = self.model
        if self.closed or self.completion_lost or model.closed:
            return "closed"
        if self.fill is None or self.fill != model.wanted_identity():
            return "fill identity changed"
        if self.generation != model.generation:
            return "model generation changed"
        if not all(block.region is region for block, region in self._stage_regions):
            return "stage region replaced"
        if self._load_source is not None and model.source is not self._load_source:
            return "load source changed"
        if not all(
            (value := site.table[site.name]) is not None and tuple(value.stride()) == stride
            for site, stride in self._load_target_strides
        ):
            return "target strides changed"
        if not all(
            (value := site.table[site.name]) is not None
            and tuple(value.shape) == shape
            and value.dtype == dtype
            for site, shape, dtype in self.targets
        ):
            return "target shape or dtype changed"
        if not all(
            (self._load_data is not None or b.data is not None) and b.generation == generation
            for b, generation in self.held
        ):
            return "held block released"
        return ""

    def resident(self) -> bool:
        return not self.unresident()

    def unresident(self) -> str:
        """Why `resident()` is false: the component and its first non-resident block."""
        name = self.model.component
        if why := self._invalid():
            return f"{name}: {why}"
        for b in self.groups:
            if b.data is None and not b.cached():
                final = f", final {_state(b.final_region)}" if b.final_bytes else ""
                return f"{name} block {b.path or '(common)'}: raw {_state(b.region)}{final}"
        return ""

    def _load_refusal(self, stage: bool, region_input: bool) -> str:
        """The first reason `prepare_load` refuses this binding, or empty when it can load."""
        memory, model = self.model.memory, self.model
        if why := self._invalid():
            return f"binding {why}"
        for why, refused in (
            ("binding already prepared", self.used),
            ("binding holds its blocks", self.held),
            ("a load slot is open", self._load_slot is not None),
            ("a stage source is open", self._stage_source is not None),
            ("memory is not CUDA", not memory.cuda),
            ("a capture is active", memory.capturing()),
            ("no native memory", memory.native is None),
            ("TensorFS cannot take region inputs", not region_input),
            ("the model has a fill", model.fill is not None),
        ):
            if refused:
                return why
        for block in self.groups:
            finals = sorted(block.finals)
            encoded = any(set(parts) != {"value"} for parts in block.parts.values())
            for why, refused in (
                ("has no declared layout", block.aliases is None),
                (f"has {len(finals)} final rows ({', '.join(finals[:3])})", finals),
                (
                    "has encoded parts outside a routed stage",
                    encoded and (not stage or model.finals is None or not model.finals.encoded),
                ),
                ("has no storage", not block.storage_bytes),
                (
                    "has packed F4 parts",
                    any(p.dtype == "F4" for parts in block.parts.values() for p in parts.values()),
                ),
                (f"is in use ({block.count})", block.count),
                (f"is retained ({block.retained}) in {model.component}", block.retained),
                ("is loading", block.loading),
            ):
                if refused:
                    return f"block {block.path or '(common)'} {why}"
        return ""

    def prepare_load(self, *, stage: bool = False) -> Any:
        """Create capture-private views for immutable raw blocks, without reading weights.

        Ordinary block data is never changed here. The native slot atomically retains ready
        regions and owns unpublished missing fills; its views are installed only by prepare().
        """
        memory, model = self.model.memory, self.model
        with self.lock:
            model._claim()
            try:
                if why := self._load_refusal(
                    stage, getattr(varena.LoadSlot, "supports_region_input", False)
                ):
                    raise StaleBinding(
                        f"native component loading needs idle immutable raw layouts: {why}"
                    )
                if any(
                    getattr(module, "_compiled_call_impl", None) is not None
                    or hasattr(module, "_orig_mod")
                    for module in model.module.modules()
                ):
                    raise StaleBinding("compiled modules require resident weight preparation")
                self._load_source = model.source
                self._load_target_strides = [
                    (site, tuple(site.table[site.name].stride())) for site, _, _ in self.targets
                ]
                # Protect every existing ready block before phase policy or a later cold
                # allocation can reclaim it. Native group admission takes its own pins before
                # these temporary owners are released. Nothing here exposes unreadable data.
                ready_uses = [
                    use
                    for block in self.groups
                    if block.region is not None
                    if (use := block.region.pin()) is not None
                ]
                # Run existing phase admission before private capacity is reserved, so
                # warmup hooks do not mistake those allocated inputs for missing bytes.
                memory._activate_phase(model)
                with memory.lock:
                    if memory._prefetch is not None:
                        memory._settle_prefetch(cancel=True)
                groups = []
                needed = 0
                for block in self.groups:
                    region = memory._native_region(block)
                    if not region.resident:
                        needed += int(region.allocated_bytes)
                    offset = 0
                    parts = []
                    for key, roles in block.parts.items():
                        for part in roles.values():
                            parts.append(
                                (
                                    f"{model.component}/{key}#{part.role or 'value'}",
                                    offset,
                                    part.nbytes,
                                )
                            )
                            offset += (part.nbytes + 255) // 256 * 256
                    groups.append((region, parts))
                if memory.budget_bytes is not None and needed > memory.room().free:
                    memory.reclaim(needed, current={model.name})
                    if needed > memory.room().free:
                        raise torch.OutOfMemoryError(
                            "native component inputs exceed local capacity"
                        )
                if stage and not getattr(varena.LoadSlot, "supports_ready_stage", False):
                    raise StaleBinding("TensorFS lacks ready-input model stage preparation")
                slot = (
                    varena.LoadSlot.from_regions(model.source.ck, groups, keep_ready=True)
                    if stage
                    else varena.LoadSlot.from_regions(model.source.ck, groups)
                )
                if (
                    stage
                    and slot is not None
                    and not callable(getattr(slot._native, "detach_source", None))
                ):
                    slot.close()
                    raise StaleBinding("TensorFS lacks detached model-stage input ownership")
                # Native now owns ready inputs, or ordinary resident capture revalidates.
                ready_uses.clear()
                if slot is None:
                    return None  # The caller must revalidate/reacquire its resident binding.
                self._load_slot = slot
                data: dict[_Block, dict[str, dict[str, Tensor]]] = {}
                for block, raw in zip(self.groups, slot.tensors, strict=True):
                    self._load_storage[block] = int(raw.untyped_storage().data_ptr())
                    offset = 0
                    rows = data[block] = {}
                    for key, parts in block.parts.items():
                        row = rows[key] = {}
                        for role, part in parts.items():
                            if part.dtype == "F4":
                                raise StaleBinding(
                                    "native raw component loading excludes packed weights"
                                )
                            value = raw[offset : offset + part.nbytes].view(
                                getattr(torch, _DTYPES[part.dtype])
                            )
                            row[role] = value.view(part.shape)
                            offset += (part.nbytes + 255) // 256 * 256
                self._load_data = data
                return slot
            except BaseException:
                if self._load_slot is not None:
                    self._load_slot.close()
                    self._load_slot = None
                self._load_data = None
                self._load_storage.clear()
                self._load_source = None
                self._load_target_strides.clear()
                raise
            finally:
                model._release()

    def immutable_buffer_ids(self, *, selected: bool = False) -> frozenset[int]:
        """Declared checkpoint-backed buffers, including idle placeholders in other blocks.

        They are immutable inputs, not mutable module state to clone while capturing one
        block. Restrict this to named final-layout targets; ordinary nonweight buffers keep
        the existing state snapshot and restoration behavior.
        """
        names: set[str] = set()
        for block in self.groups if selected else self.model.groups():
            for key in block.parts:
                if self.model.finals is not None:
                    layout = self.model.finals.layout(key)
                    if layout is not None:
                        names.update(layout)
                elif set(block.parts[key]) == {"value"}:
                    names.add(
                        self.model.source.bound.get(key, key) if self.model.source.bound else key
                    )
        return frozenset(
            id(site.table[site.name])
            for name in names
            for site in self.model.sites.get(name, ())
            if not site.parameter and site.table[site.name] is not None
        )

    def undeclared_input_ranges(self) -> tuple[tuple[Any, int, int], ...]:
        """Existing source/final reservations outside this stage, including escaped views."""
        return tuple(
            (
                self.model.memory.device,
                int(region.ptr),
                int(region.ptr) + int(region.allocated_bytes),
            )
            for block in self.model.groups()
            if block not in self.groups
            for region in (block.region, block.final_region)
            if region is not None
        )

    def detach_stage(self) -> Any:
        """Keep immutable source/VA identity after checked capture; release physical pins.

        The native source owns the Region reservations. It reacquires ready pins or missing
        fills on each sequence pass, without calling Python or replacing parameter addresses.
        """
        with self.lock:
            slot = self._load_slot
            if (
                slot is None
                or self._load_data is None
                or not self.used
                or self._stage_source is not None
                or not self.valid()
            ):
                raise StaleBinding("detachment requires a valid completed raw capture")
            if self.last is not None:
                try:
                    self.last.synchronize()
                except BaseException as incomplete:
                    self._unproven(incomplete)
            with self.model.memory.lock:
                if any(block.count or block.loading for block in self.groups):
                    raise StaleBinding("model stage capture is still using its input views")
                regions = tuple((block, block.region) for block in self.groups)
                if any(region is None for _, region in regions):
                    raise StaleBinding("model stage lost its managed input reservation")
                for block, _ in self.held:
                    block.retained -= 1
                    self.model.memory._idle(block)
                self.held.clear()
                self._load_data = None
                self._load_storage.clear()
            # Drop every private Python tensor owner before native's actual live-Pin check.
            # An escaped external view makes this refuse; no refcount guess or gc sweep.
            slot._tensor = None
            slot._tensors = None
            source = slot._native.detach_source()
            self._stage_regions = regions
            self._stage_source = source
            self._load_slot = None
            slot.close()
            return source

    def immutable_inputs(self) -> tuple[Tensor, ...]:
        """The native-loaded source ranges, including the all-ready resident fallback."""
        if self._load_source is None:
            return ()
        if self._load_slot is not None:
            return tuple(self._load_slot.immutable_inputs)
        return tuple(tensor for block in self.groups for tensor in block.tensors())

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
                if self._stage_source is not None:
                    raise StaleBinding("detached model stages prepare only through native input")
                if not self.valid() and not ephemeral:
                    raise StaleBinding("graph weight storage or fill changed")
                request = _REQUEST.get()
                if request is not None and request.idle and not self.resident():
                    raise StaleBinding("idle graph preparation cannot page weights in")
                if self._load_data is not None and self.used:
                    raise StaleBinding("a native loaded binding can only prepare once")
                self.used = True
                with torch.cuda.stream(stream) if memory.cuda else nullcontext():
                    if self.last is not None:
                        stream.wait_event(self.last)
                    entered: list[_Block] = []
                    retained: list[tuple[_Block, int]] = []
                    published = False
                    restored: list[tuple[_Block, Any, Any, Any, Any]] = []
                    try:
                        if not self.held:
                            # Claim the complete dependency set before any preparation. Neither
                            # another preparation nor a nested fill can evict its later blocks.
                            with memory.lock:
                                for block in self.groups:
                                    block.retained += 1
                                    retained.append((block, block.generation))
                        if self._load_data is not None:
                            for block in self.groups:
                                restored.append(
                                    (
                                        block,
                                        block.data,
                                        block.assigned,
                                        block.ready,
                                        block.native_storage,
                                    )
                                )
                                block.data = self._load_data[block]
                                block.assigned = block.ready = None
                                block.native_storage = self._load_storage[block]
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
                            try:
                                for block in reversed(entered):
                                    memory._exit(block)
                            finally:
                                for block, data, assigned, ready, storage in reversed(restored):
                                    block.data, block.assigned = data, assigned
                                    block.ready, block.native_storage = ready, storage
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
                    self.model.memory._idle(block)
                self.held.clear()
            if self._load_slot is not None:
                self._load_slot.close()
                self._load_slot = None
                # Abandoning the slot drops any unpublished fill, leaving its region Cold.
                # A capture-only alias may have escaped and outlive it, so retire the region
                # identity of every block that was being filled (not resident), as _ensure
                # does on a failed fill. Otherwise a later prepare() returns Busy against the
                # dropped fill and breaks the eager fallback. Resident inputs keep their cache.
                with self.model.memory.lock:
                    for block in self.groups:
                        region = block.region
                        if region is not None and not region.resident:
                            block.region = None
                            block.preferred = block.preference_set = False
            self._load_data = None
            self._load_storage.clear()
            self._load_source = None
            self._load_target_strides.clear()
            self._stage_source = None
            self._stage_regions = ()


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
    """One executor's weight residency, beside Torch's activation allocator."""

    def __init__(self, device: Any, budget_bytes: int | None = None) -> None:
        self.device = torch.device(device)
        self.cuda: bool = self.device.type == "cuda"
        self.budget_bytes = budget_bytes
        self.lock = threading.RLock()
        self.models: dict[nn.Module, Model] = {}
        self.regions: weakref.WeakSet[Region] = weakref.WeakSet()
        self.requests: set[Request] = set()
        self.last: dict[str, Any] | None = None
        self.native: Any = None
        self.granularity = 512
        self.window_bytes = 0
        #: The active phase's weights all fit beside the other phases' idle ones: nothing of it
        #: streams, so its blocks reserve no streaming window (`_activate_phase`).
        self._fits = False
        self._stage_window_owner: object | None = None
        self.prefetch_supported = False
        self._prefetch: _Prefetch | None = None
        self.phase_supported = False
        self._phase_names: frozenset[str] | None = None
        self._phase_request: str | None = None
        self._forward_depth = 0
        #: (set, block, fill identity, overlay) -> a refilled block's baked bytes in host RAM:
        #: its next refill copies them back, with no store read, factor read or bake
        self._baked: OrderedDict[tuple[object, ...], Tensor] = OrderedDict()
        self._baked_bytes = 0
        #: resident pages baked once: kept from their first refill
        self._baked_once: set[tuple[object, ...]] = set()
        #: a hooked block's pages whose fill wrote nothing: their stored bytes are the page, so a
        #: store read prefetches and the idle tier pins them as it does a plain block's
        self._plain: set[tuple[object, ...]] = set()
        #: host RAM this process could still take at the cache's last look; -1 unreadable
        self._baked_room = -1
        #: The host tier holds every streamed page (baked or stored) pinned, within half of the
        #: host's RAM. Off for the process once pinning fails (then pages stream from the store).
        self._host_tier = True
        self._host_cap = max(physical_memory(), 0) // 2
        #: stored pages to pin once the executor is idle (`pin_idle`), never on a request's path
        self._wanted: OrderedDict[tuple[object, ...], _Block] = OrderedDict()
        #: each stored page's refills in the running request, and in the last request that
        #: refilled it: the bytes a kept page saves per request, per byte of host RAM it holds
        self._refills: dict[tuple[object, ...], int] = {}
        self._rate: dict[tuple[object, ...], int] = {}
        #: stored pages whose store read completed: a resident block's page is wanted on its second
        self._read_once: set[tuple[object, ...]] = set()
        self._h2d_stream: Any = None
        self._host_looked = 0.0
        admission = inspect.signature(varena.Region.prepare).parameters
        self._host_prepare = "evict_completed" in admission  # TensorFS 0.17.4
        if self.cuda:
            try:
                if budget_bytes is not None:
                    varena.set_activation_limit(self.device.index or 0, budget_bytes)
                self.native = varena.Arena(self.device.index or 0)
                self.granularity = int(self.native.granularity)
                self.prefetch_supported = bool(
                    callable(getattr(varena, "replay", None))
                    and getattr(varena, "supports_prefetch", lambda: False)()
                )
                self.phase_supported = bool(getattr(varena, "supports_phases", lambda: False)())
            except (ImportError, AttributeError, RuntimeError) as exc:
                if fault(exc) is not Fault.OTHER:
                    raise
                raise ResidencyError(f"native CUDA weight residency is unavailable: {exc}") from exc

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

    def stream(self) -> Any:
        return torch.cuda.current_stream(self.device) if self.cuda else None

    def capturing(self) -> bool:
        return bool(
            self.cuda and torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()
        )

    def _figures(self) -> tuple[int, int, int]:
        if self.cuda and torch.cuda.is_initialized():
            resident = int(self.native.stats()["resident_bytes"]) if self.native is not None else 0
            allocated, reserved, peak = _torch_bytes(self.device)
            return allocated + resident, reserved + resident, peak + resident
        if self.device.type == "mps":
            used = int(torch.mps.current_allocated_memory())
            return used, int(torch.mps.driver_allocated_memory()), used
        return 0, 0, 0

    def room(self, current: Collection[str] = ()) -> Room:
        if self.cuda:
            free, total = torch.cuda.mem_get_info(self.device)
            # reserved - allocated includes graph-private storage that ordinary eager
            # allocation cannot necessarily reuse. Driver free is a conservative bound.
            free = max(int(free) - _RESERVE, 0)
            if self.budget_bytes is not None:
                # The card is `budget_bytes` (a development cap): what it lacks of the real one
                # is never there, whoever holds the rest. Account native backing beside Torch
                # reservations, including graph pools: unused private-pool bytes are not
                # headroom for an eager allocation.
                card = max(free - (int(total) - self.budget_bytes), 0)
                free = min(card, max(self.budget_bytes - self._figures()[1], 0))
        elif self.device.type == "mps":
            figures = allocation(torch, "mps")
            free, reserved = figures["driver_free_bytes"], figures["reserved_bytes"]
            if self.budget_bytes is not None:
                free = min(free, max(self.budget_bytes - reserved, 0))
        else:
            free = host_memory().free_bytes
        if self.native is not None:
            # Native pins, including escaped Torch storage aliases, are the authority.
            # Do not count an idle Python block as reclaimable merely because hooks closed.
            # The active phase's resident prefix (preferred) is warm, as current models' blocks
            # are below: a cut takes a finer level before it evicts the weights it runs (H3 Turbo
            # on a 4090: one coarser FFN cut evicted 20 transformer blocks, re-read and re-baked).
            stats = self.native.stats()
            reclaimable = int(stats["reclaimable_bytes"])
            own = min(int(stats.get("preferred_bytes", 0)), reclaimable)
            # Pure-cut workspace estimates still use Torch's allocation granularity.
            return Room(int(free), reclaimable - own, own)
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
                    self._discard(block, forget=True)
        return model

    def _register(self, module: nn.Module, declared: Declared) -> Model:
        model = Model(self, module, declared)
        w, (rule, paths) = declared.weights, paging(module)
        tensors = w.ck.components[w.component].tensors
        tables = [p for p, sub in module.named_modules() if p and isinstance(sub, nn.Embedding)]
        sizes: dict[str, int] = {}
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
            if path or not any(bound.startswith(t + ".") for t in tables):
                sizes[path] = sizes.get(path, 0) + sum(part.nbytes for part in parts.values())
        for index, path in enumerate(p for p in ("", *paths) if p in by_path):
            block = _Block(model, path, index, by_path[path])
            if path:
                model.blocks[path] = block
            else:
                model.common = block
        if self.native is not None:
            groups = model.groups()
            for current, following in pairwise(groups):
                current.next = following
            for block in model.groups():
                _layout(block)
            model.layout_identity = hashlib.sha256(
                repr([(b.path, b.aliases, b.finals) for b in model.groups()]).encode()
            ).hexdigest()
            model.identity = model.wanted_identity()
            self.window_bytes = max(
                [self.window_bytes, *(_STREAMED * b.allocation_bytes for b in model.groups())]
            )
            self.native.set_streaming_limit(self.window_bytes)
        why = unpaged(module, rule, sizes)
        if why is not None:
            warnings.warn(f"{w.component}: {why}", RuntimeWarning, stacklevel=2)
        _placeholders(model)
        _hooks(model)
        self.models[module] = model
        return model

    def _configure_binding(
        self, model: Model, layout: BindingLayout, *, resident_only: bool = False
    ) -> Callable[[], None]:
        """Preflight a plain final-layout change; return its checked metadata transition.

        Every allocation here is metadata, never GPU weight storage. The existing fill path
        materializes the chosen representation after its new binding is retained.
        """
        if (
            not self.cuda
            or self.native is None
            or self.device.index
            not in (
                None,
                torch.cuda.current_device(),
            )
        ):
            raise StaleBinding("prepared dtype overrides require the same native CUDA device")
        if self.capturing():
            raise StaleBinding("change weight representations before graph capture")
        with self.lock:
            generation = model.generation
            targets: dict[str, str] = {}
            target_shapes: dict[str, tuple[int, ...]] = {}
            for block in model.groups():
                for key in block.parts:
                    declared = layout.layout(key)
                    if declared is None:
                        raise StaleBinding(
                            "prepared dtype overrides require explicit final layouts"
                        )
                    for name, part in declared.items():
                        if name in targets and (
                            targets[name] != part.dtype or target_shapes[name] != part.shape
                        ):
                            raise StaleBinding("a tied weight has conflicting final dtypes")
                        targets[name] = part.dtype
                        target_shapes[name] = part.shape

            def matches() -> bool:
                return (
                    not model.closed
                    and model.binding_identity == layout.identity
                    and all(
                        (value := site.table[site.name]) is not None
                        and tuple(value.shape) == target_shapes[name]
                        and value.dtype == getattr(torch, _DTYPES[dtype])
                        for name, dtype in targets.items()
                        for site in model.sites.get(name, ())
                    )
                )

            if matches():
                return lambda: None
            if resident_only:
                raise StaleBinding(
                    "resident-only capture cannot create a new weight representation"
                )
            if (
                model.closed
                or model.retiring
                or model.users
                or any(b.loading for b in model.groups())
            ):
                raise StaleBinding("weight representation is in use or being changed")
            stored_sites = {
                (id(site.table), site.name)
                for name in targets
                for site in model.sites.get(name, ())
            }
            for sites in model.sites.values():
                for site in sites:
                    value = site.table[site.name]
                    if (
                        (id(site.table), site.name) not in stored_sites
                        and value is not None
                        and value.is_floating_point()
                    ):
                        raise StaleBinding(
                            "prepared dtype overrides require declared floating buffers"
                        )
            layouts: list[tuple[_Block, _Block]] = []
            replacements: dict[int, tuple[list[_Site], Tensor]] = {}
            for block in model.groups():
                staged = _Block(model, block.path, block.index, block.parts)
                _layout(staged, binding=layout, target_dtypes=targets)
                layouts.append((block, staged))
                for key in block.parts:
                    for name, part in (layout.layout(key) or {}).items():
                        sites = model.sites[name]
                        if id(sites) not in replacements:
                            held = _placeholder(
                                model,
                                block,
                                sites,
                                getattr(torch, _DTYPES[part.dtype]),
                                staged.raw_views,
                            )
                            replacements[id(sites)] = sites, held
            layout_identity = hashlib.sha256(
                repr([(b.path, b.aliases, b.finals) for _, b in layouts]).encode()
            ).hexdigest()
            window = max([self.window_bytes, *(_STREAMED * b.allocation_bytes for _, b in layouts)])

        def apply() -> None:
            if self.capturing():
                raise StaleBinding("change weight representations before graph capture")
            with self.lock:
                if matches():
                    return
                if (
                    model.closed
                    or model.retiring
                    or model.users
                    or model.generation != generation
                    or any(b.loading for b in model.groups())
                ):
                    raise StaleBinding("model changed after weight representation preflight")
                model.retiring = True
            try:
                # Closing graphs can join CUDA or invoke Python owner destructors. Never
                # hold the residency metadata lock across those operations.
                if model.retire_graphs is not None:
                    model.retire_graphs()
                for hold in list(model.holds):
                    hold.shed()
                with self.lock:
                    if self._prefetch is not None and self._prefetch.block.model is model:
                        self._settle_prefetch(cancel=True)
                    if model.users or any(
                        b.count or b.retained or b.loading for b in model.groups()
                    ):
                        raise StaleBinding(
                            "retire every weight user before changing representation"
                        )
                    # Check again before changing any site. External mutation is not a cast.
                    for sites, held in replacements.values():
                        if any(
                            (current := site.table[site.name]) is None
                            or tuple(current.shape) != tuple(held.shape)
                            for site in sites
                        ):
                            raise StaleBinding(
                                "weight geometry changed during representation retirement"
                            )
                    fill = "stored" if model.fill is None else model.fill.identity
                    identity = (
                        None if fill is None else repr((fill, layout.identity, layout_identity))
                    )
                    # This can refuse before metadata publication. A larger temporary
                    # streaming margin is conservative if a subsequent cache drop fails.
                    self.native.set_streaming_limit(window)
                    self.window_bytes = window
                    old_model = (
                        model.finals,
                        model.binding_identity,
                        model.layout_identity,
                        model.identity,
                        model.generation,
                        model.prefix,
                    )
                    old_sites = [
                        (site, site.table[site.name])
                        for sites, _ in replacements.values()
                        for site in sites
                    ]
                    old_layouts = [
                        (block, block.aliases, block.finals, block.raw_views)
                        for block, _ in layouts
                    ]
                    # Reserve no bytes here. Old escaped aliases own their old reservations;
                    # a later fill gets fresh final backing, never their rounded contents.
                    for block, _ in layouts:
                        self._discard(block, forget=True)
                    try:
                        for sites, held in replacements.values():
                            for site in sites:
                                site.table[site.name] = held
                        for block, staged in layouts:
                            block.aliases, block.finals = staged.aliases, staged.finals
                            block.raw_views = staged.raw_views
                        model.finals = layout
                        model.binding_identity = layout.identity
                        model.layout_identity = layout_identity
                        model.identity = identity
                        model.generation += 1
                        model.prefix = None
                    except BaseException:
                        for site, value in old_sites:
                            site.table[site.name] = value
                        for block, aliases, finals, raw_views in old_layouts:
                            block.aliases, block.finals, block.raw_views = (
                                aliases,
                                finals,
                                raw_views,
                            )
                        (
                            model.finals,
                            model.binding_identity,
                            model.layout_identity,
                            model.identity,
                            model.generation,
                            model.prefix,
                        ) = old_model
                        raise
                    finally:
                        # Next use recomputes headroom even if only cache retirement succeeded.
                        self._phase_names = frozenset()
            finally:
                with self.lock:
                    model.retiring = False

        return apply

    def _declare(self, model: Model) -> None:
        if model.finals is not None and model.finals.identity != model.binding_identity:
            raise StaleBinding("an immutable binding layout changed; reconstruct the model")
        identity = model.wanted_identity()
        if identity != model.identity or identity is None:
            graphs.retire_stale(torch, model.module)
            if any(b.retained for b in model.groups()):
                raise StaleBinding("retire old graph bindings before changing a weight fill")
            for hold in list(model.holds):
                hold.shed()
            if any(b.count for b in model.groups()):
                raise ResidencyError("a weight fill cannot change during an active use")
            for block in model.groups():
                self._discard(block, forget=True)
            self._forget_baked(model)
            model.identity = identity
            model.generation += 1

    def _keep_baked(self, page: tuple[object, ...], tensor: Tensor, *, streams: bool) -> None:
        """A block's baked bytes, kept in the host tier for its next refill: a streaming block's
        from its first bake, a resident one's from its first refill."""
        if page in self._baked:
            return
        if not streams and page not in self._baked_once:
            self._baked_once.add(page)
            return
        if not self._host_room(tensor.numel()):
            return
        kept = self._pin(torch.zeros(tensor.numel(), dtype=torch.uint8))
        if kept is not None:
            kept.tensor.copy_(tensor)  # complete when this returns
            self._baked[page] = kept
            self._baked_bytes += tensor.numel()

    def _final_from_host(self, block: _Block, overlay: Hashable) -> None:
        """A cast block's refill from its kept final bytes: one DMA into the final region, with
        no store read, raw region or cast (`_host_final`)."""
        use = block.final_use
        page = _page(block, block.model.identity, overlay, final=True)
        found = self._baked.get(page)
        if found is None or found.tensor.numel() != use.tensor.numel():
            return
        began = time.perf_counter_ns()
        use.tensor.copy_(found.tensor)  # complete when this returns
        use.commit()
        self._baked.move_to_end(page)
        self._refills[page] = self._refills.get(page, 0) + 1
        block.overlay = overlay
        if (request := _REQUEST.get()) is not None:
            ns = time.perf_counter_ns() - began
            request.read(block, ns, found.tensor.numel(), origin="baked")

    def _keep_final(self, block: _Block, tensor: Tensor) -> None:
        """A cast block's final bytes, just written from its stored ones: kept in the host tier
        as a baked page is (`_keep_baked`), so its refills move the cast bytes, not the stored."""
        if not _host_final(block):
            return
        page = _page(block, block.model.identity, block.overlay, final=True)
        self._refills[page] = self._refills.get(page, 0) + 1
        self._keep_baked(page, tensor, streams=not block.preferred)

    def _want(self, block: _Block, page: tuple[object, ...]) -> None:
        """A store read of `page` completed: the page is pinned once the executor is idle
        (`pin_idle`), a streamed one's after this read, a resident one's after its second (another
        phase's room evicted it). Pinning runs at about the copy's own rate (5.5 GiB/s on an RTX
        3070 host), so it is never on a request's path: Qwen-Image int8 at 8 GB paid 5.4 s of it
        in its first warm request (Runtime 0.35.6). A canceled or refused read does not count.
        A hooked block's page is wanted only when its fill wrote nothing (`_ensure`), and a block
        the tier keeps cast keeps its final bytes instead (`_host_final`)."""
        if (
            not _host_final(block)
            and page not in self._baked
            and (not block.preferred or page in self._read_once)
        ):
            self._wanted[page] = block
        self._read_once.add(page)

    def host_pending(self) -> bool:
        """Whether pages wait for an idle executor to pin them (`pin_idle`)."""
        return bool(self._wanted) and self._host_tier and self.cuda

    def pin_idle(self, waiting: Callable[[], bool]) -> None:
        """Pin the wanted pages while the executor is idle, each read again from the store into
        host RAM and registered once filled, until a command arrives (`waiting`). A command waits
        at most one staging window's read: the page it interrupted is wanted first at the next
        idle."""
        while self.host_pending() and not waiting():
            with self.lock:
                # The page refilled most often per request first: it saves the most bytes per
                # byte of host RAM (an H3 DiT block refilled 8 times a call before a text
                # encoder's read once).
                page = max(self._wanted, key=lambda p: self._rate.get(p, 1))
                block = self._wanted.pop(page)
            model = block.model
            if page in self._baked or model.closed or model.identity != page[2]:
                continue
            if not self._host_room(block.storage_bytes) and not self._displace(page, block):
                continue  # kept pages are worth at least as much; a smaller wanted one may fit
            tensors = model.source.ck.components[model.source.component].tensors
            tensor = torch.zeros(block.storage_bytes, dtype=torch.uint8)  # gaps included
            offset = 0
            try:
                for key, roles in block.parts.items():
                    for part in roles.values():
                        for start in range(0, part.nbytes, _HOST_STAGING):
                            if waiting():
                                with self.lock:
                                    self._wanted[page] = block
                                    self._wanted.move_to_end(page, last=False)
                                return
                            stop = min(start + _HOST_STAGING, part.nbytes)
                            into = tensor[offset + start : offset + stop]
                            tensors.read_into(key, into, role=part.role or None, offset=start)
                        offset += (part.nbytes + 255) // 256 * 256
            except Exception:  # a store or fetch failure only leaves this page unpinned
                continue
            kept = self._pin(tensor)
            if kept is None:
                return
            with self.lock:
                if page not in self._baked and not model.closed and model.identity == page[2]:
                    self._baked[page] = kept
                    self._baked_bytes += block.storage_bytes
                else:
                    kept.release()

    def _displace(self, page: tuple[object, ...], block: _Block) -> bool:
        """Room for `page` from kept pages refilled less often per request, least often and then
        least recently used first; none go unless together they make the room. The tier follows
        use, not arrival: a pod's first construction no longer keeps its pages while a later
        one's most refilled blocks stream from the store (H3 LoRA'd calls on an RTX PRO 4000:
        177 against 216 s by load order, cl-683)."""
        rate = self._rate.get(page, 1)
        with self.lock:
            cheaper = sorted(
                (p for p in self._baked if self._rate.get(p, 1) < rate),
                key=lambda p: self._rate.get(p, 1),
            )
            freed, chosen = 0, []
            for kept in cheaper:
                if self._host_fits(block.storage_bytes - freed):
                    break
                chosen.append(kept)
                freed += self._baked[kept].tensor.numel()
            if not self._host_fits(block.storage_bytes - freed):
                return False
            for kept in chosen:
                gone = self._baked.pop(kept)
                self._baked_bytes -= gone.tensor.numel()
                self._baked_room += gone.tensor.numel()
                gone.release()
        return True

    def _host_fits(self, nbytes: int) -> bool:
        """Whether `nbytes` more fit the tier as last measured (`_host_room` measures)."""
        return (
            self._baked_bytes + nbytes <= self._host_cap
            and self._baked_room - nbytes >= _BAKED_HEADROOM
        )

    def _host_room(self, nbytes: int) -> bool:
        """Whether a host page of `nbytes` fits the tier: process lifetime, never on disk, each
        page once. Only into room beyond the headroom and within half of the host's RAM, evicting
        nothing for it: blocks refill in a fixed order, so replacing the least recently used
        would miss every refill."""
        if not self._host_tier or not self.cuda or self.capturing():
            return False
        now = time.monotonic()
        if now - self._host_looked >= 1.0:  # host RAM is re-read at most once a second
            self._host_looked = now
            self._trim_baked()
        return self._host_fits(nbytes)

    def _pin(self, tensor: Tensor) -> _HostPage | None:
        """`tensor` registered as a host page; a failed pin turns the tier off, with a warning."""
        try:
            return _HostPage(tensor)
        except (RuntimeError, OSError) as error:
            self._host_tier = False
            warnings.warn(
                f"weights stream from the store: pinning host memory failed ({error})",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def _trim_baked(self) -> int:
        """The host RAM this process can still take (its tightest cgroup's, on Linux), after kept
        pages go, those refilled least often per request first and then the least recently used,
        while it is under the headroom; -1 unreadable, and then they all go."""
        room = self._baked_room = (
            proctree.available_host_bytes() if sys.platform == "linux" else host_memory().free_bytes
        )
        if room >= _BAKED_HEADROOM or not self._baked:
            return room
        for key in sorted(self._baked, key=lambda p: self._rate.get(p, 1)):
            page = self._baked.pop(key)
            self._baked_bytes -= page.tensor.numel()
            page.release()
            if room >= 0:
                room += page.tensor.numel()
                if room >= _BAKED_HEADROOM:
                    break
        self._baked_room = room
        return room

    def _forget_baked(self, model: Model) -> None:
        """`model`'s kept pages go: its fill changed, or it closed."""
        for page in [page for page in self._baked if page[0] == model.name]:
            kept = self._baked.pop(page)
            self._baked_bytes -= kept.tensor.numel()
            kept.release()
        self._baked_once = {page for page in self._baked_once if page[0] != model.name}
        self._plain = {page for page in self._plain if page[0] != model.name}

    def _host_read(
        self, region: Any, kept: _HostPage, *, evict_completed: bool
    ) -> _HostRead | None:
        """A refill from a host page: the region's fill under the native prefetch's admission,
        one async DMA; None when the region is resident already. The copy stream exists before
        admission, and a fill admitted here that fails is released before its error leaves: a
        traceback holding it kept the region `already being filled` for the retry (cr-225)."""
        if self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream(self.device)
        stream = self._h2d_stream
        use = region.prepare(reclaim=False, evict_completed=evict_completed)
        if not use.needs_fill:
            return None
        try:
            use.consume_on(stream.cuda_stream)  # its release fences on the copy
            with torch.cuda.stream(stream):
                use.tensor.copy_(kept.tensor, non_blocking=True)
                event = torch.cuda.Event()
                event.record(stream)
        except BaseException:
            use = None  # the fill and its tensor go now, not with the traceback
            raise
        kept.event = event
        return _HostRead(use, event, kept.tensor.numel())

    def _switch_overlay(self, block: _Block, overlay: Hashable) -> None:
        """`block`'s page was filled under another overlay than this use wants (a base page for
        a LoRA forward, or the reverse). As for a changed fill, its model's holds go first. A
        native page is parked under its overlay, first for reclaim to take, and the wanted
        overlay's parked page serves again: alternating overlays refill nothing while both fit.
        Off native backing the page is dropped and the next fill writes the wanted overlay."""
        for hold in list(block.model.holds):
            hold.shed()
        if block.count or block.retained or block.loading or self.capturing():
            raise StaleBinding("a weight block in use cannot change its fill overlay")
        if self.native is None:
            self._discard(block, forget=True)
            if block.paged():
                raise StaleBinding("a weight block's page under another overlay was not dropped")
            return
        self._discard(block)
        current = (block.region, block.final_region)
        if current != (None, None):
            for region in current:
                if region is not None:
                    region.set_preferred(False)
            block.parked[block.overlay] = current
        while len(block.parked) > _PARKED:
            block.parked.pop(next(iter(block.parked)))
        block.region, block.final_region = block.parked.pop(overlay, (None, None))
        block.overlay = overlay
        block.preferred = block.preference_set = False
        block.generation += 1

    def _idle(self, block: _Block) -> None:
        if self.native is not None and not (block.count or block.retained or block.loading):
            # The DLPack storage owner holds the physical pin. A surviving external alias
            # retains it after these references disappear; native eviction waits on the
            # region's last-consumer events (`_used`). record_stream is not that ownership fence.
            block.data = block.assigned = block.ready = None
            block.native_storage = block.native_use = None
            self._drop_final(block)

    def _drop_final(self, block: _Block, *, forget: bool = False) -> None:
        if block.final_use is not None and block.final_use.needs_fill:
            # A failed writer may have exported an alias. The next attempt needs a new
            # reservation; that alias retains its old owner and unspecified partial bytes.
            forget = True
        block.final_use = block.final_data = block.final_storage = block.final_native_use = None
        if forget:
            block.final_region = None

    def _discard(self, block: _Block, *, forget: bool = False) -> int:
        with self.lock:
            if (
                block.count
                or block.retained
                or block.loading
                or (block.model.users and block.model.thread != threading.get_ident())
            ):
                return 0
            if self._prefetch is not None and self._prefetch.block is block:
                self._settle_prefetch(cancel=True)
            present = block.data is not None
            block.data = block.assigned = block.ready = None
            block.native_storage = block.native_use = None
            self._drop_final(block, forget=forget)
            if forget:
                # Never overwrite an old fill beneath an escaped alias. Its pin keeps
                # the old reservation alive independently of this model's new identity.
                block.region = None
                block.parked.clear()
                block.preferred = False
                block.preference_set = False
            if present or forget:
                block.generation += 1
            return block.nbytes if present else 0

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
        available = self.room(current).free if need else 0
        if need and available < need:
            # Return actual unused activation storage before sacrificing reusable weights.
            # Private graph pools and live tensors may remain; measure instead of crediting
            # reserved-minus-allocated bytes as available memory.
            release_cached(torch, self.device.type)
            available = self.room(current).free
        dropped = 0

        def enough(released: int = 0, *, flush: bool = False) -> bool:
            nonlocal available, dropped
            if not need:
                return False
            dropped += released
            if flush or dropped >= max(need - available, 1):
                # Dropped owner bytes only trigger a batch flush. Aliases, allocator
                # fragmentation and graph pools can keep them resident; measure again.
                available = self.room(current).free
                if available < need:
                    release_cached(torch, self.device.type)
                    available = self.room(current).free
                if self.native is not None and available < need:
                    self.native.reclaim(need - available)
                    available = self.room(current).free
                dropped = 0
            return available >= need

        with self.lock:
            holds = [
                hold
                for model in self.models.values()
                if not model.users
                for hold in model.holds
                if hold.reclaimable
            ]
        for hold in holds:
            hold.shed()
        if self.native is not None:
            self.native.reclaim(
                max(need - available, 0) if need else int(self.native.stats()["resident_bytes"])
            )
            available = self.room(current).free if need else 0
        with self.lock:
            blocks = sorted(
                (b for m in self.models.values() for b in m.groups()),
                key=lambda b: (b.model.name in current, b.last),
            )
        for block in blocks:
            if enough():
                break
            enough(self._discard(block))
        # Return public allocator cache before escalating to graph retirement.
        if not enough(flush=True):
            for model in sorted(list(self.models.values()), key=lambda m: m.name in current):
                if enough():
                    break
                if self._retire(model):
                    if enough(flush=True):
                        break
                    for block in model.groups():
                        if enough():
                            break
                        enough(self._discard(block))
                    enough(flush=True)
        # A callback may discard graph owners or decoded caches. It runs outside metadata
        # locks, and a pinned/locked region is skipped rather than waited on.
        if not enough(flush=True):
            for region in sorted(list(self.regions), key=lambda r: r.last):
                if enough():
                    break
                if region.discard():
                    enough(flush=True)
        if not need:
            if self.native is not None:
                self.native.reclaim(int(self.native.stats()["resident_bytes"]))
            release_cached(torch, self.device.type)
        allocated, reserved, _ = self._figures()
        free = int(torch.cuda.mem_get_info(self.device)[0]) if self.cuda else 0
        return max(before_allocated - allocated, before_reserved - reserved, free - before_free, 0)

    def trim(self) -> int:
        """Idle offload: release eligible owners and return actual reserved-byte reduction.

        The host tier goes too. Its pages are pinned, so no other process can reclaim them, and
        the call the machine offloads this executor for needs host RAM as well: beside an idle
        H3 executor's 42 GB of pages, SeedVR2's 2160p decode on a 4090 pod was killed by the
        kernel (gate 40991023097). Its pages refill from the store on this executor's next call."""
        if self.capturing():
            raise ResidencyError("idle offload cannot run during graph capture")
        before = self._figures()[1]
        for model in list(self.models.values()):
            self._retire(model)
        self.reclaim()
        with self.lock:
            pages = list(self._baked.values())
            self._baked.clear()
            self._baked_bytes = 0
            self._wanted.clear()
        for page in pages:
            page.release()
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
        if self.device.type == "mps" and reclaim:
            # TensorFS reads into bounded host staging; account for that transient beside
            # the retained MPS part before asking Torch for its destination allocation.
            need = nbytes + min(nbytes, _HOST_STAGING)
            if need > self.room().free:
                self.reclaim(need)
        try:
            return torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        except RuntimeError as error:
            # MPS reports allocator refusal as RuntimeError. Classify only this failed
            # allocation; reads, transforms and model execution are outside the retry.
            if not isinstance(error, torch.OutOfMemoryError) and not (
                self.device.type == "mps" and fault(error) is Fault.OOM
            ):
                raise
            if not reclaim or self.capturing() or not self.reclaim():
                raise
            # Only the failed allocation is retried. No read, fill transform or forward ran.
            return torch.empty(nbytes, dtype=torch.uint8, device=self.device)

    def _active_beside(self, request: Request | None) -> bool:
        """Live request concurrency; the caller holds lock across its policy decision."""
        return any(other is not request and not other.idle for other in self.requests)

    def _headroom(self) -> int:
        """The room a preferred block leaves free beside it: the streaming window, unless the
        active phase fits."""
        return 0 if self._fits else self.window_bytes

    def _native_region(
        self, block: _Block, *, final: bool = False, speculative: bool = False
    ) -> Any:
        model = block.model
        field = "final_region" if final else "region"
        region = getattr(block, field)
        size = block.final_bytes if final else block.storage_bytes
        request = _REQUEST.get()
        if request is not None and request.idle and (region is None or not region.resident):
            raise StaleBinding("idle graph preparation cannot fill native weight backing")
        created = region is None
        if created:
            region = self.native.reserve(size)
            setattr(block, field, region)
        if speculative:
            return region  # Only demand may assign or shorten the retained prefix.
        if not block.preference_set:
            missing = block.allocation_bytes - sum(
                int(held.allocated_bytes)
                for held in (block.region, block.final_region)
                if held is not None and held.resident
            )
            block.preferred = model.prefix is None
            if block.preferred and not block.cached():
                wanted = missing + self._headroom()
                free = self.room().free
                if (
                    self.phase_supported
                    and wanted > free
                    and not self.capturing()
                    and not (request is not None and request.idle)
                ):
                    # Registration uses this same lock: another request cannot enter
                    # between the live-concurrency check and preference reclamation.
                    with self.lock:
                        if not self._active_beside(request):
                            free = self.room().free
                            if wanted > free:
                                # Torch's idle cache goes before another phase's weights (a cold
                                # Krea 2 DiT fill evicted its 4.3 GB text encoder beside it).
                                release_cached(torch, self.device.type)
                                free = self.room().free
                            if wanted > free:
                                self.native.reclaim_unpreferred(wanted - free)
                                free = self.room().free
                block.preferred = free >= wanted
            block.preference_set = True
            if not block.preferred:
                model.prefix = (
                    min(model.prefix, block.index) if model.prefix is not None else block.index
                )
            for held, protect in (
                (block.region, not block.final_only()),
                (block.final_region, True),
            ):
                if held is not None:
                    held.set_preferred(block.preferred and protect)
        elif created:
            region.set_preferred(block.preferred and (final or not block.final_only()))
        if block.preferred and (final or not block.final_only()) and not region.preferred:
            # Native activation recovery reduces the prefix without calling Python. Keep
            # later unprepared blocks on the streaming side of that same watermark.
            model.prefix = (
                min(model.prefix, block.index) if model.prefix is not None else block.index
            )
            block.preferred = False
        return region

    def _activate_phase(self, model: Model) -> None:
        """Prioritize an actual component phase, never each repeated block/denoising call."""
        request = _REQUEST.get()
        if not self.phase_supported or self.capturing() or (request is not None and request.idle):
            return
        names = request.phase(model.name) if request is not None else frozenset({model.name})
        request_id = request.id if request is not None else None
        with self.lock:
            if self._active_beside(request):
                return
            if names == self._phase_names and request_id == self._phase_request:
                return
            self._settle_prefetch(cancel=True)
            self.native.reset_preferences()  # metadata only: pins, mappings and VA stay intact
            self._phase_names, self._phase_request = names, request_id
            incoming = [m for m in self.models.values() if m.name in names]
            # An unrelated registered model must not enlarge this phase's streaming
            # headroom. Native admission still permits an unexpectedly larger group.
            self.window_bytes = _STREAMED * max(
                (b.allocation_bytes for m in incoming for b in m.groups()), default=0
            )
            self.native.set_streaming_limit(self.window_bytes)
            missing = 0
            for member in incoming:
                member.prefix = None
                for block in member.groups():
                    block.preferred = block.preference_set = False
                    required = [(block.final_region, block.final_bytes)]
                    if not block.final_only():
                        required.append((block.region, block.storage_bytes))
                    for region, size in required:
                        if not size:
                            continue
                        if region is not None and region.resident:
                            region.set_preferred(True)
                        else:
                            missing += (
                                (size + self.granularity - 1) // self.granularity * self.granularity
                            )
                    if block.cached():
                        block.preferred = block.preference_set = True
            allocated, before, _ = _torch_bytes(self.device)
            unused = max(before - allocated, 0)
            if unused:
                # The previous phase's inactive Torch segments are neither weight capacity nor
                # this phase's activations. Returned once here, not by Torch inside the first op
                # of this phase that does not fit (Krea 2's 2048² decode: 1.1 s, not 1.3 s).
                torch.cuda.empty_cache()
            returned = max(before - _torch_bytes(self.device)[1], 0)
            # Protect the incoming cached bytes above; only other phases' idle backing may go.
            # A phase whose missing bytes then fit streams nothing and reserves no window: a
            # window beside resident weights only evicts another phase's (a 4.3 GB text
            # encoder's 13 GB window took 3.9 GB of an idle DiT every request).
            need = missing - self.room().free
            reclaimed = int(self.native.reclaim_unpreferred(need)) if need > 0 else 0
            self._fits = missing <= self.room().free
            need = 0 if self._fits else missing + self.window_bytes - self.room().free
            reclaimed += int(self.native.reclaim_unpreferred(need)) if need > 0 else 0
            if request is not None:
                request.phase_changes.append(
                    {
                        "models": sorted(names),
                        "window_bytes": self.window_bytes,
                        "fits": self._fits,
                        "missing_bytes": missing,
                        "torch_unused_bytes": unused,
                        "torch_returned_bytes": returned,
                        "native_reclaimed_bytes": reclaimed,
                        # Cumulative snapshot; an older native wheel has no reason counters.
                        "native_reclaimed_by_reason": self.native.stats().get(
                            "reclaimed_by_reason"
                        ),
                    }
                )

    def _prepare_native(self, block: _Block, reclaim: bool, *, final: bool = False) -> Any:
        model = block.model
        region = self._native_region(block, final=final)
        needed = int(region.allocated_bytes)
        if (
            reclaim
            and self.budget_bytes is None
            and not region.resident
            and not self.capturing()
            and needed > self.room().free
            and (figures := _torch_bytes(self.device))[1] > figures[0]
        ):
            # Common weights may bind before their model's first forward enters its phase.
            # Return unused Torch cache first; native allocation still decides whether idle
            # weights actually need eviction. Budgeted demand already does this in reclaim().
            release_cached(torch, self.device.type)
        if not region.resident and (not reclaim or self.budget_bytes is not None):
            if reclaim and needed > self.room().free:
                self.reclaim(needed, current={model.name})
            if needed > self.room().free:
                raise torch.OutOfMemoryError("the weight block exceeds currently free local memory")
        # Converted raw bytes are working storage for an admitted preferred final block.
        # Avoid charging that temporary Fill against the idle streaming window; native
        # prefix reductions and real allocation pressure still apply to the preparation.
        provisional = (
            not final
            and block.final_only()
            and block.preferred
            and block.final_region is not None
            and block.final_region.preferred
            and not region.preferred
            and not region.resident
        )
        try:
            try:
                if provisional:
                    region.set_preferred(True)
                return region.prepare()
            finally:
                if provisional:
                    region.set_preferred(False)  # the returned Fill owns its physical pin
        except torch.OutOfMemoryError:
            if not reclaim:
                raise
            # No read or transform has run. A larger block or a different model may need
            # a smaller resident prefix. Retire eligible owners only after this miss;
            # retry the allocation only if reclamation actually returned memory.
            block.preferred = False
            model.prefix = (
                min(model.prefix, block.index) if model.prefix is not None else block.index
            )
            region.set_preferred(False)
            if not self.reclaim(needed, current={model.name}):
                raise
            return region.prepare()

    def _stage_prefetch_window(
        self, required: int, missing: int, request: Request | None, owner: object
    ) -> Callable[[], None] | None:
        """Admit bounded next-source speculation after actual graph/readout allocations.

        The streaming limit bounds idle nonpreferred cache, not active physical residency.
        Existing useful cache may remain beyond the current/next source pair. This never
        reclaims. Driver free and the request budget already exclude Torch
        reservations (including graph scratch and boundary slots) and native residency.
        The speculative native allocation independently rechecks pressure at read time.
        """
        if self.native is None or request is None or request.memory is not self or request.idle:
            return None
        with self.lock:
            native_limit = self.native.stats().get("streaming_limit_bytes")
            if (
                native_limit is None
                or native_limit != self.window_bytes
                or self._stage_window_owner is not None
                or self._prefetch is not None
                or self._active_beside(request)
                or missing > self.room().free
            ):
                return None
            previous = self.window_bytes
            admitted = max(previous, required)
            phase = self._phase_names, self._phase_request
            self.native.set_streaming_limit(admitted)
            self.window_bytes = admitted
            self._stage_window_owner = owner

        def restore() -> None:
            # Called only after checked whole-program completion, including canceled or
            # unconsumed prefetch. Never overwrite a later phase's independently changed policy.
            with self.lock:
                if self._stage_window_owner is not owner:
                    return
                if (
                    self.window_bytes == admitted
                    and (self._phase_names, self._phase_request) == phase
                    and self.native.stats().get("streaming_limit_bytes") == admitted
                ):
                    self.native.set_streaming_limit(previous)
                    self.window_bytes = previous
                self._stage_window_owner = None

        return restore

    def _prefetch_next(self, current: _Block) -> None:
        """Demand's own next block takes priority over queued-request speculation."""
        if current.next is None or current.retained:
            return
        with self.lock:
            if self._prefetch is not None and self._prefetch.queued is not None:
                self._settle_prefetch(cancel=True)
        self._prefetch_block(current.next, _REQUEST.get())

    def _prefetch_queued(self, request: Request) -> str | None:
        """Offer one raw block; return its queued assignment even when capacity refuses.

        None means no compatible assignment was ready, or current demand owned the slot.
        A resident execution may try again after CPU preflight finishes, without repeatedly
        offering the same queued assignment or changing its current memory policy.
        """
        if (
            request.queued is None
            or request.idle
            or not self.prefetch_supported
            or self.capturing()
        ):
            return
        queued = request.queued()
        if queued is None or not queued.valid():
            return
        with self.lock:
            if self._prefetch is not None or self.requests != {request}:
                return
            for model in queued.models:
                if model.memory is not self or model.closed or model.retiring:
                    return
                for block in model.groups():
                    if not block.cached():
                        # One offer, one block; no scan that admits a second reader after refusal.
                        self._prefetch_block(block, request, queued=queued)
                        return queued.request_id
            return queued.request_id  # Its first phase is already cached; no device work.

    def cancel_queued(self, request_id: str) -> None:
        """Invalidate only the named native reader; its current owner drains it later."""
        pending = self._prefetch
        if pending is not None and pending.queued is not None:
            if pending.queued.request_id == request_id:
                # A strong local reference survives concurrent settlement. Native cancel does
                # not join; no Memory or prepared-record lock is needed on the Mux thread.
                pending.read.cancel()

    def _prefetch_block(
        self,
        block: _Block,
        request: Request | None,
        *,
        queued: QueuedPrefetch | None = None,
        released: bool = False,
    ) -> None:
        """Admit one immutable read, without eviction or Python fill callbacks: a page the host
        tier keeps (stored, baked or cast) by one DMA, else the stored bytes, and for a hooked
        block only when its fill writes nothing under the overlay this forward arms. A cast block
        the tier keeps cast (`_host_final`) prefetches only its kept final bytes."""
        if (
            not self.prefetch_supported
            or self.capturing()
            or (request is not None and request.idle)
            or block.aliases is None
            or not block.storage_bytes
        ):
            return
        overlay = None
        if block.model.fill is not None:
            if queued is not None:
                return  # another request's hooks are armed by its own code, not yet run
            overlay = block.model.fill.overlay(block.path, block.parts)
            if overlay != block.overlay:
                return  # demand parks or refills a page of another overlay (`_switch_overlay`)
        final = _host_final(block)
        page = _page(block, block.model.identity, overlay, final=final)
        size = block.final_bytes if final else block.storage_bytes
        with self.lock:
            if (
                self._prefetch is not None
                or block.loading
                or block.count
                or block.retained
                or block.cached()
            ):
                return
            if block.finals and not final and block.region is not None and block.region.resident:
                return  # its stored bytes are resident; demand writes the final ones
            kept = self._baked.get(page) if self._host_prepare else None
            if kept is not None and kept.tensor.numel() != size:
                kept = None
            hooked = block.model.fill is not None and page not in self._plain
            if kept is None and (final or hooked):
                return  # only a kept page, or a hooked one whose fill writes nothing, is read
            # A queued request's read is admitted only into free room. Demand's next block may
            # also take the room of the oldest streamed block whose consumers completed: Varena
            # evicts it on its last-consumer events and refuses rather than wait or join.
            needed = block.allocation_bytes + self.window_bytes
            if queued is not None and needed > self.room().free:
                if request is not None:
                    request.prefetch_skipped += 1
                    if queued is not None:
                        request.queued_prefetch.append(
                            {
                                "request": queued.request_id,
                                "state": "capacity_skip",
                                "bytes": needed,
                            }
                        )
                return
            offset = 0
            parts = []
            for key, roles in block.parts.items():
                for part in roles.values():
                    name = f"{block.model.component}/{key}#{part.role or 'value'}"
                    parts.append((name, offset, part.nbytes))
                    offset += (part.nbytes + 255) // 256 * 256
            retry = False
            try:
                region = self._native_region(block, final=final, speculative=True)
                # Outgoing cached weights may already fill the nonpreferred window even
                # when the active phase has ample resident-prefix room. Admit its next
                # block under that same soft preference without growing a cut prefix.
                preferred = bool(region.preferred)
                provisional = (
                    queued is None
                    and self.phase_supported
                    and not preferred
                    and block.model.prefix is None
                    and block.allocation_bytes + self._headroom() <= self.room().free
                )
                try:
                    if provisional:
                        region.set_preferred(True)  # native end-of-prefix watermark still wins
                    if kept is not None:
                        read = self._host_read(region, kept, evict_completed=queued is None)
                        if read is None:
                            return  # resident already
                    else:
                        read = region.prepare_read(
                            block.model.source.ck,
                            parts,
                            reclaim=False,
                            evict_completed=queued is None,
                        )
                finally:
                    if provisional:
                        # The prepared Fill owns its pin. Only demand may assign a lasting
                        # preference, so cancellation/failure cannot leave this one behind.
                        region.set_preferred(preferred)
            except (torch.OutOfMemoryError, torch.AcceleratorError) as error:
                if fault(error) is not Fault.OOM:
                    raise
                free = int(torch.cuda.mem_get_info(self.device)[0])
                allocated, reserved, _ = _torch_bytes(self.device)
                retry = (
                    not released
                    and queued is None
                    and free < block.allocation_bytes <= (free + reserved - allocated)
                )
                if not retry:
                    if request is not None:
                        request.prefetch_skipped += 1
                    return
            except tensorfs.Error as error:
                if fault(error) is not Fault.OTHER:
                    raise
                if request is not None:
                    request.prefetch_failed += 1
                return
            except (RuntimeError, ValueError) as error:
                if queued is None or fault(error) not in (Fault.OTHER, Fault.OOM):
                    raise
                if request is not None:
                    request.prefetch_failed += 1
                return
            if retry:
                # Demand's next read takes Torch's idle cache before it is refused. Refused, it
                # is read on the request path instead (Qwen-Image int8 on 8 GB: the last streamed
                # block, every step, 9 s per request). Out of the handler, so nothing the refused
                # attempt admitted lives on in its traceback (cr-225).
                release_cached(torch, self.device.type)
                if request is not None:
                    request.prefetch_released += 1
                return self._prefetch_block(block, request, released=True)
            identity = block.model.identity
            assert identity is not None  # eligibility excludes mutable/undeclared fills
            self._prefetch = _Prefetch(
                block, region, read, identity, request, queued, overlay=overlay, final=final
            )
            try:
                # Publication precedes the last validity check, so a concurrent discard either
                # cancels this exact read or is observed here before any I/O starts.
                if queued is not None and not queued.valid():
                    self._settle_prefetch(cancel=True)
                    return
                read.start()
            except BaseException as error:
                self._settle_prefetch(cancel=True)  # a failed start still owns its admitted Fill
                if (
                    queued is not None
                    and isinstance(error, (tensorfs.Error, RuntimeError, ValueError))
                    and fault(error) in (Fault.OTHER, Fault.OOM)
                ):
                    if request is not None:
                        request.prefetch_failed += 1
                    return
                raise
            if request is not None:
                request.prefetch_started += 1
                if queued is not None:
                    request.queued_prefetch.append(
                        {
                            "request": queued.request_id,
                            "state": "started",
                            "set": block.model.name,
                            "block": block.index,
                            "bytes": block.allocation_bytes,
                            "at_ns": time.time_ns(),
                        }
                    )

    def _settle_prefetch(self, *, cancel: bool = False) -> None:
        """Checked completion owns the source and Fill independently of Python model views."""
        with self.lock:
            pending = self._prefetch
            if pending is None:
                return
            block, read, request = pending.block, pending.read, pending.request
            cancel = (
                cancel
                or pending.identity != block.model.wanted_identity()
                or (pending.queued is not None and not pending.queued.valid())
            )
            if cancel:
                read.cancel()
            began = time.perf_counter_ns()
            complete = False
            try:
                context = CURRENT.get()
                try:
                    if context is not None and not cancel:
                        # This is a host transition between blocks, never the GPU loop.
                        context.raise_if_cancelled()
                        while not read.poll()["done"]:
                            context.raise_if_cancelled()
                            time.sleep(0.001)
                    row = read.wait()
                except BaseException:
                    read.cancel()
                    try:
                        drained = read.wait()  # retain through checked producer drain
                    except (
                        tensorfs.Error,
                        torch.OutOfMemoryError,
                        torch.AcceleratorError,
                    ) as error:
                        if fault(error) not in (Fault.OTHER, Fault.OOM):
                            raise  # uncertain/sticky completion takes precedence over cancellation
                        # Ordinary native errors already prove a checked drain. Preserve the
                        # original exception (notably Context cancellation) at this boundary.
                    else:
                        if request is not None:
                            request.prefetch_canceled += int(drained["canceled"])
                    raise
                complete = not row["canceled"]
                if request is not None:
                    if pending.queued is not None:
                        request.queued_prefetch.append(
                            {
                                "request": pending.queued.request_id,
                                "state": "ready" if complete else "canceled",
                                "set": block.model.name,
                                "block": block.index,
                                "bytes": int(row.get("bytes", 0)),
                                "at_ns": time.time_ns(),
                                "native": dict(row),
                            }
                        )
                    request.prefetch_canceled += int(not complete)
                    if complete:
                        request.prefetch_completed += 1
                        request.read(
                            block,
                            0,
                            int(row.get("bytes", 0)),
                            speculative=True,
                            origin="baked" if row.get("host") else "store",
                        )
                if complete:
                    page = _page(block, pending.identity, pending.overlay, final=pending.final)
                    self._refills[page] = self._refills.get(page, 0) + 1
                    if not row.get("host"):
                        self._want(block, page)
            except (tensorfs.Error, torch.OutOfMemoryError, torch.AcceleratorError) as error:
                if fault(error) not in (Fault.OTHER, Fault.OOM):
                    raise
                # A speculative local-only source may be absent. Demand uses the normal
                # reader/fetch path after the native adapter has checked its producer drain.
                if request is not None:
                    request.prefetch_failed += 1
            finally:
                if request is not None:
                    request.prefetch_wait_ns += time.perf_counter_ns() - began
                field = "final_region" if pending.final else "region"
                if not complete and getattr(block, field) is pending.region:
                    setattr(block, field, None)
                    block.preferred = False
                    block.preference_set = False
                self._prefetch = None

    def _final_views(self, block: _Block, reclaim: bool) -> None:
        if not block.finals:
            return
        use = self._prepare_native(block, reclaim, final=True) if block.final_bytes else None
        block.final_use = block.final_native_use = use
        request = _REQUEST.get()
        if use is not None and use.needs_fill and request is not None and request.idle:
            raise StaleBinding("final weight backing changed before idle graph preparation")
        raw = use.tensor if use is not None else self._empty(0, False)
        if use is not None:
            block.final_storage = int(raw.untyped_storage().data_ptr())
            if use.needs_fill:
                raw.zero_()  # includes all alignment gaps and padded strided extents
            else:
                block.final_hits += 1
                if request is not None:
                    request.final_hits += 1
        offset = 0
        data: dict[str, dict[str, Tensor]] = {}
        for key, parts in block.finals.items():
            row = data[key] = {}
            for name, part in parts.items():
                flat = raw[offset : offset + part.nbytes].view(getattr(torch, _DTYPES[part.dtype]))
                row[name] = flat.as_strided(part.shape, part.strides)
                offset += (part.nbytes + 255) // 256 * 256
        block.final_data = data

    @torch.inference_mode(False)  # type: ignore[untyped-decorator]
    def _ensure(self, block: _Block, reclaim: bool, overlay: Hashable) -> None:
        if block.data is not None:
            return
        if self.capturing():
            raise ResidencyError("all graph weights must be resident before capture")
        with self.lock:
            block.loading = True
        data: dict[str, dict[str, Tensor]] = {}
        read_bytes = read_ns = 0
        use = kept = None
        kept_ns = 0
        try:
            with self.lock:
                if self._prefetch is not None:
                    self._settle_prefetch(cancel=self._prefetch.block is not block)
            source = block.model.source
            tensors = source.ck.components[source.component].tensors
            if self.native is not None:
                self._final_views(block, reclaim)
                final = block.final_use
                if _host_final(block) and final is not None and final.needs_fill:
                    self._final_from_host(block, overlay)
                if block.final_only() and (
                    block.final_use is None or not block.final_use.needs_fill
                ):
                    # Final bytes are sufficient for this binding. Their owning views are
                    # pinned already, so neither an eviction race nor missing raw backing
                    # can trigger a redundant source read or decode.
                    block.data = {}
                    block.generation += 1
                    return
            if self.native is not None and block.storage_bytes:
                use = self._prepare_native(block, reclaim)
                request = _REQUEST.get()
                if use.needs_fill and request is not None and request.idle:
                    raise StaleBinding(
                        "native weight backing changed before idle graph preparation"
                    )
                if use.needs_fill:
                    # commit() publishes the whole logical region, including alignment
                    # gaps between stored parts. Initialize them before publication.
                    use.tensor.zero_()
                block.native_use = use
                block.native_storage = int(use.tensor.untyped_storage().data_ptr())
            page: tuple[object, ...] | None = None
            if use is not None and use.needs_fill:
                page = _page(block, block.model.identity, overlay)
                self._refills[page] = self._refills.get(page, 0) + 1
                if self._baked:
                    self._trim_baked()  # host pressure gives kept pages back here
                found = self._baked.get(page)
                if found is not None and found.tensor.numel() == use.tensor.numel():
                    # The page's host copy (its first store read, or bake): one DMA.
                    kept, began = found.tensor, time.perf_counter_ns()
                    use.tensor.copy_(kept)  # complete when this returns
                    kept_ns = time.perf_counter_ns() - began
                    self._baked.move_to_end(page)
            offset = 0
            for key, parts in block.parts.items():
                views = data[key] = {}
                for role, part in parts.items():
                    raw = (
                        use.tensor[offset : offset + part.nbytes]
                        if use is not None
                        else self._empty(part.nbytes, reclaim)
                    )
                    offset += (part.nbytes + 255) // 256 * 256
                    if (use is None or use.needs_fill) and kept is None:
                        began = time.perf_counter_ns()
                        try:
                            self._read_into(tensors, key, raw, part.role or None)
                        except tensorfs.Error:
                            # A failed destination completion must retain its native owner.
                            # Preserve the existing sticky CUDA failure classification.
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
            changed = False
            if block.model.fill is not None and (use is None or use.needs_fill):
                # `overlay` was read on this thread just before; no package code ran since.
                changed = kept is None and block.model.fill(block.path, data)
                if changed and page is not None and use is not None:
                    self._keep_baked(page, use.tensor, streams=not block.preferred)
                block.overlay = overlay
            if page is not None and kept is None and not changed:
                with self.lock:  # this store read completed, and its bytes are the page
                    if block.model.fill is not None:
                        self._plain.add(page)
                    self._want(block, page)
            if use is not None:
                use.commit()
            ready = None
            if self.cuda:
                ready = torch.cuda.Event()
                ready.record(self.stream())
            with self.lock:
                block.data, block.ready = data, ready
                block.generation += 1
        except BaseException:
            # A failed fill may have exported storage to a transform. Retire this region
            # identity; those exports keep their own pins, never a future fill's bytes.
            if use is not None and use.needs_fill:
                block.region = None
                block.preferred = False
            block.native_storage = block.native_use = None
            self._drop_final(block)
            raise
        finally:
            request = _REQUEST.get()
            if request is not None and read_bytes:
                request.read(block, read_ns, read_bytes)
            if request is not None and kept is not None:
                request.read(block, kept_ns, kept.numel(), origin="baked")
            with self.lock:
                block.loading = False

    def _read_into(self, tensors: Any, key: str, raw: Tensor, role: str | None) -> None:
        if self.device.type != "mps":
            tensors.read_into(key, raw, role=role)
            return
        length = raw.numel()  # Stored carriers are contiguous U8, including packed parts.
        if not length:
            return
        host = torch.empty(min(length, _HOST_STAGING), dtype=torch.uint8, device="cpu")
        for offset in range(0, length, host.numel()):
            stop = min(offset + host.numel(), length)
            view = host[: stop - offset]
            tensors.read_into(key, view, role=role, offset=offset)
            # TensorFS owns only the CPU read. Torch owns MPS transfer and completion;
            # a blocking copy lets the next window reuse this same host allocation safely.
            raw[offset:stop].copy_(view, non_blocking=False)

    def _enter(self, block: _Block, reclaim: bool = True) -> None:
        declared = _CAPTURE_BLOCKS.get()
        if declared is not None and id(block) not in declared:
            raise CapabilityError(
                "model stage reads a block outside its declared paths",
                code="prepared_graph_ineligible",
            )
        global progress
        model = block.model
        overlay = None if model.fill is None else model.fill.overlay(block.path, block.parts)
        if overlay != block.overlay and (block.paged() or block.parked):
            self._switch_overlay(block, overlay)
        with self.lock:
            if not model.users or model.thread != threading.get_ident():
                raise ResidencyError("weight preparation requires an active model scope")
            block.count += 1
            nested = block.count > 1
        try:
            capturing = self.capturing()
            if capturing and (not nested or not block.retained or block.assigned is None):
                raise ResidencyError(
                    "capture requires an active prepared weight binding "
                    f"({model.component} block {block.path or '(common)'}: entered {block.count}, "
                    f"retained {block.retained}, assigned {block.assigned is not None})"
                )
            if not nested and not capturing:
                # Prefetch reads the block after this one in execution order. Registration
                # order lists a down/up block's attentions before its resnets (Diffusers'
                # UNets), so SDXL read 15 blocks a step on demand at 3 GB; the order a forward
                # actually takes replaces it, the last block's successor being the next
                # call's first.
                if model.entered is not None and model.entered is not block:
                    model.entered.next = block
                model.entered = block
            if not nested:
                request = _REQUEST.get()
                reclaim = reclaim and not (request is not None and request.idle)
                self._ensure(block, reclaim, overlay)
            # A cached Hold can keep count > 0 across sequential threads or streams.
            # Every consumer needs readiness in addition to storage ownership.
            # Capture's owner joined the prepared stream before capture began. Waiting on
            # its external readiness event now would violate CUDA capture isolation.
            if block.ready is not None and not capturing:
                self.stream().wait_event(block.ready)
            if not nested:
                if block.assigned is None:
                    assigned = _assign(block)
                    ready = block.ready
                    if self.cuda:
                        ready = torch.cuda.Event()
                        ready.record(self.stream())
                    # Binding/decoding and dtype conversion may enqueue more producer work
                    # than the fill alone. Publish their owners and completion together.
                    block.assigned, block.ready = assigned, ready
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
            # Varena records its own last-consumer event on each declared stream when the
            # storage's final alias goes, so reclaim waits on that region, not the context.
            for use in (block.native_use, block.final_native_use):
                if use is not None:
                    use.consume_on(stream.cuda_stream)
            for tensor in block.tensors():
                if tensor.untyped_storage().data_ptr() not in (
                    block.native_storage,
                    block.final_storage,
                ):
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
                    self._idle(block)

    def _read(
        self, block: _Block, site: _Site, raw_view: tuple[int, _Part] | None = None
    ) -> Tensor:
        global progress
        declared = _CAPTURE_BLOCKS.get()
        if declared is not None and id(block) not in declared:
            raise CapabilityError(
                "model stage reads a block outside its declared paths",
                code="prepared_graph_ineligible",
            )
        model = block.model
        model._claim()
        try:
            if not getattr(model.local, "depth", 0):
                self._declare(model)
            if (
                raw_view is not None
                and model.fill is None
                and not (block.count or block.retained or block.loading)
                and block.data is None
                and block.ready is None
                and ((pending := self._prefetch) is None or pending.block is not block)
                and block.region is not None
                and not self.capturing()
                and (pin := getattr(block.region, "pin", None)) is not None
            ):
                # The atomic ready-only pin never fills or reclaims. Its ordinary storage
                # alias owns the bytes until the consumer dies, without binding other sites.
                offset, part = raw_view
                stored = getattr(torch, _DTYPES[part.dtype])
                widen = _widened(model, site.name, site.table[site.name], stored)
                with torch.inference_mode(False):
                    use = pin()
                    if use is not None:
                        value = (
                            use.tensor[offset : offset + part.nbytes].view(stored).view(part.shape)
                        )
                        if widen is not None:
                            value = value.to(widen)
                        block.last = time.monotonic_ns()
                        progress += 1
                        return nn.Parameter(value, requires_grad=False) if site.parameter else value
            self._enter(block)
            try:
                value = site.table[site.name]
                if isinstance(value, Placeholder):
                    raise ResidencyError(
                        f"{model.name}: {site.name} has no value: no stored tensor binds it. A "
                        "buffer no checkpoint stores keeps its constructed value only when "
                        "construction computed one (from config or a constant), not from a "
                        "weightless parameter"
                    )
                return value
            finally:
                self._exit(block)
        finally:
            model._release()

    def _open(self, model: Model, key: str, *, executing: bool = False) -> None:
        if executing and (seen := ENTERED.get()) is not None:
            seen.add(model)
        model._claim()
        depth = getattr(model.local, "depth", 0)
        model.local.depth = depth + 1
        request = _REQUEST.get()
        outer_forward = False
        if executing:
            with self.lock:
                outer_forward = self._forward_depth == 0
                self._forward_depth += 1
        try:
            if not depth:
                self._declare(model)
            if outer_forward:
                self._activate_phase(model)
            if depth:
                return
            model.local.request = request
            if request is not None:
                request.current.add(model.name)
            if model.common is not None:
                self._enter(model.common)
        except BaseException:
            model.local.depth = depth
            if executing:
                with self.lock:
                    self._forward_depth -= 1
            if request is not None and not depth:
                request.current.discard(model.name)
            model._release()
            raise

    def _close(self, model: Model, *, executing: bool = False) -> None:
        try:
            model.local.depth -= 1
            if model.local.depth == 0:
                try:
                    try:
                        with self.lock:
                            if self._prefetch is not None and self._prefetch.block.model is model:
                                self._settle_prefetch(cancel=True)
                    finally:
                        if model.common is not None and not getattr(model.local, "lent", False):
                            self._exit(model.common)
                        model.local.lent = False
                finally:
                    request = getattr(model.local, "request", None)
                    if request is not None:
                        request.current.discard(model.name)
                        request.sample()
        finally:
            if executing:
                with self.lock:
                    self._forward_depth -= 1
            model._release()

    @contextmanager
    def request(
        self,
        id: str,
        plan: Sequence[object] = (),
        *,
        idle: bool = False,
        others: Collection[Model] = (),
    ) -> Iterator[Request]:
        """`others`: the process's other constructions' sets, whose cached weights are not this
        request's working set (`Request.sample`)."""
        request = Request(self, id, idle, others)
        request.plan(plan)
        with self.lock:
            if not idle and any(not other.idle for other in self.requests):
                request.overlap = True
                for other in self.requests:
                    if not other.idle:
                        other.overlap = True
            self.requests.add(request)
        token = _REQUEST.set(request)
        try:
            yield request
        finally:
            try:
                with self.lock:
                    if self._prefetch is not None and self._prefetch.request is request:
                        # Cancel/drain unfinished speculation. An already completed read
                        # keeps valid immutable bytes as ordinary unpinned cache for B.
                        self._settle_prefetch(cancel=True)
                with self.lock:
                    self._rate.update(self._refills)  # the pages this request refilled
                    self._refills.clear()
                request.sample()
                request.ended = request.stats()
                self.last = request.ended
            finally:
                with self.lock:
                    self.requests.discard(request)
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
            "weights": self.native.stats() if self.native is not None else None,
        }


class Request:
    """Request context; Torch observations are process-wide, including overlapping calls. Its
    `peak` is its own working set: other constructions' resident sets are left out."""

    def __init__(
        self, memory: Memory, id: str, idle: bool = False, others: Collection[Model] = ()
    ) -> None:
        self.memory, self.id, self.idle = memory, id, idle
        self.others = tuple(others)
        self.current: set[str] = set()
        self.steps: list[set[str]] = []
        self.declared_steps: list[set[str]] = []
        self.phase_changes: list[dict[str, Any]] = []
        self.peak = 0
        self.overlap = False  # Historical measurement attribution, never a live policy gate.
        self.moved = self.read_ns = 0
        self.baked_hits = self.baked_bytes = self.baked_ns = 0
        self.final_fills = self.final_hits = 0
        self.prefetch_started = self.prefetch_completed = self.prefetch_canceled = 0
        self.prefetch_skipped = self.prefetch_failed = self.prefetch_wait_ns = 0
        self.prefetch_released = 0
        self.trace: list[dict[str, object]] = []
        self.ended: dict[str, Any] | None = None
        self.queued: Callable[[], QueuedPrefetch | None] | None = None
        self.queued_prefetch: list[dict[str, Any]] = []
        self.sample()

    def prefetch_queued(self) -> str | None:
        """Only fully allocated resident executions may offer optional queued speculation."""
        return self.memory._prefetch_queued(self)

    def sample(self) -> None:
        peak = self.memory._figures()[2]
        if peak > self.peak and self.others:
            peak -= sum(_resident_bytes(b) for m in self.others for b in m.groups())
        self.peak = max(self.peak, peak)

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
        self.declared_steps = [set(step) for step in self.steps]

    def resolve(self, models: Mapping[str, Model]) -> None:
        """Resolve this request's public component names through its actual Placement."""
        for index, terms in enumerate(self.declared_steps):
            mapped = {model.name for name, model in models.items() if name in terms}
            self.steps[index].difference_update(models)
            self.steps[index].update(mapped)

    def phase(self, name: str) -> frozenset[str]:
        matching = [step for step in self.steps if name in step]
        return frozenset().union(*matching) if matching else frozenset({name})

    def room(self, step: int | None = None) -> Room:
        current = self.current if step is None or step >= len(self.steps) else self.steps[step]
        room = self.memory.room(current)
        if not self.memory.cuda or not torch.cuda.is_initialized():
            return room
        allocated, reserved, _ = _torch_bytes(self.memory.device)
        return replace(room, cached=max(reserved - allocated, 0))

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

    def read(
        self,
        block: _Block,
        ns: int,
        nbytes: int,
        *,
        speculative: bool = False,
        origin: str = "store",
    ) -> None:
        """A block filled: from the store, or (`baked`) from its kept baked bytes."""
        if origin == "store":
            self.moved += nbytes
            self.read_ns += ns
        else:
            self.baked_hits += 1
            self.baked_bytes += nbytes
            self.baked_ns += ns
        self.trace.append(
            {
                "sequence": len(self.trace),
                "set": block.model.name,
                "block": block.index,
                "bytes": nbytes,
                "origin": origin,
                "speculative": speculative,
            }
        )
        self.sample()

    def stats(self) -> dict[str, Any]:
        if self.ended is not None:
            return self.ended
        self.sample()
        allocated, reserved, _ = self.memory._figures()
        out: dict[str, Any] = {
            "request": self.id,
            "allocator_scope": "process",
            "overlapping": self.overlap,
            "sets": {
                m.name: sum(b.nbytes for b in m.groups()) for m in self.memory.models.values()
            },
            "moved": {
                "store": {"bytes": self.moved, "ns": self.read_ns},
                "baked": {"bytes": self.baked_bytes, "fills": self.baked_hits, "ns": self.baked_ns},
            },
            "baked_pages": {
                "scope": "the host tier: every streamed page's bytes, pinned",
                "pages": len(self.memory._baked),
                "bytes": self.memory._baked_bytes,
                "room": self.memory._baked_room,
                "cap": self.memory._host_cap,
                "pinned": self.memory._host_tier,
                "wanted": len(self.memory._wanted),
            },
            "trace": list(self.trace),
            "trace_scope": "fills",
            "phases": list(self.phase_changes),
            "phase_policy_supported": self.memory.phase_supported,
            "host": None,
        }
        if self.memory.cuda and torch.cuda.is_initialized():
            groups = [b for model in self.memory.models.values() for b in model.groups()]
            out.update(
                peak=self.peak,
                allocated=allocated,
                reserved=reserved,
                peak_window="Torch high-water plus sampled native backing, carried across resets; "
                "other constructions' resident sets left out",
                weight_residency=(
                    self.memory.native.stats() if self.memory.native is not None else None
                ),
                final_bindings={
                    "fills": self.final_fills,
                    "hits": self.final_hits,
                    "resident_scope": "process",
                    "raw_resident_bytes": sum(
                        int(b.region.allocated_bytes)
                        for b in groups
                        if b.region is not None and b.region.resident
                    ),
                    "final_resident_bytes": sum(
                        int(b.final_region.allocated_bytes)
                        for b in groups
                        if b.final_region is not None and b.final_region.resident
                    ),
                    "fallback_blocks": sum(b.aliases is None for b in groups),
                    "native_prefetch": self.memory.prefetch_supported,
                },
                prefetch={
                    "started": self.prefetch_started,
                    "completed": self.prefetch_completed,
                    "canceled": self.prefetch_canceled,
                    "skipped_capacity": self.prefetch_skipped,
                    "released_cache": self.prefetch_released,
                    "failed": self.prefetch_failed,
                    "wait_ns": self.prefetch_wait_ns,
                    "scope": "one exact raw block; current demand precedes queued speculation",
                    "queued": list(self.queued_prefetch),
                },
            )
        elif self.memory.device.type == "mps":
            out.update(allocated=allocated, reserved=reserved, measurement="observed_current")
        return out


def _resident_bytes(block: _Block) -> int:
    """The native backing `block` holds on the device now, raw and final."""
    return sum(
        int(region.allocated_bytes)
        for region in (block.region, block.final_region)
        if region is not None and region.resident
    )


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


def _strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    out = []
    for length in reversed(shape):
        out.append(stride)
        stride *= max(length, 1)
    return tuple(reversed(out))


def _layout(
    block: _Block,
    *,
    binding: BindingLayout | None = None,
    target_dtypes: Mapping[str, str] | None = None,
) -> None:
    """Validate final geometry before its placeholders or backing are published."""
    model = block.model
    binding = model.finals if binding is None else binding
    if binding is None and model.bind is not None:
        return  # An arbitrary binder has not declared immutable final bytes.
    aliases: dict[str, dict[str, str]] = {}
    finals: dict[str, dict[str, _Part]] = {}
    codes = {kind: code for code, kind in _DTYPES.items() if code != "F4"}
    for key, parts in block.parts.items():
        if binding is not None:
            declared = binding.layout(key)
            if declared is None:
                return  # The provider explicitly declines final-buffer caching here.
        else:
            name = model.source.bound.get(key, key) if model.source.bound else key
            sites = model.sites.get(name)
            if set(parts) != {"value"} or not sites:
                return
            target = sites[0].table[sites[0].name]
            code = codes.get(str(target.dtype).removeprefix("torch."))
            if code is None:
                return
            declared = {
                name: TensorLayout(code, tuple(target.shape), tuple(target.stride()), "value")
            }
        row_aliases = aliases[key] = {}
        row_finals = finals[key] = {}
        for name, layout in declared.items():
            if name not in model.sites:
                raise ResidencyError(f"{model.name}: final layout names no target {name}")
            shape = layout.shape
            strides = layout.strides or _strides(shape)
            target = model.sites[name][0].table[model.sites[name][0].name]
            if (
                tuple(target.shape) != shape
                or (target_dtypes or {}).get(
                    name, codes.get(str(target.dtype).removeprefix("torch."))
                )
                != layout.dtype
            ):
                raise ResidencyError(f"{name}: final layout disagrees with target shape or dtype")
            if layout.dtype not in _DTYPES or layout.dtype == "F4":
                raise ResidencyError(f"{name}: invalid final buffer geometry")
            count = layout.elements()
            dtype = getattr(torch, _DTYPES[layout.dtype])
            width = int(torch.empty((), dtype=dtype, device="meta").element_size())
            source = parts.get(layout.source) if layout.source is not None else None
            if (
                source is not None
                and source.dtype == layout.dtype
                and source.shape == shape
                and strides == _strides(source.shape)
                and source.nbytes == count * width
            ):
                assert layout.source is not None
                row_aliases[name] = layout.source
            else:
                row_finals[name] = _Part(layout.dtype, shape, count * width, "", strides)
    block.aliases = aliases
    block.finals = {key: row for key, row in finals.items() if row}
    if model.fill is None and not block.finals:
        offset = 0
        for key, parts in block.parts.items():
            for role, part in parts.items():
                for name, source in aliases[key].items():
                    if source == role:
                        for site in model.sites[name]:
                            block.raw_views[(id(site.table), site.name)] = (offset, part)
                offset += (part.nbytes + 255) // 256 * 256


def _placeholder(
    model: Model,
    block: _Block,
    sites: list[_Site],
    dtype: Any,
    raw_views: Mapping[tuple[int, str], tuple[int, _Part]],
) -> Tensor:
    meta = sites[0].table[sites[0].name]
    held = Placeholder(meta.shape, dtype, model.memory.device)
    if sites[0].parameter:
        held = nn.Parameter(held, requires_grad=False)

    def read(
        b: _Block = block,
        site: _Site = sites[0],
        raw_view: tuple[int, _Part] | None = raw_views.get((id(sites[0].table), sites[0].name)),
    ) -> Tensor:
        return model.memory._read(b, site, raw_view)

    held._varena_read = read
    return held


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
        held = _placeholder(model, block, sites, meta.dtype, block.raw_views)
        for site in sites:
            site.table[site.name] = held


def _page(
    block: _Block, identity: object, overlay: Hashable, *, final: bool = False
) -> tuple[object, ...]:
    """A block's host-tier page key: its set, path, fill identity and overlay, and `final` for
    its cast bytes."""
    page: tuple[object, ...] = (block.model.name, block.path, identity, overlay)
    return (*page, "final") if final else page


def _host_final(block: _Block) -> bool:
    """Whether the host tier keeps `block`'s final bytes rather than its stored ones: a block
    bound only from final buffers that are no larger than its stored bytes (an F32 VAE held in
    bf16 moves half). A decoded block's finals are larger, so its stored bytes are kept."""
    return block.final_only() and block.final_bytes <= block.storage_bytes


def _widened(
    model: Model, name: str, site: Tensor | None, served: torch.dtype
) -> torch.dtype | None:
    """The dtype a site widens `served` bytes to on each bind, or None. A placed component
    serves its binding's dtype (stored, or `hold_dtype`'s cast copy); only a CPU executor widens,
    to float32 (`weights._float32`). Any other mismatch is a `module.to(dtype)`, `.half()` or
    `.float()` after placement, which would cast every weight on every forward, unaccounted."""
    if site is None or site.dtype == served:
        return None
    if model.memory.device.type == "cpu" and site.dtype == torch.float32:
        return site.dtype
    raise ResidencyError(
        f"{model.name}: {name} serves {served} and the module was changed to {site.dtype} "
        "after placement: serve a placed component in another dtype with "
        "`cozy_runtime.author.hold_dtype(module, dtype)`, never `module.to(dtype)`, "
        "`.half()` or `.float()`"
    )


def _assign(block: _Block) -> list[tuple[_Site, Tensor]]:
    model = block.model
    if block.data is None:
        raise ResidencyError("weight assignment before a complete fill")
    rows: dict[str, Mapping[str, Tensor]]
    if block.aliases is not None:
        use = block.final_use
        if use is not None and use.needs_fill:
            assert block.final_data is not None
            for key, destinations in block.final_data.items():
                parts = block.data[key]
                if model.finals is not None:
                    model.finals.write(key, parts, destinations)
                else:
                    for destination in destinations.values():
                        destination.copy_(parts["value"])
            use.commit()
            model.memory._keep_final(block, use.tensor)
            block.final_fills += 1
            if (request := _REQUEST.get()) is not None:
                request.final_fills += 1
            if block.final_only():
                # Compute consumes only the completed final bytes. Retire our raw views
                # and make their backing the first victim; an escaped alias still pins it.
                if block.region is not None:
                    block.region.set_preferred(False)
                block.data = {}
                block.native_storage = block.native_use = None
        # Only ordinary owning views remain. The native region can become evictable
        # once those views (including every escaped storage alias) disappear.
        block.final_use = None
        rows = {
            key: {
                **{name: block.data[key][role] for name, role in aliases.items()},
                **(block.final_data or {}).get(key, {}),
            }
            for key, aliases in block.aliases.items()
        }
    else:
        rows = {}
        for key, parts in block.data.items():
            if model.bind is not None:
                rows[key] = model.bind(key, parts)
            elif set(parts) == {"value"}:
                name = model.source.bound.get(key, key) if model.source.bound else key
                rows[key] = {name: parts["value"]}
            else:
                raise ResidencyError(f"{model.name}: encoded {key} needs a declared binder")
    assigned: dict[tuple[int, str], tuple[_Site, Tensor]] = {}
    for values in rows.values():
        for name, value in values.items():
            sites = model.sites.get(name, ())
            if not sites:
                continue
            old = sites[0].table[sites[0].name]
            if (
                block.aliases is not None
                and old is not None
                and old.dtype != value.dtype
                and (block.retained or model.memory.capturing())
            ):
                raise StaleBinding(f"{name}: graph capture needs a matching destination dtype")
            widen = _widened(model, name, old, value.dtype)
            tensor = value if widen is None else value.to(widen)
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
            memory._open(model, key, executing=True)

        model.handles.extend(_paired(sub, opened, lambda: memory._close(model, executing=True)))
    for block in model.blocks.values():
        sub = model.module.get_submodule(block.path)

        def entered(*_: object, b: _Block = block) -> None:
            if (cut := CUTTER.get()) is not None:
                cut.enter(b)
            if (seen := ENTERED.get()) is not None:
                seen.add(b)
            model._claim()
            try:
                memory._enter(b)
                try:
                    memory._prefetch_next(b)
                except BaseException:
                    memory._exit(b)
                    raise
            except BaseException:
                model._release()
                raise

        def exited(b: _Block = block) -> None:
            try:
                memory._exit(b)
            finally:
                model._release()
            if (cut := CUTTER.get()) is not None:
                cut.exit(b)

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
