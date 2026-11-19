"""Executor-local weight residency with ordinary Torch views.

CUDA weights use TensorFS's process-local Rust reservations. A view owns its physical pin
through the Torch storage lifetime, including escaped aliases. Runtime drops idle views while
the native manager keeps eligible backing cached. CPU and MPS use Torch-owned storage.
"""

from __future__ import annotations

import functools
import hashlib
import math
import re
import sys
import threading
import time
import warnings
import weakref
from collections import deque
from collections.abc import Callable, Collection, Hashable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, NotRequired, Protocol, TypedDict, Unpack, cast

import tensorfs
import tensorfs.varena.torch as vt
import torch
from tensorfs.raw import parts as stored_parts
from tensorfs.varena import Exhausted
from tensorfs.varena import _runtime as varena
from torch import Tensor, nn
from torch.utils._pytree import tree_leaves, tree_map
from torch.utils.hooks import RemovableHandle

from cozy_runtime import graphs
from cozy_runtime.author import _phases, _retained
from cozy_runtime.author._errors import CapabilityError, DoesNotFit, Fault, fault, is_capacity
from cozy_runtime.author._model import TRANSPARENT, WARNINGS, WarningRow, recovered
from cozy_runtime.internal import cache_owners, pacing, proctree
from cozy_runtime.internal.accel import (
    ResidencyError,
    StaleBinding,
    allocation,
    host_memory,
    release_cached,
)

if TYPE_CHECKING:
    from tensorfs.varena._facts import (
        DeviceStats,
        EscapeRecord,
        PagesStats,
        PlanFacts,
        Verdict,
        WorkingStats,
    )
    from torch._ops import OpOverload

    from cozy_runtime.internal.execution.phase import WorkingPhase

__all__ = [
    "Binding",
    "BindingLayout",
    "Declared",
    "Hold",
    "Level",
    "Memory",
    "Model",
    "Region",
    "Request",
    "TensorLayout",
    "Weights",
    "blocks_of",
    "paging",
    "split",
    "unpaged",
]

def _torch_bytes(device: torch.device) -> tuple[int, int, int]:
    """Working allocation demand/backing/peak; native weight bytes are counted separately."""
    stats = vt.stats(device.index or 0)
    return int(stats["allocated_bytes"]), int(stats["mapped_bytes"]), int(stats["allocated_peak"])


#: Other overlays' pages a block keeps parked: base and a LoRA alternate with one.
_PARKED = 2
_HOST_STAGING = 64 << 20
#: Host RAM that host-resident weights leave free: the host tier's pages, a CPU's held blocks.
_HOST_HEADROOM = 8 << 30


@functools.cache
def _headroom() -> int:
    """`_HOST_HEADROOM`, or a quarter of a smaller host's RAM: 8 GiB was half a 16 GB laptop's or
    CI runner's, whose free RAM was most often less, so every CPU binding was refused (cl-885)."""
    total = host_memory().total_bytes
    return min(_HOST_HEADROOM, total // 4) if total > 0 else _HOST_HEADROOM


#: Stored float dtypes a CPU executor serves widened to float32 (`weights._float32`).
_NARROW = ("F16", "BF16")
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
#: The shape key of the component forward running now (`execution.catch.passing`), its passes'
#: shape where its blocks mark them (`_entered`): 0 where none runs.
PASS_SHAPE: ContextVar[int] = ContextVar("cozy_pass_shape", default=0)
_U64 = (1 << 64) - 1
#: Escaped bytes a run's warning names.
_NOTED = 256 << 20
_INDEX = re.compile(r"(?<![^.])\d+(?![^.])")
_MEMORIES: weakref.WeakValueDictionary[str, Memory] = weakref.WeakValueDictionary()
_OPENING = threading.Lock()
# A failed graph completion must retain the actual storage owners until executor exit.
_UNPROVEN: list[Binding] = []


class Level(Enum):
    RAM = "ram"
    GPU = "gpu"


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


class Placeholder(Tensor):
    """Metadata between binds; a real read obtains ordinary owning Torch storage."""

    _varena_read: Callable[[], Tensor]

    @staticmethod
    def __new__(
        cls: type[Placeholder], shape: Sequence[int], dtype: torch.dtype, device: torch.device
    ) -> Placeholder:
        return Tensor._make_wrapper_subclass(
            cls, tuple(shape), dtype=dtype, device=device, storage_size=0
        )

    def __repr__(self) -> str:  # type: ignore[override]  # repr() never passes Tensor's tensor_contents
        return f"UnboundWeight({tuple(self.shape)}, {self.dtype}, {self.device})"

    def _varena_read(self) -> Tensor:  # type: ignore[no-redef]  # the default; a made copy's own reads its source
        raise ResidencyError("this weight has no declared source")

    @classmethod
    def __torch_dispatch__(  # type: ignore[explicit-any, override]  # any op, any operands
        cls,
        func: OpOverload[..., Any],
        types: Sequence[type],
        args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
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


def _readable(value: object) -> object:
    return value._varena_read() if isinstance(value, Placeholder) else value


def _host(nbytes: int, *, pinned: bool) -> Tensor:
    """`nbytes` of host memory, page-locked when `pinned` and the host can lock it."""
    if pinned:
        try:
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        except RuntimeError:  # no lockable host memory now: pageable copies are slower, not wrong
            pass
    return torch.empty(nbytes, dtype=torch.uint8)


def _bound(model: Model, site: _Site) -> Tensor:
    """`site`'s value while its block is entered."""
    held = site.table[site.name]
    if isinstance(held, Placeholder):
        raise ResidencyError(
            f"{model.name}: {site.name} has no value: no stored tensor binds it. A buffer no "
            "checkpoint stores keeps its constructed value only when construction computed one "
            "(from config or a constant), not from a weightless parameter"
        )
    return cast(Tensor, held)  # a site is a slot that held a tensor


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
    # The innermost units: an outer one (a layer holding its experts) gives way to them.
    outer = {prefix for path in found for prefix in _prefixes(path)}
    found = [path for path in found if path not in outer]
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


#: Left free under a software cap: s8-h3 on an RTX 5090 peaked 57 MiB over 8 GiB with none.
#: A paging unit larger than this is a warning: a quarter of an 8 GB card.
_LARGEST_UNIT = 2 << 30
#: A paging unit holding more than this splits (`split`), the same on every card.
_SPLIT_UNIT = 1 << 30
#: An embedding table outside every block and this large pages as its own unit (`split`).
_TABLE_UNIT = 64 << 20


def _prefixes(name: str) -> Iterator[str]:
    """`name`'s proper dotted prefixes, longest first: `a.b.c` gives `a.b`, then `a`."""
    while "." in name:
        name = name.rpartition(".")[0]
        yield name


def _unit(name: str, blocks: Collection[str]) -> str:
    """The block holding tensor `name` (its longest dotted prefix that is a block), "" for common.
    `blocks` is a set or a mapping: a lookup per prefix, never a scan of the blocks."""
    return next((prefix for prefix in _prefixes(name) if prefix in blocks), "")


def split(module: nn.Module, paths: Sequence[str], sizes: Mapping[str, int]) -> list[str]:
    """`paths` with each unit over `_SPLIT_UNIT` split where child modules allow: a block gives way to
    its weighted children (its own tensors join the common weights), and the common weights give their
    largest children that hold no block, one at a time, until under. A new block over the bound splits
    the same way, down to single layers; one tensor over it stays whole. `sizes` are the stored bytes
    by module-relative tensor name. Hunyuan-Image-3 kept 10 GB of attention and dense layers in common
    beside its expert blocks. Linear in tensors and modules: each module's bytes are summed once,
    and a unit's bytes move with its blocks (Qwen3.6's 10,240 expert blocks were a scan each)."""
    modules = dict(module.named_modules())
    blocks = set(paths)
    #: each module path's stored bytes below it; the paths that hold a block (it and its parents)
    below: dict[str, int] = {}
    for name, n in sizes.items():
        for prefix in _prefixes(name):
            below[prefix] = below.get(prefix, 0) + n
    holding = {prefix for block in blocks for prefix in (block, *_prefixes(block))}
    units: dict[str, int] = {}
    for name, n in sizes.items():
        unit = _unit(name, blocks)
        units[unit] = units.get(unit, 0) + n

    def free(path: str) -> list[tuple[int, str]]:
        """`path`'s weighted child modules that hold no block, through children that hold one."""
        found, stack = [], [path]
        while stack:
            parent = stack.pop()
            for name, _ in modules[parent].named_children():
                child = f"{parent}.{name}" if parent else name
                if child in holding:
                    if child not in blocks:
                        stack.append(child)
                elif size := below.get(child, 0):
                    found.append((size, child))
        return sorted(found, reverse=True)

    def block(child: str, size: int, owner: str) -> None:
        """`child`, which holds no block and lies in unit `owner`, becomes a block of its own."""
        blocks.add(child)
        holding.update((child, *_prefixes(child)))
        units[owner] -= size
        units[child] = size

    queue = [unit for unit, n in units.items() if n > _SPLIT_UNIT]
    while queue:
        unit = queue.pop()
        if units.get(unit, 0) <= _SPLIT_UNIT:
            continue
        if unit:
            children = free(unit)
            if not children:
                continue
            # Its own tensors join the unit around it; its children's become theirs.
            blocks.discard(unit)
            around = _unit(unit + ".", blocks)
            units[around] = units.get(around, 0) + units.pop(unit)
            for size, child in children:
                block(child, size, around)
            queue += ["", *(child for size, child in children if size > _SPLIT_UNIT)]
            continue
        for size, child in free(""):
            if units.get("", 0) <= _SPLIT_UNIT:
                break
            block(child, size, "")
            if size > _SPLIT_UNIT:
                queue.append(child)
    # A table pages as its own unit, so one that does not fit the room gathers its rows on the
    # host (`Memory._gather`) and the weights beside it still fit.
    blocks.update(
        path
        for path, sub in modules.items()
        if path
        and isinstance(sub, nn.Embedding)
        and _unit(f"{path}.weight", blocks) == ""
        and sizes.get(f"{path}.weight", 0) >= _TABLE_UNIT
    )
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
    #: the module's `_parameters` or `_buffers`: a bind puts plain tensors in either
    table: dict[str, Tensor | None]
    name: str
    parameter: bool


@dataclass(frozen=True, slots=True)
class _Part:
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    role: str
    strides: tuple[int, ...] = ()


@dataclass(eq=False, slots=True, weakref_slot=True)
class _Block:
    model: Model
    path: str
    index: int
    parts: dict[str, dict[str, _Part]] = field(default_factory=dict)
    data: dict[str, dict[str, Tensor]] | None = None
    assigned: list[tuple[_Site, Tensor]] | None = None
    saved: list[tuple[_Site, Tensor | None]] = field(default_factory=list)
    count: int = 0
    retained: int = 0
    loading: bool = False
    last: int = 0
    ready: torch.cuda.Event | None = None
    region: varena.Region | None = None
    native_storage: int | None = None
    #: the Varena Uses behind the raw and final views; `_used` declares their consumer streams
    native_use: varena.Use | None = None
    final_native_use: varena.Use | None = None
    aliases: dict[str, dict[str, str]] | None = None
    raw_views: dict[tuple[int, str], tuple[int, _Part]] = field(default_factory=dict)
    finals: dict[str, dict[str, _Part]] = field(default_factory=dict)
    final_region: varena.Region | None = None
    final_data: dict[str, dict[str, Tensor]] | None = None
    final_storage: int | None = None
    final_fills: int = 0
    final_hits: int = 0
    #: the fill overlay (`Filling.overlay`) its page was written with
    overlay: Hashable = None
    #: native pages of other overlays, parked for reuse (`_switch_overlay`): (raw, final) regions
    parked: dict[Hashable, tuple[varena.Region | None, varena.Region | None]] = field(
        default_factory=dict
    )
    #: an `nn.Embedding`'s table: one that does not fit the room gathers its rows on the host
    table: bool = False
    #: its host copy while it gathers, by overlay: (overlay, raw views, final views)
    gathered: tuple[Hashable, dict[str, dict[str, Tensor]], dict[str, dict[str, Tensor]]] | None = (
        None
    )
    #: where its forward's allocations are born in the escaped-memory tally (`vt.set_site`)
    escape_site: int = 0
    #: its raw Region's Use from a binding's fill of all its blocks together (`Memory._fill`),
    #: which its next `_ensure` binds instead of filling it alone
    prefilled: varena.Use | None = None
    #: the birth site its entry replaced, back at its exit: what runs after it is not its
    born_before: int | None = None
    #: a block that is one linear layer: refused room, it stays on the host and its forward
    #: runs in windows of output rows copied in (`Memory._streamed`)
    leaf: nn.Module | None = None

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
        """What a fill of it holds: its native backing, else its stored bytes, a CPU's 16-bit
        floats widened to float32 (`_widened`)."""
        memory = self.model.memory
        if memory.native is None:
            wide = 2 if memory.device.type == "cpu" else 1
            return sum(
                part.nbytes * (wide if part.dtype in _NARROW else 1)
                for parts in self.parts.values()
                for part in parts.values()
            )
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

    def raw_needed(self) -> bool:
        return self.aliases is None or any(self.aliases.values()) or not self.finals

    @property
    def least_bytes(self) -> int:
        """The fewest device bytes it holds in use: its stored rows where they're read, and its
        final ones."""
        return (self.storage_bytes if self.raw_needed() else 0) + self.final_bytes

    def cached(self) -> bool:
        if self.final_bytes and (self.final_region is None or not self.final_region.resident):
            return False
        return not self.raw_needed() or (self.region is not None and bool(self.region.resident))

    def final_only(self) -> bool:
        return bool(self.finals) and self.aliases is not None and not any(self.aliases.values())

    def tensors(self) -> Iterator[Tensor]:
        if self.data is not None:
            yield from (tensor for parts in self.data.values() for tensor in parts.values())
        if self.assigned is not None:
            yield from (tensor for _, tensor in self.assigned)

    def computed(self) -> Iterator[Tensor]:
        """Its CUDA tensors over neither Region: what a binder decoded, each its own allocation."""
        for tensor in self.tensors():
            if tensor.is_cuda and tensor.untyped_storage().data_ptr() not in (
                self.native_storage,
                self.final_storage,
            ):
                yield tensor

    def regions(self) -> list[varena.Region]:
        """The Regions a use of it reads."""
        raw = self.region if self.storage_bytes and not self.final_only() else None
        final = self.final_region if self.final_bytes else None
        return [region for region in (raw, final) if region is not None]


class Model:
    def __init__(self, memory: Memory, module: nn.Module, declared: Declared) -> None:
        self.memory, self.module, self.source = memory, module, declared.weights
        self.component = declared.weights.component
        #: The component by its checkpoint's content, however its package was installed.
        self.content = f"{self.source.ck.id}/{self.component}"
        self.name = f"{self.source.release}/{self.content}"
        #: the construction whose load built it (`weights.Backend`): what a bake may read
        self.construction = ""
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
        self._block_sites: dict[str, list[_Site]] | None = None
        self.handles: list[RemovableHandle] = []
        self.holds: list[Hold] = []
        self.blocks: dict[str, _Block] = {}
        self.common: _Block | None = None
        self.closed = False
        self.working_phase_keys: set[str] = set()
        #: its loop's head site in the escaped-memory tally, and one block's weights with its common
        self.escape_site = 0
        self.escape_floor: int | None = None
        #: loops streaming it (`gathering`): an embedding table of it that left the device gathers
        #: its rows on the host instead of filling whole again
        self.gathers = 0

    def sites_in(self, groups: Iterable[_Block]) -> list[_Site]:
        """The weight sites `groups` hold, mapped once: a LoRA bake binds one source block per
        linear, and walking every site for each took 6 s of Qwen's first step (cr-245)."""
        if self._block_sites is None:
            paths = self.blocks
            held: dict[int, tuple[str, list[_Site]]] = {}
            for name, sites in self.sites.items():
                path = _block_of(name, paths)
                was = held.get(id(sites))
                # A tied tensor used by distinct blocks stays common, as `_register` places it.
                held[id(sites)] = (path if was is None or was[0] == path else "", sites)
            self._block_sites = {}
            for path, sites in held.values():
                self._block_sites.setdefault(path, []).extend(sites)
        return [site for block in groups for site in self._block_sites.get(block.path, ())]

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
            if self.closed:
                raise CapabilityError(
                    f"{self.name} is no longer loaded (its construction was released): a call "
                    "reads only the models its own construction loaded",
                    code="model_not_loaded",
                )
            if self.retiring:
                raise ResidencyError("the model is retiring its cached graphs")
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

    def keep(self, since_ns: int, reserve: int) -> tuple[Binding, int, int] | None:
        """For a loop of its forwards that can't hold it whole (an LM decoding streamed): of its
        blocks used since `since_ns`, the most bytes that fit beside its live working memory,
        `reserve` bytes and room for the largest of the rest to stream through, largest first. A
        binding over them, which the caller prepares and holds while the loop runs, so each
        forward reads only the others again; with the bytes kept and those still read a forward.
        None when none fits. An embedding table is no block a forward reads whole: under the loop
        it gathers its rows on the host (`gathering`)."""
        if self.memory.native is None:
            return None
        # The common group (embeddings, the head, final norms) serves every forward, entered
        # around it rather than block by block.
        used = sorted(
            (b for b in self.groups() if (b is self.common or b.last >= since_ns) and not b.table),
            key=lambda b: b.least_bytes,
            reverse=True,
        )
        room = vt.room(self.memory.device.index or 0)
        budget = room["capacity"] - room["working_live"] - reserve
        kept: list[_Block] = []
        held = streamed = 0
        for at, block in enumerate(used):
            after = used[at + 1].least_bytes if at + 1 < len(used) else 0
            if held + block.least_bytes + max(streamed, after) <= budget:
                kept.append(block)
                held += block.least_bytes
            else:
                streamed = max(streamed, block.least_bytes)
        if not kept:
            return None
        self._claim()
        try:
            self.memory._declare(self)
            return Binding(self, kept), held, sum(b.least_bytes for b in used) - held
        finally:
            self._release()

    @contextmanager
    def gathering(self) -> Iterator[None]:
        """While a loop streams it (`keep`): its embedding tables, which `keep` leaves no room,
        serve from their device page while varena keeps it and otherwise gather their rows on the
        host (`Memory._gather`), never filling whole again. Filled whole, a table took the room
        the loop's working memory had, which took it back the next token: MiniMax-H3-Image's
        expander on a 4090 refilled its 2.37 GiB table 685 times in 1,573 tokens (cr-332)."""
        with self.memory.lock:
            self.gathers += 1
        try:
            yield
        finally:
            with self.memory.lock:
                self.gathers -= 1

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
            if free_only and self.memory.device.type == "cpu":
                groups = self.memory._fitting(self, groups)
                if not groups:
                    return None
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
            if free_only and missing > self.memory._free():
                release_cached(torch, self.memory.device.type)
                if missing > self.memory._free():
                    return None
            entered: list[_Block] = []
            try:
                for block in groups:
                    self.memory._enter(block, reclaim=not free_only)
                    entered.append(block)
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
            self.memory._retire_phase_plan("model_closed", model=self)
            self.memory.forget_working_phases(self)
            self.closed = True
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            for block in self.groups():
                self.memory._discard(block, forget=True)
            self.memory._forget_pages(self)
            if self.name in self.memory._coming:
                # Its pages were read ahead for another request: their Sources hold its checkpoint.
                self.memory._staging, self.memory._coming = [], set()
            # A new map: a request beside it may be iterating the old one (a released construction).
            models = dict(self.memory.models)
            models.pop(self.module, None)
            self.memory.models = models


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


def _state(region: varena.Region | None) -> str:
    """A block region's residency, as a refusal names it."""
    return "none" if region is None else "resident" if region.resident else "cold"


class Binding:
    """What a cached graph reads, closed after its readers retire: Regions, whose addresses outlive
    their pages, so a graph holds no weight bytes between uses, and a binder's computed tensors.

    An undeclared fill may prepare once for an ephemeral direct read. It never reports a
    valid/resident graph binding and cannot be replayed or prepared a second time.
    """

    def __init__(self, model: Model, groups: list[_Block]) -> None:
        self.model, self.groups = model, groups
        self.fill = model.wanted_identity()
        self.generation = model.generation
        self.targets = [
            (site, tuple(value.shape), value.dtype)
            for site in model.sites_in(groups)
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
        #: from its first preparation on, what its graphs read
        self.regions: list[varena.Region] | None = None
        self.computed: list[Tensor] = []
        self.lock = threading.RLock()
        self.last: torch.cuda.Event | None = None

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
        if not all(
            (value := site.table[site.name]) is not None
            and tuple(value.shape) == shape
            and value.dtype == dtype
            for site, shape, dtype in self.targets
        ):
            return "target shape or dtype changed"
        if self.regions not in (None, [r for b in self.groups for r in b.regions()]):
            return "a block's Region changed"
        return ""

    def resident(self) -> bool:
        return not self.unresident()

    def immutable_buffer_ids(self) -> frozenset[int]:
        """Its model's declared checkpoint-backed buffers, in every block (an idle one's placeholder
        too): inputs a recording checks by identity and version and never copies (cr-264). Other
        buffers stay module state, copied and restored."""
        model, names = self.model, set[str]()
        for block in model.groups():
            for key in block.parts:
                if model.finals is not None:
                    names.update(model.finals.layout(key) or ())
                elif set(block.parts[key]) == {"value"}:
                    names.add(model.source.bound.get(key, key) if model.source.bound else key)
        return frozenset(
            id(site.table[site.name])
            for name in names
            for site in model.sites.get(name, ())
            if not site.parameter and site.table[site.name] is not None
        )

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

    @contextmanager
    def prepare(
        self, stream: torch.cuda.Stream | None = None, *, replay: bool = False
    ) -> Iterator[dict[varena.Region, varena.Use | None]]:
        """Its weights for one use, and no longer. A capture or an eager use binds them to the
        modules' slots. A `replay` reads its Regions at their addresses and binds nothing: it
        yields each Region's Use, None for one that left (`graphs.replaying` fills those)."""
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
                if replay and self.regions is None:
                    raise StaleBinding("a graph's weights were never prepared")
                request = _REQUEST.get()
                if request is not None and request.idle and not self.resident():
                    raise StaleBinding("idle graph preparation cannot page weights in")
                if not replay and memory.device.type == "cpu":
                    # Bound whole, its blocks cannot stream: past the host's room it refuses as
                    # an allocation would, and a decode streams them per forward instead.
                    missing = sum(b.allocation_bytes for b in self.groups if b.data is None)
                    if missing and missing > memory._free() - _headroom():
                        raise torch.OutOfMemoryError(
                            f"{self.model.component}: {missing / (1 << 30):.1f} GiB of float32 "
                            "weights to bind, more than the host can give"
                        )
                if not replay and any(b.data is None and not b.cached() for b in self.groups):
                    memory._fits(self.groups)
                self.used = True
                with torch.cuda.stream(stream) if memory.cuda else nullcontext():
                    if self.last is not None and stream is not None:
                        stream.wait_event(self.last)
                    # Claimed whole before any of it fills: no fill can change a later block.
                    with memory.lock:
                        for block in self.groups:
                            block.retained += 1
                    entered: list[_Block] = []
                    uses: dict[varena.Region, varena.Use | None] = {}
                    try:
                        if replay:
                            assert self.regions is not None
                            uses.update((region, region.pin()) for region in self.regions)
                        else:
                            memory._fill(self.groups)
                            for block in self.groups:
                                memory._enter(block)
                                entered.append(block)
                            if self.regions is None:
                                self.regions = [r for b in self.groups for r in b.regions()]
                                self.computed = [t for b in self.groups for t in b.computed()]
                        yield uses
                    finally:
                        try:
                            if memory.cuda and (entered or uses):
                                self._finish(stream)
                        finally:
                            uses.clear()
                            for block in reversed(entered):
                                memory._exit(block)
                            for block in self.groups:
                                block.prefilled = None  # a failure's: its pin goes
                            with memory.lock:
                                for block in self.groups:
                                    block.retained -= 1
                                    memory._idle(block)
            finally:
                self.model._release()

    def _unproven(self, cause: BaseException) -> None:
        if not self.completion_lost:
            self.completion_lost = True
            _UNPROVEN.append(self)
        error = torch.AcceleratorError(f"graph completion is unproven; retire executor: {cause}")
        error.error_code = 999  # type: ignore[attr-defined]  # torch's CUDA errors carry it; its stub doesn't
        raise error from cause

    def _finish(self, stream: torch.cuda.Stream | None) -> None:
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
            self.regions, self.computed = None, []


class Region:
    """An explicit owner's disposable cache; it never intercepts or remaps Torch storage."""

    def __init__(self, memory: Memory, owner: str, release: Callable[[], None] | None,
                 shrink: Callable[[], bool] | None = None) -> None:
        self.memory, self.owner, self.release = memory, owner, release
        self.shrink = shrink
        self.lock = threading.RLock()
        self.pins = 0
        self.fresh = False
        self.closed = False
        self.reclaiming = False
        self.last = time.monotonic_ns()
        # Only typed owners with an explicit nonblocking shrink path enter background reclaim.
        self._cache_owner = (cache_owners.register(memory.device, self, "discard")
                             if shrink is not None else None)

    @contextmanager
    def pin(self) -> Iterator[bool]:
        with self.lock:
            if self.closed or self.reclaiming:
                raise ResidencyError(f"cache {self.owner!r} is closed or being reclaimed")
            if self._cache_owner is not None:
                self._cache_owner.check()
            self.pins += 1
            fresh, self.fresh = self.fresh, False
        try:
            yield fresh
        finally:
            with self.lock:
                self.pins -= 1
                self.last = time.monotonic_ns()
                if self._cache_owner is not None:
                    self._cache_owner.touch()

    def discard(self) -> bool:
        if not self.lock.acquire(blocking=False):
            return False
        try:
            if self.closed or self.pins or self.reclaiming or self.release is None:
                return False
            self.reclaiming = True
            # No Memory metadata lock is held. The owner releases its ordinary tensors and
            # joins any exceptional consumers or cached graph outputs before dropping them.
            if self.shrink is None:
                self.release()
            elif not self.shrink():
                return False
            self.fresh = True
            if self._cache_owner is not None:
                self._cache_owner.suspend()
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
            if self._cache_owner is not None:
                self._cache_owner.stop()
            release, self.release = self.release, None
            self.shrink = None
            if release is not None and not self.reclaiming:
                release()
        if self._cache_owner is not None:
            self._cache_owner.close()  # drains only after releasing the Region lock


def _pages(index: int, shared: bool) -> varena.Pages:
    """This GPU's host tier. TensorFS sizes it and shares the host with every executor's tiers
    through the host's ledger page; on a shared machine (a person's computer) it pins nothing
    while the host stalls."""
    return varena.Pages(index, shared=shared)


class Memory:
    """One executor's weight residency, beside Torch's activation allocator."""

    def __init__(
        self,
        device: torch.device | str | int,
        budget_bytes: int | None = None,
        shared: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.cuda: bool = self.device.type == "cuda"
        self.budget_bytes = budget_bytes
        self.lock = threading.RLock()
        self.models: dict[nn.Module, Model] = {}
        self._working_phase_owners: dict[str, set[Model | Request]] = {}
        self.regions: weakref.WeakSet[Region] = weakref.WeakSet()
        #: what holds weights only for speed (a streamed decode's kept blocks, `lm._Keeper`): asked
        #: to let them go when a block it doesn't hold is refused room, before the refusal stands
        self.sheds: list[Callable[[int], bool]] = []
        self.requests: set[Request] = set()
        self.last: RequestStats | None = None
        self.native: varena.Device | None = None
        #: This GPU's host tier: TensorFS keeps weight pages in pinned host RAM.
        self.pages: varena.Pages | None = None
        #: each raw Region's Source, and those the tier is reading ahead (`_read_ahead`)
        self._sources: weakref.WeakKeyDictionary[varena.Region, varena.Source] = (
            weakref.WeakKeyDictionary()
        )
        self._staging: list[varena.Source] = []
        self._coming: set[str] = set()
        self.granularity = 512
        self._phase_plan: varena.Plan | None = None
        self._phase_plan_signature: tuple[tuple[list[str] | _PhaseSignature, ...], ...] | None = (
            None
        )
        self._phase_plan_keys: dict[frozenset[str], str] = {}
        self._phase_plan_models: set[Model] = set()
        #: what the active plan was built from (`_phase_stamp`): the same stamp is the same plan
        self._phase_plan_stamp: tuple[object, ...] | None = None
        self._phase_plan_status: tuple[object, ...] | None = None
        self._phase_names: frozenset[str] | None = None
        self._phase_request: str | None = None
        self._forward_depth = 0
        #: models whose own forward (or a hooked child's) is running, by how many deep
        self._executing: dict[Model, int] = {}
        #: the birth site last set on this device (`_born`)
        self._site = 0
        #: (model, pass shape) whose first block this process measured (`_Probe`): its working peak above
        #: its entry (`block_need`); and the one measuring now
        self._probed: dict[tuple[str, int], int] = {}
        self._probe: _Probe | None = None
        if self.cuda:
            try:
                if budget_bytes is not None:
                    vt.set_limit(self.device.index or 0, budget_bytes)
                self.native = varena.Device(self.device.index or 0)
                self.granularity = int(self.native.granularity)
                self.pages = _pages(self.device.index or 0, shared)
            except (ImportError, AttributeError, RuntimeError) as exc:
                if fault(exc) is not Fault.OTHER:
                    raise
                raise ResidencyError(f"native CUDA weight residency is unavailable: {exc}") from exc

    @classmethod
    def open(
        cls,
        gpu: torch.device | str | int = 0,
        *,
        budget_bytes: int | None = None,
        shared: bool = False,
    ) -> Memory:
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
            memory = cls(device, budget_bytes, shared)
            _MEMORIES[str(device)] = memory
            return memory

    def limit(self, budget_bytes: int | None) -> None:
        """A new card limit (a display GPU's follows its display): varena reclaims to it now."""
        if self.cuda and budget_bytes != self.budget_bytes:
            vt.set_limit(self.device.index or 0, budget_bytes)
            self.budget_bytes = budget_bytes

    def stream(self) -> torch.cuda.Stream | None:
        return torch.cuda.current_stream(self.device) if self.cuda else None

    def _cuda_stream(self) -> torch.cuda.Stream:
        """`stream()` where only a CUDA device gets: native Regions exist only there."""
        stream = self.stream()
        if stream is None:
            raise ResidencyError("a CUDA stream was asked of a device that is not CUDA")
        return stream

    def _device_native(self) -> varena.Device:
        """This GPU's varena Device, opened with this Memory on CUDA and only asked for there."""
        if self.native is None:
            raise ResidencyError("native CUDA weight residency is not open on this device")
        return self.native

    def _tier(self) -> varena.Pages:
        """This GPU's host tier, opened with its Device."""
        if self.pages is None:
            raise ResidencyError("native CUDA weight residency is not open on this device")
        return self.pages

    def capturing(self) -> bool:
        return bool(
            self.cuda and torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()
        )

    def _reading(self) -> tuple[int, int, int, int]:
        """Native working demand/backing/peak and weight pages, each read once."""
        if self.cuda and torch.cuda.is_initialized():
            resident = int(self.native.stats()["resident_bytes"]) if self.native is not None else 0
            return (*_torch_bytes(self.device), resident)
        if self.device.type == "mps":
            used = int(torch.mps.current_allocated_memory())
            return used, int(torch.mps.driver_allocated_memory()), used, 0
        return 0, 0, 0, 0

    def _figures(self) -> tuple[int, int]:
        allocated, reserved, _, resident = self._reading()
        return allocated + resident, reserved + resident

    def remember_working_phase(self, key: str, models: tuple[Model | Request, ...]) -> None:
        """A semantic observation lives with its model owners, or its componentless request."""
        with self.lock:
            owners = self._working_phase_owners.setdefault(key, set())
            for model in models:
                owners.add(model)
                model.working_phase_keys.add(key)

    def forget_working_phases(self, model: Model | Request) -> None:
        """Owner close runs after its execution; native retirement is metadata-only."""
        with self.lock:
            # If another scope is active, a refused retirement retains the ownership request.
            for key in model.working_phase_keys:
                if self._working_phase_owners[key] == {model}:
                    with suppress(RuntimeError):  # still active: its native record outlives it
                        vt.phase_forget(self.device.index or 0, key)
            for key in tuple(model.working_phase_keys):
                owners = self._working_phase_owners[key]
                if owners == {model}:
                    del self._working_phase_owners[key]
                else:
                    owners.remove(model)
                model.working_phase_keys.remove(key)

    def _fits(self, groups: list[_Block]) -> None:
        """`groups` held at once, refused by varena before any of them fills when the card or its
        cap could never hold them (`Device.fits`): filled partway, they would only evict everything
        else and then wait for room that can't come (cr-286: an 11.1 GiB decode set on an 8 GiB
        cap waited 300 s, run 47309). A block with no Region yet counts its bytes."""
        if self.native is None:
            return
        regions: list[varena.Region] = []
        more = 0
        for block in groups:
            for region, size in (
                (block.region, block.storage_bytes if block.raw_needed() else 0),
                (block.final_region, block.final_bytes),
            ):
                if size and region is None:
                    more += size
                elif size and region is not None:
                    regions.append(region)
        try:
            self.native.fits(regions, more)
        except Exhausted as refused:
            raise Exhausted(
                f"{groups[0].model.component}: {_gib(refused.requested_bytes)} of weights to hold "
                f"at once, more than GPU {self.device.index or 0} could ever give it beside its "
                "live working memory and other weights in use",
                refused.requested_bytes,
            ) from refused

    def _free(self) -> int:
        """Free bytes where no native owner answers (MPS, CPU). A CPU's is what the host can still
        give this process: on Linux its tightest cgroup's, page cache counted as room (a pod's
        `memory.max` is a fraction of the host's MemAvailable)."""
        if self.device.type != "mps":
            if sys.platform == "linux":
                return max(proctree.available_host_bytes(), 0)
            else:
                return host_memory().free_bytes
        figures = allocation(torch, "mps")
        free = int(figures["driver_free_bytes"])
        if self.budget_bytes is not None:
            free = min(free, max(self.budget_bytes - int(figures["reserved_bytes"]), 0))
        return free

    def _fitting(self, model: Model, groups: list[_Block]) -> list[_Block]:
        """A CPU step's hold, decided once: the longest prefix of `groups` whose bytes as served
        fit what the host can still give less its headroom (`_headroom`), once other models' holds
        outside the step have gone. The rest stream, read and widened per forward (`_idle`)."""
        request = _REQUEST.get()
        step = request.phase(model.name) if request is not None else frozenset({model.name})
        for other in list(self.models.values()):
            if other.name not in step and not other.users:
                for hold in [h for h in other.holds if h.reclaimable]:
                    hold.shed()
        room = self._free() - _headroom()
        held: list[_Block] = []
        for block in groups:
            room -= 0 if block.data is not None else block.allocation_bytes
            if room < 0:
                break
            held.append(block)
        return held

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
        blocks = frozenset(paths)
        #: each site group's names, once (a tied tensor's are several)
        named: dict[int, list[str]] = {}
        for name, sites in model.sites.items():
            named.setdefault(id(sites), []).append(name)
        for key in tensors:
            if w.keys is not None and key not in w.keys:
                continue
            bound = w.bound.get(key, key) if w.bound else key
            path = _block_of(bound, blocks)
            # A tied tensor used by distinct modules stays common, so either forward sees
            # one live parameter. The declared encoded bind routes must name their targets.
            group = model.sites.get(bound)
            if group is not None and any(
                _block_of(name, blocks) != path for name in named[id(group)]
            ):
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
                block.table = isinstance(module.get_submodule(path), nn.Embedding)
                model.blocks[path] = block
            else:
                model.common = block
        if self.native is not None:
            for block in model.groups():
                _layout(block)
            model.layout_identity = hashlib.sha256(
                repr([(b.path, b.aliases, b.finals) for b in model.groups()]).encode()
            ).hexdigest()
            model.identity = model.wanted_identity()
        why = unpaged(module, rule, sizes)
        if why is not None:
            warnings.warn(f"{w.component}: {why}", RuntimeWarning, stacklevel=2)
        _placeholders(model)
        _hooks(model)
        with self.lock:  # against a close replacing the map (`Model.close`)
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
                    old_model = (
                        model.finals,
                        model.binding_identity,
                        model.layout_identity,
                        model.identity,
                        model.generation,
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
                        ) = old_model
                        raise
                    finally:
                        # Next use refreshes native phase intent after representation changes.
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
            if any(b.retained for b in model.groups()) and model.retire_graphs is not None:
                # Programs recorded against the old fill (a kept loop baked, now per_step) still
                # bind its blocks: they go, their pools with them, and record again under the new
                # fill. Never with the metadata lock held: closing joins CUDA.
                model.retire_graphs()
            if any(b.retained for b in model.groups()):
                raise StaleBinding("retire old graph bindings before changing a weight fill")
            for hold in list(model.holds):
                hold.shed()
            if any(b.count for b in model.groups()):
                raise ResidencyError("a weight fill cannot change during an active use")
            for block in model.groups():
                self._discard(block, forget=True)
            self._forget_pages(model)
            model.identity = identity
            model.generation += 1

    def _forget_pages(self, model: Model) -> None:
        """`model`'s kept pages go: its fill changed, or it closed."""
        if self.pages is not None:
            self.pages.forget(model.name)

    def host_pending(self) -> bool:
        """Whether pages wait for an idle executor to pin them (`pin_idle`)."""
        return self.pages is not None and self.pages.pending()

    def pin_idle(self, waiting: Callable[[], bool]) -> None:
        """TensorFS pins the pages that left the card while the executor is idle, until a command
        arrives (`waiting`). A failed read only leaves a page's refills on the store."""
        self._tier().pin_idle(waiting)
        stats = self._tier().stats()
        rows: list[WarningRow] = []
        if stats["pin_failed"]:
            rows.append(recovered("a host page did not pin: its refills stay on the store"))
        if stats["refused"] is not None:
            rows.append(
                recovered(
                    "weights stream from the store: pinning host memory failed",
                    reason=stats["refused"][:300],
                )
            )
        for row in rows:
            if row not in WARNINGS:
                WARNINGS.append(row)

    def _switch_overlay(self, block: _Block, overlay: Hashable) -> None:
        """`block`'s page was filled under another overlay than this use wants (a base page for
        a LoRA forward, or the reverse). As for a changed fill, its model's holds go first. A
        native page is parked under its overlay, and the wanted overlay's parked page serves
        again: alternating overlays refill nothing while both stay resident.
        Off native backing the page is dropped and the next fill writes the wanted overlay."""
        for hold in list(block.model.holds):
            hold.shed()
        if block.count or block.retained or block.loading or self.capturing():
            raise StaleBinding("a weight block in use cannot change its fill overlay")
        self._retire_phase_plan("overlay_changed", model=block.model)
        if self.native is None:
            self._discard(block, forget=True)
            if block.paged():
                raise StaleBinding("a weight block's page under another overlay was not dropped")
            return
        self._discard(block)
        current = (block.region, block.final_region)
        if current != (None, None):
            block.parked[block.overlay] = current
        while len(block.parked) > _PARKED:
            block.parked.pop(next(iter(block.parked)))
        block.region, block.final_region = block.parked.pop(overlay, (None, None))
        block.overlay = overlay

    def _rearm(self, model: Model) -> None:
        """A forward armed another overlay, which H3 does inside it: all its blocks move now and the
        phase is declared again, so the core plans from the Regions this pass will use."""
        assert model.fill is not None
        for block in model.groups():
            overlay = model.fill.overlay(block.path, block.parts)
            if overlay == block.overlay or block.count or block.retained or block.loading:
                continue
            if block.paged() or block.parked:
                self._switch_overlay(block, overlay)
            else:
                block.overlay = overlay
        self._activate_phase(model)
        request = _REQUEST.get()
        if request is not None:
            with self.lock:
                self._read_ahead(request.phase(model.name), current=True)

    def _idle(self, block: _Block) -> None:
        """An idle block's views go: a native page stays the core's cache, and a CPU's bytes are
        copies of the store's, so there only a step's hold keeps them (`_fitting`)."""
        if (self.native is not None or self.device.type == "cpu") and not (
            block.count or block.retained or block.loading
        ):
            # The DLPack storage owner holds the physical pin. A surviving external alias
            # retains it after these references disappear; native eviction waits on the
            # region's last-consumer events (`_used`). record_stream is not that ownership fence.
            block.data = block.assigned = block.ready = None
            block.native_storage = block.native_use = None
            self._drop_final(block)

    def _drop_final(self, block: _Block, *, forget: bool = False) -> None:
        block.final_data = block.final_storage = block.final_native_use = None
        if forget:
            self._retire_phase_plan("final_representation_retired", model=block.model)
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
            present = block.data is not None
            block.data = block.assigned = block.ready = None
            block.native_storage = block.native_use = None
            self._drop_final(block, forget=forget)
            if forget:
                self._retire_phase_plan("representation_retired", model=block.model)
                # Never overwrite an old fill beneath an escaped alias. Its pin keeps
                # the old reservation alive independently of this model's new identity.
                block.region = None
                block.parked.clear()
                block.gathered = None
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

    def reclaim(self, need: int = 0) -> int:
        """Non-CUDA owner recovery; native CUDA allocation owns its bounded recovery."""
        if self.cuda:
            return 0
        before_allocated, before_reserved = self._figures()
        available = self._free() if need else 0
        if need and available < need:
            release_cached(torch, self.device.type)
            available = self._free()
        dropped = 0

        def enough(released: int = 0, *, flush: bool = False) -> bool:
            nonlocal available, dropped
            if not need:
                return False
            dropped += released
            if flush or dropped >= max(need - available, 1):
                available = self._free()
                if available < need:
                    release_cached(torch, self.device.type)
                    available = self._free()
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
        with self.lock:
            blocks = sorted((b for m in self.models.values() for b in m.groups()), key=lambda b: b.last)
        for block in blocks:
            if enough():
                break
            enough(self._discard(block))
        if not enough(flush=True):
            for model in list(self.models.values()):
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
        if not enough(flush=True):
            for region in sorted(list(self.regions), key=lambda r: r.last):
                if enough():
                    break
                if region.discard():
                    enough(flush=True)
        if not need:
            release_cached(torch, self.device.type)
        allocated, reserved = self._figures()
        return max(before_allocated - allocated, before_reserved - reserved, 0)

    def release_host(self) -> None:
        """A controller's `ram` level: this executor's host tier goes too. Its pages are pinned, so
        no other process can reclaim them, and the call the machine offloads this executor for
        needs host RAM as well: beside an idle H3 executor's 42 GB of pages, SeedVR2's 2160p decode
        on a 4090 pod was killed by the kernel (gate 40991023097). Its pages refill from the store
        on this executor's next call. Where no native core runs, its own owners go as well."""
        if self.capturing():
            raise ResidencyError("idle offload cannot run during graph capture")
        if not self.cuda:
            self.reclaim()
        if self.pages is not None:
            self.pages.forget()

    def region(
        self,
        owner: str,
        *,
        release: Callable[[], None] | None = None,
        shrink: Callable[[], bool] | None = None,
    ) -> Region:
        region = Region(self, owner, release, shrink)
        self.regions.add(region)
        return region

    def retained(self, nbytes: int) -> Retained:
        """Where a call keeps `nbytes` of cross-step state, decided once before its loop: varena
        regions. They stay on the card where half the room nothing live holds takes them, with
        none of it spilled; else each is spilled to a pinned host page behind compute and evicted
        like a weight, and varena keeps trying to fit them as it does any memory."""
        if self.native is None:
            return Retained(None)
        room = vt.room(self.device.index or 0)
        card = room["capacity"] - room["working_live"] - room["pinned"]
        return Retained(self, spill=2 * nbytes > card - room["weights_resident"])

    def _empty(self, nbytes: int, reclaim: bool) -> Tensor:
        if self.cuda:
            return torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        if self.device.type == "mps" and reclaim:
            # TensorFS reads into bounded host staging; account for that transient beside
            # the retained MPS part before asking Torch for its destination allocation.
            need = nbytes + min(nbytes, _HOST_STAGING)
            if need > self._free():
                self.reclaim(need)
            if need > (free := self._free()):
                raise DoesNotFit(
                    f"weights need {_gib(need)} on this Mac's GPU and {_gib(free)} is free with "
                    "everything evictable gone: rent a card: `cozy run … --on new`"
                )
        return torch.empty(nbytes, dtype=torch.uint8, device=self.device)

    def _active_beside(self, request: Request | None) -> bool:
        """Live request concurrency; the caller holds lock across its policy decision."""
        return any(other is not request and not other.idle for other in self.requests)

    def _native_region(self, block: _Block, *, final: bool = False) -> varena.Region:
        """Its Region under its overlay: a stable address and a Source, no bytes yet."""
        field = "final_region" if final else "region"
        region: varena.Region | None = getattr(block, field)
        if region is None:
            size = block.final_bytes if final else block.storage_bytes
            source = self._source(block, final=final)
            # Its set: varena packs a component's small units into shared pages (cr-317).
            set_ = f"{block.model.content}/final" if final else block.model.content
            region = self._device_native().region(size, source, set_)
            setattr(block, field, region)
            if not final:
                self._sources[region] = source
        return region

    def _source(
        self,
        block: _Block,
        *,
        final: bool,
        identity: object = None,
        overlay: Hashable = None,
    ) -> varena.Source:
        """Where a Region of `block` gets its bytes. The tier keeps its stored bytes, or those a
        bake wrote, or its cast bytes where those are all a use needs (`_host_final`). Under its
        model's fill identity and its overlay, or those given (a read ahead of a later use)."""
        model = block.model
        if identity is None:
            identity, overlay = model.identity, block.overlay
        kept = identity is not None and final == _host_final(block)
        page = _page(block, identity, overlay, final=final) if kept else None
        owner = weakref.ref(block)  # a Region owns its Source: never the block back
        if final:
            return varena.Source(
                self._tier(),
                model.name,
                block.index,
                block.final_bytes,
                page=page,
                derive=functools.partial(_derive, owner, "final_region", _write_final),
            )
        offset, parts = 0, []
        for key, roles in block.parts.items():
            for part in roles.values():
                name = f"{model.component}/{key}#{part.role or 'value'}"
                parts.append((name, offset, part.nbytes))
                offset += (part.nbytes + 255) // 256 * 256
        derive, slot = None, None
        if model.fill is not None:
            derive = functools.partial(_derive, owner, "region", _bake)
            if page is not None:
                slot = _slot(block, identity, overlay)
        return varena.Source(
            self._tier(),
            model.name,
            block.index,
            block.storage_bytes,
            checkpoint=model.source.ck,
            parts=parts,
            page=page,
            derive=derive,
            slot=slot,
        )

    def _use(self, block: _Block, *, final: bool = False) -> varena.Use:
        """`block`'s bytes pinned for this stream. An idle request takes only what is resident."""
        if not final and (prefilled := block.prefilled) is not None:
            block.prefilled = None
            return prefilled
        request = _REQUEST.get()
        if request is not None and request.idle:
            region = block.final_region if final else block.region
            use = None if region is None else region.pin()
            if use is None:
                raise StaleBinding("idle graph preparation cannot fill native weight backing")
            return use
        return self._native_region(block, final=final).use(self._cuda_stream().cuda_stream)

    def _fill(self, groups: list[_Block]) -> None:
        """`groups`' absent raw Regions filled together before their blocks bind (cr-318): one
        store read for all TensorFS can read so, pieces pipelined across Regions, where a block at a
        time read each alone, serialized (Qwen3.6's 10,541 expert blocks: 37.6 GB at ~1.3 GB/s).
        Each one's Use waits on its block (`_use`). A model whose fill bakes (LoRAs) fills block by
        block: its bake runs on the thread that binds it."""
        request = _REQUEST.get()
        if not self.cuda or (request is not None and request.idle):
            return
        wanted = [
            block
            for block in groups
            if block.model.fill is None
            and block.data is None
            and block.prefilled is None
            and block.storage_bytes
            and not block.final_only()
            and not block.cached()
        ]
        if len(wanted) < 2:
            return
        pairs = [(block, self._native_region(block)) for block in wanted]
        found = [(block, region, self._sources.get(region)) for block, region in pairs]
        bound = [(block, region, source) for block, region, source in found if source is not None]
        if len(bound) < 2:
            return
        wanted = [block for block, _, _ in bound]
        regions = [region for _, region, _ in bound]
        sources = [source for _, _, source in bound]
        if request is not None:
            for block in wanted:
                request.first_read(block)
        model = wanted[0].model
        with _phases.loading(
            model.component,
            sum(block.nbytes for block in wanted),
            functools.partial(self._unread, model),
        ):
            uses = varena.fill(regions, sources, self._cuda_stream().cuda_stream)
        for block, use in zip(wanted, uses, strict=True):
            block.prefilled = use

    def _note_phase_plan(self, status: str, reason: str = "", **facts: Unpack[_PlanNote]) -> None:
        request = _REQUEST.get()
        stamp = (self._phase_request, self._phase_names, status, reason)
        if request is not None and stamp != self._phase_plan_status:
            request.phase_changes.append({
                "models": sorted(self._phase_names or ()), "retention": "native_demand",
                "region_plan": {"status": status, "reason": reason, **facts},
            })
        self._phase_plan_status = stamp

    def _retire_phase_plan(self, reason: str, *, model: Model | None = None) -> None:
        """Retire native intent before a represented source/region changes; no physical release."""
        with self.lock:
            if model is not None and model not in self._phase_plan_models:
                return
            if self._phase_plan is not None:
                self._phase_plan.deactivate()
                self._phase_plan = None
                self._phase_plan_signature = None
                self._phase_plan_stamp = None
                self._phase_plan_keys = {}
                self._phase_plan_models.clear()
                self._note_phase_plan("unplanned", reason)

    def _phase_regions(
        self, names: frozenset[str]
    ) -> tuple[list[varena.Region], list[_PhaseSignature], str]:
        """A phase's Regions in order, under the overlays its blocks last used, what identifies
        them, and why it cannot be planned, if it can't. A Region is an address and a Source."""
        models = [model for model in self.models.values() if model.name in names]
        if len(models) != len(names) or {model.name for model in models} != set(names):
            return [], [], "unresolved_or_ambiguous_models"
        regions, signature = [], []
        for model in sorted(models, key=lambda model: model.name):
            identity = model.wanted_identity()
            if model.closed or model.retiring or identity is None or model.identity != identity:
                return [], [], "representation_identity_incomplete"
            bindings = []
            for block in model.groups():
                if block.gathered is not None:
                    return [], [], "host_gathered_representation"
                raw_needed = (
                    block.aliases is None or any(block.aliases.values()) or not block.finals
                )
                for role, needed, final in (
                    ("raw", raw_needed and bool(block.storage_bytes), False),
                    ("final", bool(block.final_bytes), True),
                ):
                    if not needed:
                        continue
                    region = self._native_region(block, final=final)
                    # Native identity/size are facts, not an estimated byte reserve. The
                    # native Plan deduplicates aliases by Region identity, not these rows.
                    bindings.append((
                        block.path, role, int(region.ptr), int(region.allocated_bytes),
                    ))
                    regions.append(region)
            signature.append((
                model.name, model.content, identity, model.layout_identity, tuple(bindings),
            ))
        return regions, signature, ""

    def _refresh_phase_plan(self, names: frozenset[str]) -> None:
        """The plan the core works from: this request's phases in order, every Region of each,
        with `names` active. Nothing here moves bytes."""
        request = _REQUEST.get()
        steps = [] if request is None else [frozenset(step) for step in request.steps]
        order = list(dict.fromkeys(names if names & step else step for step in steps))
        stamp = self._phase_stamp(names, order)
        if self._phase_plan is not None and stamp == self._phase_plan_stamp:
            # Its Regions can't have changed: every change of one retires the plan. Each forward
            # re-plans in O(models), not O(blocks) (cr-307: 10,541 of them for Qwen3.6).
            try:
                self._phase_plan.activate(self._phase_plan_keys[names])
            except NotImplementedError:
                self._retire_phase_plan("multiple_arenas")
            return
        phases = [] if names in order else [(names, *self._phase_regions(names))]
        phases += [(step, *self._phase_regions(step)) for step in order]
        reason = next((why for step, _, _, why in phases if step == names), "")
        if reason:
            self._retire_phase_plan(reason)
            self._note_phase_plan("unplanned", reason)
            return
        planned = [(step, regions) for step, regions, _, why in phases if not why and regions]
        if not any(step == names for step, _ in planned):
            self._retire_phase_plan("no_regions")  # A model with no weights plans nothing.
            return
        represented: tuple[tuple[list[str] | _PhaseSignature, ...], ...] = tuple(
            (sorted(step), *signature)
            for step, _, signature, why in phases
            if not why and signature
        )
        if self._phase_plan is None or represented != self._phase_plan_signature:
            self._retire_phase_plan("phase_inventory_changed")
            digest = hashlib.sha256(repr(represented).encode()).hexdigest()
            keys = {
                step: f"runtime.phase.regions/1:{digest}/{i}" for i, (step, _) in enumerate(planned)
            }
            self._phase_plan = varena.Plan(
                self._device_native(), [(keys[step], regions) for step, regions in planned]
            )
            self._phase_plan_signature, self._phase_plan_keys = represented, keys
            self._phase_plan_models = {
                model for model in self.models.values()
                if any(model.name in step for step, _ in planned)
            }
            self._note_phase_plan("active", native=self._phase_plan.facts())
        self._phase_plan_stamp = stamp
        try:
            self._phase_plan.activate(self._phase_plan_keys[names])
        except NotImplementedError:
            self._retire_phase_plan("multiple_arenas")

    def _phase_stamp(
        self, names: frozenset[str], order: list[frozenset[str]]
    ) -> tuple[object, ...]:
        """What a plan of `names` among `order` is built from, short of its Regions: the phases,
        and each model in them with its representation (`_phase_regions` reads the same)."""
        named = names.union(*order)
        return (
            names,
            tuple(order),
            tuple(
                (id(model), model.identity, model.wanted_identity(), model.layout_identity,
                 model.generation, model.closed, model.retiring)
                for model in self.models.values()
                if model.name in named
            ),
        )

    def _activate_phase(self, model: Model) -> None:
        """Declare a component-phase boundary; native policy retains/evicts actual pages."""
        request = _REQUEST.get()
        if not self.cuda or self.capturing() or (request is not None and request.idle):
            return
        names = request.phase(model.name) if request is not None else frozenset({model.name})
        request_id = request.id if request is not None else None
        with self.lock:
            if self._active_beside(request):
                self._retire_phase_plan("overlapping_requests")
                return
            changed = names != self._phase_names or request_id != self._phase_request
            if changed:
                self._phase_names, self._phase_request = names, request_id
                if request is not None:
                    request.phase_changes.append(
                        {"models": sorted(names), "retention": "native_demand"}
                    )
            self._refresh_phase_plan(names)
            if changed:
                self._read_ahead(names)

    def _read_ahead(self, names: frozenset[str], *, current: bool = False) -> None:
        """TensorFS reads pages into pinned host RAM ahead of their use (`Pages.read_ahead`),
        within free room, while this phase runs: the largest model of this phase's packages that
        the request has not used yet (text encoding reads the transformer's). This phase's own go
        before it when the last read ahead had them coming (its read goes on), or `current`: an
        overlay armed inside its forward moved its blocks. A page a derive made in an earlier
        process is read from disk, made again by no one."""
        request = _REQUEST.get()
        if self.pages is None or request is None or request.idle:
            return
        request.entered |= names
        active = [m for m in self.models.values() if m.name in names and not m.closed]
        current = current or any(m.name in self._coming for m in active)
        releases = {m.source.release for m in active}
        # The largest model it has not used: what reads it ahead pays for the read and its pages'
        # registration while this phase runs, worth it for the one a cold step waits on most.
        later = sorted(
            (
                m
                for m in self.models.values()
                if m.source.release in releases and m.name not in request.entered and not m.closed
            ),
            key=lambda m: -sum(b.storage_bytes for b in m.groups()),
        )[:1]
        self._coming = {m.name for m in later}
        reading = (active if current else []) + later
        self._staging = [source for model in reading for source in self._staged(model)]
        self._tier().read_ahead(self._staging)

    def _staged(self, model: Model) -> list[varena.Source]:
        """`model`'s raw pages' Sources in block order, under the fill identity its next use wants:
        its Regions' where they are current, else made for the read alone."""
        identity = model.wanted_identity()
        if identity is None:
            return []
        sources: list[varena.Source] = []
        for block in model.groups():
            if not block.storage_bytes or _host_final(block):
                continue
            region = block.region
            source = None
            if region is not None and identity == model.identity:
                source = self._sources.get(region)
            if source is None:
                fill = model.fill
                overlay = None if fill is None else fill.overlay(block.path, block.parts)
                source = self._source(block, final=False, identity=identity, overlay=overlay)
            sources.append(source)
        return sources

    def _final_views(self, block: _Block) -> None:
        if not block.finals:
            return
        fills = block.final_fills
        use = self._use(block, final=True) if block.final_bytes else None
        block.final_native_use = use
        raw = use.tensor if use is not None else self._empty(0, False)
        if use is not None:
            block.final_storage = int(raw.untyped_storage().data_ptr())
            if block.final_fills == fills:
                block.final_hits += 1
                if (request := _REQUEST.get()) is not None:
                    request.final_hits += 1
        block.final_data = _finals(block, raw)

    @torch.inference_mode(False)
    def _ensure(self, block: _Block, reclaim: bool, overlay: Hashable) -> None:
        if block.data is not None:
            return
        if self.capturing():
            raise ResidencyError("all graph weights must be resident before capture")
        with self.lock:
            block.loading = True
        try:
            with self._read_told(block):
                if self.native is None:
                    data = self._read_host(block, reclaim, overlay)
                else:
                    block.overlay = overlay
                    self._final_views(block)
                    data = {}
                    if not block.final_only():
                        # Final bytes alone serve a final-only binding: nothing here touches its
                        # raw Region, so an evicted one costs no read.
                        raw = self._empty(0, False)
                        if block.storage_bytes:
                            block.native_use = self._use(block)
                            raw = block.native_use.tensor
                            block.native_storage = int(raw.untyped_storage().data_ptr())
                        data = _data(block, raw)
            ready = None
            if self.cuda:
                ready = torch.cuda.Event()
                ready.record(self.stream())
            with self.lock:
                block.data, block.ready = data, ready
        except BaseException:
            block.native_storage = block.native_use = None
            self._drop_final(block)
            raise
        finally:
            with self.lock:
                block.loading = False

    def _read_told(self, block: _Block) -> AbstractContextManager[None]:
        """A read of `block`'s weights, told as its component's load (`_phases.loading`): none for
        one resident. A block this request read before is read again (paging): its step's time."""
        if not block.nbytes or block.cached():
            return nullcontext()
        model, request = block.model, _REQUEST.get()
        again = request is not None and not request.first_read(block)
        return _phases.loading(
            model.component, block.nbytes, functools.partial(self._unread, model), again=again
        )

    @staticmethod
    def _unread(model: Model) -> int:
        """The bytes of `model` no fill holds now: what a load of it that begins now reads."""
        return sum(b.nbytes for b in model.groups() if b.data is None and not b.cached())

    def _read_host(
        self, block: _Block, reclaim: bool, overlay: Hashable
    ) -> dict[str, dict[str, Tensor]]:
        """Where no native core runs (MPS, CPU): Torch-owned storage, read and baked here."""
        source = block.model.source
        tensors = source.ck.components[source.component].tensors
        data: dict[str, dict[str, Tensor]] = {}
        read_ns = read_bytes = 0
        bake: Bake | None = None
        try:
            for key, parts in block.parts.items():
                views = data[key] = {}
                for role, part in parts.items():
                    raw = self._empty(part.nbytes, reclaim)
                    began = time.perf_counter_ns()
                    self._read_into(tensors, key, raw, part.role or None)
                    read_ns += time.perf_counter_ns() - began
                    read_bytes += part.nbytes
                    views[role] = _view(key, role, part, raw)
            if block.model.fill is not None:
                baking = Bake.begin(self.stream())
                bake = baking.end(self.stream()) if block.model.fill(block.path, data) else None
                block.overlay = overlay
        finally:
            if (request := _REQUEST.get()) is not None and read_bytes:
                request.read(block, read_ns, read_bytes, bake=bake)
        return data

    def _placed_after_reads(self, block: _Block, overlay: Hashable) -> bool:
        """`block` placed after the commons bound only for reads outside their forwards, of
        components not running, went (`_let_go_reads`): False when none was bound, or it still
        has no room. LTX-2.5's denoise held its VAE's, connectors' and audio VAE's beside its
        transformer's first block, 1.3 GiB of a 6 GiB cap (var-030 §12)."""
        if not self._let_go_reads():
            return False
        try:
            self._ensure(block, True, overlay)
        except Exhausted:
            return False
        block.gathered = None
        return True

    def _placed_after_shed(self, block: _Block, overlay: Hashable) -> bool:
        """`block` placed after what holds weights only for speed let them go (`sheds`): False
        when nothing did, or it still has no room. A refused block beside them is the refusal
        that gives way first (cr-319: Klein's upsampler at a 6 GiB cap held 4.0 GiB of its 10.1
        to decode streamed, and a 0.2 GiB layer it streams had no room; gate 44062127052)."""
        need = block.allocation_bytes
        if not any([shed(need) for shed in list(self.sheds)]):
            return False
        try:
            self._ensure(block, True, overlay)
        except Exhausted:
            return False
        block.gathered = None
        return True

    @torch.inference_mode(False)
    def _gather(self, block: _Block, overlay: Hashable) -> None:
        """An embedding table that does not fit the room stays on the host: its forward looks up
        there and moves only the rows it read (`_gathering`), so a room below the table never
        fails the call (Qwen-Image's 1.2 GB text-encoder table beside a 5 GB game on 8 GB). The
        copy is the host tier's page when it keeps one, else read once and held while the table
        gathers; each entry tries the device first. A linear layer refused so (`_Block.leaf`)
        stays there the same way, in pinned memory; its windows' rows are copied in
        (`_streamed`)."""
        self._retire_phase_plan("host_gathered_representation", model=block.model)
        if block.gathered is None or block.gathered[0] != overlay:
            block.gathered = None
            model = block.model
            raw = _host(block.storage_bytes, pinned=not block.table)
            began = time.perf_counter_ns()
            self._source(block, final=False).read_host(raw.numpy(), stored=model.fill is not None)
            if (request := _REQUEST.get()) is not None:
                request.read(block, time.perf_counter_ns() - began, block.nbytes)
            data = _data(block, raw)
            if model.fill is not None:
                model.fill(block.path, data)
            finals: dict[str, dict[str, Tensor]] = {}
            for key, row in block.finals.items():
                finals[key] = {
                    name: torch.empty(part.nbytes, dtype=torch.uint8)
                    .view(getattr(torch, _DTYPES[part.dtype]))
                    .as_strided(part.shape, part.strides)
                    for name, part in row.items()
                }
                if model.finals is not None:
                    model.finals.write(key, data[key], finals[key])
                else:
                    for destination in finals[key].values():
                        destination.copy_(data[key]["value"])
            block.gathered = overlay, data, finals
        with self.lock:
            block.data, block.final_data, block.ready = block.gathered[1], block.gathered[2], None
        if (request := _REQUEST.get()) is not None and block.table:
            request.gathered += 1

    def _streams(self, block: _Block) -> bool:
        """Whether `block`, refused room, can run from the host: one linear layer whose output
        rows compute alone (`rows.streamed`), outside a capture and a graph's binding (whose
        refusal tells it to run eagerly)."""
        if block.leaf is None or not self.cuda or self.capturing() or block.retained:
            return False
        from cozy_runtime.accel import rows

        return rows.streamed(block.leaf, rows._slice, rows._same) is not None

    def _streamed(self, block: _Block, x: Tensor) -> Tensor:
        """`block`'s layer over `x` from its host copy (`_gather`): in windows of its output rows,
        each window's rows copied in on this stream and computed as `rows` computes them, the
        windows as many as the card's room takes (`catch.cut`, which goes finer on a refusal).
        One window is the whole layer's own call. Each byte crosses the bus once a forward."""
        from cozy_runtime.accel import rows
        from cozy_runtime.internal.execution import catch, sizing

        leaf = cast(nn.Module, block.leaf)
        if torch.cuda.is_current_stream_capturing():
            raise Exhausted(f"{block.model.component}/{block.path} is on the host: no capture")
        device = self.device

        def take(tensor: Tensor, start: int, stop: int) -> Tensor:
            return tensor[start:stop].to(device, non_blocking=True)

        def whole(tensor: Tensor) -> Tensor:
            return tensor.to(device, non_blocking=True)

        compute = rows.streamed(leaf, take, whole)
        if compute is None:  # its hooks changed since its entry
            raise Exhausted(f"{block.model.component}/{block.path} has no room and does not cut")
        width = cast(int, leaf.out_features)
        tile = rows.tile(leaf)
        units = -(-width // tile)
        per_row = block.storage_bytes // width
        tokens = x.numel() // max(x.shape[-1], 1)
        out_bytes = tokens * width * x.element_size()

        def need(level: int) -> tuple[int, ...]:
            n = min(width, sizing.pieces(units, level) * tile)
            return out_bytes, n * per_row, out_bytes * n // width

        def run(level: int) -> Tensor:
            n = min(width, sizing.pieces(units, level) * tile)
            if n >= width:
                return compute(x, 0, width)
            out: Tensor | None = None
            for start in range(0, width, n):
                stop = min(width, start + n)
                found = compute(x, start, stop)
                if out is None:
                    out = found.new_empty((*found.shape[:-1], width))
                out[..., start:stop].copy_(found)
                del found
            return cast(Tensor, out)

        if (request := _REQUEST.get()) is not None:
            request.streamed += 1
        name = f"{block.model.component}.{block.path}:rows"
        return catch.cut(name, sizing.levels(units), need, run)

    def _read_into(
        self, tensors: tensorfs.Tensors, key: str, raw: Tensor, role: str | None
    ) -> None:
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
        model = block.model
        graphs.entering(model.module)
        overlay = None if model.fill is None else model.fill.overlay(block.path, block.parts)
        if overlay != block.overlay and (block.paged() or block.parked):
            self._switch_overlay(block, overlay)
            self._rearm(model)
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
            if not nested:
                request = _REQUEST.get()
                reclaim = reclaim and not (request is not None and request.idle)
                if block.escape_site and request is not None and not request.idle and not capturing:
                    block.born_before = self._entered(block, request)
                try:
                    gathered = block.gathered is not None and block.gathered[0] == overlay
                    if (
                        block.table
                        and model.gathers
                        and self.native is not None
                        and not capturing
                        and (gathered or not block.cached())
                    ):  # `Model.gathering`: on the host, it stays there to the loop's end
                        with nullcontext() if gathered else self._read_told(block):
                            self._gather(block, overlay)
                    else:
                        self._ensure(block, reclaim, overlay)
                        block.gathered = None
                except Exhausted:  # varena refused its region before its forward ran
                    if not (reclaim and self.cuda and not capturing):
                        raise
                    if not (
                        self._placed_after_reads(block, overlay)
                        or self._placed_after_shed(block, overlay)
                    ):
                        if not (block.table or self._streams(block)):
                            raise
                        gathered = block.gathered is not None and block.gathered[0] == overlay
                        with nullcontext() if gathered else self._read_told(block):
                            self._gather(block, overlay)
            # A cached Hold can keep count > 0 across sequential threads or streams.
            # Every consumer needs readiness in addition to storage ownership.
            # Capture's owner joined the prepared stream before capture began. Waiting on
            # its external readiness event now would violate CUDA capture isolation.
            if block.ready is not None and not capturing:
                self._cuda_stream().wait_event(block.ready)
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
                    if self.device.type == "cpu":
                        block.data = {}  # its widened copies serve: the stored ones go
                for site, value in block.assigned:
                    block.saved.append((site, site.table[site.name]))
                    site.table[site.name] = value
            self._used(block)
            if not nested and self._probe is None:
                self._start_probe(block, capturing)
        except BaseException:
            self._exit(block)
            raise

    def block_need(self, shape: int) -> int:
        """The largest working peak above its entry this process measured for a model's first block at pass shape
        `shape` (`_end_probe`), 0 where none was: what a block of a stack at that shape needs beside what is live."""
        return max((need for (_, at), need in self._probed.items() if at == shape), default=0)

    def _start_probe(self, block: _Block, capturing: bool) -> None:
        """Measure `block` if it is its model's first at this pass shape in this process: the
        working peak resets here, and `_end_probe` reads it beside varena's room (cr-305)."""
        request = _REQUEST.get()
        key = (block.model.name, PASS_SHAPE.get())
        if (
            not self.cuda
            or capturing
            or not block.path  # the model's common block spans its whole forward
            or request is None
            or request.idle
            or key in self._probed
            or not torch.cuda.is_initialized()
        ):
            return
        device = self.device.index or 0
        request.sample()  # the request keeps its peak so far
        stats, room = vt.stats(device), vt.room(device)
        vt.reset_peak(device)
        self._probe = _Probe(
            block,
            key,
            int(stats["mapped_bytes"]),
            int(stats["allocated_bytes"]),
            int(room["weights_resident"]),
            int(room["capacity"]),
        )

    def _end_probe(self, probe: _Probe) -> None:
        self._probe = None
        self._probed[probe.key] = 0
        device = self.device.index or 0
        request = _REQUEST.get()
        if request is None:
            return
        stats, room = vt.stats(device), vt.room(device)
        peak, allocated = int(stats["allocated_peak"]), int(stats["allocated_bytes"])
        self._probed[probe.key] = peak - probe.allocated
        request.first_blocks.append(
            {
                "block": f"{probe.block.model.component}/{probe.block.path}",
                "shape": probe.key[1],
                # varena's room samples the driver: what it lost outside varena's pages
                "non_torch": probe.capacity - int(room["capacity"]),
                "mapped_growth": int(stats["mapped_bytes"]) - probe.mapped,
                "peak_above_entry": peak - probe.allocated,
                "transient": peak - allocated,
                "fragmentation": int(stats["mapped_peak"]) - peak,
                "live_at_entry": probe.allocated,
                "weights_at_entry": probe.weights,
                "capacity_at_entry": probe.capacity,
                "capacity": int(room["capacity"]),
            }
        )

    def _used(self, block: _Block) -> None:
        if self.cuda:
            stream = self._cuda_stream()
            # Varena records its own last-consumer event on each declared stream when the
            # storage's final alias goes, so reclaim waits on that region, not the context.
            for use in (block.native_use, block.final_native_use):
                if use is not None:
                    use.consume_on(stream.cuda_stream)
            for tensor in block.computed():
                tensor.record_stream(stream)

    def _exit(self, block: _Block) -> None:
        try:
            self._used(block)
        finally:
            with self.lock:
                block.count -= 1
                probe = self._probe if block.count == 0 and self._probe is not None else None
                if block.count == 0:
                    for site, previous in reversed(block.saved):
                        site.table[site.name] = previous
                    block.saved.clear()
                    if block.born_before is not None:
                        self._born(block.born_before)
                        block.born_before = None
                    block.last = time.monotonic_ns()
                    self._idle(block)
            if probe is not None and probe.block is block:
                self._end_probe(probe)

    def _read(
        self, block: _Block, site: _Site, raw_view: tuple[int, _Part] | None = None
    ) -> Tensor:
        model = block.model
        model._claim()
        try:
            if not getattr(model.local, "depth", 0):
                self._declare(model)
            elif block is model.common and self._scoped(model) and not self.capturing():
                # Read outside its forwards in a call that declared it (an overlay's factors,
                # a VAE's latent statistics): bound until the next outermost forward begins, the
                # scope ends or a refused block takes it back (`_let_go_reads`), so an overlay
                # read in every layer binds once a forward, never once an op (cr-207).
                self._enter(block)
                model.local.held = True
                return _bound(model, site)
            if (
                raw_view is not None
                and model.fill is None
                and not (block.count or block.retained or block.loading)
                and block.data is None
                and block.ready is None
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
                        raw = use.tensor[offset : offset + part.nbytes]
                        value = _view(site.name, part.role, part, raw)
                        if widen is not None:
                            value = value.to(widen)
                        block.last = time.monotonic_ns()
                        request = _REQUEST.get()
                        if block.escape_site and request is not None and not request.idle:
                            self._entered(block, request)  # a resident block's read is its entry
                        return nn.Parameter(value, requires_grad=False) if site.parameter else value
            self._enter(block)
            try:
                return _bound(model, site)
            finally:
                self._exit(block)
        finally:
            model._release()

    def _scoped(self, model: Model) -> bool:
        """Whether `model`'s common block waits for its own use on this thread: varena pages it, a
        call's scope declared it and none of its forwards runs, so nothing holds it yet."""
        return (
            self.native is not None
            and getattr(model.local, "depth", 0) > 0
            and not getattr(model.local, "running", 0)
            and not getattr(model.local, "held", False)
        )

    def _let_go_reads(self) -> bool:
        """The common blocks this thread's scopes bound for reads outside their forwards, given
        back where their component is not running: a refused block comes before a read's speed.
        A later read binds them again. Whether any went."""
        went = False
        for model in list(self.models.values()):
            local = model.local
            if getattr(local, "held", False) and not getattr(local, "running", 0):
                local.held = False
                if model.common is not None:
                    self._exit(model.common)
                    went = True
        return went

    def _open(self, model: Model, key: str, *, executing: bool = False) -> None:
        """A scope of `model` (a call that declared it) or one of its forwards begins. Under varena
        its common block is held while it runs, not while a scope only names it: LTX-2.5's
        denoise held four components' commons through the connectors, whose 1.44 GiB projection
        then had no room at a 6 GiB cap (var-030 §12). A read outside its forwards binds it then
        (`_read`). Elsewhere (CPU, MPS) the scope holds it, as their bytes are not paged."""
        model._claim()
        depth = getattr(model.local, "depth", 0)
        running = getattr(model.local, "running", 0)
        model.local.depth = depth + 1
        request = _REQUEST.get()
        outer_forward = False
        if executing:
            model.local.running = running + 1
            with self.lock:
                outer_forward = self._forward_depth == 0
                self._forward_depth += 1
                self._executing[model] = self._executing.get(model, 0) + 1
        held = (executing and not running) if self.native is not None else not depth
        try:
            if not depth:
                self._declare(model)
                # Its own forward (a hooked child's is one part of a call that may run many).
                if executing and model.escape_site and request is not None and key.startswith("|"):
                    request.forwards.add(model)
                    self._pass(model, request, hash(key) & _U64 or 1)
            if outer_forward:
                if self.native is not None and not self.capturing():
                    # What reads bound for the last forward's sake stays bound no longer: the
                    # VAE statistics a refine read held 832 MiB through its whole loop.
                    self._let_go_reads()
                self._activate_phase(model)
            if not depth:
                model.local.request = request
                if request is not None:
                    request.current.add(model.name)
            if held and model.common is not None:
                self._enter(model.common)
        except BaseException:
            model.local.depth = depth
            if executing:
                model.local.running = running
                with self.lock:
                    self._forward_depth -= 1
                    self._leave(model)
            if request is not None and not depth:
                request.current.discard(model.name)
            model._release()
            raise

    def _close(self, model: Model, *, executing: bool = False) -> None:
        try:
            model.local.depth -= 1
            if executing:
                model.local.running -= 1
            depth, running = model.local.depth, getattr(model.local, "running", 0)
            try:
                held = (executing and not running) if self.native is not None else not depth
                if held and model.common is not None:
                    self._exit(model.common)
            finally:
                if not depth:
                    if model.escape_site:
                        self._born(0)
                    try:
                        if getattr(model.local, "held", False):
                            model.local.held = False
                            if model.common is not None:
                                self._exit(model.common)
                    finally:
                        request = getattr(model.local, "request", None)
                        if request is not None:
                            request.current.discard(model.name)
                            request.sample()
        finally:
            if executing:
                with self.lock:
                    self._forward_depth -= 1
                    self._leave(model)
            model._release()

    def _leave(self, model: Model) -> None:
        if (left := self._executing[model] - 1) == 0:
            del self._executing[model]
        else:
            self._executing[model] = left

    def _born(self, site: int) -> int:
        """Allocations on this device are born at `site` from now; the site they were born at."""
        before, self._site = self._site, site
        vt.set_site(self.device.index or 0, site)
        return before

    def _entered(self, block: _Block, request: Request) -> int:
        """Its allocations are born here (the site before it, returned, comes back at its exit). A
        model whose forward runs no hook of its own (its package calls into its blocks) begins a
        pass at its first block, and at each block it enters again after others: a forward enters
        each of its blocks once, in whatever order they were registered (a mid block's attention
        before its resnet), and a weight read enters its block again with none between. A model
        whose blocks run inside another's forward (an overlay's tables and factors) is part of that
        forward's pass: marking its own at its first block would count what that forward made
        before it (its packed rows) as kept from the last pass."""
        model = block.model
        before = self._born(block.escape_site)
        if model in request.forwards:
            return before
        with self.lock:
            inside = any(other is not model for other in self._executing)
        if inside:
            return before
        seen, last = request.heads.get(model, (None, None))
        if seen is None or (block.index in seen and block.index != last):
            request.heads[model] = ({block.index}, block.index)
            self._pass(model, request, PASS_SHAPE.get())
            self._born(block.escape_site)
        elif block.index != last:
            seen.add(block.index)
            request.heads[model] = (seen, block.index)
        return before

    def _pass(self, model: Model, request: Request, shape: int) -> None:
        """A forward of `model` begins, a pass of its loop (`vt.pass_mark`; `shape` 0: unknown):
        what earlier passes left live is tallied by where it was born, and a next pass that cannot
        fit with every evictable byte gone is refused here, before it runs, never partway through it."""
        if request.idle or self.capturing():
            return
        if model.escape_floor is None:
            # A block the card need not hold whole sets no floor: an embedding table gathers its
            # rows on the host (`_gather`), and one linear layer streams its rows (`_streamed`).
            # Counted, Qwen3.8-27B's 2.37 GiB bf16 table and head refused every decode pass at a
            # 6 GiB cap ("needs 5.4 GiB ... 2.6 GiB of weights", cr-344).
            common = model.common.allocation_bytes if model.common is not None else 0
            model.escape_floor = common + max(
                (
                    b.allocation_bytes
                    for b in model.blocks.values()
                    if not (b.table or b.leaf is not None)
                ),
                default=0,
            )
        floor = model.escape_floor
        if any(b.retained for b in model.groups()):
            # Held blocks are in the pass's held bytes already: the floor is what still streams
            # (a decode keeping what fits, `keep`); an embedding table gathers on the host.
            shared = model.common
            floor = (shared.allocation_bytes if shared and not shared.retained else 0) + max(
                (
                    b.allocation_bytes
                    for b in model.blocks.values()
                    if not (b.retained or b.table or b.leaf is not None)
                ),
                default=0,
            )
        device = self.device.index or 0
        refused = vt.pass_mark(device, model.escape_site, shape, floor)
        self._born(model.escape_site)
        if refused is not None:
            raise DoesNotFit(_does_not_fit(model.component, refused))

    @contextmanager
    def request(
        self,
        id: str,
        plan: Sequence[object] = (),
        *,
        idle: bool = False,
    ) -> Iterator[Request]:
        request = Request(self, id, idle)
        request.plan(plan)
        with self.lock:
            if not idle and any(not other.idle for other in self.requests):
                self._retire_phase_plan("overlapping_requests")
                request.overlap = True
                for other in self.requests:
                    if not other.idle:
                        other.overlap = True
            self.requests.add(request)
        token = _REQUEST.set(request)
        # An idle period's work is no request to the core: what it records is a guess, taken from
        # free room only (pool_reserve), and peers order it among the idle.
        if self.cuda and not idle:
            varena.request_boundary(self.device.index or 0, begin=True)
        try:
            yield request
        finally:
            try:
                with self.lock:
                    if self._phase_request == request.id:
                        self._retire_phase_plan("request_ended")
                self.forget_working_phases(request)
                if self.pages is not None and not idle:
                    request.record_fills()
                    self.pages.request_ended()  # the pages it refilled are their worth
                    self._staging, self._coming = [], set()
                if self.cuda and not idle and torch.cuda.is_initialized():
                    # or a capacity failure raised through here, never a package's own error
                    raised = sys.exc_info()[1]
                    failed = request.failed or (raised is not None and is_capacity(raised))
                    request.escaped = vt.escapes_take(self.device.index or 0, failed=failed)
                    _escaped_warnings(request.escaped)
                request.sample()
                request.ended = request.stats()
                self.last = request.ended
                if request.hosted:  # execution imports residency: loaded by now
                    from cozy_runtime.internal.execution import pinned

                    pinned.release()
            finally:
                try:
                    with self.lock:
                        self.requests.discard(request)
                finally:
                    _REQUEST.reset(token)
                    if self.cuda and not idle:
                        varena.request_boundary(self.device.index or 0, begin=False)
                        if request.ended is not None and self.native is not None:
                            # Read again once ended, as varena measures there what this process
                            # holds outside its books: the run record's `out_of_books_bytes` (var-024).
                            request.ended["weight_residency"] = self.native.stats()

    def stats(self) -> RequestStats | None:
        return self.last

    def torch_stats(self) -> TorchStats:
        allocated, reserved = self._figures()
        peak = self._reading()[2]
        return {
            "allocated": allocated,
            "mapped": reserved,
            "peak": peak,
            "requests": {r.id: (allocated, max(r.torch_peak, peak)) for r in self.requests},
            "allocator_scope": "process",
            "weights": self.native.stats() if self.native is not None else None,
        }


class _PlanNote(TypedDict, total=False):
    native: PlanFacts


class RegionPlan(_PlanNote):
    status: str
    reason: str


class PhaseChange(TypedDict):
    """A change of the request's phase models (`Memory.phase`), and its native region plan's."""

    models: list[str]
    retention: str
    region_plan: NotRequired[RegionPlan]


class FillEntry(TypedDict):
    """One block fill (`Request.fill`): a demand fill also has its read's times, a baked one its bake's."""

    sequence: int
    set: str
    block: int
    bytes: int
    origin: str
    speculative: bool
    ns: NotRequired[int]
    at_unix_ms: NotRequired[float]
    bake_ns: NotRequired[int]
    bake_device_ns: NotRequired[int | None]


class _BakeTimes(TypedDict, total=False):
    """A bake's times, on its fill's trace entry or, when the fill is not traced, alone."""

    bake_ns: int
    bake_device_ns: int | None


class Moved(TypedDict):
    bytes: int
    fills: int
    ns: int


class BakeTotals(TypedDict):
    blocks: int
    host_ns: int
    device_ns: int


class FinalBindings(TypedDict):
    fills: int
    hits: int
    resident_scope: str
    raw_resident_bytes: int
    final_resident_bytes: int
    fallback_blocks: int


class RequestStats(TypedDict):
    """`Request.stats`: the request's memory record. CUDA adds its allocator and residency
    figures, MPS its observed allocation."""

    request: str
    allocator_scope: str
    overlapping: bool
    sets: dict[str, int]
    #: by origin (`Request.moved`), and "bake": the bakes' totals
    moved: dict[str, Moved | BakeTotals]
    baked_pages: PagesStats | None
    trace: list[FillEntry]
    trace_scope: str
    phases: list[PhaseChange]
    working_phases: list[WorkingPhase]
    working_splits: list[SplitRecord]
    escaped: list[EscapeRecord]
    retries: int
    #: `Request.hosted`
    hosted: dict[str, str]
    #: `Request.first_blocks`
    first_blocks: list[FirstBlock]
    regrown: int
    swallowed_ooms: int
    #: refusals a cheaper step answered, by the package scope they came in (`contract._Scopes`)
    answered: dict[str, int]
    retained_bytes: int
    gathered: int
    streamed: int
    host: None
    allocated: NotRequired[int]
    reserved: NotRequired[int]
    working_allocator: NotRequired[WorkingStats]
    weight_residency: NotRequired[DeviceStats | None]
    final_bindings: NotRequired[FinalBindings]
    allocator: NotRequired[dict[str, int]]
    #: "completed", the core's ahead-of-demand counters (`_ahead`) and "scope"
    prefetch: NotRequired[dict[str, int | str]]
    measurement: NotRequired[str]


class FirstBlock(TypedDict):
    """A model's first block at a pass shape, measured as vLLM's profile run splits a forward, in
    bytes: the room it lost outside varena's pages (`non_torch`: context, library workspaces,
    kernel modules), the working pages it mapped, the working peak above entry, its `transient`
    part (the peak less what the block left allocated), and mapped less allocated at the peaks
    (`fragmentation`)."""

    block: str
    shape: int
    non_torch: int
    mapped_growth: int
    peak_above_entry: int
    transient: int
    fragmentation: int
    live_at_entry: int
    weights_at_entry: int
    capacity_at_entry: int
    capacity: int


@dataclass(slots=True)
class _Probe:
    """`Memory._start_probe`'s readings at the block's entry."""

    block: _Block
    key: tuple[str, int]
    mapped: int
    allocated: int
    weights: int
    capacity: int


class SplitChoice(TypedDict):
    """What a split site's level was chosen against (`execution.catch.note`), in bytes."""

    full: int
    budget: int
    capacity: int
    live: int
    weights: int
    required: int
    available: int
    target: str | None


class SplitRefusal(TypedDict):
    level: int
    pure: bool


class SplitProbe(TypedDict):
    """A function's first call of a shape, proved row-local or not (`cuts._probe`): what it is, the
    verdict, and what the probe cost on the call's path."""

    function: str
    local: bool
    ms: float
    bytes: int


class SplitTime(TypedDict):
    """A level's calls, summed: `device_ms` between two events on the call's stream (the GPU's
    span, idle time inside it included, and the sites inside its windows), `host_ms` on the
    calling thread."""

    device_ms: float
    host_ms: float


class SplitRecord(TypedDict):
    """One split site's calls this request, for the run record: calls per level, each level's last
    choice and time, the level whose allocation was refused last, if one was, and its probe, if
    one ran. Its reader: what a capped run's windows cost each site against the same request
    uncapped (var-030)."""

    site: str
    levels: dict[str, int]
    chosen: dict[str, SplitChoice]
    time: NotRequired[dict[str, SplitTime]]
    refused: NotRequired[SplitRefusal]
    probe: NotRequired[SplitProbe]


#: A split call's span: its key, level, start and stop events, and host nanoseconds.
type SplitSpan = tuple[str, str, torch.cuda.Event, torch.cuda.Event, int]


class TorchStats(TypedDict):
    """`Memory.torch_stats`: working bytes, by process and by open request, and the weight Regions'."""

    allocated: int
    mapped: int
    peak: int
    #: request id: (allocated, peak)
    requests: dict[str, tuple[int, int]]
    allocator_scope: str
    weights: DeviceStats | None


class Retained:
    """One loop's cross-step state (`author.retained`). Each `put` is one `varena.Retained`:
    written on this stream, resident while there is room, else refilled from its host copy and
    filled ahead of its next use. Where no native core runs the tensors stay as they are. `close`
    (or leaving its `with`) ends it with the loop: its regions go then, whoever still holds a state.
    """

    def __init__(self, memory: Memory | None, *, spill: bool = True) -> None:
        self.memory, self.spill = memory, spill
        self.states: list[_Held | _Kept] = []

    def __enter__(self) -> Retained:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """The loop that reads this state is done. Kept on the card, the edit's K/V stayed pinned
        through the VAE decode after it, which then had no room (run 41362)."""
        for state in self.states:
            state.close()
        self.states.clear()

    def put(self, *tensors: Tensor) -> _retained.State:
        memory = self.memory
        if memory is None or not tensors[0].is_cuda:
            self.states.append(_Held(tensors))
            return self.states[-1]
        # Each contiguous copy lives until the write is on the stream: freed in the loop, a
        # transposed key's copy was the next value's, and HiDream-O1's prefix read back as noise.
        dense = [tensor.contiguous() for tensor in tensors]
        parts, views, end = [], [], 0
        for tensor in dense:
            parts.append((end, tensor.data_ptr(), tensor.nbytes))
            views.append((end, tensor.nbytes, tensor.dtype, tensor.shape))
            end = (end + tensor.nbytes + 255) // 256 * 256
        region = varena.Retained(memory._device_native(), memory._tier(), end)
        # None (a newer TensorFS's): the card had no room, so it went to its host copy and its
        # first use fills it from there.
        written = cast(
            "varena.Use | None",
            region.write(memory._cuda_stream().cuda_stream, parts, spill=self.spill),
        )
        ptr = int(getattr(region, "ptr", 0)) if written is None else written.tensor.data_ptr()
        del dense, written
        kept = _Kept(memory, region, views, ptr, end)
        KEPT.add(kept)
        self.states.append(kept)
        return self.states[-1]


class _Held:
    def __init__(self, tensors: tuple[Tensor, ...]) -> None:
        self.tensors: tuple[Tensor, ...] | None = tensors

    def get(self) -> tuple[Tensor, ...]:
        if self.tensors is None:
            raise RuntimeError("retained state read after its loop ended")
        return self.tensors

    def close(self) -> None:
        self.tensors = None


#: A phase's model in a plan's identity: name, content, fill identity, layout identity, and each Region
#: it binds as (block path, role, address, bytes).
type _PhaseSignature = tuple[str, str, str, str, tuple[tuple[str, str, int, int], ...]]


class _Kept:
    """One retained state: its Region, the views `put` made of it, and the Region's device
    address and bytes, which a CUDA graph that reads it pins at each replay (`graphs`)."""

    def __init__(
        self,
        memory: Memory,
        region: varena.Retained | None,
        views: list[tuple[int, int, torch.dtype, torch.Size]],
        ptr: int = 0,
        nbytes: int = 0,
    ) -> None:
        self.memory, self.region, self.views = memory, region, views
        self.ptr, self.nbytes = ptr, nbytes

    def close(self) -> None:
        # The last reference to its region: it is released now, and views still out keep their pins.
        self.region = None

    def get(self) -> tuple[Tensor, ...]:
        if self.region is None:
            raise RuntimeError("retained state read after its loop ended")
        try:
            raw = self.region.use(self.memory._cuda_stream().cuda_stream).tensor
        except Exhausted as exc:
            size = sum(n for _, n, _, _ in self.views)
            raise Exhausted(
                f"a retained state of {_gib(size)} found no room ({exc}): the states whose "
                "tensors are held now stay on the card; read each where it is used",
                exc.requested_bytes,
            ) from exc
        return tuple(raw[at : at + n].view(dtype).view(shape) for at, n, dtype, shape in self.views)


#: every live retained state: a capture that reads one records it (`graphs.observe`)
KEPT: weakref.WeakSet[_Kept] = weakref.WeakSet()


def kept_ranges() -> list[tuple[int, int, _Kept]]:
    """Each live retained state's device range, by address."""
    live = [kept for kept in list(KEPT) if kept.region is not None and kept.ptr]
    return sorted(((kept.ptr, kept.ptr + kept.nbytes, kept) for kept in live), key=lambda row: row[0])


class _Retaining:
    def retained(self, nbytes: int) -> Retained:
        request = _REQUEST.get()
        if request is None:
            return Retained(None)
        request.retained_bytes += nbytes
        return request.memory.retained(nbytes)


_retained.install(_Retaining())


@dataclass(slots=True)
class Bake:
    """A block's bake (its model's `fill` hook) after its store read: on the host clock, and on
    CUDA between events on the stream its kernels ran on."""

    began_ns: int
    start: torch.cuda.Event | None = None
    host_ns: int = 0
    stop: torch.cuda.Event | None = None

    @classmethod
    def begin(cls, stream: torch.cuda.Stream | None) -> Bake:
        start = None
        if stream is not None:
            start = torch.cuda.Event(enable_timing=True)
            start.record(stream)
        return cls(time.perf_counter_ns(), start)

    def end(self, stream: torch.cuda.Stream | None) -> Bake:
        self.host_ns = time.perf_counter_ns() - self.began_ns
        if self.start is not None:
            self.stop = torch.cuda.Event(enable_timing=True)
            self.stop.record(stream)
        return self

    def device_ns(self) -> int | None:
        if self.stop is None or self.start is None:
            return None
        self.stop.synchronize()
        return int(self.start.elapsed_time(self.stop) * 1e6)


class Request:
    """Request context; Torch observations are process-wide, including overlapping calls."""

    def __init__(self, memory: Memory, id: str, idle: bool = False) -> None:
        self.memory, self.id, self.idle = memory, id, idle
        self.current: set[str] = set()
        self.steps: list[set[str]] = []
        self.declared_steps: list[set[str]] = []
        self.phase_changes: list[PhaseChange] = []
        self.working_phases: list[WorkingPhase] = []
        self.working_phase_keys: set[str] = set()
        self.working_splits: dict[str, SplitRecord] = {}
        #: split calls whose events `stats` has not read yet (`execution.catch.timed`)
        self.split_spans: deque[SplitSpan] = deque()
        #: each loop's record of what its passes left live outside any owner (`vt.escapes_take`);
        #: the models whose forwards mark their passes, and of the others, the blocks each entered
        #: this pass, in order
        self.escaped: list[EscapeRecord] = []
        self.forwards: set[Model] = set()
        self.heads: dict[Model, tuple[set[int], int]] = {}
        #: the models whose phases this request entered: what it has not is read ahead
        self.entered: set[str] = set()
        #: the blocks it read (`Memory._read_told`), by identity: one read again is paging
        self.read_blocks: set[int] = set()
        #: the attempt failed (its executor says): a refused allocation in its last pass is learned
        self.failed = False
        #: refusals the Runtime answered by running again another way (a pure op's finer split, an LM
        #: token decoded eagerly for a capture its pool couldn't hold): zero in a normal run
        self.retries = 0
        #: split sites whose whole output the card had no room for, kept in host memory (cr-305):
        #: the card's room when its allocation was refused there, else whether selection judged
        #: the package's estimate of it or its measure (`execution.cuts._hosted`)
        self.hosted: dict[str, str] = {}
        #: refusals a CUDA graph capture answered by recording again with room for them
        #: (`graphs.capture`): what runs is the same
        self.regrown = 0
        #: each model's first block measured at a pass shape (`Memory._start_probe`)
        self.first_blocks: list[FirstBlock] = []
        #: refusals a package scope returned past (`contract._Scopes`): zero in a conforming run
        self.swallowed_ooms = 0
        #: refusals a cheaper step answered, by the package scope they came in (`contract._Scopes`)
        self.answered: dict[str, int] = {}
        #: cross-step state its loops declared (`author.retained`)
        self.retained_bytes = 0
        #: Working allocation peak over the whole request, kept across cut-site resets. Weights
        #: are not in it, and no sum of it with them ever happened: the driver reads the card.
        self.torch_peak = 0
        self.overlap = False  # Historical measurement attribution, never a live policy gate.
        #: origin -> [bytes, fills, ns]: the store, a kept page (`baked`), another set's stored
        #: bytes on this GPU (`resident`) or in a kept page (`host`, a derive's own read ahead too),
        #: or a derived page on disk (`derived`)
        self.moved = {origin: [0, 0, 0] for origin in ("store", "baked", "resident", "host")}
        #: each baked fill's trace entry and its bake (`Bake`), resolved when the request ends
        self.bakes: list[tuple[FillEntry | _BakeTimes, Bake]] = []
        self.final_fills = self.final_hits = 0
        #: table lookups served on the host (`Memory._gather`)
        self.gathered = 0
        #: forwards of a layer refused room, run from its host copy in windows (`Memory._streamed`)
        self.streamed = 0
        self.prefetch_completed = 0
        self.ahead = _ahead(memory)
        self.allocator = _allocator(memory)
        self._trace: list[FillEntry] = []
        self.ended: RequestStats | None = None
        if memory.cuda and torch.cuda.is_initialized() and not memory.requests:
            # The working peak is process-wide since its reset: alone, a request starts its own,
            # or its floor is the largest decode the process ever ran (sieg: SDXL gave nothing)
            vt.reset_peak(memory.device.index or 0)
        self.sample()

    def first_read(self, block: _Block) -> bool:
        """Whether this is its first read of `block`."""
        known = id(block) in self.read_blocks
        self.read_blocks.add(id(block))
        return not known

    def sample(self) -> None:
        if self.memory.cuda:
            self.torch_peak = max(self.torch_peak, self.memory._reading()[2])

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

    def read(self, block: _Block, ns: int, nbytes: int, *, bake: Bake | None = None) -> None:
        """A block read here, outside the core: on the host, or where no core runs."""
        self.fill(block.model.name, block.index, nbytes, "store", False, ns, time.time() * 1000)
        if bake is not None:
            self.bakes.append((self._trace[-1], bake))

    def fill(
        self, name: str, block: int, nbytes: int, origin: str, ahead: bool, ns: int, at_ms: float
    ) -> None:
        """A block's bytes filled from one origin (`moved`). A demand fill's entry carries its
        read's host time and when it ended, so a step's fills and bakes are told apart (cr-242).
        `ahead`: the core filled it before any use asked."""
        moved = self.moved.setdefault(origin, [0, 0, 0])
        moved[0] += nbytes
        moved[1] += 1
        moved[2] += ns
        self.prefetch_completed += ahead
        self._trace.append(
            {
                "sequence": len(self._trace),
                "set": name,
                "block": block,
                "bytes": nbytes,
                "origin": origin,
                "speculative": ahead,
            }
        )
        if not ahead:
            self._trace[-1].update({"ns": ns, "at_unix_ms": round(at_ms, 1)})

    def record_fills(self) -> None:
        """The fills TensorFS logged since the last look."""
        for row in self.memory._tier().drain():
            self.fill(*row)
        self.sample()

    @property
    def trace(self) -> list[FillEntry]:
        # An idle request opens inside a running one and fills nothing: the fills are that one's.
        if self.memory.pages is not None and not self.idle:
            self.record_fills()
        return self._trace

    def _bakes(self) -> BakeTotals:
        """The bakes this request ran after a store read: each one's host and device time onto
        its trace entry, and their sums."""
        host = device = 0
        for entry, bake in self.bakes:
            on_device = bake.device_ns()
            entry["bake_ns"], entry["bake_device_ns"] = bake.host_ns, on_device
            host += bake.host_ns
            device += on_device or 0
        return {"blocks": len(self.bakes), "host_ns": host, "device_ns": device}

    def split_times(self) -> None:
        """Finished split calls' spans into their sites' `time`, oldest first, up to the first the
        GPU has not passed: it never waits on the GPU, and only calls in flight stay held."""
        spans = self.split_spans
        while spans and spans[0][3].query():
            key, level, start, stop, host_ns = spans.popleft()
            record = self.working_splits.get(key)
            if record is not None:
                times = record.setdefault("time", {})
                spent = times.setdefault(level, {"device_ms": 0.0, "host_ms": 0.0})
                spent["device_ms"] += float(start.elapsed_time(stop))
                spent["host_ms"] += host_ns / 1e6

    def stats(self) -> RequestStats:
        if self.ended is not None:
            return self.ended
        trace = self.trace
        self.sample()
        self.split_times()
        allocated, reserved = self.memory._figures()
        out: RequestStats = {
            "request": self.id,
            "allocator_scope": "process",
            "overlapping": self.overlap,
            "sets": {
                m.name: sum(b.nbytes for b in m.groups()) for m in self.memory.models.values()
            },
            "moved": {
                **{
                    origin: {"bytes": nbytes, "fills": fills, "ns": ns}
                    for origin, (nbytes, fills, ns) in self.moved.items()
                },
                "bake": self._bakes(),
            },
            "baked_pages": self.memory.pages.stats() if self.memory.pages is not None else None,
            "trace": list(trace),
            "trace_scope": "fills",
            "phases": list(self.phase_changes),
            "working_phases": list(self.working_phases),
            "working_splits": list(self.working_splits.values()),
            "escaped": list(self.escaped),
            "retries": self.retries,
            "hosted": dict(self.hosted),
            "first_blocks": list(self.first_blocks),
            "regrown": self.regrown,
            "swallowed_ooms": self.swallowed_ooms,
            "answered": dict(self.answered),
            "retained_bytes": self.retained_bytes,
            "gathered": self.gathered,
            "streamed": self.streamed,
            "host": None,
        }
        if self.memory.cuda and torch.cuda.is_initialized():
            groups = [b for model in self.memory.models.values() for b in model.groups()]
            out.update(
                {
                    "allocated": allocated,
                    "reserved": reserved,
                    "working_allocator": vt.stats(self.memory.device.index or 0),
                    "weight_residency": (
                        self.memory.native.stats() if self.memory.native is not None else None
                    ),
                    "final_bindings": {
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
                    },
                    # Cumulative native working counters sampled over this request.
                    "allocator": {
                        k: v - self.allocator.get(k, 0) for k, v in _allocator(self.memory).items()
                    },
                    "prefetch": {
                        "completed": self.prefetch_completed,
                        **{k: v - self.ahead.get(k, 0) for k, v in _ahead(self.memory).items()},
                        "scope": "the core's fills ahead of demand",
                    },
                }
            )
        elif self.memory.device.type == "mps":
            out.update(
                {"allocated": allocated, "reserved": reserved, "measurement": "observed_current"}
            )
        return out


def _allocator(memory: Memory) -> dict[str, int]:
    """Cumulative native working-allocator counters; live/backing gauges are separate."""
    if not memory.cuda or memory.native is None:
        return {}
    stats = cast("Mapping[str, int]", vt.stats(memory.device.index or 0))
    return {
        key: int(stats.get(key, 0))  # a TensorFS before the last resort has none of its counters
        for key in (
            "allocations", "frees", "ooms", "driver_memory_calls",
            "event_queries", "event_records", "event_waits", "context_joins",
            # refusals a cheaper step answered, and varena's last resort: host segments pinned,
            # allocations served there and their bytes, and those host memory refused too
            "answered_ooms", "host_working_allocs", "host_working_served",
            "host_working_served_bytes", "host_working_refusals",
        )
    }


def _ahead(memory: Memory) -> dict[str, int]:
    """Cumulative fills ahead of demand that came to nothing."""
    if not memory.cuda or memory.native is None or memory.pages is None:
        return {}
    stats = memory.pages.stats()
    return {
        "canceled": int(stats["ahead_canceled"]),
        "failed": int(stats["ahead_failed"]),
        "declined_native": int(vt.stats(memory.device.index or 0)["prefetch_declines"]),
    }


def _block_of(name: str, paths: Collection[str]) -> str:
    """The block holding `name` (itself, or its longest dotted prefix in `paths`), "" for common.
    `paths` is a set or a mapping: a lookup per prefix, never a scan of the blocks."""
    return name if name in paths else _unit(name, paths)


def _sites(module: nn.Module) -> dict[str, list[_Site]]:
    groups: dict[int, list[_Site]] = {}
    names: dict[str, int] = {}
    for path, sub in module.named_modules(remove_duplicate=False):
        # torch types `_parameters` with Parameter; the Runtime binds plain tensors there too
        slots = cast(dict[str, Tensor | None], sub._parameters), sub._buffers
        for table, parameter in ((slots[0], True), (slots[1], False)):
            for attr, tensor in table.items():
                if tensor is None:
                    continue
                names[f"{path}.{attr}" if path else attr] = id(tensor)
                sites = groups.setdefault(id(tensor), [])
                if not any(s.table is table and s.name == attr for s in sites):
                    sites.append(_Site(table, attr, parameter))
    return {name: groups[key] for name, key in names.items()}


def _view(key: str, role: str, part: _Part, raw: Tensor) -> Tensor:
    """A stored part's bytes `raw` as its tensor."""
    shape = part.shape
    if part.dtype == "F4":
        if not shape or shape[-1] % 2:
            raise ResidencyError("packed F4 weights require an even last axis")
        shape = (*shape[:-1], shape[-1] // 2)
    view = raw.view(getattr(torch, _DTYPES[part.dtype]))
    if view.numel() != math.prod(shape):
        raise ResidencyError(f"stored geometry does not match {key}#{role}")
    return view.view(shape)


def _data(block: _Block, raw: Tensor) -> dict[str, dict[str, Tensor]]:
    """A block's stored parts as tensors over its region's bytes `raw`."""
    data: dict[str, dict[str, Tensor]] = {}
    offset = 0
    for key, parts in block.parts.items():
        views = data[key] = {}
        for role, part in parts.items():
            views[role] = _view(key, role, part, raw[offset : offset + part.nbytes])
            offset += (part.nbytes + 255) // 256 * 256
    return data


def _finals(block: _Block, raw: Tensor) -> dict[str, dict[str, Tensor]]:
    """A block's final buffers as tensors over its final region's bytes `raw`."""
    data: dict[str, dict[str, Tensor]] = {}
    offset = 0
    for key, parts in block.finals.items():
        row = data[key] = {}
        for name, part in parts.items():
            flat = raw[offset : offset + part.nbytes].view(getattr(torch, _DTYPES[part.dtype]))
            row[name] = flat.as_strided(part.shape, part.strides)
            offset += (part.nbytes + 255) // 256 * 256
    return data


def _derive(
    owner: weakref.ReferenceType[_Block],
    field: str,
    write: Callable[[_Block, Tensor], bool],
    view: Tensor,
) -> bool:
    """A Source's computed bytes, on the thread that called `use`. A failed one may have exported
    storage to its code: its Region's identity retires, those exports keep their own pins and
    never see a later fill's bytes."""
    block = owner()
    if block is None:
        raise ResidencyError("a fill's block is gone")
    try:
        return write(block, view)
    except BaseException:
        block.model.memory._retire_phase_plan("weight_fill_failed", model=block.model)
        setattr(block, field, None)
        raise


def _bake(block: _Block, view: Tensor) -> bool:
    """A block's stored bytes through its model's fill: whether they changed."""
    memory, model = block.model.memory, block.model
    assert model.fill is not None
    baking = Bake.begin(memory.stream())
    changed = model.fill(block.path, _data(block, view))
    if changed and (request := _REQUEST.get()) is not None:
        request.bakes.append(({}, baking.end(memory.stream())))
    return changed


def _write_final(block: _Block, view: Tensor) -> bool:
    """A block's final buffers from its stored bytes, which its own Region serves."""
    memory, model = block.model.memory, block.model
    view.zero_()  # includes all alignment gaps and padded strided extents
    use = memory._use(block)
    parts = _data(block, use.tensor)
    for key, destinations in _finals(block, view).items():
        if model.finals is not None:
            model.finals.write(key, parts[key], destinations)
        else:
            for destination in destinations.values():
                destination.copy_(parts[key]["value"])
    block.final_fills += 1
    if (request := _REQUEST.get()) is not None:
        request.final_fills += 1
    return True


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
    #: each alias's final geometry, should the block hold all its targets as finals
    whole: dict[str, dict[str, _Part]] = {}
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
            assert target is not None  # a site is a slot that held a tensor
            code = codes.get(str(target.dtype).removeprefix("torch."))
            if code is None:
                return
            declared = {
                name: TensorLayout(code, tuple(target.shape), tuple(target.stride()), "value")
            }
        row_aliases = aliases[key] = {}
        row_finals = finals[key] = {}
        row_whole = whole[key] = {}
        for name, layout in declared.items():
            if name not in model.sites:
                raise ResidencyError(f"{model.name}: final layout names no target {name}")
            shape = layout.shape
            strides = layout.strides or _strides(shape)
            target = model.sites[name][0].table[model.sites[name][0].name]
            assert target is not None  # a site is a slot that held a tensor
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
            if source is not None and source.dtype == "F4":
                # Packed two a byte: a stored F4 [..., C] is the U8 [..., C/2] its leaves read
                # (`_view`), not a second copy of it (cr-319).
                carrier = (*source.shape[:-1], source.shape[-1] // 2)
                source = replace(source, dtype="U8", shape=carrier)
            if (
                source is not None
                and source.dtype == layout.dtype
                and source.shape == shape
                and strides == _strides(source.shape)
                and source.nbytes == count * width
            ):
                assert layout.source is not None
                row_aliases[name] = layout.source
                row_whole[name] = _Part(layout.dtype, shape, count * width, "", strides)
            else:
                row_finals[name] = _Part(layout.dtype, shape, count * width, "", strides)
    if any(finals.values()) and any(aliases.values()):
        # A block with a final holds every target there and none of its stored bytes beside it:
        # aliased, its stored region stayed resident whole, its swizzled or cast parts' stored
        # bytes too (cr-319: Qwen3.6-27B nvfp4's row-major scales, 1.0 GiB, beside their blocked
        # finals). Its finals are copied from the stored bytes once a fill, or kept on the host.
        for key, row in whole.items():
            finals[key].update(row)
            aliases[key] = {}
    block.aliases = aliases
    block.finals = {key: row for key, row in finals.items() if row}
    if model.fill is None and not block.finals:
        offset = 0
        for key, parts in block.parts.items():
            for role, part in parts.items():
                for name, aliased in aliases[key].items():
                    if aliased == role:
                        for site in model.sites[name]:
                            block.raw_views[(id(site.table), site.name)] = (offset, part)
                offset += (part.nbytes + 255) // 256 * 256


def _placeholder(
    model: Model,
    block: _Block,
    sites: list[_Site],
    dtype: torch.dtype,
    raw_views: Mapping[tuple[int, str], tuple[int, _Part]],
) -> Tensor:
    meta = sites[0].table[sites[0].name]
    assert meta is not None  # a site is a slot that held a tensor
    held = Placeholder(meta.shape, dtype, model.memory.device)
    if sites[0].parameter:
        held = cast(
            Placeholder, nn.Parameter(held, requires_grad=False)
        )  # a subclass's Parameter is one

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
    #: each stored key's group, once: a lookup per site, where a scan of every group was 10,240
    #: of them per site on Qwen3.6 (cr-307)
    stored = {key: block for block in reversed(model.groups()) for key in block.parts}
    for name, sites in model.sites.items():
        meta = sites[0].table[sites[0].name]
        assert meta is not None  # a site is a slot that held a tensor
        if id(meta) in done or meta.device.type != "meta":
            continue
        done.add(id(meta))
        block = stored.get(name)
        if block is None:
            path = _block_of(name, model.blocks)
            block = model.blocks.get(path) or model.common
        if block is None:
            raise ResidencyError(f"{model.name}: no stored group for {name}")
        held = _placeholder(model, block, sites, meta.dtype, block.raw_views)
        for site in sites:
            site.table[site.name] = held


def _page(block: _Block, identity: object, overlay: Hashable, *, final: bool = False) -> str:
    """A block's host-tier page key: a digest of its component by content, path, stored layout,
    fill identity and overlay, and `final` for its cast bytes. The same bytes have one key in every
    process, whichever release of a package loaded them."""
    layout = _stored_layout(block)
    return _digest((block.model.content, block.path, layout, identity, overlay, final))


def _slot(block: _Block, identity: object, overlay: Hashable) -> str | None:
    """`block` under its fill identity whatever its overlay: what its derived pages are kept on disk
    by, when all that names them names it in every process. A factor no loaded model holds is
    named by an object of this process (`Filling.overlay`): its pages never leave it."""
    if not (_portable(identity) and _portable(overlay)):
        return None
    return _digest((block.model.content, block.path, _stored_layout(block), identity))


def _stored_layout(block: _Block) -> tuple[tuple[str, str, int], ...]:
    """Where `block`'s stored parts lie in its page."""
    parts = block.parts.items()
    return tuple((key, part.role, part.nbytes) for key, roles in parts for part in roles.values())


def _digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


def _portable(value: object) -> bool:
    """Whether `value`'s repr names it the same in every process: plain values and their tuples."""
    if isinstance(value, tuple):
        return all(_portable(item) for item in value)
    return value is None or isinstance(value, (str, int, float, bool))


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


@torch.inference_mode(False)
def _assign(block: _Block) -> list[tuple[_Site, Tensor]]:
    """`block`'s values for its sites, made outside inference mode whatever its caller's (a
    forward's hook, a read in a package's `inference_mode`): a CPU's widened copy made inside it
    was an inference tensor, and its Parameter failed `Cannot set version_counter for inference
    tensor` once a CPU model's blocks entered per forward (CPU e2e 44407890827)."""
    model = block.model
    if block.data is None:
        raise ResidencyError("weight assignment before a complete fill")
    rows: dict[str, Mapping[str, Tensor]]
    if block.aliases is not None:
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


def _gib(n: int) -> str:
    return f"{n / (1 << 30):.1f} GiB"


def _does_not_fit(component: str, refused: Verdict) -> str:
    """One line: the op, what it needs and of what, the card's room, and what to do."""
    held, kept = refused["held_bytes"], refused["escaped_bytes"]
    sites = ", ".join(site for site, _ in refused["sites"])
    born = f" in {sites}" if sites else ""
    if refused["learned"]:
        when = "its first forward, as its last run of this size went,"
        parts = f"{_gib(held)} live, {_gib(kept)} its steps keep{born}"
    else:
        when = "its next forward"
        parts = f"{_gib(held)} live" + (f" ({_gib(kept)} of it kept by earlier steps{born})" if kept else "")
    return (
        f"{component}: {when} needs {_gib(refused['need_bytes'])} with everything evictable gone "
        f"({parts}, {_gib(refused['transient_bytes'])} working, {_gib(refused['floor_bytes'])} of "
        f"weights), and this card has {_gib(refused['room_bytes'])}: run it smaller, or rent a "
        "larger card: `cozy run … --on new`"
    )


def _escaped_warnings(records: list[EscapeRecord]) -> None:
    """One line per loop whose passes kept a significant amount as plain tensors."""
    for row in records:
        if row["escaped_peak_bytes"] < _NOTED:
            continue
        sites = ", ".join(site for site, _ in row["escaped_sites"]) or row["head"]
        said = (
            f"{row['head']} kept {_gib(row['escaped_peak_bytes'])} across its steps outside the "
            f"memory manager, born in {sites}: hold it in author.retained"
        )
        WARNINGS.append((said, {"code": "escaped_memory"}))


def uncached(module: nn.Module) -> None:
    """A forward that does not ask for a cache keeps none. A text encoder's call through a library
    (Diffusers' `encode_prompt`) left its `DynamicCache` live as plain tensors, about 1 GiB for a
    long prompt; decoding asks with `use_cache=True` itself (`cozy_runtime.lm`, `generate`)."""
    for sub in module.modules():
        config = getattr(sub, "config", None)
        transformers = any(c.__module__.startswith("transformers.") for c in type(config).__mro__)
        if config is not None and transformers and getattr(config, "use_cache", None):
            config.use_cache = False


def _hooks(model: Model) -> None:
    memory = model.memory
    tally = memory.cuda
    if tally:
        model.escape_site = vt.site_id(model.component)
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

        model.handles.extend(_paired(sub, opened, lambda: memory._close(model, executing=True), model))
    for block in model.blocks.values():
        sub = model.module.get_submodule(block.path)
        if tally:  # its index starred: one site per stack
            block.escape_site = vt.site_id(f"{model.component}.{_INDEX.sub('*', block.path)}")

        def entered(*_: object, b: _Block = block) -> None:
            pacing.tick()
            model._claim()
            try:
                memory._enter(b)
            except BaseException:
                model._release()
                raise

        def exited(b: _Block = block) -> None:
            try:
                memory._exit(b)
            finally:
                model._release()

        model.handles.extend(_paired(sub, entered, exited, model))
        if block.table:
            model.handles.extend(_gathering(sub, memory))
        elif memory.cuda and _linear(sub):
            block.leaf = sub
            _streaming(sub, block, memory)


def _linear(module: nn.Module) -> bool:
    """A layer with no children whose kind may compute by output rows (`rows.streamed`)."""
    if next(module.children(), None) is not None:
        return False
    return type(module) is nn.Linear or getattr(module, "provider", None) is not None


def _streaming(module: nn.Module, block: _Block, memory: Memory) -> None:
    """`module`'s forward, from its host copy while it has no room on the card (`_streamed`)."""
    previous = vars(module).get("forward") or (
        lambda *args, **kwargs: type(module).forward(module, *args, **kwargs)
    )

    def forward(x: Tensor, *args: object, **kwargs: object) -> object:
        if block.gathered is None or args or kwargs or not isinstance(x, Tensor):
            return previous(x, *args, **kwargs)
        return memory._streamed(block, x)

    vars(forward)[TRANSPARENT] = True
    module.forward = forward


def _gathering(module: nn.Module, memory: Memory) -> list[RemovableHandle]:
    """A gathered table's forward runs on the host (`Memory._gather`): its inputs go there, and
    only the rows it read come back."""

    on_host = threading.local()

    def host(value: object) -> object:
        return value.cpu() if isinstance(value, Tensor) else value

    def device(value: object) -> object:
        return value.to(memory.device) if isinstance(value, Tensor) else value

    def pre(
        sub: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> tuple[tuple[object, ...], dict[str, object]] | None:
        weight = getattr(sub, "weight", None)
        on_host.now = memory.cuda and isinstance(weight, Tensor) and not weight.is_cuda
        if not on_host.now:
            return None
        if memory.capturing():
            raise ResidencyError("all graph weights must be resident before capture")
        return tree_map(host, args), tree_map(host, kwargs)

    def back(_: nn.Module, args: object, output: object) -> object:
        if not getattr(on_host, "now", False):
            return None
        on_host.now = False
        return tree_map(device, output)

    return [
        module.register_forward_pre_hook(pre, with_kwargs=True),
        module.register_forward_hook(back),
    ]


def _paired(
    module: nn.Module,
    enter: Callable[[tuple[object, ...], dict[str, object]], None],
    leave: Callable[[], None],
    owner: Model,
) -> list[RemovableHandle]:
    from cozy_runtime.accel import rows

    owed = threading.local()

    def pre(_: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        enter(args, kwargs)
        owed.count = getattr(owed, "count", 0) + 1

    def post(*_: object) -> None:
        if getattr(owed, "count", 0):
            owed.count -= 1
            leave()

    # The block's entry, not its arithmetic: a layer's row cut looks past them (`rows.streamed`).
    vars(pre)[rows.OWN] = vars(post)[rows.OWN] = True

    return [
        module.register_forward_pre_hook(pre, with_kwargs=True),
        module.register_forward_hook(post, always_call=True),
    ]
