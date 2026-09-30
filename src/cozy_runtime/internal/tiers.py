"""Weight units and the moves between device, host and disk. Mechanism only: `memory` decides.

A component splits into units: the innermost repeated instances of its `_no_split_modules`
classes are blocks, everything else is the common unit. Moves are synchronous on the current
stream. A free is a dropped reference, so the caching allocator reuses the bytes in stream
order and a view held elsewhere keeps its storage alive. `trim` is the one place cached device
memory goes back to the driver, after the whole device is idle.
"""

from __future__ import annotations

import functools
import weakref
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, NoReturn, Protocol

from cozy_runtime.author._errors import RuntimeFailure
from cozy_runtime.author._retry import is_declared_pure, original_callable, runtime_owned

DEVICE, HOST, DISK = "device", "host", "disk"

#: torch is not importable where the Worker runs, so its objects are named rather than typed
Torch = Any
Tensor = Any
Module = Any
Device = Any
Handle = Any
Convert = Callable[[Tensor], Tensor]
#: an op runner: (method, the bound method, args, kwargs) -> its result
Run = Callable[[str, Callable[..., object], tuple[object, ...], dict[str, object]], object]
OP_METHODS = ("forward", "encode", "decode")


class Store(Protocol):
    """The disk tier: the fill plane's validated TensorFS path."""

    def load(self, name: str) -> object: ...

    def load_unit(self, name: str, unit: Unit) -> None: ...

    def drop(self, name: str) -> None: ...

    def drop_unit(self, unit: Unit) -> None: ...

    def unit_loads(self, name: str, keys: frozenset[str]) -> bool: ...

    def load_schema(self, name: str, unit: Unit) -> RefillSchema:
        """Canonical slot dtypes and derived-buffer copy overlap, metadata only."""
        ...

    def transient(self, name: str) -> int:
        """Bytes a whole fill of `name` holds beyond the component while it arrives."""
        ...


@dataclass(frozen=True)
class DTypeConvert:
    """A canonical Torch dtype conversion, after Runtime removes device-move intent."""

    apply: Convert

    def __call__(self, tensor: Tensor) -> Tensor:
        return self.apply(tensor)


@dataclass(frozen=True)
class RefillSchema:
    """No payloads: None dtypes marks opaque encoded topology, not missing metadata."""

    dtypes: tuple[object, ...] | None
    restoration_peak_extra: int = 0


@dataclass(frozen=True)
class LoadDemand:
    resident: int
    peak: int
    #: Initial canonical fill allocation, before replay; parked metadata can be wider.
    source: int | None = field(default=None, compare=False)
    restoration_peak_extra: int = field(default=0, compare=False)
    replay_peak: int | None = field(default=None, compare=False)


def _canonical_dtype(torch: Torch, fn: Convert) -> bool:
    code = getattr(fn, "__code__", None)
    if code is None:
        return False
    methods = ("to", "half", "float", "double", "bfloat16", "type", "cpu", "cuda")
    known = any(
        code is constant
        for name in methods
        for constant in getattr(
            getattr(getattr(torch.nn.Module, name), "__code__", None), "co_consts", ()
        )
    )
    # Memory-format transforms have shape/stride allocation semantics beyond dtype.
    closure = dict(
        zip(getattr(code, "co_freevars", ()), getattr(fn, "__closure__", ()) or (), strict=True)
    )
    layout = closure.get("convert_to_format")
    return known and (layout is None or layout.cell_contents is None)


def load_demand(torch: Torch, unit: Unit, *, component: Component | None = None) -> LoadDemand:
    """Current fill backing, settled dtype bytes and conservative conversion overlap.

    Only the installed Torch's canonical dtype closures establish this bound. Calling
    an arbitrary transform on a scalar does not prove its real allocation demand.
    """
    if unit.tier != DISK:
        return LoadDemand(unit.nbytes, unit.nbytes)
    if not unit.pending and (component is None or component.store is None):
        return LoadDemand(unit.nbytes, unit.nbytes)
    if not all(isinstance(convert, DTypeConvert) for convert in unit.pending):
        raise RuntimeFailure("unbounded stored conversion", code="memory_capability")
    slots = list(unit.slots())
    if unit.pending and any(
        type(module)._apply is not torch.nn.Module._apply for _, module in unit.owners()
    ):
        raise RuntimeFailure(
            "custom owner conversion has no allocation bound", code="memory_capability"
        )
    if unit.pending and any(
        type(tensor) not in (torch.Tensor, torch.nn.Parameter)
        or tensor.layout != torch.strided
        or tensor.grad is not None
        for *_, tensor in slots
    ):
        raise RuntimeFailure(
            "stored conversion requires plain gradient-free tensors", code="memory_capability"
        )
    schema = (
        component.store.load_schema(component.name, unit)
        if component is not None and component.store is not None
        else RefillSchema(tuple(tensor.dtype for *_, tensor in slots))
    )
    if schema.dtypes is None:
        if unit.pending:
            raise RuntimeFailure(
                "stored conversion of encoded leaves is unsupported", code="memory_capability"
            )
        return LoadDemand(
            unit.nbytes,
            unit.nbytes + schema.restoration_peak_extra,
            unit.nbytes,
            schema.restoration_peak_extra,
            unit.nbytes,
        )
    dtypes = list(schema.dtypes)
    if len(dtypes) != len(slots):
        raise RuntimeFailure("canonical refill slot count changed", code="memory_capability")
    sizes = [
        tensor.numel() * torch.empty(0, dtype=dtype, device="meta").element_size()
        for (*_, tensor), dtype in zip(slots, dtypes, strict=True)
    ]
    source = sum(sizes)
    replay_peak = source
    converted = False
    for convert in unit.pending:
        changed = [convert(torch.empty(0, dtype=dtype, device="meta")).dtype for dtype in dtypes]
        next_sizes = [
            tensor.numel() * torch.empty(0, dtype=dtype, device="meta").element_size()
            for (*_, tensor), dtype in zip(slots, changed, strict=True)
        ]
        if changed != dtypes:
            converted = True
            seen: set[int] = set()
            for *_, tensor in slots:
                storage = tensor.untyped_storage()
                identity = int(storage._cdata)
                if identity in seen or storage.nbytes() != tensor.numel() * tensor.element_size():
                    raise RuntimeFailure(
                        "stored conversion cannot split aliased backing", code="memory_capability"
                    )
                seen.add(identity)
        allocated = sum(
            size
            for size, before, after in zip(next_sizes, dtypes, changed, strict=True)
            if before != after
        )
        replay_peak = max(replay_peak, sum(sizes) + allocated)
        sizes, dtypes = next_sizes, changed
    peak = max(source + schema.restoration_peak_extra, replay_peak)
    if converted or (component is not None and component.store is not None):
        return LoadDemand(sum(sizes), peak, source, schema.restoration_peak_extra, replay_peak)
    # Standalone quotes without a store must retain full unique backing size.
    return LoadDemand(unit.nbytes, max(peak, unit.nbytes), max(source, unit.nbytes))


def load_sequence(
    torch: Torch,
    units: tuple[Unit, ...],
    *,
    whole: bool = False,
    component: Component | None = None,
) -> LoadDemand:
    """New capacity for sequential unit loads, or one whole fill before replaying casts."""
    missing = [unit for unit in units if unit.tier != DEVICE]
    demands = [load_demand(torch, unit, component=component) for unit in missing]
    if whole:
        sources = [
            unit.nbytes if demand.source is None else demand.source
            for unit, demand in zip(missing, demands, strict=True)
        ]
        current = sum(sources)
        peak = current + max((d.restoration_peak_extra for d in demands), default=0)
        for source, demand in zip(sources, demands, strict=True):
            peak = max(
                peak,
                current
                - source
                + (demand.peak if demand.replay_peak is None else demand.replay_peak),
            )
            current += demand.resident - source
        return LoadDemand(current, peak)
    current = peak = 0
    for demand in demands:
        peak = max(peak, current + demand.peak)
        current += demand.resident
    return LoadDemand(current, peak)


@dataclass(eq=False)
class Unit:
    path: str
    module: Module
    #: block roots the common unit skips; empty for a block
    skip: tuple[Module, ...] = ()
    nbytes: int = 0
    tier: str = DEVICE
    #: the host copy of each slot, kept while the device copy is the one it was copied to:
    #: a clean cache, so leaving the device again only rebinds
    host: dict[tuple[int, str], Tensor] = field(default_factory=dict)
    seen: dict[tuple[int, str], int] = field(default_factory=dict)
    #: loaded by a forward hook for one forward, and sent home after it
    streamed: bool = False
    #: Explicit mutable tensor state cannot be reconstructed from checkpoint bytes.
    mutable: bool = False
    #: Checkpoint bytes start clean; only an explicit operation contract keeps them so.
    readonly: bool = True
    #: package conversions applied since the unit's last fill, and those waiting for its next
    applied: list[Convert] = field(default_factory=list)
    pending: list[Convert] = field(default_factory=list)
    #: Exact whole owners for a nonstreamed common partition; never individual slots.
    members: tuple[tuple[str, Module], ...] | None = None

    def owners(self) -> list[tuple[str, Module]]:
        """Every (path, module) whose own tensors belong to this unit."""
        if self.members is not None:
            return list(self.members)
        skip = {id(module) for module in self.skip}
        found: list[tuple[str, Module]] = []
        stack = [(self.path, self.module)]
        while stack:
            path, module = stack.pop()
            found.append((path, module))
            for name, child in reversed(module._modules.items()):
                if child is not None and id(child) not in skip:
                    stack.append((f"{path}.{name}" if path else name, child))
        return found

    def slots(self) -> Iterator[tuple[Module, str, bool, Tensor]]:
        for _, module in self.owners():
            for name, tensor in list(module._parameters.items()):
                if tensor is not None:
                    yield module, name, True, tensor
            for name, tensor in list(module._buffers.items()):
                if tensor is not None:
                    yield module, name, False, tensor

    def live(self, component: str) -> dict[str, Tensor]:
        """The persistent destinations the fill plane writes, by component-qualified key."""
        found: dict[str, Tensor] = {}
        for path, module in self.owners():
            prefix = f"{component}.{path}." if path else f"{component}."
            for name, tensor in module._parameters.items():
                if tensor is not None:
                    found[prefix + name] = tensor.detach()
            for name, tensor in module._buffers.items():
                if tensor is not None and name not in module._non_persistent_buffers_set:
                    found[prefix + name] = tensor.detach()
        return found


@dataclass(eq=False)
class Component:
    name: str
    root: Module
    common: Unit
    blocks: tuple[Unit, ...]
    store: Store | None = None
    #: the store can fill each unit alone; otherwise it refills the component whole
    unit_loads: bool = False
    # Changes only when this component actually drops backing to the store.
    disk_epoch: int = 0
    hooks: list[Handle] = field(default_factory=list)
    #: a move of ours is converting tensors: a package conversion is not intercepted
    moving: bool = False
    #: Real first-parameter owner plus alias closure; None keeps legacy common pins.
    anchor: Unit | None = None
    anchor_structure: tuple[object, ...] = ()

    @property
    def common_units(self) -> tuple[Unit, ...]:
        if self.anchor is None or self.anchor is self.common:
            return (self.common,)
        return (self.anchor, self.common)

    @property
    def units(self) -> tuple[Unit, ...]:
        return (*self.common_units, *self.blocks)

    @property
    def nbytes(self) -> int:
        return sum(unit.nbytes for unit in self.units)

    def on(self, tier: str) -> int:
        return sum(unit.nbytes for unit in self.units if unit.tier == tier)


def _backing(tensor: Tensor) -> tuple[str, int]:
    storage = tensor.untyped_storage()
    identity = int(storage._cdata) if tensor.is_meta else storage.data_ptr()
    return str(storage.device), identity if storage.nbytes() else id(tensor)


def _bytes(tensors: Iterator[Tensor]) -> int:
    sizes: dict[tuple[str, int], int] = {
        _backing(tensor): int(tensor.untyped_storage().nbytes()) for tensor in tensors
    }
    return sum(sizes.values())


def _tokens(module: Module) -> Iterator[tuple[str, int]]:
    yield "module", id(module)
    for tensor in (*module._parameters.values(), *module._buffers.values()):
        if tensor is not None:
            yield "tensor", id(tensor)
            if tensor.numel():
                device, identity = _backing(tensor)
                yield f"storage:{device}", identity


def partition(name: str, root: Module, tier: str, store: Store | None = None) -> Component:
    """Blocks are the innermost REPEATED instances of the declared no-split classes. The unit
    holding the root's first parameter stays common, so `.device` stays true; units that share
    a module or a storage join the common unit."""
    classes = set(getattr(root, "_no_split_modules", None) or ())
    found: list[tuple[str, Module]] = []

    def walk(path: str, module: Module) -> bool:
        inner = False
        for key, child in module._modules.items():
            if child is not None:
                inner = walk(f"{path}.{key}" if path else key, child) or inner
        mine = bool(path) and type(module).__name__ in classes
        if mine and not inner:
            found.append((path, module))
        return inner or mine

    walk("", root)
    repeated = Counter(type(module).__name__ for _, module in found)
    first = next(
        (
            path
            for path, module in root.named_modules()
            if any(p is not None for p in module._parameters.values())
        ),
        "",
    )
    blocks = [
        Unit(path, module, tier=tier)
        for path, module in found
        if repeated[type(module).__name__] > 1
        and first != path
        and not first.startswith(path + ".")
    ]
    common = Unit("", root, skip=tuple(unit.module for unit in blocks), tier=tier)
    owner: dict[tuple[str, int], int] = {}
    shared: set[int] = set()
    for index, unit in enumerate((common, *blocks)):
        for _, module in unit.owners():
            for token in _tokens(module):
                if owner.setdefault(token, index) != index:
                    shared.update((owner[token], index))
    if shared:
        blocks = [unit for index, unit in enumerate(blocks, 1) if index not in shared]
        common = Unit("", root, skip=tuple(unit.module for unit in blocks), tier=tier)
    component = Component(name, root, common, tuple(blocks), store)
    measure(component)
    keys = frozenset(key for unit in component.units for key in unit.live(name))
    component.unit_loads = store is not None and store.unit_loads(name, keys)
    return component


def _structure(root: Module) -> tuple[object, ...]:
    """Registered owner/slot geometry, independent of legal backing moves and casts."""
    return tuple(
        (
            path,
            id(module),
            tuple((name, id(child)) for name, child in module._modules.items()),
            tuple(
                (
                    parameter,
                    name,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    name in module._non_persistent_buffers_set,
                )
                for parameter, slots in ((True, module._parameters), (False, module._buffers))
                for name, tensor in slots.items()
                if tensor is not None
            ),
        )
        for path, module in Unit("", root).owners()
    )


def split_common(component: Component) -> None:
    """Partition complete common owners; admission/eligibility belongs to Planner."""
    common = component.common
    owners = common.owners()
    first = next(
        (
            module
            for _, module in Unit("", component.root).owners()
            if any(tensor is not None for tensor in module._parameters.values())
        ),
        None,
    )
    if first is None or not any(module is first for _, module in owners):
        return
    selected = {id(first)}
    while True:
        tokens = {
            token for _, module in owners if id(module) in selected for token in _tokens(module)
        }
        expanded = selected | {
            id(module) for _, module in owners if tokens.intersection(_tokens(module))
        }
        if expanded == selected:
            break
        selected = expanded
    anchored = tuple((path, module) for path, module in owners if id(module) in selected)
    remainder = tuple((path, module) for path, module in owners if id(module) not in selected)
    if not remainder:
        component.anchor = common
    else:

        def unit(members: tuple[tuple[str, Module], ...]) -> Unit:
            identifiers = {id(module) for _, module in members}
            return Unit(
                "",
                component.root,
                tier=common.tier,
                members=members,
                host={key: value for key, value in common.host.items() if key[0] in identifiers},
                seen={key: value for key, value in common.seen.items() if key[0] in identifiers},
                mutable=common.mutable,
                readonly=common.readonly,
                applied=list(common.applied),
                pending=list(common.pending),
            )

        component.anchor, component.common = unit(anchored), unit(remainder)
    component.anchor_structure = _structure(component.root)
    measure(component)
    validate_anchor(component)


def validate_anchor(component: Component) -> None:
    """A split immutable construction cannot silently acquire unaccounted owners/aliases."""
    if component.anchor is None:
        return
    if _structure(component.root) != component.anchor_structure:
        raise RuntimeFailure(
            "registered component owner/slot geometry changed after common partition",
            code="component_structure_changed",
        )
    owners: dict[tuple[str, int], Unit] = {}
    for unit in component.units:
        for _, module in unit.owners():
            for token in _tokens(module):
                if owners.setdefault(token, unit) is not unit:
                    raise RuntimeFailure(
                        "registered component acquired an alias across residency units",
                        code="component_structure_changed",
                    )


def measure(component: Component) -> None:
    for unit in component.units:
        unit.nbytes = _bytes(tensor for *_, tensor in unit.slots())


def _bind(module: Module, name: str, parameter: bool, tensor: Tensor) -> None:
    if parameter:
        module._parameters[name].data = tensor
    else:
        module._buffers[name] = tensor


def to_device(torch: Torch, unit: Unit, device: Device) -> int:
    """Copy each host backing once, preserving tied tensors and strided aliases."""
    return _move(torch, unit, device)


def _host_cache(unit: Unit, module: Module, name: str, tensor: Tensor) -> Tensor | None:
    key = (id(module), name)
    host = unit.host.get(key)
    return (
        host
        if (
            host is not None
            and unit.readonly
            and unit.seen.get(key) == tensor.data_ptr()
            and host.shape == tensor.shape
            and host.dtype == tensor.dtype
            and host.stride() == tensor.stride()
            and host.storage_offset() == tensor.storage_offset()
            and host.untyped_storage().nbytes() == tensor.untyped_storage().nbytes()
        )
        else None
    )


def _moves(unit: Unit, target: str) -> list[list[tuple[Module, str, bool, Tensor]]]:
    groups: dict[tuple[str, int], list[tuple[Module, str, bool, Tensor]]] = {}
    for slot in unit.slots():
        tensor = slot[3]
        if tensor.is_meta:
            raise LookupError(f"{unit.path or '<common>'} is on disk")
        if tensor.device.type != target:
            groups.setdefault(_backing(tensor), []).append(slot)
    return list(groups.values())


def _cached(unit: Unit, slots: list[tuple[Module, str, bool, Tensor]]) -> bool:
    hosts = [_host_cache(unit, module, name, tensor) for module, name, _, tensor in slots]
    return all(host is not None for host in hosts) and len({_backing(h) for h in hosts}) == 1


def host_copy_bytes(unit: Unit) -> int:
    """Fresh unique backing required by to_host; partial alias caches cannot be reused."""
    return sum(
        slots[0][3].untyped_storage().nbytes()
        for slots in _moves(unit, "cpu")
        if not _cached(unit, slots)
    )


def _copy_backing(torch: Torch, tensor: Tensor, device: Device) -> object:
    storage = tensor.untyped_storage()
    view = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )
    return view.to(device=device, copy=True).untyped_storage()


def _move(torch: Torch, unit: Unit, device: Device) -> int:
    """Prepare all views before rebinding anything; failed copies leave placement intact."""
    groups = _moves(unit, device.type)
    moves: list[tuple[Module, str, bool, Tensor, Tensor]] = []
    copied = 0
    backing = placed = None
    try:
        with torch.inference_mode(False), torch.no_grad():
            for slots in groups:
                cached = device.type == "cpu" and _cached(unit, slots)
                if not cached:
                    backing = _copy_backing(torch, slots[0][3], device)
                    copied += slots[0][3].untyped_storage().nbytes()
                for module, name, parameter, tensor in slots:
                    placed = (
                        unit.host[(id(module), name)]
                        if cached
                        else torch.empty(0, dtype=tensor.dtype, device=device).set_(
                            backing, tensor.storage_offset(), tensor.shape, tensor.stride()
                        )
                    )
                    moves.append((module, name, parameter, tensor.detach(), placed))
    except BaseException:
        moves.clear()
        backing = placed = None
        raise
    for module, name, parameter, original, placed in moves:
        if device.type != "cpu":
            unit.host[(id(module), name)] = original
            unit.seen[(id(module), name)] = placed.data_ptr()
        _bind(module, name, parameter, placed)
    if device.type == "cpu":
        unit.host.clear()
        unit.seen.clear()
    unit.tier = HOST if device.type == "cpu" else DEVICE
    return copied


class HostStaging(Protocol):
    """Exclusive, already-admitted pinned backing; release returns its borrow, not RSS."""

    buffers: tuple[Tensor, ...]
    capacity: int

    def release(self) -> None: ...


def _transfer_version(tensor: Tensor) -> int | None:
    try:
        return int(tensor._version)
    except RuntimeError:  # Inference tensors intentionally have no version counter.
        return None


def _transfer_snapshot(unit: Unit) -> tuple[object, ...]:
    return (
        unit.tier,
        unit.readonly,
        unit.mutable,
        tuple(map(id, unit.applied)),
        tuple(map(id, unit.pending)),
        tuple(
            (
                id(module),
                name,
                parameter,
                id(tensor),
                _backing(tensor),
                int(tensor.untyped_storage().nbytes()),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.dtype,
                tensor.device,
                tensor.requires_grad,
                _transfer_version(tensor),
            )
            for module, name, parameter, tensor in unit.slots()
        ),
    )


class HostTransfer:
    """One event-owned HOST copy; eligibility and pool policy belong to the Planner.

    The caller supplies an executor-lifetime custody set and an already-admitted
    exclusive pinned lease. Failed drain retains this object in that set. Publication
    joins the copy on CPU; final consumer retirement conservatively drains the device,
    because purity does not imply exclusive current-stream use. Ordinary to_device
    remains synchronous; the optional Planner experiment uses this separate custody.
    """

    def __init__(
        self,
        torch: Torch,
        unit: Unit,
        device: Device,
        stream: Any,
        valid: Callable[[], bool],
        owned: set[HostTransfer],
    ) -> None:
        self.torch, self.unit, self.device, self.stream = torch, unit, device, stream
        self.valid, self.owned = valid, owned
        self.snapshot = _transfer_snapshot(unit)
        self.groups = _moves(unit, "cuda")
        self.source_bytes = sum(
            int(group[0][3].untyped_storage().nbytes()) for group in self.groups
        )
        self.originals = [(slot, slot[3].detach()) for group in self.groups for slot in group]
        self.destinations: list[Tensor] = []
        self.staging: HostStaging | None = None
        self._buffers: tuple[Tensor, ...] = ()
        self.copy_ready: Any = None
        self.consumer_done: Any = None
        self.started = False
        self.staged_bytes = self.submitted_bytes = 0
        self.state = "new"
        self.poisoned = ""

    @classmethod
    def prepare(
        cls,
        torch: Torch,
        unit: Unit,
        device: Device,
        stream: Any,
        *,
        acquire: Callable[[int], HostStaging | None],
        device_capacity: int,
        valid: Callable[[], bool],
        owned: set[HostTransfer],
        observed: Callable[[HostTransfer], None] | None = None,
    ) -> HostTransfer | None:
        """Prepare unbound destinations or decline; never execute authored Python.

        Capacity is supplied by admission. Payload bytes are not a claim about allocator
        rounding or reusable cache; the caller's allocator cap remains authoritative.
        The caller must hold exclusive immutable source custody through publication and
        revoke ``valid`` before any writer can intervene. Geometry/version checks detect
        observable changes only: inference tensors lack versions and .data can bypass them.
        """
        if (
            device.type != "cuda"
            or device.index is None
            or getattr(stream, "device", None) != device
            or torch.is_grad_enabled()
            or unit.tier != HOST
            or not unit.readonly
            or unit.mutable
            or unit.pending
            or not valid()
            or any(
                type(tensor) not in (torch.Tensor, torch.nn.Parameter)
                or tensor.device.type != "cpu"
                or tensor.layout != torch.strided
                or tensor.is_quantized
                or tensor.grad is not None
                for *_, tensor in unit.slots()
            )
        ):
            return None
        result = cls(torch, unit, device, stream, valid, owned)
        if result.source_bytes > device_capacity:
            return None
        owned.add(result)  # Custody exists before any asynchronous operation can fail.
        try:
            if not result._prepare(acquire):
                result.abort()
                return None
        except BaseException as exc:
            if result.state != "poisoned":
                try:
                    result.abort()
                except BaseException as cleanup:
                    raise cleanup from exc
            raise
        finally:
            if observed is not None:
                observed(result)
        return result

    def _current(self) -> bool:
        return self.valid() and _transfer_snapshot(self.unit) == self.snapshot

    def _prepare(self, acquire: Callable[[int], HostStaging | None]) -> bool:
        torch = self.torch
        chunks: list[tuple[int, int, Tensor]] = []
        if self.source_bytes:
            self.staging = acquire(self.source_bytes)
            if self.staging is None:
                return False
            buffers = self._buffers = tuple(self.staging.buffers)
            if (
                self.staging.capacity != sum(int(b.untyped_storage().nbytes()) for b in buffers)
                or self.staging.capacity < self.source_bytes
                or any(
                    b.device.type != "cpu"
                    or b.dtype != torch.uint8
                    or not b.is_pinned()
                    or not b.is_contiguous()
                    or b.storage_offset() != 0
                    or b.numel() != b.untyped_storage().nbytes()
                    for b in buffers
                )
            ):
                return False
            # No borrowed range can alias another slot or the immutable source it stages.
            ranges = [
                (int(b.data_ptr()), int(b.data_ptr()) + b.numel()) for b in buffers if b.numel()
            ]
            sources = [
                (
                    int(g[0][3].untyped_storage().data_ptr()),
                    int(g[0][3].untyped_storage().data_ptr())
                    + int(g[0][3].untyped_storage().nbytes()),
                )
                for g in self.groups
                if g[0][3].untyped_storage().nbytes()
            ]
            all_ranges = sorted(ranges + sources)
            if any(a[1] > b[0] for a, b in zip(all_ranges, all_ranges[1:], strict=False)):
                return False
            slot, offset = 0, 0
            for index, group in enumerate(self.groups):
                storage = group[0][3].untyped_storage()
                source = torch.empty(0, dtype=torch.uint8, device="cpu").set_(
                    storage, 0, (storage.nbytes(),), (1,)
                )
                cursor = 0
                while cursor < source.numel():
                    while offset == buffers[slot].numel():
                        slot, offset = slot + 1, 0
                    count = min(source.numel() - cursor, buffers[slot].numel() - offset)
                    target = buffers[slot].view(-1)[offset : offset + count]
                    target.copy_(source[cursor : cursor + count])
                    self.staged_bytes += count
                    chunks.append((index, cursor, target))
                    cursor, offset = cursor + count, offset + count
        if not self._current():
            return False
        with torch.inference_mode(False), torch.no_grad(), torch.cuda.stream(self.stream):
            self.started = True
            for group in self.groups:
                self.destinations.append(
                    torch.empty(
                        group[0][3].untyped_storage().nbytes(),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                )
            for index, offset, source in chunks:
                self.destinations[index][offset : offset + source.numel()].copy_(
                    source, non_blocking=True
                )
                self.submitted_bytes += source.numel()
            self.copy_ready = torch.cuda.Event()
            self.copy_ready.record(self.stream)
        self.state = "prepared"
        return True

    def _poison(self, exc: BaseException, phase: str = "drain") -> NoReturn:
        self.poisoned = f"HOST transfer {phase} failed: {type(exc).__name__}: {exc}"
        self.state = "poisoned"
        raise RuntimeFailure(self.poisoned, code="device_fault") from exc

    def _release_staging(self) -> None:
        if self.staging is not None:
            try:
                self.staging.release()
            except BaseException as exc:
                self._poison(exc, "staging release")
            self.staging = None
            self._buffers = ()

    def publish(self) -> bool:
        """At actual demand, join and revalidate before atomically publishing any views."""
        if self.state != "prepared":
            raise RuntimeError("only a prepared transfer can publish")
        try:
            self.copy_ready.synchronize()
        except BaseException as exc:
            self._poison(exc, "copy completion")
        try:
            if not self._current():
                self.abort()
                return False
            torch = self.torch
            replacements: list[tuple[tuple[Module, str, bool, Tensor], Tensor, Tensor]] = []
            cursor = 0
            with torch.inference_mode(False), torch.no_grad():
                for group, backing in zip(self.groups, self.destinations, strict=True):
                    for slot in group:
                        tensor = slot[3]
                        placed = torch.empty(0, dtype=tensor.dtype, device=self.device).set_(
                            backing.untyped_storage(),
                            tensor.storage_offset(),
                            tensor.shape,
                            tensor.stride(),
                        )
                        placed.requires_grad_(tensor.requires_grad)
                        replacements.append((slot, self.originals[cursor][1], placed))
                        cursor += 1
        except BaseException as exc:
            if self.state != "poisoned":
                try:
                    self.abort()
                except BaseException as cleanup:
                    raise cleanup from exc
            raise
        host, seen = dict(self.unit.host), dict(self.unit.seen)
        attempted = []
        try:
            for slot, original, placed in replacements:
                module, name, parameter, tensor = slot
                attempted.append((slot, original))
                _bind(module, name, parameter, placed)
            for slot, original, placed in replacements:
                module, name, parameter, tensor = slot
                self.unit.host[(id(module), name)] = original if parameter else tensor
                self.unit.seen[(id(module), name)] = placed.data_ptr()
            self.unit.tier = DEVICE
        except BaseException:
            try:
                for slot, original in reversed(attempted):
                    module, name, parameter, tensor = slot
                    _bind(module, name, parameter, original if parameter else tensor)
                self.unit.host, self.unit.seen, self.unit.tier = host, seen, HOST
            except BaseException as exc:
                self._poison(exc, "binding rollback")
            self.abort()
            raise
        self.state = "published"
        self._release_staging()  # Copy was joined; buffers are no longer DMA-live.
        return True

    def finish(self) -> None:
        """After final consumer submission, prove device completion before retiring custody.

        A recorded current-stream event is evidence, not permission to assume there were
        no other using streams. The conservative device-wide drain is intentional.
        The caller may home the unit only after this method succeeds.
        """
        if self.state == "retired":
            return
        if self.state == "poisoned":
            raise RuntimeFailure(self.poisoned, code="device_fault")
        try:
            if self.state == "published":
                self.consumer_done = self.torch.cuda.Event()
                self.consumer_done.record(self.torch.cuda.current_stream(self.device))
                self.torch.cuda.synchronize(self.device)
            elif self.started:
                self.stream.synchronize()
        except BaseException as exc:
            self._poison(exc)
        self._release_staging()
        self.destinations.clear()
        self.originals.clear()
        self.groups.clear()
        self.copy_ready = self.consumer_done = None
        self.state = "retired"
        self.owned.discard(self)

    def abort(self) -> None:
        """Cancel optional work; published bindings remain for the Planner to home safely."""
        self.finish()


def to_host(torch: Torch, unit: Unit) -> int:
    """Rebind clean host backings, copying fresh unique storages only when needed."""
    return _move(torch, unit, torch.device("cpu"))


def store_safe(unit: Unit) -> bool:
    """The fill plane reconstructs slots, not alias/view storage geometry."""
    seen: set[tuple[str, int]] = set()
    for *_, tensor in unit.slots():
        key = _backing(tensor)
        if (
            key in seen
            or tensor.storage_offset() != 0
            or not tensor.is_contiguous()
            or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
        ):
            return False
        seen.add(key)
    return True


def to_disk(torch: Torch, component: Component, unit: Unit | None) -> None:
    """Drop one unit's bytes, or with None the whole component's, wherever they are."""
    store = component.store
    if store is None:
        raise LookupError(f"{component.name} has no store to refill from")
    units = component.units if unit is None else (unit,)
    if not all(store_safe(each) for each in units):
        raise RuntimeFailure("store cannot reconstruct aliased backing", code="memory_capability")
    component.moving = True
    try:
        with torch.inference_mode(False), torch.no_grad():
            if unit is None and not component.unit_loads:
                store.drop(component.name)
            else:
                for each in units:
                    store.drop_unit(each)
    finally:
        component.moving = False
    if any(each.tier != DISK for each in units):
        component.disk_epoch += 1
    for each in units:
        each.host.clear()
        each.seen.clear()
        each.pending[:0] = each.applied
        each.applied.clear()
        each.tier = DISK


def from_disk(torch: Torch, component: Component, unit: Unit | None) -> None:
    """Fill one unit, or with None the whole component, from the store onto the device; then
    apply the conversions a package made while it was on disk. Units dropped one at a time
    come back one at a time, as `to_disk` sent them."""
    store = component.store
    if store is None:
        raise LookupError(f"{component.name} has no store to refill from")

    def filled(each: Unit) -> None:
        for convert in each.pending:
            for _, module in each.owners():
                type(module)._apply(module, convert, recurse=False)
        each.applied, each.pending = each.pending, []
        each.tier = DEVICE

    whole = unit is None and not component.unit_loads
    units = component.units if unit is None else (unit,)
    if whole and any(each.tier != DISK for each in units):
        raise RuntimeFailure(
            "whole refill requires wholly stored weights", code="memory_capability"
        )
    loading = [
        (each, list(each.pending), list(each.applied)) for each in units if each.tier == DISK
    ]
    if any(not store_safe(each) for each, _, _ in loading):
        raise RuntimeFailure("store cannot reconstruct aliased backing", code="memory_capability")
    original = [slot for each, _, _ in loading for slot in each.slots()]
    component.moving = True
    try:
        with torch.inference_mode(False), torch.no_grad():
            if whole:
                store.load(component.name)
                for each, _, _ in loading:
                    filled(each)
            else:
                for each, _, _ in loading:
                    store.load_unit(component.name, each)
                    filled(each)
    except BaseException as exc:
        # A conversion can fail after fill has committed. Discard the preparation as
        # one transaction, including earlier units, and retain the original conversions.
        exc.__traceback__ = None
        try:
            with torch.inference_mode(False), torch.no_grad():
                if whole:
                    store.drop(component.name)
                else:
                    for each, _, _ in loading:
                        store.drop_unit(each)
            for module, name, parameter, tensor in original:
                if not tensor.is_meta:
                    raise RuntimeError("stored tensor identity changed during failed preparation")
                slots = module._parameters if parameter else module._buffers
                slots[name] = tensor
            for each, pending, applied in loading:
                each.pending, each.applied, each.tier = pending, applied, DISK
            measure(component)
        except BaseException as cleanup:
            cleanup.__traceback__ = None
            raise RuntimeFailure(
                "failed to restore stored placement", code="device_fault"
            ) from None
        raise exc from None
    finally:
        component.moving = False
    measure(component)


def install(
    component: Component,
    enter: Callable[[Component, Unit], None],
    leave: Callable[[Component, Unit], None],
) -> None:
    """A block loads before its forward if it is absent, and a streamed block goes home after."""

    def pre(unit: Unit, module: Module, args: object) -> None:
        enter(component, unit)

    def post(unit: Unit, module: Module, args: object, output: object) -> None:
        leave(component, unit)

    for unit in component.blocks:
        before, after = functools.partial(pre, unit), functools.partial(post, unit)
        before.__cozy_op__ = weakref.ref(before)  # type: ignore[attr-defined]
        after.__cozy_op__ = weakref.ref(after)  # type: ignore[attr-defined]
        component.hooks.append(unit.module.register_forward_pre_hook(before, prepend=True))
        component.hooks.append(unit.module.register_forward_hook(after, always_call=True))


def intercept_apply(
    torch: Torch,
    component: Component,
    applied: Callable[[], None],
    *,
    host_copy: Callable[[int], AbstractContextManager[bool]] | None = None,
) -> None:
    """A package's `.to(dtype)` on the root converts the units on device and host now, and
    queues the conversion for units on disk, whose meta tensors keep the store's dtype. A
    device move (`.to(device)`, `.cuda()`, `.cpu()`) is the planner's to make, never a
    package's: only its dtype part is kept."""
    root = component.root

    def _apply(fn: Convert, recurse: bool = True) -> Module:
        if component.moving or not recurse:
            return type(root)._apply(root, fn, recurse)
        canonical = _canonical_dtype(torch, fn)
        if not canonical and any(unit.tier == DISK for unit in component.units):
            raise RuntimeFailure("unbounded transform on stored weights", code="memory_capability")
        # A real meta tensor cannot execute .to(cpu/cuda). FakeTensor traces device
        # intent without allocating or initializing that device, including .cuda().
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            probes = [
                fn(torch.empty((), device="meta", dtype=dtype))
                for dtype in (torch.float16, torch.float32)
            ]
        if not all(isinstance(probe, torch.Tensor) for probe in probes):
            return type(root)._apply(root, fn, recurse)
        if any(probe.device.type != "meta" for probe in probes):
            # Two input dtypes distinguish a device-only move from .to(device,
            # dtype=float16), even when the first probe already has that dtype.
            if probes[0].dtype != probes[1].dtype:
                if [probe.dtype for probe in probes] != [torch.float16, torch.float32]:
                    raise TypeError("unsupported device conversion on a managed component")
                return root
            fn = functools.partial(_cast, probes[0].dtype)
        if canonical:
            fn = DTypeConvert(fn)
        # Prove every parked replay before mutating any resident owner or queue.
        for unit in component.units:
            if unit.tier == DISK:
                unit.pending.append(fn)
                try:
                    load_demand(torch, unit, component=component)
                finally:
                    unit.pending.pop()
        if any(
            getattr(module, "_compiled_call_impl", None) is not None for module in root.modules()
        ):
            raise RuntimeError("a compiled component cannot change its captured tensor storage")
        if any(unit.mutable for unit in component.units):
            raise RuntimeError("a component with declared mutable state cannot change dtype")
        # Arbitrary authored transforms have no storage-level alias contract. Refuse an
        # alias-changing conversion before mutating any unit, instead of splitting views.
        aliases: dict[tuple[str, int], list[Tensor]] = {}
        for unit in component.units:
            for *_, tensor in unit.slots():
                if not tensor.is_meta:
                    aliases.setdefault(_backing(tensor), []).append(tensor)
        with FakeTensorMode() as mode:
            for tensors in aliases.values():
                if len(tensors) > 1:
                    for tensor in tensors:
                        fake = mode.from_tensor(tensor)
                        if fn(fake) is not fake:
                            raise RuntimeError("a managed conversion cannot split aliased storage")
        # Invalidate before the first mutation too: an exception can leave a partially
        # converted resident module, which must not reuse old resource observations.
        applied()
        try:
            for unit in component.units:
                if unit.tier == DISK:
                    unit.pending.append(fn)
                    continue
                needed = 0
                with FakeTensorMode() as mode:
                    for *_, tensor in unit.slots():
                        if tensor.device.type == "cpu":
                            fake = mode.from_tensor(tensor)
                            converted = fn(fake)
                            if converted is not fake:
                                needed += converted.untyped_storage().nbytes()
                reservation = host_copy(needed) if host_copy is not None else nullcontext(True)
                with reservation as admitted:
                    if not admitted:
                        raise RuntimeFailure(
                            f"dtype conversion needs {needed} B of host memory",
                            code="host_shortfall",
                        )
                    for _, module in unit.owners():
                        type(module)._apply(module, fn, recurse=False)
                unit.applied.append(fn)
                unit.seen.clear()
        finally:
            # Earlier owners may already have converted when a later reservation or
            # cast fails. Keep their real backing sizes, without claiming that the
            # failed unit completed its replay or replacing the original exception.
            measure(component)
        applied()
        return root

    _apply.__cozy_op__ = weakref.ref(_apply)  # type: ignore[attr-defined]
    _remember(root, "_apply", _apply)


def _cast(dtype: object, tensor: Tensor) -> Tensor:
    return tensor.to(dtype) if tensor.is_floating_point() else tensor


def _remember(root: Module, name: str, value: object) -> None:
    saved = root.__dict__.setdefault("_cozy_wrapped", {})
    saved[name] = (name in vars(root), vars(root).get(name))
    root.__dict__[name] = value


def release(component: Component) -> None:
    """Remove our adapters, restoring package-owned instance methods exactly."""
    for hook in component.hooks:
        hook.remove()
    component.hooks.clear()
    for name, (present, prior) in component.root.__dict__.pop("_cozy_wrapped", {}).items():
        if not installed(vars(component.root).get(name)):
            continue  # A package replaced it after installation; it owns that method.
        if present:
            component.root.__dict__[name] = prior
        else:
            component.root.__dict__.pop(name, None)
    component.root.__dict__.pop("_cozy_op", None)


def wrap(root: Module, run: Run) -> None:
    """Own the complete eager call, including root pre/post hooks.

    Direct forward/encode/decode calls still have their own entry. Calls nested inside
    the complete module invocation join its existing planner frame. Compiled dispatch
    has an outer adapter too, but is resident-only and never automatically replayed.
    """
    for name in ("_call_impl", "_compiled_call_impl", *OP_METHODS):
        replaced = vars(root).get(name)
        router = getattr(replaced, "__cozy_group_router__", None)
        if (
            name == "_call_impl"
            and isinstance(router, weakref.ReferenceType)
            and router() is replaced
        ):
            # The group must broadcast before the local planner reaches its ready gate.
            # Its captured local call already contains our full eager-call adapter.
            continue
        if installed(replaced):
            continue
        bound = replaced if replaced is not None else getattr(root, name, None)
        if callable(bound):
            method = "forward" if name.startswith("_") else name
            _remember(root, name, op(method, bound, run))


def op(
    method: str,
    bound: Callable[..., object],
    run: Run,
) -> Callable[..., object]:
    @functools.wraps(bound)
    def op(*args: object, **kwargs: object) -> object:
        return run(method, bound, args, kwargs)

    op.__cozy_op__ = weakref.ref(op)  # type: ignore[attr-defined]
    op.__cozy_bound__ = bound  # type: ignore[attr-defined]
    return op


def original(value: object) -> object:
    """Unwrap only our adapters; a package callable never acquires their authority."""
    return original_callable(value)


def installed(value: object) -> bool:
    """This module put it there: a wrapper, hook or `_apply` it installs and removes."""
    return runtime_owned(value)


def ours(value: object) -> bool:
    """Safe to run again after an OOM: installed here, or marked pure by its author."""
    return installed(value) or is_declared_pure(value)


def trim(torch: Torch) -> None:
    """Return cached device memory to the driver once every stream is idle."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
