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
from typing import Any, Protocol

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
class LoadDemand:
    resident: int
    peak: int


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


def load_demand(torch: Torch, unit: Unit) -> LoadDemand:
    """Current fill backing, settled dtype bytes and conservative conversion overlap.

    Only the installed Torch's canonical dtype closures establish this bound. Calling
    an arbitrary transform on a scalar does not prove its real allocation demand.
    """
    if unit.tier != DISK or not unit.pending:
        return LoadDemand(unit.nbytes, unit.nbytes)
    if not all(isinstance(convert, DTypeConvert) for convert in unit.pending):
        raise RuntimeFailure("unbounded stored conversion", code="memory_capability")
    slots = list(unit.slots())
    if any(type(module)._apply is not torch.nn.Module._apply for _, module in unit.owners()):
        raise RuntimeFailure(
            "custom owner conversion has no allocation bound", code="memory_capability"
        )
    if any(
        type(tensor) not in (torch.Tensor, torch.nn.Parameter)
        or tensor.layout != torch.strided
        or tensor.grad is not None
        for *_, tensor in slots
    ):
        raise RuntimeFailure(
            "stored conversion requires plain gradient-free tensors", code="memory_capability"
        )
    dtypes = [tensor.dtype for *_, tensor in slots]
    sizes = [tensor.numel() * tensor.element_size() for *_, tensor in slots]
    peak = max(unit.nbytes, sum(sizes))
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
        peak = max(peak, sum(sizes) + allocated)
        sizes, dtypes = next_sizes, changed
    return LoadDemand(sum(sizes), peak) if converted else LoadDemand(unit.nbytes, peak)


def load_sequence(torch: Torch, units: tuple[Unit, ...], *, whole: bool = False) -> LoadDemand:
    """New capacity for sequential unit loads, or one whole fill before replaying casts."""
    missing = [unit for unit in units if unit.tier != DEVICE]
    demands = [load_demand(torch, unit) for unit in missing]
    if whole:
        current = sum(unit.nbytes for unit in missing)
        peak = current
        for unit, demand in zip(missing, demands, strict=True):
            peak = max(peak, current - unit.nbytes + demand.peak)
            current += demand.resident - unit.nbytes
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

    def owners(self) -> list[tuple[str, Module]]:
        """Every (path, module) whose own tensors belong to this unit."""
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

    @property
    def units(self) -> tuple[Unit, ...]:
        return (self.common, *self.blocks)

    @property
    def nbytes(self) -> int:
        return sum(unit.nbytes for unit in self.units)

    def on(self, tier: str) -> int:
        return sum(unit.nbytes for unit in self.units if unit.tier == tier)


def _backing(tensor: Tensor) -> tuple[str, int]:
    storage = tensor.untyped_storage()
    return str(storage.device), storage.data_ptr() if storage.nbytes() else id(tensor)


def _bytes(tensors: Iterator[Tensor]) -> int:
    sizes: dict[object, int] = {}
    for tensor in tensors:
        key = id(tensor) if tensor.is_meta else _backing(tensor)
        sizes[key] = (
            tensor.numel() * tensor.element_size()
            if tensor.is_meta
            else tensor.untyped_storage().nbytes()
        )
    return sum(sizes.values())


def _tokens(module: Module) -> Iterator[tuple[str, int]]:
    yield "module", id(module)
    for tensor in (*module._parameters.values(), *module._buffers.values()):
        if tensor is not None:
            yield "tensor", id(tensor)
            if not tensor.is_meta and tensor.numel():
                yield f"storage:{tensor.device}", tensor.untyped_storage().data_ptr()


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


def to_host(torch: Torch, unit: Unit) -> int:
    """Rebind clean host backings, copying fresh unique storages only when needed."""
    return _move(torch, unit, torch.device("cpu"))


def store_safe(unit: Unit) -> bool:
    """The fill plane reconstructs slots, not alias/view storage geometry."""
    seen: set[tuple[str, int]] = set()
    for *_, tensor in unit.slots():
        if tensor.is_meta:
            continue
        key = _backing(tensor)
        if (
            key in seen
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
                    load_demand(torch, unit)
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
