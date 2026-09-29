"""Weight units and the moves between device, host and disk. Mechanism only: `memory` decides.

A component splits into units: the innermost repeated instances of its `_no_split_modules`
classes are blocks, everything else is the common unit. Moves are synchronous on the current
stream. A free is a dropped reference, so the caching allocator reuses the bytes in stream
order and a view held elsewhere keeps its storage alive. `trim` is the one place cached device
memory goes back to the driver, after the whole device is idle.
"""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

DEVICE, HOST, DISK = "device", "host", "disk"
OP_METHODS = ("forward", "encode", "decode")


class Store(Protocol):
    """The disk tier: the fill plane's validated TensorFS path."""

    def load(self, name: str) -> None: ...

    def load_unit(self, name: str, unit: Unit) -> None: ...

    def drop(self, name: str) -> None: ...

    def drop_unit(self, unit: Unit) -> None: ...

    def unit_loads(self, name: str, keys: frozenset[str]) -> bool: ...


@dataclass(eq=False)
class Unit:
    path: str
    module: Any
    #: block roots the common unit skips; empty for a block
    skip: tuple[Any, ...] = ()
    nbytes: int = 0
    tier: str = DEVICE
    #: the host copy of each slot, kept while the device copy is the one it was copied to:
    #: a clean cache, so leaving the device again only rebinds
    host: dict[tuple[int, str], Any] = field(default_factory=dict)
    seen: dict[tuple[int, str], int] = field(default_factory=dict)
    #: loaded by a forward hook for one forward, and sent home after it
    streamed: bool = False
    #: package conversions applied since the unit's last fill, and those waiting for its next
    applied: list[Callable[[Any], Any]] = field(default_factory=list)
    pending: list[Callable[[Any], Any]] = field(default_factory=list)

    def owners(self) -> list[tuple[str, Any]]:
        """Every (path, module) whose own tensors belong to this unit."""
        skip = {id(module) for module in self.skip}
        found: list[tuple[str, Any]] = []
        stack = [(self.path, self.module)]
        while stack:
            path, module = stack.pop()
            found.append((path, module))
            for name, child in reversed(module._modules.items()):
                if child is not None and id(child) not in skip:
                    stack.append((f"{path}.{name}" if path else name, child))
        return found

    def slots(self) -> Iterator[tuple[Any, str, bool, Any]]:
        for _, module in self.owners():
            for name, tensor in list(module._parameters.items()):
                if tensor is not None:
                    yield module, name, True, tensor
            for name, tensor in list(module._buffers.items()):
                if tensor is not None:
                    yield module, name, False, tensor

    def live(self, component: str) -> dict[str, Any]:
        """The persistent destinations the fill plane writes, by component-qualified key."""
        found: dict[str, Any] = {}
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
    root: Any
    common: Unit
    blocks: tuple[Unit, ...]
    store: Store | None = None
    #: the store can fill each unit alone; otherwise it refills the component whole
    unit_loads: bool = False
    hooks: list[Any] = field(default_factory=list)
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


def _bytes(tensors: Iterator[Any]) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in tensors:
        if id(tensor) not in seen:
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
    return total


def _tokens(module: Any) -> Iterator[tuple[str, int]]:
    yield "module", id(module)
    for tensor in (*module._parameters.values(), *module._buffers.values()):
        if tensor is not None:
            yield "tensor", id(tensor)
            if not tensor.is_meta and tensor.numel():
                yield "storage", tensor.untyped_storage().data_ptr()


def partition(name: str, root: Any, tier: str, store: Store | None = None) -> Component:
    """Blocks are the innermost REPEATED instances of the declared no-split classes. The unit
    holding the root's first parameter stays common, so `.device` stays true; units that share
    a module or a storage join the common unit."""
    classes = set(getattr(root, "_no_split_modules", None) or ())
    found: list[tuple[str, Any]] = []

    def walk(path: str, module: Any) -> bool:
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


def _bind(module: Any, name: str, parameter: bool, tensor: Any) -> None:
    if parameter:
        module._parameters[name].data = tensor
    else:
        module._buffers[name] = tensor


def to_device(torch: Any, unit: Unit, device: Any) -> int:
    """Copy a host unit to the device; its host copy stays for a free return. Bytes copied."""
    moves: list[tuple[Any, str, bool, Any, Any]] = []
    with torch.inference_mode(False), torch.no_grad():
        for module, name, parameter, tensor in list(unit.slots()):
            if tensor.device.type == device.type:
                continue
            if tensor.is_meta:
                raise LookupError(f"{unit.path or '<common>'} is on disk")
            moves.append((module, name, parameter, tensor.detach(), tensor.detach().to(device)))
    # Bound only once every copy exists: an allocation failure part way leaves the unit whole
    # where it was, and the copies made so far are dropped with the list.
    for module, name, parameter, host, placed in moves:
        unit.host[(id(module), name)] = host
        unit.seen[(id(module), name)] = placed.data_ptr()
        _bind(module, name, parameter, placed)
    unit.tier = DEVICE
    return sum(placed.numel() * placed.element_size() for *_, placed in moves)


def to_host(torch: Any, unit: Unit) -> int:
    """Rebind a device unit to its host copy, copying back only what changed. Bytes copied."""
    copied = 0
    with torch.inference_mode(False), torch.no_grad():
        for module, name, parameter, tensor in list(unit.slots()):
            if tensor.is_meta or tensor.device.type == "cpu":
                continue
            key = (id(module), name)
            host = unit.host.get(key)
            if (
                host is None
                or unit.seen.get(key) != tensor.data_ptr()
                or host.shape != tensor.shape
                or host.dtype != tensor.dtype
            ):
                host = tensor.detach().to("cpu")
                copied += host.numel() * host.element_size()
            _bind(module, name, parameter, host)
    unit.seen.clear()
    unit.tier = HOST
    return copied


def to_disk(torch: Any, component: Component, unit: Unit | None) -> None:
    """Drop one unit's bytes, or with None the whole component's, wherever they are."""
    store = component.store
    if store is None:
        raise LookupError(f"{component.name} has no store to refill from")
    units = component.units if unit is None else (unit,)
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
    for each in units:
        each.host.clear()
        each.seen.clear()
        each.pending[:0] = each.applied
        each.applied.clear()
        each.tier = DISK


def from_disk(torch: Any, component: Component, unit: Unit | None) -> None:
    """Fill one unit, or with None the whole component, from the store onto the device; then
    apply the conversions a package made while it was on disk."""
    store = component.store
    if store is None:
        raise LookupError(f"{component.name} has no store to refill from")
    units = component.units if unit is None else (unit,)
    component.moving = True
    try:
        with torch.inference_mode(False), torch.no_grad():
            if unit is None:
                store.load(component.name)
            else:
                store.load_unit(component.name, unit)
            for each in units:
                for convert in each.pending:
                    for _, module in each.owners():
                        type(module)._apply(module, convert, recurse=False)
                each.applied, each.pending = each.pending, []
                each.tier = DEVICE
    finally:
        component.moving = False
    measure(component)


def install(
    component: Component,
    enter: Callable[[Component, Unit], None],
    leave: Callable[[Component, Unit], None],
) -> None:
    """A block loads before its forward if it is absent, and a streamed block goes home after."""

    def pre(unit: Unit, module: Any, args: Any) -> None:
        enter(component, unit)

    def post(unit: Unit, module: Any, args: Any, output: Any) -> None:
        leave(component, unit)

    for unit in component.blocks:
        before, after = functools.partial(pre, unit), functools.partial(post, unit)
        before.__cozy_op__ = after.__cozy_op__ = True  # type: ignore[attr-defined]
        component.hooks.append(unit.module.register_forward_pre_hook(before, prepend=True))
        component.hooks.append(unit.module.register_forward_hook(after, always_call=True))


def intercept_apply(torch: Any, component: Component, applied: Callable[[], None]) -> None:
    """A package's `.to(dtype)` on the root converts the units on device and host now, and
    queues the conversion for units on disk, whose meta tensors keep the store's dtype. A
    device move (`.to(device)`, `.cuda()`, `.cpu()`) is the planner's to make, never a
    package's: only its dtype part is kept."""
    root = component.root

    def _apply(fn: Callable[[Any], Any], recurse: bool = True) -> Any:
        if component.moving or not recurse:
            return type(root)._apply(root, fn, recurse)
        probe = fn(torch.empty((), device="meta", dtype=torch.float16))
        if not isinstance(probe, torch.Tensor):
            return type(root)._apply(root, fn, recurse)
        if probe.device.type != "meta":
            if probe.dtype == torch.float16:
                return root
            dtype = probe.dtype
            fn = functools.partial(_cast, dtype)
        for unit in component.units:
            if unit.tier == DISK:
                unit.pending.append(fn)
                continue
            for _, module in unit.owners():
                type(module)._apply(module, fn, recurse=False)
            unit.applied.append(fn)
            unit.seen.clear()
        measure(component)
        applied()
        return root

    _apply.__cozy_op__ = True  # type: ignore[attr-defined]
    root.__dict__["_apply"] = _apply


def _cast(dtype: Any, tensor: Any) -> Any:
    return tensor.to(dtype) if tensor.is_floating_point() else tensor


def release(component: Component) -> None:
    """Remove every hook and wrapper this module installed on the component."""
    for hook in component.hooks:
        hook.remove()
    component.hooks.clear()
    for name in (*OP_METHODS, "_apply", "_cozy_op"):
        value = vars(component.root).get(name)
        if value is not None and (ours(value) or name == "_cozy_op"):
            del component.root.__dict__[name]


def wrap(root: Any, run: Callable[[str, Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any]) -> None:
    """Route the root's forward, encode and decode through `run`, keeping each signature."""
    for method in OP_METHODS:
        bound = getattr(root, method, None)
        if method not in vars(root) and callable(bound):
            root.__dict__[method] = op(method, bound, run)


def op(
    method: str,
    bound: Callable[..., Any],
    run: Callable[[str, Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any],
) -> Callable[..., Any]:
    @functools.wraps(bound)
    def op(*args: Any, **kwargs: Any) -> Any:
        return run(method, bound, args, kwargs)

    op.__cozy_op__ = True  # type: ignore[attr-defined]
    return op


def ours(value: Any) -> bool:
    return bool(getattr(value, "__cozy_op__", False) or getattr(value, "__cozy_pure__", False))


def trim(torch: Any) -> None:
    """Return cached device memory to the driver once every stream is idle."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
