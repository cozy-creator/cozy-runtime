"""Construction-owned tensor slots, independent of later adapter module nesting."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from cozy_runtime.author._loader import Census


@dataclass(frozen=True, slots=True)
class _Slot:
    owner: Any
    name: str
    parameter: bool

    def value(self) -> Any:
        return getattr(self.owner, self.name)


class BaseWeights:
    """Keep logical source names and base ownership fixed at construction.

    References are to parameter *slots*, not their current storage. A request may
    insert PEFT wrappers without changing source identity or making its adapters
    part of base eviction. Device data is dropped through normal Tensor lifetime;
    storage retained by another owner is never resized underneath that owner.
    """

    def __init__(self, component: str, root: Any, walked: Census) -> None:
        self.component = component
        self.destinations = tuple(d for d in walked.destinations if d.component == component)
        self._slots: dict[str, _Slot] = {}
        aliases: dict[int, int] = {}
        for path, module in root.named_modules(remove_duplicate=False):
            for parameter, table in ((True, module._parameters), (False, module._buffers)):
                for name, value in table.items():
                    if value is None:
                        continue
                    key = ".".join(p for p in (component, path, name) if p)
                    self._slots[key] = _Slot(module, name, parameter)
                    storage = value.untyped_storage()._cdata
                    previous = aliases.setdefault(storage, id(value))
                    if previous != id(value):
                        raise ValueError("standard binding does not support distinct shared views")
        if any(d.key not in self._slots for d in self.destinations):
            raise ValueError("standard binding requires registered logical parameter/buffer slots")
        self.live()

    def tensors(self) -> Iterator[Any]:
        for slot in self._slots.values():
            yield slot.value()

    def owner(self, key: str) -> tuple[Any, str]:
        slot = self._slots[key]
        return slot.owner, slot.name

    def live(self) -> dict[str, Any]:
        from cozy_runtime.author._loader import canonical_dtype

        live = {}
        for destination in self.destinations:
            value = self._slots[destination.key].value()
            if (
                tuple(value.shape) != destination.spec.shape
                or canonical_dtype(str(value.dtype).removeprefix("torch."))
                != destination.spec.dtype
            ):
                raise ValueError(f"logical base slot changed: {destination.key}")
            live[destination.key] = value.detach()
        return live

    def evict(self) -> None:
        self._move("meta", empty=False)

    def reserve(self, device: Any, *, initial: bool = False) -> None:
        if not initial and any(value.device.type != "meta" for value in self.tensors()):
            raise ValueError("base reservation requires evicted parameters")
        self._move(device, empty=True)

    def _move(self, device: Any, *, empty: bool) -> None:
        import torch

        moved: dict[int, Any] = {}
        for slot, value in [(slot, slot.value()) for slot in self._slots.values()]:
            replacement = moved.get(id(value))
            if replacement is None:
                if empty:
                    # Restaging reserves logical destinations for the existing fill
                    # transaction; encoded parts replace them after verified transfer.
                    replacement = torch.empty_strided(
                        value.shape, value.stride(), dtype=value.dtype, device=device
                    )
                else:
                    replacement = value.to(device=device)
                if slot.parameter:
                    replacement = torch.nn.Parameter(replacement, requires_grad=value.requires_grad)
                moved[id(value)] = replacement
            setattr(slot.owner, slot.name, replacement)
