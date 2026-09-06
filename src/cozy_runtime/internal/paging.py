"""Runtime-owned paging units from a model's existing no-split forward boundaries.

The layout carries module references and shape facts, never tensor payloads or a CPU
state dict. Physical placement and completion fences remain in fill/residency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PageUnit:
    path: str
    owners: tuple[tuple[str, Any], ...]
    nbytes: int
    keys: tuple[str, ...]

    def live(self, component: str) -> dict[str, Any]:
        """The current registered destinations, including after a to_empty replacement."""
        result: dict[str, Any] = {}
        for path, module in self.owners:
            prefix = f"{component}.{path}." if path else f"{component}."
            for name, tensor in module.named_parameters(recurse=False, remove_duplicate=False):
                result[prefix + name] = tensor.detach()
            for name, tensor in module.named_buffers(recurse=False, remove_duplicate=False):
                if name not in module._non_persistent_buffers_set:
                    result[prefix + name] = tensor.detach()
        return result


@dataclass(frozen=True)
class BlockLayout:
    common: PageUnit
    blocks: tuple[PageUnit, ...]

    @property
    def working_bytes(self) -> int:
        return self.common.nbytes + max(block.nbytes for block in self.blocks)

    @property
    def total_bytes(self) -> int:
        return self.common.nbytes + sum(block.nbytes for block in self.blocks)

    def document(self) -> dict[str, int]:
        return {
            "total_bytes": self.total_bytes,
            "common_bytes": self.common.nbytes,
            "largest_block_bytes": max(block.nbytes for block in self.blocks),
            "working_bytes": self.working_bytes,
            "blocks": len(self.blocks),
        }


def partition(root: Any) -> BlockLayout | None:
    """Use declared indivisible blocks; never guess an arbitrary ModuleList's semantics.

    Cross-unit module/tensor aliases and custom state-dict layouts cannot safely be
    independently parked. Refuse before any device allocation rather than break ties.
    """
    classes = getattr(root, "_no_split_modules", None)
    if not classes:
        return None
    if not isinstance(classes, (tuple, list, set, frozenset)) or any(
        not isinstance(name, str) for name in classes
    ):
        raise ValueError("no-split module declarations must contain class names")
    modules = list(root.named_modules(remove_duplicate=False))
    blocks = [path for path, module in modules if path and type(module).__name__ in classes]
    if not blocks:
        return None
    if any(a != b and b.startswith(a + ".") for a in blocks for b in blocks):
        raise ValueError("declared paging blocks overlap")
    owners: dict[str, list[tuple[str, Any]]] = {"": [], **{path: [] for path in blocks}}
    sizes = dict.fromkeys(owners, 0)
    keys: dict[str, list[str]] = {path: [] for path in owners}
    storage_units: dict[Any, str] = {}
    module_units: dict[int, str] = {}
    for path, module in modules:
        unit = next(
            (block for block in blocks if path == block or path.startswith(block + ".")), ""
        )
        prior = module_units.setdefault(id(module), unit)
        if prior != unit:
            raise ValueError("one module is shared across paging units")
        owners[unit].append((path, module))
        tensors = list(module.named_parameters(recurse=False, remove_duplicate=False)) + list(
            module.named_buffers(recurse=False, remove_duplicate=False)
        )
        for local, tensor in tensors:
            storage = tensor.untyped_storage()
            if tensor.numel():
                prior = storage_units.setdefault(storage, unit)
                if prior != unit:
                    raise ValueError("one tensor storage is shared across paging units")
            sizes[unit] += int(tensor.numel() * tensor.element_size())
            if local not in module._non_persistent_buffers_set:
                keys[unit].append(f"{path}.{local}" if path else local)
    declared = {key for group in keys.values() for key in group}
    if declared != set(root.state_dict()):
        raise ValueError("custom state-dict keys do not map to registered paging destinations")
    units = {
        path: PageUnit(path, tuple(owners[path]), sizes[path], tuple(keys[path])) for path in owners
    }
    return BlockLayout(units[""], tuple(units[path] for path in blocks))
