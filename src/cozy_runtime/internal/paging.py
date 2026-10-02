"""A component's weight-plane regions, from the model's own no-split forward boundaries.

The layout carries module references and shape facts, never tensor payloads. `common` holds
every weight outside a block; each innermost declared block is one region the plane moves
as a unit and the executor hooks once per call (`weights.py`).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module


@dataclass(frozen=True)
class PageUnit:
    path: str
    owners: tuple[tuple[str, Module], ...]
    nbytes: int
    keys: tuple[str, ...]


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


#: A sub-block unit holds at least this fraction of the region limit (64 KiB of 16 MiB).
SMALL = 256


def partition(root: Module, limit: int | None = None) -> BlockLayout | None:
    """Use declared indivisible blocks; never guess an arbitrary ModuleList's semantics.

    With `limit`, the units are instead the largest module subtrees of at most `limit` bytes
    anywhere in the component (a single larger leaf stays whole). The weights of the modules
    that had to be split stay in `common`, and so does a subtree under `limit // SMALL`: a
    region costs 2 MiB of the budget however little it holds, and weights that small (norm
    scales) are what fused kernels read without calling their module. That is the sub-block
    grain a budget below the block floor needs (proving-cpu.md C2: SDXL at 1 GiB); every
    unit's own forward is its hook, and a weight read any other way is mapped for that op
    (`weights.Weights.reading`).
    Cross-unit module/tensor aliases and custom state-dict layouts cannot safely be
    independently placed; they refuse before any device allocation.
    """
    classes = getattr(root, "_no_split_modules", None)
    if limit is not None:
        classes = ()
    elif not classes:
        return None
    if not isinstance(classes, (tuple, list, set, frozenset)) or any(
        not isinstance(name, str) for name in classes
    ):
        raise ValueError("no-split module declarations must contain class names")
    modules = list(root.named_modules(remove_duplicate=False))
    declared = [path for path, module in modules if path and type(module).__name__ in classes]
    # The innermost declared instances are the blocks: an outer declared container (SDXL's
    # CrossAttnUpBlock2D around its resnets and transformer blocks) keeps only its own
    # weights, which fall to `common`.
    blocks = [a for a in declared if not any(b.startswith(a + ".") for b in declared)]
    if limit is not None:
        blocks = [unit for unit in _split(root, "", limit) if unit]
    if not blocks:
        return None
    owners: dict[str, list[tuple[str, Module]]] = {"": [], **{path: [] for path in blocks}}
    sizes = dict.fromkeys(owners, 0)
    keys: dict[str, list[str]] = {path: [] for path in owners}
    storage_units: dict[object, str] = {}
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
    mapped = {key for group in keys.values() for key in group}
    if mapped != set(root.state_dict()):
        raise ValueError("custom state-dict keys do not map to registered paging destinations")
    units = {
        path: PageUnit(path, tuple(owners[path]), sizes[path], tuple(keys[path])) for path in owners
    }
    return BlockLayout(units[""], tuple(units[path] for path in blocks))


def _split(root: Module, path: str, limit: int) -> list[str]:
    """`path`, or its children's units when its weights exceed `limit`."""
    module = root.get_submodule(path) if path else root
    children = [
        f"{path}.{name}" if path else name
        for name, child in module.named_children()
        if next(_tensors(child), None) is not None
    ]
    if not children or (path and _bytes(module) <= limit and not _container(module)):
        return [path] if _bytes(module) >= limit // SMALL else []
    return [unit for child in children for unit in _split(root, child, limit)]


def _container(module: Module) -> bool:
    """A ModuleList or ModuleDict: never called itself (its owner calls its members), so it
    can never be a unit, whose module call is what binds it."""
    return getattr(type(module).forward, "__name__", "") == "_forward_unimplemented"


def _tensors(module: Module) -> Iterator[Tensor]:
    """Parameters and buffers: an encoded leaf holds its stored roles as buffers."""
    return chain(module.parameters(), module.buffers())


def _bytes(module: Module) -> int:
    return sum(int(t.numel() * t.element_size()) for t in _tensors(module))
