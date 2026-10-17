"""A component's weight-plane regions, from the model's own no-split forward boundaries.

The layout carries module paths and shape facts, never tensor payloads. `common` holds every
weight outside a unit; each unit is one region the plane moves as a unit and the executor hooks
once per call, at its first module's forward (`weights.py`).
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
    #: the module subtrees whose weights it holds; the first one's forward is its hook
    paths: tuple[str, ...]
    nbytes: int
    keys: tuple[str, ...]

    @property
    def path(self) -> str:
        return self.paths[0]


@dataclass(frozen=True)
class BlockLayout:
    common: PageUnit
    blocks: tuple[PageUnit, ...]

    @property
    def total_bytes(self) -> int:
        return self.common.nbytes + sum(block.nbytes for block in self.blocks)


#: A sub-block unit holds at least this fraction of the region limit (64 KiB of 16 MiB).
SMALL = 256


def partition(root: Module, limit: int | None = None) -> BlockLayout | None:
    """Use declared indivisible blocks (instances of a declared class or of any subclass of
    one); never guess an arbitrary ModuleList's semantics. Blocks that share a tensor storage
    cannot be placed apart: they refuse before any device allocation.

    With `limit`, the units are instead the largest module subtrees of at most `limit` bytes
    anywhere in the component (a single larger leaf stays whole, and so does a LoRA'd linear
    with its factors: they are baked into it when its bytes are placed), and this grain never
    refuses for aliases: units that share a storage are one region (tied embeddings), hooked
    at the first, the others' weights read through their stand-ins; one that shares with
    `common` joins it. The weights of the modules that had to be split stay
    in `common`, and so do a subtree under `limit // SMALL` (a region costs 2 MiB of the
    budget however little it holds, and weights that small, norm scales, are what fused
    kernels read without calling their module) and a module nothing calls (`_container`).
    Custom state-dict layouts refuse at either grain.
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
    # A declared class names its subclasses too (SeedVR2's VAE declares diffusers'
    # `ResnetBlock2D` and builds `ResnetBlock3D(ResnetBlock2D)`: run 5439).
    declared = [
        path
        for path, module in modules
        if path and any(kind.__name__ in classes for kind in type(module).__mro__)
    ]
    # The innermost declared instances are the blocks: an outer declared container (SDXL's
    # CrossAttnUpBlock2D around its resnets and transformer blocks) keeps only its own
    # weights, which fall to `common`.
    blocks = [a for a in declared if not any(b.startswith(a + ".") for b in declared)]
    if limit is not None:
        blocks = [unit for unit in _split(root, "", limit) if unit]
    if not blocks:
        return None
    rank = {path: index for index, path in enumerate(["", *blocks])}
    group = {path: path for path in rank}

    def find(unit: str) -> str:
        while group[unit] != unit:
            unit = group[unit]
        return unit

    units = [
        (path, module, next((b for b in blocks if path == b or path.startswith(b + ".")), ""))
        for path, module in modules
    ]
    storages: dict[object, str] = {}
    for _, module, unit in units:
        for tensor in chain(module.parameters(recurse=False), module.buffers(recurse=False)):
            if not tensor.numel():
                continue
            prior = storages.setdefault(tensor.untyped_storage(), unit)
            a, b = sorted((find(prior), find(unit)), key=rank.__getitem__)
            if a != b and limit is None:
                raise ValueError("one tensor storage is shared across paging units")
            group[b] = a
    sizes = dict.fromkeys(rank, 0)
    keys: dict[str, list[str]] = {path: [] for path in rank}
    counted: set[int] = set()
    for path, module, unit in units:
        unit = find(unit)
        tensors = chain(
            module.named_parameters(recurse=False, remove_duplicate=False),
            module.named_buffers(recurse=False, remove_duplicate=False),
        )
        for local, tensor in tensors:
            if id(tensor) not in counted:  # a tied tensor's bytes count once
                counted.add(id(tensor))
                sizes[unit] += int(tensor.numel() * tensor.element_size())
            if local not in module._non_persistent_buffers_set:
                keys[unit].append(f"{path}.{local}" if path else local)
    if {key for held in keys.values() for key in held} != set(root.state_dict()):
        raise ValueError("custom state-dict keys do not map to registered paging destinations")

    return BlockLayout(
        PageUnit(("",), sizes[""], tuple(keys[""])),
        tuple(
            PageUnit(tuple(p for p in blocks if find(p) == head), sizes[head], tuple(keys[head]))
            for head in blocks
            if find(head) == head
        ),
    )


def _split(root: Module, path: str, limit: int) -> list[str]:
    """`path`, or its children's units when its weights exceed `limit`."""
    module = root.get_submodule(path) if path else root
    children = [
        f"{path}.{name}" if path else name
        for name, child in module.named_children()
        if next(_tensors(child), None) is not None
    ]
    whole = _bytes(module) <= limit or hasattr(module, "lora_A")
    if not children or (path and whole and not _container(module)):
        unit = _bytes(module) >= limit // SMALL and not _container(module)
        return [path] if unit else []
    return [unit for child in children for unit in _split(root, child, limit)]


def _container(module: Module) -> bool:
    """A module without a forward of its own (a ModuleList, a holder of factors its owner
    reads, as H3's turbo LoRA's): never called itself, so it can never be a unit, whose
    module call is what binds it. Its own weights stay in `common`."""
    return getattr(type(module).forward, "__name__", "") == "_forward_unimplemented"


def _tensors(module: Module) -> Iterator[Tensor]:
    """Parameters and buffers: an encoded leaf holds its stored roles as buffers."""
    return chain(module.parameters(), module.buffers())


def _bytes(module: Module) -> int:
    return sum(int(t.numel() * t.element_size()) for t in _tensors(module))
