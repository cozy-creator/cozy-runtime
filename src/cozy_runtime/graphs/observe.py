"""What a captured call touches outside its own allocations, as its torch ops show it.

A graph bakes in every address its kernels read. Its static inputs, its pool and its
component's weights (Regions, whose addresses outlive their pages) are safe; anything else a
forward reaches through Python state (a producer's loop tables, a turbo head index, a cache a
module keeps) is foreign. `Observer` runs over the capture and sorts each op's operands by
storage: a static input, made in the capture, a weight, or foreign. A replay then needs every
foreign storage alive (`_Recording.live`): a new loop's tables at new addresses make the graph
stale, never wrong. Writes are kept too: one into a static input is copied back to the
caller's tensor after each replay (the graph wrote the copy), and one into a foreign tensor
(H3's key-mean slots) is saved and put back around the capture's one check, so the replay and
the call as written each start from the same state. A kernel launched outside torch's
dispatcher (a ctypes or Triton launch) is not seen: only the capture's one check, replay
against the call as written, sees what it does.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

    from torch import Tensor
    from torch.multiprocessing.reductions import StorageWeakRef

#: an operand's storage made in the capture (its pool)
MADE = -1
#: one of its component's weights
WEIGHT = -2
#: anything else the call reached
FOREIGN = -3
#: a retained state's Region (a prefix's K/V): pinned at each replay, like a weight
RETAINED = -4
#: the foreign bytes a capture may write before it is refused: saving them around its check
#: costs that much again
WRITE_LIMIT = 256 << 20


class Observed:
    """A captured call's foreign storages, the static inputs it wrote and its foreign writes."""

    def __init__(self) -> None:
        self.foreign: list[StorageWeakRef] = []
        self.arg_writes: set[int] = set()
        #: each foreign view written in place, once by its layout (held until its check ends)
        self.writes: dict[tuple[object, ...], Tensor] = {}
        self.weight_writes = 0
        self.foreign_bytes = 0
        #: the retained states it read, by identity (`residency.kept_ranges`)
        self.retained: dict[int, object] = {}

    def refusal(self) -> str:
        """Why the capture can't be kept ("" when it can)."""
        if self.weight_writes:
            return f"it writes its weights ({self.weight_writes} ops)"
        written = sum(t.nbytes for t in self.writes.values())
        if written > WRITE_LIMIT:
            return f"it writes {written / (1 << 20):.0f} MiB of state outside the call"
        return ""


def ranges(tensors: Iterable[Tensor]) -> list[tuple[int, int]]:
    """The address ranges of these tensors' storages (an unbound weight has none)."""
    spans = []
    for tensor in tensors:
        if not tensor.is_cuda:
            continue
        try:
            storage = tensor.untyped_storage()
        except (RuntimeError, NotImplementedError):  # a wrapper subclass with no storage
            continue
        if ptr := storage.data_ptr():
            spans.append((ptr, ptr + storage.nbytes()))
    return spans


def merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Address ranges sorted and merged, so one search finds any address in them."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def observer(
    torch: ModuleType,
    statics: Sequence[Tensor | None],
    weights: list[tuple[int, int]],
    retained: Sequence[tuple[int, int, object]] = (),
) -> tuple[object, Observed]:
    """A dispatch mode for one capture over `statics` (by position), `weights` and `retained`
    states (address ranges), and what it observes."""
    from torch.multiprocessing.reductions import StorageWeakRef
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_leaves

    seen = Observed()
    kind: dict[int, int] = {}
    for index, static in enumerate(statics):
        if static is not None and (ptr := static.untyped_storage().data_ptr()):
            kind[ptr] = index
    starts = [start for start, _ in weights]
    tensor_type = torch.Tensor

    def weight(ptr: int) -> bool:
        at = bisect.bisect_right(starts, ptr) - 1
        return at >= 0 and ptr < weights[at][1]

    kept_starts = [start for start, _, _ in retained]

    def kept(ptr: int) -> object | None:
        at = bisect.bisect_right(kept_starts, ptr) - 1
        return retained[at][2] if at >= 0 and ptr < retained[at][1] else None

    def classify(tensor: Tensor) -> tuple[int, int]:
        storage = tensor.untyped_storage()
        ptr = storage.data_ptr()
        if not ptr:  # no bytes of its own (an unbound weight's metadata)
            return ptr, MADE
        found = kind.get(ptr)
        if found is None and (state := kept(ptr)) is not None:
            found = kind[ptr] = RETAINED
            seen.retained[id(state)] = state
        if found is None:
            found = kind[ptr] = WEIGHT if weight(ptr) else FOREIGN
            if found == FOREIGN:
                seen.foreign.append(StorageWeakRef(storage))
                seen.foreign_bytes += storage.nbytes()
        return ptr, found

    def operands(value: object) -> list[Tensor]:
        return [
            v for v in tree_leaves(value)
            if isinstance(v, tensor_type) and v.is_cuda and v.numel() and not v.is_meta
        ]

    class Observer(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: object,
            types: tuple[type, ...],
            args: tuple[object, ...] = (),
            kwargs: dict[str, object] | None = None,
        ) -> object:
            kwargs = kwargs or {}
            written: set[int] = set()
            for at, argument in enumerate(func._schema.arguments):  # type: ignore[attr-defined]  # an OpOverload
                alias = argument.alias_info
                if alias is None or not alias.is_write:
                    continue
                value = (
                    kwargs.get(argument.name)
                    if argument.kwarg_only or at >= len(args)
                    else args[at]
                )
                written.update(id(t) for t in operands(value))
            for tensor in operands((args, kwargs)):
                try:
                    _, found = classify(tensor)
                except (RuntimeError, NotImplementedError):  # a wrapper with no storage
                    continue
                if id(tensor) not in written:
                    continue
                if found >= 0:
                    seen.arg_writes.add(found)
                elif found in (WEIGHT, RETAINED):
                    seen.weight_writes += 1
                elif found == FOREIGN:
                    layout = (
                        tensor.untyped_storage().data_ptr(),
                        tensor.storage_offset(),
                        tuple(tensor.shape),
                        tuple(tensor.stride()),
                        tensor.dtype,
                    )
                    seen.writes.setdefault(layout, tensor)
            out = func(*args, **kwargs)  # type: ignore[operator]  # an OpOverload
            for tensor in operands(out):
                try:
                    ptr = tensor.untyped_storage().data_ptr()
                except (RuntimeError, NotImplementedError):
                    continue
                kind.setdefault(ptr, MADE)
            return out

    return Observer(), seen  # type: ignore[no-untyped-call]  # TorchDispatchMode's __init__
