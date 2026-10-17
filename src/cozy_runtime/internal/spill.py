"""An executor's own tensors on its GPU, moved to page-locked host memory in place.

The last room an op can be given, once every weight it does not read is unmapped and every
idle tenant has given back: the device bytes of the tensors this process holds (a package's
caches, an earlier stage's outputs, a decoder's feature maps). Each storage torch's allocator
holds is copied to page-locked host memory the GPU addresses directly (unified addressing),
and the storage is repointed there. Every tensor over it stays a CUDA tensor, read and
written over the bus from then on, so whatever holds one is unchanged and nothing has to bring
it back; its device bytes go back to torch. Least recently used first (`touch`), the op's
own operands last. Only a stage under pressure keeps that order: from its first spill or host
result until its host blocks are gone, so a stage with room pays nothing per op.

Locking new pages runs at about 1.6 GB/s, most of a spill's cost: a block no tensor holds any
more stays locked for the next spill or host map of its size, until the stage closes.
"""

from __future__ import annotations

import gc
import time
import weakref
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from torch import Tensor, UntypedStorage
    from torch import device as Device

#: Storages smaller than this are not worth a copy.
SMALLEST = 1 << 20
#: `cudaHostRegisterPortable | cudaHostRegisterMapped`
_MAPPED = 3
#: Host blocks are locked in multiples of this, so maps of about one size share them.
GRAIN = 2 << 20
#: cudaMalloc's alignment, which CUDA libraries take for granted (cuBLASLt answers INVALID_VALUE
#: for a workspace off it: run 9141); torch's host allocator gives 64.
ALIGN = 256
#: [times tensors moved, their bytes, nanoseconds it took] since the process began
MOVED = [0, 0, 0]
#: [times tensors came back (`restore`), their bytes] since the process began
RESTORED = [0, 0]
#: locked blocks no tensor holds, by size; and the address of every block one holds -> its size
_FREE: dict[int, list[Any]] = {}
HELD: dict[int, int] = {}
#: storage address -> when an op last read or made it, in ops, and that tensor (`touch`):
#: the candidates a spill looks at first; entries this many ops past are forgotten
USED: dict[int, tuple[int, weakref.ref[Tensor]]] = {}
FORGET = 1 << 16
_CLOCK = [0]
#: whether ops are noted (`touch`): from a stage's first spill or host result until its host
#: blocks are all gone (`release`)
PRESSED = [False]


def _size(nbytes: int) -> int:
    return -(-max(nbytes, 1) // GRAIN) * GRAIN


def reusable(nbytes: int) -> bool:
    """Whether a free block can take `nbytes` without locking more."""
    return bool(_FREE.get(_size(nbytes)))


class _Host:
    """A host block: page-locked memory, as CUDA addresses it. Locked anew only when no free
    block has its size; freed, it waits for the next one."""

    def __init__(self, torch: ModuleType, nbytes: int) -> None:
        self.size = _size(nbytes)
        free = _FREE.get(self.size)
        if free:
            self.data = free.pop()
        else:
            self.data = torch.empty(self.size + ALIGN, dtype=torch.uint8)
            cudart = torch.cuda.cudart()
            torch.cuda.check_error(
                cudart.cudaHostRegister(self.data.data_ptr(), self.size + ALIGN, _MAPPED)
            )
        start = -self.data.data_ptr() % ALIGN
        self.view = self.data[start : start + self.size]
        pointer = self.view.data_ptr()
        HELD[pointer] = self.size
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }

    def __del__(self) -> None:
        HELD.pop(self.view.data_ptr(), None)
        _FREE.setdefault(self.size, []).append(self.data)


def release(torch: ModuleType) -> None:
    """Free blocks go back to the host: a closed stage keeps none locked. Once no tensor is in
    host memory, ops are no longer noted."""
    if not HELD:
        PRESSED[0] = False
        USED.clear()
    if not _FREE:
        return
    cudart = torch.cuda.cudart()
    for blocks in _FREE.values():
        for data in blocks:
            cudart.cudaHostUnregister(data.data_ptr())
    _FREE.clear()


def stored(values: Iterable[object]) -> Iterator[Tensor]:
    """The tensors among `values` (one level of sequences) whose storage holds their bytes: a
    plain strided tensor, or a traceable wrapper's inner ones (an AsyncCollectiveTensor, a
    DTensor). A wrapper's own storage has no bytes (run 6349: `data_ptr` raises on it), nor
    has another subclass's (a weight's unbound stand-in), a meta or a sparse tensor's."""
    for value in values:
        for item in value if isinstance(value, (tuple, list)) else (value,):
            if type(item).__name__ in ("Tensor", "Parameter"):
                if not item.is_meta and str(item.layout) == "torch.strided":
                    yield cast("Tensor", item)
            elif (flatten := getattr(item, "__tensor_flatten__", None)) is not None:
                yield from stored([getattr(item, name) for name in flatten()[0]])


def touch(values: Iterable[object]) -> None:
    """An op read or made `values`: their device storages are the last to move."""
    if not PRESSED[0]:
        return
    clock = _CLOCK[0] = _CLOCK[0] + 1
    for value in stored(values):
        storage = value.untyped_storage()
        if value.is_cuda and storage.nbytes() >= SMALLEST:
            USED[storage.data_ptr()] = (clock, weakref.ref(value))
    if len(USED) > FORGET:
        for key in [k for k, (at, _) in USED.items() if at < clock - FORGET]:
            del USED[key]


def hosted(value: object) -> bool:
    """Whether `value`'s bytes are in a host block."""
    return bool(HELD) and any(t.untyped_storage().data_ptr() in HELD for t in stored((value,)))


def empty(torch: ModuleType, device: Device, like: Tensor) -> Tensor:
    """A CUDA tensor of `like`'s shape, strides and dtype whose bytes are a host block."""
    PRESSED[0] = True
    span = 1 + sum((n - 1) * s for n, s in zip(like.shape, like.stride(), strict=True))
    nbytes = span * like.element_size() if like.numel() else 0
    raw = torch.as_tensor(_Host(torch, nbytes), device=device)
    return raw.view(like.dtype).as_strided(like.shape, like.stride())


def spill(
    torch: ModuleType, device: Device, need: int, operands: Iterable[object], room: int
) -> int:
    """Move this process's tensors on `device` to host memory, least recently used first,
    until `need` device bytes came back, the storages `operands` read last, at most `room`
    bytes. Returns the bytes moved."""
    PRESSED[0] = True
    started = time.perf_counter_ns()
    torch.cuda.synchronize(device)
    segments = sorted(
        (s["address"], s["address"] + s["total_size"])
        for s in torch.cuda.memory_snapshot()
        if s["device"] == device.index
    )
    starts = [start for start, _ in segments]
    read = {int(t.untyped_storage().data_ptr()) for t in stored(operands) if t.is_cuda}
    found: dict[int, UntypedStorage] = {}

    def add(values: Iterable[object]) -> int:
        """Storages torch's allocator holds on `device` among `values`; the bytes of those
        no operand reads."""
        spare = 0
        for value in values:
            if (
                type(value) not in (torch.Tensor, torch.nn.Parameter)
                or value.layout is not torch.strided
                or value.device != device
            ):
                continue
            storage = value.untyped_storage()
            address, at = storage.data_ptr(), bisect_right(starts, storage.data_ptr()) - 1
            fresh = address not in found and storage.nbytes() >= SMALLEST
            if fresh and at >= 0 and address < segments[at][1]:
                found[address] = storage
                spare += 0 if address in read else storage.nbytes()
        return spare

    # The tensors ops touched cover a decode's maps; every object only when they fall short.
    if add(ref() for _, ref in list(USED.values())) < need:
        add(gc.get_objects())
    moved = 0
    order = sorted(
        found.items(), key=lambda r: (r[0] in read, USED.get(r[0], (0,))[0], -r[1].nbytes())
    )
    for _, storage in order:
        if moved >= need:
            break
        if storage.nbytes() <= room - moved:
            moved += _move(torch, device, storage)
    MOVED[0] += moved > 0
    MOVED[1] += moved
    MOVED[2] += time.perf_counter_ns() - started
    return moved


def restore(torch: ModuleType, device: Device, room: int) -> int:
    """The storages in host blocks come back to `device`, largest first, while `room` bytes
    allow: each is copied back and repointed in place, and its host block is free again. A
    moved tensor nobody brings back is read over the bus for as long as it lives, and a cache
    or a decoder's state lives across calls. Returns the bytes restored."""
    if not HELD or room <= 0:
        return 0
    torch.cuda.synchronize(device)
    starts = sorted(HELD)
    found: dict[int, UntypedStorage] = {}
    for value in gc.get_objects():
        if (
            type(value) not in (torch.Tensor, torch.nn.Parameter)
            or value.layout is not torch.strided
            or value.device != device
        ):
            continue
        storage = value.untyped_storage()
        address, at = storage.data_ptr(), bisect_right(starts, storage.data_ptr()) - 1
        if at >= 0 and address < starts[at] + HELD[starts[at]]:
            found.setdefault(address, storage)
    back = 0
    for storage in sorted(found.values(), key=lambda s: -s.nbytes()):
        nbytes = int(storage.nbytes())
        if back + nbytes > room:
            continue
        fresh = torch.empty(nbytes, dtype=torch.uint8, device=device)
        fresh.copy_(torch.empty(0, dtype=torch.uint8, device=device).set_(storage))
        # `fresh` takes the host pointer and frees its block when it goes
        storage._swap_data_ptr_(fresh.untyped_storage())
        back += nbytes
    torch.cuda.synchronize(device)
    RESTORED[0] += back > 0
    RESTORED[1] += back
    return back


def _move(torch: ModuleType, device: Device, storage: UntypedStorage) -> int:
    nbytes = int(storage.nbytes())
    host = _Host(torch, nbytes)
    host.view[:nbytes].copy_(torch.empty(0, dtype=torch.uint8, device=device).set_(storage))
    alias = torch.as_tensor(host, device=device)
    storage._swap_data_ptr_(alias.untyped_storage())
    return nbytes
