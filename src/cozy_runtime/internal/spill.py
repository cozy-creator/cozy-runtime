"""An executor's own tensors on its GPU, moved to page-locked host memory in place.

The last room an op can be given, once every weight it does not read is unmapped and every
idle tenant has given back: the device bytes of the tensors this process holds (a package's
caches, an earlier stage's outputs, a decoder's feature maps). Each storage torch's allocator
holds is copied to page-locked host memory the GPU addresses directly (unified addressing),
and the storage is repointed there. Every tensor over it stays a CUDA tensor, read and
written over the bus from then on, so whatever holds one is unchanged and nothing has to bring
it back; its device bytes go back to torch. Least recently used first (`touch`), the op's
own operands last.

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
#: [times tensors moved, their bytes, nanoseconds it took] since the process began
MOVED = [0, 0, 0]
#: locked blocks no tensor holds, by size; and the address of every block one holds -> its size
_FREE: dict[int, list[Any]] = {}
HELD: dict[int, int] = {}
#: storage address -> when an op last read or made it, in ops, and that tensor (`touch`):
#: the candidates a spill looks at first; entries this many ops past are forgotten
USED: dict[int, tuple[int, weakref.ref[Tensor]]] = {}
FORGET = 1 << 16
_CLOCK = [0]


class _Host:
    """A host block: page-locked memory, as CUDA addresses it. Locked anew only when no free
    block has its size; freed, it waits for the next one."""

    def __init__(self, torch: ModuleType, nbytes: int) -> None:
        self.size = -(-max(nbytes, 1) // GRAIN) * GRAIN
        free = _FREE.get(self.size)
        if free:
            self.data = free.pop()
        else:
            self.data = torch.empty(self.size, dtype=torch.uint8)
            cudart = torch.cuda.cudart()
            torch.cuda.check_error(
                cudart.cudaHostRegister(self.data.data_ptr(), self.size, _MAPPED)
            )
        pointer = self.data.data_ptr()
        HELD[pointer] = self.size
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }

    def __del__(self) -> None:
        HELD.pop(self.data.data_ptr(), None)
        _FREE.setdefault(self.size, []).append(self.data)


def release(torch: ModuleType) -> None:
    """Free blocks go back to the host: a closed stage keeps none locked."""
    if not _FREE:
        return
    cudart = torch.cuda.cudart()
    for blocks in _FREE.values():
        for data in blocks:
            cudart.cudaHostUnregister(data.data_ptr())
    _FREE.clear()


def touch(tensors: Iterable[Tensor]) -> None:
    """An op read or made `tensors`: their storages are the last to move."""
    clock = _CLOCK[0] = _CLOCK[0] + 1
    for value in tensors:
        if getattr(value, "pending", False):  # a pending result has no storage yet
            continue
        try:
            storage = value.untyped_storage()
        except RuntimeError:  # a sparse or wrapper tensor has no storage of its own
            continue
        if storage.nbytes() >= SMALLEST:
            USED[storage.data_ptr()] = (clock, weakref.ref(value))
    if len(USED) > FORGET:
        for key in [k for k, (at, _) in USED.items() if at < clock - FORGET]:
            del USED[key]


def hosted(value: Tensor) -> bool:
    """Whether `value`'s bytes are in a host block."""
    if not HELD or getattr(value, "pending", False):
        return False
    try:
        return value.untyped_storage().data_ptr() in HELD
    except RuntimeError:  # a sparse or wrapper tensor has no storage of its own
        return False


def empty(torch: ModuleType, device: Device, like: Tensor) -> Tensor:
    """A CUDA tensor of `like`'s shape, strides and dtype whose bytes are a host block."""
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
    started = time.perf_counter_ns()
    torch.cuda.synchronize(device)
    segments = sorted(
        (s["address"], s["address"] + s["total_size"])
        for s in torch.cuda.memory_snapshot()
        if s["device"] == device.index
    )
    starts = [start for start, _ in segments]
    read = {int(t.untyped_storage().data_ptr()) for t in _tensors(operands)}
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


def _move(torch: ModuleType, device: Device, storage: UntypedStorage) -> int:
    nbytes = int(storage.nbytes())
    host = _Host(torch, nbytes)
    host.data[:nbytes].copy_(torch.empty(0, dtype=torch.uint8, device=device).set_(storage))
    alias = torch.as_tensor(host, device=device)
    storage._swap_data_ptr_(alias.untyped_storage())
    return nbytes


def _tensors(operands: Iterable[object]) -> Iterator[Tensor]:
    for value in operands:
        for item in value if isinstance(value, (tuple, list)) else (value,):
            if type(item).__name__ in ("Tensor", "Parameter") and getattr(item, "is_cuda", False):
                yield cast("Tensor", item)
