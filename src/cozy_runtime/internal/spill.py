"""An executor's own tensors on its GPU, moved to page-locked host memory in place.

The last room an op can be given, once every weight it does not read is unmapped and every
idle tenant has given back: the device bytes of the tensors this process holds (a package's
caches, an earlier stage's outputs, a decoder's feature maps). Each storage torch's allocator
holds is copied to page-locked host memory the GPU addresses directly (unified addressing),
and the storage is repointed there. Every tensor over it stays a CUDA tensor, read and
written over the bus from then on, so whatever holds one is unchanged and nothing has to bring
it back; its device bytes go back to torch. Largest first; the op's own operands last.
"""

from __future__ import annotations

import gc
import time
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from types import ModuleType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor, UntypedStorage
    from torch import device as Device

#: Storages smaller than this are not worth a copy.
SMALLEST = 1 << 20
#: `cudaHostRegisterPortable | cudaHostRegisterMapped`
_MAPPED = 3
#: [times tensors moved, their bytes, nanoseconds it took] since the process began
MOVED = [0, 0, 0]


class _Host:
    """A moved storage's bytes: page-locked host memory, as CUDA addresses it."""

    def __init__(self, torch: ModuleType, nbytes: int) -> None:
        self.cudart = torch.cuda.cudart()
        self.data = torch.empty(nbytes, dtype=torch.uint8)
        pointer = self.data.data_ptr()
        torch.cuda.check_error(self.cudart.cudaHostRegister(pointer, nbytes, _MAPPED))
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }

    def __del__(self) -> None:
        self.cudart.cudaHostUnregister(self.data.data_ptr())


def spill(
    torch: ModuleType, device: Device, need: int, operands: Iterable[object], room: int
) -> int:
    """Move this process's largest tensors on `device` to host memory until `need` device
    bytes came back, the storages `operands` read last, at most `room` bytes. Returns the
    bytes moved."""
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
    for value in gc.get_objects():
        if (
            type(value) not in (torch.Tensor, torch.nn.Parameter)
            or value.layout is not torch.strided
            or value.device != device
        ):
            continue
        storage = value.untyped_storage()
        address, at = storage.data_ptr(), bisect_right(starts, storage.data_ptr()) - 1
        if storage.nbytes() >= SMALLEST and at >= 0 and address < segments[at][1]:
            found[address] = storage
    moved = 0
    for _, storage in sorted(found.items(), key=lambda r: (r[0] in read, -r[1].nbytes())):
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
    host.data.copy_(torch.empty(0, dtype=torch.uint8, device=device).set_(storage))
    alias = torch.as_tensor(host, device=device)
    storage._swap_data_ptr_(alias.untyped_storage())  # type: ignore[attr-defined]
    return nbytes


def _tensors(operands: Iterable[object]) -> Iterator[Tensor]:
    for value in operands:
        for item in value if isinstance(value, (tuple, list)) else (value,):
            if type(item).__name__ in ("Tensor", "Parameter") and getattr(item, "is_cuda", False):
                yield cast("Tensor", item)
