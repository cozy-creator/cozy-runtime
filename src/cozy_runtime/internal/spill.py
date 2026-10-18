"""An executor's own tensors off its GPU: memory rungs 4 and 5, after every weight an op does
not read is unmapped and every idle tenant gave back. Least recently used first (`touch`), the
op's operands last, a storage leaves two ways:

- Mapped: repointed in place to page-locked host memory the GPU reads over the bus, so whatever
  holds its tensors, native code included, is unchanged. While the host can give it.
- Out: one no op touched since its stage began (`began`) is copied to pageable host memory, or
  to disk once the host can give no more (rung 5), and its device bytes are released; the
  dispatch mode brings it back before an op reads it (`fault`), and a launch outside it before
  the kernel does (`arrive`). Never a CUDA graph pool's, nor what a graph holds (`home`): its
  graphs read it unseen. Unified memory (Apple) has only this way.

Ops are noted only under pressure, from the first move until nothing is off the device. Locking
new pages runs at about 1.6 GB/s: a freed block stays locked for the next of its size until the
stage closes.
"""

from __future__ import annotations

import gc
import mmap
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, cast

from cozy_runtime.author._errors import is_device_oom
from cozy_runtime.author._native import NATIVE_READS

if TYPE_CHECKING:
    from io import BufferedRandom

    from torch import Tensor, UntypedStorage
    from torch import device as Device
    from torch.multiprocessing.reductions import StorageWeakRef

#: `Memory.retry`: an attempt, what it is, its operands
type Retry = Callable[[Callable[[], None], str, Sequence[object]], None]

#: Storages smaller than this are not worth a copy.
SMALLEST = 1 << 20
#: `cudaHostRegisterPortable | cudaHostRegisterMapped`
_MAPPED = 3
#: Host blocks are locked in multiples of this, so maps of about one size share them.
GRAIN = 2 << 20
#: cudaMalloc's alignment, which CUDA libraries take for granted (cuBLASLt answers INVALID_VALUE
#: for a workspace off it: run 9141); torch's host allocator gives 64.
ALIGN = 256
#: since the process began: [moves, their bytes, nanoseconds they took, bytes mapped, bytes
#: out to host memory, bytes out to disk, bytes back on the device (`restore`, `fault`)]
TOTALS = [0, 0, 0, 0, 0, 0, 0]
#: locked blocks no tensor holds, by size; and the address of every block one holds -> its size
_FREE: dict[int, list[Tensor]] = {}
HELD: dict[int, int] = {}
#: storage -> (when an op last read or made it, in ops; a weak reference to it) (`touch`)
USED: dict[int, tuple[int, StorageWeakRef]] = {}
#: USED's size past which its dead storages are swept out
_SWEEP = [1 << 12]
_CLOCK = [0]
#: the clock when the outermost stage began (`began`)
BEGAN = [0]
#: whether ops are noted (`touch`): from a stage's first move until nothing is off the device
PRESSED = [False]


@dataclass(slots=True)
class Off:
    """A storage off its device: mapped in place, or out, its bytes a pageable host copy or a
    file."""

    ref: StorageWeakRef  # the storage goes with its last tensor
    nbytes: int
    host: Tensor | None = None
    file: BufferedRandom | None = None

    @property
    def out(self) -> bool:
        return self.host is not None or self.file is not None


#: every storage off its device, by `_cdata`
OFF: dict[int, Off] = {}
#: how many of them are out: a native read has work only while there is one, or pressure
OUT = [0]
#: the torch a move ran with, and the memory ladder that makes room for a storage's return
#: (`Memory.retry`, weakly: the process's GPU executor sets it)
_TORCH: list[ModuleType] = []
RETRY: list[Callable[[], Retry | None]] = []
#: storages code outside the dispatch mode reads (`native`), by `_cdata`
NATIVE: dict[int, StorageWeakRef] = {}


def _size(nbytes: int) -> int:
    return -(-max(nbytes, 1) // GRAIN) * GRAIN


def reusable(nbytes: int) -> bool:
    """Whether a free block can take `nbytes` without locking more."""
    return bool(_FREE.get(_size(nbytes)))


class _Host:
    """A host block as CUDA addresses it; freed, it waits for the next one of its size."""

    def __init__(self, data: Tensor, size: int, nbytes: int) -> None:
        self.size, self.data = size, data
        start = -data.data_ptr() % ALIGN
        self.view = data[start : start + size]
        pointer = self.view.data_ptr()
        HELD[pointer] = size
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }

    def __del__(self) -> None:
        HELD.pop(self.view.data_ptr(), None)
        _FREE.setdefault(self.size, []).append(self.data)


def _host(torch: ModuleType, nbytes: int) -> _Host:
    """A page-locked block for `nbytes`: a free one of its size, else newly locked."""
    size = _size(nbytes)
    if free := _FREE.get(size):
        return _Host(free.pop(), size, nbytes)
    data = torch.empty(size + ALIGN, dtype=torch.uint8)
    status = torch.cuda.cudart().cudaHostRegister(data.data_ptr(), size + ALIGN, _MAPPED)
    if status != torch.cuda.cudart().cudaError.success:
        with suppress(RuntimeError):  # CUDA's last error until a launch's check reads it
            torch.empty(1, device="cuda").zero_()
        torch.cuda.check_error(status)
    return _Host(data, size, nbytes)


def release(torch: ModuleType) -> None:
    """A stage closed: free blocks go back to the host, and the copies of storages no tensor
    holds any more are deleted. Once nothing is off the device, ops are no longer noted."""
    _sweep()
    if not OFF:
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
    """The tensors among `values` (and their sequences) whose storage holds their bytes: a
    plain strided tensor, or a traceable wrapper's inner ones (an AsyncCollectiveTensor, a
    DTensor). A wrapper's own storage has no bytes (run 6349: `data_ptr` raises on it), nor
    has another subclass's (a weight's unbound stand-in), a meta or a sparse tensor's."""
    for item in values:
        if isinstance(item, (tuple, list)):
            yield from stored(item)
        elif type(item).__name__ in ("Tensor", "Parameter"):
            tensor = cast("Tensor", item)
            if not tensor.is_meta and str(tensor.layout) == "torch.strided":
                yield tensor
        elif (flatten := getattr(item, "__tensor_flatten__", None)) is not None:
            yield from stored([getattr(item, name) for name in flatten()[0]])


def native(torch: ModuleType, *values: object) -> None:
    """Code outside the dispatch mode reads these tensors' storages (a CUDA graph's static
    inputs, an LM engine's buffers under its graphs, a plane's weights): they never leave the
    device, since neither a pointer it holds nor a read it makes would follow them."""
    for tensor in stored(values):
        storage = tensor.untyped_storage()
        NATIVE[int(storage._cdata)] = torch.multiprocessing.reductions.StorageWeakRef(storage)


def home(torch: ModuleType, values: tuple[object, ...]) -> None:
    """`values` on their device for good, before code that keeps their addresses (a CUDA graph's
    capture): back from wherever they are, and never moved again (`native`)."""
    if OFF:
        with torch.utils._python_dispatch._disable_current_modes():
            fault(torch, values, RETRY[0]() if RETRY else None, mapped=True)
    native(torch, *values)


def arrive(values: tuple[object, ...]) -> None:
    """Native code (a Triton or CUDA launch, a Kitchen call) reads `values` by pointer: a storage
    out of the device comes back first, and under pressure they count as read, as an op's
    operands do, so a tensor only kernels read never looks idle. Constant time while nothing
    is out and nothing is pressed."""
    if not (OUT[0] or PRESSED[0]) or not _TORCH:
        return
    torch = _TORCH[0]
    with torch.utils._python_dispatch._disable_current_modes():
        if OUT[0]:
            fault(torch, values, RETRY[0]() if RETRY else None)
        touch(torch, values)


NATIVE_READS.append(arrive)


def began() -> None:
    """The outermost stage begins: what no op touches from now on may leave the device whole."""
    BEGAN[0] = _CLOCK[0] + 1


def touch(torch: ModuleType, values: Iterable[object]) -> None:
    """An op read or made `values`: their storages are the last to move, and not out while
    this stage touches them."""
    if not PRESSED[0]:
        return
    clock = _CLOCK[0] = _CLOCK[0] + 1
    for value in stored(values):
        storage = value.untyped_storage()
        if storage.nbytes() < SMALLEST:
            continue
        seen = USED.get(key := storage._cdata)
        weak = torch.multiprocessing.reductions.StorageWeakRef
        USED[key] = (clock, seen[1] if seen is not None else weak(storage))
    if len(USED) > _SWEEP[0]:
        for key in [key for key, (_, ref) in USED.items() if ref.expired()]:
            del USED[key]
        _SWEEP[0] = max(2 * len(USED), 1 << 12)


def hosted(value: object) -> bool:
    """Whether `value`'s bytes are in a host block."""
    return bool(HELD) and any(t.untyped_storage().data_ptr() in HELD for t in stored((value,)))


def summary(since: Sequence[int]) -> str:
    """What moved since `TOTALS` read `since`: one call's moves, not the process's."""
    n, moved, ns, mapped, host, disk, back = (a - b for a, b in zip(TOTALS, since, strict=True))
    return (
        f"{n}x {moved} B in {ns / 1e9:.1f} s ({mapped} B mapped, {host} B to host, {disk} B "
        f"to disk), {back} B back"
    )


def empty(torch: ModuleType, device: Device, like: Tensor) -> Tensor:
    """A CUDA tensor of `like`'s shape, strides and dtype whose bytes are a host block."""
    PRESSED[0] = True
    span = 1 + sum((n - 1) * s for n, s in zip(like.shape, like.stride(), strict=True))
    nbytes = span * like.element_size() if like.numel() else 0
    raw = torch.as_tensor(_host(torch, nbytes), device=device)
    _note(torch, raw.untyped_storage())
    return raw.view(like.dtype).as_strided(like.shape, like.stride())


def keep[T: (Tensor, UntypedStorage)](value: T) -> T:
    """A copy of what an op or a journal writes into, put back if it runs again: in host
    memory under pressure, where a device copy would take the room the attempt lacks."""
    return value.cpu() if PRESSED[0] and value.device.type != "cpu" else value.clone()


def spill(
    torch: ModuleType,
    device: Device,
    need: int,
    operands: Iterable[object],
    room: int,
    mapped: bool = True,
    disk: bool = False,
) -> int:
    """This process's storages off `device` until `need` bytes came back, taking at most `room`
    of host memory (-1: unknown): mapped while it locks (`mapped`), else out to it, and with
    `disk` to disk past it. Returns the bytes moved."""
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return 0  # a capture can neither wait for the device nor copy from pageable memory
    PRESSED[0] = True
    _TORCH[:] = [torch]
    started = time.perf_counter_ns()
    _sweep()
    _fence(torch, device)
    read = {int(t.untyped_storage()._cdata) for t in stored(operands) if t.device == device}
    left, moved, disk = room if room >= 0 else need, 0, disk and room >= 0

    def movable(key: int, nbytes: int) -> bool:
        out = key not in read and _last(key) < BEGAN[0]
        return key not in read and ((nbytes <= left and (mapped or out)) or (disk and out))

    found = _found(torch, device, need, movable)
    for key, storage in sorted(
        found.items(), key=lambda r: (r[0] in read, _last(r[0]), -r[1].nbytes())
    ):
        if moved >= need:
            break
        nbytes, out = int(storage.nbytes()), key not in read and _last(key) < BEGAN[0]
        if nbytes <= left and mapped:
            try:
                _map(torch, device, storage)
            except RuntimeError:  # the host locks no more: what can leave whole does
                mapped = False
            else:
                left, moved, TOTALS[3] = left - nbytes, moved + nbytes, TOTALS[3] + nbytes
                continue
        if out and (nbytes <= left or disk):
            on_disk = nbytes > left
            _out(torch, storage, on_disk)
            left -= 0 if on_disk else nbytes
            moved += nbytes
            TOTALS[5 if on_disk else 4] += nbytes
    TOTALS[0] += moved > 0
    TOTALS[1] += moved
    TOTALS[2] += time.perf_counter_ns() - started
    return moved


def fault(
    torch: ModuleType,
    operands: Sequence[object],
    retry: Retry | None,
    mapped: bool = False,
) -> None:
    """Storages among an op's `operands` that are out (and with `mapped`, those mapped too)
    come back before it runs, each allocation making its room as any op's does (`retry`)."""
    for tensor in stored(operands):
        storage = tensor.untyped_storage()
        off = OFF.get(storage._cdata)
        if off is None or not (off.out or mapped):
            continue
        back = partial(_in, torch, storage, off)
        if retry is None:
            back()
        else:
            retry(back, f"a storage of {off.nbytes} B back from host", operands)


def restore(torch: ModuleType, device: Device, room: int) -> int:
    """The storages off `device` come back while `room` bytes allow, largest first: a cache or a
    decoder's state left mapped is read over the bus for as long as it lives. Returns the bytes
    restored."""
    _sweep()
    back = 0
    for off in sorted(OFF.values(), key=lambda off: -off.nbytes):
        storage = _strong(torch, off.ref)
        if storage is None or storage.device != device or back + off.nbytes > room:
            continue
        if back == 0:
            _fence(torch, device)
        try:
            _in(torch, storage, off)
        except Exception as exc:
            if not is_device_oom(exc):
                raise
            break  # less room than the driver said: the rest come back when an op reads them
        back += off.nbytes
    return back


def _flat(torch: ModuleType, storage: UntypedStorage) -> Tensor:
    return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)


def _fence(torch: ModuleType, device: Device) -> None:
    """Every stream that touched the device's bytes is done with them: the one wait before any
    of them is released or repointed (#928)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _found(
    torch: ModuleType, device: Device, need: int, movable: Callable[[int, int], bool]
) -> dict[int, UntypedStorage]:
    """Storages torch's allocator holds on `device` for this process (resizable: neither a
    plane's nor a host block's), by `_cdata`. Every object is looked at when the `movable`
    ones ops touched cannot give `need`."""
    found: dict[int, UntypedStorage] = {}

    def add(storages: Iterable[UntypedStorage | None]) -> int:
        """Those among `storages`; the bytes of those that may leave."""
        spare = 0
        for storage in storages:
            if storage is None or storage.device != device or not storage.resizable():
                continue
            key = int(storage._cdata)
            if key in found or key in OFF or key in NATIVE or storage.nbytes() < SMALLEST:
                continue
            found[key] = storage
            spare += int(storage.nbytes()) if movable(key, int(storage.nbytes())) else 0
        return spare

    if add(_strong(torch, ref) for _, ref in list(USED.values())) < need:
        add(
            value.untyped_storage()
            for value in gc.get_objects()
            if type(value) in (torch.Tensor, torch.nn.Parameter)
            and value.layout is torch.strided
            and value.device == device
        )
    return found


def _strong(torch: ModuleType, ref: StorageWeakRef) -> UntypedStorage | None:
    if ref.expired():
        return None
    return cast("UntypedStorage | None", torch.UntypedStorage._new_with_weak_ptr(ref.cdata))


def _last(key: int) -> int:
    """When an op last touched the storage (-1: none since ops were noted). A stage's first
    move knows only what ops touched since then: a live tensor may leave once, and its next op
    brings it back to stay."""
    seen = USED.get(key)
    return seen[0] if seen is not None else -1


def _note(
    torch: ModuleType,
    storage: UntypedStorage,
    host: Tensor | None = None,
    file: BufferedRandom | None = None,
) -> None:
    weak = torch.multiprocessing.reductions.StorageWeakRef(storage)
    OFF[int(storage._cdata)] = off = Off(weak, int(storage.nbytes()), host, file)
    OUT[0] += off.out
    _TORCH[:] = [torch]


def _map(torch: ModuleType, device: Device, storage: UntypedStorage) -> None:
    nbytes = int(storage.nbytes())
    host = _host(torch, nbytes)
    host.view[:nbytes].copy_(_flat(torch, storage))
    alias = torch.as_tensor(host, device=device)
    storage._swap_data_ptr_(alias.untyped_storage())
    _note(torch, storage)


def _out(torch: ModuleType, storage: UntypedStorage, disk: bool) -> None:
    """The storage's bytes to pageable host memory or a file, its device bytes released (the
    device is fenced: `spill`)."""
    flat, nbytes = _flat(torch, storage), int(storage.nbytes())
    if disk:
        # In the executor's scratch scope on the machine's disk (`TMPDIR`), never RAM-backed;
        # unnamed, so it goes with the process at the latest.
        file = tempfile.TemporaryFile()  # noqa: SIM115 - held until the storage is back
        os.ftruncate(file.fileno(), nbytes)
        with mmap.mmap(file.fileno(), nbytes) as pages:
            torch.frombuffer(pages, dtype=torch.uint8).copy_(flat)
            pages.flush()
        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)  # its pages leave RAM
        _note(torch, storage, file=file)
    else:
        _note(torch, storage, torch.empty(nbytes, dtype=torch.uint8).copy_(flat))
    storage.resize_(0)


def _in(torch: ModuleType, storage: UntypedStorage, off: Off) -> None:
    """A storage's bytes back on its device: a mapped one copied and repointed in place (its
    host block free again), an out one allocated and filled. The allocation is the only step
    that can fail, and it fails before anything changed."""
    if not off.out:
        fresh = torch.empty(off.nbytes, dtype=torch.uint8, device=storage.device)
        storage._swap_data_ptr_(fresh.copy_(_flat(torch, storage)).untyped_storage())
    else:
        storage.resize_(off.nbytes)
    flat = _flat(torch, storage)
    if off.host is not None:
        flat.copy_(off.host)
    elif (file := off.file) is not None:
        with mmap.mmap(file.fileno(), off.nbytes) as pages:
            flat.copy_(torch.frombuffer(pages, dtype=torch.uint8))
        file.close()
    del OFF[storage._cdata]
    OUT[0] -= off.out
    TOTALS[6] += off.nbytes


def _sweep() -> None:
    """What was kept for storages no tensor holds any more goes."""
    for key in [key for key, off in OFF.items() if off.ref.expired()]:
        off = OFF.pop(key)
        OUT[0] -= off.out
        if off.file is not None:
            off.file.close()
    for key in [key for key, ref in NATIVE.items() if ref.expired()]:
        del NATIVE[key]
