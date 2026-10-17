"""An executor's own tensors off its GPU, the last room an op can be given (memory rungs 4-5).

Once every weight an op does not read is unmapped and every idle tenant has given back, the
device bytes left are the tensors this process holds (a package's caches, an earlier stage's
outputs, a decoder's feature maps). They leave least recently used first (`touch`), the op's
own operands last, two ways:

- Mapped: copied to page-locked host memory the GPU addresses directly, the storage repointed
  there. Every tensor over it stays a CUDA tensor read over the bus, so whatever holds one,
  native code included, is unchanged. While the host can lock it (`Weights.lockable`).
- Out: a storage no op has touched since its stage began (`began`: an earlier stage's state,
  a decoder's leftovers) is copied to pageable host memory, or to a file on the machine's disk
  once the host can give no more (rung 5), and its device bytes are released; the dispatch mode
  brings it back before an op touches it (`fault`). Never a CUDA graph pool's: its graphs read
  it without the mode seeing them. Unified memory (Apple) has only this way.

Ops note what they touch only under pressure, from a stage's first move until nothing is off
the device: a stage with room pays nothing per op. Locking new pages runs at about 1.6 GB/s,
most of a mapped move's cost: a block no tensor holds any more stays locked for the next move
or host result of its size, until the stage closes.
"""

from __future__ import annotations

import gc
import os
import tempfile
import time
from bisect import bisect_right
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from io import BufferedRandom

    from torch import Tensor, UntypedStorage
    from torch import device as Device
    from torch.multiprocessing.reductions import StorageWeakRef

#: Storages smaller than this are not worth a copy.
SMALLEST = 1 << 20
#: `cudaHostRegisterPortable | cudaHostRegisterMapped`
_MAPPED = 3
#: Host blocks are locked in multiples of this, so maps of about one size share them.
GRAIN = 2 << 20
#: cudaMalloc's alignment, which CUDA libraries take for granted (cuBLASLt answers INVALID_VALUE
#: for a workspace off it: run 9141); torch's host allocator gives 64.
ALIGN = 256
#: A move to or from disk crosses host memory in pieces of this size.
CHUNK = 64 << 20
#: [times tensors moved, their bytes, nanoseconds it took] since the process began
MOVED = [0, 0, 0]
#: bytes moved [mapped, out to host memory, out to disk] since the process began
TIERS = [0, 0, 0]
#: [times tensors came back (`restore`, `fault`), their bytes] since the process began
RESTORED = [0, 0]
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


#: every storage off its device, by `_cdata`
OFF: dict[int, Off] = {}


def _size(nbytes: int) -> int:
    return -(-max(nbytes, 1) // GRAIN) * GRAIN


def reusable(nbytes: int) -> bool:
    """Whether a free block can take `nbytes` without locking more."""
    return bool(_FREE.get(_size(nbytes)))


def _locked(torch: ModuleType, size: int) -> Tensor:
    """A page-locked block of `size` bytes (and the alignment's slack): a free one, else new."""
    free = _FREE.get(size)
    if free:
        return free.pop()
    data = torch.empty(size + ALIGN, dtype=torch.uint8)
    status = torch.cuda.cudart().cudaHostRegister(data.data_ptr(), size + ALIGN, _MAPPED)
    if status != torch.cuda.cudart().cudaError.success:
        with suppress(RuntimeError):  # CUDA's last error until a launch's check reads it
            torch.empty(1, device="cuda").zero_()
        torch.cuda.check_error(status)
    return data


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
    size = _size(nbytes)
    return _Host(_locked(torch, size), size, nbytes)


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


def summary() -> str:
    seconds = MOVED[2] / 1e9
    return (
        f"{MOVED[0]}x {MOVED[1]} B in {seconds:.1f} s: {TIERS[0]} B mapped, {TIERS[1]} B to "
        f"host, {TIERS[2]} B to disk; {RESTORED[1]} B back"
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
    placed: Collection[int] = (),
) -> int:
    """Move this process's storages on `device` off it until `need` device bytes came back,
    least recently used first and `operands` last, taking at most `room` bytes of host memory
    (-1: unknown, as much as `need`): mapped while the host locks it (`mapped`), else out to
    host memory, and, with `disk`, out to disk once the host can give no more. `placed`:
    storages (`_cdata`) another owner places (a unified plane's weights). Returns the bytes
    moved."""
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return 0  # a capture can neither wait for the device nor copy from pageable memory
    PRESSED[0] = True
    started = time.perf_counter_ns()
    _sweep()
    _fence(torch, device)
    read = {int(t.untyped_storage()._cdata) for t in stored(operands) if t.device == device}
    found = _found(torch, device, need, read, placed)
    left, moved, disk = room if room >= 0 else need, 0, disk and room >= 0
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
                left, moved, TIERS[0] = left - nbytes, moved + nbytes, TIERS[0] + nbytes
                continue
        if out and (nbytes <= left or disk):
            on_disk = nbytes > left
            _out(torch, storage, on_disk)
            left -= 0 if on_disk else nbytes
            moved += nbytes
            TIERS[2 if on_disk else 1] += nbytes
    MOVED[0] += moved > 0
    MOVED[1] += moved
    MOVED[2] += time.perf_counter_ns() - started
    return moved


def fault(
    torch: ModuleType,
    operands: Sequence[object],
    retry: Callable[[Callable[[], None], str, Sequence[object]], None] | None,
) -> None:
    """Storages among an op's `operands` that are out come back before it runs, each
    allocation making its room as any op's does (`retry`)."""
    for tensor in stored(operands):
        storage = tensor.untyped_storage()
        off = OFF.get(storage._cdata)
        if off is None or (off.host is None and off.file is None):
            continue
        back = partial(_in, torch, storage, off)
        if retry is None:
            back()
        else:
            retry(back, f"a storage of {off.nbytes} B back from host", operands)


def restore(torch: ModuleType, device: Device, room: int) -> int:
    """The storages off `device` come back while `room` bytes allow, largest first: a mapped
    one is copied back and repointed in place (its host block free again), an out one brought
    in. A tensor left mapped is read over the bus for as long as it lives, and a cache or a
    decoder's state lives across calls. Returns the bytes restored."""
    _sweep()
    back = 0
    for off in sorted(OFF.values(), key=lambda off: -off.nbytes):
        storage = _strong(torch, off.ref)
        if storage is None or storage.device != device or back + off.nbytes > room:
            continue
        if back == 0:
            _fence(torch, device)
        if off.host is None and off.file is None:
            fresh = torch.empty(off.nbytes, dtype=torch.uint8, device=device)
            fresh.copy_(_flat(torch, storage))
            # `fresh` takes the host pointer and frees its block when it goes
            storage._swap_data_ptr_(fresh.untyped_storage())
            del OFF[storage._cdata]
            RESTORED[0] += 1
            RESTORED[1] += off.nbytes
        else:
            _in(torch, storage, off)
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
    torch: ModuleType, device: Device, need: int, read: set[int], placed: Collection[int]
) -> dict[int, UntypedStorage]:
    """Storages torch's allocator holds on `device` for this process, by `_cdata`: in its own
    pool (a CUDA graph's pool holds what its graphs read) or, without segments, resizable."""
    segments: list[tuple[int, int]] | None = None
    if device.type == "cuda":
        segments = sorted(
            (s["address"], s["address"] + s["total_size"])
            for s in torch.cuda.memory_snapshot()
            if s["device"] == device.index and tuple(s.get("segment_pool_id", (0, 0))) == (0, 0)
        )
    starts = [start for start, _ in segments or ()]
    found: dict[int, UntypedStorage] = {}

    def add(storages: Iterable[UntypedStorage | None]) -> int:
        """Those among `storages`; the bytes of those no operand reads."""
        spare = 0
        for storage in storages:
            if storage is None or storage.device != device:
                continue
            key = int(storage._cdata)
            if key in found or key in placed or storage.nbytes() < SMALLEST:
                continue
            if segments is None:
                if not storage.resizable():
                    continue
            else:
                address = storage.data_ptr()
                at = bisect_right(starts, address) - 1
                if at < 0 or address >= segments[at][1]:
                    continue
            found[key] = storage
            spare += 0 if key in read else int(storage.nbytes())
        return spare

    # The storages ops touched cover a decode's maps; every object only when they fall short.
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
    OFF[int(storage._cdata)] = Off(weak, int(storage.nbytes()), host, file)


def _chunks(torch: ModuleType, nbytes: int) -> Iterator[tuple[slice, Tensor]]:
    """A disk copy's pieces: each one's span, and host memory to cross in."""
    staging = torch.empty(min(CHUNK, nbytes), dtype=torch.uint8)
    for at in range(0, nbytes, CHUNK):
        part = staging[: min(CHUNK, nbytes - at)]
        yield slice(at, at + part.numel()), part


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
    flat = _flat(torch, storage)
    if disk:
        # In the executor's scratch scope on the machine's disk (`TMPDIR`), never RAM-backed;
        # unnamed, so it goes with the process at the latest.
        file = tempfile.TemporaryFile()  # noqa: SIM115 - held until the storage is back
        for span, part in _chunks(torch, flat.numel()):
            file.write(part.copy_(flat[span]).numpy().data)
        file.flush()
        os.fdatasync(file.fileno())
        os.posix_fadvise(file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)  # its pages leave RAM
        _note(torch, storage, file=file)
    else:
        _note(torch, storage, torch.empty(flat.numel(), dtype=torch.uint8).copy_(flat))
    storage.resize_(0)


def _in(torch: ModuleType, storage: UntypedStorage, off: Off) -> None:
    """An out storage's bytes back on its device. Its allocation is the only step that can
    fail, and it fails before anything changed."""
    storage.resize_(off.nbytes)
    flat = _flat(torch, storage)
    if off.host is not None:
        flat.copy_(off.host)
    elif (file := off.file) is not None:
        file.seek(0)
        for span, part in _chunks(torch, off.nbytes):
            file.readinto(part.numpy().data)
            flat[span].copy_(part)
        file.close()
    del OFF[storage._cdata]
    RESTORED[0] += 1
    RESTORED[1] += off.nbytes


def _sweep() -> None:
    """What was kept for storages no tensor holds any more goes."""
    for key in [key for key, off in OFF.items() if off.ref.expired()]:
        if (file := OFF.pop(key).file) is not None:
            file.close()
