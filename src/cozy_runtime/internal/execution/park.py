"""Park, the last rung at a pure cut site (varena-slim §8 Q2/Q3, varena-leases §8; TensorFS 0.13's
rung 5): once its finest level is refused, the idle tensors this request's frames hold move to
page-locked host memory the GPU maps, in place, and the site runs again.

A parked storage keeps its tensors and views: only its bytes move, read over the bus until it comes
back to the device at its next pure site that reads it (`restore`), or until it dies. Park never
runs on a capturing thread or outside a pure site (`catch.cut`), and moves only what it can prove
is a request's activation: a plain tensor a frame holds, made by torch's allocator, read by no
graph (`graphs._LENT`), no loaded model's parameter, buffer or attribute, and no block an earlier
request outlived. Its pages are bounded by measured host headroom: no floor, but pinned pages
never page out, so it leaves a tenth of the memory the process can reach (varena-leases §6).
"""

from __future__ import annotations

import ctypes
import functools
import inspect
import sys
import threading
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import CodeType, FrameType

import torch
from tensorfs import varena
from tensorfs.varena import Stats
from tensorfs.varena.torch import Request
from torch import Tensor, UntypedStorage

from cozy_runtime import graphs

#: a storage smaller than this never parks (the allocator's granule: its bus reads cost more)
SMALLEST = 2 << 20
#: blocks earlier requests' audits reported outliving them, newest last (`ended`)
_OUTLIVED: OrderedDict[int, None] = OrderedDict()
_OUTLIVED_MOST = 4096
#: each parked storage's device address (its host bytes, mapped): its bytes
_PARKED: dict[int, int] = {}
#: host blocks whose storage died, freed once the device passed their readers (`_reclaim`)
_DEAD: list[int] = []
#: per request (by identity): bytes parked, bytes brought back
_MOVED: dict[int, list[int]] = {}
_LOCK = threading.Lock()


def held() -> bool:
    """Something is parked: a pure site's entry looks for its reads among them."""
    return bool(_PARKED)


def park(request: Request, reads: Iterable[object], need: int, block: CodeType) -> int:
    """At a pure site whose finest level was refused: idle tensors of the model call it runs in
    (the frames up to the outermost running `block` wrapper) to the host, largest first, until
    `need` bytes left the device or host headroom runs out. Returns the bytes moved."""
    device = _device(reads)
    if device is None:
        return 0
    _reclaim(device)  # also every stream's pending work on what moves
    room, moved, devices = _host_room(), 0, []
    for storage in _idle(request, device, reads, block):
        nbytes = storage.nbytes()
        if moved >= need:
            break
        if nbytes > room:
            continue
        mapped = _Host.make(nbytes, device)
        if mapped is None:
            break
        mapped.copy_(_view(storage, device))
        storage._swap_data_ptr_(mapped.untyped_storage())  # type: ignore[attr-defined]
        devices.append(mapped)  # now the device block: freed after the sync below
        _PARKED[storage.data_ptr()] = nbytes
        room, moved = room - nbytes, moved + nbytes
    if moved:
        torch.cuda.synchronize(device)  # every reader of the device blocks passed, on every stream
        devices.clear()
        with _LOCK:
            _MOVED.setdefault(id(request), [0, 0])[0] += moved
    return moved


def restore(request: Request, reads: Iterable[object], spare: int) -> None:
    """A pure site's parked reads come back to the device while free and cold bytes cover each
    and `spare` (the site's finest need); otherwise they stay mapped, read over the bus."""
    back = 0
    for t in _tensors(reads):
        storage = t.untyped_storage()
        nbytes = _PARKED.get(storage.data_ptr())
        if nbytes is None:
            continue
        room = request.room()
        if room.free + room.cold < nbytes + spare:
            continue
        try:
            device = torch.empty(nbytes, dtype=torch.uint8, device=t.device)
        except torch.OutOfMemoryError:
            continue
        device.copy_(_view(storage, t.device))
        storage._swap_data_ptr_(device.untyped_storage())  # type: ignore[attr-defined]
        back += nbytes
        del device  # the host block: freed once the device passed the copy (`_reclaim`)
    if back:
        with _LOCK:
            _MOVED.setdefault(id(request), [0, 0])[1] += back


def ended(request: Request, stats: Stats) -> tuple[int, int]:
    """At a request's end: its record's outlived blocks never park, dead host blocks go, and
    the bytes it parked and brought back."""
    for o in stats.outlived:
        _OUTLIVED[o.ptr] = None
        _OUTLIVED.move_to_end(o.ptr)
    while len(_OUTLIVED) > _OUTLIVED_MOST:
        _OUTLIVED.popitem(last=False)
    if _DEAD and torch.cuda.is_initialized():
        _reclaim(torch.device("cuda", torch.cuda.current_device()))
    with _LOCK:
        parked, back = _MOVED.pop(id(request), (0, 0))
    return parked, back


class _Host:
    """Page-locked host bytes the GPU maps, of exactly a storage's size (torch's host allocator
    rounds to a power of two): the CUDA face of a parked storage, freed when it dies."""

    def __init__(self, host: int, mapped: int, nbytes: int) -> None:
        self.host, self.mapped = host, mapped
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (mapped, False),
            "version": 3,
        }

    @staticmethod
    def make(nbytes: int, device: torch.device) -> Tensor | None:
        """A CUDA uint8 tensor over `nbytes` of them; None when the driver refuses."""
        cuda = _driver()
        host, mapped = ctypes.c_void_p(), ctypes.c_uint64()
        if cuda.cuMemHostAlloc(ctypes.byref(host), nbytes, _PORTABLE | _DEVICEMAP):
            return None
        if cuda.cuMemHostGetDevicePointer_v2(ctypes.byref(mapped), host, 0):
            cuda.cuMemFreeHost(host)
            return None
        try:
            tensor = torch.as_tensor(_Host(host.value or 0, mapped.value, nbytes), device=device)
        except (RuntimeError, TypeError):  # torch would not take the mapping
            return None
        return tensor if tensor.data_ptr() == mapped.value else None  # else torch copied it

    def __del__(self) -> None:
        _PARKED.pop(self.mapped, None)
        _DEAD.append(self.host)


_PORTABLE, _DEVICEMAP = 1, 2


@functools.cache
def _driver() -> ctypes.CDLL:
    cuda = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    cuda.cuMemHostAlloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint)
    cuda.cuMemHostGetDevicePointer_v2.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)
    cuda.cuMemFreeHost.argtypes = (ctypes.c_void_p,)
    return cuda


def _reclaim(device: torch.device) -> None:
    """Dead host blocks go back to the OS once the device passed every kernel that may read them."""
    torch.cuda.synchronize(device)
    n = len(_DEAD)
    for host in _DEAD[:n]:
        _driver().cuMemFreeHost(host)
    del _DEAD[:n]


def _view(storage: UntypedStorage, device: torch.device) -> Tensor:
    return torch.empty(0, dtype=torch.uint8, device=device).set_(storage)


def _device(reads: Iterable[object]) -> torch.device | None:
    """The site's GPU: its first CUDA read's, else the current one; None off CUDA."""
    if not torch.cuda.is_initialized():
        return None
    for t in _tensors(reads):
        if t.is_cuda:
            return t.device
    return torch.device("cuda", torch.cuda.current_device())


def tensors(value: object, depth: int = 3) -> list[Tensor]:
    """The tensors in `value`, `depth` levels of sequences and mappings deep."""
    if isinstance(value, Tensor):
        return [value]
    if depth and isinstance(value, list | tuple | Mapping):
        items = value.values() if isinstance(value, Mapping) else value
        return [t for item in items for t in tensors(item, depth - 1)]
    return []


def _tensors(values: Iterable[object], depth: int = 2) -> Iterator[Tensor]:
    """The tensors in `values`, and those a callable among them closes over (a `split`'s window
    function), `depth` closures deep."""
    for value in values:
        yield from tensors(value)
        for cell in (getattr(value, "__closure__", None) or ()) if depth else ():
            try:
                contents = cell.cell_contents
            except ValueError:  # an empty cell
                continue
            yield from _tensors((contents,), depth - 1)


def _idle(
    request: Request, device: torch.device, reads: Iterable[object], block: CodeType
) -> list[UntypedStorage]:
    """The storages the model call's frames hold that may park, largest first."""
    skip = {t.untyped_storage().data_ptr() for t in _tensors(reads) if t.is_cuda}
    skip |= _kept(request)
    lent = list(graphs._LENT)
    found: dict[int, UntypedStorage] = {}
    for value in _frames(sys._getframe(2), block):
        for t in tensors(value):
            if type(t) is not Tensor or t.device != device:
                continue
            storage = t.untyped_storage()
            at, nbytes = storage.data_ptr(), storage.nbytes()
            if (
                nbytes < SMALLEST
                or not storage.resizable()  # not torch's allocation: a weight's view, a mapping
                or at in skip
                or at in _PARKED
                or at in _OUTLIVED
                or any(at <= p < at + nbytes for p in lent)
            ):
                continue
            found.setdefault(at, storage)
    return sorted(found.values(), key=lambda s: -s.nbytes())


def _frames(frame: FrameType, block: CodeType) -> Iterator[object]:
    """The locals of `frame` and the frames under it, down to the outermost running `block`
    wrapper (the model call; every frame when none runs). Python 3.12 caches an optimized frame's
    locals in a dict that would hold them past their use, so it is emptied after the read (3.13's
    `f_locals` is a view of the frame itself)."""
    frames: list[FrameType] = []
    at: FrameType | None = frame
    while at is not None:
        frames.append(at)
        at = at.f_back
    outermost = max((i for i, f in enumerate(frames) if f.f_code is block), default=len(frames))
    for f in frames[: outermost + 1]:
        values = f.f_locals
        yield from list(values.values())
        if sys.version_info < (3, 13) and f.f_code.co_flags & inspect.CO_OPTIMIZED:
            values.clear()


def _kept(request: Request) -> set[int]:
    """The storages of every loaded model's parameters, buffers and tensor attributes."""
    kept: set[int] = set()
    for root in list(request.memory.models):
        for module in root.modules():
            for t in tensors(vars(module)):
                try:
                    kept.add(t.untyped_storage().data_ptr())
                except RuntimeError:  # a placeholder: no storage
                    continue
    return kept


def _host_room() -> int:
    """Host bytes Park may pin now: what is available less a tenth of what the process can reach;
    0 where either is unreadable."""
    limit, available = varena.host_limit(), _available()
    if limit is None or available is None:
        return 0
    return max(available - limit // 10, 0)


def _available() -> int | None:
    """Available host memory: the OS's estimate, and within a cgroup v2 limit, that limit less
    what the cgroup holds beyond its reclaimable file pages."""
    if sys.platform == "win32":
        return _windows_available()
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    available = next(
        (
            int(line.split()[1]) << 10
            for line in meminfo.splitlines()
            if line.startswith("MemAvailable:")
        ),
        None,
    )
    if available is None:
        return None
    try:
        path = next(
            line[3:]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        group = Path("/sys/fs/cgroup") / path.strip().lstrip("/")
        limit = (group / "memory.max").read_text().strip()
        if limit != "max":
            current = int((group / "memory.current").read_text())
            stat = dict(line.split() for line in (group / "memory.stat").read_text().splitlines())
            available = min(available, int(limit) - current + int(stat.get("inactive_file", 0)))
    except (OSError, StopIteration, ValueError):
        pass
    return max(available, 0)


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("load", ctypes.c_ulong),
        ("total", ctypes.c_ulonglong),
        ("available", ctypes.c_ulonglong),
        ("page_total", ctypes.c_ulonglong),
        ("page_available", ctypes.c_ulonglong),
        ("virtual_total", ctypes.c_ulonglong),
        ("virtual_available", ctypes.c_ulonglong),
        ("extended", ctypes.c_ulonglong),
    ]


def _windows_available() -> int | None:
    status = _MemoryStatus(length=ctypes.sizeof(_MemoryStatus))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    return int(status.available)
