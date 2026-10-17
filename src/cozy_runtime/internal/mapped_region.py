"""Regions read in place on an integrated CUDA GPU (GB10: `accel.memory_class` unified).

The store's object files are mapped into one region's address range, registered read-only with
the driver and handed to torch as a device tensor through DLPack: the bytes live once, in the
page cache, and the GPU reads them there. GB10 reads no pageable memory (its
PAGEABLE_MEMORY_ACCESS is 0), so the pages are registered; a registered page's host address is
its device address there (CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM). The registration and the
mapping go when torch frees the last view of the region.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

PAGE = mmap.PAGESIZE

_PROT_READ, _PROT_WRITE = 0x1, 0x2
_MAP_SHARED, _MAP_PRIVATE, _MAP_FIXED, _MAP_ANONYMOUS = 0x01, 0x02, 0x10, 0x20
_FAILED = ctypes.c_void_p(-1).value
_REGISTER_MAPPED, _REGISTER_READ_ONLY = 0x02, 0x08
# CUdevice_attribute
_INTEGRATED, _HOST_POINTER_FOR_REGISTERED, _REGISTER_READ_ONLY_SUPPORTED = 18, 91, 113

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = (
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
)
_libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)


def _mmap(at: int | None, nbytes: int, prot: int, flags: int, fd: int = -1, off: int = 0) -> int:
    got = _libc.mmap(at, nbytes, prot, flags, fd, off)
    if got is None or got == _FAILED:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return int(got)


def supported(index: int) -> bool:
    """An integrated GPU whose registered host pages are device pointers, registrable read-only."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return False
    device, value = ctypes.c_int(), ctypes.c_int()
    if cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(device), index):
        return False
    for attribute in (_INTEGRATED, _HOST_POINTER_FOR_REGISTERED, _REGISTER_READ_ONLY_SUPPORTED):
        if cuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, device) or value.value != 1:
            return False
    return True


class _Device(ctypes.Structure):
    _fields_ = (("type", ctypes.c_int), ("id", ctypes.c_int))


class _DType(ctypes.Structure):
    _fields_ = (("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16))


class _DLTensor(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.c_void_p),
        ("device", _Device),
        ("ndim", ctypes.c_int),
        ("dtype", _DType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    )


class _Managed(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_Managed))
_Managed._fields_ = (
    ("tensor", _DLTensor),
    ("context", ctypes.c_void_p),
    ("deleter", _DELETER),
)

_capsule = ctypes.pythonapi.PyCapsule_New
_capsule.restype = ctypes.py_object
_capsule.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)

#: address of each live region's DLManagedTensor -> what keeps it valid, and how it goes
_live: dict[int, tuple[_Managed, object, Callable[[], None]]] = {}
_lock = threading.Lock()


@_DELETER
def _release(managed: ctypes._Pointer[_Managed]) -> None:  # type: ignore[name-defined]
    with _lock:
        held = _live.pop(ctypes.addressof(managed.contents), None)
    if held is not None:
        held[2]()


#: (fd, file offset, length, offset in the region); both offsets on a page
type Piece = tuple[int, int, int, int]


def region(
    torch: ModuleType,
    index: int,
    span: int,
    pieces: Sequence[Piece],
    rest: Callable[[memoryview], None],
) -> Tensor:
    """`span` bytes as a uint8 tensor on GPU `index`: each piece's file bytes mapped in place at
    its offset, the rest (inline parts) written by `rest` into the anonymous pages around them.
    `rest` must write only outside the pieces' pages: those are read-only file pages."""
    base = _mmap(None, span, _PROT_READ | _PROT_WRITE, _MAP_PRIVATE | _MAP_ANONYMOUS)
    cudart = torch.cuda.cudart()
    try:
        for fd, off, nbytes, at in pieces:
            pages = -(-nbytes // PAGE) * PAGE
            _mmap(base + at, pages, _PROT_READ, _MAP_SHARED | _MAP_FIXED, fd, off)
        rest(memoryview((ctypes.c_char * span).from_address(base)).cast("B"))
        torch.cuda.check_error(
            cudart.cudaHostRegister(base, span, _REGISTER_MAPPED | _REGISTER_READ_ONLY)
        )
    except BaseException:
        _libc.munmap(base, span)
        raise

    def close() -> None:
        # Kernels still reading the pages finish before they go.
        torch.cuda.synchronize(index)
        cudart.cudaHostUnregister(base)
        _libc.munmap(base, span)

    shape = (ctypes.c_int64 * 1)(span)
    managed = _Managed(
        _DLTensor(
            base,
            _Device(2, index),  # kDLCUDA
            1,
            _DType(1, 8, 1),  # uint8
            ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64)),
            None,
            0,
        )
    )
    managed.deleter = _release
    with _lock:
        _live[ctypes.addressof(managed)] = (managed, shape, close)
    return torch.utils.dlpack.from_dlpack(_capsule(ctypes.addressof(managed), b"dltensor", None))
