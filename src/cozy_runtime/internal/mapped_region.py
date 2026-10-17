"""Regions read in place on an integrated CUDA GPU (GB10: `accel.memory_class` unified).

The store's object files are mapped into one region's address range and handed to torch as a
device tensor through DLPack: the bytes live once, in the page cache, and the GPU reads them
there. A GPU that walks the CPU's page tables (ATS: PAGEABLE_MEMORY_ACCESS with
USES_HOST_PAGE_TABLES, GB10 on driver 595) reads the mapping as it is; one that does not (GB10 on
driver 580) reads it once the pages are registered read-only, a registered page's host address
being its device address (CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM). The mapping (and any
registration) goes when torch frees the last view of the region.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import threading
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch import Tensor

PAGE = mmap.PAGESIZE

_PROT_READ, _PROT_WRITE = 0x1, 0x2
_MAP_SHARED, _MAP_PRIVATE, _MAP_FIXED, _MAP_ANONYMOUS = 0x01, 0x02, 0x10, 0x20
_FAILED = ctypes.c_void_p(-1).value
_REGISTER_MAPPED, _REGISTER_READ_ONLY = 0x02, 0x08
# CUdevice_attribute
_INTEGRATED, _PAGEABLE, _HOST_POINTER_FOR_REGISTERED, _HOST_PAGE_TABLES = 18, 88, 91, 100
_REGISTER_SUPPORTED, _REGISTER_READ_ONLY_SUPPORTED = 99, 113

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


def mode(index: int) -> Literal["pageable", "registered"] | None:
    """How an integrated GPU reads a mapped file in place: through the CPU's page tables, or once
    registered read-only; None for a discrete GPU or one that can do neither."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    device, value = ctypes.c_int(), ctypes.c_int()
    if cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(device), index):
        return None

    def has(*attributes: int) -> bool:
        return all(
            not cuda.cuDeviceGetAttribute(ctypes.byref(value), a, device) and value.value == 1
            for a in attributes
        )

    if not has(_INTEGRATED):
        return None
    if has(_PAGEABLE, _HOST_PAGE_TABLES):
        return "pageable"
    if has(_REGISTER_SUPPORTED, _REGISTER_READ_ONLY_SUPPORTED, _HOST_POINTER_FOR_REGISTERED):
        return "registered"
    return None


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
    register: bool,
) -> Tensor:
    """`span` bytes as a uint8 tensor on GPU `index`: each piece's file bytes mapped in place at
    its offset, the rest (inline parts) written by `rest` into the anonymous pages around them,
    registered read-only with the driver where the GPU needs it (`mode`). `rest` must write only
    outside the pieces' pages: those are read-only file pages."""
    base = _mmap(None, span, _PROT_READ | _PROT_WRITE, _MAP_PRIVATE | _MAP_ANONYMOUS)
    cudart = torch.cuda.cudart()
    try:
        for fd, off, nbytes, at in pieces:
            pages = -(-nbytes // PAGE) * PAGE
            _mmap(base + at, pages, _PROT_READ, _MAP_SHARED | _MAP_FIXED, fd, off)
        rest(memoryview((ctypes.c_char * span).from_address(base)).cast("B"))
        if register:
            torch.cuda.check_error(
                cudart.cudaHostRegister(base, span, _REGISTER_MAPPED | _REGISTER_READ_ONLY)
            )
    except BaseException:
        _libc.munmap(base, span)
        raise

    def close() -> None:
        # Kernels still reading the pages finish before they go.
        torch.cuda.synchronize(index)
        if register:
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
