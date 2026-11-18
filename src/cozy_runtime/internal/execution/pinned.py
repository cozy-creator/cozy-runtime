"""Pinned host memory the device addresses, for what the card has no room for: a split's whole
output (cr-305).

With unified addressing a kernel reads and writes pinned host pages directly, so a CUDA tensor
made over them works in any op, at the host link's speed. The Runtime's own copies to and from
them (`write`, `read`) are copies the stream orders: a contiguous window an ordinary host copy, a
window of rows at a fixed pitch (a head window of a whole cut by heads) one pitched copy, so no
device-to-device copy ever names a host address and no kernel writes a window over the link.
"""

from __future__ import annotations

import ctypes
import math
import weakref
from collections.abc import Callable, Sequence

import time

import torch
from torch import Tensor

from cozy_runtime.internal import var030  # PROOF ONLY (var-030), never merge


class _Buffer:
    """One pinned buffer offered to Torch as device memory (`__cuda_array_interface__`): the CUDA
    tensor made over it holds this until its storage goes."""

    def __init__(self, nbytes: int) -> None:
        began = time.perf_counter()
        self.host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        var030.count("pinned_alloc", nbytes, time.perf_counter() - began)
        self.address = self.host.data_ptr()
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (self.address, False),
            "version": 2,
        }

    def __del__(self) -> None:
        # Copies and kernels queued before here may still use it: it goes once the stream passes
        # this point (inside a capture, once the device is idle).
        try:
            event: torch.cuda.Event | None = None
            if not torch.cuda.is_current_stream_capturing():
                event = torch.cuda.Event()
                event.record()
            _GOING.append((event, self.host))
        except Exception:  # the device or interpreter is going down: nothing queued reads it
            pass


#: each live buffer by its first byte's device address
_LIVE: weakref.WeakValueDictionary[int, _Buffer] = weakref.WeakValueDictionary()
#: buffers whose tensors went, held until the stream passes their last use
_GOING: list[tuple[torch.cuda.Event | None, Tensor]] = []
#: host wholes made in this process (`made`)
_MADE = 0


def made() -> int:
    """How many host wholes this process has made: a pass that made one took the host rung, and a
    graph of it would hold that whole on the card (`graphs`)."""
    return _MADE


def empty(shape: Sequence[int], dtype: torch.dtype, device: torch.device) -> Tensor:
    """An uninitialized CUDA tensor of `shape` on `device`, in pinned host memory. One at a time
    stays pinned: a new one waits for the stream to pass the last use of those that went."""
    global _MADE
    settle()
    _MADE += 1
    nbytes = math.prod(shape) * dtype.itemsize
    buffer = _Buffer(max(nbytes, dtype.itemsize))
    flat = torch.as_tensor(buffer)  # its storage holds `buffer`
    if flat.device != device:
        raise RuntimeError(f"pinned memory is addressed as {flat.device}, not {device}")
    _LIVE[buffer.address] = buffer
    return flat[:nbytes].view(dtype).view(tuple(shape))


#: bytes that crossed the link for host wholes in this process: by copies to host and to the
#: card, and by kernels over pinned memory; and the wholes brought back (`cuts.landed`)
_CROSSED = {"to_host": 0, "to_card": 0, "kernel": 0, "landed": 0}


def count(how: str, n: int) -> None:
    _CROSSED[how] += n


def crossed() -> dict[str, int]:
    """`_CROSSED` now: a request's record is what it added (`Request.stats`)."""
    return dict(_CROSSED)


def write(dst: Tensor, src: Tensor) -> None:
    """`dst.copy_(src)`: into pinned memory a contiguous window is a host copy the stream orders,
    a window of rows at a pitch one pitched copy; a kernel writes it in place only where its
    strides are neither, or `src` is not the same tensor on the card."""
    host = _host(dst)
    nbytes = dst.numel() * dst.element_size()
    var030.at("write")
    if host is None:
        dst.copy_(src)
        var030.moved("wrote_card", nbytes)
        var030.at("w")
        return
    if dst.is_contiguous():
        host.copy_(src, non_blocking=True)
        count("to_host", nbytes)
        var030.moved("wrote_host", nbytes)
        var030.at("w")
        return
    rows = _rows(dst)
    same = src.is_cuda and src.dtype == dst.dtype and src.shape == dst.shape and not held(src)
    if rows is None or not same or not nbytes:
        dst.copy_(src)
        count("kernel", nbytes)
        var030.moved("wrote_host_strided", nbytes)
        var030.at("w")
        return
    width, height, pitch = rows
    src = src.contiguous()
    _pitched(dst.data_ptr(), pitch, src.data_ptr(), width, width, height, dst.device)
    count("to_host", nbytes)
    var030.moved("wrote_host_pitched", nbytes)
    var030.count(f"pitched_write_width_{width}", nbytes)
    var030.at("w")


def read(src: Tensor) -> Tensor:
    """`src` on the device: a window of pinned memory is copied there, since an op may read its
    input many times; by a pitched copy where it is rows at a pitch."""
    host = _host(src)
    if host is None:
        return src
    var030.at("read")
    out = torch.empty(src.shape, dtype=src.dtype, device=src.device)
    nbytes = src.numel() * src.element_size()
    rows = None if src.is_contiguous() else _rows(src)
    if src.is_contiguous():
        out.copy_(host, non_blocking=True)
        var030.moved("read_host", nbytes)
    elif rows is not None and nbytes:
        width, height, pitch = rows
        _pitched(out.data_ptr(), width, src.data_ptr(), pitch, width, height, src.device)
        var030.moved("read_host_pitched", nbytes)
        var030.count(f"pitched_read_width_{width}", nbytes)
    else:
        out.copy_(src)
        count("kernel", nbytes)
        var030.moved("read_host_strided", nbytes)
        var030.at("w")
        return out
    count("to_card", nbytes)
    var030.at("w")
    return out


def _rows(tensor: Tensor) -> tuple[int, int, int] | None:
    """`tensor`'s bytes as `(width, height, pitch)`: `height` rows of `width` contiguous bytes,
    `pitch` bytes apart, in its own element order; None where its strides are not that, or no
    pitched copy is at hand."""
    if _copy2d() is None:
        return None
    dims = [(n, s) for n, s in zip(tensor.shape, tensor.stride(), strict=True) if n != 1]
    run = 1  # elements in the innermost contiguous run
    while dims and dims[-1][1] == run:
        run *= dims.pop()[0]
    item = tensor.element_size()
    if not dims:
        return run * item, 1, run * item
    height, step = dims.pop()
    if step < run:
        return None
    while dims:  # outer dims fold into the rows where each steps over the ones inside it
        n, s = dims.pop()
        if s != step * height:
            return None
        height *= n
    return run * item, height, step * item


class _Copy2D(ctypes.Structure):
    """The driver's `CUDA_MEMCPY2D`."""

    _fields_ = [
        ("srcXInBytes", ctypes.c_size_t),
        ("srcY", ctypes.c_size_t),
        ("srcMemoryType", ctypes.c_int),
        ("srcHost", ctypes.c_void_p),
        ("srcDevice", ctypes.c_uint64),
        ("srcArray", ctypes.c_void_p),
        ("srcPitch", ctypes.c_size_t),
        ("dstXInBytes", ctypes.c_size_t),
        ("dstY", ctypes.c_size_t),
        ("dstMemoryType", ctypes.c_int),
        ("dstHost", ctypes.c_void_p),
        ("dstDevice", ctypes.c_uint64),
        ("dstArray", ctypes.c_void_p),
        ("dstPitch", ctypes.c_size_t),
        ("WidthInBytes", ctypes.c_size_t),
        ("Height", ctypes.c_size_t),
    ]


_UNIFIED = 4  # CU_MEMORYTYPE_UNIFIED: an address in the unified space, host or device
#: `cuMemcpy2DAsync` once loaded: None where the driver library has none
_COPY2D: list[Callable[..., int] | None] = []


def _copy2d() -> Callable[..., int] | None:
    if not _COPY2D:
        try:
            fn = ctypes.CDLL("libcuda.so.1").cuMemcpy2DAsync_v2
        except (OSError, AttributeError):
            _COPY2D.append(None)
        else:
            fn.argtypes = [ctypes.POINTER(_Copy2D), ctypes.c_void_p]
            fn.restype = ctypes.c_int
            _COPY2D.append(fn)
    return _COPY2D[0]


def _pitched(
    dst: int, dpitch: int, src: int, spitch: int, width: int, height: int, device: torch.device
) -> None:
    """One copy of `height` rows of `width` bytes, ordered on `device`'s current stream."""
    copy = _Copy2D(
        srcMemoryType=_UNIFIED,
        srcDevice=src,
        srcPitch=spitch,
        dstMemoryType=_UNIFIED,
        dstDevice=dst,
        dstPitch=dpitch,
        WidthInBytes=width,
        Height=height,
    )
    fn = _copy2d()
    assert fn is not None  # `_rows` found it
    stream = torch.cuda.current_stream(device).cuda_stream
    code = fn(ctypes.byref(copy), ctypes.c_void_p(stream))
    if code:
        raise RuntimeError(f"cuMemcpy2DAsync: {height} rows of {width} bytes, CUresult {code}")


def held(tensor: Tensor) -> bool:
    """Whether `tensor` is a view of a pinned buffer."""
    return _host(tensor) is not None


def settle() -> None:
    """Free the buffers whose tensors went, once the stream has passed their last use."""
    while _GOING:
        event, _ = _GOING[0]
        began = time.perf_counter()
        if event is None:
            torch.cuda.synchronize()
        else:
            event.synchronize()
        var030.count("pinned_settle_wait", 0, time.perf_counter() - began)
        _GOING.pop(0)


def release() -> None:
    """A request that used pinned wholes ended: free them and the host allocator's cached pages."""
    settle()
    torch._C._host_emptyCache()


def _host(tensor: Tensor) -> Tensor | None:
    """The pinned host tensor `tensor` (a CUDA view of a buffer) stands for, or None."""
    if not tensor.is_cuda:
        return None
    buffer = _LIVE.get(tensor.untyped_storage().data_ptr())
    if buffer is None:
        return None
    return buffer.host.view(tensor.dtype).as_strided(
        tensor.shape, tensor.stride(), tensor.storage_offset()
    )
