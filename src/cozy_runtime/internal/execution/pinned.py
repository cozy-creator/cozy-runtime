"""Pinned host memory the device addresses, for what the card has no room for: a split's whole
output (cr-305).

With unified addressing a kernel reads and writes pinned host pages directly, so a CUDA tensor
made over them works in any op, at the host link's speed. The Runtime's own copies to and from
them (`write`, `read`) go through the pinned host tensor as ordinary host copies, so no
device-to-device copy ever names a host address.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence

import tensorfs
import torch
from tensorfs.varena.api import Exhausted
from torch import Tensor


class _Buffer:
    """One pinned buffer offered to Torch as device memory (`__cuda_array_interface__`): the CUDA
    tensor made over it holds this until its storage goes."""

    def __init__(self, nbytes: int) -> None:
        admit(nbytes, f"a {nbytes / 2**30:.2f} GiB whole in pinned host memory")
        self.host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
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


def admit(nbytes: int, what: str) -> None:
    """The machine's one host admission for `nbytes` more host memory, before it is taken: `what`
    it can't hold is refused in one line (`Exhausted`), never killed by the kernel (var-030 W5)."""
    check = getattr(tensorfs._ext, "_varena_host_admit", None)
    if check is not None and (why := check(nbytes)) is not None:
        raise Exhausted(f"{what}: {why}", nbytes)


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


def write(dst: Tensor, src: Tensor) -> None:
    """`dst.copy_(src)`: into pinned memory a contiguous window is a host copy the stream orders."""
    host = _host(dst)
    if host is not None and dst.is_contiguous():
        host.copy_(src, non_blocking=True)
    else:
        dst.copy_(src)  # a strided window of pinned memory: a kernel writes it in place


def read(src: Tensor) -> Tensor:
    """`src` on the device: a window of pinned memory is copied there, since an op may read its
    input many times."""
    host = _host(src)
    if host is None:
        return src
    out = torch.empty(src.shape, dtype=src.dtype, device=src.device)
    out.copy_(host if src.is_contiguous() else src, non_blocking=True)
    return out


def held(tensor: Tensor) -> bool:
    """Whether `tensor` is a view of a pinned buffer."""
    return _host(tensor) is not None


def settle() -> None:
    """Free the buffers whose tensors went, once the stream has passed their last use."""
    while _GOING:
        event, _ = _GOING[0]
        if event is None:
            torch.cuda.synchronize()
        else:
            event.synchronize()
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
