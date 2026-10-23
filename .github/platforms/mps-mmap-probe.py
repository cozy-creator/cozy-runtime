"""Can an MPS tensor be a view of an mmap'd weights file, with no copy (cr-021)?

A file of fp16 weights is mapped read-only; Metal wraps the mapping as a shared buffer
(`newBufferWithBytesNoCopy`), and torch adopts the buffer as an MPS tensor through DLPack
(`kDLMetal`: the data pointer is the MTLBuffer, the tensor's place in it a byte offset). Its ops
must match CPU, and neither torch's MPS pool nor the process's own memory may grow by the file.
Writes mmap-probe.json to argv[1].
"""

import ctypes
import json
import mmap
import os
import sys
import tempfile
from pathlib import Path

import torch

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
facts: dict[str, object] = {}
libc = ctypes.CDLL(None, use_errno=True)
objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
objc.sel_registerName.restype = ctypes.c_void_p
objc.sel_registerName.argtypes = [ctypes.c_char_p]
metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]


def send(restype, *argtypes):
    fn = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)
    return fn(ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value)


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int),
        ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64),
    ]


class DLManaged(ctypes.Structure):
    pass


DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManaged))
DLManaged._fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", DELETER)]
KEEP: list[object] = []
freed = []
on_free = DELETER(lambda _: freed.append(1))
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def mps_view(buffer: int, offset: int, shape: tuple[int, ...]) -> torch.Tensor:
    """An fp16 MPS tensor over `buffer` (an MTLBuffer) at byte `offset`, contiguous."""
    sizes = (ctypes.c_int64 * len(shape))(*shape)
    strides = [1]
    for n in reversed(shape[1:]):
        strides.insert(0, strides[0] * n)
    stride = (ctypes.c_int64 * len(shape))(*strides)
    managed = DLManaged(DLTensor(buffer, DLDevice(8, 0), len(shape), DLDataType(2, 16, 1), sizes, stride, offset), None, on_free)
    KEEP.extend((sizes, stride, managed))
    return torch.from_dlpack(ctypes.pythonapi.PyCapsule_New(ctypes.addressof(managed), b"dltensor", None))


def leg(name, fn):
    try:
        facts[name] = fn()
    except Exception as e:  # a refusal is a result here
        facts[name] = f"REFUSED {type(e).__name__}: {str(e)[:300]}"
    print(f"{name}: {facts[name]}", flush=True)


libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def usage() -> tuple[int, int, int]:
    """(wired, resident, phys_footprint) from `proc_pid_rusage` (rusage_info_v4): resident counts
    mapped file pages, the footprint does not count clean ones."""
    info = (ctypes.c_uint64 * 64)()
    assert libproc.proc_pid_rusage(os.getpid(), 4, info) == 0
    return info[2 + 5], info[2 + 6], info[2 + 7]


page = mmap.PAGESIZE
n = 4096
weights = torch.randn(3, n, n, generator=torch.Generator().manual_seed(1), dtype=torch.float32).half()
path = Path(tempfile.mkdtemp()) / "weights.bin"
head = 3 * page  # the tensors start past a header, as a chunk's do
path.write_bytes(b"\0" * head + weights.numpy().tobytes())
length = -(-path.stat().st_size // page) * page
fd = os.open(path, os.O_RDONLY)
base = libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
assert base not in (None, ctypes.c_void_p(-1).value), ctypes.get_errno()
device = metal.MTLCreateSystemDefaultDevice()
nocopy = send(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p)
torch.zeros(1, device="mps")  # torch's own device and pool, first
torch.mps.synchronize()
before = (int(torch.mps.driver_allocated_memory()), int(torch.mps.current_allocated_memory()), *usage())
buffer = nocopy(device, objc.sel_registerName(b"newBufferWithBytesNoCopy:length:options:deallocator:"), base, length, 0, None)
leg("buffer", lambda: bool(buffer))
leg("buffer_contents_is_the_mapping", lambda: send(ctypes.c_void_p)(buffer, objc.sel_registerName(b"contents")) == base)
views = [mps_view(buffer, head + i * n * n * 2, (n, n)) for i in range(3)]
x = torch.randn(64, n, generator=torch.Generator().manual_seed(2)).half()
cpu = [w.float() for w in weights]


def rel(a, b):
    a, b = a.cpu().double(), b.cpu().double()
    return float((a - b).norm() / b.norm())


leg("view_device", lambda: str(views[0].device))
leg("view_bytes_exact", lambda: bool(torch.equal(views[1].cpu(), weights[1])))
leg("linear_rel", lambda: rel(torch.nn.functional.linear(x.to("mps"), views[0]).float(), x.float() @ cpu[0].T))
leg("matmul_rel", lambda: rel((views[1] @ views[2]).float(), cpu[1] @ cpu[2]))
q = torch.randn(1, 8, 512, 64, generator=torch.Generator().manual_seed(3)).half()
kv = views[2][:512, :512].reshape(1, 8, 512, 64)
leg("sdpa_rel", lambda: rel(torch.nn.functional.scaled_dot_product_attention(q.to("mps"), kv, kv).float(),
                            torch.nn.functional.scaled_dot_product_attention(q.float(), kv.cpu().float(), kv.cpu().float())))
conv = views[0][:1152, :576].reshape(128, 64, 3, 3)
img = torch.randn(1, 64, 64, 64, generator=torch.Generator().manual_seed(4)).half()
leg("conv_rel", lambda: rel(torch.nn.functional.conv2d(img.to("mps"), conv, padding=1).float(),
                            torch.nn.functional.conv2d(img.float(), conv.cpu().float(), padding=1)))
leg("parameter", lambda: str(torch.nn.Parameter(views[0], requires_grad=False).device))
torch.mps.synchronize()
after = (int(torch.mps.driver_allocated_memory()), int(torch.mps.current_allocated_memory()), *usage())
facts["file_mib"] = round(length / 2**20, 1)
names = ("driver", "allocated", "wired", "resident", "footprint")
facts["growth_mib"] = {k: round((a - b) / 2**20, 1) for k, a, b in zip(names, after, before)}
print("growth_mib:", facts["growth_mib"], "of a", facts["file_mib"], "MiB file", flush=True)
del views, conv, kv
torch.mps.synchronize()
facts["deleters_ran"] = len(freed)
(out / "mmap-probe.json").write_text(json.dumps(facts, indent=2))
