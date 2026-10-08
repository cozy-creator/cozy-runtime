"""Small, owned mapping diagnostic; no model buffers or memory-manager mutations."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import mmap
import os
import resource
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import msgspec

from cozy_runtime.author import App, Context

app = App()

# Linux UAPI constants; uv CPython builds may omit the optional fcntl/os aliases.
F_ADD_SEALS, F_GET_SEALS = 1033, 1034
SEALS = 0x1 | 0x2 | 0x4 | 0x10


class Request(msgspec.Struct, forbid_unknown_fields=True):
    pass


class Result(msgspec.Struct):
    gpu: str
    evidence_json: str


@contextmanager
def mapping(size: int, sealed: bool) -> Iterator[tuple[int, dict[str, object]]]:
    """TensorFS0.4 adopter layout: shared RO mapping of a FUTURE_WRITE-sealed memfd."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.mmap.restype = ctypes.c_void_p
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    libc.memfd_create.restype = ctypes.c_int
    fd = libc.memfd_create(b"h3-host-registration-diagnostic", 0x1 | 0x2)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create diagnostic failed")
    adopter = -1
    ptr = None
    try:
        data = b"\x5a" * size
        offset = 0
        while offset < size:
            offset += os.write(fd, data[offset:])
        if sealed:
            fcntl.fcntl(
                fd,
                F_ADD_SEALS,
                SEALS,
            )
        adopter = os.open(f"/proc/self/fd/{fd}", os.O_RDONLY if sealed else os.O_RDWR)
        permissions = mmap.PROT_READ if sealed else mmap.PROT_READ | mmap.PROT_WRITE
        ptr = libc.mmap(None, size, permissions, mmap.MAP_SHARED, adopter, 0)
        if ptr == ctypes.c_void_p(-1).value:
            ptr = None
            raise OSError(ctypes.get_errno(), "mmap diagnostic memfd failed")
        if ptr is None:
            raise RuntimeError("mmap returned a null mapping")
        row = next(
            line
            for line in Path("/proc/self/maps").read_text().splitlines()
            if int(line.split()[0].split("-")[0], 16) == ptr
        )
        fact = {
            "size_bytes": size,
            "sealed": sealed,
            "fd_seals": fcntl.fcntl(adopter, F_GET_SEALS),
            "mapping_permissions": row.split()[1],
            "sha256": hashlib.sha256(ctypes.string_at(ptr, size)).hexdigest(),
        }
        yield ptr, fact
        if hashlib.sha256(ctypes.string_at(ptr, size)).hexdigest() != fact["sha256"]:
            raise RuntimeError("Registration changed the diagnostic source bytes")
    finally:
        unmap_error = None
        if ptr is not None and libc.munmap(ptr, size) != 0:
            unmap_error = OSError(ctypes.get_errno(), "munmap diagnostic memfd failed")
        if adopter >= 0:
            os.close(adopter)
        os.close(fd)
        if unmap_error is not None:
            raise unmap_error


def memory() -> dict[str, object]:
    facts: dict[str, object] = {
        "memlock_rlimit_bytes": list(resource.getrlimit(resource.RLIMIT_MEMLOCK))
    }
    wanted = {"VmRSS", "VmLck", "RssAnon", "RssFile", "RssShmem"}
    facts["proc_status"] = {
        k: v.strip()
        for k, v in (
            line.split(":", 1)
            for line in Path("/proc/self/status").read_text().splitlines()
            if ":" in line
        )
        if k in wanted
    }
    for name in ("memory.current", "memory.max", "memory.swap.current", "memory.events"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            facts[name] = path.read_text().strip()
    return facts


@app.job(accelerator=True)
def verify(payload: Request, ctx: Context) -> Result:
    import torch

    if ctx.device.type != "cuda":
        raise ValueError("Host registration diagnostic requires the allocated rental GPU")
    device = torch.device(str(ctx.device))
    torch.cuda.set_device(device)
    anchor = torch.empty(1, device=device, dtype=torch.uint8)
    torch.cuda.synchronize(device)
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    driver.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    driver.cuMemHostUnregister.argtypes = [ctypes.c_void_p]

    def error(code: int) -> dict[str, object]:
        name, detail = ctypes.c_char_p(), ctypes.c_char_p()
        driver.cuGetErrorName(code, ctypes.byref(name))
        driver.cuGetErrorString(code, ctypes.byref(detail))
        return {
            "code": code,
            "name": name.value.decode() if name.value else None,
            "detail": detail.value.decode() if detail.value else None,
        }

    ordinal, version = ctypes.c_int(), ctypes.c_int()
    code = driver.cuCtxGetDevice(ctypes.byref(ordinal))
    if code:
        raise RuntimeError(json.dumps(error(code)))
    driver.cuDriverGetVersion(ctypes.byref(version))
    attributes = {}
    for name, number in (
        ("HOST_REGISTER_SUPPORTED", 99),
        ("PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES", 100),
        ("READ_ONLY_HOST_REGISTER_SUPPORTED", 113),
    ):
        value = ctypes.c_int()
        code = driver.cuDeviceGetAttribute(ctypes.byref(value), number, ordinal.value)
        attributes[name] = {
            "number": number,
            "value": value.value if code == 0 else None,
            "result": error(code),
        }
    cases = []
    for size in (64 << 10, 8 << 20):
        for sealed in (False, True):
            row: dict[str, object] = {"before": memory()}
            with mapping(size, sealed) as (ptr, source):
                flags = 0x1 | (0x8 if sealed else 0)
                row.update(source)
                row["flags"] = flags
                code = driver.cuMemHostRegister_v2(ptr, size, flags)
                row["register"] = error(code)
                row["after_register"] = memory()
                if code == 0:
                    unregistered = driver.cuMemHostUnregister(ptr)
                    row["unregister"] = error(unregistered)
                    if unregistered:
                        # Preserve evidence rather than claim an unregister succeeded.
                        raise RuntimeError(json.dumps(row, sort_keys=True))
            row["after_cleanup"] = memory()
            cases.append(row)
    del anchor
    return Result(
        torch.cuda.get_device_name(device),
        json.dumps(
            {
                "driver_version": version.value,
                "device_ordinal": ordinal.value,
                "attributes": attributes,
                "cases": cases,
                "no_model_buffers_touched": True,
                "purpose": "diagnostic only; not a transfer-throughput or inference benchmark",
            },
            sort_keys=True,
        ),
    )
