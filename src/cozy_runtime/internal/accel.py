"""AcceleratorOps: the device boundary, the second of exactly two platform seams (#447).

Backends are `cuda` and `mps`, period — the no-device-zoo rule of #422 stands; what #447
amends is WHERE the branch lives. A process that sees no GPU computes on `cpu`: the degraded
tier every model still runs on (owner, 10-23), slowly, with host RAM as its device memory.
Every `if backend == "mps"` belongs HERE, not spread through lifecycle, fill and ledger
paths. Enumerated implementations, no registration mechanism: an unimplemented (backend,
operation) pair refuses TYPED.

The surface is exactly the set of device operations the runtime actually performs, and no
more: context init, the OOM primitive, the memory meters, the eviction fence, quiescence,
device identity, the memory class and its budget, the forced child-env law, the
qualification gate. Callers hand in their `torch` module; this module imports none.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import subprocess
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from cozy_runtime.internal.config import inherited_environment, sees_no_gpu
from cozy_runtime.internal.hostfacts import available_memory, physical_memory

#: The #450 law, imposed on every MPS executor's sealed environment: enabling the fallback
#: silently runs unsupported operations on CPU, which violates the CPU-refusal law. Forced,
#: not defaulted — a package cannot opt back in.
MPS_CHILD_ENV_LAW = ("PYTORCH_ENABLE_MPS_FALLBACK", "0")

BACKENDS = ("cuda", "mps")

#: NVML's own answers, by name. `nvmlReturn_t` is an enum; these are the two this module
#: branches on.
_NVML_SUCCESS = 0
_NVML_ERROR_NOT_SUPPORTED = 3
_NVML_ERROR_INSUFFICIENT_SIZE = 7


class AcceleratorUnsupported(Exception):
    """This (backend, operation) has no implementation — typed."""

    def __init__(self, backend: str, operation: str) -> None:
        super().__init__(
            f"accel.{operation} has no {backend!r} implementation; refusing typed beats "
            "silently serving through a path that was never measured"
        )
        self.backend = backend
        self.operation = operation


@dataclass(frozen=True, slots=True)
class ProcessMemory:
    """The driver's observation of one process, with absence distinct from unreadable."""

    state: Literal["absent", "present", "unreadable"]
    bytes: int = -1


def _smi(*query: str) -> subprocess.CompletedProcess[str]:
    """One `nvidia-smi` query, run to completion. A process that sees no GPU has no driver to
    ask, as on a host without the tool."""
    if sees_no_gpu():
        raise FileNotFoundError("this process sees no GPU: nvidia-smi is not asked")
    return subprocess.run(
        ["nvidia-smi", *query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )


def process_memories(pids: set[int], kind: str) -> dict[int, ProcessMemory]:
    """One coherent driver-table read for all requested process IDs.

    The ANSWER decides, never its arrival time. NVML is asked in-process first — a library
    call that answers in milliseconds and needs no fork on a host already under load — and
    `nvidia-smi` is the fallback where the library is absent, run to completion: a query
    that took four seconds on a busy eight-GPU host used to be reported UNREADABLE at two,
    and an unreadable row blocks every successor spawn. UNREADABLE now means exactly what it
    says — no driver answered — and never "the driver was slow".
    """
    if not pids:
        return {}
    if kind in ("mps", "cpu"):
        return {pid: ProcessMemory("unreadable") for pid in pids}
    table = _nvml_compute_table()
    if table is None:
        try:
            out = _smi("--query-compute-apps=pid,used_memory")
        except (OSError, subprocess.SubprocessError):
            return {pid: ProcessMemory("unreadable") for pid in pids}
        table = {}
        for line in out.stdout.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                try:
                    table[int(parts[0])] = int(parts[1]) * 1024 * 1024
                except ValueError:
                    table[int(parts[0])] = -1
    return _table_rows(pids, table)


class _NvmlProcessInfo(ctypes.Structure):
    """`nvmlProcessInfo_t` as `nvmlDeviceGetComputeRunningProcesses_v3` fills it."""

    _fields_ = (
        ("pid", ctypes.c_uint),
        ("usedGpuMemory", ctypes.c_ulonglong),
        ("gpuInstanceId", ctypes.c_uint),
        ("computeInstanceId", ctypes.c_uint),
    )


_nvml_lock = threading.Lock()
_nvml: ctypes.CDLL | None = None
_nvml_absent = False


def _nvml_library() -> ctypes.CDLL | None:
    """The driver's management library, initialized once per process, or None where the
    host has none — a CPU box, or a container without the driver mounted."""
    global _nvml, _nvml_absent
    with _nvml_lock:
        if _nvml is not None or _nvml_absent or sees_no_gpu():
            return _nvml
        try:
            library = ctypes.CDLL("libnvidia-ml.so.1")
            if library.nvmlInit_v2() != _NVML_SUCCESS:
                raise OSError("nvmlInit_v2 refused")
        except (OSError, AttributeError):
            _nvml_absent = True
            return None
        _nvml = library
        return library


def nvml_index(uuid: str) -> int:
    """nvidia-smi's number for a `GPU-…` UUID; -1 when NVML cannot say."""
    library = _nvml_library() if uuid else None
    handle = _nvml_handle(library, uuid) if library is not None else None
    index = ctypes.c_uint()
    if handle is None or library is None:
        return -1
    if library.nvmlDeviceGetIndex(handle, ctypes.byref(index)) != _NVML_SUCCESS:
        return -1
    return int(index.value)


def _nvml_handle(library: ctypes.CDLL, device: str) -> ctypes.c_void_p | None:
    """One envelope entry's handle: an index by index, anything else by UUID."""
    handle = ctypes.c_void_p()
    try:
        if device.isdigit():
            answer = library.nvmlDeviceGetHandleByIndex_v2(int(device), ctypes.byref(handle))
        else:
            answer = library.nvmlDeviceGetHandleByUUID(device.encode(), ctypes.byref(handle))
    except (OverflowError, ctypes.ArgumentError, UnicodeEncodeError):
        return None
    return handle if answer == _NVML_SUCCESS else None


def _nvml_device_compute_table(
    library: ctypes.CDLL, handle: ctypes.c_void_p
) -> dict[int, int] | None:
    """One device's compute processes: pid -> used bytes (-1 when the driver does not
    account it). None when the driver would not answer for this device."""
    table: dict[int, int] = {}
    rows = ctypes.c_uint(0)
    answer = library.nvmlDeviceGetComputeRunningProcesses_v3(handle, ctypes.byref(rows), None)
    if answer == _NVML_SUCCESS:
        return table
    if answer != _NVML_ERROR_INSUFFICIENT_SIZE:
        return None
    # The count is a snapshot; a process can start between the two calls. Ask for room.
    rows = ctypes.c_uint(rows.value * 2 + 8)
    infos = (_NvmlProcessInfo * rows.value)()
    answer = library.nvmlDeviceGetComputeRunningProcesses_v3(handle, ctypes.byref(rows), infos)
    if answer != _NVML_SUCCESS:
        return None
    for info in infos[: rows.value]:
        used = int(info.usedGpuMemory)
        # NVML reports "not available" as all bits set.
        table[int(info.pid)] = -1 if used == (1 << 64) - 1 else used
    return table


def _nvml_compute_table() -> dict[int, int] | None:
    """Every compute process on every device: pid -> used bytes (-1 when the driver does
    not account it). None when NVML is unavailable, so the caller falls back."""
    library = _nvml_library()
    if library is None:
        return None
    table: dict[int, int] = {}
    count = ctypes.c_uint(0)
    if library.nvmlDeviceGetCount_v2(ctypes.byref(count)) != _NVML_SUCCESS:
        return None
    for index in range(count.value):
        handle = ctypes.c_void_p()
        if library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != _NVML_SUCCESS:
            return None
        rows = _nvml_device_compute_table(library, handle)
        if rows is None:
            return None
        table.update(rows)
    return table


def compute_processes(device: str) -> dict[int, int] | None:
    """`device`'s compute processes as NVML counts them: pid -> used bytes (-1 unaccounted).
    None where the driver will not say."""
    library = _nvml_library()
    handle = _nvml_handle(library, device) if library is not None and device else None
    return (
        _nvml_device_compute_table(library, handle)
        if library is not None and handle is not None
        else None
    )


def _table_rows(pids: set[int], table: dict[int, int]) -> dict[int, ProcessMemory]:
    return {
        pid: (
            ProcessMemory("absent", 0)
            if pid not in table
            else ProcessMemory("present", table[pid])
            if table[pid] >= 0
            else ProcessMemory("unreadable")
        )
        for pid in pids
    }


def process_memories_by_device(
    pids: set[int], kind: str, devices: tuple[str, ...]
) -> dict[str, dict[int, ProcessMemory]]:
    """`process_memories`, PER DEVICE of a lane (cr-068): entry -> pid -> the driver's row.

    A group lane's executor is K processes over K cards, and its reclaim is proved absent
    on every one of them. The merged table is what the reclaim loop waits on (absent on
    every device the driver has is absent on these); this is the per-entry evidence the
    record carries. An entry the driver does not have, or a query it refuses, is
    UNREADABLE for every pid - never absent by omission. In a PID namespace the driver
    reports HOST pids, so a row here can only ever confirm the OS-state half, never
    replace it.
    """
    if not pids or kind in ("mps", "cpu"):
        return {device: {pid: ProcessMemory("unreadable") for pid in pids} for device in devices}
    library = _nvml_library()
    out: dict[str, dict[int, ProcessMemory]] = {}
    for device in devices:
        table: dict[int, int] | None = None
        if library is not None:
            handle = _nvml_handle(library, device)
            table = _nvml_device_compute_table(library, handle) if handle is not None else None
        else:
            try:
                lines = (
                    _smi(f"--id={device}", "--query-compute-apps=pid,used_memory")
                    .stdout.strip()
                    .splitlines()
                )
            except (OSError, subprocess.SubprocessError):
                lines = None
            if lines is not None:
                table = {}
                for line in lines:
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) == 2 and parts[0].isdigit():
                        try:
                            table[int(parts[0])] = int(parts[1]) * 1024 * 1024
                        except ValueError:
                            table[int(parts[0])] = -1
        out[device] = (
            {pid: ProcessMemory("unreadable") for pid in pids}
            if table is None
            else _table_rows(pids, table)
        )
    return out


@dataclass(frozen=True, slots=True)
class DeviceMemory:
    """The driver's view of ONE device's memory, read from outside any CUDA context."""

    state: Literal["measured", "unreadable"]
    free_bytes: int = -1
    total_bytes: int = -1


class _NvmlMemory(ctypes.Structure):
    """`nvmlMemory_t` as `nvmlDeviceGetMemoryInfo` fills it: bytes, not MiB."""

    _fields_ = (
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    )


def _nvml_device_memory(device: str) -> DeviceMemory | None:
    """One envelope entry's free/total through NVML, or None where NVML is absent.

    An index resolves by index, anything else (`GPU-…`, `MIG-…`) by UUID — the two
    spellings `CUDA_VISIBLE_DEVICES` admits. An entry the driver does not have is
    UNREADABLE, not None: NVML answered, and its answer is that there is no such device.
    """
    library = _nvml_library()
    if library is None:
        return None
    handle = _nvml_handle(library, device)
    if handle is None:
        return DeviceMemory("unreadable")
    memory = _NvmlMemory()
    answer = library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
    if answer == _NVML_ERROR_NOT_SUPPORTED:
        return _host_pool()
    if answer != _NVML_SUCCESS:
        return DeviceMemory("unreadable")
    return _device_memory_of(int(memory.free), int(memory.total))


def _host_pool() -> DeviceMemory:
    """An integrated GPU's memory (GB10, Jetson): it has no framebuffer and allocates from the
    host's RAM, so the driver answers Not Supported / `[N/A]` and its free and total are the
    host's MemAvailable and MemTotal."""
    try:
        rows = dict(row.split(":", 1) for row in Path("/proc/meminfo").read_text().splitlines())
        free, total = (int(rows[key].split()[0]) * 1024 for key in ("MemAvailable", "MemTotal"))
    except (OSError, KeyError, ValueError, IndexError):
        return DeviceMemory("unreadable")
    return _device_memory_of(free, total)


def device_capability(device: str) -> int:
    """One envelope entry's CUDA capability as `major * 10 + minor` (sm_120 is 120), from
    the driver, torch-free; zero where NVML is absent or does not know the entry."""
    library = _nvml_library()
    handle = _nvml_handle(library, device) if library is not None else None
    if library is None or handle is None:
        return 0
    major, minor = ctypes.c_int(), ctypes.c_int()
    answer = library.nvmlDeviceGetCudaComputeCapability(
        handle, ctypes.byref(major), ctypes.byref(minor)
    )
    return major.value * 10 + minor.value if answer == _NVML_SUCCESS else 0


_NVML_NOT_SUPPORTED = 3


def device_settings(index: int) -> tuple[str, str, str]:
    """NVML device `index`'s driver version, power limit (W) and MIG mode, spelled as
    `nvidia-smi --query-gpu=driver_version,power.limit,mig.mode.current` spells them. In-process
    where the library is present, else `nvidia-smi` run to its answer: the answer decides, never
    how long it took. `[N/A]` is the driver saying so; `unreadable` is no driver answering."""
    library = _nvml_library()
    handle = _nvml_handle(library, str(index)) if library is not None else None
    if library is None or handle is None:
        try:
            out = _smi("-i", str(index), "--query-gpu=driver_version,power.limit,mig.mode.current")
        except (OSError, subprocess.SubprocessError):
            return ("unreadable",) * 3
        fields = [field.strip().replace(" ", "") or "unreadable" for field in out.stdout.split(",")]
        if len(fields) != 3:
            return ("unreadable",) * 3
        return fields[0], fields[1], fields[2]
    version = ctypes.create_string_buffer(96)
    driver = (
        version.value.decode()
        if library.nvmlSystemGetDriverVersion(version, ctypes.c_uint(96)) == _NVML_SUCCESS
        else "unreadable"
    )
    milliwatts, current, pending = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()

    def spelled(answer: int, value: str) -> str:
        if answer == _NVML_SUCCESS:
            return value
        return "[N/A]" if answer == _NVML_NOT_SUPPORTED else "unreadable"

    answer = library.nvmlDeviceGetPowerManagementLimit(handle, ctypes.byref(milliwatts))
    power = spelled(answer, f"{milliwatts.value / 1000:.2f}")
    answer = library.nvmlDeviceGetMigMode(handle, ctypes.byref(current), ctypes.byref(pending))
    mig = spelled(answer, "Enabled" if current.value else "Disabled")
    return driver, power, mig


def driver_version() -> str:
    """The loaded NVIDIA driver's version, read from the kernel: no NVML and no thread, so a
    fork parent may ask. A compiled kernel's key carries it. Empty where there is none."""
    for path in ("/sys/module/nvidia/version", "/proc/driver/nvidia/version"):
        with suppress(OSError):
            found = re.search(r"\d+\.\d+(\.\d+)*", Path(path).read_text().splitlines()[0])
            if found:
                return found.group()
    return ""


def drives_display(index: int) -> bool:
    """Whether a display is initialized on CUDA device `index` of this process (NVML's display
    active state: a display server holds memory for a screen there); False where NVML cannot
    say. A connector with a display attached (NVML's display mode: a dummy plug on many
    datacenter GPUs) is not one."""
    visible = inherited_environment().get("CUDA_VISIBLE_DEVICES", "")
    entries = [e.strip() for e in visible.split(",")]
    entries = [e for e in entries if e]
    library = _nvml_library()
    entry = entries[index] if index < len(entries) else str(index)
    handle = _nvml_handle(library, entry) if library is not None else None
    if library is None or handle is None:
        return False
    state = ctypes.c_int()
    active = library.nvmlDeviceGetDisplayActive(handle, ctypes.byref(state))
    return bool(active == _NVML_SUCCESS and state.value == 1)


def host_memory() -> DeviceMemory:
    """The host's RAM as a CPU device's memory: what the OS can give now, and all of it."""
    if sys.platform == "darwin":
        return _device_memory_of(_available(), physical_memory())
    if sys.platform == "win32":
        return _device_memory_of(available_memory(), physical_memory())
    return _host_pool()


def _device_memory_of(free: int, total: int) -> DeviceMemory:
    if total <= 0 or free < 0 or free > total:
        return DeviceMemory("unreadable")
    return DeviceMemory("measured", free, total)


def device_memory(device: str, kind: str) -> DeviceMemory:
    """Free and total bytes of one envelope entry, measured by the DRIVER, torch-free.

    The worker assigns residence ceilings from this number (cr-066): it is the only view
    that sees every process on the device, and it is readable before any executor exists.
    `device` is one `CUDA_VISIBLE_DEVICES` entry verbatim — an index or a GPU UUID.

    NVML in-process first (xs-007 row 19's rule: the ANSWER decides, never its arrival
    time): `nvmlDeviceGetMemoryInfo` is the exact byte count `nvidia-smi` rounds to MiB,
    measured at 0.02 ms against the subprocess's 22 ms on the 4070. `nvidia-smi --id` is
    the fallback where the library is absent, run to completion. UNREADABLE means neither
    answered, or the driver has no such device — never a zero and never a guess: a lane
    whose capacity cannot be read cannot be shown to fit anything. MPS has no per-device
    driver query.
    """
    if kind == "cpu":
        return host_memory()
    if kind == "mps" or not device:
        return DeviceMemory("unreadable")
    through_nvml = _nvml_device_memory(device)
    if through_nvml is not None:
        return through_nvml
    try:
        out = _smi(f"--id={device}", "--query-gpu=memory.free,memory.total")
        line = out.stdout.strip().splitlines()[0]
        if line.replace(" ", "") == "[N/A],[N/A]":
            return _host_pool()
        free, total = (int(part.strip()) * 1024 * 1024 for part in line.split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return DeviceMemory("unreadable")
    return _device_memory_of(free, total)


def synchronize(torch: Any, kind: str) -> None:
    """A device-order fence for `kind`. `cpu` is a no-op (host code is already ordered)."""
    if kind == "mps":
        torch.mps.synchronize()
    elif kind == "cuda":
        torch.cuda.synchronize()


def healthy(torch: ModuleType, kind: str) -> bool:
    """Whether the device still answers a fence: a fault is sticky (an illegal address, a copy
    engine's), and every later call of this process fails with it."""
    try:
        synchronize(torch, kind)
    except Exception:  # the fault itself, whatever torch calls it
        return False
    return True


def drain(torch: Any, kind: str) -> None:
    """Wait for the CURRENT stream only: the caller's own queued work, not other streams'.
    MPS has one command queue per process: its drain is a synchronize."""
    if kind == "mps":
        torch.mps.synchronize()
    elif kind == "cuda":
        torch.cuda.current_stream().synchronize()


#: torch's caching allocator, set in code before the first allocation: a segment grows by
#: mapping only the pages a request adds and gives freed pages back the same way, so torch
#: fragments less beside the plane's maps. The machine erases every `PYTORCH_` variable from
#: an executor's environment: nothing outside the Runtime changes this.
ALLOCATOR = "expandable_segments:True"


def initialize(torch: Any, kind: str) -> None:
    """Bring the accelerator context up EAGERLY, so its cost is a measured prepare step
    rather than a surprise inside the first kernel. torch's allocator is configured first.
    A CPU executor has no accelerator to bring up."""
    if kind == "cpu":
        return
    if sees_no_gpu():
        raise RuntimeError("this process sees no GPU (CUDA_VISIBLE_DEVICES is empty)")
    if kind == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("this Mac's torch has no usable MPS device")
    if kind == "cuda":
        torch._C._accelerator_setAllocatorSettings(ALLOCATOR)
        torch.cuda.init()


def first_launches(torch: ModuleType, kind: str) -> None:
    """Create the GPU libraries' handles, workspaces and first kernels while the device has
    room. cuBLAS and cuDNN take device memory for themselves at a first launch, and inside a
    stage whose budget left none they fail under names that are not out-of-memory (run 3095:
    CUBLAS_STATUS_INTERNAL_ERROR in a block's first Linear, beside another model's weights)."""
    if kind != "cuda":
        return
    conv = torch.nn.functional
    with torch.inference_mode():
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            a = torch.ones(8, 8, dtype=dtype, device="cuda")
            kernel = a[:3, :3]
            with suppress(RuntimeError):  # a card without this dtype
                torch.mm(a, a)
                conv.conv2d(a.view(1, 1, 8, 8), kernel.reshape(1, 1, 3, 3))
                conv.conv3d(a.view(1, 1, 1, 8, 8), kernel.reshape(1, 1, 1, 3, 3))
        torch.cuda.synchronize()


def initialized(torch: Any, kind: str) -> bool:
    """Whether device initialization already escaped Runtime's explicit init boundary. MPS
    has no context to pick a device with: there is one GPU."""
    if kind == "cuda":
        return bool(torch.cuda.is_initialized())
    return False


def oom_error(torch: Any, kind: str) -> type[BaseException]:
    """The OOM exception CLASS, for an `except` clause that must name the type before an
    instance exists. An instance is classified by `author._errors.fault`."""
    cls = getattr(torch, "OutOfMemoryError", None)
    if not isinstance(cls, type):  # pragma: no cover - a torch without the class
        raise AcceleratorUnsupported(kind, "oom_error")
    return cls


def allocation(torch: Any, kind: str, memory: Any = None, device: Any = None) -> dict[str, int]:
    """The allocator's live numbers plus the driver's own view — the only number that sees
    other processes. On CUDA the allocator is varena's (`memory`), whose figures these are:
    torch's own statistics raise under it. On unified memory the device is this process's
    budget (`unified_budget`) and the driver's figure counts this process's Metal buffers only.
    A CPU's device memory is the host's RAM, and no allocator of ours holds any of it."""
    if kind == "cpu":
        host = host_memory()
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "driver_free_bytes": host.free_bytes,
            "driver_total_bytes": host.total_bytes,
        }
    if kind == "mps":
        driver, total = int(torch.mps.driver_allocated_memory()), unified_budget(torch)
        return {
            "allocated_bytes": int(torch.mps.current_allocated_memory()),
            "reserved_bytes": driver,
            "driver_free_bytes": max(total - driver, 0),
            "driver_total_bytes": total,
        }
    free, total = torch.cuda.mem_get_info(device)
    held = varena_torch(memory)
    return {
        "allocated_bytes": int(held["allocated"]),
        "reserved_bytes": int(held["mapped"]),
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
    }


def varena_torch(memory: Any) -> dict[str, Any]:
    """varena's figures for torch's blocks on a CUDA `memory`: allocated, mapped and peak bytes,
    and per open request its (allocated, peak). None without one (no torch block is varena's)."""
    if memory is None or not memory.cuda:
        return {"allocated": 0, "mapped": 0, "peak": 0, "requests": {}}
    return dict(memory.torch_stats())


def allocated(torch: Any, kind: str, memory: Any = None) -> int:
    """The allocator's live bytes ALONE. Separate from `allocation` on purpose: the
    residency plane reads this on every stage and every eviction, and the driver query
    `allocation` also makes is far too expensive to ride along on that path."""
    if kind == "mps":
        return int(torch.mps.current_allocated_memory())
    return int(varena_torch(memory)["allocated"])


def reserved(torch: ModuleType, kind: str, memory: Any = None) -> int:
    """The allocator's reserved bytes ALONE: live tensors plus what it holds mapped for them."""
    if kind == "mps":
        return int(torch.mps.driver_allocated_memory())
    return int(varena_torch(memory)["mapped"])


def arena_peak(memory: Any, request: str) -> int:
    """Open request `request`'s torch high-water in its arena (varena keeps it per request,
    so concurrent requests never share a mark); 0 off CUDA."""
    return int(varena_torch(memory)["requests"].get(request, (0, 0))[1])


#: This process's own cache releases that returned something, by the function that asked:
#: how many, and the bytes. (The caller's name is read off its frame: no call site says it.)
RELEASED: dict[str, list[int]] = {}


def release_cached(torch: Any, kind: str) -> None:
    """Return MPS's caching allocator's UNUSED segments to the driver. Frees nothing the
    process still holds; it is what makes an eviction visible to other processes. CUDA's
    allocator is varena's, which returns what it doesn't hold itself."""
    if kind == "mps":
        before = reserved(torch, kind)
        torch.mps.empty_cache()
        if (freed := before - reserved(torch, kind)) > 0:
            row = RELEASED.setdefault(sys._getframe(1).f_code.co_name, [0, 0])
            row[0] += 1
            row[1] += freed


def completion_event(torch: Any, kind: str) -> Any:
    """An event recorded on the CURRENT stream — "everything queued so far has run".
    The residency plane's eviction fence: bytes a kernel may still read are not evictable
    until this queries true."""
    if kind == "mps":
        event = torch.mps.Event()
        event.record()
        return event
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    return event


def streams_idle(torch: Any, kind: str) -> bool:
    """Whether the device has nothing outstanding. The device's OWN answer to quiescence,
    which is why a terminal may be recorded on it rather than on a timer. Host code is done."""
    if kind == "cpu":
        return True
    if kind == "mps":
        return bool(completion_event(torch, kind).query())
    return bool(torch.cuda.current_stream().query() and torch.cuda.default_stream().query())


def present(torch: Any, kind: str) -> bool:
    """Whether THIS backend is actually usable in this process. `False` is a fact, not a
    refusal: a torch-bearing process with no card is a legitimate weightless executor, and a
    CPU executor has no accelerator to be present."""
    if kind == "cpu":
        return False
    if kind == "mps":
        return not sees_no_gpu() and bool(torch.backends.mps.is_available())
    return not sees_no_gpu() and bool(torch.cuda.is_available())


def device_identity(torch: Any, kind: str, index: int = 0) -> dict[str, Any]:
    """One device's STABLE identity, from TORCH. Everything here holds for the life of the
    machine, which is what lets a probe result be keyed by it.

    **`driver` used to be `torch.version.cuda` and that was wrong** (#549.9). That value is
    the CUDA RUNTIME torch was compiled against — a property of the wheel, not of the
    machine — so two pods with one torch build and different NVIDIA drivers produced
    identical capability keys, and a record minted under a driver with a miscompiled kernel
    was reused under one without it. It is now spelled `cuda_runtime`, which is what it is.

    Two DRIVER facts replace it, and neither is invented here: `driver_api` is the CUDA
    driver API version torch itself reports (`_cuda_getDriverVersion`, e.g. 13000), and the
    display driver version proper is an NVML query that `encoding.measure_device` makes with
    the SELECTED device index. Both ride the key.
    """
    if kind == "cpu":
        return {"name": cpu_name(), "sm": 0, "cuda_runtime": "", "driver_api": ""}
    if kind == "mps":
        # Apple's GPU is the chip's; its "driver" is the macOS release.
        return {
            "name": apple_chip(),
            "sm": 0,
            "cuda_runtime": "",
            "driver_api": f"macOS {platform.mac_ver()[0] or 'unreadable'}",
        }
    major, minor = torch.cuda.get_device_capability(index)
    api = getattr(getattr(torch, "_C", None), "_cuda_getDriverVersion", None)
    return {
        "name": str(torch.cuda.get_device_properties(index).name),
        "sm": major * 10 + minor,
        # The CUDA runtime the WHEEL was built against. A build fact, named as one.
        "cuda_runtime": str(torch.version.cuda or "unknown"),
        # The CUDA DRIVER API version, from the driver. Unreadable is SPELLED, never
        # defaulted to a number that would collide with a real one.
        "driver_api": str(api() if callable(api) else "unreadable"),
    }


def host_backend_family() -> Literal["cuda", "mps", "cpu"]:
    """The device FAMILY this process serves — `mps` on Apple silicon, `cpu` where it sees no
    GPU (the degraded tier: every model runs, slowly), else `cuda`. A platform fact, not a
    capability claim (present-but-unqualified, #446); cheap enough for the spawn path, unlike
    hostfacts' nvidia-smi measurement."""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mps"
    return "cpu" if sees_no_gpu() else "cuda"


def child_env_law(kind: str) -> dict[str, str]:
    """Backend-mandated child-env impositions, merged into the reviewed seal (§3.5)."""
    if kind == "mps":
        return dict((MPS_CHILD_ENV_LAW,))
    return {}


def qualification_gate(kind: str) -> None:
    """Refuses qualification on a backend with no probe suite. The decode suite is the same
    arithmetic on every backend, so it qualifies `cuda`, `mps` and `cpu` alike and mints
    records predicated on the device it ran on, which no other device's fill can satisfy."""
    if kind != "cpu" and kind not in BACKENDS:
        raise AcceleratorUnsupported(kind, "qualification")


# ------------------------------------------------------------------- memory classes

#: What macOS and the user's other apps keep of a unified pool, beside Metal's own limit.
HOST_RESERVE = 2 << 30


def memory_class(kind: str) -> Literal["discrete", "unified"]:
    """How the device's memory relates to the host's (design §6.2). `unified`: one pool, so
    weights never page between a host and a device tier and "offload to CPU" frees nothing.
    Apple GPUs are unified; CUDA cards are discrete here (coherent and integrated CUDA
    machines are TensorD's to detect). A CPU's device memory is the host's own: unified."""
    return "unified" if kind in ("mps", "cpu") else "discrete"


#: torch's own MPS high watermark, as a fraction of Metal's recommended working set
MPS_WATERMARK = 1.7


def hold_pool(torch: ModuleType, cap: int) -> None:
    """Hold MPS's allocator, which also holds a unified plane's regions, to an owner's cap of
    `cap` bytes; -1 gives torch its own watermark back (past the budget the system pages)."""
    recommended = max(int(torch.mps.recommended_max_memory()), 1)
    torch.mps.set_per_process_memory_fraction(
        min(cap / recommended, 2.0) if cap >= 0 else MPS_WATERMARK
    )


def unified_ceiling(torch: ModuleType) -> int:
    """The most one unified-memory process may ever hold: Metal's recommended working set,
    within physical RAM less what the system keeps."""
    ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return max(min(int(torch.mps.recommended_max_memory()), ram - HOST_RESERVE), 0)


def unified_budget(torch: ModuleType) -> int:
    """What it may hold now: the ceiling, within what it holds plus what the system has
    available (shared mode follows memory pressure: past it, macOS swaps)."""
    held = int(torch.mps.driver_allocated_memory())
    return max(min(unified_ceiling(torch), held + _available()), 0)


class _VmStatistics64(ctypes.Structure):
    """`vm_statistics64` (`<mach/vm_statistics.h>`), as `host_statistics64` fills it."""

    _fields_ = (
        ("free", ctypes.c_uint32),
        ("active", ctypes.c_uint32),
        ("inactive", ctypes.c_uint32),
        ("wired", ctypes.c_uint32),
        ("events", ctypes.c_uint64 * 9),
        ("purgeable", ctypes.c_uint32),
        ("speculative", ctypes.c_uint32),
        ("compressor", ctypes.c_uint64 * 4),
        ("pages", ctypes.c_uint32 * 4),
        ("uncompressed", ctypes.c_uint64),
    )


_HOST_VM_INFO64 = 4
_libc = ctypes.CDLL(None) if sys.platform == "darwin" else None


def _available() -> int:
    """Bytes macOS can give without paging: free, speculative, purgeable and inactive pages."""
    assert _libc is not None
    stats = _VmStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(stats) // 4)
    host = _libc.mach_host_self()
    if _libc.host_statistics64(host, _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count)):
        raise OSError("host_statistics64 refused")
    pages = int(stats.free) + int(stats.speculative) + int(stats.purgeable) + int(stats.inactive)
    return pages * os.sysconf("SC_PAGE_SIZE")


def cpu_name() -> str:
    """The host CPU's own name (`AMD EPYC 9654 96-Core Processor`), or `cpu` where the OS says
    nothing."""
    with suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip() or "cpu"
    if sys.platform == "darwin":
        return apple_chip()
    return platform.processor() or "cpu"


def apple_chip() -> str:
    """The Apple chip's own name (`Apple M4 Pro`), or `apple` where sysctl says nothing."""
    with suppress(OSError, subprocess.SubprocessError):
        found = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if found:
            return found
    return "apple"


#: What a backend serves without, and what serves instead: one run warning each.
UNAVAILABLE: dict[str, tuple[tuple[str, str], ...]] = {
    "cpu": (
        ("GPU execution", "degraded mode on the CPU, many times slower; `--on new` rents a GPU"),
        ("fp8, int8 and nvfp4 lanes", "16-bit weights, computed in float32"),
        ("custom attention kernels", "PyTorch SDPA"),
    ),
    "mps": (
        ("weight plane GPU and pinned tiers", "every weight is copied to the GPU once and stays"),
        ("native CUDA kernels and Triton", "PyTorch's own MPS kernels"),
        ("fp8 and nvfp4 lanes", "16-bit weights"),
        ("custom attention kernels", "PyTorch SDPA"),
    ),
}


def unavailable(kind: str) -> tuple[tuple[str, str], ...]:
    """The features this backend turns off, each with what serves in its place."""
    return UNAVAILABLE.get(kind, ())
