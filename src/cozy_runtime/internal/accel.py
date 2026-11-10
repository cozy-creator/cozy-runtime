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
from typing import TYPE_CHECKING, Literal, TypedDict

import tensorfs.varena.torch as vt
from tensorfs.varena import _runtime as varena

from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.internal.config import sees_no_gpu
from cozy_runtime.internal.hostfacts import available_memory, physical_memory

if TYPE_CHECKING:
    from torch.cuda import Event, Stream
    from torch.mps.event import Event as MPSEvent

    from cozy_runtime.internal.residency import Memory


class ResidencyError(RuntimeError):
    """Device weight residency cannot serve: raised here and by `residency`."""

    code = "RESIDENCY"
    sticky = False
    cuda_code = None


class StaleBinding(ResidencyError):
    """A cached graph must retire before these weights can change storage or fill."""

    code = "STALE_BINDING"


#: The #450 law, imposed on every MPS executor's sealed environment: enabling the fallback
#: silently runs unsupported operations on CPU, which violates the CPU-refusal law. Forced,
#: not defaulted — a package cannot opt back in.
MPS_CHILD_ENV_LAW = ("PYTORCH_ENABLE_MPS_FALLBACK", "0")

BACKENDS = ("cuda", "mps")

#: NVML's own answers, by name. `nvmlReturn_t` is an enum; these are the two this module
#: branches on.
_NVML_SUCCESS = 0
_NVML_ERROR_NOT_SUPPORTED = 3


class AcceleratorUnsupported(Exception):
    """This (backend, operation) has no implementation — typed."""

    def __init__(self, backend: str, operation: str) -> None:
        super().__init__(
            f"accel.{operation} has no {backend!r} implementation; refusing typed beats "
            "silently serving through a path that was never measured"
        )
        self.backend = backend
        self.operation = operation


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


@dataclass(frozen=True, slots=True)
class DeviceMemory:
    """The driver's view of ONE device's memory, read from outside any CUDA context."""

    state: Literal["measured", "unreadable"]
    free_bytes: int = -1
    total_bytes: int = -1


def _host_pool() -> DeviceMemory:
    """Linux's MemAvailable and MemTotal, for a CPU device's report. (An integrated GPU's room
    is varena's `room()`, from the platform's reading of the host.)"""
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


def synchronize(torch: ModuleType, kind: str) -> None:
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


def drain(torch: ModuleType, kind: str) -> None:
    """Wait for the CURRENT stream only: the caller's own queued work, not other streams'.
    MPS has one command queue per process: its drain is a synchronize."""
    if kind == "mps":
        torch.mps.synchronize()
    elif kind == "cuda":
        completion_event(torch, kind).synchronize()


def bind_allocator(torch: ModuleType, kind: str) -> None:
    """Select Runtime's allocator without device activation or standalone defaults."""
    if kind != "cuda":
        return
    try:
        varena.bind_torch_allocator()
    except (ImportError, AttributeError, RuntimeError) as exc:
        if fault(exc) is not Fault.OTHER:
            raise
        raise ResidencyError(str(exc)) from exc


def initialize(torch: ModuleType, kind: str) -> None:
    """Initialize after allocator binding and the caller's explicit native custody."""
    if kind == "cpu":
        return
    if sees_no_gpu():
        raise RuntimeError("this process sees no GPU (CUDA_VISIBLE_DEVICES is empty)")
    if kind == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("this Mac's torch has no usable MPS device")
    if kind == "cuda":

        torch.cuda.init()


def owned_stream(torch: ModuleType, device: int) -> Stream:
    """A unique Runtime-owned blocking stream. It orders with the legacy default stream, where a
    CUDA extension launches when it passes no stream, and a capture on it refuses that launch
    instead of leaving it out of the graph. CUDA's runtime makes it, so a thread that has not yet
    touched the device (the idle recorder's) needs no context of its own.

    The caller owns destruction after all library handles, tensors and events using it
    have retired. Executor streams live until process exit; graph domains close theirs.
    """
    pointer = ctypes.c_void_p()
    with torch.cuda.device(device):
        code = int(torch.cuda.cudart().cudaStreamCreate(ctypes.addressof(pointer)))
        if code:
            error = torch.AcceleratorError(f"cudaStreamCreate failed with CUDA error {code}")
            error.error_code = code
            raise error
        try:
            stream: Stream = torch.cuda.ExternalStream(pointer.value, device=device)
            return stream
        except BaseException:
            torch.cuda.cudart().cudaStreamDestroy(pointer.value)
            raise


def destroy_stream(torch: ModuleType, stream: Stream) -> None:
    """Destroy an owned stream only after its owner retires and joins every consumer."""
    with torch.cuda.device(stream.device):
        code = int(torch.cuda.cudart().cudaStreamDestroy(stream.cuda_stream))
    if code:
        error = torch.AcceleratorError(f"cudaStreamDestroy failed with CUDA error {code}")
        error.error_code = code
        raise error


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
        completion_event(torch, kind).synchronize()


def initialized(torch: ModuleType, kind: str) -> bool:
    """Whether device initialization already escaped Runtime's explicit init boundary. MPS
    has no context to pick a device with: there is one GPU."""
    if kind == "cuda":
        return bool(torch.cuda.is_initialized())
    return False


def allocation(
    torch: ModuleType, kind: str, memory: Memory | None = None, device: vt.DeviceLike = None
) -> dict[str, int]:
    """The allocator's live numbers, and the card as varena's `room()` reports it: capacity (the
    card or its cap, less what other processes hold) and what maps free. A report, never sizing.
    CUDA figures include working allocations and native weight backing. On unified memory the
    device is this process's budget (`unified_budget`) and the driver's figure counts this
    process's Metal buffers only. A CPU's device memory is the host's RAM, and no allocator of
    ours holds any of it."""
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
    room = vt.room(device)
    weights = _weight_bytes(memory)
    working = vt.stats(device)
    return {
        "allocated_bytes": int(working["allocated_bytes"]) + weights,
        "reserved_bytes": int(working["mapped_bytes"]) + weights,
        "driver_free_bytes": int(room["free"]),
        "driver_total_bytes": int(room["capacity"]),
    }


def _weight_bytes(memory: Memory | None) -> int:
    native = memory.native if memory is not None else None
    return int(native.stats()["resident_bytes"]) if native is not None else 0


def allocated(torch: ModuleType, kind: str, memory: Memory | None = None) -> int:
    """Process-local live allocations, without querying the driver."""
    if kind == "mps":
        return int(torch.mps.current_allocated_memory())
    if kind == "cpu":
        return 0
    return int(
        vt.stats(memory.device if memory is not None else None)["allocated_bytes"]
    ) + _weight_bytes(memory)


def reserved(torch: ModuleType, kind: str, memory: Memory | None = None) -> int:
    """Process-local allocator storage, including unused cached blocks."""
    if kind == "mps":
        return int(torch.mps.driver_allocated_memory())
    if kind == "cpu":
        return 0
    return int(
        vt.stats(memory.device if memory is not None else None)["mapped_bytes"]
    ) + _weight_bytes(memory)


def allocator_peak(memory: Memory | None, request: str | None = None) -> int:
    """Absolute process high-water, including samples carried across pure-cut resets.

    A request names the observation window, not exclusive ownership; overlapping calls share it.
    """
    if memory is None or not memory.cuda:
        return 0
    figures = memory.torch_stats()
    carried = figures["requests"].get(request, (0, 0))[1] if request is not None else 0
    return max(int(figures["peak"]), int(carried))


def release_cached(torch: ModuleType, kind: str) -> None:
    """MPS's unused allocator blocks back to its driver. On CUDA the native core alone does this."""
    if kind == "mps":
        torch.mps.empty_cache()


def completion_event(torch: ModuleType, kind: str) -> Event | MPSEvent:
    """An event recorded on the CURRENT stream — "everything queued so far has run".
    The residency plane's eviction fence: bytes a kernel may still read are not evictable
    until this queries true."""
    if kind == "mps":
        fence: MPSEvent = torch.mps.Event()
        fence.record()
        return fence
    event: Event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    return event


def streams_idle(torch: ModuleType, kind: str, *, stream: Stream) -> bool:
    """Whether the execution stream and its explicitly joined producers are complete.

    The caller supplies its owned stream; another thread's current/default stream cannot
    prove completion. Consumers on other streams must join before the terminal boundary.
    """
    if kind == "cpu":
        return True
    if kind == "mps":
        # Recording a new event after a drain adds work, so its immediate query can fail.
        # These completion boundaries need the MPS queue fence, not a fresh poll marker.
        synchronize(torch, kind)
        return True
    return bool(stream.query())


def present(torch: ModuleType, kind: str) -> bool:
    """Whether THIS backend is actually usable in this process. `False` is a fact, not a
    refusal: a torch-bearing process with no card is a legitimate weightless executor, and a
    CPU executor has no accelerator to be present."""
    if kind == "cpu":
        return False
    if kind == "mps":
        return not sees_no_gpu() and bool(torch.backends.mps.is_available())
    return not sees_no_gpu() and bool(torch.cuda.is_available())


class DeviceIdentity(TypedDict):
    """`device_identity`: the device's name, SM (0 off CUDA), the wheel's CUDA runtime and the driver API."""

    name: str
    sm: int
    cuda_runtime: str
    driver_api: str


def device_identity(torch: ModuleType, kind: str, index: int = 0) -> DeviceIdentity:
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
