"""AcceleratorOps: the device boundary, the second of exactly two platform seams (#447).

Backends are `cuda` and `mps`, period — the no-device-zoo rule of #422 stands; what #447
amends is WHERE the branch lives. Every `if backend == "mps"` belongs HERE, not spread
through lifecycle, fill and ledger paths. Enumerated implementations, no registration
mechanism: an unimplemented (backend, operation) pair refuses TYPED, naming cr-021 as the
lane that fills it.

The surface is exactly the set of device operations the runtime actually performs, and no
more: context init, the OOM primitive, the memory meters, the eviction fence, quiescence,
device identity, the forced child-env law, the qualification gate. It grew to that set
rather than being guessed at — every function here has a caller, because a half-populated
boundary hands the cr-021 MPS lane the exact scatter #447 was written to prevent (#496c).

`fill.py`'s stream/event machinery is CUDA IMPLEMENTATION CODE behind this boundary (the
MPS lane lands its own fill path behind the same names), not a violation of it. Callers
hand in their `torch` module; this module imports none.
"""

from __future__ import annotations

import ctypes
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Literal

#: The #450 law, imposed on every MPS executor's sealed environment: enabling the fallback
#: silently runs unsupported operations on CPU, which violates the CPU-refusal law. Forced,
#: not defaulted — a package cannot opt back in.
MPS_CHILD_ENV_LAW = ("PYTORCH_ENABLE_MPS_FALLBACK", "0")

BACKENDS = ("cuda", "mps")

#: NVML's own answers, by name. `nvmlReturn_t` is an enum; these are the two this module
#: branches on.
_NVML_SUCCESS = 0
_NVML_ERROR_INSUFFICIENT_SIZE = 7


class AcceleratorUnsupported(Exception):
    """This (backend, operation) has no implementation yet — typed, with its lane."""

    def __init__(self, backend: str, operation: str) -> None:
        lane = "cr-021" if backend == "mps" else "unplanned"
        super().__init__(
            f"accel.{operation} has no {backend!r} implementation yet — {lane} owns it; "
            "refusing typed beats silently serving through a path that was never measured"
        )
        self.backend = backend
        self.operation = operation


@dataclass(frozen=True, slots=True)
class ProcessMemory:
    """The driver's observation of one process, with absence distinct from unreadable."""

    state: Literal["absent", "present", "unreadable"]
    bytes: int = -1


def process_memory(pid: int, kind: str) -> ProcessMemory:
    """Read per-process device memory from outside the executor.

    A successful query with no row is measured absence. A failed query is unreadable and
    never masquerades as zero. MPS has no per-process driver query yet.
    """
    return process_memories({pid}, kind)[pid]


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
    if kind == "mps":
        return {pid: ProcessMemory("unreadable") for pid in pids}
    table = _nvml_compute_table()
    if table is None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
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
        if _nvml is not None or _nvml_absent:
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
    if not pids or kind == "mps":
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
                    subprocess.run(
                        [
                            "nvidia-smi",
                            f"--id={device}",
                            "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
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
    if library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != _NVML_SUCCESS:
        return DeviceMemory("unreadable")
    return _device_memory_of(int(memory.free), int(memory.total))


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
    if kind == "mps" or not device:
        return DeviceMemory("unreadable")
    through_nvml = _nvml_device_memory(device)
    if through_nvml is not None:
        return through_nvml
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = out.stdout.strip().splitlines()[0]
        free, total = (int(part.strip()) * 1024 * 1024 for part in line.split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return DeviceMemory("unreadable")
    return _device_memory_of(free, total)


def synchronize(torch: Any, kind: str) -> None:
    """A device-order fence for `kind`. `cpu` is a no-op (host code is already ordered)."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "synchronize")
    if kind == "cuda":
        torch.cuda.synchronize()


def initialize(torch: Any, kind: str) -> None:
    """Bring the accelerator context up EAGERLY, so its cost is a measured prepare step
    rather than a surprise inside the first kernel."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "initialize")
    if kind == "cuda":
        torch.cuda.init()


def initialized(torch: Any, kind: str) -> bool:
    """Whether device initialization already escaped Runtime's explicit init boundary."""
    if kind == "mps":
        return False
    if kind == "cuda":
        return bool(torch.cuda.is_initialized())
    return False


def oom_error(torch: Any, kind: str) -> type[BaseException]:
    """The OOM exception CLASS, for an `except` clause that must name the type before an
    instance exists. An instance is classified by `author._errors.is_device_oom`."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "oom_error")
    cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if not isinstance(cls, type):  # pragma: no cover - a torch without the class
        raise AcceleratorUnsupported(kind, "oom_error")
    return cls


def allocation(torch: Any, kind: str, device: Any = None) -> dict[str, int]:
    """The allocator's live numbers plus the driver's own view — the only number that sees
    other processes. MPS lands here with driver_allocated vs recommended_max (#450)."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "allocation")
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "driver_free_bytes": int(free),
        "driver_total_bytes": int(total),
    }


def allocated(torch: Any, kind: str, device: Any = None) -> int:
    """The allocator's live bytes ALONE. Separate from `allocation` on purpose: the
    residency plane reads this on every stage and every eviction, and the driver query
    `allocation` also makes is far too expensive to ride along on that path."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "allocated")
    return int(torch.cuda.memory_allocated(device))


def reset_peak(torch: Any, kind: str) -> None:
    """Zero the high-water mark, so the next `peak_allocated` measures ONE attempt."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "reset_peak")
    if kind == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_allocated(torch: Any, kind: str) -> int:
    """The high-water mark since the last `reset_peak` — the number the ledger prices an
    attempt's activations from, which the live figure structurally cannot give."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "peak_allocated")
    return int(torch.cuda.max_memory_allocated())


def release_cached(torch: Any, kind: str) -> None:
    """Return the caching allocator's UNUSED segments to the driver. Frees nothing the
    process still holds; it is what makes an eviction visible to other processes."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "release_cached")
    if kind == "cuda":
        torch.cuda.empty_cache()


def completion_event(torch: Any, kind: str) -> Any:
    """An event recorded on the CURRENT stream — "everything queued so far has run".
    The residency plane's eviction fence: bytes a kernel may still read are not evictable
    until this queries true."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "completion_event")
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream())
    return event


def streams_idle(torch: Any, kind: str) -> bool:
    """Whether the device has nothing outstanding. The device's OWN answer to quiescence,
    which is why a terminal may be recorded on it rather than on a timer."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "streams_idle")
    return bool(torch.cuda.current_stream().query() and torch.cuda.default_stream().query())


def present(torch: Any, kind: str) -> bool:
    """Whether THIS backend is actually usable in this process. `False` is a fact, not a
    refusal: a torch-bearing process with no card is a legitimate weightless executor."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "present")
    return bool(torch.cuda.is_available())


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
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "device_identity")
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


def host_backend_family() -> Literal["cuda", "mps"]:
    """The accelerator FAMILY this host could serve — `mps` on Apple silicon, else `cuda`.
    A platform fact, not a capability claim (present-but-unqualified, #446); cheap enough
    for the spawn path, unlike hostfacts' nvidia-smi measurement."""
    import platform as _platform
    import sys as _sys

    if _sys.platform == "darwin" and _platform.machine() == "arm64":
        return "mps"
    return "cuda"


def child_env_law(kind: str) -> dict[str, str]:
    """Backend-mandated child-env impositions, merged into the reviewed seal (§3.5)."""
    if kind == "mps":
        return dict((MPS_CHILD_ENV_LAW,))
    return {}


def qualification_gate(kind: str) -> None:
    """Refuses qualification on a backend whose probe suite does not exist yet. cr-006's
    suite is CUDA; the MPS suite must qualify the ACTUAL package graph, not one matmul
    (#450), and lands with cr-021.

    `cpu` PASSES, and it is the same reading `present` already takes: a torch-bearing
    process with no card is a legitimate weightless executor, not a refusal, and cr-006's
    decode arithmetic is the identical arithmetic there — the suite qualifies it honestly
    and mints records predicated on `cpu`, which no CUDA fill can then satisfy. The whole
    point of this gate is that a backend with no measured suite mints NO records rather
    than inheriting someone else's, and that is a statement about `mps`, not about the
    absence of an accelerator.
    """
    if kind == "cpu":
        return
    if kind not in BACKENDS:
        raise AcceleratorUnsupported(kind, "qualification")
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "qualification")
