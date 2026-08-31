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

import subprocess
from dataclasses import dataclass
from typing import Any, Literal

#: The #450 law, imposed on every MPS executor's sealed environment: enabling the fallback
#: silently runs unsupported operations on CPU, which violates the CPU-refusal law. Forced,
#: not defaulted — a package cannot opt back in.
MPS_CHILD_ENV_LAW = ("PYTORCH_ENABLE_MPS_FALLBACK", "0")

BACKENDS = ("cuda", "mps")
PROCESS_QUERY_SECONDS = 2.0


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
    """One coherent driver-table read for all requested process IDs."""
    if not pids:
        return {}
    if kind == "mps":
        return {pid: ProcessMemory("unreadable") for pid in pids}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=PROCESS_QUERY_SECONDS,
            check=True,
        )
    except OSError, subprocess.SubprocessError:
        return {pid: ProcessMemory("unreadable") for pid in pids}
    result = {pid: ProcessMemory("absent", 0) for pid in pids}
    for line in out.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in pids:
            pid = int(parts[0])
            try:
                result[pid] = ProcessMemory("present", int(parts[1]) * 1024 * 1024)
            except ValueError:
                result[pid] = ProcessMemory("unreadable")
    return result


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


def is_oom(torch: Any, exc: BaseException) -> bool:
    """The OOM PRIMITIVE only — what the failure IS. What it MEANS (#418's
    accepted_envelope_breach classification) stays with the attempt lifecycle."""
    oom = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(exc, oom)


def oom_error(torch: Any, kind: str) -> type[BaseException]:
    """The OOM exception CLASS, for the one shape `is_oom` cannot serve: an `except`
    clause that must name the type before an instance exists."""
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
