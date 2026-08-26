"""AcceleratorOps: the device boundary, the second of exactly two platform seams (#447).

Backends are `cuda` and `mps`, period — the no-device-zoo rule of #422 stands; what #447
amends is WHERE the branch lives. Every `if backend == "mps"` belongs HERE, not spread
through lifecycle, fill and ledger paths. Enumerated implementations, no registration
mechanism: an unimplemented (backend, operation) pair refuses TYPED, naming cr-021 as the
lane that fills it.

The surface is deliberately narrow — synchronize, OOM primitive, allocation snapshot, the
forced child-env law, qualification gate. `fill.py`'s stream/event machinery is CUDA
IMPLEMENTATION CODE behind this boundary (the MPS lane lands its own fill path behind the
same names), not a violation of it. Callers hand in their `torch` module; this module
imports none.
"""

from __future__ import annotations

from typing import Any

#: The #450 law, imposed on every MPS executor's sealed environment: enabling the fallback
#: silently runs unsupported operations on CPU, which violates the CPU-refusal law. Forced,
#: not defaulted — an endpoint cannot opt back in.
MPS_CHILD_ENV_LAW = ("PYTORCH_ENABLE_MPS_FALLBACK", "0")

BACKENDS = ("cuda", "mps")


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


def synchronize(torch: Any, kind: str) -> None:
    """A device-order fence for `kind`. `cpu` is a no-op (host code is already ordered)."""
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "synchronize")
    if kind == "cuda":
        torch.cuda.synchronize()


def is_oom(torch: Any, exc: BaseException) -> bool:
    """The OOM PRIMITIVE only — what the failure IS. What it MEANS (#418's
    accepted_envelope_breach classification) stays with the attempt lifecycle."""
    oom = getattr(torch.cuda, "OutOfMemoryError", ())
    return isinstance(exc, oom)


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


def host_backend_family() -> str:
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
    suite is CUDA; the MPS suite must qualify the ACTUAL endpoint graph, not one matmul
    (#450), and lands with cr-021."""
    if kind not in BACKENDS:
        raise AcceleratorUnsupported(kind, "qualification")
    if kind == "mps":
        raise AcceleratorUnsupported("mps", "qualification")
