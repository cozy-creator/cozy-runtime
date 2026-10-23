"""`verify_op`: every native kernel is checked on the device before it serves (design §6.3).

A prebuilt, local or JIT kernel runs its probe once per process against an exact reference on
the card it will serve: a non-finite output fails before any ratio, then the relative L2 error
must be within the kernel's budget. A failure stays in this process: its caller serves the next
kernel of the family's order, the run says so (`gemm.fallback`), and the next process checks
again. No stored verdict ever turns a kernel off (owner, 2026-10-22).

The one thing kept on the machine is a probe that never finished: a marker written before the
probe and removed after it, so a kernel that ends its process (a sticky CUDA error) is refused by
the next process instead of crash-looping it. The marker is keyed by the kernel, its artifact's
digest, the card, the driver and the Runtime build, so an upgrade or a driver update checks it
again. A device that ran out of memory during the probe keeps nothing: the error raises for the
memory manager.

A kernel first opens at a leaf's bind, inside its model's construction: the probe runs with the
construction guard stepped aside (`derive.placing`), as the runtime's own placement does.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

from cozy_runtime import __version__
from cozy_runtime._build_provenance import COMMIT
from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._errors import is_capacity
from cozy_runtime.internal import accel, filelock, kernel_cache
from cozy_runtime.internal.derive import placing
from cozy_runtime.internal.encoding.formats import Tensor

#: The unfinished-probe markers under the machine's kernel store.
VERDICTS = "verified"
#: The Runtime build a marker belongs to: an upgrade checks every kernel again.
RUNTIME = f"{__version__}+{COMMIT[:12]}" if COMMIT else __version__
#: What a kernel whose probe ended a process reads in the next one.
CRASHED = "its probe ended a process before it finished; this Runtime build will not run it here"


@dataclass(frozen=True, slots=True)
class Verdict:
    ok: bool
    #: what was measured, or why it failed
    detail: str


#: (kernel, artifact, sm, driver, Runtime build) -> its verdict, for this process only.
type Key = tuple[str, str, int, str, str]
_KNOWN: dict[Key, Verdict] = {}


def error(torch: ModuleType, got: Tensor, want: Tensor) -> float:
    """Relative L2 error of `got` against the exact `want`; inf where `got` is not finite."""
    got = got.double()
    if not bool(torch.isfinite(got).all()):
        return math.inf
    want = want.double()
    return float((got - want).norm() / want.norm().clamp(min=1e-30))


def within(errors: dict[str, float], budget: float) -> Verdict:
    """Every probed case within `budget`, said with the worst one."""
    if not errors:
        return Verdict(False, "no case ran")
    worst = max(errors, key=lambda case: errors[case])
    detail = f"rel L2 {errors[worst]:.2e} at {worst} (budget {budget:g}, {len(errors)} cases)"
    return Verdict(errors[worst] <= budget, detail)


def verified(kernel: str, artifact: str, sm: int, probe: Callable[[], Verdict]) -> Verdict:
    """`kernel`'s verdict on this card for this process: probed now, unless a probe of it ended
    an earlier process of this Runtime build here. A probe that raises fails with its exception."""
    key = (kernel, artifact, sm, accel.driver_version(), RUNTIME)
    known = _KNOWN.get(key)
    if known is None:
        with request_path.forbid(f"kernel self-check: {kernel}"):
            known = _KNOWN[key] = _probed(key, probe)
    return known


def _probed(key: Key, probe: Callable[[], Verdict]) -> Verdict:
    path = _path(key)
    if path is None:
        return _run(probe)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".lock").open("a+")
    except OSError:
        return _run(probe)
    with lock:
        filelock.lock(lock)  # another process of this machine probing it finishes first
        if path.exists():  # its marker outlived the probe: that probe ended its process
            return Verdict(False, CRASHED)
        _write(path, key, Verdict(False, CRASHED))
        try:
            return _run(probe)
        finally:
            path.unlink(missing_ok=True)


def _run(probe: Callable[[], Verdict]) -> Verdict:
    """The probe's verdict; one that raises fails, except when the device ran out, which says
    nothing about the kernel and raises for the memory manager to answer."""
    try:
        with placing():  # the runtime's own allocations, never an author's construction
            return probe()
    except Exception as exc:
        if is_capacity(exc):
            raise
        return Verdict(False, f"{type(exc).__name__}: {exc}"[:400])


def _path(key: Key) -> Path | None:
    store = kernel_cache.configured()
    if store is None:
        return None
    digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:32]
    return store.own / VERDICTS / f"{key[0]}-{digest}.json"


def _write(path: Path, key: Key, verdict: Verdict) -> None:
    """Kept whole or not at all; a store that cannot take it guards nothing next process."""
    kernel, artifact, sm, driver, runtime = key
    row = {"kernel": kernel, "artifact": artifact, "sm": sm, "driver": driver, "runtime": runtime,
           **asdict(verdict)}  # fmt: skip
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(row))
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
