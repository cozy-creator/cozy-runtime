"""`verify_op`: every native kernel is checked on the device before it serves (design §6.3).

A prebuilt, local or JIT kernel runs its probe once against an exact reference on the card it
will serve: a non-finite output fails before any ratio, then the relative L2 error must be within
the kernel's budget. The verdict is kept per machine under the kernel store, keyed by the
kernel, its artifact's digest, the card's architecture and the driver, so a driver update
re-verifies and never recompiles. One process probes a key while others wait for its verdict, and
a failing verdict is written before the probe starts: a kernel that ends its process (a sticky
CUDA error) stays refused. A failure is final for that key: its caller serves the next kernel of
the family's order and the run says so.
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

from cozy_runtime.internal import accel, filelock, kernel_cache
from cozy_runtime.internal.encoding.formats import Tensor

#: The verdict files under the machine's kernel store.
VERDICTS = "verified"


@dataclass(frozen=True, slots=True)
class Verdict:
    ok: bool
    #: what was measured, or why it failed
    detail: str


#: (kernel, artifact, sm, driver) -> its verdict, for this process.
_KNOWN: dict[tuple[str, str, int, str], Verdict] = {}


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
    """`kernel`'s verdict on this card: kept, else `probe`d now and kept. A probe that raises
    fails with its exception."""
    key = (kernel, artifact, sm, accel.driver_version())
    known = _KNOWN.get(key)
    if known is not None:
        return known
    path = _path(key)
    if path is None:
        known = _KNOWN[key] = _run(probe)
        return known
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".lock").open("a+")
    except OSError:
        known = _KNOWN[key] = _run(probe)
        return known
    with lock:
        filelock.lock(lock)
        known = _read(path)
        if known is None:
            _write(path, key, Verdict(False, "its probe ended the process before a verdict"))
            known = _run(probe)
            _write(path, key, known)
    _KNOWN[key] = known
    return known


def _run(probe: Callable[[], Verdict]) -> Verdict:
    try:
        return probe()
    except Exception as exc:
        return Verdict(False, f"{type(exc).__name__}: {exc}"[:400])


def _path(key: tuple[str, str, int, str]) -> Path | None:
    store = kernel_cache.configured()
    if store is None:
        return None
    digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:32]
    return store.own / VERDICTS / f"{key[0]}-{digest}.json"


def _read(path: Path | None) -> Verdict | None:
    if path is None:
        return None
    try:
        row = json.loads(path.read_text())
        return Verdict(bool(row["ok"]), str(row["detail"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write(path: Path, key: tuple[str, str, int, str], verdict: Verdict) -> None:
    """Kept whole or not at all; a store that cannot take it verifies again next process."""
    kernel, artifact, sm, driver = key
    row = {"kernel": kernel, "artifact": artifact, "sm": sm, "driver": driver, **asdict(verdict)}
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(row))
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
