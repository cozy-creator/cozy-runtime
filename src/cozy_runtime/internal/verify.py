"""`verify_op`: every native kernel is checked on the device before it serves (design §6.3).

A prebuilt, local or JIT kernel runs its probe once per process against an exact reference on
the card it will serve: a non-finite output fails before any ratio, then the relative L2 error
must be within the kernel's budget. A failure stays in this process: its caller serves the next
kernel of the family's order, the run says so (`gemm.fallback`), and the next process checks
again. No stored verdict ever turns a kernel off (owner, 2026-10-22).

The one thing kept on the machine is a probe that crashed its process: a marker written before
the probe and removed after it, naming the probing process, while a fatal signal's traceback
(`faulthandler`) goes to that process's crash file. A marker whose process crashed refuses its
kernel in the next process of this boot instead of crash-looping it; one whose process was ended
mid-probe (TensorD ending it for room or an update, a kill) says nothing of the kernel, and the
next process probes again (cr-227: an ended executor turned cozy-int8 off for the pod's life). The
marker is keyed by the kernel, its artifact's digest, the card, the driver and the Runtime build,
so an upgrade or a driver update checks it again, and it names its boot: a probe a reboot
interrupted says nothing of its kernel. A device that ran out of memory during the probe keeps
nothing: the error raises for the memory manager.

A kernel first opens at a leaf's bind, inside its model's construction: the probe runs with the
construction guard stepped aside (`derive.placing`), as the runtime's own placement does.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import math
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

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
#: What a kernel whose probe crashed a process reads in the next one.
CRASHED = "its probe crashed a process before it finished; this Runtime build will not run it here"


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


#: How much more a kernel with its own rounding of a format may lose than that format's exact law
#: (comfy-kitchen's int8 linear: 1.18x on sm89, run 17614); garbage loses O(100)x.
SLACK = 1.5


def against_law(errors: dict[str, tuple[float, float]], slack: float = SLACK) -> Verdict:
    """Each case's (kernel's, exact law's) relative L2 error against the unquantized product: a
    kernel that rounds the format its own way may lose at most `slack` times what the law does.
    Said with the worst ratio."""
    if not errors:
        return Verdict(False, "no case ran")
    ratio = {case: got / max(law, 1e-30) for case, (got, law) in errors.items()}
    worst = max(ratio, key=lambda case: ratio[case])
    got, law = errors[worst]
    detail = (f"rel L2 {got:.2e} at {worst}, {ratio[worst]:.2f}x the law's {law:.2e} "
              f"(slack {slack:g}, {len(errors)} cases)")  # fmt: skip
    return Verdict(ratio[worst] <= slack, detail)


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
        if _crashed(path):  # its marker outlived the probe, which crashed its process
            return Verdict(False, CRASHED)
        with _armed(path.parent) as crash:
            _write(path, key, Verdict(False, CRASHED), crash)
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


def _read_boot() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return ""


#: This boot. Where the OS does not say, every marker counts as this boot's.
_BOOT = _read_boot()


def _crashed(path: Path) -> bool:
    """Whether `path`'s marker is a probe that crashed its process this boot: the process's
    crash file holds a fatal signal's traceback. One whose process was ended mid-probe holds none:
    its marker and empty crash file go, and the kernel is probed again."""
    try:
        row = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(row, dict) or str(row.get("boot", "")) != _BOOT:
        return False
    if not row.get("crash"):
        return False
    crash = path.parent / str(row["crash"])
    try:
        if crash.stat().st_size > 0:
            return True
        crash.unlink()
    except OSError:
        pass
    return False


#: This process's probes in flight, the crash file faulthandler writes while any is, and whether
#: faulthandler wrote to stderr before.
_ARMED: list[Any] = [0, None, False]
_ARMING = threading.Lock()


@contextmanager
def _armed(directory: Path) -> Iterator[str]:
    """While a probe runs, a fatal signal's traceback goes to this process's crash file in
    `directory` (named by pid and start time), not stderr: it is what tells a probe that crashed
    its process from one whose process was ended. Its name; empty where it cannot be written."""
    name = f"crash-{os.getpid()}-{_START}"
    with _ARMING:
        if _ARMED[0] == 0:
            try:
                _ARMED[2] = faulthandler.is_enabled()
                _ARMED[1] = (directory / name).open("w")
                faulthandler.enable(_ARMED[1], all_threads=True)
            except OSError:
                _ARMED[1] = None
        _ARMED[0] += 1
    try:
        yield name if _ARMED[1] is not None else ""
    finally:
        with _ARMING:
            _ARMED[0] -= 1
            if _ARMED[0] == 0 and _ARMED[1] is not None:
                if _ARMED[2]:
                    faulthandler.enable(all_threads=True)  # stderr again, as the executor keeps it
                else:
                    faulthandler.disable()
                _ARMED[1].close()
                (directory / name).unlink(missing_ok=True)
                _ARMED[1] = None


def _read_start() -> str:
    """This process's start time in clock ticks since boot: with its pid, its name this boot."""
    try:
        return Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return "0"


_START = _read_start()


def _write(path: Path, key: Key, verdict: Verdict, crash: str) -> None:
    """Kept whole or not at all; a store that cannot take it guards nothing next process."""
    kernel, artifact, sm, driver, runtime = key
    row = {"kernel": kernel, "artifact": artifact, "sm": sm, "driver": driver, "runtime": runtime,
           "boot": _BOOT, "crash": crash, **asdict(verdict)}  # fmt: skip
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(row))
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
