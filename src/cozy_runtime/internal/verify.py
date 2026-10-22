"""Offline kernel evidence and read-only consumer admission.

`admitted` never runs a consumer probe. `measuring` is an explicit development-only scope.
The machine cache below belongs only to development qualification, not inference admission.

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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from cozy_runtime.internal import accel, filelock, kernel_cache
from cozy_runtime.internal.encoding.formats import Tensor

if TYPE_CHECKING:
    from cozy_runtime.internal.encoding import RuntimeIdentity


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


# Consumer dispatch reads immutable release facts. Only the explicit development scope below
# may execute a probe; the machine-local cache above is never consumer admission evidence.
_OFFLINE: ContextVar[dict[tuple[str, str, int], Verdict] | None] = ContextVar(
    "kernel_offline_verdicts", default=None
)


def in_oracle() -> bool:
    return _OFFLINE.get() is not None


@contextmanager
def measuring() -> Iterator[dict[tuple[str, str, int], Verdict]]:
    """Development oracle scope: collect the exact kernel arms actually qualified."""
    records: dict[tuple[str, str, int], Verdict] = {}
    token = _OFFLINE.set(records)
    try:
        yield records
    finally:
        _OFFLINE.reset(token)


@lru_cache(maxsize=1)
def _identity() -> tuple[RuntimeIdentity, str, str]:
    import torch

    from cozy_runtime.internal.encoding import measure_runtime

    runtime = measure_runtime(torch, "offline")
    return runtime, runtime.source_digest, runtime.torch_build


def _name(sm: int, source: str, torch_build: str) -> str:
    build = hashlib.sha256(torch_build.encode()).hexdigest()[:12]
    return f"kernels-sm{sm}-{source.removeprefix('sha256:')[:16]}-{build}.json"


def publish(directory: Path, records: dict[tuple[str, str, int], Verdict]) -> list[Path]:
    """Export real offline verdicts beside format qualification data, without rewriting identity."""
    runtime, source, torch_build = _identity()
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for sm in sorted({key[2] for key in records}):
        path = directory / _name(sm, source, torch_build)
        rows = [
            {"kernel": kernel, "artifact": artifact, "sm": card, **asdict(verdict)}
            for (kernel, artifact, card), verdict in sorted(records.items())
            if card == sm
        ]
        data = {"schema": 1, "runtime": asdict(runtime), "sm": sm, "verdicts": rows}
        temporary = path.with_name(f".{path.name}.{os.getpid()}")
        temporary.write_text(json.dumps(data, sort_keys=True))
        os.replace(temporary, path)
        written.append(path)
    return written


def admitted(
    kernel: str, artifact: str, sm: int, probe: Callable[[], Verdict], *, tables: Path | None = None
) -> Verdict:
    """A shipped verdict, or a real probe only within the explicit offline oracle scope."""
    if _OFFLINE.get() is not None:
        return measured(kernel, artifact, sm, probe)
    found = shipped(kernel, artifact, sm, tables=tables)
    return found or Verdict(False, f"no matching offline verdict for {kernel} on sm{sm}")


def measured(kernel: str, artifact: str, sm: int, probe: Callable[[], Verdict]) -> Verdict:
    """The development oracle's probe, recorded for publication (inside `measuring` only)."""
    records = _OFFLINE.get()
    assert records is not None, "a consumer never probes"
    _, source, torch_build = _identity()
    # A source change invalidates old development-cache verdicts as well as shipped facts.
    verdict = verified(kernel, f"{artifact};source={source};torch={torch_build}", sm, probe)
    records[kernel, artifact, sm] = verdict
    return verdict


def shipped(kernel: str, artifact: str, sm: int, *, tables: Path | None = None) -> Verdict | None:
    """The release's offline verdict for this exact kernel, artifact and card, or None."""
    _, source, torch_build = _identity()
    directory = Path(__file__).parent / "qualification_tables" if tables is None else tables
    try:
        document = json.loads((directory / _name(sm, source, torch_build)).read_bytes())
        runtime = document["runtime"]
        if (
            document["schema"] != 1
            or document["sm"] != sm
            or runtime["source_digest"] != source
            or runtime["torch_build"] != torch_build
        ):
            raise ValueError("kernel verdict identity mismatch")
        for row in document["verdicts"]:
            if row["kernel"] == kernel and row["artifact"] == artifact and row["sm"] == sm:
                if type(row["ok"]) is not bool or not isinstance(row["detail"], str):
                    raise ValueError("invalid kernel verdict")
                return Verdict(row["ok"], row["detail"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


@lru_cache(maxsize=32)
def implementation_artifact(module: ModuleType) -> str:
    """Hash the imported wrapper and native implementation, not its recipe's display name.

    Package trees include their fixed JIT sources. Single-file wrappers also bind imported
    extension modules, such as flash_attn_interface's separately installed CUDA module.
    """
    from cozy_runtime.internal import canonical

    roots = tuple(Path(p) for p in getattr(module, "__path__", ()))
    files = {
        path
        for root in roots
        for path in root.rglob("*")
        if path.suffix
        in (
            ".py",
            ".so",
            ".pyd",
            ".dll",
            ".dylib",
            ".cu",
            ".cuh",
            ".h",
            ".hpp",
            ".cpp",
            ".cc",
            ".ptx",
            ".cubin",
            ".json",
        )
    }
    if getattr(module, "__file__", None):
        files.add(Path(module.__file__))
    for imported in vars(module).values():
        if isinstance(imported, ModuleType) and getattr(imported, "__file__", None):
            path = Path(imported.__file__)
            if path.suffix in (".so", ".pyd", ".dll", ".dylib"):
                files.add(path)
    if not files:
        return "ABSENT"
    # Paths outside the installed tree are not part of identity; compiled bytes are.
    return canonical.digest(sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in files))
