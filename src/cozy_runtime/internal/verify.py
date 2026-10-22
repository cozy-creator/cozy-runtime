"""Offline kernel evidence and read-only consumer admission.

A shipped verdict binds to the kernel's own identity (`Kernel`: its name, which carries its
family, the digest of the bytes that run and the digest of their declared interface), never to
the rest of Runtime's source. An unrelated edit, or a rebuild that reproduces the same kernel
bytes, keeps every fact. A changed kernel matches none, and `scripts/verify-wheels.py` fails the
release when a shipped verdict names a kernel the wheel no longer carries.

Consumers only read (`shipped`, `rules`). Probes run only inside `measuring`, the development
oracle's explicit scope (`cli.qualify`). A probe's output is checked against an exact
reference: a non-finite output fails before any ratio, then the relative L2 error must be within
the kernel's budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import msgspec

from cozy_runtime.internal import canonical
from cozy_runtime.internal.encoding.formats import Tensor

#: Where a release ships its offline facts, one kernel file per card architecture.
TABLES = Path(__file__).parent / "qualification_tables"
SCHEMA = 2
#: A rule's non-tile choices: cuBLASLt's GEMM (torch), or the one-launch GEMV.
CUBLASLT, GEMV = "cublaslt", "gemv"


@dataclass(frozen=True, slots=True)
class Verdict:
    ok: bool
    #: what was measured, or why it failed
    detail: str


@dataclass(frozen=True, slots=True)
class Kernel:
    """What a verdict binds to: `name` (family included), the digest of the bytes that run,
    and the digest of their declared native interface ("" where the bytes declare it)."""

    name: str
    build: str
    abi: str = ""


class Rule(msgspec.Struct, frozen=True):
    """A measured workload rule: a GEMM of `shape` class (`shape_class`) with
    `m_min < M <= m_max` rows takes `choice`, a tile config or `CUBLASLT`/`GEMV`. Every choice
    takes any M in its bucket: partial tiles are predicated, nothing is padded."""

    m_min: int
    m_max: int
    shape: str
    choice: int | str


class _Row(msgspec.Struct, frozen=True):
    name: str
    build: str
    abi: str
    ok: bool
    detail: str
    rules: tuple[Rule, ...] = ()


class _Document(msgspec.Struct, frozen=True):
    schema: int
    sm: int
    verdicts: tuple[_Row, ...]
    #: Provenance (card, clocks, runtime, torch) and the raw measurements: never matched.
    measured: msgspec.Raw = msgspec.Raw(b"null")
    evidence: msgspec.Raw = msgspec.Raw(b"null")


def shape_class(n: int, k: int) -> str:
    """A projection's class: an FFN's up and down projections are 3-8x wide, attention's ~1x;
    a 2x ratio separates them."""
    return "up" if n >= 2 * k else "down" if k >= 2 * n else "square"


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
    detail = f"rel L2 {errors[worst]:.2e} at {worst} (budget {budget:.2e}, {len(errors)} cases)"
    return Verdict(errors[worst] <= budget, detail)


_OFFLINE: ContextVar[dict[tuple[Kernel, int], Verdict] | None] = ContextVar(
    "kernel_offline_verdicts", default=None
)


def in_oracle() -> bool:
    return _OFFLINE.get() is not None


@contextmanager
def measuring() -> Iterator[dict[tuple[Kernel, int], Verdict]]:
    """Development oracle scope: collect the exact kernel arms actually qualified."""
    records: dict[tuple[Kernel, int], Verdict] = {}
    token = _OFFLINE.set(records)
    try:
        yield records
    finally:
        _OFFLINE.reset(token)


def measured(kernel: Kernel, sm: int, probe: Callable[[], Verdict]) -> Verdict:
    """The development oracle's probe, recorded for publication (inside `measuring` only)."""
    records = _OFFLINE.get()
    assert records is not None, "a consumer never probes"
    try:
        verdict = probe()
    except Exception as exc:
        verdict = Verdict(False, f"{type(exc).__name__}: {exc}"[:400])
    records[kernel, sm] = verdict
    return verdict


def admitted(
    kernel: Kernel, sm: int, probe: Callable[[], Verdict], *, tables: Path | None = None
) -> Verdict:
    """A shipped verdict, or a real probe only within the explicit offline oracle scope."""
    if in_oracle():
        return measured(kernel, sm, probe)
    found = shipped(kernel, sm, tables=tables)
    return found or Verdict(False, f"no matching offline verdict for {kernel.name} on sm{sm}")


def path(sm: int, tables: Path | None = None) -> Path:
    return (TABLES if tables is None else tables) / f"kernels-sm{sm}.json"


@lru_cache(maxsize=16)
def _rows(file: Path, changed: int, size: int) -> dict[Kernel, _Row]:
    try:
        document = msgspec.json.decode(file.read_bytes(), type=_Document)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError):
        return {}
    if document.schema != SCHEMA or file.name != path(document.sm, file.parent).name:
        return {}
    return {Kernel(row.name, row.build, row.abi): row for row in document.verdicts}


def _shipped_row(kernel: Kernel, sm: int, tables: Path | None) -> _Row | None:
    file = path(sm, tables)
    try:
        stat = file.stat()
    except OSError:
        return None
    return _rows(file, stat.st_mtime_ns, stat.st_size).get(kernel)


def shipped(kernel: Kernel, sm: int, *, tables: Path | None = None) -> Verdict | None:
    """The release's offline verdict for this exact kernel on this card, or None."""
    row = _shipped_row(kernel, sm, tables)
    return None if row is None else Verdict(row.ok, row.detail)


def rules(kernel: Kernel, sm: int, *, tables: Path | None = None) -> tuple[Rule, ...]:
    """The measured workload rules refining this kernel's family default on this card."""
    row = _shipped_row(kernel, sm, tables)
    return row.rules if row is not None and row.ok else ()


def publish(
    directory: Path,
    sm: int,
    records: Mapping[tuple[Kernel, int], Verdict],
    rules: Mapping[Kernel, Sequence[Rule]],
    measured: Mapping[str, object],
    evidence: object,
) -> Path:
    """Export one card's real offline verdicts, rules and raw measurements."""
    rows = [
        {
            "name": kernel.name,
            "build": kernel.build,
            "abi": kernel.abi,
            "ok": verdict.ok,
            "detail": verdict.detail,
            "rules": [msgspec.to_builtins(rule) for rule in rules.get(kernel, ())],
        }
        for (kernel, card), verdict in sorted(records.items(), key=lambda item: item[0][0].name)
        if card == sm
    ]
    document = {
        "schema": SCHEMA,
        "sm": sm,
        "verdicts": rows,
        "measured": measured,
        "evidence": evidence,
    }
    directory.mkdir(parents=True, exist_ok=True)
    target = path(sm, directory)
    temporary = target.with_name(f".{target.name}.{os.getpid()}")
    temporary.write_text(json.dumps(document, sort_keys=True, allow_nan=False))
    os.replace(temporary, target)
    return target


@lru_cache(maxsize=32)
def implementation_artifact(module: ModuleType) -> str:
    """Hash the imported wrapper and native implementation, not its recipe's display name.

    Package trees include their fixed JIT sources. Single-file wrappers also bind imported
    extension modules, such as flash_attn_interface's separately installed CUDA module.
    """
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
    if module.__file__:
        files.add(Path(module.__file__))
    for imported in vars(module).values():
        if isinstance(imported, ModuleType) and imported.__file__:
            path = Path(imported.__file__)
            if path.suffix in (".so", ".pyd", ".dll", ".dylib"):
                files.add(path)
    if not files:
        return "ABSENT"
    # Paths outside the installed tree are not part of identity; compiled bytes are.
    return canonical.digest(sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in files))
