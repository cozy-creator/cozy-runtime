"""How many GPUs of a group a sharded stage runs on (cr-348 C): its group's degree or one GPU,
whichever was measured faster for its key, never estimated. Ulysses halves a step's device work,
not its host work, so no figure taken at one degree predicts the other where the host bounds the
step (LTX-2.5's base stage: 1.67 s a step on two GPUs, 0.925 s on one).

A key is the package and its build, the function, the stage, its shape class (the tensor shapes
of the stage's first sharded call, where the degree is chosen: no seed or other value), the card,
the group's degree and the Runtime build: a new release measures again. A key with no steady step
time at a degree runs there and records it (steps 2 on: step 1 carries fills and compiles; and
only steps no GPU read weights from disk or saw a kernel build in, else the next request measures
again); once both are known, the faster is the key's for good, so a request's bytes stop changing
with its runs. The record keeps the attention kernels each degree served, so a reviewer can tell a
degree's head-group numerics from a bug. Records live in the machine's kernel cache, beside its
link measurements.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from cozy_runtime.internal import filelock, kernel_cache


@dataclass(frozen=True, slots=True)
class Key:
    package: str
    function: str
    stage: str
    shape: str
    card: str
    world: int
    runtime: str

    @classmethod
    def of(cls, document: Mapping[str, object]) -> Key:
        return cls(
            str(document["package"]),
            str(document["function"]),
            str(document["stage"]),
            str(document["shape"]),
            str(document["card"]),
            int(cast("int", document["world"])),
            str(document["runtime"]),
        )

    def digest(self) -> str:
        """Its record's name on this machine."""
        return hashlib.sha256(json.dumps(self.document(), sort_keys=True).encode()).hexdigest()[:32]

    def document(self) -> dict[str, object]:
        return {
            "package": self.package,
            "function": self.function,
            "stage": self.stage,
            "shape": self.shape,
            "card": self.card,
            "world": self.world,
            "runtime": self.runtime,
        }


def _directory() -> Path | None:
    own = os.environ.get(kernel_cache.OWN_ENV, "").strip()
    return Path(own) / "degrees" if own and Path(own).is_absolute() else None


def _path(key: Key) -> Path | None:
    directory = _directory()
    if directory is None:
        return None
    return directory / f"{key.digest()}.json"


def known(key: Key) -> dict[int, dict[str, object]]:
    """Each degree's steady step record for `key` on this machine."""
    path = _path(key)
    if path is None:
        return {}
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {int(degree): row for degree, row in document.get("degrees", {}).items()}


def choose(key: Key) -> int:
    """The degree a stage of `key` runs at: the group's while it has no time there, then one GPU
    while it has none there, then the faster."""
    times = known(key)
    if key.world not in times:
        return key.world
    if 1 not in times:
        return 1
    return min((1, key.world), key=lambda d: float(str(times[d]["steady_ms"])))


def record(key: Key, degree: int, row: Mapping[str, object]) -> None:
    """`degree`'s steady step time and the kernels it served, for `key`; a degree already known
    keeps its first record (its choice must not move)."""
    path = _path(key)
    if path is None:
        return
    with contextlib.suppress(OSError, ValueError):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".lock"), "a+") as lock:
            filelock.lock(lock)
            document = json.loads(path.read_text()) if path.exists() else {"key": key.document()}
            degrees = document.setdefault("degrees", {})
            if str(degree) in degrees:
                return
            degrees[str(degree)] = dict(row)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(document, sort_keys=True))
            temporary.replace(path)
