"""How many GPUs of a group a sharded stage runs on (cr-348 C): its group's degree or one GPU,
whichever was measured faster for its key. Ulysses halves a step's device work, not its host work
or its weights, so no figure taken at one degree predicts the other (LTX-2.5's base stage: 1.67 s
a step on two GPUs, 0.925 s on one; on two 24 GB cards that page its weights, 4-6x slower).

A key is the package and its build, the function, the stage, its shape class (the stage's first
sharded call, where the degree is chosen: its video's exact shape, its other tensors' dimensions
to the next power of two, no seed or other value: `spmd._shape_class`), the card, the group's
degree and the Runtime build: a new release measures again.

Two GPUs should simply be faster (Paul 11-19), so a key runs at its group's degree until measured
otherwise, and wherever nothing can be measured (no kernel cache to keep records in, a stage of
one sharded call). The group's degree is measured first, then one GPU (one GPU runs each sharded
call and its output crosses, `spmd`). A measurement whose steps loaded weights or saw a kernel
build or a graph recorded is cold, and is measured once more, warm, then kept (`measured`). Where
one GPU is known first, the group's degree is tried against its second step, and a slower one
moves the rest of that stage to one GPU (`spmd._Choice._trial`). Once both are known, one GPU is
the key's only where it is more than the ledger's noise faster (Paul 11-15: within 5% is a tie),
and for good. The record keeps the attention kernels each degree served, so a reviewer can tell a
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


#: Paul 11-15: a difference within 5% is a tie; a tie goes to the group's degree.
NOISE = 0.05


@dataclass(frozen=True, slots=True)
class Decision:
    degree: int
    measure: bool = False
    #: trying the group's degree: one GPU's second step, which its own must not exceed by NOISE
    against_ms: float | None = None


def _open(row: Mapping[str, object] | None) -> bool:
    """No record, or a cold one not yet measured again."""
    return row is None or (bool(row.get("cold")) and int(cast("int", row.get("measured", 1))) < 2)


def choose(key: Key) -> Decision:
    """How a stage of `key` runs."""
    if _directory() is None:  # nothing can be recorded: the group's degree
        return Decision(key.world)
    times = known(key)
    group, one = times.get(key.world), times.get(1)
    if _open(group):
        settled = one is not None and not _open(one)
        against = float(str(one.get("second_ms", one["steady_ms"]))) if settled else None
        return Decision(key.world, measure=True, against_ms=against)
    if _open(one):
        return Decision(1, measure=True)
    assert group is not None and one is not None
    faster = float(str(one["steady_ms"])) * (1 + NOISE) < float(str(group["steady_ms"]))
    return Decision(1 if faster else key.world)


def record(key: Key, degree: int, row: Mapping[str, object]) -> None:
    """`degree`'s steady step time and the kernels it served, for `key`. A degree already known
    keeps its record (its choice must not move), except a cold one, which its next measurement
    replaces once (`measured`)."""
    path = _path(key)
    if path is None:
        return
    with contextlib.suppress(OSError, ValueError):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".lock"), "a+") as lock:
            filelock.lock(lock)
            document = json.loads(path.read_text()) if path.exists() else {"key": key.document()}
            degrees = document.setdefault("degrees", {})
            held = degrees.get(str(degree))
            if not _open(held):
                return
            measured = 1 + (int(held.get("measured", 1)) if held is not None else 0)
            degrees[str(degree)] = {**row, "measured": measured}
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(document, sort_keys=True))
            temporary.replace(path)
