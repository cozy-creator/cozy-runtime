"""How many GPUs of a group a sharded stage runs on (cr-348 C): its group's degree or one GPU,
whichever was measured faster for its key. Ulysses halves a step's device work, not its host work
or its weights, so no figure taken at one degree predicts the other (LTX-2.5's base stage: 1.67 s
a step on two GPUs, 0.925 s on one; on two 24 GB cards that page its weights, 4-6x slower).

A key is the package and its build, the function, the stage, its shape class (the stage's first
sharded call, where the degree is chosen: its video's exact shape, its other tensors' dimensions
to the next power of two, no seed or other value: `spmd._shape_class`), the card, the group's
degree and the Runtime build: a new release measures again.

One GPU is measured first: GPU 0 runs each sharded call and its output crosses (`spmd`), so it
costs a request what one GPU does. It is measured again while its record is cold (a step read
weights from disk or saw a kernel build), as that costs nothing. The group's degree is then tried
once against it, and the trial moves the rest of its stage to one GPU after a slower second step
(`spmd._Choice`); its record, cold or not, is never measured again. It is not tried at all where
one GPU's last sharded call paged weights onto the card and the group crosses through host
memory: Ulysses splits the sequence, not the weights, so every GPU pages them at once through the
path the exchange takes. Once both are known the faster is the key's for good. The record keeps
the attention kernels each degree served, so a reviewer can tell a degree's head-group numerics
from a bug. Records live in the machine's kernel cache, beside its link measurements.
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


@dataclass(frozen=True, slots=True)
class Decision:
    degree: int
    measure: bool = False
    #: trying the group's degree: one GPU's steady step its second step must not exceed
    against_ms: float | None = None
    #: why the group's degree is not tried
    skipped: str = ""


def choose(key: Key, route: str) -> Decision:
    """How a stage of `key` runs where its group's exchange takes `route` (`Transport.route`)."""
    times = known(key)
    one = times.get(1)
    if one is None:
        return Decision(1, measure=True)
    cold = bool(one.get("cold"))
    if key.world not in times:
        if route == "host-staged" and int(cast("int", one.get("paged_bytes", 0))):
            why = "one GPU's last sharded call paged weights, and the exchange crosses host memory"
            return Decision(1, measure=cold, skipped=why)
        if cold:  # a steady step to try the group's degree against
            return Decision(1, measure=True)
        return Decision(key.world, measure=True, against_ms=float(str(one["steady_ms"])))
    return Decision(min((1, key.world), key=lambda d: float(str(times[d]["steady_ms"]))))


def record(key: Key, degree: int, row: Mapping[str, object]) -> None:
    """`degree`'s steady step time and the kernels it served, for `key`. A degree already known
    keeps its record (its choice must not move), except one GPU's cold record, which a later
    measurement replaces."""
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
            if held is not None and not (degree == 1 and held.get("cold")):
                return
            degrees[str(degree)] = dict(row)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(document, sort_keys=True))
            temporary.replace(path)
