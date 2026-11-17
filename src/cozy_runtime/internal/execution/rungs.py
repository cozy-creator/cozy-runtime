"""Each split site's rung, remembered per machine (cr-305, design `varena-activation-plan`): where its
whole output lives (the card or pinned host memory), the whole's measured shape and dtype, and the room
the rung was chosen in.

A rung is kept under a key naming everything its measure depends on: the GPU and its total memory,
torch's and CUDA's builds, the Runtime's version, the running models' names (each carries its package
release and checkpoint digest, so its lane) and the site with its shapes. A rung is trusted only while
the room it was chosen in still holds: a card rung while the room is no smaller, a host rung while it
is no larger; otherwise the site chooses again. The table lives beside the executor's kernels
(`COZY_KERNEL_CACHE`), the machine's own namespace, which a fair A/B arm resets.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import torch

from cozy_runtime.internal.kernel_cache import OWN_ENV

#: A room within this fraction of the recorded one is the same room: capacity moves by context and
#: library growth (a few MiB) between calls.
_SAME_ROOM = 64


@dataclass(frozen=True, slots=True)
class Rung:
    host: bool
    shape: tuple[int, ...]
    dtype: str
    room: int

    def trusted(self, room: int) -> bool:
        """Whether this rung still holds in `room`: a host rung while the room did not grow, a card
        rung while it did not shrink."""
        slack = self.room // _SAME_ROOM
        return room <= self.room + slack if self.host else room >= self.room - slack


_LOCK = threading.Lock()
_TABLE: dict[str, Rung] | None = None


def key(site: str, models: list[str], device: torch.device) -> str:
    """The rung's key: everything the measure behind it depends on."""
    parts = [*_machine(device.index or 0), *sorted(models), site]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:32]


@functools.cache
def _machine(index: int) -> tuple[str, ...]:
    """The GPU, its total memory, torch's and CUDA's builds and the Runtime's version."""
    return (
        torch.cuda.get_device_name(index),
        str(torch.cuda.get_device_properties(index).total_memory),
        torch.__version__,
        str(torch.version.cuda),
        metadata.version("cozy-runtime"),
    )


def get(rung_key: str, room: int) -> Rung | None:
    """The rung kept for `rung_key`, if it holds in `room`."""
    with _LOCK:
        found = _table().get(rung_key)
    return found if found is not None and found.trusted(room) else None


def put(rung_key: str, rung: Rung) -> None:
    """Keep `rung` for `rung_key`, on disk when the machine gives the executor a namespace."""
    with _LOCK:
        table = _table()
        if table.get(rung_key) == rung:
            return
        table[rung_key] = rung
        path = _path()
        if path is None:
            return
        with contextlib.suppress(OSError):  # a table that can't be written is chosen again
            on_disk = _read(path)
            on_disk[rung_key] = rung
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(f".{os.getpid()}.tmp")
            partial.write_text(json.dumps({k: asdict(v) for k, v in on_disk.items()}))
            os.replace(partial, path)


def _table() -> dict[str, Rung]:
    global _TABLE
    if _TABLE is None:
        path = _path()
        _TABLE = _read(path) if path is not None else {}
    return _TABLE


def _path() -> Path | None:
    root = os.environ.get(OWN_ENV)
    return Path(root) / "rungs.json" if root else None


def _read(path: Path) -> dict[str, Rung]:
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    table: dict[str, Rung] = {}
    for rung_key, row in rows.items() if isinstance(rows, dict) else ():
        with contextlib.suppress(TypeError, ValueError):
            table[rung_key] = Rung(
                bool(row["host"]),
                tuple(int(n) for n in row["shape"]),
                str(row["dtype"]),
                int(row["room"]),
            )
    return table
