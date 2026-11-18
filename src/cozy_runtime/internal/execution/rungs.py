"""Each graph's measured capture pool, remembered per machine (cr-305, design
`varena-activation-plan`): the graph rung is chosen only when its weights and that pool fit the room
at the time. A split's host rung is not remembered: it holds for its request only (cr-337).

A pool is kept under a key naming everything its measure depends on: the GPU and its total memory,
torch's and CUDA's builds, the Runtime's version, the running models' names (each carries its
package release and checkpoint digest, so its lane) and the graph with its shapes. The table lives
beside the executor's kernels (`COZY_KERNEL_CACHE`), the machine's own namespace, which a fair A/B
arm resets.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import threading
from importlib import metadata
from pathlib import Path

import torch

from cozy_runtime.internal.kernel_cache import OWN_ENV

_LOCK = threading.Lock()
_TABLE: dict[str, int] | None = None


def key(site: str, models: list[str], device: int) -> str:
    """The pool's key: everything the measure behind it depends on."""
    parts = [*_machine(device), *sorted(models), site]
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


def pool(pool_key: str) -> int:
    """The capture pool a graph kept under `pool_key` measured, 0 when none did."""
    with _LOCK:
        return _table().get(pool_key, 0)


def put(pool_key: str, measured: int) -> None:
    """Keep a graph's measured pool for `pool_key`, on disk when the machine gives the executor a
    namespace."""
    with _LOCK:
        table = _table()
        if table.get(pool_key) == measured:
            return
        table[pool_key] = measured
        path = _path()
        if path is None:
            return
        with contextlib.suppress(OSError):  # a table that can't be written is measured again
            on_disk = _read(path)
            on_disk[pool_key] = measured
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(f".{os.getpid()}.tmp")
            partial.write_text(json.dumps({k: {"pool": v} for k, v in on_disk.items()}))
            os.replace(partial, path)


def _table() -> dict[str, int]:
    global _TABLE
    if _TABLE is None:
        path = _path()
        _TABLE = _read(path) if path is not None else {}
    return _TABLE


def _path() -> Path | None:
    root = os.environ.get(OWN_ENV)
    return Path(root) / "rungs.json" if root else None


def _read(path: Path) -> dict[str, int]:
    """The pools on disk. A split's host rung an earlier build kept there is dropped."""
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    table: dict[str, int] = {}
    for pool_key, row in rows.items() if isinstance(rows, dict) else ():
        with contextlib.suppress(TypeError, ValueError, KeyError):
            if "pool" in row:
                table[pool_key] = int(row["pool"])
    return table
