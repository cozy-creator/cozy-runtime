"""The weight plane's one door into cozy-runtime (`tensorfs.plane`, weight-plane.md §3).

The plane is mechanism: it never measures free memory, never picks a victim among equals and
never reorders a schedule. It maps, copies, streams and counts; `weights.py` and
`weight_policy.py` decide. Everything the runtime reads back from it is typed here.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import msgspec

_plane: ModuleType | None
try:
    from tensorfs import plane as _plane
except ImportError:  # an older TensorFS: the executor offers no `weight_plane/1`
    _plane = None

#: The executor capability a Worker negotiates on (`hello.memory`).
CAPABILITY = "weight_plane/1"
PINNED = "pinned"
#: Priorities are plain integers; the plane evicts ascending and never picks among equals, so
#: every band carries the LRU clock. Idle regions (a finished stage's) sit lowest; a running
#: stage's streamed and demand-filled blocks above them, each newer one above the older; its
#: resident prefix above everything, so nothing in the stage evicts it.
STREAMED = 1 << 40
PREFIX = 2 << 40


class Unavailable(Exception):
    """This environment has no weight plane."""


class HostStats(msgspec.Struct, frozen=True, kw_only=True):
    budget: int = -1
    used: int = -1
    ready_bytes: int = 0
    registered_bytes: int = 0
    fill_bytes: int = 0
    fill_ns: int = 0
    disk_read_bytes: int = 0
    evictions: int = 0


class DeviceStats(msgspec.Struct, frozen=True, kw_only=True):
    ordinal: int
    budget: int = -1
    committed: int = -1
    leased_bytes: int = 0
    held_bytes: int = 0
    host_copy_bytes: int = 0
    host_copy_ns: int = 0
    disk_copy_bytes: int = 0
    disk_copy_ns: int = 0
    misses: int = 0
    evictions: int = 0
    evicted_bytes: int = 0


class SetDevice(msgspec.Struct, frozen=True, kw_only=True):
    ordinal: int
    ready_bytes: int = 0


class SetStats(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    nbytes: int = 0
    host_ready_bytes: int = 0
    devices: tuple[SetDevice, ...] = ()


class StageStats(msgspec.Struct, frozen=True, kw_only=True):
    ws: str = ""
    device: int = -1
    tag: str = ""
    acquires: int = 0
    skipped: int = 0
    misses: int = 0
    late: int = 0
    late_bytes: int = 0
    stall_ns: int = 0
    streamed_bytes: int = 0


class RegionStats(msgspec.Struct, frozen=True, kw_only=True):
    ws: str = ""
    device: int = -1
    tag: str = ""
    region: int = -1
    uses: int = 0
    compute_ns: int = 0
    stall_ns: int = 0
    late: int = 0


class Stats(msgspec.Struct, frozen=True, kw_only=True):
    """`Plane.stats()` as the runtime reads it. Unknown keys are ignored."""

    host: HostStats = HostStats()
    devices: tuple[DeviceStats, ...] = ()
    sets: tuple[SetStats, ...] = ()
    stages: tuple[StageStats, ...] = ()
    regions: tuple[RegionStats, ...] = ()
    poisoned: str | None = None

    def device(self, ordinal: int) -> DeviceStats:
        return next((d for d in self.devices if d.ordinal == ordinal), DeviceStats(ordinal=ordinal))

    def ready(self, name: str, ordinal: int) -> int:
        for row in self.sets:
            if row.name == name:
                return sum(d.ready_bytes for d in row.devices if d.ordinal == ordinal)
        return 0


class Event(msgspec.Struct, frozen=True, kw_only=True):
    kind: str
    ws: str | None = None
    region: int | None = None
    device: int | None = None
    bytes: int = 0
    ns: int = 0
    detail: str = ""


def available() -> bool:
    return _plane is not None


def open_plane(device: int | None) -> Any:
    """One plane per executor process (phase 1), over this rank's device; `None` opens the
    host tier alone (no CUDA), which fills pinned memory before any device is granted."""
    if _plane is None:
        raise Unavailable("this TensorFS build carries no weight plane")
    try:
        return _plane.Plane(devices=[] if device is None else [device])
    except _plane.PlaneError as exc:  # no CUDA driver here (CPU or MPS executor)
        raise Unavailable(f"the weight plane cannot open device {device}: {exc}") from exc


def source(plane: Any, store: Any, lease: Any) -> Any:
    """One manifest's read source, shared by its components' weight sets; consumes `lease`."""
    return plane.source(store, lease)


def stats(plane: Any) -> Stats:
    return msgspec.convert(plane.stats(), Stats)


def events(plane: Any) -> list[Event]:
    return msgspec.convert(plane.events(), list[Event])


def error(name: str) -> type[BaseException]:
    """The plane's typed error class `name` (`Shortfall`, `BudgetExceeded`, ...)."""
    assert _plane is not None
    found: type[BaseException] = getattr(_plane, name)
    return found
