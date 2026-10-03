"""The weight plane's one door into cozy-runtime (`tensorfs.plane`, weight-plane.md §3).

The plane is mechanism: it never measures free memory, never picks a victim among equals and
never reorders a schedule. It maps, copies, streams and counts; `weights.py` and
`weight_policy.py` decide. Everything the runtime reads back from it is typed here.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
from types import ModuleType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from tensorfs import ReadLease, Store
    from tensorfs.plane import Plane, Source

import msgspec

_plane: ModuleType | None
try:
    from tensorfs import plane as _plane
except ImportError:  # an older TensorFS: the executor offers no `weight_plane/1`
    _plane = None

#: The executor capability a Worker negotiates on (`hello.memory`).
CAPABILITY = "weight_plane/1"
#: GPU memory the machine keeps across executors: attach on `Load.device_weights`, `share`
#: after a call, `revoke` (TensorFS `Plane.export_device`).
SHARING = ("weights.attach/1", "weights.revoke/1")
PINNED: Final = "pinned"
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
    #: -1: a TensorFS before 0.3.92 does not count it (a 0 there was never a measurement)
    disk_read_bytes: int = -1
    #: RAM of the machine's sealed layouts this plane maps (theirs, not this process's)
    sealed_bytes: int = 0
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
    #: shared regions mapped here (exported or attached): outside `committed` and the budget
    shared_bytes: int = 0
    #: of the host pair, the copies from mappings of objects the page cache held; the rest
    #: came from the pinned tier (TensorFS 0.3.92; 0 before)
    mapped_copy_bytes: int = 0
    mapped_copy_ns: int = 0
    misses: int = 0
    evictions: int = 0
    evicted_bytes: int = 0


class SetDevice(msgspec.Struct, frozen=True, kw_only=True):
    ordinal: int
    ready_bytes: int = 0
    shared_bytes: int = 0


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


def sharing() -> bool:
    """This TensorFS can export and attach GPU regions (whether the GPU can is the plane's
    `device_sharing(ordinal)`)."""
    return _plane is not None and hasattr(_plane.Plane, "export_device")


#: Linux `fcntl` seals, by number (python-build-standalone may lack the names).
_GET_SEALS, _ADD_SEALS = 1034, 1033
#: seal, shrink, grow, write: what a plan memfd carries
_SEALS = 0x8 | 0x4 | 0x2 | 0x1
#: a sealed layout: shrink, grow and sealing itself, and write or future write (its filler may
#: still be unmapping; no other mapping or write can ever change a byte)
_LAYOUT_SEALS, _WRITE_SEALS = 0x1 | 0x2 | 0x4, 0x8 | 0x10


def sealed_available() -> bool:
    """This TensorFS adopts sealed host layouts (`Plane.register_sealed`)."""
    return _plane is not None and hasattr(_plane.Plane, "register_sealed")


def sealed_filling() -> bool:
    """This TensorFS adopts a sealed layout still filling, waiting per region."""
    return sealed_available() and getattr(_plane, "SEALED_WHILE_FILLING", False) is True


def is_sealed(fd: int) -> bool:
    """Whether `fd` is a memfd nobody can write, resize or unseal: a sealed host layout."""
    try:
        seals = fcntl.fcntl(fd, _GET_SEALS)
    except OSError:
        return False
    return seals & _LAYOUT_SEALS == _LAYOUT_SEALS and bool(seals & _WRITE_SEALS)


def sealed_memfd(name: str, body: bytes) -> int:
    """`body` in a new memfd, sealed: what crosses beside a request too large for a frame."""
    # libc's own: python-build-standalone's old glibc build lacks `os.memfd_create`.
    # MFD_CLOEXEC | MFD_ALLOW_SEALING
    fd: int = ctypes.CDLL(None, use_errno=True).memfd_create(name.encode(), 0x1 | 0x2)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "memfd_create")
    try:
        view = memoryview(body)
        while view:
            view = view[os.write(fd, view) :]
        fcntl.fcntl(fd, _ADD_SEALS, _SEALS)
    except BaseException:
        os.close(fd)
        raise
    return fd


def open_plane(device: int | None) -> Plane:
    """One plane per executor process (phase 1), over this rank's device; `None` opens the
    host tier alone (no CUDA), which fills pinned memory before any device is granted."""
    if _plane is None:
        raise Unavailable("this TensorFS build carries no weight plane")
    try:
        opened: Plane = _plane.Plane(devices=[] if device is None else [device])
        return opened
    except _plane.PlaneError as exc:  # no CUDA driver here (CPU or MPS executor)
        raise Unavailable(f"the weight plane cannot open device {device}: {exc}") from exc


def source(plane: Plane, store: Store, lease: ReadLease) -> Source:
    """One manifest's read source, shared by its components' weight sets; consumes `lease`."""
    return plane.source(store, lease)


def stats(plane: Plane) -> Stats:
    return msgspec.convert(plane.stats(), Stats)


def events(plane: Plane) -> list[Event]:
    return msgspec.convert(plane.events(), list[Event])


def hold_tier(fd: int) -> int:
    """Claim every region of a pinned-tier memfd for the Worker (no plane of its own): while
    it holds the returned descriptor no executor's eviction or close frees the tier's RAM. A
    TensorFS before claims keeps the descriptor as it is."""
    hold = getattr(_plane, "hold_tier", None)
    return int(hold(fd)) if hold is not None else os.dup(fd)


def release_tier(hold: int) -> None:
    """End `hold_tier`: frees every region no executor claims, and closes the descriptor."""
    release = getattr(_plane, "release_tier", None)
    if release is not None:
        release(hold)
    else:
        os.close(hold)


def error(name: str) -> type[BaseException]:
    """The plane's typed error class `name` (`Shortfall`, `BudgetExceeded`, ...)."""
    assert _plane is not None
    found: type[BaseException] = getattr(_plane, name)
    return found
