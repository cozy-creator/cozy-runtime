"""The weight plane's one door into cozy-runtime (`tensorfs-plane`, weight-plane.md §3).

Everything the runtime asks of the plane goes through this file, so the plane's Python API
can move without touching the executor. The plane is mechanism: it never measures free
memory, never picks a victim among equals and never reorders a schedule. It maps, copies,
streams and counts; `weights.py` and `weight_policy.py` decide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import msgspec

try:
    from tensorfs import plane as _plane
except ImportError:  # an older TensorFS: the executor serves without `weight_plane/1`
    _plane = None

#: The executor capability a Worker negotiates on (`hello.memory`).
CAPABILITY = "weight_plane/1"
#: Byte alignment of every stored part inside a region (R3): GEMM operands want 16 B, the
#: plane's DMA wants 256 B.
ALIGN = 256
#: Priority classes. The plane evicts only unleased regions of strictly lower priority; within
#: a class the LRU clock orders them.
IDLE, ACTIVE = 1, 2


class Unavailable(Exception):
    """This environment has no weight plane."""


class Lease(Protocol):
    """One region held for one consumer stream until `release`."""

    @property
    def address(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    def view(self) -> Any: ...

    def release(self, stream: int) -> None: ...


class Cursor(Protocol):
    def acquire(self, region: int, stream: int) -> Lease: ...

    def close(self) -> None: ...


class Plane(Protocol):
    """The mechanism surface the runtime drives (§5.1 of the substrate study)."""

    def register(
        self,
        content: tuple[str, str, str],
        regions: Sequence[Region],
        *,
        source: Any,
        header: bytes,
    ) -> int: ...

    def unregister(self, weights: int) -> None: ...

    def set_vram_budget(self, device: int, nbytes: int) -> Any: ...

    def set_pinned_budget(self, nbytes: int) -> Any: ...

    def want(
        self, weights: int, tier: Any, regions: Sequence[int], priority: tuple[int, int]
    ) -> Any: ...

    def drop(self, weights: int, tier: Any, regions: Sequence[int]) -> int: ...

    def stream(
        self, weights: int, device: int, order: Sequence[int], window: int, repeat: int
    ) -> Cursor: ...

    def acquire(self, weights: int, device: int, region: int, stream: int) -> Lease: ...

    def resident(self, weights: int, device: int) -> frozenset[int]: ...

    def stats(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Part:
    """One stored role of one checkpoint tensor inside a region."""

    what: str
    """TensorFS's `<component>/<name>#<role>`: the plane plans the read over exactly these."""
    offset: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class Region:
    index: int
    parts: tuple[Part, ...]
    nbytes: int


class Stats(msgspec.Struct, frozen=True, kw_only=True):
    """The counters the runtime reads from `Plane.stats()` (R7). Unknown keys are ignored."""

    budget: int = -1
    committed: int = -1
    leased: int = -1
    pinned_budget: int = -1
    pinned: int = -1
    resident: dict[str, int] = {}
    h2d_bytes: int = 0
    h2d_gbps: float = 0.0
    late: int = 0
    stall_ns: int = 0
    misses: int = 0
    evictions: int = 0


def available() -> bool:
    return _plane is not None


def layout(parts: Sequence[tuple[str, int]]) -> tuple[tuple[Part, ...], int]:
    """Lay stored parts out contiguously in the given order, each aligned (R3)."""
    placed: list[Part] = []
    offset = 0
    for what, nbytes in parts:
        offset = -(-offset // ALIGN) * ALIGN
        placed.append(Part(what, offset, nbytes))
        offset += nbytes
    return tuple(placed), offset


def open_plane(device: int, *, host_tier: Any = None) -> Plane:
    """One plane per executor process (phase 1), over this rank's device."""
    if _plane is None:
        raise Unavailable("this TensorFS build carries no weight plane")
    opened: Plane = _plane.Plane.open(devices=[device], host_tier=host_tier)
    return opened


def vram(device: int) -> Any:
    assert _plane is not None
    return _plane.Vram(device)


def pinned() -> Any:
    assert _plane is not None
    return _plane.Pinned


def stats(plane: Plane) -> Stats:
    return msgspec.convert(dict(plane.stats()), Stats)
