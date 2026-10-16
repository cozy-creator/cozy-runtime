"""The weight plane of the unified memory class (design §6.2): one pool, so the host tier is the
OS page cache of the store's objects, and a resident region is one device tensor read from it.

It speaks `tensorfs.plane`'s contract (`crates/tensorfs-plane/API.md`) for the calls Runtime
makes, so `weights.py`'s residency, paging and out-of-memory recovery run unchanged on Apple
GPUs. What it has not: a pinned tier (the page cache is it), a byte ring (a streamed region is
read where it is used and its cursor lets it go `window` regions later), sealed layouts and
shared GPU handles. Fills copy on the caller's thread. A region's bytes are freed when nothing
binds them any more: `weights.py` unbinds an evicted region as on a discrete card.
"""

from __future__ import annotations

import mmap
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING

import tensorfs
from tensorfs import ReadLease, ReadPlan, Store
from tensorfs import plane as _plane

if TYPE_CHECKING:
    from tensorfs.plane import Part, Region, Trim
    from tensorfs.records import Header, ReadItem
    from torch import Tensor
    from torch import device as Device

#: `stats()` and an event, as `tensorfs.plane` spells them (Runtime reads them typed)
type Facts = dict[str, object]

#: Every part starts on a page, so any dtype views it.
ALIGN = 4096


def _aligned(n: int) -> int:
    return -(-n // ALIGN) * ALIGN


class _Source:
    """Bytes of the plan's rows: objects by digest, inline parts from the parsed header."""

    def __init__(self, header: Header) -> None:
        self.components = header["components"]

    def object(self, row: ReadItem, into: memoryview) -> None:
        raise NotImplementedError

    def read(self, row: ReadItem, within: int, into: memoryview) -> None:
        if row["source"] == "object":
            self.object(row, into)
            return
        component, _, rest = row["what"].partition("/")
        key, _, role = rest.rpartition("#")
        inline = self.components[component][key]["parts"][role]["inline"]  # type: ignore[typeddict-item]
        into[:] = inline[within : within + row["len"]]

    def close(self) -> None:
        pass


class _LeaseSource(_Source):
    """An explicitly opened Store's read lease (standalone use)."""

    def __init__(self, store: Store, lease: ReadLease) -> None:
        header = store.manifest(lease.manifest)["header"]
        if header is None:
            raise _plane.Invalid(f"{lease.manifest} has no cozytensors header")
        super().__init__(tensorfs.parse_header(header))
        self.lease = lease

    def object(self, row: ReadItem, into: memoryview) -> None:
        self.lease.read_into(row["object"], row["object_length"], row["off"], row["len"], into)

    def close(self) -> None:
        self.lease.release()


class _FileSource(_Source):
    """The machine's verified object files, mapped read-only: their pages are the host tier."""

    def __init__(self, files: Iterable[tuple[str, int, int]], header: Header) -> None:
        super().__init__(header)
        self.maps = {
            sha.removeprefix("sha256:"): mmap.mmap(fd, length, access=mmap.ACCESS_READ)
            for sha, length, fd in files
        }

    def object(self, row: ReadItem, into: memoryview) -> None:
        mapped = self.maps[row["object"].removeprefix("sha256:")]
        with memoryview(mapped) as view:
            into[:] = view[row["off"] : row["off"] + row["len"]]

    def close(self) -> None:
        for mapped in self.maps.values():
            mapped.close()


@dataclass(slots=True)
class _Region:
    index: int
    offset: int
    span: int
    #: (plan row, offset in this region, offset in its part)
    copies: list[tuple[ReadItem, int, int]] = field(default_factory=list)
    tensor: Tensor | None = None
    priority: int = 0
    pinned: bool = False
    leases: int = 0
    held: int = 0


class WeightSet:
    """One component's layout: its regions and parts as `tensorfs.plane.WeightSet` states them."""

    def __init__(
        self,
        plane: UnifiedPlane,
        name: str,
        source: _Source,
        plan: ReadPlan,
        regions: Sequence[Sequence[str]],
        serial: int,
    ) -> None:
        self.plane, self.name, self.source, self.serial = plane, name, source, serial
        self.digest = name
        member = {what: r for r, names in enumerate(regions) for what in names}
        grouped: list[dict[str, list[ReadItem]]] = [{} for _ in regions]
        for row in plan.items():
            what = row["what"]
            r = member.get(what, member.get(what.partition("#")[0]))
            if r is None:
                raise _plane.Invalid(f"{what} is in no region of {name}")
            grouped[r].setdefault(what, []).append(row)
        self.regions: list[Region] = []
        self.parts: list[Part] = []
        self._regions: list[_Region] = []
        offset = 0
        for r, parts in enumerate(grouped):
            if not parts:
                raise _plane.Invalid(f"region {r} of {name} is empty")
            region = _Region(r, offset, 0)
            for what, rows in parts.items():
                base = min(row["dest_off"] for row in rows)
                nbytes = max(row["dest_off"] + row["len"] for row in rows) - base
                self.parts.append({"what": what, "region": r, "offset": offset, "nbytes": nbytes})
                for row in rows:
                    within = row["dest_off"] - base
                    region.copies.append((row, offset - region.offset + within, within))
                offset = _aligned(offset + nbytes)
            region.span = offset - region.offset
            self._regions.append(region)
            self.regions.append(
                {"index": r, "offset": region.offset, "nbytes": region.span, "span": region.span}
            )
        self.nbytes = offset

    def close(self) -> None:
        with self.plane.lock:
            if any(r.leases or r.held for r in self._regions):
                raise _plane.LeaseViolation(f"{self.name} has live leases")
            for region in self._regions:
                self.plane._free(self, region)
            self.plane.sets.pop(self.name, None)
        self.source.close()


class Ticket:
    """Fills here are synchronous: a ticket is done when it is given."""

    def done(self) -> bool:
        return True

    def wait(self) -> None:
        return None


class Lease:
    def __init__(self, plane: UnifiedPlane, ws: WeightSet, region: _Region) -> None:
        self.plane, self.ws, self._region = plane, ws, region
        assert region.tensor is not None  # leased while resident
        self.tensor: Tensor = region.tensor
        self.ptr = int(self.tensor.data_ptr())
        self.nbytes = region.span
        self.region = region.index
        self.ring_offset = None
        self.live = True

    def view(self) -> Tensor:
        """The region's bytes as a uint8 device tensor (not DLPack: MPS has none)."""
        return self.tensor

    def release(self, stream: int | None = None) -> None:
        with self.plane.lock:
            if self.live:
                self.live = False
                self._region.leases -= 1

    def __enter__(self) -> Lease:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class Cursor:
    """A stage's streamed blocks, read where they are used: those resident at its start are held
    in place, and each streamed one lets the oldest go once `window` are resident."""

    def __init__(
        self, plane: UnifiedPlane, ws: WeightSet, order: Sequence[int], window: int, priority: int
    ) -> None:
        self.plane, self.ws, self.order = plane, ws, list(order)
        self.window, self.priority = max(window, 1), priority
        self.home = [ws._regions[r].tensor is not None for r in self.order]
        self.held = [ws._regions[r] for r, home in zip(self.order, self.home, strict=True) if home]
        for region in self.held:
            region.held += 1
        self.streamed: deque[_Region] = deque()
        self.ring_ptr = self.ring_nbytes = self.position = self.passes = 0

    def acquire(self, index: int, stream: int = 0) -> Lease:
        region = self.ws._regions[index]
        with self.plane.lock:
            if region.tensor is None:
                while self.streamed and (
                    len(self.streamed) >= self.window
                    or self.plane.committed + region.span > self.plane.budget
                ):
                    old = self.streamed.popleft()
                    if old.tensor is not None and not (old.leases or old.held or old.pinned):
                        self.plane._evict(self.ws, old)
                self.streamed.append(region)
        lease = self.plane.acquire(self.ws, 0, index, stream, priority=self.priority)
        self.position += 1
        self.passes = self.position // max(len(self.order), 1)
        return lease

    def ring_view(self) -> Tensor:
        raise _plane.Invalid("a unified plane streams without a ring")

    def close(self) -> None:
        with self.plane.lock:
            for region in self.held:
                region.held -= 1
            self.held = []


class UnifiedPlane:
    """`tensorfs.plane.Plane` for one unified-memory device (`device`): budgets, residency
    intent, leases, cursors, stats and events, over regions read from the page cache."""

    def __init__(self, torch: ModuleType, device: Device) -> None:
        self.torch, self.device = torch, device
        self.lock = threading.RLock()
        self.devices = [(0, device.type)]
        self.sets: dict[str, WeightSet] = {}
        self.budget = self.pinned_budget = self.committed = 0
        self.serial = 0
        self.counts = {"misses": 0, "evictions": 0, "evicted_bytes": 0, "fills": 0}
        self.fill_bytes = self.fill_ns = 0
        self._events: list[Facts] = []

    # -- sources and weight sets

    def source(self, store: Store, lease: ReadLease) -> _Source:
        return _LeaseSource(store, lease)

    def files_source(self, files: Iterable[tuple[str, int, int]], header: Header) -> _Source:
        """The machine's verified object files (`(sha256, length, fd)`; the caller keeps the
        fds, the maps are this plane's) and the checkpoint's parsed header."""
        return _FileSource(files, header)

    def register(
        self,
        name: str,
        source: _Source,
        plan: ReadPlan,
        regions: Sequence[Sequence[str]],
        host_fd: int | None = None,
    ) -> WeightSet:
        if host_fd is not None:
            raise _plane.Invalid("a unified plane adopts no host layout")
        with self.lock:
            if name in self.sets:
                raise _plane.Invalid(f"{name} is registered")
            self.serial += 1
            ws = self.sets[name] = WeightSet(self, name, source, plan, regions, self.serial)
            return ws

    def device_sharing(self, device: int) -> bool:
        return False

    # -- budgets and residency

    def set_vram_budget(self, device: int, nbytes: int) -> Trim:
        with self.lock:
            self.budget = max(int(nbytes), 0)
            return self._trim(self.committed - self.budget, None)

    def set_pinned_budget(self, nbytes: int) -> Trim:
        self.pinned_budget = int(nbytes)
        return {"freed": 0, "evicted": [], "over_budget_unreleasable": 0, "blockers": []}

    def want(
        self,
        ws: WeightSet,
        tier: int | str,
        regions: Sequence[int] | None = None,
        priority: int = 0,
        pin: bool = False,
    ) -> Ticket:
        if tier == "pinned":  # the page cache is the host tier: nothing to fill
            return Ticket()
        with self.lock:
            chosen = self._chosen(ws, regions)
            whole = sum(r.span for r in chosen)
            if whole > self.budget:  # they are wanted resident at once: nothing helps
                raise self._error(_plane.BelowFloor, whole, self.budget, [], "budget")
            for region in chosen:  # what stays is not evicted to make room for the rest
                region.priority = max(region.priority, priority)
            self._room(sum(r.span for r in chosen if r.tensor is None), priority)
            for region in chosen:
                region.priority = priority
                region.pinned = region.pinned or pin
                if region.tensor is None:
                    self._fill(ws, region)
        return Ticket()

    def prioritise(
        self,
        ws: WeightSet,
        tier: int | str,
        regions: Sequence[int] | None = None,
        priority: int = 0,
        pin: bool | None = None,
    ) -> None:
        with self.lock:
            for region in self._chosen(ws, regions):
                region.priority = priority
                if pin is not None:
                    region.pinned = pin

    def drop(self, ws: WeightSet, tier: int | str, regions: Sequence[int] | None = None) -> int:
        if tier == "pinned":
            return 0
        with self.lock:
            chosen = self._chosen(ws, regions)
            busy = [f"{ws.name}#{r.index}" for r in chosen if r.leases or r.held]
            if busy:
                raise _plane.LeaseViolation(f"leased or held: {', '.join(busy[:4])}")
            freed = 0
            for region in chosen:
                region.pinned = False
                freed += self._free(ws, region)
            return freed

    def acquire(
        self, ws: WeightSet, device: int, index: int, stream: int = 0, priority: int | None = None
    ) -> Lease:
        with self.lock:
            region = ws._regions[index]
            if priority is not None:
                region.priority = priority
            if region.tensor is None:
                self.counts["misses"] += 1
                if region.span > self.budget:
                    raise self._error(_plane.BelowFloor, region.span, self.budget, [], "budget")
                self._room(region.span, region.priority)
                self._fill(ws, region)
            region.leases += 1
            return Lease(self, ws, region)

    def stream(
        self,
        ws: WeightSet,
        device: int,
        order: Sequence[int],
        window: int,
        repeat: int = 1,
        priority: int = 0,
        tag: str = "",
        ring_bytes: int | None = None,
    ) -> Cursor:
        with self.lock:
            return Cursor(self, ws, order, window, priority)

    # -- facts

    def stats(self) -> Facts:
        with self.lock:
            regions = [(ws, r) for ws in self.sets.values() for r in ws._regions]
            leased = sum(r.span for _, r in regions if r.tensor is not None and r.leases)
            held = sum(r.span for _, r in regions if r.tensor is not None and r.held)
            return {
                "host": {"budget": self.pinned_budget, "used": 0},
                "devices": [
                    {
                        "ordinal": 0,
                        "budget": self.budget,
                        "committed": self.committed,
                        "leased_bytes": leased,
                        "held_bytes": held,
                        "disk_copy_bytes": self.fill_bytes,
                        "disk_copy_ns": self.fill_ns,
                        "misses": self.counts["misses"],
                        "evictions": self.counts["evictions"],
                        "evicted_bytes": self.counts["evicted_bytes"],
                    }
                ],
                "sets": [
                    {
                        "name": ws.name,
                        "nbytes": ws.nbytes,
                        "devices": [
                            {
                                "ordinal": 0,
                                "ready_bytes": sum(
                                    r.span for r in ws._regions if r.tensor is not None
                                ),
                            }
                        ],
                    }
                    for ws in self.sets.values()
                ],
                "stages": [],
                "regions": [],
                "poisoned": None,
            }

    def events(self) -> list[Facts]:
        with self.lock:
            out, self._events = self._events, []
            return out

    def close(self) -> None:
        for ws in list(self.sets.values()):
            ws.close()

    # -- internals (under `lock`)

    def _chosen(self, ws: WeightSet, regions: Sequence[int] | None) -> list[_Region]:
        return list(ws._regions) if regions is None else [ws._regions[i] for i in regions]

    def _fill(self, ws: WeightSet, region: _Region) -> None:
        torch = self.torch
        started = time.perf_counter_ns()
        staging = torch.empty(region.span, dtype=torch.uint8)
        into = memoryview(staging.numpy())
        for row, at, within in region.copies:
            ws.source.read(row, within, into[at : at + row["len"]])
        region.tensor = staging.to(self.device)
        self.committed += region.span
        self.counts["fills"] += 1
        self.fill_bytes += region.span
        self.fill_ns += time.perf_counter_ns() - started

    def _free(self, ws: WeightSet, region: _Region) -> int:
        if region.tensor is None:
            return 0
        region.tensor = None
        self.committed -= region.span
        return region.span

    def _evict(self, ws: WeightSet, region: _Region) -> None:
        freed = self._free(ws, region)
        self.counts["evictions"] += 1
        self.counts["evicted_bytes"] += freed
        self._events.append(
            {"kind": "evicted", "ws": ws.name, "region": region.index, "device": 0}
            | {"bytes": freed, "ns": 0, "detail": ""}
        )

    def _victims(self, below: int | None) -> list[tuple[WeightSet, _Region]]:
        """Eviction order: ascending priority, later-registered sets first, then the higher
        region index; never a pinned, leased or held region, and only strictly below `below`."""
        found = [
            (ws, r)
            for ws in self.sets.values()
            for r in ws._regions
            if r.tensor is not None
            and not (r.pinned or r.leases or r.held)
            and (below is None or r.priority < below)
        ]
        return sorted(found, key=lambda pair: (pair[1].priority, -pair[0].serial, -pair[1].index))

    def _trim(self, excess: int, below: int | None) -> Trim:
        evicted: list[tuple[str, int]] = []
        freed = 0
        for ws, region in self._victims(below):
            if freed >= excess:
                break
            span = region.span
            self._evict(ws, region)
            freed += span
            evicted.append((ws.name, region.index))
        blockers = [
            f"{ws.name}#{r.index}@dev0"
            for ws in self.sets.values()
            for r in ws._regions
            if r.tensor is not None and (r.leases or r.held or r.pinned)
        ]
        return {
            "freed": freed,
            "evicted": evicted,
            "over_budget_unreleasable": max(excess - freed, 0),
            "blockers": blockers[:16],
        }

    def _room(self, need: int, below: int) -> None:
        excess = self.committed + need - self.budget
        if excess <= 0:
            return
        trim = self._trim(excess, below)
        if trim["over_budget_unreleasable"] > 0:
            available = self.budget - self.committed
            raise self._error(_plane.Shortfall, need, available, trim["blockers"], "available")

    @staticmethod
    def _error(
        kind: type[BaseException], need: int, other: int, blockers: list[str], name: str
    ) -> BaseException:
        error = kind(f"need {need} B, {name} {other} B")
        error.pool, error.need, error.blockers = "device", need, blockers  # type: ignore[attr-defined]
        setattr(error, name, other)
        return error
