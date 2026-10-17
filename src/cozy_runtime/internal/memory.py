"""One owner of an executor's device memory per GPU (runtime-memory-v2.md §1-3).

Every memory signal (an op out of memory, the allocator's refusal, the plane short for a map)
asks `free_for`, which walks one ladder in order: each action runs again while it gives device
bytes back and is left when a pass gives none, so it ends without counts or timers. Only
`exhausted` raises for memory, with the ledger of every byte on the GPU.
"""

from __future__ import annotations

import gc
import os
import re
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, NoReturn

from cozy_runtime.author._errors import (
    MEMORY_EXHAUSTED,
    RuntimeFailure,
    is_device_oom,
    is_library_refusal,
)
from cozy_runtime.author._model import RELEASERS
from cozy_runtime.internal import accel, plane, spill, weight_policy

if TYPE_CHECKING:
    from cozy_runtime.internal.weights import Component, Weights

MARGIN = weight_policy.MARGIN
#: torch's out-of-memory message: CUDA's units are KiB/MiB/GiB, MPS's KB/MB/GB of 1024
_REQUESTED = re.compile(r"Tried to allocate ([0-9.]+) (B|KiB|MiB|GiB|KB|MB|GB)")
_UNITS = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30}
_UNITS |= {"KB": _UNITS["KiB"], "MB": _UNITS["MiB"], "GB": _UNITS["GiB"]}


class Rung(IntEnum):
    SLACK = 0  # torch's idle cache
    IDLE = 1  # orphaned rings, weights no stage holds, other tenants (`DeviceRoom`)
    FINER = 2  # the running stage's resident set, its components not in use
    CACHES = 3  # what is kept for speed alone: reservations (`Weights.reserve`), `RELEASERS`
    HOST = 4  # live tensors to page-locked host memory, the op's operands last
    DISK = 5  # the pinned tier toward 0, so more can lock


@dataclass(frozen=True, slots=True)
class Ledger:
    """Every byte of one GPU in one class, as the driver, torch and the plane count it."""

    total: int
    free: int
    context: int
    #: other processes on the GPU (NVML); -1 when the driver does not show this one
    foreign: int
    reserved: int
    allocated: int
    mapped: int
    leased: int
    budget: int
    shared: int
    hosted: int
    floor: int
    cap: int

    @property
    def residue(self) -> int:
        """What the driver charges that no class accounts for: never working set."""
        if self.foreign < 0:
            return 0
        held = self.foreign + self.context + self.reserved + self.mapped + self.shared
        return max(self.total - self.free - held, 0)

    def text(self) -> str:
        others = "unknown" if self.foreign < 0 else self.foreign
        return (
            f"driver {self.free} B free of {self.total}; context {self.context}; torch "
            f"{self.reserved} reserved, {self.allocated} allocated; plane {self.mapped} mapped, "
            f"{self.leased} leased, budget {self.budget}; {self.shared} from the machine; "
            f"other processes {others}; unattributed {self.residue}; {self.hosted} in host "
            f"memory; floor {self.floor}, cap {self.cap}"
        )


class MemoryExhausted(RuntimeFailure):
    """The one memory refusal: the ladder ran to its end and the op still does not fit."""

    def __init__(self, detail: str, op: str, nbytes: int, ledger: Ledger, tried: str) -> None:
        super().__init__(detail, code=MEMORY_EXHAUSTED)
        self.detail, self.op, self.nbytes = detail, op, nbytes
        self.ledger, self.tried = ledger, tried
        self.shortfall: dict[str, int | str] = {
            "resource": "vram",
            "needed_bytes": nbytes,
            "available_bytes": max(ledger.free - ledger.floor, 0),
            "request_shape": op,
        }


def requested(exc: BaseException) -> int:
    """The allocation torch could not make, from its own message (0: not stated)."""
    found = _REQUESTED.search(str(exc))
    return int(float(found.group(1)) * _UNITS[found.group(2)]) if found else 0


def collective(exc: BaseException) -> bool:
    """NCCL ran out of memory inside a collective: its peers wait in it, so no GPU runs it
    again alone; the group runs the call again (`mirror.mirror_component`)."""
    return "NCCL" in str(exc) and is_device_oom(exc)


def shortage(exc: BaseException) -> tuple[int, bool] | None:
    """(bytes short, whether the plane's own budget was) for a memory failure one op can run
    again after; None for any other. A plane refusal carries its real deficit, never a plan's
    totals (run 10562)."""
    if isinstance(exc, plane.error("Shortfall")):
        return max(int(getattr(exc, "need", 0)) - int(getattr(exc, "available", 0)), 1), True
    if isinstance(exc, plane.error("BelowFloor")):
        return max(int(getattr(exc, "need", 0)) - int(getattr(exc, "budget", 0)), 1), True
    if isinstance(exc, plane.error("BudgetExceeded")):
        asked = int(getattr(exc, "requested", 0))
        if getattr(exc, "pool", "") == "device":  # the device itself is full
            return asked, False
        return max(asked - int(getattr(exc, "available", 0)), 1), True
    if is_device_oom(exc):
        return None if collective(exc) else (requested(exc), False)
    if is_library_refusal(exc):  # a cuBLAS/cuDNN handle or workspace that found no room
        return weight_policy.LIBRARY_ROOM, False
    return None


def reap(component: Component) -> bool:
    """Close the cursors a view into their ring kept open (`close_cursor`): whether none is."""
    for cursor in list(component.orphans):
        try:
            cursor.close()
        except plane.error("Poisoned"):
            raise
        except plane.error("PlaneError"):
            continue
        component.orphans.remove(cursor)
    return not component.orphans


def _caller() -> str:
    """Where the refused op was issued: the innermost frames outside torch and this ladder."""
    frames = [
        f
        for f in traceback.extract_stack()[:-3]
        if "/torch/" not in f.filename
        and os.path.basename(f.filename) not in ("memory.py", "weights.py", "contextlib.py")
    ]
    return " < ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in frames[:-5:-1])


class Memory:
    """The ladder and the refusal for one executor's GPU (`Weights.memory`)."""

    def __init__(self, weights: Weights) -> None:
        self.weights = weights
        #: rung -> [passes that gave bytes back, bytes] this attempt (`modes["memory"]`)
        self.used: dict[Rung, list[int]] = {}
        #: the bytes the last `free_for` gave back
        self.freed = 0
        #: the attempt's refusal, for its reply
        self.refusal: dict[str, int | str] | None = None
        #: the running stage gave weights back: its next block boundary plans it again, in
        #: `plan` bytes when rung 2 set it (-1: the budget it has)
        self.shrunk = False
        self.plan = -1
        #: rung 1 asks the owner; never from inside torch's allocator
        self.asking = True
        self._collected = False

    def free(self) -> int:
        """Device bytes torch or the plane can take now: free above the floor, in the cap."""
        return self.weights.headroom(self.weights.allocation())

    def ledger(self) -> Ledger:
        weights = self.weights
        memory = weights.allocation()
        device = plane.stats(weights.plane).device(weights.index)
        foreign = -1
        rows = accel.compute_processes(dict(weights.plane.devices).get(weights.index, ""))
        if rows is not None and os.getpid() in rows:
            foreign = sum(n for pid, n in rows.items() if pid != os.getpid() and n > 0)
        return Ledger(
            total=memory["driver_total_bytes"],
            free=memory["driver_free_bytes"],
            context=max(weights.context_bytes, 0),
            foreign=foreign,
            reserved=memory["reserved_bytes"],
            allocated=memory["allocated_bytes"],
            mapped=max(device.committed, 0),
            leased=device.leased_bytes + device.held_bytes,
            budget=device.budget,
            shared=device.shared_bytes,
            hosted=sum(spill.HELD.values()),
            floor=weights.floor_free,
            cap=weights.cap,
        )

    def free_for(
        self,
        nbytes: int,
        op: str,
        operands: Sequence[object] = (),
        upto: Rung = Rung.DISK,
    ) -> bool:
        """Whether `nbytes` (and a margin) are free once the ladder ran as far as `upto`:
        rung by rung, each action again while it gives bytes back, until they are."""
        weights = self.weights
        self.freed = 0
        if weights.host_only or nbytes + max(weights.context_bytes, 0) > weights.total:
            return False  # no rung makes a GPU larger: nothing moves for it
        want, start = max(nbytes, 0) + MARGIN, self.free()
        placing, weights.placing, self._collected = weights.placing, True, False
        try:
            self._settle()  # rung 0: torch's idle cache, which counts as given back
            have = self.free()
            for rung, act in self._ladder(operands):
                if rung > upto or have >= want:
                    break
                while have < want:
                    act(want - have)
                    self._settle()
                    now = self.free()
                    if now <= have:
                        break
                    self._count(rung, now - have)
                    have = now
            return have >= want
        finally:
            weights.placing = placing
            self.freed = max(self.free() - start, 0)

    def again(
        self,
        nbytes: int,
        op: str,
        operands: Sequence[object] = (),
        seen: int = -1,
        upto: Rung = Rung.DISK,
    ) -> int:
        """The room a failed attempt runs again with: more is free than at its last try
        (`seen`), and its bytes are (the margin is the ladder's aim, not the op's need). One that
        failed with the room the ledger gave it was
        refused by more than the ledger shows (fragmentation, a library's own bytes): its ask
        doubles, walking the ladder further, until more comes free or it is more than the
        GPU has. -1: it does not run again."""
        weights, ask = self.weights, nbytes
        most = weights.total - max(weights.context_bytes, 0) - MARGIN
        while True:
            fits = self.free_for(ask, op, operands, upto)
            if (now := self.free()) > seen and (fits or now >= nbytes or ask > nbytes):
                weights.oom_retries += 1
                return now
            if not fits or ask >= most:
                return -1
            ask = min(2 * max(ask, now), most)

    def retry[T](
        self,
        attempt: Callable[[], T],
        op: str,
        operands: Sequence[object] = (),
        upto: Rung = Rung.DISK,
    ) -> T:
        """`attempt()` until it allocates. A plane short of its own budget grows it into room
        the device has (each growth asks for all of it, so it ends); any other shortage runs
        again only after more came free."""
        weights, seen = self.weights, -1
        while True:
            try:
                return attempt()
            except Exception as exc:
                short = shortage(exc)
                if short is None:
                    raise
                exc.__traceback__ = None  # the failed attempt's frames free their tensors
            nbytes, own = short
            nbytes = nbytes or MARGIN
            if own:
                unused = max(weights.applied - weights._committed(), 0) if weights.cap < 0 else 0
                if not self.free_for(nbytes + unused, op, operands, upto):
                    self.exhausted(nbytes, op)
                weights._apply(weights.applied + nbytes)
                weights.oom_retries += 1
            elif (seen := self.again(nbytes, op, operands, seen, upto)) < 0:
                self.exhausted(nbytes, op)

    def exhausted(self, nbytes: int, op: str, why: str = "") -> NoReturn:
        """Raise the one memory refusal: the ledger, the op and what each rung gave."""
        ledger = self.ledger()
        slack = ledger.total - ledger.context - ledger.floor - ledger.leased - nbytes
        verdict = (
            f"it should have fit by {slack} B beside the context and its leased weights"
            if slack >= 0
            else f"{-slack} B more than the GPU has beside the context and its leased weights"
        )
        tried = self.summary() or "no rung gave bytes back"
        detail = (
            f"{op} needs {nbytes} B of device memory at once"
            f"{f' ({why})' if why else ''}; {verdict}; ladder: {tried}; {ledger.text()}; "
            f"issued from {_caller()}"
        )
        failure = MemoryExhausted(detail, op, nbytes, ledger, tried)
        self.refusal = failure.shortfall
        raise failure

    def ask(self, nbytes: int) -> None:
        """The owner's idle tenants on this GPU give `nbytes` (`DeviceRoom`): the one place a
        process asks, and never for nothing or less (runs 4267, 10078)."""
        if nbytes > 0 and self.asking and self.weights.room is not None:
            self.weights.room(nbytes)

    def summary(self) -> str:
        return ", ".join(f"{r.name.lower()} {n}x {b} B" for r, (n, b) in sorted(self.used.items()))

    def _ladder(self, operands: Sequence[object]) -> list[tuple[Rung, Callable[[int], object]]]:
        weights = self.weights
        return [
            (Rung.IDLE, self._collect),
            (Rung.IDLE, lambda short: weights.shrink(short, keep_holds=True)),
            (Rung.IDLE, lambda short: weights.page_out(None, short)),
            (Rung.IDLE, lambda short: self.ask(self.free() + short)),  # all it wants free
            (Rung.FINER, self._finer),
            (Rung.FINER, lambda _: weights.spare(operands)),
            (Rung.CACHES, lambda short: weights.shed(self.free() + short - MARGIN)),
            (Rung.CACHES, lambda _: [release() for release in RELEASERS]),
            (Rung.HOST, lambda short: weights.spill(short, operands)),
            (Rung.DISK, lambda short: weights.unpin(short, operands)),
        ]

    def _settle(self) -> None:
        """What the plane and torch just gave back reaches the driver, and torch's limit
        follows it."""
        weights = self.weights
        accel.release_cached(weights.torch, weights.kind)
        weights.drain()
        weights.limit_torch()

    def collect(self) -> bool:
        """Rings whose cursor could not close (a failed block's frames hold a view in a
        reference cycle, run 10562) and bytes no class accounts for: garbage is collected and
        the rings whose last view it held close. Whether one did."""
        components = list(self.weights.components.values())
        orphans = sum(len(c.orphans) for c in components)
        if not orphans and self.ledger().residue <= MARGIN:
            return False
        gc.collect()
        return sum(len(c.orphans) for c in components if not reap(c)) < orphans

    def _collect(self, _: int) -> None:
        if not self._collected:  # once a pass
            self._collected = True
            self.collect()

    def _finer(self, short: int) -> None:
        """The running stage's regions no block leases now leave, and its next block boundary
        plans it again in `short` less than it holds (`WeightResidency._replan`): a window of
        one, then its components' finest grain, down to their floor. Nothing here moves what
        the running block leases, so the failing op goes on to the next rung."""
        weights = self.weights
        freed = weights.shrink(short)
        if weights.open:
            held = min(weights.applied, weights._committed())
            held = min(held, self.plan) if self.plan >= 0 else held
            self.plan = max(held - short, weights.floor())
        self.shrunk = self.shrunk or bool(freed or weights.open)

    def _count(self, rung: Rung, freed: int) -> None:
        modes = self.weights.modes
        if "memory" not in modes:  # a new attempt cleared its modes
            self.used.clear()
        row = self.used.setdefault(rung, [0, 0])
        row[0] += 1
        row[1] += freed
        modes["memory"] = self.summary()
