"""The machine's device memory: who holds what on every device, and who gives room to whom.

ONE manager per Worker owns every device byte across every executor, package and rank. A
TENANT is one placement's executor over its lane's devices (a rank group is one tenant over
K devices); its row lives on its lane. THE DEVICE LOCK IS THE RESERVATION: a tenant holding
its devices' locks may use every byte no other tenant holds, and every IDLE tenant on those
devices is evictable for it, least recently used first. An idle tenant cannot be mid-call or
mid-load, because both need a lock the requester holds, so the executing tenant is never a
victim.

Every number that decides is the DRIVER's: NVML free bytes per device, read fresh under the
lock. Executor replies never size a demand, so an executor of any version cannot under-declare.
While a tenant holds its devices, only it moves their bytes, so the driver's deltas are its:

- `held[d]`: what the tenant holds on `d` beyond its process's context, moved by every load,
  restore and call it makes, and zero once it is evicted;
- `peak[shape][d]`: the most it has held on `d` during a successful call of that shape (an
  entrypoint and its request's shape cell), sampled while the call runs.

A call needs `peak - held` more free bytes on each of its devices. A shape never measured takes
the peak of a measured shape at least as large on every axis (memory does not shrink as a shape
grows); with none, or after a failure on the device, it needs the whole device: every idle
tenant there goes, and the call measures what it really takes.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

from cozy_runtime.internal import accel
from cozy_runtime.internal.accel import DeviceMemory
from cozy_runtime.internal.execution_evidence import gpu_name, gpu_names
from cozy_runtime.internal.executor_commands import Vacate
from cozy_runtime.internal.worker import child
from cozy_runtime.internal.worker.child import ExecutorGone
from cozy_runtime.internal.worker.lanes import DeviceLane, LaneRow, LaneSet

#: How often the driver is read while a call runs, to see its peak. A sampling cadence, not a
#: deadline: nothing waits on it. NVML answers in ~0.02 ms per device.
SAMPLE_SECONDS = 0.01

#: Bytes per ordinal a tenant needs free, or None: the whole device (never measured).
Need = dict[int, int] | None
#: An entrypoint and its request's shape cell (`plan.shape_cell`): what a peak is banked by.
Shape = tuple[str, str]


@dataclass(slots=True)
class Watch:
    """One transition's device picture: free bytes before, the lowest seen, and after."""

    before: dict[int, int]
    low: dict[int, int]
    after: dict[int, int] = field(default_factory=dict)


def axes(cell: str) -> dict[str, int] | None:
    """A shape cell's axes (`plan.shape_cell`), or None when one is not a count."""
    try:
        return {
            axis: int(value)
            for axis, value in (part.split("=", 1) for part in cell.split(",") if part != "-")
        }
    except ValueError:
        return None


def bound(shape: Shape, peaks: Mapping[Shape, dict[int, int]]) -> dict[int, int] | None:
    """The least peak among measured shapes of the same entrypoint at least as large as
    `shape` on every axis."""
    entrypoint, cell = shape
    mine = axes(cell) if cell else None
    if mine is None:
        return None
    covering = [
        peak
        for (other_entrypoint, other), peak in peaks.items()
        if other_entrypoint == entrypoint
        and (theirs := axes(other)) is not None
        and theirs.keys() == mine.keys()
        and all(theirs[axis] >= value for axis, value in mine.items())
    ]
    if not covering:
        return None
    return {o: min(peak.get(o, 0) for peak in covering) for o in covering[0]}


class MemoryManager:
    """Admission, eviction and reclaim over every tenant of every device."""

    def __init__(
        self,
        lanes: LaneSet,
        note: Callable[[str, str], None],
        end: Callable[[str, child.Executor, str], bool],
    ) -> None:
        self.lanes = lanes
        self.note = note
        #: kill and reap one tenant's executor now (True once proved gone); it rebuilds later
        self.end = end
        self.kind = accel.host_backend_family()

    def tenants(self, ordinals: set[int]) -> list[tuple[str, DeviceLane, LaneRow]]:
        """Every model-bearing tenant with a device among `ordinals`, each once."""
        found: dict[str, tuple[str, DeviceLane, LaneRow]] = {}
        for lane in self.lanes.lanes:
            if ordinals & set(lane.ordinals):
                for slot, row in lane.rows.items():
                    if row.model_bearing:
                        found.setdefault(slot, (slot, lane, row))
        return list(found.values())

    # ------------------------------------------------------------------ admission

    @contextmanager
    def starting(self, lane: DeviceLane, slot: str) -> Iterator[None]:
        """Admit a new device context under the same physical locks as a call.

        Startup has no measured bound yet. Use the existing unknown-need policy:
        every idle managed tenant on these devices vacates before CUDA or the
        rank group initializes. Active work keeps its lock; external allocations
        remain external, and a genuine startup refusal is never retried here.
        """
        with lane.device:
            self.make_room(lane, slot, None, "executor startup")
            yield

    def need(self, lane: DeviceLane, slot: str, shape: Shape) -> Need:
        """What one call of `shape` needs free on each device, beyond what `slot` holds."""
        row = lane.row(slot)
        peak = row.peak.get(shape)
        if peak is None:
            peak = bound(shape, row.peak)
        if peak is None:
            return None
        return {o: max(peak.get(o, 0) - row.held.get(o, 0), 0) for o in lane.ordinals}

    def admit(self, lane: DeviceLane, slot: str, shape: Shape, why: str) -> bool:
        """Make room for one call of `slot` on EVERY device of its lane, under the lane's
        devices (the caller holds them). True when the measured need fits everywhere, or
        when it was never measured and every idle tenant has gone."""
        lane.touch(slot)
        need = self.need(lane, slot, shape)
        evicted: list[str] = []
        measured = self.make_room(lane, slot, need, why, evicted)
        fits = need is None or all(
            measured[o].state == "measured" and measured[o].free_bytes >= n for o, n in need.items()
        )
        if evicted or need is None:
            self.note(
                "residency",
                f"{lane.lane_id}: admitted {slot!r} for {why} ({shape[0]} {shape[1] or '-'}): "
                + (
                    "never measured, so it has the whole of its devices"
                    if need is None
                    else "needs " + ", ".join(f"{need[o]} B on device {o}" for o in sorted(need))
                )
                + f"; evicted {evicted or 'nobody'}; free "
                + ", ".join(f"{measured[o].free_bytes} B on device {o}" for o in sorted(measured))
                + ("" if fits else "; it runs staged in what is left"),
            )
        return fits

    def make_room(
        self, lane: DeviceLane, slot: str, need: Need, why: str, evicted: list[str] | None = None
    ) -> dict[int, DeviceMemory]:
        """Evict idle tenants from `lane`'s short devices until `need` fits on every one:
        dead tenants first (killed and reaped), then weights, least recently used first.
        Returns the driver's picture once it fits or nothing is left to evict."""
        tried: set[str] = set()
        while True:
            measured = lane.measure(self.kind)
            if any(memory.state != "measured" for memory in measured.values()):
                return measured
            short = {o for o in lane.ordinals if need is None or measured[o].free_bytes < need[o]}
            if not short:
                return measured
            victim = self._victim(slot, short, tried)
            if victim is None:
                if need is not None:
                    self.note(
                        "residency",
                        f"{lane.lane_id}: no idle tenant is left to evict for {slot!r} on {why}: "
                        + ", ".join(
                            f"device {o} needs {need[o]} B free, has {measured[o].free_bytes} "
                            f"of {measured[o].total_bytes}"
                            for o in sorted(short)
                        ),
                    )
                return measured
            other, other_lane, row, executor = victim
            tried.add(other)
            if evicted is not None:
                evicted.append(other)
            if executor.poisoned or not executor.alive():
                why_dead = executor.poisoned or "exited"
                if self.end(other, executor, f"reclaimed for {slot!r} on {why}: {why_dead}"):
                    row.held.clear()
                continue
            self._evict(other, other_lane, row, executor, slot, why, measured, short)

    def _victim(
        self, slot: str, short: set[int], tried: set[str]
    ) -> tuple[str, DeviceLane, LaneRow, child.Executor] | None:
        """The next tenant to give room on `short`: a dead one, else the least recently used
        idle one holding bytes there. Never `slot`, never twice."""
        dead = []
        idle = []
        for other, lane, row in self.tenants(short):
            executor = row.supervision.current if row.supervision is not None else None
            if other == slot or other in tried or executor is None:
                continue
            if executor.poisoned or not executor.alive():
                dead.append((other, lane, row, executor))
            elif any(row.held.get(o, 0) for o in short) or (not row.held and row.resident_bytes()):
                idle.append((other, lane, row, executor))
        for tier in (dead, idle):
            if tier:
                return min(tier, key=lambda item: item[2].last_used)
        return None

    def _evict(
        self,
        victim: str,
        lane: DeviceLane,
        row: LaneRow,
        executor: child.Executor,
        slot: str,
        why: str,
        before: Mapping[int, DeviceMemory],
        short: set[int],
    ) -> None:
        """Vacate one idle tenant: the process stays warm and its weights leave the device,
        only on `short` cards when it is a group that can say which ranks. An executor that
        refuses or dies is killed and reaped instead: death frees all."""
        on = [o for o in lane.ordinals if o in short]
        ranks = (
            tuple(lane.ordinals.index(o) for o in on)
            if lane.group and "vacate_ranks" in executor.hello.get("memory", ())
            else ()
        )
        try:
            with executor.watched("vacate"):
                reply = executor.call(Vacate(ranks=ranks), timeout=None)
        except ExecutorGone as exc:
            reply = {"ok": False, "code": "executor_gone", "detail": str(exc)}
        if not reply.get("ok"):
            self.note(
                "residency",
                f"{victim!r} could not vacate ({reply.get('code')}: "
                f"{str(reply.get('detail', ''))[:200]}); ending it instead",
            )
            if self.end(victim, executor, f"evicted for {slot!r} on {why}"):
                row.held.clear()
            return
        current = row.supervision is not None and row.supervision.current is executor
        if current and (not ranks or 0 in ranks):
            row.ledger.observe_vacate(reply, why)
        emptied = on if ranks else list(lane.ordinals)
        for o in emptied:
            row.held.pop(o, None)
        row.emptied.update(emptied)
        after = lane.measure(self.kind)
        self.note(
            "residency",
            f"{lane.lane_id}: {victim!r} (epoch {executor.epoch} pid {executor.pid}) vacated "
            + (f"{gpu_names([lane.entries[r] for r in ranks])} " if ranks else "")
            + f"for {slot!r} on {why}: "
            + ", ".join(
                f"{gpu_name(lane.entries, i)} "
                f"{before[o].free_bytes} -> {after[o].free_bytes} B free"
                for i, o in enumerate(lane.ordinals)
                if o in before and after[o].state == "measured"
            ),
        )

    # ------------------------------------------------------------------ measurement

    @contextmanager
    def watch(self, lane: DeviceLane, *, sample: bool = True) -> Iterator[Watch]:
        """`lane`'s devices before and after the body; sampled while it runs, the lowest
        free bytes are its peak."""
        before = {
            o: m.free_bytes for o, m in lane.measure(self.kind).items() if m.state == "measured"
        }
        seen = Watch(before=before, low=dict(before))
        stop = threading.Event()

        def sampling() -> None:
            while not stop.wait(SAMPLE_SECONDS):
                for o, memory in lane.measure(self.kind).items():
                    if o in seen.low and memory.state == "measured":
                        seen.low[o] = min(seen.low[o], memory.free_bytes)

        sampler = threading.Thread(target=sampling, name=f"memory:{lane.lane_id}", daemon=True)
        if sample and before:
            sampler.start()
        try:
            yield seen
        finally:
            stop.set()
            if sampler.is_alive():
                sampler.join()
            seen.after = {
                o: m.free_bytes for o, m in lane.measure(self.kind).items() if m.state == "measured"
            }
            for o, free in seen.after.items():
                if o in seen.low:
                    seen.low[o] = min(seen.low[o], free)

    def moved(self, lane: DeviceLane, slot: str, seen: Watch) -> None:
        """A load, restore or unload `slot` made under its lock: bank what it took."""
        row = lane.row(slot)
        for o, free in seen.before.items():
            if o in seen.after:
                row.held[o] = max(row.held.get(o, 0) + free - seen.after[o], 0)

    def settle(
        self, lane: DeviceLane, slot: str, shape: Shape, seen: Watch, *, ok: bool, forget: bool
    ) -> None:
        """Bank one call. A success raises its shape's peak; a failure on the device forgets
        it, so the next call of that shape has the whole device again."""
        row = lane.row(slot)
        if ok:
            peak = row.peak.setdefault(shape, {})
            for o, free in seen.before.items():
                peak[o] = max(peak.get(o, 0), row.held.get(o, 0) + free - seen.low.get(o, free))
        elif forget:
            row.peak.pop(shape, None)
        if ok:
            row.emptied.clear()
        self.moved(lane, slot, seen)
