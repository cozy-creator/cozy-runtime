"""Device lanes: the serialized per-device resource inside ONE worker (cr-066).

`device_lane()` always said ONE SERIALIZED PER-DEVICE STATE MACHINE and was one thread per
WORKER. This module makes the sentence literally true: a LANE owns a set of envelope-local
device ordinals, one attempt thread, one ledger row per resident executor generation, and
the placements assigned to those devices. Law 8 (one active attempt per PHYSICAL device,
#451) is a lane fact; `admission_epoch` and `admission_state` stay worker facts.

The worker's envelope is a launcher fact (`--devices 0,1`, pod inventory). It is split
into one lane per entry, in envelope order, so ordinal `i` is `entries[i]` and the seal a
lane imposes is exactly that entry. A worker given one device has one lane and is
byte-identical on the wire to the worker before lanes existed. The ENVELOPE lane is the
whole envelope as one group lane with one seat: it is the shape a job takes (cr-018's
door), never a serving placement's.

Residence ceilings are ASSIGNED by the worker, not derived in the executor (cr-025's
`authorized_device_limit_bytes`): a process ceiling is what a device measures less what its
co-tenants hold there, and loads are serialized by the device lock, so two executors on one
device cannot both be handed the same headroom. Every input is a measurement.

A GROUP lane has d > 1 ordinals (cr-068): ONE executor sealed to all d devices, one seat
and one staged, one `DeviceFacts` row per device. Lanes OVERLAP: a group over 0-3 and the
device lanes 0..3 exist together, so a device holds any number of tenants (the owner's
"allow models to share GPUs"). `gpu_scheduler` decides which ordinals a placement gets and
`bind` binds exactly those; nothing is refused or retired for occupancy.

EXECUTION is exclusive per physical device (law 8): each device has one lock and a lane
takes its devices' locks in ordinal order. RESIDENCY is shared: co-tenants of a device
hold weights side by side while they fit; when a tenant's demand does not fit, co-tenants
on its tightest device VACATE, LRU-first — the process and its construction stay, the
weights re-stage on that tenant's next attempt. No timer decides anything here; a tenant
that fits beside its neighbours is never moved, and one on the card is never a victim:
the arbiter holds that device's lock.
"""

from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from cozy_runtime.internal import accel
from cozy_runtime.internal.accel import DeviceMemory
from cozy_runtime.internal.worker import child
from cozy_runtime.internal.worker.attempts import AttemptRecord, Released
from cozy_runtime.internal.worker.ledger import DeviceFacts, Ledger
from cozy_runtime.internal.worker.plan import PlanChooser

#: The seats one serving lane advertises: ONE attempt on the device (#451/law 8) and ONE
#: staged behind it (proto-061 G). The staged attempt is accepted, hydrated and journaled
#: while the device runs the one ahead of it, and enters the device the moment that one
#: releases — so the device never waits on an offer round trip or an input download. It is
#: on the wire as QUEUED (proto-026): bound to no executor and no plan until device entry.
#: A job lane stages nothing (one bounded attempt per build) and advertises LANE_RUNNING.
#:
#: A seat is held from the offer's hold to DEVICE RELEASE (cr-079 D3) — the executor's
#: reply — and not through the ack. The post phase (encode, write, outcome, hold to ack) is
#: bounded structurally instead: while more than LANE_RELEASED released attempts on a lane
#: are unacknowledged the lane advertises nothing, so an owner that stops acking idles its
#: own device rather than growing this worker's held set.
LANE_RUNNING = 1
LANE_STAGED = 1
LANE_RELEASED = 1

#: The lane name a job's executor is sealed under: the whole envelope, one seat.
ENVELOPE_LANE = "envelope"
#: A weightless machine replica's own lane: sealed to no device, so it never waits for one.
HOST_LANE = "host-"
#: The LRU clock, worker-wide so tenants of different lanes on one device compare
_CLOCK = itertools.count(1)


class Devices:
    """A lane's devices as ONE re-entrant lock: each device's lock, taken in ordinal order,
    so lanes sharing a device serialize on it and overlapping lanes never deadlock."""

    __slots__ = ("locks",)

    def __init__(self, locks: tuple[threading.RLock, ...]) -> None:
        self.locks = locks

    def __enter__(self) -> None:
        for lock in self.locks:
            lock.acquire()

    def __exit__(self, *_: object) -> None:
        for lock in reversed(self.locks):
            lock.release()


class LaneRefusal(Exception):
    """A typed placement-to-lane refusal. Carries the reason an owner reads."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ExecutorDeath:
    """One exact child exit awaiting its lane's rebuild decision."""

    executor: child.Executor
    placement_id: str = ""
    #: a deferred replica rebuild released by its own scheduler grant
    granted: bool = False


@dataclass(slots=True)
class LaneRow:
    """One resident executor generation on a lane: its ledger and its plan chooser."""

    ledger: Ledger
    chooser: PlanChooser
    #: TRUE for a slot whose executor holds device memory (a model-bearing placement); a
    #: weightless executor never reaches the device and prices nothing on it
    model_bearing: bool = False
    #: the executor slot this row's generation lives in, so the lane can ask it to vacate
    supervision: child.ExecutorSupervision | None = None
    #: the lane clock at this tenant's latest admission or prepare: LRU victim order
    last_used: int = 0
    #: invocation spec digest -> THE LANE RESIDENCY SIGNATURE under which an attempt of
    #: that exact invocation died of a device OOM (cr-097). The same invocation under the
    #: SAME signature cannot get a different answer -- measured as four identical 38.3 s
    #: failures, each followed by a 25.8 s executor rebuild, while an IDLE co-tenant held
    #: 6.07 GiB. The arbitration reads it to make that retry a different request, or to
    #: refuse it typed instead of spending the handler again. Keyed by invocation (cr-190):
    #: one oversized request must not refuse every other request of the placement.
    oom_signatures: dict[bytes, str] = field(default_factory=dict)

    def resident_bytes(self) -> int:
        return sum(self.ledger.resident.values())

    def executor(self) -> child.Executor | None:
        """The live, unpoisoned executor holding this row's bytes, or None. A poisoned or
        dead one is not a vacate candidate: its own rebuild reclaims it, by process death,
        which frees more surely than any eviction."""
        current = self.supervision.current if self.supervision is not None else None
        if current is None or current.poisoned or not current.alive():
            return None
        return current


class DeviceLane:
    """ONE serialized resource: devices, one attempt thread, one row per executor."""

    def __init__(
        self,
        lane_id: str,
        ordinals: tuple[int, ...],
        devices: str,
        *,
        worker_pid: int,
        locks: tuple[threading.RLock, ...] = (),
    ) -> None:
        self.lane_id = lane_id
        #: envelope-local ordinals (position in the worker's `--devices` list)
        self.ordinals = ordinals
        #: what the seal imposes as `CUDA_VISIBLE_DEVICES`: the envelope's entries at
        #: `ordinals`, verbatim — an index or a UUID, whatever the launcher wrote
        self.devices = devices
        self.worker_pid = worker_pid
        #: slot key -> that executor generation's row. The primary placement's slot is its
        #: placement id; a hosted placement's is its placement id; the job lane's is "".
        self.rows: dict[str, LaneRow] = {}
        #: placement ids assigned to this lane, whether or not they are dispatchable yet
        self.placements: set[str] = set()
        #: STAGING (proto-061 G): serving offers, in arrival order, to be accepted beside
        #: the device. Never takes `device`: staging touches no executor.
        self.stage: queue.Queue[AttemptRecord | None] = queue.Queue()
        #: the DEVICE queue: staged attempts, job offers and executor deaths, in order
        self.work: queue.Queue[AttemptRecord | ExecutorDeath | None] = queue.Queue()
        #: the POST PHASE (cr-079): released attempts, one thread per lane, in release
        #: order. It never takes `device`: everything it reads left the executor at release.
        self.post: queue.Queue[Released | None] = queue.Queue()
        #: this lane's gate: has it STOPPED moving? A rebuild on another lane leaves it set.
        self.settled = threading.Event()
        #: one lock per physical device, shared with every lane over that device
        self.locks = locks or (threading.RLock(),)
        #: THE DEVICES, as a lock: held by the lane thread from an attempt's admission to its
        #: device release, and by any model-bearing prepare on this lane. Residency arbitration
        #: vacates co-tenants under it, so a tenant is never evicted while its own attempt
        #: is on the card. Re-entrant because the lane thread rebuilds (prepares) inside
        #: its own hold.
        self.device = Devices(self.locks)

    def __repr__(self) -> str:
        return f"DeviceLane({self.lane_id!r}, devices={self.devices!r})"

    @property
    def entries(self) -> tuple[str, ...]:
        """The seal's entries, one per ordinal, in ordinal order."""
        return split_envelope(self.devices)

    @property
    def group(self) -> bool:
        """K > 1: one executor over K devices (cr-068)."""
        return len(self.ordinals) > 1

    @property
    def degree(self) -> int:
        return len(self.ordinals)

    def measure(self, kind: str) -> dict[int, DeviceMemory]:
        """Every device of this lane as the driver reports it now, keyed by ordinal."""
        return {
            ordinal: accel.device_memory(entry, kind)
            for ordinal, entry in zip(self.ordinals, self.entries, strict=True)
        }

    def memory(self, kind: str) -> DeviceMemory:
        """The lane's ONE reading: the MIN over its devices, UNREADABLE if any is.

        Every rank follows one residency plan, so what the lane can hold is what
        its tightest card can hold; one card the driver cannot read makes the lane
        unreadable rather than "the others looked fine" (never authorized zero).
        """
        return fold_memory(self.measure(kind))

    def device_facts(self, measured: Mapping[int, DeviceMemory]) -> dict[int, DeviceFacts]:
        """This lane's per-device ledger rows from a measurement keyed by ordinal."""
        rows: dict[int, DeviceFacts] = {}
        for ordinal, entry in zip(self.ordinals, self.entries, strict=True):
            memory = measured.get(ordinal)
            if memory is None or memory.state != "measured":
                continue
            rows[ordinal] = DeviceFacts(ordinal, entry, memory.free_bytes, memory.total_bytes)
        return rows

    def row(self, slot: str) -> LaneRow:
        """The row for `slot`, created on first use with an unknown generation."""
        held = self.rows.get(slot)
        if held is None:
            ledger = Ledger(worker_pid=self.worker_pid)
            held = self.rows[slot] = LaneRow(ledger=ledger, chooser=PlanChooser(ledger))
        return held

    def forget(self, slot: str) -> None:
        """Drop a slot's row when its placement leaves this lane."""
        self.rows.pop(slot, None)

    def touch(self, slot: str) -> None:
        """`slot`'s tenant used the device (an admission or a prepare): it is now MRU."""
        self.row(slot).last_used = next(_CLOCK)

    def holding(self) -> list[tuple[str, LaneRow]]:
        """Every slot whose executor holds device bytes NOW: a model-bearing row with
        resident components and a live, unpoisoned executor. A row mid-load (no
        construction facts yet) holds nothing the driver has settled."""
        return [
            (key, row)
            for key, row in self.rows.items()
            if row.model_bearing and row.resident_bytes() > 0 and row.executor() is not None
        ]

    def resident_placement_ids(self) -> list[str]:
        """`DeviceLane.resident_placement_ids` (proto-026 / cr-080): the placements of this
        lane whose executor holds device bytes now, sorted — `holding()` as a SET of names.
        Membership only; the bytes stay on the row."""
        return sorted(key for key, _row in self.holding() if key in self.placements)

    def summary(self) -> dict[str, object]:
        return {
            "lane_id": self.lane_id,
            "ordinals": list(self.ordinals),
            "devices": self.devices,
            "degree": self.degree,
            "placements": sorted(self.placements),
            "rows": {slot: row.ledger.rows() for slot, row in sorted(self.rows.items())},
        }


def split_envelope(devices: str) -> tuple[str, ...]:
    """The launcher's `--devices` value as its entries, in order, empty entries dropped."""
    return tuple(entry.strip() for entry in devices.split(",") if entry.strip())


def lane_id_for(ordinals: tuple[int, ...]) -> str:
    """`lane-0` for a device lane, `lane-0+1` for a group: opaque on the wire, legible here."""
    return "lane-" + "+".join(str(ordinal) for ordinal in ordinals)


def fold_memory(measured: Mapping[int, DeviceMemory]) -> DeviceMemory:
    """K device readings as ONE: measured only if every one is, MIN free and MIN total."""
    if not measured or any(memory.state != "measured" for memory in measured.values()):
        return DeviceMemory("unreadable")
    return DeviceMemory(
        "measured",
        min(memory.free_bytes for memory in measured.values()),
        min(memory.total_bytes for memory in measured.values()),
    )


@dataclass(slots=True)
class LaneSet:
    """Every lane this worker drives: one per envelope entry, one per GROUP of entries a
    placement is bound to (overlapping them), plus the envelope and orchestration lanes."""

    entries: tuple[str, ...]
    lanes: tuple[DeviceLane, ...]
    envelope: DeviceLane
    orchestration: DeviceLane
    worker_pid: int
    by_id: dict[str, DeviceLane] = field(default_factory=dict)
    composition: dict[str, DeviceLane] = field(default_factory=dict)

    @classmethod
    def from_envelope(cls, devices: str, *, worker_pid: int) -> LaneSet:
        entries = split_envelope(devices) or ("0",)
        lanes = tuple(
            DeviceLane(lane_id_for((ordinal,)), (ordinal,), entry, worker_pid=worker_pid)
            for ordinal, entry in enumerate(entries)
        )
        envelope = DeviceLane(
            ENVELOPE_LANE, tuple(range(len(entries))), ",".join(entries), worker_pid=worker_pid
        )
        orchestration = DeviceLane("orchestration", (), "", worker_pid=worker_pid)
        by_id = {lane.lane_id: lane for lane in (*lanes, envelope, orchestration)}
        return cls(
            entries=entries,
            lanes=lanes,
            envelope=envelope,
            orchestration=orchestration,
            worker_pid=worker_pid,
            by_id=by_id,
        )

    def __iter__(self) -> Iterator[DeviceLane]:
        return iter((*self.lanes, self.envelope, self.orchestration, *self.composition.values()))

    @property
    def wide(self) -> bool:
        """More than one device in the envelope: the shape a pin decides (cr-068)."""
        return len(self.entries) > 1

    def lane_of(self, placement_id: str) -> DeviceLane | None:
        return next(
            (
                lane
                for lane in (*self.lanes, *self.composition.values())
                if placement_id in lane.placements
            ),
            None,
        )

    def group(self, ordinals: tuple[int, ...]) -> DeviceLane:
        """The lane over exactly `ordinals` (K >= 2), sharing each device's lock with its
        device lane; created on first use and gone when its last placement leaves."""
        lane = self.by_id.get(lane_id_for(ordinals))
        if lane is None:
            devices = [self.by_id[lane_id_for((ordinal,))] for ordinal in ordinals]
            lane = DeviceLane(
                lane_id_for(ordinals),
                ordinals,
                ",".join(device.devices for device in devices),
                worker_pid=self.worker_pid,
                locks=tuple(device.locks[0] for device in devices),
            )
            self.lanes = tuple(sorted((*self.lanes, lane), key=lambda other: other.ordinals))
            self.by_id[lane.lane_id] = lane
        return lane

    def release(self, placement_id: str) -> bool:
        """A placement leaves; a group or host lane it held alone goes with it. True when a
        lane left the set."""
        gone = False
        for lane in self.lanes:
            lane.placements.discard(placement_id)
            lane.forget(placement_id)
            if lane.group and not lane.placements:
                self.by_id.pop(lane.lane_id, None)
                gone = True
        self.lanes = tuple(lane for lane in self.lanes if lane.lane_id in self.by_id)
        host = self.composition.pop(HOST_LANE + placement_id, None)
        if host is not None:
            self.by_id.pop(host.lane_id, None)
        return gone or host is not None

    # ------------------------------------------------------------------ residency

    def sharing(self, lane: DeviceLane) -> list[DeviceLane]:
        """`lane` and every lane over one of its devices: whose tenants share its memory."""
        mine = set(lane.ordinals)
        return [lane, *(o for o in self.lanes if o is not lane and mine & set(o.ordinals))]

    def tenants(self, lane: DeviceLane, ordinal: int | None = None) -> list[tuple[str, LaneRow]]:
        """Model-bearing rows on `lane`'s devices (on `ordinal` alone, if given)."""
        return [
            (slot, row)
            for other in self.sharing(lane)
            if ordinal is None or ordinal in other.ordinals
            for slot, row in other.rows.items()
            if row.model_bearing
        ]

    def victims(self, lane: DeviceLane, slot: str, ordinal: int) -> list[tuple[str, LaneRow]]:
        """`slot`'s live co-tenants on `ordinal`, one of `lane`'s devices, LRU first. None
        is on the card: the caller holds `lane.device`, which any of them needs to execute.
        One holding no residents may still hold an allocator cache."""
        held = [
            (key, row)
            for key, row in self.tenants(lane, ordinal)
            if key != slot and row.executor() is not None
        ]
        return sorted(held, key=lambda item: item[1].last_used)

    def beside(self, lane: DeviceLane, slot: str) -> int:
        """The most any one of `lane`'s devices holds for tenants other than `slot`."""
        return max(
            (
                sum(row.resident_bytes() for key, row in self.tenants(lane, o) if key != slot)
                for o in lane.ordinals
            ),
            default=0,
        )

    def signature(self, lane: DeviceLane) -> str:
        """WHO HOLDS WHAT ON `lane`'s DEVICES, right now, as one comparable word (cr-097).

        The question a retry has to be able to answer: is the device in the same state it
        was in when the last attempt died? Bytes rather than names, because a co-tenant
        that vacated half its components changed the answer even though the set of tenants
        did not.
        """
        rows = sorted(
            ((slot, row) for other in self.sharing(lane) for slot, row in other.rows.items()),
            key=lambda item: item[0],
        )
        return ";".join(f"{slot}={row.resident_bytes()}" for slot, row in rows)

    # ------------------------------------------------------------------- assignment

    def bind(
        self,
        placement_id: str,
        ordinals: tuple[int, ...],
        *,
        model_bearing: bool,
        measured: Mapping[int, DeviceMemory],
    ) -> DeviceLane:
        """Bind a placement to EXACTLY `ordinals`, or refuse typed. Occupancy never refuses:
        devices are shared and residency arbitration decides who holds memory.

        A group never degrades to fewer devices: `device_pin_infeasible` (envelope),
        `device_group_unsupported` (a weightless group), `device_group_infeasible` (an
        unreadable device).
        """
        current = self.lane_of(placement_id)
        if current is not None:
            return current
        if not ordinals and not model_bearing:
            host = DeviceLane(HOST_LANE + placement_id, (), "", worker_pid=self.worker_pid)
            self.composition[host.lane_id] = self.by_id[host.lane_id] = host
            host.placements.add(placement_id)
            return host
        if (
            not ordinals
            or tuple(sorted(set(ordinals))) != tuple(ordinals)
            or ordinals[-1] >= len(self.entries)
        ):
            raise LaneRefusal(
                "device_pin_infeasible",
                f"placement {placement_id!r}: {list(ordinals)} is not sorted unique ordinals "
                f"of this worker's {len(self.entries)}-device envelope",
            )
        unreadable = [
            f"ordinal {o} ({self.entries[o]!r}) free bytes unreadable"
            for o in ordinals
            if model_bearing and measured.get(o, DeviceMemory("unreadable")).state != "measured"
        ]
        if len(ordinals) > 1 and not model_bearing:
            raise LaneRefusal(
                "device_group_unsupported",
                f"placement {placement_id!r} is weightless and granted {len(ordinals)} "
                "devices; a group shards a model's attention and there is no model",
            )
        if unreadable:
            raise LaneRefusal(
                "device_group_infeasible" if len(ordinals) > 1 else "device_pin_infeasible",
                f"placement {placement_id!r} cannot take {list(ordinals)}: "
                + "; ".join(unreadable),
            )
        lane = self.group(ordinals) if len(ordinals) > 1 else self.by_id[lane_id_for(ordinals)]
        lane.placements.add(placement_id)
        return lane

    def choose(
        self,
        candidates: tuple[int, ...],
        width: int,
        *,
        model_bearing: bool,
        declared_bytes: int,
        measured: Mapping[int, DeviceMemory],
    ) -> tuple[int, ...]:
        """Pick `width` ordinals for a placement no scheduler grant pinned exactly.

        A group takes the first `width` candidates. One device is chosen by measured fit,
        envelope order within each step: a device with no model-bearing tenant whose free
        bytes cover the declared weights; one whose free bytes cover them beside its
        tenants (cr-022 co-fit); one whose total covers them (time-sliced); any readable
        device (bounded preparation decides the working set). A weightless placement goes
        to the candidate device lane with the fewest placements.
        """
        if not candidates or not set(candidates) <= set(range(len(self.entries))):
            raise LaneRefusal(
                "device_pin_infeasible",
                f"{list(candidates)} names ordinals outside this worker's "
                f"{len(self.entries)}-device envelope",
            )
        if width > 1:
            return tuple(sorted(candidates)[:width])
        pool = [self.by_id[lane_id_for((o,))] for o in dict.fromkeys(candidates)]
        if not model_bearing:
            return min(pool, key=lambda lane: (len(lane.placements), lane.ordinals)).ordinals
        readable = [
            (lane, memory)
            for lane in pool
            if (memory := measured.get(lane.ordinals[0], DeviceMemory("unreadable"))).state
            == "measured"
        ]

        def crowd(lane: DeviceLane) -> tuple[int, tuple[int, ...]]:
            return len(self.tenants(lane)), lane.ordinals

        steps = (
            [
                lane
                for lane, m in readable
                if not self.tenants(lane) and m.free_bytes >= declared_bytes
            ],
            [lane for lane, m in readable if m.free_bytes >= declared_bytes],
            sorted((lane for lane, m in readable if m.total_bytes >= declared_bytes), key=crowd),
            sorted((lane for lane, _m in readable), key=crowd),
        )
        for step in steps:
            if step:
                return step[0].ordinals
        raise LaneRefusal(
            "device_lane_infeasible",
            f"no lane among {list(candidates)} has readable free bytes",
        )
