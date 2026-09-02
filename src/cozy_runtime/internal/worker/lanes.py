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

Residence ceilings are ASSIGNED here, not derived in the executor (cr-025's
`authorized_device_limit_bytes`): the worker measures the lane's free device bytes and
subtracts what other executors on the lane were authorized but have not yet settled, so
two executors preparing on one device cannot both be handed the same headroom. Every
input is a measurement; nothing here is configured.

Two model-bearing placements on ONE lane share it by RESIDENCY ARBITRATION, never by
concurrency (cr-022, the owner's "one gets to use it first, then the next gets to use it
for a different request"). Both executors exist, sealed to the lane; at most one holds
its weights on the device at a time unless the measured card holds both. The trigger is
an attempt (or a prepare) for a tenant whose demand does not fit what the driver reports
free: the other tenants VACATE, LRU-first, until it does — eviction FREES (weights are
immutable, nothing is copied out) and the vacated tenant re-stages on its own next
attempt through the ordinary demand-pull admission. No timer and no idleness clock
decides anything here; a tenant that fits beside its neighbours is never moved.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from cozy_runtime.internal.accel import DeviceMemory
from cozy_runtime.internal.worker import child
from cozy_runtime.internal.worker.attempts import AttemptRecord
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import PlanChooser

#: The seats one lane advertises. ONE, by #451/law 8: a lane is a physical device (or one
#: group of devices under one attempt), and a second held attempt on it could only wait in a
#: state the wire cannot spell. This is the structural fact `ADMISSION_WINDOW` used to hold
#: for the whole worker; the worker-level count is now the SUM over its lanes.
LANE_SEATS = 1

#: The lane name a job's executor is sealed under: the whole envelope, one seat.
ENVELOPE_LANE = "envelope"


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


@dataclass(slots=True)
class LaneRow:
    """One resident executor generation on a lane: its ledger and its plan chooser."""

    ledger: Ledger
    chooser: PlanChooser
    #: the ceiling this generation was ASSIGNED for its prepare and has not yet settled
    #: under a construction reply; -1 is nothing outstanding
    outstanding: int = -1
    #: TRUE for a slot whose executor holds device memory (a model-bearing placement); a
    #: weightless executor never reaches the device and prices nothing on it
    model_bearing: bool = False
    #: the executor slot this row's generation lives in, so the lane can ask it to vacate
    supervision: child.ExecutorSupervision | None = None
    #: the lane clock at this tenant's latest admission or prepare: LRU victim order
    last_used: int = 0

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
        self, lane_id: str, ordinals: tuple[int, ...], devices: str, *, worker_pid: int
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
        self.work: queue.Queue[AttemptRecord | ExecutorDeath | None] = queue.Queue()
        #: this lane's gate: has it STOPPED moving? A rebuild on another lane leaves it set.
        self.settled = threading.Event()
        #: THE DEVICE, as a lock: held by the lane thread from an attempt's admission to its
        #: terminal, and by any model-bearing prepare on this lane. Residency arbitration
        #: vacates co-tenants under it, so a tenant is never evicted while its own attempt
        #: is on the card. Re-entrant because the lane thread rebuilds (prepares) inside
        #: its own hold.
        self.device = threading.RLock()
        #: the LRU clock: advanced by `touch`, never by time
        self.clock = 0

    def __repr__(self) -> str:
        return f"DeviceLane({self.lane_id!r}, devices={self.devices!r})"

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

    def model_bearing_slots(self) -> list[str]:
        return sorted(slot for slot, row in self.rows.items() if row.model_bearing)

    def touch(self, slot: str) -> None:
        """`slot`'s tenant used the device (an admission or a prepare): it is now MRU."""
        self.clock += 1
        self.row(slot).last_used = self.clock

    def victims(self, slot: str) -> list[tuple[str, LaneRow]]:
        """The co-tenants of `slot` that hold device bytes, LRU first (the arbitration's
        eviction order, design §4.6): model-bearing rows with resident components and a
        live, unpoisoned executor to ask. A row mid-prepare (outstanding authorization,
        no construction facts yet) holds nothing the driver has settled and is skipped."""
        held = [
            (key, row)
            for key, row in self.rows.items()
            if key != slot
            and row.model_bearing
            and row.resident_bytes() > 0
            and row.executor() is not None
        ]
        return sorted(held, key=lambda item: item[1].last_used)

    # ------------------------------------------------------------- assigned ceilings

    def authorize(self, slot: str, measured_free: int) -> int:
        """ASSIGN `slot`'s residence ceiling from the lane's MEASURED free bytes.

        `measured_free` is the driver's view of the lane's device right now, which already
        counts every executor whose fill has settled. What it cannot see is a ceiling this
        lane handed out that has not settled yet — an executor mid-prepare — so those are
        subtracted, and the same headroom can never be authorized twice. The executor still
        measures its own allocatable and takes the smaller of the two (cr-025: predictions
        authorize, observations verify).
        """
        outstanding = sum(
            row.outstanding for key, row in self.rows.items() if key != slot and row.outstanding > 0
        )
        granted = max(int(measured_free) - outstanding, 0)
        row = self.row(slot)
        row.outstanding = granted
        row.ledger.authorized_device_limit_bytes = granted
        return granted

    def settle(self, slot: str) -> None:
        """The prepare under this slot's authorization replied: the driver sees it now."""
        row = self.rows.get(slot)
        if row is not None:
            row.outstanding = -1

    def outstanding_bytes(self) -> int:
        return sum(row.outstanding for row in self.rows.values() if row.outstanding > 0)

    def summary(self) -> dict[str, object]:
        return {
            "lane_id": self.lane_id,
            "ordinals": list(self.ordinals),
            "devices": self.devices,
            "placements": sorted(self.placements),
            "rows": {slot: row.ledger.rows() for slot, row in sorted(self.rows.items())},
        }


def split_envelope(devices: str) -> tuple[str, ...]:
    """The launcher's `--devices` value as its entries, in order, empty entries dropped."""
    return tuple(entry.strip() for entry in devices.split(",") if entry.strip())


@dataclass(slots=True)
class LaneSet:
    """Every lane this worker drives: one per envelope entry, plus the envelope lane."""

    lanes: tuple[DeviceLane, ...]
    envelope: DeviceLane
    by_id: dict[str, DeviceLane] = field(default_factory=dict)

    @classmethod
    def from_envelope(cls, devices: str, *, worker_pid: int) -> LaneSet:
        entries = split_envelope(devices) or ("0",)
        lanes = tuple(
            DeviceLane(f"lane-{ordinal}", (ordinal,), entry, worker_pid=worker_pid)
            for ordinal, entry in enumerate(entries)
        )
        envelope = DeviceLane(
            ENVELOPE_LANE, tuple(range(len(entries))), ",".join(entries), worker_pid=worker_pid
        )
        by_id = {lane.lane_id: lane for lane in (*lanes, envelope)}
        return cls(lanes=lanes, envelope=envelope, by_id=by_id)

    def __iter__(self) -> Iterator[DeviceLane]:
        return iter((*self.lanes, self.envelope))

    def __len__(self) -> int:
        return len(self.lanes)

    def lane_of(self, placement_id: str) -> DeviceLane | None:
        return next((lane for lane in self.lanes if placement_id in lane.placements), None)

    def release(self, placement_id: str) -> None:
        for lane in self.lanes:
            lane.placements.discard(placement_id)
            lane.forget(placement_id)

    def assign(
        self,
        placement_id: str,
        *,
        model_bearing: bool,
        declared_bytes: int,
        measured: Mapping[str, DeviceMemory],
    ) -> DeviceLane:
        """Choose the lane for a placement by MEASURED fit, or refuse typed.

        The worker never orders attempts (#475); it does own GPU allocation (#659), and this
        is the whole of it. A model-bearing placement is placed in this order, envelope
        order within each step, and every number is the driver's:

        1. a lane holding no other model-bearing placement whose measured free bytes, less
           the ceilings the lane has outstanding, cover the declared weight bytes — the
           placement has the device to itself (cr-066);
        2. a lane whose measured headroom still covers the declared bytes BESIDE its
           tenants — co-residency, no swap will ever be needed unless the measurements
           move (cr-022's co-fit arm);
        3. a lane whose measured TOTAL covers the declared bytes — time-sliced sharing
           (cr-022's swap arm): the placement fits the card alone, and the lane's residency
           arbitration gives it the card when its attempts need it. Fewest model-bearing
           tenants first, so sharing spreads before it stacks.

        Nothing fitting anywhere is `device_lane_infeasible`, and moves no other
        placement. A lane absent from `measured` is UNREADABLE and can be shown to fit
        nothing, so it is never chosen for a model-bearing placement. A weightless
        placement holds no device memory and goes to the lane with the fewest placements,
        so its attempts do not all queue behind one seat.
        """
        current = self.lane_of(placement_id)
        if current is not None:
            return current
        if not model_bearing:
            chosen = min(self.lanes, key=lambda lane: (len(lane.placements), lane.ordinals))
            chosen.placements.add(placement_id)
            return chosen
        shortfalls: list[str] = []
        readable: list[tuple[DeviceLane, DeviceMemory]] = []
        for lane in self.lanes:
            memory = measured.get(lane.lane_id)
            if memory is None or memory.state != "measured":
                shortfalls.append(f"{lane.lane_id} free bytes unreadable")
                continue
            readable.append((lane, memory))

        def headroom(lane: DeviceLane, memory: DeviceMemory) -> int:
            return memory.free_bytes - lane.outstanding_bytes()

        alone = [
            lane
            for lane, memory in readable
            if not lane.model_bearing_slots() and headroom(lane, memory) >= declared_bytes
        ]
        beside = [lane for lane, memory in readable if headroom(lane, memory) >= declared_bytes]
        shared = sorted(
            (lane for lane, memory in readable if memory.total_bytes >= declared_bytes),
            key=lambda lane: (len(lane.model_bearing_slots()), lane.ordinals),
        )
        for candidates in (alone, beside, shared):
            if candidates:
                chosen = candidates[0]
                chosen.placements.add(placement_id)
                return chosen
        for lane, memory in readable:
            shortfalls.append(
                f"{lane.lane_id} has {headroom(lane, memory)} B measured headroom and "
                f"{memory.total_bytes} B total for {declared_bytes} B of declared weights"
            )
        raise LaneRefusal(
            "device_lane_infeasible",
            f"no lane fits placement {placement_id!r}: " + "; ".join(shortfalls),
        )
