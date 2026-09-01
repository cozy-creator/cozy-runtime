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
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field

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
        measured_free: dict[str, int],
    ) -> DeviceLane:
        """Choose the lane for a placement by MEASURED fit, or refuse typed.

        The worker never orders attempts (#475); it does own GPU allocation (#659), and this
        is the whole of it. A model-bearing placement takes the first lane (envelope order)
        that holds no other model-bearing placement and whose measured free bytes, less the
        ceilings the lane has outstanding, cover the placement's declared weight bytes. One
        model-bearing placement per lane is the shape this issue builds; same-lane co-fit
        over these rows is cr-022's arithmetic and is refused here, not guessed at. A
        weightless placement holds no device memory and goes to the lane with the fewest
        placements, so its attempts do not all queue behind one seat.

        `measured_free` maps lane id to the lane's measured free bytes, and a lane absent
        from it is UNREADABLE: it cannot be shown to fit, so it is never chosen for a
        model-bearing placement.
        """
        current = self.lane_of(placement_id)
        if current is not None:
            return current
        if not model_bearing:
            chosen = min(self.lanes, key=lambda lane: (len(lane.placements), lane.ordinals))
            chosen.placements.add(placement_id)
            return chosen
        shortfalls: list[str] = []
        for lane in self.lanes:
            occupied = lane.model_bearing_slots()
            if occupied:
                shortfalls.append(f"{lane.lane_id} holds model-bearing {occupied}")
                continue
            free = measured_free.get(lane.lane_id)
            if free is None:
                shortfalls.append(f"{lane.lane_id} free bytes unreadable")
                continue
            headroom = free - lane.outstanding_bytes()
            if headroom < declared_bytes:
                shortfalls.append(
                    f"{lane.lane_id} has {headroom} B measured headroom for {declared_bytes} B "
                    "of declared weights"
                )
                continue
            lane.placements.add(placement_id)
            return lane
        raise LaneRefusal(
            "device_lane_infeasible",
            f"no lane fits placement {placement_id!r}: " + "; ".join(shortfalls),
        )
