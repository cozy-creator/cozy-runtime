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
`authorized_device_limit_bytes`): a process ceiling is the lane's measured capacity less
what its co-tenants hold, and loads on a lane are serialized by its device lock, so two
executors on one device cannot both be handed the same headroom. Every input is a
measurement; nothing here is configured.

Two model-bearing placements on ONE lane share it by RESIDENCY ARBITRATION, never by
concurrency (cr-022, the owner's "one gets to use it first, then the next gets to use it
for a different request"). Both executors exist, sealed to the lane; at most one holds
its weights on the device at a time unless the measured card holds both. The trigger is
an attempt (or a prepare) for a tenant whose demand does not fit what the driver reports
free: the other tenants VACATE, LRU-first, until it does — eviction FREES (weights are
immutable, nothing is copied out) and the vacated tenant re-stages on its own next
attempt through the ordinary demand-pull admission. No timer and no idleness clock
decides anything here; a tenant that fits beside its neighbours is never moved.

A GROUP lane is a lane whose `ordinals` has d > 1 entries (cr-068): ONE executor sealed to
all d devices, one device seat and one staged, one ledger with one `DeviceFacts` row per
device. The RecordOwner PINS a model-bearing placement to K ordinals; the worker runs it
at the largest declared sequence-parallel degree d <= K (or on one device when none fits),
FUSES d empty device
lanes into one, leaves the other K-d as device lanes and records a note saying so. The lane
stays a group while its placement holds it and SPLITS back when the placement leaves.
`revision` moves on every fuse and split, which is the worker's cue to bump
`admission_epoch`: the seat count changed meaning.
"""

from __future__ import annotations

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


def effective_degree(width: int, degrees: tuple[int, ...]) -> int:
    """The largest declared degree that `width` devices can form, or 1."""
    return max((degree for degree in degrees if 1 < degree <= width), default=1)


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
    #: TRUE for a slot whose executor holds device memory (a model-bearing placement); a
    #: weightless executor never reaches the device and prices nothing on it
    model_bearing: bool = False
    #: the executor slot this row's generation lives in, so the lane can ask it to vacate
    supervision: child.ExecutorSupervision | None = None
    #: the lane clock at this tenant's latest admission or prepare: LRU victim order
    last_used: int = 0
    #: THE LANE RESIDENCY SIGNATURE under which this tenant's last attempt died of a device
    #: OOM, or "" when it did not (cr-097). A retry under the SAME signature is the same
    #: request asked twice and cannot get a different answer -- measured as four identical
    #: 38.3 s failures, each followed by a 25.8 s executor rebuild, while an IDLE co-tenant
    #: held 6.07 GiB. The arbitration reads it to make the retry a different request, or to
    #: refuse it typed instead of spending the handler again.
    oom_signature: str = ""

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
        #: THE DEVICE, as a lock: held by the lane thread from an attempt's admission to its
        #: device release, and by any model-bearing prepare on this lane. Residency arbitration
        #: vacates co-tenants under it, so a tenant is never evicted while its own attempt
        #: is on the card. Re-entrant because the lane thread rebuilds (prepares) inside
        #: its own hold.
        self.device = threading.RLock()
        #: the LRU clock: advanced by `touch`, never by time
        self.clock = 0

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

    def model_bearing_slots(self) -> list[str]:
        return sorted(slot for slot, row in self.rows.items() if row.model_bearing)

    def touch(self, slot: str) -> None:
        """`slot`'s tenant used the device (an admission or a prepare): it is now MRU."""
        self.clock += 1
        self.row(slot).last_used = self.clock

    def holding(self) -> list[tuple[str, LaneRow]]:
        """Every slot whose executor holds device bytes NOW: a model-bearing row with
        resident components and a live, unpoisoned executor. A row mid-load (no
        construction facts yet) holds nothing the driver has settled."""
        return [
            (key, row)
            for key, row in self.rows.items()
            if row.model_bearing and row.resident_bytes() > 0 and row.executor() is not None
        ]

    def victims(self, slot: str) -> list[tuple[str, LaneRow]]:
        """The co-tenants of `slot` that hold device bytes, LRU first (the arbitration's
        eviction order, design §4.6)."""
        held = [(key, row) for key, row in self.holding() if key != slot]
        return sorted(held, key=lambda item: item[1].last_used)

    def residency_signature(self) -> str:
        """WHO HOLDS WHAT ON THIS LANE, right now, as one comparable word (cr-097).

        The question a retry has to be able to answer: is the device in the same state it
        was in when the last attempt died? Bytes rather than names, because a co-tenant
        that vacated half its components changed the answer even though the set of tenants
        did not.
        """
        return ";".join(f"{slot}={row.resident_bytes()}" for slot, row in sorted(self.rows.items()))

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
    """Every lane this worker drives: one per envelope entry (or one per GROUP of entries
    while a placement holds them), plus the envelope lane."""

    entries: tuple[str, ...]
    lanes: tuple[DeviceLane, ...]
    envelope: DeviceLane
    orchestration: DeviceLane
    worker_pid: int
    by_id: dict[str, DeviceLane] = field(default_factory=dict)
    composition: dict[str, DeviceLane] = field(default_factory=dict)
    #: moves on every fuse and split: the seat set changed meaning (`admission_epoch`)
    revision: int = 0
    #: placement id -> why it runs on fewer devices than it was pinned to
    notes: dict[str, str] = field(default_factory=dict)

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

    def __len__(self) -> int:
        return len(self.lanes)

    @property
    def wide(self) -> bool:
        """More than one device in the envelope: the shape a pin decides (cr-068)."""
        return len(self.entries) > 1

    def lane_of(self, placement_id: str) -> DeviceLane | None:
        return next((lane for lane in self.lanes if placement_id in lane.placements), None)

    def holder_of(self, ordinal: int) -> DeviceLane | None:
        """The lane (device or group) that holds `ordinal` right now."""
        return next((lane for lane in self.lanes if ordinal in lane.ordinals), None)

    def release(self, placement_id: str) -> None:
        """A placement leaves: its rows go, and a group it held alone SPLITS."""
        self.notes.pop(placement_id, None)
        for lane in list(self.lanes):
            lane.placements.discard(placement_id)
            lane.forget(placement_id)
            if lane.group and not lane.placements:
                self.split(lane)

    # -------------------------------------------------------------------- fuse/split

    def _replace(self, remove: tuple[DeviceLane, ...], add: tuple[DeviceLane, ...]) -> None:
        kept = [lane for lane in self.lanes if lane not in remove]
        self.lanes = tuple(sorted((*kept, *add), key=lambda lane: lane.ordinals[0]))
        for lane in remove:
            self.by_id.pop(lane.lane_id, None)
        for lane in add:
            self.by_id[lane.lane_id] = lane
        self.revision += 1

    def fuse(self, ordinals: tuple[int, ...]) -> DeviceLane:
        """FUSE K empty device lanes into ONE group lane sealed to all K (cr-068).

        Every ordinal must be in the envelope and held by a device lane with no placement
        on it — a group never steals a device from a tenant, and never nests. The group's
        `devices` is the K entries joined, which is exactly what its seal imposes.
        """
        wanted = tuple(sorted(set(ordinals)))
        if len(wanted) < 2 or wanted != tuple(ordinals):
            raise LaneRefusal(
                "device_pin_infeasible",
                f"a group pin is K >= 2 sorted unique ordinals; got {list(ordinals)}",
            )
        members: list[DeviceLane] = []
        for ordinal in wanted:
            holder = self.holder_of(ordinal)
            if ordinal >= len(self.entries) or holder is None:
                raise LaneRefusal(
                    "device_pin_infeasible",
                    f"ordinal {ordinal} is outside this worker's envelope of "
                    f"{len(self.entries)} device(s)",
                )
            if holder.group:
                raise LaneRefusal(
                    "device_pin_infeasible",
                    f"ordinal {ordinal} is held by group {holder.lane_id} "
                    f"(placements {sorted(holder.placements)})",
                )
            if holder.placements:
                raise LaneRefusal(
                    "device_pin_infeasible",
                    f"{holder.lane_id} holds {sorted(holder.placements)}; a group takes only "
                    "empty devices and moves no other placement",
                )
            members.append(holder)
        fused = DeviceLane(
            lane_id_for(wanted),
            wanted,
            ",".join(self.entries[ordinal] for ordinal in wanted),
            worker_pid=self.worker_pid,
        )
        self._replace(tuple(members), (fused,))
        return fused

    def split(self, lane: DeviceLane) -> tuple[DeviceLane, ...]:
        """SPLIT an empty group lane back into its device lanes."""
        if not lane.group or lane.lane_id not in self.by_id:
            raise LaneRefusal("device_pin_infeasible", f"{lane.lane_id} is not a group lane")
        if lane.placements:
            raise LaneRefusal(
                "device_pin_infeasible",
                f"{lane.lane_id} still holds {sorted(lane.placements)}",
            )
        parts = tuple(
            DeviceLane(
                lane_id_for((ordinal,)),
                (ordinal,),
                self.entries[ordinal],
                worker_pid=self.worker_pid,
            )
            for ordinal in lane.ordinals
        )
        self._replace((lane,), parts)
        return parts

    # ------------------------------------------------------------------- assignment

    def assign(
        self,
        placement_id: str,
        *,
        model_bearing: bool,
        declared_bytes: int,
        measured: Mapping[int, DeviceMemory],
        pin: tuple[int, ...] | None = None,
        degrees: tuple[int, ...] = (),
    ) -> DeviceLane:
        """Choose a measured lane under its PIN; preparation authorizes its working set.

        The worker never orders attempts (#475); it does own GPU allocation (#659), and this
        is the whole of it. `measured` is the driver's reading of every envelope device,
        keyed by ordinal. `pin` is the RecordOwner's placement of this placement on THIS
        worker (proto-024 `device_pins`); `degrees` is what the package declares it can
        shard at (package interface `sequence_parallel.degrees`).

        A model-bearing placement is placed in this order, envelope order within each
        step, over its CANDIDATE lanes — the pinned lane, or every device lane of a
        one-device envelope — and every number is the driver's:

        1. a lane holding no other model-bearing placement whose measured free bytes cover
           the declared weight bytes — the placement has the device to itself (cr-066);
        2. a lane whose measured headroom still covers the declared bytes BESIDE its
           tenants — co-residency, no swap will ever be needed unless the measurements
           move (cr-022's co-fit arm);
        3. a lane whose measured TOTAL covers the declared bytes — time-sliced sharing
           (cr-022's swap arm): the placement fits the card alone, and the lane's residency
           arbitration gives it the card when its attempts need it.
        4. a readable lane for bounded preparation. Total weights can exceed VRAM when
           component staging or block paging is available. Assignment reserves no bytes:
           the measured prepare ceiling and actual construction/residency working set
           still decide whether the model can run.

        A pin of K > 1 ordinals is a GROUP pin (cr-068), placed at d = `effective_degree`:
        d == K fuses the whole pin; 1 < d < K fuses the first d empty pinned ordinals and
        leaves the rest as device lanes; d == 1 places by fit over the pinned device lanes.
        Either shortfall records a note in `notes`. Every fused device must report readable
        capacity; like a single-device lane, a group permits bounded preparation when header
        weight sizes exceed VRAM. Refusals move no other placement: `device_pin_infeasible`
        (envelope, occupancy), `device_group_unsupported` (a weightless group pin),
        `device_group_infeasible` (a fused device unreadable). A model-bearing placement with
        NO pin on a WIDE envelope is placed as if pinned to the whole envelope when every
        device is empty, and refuses `device_group_unsupported` otherwise: the worker does
        not choose among devices other placements hold. A weightless placement holds no
        device memory and goes to the device lane with the fewest placements (or its
        pinned lane), so its attempts do not all queue behind one seat.
        """
        current = self.lane_of(placement_id)
        if current is not None:
            return current
        if pin is not None and (not pin or tuple(sorted(set(pin))) != tuple(pin)):
            raise LaneRefusal(
                "device_pin_infeasible",
                f"placement {placement_id!r}: a pin is >= 1 sorted unique ordinals; "
                f"got {list(pin)}",
            )
        if (
            pin is None
            and model_bearing
            and self.wide
            and all(not lane.group and not lane.placements for lane in self.lanes)
        ):
            pin = tuple(range(len(self.entries)))
        if pin is not None and len(pin) > 1:
            return self._assign_group(
                placement_id, pin, model_bearing, declared_bytes, measured, degrees
            )
        if not model_bearing:
            if pin is None:
                pool = [lane for lane in self.lanes if not lane.group] or list(self.lanes)
                chosen = min(pool, key=lambda lane: (len(lane.placements), lane.ordinals))
            else:
                chosen = self._pinned_device_lane(placement_id, pin[0])
            chosen.placements.add(placement_id)
            return chosen
        if pin is None:
            if self.wide:
                raise LaneRefusal(
                    "device_group_unsupported",
                    f"placement {placement_id!r} carries no device pin on a "
                    f"{len(self.entries)}-device envelope and declares "
                    + (f"degrees {list(degrees)}" if degrees else "no sequence-parallel degree")
                    + "; the RecordOwner pins a placement to its devices and this worker "
                    "does not choose among devices other placements hold",
                )
            candidates = list(self.lanes)
        else:
            candidates = [self._pinned_device_lane(placement_id, pin[0])]
        return self._assign_by_fit(
            placement_id, candidates, declared_bytes, measured, pinned=pin is not None
        )

    def _pinned_device_lane(self, placement_id: str, ordinal: int) -> DeviceLane:
        holder = self.holder_of(ordinal)
        if ordinal >= len(self.entries) or holder is None:
            raise LaneRefusal(
                "device_pin_infeasible",
                f"placement {placement_id!r} is pinned to ordinal {ordinal}, outside this "
                f"worker's envelope of {len(self.entries)} device(s)",
            )
        if holder.group:
            raise LaneRefusal(
                "device_pin_infeasible",
                f"placement {placement_id!r} is pinned to ordinal {ordinal}, held by group "
                f"{holder.lane_id} for {sorted(holder.placements)}",
            )
        return holder

    def _assign_by_fit(
        self,
        placement_id: str,
        candidates: list[DeviceLane],
        declared_bytes: int,
        measured: Mapping[int, DeviceMemory],
        *,
        pinned: bool,
    ) -> DeviceLane:
        shortfalls: list[str] = []
        readable: list[tuple[DeviceLane, DeviceMemory]] = []
        for lane in candidates:
            memory = fold_memory(
                {
                    ordinal: measured.get(ordinal, DeviceMemory("unreadable"))
                    for ordinal in lane.ordinals
                }
            )
            if memory.state != "measured":
                shortfalls.append(f"{lane.lane_id} free bytes unreadable")
                continue
            readable.append((lane, memory))

        def headroom(lane: DeviceLane, memory: DeviceMemory) -> int:
            return memory.free_bytes

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
        bounded = sorted(
            (lane for lane, _memory in readable),
            key=lambda lane: (len(lane.model_bearing_slots()), lane.ordinals),
        )
        for chosen_set in (alone, beside, shared, bounded):
            if chosen_set:
                chosen = chosen_set[0]
                chosen.placements.add(placement_id)
                return chosen
        raise LaneRefusal(
            "device_pin_infeasible" if pinned else "device_lane_infeasible",
            f"no lane fits placement {placement_id!r}: " + "; ".join(shortfalls),
        )

    def _assign_group(
        self,
        placement_id: str,
        pin: tuple[int, ...],
        model_bearing: bool,
        declared_bytes: int,
        measured: Mapping[int, DeviceMemory],
        degrees: tuple[int, ...],
    ) -> DeviceLane:
        """Place a K-device pin at its effective degree, noting any device it leaves out.

        Header weight bytes cannot describe the working set of Runtime block paging.
        Preparation assigns one measured ceiling to every rank; construction and the
        attempt planner remain responsible for the actual simultaneous working set.
        """
        width = len(pin)
        if not model_bearing:
            raise LaneRefusal(
                "device_group_unsupported",
                f"placement {placement_id!r} is weightless and pinned to {width} devices; "
                "a group shards a model's attention and there is no model",
            )
        holders: list[DeviceLane] = []
        for ordinal in pin:
            holder = self.holder_of(ordinal)
            if ordinal >= len(self.entries) or holder is None:
                raise LaneRefusal(
                    "device_pin_infeasible",
                    f"placement {placement_id!r} is pinned to ordinal {ordinal}, outside this "
                    f"worker's envelope of {len(self.entries)} device(s)",
                )
            holders.append(holder)
        degree = effective_degree(width, degrees)
        if degree == 1:
            candidates = [holder for holder in holders if not holder.group]
            if not candidates:
                raise LaneRefusal(
                    "device_pin_infeasible",
                    f"placement {placement_id!r} is pinned to {list(pin)}, every one held by "
                    f"a group ({sorted({holder.lane_id for holder in holders})})",
                )
            lane = self._assign_by_fit(
                placement_id, candidates, declared_bytes, measured, pinned=True
            )
            self._note_degree(placement_id, pin, degrees, lane.ordinals)
            return lane
        empty = [
            ordinal
            for ordinal, holder in zip(pin, holders, strict=True)
            if not holder.group and not holder.placements
        ]
        if len(empty) < degree:
            held = {
                holder.lane_id: sorted(holder.placements)
                for holder in holders
                if holder.group or holder.placements
            }
            raise LaneRefusal(
                "device_pin_infeasible",
                f"placement {placement_id!r} is pinned to {list(pin)} and runs at degree "
                f"{degree}, but only {empty} are empty ({held} hold the rest); a group takes "
                "only empty devices and moves no other placement",
            )
        taken = tuple(empty[:degree])
        short = [
            f"ordinal {ordinal} ({self.entries[ordinal]!r}) free bytes unreadable"
            for ordinal in taken
            if (memory := measured.get(ordinal)) is None or memory.state != "measured"
        ]
        if short:
            raise LaneRefusal(
                "device_group_infeasible",
                f"placement {placement_id!r} cannot form a {degree}-device group: "
                + "; ".join(short),
            )
        fused = self.fuse(taken)
        fused.placements.add(placement_id)
        self._note_degree(placement_id, pin, degrees, taken)
        return fused

    def _note_degree(
        self,
        placement_id: str,
        pin: tuple[int, ...],
        degrees: tuple[int, ...],
        used: tuple[int, ...],
    ) -> None:
        left = len(pin) - len(used)
        if not left:
            return
        declared = f"package declares {list(degrees)}" if degrees else "package declares none"
        self.notes[placement_id] = (
            f"running at sequence-parallel degree {len(used)} on GPUs {list(used)} of pin "
            f"{list(pin)} ({declared}); {left} GPU(s) unused - rent a GPU count matching a "
            f"declared degree for full parallelism; placement {placement_id!r}"
        )
