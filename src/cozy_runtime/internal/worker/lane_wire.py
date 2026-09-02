"""Device lanes ON THE WIRE (proto-024, minor 22): the read and emit sites, isolated.

* `read_device_pins`      - `DesiredPlacementSet.device_pins[4]` -> {placement_id: ordinals}
* `emit_lanes`            - `ObservedWorkerState.lanes[26]` / `WorkerSnapshotBody.lanes[12]`,
                            each row with its `resident_placement_ids[5]` (proto-026)
* `emit_held_manifests`   - `ObservedWorkerState.held_manifests[27]` /
                            `WorkerSnapshotBody.held_manifests[13]` (proto-026)
* `emit_device_lane_id`   - `PlacementStatus.device_lane_id[19]`
* `lanes_on_wire`         - the fence condition: the NEGOTIATED minor carries lanes

RESIDENCY CROSSES AS TWO SETS AND NOTHING ELSE (residency-aware-routing.md §5, cr-080):
the placements resident on a lane and the manifests the verified store holds. No bytes,
ceiling, headroom, plan or LRU order is named in this module — `checks/architecture.py`'s
`residency-sets-only` fence refuses the vocabulary here and on every construction of the
three residency-bearing messages.

The vendored bindings carry every field; what varies per session is the RecordOwner's
minor. An owner below 22 cannot read `lanes[]` (one worker-level count over N lanes is the
mirror defect cr-066 fenced) and cannot send a pin, so a wide serving envelope still
refuses `device_lanes_unsupported` typed for that owner; the group machinery is reachable
only through a pin, which only a 22-owner can send.
"""

from __future__ import annotations

from typing import Any

from cozy_runtime.internal.worker.lanes import DeviceLane
from cozy_runtime.protocol import WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb

#: The minor that put lanes, pins and `device_lane_id` on the wire (worker-protocol#42).
LANES_WIRE_MINOR = 22

assert WIRE_MINOR >= LANES_WIRE_MINOR, f"vendored minor {WIRE_MINOR} predates device lanes"


def lanes_on_wire(negotiated_minor: int) -> bool:
    """Whether the RecordOwner of this session reads lanes: it negotiated at least the
    minor that carries them. This is the condition that lifts cr-066's fence."""
    return negotiated_minor >= LANES_WIRE_MINOR


def read_device_pins(desired_set: pb.DesiredPlacementSet) -> dict[str, tuple[int, ...]]:
    """`DesiredPlacementSet.device_pins` as {placement_id: ordinals, as sent}. `LaneSet.assign`
    validates the ordinals (sorted, unique, inside the envelope); this only reads them."""
    return {
        str(pin.placement_id): tuple(int(o) for o in pin.device_ordinals)
        for pin in desired_set.device_pins
    }


def lane_document(lane: DeviceLane, available_attempt_slots: int) -> dict[str, Any]:
    """ONE `DeviceLane` message's fields as a plain document: what `emit_lanes` writes and
    what the runtime's own records carry."""
    return {
        "lane_id": lane.lane_id,
        "device_ordinals": list(lane.ordinals),
        "available_attempt_slots": int(available_attempt_slots),
        "placement_ids": sorted(lane.placements),
        "resident_placement_ids": lane.resident_placement_ids(),
    }


def emit_lanes(
    target: pb.ObservedWorkerState | pb.WorkerSnapshotBody, rows: list[dict[str, Any]]
) -> None:
    """Append `rows` (from `lane_document`) to `target.lanes`, sorted by lane_id."""
    for row in sorted(rows, key=lambda row: str(row["lane_id"])):
        target.lanes.append(
            pb.DeviceLane(
                lane_id=str(row["lane_id"]),
                device_ordinals=[int(o) for o in row["device_ordinals"]],
                available_attempt_slots=int(row["available_attempt_slots"]),
                placement_ids=[str(p) for p in row["placement_ids"]],
                resident_placement_ids=[str(p) for p in row["resident_placement_ids"]],
            )
        )


def emit_held_manifests(
    target: pb.ObservedWorkerState | pb.WorkerSnapshotBody, digests: list[str]
) -> None:
    """`held_manifests`: the manifests the verified store holds complete, sorted unique."""
    target.held_manifests.extend(sorted(set(str(d) for d in digests)))


def emit_device_lane_id(status: pb.PlacementStatus, lane_id: str) -> None:
    """Set `PlacementStatus.device_lane_id`: the lane the placement's executor is on."""
    status.device_lane_id = lane_id
