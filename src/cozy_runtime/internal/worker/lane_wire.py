"""Device lanes ON THE WIRE (proto-024, minor 22): the read and emit sites, isolated.

Everything about group lanes is runtime-internal and built against the CURRENT vendored
minor; this module is the whole of what changes when `cozy.worker.v1` is re-vendored at 22.
Each function below carries the minor-22 body it will have, gated on `LANES_WIRE_MINOR`
against the vendored `WIRE_MINOR`, so that a worker at 20 or 21 is BYTE-IDENTICAL on the
wire (no field it does not have is touched) and the flip is mechanical:

* `read_device_pins`  - `DesiredPlacementSet.device_pins[4]` -> {placement_id: ordinals}
* `emit_lanes`        - `ObservedWorkerState.lanes[26]` / `WorkerSnapshotBody.lanes[12]`
* `emit_device_lane_id` - `PlacementStatus.device_lane_id[19]`
* `lanes_on_wire`     - the fence condition: the negotiated minor carries lanes

Until then a pin cannot arrive on the wire at all (a 22-owner's `device_pins` lands in a
20-worker's unknown fields), so the serving fence over a wide envelope stays
(`device_lanes_unsupported`) and the group machinery is reachable only in-process.
"""

from __future__ import annotations

from typing import Any

from cozy_runtime.internal.worker.lanes import DeviceLane
from cozy_runtime.protocol import WIRE_MINOR
from cozy_runtime.protocol import worker_pb2 as pb

#: The minor that puts lanes, pins and `device_lane_id` on the wire (worker-protocol#42).
LANES_WIRE_MINOR = 22


def lanes_on_wire(negotiated_minor: int) -> bool:
    """Whether BOTH peers read lanes: the vendored bindings carry them AND the owner
    negotiated at least that minor. This is the condition that lifts cr-066's fence."""
    return WIRE_MINOR >= LANES_WIRE_MINOR and negotiated_minor >= LANES_WIRE_MINOR


def read_device_pins(desired_set: pb.DesiredPlacementSet) -> dict[str, tuple[int, ...]]:
    """`DesiredPlacementSet.device_pins` as {placement_id: sorted unique ordinals}.

    FLIP AT 22: replace the body with
        return {
            str(pin.placement_id): tuple(int(o) for o in pin.device_ordinals)
            for pin in desired_set.device_pins
        }
    """
    del desired_set
    return {}


def lane_document(lane: DeviceLane, available_attempt_slots: int) -> dict[str, Any]:
    """ONE `DeviceLane` message's fields, as the runtime's own document until the message
    exists in the vendored bindings; the same rows land in records today."""
    return {
        "lane_id": lane.lane_id,
        "device_ordinals": list(lane.ordinals),
        "available_attempt_slots": int(available_attempt_slots),
        "placement_ids": sorted(lane.placements),
    }


def emit_lanes(
    target: pb.ObservedWorkerState | pb.WorkerSnapshotBody, rows: list[dict[str, Any]]
) -> None:
    """Append `rows` (from `lane_document`, sorted by lane_id) to `target.lanes`.

    FLIP AT 22: replace the body with
        for row in sorted(rows, key=lambda row: str(row["lane_id"])):
            target.lanes.append(
                pb.DeviceLane(
                    lane_id=str(row["lane_id"]),
                    device_ordinals=[int(o) for o in row["device_ordinals"]],
                    available_attempt_slots=int(row["available_attempt_slots"]),
                    placement_ids=[str(p) for p in row["placement_ids"]],
                )
            )
    """
    del target, rows


def emit_device_lane_id(status: pb.PlacementStatus, lane_id: str) -> None:
    """Set `PlacementStatus.device_lane_id`.

    FLIP AT 22: replace the body with
        status.device_lane_id = lane_id
    """
    del status, lane_id
