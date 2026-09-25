"""Machine-owned serving replicas share the existing placement and device-lane engine."""

from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import machine_capture
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker


def state(preparation: dict[str, Any]) -> pb.DesiredWorkerState:
    return pb.DesiredWorkerState.FromString(base64.b64decode(preparation["state"], validate=True))


def replica(
    worker: Worker, owner: str, placement: dict[str, Any], models: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[pb.PlacementDevicePin]]:
    """Reserve a stable replica through the durable child submission, not a GPU hint.

    Outstanding requests charge their selected devices, including queued preparations.
    A singleton model therefore fills all granted devices before queuing another round.
    Model-declared sequence parallelism chooses the largest supported group.
    """
    if not models or not worker.lanes.wide:
        return placement, []
    degrees = set.intersection(
        *(set(model.get("sequence_parallel", {}).get("degrees", [1])) for model in models)
    )
    width = len(worker.lanes.entries)
    degree = max((value for value in degrees if 0 < value <= width), default=width)
    groups = [tuple(range(start, start + degree)) for start in range(0, width - degree + 1, degree)]
    loads = [0] * width
    executions = worker.executions
    assert executions is not None
    for request in executions.pending(owner):
        preparation = executions.preparation(owner, request)
        if not preparation:
            continue
        desired = state(preparation)
        if desired.WhichOneof("mode") != "placement_set":
            continue
        for pin in desired.placement_set.device_pins:
            for ordinal in pin.device_ordinals:
                if ordinal < width:
                    loads[ordinal] += 1
    ordinals = min(groups, key=lambda group: (sum(loads[index] for index in group), group))
    # The replica is a placement instance, not a new package or binding. Reusing the
    # same immutable code/models on the same devices reuses its supervised executor.
    identity = canonical_json.encode(
        [machine_capture.code_identity(placement), placement["bindings_digest"], list(ordinals)]
    )
    selected = dict(placement)
    selected["placement_id"] = "machine-" + hashlib.sha256(identity).hexdigest()[:32]
    return selected, [
        pb.PlacementDevicePin(placement_id=selected["placement_id"], device_ordinals=ordinals)
    ]


def compose(worker: Worker, desired: pb.DesiredWorkerState) -> pb.DesiredWorkerState | None:
    """Keep disjoint placements; wait for overlapping work before topology replacement.

    Called under the worker control lock. Existing convergence retires obsolete executor
    processes before releasing/fusing lanes, so this owner never pretends cached VRAM is free.
    """
    selected = canonical_json.decode(desired.placement_set.placement_set_canonical_bytes)
    incoming = {row["placement_id"]: row for row in selected["placements"]}
    pins = {
        pin.placement_id: tuple(pin.device_ordinals) for pin in desired.placement_set.device_pins
    }
    wanted = {ordinal for values in pins.values() for ordinal in values}
    # Unpinned single-device/weightless preparations retain the existing serialized path.
    if not wanted:
        return desired
    current = ([worker.placement] if worker.placement is not None else []) + [
        hosted.placement for hosted in worker.hosted.values()
    ]
    kept = dict(incoming)
    for placement in current:
        identity = placement.placement_id
        if identity in incoming:
            continue
        held_pin = placement.device_pin
        if held_pin is not None and not wanted.intersection(held_pin):
            kept[identity] = placement.document
            pins[identity] = held_pin
    for attempt in worker.engine.history.values():
        if attempt.state not in (*worker.engine.PRE_RELEASE, *worker.engine.HELD):
            continue
        if attempt.lane_id.startswith("cpu-") and attempt.state != "queued":
            continue
        held = worker._placement_by_id(attempt.placement_id)
        if (
            held is None
            or held.placement_id not in kept
            or held.document != kept[held.placement_id]
            or held.device_pin != pins.get(held.placement_id)
        ):
            return None
    raw = canonical_json.encode(
        {
            "format": "cozy.worker.v1.PlacementSet/1",
            "placements": [kept[key] for key in sorted(kept)],
        }
    )
    combined = pb.DesiredWorkerState()
    combined.CopyFrom(desired)
    combined.placement_set.CopyFrom(
        pb.DesiredPlacementSet(
            placement_set_digest=hashlib.sha256(raw).digest(),
            placement_set_canonical_bytes=raw,
            device_pins=[
                pb.PlacementDevicePin(placement_id=identity, device_ordinals=pins[identity])
                for identity in sorted(pins)
            ],
        )
    )
    return combined
