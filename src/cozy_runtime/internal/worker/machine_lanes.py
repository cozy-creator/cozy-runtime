"""Machine-owned serving replicas share the existing placement and device-lane engine."""

from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.internal.worker import machine_capture
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker


def state(preparation: dict[str, Any]) -> pb.DesiredWorkerState:
    return pb.DesiredWorkerState.FromString(base64.b64decode(preparation["state"], validate=True))


def replica(
    worker: Worker,
    owner: str,
    placement: dict[str, Any],
    models: list[dict[str, Any]],
    defaults: list[dict[str, Any]] | None = None,
    *,
    parent_request: str = "",
) -> tuple[dict[str, Any], list[pb.PlacementDevicePin]] | None:
    """Reserve a stable replica through the durable child submission, not a GPU hint.

    Outstanding requests charge their selected devices, including queued preparations.
    A singleton model therefore fills all granted devices before queuing another round.
    An explicit authored count reserves exactly that group. Earlier captures without
    a count retain the largest supported group behavior.
    """
    selected_degree = degree(worker, models, defaults)
    if not models or not worker.lanes.wide:
        return placement, []
    width = len(worker.lanes.entries)
    loads = [0] * width
    executions = worker.executions
    assert executions is not None
    charged: set[str] = set()
    for request in executions.pending(owner):
        preparation = executions.preparation(owner, request)
        if not preparation:
            continue
        desired = state(preparation)
        if desired.WhichOneof("mode") != "placement_set":
            continue
        charged.add(request)
        for pin in desired.placement_set.device_pins:
            for ordinal in pin.device_ordinals:
                if ordinal < width:
                    loads[ordinal] += 1
    # Direct offers and a prior fenced owner's draining attempts share these same
    # physical lanes. Do not route onto a busy device merely because its work was
    # admitted outside this owner's durable execution queue.
    for attempt in worker.engine.history.values():
        if attempt.request_id in charged or attempt.state not in (
            *worker.engine.PRE_RELEASE,
            *worker.engine.HELD,
        ):
            continue
        lane = worker.lanes.by_id.get(attempt.lane_id)
        if lane is not None:
            for ordinal in lane.ordinals:
                loads[ordinal] += 1
            charged.add(attempt.request_id)
    available = tuple(range(width))
    affinity = getattr(worker, "machine_groups", None)
    if parent_request and affinity is not None:
        available = affinity.available(owner, parent_request, selected_degree, loads)
    if len(available) < selected_degree:
        return None
    # Additive lane load admits any free compatible set, without aligned buckets
    # or enumerating every possible group on large machines.
    ordinals = tuple(
        sorted(sorted(available, key=lambda index: (loads[index], index))[:selected_degree])
    )
    # The replica is a placement instance, not a new package or binding. Reusing the
    # same immutable code/models on the same devices reuses its supervised executor.
    identity = canonical_json.encode(
        [
            machine_capture.installation_identity(placement),
            placement["bindings_digest"],
            list(ordinals),
        ]
    )
    selected = dict(placement)
    selected["placement_id"] = "machine-" + hashlib.sha256(identity).hexdigest()[:32]
    return selected, [
        pb.PlacementDevicePin(placement_id=selected["placement_id"], device_ordinals=ordinals)
    ]


def degree(
    worker: Worker, models: list[dict[str, Any]], defaults: list[dict[str, Any]] | None = None
) -> int:
    """Selected model capacity; planning this count never reserves a GPU."""
    requested = {row.get("gpus", 0) for row in (defaults or [])} - {0}
    if len(requested) > 1:
        raise WorkspaceRefusal("one serving construction has conflicting default GPU counts")
    requested_degree = next(iter(requested), 0)
    if requested_degree > len(worker.lanes.entries):
        raise WorkspaceRefusal("the requested default GPU group exceeds the granted device pool")
    if not models:
        return 0
    if not worker.lanes.wide:
        return 1
    degrees = set.intersection(
        *(set(model.get("sequence_parallel", {}).get("degrees", [])) | {1} for model in models)
    )
    width = len(worker.lanes.entries)
    degree = (
        requested_degree
        if requested_degree
        else max((value for value in degrees if 0 < value <= width), default=width)
    )
    if degree not in degrees or degree > width:
        raise WorkspaceRefusal("the requested default GPU group is not supported on this machine")
    return degree


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
