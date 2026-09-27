"""Machine replicas: a durable placement template bound to its scheduler grant at admission."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cozy_runtime import canonical_json
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.worker import machine_capture
from cozy_runtime.internal.worker.gpu_scheduler import Ordinals, model_degrees
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker

#: A replica's placement id prefix; the id hashes its template and its granted ordinals.
REPLICA = "machine-"


def state(preparation: dict[str, Any]) -> pb.DesiredWorkerState:
    return pb.DesiredWorkerState.FromString(base64.b64decode(preparation["state"], validate=True))


@dataclass(frozen=True)
class Shape:
    """What one durable execution asks of the GPUs, read once from its preparation."""

    root: str
    kind: str  # "cpu" (no GPU), "job" (every GPU: job mode is worker-wide) or "serving"
    degrees: tuple[frozenset[int], ...] = ()
    template: tuple[str, str] = ("", "")
    refusal: str = ""


def shape(worker: Worker, owner: str, request: str) -> Shape:
    executions = worker.executions
    assert executions is not None
    root, parent = executions.scheduling_root(owner, request)
    preparation = executions.preparation(owner, request)
    desired = state(preparation) if preparation else pb.DesiredWorkerState()
    mode = desired.WhichOneof("mode")
    if mode == "job":
        found = Shape(root, "cpu" if worker._cpu_job(desired.job) else "job")
    elif mode == "placement_set":
        offered = executions.offer(owner, request)
        spec = documents.read(offered.invocation_spec_canonical_bytes, pb.InvocationSpec)
        row = _selected(desired, offered.placement_id)
        interface = package_interface.read_bytes(base64.b64decode(row["package_interface"]))
        name = next(
            entry["name"]
            for entry in row["entrypoints"]
            if entry["entrypoint_binding_digest"] == spec["serving"]["entrypoint_binding_digest"]
        )
        models = next(e for e in interface["entrypoints"] if e["name"] == name).get("models", [])
        template = (machine_capture.installation_identity(row), row["bindings_digest"])
        degrees = tuple(frozenset(d) for d in model_degrees(models))
        found = Shape(root, "serving" if models else "cpu", degrees, template)
    else:
        return Shape(root, "cpu")  # an owner-placed serving root already has its lane
    held = worker.engine.live.get(parent) if parent else None
    lane = worker.lanes.by_id.get(held.lane_id) if held is not None else None
    if found.kind != "cpu" and lane is not None and lane.ordinals:
        return Shape(
            root,
            found.kind,
            refusal=f"nested_gpu_call: {parent} holds GPUs {list(lane.ordinals)} and cannot "
            "await another GPU call; make the caller a CPU orchestration job",
        )
    return found


def _selected(desired: pb.DesiredWorkerState, placement_id: str) -> dict[str, Any]:
    rows = canonical_json.decode(desired.placement_set.placement_set_canonical_bytes)["placements"]
    return next(row for row in rows if row["placement_id"] == placement_id)


def placement_for(
    preparation: dict[str, Any], offered: pb.AttemptOffer, ordinals: Ordinals
) -> dict[str, Any]:
    """The replica of the durable template on exactly `ordinals`; never persisted.

    The same template on the same devices reuses its supervised executor. No ordinals is
    a weightless replica sealed to no device.
    """
    if not preparation or state(preparation).WhichOneof("mode") != "placement_set":
        return preparation
    desired = state(preparation)
    selected = _selected(desired, offered.placement_id)
    identity = canonical_json.encode(
        [
            machine_capture.installation_identity(selected),
            selected["bindings_digest"],
            list(ordinals),
        ]
    )
    replica = dict(selected, placement_id=REPLICA + hashlib.sha256(identity).hexdigest()[:32])
    raw = canonical_json.encode(
        {"format": "cozy.worker.v1.PlacementSet/1", "placements": [replica]}
    )
    desired.placement_set.placement_set_canonical_bytes = raw
    desired.placement_set.placement_set_digest = hashlib.sha256(raw).digest()
    del desired.placement_set.device_pins[:]
    if ordinals:
        desired.placement_set.device_pins.add(
            placement_id=replica["placement_id"], device_ordinals=ordinals
        )
    offered.placement_id = replica["placement_id"]
    return dict(preparation, state=base64.b64encode(desired.SerializeToString()).decode())


def compose(worker: Worker, desired: pb.DesiredWorkerState) -> pb.DesiredWorkerState | None:
    """Keep every placement the incoming replica does not overlap; wait for held overlap.

    Called under the worker control lock. Existing convergence retires obsolete executor
    processes before releasing/fusing lanes, so this owner never pretends cached VRAM is free.
    """
    selected = canonical_json.decode(desired.placement_set.placement_set_canonical_bytes)
    incoming = {row["placement_id"]: row for row in selected["placements"]}
    pins = {
        pin.placement_id: tuple(pin.device_ordinals) for pin in desired.placement_set.device_pins
    }
    wanted = {ordinal for values in pins.values() for ordinal in values}
    current = ([worker.placement] if worker.placement is not None else []) + [
        hosted.placement for hosted in worker.hosted.values()
    ]
    kept = dict(incoming)
    for placement in current:
        identity = placement.placement_id
        if identity in incoming:
            continue
        held_pin = placement.device_pin
        if not wanted.intersection(held_pin or ()):
            kept[identity] = placement.document
            if held_pin is not None:
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
