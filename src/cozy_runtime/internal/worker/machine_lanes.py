"""Machine replicas: a durable placement template bound to its scheduler grant at admission."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec

from cozy_runtime import canonical_json
from cozy_runtime.internal import package_interface
from cozy_runtime.internal.worker.stage_scheduler import Ordinals, Want
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_executions import Preparation
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb

if TYPE_CHECKING:
    from .session import Worker

#: A replica's placement id prefix; the id hashes its template and its granted ordinals.
REPLICA = "machine-"


def state(preparation: Mapping[str, object]) -> pb.DesiredWorkerState:
    """The desired state of a retained preparation document."""
    return Preparation.of(preparation).desired()


@dataclass(frozen=True)
class Shape:
    """What one durable execution asks of the GPUs, read once from its preparation."""

    root: str
    kind: str  # "cpu" (no GPU), "job" (every GPU: job mode is worker-wide) or "serving"
    degrees: tuple[frozenset[int], ...] = ()
    template: tuple[str, str] = ("", "")
    refusal: str = ""
    gpus: int = 0  # the exact group width the owner submitted; 0 lets the scheduler choose
    interface: bytes = b""  # the template's PackageInterface, for its executor's prespawn
    entrypoint: str = ""  # the template entrypoint the call serves


def shape(worker: Worker, owner: str, request: str) -> Shape:
    executions = worker.executions
    assert executions is not None
    root, parent = executions.scheduling_root(owner, request)
    desired = executions.prepared(owner, request).desired()
    return _shape(worker, root, parent, desired, executions.offer(owner, request))


def recorded(worker: Worker, owner: str, request: str) -> Shape:
    """What a journaled execution, finished or not, asked of the GPUs, from its own offer."""
    executions = worker.executions
    assert executions is not None
    row = executions.row(owner, request)
    if row is None:
        return Shape(request, "cpu")
    return _shape(
        worker, "", "", executions.prepared(owner, request).desired(), row.attempt_offer()
    )


def _shape(
    worker: Worker, root: str, parent: str, desired: pb.DesiredWorkerState, offered: pb.AttemptOffer
) -> Shape:
    mode = desired.WhichOneof("mode")
    if mode == "job":
        found = Shape(root, "cpu" if worker._cpu_job(desired.job) else "job")
    elif mode == "placement_set":
        spec = documents.parse(offered.invocation_spec_canonical_bytes, pb.InvocationSpec)
        row = _selected(desired, offered.placement_id)
        wanted = documents.raw(spec.serving.entrypoint_binding_digest)
        name = next(e.name for e in row.entrypoints if e.entrypoint_binding_digest == wanted)
        models = next(
            e for e in package_interface.parse(row.package_interface).entrypoints if e.name == name
        ).models
        template = (row.installation_id, documents.spell(row.bindings_digest))
        degrees = tuple(
            frozenset(() if m.sequence_parallel is msgspec.UNSET else m.sequence_parallel.degrees)
            for m in models
        )
        gpus = desired.placement_set.execution_gpus
        found = Shape(
            root,
            "serving" if models else "cpu",
            degrees,
            template,
            gpus=gpus,
            interface=row.package_interface,
            entrypoint=name,
        )
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


def widths(found: Shape, readable: int) -> tuple[int, ...] | str:
    """The GPU counts `found` may run at on `readable` GPUs: each count every model slot
    declares (one always is), or the owner's exact one. A count no slot set declares is
    refused here; one the machine cannot form, the scheduler refuses with numbers."""
    if found.refusal:
        return found.refusal
    common = set.intersection(*({1, *d} for d in found.degrees)) if found.degrees else {1}
    if found.gpus:
        if found.gpus not in common:
            return (
                f"gpu_count_unavailable: exactly {found.gpus} GPUs is not a group every model "
                "slot declares"
            )
        return (found.gpus,)
    return tuple(sorted(w for w in common if w <= readable)) or (min(common),)


def want(worker: Worker, found: Shape) -> Want | str:
    """What a call of `found` asks of the stage scheduler, or why it can never ask."""
    devices = len(worker.lanes.entries)
    if found.kind == "job":  # a device job is worker-wide
        return Want(root=found.root, widths=(devices,), exclusive=True)
    readable = worker.readable_gpus()
    allowed = widths(found, len(readable))
    if isinstance(allowed, str):
        return allowed
    return Want(
        root=found.root,
        widths=allowed,
        template=found.template,
        exclude=tuple(o for o in range(devices) if o not in readable),
        entrypoint=found.entrypoint,
    )


def admit(worker: Worker, parent: str, preparation: bytes, offered: pb.AttemptOffer) -> None:
    """Refuse at admission what this machine can never run, with numbers, so it is never
    queued: below the measured floor of some stage at every width it may run at."""
    try:
        found = _shape(worker, "", parent, Preparation.read(preparation).desired(), offered)
    except Exception as exc:
        raise WorkspaceRefusal(f"gpu_shape_unreadable: {type(exc).__name__}: {exc}") from exc
    if found.refusal:
        raise WorkspaceRefusal(found.refusal)
    if found.kind != "serving":
        return
    asked = want(worker, found)
    if isinstance(asked, str):
        raise WorkspaceRefusal(asked)
    if (refused := worker.stages.refuse(asked)) is not None:
        raise WorkspaceRefusal(f"no_capacity: {refused.detail}")


def _selected(desired: pb.DesiredWorkerState, placement_id: str) -> pb.Placement:
    rows = documents.parse(desired.placement_set.placement_set_canonical_bytes, pb.PlacementSet)
    return next(row for row in rows.placements if row.placement_id == placement_id)


def placement_for(
    preparation: dict[str, object], offered: pb.AttemptOffer, ordinals: Ordinals
) -> dict[str, object]:
    """The replica of the durable template on exactly `ordinals`; never persisted.

    The same template on the same devices reuses its supervised executor. No ordinals is
    a weightless replica sealed to no device.
    """
    if not preparation:
        return preparation
    desired = state(preparation)
    if desired.WhichOneof("mode") != "placement_set":
        return preparation
    replica = _selected(desired, offered.placement_id)
    identity = canonical_json.encode(
        [replica.installation_id, documents.spell(replica.bindings_digest), list(ordinals)]
    )
    replica.placement_id = REPLICA + hashlib.sha256(identity).hexdigest()[:32]
    raw = documents.canonical_bytes(pb.PlacementSet(placements=[replica]))
    desired.placement_set.placement_set_canonical_bytes = raw
    desired.placement_set.placement_set_digest = hashlib.sha256(raw).digest()
    del desired.placement_set.device_pins[:]
    if ordinals:
        desired.placement_set.device_pins.add(
            placement_id=replica.placement_id, device_ordinals=ordinals
        )
    offered.placement_id = replica.placement_id
    return dict(preparation, state=base64.b64encode(desired.SerializeToString()).decode())
