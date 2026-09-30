"""The `cozy.worker.v1` WORKER: the machine's one control process, hosting `WorkerControl`.

The worker is the gRPC SERVER; the record-plane owner dials it (worker-protocol/01,
decisions #436/#454 — the 2026-08-25 re-landing). Two RPCs run at launch and their
durability attributes are the whole design:

* **Control** — one durable bidi stream per accepted claim, fenced by the three-field
  envelope. Durable frames are never shed. Terminal authority lives only here.
* **WatchProgress** — bounded and LOSSY, owner-opened on its own connection. Saturating it
  can never block Control: a different RPC, a different queue, a different thread.

The conversation on every accepted stream is fixed (02): Claim -> ClaimAck -> ONE
`WorkerSnapshot` -> the RecordOwner's `SnapshotAck` -> dispatch opens. The ownership fence is
`(record_owner_epoch, control_stream_epoch, worker_boot_id)`, checked on every frame
before any body field is read. LAUNCH TIER (single-owner): the epoch is a provisioned
constant — a claim with a strictly higher epoch supersedes, an equal epoch from the same
RecordOwner is a reconnect, anything else refuses typed.

Threads, and why each exists: each accepted STREAM runs its conversation on the host's
thread and must never do slow work (a cancel has to arrive while an attempt runs); ONE
DEVICE LANE PER DEVICE of the envelope (`worker/lanes.py`, cr-066) holds that device's lease
and serializes its attempts in arrival order — the whole of worker-local arbitration, per
lane; the REPORTER ticks the durable Report; the WATCHDOG owns the deadline and the cancel
escalation, because the deadline is worker-enforced and never merely handed to author code.

This process never imports torch. `scripts/worker-live.py nocuda` proves that against the
RUNNING process's loaded modules and mapped libraries, not against the import graph.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import os
import queue
import re
import resource
import shutil
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import msgspec

from cozy_runtime.author._executor_requests import (
    Answer,
    DeviceRoom,
    Handler,
    ModelPrefetch,
    Request,
    Room,
    refuse,
)
from cozy_runtime.internal import (
    accel,
    canonical,
    child_env,
    execution_evidence,
    fill,
    hostfacts,
    job_plan,
    liveness,
    machine_kernels,
    package_environment,
    package_installation,
    prepare_diagnostics,
    proctree,
    readiness,
    tolerant,
    weights_sink,
)
from cozy_runtime.internal.config import RuntimeConfig
from cozy_runtime.internal.executor_commands import (
    Activate,
    Binding,
    Budgets,
    Custody,
    DescribeInstalled,
    Load,
    ModelLoad,
    Restore,
    Start,
    Unload,
)
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.worker import (
    acquire,
    activity,
    child,
    downloads,
    execution_unit,
    gpu_scheduler,
    grants,
    internal_calls,
    lane_wire,
    machine_byte_inputs,
    machine_capture,
    machine_lanes,
    machine_model_defaults,
    machine_model_resolve,
    machine_release_catalog,
    machine_slots,
    model_source_prepare,
    observe,
    page_warm,
    prespawn,
    store_gc,
    triage,
    weights,
    workspace_executions,
)
from cozy_runtime.internal.worker.attempts import (
    CAUSE,
    DEVICE_SHORTFALL_CODES,
    GROUP_FAULT_CODES,
    ORIGIN,
    STATUS,
    AttemptEngine,
    AttemptRecord,
    AttemptRefusal,
    AttemptSlot,
    Released,
    safe,
)
from cozy_runtime.internal.worker.child import (
    INTERFACE_DOCUMENT,
    ExecutorGone,
    ExecutorProtocolMismatch,
    ExecutorSupervision,
    socket_path_refusal,
)
from cozy_runtime.internal.worker.control import (
    ControlHost,
    WatchFanout,
)
from cozy_runtime.internal.worker.gpu_scheduler import GpuBroker, width_for
from cozy_runtime.internal.worker.lanes import (
    DeviceLane,
    LaneRefusal,
    LaneRow,
    LaneSet,
)
from cozy_runtime.internal.worker.ledger import Ledger, read_vmrss
from cozy_runtime.internal.worker.machine_owner_memo import OwnerMemo
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority, origin_key
from cozy_runtime.internal.worker.memory import MemoryManager
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    JobBinding,
    PlanRefusal,
    PreparedModel,
)
from cozy_runtime.internal.worker.products import Products
from cozy_runtime.internal.worker.records import WorkerRecords
from cozy_runtime.internal.worker.supervisor import Stopping, Supervisor, Unit, fault_text
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.internal.worker.workspace_calls import Call
from cozy_runtime.internal.worker.workspace_executions import ExecutionChanged
from cozy_runtime.protocol import (
    CAPABILITY_UNAVAILABLE_CODE,
    MIN_COMPATIBLE_WIRE_MINOR,
    WIRE_MINOR,
    documents,
)
from cozy_runtime.protocol import worker_pb2 as pb

#: The lossy lane's per-watch depth. Oldest is shed first; a durable frame never enters it.
PROGRESS_QUEUE = 256

#: Report cadence. Report is DURABLE and carries the applied baseline, so the owner can
#: always see a directive it issued but the worker did not apply.
REPORT_SECONDS = 2.0


class _Progress(msgspec.Struct, frozen=True):
    """A `progress` frame's measured coordinates; an unknown one stays None and is omitted."""

    stage: str = ""
    step_ms: float = 0.0
    stage_fraction: float | None = None
    position: int | None = None
    total: int | None = None
    overall_fraction: float | None = None
    unit: str | None = None
    rate: float | None = None
    call_request: str | None = None
    call_attempt: int | None = None


def _progress_document(frame: Mapping[str, canonical.Json]) -> dict[str, canonical.Json]:
    """Preserve the measured progress coordinates the executor already supplied.

    Stage-local and overall fractions stay separate. Optional coordinates are omitted when
    the package does not know them; no layer fabricates whole-invocation completion. A
    byte stage names its `unit` and the `rate` it measured since its previous sample.
    Other events retain the executor's public record, already bounded and redacted at
    emission. Only seam routing fields are removed; a log's scalar value is its level,
    not its message, and a load's value may be None beside a measured position.
    """
    kind = frame.get("kind", "progress")
    if kind != "progress":
        routing = {"event", "request_id", "kind"}
        return {"type": kind, "payload": {k: v for k, v in frame.items() if k not in routing}}
    progress, _ = tolerant.read(frame, _Progress, _Progress.__struct_fields__)
    payload = {k: v for k, v in msgspec.structs.asdict(progress).items() if v is not None}
    payload["stage"] = progress.stage[:120]
    return {"type": kind, "payload": payload}


class _Delivery(msgspec.Struct, frozen=True):
    """A load reply's `delivery` (`resolution.Plan.wire_document`), as the worker narrates it."""

    variant: str = ""
    label: str = "unlabelled"
    rung: str = "unreported"
    delivery: str = "unreported"
    materialization: str = ""
    """Empty when nothing decodes: the bytes are served verbatim."""
    routes: dict[str, int] | None = None
    kernels: dict[str, int] | None = None
    leaf_speedup_x: float = -1.0
    calibrated: bool = False
    objective: str = "unreported"
    confession: str = ""
    models: dict[str, object] = {}


def _delivery_observation(delivery: _Delivery) -> str:
    """Render only the current resolution-plan projection; narration never gates serving."""

    def census(value: dict[str, int] | None) -> str:
        if value is None:
            return "unreported"
        return ", ".join(f"{key} x{count}" for key, count in sorted(value.items())) or "none"

    # The leaf-speed evidence rides BESIDE the route census it qualifies: a plan that routes
    # `encoded_gemm x453` on a card whose probe measured the leaf at 0.61x its float GEMM
    # says so on the same line, rather than banking the number where nothing reads it.
    speedup = delivery.leaf_speedup_x
    leaf = f"; leaf {speedup:.2f}x its float GEMM" if speedup >= 0.0 else ""
    selection = "per-model delivery" if delivery.models else f"on rung {delivery.rung}"
    calibrated = "calibrated" if delivery.calibrated else "uncalibrated"
    return (
        f"{delivery.variant!r} / {delivery.label} {selection} "
        f"({delivery.delivery}/{delivery.materialization or 'verbatim'}; "
        f"routes {census(delivery.routes)}; kernels {census(delivery.kernels)}{leaf}; "
        f"{calibrated}) chosen on the {delivery.objective} axis"
        + (f" - {delivery.confession}" if delivery.confession else "")
    )


class _InterfaceFault(Exception):
    """A preparation step that FAILED rather than refused.

    It is deliberately not a `PreparationRefusal`: the loopback servicer answers a typed
    refusal with FAILED_PRECONDITION, which the pod host journals as the permanent answer
    to an immutable request, and a fault about this pod's condition must never be written
    into that row. This one reaches the servicer's default arm and is answered INTERNAL —
    the owner is told now, and nothing is cached.
    """


def _interface_failure(reply: Mapping[str, object]) -> Exception:
    """Turn one not-ok executor reply into the right kind of exception.

    `executor.classify` already computed `terminal` and `origin` and put them in the reply;
    reading only `code` and `detail` discarded the verdict the executor had made. A package
    author's ConformanceError and a PermissionError from the runtime's own plumbing are not
    the same failure and must not settle the same way.

    `origin` is read too, and it is the reason this is not simply `if terminal`. The
    unhandled-exception default attributes an unrecognised exception to the AUTHOR, so an
    infrastructure fault arrives here already mislabelled; a failure the executor did not
    positively attribute to the package is not treated as a verdict about the package.
    """
    from cozy_runtime.internal.worker import package_prepare

    code = str(reply.get("code", "invalid reply"))
    detail = str(reply.get("detail", ""))
    said = f"{code}: {detail}" if detail else code
    terminal = bool(reply.get("terminal", True))
    origin = str(reply.get("origin", ""))
    if terminal and origin in ("author", "package"):
        return package_prepare.PreparationRefusal("package_prepare_interface_failed", said)
    return _InterfaceFault(said)


@dataclass(frozen=True, slots=True)
class WorkerOptions:
    """One worker's settings, as a typed value rather than a parser's leftovers.

    The worker has two entrances — the long-lived `serve` process and the ephemeral local
    worker a one-shot `run` builds in memory (§8) — and only one of them has a command
    line. The owner is NOT here: who drives this worker is a transport
    (`worker/control.py`), not a setting.
    """

    root: Path
    worker_id: str = "wrk-local"
    instance_id: str = ""
    worker_boot_id: str = ""
    """Externally provisioned boot id for fixed pods; local workers mint one in-process."""
    ownership_required: bool = False
    """A retained pod readiness envelope requires existing accepted control history."""
    sole_supervisor: bool = False
    """Private machine-agent contract: authentication without durable client ownership."""
    process_incarnation: str = ""
    """Identifies this supervised process; execution and checkpoint identities outlive it."""
    owns_tensorfs_store: bool = False
    """The fixed pod owns this Store; local workers may share one and own only their work."""
    #: PROVENANCE, and every one of these is EMPTY when the launch did not supply it (#494g).
    #: The defaults used to name a real SDXL release and a `sha256:00…00` artifact, which rode
    #: real ClaimAcks and real BootFailures as if they had been measured. Absent is absent: an
    #: owner reading an empty `release_id` learns the truth, and an owner reading a plausible
    #: lie learns nothing it can act on.
    release_id: str = ""
    git_commit: str = ""
    python: str = ""
    accelerator_backend: str = ""
    """Baked pod profile backend (`cuda`/`none`), or empty for local auto-detection."""
    devices: str = "0"
    gpus: tuple[readiness.RuntimeGPU, ...] = ()
    """The measured inventory `devices` spans, in ordinal order, reported to the controller.
    Tests supply a virtual one here; nothing reads an environment variable for it."""
    grant_roots: tuple[str, ...] = ()
    """Every directory tree a `file://` DeliveryGrant may address on this machine (#506b).
    A launcher that knows its own root states it — cozy-creator passes its local root, a
    pod's provisioner passes the media subtree its co-resident server owns — and the
    Authorizer refuses a grant outside every one of them. Empty declares no fence."""
    install_root: Path | None = None
    """The immutable install root whose receipt refs select content-keyed package venvs.

    The provisioner/local installer chooses storage; the worker receives that path and owns
    no retention or GC policy.  Absent means no already-materialized generation is selectable.
    """
    environment_python: Path | None = None
    """An already-complete local package venv; mutually exclusive with materialization root."""
    publication_authority: PublicationAuthority | None = None
    """Privileged worker publication authority, never an executor or captured-code value."""
    hubs: tuple[PublicationAuthority, ...] = ()
    """Every other Hub this machine is registered at (wire 67): a run naming one reads its
    release, index and Models there. Publication stays at the default Hub."""
    hub_access_path: Path | None = None
    """Atomic machine-agent access projection, re-read when privileged work uses a Hub."""
    tensorfs_root: Path | None = None
    """The one host-local executable TensorFS Store."""
    artifact_cache: Path | None = None
    """Where the dynamic acquisition lane writes the exact bytes it fetches (cr-049).

    Content-addressed and re-readable: a complete object is never re-fetched, which is what
    makes a kill -9 mid-materialize resume by digest. Absent means no dynamic acquisition
    was provisioned and a placement naming artifacts this worker does not already hold
    reports a typed fault rather than guessing a directory."""
    worker_tls_certificate_digest: str = ""
    alloc_conf: str = "expandable_segments:True"
    threads: int = 4
    claim_ready: bool = True
    """False only for fixed product-host boot until listener/readiness proof commits."""
    executor_uid: int = -1
    executor_gid: int = -1
    """Fixed package identity, or ``-1/-1`` for the native same-user development lane."""
    triage_root: Path | None = None
    """Where terminal triage bundles are committed; absent means ``<root>/triage``.

    A pod worker commits them under the media plane's subtree so the owner can fetch one by
    its opaque subject over the same plane the outputs cross (cl-101); a local worker's
    owner reads them from the worker root directly."""
    triage_owner: tuple[int, int] | None = None
    """The uid/gid each committed bundle is handed to — the pod media plane's, so a plane
    that runs as its own uid can open a file the root worker wrote. Absent keeps the
    writer's identity."""
    activity_path: Path | None = None
    """Where the reporter publishes what this worker is doing, for the pod supervisor that
    exec'd it (`worker/activity.py`). Absent — every local worker — publishes nothing:
    there is no second process holding a rental open."""
    activity_owner: tuple[int, int] | None = None
    """The uid/gid each published activity file is handed to, the same handoff the readiness
    receipt makes. Absent keeps the writer's identity."""


@dataclass(slots=True)
class Placement:
    """The ONE hosted placement (launch enforces len(placements) <= 1, §2). The id is the
    RecordOwner-minted routing key; it is never invocation identity.

    `placement_set_digest` plus `placement_id` is the placement's identity. A single-placement
    replace is this SAME `placement_id` in a new set, which is why routing survives rollout.
    """

    document: pb.Placement
    """The accepted Placement, decoded and checked once by `read_placement_set`."""
    placement_set_digest: bytes
    acquisition: pb.PlacementAcquisitionObservation = field(
        default_factory=pb.PlacementAcquisitionObservation
    )
    #: TWO INDEPENDENT AXES (§8, #473/#482), because one enum cannot say "staged on disk and
    #: offline" — the exact state an outgoing placement holds under fallback-retention.
    materialization: pb.MaterializationState = pb.MaterializationState.MATERIALIZATION_STATE_ABSENT
    serving: pb.ServingState = pb.ServingState.SERVING_STATE_OFFLINE
    #: #474/#485c: the predecessor kept for restore. EMPTY means replacement is PAUSED —
    #: fallback capability is a PRECONDITION of replacement, not a nicety.
    retained_fallback_placement_set_digest: bytes = b""
    #: The lane this placement's executor is sealed to (cr-066): assigned by the worker by
    #: measured fit (or the RecordOwner's pin) before activation, empty while unassigned;
    #: on the wire as `PlacementStatus.device_lane_id` (proto-024).
    lane_id: str = ""
    #: The operational pin this placement was converged under, including a failed assignment.
    device_pin: tuple[int, ...] | None = None
    #: The environment THIS boot materialized when it prepared a local revision. The
    #: wheels were the pod host's carriers and are gone once the PlacementSet is journaled,
    #: so the placement over that revision serves from this environment instead of
    #: re-acquiring bytes nothing on the pod still holds.
    installed: package_installation.InstalledEnvironment | None = None

    @property
    def placement_id(self) -> str:
        return self.document.placement_id

    @property
    def installation_id(self) -> str:
        return self.document.installation_id

    @property
    def package_interface(self) -> bytes:
        return self.document.package_interface

    @property
    def bindings_digest(self) -> bytes:
        return self.document.bindings_digest

    @property
    def prepared_key(self) -> tuple[str, str, bytes]:
        """A prepared placement's exact identity: one release prepared for several model
        selections holds one placement per selection, and none displaces another."""
        return self.placement_id, self.installation_id, self.bindings_digest


#: Attempt states between device entry and device release.
ON_DEVICE = frozenset({"entering", "accepted", "running", "reclaiming", "finalizing"})


@dataclass(slots=True)
class HostedPlacement:
    """One additional placement and the process that imports only its package.

    The original single-placement fields remain the primary slot.  Keeping this value small
    makes the co-fitting cut equally small: package-local bindings, failures and supervision
    never become a second scheduler or a shared environment. A hosted placement may bear a
    model (cr-022): its executor is sealed to the lane the worker assigned it, prepared
    through the same generation prepare as the primary, and shares that lane's device by
    residency arbitration when it does not co-fit.
    """

    placement: Placement
    supervision: ExecutorSupervision
    bindings: dict[str, DeclaredBinding] = field(default_factory=dict)
    failed_bindings: dict[str, str] = field(default_factory=dict)
    latched: str = ""


@dataclass(frozen=True, slots=True)
class Tenant:
    """ONE placement's executor slot on its lane, as a prepare reads it.

    The primary slot and a hosted one keep their state in different places; the generation
    prepare (`_prepare_generation`) reads neither directly. It reads this, so the two
    cannot drift apart again — the hosted prepare used to be a second, thinner path with no
    assigned ceiling, no construction facts and no per-binding faults, which is why it could
    host weightless packages only.
    """

    placement: Placement
    supervision: ExecutorSupervision
    bindings: dict[str, DeclaredBinding]
    failed_bindings: dict[str, str]
    lane: DeviceLane
    slot: str

    def row(self) -> LaneRow:
        return self.lane.row(self.slot)


@dataclass(slots=True)
class OwnershipFence:
    """WHO may drive this worker, and on WHICH stream (02 §0/§4).

    Split out of the old single `Fences` bag (#484): ownership and accepted intent are two
    unrelated lifetimes — a reconnect mints a new `control_stream_epoch` and touches no
    accepted revision at all, and #494a's reconnect defect was exactly that confusion made
    structural. `activity` lives in NEITHER of the two halves; it is the worker's own
    narration lane and now sits on the worker.

    `record_owner_epoch` is authority-granter-minted; `control_stream_epoch` is
    WORKER-minted, incremented for every accepted stream. Fixed pods retain both the boot
    identity and accepted counters across process replacement; local workers mint a fresh
    boot. The executor epoch is PLACEMENT truth (#454) and deliberately not here.
    """

    worker_boot_id: str = ""
    record_owner_epoch: int = 0
    record_owner_id: str = ""
    #: TRUE once this pod boot accepted a (record_owner_epoch, record_owner_id) pair.
    record_owner_recorded: bool = False
    control_stream_epoch: int = 0  # worker-minted; 0 = never claimed
    snapshot_id: str = ""  # the snapshot the CURRENT stream was sent
    wire_minor: int = 0


@dataclass(slots=True)
class AcceptedDesiredState:
    """WHAT this worker was told to be, and the exact bytes it accepted (§4/§8).

    `accepted_desired_state_revision` advances on ACCEPTANCE — which is all it ever honestly
    meant — and `converged_revision` (computed, never stored here) advances only when the
    observed placement state satisfies it. `applied_revision` is retired as dishonest (#473):
    it advanced on acceptance while claiming to report reality.

    `placement_set_canonical_bytes` is the EXACT accepted bytes, recorded and re-quoted —
    never a re-serialization and never a structured projection (§4). The snapshot carries
    these bytes verbatim.
    """

    accepted_desired_state_revision: int = 0
    desired_state_body: bytes = b""  # the revision's own desired-state bytes (#494a)
    accepted_placement_set_digest: bytes = b""
    placement_set_canonical_bytes: bytes = b""
    posture: pb.Posture = pb.Posture.POSTURE_UNSPECIFIED
    drain_grace_ms: int = 0
    mode: str = ""  # the DesiredWorkerState `mode` oneof this worker APPLIED
    #: placement_id -> the ordinals the RecordOwner pinned it to (proto-024 `device_pins`,
    #: cr-068). Beside the digest-fenced document, never inside it.
    device_pins: dict[str, tuple[int, ...]] = field(default_factory=dict)
    #: The revision whose placements are all materially STAGED and DISPATCHABLE. It may NEVER
    #: advance past a placement that is not — the field is a DERIVED fact, and a worker that
    #: advances it while observed state disagrees is in breach, not merely optimistic (§8).
    converged_revision: int = 0


#: Every envelope-bearing message type sets the same three fields; the stamp is central so
#: no constructor can forget one (the fence is checked before any body field is read).
def _stamped(message: Any, fence: OwnershipFence) -> Any:
    message.record_owner_epoch = fence.record_owner_epoch
    message.control_stream_epoch = fence.control_stream_epoch
    message.worker_boot_id = fence.worker_boot_id
    return message


def _refused(revision: int, **fault: Any) -> pb.Fault:
    """A desired revision left unapplied."""
    return pb.Fault(desired_state_revision=revision, **fault)


def _capability_unavailable(operation: str, minor: int) -> str:
    return (
        f"{operation} needs wire minor {MIN_COMPATIBLE_WIRE_MINOR} or newer and the owner "
        f"speaks {minor}; update Creator. Other work on this worker continues."
    )


def _same_parent(left: pb.JobDirective, right: pb.JobDirective) -> bool:
    """One CPU orchestration parent: the prepared job it runs. Policy fields (limits,
    publication) may change without replacing a parent, which keeps the contract
    it captured when it started."""
    return (left.installation_id, left.job_descriptor_id, left.orchestration) == (
        right.installation_id,
        right.job_descriptor_id,
        right.orchestration,
    )


def desired_state_body(desired: pb.DesiredWorkerState) -> bytes:
    """The canonical bytes of a DesiredWorkerState's DESIRED STATE — envelope excluded (#494a).

    A DesiredWorkerState carries two unrelated things: what the RecordOwner wants this worker
    to be (revision, posture, drain grace, the mode's set or job), and the per-connection
    facts that route it (record_owner_epoch, control_stream_epoch, worker_boot_id, and the
    wire minor the sender speaks). Only the first is the body §3's same-revision rule is
    about: an owner restarted on a newer Creator re-sends the same revision at its own minor.

    Hashing the whole message made `control_stream_epoch` — which the WORKER mints
    afresh for every accepted stream, by design — part of the body, so the identical desired
    revision re-sent after an ordinary reconnect classified as `same_revision_changed_body`
    and took a permanent CONFIG_REFUSED fault. Same revision + same desired state is now the
    idempotent re-apply it always was; same revision + genuinely different desired state
    stays the protocol refusal, because that is the case the rule exists for.
    """
    body = pb.DesiredWorkerState()
    body.CopyFrom(desired)
    body.record_owner_epoch = 0
    body.control_stream_epoch = 0
    body.worker_boot_id = ""
    body.wire_minor = 0
    return documents.canonical_bytes(body)


#: The fault reason asking the owner to prepare a stale-shaped placement set again.
REPREPARE = "placement_set_reprepare_required"
SEMVER = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


class _StalePlacement(msgspec.Struct, frozen=True):
    """The shape an older Runtime prepared: the interface by Ref, no installation."""

    package_interface: dict[str, object]


class _StaleSet(msgspec.Struct, frozen=True):
    placements: tuple[_StalePlacement, ...]


def _stale(data: bytes) -> bool:
    try:
        return bool(msgspec.json.decode(data, type=_StaleSet).placements)
    except msgspec.DecodeError:
        return False


def _reprepare() -> documents.DocumentError:
    # Nothing is wrong with the request: its owner prepares the set again on this worker.
    return documents.DocumentError(
        REPREPARE,
        "this placement set was prepared by an older Runtime; prepare the package set again "
        "on this worker",
    )


def read_placement_set(desired_set: pb.DesiredPlacementSet) -> list[pb.Placement]:
    """§4's ORDER OF OPERATIONS, and it is not negotiable.

    RECOMPUTE `sha256(placement_set_canonical_bytes)` and compare BEFORE parsing a single
    field. The frozen wire "digest-addressed" the set by an ECHOED claim, which is a
    statement the sender makes about bytes it also sent; this is a fence the receiver
    computes. Only on a match is the document decoded, once, into the typed message.
    Fields this worker does not consume are ignored at every depth, and the nested binding
    digests are the producer's names.

    Raises `documents.DocumentError`; the caller turns that into the typed fault and leaves
    the previous set serving.
    """
    data = desired_set.placement_set_canonical_bytes
    recomputed = documents.digest_of(data)
    if recomputed != desired_set.placement_set_digest:
        raise documents.DocumentError(
            "placement_set_digest_mismatch",
            f"the set claims {desired_set.placement_set_digest.hex()[:16]}… and its own "
            f"{len(data)} B hash to {recomputed.hex()[:16]}… — nothing in it has been parsed",
        )
    try:
        placements = list(documents.parse(data, pb.PlacementSet).placements)
    except documents.DocumentError as exc:
        if _stale(data):
            raise _reprepare() from exc
        raise
    for placement in placements:
        _check_placement(placement)
    if len({placement.placement_id for placement in placements}) != len(placements):
        raise documents.DocumentError("duplicate_placement", "placement_id must be unique")
    return placements


def _unique(values: list[str], name: str) -> set[str]:
    if len(values) != len(set(values)):
        raise documents.DocumentError("duplicate_entry", f"{name} repeats an entry")
    return set(values)


def _identifier(value: str, name: str) -> str:
    if not value or value.strip() != value:
        raise documents.DocumentError("invalid_identifier", f"{name} is empty or padded")
    if len(value) > 256 or any(not 0x20 <= ord(char) <= 0x7E for char in value):
        raise documents.DocumentError("invalid_identifier", f"{name} is not bounded ASCII")
    return value


def _check_placement(placement: pb.Placement) -> None:
    """What the schema cannot say: bounded identifiers, exact refs, one package mode, and
    every model a slot, component or adapter names declared once in `models`."""
    if not placement.installation_id:
        raise _reprepare()
    _identifier(placement.placement_id, "Placement.placement_id")
    _identifier(placement.installation_id, "Placement.installation_id")
    if not 0 < len(placement.package_interface) <= canonical.DOC_MAX_BYTES:
        raise documents.DocumentError("package_interface_invalid", "invalid interface bytes")
    if not placement.bindings_digest:
        raise documents.DocumentError("malformed_digest", "Placement.bindings_digest")
    match placement.WhichOneof("package_mode"):
        case "package":
            package = _identifier(placement.package.package, "PackageSelection.package")
            _identifier(placement.package.release, "PackageSelection.release")
            if package.count("/") != 1 or not all(package.split("/")):
                raise documents.DocumentError("package_invalid", "package identity is invalid")
        case "development":
            development = placement.development
            _identifier(development.package, "DevelopmentPackage.package")
            _identifier(development.release, "DevelopmentPackage.release")
            _identifier(development.installation_id, "DevelopmentPackage.installation_id")
        case _:
            raise documents.DocumentError("package_mode_invalid", "select exactly one package mode")
    known = _unique([_identifier(model.id, "Model.id") for model in placement.models], "models")
    for model in placement.models:
        _identifier(model.repo, f"{model.id}.repo")
        if model.version:
            _identifier(model.version, f"{model.id}.version")
        if model.lane:
            _identifier(model.lane, f"{model.id}.lane")
        if not model.manifest.digest or model.manifest.length <= 0:
            raise documents.DocumentError("invalid_ref", f"{model.id}.manifest is not exact")
    names = [
        _identifier(entrypoint.name, "Entrypoint.name") for entrypoint in placement.entrypoints
    ]
    _unique(names, "Placement.entrypoints")
    for entrypoint in placement.entrypoints:
        if not entrypoint.entrypoint_binding_digest:
            raise documents.DocumentError(
                "malformed_digest", "Entrypoint.entrypoint_binding_digest"
            )
        _unique([_slot(slot, known) for slot in entrypoint.slots], "Entrypoint.slots")


def _slot(slot: pb.Slot, known: set[str]) -> str:
    # Retired `stamps` and `model_construction_contract` are read and never consulted:
    # fit is the header and the census.
    named = [slot.reference_model_id, *(row.model_id for row in slot.components)]
    if any(model_id not in known for model_id in named):
        raise documents.DocumentError("unknown_model", f"Slot {slot.slot!r} names a model")
    _unique(
        [_identifier(row.component, "Component.component") for row in slot.components],
        "Slot.components",
    )
    for adapter in slot.adapters:
        if adapter.model_id not in known:
            raise documents.DocumentError("unknown_model", "ModelAdapter.model_id")
        _identifier(adapter.component, "ModelAdapter.component")
        _identifier(adapter.source_component, "ModelAdapter.source_component")
        if not adapter.scale.strip():
            raise documents.DocumentError("invalid_scale", "ModelAdapter.scale")
    return _identifier(slot.slot, "Slot.slot")


def _rank_degree(lane: DeviceLane, bindings: Iterable[DeclaredBinding]) -> int:
    """The executor's world: the group lane's K for a model-bearing placement, else 1.
    A weightless package has nothing to shard, so it runs one rank whatever the seal's
    width (a job's envelope lane is the same case)."""
    return lane.degree if lane.group and any(not b.weightless for b in bindings) else 1


def _group_parameters(group: Sequence[DeclaredBinding]) -> frozenset[str]:
    """Every model parameter name the group's bindings spell (h3a-018)."""
    return frozenset(
        model.model_parameter_name for binding in group for model in binding.model_bindings()
    )


def _parameter_names_by_model(group: Sequence[DeclaredBinding]) -> list[list[str]]:
    """Per model position, the parameter names the group binds it under. Bindings of one
    construction hold their models in the same order (the key sorts them by path)."""
    width = max((len(binding.model_bindings()) for binding in group), default=0)
    return [
        sorted(
            {
                binding.model_bindings()[index].model_parameter_name
                for binding in group
                if index < len(binding.model_bindings())
            }
        )
        for index in range(width)
    ]


def executor_load_command(
    binding: DeclaredBinding,
    *,
    construction: str,
    devices: str,
    authorized_device_limit_bytes: int = 0,
    sequence_parallel_degree: int = 1,
    parameter_names: Sequence[Sequence[str]] = (),
    attention_pin: str = "",
) -> Load:
    """Translate one resolved callable binding into the executor's closed `load` shape.

    `construction` is the binding's construction key: the executor builds it once and
    answers a second load of the same key from what it built.

    `devices` is what the executor's seal must say (`CUDA_VISIBLE_DEVICES`), so the executor
    can refuse `env_seal_broken` if it is preparing under any other seal than its lane's.
    `authorized_device_limit_bytes` is the ceiling the worker ASSIGNED for this prepare
    (cr-025/cr-066); the executor fits under the smaller of it and what it measures, and
    never derives one of its own. `sequence_parallel_degree` rides only when > 1 (cr-068):
    a group lane's K, the ONE new fact its executor receives; a plain prepare is unchanged.
    `parameter_names[i]` is every parameter name the i-th model is bound under across the
    bindings that share this construction (h3a-018); absent, the model binds under this
    binding's own name. `attention_pin` rides the same way when a caller prepares a
    construction for one kernel (cr-125). The worker never does: a request's pin is an
    execution parameter of `invoke`.
    """

    models = []
    for index, model in enumerate(binding.model_bindings()):
        names = {model.model_parameter_name}
        if index < len(parameter_names):
            names.update(parameter_names[index])
        models.append(
            ModelLoad(
                binding=Binding(
                    application=binding.application,
                    package_interface=binding.interface_path,
                    model_class=model.model_class,
                    model_binding_path=model.model_binding_path,
                    model_parameter_name=model.model_parameter_name,
                    model_parameter_names=tuple(sorted(names)),
                    component=model.components[0] if model.components else "",
                    components=tuple(model.components),
                    snapshots=dict(model.snapshots),
                    store=model.store,
                    snapshot=model.reference_snapshot,
                    release=binding.release,
                    variant=model.variant,
                    development=binding.development,
                    # The executor's decode refuses any other custody.
                    custody=cast(Custody, model.custody),
                    objective=model.objective,
                    steps_basis=model.steps_basis,
                    package=binding.package,
                    model=model.model,
                ),
                budgets=Budgets(logical_weight_bytes=model.logical_weight_bytes),
            )
        )
    command = Load(
        construction=construction,
        devices=devices,
        authorized_device_limit_bytes=int(authorized_device_limit_bytes),
        sequence_parallel_degree=sequence_parallel_degree,
        attention_pin=attention_pin,
    )
    if len(models) > 1:
        return msgspec.structs.replace(command, models=tuple(models))
    single = (
        models[0]
        if models
        else ModelLoad(
            binding=Binding(
                application=binding.application,
                package_interface=binding.interface_path,
            )
        )
    )
    return msgspec.structs.replace(command, binding=single.binding, budgets=single.budgets)


def _claim_descriptor_headroom() -> None:
    """Raise the soft RLIMIT_NOFILE to what the host allows before the worker serves.

    The TensorFS reader pins one verified descriptor per object for a lease's whole life
    and refuses FD_HEADROOM rather than re-resolve a pathname (tfs-021); a container's
    default soft limit (1024) is below one real checkpoint (paul/anima pins 979). The
    worker claims the headroom here and every executor it spawns inherits it.
    """

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or soft >= hard:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except (ValueError, OSError) as exc:
        print(f"[worker] descriptor headroom stays at soft={soft}: {exc}", flush=True)
        return
    print(f"[worker] descriptor headroom soft={soft} -> {hard}", flush=True)


class Worker:
    """The no-CUDA control parent — the machine's WORKER process (#455/#484; the name
    `Supervisor` is retired everywhere, and "supervision" survives only as the verb for its
    executor management)."""

    def __init__(self, config: RuntimeConfig, options: WorkerOptions, host: ControlHost) -> None:
        if host.persistent and not config.record_owner_public_key:
            raise ValueError("a worker listener needs the RecordOwner's public key")
        self.config = config
        self.options = options
        self.hub_access_lock = threading.Lock()
        self.hub_access_principals = {
            origin_key(held.origin): held.principal
            for held in options.hubs
            + (() if options.publication_authority is None else (options.publication_authority,))
            if held.access_token
        }
        self.host = host
        self.root = Path(options.root)
        #: Per-slot executor uids (pod co-hosting). Each additional executor slot owns a
        #: DISJOINT uid domain: the pod census cannot attribute processes by token — a
        #: setuid child is non-dumpable and docker-default pods carry no SYS_PTRACE, so
        #: /proc/<pid>/environ is unreadable even to the worker — so one uid per slot is
        #: what keeps one slot's reclaim from freezing a sibling's serving executor.
        #: The primary keeps the seat's own uid.
        self._slot_uids: dict[str, int] = {}
        #: (store, manifest, component) this worker already read into the page cache
        self._pages_warmed: set[tuple[str, str, str]] = set()
        #: per placement: bindings digests of superseded documents -> {binding: construction}
        #: whose offers are still accepted when the binding and its construction are unchanged
        self._superseded: dict[str, dict[str, dict[str, str]]] = {}
        #: `host_facts()`: measured once per boot, forgotten when an executor dies
        self._host_facts: hostfacts.HostFacts | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        _claim_descriptor_headroom()
        self.base = package_environment.observe_base(
            options.environment_python or Path(options.python or sys.executable)
        )
        if options.executor_uid >= 0:
            # Set before any package child exists and before the worker starts serving.
            # The uid split closes ordinary ptrace; non-dumpable is the explicit second belt.
            proctree.deny_process_inspection()
        self.fence = OwnershipFence(
            worker_boot_id=options.worker_boot_id or f"boot-{uuid.uuid4().hex[:20]}"
        )
        self.accepted = AcceptedDesiredState()
        #: The non-attempt narration lane. In NEITHER fence half (#484): it is neither an
        #: authority fact nor accepted intent, it is what this worker has been doing.
        self.activity: list[pb.ActivityEvent] = []
        #: MACHINE lifecycle, pulled out of the placement enum (#482). BOOTING until the
        #: first accepted claim finds a worker that booted; FAILED is terminal for this boot.
        self.phase: pb.WorkerPhase = pb.WorkerPhase.WORKER_PHASE_BOOTING
        #: THE ONE WORKER-LEVEL ADMISSION FENCE (§6). It bumps whenever the MEANING of
        #: admission changes — executor respawn, window resize, phase change, cutover — and
        #: an `AttemptOffer` echoing anything else refuses DETERMINISTICALLY. Per-placement
        #: `attempt_credits` are deleted: N counters over ONE serialized device advertise
        #: N x the real capacity, which is an arithmetic defect, not a tuning problem.
        self.admission_epoch = 1
        self.seq = itertools.count(1)  # WORKER-owned monotone counter
        self.progress_seq: dict[tuple[str, int], itertools.count[int]] = {}
        self.records = WorkerRecords()
        #: THE LANES (cr-066): one serialized device lane per envelope entry, in envelope
        #: order, plus the whole-envelope lane a job's executor is sealed under. Each lane
        #: owns its attempt thread, its work queue, its `settled` gate and one ledger row per
        #: resident executor generation — nothing device-scoped lives on the worker any more.
        self.lanes = LaneSet.from_envelope(options.devices, worker_pid=os.getpid())
        #: every device byte of this machine, across every executor, package and rank
        self.memory = MemoryManager(self.lanes, self.note, self._end_tenant)
        self.authorizer = grants.Authorizer(roots=tuple(options.grant_roots))
        # Before any executor starts: an older executor reads the image kernel site once.
        self.machine_kernels = machine_kernels.boot(
            config.kernel_cache, options.devices, dict(config.child_base_env)
        )
        self.supervision = ExecutorSupervision(
            root=self.root,
            python=options.python or sys.executable,
            base_env=config.child_base_env,
            cozy_home=config.cozy_home,
            executor_uid=options.executor_uid,
            executor_gid=options.executor_gid,
            jit_pod_scope=self.fence.worker_boot_id,
            kernel_cache=config.kernel_cache,
        )
        # ExecutorSupervision already holds the process-lifetime worker-root lease.
        # Restore only under that lease so two processes cannot allocate one stream.
        self.ownership_path = (
            self.root / "ownership.json"
            if options.worker_boot_id and not options.sole_supervisor
            else None
        )
        if self.ownership_path is not None:
            from cozy_runtime.internal.worker import ownership

            try:
                owner, identity, stream = ownership.read(
                    self.ownership_path,
                    self.fence.worker_boot_id,
                    required=options.ownership_required,
                )
            except Exception:
                self.supervision.close()
                raise
            self.fence.record_owner_epoch = owner
            self.fence.record_owner_id = identity
            self.fence.control_stream_epoch = stream
            self.fence.record_owner_recorded = stream != 0
        if options.sole_supervisor:
            # Keep the existing machine journal namespace on adoption. These fixed wire
            # fields name records; they do not elect an owner or survive as control history.
            self.fence.record_owner_epoch = 1
            self.fence.record_owner_id = "cozy-local-client"
            self.fence.record_owner_recorded = True
            self.fence.wire_minor = WIRE_MINOR
        self.supervision.on_change = self.executor_changed
        # One accepted stream at a time may mutate control state. Claims take this through
        # their snapshot, so ownership cannot move in the middle of convergence and a frame
        # from the superseded stream cannot finish under its successor's epoch.
        self.control_lock = threading.RLock()
        self.preparation_lock = threading.Lock()
        #: The one preparation executor slot (`placements/prepare`). A package preparation and
        #: a submission's built-in capture both describe in it; two at once collide on its
        #: worker-root lease, which refused concurrent first submissions on a fresh pod.
        self.preparation_slot = threading.Lock()
        self.model_source_operations: set[str] = set()
        self.restart_requested = False
        self.restart_sequence = 0
        self.idle_restart_pending = threading.Event()
        self.compute_idle_since = time.monotonic()
        if options.owns_tensorfs_store:
            try:
                self.model_source_operations.update(
                    model_source_prepare.source_operations(
                        tensorfs_root=Path(options.tensorfs_root or "")
                    )
                )
            except Exception:
                self.supervision.close()
                raise
        # Exact code/interface/environment revisions prepared over the existing seam.
        # Captured machine executions persist their immutable receipt and restore this
        # same inventory after restart, independent of their code's publication origin.
        self.prepared_installations: dict[tuple[str, str, bytes], Placement] = {}
        #: Successful interfaces of owned environment incarnations, shared by all prepares.
        self._installed_interfaces: dict[
            tuple[str, tuple[int, int, int, int], Path, str, str], bytes
        ] = {}
        self.local_operations: dict[tuple[str, str], Placement] = {}
        self.claim_ready = threading.Event()
        if options.claim_ready:
            self.claim_ready.set()
        self.bindings: dict[str, DeclaredBinding] = {}
        #: plan id -> why THIS binding could not be prepared (#572d). A faulted binding
        #: refuses its own entrypoints and denies its siblings nothing.
        self.failed_bindings: dict[str, str] = {}
        self.jobs: dict[str, JobBinding] = {}
        from cozy_runtime.internal.worker.job_slots import JobSlot

        self.job_slots: dict[str, JobSlot] = {}
        #: idle CPU executors by (installation, job, directive), adopted by the next request
        self.warm_cpu: dict[tuple[str, str, bytes], JobSlot] = {}
        #: the last (owner epoch, proof) whose Ed25519 signature this worker verified
        self._proven_claim: tuple[int, bytes] = (0, b"")
        self.placement: Placement | None = None
        #: The INCOMING placement being staged while `placement` keeps serving (#474). It is
        #: a separate field precisely because one field cannot hold both, which is the same
        #: reason the observed state has two axes.
        self.pending: Placement | None = None
        #: Additional placements hosted beside the primary one, weightless or model-bearing.
        #: Each gets its own immutable uv environment and executor, sealed to the lane the
        #: worker assigned it; an attempt takes its own placement's lane seat and no other.
        self.hosted: dict[str, HostedPlacement] = {}
        self.pending_hosted: list[pb.Placement] = []
        #: Job mode has no placement, so it carries its own dispatchability edge (cr-009).
        self.job_ready = False
        #: An activation failure LATCHES for the accepted revision (§8): non-empty means the
        #: worker has stopped retrying and `converged_revision` stays behind on purpose.
        self.latched = ""
        #: The latch this worker has already TOLD an owner about. A latch it never published
        #: is a refusal only this process can read (cr-061); see `_publish_latch`.
        self._published_latch = ""
        self.boot = observe.BootRecord()
        self.monitor = observe.SilenceMonitor(self.boot.ring, busy=self.read_device_work)
        #: thread name -> kernel thread id of each `_lane`-wrapped thread (read by `pollers`)
        self.lane_threads: dict[str, int] = {}
        self.supervisor = Supervisor(
            integrity=self.fail_machine, note=lambda step: self.note("supervision", step)
        )
        #: units waiting for an attempt to leave or a placement to settle
        self._capacity_waiters: set[str] = set()
        #: signalled whenever a held attempt leaves (placement drains wait on it)
        self.attempts_left = threading.Condition()
        #: replicas one execution is activating right now; the others wait for it
        self._activating: set[str] = set()
        #: the owner this process's one boot reconcile ran for
        self._reconciled = ""
        #: terminal executions whose hold this boot releases once their effects settle
        self.releasing: set[str] = set()
        self.failure_reason = ""
        self.bundles = triage.BundleStore(
            options.triage_root or self.root / "triage",
            self.records,
            owner=options.triage_owner,
        )
        self.stop = threading.Event()
        from cozy_runtime.internal.worker.workspace import Workspace
        from cozy_runtime.internal.worker.workspace_rpc import Service as WorkspaceService

        self.workspace = (
            Workspace(options.tensorfs_root) if options.tensorfs_root is not None else None
        )
        self.executions = (
            workspace_executions.Executions(self.workspace) if self.workspace else None
        )
        self._pressure_registration = None
        if self.workspace is not None:
            from cozy_runtime.internal import storage_admission

            from . import workspace_memo

            workspace = self.workspace
            self._pressure_registration = storage_admission.register_workspace(
                workspace.store_root,
                lambda target: workspace_memo.reclaim_unused(workspace, target_bytes=target),
            )
        self.resolutions = machine_model_resolve.Resolutions()
        self.workspace_service = (
            WorkspaceService(
                self.workspace,
                self.authorize_workspace,
                self.resolutions,
                private_egress_allowed=lambda: machine_model_defaults.local_development(self),
            )
            if self.workspace is not None
            else None
        )
        self.weights = weights.WeightsExchange(
            store_root=options.tensorfs_root,
            stamp=self.stamp,
            stop=self.stop,
            owner_scope=lambda: self.fence.record_owner_id,
            note=self.note,
            workspace=self.workspace,
        )
        self.engine = AttemptEngine(
            records=self.records,
            authorizer=self.authorizer,
            jobs={},
            spool_root=self.root / "spool",
            instance_id=options.instance_id,
            progress=self.emit_progress,
            dispatched=self._dispatched,
            bundles=self.bundles,
            monitor=self.monitor,
            posture=self.applied_posture,
            device_process=self.read_device_process,
            weights=self.weights,
            owner_scope=lambda: self.fence.record_owner_id,
            tensorfs_root=options.tensorfs_root,
            inventory=options.gpus,
            room=lambda attempt: self._room(
                self.lanes.by_id[self.engine.slot_of(attempt).lane_id], attempt.placement_id
            ),
            work_fingerprint=lambda attempt: weights_sink.work_fingerprint(
                attempt.spec,
                self.numerical_environment(),
                operation_identity=(
                    self.engine._job_declaration(attempt.job, attempt.job.job_descriptor_id)
                    .get("invocable", {})
                    .get("operation_identity", "")
                    if attempt.job
                    else ""
                ),
            ),
        )
        self.engine.boot_id = self.fence.worker_boot_id
        from cozy_runtime.internal.worker.calls import Calls

        self.model_transfer_lock = threading.Lock()
        self.model_transfers: dict[str, threading.Lock] = {}
        self.calls = Calls(
            self._machine_call,
            workspace=self.workspace,
            owner=lambda: self.fence.record_owner_id,
            prefetch=self._machine_prefetch,
            release_gpus=self._release_gpus,
            installed_bindings=self._installed_call_bindings,
        )
        from .machine_calls import Calls as MachineCalls
        from .machine_effects import Effects

        self.machine_calls = MachineCalls(self) if self.workspace is not None else None
        if self.workspace is not None and self.executions is not None:
            self.engine.products = Products(self.workspace, self.executions, self.calls)
        if self.executions is not None:
            self.executions.changed = self._journal_changed
        self.gpu = GpuBroker(
            len(self.lanes.entries),
            granted_to=lambda key: self.supervisor.poke(execution_unit.key(key.rpartition("#")[0])),
            observed=self._gpu_observed,
            warm=self._warm,
            fault=self.fail_machine,
        )
        self.prespawns = prespawn.Prespawns(self)
        self._dead_replicas: dict[str, child.Executor] = {}
        from .machine_sources import Sources as MachineSources

        self.machine_sources = MachineSources(self)
        self.machine_effects = (
            Effects(
                self.workspace,
                self.executions,
                options.publication_authority,
                authority_for=self._execution_publication_authority,
                progress=self.native_progress,
            )
            if self.workspace is not None and self.executions is not None
            else None
        )
        if self.machine_effects is not None:
            self.machine_effects.settled = self._effect_settled
        self.engine.calls = self.calls
        self.owner_memo = OwnerMemo(self.executions) if self.executions is not None else None
        from cozy_runtime.internal.worker.source_calls import SourceCalls

        self.source_calls = (
            SourceCalls(
                self.workspace,
                lambda: self.fence.record_owner_id,
                self.machine_sources.observe,
                lambda command: (
                    command.record_owner_epoch == self.fence.record_owner_epoch
                    and command.control_stream_epoch == self.fence.control_stream_epoch
                    and command.worker_boot_id == self.fence.worker_boot_id
                ),
                spool_root=self.engine.spool_root,
                progress=self.native_progress,
                publication=(
                    self.machine_effects.source_client if self.machine_effects is not None else None
                ),
                memo=self.owner_memo,
            )
            if self.workspace is not None
            else None
        )
        if self.source_calls is not None:
            self.source_calls.finished = self.machine_sources.poke

        #: The durable lane's outbound queue, and it belongs to ONE accepted stream. A new
        #: claim installs a FRESH queue under `claim_lock`, so the superseded stream's sender
        #: keeps draining its own (now unread) queue and can never consume a frame authored
        #: for the live stream. A single process-wide queue made that theft routine: the dead
        #: epoch's pump sits in `get(timeout=REPORT_SECONDS)`, and for up to that long
        #: after a reconnect it woke on the NEXT frame — a live AttemptTerminal — saw the
        #: epoch had moved, and dropped it. The terminal was recorded, so it replayed on
        #: the following claim and the replay law held; live delivery on the stream that
        #: dispatched it did not.
        self.outbound: queue.Queue[pb.WorkerFrame | None] = queue.Queue()
        self.watches = WatchFanout(PROGRESS_QUEUE)
        # Slow source/custom-wheel and model-document acquisition never runs on a control
        # stream thread. One lane serializes revisions; the revision check after every slow
        # leg discards a superseded result before it can move serving state.
        self.convergences: queue.Queue[int | None] = queue.Queue()
        self.supervision.on_invalidate = self.capacity_dropped
        self.supervision.on_exit = self.executor_exited
        self.supervision.on_lane_failure = self.fail_machine
        #: The `control_stream_epoch` of every accepted RecordOwner stream being served right
        #: now: an owner holding this pod, the fact the supervisor's release decision rests on.
        self.attached_streams: set[int] = set()
        #: What this worker is doing, for the pod supervisor that outlives it. Absent off a
        #: pod: there is no second process there to publish to. (`self.activity` above is
        #: the narration lane the ObservedWorkerState tick carries — a different thing under
        #: a near name, and neither reads the other.)
        self.activity_file = (
            activity.ActivityFile(options.activity_path, owner=options.activity_owner)
            if options.activity_path is not None
            else None
        )
        self.exit_code: int = int(Exit.ok)
        self.boot_at = time.perf_counter()
        self.job_caps = pb.ResourceCaps()
        self.boot_fatal: pb.BootFailure | None = None
        # Same-boot verification receipts. A worker restart intentionally loses this table
        # and re-hashes every retained cache file once before trusting it again.
        self.verified_artifacts = acquire.VerifiedArtifactCache()
        #: The manifests the verified store holds complete (cr-080): rebuilt at boot from
        #: the store, kept by the acquirer's model leg, reported on every state and in the
        #: snapshot. A change is a lane-grade fact and emits a state like any other.
        self.held_manifests = acquire.HeldManifests(on_change=self._residency_changed)
        #: serializes claim acceptance: one fence, whichever host thread observes the claim
        self.claim_lock = threading.Lock()

    def _slot_executor_uid(self, slot: str) -> int:
        """The slot's own executor uid: deterministic by slot key, probed past in-boot
        collisions, in a range (64000..64511) disjoint from the pod seat uids (65533
        executor, 65532 media) and system users; setuid and chown need no passwd row."""

        base = self.options.executor_uid
        if base < 0:
            return base
        held = self._slot_uids.get(slot)
        if held is not None:
            return held
        span = 512
        start = int(hashlib.sha256(slot.encode()).hexdigest()[:8], 16) % span
        used = set(self._slot_uids.values())
        for probe in range(span):
            uid = 64000 + (start + probe) % span
            if uid not in used:
                self._slot_uids[slot] = uid
                return uid
        raise RuntimeError("executor slot uid range exhausted")

    def _new_placement_supervision(
        self, placement_id: str, *, uid_key: str = ""
    ) -> ExecutorSupervision:
        """Create the one executor slot owned by an additional package placement. `uid_key`
        names whose executor uid it runs as, when that is not the placement's own."""

        key = hashlib.sha256(placement_id.encode()).hexdigest()[:24]
        supervision = ExecutorSupervision(
            root=self.root / "placements" / key,
            python=self.options.python or sys.executable,
            base_env=self.config.child_base_env,
            cozy_home=self.config.cozy_home,
            executor_uid=self._slot_executor_uid(uid_key or placement_id),
            executor_gid=self.options.executor_gid,
            jit_pod_scope=f"{self.fence.worker_boot_id}-{key}",
            kernel_cache=self.config.kernel_cache,
        )
        supervision.on_invalidate = lambda why: self._hosted_capacity_dropped(placement_id, why)
        supervision.on_exit = lambda executor: self._hosted_executor_exited(placement_id, executor)
        supervision.on_change = lambda executor: self._row_changed(placement_id, executor)
        supervision.on_lane_failure = self.fail_machine
        killed = supervision.sweep_orphans()
        if killed:
            self.note(
                "recovery",
                f"{placement_id!r}: reclaimed {len(killed)} prior executor(s)",
            )
        return supervision

    def _placement_by_id(self, placement_id: str) -> Placement | None:
        if self.placement is not None and self.placement.placement_id == placement_id:
            return self.placement
        hosted = self.hosted.get(placement_id)
        return hosted.placement if hosted is not None else None

    def _placement_dispatchable(self, placement_id: str) -> bool:
        placement = self._placement_by_id(placement_id)
        return bool(
            placement is not None
            and placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
        )

    def _all_placements_dispatchable(self) -> bool:
        return bool(
            self.placement is not None
            and self.pending is None
            and not self.pending_hosted
            and self._placement_dispatchable(self.placement.placement_id)
            and all(
                hosted.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
                for hosted in self.hosted.values()
            )
        )

    # ------------------------------------------------------------------ lanes

    def _primary_lane(self) -> DeviceLane:
        """The lane the PRIMARY slot's executor (`self.supervision`) is sealed to.

        A job takes the envelope lane; a serving placement takes the lane the worker
        assigned it; before either exists (package describe, boot) it is the first lane.
        """
        if self.accepted.mode == "job":
            return self.lanes.envelope
        if self.placement is not None and self.placement.lane_id:
            return self.lanes.by_id[self.placement.lane_id]
        return self.lanes.lanes[0]

    def _primary_slot(self) -> str:
        """The primary slot's row key on its lane: the placement id, or "" for a job."""
        if self.accepted.mode != "job" and self.placement is not None:
            return self.placement.placement_id
        return ""

    def _primary_row(self) -> Ledger:
        return self._primary_lane().row(self._primary_slot()).ledger

    def primary_ledger(self) -> Ledger:
        """The primary slot's ledger row, for the one-shot door's result document."""
        return self._primary_row()

    def _lane_of(self, placement_id: str) -> DeviceLane:
        """The lane a HOSTED placement was assigned; a hosted placement is never unassigned
        by the time anything asks (assignment precedes activation)."""
        lane = self.lanes.lane_of(placement_id)
        if lane is None:
            raise RuntimeError(f"placement {placement_id!r} has no device lane")
        return lane

    def _assign_lane(
        self,
        placement: Placement,
        bindings: Mapping[str, DeclaredBinding],
        supervision: ExecutorSupervision,
    ) -> bool:
        """Bind one placement to a lane, or latch a typed lane refusal.

        A machine replica runs on exactly its scheduler grant. A placement the owner sent
        runs at the largest degree its bindings declare within its pin (or the envelope),
        chosen by measured fit. The row remembers `supervision`, the slot the lane asks to
        vacate when a co-tenant needs the card (cr-022).
        """
        model_bearing = any(not binding.weightless for binding in bindings.values())
        # A placement executes one entrypoint at a time. Each binding already sums
        # its independent model slots; other entrypoints are alternative calls.
        declared = max((binding.logical_weight_bytes for binding in bindings.values()), default=0)
        measured: dict[int, accel.DeviceMemory] = {}
        if model_bearing:
            kind = accel.host_backend_family()
            for ordinal, entry in enumerate(self.lanes.entries):
                measured[ordinal] = accel.device_memory(entry, kind)
        pin = placement.device_pin
        try:
            if placement.placement_id.startswith(machine_lanes.REPLICA):
                ordinals = pin or ()  # its scheduler grant exactly; none for weightless work
            else:
                candidates = pin or tuple(range(len(self.lanes.entries)))
                degrees = [
                    b.sequence_parallel_degrees() for b in bindings.values() if not b.weightless
                ]
                ordinals = self.lanes.choose(
                    candidates,
                    width_for(degrees, len(candidates)) if model_bearing else 1,
                    model_bearing=model_bearing,
                    declared_bytes=declared,
                    measured=measured,
                )
            lane = self.lanes.bind(
                placement.placement_id, ordinals, model_bearing=model_bearing, measured=measured
            )
        except LaneRefusal as exc:
            placement.lane_id = ""
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    subject=placement.placement_id,
                    reason=exc.code,
                    detail=safe(exc.detail)[:1024],
                )
            )
            self.note("materialization", f"{placement.placement_id!r} refused: {exc}"[:256])
            return False
        placement.lane_id = lane.lane_id
        row = lane.row(placement.placement_id)
        row.model_bearing = model_bearing
        row.supervision = supervision
        row.ledger.observe_devices(lane.device_facts(measured))
        tenants = sorted(
            slot
            for slot, _lane, _row in self.memory.tenants(set(lane.ordinals))
            if slot != placement.placement_id
        )
        self.note(
            "materialization",
            f"{placement.placement_id!r} assigned to {lane.lane_id} "
            f"(CUDA_VISIBLE_DEVICES={lane.devices!r}; "
            + ("model-bearing" if model_bearing else "weightless")
            + (f"; a {lane.degree}-device GROUP, one seat" if lane.group else "")
            + (f"; shares its devices with {tenants} by residency arbitration" if tenants else "")
            + ")",
        )
        return True

    def _release_lane(self, placement_id: str) -> None:
        placement = self._placement_by_id(placement_id)
        if placement is not None:
            placement.lane_id = ""
        self.lanes.release(placement_id)

    def lane_documents(self) -> list[dict[str, Any]]:
        """`lanes[]` (proto-024): every device or group lane of the envelope with its seats
        now, sorted by lane_id; `lane_wire.emit_lanes` writes it, the records carry it."""
        return sorted(
            (lane_wire.lane_document(lane) for lane in self.lanes.lanes),
            key=lambda row: str(row["lane_id"]),
        )

    def _bind_attempt_slot(self, attempt: AttemptRecord) -> AttemptSlot:
        """Capture the placement slot an attempt is admitted against — once, under the lock.

        Nothing on the engine is re-pointed any more: lanes admit concurrently, and the
        slot is what makes an attempt on lane B unable to observe lane A's placement. The
        ledger row is the placement's own on its lane; switching between two placements
        wipes nothing (cr-066's red arm: it used to `begin_generation` the one ledger).
        """
        selected_job = None
        bindings_digest = ""
        if attempt.spec.get("job") is not None or self.accepted.mode == "job":
            descriptor = (attempt.spec.get("job") or {}).get("job_descriptor_id", "")
            selected_job = self.job_slots.get(machine_slots.key(attempt.request_id)) or next(
                (
                    slot
                    for slot in self.job_slots.values()
                    if slot.binding.job_descriptor_id == descriptor
                ),
                None,
            )
            lane = selected_job.lane if selected_job is not None else self.lanes.envelope
            supervision = selected_job.supervision if selected_job is not None else self.supervision
            row = lane.row("")
            bindings: Mapping[str, DeclaredBinding] = {}
            failed: Mapping[str, str] = {}
        else:
            hosted = self.hosted.get(attempt.placement_id)
            if hosted is None:
                lane = self._primary_lane()
                supervision = self.supervision
                row = lane.row(self._primary_slot())
                bindings = self.bindings
                failed = self.failed_bindings
                if self.placement is not None:
                    bindings_digest = documents.spell(self.placement.bindings_digest)
            else:
                lane = self._lane_of(attempt.placement_id)
                supervision = hosted.supervision
                row = lane.row(attempt.placement_id)
                bindings = hosted.bindings
                failed = hosted.failed_bindings
                bindings_digest = documents.spell(hosted.placement.bindings_digest)
        if attempt.lane_id and attempt.lane_id != lane.lane_id:
            raise AttemptRefusal(
                "placement_lane_moved",
                f"the offer took a seat on {attempt.lane_id} and placement "
                f"{attempt.placement_id!r} is now on {lane.lane_id}; the RecordOwner "
                "re-dispatches against the current snapshot",
                cause=CAUSE.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE,
            )
        bindings_digest = self._offered_bindings_digest(attempt, bindings, bindings_digest)
        executor = supervision.current
        executor_pid = executor.pid if executor is not None else 0
        if row.ledger.executor_pid != executor_pid:
            # This ROW's generation moved under it (a respawn this row's `on_change` already
            # announced, or a slot that never saw one): re-key the row, touching no other.
            row.ledger.begin_generation(executor_pid)
        slot = AttemptSlot(
            supervision=supervision,
            ledger=row.ledger,
            chooser=row.chooser,
            bindings=bindings,
            failed_bindings=failed,
            lane_id=lane.lane_id,
            devices=lane.devices,
            job_device_count=(
                selected_job.directive.device_count
                if selected_job is not None
                else self.engine.job_device_count
            )
            if attempt.spec.get("job") is not None
            else 0,
            orchestration=not lane.ordinals,
            job_binding=selected_job.binding if selected_job is not None else None,
            job_resource_caps=documents.body(selected_job.directive.resource_caps)
            if selected_job is not None
            else documents.body(self.job_caps),
            bindings_digest=bindings_digest,
        )
        attempt.slot = slot
        attempt.lane_id = lane.lane_id
        return slot

    def _offered_bindings_digest(
        self, attempt: AttemptRecord, bindings: Mapping[str, DeclaredBinding], current: str
    ) -> str:
        """The placement bindings digest this attempt is admitted under.

        An in-place update changes the placement's bindings digest while offers made under
        the superseded document are still queued. Such an offer is admitted when its own
        binding survived the update over the very same construction: the bytes it pins are
        the bytes it will run, so the placement growing around it is no reason to refuse.
        """
        serving = attempt.spec.get("serving") or {}
        offered = str(serving.get("bindings_digest", ""))
        if not offered or offered == current:
            return current
        plan_id = str(serving.get("entrypoint_binding_digest", ""))
        superseded = self._superseded.get(attempt.placement_id, {}).get(offered, {})
        declared = bindings.get(plan_id)
        if declared is not None and superseded.get(plan_id) == declared.construction_key():
            return offered
        return current

    def _row_changed(self, placement_id: str, executor: child.Executor | None) -> None:
        """A HOSTED placement's executor was published or cleared: move ITS row only."""
        lane = self.lanes.lane_of(placement_id)
        if lane is not None:
            lane.row(placement_id).ledger.begin_generation(executor.pid if executor else 0)

    def _lane_dispatchable(self, lane: DeviceLane) -> bool:
        """Whether THIS lane has a placement that takes offers (job mode: the envelope)."""
        if lane.lane_id.startswith("cpu-") and (slot := self.job_slots.get(lane.lane_id)):
            return slot.ready()
        if self.accepted.mode == "job":
            slot = self.job_slots.get(lane.lane_id)
            return (
                slot.ready() if slot is not None else lane is self.lanes.envelope and self.job_ready
            )
        return any(self._placement_dispatchable(placement_id) for placement_id in lane.placements)

    def _hosted_set_serving(self, placement_id: str, state: pb.ServingState, why: str = "") -> None:
        hosted = self.hosted.get(placement_id)
        if hosted is None or hosted.placement.serving == state:
            return
        hosted.placement.serving = state
        self.bump_admission(why or f"{placement_id} serving -> {pb.ServingState.Name(state)}")
        self._settle()

    def _hosted_capacity_dropped(self, placement_id: str, why: str) -> None:
        if not self._placement_dispatchable(placement_id):
            return
        self.note("serving", f"{placement_id} capacity withdrawn: {why[:180]}")
        self._hosted_set_serving(
            placement_id,
            pb.ServingState.SERVING_STATE_ACTIVATING,
            f"{placement_id} executor invalidated",
        )

    def _hosted_executor_exited(self, placement_id: str, executor: child.Executor) -> None:
        with self.control_lock:
            hosted = self.hosted.get(placement_id)
            if hosted is None:
                return
            rebuild = self._placement_dispatchable(placement_id)
            if not hosted.supervision.invalidate(
                executor,
                f"executor epoch {executor.epoch} pid {executor.pid} exited",
            ):
                return
        if self.stop.is_set():
            return
        if rebuild:
            self._executor_died(self._lane_of(placement_id), executor, placement_id)
            return
        # Withdrawn: nothing rebuilds it, and its process is reaped and proved gone now.
        self.supervisor.spawn(
            f"reap:{placement_id}:{executor.epoch}",
            lambda _: self._reclaim_dead(placement_id, executor, "exited after withdrawal"),
        )

    # ------------------------------------------------------------------ plumbing

    def prepare_package_set(
        self, request: pb.PreparePackageSetRequest
    ) -> pb.PreparePackageSetResult:
        return self._prepare_package(request, local=False)

    def prepare_local_package(
        self, request: pb.PrepareLocalPackageRequest
    ) -> pb.PreparePackageSetResult:
        return self._prepare_package(request, local=True)

    def _register_local_package(
        self,
        request: pb.PrepareLocalPackageRequest,
        result: pb.PreparePackageSetResult,
        installed: package_installation.InstalledEnvironment,
    ) -> pb.PreparePackageSetResult:
        from cozy_runtime.internal.worker import package_prepare

        prepared = read_placement_set(result.placement_set)
        if len(prepared) != 1 or not prepared[0].HasField("development"):
            raise package_prepare.PreparationRefusal(
                "local_package_result_invalid",
                "local preparation did not return one development Placement",
            )
        stored = self._placement_from_entry(prepared[0], result.placement_set.placement_set_digest)
        stored.installed = installed
        revision = stored.installation_id
        if revision != request.package.installation_id:
            raise package_prepare.PreparationRefusal(
                "local_package_installation_invalid",
                "prepared Placement changed the installation ID",
            )
        self.local_operations[(request.operation_id, revision)] = stored
        self.prepared_installations[stored.prepared_key] = stored
        return result

    def prepare_unpublished_placement(
        self, request: pb.PreparePrivatePlacementRequest
    ) -> pb.PreparePackageSetResult:
        from cozy_runtime.internal.worker import package_prepare

        with self.preparation_lock:
            revision = request.installation_id
            prepared = self.local_operations.get((request.operation_id, revision))
            if prepared is None:
                raise package_prepare.PreparationRefusal("local_revision_unprepared", revision)

            from . import machine_materialization

            machine_materialization.ensure(
                self, request.download_delegation, native=request.native_models
            )

            def mark_verified(digest: str, length: int, path: Path) -> None:
                self.verified_artifacts.record(digest, length, path)

            with contextlib.ExitStack() as access:
                if request.native_models or request.model_choices:
                    if self.workspace is None:
                        raise package_prepare.PreparationRefusal(
                            "private_workspace_absent",
                            "native model preparation needs its retained workspace",
                        )
                    owner = self.authorize_workspace(request.claim)

                    def guard() -> None:
                        self.authorize_workspace(request.claim)

                    access.enter_context(self.workspace.authorized(guard))
                    for model in request.native_models:
                        access.enter_context(
                            self.workspace.held_model(owner, model.retention, model.manifest)
                        )
                overrides = None
                if request.model_choices:
                    from . import machine_model_overrides

                    def check_model_preparation() -> None:
                        self.authorize_workspace(request.claim)
                        if self.stop.is_set():
                            raise WorkspaceRefusal("machine stopped during model preparation")

                    overrides = machine_model_overrides.private(
                        self, request, prepared.document, check=check_model_preparation
                    )
                result = package_prepare.prepare_unpublished_placement(
                    request,
                    prepared=prepared.document,
                    installed=prepared.installed,
                    artifact_cache=Path(self.options.artifact_cache or ""),
                    tensorfs_root=(
                        Path(self.options.tensorfs_root)
                        if self.options.tensorfs_root is not None
                        else None
                    ),
                    verified=mark_verified,
                    model_overrides=overrides,
                )
                if request.native_models or request.model_choices:
                    self.authorize_workspace(request.claim)
            joined = read_placement_set(result.placement_set)
            if len(joined) != 1 or joined[0].installation_id != revision:
                raise package_prepare.PreparationRefusal(
                    "private_placement_result_invalid", revision
                )
            # A captured root names this model-bound placement, not the code-only one.
            stored = self._placement_from_entry(
                joined[0], result.placement_set.placement_set_digest
            )
            stored.installed = prepared.installed
            self.prepared_installations[stored.prepared_key] = stored
            return result

    def _prepare_package(
        self,
        request: pb.PreparePackageSetRequest | pb.PrepareLocalPackageRequest,
        *,
        local: bool,
    ) -> pb.PreparePackageSetResult:
        """Derive one local PlacementSet while package imports stay in the isolated executor."""

        with self.preparation_lock:
            self.compute_idle_since = time.monotonic()
            # Publish stack warnings on the durable activity lane consumed by Creator,
            # in addition to the installer's process-local diagnostics.
            raw_requirements = (
                request.dependency_requirements
                if isinstance(request, pb.PrepareLocalPackageRequest)
                else request.locked_requirements
            )
            if raw_requirements:
                with contextlib.suppress(package_environment.EnvironmentRefusal):
                    selected = package_environment.read_locked_requirements(bytes(raw_requirements))
                    for warning in package_environment.accelerator_warnings(
                        {row.name: row.version for row in selected.rows}, self.base
                    ):
                        self.note("warning", warning)
            if not local:
                assert isinstance(request, pb.PreparePackageSetRequest)
                return self._prepare_materialized_package(request)
            assert isinstance(request, pb.PrepareLocalPackageRequest)
            return self._prepare_local_in(request)

    def _prepare_materialized_package(
        self, request: pb.PreparePackageSetRequest
    ) -> pb.PreparePackageSetResult:
        """Runtime owns model acquisition, including the models-only preparation lane."""
        from concurrent.futures import ThreadPoolExecutor

        from . import machine_materialization

        selected = documents.parse(request.download_delegation, pb.DownloadDelegation)
        # Older Hosts may still ask for a code-only warm phase before their own transfer.
        if request.models_landing:
            return self._prepare_published(request)
        if not selected.packages:
            if request.application or request.locked_requirements:
                raise WorkspaceRefusal("models-only preparation cannot name package requirements")
            machine_materialization.ensure(self, request.download_delegation, hub=request.hub)
            raw, digest = documents.identity(pb.PlacementSet())
            return pb.PreparePackageSetResult(
                placement_set=pb.DesiredPlacementSet(
                    placement_set_canonical_bytes=raw, placement_set_digest=digest
                )
            )
        if selected.models:
            landing = pb.PreparePackageSetRequest()
            landing.CopyFrom(request)
            landing.models_landing = True
            # Package installation/prespawn overlaps disk transfer, as on either deployment.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="package-prepare") as pool:
                installed = pool.submit(self._prepare_published, landing)
                machine_materialization.ensure(self, request.download_delegation, hub=request.hub)
                installed.result()
        return self._prepare_published(request)

    def _execution_publication_authority(
        self, owner: str, request: str
    ) -> PublicationAuthority | None:
        assert self.executions is not None
        root = self.executions.capture_root(owner, request)
        return machine_model_resolve.registration(self, self.executions.prepared(owner, root).hub)

    def _installed_call_bindings(self, attempt: AttemptRecord) -> list[dict[str, Any]]:
        owner = self.fence.record_owner_id
        if (
            self.executions is None
            or attempt.job is None
            or not self.executions.owns(owner, attempt.request_id)
        ):
            return []
        root = self.executions.capture_root(owner, attempt.request_id)
        capture = self.executions.capture(owner, root)
        prepared = self.executions.preparation(owner, root)["installations"]
        from cozy_runtime.internal import package_interface

        from . import machine_child_target, machine_deferred

        deferred = machine_deferred.rows(capture)
        callers = {
            attempt.job.installation_id,
            machine_child_target._deferred_key(self, attempt.job.installation_id),
        }
        rows = []
        described_modules: set[tuple[str, str]] = set()
        for binding in capture.get("bindings", []):
            if (
                binding.get("caller_installation_id") or binding.get("caller_deferred_key")
            ) not in callers:
                continue
            identifier = binding.get("callee_installation_id") or binding.get(
                "callee_deferred_key", ""
            )
            if identifier in deferred:
                body = machine_deferred.interface(deferred[identifier])
            else:
                placement = prepared[identifier]["placement"]
                raw = base64.b64decode(placement["package_interface"], validate=True)
                body = package_interface.read_bytes(raw)
            kind = (
                "job"
                if any(row["name"] == binding["entrypoint"] for row in body["jobs"])
                else "entrypoint"
            )
            row: dict[str, Any] = {
                "module": binding["module"],
                "export": binding["export"],
                "kind": kind,
            }
            if identifier == attempt.job.installation_id:
                row["self"] = True
            elif (identifier, binding["module"]) not in described_modules:
                # Private source projects have ordinary Python implementations,
                # including synchronous serving functions. Supply the installed
                # schema so the caller executor can expose managed imports before
                # importing the workflow, without rewriting either source tree.
                # A package can export many functions across several modules.
                # Transport each module's schemas once, not the whole package for
                # every export, so ordinary dependency graphs fit the control seam.
                described_modules.add((identifier, binding["module"]))
                row["interface_document"] = {
                    **body,
                    **{
                        collection: [
                            entry
                            for entry in body[collection]
                            if entry.get("invocable", {}).get("module") == binding["module"]
                        ]
                        for collection in ("jobs", "entrypoints")
                    },
                }
            rows.append(row)
        return rows

    def _new_preparation_supervision(self) -> ExecutorSupervision:
        """The preparation-scoped executor slot for an installed package's describe. Package
        describe needs an isolated executor (package code never gets worker authority), and
        borrowing the primary slot while placements serve would replace a serving executor
        with the prepared package's environment. Modern published sets carry their interface."""

        # Keep preparation's socket shorter than the boot-validated primary
        # socket, independent of placement and operation identifiers. Its state
        # stays in the original directory so orphan ownership still reconciles.
        socket_path = self.root / "p.sock"
        if refusal := socket_path_refusal(str(socket_path)):
            raise ExecutorGone(refusal)
        supervision = ExecutorSupervision(
            root=self.root / "placements" / "prepare",
            socket_path=socket_path,
            python=self.options.python or sys.executable,
            base_env=self.config.child_base_env,
            cozy_home=self.config.cozy_home,
            executor_uid=self._slot_executor_uid("prepare"),
            executor_gid=self.options.executor_gid,
            jit_pod_scope=f"{self.fence.worker_boot_id}-prepare",
            kernel_cache=self.config.kernel_cache,
        )
        supervision.on_lane_failure = self.fail_machine
        killed = supervision.sweep_orphans()
        if killed:
            self.note("recovery", f"preparation: reclaimed {len(killed)} prior executor(s)")
        return supervision

    def _describe_installed(
        self,
        installed: package_installation.InstalledEnvironment,
        distribution: str,
        *,
        application: str = "",
    ) -> bytes:
        """Describe an owned environment incarnation once, in an isolated executor.

        Installation validation/refresh precedes this call. Adopted mutable environments
        have no incarnation and must be rediscovered. The record is replaced on every
        rebuild, including repairs and SDK rollbacks to a previously used venv path.
        """
        from cozy_runtime.internal import package_interface

        key = (
            (
                installed.installation_id,
                installed.incarnation,
                installed.generation,
                distribution,
                application,
            )
            if installed.incarnation is not None
            else None
        )
        with self.preparation_slot:
            if key is not None and key in self._installed_interfaces:
                return self._installed_interfaces[key]
            raw = self._describe_in_slot(installed, distribution, application=application)
            package_interface.parse(raw, "installed package interface")
            if key is not None:
                self._installed_interfaces[key] = raw
            return raw

    def _describe_in_slot(
        self,
        installed: package_installation.InstalledEnvironment,
        distribution: str,
        *,
        application: str,
    ) -> bytes:
        from cozy_runtime.internal.worker import package_prepare

        supervision = self._new_preparation_supervision()
        try:
            supervision.use_environment(str(installed.python), installed.installation_id)
            lane = self._primary_lane()
            try:
                executor = supervision.spawn(imposed=self.imposed(lane))
                output = supervision.handoff(executor.epoch, INTERFACE_DOCUMENT)
                # A cold import, torch included: bound by the executor's own measured
                # work and its death, never by a number of seconds (xs-007 row 8).
                with executor.watched("describe_installed"):
                    reply = executor.call(
                        DescribeInstalled(
                            distribution=distribution,
                            application=application,
                            output=str(output),
                        ),
                        timeout=None,
                    )
            except ExecutorProtocolMismatch as exc:
                raise package_prepare.PreparationRefusal(
                    "package_prepare_executor_failed", exc.detail
                ) from exc
            except ExecutorGone as exc:
                # A dead or silent describe executor is this pod's condition, not a verdict
                # on the package: answer it without recording a permanent refusal.
                raise _InterfaceFault(f"package_prepare_executor_failed: {exc.detail}") from exc
            if not reply.get("ok"):
                # THE EXECUTOR ALREADY DECIDED THIS AND WE WERE THROWING IT AWAY. Its
                # reply carries `terminal` and `origin` beside `code` and `detail`
                # (executor.classify), and reading only two of the four collapsed a
                # package's ConformanceError, a device OOM, and a PermissionError raised by
                # the runtime's own plumbing into one terminal refusal. The origin matters
                # as much as the verdict: `unhandled_exception` defaults to blaming the
                # AUTHOR, so an infrastructure fault sent an operator to debug the package.
                raise _interface_failure(reply)
            try:
                raw = output.read_bytes()
            except OSError as exc:
                # An absent output after an `ok` reply is a filesystem fact about this pod,
                # not a statement about the request, and the errno is the whole diagnosis.
                raise _InterfaceFault(
                    f"package-interface output is absent: {type(exc).__name__}: {exc}"
                ) from exc
            finally:
                output.unlink(missing_ok=True)
            return raw
        finally:
            supervision.close()

    def _prepare_local_in(
        self, request: pb.PrepareLocalPackageRequest
    ) -> pb.PreparePackageSetResult:
        from cozy_runtime.internal.worker import package_prepare

        def mark_verified(digest: str, length: int, path: Path) -> None:
            self.verified_artifacts.record(digest, length, path)

        authority = machine_model_resolve.registration(
            self, machine_model_resolve.hub_key(self, request.hub)
        )
        materialized: list[package_installation.InstalledEnvironment] = []
        result = package_prepare.prepare_local_package(
            request,
            python_root=self.config.managed_python_root,
            cancel=self.stop.is_set,
            progress=lambda message: self.note("python", message),
            artifact_cache=Path(self.options.artifact_cache or ""),
            install_root=Path(self.options.install_root or ""),
            python=Path(self.options.python or sys.executable),
            describe=self._describe_installed,
            verified=mark_verified,
            job_plan_root=self.config.cozy_home / "job-plans",
            base=self.base,
            materialized=materialized.append,
            dependency_python=self.options.environment_python,
            dependency_cache=self.config.dependency_cache,
            hub=package_installation.HubIndex(authority.origin, authority.ca)
            if authority is not None
            else None,
        )
        installed = materialized[0]
        return self._register_local_package(request, result, installed)

    def _prepare_published(
        self, request: pb.PreparePackageSetRequest
    ) -> pb.PreparePackageSetResult:
        """Install a release and retain its worker-discovered callable interface."""

        from cozy_runtime.internal.worker import package_prepare

        def mark_verified(digest: str, length: int, path: Path) -> None:
            self.verified_artifacts.record(digest, length, path)

        published: list[package_installation.InstalledEnvironment] = []
        result = package_prepare.prepare_package_set(
            machine_release_catalog.complete(self, request),
            describe=self._describe_installed,
            python_root=self.config.managed_python_root,
            cancel=self.stop.is_set,
            progress=lambda message: self.note("python", message),
            artifact_cache=Path(self.options.artifact_cache or ""),
            tensorfs_root=Path(self.options.tensorfs_root or ""),
            install_root=Path(self.options.install_root or ""),
            python=Path(self.options.python or sys.executable),
            verified=mark_verified,
            job_plan_root=self.config.cozy_home / "job-plans",
            base=self.base,
            preinstalled_python=Path(self.options.environment_python)
            if self.options.environment_python is not None
            else None,
            materialized=published.append,
            dependency_cache=self.config.dependency_cache,
        )
        entries = read_placement_set(result.placement_set)
        if len(entries) != 1 or not entries[0].HasField("package") or len(published) != 1:
            raise package_prepare.PreparationRefusal(
                "published_package_result_invalid", "published preparation needs one exact revision"
            )
        if request.models_landing:
            # The Host is still landing the weights (h3a-089). Nothing read a model, so no
            # placement is retained; the installation starts its executor meanwhile.
            self._collect_superseded_installations(published[0])
            interface = bytes(result.installed_package.package_interface)
            self.prespawns.request_landing(
                published[0].installation_id,
                interface,
                package_prepare.landing_models(request, interface),
            )
            return pb.PreparePackageSetResult(installed_package=result.installed_package)
        stored = self._placement_from_entry(entries[0], result.placement_set.placement_set_digest)
        stored.installed = published[0]
        self.prepared_installations[stored.prepared_key] = stored
        self._collect_superseded_installations(published[0])
        return result

    def _referenced_installations(self) -> set[str]:
        """Every installation a live placement, job, executor or retained execution uses."""
        placements = [self.placement, self.pending, *(h.placement for h in self.hosted.values())]
        referenced = {p.installation_id for p in placements if p is not None}
        referenced |= {binding.installation_id for binding in self.jobs.values()}
        referenced |= {slot.binding.installation_id for slot in self.job_slots.values()}
        supervisions = [
            self.supervision,
            *(hosted.supervision for hosted in self.hosted.values()),
            *(slot.supervision for slot in self.job_slots.values()),
        ]
        referenced |= {s.environment_installation_id for s in supervisions}
        if self.executions is not None:
            referenced |= self.executions.retained_installations()
        return referenced

    def _collect_superseded_installations(
        self, current: package_installation.InstalledEnvironment
    ) -> None:
        """Remove this package's other published environments that nothing live references.

        Only content-keyed published installations are collected; a development
        installation or one a placement, job, executor or retained execution names stays.
        """
        root = Path(self.options.install_root or "")
        prefix = package_installation.PUBLISHED_PREFIX
        referenced = self._referenced_installations()
        for directory in sorted((root / "installations").glob(prefix + "*")):
            identifier = directory.name
            if identifier == current.installation_id or identifier in referenced:
                continue
            try:
                installed = package_installation.open_installation(root, identifier)
            except package_environment.EnvironmentRefusal:
                continue
            if installed.package != current.package:
                continue
            self.prespawns.forget(identifier)
            machine_slots.forget(self, identifier)
            package_installation.remove(root, identifier)
            shutil.rmtree(self.config.cozy_home / "job-plans" / identifier, ignore_errors=True)
            for key in [key for key in self.prepared_installations if key[1] == identifier]:
                del self.prepared_installations[key]
            for described in [key for key in self._installed_interfaces if key[0] == identifier]:
                del self._installed_interfaces[described]
            self.note("installation", f"collected superseded {installed.package} {identifier}")

    def host_facts(self) -> hostfacts.HostFacts:
        """This host's accelerator facts, measured once rather than one `nvidia-smi` per call.

        The inventory is fixed for a boot. An unreadable accelerator is measured again at its
        next use, and an executor death (the one device event a worker sees) forgets them.
        """
        facts = self._host_facts
        if facts is None:
            facts = hostfacts.measure(self.options.accelerator_backend)
            if "gpu_name" not in facts.unreadable:
                self._host_facts = facts
        return facts

    def holds_construction(self, binding: str) -> bool:
        """A live executor has built this exact entrypoint binding: its models are resident."""
        supervisions = [self.supervision, *(hosted.supervision for hosted in self.hosted.values())]
        return any(
            s.current is not None and s.current.alive() and binding in s.current.loaded_bindings()
            for s in supervisions
        )

    def numerical_environment(self) -> bytes:
        from cozy_runtime.internal.numerical_environment import fingerprint

        return fingerprint(
            self.host_facts(),
            threads=self.options.threads,
            inherited=dict(self.config.child_base_env),
        )

    def authorize_workspace(self, claim: pb.Claim) -> str:
        from cozy_runtime.internal.worker.workspace import WorkspaceRefusal

        with self.claim_lock:
            if self.options.sole_supervisor and (
                not self.claim_ready.is_set() or self.stop.is_set()
            ):
                raise WorkspaceRefusal("machine coordinator is not accepting calls")
            if (
                not self.fence.record_owner_recorded
                or claim.record_owner_id != self.fence.record_owner_id
                or claim.record_owner_epoch != self.fence.record_owner_epoch
                or claim.control_stream_epoch != 0
            ):
                raise WorkspaceRefusal(
                    "workspace call does not name the current authenticated owner"
                )
            if claim.wire_minor < MIN_COMPATIBLE_WIRE_MINOR:
                raise WorkspaceRefusal(
                    f"{CAPABILITY_UNAVAILABLE_CODE}: "
                    + _capability_unavailable("workspace calls", claim.wire_minor)
                )
            if (
                claim.worker_boot_id != self.fence.worker_boot_id
                or claim.worker_id != self.options.worker_id
            ):
                raise WorkspaceRefusal("workspace proof names another worker lifetime")
            # The transcript is (epoch, boot, worker, certificate); all but the epoch are
            # this process's own, so one verified signature stands for the epoch.
            proven = (claim.record_owner_epoch, bytes(claim.proof))
            if proven != self._proven_claim:
                try:
                    downloads.verify_claim_proof(
                        claim,
                        self.config.record_owner_public_key,
                        worker_id=self.options.worker_id,
                        worker_boot_id=self.fence.worker_boot_id,
                        worker_tls_certificate_digest=self.options.worker_tls_certificate_digest,
                    )
                except downloads.DownloadRefusal as exc:
                    raise WorkspaceRefusal("workspace owner proof was refused") from exc
                self._proven_claim = proven
            return self.fence.record_owner_id

    def _execution_service(self, claim: pb.Claim) -> tuple[workspace_executions.Executions, str]:
        owner = self.authorize_workspace(claim)
        if self.executions is None:
            raise ValueError("worker has no execution workspace")
        return self.executions, owner

    def _machine_prefetch(self, attempt: AttemptRecord, request: ModelPrefetch) -> Answer:
        assert self.machine_calls is not None
        return self.machine_calls.serving.prefetch(attempt, request)

    def _release_gpus(self, attempt: AttemptRecord) -> None:
        """`ctx.release_gpus()`: the attempt's root ends its GPU lease now."""
        if self.executions is not None:
            owner = self.fence.record_owner_id
            self.gpu.release_root(self.executions.scheduling_root(owner, attempt.request_id)[0])

    def _machine_call(
        self, attempt: AttemptRecord, request: pb.ChildCallRequest, action: str
    ) -> pb.ChildCallResult:
        assert self.machine_calls is not None
        return self.machine_calls.handle(attempt, request, action)

    def _journal_changed(self, request: str, kind: str) -> None:
        """A child's progress moved: the one parent awaiting it re-polls."""
        if kind == "progress" and self.machine_calls is not None:
            with self.machine_calls.lock:
                called = self.machine_calls.requests.get(request)
            if called is not None:
                self.calls.wake(called.parent_request_id)

    def execution(self, owner: str, request: str) -> None:
        """Start the unit that owns this execution, or wake the one that does."""
        if self.machine_calls is None or self.stop.is_set():
            return
        unit = execution_unit.Execution(self, self.machine_calls, owner, request)
        self.supervisor.spawn(execution_unit.key(request), unit.run, unit.crashed)

    def _effect_settled(self, owner: str, call: Call) -> None:
        """An effect settled: its parent re-reads it and its root may now release."""
        self.calls.wake(call.parent_request)
        if self.executions is not None:
            self.execution(owner, self.executions.scheduling_root(owner, call.parent_request)[0])

    def control_tree(
        self, owner: str, request: str, action: Literal["pause", "cancel"], generation: int = 0
    ) -> None:
        """Stop `request` and every open descendant now: each is controlled durably, its
        held attempt is signalled, and its unit is woken to finish the stop itself."""
        assert self.executions is not None and self.machine_calls is not None
        pending = [request]
        while pending:
            current = pending.pop()
            row = self.executions.row(owner, current)
            if row is None:  # a call not yet submitted: its unit sees its parent stopped
                self.supervisor.poke(execution_unit.key(current))
                continue
            # A running or queued execution stops; a canceled tree releases what even its
            # finished executions retain.
            if row.state in ("queued", "running") or (
                action == "cancel" and row.desired != "cancel"
            ):
                with contextlib.suppress(workspace_executions.StaleExecutionGeneration):
                    self.executions.control(
                        owner,
                        current,
                        f"tree-{action}.{request}.{generation}.{row.generation}",
                        row.generation,
                        action,
                    )
            held = self.engine.live.get(current)
            if held is not None and held.attempt == row.ordinal:
                self.cancel(
                    pb.CancelAttempt(
                        request_id=current,
                        attempt_ordinal=held.attempt,
                        invocation_spec_digest=held.digest,
                        reason=pb.CANCEL_REASON_CLIENT,
                    )
                )
                if self.source_calls is not None:
                    self.source_calls.cancel_parent(current, ordinal=held.attempt, spec=held.digest)
            self.execution(owner, current)
            pending.extend(
                call.child_request for call in self.machine_calls.journal.children(owner, current)
            )

    def _dispatched(self, attempt: AttemptRecord) -> None:
        """Execution STARTS, journaled just before the executor gets its command: a terminal
        recorded first (a stop that won) keeps the executor from ever being called."""
        if self.executions is not None:
            self.executions.dispatched(
                self.fence.record_owner_id, attempt.request_id, attempt.attempt
            )

    def reconcile_executions(self) -> None:
        """BOOT, once per process: rebuild live work from the journal. A dead process's
        dispatches fail, its undispatched work runs once more, unread custody and terminal
        holds are released, and every owed execution gets its unit."""
        if (
            self.executions is None
            or self.machine_calls is None
            or not self.fence.record_owner_recorded
            or self._reconciled == self.fence.record_owner_id
        ):
            return
        owner = self._reconciled = self.fence.record_owner_id
        marker = self.root / FAILED_MARKER
        failed = ""
        with contextlib.suppress(OSError, ValueError):
            failed = str(json.loads(marker.read_text()).get("reason", "")) or "the worker failed"
        self.executions.workspace.recover(owner)
        owed = self.executions.owed(owner, self.options.worker_id)
        for request in owed:
            if failed:
                with contextlib.suppress(ExecutionChanged):
                    self.executions.fail(owner, request, safe(f"the worker failed: {failed}", 4096))
            self.executions.reconcile(
                owner, request, requeue="" if failed else self.fence.worker_boot_id
            )
        with contextlib.suppress(OSError):
            marker.unlink()
        self.machine_calls.release_stranded(owner)
        for request in self.executions.held(owner, self.options.worker_id):
            if not self.executions.release(owner, request, boot=True):
                self.releasing.add(request)  # an effect is in doubt: once it settles
        for request in owed:
            self.execution(owner, request)
        for request in self.machine_calls.journal.unsettled_effects(owner):
            self.execution(owner, request)

    def submit_execution(
        self,
        claim: pb.Claim,
        submission_id: str,
        capture_digest: bytes,
        offer: pb.AttemptOffer,
        *,
        expected_execution_workspace_id: str,
        capture_document: bytes = b"",
        preparation: bytes = b"",
        publication_authorization_id: str = "",
        arguments: dict[str, Any] | None = None,
        owner_memo: bool = False,
        source_credentials: Sequence[pb.SourceCredential] = (),
    ) -> workspace_executions.Receipt:
        """Authenticated root intake; accepted execution never depends on the caller's stream."""
        with self.control_lock:
            if self.stop.is_set():
                raise WorkspaceRefusal("worker is stopping; root admission is closed")
            executions, owner = self._execution_service(claim)
            executions.require_workspace(expected_execution_workspace_id)
            machine_lanes.admit(self, "", preparation, offer)
            if (
                publication_authorization_id
                and machine_model_resolve.registration(
                    self, workspace_executions.Preparation.read(preparation).hub
                )
                is None
            ):
                raise WorkspaceRefusal("worker has no independent publication authority configured")
            result_schema, memoize = b"", False
            spec = documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
            if "job" in spec:
                descriptor = spec["job"]["job_descriptor_id"]
                binding = self.engine.jobs.get(descriptor)
                if binding is None or binding.installation_id != spec["job"]["installation_id"]:
                    binding = self._job_plan(spec["job"]["installation_id"], descriptor)
                declaration = self.engine._job_declaration(binding, descriptor)
                result_schema = canonical.write(declaration["result"])
                memoize = declaration.get("invocable", {}).get("memoize", False) is True
            from . import machine_checkpoint_inputs

            with machine_checkpoint_inputs.admission(executions.workspace, owner, offer.request_id):
                if "job" in spec:
                    from . import machine_model_inputs

                    assert binding is not None
                    declared = set(self.engine._job_model_declarations(binding, descriptor))
                    machine_model_inputs.retain(executions.workspace, owner, offer, declared)
                    from . import machine_byte_inputs

                    machine_byte_inputs.retain_root(
                        executions.workspace,
                        owner,
                        offer,
                        declaration["request"],
                        arguments or {},
                    )
                elif any(
                    str(row.get("input_id", "")).startswith(grants.MODEL_PREFIX)
                    for row in spec.get("inputs", [])
                ):
                    # A root this machine resolved names its Models as catalog inputs.
                    catalog = grants.model_inputs(
                        grants.bind(spec, offer.grant, offer.invocation_spec_digest)
                    )
                    if catalog:
                        workspace = executions.workspace
                        machine_checkpoint_inputs.preflight(workspace, owner, offer, catalog)
                        for entry in catalog.values():
                            machine_checkpoint_inputs.retain(workspace, owner, offer, entry)
                receipt = executions.submit(
                    owner,
                    submission_id,
                    capture_digest,
                    offer,
                    expected_execution_workspace_id=expected_execution_workspace_id,
                    result_schema=result_schema,
                    worker_boot=self.fence.worker_boot_id,
                    memoize=memoize,
                    capture_document=capture_document,
                    preparation=preparation,
                    worker_id=self.options.worker_id,
                    publication_authorization_id=publication_authorization_id,
                    owner_memo=owner_memo,
                )
            self.machine_sources.hold(owner, offer.request_id, source_credentials)
            self.execution(owner, offer.request_id)
            self.publish_activity(strict=True)
            return receipt

    def execution_status(self, claim: pb.Claim, request_id: str) -> workspace_executions.State:
        executions, owner = self._execution_service(claim)
        return executions.status(owner, request_id)

    def execution_events(
        self, claim: pb.Claim, request_id: str, after: int = 0, limit: int = 256
    ) -> workspace_executions.EventPage:
        executions, owner = self._execution_service(claim)
        return executions.events(owner, request_id, after, limit)

    def control_execution(
        self,
        claim: pb.Claim,
        request_id: str,
        command_id: str,
        generation: int,
        action: Literal["pause", "resume", "cancel"],
    ) -> workspace_executions.State:
        if action not in ("pause", "resume", "cancel"):
            raise ValueError("unknown execution control")
        if action == "resume" and self.stop.is_set():
            raise WorkspaceRefusal("worker is stopping; execution resume is closed")
        executions, owner = self._execution_service(claim)
        prior = executions.status(owner, request_id)
        state = executions.control(
            owner,
            request_id,
            command_id,
            generation,
            action,
            worker_boot=self.fence.worker_boot_id,
        )
        if action != "resume":
            self.control_tree(owner, request_id, action, state.generation)
        elif state.attempt_ordinal > prior.attempt_ordinal and self.machine_calls is not None:
            self.machine_calls.timing.phase(
                owner, request_id, state.attempt_ordinal, "queued", fresh=True
            )
        self.execution(owner, request_id)
        self.publish_activity(strict=True)
        return state

    def reconcile_publication(
        self, claim: pb.Claim, request_id: str, reconciliation: pb.MachinePublicationReconciliation
    ) -> workspace_executions.State:
        """Settle one sent publication of this execution from its owner's Hub read."""
        executions, owner = self._execution_service(claim)
        if self.machine_effects is None or not executions.owns(owner, request_id):
            raise WorkspaceRefusal("execution holds no publication effects")
        self.machine_effects.reconcile(
            owner,
            request_id,
            reconciliation.call_index,
            reconciliation.http_status,
            reconciliation.finalization,
        )
        self.execution(owner, executions.scheduling_root(owner, request_id)[0])
        return executions.status(owner, request_id)

    def _gpu_observed(
        self,
        events: list[gpu_scheduler.Event],
        roots: dict[str, int],
        view: gpu_scheduler.Document,
        installations: dict[str, str],
    ) -> None:
        """One scheduling pass's observations, journaled on the roots they concern."""
        self.prespawns.reconcile(roots, view, installations)
        if self.executions is None:
            return
        owner = self.fence.record_owner_id
        for root, event, body in events:
            if event == "gpu.wait":
                body = {**body, **self._waiting_call(owner, str(body["key"]))}
            if event in ("gpu.grant", "gpu.release") and not body.get("gpus"):
                # `ordinals` index this worker's envelope; people name a GPU by nvidia-smi's
                # number. `ordinals` stays for readers that predate `gpus`.
                body = {**body, "gpus": [self.gpu_identity(o) for o in body["ordinals"]]}
            with contextlib.suppress(WorkspaceRefusal):
                self.executions.record(owner, root, event, body)
            if event == "gpu.wait":
                self._gpu_waiting(owner, root, body, view)

    def _record_executor(self, attempt: AttemptRecord) -> None:
        """The SDK the attempt's executor loaded, as it stated in its hello, on the execution's
        root: what ran it, which a package environment built against another Runtime shows."""
        executor = attempt.executor
        if self.executions is None or executor is None:
            return
        owner = self.fence.record_owner_id
        body = {
            "request": attempt.request_id,
            "attempt": attempt.attempt,
            "pid": executor.pid,
            "runtime_version": str(executor.hello.get("runtime_version", "")),
            "tensorfs_version": str(executor.hello.get("tensorfs_version", "")),
        }
        with contextlib.suppress(WorkspaceRefusal):
            root, _ = self.executions.scheduling_root(owner, attempt.request_id)
            self.executions.record(owner, root, "executor", body)

    def gpu_identity(self, ordinal: int) -> execution_evidence.GpuIdentity:
        """Envelope ordinal `ordinal` as the GPU a person names (`execution_evidence.identify`)."""
        entries = self.lanes.entries
        entry = entries[ordinal] if 0 <= ordinal < len(entries) else ""
        return execution_evidence.identify(entry, self.options.gpus)

    def readable_gpus(self) -> tuple[int, ...]:
        """Ordinals whose memory the driver answers; an unreadable GPU cannot fit anything."""
        family = accel.host_backend_family()
        return tuple(
            o
            for o, entry in enumerate(self.lanes.entries)
            if accel.device_memory(entry, family).state == "measured"
        )

    def gpu_status(self, request: str) -> pb.MachineExecutionGpu:
        """What the GPU scheduler holds for this execution and, for a root, its calls."""

        def on_device(key: str) -> bool:
            held = self.engine.live.get(key.rpartition("#")[0])
            return held is not None and held.state in ON_DEVICE

        return pb.MachineExecutionGpu(**self.gpu.status(request, on_device))

    def _warm(self, template: tuple[str, str]) -> tuple[int, ...]:
        """Ordinals where a replica of this template is resident, or where its installation's
        process was started beside the download, so a grant reuses it."""
        placements = [self.placement, *(hosted.placement for hosted in self.hosted.values())]
        return tuple(
            sorted(
                {
                    ordinal
                    for placement in placements
                    if placement is not None
                    and placement.placement_id.startswith(machine_lanes.REPLICA)
                    and placement.bindings_digest
                    and (placement.installation_id, documents.spell(placement.bindings_digest))
                    == template
                    for ordinal in placement.device_pin or ()
                }
                | set(self.prespawns.ordinals(template[0]))
            )
        )

    def _waiting_call(self, owner: str, key: str) -> dict[str, str]:
        """The waiting call's function and author label: what waits, not only for what."""
        assert self.machine_calls is not None
        call = self.machine_calls.journal.child(owner, key.rpartition("#")[0])
        if call is None:
            return {}
        label = self.calls.progress_label(call.parent_request, call.parent_ordinal, call.call_index)
        return {"function": call.target().export, **({"label": label} if label else {})}

    def _gpu_waiting(
        self, owner: str, root: str, body: gpu_scheduler.Document, view: gpu_scheduler.Document
    ) -> None:
        """Once per transition, tell the waiting call's parent (or root) why it waits. A root's
        own calls holding the cards are not a run it is behind."""
        assert self.executions is not None
        request = str(body["key"]).rpartition("#")[0]
        _root, parent = self.executions.scheduling_root(owner, request)
        target = self.engine.live.get(parent or request)
        if target is None:
            return
        others = [blocker for blocker in body["blocked_by"] if blocker != root]
        if others or not body["blocked_by"]:
            why = "behind " + (", ".join(others) or "released devices")
        else:
            held = {
                o
                for held_by, ordinals in view["grants"].values()
                if held_by == root
                for o in ordinals
            }
            why = f"{len(held)} in use by this run's other calls"
        self.emit_progress(
            target.request_id,
            target.attempt,
            {"kind": "progress", "stage": f"Waiting for GPU (needs {body['width']}, {why})"[:120]},
        )

    def _defer_rebuild(self, placement_id: str) -> child.Executor | None:
        """A dead GPU replica rebuilds on its next grant, never on devices it no longer holds.
        Returns the dead executor to reclaim now, or None when the rebuild is not deferred."""
        placement = self._placement_by_id(placement_id)
        if (
            placement is None
            or not placement.placement_id.startswith(machine_lanes.REPLICA)
            or not placement.device_pin
            or any(
                attempt.placement_id == placement_id and attempt.state in self.engine.PRE_RELEASE
                for attempt in self.engine.history.values()
            )
        ):
            return None
        hosted = self.hosted.get(placement_id)
        supervision = hosted.supervision if hosted is not None else self.supervision
        # An already reclaimed generation stays deferred: only a grant rebuilds it.
        current = supervision.current or self._dead_replicas.get(placement_id)
        if current is not None:
            self._dead_replicas[placement_id] = current
        return current

    def _reclaim_dead(self, placement_id: str, dead: child.Executor, why: str = "") -> bool:
        """Kill, reap and prove absent one tenant's executor NOW, under its lane's device
        hold: only its successor waits. A poisoned process left alive kept 62 GB of a card
        another tenant was then granted (darkness 1514). True once gone."""
        hosted = self.hosted.get(placement_id)
        supervision = hosted.supervision if hosted is not None else self.supervision
        try:
            evidence = supervision.retire_current(dead, (why or dead.poisoned or "exited")[:200])
        except ExecutorGone as exc:
            if hosted is not None:
                hosted.latched = f"executor_reclaim_failed: {exc}"[:200]
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                    subject=placement_id,
                    reason="executor_reclaim_failed",
                    detail=safe(str(exc)),
                )
            )
            return False
        members = list(evidence.members) if evidence is not None else [dead.pid]
        self.note(
            "recovery",
            f"{placement_id!r}: reclaimed executor epoch {dead.epoch} (pids {members}): "
            f"{why or dead.poisoned or 'exited'}"[:400],
        )
        return True

    def _revive(self, placement_id: str) -> None:
        """Rebuild a deferred dead replica now that a grant holds its devices, here."""
        executor = self._dead_replicas.pop(placement_id, None)
        lane = self.lanes.lane_of(placement_id)
        placement = self._placement_by_id(placement_id)
        if executor is not None and lane is not None and placement is not None:
            # Its successor may already be starting beside this root's downloads.
            warm = self.prespawns.claim(placement.installation_id, placement.device_pin or ())
            if warm is not None:
                warm.started.wait()
            self._rebuild_lane(lane, placement_id, executor, granted=True)

    def _cpu_job(self, job: pb.JobDirective) -> bool:
        """An orchestration job granted no device holds no GPU (`machine_slots.admits`)."""
        if not job.orchestration:
            return False
        binding = self._job_plan(job.installation_id, job.job_descriptor_id)
        declaration = self.engine._job_declaration(binding, binding.job_descriptor_id)
        return machine_slots.admits(job, declaration)

    def _job_plan(self, installation_id: str, descriptor: str) -> JobBinding:
        plan = job_plan.path(self.config.cozy_home / "job-plans", installation_id, descriptor)
        return JobBinding.read(json.loads(plan.read_bytes()))

    def _failed_execution_binding(self, attempt: AttemptRecord) -> str | None:
        """A known construction refusal is terminal before waiting for a GPU seat.

        STAGED/OFFLINE alone is waitable. Only the exact selected binding on the current
        placement and environment owns this failure; other bindings and replaced
        placements must not poison a queued request.
        """
        serving = attempt.spec.get("serving")
        if serving is None:
            return None
        hosted = self.hosted.get(attempt.placement_id)
        placement = hosted.placement if hosted is not None else self.placement
        if (
            placement is None
            or placement.placement_id != attempt.placement_id
            or not placement.installation_id
            or len(placement.bindings_digest) != 32
            or placement.installation_id != attempt.spec.get("installation_id")
            or documents.spell(placement.bindings_digest) != serving.get("bindings_digest")
        ):
            return None
        failed = hosted.failed_bindings if hosted is not None else self.failed_bindings
        return failed.get(serving.get("entrypoint_binding_digest", ""))

    def _journal_row(self, request: str, attempt: int, frame: dict[str, Any]) -> None:
        """One worker observation on an execution's own journal; narration never fails it."""
        with contextlib.suppress(WorkspaceRefusal, OSError):
            self.emit_progress(request, attempt, frame)

    @contextlib.contextmanager
    def _call_phase(self, request: str, attempt: int, name: str) -> Iterator[None]:
        """Name one piece of a call's work before its handler, as `Checking model inputs` is
        named: the parent's journal (a root's own) shows it under the call's label and times
        it, so an executor start or a weight load never reads as the input check (run 1516)."""
        calls = self.machine_calls
        call = calls.journal.child(self.fence.record_owner_id, request) if calls else None
        journal, label = (request, attempt), ""
        if call is not None:
            journal = (call.parent_request, call.parent_ordinal)
            label = self.calls.progress_label(*journal, call.call_index)
        self._journal_row(
            *journal, {"kind": "progress", "stage": f"{label} / {name}" if label else name}
        )
        started, began, completed = time.time(), time.perf_counter(), False
        try:
            yield
            completed = True
        finally:
            fields = {
                "child_request": request,
                "phase": name,
                "completed": completed,
                "started_unix_ms": int(started * 1000),
                "elapsed_ms": round((time.perf_counter() - began) * 1000, 3),
            }
            self._journal_row(
                *journal,
                {
                    "kind": "log",
                    "name": name,
                    "value": "info",
                    "at_unix_ms": int(time.time() * 1000),
                    "fields": fields,
                },
            )

    def dispatch_machine(
        self,
        unit: Unit,
        offered: pb.AttemptOffer,
        ordinals: tuple[int, ...],
        moved: Callable[[], bool],
    ) -> AttemptRecord | None:
        """Prepare this execution's placement on its grant and hold its attempt, on the
        execution's own thread. None when it was refused (its terminal is recorded) or when
        what it waited on moved (the caller re-reads its row)."""
        assert self.executions is not None
        owner, request = self.fence.record_owner_id, offered.request_id
        machine_byte_inputs.bind_root_grant(owner, offered)
        try:
            preparation = machine_lanes.placement_for(
                self.executions.preparation(owner, request), offered, ordinals
            )
            if preparation and not self._prepare_execution(
                preparation, request, offered.attempt_ordinal, unit, moved
            ):
                return None
        except (Stopping, *execution_unit.FENCES):
            raise
        except Exception as exc:
            self.refuse(
                offered,
                "execution_preparation_unavailable",
                f"Retained preparation is unavailable ({type(exc).__name__}): {exc}",
                CAUSE.CAUSE_CODE_CAPABILITY_UNAVAILABLE,
            )
            return None
        try:
            attempt = self.engine.offer(offered)
        except AttemptRefusal as exc:
            self.refuse(offered, exc.code, exc.detail, exc.cause, exc.origin)
            return None
        if attempt.state in ("outcome", "closed"):
            return None
        if fault := self._failed_execution_binding(attempt):
            self.refuse(
                offered,
                "binding_faulted",
                f"Model preparation failed: {fault}",
                CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
            return None
        # Desired-state replacement can still expose an older placement while the selected
        # one converges: capacity on it is not readiness for this exact offer.
        if "serving" in attempt.spec and not self._await_capacity(
            unit, lambda: self._placement_dispatchable(attempt.placement_id), moved
        ):
            return None
        if refusal := self._admission_refusal(offered, attempt, machine_owned=True):
            self.refuse(offered, *refusal)
            return None
        attempt.lane_id = self._offer_lane(offered.placement_id, attempt).lane_id
        self.engine.hold(attempt)
        if attempt.deadline_ms:
            self._guarded(attempt)
        return attempt

    def _await_capacity(
        self, unit: Unit, ready: Callable[[], bool], moved: Callable[[], bool]
    ) -> bool:
        """Wait for a fact that moves only when an attempt leaves or a placement settles
        (`_capacity_changed`). False when what the caller waits on moved first."""
        self._capacity_waiters.add(unit.key)
        try:
            unit.wait_for(lambda: ready() or moved())
            return ready()
        finally:
            self._capacity_waiters.discard(unit.key)

    def _capacity_changed(self) -> None:
        """An attempt left or a placement settled: wake the units waiting on it."""
        self.compute_idle_since = time.monotonic()
        for key in tuple(self._capacity_waiters):
            self.supervisor.poke(key)
        with self.attempts_left:
            self.attempts_left.notify_all()

    def _prepare_execution(
        self,
        preparation: dict[str, Any],
        request_id: str,
        attempt: int,
        unit: Unit,
        moved: Callable[[], bool],
    ) -> bool:
        """Reuse the preparation engine on the execution's own thread, beside live work.
        False when what it waited on moved."""
        if self.phase == pb.WorkerPhase.WORKER_PHASE_FAILED:
            raise WorkspaceRefusal(f"worker preparation stopped: {self.failure_reason}")
        desired = machine_lanes.state(preparation)
        if desired.WhichOneof("mode") == "job" and self._cpu_job(desired.job):
            held = self.job_slots.get(machine_slots.key(request_id))
            if held is not None and held.ready():
                return True
            binding = self._job_plan(desired.job.installation_id, desired.job.job_descriptor_id)
            with self.control_lock:
                machine_capture.restore(self, preparation)
            return machine_slots.ensure(self, request_id, desired.job, binding)
        if desired.WhichOneof("mode") == "placement_set":
            return self._bind_replica(preparation, desired, request_id, attempt, unit, moved)
        # A device job replaces the whole worker: it waits for every held attempt to leave.
        with self.control_lock:
            desired.revision = self.accepted.accepted_desired_state_revision
            if desired_state_body(desired) == self.accepted.desired_state_body:
                return True

        def drained() -> bool:
            return not any(
                (attempt.state in self.engine.PRE_RELEASE or attempt.state in self.engine.HELD)
                and not (
                    not desired.job.orchestration
                    and attempt.lane_id.startswith("cpu-")
                    and attempt.state != "queued"
                )
                for attempt in list(self.engine.history.values())
            )

        if not self._await_capacity(unit, drained, moved):
            return False
        with self.control_lock:
            machine_capture.restore(self, preparation)
            desired.revision = self.accepted.accepted_desired_state_revision + 1
            self._apply_desired_state(desired)
        if not self._await_capacity(unit, lambda: self.dispatchable() or bool(self.latched), moved):
            return False
        if self.latched:  # its executor never booted: this execution's typed refusal
            raise WorkspaceRefusal(f"the job could not start: {self.latched}")
        return True

    def _bind_replica(
        self,
        preparation: dict[str, Any],
        desired: pb.DesiredWorkerState,
        request_id: str,
        attempt: int,
        unit: Unit,
        moved: Callable[[], bool],
    ) -> bool:
        """Activate this machine replica on its grant, beside every other placement. One
        execution activates (or revives) it and the others wait; a failed one refuses the
        demand that met it and retires, so the next demand activates it afresh."""
        (row,) = read_placement_set(desired.placement_set)
        pins = desired.placement_set.device_pins
        pin = tuple(pins[0].device_ordinals) if pins else None
        placement_id = row.placement_id
        while True:
            with self.control_lock:
                hosted = self.hosted.get(placement_id)
                if hosted is not None and self._placement_dispatchable(placement_id):
                    return True
                busy = placement_id in self._activating
                failed = hosted is not None and not busy and self._replica_failed(hosted)
                mine = (
                    not busy
                    and not failed
                    and (hosted is None or placement_id in self._dead_replicas)
                )
                if mine:
                    self._activating.add(placement_id)
            if failed:
                break
            if not mine:
                if not self._await_capacity(
                    unit,
                    lambda: (
                        placement_id not in self._activating
                        and (
                            self._placement_dispatchable(placement_id)
                            or self._replica_gone(placement_id)
                        )
                    ),
                    moved,
                ):
                    return False
                continue
            try:
                if hosted is None:
                    self._activate_replica(row, desired, preparation, pin, request_id, attempt)
                else:
                    with self._call_phase(request_id, attempt, "Starting model executor"):
                        self._revive(placement_id)
            finally:
                with self.control_lock:
                    self._activating.discard(placement_id)
                self._capacity_changed()
        assert hosted is not None
        self.supervisor.spawn(f"retire:{placement_id}", lambda _: self._retire_one(placement_id))
        fault = next((f for f in reversed(self.engine.faults) if f.subject == placement_id), None)
        detail = f"{fault.reason}: {fault.detail}" if fault else hosted.latched
        raise WorkspaceRefusal(f"model preparation failed: {detail or placement_id}")

    def _replica_failed(self, hosted: HostedPlacement) -> bool:
        placement = hosted.placement
        return bool(
            hosted.latched
            or placement.serving == pb.ServingState.SERVING_STATE_OFFLINE
            or placement.materialization
            in (pb.MATERIALIZATION_STATE_FAILED, pb.MATERIALIZATION_STATE_ABSENT)
        )

    def _replica_gone(self, placement_id: str) -> bool:
        hosted = self.hosted.get(placement_id)
        return hosted is None or self._replica_failed(hosted) or placement_id in self._dead_replicas

    def _activate_replica(
        self,
        row: pb.Placement,
        desired: pb.DesiredWorkerState,
        preparation: dict[str, Any],
        pin: tuple[int, ...] | None,
        request_id: str,
        attempt: int,
    ) -> None:
        """Acquire, bind and start this replica's executor; a device job's leaves first."""
        with self.control_lock:
            machine_capture.restore(self, preparation)
        placement = self._placement_from_entry(row, desired.placement_set.placement_set_digest)
        placement.device_pin = pin
        # The process started beside this replica's download (h3a-087), when there is one.
        journal = functools.partial(self._journal_row, request_id, attempt)
        warm = self.prespawns.claim(placement.installation_id, pin or (), journal)
        supervision = (
            warm.supervision
            if warm is not None
            else self._new_placement_supervision(
                placement.placement_id,
                uid_key=prespawn.uid_key(placement.installation_id, pin or ()),
            )
        )
        hosted = HostedPlacement(placement, supervision)
        with self.control_lock:
            self.hosted[placement.placement_id] = hosted
            job = self.accepted.mode == "job"
            if job:
                # A device job's slot leaves with it; CPU job slots keep serving beside replicas.
                self.lanes.envelope.forget("")
                self.job_slots.pop("envelope", None)
            self.accepted.mode = "serving"
        # A claimed prespawn journals its own start; only a start this call pays is named here.
        starting = (
            self._call_phase(request_id, attempt, "Starting model executor")
            if warm is None
            else contextlib.nullcontext()
        )
        try:
            with starting:
                if warm is not None:
                    self.prespawns.settle(warm, placement.placement_id)
                current = self.supervision.current
                if job and current is not None:
                    self.supervision.retire_current(current, "a machine replica takes the devices")
                self._activate_hosted(hosted, None)
        except Exception as exc:
            hosted.latched = f"replica_activation_failed: {type(exc).__name__}: {exc}"[:200]
            self._settle()

    def collect_execution(
        self, claim: pb.Claim, request_id: str, ordinal: int | None = None
    ) -> pb.AttemptOutcome:
        executions, owner = self._execution_service(claim)
        return executions.collect(owner, request_id, ordinal)

    def acknowledge_execution_collection(
        self, claim: pb.Claim, ack: pb.AttemptOutcomeAck
    ) -> workspace_executions.State:
        executions, owner = self._execution_service(claim)
        state = executions.acknowledge_collection(owner, ack)
        if (
            self.engine.products is not None
            and ack.attempt_ordinal == state.attempt_ordinal
            and state.state in workspace_executions.TERMINAL
        ):
            # The owner holds the whole log: delivered. Its products stay until GC.
            self.engine.products.deliver(owner, ack.request_id)
        return state

    def prepare_model_source(
        self, request: pb.PrepareModelSourceRequest
    ) -> pb.PrepareModelSourceResult:
        """Hand exact supervisor-verified source facts to the shared TensorFS core."""

        with self.preparation_lock:
            if self.stop.is_set():
                return pb.PrepareModelSourceResult(
                    outcome=pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED,
                    safe_code="model_source_preparer_unavailable",
                    safe_detail="the worker is stopping",
                )
            result = model_source_prepare.prepare_model_source(
                request, tensorfs_root=Path(self.options.tensorfs_root or "")
            )
            if result.outcome in {
                pb.MODEL_SOURCE_PREPARE_OUTCOME_PREPARED,
                pb.MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED,
                pb.MODEL_SOURCE_PREPARE_OUTCOME_INCOMPLETE,
            }:
                self.model_source_operations.add(request.operation_id)
            return result

    def checkpoint_page(self, request: pb.CheckpointPageRequest) -> pb.CheckpointPageResult:
        from . import checkpoint_transport

        return checkpoint_transport.page(
            request, tensorfs_root=Path(self.options.tensorfs_root or "")
        )

    def checkpoint_transfer(
        self, request: pb.CheckpointTransferRequest
    ) -> pb.CheckpointTransferStatus:
        from . import checkpoint_transport

        return checkpoint_transport.transfer(
            request,
            tensorfs_root=Path(self.options.tensorfs_root or ""),
            restore_scope=self.weights.restore_checkpoint,
        )

    def release_model_source(self, operation_id: str) -> None:
        """Release one native source operation under the preparation lock."""

        with self.preparation_lock:
            model_source_prepare.release_model_source(
                tensorfs_root=Path(self.options.tensorfs_root or ""),
                operation_id=operation_id,
            )
            self.model_source_operations.discard(operation_id)

    def retain_derived_result(
        self, request: pb.DerivedRetentionRequest, release: bool
    ) -> pb.DerivedRetentionResult:
        from . import derived_retention

        return derived_retention.change(
            request, tensorfs_root=Path(self.options.tensorfs_root or ""), release=release
        )

    def release_derived_result(
        self, request: pb.DerivedResultReleaseRequest
    ) -> pb.DerivedResultReleaseResult:
        from . import derived_retention

        result = derived_retention.release_result(
            request, tensorfs_root=Path(self.options.tensorfs_root or "")
        )
        self.weights.abandon(request.weights_transaction_id)
        return result

    def collect_store_garbage(self) -> pb.CollectStoreGarbageResult:
        return store_gc.collect_worker(self)

    def note(self, kind: str, step: str) -> None:
        """The NON-ATTEMPT activity lane: typed, sequence-bearing, on the DURABLE Report —
        never overloaded onto the lossy progress stream."""
        event = pb.ActivityEvent(
            seq=next(self.seq), kind=kind, step=step[:256], at_unix_ms=int(time.time() * 1000)
        )
        self.activity.append(event)
        del self.activity[:-64]
        print(f"[worker] {kind}: {step}", flush=True)

    def stamp(self, message: Any) -> Any:
        target = message
        if message.DESCRIPTOR.name in ("WorkerFrame", "RecordOwnerFrame"):
            kind = message.WhichOneof("msg")
            if kind is None:
                return message
            target = getattr(message, kind)
        _stamped(target, self.fence)
        return message

    def send(self, frame: pb.WorkerFrame) -> None:
        """Durable. Queued, never shed; the inner message is stamped with the CURRENT fence
        so a frame authored before a re-claim still leaves under the live envelope.

        STAMP AND ENQUEUE ARE ONE STEP, under the same lock claim acceptance takes: the
        envelope a frame carries and the stream it is queued on are then the same epoch
        by construction, with no window where a frame is stamped for one stream and handed to
        another's sender.
        """
        with self.claim_lock:
            self.stamp(getattr(frame, frame.WhichOneof("msg")))
            self.outbound.put(frame)

    def settle(self, outcome: pb.AttemptOutcome) -> None:
        """An attempt's terminal lands on its execution and closes the attempt at once; the
        execution's unit settles custody. Outcome forwarding follows the same workspace
        commit that observes native receipts."""
        owner = self.fence.record_owner_id
        request, ordinal = outcome.request_id, outcome.attempt_ordinal
        if self.workspace is not None and self.workspace.contains(owner, request, ordinal):
            self.workspace.outcome(owner, outcome)
        if self.executions is not None and self.executions.owns(owner, request):
            self.executions.reconcile(owner, request)
        self._close(outcome)

    def set_serving(self, state: pb.ServingState, why: str = "") -> None:
        """Move the placement's SERVING axis and REPORT IT IMMEDIATELY — dispatchability is
        an EDGE, and an ObservedWorkerState that arrives on the next tick is a stale one.

        `IntakeState` is retired with both of its uses (#482/#510b). What it conflated was
        machine lifecycle (now `worker_phase`), what is on disk (now the MATERIALIZATION
        axis) and what a placement will take (this one) — and a single enum could not say
        "staged on disk but offline", which is exactly the state an outgoing placement holds under
        fallback-retention.

        ADMISSION MEANING MOVES WITH IT, so `admission_epoch` bumps here (§6): an offer
        minted against the old epoch refuses deterministically rather than racing a
        placement that stopped being dispatchable while it was in flight.
        """
        if self.placement is None or state == self.placement.serving:
            return
        self.placement.serving = state
        self.bump_admission(why or f"serving -> {pb.ServingState.Name(state)}")
        self._settle()

    def set_job_ready(self, ready: bool) -> None:
        """Job mode has no placement, so its admission edge is its own (cr-009)."""
        if ready == self.job_ready:
            return
        self.job_ready = ready
        self.bump_admission(f"job_ready={ready}")
        self._settle()

    def _residency_changed(self) -> None:
        """A residency set moved (a manifest held or gone, a fill or a vacate): the owner
        reads it from the next state, so one goes now rather than at the next tick."""

    def _rebuild_held_manifests(self) -> None:
        """Boot: `held_manifests` from the store's own verified walk, no network. A worker
        without a store or without the TensorFS reader holds nothing and says so."""
        root = self.options.tensorfs_root
        if root is None or not Path(root).is_dir():
            return
        started = time.perf_counter()
        try:
            store = fill.open_store(Path(root))
            held = self.held_manifests.rebuild(store)
        except Exception as exc:
            self.note("boot", f"held_manifests: none — {exc}"[:256])
            return
        self.note(
            "boot",
            f"held_manifests: {held} manifest(s) in {root} verified complete in "
            f"{(time.perf_counter() - started) * 1000:.0f} ms",
        )

    def _settle(self) -> None:
        """Each lane's `settled` is that lane's gate: has IT stopped moving? A latched
        activation failure settles too — the lane must refuse through it, not wait forever.
        A rebuild on lane A clears A's gate and no other, which is what keeps B's queued
        attempts from waiting on a rebuild they did not cause (cr-066)."""
        for lane in self.lanes:
            if self._lane_dispatchable(lane) or self.latched or self._lane_latched(lane):
                lane.settled.set()
            else:
                lane.settled.clear()
        self._capacity_changed()

    def _lane_latched(self, lane: DeviceLane) -> bool:
        """Every placement on the lane has stopped moving for the accepted revision."""
        if not lane.placements:
            return False
        for placement_id in list(lane.placements):
            hosted = self.hosted.get(placement_id)
            if not (hosted.latched if hosted is not None else self.latched):
                return False
        return True

    def _publish_latch(self) -> None:
        """A LATCH IS A REFUSAL, AND A REFUSAL AN OWNER CANNOT READ IS A HANG.

        `latched` is written at a dozen sites and every one of them says the same thing: this
        worker has stopped moving for the accepted revision and dispatchability will never
        arrive. Half of those sites also typed a `Fault`; the rest only wrote the string and
        printed it, which is worker-local — the refusal was correct, rendered on this
        process's own stderr, and completely invisible on the wire, so an owner waiting for
        dispatchability waited for something no longer being produced (cr-061).

        Publication belongs on the REPORT, once, and not at each site remembering to: a site
        can forget, and eight of them had. The typed cause fault a site raises is about the
        THING that failed; this one is about the WORKER having stopped, which is the fact an
        owner without retry authority has to act on.
        """
        if self.latched == self._published_latch:
            return
        self._published_latch = self.latched
        if not self.latched:
            return
        subject = self.options.worker_id
        if self.accepted.mode == "serving" and self.placement is not None:
            subject = self.placement.placement_id or subject
        reason, _, cause = self.latched.partition(": ")
        self.engine.fault(
            pb.Fault(
                kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                subject=subject,
                reason=reason[:200] or "latched",
                detail=safe(cause or self.latched)[:1024],
            )
        )

    def bump_admission(self, why: str) -> None:
        """The ONE worker-level admission fence moves (§6). Callers: executor respawn,
        window resize, phase change, cutover, and every serving-axis edge above."""
        self.admission_epoch += 1
        self.note("admission", f"epoch {self.admission_epoch}: {why}"[:256])

    def dispatchable(self) -> bool:
        """Can this worker take an attempt AT ALL right now? Serving asks the placement's
        axis; job mode asks whether its one build is ready."""
        if any(slot.ready() for name, slot in self.job_slots.items() if name.startswith("cpu-")):
            return True
        if self.accepted.mode == "job":
            return self.job_ready
        return bool(
            (
                self.placement is not None
                and self.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
            )
            or any(
                hosted.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
                for hosted in self.hosted.values()
            )
        )

    def admission_state(self) -> pb.AdmissionState:
        """CLOSED: no attempt is offered over the Control stream; work arrives through the
        execution API."""
        return pb.AdmissionState.ADMISSION_STATE_CLOSED

    def set_phase(self, phase: pb.WorkerPhase, why: str) -> None:
        if phase == self.phase:
            return
        self.phase = phase
        self.bump_admission(f"worker_phase -> {pb.WorkerPhase.Name(phase)}: {why}")
        self._settle()

    def latch(self, fault: pb.Fault) -> None:
        """An activation failure LATCHES for this desired revision (§8): the worker does not
        evict-and-retry forever, `converged_revision` stays behind, and the fault is typed on
        the placement. The previous set keeps serving (fallback-retention, #474)."""
        self.latched = f"{pb.FaultKind.Name(fault.kind)}: {fault.reason}"[:200]
        self.engine.fault(fault)
        # This fault IS the publication; `_publish_latch` must not repeat it.
        self._published_latch = self.latched
        self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, "activation failed; LATCHED")
        self._settle()

    def fail_machine(self, why: str) -> None:
        """A worker INTEGRITY failure (`WORKER_PHASE_FAILED`, #482), for what no one execution
        owns: admission closes and each held execution's unit ends it FAILED with this reason,
        so a failed worker never holds a rental. A marker makes the next boot fail what this
        one held rather than run it again."""
        if self.phase == pb.WorkerPhase.WORKER_PHASE_FAILED:
            return
        self.failure_reason = safe(f"the worker failed: {why}", 4096)
        self.engine.fault(
            pb.Fault(
                kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                subject=self.options.worker_id,
                reason=why[:1024],
                detail="the worker hit an unrecoverable local safety fault, so it stops "
                "admitting further work",
            )
        )
        self.set_phase(pb.WorkerPhase.WORKER_PHASE_FAILED, why)
        with contextlib.suppress(OSError):
            (self.root / FAILED_MARKER).write_text(
                json.dumps({"worker_boot_id": self.fence.worker_boot_id, "reason": why[:4096]})
            )
        for held in list(self.engine.live.values()):
            with contextlib.suppress(Exception):
                self.cancel(
                    pb.CancelAttempt(
                        request_id=held.request_id,
                        attempt_ordinal=held.attempt,
                        invocation_spec_digest=held.digest,
                    ),
                    failure=self.failure_reason,
                )
        for key in self.supervisor.keys(execution_unit.EXECUTION):
            self.supervisor.poke(key)

    def _lane(
        self, name: str, body: Callable[..., None], *args: Any, fatal: bool = True
    ) -> Callable[[], None]:
        """Wrap one worker lane so its DEATH is a reported fact rather than a silence.

        Each lane below is the sole producer of some terminal: the device lane produces
        attempt outcomes, convergence produces a placement's verdict, the sender is the only
        thing that puts a frame on the stream. Nothing on the wire describes a dead Python
        thread, so a lane that dies leaves whoever waits on that terminal waiting on a
        producer that no longer exists — the exact shape cr-061 hit from the other direction.

        The answer is the one fact every owner already reads: `WORKER_PHASE_FAILED`. And
        where the host IS the stream — the local one-shot door, whose `persistent` is False —
        the stream is closed too, so the process EXITS with a non-zero code instead of
        sitting on a queue nothing will ever fill again. A helper that is no one's sole
        producer (`fatal=False`: an unload, a page warm) only notes its failure.
        """

        def guarded() -> None:
            native_id = threading.get_native_id()
            self.lane_threads[name] = native_id
            try:
                body(*args)
            except BaseException as exc:
                if not fatal and isinstance(exc, Exception):
                    self.note("recovery", f"{name} stopped: {fault_text(exc)}"[:400])
                    return
                detail = f"the {name} lane died: {type(exc).__name__}: {exc}"[:400]
                traceback.print_exc()
                self.exit_code = int(Exit.internal)
                with contextlib.suppress(Exception):
                    self.note("recovery", detail)
                with contextlib.suppress(Exception):
                    self.fail_machine(detail)
                if not self.host.persistent:
                    with contextlib.suppress(Exception):
                        self.host.stop()
            finally:
                if self.lane_threads.get(name) == native_id:
                    self.lane_threads.pop(name, None)

        return guarded

    def capacity_dropped(self, why: str) -> None:
        """The executor's state was invalidated: STOP ADVERTISING DISPATCHABLE, right now."""
        if self.accepted.mode != "job" and (
            self.placement is None
            or self.placement.serving != pb.ServingState.SERVING_STATE_DISPATCHABLE
        ):
            return
        self.note("serving", f"capacity withdrawn: {why[:200]}")
        if self.accepted.mode == "job":
            self.set_job_ready(False)
        else:
            self.set_serving(pb.ServingState.SERVING_STATE_ACTIVATING)

    def executor_exited(self, executor: child.Executor) -> None:
        """Withdraw an exact dead epoch and rebuild it under its lane's device hold.

        The pidfd watcher does not reap. Invalidation clears the ready-plan set and moves
        admission before this callback returns; the rebuild waits behind whichever attempt
        already holds the device. Both target ``executor``, so a watcher for E1 cannot
        invalidate or rebuild E2.
        """
        with self.control_lock:
            rebuild = bool(
                (self.accepted.mode == "job" and self.job_ready)
                or (
                    self.placement is not None
                    and self.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
                )
            )
            self._host_facts = None
            if not self.supervision.invalidate(
                executor,
                f"executor epoch {executor.epoch} pid {executor.pid} exited",
            ):
                return
        if rebuild and not self.stop.is_set():
            self._executor_died(self._primary_lane(), executor, "")

    def executor_changed(self, executor: child.Executor | None) -> None:
        """Move the PRIMARY slot's ledger row at the boundary that publishes its executor.

        A job generation has no construction report, so resetting only in the serving fill
        path lets it inherit the previous serving process's baseline and residency. The
        supervision callback covers every spawn and retirement before either can be
        observed through the session. Only this slot's row moves: a hosted placement on the
        same or another lane keeps its facts (cr-066).
        """
        self._primary_row().begin_generation(executor.pid if executor is not None else 0)

    def read_device_process(self, pid: int) -> int:
        """NVML bytes for one executor pid; -1 remains UNREADABLE/ABSENT here.

        The ledger's legacy scalar cannot express absence separately. Lifecycle reclaim uses
        AcceleratorOps' tri-state observation directly, where that distinction is authoritative.
        """
        if pid <= 0:
            return -1
        observed = accel.process_memory(pid, accel.host_backend_family())
        return observed.bytes if observed.state == "present" else -1

    def read_device_work(self) -> int:
        """How much work EVERY live executor has done, read by the parent — THE meter.

        Summed over the lanes: the silence monitor asks whether the device side is moving
        at all, and with one executor per lane any of them advancing is the answer.
        """
        supervisions = [self.supervision, *(h.supervision for h in self.hosted.values())]
        return sum(
            child.work(supervision.current.pid)
            for supervision in supervisions
            if supervision.current is not None
        )

    @property
    def pollers(self) -> set[int]:
        """The kernel thread ids of the lanes that wake on a CLOCK — the reporter and the
        sender that delivers what the reporter ticks — whose CPU says nothing about whether
        this worker is getting anywhere. Every other thread burns only when it works, which
        is what makes the worker's own burn, less these, an honest progress meter."""
        return {
            tid for name, tid in self.lane_threads.items() if name in ("report", "control-send")
        }

    def applied_posture(self) -> dict[str, Any]:
        """The state the worker was in when something happened (§3.6). `readiness_epoch` is
        NOT here: it is DELETED, not renamed (#486b) — the serving axis says what the
        placement can serve and `admission_epoch` says whether an offer is admissible,
        which is one fence instead of two to keep consistent."""
        return {
            "posture": pb.Posture.Name(self.accepted.posture),
            "worker_phase": pb.WorkerPhase.Name(self.phase),
            "materialization": pb.MaterializationState.Name(
                self.placement.materialization
                if self.placement
                else pb.MaterializationState.MATERIALIZATION_STATE_UNSPECIFIED
            ),
            "serving": pb.ServingState.Name(
                self.placement.serving
                if self.placement
                else pb.ServingState.SERVING_STATE_UNSPECIFIED
            ),
            "accepted_desired_state_revision": self.accepted.accepted_desired_state_revision,
            "converged_revision": self.converged_revision(),
            "record_owner_epoch": self.fence.record_owner_epoch,
            "control_stream_epoch": self.fence.control_stream_epoch,
            "applied_wire_minor": self.fence.wire_minor,
            "admission_epoch": self.admission_epoch,
            "executor_epoch": self.supervision.epoch,
            "placement_id": self.placement.placement_id if self.placement else "",
            "worker_id": self.options.worker_id,
            "worker_release_id": self.options.release_id,
            "boot_steps": list(self.boot.steps),
        }

    def emit_progress(self, request_id: str, attempt: int, frame: dict[str, Any]) -> None:
        """The LOSSY lane. Bounded per watch, oldest-first shed, worker-owned sequence."""
        if self.machine_calls is not None:
            timing = self.machine_calls.run_timing
            owner = self.fence.record_owner_id
            if frame.get("kind") == "execution_activity":
                timing.observe(owner, request_id, attempt, frame)
                return
            timing.snapshot(owner, request_id, attempt)
        key = (request_id, attempt)
        # A load's positions are liveness for the monitor and watches; its start is history.
        if self.executions is not None and not (
            frame.get("kind") == "load" and frame.get("position")
        ):
            self.executions.progress(
                self.fence.record_owner_id, request_id, attempt, _progress_document(frame)
            )
        position = (
            frame.get("advance") if frame.get("kind") == "progress" else frame.get("position")
        )
        if isinstance(position, int):
            self.monitor.advance(f"{request_id}#{attempt}", position)
        if frame.get("kind") != "progress":
            self.engine.forward(request_id, attempt, frame)
        counter = self.progress_seq.setdefault(key, itertools.count(1))
        self.watches.publish(
            self.stamp(
                pb.AttemptProgress(
                    request_id=request_id,
                    attempt_ordinal=attempt,
                    seq=next(counter),
                    content_type="application/x-cozy-event+json",
                    data=json.dumps(_progress_document(frame)).encode()[:65536],
                    placement_id=self.placement.placement_id if self.placement else "",
                )
            )
        )

    def native_progress(
        self, request_id: str, attempt: int, index: int, frame: dict[str, Any]
    ) -> None:
        """Native child work reports on its parent's lane, under the call's author label."""
        label = self.calls.progress_label(request_id, attempt, index)
        if label and frame.get("kind") == "progress":
            frame = {**frame, "stage": f"{label} / {frame['stage']}"[:120]}
        # Optional narration can outlive its parent attempt; it never fails the work.
        with contextlib.suppress(WorkspaceRefusal, OSError):
            self.emit_progress(request_id, attempt, frame)

    # ------------------------------------------------------------------ the stream

    def serve_stream(
        self, inbound: Iterator[pb.RecordOwnerFrame], send: Callable[[pb.WorkerFrame], None]
    ) -> None:
        """ONE control stream: Claim -> ClaimAck -> WorkerSnapshot -> (SnapshotAck) -> frames.

        Runs on the host's thread. The first frame MUST be a Claim; acceptance fences any
        previous stream by minting the next `control_stream_epoch`, so the superseded
        stream's sender exits on its next frame and its watches end.
        """
        try:
            first = next(inbound)
        except (StopIteration, Exception):
            return
        claim = first.claim if first.WhichOneof("msg") == "claim" else None
        if claim is None:
            self.note("claim", "dropped a stream whose first frame is not a Claim")
            return
        # Acceptance and the snapshot are one control transition. In particular, the claim
        # cannot snapshot half of a desired-state replacement running on another stream.
        with self.control_lock:
            epoch, outbound = self.accept_claim(claim, send)
            if epoch == 0:
                return  # refused; the rejection already went out and the stream ends
            if self.boot_fatal is not None:
                # The boot-fatal verdict INSTEAD of ClaimAck (01, #494c): the claim was
                # AUTHENTICATED and FENCED above — that is what makes the verdict deliverable
                # to this dialer — and then the worker says the one true thing it has to say.
                send(self.stamp(pb.WorkerFrame(boot_failure=self.boot_fatal)))
                self.note("boot", "boot failure delivered INSTEAD of ClaimAck; stopping")
                self.shutdown_from_stream()
                return
            self._send_claim_ack(claim, epoch, send)
            self.attached_streams.add(epoch)

        sender = threading.Thread(
            target=self._lane("control-send", self._pump_outbound, epoch, send, outbound),
            daemon=True,
            name=f"control-send-{epoch}",
        )
        sender.start()
        try:
            for frame in inbound:
                if self.fence.control_stream_epoch != epoch or self.stop.is_set():
                    self.note("fence", "a frame arrived on a superseded stream; ending it")
                    return
                self._dispatch_frame(frame, epoch)
        except Exception as exc:
            # THE MESSAGE, not the class name. A typed refusal carries the one sentence that
            # says what to do about it, and reducing it to `PlanRefusal` produced a log that
            # repeated forever and named nothing — the refusal had to be reproduced by hand
            # off the record to find out which key it was about.
            # A peer cancelling the stream (the Host superseding it for a newer owner stream,
            # or the owner hanging up) surfaces as an RpcError with no text of its own.
            self.note(
                "session",
                f"the control stream ended: {type(exc).__name__}: "
                f"{str(exc) or 'the peer closed the stream'}",
            )
        finally:
            # Unconditional: a SUPERSEDED stream is gone too, and leaving its epoch here
            # would report an owner that has been fenced out as still holding the pod.
            self.attached_streams.discard(epoch)

    def accept_claim(
        self, claim: pb.Claim, send: Callable[[pb.WorkerFrame], None]
    ) -> tuple[int, queue.Queue[pb.WorkerFrame | None]]:
        """The ownership fence (02 §4). AUTHENTICATE, fence, RECORD the authority durably,
        mint the epoch. Returns `(epoch, this stream's outbound queue)`, or
        `(0, …)` on refusal — the ClaimAck carrying the rejection has already gone out.

        Nothing is acknowledged here. The caller decides what the accepted claim is owed: a
        BootFailure when this worker never booted, a ClaimAck plus a snapshot otherwise.
        """

        def reject(reason: pb.ClaimRejection, why: str) -> tuple[int, queue.Queue[Any]]:
            # Naming the claimant tells the Runtime's own readiness proof (a foreign Claim it
            # must refuse) apart from a real owner's refused Claim.
            self.note(
                "claim",
                f"REFUSED ({pb.ClaimRejection.Name(reason)}) "
                f"for {claim.record_owner_id or 'an unnamed claimant'}: {why}",
            )
            ack = pb.ClaimAck(
                accepted=False,
                rejection=reason,
                wire_minor=WIRE_MINOR,
            )
            ack.record_owner_epoch = claim.record_owner_epoch
            ack.control_stream_epoch = self.fence.control_stream_epoch
            ack.worker_boot_id = self.fence.worker_boot_id
            send(pb.WorkerFrame(claim_ack=ack))
            return 0, queue.Queue()

        # THE CREDENTIAL CHECK: every Claim is the RecordOwner's signed ClaimProof.
        try:
            downloads.verify_claim_proof(
                claim,
                self.config.record_owner_public_key,
                worker_id=self.options.worker_id,
                worker_boot_id=self.fence.worker_boot_id,
                worker_tls_certificate_digest=self.options.worker_tls_certificate_digest,
            )
        except downloads.DownloadRefusal as exc:
            return reject(pb.ClaimRejection.CLAIM_REJECTION_UNAUTHENTICATED, exc.detail)
        if claim.worker_id and claim.worker_id != self.options.worker_id:
            return reject(
                pb.ClaimRejection.CLAIM_REJECTION_WORKER_ID_MISMATCH,
                f"this worker is {self.options.worker_id!r}",
            )
        with self.claim_lock:
            if not self.claim_ready.is_set() or self.stop.is_set():
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_UNDURABLE,
                    "the worker listener is bound but the readiness barrier is not committed",
                )
            if self.options.sole_supervisor and (
                claim.record_owner_id != "cozy-local-client" or claim.record_owner_epoch != 1
            ):
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_UNAUTHENTICATED,
                    "private supervisor call names another machine execution namespace",
                )
            if (
                self.fence.record_owner_recorded
                and claim.record_owner_id != self.fence.record_owner_id
                and self.executions is not None
                and self.executions.retention_required(self.fence.record_owner_id)
            ):
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_EPOCH_HELD,
                    "accepted machine execution retains its owner; reconnect is observation",
                )
            # `record_owner_recorded`, not the epoch: the high-water authority this worker
            # accepted is DURABLE (#494e), so a restart still refuses the owner it superseded
            # before the restart instead of welcoming it back as the first claim of a fresh
            # process.
            if (
                self.fence.record_owner_recorded
                and claim.record_owner_epoch < self.fence.record_owner_epoch
            ):
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_STALE_RECORD_OWNER_EPOCH,
                    f"epoch {claim.record_owner_epoch} is older than the accepted "
                    f"{self.fence.record_owner_epoch}",
                )
            if (
                self.fence.record_owner_recorded
                and claim.record_owner_epoch == self.fence.record_owner_epoch
                and claim.record_owner_id != self.fence.record_owner_id
            ):
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_EPOCH_HELD,
                    "equal epoch, different RecordOwner: the epoch is held",
                )
            # The existing stream counter is synced before its ClaimAck. A failed
            # write closes admission; a fresh process must read the stored authority.
            try:
                self.record_ownership(claim)
            except (OSError, ValueError):
                self.claim_ready.clear()
                self.request_stop()
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_UNDURABLE,
                    "the next control stream could not be durably recorded",
                )
            superseded = self.fence.control_stream_epoch
            if self.source_calls is not None:
                self.source_calls.fence_owner(claim.record_owner_id)
            self.fence.record_owner_epoch = claim.record_owner_epoch
            self.fence.record_owner_id = claim.record_owner_id
            self.fence.record_owner_recorded = True
            self.fence.control_stream_epoch += 1
            self.fence.wire_minor = min(WIRE_MINOR, claim.wire_minor)
            epoch = self.fence.control_stream_epoch
            # A FRESH queue for this stream. The superseded sender keeps its own and drains
            # into a dead call; nothing it still holds was ever this stream's to deliver, and
            # everything the law owes regenerates — unacked terminals replay after
            # SnapshotAck, reports re-tick.
            outbound: queue.Queue[pb.WorkerFrame | None] = queue.Queue()
            self.outbound = outbound
        if superseded:
            self.watches.end_epoch(superseded)
            self.note(
                "claim",
                f"stream epoch {superseded} fenced; stream epoch {epoch} claims "
                f"under owner epoch {claim.record_owner_epoch}",
            )
        # The owner's durable work resumes before its snapshot is built (once per process).
        self.reconcile_executions()
        return epoch, outbound

    def mark_claim_ready(self) -> None:
        """Open Claim after fixed-host listener proof, before readiness publication."""

        self._publish_restart_fact("restart-capability.json", {"supports_guarded_restart": True})
        self.claim_ready.set()

    def record_ownership(self, claim: pb.Claim) -> None:
        """Record the authority this claim carries before ClaimAck.

        Called under `claim_lock` and the existing process-lifetime worker-root lease.
        """
        if self.options.sole_supervisor:
            return  # authenticated private framing has no durable ownership history
        if self.ownership_path is not None:
            from cozy_runtime.internal.worker import ownership

            ownership.write(
                self.ownership_path,
                self.fence.worker_boot_id,
                claim.record_owner_epoch,
                claim.record_owner_id,
                self.fence.control_stream_epoch + 1,
            )
        self.records.append(
            "ownership",
            {
                "worker_boot_id": self.fence.worker_boot_id,
                "instance_id": self.options.instance_id,
                "record_owner_epoch": int(claim.record_owner_epoch),
                "record_owner_id": claim.record_owner_id,
            },
        )

    def _send_claim_ack(
        self, claim: pb.Claim, epoch: int, send: Callable[[pb.WorkerFrame], None]
    ) -> None:
        """ClaimAck + the convergence snapshot, for a claim this worker can actually serve.

        Exact control-wheel digest remains Hub-owned image provenance. Installed files cannot
        rederive a wheel archive digest, so the protocol's transitional field stays empty.
        """
        ack = pb.ClaimAck(
            accepted=True,
            wire_minor=WIRE_MINOR,
            worker_id=self.options.worker_id,
            worker_instance_id=self.options.instance_id,
            worker_release_id=self.options.release_id,
            control_runtime_digest="",
            git_commit=self.options.git_commit,
            resources=self.resources(),
        )
        send(self.stamp(pb.WorkerFrame(claim_ack=ack)))
        self._send_snapshot(send)
        self.note(
            "session",
            f"claimed by {claim.record_owner_id or 'an unnamed RecordOwner'} at owner epoch "
            f"{claim.record_owner_epoch}, stream epoch {epoch} "
            f"(wire_minor effective {self.fence.wire_minor})",
        )

    def snapshot_body(self) -> pb.WorkerSnapshotBody:
        """The whole of what this worker holds, in ONE document (§5).

        The three-message Begin/Entry/End form could only be bounded by a counted-entries
        check, and a count is a claim the sender makes about a stream the receiver is
        reassembling. A single canonical document is bounded by its own digest: a truncated
        or re-ordered snapshot cannot match, so there is no partial reconciliation to get
        wrong because there is no partial message.
        """
        body = pb.WorkerSnapshotBody(
            accepted_desired_state_revision=self.accepted.accepted_desired_state_revision,
            accepted_placement_set_digest=self.accepted.accepted_placement_set_digest,
            worker_phase=self.phase,
            converged_revision=self.converged_revision(),
            admission_epoch=self.admission_epoch,
            admission_state=self.admission_state(),
        )
        if self.placement is not None:
            body.placements.append(self.placement_status())
        # Machine executions own their attempts and collection through the execution API.
        # The legacy RecordOwner never assigned those ordinals and cannot reconcile them.
        body.held_attempts.extend(
            entry
            for entry in self.engine.snapshot_entries()
            if self.executions is None
            or not self.executions.owns(self.fence.record_owner_id, entry.request_id)
        )
        if self.workspace is not None:
            known = {(entry.request_id, entry.attempt_ordinal) for entry in body.held_attempts}
            for outcome in self.workspace.retained_outcomes(self.fence.record_owner_id):
                if self.executions is not None and self.executions.owns(
                    self.fence.record_owner_id, outcome.request_id
                ):
                    continue
                if (outcome.request_id, outcome.attempt_ordinal) not in known:
                    body.held_attempts.append(
                        pb.HeldAttempt(
                            request_id=outcome.request_id,
                            attempt_ordinal=outcome.attempt_ordinal,
                            kind=pb.ATTEMPT_KIND_JOB,
                            state=pb.ATTEMPT_STATE_OUTCOME_PENDING_ACK,
                            invocation_spec_digest=outcome.invocation_spec_digest,
                            outcome_id=outcome.outcome_id,
                            outcome_digest=outcome.outcome_digest,
                        )
                    )
        if self.accepted.mode == "serving":
            lane_wire.emit_lanes(body, self.lane_documents())
        lane_wire.emit_held_manifests(body, self.held_manifests.digests())
        return body

    def _send_snapshot(self, send: Callable[[pb.WorkerFrame], None]) -> None:
        """ONE digest-fenced `WorkerSnapshot`: what this worker holds, for its RecordOwner."""
        canonical_bytes, digest = documents.identity(self.snapshot_body())
        self.fence.snapshot_id = f"snp-{uuid.uuid4().hex[:16]}"
        held = len(self.snapshot_body().held_attempts)
        send(
            self.stamp(
                pb.WorkerFrame(
                    snapshot=pb.WorkerSnapshot(
                        snapshot_id=self.fence.snapshot_id,
                        snapshot_digest=digest,
                        snapshot_canonical_bytes=canonical_bytes,
                        # THE EXACT ACCEPTED BYTES, as recorded — never a re-serialization
                        # and never a structured projection (§4). Its digest lives INSIDE
                        # the snapshot document, so the two cannot drift apart.
                        accepted_placement_set_canonical_bytes=(
                            self.accepted.placement_set_canonical_bytes
                        ),
                    )
                )
            )
        )
        self.note(
            "snapshot",
            f"snapshot {self.fence.snapshot_id} ({len(canonical_bytes)} B, "
            f"{documents.spell(digest)[:23]}…) reports {held} held attempt(s)",
        )

    def _pump_outbound(
        self,
        epoch: int,
        send: Callable[[pb.WorkerFrame], None],
        outbound: queue.Queue[pb.WorkerFrame | None],
    ) -> None:
        """ONE stream's durable sender, draining ONE stream's queue.

        The queue is passed in rather than read off `self`: a superseded pump that keeps
        looping for a moment must drain the queue it was born with, never the live stream's.
        """
        while not self.stop.is_set() and self.fence.control_stream_epoch == epoch:
            try:
                frame = outbound.get(timeout=REPORT_SECONDS)
            except queue.Empty:
                continue  # the epoch check above is this thread's fence
            if frame is None:
                return
            try:
                send(frame)
            except Exception:
                return

    def fenced(self, message: Any) -> bool:
        """The three-field envelope, in its fixed order, BEFORE any body field (02 §0)."""
        if message.record_owner_epoch != self.fence.record_owner_epoch:
            self.note("fence", f"dropped: epoch {message.record_owner_epoch}")
            return True
        if message.control_stream_epoch != self.fence.control_stream_epoch:
            self.note(
                "fence",
                f"dropped: superseded stream epoch {message.control_stream_epoch}",
            )
            return True
        if message.worker_boot_id != self.fence.worker_boot_id:
            self.note("fence", f"dropped: wrong boot {message.worker_boot_id!r}")
            return True
        return False

    def _dispatch_frame(self, frame: pb.RecordOwnerFrame, epoch: int) -> None:
        """Apply one frame while its stream still owns the worker.

        The stream epoch is checked inside the same lock claims take. Trusting only the
        frame envelope left a check/use window in which a new claim could supersede this
        stream and the old handler could then mutate state and stamp replies as the new one.
        """
        with self.control_lock:
            if epoch != self.fence.control_stream_epoch:
                self.note("fence", f"dropped a frame from superseded stream {epoch}")
                return
            self._dispatch_current_frame(frame)

    def _dispatch_current_frame(self, frame: pb.RecordOwnerFrame) -> None:
        """The Control stream carries only the Claim; work arrives through the execution API."""
        kind = frame.WhichOneof("msg")
        if kind is not None:
            self.note("control", f"dropped a {kind}: the Control stream carries only the Claim")

    def _close(self, outcome: pb.AttemptOutcome) -> None:
        """A closed attempt leaves its lane; it does not bump `admission_epoch` (§6). Its work
        stays retained until its execution releases it."""
        owner = self.fence.record_owner_id
        ack = pb.AttemptOutcomeAck(
            request_id=outcome.request_id,
            attempt_ordinal=outcome.attempt_ordinal,
            invocation_spec_digest=outcome.invocation_spec_digest,
            outcome_id=outcome.outcome_id,
            outcome_digest=outcome.outcome_digest,
            retain_work=True,
        )
        if self.workspace is not None and self.workspace.contains(
            owner, ack.request_id, ack.attempt_ordinal
        ):
            try:
                self.workspace.acknowledge(owner, ack)
            except ValueError as exc:
                self.note("outcome", f"workspace acknowledgment refused: {exc}")
                return
        disposition = self.engine.close(ack)
        self.note("outcome", disposition)
        if disposition != "closed":
            return
        self.calls.close(ack.request_id, ack.attempt_ordinal)
        machine_slots.release(self, ack.request_id, ordinal=ack.attempt_ordinal)
        # A closed child is its parent's result: its unit settles custody now.
        self.supervisor.poke(execution_unit.key(ack.request_id))
        self._capacity_changed()

    def request_stop(self) -> None:
        """Wake every unit and the sender so the worker can leave; the host is the caller's."""
        self.stop.set()
        self.outbound.put(None)
        self.supervisor.stop()
        with self.attempts_left:
            self.attempts_left.notify_all()
        for attempt in list(self.engine.live.values()):
            attempt.poke()

    def request_restart(self) -> None:
        """A local operator requests process replacement while retaining source work."""
        if not self.stop.is_set():
            self.restart_requested = True
            self.stop.set()

    def request_idle_restart(self) -> bool:
        """An operator update cannot interrupt work admitted by a detached client."""
        with self.control_lock:
            active_executions = len(self.supervisor.keys(execution_unit.EXECUTION))
            active_attempts = sum(
                attempt.state in (*self.engine.PRE_RELEASE, "released")
                for attempt in list(self.engine.history.values())
            )
            preparing = self.preparation_lock.locked()
            accepted = not (active_executions or active_attempts or preparing)
            self.restart_sequence += 1
            document = {
                "sequence": self.restart_sequence,
                "status": "accepted" if accepted else "busy",
                "active_executions": active_executions,
                "active_attempts": active_attempts,
                "preparing": preparing,
            }
            try:
                self._publish_restart_fact("restart-status.json", document)
            except OSError as exc:
                self.note("restart_refused", f"could not record the restart verdict: {exc}")
                return False
            if accepted:
                self.request_restart()
            else:
                self.note("restart_refused", "worker update refused while execution is active")
            return accepted

    def _publish_restart_fact(self, name: str, document: dict[str, Any]) -> None:
        process = proctree.process_identity(os.getpid())
        body = {
            **document,
            "worker_boot_id": self.fence.worker_boot_id,
            "pid": process.pid,
            "started_ticks": process.started_ticks,
        }
        staged = self.root / f".{name}-{process.pid}"
        with staged.open("wb") as handle:
            handle.write(canonical.write(body))
            handle.flush()
            os.fsync(handle.fileno())
        staged.chmod(0o400)
        os.replace(staged, self.root / name)

    def schedule_idle_restart(self) -> None:
        """Signal handlers only wake the supervisor thread; they cannot reenter admission."""
        self.idle_restart_pending.set()

    def shutdown_from_stream(self) -> None:
        self.request_stop()
        self.weights.close()
        self.watches.end_all()
        self.host.stop()

    def watch(
        self,
        opened: pb.ProgressOpen,
        *,
        on_cancel: Callable[[Callable[[], None]], bool] | None = None,
    ) -> Iterator[pb.AttemptProgress]:
        """One WatchProgress call: bound to the fenced control-stream epoch (#436)."""
        if self.fenced(opened):
            return iter(())
        watch = self.watches.open(opened.request_id, opened.control_stream_epoch)
        if on_cancel is not None and not on_cancel(watch.end):
            # gRPC declines registration if cancellation already won the race.
            watch.end()
        return watch.frames(self.watches)

    # ------------------------------------------------------------- the desired state

    def _placement_from_entry(self, entry: pb.Placement, placement_set_digest: bytes) -> Placement:
        return Placement(
            entry,
            placement_set_digest,
            device_pin=self.accepted.device_pins.get(entry.placement_id),
        )

    def _apply_desired_state(self, desired: pb.DesiredWorkerState) -> None:
        """FULL REPLACE. A field absent means absent, never unchanged (§3).

        ORDER, and none of it is negotiable:

            recompute the set digest over the canonical bytes -> the launch clamp ->
            revision arithmetic -> RECORD the exact accepted bytes -> parse -> converge

        The BYTE FENCE runs before the clamp counts anything, because counting placements
        means parsing, and §4 forbids parsing a single field of a set whose bytes do not hash
        to what it claims. A mismatch is `FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH`, the
        desired state is UNAPPLIED, `accepted_desired_state_revision` does NOT advance, and
        the previous set keeps serving.
        """
        mode = desired.WhichOneof("mode") or ""
        retained_parent = None
        if mode == "placement_set" and desired.placement_set.HasField("orchestration_parent"):
            retained_parent = self.job_slots.get("orchestration")
            supplied_parent = desired.placement_set.orchestration_parent
            if (
                retained_parent is None
                or not _same_parent(supplied_parent, retained_parent.directive)
                or not supplied_parent.orchestration
                or supplied_parent.HasField("orchestration_parent")
                or supplied_parent.device_count
                or supplied_parent.resource_caps.device_required
            ):
                self.engine.fault(
                    _refused(
                        desired.revision,
                        kind=pb.FAULT_KIND_CONFIG_REFUSED,
                        reason="orchestration_parent_changed",
                        detail="serving must retain the exact already-prepared CPU parent",
                    )
                )
                return
        if (
            mode != "job"
            and self.job_slots
            and any(
                self._job_lane_busy(slot.lane)
                for name, slot in self.job_slots.items()
                if slot is not retained_parent
                and (
                    not name.startswith("cpu-")
                    or any(
                        attempt.lane_id == slot.lane.lane_id and attempt.state == "queued"
                        for attempt in self.engine.history.values()
                    )
                )
            )
        ):
            self.engine.fault(
                _refused(
                    desired.revision,
                    kind=pb.FAULT_KIND_CONFIG_REFUSED,
                    reason="job_set_busy",
                    detail="stop unpublished package attempts before replacing their job set",
                )
            )
            return
        placements: list[pb.Placement] = []
        if mode == "placement_set":
            try:
                placements = read_placement_set(desired.placement_set)
            except documents.DocumentError as exc:
                kind = (
                    pb.FaultKind.FAULT_KIND_PLACEMENT_SET_DIGEST_MISMATCH
                    if exc.code == "placement_set_digest_mismatch"
                    else pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED
                    if exc.code == REPREPARE
                    else pb.FaultKind.FAULT_KIND_CONFIG_REFUSED
                )
                self.note(
                    "desired_state",
                    f"revision {desired.revision} REFUSED before any field was parsed: "
                    f"{exc.code} — UNAPPLIED, the previous set keeps serving",
                )
                self.engine.fault(
                    _refused(
                        desired.revision,
                        kind=kind,
                        reason=exc.code,
                        detail=safe(exc.detail),
                    )
                )
                return
        body = desired_state_body(desired)
        if desired.revision < self.accepted.accepted_desired_state_revision:
            self.note("desired_state", f"revision {desired.revision} ignored (older)")
            return
        if desired.revision == self.accepted.accepted_desired_state_revision:
            if body != self.accepted.desired_state_body:
                self.note(
                    "desired_state",
                    f"PROTOCOL ERROR: revision {desired.revision} arrived with a different "
                    "desired state; a changed body must carry a new revision",
                )
                self.engine.fault(
                    _refused(
                        desired.revision,
                        kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        reason="same_revision_changed_body",
                        detail="a DesiredWorkerState body is a complete replacement; changing "
                        "it without a new revision is unrepresentable",
                    )
                )
            else:
                self.note("desired_state", f"revision {desired.revision} re-applied (idempotent)")
            return

        self.accepted.accepted_desired_state_revision = desired.revision
        self.accepted.desired_state_body = body
        self.accepted.posture = desired.posture
        self._capacity_changed()  # a drain of the superseded revision stops waiting
        # DRAINING is machine lifecycle too: a worker under a draining posture is not a
        # placement that stopped serving, it is a machine on its way out.
        if self.phase != pb.WorkerPhase.WORKER_PHASE_FAILED:
            self.set_phase(
                pb.WorkerPhase.WORKER_PHASE_DRAINING
                if desired.posture == pb.Posture.POSTURE_DRAINING
                else pb.WorkerPhase.WORKER_PHASE_ONLINE,
                f"posture {pb.Posture.Name(desired.posture)}",
            )
        # Never report a RecordOwner's future minor as one this worker applied.
        self.fence.wire_minor = min(WIRE_MINOR, desired.wire_minor or self.fence.wire_minor)
        self.accepted.drain_grace_ms = desired.drain_grace_ms
        if self.accepted.mode == "job" and mode != "job":
            self.lanes.envelope.forget("")  # the job generation's row leaves with the job
        self.accepted.mode = "serving" if mode == "placement_set" else mode
        if mode != "job" and self.job_slots:
            cpu = self.job_slots.get("orchestration")
            if cpu is not None and cpu is not retained_parent:
                cpu.supervision.close()
                self.lanes.orchestration.forget("")
            self.job_slots = {
                name: slot for name, slot in self.job_slots.items() if name.startswith("cpu-")
            }
            if retained_parent is not None:
                self.job_slots["orchestration"] = retained_parent
            self.jobs = {
                slot.binding.job_descriptor_id: slot.binding for slot in self.job_slots.values()
            }
            self.engine.jobs = self.jobs
        self.latched = ""  # a NEW revision un-latches: the latch is per desired revision
        if mode == "job":
            self.apply_job_directive(desired.job)
            return
        if mode != "placement_set":
            self.note("desired_state", "no mode: nothing to converge to")
            executor = self.supervision.current
            if executor is not None:
                try:
                    self.supervision.retire_current(executor, "empty desired state")
                except ExecutorGone as exc:
                    self.latched = f"executor_reclaim_failed: {exc}"[:200]
                    self.note("recovery", self.latched)
                    self._settle()
            if self.placement is not None:
                self._release_lane(self.placement.placement_id)
            self.placement = None
            self.pending = None
            self._retire_hosted(set())
            self.bindings = {}
            self.set_job_ready(False)
            return

        desired_set = desired.placement_set
        # RECORD THE EXACT ACCEPTED BYTES (§4) — never a re-serialization, never a
        # structured projection. Recovery, status attribution and the snapshot all quote
        # these, which is what makes "the set this worker accepted" a byte fact rather than
        # a reconstruction that is free to disagree with itself.
        self.records.append(
            "placement_set",
            {
                "worker_boot_id": self.fence.worker_boot_id,
                "accepted_desired_state_revision": desired.revision,
                "placement_set_digest": documents.spell(desired_set.placement_set_digest),
                "placement_set_canonical_b64": base64.b64encode(
                    desired_set.placement_set_canonical_bytes
                ).decode(),
            },
        )
        self.accepted.accepted_placement_set_digest = desired_set.placement_set_digest
        self.accepted.placement_set_canonical_bytes = desired_set.placement_set_canonical_bytes
        self.accepted.device_pins = lane_wire.read_device_pins(desired_set)
        previous = self.placement
        self.jobs = {}
        self.engine.jobs = {}
        if not placements:
            self.bindings = {}
            executor = self.supervision.current
            if executor is not None:
                try:
                    self.supervision.retire_current(executor, "accepted placement set is empty")
                except ExecutorGone as exc:
                    self.latched = f"executor_reclaim_failed: {exc}"[:200]
                    self.note("recovery", self.latched)
                    self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
                    self._settle()
                    return
            if self.placement is not None:
                self._release_lane(self.placement.placement_id)
            self.placement = None
            self.pending = None
            self._retire_hosted(set())
            self.accepted.converged_revision = desired.revision
            self.bump_admission("the accepted set is EMPTY; nothing is dispatchable")
            self.note("desired_state", f"revision {desired.revision} accepted: an EMPTY set")
            return
        # THE OUTGOING PLACEMENT KEEPS SERVING FROM HERE (fallback-retention, #474 clause 1).
        # Nothing below touches `self.bindings`, the prepared models, or the executor until
        # the incoming placement is materially STAGED — which is what makes the incoming's
        # every materialization failure free.
        primary_index = 0
        if previous is not None:
            primary_index = next(
                (
                    index
                    for index, row in enumerate(placements)
                    if row.placement_id == previous.placement_id
                ),
                0,
            )
        entry = placements[primary_index]
        self.pending_hosted = [
            row for index, row in enumerate(placements) if index != primary_index
        ]
        placement_id = entry.placement_id
        candidate = self._placement_from_entry(entry, desired_set.placement_set_digest)
        if (
            previous is not None
            and previous.placement_id == placement_id
            and previous.document == candidate.document
            and previous.device_pin == candidate.device_pin
            and previous.materialization == pb.MaterializationState.MATERIALIZATION_STATE_STAGED
            and previous.serving != pb.ServingState.SERVING_STATE_DRAINING
        ):
            previous.placement_set_digest = desired_set.placement_set_digest
            self.pending = None
            self.note(
                "desired_state",
                f"revision {desired.revision} keeps {placement_id!r} warm and reconciles "
                f"{len(self.pending_hosted)} additional placement(s)",
            )
            self.schedule_convergence(desired.revision)
            return
        candidate.retained_fallback_placement_set_digest = (
            previous.placement_set_digest
            if previous is not None and previous.placement_id == placement_id
            else b""
        )
        replacement = (
            previous is not None
            and previous.placement_id == candidate.placement_id
            and previous.placement_set_digest != candidate.placement_set_digest
        )
        if replacement:
            assert previous is not None
            # #485c: FALLBACK CAPABILITY IS A PRECONDITION OF REPLACEMENT, not a nicety. A
            # placement whose rollback pin is gone — evicted under a bytes/age cap, or never
            # materialized — cannot accept a new spec until it re-materializes one, so the
            # replacement PAUSES here and is recorded as such. Proceeding without a pin is an
            # explicit administrative act and there is no default that performs one. A
            # placement that is not serving has nothing to fall back to, so replacing it is a
            # standing start rather than a lost rollback.
            if (
                previous.materialization != pb.MaterializationState.MATERIALIZATION_STATE_STAGED
                and previous.serving != pb.ServingState.SERVING_STATE_OFFLINE
            ):
                self.note(
                    "desired_state",
                    f"revision {desired.revision} PAUSES the replacement of "
                    f"{previous.placement_id or '(unnamed)'}: no live rollback pin",
                )
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_FALLBACK_PIN_MISSING,
                        subject=previous.placement_id,
                        reason="rollback_pin_absent",
                        detail="the outgoing placement is "
                        f"{pb.MaterializationState.Name(previous.materialization)}, so it "
                        "cannot restore this placement if the incoming activation fails; "
                        "replacement is PAUSED until the pin re-materializes (#485c)",
                    )
                )
                return
        self.pending = candidate
        self.note(
            "desired_state",
            f"revision {desired.revision} accepted: staging placement "
            f"{candidate.placement_id or '(unnamed)'} in set "
            f"{documents.spell(candidate.placement_set_digest)[:23]}…"
            + (
                f" while {previous.placement_id or '(unnamed)'} keeps serving"
                if replacement and previous is not None
                else " from a standing start"
            ),
        )
        self.schedule_convergence(desired.revision)

    def schedule_convergence(self, revision: int) -> None:
        self.convergences.put(revision)

    def convergence_lane(self) -> None:
        """One non-control lane for slow, epoch-fenced placement convergence."""

        while (revision := self.convergences.get()) is not None:
            if revision != self.accepted.accepted_desired_state_revision:
                continue
            try:
                self.converge_placement(revision)
            except Exception as exc:
                # One revision's failure latches that revision; the lane converges the next.
                self.note("convergence", f"revision {revision} faulted: {fault_text(exc)}"[:400])
                self.latch(
                    _refused(
                        revision,
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        reason="convergence_faulted",
                        detail=safe(fault_text(exc)),
                    )
                )

    def converge_placement(self, revision: int) -> None:
        """ACQUIRE, materialize, verify, then hand off the device — or refuse to lie.

        The order is §1's, and none of it is negotiable:

            acquire (fetch by digest, build the environment, stage the plan closure)
                -> re-measure the published generation
                -> resolve the provisioned bindings
                -> HAND OFF the device (this is the only step that serializes)

        Everything before the hand-off runs while the OUTGOING placement keeps serving
        (fallback-retention #474 clauses 1-2), so an incoming failure costs zero intake
        interruption and LATCHES for this desired revision rather than looping (clause 5).

        `MATERIALIZATION_STATE_STAGED` asserts that the exact package environment installed
        and imported; it is unspeakable otherwise (§8).
        """
        if revision != self.accepted.accepted_desired_state_revision:
            return
        if self.pending is None and self.placement is not None:
            self._reconcile_hosted(revision)
            return
        desired_hosted = {
            entry.placement_id: self._placement_from_entry(
                entry, self.accepted.accepted_placement_set_digest
            )
            for entry in self.pending_hosted
        }
        candidate = self.pending or self.placement
        assert candidate is not None
        outgoing = self.placement if self.pending is not None else None
        candidate.materialization = pb.MaterializationState.MATERIALIZATION_STATE_MATERIALIZING
        if outgoing is None:
            # There is nothing to fall back to, so the incoming placement IS the reported
            # one from the first instant. An empty pod's honest report is the incoming
            # placement MATERIALIZING, never a plausible OFFLINE against no spec at all.
            self.placement = candidate
            self.pending = None
            self.bump_admission(
                f"placement {candidate.placement_id or '(unnamed)'} in set "
                f"{documents.spell(candidate.placement_set_digest)[:23]}… is materializing"
            )

        acquired = self._acquire(candidate, revision)
        if acquired is None:
            return
        if revision != self.accepted.accepted_desired_state_revision:
            self.note("materialization", f"revision {revision} was superseded after acquisition")
            return
        installed = acquired.installed
        if installed is not None:
            changed = self.supervision.use_environment(
                str(installed.python), installed.installation_id
            )
            if changed and self.supervision.current is not None:
                self.supervision.current.poisoned = "package environment generation changed"
        bindings = dict(acquired.bindings)
        if revision != self.accepted.accepted_desired_state_revision:
            self.note("materialization", f"revision {revision} was superseded before hand-off")
            return
        if outgoing is not None and self._grows_in_place(outgoing, candidate, bindings):
            self._grow_in_place(revision, outgoing, candidate, bindings)
            return

        # ---- THE DEVICE HAND-OFF. Everything above was concurrent with serving. ----------
        candidate.materialization = pb.MaterializationState.MATERIALIZATION_STATE_STAGED
        if outgoing is not None:
            self.note(
                "materialization",
                f"incoming set {documents.spell(candidate.placement_set_digest)[:23]}… is STAGED; "
                f"draining {outgoing.placement_id or '(unnamed)'} and handing off the device",
            )
            self.await_drained(revision)
            if revision != self.accepted.accepted_desired_state_revision:
                return
        # Obsolete hosted owners must release their devices before the primary
        # can fuse them. Keep unchanged tenants on disjoint devices running.
        keep = {
            identity
            for identity, placement in desired_hosted.items()
            if identity in self.hosted
            and self.hosted[identity].placement.document == placement.document
            and self.hosted[identity].placement.device_pin == placement.device_pin
        }
        if not self._retire_hosted(keep, revision):
            return
        # Retire the old owner before releasing its lane: otherwise a replacement
        # on the same pins measures the old executor's cached weights as occupied.
        if outgoing is not None and (
            outgoing.placement_id != candidate.placement_id
            or outgoing.device_pin != candidate.device_pin
        ):
            current = self.supervision.current
            if current is not None:
                try:
                    self.supervision.retire_current(current, "placement device ownership changed")
                except ExecutorGone as exc:
                    self.latched = f"executor_reclaim_failed: {exc}"[:200]
                    self._settle()
                    return
        with self.control_lock:
            if revision != self.accepted.accepted_desired_state_revision:
                return
            if outgoing is not None and (
                outgoing.placement_id != candidate.placement_id
                or outgoing.device_pin != candidate.device_pin
            ):
                self._release_lane(outgoing.placement_id)
            # THE LANE, before the executor exists to be sealed to it (cr-066): the same
            # placement id keeps its lane across a replacement; a new one is placed by fit.
            if not self._assign_lane(candidate, bindings, self.supervision):
                candidate.materialization = pb.MaterializationState.MATERIALIZATION_STATE_FAILED
                self.placement = candidate
                self.pending = None
                self.latched = "device_lane_infeasible"
                self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
                self._settle()
                return
            self.placement = candidate
            self.pending = None
            self.bindings = bindings
            self.bump_admission(
                f"placement {candidate.placement_id or '(unnamed)'} moved to set "
                f"{documents.spell(candidate.placement_set_digest)[:23]}…"
            )
            self.note(
                "desired_state",
                f"revision {revision} STAGED: placement "
                f"{candidate.placement_id or '(unnamed)'} in set "
                f"{documents.spell(candidate.placement_set_digest)[:23]}… "
                f"with {len(self.bindings)} of "
                f"{len(candidate.document.entrypoints)} binding(s) resolved, "
                f"posture={pb.Posture.Name(self.accepted.posture)}",
            )
        self.activate_executor()
        if self._placement_dispatchable(candidate.placement_id):
            self._reconcile_hosted(revision)

    def _grows_in_place(
        self, outgoing: Placement, candidate: Placement, bindings: Mapping[str, DeclaredBinding]
    ) -> bool:
        """Whether a new document for the SAME process identity updates it in place.

        Process identity is (environment, lane): the same placement id and package, the
        same environment digest, the same device pin and a lane of the same shape. Then the
        binding set changes inside the live executor — constructions load on first use and
        unreferenced ones unload once their attempts settle — and nothing drains.
        """
        executor = self.supervision.current

        def shape(rows: Mapping[str, DeclaredBinding]) -> tuple[bool, tuple[int, ...]]:
            degrees: set[int] | None = None
            for binding in rows.values():
                if binding.weightless:
                    continue
                held = set(binding.sequence_parallel_degrees())
                degrees = held if degrees is None else degrees & held
            model_bearing = any(not binding.weightless for binding in rows.values())
            return model_bearing, tuple(sorted(degrees or ()))

        return bool(
            outgoing.placement_id == candidate.placement_id
            and outgoing.document.package == candidate.document.package
            and outgoing.document.development == candidate.document.development
            and outgoing.installation_id == candidate.installation_id
            and outgoing.device_pin == candidate.device_pin
            and outgoing.materialization == pb.MaterializationState.MATERIALIZATION_STATE_STAGED
            and outgoing.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
            and executor is not None
            and executor.started
            and executor.alive()
            and not executor.poisoned
            and shape(self.bindings) == shape(bindings)
        )

    def _grow_in_place(
        self,
        revision: int,
        outgoing: Placement,
        candidate: Placement,
        bindings: dict[str, DeclaredBinding],
    ) -> None:
        """Swap the placement's bindings inside the live executor (proto-061 D2).

        Under the control lock and with admission OPEN: the admission epoch does not move,
        the placement stays DISPATCHABLE, an offer queued under the superseded document is
        still admitted for a binding whose construction is unchanged, and a new binding over
        a construction the process already built is loaded from the moment it exists.
        """
        placement_id = candidate.placement_id
        with self.control_lock:
            if revision != self.accepted.accepted_desired_state_revision:
                return
            executor = self.supervision.current
            assert executor is not None
            before = {
                digest: binding.construction_key() for digest, binding in self.bindings.items()
            }
            self._superseded[placement_id] = {documents.spell(outgoing.bindings_digest): before}
            kept = {
                digest
                for digest, binding in bindings.items()
                if before.get(digest) == binding.construction_key()
            }
            candidate.lane_id = outgoing.lane_id
            candidate.materialization = pb.MaterializationState.MATERIALIZATION_STATE_STAGED
            candidate.serving = outgoing.serving
            candidate.installed = candidate.installed or outgoing.installed
            self.placement = candidate
            self.pending = None
            self.bindings = bindings
            self.failed_bindings = {
                digest: why for digest, why in self.failed_bindings.items() if digest in kept
            }
            executor.ready_bindings = set(bindings) - set(self.failed_bindings)
            for key, row in executor.loaded.items():
                row.bindings = frozenset(
                    digest
                    for digest in row.bindings
                    if digest in bindings and bindings[digest].construction_key() == key
                )
            for digest, binding in bindings.items():
                loaded = executor.loaded.get(binding.construction_key())
                if (
                    loaded is not None
                    and {m.model_parameter_name for m in binding.model_bindings()}
                    <= loaded.parameters
                ):
                    loaded.bindings = loaded.bindings | {digest}
            wanted = {binding.construction_key() for binding in bindings.values()}
            stale = sorted(key for key in executor.loaded if key not in wanted)
            served = {digest for digest, key in before.items() if key in stale}
            added = sorted(set(bindings) - set(before))
            removed = sorted(set(before) - set(bindings))
            rebound = sorted(set(bindings) & set(before) - kept)
            self.note(
                "desired_state",
                f"revision {revision} updated {placement_id!r} IN PLACE: +{len(added)} "
                f"-{len(removed)} ~{len(rebound)} binding(s), {len(wanted)} construction(s) "
                "wanted, "
                f"{len(stale)} to unload; epoch {executor.epoch} pid {executor.pid} kept, "
                f"admission stays OPEN at epoch {self.admission_epoch}",
            )
        self._warm_pages(self._tenant(placement_id))
        if stale:
            self._unload_when_settled(placement_id, executor, stale, served)
        self._reconcile_hosted(revision)

    def _unload_when_settled(
        self, placement_id: str, executor: child.Executor, keys: list[str], served: set[str]
    ) -> None:
        """Unload constructions no binding references once the attempts of the bindings
        they served (`served`, before the update) have settled."""

        def held() -> bool:
            return any(
                attempt.placement_id == placement_id
                and attempt.state in self.engine.HELD
                and str((attempt.spec.get("serving") or {}).get("entrypoint_binding_digest", ""))
                in served
                for attempt in list(self.engine.history.values())
            )

        def run() -> None:
            with self.attempts_left:
                self.attempts_left.wait_for(lambda: not held() or self.stop.is_set())
            if self.stop.is_set():
                return
            lane = self._lane_of(placement_id)
            for key in keys:
                label = key.removeprefix("sha256:")[:12]
                with lane.device:
                    with self.control_lock:
                        referenced = any(
                            binding.construction_key() == key for binding in self.bindings.values()
                        )
                        if (
                            referenced
                            or self.supervision.current is not executor
                            or key not in executor.loaded
                        ):
                            continue
                    try:
                        reply = executor.call(Unload(construction=key), timeout=None)
                    except ExecutorGone as exc:
                        self.note(
                            "residency",
                            f"{placement_id!r} lost the executor unloading {label}: {exc}"[:256],
                        )
                        return
                    with self.control_lock:
                        executor.loaded.pop(key, None)
                        if executor.active == key:
                            executor.active = ""
                        if reply.get("ok"):
                            lane.row(placement_id).ledger.observe_vacate(
                                {**reply, "vacated": {"freed_bytes": reply.get("freed_bytes", 0)}},
                                f"unloaded {label}",
                            )
                if not reply.get("ok"):
                    self.note(
                        "residency",
                        f"{placement_id!r} refused to unload {label}: {reply.get('code')}: "
                        f"{reply.get('detail', '')}"[:256],
                    )
                    continue
                self.note(
                    "residency",
                    f"{placement_id!r} construction {label} unloaded "
                    f"(freed {int(reply.get('freed_bytes', 0))} B)",
                )

        threading.Thread(
            target=self._lane(f"unload:{placement_id}", run, fatal=False),
            name=f"unload:{placement_id}",
            daemon=True,
        ).start()

    def _retire_hosted(self, keep: set[str], revision: int | None = None) -> bool:
        """Remove only placements absent from the complete desired set."""
        return all(
            self._retire_one(placement_id, revision)
            for placement_id in sorted(set(self.hosted) - keep)
        )

    def _retire_one(self, placement_id: str, revision: int | None = None) -> bool:
        """Drain one hosted placement, reclaim its executor and free its lane."""

        def superseded() -> bool:
            return (
                revision is not None and revision != self.accepted.accepted_desired_state_revision
            )

        hosted = self.hosted.get(placement_id)
        if hosted is None:
            return True
        if superseded():
            return False
        self._hosted_set_serving(
            placement_id,
            pb.ServingState.SERVING_STATE_DRAINING,
            f"placement {placement_id!r} removed",
        )

        def draining() -> bool:
            return any(
                attempt.placement_id == placement_id and attempt.state in self.engine.HELD
                for attempt in list(self.engine.history.values())
            )

        with self.attempts_left:
            self.attempts_left.wait_for(
                lambda: not draining() or superseded() or self.stop.is_set()
            )
        if self.stop.is_set():
            return False
        if superseded():
            return False
        current = hosted.supervision.current
        if current is not None:
            try:
                hosted.supervision.retire_current(current, "placement removed")
            except ExecutorGone as exc:
                hosted.latched = f"executor_reclaim_failed: {exc}"[:200]
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_LOCAL_SAFETY_REFUSAL,
                        subject=placement_id,
                        reason="executor_reclaim_failed",
                        detail=safe(str(exc)),
                    )
                )
                return False
        with self.control_lock:
            if superseded():
                return False
            hosted.supervision.close()
            self.hosted.pop(placement_id, None)
            self._release_lane(placement_id)
            # No second bump: the DRAINING edge above moved the fence, and an offer naming
            # the gone placement is refused `unknown_placement` on its own.
            self.note("admission", f"placement {placement_id!r} removed")
        return True

    def _reconcile_hosted(self, revision: int) -> None:
        """Converge the additional placements without disturbing the primary one."""

        if revision != self.accepted.accepted_desired_state_revision:
            return
        desired = {
            entry.placement_id: self._placement_from_entry(
                entry, self.accepted.accepted_placement_set_digest
            )
            for entry in self.pending_hosted
        }
        changed = {
            placement_id
            for placement_id, placement in desired.items()
            if placement_id in self.hosted
            and (
                self.hosted[placement_id].placement.document != placement.document
                or self.hosted[placement_id].placement.device_pin != placement.device_pin
            )
        }
        if not self._retire_hosted(set(desired) - changed, revision):
            return
        for placement_id, placement in desired.items():
            if revision != self.accepted.accepted_desired_state_revision:
                return
            hosted = self.hosted.get(placement_id)
            if hosted is not None:
                hosted.placement.placement_set_digest = self.accepted.accepted_placement_set_digest
                if hosted.placement.serving == pb.ServingState.SERVING_STATE_DRAINING:
                    self._prepare_hosted_executor(hosted)
                continue
            hosted = HostedPlacement(
                placement=placement,
                supervision=self._new_placement_supervision(placement_id),
            )
            self.hosted[placement_id] = hosted
            self._activate_hosted(hosted, revision)
        with self.control_lock:
            if revision != self.accepted.accepted_desired_state_revision:
                return
            self.pending_hosted = []
            if (
                self.placement is not None
                and self._placement_dispatchable(self.placement.placement_id)
                and all(
                    hosted.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
                    for hosted in self.hosted.values()
                )
            ):
                self.accepted.converged_revision = revision
            self._settle()

    def _activate_hosted(self, hosted: HostedPlacement, revision: int | None) -> None:
        """Acquire, place and activate one hosted placement in its own uv environment.

        A machine replica passes no revision: no desired state supersedes it."""

        placement = hosted.placement
        placement.materialization = pb.MaterializationState.MATERIALIZATION_STATE_MATERIALIZING
        acquired = self._acquire(placement, revision or 0)
        if acquired is None or (
            revision is not None and revision != self.accepted.accepted_desired_state_revision
        ):
            return
        bindings = dict(acquired.bindings)
        if not bindings:
            # A placement with nothing to serve. Model-bearing hosted placements are admitted
            # (cr-022): the lane assignment below places them by measured fit — another lane
            # of their own, co-resident beside a tenant that leaves room, or time-sliced on
            # a shared lane — and the generation prepare is the primary's own.
            placement.materialization = pb.MaterializationState.MATERIALIZATION_STATE_FAILED
            hosted.latched = "placement_without_bindings"
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    subject=placement.placement_id,
                    reason=hosted.latched,
                    detail="the placement resolved no entrypoint binding to serve",
                )
            )
            self.note(
                "materialization",
                f"{placement.placement_id!r} refused without disturbing other placements: "
                f"{hosted.latched}",
            )
            self._settle()
            return
        installed = acquired.installed
        if installed is not None:
            hosted.supervision.use_environment(str(installed.python), installed.installation_id)
        hosted.bindings = bindings
        hosted.failed_bindings = {}
        with self.control_lock:
            assigned = self._assign_lane(placement, bindings, hosted.supervision)
        if not assigned:
            placement.materialization = pb.MaterializationState.MATERIALIZATION_STATE_FAILED
            hosted.latched = "device_lane_infeasible"
            self._settle()
            return
        placement.materialization = pb.MaterializationState.MATERIALIZATION_STATE_STAGED
        self._prepare_hosted_executor(hosted)

    def _prepare_hosted_executor(self, hosted: HostedPlacement, *, fresh: bool = True) -> None:
        """ACTIVATE one hosted placement through the generation prepare the primary uses.

        The serving edge is the hosted placement's own; the worker-level latch is not
        touched — a hosted placement that cannot prepare latches itself and the others keep
        serving. The author's `Model.warm` ran inside its prepare, as for every placement.
        """
        placement = hosted.placement
        self._hosted_set_serving(
            placement.placement_id,
            pb.ServingState.SERVING_STATE_ACTIVATING,
            f"activating placement {placement.placement_id!r}",
        )
        if fresh:
            hosted.failed_bindings.clear()
        latch, _reused = self._prepare_generation(self._tenant(placement.placement_id))
        if latch:
            hosted.latched = latch
            self._hosted_set_serving(
                placement.placement_id,
                pb.ServingState.SERVING_STATE_OFFLINE,
                f"placement {placement.placement_id!r} latched: {latch}"[:200],
            )
            self.note(
                "boot",
                f"{placement.placement_id!r} activation failed; other placements stay live: "
                f"{latch}",
            )
            self._settle()
            return
        hosted.latched = ""
        self._hosted_set_serving(
            placement.placement_id,
            pb.ServingState.SERVING_STATE_DISPATCHABLE,
            f"placement {placement.placement_id!r} dispatchable",
        )

    def await_drained(self, revision: int) -> None:
        """Step 2 of the hand-off: admission stops, ALREADY-ACCEPTED attempts settle.

        NO DRAIN CONSTANT EXISTS (#434/#479). This waits on the attempts the RecordOwner
        already accepted and on THEIR own deadlines — nothing else. Cutting the tail short is
        `force`, an administrative act with a recorded actor, and there is no default number
        in any config file that performs one.
        """
        self.set_serving(pb.ServingState.SERVING_STATE_DRAINING, "handing off the device")
        for attempt in list(self.engine.history.values()):
            while (
                self.placement is not None
                and attempt.placement_id == self.placement.placement_id
                and attempt.state in self.engine.HELD
            ):
                if revision != self.accepted.accepted_desired_state_revision or self.stop.wait(
                    0.02
                ):
                    return

    def _acquire(self, placement: Placement, revision: int) -> acquire.AcquiredPlacement | None:
        """Fetch what the spec reaches and join its receipt to content-keyed installed bytes.

        Returns the installed generation, or None having already typed the fault and left the
        outgoing placement serving.
        """
        prepared_environment = next(
            (
                prepared.installed
                for prepared in self.prepared_installations.values()
                if prepared.placement_id == placement.placement_id
                and prepared.installation_id == placement.installation_id
            ),
            None,
        )
        if (
            self.options.install_root is None and self.options.environment_python is None
        ) or self.options.tensorfs_root is None:
            self._materialization_failed(
                placement,
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    subject=placement.placement_id,
                    reason="environment_materialization_input_missing",
                    detail="this worker was launched without --install-root or "
                    "--environment-python and --tensorfs-root; the worker will not "
                    "guess where the worker put package environments or TensorFS objects",
                ),
                terminal=False,
            )
            return None
        if self.options.artifact_cache is None:
            self._materialization_failed(
                placement,
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    subject=placement.placement_id,
                    reason="artifact_cache_absent",
                    detail="--artifact-cache is the one local landing area for exact "
                    "PlacementSet references",
                ),
                terminal=False,
            )
            return None
        try:
            installed_environment = prepared_environment
            if (
                installed_environment is None
                and self.options.install_root is None
                and self.options.environment_python is not None
            ):
                installed_environment = package_installation.open_existing_environment(
                    self.options.environment_python,
                    placement.installation_id,
                    package=acquire.package_selection(placement.document).package,
                    release=acquire.package_selection(placement.document).release,
                )
            install_root = self.options.install_root or (
                installed_environment.generation if installed_environment is not None else None
            )
            assert install_root is not None
            acquirer = acquire.Acquirer(
                cache_root=self.options.artifact_cache,
                install_root=install_root,
                selection_root=self.options.artifact_cache / "selections",
                tensorfs_root=self.options.tensorfs_root,
                python=self.options.environment_python or Path(sys.executable),
                base=self.base,
                installed_environment=installed_environment,
                verified_cache=self.verified_artifacts,
                held_manifests=self.held_manifests,
                on_observation=lambda value: self._observe_acquisition(placement, value),
            )

            acquired = acquirer.acquire(placement.document)
            self.note(
                "materialization",
                f"revision {revision}: {acquired.manifests} manifest(s) held; "
                f"{acquired.fetched_bytes} byte(s) new; "
                + (
                    f"environment {acquired.installed.installation_id} at "
                    f"{acquired.installed.generation}"
                    if acquired.installed is not None
                    else "source-checkout environment"
                ),
            )
            return acquired
        except acquire.AcquisitionRefusal as exc:
            self._materialization_failed(
                placement,
                pb.Fault(
                    kind=exc.kind,
                    subject=placement.placement_id,
                    reason=exc.code,
                    detail=safe(exc.detail),
                ),
                terminal=exc.kind in (pb.FaultKind.FAULT_KIND_ARTIFACT_DIGEST_MISMATCH,),
            )
            return None
        except (OSError, package_environment.EnvironmentRefusal) as exc:
            reason = (
                exc.code if isinstance(exc, package_environment.EnvironmentRefusal) else "io_error"
            )
            self._materialization_failed(
                placement,
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED
                    if isinstance(exc, package_environment.EnvironmentRefusal)
                    else pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                    subject=placement.placement_id,
                    reason=reason,
                    detail=safe(str(exc)),
                ),
                terminal=isinstance(exc, package_environment.EnvironmentRefusal),
            )
            return None

    def _observe_acquisition(
        self, placement: Placement, value: pb.PlacementAcquisitionObservation
    ) -> None:
        """Publish observation-only partial/terminal progress; never decide convergence."""

        placement.acquisition.CopyFrom(value)

    def _materialization_failed(
        self, placement: Placement, fault: pb.Fault, *, terminal: bool
    ) -> None:
        """One exit for every materialization refusal, and it never serves-nothing silently.

        `terminal` is the ABSENT/FAILED distinction (§8): a fault about the ARTIFACTS —
        a wrong digest, a receipt that did not match — is a `FAILED(cause)` verdict about
        this placement; a fault about ACCESS or a missing local input is `ABSENT`, because
        a grant refresh or a provisioner repair can still clear it.
        """
        placement.materialization = (
            pb.MaterializationState.MATERIALIZATION_STATE_FAILED
            if terminal
            else pb.MaterializationState.MATERIALIZATION_STATE_ABSENT
        )
        hosted = self.hosted.get(placement.placement_id)
        if hosted is not None and hosted.placement is placement:
            placement.serving = pb.ServingState.SERVING_STATE_OFFLINE
            hosted.latched = f"{pb.FaultKind.Name(fault.kind)}: {fault.reason}"[:200]
            self.engine.fault(fault)
            self.note(
                "materialization",
                f"{placement.placement_id!r} refused ({fault.reason}); other placements "
                "remain unchanged",
            )
            self._settle()
            return
        if self.placement is placement:
            # No outgoing placement exists to keep serving: latch on this placement, which moves
            # its serving axis OFFLINE and leaves `converged_revision` behind, typed.
            self.note(
                "materialization",
                f"placement REFUSED ({fault.reason}): {safe(fault.detail)}",
            )
            self.latch(fault)
            return
        # FALLBACK-RETENTION (#474 clauses 1, 4 and 5): the outgoing placement never stopped
        # serving, the failure is typed against the incoming set, and it LATCHES for this
        # desired revision. Retry is a RecordOwner act — a NEW revision, or a grant refresh.
        self.latched = f"{pb.FaultKind.Name(fault.kind)}: {fault.reason}"[:200]
        self.engine.fault(fault)
        self.note(
            "materialization",
            f"incoming set REFUSED ({fault.reason}); "
            f"{self.placement.placement_id if self.placement else '(none)'} keeps serving and "
            "the failure LATCHES for this desired revision",
        )
        self._settle()

    def apply_job_directive(self, job: pb.JobDirective) -> None:
        """Apply a complete ordinary-job plus optional CPU-parent admission set."""
        from cozy_runtime.internal.worker.job_slots import JobSlot

        has_parent = job.HasField("orchestration_parent")
        if not job.orchestration and not has_parent and not self.job_slots:
            self._apply_ordinary_job_directive(job)
            return
        try:
            if job.orchestration and has_parent:
                raise ValueError("orchestration takes one CPU parent")
            parent = job if job.orchestration else job.orchestration_parent if has_parent else None
            if parent is not None and (
                not parent.orchestration or parent.HasField("orchestration_parent")
            ):
                raise ValueError("orchestration parent cannot be ordinary or nested")
            if parent is not None:
                from cozy_runtime.internal import job_plan

                record = job_plan.path(
                    self.config.cozy_home / "job-plans",
                    parent.installation_id,
                    parent.job_descriptor_id,
                )
                binding = JobBinding.read(json.loads(record.read_text()))
                declaration = self.engine._job_declaration(binding, parent.job_descriptor_id)
                if (
                    parent.device_count
                    or parent.resource_caps.device_required
                    or parent.resource_caps.max_device_memory_bytes
                    or declaration.get("models")
                    or declaration.get("weights_outputs")
                ):
                    raise ValueError("CPU orchestration cannot hold a Model, weights, or GPU grant")
                if not job.orchestration and parent.job_descriptor_id == job.job_descriptor_id:
                    raise ValueError("one job cannot occupy the parent and ordinary slots together")
            old_cpu = self.job_slots.get("orchestration")
            cpu_same = bool(
                parent is not None
                and old_cpu is not None
                and _same_parent(old_cpu.directive, parent)
            )
            old_gpu = self.job_slots.get("envelope")
            ordinary = pb.JobDirective()
            ordinary.CopyFrom(job)
            ordinary.ClearField("orchestration_parent")
            gpu_same = bool(
                not job.orchestration
                and old_gpu is not None
                and old_gpu.directive == ordinary
                and old_gpu.ready()
            )
            if not cpu_same and self._job_lane_busy(self.lanes.orchestration):
                raise ValueError("stop the active CPU parent before replacing or removing it")
            if not gpu_same and self._job_lane_busy(self.lanes.envelope):
                raise ValueError("stop the active ordinary job before replacing or removing it")
            if any(
                attempt.kind != "job" and attempt.state in self.engine.PRE_RELEASE
                for attempt in self.engine.history.values()
            ):
                raise ValueError(
                    "stop active serving attempts before entering the unpublished package job set"
                )

            if not cpu_same and old_cpu is not None:
                old_cpu.supervision.close()
                self.job_slots.pop("orchestration", None)
                self.lanes.orchestration.forget("")
            if parent is not None and not cpu_same:
                supervision = self._new_orchestration_supervision()
                supervision.use_environment(binding.python, binding.installation_id)
                copied = pb.JobDirective()
                copied.CopyFrom(parent)
                self.job_slots["orchestration"] = JobSlot(
                    copied, binding, supervision, self.lanes.orchestration
                )
                executor = supervision.spawn(imposed=self.imposed(self.lanes.orchestration))
                executor.reserved_for_job = True
            if not job.orchestration:
                if not gpu_same:
                    self._apply_ordinary_job_directive(ordinary)
                    selected = self.jobs.get(ordinary.job_descriptor_id)
                    if selected is not None:
                        self.job_slots["envelope"] = JobSlot(
                            ordinary, selected, self.supervision, self.lanes.envelope
                        )
            else:
                ordinary_executor = self.supervision.current
                if ordinary_executor is not None:
                    self.supervision.retire_current(
                        ordinary_executor, "CPU orchestration holds no ordinary job executor"
                    )
                self.job_slots.pop("envelope", None)
                self.bindings = {}
                if self.placement is not None:
                    self._release_lane(self.placement.placement_id)
                self.placement = None
                self.pending = None
                self._retire_hosted(set())
            self.jobs = {
                slot.binding.job_descriptor_id: slot.binding for slot in self.job_slots.values()
            }
            self.engine.jobs = self.jobs
            self.set_job_ready(
                bool(self.job_slots)
                and all(slot.ready() for slot in self.job_slots.values())
                and not self.latched
            )
            self._settle()
        except (OSError, ValueError, ExecutorGone) as exc:
            self.latched = "job_set_refused: " + safe(str(exc))
            self.engine.fault(
                pb.Fault(
                    kind=pb.FAULT_KIND_CONFIG_REFUSED,
                    subject=job.job_descriptor_id,
                    reason="job_set_refused",
                    detail=safe(str(exc)),
                )
            )
            self._settle()

    def _job_lane_busy(self, lane: DeviceLane) -> bool:
        return any(
            attempt.lane_id == lane.lane_id and attempt.state in self.engine.PRE_RELEASE
            for attempt in self.engine.history.values()
        )

    def _new_orchestration_supervision(self) -> ExecutorSupervision:
        supervision = ExecutorSupervision(
            root=self.root / "placements" / "orchestration",
            python=self.options.python or sys.executable,
            base_env=self.config.child_base_env,
            cozy_home=self.config.cozy_home,
            executor_uid=self._slot_executor_uid("orchestration"),
            executor_gid=self.options.executor_gid,
            jit_pod_scope=f"{self.fence.worker_boot_id}-orchestration",
            kernel_cache=self.config.kernel_cache,
        )
        supervision.on_change = lambda executor: self.lanes.orchestration.row(
            ""
        ).ledger.begin_generation(executor.pid if executor else 0)
        supervision.on_invalidate = lambda why: self._orchestration_unavailable(why)
        supervision.on_exit = lambda executor: self._executor_died(
            self.lanes.orchestration, executor, ""
        )
        supervision.on_lane_failure = self.fail_machine
        supervision.sweep_orphans()
        return supervision

    def _orchestration_unavailable(self, why: str) -> None:
        self.set_job_ready(False)
        self.note("orchestration", "capacity withdrawn: " + safe(why))
        self._settle()

    def _rebuild_orchestration(self) -> None:
        with self.control_lock:
            slot = self.job_slots.get("orchestration")
            if slot is None or slot.ready() or self.stop.is_set():
                return
            try:
                if slot.supervision.current is not None:
                    slot.supervision.retire_current(
                        slot.supervision.current, "rebuild stopped CPU parent"
                    )
                executor = slot.supervision.spawn(imposed=self.imposed(slot.lane))
                executor.reserved_for_job = True
                self.set_job_ready(all(value.ready() for value in self.job_slots.values()))
            except ExecutorGone as exc:
                self.latched = "orchestration_executor_absent: " + safe(str(exc))
            self._settle()

    def _apply_ordinary_job_directive(self, job: pb.JobDirective) -> None:
        """The JOB lane: resolve the descriptor, spawn a fresh executor, become dispatchable.

        Job mode carries no placement, so its dispatchability edge is `job_ready` and its
        capacity is `JobCapacity` — the placement axes are unspeakable here and stay unset
        rather than being filled with a plausible OFFLINE.
        """
        self.bindings = {}
        self.jobs = {}
        previous = self.supervision.current
        if previous is not None:
            try:
                # Retired while the serving placement still names its lane, so the
                # retirement clears THAT slot's row; the envelope lane's row starts clean.
                self.supervision.retire_current(
                    previous, "entering job mode retires the serving generation"
                )
            except ExecutorGone as exc:
                self.note("boot", f"the serving executor could not be reclaimed: {exc}")
                self.latched = f"job_executor_reclaim_failed: {exc}"[:200]
                self._settle()
                return
        if self.placement is not None:
            self._release_lane(self.placement.placement_id)
        self.placement = None
        self.pending = None
        self._retire_hosted(set())
        self.set_job_ready(False)
        plans = self.config.cozy_home / "job-plans"
        from cozy_runtime.internal import job_plan

        try:
            record = job_plan.path(plans, job.installation_id, job.job_descriptor_id)
        except ValueError:
            self.latched = "job_plan_identity_invalid"
            self._settle()
            return
        if not record.is_file():
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    subject=job.job_descriptor_id,
                    reason="unresolved_job_descriptor",
                    detail=f"no local record for {job.job_descriptor_id}",
                )
            )
            self.latched = "unresolved_job_descriptor"
            self._settle()
            return
        # The plan is found by the directive's own installation and descriptor ids, and
        # the directive's reclaim policy is applied below as sent.
        try:
            binding = JobBinding.read(json.loads(record.read_text()))
        except (OSError, ValueError, TypeError, PlanRefusal) as exc:
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    subject=job.job_descriptor_id,
                    reason="job_plan_unreadable",
                    detail=f"re-prepare the package: {type(exc).__name__}: {exc}"[:512],
                )
            )
            self.latched = "job_plan_unreadable"
            self._settle()
            return
        self.supervision.use_environment(binding.python, binding.installation_id)
        self.jobs[job.job_descriptor_id] = binding
        self.engine.jobs = self.jobs
        self.engine.job_device_count = job.device_count
        self.job_caps = job.resource_caps
        self.note(
            "desired_state",
            f"revision {self.accepted.accepted_desired_state_revision} accepted in JOB mode: "
            f"{self.jobs[job.job_descriptor_id].job!r}, device floor {job.device_count}",
        )
        try:
            # A JOB IS SEALED TO THE WHOLE ENVELOPE: the group lane, one seat (cr-018's
            # door; `device_count` is a floor, never concurrency).
            executor = self.supervision.spawn(imposed=self.imposed(self.lanes.envelope))
        except ExecutorGone as exc:
            self.note("boot", f"the job executor never dialled back: {exc}")
            # The CAUSE rides the latch. `job_executor_absent` alone names the shape and
            # nothing an operator can act on, and the latch is what reaches the owner.
            self.latched = f"job_executor_absent: {exc}"[:200]
            self._settle()
            return
        executor.reserved_for_job = True
        self.note(
            "boot",
            f"job executor epoch {executor.epoch} pid {executor.pid} dialled "
            f"back in {(executor.ready_at - executor.spawned_at) * 1000:.0f} ms "
            f"(torch_loaded={executor.hello.get('torch_loaded')})",
        )
        self.set_job_ready(True)

    # ------------------------------------------------------------- activation

    def activate_executor(self, *, fresh: bool = True) -> None:
        """Contain every activation defect as observable OFFLINE state, never a dead lane."""

        try:
            self.prepare_executor(fresh=fresh)
        except Exception as exc:
            detail = safe(f"{type(exc).__name__}: {exc}")[:1024]
            subject = self.placement.placement_id if self.placement is not None else "placement"
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    subject=subject,
                    reason="activation_exception",
                    detail=detail,
                )
            )
            self.latched = "activation_exception"
            self.note("boot", f"activation failed after staging: {detail}"[:256])
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
            self._settle()

    def prepare_executor(self, *, fresh: bool = True) -> None:
        """ACTIVATION: prepare the executor BEFORE the placement becomes DISPATCHABLE.

        This is the SERVING axis moving OFFLINE -> ACTIVATING -> DISPATCHABLE. Serving never
        accepts-then-cold-loads, which is the whole reason the axis exists separately from
        materialization: bytes on disk and a process that will take work are two facts, and
        one enum could not hold both.
        """
        if not self.bindings:
            executor = self.supervision.current
            if executor is not None:
                try:
                    self.supervision.retire_current(executor, "placement has no bindings")
                except ExecutorGone as exc:
                    self.latched = f"executor_reclaim_failed: {exc}"[:200]
                    self.note("recovery", self.latched)
                    self._settle()
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, "no bindings")
            return
        assert self.placement is not None
        # A NEW desired revision re-earns every verdict. A rebuild does not: a construction
        # whose load refused replaced the process, and is not tried again until the desired
        # state changes.
        if fresh:
            self.failed_bindings.clear()
        self.set_serving(pb.ServingState.SERVING_STATE_ACTIVATING)
        latch, reused = self._prepare_generation(self._tenant(self.placement.placement_id))
        if latch:
            self.latched = latch
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, latch[:200])
            self._settle()
            return
        if reused:
            # The high-water mark moves BEFORE the edge is published. `set_serving`
            # reports immediately — dispatchability is an edge — and `converged_revision`
            # CLAMPS itself back until serving actually says DISPATCHABLE, so writing it
            # first cannot publish an early claim, while writing it second published one
            # frame that said DISPATCHABLE and not-converged in the same breath.
            self.accepted.converged_revision = self.accepted.accepted_desired_state_revision
            self.set_serving(pb.ServingState.SERVING_STATE_DISPATCHABLE)
            return
        if self.failed_bindings:
            self.note(
                "boot",
                f"{len(self.bindings) - len(self.failed_bindings)} of {len(self.bindings)} "
                f"binding(s) prepared; faulted: " + ", ".join(sorted(self.failed_bindings)),
            )
        self._finish_activation()

    def _tenant(self, placement_id: str) -> Tenant:
        """The prepare's view of one placement's slot: the primary's or a hosted one's."""
        hosted = self.hosted.get(placement_id)
        if hosted is None:
            assert self.placement is not None and self.placement.placement_id == placement_id
            return Tenant(
                placement=self.placement,
                supervision=self.supervision,
                bindings=self.bindings,
                failed_bindings=self.failed_bindings,
                lane=self._primary_lane(),
                slot=self._primary_slot(),
            )
        return Tenant(
            placement=hosted.placement,
            supervision=hosted.supervision,
            bindings=hosted.bindings,
            failed_bindings=hosted.failed_bindings,
            lane=self._lane_of(placement_id),
            slot=placement_id,
        )

    def _prepare_generation(self, tenant: Tenant) -> tuple[str, bool]:
        """ONE placement's PROCESS: spawn or keep its executor and START it (proto-061 D2).

        Process identity is (environment, lane). Nothing here fills: `start` brings every
        rank up and imports torch, the runtime and the package, and the placement is
        DISPATCHABLE when it returns. Constructions load on the attempt path, under the
        lane's device lock, the first time a binding needs one (`_ensure_construction`).

        Returns `(latch, reused)`: an empty latch means the process is started; `reused`
        means an already started process of the same seal serves the placement unchanged.
        """
        placement_id = tenant.placement.placement_id
        wanted = {digest for digest in tenant.bindings if digest not in tenant.failed_bindings}
        executor = tenant.supervision.current
        if (
            executor is not None
            and executor.started
            and executor.alive()
            and not executor.poisoned
            and not executor.reserved_for_job
            and executor.sealed_devices == tenant.lane.devices
        ):
            executor.ready_bindings = set(wanted)
            self._load_weightless(tenant, executor)
            self.note(
                "boot",
                f"{placement_id!r} reused started epoch {executor.epoch} pid {executor.pid} "
                f"({len(wanted)} binding(s), {len(executor.loaded)} construction(s) built)",
            )
            return ("", True) if wanted else ("prepare refused: every binding", False)
        owned = self._own_executor(tenant)
        if isinstance(owned, str):
            return owned, False
        started = self._start_executor(tenant, owned)
        if started:
            return started, False
        owned.ready_bindings = set(wanted)
        self._load_weightless(tenant, owned)
        self._warm_pages(tenant)
        wanted -= set(tenant.failed_bindings)
        return ("", False) if wanted else ("prepare refused: every binding", False)

    def _load_weightless(self, tenant: Tenant, executor: child.Executor) -> None:
        """A weightless construction fills nothing, so it is built with the process: its
        bindings are served from the first attempt without a load on the attempt path."""
        groups: dict[str, list[DeclaredBinding]] = {}
        for digest, binding in tenant.bindings.items():
            if binding.weightless and digest not in tenant.failed_bindings:
                groups.setdefault(binding.construction_key(), []).append(binding)
        for key, group in groups.items():
            try:
                if key not in executor.loaded:
                    self._load_construction(tenant, executor, key, group)
                if not executor.active:
                    self._activate_construction(tenant, executor, key)
            except AttemptRefusal as exc:
                self.note("boot", f"weightless construction refused: {exc.code}"[:256])
                return

    def _own_executor(self, tenant: Tenant) -> child.Executor | str:
        """Spawn, keep or replace this tenant's executor. Returns it, or a latch string.

        A live, unpoisoned process sealed to this lane is kept whatever it has loaded:
        constructions are loaded into it, never by replacing it. A seal is set once, at
        spawn, so a lane of another width is an ordinary replacement (cr-068), and a
        process that ran a job is never reused for serving. Nothing here fills.
        """
        lane = tenant.lane
        placement_id = tenant.placement.placement_id
        executor = tenant.supervision.current
        try:
            if executor is None:
                self.note(
                    "boot",
                    f"{placement_id!r}: spawning the device executor (exec-first trampoline) "
                    f"on {lane.lane_id}",
                )
                executor = tenant.supervision.spawn(imposed=self.imposed(lane))
            elif (
                executor.alive()
                and not executor.poisoned
                and not executor.reserved_for_job
                and executor.sealed_devices == lane.devices
            ):
                self.note(
                    "boot",
                    f"{placement_id!r}: starting epoch {executor.epoch} pid {executor.pid}",
                )
            else:
                self.note(
                    "boot",
                    f"{placement_id!r}: replacing the device executor "
                    f"({executor.poisoned or 'sealed to other devices or exited'})"[:256],
                )
                with lane.device:
                    executor = tenant.supervision.replace(
                        executor,
                        imposed=self.imposed(lane),
                        why=executor.poisoned or "executor seal or process changed",
                    )
            self.note(
                "boot",
                f"{placement_id!r} epoch {executor.epoch} pid {executor.pid} dialled back in "
                f"{(executor.ready_at - executor.spawned_at) * 1000:.0f} ms "
                f"(no_new_privs={executor.hello.get('no_new_privs')}, "
                f"oom_score_adj={executor.hello.get('oom_score_adj')}, "
                f"torch_loaded={executor.hello.get('torch_loaded')})",
            )
        except ExecutorGone as exc:
            self.note("boot", f"{placement_id!r}: executor_activation_failed: {exc}"[:256])
            return f"executor_activation_failed: {exc}"[:200]
        return executor

    def _start_executor(self, tenant: Tenant, executor: child.Executor) -> str:
        """Start every rank, admitting new model-bearing contexts through MemoryManager.

        Weightless starts keep their CPU overlap. Returns the actual startup
        refusal, or "" once the process is started.
        """
        lane = tenant.lane
        placement_id = tenant.placement.placement_id
        first = next(iter(tenant.bindings.values()), None)
        if first is None:
            return "prepare refused: every binding"
        command = Start(
            devices=lane.devices,
            sequence_parallel_degree=_rank_degree(lane, tenant.bindings.values()),
            application=first.application,
            package_interface=first.interface_path,
        )
        began = time.perf_counter()
        self.monitor.start(f"start:{placement_id}", "load")
        try:
            admission = (
                self.memory.starting(lane, tenant.slot)
                if any(not binding.weightless for binding in tenant.bindings.values())
                else contextlib.nullcontext()
            )
            with admission:
                reply = executor.call(command, timeout=None)
        except ExecutorGone as exc:
            self.monitor.end(f"start:{placement_id}", "executor_gone")
            reply = {"ok": False, "code": "executor_gone_during_start", "detail": str(exc)}
        took = (time.perf_counter() - began) * 1000
        if not reply.get("ok"):
            self.monitor.end(f"start:{placement_id}", str(reply.get("code")))
            detail = str(reply.get("detail"))[:1024]
            for binding in tenant.bindings.values():
                tenant.failed_bindings[binding.entrypoint_binding_digest] = (
                    f"{reply.get('code')}: {detail}"
                )
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=binding.entrypoint_binding_digest,
                        reason=str(reply.get("code")),
                        detail=safe(detail),
                    )
                )
            self.note(
                "boot", f"{placement_id!r} start REFUSED: {reply.get('code')} - {detail}"[:256]
            )
            for line in prepare_diagnostics.activity_lines(reply.get("traceback")):
                self.note("boot", f"start traceback {placement_id!r}: {line}")
            try:
                tenant.supervision.retire_current(executor, f"start refused: {reply.get('code')}")
            except ExecutorGone as exc:
                self.note("recovery", f"start refusal reclaim failed: {exc}")
            return f"{reply.get('code') or 'executor_start_refused'}: {detail}"
        self.monitor.end(f"start:{placement_id}", "started")
        executor.started = dict(reply)
        stages = reply.get("stages") or ()
        self.note(
            "boot",
            f"{placement_id!r} started epoch {executor.epoch} in {took:.0f} ms"
            + (
                f" with followers {reply.get('follower_pids')}"
                if reply.get("follower_pids")
                else ""
            )
            + ": "
            + ", ".join(f"{name} {ms:.0f}ms" for name, ms in stages)
            + (
                f" [warmup failed: {reply.get('warmup_error')}]"
                if reply.get("warmup_error")
                else ""
            ),
        )
        return ""

    def _ensure_construction(self, attempt: AttemptRecord) -> None:
        """LOAD the attempt's construction if its process has not built it, then ACTIVATE it.

        Runs under the lane's device lock and WITHOUT the control lock: a load moves
        gigabytes and the control plane keeps answering while it does. The placement is
        read under the lock, the executor is commanded outside it, and what came back is
        published under it again. A refusal is the attempt's typed pre-acceptance refusal.
        """
        serving = attempt.spec.get("serving")
        if serving is None or self.accepted.mode != "serving" or "job" in attempt.spec:
            return
        with self.control_lock:
            if self._placement_by_id(attempt.placement_id) is None:
                return
            tenant = self._tenant(attempt.placement_id)
            plan_id = str(serving.get("entrypoint_binding_digest", ""))
            declared = tenant.bindings.get(plan_id)
            executor = tenant.supervision.current
            if (
                declared is None
                or plan_id in tenant.failed_bindings
                or executor is None
                or not executor.started
                or executor.poisoned
                or not executor.alive()
            ):
                return  # admission refuses it with the real reason
            key = declared.construction_key()
            group = [
                binding
                for binding in tenant.bindings.values()
                if binding.construction_key() == key
                and binding.entrypoint_binding_digest not in tenant.failed_bindings
            ]
            loaded = executor.loaded.get(key)
            parameters = _group_parameters(group)
            load = (
                loaded is None
                or not parameters <= loaded.parameters
                or not {b.entrypoint_binding_digest for b in group} <= loaded.bindings
            )
        if load:
            with self._call_phase(attempt.request_id, attempt.attempt, "Loading model weights"):
                self._load_construction(tenant, executor, key, group)
        if executor.active != key:
            self._activate_construction(tenant, executor, key)

    def _current_tenant(self, tenant: Tenant) -> Tenant:
        """`tenant`'s slot as it stands now (the caller holds the control lock)."""
        if self._placement_by_id(tenant.placement.placement_id) is None:
            return tenant
        return self._tenant(tenant.placement.placement_id)

    def _ceiling(self, lane: DeviceLane, slot: str) -> int:
        """What a load or activation may hold: the whole device. The device lock is the
        reservation (`memory.py`); the executor verifies against its own allocator. A lane
        with an unreadable device refuses typed."""
        measured = lane.measure(accel.host_backend_family())
        unreadable = [
            repr(entry)
            for entry, ordinal in zip(lane.entries, lane.ordinals, strict=True)
            if measured[ordinal].state != "measured"
        ]
        if unreadable:
            raise LaneRefusal(
                "device_capacity_unreadable",
                f"{lane.lane_id}: the driver could not report free bytes for "
                f"{', '.join(unreadable)} (NVML nvmlDeviceGetMemoryInfo, else nvidia-smi)",
            )
        lane.row(slot).ledger.observe_devices(lane.device_facts(measured))
        return min(memory.total_bytes for memory in measured.values())

    def _load_construction(
        self,
        tenant: Tenant,
        executor: child.Executor,
        key: str,
        group: list[DeclaredBinding],
        *,
        attention_pin: str = "",
    ) -> None:
        """Build one construction in the live process. Raises the attempt's refusal."""
        lane, slot = tenant.lane, tenant.slot
        placement_id = tenant.placement.placement_id
        first = group[0]
        if any(model.prepared_adapters for model in first.model_bindings()):
            from cozy_runtime.internal.lora_contract import CAPABILITY

            if CAPABILITY not in executor.hello.get("model_adapters", ()):
                raise AttemptRefusal(
                    "adapter_composition_unsupported",
                    "the package executor SDK cannot execute prepared LoRA graphs; "
                    "update its Runtime dependency",
                    cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE,
                )
        label = key.removeprefix("sha256:")[:12]
        if not first.weightless:
            # Idle tenants of every device give room for the weights first. Bindings of one
            # construction share its weights, so they are counted once.
            declared = max(binding.logical_weight_bytes for binding in group)
            lane.touch(slot)
            self.memory.make_room(
                lane,
                slot,
                {ordinal: declared for ordinal in lane.ordinals},
                f"load of {label} for {placement_id!r} declaring {declared} B",
            )
        try:
            authorized = 0 if first.weightless else self._ceiling(lane, slot)
        except LaneRefusal as exc:
            raise AttemptRefusal(
                exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            ) from exc
        command = executor_load_command(
            first,
            construction=key,
            devices=lane.devices,
            authorized_device_limit_bytes=authorized,
            sequence_parallel_degree=_rank_degree(lane, tenant.bindings.values()),
            parameter_names=_parameter_names_by_model(group),
            attention_pin=attention_pin,
        )
        subject = f"load:{label}"
        began = time.perf_counter()
        self.monitor.start(subject, "load")
        try:
            with self.memory.watch(lane, sample=False) as seen:
                reply = executor.call(
                    command,
                    timeout=None,
                    on_progress=lambda frame: self.monitor.advance(
                        subject, int(frame.get("position", 0) or 0)
                    ),
                    on_request=self._room(lane, slot),
                )
            if reply.get("ok") and not first.weightless:
                self.memory.moved(lane, slot, seen)
        except ExecutorGone as exc:
            self.monitor.end(subject, "executor_gone")
            reply = {"ok": False, "code": "executor_gone_during_prepare", "detail": str(exc)}
        took = (time.perf_counter() - began) * 1000
        if not reply.get("ok"):
            self._load_refused(tenant, executor, group, label, reply, took)
            code = str(reply.get("code") or "construction_load_refused")
            raise AttemptRefusal(
                code,
                f"construction {label} could not be loaded: {str(reply.get('detail'))[:512]}",
                # A load refused before it touched the process is this request's constraint
                # (a pin, a fit, the device's free bytes); a poisoned one is the executor's.
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT
                if reply.get("poisoned", "unreported") and code not in DEVICE_SHORTFALL_CODES
                else CAUSE.CAUSE_CODE_INVALID_REQUEST
                if code.startswith("attention_kernel_")
                else CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE,
            )
        self.monitor.end(subject, "filled")
        facts = dict(reply.get("facts") or {})
        try:
            resolved = PreparedModel.from_reply(reply)
            narrated = reply.get("delivery")
            delivery = (
                None
                if narrated is None
                else tolerant.read(narrated, _Delivery, _Delivery.__struct_fields__)[0]
            )
        except (KeyError, TypeError, ValueError) as exc:
            # An executor from another Runtime version answered without a fact this worker
            # needs: refuse this attempt, never the lane or the worker.
            raise AttemptRefusal(
                "construction_reply_incomplete",
                f"construction {label} reply lacks {exc}",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            ) from exc
        with self.control_lock:
            executor.loaded[key] = child.LoadedConstruction(
                key=key,
                prepared_model=resolved,
                parameters=_group_parameters(group),
                bindings=frozenset(binding.entrypoint_binding_digest for binding in group),
                facts=facts,
            )
            row = tenant.row()
            row.model_bearing = row.model_bearing or not first.weightless
            if row.ledger.executor_pid != executor.pid:
                row.ledger.begin_generation(executor.pid)
            row.ledger.observe_construction(facts)
        self.boot.step(
            "load",
            "reused" if reply.get("reused") else "filled",
            took,
            component=first.entrypoint,
            destinations=int(facts.get("filled", 0)),
            bytes=int(facts.get("filled_bytes", 0)),
            fill_ms=float(facts.get("fill_ms", 0)),
            warm_ms=float(facts.get("warm_ms", 0)),
            custody=str(facts.get("custody", "canonical")),
            ignored_extra_keys=", ".join(facts.get("ignored_extra_keys", ())),
        )
        self._note_construction(placement_id, label, reply, facts)
        parked = [str(name).removeprefix("sha256:")[:12] for name in reply.get("parked") or ()]
        self.note(
            "residency",
            f"{placement_id!r} construction {label} loaded in {took:.0f} ms "
            f"({len(group)} binding(s), {facts.get('filled_bytes', 0) / 2**30:.3f} GiB, "
            f"resident {int(reply.get('resident_bytes', 0))} B"
            + (f", evicted {parked}" if parked else ", evicted nothing")
            + (", reused" if reply.get("reused") else "")
            + ")",
        )
        if delivery is not None:
            self.note("delivery", _delivery_observation(delivery))

    def _load_refused(
        self,
        tenant: Tenant,
        executor: child.Executor,
        group: list[DeclaredBinding],
        label: str,
        reply: Mapping[str, Any],
        took: float,
    ) -> None:
        """A load refusal fails its request; only a poisoned process is replaced.

        The executor reports `poisoned` on every refused load: empty when the load was
        refused before it touched the process (a plan, fit or device-shortfall refusal on one
        rank), so its other constructions keep serving and the next request tries again. An
        older executor that does not report it is treated as poisoned. A capacity refusal
        is never held against the binding: the card was short for this attempt, and the
        next one prepares afresh even when the process had to be replaced.
        """
        placement_id = tenant.placement.placement_id
        first = group[0]
        code = str(reply.get("code"))
        self.monitor.end(f"load:{label}", code)
        self.boot.step(
            "load",
            "refused",
            took,
            component=first.components[0] if first.components else "",
            code=code,
        )
        detail = str(reply.get("detail"))[:1024]
        poisoned = str(reply.get("poisoned", "unreported"))
        shortfall = reply.get("shortfall") or {}
        if shortfall:
            detail = (
                f"{shortfall['resource']} shortfall at {shortfall['scope']}: "
                f"needed {shortfall['needed_bytes']} B, "
                f"{shortfall['available_bytes']} B free (measured) - {detail}"
            )[:1024]
        for line in prepare_diagnostics.activity_lines(reply.get("traceback")):
            self.note("boot", f"load traceback {label}: {line}")
        if not poisoned:
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_DEGRADED,
                    subject=first.entrypoint_binding_digest,
                    reason=code,
                    detail=safe(detail),
                )
            )
            self.note(
                "boot",
                f"{placement_id!r} load of construction {label} REFUSED for this request: "
                f"{code} - {detail}. The executor held nothing for it and keeps serving",
            )
            return
        capacity = code in DEVICE_SHORTFALL_CODES
        with self.control_lock:
            current = self._current_tenant(tenant)
            for member in group:
                # PER-BINDING (#572d), and it OUTLIVES the rebuild the refusal causes: the
                # process is replaced, but this construction is not tried again until the
                # desired state changes. A capacity refusal is this attempt's alone.
                if not capacity:
                    current.failed_bindings[member.entrypoint_binding_digest] = f"{code}: {detail}"
                    executor.ready_bindings.discard(member.entrypoint_binding_digest)
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_DEGRADED
                        if capacity
                        else pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=member.entrypoint_binding_digest,
                        reason=code,
                        detail=safe(detail),
                    )
                )
        self.note(
            "boot",
            f"{placement_id!r} load of construction {label} REFUSED: {code} - {detail} "
            f"[{prepare_diagnostics.memory_summary(dict(reply))}. The executor is poisoned "
            f"({poisoned}) and replaced; the card returns with the process (#613)]",
        )
        try:
            tenant.supervision.retire_current(executor, f"load refused: {code}")
        except ExecutorGone as exc:
            self.note("recovery", f"load refusal reclaim failed: {exc}")

    def _activate_construction(self, tenant: Tenant, executor: child.Executor, key: str) -> None:
        """Make the attempt's construction the resident, active one in its process."""
        label = key.removeprefix("sha256:")[:12]
        built = executor.loaded.get(key)
        try:
            authorized = (
                0
                if built is not None and built.facts.get("weightless")
                else self._ceiling(tenant.lane, tenant.slot)
            )
        except LaneRefusal as exc:
            raise AttemptRefusal(
                exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            ) from exc
        try:
            with executor.watched("activate"), self.memory.watch(tenant.lane, sample=False) as seen:
                reply = executor.call(
                    Activate(construction=key, authorized_device_limit_bytes=authorized),
                    timeout=None,
                    on_request=self._room(tenant.lane, tenant.slot),
                )
            self.memory.moved(tenant.lane, tenant.slot, seen)
        except ExecutorGone as exc:
            raise AttemptRefusal(
                "executor_absent",
                f"the executor died while activating construction {label}: {exc}",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            ) from exc
        if not reply.get("ok"):
            tenant.supervision.invalidate(executor, f"activate refused: {reply.get('code')}")
            raise AttemptRefusal(
                str(reply.get("code") or "construction_activation_refused"),
                f"construction {label} could not be activated: {str(reply.get('detail'))[:512]}",
                cause=CAUSE.CAUSE_CODE_EXECUTOR_FAULT,
            )
        with self.control_lock:
            executor.active = key
            loaded = executor.loaded.get(key)
            row = tenant.row()
            if loaded is not None:
                # The ledger prices THIS construction now; its baseline moves to where the
                # activation left the device, so parking a neighbour is never a "leak".
                row.ledger.observe_construction(loaded.facts)
            row.ledger.observe_restore(
                {**reply, "restored": {"restored_bytes": reply.get("restored_bytes", 0)}},
                f"activated {label}",
            )
        parked = [str(name).removeprefix("sha256:")[:12] for name in reply.get("parked") or ()]
        self.note(
            "residency",
            f"{tenant.placement.placement_id!r} activated {label} in "
            f"{float(reply.get('ms', 0)):.0f} ms (restored {int(reply.get('restored_bytes', 0))} B"
            + (f", parked {parked}" if parked else "")
            + ")",
        )

    def _note_construction(
        self, placement_id: str, label: str, reply: Mapping[str, Any], facts: Mapping[str, Any]
    ) -> None:
        """The construction's own boot record: legs, substrate, plan, kernels."""
        _stages: tuple[tuple[str, float], ...] = facts.get("stages") or ()
        if _stages:
            # THE EXPENSIVE LEGS AND THEIR PROVENANCE BOTH SURVIVE THE CAP (cr-141 over
            # cr-103): `note` caps a step at 256 characters, so the legs are shown most
            # expensive first and the provenance suffixes are priced into the budget first.
            _q = facts.get("qualification") or {}
            _l = facts.get("leases") or {}
            _leaf = min((_q.get("leaf_speedup_x") or {}).values(), default=-1.0)
            tail = (
                f" [qualification {_q['source']} {_q['key'][:12]}"
                + (f" leaf {_leaf:.2f}x" if _leaf >= 0.0 else "")
                + "]"
                if _q
                else ""
            ) + (
                f" [{_l['leases']} lease(s) over {_l['objects']} object(s) in "
                f"{_l['acquisitions']} acquisition(s)]"
                if _l
                else ""
            )
            legs = [(name, ms) for name, ms in _stages if ms >= 0.5]
            head = f"{placement_id!r} load legs: "
            shown: list[str] = []
            budget = 256 - len(head) - len(tail) - 24
            for name, ms in sorted(legs, key=lambda leg: -leg[1]):
                text = f"{name} {ms:.0f}ms"
                if budget - len(text) - 2 < 0:
                    break
                budget -= len(text) + 2
                shown.append(text)
            order = {name: index for index, (name, _) in enumerate(legs)}
            hidden = len(legs) - len(shown)
            floor = min((ms for name, ms in legs if f"{name} {ms:.0f}ms" in shown), default=0.0)
            self.note(
                "boot",
                head
                + ", ".join(sorted(shown, key=lambda text: order[text.rsplit(" ", 1)[0]]))
                + (f" (+{hidden} under {floor:.0f}ms)" if hidden else "")
                + tail,
            )
        _w: tuple[int, ...] = facts.get("substrate_census") or ()
        if len(_w) == 3 and _w[1]:
            self.note(
                "boot",
                f"{placement_id!r} WEIGHTLESS CONSTRUCTION DID NOT HOLD: {_w[1]} of "
                f"{_w[0] + _w[1]} parameter(s) were allocated on a real device during "
                f"construction ({_w[2] / 2**30:.3f} GiB) and every one of them is "
                "discarded by `to_empty` — the substrate did not fire",
            )
        if facts.get("weightless"):
            return
        _opt = facts.get("execution_optimization")
        self.note(
            "boot",
            f"{placement_id!r} execution plan: "
            + (
                f"APPLIED {_opt.get('identity', '?')} operations={list(_opt.get('operations', ()))}"
                if _opt
                else "NOT APPLIED — this generation runs the unfused path"
                + (
                    f" ({facts.get('execution_optimization_refusal')})"
                    if facts.get("execution_optimization_refusal")
                    else ""
                )
            ),
        )
        _attention = facts.get("attention")
        if _attention:
            self.note("boot", f"{placement_id!r} attention: {_attention.get('line', '?')}")
        _fused = facts.get("execution_fusion")
        if _fused is not None:
            self.note(
                "boot",
                f"{placement_id!r} fused glue: "
                + (
                    f"APPLIED {_fused.get('identity', '?')} sites={_fused.get('sites')} "
                    f"fp8_sites={_fused.get('fp8_sites')} warm={_fused.get('warm_ms')}ms"
                    if _fused.get("applied")
                    else f"NOT APPLIED [{_fused.get('code')}] {_fused.get('reason')}"
                ),
            )
        self.note(
            "boot",
            f"{placement_id!r} construction {label} — "
            f"{facts.get('filled', 0)} destinations, "
            f"{facts.get('filled_bytes', 0) / 2**30:.3f} GiB, fill leg "
            f"{float(facts.get('fill_ms', 0)) / 1000:.3f} s, "
            f"generation {str(reply.get('constructed_model_digest'))[:23]}… "
            f"resident={sorted(facts.get('resident', {}))} "
            f"parked={facts.get('parked', [])}; "
            f"fit: {(facts.get('fit') or {}).get('detail', 'unjudged')}",
        )

    def _warm_pages(self, tenant: Tenant) -> None:
        """Read every desired construction's stored bytes once, in the background (h3a-018).

        A construction loads on its first attempt, from the store. Measured on an H100 pod,
        that read runs at 23-33 GB/s when its pages are cached and at 0.73 GB/s when they
        are not. The read here is the same store stream the fill performs, into host
        buffers that are dropped, so the first load of each construction finds its pages
        warm. It moves no device byte, holds no lane lock, and decides nothing; a component
        already warmed by this worker is not read again.
        """
        placement_id = tenant.placement.placement_id
        work = []
        for binding in tenant.bindings.values():
            for model in binding.model_bindings():
                for name in model.components:
                    item = (model.store, model.snapshot_for(name), name)
                    if item not in self._pages_warmed:
                        self._pages_warmed.add(item)
                        work.append(item)
        if not work:
            return

        def run() -> None:
            for store, manifest, name in work:
                try:
                    report = page_warm.warm_component(store, manifest, name)
                except Exception as exc:
                    self._pages_warmed.discard((store, manifest, name))
                    self.note(
                        "residency",
                        f"{placement_id!r} page warm of {name!r} failed: "
                        f"{type(exc).__name__}: {exc}"[:256],
                    )
                    continue
                if report.get("skipped"):
                    self.note(
                        "residency",
                        f"{placement_id!r} page warm skipped {name!r}: {report['skipped']}"[:256],
                    )
                    continue
                self.note("residency", f"{placement_id!r} {page_warm.describe(report)}"[:256])

        # A LANE like every other worker thread (cr-061's fence): `run` reports every
        # failure it can see; the wrap turns the one it cannot into WORKER_PHASE_FAILED.
        threading.Thread(
            target=self._lane(f"page-warm:{placement_id}", run, fatal=False),
            name=f"page-warm:{placement_id}",
            daemon=True,
        ).start()

    def _finish_activation(self) -> None:
        """Publish the activation edge, or its typed failure when the executor was lost.

        No entrypoint-level warm pass runs here any more (cr-110): the author's `Model.warm`
        ran INSIDE the prepare, on every worker, before the executor reported Ready.
        """
        executor = self.supervision.current
        if executor is None or not self.supervision.owns(executor):
            self.latched = self.latched or "executor_lost_during_activation"
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
            self._settle()
            return
        if not self.latched:
            # Before the edge, for the reason above: the clamp in `converged_revision` is
            # what keeps this honest, and the ORDER is what keeps it consistent.
            self.accepted.converged_revision = self.accepted.accepted_desired_state_revision
        self.set_serving(pb.ServingState.SERVING_STATE_DISPATCHABLE)

    def imposed(self, lane: DeviceLane) -> dict[str, str]:
        """The IMPOSE half of the seal, for ONE lane. Every name is in the reviewed allowlist.

        `CUDA_VISIBLE_DEVICES` is the LANE's entries, not the worker's envelope (cr-066):
        the seal IS the device selection, so ordinal 0 inside the executor is the lane's
        device and nothing downstream picks a card. It used to be the whole envelope, which
        put every executor on card 0 of everything the worker was given.
        """
        imposed = {
            "CUDA_VISIBLE_DEVICES": lane.devices,
            "PYTORCH_CUDA_ALLOC_CONF": self.options.alloc_conf,
            "OMP_NUM_THREADS": str(self.options.threads),
        }
        if lane.group:
            # A GROUP's K ranks form an NCCL communicator (cr-068); NVLS multicast is off
            # BEFORE it forms, by the seal, and every rank asserts it (pgw#929).
            imposed["NCCL_NVLS_ENABLE"] = "0"
        # The FILL runs in the executor, so its capture directory has to cross the seal or
        # it instruments the one process that never reserves a destination. Projected only
        # when one is configured: an empty value is an imposition of nothing.
        where = self.config.fill_forensics_dir
        if where:
            imposed[child_env.FILL_FORENSICS_DIR] = where
        return imposed

    # ------------------------------------------------------ residency arbitration

    def _arbitrate_residency(self, attempt: AttemptRecord, slot: AttemptSlot) -> None:
        """Make room for THIS attempt on every device of its lane, then refill what an
        eviction took from it (`memory.py`). The caller holds the lane's devices, so every
        other tenant there is idle and none is mid-load."""
        serving = attempt.spec.get("serving")
        if serving is None or self.accepted.mode != "serving":
            return
        declared = slot.bindings.get(str(serving.get("entrypoint_binding_digest", "")))
        if declared is None or declared.weightless:
            return
        lane = self.lanes.by_id[slot.lane_id]
        tenant = attempt.placement_id
        why = f"attempt {attempt.request_id}#{attempt.attempt}"
        prepared = attempt.prepared
        shape = (prepared.entrypoint, prepared.cell()) if prepared else (declared.entrypoint, "")
        attempt.room = self.memory.admit(lane, tenant, shape, why)
        row = lane.row(tenant)
        self._restore_residency(lane, tenant, row, why)
        row.ledger.observe_devices(lane.device_facts(lane.measure(accel.host_backend_family())))

    def _restore_residency(self, lane: DeviceLane, slot: str, row: LaneRow, why: str) -> None:
        """Refill what an eviction took from `slot`, now that room was made for it. What
        does not fit stays absent and the attempt stages it."""
        missing = row.ledger.missing_resident()
        executor = row.executor()
        if not (missing or row.emptied) or executor is None:
            return
        kind = accel.host_backend_family()
        memory = lane.memory(kind)
        try:
            with executor.watched("restore"), self.memory.watch(lane, sample=False) as seen:
                reply = executor.call(Restore(names=tuple(sorted(missing))), timeout=None)
            self.memory.moved(lane, slot, seen)
        except ExecutorGone as exc:
            row.ledger.device_unreadable(f"the executor died while restoring: {exc}")
            self.note("residency", f"{slot!r} died while restoring: {exc}"[:256])
            return
        if not reply.get("ok", True):
            self.note(
                "residency",
                f"{slot!r} refused to restore: {reply.get('code')}: {reply.get('detail', '')}"[
                    :256
                ],
            )
            return
        gained = row.ledger.observe_restore(reply, why)
        row.ledger.device_process = self.read_device_process(executor.pid)
        restored = (reply.get("restored") or {}).get("restored") or {}
        held = (reply.get("restored") or {}).get("held") or {}
        if not held:
            row.emptied.clear()
        if gained or restored:
            after = lane.memory(kind)
            # THE OTHER HALF OF A SWAP, MEASURED RATHER THAN INFERRED (cr-100). `restore_ms`
            # is the whole re-fill, `fill_ms` the host->device leg inside it, and
            # `read_bytes` what the block layer actually served — a warm page cache reads
            # ~0 there, so "the re-stage was warm" stops being an assumption. Nothing here
            # decides anything: a duration that starts governing behaviour is the next
            # measurement to be enforced outside its conditions.
            report = reply.get("restored") or {}
            took = float(report.get("restore_ms", 0.0))
            fill_ms = float(report.get("fill_ms", 0.0))
            read = int(report.get("read_bytes", -1))
            # THE SWAP'S OWN LEGS (cr-103), beside the total they explain — the same
            # treatment `prepare` got. "9018 ms outside the fill leg" named a duration and
            # attributed none of it, which is how a restore stayed unexplained across two
            # issues; these four legs say which part of the re-stage spent it.
            legs = dict(report.get("legs") or {})
            objects = int(legs.pop("objects", 0))
            destinations = int(legs.pop("destinations", 0))
            copy_ms = float(legs.get("copy_ms", 0))
            # MS PER OBJECT, OUTSIDE THE COPY. The arithmetic says ~7 GB from page cache
            # over PCIe 4.0 x16 is ~1.5 s and from NVMe ~3.5 s, and the measured restore
            # was 9.7 s — so if what is left over scales with the OBJECT COUNT rather than
            # with bytes, the fix is batching and not faster I/O. That is one division and
            # it decides which, so it is printed rather than left to be inferred.
            per_object = (
                f", {(took - copy_ms) / objects:.2f} ms/object outside the copy" if objects else ""
            )
            self.note(
                "residency",
                f"{lane.lane_id}: {slot!r} restored {gained} B "
                f"({', '.join(sorted(restored))}) in {took:.0f} ms — fill leg "
                f"{fill_ms / 1000:.3f} s, {took - fill_ms:.0f} ms outside it"
                + (
                    " ["
                    + ", ".join(f"{name} {int(ms)}ms" for name, ms in legs.items())
                    + f" over {objects} object(s), {destinations} destination(s)"
                    + per_object
                    + "]"
                    if legs
                    else ""
                )
                + ", "
                + (
                    "block-device reads unreadable"
                    if read < 0
                    else f"{read} B read from the block layer "
                    f"({'cold' if read > gained // 2 else 'warm page cache'})"
                )
                + f" — for {why}; the driver reports "
                f"{memory.free_bytes} -> {after.free_bytes} B free",
            )
        if held:
            self.note(
                "residency",
                f"{slot!r} left {sorted(held)} absent: "
                + "; ".join(f"{name}: {reason}" for name, reason in sorted(held.items()))[:300],
            )

    def _room(self, lane: DeviceLane, slot: str) -> Handler:
        """Answer an executor that asks for room mid-load or mid-call (released executors'
        rank 0 does): idle tenants of every device of its lane give theirs back."""

        def answer(request: Request) -> Answer:
            if not isinstance(request, DeviceRoom):
                return refuse("no_durable_exchange", "device_room only")
            need = request.free_bytes
            why = f"its executor needs {need} B free"
            self.memory.make_room(lane, slot, {ordinal: need for ordinal in lane.ordinals}, why)
            try:
                return Room(ok=True, authorized_device_limit_bytes=self._ceiling(lane, slot))
            except LaneRefusal as exc:
                return refuse(exc.code, exc.detail)

        return answer

    def _end_tenant(self, placement_id: str, executor: child.Executor, why: str) -> bool:
        """End a dead tenant for the memory manager: reclaimed now, so no byte it held is
        granted twice; rebuilt afterwards on its own lane (a machine replica on its next
        grant, anything else once its devices are free)."""
        with self.control_lock:
            deferred = self._defer_rebuild(placement_id) is not None
        if not self._reclaim_dead(placement_id, executor, why):
            return False
        lane = self.lanes.lane_of(placement_id)
        if not deferred and lane is not None:
            self.supervisor.spawn(
                f"rebuild:{placement_id}:{executor.epoch}",
                lambda _: self._rebuild_lane(lane, placement_id),
            )
        return True

    # ------------------------------------------------------------------ dispatch

    def _admission_refusal(
        self, offered: pb.AttemptOffer, attempt: AttemptRecord, *, machine_owned: bool
    ) -> tuple[str, str, pb.CauseCode] | None:
        """What this attempt's own declaration refuses: an internal callable offered by
        anything but its managed parent, or a job whose binding is not prepared here."""
        internal: tuple[str, str] | None = None  # (installation, entrypoint)
        if "serving" in attempt.spec:
            hosted = self.hosted.get(offered.placement_id)
            bindings = hosted.bindings if hosted is not None else self.bindings
            declared = bindings.get(attempt.spec["serving"]["entrypoint_binding_digest"])
            if declared is not None and declared.internal:
                placement = self._placement_by_id(offered.placement_id)
                revision = placement.installation_id if placement else ""
                internal = revision, declared.entrypoint
        elif "job" in attempt.spec:
            descriptor = (attempt.spec.get("job") or {}).get("job_descriptor_id", "")
            slot = self.job_slots.get(machine_slots.key(attempt.request_id))
            binding = slot.binding if slot is not None else self.engine.jobs.get(descriptor)
            if binding is None or binding.installation_id != attempt.spec["job"]["installation_id"]:
                return (
                    "job_binding_unavailable",
                    "the selected job is not prepared",
                    CAUSE.CAUSE_CODE_LOCAL_SAFETY,
                )
            try:
                declaration = self.engine._job_declaration(binding, descriptor)
            except AttemptRefusal as exc:
                return exc.code, exc.detail, exc.cause
            if declaration.get("internal", False):
                internal = binding.installation_id, binding.job
        if internal is not None:
            try:
                internal_calls.require_child(
                    self.workspace,
                    self.fence.record_owner_id,
                    offered,
                    *internal,
                    machine_owned=machine_owned,
                )
            except WorkspaceRefusal as exc:
                return "internal_callable", str(exc), CAUSE.CAUSE_CODE_INVALID_REQUEST
        return None

    def _stages(self, attempt: AttemptRecord) -> bool:
        """A serving attempt is staged beside the device; a job admits on its lane."""
        return self.accepted.mode == "serving" and "job" not in attempt.spec

    def _offer_lane(self, placement_id: str, attempt: AttemptRecord | None = None) -> DeviceLane:
        """The lane an offer draws its seat from: the placement's, or the envelope for a job.
        Called after the placement is known to be hosted, so an unassigned placement here
        is one that is not dispatchable, and its lane is whichever its slot would take."""
        if self.accepted.mode == "job" or (attempt is not None and "job" in attempt.spec):
            if attempt is not None:
                if local := self.job_slots.get(machine_slots.key(attempt.request_id)):
                    return local.lane
                descriptor = (attempt.spec.get("job") or {}).get("job_descriptor_id", "")
                for slot in self.job_slots.values():
                    if slot.binding.job_descriptor_id == descriptor:
                        return slot.lane
            return self.lanes.envelope
        if placement_id in self.hosted:
            return self.lanes.lane_of(placement_id) or self.lanes.lanes[0]
        return self._primary_lane()

    def await_settled(self, attempt: AttemptRecord) -> None:
        """QUEUE behind THIS LANE's rebuild until it settles or the attempt's own deadline
        expires. Another lane's rebuild is not this lane's wait (cr-066)."""
        settled_gate = self.lanes.by_id[attempt.lane_id].settled
        if settled_gate.is_set():
            return
        started = time.perf_counter()
        key = f"{attempt.request_id}#{attempt.attempt}"
        self.note(
            "readiness",
            f"{key} reached the lane while the worker is rebuilding: waiting while the "
            "rebuild makes progress rather than refusing it",
        )
        timeout = attempt.deadline_ms / 1000 - time.time() if attempt.deadline_ms else None
        settled = settled_gate.wait(None if timeout is None else max(0.0, timeout))
        waited = (time.perf_counter() - started) * 1000
        self.note(
            "readiness",
            f"{key} waited {waited:.0f} ms; the placement is "
            f"{pb.ServingState.Name(self.placement.serving) if self.placement else 'absent'}"
            + ("" if settled else " (the attempt's own deadline expired while it queued)"),
        )

    def refuse(
        self,
        offered: pb.AttemptOffer,
        code: str,
        detail: str,
        cause: pb.CauseCode,
        origin: pb.CauseOrigin = ORIGIN.CAUSE_ORIGIN_WORKER,
    ) -> None:
        """A typed refusal is an outcome recorded before send, like any other (§7).

        `execution_started` is FALSE here by construction — nothing ran — which is the
        BILLING FACT stated structurally rather than inferred from a cause-code allowlist
        (#480c). The RecordOwner reads it and knows the ordinal was consumed and the budget
        was not. A machine execution's refusal is its FAILED terminal: nothing reruns it.
        """
        held = self.engine.history.get((offered.request_id, offered.attempt_ordinal))
        if held is not None and held.state in ("outcome", "closed"):
            held = None
        if held is None:
            spec: dict[str, Any] = {}
            if (
                documents.digest_of(offered.invocation_spec_canonical_bytes)
                == offered.invocation_spec_digest
            ):
                with contextlib.suppress(documents.DocumentError):
                    spec = documents.read(
                        offered.invocation_spec_canonical_bytes, pb.InvocationSpec
                    )
            attempt = AttemptRecord(
                request_id=offered.request_id,
                attempt=offered.attempt_ordinal,
                digest=offered.invocation_spec_digest,
                spec=spec,
                kind="job" if "job" in spec else "serving",
            )
        else:
            attempt = held
        outcome = self.engine.outcome(
            attempt,
            STATUS.OUTCOME_STATUS_REFUSED,
            cause,
            origin,
            f"{code}: {detail}",
        )
        self.engine.history[attempt.key()] = attempt
        self.settle(outcome)

    # ------------------------------------------------------------------ the device lane

    def run_attempt(self, attempt: AttemptRecord) -> None:
        """ONE held attempt, start to terminal, on the calling thread: staged beside the
        device, then the device under its lane's lock (entry to release, cr-022, cr-079) with
        any rebuild it leaves behind, then its post phase. A fault is its terminal."""
        lane = self.lanes.by_id[attempt.lane_id]
        released: Released | None = None
        try:
            staged = self._stages(attempt)
            self.await_settled(attempt)
            if staged and not self.stage(attempt):
                return
            self.await_settled(attempt)
            with lane.device:
                # A cancel or its deadline may have taken a staged attempt off the queue
                # while it waited for the device: then its terminal is already sent.
                if staged and not self.engine.claim_staged(attempt):
                    return
                if self.enter(attempt) if staged else self.admit(attempt):
                    released = self.execute(attempt, lane)
                # On REFUSALS too (#458): a poisoned generation refuses every attempt.
                self._rebuild_lane(lane, attempt.placement_id)
        except Exception as exc:
            self._attempt_faulted(attempt, exc)
        finally:
            # Per-GPU pid, card and interval (execution_evidence) prove where it ran.
            self.gpu.release(
                f"{attempt.request_id}#{attempt.attempt}",
                ranks=attempt.execution.get("ranks", []),
                gpus=attempt.execution.get("gpus", []),
            )
            self._capacity_changed()
        if released is not None:
            with lane.posting:
                self.finish(released)

    def _rebuild_lane(
        self,
        lane: DeviceLane,
        placement_id: str,
        executor: child.Executor | None = None,
        *,
        granted: bool = False,
    ) -> None:
        """Rebuild under the lane's device hold, so no attempt enters a half-built executor:
        the exact dead `executor`, or whatever the last attempt left in need."""
        with lane.device:
            if lane.lane_id.startswith("cpu-"):
                return
            if lane is self.lanes.orchestration:
                self._rebuild_orchestration()
            elif executor is None:
                self.rebuild_if_needed(placement_id)
            else:
                self.rebuild_exited(executor, placement_id, granted=granted)

    def _executor_died(self, lane: DeviceLane, executor: child.Executor, placement_id: str) -> None:
        self.supervisor.spawn(
            f"rebuild:{placement_id or lane.lane_id}:{executor.epoch}",
            lambda _: self._rebuild_lane(lane, placement_id, executor),
        )

    def _attempt_faulted(self, attempt: AttemptRecord, exc: BaseException) -> None:
        """This attempt's own failure. An executor it may have left mid-command is retired:
        the worker no longer knows what that process is doing."""
        why = fault_text(exc)
        self.note("attempt", f"{attempt.request_id}#{attempt.attempt} faulted: {why}"[:400])
        executor = attempt.executor
        dispatched = bool(attempt.executions and executor is not None and attempt.slot)
        if dispatched and executor is not None:
            self.engine.supervision_for(attempt).invalidate(
                executor, f"attempt faulted: {why}"[:200]
            )
        if attempt.state not in ("outcome", "closed"):
            if attempt.spool is not None:
                grants.abort_outputs(attempt.spool)
            self.terminate(
                attempt,
                self.engine.outcome,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_LOCAL_SAFETY,
                ORIGIN.CAUSE_ORIGIN_WORKER,
                why,
            )
        lane = self.lanes.by_id.get(attempt.lane_id)
        if dispatched and lane is not None:
            # What it left on the card is reclaimed now; only its successor waits.
            self._rebuild_lane(lane, attempt.placement_id)

    def _guarded(self, attempt: AttemptRecord) -> None:
        """Start this attempt's guard, once: one per attempt, so escalation is single-flight."""
        key = f"guard:{attempt.request_id}#{attempt.attempt}"
        with attempt.transition_lock:
            if attempt.guarded:
                return
            attempt.guarded = True
            attempt.poke = lambda: self.supervisor.poke(key)
        self.supervisor.spawn(key, lambda unit: self._guard(unit, attempt))

    def _guard(self, unit: Unit, attempt: AttemptRecord) -> None:
        """This attempt's clock, the WORKER's and never author code's: its deadline, then a
        cancel's stall. A cancelled executor is killed only when its frames stop against their
        own measured cadence (`AttemptRecord.cancel_stall`), never after a flat grace."""
        floor = liveness.noise_floor()
        while not self.stop.is_set():
            if (
                attempt.state in ("released", "outcome", "closed")
                or attempt.cancel_started == math.inf
            ):
                return
            now = time.monotonic()
            due: float | None = None
            if not attempt.canceling:
                if attempt.deadline_ms:
                    due = now + attempt.deadline_ms / 1000 - time.time()
                    if due <= now:
                        self._expire(attempt)
                        continue
            elif attempt.state in ("running", "reclaiming"):
                due = max(attempt.pace.moved_at, attempt.cancel_started) + attempt.pace.patience(
                    floor
                )
                if due <= now:
                    if (stall := attempt.cancel_stall(floor)) is not None:
                        self._escalate(attempt, stall)
                    continue
            unit.wait(due)

    def _expire(self, attempt: AttemptRecord) -> None:
        """The attempt's deadline passed: staged, it ends CANCELED(deadline) now, not when the
        device reaches it; running, it is cancelled cooperatively and its stall is watched."""
        key = f"{attempt.request_id}#{attempt.attempt}"
        if attempt.state == "staged" and self.engine.claim_staged(attempt):
            self.note("cancel", f"deadline expired for staged {key}")
            self.terminate(attempt, self.engine.finish_cancel, "deadline")
            return
        with attempt.transition_lock:
            if attempt.canceling or attempt.state in (
                "finalizing",
                "released",
                "outcome",
                "closed",
            ):
                return
            attempt.canceling, attempt.cancel_started = "deadline", time.monotonic()
            executor = attempt.executor if attempt.state == "running" else None
        self.note("cancel", f"deadline expired for {key}")
        if executor is not None:
            self.engine.supervision_for(attempt).cooperative_cancel(executor, key)
            self.calls.wake(attempt.request_id)

    def _escalate(self, attempt: AttemptRecord, stall: Mapping[str, float]) -> None:
        self.note(
            "cancel",
            f"no measured progress for {stall['still_s']:g} s after the cancel (longest earlier "
            f"gap {stall['worst_gap_s']:g} s over {stall['frames']} frame(s); patience "
            f"{stall['patience_s']:g} s): escalating to a forceful kill",
        )
        try:
            facts = self.engine.escalate(attempt, stall)
        except Exception as exc:
            # A reclaim that is not proved leaves a process that may still hold the device.
            attempt.cancel_started = math.inf
            self.fail_machine(
                f"executor reclaim for {attempt.request_id}#{attempt.attempt} is not proved: "
                + fault_text(exc)
            )
            return
        self.note("cancel", f"forceful: {json.dumps(facts)}")
        attempt.cancel_started = math.inf

    def stage(self, attempt: AttemptRecord) -> bool:
        """VALIDATE -> FETCH -> HYDRATE -> JOURNAL -> ACCEPT, or a typed terminal."""
        key = f"{attempt.request_id}#{attempt.attempt}"
        with self.control_lock:
            if self._admission_refused(attempt):
                return False
            try:
                self._bind_attempt_slot(attempt)
            except AttemptRefusal as exc:
                self.note("attempt", f"{key} refused: {exc.code}")
                self.terminate(
                    attempt, self.engine.refuse_held, exc.code, exc.detail, exc.cause, exc.origin
                )
                return False
        # Fetch and hydrate move bytes; the control plane keeps answering meanwhile. The
        # slot bound above is re-bound under the lock at device entry.
        try:
            self.engine.stage(attempt)
        except AttemptRefusal as exc:
            self.note("attempt", f"{key} refused: {exc.code}")
            self.terminate(
                attempt, self.engine.refuse_held, exc.code, exc.detail, exc.cause, exc.origin
            )
            return False
        if attempt.canceling and self.engine.claim_staged(attempt):
            # cancelled while it hydrated: nothing reached the device
            self.terminate(attempt, self.engine.finish_cancel, attempt.canceling)
        return attempt.state == "staged"

    def enter(self, attempt: AttemptRecord) -> bool:
        """DEVICE ENTRY for a claimed staged attempt: LOAD/ACTIVATE -> RESOLVE -> PLAN, or a
        typed terminal. The lane's device lock is held by the caller."""
        key = f"{attempt.request_id}#{attempt.attempt}"
        with self.control_lock:
            if self._entry_refused(attempt):
                return False
        try:
            self._ensure_construction(attempt)
            with self.control_lock:
                if self._entry_refused(attempt):
                    return False
                slot = self._bind_attempt_slot(attempt)
                self.engine.enter(
                    attempt, before_plan=lambda: self._arbitrate_residency(attempt, slot)
                )
        except AttemptRefusal as exc:
            self.note("attempt", f"{key} refused at device entry: {exc.code}")
            self.terminate(
                attempt, self.engine.refuse_held, exc.code, exc.detail, exc.cause, exc.origin
            )
            return False
        return True

    def _entry_refused(self, attempt: AttemptRecord) -> bool:
        """An ACCEPTED attempt's device-entry fences, under the control lock: TRUE once
        terminated. What can still stop it is a cancel, its own deadline, or its placement
        leaving this worker."""
        key = f"{attempt.request_id}#{attempt.attempt}"
        if attempt.canceling:
            self.note("cancel", f"{key} was cancelled before device entry")
            self.terminate(attempt, self.engine.finish_cancel, attempt.canceling)
            return True
        deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
        if deadline_ms and time.time() * 1000 > deadline_ms:
            self.note("cancel", f"{key} reached device entry past its deadline")
            self.terminate(attempt, self.engine.finish_cancel, "deadline")
            return True
        if self._placement_by_id(attempt.placement_id) is None:
            self.note("admission", f"refusing staged {key}: placement left this worker")
            self.terminate(
                attempt,
                self.engine.refuse_held,
                "unknown_placement",
                f"the staged attempt names placement {attempt.placement_id!r} and this "
                "worker no longer hosts it",
                CAUSE.CAUSE_CODE_UNKNOWN_PLACEMENT,
            )
            return True
        return False

    def cancel(self, message: pb.CancelAttempt, *, failure: str = "") -> str:
        """Mark the cancel; a STAGED attempt has nothing on the device, so its terminal is
        recorded now. Anything later is watched by the attempt's guard."""
        done = self.engine.cancel(message, failure=failure)
        held = self.engine.live.get(message.request_id)
        if held is None or held.attempt != message.attempt_ordinal or not held.canceling:
            return done
        if self.engine.claim_staged(held):
            self.terminate(held, self.engine.finish_cancel, held.canceling)
            return f"cancelled while staged ({held.canceling}); nothing reached the device"
        self._guarded(held)
        return done

    def admit(self, attempt: AttemptRecord) -> bool:
        """FETCH -> HYDRATE -> LOAD -> CHOOSE -> ACCEPT, or a typed terminal. Never a silence."""
        # Desired-state convergence and executor rebuilds take this same lock. Recheck the
        # offer's immutable routing facts under it, then hold it through the acceptance:
        # a queued offer cannot be accepted on the far side of a change to ITS placement
        # merely because it was admitted earlier. The construction the attempt needs
        # is loaded between the two holds, under the lane's device lock alone, so a load
        # never stalls the control plane.
        key = f"{attempt.request_id}#{attempt.attempt}"
        with self.control_lock:
            if self._admission_refused(attempt):
                return False
        try:
            self._ensure_construction(attempt)
        except AttemptRefusal as exc:
            self.note("attempt", f"{key} refused: {exc.code}")
            self.terminate(
                attempt, self.engine.refuse_held, exc.code, exc.detail, exc.cause, exc.origin
            )
            return False
        with self.control_lock:
            if self._admission_refused(attempt):
                return False
            try:
                slot = self._bind_attempt_slot(attempt)
                self.engine.prepare_and_accept(
                    attempt, before_plan=lambda: self._arbitrate_residency(attempt, slot)
                )
            except AttemptRefusal as exc:
                self.note("attempt", f"{key} refused: {exc.code}")
                self.terminate(
                    attempt, self.engine.refuse_held, exc.code, exc.detail, exc.cause, exc.origin
                )
                return False
        return True

    def _admission_refused(self, attempt: AttemptRecord) -> bool:
        """The queued offer's own facts, re-read under the control lock: TRUE once refused.

        The worker-wide admission epoch fenced the owner's dispatch in `on_offer`; it is not
        re-read here. It moves on ANY placement's edge, and another root's replica leaving
        must not refuse a call already waiting on its lane (#790's proof). What can refuse
        it is its cancel, its deadline, its own placement, or the worker's own admission.
        """
        key = f"{attempt.request_id}#{attempt.attempt}"
        if attempt.canceling:
            self.note("cancel", f"{key} was cancelled before admission")
            self.terminate(attempt, self.engine.finish_cancel, attempt.canceling)
            return True
        deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
        if deadline_ms and time.time() * 1000 > deadline_ms:
            self.note("cancel", f"{key} reached the lane past its deadline")
            self.terminate(attempt, self.engine.finish_cancel, "deadline")
            return True
        if "job" not in attempt.spec:
            # A serving call is re-checked on its OWN placement, in either mode: a machine
            # replica can leave while its call is still queued (`queued` is not HELD).
            selected = self._placement_by_id(attempt.placement_id)
            if selected is None:
                self.note(
                    "admission",
                    f"refusing queued {key}: placement {attempt.placement_id!r} is no "
                    "longer hosted",
                )
                self.terminate(
                    attempt,
                    self.engine.refuse_held,
                    "unknown_placement",
                    f"the queued offer names placement {attempt.placement_id!r} and "
                    "this worker no longer hosts it",
                    CAUSE.CAUSE_CODE_UNKNOWN_PLACEMENT,
                )
                return True
            if not self._placement_dispatchable(attempt.placement_id):
                self.note(
                    "admission",
                    f"refusing queued {key}: placement {attempt.placement_id!r} is not "
                    "dispatchable",
                )
                self.terminate(
                    attempt,
                    self.engine.refuse_held,
                    "placement_not_dispatchable",
                    f"placement {attempt.placement_id!r} stopped being dispatchable "
                    "while this offer was queued",
                    CAUSE.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE,
                )
                return True
        elif self.accepted.mode != "serving" and not self.dispatchable():
            self.note("admission", f"refusing queued {key}: worker is not dispatchable")
            self.terminate(
                attempt,
                self.engine.refuse_held,
                "placement_not_dispatchable",
                "the worker stopped being dispatchable while this offer was queued",
                CAUSE.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE,
            )
            return True
        if self.accepted.posture == pb.Posture.POSTURE_DRAINING:
            self.note("intake", f"draining: refusing queued {key}")
            self.terminate(
                attempt,
                self.engine.refuse_held,
                "draining",
                "the worker began draining while this attempt was queued",
                CAUSE.CAUSE_CODE_LOCAL_SAFETY,
            )
            return True
        return False

    def execute(self, attempt: AttemptRecord, lane: DeviceLane) -> Released | None:
        """RUN -> RELEASE. The ledger closes at release, while the device is still held; the
        caller runs the post phase after letting the device go. The driver is sampled while
        the attempt runs: its peak is what this shape costs on each device (`memory.py`).

        A terminal built here is one the device phase itself produced (cancelled before
        dispatch, the executor died, the worker faulted) and is sent at once."""
        self._record_executor(attempt)
        with self.memory.watch(lane) as seen:
            try:
                if self.machine_calls is not None:
                    self.machine_calls.run_timing.begin(
                        self.fence.record_owner_id, attempt.request_id, attempt.attempt
                    )
                    self.machine_calls.timing.phase(
                        self.fence.record_owner_id, attempt.request_id, attempt.attempt, "running"
                    )
                try:
                    released = self.engine.execute(attempt)
                finally:
                    if self.machine_calls is not None:
                        self.machine_calls.run_timing.end(
                            self.fence.record_owner_id, attempt.request_id, attempt.attempt
                        )
                        self.machine_calls.timing.phase(
                            self.fence.record_owner_id,
                            attempt.request_id,
                            attempt.attempt,
                            "finalizing",
                        )
            except Exception as exc:
                self.note("attempt", f"execution faulted: {type(exc).__name__}: {exc}")
                try:
                    released = self.engine.abandon(
                        attempt, f"the worker faulted while running this attempt: {exc}"
                    )
                except Exception as inner:
                    self.note("attempt", f"even ABANDONED could not be built: {inner}")
                    self.terminate(
                        attempt, self.engine.last_resort, f"{type(inner).__name__}: {inner}"
                    )
                    return None
        if not isinstance(released, Released):
            self.settle(released)
            return None
        prepared = attempt.prepared
        if attempt.placement_id in lane.rows and prepared is not None:
            outcome = released.reply.outcome
            code = outcome.code if outcome is not None else ""
            self.memory.settle(
                lane,
                attempt.placement_id,
                (prepared.entrypoint, prepared.cell()),
                seen,
                ok=outcome is not None and outcome.terminal == "succeeded",
                forget=code in DEVICE_SHORTFALL_CODES or code in GROUP_FAULT_CODES,
            )
        return released

    def finish(self, released: Released) -> None:
        """POST PHASE -> OUTCOME -> SEND, on the lane's post thread.

        A fault here is the WORKER's: the executor gave everything it had at release and is
        not consulted, so it is never invalidated for a codec or a write that failed after
        it was done — the attempt on the device now is untouched.
        """
        attempt = released.attempt
        try:
            outcome = self.engine.finish(released)
        except Exception as exc:
            self.note("attempt", f"post phase faulted: {type(exc).__name__}: {exc}")
            if attempt.spool is not None:
                grants.abort_outputs(attempt.spool)
            self.terminate(
                attempt,
                self.engine.outcome,
                STATUS.OUTCOME_STATUS_FAILED,
                CAUSE.CAUSE_CODE_CAPABILITY_UNAVAILABLE,
                ORIGIN.CAUSE_ORIGIN_WORKER,
                f"post_phase_faulted: {type(exc).__name__}: {exc}",
            )
            return
        self._attention_record(attempt)
        self._ignored_record(attempt, released.reply.ignored)
        self.settle(outcome)

    def _ignored_record(self, attempt: AttemptRecord, ignored: tuple[str, ...]) -> None:
        """The request fields its types did not declare were dropped, never refused (owner,
        2026-09-28); the run's root says which, once."""
        owner = self.fence.record_owner_id
        executions = self.executions
        if not ignored or executions is None or not executions.owns(owner, attempt.request_id):
            return
        with contextlib.suppress(WorkspaceRefusal):
            executions.notice(
                owner,
                executions.scheduling_root(owner, attempt.request_id)[0],
                "warning",
                {
                    "code": "request_fields_ignored",
                    "message": f"ignored unknown field{'s' if len(ignored) > 1 else ''} "
                    f"{', '.join(ignored)} — not in {self.engine._subject(attempt)}'s interface",
                    "fields": list(ignored),
                },
            )

    def _attention_record(self, attempt: AttemptRecord) -> None:
        """A request's pin is the run's: whichever of its calls hold attention sites apply
        it. The run's root records the first call that did, or warns that none did."""
        pin = str(attempt.spec.get("attention_kernel") or "")
        owner = self.fence.record_owner_id
        executions = self.executions
        if not pin or executions is None or not executions.owns(owner, attempt.request_id):
            return
        with contextlib.suppress(WorkspaceRefusal):
            root = executions.scheduling_root(owner, attempt.request_id)[0]
            if executions.noticed(owner, root, "attention.applied"):
                return
            if attempt.attention_applied:
                executions.notice(
                    owner, root, "attention.applied", {"pin": pin, "request": attempt.request_id}
                )
            elif root == attempt.request_id:
                executions.notice(
                    owner,
                    root,
                    "warning",
                    {
                        "code": "attention_pin_unapplied",
                        "message": f"attention pin {pin} was never applied",
                    },
                )

    def terminate(
        self, attempt: AttemptRecord, build: Callable[..., pb.AttemptOutcome], *args: Any
    ) -> None:
        """Build one outcome for a held attempt, record it, and send it."""
        why = ""
        try:
            self.settle(build(attempt, *args))
            return
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
            self.note("attempt", f"the outcome could not be built ({why}); last resort")
        try:
            self.settle(self.engine.last_resort(attempt, why))
        except Exception as inner:
            self.note("attempt", f"even the last-resort outcome failed: {inner}")
            self.fail_machine("last_resort_failed")

    def rebuild_if_needed(self, placement_id: str = "") -> None:
        """The lane's queue starts behind the rebuild because BOTH live on this lane.

        The DECISION is taken under the control lock; the rebuild itself is not (cr-066
        step 5). A rebuild is a spawn plus a prepare — minutes for a real model — and a
        worker-wide lock held for that long stalled every other lane's admission and every
        stream frame behind one placement's refill. The convergence lane already prepares
        without the lock, fenced by the accepted revision; the rebuild takes the same
        posture, and `ExecutorSupervision._lifecycle` keeps replacement single-flight.
        """
        with self.control_lock:
            hosted = self.hosted.get(placement_id)
            if hosted is not None:
                current = hosted.supervision.current
                rebuild = current is None or not current.alive() or bool(current.poisoned)
            else:
                current = self.supervision.current
                rebuild = current is None or not current.alive() or bool(current.poisoned)
            if not rebuild:
                return
            dead = self._defer_rebuild(placement_id)
        if dead is not None:
            self._reclaim_dead(placement_id, dead)
        elif hosted is not None:
            self._prepare_hosted_executor(hosted, fresh=False)
        elif current is None or not current.alive():
            self._rebuild("the executor is gone")
        else:
            self._rebuild(f"poisoned: {current.poisoned}")

    def rebuild_exited(
        self, executor: child.Executor, placement_id: str = "", *, granted: bool = False
    ) -> None:
        """Rebuild only if the queued death still names the owned dead epoch, or its grant
        found that epoch already reclaimed (`_end_tenant`)."""

        with self.control_lock:
            hosted = self.hosted.get(placement_id)
            supervision = hosted.supervision if hosted is not None else self.supervision
            reclaimed = granted and supervision.current is None
            if not reclaimed and (
                supervision.current is not executor or (executor.alive() and not executor.poisoned)
            ):
                return
            primary = self.placement.placement_id if self.placement is not None else ""
            dead = None if granted else self._defer_rebuild(placement_id or primary)
        if dead is not None:
            self._reclaim_dead(placement_id or primary, dead)
            return
        if hosted is not None:
            self._prepare_hosted_executor(hosted, fresh=False)
            return
        self._rebuild(f"observed exit of executor epoch {executor.epoch} pid {executor.pid}")

    def _rebuild(self, why: str) -> None:
        """Replace the PRIMARY slot's executor and RELOAD before it is dispatchable again.

        Runs on the primary slot's lane thread with the control lock RELEASED (see
        `rebuild_if_needed`): the spawn is single-flight under the supervision's own
        lifecycle lock, and the prepare reads the bindings the accepted revision holds now.
        """
        job = self.accepted.mode == "job"
        if job:
            self.set_job_ready(False)
        else:
            self.set_serving(pb.ServingState.SERVING_STATE_ACTIVATING, "executor rebuild")
        started = time.perf_counter()
        self.note("recovery", f"rebuilding the executor: {why}")
        current = self.supervision.current
        lane = self._primary_lane()
        try:
            if current is None:
                self.supervision.spawn(imposed=self.imposed(lane))
            else:
                self.supervision.replace(current, imposed=self.imposed(lane), why=why)
        except ExecutorGone as exc:
            self.note("recovery", f"respawn failed: {exc}")
            self.latched = f"executor_respawn_failed: {exc}"[:200]
            if job:
                self.set_job_ready(False)
            else:
                self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, "respawn failed")
            self._settle()
            return
        self.bump_admission("executor respawned")
        if job:
            executor = self.supervision.current
            assert executor is not None
            executor.reserved_for_job = True
            self.set_job_ready(True)
        else:
            self.activate_executor(fresh=False)
        self.note(
            "recovery",
            f"dispatchable again in {(time.perf_counter() - started) * 1000:.0f} ms at "
            f"executor epoch {self.supervision.epoch} / admission epoch "
            f"{self.admission_epoch}",
        )

    def reporter(self) -> None:
        """The ObservedWorkerState tick, and the liveness SAMPLE that rides it."""
        while not self.stop.wait(REPORT_SECONDS):
            try:
                if self.machine_calls is not None:
                    self.machine_calls.run_timing.sample()
                for subject in list(self.monitor.cursors):
                    self.monitor.sample(subject)
                self.publish_activity()
                from . import machine_compute

                machine_compute.park_if_idle(self)
                # Observation follows authentication, independently of the dispatch
                # barrier. Only the current stream is eligible, after its snapshot.
                if self.fence.control_stream_epoch in self.attached_streams:
                    self.send(pb.WorkerFrame(observed_state=self.observed_state()))
            except Exception as exc:  # one tick's failure is that tick's
                self.note("report", f"tick faulted: {fault_text(exc)}"[:400])

    def publish_activity(self, *, strict: bool = False) -> None:
        """Tell the pod supervisor what this worker is doing (`worker/activity.py`).

        It rides the reporter for one reason: a tick that does not happen is the evidence
        the supervisor needs. A failure to write is NOTED and never raised — the reporter
        lane dying would silence the file permanently, and a silent file reads as a wedge.
        """
        if self.activity_file is None:
            return
        try:
            held = activity.holding(self)
            self.activity_file.publish(
                owner_attached=bool(self.attached_streams),
                attempts_in_flight=self.engine.in_flight(),
                active_work=bool(held),
                holding=held,
                execution_retention_required=(
                    self.executions is not None
                    and self.fence.record_owner_recorded
                    and self.executions.retention_required(self.fence.record_owner_id)
                ),
                meter=activity.meter(self),
            )
        except OSError as exc:
            if strict:
                raise
            self.note("activity", f"the supervisor's activity file was not written: {exc}")

    def observed_state(self) -> pb.ObservedWorkerState:
        """The observed half of the pair (§8; was `Report`).

        ACCEPTANCE AND CONVERGENCE ARE TWO SEPARATE, CHECKABLE FACTS.
        `accepted_desired_state_revision` advances when this worker durably accepts intent —
        which is all `applied_revision` ever honestly meant, while claiming to report
        reality. `converged_revision` advances ONLY when observed state satisfies the
        accepted set, and `converged < accepted` is the normal, visible state of a
        convergence in progress or a latched failure. Never an error; always readable.

        Serving truth is PER PLACEMENT; job mode reports its capacity credit. Before any
        desired state it reports NEITHER — an honest "no claim yet" rather than a fabricated
        zero.

        The LATCH is folded in here rather than at the sites that set it, so that no report
        can leave describing a worker that has stopped without saying that it has.
        """
        self._publish_latch()
        observed = pb.ObservedWorkerState(
            accepted_desired_state_revision=self.accepted.accepted_desired_state_revision,
            converged_revision=self.converged_revision(),
            worker_phase=self.phase,
            held_attempts=self.engine.active(),
            faults=self.engine.faults[-8:],
            activity=self.activity[-8:],
            applied_wire_minor=self.fence.wire_minor,
            # THE ONE WORKER-LEVEL ADMISSION FENCE (§6). Not per placement: N counters over
            # ONE serialized device advertise N x the real capacity.
            admission_epoch=self.admission_epoch,
            admission_state=self.admission_state(),
        )
        if self.accepted.mode == "job":
            observed.job_capacity.CopyFrom(self.job_capacity())
        elif self.accepted.mode == "serving":
            observed.accepted_placement_set_digest = self.accepted.accepted_placement_set_digest
            if self.placement is not None:
                observed.placements.append(self.placement_status())
            for placement_id in sorted(self.hosted):
                observed.placements.append(self.placement_status(placement_id))
            lane_wire.emit_lanes(observed, self.lane_documents())
        lane_wire.emit_held_manifests(observed, self.held_manifests.digests())
        return observed

    def converged_revision(self) -> int:
        """A DERIVED fact, recomputed on every read — never a stored optimism.

        It may NEVER advance past a placement whose serving state is not DISPATCHABLE for a
        desired spec (§8): a worker that advances it while observed state disagrees is in
        breach, not merely optimistic. So the stored value is the high-water mark
        `prepare_executor` reached, and this read CLAMPS it back the moment reality moves —
        a placement that stops being dispatchable un-converges immediately, which is exactly
        what a RecordOwner watching a cutover needs to see.
        """
        if self.accepted.mode == "serving" and not self._all_placements_dispatchable():
            return min(
                self.accepted.converged_revision,
                max(0, self.accepted.accepted_desired_state_revision - 1),
            )
        return self.accepted.converged_revision

    def placement_status(self, placement_id: str = "") -> pb.PlacementStatus:
        """Per-placement truth on TWO AXES (#473/#482), and `readiness_epoch` is not among
        them: it is DELETED, not renamed (#486b), because the serving axis plus
        `dispatchable_binding_digests` carry what a placement can serve and the worker-level
        `admission_epoch` carries whether an offer is admissible. `attempt_credits` is
        gone for the same reason, one level up.

        The accelerator qualification is deliberately ABSENT: this torch-free process cannot
        measure capability facts, and absent IS the present-but-unqualified state (#454 —
        registration honesty is schema). The executor-measured qualification lands with
        cr-020/cr-021's lanes.
        """
        hosted = self.hosted.get(placement_id)
        placement = hosted.placement if hosted is not None else self.placement
        supervision = hosted.supervision if hosted is not None else self.supervision
        assert placement is not None
        executor = supervision.current
        dispatchable = sorted(executor.ready_bindings) if executor is not None else []
        loaded = sorted(executor.loaded_bindings()) if executor is not None else []
        selected = {
            documents.spell(entrypoint.entrypoint_binding_digest)
            for entrypoint in placement.document.entrypoints
        }
        fault_subjects = selected | {placement.placement_id}
        # DISJOINT by construction (the proto says so): a binding is either servable now or
        # still to be materialized, never counted in both.
        materializable = sorted(selected - set(dispatchable))
        status = pb.PlacementStatus(
            placement_id=placement.placement_id,
            placement_set_digest=placement.placement_set_digest,
            materialization=placement.materialization,
            serving=placement.serving,
            retained_fallback_placement_set_digest=(
                placement.retained_fallback_placement_set_digest
            ),
            executor_epoch=supervision.epoch,
            dispatchable_binding_digests=[documents.raw(value) for value in dispatchable],
            materializable_binding_digests=[documents.raw(value) for value in materializable],
            # proto-061: the dispatchable bindings whose construction is BUILT in the live
            # executor, resident or parked. Serving one of them moves no checkpoint byte.
            loaded_binding_digests=[documents.raw(value) for value in loaded],
            faults=[
                f
                for f in self.engine.faults[-8:]
                if (hosted is None or f.subject in fault_subjects)
                and f.kind
                in (
                    pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    pb.FaultKind.FAULT_KIND_BINDING_DEGRADED,
                    pb.FaultKind.FAULT_KIND_EXECUTOR_POISONED,
                    pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    pb.FaultKind.FAULT_KIND_FALLBACK_PIN_MISSING,
                )
            ],
        )
        if (
            placement.materialization == pb.MaterializationState.MATERIALIZATION_STATE_STAGED
            and placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
        ):
            status.installation_id = placement.installation_id
            if not status.installation_id:
                raise RuntimeError("dispatchable placement has no installed environment")
        if placement.acquisition.ListFields():
            status.acquisition.CopyFrom(placement.acquisition)
        lane_wire.emit_device_lane_id(status, placement.lane_id)
        return status

    def job_capacity(self) -> pb.JobCapacity:
        """How many JOB attempts this worker can take right now (cr-009)."""
        in_flight = sum(
            1
            for a in self.engine.live.values()
            if a.job is not None and a.lane_id != "orchestration"
        )
        orchestrations = sum(
            1
            for a in self.engine.live.values()
            if a.job is not None and a.lane_id == "orchestration"
        )
        ready = bool(self.jobs) and self.job_ready
        cpu = self.job_slots.get("orchestration")
        ordinary = self.job_slots.get("envelope")
        return pb.JobCapacity(
            jobs_in_flight=in_flight,
            jobs_available=max(
                0, (1 if ready and (ordinary is not None or not self.job_slots) else 0) - in_flight
            ),
            orchestration_in_flight=orchestrations,
            orchestration_available=max(
                0, (1 if ready and cpu is not None and cpu.ready() else 0) - orchestrations
            ),
        )

    # ------------------------------------------------------------------ identity

    def resources(self) -> pb.WorkerResources:
        """Torch-free statics measured ONCE at claim time (#446): the worker NEVER
        fabricates capability facts — a discovered accelerator is present-but-unqualified
        until a deployment executor qualifies it.

        The devices are the inventory this worker schedules on (`options.gpus`, measured by
        its launcher or supplied virtual), the same rows its workspace and readiness report.
        """
        gpus = self.options.gpus
        facts = hostfacts.measure("none") if gpus else self.host_facts()
        if gpus:
            names = {gpu["device_name"] for gpu in gpus}
            memory = {gpu["memory_bytes"] for gpu in gpus}
            drivers = {gpu["driver_version"] for gpu in gpus}
            # One (name, memory, driver) describes a homogeneous inventory and nothing else.
            unreadable = [
                fact
                for fact, values in (
                    ("gpu_name", names),
                    ("vram_total_bytes", memory),
                    ("driver_version", drivers),
                )
                if len(values) != 1
            ]
            facts = dataclasses.replace(
                facts,
                backend="cuda",
                gpu_count=len(gpus),
                gpu_name=next(iter(names)) if len(names) == 1 else "",
                vram_total_bytes=next(iter(memory)) if len(memory) == 1 else 0,
                driver_version=next(iter(drivers)) if len(drivers) == 1 else "",
                unreadable=tuple(sorted({*facts.unreadable, *unreadable})),
            )
        memory_model = ""
        if facts.backend == "cuda":
            memory_model = "discrete"
        elif facts.backend == "mps":
            memory_model = "unified"
        elif facts.backend == "none":
            memory_model = "host"
        return pb.WorkerResources(
            platform=facts.platform,
            backend=facts.backend,
            memory_model=memory_model,
            device_count=facts.gpu_count,
            device_name=facts.gpu_name,
            device_memory_total_bytes=facts.vram_total_bytes,
            driver_version=facts.driver_version,
            backend_version=facts.cuda_version,
            host_ram_total_bytes=facts.host_ram_total_bytes,
            vcpu_count=facts.vcpu_count,
            unreadable=list(facts.unreadable),
        )

    def disk_invoice(self) -> dict[str, int | str]:
        """The numbers a `DISK_SHAPE` claim must carry, read from the filesystem itself."""
        try:
            stat = os.statvfs(self.root)
        except OSError:
            return {"readable": "unreadable"}
        return {
            "readable": "measured",
            "total_bytes": stat.f_blocks * stat.f_frsize,
            "available_bytes": stat.f_bavail * stat.f_frsize,
            "free_inodes": stat.f_favail,
            "block_size": stat.f_frsize,
        }

    def boot_failure(self, reason: pb.BootFailureReason, detail: str) -> pb.BootFailure:
        """One boot verdict, classified ONCE, with its invoice attached. Delivered on the
        first claimed stream INSTEAD of ClaimAck (01) — the owner dials, so the verdict
        waits for it."""
        invoice = self.disk_invoice()
        self.boot.step("load", "boot_failure", 0.0, reason=pb.BootFailureReason.Name(reason))
        return pb.BootFailure(
            worker_id=self.options.worker_id,
            worker_instance_id=self.options.instance_id,
            reason=reason,
            detail=f"{detail} | disk invoice: {json.dumps(invoice, sort_keys=True)}"[:1024],
            resources=self.resources(),
            worker_release_id=self.options.release_id,
            control_runtime_digest="",
        )

    def run(self) -> int:
        """Serve until stopped. The worker outlives any one stream: an owner that drops
        the connection re-dials and re-claims the SAME boot; the in-memory host ending its
        one stream is the run ending."""
        # Executor grandchildren orphan here, so retirement can reap every group member
        # and prove its exact PID absent before publishing a successor.
        proctree.arm_subreaper()
        self.records.append(
            "session",
            {
                "worker_boot_id": self.fence.worker_boot_id,
                "instance_id": self.options.instance_id,
            },
        )
        if self.boot_fatal is None:
            unbindable = socket_path_refusal(self.supervision.socket_path)
            if unbindable:
                self.boot_fatal = self.boot_failure(
                    pb.BootFailureReason.BOOT_FAILURE_REASON_CONFIG_INVALID, unbindable
                )
                self.exit_code = int(Exit.structural)
                self.note("boot", f"BOOT FAILURE (config): {unbindable}")
        if self.boot_fatal is None:
            killed = self.supervision.sweep_orphans()
            if killed:
                self.note(
                    "recovery",
                    f"killed {len(killed)} orphan executor(s) from before this restart: {killed}",
                )
            # THE MACHINE IS UP (#482). `worker_phase` is machine lifecycle, pulled out of
            # the placement enum: BOOTING is true only until the socket, orphan sweep and
            # lanes are all real. It moves
            # here and not at the first claim, because a worker is online whether or not a
            # RecordOwner has dialled it — and admission is CLOSED until the barrier anyway.
            self._rebuild_held_manifests()
            self.set_phase(pb.WorkerPhase.WORKER_PHASE_ONLINE, "boot completed")
            threading.Thread(
                target=self._lane("convergence", self.convergence_lane),
                daemon=True,
                name="convergence-lane",
            ).start()
            threading.Thread(
                target=self._lane("report", self.reporter), daemon=True, name="report"
            ).start()
            # The durable owner is known from the last boot: its work resumes from the
            # journal now. Otherwise the first accepted claim runs this one pass.
            self.reconcile_executions()
        try:
            self.host.serve(self)
        finally:
            self.shutdown()
        return self.exit_code

    def shutdown(self) -> None:
        # Fence new claims before the existing worker-root lease can be released.
        # Any claim already syncing its counter finishes under the same lock.
        with self.claim_lock:
            self.claim_ready.clear()
        supervisions = [self.supervision] + [hosted.supervision for hosted in self.hosted.values()]
        for name, slot in self.job_slots.items():
            if name == "orchestration" or name.startswith("cpu-"):
                supervisions.append(slot.supervision)
        supervisions += [slot.supervision for slot in self.warm_cpu.values()]
        seam = [
            supervision.current.channel.accounting()
            for supervision in supervisions
            if supervision.current is not None
        ]
        self.request_stop()
        self.prespawns.close()
        if self.machine_calls is not None:
            self.machine_calls.close()
        if self.machine_effects is not None:
            self.machine_effects.close()
        self.weights.close()
        self.convergences.put(None)
        self.watches.end_all()
        self.host.stop()
        # Source custody belongs to the RecordOwner, not this disposable process.
        # A normal recycle can happen while another request is paused or blocked;
        # sweeping the Store here would destroy its expensive retained work.
        # Drain entered preparation, but release a source only through the explicit
        # owner-authorized release operation after its last dependency is gone.
        with self.preparation_lock:
            pass
        # Every executor goes first, so every attempt on one ends and its unit records it;
        # the primary slot's supervision, which holds the worker-root lease, closes LAST: a
        # successor that takes the lease knows this process has left its journal rows alone.
        with contextlib.suppress(ExecutorGone):
            if (current := self.supervision.current) is not None:
                self.supervision.retire_current(current, "worker shutdown")
        for supervision in [*supervisions[1:], supervisions[0]]:
            if supervision is supervisions[0]:
                self.supervisor.join()
            try:
                supervision.close()
            except ExecutorGone as exc:
                self.exit_code = int(Exit.internal)
                self.note("shutdown", f"executor reclaim failed: {exc}")
        if self._pressure_registration is not None:
            self._pressure_registration.close()
            self._pressure_registration = None
        print(
            "[worker] summary "
            + json.dumps(
                {
                    "worker_boot_id": self.fence.worker_boot_id,
                    "control_stream_epoch": self.fence.control_stream_epoch,
                    "restart_requested": self.restart_requested,
                    "executor_epochs": sum(supervision.spawns for supervision in supervisions),
                    "admission_epoch": self.admission_epoch,
                    "executed": self.engine.executed,
                    "record_count": self.records.appends,
                    "progress_sent": self.watches.sent,
                    "progress_shed": self.watches.shed,
                    "hub_calls_allowed": self.authorizer.allowed,
                    "hub_calls_refused": len(self.authorizer.refusals),
                    "seam": seam,
                    "lanes": [lane.summary() for lane in self.lanes],
                    "rss_bytes": read_vmrss(os.getpid()),
                }
            ),
            flush=True,
        )


#: Written when the worker FAILS: the next boot fails what it held instead of running it.
FAILED_MARKER = "worker-failed.json"
