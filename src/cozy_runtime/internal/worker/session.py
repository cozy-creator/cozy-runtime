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

import argparse
import base64
import contextlib
import dataclasses
import functools
import hashlib
import hmac
import itertools
import json
import math
import os
import queue
import re
import resource
import shutil
import signal
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from cozy_runtime.internal import (
    accel,
    canonical,
    child_env,
    fill,
    hostfacts,
    job_plan,
    liveness,
    package_environment,
    package_installation,
    prepare_diagnostics,
    proctree,
    readiness,
    weights_sink,
)
from cozy_runtime.internal.config import ConfigError, RuntimeConfig, read_config
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.worker import (
    acquire,
    activity,
    child,
    downloads,
    grants,
    lane_wire,
    machine_byte_inputs,
    machine_capture,
    machine_lanes,
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
    GrpcControlHost,
    WatchFanout,
    _loopback_preparation_listener,
)
from cozy_runtime.internal.worker.gpu_scheduler import Demand, GpuScheduler, width_for
from cozy_runtime.internal.worker.lanes import (
    LANE_RELEASED,
    LANE_RUNNING,
    LANE_STAGED,
    DeviceLane,
    ExecutorDeath,
    LaneRefusal,
    LaneRow,
    LaneSet,
    fold_memory,
)
from cozy_runtime.internal.worker.ledger import Ledger, read_vmrss
from cozy_runtime.internal.worker.machine_publication import PublicationAuthority
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    JobBinding,
    PlanRefusal,
    PreparedModel,
    PreparedRequest,
)
from cozy_runtime.internal.worker.records import WorkerRecords
from cozy_runtime.internal.worker.workspace import WorkspaceRefusal
from cozy_runtime.protocol import (
    CAPABILITY_UNAVAILABLE_CODE,
    MIN_COMPATIBLE_WIRE_MINOR,
    WIRE_MINOR,
    documents,
    weights_limits,
)
from cozy_runtime.protocol import worker_pb2 as pb

#: The lossy lane's per-watch depth. Oldest is shed first; a durable frame never enters it.
PROGRESS_QUEUE = 256

#: Report cadence. Report is DURABLE and carries the applied baseline, so the owner can
#: always see a directive it issued but the worker did not apply.
REPORT_SECONDS = 2.0

#: The RUN-ONCE RECYCLE exit (cr-009, v1 rc 75). A job worker that completed its one attempt
#: and had its terminal acknowledged asks to be RECLAIMED, and that is not a death.
RECYCLE_EXIT = 75

#: How often the rebuild queue LOOKS. A sampling cadence is the resolution of an
#: observation, never a bound on one (#434); it is the Report cadence because that is
#: already the rate at which this worker's non-attempt facts move.
REBUILD_SAMPLE_SECONDS = REPORT_SECONDS

#: The worker's accepted-attempt WINDOW behind `available_attempt_slots` (#472e/#494f) is
#: each lane's seats summed over the lanes with a dispatchable placement: one counter per
#: SERIALIZED RESOURCE, and that resource is the device LANE (cr-066). A serving lane has
#: `LANE_RUNNING + LANE_STAGED` seats (proto-061 G): one attempt on the device and one
#: STAGED behind it — accepted, hydrated and journaled while the device runs, visible to the
#: RecordOwner as QUEUED, entering the device at the running one's release. A seat is held
#: from the offer's hold to DEVICE RELEASE (cr-079 D3); a released attempt holds none, but
#: while more than `LANE_RELEASED` released attempts on a lane await `AttemptOutcomeAck` the
#: lane advertises nothing. An offer over its lane's seats is a JOURNALED REFUSED outcome,
#: never a silent hold. A multi-lane worker REFUSES a RecordOwner whose negotiated minor
#: cannot read `lanes[]` (`device_lanes_unsupported`, proto-024 minor 22).


def _progress_document(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the measured progress coordinates the executor already supplied.

    Stage-local and overall fractions stay separate. Optional coordinates are omitted when
    the package does not know them; no layer fabricates whole-invocation completion. A
    byte stage names its `unit` and the `rate` it measured since its previous sample.
    Other events retain the executor's public record, already bounded and redacted at
    emission. Only seam routing fields are removed; a log's scalar value is its level,
    not its message, and a load's value may be None beside a measured position.
    """
    kind = frame.get("kind", "progress")
    payload: Any = {
        key: value for key, value in frame.items() if key not in {"event", "request_id", "kind"}
    }
    if kind == "progress":
        payload = {
            "stage": str(frame.get("stage", ""))[:120],
            "step_ms": frame.get("step_ms", 0.0),
        }
        for field in ("stage_fraction", "position", "total", "overall_fraction", "unit", "rate"):
            if frame.get(field) is not None:
                payload[field] = frame[field]
    return {"type": kind, "payload": payload}


def _delivery_observation(delivery: Mapping[str, Any]) -> str:
    """Render only the current resolution-plan projection; narration never gates serving."""

    def census(name: str) -> str:
        value = delivery.get(name)
        if not isinstance(value, Mapping):
            return "unreported"
        rows = ", ".join(f"{key} x{count}" for key, count in sorted(value.items()))
        return rows or "none"

    materialization = str(delivery.get("materialization") or "verbatim")
    calibrated = "calibrated" if delivery.get("calibrated") else "uncalibrated"
    confession = str(delivery.get("confession") or "")
    # The leaf-speed evidence rides BESIDE the route census it qualifies: a plan that routes
    # `encoded_gemm x453` on a card whose probe measured the leaf at 0.61x its float GEMM
    # says so on the same line, rather than banking the number where nothing reads it.
    speedup = delivery.get("leaf_speedup_x")
    leaf = (
        f"; leaf {speedup:.2f}x its float GEMM"
        if isinstance(speedup, int | float) and speedup >= 0.0
        else ""
    )
    selection = (
        "per-model delivery"
        if delivery.get("models")
        else f"on rung {delivery.get('rung', 'unreported')}"
    )
    return (
        f"{delivery.get('variant', '')!r} / {delivery.get('label', 'unlabelled')} {selection} "
        f"({delivery.get('delivery', 'unreported')}/{materialization}; "
        f"routes {census('routes')}; kernels {census('kernels')}{leaf}; {calibrated}) chosen "
        f"on the {delivery.get('objective', 'unreported')} axis"
        + (f" - {confession}" if confession else "")
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
    tensorfs_root: Path | None = None
    """The one host-local executable TensorFS Store."""
    artifact_cache: Path | None = None
    """Where the dynamic acquisition lane writes the exact bytes it fetches (cr-049).

    Content-addressed and re-readable: a complete object is never re-fetched, which is what
    makes a kill -9 mid-materialize resume by digest. Absent means no dynamic acquisition
    was provisioned and a placement naming artifacts this worker does not already hold
    reports a typed fault rather than guessing a directory."""
    development_project: Path | None = None
    """Explicit same-user source-checkout execution; product published venvs leave absent."""
    development_package: str = ""
    development_release: str = ""
    development_installation_id: str = ""
    """Exact launch facts the development Placement must echo."""
    development_binding_overrides: Mapping[str, str] = field(default_factory=dict)
    """One-shot local resolver inputs; long-lived Creator launches use package.toml."""
    attention_pin: str = ""
    """THE EXECUTION-PATH OVERRIDE (cr-125): one attention kernel by name, or empty.

    Empty is the whole product. A serving worker picks the fastest kernel this device and
    image support by itself (cr-124) and has no knob at all — `_SERVE_FLAGS` is a closed
    set and this is deliberately not in it, so the guard is the absence of a way to say it
    rather than a condition somebody could get wrong. Only the ephemeral worker
    `cozy-runtime run` builds in memory can carry one, which is exactly the dev mode the
    A/B needs: two runs, two pins, one honest comparison.

    It is the pin of every request that names none: an execution parameter of `invoke`,
    applied on every rank for that attempt, never a construction of its own (cr-191)."""
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

    placement_id: str
    placement_set_digest: bytes
    package: dict[str, Any]
    development: dict[str, Any]
    installation_id: str
    package_interface: bytes
    bindings_digest: bytes
    models: tuple[dict[str, Any], ...]
    entrypoints: tuple[dict[str, Any], ...]
    document: dict[str, Any]
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
    snapshot_id: str = ""  # the snapshot the CURRENT stream must ack
    snapshot_digest: bytes = b""  # ...bound by BOTH; an ack naming either alone is not one
    dispatch_open: bool = False
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


def _capability_unavailable(operation: str, minor: int) -> str:
    return (
        f"{operation} needs wire minor {MIN_COMPATIBLE_WIRE_MINOR} or newer and the owner "
        f"speaks {minor}; update Creator. Other work on this worker continues."
    )


def _same_parent(left: pb.JobDirective, right: pb.JobDirective) -> bool:
    """One CPU orchestration parent: the prepared job it runs. Policy fields (limits,
    publication, reclaim) may change without replacing a parent, which keeps the contract
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


def _digest_field(spelled: Any) -> bytes:
    """A document spells a digest `sha256:<hex>`; the wire carries the 32 raw bytes. Absent
    is absent — an empty field is not a zero digest (#494g)."""
    return documents.raw(str(spelled)) if spelled else b""


def read_placement_set(desired_set: pb.DesiredPlacementSet) -> list[dict[str, Any]]:
    """§4's ORDER OF OPERATIONS, and it is not negotiable.

    RECOMPUTE `sha256(placement_set_canonical_bytes)` and compare BEFORE parsing a single
    field. The frozen wire "digest-addressed" the set by an ECHOED claim, which is a
    statement the sender makes about bytes it also sent; this is a fence the receiver
    computes. Only on a match is the document parsed. Fields this worker does not consume
    are ignored at every depth, and the nested binding digests are the producer's names.

    Raises `documents.DocumentError`; the caller turns that into the typed fault and leaves
    the previous set serving.
    """
    recomputed = documents.digest_of(desired_set.placement_set_canonical_bytes)
    if recomputed != desired_set.placement_set_digest:
        raise documents.DocumentError(
            "placement_set_digest_mismatch",
            f"the set claims {desired_set.placement_set_digest.hex()[:16]}… and its own "
            f"{len(desired_set.placement_set_canonical_bytes)} B hash to "
            f"{recomputed.hex()[:16]}… — nothing in it has been parsed",
        )
    document = documents.read(desired_set.placement_set_canonical_bytes, pb.PlacementSet)
    placements = document.get("placements", [])
    if not isinstance(placements, list):
        raise documents.DocumentError("wrong_type", "`placements` is a list of Placement")
    for entry in placements:
        if not isinstance(entry, dict):
            raise documents.DocumentError("wrong_type", "a Placement is a JSON object")
        _placement_profile(entry)
    placement_ids = [str(entry["placement_id"]) for entry in placements]
    if len(placement_ids) != len(set(placement_ids)):
        raise documents.DocumentError("duplicate_placement", "placement_id must be unique")
    return [entry for entry in placements if isinstance(entry, dict)]


HASH_ENTRY = re.compile(r"^sha256:[0-9a-f]{64}$")
#: The fault reason asking the owner to prepare a stale-shaped placement set again.
REPREPARE = "placement_set_reprepare_required"
SEMVER = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def _document(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise documents.DocumentError("wrong_type", f"{name} is a JSON object")
    return value


def _unique(values: list[str], name: str) -> None:
    if len(values) != len(set(values)):
        raise documents.DocumentError("duplicate_entry", f"{name} repeats an entry")


def _required(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    missing = fields - set(value)
    if missing:
        raise documents.DocumentError("missing_field", f"{name} lacks {sorted(missing)[0]}")


def _exact_ref(value: object, name: str) -> dict[str, Any]:
    row = _document(value, name)
    _required(row, {"digest", "length"}, name)
    digest = row["digest"]
    length = row["length"]
    if (
        not isinstance(digest, str)
        or HASH_ENTRY.fullmatch(digest) is None
        or not isinstance(length, int)
        or isinstance(length, bool)
        or length <= 0
    ):
        raise documents.DocumentError("invalid_ref", f"{name} is not an exact positive Ref")
    return row


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise documents.DocumentError("invalid_identifier", f"{name} is empty or padded")
    if len(value) > 256 or any(not 0x20 <= ord(char) <= 0x7E for char in value):
        raise documents.DocumentError("invalid_identifier", f"{name} is not bounded ASCII")
    return value


def _strings(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise documents.DocumentError("wrong_type", f"{name} is a list of strings")
    return value


def _wheel_fact(value: object, name: str) -> dict[str, Any]:
    row = _document(value, name)
    _required(row, {"ref", "distribution", "version", "filename"}, name)
    _exact_ref(row["ref"], f"{name}.ref")
    distribution = _identifier(row["distribution"], f"{name}.distribution")
    if distribution != distribution.lower() or any(
        not part or not part.isalnum() for part in distribution.split("-")
    ):
        raise documents.DocumentError("wheel_fact_invalid", f"{name}.distribution")
    _identifier(row["version"], f"{name}.version")
    _identifier(row["filename"], f"{name}.filename")
    _strings(row.get("import_roots", []), f"{name}.import_roots")
    _strings(row.get("tags", []), f"{name}.tags")
    return row


def _package_selection(value: object) -> dict[str, Any]:
    row = _document(value, "PackageSelection")
    _required(row, {"package", "release"}, "PackageSelection")
    package = _identifier(row["package"], "PackageSelection.package")
    _identifier(row["release"], "PackageSelection.release")
    if package.count("/") != 1 or not all(package.split("/")):
        raise documents.DocumentError("package_invalid", "package identity is invalid")
    return row


def _development_package(value: object) -> dict[str, Any]:
    row = _document(value, "DevelopmentPackage")
    _required(row, {"package", "release", "installation_id"}, "DevelopmentPackage")
    _identifier(row["package"], "DevelopmentPackage.package")
    _identifier(row["release"], "DevelopmentPackage.release")
    _identifier(row["installation_id"], "DevelopmentPackage.installation_id")
    return row


def _placement_profile(entry: dict[str, Any]) -> None:
    if "installation_id" not in entry or isinstance(entry.get("package_interface"), dict):
        # The shape an older Runtime prepared (interface by Ref, no installation). Nothing
        # is wrong with the request: its owner prepares the set again on this worker.
        raise documents.DocumentError(
            REPREPARE,
            "this placement set was prepared by an older Runtime; prepare the package set "
            "again on this worker",
        )
    _required(
        entry,
        {"placement_id", "installation_id", "package_interface", "bindings_digest", "entrypoints"},
        "Placement",
    )
    _identifier(entry["placement_id"], "Placement.placement_id")
    _identifier(entry["installation_id"], "Placement.installation_id")
    try:
        interface = base64.b64decode(entry["package_interface"], validate=True)
        if not 0 < len(interface) <= canonical.DOC_MAX_BYTES:
            raise ValueError("interface bound")
    except (ValueError, TypeError) as exc:
        raise documents.DocumentError(
            "package_interface_invalid", "invalid interface bytes"
        ) from exc
    if (
        not isinstance(entry["bindings_digest"], str)
        or HASH_ENTRY.fullmatch(entry["bindings_digest"]) is None
    ):
        raise documents.DocumentError("malformed_digest", "Placement.bindings_digest")
    published, development = "package" in entry, "development" in entry
    if published == development:
        raise documents.DocumentError("package_mode_invalid", "select exactly one package mode")
    if published:
        _package_selection(entry["package"])
    else:
        _development_package(entry["development"])

    models = entry.get("models", [])
    if not isinstance(models, list):
        raise documents.DocumentError("wrong_type", "Placement.models")
    model_ids: list[str] = []
    for index, value in enumerate(models):
        model = _document(value, "Model")
        _required(model, {"id", "repo", "manifest"}, "Model")
        model_ids.append(_identifier(model["id"], f"models[{index}].id"))
        _identifier(model["repo"], f"models[{index}].repo")
        if "version" in model:
            _identifier(model["version"], f"models[{index}].version")
        if "lane" in model:
            _identifier(model["lane"], f"models[{index}].lane")
        _exact_ref(model["manifest"], f"models[{index}].manifest")
    _unique(model_ids, "Placement.models")
    known_models = set(model_ids)

    entrypoints = entry["entrypoints"]
    if not isinstance(entrypoints, list):
        raise documents.DocumentError("wrong_type", "Placement.entrypoints")
    entrypoint_names: list[str] = []
    for entrypoint_index, value in enumerate(entrypoints):
        entrypoint = _document(value, "Entrypoint")
        _required(entrypoint, {"name", "entrypoint_binding_digest"}, "Entrypoint")
        entrypoint_names.append(
            _identifier(entrypoint["name"], f"entrypoints[{entrypoint_index}].name")
        )
        slots = entrypoint.get("slots", [])
        if not isinstance(slots, list):
            raise documents.DocumentError("wrong_type", "Entrypoint.slots")
        slot_names: list[str] = []
        for slot_index, value in enumerate(slots):
            slot = _document(value, "Slot")
            _required(slot, {"slot", "reference_model_id"}, "Slot")
            # Retired `stamps` and `model_construction_contract` from older writers are
            # ignored like any other unknown key: fit is the header and the census.
            slot_names.append(_identifier(slot["slot"], f"slots[{slot_index}].slot"))
            if slot["reference_model_id"] not in known_models:
                raise documents.DocumentError("unknown_model", "Slot.reference_model_id")
            components = slot.get("components", [])
            if not isinstance(components, list):
                raise documents.DocumentError("wrong_type", "Slot.components")
            component_names: list[str] = []
            for value in components:
                component = _document(value, "Component")
                _required(component, {"component", "model_id"}, "Component")
                component_names.append(_identifier(component["component"], "Component.component"))
                if component["model_id"] not in known_models:
                    raise documents.DocumentError("unknown_model", "Component.model_id")
            _unique(component_names, "Slot.components")
            adapters = slot.get("adapters", [])
            if not isinstance(adapters, list):
                raise documents.DocumentError("wrong_type", "Slot.adapters")
            for adapter in adapters:
                row = _document(adapter, "ModelAdapter")
                _required(
                    row, {"component", "model_id", "source_component", "scale"}, "ModelAdapter"
                )
                if row["model_id"] not in known_models:
                    raise documents.DocumentError("unknown_model", "ModelAdapter.model_id")
                _identifier(row["component"], "ModelAdapter.component")
                _identifier(row["source_component"], "ModelAdapter.source_component")
                if not isinstance(row["scale"], str) or not row["scale"].strip():
                    raise documents.DocumentError("invalid_scale", "ModelAdapter.scale")
        _unique(slot_names, "Entrypoint.slots")
        digest = entrypoint["entrypoint_binding_digest"]
        if not isinstance(digest, str) or HASH_ENTRY.fullmatch(digest) is None:
            raise documents.DocumentError(
                "malformed_digest", "Entrypoint.entrypoint_binding_digest"
            )
    _unique(entrypoint_names, "Placement.entrypoints")


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
) -> dict[str, Any]:
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
    construction for one kernel (cr-125). The worker never does: a request's pin, and the
    dev-mode `WorkerOptions.attention_pin` default, are execution parameters of `invoke`.
    """

    models = []
    for index, model in enumerate(binding.model_bindings()):
        names = {model.model_parameter_name}
        if index < len(parameter_names):
            names.update(parameter_names[index])
        models.append(
            {
                "binding": {
                    "application": binding.application,
                    "package_interface": binding.interface_path,
                    "development_project": binding.development_project,
                    "model_class": model.model_class,
                    "model_binding_path": model.model_binding_path,
                    "model_parameter_name": model.model_parameter_name,
                    "model_parameter_names": sorted(names),
                    "component": model.components[0] if model.components else "",
                    "components": list(model.components),
                    "snapshots": dict(model.snapshots),
                    "store": model.store,
                    "snapshot": model.reference_snapshot,
                    "release": binding.release,
                    "variant": model.variant,
                    "development": binding.development,
                    "custody": model.custody,
                    "objective": model.objective,
                    "steps_basis": model.steps_basis,
                    "package": binding.package,
                    "model": model.model,
                },
                "budgets": {"declared_weight_bytes": model.model_weight_bytes},
            }
        )
    command: dict[str, Any] = {
        "cmd": "load",
        "construction": construction,
        "devices": devices,
        "authorized_device_limit_bytes": int(authorized_device_limit_bytes),
    }
    if sequence_parallel_degree > 1:
        command["sequence_parallel_degree"] = int(sequence_parallel_degree)
    if attention_pin:
        command["attention_pin"] = attention_pin
    if not models:
        command.update(
            {
                "binding": {
                    "application": binding.application,
                    "package_interface": binding.interface_path,
                    "development_project": binding.development_project,
                    "model_class": "",
                },
                "budgets": {},
            }
        )
    elif len(models) == 1:
        command.update(models[0])
    else:
        command["models"] = models
    return command


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
        self.config = config
        self.options = options
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
        self.authorizer = grants.Authorizer(roots=tuple(options.grant_roots))
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
        self.ownership_path = self.root / "ownership.json" if options.worker_boot_id else None
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
        self.supervision.on_change = self.executor_changed
        # One accepted stream at a time may mutate control state. Claims take this through
        # their snapshot, so ownership cannot move in the middle of convergence and a frame
        # from the superseded stream cannot finish under its successor's epoch.
        self.control_lock = threading.RLock()
        self.preparation_lock = threading.Lock()
        self.model_source_operations: set[str] = set()
        self.restart_requested = False
        self.restart_sequence = 0
        self.idle_restart_pending = threading.Event()
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
        self.placement: Placement | None = None
        #: The INCOMING placement being staged while `placement` keeps serving (#474). It is
        #: a separate field precisely because one field cannot hold both, which is the same
        #: reason the observed state has two axes.
        self.pending: Placement | None = None
        #: Additional placements hosted beside the primary one, weightless or model-bearing.
        #: Each gets its own immutable uv environment and executor, sealed to the lane the
        #: worker assigned it; an attempt takes its own placement's lane seat and no other.
        self.hosted: dict[str, HostedPlacement] = {}
        self.pending_hosted: list[dict[str, Any]] = []
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
        #: thread name -> its kernel thread id, recorded by each `_lane`-wrapped thread as
        #: it starts (every device lane, convergence, report, watchdog, sender). An observer
        #: judging this worker's PROGRESS by CPU reads the threads that work and leaves out
        #: the ones that tick on a clock (`pollers`). Not the device lanes: those are
        #: `self.lanes`, the resource the threads serve.
        self.lane_threads: dict[str, int] = {}
        self._device_consumers: dict[DeviceLane, tuple[threading.Thread, ...]] = {}
        self._retired_consumers: list[threading.Thread] = []
        self._consumers_started = False
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
        self.workspace_service = (
            WorkspaceService(self.workspace, self.authorize_workspace, self.workspace_weights_ready)
            if self.workspace is not None
            else None
        )
        self.weights = weights.WeightsExchange(
            store_root=options.tensorfs_root,
            send=self.send,
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
            attention_pin=options.attention_pin,
            send=self.send,
            progress=self.emit_progress,
            bundles=self.bundles,
            monitor=self.monitor,
            posture=self.applied_posture,
            device_process=self.read_device_process,
            weights=self.weights,
            owner_scope=lambda: self.fence.record_owner_id,
            tensorfs_root=options.tensorfs_root,
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
            self.send,
            lambda: self.fence.control_stream_epoch,
            lambda: self.fence.wire_minor,
            workspace=self.workspace,
            owner=lambda: self.fence.record_owner_id,
            local=self._machine_call,
            prefetch=self._machine_prefetch,
            release_gpus=self._release_gpus,
            installed_bindings=self._installed_call_bindings,
        )
        from .machine_calls import Calls as MachineCalls
        from .machine_effects import Effects

        self.machine_calls = MachineCalls(self) if self.workspace is not None else None
        self.gpu = GpuScheduler(len(self.lanes.entries))
        self.prespawns = prespawn.Prespawns(self)
        self._gpu_shapes: dict[str, machine_lanes.Shape] = {}
        self._dead_replicas: dict[str, child.Executor] = {}
        self._activating: dict[str, threading.Thread] = {}
        from .machine_sources import Sources as MachineSources

        self.machine_sources = MachineSources(self)
        self.machine_effects = (
            Effects(
                self.workspace,
                self.executions,
                options.publication_authority,
                progress=self.native_progress,
            )
            if self.workspace is not None and self.executions is not None
            else None
        )
        self.engine.calls = self.calls
        from cozy_runtime.internal.worker.source_calls import SourceCalls

        self.source_calls = (
            SourceCalls(
                self.workspace,
                lambda: self.fence.record_owner_id,
                self.send,
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
            )
            if self.workspace is not None
            else None
        )
        from cozy_runtime.internal.worker.artifact_transfer import NativeArtifactTransfers

        self.artifact_transfers = (
            NativeArtifactTransfers(
                self.workspace,
                lambda: self.fence.record_owner_id,
                self.send,
                lambda command: (
                    command.record_owner_epoch == self.fence.record_owner_epoch
                    and command.control_stream_epoch == self.fence.control_stream_epoch
                    and command.worker_boot_id == self.fence.worker_boot_id
                ),
            )
            if self.workspace is not None
            else None
        )

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
        self.connected = threading.Event()
        #: The `control_stream_epoch` of every accepted RecordOwner stream being served right
        #: now. `connected` is narrower — it opens only at SnapshotAck — and a claim that has
        #: been accepted but not yet acked is still an owner holding this pod, which is the
        #: fact the supervisor's release decision rests on.
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
        self.reclaim_on_terminal = False
        self.boot_fatal: pb.BootFailure | None = None
        #: The per-spawn bootstrap credential (#449/#463): the launcher that spawned this
        #: worker handed it through the environment — read by the ONE config authority,
        #: never here — and a Claim must present it as `proof`. The check is constant-time;
        #: the value never appears in a log. Where no launcher minted one (a hand-started
        #: serve), the fence is the socket's own filesystem authority and it is absent.
        self.bootstrap = config.bootstrap_credential
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
        declared = max((binding.model_weight_bytes for binding in bindings.values()), default=0)
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
            slot for slot, _row in self.lanes.tenants(lane) if slot != placement.placement_id
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
        self._sync_lane_consumers()
        return True

    def _release_lane(self, placement_id: str) -> None:
        placement = self._placement_by_id(placement_id)
        if placement is not None:
            placement.lane_id = ""
        if self.lanes.release(placement_id):
            self._sync_lane_consumers()

    def lane_documents(self) -> list[dict[str, Any]]:
        """`lanes[]` (proto-024): every device or group lane of the envelope with its seats
        now, sorted by lane_id; `lane_wire.emit_lanes` writes it, the records carry it."""
        return sorted(
            (lane_wire.lane_document(lane, self._lane_slots(lane)) for lane in self.lanes.lanes),
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
            from . import machine_slots

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

    def _lane_phases(self, lane: DeviceLane) -> tuple[int, int]:
        """(attempts queued for or on the lane's DEVICES, from any lane over them; released
        attempts awaiting ack on ONE lane)."""
        sharing = {other.lane_id for other in self.lanes.sharing(lane)}
        seated = released = 0
        for held in list(self.engine.history.values()):
            if held.lane_id in sharing and held.state in self.engine.PRE_RELEASE:
                seated += 1
            elif held.lane_id == lane.lane_id and held.state in self.engine.RELEASED:
                released += 1
        return seated, released

    def _lane_capacity(self, lane: DeviceLane) -> int:
        """The seats `lane` has: one running plus one staged on a serving device lane, one
        on a job lane (the envelope, a CPU slot, orchestration), which stages nothing."""
        if self.accepted.mode == "serving" and lane in self.lanes.lanes:
            return LANE_RUNNING + LANE_STAGED
        return LANE_RUNNING

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

    def _lane_slots(self, lane: DeviceLane, *, machine_owned: bool = False) -> int:
        """Free seats on ONE lane, now: its capacity less what is queued for or on its
        device, NONE while
        more than `LANE_RELEASED` released attempts await their ack (cr-079 D3), and NONE
        while the worker's admission is CLOSED — that is one structural fact per worker,
        and `lanes[]` must never show a seat the worker-level count denies."""
        if (
            self.admission_state(machine_owned=machine_owned)
            != pb.AdmissionState.ADMISSION_STATE_OPEN
        ):
            return 0
        if not self._lane_dispatchable(lane):
            return 0
        seated, released = self._lane_phases(lane)
        if released > LANE_RELEASED:
            return 0
        return max(0, self._lane_capacity(lane) - seated)

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
            if rebuild and not self.stop.is_set():
                self._lane_of(placement_id).work.put(ExecutorDeath(executor, placement_id))

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
        if len(prepared) != 1 or not prepared[0].get("development"):
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

            def mark_verified(digest: str, length: int, path: Path) -> None:
                self.verified_artifacts.record(digest, length, path)

            with contextlib.ExitStack() as access:
                if request.native_models:
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
                )
                if request.native_models:
                    self.authorize_workspace(request.claim)
            joined = read_placement_set(result.placement_set)
            if len(joined) != 1 or joined[0].get("installation_id") != revision:
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
                return self._prepare_published(request)
            assert isinstance(request, pb.PrepareLocalPackageRequest)
            return self._prepare_local_in(request)

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
        """Import package code only inside a dedicated preparation executor."""
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
                        {
                            "cmd": "describe_installed",
                            "distribution": distribution,
                            "application": application,
                            "output": str(output),
                        },
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
            request,
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
        if len(entries) != 1 or not entries[0].get("package") or len(published) != 1:
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
            package_installation.remove(root, identifier)
            shutil.rmtree(self.config.cozy_home / "job-plans" / identifier, ignore_errors=True)
            for key in [key for key in self.prepared_installations if key[1] == identifier]:
                del self.prepared_installations[key]
            self.note("installation", f"collected superseded {installed.package} {identifier}")

    def numerical_environment(self) -> bytes:
        from cozy_runtime.internal.numerical_environment import fingerprint

        return fingerprint(
            hostfacts.measure(self.options.accelerator_backend),
            threads=self.options.threads,
            inherited=dict(self.config.child_base_env),
        )

    def authorize_workspace(self, claim: pb.Claim) -> str:
        from cozy_runtime.internal.worker.workspace import WorkspaceRefusal

        with self.claim_lock:
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
            if self.config.record_owner_public_key:
                if (
                    claim.worker_boot_id != self.fence.worker_boot_id
                    or claim.worker_id != self.options.worker_id
                ):
                    raise WorkspaceRefusal("workspace proof names another worker lifetime")
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
            elif (
                not self.bootstrap
                or not hmac.compare_digest(claim.proof, self.bootstrap.encode())
                or claim.worker_id not in ("", self.options.worker_id)
                or claim.worker_boot_id not in ("", self.fence.worker_boot_id)
            ):
                raise WorkspaceRefusal("workspace bootstrap proof was refused")
            return self.fence.record_owner_id

    def _execution_service(self, claim: pb.Claim) -> tuple[workspace_executions.Executions, str]:
        owner = self.authorize_workspace(claim)
        if self.executions is None:
            raise ValueError("worker has no execution workspace")
        return self.executions, owner

    def _machine_prefetch(self, attempt: AttemptRecord, frame: Mapping[str, Any]) -> dict[str, Any]:
        if self.executions is None or not self.executions.owns(
            self.fence.record_owner_id, attempt.request_id
        ):
            return {"ok": True, "scheduled": False}
        assert self.machine_calls is not None
        return self.machine_calls.serving.prefetch(attempt, frame)

    def _release_gpus(self, attempt: AttemptRecord) -> None:
        """`ctx.release_gpus()`: the attempt's root ends its GPU lease now (the next drive
        tick grants the next waiter). A direct attempt holds no lease."""
        owner = self.fence.record_owner_id
        if self.executions is None or not self.executions.owns(owner, attempt.request_id):
            return
        root, _parent = self.executions.scheduling_root(owner, attempt.request_id)
        self.gpu.release_root(root)

    def _machine_call(
        self, attempt: AttemptRecord, request: pb.ChildCallRequest, action: str
    ) -> pb.ChildCallResult | None:
        if self.executions is None or not self.executions.owns(
            self.fence.record_owner_id, attempt.request_id
        ):
            return None
        assert self.machine_calls is not None
        return self.machine_calls.handle(attempt, request, action)

    def submit_execution(
        self,
        claim: pb.Claim,
        submission_id: str,
        capture_digest: bytes,
        offer: pb.AttemptOffer,
        *,
        expected_execution_workspace_id: str,
        max_attempts: int = 1,
        retry_delay_ms: int = 0,
        capture_document: bytes = b"",
        preparation: bytes = b"",
        publication_authorization_id: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> workspace_executions.Receipt:
        """Authenticated root intake; accepted execution never depends on the caller's stream."""
        with self.control_lock:
            if self.stop.is_set():
                raise WorkspaceRefusal("worker is stopping; root admission is closed")
            executions, owner = self._execution_service(claim)
            executions.require_workspace(expected_execution_workspace_id)
            if publication_authorization_id and self.options.publication_authority is None:
                raise WorkspaceRefusal("worker has no independent publication authority configured")
            result_schema, memoize = b"", False
            spec = documents.read(offer.invocation_spec_canonical_bytes, pb.InvocationSpec)
            if "job" in spec:
                descriptor = spec["job"]["job_descriptor_id"]
                binding = self.engine.jobs.get(descriptor)
                if binding is None or binding.installation_id != spec["job"]["installation_id"]:
                    from cozy_runtime.internal import job_plan

                    binding = JobBinding.read(
                        json.loads(
                            job_plan.path(
                                self.config.cozy_home / "job-plans",
                                spec["job"]["installation_id"],
                                descriptor,
                            ).read_bytes()
                        )
                    )
                declaration = self.engine._job_declaration(binding, descriptor)
                result_schema = canonical.write(declaration["result"])
                memoize = declaration.get("invocable", {}).get("memoize", False) is True
            from . import machine_checkpoint_inputs

            with machine_checkpoint_inputs.admission(executions.workspace, owner, offer.request_id):
                if "job" in spec:
                    from . import machine_model_inputs

                    assert binding is not None
                    declared = set(self.engine._job_model_declarations(binding, descriptor))
                    if declared and claim.wire_minor < weights_limits.NATIVE_ROOT_INPUTS_WIRE_MINOR:
                        raise WorkspaceRefusal("durable root Model inputs require Runtime wire55")
                    machine_model_inputs.retain(executions.workspace, owner, offer, declared)
                    from . import machine_byte_inputs

                    machine_byte_inputs.retain_root(
                        executions.workspace,
                        owner,
                        offer,
                        declaration["request"],
                        arguments or {},
                        wire_minor=claim.wire_minor,
                    )
                receipt = executions.submit(
                    owner,
                    submission_id,
                    capture_digest,
                    offer,
                    expected_execution_workspace_id=expected_execution_workspace_id,
                    result_schema=result_schema,
                    worker_boot=self.fence.worker_boot_id,
                    memoize=memoize,
                    max_attempts=max_attempts,
                    retry_delay_ms=retry_delay_ms,
                    capture_document=capture_document,
                    preparation=preparation,
                    worker_id=self.options.worker_id,
                    publication_authorization_id=publication_authorization_id,
                )
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
        with self.control_lock:
            if action == "resume" and self.stop.is_set():
                raise WorkspaceRefusal("worker is stopping; execution resume is closed")
            executions, owner = self._execution_service(claim)
            state = executions.control(
                owner,
                request_id,
                command_id,
                generation,
                action,
                worker_boot=self.fence.worker_boot_id,
            )
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
        return executions.status(owner, request_id)

    def drive_executions(self) -> None:
        """A custody handoff fences dispatch without killing the lifecycle watchdog."""
        from .workspace import WorkspaceInactive

        try:
            self._drive_executions()
        except WorkspaceInactive:
            return

    def _drive_executions(self) -> None:
        """Admit retained executions; every GPU phase runs on its scheduler grant."""
        if self.executions is None or not self.fence.record_owner_recorded:
            return
        with self.control_lock:
            if self.stop.is_set():
                return
            owner = self.fence.record_owner_id
            if self.machine_effects is not None:
                self.machine_effects.drive(owner)
            if self.machine_calls is not None:
                self.machine_calls.drive()
            self.executions.retry_due(owner, worker_boot=self.fence.worker_boot_id)
            states = {r: self.executions.status(owner, r) for r in self.executions.pending(owner)}
            grants = self._schedule_gpus(owner, states)
            for request, state in states.items():
                held = self.engine.history.get((request, state.attempt_ordinal))
                if state.state in workspace_executions.TERMINAL:
                    if held is None or held.state == "closed":
                        self.executions.abandon(owner, request)
                    continue
                if not self.executions.assigned_here(
                    owner, request, state.attempt_ordinal, self.fence.worker_boot_id
                ):
                    continue
                if state.state in ("pausing", "canceling"):
                    if held is None:
                        self.executions.stop_queued(owner, request)
                    else:
                        self.cancel(
                            pb.CancelAttempt(
                                request_id=request,
                                attempt_ordinal=state.attempt_ordinal,
                                invocation_spec_digest=held.digest,
                                reason=pb.CANCEL_REASON_CLIENT,
                            )
                        )
                        if self.source_calls is not None:
                            self.source_calls.cancel_parent(
                                request, ordinal=state.attempt_ordinal, spec=held.digest
                            )
                    continue
                if state.state != "queued" or held is not None:
                    continue
                if any(
                    attempt.request_id == request
                    and attempt.attempt < state.attempt_ordinal
                    and attempt.state != "closed"
                    for attempt in self.engine.history.values()
                ):
                    continue
                offered = self.executions.offer(owner, request)
                machine_byte_inputs.bind_root_grant(owner, offered)
                spec = documents.read(offered.invocation_spec_canonical_bytes, pb.InvocationSpec)
                deadline = int(spec.get("deadline_unix_ms", 0))
                if deadline and self.executions.clock_ms() >= deadline:
                    self.executions.stop_queued(owner, request, deadline=True)
                    continue
                shape = self._gpu_shapes[request]
                if shape.refusal:
                    self.refuse(
                        offered,
                        shape.refusal.partition(":")[0],
                        shape.refusal,
                        pb.CAUSE_CODE_CONSTRAINT_INFEASIBLE,
                    )
                    continue
                ordinals = grants.get(f"{request}#{state.attempt_ordinal}", ())
                if shape.kind != "cpu" and not ordinals:
                    continue
                try:
                    preparation = machine_lanes.placement_for(
                        self.executions.preparation(owner, request), offered, ordinals
                    )
                    if preparation and not self._prepare_execution(
                        preparation,
                        request,
                        functools.partial(self._journal_row, request, state.attempt_ordinal),
                    ):
                        continue
                except Exception as exc:
                    self.refuse(
                        offered,
                        "execution_preparation_unavailable",
                        f"Retained preparation is unavailable ({type(exc).__name__}): {exc}",
                        pb.CAUSE_CODE_CAPABILITY_UNAVAILABLE,
                    )
                    continue
                # Validation precedes waiting for capacity, so malformed work does not
                # occupy a durable queue forever. The engine still authors its refusal.
                try:
                    attempt = self.engine.offer(offered)
                except AttemptRefusal:
                    offered.admission_epoch = self.admission_epoch
                    self.on_offer(offered, machine_owned=True)
                    continue
                if fault := self._failed_execution_binding(attempt):
                    self.refuse(
                        offered,
                        "binding_faulted",
                        f"Model preparation failed: {fault}",
                        pb.CAUSE_CODE_EXECUTOR_FAULT,
                    )
                    continue
                # Desired-state replacement can still expose an older warm placement
                # while the selected package is materializing or its executor starts.
                # Capacity on that old placement is not readiness for this exact offer.
                # Keep the durable child queued; explicit preparation faults above still
                # fail immediately, and malformed offers were already validated.
                if "serving" in attempt.spec and not self._placement_dispatchable(
                    attempt.placement_id
                ):
                    continue
                lane = self._offer_lane(offered.placement_id, attempt)
                if self._lane_slots(lane, machine_owned=True) < 1:
                    continue
                offered.admission_epoch = self.admission_epoch
                self.on_offer(offered, machine_owned=True)

    def _schedule_gpus(
        self, owner: str, states: Mapping[str, workspace_executions.State]
    ) -> dict[str, tuple[int, ...]]:
        """One level-triggered scheduler pass over every execution that needs a GPU now.

        A queued execution demands; an attempt keeps its grant until device release. A
        GPU held outside this owner's executions (a direct offer) is excluded, and so is
        an unreadable one for model work: its width is what the readable GPUs can form.
        """
        assert self.executions is not None
        self._gpu_shapes = {r: s for r, s in self._gpu_shapes.items() if r in states}
        outside = {
            ordinal
            for attempt in self.engine.history.values()
            if attempt.state in self.engine.PRE_RELEASE and attempt.request_id not in states
            for ordinal in getattr(self.lanes.by_id.get(attempt.lane_id), "ordinals", ())
        }
        readable: tuple[int, ...] | None = None
        demands: list[Demand] = []
        #: grant key -> the installation it runs, so a Host-started process finds its root
        installations: dict[str, str] = {}
        for request, state in states.items():
            key = f"{request}#{state.attempt_ordinal}"
            held = self.engine.history.get((request, state.attempt_ordinal))
            if held is not None:
                granted = self.gpu.granted(key)
                if held.state in self.engine.PRE_RELEASE and granted is not None:
                    demands.append(Demand(key, self._gpu_shapes[request].root, len(granted)))
                    installations[key] = self._gpu_shapes[request].template[0]
                continue
            if state.state != "queued" or not self.executions.assigned_here(
                owner, request, state.attempt_ordinal, self.fence.worker_boot_id
            ):
                continue
            if request not in self._gpu_shapes:
                try:
                    self._gpu_shapes[request] = machine_lanes.shape(self, owner, request)
                except Exception as exc:  # only this execution is refused
                    self._gpu_shapes[request] = machine_lanes.Shape(
                        request, "cpu", refusal=f"gpu_shape_unreadable: {type(exc).__name__}: {exc}"
                    )
            shape = self._gpu_shapes[request]
            if shape.kind == "cpu" or shape.refusal:
                continue
            exclude = set(outside)
            width = len(self.lanes.entries)
            if shape.kind == "serving":
                if readable is None:
                    family = accel.host_backend_family()
                    readable = tuple(
                        o
                        for o, entry in enumerate(self.lanes.entries)
                        if accel.device_memory(entry, family).state == "measured"
                    )
                if not readable:
                    self._gpu_shapes[request] = dataclasses.replace(
                        shape, refusal="gpu_unavailable: no GPU on this worker is readable"
                    )
                    continue
                exclude.update(set(range(width)) - set(readable))
                width = width_for(shape.degrees, len(readable), shape.gpus)
                if not width:
                    self._gpu_shapes[request] = dataclasses.replace(
                        shape,
                        refusal=f"gpu_count_unavailable: exactly {shape.gpus} GPUs is not a "
                        f"group every model slot declares on {len(readable)} readable GPUs",
                    )
                    continue
            demands.append(
                Demand(key, shape.root, width, self._warm(shape.template), tuple(sorted(exclude)))
            )
            installations[key] = shape.template[0]
        roots = self.executions.scheduling_roots(owner)
        grants = self.gpu.sync(roots, demands)
        self.prespawns.reconcile(roots, self.gpu.view(), installations)
        for root, event, body in self.gpu.drain():
            with contextlib.suppress(WorkspaceRefusal):
                self.executions.record(owner, root, event, body)
            if event == "gpu.wait":
                self._gpu_waiting(owner, body)
        return grants

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
                    and (placement.installation_id, placement.document.get("bindings_digest"))
                    == template
                    for ordinal in placement.device_pin or ()
                }
                | set(self.prespawns.ordinals(template[0]))
            )
        )

    def _gpu_waiting(self, owner: str, body: dict[str, Any]) -> None:
        """Once per transition, tell the waiting call's parent (or root) why it waits."""
        assert self.executions is not None
        request = str(body["key"]).rpartition("#")[0]
        _root, parent = self.executions.scheduling_root(owner, request)
        target = self.engine.live.get(parent or request)
        blocked = ", ".join(body["blocked_by"]) or "released devices"
        if target is not None:
            self.emit_progress(
                target.request_id,
                target.attempt,
                {
                    "kind": "progress",
                    "stage": f"Waiting for GPU (needs {body['width']}, behind {blocked})"[:120],
                },
            )

    def _defer_rebuild(self, placement_id: str) -> bool:
        """A dead GPU replica rebuilds on its next grant, never on devices it no longer holds."""
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
            return False
        hosted = self.hosted.get(placement_id)
        current = (hosted.supervision if hosted is not None else self.supervision).current
        if current is not None:
            self._dead_replicas[placement_id] = current
        return current is not None

    def _revive(self, placement_id: str) -> None:
        """Rebuild a deferred dead replica now that a grant holds its devices."""
        executor = self._dead_replicas.pop(placement_id, None)
        if executor is not None and self._placement_by_id(placement_id) is not None:
            lane = self.lanes.lane_of(placement_id)
            if lane is not None:
                lane.work.put(ExecutorDeath(executor, placement_id, granted=True))

    def _cpu_job(self, job: pb.JobDirective) -> bool:
        """An orchestration job granted no device holds no GPU (`machine_slots.admits`)."""
        if not job.orchestration:
            return False
        binding = JobBinding.read(
            json.loads(
                job_plan.path(
                    self.config.cozy_home / "job-plans", job.installation_id, job.job_descriptor_id
                ).read_bytes()
            )
        )
        declaration = self.engine._job_declaration(binding, binding.job_descriptor_id)
        return machine_slots.admits(job, declaration)

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

    def _prepare_execution(
        self,
        preparation: dict[str, Any],
        request_id: str = "",
        journal: Callable[[dict[str, Any]], None] | None = None,
    ) -> bool:
        """Reuse the existing preparation/convergence engine without replacing live work."""
        if self.phase == pb.WorkerPhase.WORKER_PHASE_FAILED:
            # A failed convergence lane can never satisfy the pending-revision
            # wait below. Refuse through the existing durable execution path so
            # a parent awaiting this child receives the originating failure.
            fault = next(
                (
                    fault
                    for fault in reversed(self.engine.faults)
                    if fault.subject == self.options.worker_id
                ),
                None,
            )
            detail = fault.reason if fault is not None else "worker entered FAILED state"
            raise WorkspaceRefusal(f"worker preparation stopped: {detail}")

        desired = machine_lanes.state(preparation)
        if request_id and desired.WhichOneof("mode") == "job" and self._cpu_job(desired.job):
            held = self.job_slots.get(machine_slots.key(request_id))
            if held is not None and held.ready():
                return True
            binding = JobBinding.read(
                json.loads(
                    job_plan.path(
                        self.config.cozy_home / "job-plans",
                        desired.job.installation_id,
                        desired.job.job_descriptor_id,
                    ).read_bytes()
                )
            )
            machine_capture.restore(self, preparation)
            return machine_slots.ensure(self, request_id, desired.job, binding)
        if request_id and desired.WhichOneof("mode") == "placement_set":
            return self._bind_replica(preparation, desired, journal)
        desired.revision = self.accepted.accepted_desired_state_revision
        if desired_state_body(desired) == self.accepted.desired_state_body:
            return True
        if any(
            (attempt.state in self.engine.PRE_RELEASE or attempt.state in self.engine.HELD)
            and not (
                desired.WhichOneof("mode") == "job"
                and not desired.job.orchestration
                and attempt.lane_id.startswith("cpu-")
                and attempt.state != "queued"
            )
            for attempt in self.engine.history.values()
        ):
            # A device job replaces the whole worker: it waits for every held attempt.
            return False
        machine_capture.restore(self, preparation)
        desired.revision = self.accepted.accepted_desired_state_revision + 1
        self._apply_desired_state(desired)
        return self.dispatchable()

    def _bind_replica(
        self,
        preparation: dict[str, Any],
        desired: pb.DesiredWorkerState,
        journal: Callable[[dict[str, Any]], None] | None = None,
    ) -> bool:
        """Activate this machine replica on its grant, beside every other placement.

        No desired-state revision is written, so replicas on other GPUs activate at once
        and a failed one refuses only its own demand (the next demand starts it afresh).
        Replicas already on the granted devices stay: a device holds several tenants and
        residency arbitration, not activation, decides whose weights are on the card.
        """
        (row,) = read_placement_set(desired.placement_set)
        pins = desired.placement_set.device_pins
        pin = tuple(pins[0].device_ordinals) if pins else None
        placement_id = str(row["placement_id"])
        hosted = self.hosted.get(placement_id)
        if hosted is not None:
            busy = self._activating.get(placement_id)
            if self._placement_dispatchable(placement_id) or (busy and busy.is_alive()):
                return self._placement_dispatchable(placement_id)
            placement = hosted.placement
            if (
                hosted.latched
                or placement.serving == pb.ServingState.SERVING_STATE_OFFLINE
                or placement.materialization
                in (pb.MATERIALIZATION_STATE_FAILED, pb.MATERIALIZATION_STATE_ABSENT)
            ):
                # This demand is refused; the replica leaves in the background so the next
                # demand activates it afresh instead of inheriting the failure.
                self._lifecycle(placement_id, self._retire_one, placement_id)
                fault = next(
                    (f for f in reversed(self.engine.faults) if f.subject == placement_id), None
                )
                detail = f"{fault.reason}: {fault.detail}" if fault else hosted.latched
                raise WorkspaceRefusal(f"model preparation failed: {detail or placement_id}")
            self._revive(placement_id)
            return False
        machine_capture.restore(self, preparation)
        placement = self._placement_from_entry(row, desired.placement_set.placement_set_digest)
        placement.device_pin = pin
        # The process started beside this replica's download (h3a-087), when there is one.
        # One the Host started for a root it had not submitted yet journals its phases here.
        warm = self.prespawns.claim(placement.installation_id, pin or (), journal)
        supervision = (
            warm.supervision
            if warm is not None
            else self._new_placement_supervision(
                placement_id, uid_key=prespawn.uid_key(placement.installation_id, pin or ())
            )
        )
        hosted = HostedPlacement(placement, supervision)
        self.hosted[placement_id] = hosted
        job = self.accepted.mode == "job"
        if job:
            # A device job's slot leaves with it; CPU job slots keep serving beside replicas.
            self.lanes.envelope.forget("")
            self.job_slots.pop("envelope", None)
        self.accepted.mode = "serving"
        self._lifecycle(placement_id, self._activate_replica, hosted, job, warm)
        return False

    def _lifecycle(self, placement_id: str, body: Callable[..., Any], *args: Any) -> None:
        """One replica's activation or retirement, off the drive tick."""
        self._activating = {k: v for k, v in self._activating.items() if v.is_alive()}
        name = f"replica-{placement_id}"
        thread = threading.Thread(target=self._lane(name, body, *args), name=name, daemon=True)
        self._activating[placement_id] = thread
        thread.start()

    def _activate_replica(
        self, hosted: HostedPlacement, job: bool, warm: prespawn.Slot | None = None
    ) -> None:
        """Acquire, bind and start this replica's executor; a device job's leaves first."""
        try:
            if warm is not None:
                self.prespawns.settle(warm, hosted.placement.placement_id)
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
        return executions.acknowledge_collection(owner, ack)

    def workspace_weights_ready(self, call: pb.WeightsIntentReadyCall) -> pb.WeightsHostAck:
        from cozy_runtime.internal.worker.workspace import WorkspaceRefusal

        owner = self.authorize_workspace(call.claim)
        if self.workspace is None:
            raise WorkspaceRefusal("worker has no workspace")
        value = self.workspace.ready_weights(owner, call.request)
        row = value
        self.send(pb.WorkerFrame(weights_transaction=self.workspace.status(value)))
        ack = pb.WeightsHostAck(
            request_id=row["request"],
            attempt_ordinal=row["ordinal"],
            invocation_spec_digest=row["spec"],
            output_slot=row["slot"],
            weights_transaction_id=row["id"],
            writer_epoch=row["epoch"],
            tensorfs_declaration_digest=row["declaration_digest"],
            stage=pb.WEIGHTS_HOST_STAGE_INTENT,
            outcome=pb.WEIGHTS_HOST_OUTCOME_RECORDED,
            record_owner_epoch=call.request.record_owner_epoch,
            worker_boot_id=call.request.worker_boot_id,
        )
        if row["restore_checkpoint"]:
            ack.checkpoint.CopyFrom(pb.CheckpointRef.FromString(row["restore_checkpoint"]))
        return ack

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
        so a frame authored before a re-claim still leaves under the live envelope, and an
        unacknowledged terminal is replayed after a reconnect.

        STAMP AND ENQUEUE ARE ONE STEP, under the same lock claim acceptance takes: the
        envelope a frame carries and the stream it is queued on are then the same epoch
        by construction, with no window where a frame is stamped for one stream and handed to
        another's sender.
        """
        inner = getattr(frame, frame.WhichOneof("msg"))
        if frame.HasField("attempt_accepted") and self.workspace is not None:
            self.workspace.mark_running(self.fence.record_owner_id, frame.attempt_accepted)
        if (
            frame.HasField("attempt_outcome")
            and self.workspace is not None
            and self.workspace.contains(
                self.fence.record_owner_id,
                frame.attempt_outcome.request_id,
                frame.attempt_outcome.attempt_ordinal,
            )
        ):
            # Outcome forwarding follows the same workspace commit that observes
            # native receipts. A transport projection is never the authority.
            self.workspace.outcome(self.fence.record_owner_id, frame.attempt_outcome)
        execution_subject = (
            frame.attempt_outcome
            if frame.HasField("attempt_outcome")
            else frame.attempt_accepted
            if frame.HasField("attempt_accepted")
            else None
        )
        managed = (
            execution_subject is not None
            and self.executions is not None
            and self.executions.owns(self.fence.record_owner_id, execution_subject.request_id)
        )
        if managed and self.executions is not None:
            self.executions.observe(self.fence.record_owner_id, frame)
        subject_id = getattr(inner, "request_id", "") or getattr(inner, "parent_request_id", "")
        machine_message = (
            self.executions is not None
            and bool(subject_id)
            and self.executions.owns(self.fence.record_owner_id, subject_id)
        )
        if machine_message and frame.HasField("native_source_status"):
            self.machine_sources.observe(frame.native_source_status)
        if machine_message and frame.HasField("child_call_request"):
            request = frame.child_call_request
            self.calls.absorb(
                pb.ChildCallResult(
                    parent_request_id=request.parent_request_id,
                    parent_attempt_ordinal=request.parent_attempt_ordinal,
                    parent_invocation_spec_digest=request.parent_invocation_spec_digest,
                    call_index=request.call_index,
                    intent_digest=request.intent_digest,
                    state=pb.CHILD_CALL_STATE_REFUSED,
                    safe_code="machine_children_unavailable",
                    safe_detail="This Runtime does not yet support child operations for scripts.",
                )
            )
        with self.claim_lock:
            if frame.WhichOneof("msg") != "weights_transaction":
                self.stamp(inner)
            if not machine_message:
                self.outbound.put(frame)
        if managed and frame.HasField("attempt_outcome"):
            terminal = frame.attempt_outcome
            # Runtime has committed exact result custody. Free the lane without
            # waiting for an observer; collection is a separate durable obligation.
            self.on_outcome_ack(
                self.stamp(
                    pb.AttemptOutcomeAck(
                        request_id=terminal.request_id,
                        attempt_ordinal=terminal.attempt_ordinal,
                        invocation_spec_digest=terminal.invocation_spec_digest,
                        outcome_id=terminal.outcome_id,
                        outcome_digest=terminal.outcome_digest,
                        retain_work=True,
                    )
                )
            )

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
        if self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))

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
        if self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))

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
        """Can this worker take an offer AT ALL right now? Serving asks the placement's
        axis; job mode asks whether its one build is ready. Neither answer is a SLOT count —
        `available_attempt_slots` is the other half, and #486c keeps them apart on purpose."""
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

    def admission_state(self, *, machine_owned: bool = False) -> pb.AdmissionState:
        """CLOSED is STRUCTURAL (#486c): before the snapshot barrier, while draining, while
        the placement is not dispatchable. OPEN with zero slots is TRANSIENT saturation, and
        a RecordOwner backs off differently for each — so the two never collapse into one
        counter here either."""
        if (
            (not machine_owned and not self.fence.dispatch_open)
            or self.accepted.posture == pb.Posture.POSTURE_DRAINING
            or self.phase != pb.WorkerPhase.WORKER_PHASE_ONLINE
            or not self.dispatchable()
        ):
            return pb.AdmissionState.ADMISSION_STATE_CLOSED
        return pb.AdmissionState.ADMISSION_STATE_OPEN

    def available_attempt_slots(self) -> int:
        """Free seats across this worker's lanes, NOW — the SUM of each lane's free seat.

        BOUNDEDNESS IS STRUCTURAL, NOT A PROMISE (§5): a seat is held from an offer's hold
        to DEVICE RELEASE, and a lane whose released attempts awaiting `AttemptOutcomeAck`
        exceed `LANE_RELEASED` advertises none (cr-079 D3), so a RecordOwner that stops
        acking runs itself out of admission instead of growing this worker's held set. On the
        one-lane worker this is exactly the old single window; on a multi-lane worker the
        sum is honest only for an owner that reads which lane an offer draws from
        (`lanes[]`, proto-024), which is why serving on more than one lane refuses an owner
        below minor 22. CLOSED admission is zero on every lane, so the sum is zero.
        """
        return sum(self._lane_slots(lane) for lane in self.lanes)

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
        """A machine-scope failure that closes admission.

        `WORKER_PHASE_FAILED` is where that fact lives now (#482) — machine lifecycle is out
        of the placement enum, because "the placement cannot serve" and "the worker cannot
        write" are different facts with different remedies, and one `INTAKE_STATE_ERROR`
        said both.
        """
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

    def _lane(self, name: str, body: Callable[..., None], *args: Any) -> Callable[[], None]:
        """Wrap one worker lane so its DEATH is a reported fact rather than a silence.

        Each lane below is the sole producer of some terminal: the device lane produces
        attempt outcomes, convergence produces a placement's verdict, the sender is the only
        thing that puts a frame on the stream. Nothing on the wire describes a dead Python
        thread, so a lane that dies leaves whoever waits on that terminal waiting on a
        producer that no longer exists — the exact shape cr-061 hit from the other direction.

        The answer is the one fact every owner already reads: `WORKER_PHASE_FAILED`. And
        where the host IS the stream — the local one-shot door, whose `persistent` is False —
        the stream is closed too, so the process EXITS with a non-zero code instead of
        sitting on a queue nothing will ever fill again.
        """

        def guarded() -> None:
            native_id = threading.get_native_id()
            self.lane_threads[name] = native_id
            try:
                body(*args)
            except BaseException as exc:
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
        """Withdraw an exact dead epoch and ask the device lane to rebuild it.

        The pidfd watcher does not reap.  Invalidation clears the ready-plan set and moves
        admission before this callback returns; replacement remains queued behind whichever
        attempt already owns the serialized lane.  Both transitions target ``executor``, so
        a watcher for E1 cannot invalidate or rebuild E2.
        """

        with self.control_lock:
            rebuild = bool(
                (self.accepted.mode == "job" and self.job_ready)
                or (
                    self.placement is not None
                    and self.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
                )
            )
            if not self.supervision.invalidate(
                executor,
                f"executor epoch {executor.epoch} pid {executor.pid} exited",
            ):
                return
            if rebuild and not self.stop.is_set():
                self._primary_lane().work.put(ExecutorDeath(executor))

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
        """The kernel thread ids of the lanes that wake on a CLOCK — the reporter, the
        watchdog, and the sender that delivers what the reporter ticks — whose CPU says
        nothing about whether this worker is getting anywhere. Every other thread burns
        only when it works, which is what makes the worker's own burn, less these, an
        honest progress meter for an outside observer."""
        clocks = {
            tid
            for name, tid in self.lane_threads.items()
            if name in ("report", "watchdog", "control-send")
        }
        if self.machine_effects is not None:
            clocks.update(self.machine_effects.pollers)
        return clocks

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
        key = (request_id, attempt)
        if self.executions is not None and self.executions.owns(
            self.fence.record_owner_id, request_id
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
            self.note("session", f"the control stream ended: {type(exc).__name__}: {exc}")
        finally:
            # Unconditional: a SUPERSEDED stream is gone too, and leaving its epoch here
            # would report an owner that has been fenced out as still holding the pod.
            self.attached_streams.discard(epoch)
            if self.fence.control_stream_epoch == epoch:
                self.connected.clear()
                self.fence.dispatch_open = False

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
            self.note("claim", f"REFUSED ({pb.ClaimRejection.Name(reason)}): {why}")
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

        # THE CREDENTIAL CHECK, in whichever of the two launch modes this worker was
        # started in (#560g). A rented pod verifies BY HASH — it holds no copy of the
        # token, so neither does whoever provisioned it — and a self-spawned worker
        # compares against the credential its own launcher minted. Both are constant-time
        # and neither ever renders what it holds into a refusal.
        if self.config.record_owner_public_key:
            try:
                downloads.verify_claim_proof(
                    claim,
                    self.config.record_owner_public_key,
                    worker_id=self.options.worker_id,
                    worker_boot_id=self.fence.worker_boot_id,
                    worker_tls_certificate_digest=self.options.worker_tls_certificate_digest,
                )
            except downloads.DownloadRefusal as exc:
                return reject(
                    pb.ClaimRejection.CLAIM_REJECTION_UNAUTHENTICATED,
                    exc.detail,
                )
        elif self.bootstrap and not hmac.compare_digest(
            claim.proof.decode(errors="replace"), self.bootstrap
        ):
            return reject(
                pb.ClaimRejection.CLAIM_REJECTION_UNAUTHENTICATED,
                "the bootstrap credential is absent or wrong",
            )
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
                self.fence.dispatch_open = False
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
            self.fence.dispatch_open = False
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
        return epoch, outbound

    def mark_claim_ready(self) -> None:
        """Open Claim after fixed-host listener proof, before readiness publication."""

        self._publish_restart_fact("restart-capability.json", {"supports_guarded_restart": True})
        self.claim_ready.set()

    def record_ownership(self, claim: pb.Claim) -> None:
        """Record the authority this claim carries before ClaimAck.

        Called under `claim_lock` and the existing process-lifetime worker-root lease.
        """
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
            available_attempt_slots=self.available_attempt_slots(),
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
            body.weights_transactions.extend(
                self.workspace.status(row)
                for row in self.workspace.projected_weights(
                    self.fence.record_owner_id, self.fence.worker_boot_id
                )
            )
        if self.accepted.mode == "serving":
            lane_wire.emit_lanes(body, self.lane_documents())
        lane_wire.emit_held_manifests(body, self.held_manifests.digests())
        return body

    def _send_snapshot(self, send: Callable[[pb.WorkerFrame], None]) -> None:
        """ONE `WorkerSnapshot`, digest-fenced. Dispatch stays CLOSED until the
        RecordOwner's `SnapshotAck` names exactly this `snapshot_id` AND this
        `snapshot_digest` — an ack naming a different one is NOT an ack (02 §6, §5)."""
        canonical_bytes, digest = documents.identity(self.snapshot_body())
        self.fence.snapshot_id = f"snp-{uuid.uuid4().hex[:16]}"
        self.fence.snapshot_digest = digest
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
            f"{documents.spell(digest)[:23]}…) reports {held} held attempt(s); the "
            "RecordOwner reconciles them durably before any dispatch",
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

    def _weights_refused(self, refused: pb.WeightsTransactionRefused) -> None:
        """The Host holds no custody for one transaction: stop it and fail only its attempt.

        The envelope was fenced by the caller. A refusal naming no transaction this worker
        opened, or one already refused, is a no-op; the control stream always continues.
        """
        owner = self.fence.record_owner_id
        transaction = refused.weights_transaction_id
        if self.workspace is None:
            return
        try:
            row = self.workspace.weights_row(owner, transaction)
        except WorkspaceRefusal:
            row = None
        if (
            row is None
            or row["request"] != refused.request_id
            or row["ordinal"] != refused.attempt_ordinal
            or row["epoch"] != refused.writer_epoch
        ):
            self.note("weights", f"ignored a custody refusal for unknown {transaction[:23]}")
            return
        self.weights.abandon(transaction)
        if not self.workspace.refuse_weights(owner, transaction):
            return
        detail = safe(
            f"output {refused.output_slot!r}: the Host refused "
            f"{pb.WeightsHostStage.Name(refused.stage)} "
            f"({pb.WeightsHostRefusal.Name(refused.refusal)}): {refused.safe_detail}"
        )
        size = len(row["declaration"])
        if (
            refused.stage == pb.WEIGHTS_HOST_STAGE_INTENT
            and refused.refusal == pb.WEIGHTS_HOST_REFUSAL_INTENT_CONFLICT
            and size > weights_limits.MAX_WEIGHTS_DECLARATION_BYTES
        ):
            # A Host that predates by-reference intents refuses every declaration over its
            # inline bound; only a newer worker image carries one that accepts it.
            detail = safe(
                f"output {refused.output_slot!r}: its weights declaration is {size} bytes, over "
                f"this pod Host's {weights_limits.MAX_WEIGHTS_DECLARATION_BYTES}-byte inline "
                "limit; the machine needs a worker image whose Host takes declarations by "
                "reference"
            )
        attempt = self.engine.history.get((refused.request_id, refused.attempt_ordinal))
        if attempt is not None and attempt.state not in ("outcome", "closed"):
            attempt.weights_refused = detail
        self.note("weights", f"{refused.request_id}#{refused.attempt_ordinal} {detail}")

    def _dispatch_current_frame(self, frame: pb.RecordOwnerFrame) -> None:
        kind = frame.WhichOneof("msg")
        if kind is None:
            return
        inner = getattr(frame, kind)
        if kind == "claim":
            self.note("claim", "PROTOCOL: a second Claim on an established stream; ignored")
            return
        if self.fenced(inner):
            return
        if kind == "weights_transaction_refused":
            self._weights_refused(inner)
            return
        if kind == "snapshot_ack":
            # THE BARRIER IS BOUND BY BOTH (§5). An ack naming a different `snapshot_id` OR
            # a different `snapshot_digest` is NOT an ack: it would open dispatch against a
            # reconciliation of some other snapshot, which is the one thing the barrier
            # exists to prevent.
            if (
                inner.snapshot_id != self.fence.snapshot_id
                or inner.snapshot_digest != self.fence.snapshot_digest
            ):
                self.note(
                    "fence",
                    f"SnapshotAck names {inner.snapshot_id!r}/"
                    f"{inner.snapshot_digest.hex()[:12]} and this snapshot is "
                    f"{self.fence.snapshot_id!r}/{self.fence.snapshot_digest.hex()[:12]}: "
                    "not an ack, the barrier stays closed",
                )
                return
            self.fence.dispatch_open = True
            self.connected.set()
            self.bump_admission(f"snapshot {inner.snapshot_id} acknowledged; dispatch opens")
            self.note("session", f"snapshot {inner.snapshot_id} acknowledged; dispatch is open")
            for attempt in self.engine.pending_outcomes():
                self.note(
                    "outcome",
                    f"replaying the recorded outcome for {attempt.request_id}#"
                    f"{attempt.attempt} in a fresh envelope ({attempt.outcome_id})",
                )
                self.send(pb.WorkerFrame(attempt_outcome=self.engine.wrap(attempt)))
            if self.workspace is not None:
                # Completed historical work may belong to an executor that no
                # longer exists. Never project another worker's live attempt.
                for value in self.workspace.projected_weights(
                    self.fence.record_owner_id, self.fence.worker_boot_id
                ):
                    row = value
                    if row["receipt"]:
                        self.send(
                            pb.WorkerFrame(
                                weights_receipt=pb.WeightsReceiptFrame(
                                    request_id=row["request"],
                                    attempt_ordinal=row["ordinal"],
                                    invocation_spec_digest=row["spec"],
                                    output_slot=row["slot"],
                                    weights_transaction_id=row["id"],
                                    writer_epoch=row["epoch"],
                                    tensorfs_declaration_digest=row["declaration_digest"],
                                    weights_receipt=pb.WeightsReceiptRef(
                                        weights_receipt_digest=row["receipt_digest"],
                                        weights_receipt_canonical_bytes=row["receipt"],
                                    ),
                                    manifest=pb.Ref(
                                        digest=row["manifest"], length=row["manifest_length"]
                                    ),
                                    objects=[
                                        pb.WeightsObjectSource(**obj)
                                        for obj in json.loads(row["objects"])
                                    ],
                                )
                            )
                        )
                    self.send(pb.WorkerFrame(weights_transaction=self.workspace.status(value)))
                for outcome in self.workspace.retained_outcomes(self.fence.record_owner_id):
                    self.send(pb.WorkerFrame(attempt_outcome=outcome))
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))
            return
        if kind == "desired_state":
            minor = inner.wire_minor or self.fence.wire_minor
            if minor < MIN_COMPATIBLE_WIRE_MINOR:
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FAULT_KIND_CONFIG_REFUSED,
                        subject=f"revision {inner.revision}",
                        reason=CAPABILITY_UNAVAILABLE_CODE,
                        detail=_capability_unavailable("package preparation", minor),
                    )
                )
                self.send(pb.WorkerFrame(observed_state=self.observed_state()))
                return
            if self.executions is not None and self.executions.active(self.fence.record_owner_id):
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FAULT_KIND_CONFIG_REFUSED,
                        subject=f"revision {inner.revision}",
                        reason="machine_execution_owns_worker",
                        detail="Accepted execution controls its prepared state; "
                        "submit through the execution API.",
                    )
                )
                self.send(pb.WorkerFrame(observed_state=self.observed_state()))
                return
            try:
                self._apply_desired_state(inner)
            except ExecutorGone as exc:
                self.note("desired_state", f"apply failed: {exc}")
                self.latch(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=f"revision {inner.revision}",
                        reason="desired_state_preparation_failed",
                        detail=safe(f"{type(exc).__name__}: {exc}"),
                    )
                )
            return
        if not self.fence.dispatch_open:
            self.note("fence", f"dropped a {kind} before SnapshotAck: dispatch is closed")
            return
        if (
            kind in ("attempt_offer", "cancel_attempt", "outcome_ack", "weights_finalize_request")
            and self.executions is not None
            and self.executions.owns(self.fence.record_owner_id, inner.request_id)
        ):
            self.note("execution", "machine-owned request requires the execution control API")
            return
        if kind == "attempt_offer":
            self.on_offer(inner)
        elif kind == "cancel_attempt":
            if self.source_calls is not None:
                self.source_calls.cancel_parent(
                    inner.request_id,
                    ordinal=inner.attempt_ordinal,
                    spec=inner.invocation_spec_digest,
                )
            self.note("cancel", self.cancel(inner))
        elif kind == "outcome_ack":
            self.on_outcome_ack(inner)
        elif kind == "checkpoint_receipt":
            self.engine.absorb_receipt(inner)
            ack = pb.JobCheckpointAck(
                request_id=inner.request_id,
                attempt_ordinal=inner.attempt_ordinal,
                operation_key=inner.operation_key,
                receipt_id=inner.receipt_id,
                content_digest=inner.content_digest,
            )
            self.send(pb.WorkerFrame(checkpoint_ack=ack))
        elif kind == "weights_host_ack":
            self.note("artifact", "retired Host writer acknowledgment ignored")
        elif kind == "native_source_command":
            from cozy_runtime.internal.worker.source_calls import refusal_frame

            if self.source_calls is None or self.fence.wire_minor < 42:
                self.send(refusal_frame(inner, "native_source_unavailable"))
            else:
                try:
                    self.source_calls.handle(inner)
                except Exception:
                    self.send(refusal_frame(inner, "native_source_refused"))
        elif kind == "native_artifact_transfer":
            if self.artifact_transfers is not None and self.fence.wire_minor >= 42:
                self.artifact_transfers.handle(inner)
        elif kind == "child_call_result":
            if not self.calls.absorb(inner):
                self.note("child", "dropped stale or invalid child-call result")
            elif inner.state == pb.CHILD_CALL_STATE_SUCCEEDED and self.workspace is not None:
                from . import workspace_memo

                try:
                    workspace_memo.delivered(
                        self.workspace, self.fence.record_owner_id, inner.child_request_id
                    )
                except Exception as exc:
                    # The owner already committed this result. Optional metadata
                    # compaction cannot turn its successful computation into a failure.
                    self.note("workspace", f"result compaction deferred ({type(exc).__name__})")
        elif kind == "weights_read_request":
            self.send(self.weights.read(inner))
        elif kind == "weights_upload_request":
            # The pre-21 upload lane. A supervisor on RuntimeWeights calls Upload unary
            # instead; this frame answer stays for one that has not opened it.
            self.send(self.weights.upload(inner))
        elif kind == "weights_finalize_request":
            from . import workspace_finalize

            try:
                if self.workspace is None:
                    raise ValueError("workspace custody is unavailable")
                result = workspace_finalize.finalize(
                    self.workspace,
                    self.fence.record_owner_id,
                    inner,
                    legacy_root=self.root,
                    on_abandon=self.weights.abandon,
                )
            except Exception as exc:
                self.note(
                    "artifact", f"weights finalization refused: {type(exc).__name__}: {exc}"[:512]
                )
            else:
                self.send(pb.WorkerFrame(weights_finalize_result=result))

    def on_outcome_ack(
        self, acked: pb.AttemptOutcomeAck, *, native_finalized: bool = False
    ) -> None:
        """A closed outcome leaves the lane's RELEASED set — a SLOT COUNT change when the
        device was waiting on it (cr-079 D3), not a change in admission MEANING, so it does
        NOT bump `admission_epoch` (§6).

        The distinction is load-bearing and it cost a live run to find. `admission_epoch`
        moves when the meaning of an offer changes — executor respawn, window resize, phase
        change, cutover — and a RecordOwner dispatching against a stale one is refused
        deterministically. Freeing a seat changes neither: bumping here made EVERY
        steady-state dispatch refuse ADMISSION_EPOCH_STALE, because the ack that freed the
        seat invalidated the epoch the RecordOwner was about to dispatch against.
        `available_attempt_slots` is the field that carries this, and it is recomputed on
        every tick.
        """
        if not native_finalized and self.source_calls is not None:

            def finished() -> None:
                with self.control_lock:
                    if not self.fenced(acked) or (
                        self.executions is not None
                        and self.executions.owns(self.fence.record_owner_id, acked.request_id)
                    ):
                        self.on_outcome_ack(acked, native_finalized=True)

            try:
                if self.source_calls.acknowledge(acked, finished):
                    return
            except ValueError:
                self.note("source", "native source finalization was refused")
                return
        if self.workspace is not None and self.workspace.contains(
            self.fence.record_owner_id, acked.request_id, acked.attempt_ordinal
        ):
            try:
                self.workspace.acknowledge(self.fence.record_owner_id, acked)
            except ValueError as exc:
                self.note("outcome", f"workspace acknowledgment refused: {exc}")
                return
        closed = self.engine.history.get((acked.request_id, acked.attempt_ordinal))
        job = closed.job if closed is not None else None
        disposition = self.engine.close(acked)
        self.note("outcome", disposition)
        if disposition != "closed":
            return
        self.calls.close(acked.request_id, acked.attempt_ordinal)
        from . import machine_slots

        machine_slots.release(self, acked.request_id, ordinal=acked.attempt_ordinal)
        if self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))
        # An acknowledged stopped attempt can leave retained work for another
        # attempt on this worker. The owner's per-outcome retention decision
        # overrides the job's ordinary one-shot process recycling policy.
        if (
            self.reclaim_on_terminal
            and not acked.retain_work
            and not self.engine.live
            # Managed composition remains under the owner's lifetime, including
            # independent native holds which Runtime does not own or enumerate.
            and not self.job_slots
            and closed is not None
            and closed.kind == "job"
            and job is not None
            and self.jobs.get(job.job_descriptor_id) is job
        ):
            self.note(
                "session",
                f"job attempt {acked.request_id}#{acked.attempt_ordinal} is acknowledged and "
                f"reclaim_on_terminal is set: exiting {RECYCLE_EXIT} (recycle, not a death)",
            )
            self.exit_code = RECYCLE_EXIT
            self.shutdown_from_stream()

    def request_stop(self) -> None:
        """Wake every lane and the sender so the worker can leave; the host is the caller's."""
        self.stop.set()
        self.outbound.put(None)
        for lane in self.lanes:
            lane.stage.put(None)
            lane.work.put(None)
            lane.post.put(None)

    def request_restart(self) -> None:
        """A local operator requests process replacement while retaining source work."""
        if not self.stop.is_set():
            self.restart_requested = True
            self.stop.set()

    def request_idle_restart(self) -> bool:
        """An operator update cannot interrupt work admitted by a detached client."""
        with self.control_lock:
            active_executions = self.executions.active_count() if self.executions is not None else 0
            active_attempts = sum(
                attempt.state in (*self.engine.PRE_RELEASE, "released")
                for attempt in self.engine.history.values()
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

    def _placement_from_entry(
        self, entry: Mapping[str, Any], placement_set_digest: bytes
    ) -> Placement:
        return Placement(
            placement_id=str(entry.get("placement_id", "")),
            placement_set_digest=placement_set_digest,
            package=dict(entry.get("package", {})),
            development=dict(entry.get("development", {})),
            installation_id=str(entry.get("installation_id", "")),
            package_interface=base64.b64decode(entry.get("package_interface", ""), validate=True),
            bindings_digest=_digest_field(entry.get("bindings_digest")),
            models=tuple(dict(row) for row in entry.get("models", [])),
            entrypoints=tuple(dict(row) for row in entry.get("entrypoints", [])),
            document=dict(entry),
            device_pin=self.accepted.device_pins.get(str(entry.get("placement_id", ""))),
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
                    pb.Fault(
                        kind=pb.FAULT_KIND_CONFIG_REFUSED,
                        subject=f"revision {desired.revision}",
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
                pb.Fault(
                    kind=pb.FAULT_KIND_CONFIG_REFUSED,
                    subject=f"revision {desired.revision}",
                    reason="job_set_busy",
                    detail="stop unpublished package attempts before replacing their job set",
                )
            )
            return
        placements: list[dict[str, Any]] = []
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
                    pb.Fault(
                        kind=kind,
                        subject=f"revision {desired.revision}",
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
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        subject=f"revision {desired.revision}",
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
        # DRAINING is machine lifecycle too: a worker under a draining posture is not a
        # placement that stopped serving, it is a machine on its way out.
        if self.phase != pb.WorkerPhase.WORKER_PHASE_FAILED:
            self.set_phase(
                pb.WorkerPhase.WORKER_PHASE_DRAINING
                if desired.posture == pb.Posture.POSTURE_DRAINING
                else pb.WorkerPhase.WORKER_PHASE_ONLINE,
                f"posture {pb.Posture.Name(desired.posture)}",
            )
        # Minor skew is additive compatibility, not an equality gate. Never report a
        # RecordOwner's future minor as if this worker had applied capabilities it does not
        # implement; an older owner's lower minor remains visible as the effective level.
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
        if self.lanes.wide and not lane_wire.lanes_on_wire(self.fence.wire_minor):
            # THE FENCE (cr-066 step 0, ruling R3). This worker drives one lane per device
            # of its envelope; a RecordOwner below minor 22 reads ONE `available_attempt_slots`
            # and no lane per placement: reading the sum over lanes it would offer seats a
            # placement cannot use, and reading one lane it would leave the paid cards idle —
            # silently, since nothing it reads says a card is idle. Refusing typed is the
            # "never advertise capacity it cannot use" rule applied to the silent-GPU-0
            # defect; jobs keep the whole envelope and are untouched. An owner at 22 reads
            # `lanes[]` and pins placements to devices (`DesiredPlacementSet.device_pins`),
            # and `LaneSet.assign` decides per placement.
            self.latched = "device_lanes_unsupported"
            self._published_latch = self.latched  # this fault IS the publication
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    subject=f"revision {desired.revision}",
                    reason=self.latched,
                    detail=(
                        f"this worker's envelope {self.options.devices!r} is "
                        f"{len(self.lanes.entries)} device lanes and the RecordOwner "
                        f"negotiated wire minor {self.fence.wire_minor}, which carries no "
                        "per-lane seats (proto-024 `lanes[]` is minor "
                        f"{lane_wire.LANES_WIRE_MINOR}); a serving set here would idle "
                        "every card but one, so it is refused (jobs still take the whole "
                        "envelope)"
                    ),
                )
            )
            self.note(
                "desired_state",
                f"revision {desired.revision} REFUSED: {self.latched} "
                f"({len(self.lanes.entries)} lanes over {self.options.devices!r}); the worker "
                "stays ONLINE",
            )
            self.bump_admission(f"revision {desired.revision} refused: {self.latched}")
            self._settle()
            return

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
                    if str(row.get("placement_id", "")) == previous.placement_id
                ),
                0,
            )
        entry = placements[primary_index]
        self.pending_hosted = [
            dict(row) for index, row in enumerate(placements) if index != primary_index
        ]
        placement_id = str(entry.get("placement_id", ""))
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
            self.converge_placement(revision)

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
            str(entry.get("placement_id", "")): self._placement_from_entry(
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
                f"{len(candidate.entrypoints)} binding(s) resolved, "
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
            and outgoing.package == candidate.package
            and outgoing.development == candidate.development
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
            while held():
                if self.stop.wait(0.02):
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
                        reply = executor.call({"cmd": "unload", "construction": key}, timeout=None)
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
                if self.connected.is_set():
                    self.send(pb.WorkerFrame(observed_state=self.observed_state()))

        threading.Thread(
            target=self._lane(f"unload:{placement_id}", run),
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
        for attempt in list(self.engine.history.values()):
            while attempt.placement_id == placement_id and attempt.state in self.engine.HELD:
                if superseded() or self.stop.wait(0.02):
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
            self.bump_admission(f"placement {placement_id!r} removed")
        return True

    def _reconcile_hosted(self, revision: int) -> None:
        """Converge the additional placements without disturbing the primary one."""

        if revision != self.accepted.accepted_desired_state_revision:
            return
        desired = {
            str(entry.get("placement_id", "")): self._placement_from_entry(
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
        if self.options.development_project is not None:
            expected_development = {
                "package": self.options.development_package,
                "release": self.options.development_release,
                "installation_id": self.options.development_installation_id,
            }
            if placement.development != expected_development or placement.package:
                self._materialization_failed(
                    placement,
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        subject=placement.placement_id,
                        reason="development_launch_mismatch",
                        detail="the development Placement does not equal this process's exact "
                        "package, release, and source digest",
                    ),
                    terminal=True,
                )
                return None
            if self.options.artifact_cache is None or self.options.tensorfs_root is None:
                self._materialization_failed(
                    placement,
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=placement.placement_id,
                        reason="artifact_cache_absent",
                        detail="source-checkout selection bytes require --artifact-cache and "
                        "--tensorfs-root",
                    ),
                    terminal=False,
                )
                return None
            try:
                from cozy_runtime.internal import local as development_local
                from cozy_runtime.internal import static_interface

                # A SOURCE reading (#713): the session supervises, the executor imports.
                staged = development_local.stage_development_placement(
                    self.config.cozy_home,
                    static_interface.build(self.options.development_project),
                    package=self.options.development_package,
                    release=self.options.development_release,
                    installation_id=self.options.development_installation_id,
                    artifact_cache=self.options.artifact_cache,
                    tensorfs_root=self.options.tensorfs_root,
                    project=self.options.development_project,
                    binding_overrides=self.options.development_binding_overrides,
                )
                return acquire.preinstalled_development(
                    placement,
                    self.options.artifact_cache,
                    staged.placement,
                    staged.declared_bindings,
                )
            except acquire.AcquisitionRefusal as exc:
                self._materialization_failed(
                    placement,
                    pb.Fault(
                        kind=exc.kind,
                        subject=placement.placement_id,
                        reason=exc.code,
                        detail=safe(exc.detail),
                    ),
                    terminal=True,
                )
                return None
            except Exception as exc:
                code, separator, detail = str(exc).partition(": ")
                if not code.startswith("development_"):
                    code = "development_source_invalid"
                    detail = str(exc)
                self._materialization_failed(
                    placement,
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        subject=placement.placement_id,
                        reason=code,
                        detail=safe(detail if separator else str(exc)),
                    ),
                    terminal=True,
                )
                return None
        from .machine_capture import installation_identity

        prepared_environment = next(
            (
                prepared.installed
                for prepared in self.prepared_installations.values()
                if prepared.placement_id == placement.placement_id
                and prepared.installation_id == installation_identity(placement.document)
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
                    package=(placement.package or placement.development).get("package", ""),
                    release=(placement.package or placement.development).get("release", ""),
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

            acquired = acquirer.acquire(placement)
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
        if (
            placement is self.placement
            or any(hosted.placement is placement for hosted in self.hosted.values())
        ) and self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))

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
            if self.fence.wire_minor < 40 or (job.orchestration and has_parent):
                raise ValueError("orchestration requires wire40 and one CPU parent")
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
                self.reclaim_on_terminal = (
                    parent.reclaim_on_terminal if parent is not None else True
                )
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
        supervision.on_exit = lambda executor: self.lanes.orchestration.work.put(
            ExecutorDeath(executor, "")
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
        self.reclaim_on_terminal = job.reclaim_on_terminal
        self.note(
            "desired_state",
            f"revision {self.accepted.accepted_desired_state_revision} accepted in JOB mode: "
            f"{self.jobs[job.job_descriptor_id].job!r}, device floor {job.device_count}, "
            f"reclaim_on_terminal={job.reclaim_on_terminal}",
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
            if (
                latch == "prepare refused: every binding"
                and self.options.development_project is not None
            ):
                # A source-checkout generation has one placement and no retained published
                # fallback. Creator's readiness owner treats FAILED materialization as the
                # terminal answer and replaces this generation; leaving it STAGED/OFFLINE
                # made a permanent prepare refusal look like an endlessly activating model.
                self.placement.materialization = (
                    pb.MaterializationState.MATERIALIZATION_STATE_FAILED
                )
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
        """`start` the process OFF the device lock: every rank spawned, joined, imported and
        warmed. Returns a latch, or "" once the process is started."""
        lane = tenant.lane
        placement_id = tenant.placement.placement_id
        first = next(iter(tenant.bindings.values()), None)
        if first is None:
            return "prepare refused: every binding"
        command = {
            "cmd": "start",
            "devices": lane.devices,
            "sequence_parallel_degree": _rank_degree(lane, tenant.bindings.values()),
            "application": first.application,
            "package_interface": first.interface_path,
            "development_project": first.development_project,
        }
        began = time.perf_counter()
        self.monitor.start(f"start:{placement_id}", "load")
        try:
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
            return "prepare refused: every binding"
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
            self._load_construction(tenant, executor, key, group)
        if executor.active != key:
            self._activate_construction(tenant, executor, key)

    def _current_tenant(self, tenant: Tenant) -> Tenant:
        """`tenant`'s slot as it stands now (the caller holds the control lock)."""
        if self._placement_by_id(tenant.placement.placement_id) is None:
            return tenant
        return self._tenant(tenant.placement.placement_id)

    def _process_ceiling(self, lane: DeviceLane, slot: str) -> int:
        """The PROCESS ceiling for a load or an activation: the lane's measured capacity
        less the most its co-tenants hold on any one of its devices. The executor fits under
        it less what it already holds, verified against what its own allocator can get."""
        measured = lane.measure(accel.host_backend_family())
        memory = fold_memory(measured)
        if memory.state != "measured":
            unreadable = [
                f"{lane.entries[index]!r}"
                for index, ordinal in enumerate(lane.ordinals)
                if measured[ordinal].state != "measured"
            ]
            raise LaneRefusal(
                "device_capacity_unreadable",
                f"{lane.lane_id}: the driver could not report free bytes for "
                f"{', '.join(unreadable)}, so no residence ceiling can be assigned (NVML "
                "nvmlDeviceGetMemoryInfo, else nvidia-smi --query-gpu=memory.free, is the "
                "reading); a lane with any unreadable device is unreadable",
            )
        row = lane.row(slot)
        row.ledger.observe_devices(lane.device_facts(measured))
        others = self.lanes.beside(lane, slot)
        row.ledger.authorized_device_limit_bytes = max(int(memory.total_bytes) - others, 0)
        return row.ledger.authorized_device_limit_bytes

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
        label = key.removeprefix("sha256:")[:12]
        if not first.weightless:
            # A CO-TENANT PROCESS on a shared single-device lane gives the card up first
            # (cr-022); a lane with one process has no victim and moves nothing.
            declared = sum(binding.model_weight_bytes for binding in group)
            lane.touch(slot)
            self._make_room(
                lane,
                slot,
                lambda memory: memory.free_bytes >= declared,
                f"load of {label} for {placement_id!r} declaring {declared} B",
            )
        try:
            authorized = 0 if first.weightless else self._process_ceiling(lane, slot)
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
            reply = executor.call(
                command,
                timeout=None,
                on_progress=lambda frame: self.monitor.advance(
                    subject, int(frame.get("position", 0) or 0)
                ),
            )
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
                if reply.get("poisoned", "unreported")
                else CAUSE.CAUSE_CODE_INVALID_REQUEST
                if code.startswith("attention_kernel_")
                else CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE,
            )
        self.monitor.end(subject, "filled")
        facts = dict(reply.get("facts") or {})
        try:
            resolved = PreparedModel.from_reply(reply)
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
        delivery = reply.get("delivery") or {}
        if delivery:
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
        older executor that does not report it is treated as poisoned.
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
        with self.control_lock:
            current = self._current_tenant(tenant)
            for member in group:
                # PER-BINDING (#572d), and it OUTLIVES the rebuild the refusal causes: the
                # process is replaced, but this construction is not tried again until the
                # desired state changes.
                current.failed_bindings[member.entrypoint_binding_digest] = f"{code}: {detail}"
                executor.ready_bindings.discard(member.entrypoint_binding_digest)
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
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
                else self._process_ceiling(tenant.lane, tenant.slot)
            )
        except LaneRefusal as exc:
            raise AttemptRefusal(
                exc.code, exc.detail, cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE
            ) from exc
        try:
            with executor.watched("activate"):
                reply = executor.call(
                    {
                        "cmd": "activate",
                        "construction": key,
                        "authorized_device_limit_bytes": authorized,
                    },
                    timeout=None,
                )
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
            target=self._lane(f"page-warm:{placement_id}", run),
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
        """Give THIS attempt's tenant the device its plan needs, before the plan is priced.

        cr-022's swap arm, the owner's "one gets to use it first, then the next gets to use
        it for a different request". The tenant's row is refreshed with the driver's free
        bytes as they stand now — the ledger's own number is the executor's last probe,
        and co-tenants move gigabytes between probes — and the chooser is asked whether any
        proven rung fits. If none does, co-tenants vacate LRU-first until one does (or
        nobody holds anything, in which case the chooser refuses or deepens exactly as it
        would alone). The lane seat is held by this attempt throughout, so no co-tenant
        is mid-attempt; the device lock is held by this lane thread, so no co-tenant's
        prepare is either. Both fitting means nothing moves, ever.
        """
        serving = attempt.spec.get("serving")
        if serving is None or self.accepted.mode != "serving":
            return
        declared = slot.bindings.get(str(serving.get("entrypoint_binding_digest", "")))
        if declared is None or declared.weightless:
            return
        lane = self.lanes.by_id[slot.lane_id]
        tenant = attempt.placement_id
        lane.touch(tenant)
        row = lane.row(tenant)
        why = f"attempt {attempt.request_id}#{attempt.attempt}"
        cell = attempt.prepared.cell() if attempt.prepared is not None else ""

        def fits(memory: accel.DeviceMemory) -> bool:
            row.ledger.observe_device_free(memory.free_bytes, memory.total_bytes)
            return bool(row.chooser.fits(declared, cell))

        # FIRST, TAKE BACK WHAT WAS TAKEN (cr-097). A tenant a co-tenant's prepare emptied
        # is priced, otherwise, against a hole nobody put back — which is how a placement
        # that had 10 GiB of room to itself spent seven consecutive attempts at +118%.
        # This evicts nothing, so it runs before the arbitration below rather than instead
        # of it: what it cannot restore, `_make_room` is still free to make room for.
        self._restore_residency(lane, tenant, row, why, declared, attempt.prepared)
        # THEN, THE RETRY RULE. The same invocation under the residency picture that already
        # produced its device OOM cannot get a different answer. Any other request can.
        self._break_oom_repeat(lane, tenant, row, attempt.digest, cell, why)
        self._make_room(lane, tenant, fits, why)

    def _restore_residency(
        self,
        lane: DeviceLane,
        slot: str,
        row: LaneRow,
        why: str,
        declared: DeclaredBinding | None = None,
        prepared: PreparedRequest | None = None,
    ) -> None:
        """RE-FILL a tenant the lane emptied, when the card has room. Evicts NOBODY (cr-097).

        `vacate` had no counterpart, and the gap was the defect: a tenant vacated for a
        co-tenant's prepare stayed empty for the life of its generation, re-admitting per
        scope on every attempt at a cost the device had room to stop paying. Measured live
        on an A4000: 8739 ms fully resident, 18909 ms after the vacate, 9056 ms once an
        unrelated rebuild happened to re-prepare it — with the co-tenant resident and
        untouched throughout, which is the proof the room was there all along.

        Unknown shapes and shapes that have already completed without moving weights
        retain prepare's measured-capacity restoration rule. An activation peak from a
        roomy device alone must not prevent their co-tenant recovery. A shape with only
        staged execution behind it is different: refilling all weights cannot help when
        its plan must immediately evict them again to make room for activations.
        """
        missing = row.ledger.missing_resident()
        if not missing:
            return
        executor = row.executor()
        if executor is None:
            return
        kind = accel.host_backend_family()
        memory = lane.memory(kind)
        if memory.state != "measured":
            return
        row.ledger.observe_device_free(memory.free_bytes, memory.total_bytes)
        if (
            declared is not None
            and prepared is not None
            and not row.chooser.should_restore(declared, prepared)
        ):
            return
        try:
            with executor.watched("restore"):
                reply = executor.call({"cmd": "restore", "names": sorted(missing)}, timeout=None)
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

    def _break_oom_repeat(
        self,
        lane: DeviceLane,
        slot: str,
        row: LaneRow,
        invocation: bytes,
        cell: str,
        why: str,
    ) -> None:
        """A RETRY MUST BE A DIFFERENT REQUEST, or it must refuse (cr-097).

        proto-037 fixed the mirror of this: a refusal is resumable only if the same request
        could get a different answer later. Here the answer COULD change, but only if
        something about the device changes first — and nothing did. Measured live: four
        attempts for one request, each 38.3 s of execution plus a 25.8 s executor rebuild,
        every one of them dying on the same allocator while an IDLE co-tenant held 6.07 GiB,
        until the requeue budget ran out. Roughly two and a half minutes of paid GPU to
        prove the same thing four times.

        So when THIS invocation already died on the device and the lane looks exactly as it
        did then, the co-tenants vacate — LRU-first, the ordinary arbitration, just with a
        demand the chooser's own `fits` is not allowed to talk it out of. If nothing can be
        freed, the picture cannot change and the attempt refuses TYPED and quantified
        instead of spending the handler to rediscover the allocator.

        The memory is per invocation, never per placement (cr-190): one oversized request
        is not evidence about a different, smaller one.
        """
        signature = row.oom_signatures.get(invocation, "")
        if not signature or signature != self.lanes.signature(lane):
            return
        self.note(
            "residency",
            f"{lane.lane_id}: {slot!r} already died on the device for this invocation under "
            f"this exact residency ({signature}); making room before {why} rather than "
            "repeating it",
        )
        # Never the chooser's `fits`: its verdict is what admitted the attempt that OOMed.
        self._make_room(lane, slot, lambda _memory: False, f"{why} after a device OOM")
        if signature != self.lanes.signature(lane):
            row.oom_signatures.pop(invocation, None)
            return
        kind = accel.host_backend_family()
        memory = lane.memory(kind)
        floor = row.ledger.carried_shortfall_by_cell.get(cell, 0)
        raise AttemptRefusal(
            "device_shortfall",
            f"this exact invocation already failed on the device with {floor} B free on "
            f"{lane.lane_id}, no co-tenant holds bytes that could be given back, and the "
            f"lane is unchanged ({signature}). Retrying would run the same request against "
            f"the same device and reach the same allocator; the driver reports "
            f"{memory.free_bytes} B free",
            cause=CAUSE.CAUSE_CODE_CONSTRAINT_INFEASIBLE,
            origin=ORIGIN.CAUSE_ORIGIN_RUNTIME,
        )

    def _make_room(
        self,
        lane: DeviceLane,
        slot: str,
        fits: Callable[[accel.DeviceMemory], bool],
        why: str,
    ) -> None:
        """RESIDENCY ARBITRATION on a lane's devices: vacate co-tenants of its TIGHTEST
        device, LRU-first, until `fits`.

        `fits` reads the lane as the driver measures it NOW (cr-066's NVML reading, the
        same meter ceilings are assigned from) and says whether `slot`'s demand holds. It is
        asked before anything moves and again after each vacate. Co-tenants come from every
        lane over that device (a group's ranks, other singletons), and only from the
        tightest one: freeing another device cannot raise the lane's MIN. A lane the driver
        cannot read is left alone: the prepare or the plan refuses typed on the same fact.
        A victim's process stays — only its residency changes, so its executor epoch and
        its placement's serving state do not move; its next attempt re-stages on demand.
        """
        kind = accel.host_backend_family()
        tried: set[str] = set()
        while True:
            measured = lane.measure(kind)
            memory = fold_memory(measured)
            if memory.state != "measured" or fits(memory):
                return
            tightest = min(measured, key=lambda ordinal: measured[ordinal].free_bytes)
            found = [v for v in self.lanes.victims(lane, slot, tightest) if v[0] not in tried]
            if not found:
                break
            victim, row = found[0]
            tried.add(victim)
            executor = row.executor()
            if executor is None:
                continue
            freed, took = self._vacate(victim, row, executor, why)
            self.note(
                "residency",
                f"{lane.lane_id}: {victim!r} (epoch {executor.epoch} pid {executor.pid}) "
                f"vacated {freed} B in {took:.0f} ms from device {tightest} for {slot!r} on "
                f"{why}; the driver reported {memory.free_bytes} B free before",
            )
        self.note(
            "residency",
            f"{lane.lane_id}: no co-tenant of {slot!r} holds bytes the driver could give "
            f"back for {why}; the plan decides on {memory.free_bytes} B free",
        )

    def _vacate(
        self, victim: str, row: LaneRow, executor: child.Executor, why: str
    ) -> tuple[int, float]:
        """Ask one co-tenant's executor to free every component it holds; bank the
        after-state on its row. Returns the bytes it reports freed and how long the free
        took (0 on refusal or death: a dead executor freed everything, and the
        re-measurement says so). HALF OF A SWAP IS AN EVICTION and the other half is a
        re-fill; both halves are timed now, because the cost of alternating two models on
        one card was being inferred from a construction's fill leg instead of measured
        (cr-100)."""
        try:
            with executor.watched("vacate"):
                reply = executor.call({"cmd": "vacate"}, timeout=None)
        except ExecutorGone as exc:
            row.ledger.device_unreadable(f"the executor died while vacating: {exc}")
            self.note("residency", f"{victim!r} died while vacating: {exc}"[:256])
            return 0, 0.0
        if not reply.get("ok"):
            self.note(
                "residency",
                f"{victim!r} refused to vacate: {reply.get('code')}: {reply.get('detail', '')}"[
                    :256
                ],
            )
            return 0, 0.0
        freed = row.ledger.observe_vacate(reply, why)
        row.ledger.device_process = self.read_device_process(executor.pid)
        held = (reply.get("vacated") or {}).get("held") or {}
        if held:
            self.note(
                "residency",
                f"{victim!r} kept {sorted(held)}: "
                + "; ".join(f"{name}: {reason}" for name, reason in sorted(held.items()))[:300],
            )
        return freed, float((reply.get("vacated") or {}).get("vacate_ms", 0.0))

    # ------------------------------------------------------------------ dispatch

    def on_offer(self, offered: pb.AttemptOffer, *, machine_owned: bool = False) -> None:
        """STREAM THREAD: fence, record, enqueue. Nothing here waits, fetches or leases.

        EVERY OFFER GETS A JOURNALED OUTCOME (§7). The outcome-less third state is closed:
        an `AttemptOffer` receives either `AttemptAccepted` or a JOURNALED
        `AttemptOutcome(REFUSED)` — never silence, never an unrecorded soft decline. An
        unrecorded refusal is unreconcilable after a crash, because the RecordOwner cannot
        then distinguish "never landed" from "refused", which is precisely the distinction
        02 §6 is built on.

        The rename is load-bearing: receiving an OFFER begins VALIDATION, not execution, so
        refusing one is an ordinary outcome rather than an exception. A pre-execution refusal
        consumes the `attempt_ordinal` and ZERO billed execution budget — `execution_started`
        stays false — and the RecordOwner may immediately mint `attempt_ordinal + 1` and
        dispatch it elsewhere.
        """
        # THE ADMISSION FENCE FIRST (§6), because it is the cheapest deterministic verdict
        # and because a stale epoch means the RecordOwner dispatched against a picture
        # of this worker that no longer holds.
        if self.stop.is_set():
            self.refuse(
                offered,
                "worker_stopping",
                "the worker is stopping and admits no new attempts",
                CAUSE.CAUSE_CODE_NO_CAPACITY,
            )
            return
        if self.fence.wire_minor < MIN_COMPATIBLE_WIRE_MINOR:
            self.refuse(
                offered,
                CAPABILITY_UNAVAILABLE_CODE,
                _capability_unavailable("execution", self.fence.wire_minor),
                CAUSE.CAUSE_CODE_CAPABILITY_UNAVAILABLE,
            )
            return
        if offered.admission_epoch != self.admission_epoch:
            self.refuse(
                offered,
                "admission_epoch_stale",
                f"the offer echoes admission epoch {offered.admission_epoch} and "
                f"this worker is at {self.admission_epoch}; the verdict is the same on "
                "every replay of these bytes, and the ordinal is consumed with no budget draw",
                CAUSE.CAUSE_CODE_ADMISSION_EPOCH_STALE,
            )
            return
        try:
            job_offer = machine_owned and "job" in documents.read(
                offered.invocation_spec_canonical_bytes, pb.InvocationSpec
            )
        except documents.DocumentError:
            job_offer = False  # The ordinary AttemptEngine records the malformed offer refusal.
        if self.accepted.mode == "serving" and not job_offer:
            selected = self._placement_by_id(offered.placement_id)
            if selected is None:
                hosted_ids = sorted(self.hosted)
                if self.placement is not None:
                    hosted_ids.insert(0, self.placement.placement_id)
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_UNKNOWN_PLACEMENT,
                        subject=offered.placement_id,
                        reason="unknown_placement",
                        detail=f"the offer names placement {offered.placement_id!r} and this "
                        f"worker hosts {hosted_ids!r}",
                    )
                )
                self.refuse(
                    offered,
                    "unknown_placement",
                    f"no placement {offered.placement_id!r} is hosted here",
                    CAUSE.CAUSE_CODE_UNKNOWN_PLACEMENT,
                )
                return
            if not self._placement_dispatchable(offered.placement_id):
                hosted = self.hosted.get(offered.placement_id)
                latched = hosted.latched if hosted is not None else self.latched
                self.refuse(
                    offered,
                    "placement_not_dispatchable",
                    f"placement {offered.placement_id!r} is "
                    + pb.ServingState.Name(selected.serving)
                    + (f" (LATCHED: {latched})" if latched else ""),
                    CAUSE.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE,
                )
                return
        if self.accepted.posture == pb.Posture.POSTURE_DRAINING:
            self.refuse(
                offered,
                "draining",
                "the worker is draining and admits no new attempts",
                CAUSE.CAUSE_CODE_NO_CAPACITY,
            )
            return
        try:
            attempt = self.engine.offer(offered)
        except AttemptRefusal as exc:
            self.note(
                "attempt", f"{offered.request_id}#{offered.attempt_ordinal} refused: {exc.code}"
            )
            self.refuse(offered, exc.code, exc.detail, exc.cause, exc.origin)
            return
        if "serving" in attempt.spec:
            hosted = self.hosted.get(offered.placement_id)
            bindings = hosted.bindings if hosted is not None else self.bindings
            declared = bindings.get(attempt.spec["serving"]["entrypoint_binding_digest"])
            if declared is not None and declared.internal:
                from . import internal_calls, machine_capture

                placement = self._placement_by_id(offered.placement_id)
                revision = (
                    machine_capture.installation_identity(placement.document) if placement else ""
                )
                try:
                    internal_calls.require_child(
                        self.workspace,
                        self.fence.record_owner_id,
                        offered,
                        revision,
                        declared.entrypoint,
                        machine_owned=machine_owned,
                    )
                except WorkspaceRefusal as exc:
                    self.refuse(
                        offered, "internal_callable", str(exc), CAUSE.CAUSE_CODE_INVALID_REQUEST
                    )
                    return
        if ("job" in attempt.spec or self.fence.wire_minor >= 43) and attempt.state not in (
            "outcome",
            "closed",
        ):
            result_schema, memoize = b"", False
            if "job" in attempt.spec:
                descriptor = (attempt.spec.get("job") or {}).get("job_descriptor_id", "")
                # Execution attaches attempt.job later. The prepared engine binding
                # is available on both local and placement-hosted paths at offer time.
                from . import machine_slots

                child_slot = self.job_slots.get(machine_slots.key(attempt.request_id))
                binding = (
                    child_slot.binding
                    if child_slot is not None
                    else self.engine.jobs.get(descriptor)
                )
                if (
                    binding is None
                    or binding.installation_id != attempt.spec["job"]["installation_id"]
                ):
                    self.refuse(
                        offered,
                        "job_binding_unavailable",
                        "the selected job is not prepared",
                        CAUSE.CAUSE_CODE_LOCAL_SAFETY,
                    )
                    return
                try:
                    declaration = self.engine._job_declaration(binding, descriptor)
                except AttemptRefusal as exc:
                    self.refuse(offered, exc.code, exc.detail, exc.cause, exc.origin)
                    return
                if declaration.get("internal", False):
                    from . import internal_calls

                    try:
                        internal_calls.require_child(
                            self.workspace,
                            self.fence.record_owner_id,
                            offered,
                            binding.installation_id,
                            binding.job,
                            machine_owned=machine_owned,
                        )
                    except WorkspaceRefusal as exc:
                        self.refuse(
                            offered, "internal_callable", str(exc), CAUSE.CAUSE_CODE_INVALID_REQUEST
                        )
                        return
                result_schema = canonical.write(declaration["result"])
                memoize = declaration.get("invocable", {}).get("memoize", False) is True
            replay = (
                self.workspace.accept(
                    self.fence.record_owner_id,
                    offered,
                    result_schema=result_schema,
                    worker_boot=self.fence.worker_boot_id,
                    memoize=memoize,
                )
                if self.workspace is not None
                else None
            )
            if replay is not None:
                self.send(pb.WorkerFrame(attempt_outcome=replay))
                return
        if attempt.state in ("outcome", "closed"):
            self.note(
                "attempt",
                f"duplicate AttemptOffer for {offered.request_id}#{offered.attempt_ordinal}: "
                f"replaying the recorded outcome, executing NOTHING "
                f"(executions={attempt.executions})",
            )
            self.send(pb.WorkerFrame(attempt_outcome=self.engine.wrap(attempt)))
            return
        if attempt.state == "queued":
            self.note(
                "attempt",
                "duplicate AttemptOffer for the attempt currently crossing to its lane's "
                "stage: it is held, and its acceptance publishes when the lane stages it",
            )
            return
        if attempt.state in (
            "staged",
            "entering",
            "accepted",
            "running",
            "reclaiming",
            "finalizing",
            "released",
        ):
            self.note(
                "attempt",
                "duplicate AttemptOffer for a held attempt: re-accepting "
                "the SAME acceptance; nothing starts a second time",
            )
            self.send(pb.WorkerFrame(attempt_accepted=self.engine.accepted_message(attempt)))
            return
        # THE SEAT IS THE PLACEMENT'S LANE'S, AND OVER IT THE ANSWER IS A JOURNALED
        # OUTCOME (#494f/§7): a refusal the RecordOwner can read and re-dispatch. An attempt
        # for placement P may take P's lane's seat and no other.
        lane = self._offer_lane(offered.placement_id, attempt)
        if self.admission_state(
            machine_owned=machine_owned
        ) != pb.AdmissionState.ADMISSION_STATE_OPEN or (
            self._lane_slots(lane, machine_owned=machine_owned) < 1
        ):
            busy = [
                held
                for held in self.engine.history.values()
                if held is not attempt
                and held.state in self.engine.HELD
                and held.lane_id == lane.lane_id
            ]
            other = busy[0] if busy else None
            self.note(
                "attempt",
                f"{offered.request_id}#{offered.attempt_ordinal} over {lane.lane_id}'s seat",
            )
            self.refuse(
                offered,
                "no_capacity",
                f"{lane.lane_id} admits {self._lane_capacity(lane)} attempt(s) to its "
                "queue and device at a time and "
                + (
                    f"holds {other.request_id}#{other.attempt} ({other.state})"
                    if other is not None
                    else "its seat is closed"
                )
                + "; the seat frees at device release, and the device waits while more "
                f"than {LANE_RELEASED} released attempt awaits AttemptOutcomeAck, so acking "
                "frees one",
                CAUSE.CAUSE_CODE_NO_CAPACITY,
            )
            return
        attempt.lane_id = lane.lane_id
        self.engine.hold(attempt)
        (lane.stage if self._stages(attempt) else lane.work).put(attempt)

    def _stages(self, attempt: AttemptRecord) -> bool:
        """A serving attempt is staged beside the device; a job admits on its lane."""
        return self.accepted.mode == "serving" and "job" not in attempt.spec

    def _offer_lane(self, placement_id: str, attempt: AttemptRecord | None = None) -> DeviceLane:
        """The lane an offer draws its seat from: the placement's, or the envelope for a job.
        Called after the placement is known to be hosted, so an unassigned placement here
        is one that is not dispatchable, and its lane is whichever its slot would take."""
        if self.accepted.mode == "job" or (attempt is not None and "job" in attempt.spec):
            if attempt is not None:
                from . import machine_slots

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
        deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
        self.note(
            "readiness",
            f"{key} reached the lane while the worker is rebuilding: waiting while the "
            "rebuild makes progress rather than refusing it",
        )
        why = "the attempt's own deadline expired while it queued"
        while not (settled := settled_gate.wait(REBUILD_SAMPLE_SECONDS)):
            if deadline_ms and time.time() * 1000 > deadline_ms:
                break
        waited = (time.perf_counter() - started) * 1000
        self.note(
            "readiness",
            f"{key} waited {waited:.0f} ms; the placement is "
            f"{pb.ServingState.Name(self.placement.serving) if self.placement else 'absent'}"
            + ("" if settled else f" ({why})"),
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
        was not.
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
        self.send(pb.WorkerFrame(attempt_outcome=outcome))

    # ------------------------------------------------------------------ the device lane

    def _sync_lane_consumers(self) -> None:
        """Each current queue has exactly one consumer; replaced empty lanes stop."""
        with self.control_lock:
            if not self._consumers_started or self.stop.is_set():
                return
            current = set(self.lanes)
            for lane in self._device_consumers.keys() - current:
                lane.stage.put(None)
                lane.work.put(None)
                lane.post.put(None)
                self._retired_consumers.extend(self._device_consumers.pop(lane))
            for lane in current - self._device_consumers.keys():
                consumers = tuple(
                    threading.Thread(
                        target=self._lane(name.removeprefix("device-"), target, lane),
                        daemon=True,
                        name=name,
                    )
                    for name, target in (
                        (f"stage-{lane.lane_id}", self.stage_lane),
                        (f"device-{lane.lane_id}", self.device_lane),
                        (f"post-{lane.lane_id}", self.post_lane),
                    )
                )
                self._device_consumers[lane] = consumers
                for consumer in consumers:
                    consumer.start()
            # Called under the control lock during assignment. Never wait here for a
            # retiring consumer that may itself need that lock to finish its last event.
            for consumer in self._retired_consumers:
                if consumer is not threading.current_thread():
                    consumer.join(timeout=0)
            self._retired_consumers = [t for t in self._retired_consumers if t.is_alive()]

    def device_lane(self, lane: DeviceLane) -> None:
        """ONE SERIALIZED PER-DEVICE STATE MACHINE: admit, execute, reconcile, rebuild.

        One thread per lane (cr-066). Lanes share nothing but the worker's control state:
        an attempt on lane B admits and runs while lane A rebuilds, and A's rebuild queues
        only A's attempts behind it.
        """
        while not self.stop.is_set():
            item = lane.work.get()
            if item is None:
                return
            if isinstance(item, ExecutorDeath):
                if lane.lane_id.startswith("cpu-"):
                    continue
                elif lane is self.lanes.orchestration:
                    self._rebuild_orchestration()
                else:
                    self.rebuild_exited(item.executor, item.placement_id, granted=item.granted)
                continue
            attempt = item
            # A job offer arrives `queued` and admits here; a serving attempt arrives
            # staged and is claimed for entry — unless a cancel or its deadline took it
            # off the queue first, and then its terminal is already sent.
            staged = attempt.state != "queued"
            if staged and not self.engine.claim_staged(attempt):
                continue
            self.await_settled(attempt)
            # THE DEVICE IS HELD from entry to DEVICE RELEASE (cr-022, cr-079): a
            # co-tenant's prepare on this lane waits here rather than vacating an executor
            # whose attempt is on the card. The post phase runs on `post_lane` and never
            # takes this lock: at release the attempt's bytes have left the executor.
            with lane.device:
                if self.enter(attempt) if staged else self.admit(attempt):
                    self.execute(attempt, lane)
            # Per-rank pid, GPU and interval (execution_evidence) prove where it ran.
            self.gpu.release(
                f"{attempt.request_id}#{attempt.attempt}", ranks=attempt.execution.get("ranks", [])
            )
            # Rebuild on REFUSALS too (#458): a poisoned generation refuses every attempt,
            # so a rebuild gated on admission could never run again.
            if lane.lane_id.startswith("cpu-"):
                continue
            elif lane is self.lanes.orchestration:
                self._rebuild_orchestration()
            else:
                self.rebuild_if_needed(attempt.placement_id)

    def stage_lane(self, lane: DeviceLane) -> None:
        """STAGING, one thread per lane (proto-061 G): accept the next serving offer —
        fetch, hydrate, journal, AttemptAccepted — while the device runs the one ahead of
        it, then queue it for device entry. It never takes `device` or touches an
        executor."""
        while not self.stop.is_set():
            attempt = lane.stage.get()
            if attempt is None:
                return
            self.await_settled(attempt)
            if self.stage(attempt):
                lane.work.put(attempt)

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
            accepted = self.engine.stage(attempt)
        except AttemptRefusal as exc:
            self.note("attempt", f"{key} refused: {exc.code}")
            self.terminate(
                attempt, self.engine.refuse_held, exc.code, exc.detail, exc.cause, exc.origin
            )
            return False
        self.send(pb.WorkerFrame(attempt_accepted=accepted))
        if attempt.canceling and self.engine.claim_staged(attempt):
            # cancelled while it hydrated: nothing reached the device
            self.terminate(attempt, self.engine.finish_cancel, attempt.canceling)
        if self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))
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
        terminated. Acceptance already passed the admission epoch; what can still stop it
        is a cancel, its own deadline, or its placement leaving this worker."""
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

    def cancel(self, message: pb.CancelAttempt) -> str:
        """Mark the cancel; a STAGED attempt has nothing on the device, so its CANCELED
        terminal is recorded now rather than when the device reaches it."""
        done = self.engine.cancel(message)
        held = self.engine.live.get(message.request_id)
        if (
            held is not None
            and held.attempt == message.attempt_ordinal
            and held.canceling
            and self.engine.claim_staged(held)
        ):
            self.terminate(held, self.engine.finish_cancel, held.canceling)
            if self.connected.is_set():
                self.send(pb.WorkerFrame(observed_state=self.observed_state()))
            return f"cancelled while staged ({held.canceling}); nothing reached the device"
        return done

    def post_lane(self, lane: DeviceLane) -> None:
        """THE POST PHASE, one thread per lane (cr-079 D4): encode, bound, write under the
        grant, journal the outcome, send it. It overlaps the next attempt's device phase on
        the same lane, and its failure is a typed WORKER outcome, never a poisoned executor.
        """
        while not self.stop.is_set():
            item = lane.post.get()
            if item is None:
                return
            self.finish(item)

    def admit(self, attempt: AttemptRecord) -> bool:
        """FETCH -> HYDRATE -> LOAD -> CHOOSE -> ACCEPT, or a typed terminal. Never a silence."""
        # Desired-state convergence and executor rebuilds take this same lock. Recheck the
        # offer's immutable routing facts under it, then hold it through the acceptance:
        # a queued offer cannot be accepted on the far side of a placement or
        # admission-epoch change merely because it passed `on_offer()` earlier. The
        # construction the attempt needs is loaded between the two holds, under the lane's
        # device lock alone, so a load never stalls the control plane.
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
                accepted = self.engine.prepare_and_accept(
                    attempt, before_plan=lambda: self._arbitrate_residency(attempt, slot)
                )
            except AttemptRefusal as exc:
                self.note("attempt", f"{key} refused: {exc.code}")
                self.terminate(
                    attempt, self.engine.refuse_held, exc.code, exc.detail, exc.cause, exc.origin
                )
                return False
        self.send(pb.WorkerFrame(attempt_accepted=accepted))
        return True

    def _admission_refused(self, attempt: AttemptRecord) -> bool:
        """The offer's routing facts, re-read under the control lock: TRUE once refused."""
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
        if attempt.admission_epoch != self.admission_epoch:
            self.note(
                "admission",
                f"refusing queued {key}: offered epoch "
                f"{attempt.admission_epoch}, current {self.admission_epoch}",
            )
            self.terminate(
                attempt,
                self.engine.refuse_held,
                "admission_epoch_stale",
                f"the queued offer echoes admission epoch "
                f"{attempt.admission_epoch} and this worker is now at "
                f"{self.admission_epoch}",
                CAUSE.CAUSE_CODE_ADMISSION_EPOCH_STALE,
            )
            return True
        if self.accepted.mode == "serving" and "job" not in attempt.spec:
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
        if self.accepted.mode != "serving" and not self.dispatchable():
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

    def execute(self, attempt: AttemptRecord, lane: DeviceLane) -> None:
        """RUN -> RELEASE. The ledger closes at release; the post thread takes it from there.

        A terminal built here is one the device phase itself produced (cancelled before
        dispatch, the executor died, the worker faulted) and is sent at once. A released
        attempt is handed to `post_lane`, and the seat it held is advertised free in the
        same breath so the owner can offer the next attempt while this one encodes.
        """
        try:
            released = self.engine.execute(attempt)
        except Exception as exc:
            self.note("attempt", f"execution faulted: {type(exc).__name__}: {exc}")
            try:
                released = self.engine.abandon(
                    attempt, f"the worker faulted while running this attempt: {exc}"
                )
            except Exception as inner:
                self.note("attempt", f"even ABANDONED could not be built: {inner}")
                self.fail_machine(f"{type(inner).__name__}")
                self.terminate(attempt, self.engine.last_resort, f"{type(inner).__name__}: {inner}")
                return
        if not isinstance(released, Released):
            self.send(pb.WorkerFrame(attempt_outcome=released))
            return
        self._mark_device_outcome(attempt, released, lane)
        lane.post.put(released)
        if self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))

    def _mark_device_outcome(
        self, attempt: AttemptRecord, released: Released, lane: DeviceLane
    ) -> None:
        """Remember WHETHER THE DEVICE KILLED THIS ATTEMPT, and under what picture (cr-097).

        Recorded on the lane row rather than on the ledger because the ledger restarts with
        the next generation and an OOM is precisely what replaces the generation: the fact
        would die in the rebuild it caused, which is the whole shape of this defect. The row
        outlives the rebuild — `lane.forget` runs only when the placement LEAVES the lane —
        so the next attempt can see what the last one hit.
        """
        outcome = released.reply.get("outcome") or {}
        row = lane.rows.get(attempt.placement_id)
        if row is None:
            return
        if str(outcome.get("terminal", "")) == "succeeded":
            # This invocation got through, so its earlier OOM is no longer a trap.
            row.oom_signatures.pop(attempt.digest, None)
            return
        if str(outcome.get("code", "")) in DEVICE_SHORTFALL_CODES:
            signature = self.lanes.signature(lane)
            # An entry under another picture describes a lane that no longer exists.
            row.oom_signatures = {
                invocation: seen
                for invocation, seen in row.oom_signatures.items()
                if seen == signature
            }
            row.oom_signatures[attempt.digest] = signature

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
        self.send(pb.WorkerFrame(attempt_outcome=outcome))

    def terminate(
        self, attempt: AttemptRecord, build: Callable[..., pb.AttemptOutcome], *args: Any
    ) -> None:
        """Build one outcome for a held attempt, record it, and send it."""
        why = ""
        try:
            self.send(pb.WorkerFrame(attempt_outcome=build(attempt, *args)))
            return
        except Exception as exc:
            why = f"{type(exc).__name__}: {exc}"
            self.note("attempt", f"the outcome could not be built ({why}); last resort")
        try:
            self.send(pb.WorkerFrame(attempt_outcome=self.engine.last_resort(attempt, why)))
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
            if not rebuild or self._defer_rebuild(placement_id):
                return
        if hosted is not None:
            self._prepare_hosted_executor(hosted, fresh=False)
        elif current is None or not current.alive():
            self._rebuild("the executor is gone")
        else:
            self._rebuild(f"poisoned: {current.poisoned}")

    def rebuild_exited(
        self, executor: child.Executor, placement_id: str = "", *, granted: bool = False
    ) -> None:
        """Rebuild only if the queued death still names the owned dead epoch."""

        with self.control_lock:
            hosted = self.hosted.get(placement_id)
            supervision = hosted.supervision if hosted is not None else self.supervision
            if supervision.current is not executor or (executor.alive() and not executor.poisoned):
                return
            primary = self.placement.placement_id if self.placement is not None else ""
            if not granted and self._defer_rebuild(placement_id or primary):
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

    def _expire_staged(self, attempt: AttemptRecord) -> None:
        """A staged attempt whose deadline passed while the device ran the one ahead of it
        ends CANCELED(deadline) now, not when the device reaches it."""
        deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
        if not deadline_ms or time.time() * 1000 <= deadline_ms:
            return
        if not self.engine.claim_staged(attempt):
            return
        self.note("cancel", f"deadline expired for staged {attempt.request_id}")
        self.terminate(attempt, self.engine.finish_cancel, "deadline")

    def watchdog(self) -> None:
        """The deadline and cancel escalation — both the WORKER's, never author code's.

        A cancelled attempt stops at its executor's next safe point and the executor stays
        warm. It is killed only when its frames stop against their own measured cadence
        (`AttemptRecord.cancel_stall`), never after a flat grace.
        """
        while not self.stop.wait(0.05):
            self.drive_executions()
            for attempt in list(self.engine.live.values()):
                if attempt.state == "staged":
                    self._expire_staged(attempt)
                    continue
                if attempt.state not in ("running", "reclaiming"):
                    continue
                deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
                deadline_expired = False
                with attempt.transition_lock:
                    if (
                        attempt.state == "running"
                        and deadline_ms
                        and time.time() * 1000 > deadline_ms
                        and not attempt.canceling
                    ):
                        attempt.canceling = "deadline"
                        executor = attempt.executor
                        if executor is not None:
                            self.engine.supervision_for(attempt).cooperative_cancel(
                                executor, f"{attempt.request_id}#{attempt.attempt}"
                            )
                        attempt.cancel_started = time.monotonic()
                        deadline_expired = True
                if deadline_expired:
                    self.note("cancel", f"deadline expired for {attempt.request_id}")
                if attempt.canceling:
                    attempt.cancel_started = attempt.cancel_started or time.monotonic()
                    stall = attempt.cancel_stall(liveness.noise_floor())
                    if stall is not None:
                        self.note(
                            "cancel",
                            f"no measured progress for {stall['still_s']:g} s after the "
                            f"cancel (longest earlier gap {stall['worst_gap_s']:g} s over "
                            f"{stall['frames']} frame(s); patience {stall['patience_s']:g} s): "
                            "escalating to a forceful kill",
                        )
                        try:
                            facts = self.engine.escalate(attempt, stall)
                        except Exception as exc:
                            # A failed reclaim is NOT a cancellation result. Keep the
                            # executor owned, withdraw machine admission, and leave this
                            # watchdog as the retrying owner. A one-shot exception here used
                            # to kill the watchdog thread and strand the executor forever.
                            self.note(
                                "cancel",
                                f"forceful reclaim failed; retrying after the same window: "
                                f"{type(exc).__name__}: {exc}",
                            )
                            if attempt.placement_id in self.hosted:
                                self._hosted_capacity_dropped(
                                    attempt.placement_id, "executor reclaim is not proved"
                                )
                            else:
                                self.capacity_dropped("executor reclaim is not proved")
                            attempt.cancel_started = time.monotonic()
                            continue
                        self.note("cancel", f"forceful: {json.dumps(facts)}")
                        attempt.canceling = attempt.canceling or "client"
                        attempt.cancel_started = math.inf

    def reporter(self) -> None:
        """The ObservedWorkerState tick, and the liveness SAMPLE that rides it."""
        while not self.stop.wait(REPORT_SECONDS):
            for subject in list(self.monitor.cursors):
                self.monitor.sample(subject)
            self.publish_activity()
            # Observation follows authentication, independently of the dispatch
            # barrier. Only the current stream is eligible, after its snapshot.
            if self.fence.control_stream_epoch in self.attached_streams:
                self.send(pb.WorkerFrame(observed_state=self.observed_state()))

    def publish_activity(self, *, strict: bool = False) -> None:
        """Tell the pod supervisor what this worker is doing (`worker/activity.py`).

        It rides the reporter for one reason: a tick that does not happen is the evidence
        the supervisor needs. A failure to write is NOTED and never raised — the reporter
        lane dying would silence the file permanently, and a silent file reads as a wedge.
        """
        if self.activity_file is None:
            return
        try:
            self.activity_file.publish(
                owner_attached=bool(self.attached_streams),
                attempts_in_flight=self.engine.in_flight(),
                active_work=activity.active_work(self),
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
            available_attempt_slots=self.available_attempt_slots(),
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
            str(entrypoint["entrypoint_binding_digest"]) for entrypoint in placement.entrypoints
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
        until a deployment executor qualifies it."""
        facts = hostfacts.measure(self.options.accelerator_backend)
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
            self._consumers_started = True
            self._sync_lane_consumers()
            threading.Thread(
                target=self._lane("convergence", self.convergence_lane),
                daemon=True,
                name="convergence-lane",
            ).start()
            threading.Thread(
                target=self._lane("report", self.reporter), daemon=True, name="report"
            ).start()
            threading.Thread(
                target=self._lane("watchdog", self.watchdog), daemon=True, name="watchdog"
            ).start()
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
        if self.artifact_transfers is not None:
            self.artifact_transfers.close()
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
        for supervision in supervisions:
            try:
                supervision.close()
            except ExecutorGone as exc:
                self.exit_code = int(Exit.internal)
                self.note("shutdown", f"executor reclaim failed: {exc}")
        for consumer in [
            *self._retired_consumers,
            *(thread for pair in self._device_consumers.values() for thread in pair),
        ]:
            consumer.join()
        self._retired_consumers.clear()
        self._device_consumers.clear()
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


#: The loopback names a worker may serve plaintext, and without a bootstrap credential.
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def serve_worker(config: RuntimeConfig, argv: list[str] | None = None) -> int:
    """Parse the launch flags and run ONE worker hosting the protocol.

    Not an entrypoint: the config arrives from whoever already read it (one config per
    process, cl-004). `--listen` is the local binding this worker SERVES on — the owner
    dials it; the worker never dials anyone (#436).
    """
    parser = argparse.ArgumentParser(description="cozy-runtime worker (no CUDA)")
    parser.add_argument(
        "--listen", required=True, help="local binding, e.g. unix:/path/control.sock or 127.0.0.1:0"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--worker-id", default="wrk-local")
    parser.add_argument("--release-id", default="")
    parser.add_argument("--git-commit", default="")
    parser.add_argument("--python", default="")
    parser.add_argument("--accelerator-backend", choices=("", "cuda", "none"), default="")
    parser.add_argument("--devices", default="0")
    parser.add_argument(
        "--grant-root",
        default="",
        help="comma-separated directory trees a file:// grant may address (#506b)",
    )
    parser.add_argument(
        "--install-root",
        default="",
        help="immutable install root containing exact receipts and content-keyed generations",
    )
    parser.add_argument(
        "--environment-python",
        default="",
        help="installed local package Python; with --install-root, supplies dependencies "
        "for a captured immutable generation",
    )
    parser.add_argument(
        "--artifact-cache",
        default="",
        help="where the dynamic acquisition lane writes the exact bytes a placement reaches "
        "(cr-049); content-addressed, so a replay re-fetches only what is missing",
    )
    parser.add_argument(
        "--tensorfs-root",
        required=True,
        help="verified TensorFS store populated from selected model Manifests",
    )
    parser.add_argument("--development-project", default="", help=argparse.SUPPRESS)
    parser.add_argument("--development-package", default="", help=argparse.SUPPRESS)
    parser.add_argument("--development-release", default="", help=argparse.SUPPRESS)
    parser.add_argument("--development-installation-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--alloc-conf", default="expandable_segments:True")
    parser.add_argument("--threads", type=int, default=4)
    # Ignored: cancel escalation is measured, not a grace. Deployed launchers still pass it.
    parser.add_argument("--grace", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--executor-uid", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--executor-gid", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--tls-cert", default="", help="PEM cert for a non-loopback listener")
    parser.add_argument("--tls-key", default="", help="PEM key for a non-loopback listener")
    args = parser.parse_args(argv)

    isolated = args.executor_uid >= 0 or args.executor_gid >= 0
    if isolated and (
        args.executor_uid <= 0
        or args.executor_gid <= 0
        or os.geteuid() != 0
        or args.executor_uid == os.geteuid()
    ):
        print(
            "[worker] refusing executor identity split: --executor-uid/--executor-gid must "
            "name a positive non-worker identity and the trusted worker must run as root",
            file=sys.stderr,
            flush=True,
        )
        return int(Exit.structural)
    development_values = (
        args.development_project,
        args.development_package,
        args.development_release,
        args.development_installation_id,
    )
    if any(development_values) and not all(development_values):
        print(
            "[worker] development project, package, release, and source digest come together",
            file=sys.stderr,
            flush=True,
        )
        return int(Exit.structural)
    if all(development_values) and (
        args.development_package.count("/") != 1
        or not all(args.development_package.split("/"))
        or SEMVER.fullmatch(args.development_release) is None
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", args.development_installation_id)
        is None
    ):
        print(
            "[worker] development package identity, semantic release, or source digest is invalid",
            file=sys.stderr,
            flush=True,
        )
        return int(Exit.structural)
    options = WorkerOptions(
        root=Path(args.root),
        worker_id=args.worker_id,
        instance_id=f"ins-{uuid.uuid4().hex[:16]}",
        release_id=args.release_id,
        git_commit=args.git_commit,
        python=args.python,
        accelerator_backend=args.accelerator_backend,
        devices=args.devices,
        grant_roots=tuple(r for r in args.grant_root.split(",") if r),
        install_root=Path(args.install_root).resolve() if args.install_root else None,
        environment_python=(
            Path(args.environment_python).expanduser().parent.resolve()
            / Path(args.environment_python).name
            if args.environment_python
            else None
        ),
        tensorfs_root=Path(args.tensorfs_root).resolve(),
        artifact_cache=Path(args.artifact_cache).resolve() if args.artifact_cache else None,
        development_project=(
            Path(args.development_project).resolve() if args.development_project else None
        ),
        development_package=args.development_package,
        development_release=args.development_release,
        development_installation_id=args.development_installation_id,
        alloc_conf=args.alloc_conf,
        threads=args.threads,
        executor_uid=args.executor_uid,
        executor_gid=args.executor_gid,
    )
    listen = args.listen
    if listen.startswith("unix:"):
        refusal = socket_path_refusal(listen[len("unix:") :])
        if refusal:
            print(f"[worker] {refusal}", file=sys.stderr, flush=True)
            return int(Exit.structural)
    tls_cert = Path(args.tls_cert) if args.tls_cert else None
    tls_key = Path(args.tls_key) if args.tls_key else None
    if (tls_cert is None) != (tls_key is None):
        print("[worker] --tls-cert and --tls-key come together or not at all", file=sys.stderr)
        return int(Exit.structural)
    host_part = listen.rsplit(":", 1)[0]
    remote = not listen.startswith("unix:") and host_part not in LOOPBACK
    if remote and tls_cert is None:
        # A worker never listens plaintext off-loopback: the remote leg is TLS + the
        # bearer owner token, or it does not exist (#445; cr-020 item 4 keeps loopback).
        print(
            f"[worker] refusing to bind {listen} without TLS: a non-loopback listener "
            "requires --tls-cert/--tls-key",
            file=sys.stderr,
            flush=True,
        )
        return int(Exit.structural)
    if remote and not config.bootstrap_credential and not config.record_owner_public_key:
        print(
            f"[worker] refusing to bind {listen} without RecordOwner authority: use a "
            "launcher bootstrap credential or the fixed product-host signed Claim lane",
            file=sys.stderr,
            flush=True,
        )
        return int(Exit.structural)
    host = GrpcControlHost(
        listen, Path(args.root) / "control.addr", tls_cert=tls_cert, tls_key=tls_key
    )
    worker = Worker(config, options, host)
    if _loopback_preparation_listener(listen):
        host.preparer = worker.prepare_package_set
        host.local_package_preparer = worker.prepare_local_package
        host.unpublished_placement_preparer = worker.prepare_unpublished_placement
        host.numerical_environment = worker.numerical_environment
        host.workspace_service = worker.workspace_service

    def terminate(signum: int, frame: object) -> None:
        worker.request_stop()
        host.stop()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    return worker.run()


def main(argv: list[str] | None = None) -> int:
    """The PROCESS entrypoint: read the ONE config here, then serve.

    A configuration refusal is the worker's own verdict, printed the way every other
    pre-bind refusal is: a launcher reading this process's output must see WHY nothing
    bound, and a traceback is not that.
    """
    try:
        config = read_config()
    except ConfigError as err:
        print(f"[worker] {err.message}", file=sys.stderr, flush=True)
        print(f"[worker] remedy: {err.remedy}", file=sys.stderr, flush=True)
        return int(err.code)
    return serve_worker(config, argv)


if __name__ == "__main__":
    raise SystemExit(main())
