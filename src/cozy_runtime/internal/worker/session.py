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
lane; the REPORTER ticks the durable Report; the WATCHDOG owns the deadline and the grace
clock, because the deadline is worker-enforced and never merely handed to author code.

This process never imports torch. `scripts/worker-live.py nocuda` proves that against the
RUNNING process's loaded modules and mapped libraries, not against the import graph.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import hmac
import itertools
import json
import os
import queue
import re
import resource
import signal
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import (
    accel,
    canonical,
    child_env,
    fill,
    hostfacts,
    package_environment,
    proctree,
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
    model_source_prepare,
    observe,
    page_warm,
    triage,
    weights,
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
    GRACE_SECONDS,
    INTERFACE_DOCUMENT,
    ExecutorGone,
    ExecutorSupervision,
    socket_path_refusal,
)
from cozy_runtime.internal.worker.control import (
    ControlHost,
    GrpcControlHost,
    WatchFanout,
    _loopback_preparation_listener,
)
from cozy_runtime.internal.worker.lanes import (
    LANE_RELEASED,
    LANE_SEATS,
    DeviceLane,
    ExecutorDeath,
    LaneRefusal,
    LaneRow,
    LaneSet,
    fold_memory,
)
from cozy_runtime.internal.worker.ledger import Ledger, read_vmrss
from cozy_runtime.internal.worker.plan import DeclaredBinding, JobBinding, PreparedModel
from cozy_runtime.internal.worker.records import WorkerRecords
from cozy_runtime.protocol import MIN_COMPATIBLE_WIRE_MINOR, WIRE_MINOR, documents
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
#: `lanes.LANE_SEATS` PER LANE, summed over the lanes with a dispatchable placement — ONE on
#: the one-device worker every deployment produces today, byte-identical on the wire.
#:
#: It was 2 per worker, on the claim that attempt N+1's CPU phase overlaps attempt N's device
#: phase. That overlap does not exist: a lane is a single thread that admits, executes and
#: reconciles one attempt at a time, so a second admitted attempt could only WAIT — and it
#: waited in a state (`queued`) the wire cannot spell, invisible to the snapshot and to the
#: held-attempt set. A worker holding work the RecordOwner cannot see breaks the
#: snapshot-is-complete law, so the false state is DELETED rather than surfaced: an offer
#: arriving while its placement's lane already holds one is a JOURNALED REFUSED outcome the
#: RecordOwner reads, never a silent hold.
#:
#: rev-2 makes the SHAPE of the bound structural rather than advisory (§6): per-placement
#: `attempt_credits` are gone — N counters over ONE serialized device advertised N x the real
#: capacity. The principle is one counter per SERIALIZED RESOURCE, and with cr-066 that
#: resource is the device LANE: one seat per lane, consumed from the offer's hold to DEVICE
#: RELEASE (cr-079 D3: the executor's reply; the encode is the post thread's). A released
#: attempt holds no seat, but the post phase is bounded one-per-lane: while more than
#: `LANE_RELEASED` released attempts on a lane await `AttemptOutcomeAck`, the lane advertises
#: nothing, so a RecordOwner that stops acking idles its own device instead of growing this
#: worker's held set. The dual defect — one count of N
#: over N lanes, where an offer can use only its placement's — is why a multi-lane worker
#: REFUSES to serve a RecordOwner whose negotiated minor cannot read `lanes[]`
#: (`device_lanes_unsupported`, proto-024 minor 22).


def _progress_document(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the measured progress coordinates the executor already supplied.

    Stage-local and overall fractions stay separate. Optional coordinates are omitted when
    the package does not know them; no layer fabricates whole-invocation completion.
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
        for field in ("stage_fraction", "position", "total", "overall_fraction"):
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
    return (
        f"{delivery.get('variant', '')!r} / {delivery.get('label', 'unlabelled')} on rung "
        f"{delivery.get('rung', 'unreported')} "
        f"({delivery.get('delivery', 'unreported')}/{materialization}; "
        f"routes {census('routes')}; kernels {census('kernels')}; {calibrated}) chosen on "
        f"the {delivery.get('objective', 'unreported')} axis"
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
    development_source_digest: str = ""
    """Exact launch facts the development Placement must echo."""
    development_binding_overrides: Mapping[str, str] = field(default_factory=dict)
    """One-shot local resolver inputs; long-lived Creator launches use package.toml."""
    worker_tls_certificate_digest: str = ""
    alloc_conf: str = "expandable_segments:True"
    threads: int = 4
    grace: float = GRACE_SECONDS
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
    environment_digest: bytes
    package_interface: tuple[str, int]
    bindings_digest: bytes
    models: tuple[dict[str, Any], ...]
    entrypoints: tuple[dict[str, Any], ...]
    environment: dict[str, Any]
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
    #: The environment THIS boot materialized when it prepared a local revision. The
    #: wheels were the pod host's carriers and are gone once the PlacementSet is journaled,
    #: so the placement over that revision serves from this environment instead of
    #: re-acquiring bytes nothing on the pod still holds.
    installed: package_environment.InstalledEnvironment | None = None


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
    prepared_models: dict[str, PreparedModel] = field(default_factory=dict)
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
    prepared_models: dict[str, PreparedModel]
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


def desired_state_body(desired: pb.DesiredWorkerState) -> bytes:
    """The canonical bytes of a DesiredWorkerState's DESIRED STATE — envelope excluded (#494a).

    A DesiredWorkerState carries two unrelated things: what the RecordOwner wants this worker
    to be (revision, posture, wire minor, drain grace, the mode's set or job), and the
    per-stream envelope that routes it (record_owner_epoch, control_stream_epoch,
    worker_boot_id). Only the first is the body §3's same-revision rule is about.

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
    computes. Only on a match is the document parsed, under unknown-field REFUSAL — at every
    depth. There is no additive versioning inside a digest-fenced document.

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
SEMVER = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def _closed_document(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value).issubset(fields):
        raise documents.DocumentError("unknown_field", f"{name} has a non-closed field set")
    return value


def _required(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    missing = fields - set(value)
    if missing:
        raise documents.DocumentError("missing_field", f"{name} lacks {sorted(missing)[0]}")


def _exact_ref(value: object, name: str) -> dict[str, Any]:
    row = _closed_document(value, {"digest", "length"}, name)
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


def _sorted_unique_strings(value: object, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        or value != sorted(set(value))
    ):
        raise documents.DocumentError("sort_order", f"{name} is not sorted unique")
    return value


def _wheel_fact(value: object, name: str) -> dict[str, Any]:
    row = _closed_document(
        value, {"ref", "distribution", "version", "filename", "import_roots", "tags"}, name
    )
    _required(row, {"ref", "distribution", "version", "filename"}, name)
    _exact_ref(row["ref"], f"{name}.ref")
    distribution = _identifier(row["distribution"], f"{name}.distribution")
    if distribution != distribution.lower() or any(
        not part or not part.isalnum() for part in distribution.split("-")
    ):
        raise documents.DocumentError("wheel_fact_invalid", f"{name}.distribution")
    _identifier(row["version"], f"{name}.version")
    _identifier(row["filename"], f"{name}.filename")
    _sorted_unique_strings(row.get("import_roots", []), f"{name}.import_roots")
    _sorted_unique_strings(row.get("tags", []), f"{name}.tags")
    return row


def _package_selection(value: object) -> dict[str, Any]:
    row = _closed_document(value, {"package", "release"}, "PackageSelection")
    _required(row, {"package", "release"}, "PackageSelection")
    package = _identifier(row["package"], "PackageSelection.package")
    release = _identifier(row["release"], "PackageSelection.release")
    if package.count("/") != 1 or not all(package.split("/")) or SEMVER.fullmatch(release) is None:
        raise documents.DocumentError("package_invalid", "package identity or release is invalid")
    return row


def _development_package(value: object) -> dict[str, Any]:
    row = _closed_document(
        value,
        {
            "package",
            "release",
            "source_digest",
            "project_wheel",
            "local_revision_digest",
        },
        "DevelopmentPackage",
    )
    _required(row, {"package", "release", "source_digest"}, "DevelopmentPackage")
    package = _identifier(row["package"], "DevelopmentPackage.package")
    release = _identifier(row["release"], "DevelopmentPackage.release")
    if package.count("/") != 1 or not all(package.split("/")) or SEMVER.fullmatch(release) is None:
        raise documents.DocumentError(
            "development_invalid", "package identity or release is invalid"
        )
    if (
        not isinstance(row["source_digest"], str)
        or HASH_ENTRY.fullmatch(row["source_digest"]) is None
    ):
        raise documents.DocumentError("development_invalid", "source_digest is malformed")
    if "project_wheel" in row:
        project = _wheel_fact(row["project_wheel"], "DevelopmentPackage.project_wheel")
        if project["distribution"] != package.split("/", 1)[1] or project["version"] != release:
            raise documents.DocumentError("project_wheel_mismatch", package)
    remote = "project_wheel" in row
    local_revision = "local_revision_digest" in row
    if remote != local_revision:
        raise documents.DocumentError(
            "development_invalid",
            "project_wheel and local_revision_digest must be present together",
        )
    if local_revision and (
        not isinstance(row["local_revision_digest"], str)
        or HASH_ENTRY.fullmatch(row["local_revision_digest"]) is None
    ):
        raise documents.DocumentError("development_invalid", "local_revision_digest is malformed")
    return row


def _placement_profile(entry: dict[str, Any]) -> None:
    fields = {
        "placement_id",
        "package",
        "development",
        "environment_digest",
        "package_interface",
        "bindings_digest",
        "models",
        "entrypoints",
        "environment",
    }
    _closed_document(entry, fields, "Placement")
    _required(
        entry,
        {"placement_id", "package_interface", "bindings_digest", "entrypoints"},
        "Placement",
    )
    _identifier(entry["placement_id"], "Placement.placement_id")
    _exact_ref(entry["package_interface"], "Placement.package_interface")
    if (
        not isinstance(entry["bindings_digest"], str)
        or HASH_ENTRY.fullmatch(entry["bindings_digest"]) is None
    ):
        raise documents.DocumentError("malformed_digest", "Placement.bindings_digest")
    published = "package" in entry
    development = "development" in entry
    if published == development:
        raise documents.DocumentError("package_mode_invalid", "select exactly one package mode")
    package = (
        _package_selection(entry["package"])
        if published
        else _development_package(entry["development"])
    )
    remote_development = development and "project_wheel" in package
    if published or remote_development:
        _required(
            entry,
            {"environment_digest", "environment"},
            "Placement",
        )
        if (
            not isinstance(entry["environment_digest"], str)
            or HASH_ENTRY.fullmatch(entry["environment_digest"]) is None
        ):
            raise documents.DocumentError("malformed_digest", "Placement.environment_digest")
        environment = _closed_document(
            entry["environment"], {"local_wheels", "locked_requirements"}, "Environment"
        )
        wheels = environment.get("local_wheels", [])
        if not isinstance(wheels, list):
            raise documents.DocumentError("wrong_type", "Environment.local_wheels")
        wheel_rows = [
            _wheel_fact(row, f"Environment.local_wheels[{index}]")
            for index, row in enumerate(wheels)
        ]
        names = [str(row["distribution"]) for row in wheel_rows]
        if names != sorted(set(names)):
            raise documents.DocumentError("sort_order", "Environment.local_wheels")
        if published:
            # A published install is identified by the exact export it materialized from;
            # supplied wheels belong to editable revisions only.
            if wheels:
                raise documents.DocumentError(
                    "environment_invalid", "published Environment carries local_wheels"
                )
            _exact_ref(environment.get("locked_requirements"), "Environment.locked_requirements")
        else:
            if "locked_requirements" in environment:
                raise documents.DocumentError(
                    "environment_invalid", "development Environment carries a lock ref"
                )
            if package["project_wheel"]["distribution"] in names:
                raise documents.DocumentError(
                    "project_wheel_mismatch", "project wheel is in dependency wheels"
                )
        measured_environment = documents.spell(
            documents.embedded_document_digest(environment, "cozy.worker.v1.Environment")
        )
        if measured_environment != entry["environment_digest"]:
            raise documents.DocumentError(
                "environment_digest_mismatch",
                f"measured {measured_environment}, expected {entry['environment_digest']}",
            )
    else:
        forbidden = {
            "environment_digest",
            "environment",
        } & set(entry)
        if forbidden:
            raise documents.DocumentError(
                "development_fields_invalid", f"development carries {sorted(forbidden)[0]}"
            )

    models = entry.get("models", [])
    if not isinstance(models, list):
        raise documents.DocumentError("wrong_type", "Placement.models")
    model_ids: list[str] = []
    for index, value in enumerate(models):
        model = _closed_document(value, {"id", "repo", "version", "lane", "manifest"}, "Model")
        _required(model, {"id", "repo", "version", "manifest"}, "Model")
        model_ids.append(_identifier(model["id"], f"models[{index}].id"))
        for name in ("repo", "version"):
            _identifier(model[name], f"models[{index}].{name}")
        if "lane" in model:
            _identifier(model["lane"], f"models[{index}].lane")
        _exact_ref(model["manifest"], f"models[{index}].manifest")
    if model_ids != sorted(set(model_ids)):
        raise documents.DocumentError("sort_order", "Placement.models")
    known_models = set(model_ids)

    entrypoints = entry["entrypoints"]
    if not isinstance(entrypoints, list):
        raise documents.DocumentError("wrong_type", "Placement.entrypoints")
    entrypoint_names: list[str] = []
    for entrypoint_index, value in enumerate(entrypoints):
        entrypoint = _closed_document(
            value, {"name", "entrypoint_binding_digest", "slots"}, "Entrypoint"
        )
        _required(entrypoint, {"name", "entrypoint_binding_digest"}, "Entrypoint")
        entrypoint_names.append(
            _identifier(entrypoint["name"], f"entrypoints[{entrypoint_index}].name")
        )
        slots = entrypoint.get("slots", [])
        if not isinstance(slots, list):
            raise documents.DocumentError("wrong_type", "Entrypoint.slots")
        slot_names: list[str] = []
        for slot_index, value in enumerate(slots):
            slot = _closed_document(
                value,
                {
                    "slot",
                    "reference_model_id",
                    "components",
                    "model_construction_contract",
                    "stamps",
                },
                "Slot",
            )
            _required(slot, {"slot", "reference_model_id"}, "Slot")
            # THE DELETED PLANE (model-code-fit D1, D12; the proto reserves both at
            # xs-010). `stamps` is `[]` on every wire document — the canonical writer spells
            # closed repeated fields even when empty — and anything in it refuses typed; a
            # `model_construction_contract` ref refuses at all.
            if "model_construction_contract" in slot:
                raise documents.DocumentError(
                    "slot.model_construction_contract_retired",
                    "Slot.model_construction_contract: the ModelConstructionContract is "
                    "deleted (model-code-fit D12); fit is a function of the header and the census",
                )
            stamps = slot.get("stamps", [])
            if not isinstance(stamps, list):
                raise documents.DocumentError("wrong_type", "Slot.stamps")
            if stamps:
                raise documents.DocumentError(
                    "slot.stamps_retired",
                    "Slot.stamps: stamps are deleted (model-code-fit D1); a semantic twin is a "
                    "component name or an inline CozyTensors config fact",
                )
            slot_names.append(_identifier(slot["slot"], f"slots[{slot_index}].slot"))
            if slot["reference_model_id"] not in known_models:
                raise documents.DocumentError("unknown_model", "Slot.reference_model_id")
            components = slot.get("components", [])
            if not isinstance(components, list):
                raise documents.DocumentError("wrong_type", "Slot.components")
            component_names: list[str] = []
            for value in components:
                component = _closed_document(value, {"component", "model_id"}, "Component")
                _required(component, {"component", "model_id"}, "Component")
                component_names.append(_identifier(component["component"], "Component.component"))
                if component["model_id"] not in known_models:
                    raise documents.DocumentError("unknown_model", "Component.model_id")
            if len(component_names) != len(set(component_names)):
                raise documents.DocumentError("sort_order", "Slot.components")
        if slot_names != sorted(set(slot_names)):
            raise documents.DocumentError("sort_order", "Entrypoint.slots")
        expected_entrypoint = (
            "sha256:"
            + hashlib.sha256(
                canonical.write({"name": entrypoint["name"], "slots": slots})
            ).hexdigest()
        )
        if entrypoint["entrypoint_binding_digest"] != expected_entrypoint:
            raise documents.DocumentError(
                "entrypoint_binding_digest_mismatch", str(entrypoint["name"])
            )
    if entrypoint_names != sorted(set(entrypoint_names)):
        raise documents.DocumentError("sort_order", "Placement.entrypoints")
    expected_bindings = (
        "sha256:"
        + hashlib.sha256(
            canonical.write({"entrypoints": entrypoints, "models": models})
        ).hexdigest()
    )
    if entry["bindings_digest"] != expected_bindings:
        raise documents.DocumentError(
            "bindings_digest_mismatch",
            f"measured {expected_bindings}, expected {entry['bindings_digest']}",
        )


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


def executor_prepare_command(
    binding: DeclaredBinding,
    *,
    devices: str,
    authorized_device_limit_bytes: int = 0,
    sequence_parallel_degree: int = 1,
    parameter_names: Sequence[Sequence[str]] = (),
) -> dict[str, Any]:
    """Translate one resolved callable binding into the executor's closed prepare shape.

    `devices` is what the executor's seal must say (`CUDA_VISIBLE_DEVICES`), so the executor
    can refuse `env_seal_broken` if it is preparing under any other seal than its lane's.
    `authorized_device_limit_bytes` is the ceiling the worker ASSIGNED for this prepare
    (cr-025/cr-066); the executor fits under the smaller of it and what it measures, and
    never derives one of its own. `sequence_parallel_degree` rides only when > 1 (cr-068):
    a group lane's K, the ONE new fact its executor receives; a plain prepare is unchanged.
    `parameter_names[i]` is every parameter name the i-th model is bound under across the
    bindings that share this construction (h3a-018); absent, the model binds under this
    binding's own name.
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
        "cmd": "prepare",
        "devices": devices,
        "authorized_device_limit_bytes": int(authorized_device_limit_bytes),
    }
    if sequence_parallel_degree > 1:
        command["sequence_parallel_degree"] = int(sequence_parallel_degree)
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
        # Exact code/PackageInterface/Environment revisions prepared over the loopback seam.
        # This is same-process admission state, not a Runtime journal. After worker restart the
        # RecordOwner replays preparation before a model-bound PlacementSet can join it.
        self.local_revisions: dict[tuple[str, str], Placement] = {}
        self.local_operations: dict[tuple[str, str], Placement] = {}
        self.claim_ready = threading.Event()
        if options.claim_ready:
            self.claim_ready.set()
        self.bindings: dict[str, DeclaredBinding] = {}
        #: plan id -> why THIS binding could not be prepared (#572d). A faulted binding
        #: refuses its own entrypoints and denies its siblings nothing.
        self.failed_bindings: dict[str, str] = {}
        self.prepared_models: dict[str, PreparedModel] = {}
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
                attempt.job.environment_content_digest if attempt.job else "",
                self.base,
                hostfacts.measure(options.accelerator_backend),
            ),
        )
        self.engine.boot_id = self.fence.worker_boot_id
        from cozy_runtime.internal.worker.calls import Calls

        self.calls = Calls(
            self.send, lambda: self.fence.control_stream_epoch, lambda: self.fence.wire_minor
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

    def _new_placement_supervision(self, placement_id: str) -> ExecutorSupervision:
        """Create the one executor slot owned by an additional package placement."""

        key = hashlib.sha256(placement_id.encode()).hexdigest()[:24]
        supervision = ExecutorSupervision(
            root=self.root / "placements" / key,
            python=self.options.python or sys.executable,
            base_env=self.config.child_base_env,
            cozy_home=self.config.cozy_home,
            executor_uid=self._slot_executor_uid(placement_id),
            executor_gid=self.options.executor_gid,
            jit_pod_scope=f"{self.fence.worker_boot_id}-{key}",
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
        """Assign one placement to a measured lane, or latch a typed lane refusal.

        The declared bytes are the placement's model weight bytes from its bindings' MCC
        census; the measured side is each lane's driver-reported free and total bytes, read
        now. Full weight size ranks lanes, while the measured prepare ceiling and actual
        construction/residency working set determine fit. Assignment grants no bytes. The row
        the lane keeps for it remembers `supervision`, the slot the lane asks to vacate
        when a co-tenant needs the card (cr-022).
        """
        model_bearing = any(not binding.weightless for binding in bindings.values())
        declared = sum(binding.model_weight_bytes for binding in bindings.values())
        measured: dict[int, accel.DeviceMemory] = {}
        if model_bearing:
            kind = accel.host_backend_family()
            for ordinal, entry in enumerate(self.lanes.entries):
                measured[ordinal] = accel.device_memory(entry, kind)
        # THE PIN (cr-068): the RecordOwner's placement of this placement on THIS worker,
        # validated here and never chosen here. The declared degrees are the package's.
        pin = self.accepted.device_pins.get(placement.placement_id)
        degrees: set[int] | None = None
        for binding in bindings.values():
            if binding.weightless:
                continue
            held = set(binding.sequence_parallel_degrees())
            degrees = held if degrees is None else degrees & held
        revision = self.lanes.revision
        try:
            lane = self.lanes.assign(
                placement.placement_id,
                model_bearing=model_bearing,
                declared_bytes=declared,
                measured=measured,
                pin=pin,
                degrees=tuple(sorted(degrees or ())),
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
        tenants = [slot for slot in lane.model_bearing_slots() if slot != placement.placement_id]
        self.note(
            "materialization",
            f"{placement.placement_id!r} assigned to {lane.lane_id} "
            f"(CUDA_VISIBLE_DEVICES={lane.devices!r}; "
            + ("model-bearing" if model_bearing else "weightless")
            + (f"; a {lane.degree}-device GROUP, one seat" if lane.group else "")
            + (f"; shares the device with {tenants} by residency arbitration" if tenants else "")
            + ")",
        )
        if self.lanes.revision != revision:
            self.bump_admission(
                f"{lane.lane_id} fused from {list(lane.ordinals)} for {placement.placement_id!r}"
            )
        return True

    def _release_lane(self, placement_id: str) -> None:
        placement = self._placement_by_id(placement_id)
        if placement is not None:
            placement.lane_id = ""
        revision = self.lanes.revision
        self.lanes.release(placement_id)
        if self.lanes.revision != revision:
            self.bump_admission(f"a group lane split when {placement_id!r} left it")

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
        if attempt.spec.get("job") is not None or self.accepted.mode == "job":
            descriptor = (attempt.spec.get("job") or {}).get("job_descriptor_id", "")
            selected_job = next(
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
            prepared: Mapping[str, PreparedModel] = {}
            failed: Mapping[str, str] = {}
        else:
            hosted = self.hosted.get(attempt.placement_id)
            if hosted is None:
                lane = self._primary_lane()
                supervision = self.supervision
                row = lane.row(self._primary_slot())
                bindings = self.bindings
                prepared = self.prepared_models
                failed = self.failed_bindings
            else:
                lane = self._lane_of(attempt.placement_id)
                supervision = hosted.supervision
                row = lane.row(attempt.placement_id)
                bindings = hosted.bindings
                prepared = hosted.prepared_models
                failed = hosted.failed_bindings
        if attempt.lane_id and attempt.lane_id != lane.lane_id:
            raise AttemptRefusal(
                "placement_lane_moved",
                f"the offer took a seat on {attempt.lane_id} and placement "
                f"{attempt.placement_id!r} is now on {lane.lane_id}; the RecordOwner "
                "re-dispatches against the current snapshot",
                cause=CAUSE.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE,
            )
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
            prepared_models=prepared,
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
            orchestration=lane is self.lanes.orchestration,
        )
        attempt.slot = slot
        attempt.lane_id = lane.lane_id
        return slot

    def _row_changed(self, placement_id: str, executor: child.Executor | None) -> None:
        """A HOSTED placement's executor was published or cleared: move ITS row only."""
        lane = self.lanes.lane_of(placement_id)
        if lane is not None:
            lane.row(placement_id).ledger.begin_generation(executor.pid if executor else 0)

    def _lane_phases(self, lane: DeviceLane) -> tuple[int, int]:
        """(attempts holding the device, released attempts awaiting ack) on ONE lane."""
        on_device = released = 0
        for held in list(self.engine.history.values()):
            if held.lane_id != lane.lane_id:
                continue
            if held.state in self.engine.ON_DEVICE:
                on_device += 1
            elif held.state in self.engine.RELEASED:
                released += 1
        return on_device, released

    def _lane_dispatchable(self, lane: DeviceLane) -> bool:
        """Whether THIS lane has a placement that takes offers (job mode: the envelope)."""
        if self.accepted.mode == "job":
            slot = self.job_slots.get(lane.lane_id)
            return (
                slot.ready() if slot is not None else lane is self.lanes.envelope and self.job_ready
            )
        return any(self._placement_dispatchable(placement_id) for placement_id in lane.placements)

    def _lane_slots(self, lane: DeviceLane) -> int:
        """Free seats on ONE lane, now: `LANE_SEATS` less what holds its device, NONE while
        more than `LANE_RELEASED` released attempts await their ack (cr-079 D3), and NONE
        while the worker's admission is CLOSED — that is one structural fact per worker,
        and `lanes[]` must never show a seat the worker-level count denies."""
        if self.admission_state() != pb.AdmissionState.ADMISSION_STATE_OPEN:
            return 0
        if not self._lane_dispatchable(lane):
            return 0
        on_device, released = self._lane_phases(lane)
        if released > LANE_RELEASED:
            return 0
        return max(0, LANE_SEATS - on_device)

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

    def check_package_set_compatibility(
        self, request: pb.PreparePackageSetRequest
    ) -> pb.CheckPackageSetCompatibilityResult:
        """Cold metadata-only CPU/meta fit; no device lane or executor is acquired."""

        from cozy_runtime.internal.worker import package_prepare

        with self.preparation_lock:
            try:
                return package_prepare.check_package_set_compatibility(
                    request,
                    tensorfs_root=Path(self.options.tensorfs_root or ""),
                    install_root=Path(self.options.install_root or ""),
                    python=Path(self.options.python or sys.executable),
                    base=self.base,
                )
            except package_prepare.PreparationRefusal as exc:
                return pb.CheckPackageSetCompatibilityResult(
                    refusal_code=exc.code,
                    refusal_detail="".join(
                        character if " " <= character <= "~" else "?" for character in exc.detail
                    )[:1024],
                )

    def prepare_package_set(
        self, request: pb.PreparePackageSetRequest
    ) -> pb.PreparePackageSetResult:
        return self._prepare_package(request, local=False)

    def prepare_local_package(
        self, request: pb.PrepareLocalPackageRequest
    ) -> pb.PreparePackageSetResult:
        return self._prepare_package(request, local=True)

    def prepare_private_placement(
        self, request: pb.PreparePrivatePlacementRequest
    ) -> pb.PreparePackageSetResult:
        from cozy_runtime.internal.worker import package_prepare

        with self.preparation_lock:
            revision = documents.spell(request.local_revision_digest)
            prepared = self.local_operations.get((request.operation_id, revision))
            if prepared is None:
                raise package_prepare.PreparationRefusal("local_revision_unprepared", revision)

            def mark_verified(digest: str, length: int, path: Path) -> None:
                self.verified_artifacts.record(digest, length, path)

            result = package_prepare.prepare_private_placement(
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
            joined = read_placement_set(result.placement_set)
            if (
                len(joined) != 1
                or joined[0].get("development", {}).get("local_revision_digest") != revision
            ):
                raise package_prepare.PreparationRefusal(
                    "private_placement_result_invalid", revision
                )
            return result

    def _prepare_package(
        self,
        request: pb.PreparePackageSetRequest | pb.PrepareLocalPackageRequest,
        *,
        local: bool,
    ) -> pb.PreparePackageSetResult:
        """Derive one local PlacementSet while package imports stay in the isolated executor."""

        with self.preparation_lock:
            # A worker already serving placements prepares BESIDE them: the describe
            # executor runs in a preparation-scoped slot, so the serving placements'
            # executors are never replaced by a preparation. One preparation at a time
            # (this lock); the slot is retired when the preparation returns. An idle
            # worker keeps using the primary slot so the desired state that follows
            # reuses the already-spawned executor and its environment.
            busy = (
                self.placement is not None
                or self.pending is not None
                or bool(self.bindings)
                or bool(self.hosted)
            )
            supervision = self._new_preparation_supervision() if busy else self.supervision
            try:
                return self._prepare_package_in(request, supervision, local=local)
            finally:
                if busy:
                    executor = supervision.current
                    if executor is not None:
                        try:
                            supervision.retire_current(executor, "package preparation finished")
                        except ExecutorGone as exc:
                            self.note("recovery", f"preparation executor reclaim failed: {exc}")
                    supervision.close()

    def _new_preparation_supervision(self) -> ExecutorSupervision:
        """The preparation-scoped executor slot. Package describe needs an isolated
        executor (package code never gets worker authority), and borrowing the primary
        slot while placements serve would replace a serving executor with the prepared
        package's environment."""

        supervision = ExecutorSupervision(
            root=self.root / "placements" / "prepare",
            python=self.options.python or sys.executable,
            base_env=self.config.child_base_env,
            cozy_home=self.config.cozy_home,
            executor_uid=self._slot_executor_uid("prepare"),
            executor_gid=self.options.executor_gid,
            jit_pod_scope=f"{self.fence.worker_boot_id}-prepare",
        )
        supervision.on_lane_failure = self.fail_machine
        killed = supervision.sweep_orphans()
        if killed:
            self.note("recovery", f"preparation: reclaimed {len(killed)} prior executor(s)")
        return supervision

    def _prepare_package_in(
        self,
        request: pb.PreparePackageSetRequest | pb.PrepareLocalPackageRequest,
        supervision: ExecutorSupervision,
        *,
        local: bool,
    ) -> pb.PreparePackageSetResult:
        from cozy_runtime.internal.worker import package_prepare

        def describe(
            installed: package_environment.InstalledEnvironment, distribution: str
        ) -> bytes:
            changed = supervision.use_environment(str(installed.python), installed.content_digest)
            executor = supervision.current
            lane = self._primary_lane()
            try:
                if executor is None:
                    executor = supervision.spawn(imposed=self.imposed(lane))
                elif changed or not executor.alive():
                    executor = supervision.replace(
                        executor, imposed=self.imposed(lane), why="prepared package changed"
                    )
                output = supervision.handoff(executor.epoch, INTERFACE_DOCUMENT)
                # A cold import, torch included: bound by the executor's own measured
                # work and its death, never by a number of seconds (xs-007 row 8).
                with executor.watched("describe_installed"):
                    reply = executor.call(
                        {
                            "cmd": "describe_installed",
                            "distribution": distribution,
                            "output": str(output),
                        },
                        timeout=None,
                    )
            except ExecutorGone as exc:
                raise package_prepare.PreparationRefusal(
                    "package_prepare_executor_failed", exc.detail
                ) from exc
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
            digest = "sha256:" + hashlib.sha256(raw).hexdigest()
            if reply.get("digest") != digest or reply.get("length") != len(raw):
                # QUOTE BOTH SIDES OF BOTH COMPARISONS. This reported a digest/length
                # disagreement as `reply.get("code")` — absent on an ok reply, so the
                # detail was the literal words "invalid reply" and an operator could not
                # tell which of the two values disagreed, or by how much.
                raise package_prepare.PreparationRefusal(
                    "package_prepare_interface_failed",
                    f"the executor's document does not match its own claim: "
                    f"digest {reply.get('digest')!r} vs {digest!r}, "
                    f"length {reply.get('length')!r} vs {len(raw)}",
                )
            return raw

        def mark_verified(digest: str, length: int, path: Path) -> None:
            self.verified_artifacts.record(digest, length, path)

        if local:
            assert isinstance(request, pb.PrepareLocalPackageRequest)
            materialized: list[package_environment.InstalledEnvironment] = []
            result = package_prepare.prepare_local_package(
                request,
                artifact_cache=Path(self.options.artifact_cache or ""),
                install_root=Path(self.options.install_root or ""),
                python=Path(self.options.python or sys.executable),
                describe=describe,
                verified=mark_verified,
                job_plan_root=self.config.cozy_home / "job-plans",
                base=self.base,
                materialized=materialized.append,
                dependency_python=self.options.environment_python,
            )
            prepared = read_placement_set(result.placement_set)
            if len(prepared) != 1 or not prepared[0].get("development"):
                raise package_prepare.PreparationRefusal(
                    "local_package_result_invalid",
                    "local preparation did not return one development Placement",
                )
            entry = prepared[0]
            package_interface = dict(entry.get("package_interface", {}))
            stored = Placement(
                placement_id=str(entry.get("placement_id", "")),
                placement_set_digest=result.placement_set.placement_set_digest,
                package=dict(entry.get("package", {})),
                development=dict(entry.get("development", {})),
                environment_digest=_digest_field(entry.get("environment_digest")),
                package_interface=(
                    str(package_interface.get("digest", "")),
                    int(package_interface.get("length", 0)),
                ),
                bindings_digest=_digest_field(entry.get("bindings_digest")),
                models=tuple(dict(row) for row in entry.get("models", [])),
                entrypoints=tuple(dict(row) for row in entry.get("entrypoints", [])),
                environment=dict(entry.get("environment", {})),
                document=dict(entry),
                installed=materialized[0] if materialized else None,
            )
            revision = str(stored.development.get("local_revision_digest", ""))
            if revision != documents.spell(request.package.local_revision_digest):
                raise package_prepare.PreparationRefusal(
                    "local_package_revision_mismatch",
                    "prepared Placement changed the declared local revision",
                )
            self.local_operations[(request.operation_id, revision)] = stored
            self.local_revisions[(stored.placement_id, revision)] = stored
            return result
        assert isinstance(request, pb.PreparePackageSetRequest)
        return package_prepare.prepare_package_set(
            request,
            artifact_cache=Path(self.options.artifact_cache or ""),
            tensorfs_root=Path(self.options.tensorfs_root or ""),
            install_root=Path(self.options.install_root or ""),
            python=Path(self.options.python or sys.executable),
            describe=describe,
            verified=mark_verified,
            job_plan_root=self.config.cozy_home / "job-plans",
            base=self.base,
            preinstalled_python=Path(self.options.environment_python)
            if self.options.environment_python is not None
            else None,
        )

    def numerical_environment(self) -> bytes:
        from cozy_runtime.internal.numerical_environment import fingerprint

        return fingerprint(
            self.base,
            hostfacts.measure(self.options.accelerator_backend),
            threads=self.options.threads,
            alloc_conf=self.options.alloc_conf,
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
                or claim.wire_minor < MIN_COMPATIBLE_WIRE_MINOR
            ):
                raise WorkspaceRefusal(
                    "workspace call does not name the current authenticated owner"
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
        from . import store_gc

        return store_gc.collect(Path(self.options.tensorfs_root or ""))

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
        with self.claim_lock:
            if frame.WhichOneof("msg") != "weights_transaction":
                self.stamp(inner)
            self.outbound.put(frame)

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
            tensorfs = fill.tensorfs_module()
            store = fill.open_store(tensorfs, Path(root))
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
        """CLOSED is STRUCTURAL (#486c): before the snapshot barrier, while draining, while
        the placement is not dispatchable. OPEN with zero slots is TRANSIENT saturation, and
        a RecordOwner backs off differently for each — so the two never collapse into one
        counter here either."""
        if (
            not self.fence.dispatch_open
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
            self.lane_threads[name] = threading.get_native_id()
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
        return {
            tid
            for name, tid in self.lane_threads.items()
            if name in ("report", "watchdog", "control-send")
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
        key = (request_id, attempt)
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
        if claim.wire_minor < MIN_COMPATIBLE_WIRE_MINOR:
            return reject(
                pb.ClaimRejection.CLAIM_REJECTION_PROTOCOL_INCOMPATIBLE,
                "peer is below the supported wire compatibility floor",
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
        """Open Claim only after fixed-host listener proof and readiness publication."""

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
        body.held_attempts.extend(self.engine.snapshot_entries())
        if self.workspace is not None:
            known = {(entry.request_id, entry.attempt_ordinal) for entry in body.held_attempts}
            for outcome in self.workspace.retained_outcomes(self.fence.record_owner_id):
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
        if kind == "attempt_offer":
            self.on_offer(inner)
        elif kind == "cancel_attempt":
            if self.source_calls is not None:
                self.source_calls.cancel_parent(
                    inner.request_id,
                    ordinal=inner.attempt_ordinal,
                    spec=inner.invocation_spec_digest,
                )
            self.note("cancel", self.engine.cancel(inner))
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
                    if not self.fenced(acked):
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
            lane.work.put(None)
            lane.post.put(None)

    def request_restart(self) -> None:
        """A local operator requests process replacement while retaining source work."""
        if not self.stop.is_set():
            self.restart_requested = True
            self.stop.set()

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

    @staticmethod
    def _placement_from_entry(entry: Mapping[str, Any], placement_set_digest: bytes) -> Placement:
        package_interface = dict(entry.get("package_interface", {}))
        return Placement(
            placement_id=str(entry.get("placement_id", "")),
            placement_set_digest=placement_set_digest,
            package=dict(entry.get("package", {})),
            development=dict(entry.get("development", {})),
            environment_digest=_digest_field(entry.get("environment_digest")),
            package_interface=(
                str(package_interface.get("digest", "")),
                int(package_interface.get("length", 0)),
            ),
            bindings_digest=_digest_field(entry.get("bindings_digest")),
            models=tuple(dict(row) for row in entry.get("models", [])),
            entrypoints=tuple(dict(row) for row in entry.get("entrypoints", [])),
            environment=dict(entry.get("environment", {})),
            document=dict(entry),
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
        if (
            mode != "job"
            and self.job_slots
            and any(self._job_lane_busy(slot.lane) for slot in self.job_slots.values())
        ):
            self.engine.fault(
                pb.Fault(
                    kind=pb.FAULT_KIND_CONFIG_REFUSED,
                    subject=f"revision {desired.revision}",
                    reason="job_set_busy",
                    detail="stop private attempts before replacing their job set",
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
        self.fence.wire_minor = min(WIRE_MINOR, desired.wire_minor)
        self.accepted.drain_grace_ms = desired.drain_grace_ms
        if self.accepted.mode == "job" and mode != "job":
            self.lanes.envelope.forget("")  # the job generation's row leaves with the job
        self.accepted.mode = "serving" if mode == "placement_set" else mode
        if mode != "job" and self.job_slots:
            cpu = self.job_slots.get("orchestration")
            if cpu is not None:
                cpu.supervision.close()
                self.lanes.orchestration.forget("")
            self.job_slots.clear()
            self.jobs.clear()
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
            self.prepared_models = {}
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
            # and `LaneSet.assign` decides per placement (ruling 3: a model-bearing placement
            # with no pin on a wide envelope refuses `device_group_unsupported`).
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
            self.prepared_models = {}
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
            # explicit administrative act and there is no default that performs one.
            if previous.materialization != pb.MaterializationState.MATERIALIZATION_STATE_STAGED:
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
        if self.pending is None and self.placement is not None:
            self._reconcile_hosted(revision)
            return
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
                str(installed.python), installed.content_digest
            )
            if changed and self.supervision.current is not None:
                self.supervision.current.poisoned = "package environment generation changed"
        bindings = dict(acquired.bindings)
        if revision != self.accepted.accepted_desired_state_revision:
            self.note("materialization", f"revision {revision} was superseded before hand-off")
            return

        # ---- THE DEVICE HAND-OFF. Everything above was concurrent with serving. ----------
        candidate.materialization = pb.MaterializationState.MATERIALIZATION_STATE_STAGED
        if outgoing is not None:
            self.note(
                "materialization",
                f"incoming set {documents.spell(candidate.placement_set_digest)[:23]}… is STAGED; "
                f"draining {outgoing.placement_id or '(unnamed)'} and handing off the device",
            )
            self.await_drained()
            if outgoing.placement_id != candidate.placement_id:
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
        self.prepared_models = {}
        self.bump_admission(
            f"placement {candidate.placement_id or '(unnamed)'} moved to set "
            f"{documents.spell(candidate.placement_set_digest)[:23]}…"
        )
        self.note(
            "desired_state",
            f"revision {revision} STAGED: placement "
            f"{candidate.placement_id or '(unnamed)'} in set "
            f"{documents.spell(candidate.placement_set_digest)[:23]}… with {len(self.bindings)} of "
            f"{len(candidate.entrypoints)} binding(s) resolved, "
            f"posture={pb.Posture.Name(self.accepted.posture)}",
        )
        self.activate_executor()
        if self._placement_dispatchable(candidate.placement_id):
            self._reconcile_hosted(revision)

    def _retire_hosted(self, keep: set[str]) -> None:
        """Remove only placements absent from the complete desired set."""

        for placement_id in sorted(set(self.hosted) - keep):
            hosted = self.hosted[placement_id]
            self._hosted_set_serving(
                placement_id,
                pb.ServingState.SERVING_STATE_DRAINING,
                f"placement {placement_id!r} removed from desired state",
            )
            for attempt in list(self.engine.history.values()):
                while attempt.placement_id == placement_id and attempt.state in self.engine.HELD:
                    time.sleep(0.02)
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
                    continue
            hosted.supervision.close()
            del self.hosted[placement_id]
            self._release_lane(placement_id)
            self.bump_admission(f"placement {placement_id!r} removed")

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
            and self.hosted[placement_id].placement.document != placement.document
        }
        self._retire_hosted(set(desired) - changed)
        for placement_id, placement in desired.items():
            if revision != self.accepted.accepted_desired_state_revision:
                return
            hosted = self.hosted.get(placement_id)
            if hosted is not None:
                hosted.placement.placement_set_digest = self.accepted.accepted_placement_set_digest
                continue
            hosted = HostedPlacement(
                placement=placement,
                supervision=self._new_placement_supervision(placement_id),
            )
            self.hosted[placement_id] = hosted
            self._activate_hosted(hosted, revision)
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

    def _activate_hosted(self, hosted: HostedPlacement, revision: int) -> None:
        """Acquire, place and activate one hosted placement in its own uv environment."""

        placement = hosted.placement
        placement.materialization = pb.MaterializationState.MATERIALIZATION_STATE_MATERIALIZING
        acquired = self._acquire(placement, revision)
        if acquired is None or revision != self.accepted.accepted_desired_state_revision:
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
            hosted.supervision.use_environment(str(installed.python), installed.content_digest)
        hosted.bindings = bindings
        hosted.prepared_models = {}
        hosted.failed_bindings = {}
        if not self._assign_lane(placement, bindings, hosted.supervision):
            placement.materialization = pb.MaterializationState.MATERIALIZATION_STATE_FAILED
            hosted.latched = "device_lane_infeasible"
            self._settle()
            return
        placement.materialization = pb.MaterializationState.MATERIALIZATION_STATE_STAGED
        self._prepare_hosted_executor(hosted)

    def _prepare_hosted_executor(self, hosted: HostedPlacement) -> None:
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

    def await_drained(self) -> None:
        """Step 2 of the hand-off: admission stops, ALREADY-ACCEPTED attempts settle.

        NO DRAIN CONSTANT EXISTS (#434/#479). This waits on the attempts the RecordOwner
        already accepted and on THEIR own deadlines — nothing else. Cutting the tail short is
        `force`, an administrative act with a recorded actor, and there is no default number
        in any config file that performs one.
        """
        self.set_serving(pb.ServingState.SERVING_STATE_DRAINING, "handing off the device")
        for attempt in list(self.engine.history.values()):
            if attempt.state in self.engine.HELD:
                self.await_settled(attempt)

    def _acquire(self, placement: Placement, revision: int) -> acquire.AcquiredPlacement | None:
        """Fetch what the spec reaches and join its receipt to content-keyed installed bytes.

        Returns the installed generation, or None having already typed the fault and left the
        outgoing placement serving.
        """
        if self.options.development_project is not None:
            expected_development = {
                "package": self.options.development_package,
                "release": self.options.development_release,
                "source_digest": self.options.development_source_digest,
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
                    source_digest=self.options.development_source_digest,
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
        if placement.development and "project_wheel" not in placement.development:
            self._materialization_failed(
                placement,
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    subject=placement.placement_id,
                    reason="development_project_absent",
                    detail="a live development Placement requires an explicit same-user "
                    "--development-project launch; a remote snapshot must carry its exact wheel",
                ),
                terminal=True,
            )
            return None
        if placement.development:
            local_revision = str(placement.development.get("local_revision_digest", ""))
            prepared = self.local_revisions.get((placement.placement_id, local_revision))
            if prepared is None:
                self._materialization_failed(
                    placement,
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        subject=placement.placement_id,
                        reason="local_revision_unprepared",
                        detail="prepare this exact local code revision before binding models",
                    ),
                    terminal=True,
                )
                return None
            core_fields = (
                "development",
                "environment_digest",
                "environment",
                "package_interface",
            )
            changed = next(
                (
                    name
                    for name in core_fields
                    if placement.document.get(name) != prepared.document.get(name)
                ),
                "",
            )
            if changed:
                self._materialization_failed(
                    placement,
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        subject=placement.placement_id,
                        reason="local_revision_changed",
                        detail=f"model-bound Placement changed prepared field {changed}",
                    ),
                    terminal=True,
                )
                return None
            prepared_environment = prepared.installed
        else:
            prepared_environment = None
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
                package = placement.package or placement.development
                project_wheel = package.get("project_wheel", {})
                project_ref = (
                    project_wheel.get("ref", {}) if isinstance(project_wheel, dict) else {}
                )
                # Published placements carry no wheel fact at wire 30: the derived
                # package interface is the exact code surface, so it is the code digest.
                package_code_digest = (
                    placement.package_interface[0]
                    if placement.package
                    else str(project_ref.get("digest", ""))
                    or str(placement.development.get("source_digest", ""))
                )
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", package_code_digest):
                    raise package_environment.EnvironmentRefusal(
                        "package_environment_code_digest_absent",
                        "preinstalled package selection has no exact interface, "
                        "project-wheel, or source digest",
                    )
                content_digest = canonical.digest(
                    {
                        "package_code_digest": package_code_digest,
                        "package_interface": {
                            "digest": placement.package_interface[0],
                            "length": placement.package_interface[1],
                        },
                        "environment_digest": documents.spell(placement.environment_digest),
                    }
                )
                installed_environment = package_environment.open_existing_environment(
                    self.options.environment_python, content_digest
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
                f"revision {revision} acquired {acquired.wheels} wheel(s), "
                f"{acquired.documents} selected document(s), hydrated "
                f"{acquired.model_objects} model object(s) — "
                f"{acquired.fetched_bytes} new byte(s) — and published "
                + (
                    f"content {acquired.installed.content_digest} at "
                    f"{acquired.installed.generation}; "
                    if acquired.installed is not None
                    else "source-checkout content; "
                )
                + f"verified {acquired.verified_bytes} byte(s), "
                f"trusted same-boot reuse {acquired.trusted_reuse_bytes} byte(s)",
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
                    self.config.cozy_home / "job-plans", parent.build_id, parent.job_descriptor_id
                )
                binding = JobBinding.read(json.loads(record.read_text()))
                declaration = self.engine._job_declaration(binding, parent.job_descriptor_id)
                if (
                    binding.build_id != parent.build_id
                    or binding.job_descriptor_id != parent.job_descriptor_id
                    or not parent.reclaim_on_terminal
                ):
                    raise ValueError("orchestration directive differs from its prepared job")
                if (
                    parent.device_count
                    or parent.resource_caps.device_required
                    or parent.resource_caps.max_device_memory_bytes
                    or declaration.get("models")
                    or declaration.get("weights_outputs")
                    or not binding.call_interfaces
                ):
                    raise ValueError(
                        "orchestration requires frozen child interfaces "
                        "and no Model, weights, or GPU grant"
                    )
                if not job.orchestration and parent.job_descriptor_id == job.job_descriptor_id:
                    raise ValueError("one job cannot occupy the parent and ordinary slots together")
            old_cpu = self.job_slots.get("orchestration")
            cpu_same = bool(
                parent is not None and old_cpu is not None and old_cpu.directive == parent
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
                attempt.kind != "job" and attempt.state in self.engine.ON_DEVICE
                for attempt in self.engine.history.values()
            ):
                raise ValueError("stop active serving attempts before entering the private job set")

            if not cpu_same and old_cpu is not None:
                old_cpu.supervision.close()
                self.job_slots.pop("orchestration", None)
                self.lanes.orchestration.forget("")
            if parent is not None and not cpu_same:
                supervision = self._new_orchestration_supervision()
                supervision.use_environment(binding.python, binding.environment_content_digest)
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
                self.prepared_models = {}
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
            attempt.lane_id == lane.lane_id and attempt.state in self.engine.ON_DEVICE
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
        self.prepared_models = {}
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
            record = job_plan.path(plans, job.build_id, job.job_descriptor_id)
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
        binding = JobBinding.read(json.loads(record.read_text()))
        if (
            binding.job_descriptor_id != job.job_descriptor_id
            or binding.build_id != job.build_id
            or not job.reclaim_on_terminal
        ):
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                    subject=job.job_descriptor_id,
                    reason="job_plan_mismatch",
                    detail="JobDirective does not equal the prepared package job plan",
                )
            )
            self.latched = "job_plan_mismatch"
            self._settle()
            return
        self.supervision.use_environment(binding.python, binding.environment_content_digest)
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

    def activate_executor(self) -> None:
        """Contain every activation defect as observable OFFLINE state, never a dead lane."""

        try:
            self.prepare_executor()
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

    def prepare_executor(self) -> None:
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
        # A fresh activation re-earns every verdict; a stale fault must not outlive the
        # prepare that produced it.
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
                prepared_models=self.prepared_models,
                failed_bindings=self.failed_bindings,
                lane=self._primary_lane(),
                slot=self._primary_slot(),
            )
        return Tenant(
            placement=hosted.placement,
            supervision=hosted.supervision,
            bindings=hosted.bindings,
            prepared_models=hosted.prepared_models,
            failed_bindings=hosted.failed_bindings,
            lane=self._lane_of(placement_id),
            slot=placement_id,
        )

    def _prepare_generation(self, tenant: Tenant) -> tuple[str, bool]:
        """ONE placement's generation: spawn or reuse its executor, prepare its ONE
        construction under the lane's assigned ceiling, bank the construction facts on its
        row. The same code for the primary slot and for every hosted placement (cr-022).

        Returns `(latch, reused)`: an empty latch means at least one binding is ready;
        `reused` means the live generation already held this construction and nothing was
        prepared. A model-bearing prepare holds the lane's DEVICE lock: an attempt on a
        co-tenant cannot be on the card while this prepare vacates it or fills beside it.
        """
        by_construction: dict[str, list[DeclaredBinding]] = {}
        for binding in tenant.bindings.values():
            by_construction.setdefault(binding.construction_key(), []).append(binding)
        placement_id = tenant.placement.placement_id
        if len(by_construction) != 1:
            # ONE construction per executor. Two model-bearing placements share a lane as
            # two executors (residency arbitration between them); two constructions in one
            # placement would be one executor with two generations, which nothing prices.
            detail = (
                f"placement {placement_id!r} received {len(by_construction)} distinct model "
                "constructions, but one executor can own exactly one prepared generation"
            )
            for binding in tenant.bindings.values():
                tenant.failed_bindings[binding.entrypoint_binding_digest] = detail
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=binding.entrypoint_binding_digest,
                        reason="multiple_constructions_unsupported",
                        detail=detail,
                    )
                )
            executor = tenant.supervision.current
            if executor is not None:
                try:
                    tenant.supervision.retire_current(executor, detail)
                except ExecutorGone as exc:
                    self.note("recovery", f"multi-construction reclaim failed: {exc}")
            return "multiple_constructions_unsupported", False

        construction_key, group = next(iter(by_construction.items()))
        # THE GENERATION IS THE CONSTRUCTION, NOT THE BINDING SET (h3a-018). A desired set
        # that adds an entrypoint over bytes this process already holds reuses the process:
        # the prepared model serves every binding of the construction, provided the
        # executor bound it under each parameter name the new set spells. Equality on the
        # binding set rebuilt the executor for a placement that changed nothing it fills.
        parameters = _group_parameters(group)
        executor = tenant.supervision.current
        if (
            executor is not None
            and executor.alive()
            and not executor.poisoned
            and executor.construction_key == construction_key
            and parameters <= executor.parameters
            and executor.prepared_model is not None
        ):
            executor.ready_bindings.clear()
            for binding in group:
                executor.ready_bindings.add(binding.entrypoint_binding_digest)
                tenant.prepared_models[binding.entrypoint_binding_digest] = executor.prepared_model
            self.note(
                "boot",
                f"{placement_id!r} reused epoch {executor.epoch} pid {executor.pid} for the "
                f"same construction ({len(executor.ready_bindings)} binding(s))",
            )
            if executor.ready_bindings:
                return "", True
            return "logical model digest mismatch", False

        first = group[0]
        # THE CARD IS TAKEN LATE (cr-104). Owning the process and warming its device-free
        # half hold no device bytes and need no lock; only the half that MOVES bytes does.
        # Held for the whole prepare, the lock kept a co-tenant off a card that the tenant
        # taking it would not touch for another 4-20 s of Torch, runtime and package
        # imports -- and `_make_room` had already vacated that co-tenant to get there, so
        # the card was emptied and then left idle for the length of an import.
        lock = contextlib.nullcontext() if first.weightless else tenant.lane.device
        with lock:
            owned = self._own_executor(tenant, group)
        if isinstance(owned, str):
            return owned, False
        if not first.weightless:
            self._preconstruct(tenant, owned, first)
        with lock:
            return self._prepare_construction(tenant, group, construction_key, parameters, owned)

    def _own_executor(self, tenant: Tenant, group: list[DeclaredBinding]) -> child.Executor | str:
        """Spawn, keep or replace this tenant's executor. Returns it, or a latch string.

        Separated from the prepare so the process exists BEFORE the device-free warmup that
        overlaps a co-tenant (cr-104). A replacement kills a process holding device bytes,
        so this runs under the lane's lock like every other act that moves the card's
        occupancy; nothing here fills.
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
                and not executor.ready_bindings
                and executor.prepared_model is None
            ):
                self.note(
                    "boot",
                    f"{placement_id!r}: preparing fresh epoch {executor.epoch} pid {executor.pid}",
                )
            else:
                self.note(
                    "boot",
                    f"{placement_id!r}: replacing the device executor for a new construction",
                )
                executor = tenant.supervision.replace(
                    executor,
                    imposed=self.imposed(lane),
                    why="desired construction changed",
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

    def _preconstruct(
        self, tenant: Tenant, executor: child.Executor, first: DeclaredBinding
    ) -> None:
        """Warm the executor's device-free half WITHOUT the lane's device lock (cr-104).

        This is the whole of the overlap: while a co-tenant computes on the card, the next
        tenant's process imports Torch, the runtime's construction modules, its package and
        whatever the package defers. It moves no weight byte and asks the lane for nothing,
        so EVERY failure here is a missed saving and never a refusal -- the prepare that
        follows does the same work under the lock and refuses there, typed, exactly as it
        does today. That is why this returns nothing and latches nothing.
        """
        command = executor_prepare_command(
            first, devices=tenant.lane.devices, authorized_device_limit_bytes=0
        )
        command["cmd"] = "preconstruct"
        command.pop("authorized_device_limit_bytes", None)
        placement_id = tenant.placement.placement_id
        started = time.perf_counter()
        try:
            reply = executor.call(command, timeout=None)
        except ExecutorGone as exc:
            # The prepare below will spawn or refuse on the same fact, with the record.
            self.note("boot", f"{placement_id!r}: preconstruct lost epoch {executor.epoch}: {exc}")
            return
        took = (time.perf_counter() - started) * 1000
        if not reply.get("ok"):
            self.note(
                "boot",
                f"{placement_id!r}: preconstruct declined ({reply.get('code')}): "
                f"{str(reply.get('detail'))[:160]}",
            )
            return
        if reply.get("skipped"):
            return
        stages = reply.get("stages") or ()
        self.note(
            "boot",
            f"{placement_id!r}: preconstruct warmed epoch {executor.epoch} in {took:.0f} ms "
            "OFF the lane lock: "
            + ", ".join(f"{name} {ms:.0f}ms" for name, ms in stages)
            + (
                f" [warmup failed: {reply.get('warmup_error')}]"
                if reply.get("warmup_error")
                else ""
            )
            + (
                " [the package opened CUDA itself]"
                if reply.get("adopted_package_cuda_init")
                else ""
            ),
        )

    def _prepare_construction(
        self,
        tenant: Tenant,
        group: list[DeclaredBinding],
        construction_key: str,
        parameters: frozenset[str],
        executor: child.Executor,
    ) -> tuple[str, bool]:
        """The device half of `_prepare_generation`, under the lane's device lock."""
        lane, slot = tenant.lane, tenant.slot
        placement_id = tenant.placement.placement_id
        if tenant.supervision.current is not executor:
            # The lock was released for the warmup, so ownership could have moved under it.
            # Whatever owns the slot now prepares on the next convergence; this generation
            # does not fill into a process the supervision has already let go.
            self.note(
                "boot",
                f"{placement_id!r}: epoch {executor.epoch} was replaced during its warmup",
            )
            return "executor_replaced_during_warmup", False

        first = group[0]
        started = time.perf_counter()
        subject = f"prepare:{first.components[0] if first.components else first.entrypoint}"
        if not first.weightless:
            # THE LANE IS SHARED BY RESIDENCY (cr-022): before this generation is handed a
            # ceiling, co-tenants that leave it no room for its declared weights give the
            # card up. What is measured is the driver's free bytes less the lane's other
            # outstanding ceilings; what is declared is the binding's own MCC census.
            declared = sum(binding.model_weight_bytes for binding in group)
            lane.touch(slot)
            self._make_room(
                lane,
                slot,
                lambda memory: memory.free_bytes - lane.outstanding_bytes() >= declared,
                f"prepare of {placement_id!r} declaring {declared} B",
            )
        try:
            authorized = self._authorize_prepare(lane, slot, first)
        except LaneRefusal as exc:
            # No byte moved and the executor is untouched: the BINDINGS fault, typed,
            # and the placement reports what the driver would not say.
            for member in group:
                tenant.failed_bindings[member.entrypoint_binding_digest] = str(exc)
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=member.entrypoint_binding_digest,
                        reason=exc.code,
                        detail=safe(exc.detail),
                    )
                )
            self.note("boot", f"{placement_id!r} prepare REFUSED before the executor: {exc}"[:256])
            return "prepare refused: every binding", False
        self.monitor.start(subject, "load")
        try:
            reply = executor.call(
                executor_prepare_command(
                    first,
                    devices=lane.devices,
                    authorized_device_limit_bytes=authorized,
                    sequence_parallel_degree=lane.degree if lane.group else 1,
                    parameter_names=_parameter_names_by_model(group),
                ),
                timeout=None,
                on_progress=lambda frame: self.monitor.advance(
                    f"prepare:{frame.get('name', '')}", int(frame.get("position", 0) or 0)
                ),
            )
        except ExecutorGone as exc:
            lane.settle(slot)
            self.monitor.end(subject, "executor_gone")
            detail = f"executor disappeared during prepare: {exc}"[:1024]
            for member in group:
                tenant.failed_bindings[member.entrypoint_binding_digest] = detail
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=member.entrypoint_binding_digest,
                        reason="executor_gone_during_prepare",
                        detail=safe(detail),
                    )
                )
            try:
                tenant.supervision.retire_current(executor, detail)
            except ExecutorGone as reclaim:
                self.note("recovery", f"prepare-loss reclaim failed: {reclaim}")
            return "prepare refused: every binding", False
        took = (time.perf_counter() - started) * 1000
        # The authorization SETTLED with this reply, refused or filled: whatever the
        # process holds is the driver's to see now, and a refused one is replaced.
        lane.settle(slot)
        if not reply.get("ok"):
            self.monitor.end(subject, str(reply.get("code")))
            self.boot.step(
                "load",
                "refused",
                took,
                component=first.components[0] if first.components else "",
                code=str(reply.get("code")),
            )
            detail = str(reply.get("detail"))[:1024]
            shortfall = reply.get("shortfall") or {}
            if shortfall:
                detail = (
                    f"{shortfall['resource']} shortfall at {shortfall['scope']}: "
                    f"needed {shortfall['needed_bytes']} B, "
                    f"{shortfall['available_bytes']} B free (measured) - {detail}"
                )[:1024]
            self.engine.fault(
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    subject=first.entrypoint_binding_digest,
                    reason=str(reply.get("code")),
                    # SANITIZED, like every other detail that crosses this wire (#574b).
                    # The document profile is printable ASCII; refusal prose in this
                    # codebase is written with em dashes, and one of them in a Fault
                    # made EVERY later claim ack raise `non_ascii_field` — the worker
                    # became unclaimable over a refusal it could not spell.
                    detail=safe(detail),
                )
            )
            # AN ACTIVATION FAILURE LATCHES for this desired revision (§8): the worker
            # does not evict-and-retry forever, `converged_revision` stays behind, and
            # the fault is typed on the placement. PREPARE IS PER-BINDING (#572d): the
            # fault is typed on the bindings that earned it and its entrypoints refuse
            # naming it (#548d's refusal-vs-fault split applied to bindings).
            for member in group:
                tenant.failed_bindings[member.entrypoint_binding_digest] = (
                    f"{reply.get('code')}: {detail}"
                )
            # #567c's SECOND HALF: a refusal reduced to its class name is a refusal
            # that does not exist. The detail leads; the memory state follows it, because
            # on a genuine shortfall those numbers are the answer and on everything else
            # they are context.
            self.note(
                "boot",
                f"{placement_id!r} prepare REFUSED for {first.entrypoint_binding_digest[:20]}: "
                f"{reply.get('code')} - {detail} "
                f"[peak allocator {reply.get('peak_allocator_bytes', 0)} B was LIVE when "
                f"it failed; the allocator holds {reply.get('allocator_bytes', -1)} B and "
                f"the device has {reply.get('device_free_bytes', -1)} B free. The executor "
                "is poisoned and replaced; the card returns with the process (#613)]",
            )
            # A failed construction owns no retry policy. Retire its possibly retained
            # allocations and remain offline; convergence decides whether to try again.
            try:
                tenant.supervision.retire_current(executor, f"prepare refused: {reply.get('code')}")
            except ExecutorGone as exc:
                self.note("recovery", f"prepare refusal reclaim failed: {exc}")
            return "prepare refused: every binding", False
        delivery = reply.get("delivery") or {}
        resolved = PreparedModel(
            constructed_model_digest=str(reply["constructed_model_digest"]),
            delivery_rung=str(delivery.get("rung", "")),
            delivery_variant=str(delivery.get("variant", "")),
            weight_bytes=int(reply["facts"]["filled_bytes"]),
            attention=dict(reply["facts"].get("attention") or {}),
        )
        executor.construction_key = construction_key
        executor.parameters = parameters
        executor.prepared_model = resolved
        for binding in group:
            executor.ready_bindings.add(binding.entrypoint_binding_digest)
            tenant.prepared_models[binding.entrypoint_binding_digest] = resolved
        row = tenant.row()
        if row.ledger.executor_pid != executor.pid:
            row.ledger.begin_generation(executor.pid)
        row.ledger.observe_construction(reply["facts"])
        self.monitor.end(subject, "filled")
        self.boot.step(
            "load",
            "filled",
            took,
            component=first.entrypoint,
            destinations=int(reply["facts"]["filled"]),
            bytes=int(reply["facts"]["filled_bytes"]),
            fill_ms=float(reply["facts"]["fill_ms"]),
            warm_ms=float(reply["facts"].get("warm_ms", 0)),
            custody=str(reply["facts"].get("custody", "canonical")),
            ignored_extra_keys=", ".join(reply["facts"].get("ignored_extra_keys", ())),
        )
        # THE SEGMENT LEDGER AND THE OPTIMIZER'S VERDICT (cr-102). The prepare's own
        # breakdown and whether the fused execution plan applied are facts an operator
        # needs on the line that reports the time, not values that die in the reply.
        _stages: tuple[tuple[str, float], ...] = reply["facts"].get("stages") or ()
        if _stages:
            # THE EXPENSIVE LEGS AND THEIR PROVENANCE BOTH SURVIVE THE CAP (cr-141 over
            # cr-103). `note` caps a step at 256 characters, and the ledger is emitted in
            # EXECUTION order -- which puts the cheap prologue first and cut the line off
            # at `admission`, hiding `lease_acquire`, `construct_and_fill` and
            # `execution_plan`. Measured on a rented L40, that was the last 4 s of a 7.5 s
            # prepare: the four legs an operator most needs were exactly the four the cap
            # ate. The two suffixes are priced into the budget FIRST rather than appended
            # and hoped for -- "measured" vs "reused" is what tells a warm pod from a cold
            # one, and it is worthless if the cap eats it instead of a 78 ms leg.
            _q = reply["facts"].get("qualification") or {}
            _l = reply["facts"].get("leases") or {}
            tail = (f" [qualification {_q['source']} {_q['key'][:12]}]" if _q else "") + (
                f" [{_l['leases']} lease(s) over {_l['objects']} object(s) in "
                f"{_l['acquisitions']} acquisition(s)]"
                if _l
                else ""
            )
            legs = [(name, ms) for name, ms in _stages if ms >= 0.5]
            head = f"{placement_id!r} prepare legs: "
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
            # The hidden legs are all cheaper than the cheapest SHOWN one. That bound is
            # what an operator can act on: it says what the omission is worth at most.
            floor = min((ms for name, ms in legs if f"{name} {ms:.0f}ms" in shown), default=0.0)
            self.note(
                "boot",
                head
                + ", ".join(sorted(shown, key=lambda text: order[text.rsplit(" ", 1)[0]]))
                + (f" (+{hidden} under {floor:.0f}ms)" if hidden else "")
                + tail,
            )
        _w: tuple[int, ...] = reply["facts"].get("substrate_census") or ()
        if len(_w) == 3 and _w[1]:
            self.note(
                "boot",
                f"{placement_id!r} WEIGHTLESS CONSTRUCTION DID NOT HOLD: {_w[1]} of "
                f"{_w[0] + _w[1]} parameter(s) were allocated on a real device during "
                f"construction ({_w[2] / 2**30:.3f} GiB) and every one of them is "
                "discarded by `to_empty` — the substrate did not fire",
            )
        _opt = reply["facts"].get("execution_optimization")
        self.note(
            "boot",
            f"{placement_id!r} execution plan: "
            + (
                f"APPLIED {_opt.get('identity', '?')} operations={list(_opt.get('operations', ()))}"
                if _opt
                else "NOT APPLIED — this generation runs the unfused path"
                + (
                    f" ({reply['facts'].get('execution_optimization_refusal')})"
                    if reply["facts"].get("execution_optimization_refusal")
                    else ""
                )
            ),
        )
        _attention = reply["facts"].get("attention")
        if _attention:
            # WHAT THE RUNTIME PINNED (cr-124), ONE LINE, ONCE PER POD. This is the whole
            # per-run attention record and it can be, because selection is a static rule over
            # (device, image): there is no per-attempt CHOICE to audit, and the per-attempt
            # observation this replaces is what faulted an executor on its own telemetry
            # (cr-122). The uncapped map rides `plan.attention` in triage.
            self.note("boot", f"{placement_id!r} attention: {_attention.get('line', '?')}")
        _fused = reply["facts"].get("execution_fusion")
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
            f"{placement_id!r} prepared {len(group)} binding(s) in {took:.0f} ms — "
            f"{reply['facts']['filled']} destinations, "
            f"{reply['facts']['filled_bytes'] / 2**30:.3f} GiB, fill leg "
            f"{reply['facts']['fill_ms'] / 1000:.3f} s, "
            f"generation {str(reply['constructed_model_digest'])[:23]}… "
            f"resident={sorted(reply['facts'].get('resident', {}))} "
            f"parked={reply['facts'].get('parked', [])}; "
            f"fit: {(reply['facts'].get('fit') or {}).get('detail', 'unjudged')}",
        )
        if delivery:
            self.note("delivery", _delivery_observation(delivery))
        if not executor.ready_bindings:
            return "prepare refused: every binding", False
        self._warm_parked_pages(placement_id, first, reply["facts"])
        return "", False

    def _warm_parked_pages(
        self, placement_id: str, first: DeclaredBinding, facts: Mapping[str, Any]
    ) -> None:
        """Read every PARKED component's stored bytes once, in the background (h3a-018).

        A parked component is filled on its first declared use, from the store. Measured on
        an H100 pod, that stage runs at 23-33 GB/s when its pages are cached and at
        0.73 GB/s when they are not: a 40 GB DiT nobody had touched cost 55 s on the first
        request that named it. The read here is the same store stream the fill performs,
        into host buffers that are dropped, so the first stage of each parked component
        finds its pages warm. It moves no device byte, holds no lane lock, and reports what
        it read and whether the block layer served it -- it decides nothing.
        """
        parked = set(facts.get("parked") or ())
        work = [
            (model.store, model.snapshot_for(name), name)
            for model in first.model_bindings()
            for name in model.components
            if name in parked
        ]
        if not work:
            return

        def run() -> None:
            for store, manifest, name in work:
                try:
                    report = page_warm.warm_component(store, manifest, name)
                except Exception as exc:
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

    def _authorize_prepare(self, lane: DeviceLane, slot: str, binding: DeclaredBinding) -> int:
        """ASSIGN the residence ceiling for one prepare from the lane's MEASURED free bytes.

        The worker, not the executor, hands out headroom (cr-025's
        `authorized_device_limit_bytes`; cr-066 assigns it per lane): the lane subtracts what
        it already authorized to executors that have not settled, so the same bytes are
        never promised twice. A weightless binding holds no device memory and is authorized
        nothing. A device the driver cannot read is refused typed (`LaneRefusal`), never
        authorized zero: unreadable is not a number, and a prepare against it would be this
        worker guessing what the card can hold.
        """
        if binding.weightless:
            return 0
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
        # THE MIN OVER THE LANE'S DEVICES (cr-068): every rank of a group holds the full
        # weights, so the ceiling every rank is assigned is what the tightest card has.
        free = memory.free_bytes
        lane.row(slot).ledger.observe_devices(lane.device_facts(measured))
        granted = lane.authorize(slot, free)
        self.note(
            "boot",
            f"{lane.lane_id}: authorized {granted} B for {slot!r} from {free} B measured "
            f"free on {lane.devices!r}"
            + (f" (MIN over {lane.degree} devices)" if lane.group else "")
            + f" ({lane.outstanding_bytes() - granted} B outstanding to other executors)",
        )
        return granted

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

        def fits(memory: accel.DeviceMemory) -> bool:
            row.ledger.observe_device_free(memory.free_bytes, memory.total_bytes)
            return bool(row.chooser.fits(declared))

        # FIRST, TAKE BACK WHAT WAS TAKEN (cr-097). A tenant a co-tenant's prepare emptied
        # is priced, otherwise, against a hole nobody put back — which is how a placement
        # that had 10 GiB of room to itself spent seven consecutive attempts at +118%.
        # This evicts nothing, so it runs before the arbitration below rather than instead
        # of it: what it cannot restore, `_make_room` is still free to make room for.
        self._restore_residency(lane, tenant, row, why)
        # THEN, THE RETRY RULE. A retry under the residency picture that already produced a
        # device OOM is the same request asked twice and cannot get a different answer.
        self._break_oom_repeat(lane, tenant, row, fits, why)
        self._make_room(lane, tenant, fits, why)

    def _restore_residency(self, lane: DeviceLane, slot: str, row: LaneRow, why: str) -> None:
        """RE-FILL a tenant the lane emptied, when the card has room. Evicts NOBODY (cr-097).

        `vacate` had no counterpart, and the gap was the defect: a tenant vacated for a
        co-tenant's prepare stayed empty for the life of its generation, re-admitting per
        scope on every attempt at a cost the device had room to stop paying. Measured live
        on an A4000: 8739 ms fully resident, 18909 ms after the vacate, 9056 ms once an
        unrelated rebuild happened to re-prepare it — with the co-tenant resident and
        untouched throughout, which is the proof the room was there all along.

        It is governed by THE SAME RULE PREPARE USES and by no other. An earlier cut gated
        each component on the attempt's banked activation envelope; that refused `unet` by
        1.1 GiB on every attempt and left the tenant at 2x its solo handler time — because
        the envelope was measured on an empty card, and kernels expand into whatever
        headroom they are given. Prepare and restore are the same physical act, putting a
        generation's weights on a card, and a private stricter test for one of them was not
        a safety measure but the very asymmetry this issue is about.
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
        fits: Callable[[accel.DeviceMemory], bool],
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

        So when this tenant's last attempt died on the device and the lane looks exactly as
        it did then, the co-tenants vacate — LRU-first, the ordinary arbitration, just with
        a demand the chooser's own `fits` is not allowed to talk it out of. If nothing can
        be freed, the picture cannot change and the attempt refuses TYPED and quantified
        instead of spending the handler to rediscover the allocator.
        """
        if not row.oom_signature or row.oom_signature != lane.residency_signature():
            return
        self.note(
            "residency",
            f"{lane.lane_id}: {slot!r} last died on the device under this exact residency "
            f"({row.oom_signature}); making room before {why} rather than repeating it",
        )
        # `fits` deliberately ignored: it is what admitted the attempt that OOMed.
        self._make_room(lane, slot, lambda _memory: False, f"{why} after a device OOM")
        if row.oom_signature != lane.residency_signature():
            row.oom_signature = ""
            return
        kind = accel.host_backend_family()
        memory = lane.memory(kind)
        floor = max(row.ledger.carried_shortfall_by_cell.values(), default=0)
        raise AttemptRefusal(
            "device_shortfall",
            f"an attempt for this placement already failed on the device with "
            f"{floor} B free on {lane.lane_id}, no co-tenant holds bytes that could be "
            f"given back, and the lane is unchanged ({row.oom_signature}). Retrying would "
            f"run the same request against the same device and reach the same allocator; "
            f"the driver reports {memory.free_bytes} B free",
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
        """RESIDENCY ARBITRATION on one lane: vacate co-tenants, LRU-first, until `fits`.

        `fits` reads the lane as the driver measures it NOW (cr-066's NVML reading, the
        same meter ceilings are assigned from) and says whether `slot`'s demand holds. It is
        asked before anything moves and again after each vacate. A lane the driver cannot
        read is left alone: the prepare or the plan refuses typed on the same fact. A
        victim's process stays — only its residency changes, so its executor epoch and its
        placement's serving state do not move; its next attempt re-stages on demand.
        """
        kind = accel.host_backend_family()
        memory = lane.memory(kind)
        if memory.state != "measured" or fits(memory):
            return
        for victim, row in lane.victims(slot):
            executor = row.executor()
            if executor is None:
                continue
            before = memory.free_bytes
            freed, took = self._vacate(victim, row, executor, why)
            memory = lane.memory(kind)
            self.note(
                "residency",
                f"{lane.lane_id}: {victim!r} (epoch {executor.epoch} pid {executor.pid}) "
                f"vacated {freed} B in {took:.0f} ms for {slot!r} on {why}; the driver "
                f"reports {before} -> {memory.free_bytes} B free",
            )
            if memory.state != "measured" or fits(memory):
                return
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

    def on_offer(self, offered: pb.AttemptOffer) -> None:
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
        if self.accepted.mode == "serving":
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
        if (
            self.workspace is not None
            and "job" in attempt.spec
            and attempt.state not in ("outcome", "closed")
        ):
            descriptor = (attempt.spec.get("job") or {}).get("job_descriptor_id", "")
            # Execution attaches attempt.job later. The prepared engine binding
            # is available on both local and placement-hosted paths at offer time.
            binding = self.engine.jobs.get(descriptor)
            if binding is None:
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
            result_schema = canonical.write(declaration["result"])
            memoize = declaration.get("invocable", {}).get("memoize", False) is True
            replay = self.workspace.accept(
                self.fence.record_owner_id,
                offered,
                result_schema=result_schema,
                worker_boot=self.fence.worker_boot_id,
                memoize=memoize,
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
                "duplicate AttemptOffer for the attempt currently crossing to the device "
                "lane: it is held, and its acceptance publishes when the lane admits it",
            )
            return
        if attempt.state in ("accepted", "running", "reclaiming", "finalizing", "released"):
            self.note(
                "attempt",
                "duplicate AttemptOffer for a held attempt: re-accepting "
                "the SAME acceptance; nothing starts a second time",
            )
            self.send(pb.WorkerFrame(attempt_accepted=self.engine.accepted_message(attempt)))
            return
        # THE SEAT IS THE PLACEMENT'S LANE'S, AND OVER IT THE ANSWER IS A JOURNALED
        # OUTCOME (#494f/§7). A lane is a single serialized state machine, so a second held
        # attempt on it could only wait — and it would wait in a state the wire cannot
        # spell, absent from the snapshot and from the held-attempt set. A refusal the
        # RecordOwner can read and re-dispatch is strictly more informative than a hold it
        # cannot see. An attempt for placement P may take P's lane's seat and no other.
        lane = self._offer_lane(offered.placement_id, attempt)
        if self.admission_state() != pb.AdmissionState.ADMISSION_STATE_OPEN or (
            self._lane_slots(lane) < 1
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
                f"{lane.lane_id} admits {LANE_SEATS} attempt to its device at a time and "
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
        lane.work.put(attempt)

    def _offer_lane(self, placement_id: str, attempt: AttemptRecord | None = None) -> DeviceLane:
        """The lane an offer draws its seat from: the placement's, or the envelope for a job.
        Called after the placement is known to be hosted, so an unassigned placement here
        is one that is not dispatchable, and its lane is whichever its slot would take."""
        if self.accepted.mode == "job":
            if attempt is not None:
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
                if lane is self.lanes.orchestration:
                    self._rebuild_orchestration()
                else:
                    self.rebuild_exited(item.executor, item.placement_id)
                continue
            attempt = item
            self.await_settled(attempt)
            # THE DEVICE IS HELD from admission to DEVICE RELEASE (cr-022, cr-079): a
            # co-tenant's prepare on this lane waits here rather than vacating an executor
            # whose attempt is on the card. The post phase runs on `post_lane` and never
            # takes this lock: at release the attempt's bytes have left the executor.
            with lane.device:
                if self.admit(attempt):
                    self.execute(attempt, lane)
            # Rebuild on REFUSALS too (#458): a poisoned generation refuses every attempt,
            # so a rebuild gated on admission could never run again.
            if lane is self.lanes.orchestration:
                self._rebuild_orchestration()
            else:
                self.rebuild_if_needed(attempt.placement_id)

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
        """FETCH -> HYDRATE -> CHOOSE -> ACCEPT, or a typed terminal. Never a silence."""
        key = f"{attempt.request_id}#{attempt.attempt}"
        # Desired-state convergence and executor rebuilds take this same lock. Recheck the
        # offer's immutable routing facts under it, then hold it through the acceptance
        # A queued offer cannot be accepted on the far side of a placement or
        # admission-epoch change merely because it passed `on_offer()` earlier.
        with self.control_lock:
            if attempt.canceling:
                self.note("cancel", f"{key} was cancelled before admission")
                self.terminate(attempt, self.engine.finish_cancel, attempt.canceling)
                return False
            deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
            if deadline_ms and time.time() * 1000 > deadline_ms:
                self.note("cancel", f"{key} reached the lane past its deadline")
                self.terminate(attempt, self.engine.finish_cancel, "deadline")
                return False
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
                return False
            if self.accepted.mode == "serving":
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
                    return False
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
                    return False
            if self.accepted.mode != "serving" and not self.dispatchable():
                self.note("admission", f"refusing queued {key}: worker is not dispatchable")
                self.terminate(
                    attempt,
                    self.engine.refuse_held,
                    "placement_not_dispatchable",
                    "the worker stopped being dispatchable while this offer was queued",
                    CAUSE.CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE,
                )
                return False
            if self.accepted.posture == pb.Posture.POSTURE_DRAINING:
                self.note("intake", f"draining: refusing queued {key}")
                self.terminate(
                    attempt,
                    self.engine.refuse_held,
                    "draining",
                    "the worker began draining while this attempt was queued",
                    CAUSE.CAUSE_CODE_LOCAL_SAFETY,
                )
                return False
            try:
                slot = self._bind_attempt_slot(attempt)
                self._arbitrate_residency(attempt, slot)
                accepted = self.engine.prepare_and_accept(attempt)
            except AttemptRefusal as exc:
                self.note("attempt", f"{key} refused: {exc.code}")
                self.terminate(
                    attempt, self.engine.refuse_held, exc.code, exc.detail, exc.cause, exc.origin
                )
                return False
        self.send(pb.WorkerFrame(attempt_accepted=accepted))
        return True

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
            # A serve got through under this picture, so nothing about it is a trap.
            row.oom_signature = ""
            return
        if str(outcome.get("code", "")) in DEVICE_SHORTFALL_CODES:
            row.oom_signature = lane.residency_signature()

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
        if not rebuild:
            return
        if hosted is not None:
            self._prepare_hosted_executor(hosted)
        elif current is None or not current.alive():
            self._rebuild("the executor is gone")
        else:
            self._rebuild(f"poisoned: {current.poisoned}")

    def rebuild_exited(self, executor: child.Executor, placement_id: str = "") -> None:
        """Rebuild only if the queued death still names the owned dead epoch."""

        with self.control_lock:
            hosted = self.hosted.get(placement_id)
            if hosted is not None:
                if hosted.supervision.current is not executor or executor.alive():
                    return
            elif self.supervision.current is not executor or executor.alive():
                return
        if hosted is not None:
            self._prepare_hosted_executor(hosted)
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
            self.activate_executor()
        self.note(
            "recovery",
            f"dispatchable again in {(time.perf_counter() - started) * 1000:.0f} ms at "
            f"executor epoch {self.supervision.epoch} / admission epoch "
            f"{self.admission_epoch}",
        )

    def watchdog(self) -> None:
        """The deadline and the grace clock — both the WORKER's, never author code's."""
        while not self.stop.wait(0.05):
            for attempt in list(self.engine.live.values()):
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
                        attempt.cancel_started = time.perf_counter()
                        deadline_expired = True
                if deadline_expired:
                    self.note("cancel", f"deadline expired for {attempt.request_id}")
                if attempt.canceling:
                    started = getattr(attempt, "cancel_started", 0.0) or time.perf_counter()
                    attempt.cancel_started = started
                    if time.perf_counter() - started > self.options.grace:
                        self.note(
                            "cancel",
                            f"grace of {self.options.grace:g} s expired: escalating to a "
                            "forceful kill (the handler never asked whether it was cancelled)",
                        )
                        try:
                            facts = self.engine.escalate(attempt, self.options.grace)
                        except Exception as exc:
                            # A failed reclaim is NOT a cancellation result. Keep the
                            # executor owned, withdraw machine admission, and leave this
                            # watchdog as the retrying owner. A one-shot exception here used
                            # to kill the watchdog thread and strand the executor forever.
                            self.note(
                                "cancel",
                                f"forceful reclaim failed; retrying after the grace window: "
                                f"{type(exc).__name__}: {exc}",
                            )
                            if attempt.placement_id in self.hosted:
                                self._hosted_capacity_dropped(
                                    attempt.placement_id, "executor reclaim is not proved"
                                )
                            else:
                                self.capacity_dropped("executor reclaim is not proved")
                            attempt.cancel_started = time.perf_counter()
                            continue
                        self.note("cancel", f"forceful: {json.dumps(facts)}")
                        attempt.canceling = attempt.canceling or "client"
                        attempt.cancel_started = time.perf_counter() + 1e9

    def reporter(self) -> None:
        """The ObservedWorkerState tick, and the liveness SAMPLE that rides it."""
        while not self.stop.wait(REPORT_SECONDS):
            for subject in list(self.monitor.cursors):
                self.monitor.sample(subject)
            self.publish_activity()
            if self.connected.is_set():
                self.send(pb.WorkerFrame(observed_state=self.observed_state()))

    def publish_activity(self) -> None:
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
                meter=activity.meter(self),
            )
        except OSError as exc:
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
            status.environment_digest = (
                documents.spell(placement.environment_digest)
                if placement.environment_digest
                else canonical.digest({})
            )
            if re.fullmatch(r"sha256:[0-9a-f]{64}", status.environment_digest) is None:
                raise RuntimeError("dispatchable placement has incomplete invocation identity")
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
            for lane in self.lanes:
                threading.Thread(
                    target=self._lane(lane.lane_id, self.device_lane, lane),
                    daemon=True,
                    name=f"device-{lane.lane_id}",
                ).start()
                threading.Thread(
                    target=self._lane(f"post-{lane.lane_id}", self.post_lane, lane),
                    daemon=True,
                    name=f"post-{lane.lane_id}",
                ).start()
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
        if (cpu := self.job_slots.get("orchestration")) is not None:
            supervisions.append(cpu.supervision)
        seam = [
            supervision.current.channel.accounting()
            for supervision in supervisions
            if supervision.current is not None
        ]
        self.request_stop()
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
    parser.add_argument("--development-source-digest", default="", help=argparse.SUPPRESS)
    parser.add_argument("--alloc-conf", default="expandable_segments:True")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--grace", type=float, default=GRACE_SECONDS)
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
        args.development_source_digest,
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
        or HASH_ENTRY.fullmatch(args.development_source_digest) is None
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
        development_source_digest=args.development_source_digest,
        alloc_conf=args.alloc_conf,
        threads=args.threads,
        grace=args.grace,
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
        host.compatibility_checker = worker.check_package_set_compatibility
        host.preparer = worker.prepare_package_set
        host.local_package_preparer = worker.prepare_local_package
        host.private_placement_preparer = worker.prepare_private_placement
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
