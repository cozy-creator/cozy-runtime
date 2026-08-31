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
`(record_owner_epoch, control_stream_generation, worker_boot_id)`, checked on every frame
before any body field is read. LAUNCH TIER (single-owner): the epoch is a provisioned
constant — a claim with a strictly higher epoch supersedes, an equal epoch from the same
RecordOwner is a reconnect, anything else refuses typed.

Threads, and why each exists: each accepted STREAM runs its conversation on the host's
thread and must never do slow work (a cancel has to arrive while an attempt runs); the
DEVICE LANE holds the single device lease and serializes attempts in arrival order — the
whole of worker-local arbitration; the REPORTER ticks the durable Report; the WATCHDOG owns
the deadline and the grace clock, because the deadline is worker-enforced and never merely
handed to author code.

This process never imports torch. `scripts/worker-live.py nocuda` proves that against the
RUNNING process's loaded modules and mapped libraries, not against the import graph.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import hmac
import itertools
import json
import os
import queue
import re
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cozy_runtime.internal import (
    accel,
    canonical,
    child_env,
    hostfacts,
    overlay,
    proctree,
)
from cozy_runtime.internal.config import ConfigError, RuntimeConfig, read_config
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.worker import (
    acquire,
    artifacts,
    child,
    downloads,
    grants,
    model_source_prepare,
    observe,
    triage,
)
from cozy_runtime.internal.worker.attempts import (
    CAUSE,
    ORIGIN,
    STATUS,
    AttemptEngine,
    AttemptRecord,
    AttemptRefusal,
    safe,
)
from cozy_runtime.internal.worker.child import (
    GRACE_SECONDS,
    ExecutorGone,
    ExecutorSupervision,
    socket_path_refusal,
)
from cozy_runtime.internal.worker.control import (
    ControlHost,
    GrpcControlHost,
    WatchFanout,
)
from cozy_runtime.internal.worker.ledger import Ledger, read_vmrss
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    JobBinding,
    PlanChooser,
    PreparedModel,
)
from cozy_runtime.internal.worker.records import WorkerRecords
from cozy_runtime.protocol import WIRE_MINOR, documents
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

#: The worker's accepted-attempt WINDOW — the ONE shared acceptance window behind
#: `available_attempt_slots` (#472e/#494f). ONE.
#:
#: It was 2, on the claim that attempt N+1's CPU phase overlaps attempt N's device phase.
#: That overlap does not exist: `device_lane()` is a single thread that admits, executes and
#: reconciles one attempt at a time, so a second admitted attempt could only WAIT — and it
#: waited in a state (`queued`) the wire cannot spell, invisible to the snapshot and to the
#: held-attempt set. A worker holding work the RecordOwner cannot see breaks the
#: snapshot-is-complete law, so the false state is DELETED rather than surfaced: an offer
#: arriving while this worker already holds one is a JOURNALED REFUSED outcome the
#: RecordOwner reads, never a silent hold.
#:
#: rev-2 makes the SHAPE of the bound structural rather than advisory (§6): per-placement
#: `attempt_credits` are gone — N counters over ONE serialized device advertised N x the real
#: capacity — and the seats here are consumed by running attempts AND by outcomes awaiting
#: `AttemptOutcomeAck`, so a RecordOwner that stops acking starves its own admission instead
#: of growing this worker's held set.
ADMISSION_WINDOW = 1


def _progress_document(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the measured progress coordinates the executor already supplied.

    The old worker serialized only ``value`` and silently discarded stage, position, and
    step time. That left clients with a percentage but no honest way to name a denoising
    stage, count its steps, or estimate its remaining wall time.
    """
    kind = frame.get("kind", "progress")
    payload: Any = frame.get("value")
    if kind == "progress":
        payload = {
            "name": str(frame.get("name", ""))[:120],
            "fraction": frame.get("value"),
            "position": frame.get("position"),
            "step_ms": frame.get("step_ms", 0.0),
        }
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
    #: PROVENANCE, and every one of these is EMPTY when the launch did not supply it (#494g).
    #: The defaults used to name a real SDXL release and a `sha256:00…00` artifact, which rode
    #: real ClaimAcks and real BootFailures as if they had been measured. Absent is absent: an
    #: owner reading an empty `release_id` learns the truth, and an owner reading a plausible
    #: lie learns nothing it can act on.
    release_id: str = ""
    control_runtime_digest: str = ""
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
    environment_root: Path | None = None
    """The materializer root whose receipt refs select content-keyed immutable overlays.

    The provisioner/local installer chooses storage; the worker receives that path and owns
    no retention or GC policy.  Absent means no already-materialized generation is selectable.
    """
    base_manifest: Path = overlay.DEFAULT_BASE_MANIFEST
    """Exact base-worker-image WheelhouseManifest path; fixed default in production."""
    artifact_store: Path | None = None
    """The verified TensorFS store populated from the canonical model-root closure."""
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
    host_budget: int = 8 << 30
    pinned_budget: int = 512 << 20
    no_warm: bool = False
    claim_ready: bool = True
    """False only for fixed product-host boot until listener/readiness proof commits."""
    executor_uid: int = -1
    executor_gid: int = -1
    """Fixed package identity, or ``-1/-1`` for the native same-user development lane."""


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
    package_descriptor: tuple[str, int]
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

    @property
    def execution_digest(self) -> str:
        """Published release.json provenance or explicit development source identity."""

        return str(
            self.package.get("release_digest")
            or self.development.get("private_revision_digest")
            or self.development.get("source_digest")
            or ""
        )

    @property
    def config_digest(self) -> str:
        selected = {
            str(slot["config"]["document"]["digest"])
            for entrypoint in self.entrypoints
            for slot in entrypoint.get("slots", [])
        }
        if not selected:
            return canonical.digest({})
        return next(iter(selected)) if len(selected) == 1 else ""


@dataclass(frozen=True, slots=True)
class ExecutorDeath:
    """One exact child exit awaiting the serialized device lane's rebuild decision."""

    executor: child.Executor


@dataclass(slots=True)
class OwnershipFence:
    """WHO may drive this worker, and on WHICH stream (02 §0/§4).

    Split out of the old single `Fences` bag (#484): ownership and accepted intent are two
    unrelated lifetimes — a reconnect mints a new `control_stream_generation` and touches no
    accepted revision at all, and #494a's reconnect defect was exactly that confusion made
    structural. `activity` lives in NEITHER of the two halves; it is the worker's own
    narration lane and now sits on the worker.

    `record_owner_epoch` is authority-granter-minted; `control_stream_generation` is
    WORKER-minted, incremented for every accepted stream; `worker_boot_id` is minted at worker
    boot and never reused. The caller owns recovery across worker processes. The executor
    generation is PLACEMENT truth (#454) and deliberately not here.
    """

    worker_boot_id: str = ""
    record_owner_epoch: int = 0
    record_owner_id: str = ""
    #: TRUE once this process accepted a (record_owner_epoch, record_owner_id) pair.
    record_owner_recorded: bool = False
    control_stream_generation: int = 0  # worker-minted; 0 = never claimed
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
    #: The revision whose placements are all materially STAGED and DISPATCHABLE. It may NEVER
    #: advance past a placement that is not — the field is a DERIVED fact, and a worker that
    #: advances it while observed state disagrees is in breach, not merely optimistic (§8).
    converged_revision: int = 0


#: Every envelope-bearing message type sets the same three fields; the stamp is central so
#: no constructor can forget one (the fence is checked before any body field is read).
def _stamped(message: Any, fence: OwnershipFence) -> Any:
    message.record_owner_epoch = fence.record_owner_epoch
    message.control_stream_generation = fence.control_stream_generation
    message.worker_boot_id = fence.worker_boot_id
    return message


def desired_state_body(desired: pb.DesiredWorkerState) -> bytes:
    """The canonical bytes of a DesiredWorkerState's DESIRED STATE — envelope excluded (#494a).

    A DesiredWorkerState carries two unrelated things: what the RecordOwner wants this worker
    to be (revision, posture, wire minor, drain grace, the mode's set or job), and the
    per-stream envelope that routes it (record_owner_epoch, control_stream_generation,
    worker_boot_id). Only the first is the body §3's same-revision rule is about.

    Hashing the whole message made `control_stream_generation` — which the WORKER mints
    afresh for every accepted stream, by design — part of the body, so the identical desired
    revision re-sent after an ordinary reconnect classified as `same_revision_changed_body`
    and took a permanent CONFIG_REFUSED fault. Same revision + same desired state is now the
    idempotent re-apply it always was; same revision + genuinely different desired state
    stays the protocol refusal, because that is the case the rule exists for.
    """
    body = pb.DesiredWorkerState()
    body.CopyFrom(desired)
    body.record_owner_epoch = 0
    body.control_stream_generation = 0
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
    row = _closed_document(
        value, {"package", "release", "release_digest", "project_wheel"}, "PackageSelection"
    )
    _required(row, {"package", "release", "release_digest", "project_wheel"}, "PackageSelection")
    package = _identifier(row["package"], "PackageSelection.package")
    release = _identifier(row["release"], "PackageSelection.release")
    if package.count("/") != 1 or not all(package.split("/")) or SEMVER.fullmatch(release) is None:
        raise documents.DocumentError("package_invalid", "package identity or release is invalid")
    if (
        not isinstance(row["release_digest"], str)
        or HASH_ENTRY.fullmatch(row["release_digest"]) is None
    ):
        raise documents.DocumentError("package_invalid", "release_digest is malformed")
    project = _wheel_fact(row["project_wheel"], "PackageSelection.project_wheel")
    if project["distribution"] != package.split("/", 1)[1] or project["version"] != release:
        raise documents.DocumentError("project_wheel_mismatch", package)
    return row


def _development_package(value: object) -> dict[str, Any]:
    row = _closed_document(
        value,
        {
            "package",
            "release",
            "source_digest",
            "project_wheel",
            "private_revision_digest",
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
    private_revision = "private_revision_digest" in row
    if remote != private_revision:
        raise documents.DocumentError(
            "development_invalid",
            "project_wheel and private_revision_digest must be present together",
        )
    if private_revision and (
        not isinstance(row["private_revision_digest"], str)
        or HASH_ENTRY.fullmatch(row["private_revision_digest"]) is None
    ):
        raise documents.DocumentError("development_invalid", "private_revision_digest is malformed")
    return row


def _placement_profile(entry: dict[str, Any]) -> None:
    fields = {
        "placement_id",
        "package",
        "development",
        "environment_digest",
        "package_descriptor",
        "bindings_digest",
        "models",
        "entrypoints",
        "environment",
    }
    _closed_document(entry, fields, "Placement")
    _required(
        entry,
        {"placement_id", "package_descriptor", "bindings_digest", "entrypoints"},
        "Placement",
    )
    _identifier(entry["placement_id"], "Placement.placement_id")
    _exact_ref(entry["package_descriptor"], "Placement.package_descriptor")
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
        environment = _closed_document(entry["environment"], {"wheels"}, "Environment")
        wheels = environment.get("wheels", [])
        if not isinstance(wheels, list):
            raise documents.DocumentError("wrong_type", "Environment.wheels")
        wheel_rows = [
            _wheel_fact(row, f"Environment.wheels[{index}]") for index, row in enumerate(wheels)
        ]
        names = [str(row["distribution"]) for row in wheel_rows]
        if names != sorted(set(names)):
            raise documents.DocumentError("sort_order", "Environment.wheels")
        if package["project_wheel"]["distribution"] in names:
            raise documents.DocumentError("project_wheel_mismatch", "project wheel is in overlay")
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
                    "config",
                    "components",
                    "model_construction_contract",
                    "stamps",
                },
                "Slot",
            )
            _required(slot, {"slot", "reference_model_id", "config"}, "Slot")
            if development and "model_construction_contract" in slot:
                raise documents.DocumentError(
                    "development_fields_invalid",
                    "development derives model_construction_contract locally",
                )
            slot_names.append(_identifier(slot["slot"], f"slots[{slot_index}].slot"))
            if slot["reference_model_id"] not in known_models:
                raise documents.DocumentError("unknown_model", "Slot.reference_model_id")
            config = _closed_document(slot["config"], {"document", "assets"}, "Config")
            _required(config, {"document"}, "Config")
            _exact_ref(config["document"], "Config.document")
            assets = config.get("assets", [])
            if not isinstance(assets, list):
                raise documents.DocumentError("wrong_type", "Config.assets")
            asset_names: list[str] = []
            for asset_index, value in enumerate(assets):
                asset = _closed_document(value, {"name", "model_id", "ref"}, "Asset")
                _required(asset, {"name", "model_id", "ref"}, "Asset")
                asset_names.append(_identifier(asset["name"], f"assets[{asset_index}].name"))
                if asset["model_id"] not in known_models:
                    raise documents.DocumentError("unknown_model", "Asset.model_id")
                _exact_ref(asset["ref"], "Asset.ref")
            if asset_names != sorted(set(asset_names)):
                raise documents.DocumentError("sort_order", "Config.assets")
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
            if "model_construction_contract" in slot:
                _exact_ref(slot["model_construction_contract"], "Slot.model_construction_contract")
            stamps = slot.get("stamps", [])
            if not isinstance(stamps, list):
                raise documents.DocumentError("wrong_type", "Slot.stamps")
            stamp_ids: list[tuple[str, str]] = []
            for value in stamps:
                stamp = _closed_document(value, {"component", "key", "values"}, "Stamp")
                _required(stamp, {"component", "key", "values"}, "Stamp")
                component_name = _identifier(stamp["component"], "Stamp.component")
                key = _identifier(stamp["key"], "Stamp.key")
                if component_name not in component_names:
                    raise documents.DocumentError("unknown_component", "Stamp.component")
                if not _sorted_unique_strings(stamp["values"], "Stamp.values"):
                    raise documents.DocumentError("missing_field", "Stamp.values")
                stamp_ids.append((component_name, key))
            if stamp_ids != sorted(set(stamp_ids)):
                raise documents.DocumentError("sort_order", "Slot.stamps")
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


class Worker:
    """The no-CUDA control parent — the machine's WORKER process (#455/#484; the name
    `Supervisor` is retired everywhere, and "supervision" survives only as the verb for its
    executor management)."""

    def __init__(self, config: RuntimeConfig, options: WorkerOptions, host: ControlHost) -> None:
        self.config = config
        self.options = options
        self.host = host
        self.root = Path(options.root)
        self.root.mkdir(parents=True, exist_ok=True)
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
        self.admission_generation = 1
        self.seq = itertools.count(1)  # WORKER-owned monotone counter
        self.progress_seq: dict[tuple[str, int], itertools.count[int]] = {}
        self.records = WorkerRecords()
        self.ledger = Ledger(
            host_budget=options.host_budget,
            pinned_budget=options.pinned_budget,
            worker_pid=os.getpid(),
        )
        self.authorizer = grants.Authorizer(roots=tuple(options.grant_roots))
        self.supervision = ExecutorSupervision(
            root=self.root,
            python=options.python or sys.executable,
            base_env=config.child_base_env,
            cozy_home=config.cozy_home,
            executor_uid=options.executor_uid,
            executor_gid=options.executor_gid,
            jit_pod_scope=self.fence.worker_boot_id,
            scratch_root=self.root.parent / "scratch",
        )
        self.supervision.on_change = self.executor_changed
        # One accepted stream at a time may mutate control state. Claims take this through
        # their snapshot, so ownership cannot move in the middle of convergence and a frame
        # from the superseded stream cannot finish under its successor's generation.
        self.control_lock = threading.RLock()
        self.preparation_lock = threading.Lock()
        self.model_source_operations: set[str] = set()
        # Exact code/descriptor/Environment revisions prepared over the private loopback seam.
        # This is same-process admission state, not a Runtime journal. After worker restart the
        # RecordOwner replays preparation before a model-bound PlacementSet can join it.
        self.private_revisions: dict[tuple[str, str], Placement] = {}
        self.private_operations: dict[tuple[str, str], Placement] = {}
        self.claim_ready = threading.Event()
        if options.claim_ready:
            self.claim_ready.set()
        self.chooser = PlanChooser(self.ledger)
        self.bindings: dict[str, DeclaredBinding] = {}
        #: plan id -> why THIS binding could not be prepared (#572d). A faulted binding
        #: refuses its own entrypoints and denies its siblings nothing.
        self.failed_bindings: dict[str, str] = {}
        self.prepared_models: dict[str, PreparedModel] = {}
        self.jobs: dict[str, JobBinding] = {}
        self.placement: Placement | None = None
        #: The INCOMING placement being staged while `placement` keeps serving (#474). It is
        #: a separate field precisely because one field cannot hold both, which is the same
        #: reason the observed state has two axes.
        self.pending: Placement | None = None
        #: Job mode has no placement, so it carries its own dispatchability edge (cr-009).
        self.job_ready = False
        #: An activation failure LATCHES for the accepted revision (§8): non-empty means the
        #: worker has stopped retrying and `converged_revision` stays behind on purpose.
        self.latched = ""
        self.boot = observe.BootRecord()
        self.monitor = observe.SilenceMonitor(self.boot.ring, busy=self.read_device_work)
        self.bundles = triage.BundleStore(self.root / "triage", self.records)
        self.stop = threading.Event()
        self.artifacts = artifacts.ArtifactExchange(
            store_root=options.artifact_store,
            send=self.send,
            stop=self.stop,
            owner_scope=lambda: self.fence.record_owner_id,
        )
        self.engine = AttemptEngine(
            records=self.records,
            ledger=self.ledger,
            chooser=self.chooser,
            supervision=self.supervision,
            authorizer=self.authorizer,
            bindings={},
            prepared_models={},
            jobs={},
            spool_root=self.root / "spool",
            scratch_root=self.root.parent / "scratch",
            instance_id=options.instance_id,
            send=self.send,
            progress=self.emit_progress,
            bundles=self.bundles,
            monitor=self.monitor,
            posture=self.applied_posture,
            device_process=self.read_device_process,
            artifacts=self.artifacts,
            owner_scope=lambda: self.fence.record_owner_id,
            artifact_store=options.artifact_store,
        )
        self.engine.boot_id = self.fence.worker_boot_id
        #: The durable lane's outbound queue, and it belongs to ONE accepted stream. A new
        #: claim installs a FRESH queue under `claim_lock`, so the superseded stream's sender
        #: keeps draining its own (now unread) queue and can never consume a frame authored
        #: for the live stream. A single process-wide queue made that theft routine: the dead
        #: generation's pump sits in `get(timeout=REPORT_SECONDS)`, and for up to that long
        #: after a reconnect it woke on the NEXT frame — a live AttemptTerminal — saw the
        #: generation had moved, and dropped it. The terminal was recorded, so it replayed on
        #: the following claim and the replay law held; live delivery on the stream that
        #: dispatched it did not.
        self.outbound: queue.Queue[pb.WorkerFrame | None] = queue.Queue()
        self.watches = WatchFanout(PROGRESS_QUEUE)
        self.work: queue.Queue[AttemptRecord | ExecutorDeath | None] = queue.Queue()
        # Slow source/custom-wheel and model-document acquisition never runs on a control
        # stream thread. One lane serializes revisions; the revision check after every slow
        # leg discards a superseded result before it can move serving state.
        self.convergences: queue.Queue[int | None] = queue.Queue()
        self.settled = threading.Event()
        self.supervision.on_invalidate = self.capacity_dropped
        self.supervision.on_exit = self.executor_exited
        self.connected = threading.Event()
        self.exit_code: int = int(Exit.ok)
        self.boot_at = time.perf_counter()
        self.warm: list[dict[str, Any]] = []
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
        #: serializes claim acceptance: one fence, whichever host thread observes the claim
        self.claim_lock = threading.Lock()

    # ------------------------------------------------------------------ plumbing

    def prepare_package_set(
        self, request: pb.PreparePackageSetRequest
    ) -> pb.PreparePackageSetResult:
        return self._prepare_package(request, private=False)

    def prepare_private_package(
        self, request: pb.PreparePrivatePackageRequest
    ) -> pb.PreparePackageSetResult:
        return self._prepare_package(request, private=True)

    def prepare_private_placement(
        self, request: pb.PreparePrivatePlacementRequest
    ) -> pb.PreparePackageSetResult:
        from cozy_runtime.internal.worker import package_prepare

        with self.preparation_lock:
            revision = documents.spell(request.private_revision_digest)
            prepared = self.private_operations.get((request.operation_id, revision))
            if prepared is None:
                raise package_prepare.PreparationRefusal("private_revision_unprepared", revision)

            def mark_verified(digest: str, length: int, path: Path) -> None:
                self.verified_artifacts.record(digest, length, path)

            result = package_prepare.prepare_private_placement(
                request,
                prepared=prepared.document,
                artifact_cache=Path(self.options.artifact_cache or ""),
                artifact_store=(
                    Path(self.options.artifact_store)
                    if self.options.artifact_store is not None
                    else None
                ),
                verified=mark_verified,
            )
            joined = read_placement_set(result.placement_set)
            if (
                len(joined) != 1
                or joined[0].get("development", {}).get("private_revision_digest") != revision
            ):
                raise package_prepare.PreparationRefusal(
                    "private_placement_result_invalid", revision
                )
            return result

    def _prepare_package(
        self,
        request: pb.PreparePackageSetRequest | pb.PreparePrivatePackageRequest,
        *,
        private: bool,
    ) -> pb.PreparePackageSetResult:
        """Derive one local PlacementSet while package imports stay in the isolated executor."""

        from cozy_runtime.internal.worker import package_prepare

        with self.preparation_lock:
            if self.placement is not None or self.pending is not None or self.bindings:
                raise package_prepare.PreparationRefusal(
                    "package_prepare_worker_busy",
                    "replace the active package through desired state",
                )

            def describe(installed: overlay.InstalledEnvironment, distribution: str) -> bytes:
                changed = self.supervision.use_environment(
                    str(installed.python), installed.content_digest
                )
                executor = self.supervision.current
                try:
                    if executor is None:
                        executor = self.supervision.spawn(imposed=self.imposed())
                    elif changed or not executor.alive():
                        executor = self.supervision.replace(
                            executor, imposed=self.imposed(), why="prepared package changed"
                        )
                    scratch = self.supervision.generation_scratch(executor.incarnation)
                    output = scratch / f"package-descriptor-{uuid.uuid4().hex}.json"
                    reply = executor.call(
                        {
                            "cmd": "describe_installed",
                            "distribution": distribution,
                            "output": str(output),
                        },
                        timeout=1.0,
                        total_timeout=65.0,
                    )
                except ExecutorGone as exc:
                    raise package_prepare.PreparationRefusal(
                        "package_prepare_executor_failed", exc.detail
                    ) from exc
                try:
                    raw = output.read_bytes()
                finally:
                    output.unlink(missing_ok=True)
                digest = "sha256:" + hashlib.sha256(raw).hexdigest()
                if (
                    not reply.get("ok")
                    or reply.get("digest") != digest
                    or reply.get("length") != len(raw)
                ):
                    raise package_prepare.PreparationRefusal(
                        "package_prepare_descriptor_failed", str(reply.get("code", "invalid reply"))
                    )
                return raw

            def mark_verified(digest: str, length: int, path: Path) -> None:
                self.verified_artifacts.record(digest, length, path)

            if private:
                assert isinstance(request, pb.PreparePrivatePackageRequest)
                result = package_prepare.prepare_private_package(
                    request,
                    artifact_cache=Path(self.options.artifact_cache or ""),
                    environment_root=Path(self.options.environment_root or ""),
                    python=Path(self.options.python or sys.executable),
                    describe=describe,
                    verified=mark_verified,
                    job_plan_root=self.config.cozy_home / "job-plans",
                )
                prepared = read_placement_set(result.placement_set)
                if len(prepared) != 1 or not prepared[0].get("development"):
                    raise package_prepare.PreparationRefusal(
                        "private_package_result_invalid",
                        "private preparation did not return one development Placement",
                    )
                entry = prepared[0]
                package_descriptor = dict(entry.get("package_descriptor", {}))
                stored = Placement(
                    placement_id=str(entry.get("placement_id", "")),
                    placement_set_digest=result.placement_set.placement_set_digest,
                    package=dict(entry.get("package", {})),
                    development=dict(entry.get("development", {})),
                    environment_digest=_digest_field(entry.get("environment_digest")),
                    package_descriptor=(
                        str(package_descriptor.get("digest", "")),
                        int(package_descriptor.get("length", 0)),
                    ),
                    bindings_digest=_digest_field(entry.get("bindings_digest")),
                    models=tuple(dict(row) for row in entry.get("models", [])),
                    entrypoints=tuple(dict(row) for row in entry.get("entrypoints", [])),
                    environment=dict(entry.get("environment", {})),
                    document=dict(entry),
                )
                revision = stored.execution_digest
                if revision != documents.spell(request.package.private_revision_digest):
                    raise package_prepare.PreparationRefusal(
                        "private_package_revision_mismatch",
                        "prepared Placement changed the declared private revision",
                    )
                self.private_operations[(request.operation_id, revision)] = stored
                self.private_revisions[(stored.placement_id, revision)] = stored
                return result
            assert isinstance(request, pb.PreparePackageSetRequest)
            return package_prepare.prepare_package_set(
                request,
                artifact_cache=Path(self.options.artifact_cache or ""),
                artifact_store=Path(self.options.artifact_store or ""),
                environment_root=Path(self.options.environment_root or ""),
                python=Path(self.options.python or sys.executable),
                describe=describe,
                verified=mark_verified,
                job_plan_root=self.config.cozy_home / "job-plans",
            )

    def prepare_model_source(
        self, request: pb.PrepareModelSourceRequest
    ) -> pb.PrepareModelSourceResult:
        """Hand exact supervisor-verified source facts to the shared TensorFS core."""

        with self.preparation_lock:
            result = model_source_prepare.prepare_model_source(
                request, artifact_store=Path(self.options.artifact_store or "")
            )
            if result.outcome in {
                pb.MODEL_SOURCE_PREPARE_OUTCOME_PREPARED,
                pb.MODEL_SOURCE_PREPARE_OUTCOME_REPLAYED,
            }:
                self.model_source_operations.add(request.operation_id)
            return result

    def release_model_source(self, operation_id: str) -> None:
        """Explicit cooperative cleanup; pod teardown remains the remote durability boundary."""

        with self.preparation_lock:
            model_source_prepare.release_model_source(
                artifact_store=Path(self.options.artifact_store or ""),
                operation_id=operation_id,
            )
            self.model_source_operations.discard(operation_id)

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
        envelope a frame carries and the stream it is queued on are then the same generation
        by construction, with no window where a frame is stamped for one stream and handed to
        another's sender.
        """
        inner = getattr(frame, frame.WhichOneof("msg"))
        with self.claim_lock:
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

        ADMISSION MEANING MOVES WITH IT, so `admission_generation` bumps here (§6): an offer
        minted against the old generation refuses deterministically rather than racing a
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

    def _settle(self) -> None:
        """`settled` is the device lane's gate: has this worker STOPPED moving? A latched
        activation failure settles too — the lane must refuse through it, not wait forever."""
        if self.dispatchable() or self.latched:
            self.settled.set()
        else:
            self.settled.clear()
        if self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))

    def bump_admission(self, why: str) -> None:
        """The ONE worker-level admission fence moves (§6). Callers: executor respawn,
        window resize, phase change, cutover, and every serving-axis edge above."""
        self.admission_generation += 1
        self.note("admission", f"generation {self.admission_generation}: {why}"[:256])

    def dispatchable(self) -> bool:
        """Can this worker take an offer AT ALL right now? Serving asks the placement's
        axis; job mode asks whether its one build is ready. Neither answer is a SLOT count —
        `available_attempt_slots` is the other half, and #486c keeps them apart on purpose."""
        if self.accepted.mode == "job":
            return self.job_ready
        return (
            self.placement is not None
            and self.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
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
        """Free seats in the ONE shared acceptance window, NOW.

        BOUNDEDNESS IS STRUCTURAL, NOT A PROMISE (§5): a seat is held by a running attempt
        AND by an outcome awaiting `AttemptOutcomeAck`, so a RecordOwner that stops acking
        runs itself out of admission instead of growing this worker's held set.
        """
        if self.admission_state() != pb.AdmissionState.ADMISSION_STATE_OPEN:
            return 0
        held = sum(1 for a in self.engine.history.values() if a.state in self.engine.HELD)
        return max(0, ADMISSION_WINDOW - held)

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

    def capacity_dropped(self, why: str) -> None:
        """The executor's state was invalidated: STOP ADVERTISING DISPATCHABLE, right now."""
        if not self.dispatchable():
            return
        self.note("serving", f"capacity withdrawn: {why[:200]}")
        if self.accepted.mode == "job":
            self.set_job_ready(False)
        else:
            self.set_serving(pb.ServingState.SERVING_STATE_ACTIVATING)

    def executor_exited(self, executor: child.Executor) -> None:
        """Withdraw an exact dead incarnation and ask the device lane to rebuild it.

        The pidfd watcher does not reap.  Invalidation clears the ready-plan set and moves
        admission before this callback returns; replacement remains queued behind whichever
        attempt already owns the serialized lane.  Both transitions target ``executor``, so
        a watcher for E1 cannot invalidate or rebuild E2.
        """

        with self.control_lock:
            rebuild = self.dispatchable()
            if not self.supervision.invalidate(
                executor,
                f"executor incarnation {executor.incarnation} pid {executor.pid} exited",
            ):
                return
            if rebuild and not self.stop.is_set():
                self.work.put(ExecutorDeath(executor))

    def executor_changed(self, executor: child.Executor | None) -> None:
        """Move the ledger at the same boundary that publishes executor ownership.

        A job generation has no construction report, so resetting only in the serving fill
        path lets it inherit the previous serving process's baseline and residency. The
        supervision callback covers every spawn and retirement before either can be
        observed through the session.
        """
        self.ledger.begin_generation(executor.pid if executor is not None else 0)

    def read_device_process(self) -> int:
        """NVML bytes for the current executor; -1 remains UNREADABLE/ABSENT here.

        The ledger's legacy scalar cannot express absence separately. Lifecycle reclaim uses
        AcceleratorOps' tri-state observation directly, where that distinction is authoritative.
        """
        executor = self.supervision.current
        if executor is None:
            return -1
        observed = accel.process_memory(executor.pid, accel.host_backend_family())
        return observed.bytes if observed.state == "present" else -1

    def read_device_work(self) -> int:
        """How much work the EXECUTOR has done, read by the parent — THE meter."""
        executor = self.supervision.current
        return child.work(executor.pid if executor is not None else 0)

    def applied_posture(self) -> dict[str, Any]:
        """The state the worker was in when something happened (§3.6). `readiness_epoch` is
        NOT here: it is DELETED, not renamed (#486b) — the serving axis says what the
        placement can serve and `admission_generation` says whether an offer is admissible,
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
            "control_stream_generation": self.fence.control_stream_generation,
            "applied_wire_minor": self.fence.wire_minor,
            "admission_generation": self.admission_generation,
            "executor_generation": self.supervision.incarnation,
            "placement_id": self.placement.placement_id if self.placement else "",
            "worker_id": self.options.worker_id,
            "worker_release_id": self.options.release_id,
            "boot_steps": list(self.boot.steps),
        }

    def emit_progress(self, request_id: str, attempt: int, frame: dict[str, Any]) -> None:
        """The LOSSY lane. Bounded per watch, oldest-first shed, worker-owned sequence."""
        key = (request_id, attempt)
        position = frame.get("position")
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
        previous stream by minting the next `control_stream_generation`, so the superseded
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
            generation, outbound = self.accept_claim(claim, send)
            if generation == 0:
                return  # refused; the rejection already went out and the stream ends
            if self.boot_fatal is not None:
                # The boot-fatal verdict INSTEAD of ClaimAck (01, #494c): the claim was
                # AUTHENTICATED and FENCED above — that is what makes the verdict deliverable
                # to this dialer — and then the worker says the one true thing it has to say.
                send(self.stamp(pb.WorkerFrame(boot_failure=self.boot_fatal)))
                self.note("boot", "boot failure delivered INSTEAD of ClaimAck; stopping")
                self.shutdown_from_stream()
                return
            self._send_claim_ack(claim, generation, send)

        sender = threading.Thread(
            target=self._pump_outbound,
            args=(generation, send, outbound),
            daemon=True,
            name=f"control-send-{generation}",
        )
        sender.start()
        try:
            for frame in inbound:
                if self.fence.control_stream_generation != generation or self.stop.is_set():
                    self.note("fence", "a frame arrived on a superseded stream; ending it")
                    return
                self._dispatch_frame(frame, generation)
        except Exception as exc:
            # THE MESSAGE, not the class name. A typed refusal carries the one sentence that
            # says what to do about it, and reducing it to `PlanRefusal` produced a log that
            # repeated forever and named nothing — the refusal had to be reproduced by hand
            # off the record to find out which key it was about.
            self.note("session", f"the control stream ended: {type(exc).__name__}: {exc}")
        finally:
            if self.fence.control_stream_generation == generation:
                self.connected.clear()
                self.fence.dispatch_open = False

    def accept_claim(
        self, claim: pb.Claim, send: Callable[[pb.WorkerFrame], None]
    ) -> tuple[int, queue.Queue[pb.WorkerFrame | None]]:
        """The ownership fence (02 §4). AUTHENTICATE, fence, RECORD the authority durably,
        mint the generation. Returns `(generation, this stream's outbound queue)`, or
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
            ack.control_stream_generation = self.fence.control_stream_generation
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
        if not self.claim_ready.is_set():
            return reject(
                pb.ClaimRejection.CLAIM_REJECTION_UNDURABLE,
                "the worker listener is bound but the readiness barrier is not committed",
            )
        if claim.worker_id and claim.worker_id != self.options.worker_id:
            return reject(
                pb.ClaimRejection.CLAIM_REJECTION_WORKER_ID_MISMATCH,
                f"this worker is {self.options.worker_id!r}",
            )
        with self.claim_lock:
            # `record_owner_recorded`, not the generation: the high-water authority this worker
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
            # RECORD BEFORE ACKNOWLEDGING so a reconnect to this process cannot race the
            # ownership fence. The caller owns recovery across worker processes.
            self.record_ownership(claim)
            superseded = self.fence.control_stream_generation
            self.fence.record_owner_epoch = claim.record_owner_epoch
            self.fence.record_owner_id = claim.record_owner_id
            self.fence.record_owner_recorded = True
            self.fence.control_stream_generation += 1
            self.fence.dispatch_open = False
            self.fence.wire_minor = min(WIRE_MINOR, claim.wire_minor)
            generation = self.fence.control_stream_generation
            # A FRESH queue for this stream. The superseded sender keeps its own and drains
            # into a dead call; nothing it still holds was ever this stream's to deliver, and
            # everything the law owes regenerates — unacked terminals replay after
            # SnapshotAck, reports re-tick.
            outbound: queue.Queue[pb.WorkerFrame | None] = queue.Queue()
            self.outbound = outbound
        if superseded:
            self.watches.end_generation(superseded)
            self.note(
                "claim",
                f"stream generation {superseded} fenced; generation {generation} claims "
                f"under epoch {claim.record_owner_epoch}",
            )
        return generation, outbound

    def mark_claim_ready(self) -> None:
        """Open Claim only after fixed-host listener proof and readiness publication."""

        self.claim_ready.set()

    def record_ownership(self, claim: pb.Claim) -> None:
        """Record the authority this claim carries before ClaimAck.

        Called with `claim_lock` held so reconnects cannot race the in-process fence.
        """
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
        self, claim: pb.Claim, generation: int, send: Callable[[pb.WorkerFrame], None]
    ) -> None:
        """ClaimAck + the convergence snapshot, for a claim this worker can actually serve.

        `control_runtime_digest` is the exact image-owned cozy-runtime wheel fact from
        WheelhouseManifest/1. OCI identity remains the provider-readback axis; neither is
        substituted for the other.
        """
        ack = pb.ClaimAck(
            accepted=True,
            wire_minor=WIRE_MINOR,
            worker_id=self.options.worker_id,
            worker_instance_id=self.options.instance_id,
            worker_release_id=self.options.release_id,
            control_runtime_digest=self.options.control_runtime_digest,
            git_commit=self.options.git_commit,
            resources=self.resources(),
        )
        send(self.stamp(pb.WorkerFrame(claim_ack=ack)))
        self._send_snapshot(send)
        self.note(
            "session",
            f"claimed by {claim.record_owner_id or 'an unnamed RecordOwner'} at epoch "
            f"{claim.record_owner_epoch}, generation {generation} "
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
            admission_generation=self.admission_generation,
            admission_state=self.admission_state(),
            available_attempt_slots=self.available_attempt_slots(),
        )
        if self.placement is not None:
            body.placements.append(self.placement_status())
        body.held_attempts.extend(self.engine.snapshot_entries())
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
        generation: int,
        send: Callable[[pb.WorkerFrame], None],
        outbound: queue.Queue[pb.WorkerFrame | None],
    ) -> None:
        """ONE stream's durable sender, draining ONE stream's queue.

        The queue is passed in rather than read off `self`: a superseded pump that keeps
        looping for a moment must drain the queue it was born with, never the live stream's.
        """
        while not self.stop.is_set() and self.fence.control_stream_generation == generation:
            try:
                frame = outbound.get(timeout=REPORT_SECONDS)
            except queue.Empty:
                continue  # the generation check above is this thread's fence
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
        if message.control_stream_generation != self.fence.control_stream_generation:
            self.note(
                "fence",
                f"dropped: superseded generation {message.control_stream_generation}",
            )
            return True
        if message.worker_boot_id != self.fence.worker_boot_id:
            self.note("fence", f"dropped: wrong boot {message.worker_boot_id!r}")
            return True
        return False

    def _dispatch_frame(self, frame: pb.RecordOwnerFrame, generation: int) -> None:
        """Apply one frame while its stream still owns the worker.

        The stream generation is checked inside the same lock claims take. Trusting only the
        frame envelope left a check/use window in which a new claim could supersede this
        stream and the old handler could then mutate state and stamp replies as the new one.
        """
        with self.control_lock:
            if generation != self.fence.control_stream_generation:
                self.note("fence", f"dropped a frame from superseded stream {generation}")
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
            self.artifacts.replay_pending()
            for attempt in self.engine.pending_outcomes():
                self.note(
                    "outcome",
                    f"replaying the recorded outcome for {attempt.request_id}#"
                    f"{attempt.attempt} in a fresh envelope ({attempt.outcome_id})",
                )
                self.send(pb.WorkerFrame(attempt_outcome=self.engine.wrap(attempt)))
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
        elif kind == "artifact_host_ack":
            try:
                self.artifacts.absorb_ack(inner)
            except artifacts.ArtifactExchangeError as exc:
                self.note("artifact", f"dropped host ACK: {exc.code}: {exc.detail}")
        elif kind == "artifact_read_request":
            self.send(self.artifacts.read(inner))

    def on_outcome_ack(self, acked: pb.AttemptOutcomeAck) -> None:
        """A closed outcome FREES ITS ADMISSION SEAT — and that is a SLOT COUNT change, not
        a change in admission MEANING, so it does NOT bump `admission_generation` (§6).

        The distinction is load-bearing and it cost a live run to find. `admission_generation`
        moves when the meaning of an offer changes — executor respawn, window resize, phase
        change, cutover — and a RecordOwner dispatching against a stale one is refused
        deterministically. Freeing a seat changes neither: bumping here made EVERY
        steady-state dispatch refuse ADMISSION_GENERATION_STALE, because the ack that freed
        the seat invalidated the generation the RecordOwner was about to dispatch against.
        `available_attempt_slots` is the field that carries this, and it is recomputed on
        every tick.
        """
        self.note("outcome", self.engine.close(acked))
        if self.connected.is_set():
            self.send(pb.WorkerFrame(observed_state=self.observed_state()))
        closed = self.engine.history.get((acked.request_id, acked.attempt_ordinal))
        if self.reclaim_on_terminal and closed is not None and closed.kind == "job":
            self.note(
                "session",
                f"job attempt {acked.request_id}#{acked.attempt_ordinal} is acknowledged and "
                f"reclaim_on_terminal is set: exiting {RECYCLE_EXIT} (recycle, not a death)",
            )
            self.exit_code = RECYCLE_EXIT
            self.shutdown_from_stream()

    def shutdown_from_stream(self) -> None:
        self.stop.set()
        self.artifacts.close()
        self.outbound.put(None)
        self.work.put(None)
        self.watches.end_all()
        self.host.stop()

    def watch(self, opened: pb.ProgressOpen) -> Iterator[pb.AttemptProgress]:
        """One WatchProgress call: bound to the fenced control generation (#436)."""
        if self.fenced(opened):
            return iter(())
        return self.watches.open(opened.request_id, opened.control_stream_generation).frames(
            self.watches
        )

    # ------------------------------------------------------------- the desired state

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
            if len(placements) > 1:
                self.note(
                    "desired_state",
                    f"revision {desired.revision} REFUSED: {len(placements)} placements; "
                    "launch enforces len(placements) <= 1 and the desired state is UNAPPLIED",
                )
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_PLACEMENT_SET_UNSUPPORTED,
                        subject=f"revision {desired.revision}",
                        reason="placement_set_over_launch_clamp",
                        detail=f"the desired set names {len(placements)} placements and this "
                        "worker hosts at most one until cr-023's reconciler lands; #475 adds "
                        "that when the clamp lifts, multi-placement serving is CO-FITTING ONLY",
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
        self.accepted.mode = "serving" if mode == "placement_set" else mode
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
            self.placement = None
            self.pending = None
            self.bindings = {}
            self.prepared_models = {}
            self.engine.bindings = {}
            self.engine.prepared_models = {}
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
            self.placement = None
            self.pending = None
            self.engine.bindings = {}
            self.engine.prepared_models = {}
            self.accepted.converged_revision = desired.revision
            self.bump_admission("the accepted set is EMPTY; nothing is dispatchable")
            self.note("desired_state", f"revision {desired.revision} accepted: an EMPTY set")
            return
        # THE OUTGOING PLACEMENT KEEPS SERVING FROM HERE (fallback-retention, #474 clause 1).
        # Nothing below touches `self.bindings`, the prepared models, or the executor until
        # the incoming placement is materially STAGED — which is what makes the incoming's
        # every materialization failure free.
        entry = placements[0]
        placement_id = str(entry.get("placement_id", ""))
        package = dict(entry.get("package", {}))
        development = dict(entry.get("development", {}))
        package_descriptor = dict(entry.get("package_descriptor", {}))
        candidate = Placement(
            placement_id=placement_id,
            placement_set_digest=desired_set.placement_set_digest,
            package=package,
            development=development,
            environment_digest=_digest_field(entry.get("environment_digest")),
            package_descriptor=(
                str(package_descriptor.get("digest", "")),
                int(package_descriptor.get("length", 0)),
            ),
            bindings_digest=_digest_field(entry.get("bindings_digest")),
            models=tuple(dict(row) for row in entry.get("models", [])),
            entrypoints=tuple(dict(row) for row in entry.get("entrypoints", [])),
            environment=dict(entry.get("environment", {})),
            document=dict(entry),
            # #474/#485c: the predecessor is the rollback pin. A REPLACEMENT (same
            # placement_id, new set digest) retains the outgoing set as the pin; a fresh
            # placement has none, and an empty pin means replacement is PAUSED — which is
            # why the pin is set from the PREVIOUS spec and never invented.
            retained_fallback_placement_set_digest=(
                previous.placement_set_digest
                if previous is not None and previous.placement_id == placement_id
                else b""
            ),
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
        """One non-control lane for slow, generation-fenced placement convergence."""

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

        `MATERIALIZATION_STATE_STAGED` asserts that the exact overlay installed and imported;
        it is unspeakable otherwise (§8).
        """
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
                self.supervision.current.poisoned = "package overlay generation changed"
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
        self.placement = candidate
        self.pending = None
        self.bindings = bindings
        self.prepared_models = {}
        self.engine.bindings = self.bindings
        self.engine.prepared_models = {}
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
            if self.options.artifact_cache is None:
                self._materialization_failed(
                    placement,
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=placement.placement_id,
                        reason="artifact_cache_absent",
                        detail="source-checkout selection bytes require --artifact-cache",
                    ),
                    terminal=False,
                )
                return None
            try:
                from cozy_runtime.internal import descriptor as development_descriptor
                from cozy_runtime.internal import local as development_local
                from cozy_runtime.internal.discovery import discover

                found = discover(self.options.development_project)
                staged = development_local.stage_development_placement(
                    self.config.cozy_home,
                    development_descriptor.build(found),
                    package=self.options.development_package,
                    release=self.options.development_release,
                    source_digest=self.options.development_source_digest,
                    found=found,
                    artifact_store=self.options.artifact_store,
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
            prepared = self.private_revisions.get(
                (placement.placement_id, placement.execution_digest)
            )
            if prepared is None:
                self._materialization_failed(
                    placement,
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED,
                        subject=placement.placement_id,
                        reason="private_revision_unprepared",
                        detail="prepare this exact private code revision before binding models",
                    ),
                    terminal=True,
                )
                return None
            core_fields = (
                "development",
                "environment_digest",
                "environment",
                "package_descriptor",
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
                        reason="private_revision_changed",
                        detail=f"model-bound Placement changed prepared field {changed}",
                    ),
                    terminal=True,
                )
                return None
        if self.options.environment_root is None or self.options.artifact_store is None:
            self._materialization_failed(
                placement,
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                    subject=placement.placement_id,
                    reason="environment_materialization_input_missing",
                    detail="this worker was launched without --environment-root and "
                    "--artifact-store; the worker will not "
                    "guess where the worker put package overlays or TensorFS objects",
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
            acquirer = acquire.Acquirer(
                downloads.DownloadPlan(),
                cache_root=self.options.artifact_cache,
                environment_root=self.options.environment_root,
                selection_root=self.options.environment_root / "selections",
                artifact_store=self.options.artifact_store,
                python=Path(sys.executable),
                base_manifest=self.options.base_manifest,
                verified_cache=self.verified_artifacts,
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
        except (OSError, overlay.OverlayRefusal) as exc:
            reason = exc.code if isinstance(exc, overlay.OverlayRefusal) else "io_error"
            self._materialization_failed(
                placement,
                pb.Fault(
                    kind=pb.FaultKind.FAULT_KIND_CONFIG_REFUSED
                    if isinstance(exc, overlay.OverlayRefusal)
                    else pb.FaultKind.FAULT_KIND_ARTIFACT_FETCH_FAILED,
                    subject=placement.placement_id,
                    reason=reason,
                    detail=safe(str(exc)),
                ),
                terminal=isinstance(exc, overlay.OverlayRefusal),
            )
            return None

    def _observe_acquisition(
        self, placement: Placement, value: pb.PlacementAcquisitionObservation
    ) -> None:
        """Publish observation-only partial/terminal progress; never decide convergence."""

        placement.acquisition.CopyFrom(value)
        if placement is self.placement and self.connected.is_set():
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
        """The JOB lane: resolve the descriptor, spawn a fresh executor, become dispatchable.

        Job mode carries no placement, so its dispatchability edge is `job_ready` and its
        capacity is `JobCapacity` — the placement axes are unspeakable here and stay unset
        rather than being filled with a plausible OFFLINE.
        """
        self.bindings = {}
        self.prepared_models = {}
        self.engine.bindings = {}
        self.engine.prepared_models = {}
        self.jobs = {}
        self.placement = None
        self.pending = None
        self.set_job_ready(False)
        previous = self.supervision.current
        if previous is not None:
            try:
                self.supervision.retire_current(
                    previous, "entering job mode retires the serving generation"
                )
            except ExecutorGone as exc:
                self.note("boot", f"the serving executor could not be reclaimed: {exc}")
                self.latched = "job_executor_reclaim_failed"
                self._settle()
                return
        plans = self.config.cozy_home / "job-plans"
        record = plans / f"{job.job_descriptor_id.split(':')[-1]}.json"
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
            return
        self.supervision.use_environment(
            binding.python, binding.environment_content_digest
        )
        self.jobs[job.job_descriptor_id] = binding
        self.engine.jobs = self.jobs
        self.job_caps = job.resource_caps
        self.reclaim_on_terminal = job.reclaim_on_terminal
        self.note(
            "desired_state",
            f"revision {self.accepted.accepted_desired_state_revision} accepted in JOB mode: "
            f"{self.jobs[job.job_descriptor_id].job!r}, device floor {job.device_count}, "
            f"reclaim_on_terminal={job.reclaim_on_terminal}",
        )
        try:
            executor = self.supervision.spawn(imposed=self.imposed())
        except ExecutorGone as exc:
            self.note("boot", f"the job executor never dialled back: {exc}")
            self.latched = "job_executor_absent"
            self._settle()
            return
        executor.reserved_for_job = True
        self.note(
            "boot",
            f"job executor generation {executor.incarnation} pid {executor.pid} dialled "
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
        # A fresh activation re-earns every verdict; a stale fault must not outlive the
        # prepare that produced it.
        self.failed_bindings.clear()
        self.set_serving(pb.ServingState.SERVING_STATE_ACTIVATING)
        by_construction: dict[str, list[DeclaredBinding]] = {}
        for binding in self.bindings.values():
            by_construction.setdefault(binding.construction_key(), []).append(binding)
        if len(by_construction) != 1:
            detail = (
                f"the launch worker received {len(by_construction)} distinct model "
                "constructions, but one executor can own exactly one prepared generation"
            )
            for binding in self.bindings.values():
                self.failed_bindings[binding.entrypoint_binding_digest] = detail
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                        subject=binding.entrypoint_binding_digest,
                        reason="multiple_constructions_unsupported",
                        detail=detail,
                    )
                )
            executor = self.supervision.current
            if executor is not None:
                try:
                    self.supervision.retire_current(executor, detail)
                except ExecutorGone as exc:
                    self.note("recovery", f"multi-construction reclaim failed: {exc}")
            self.latched = "multiple_constructions_unsupported"
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
            self._settle()
            return

        construction_key, group = next(iter(by_construction.items()))
        binding_set = frozenset(binding.entrypoint_binding_digest for binding in group)
        executor = self.supervision.current
        if (
            executor is not None
            and executor.alive()
            and not executor.poisoned
            and executor.construction_key == construction_key
            and executor.prepared_binding_set == binding_set
            and executor.prepared_model is not None
        ):
            executor.ready_bindings.clear()
            for binding in group:
                if binding.expected_logical_model_digest and (
                    binding.expected_logical_model_digest
                    != executor.prepared_model.logical_model_digest
                ):
                    self.failed_bindings[binding.entrypoint_binding_digest] = (
                        "logical model digest mismatch"
                    )
                    continue
                executor.ready_bindings.add(binding.entrypoint_binding_digest)
                self.prepared_models[binding.entrypoint_binding_digest] = executor.prepared_model
            self.engine.bindings = self.bindings
            self.engine.prepared_models = self.prepared_models
            self.note(
                "boot",
                f"reused generation {executor.incarnation} pid {executor.pid} for the same "
                f"construction ({len(executor.ready_bindings)} binding(s))",
            )
            if executor.ready_bindings:
                # The high-water mark moves BEFORE the edge is published. `set_serving`
                # reports immediately — dispatchability is an edge — and `converged_revision`
                # CLAMPS itself back until serving actually says DISPATCHABLE, so writing it
                # first cannot publish an early claim, while writing it second published one
                # frame that said DISPATCHABLE and not-converged in the same breath.
                self.accepted.converged_revision = self.accepted.accepted_desired_state_revision
                self.set_serving(pb.ServingState.SERVING_STATE_DISPATCHABLE)
            else:
                self.latched = "logical model digest mismatch"
                self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
            return

        try:
            if executor is None:
                self.note("boot", "spawning the device executor (exec-first trampoline)")
                executor = self.supervision.spawn(imposed=self.imposed())
            elif (
                executor.alive()
                and not executor.poisoned
                and not executor.reserved_for_job
                and not executor.ready_bindings
                and executor.prepared_model is None
            ):
                self.note(
                    "boot",
                    f"preparing fresh generation {executor.incarnation} pid {executor.pid}",
                )
            else:
                self.note("boot", "replacing the device executor for a new construction")
                executor = self.supervision.replace(
                    executor,
                    imposed=self.imposed(),
                    why="desired construction changed",
                )
            self.note(
                "boot",
                f"generation {executor.incarnation} pid {executor.pid} dialled back in "
                f"{(executor.ready_at - executor.spawned_at) * 1000:.0f} ms "
                f"(no_new_privs={executor.hello.get('no_new_privs')}, "
                f"oom_score_adj={executor.hello.get('oom_score_adj')}, "
                f"torch_loaded={executor.hello.get('torch_loaded')})",
            )
        except ExecutorGone as exc:
            self.latched = f"executor_activation_failed: {exc}"[:200]
            self.note("boot", self.latched)
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, "executor absent")
            self._settle()
            return

        for group in by_construction.values():
            first = group[0]
            started = time.perf_counter()
            subject = f"prepare:{first.components[0] if first.components else first.entrypoint}"
            self.monitor.start(subject, "load")
            try:
                reply = executor.call(
                    {
                        "cmd": "prepare",
                        "binding": {
                            "application": first.application,
                            "package_descriptor": first.descriptor_path,
                            "development_project": first.development_project,
                            "model_class": first.model_class,
                            "model_binding_path": first.model_binding_path,
                            "model_parameter_name": first.model_parameter_name,
                            "component": first.components[0] if first.components else "",
                            "components": list(first.components),
                            "snapshots": dict(first.snapshots),
                            "store": first.store,
                            "config": first.config,
                            "snapshot": first.reference_snapshot,
                            "release": first.release,
                            "variant": first.variant,
                            "development": first.development,
                            # THE STAMP PLANE'S LAST HOP (#567d). The index row carries what the
                            # hub verified at publish; the executor hands it to `Artifact`, and
                            # a model class's keywords join it there. Dropping it here is what
                            # made `artifact.stamps` unconditionally `{}` and refused every
                            # stamped class on every checkpoint.
                            "stamps": {k: list(v) for k, v in first.stamps.items()},
                            "custody": first.custody,
                            "strict_keys": first.strict_keys,
                            "variants": [dataclasses.asdict(offer) for offer in first.offered],
                            "objective": first.objective,
                            "steps_basis": first.steps_basis,
                        },
                        "budgets": {
                            # cr-037/law 19: a prediction BASIS, not a budget — the wire
                            # field says what it carries so it cannot be read as a ceiling.
                            "declared_weight_bytes": first.model_weight_bytes,
                            "pinned": first.pinned_bytes,
                            # cl-027: None means DERIVE at prepare; the floor rides with it.
                            "resident": first.resident_budget_bytes,
                            "vram_floor": first.vram_floor_bytes,
                        },
                        "expect_construction_digest": first.expected_logical_model_digest,
                    },
                    timeout=None,
                    on_progress=lambda frame: self.monitor.advance(
                        f"prepare:{frame.get('name', '')}", int(frame.get("position", 0) or 0)
                    ),
                )
            except ExecutorGone as exc:
                self.monitor.end(subject, "executor_gone")
                detail = f"executor disappeared during prepare: {exc}"[:1024]
                for member in group:
                    self.failed_bindings[member.entrypoint_binding_digest] = detail
                    self.engine.fault(
                        pb.Fault(
                            kind=pb.FaultKind.FAULT_KIND_BINDING_UNAVAILABLE,
                            subject=member.entrypoint_binding_digest,
                            reason="executor_gone_during_prepare",
                            detail=safe(detail),
                        )
                    )
                try:
                    self.supervision.retire_current(executor, detail)
                except ExecutorGone as reclaim:
                    self.note("recovery", f"prepare-loss reclaim failed: {reclaim}")
                continue
            took = (time.perf_counter() - started) * 1000
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
                # the fault is typed on the placement.
                # PREPARE IS PER-BINDING (#572d). This used to latch the whole placement
                # and `return`, so ONE binding's failure denied every sibling the card —
                # on H3 a hidden, unbuildable ref2va binding took the working T2VA one
                # down with it, and the working one never got its turn to try.
                #
                # It is #548d's refusal-vs-fault split applied to bindings: a placement can
                # hold a FAULTED binding and still serve the others. The fault is typed on
                # the binding that earned it, its entrypoints refuse naming it, and the
                # loop goes on to the next group.
                for member in group:
                    self.failed_bindings[member.entrypoint_binding_digest] = (
                        f"{reply.get('code')}: {detail}"
                    )
                # #567c's SECOND HALF: a refusal reduced to its class name is a refusal
                # that does not exist. This line carried the code and four memory numbers
                # and DROPPED the executor's own message — so an operator watching a boot
                # fail read "unhandled_exception" beside "the allocator holds -1 B", which
                # names neither what broke nor where. The detail leads; the memory state
                # follows it, because on a genuine shortfall those numbers are the answer
                # and on everything else they are context.
                self.note(
                    "boot",
                    f"prepare REFUSED for {first.entrypoint_binding_digest[:20]}: "
                    f"{reply.get('code')} - {detail} "
                    f"[peak allocator {reply.get('peak_allocator_bytes', 0)} B was LIVE when "
                    f"it failed; the allocator holds {reply.get('allocator_bytes', -1)} B and "
                    f"the device has {reply.get('device_free_bytes', -1)} B free. The executor "
                    "is poisoned and replaced; the card returns with the process (#613)]" + "]",
                )
                # A failed construction owns no retry policy. Retire its possibly retained
                # allocations and remain offline; convergence decides whether to try again.
                try:
                    self.supervision.retire_current(
                        executor, f"prepare refused: {reply.get('code')}"
                    )
                except ExecutorGone as exc:
                    self.note("recovery", f"prepare refusal reclaim failed: {exc}")
                continue
            digest = reply["model_construction_digest"]
            delivery = reply.get("delivery") or {}
            resolved = PreparedModel(
                logical_model_digest=digest,
                constructed_model_digest=str(reply["constructed_model_digest"]),
                delivery_rung=str(delivery.get("rung", "")),
                delivery_variant=str(delivery.get("variant", "")),
                weight_bytes=int(reply["facts"]["filled_bytes"]),
            )
            executor.construction_key = construction_key
            executor.prepared_binding_set = binding_set
            executor.prepared_model = resolved
            for binding in group:
                if binding.expected_logical_model_digest and (
                    binding.expected_logical_model_digest != digest
                ):
                    self.note(
                        "boot",
                        f"{binding.entrypoint_binding_digest[:20]}… construction digest mismatch",
                    )
                    continue
                executor.ready_bindings.add(binding.entrypoint_binding_digest)
                self.prepared_models[binding.entrypoint_binding_digest] = resolved
            self.engine.bindings = self.bindings
            self.engine.prepared_models = self.prepared_models
            self.ledger.observe_construction(reply["facts"])
            self.monitor.end(subject, "filled")
            self.boot.step(
                "load",
                "filled",
                took,
                component=first.entrypoint,
                destinations=int(reply["facts"]["filled"]),
                bytes=int(reply["facts"]["filled_bytes"]),
                fill_ms=float(reply["facts"]["fill_ms"]),
                custody=str(reply["facts"].get("custody", "canonical")),
                ignored_extra_keys=", ".join(reply["facts"].get("ignored_extra_keys", ())),
                transcribed_markers=", ".join(reply["facts"].get("transcribed_markers", ())),
            )
            self.note(
                "boot",
                f"prepared {len(group)} binding(s) in {took:.0f} ms — "
                f"{reply['facts']['filled']} destinations, "
                f"{reply['facts']['filled_bytes'] / 2**30:.3f} GiB, fill leg "
                f"{reply['facts']['fill_ms'] / 1000:.3f} s, logical model {digest[:23]}… "
                f"generation {str(reply['constructed_model_digest'])[:23]}… "
                f"resident={sorted(reply['facts'].get('resident', {}))} "
                f"parked={reply['facts'].get('parked', [])}",
            )
            if delivery:
                self.note("delivery", _delivery_observation(delivery))
        # A placement with NO working binding has nothing to serve and says so; one with
        # any working binding becomes dispatchable and refuses only what actually failed.
        if len(self.failed_bindings) >= len(self.bindings):
            self.latched = "prepare refused: every binding"
            if self.options.development_project is not None and self.placement is not None:
                # A source-checkout generation has one placement and no retained published
                # fallback. Creator's readiness owner treats FAILED materialization as the
                # terminal answer and replaces this generation; leaving it STAGED/OFFLINE
                # made a permanent prepare refusal look like an endlessly activating model.
                self.placement.materialization = (
                    pb.MaterializationState.MATERIALIZATION_STATE_FAILED
                )
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, "prepare REFUSED")
            return
        if self.failed_bindings:
            self.note(
                "boot",
                f"{len(self.bindings) - len(self.failed_bindings)} of {len(self.bindings)} "
                f"binding(s) prepared; faulted: " + ", ".join(sorted(self.failed_bindings)),
            )
        self._finish_activation()

    def _finish_activation(self) -> None:
        """Run optional warm work, then publish the activation edge or its typed failure."""

        if not self.run_warm_pass():
            self.latched = "executor_lost_during_warm"
            self.set_serving(pb.ServingState.SERVING_STATE_OFFLINE, self.latched)
            self._settle()
            return
        if not self.latched:
            # Before the edge, for the reason above: the clamp in `converged_revision` is
            # what keeps this honest, and the ORDER is what keeps it consistent.
            self.accepted.converged_revision = self.accepted.accepted_desired_state_revision
        self.set_serving(pb.ServingState.SERVING_STATE_DISPATCHABLE)

    def run_warm_pass(self) -> bool:
        """One attempted warm case per WARMABLE entrypoint, through ORDINARY dispatch."""
        # Dynamic placement workers execute package source only for a real invocation,
        # after the placement is DISPATCHABLE. Their environment root is the structural
        # identity of that mode; the fixed worker's `no_warm` remains a belt, but correctness
        # cannot depend on one argv translation surviving every launch layer. A real pod
        # proved why: its model prepared successfully, then a synthetic warm handler blocked
        # forever in its first CUDA scalar read and kept the placement ACTIVATING.
        if self.options.environment_root is not None:
            self.note(
                "warm",
                "dynamic placement worker: boot warm pass disabled; package execution "
                "begins only after DISPATCHABLE",
            )
            executor = self.supervision.current
            return executor is not None and self.supervision.owns(executor)
        if self.options.no_warm:
            self.note("warm", "boot warm pass disabled by launch option")
            executor = self.supervision.current
            return executor is not None and self.supervision.owns(executor)
        executor = self.supervision.current
        if executor is None:
            return False
        for plan_id, binding in self.bindings.items():
            if plan_id not in executor.ready_bindings:
                continue
            spool = self.root / "warm" / binding.entrypoint
            spool.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            try:
                reply = executor.call(
                    {
                        "cmd": "warm",
                        "request_id": f"warm-{binding.entrypoint}",
                        "entrypoint": binding.entrypoint,
                        # NO BOUND: warm is a BOOT STEP and boot is silence-bounded (§3.3).
                        "deadline_s": None,
                        "spool": str(spool),
                    },
                    timeout=None,
                )
                outcome: dict[str, Any] = {
                    "entrypoint": binding.entrypoint,
                    "outcome": str(reply.get("outcome", "failed")),
                    "reason": str(reply.get("reason", ""))[:200],
                    "boot_warmup_observed": bool(reply.get("boot_warmup_observed", False)),
                }
            except ExecutorGone as exc:
                self.supervision.invalidate(executor, "executor died during warmup")
                try:
                    self.supervision.retire_current(executor, "executor died during warmup")
                except ExecutorGone as reclaim:
                    self.note("recovery", f"warmup reclaim failed: {reclaim}")
                self.note("warm", f"{binding.entrypoint}: executor lost: {exc}"[:200])
                return False
            outcome["ms"] = round((time.perf_counter() - started) * 1000, 1)
            outcome["demand_evidence"] = False
            self.boot.step(
                "warm",
                str(outcome["outcome"]),
                float(outcome["ms"]),
                entrypoint=binding.entrypoint,
                benchmark_override=True,
                reason=str(outcome["reason"])[:200],
            )
            self.records.append("warm", dict(outcome))
            self.warm.append(outcome)
            self.note(
                "warm",
                f"{binding.entrypoint}: {outcome['outcome']} ({outcome['ms']} ms) "
                f"{outcome['reason']}"[:200],
            )
            if outcome["outcome"] == "failed":
                self.engine.degraded[plan_id] = str(outcome["reason"])[:160]
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_BINDING_DEGRADED,
                        subject=plan_id,
                        reason="serve_degrade",
                        detail=f"boot warm pass failed for {binding.entrypoint}: "
                        f"{outcome['reason']} - the worker still serves",
                    )
                )
        if self.warm and executor.alive():
            try:
                self.ledger.observe_probe(
                    executor.call({"cmd": "probe"}, timeout=30.0, total_timeout=30.0)
                )
            except ExecutorGone:
                self.supervision.invalidate(executor, "executor died during warmup probe")
                try:
                    self.supervision.retire_current(executor, "executor died during warmup probe")
                except ExecutorGone as reclaim:
                    self.note("recovery", f"warmup probe reclaim failed: {reclaim}")
                return False
            shift = self.ledger.rebaseline("boot warm pass")
            self.note("warm", f"the resident device baseline moved {shift:+d} B on the warm pass")
        return self.supervision.owns(executor)

    def imposed(self) -> dict[str, str]:
        """The IMPOSE half of the seal. Every name here is in the reviewed allowlist."""
        imposed = {
            "CUDA_VISIBLE_DEVICES": self.options.devices,
            "PYTORCH_CUDA_ALLOC_CONF": self.options.alloc_conf,
            "OMP_NUM_THREADS": str(self.options.threads),
        }
        # The FILL runs in the executor, so its capture directory has to cross the seal or
        # it instruments the one process that never reserves a destination. Projected only
        # when one is configured: an empty value is an imposition of nothing.
        where = self.config.fill_forensics_dir
        if where:
            imposed[child_env.FILL_FORENSICS_DIR] = where
        return imposed

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
        # and because a stale generation means the RecordOwner dispatched against a picture
        # of this worker that no longer holds.
        if offered.admission_generation != self.admission_generation:
            self.refuse(
                offered,
                "admission_generation_stale",
                f"the offer echoes admission generation {offered.admission_generation} and "
                f"this worker is at {self.admission_generation}; the verdict is the same on "
                "every replay of these bytes, and the ordinal is consumed with no budget draw",
                CAUSE.CAUSE_CODE_ADMISSION_GENERATION_STALE,
            )
            return
        if self.accepted.mode == "serving":
            wanted = self.placement.placement_id if self.placement is not None else ""
            if offered.placement_id != wanted:
                self.engine.fault(
                    pb.Fault(
                        kind=pb.FaultKind.FAULT_KIND_UNKNOWN_PLACEMENT,
                        subject=offered.placement_id,
                        reason="unknown_placement",
                        detail=f"the offer names placement {offered.placement_id!r} and this "
                        f"worker hosts {wanted!r}",
                    )
                )
                self.refuse(
                    offered,
                    "unknown_placement",
                    f"no placement {offered.placement_id!r} is hosted here",
                    CAUSE.CAUSE_CODE_UNKNOWN_PLACEMENT,
                )
                return
            if not self.dispatchable():
                self.refuse(
                    offered,
                    "placement_not_dispatchable",
                    f"placement {wanted!r} is "
                    + (pb.ServingState.Name(self.placement.serving) if self.placement else "absent")
                    + (f" (LATCHED: {self.latched})" if self.latched else ""),
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
        if attempt.state in ("accepted", "running", "finalizing"):
            self.note(
                "attempt",
                "duplicate AttemptOffer for a held attempt: re-accepting "
                "the SAME acceptance; nothing starts a second time",
            )
            self.send(pb.WorkerFrame(attempt_accepted=self.engine.accepted_message(attempt)))
            return
        # THE WINDOW IS ONE, AND OVER IT THE ANSWER IS A JOURNALED OUTCOME (#494f/§7). The
        # device lane is a single serialized state machine, so a second held attempt could
        # only wait — and it would wait in a state the wire cannot spell, absent from the
        # snapshot and from the held-attempt set. A refusal the RecordOwner can read and
        # re-dispatch is strictly more informative than a hold it cannot see.
        if self.available_attempt_slots() < 1:
            busy = [
                held
                for held in self.engine.history.values()
                if held is not attempt and held.state in self.engine.HELD
            ]
            other = busy[0] if busy else None
            self.note(
                "attempt",
                f"{offered.request_id}#{offered.attempt_ordinal} over the admission window",
            )
            self.refuse(
                offered,
                "no_capacity",
                f"this worker admits {ADMISSION_WINDOW} attempt at a time and the window is "
                + (
                    f"held by {other.request_id}#{other.attempt} ({other.state})"
                    if other is not None
                    else "closed"
                )
                + "; a seat is held by a running attempt AND by an outcome awaiting "
                "AttemptOutcomeAck, so acking frees one",
                CAUSE.CAUSE_CODE_NO_CAPACITY,
            )
            return
        self.engine.hold(attempt)
        self.work.put(attempt)

    def await_settled(self, attempt: AttemptRecord) -> None:
        """QUEUE behind a rebuild instead of refusing through one — for exactly as long as
        the rebuild is MAKING PROGRESS (#434: caller deadline or observed stall, never a
        wall-clock guess)."""
        if self.settled.is_set():
            return
        started = time.perf_counter()
        key = f"{attempt.request_id}#{attempt.attempt}"
        deadline_ms = int(attempt.spec.get("deadline_unix_ms", 0))
        self.note(
            "readiness",
            f"{key} reached the lane while the worker is rebuilding: waiting while the "
            "rebuild makes progress rather than refusing it",
        )
        still = 0
        position, burned = -1, -1
        why = ""
        while not (settled := self.settled.wait(REBUILD_SAMPLE_SECONDS)):
            if deadline_ms and time.time() * 1000 > deadline_ms:
                why = "the attempt's own deadline expired while it queued"
                break
            still += 1
            (position, before_position) = (self.boot.ring.admitted, position)
            (burned, before_burned) = (self.read_device_work(), burned)
            if position > before_position >= 0 or burned > before_burned >= 0:
                still = 0
            if still >= observe.GIVE_UP_AFTER:
                why = (
                    f"{still} consecutive observations found the rebuild at ring position "
                    f"{position} having burned {burned} ns: it is stalled, not slow"
                )
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
        attempt = held or AttemptRecord(
            request_id=offered.request_id,
            attempt=offered.attempt_ordinal,
            digest=offered.invocation_spec_digest,
            spec={},
        )
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

    def device_lane(self) -> None:
        """ONE SERIALIZED PER-DEVICE STATE MACHINE: admit, execute, reconcile, rebuild."""
        while not self.stop.is_set():
            item = self.work.get()
            if item is None:
                return
            if isinstance(item, ExecutorDeath):
                self.rebuild_exited(item.executor)
                continue
            attempt = item
            if self.admit(attempt):
                self.execute(attempt)
            # Rebuild on REFUSALS too (#458): a poisoned generation refuses every attempt,
            # so a rebuild gated on admission could never run again.
            self.rebuild_if_needed()

    def admit(self, attempt: AttemptRecord) -> bool:
        """FETCH -> HYDRATE -> CHOOSE -> ACCEPT, or a typed terminal. Never a silence."""
        key = f"{attempt.request_id}#{attempt.attempt}"
        self.await_settled(attempt)
        # Desired-state convergence and executor rebuilds take this same lock. Recheck the
        # offer's immutable routing facts under it, then hold it through the acceptance
        # A queued offer cannot be accepted on the far side of a placement or
        # admission-generation change merely because it passed `on_offer()` earlier.
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
            if attempt.admission_generation != self.admission_generation:
                self.note(
                    "admission",
                    f"refusing queued {key}: offered generation "
                    f"{attempt.admission_generation}, current {self.admission_generation}",
                )
                self.terminate(
                    attempt,
                    self.engine.refuse_held,
                    "admission_generation_stale",
                    f"the queued offer echoes admission generation "
                    f"{attempt.admission_generation} and this worker is now at "
                    f"{self.admission_generation}",
                    CAUSE.CAUSE_CODE_ADMISSION_GENERATION_STALE,
                )
                return False
            if self.accepted.mode == "serving":
                wanted = self.placement.placement_id if self.placement is not None else ""
                if attempt.placement_id != wanted:
                    self.note(
                        "admission",
                        f"refusing queued {key}: placement {attempt.placement_id!r} is no "
                        f"longer {wanted!r}",
                    )
                    self.terminate(
                        attempt,
                        self.engine.refuse_held,
                        "unknown_placement",
                        f"the queued offer names placement {attempt.placement_id!r} and "
                        f"this worker now hosts {wanted!r}",
                        CAUSE.CAUSE_CODE_UNKNOWN_PLACEMENT,
                    )
                    return False
            if not self.dispatchable():
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
                accepted = self.engine.prepare_and_accept(attempt)
            except AttemptRefusal as exc:
                self.note("attempt", f"{key} refused: {exc.code}")
                self.terminate(
                    attempt, self.engine.refuse_held, exc.code, exc.detail, exc.cause, exc.origin
                )
                return False
        self.send(pb.WorkerFrame(attempt_accepted=accepted))
        return True

    def execute(self, attempt: AttemptRecord) -> None:
        """RUN -> OUTCOME -> RECONCILE. The ledger closes inside `outcome()`."""
        try:
            outcome = self.engine.execute(attempt)
        except Exception as exc:
            self.note("attempt", f"execution faulted: {type(exc).__name__}: {exc}")
            try:
                outcome = self.engine.abandon(
                    attempt, f"the worker faulted while running this attempt: {exc}"
                )
            except Exception as inner:
                self.note("attempt", f"even ABANDONED could not be built: {inner}")
                self.fail_machine(f"{type(inner).__name__}")
                self.terminate(attempt, self.engine.last_resort, f"{type(inner).__name__}: {inner}")
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

    def rebuild_if_needed(self) -> None:
        """The queue starts behind the rebuild because BOTH live on this lane."""
        with self.control_lock:
            current = self.supervision.current
            if current is None or not current.alive():
                self._rebuild_locked("the executor is gone")
            elif current.poisoned:
                self._rebuild_locked(f"poisoned: {current.poisoned}")

    def rebuild(self, why: str) -> None:
        """Replace the executor and RELOAD before the placement is dispatchable again."""
        with self.control_lock:
            self._rebuild_locked(why)

    def rebuild_exited(self, executor: child.Executor) -> None:
        """Rebuild only if the queued death still names the owned dead incarnation."""

        with self.control_lock:
            if self.supervision.current is not executor or executor.alive():
                return
            self._rebuild_locked(
                f"observed exit of executor incarnation {executor.incarnation} pid {executor.pid}"
            )

    def _rebuild_locked(self, why: str) -> None:
        """Rebuild while desired-state convergence is excluded."""
        job = self.accepted.mode == "job"
        if job:
            self.set_job_ready(False)
        else:
            self.set_serving(pb.ServingState.SERVING_STATE_ACTIVATING, "executor rebuild")
        started = time.perf_counter()
        self.note("recovery", f"rebuilding the executor: {why}")
        current = self.supervision.current
        try:
            if current is None:
                self.supervision.spawn(imposed=self.imposed())
            else:
                self.supervision.replace(current, imposed=self.imposed(), why=why)
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
            f"executor generation {self.supervision.incarnation} / admission generation "
            f"{self.admission_generation}",
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
                            self.supervision.cooperative_cancel(
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
                            self.capacity_dropped("executor reclaim is not proved")
                            attempt.cancel_started = time.perf_counter()
                            continue
                        self.note("cancel", f"forceful: {json.dumps(facts)}")
                        attempt.canceling = attempt.canceling or "client"
                        attempt.cancel_started = time.perf_counter() + 1e9

    def reporter(self) -> None:
        """The ObservedWorkerState tick, and the liveness SAMPLE that rides it."""
        seen: dict[str, str] = {}
        while not self.stop.wait(REPORT_SECONDS):
            for subject in list(self.monitor.cursors):
                verdict = self.monitor.sample(subject)
                before = seen.get(subject)
                if verdict == "wedged" and before != verdict:
                    self.note("liveness", f"{subject} is WEDGED by silence (no clock involved)")
                elif before == "wedged" and verdict in ("advancing", "busy"):
                    # THE RETRACTION (cl-025 residue). A WEDGED note is a durable fact on
                    # the lane; a subject that moves again must say so on the same lane, or
                    # the last word about it stays a wedge that no longer exists.
                    self.note(
                        "liveness",
                        f"{subject} resumed ({verdict}): the earlier WEDGED verdict is retracted",
                    )
                seen[subject] = verdict
            if self.connected.is_set():
                self.send(pb.WorkerFrame(observed_state=self.observed_state()))

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
        """
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
            admission_generation=self.admission_generation,
            admission_state=self.admission_state(),
            available_attempt_slots=self.available_attempt_slots(),
        )
        if self.accepted.mode == "job":
            observed.job_capacity.CopyFrom(self.job_capacity())
        elif self.accepted.mode == "serving":
            observed.accepted_placement_set_digest = self.accepted.accepted_placement_set_digest
            if self.placement is not None:
                observed.placements.append(self.placement_status())
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
        if self.accepted.mode == "serving" and not self.dispatchable():
            return min(
                self.accepted.converged_revision,
                max(0, self.accepted.accepted_desired_state_revision - 1),
            )
        return self.accepted.converged_revision

    def placement_status(self) -> pb.PlacementStatus:
        """Per-placement truth on TWO AXES (#473/#482), and `readiness_epoch` is not among
        them: it is DELETED, not renamed (#486b), because the serving axis plus
        `dispatchable_binding_digests` carry what a placement can serve and the worker-level
        `admission_generation` carries whether an offer is admissible. `attempt_credits` is
        gone for the same reason, one level up.

        The accelerator qualification is deliberately ABSENT: this torch-free process cannot
        measure capability facts, and absent IS the present-but-unqualified state (#454 —
        registration honesty is schema). The executor-measured qualification lands with
        cr-020/cr-021's lanes.
        """
        assert self.placement is not None
        executor = self.supervision.current
        dispatchable = sorted(executor.ready_bindings) if executor is not None else []
        selected = {
            str(entrypoint["entrypoint_binding_digest"])
            for entrypoint in self.placement.entrypoints
        }
        # DISJOINT by construction (the proto says so): a binding is either servable now or
        # still to be materialized, never counted in both.
        materializable = sorted(selected - set(dispatchable))
        status = pb.PlacementStatus(
            placement_id=self.placement.placement_id,
            placement_set_digest=self.placement.placement_set_digest,
            materialization=self.placement.materialization,
            serving=self.placement.serving,
            retained_fallback_placement_set_digest=(
                self.placement.retained_fallback_placement_set_digest
            ),
            executor_generation=self.supervision.incarnation,
            dispatchable_binding_digests=[documents.raw(value) for value in dispatchable],
            materializable_binding_digests=[documents.raw(value) for value in materializable],
            faults=[
                f
                for f in self.engine.faults[-8:]
                if f.kind
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
            self.placement.materialization == pb.MaterializationState.MATERIALIZATION_STATE_STAGED
            and self.placement.serving == pb.ServingState.SERVING_STATE_DISPATCHABLE
        ):
            status.package_revision_digest = self.placement.execution_digest
            status.environment_digest = (
                documents.spell(self.placement.environment_digest)
                if self.placement.environment_digest
                else canonical.digest({})
            )
            status.config_digest = self.placement.config_digest
            if not all(
                re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                for value in (
                    status.package_revision_digest,
                    status.environment_digest,
                    status.config_digest,
                )
            ):
                raise RuntimeError("dispatchable placement has incomplete invocation identity")
        if self.placement.acquisition.ListFields():
            status.acquisition.CopyFrom(self.placement.acquisition)
        return status

    def job_capacity(self) -> pb.JobCapacity:
        """How many JOB attempts this worker can take right now (cr-009)."""
        in_flight = sum(1 for a in self.engine.live.values() if a.job is not None)
        ready = bool(self.jobs) and self.job_ready
        return pb.JobCapacity(
            jobs_in_flight=in_flight,
            jobs_available=max(0, (1 if ready else 0) - in_flight),
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
            control_runtime_digest=self.options.control_runtime_digest,
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
            self.set_phase(pb.WorkerPhase.WORKER_PHASE_ONLINE, "boot completed")
            threading.Thread(target=self.device_lane, daemon=True, name="device-lane").start()
            threading.Thread(
                target=self.convergence_lane, daemon=True, name="convergence-lane"
            ).start()
            threading.Thread(target=self.reporter, daemon=True, name="report").start()
            threading.Thread(target=self.watchdog, daemon=True, name="watchdog").start()
        try:
            self.host.serve(self)
        finally:
            self.shutdown()
        return self.exit_code

    def shutdown(self) -> None:
        seam = (
            self.supervision.current.channel.accounting()
            if self.supervision.current is not None
            else {}
        )
        self.stop.set()
        self.artifacts.close()
        self.work.put(None)
        self.convergences.put(None)
        self.outbound.put(None)
        self.watches.end_all()
        self.host.stop()
        for operation_id in sorted(self.model_source_operations):
            try:
                self.release_model_source(operation_id)
            except Exception as exc:
                self.note(
                    "shutdown",
                    f"model source cleanup {operation_id} failed: {type(exc).__name__}",
                )
        try:
            self.supervision.close()
        except ExecutorGone as exc:
            self.exit_code = int(Exit.internal)
            self.note("shutdown", f"executor reclaim failed: {exc}")
        print(
            "[worker] summary "
            + json.dumps(
                {
                    "worker_boot_id": self.fence.worker_boot_id,
                    "control_generation": self.fence.control_stream_generation,
                    "executor_generations": self.supervision.spawns,
                    "admission_generation": self.admission_generation,
                    "executed": self.engine.executed,
                    "record_count": self.records.appends,
                    "progress_sent": self.watches.sent,
                    "progress_shed": self.watches.shed,
                    "hub_calls_allowed": self.authorizer.allowed,
                    "hub_calls_refused": len(self.authorizer.refusals),
                    "seam": seam,
                    "ledger": self.ledger.rows(),
                    "warm": self.warm,
                    "rss_bytes": read_vmrss(os.getpid()),
                }
            ),
            flush=True,
        )


#: A provenance digest is `sha256:` plus 64 lowercase hex, or it is not one.
_CONTROL_RUNTIME_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")

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
    parser.add_argument(
        "--control-runtime-digest",
        default="",
        help="the exact image-owned cozy-runtime wheel digest from WheelhouseManifest/1; "
        "absent for native workers",
    )
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
        "--environment-root",
        default="",
        help="package-overlay root containing exact receipts and content-keyed generations",
    )
    parser.add_argument(
        "--artifact-cache",
        default="",
        help="where the dynamic acquisition lane writes the exact bytes a placement reaches "
        "(cr-049); content-addressed, so a replay re-fetches only what is missing",
    )
    parser.add_argument(
        "--base-manifest",
        default=str(overlay.DEFAULT_BASE_MANIFEST),
        help="exact base worker image WheelhouseManifest used for overlay ownership",
    )
    parser.add_argument(
        "--artifact-store",
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
    parser.add_argument("--host-budget", type=int, default=8 << 30)
    parser.add_argument("--pinned-budget", type=int, default=512 << 20)
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--executor-uid", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--executor-gid", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--tls-cert", default="", help="PEM cert for a non-loopback listener")
    parser.add_argument("--tls-key", default="", help="PEM key for a non-loopback listener")
    args = parser.parse_args(argv)

    if args.control_runtime_digest and not _CONTROL_RUNTIME_DIGEST.fullmatch(
        args.control_runtime_digest
    ):
        print(
            f"[worker] --control-runtime-digest {args.control_runtime_digest!r} is not "
            "`sha256:<64 hex>`: the image-owned Runtime wheel digest is measured or absent",
            file=sys.stderr,
            flush=True,
        )
        return int(Exit.structural)
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
        control_runtime_digest=args.control_runtime_digest,
        git_commit=args.git_commit,
        python=args.python,
        accelerator_backend=args.accelerator_backend,
        devices=args.devices,
        grant_roots=tuple(r for r in args.grant_root.split(",") if r),
        environment_root=Path(args.environment_root).resolve() if args.environment_root else None,
        base_manifest=Path(args.base_manifest).resolve(),
        artifact_store=Path(args.artifact_store).resolve(),
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
        host_budget=args.host_budget,
        pinned_budget=args.pinned_budget,
        no_warm=args.no_warm,
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

    def terminate(signum: int, frame: object) -> None:
        worker.stop.set()
        worker.outbound.put(None)
        worker.work.put(None)
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
