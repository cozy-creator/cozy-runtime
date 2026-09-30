"""The long-lived device executor: Python, CUDA, loaded constructions, one GPU attempt.

This is the ONLY process in the worker that imports torch or touches a device. It lives
for its (environment, lane) and holds every construction it has loaded; it is still
disposable in the strong sense: `kill -9` at any instant is a supported input, and the
worker converges to a typed terminal and a fresh epoch from what it recorded.
Nothing durable lives here — no records, no credential, no monotone wire counter.

Its whole surface is the seam (`internal/seam.py`), and its whole vocabulary is:

    hello     process facts: pid, parent, no_new_privs, oom_score_adj, the SEALED env
    describe_installed  derive one installed PackageInterface into brokered scratch
    start     spawn and join every rank, import torch, the runtime and the package, warm up
    load      construct + fill one construction under its key; idempotent per key
    activate  make one loaded construction resident, parking LRU constructions for room
    unload    drop one construction and free its device bytes
    residency every loaded construction, resident or parked
    prepare_request  resolve ONE request (decode, normalize, preflight) and retain it, so
              the worker prices a plan against the record the handler will actually run
    invoke    run ONE PREPARED attempt through the author invocation kernel and report its facts
    run_job   run ONE @app.job to completion through the SAME kernel, with job services
    probe     device/host memory facts, for the worker's ledger reconciliation
    shutdown  cooperative exit

Cancellation is attempt-keyed: `ctx.cancelled` reads the marker the worker publishes for
this exact request. A stale cancellation therefore cannot cross into the next attempt.
Every telemetry step, stage and progress call and, in a group, every mirrored call is a safe
point; the attempt ends CANCELED there and the process stays Ready. The worker reclaims the
whole executor scope only when the attempt stops moving against its own measured cadence.

The env seal is enforced, not trusted: the sealed values are snapshotted BEFORE torch is
imported, re-imposed before Torch import, CUDA init and every reply, and any names package
code changed are reported in `env_changed` rather than silently ignored.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import msgspec
from msgspec import UNSET, UnsetType

from cozy_runtime import __version__ as RUNTIME_VERSION
from cozy_runtime.author._assets import GrantedInput
from cozy_runtime.author._attention import AttentionContext
from cozy_runtime.author._capture import ActivationCapture
from cozy_runtime.author._context import AdapterRef, Device
from cozy_runtime.author._errors import (
    CapabilityError,
    InvalidRequest,
    Outcome,
    classify,
    is_device_oom,
)
from cozy_runtime.author._executor_requests import (
    Answer,
    Checkpoint,
    CheckpointReceipt,
    ChildEvents,
    DeviceRoom,
    Handoff,
    Publish,
    Published,
    Request,
    Room,
)
from cozy_runtime.author._executor_requests import encode as encode_request
from cozy_runtime.author._invoke import (
    Invocation,
    PreparedRequest,
    attempt,
    prepare,
    run_prepared,
)
from cozy_runtime.author._model import Model, _derive_model, component_use
from cozy_runtime.author._services import Attempt as AuthorAttempt
from cozy_runtime.author._services import CheckpointConflict, ProgressFrame
from cozy_runtime.internal import (
    accel,
    attention,
    attention_sol,
    attention_ulysses,
    canonical,
    child_env,
    execution_evidence,
    executor_commands,
    kernel_cache,
    model_config,
    native_interfaces,
    package_interface,
    prepare_diagnostics,
    proctree,
    sandbox,
    tolerant,
)
from cozy_runtime.internal.config import restore_seal, seal_snapshot
from cozy_runtime.internal.discovery import discover_distribution, discover_installed
from cozy_runtime.internal.executor_commands import (
    Activate,
    Attention,
    Binding,
    CallInterface,
    DescribeInstalled,
    GroupHardware,
    Hello,
    Invoke,
    Join,
    KnownCommand,
    Load,
    ModelLoad,
    PrepareRequest,
    Probe,
    Release,
    Residency,
    Restore,
    RunJob,
    Shutdown,
    Start,
    Unload,
    Vacate,
    Warm,
)
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.parallel import cp, mirror, wire
from cozy_runtime.internal.parallel.group import (
    RankGroup,
    RankSpec,
    join_rank,
    receive_residence,
)
from cozy_runtime.internal.parallel.mirror import mirror_component
from cozy_runtime.internal.parallel.plan import (
    GpuDivergence,
    GroupPlan,
    GroupRefusal,
    rank_invariant_digest,
)
from cozy_runtime.internal.residency import (
    Construction,
    ConstructionArbiter,
    ResidencyRefusal,
    RestoreReport,
)
from cozy_runtime.internal.seam import (
    EXECUTOR_PROTOCOL_REVISION,
    RESULT_DOCUMENT,
    Channel,
    SeamError,
    connect,
)
from cozy_runtime.internal.worker.plan import shape_cell

#: The inline-result door (worker-protocol/01 `ResultEnvelope`): a small typed result rides
#: the terminal envelope and skips the file-store round trip. Larger than this REFUSES at
#: cr-007 — the blob-receipt branch needs a real write-and-receipt transaction, and minting
#: a receipt for bytes nobody wrote is the fabrication class this runtime exists to prevent.
INLINE_RESULT_MAX = 4 * 1024 * 1024

#: How many PREPARED requests this epoch holds at once. A serving lane admits and
#: runs one attempt at a time, so anything past the newest few is a request the
#: worker prepared and then never dispatched.
PREPARED_HELD = 8

#: The environment as it was when this process started, restricted to the sealed names.
#: Captured before any import that could initialize CUDA.
_SEALED: dict[str, str] = {}


def _capture_seal() -> None:
    """Snapshot the seal BEFORE anything that could initialize CUDA."""
    _SEALED.update(seal_snapshot())


def _env_changes() -> tuple[str, ...]:
    """Names whose current values differ from the pre-Torch executor seal.

    Values never cross the seam. The names are enough to diagnose which reviewed
    imposition moved without leaking paths or future secret-bearing values into a terminal.
    """
    current = seal_snapshot()
    return tuple(name for name in sorted(_SEALED) if current.get(name, "") != _SEALED[name])


def _sealed_entries() -> tuple[str, ...]:
    """The seal's `CUDA_VISIBLE_DEVICES` entries: one per device this process may use."""
    return tuple(
        entry.strip()
        for entry in _SEALED.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if entry.strip()
    )


def _gpu(rank: int) -> str:
    """How people name the GPU group process `rank` drives: `GPU 3`, nvidia-smi's number."""
    return execution_evidence.gpu_name(_sealed_entries(), rank)


def _qualification_cache() -> str:
    """Where a measured qualification is kept between executors, from the SEAL.

    `COZY_HOME` is already a reviewed imposition — the executor resolves the same CAS as
    the worker through it — and the worker already keeps node-local caches under it. This
    is a path, not a switch: an unset or relative value means nowhere to keep a result and
    the suite simply runs, which is what every executor did before cr-103. The directory
    is not created here; `probe.qualified` makes it when it has something to keep.
    """
    home = _SEALED.get("COZY_HOME", "").strip()
    if not home or not Path(home).is_absolute():
        return ""
    return str(Path(home) / "qualification")


def _restore_seal() -> tuple[str, ...]:
    """Re-impose sealed values that package code changed, and name them.

    The seal compares a reviewed allowlist of environment VALUES (device list, NCCL
    and allocator settings, thread cap, cache and store paths), not files or hashes.
    Torch and CUDA read them once, so re-imposing before CUDA init keeps exactly the
    lane's devices and settings, and after it keeps the process consistent with the
    configuration it runs. A package's change is a no-op either way: warn, never refuse.
    """
    changed = _env_changes()
    restore_seal(_SEALED, changed)
    if changed:
        print(
            f"[executor] restored sealed environment names changed by package code: "
            f"{', '.join(changed)}",
            file=sys.stderr,
            flush=True,
        )
    return changed


def _env_document() -> dict[str, Any]:
    """The seal holds for every reply; names package code changed are reported."""
    return {"env_intact": True, "env_changed": list(_restore_seal())}


def _cancelled(root: Path, request_id: str) -> bool:
    """Whether the worker's one cooperative marker names this exact attempt."""
    try:
        return (root / "executor.cancel").read_text() == request_id
    except (OSError, UnicodeError):
        return False


def _distribution_version(name: str) -> str:
    """An installed distribution's version in this environment, or ""."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _proc_status(field: str) -> str:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return line.split(":", 1)[1].strip()
    return ""


def _rss() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _deadline(seconds: float | None) -> float:
    """The monotonic instant this attempt must not outlive — or NO INSTANT AT ALL.

    `None` is "does not expire", and it is deliberately not 0: a duration of zero is a real
    remaining time, so reading the two as one would hand an attempt whose deadline had just
    lapsed an unbounded one. An attempt whose caller minted no deadline has no bound to
    derive one from, and inventing one here would put a ceiling on what an attempt may be
    that the worker's own watchdog does not enforce — two clocks disagreeing about the
    same attempt (cr-009).
    """
    return math.inf if seconds is None else time.monotonic() + seconds


def _granted_inputs(command: Invoke | RunJob) -> dict[str, GrantedInput]:
    """Verified input paths, projected identically into either execution lane."""
    return {input_id: row.granted(input_id) for input_id, row in command.inputs.items()}


def _trees(command: Invoke | RunJob) -> dict[str, tuple[Path, str]]:
    return {ref: (Path(root), digest) for ref, (root, digest) in command.trees.items()}


def _working_peak(peak: int, at_entry: int, residency: Mapping[str, Any]) -> int:
    """The attempt's working memory: the allocator peak over what was allocated at device
    entry, less the weights the attempt itself staged in; never below a declared scope's
    own activation peak, which is measured against that scope's resident baseline."""
    moved = max(int(residency.get("allocator_delta_bytes", 0)), 0)
    return max(
        int(peak) - int(at_entry) - moved,
        int(residency.get("attempt_activation_peak_bytes", 0)),
        0,
    )


def residence_ceiling(*, destinations: int, authorized: int, allocatable: int) -> int:
    """The construction's residence ceiling: ASSIGNED by the worker, VERIFIED by the card.

    `authorized` is the ceiling the worker handed this executor for this prepare (cr-025's
    `authorized_device_limit_bytes`; cr-066 assigns it from the lane's measured free bytes
    less what other executors on the lane hold outstanding). `allocatable` is what this
    process measures it can actually get. Predictions authorize, observations verify: the
    ceiling is the smaller of the two, capped at the artifact's own complete destination
    bytes so a card with room reports the artifact's size rather than its own. Nothing here
    derives a ceiling from free bytes alone any more — that is how two executors preparing
    on one device were each handed the same headroom. Whatever it comes to is what the
    generation's digest carries; zero is a typed shortfall at the caller.
    """
    return min(int(destinations), max(int(authorized), 0), max(int(allocatable), 0))


def settled_capacity(*, resident_budget: int, fill_overhead: int) -> int:
    """What a rank holds for this construction once its fill has settled.

    `resident_budget` is the FILL ceiling: the card less the widest component's fill
    transient, which for H3's fp8 DiT (float destinations beside the encoded payload) is
    ~20 GB. Placing a component on a rank is a steady-state question, so that transient
    is given back; priced against the fill ceiling, H3's text encoder was never hosted on
    an H100 (runs 1435-1441).
    """
    return int(resident_budget) + int(fill_overhead)


# --------------------------------------------------------------------------- state


class _CompositeResidency:
    """Joint view over independently exact model-slot residencies.

    Each slot may evict only its own components. A sibling model's residency is
    never offered as capacity to another slot; sequential preparation therefore
    admits only the conservative all-slot resident sum plus each slot's own rung.
    """

    def __init__(self, rows: Mapping[str, Any]) -> None:
        self.rows = dict(rows)

    @property
    def poisoned(self) -> str:
        return next((str(row.poisoned) for row in self.rows.values() if row.poisoned), "")

    @property
    def last_shortfall(self) -> Any:
        return next(
            (row.last_shortfall for row in self.rows.values() if row.last_shortfall is not None),
            None,
        )

    def open_attempt(
        self,
        placement: str,
        headroom_bytes: int,
        scope_headroom_bytes: dict[str, int] | None = None,
        measured: tuple[str, ...] | frozenset[str] = (),
    ) -> None:
        for name, row in self.rows.items():
            prefix = name + "."
            scoped = {
                key.removeprefix(prefix): value
                for key, value in (scope_headroom_bytes or {}).items()
                if key.startswith(prefix)
            }
            mine = tuple(key.removeprefix(prefix) for key in measured if key.startswith(prefix))
            row.open_attempt(placement, headroom_bytes, scoped, mine)

    def vacate(self) -> dict[str, Any]:
        """Every slot frees its own components; the reports merge under the slot prefix."""
        reports = {name: row.vacate() for name, row in sorted(self.rows.items())}
        prefix = len(reports) > 1

        def keyed(name: str, component: str) -> str:
            return f"{name}/{component}" if prefix else str(component)

        return {
            "freed": {
                keyed(name, component): int(size)
                for name, report in reports.items()
                for component, size in report["freed"].items()
            },
            "held": {
                keyed(name, component): str(why)
                for name, report in reports.items()
                for component, why in report["held"].items()
            },
            "freed_bytes": sum(int(report["freed_bytes"]) for report in reports.values()),
            "allocator_delta_bytes": sum(
                int(report["allocator_delta_bytes"]) for report in reports.values()
            ),
            "vacate_ms": round(sum(float(report["vacate_ms"]) for report in reports.values()), 2),
        }

    def restore(self, names: tuple[str, ...] = ()) -> dict[str, Any]:
        """Every slot re-fills what it can without evicting; the reports merge as above."""
        prefix = len(self.rows) > 1
        reports = {}
        for name, row in sorted(self.rows.items()):
            local = (
                tuple(
                    value.removeprefix(name + "/")
                    for value in names
                    if value.startswith(name + "/")
                )
                if prefix
                else names
            )
            # An empty selection means "restore everything" to a child, so do not call
            # unselected models when the worker named specific global components.
            if names and not local:
                continue
            reports[name] = row.restore(names=local)

        def keyed(name: str, component: str) -> str:
            return f"{name}/{component}" if prefix else str(component)

        return {
            "restored": {
                keyed(name, component): int(size)
                for name, report in reports.items()
                for component, size in report["restored"].items()
            },
            "held": {
                **{
                    value: "this generation has no checkpoint for it"
                    for value in names
                    if prefix and ("/" not in value or value.partition("/")[0] not in self.rows)
                },
                **{
                    keyed(name, component): str(why)
                    for name, report in reports.items()
                    for component, why in report["held"].items()
                },
            },
            "restored_bytes": sum(int(report["restored_bytes"]) for report in reports.values()),
            "allocator_delta_bytes": sum(
                int(report["allocator_delta_bytes"]) for report in reports.values()
            ),
            "restore_ms": round(sum(float(report["restore_ms"]) for report in reports.values()), 2),
        }

    def document(self) -> dict[str, Any]:
        documents = {name: row.document() for name, row in sorted(self.rows.items())}
        prefix = len(documents) > 1
        resident = {
            f"{name}/{component}" if prefix else str(component): int(size)
            for name, document in documents.items()
            for component, size in document.get("resident", {}).items()
        }
        evicted = {
            f"{name}/{component}" if prefix else str(component): int(size)
            for name, document in documents.items()
            for component, size in document.get("evicted", {}).items()
        }
        activation_peaks = {
            f"{name}.{method}" if prefix else str(method): int(value)
            for name, document in documents.items()
            for method, value in (document.get("attempt_activation_peaks") or {}).items()
        }
        scope_headrooms = {
            f"{name}.{method}" if prefix else str(method): int(value)
            for name, document in documents.items()
            for method, value in (document.get("scope_headroom_bytes") or {}).items()
        }
        return {
            "placement": next(
                (document.get("placement", "") for document in documents.values()), ""
            ),
            "headroom_bytes": max(
                (int(document.get("headroom_bytes", 0)) for document in documents.values()),
                default=0,
            ),
            "scope_headroom_bytes": scope_headrooms,
            "admissions": sum(int(row.get("admissions", 0)) for row in documents.values()),
            "stages": sum(int(row.get("stages", 0)) for row in documents.values()),
            "evictions": sum(int(row.get("evictions", 0)) for row in documents.values()),
            "attempt_stages": sum(int(row.get("attempt_stages", 0)) for row in documents.values()),
            "attempt_evictions": sum(
                int(row.get("attempt_evictions", 0)) for row in documents.values()
            ),
            # THE DURATIONS TRAVEL WITH THEIR COUNTS (cr-134). The fold carried the counts
            # and dropped the milliseconds, so a multi-model placement's staged rung
            # confessed "N stage(s) costing 0 ms". Attempt-scoped and cumulative both, so
            # a reader can never take one for the other.
            "attempt_staged_ms": round(
                sum(float(row.get("attempt_staged_ms", 0.0)) for row in documents.values()), 2
            ),
            "attempt_evicted_ms": round(
                sum(float(row.get("attempt_evicted_ms", 0.0)) for row in documents.values()), 2
            ),
            "staged_ms": round(
                sum(float(row.get("staged_ms", 0.0)) for row in documents.values()), 2
            ),
            "evicted_ms": round(
                sum(float(row.get("evicted_ms", 0.0)) for row in documents.values()), 2
            ),
            "attempt_activation_peak_bytes": max(
                (int(row.get("attempt_activation_peak_bytes", 0)) for row in documents.values()),
                default=0,
            ),
            "attempt_activation_peaks": activation_peaks,
            "attempt_absolute_peak_bytes": max(
                (int(row.get("attempt_absolute_peak_bytes", 0)) for row in documents.values()),
                default=0,
            ),
            "allocator_delta_bytes": sum(
                int(row.get("allocator_delta_bytes", 0)) for row in documents.values()
            ),
            "resident": resident,
            "evicted": evicted,
            "poisoned": self.poisoned,
            "cross_model_eviction": False,
            "models": documents,
        }


#: The runtime modules a construction imports, named ONCE so `start` can warm
#: exactly the set `_prepare_first` re-imports (cr-104). Warming a name this tuple omits
#: is a missed saving and never a wrong answer: the prepare imports what it needs either
#: way, and a warm `sys.modules` is the only difference.
_CONSTRUCTION_MODULES = (
    "cozy_runtime.author",
    "cozy_runtime.author._loader",
    "cozy_runtime.author._model",
    "cozy_runtime.internal.anima_optimization",
    "cozy_runtime.internal.derive",
    "cozy_runtime.internal.encoding",
    "cozy_runtime.internal.fill",
    "cozy_runtime.internal.fusion",
    "cozy_runtime.internal.planfacts",
    "cozy_runtime.internal.warm",
    "cozy_runtime.internal.probe",
    "cozy_runtime.internal.residency",
    "cozy_runtime.internal.resolution",
)


def _local_attention_pin(pin: str, parameters: tuple[str, ...]) -> str:
    scope, _ = attention._pin_parts(pin)
    if "/" in scope and scope.split("/", 1)[0] not in parameters:
        raise attention.AttentionRefusal(
            "attention_kernel_unsupported", f"{pin} matches no model in {list(parameters)}"
        )
    return attention.for_model(pin, parameters)


class Executor:
    """One epoch's mutable state. Everything here dies with the process.

    The PROCESS is one instance (`host is self`); each loaded construction is a child
    instance holding that construction's generation (`host` is the process). Process facts
    - the seam, the group, torch, the discovered package - live on the host and are copied
    into a child when it is made; generation facts live only on the child.
    """

    def __init__(
        self,
        channel: Channel,
        root: Path,
        fill_forensics_dir: str = "",
        *,
        rank: int = 0,
        world: int = 1,
        host: Executor | None = None,
        construction: str = "",
    ) -> None:
        self.channel = channel
        self.root = root
        self.host: Executor = host or self
        #: the construction key this generation was loaded under; "" on the host
        self.construction = construction
        #: HOST ONLY: every loaded construction, the active one, and the LRU arbiter
        self.constructions: dict[str, Executor] = {}
        self.arbiter = ConstructionArbiter()
        self.active = ""
        self.started = False
        self.package_warmed = False
        #: rank 0 only: each follower's own start ledger, in rank order
        self.follower_stages: list[list[Any]] = []
        #: the process-wide device ceiling the worker last assigned (load/activate)
        self.authorized = 0
        #: this generation's load reply and the constructions its load parked
        self.load_reply: dict[str, Any] = {}
        self.load_parked: list[str] = []
        #: follower only: the attempt spool and allocator bytes at its first mirrored call
        self._run_entry: tuple[Path, int] | None = None
        #: follower only: the attention its processors held at that first call, and its kernels
        self._run_attention = ""
        self._run_served: tuple[str, ...] = ()
        #: HOST ONLY: this process's start, for the execution record (wall ms and its legs)
        self.boot: dict[str, Any] = {}
        #: HOST ONLY: this rank's pid, host ordinal, GPU uuid and arch, read once after device init
        self._identity: dict[str, Any] | None = None
        #: THIS process's rank in its generation's group and the group's size (cr-068). A
        #: plain executor is rank 0 of a world of 1 and forms nothing; rank 0 of a wider
        #: world spawns and drives the followers; a follower's channel is rank 0's seam.
        self.rank = int(rank)
        self.world = int(world)
        self.group: RankGroup | None = None
        self.pg: Any = None
        self._device_ready = False
        self._startup_reported = False
        #: the attempt spool while an attempt runs: where a mirrored call's payloads go
        self._attempt_spool: Path | None = None
        #: rank 0 only: the attempt whose cancel marker the mirror reads before each call
        self._attempt_request = ""
        #: cr-038: the fill plane's capture directory, taken from the sealed environment
        #: snapshot at process entry and passed down — never read ambiently.
        self.fill_forensics_dir = fill_forensics_dir
        self.torch: Any = None
        #: The BACKEND this epoch serves. Every device operation below goes through
        #: `accel` with it, so the whole process holds exactly one backend conditional and
        #: it is this line (#447). A weightless executor never reaches a device at all.
        self.device_kind = accel.host_backend_family()
        self.registry: Any = None
        self.backend: Any = None
        self.residency: Any = None
        self.model_executors: list[Executor] = []
        self.models: dict[str, Any] = {}
        self.group_components: dict[tuple[str, str], tuple[Any, Any]] = {}
        self.group_sharded: set[tuple[str, str]] = set()
        #: component -> the follower rank that keeps it resident for the whole group
        self.group_hosted: dict[tuple[str, str], int] = {}
        #: the many-model construction this per-model generation belongs to, if any
        self._group_parent: Executor | None = None
        self._group_model_key = ""
        self._group_attempts: dict[str, Path] = {}
        self.surface_names: dict[str, Any] = {}
        self.discovered: Any = None
        self.ready = False
        self.poisoned = ""
        self.constructed_model_digest = ""
        #: What the runtime pinned on this generation's attention sites (cr-124). One BOOT
        #: line, not a per-attempt record: the choice is a function of device and image.
        self.attention: attention.Applied | None = None
        #: selects this construction again, when a kernel it skipped while compiling is ready
        self._reselect: Callable[[], attention.Applied] | None = None
        self.attention_defaults = attention.Snapshot(())
        #: Device facts for this prepared generation, reused for per-request pin checks.
        self.device_facts: Any = None
        #: a follower's hold of rank 0's attention selection for the current attempt
        self._attention_hold = contextlib.ExitStack()
        self.facts: dict[str, Any] = {}
        self.schema_digests: dict[str, str] = {}
        #: request_id -> the record `prepare_request` resolved, run by `invoke`
        self.prepared: dict[str, PreparedRequest] = {}
        self.attempts = 0
        #: the DURABLE exchange counter (cr-009). Worker-answered, never sheddable.
        self._exchange = 0
        #: one exchange at a time: a publish from a joining thread and a child poll from the
        #: event loop share the seam, and each answer belongs to its own request.
        self._durable_lock = threading.Lock()

    # ------------------------------------------------------------------ prepare

    def handle(self, command: KnownCommand) -> dict[str, Any]:
        """Answer one decoded worker command (`shutdown` ends the loop before this)."""
        match command:
            case Hello():
                return self.hello()
            case Join():
                return self.join(command)
            case Attention():
                return self.select_attention(command)
            case DescribeInstalled():
                return self.describe_installed(command)
            case Start():
                return self.start(command)
            case Warm():
                return self.warm(command)
            case Load():
                return self.load(command)
            case Activate():
                return self.activate(command)
            case Unload():
                return self.unload(command)
            case Residency():
                return self.residency_document()
            case PrepareRequest():
                return self.serve_request(command)
            case Invoke():
                return self.serve_invoke(command)
            case RunJob():
                return self.run_job(command)
            case Probe():
                return self.probe()
            case Vacate():
                return self.vacate(residents=command.residents, ranks=command.ranks)
            case Restore():
                return self.restore(names=command.names)
            case Shutdown():
                raise AssertionError("shutdown ends the command loop")

    def hello(self) -> dict[str, Any]:
        return {
            "ok": True,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "python_version": platform.python_version(),
            "python_abi": (sys.implementation.cache_tag or "").replace("cpython-", "cp"),
            "python_executable": sys.executable,
            "runtime_version": RUNTIME_VERSION,
            "tensorfs_version": _distribution_version("tensorfs"),
            "executor_protocol_revision": EXECUTOR_PROTOCOL_REVISION,
            "native_interfaces": native_interfaces.stated(),
            "model_adapters": ["peft_graph/1"],
            #: what the worker's memory manager may ask beyond the floor's commands
            "memory": ["vacate_ranks"],
            "pgid": os.getpgrp(),
            "no_new_privs": _proc_status("NoNewPrivs"),
            "oom_score_adj": Path("/proc/self/oom_score_adj").read_text().strip(),
            "uid": os.getuid(),
            "euid": os.geteuid(),
            "sealed": dict(_SEALED),
            "torch_loaded": "torch" in sys.modules,
            "rss_bytes": _rss(),
            "rank": self.rank,
            "world": self.world,
        }

    def describe_installed(self, command: DescribeInstalled) -> dict[str, Any]:
        """Import one installed package and write its interface to brokered scratch."""
        output = Path(command.output)
        if not command.distribution or not output.is_absolute():
            return {"ok": False, "code": "package_prepare_path_invalid", "detail": "paths"}
        # Worker supervises this executor by observed progress and death. An import
        # slowed by host pressure is not a package conformance failure after 60 seconds.
        discovered = (
            discover_installed(command.application, seconds=None)
            if command.application
            else discover_distribution(command.distribution, seconds=None)
        )
        body = package_interface.build(discovered)
        raw = package_interface.canonical_bytes(body)
        with output.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        output.chmod(0o400)
        self.discovered = discovered
        self.surface_names = {surface.name: surface for surface in discovered.surfaces}
        return {
            "ok": True,
            "application": discovered.application,
            "length": len(raw),
        }

    # ------------------------------------------------------------- the group (cr-068)

    def _initialize_device(self, torch: Any) -> dict[str, Any] | None:
        """Runtime's ONE explicit accelerator init boundary, idempotent per process.

        Under a group every rank pins its own card: rank r of a seal naming K entries is
        card r of that seal, made current before any allocation. A CUDA context that
        already exists when this runs is a refusal (`cuda_initialized_before_runtime`) -
        something else picked a device first.
        """
        owner = self.host
        if not owner._startup_reported:
            # Observe before the explicit boundary on every rank, including a
            # previously adopted context. Stderr survives even a pre-reply crash.
            print(
                "[executor] prepare startup: "
                + json.dumps(
                    prepare_diagnostics.startup_facts(
                        rank=self.rank, world=self.world, device_kind=self.device_kind
                    ),
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            owner._startup_reported = True
        if self._device_ready:
            return None
        if accel.initialized(torch, self.device_kind):
            return {
                "ok": False,
                "code": "cuda_initialized_before_runtime",
                "detail": "Torch CUDA was already initialized before Runtime's explicit "
                "allocator boundary; package execution is refused",
            }
        accel.initialize(torch, self.device_kind)
        if self.world > 1 and self.device_kind == "cuda":
            torch.cuda.set_device(self.rank)
        accel.reset_peak(torch, self.device_kind)
        self._device_ready = True
        return None

    def _rank_identity(self) -> dict[str, Any]:
        owner = self.host
        if owner._identity is None:
            owner._identity = execution_evidence.identity(
                self.torch,
                self.device_kind == "cuda" and accel.present(self.torch, self.device_kind),
                self.rank if self.world > 1 else 0,
                _sealed_entries(),
            )
        return owner._identity

    def _execution(
        self,
        pin: str,
        choices: attention.Applied | None,
        start_us: int,
        end_us: int,
        impl: str = "",
    ) -> dict[str, Any]:
        """The attempt's execution record: degree, every rank, and the setup it ran on."""
        ranks = [
            execution_evidence.record(
                0,
                self._rank_identity() if self.torch is not None else {"pid": os.getpid()},
                start_us,
                end_us,
                execution_evidence.attention(
                    pin,
                    execution_evidence.observed(choices.hosts) if choices else "",
                    impl,
                    attention.evidence(choices.totals() if choices else ()),
                ),
            )
        ]
        if self.group is not None:
            for follower in self.group.followers:
                row = self.group.rank_records.get(follower.rank) or execution_evidence.record(
                    follower.rank, {"pid": follower.pid}, 0, 0
                )
                ranks.append({**row, "attention": {**row["attention"], "requested": pin}})
        facts = self.facts
        return {
            "degree": self.world,
            "ranks": ranks,
            # per spread method: the calls each rank ran, and why a rank declined
            "spread": dict(self.group.spread_log) if self.group is not None else {},
            "executor": dict(self.host.boot),
            "construction": {
                "prepared_unix_ms": int(facts.get("prepared_unix_ms", 0)),
                "prepare_ms": float(facts.get("prepare_ms", 0)),
                "fill_ms": float(facts.get("fill_ms", 0)),
                "warm_ms": float(facts.get("warm_ms", 0)),
                "legs_ms": dict(facts.get("stages") or ()),
            },
        }

    def _rank_device(self, torch: Any) -> Any:
        if self.device_kind == "cuda" and accel.present(torch, self.device_kind):
            return torch.device("cuda", self.rank if self.world > 1 else 0)
        return torch.device("cpu")

    def join(self, command: Join) -> dict[str, Any]:
        """A FOLLOWER joins rank 0's store and forms its half of the process group."""
        if self.rank == 0:
            return {"ok": False, "code": "not_a_follower", "detail": f"{_gpu(0)} forms the group"}
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        _restore_seal()
        import torch

        self.torch = torch
        spec = RankSpec(
            rank=self.rank,
            world=self.world,
            port=command.port,
            backend=command.backend or "nccl",
        )
        if spec.backend == "nccl":
            # This rank's card becomes current HERE, through the one init boundary: the
            # NCCL communicator binds to the current device. A gloo group (the CPU stand-in
            # the suite proves rendezvous on) touches no device.
            refused = self._initialize_device(torch)
            if refused is not None:
                self.poisoned = f"join refused: {refused['code']}"
                return refused
        try:
            self.pg = join_rank(torch, spec, _SEALED)
        except Exception as exc:
            self.poisoned = f"join failed: {type(exc).__name__}"
            return {
                "ok": False,
                "code": "group_unformed",
                "detail": f"{_gpu(self.rank)} could not join the group: "
                f"{type(exc).__name__}: {exc}"[:900],
                "traceback": prepare_diagnostics.exception_trace(exc),
            }
        return {"ok": True, "rank": self.rank, "world": self.world, "backend": spec.backend}

    def run(self, command: dict[str, Any]) -> dict[str, Any]:
        """Run a prepared sharded component, or one this rank hosts, under its leader's
        scope and execution modes. A hosted call's result returns through the spool."""
        if self.rank == 0:
            return {
                "ok": False,
                "code": "not_a_follower",
                "detail": f"{_gpu(0)} leads the group's calls; it is not sent them",
            }
        if not self.ready or self.torch is None:
            return {"ok": False, "code": "executor_not_ready", "detail": "no generation is Ready"}
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        named = command.get("component")
        if (
            not isinstance(named, list)
            or len(named) != 2
            or not all(isinstance(part, str) for part in named)
        ):
            return {
                "ok": False,
                "code": "gpu_divergence",
                "detail": "component identity is not a model/root pair",
            }
        name = (named[0], named[1])
        target = self.group_components.get(name)
        hosted = self.group_hosted.get(name) == self.rank
        spread = "method" in command
        if target is None or not (hosted or spread or name in self.group_sharded):
            self.poisoned = "mirrored call names no prepared sharded component"
            return {
                "ok": False,
                "code": "gpu_divergence",
                "detail": f"{_gpu(self.rank)} holds no sharded component {name!r}",
                "rank": self.rank,
            }
        _model, component = target
        sol_counts: dict[str, int] = {}
        result: Any = None
        models = {key[0]: model for key, (model, _root) in self.group_components.items()}
        started_us = execution_evidence.now_us()
        homes = any(rank == self.rank for rank in self.group_hosted.values())
        # A spread call rank 0 waits on over the group: a header always answers it.
        answered = False
        try:
            spool = Path(command["spool"])
            if self._run_entry is None or self._run_entry[0] != spool:
                accel.reset_peak(self.torch, self.device_kind)
                self._run_entry = (spool, accel.allocated(self.torch, self.device_kind))
                held = attention.observed(
                    {f"{m}/{c}": root for (m, c), (_, root) in self.group_components.items()}
                )
                self._run_attention = execution_evidence.observed(held.hosts)
                self._run_served = tuple(held.totals())
            scopes = command["scopes"]
            if not isinstance(scopes, list) or not 1 <= len(scopes) <= len(models):
                raise GroupRefusal("mirrored scope set is not a bounded prepared model set")
            seen: set[str] = set()
            active: dict[str, tuple[str, ...]] = {}
            with contextlib.ExitStack() as stack:
                for scope in scopes:
                    key = scope["model"]
                    if key not in models or key in seen:
                        raise GroupRefusal("mirrored scope names an unknown or duplicate model")
                    seen.add(key)
                    components = tuple(scope["components"])
                    if not components or any(
                        (key, root) not in self.group_components for root in components
                    ):
                        raise GroupRefusal("mirrored scope names no prepared component")
                    residency = models[key]._cozy_residency
                    if residency is not None and self._group_attempts.get(key) != spool:
                        # A rank that hosts a component holds a set rank 0's plan never
                        # priced, so it may always make room: demand-pull when measured.
                        residency.open_attempt(
                            "component_staged" if homes else scope["placement"],
                            scope["headroom_bytes"],
                            scope["scope_headroom_bytes"],
                            tuple(scope.get("measured_scopes") or ()),
                        )
                        self._group_attempts[key] = spool
                    if spread and residency is not None and (homes or command.get("spare")):
                        # An optional call: taken on room alone, declined rather than evict.
                        stack.enter_context(residency.sparing())
                    stack.enter_context(models[key]._cozy_scope(scope["method"], components))
                    active[key] = components
                if name[0] not in active or name[1] not in active[name[0]]:
                    raise GroupRefusal("sharded component has no active owning scope")
                references = {
                    key: root
                    for key, (_owner, root) in self.group_components.items()
                    if key[0] in active and key[1] in active[key[0]]
                }
                # Stage the same scopes before allocating the incoming activations.
                stack.enter_context(self.torch.inference_mode(command["inference_mode"]))
                stack.enter_context(self.torch.set_grad_enabled(command["grad_enabled"]))
                if not (hosted or spread):
                    stack.enter_context(cp.gated_call())
                for kind, dtype in (command.get("autocast") or {}).items():
                    stack.enter_context(
                        self.torch.autocast(device_type=kind, dtype=getattr(self.torch, dtype))
                    )
                stack.enter_context(wire.attention_context(command))
                sol_counts = stack.enter_context(attention_sol.observing())
                served = stack.enter_context(attention_ulysses.observing())
                args, kwargs = wire.run_call(
                    command,
                    wire.TensorSpool(spool / str(command.get("payloads", "ranks")), references),
                    device=self._rank_device(self.torch),
                )
                if spread:
                    # A pure method call whose one tensor result goes straight to rank 0.
                    output = getattr(component, str(command["method"]))(*args, **kwargs)
                    answered = True
                    wire.send_tensor(output, 0, self.pg)
                elif hosted:
                    plane = models[name[0]]._cozy_residency
                    call = component.get_submodule(str(command.get("module", "")))
                    try:
                        output = call(*args, **kwargs)
                    except accel.oom_error(self.torch, self.device_kind):
                        # The call has no effect beside its result: free what the scope
                        # does not hold and run it once more before refusing.
                        if plane is None:
                            raise
                        plane.shed()
                        accel.release_cached(self.torch, self.device_kind)
                        output = call(*args, **kwargs)
                    result = wire.marshal(
                        output, wire.TensorSpool(spool / str(command.get("results", "hosted")))
                    )
                else:
                    component(*args, **kwargs)
                accel.synchronize(self.torch, self.device_kind)
            self._attempt_spool = spool
            moved = sum(
                max(int(models[key]._cozy_residency.document().get("allocator_delta_bytes", 0)), 0)
                for key in active
                if models[key]._cozy_residency is not None
            )
            working = max(
                accel.peak_allocated(self.torch, self.device_kind) - self._run_entry[1] - moved, 0
            )
        except Exception as exc:
            if spread and not answered:
                declined = self._declines(exc)
                with contextlib.suppress(Exception):
                    wire.send_nothing(
                        0, self.pg, self._rank_device(self.torch), failed=declined is None
                    )
                if declined is not None:
                    # Nothing moved that this rank must keep: the call is pure and runs on
                    # another rank. The generation stays Ready.
                    return {
                        "ok": True,
                        "rank": self.rank,
                        "component": list(name),
                        "sol_calls": {},
                        "working_peak_bytes": 0,
                        "declined": declined,
                    }
            outcome = classify(exc)
            self.poisoned = f"mirrored {name} failed: {outcome.code}"
            return {
                "ok": False,
                "code": outcome.code,
                "detail": f"{type(exc).__name__}: {exc}"[:900],
                "rank": self.rank,
            }
        return {
            "ok": True,
            "rank": self.rank,
            "component": list(name),
            "sol_calls": sol_counts,
            "working_peak_bytes": working,
            **({"result": result} if hosted else {}),
            "evidence": execution_evidence.record(
                self.rank,
                self._rank_identity(),
                started_us,
                execution_evidence.now_us(),
                execution_evidence.attention(
                    observed=self._run_attention,
                    impl=attention.implementations(served),
                    kernels=attention.evidence(self._run_served),
                ),
            ),
        }

    def _declines(self, exc: Exception) -> str | None:
        """Why a spread call is declined rather than failed: no room without evicting
        (`device_spared`), or the device ran out inside the call."""
        if isinstance(exc, ResidencyRefusal) and exc.code == "device_spared":
            return exc.detail
        if self.device_kind == "cuda" and isinstance(
            exc, accel.oom_error(self.torch, self.device_kind)
        ):
            accel.release_cached(self.torch, self.device_kind)
            return f"device_out_of_memory: {exc}"[:300]
        return None

    def _group_refusal(self, exc: GroupRefusal, *, closing: bool = True) -> dict[str, Any]:
        """A group that could not form or hold is THIS process's poison: the worker
        rebuilds the whole cgroup, never a partial group."""
        self.poisoned = f"group: {exc.code}"
        if closing and self.group is not None:
            self.group.close()
        return {"ok": False, "code": exc.code, "detail": exc.message[:900]}

    def _install_group(self, torch: Any, model: Any, capacity: int = 0) -> dict[str, Any] | None:
        """Every rank shards its replica, takes its side of the mirror, and agrees which
        follower keeps each placeable component that rank 0 has no room for. `capacity` is
        the group's agreed settled capacity (`settled_capacity`)."""
        if self.world <= 1:
            return None
        owner = self.host
        roots = {**self.backend.components, **self.backend.parked}
        for component, root in roots.items():
            owner.group_components[(self._group_model_key, component)] = (model, root)
        candidates = cp.sharding_candidates(model)
        # An explicitly group-compatible auxiliary model can be replicated beside a
        # sharded model. The complete multi-model prepare still requires a sharded root.
        installed: tuple[str, ...] = ()
        if candidates or self._group_parent is None:
            comms = cp.CpComms(self.pg, self.rank, self._rank_device(torch))
            try:
                installed = cp.install_context_parallel(model, degree=self.world, comms=comms)
            except GroupRefusal as exc:
                return self._group_refusal(exc)
        for _label, component in candidates:
            component_name = next((name for name, root in roots.items() if root is component), None)
            if component_name is None:
                return self._group_refusal(
                    GroupRefusal("a sharded component is not a prepared root")
                )
            name = (self._group_model_key, component_name)
            owner.group_sharded.add(name)
            if self.rank == 0:
                assert self.group is not None
                mirror_component(
                    component,
                    name=name,
                    prepared=owner.group_components,
                    group=self.group,
                    spool=lambda: owner._attempt_spool,
                    cancelled=lambda: (
                        bool(owner._attempt_request)
                        and _cancelled(owner.root, owner._attempt_request)
                    ),
                )
        hosted = mirror.hosting_plan(
            placeable=tuple(getattr(type(model), "__placeable__", ())),
            sizes={
                name: int(getattr(self.backend, "component_bytes", {}).get(name, 0))
                for name in roots
            },
            sharded={name[1] for name in owner.group_sharded if name[0] == self._group_model_key},
            scopes=component_use(type(model)),
            capacity=capacity,
            world=self.world,
        )
        for component_name, home in hosted.items():
            name = (self._group_model_key, component_name)
            owner.group_hosted[name] = home
            if self.rank != home and self.residency is not None:
                self.residency.host(component_name)
            if self.rank == 0:
                assert self.group is not None
                mirror.host_component(
                    roots[component_name],
                    name=name,
                    prepared=owner.group_components,
                    group=self.group,
                    spool=lambda: owner._attempt_spool,
                    rank=home,
                    device=self._rank_device(torch),
                    cancelled=lambda: (
                        bool(owner._attempt_request)
                        and _cancelled(owner.root, owner._attempt_request)
                    ),
                )
        # Every rank may take a spread call. A rank homing a placeable component takes one
        # only beside everything it holds (measured room), so it never gives up the
        # component it hosts for a clip (`Executor.run`, `ComponentResidency.sparing`).
        sparing = sorted(set(owner.group_hosted.values()))
        if self.rank == 0:
            assert self.group is not None
            if hosted:
                object.__setattr__(
                    model, "_cozy_placement", mirror.GroupPlacement(hosted, self.group, torch)
                )
            for component_name, root in roots.items():
                name = (self._group_model_key, component_name)
                if name in owner.group_sharded or name in owner.group_hosted:
                    continue
                mirror.install_spread(
                    root,
                    name=name,
                    prepared=owner.group_components,
                    group=self.group,
                    spool=lambda: owner._attempt_spool,
                    device=self._rank_device(torch),
                    cancelled=lambda: (
                        bool(owner._attempt_request)
                        and _cancelled(owner.root, owner._attempt_request)
                    ),
                )
        self.facts["sequence_parallel"] = {
            "degree": self.world,
            "rank": self.rank,
            "components": list(installed),
            "hosted": hosted,
            "spread_ranks": list(range(self.world)),
            "sparing_ranks": sparing,
            "followers": list(self.group.pids()) if self.group is not None else [],
        }
        return None

    def _warm_model(self, model: Any) -> float:
        """Preparation has its own spool; component calls need no request to warm."""
        from cozy_runtime.internal.warm import warm_generation

        # Warm has no admitted request rung. Its declared component-use scopes must
        # free unrelated weights before kernel and collective scratch is allocated.
        # The mirror carries this same existing staged contract to every follower.
        if self.residency is not None:
            self.residency.open_attempt("component_staged", 0)
        owner = self.host
        owner._attempt_spool = (
            self.root / "warm" / hashlib.sha256(self._group_model_key.encode()).hexdigest()
        )
        try:
            return warm_generation(model, device=self._device(), cancel=lambda: bool(self.poisoned))
        finally:
            owner._attempt_spool = None

    def _warm_package(self) -> str:
        """Give the package its ONE chance to pay its deferred import cost off the lock.

        A package's `__init__` is deliberately lazy, and that is right for every caller but
        this one: deferred, the cost lands inside the construction, under the device lock,
        on the far side of the vacate. A module-level `warmup()` is where a package says
        what it would rather pay early. It takes no arguments, returns nothing, and MUST
        NOT touch the device -- another tenant owns the card while this runs.
        """
        if self.package_warmed:
            return ""
        self.package_warmed = True
        module = getattr(self.discovered, "module", None)
        warm = getattr(module, "warmup", None)
        if not callable(warm):
            return ""
        try:
            warm()
        except Exception as exc:
            # A WARMUP IS AN OPTIMIZATION AND NEVER A VERDICT. Whatever it failed to do the
            # construction does again for real under the lock and refuses there with the
            # real code; poisoning the generation here would turn a missed saving into a
            # lost placement.
            return f"{type(exc).__name__}: {exc}"[:200]
        return ""

    # ------------------------------------------------------------ the process (proto-061)

    def start(self, command: Start) -> dict[str, Any]:
        """Bring EVERY rank up before any construction: seal, group, torch, package, warmup.

        Rank 0 spawns its followers first and hands them this command, so every rank's
        Torch, runtime and package imports and the package `warmup()` run in parallel; the
        group then forms over the loopback store. Nothing here moves a weight byte. A
        package whose interface binds no model on any entrypoint imports no Torch at all,
        which is what keeps a weightless package servable on a cardless runner (cl-013).
        Idempotent: a started process answers with what it started as.
        """
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        if self.started:
            return self._start_reply([], reused=True)
        sealed_devices = _SEALED.get("CUDA_VISIBLE_DEVICES", "")
        if command.devices != sealed_devices:
            self.poisoned = "env seal names other devices"
            return {
                "ok": False,
                "code": "env_seal_broken",
                "detail": "this executor is sealed to CUDA_VISIBLE_DEVICES="
                f"{sealed_devices!r} and the start names {command.devices!r}: "
                "an executor serves exactly its lane's devices",
            }
        degree = command.sequence_parallel_degree or 1
        expected = self.world if self.rank > 0 else max(degree, 1)
        if degree != expected or (degree > 1 and degree != len(_sealed_entries())):
            self.poisoned = "sequence-parallel degree mismatch"
            return {
                "ok": False,
                "code": "sequence_parallel_degree_mismatch",
                "detail": f"the start names {degree} GPU(s); this process drives "
                f"{_gpu(self.rank)} of a {self.world}-GPU group sealed to "
                f"{len(_sealed_entries())} device(s) ({sealed_devices!r})",
            }
        _restore_seal()
        try:
            interface = package_interface.read_bytes(
                Path(command.package_interface).read_bytes(), command.package_interface
            )
        except (OSError, ValueError) as exc:
            self.poisoned = "start without a readable package interface"
            return {
                "ok": False,
                "code": "package_prepare_path_invalid",
                "detail": f"the start names no readable PackageInterface: {exc}"[:900],
            }
        entrypoints = interface.get("entrypoints") or []
        model_bearing = (
            any(isinstance(doc, dict) and doc.get("models") for doc in entrypoints)
            if isinstance(entrypoints, list)
            else False
        )
        if degree > 1 and not model_bearing:
            self.poisoned = "sequence-parallel construction unsupported"
            return {
                "ok": False,
                "code": "sequence_parallel_unsupported",
                "detail": "a group requires a model-bearing package",
            }
        started = time.perf_counter()
        stages: list[tuple[str, float]] = []
        _mark = [started]

        def stage(name: str) -> None:
            now = time.perf_counter()
            stages.append((name, round((now - _mark[0]) * 1000, 1)))
            _mark[0] = now

        try:
            if degree > 1 and self.rank == 0:
                self.world = degree
                if self.group is None:
                    self.group = RankGroup(
                        degree=degree,
                        backend="nccl" if self.device_kind == "cuda" else "cpu:gloo",
                        python=sys.executable,
                        root=str(self.root),
                        forward=self.channel.send,
                        devices=_sealed_entries(),
                    )
                if not self.group.followers:
                    self.group.spawn()
                    self.group.dial_back(_SEALED)
                self.group.broadcast(executor_commands.encode(command))
                stage("followers_spawned")
            if model_bearing:
                import torch

                self.torch = torch
                stage("torch_import")
                try:
                    refused = self._initialize_device(torch)
                except RuntimeError as exc:
                    refused = {
                        "ok": False,
                        "code": "accelerator_unavailable",
                        "detail": f"the sealed devices {_SEALED.get('CUDA_VISIBLE_DEVICES')!r} "
                        f"could not be initialized: {exc}"[:900],
                    }
                if refused is not None:
                    self.poisoned = f"start refused: {refused['code']}"
                    return self._close_after(refused)
                stage("cuda_init")
                for name in _CONSTRUCTION_MODULES:
                    importlib.import_module(name)
                stage("runtime_imports")
            if self.discovered is None:
                self._discover_provisioned(command.release())
            stage("package_import")
            warmup_error = self._warm_package()
            stage("package_warmup")
            if self.group is not None:
                replies = self.group.collect("start")
                self.follower_stages = [list(reply.get("stages") or []) for reply in replies]
                for follower, reply in zip(self.group.followers, replies, strict=True):
                    if not reply.get("ok"):
                        raise GroupRefusal(
                            f"{follower.gpu} refused the start: {reply.get('code')}: "
                            f"{reply.get('detail', '')}"[:800],
                            code="group_unformed",
                        )
                self.pg = self.group.form(self.torch, _SEALED)
                stage("group_form")
        except GroupRefusal as exc:
            return self._group_refusal(exc)
        except Exception:
            self.poisoned = self.poisoned or "start raised"
            if self.group is not None:
                self.group.close()
            raise
        self.started = True
        took_ms = (time.perf_counter() - started) * 1000
        self.boot = {
            "started_unix_ms": int(time.time() * 1000 - took_ms),
            "ms": round(took_ms, 1),
            "legs_ms": dict(stages),
        }
        return self._start_reply(stages, warmup_error=warmup_error)

    def _start_reply(
        self, stages: list[tuple[str, float]], *, reused: bool = False, warmup_error: str = ""
    ) -> dict[str, Any]:
        total = free = -1
        if self.torch is not None and accel.present(self.torch, self.device_kind):
            memory = accel.allocation(self.torch, self.device_kind)
            total, free = memory["driver_total_bytes"], memory["driver_free_bytes"]
        return {
            "ok": True,
            "stages": stages,
            "reused": reused,
            "follower_pids": list(self.group.pids()) if self.group is not None else [],
            "follower_stages": self.follower_stages,
            "device_total_bytes": total,
            "device_free_bytes": free,
            "torch": self.torch is not None,
            "warmed": not warmup_error,
            "warmup_error": warmup_error,
            "rss_bytes": _rss(),
        }

    def warm(self, command: Warm) -> dict[str, Any]:
        """The startup warm (h3a-087): a started process with nothing constructed starts, on
        every rank's own card, the compiles its first construction will want (the fused glue
        and every kernel of the model's preferred chain and the ranked table), so they run
        beside the weights' download. It returns at once: nothing here waits on a compile.

        An optimization and never a verdict: the construction selects again for real."""
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        if not self.started or self.torch is None or self.device_kind != "cuda":
            return {"ok": True, "ranks": [], "skipped": "no started device process"}
        started = time.perf_counter()
        try:
            if self.group is not None and self.rank == 0:
                self.group.broadcast(executor_commands.encode(command))
            rows = [self._warm_rank()]
            if self.group is not None and self.rank == 0:
                rows += [reply.get("rank_warm") or {} for reply in self.group.collect("warm")]
        except GroupRefusal as exc:
            return self._group_refusal(exc)
        if self.rank > 0:
            return {"ok": True, "rank_warm": rows[0]}
        took = (time.perf_counter() - started) * 1000
        if "started_unix_ms" in self.boot:
            # The boot record spans the warm too: `cozy run show` reads one executor boot.
            for name in ("fusion_compile", "attention_compile"):
                self.boot["legs_ms"][name] = max(r.get("legs_ms", {}).get(name, 0.0) for r in rows)
            self.boot["ms"] = round(time.time() * 1000 - self.boot["started_unix_ms"], 1)
        return {"ok": True, "ranks": rows, "ms": round(took, 1)}

    def _warm_rank(self) -> dict[str, Any]:
        from cozy_runtime.internal import fusion
        from cozy_runtime.internal.encoding import measure_device

        legs: dict[str, float] = {}
        mark = time.perf_counter()
        device = measure_device(self.torch, self.rank if self.world > 1 else 0)
        fused = ""
        classes = self._model_classes()
        if any(getattr(cls, "__fusion__", "refuse") in ("accept", "require") for cls in classes):
            fused = fusion.expect(self._rank_device(self.torch))[:200]
            legs["fusion_compile"] = round((time.perf_counter() - mark) * 1000, 1)
        mark = time.perf_counter()
        kernels = attention.expect(device, self._preferred_attention(classes, device))
        # Launch checks' scratch goes back to the driver: the card may be another call's.
        accel.release_cached(self.torch, self.device_kind)
        legs["attention_compile"] = round((time.perf_counter() - mark) * 1000, 1)
        return {
            "rank": self.rank,
            "legs_ms": legs,
            "kernels": kernels,
            **({"fusion": fused} if fused else {}),
        }

    def _model_classes(self) -> list[type[Model[Any]]]:
        module = getattr(self.discovered, "module", None)
        values = vars(module).values() if module is not None else ()
        return [v for v in values if isinstance(v, type) and issubclass(v, Model) and v != Model]

    def _preferred_attention(self, classes: list[type[Model[Any]]], device: Any) -> list[str]:
        """Each model class's stated preference, asked per declared component with the facts
        a site would carry. A class whose choice needs its constructed state says nothing."""
        order: list[str] = []
        for cls in classes:
            if cls.choose_attention is Model.choose_attention:
                continue
            components = {c for used in component_use(cls).values() for c in used}
            for component in sorted(components):
                context = AttentionContext(
                    component,
                    "",
                    device.kind,
                    device.name,
                    device.sm,
                    "bfloat16",
                    128,
                    self.world,
                    (),
                    "",
                    "",
                )
                try:
                    chosen = cls.choose_attention(object.__new__(cls), context)
                except Exception:
                    continue
                names = [chosen] if isinstance(chosen, str) else list(chosen or ())
                order += [name for name in names if isinstance(name, str) and name not in order]
        return order

    def _close_after(self, reply: dict[str, Any]) -> dict[str, Any]:
        """A refusal after followers were commanded leaves them mid-protocol: the process
        is poisoned and its group ends with it."""
        if self.host.group is not None and self.rank == 0:
            self.host.group.close()
        return reply

    def _construction_child(self, construction: str) -> Executor:
        """A new generation object for one construction, carrying the process facts."""
        child = Executor(
            self.channel,
            self.root,
            self.fill_forensics_dir,
            rank=self.rank,
            world=self.world,
            host=self.host,
            construction=construction,
        )
        child.group, child.pg = self.group, self.pg
        child.torch = self.torch
        child.device_kind = self.device_kind
        child._device_ready = self._device_ready
        child.discovered = self.discovered
        child.surface_names = dict(self.surface_names)
        return child

    def _refusal(self, code: str, detail: str) -> dict[str, Any]:
        return {"ok": False, "code": code, "detail": detail}

    def load(self, command: Load) -> dict[str, Any]:
        """Construct and fill ONE construction under its key; a loaded key answers again.

        The worker names the key (its construction key) and assigns the process-wide
        ceiling: `authorized_device_limit_bytes` bounds everything this process holds, so a
        construction fits under it less what the others already hold, after parking the
        least recently used of them if that is what makes room. A load refused before it
        touched the process leaves the other constructions serving; one that allocated,
        commanded followers or raised poisons the process.
        """
        if self.rank > 0:
            return self._follower_load(command)
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        if not self.started:
            return self._refusal("executor_not_started", "load before start")
        key = command.construction
        if not key:
            return self._refusal("construction_key_absent", "a load names its construction")
        loaded = self.constructions.get(key)
        if loaded is not None:
            loaded._bind_parameters(command)
            self.arbiter.touch(key)
            return self._load_reply(key, loaded, reused=True)
        if command.authorized_device_limit_bytes is not UNSET:
            self.authorized = command.authorized_device_limit_bytes
        child = self._construction_child(key)
        try:
            reply = child._prepare_first(command)
        except Exception:
            self.poisoned = self.poisoned or child.poisoned or "load raised"
            self._close_after({})
            raise
        if not reply.get("ok"):
            if not child.poisoned:
                # Refused before touching the process: its other constructions keep serving.
                return {**reply, "poisoned": ""}
            self.poisoned = self.poisoned or child.poisoned
            return self._close_after({**reply, "poisoned": self.poisoned})
        self._register(key, child, reply)
        return self._load_reply(key, child, reused=False)

    def _bind_parameters(self, command: Load) -> None:
        """A reload names the construction under every parameter its bindings now spell
        (h3a-018): an added entrypoint over loaded bytes binds the same model object."""
        members = self.model_executors or [self]
        for row, member in zip(command.rows(), members, strict=False):
            model = next(iter(member.models.values()), None)
            if model is None:
                continue
            for name in row.binding.parameter_names():
                member.models.setdefault(name, model)
                self.models.setdefault(name, model)

    def _register(self, key: str, child: Executor, reply: dict[str, Any]) -> None:
        child.load_reply = reply
        self.constructions[key] = child
        self.arbiter.add(key, Construction(residency=child.residency, need=child._restore_need))
        self.discovered = child.discovered
        self.surface_names = dict(child.surface_names)
        self.torch = child.torch or self.torch
        self._device_ready = child._device_ready or self._device_ready
        self.ready = True

    def _load_reply(self, key: str, child: Executor, *, reused: bool) -> dict[str, Any]:
        row = self.arbiter.rows[key]
        return {
            **child.load_reply,
            "construction": key,
            "reused": reused,
            "parked": [] if reused else list(child.load_parked),
            "resident_bytes": row.resident_bytes(),
        }

    def _follower_load(self, command: Load) -> dict[str, Any]:
        """Rank r's half of rank 0's load: one construction, or one model of a many-model
        construction (`model_key`), filled under rank 0's agreed plan and ceiling."""
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        key, model_key = command.construction, command.model_key
        if not key or (model_key is UNSET and key in self.constructions):
            self.poisoned = "GPU divergence"
            return self._refusal("gpu_divergence", f"{_gpu(0)} loaded an unnamed or loaded key")
        if model_key is UNSET:
            child = self._construction_child(key)
            reply = child._prepare_first(command)
            if not reply.get("ok"):
                self.poisoned = child.poisoned or f"load refused: {reply.get('code')}"
                return reply
            self._register(key, child, reply)
            return reply
        container = self.constructions.get(key)
        if container is None:
            container = self._construction_child(key)
        if any(member._group_model_key == model_key for member in container.model_executors):
            self.poisoned = "group model identity changed or was prepared twice"
            return self._refusal("gpu_divergence", self.poisoned)
        member = container._construction_child(key)
        member._group_parent = container
        container.model_executors.append(member)
        reply = member._prepare_first(command, optional_attention_scope=True)
        if not reply.get("ok"):
            self.poisoned = member.poisoned or f"load refused: {reply.get('code')}"
            return reply
        container.models.update(member.models)
        container.residency = _CompositeResidency(
            {
                next(iter(item.models)): item.residency
                for item in container.model_executors
                if item.residency is not None
            }
        )
        container.ready = True  # prepared roots can receive their leader's warm calls
        if key in self.constructions:
            self.arbiter.rows[key].residency = container.residency
        else:
            self._register(key, container, reply)
        return reply

    def _park(self, keys: list[str]) -> None:
        """Park exactly these constructions (a follower obeying rank 0's decision)."""
        for key in keys:
            if key not in self.arbiter.rows:
                raise GroupRefusal(
                    f"{_gpu(0)} parked {key!r}, which {_gpu(self.rank)} never loaded"
                )
            self._synchronize()
            self.arbiter.park(key)

    def _make_room(self, exclude: str, need: int) -> list[str]:
        """Park LRU constructions until `need` more bytes fit the ceiling and the card."""
        torch, kind = self.torch, self.device_kind
        if torch is None:
            return []

        def fits() -> bool:
            memory = accel.allocation(torch, kind)
            allocatable = memory["driver_free_bytes"] + max(
                memory["reserved_bytes"] - memory["allocated_bytes"], 0
            )
            return memory["allocated_bytes"] + need <= self.authorized and allocatable >= need

        self._synchronize()
        parked = self.arbiter.make_room(exclude, fits)
        if parked:
            accel.release_cached(torch, kind)
        if self.world > 1 or not fits():
            # A group asks for every rank: a follower's card may be short where this one
            # is not, and the worker frees each device of the lane until its tightest fits.
            self._room(need)
        return parked

    def _room(self, need: int) -> None:
        """Ask the worker for `need` allocatable bytes this process cannot free itself:
        idle co-tenants of its devices give theirs back. Rank 0 only; followers answer to
        rank 0."""
        host = self.host
        if host.rank != 0 or host.torch is None or host.device_kind != "cuda":
            return
        memory = accel.allocation(host.torch, host.device_kind)
        slack = max(memory["reserved_bytes"] - memory["allocated_bytes"], 0)
        answer = host._durable(DeviceRoom(free_bytes=need - slack), Room)
        if answer.ok and not isinstance(answer.authorized_device_limit_bytes, UnsetType):
            host.authorized = answer.authorized_device_limit_bytes

    def _synchronize(self) -> None:
        if self.torch is not None:
            accel.synchronize(self.torch, self.device_kind)

    def _restore_need(self, names: tuple[str, ...]) -> int:
        """Bytes to stage `names` back: their settled sum plus the widest one's fill
        transient. An fp8 leaf is filled as its bf16 destination beside the encoded payload
        before the swap, so `stage_bytes` (the component's fill peak) is what must fit
        while it arrives - pricing only the settled bytes is how a restore OOMs."""
        members = self.model_executors or [self]
        prefixed = len(members) > 1
        settled = transient = 0
        for member in members:
            backend = member.backend
            if backend is None:
                continue
            prefix = (next(iter(member.models)) + "/") if prefixed and member.models else ""
            for component in {*backend.components, *backend.parked}:
                if prefix + component not in names:
                    continue
                peak = int(backend.stage_bytes(component))
                resting = (
                    int(backend.paging[component].working_bytes)
                    if component in backend.paging
                    else int(backend.component_memory[component].settled)
                )
                settled += resting
                transient = max(transient, peak - resting)
        return settled + transient

    def activate(self, command: Activate) -> dict[str, Any]:
        """Make one loaded construction the resident, active one.

        A resident construction activates in 0 ms. A parked one first parks least recently
        used constructions until its whole parked set plus the widest component's fill
        transient fits the ceiling and the card, then stages that set back. Every rank
        parks and restores the same constructions: rank 0 decides and says which.
        """
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        key = command.construction
        if key not in self.constructions:
            return self._refusal(
                "construction_not_loaded", f"{key!r} is not loaded in this executor"
            )
        if command.authorized_device_limit_bytes is not UNSET:
            self.authorized = command.authorized_device_limit_bytes
        started = time.perf_counter()
        row = self.arbiter.rows[key]
        restored: RestoreReport = {"restored": {}, "held": {}, "restored_bytes": 0}
        try:
            if self.rank == 0:
                parked = self._make_room(key, self.arbiter.need(key)) if row.parked else []
                if self.group is not None:
                    self.group.broadcast(
                        executor_commands.encode(Activate(construction=key, park=tuple(parked)))
                    )
            else:
                parked = list(command.park)
                self._park(parked)
            if row.parked is not None:
                restored = self.arbiter.restore(key)
                self._synchronize()
            if self.rank == 0 and self.group is not None:
                for reply in self.group.collect("activate"):
                    if not reply.get("ok"):
                        raise GroupRefusal(
                            f"a follower refused to activate: {reply.get('code')}: "
                            f"{reply.get('detail', '')}"[:600],
                            code="group_broken",
                        )
        except GroupRefusal as exc:
            return self._group_refusal(exc, closing=False)
        except ResidencyRefusal as exc:
            self.poisoned = self.poisoned or f"activate: {exc.code}"
            return {"ok": False, "code": exc.code, "detail": exc.detail[:900]}
        self.arbiter.touch(key)
        self.active = key
        return {
            **self.probe(),
            "ok": True,
            "construction": key,
            "parked": parked,
            "restored_bytes": int(restored.get("restored_bytes", 0)),
            # what the block layer served for it: ~0 is a warm page cache
            "read_bytes": int(restored.get("read_bytes", -1)),
            "fill_ms": int(restored.get("fill_ms", 0)),
            "held": dict(restored.get("held") or {}),
            "resident_bytes": row.resident_bytes(),
            "ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def unload(self, command: Unload) -> dict[str, Any]:
        """Drop one construction: vacate it, release its leases, drop every reference."""
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        key = command.construction
        child = self.constructions.pop(key, None)
        if child is None:
            return {"ok": True, "construction": key, "freed_bytes": 0, "absent": True}
        try:
            if self.rank == 0 and self.group is not None:
                self.group.broadcast(executor_commands.encode(Unload(construction=key)))
            self._synchronize()
            before = accel.allocated(self.torch, self.device_kind) if self.torch else 0
            self.arbiter.remove(key)
            members = child.model_executors or [child]
            if child.residency is not None:
                child.residency.vacate()
            for member in members:
                if member.backend is not None:
                    member.backend.close()
            prefix = key + "#"
            for name in [name for name in self.group_components if name[0].startswith(prefix)]:
                del self.group_components[name]
            self.group_sharded = {n for n in self.group_sharded if not n[0].startswith(prefix)}
            self.group_hosted = {
                n: r for n, r in self.group_hosted.items() if not n[0].startswith(prefix)
            }
            for member in (*members, child):
                member.models.clear()
                member.prepared.clear()
                member.model_executors = []
                member.backend = member.residency = member.registry = None
            del child, members, member
            if self.active == key:
                self.active = ""
            gc.collect()
            if self.torch is not None:
                accel.release_cached(self.torch, self.device_kind)
            after = accel.allocated(self.torch, self.device_kind) if self.torch else 0
            if self.rank == 0 and self.group is not None:
                for reply in self.group.collect("unload"):
                    if not reply.get("ok"):
                        raise GroupRefusal(
                            f"a follower refused to unload: {reply.get('code')}: "
                            f"{reply.get('detail', '')}"[:600],
                            code="group_broken",
                        )
        except GroupRefusal as exc:
            return self._group_refusal(exc, closing=False)
        except ResidencyRefusal as exc:
            self.poisoned = self.poisoned or f"unload: {exc.code}"
            return {"ok": False, "code": exc.code, "detail": exc.detail[:900]}
        self.ready = bool(self.constructions)
        return {
            **self.probe(),
            "ok": True,
            "construction": key,
            "freed_bytes": max(before - after, 0),
        }

    def residency_document(self) -> dict[str, Any]:
        free = -1
        if self.torch is not None and accel.present(self.torch, self.device_kind):
            free = accel.allocation(self.torch, self.device_kind)["driver_free_bytes"]
        return {
            "ok": True,
            "constructions": self.arbiter.document(self.active),
            "active": self.active,
            "free_bytes": free,
        }

    def _device_bytes(self) -> int:
        """CUDA allocator bytes this process holds; -1 where no allocator can say (MPS)."""
        if self.torch is None or self.device_kind == "mps":
            return -1
        return int(self.torch.cuda.memory_allocated())

    def _holds_nothing(self, baseline: int) -> bool:
        """A refused single-rank load that allocated nothing leaves the process intact."""
        return self.world == 1 and 0 <= self._device_bytes() <= baseline

    def _discard_unallocated(self, backend: Any, baseline: int, poison: str) -> None:
        """Close a generation refused before it held device bytes; otherwise poison (#613)."""
        if self._holds_nothing(baseline):
            backend.close()
            return
        backend.poison(())
        self.poisoned = poison

    def _prepare_first(
        self, command: Load, *, optional_attention_scope: bool = False
    ) -> dict[str, Any]:
        """Construct and fill one construction (or one model of a many-model one)."""
        if self.poisoned:
            return {
                "ok": False,
                "code": "poisoned_generation",
                "detail": f"this executor is poisoned ({self.poisoned}); a poisoned "
                "generation has no recovery edge to Ready — the worker replaces it",
            }
        sealed_devices = _SEALED.get("CUDA_VISIBLE_DEVICES", "")
        if command.devices != sealed_devices:
            # THE SEAL IS THE DEVICE SELECTION (cr-066). The worker names the lane it
            # sealed this process to; a prepare for any other set of devices is a prepare
            # in the wrong process, refused before Torch can pick a card.
            self.poisoned = "env seal names other devices"
            return {
                "ok": False,
                "code": "env_seal_broken",
                "detail": "this executor is sealed to CUDA_VISIBLE_DEVICES="
                f"{sealed_devices!r} and the prepare names {command.devices!r}: "
                "an executor serves exactly its lane's devices",
            }
        # THE DEGREE (cr-068): the process started at K; every load names the same K, and a
        # mismatch refuses before any weight moves. Degree 1 (or absent) is a plain
        # executor, whatever the seal's width - a job's envelope lane.
        degree = command.sequence_parallel_degree or 1
        if degree != max(self.world, 1) or (degree > 1 and degree != len(_sealed_entries())):
            self.poisoned = "sequence-parallel degree mismatch"
            return {
                "ok": False,
                "code": "sequence_parallel_degree_mismatch",
                "detail": f"the load names {degree} GPU(s); this process drives "
                f"{_gpu(self.rank)} of a {self.world}-GPU group sealed to "
                f"{len(_sealed_entries())} device(s) ({sealed_devices!r})",
            }
        if self.rank > 0:
            # THE BROADCAST HARDWARE HALF: the degree and the devices are this rank's own
            # facts already, and a rank 0 that planned for other cards is divergence, not
            # something to fill under.
            hardware = command.group
            if hardware is UNSET or (hardware.degree, hardware.devices) != (
                self.world,
                _sealed_entries(),
            ):
                self.poisoned = "GPU divergence"
                return {
                    "ok": False,
                    "code": "gpu_divergence",
                    "detail": f"{_gpu(self.rank)} is sealed to {list(_sealed_entries())} in a "
                    f"{self.world}-GPU group; {_gpu(0)} broadcast "
                    + (
                        "no hardware plan"
                        if hardware is UNSET
                        else f"{hardware.degree} GPUs over {list(hardware.devices)}"
                    ),
                    "rank": self.rank,
                }
        if degree > 1 and command.models is UNSET and not command.single().binding.model_class:
            self.poisoned = "sequence-parallel construction unsupported"
            return {
                "ok": False,
                "code": "sequence_parallel_unsupported",
                "detail": "a group requires a model-bearing construction",
            }
        if degree > 1 and (self.group is None if self.rank == 0 else self.pg is None):
            self.poisoned = "group unformed"
            return {
                "ok": False,
                "code": "group_unformed",
                "detail": "a group load needs a started process whose GPUs have joined",
            }
        if command.models is not UNSET:
            return self._prepare_many(command, command.models)
        single = command.single()
        binding, budgets = single.binding, single.budgets
        self._group_model_key = f"{self.construction}#{binding.model_binding_path}"
        started = time.perf_counter()
        # THE SEGMENT LEDGER (cr-102). A `prepare` that reports one total and one fill leg
        # cannot say which of construction, qualification or lease acquisition spent the
        # wall clock, and that ambiguity kept a 20s cost unattributed across three issues.
        stages: list[tuple[str, float]] = []
        _mark = [started]

        def stage(name: str) -> None:
            now = time.perf_counter()
            stages.append((name, round((now - _mark[0]) * 1000, 1)))
            _mark[0] = now

        self._stages = stages
        pin = command.attention_pin
        if not binding.model_class:
            if pin:
                # A PIN IS NEVER SILENTLY IGNORED (cr-125). A weightless binding constructs
                # nothing and therefore holds no attention site, so there is nowhere to put
                # the kernel that was named — and a run that reported success while serving
                # no pin at all is the same lie as one that quietly served another kernel.
                return {
                    "ok": False,
                    "code": "attention_kernel_unsupported",
                    "detail": f"{pin} was pinned on a binding that declares no model; a "
                    "weightless package holds no attention site and a pin is never ignored",
                }
            return self._prepare_weightless(binding, started)
        try:
            attention.admit(_local_attention_pin(pin, binding.parameter_names()), self.world)
        except attention.AttentionRefusal as exc:
            return {"ok": False, "code": exc.code, "detail": str(exc)}
        authorized = command.authorized_device_limit_bytes
        if authorized is UNSET:
            # NO SELF-DERIVED CEILING (cr-025/cr-066): the worker assigns one before every
            # model-bearing prepare. An absent assignment is a worker defect, refused
            # rather than defaulted to a read of free bytes.
            self.poisoned = "prepare without an authorized device limit"
            return {
                "ok": False,
                "code": "authorization_absent",
                "detail": "the prepare carries no authorized_device_limit_bytes; the worker "
                "assigns the residence ceiling and this executor derives none",
            }

        _restore_seal()

        import torch  # the ONE torch import in the runtime, and it is in THIS process

        self.torch = torch
        stage("torch_import")
        # Kernel integration imports may query the CUDA device (for example,
        # Diffusers importing TorchAO's optional Triton kernels). Runtime must
        # establish the selected device before loading those modules.
        refused = self._initialize_device(torch)
        if refused is not None:
            self.poisoned = f"prepare refused: {refused['code']}"
            return refused
        cuda_ms = (time.perf_counter() - started) * 1000
        stage("cuda_init")
        held = [self._device_bytes()]
        from cozy_runtime.author import ModelRegistry
        from cozy_runtime.author._loader import Artifact, Config, ModelFitRefused
        from cozy_runtime.author._model import component_use
        from cozy_runtime.internal.anima_optimization import (
            AnimaOptimizationRefusal,
            apply_anima_execution_plan,
        )
        from cozy_runtime.internal.derive import Observations, serving_substrate
        from cozy_runtime.internal.encoding import launch_providers, measure_device, measure_runtime
        from cozy_runtime.internal.fill import (
            Checkpoint,
            FillRefusal,
            StreamingFillBackend,
            complete_device_envelope,
            dtype_name,
            tensor_schema_of,
        )
        from cozy_runtime.internal.fusion_install import FusionRefusal, apply_fusion_plan
        from cozy_runtime.internal.planfacts import PlanFacts
        from cozy_runtime.internal.probe import qualified
        from cozy_runtime.internal.residency import ComponentResidency
        from cozy_runtime.internal.resolution import (
            ConstructionFacts,
            PlanRefusal,
            Variant,
            resolve,
        )
        from cozy_runtime.internal.warm import WarmFailed

        stage("runtime_imports")
        _restore_seal()
        if self.world > 1 and self.rank == 0:
            # THE FOLLOWERS RESOLVE BESIDE RANK 0, not after it: the load goes to every rank
            # before this one reads a header, and the plans are compared at the residence
            # agreement, before any rank reserves a destination.
            assert self.group is not None
            try:
                self.group.broadcast(
                    executor_commands.encode(
                        msgspec.structs.replace(
                            command,
                            construction=self.construction,
                            group=GroupHardware(degree=self.world, devices=_sealed_entries()),
                        )
                    )
                )
            except GroupRefusal as exc:
                return self._group_refusal(exc)
            stage("group_broadcast")

        if self.discovered is None:
            self._discover_provisioned(binding)
        stage("package_import")

        # The store IS the artifact identity: one snapshot digest, and TensorFS answers with
        # the header, the plan and the verified lease. The interim path needed a store root
        # AND a pre-parsed read-plan file beside it; tfs-007 deleted the second one.
        # The component SET, in the contract's own declared order. A single-component
        # binding is the same code with a one-element set: cr-008b did not add a second
        # path, it widened the one that existed. `snapshots` names ONE snapshot per
        # component, because a TensorFS read plan is complete over its header and a
        # partial read of a whole-pipeline snapshot has no representation (tfs-007).
        names = list(binding.components or [binding.component])
        per_component = dict(binding.snapshots)

        # --------------------------------------------------------- PLAN RESOLUTION (#549.1)
        #
        # ONE decision, made once, here, before a destination is reserved — because what a
        # generation reads decides what is resident, and that is a property of the
        # construction rather than of a request.
        #
        # This used to be TWO decisions that could not see each other: a `DeliveryChooser`
        # picking one artifact-wide RUNG, and a `Selector` picking a provider per tensor. A
        # mixed artifact reported the rung and contained the tensors, and the rung was hashed
        # into generation identity while having no execution authority at all. What is
        # resolved here is the whole plan — the exact snapshot map, every tensor's provider,
        # route and validated geometry, the device and runtime identity, the placement and
        # the objective — and its digest is what enters identity. The LABEL derives from it.
        #
        # The artifact is surveyed from its HEADER (cr-006 measured the real fp8 UNet's
        # whole per-tensor encoding plan at 400,592 B read, zero tensor bytes), so this
        # costs one header read and no load.
        # Package installation carries no pre-qualified hardware variant. Derive it from
        # the executor's device; a repository lane is never a hardware claim. THE DEVICE IS
        # ORDINAL 0 BY CONSTRUCTION (cr-066): the worker sealed this process to exactly its
        # lane's devices, so there is nothing to select here and no index to carry.
        device_index = self.rank if self.world > 1 else 0
        device_facts = measure_device(torch, device_index)
        stage("measure_device")
        hardware_variant = binding.variant
        if not hardware_variant:
            if device_facts.kind != "cuda" or device_facts.sm <= 0:
                self.poisoned = "modeled accelerator unavailable"
                return {
                    "ok": False,
                    "code": "modeled_accelerator_unavailable",
                    "detail": "modeled execution requires a measured CUDA device; no "
                    "caller-authored hardware variant is accepted",
                }
            hardware_variant = f"sm{device_facts.sm}"
        reference = Variant(
            name=hardware_variant,
            store=binding.store,
            snapshot=binding.snapshot,
            snapshots=dict(per_component),
            reference=True,
        )
        vmap = reference.snapshot_map()
        checkpoints = {
            name: Checkpoint(reference.store, vmap.get(name, reference.snapshot)) for name in names
        }
        # `names` is what the DERIVE constructed (model-code-fit D9), not what the header
        # carries: a constructed component the checkpoint lacks is the fit's own
        # `component_missing`, judged at construction with the §3 text, so the plan is
        # resolved over the rows the header does supply and nothing here refuses first.
        present = [name for name in names if checkpoints[name].has(name)]
        rows = [r for name in present for r in checkpoints[name].rows(name)]
        stage("checkpoint_headers")
        # NVML is queried with the sealed device (#549.9): inside the seal, card zero IS the
        # lane's card, so a multi-GPU pod's records name the card the fill actually uses.
        runtime_identity = measure_runtime(torch, binding.release)
        providers = launch_providers()
        # MEASURED ONCE PER (card, build, providers, dtypes), NOT ONCE PER PREPARE
        # (cr-103). The decode suite is 3.2-6.1 s of every model-bearing prepare and none
        # of its inputs change between two models on one warm pod, which is the swap case
        # exactly. `qualified` keys the result on the whole of what determines it and
        # keeps it under COZY_HOME; a term that moves is a miss and the suite runs.
        qualification = qualified(
            torch,
            providers,
            device_facts,
            sorted({dtype_name(r.dtype) for r in rows}),
            release=binding.release,
            runtime=runtime_identity,
            cache=_qualification_cache(),
        )
        qualified_pair = (qualification.capabilities, qualification.result)
        stage("qualify")
        objective = binding.objective or "latency"
        model_cls = getattr(self.discovered.module, binding.model_class)
        config_checkpoint = next(
            (
                checkpoint
                for checkpoint in checkpoints.values()
                if checkpoint.manifest_id == reference.snapshot
            ),
            None,
        ) or Checkpoint(reference.store, reference.snapshot)
        config_bytes = model_config.construction_config(config_checkpoint.header.get("configs"))
        config_document = canonical.parse_canonical(config_bytes)
        if not isinstance(config_document, dict):
            raise RuntimeError("the CozyTensors construction config is not an object")
        stage("construction_config")
        construction = ConstructionFacts(
            release=binding.release,
            model_class=f"{model_cls.__module__}:{model_cls.__qualname__}",
            components=tuple(names),
            custody=binding.custody,
        )
        # CONSENT IS AN INPUT TO RESOLUTION (#549.1). It used to fire at `materialize`, with
        # the plan already chosen and the package already imported; a plan whose route the
        # package cannot consent to is not a candidate, so it is never built.
        consent = str(getattr(model_cls, "__encoded_leaves__", "refuse"))
        facts = PlanFacts()
        try:
            plan = resolve(
                variants=[reference],
                rows_for={reference.name: rows},
                providers=providers,
                capabilities=qualification.capabilities,
                device=device_facts,
                runtime=runtime_identity,
                construction=construction,
                facts=facts,
                objective=objective,
                placement=binding.placement or "all_resident",
                encoded_leaves=consent,
                steps_basis=binding.steps_basis or 20,
                dtype_name=dtype_name,
            )
        except PlanRefusal as exc:
            if not self._holds_nothing(held[0]):
                self.poisoned = f"plan refused: {exc.code}"
            return {
                "ok": False,
                "code": exc.code,
                "detail": str(exc).splitlines()[0][:900],
                "plan": {"walk": msgspec.to_builtins(exc.steps), "objective": binding.objective},
            }
        stage("resolve")
        mine = GroupPlan(
            degree=self.world,
            devices=_sealed_entries(),
            plan_digest=rank_invariant_digest(plan.identity()),
            authorized_device_limit_bytes=authorized,
        )

        # The LOAD leg's liveness feed. Integral-MiB positions on the lossy lane, so the
        # worker can report measured progress for a multi-GB load without inventing a timeout.
        # `started` is published at position 0 before the first byte moves, which is what
        # makes a pre-first-unit wedge render at all.
        def position(mib: int) -> None:
            with contextlib.suppress(SeamError):
                self.channel.send(
                    {
                        "event": "progress",
                        "request_id": f"prepare:{binding.component}",
                        "kind": "load",
                        "name": binding.component,
                        "value": None,
                        "position": mib,
                    }
                )

        position(0)
        backend = StreamingFillBackend(
            checkpoints,
            rows,
            plan=plan,
            device=str(self._rank_device(torch)),
            on_position=position,
            window_bytes=binding.window_bytes,
            slots=binding.slots,
            readers=binding.readers,
            inflight=binding.inflight,
            release=binding.release,
            # The suite already ran, above, to resolve the plan. Running it a second time
            # here would re-measure a device fact that has not changed.
            qualified=qualified_pair,
            forensics_dir=self.fill_forensics_dir,
            custody=binding.custody,
        )
        stage("backend_ctor")
        # THE ARITHMETIC, BEFORE THE FIRST BYTE (#574b). The binding declares its components'
        # logical bytes from their headers; the header envelope prices the rows this plan
        # fills and the selected route's native/decode overhead, and the larger destination
        # count wins where the two have not met. Derived buffers are not priced until
        # construction. This refusal runs before allocation instead of an OOM mid-fill.
        declared = budgets.logical_weight_bytes
        envelope = complete_device_envelope(backend.device_envelope(), declared)
        destinations = int(envelope["base_bytes"])
        overhead = int(envelope["fill_overhead_bytes"])
        # THE PROCESS CEILING (proto-061). `authorized` bounds everything this process
        # holds; the constructions already loaded are parked, least recently used first,
        # until this one's destinations plus its widest fill transient fit beside the rest.
        # Under a group rank 0 decides from the largest need any rank reported, and every
        # rank parks the same constructions before measuring its own ceiling.
        need = destinations + overhead
        host = self.host
        measured: dict[str, int] = {}

        def local_ceiling() -> int:
            memory = accel.allocation(torch, self.device_kind)
            measured["allocatable"] = memory["driver_free_bytes"] + max(
                memory["reserved_bytes"] - memory["allocated_bytes"], 0
            )
            # Rank 0's ceiling moves when the worker vacated co-tenants for it (`_room`).
            ceiling = host.authorized if self.rank == 0 else authorized
            measured["authorized"] = ceiling - memory["allocated_bytes"]
            # THE RESIDENCE CEILING (cl-027 / cr-025 / cr-066): the worker's ASSIGNED
            # limit less what this process already holds, verified against measured
            # capacity, capped at the artifact's complete destination bytes; the APPLIED
            # value is what the generation's digest carries. A staging generation re-reads
            # the store after construction, so its GC-safe hold must outlive the first
            # commit; that is `hold_lease`.
            return residence_ceiling(
                destinations=destinations,
                authorized=measured["authorized"] - overhead,
                allocatable=measured["allocatable"] - overhead,
            )

        try:
            if self.world > 1 and self.rank == 0:
                assert self.group is not None
                resident_budget, self.load_parked = self.group.agree_residence(
                    model=self._group_model_key,
                    plan=mine.document(),
                    required=need,
                    park=lambda most: host._make_room(self.construction, most),
                    ceiling=local_ceiling,
                )
            elif self.world > 1:
                resident_budget = receive_residence(
                    self.channel,
                    rank=self.rank,
                    model=self._group_model_key,
                    plan=mine.document(),
                    required=need,
                    park=host._park,
                    ceiling=local_ceiling,
                )
            else:
                self.load_parked = host._make_room(self.construction, need)
                held[0] = self._device_bytes()
                resident_budget = local_ceiling()
        except (GroupRefusal, SeamError) as exc:
            return self._group_refusal(
                exc if isinstance(exc, GroupRefusal) else GroupRefusal(str(exc))
            )
        except ResidencyRefusal as exc:
            self.poisoned = f"park refused: {exc.code}"
            return {"ok": False, "code": exc.code, "detail": exc.detail[:900]}
        local_resident_ceiling = residence_ceiling(
            destinations=destinations,
            authorized=measured["authorized"] - overhead,
            allocatable=measured["allocatable"] - overhead,
        )
        allocatable = measured["allocatable"]
        authorized_here = measured["authorized"]
        admissible = min(authorized_here, int(allocatable))
        if resident_budget <= 0:
            self._discard_unallocated(backend, held[0], "fill refused: device_shortfall")
            return {
                "ok": False,
                "code": "device_shortfall",
                "detail": (
                    f"{authorized_here} B left of the {authorized} B the worker authorized "
                    f"and {allocatable} B allocatable leave no residence ceiling — nothing "
                    "of this construction can stay resident, so there is no plan to derive"
                ),
                "shortfall": {
                    "resource": "vram",
                    "scope": "component_generation",
                    "needed_bytes": destinations,
                    "available_bytes": admissible,
                    "evidence_class": "measured",
                },
            }
        # The ceiling is capped at `destinations`, an estimate that misses derived buffers
        # when the stored bytes undercut the logical ones (an encoded checkpoint). Capped
        # there, the card holds the whole construction, so the fill parks nothing (0).
        backend.resident_budget = resident_budget if resident_budget < destinations else 0
        backend.hold_lease = 0 < resident_budget < destinations
        # ADMISSION prices the RESIDENT set, not every destination: a construction whose
        # total exceeds the card but whose ceiling fits prepares and stages (cr-025 done-when).
        resident_destinations = (
            min(destinations, resident_budget) if resident_budget else destinations
        )
        required = resident_destinations + overhead
        envelope = {
            **envelope,
            "resident_destination_bytes": resident_destinations,
            "local_resident_ceiling_bytes": local_resident_ceiling,
            "resident_budget_bytes": resident_budget,
            "authorized_device_limit_bytes": authorized,
            "required_bytes": required,
        }
        if required > admissible:
            self._discard_unallocated(backend, held[0], "fill refused: device_shortfall")
            return {
                "ok": False,
                "code": "device_shortfall",
                "detail": (
                    f"this fill will hold {required} B on the device "
                    f"({resident_destinations} B of resident destinations under an "
                    f"assigned ceiling of {resident_budget} B; "
                    f"{envelope['destination_bytes']} B of logical destinations, "
                    f"{envelope['derived_destination_bytes']} B of additional MCC-derived "
                    "destinations, "
                    f"{overhead} B of overlapping fill storage at its widest component) "
                    f"against {authorized_here} B left of the {authorized} B process "
                    f"ceiling and {allocatable} B allocatable - refused on the arithmetic, "
                    "before a byte moved. The "
                    f"binding declared {declared} B of logical destinations; the header "
                    "independently prices stored encoding overhead"
                ),
                "envelope": envelope,
                "logical_weight_bytes": declared,
                "shortfall": {
                    "resource": "vram",
                    "scope": "component_generation",
                    "needed_bytes": required,
                    "available_bytes": admissible,
                    "evidence_class": "measured",
                },
            }
        stage("admission")
        artifact = Artifact(
            reference.snapshot,
            tensor_schema_of(rows),
            Config(config_document, "model.config"),
            assets=config_checkpoint.model_assets(),
            # CUSTODY is a BINDING fact, not a runtime constant: a checkpoint that came
            # through the border is canonical and a serve-time extra key is drift, while a
            # local bind of an un-bordered checkpoint tiers its extras (§1.1). Defaulting
            # to canonical keeps the strict reading for any record that does not say.
            custody=binding.custody,
            unnormalized=model_config.unnormalized(config_checkpoint.header.get("configs")),
        )
        from cozy_runtime.internal.lora_composition import bind as bind_adapters

        from cozy_runtime.internal.lora_contract import read as read_adapter_graph

        adapter_graphs = [
            (checkpoint.manifest_id, data)
            for checkpoint in checkpoints.values()
            if (data := model_config.adapter_graph(checkpoint.header.get("configs")))
        ]
        if adapter_graphs:
            manifest, data = adapter_graphs[0]
            graph = read_adapter_graph(data)
            if any(read_adapter_graph(other) != graph for _, other in adapter_graphs[1:]):
                raise CapabilityError(
                    "one model slot selects different prepared adapter graphs",
                    code="adapter_graph_conflict",
                )
            artifact = bind_adapters(artifact, model_cls, data, prepared_snapshot=manifest)
        seen = Observations()
        registry = ModelRegistry(
            release=binding.release,
            backend=backend,
            substrate=lambda: serving_substrate(hardware_variant, seen),
        )
        try:
            # `expect` is INSIDE the handler because it now takes the verified read lease:
            # a corrupt artifact refuses here, before construction and before a single
            # destination is reserved, and that refusal is a typed fill refusal like any
            # other rather than an exception escaping the prepare path.
            backend.expect(
                {name: tuple(r.key for r in checkpoints[name].rows(name)) for name in present}
            )
            stage("lease_acquire")
            adapter_rows = tuple(
                AdapterRef(
                    ref=row.model_id or row.ref,
                    scale=row.scale,
                    kind=row.kind,
                    component=row.component,
                    source_component=row.source_component,
                    family=row.family,
                )
                for row in binding.adapters
            )
            model = registry.acquire(
                binding.model_binding_path, model_cls, artifact, adapters=adapter_rows
            )
            stage("construct_and_fill")
            optimization, optimization_refusal = apply_anima_execution_plan(
                model,
                application=binding.application,
                model_class=f"{model_cls.__module__}:{model_cls.__qualname__}",
            )
            # THE FUSED GLUE (h3a-015), under the class's `fusion=` consent only: a
            # substitution of forwards after fill. `accept` is CONSENT, not a requirement —
            # an image without the built kernels serves eager and the record says so;
            # only `require` refuses the placement typed.
            consent = str(getattr(model_cls, "__fusion__", "refuse"))
            fused = (
                apply_fusion_plan(
                    model,
                    torch=torch,
                    device=self._rank_device(torch),
                    required=consent == "require",
                )
                if consent in ("accept", "require")
                else None
            )
            stage("execution_plan")
            # THE FASTEST KERNEL THIS DEVICE AND IMAGE SUPPORT, PINNED POST-FILL (cr-124),
            # per attention SITE and before any warm step. No checkpoint is consulted:
            # attention is weightless, so the choice is (device, image) and nothing else.
            # `pin` is the execution-path override (cr-125) and is empty on every serving
            # worker; when it is set it names ONE kernel and refuses rather than falling
            # back. HERE is the only place it can act: `warm` may compile the module in
            # place and a group installs its collectives next, and cp.py's ordering is
            # forced — set the backend, then parallelism, then compile. A pin applied after
            # either is a field nobody reads, which is why this is a construction fact and
            # not a per-attempt one. The degree rides in because a sharded site admits only
            # a kernel `cp.py` can shard, and the fp8 rank is not one (cr-138).
            roots = {**backend.components, **backend.parked}
            reselect = functools.partial(
                self._prepare_attention,
                binding,
                roots,
                pin,
                device_facts,
                choose=model.choose_attention
                if type(model).choose_attention is not Model.choose_attention
                else None,
                optional_scope=optional_attention_scope,
            )
            applied = reselect()
            self.device_facts = device_facts
            stage("attention_pin")
        except ModelFitRefused as exc:
            # THE FIT VERDICT (model-code-fit §3), verbatim. The loader asked before
            # `materialize`: when the device holds nothing new, only the read lease is
            # undone and the process keeps serving its other constructions.
            self._discard_unallocated(backend, held[0], f"fit refused: {exc.fit.get('code')}")
            return {
                "ok": False,
                "code": exc.code,
                "detail": (
                    f"{binding.package or binding.application} "
                    f"{binding.model_binding_path} does not fit "
                    f"{binding.model or reference.snapshot}: {exc.message}"
                )[:900],
                "fit": exc.fit,
                "state": backend.state,
                "envelope": envelope,
            }
        except (AnimaOptimizationRefusal, FusionRefusal) as exc:
            backend.poison(())
            self.poisoned = f"execution plan refused: {exc.code}"
            return {
                "ok": False,
                "code": exc.code,
                "detail": str(exc)[:900],
                "state": backend.state,
                "envelope": envelope,
                "poison": backend.poison_report,
            }
        except attention.AttentionRefusal as exc:
            backend.poison(())
            self.poisoned = f"attention refused: {exc.code}"
            return {
                "ok": False,
                "code": exc.code,
                "detail": str(exc)[:900],
                "state": backend.state,
                "envelope": envelope,
                "poison": backend.poison_report,
            }
        except Exception as exc:
            # The two failures this path owns, and nothing else: a typed fill refusal, or
            # the backend's own OOM — which is asked of the accelerator boundary rather
            # than named here, so a second backend spells its allocator error once (#447).
            refusal = exc if isinstance(exc, FillRefusal) else None
            if refusal is None and not is_device_oom(exc):
                raise
            # The generation is already rolled back, so the numbers that matter are the
            # HIGH-WATER marks beside the after state — "released 0 B" would be true and
            # useless. A device OOM here is a RESOURCE SHORTFALL, not a crash: it is the
            # ordinary outcome of a neighbour holding the card, and it carries the two
            # numbers an operator needs (cr-005).
            oom = refusal is None
            # NO ROLLBACK (#613). The loader's abort already poisoned the backend for a
            # typed refusal; an allocator OOM that escaped it is poisoned here. Nothing
            # frees a byte in this process: the prepare latch is consumed, the worker
            # replaces the executor and external reclaim returns the card. What follows
            # is EVIDENCE — the allocator and driver as they stood when the fill failed.
            if backend.state != "poisoned":
                backend.poison(())
            memory = accel.allocation(torch, self.device_kind)
            free, total = memory["driver_free_bytes"], memory["driver_total_bytes"]
            code = "device_shortfall" if refusal is None else refusal.code
            self.poisoned = f"fill refused: {code}"
            reply: dict[str, Any] = {
                "ok": False,
                "code": code,
                "detail": str(exc).splitlines()[0][:400],
                "state": backend.state,
                "peak_allocator_bytes": accel.peak_allocated(torch, self.device_kind),
                "allocator_bytes": memory["allocated_bytes"],
                "device_free_bytes": free,
                "device_total_bytes": total,
                "envelope": envelope,
                # WHAT THE FILL HELD when it failed. The process is replaced, so these
                # numbers explain the refusal rather than promise a return.
                "poison": backend.poison_report,
            }
            if oom:
                reply["shortfall"] = {
                    "resource": "vram",
                    "scope": "component_generation",
                    "needed_bytes": budgets.logical_weight_bytes,
                    "available_bytes": int(free),
                    "evidence_class": "measured",
                }
            return reply
        self.backend = backend
        self.registry = registry
        # ONE generation under EVERY parameter name the placement's bindings spell
        # (h3a-018): two entrypoints over one construction share the object.
        self.models = {name: model for name in binding.parameter_names()}
        # THE COMPONENT-USE CONTRACT'S RUNTIME BODY, installed on the constructed
        # generation. From here on, entering a declared scope admits and materializes its
        # set before the body runs; the package cannot observe that it happened.
        self.residency = ComponentResidency(
            backend=backend, torch=torch, placement="all_resident", room=self._room
        )
        object.__setattr__(model, "_cozy_residency", self.residency)
        record = registry.generations()[0].record
        # THE CONSTRUCTED GENERATION is what actually got filled, and since #549.1 it is
        # THE RESOLVED PLAN'S OWN DIGEST that says so. The plan digest binds the exact
        # snapshot map, every tensor's implementation digest and route, the device identity,
        # the runtime source digest and the construction facts, so two
        # materially different fills of one artifact never share it. There is no second,
        # "logical" identity beside it: the MCC digest and its `expect_construction_digest`
        # fence are deleted whole (model-code-fit D12) — its only consumer was a fence the
        # planner had already disabled.
        self.plan = plan
        self.attention, self._reselect = applied, reselect
        constructed_preimage = canonical.write(
            {
                "plan_digest": plan.digest(),
                # THE APPLIED CEILING, derived from the card (cl-027).
                "resident_budget_bytes": resident_budget,
                "execution_optimization": optimization.document() if optimization else None,
                "execution_fusion": fused.identity_document() if fused is not None else None,
            }
        )
        self.constructed_model_digest = (
            "sha256:"
            + hashlib.sha256(
                b"cozy.runtime.constructed-generation\0" + constructed_preimage
            ).hexdigest()
        )
        if self.group is not None:
            # THE FOLLOWERS FILLED THE SAME PLAN, or the group is not a group. Every reply
            # is held to rank 0's own construction identities.
            try:
                replies = self.group.collect("load")
            except GroupRefusal as exc:
                return self._group_refusal(exc)
            for follower, reply in zip(self.group.followers, replies, strict=True):
                if not reply.get("ok"):
                    code = str(reply.get("code") or "group_unformed")
                    return self._group_refusal(
                        GroupRefusal(
                            f"{follower.gpu} refused the prepare: {code}: "
                            f"{reply.get('detail', '')}"[:800],
                            code=code if code in _GROUP_CODES else "group_unformed",
                        )
                    )
                # The CONSTRUCTION must be the same on every rank; the constructed digest
                # folds the card's index and is rank-local, so what is compared beside it
                # is what the fill committed and the ceiling it committed under.
                theirs = dict(reply.get("facts") or {})
                agreed: list[tuple[str, object, object]] = [
                    ("filled_bytes", record.filled_bytes, theirs.get("filled_bytes")),
                    (
                        "resident_budget_bytes",
                        resident_budget,
                        theirs.get("resident_budget_bytes"),
                    ),
                ]
                for key, ours, other in agreed:
                    if other != ours:
                        return self._group_refusal(
                            GpuDivergence(
                                follower.gpu,
                                key,
                                f"{_gpu(0)} built {ours!r}, {follower.gpu} built {other!r}",
                            )
                        )
            refused = self._agree_attention(applied)
            if refused is not None:
                return refused
        self.facts = {}
        refused = self._install_group(
            torch,
            model,
            settled_capacity(resident_budget=resident_budget, fill_overhead=overhead),
        )
        if refused is not None:
            return refused
        stage("install")
        # Warm-up may stage components one scope at a time even on a roomy card.
        # Preserve what the construction actually fitted before those temporary moves:
        # the worker must restore that set, not treat warm's last scope as a capacity floor.
        constructed_resident = dict(backend.vram_charge)
        # A component another rank hosts is not this rank's to hold, restore or price.
        hosted = self.residency.hosted if self.residency is not None else frozenset()
        construction_parked = sorted((set(backend.parked) | set(backend.paging)) - hosted)
        # THE AUTHOR'S ONE WARM MOMENT (cr-110, model-lifecycle.md): after the fill, its
        # verification, the execution plan, the residency install and the group mirror;
        # before this generation is Ready, so a placement is never DISPATCHABLE over a cold
        # generation. Rank 0 runs the warm method; its sharded component forwards reach
        # the followers through the same mirror as ordinary inference.
        warm_ms = 0.0
        if self.rank == 0:
            try:
                warm_ms = self._warm_model(model)
            except WarmFailed as exc:
                # A CONSTRUCTION FAILURE, typed on the author's exception: the executor is
                # poisoned and replaced like every other prepare refusal, and the binding
                # faults instead of serving a generation whose warm did not complete. The
                # device running short is a capacity fact and is typed as one, so the
                # refusal stays this attempt's and is never held against the binding.
                backend.poison(())
                self.poisoned = f"warm failed: {exc}"[:200]
                cause = exc.__cause__
                short = cause if isinstance(cause, ResidencyRefusal) else None
                oom = cause is not None and is_device_oom(cause)
                return {
                    "ok": False,
                    "code": short.code if short else "device_out_of_memory" if oom else exc.code,
                    "shortfall": short.shortfall if short else None,
                    "detail": str(exc)[:900],
                    "traceback": prepare_diagnostics.exception_trace(exc),
                    "state": backend.state,
                    "envelope": envelope,
                }
        stage("warm")
        if "sol-attn" in applied.totals():
            frozen = attention._baked_in({**backend.components, **backend.parked})
            if frozen:
                backend.poison(())
                self.poisoned = "attention refused: attention_kernel_frozen"
                return {
                    "ok": False,
                    "code": "attention_kernel_frozen",
                    "detail": f"Sol requires eager execution: {frozen}",
                }
        self.attention_defaults = attention.snapshot({**backend.components, **backend.parked})
        group_facts = dict(self.facts)
        memory = accel.allocation(torch, self.device_kind)
        self.ready = True
        self.facts = {
            # `sequence_parallel` rides here under a group ONLY: a plain executor's facts
            # are exactly what they were.
            **group_facts,
            "cuda_init_ms": round(cuda_ms, 2),
            "prepare_ms": round((time.perf_counter() - started) * 1000, 2),
            "prepared_unix_ms": int(time.time() * 1000),
            # THE SEGMENT LEDGER (cr-102), in execution order. `prepare_ms` minus `fill_ms`
            # was a 20s number nobody could attribute; these are the legs that spend it.
            "stages": list(stages) + [(f"construct.{name}", ms) for name, ms in record.stage_ms],
            # WHERE THE QUALIFY LEG'S ANSWER CAME FROM (cr-103). A cache nobody can see
            # fire is a cache nobody can prove fires, and "it got faster" is not evidence
            # on a thermally variable card. `source` says measured or reused, per prepare.
            "qualification": qualification.document(),
            # THE WEIGHTLESS PROMISE, OBSERVED (cr-102). `serving_substrate` swaps every
            # parameter to `meta` as it is registered; this is the count that says it did.
            # NOT `weightless`: that key already meant "this binding declares no model" on
            # a weightless prepare's own facts, and one name for a bool and a census made
            # the worker read `len(True)` and crash EVERY weightless activation (cr-104).
            "substrate_census": list(record.weightless),
            # WHY the fused lane was unavailable, when it was (cr-102 / proto-038). The
            # verdict itself already rode in `execution_optimization` below and nothing
            # rendered it; a `None` with no reason is the part an operator cannot act on.
            "execution_optimization_refusal": optimization_refusal,
            # The fused-glue plan's record (h3a-015): None where the class did not consent,
            # else what was substituted (or why nothing was) on this device.
            "execution_fusion": fused.document() if fused is not None else None,
            # An all-inline checkpoint commits without opening the object stream. No stream
            # wall was measured in that case, and zero is the exact value rather than an
            # absent-stat KeyError after a successful construction.
            "fill_ms": backend.stats.get("fill_ms", 0),
            "warm_ms": round(warm_ms, 2),
            "filled": record.filled,
            "filled_bytes": record.filled_bytes,
            "device_free_bytes": memory["driver_free_bytes"],
            "device_total_bytes": memory["driver_total_bytes"],
            "allocator_bytes": memory["allocated_bytes"],
            "reserved_bytes": memory["reserved_bytes"],
            "resident": dict(backend.vram_charge),
            "constructed": constructed_resident,
            "parked": construction_parked,
            "paging": {name: layout.document() for name, layout in backend.paging.items()},
            # WITH THEIR SIZES, like the residency document's `evicted`: a parked component
            # is a component the next attempt may have to bring back, and a chooser that
            # sees its name and not its bytes prices that stage-in at nothing.
            "evicted": {
                name: backend.component_bytes.get(name, 0)
                for name in backend.parked
                if name not in hosted
            },
            "resident_budget_bytes": resident_budget,
            "authorized_device_limit_bytes": authorized,
            "rss_bytes": _rss(),
            "pinned": dict(backend.pinned),
            # WHAT THIS GENERATION HOLDS OPEN (cr-103). `lease_acquire` was 1.9-3.4 s of
            # the prepare and three quarters of it was the same snapshot leased once per
            # component; the count is here so the saving is a number an operator reads
            # rather than a claim, and so the descriptors pinned against FD_HEADROOM are
            # visible beside the admission that spends them.
            "leases": backend.leases.document(),
            "store": backend.stats.get("store", {}),
            "gpu_name": accel.device_identity(torch, self.device_kind)["name"],
            "device_driver": device_facts.driver,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "",
            # THE CONFESSION (§1.1). "Loudly ignored, never silently" is only true if the
            # ignoring is named where a person and a RecordOwner both see it, so the extras
            # this bind dropped ride the bind result.
            "custody": binding.custody,
            # THE COMPONENT-USE CONTRACT, as a construction fact. The worker prices a
            # staged rung against it and cannot read it any other way: the declaration lives
            # on the package's class, and that class is only ever imported here.
            "declared_scopes": {
                method: list(names) for method, names in sorted(component_use(model_cls).items())
            },
            "ignored_extra_keys": list(record.ignored_extras),
            # THE FIT this generation was judged by (§3), verbatim: the eight-code document
            # `tensorfs.fit` returned, `ok` here by construction.
            "fit": record.fit,
            # THE SEAM FORM (#574b). The full document carries a row per destination and a
            # control frame is 64 KiB; on an N-ary artifact that reply could not be sent.
            "plan": plan.seam_document(),
            # The WIRE's own document, DERIVED from the plan's route census and placement
            # rather than copied off a chooser's rung. The key set is the worker's — that
            # seam is closed and reads `rung`/`variant` — and `label` rides beside them, so
            # a plan that resolved three routes prices as what it COSTS and reports
            # `mixed(...)` as what it IS.
            "delivery": plan.wire_document(),
            "execution_optimization": optimization.document() if optimization else None,
            # WHAT THE RUNTIME PINNED (cr-124): kernel per component, and what it chose among.
            "attention": applied.document() if applied is not None else None,
        }
        return {
            "ok": True,
            "ready": True,
            "constructed_model_digest": self.constructed_model_digest,
            "entrypoints": sorted(self.surface_names),
            "plan": plan.seam_document(),
            "delivery": self.facts["delivery"],
            "facts": self.facts,
        }

    def _prepare_many(self, command: Load, rows: tuple[ModelLoad, ...]) -> dict[str, Any]:
        """Prepare one complete bounded multi-model callable in this device process."""
        if not 2 <= len(rows) <= 16:
            return {
                "ok": False,
                "code": "model_binding_set_invalid",
                "detail": "models must be an array of 2 through 16 binding rows with budgets",
            }
        paths = [row.binding.model_binding_path for row in rows]
        parameters = [row.binding.model_parameter_name for row in rows]
        if (
            not all(paths)
            or paths != sorted(set(paths))
            or not all(parameters)
            or len(parameters) != len(set(parameters))
        ):
            return {
                "ok": False,
                "code": "model_binding_set_invalid",
                "detail": "model paths must be sorted unique and parameters must be unique",
            }

        authorized = command.authorized_device_limit_bytes
        if authorized is UNSET:
            self.poisoned = "prepare without an authorized device limit"
            return {
                "ok": False,
                "code": "authorization_absent",
                "detail": "the prepare carries no authorized_device_limit_bytes; the worker "
                "assigns the residence ceiling and this executor derives none",
            }
        aggregate_pin = command.attention_pin
        try:
            scope, _ = attention._pin_parts(aggregate_pin)
            attention.admit(aggregate_pin, self.world)
        except attention.AttentionRefusal as exc:
            return {"ok": False, "code": exc.code, "detail": str(exc)}
        declared_parameters = tuple(
            dict.fromkeys(name for row in rows for name in row.binding.parameter_names())
        )
        if "/" in scope and scope.split("/", 1)[0] not in declared_parameters:
            return {
                "ok": False,
                "code": "attention_kernel_unsupported",
                "detail": f"{aggregate_pin} matches no model in {list(declared_parameters)}",
            }

        # ONE authorization for the PROCESS: each model fits under what everything already
        # loaded left of it, measured by this process's own allocator before its fill.
        children: list[tuple[str, Executor, dict[str, Any]]] = []
        for row in rows:
            binding = row.binding
            parameter = binding.model_parameter_name
            child = self._construction_child(self.construction)
            child._group_parent = self
            reply = child._prepare_first(
                Load(
                    construction=self.construction,
                    binding=binding,
                    budgets=row.budgets,
                    devices=command.devices,
                    authorized_device_limit_bytes=authorized,
                    attention_pin=aggregate_pin
                    if attention.for_model(aggregate_pin, binding.parameter_names())
                    else "",
                    model_key=f"{self.construction}#{binding.model_binding_path}"
                    if self.world > 1
                    else UNSET,
                    sequence_parallel_degree=max(self.world, 1),
                ),
                optional_attention_scope=True,
            )
            self.load_parked += child.load_parked
            if not reply.get("ok"):
                self.poisoned = child.poisoned or (
                    f"model {parameter} refused: {reply.get('code', 'unknown')}"
                )
                return {**reply, "model_parameter": parameter}
            self._device_ready = child._device_ready
            self.discovered = child.discovered
            self.surface_names = dict(child.surface_names)
            children.append((parameter, child, reply))

        if self.world > 1 and not any(
            name[0].startswith(self.construction + "#") for name in self.host.group_sharded
        ):
            return self._group_refusal(GroupRefusal("the model set contains no sharded component"))
        if aggregate_pin and not any(
            child.attention and child.attention.pin for _, child, _ in children
        ):
            self.poisoned = "attention override matches no component"
            return {
                "ok": False,
                "code": "attention_kernel_unsupported",
                "detail": f"{aggregate_pin} matches no attention component",
            }
        self.model_executors = [child for _, child, _ in children]
        self.torch = children[-1][1].torch
        self.models = {}
        for parameter, child, _ in children:
            if parameter not in child.models or set(child.models) & set(self.models):
                self.poisoned = "model binding parameter mismatch"
                return {
                    "ok": False,
                    "code": "model_binding_set_invalid",
                    "detail": f"prepared model object did not bind parameter {parameter!r}",
                }
            self.models.update(child.models)
        residencies = {
            parameter: child.residency
            for parameter, child, _ in children
            if child.residency is not None
        }
        self.residency = _CompositeResidency(residencies)

        constructed = [
            {"parameter": parameter, "digest": str(reply["constructed_model_digest"])}
            for parameter, _, reply in children
        ]
        self.constructed_model_digest = (
            "sha256:"
            + hashlib.sha256(
                b"cozy.runtime.constructed-generation-set\0" + canonical.write(constructed)
            ).hexdigest()
        )

        facts_rows = [(parameter, reply["facts"]) for parameter, _, reply in children]
        last = facts_rows[-1][1]
        resident = {
            f"{parameter}/{component}": int(size)
            for parameter, facts in facts_rows
            for component, size in (facts.get("resident") or {}).items()
        }
        parked = [
            f"{parameter}/{component}"
            for parameter, facts in facts_rows
            for component in facts.get("parked", [])
        ]
        evicted = {
            f"{parameter}/{component}": int(size)
            for parameter, facts in facts_rows
            for component, size in (facts.get("evicted") or {}).items()
        }
        scopes = {
            f"{parameter}.{method}": [f"{parameter}/{component}" for component in components]
            for parameter, facts in facts_rows
            for method, components in (facts.get("declared_scopes") or {}).items()
        }
        paging = {
            f"{parameter}/{component}": layout
            for parameter, facts in facts_rows
            for component, layout in (facts.get("paging") or {}).items()
        }
        pinned_rows = [facts.get("pinned") or {} for _, facts in facts_rows]
        # WHAT EACH MODEL PINNED, MERGED (cr-124). A group prepare has one boot note, so a
        # per-model `attention` fact would simply be dropped and the placement would report
        # no kernel at all — the one fact the boot line exists to carry.
        attention_fact = attention.merge(
            {parameter: facts.get("attention") or {} for parameter, facts in facts_rows}
        )
        routes: dict[str, int] = {}
        kernels: dict[str, int] = {}
        for _, _, reply in children:
            delivery = reply.get("delivery") or {}
            for name, count in (delivery.get("routes") or {}).items():
                routes[str(name)] = routes.get(str(name), 0) + int(count)
            for name, count in (delivery.get("kernels") or {}).items():
                kernels[str(name)] = kernels.get(str(name), 0) + int(count)
        deliveries = {parameter: reply["delivery"] for parameter, _, reply in children}
        materializations = {row.get("materialization", "") for row in deliveries.values()}
        delivery = {
            "variant": ", ".join(
                f"{parameter}={row.get('variant', '')}" for parameter, row in deliveries.items()
            ),
            "label": ", ".join(
                f"{parameter}: {row.get('label', '')}" for parameter, row in deliveries.items()
            ),
            # This is a summary over independently resolved models, never a new rung.
            "models": deliveries,
            "delivery": "float" if routes.get("decoded_float") else "native",
            "materialization": (
                "staged_decode"
                if "staged_decode" in materializations
                else "aot_decode"
                if "aot_decode" in materializations
                else ""
            ),
            "routes": routes,
            "kernels": kernels,
            "calibrated": all(
                bool(reply.get("delivery", {}).get("calibrated")) for _, _, reply in children
            ),
            "objective": ", ".join(
                sorted({str(row.get("objective", "")) for row in deliveries.values()})
            ),
        }
        self.facts = {
            "cuda_init_ms": sum(float(facts.get("cuda_init_ms", 0)) for _, facts in facts_rows),
            "prepare_ms": sum(float(facts.get("prepare_ms", 0)) for _, facts in facts_rows),
            "prepared_unix_ms": int(time.time() * 1000),
            # THE SEGMENT LEDGER AND THE SUBSTRATE CENSUS, FOR A MANY-MODEL PREPARE TOO
            # (cr-130). cr-102 added both to `_prepare_first` and nowhere else, so a
            # placement with two or more models reported one summed `prepare_ms` and no
            # legs at all -- exactly the unattributable total cr-102 exists to end, kept
            # alive on the path where it is hardest to reason about because there are now
            # several constructions inside the one number. The legs carry their model's
            # parameter name, because "which model spent it" is the first question.
            "stages": [
                (f"{parameter}.{name}", ms)
                for parameter, facts in facts_rows
                for name, ms in (facts.get("stages") or ())
            ],
            # Summed elementwise: (parameters on meta, parameters on a real device, the
            # bytes those real ones hold). One model allocating its weights on the host is
            # the alarm whether or not its siblings behaved.
            "substrate_census": [
                sum(int((facts.get("substrate_census") or (0, 0, 0))[i]) for _, facts in facts_rows)
                for i in range(3)
            ],
            "fill_ms": sum(float(facts.get("fill_ms", 0)) for _, facts in facts_rows),
            "warm_ms": sum(float(facts.get("warm_ms", 0)) for _, facts in facts_rows),
            "filled": sum(int(facts.get("filled", 0)) for _, facts in facts_rows),
            "filled_bytes": sum(int(facts.get("filled_bytes", 0)) for _, facts in facts_rows),
            "device_free_bytes": int(last.get("device_free_bytes", -1)),
            "device_total_bytes": int(last.get("device_total_bytes", -1)),
            "allocator_bytes": int(last.get("allocator_bytes", -1)),
            "reserved_bytes": int(last.get("reserved_bytes", -1)),
            "resident": resident,
            "constructed": {
                f"{parameter}/{component}": int(size)
                for parameter, facts in facts_rows
                for component, size in facts.get("constructed", facts.get("resident", {})).items()
            },
            "parked": parked,
            "evicted": evicted,
            "paging": paging,
            "declared_scopes": scopes,
            "rss_bytes": max(int(facts.get("rss_bytes", 0)) for _, facts in facts_rows),
            "pinned": {
                "registered_bytes": sum(int(row.get("registered_bytes", 0)) for row in pinned_rows),
                "in_flight_bytes": sum(int(row.get("in_flight_bytes", 0)) for row in pinned_rows),
            },
            "attention": attention_fact,
            "gpu_name": str(last.get("gpu_name", "")),
            "torch_version": str(last.get("torch_version", "")),
            "cuda_version": str(last.get("cuda_version", "")),
            "custody": "joint",
            "ignored_extra_keys": [
                f"{parameter}/{key}"
                for parameter, facts in facts_rows
                for key in facts.get("ignored_extra_keys", [])
            ],
            "delivery": delivery,
            "models": {parameter: facts for parameter, facts in facts_rows},
            "cross_model_eviction": False,
        }
        self.ready = True
        return {
            "ok": True,
            "ready": True,
            "constructed_model_digest": self.constructed_model_digest,
            "entrypoints": sorted(self.surface_names),
            "delivery": delivery,
            "facts": self.facts,
        }

    def _discover_provisioned(self, release: Release) -> None:
        """Import the installed release and bind its PackageInterface."""
        if not release.application or not release.package_interface:
            raise RuntimeError("executor requires the canonical installed release identity")
        discovered = discover_installed(release.application)
        self.discovered = discovered
        self.surface_names = {surface.name: surface for surface in discovered.surfaces}

    def _prepare_weightless(self, binding: Binding, started: float) -> dict[str, Any]:
        """A binding that declares NO MODEL: discover the release, be ready, import nothing.

        The whole cold path below is a fill, and there is nothing to fill. Returning here is
        what makes a weightless package servable at all — and it is the ONLY thing a
        cardless runner can ever pass, because everything past this line imports torch
        (cl-013). There is no construction, so there is no construction identity: the
        digests are empty and say so rather than naming a build that never happened.
        """
        if self.discovered is None:
            self._discover_provisioned(binding)
        self.ready = True
        self.facts = {
            "cuda_init_ms": 0.0,
            "prepare_ms": round((time.perf_counter() - started) * 1000, 2),
            "fill_ms": 0.0,
            "filled": 0,
            "filled_bytes": 0,
            "device_free_bytes": -1,
            "device_total_bytes": -1,
            "allocator_bytes": -1,
            "reserved_bytes": -1,
            "resident": {},
            "parked": [],
            "declared_scopes": {},
            "rss_bytes": _rss(),
            "weightless": True,
            "custody": "canonical",
            "ignored_extra_keys": [],
        }
        return {
            "ok": True,
            "ready": True,
            "weightless": True,
            "constructed_model_digest": "",
            "entrypoints": sorted(self.surface_names),
            "delivery": {},
            "facts": self.facts,
        }

    # ------------------------------------------------------------------ invoke

    def _target(
        self, command: PrepareRequest | Invoke, *, attempt: bool
    ) -> Executor | dict[str, Any]:
        """The construction a request or attempt runs on: the one it names, or the active."""
        key = command.construction or self.active
        child = self.constructions.get(key)
        if child is None:
            return {
                "ok": False,
                "code": "construction_not_loaded" if self.constructions else "executor_not_ready",
                "detail": f"no loaded construction {key!r} in this executor",
            }
        if attempt and key != self.active:
            return {
                "ok": False,
                "code": "construction_inactive",
                "detail": f"{key!r} is loaded but {self.active!r} is active; activate it first",
            }
        return child

    def serve_request(self, command: PrepareRequest) -> dict[str, Any]:
        target = self._target(command, attempt=False)
        if isinstance(target, dict):
            return target
        return target.prepare_request(command)

    def serve_invoke(self, command: Invoke) -> dict[str, Any]:
        target = self._target(command, attempt=True)
        if isinstance(target, dict):
            return target
        key = target.construction
        row = self.arbiter.rows[key]
        # A lane vacate parked the active construction; its admissions stage on demand.
        row.parked = None
        self.arbiter.touch(key)
        reply = target.invoke(command)
        self.attempts += 1
        self.poisoned = self.poisoned or target.poisoned
        reply["poisoned"] = self.poisoned
        return reply

    def prepare_request(self, command: PrepareRequest) -> dict[str, Any]:
        """Resolve ONE request through the kernel's pre-admission half, and retain it.

        The worker calls this BEFORE it prices a plan, so the normalized features and
        the preflight digests it journals into the AttemptPlan are the exact ones the
        handler then runs on. The resolved payload is an author-typed object and cannot
        cross the seam, so it stays here under the request id and only its digests travel;
        `invoke` runs the record this produced and never resolves a second time.

        No device is touched and no handler is entered, so a refusal here leaves a Ready
        executor exactly as it found it — it is a PRE-ENTRY refusal in the strongest sense,
        arriving before the attempt is even accepted.
        """
        if not self.ready:
            return {
                "ok": False,
                "code": "executor_not_ready",
                "detail": "a request prepared against a non-Ready executor",
            }
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        if self.surface_names.get(command.entrypoint) is None:
            return {
                "ok": False,
                "code": "unknown_entrypoint",
                "detail": f"{command.entrypoint!r} is not an entrypoint of this release",
            }
        registration = self.discovered.app.get(command.entrypoint)
        pin = command.attention_kernel
        if pin and self._attention_reaches(pin):
            try:
                self._attention_plan(pin)
            except attention.AttentionRefusal as exc:
                return {
                    "ok": False,
                    "code": exc.code,
                    "detail": str(exc)[:900],
                    "origin": "request",
                }
        try:
            self._capture_options(command, registration.surface)
            prepared = prepare(registration, command.payload, input_metadata=command.input_metadata)
        except Exception as exc:
            outcome = classify(exc)
            return {
                "ok": False,
                "code": outcome.code,
                "detail": outcome.message[:1024],
                "terminal": outcome.terminal,
                "origin": outcome.origin,
                "fields": list(outcome.fields),
            }
        self.prepared[command.request_id] = prepared
        # A prepared request the worker never dispatched (a refusal after the plan, a
        # cancel while it queued) would otherwise be retained for the generation's life.
        for stale in list(self.prepared)[: -PREPARED_HELD or None]:
            del self.prepared[stale]
        return {
            "ok": True,
            "features": dict(prepared.features.values),
            "adjustments": [
                {"field": row.field, "requested": str(row.requested), "applied": str(row.applied)}
                for row in prepared.overlay.rows
            ],
            **prepared.digests(),
        }

    def invoke(self, command: Invoke) -> dict[str, Any]:
        """ONE attempt through the author kernel, with the §3.4 freeze point held."""
        if not self.ready:
            return {
                "ok": False,
                "code": "executor_not_ready",
                "detail": "an online attempt against a non-Ready executor: the directive "
                "prepares the executor first — serving never accepts-then-cold-loads",
            }
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        surface = self.surface_names.get(command.entrypoint)
        if surface is None:
            return {
                "ok": False,
                "code": "unknown_entrypoint",
                "detail": f"{command.entrypoint!r} is not an entrypoint of this release",
            }
        prepared = self.prepared.pop(command.request_id, None)
        if prepared is None:
            return {
                "ok": False,
                "code": "prepared_request_absent",
                "detail": f"nothing was prepared for {command.request_id!r}: an attempt "
                "runs the record its plan was priced against, and this executor holds none "
                "(a fresh epoch holds nothing at all)",
            }
        requested = command.attention_kernel
        # A pin is the run's: a call holding no site in its scope runs its own selection,
        # and the worker reports a run none of whose calls applied it.
        pin = requested if requested and self._attention_reaches(requested) else ""
        try:
            if not pin:
                self._upgrade_attention()
            swap = self._attention_plan(pin) if pin else None
            self._restore_attention()
        except attention.AttentionRefusal as exc:
            self._restore_attention()
            return {"ok": False, "code": exc.code, "detail": str(exc)[:900], "origin": "request"}
        spool = Path(command.spool)
        deadline = _deadline(command.deadline_s)
        torch = self.torch
        self._attempt_spool = self.host._attempt_spool = spool
        self.host._attempt_request = command.request_id
        if self.group is not None:
            self.group.working_peaks.clear()
            self.group.rank_records.clear()
            self.group.spread_log.clear()

        # The lossy lane, carrying BOTH kinds of live record. A progress frame is not
        # retained anywhere: its `position` is the monotone unit count the worker's
        # silence detector reads, and that is all it is for. A RETAINED observation rides the
        # same lane so the worker holds a copy BEFORE this process can die — a ring that
        # only crosses in the reply crosses never when the process is killed, which is the
        # exact case triage exists for. A seam refusal there is dropped live output, the ONE
        # loss this lane may incur.
        progress = self._progress_sink(command.request_id)

        # The attempt's PLACEMENT RUNG, fixed by the PlanChooser before durable acceptance
        # and applied here. The executor decides nothing: it is handed a rung and obeys it,
        # which is what keeps `all_resident` from quietly starting to swap.
        if self.residency is not None:
            self.residency.open_attempt(
                command.placement,
                command.headroom_bytes,
                command.scope_headroom_bytes,
                command.measured_scopes,
            )
        if torch is not None:
            accel.reset_peak(torch, self.device_kind)
        allocated_at_start = accel.allocated(torch, self.device_kind) if torch is not None else 0
        lease_acquired = time.perf_counter()
        started_wall = time.time()
        started_us = execution_evidence.now_us()
        served_attention: attention.Applied | None = None
        sol_counts: dict[str, int] = {}
        sol_rank_counts: dict[int, dict[str, int]] = {}
        sol_kernels: list[dict[str, Any]] = []
        served_impls: dict[str, set[str]] = {}
        capture = None
        capture_result: dict[str, Any] | None = None
        record: AuthorAttempt | None = None
        try:
            with (
                attention.override(
                    self._attention_snapshot(), self._attention_roots(), swap, pin
                ) as served_attention,
                self._ranks_attention(pin, served_attention),
                attention_sol.observing(ranks=sol_rank_counts, kernels=sol_kernels) as sol_counts,
                attention_ulysses.observing() as served_impls,
            ):
                capture = self._capture_session(command, surface)
                with capture if capture is not None else contextlib.nullcontext():
                    envelope, outcome, record = run_prepared(
                        prepared,
                        Invocation(
                            request_id=command.request_id,
                            spool=spool,
                            deadline=deadline,
                            device=self._device(),
                            cancel=lambda: _cancelled(self.root, command.request_id),
                            models=self.models,
                            progress=progress,
                            activity=self._activity_sink(command.request_id),
                            # cr-012: the inputs the WORKER already fetched, verified and spooled.
                            # Every value is a path inside this attempt's own spool: this process is
                            # `sandbox.refuse_network` rejects ungranted network access.
                            # Input bytes arrive only through the worker grant.
                            # That fence is Python-level (cr-042) — it bounds a buggy package, not a
                            # hostile one, and there is no netns or seccomp under it.
                            trees=_trees(command),
                            assets=_granted_inputs(command),
                            max_input_bytes=command.max_input_bytes or 64 << 20,
                            max_output_bytes=command.max_output_bytes,
                            publish=self._publish,
                        ),
                    )
                if capture is not None and outcome.terminal == "succeeded":
                    capture_result = capture.write(spool)
        except Exception as exc:
            envelope, outcome = None, classify(exc)
            if record is None:
                record = AuthorAttempt(command.request_id, spool, 0, sink=progress)
        assert record is not None
        if served_attention is not None:
            self._emit_attention(record, served_attention, "attention.selected")
            if "sol-attn" in served_attention.totals():
                record.emit(
                    "confession",
                    "attention.sol.calls",
                    "sol-attn",
                    rank=0,
                    world=self.world,
                    dense_reference=attention_sol.dense_identity(),
                    **sol_counts,
                )
                for rank, counts in sorted(sol_rank_counts.items()):
                    record.emit(
                        "confession",
                        "attention.sol.calls",
                        "sol-attn",
                        rank=rank,
                        world=self.world,
                        **counts,
                    )
            # Where this request's Sol kernel came from (h3a-087): loaded from the machine
            # store, waited for behind a builder, or compiled here, and how long each took.
            for row in sol_kernels:
                fields = dict(row)
                record.emit(
                    "confession", "attention.sol.kernel", fields.pop("source"), rank=0, **fields
                )
        self._emit_attention(
            record, attention.observed(self._attention_roots()), "attention.restored"
        )
        handler_ms = (time.perf_counter() - lease_acquired) * 1000
        self._attempt_spool = self.host._attempt_spool = None
        self.host._attempt_request = ""
        if self.group is not None and self.group.broken and not self.poisoned:
            # A FOLLOWER DIED OR DIVERGED UNDER THIS ATTEMPT (cr-068): the group is one
            # executor, so this process is poisoned with it and the worker rebuilds all K.
            self.poisoned = f"group: {self.group.broken}"[:200]

        # §3.4 step 2: the exact D2H wait BEFORE the device lease releases. Everything the
        # tail touches must already be immutable and independent of package GPU state. A
        # WEIGHTLESS generation holds no stream, so its quiescence is structural — there is
        # nothing that could still be moving.
        d2h_started = time.perf_counter()
        if torch is not None:
            accel.synchronize(torch, self.device_kind)
        d2h_ms = (time.perf_counter() - d2h_started) * 1000
        released_us = execution_evidence.now_us()
        peak_vram = accel.peak_allocated(torch, self.device_kind) if torch is not None else 0
        residency_document = self.residency.document() if self.residency is not None else {}
        peak_vram = max(peak_vram, int(residency_document.get("attempt_absolute_peak_bytes", 0)))
        working_peak = _working_peak(peak_vram, allocated_at_start, residency_document)
        if self.group is not None:
            working_peak = max(working_peak, *self.group.working_peaks.values(), 0)
        if torch is not None:
            # An idle tenant holds its weights, not its last call's scratch: the cache goes
            # back to the card, where the next tenant the worker admits can use it.
            accel.release_cached(torch, self.device_kind)
        # step 3: the lease releases here, and nothing after this line touches the device.
        # DEVICE RELEASE IS THIS REPLY (cr-079): the handler registered its outputs as raw
        # host frames, so `device_lease_ms` is device time and the encode is the worker's.
        lease_ms = (time.perf_counter() - lease_acquired) * 1000
        quiescent = self._quiescent() if torch is not None else True
        self.attempts += 1
        result_ref, outputs, oversize = _serialize_result(envelope, spool)
        # Only a failure inside the handler leaves package state unknown. A request refusal,
        # a cooperative cancel/deadline or a failed package call stops at a safe point and
        # keeps the generation Ready.
        if outcome.terminal == "failed" and record.entered and not record.failed_at_call:
            self.poisoned = f"{outcome.terminal}/{outcome.code}"
        terminal = {
            "terminal": outcome.terminal,
            "origin": outcome.origin,
            "code": outcome.code,
            "message": outcome.message[:2048],
            "fields": list(outcome.fields),
            "traceback": outcome.traceback,
        }
        # THE RESIDENCY PLANE'S POISON IS THIS PROCESS'S POISON (#613). A mid-mutation
        # failure or an accounting violation there can fire BEFORE handler entry (admission
        # runs at scope enter), so the phase bit above cannot see it — and the transaction
        # is the process, so the worker must replace it either way.
        if self.residency is not None and self.residency.poisoned:
            self.poisoned = self.poisoned or f"residency: {self.residency.poisoned[:200]}"
        # The BOUNDED observation tail and the O(1) time attribution cross the seam on every
        # path, success or fault. A failed attempt's log tail is the one triage most needs,
        # and cr-011 hoisted the record out of `invoke` precisely so it survives the failure.
        shortfall = self.residency.last_shortfall if self.residency is not None else None
        return {
            "ok": True,
            "shortfall": shortfall,
            "residency": residency_document,
            "observations": record.ring.rows(),
            "observation_caps": record.ring.caps(),
            "attribution": record.attribution.document(),
            "position": record.position,
            "outcome": terminal,
            "execution_observation": self._execution_observation(served_attention),
            "attention_applied": bool(pin),
            "execution": self._execution(
                requested,
                served_attention,
                started_us,
                released_us,
                attention.implementations(served_impls),
            ),
            "capture": capture_result,
            "quiescent": quiescent,
            **_env_document(),
            "poisoned": self.poisoned,
            "outputs": outputs,
            # cr-079: the host frames the saves REGISTERED. No codec ran in this process;
            # the worker's post thread encodes them after the device is released.
            "frames": [frame.row() for frame in record.frames.values()],
            "max_output_bytes": record.max_output_bytes,
            "result_ref": result_ref,
            "result_schema_digest": self._result_schema_digest(surface),
            "adjustments": [
                {
                    "field": row.field,
                    "requested": str(row.requested),
                    "applied": str(row.applied),
                    "reason": f"{row.kind}: {row.reason}"[:256],
                }
                for row in (envelope.adjustments if envelope else ())
            ],
            "ignored": list(record.ignored),
            "oversize_result_bytes": oversize,
            "metrics": {
                "handler_ms": round(handler_ms, 3),
                "device_lease_ms": round(lease_ms, 3),
                "d2h_wait_ms": round(d2h_ms, 3),
                "peak_vram_bytes": peak_vram,
                "activation_peak_bytes": int(
                    residency_document.get("attempt_activation_peak_bytes", 0)
                ),
                "activation_peaks": dict(residency_document.get("attempt_activation_peaks") or {}),
                "allocated_at_start_bytes": allocated_at_start,
                "working_peak_vram_bytes": working_peak,
                "shape_cell": shape_cell(prepared.features.values),
                "rss_at_end_bytes": _rss(),
                "started_unix": started_wall,
                "gpu_count": self.world if torch is not None else 0,
            },
        }

    # ------------------------------------------------------- the per-request attention pin

    def _prepare_attention(
        self,
        binding: Binding,
        roots: Mapping[str, Any],
        pin: str,
        device: Any,
        *,
        choose: Any = None,
        optional_scope: bool = False,
    ) -> attention.Applied:
        """Select locally while preserving the caller's explicit qualified pin."""
        local = _local_attention_pin(pin, binding.parameter_names())
        scope, _ = attention._pin_parts(local)
        if (
            optional_scope
            and "/" not in attention._pin_parts(pin)[0]
            and scope
            and not any(
                attention._matches(component, scope)
                for component, _, _, _ in attention._attention_records(roots)
            )
        ):
            local = ""
        applied = attention.select(device, roots, local, degree=self.world, choose=choose)
        return replace(applied, pin=pin) if local else applied

    def _upgrade_attention(self) -> None:
        """A kernel this construction skipped while it compiled joins at a request boundary,
        never inside a request: the construction selects again and the group agrees. A
        compiled graph holds its kernel, so a construction warmed into one keeps it."""
        for executor in self.model_executors or [self]:
            applied, device = executor.attention, executor.device_facts
            if applied is None or executor._reselect is None or device is None:
                continue
            if not attention.pending(applied, device) or executor.backend is None:
                continue
            roots = {**executor.backend.components, **executor.backend.parked}
            if attention._baked_in(roots):
                continue
            upgraded = executor._reselect()
            executor.attention = replace(upgraded, pin=applied.pin)
            executor.attention_defaults = attention.snapshot(roots)
            print(f"[attention] upgraded: {upgraded.line()}", file=sys.stderr, flush=True)
            if self.group is not None and self.rank == 0:
                refused = executor._agree_attention(upgraded)
                if refused is not None:
                    raise attention.AttentionRefusal(
                        str(refused.get("code")), str(refused.get("detail"))
                    )

    def _attention_snapshot(self) -> attention.Snapshot:
        return attention.Snapshot(
            tuple(
                choice
                for executor in (self.model_executors or [self])
                for choice in executor.attention_defaults.choices
            )
        )

    def _restore_attention(self) -> None:
        self._attention_snapshot().restore()

    def _attention_roots(self) -> dict[str, Any]:
        roots: dict[str, Any] = {}
        executors = self.model_executors or [self]
        for executor in executors:
            if executor.backend is None:
                continue
            # A qualified request names its model even when it is the only one.
            # Component-only pins still match through attention._matches.
            prefix = "+".join(sorted(executor.models)) + "/" if executor.models else ""
            for name, root in {**executor.backend.components, **executor.backend.parked}.items():
                roots[prefix + name] = root
        return roots

    def _attention_reaches(self, pin: str) -> bool:
        try:
            return attention.reaches(self._attention_roots(), pin)
        except attention.AttentionRefusal:
            return True  # a malformed pin refuses where it is planned

    def _attention_device(self, pin: str) -> Any:
        device = next(
            (e.device_facts for e in (self.model_executors or [self]) if e.device_facts), None
        )
        if device is None:
            raise attention.AttentionRefusal(
                "attention_kernel_unsupported",
                f"{pin} was pinned on a construction that reached no device",
            )
        return device

    def _attention_plan(self, pin: str) -> attention.Swap | None:
        """This rank's swap for one request's pin, admitted by every site, the device and
        the group degree; none when a compiled construction already holds exactly it."""
        roots = self._attention_roots()
        if attention._baked_in(roots) and attention._is_selected(
            self._attention_snapshot(), roots, pin
        ):
            return None
        return attention.plan(self._attention_device(pin), roots, pin, self.world)

    def _agree_attention(self, applied: attention.Applied) -> dict[str, Any] | None:
        """THE GROUP SERVES RANK 0'S PREPARED SELECTION. Each rank selected beside rank 0
        under its own probes, so a follower takes rank 0's per-component kernels before the
        group installs; one that cannot serve them refuses the group instead of serving a
        mixed selection that every request would then refuse."""
        assert self.group is not None
        chosen = {c: next(iter(k)) for c, k in applied.hosts.items() if len(k) == 1}
        try:
            self.group.broadcast(
                executor_commands.encode(
                    Attention(
                        construction=self.construction,
                        model_key=self._group_model_key,
                        adopt=chosen,
                    )
                )
            )
            replies = self.group.collect("attention")
        except GroupRefusal as exc:
            return self._group_refusal(exc)
        for follower, reply in zip(self.group.followers, replies, strict=True):
            if not reply.get("ok") or reply.get("hosts") != applied.hosts:
                return self._group_refusal(
                    GpuDivergence(
                        follower.gpu,
                        "attention",
                        f"{_gpu(0)} prepared {applied.hosts}, {follower.gpu} holds "
                        f"{reply.get('hosts')} {reply.get('code', '')} {reply.get('detail', '')}",
                    )
                )
        return None

    def _adopt_attention(self, command: Attention, adopt: dict[str, str]) -> dict[str, Any]:
        """A follower's half of `_agree_attention`: take rank 0's kernel per component as
        this construction's prepared selection."""
        container = self.constructions.get(command.construction)
        members = (container.model_executors or [container]) if container is not None else []
        target = next((m for m in members if m._group_model_key == command.model_key), None)
        if self.rank == 0 or target is None or target.backend is None:
            return {"ok": False, "code": "gpu_divergence", "detail": "no such construction"}
        roots = {**target.backend.components, **target.backend.parked}
        device, applied = target.device_facts, target.attention
        if (
            target._reselect is not None
            and applied is not None
            and device is not None
            and attention.pending(applied, device)
        ):
            # Rank 0 upgraded at a request boundary; a kernel this rank skipped while it
            # compiled joins here too, e.g. Sol's dense reference, which `adopt` cannot name.
            target.attention = replace(target._reselect(), pin=applied.pin)
        held = attention.observed(roots).hosts
        try:
            for component, name in sorted(adopt.items()):
                if set(held.get(component) or ()) == {name}:
                    continue
                if component not in roots or target.device_facts is None:
                    raise attention.AttentionRefusal(
                        "attention_kernel_unsupported", f"{_gpu(self.rank)} holds no {component}"
                    )
                attention.plan(
                    target.device_facts, {component: roots[component]}, name, self.world
                ).apply()
                print(
                    f"[attention] {_gpu(self.rank)} {component}: took {_gpu(0)}'s {name} over "
                    f"its own {sorted(held.get(component) or ())}",
                    file=sys.stderr,
                    flush=True,
                )
        except attention.AttentionRefusal as exc:
            return {"ok": False, "code": exc.code, "detail": str(exc)[:900], "rank": self.rank}
        target.attention_defaults = attention.snapshot(roots)
        hosts = attention.observed(roots).hosts
        if target.attention is not None:
            target.attention = replace(target.attention, hosts=hosts)
        return {"ok": True, "rank": self.rank, "hosts": hosts}

    def select_attention(self, command: Attention) -> dict[str, Any]:
        """A FOLLOWER restores its prepared selection and, given rank 0's `pin` for the next
        attempt (empty is the prepared one), holds it and answers what its sites now hold."""
        if command.adopt is not UNSET:
            return self._adopt_attention(command, command.adopt)
        self._attention_hold.close()
        if command.pin is UNSET:
            return {"ok": True, "rank": self.rank}
        target = self.constructions.get(command.construction)
        if self.rank == 0 or target is None:
            return {"ok": False, "code": "gpu_divergence", "detail": "no such construction"}
        pin = command.pin
        try:
            swap = target._attention_plan(pin) if pin else None
            held = self._attention_hold.enter_context(
                attention.override(
                    target._attention_snapshot(), target._attention_roots(), swap, pin
                )
            )
        except attention.AttentionRefusal as exc:
            return {"ok": False, "code": exc.code, "detail": str(exc)[:900], "rank": self.rank}
        return {"ok": True, "rank": self.rank, "hosts": held.hosts}

    def _followers_attention(self, pin: str | UnsetType = UNSET) -> list[dict[str, Any]]:
        assert self.group is not None
        self.group.broadcast(
            executor_commands.encode(Attention(construction=self.construction, pin=pin))
        )
        replies = self.group.collect("attention")
        for reply in (reply for reply in replies if not reply.get("ok")):
            detail = f"{_gpu(int(reply.get('rank', -1)))}: {reply.get('detail', '')}"[:900]
            raise InvalidRequest(detail, code=str(reply.get("code")))
        return replies

    @contextlib.contextmanager
    def _ranks_attention(self, pin: str, applied: attention.Applied) -> Iterator[None]:
        """Every follower serves this attempt's selection and restores its prepared one
        after. Diffusers reads a processor's backend and parallel config on every call, so a
        group swaps per attempt as one process does; rank 0 refuses what a rank does not hold.
        """
        if self.group is None:
            yield
            return
        try:
            held = {r["rank"]: r["hosts"] for r in self._followers_attention(pin=pin)}
            if diverged := sorted(r for r, hosts in held.items() if hosts != applied.hosts):
                gpus = ", ".join(_gpu(rank) for rank in diverged)
                detail = f"{gpus} do not hold {_gpu(0)}'s {applied.hosts}"[:900]
                raise InvalidRequest(detail, code="attention_gpu_divergence")
            yield
        finally:
            self._followers_attention()

    @staticmethod
    def _capture_options(
        command: PrepareRequest | Invoke, surface: Any
    ) -> ActivationCapture | None:
        from cozy_runtime.probe._capture import validate_components

        if command.capture is None:
            return None
        options = msgspec.convert(command.capture, type=ActivationCapture, strict=True)
        validate_components(options, [binding.model_class for binding in surface.model_bindings])
        return options

    def _capture_session(self, command: PrepareRequest | Invoke, surface: Any) -> Any:
        from cozy_runtime.author._errors import InvalidRequest
        from cozy_runtime.probe._capture import CaptureSession

        options = self._capture_options(command, surface)
        if options is None:
            return None
        roots: dict[str, Any] = {}
        for executor in self.model_executors or [self]:
            if executor.backend is None:
                continue
            for name, root in {**executor.backend.components, **executor.backend.parked}.items():
                if name not in options.components:
                    continue
                if name in roots and roots[name] is not root:
                    raise InvalidRequest(
                        f"capture component {name!r} occurs in more than one bound model",
                        code="capture_component_ambiguous",
                    )
                roots[name] = root
        return CaptureSession(options, roots)

    @staticmethod
    def _emit_attention(record: AuthorAttempt, choices: attention.Applied, name: str) -> None:
        for kernel, artifact in (choices.artifacts or {}).items():
            record.emit(
                "confession",
                "attention.artifact",
                kernel,
                selection=name,
                **artifact,
            )
        for component, kernels in choices.hosts.items():
            for kernel, sites in kernels.items():
                record.emit(
                    "confession",
                    name,
                    kernel,
                    component=component,
                    sites=sites,
                    pin=choices.pin,
                    dense_reference=attention_sol.dense_identity() if kernel == "sol-attn" else "",
                )

    def _execution_observation(self, choices: attention.Applied | None = None) -> dict[str, Any]:
        from cozy_runtime import __version__

        # Observed generation facts; image and boot identity are added by the worker.
        # Optional kernel/lane fields stay absent until their execution is observed.
        return {
            "runtime_version": __version__,
            "accelerator": str(self.facts.get("gpu_name") or "CPU"),
            "cuda": str(self.facts.get("cuda_version") or ""),
            "driver": str(self.facts.get("device_driver") or ""),
            "execution_lane": "",  # no generic eager/compiled observation exists yet
            "execution_contract_digest": "",
            "kernel_symbol": ",".join(sorted(choices.totals())) if choices is not None else "",
        }

    # ------------------------------------------------------------------ job

    def _job_models(self, surface: Any, command: RunJob) -> dict[str, Any]:
        """Create typed Manifest-only Model instances; construct/load no package state."""
        declared = {binding.param: binding for binding in surface.model_bindings}
        rows = command.models
        if not set(declared) <= set(rows):
            raise CapabilityError(
                f"job declares Model parameters {sorted(declared) or 'none'} and the worker "
                f"passed {sorted(rows)}",
                code="job_model_binding_mismatch",
            )
        payload = command.payload if isinstance(command.payload, Mapping) else {}
        models: dict[str, Any] = {}
        for parameter in sorted(declared):
            row, binding = rows[parameter], declared[parameter]
            if row.class_key != binding.class_key:
                raise CapabilityError(
                    f"job Model parameter {parameter!r} changed its interface class",
                    code="job_model_class_mismatch",
                )
            if surface.invocable and (artifact := payload.get(parameter)) is not None:
                expected = artifact.get("manifest") if isinstance(artifact, Mapping) else None
                if (
                    not isinstance(expected, Mapping)
                    or expected.get("digest") != row.manifest
                    or expected.get("length") != row.length
                ):
                    raise CapabilityError(
                        "model call artifact differs from its admitted native input",
                        code="model_artifact_binding",
                    )
            models[parameter] = _derive_model(binding.model_class, row.manifest)
        return models

    def run_job(self, command: RunJob) -> dict[str, Any]:
        """ONE `@app.job`, run to completion through the SAME invocation kernel (cr-009).

        Everything structural about a job is here and it is short, which is the point: a job
        is not a second runtime, it is the one attempt kernel with a different service set
        and no serving loop underneath it. The three differences from `invoke`, out loud:

        1. **No prepared generation is required.** The admitted slot's device count
           determines accelerator initialization. A CPU job imports no Torch and opens no
           CUDA context; a GPU transform can use its device without constructing an
           inference model. A job Model remains a derive-only Manifest capability.
        2. **Job services are injected**: a resumable Scratch tree keyed on the RUN, the
           durable checkpoint exchange, and the typed budget facts.
        3. **Quiescence is proven by what exists.** A torch-free job has no stream that
           could still be moving, so the proof is structural rather than a device query.
        """
        if command.capture is not None:
            return {
                "ok": False,
                "code": "capture_mode",
                "detail": "activation capture requires an ordinary serving invocation",
                "terminal": "refused",
                "origin": "request",
            }
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        budget = command.budget
        if budget.gpu_count > 0:
            import torch

            self.torch = torch
            refused = self._initialize_device(torch)
            if refused is not None:
                return refused
        if self.discovered is None:
            try:
                self._install_call_proxies(command.call_interfaces)
                self._discover_provisioned(
                    Release(
                        application=command.application,
                        package_interface=command.package_interface,
                    )
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "code": "job_release_undiscoverable",
                    "detail": f"{type(exc).__name__}: {exc}"[:1024],
                }
        name = command.job
        surface = self.surface_names.get(name)
        if surface is None or surface.kind != "job":
            return {
                "ok": False,
                "code": "unknown_job",
                "detail": f"{name!r} is not a registered @app.job of this release (it has "
                f"{sorted(s.name for s in self.surface_names.values() if s.kind == 'job')})",
            }
        registration = self.discovered.app.get(name)
        spool = Path(command.spool)
        scratch = Path(command.scratch) if command.scratch else None
        if scratch is not None:
            scratch.mkdir(parents=True, exist_ok=True)

        try:
            job_models = self._job_models(surface, command)
        except CapabilityError as exc:
            return {"ok": False, "code": exc.code, "detail": str(exc)[:1024]}

        weights_values = None
        if "weights_read" in surface.capabilities:
            from cozy_runtime.internal.model_values import read_values_into

            weights_values = read_values_into

        from cozy_runtime.internal.weights_writer import ExecutionStorage

        execution_storage = ExecutionStorage(
            spool,
            self._durable,
            {output.name: output.max_new_bytes for output in surface.weights_outputs},
        )
        tensorfs_source = execution_storage.source
        tensorfs_output = execution_storage.open_output
        tensorfs_adopt = execution_storage.adopt_model

        from cozy_runtime.author._services import (
            BudgetFacts,
            CheckpointDeclaration,
            CheckpointSave,
        )

        def durable_checkpoint(save: CheckpointSave) -> CheckpointDeclaration:
            """Block until the worker has JOURNALED the declaration. Never the lossy lane.

            What comes back is a recorded declaration id, not a durability receipt: this
            frame carries keys, a digest and a length, and the bytes stay in the spool
            (#553a).
            """
            answer = self._durable(
                Checkpoint(
                    operation_key=save.operation_key,
                    logical_key=save.logical_key,
                    content_digest=save.content_digest,
                    length=save.length,
                ),
                CheckpointReceipt,
            )
            if not answer.ok:
                raise CheckpointConflict(f"{save.logical_key}: {answer.detail}", code=answer.code)
            return CheckpointDeclaration(
                declaration_id=answer.receipt_id,
                operation_key=save.operation_key,
                logical_key=save.logical_key,
                content_digest=save.content_digest,
                length=save.length,
                replayed=answer.replayed,
            )

        started = time.perf_counter()
        started_wall = time.time()
        if self.torch is not None:
            accel.reset_peak(self.torch, self.device_kind)
        envelope, outcome, record = attempt(
            registration,
            command.payload,
            Invocation(
                request_id=command.request_id,
                spool=spool,
                deadline=_deadline(command.deadline_s),
                calls=self._call_broker(command.request_id, command.call_interfaces),
                publish_to=command.publish_to,
                publish=self._publish,
                device=self._device() if self.torch is not None else Device(),
                cancel=lambda: _cancelled(self.root, command.request_id),
                models=job_models,
                trees=_trees(command),
                assets=_granted_inputs(command),
                max_input_bytes=command.max_input_bytes or 64 << 20,
                max_output_bytes=command.max_output_bytes,
                scratch=scratch,
                checkpoints=durable_checkpoint,
                weights_values=weights_values,
                tensorfs_source=tensorfs_source,
                tensorfs_output=tensorfs_output,
                tensorfs_adopt=tensorfs_adopt,
                budget=BudgetFacts(
                    gpu_rate_usd_per_hour=budget.gpu_rate_micro_usd_per_hour / 1e6,
                    gpu_count=budget.gpu_count,
                    cap_usd=budget.cap_micro_usd / 1e6,
                    started_monotonic=time.monotonic(),
                ),
                progress=self._progress_sink(command.request_id),
                activity=self._activity_sink(command.request_id),
            ),
        )
        handler_ms = (time.perf_counter() - started) * 1000
        peak_vram = 0
        if self.torch is not None:
            accel.synchronize(self.torch, self.device_kind)
            peak_vram = accel.peak_allocated(self.torch, self.device_kind)
        self.attempts += 1
        result_ref, outputs, oversize = _serialize_result(envelope, spool)
        if outcome.terminal == "failed" and record.entered and not record.failed_at_call:
            self.poisoned = f"{outcome.terminal}/{outcome.code}"
        return {
            "ok": True,
            "kind": "job",
            "shortfall": None,
            "residency": {},
            "execution_observation": self._execution_observation(),
            "observations": record.ring.rows(),
            "observation_caps": {**record.ring.caps(), "metrics_dropped": record.metrics_dropped},
            "attribution": record.attribution.document(),
            "position": record.position,
            "outcome": {
                "terminal": outcome.terminal,
                "origin": outcome.origin,
                "code": outcome.code,
                "message": outcome.message[:2048],
                "fields": list(outcome.fields),
            },
            # A torch-free job holds no stream, so quiescence is a STRUCTURAL fact rather
            # than a device query: there is nothing that could still be moving.
            "quiescent": self._quiescent() if self.torch is not None else True,
            **_env_document(),
            "poisoned": self.poisoned,
            "outputs": outputs,
            "frames": [frame.row() for frame in record.frames.values()],
            "max_output_bytes": record.max_output_bytes,
            "result_ref": result_ref,
            "result_schema_digest": self._result_schema_digest(surface),
            "adjustments": [],
            "ignored": list(record.ignored),
            "oversize_result_bytes": oversize,
            "metrics": {
                "handler_ms": round(handler_ms, 3),
                "device_lease_ms": round(handler_ms, 3),
                "d2h_wait_ms": 0.0,
                "peak_vram_bytes": peak_vram,
                "allocated_at_start_bytes": 0,
                "rss_at_end_bytes": _rss(),
                "started_unix": started_wall,
                "gpu_count": 1 if self.torch is not None else 0,
            },
        }

    def _install_call_proxies(self, rows: tuple[CallInterface, ...]) -> None:
        """Expose managed dependency imports before importing a private workflow.

        App registrations keep their original implementations. Only public module
        exports in this caller process become typed proxies; installed files and
        the independently executing callee remain ordinary uv installations.
        """
        from types import ModuleType

        from cozy_runtime.internal import call_types, interface_wheel

        prepared: set[str] = set()
        for row in rows:
            document, module = row.interface_document, row.module
            if not document or module in prepared:
                continue
            prepared.add(module)
            imported = importlib.import_module(module)
            generated = interface_wheel.generate(package_interface.canonical_bytes(document))
            source = generated.get(module.replace(".", "/") + "/__init__.py")
            if source is None:
                raise CapabilityError(
                    "installed dependency export is absent", code="child_interface"
                )
            companion_name = module + "._cozy_runtime_callers"
            companion = ModuleType(companion_name)
            sys.modules[companion_name] = companion
            exec(compile(source, "<installed package callers>", "exec"), companion.__dict__)
            bindings = {
                export: call_types.public_result(imported, binding)
                for export, binding in companion.__dict__["__cozy_bindings__"].items()
            }
            imported.__dict__["__cozy_bindings__"] = bindings
            for export in bindings:
                imported.__dict__[export] = companion.__dict__[export]

    def _call_broker(self, request_id: str, rows: tuple[CallInterface, ...]) -> Any:
        from cozy_runtime.author._calls import _Broker, _CallType

        if not rows:
            return None
        if len(rows) > 256:
            raise CapabilityError(
                "call interface inventory exceeds its bound", code="child_interface"
            )
        bindings: dict[tuple[str, str], _CallType] = {}
        own = native_interfaces.bindings()
        for row in rows:
            interface, module, export = row.interface_digest, row.module, row.export
            if row.native_source or row.native_effect:
                # Admission is per operation: the worker judged this one against what this
                # Runtime stated at hello. One this Runtime lacks cannot be called from here.
                native = own.get((module, export))
                if native is None:
                    continue
                if row.request_fields is not None:
                    native = dataclasses.replace(
                        native, served_fields=frozenset(row.request_fields)
                    )
                if row.unavailable:
                    native = dataclasses.replace(native, unavailable=row.unavailable[:1024])
                bindings[(module, export)] = native
            elif row.builtin:
                from cozy_runtime.internal import builtin_operations

                if row.builtin != "operations":
                    raise CapabilityError("unknown Runtime builtin", code="child_interface")
                builtin_operations.validate_binding(interface, module, export)
                from cozy_runtime.author import describe
                from cozy_runtime.derive.operations import app

                describe(app)
                builtin_surface = app.get(export).surface
                request_type, result_type = (
                    builtin_surface.payload_type,
                    builtin_surface.result_type,
                )
                if not (
                    isinstance(request_type, type)
                    and issubclass(request_type, msgspec.Struct)
                    and isinstance(result_type, type)
                    and issubclass(result_type, msgspec.Struct)
                ):
                    raise CapabilityError("Runtime builtin types changed", code="child_interface")
                binding = _CallType(interface, module, export, request_type, result_type)
                bindings[(module, export)] = binding
            elif row.self_call:
                assert self.discovered is not None
                surface = next(
                    (
                        surface
                        for surface in self.discovered.surfaces
                        if (surface.invocable or surface.kind == "entrypoint")
                        and surface.fn.__module__ == module
                        and surface.fn.__name__ == export
                    ),
                    None,
                )
                if surface is None:
                    raise CapabilityError(
                        "self call is not a registered App export", code="child_interface"
                    )
                binding = _CallType(
                    interface, module, export, surface.payload_type, surface.result_type
                )
                bindings[(module, export)] = binding
            else:
                imported = importlib.import_module(module)
                from cozy_runtime.author._app import Registration
                from cozy_runtime.author._calls import _export

                function = getattr(imported, export, None)
                declared = _export(function) if callable(function) else None
                if declared is not None:
                    surface = Registration(
                        name=export,
                        kind=row.kind,
                        fn=declared.implementation,
                        invocable=True,
                        memoize=declared.memoize,
                        model_defaults=declared.defaults,
                    ).surface
                    request_type, result_type = surface.payload_type, surface.result_type
                    if not (
                        isinstance(request_type, type)
                        and issubclass(request_type, msgspec.Struct)
                        and isinstance(result_type, type)
                        and issubclass(result_type, msgspec.Struct)
                    ):
                        raise CapabilityError(
                            "invocable types are not structured", code="child_interface"
                        )
                    binding = _CallType("", module, export, request_type, result_type)
                else:
                    generated_binding = getattr(imported, "__cozy_bindings__", {}).get(export)
                    if not isinstance(generated_binding, _CallType):
                        raise CapabilityError(
                            "installed export is not invocable", code="child_interface"
                        )
                    binding = generated_binding
                # Source-preserving caller wheels retain ordinary @invocable job
                # wrappers, whose lookup is by their exact module/export pair.
                prior = bindings.get((module, export))
                if prior is not None and prior != binding:
                    raise CapabilityError("ambiguous managed export", code="child_interface")
                bindings[(module, export)] = binding
        for module, export in own.keys() - bindings.keys():
            bindings[(module, export)] = native_interfaces.unserved(module, export)
        return _Broker(request_id, bindings, self._durable, self._child_events)

    def _publish(self, request: Publish) -> Published:
        """One product onto the run's output log, durable before this returns."""
        return self._durable(request, Published)

    def _child_events(self) -> int | None:
        """The worker's nudge socket for this attempt's calls; None from an older worker."""
        answer = self._durable(ChildEvents(), Handoff)
        return answer.descriptor if answer.ok else None

    def _durable[A: Answer](self, request: Request, into: type[A]) -> A:
        """One DURABLE mid-attempt exchange. Blocks until the worker answers.

        This is the seam's other half of `child.Executor.call(on_request=…)`. It is a
        separate lane from progress on purpose: a durable fact must not be sheddable, and
        the lossy lane's whole contract is that it may shed.
        """
        body = encode_request(request)
        with self._durable_lock:
            self._exchange += 1
            exchange = f"{body['kind']}#{self._exchange}"
            self.channel.send({"event": "request", "seq": self._exchange, **body})
            frame = self.channel.recv()
            if frame is None:
                raise SeamError("seam_closed", f"the worker closed during {exchange}")
            if frame.get("event") != "answer" or frame.get("seq") != self._exchange:
                raise SeamError(
                    "seam_out_of_order",
                    f"expected the answer to {exchange}, got {frame.get('event')!r}",
                )
            handoff = issubclass(into, Handoff)
            if frame.get("descriptor") is True:
                if not handoff:
                    self.channel.close()
                    raise SeamError("seam_descriptor", "unexpected capability in control reply")
                frame["descriptor"] = self.channel.recv_descriptor()
            elif handoff and frame.get("ok") is True:
                raise SeamError("seam_descriptor", "successful reply omitted its capability")
        try:
            return tolerant.read(frame, into, ("progress", "observation"))[0]
        except msgspec.ValidationError as exc:
            raise SeamError("seam_answer_malformed", f"{exchange}: {exc}") from exc

    def _activity_sink(self, request_id: str) -> Callable[[int, bool, bool, bool], None]:
        def activity(sequence: int, active: bool, known: bool, finished: bool) -> None:
            self.channel.send(
                {
                    "event": "progress",
                    "request_id": request_id,
                    "kind": "execution_activity",
                    "sequence": sequence,
                    "active": active,
                    "known": known,
                    "finished": finished,
                }
            )

        return activity

    def _progress_sink(self, request_id: str) -> Any:
        """The lossy lane's sink for one attempt — the same one `invoke` installs."""

        def progress(frame: Any) -> None:
            with contextlib.suppress(SeamError):
                if isinstance(frame, ProgressFrame):
                    self.channel.send(
                        {
                            "event": "progress",
                            "request_id": request_id,
                            "kind": "progress",
                            "stage": frame.stage,
                            "stage_fraction": frame.stage_fraction,
                            "overall_fraction": frame.overall_fraction,
                            "position": frame.position,
                            "total": frame.total,
                            "advance": frame.advance,
                            "step_ms": frame.step_ms,
                            "call_request": frame.call_request,
                            "call_attempt": frame.call_attempt,
                        }
                    )
                else:
                    self.channel.send(
                        {
                            "event": "progress",
                            "request_id": request_id,
                            "kind": frame.kind,
                            "name": frame.name,
                            "value": frame.value,
                            "fields": dict(frame.fields),
                            "seq": frame.seq,
                            "at_unix_ms": frame.at_unix_ms,
                        }
                    )

        return progress

    def _result_schema_digest(self, surface: Any) -> str:
        """The identity of the DECLARED annotated result surface — never of the value.

        A digest over the value document would change with every request and would fence
        nothing; the RecordOwner needs to know which CONTRACT the bytes satisfy.
        """
        from cozy_runtime.internal import schema

        cached = self.schema_digests.get(surface.name)
        if cached is None:
            cached = self.schema_digests[surface.name] = canonical.digest(
                schema.render(surface.result_type, decoded_bounds=True)
            )
        return cached

    def _device(self) -> Device:
        """No torch in this process means no device to name (cl-013). Ordinal 0 IS the
        lane's device: the worker sealed this process to exactly its lane (cr-066)."""
        return Device(self.device_kind, 0) if self.torch is not None else Device()

    def _quiescent(self) -> bool:
        """No author computation may still mutate GPU or output state when a terminal is
        recorded. The proof is the device's own: every stream is complete."""
        return accel.streams_idle(self.torch, self.device_kind)

    def vacate(self, residents: bool = True, ranks: tuple[int, ...] = ()) -> dict[str, Any]:
        """FREE this process's device bytes for another tenant (cr-022), on `ranks` of a
        group (every rank when empty): only the cards the worker needs are given back.

        The worker calls this between attempts, when a co-tenant's attempt or load does not
        fit beside what this process holds. Every resident construction is parked; the
        process stays, `restore` brings them back and an attempt's own admissions stage
        whatever is still missing. Without `residents` only the allocator's cache goes back.
        The reply is the probe's after-state plus what moved.
        """
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        empty = {"freed": {}, "held": {}, "freed_bytes": 0}
        if self.torch is None:
            return {**self.probe(), "vacated": empty}
        chosen = set(ranks) if ranks else set(range(self.world))
        followers = sorted(rank for rank in chosen if 0 < rank < self.world)
        command = executor_commands.encode(Vacate(residents=residents))
        if self.group is not None:
            try:
                for rank in followers:
                    self.group.send_one(rank, command)
            except GroupRefusal as exc:
                return self._group_refusal(exc, closing=False)
        reports: list[Any] = []
        if 0 in chosen:
            self._synchronize()
            try:
                victims = self.arbiter.victims(exclude="") if residents else []
                reports = [self.arbiter.park(key) for key in victims]
            except ResidencyRefusal as exc:
                return {"ok": False, "code": exc.code, "detail": exc.detail[:900]}
            accel.release_cached(self.torch, self.device_kind)
        vacated = {
            "freed": {k: v for report in reports for k, v in report.get("freed", {}).items()},
            "held": {k: v for report in reports for k, v in report.get("held", {}).items()},
            "freed_bytes": sum(int(report.get("freed_bytes", 0)) for report in reports),
            "allocator_delta_bytes": sum(
                int(report.get("allocator_delta_bytes", 0)) for report in reports
            ),
            "vacate_ms": round(sum(float(report.get("vacate_ms", 0)) for report in reports), 2),
        }
        if self.group is not None:
            try:
                for rank in followers:
                    reply = self.group.reply_one(rank, "vacate")
                    if not reply.get("ok"):
                        raise GroupRefusal(
                            f"{_gpu(rank)} refused to vacate: {reply.get('code')}: "
                            f"{reply.get('detail', '')}"[:600],
                            code="group_broken",
                        )
            except GroupRefusal as exc:
                return self._group_refusal(exc, closing=False)
        return {**self.probe(), "vacated": vacated}

    def restore(self, names: tuple[str, ...] = ()) -> dict[str, Any]:
        """RE-FILL what a vacate took from the ACTIVE construction, when the card has room
        (cr-097).

        The worker calls this between attempts, under the same lane seat, BEFORE it prices
        the attempt -- so the plan is chosen against the residency the tenant is entitled
        to rather than against the hole a co-tenant left in it. It evicts nothing; a
        component that does not fit is simply not restored.
        """
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        empty = {"restored": {}, "held": {}, "restored_bytes": 0}
        if self.torch is None or self.active not in self.constructions:
            return {**self.probe(), "restored": empty}
        if self.group is not None:
            # EVERY RANK RESTORES (cr-068), for the reason every rank vacates: the ranks
            # hold one plan between them and a partially restored group is not a plan.
            try:
                self.group.broadcast(executor_commands.encode(Restore(names=names)))
            except GroupRefusal as exc:
                return self._group_refusal(exc, closing=False)
        try:
            restored = self.arbiter.restore(self.active, names)
        except ResidencyRefusal as exc:
            return {"ok": False, "code": exc.code, "detail": exc.detail[:900]}
        self._synchronize()
        if self.group is not None:
            try:
                for reply in self.group.collect("restore"):
                    if not reply.get("ok"):
                        raise GroupRefusal(
                            f"a follower refused to restore: {reply.get('code')}: "
                            f"{reply.get('detail', '')}"[:600],
                            code="group_broken",
                        )
            except GroupRefusal as exc:
                return self._group_refusal(exc, closing=False)
        return {**self.probe(), "restored": restored}

    def _active_residency(self) -> dict[str, Any]:
        child = self.constructions.get(self.active)
        if child is None or child.residency is None:
            return {}
        return dict(child.residency.document())

    def process_incarnations(self) -> list[dict[str, Any]]:
        """The actual rank births, including a follower that vanished before this probe."""
        ranks = [(self.rank, os.getpid())]
        if self.group is not None:
            ranks.extend((follower.rank, follower.pid) for follower in self.group.followers)
        rows: list[dict[str, Any]] = []
        for rank, pid in ranks:
            row: dict[str, Any] = {"rank": rank, "pid": pid, "started_ticks": None}
            with contextlib.suppress(OSError, proctree.ProcessTreeUnsupported):
                row["started_ticks"] = proctree.process_identity(pid).started_ticks
            rows.append(row)
        return rows

    def probe(self) -> dict[str, Any]:
        processes = self.process_incarnations()
        if self.torch is None:
            # A process can be poisoned before Torch loads (a follower's rank divergence).
            return {
                "ok": True,
                "torch": False,
                "rss_bytes": _rss(),
                "processes": processes,
                "poisoned": self.poisoned,
            }
        # A torch-bearing process with no card (a CPU executor, a gloo rank) has no device
        # numbers to report; readers take absent as unknown.
        device: dict[str, int] = {}
        if self.device_kind != "mps" and accel.present(self.torch, self.device_kind):
            memory = accel.allocation(self.torch, self.device_kind)
            device = {
                "device_free_bytes": memory["driver_free_bytes"],
                "device_total_bytes": memory["driver_total_bytes"],
                "allocator_bytes": memory["allocated_bytes"],
                "reserved_bytes": memory["reserved_bytes"],
            }
        return {
            "ok": True,
            "torch": True,
            **device,
            "residency": self._active_residency(),
            "constructions": self.arbiter.document(self.active),
            "rss_bytes": _rss(),
            "attempts": self.attempts,
            "processes": processes,
            "ready": self.ready,
            "poisoned": self.poisoned,
            **_env_document(),
        }


#: Telemetry a reply can lose to fit one frame; the worker reads each as optional.
_SHEDDABLE = ("execution", "attribution")

#: The keys the minimal reply below actually carries. A reply whose own keys are a subset
#: of these loses nothing by being reduced; anything else would be silently mutilated.
_MINIMAL_KEYS = frozenset(
    {
        "reply",
        "ok",
        "code",
        "detail",
        "outcome",
        "execution_observation",
        "capture",
        "quiescent",
        "env_intact",
        "env_changed",
        "poisoned",
        "metrics",
        "outputs",
        "result_ref",
        "result_schema_digest",
        "oversize_result_bytes",
        "shortfall",
        "residency",
        "adjustments",
        "observations",
        "observation_caps",
        "terminal",
        "origin",
        "fields",
    }
)


def _send_reply(channel: Channel, reply: dict[str, Any]) -> None:
    """Put ONE reply on the seam, and MAKE IT FIT.

    The observation ring's own budget is 128 KiB and one control frame is 64 KiB, so a tail
    the emit boundary bounded correctly can still produce a reply the seam refuses. That
    refusal used to fire HERE, outside every handler, killing the executor — and the attempt
    lost its real terminal to an ABANDONED that named the death instead of the refusal.
    Reproduced from job-001's shape: a body that narrates a border and then refuses at the
    host-fit point sends 134,743 B over a 65,536 B cap.

    The TAIL is the only part of a reply that can grow, so the tail is what gives way, and
    what it gave up is a NUMBER on `observation_caps` — the same rule the ring already
    follows. A reply that does not fit even empty ships the OUTCOME, which is the one thing
    a RecordOwner cannot reconstruct from anywhere else.

    AND THE FALLBACK NEVER CLAIMS SUCCESS. It was written for INVOKE and used for every
    command, and it copied `ok` through while dropping every key that makes an ok reply
    mean anything — so an oversized PREPARE reply arrived as `{"ok": True}` with no
    construction digest, no plan and no facts, and the worker read a key that was not
    there. That KeyError killed the control stream AFTER the fill had put 90.84 GiB on the
    card; the client reconnected, the placement was re-accepted, and prepare ran a SECOND
    time in the SAME executor — 90.84 + 48.18 GiB on a 139.81 GiB card, which is the whole
    of the "the fill allocates 63% more than it accounts for" gap (#574b). The fill plane
    was never wrong. A reply this path had to mutilate is a REFUSAL that names itself, so
    the caller takes the refusal branch instead of reading a success that is not one.
    """
    held = len(reply.get("observations") or ())
    rows = list(reply.get("observations") or ())
    # Telemetry the worker reads as optional goes next, before any result is refused.
    shed = [key for key in _SHEDDABLE if key in reply]
    while True:
        try:
            channel.send(reply)
            return
        except SeamError as exc:
            # A telemetry key the seam reads as a credential sheds the same way.
            if exc.code not in ("frame_too_large", "credential_at_seam") or not (rows or shed):
                break
            caps = dict(reply.get("observation_caps") or {})
            if rows:
                rows = rows[(len(rows) + 1) // 2 :]
                reply["observations"] = rows
                caps["dropped_for_frame"] = held - len(rows)
            else:
                key = shed.pop()
                del reply[key]
                caps[f"dropped_{key}_for_frame"] = 1
            reply["observation_caps"] = caps
    reduced = not reply.get("ok") or set(reply) <= _MINIMAL_KEYS
    channel.send(
        {
            "reply": reply.get("reply"),
            "ok": reply.get("ok") if reduced else False,
            "code": reply.get("code") if reduced else "reply_too_large",
            "detail": str(reply.get("detail", ""))[:512]
            if reduced
            else (
                f"the {reply.get('reply')!r} reply does not fit one control frame and the "
                f"keys it declares ({', '.join(sorted(set(reply) - _MINIMAL_KEYS))[:200]}) "
                "cannot be dropped without inventing a success. It refuses instead: a "
                "RecordOwner acting on a reply that lost its own contents is worse than one "
                "told the reply could not be sent"
            ),
            "outcome": reply.get("outcome"),
            "quiescent": reply.get("quiescent"),
            "env_intact": reply.get("env_intact"),
            "env_changed": reply.get("env_changed") or [],
            "poisoned": reply.get("poisoned"),
            **(
                {"traceback": "\n".join(prepare_diagnostics.activity_lines(reply["traceback"]))}
                if reply.get("reply") in _TRACED_COMMANDS and reply.get("traceback")
                else {}
            ),
            "metrics": reply.get("metrics"),
            "outputs": reply.get("outputs") or [],
            "result_ref": reply.get("result_ref"),
            "result_schema_digest": reply.get("result_schema_digest"),
            "oversize_result_bytes": reply.get("oversize_result_bytes"),
            "shortfall": reply.get("shortfall"),
            "residency": reply.get("residency") or {},
            "adjustments": reply.get("adjustments") or [],
            "observations": [],
            "observation_caps": {"dropped_for_frame": held, "minimal_reply": 1},
        }
    )


def _serialize_result(
    envelope: Any, spool: Path
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int]:
    """Serialize the typed result EXACTLY ONCE, before its types are lost.

    msgspec encodes the whole annotated result struct — nested structs, lists and all — in
    one pass, so nothing is stringified on the way out and a nested typed value survives as
    a typed value. Assets are the one thing that cannot go on the wire as themselves: the
    hook replaces each with its MANIFEST ROW (`asset_ref`, kind, media type — and digest and
    size once encoded; a frame the post phase still owes carries neither, cr-079), which is
    exactly what `Asset.row()` already publishes and carries no local path.

    Output identity comes from the result's own FIELD PATH (`image`, `pair.thumb`,
    `frames.0`) — a stable name the RecordOwner can grant against. It is never a list index
    into the grant: two outputs arriving in a different order must still land in their own
    destinations, and position cannot express that.

    Returns `(result_ref, outputs, oversize_bytes)`. The BYTES land in the attempt spool and
    only their digest and length ride the reply: the public door admits 4 MiB inline and one
    control frame admits 64 KiB, so a result between the two was an executor fault instead of
    a served result until the bytes moved to the spool the worker already brokers.
    `oversize_bytes` is non-zero when the canonical result exceeds the inline door, which
    cr-007 REFUSES rather than papering over with a receipt for bytes nobody wrote.
    """
    import msgspec

    from cozy_runtime.author._assets import Asset, Tree

    if envelope is None:
        return None, [], 0

    def hook(value: object) -> object:
        if isinstance(value, (Asset, Tree)):
            row = value.row()
            return {"asset_ref": row.pop("ref"), **row}
        raise TypeError(f"{type(value).__name__} has no wire spelling in a result")

    document = json.loads(msgspec.json.encode(envelope.result, enc_hook=hook))
    assert isinstance(document, dict)
    outputs = sorted(_assets_at(document, ""), key=lambda row: str(row["output_id"]))
    data = canonical.write(document)
    if len(data) > INLINE_RESULT_MAX:
        return None, [], len(data)
    (spool / RESULT_DOCUMENT).write_bytes(data)
    return (
        {"digest": "sha256:" + hashlib.sha256(data).hexdigest(), "length": len(data)},
        outputs,
        0,
    )


def _assets_at(node: Any, path: str) -> list[dict[str, Any]]:
    """Every asset node in the serialized result, with the field path that names it."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "asset_ref" in node:
            return [{"output_id": path, **node}]
        for key, value in node.items():
            found += _assets_at(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found += _assets_at(value, f"{path}.{index}" if path else str(index))
    return found


# --------------------------------------------------------------------------- main

#: What a follower rank answers. It has no worker: rank 0 is its whole control plane.
_FOLLOWER_COMMANDS = frozenset(
    {
        "hello",
        "start",
        "warm",
        "join",
        "load",
        "activate",
        "unload",
        "run",
        "attention",
        "vacate",
        "restore",
        "probe",
        "shutdown",
    }
)

#: The commands whose refusals carry the raising traceback to the worker.
_TRACED_COMMANDS = frozenset({"start", "load", "join"})

#: A follower's refusal codes rank 0 relays as-is; any other refusal is the group not
#: forming, named with the follower's own detail.
_GROUP_CODES = frozenset({"gpu_divergence", "context_parallel_unavailable", "group_broken"})


def _remember_distributions() -> None:
    """Diffusers and Transformers each map every package to its distribution at import by
    walking every installed file's RECORD (~0.3 s apiece in the H3 environment, run 1516). A
    sealed installation cannot change under a live executor, so the second walk reads the
    first; each caller still gets its own copy."""
    walk = functools.cache(importlib.metadata.packages_distributions)
    importlib.metadata.packages_distributions = lambda: {k: list(v) for k, v in walk().items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cozy-runtime device executor")
    parser.add_argument("--socket", default="")
    parser.add_argument("--root", required=True)
    # A FOLLOWER RANK (cr-068): the same program, told its rank, its world, the inherited
    # seam fd to rank 0 and rank 0's pid. It never has a worker socket of its own.
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--rank-fd", type=int, default=-1)
    parser.add_argument("--leader", type=int, default=0)
    args = parser.parse_args(argv)
    if args.rank > 0:
        if args.rank_fd < 0 or args.world <= args.rank or not args.leader:
            parser.error("a follower rank needs --rank-fd, --world > rank and --leader")
        if os.getppid() != args.leader:
            # pdeathsig was armed before exec and covers FUTURE deaths only; an already
            # orphaned follower never becomes a rank.
            return int(Exit.structural)
    elif not args.socket:
        parser.error("--socket is required for rank 0")

    _capture_seal()
    _remember_distributions()
    kernel_cache.configure(kernel_cache.from_sealed(_SEALED))
    # cr-042: the executor's ONE network fence, installed BEFORE the seam dials so the
    # AF_UNIX exemption is exercised on every boot rather than asserted. It refuses
    # Python-level egress from package and job code; it is not a kernel bound and
    # `sandbox.refuse_network` says exactly which adversary that leaves.
    sandbox.refuse_network(
        "the executor is handed verified files in its attempt spool and never an address "
        "(cr-012). Package code that dials is a bug, and this is where the bug stops",
        allow_unix_seam=True,
    )
    channel = (
        connect(args.socket)
        if args.rank == 0
        else Channel(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=args.rank_fd))
    )
    executor = Executor(
        channel,
        Path(args.root),
        fill_forensics_dir=_SEALED.get(child_env.FILL_FORENSICS_DIR, "").strip(),
        rank=args.rank,
        world=args.world,
    )

    while True:
        try:
            frame = channel.recv()
        except SeamError:
            return int(Exit.unavailable)
        if frame is None:
            return int(Exit.ok)
        name = str(frame.get("cmd", ""))
        if executor.group is not None and executor.group.broken and not executor.poisoned:
            # A FOLLOWER DIED BETWEEN COMMANDS (cr-068): the group is one executor, so this
            # process is poisoned before it answers anything, and the worker's next
            # command reads `poisoned_generation` and rebuilds the whole cgroup.
            executor.poisoned = f"group: {executor.group.broken}"[:200]
        try:
            if executor.rank > 0 and name not in _FOLLOWER_COMMANDS:
                reply = {
                    "ok": False,
                    "code": "follower_command_unsupported",
                    "detail": f"{_gpu(executor.rank)} answers {sorted(_FOLLOWER_COMMANDS)}, "
                    f"not {name!r}",
                }
            elif name == "run":
                reply = executor.run(frame)
            elif name not in executor_commands.NAMES:
                reply = {"ok": False, "code": "unknown_command", "detail": name}
            else:
                command = executor_commands.decode(frame)
                if isinstance(command, Shutdown):
                    if executor.group is not None:
                        executor.group.close()
                    channel.send({"reply": name, "ok": True})
                    return int(Exit.ok)
                reply = executor.handle(command)
        except msgspec.ValidationError as exc:
            reply = {"ok": False, "code": "command_malformed", "detail": str(exc)[:1024]}
        except Exception as exc:  # the executor reports its own faults; it never hides one
            outcome: Outcome = classify(exc)
            reply = {
                "ok": False,
                "code": outcome.code,
                "detail": f"{type(exc).__name__}: {exc}"[:1024],
                "terminal": outcome.terminal,
                "origin": outcome.origin,
            }
            if name in _TRACED_COMMANDS:
                reply["traceback"] = prepare_diagnostics.exception_trace(exc)
        reply["reply"] = name
        _send_reply(channel, reply)


if __name__ == "__main__":
    raise SystemExit(main())
