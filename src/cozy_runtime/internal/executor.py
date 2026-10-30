"""The long-lived device executor: Python, CUDA, loaded constructions, one GPU attempt.

Runtime imports the package's Torch stack and constructs its models in this process.
TensorD makes model data available, prepares shared CPU buffers through TensorFS and
coordinates machine budgets. Runtime allocates ordinary Torch tensors and lends their storage
through TensorFS's read interface. Each executor owns its allocations; TensorFS runs in both
processes as a library.

The executor lives for its (environment, lane) and holds its loaded constructions.
TensorD supervises its disposable lifetime and owns durable execution records; source and
output transaction durability remain with the native TensorFS store and execution owner.

Its whole surface is the seam (`internal/seam.py`), and its whole vocabulary is:

    hello     process facts: pid, parent, no_new_privs, oom_score_adj, the SEALED env
    start     spawn and join every rank, import torch, the runtime and the package, warm up
    load      construct + fill one construction under its key; idempotent per key
    activate  make one loaded construction resident, parking LRU constructions for room
    unload    drop one construction and free its device bytes
    prepare_request  resolve ONE request (decode, normalize, preflight) and retain it, so
              TensorD prices a plan against the record the handler will actually run
    invoke    run ONE PREPARED attempt through the author invocation kernel and report its facts
    probe     device/host memory facts, for TensorD's ledger reconciliation
    shutdown  cooperative exit

Calls run side by side, as tasks on one bounded pool (`serve`): every per-call command,
reply and frame names its `request_id`, and answers to the executor's requests come back by
`seq` (`seam.Mux`). Cancellation is call-keyed: `ctx.raise_if_cancelled` reads the mark a
`cancel` command left for this exact request. Every telemetry step, stage and progress call
and, in a group, every mirrored call is a safe point. TensorD supervises executor lifetime
and records each attempt's terminal outcome.

The env seal is enforced, not trusted: the sealed values are snapshotted BEFORE torch is
imported, re-imposed before Torch import, CUDA init and every reply, and any names package
code changed are reported in `env_changed` rather than silently ignored.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import contextvars
import ctypes
import dataclasses
import faulthandler
import functools
import gc
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import msgspec
import tensorfs
from msgspec import UNSET, UnsetType

from cozy_runtime import __version__ as RUNTIME_VERSION
from cozy_runtime import graphs, weights_store
from cozy_runtime.author import ModelRegistry, _split, describe
from cozy_runtime.author import _request_path as request_path
from cozy_runtime.author._activity import ThreadPoolExecutor
from cozy_runtime.author._app import Registration
from cozy_runtime.author._artifacts import ModelArtifact
from cozy_runtime.author._assets import Asset, GrantedInput, Tree
from cozy_runtime.author._attention import _close as close_producer_states
from cozy_runtime.author._calls import Nudges, _Broker, _CallType, _export
from cozy_runtime.author._context import CURRENT, AdapterRef, Device, place
from cozy_runtime.author._demand import shape_cell
from cozy_runtime.author._errors import (
    DEVICE_FAULT,
    DEVICE_OOM,
    MEMORY_EXHAUSTED,
    CapabilityError,
    ConformanceError,
    Fault,
    InvalidRequest,
    Outcome,
    RuntimeFailure,
    classify,
    ends,
    fault,
    unwind,
)
from cozy_runtime.author._executor_requests import (
    Answer,
    CheckpointItem,
    ChildEvents,
    Exchange,
    Handoff,
    Published,
    Request,
)
from cozy_runtime.author._executor_requests import encode as encode_request
from cozy_runtime.author._invoke import (
    Invocation,
    PreparedRequest,
    prepare,
    run_prepared,
)
from cozy_runtime.author._loader import Artifact, Config, ModelFitRefused
from cozy_runtime.author._model import (
    LANDED,
    SCOPES,
    WARNINGS,
    Model,
    WarningRow,
    component_use,
)
from cozy_runtime.author._observations import DEVICE, Attribution
from cozy_runtime.author._prepared_use import PreparedUse, prepared_use
from cozy_runtime.author._services import AdjustmentRow, ProgressFrame
from cozy_runtime.author._services import Attempt as AuthorAttempt
from cozy_runtime.author._weights import Weights as RawWeights
from cozy_runtime.author.kernels import _SETS as KERNEL_SETS
from cozy_runtime.author.kernels import Launch, Plan, TritonKernels
from cozy_runtime.derive import recipes as declared_recipes
from cozy_runtime.derive.operations import app as builtin_operations_app
from cozy_runtime.derive.operations import run_recipe
from cozy_runtime.internal import (
    accel,
    arch,
    attention,
    attention_sol,
    attention_ulysses,
    builtin_operations,
    call_types,
    canonical,
    child_env,
    effect_interfaces,
    execution_evidence,
    executor_commands,
    fp8_linear,
    int8_linear,
    interface_wheel,
    kernel_cache,
    kernel_compile,
    kernel_ready,
    model_config,
    package_interface,
    prebuilt,
    prepare_diagnostics,
    proctree,
    sandbox,
    schema,
    spawn,
    stage_memo,
    tolerant,
    tools,
    w4a8_linear,
    zygote,
)
from cozy_runtime.internal.accel import ResidencyError
from cozy_runtime.internal.config import inherited_environment, restore_seal, seal_snapshot
from cozy_runtime.internal.derive import Observations, serving_substrate
from cozy_runtime.internal.discovery import discover_installed
from cozy_runtime.internal.encoding.leaves import FORMATS, serving
from cozy_runtime.internal.encoding.selection import (
    launch_providers,
    measure_device,
    measure_runtime,
)
from cozy_runtime.internal.executor_commands import (
    Activate,
    Attention,
    Binding,
    CallInterface,
    Fork,
    Form,
    GroupHardware,
    Hello,
    Invoke,
    Join,
    Kernels,
    KnownCommand,
    Load,
    Map,
    ModelLoad,
    PrepareRequest,
    Probe,
    Release,
    Shutdown,
    Start,
    Supplied,
    TensorFSOptions,
    Unload,
    View,
)
from cozy_runtime.internal.executor_replies import GpuMemory, RankMemory
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.fill import (
    Checkpoint,
    FillRefusal,
    dtype_name,
    normalized,
    store_refusals,
    tensor_schema_of,
)
from cozy_runtime.internal.lora_composition import bind as bind_adapters
from cozy_runtime.internal.lora_contract import read as read_adapter_graph
from cozy_runtime.internal.model_values import read_values_into
from cozy_runtime.internal.parallel import copy_engine, cp, links, mirror, wire
from cozy_runtime.internal.parallel.group import (
    ChildProcess,
    RankGroup,
    RankSpec,
    abort_process_group,
    form_rank,
    ready_rank,
)
from cozy_runtime.internal.parallel.mirror import mirror_component
from cozy_runtime.internal.parallel.plan import (
    GpuDivergence,
    GroupPlan,
    GroupRefusal,
    rank_invariant_digest,
)
from cozy_runtime.internal.resolution import (
    ConstructionFacts,
    PlanRefusal,
    RefusedLeaves,
    Variant,
    resolve,
)
from cozy_runtime.internal.seam import (
    RESULT_DOCUMENT,
    Channel,
    Frame,
    Mux,
    SeamError,
    connect,
)

if sys.platform == "win32":
    from ctypes import wintypes

    from cozy_runtime.internal import winpipe
else:
    import resource

if TYPE_CHECKING:
    from cozy_runtime.internal.residency import Memory, QueuedPrefetch
    from cozy_runtime.internal.residency import Request as MemoryRequest

#: The native inline-result door: a small typed result rides
#: the terminal envelope and skips the file-store round trip. Larger than this REFUSES at
#: cr-007 — the blob-receipt branch needs a real write-and-receipt transaction, and minting
#: a receipt for bytes nobody wrote is the fabrication class this runtime exists to prevent.
INLINE_RESULT_MAX = 4 * 1024 * 1024

#: Outcomes that are capacity, not broken state: they fail the attempt and keep the executor.
CAPACITY_CODES = frozenset({DEVICE_OOM, MEMORY_EXHAUSTED})

#: Calls this executor runs at once: its pool's workers. More wait their turn in the pool.
CALLS_IN_FLIGHT = 16

#: How many PREPARED requests this epoch holds at once: anything past the newest few beyond
#: the calls in flight is a request the worker prepared and then never dispatched.
PREPARED_HELD = 2 * CALLS_IN_FLIGHT

#: The environment as it was when this process started, restricted to the sealed names.
#: Captured before any import that could initialize CUDA.
_SEALED: dict[str, str] = {}

#: macOS's ceiling on a process's soft descriptor limit (`<sys/syslimits.h>`).
_OPEN_MAX = 10240


#: Each callable an attempt may call, by (module, export).
type CallBindings = dict[tuple[str, str], _CallType]


def _capture_seal() -> None:
    """Snapshot the seal BEFORE anything that could initialize CUDA."""
    _SEALED.update(seal_snapshot())


def _env_changes() -> tuple[str, ...]:
    """Names whose current values differ from the pre-Torch executor seal.

    Values never cross the seam. The names are enough to diagnose which reviewed
    imposition moved without leaking paths or future secret-bearing values into a terminal.
    """
    current = seal_snapshot()
    return tuple(
        name
        for name in sorted(_SEALED.keys() | current.keys())
        if current.get(name) != _SEALED.get(name)
    )


def _attention_pin(kernel: str, bits: int) -> str:
    """A request's attention override: its pin, else level 16 at 16 bits (C4)."""
    return kernel or (attention.EXACT if bits == 16 else "")


def _sealed_entries() -> tuple[str, ...]:
    """The seal's `CUDA_VISIBLE_DEVICES` entries: one per device this process may use."""
    return tuple(
        entry.strip()
        for entry in _SEALED.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if entry.strip()
    )


def _gpu_numbers(degree: int) -> list[int]:
    """The group's GPUs as nvidia-smi numbers them, from the seal's entries."""
    entries = _sealed_entries()[:degree]
    return [execution_evidence.identify(entry)["gpu"] for entry in entries]


def _gpu(rank: int) -> str:
    """How people name the GPU group process `rank` drives: `GPU 3`, nvidia-smi's number."""
    return execution_evidence.gpu_name(_sealed_entries(), rank)


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
    """The seal holds for every reply once re-imposed; names package code changed are reported."""
    changed = _restore_seal()
    return {"env_intact": not _env_changes(), "env_changed": list(changed)}


#: The call the current thread serves ("" outside every call): its requests and its compile
#: waits' progress name it.
_CALL: contextvars.ContextVar[str] = contextvars.ContextVar("cozy_call", default="")


@contextlib.contextmanager
def _held(
    mux: Mux, request_id: str, waits: Attribution, waiter: str, wake: Callable[[], None]
) -> Iterator[Callable[[], bool]]:
    """A compile wait of this request's preparation (`kernel_compile.held`): a stage of its
    record, woken at once when a cancel names it."""
    started = time.perf_counter()
    try:
        with mux.watch(request_id, wake):
            yield functools.partial(mux.cancelled, request_id)
    finally:
        waits.stage(f"waiting for {waiter} compile", (time.perf_counter() - started) * 1000)


def _supplied(bearers: weights_store.Bearers, frame: Frame) -> None:
    """A bearer as the seam reads it, before any later frame; a malformed one is its command's
    to refuse."""
    try:
        supplied = msgspec.convert(frame, Supplied)
    except msgspec.ValidationError:
        return
    options = supplied.tensorfs
    if options is not None and options.token:
        bearers.note(supplied.actor, options.endpoint, options.token, options.expires_unix_ms)


def _connection(
    options: TensorFSOptions | None, actor: str, bearers: weights_store.Bearers
) -> weights_store.Connection | None:
    """The request's endpoint and, when it was supplied a token, its signer's newest bearer for
    that endpoint; its private CA, if any, joins TensorFS's trusted roots for this process
    (`configure` adds a root, so setting it again is harmless)."""
    if options is None:
        return None
    if options.ca_der is not None:
        try:
            ca_der = base64.b64decode(options.ca_der, validate=True)
        except ValueError as exc:
            raise InvalidRequest("TensorFS CA is not base64 DER", code="tensorfs_config") from exc
        tensorfs.configure(ca_der=ca_der)
    if options.token is None:
        return weights_store.Connection(options.endpoint)
    bearers.offer(actor, options.endpoint, options.token, options.expires_unix_ms)
    bearer = weights_store.Bearer(bearers, actor, options.endpoint, options.token)
    return weights_store.Connection(options.endpoint, bearer)


def _replay_code(command: Invoke, request: str) -> str:
    """The code and request a weights output derives from: the attempt's operation and resolved
    request (`request`, its digest), with its file inputs and precision; without an operation,
    its run, so only its own attempts resume it."""
    if not command.operation:
        return command.request_id
    files = sorted([name, row.digest, row.length] for name, row in command.inputs.items())
    asked = {"request": request, "files": files, "bits": command.bits}
    return f"{command.operation}\0{hashlib.sha256(canonical.write(asked)).hexdigest()}"


def _show_checkpoint(
    exchange: Exchange, slot: str, repo_id: str, revision: str
) -> None:
    """A weights output's checkpoint as the run's checkpoint item (`CheckpointItem`)."""
    item = CheckpointItem(output=slot, repo_id=repo_id, revision=revision)
    answer = exchange(item, Answer)
    if not answer.ok:
        raise RuntimeFailure(
            answer.detail or "the machine refused the checkpoint item",
            code=answer.code or "runtime_failure",
        )


@dataclasses.dataclass(slots=True)
class _Slot:
    first: bool
    leaving: bool = False
    last: bool = False


class _InFlight:
    """The attempts in flight, side by side. The first in readies the plane and the last out
    drains it, each with no other attempt running. One that changes what every attempt runs on
    (a pinned kernel, a GPU group's call) runs alone, and later ones wait for it, told which
    runs they wait on."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        #: the runs in flight, by request id
        self._running: list[str] = []
        self._leaving = 0
        self._alone = False
        self._queued = 0

    @property
    def count(self) -> int:
        return len(self._running)

    @contextlib.contextmanager
    def attempt(self, alone: bool, request_id: str) -> Iterator[_Slot]:
        with self._changed:
            self._queued += alone
            request_path.wait(
                self._changed,
                lambda: not self._alone and not (self.count if alone else self._queued),
                lambda: ", ".join(f"run {running}" for running in self._running),
            )
            self._queued -= alone
            self._running.append(request_id)
            slot = _Slot(first=self.count == 1)
            self._alone = alone or slot.first
        try:
            yield slot
        finally:
            with self._changed:
                self._running.remove(request_id)
                self._leaving -= slot.leaving
                self._alone = self._alone and self.count > 0
                self._changed.notify_all()

    def ready(self) -> None:
        """The first in readied the plane: others may join it."""
        with self._changed:
            self._alone = False
            self._changed.notify_all()

    def last(self, slot: _Slot) -> bool:
        """This attempt is leaving. Whether every attempt in flight is, so that this one drains
        the plane, alone."""
        with self._changed:
            slot.leaving = True
            self._leaving += 1
            slot.last = self._leaving == self.count
            self._alone = self._alone or slot.last
            return slot.last


def _distribution_version(name: str) -> str:
    """An installed distribution's version in this environment, or ""."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


if sys.platform == "win32":

    class _MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            *(
                (name, ctypes.c_size_t)
                for name in (
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ),
        ]

    def _rss() -> int:
        """The peak working set: what `ru_maxrss` is elsewhere."""
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        counters = _MemoryCounters(cb=ctypes.sizeof(_MemoryCounters))
        if not kernel32.K32GetProcessMemoryInfo(
            wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)

    def _group() -> int:
        """A Windows executor is its own group (its Job Object)."""
        return os.getpid()

else:

    def _rss() -> int:
        # Kilobytes on Linux, bytes on macOS.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )

    def _group() -> int:
        return os.getpgrp()


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


def adjustment_row(row: AdjustmentRow) -> dict[str, str]:
    """The reply's adjustment row: kind and source stay fields; values cross as text."""
    return {
        "field": row.field,
        "kind": row.kind,
        "requested": "" if row.requested is None else str(row.requested),
        "applied": "" if row.applied is None else str(row.applied),
        "reason": row.reason[:256],
        "source": row.source,
    }


def _granted_inputs(command: Invoke) -> dict[str, GrantedInput]:
    """Verified input paths in the attempt's spool."""
    return {input_id: row.granted(input_id) for input_id, row in command.inputs.items()}


def _trees(command: Invoke) -> dict[str, tuple[Path, str]]:
    return {ref: (Path(root), digest) for ref, (root, digest) in command.trees.items()}


def _serving(module: str, export: str, surface: Any) -> _CallType:
    """A callable bound from its own code, called as every function is: `{payload, models}`.
    Its direct caller passes the model slots beside the request fields (`_CallType.models`)."""
    slots = frozenset(binding.param for binding in surface.model_bindings)
    request, result = surface.payload_type, surface.result_type
    if not all(isinstance(t, type) and issubclass(t, msgspec.Struct) for t in (request, result)):
        raise CapabilityError("callable types are not structured", code="child_interface")
    fields = msgspec.structs.fields(request)
    if any(field.name in slots for field in fields):
        request = msgspec.defstruct(
            request.__name__,
            # An invocable's defaults are immutable scalars (`_wire_default`), never factories.
            [(f.name, f.type, f.default) for f in fields if f.name not in slots],
            module=module,
            kw_only=True,
            frozen=True,
            forbid_unknown_fields=True,
        )
    artifact: Any = ModelArtifact | None
    models = msgspec.defstruct(
        f"{export}Models",
        [(slot, artifact, None) for slot in sorted(slots)],
        module=module,
        kw_only=True,
        frozen=True,
    )
    call = msgspec.defstruct(
        f"{export}Call", [("payload", request), ("models", models)], module=module, frozen=True
    )
    return _CallType("", module, export, call, result, models=slots)


# --------------------------------------------------------------------------- state


@dataclasses.dataclass(slots=True)
class _RankRequest:
    """A follower's part of rank 0's attempt: its open request and the record it lands in."""

    construction: str
    id: str
    context: contextlib.AbstractContextManager[list[GpuMemory]]
    rows: list[GpuMemory]


@dataclasses.dataclass(frozen=True, slots=True)
class _Install:
    """A plan's install waiting for its kernel set's build (`Plan.when_ready`): the set, its build
    on this GPU, the package's install, and its construction's plan reports."""

    name: str
    ready: Callable[[], Launch | None]
    install: Callable[[Launch], None]
    reports: dict[str, Mapping[str, object]]


class _CompositeResidency:
    """The placements of a many-model construction, one per model slot, as one."""

    def __init__(self, rows: Mapping[str, Any]) -> None:
        self.rows = dict(rows)

    def open_attempt(self) -> None:
        for row in self.rows.values():
            row.open_attempt()

    def plan(
        self, steps: Sequence[str | Collection[str]], request: MemoryRequest | None = None
    ) -> None:
        for row in self.rows.values():
            row.plan(steps, request=request)

    def release(self) -> None:
        for row in self.rows.values():
            row.release()

    @property
    def notes(self) -> dict[str, str]:
        return {f"{name}.{k}": v for name, row in self.rows.items() for k, v in row.notes.items()}

    def baked(self) -> int:
        return sum(row.baked() for row in self.rows.values())

    def document(self) -> dict[str, dict[str, object]]:
        """Each slot's placement's document as one: its notes and graphs keyed by slot."""
        docs = {name: row.document() for name, row in self.rows.items()}
        return {
            part: {
                f"{name}.{key}": value
                for name, doc in docs.items()
                for key, value in doc[part].items()
            }
            for part in ("notes", "graphs")
        }


def _construction_record(facts: Mapping[str, Any]) -> dict[str, Any]:
    """A prepare's run-record entry: when it ended, how long it took and its legs."""
    return {
        "prepared_unix_ms": int(facts.get("prepared_unix_ms", 0)),
        "prepare_ms": float(facts.get("prepare_ms", 0)),
        "fill_ms": float(facts.get("fill_ms", 0)),
        "warm_ms": float(facts.get("warm_ms", 0)),
        "legs_ms": dict(facts.get("stages") or ()),
    }


class _OnDemand:
    """An `on_demand=` model argument: built, and its weights fetched, when the function first
    reads it. Until then the runtime's probes see no model, and a request that never reads it
    builds and fetches nothing of it."""

    def __init__(self, build: Callable[[], Any]) -> None:
        self._build = build
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def _cozy_residency(self) -> Any:
        return getattr(self._model, "_cozy_residency", None)

    @property
    def _cozy_active(self) -> Any:
        return getattr(self._model, "_cozy_active", None)

    def __getattr__(self, name: str) -> Any:
        with self._lock:
            if self._model is None:
                self._model = self._build()
        return getattr(self._model, name)


#: The runtime modules a construction imports, named ONCE so `start` can warm
#: exactly the set `_prepare_first` re-imports (cr-104). Warming a name this tuple omits
#: is a missed saving and never a wrong answer: the prepare imports what it needs either
#: way, and a warm `sys.modules` is the only difference.
_CONSTRUCTION_MODULES = (
    "cozy_runtime.author",
    "cozy_runtime.author._loader",
    "cozy_runtime.author._model",
    "cozy_runtime.internal.derive",
    "cozy_runtime.internal.encoding",
    "cozy_runtime.internal.fill",
    "cozy_runtime.internal.weights",
    "cozy_runtime.internal.resolution",
    "cozy_runtime.internal.triton_kernels",
)


class _PlanRefusal(RuntimeError):
    """A model's execution plan refused its construction."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _dense(kernels: Iterable[str]) -> str:
    """The dense reference the Sol kernels among `kernels` recompute rows with."""
    sparse = attention.sparse_names(kernels)
    return ", ".join(dict.fromkeys(attention.dense_identity(name) for name in sparse))


def _attention_divergence(mine: executor_commands.Choice, reply: Mapping[str, object]) -> str:
    """How a follower's answer to rank 0's selection differs from it; "" when it serves it."""
    if not reply.get("ok"):
        return f"{reply.get('code', '')}: {reply.get('detail', '')}"
    try:
        theirs = msgspec.convert(reply.get("choice"), executor_commands.Choice)
    except msgspec.ValidationError as exc:
        return f"an unreadable selection: {exc}"
    return attention.difference(mine, theirs)


def _segment(stage: str, step: int, held: attention.Applied, at_us: int = 0) -> dict[str, Any]:
    """A selection a run served from step `step` of `stage` on (`execution_evidence`)."""
    return {
        "stage": stage,
        "step": step,
        "at_us": at_us or execution_evidence.now_us(),
        "observed": execution_evidence.observed(held.hosts),
        "dense": _dense(held.totals()),
    }


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
        #: HOST ONLY: each signer's newest bearer for each endpoint, which the seam keeps
        #: current as it reads (`Mux`) and every TensorFS request asks (`weights_store.Bearer`)
        self.bearers: weights_store.Bearers = (
            host.bearers if host is not None else weights_store.Bearers()
        )
        #: HOST ONLY: the seam every thread reads through, the attempts in flight and the lock
        #: their fetch channel opens and closes under
        self.mux: Mux = (
            host.mux
            if host is not None
            else Mux(channel, functools.partial(_supplied, self.bearers), self._discard)
        )
        self.in_flight: _InFlight = host.in_flight if host is not None else _InFlight()
        #: the construction key this generation was loaded under; "" on the host
        self.construction = construction
        #: HOST ONLY: every loaded construction and the active one. One memory decides where
        #: all their weights live (`internal.residency`), and the execution layer runs
        #: torch under it (`internal.execution`, imported with the device).
        self.constructions: dict[str, Executor] = {}
        self.memory: Memory | None = None
        #: HOST ONLY: Start's development GPU memory limit, the memory's budget (the card's size)
        self.gpu_memory_limit: int | None = None
        self.execution: ModuleType | None = None
        #: HOST ONLY: each attempt's open request, by request id (`ctx.plan` replaces its plan)
        self.requests: dict[str, MemoryRequest] = {}
        #: HOST ONLY: each open or just-ended request's cut rows (`measurements.execution.cuts`)
        self.cuts: dict[str, object] = {}
        #: each request's `author.denoise_loop` rows (`internal.loop.ROWS`), once memory opens
        self.loops: dict[str, list[dict[str, object]]] = {}
        #: HOST ONLY: TensorD's lease namespace from `Start` (None: no lease, its rows alone);
        #: rank 0 forwards it in its followers' Start
        #: The machine's sealed TensorFS cache directory.
        self.store: Path | None = None
        #: HOST ONLY: the constructed slots built on first use (`on_demand=`), by the same path
        self.on_demand: frozenset[str] = frozenset()
        #: each on-demand model built so far, by parameter: its prepare facts (the run record's)
        self.built_on_demand: dict[str, dict[str, Any]] = {}
        self.active = ""
        self.started = False
        #: torch, the Runtime and the package are imported (`Start.import_only`); no device yet
        self.imported = False
        #: children forked from this import-only process and not reaped yet (`fork/1`)
        self.forked: set[int] = set()
        self.package_warmed = False
        #: rank 0 only: each follower's own start ledger, in rank order
        self.follower_stages: list[list[Any]] = []
        #: this generation's load reply
        self.load_reply: dict[str, Any] = {}
        #: follower only: the attempt spool and allocator bytes at its first mirrored call
        self._run_entry: Path | None = None
        #: follower only: every kernel its processors held across that attempt's calls
        self._run_held = attention.Applied({}, ())
        self._rank_request: _RankRequest | None = None
        #: HOST ONLY: this process's start, for the execution record (wall ms and its legs)
        self.boot: dict[str, Any] = {}
        #: HOST ONLY: this rank's pid, host ordinal, GPU uuid and arch, read once after device init
        self._identity: dict[str, Any] | None = None
        #: THIS process's rank in its generation's group and the group's size (cr-068). A
        #: plain executor is rank 0 of a world of 1 and forms nothing; rank 0 of a wider
        #: world spawns and drives the followers; a follower's channel is rank 0's seam.
        self.rank = int(rank)
        self.world = int(world)
        if self.world > 1:
            kernel_ready.group(self.rank, self.world)
        self.group: RankGroup | None = None
        #: the kernels that began serving as this attempt was admitted (`_grant`)
        self._granted: list[str] = []
        self.pg: Any = None
        #: between `join` and `form`: this follower's rank spec and the store it reached
        self._forming: tuple[RankSpec, Any] | None = None
        #: HOST ONLY: the group's handle as every installed component reads it (`CpComms`)
        self.comms: cp.CpComms | None = None
        #: rank 0 only: the measured links and route this group formed over
        self.transport: links.Transport | None = None
        #: a follower's watcher aborted its collectives; cleared when the group forms again
        self._interrupted = threading.Event()
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
        # Accepted with this member (or its deferred container), never a per-alias cache.
        self._prepared_use: PreparedUse | None = None
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
        #: HOST ONLY: what its attempts may call (`Start.call_interfaces`), and their bindings
        self.call_interfaces: tuple[CallInterface, ...] = ()
        self._bound_calls: tuple[tuple[CallInterface, ...], CallBindings] | None = None
        self.ready = False
        self.poisoned = ""
        #: its followers were sent the load: from here a refusal leaves them mid-protocol
        self.commanded = False
        #: by model binding path: the execution plan it installed (its fused glue's identity),
        #: "" when eager
        self.planned: dict[str, str] = {}
        #: per binding path, what its execution plan reported (`Plan.report`)
        self.plan_reports: dict[str, dict[str, Mapping[str, object]]] = {}
        #: by request: an attempt's handler outcome, the stage it last reported and its record,
        #: the reason an attempt whose tail raised still answers with (`unsettled`)
        self.handled: dict[str, tuple[Outcome, str, AuthorAttempt]] = {}
        self.constructed_model_digest = ""
        #: What the runtime pinned on this generation's attention sites (cr-124). One BOOT
        #: line, not a per-attempt record: the choice is a function of device and image.
        self.attention: attention.Applied | None = None
        self.attention_defaults = attention.Snapshot(())
        #: this construction's selection, made again between requests (`_upgrade_attention`),
        #: passing over what `passed` names
        self._reselect: Callable[..., attention.Applied] | None = None
        #: kernel installs waiting for their build on every GPU (`Plan.when_ready`), by set name
        self._installs: list[_Install] = []
        #: Device facts for this prepared generation, reused for per-request pin checks.
        self.device_facts: Any = None
        #: a follower's hold of rank 0's attention selection for the current attempt
        self._attention_hold = contextlib.ExitStack()
        self._producer_hold: tuple[str, dict[str, object]] | None = None
        self.facts: dict[str, Any] = {}
        #: what this generation's plan serves decoded because the device refused its leaf
        self.refused_leaves: tuple[RefusedLeaves, ...] = ()
        self.schema_digests: dict[str, str] = {}
        #: request_id -> the record `prepare_request` resolved, run by `invoke`
        self.prepared: dict[str, PreparedRequest] = {}
        # A cancel may arrive on the seam reader while preflight is still running.
        self._prepared_lock = threading.Lock()
        self.attempts = 0

    # ------------------------------------------------------------------ prepare

    def handle(self, command: KnownCommand) -> dict[str, Any]:
        """Answer one decoded worker command (`shutdown` ends the loop before this)."""
        match command:
            case Hello():
                return self.hello()
            case Join():
                return self.join(command)
            case Form():
                return self.form(command)
            case Attention():
                return self.select_attention(command)
            case Kernels():
                return self.kernels(command)
            case Fork():
                return self.fork(command)
            case Start():
                return self.start(command)
            case Load():
                return self.load(command)
            case Activate():
                return self.activate(command)
            case Unload():
                return self.unload(command)
            case PrepareRequest():
                return self.serve_request(command)
            case Invoke():
                try:
                    return self._settled(self.serve_invoke(command))
                except Exception as exc:
                    key = command.construction or self.active
                    target = self.constructions.get(key, self)
                    return self.unsettled(target, exc, command.request_id)
            case Probe():
                if command.collect:
                    gc.collect()
                return self.probe()
            case Map():
                return self.map(command)
            case executor_commands.RequestMemory():
                reply = msgspec.to_builtins(self.request_memory(command))
                assert isinstance(reply, dict)
                return reply
            case Shutdown():
                raise AssertionError("shutdown ends the command loop")

    def hello(self) -> dict[str, Any]:
        """What TensorD (`Hello`) and rank 0 (`group._Hello`) read: who this process is, its
        versions, its memory protocol and its seal."""
        return {
            "ok": True,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "pgid": _group(),
            "runtime_version": RUNTIME_VERSION,
            "tensorfs_version": _distribution_version("tensorfs"),
            "memory_protocol": 0,  # all weight and activation storage is process-local
            "memory_backend": "rust-local" if self.device_kind == "cuda" else "torch",
            # Correlated PrepareRequest replies and cancellation-safe publication.
            "request_lookahead": True,
            "sealed": dict(_SEALED),
            "rank": self.rank,
            "world": self.world,
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
        try:
            accel.initialize(torch, self.device_kind)
            if self.world > 1 and self.device_kind == "cuda":
                torch.cuda.set_device(self.rank)
                kernel_ready.DEVICE[0] = functools.partial(torch.cuda.device, self.rank)
            self._open_memory(torch)
        except ResidencyError as exc:
            return {"ok": False, "code": "weight_residency_unavailable", "detail": str(exc)}
        accel.first_launches(torch, self.device_kind)
        if self.device_kind == "cuda" and self.world == 1:
            # The prebuilt 8-bit op opens here, before any request: a component first built
            # mid-request (Klein's on-demand upsampler under a 16-bit main model) would ask for it
            # only then, and serve its whole stage on a fallback while it lands.
            int8_linear.native(torch, torch.cuda.current_device())
        self._device_ready = True
        place(self._device())
        return None

    def _open_memory(self, torch: Any) -> None:
        """This executor's local residency and its pure-site execution helpers."""
        from cozy_runtime.internal import execution, loop
        from cozy_runtime.internal.residency import Memory

        host = self.host
        if host.memory is not None:
            return
        device = self._rank_device(torch)
        host.memory = Memory.open(device, budget_bytes=host.gpu_memory_limit)
        host.execution = execution
        if device.type == "cuda":  # steps report the device's time beside the host's
            DEVICE[:] = [execution.Steps()]
        _split.install(execution)
        attention_ulysses.install(execution)
        loop.install()
        host.loops = loop.ROWS

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
        segments: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """The attempt's execution record: degree, every GPU, and the setup it ran on."""
        gpus = [
            execution_evidence.record(
                self._rank_identity() if self.torch is not None else {"pid": os.getpid()},
                start_us,
                end_us,
                execution_evidence.attention(
                    pin,
                    execution_evidence.observed(choices.hosts) if choices else "",
                    impl,
                    attention.evidence(choices.totals() if choices else ()),
                    segments,
                    choices.fallbacks if choices else (),
                ),
            )
        ]
        if self.group is not None:
            numbers = _gpu_numbers(self.world)
            for follower in self.group.followers:
                gpu = numbers[follower.rank] if follower.rank < len(numbers) else -1
                row = self.group.rank_records.get(follower.rank) or execution_evidence.record(
                    {"pid": follower.pid, "gpu": gpu}, 0, 0
                )
                gpus.append({**row, "attention": {**row["attention"], "requested": pin}})
        facts = self.facts
        transport = self.host.transport
        return {
            "degree": self.world,
            "gpus": gpus,
            **({"transport": transport.document()} if transport is not None else {}),
            # per spread method: the calls each rank ran, and why a rank declined
            "spread": dict(self.group.spread_log) if self.group is not None else {},
            "executor": dict(self.host.boot),
            "plan": {k: v for m in self.model_executors or [self] for k, v in m.planned.items()},
            "plan_reports": {
                k: dict(v)
                for m in self.model_executors or [self]
                for k, v in m.plan_reports.items()
            },
            "construction": _construction_record(facts),
            # Each on-demand model's build, made inside the stage that first read it.
            "on_demand": {p: _construction_record(f) for p, f in self.built_on_demand.items()},
        }

    def _rank_device(self, torch: Any) -> Any:
        if self.device_kind == "cuda" and accel.present(torch, self.device_kind):
            return torch.device("cuda", self.rank if self.world > 1 else 0)
        if self.device_kind == "mps" and accel.present(torch, self.device_kind):
            return torch.device("mps")
        return torch.device("cpu")

    def join(self, command: Join) -> dict[str, Any]:
        """A FOLLOWER's first half of the group: rank 0's store reached and its device current.
        Its answer is the readiness every GPU gives before any enters group init (`form`)."""
        if self.rank == 0:
            return {"ok": False, "code": "not_a_follower", "detail": f"{_gpu(0)} forms the group"}
        self._forming = None
        if command.abort:
            watcher = socket.socket(fileno=self.channel.recv_descriptor())
            threading.Thread(
                target=self._watch_abort, args=(watcher,), name="group-abort", daemon=True
            ).start()
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        _restore_seal()
        import torch

        self.torch = torch
        if command.again:
            # Rank 0 interrupted before it asked: the watcher has aborted the old group.
            self._interrupted.wait()
            self._interrupted.clear()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        spec = RankSpec(
            rank=self.rank,
            world=self.world,
            port=command.port,
            backend=command.backend or "nccl",
            nccl=dict(command.nccl),
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
            self._forming = (spec, ready_rank(torch, spec, _SEALED))
        except Exception as exc:
            return self._unformed(exc, "join")
        return {"ok": True, "rank": self.rank, "world": self.world, "backend": spec.backend}

    def form(self, command: Form) -> dict[str, Any]:
        """A FOLLOWER's second half: every GPU said it is ready, so it enters group init."""
        forming, self._forming = self._forming, None
        if forming is None:
            return {
                "ok": False,
                "code": "group_unformed",
                "detail": f"{_gpu(self.rank)} was not joined",
            }
        spec, store = forming
        try:
            self.pg = form_rank(self.torch, spec, store)
        except Exception as exc:
            return self._unformed(exc, "form")
        if self.comms is not None:
            self.comms.pg = self.pg
        reply = {"ok": True, "rank": self.rank, "world": self.world, "backend": spec.backend}
        if command.calibrate:
            exchange, reply["calibration"] = copy_engine.calibrate(
                self.torch, self.pg, self.rank, self.world
            )
            copy_engine.install(exchange, self.pg)
        return reply

    def _unformed(self, exc: Exception, step: str) -> dict[str, Any]:
        self.poisoned = f"{step} failed: {type(exc).__name__}"
        detail = f"{_gpu(self.rank)} could not {step} the group: {type(exc).__name__}: {exc}"
        return {
            "ok": False,
            "code": "group_unformed",
            "detail": detail[:900],
            "traceback": prepare_diagnostics.exception_trace(exc),
        }

    def _calibrated(self, pg: Any) -> None:
        """Rank 0's half of the formation's calibration: NCCL against the copy engines."""
        assert self.transport is not None
        exchange, self.transport.calibration = copy_engine.calibrate(self.torch, pg, 0, self.world)
        copy_engine.install(exchange, pg)
        self.transport.route = "copy-engine" if exchange is not None else "nccl"

    def _watch_abort(self, watcher: socket.socket) -> None:
        """A follower's collectives end when rank 0 says so (`RankGroup.interrupt`): its main
        thread may be inside one, where no command reaches it."""
        with watcher:
            try:
                while watcher.recv(1):
                    abort_process_group()
                    self._interrupted.set()
            finally:
                # Ended (rank 0 closed it, or the abort raised): a join waiting on it goes on.
                self._interrupted.set()

    def _group_reform(self) -> None:
        """Rank 0's half of `mirror.mirror_component`: after a GPU ran short of memory in a
        call, the group forms again for the next attempt (§5.5)."""
        assert self.group is not None and self.torch is not None
        transport = self.host.transport
        recalibrate = transport is not None and transport.links.copy_engines
        self.host.pg = self.group.form(
            self.torch, _SEALED, again=True, formed=self.host._calibrated if recalibrate else None
        )
        if self.host.comms is not None:
            self.host.comms.pg = self.host.pg

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
        # A spread call rank 0 waits on over the group: a header always answers it.
        answered = False
        try:
            scoped = {scope.model: scope for scope in wire.RunCommand.read(command).scopes}
            spool = Path(command["spool"])
            held = attention.observed(
                {f"{m}/{c}": root for (m, c), (_, root) in self.group_components.items()}
            )
            if self._run_entry != spool:
                self._run_entry = spool
                self._run_held = held
            # A run takes rank 0's kernel at a step boundary (`_upgrade_attention`).
            self._run_held = self._run_held.joined(held)
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
                        residency.open_attempt()
                        self._group_attempts[key] = spool
                    if residency is not None:
                        choices = scoped[key].lora
                        if set(choices) != set(components):
                            raise GroupRefusal(
                                "mirrored LoRA choices do not cover the scope's components"
                            )
                        stack.enter_context(residency.mirrored_lora(choices))
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
                states = None
                if self._producer_hold is not None:
                    construction, held_store = self._producer_hold
                    if name[0].startswith(construction + "#"):
                        states = held_store
                sol_counts = stack.enter_context(attention_sol.observing(held=states))
                served = stack.enter_context(attention_ulysses.observing())
                args, kwargs = wire.run_call(
                    command,
                    wire.Broadcast(references).receive(
                        command, self.pg, self._rank_device(self.torch)
                    )
                    if command.get("broadcast")
                    else wire.TensorSpool(
                        spool / str(command.get("payloads", "ranks")), references
                    ),
                    device=self._rank_device(self.torch),
                )
                if spread:
                    # A pure method call whose one tensor result goes straight to rank 0.
                    output = getattr(component, str(command["method"]))(*args, **kwargs)
                    head = wire.p2p_head(output)
                    # From the header on, rank 0 reads this answer; before it, `send_nothing`.
                    answered = True
                    wire.send_tensor(output, head, 0, self.pg)
                elif hosted:
                    call = component.get_submodule(str(command.get("module", "")))
                    result = wire.marshal(
                        call(*args, **kwargs),
                        wire.TensorSpool(spool / str(command.get("results", "hosted"))),
                    )
                else:
                    component(*args, **kwargs)
                accel.synchronize(self.torch, self.device_kind)
            self._attempt_spool = spool
            ranked = self.host._rank_request
            working = accel.allocator_peak(self.host.memory, ranked.id) if ranked is not None else 0
        except Exception as exc:
            ending = mirror.ending(exc)
            # where it was raised, innermost first: a GPU's own traceback stays on that GPU
            at = " < ".join(
                f"{Path(f.filename).name}:{f.lineno} {f.name}"
                for f in reversed(traceback.extract_tb(exc.__traceback__)[-3:])
            )
            if spread and not answered:
                declined = self._declines(exc, ending)
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
            if ending is not None and not spread:
                # Short of memory, or rank 0 ended this collective: never this process's
                # poison. A mirrored call's GPU waits for the group to form again.
                return {
                    "ok": False,
                    "code": ending,
                    "detail": f"{type(exc).__name__}: {str(exc)[:640]} (at {at})"[:900],
                    "rank": self.rank,
                    **({} if hosted else {"rerun": True}),
                }
            outcome = classify(exc)
            self.poisoned = f"mirrored {name} failed: {outcome.code}"
            return {
                "ok": False,
                "code": outcome.code,
                "detail": f"{type(exc).__name__}: {str(exc)[:640]} (at {at})"[:900],
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
                self._rank_identity(),
                started_us,
                execution_evidence.now_us(),
                execution_evidence.attention(
                    observed=execution_evidence.observed(self._run_held.hosts),
                    impl=attention.implementations(served),
                    kernels=attention.evidence(self._run_held.totals()),
                ),
            ),
        }

    def _declines(self, exc: Exception, ending: str | None) -> str | None:
        """Why a spread call is declined rather than failed: this GPU ran short of memory in
        it (`mirror.ending`)."""
        if ending == MEMORY_EXHAUSTED:
            accel.release_cached(self.torch, self.device_kind)
            return f"{getattr(exc, 'code', DEVICE_OOM)}: {exc}"[:300]
        return None

    def _group_refusal(self, exc: GroupRefusal, *, closing: bool = True) -> dict[str, Any]:
        """A group that could not form or hold is THIS process's poison: the worker
        rebuilds the whole cgroup, never a partial group."""
        self.poisoned = f"group: {exc.code}"
        if closing and self.group is not None:
            self.group.close()
        return {"ok": False, "code": exc.code, "detail": exc.message[:900]}

    def _install_group(
        self, torch: Any, model: Any, homes: Mapping[str, int]
    ) -> dict[str, Any] | None:
        """Every rank shards its replica, takes its side of the mirror, and keeps the placeable
        components TensorD homed on its GPU (`Binding.homes`, decided by varena)."""
        if self.world <= 1:
            return None
        owner = self.host
        roots = self.backend.roots
        for component, root in roots.items():
            owner.group_components[(self._group_model_key, component)] = (model, root)
        candidates = cp.sharding_candidates(model)
        # An explicitly group-compatible auxiliary model can be replicated beside a
        # sharded model. The complete multi-model prepare still requires a sharded root.
        installed: tuple[str, ...] = ()
        if candidates or self._group_parent is None:
            if owner.comms is None:
                owner.comms = cp.CpComms(owner.pg, self.rank, self._rank_device(torch))
            try:
                installed = cp.install_context_parallel(model, degree=self.world, comms=owner.comms)
            except GroupRefusal as exc:
                return self._group_refusal(exc)
        record = model._cozy_record
        skipped = set(record.skipped) if record is not None else set()
        for label, component in candidates:
            component_name = next((name for name, root in roots.items() if root is component), None)
            if component_name is None and label.rpartition(".")[2] in skipped:
                continue  # on meta: the binding does not select it
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
                    cancelled=owner._attempt_cancelled,
                    reform=owner._group_reform,
                    device=self._rank_device(torch),
                )
        sharded = {name[1] for name in owner.group_sharded if name[0] == self._group_model_key}
        hosted = {component: gpu for component, gpu in homes.items() if gpu}
        for component, gpu in hosted.items():
            if component not in roots or component in sharded or not 0 < gpu < self.world:
                return self._group_refusal(
                    GroupRefusal(f"{component} cannot be homed on GPU {gpu}", code="group_unformed")
                )
        for component_name, home in hosted.items():
            name = (self._group_model_key, component_name)
            owner.group_hosted[name] = home
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
                    cancelled=owner._attempt_cancelled,
                )
        # Every rank may take a spread call. A rank homing a placeable component takes one
        # only beside everything it holds (measured room), so it never gives up the
        # component it hosts for a clip (`Executor.run`).
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
                    cancelled=owner._attempt_cancelled,
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

    def _attempt_cancelled(self) -> bool:
        """A group's one attempt is cancelled (its calls on the other GPUs read this)."""
        return bool(self._attempt_request) and self.mux.cancelled(self._attempt_request)

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
        session with GPUs initializes them here, constructed models or not; a session
        without (`accelerator=False`), or whose environment has no Torch, imports none and
        never opens CUDA.
        Idempotent: a started process answers with what it started as.

        `import_only` (a prespawn, before the GPU grant) stops after the imports: no device
        is initialized and `self.torch` stays unset, so the process holds no device memory
        and reads as unstarted. The ordinary start that follows only initializes devices,
        warms the package and forms the group.
        """
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        self.host.gpu_memory_limit = command.gpu_memory_limit
        if command.memo is not None:
            stage_memo.ENGINE.configure(command.memo)
        if self.started or (command.import_only and self.imported):
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
            interface = package_interface.parse(
                Path(command.package_interface).read_bytes(), command.package_interface
            )
        except (OSError, ValueError, ConformanceError) as exc:
            self.poisoned = "start without a readable package interface"
            return {
                "ok": False,
                "code": "package_prepare_path_invalid",
                "detail": f"the start names no readable PackageInterface: {exc}"[:900],
            }
        # Run 6132: an environment without torch runs on the devices it was given, untouched.
        device = bool(command.devices) and importlib.util.find_spec("torch") is not None
        constructs = device and any(
            not slot.weights for entry in interface.entrypoints for slot in entry.models
        )
        if degree > 1 and not constructs:
            self.poisoned = "sequence-parallel construction unsupported"
            return {
                "ok": False,
                "code": "sequence_parallel_unsupported",
                "detail": "a group runs constructed models",
            }
        # A group's GPUs construct in lockstep at its load: it builds every model there.
        self.on_demand = frozenset(
            slot.path
            for entry in interface.entrypoints
            for slot in entry.models
            if slot.on_demand and degree == 1
        )
        started = time.perf_counter()
        stages: list[tuple[str, float]] = []
        _mark = [started]

        def stage(name: str) -> None:
            now = time.perf_counter()
            stages.append((name, round((now - _mark[0]) * 1000, 1)))
            _mark[0] = now

        imported = self.imported
        warmup_error = ""
        driverless = contextlib.ExitStack()
        try:
            if degree > 1 and self.rank == 0:
                self.world = degree
                kernel_ready.group(0, degree)
                if self.group is None:
                    self.group = RankGroup(
                        degree=degree,
                        backend="nccl" if self.device_kind == "cuda" else "cpu:gloo",
                        python=sys.executable,
                        root=str(self.root),
                        forward=self.channel.send,
                        devices=_sealed_entries(),
                        launch=self._follower,
                        waiting=functools.partial(_waiting, self.channel),
                    )
                if not self.group.followers:
                    self.group.spawn()
                    self.group.dial_back(_SEALED)
                # TensorD's lease listeners admit only its tenants: it registers each follower
                # from this frame, before the follower's Start opens its memory, pairing them
                # with GPUs by position (rank 1 first).
                self.channel.send({"event": "progress", "follower_pids": list(self.group.pids())})
                self.group.start(executor_commands.encode(command))
                stage("followers_spawned")
            if device:
                import torch

                if command.import_only:
                    driverless.enter_context(zygote.driverless(torch, len(_sealed_entries())))
                if not imported:
                    stage("torch_import")
                if not command.import_only:
                    self.torch = torch
                    try:
                        refused = self._initialize_device(torch)
                    except RuntimeError as exc:
                        # No room for a context is the device's memory, not its absence: the
                        # worker starts again once a call beside it has given the room.
                        refused = {
                            "ok": False,
                            "code": "device_out_of_memory"
                            if fault(exc) is Fault.OOM
                            else "accelerator_unavailable",
                            "detail": "the sealed devices "
                            f"{_SEALED.get('CUDA_VISIBLE_DEVICES')!r} "
                            f"could not be initialized: {exc}"[:900],
                        }
                    if refused is not None:
                        self.poisoned = f"start refused: {refused['code']}"
                        return self._close_after(refused)
                    stage("cuda_init")
                if constructs and not imported:
                    for name in _CONSTRUCTION_MODULES:
                        importlib.import_module(name)
                    stage("runtime_imports")
            if self.discovered is None:
                self._install_call_proxies(command.call_interfaces)
                self._discover_provisioned(command.release())
                stage("package_import")
            # A fork's own Start names them again; its import-only parent installed them.
            self.call_interfaces = command.call_interfaces or self.call_interfaces
            compiles = (
                self._expect_kernels(isolated=command.import_only)
                if constructs and self.rank == 0 and not imported
                else ""
            )
            if command.import_only:
                if device and accel.initialized(torch, self.device_kind):
                    self.poisoned = "the package import initialized a device before its grant"
                    return self._close_after(
                        self._refusal("cuda_initialized_before_runtime", self.poisoned)
                    )
            else:
                warmup_error = self._warm_package()
                stage("package_warmup")
            if self.group is not None:
                replies = self.group.collect("start")
                stage("group_start")
                self.follower_stages = [list(reply.get("stages") or []) for reply in replies]
                for follower, reply in zip(self.group.followers, replies, strict=True):
                    if not reply.get("ok"):
                        raise GroupRefusal(
                            f"{follower.gpu} refused the start: {reply.get('code')}: "
                            f"{reply.get('detail', '')}"[:800],
                            code="group_unformed",
                        )
                if not command.import_only:
                    measured = links.measured(
                        links.directory(_SEALED),
                        _sealed_entries()[:degree],
                        functools.partial(_waiting, self.channel),
                    )
                    stage("group_links")
                    self.transport = links.Transport(
                        degree, _gpu_numbers(degree), "nccl", measured.nccl(), measured
                    )
                    self.pg = self.group.form(
                        self.torch,
                        _SEALED,
                        nccl=self.transport.nccl,
                        formed=self._calibrated if measured.copy_engines else None,
                    )
                    stage("group_form")
        except GroupRefusal as exc:
            return self._group_refusal(exc)
        except Exception:
            self.poisoned = self.poisoned or "start raised"
            if self.group is not None:
                self.group.close()
            raise
        finally:
            driverless.close()
        self.imported = True
        self.started = not command.import_only
        took_ms = (time.perf_counter() - started) * 1000
        # A prespawned process's record keeps its import legs beside the device start's.
        self.boot = {
            "started_unix_ms": int(time.time() * 1000 - took_ms),
            "ms": round(took_ms, 1),
            "legs_ms": {**self.boot.get("legs_ms", {}), **dict(stages)},
        }
        return self._start_reply(stages, warmup_error=warmup_error, compiles=compiles)

    def fork(self, command: Fork) -> dict[str, Any]:
        """Fork this import-only process into an executor of `command.root` (`fork/1`).

        The child's imports are done; its device start initializes CUDA for the first time
        in its lineage. A process that is not provably single-threaded and driver-free
        refuses, and the machine spawns instead."""
        if self.rank or self.world != 1 or self.group is not None:
            return self._refusal("fork_unsupported", "only a world-one executor forks")
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        if not self.imported or self.started or self.constructions:
            return self._refusal(
                "fork_unready", "an executor forks after its import-only start, before a device"
            )
        moved = [n for n in zygote.LOADED if command.environment.get(n, "") != _SEALED.get(n, "")]
        if moved:
            return self._refusal(
                "fork_environment", f"{moved} were read at this process's imports and differ"
            )
        # Exited children stay zombies until now, so the machine reads their exit status.
        self.forked -= zygote.collect(self.forked)
        zygote.reap()
        reason = zygote.unsafe(sys.modules.get("torch"))
        if reason:
            return self._refusal("fork_unsafe", reason)
        started = time.perf_counter()
        pid = zygote.fork(command.environment, functools.partial(self._forked, command, started))
        self.forked.add(pid)
        return {"ok": True, "pid": pid}

    def _follower(self, module_argv: list[str], rank_fd: int) -> ChildProcess:
        """A follower rank forked from this process when its imports are done and it holds no
        thread or device (`zygote`): it starts with torch, the Runtime and the package already
        imported, which a spawned one pays again (run 6351: 6.6 s of rank 0 waiting on them).
        Otherwise it is spawned."""
        assert self.group is not None
        if not self.imported or zygote.unsafe(sys.modules.get("torch")):
            return self.group._spawn_follower(module_argv, rank_fd)
        rank, world = (int(module_argv[module_argv.index(f) + 1]) for f in ("--rank", "--world"))
        siblings = [follower.channel for follower in self.group.followers]

        def follower() -> int:
            self.channel.sock.close()
            for sibling in siblings:
                sibling.sock.close()
            stdin = os.open(os.devnull, os.O_RDONLY)
            os.dup2(stdin, 0)
            os.close(stdin)
            channel = Channel(socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=rank_fd))
            child = Executor(
                channel,
                self.root,
                fill_forensics_dir=_SEALED.get(child_env.FILL_FORENSICS_DIR, "").strip(),
                rank=rank,
                world=world,
            )
            child.imported = True
            child.discovered = self.discovered
            child.surface_names = dict(self.surface_names)
            return serve(child, channel)

        pid = zygote.fork(inherited_environment(), follower, group=False)
        gc.unfreeze()  # rank 0 serves on: its own heap is collected as before
        return spawn.Child(pid)

    def _forked(self, command: Fork, started: float) -> int:
        """In the child: an executor on its own seam, keeping this process's imports."""
        self.channel.sock.close()
        for fd, name in ((1, "stdout.log"), (2, "stderr.log")):
            log = os.open(Path(command.root) / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.dup2(log, fd)
            os.close(log)
        _capture_seal()
        kernel_cache.configure(kernel_cache.from_sealed(_SEALED))
        channel = connect(command.socket)
        child = Executor(
            channel,
            Path(command.root),
            fill_forensics_dir=_SEALED.get(child_env.FILL_FORENSICS_DIR, "").strip(),
        )
        child.imported = True
        child.discovered = self.discovered
        child.surface_names = dict(self.surface_names)
        legs = {
            **self.boot.get("legs_ms", {}),
            "fork": round((time.perf_counter() - started) * 1000, 1),
        }
        child.boot = {**self.boot, "legs_ms": legs}
        return serve(child, channel)

    def _start_reply(
        self,
        stages: list[tuple[str, float]],
        *,
        reused: bool = False,
        warmup_error: str = "",
        compiles: str = "",
    ) -> dict[str, Any]:
        total = free = -1
        if self.torch is not None and accel.present(self.torch, self.device_kind):
            memory = accel.allocation(self.torch, self.device_kind, self.host.memory)
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
            **({"compiles": compiles} if compiles else {}),
            "memory_backend": "rust-local" if self.device_kind == "cuda" else "torch",
        }

    def _model_classes(self) -> list[type[Model[Any]]]:
        """The classes the package's constructed Model slots declare, wherever they are
        defined (a library's, not only the entry module's)."""
        surfaces = getattr(self.discovered, "surfaces", ())
        slots = [s for surface in surfaces for s in surface.model_bindings if not s.weights]
        return list(dict.fromkeys(slot.model_class for slot in slots))

    def _expect_kernels(self, *, isolated: bool) -> str:
        """Start the package's kernels' compiles for each sealed card's architecture, read
        from the driver's management library: no device context, so they run beside the wait
        for the grant and the weights (h3a-087). Construction takes each build once it is
        ready. `isolated` (an import-only start, which may be forked): they start in a child."""
        sets = sorted(KERNEL_SETS)
        if not sets:
            return ""
        try:
            entries = _sealed_entries()
            if isolated:
                modules = set().union(*(KERNEL_SETS[name].modules for name in sets))
                return zygote.expect_kernels(entries, sorted(modules))
            archs = sorted({accel.device_capability(entry) for entry in entries} - {0})
            from cozy_runtime.internal import triton_kernels as built

            return "; ".join(
                f"{name} sm{arch} {built.expect(KERNEL_SETS[name], arch)}"
                for name in sets
                for arch in archs
            )
        except Exception as exc:  # an optimization, never the start's verdict
            return f"kernel compiles not started: {type(exc).__name__}: {exc}"[:300]

    def _execution_plan(
        self,
        model: Model[object],
        torch: object,
        load: Callable[..., Launch],
        ready: Callable[..., Launch | None],
        reports: dict[str, Mapping[str, object]],
    ) -> str:
        """The model's own execution plan on its filled components, on this GPU: its
        identity, "" when it installs none, and what it reports into `reports`. A raise refuses
        the construction, typed."""
        if type(model).execution_plan is Model.execution_plan:
            return ""
        device = self._rank_device(torch)

        def when_ready(kernels: TritonKernels, install: Callable[[Launch], None]) -> bool:
            built = functools.partial(ready, kernels, device=device)
            here = built() is not None
            self._installs.append(_Install(kernels.name, built, install, reports))
            return here

        try:
            identity = model.execution_plan(
                Plan(
                    device, functools.partial(load, device=device), when_ready, reports.__setitem__
                )
            )
        except Exception as exc:
            code = getattr(exc, "code", "")
            raise _PlanRefusal(
                code if isinstance(code, str) and code else "execution_plan_failed",
                f"{type(model).__name__}.execution_plan: {type(exc).__name__}: {exc}",
            ) from exc
        if not isinstance(identity, str):
            raise _PlanRefusal(
                "execution_plan_failed",
                f"{type(model).__name__}.execution_plan returned {type(identity).__name__}, "
                "not its identity string",
            )
        return identity

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

        TensorD names the key and supplies endpoint options and memory budgets. Runtime
        constructs the model and registers weight regions with its local TensorFS plane;
        the first stage requests their GPU residency. A load refused before
        it touched the process leaves the other constructions serving; one that commanded
        followers or raised poisons the process.
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
            if refused := loaded._bind_parameters(command):
                return refused
            return self._load_reply(key, loaded, reused=True)
        child = self._construction_child(key)
        try:
            reply = child._prepare_first(command)
        except Exception:
            self.poisoned = self.poisoned or child.poisoned or "load raised"
            self._close_after({})
            raise
        if not reply.get("ok"):
            if not child.poisoned and not child.commanded:
                # Refused before touching the process: its other constructions keep serving.
                return {**reply, "poisoned": ""}
            self.poisoned = self.poisoned or child.poisoned or f"group: {reply.get('code')}"
            return self._close_after({**reply, "poisoned": self.poisoned})
        self._register(key, child, reply)
        return self._load_reply(key, child, reused=False)

    def _bind_parameters(self, command: Load) -> dict[str, Any] | None:
        """A reuse binds aliases to the same path-ordered members and immutable uses."""
        members = self.model_executors or [self]
        rows = tuple(sorted(command.rows(), key=lambda row: row.binding.model_binding_path))
        if len(rows) != len(members):
            return self._refusal("model_binding_set_invalid", "a reuse must bind every member")
        paths = [row.binding.model_binding_path for row in rows]
        if len(paths) != len(set(paths)):
            return self._refusal("model_binding_set_invalid", "model paths must be unique")
        claimed = {name: member for member in members for name in member.models}
        pairs = tuple(zip(rows, members, strict=True))
        for row, member in pairs:
            try:
                use = None if row.binding.prepared_use is None else prepared_use(
                    msgspec.to_builtins(row.binding.prepared_use)
                )
            except InvalidRequest as exc:
                return self._refusal(exc.code, str(exc))
            if use != member._prepared_use:
                return self._refusal("prepared_use_binding", "changed use needs a new construction")
            for name in row.binding.parameter_names():
                if claimed.setdefault(name, member) is not member:
                    return self._refusal(
                        "model_binding_set_invalid", "a reuse aliases different model members"
                    )
        # Every row agreed before any alias is published: later refusal is atomic.
        for row, member in pairs:
            model = next(iter(member.models.values()), None)
            if model is None:
                continue
            for name in row.binding.parameter_names():
                member.models.setdefault(name, model)
                self.models.setdefault(name, model)
        return None

    def _register(self, key: str, child: Executor, reply: dict[str, Any]) -> None:
        child.load_reply = reply
        self.constructions[key] = child
        self.discovered = child.discovered
        self.surface_names = dict(child.surface_names)
        self.torch = child.torch or self.torch
        self._device_ready = child._device_ready or self._device_ready
        self.ready = True

    def _load_reply(self, key: str, child: Executor, *, reused: bool) -> dict[str, Any]:
        return {
            **child.load_reply,
            "construction": key,
            "reused": reused,
        }

    def _members(self) -> list[Executor]:
        """Every loaded model's generation in this process."""
        return [
            member
            for child in self.constructions.values()
            for member in child.model_executors or [child]
        ]

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
        reply = member._prepare_first(command)
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
        if key not in self.constructions:
            self._register(key, container, reply)
        return reply

    def activate(self, command: Activate) -> dict[str, Any]:
        """Make one loaded construction the active one. Nothing moves: its weights are
        wanted by its first stage, and an idle construction's bytes leave the device only
        when a stage needs the room (least recently used first)."""
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        key = command.construction
        child = self.constructions.get(key)
        if child is None:
            return self._refusal(
                "construction_not_loaded", f"{key!r} is not loaded in this executor"
            )
        self.active = key
        return {
            **self.probe(),
            "ok": True,
            "construction": key,
            "parked": [],
            "restored_bytes": 0,
            "held": {},
            "ms": 0.0,
        }

    def unload(self, command: Unload) -> dict[str, Any]:
        """Drop one construction: forget its weight sets, release its leases, drop every
        reference."""
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        key = command.construction
        if self._producer_hold is not None and self._producer_hold[0] == key:
            self._attention_hold.close()
            self._producer_hold = None
        child = self.constructions.pop(key, None)
        if child is None:
            return {"ok": True, "construction": key, "absent": True}
        try:
            if self.rank == 0 and self.group is not None:
                self.group.broadcast(executor_commands.encode(Unload(construction=key)))
            members = child.model_executors or [child]
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
                member._prepared_use = None
                member.prepared.clear()
                member.model_executors = []
                member.backend = member.residency = member.registry = None
            del child, members, member
            if self.active == key:
                self.active = ""
            gc.collect()
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
        self.ready = bool(self.constructions)
        return {
            **self.probe(),
            "ok": True,
            "construction": key,
        }

    def residency_document(self) -> dict[str, Any]:
        free = -1
        if self.torch is not None and accel.present(self.torch, self.device_kind):
            free = accel.allocation(self.torch, self.device_kind, self.host.memory)[
                "driver_free_bytes"
            ]
        return {
            "ok": True,
            "constructions": [
                {
                    "key": key,
                    "state": "resident",
                    "active": key == self.active,
                }
                for key, child in self.constructions.items()
            ],
            "active": self.active,
            "free_bytes": free,
        }

    def _prepare_first(self, command: Load) -> dict[str, Any]:
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
        # executor, whatever the seal's width.
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
        if self._group_parent is None and self._deferred(single.binding):
            return self._prepare_many(command, (single,))
        binding = single.binding
        try:
            use = None if binding.prepared_use is None else prepared_use(
                msgspec.to_builtins(binding.prepared_use)
            )
        except InvalidRequest as exc:
            return self._refusal(exc.code, str(exc))
        if use is not None and not binding.model_class:
            return self._refusal("prepared_use_binding", "prepared use requires a model")
        self._group_model_key = f"{self.construction}#{binding.model_binding_path}"
        started = time.perf_counter()
        # THE SEGMENT LEDGER (cr-102). A `prepare` that reports one total and one fill leg
        # cannot say which of construction, resolution or lease acquisition spent the
        # wall clock, and that ambiguity kept a 20s cost unattributed across three issues.
        stages: list[tuple[str, float]] = []
        _mark = [started]

        def stage(name: str) -> None:
            now = time.perf_counter()
            stages.append((name, round((now - _mark[0]) * 1000, 1)))
            _mark[0] = now

        self._stages = stages
        if not binding.model_class:
            return self._prepare_weightless(binding, started)
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
        from cozy_runtime.internal.triton_kernels import load as load_kernels
        from cozy_runtime.internal.triton_kernels import ready as kernels_ready
        from cozy_runtime.internal.weights import Backend, Placement

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
            self.commanded = True
            stage("group_broadcast")

        if self.discovered is None:
            self._discover_provisioned(binding)
        stage("package_import")

        registration = None
        if use is not None:
            entrypoint = binding.model_binding_path.partition(".models.")[0]
            registration = self.discovered.app.get(entrypoint)
            if binding.model_parameter_name not in registration.model_prepared_use:
                return self._refusal(
                    "prepared_use_binding", "the model declares no prepared-use field"
                )

        # The header ref identifies stored weights and metadata. Runtime opens it through
        # TensorFS in the shared store, then plans reads from its components.
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
        # route and validated geometry, the device and runtime identity and the placement —
        # and its digest is what enters identity. The LABEL derives from it.
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
        # The request this executor loads for pins a kernel: its build is imported first.
        if command.attention_pin:
            attention.reserve(command.attention_pin, device_facts)
        stage("measure_device")
        hardware_variant = binding.variant
        if not hardware_variant and device_facts.kind in ("mps", "cpu"):
            hardware_variant = device_facts.kind
        if not hardware_variant:
            if device_facts.kind != "cuda" or device_facts.sm <= 0:
                self.poisoned = "modeled accelerator unavailable"
                return {
                    "ok": False,
                    "code": "modeled_accelerator_unavailable",
                    "detail": "a model needs a GPU, and this machine has none it can use (an "
                    "NVIDIA GPU, or an Apple Silicon Mac's own): run it --on new",
                }
            hardware_variant = f"sm{device_facts.sm}"
        reference = Variant(
            name=hardware_variant,
            store=binding.store,
            snapshot=binding.snapshot,
            snapshots=dict(per_component),
        )
        vmap = reference.snapshot_map()
        host = self.host
        # One checkpoint per header ref, through the public TensorFS API.
        opened: dict[str, Checkpoint] = {}
        for manifest in {vmap.get(name, reference.snapshot) for name in names}:
            opened[manifest] = host._checkpoint(
                manifest, _connection(command.tensorfs, command.actor, host.bearers), binding.views
            )
        checkpoints = {name: opened[vmap.get(name, reference.snapshot)] for name in names}
        # `names` is what the DERIVE constructed (model-code-fit D9), not what the header
        # carries: a constructed component the checkpoint lacks is the fit's own
        # `component_missing`, judged at construction with the §3 text, so the plan is
        # resolved over the rows the header does supply and nothing here refuses first.
        present = [name for name in names if checkpoints[name].has(name)]
        rows = [r for name in present for r in checkpoints[name].rows(name)]
        stage("checkpoint_headers")
        if self.device_kind == "mps":
            # One pool, every weight resident (cr-021): past the GPU's working set never fits.
            # A CPU's weights are host memory, which pages: it is never refused here.
            need, room = sum(r.stored_nbytes for r in rows), accel.unified_ceiling(torch)
            if need > room:
                return self._refusal(
                    MEMORY_EXHAUSTED,
                    f"{binding.model or reference.snapshot} keeps {need / 2**30:.1f} GiB of "
                    f"weights on the GPU, and {device_facts.name} holds {room / 2**30:.1f} GiB: "
                    "run it on a Mac with more memory, or --on new",
                )
        # NVML is queried with the sealed device (#549.9): inside the seal, card zero IS the
        # lane's card, so a multi-GPU pod's plan names the card the fill actually uses.
        runtime_identity = measure_runtime(torch, binding.release)
        # Only the schemes these rows store. Which may serve here is a rule of the card's
        # tensor cores (`selection.admit`, xs-065): no suite runs and nothing is measured.
        stored = {row.encoded.scheme for row in rows}
        providers = {s: c for s, c in launch_providers().items() if s in stored}
        model_cls = next(
            cls for cls in self._model_classes() if cls.__name__ == binding.model_class
        )
        config_checkpoint = next(
            (
                checkpoint
                for checkpoint in checkpoints.values()
                if checkpoint.id == reference.snapshot
            ),
            None,
        ) or host._checkpoint(
            reference.snapshot,
            _connection(command.tensorfs, command.actor, host.bearers),
            binding.views,
        )
        config_bytes = model_config.construction_config(config_checkpoint.configs())
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
        prepared_ineligible = ""
        if use is not None:
            assert registration is not None
            if self.world != 1 or len(registration.surface.slots) != 1:
                prepared_ineligible = (
                    "multi-rank or multi-slot preparation remains on its existing route"
                )
            elif binding.adapters or any(
                model_config.ADAPTERS in held.assets
                for checkpoint in checkpoints.values() for held in checkpoint.ck.components.values()
            ):
                prepared_ineligible = (
                    "adapter preparation has no fixed complete-footprint allowance"
                )
        prepared_memory = host.memory
        try:
            plan = resolve(
                variant=reference,
                rows=rows,
                providers=providers,
                device=device_facts,
                runtime=runtime_identity,
                construction=construction,
                placement=binding.placement or "all_resident",
                encoded_leaves=consent,
                dtype_name=dtype_name,
                prepared_use=use,
                prepared_free_bytes=(
                    prepared_memory.room().free
                    if use is not None and prepared_memory is not None else 0
                ),
                prepared_granularity=(
                    prepared_memory.granularity if prepared_memory is not None else 0
                ),
                prepared_ineligible=prepared_ineligible,
            )
        except PlanRefusal as exc:
            return {
                "ok": False,
                "code": exc.code,
                "detail": str(exc).splitlines()[0][:900],
                "plan": {"walk": msgspec.to_builtins(exc.steps)},
            }
        stage("resolve")
        mine = GroupPlan(
            degree=self.world,
            devices=_sealed_entries(),
            plan_digest=rank_invariant_digest(plan.identity()),
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
        host = self.host
        device = self._rank_device(torch)
        assert host.memory is not None  # opened with the device (`_open_memory`)
        backend = Backend(
            checkpoints,
            rows,
            torch=torch,
            plan=plan,
            memory=host.memory,
            device=device,
            construction=self._group_model_key,
            device_facts=device_facts,
            release=binding.release,
            custody=binding.custody,
        )
        stage("backend_ctor")
        artifact = Artifact(
            reference.snapshot,
            tensor_schema_of(rows),
            Config(config_document, "model.config"),
            assets=config_checkpoint.assets(),
            # CUSTODY is a BINDING fact: canonical refuses a serve-time extra key, local
            # tiers it (§1.1).
            custody=binding.custody,
            unnormalized=normalized(config_checkpoint),
        )
        adapter_graphs = [
            (checkpoint.id, checkpoint.asset(component, model_config.ADAPTERS))
            for checkpoint in {c.id: c for c in checkpoints.values()}.values()
            for component, held in checkpoint.ck.components.items()
            if model_config.ADAPTERS in held.assets
        ]
        if adapter_graphs:
            manifest, data = adapter_graphs[0]
            graph = read_adapter_graph(data)
            if any(read_adapter_graph(other) != graph for _, other in adapter_graphs[1:]):
                raise CapabilityError(
                    "one model slot selects different prepared adapter graphs",
                    code="adapter_graph_conflict",
                )
            artifact = bind_adapters(
                artifact,
                model_cls,
                data,
                lambda component, name: checkpoints[component].binding(component)(name),
                prepared_snapshot=manifest,
            )
        seen = Observations()
        registry = ModelRegistry(
            release=binding.release,
            backend=backend,
            substrate=lambda: serving_substrate(hardware_variant, seen),
        )
        try:
            backend.expect(
                {name: tuple(r.key for r in checkpoints[name].rows(name)) for name in present}
            )
            stage("expect")
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
            stage("construct_and_register")
        except ModelFitRefused as exc:
            # THE FIT VERDICT (model-code-fit §3), verbatim. Nothing was allocated: the
            # process keeps serving its other constructions.
            backend.close()
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
            }
        except FillRefusal as exc:
            # Refused before a weight byte moved: the weights live in the plane, and nothing
            # of this construction was mapped yet.
            backend.close()
            return {"ok": False, "code": exc.code, "detail": str(exc).splitlines()[0][:900]}
        try:
            # THE MODEL'S OWN EXECUTION PLAN: its fused kernels, glue and hooks.
            reports = self.plan_reports[binding.model_binding_path] = {}
            planned = self._execution_plan(model, torch, load_kernels, kernels_ready, reports)
            self.planned[binding.model_binding_path] = planned
            # Runtime's cuts and block names after the plan: its substitutions stand, and cuts
            # land only on what stays Diffusers (v13: H3's glue refused modules they had wrapped).
            backend.install_execution()
            roots = backend.roots
            stage("execution_plan")
            # EACH SITE'S KERNEL FROM THE GROUP'S FAMILY CELL AT THE MODEL'S ATTENTION LEVEL,
            # PINNED POST-FILL (cr-124), before any warm step (`attention.select`).
            sms = [accel.device_capability(entry) for entry in _sealed_entries()]
            self._reselect = functools.partial(
                attention.select,
                device_facts,
                roots,
                degree=self.world,
                bits=type(model).__attention_bits__,
                row=arch.common(map(arch.row, sms)) if self.world > 1 and all(sms) else None,
            )
            applied = self._reselect()
            self.device_facts = device_facts
            stage("attention_pin")
        except (attention.AttentionRefusal, _PlanRefusal) as exc:
            backend.poison(())
            self.poisoned = f"execution plan refused: {exc.code}"
            return {"ok": False, "code": exc.code, "detail": str(exc)[:900]}
        self.backend = backend
        self.registry = registry
        # Keep aliases accepted while an on-demand member was still a proxy (h3a-018).
        aliases = dict.fromkeys((*binding.parameter_names(), *self.models))
        self.models = dict.fromkeys(aliases, model)
        # THE COMPONENT-USE CONTRACT'S RUNTIME BODY: from here on a declared scope is a
        # shared `use` of its components, and each block binds its weights while it runs.
        self.residency = Placement(host.memory, backend)
        object.__setattr__(model, "_cozy_residency", self.residency)
        # Its components' calls replay CUDA graphs once recorded (`graphs`), unless uncaptured.
        graphs.install(self.torch, model, {name: m.module for name, m in backend.models.items()})
        stage_memo.install(
            model,
            torch=torch,
            headers={
                name: {
                    "components": {name: checkpoints[name].table(name)},
                    "configs": checkpoints[name].configs(),
                }
                for name in present
            },
            assets={
                "assets": config_checkpoint.asset_digests(),
                "configs": config_checkpoint.configs(),
            },
            plan_rows=({**row.identity(), "component": row.component} for row in plan.tensors),
            adapters=(msgspec.to_builtins(row) for row in binding.adapters),
            roots=roots,
            device=device_facts,
            world=self.world,
            sealed=_SEALED,
            plan=planned or None,
        )
        record = registry.generations()[0].record
        self.plan = plan
        self.refused_leaves = plan.refused_leaves()
        for group in self.refused_leaves:
            print(
                f"[executor] {group.component}: {group.tensors} {group.lane} tensor(s) serve "
                f"decoded, leaf {group.leaf} refused on {device_facts.predicate}: {group.reason}",
                file=sys.stderr,
                flush=True,
            )
        self.attention = applied
        constructed_preimage = canonical.write(
            {
                "plan_digest": plan.digest(),
                "execution_plan": planned,
            }
        )
        self.constructed_model_digest = (
            "sha256:"
            + hashlib.sha256(
                b"cozy.runtime.constructed-generation\0" + constructed_preimage
            ).hexdigest()
        )
        if self.group is not None:
            # THE FOLLOWERS BUILT THE SAME PLAN, or the group is not a group.
            try:
                replies = self.group.collect("load")
            except GroupRefusal as exc:
                return self._group_refusal(exc)
            for follower, reply in zip(self.group.followers, replies, strict=True):
                kernel_ready.report(follower.rank, reply.get("kernels") or {})
                if not reply.get("ok"):
                    code = str(reply.get("code") or "group_unformed")
                    return self._group_refusal(
                        GroupRefusal(
                            f"{follower.gpu} refused the prepare: {code}: "
                            f"{reply.get('detail', '')}"[:800],
                            code=code if code in _GROUP_CODES else "group_unformed",
                        )
                    )
                theirs = dict(reply.get("facts") or {})
                try:  # THE FOLLOWERS RESOLVED THE SAME PLAN (cr-068)
                    GroupPlan.read(theirs.get("group_plan") or {}).assert_agrees(
                        mine, rank=follower.rank
                    )
                except (GpuDivergence, msgspec.ValidationError) as exc:
                    return self._group_refusal(
                        exc
                        if isinstance(exc, GpuDivergence)
                        else GpuDivergence(follower.gpu, "group_plan", str(exc))
                    )
                if theirs.get("filled_bytes") != record.filled_bytes:
                    return self._group_refusal(
                        GpuDivergence(
                            follower.gpu,
                            "filled_bytes",
                            f"{_gpu(0)} built {record.filled_bytes!r}, {follower.gpu} built "
                            f"{theirs.get('filled_bytes')!r}",
                        )
                    )
            refused = self._agree_attention()
            if refused is not None:
                return refused
        self.facts = {}
        refused = self._install_group(torch, model, binding.homes)
        if refused is None and self.rank == 0:
            # The execution plan's kernels already built for every GPU install now, before
            # the first request, on every GPU (`_install_ready_kernels`).
            refused = self._install_ready_kernels()
        if refused is not None:
            return refused
        stage("install")
        applied = self.attention  # the group may have selected again (`_agree_attention`)
        if attention.sparse_names(applied.totals()):
            frozen = attention._baked_in(dict(backend.roots))
            if frozen:
                backend.poison(())
                self.poisoned = "attention refused: attention_kernel_frozen"
                return {
                    "ok": False,
                    "code": "attention_kernel_frozen",
                    "detail": f"Sol requires eager execution: {frozen}",
                }
        self.attention_defaults = attention.snapshot(dict(backend.roots))
        group_facts = dict(self.facts)
        memory = accel.allocation(torch, self.device_kind, self.host.memory)
        self.ready = True
        self.facts = {
            **group_facts,
            "cuda_init_ms": round(cuda_ms, 2),
            "prepare_ms": round((time.perf_counter() - started) * 1000, 2),
            "prepared_unix_ms": int(time.time() * 1000),
            # THE SEGMENT LEDGER (cr-102), in execution order.
            "stages": list(stages)
            + [(f"construct.{name}", ms) for name, ms in record.stage_ms]
            + [(f"register.{name}", ms) for name, ms in backend.legs],
            "substrate_census": list(record.weightless),
            "execution_plan": planned,
            "filled": record.filled,
            "filled_bytes": record.filled_bytes,
            **({"group_plan": mine.document()} if self.rank else {}),
            "device_free_bytes": memory["driver_free_bytes"],
            "device_total_bytes": memory["driver_total_bytes"],
            "allocator_bytes": memory["allocated_bytes"],
            "reserved_bytes": memory["reserved_bytes"],
            "rss_bytes": _rss(),
            "gpu_name": accel.device_identity(torch, self.device_kind)["name"],
            "device_driver": device_facts.driver,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "",
            # THE CONFESSION (§1.1): the extras this bind dropped ride the bind result.
            "custody": binding.custody,
            "declared_scopes": {
                method: list(names) for method, names in sorted(component_use(model_cls).items())
            },
            "ignored_extra_keys": list(record.ignored_extras),
            "ignored_extra_warnings": list(record.ignored_warnings),
            # built components the binding does not select: left on meta
            "skipped_components": list(record.skipped),
            "fit": record.fit,
            "plan": plan.seam_document(),
            "delivery": plan.wire_document(),
            "attention": applied.document() if applied is not None else None,
        }
        self._prepared_use = use
        return {
            "ok": True,
            "ready": True,
            "constructed_model_digest": self.constructed_model_digest,
            "entrypoints": sorted(self.surface_names),
            "facts": self.facts,
        }

    def _deferred(self, binding: Binding) -> bool:
        """Whether this model is built on first use (`on_demand=`)."""
        return self.world == 1 and binding.model_binding_path in self.host.on_demand

    def _prepare_many(self, command: Load, rows: tuple[ModelLoad, ...]) -> dict[str, Any]:
        """Prepare one complete bounded multi-model callable in this device process; an
        on-demand model is built when the function first reads it (`_build_on_demand`)."""
        if not 1 <= len(rows) <= 16:
            return {
                "ok": False,
                "code": "model_binding_set_invalid",
                "detail": "models must be an array of 1 through 16 binding rows with budgets",
            }
        # The machine sends a function's slots in its declaration order; one order is ours.
        rows = tuple(sorted(rows, key=lambda row: row.binding.model_binding_path))
        paths = [row.binding.model_binding_path for row in rows]
        parameters = [row.binding.model_parameter_name for row in rows]
        aliases = [name for row in rows for name in row.binding.parameter_names()]
        if (
            not all(paths)
            or len(paths) != len(set(paths))
            or not all(parameters)
            or len(aliases) != len(set(aliases))
        ):
            return {
                "ok": False,
                "code": "model_binding_set_invalid",
                "detail": "model paths and parameters must be named and unique",
            }

        try:
            uses = tuple(
                None if row.binding.prepared_use is None else prepared_use(
                    msgspec.to_builtins(row.binding.prepared_use)
                )
                for row in rows
            )
        except InvalidRequest as exc:
            return self._refusal(exc.code, str(exc))
        children: list[tuple[str, Executor, dict[str, Any]]] = []
        for row in rows:
            binding = row.binding
            parameter = binding.model_parameter_name
            child = self._construction_child(self.construction)
            child._group_parent = self
            # The command with one model in place of its set: everything the machine said
            # about the load holds for each model.
            load = msgspec.structs.replace(
                command,
                construction=self.construction,
                binding=binding,
                models=UNSET,
                group=UNSET,
                model_key=f"{self.construction}#{binding.model_binding_path}"
                if self.world > 1
                else UNSET,
                sequence_parallel_degree=max(self.world, 1),
            )
            if self._deferred(binding):
                # Built inside the attempt that reads it.
                build = functools.partial(self._build_on_demand, child, load)
                child.models = dict.fromkeys(binding.parameter_names(), _OnDemand(build))
                children.append((parameter, child, {}))
                continue
            reply = child._prepare_first(load)
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
        built = [(parameter, child, reply) for parameter, child, reply in children if reply]
        if self.discovered is None:
            self._discover_provisioned(rows[0].binding)
        self.model_executors = [child for _, child, _ in children]
        self.torch = built[-1][1].torch if built else self.torch
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
            for parameter, _, reply in built
        ]
        self.constructed_model_digest = (
            "sha256:"
            + hashlib.sha256(
                b"cozy.runtime.constructed-generation-set\0" + canonical.write(constructed)
            ).hexdigest()
        )

        facts_rows = [(parameter, reply["facts"]) for parameter, _, reply in built]
        last = facts_rows[-1][1] if facts_rows else {}
        scopes = {
            f"{parameter}.{method}": [f"{parameter}/{component}" for component in components]
            for parameter, facts in facts_rows
            for method, components in (facts.get("declared_scopes") or {}).items()
        }
        layouts = {
            f"{parameter}/{component}": layout
            for parameter, facts in facts_rows
            for component, layout in (facts.get("layouts") or {}).items()
        }
        # WHAT EACH MODEL PINNED, MERGED (cr-124). A group prepare has one boot note, so a
        # per-model `attention` fact would simply be dropped and the placement would report
        # no kernel at all — the one fact the boot line exists to carry.
        attention_fact = attention.merge(
            {parameter: facts.get("attention") or {} for parameter, facts in facts_rows}
        )
        routes: dict[str, int] = {}
        kernels: dict[str, int] = {}
        deliveries = {parameter: facts.get("delivery") or {} for parameter, facts in facts_rows}
        for delivery in deliveries.values():
            for name, count in (delivery.get("routes") or {}).items():
                routes[str(name)] = routes.get(str(name), 0) + int(count)
            for name, count in (delivery.get("kernels") or {}).items():
                kernels[str(name)] = kernels.get(str(name), 0) + int(count)
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
            "filled": sum(int(facts.get("filled", 0)) for _, facts in facts_rows),
            "filled_bytes": sum(int(facts.get("filled_bytes", 0)) for _, facts in facts_rows),
            "device_free_bytes": int(last.get("device_free_bytes", -1)),
            "device_total_bytes": int(last.get("device_total_bytes", -1)),
            "allocator_bytes": int(last.get("allocator_bytes", -1)),
            "reserved_bytes": int(last.get("reserved_bytes", -1)),
            "layouts": layouts,
            "declared_scopes": scopes,
            "rss_bytes": max(
                (int(facts.get("rss_bytes", 0)) for _, facts in facts_rows), default=0
            ),
            "attention": attention_fact,
            "gpu_name": str(last.get("gpu_name", "")),
            "torch_version": str(last.get("torch_version", "")),
            "cuda_version": str(last.get("cuda_version", "")),
            "custody": "joint",
            "delivery": delivery,
            "models": {parameter: facts for parameter, facts in facts_rows},
        }
        # Deferred members are accepted bindings even before their first materialization.
        for (_, member, reply), use in zip(children, uses, strict=True):
            if not reply:
                member._prepared_use = use
        self.ready = True
        return {
            "ok": True,
            "ready": True,
            "constructed_model_digest": self.constructed_model_digest,
            "entrypoints": sorted(self.surface_names),
            "facts": self.facts,
        }

    def _build_on_demand(self, member: Executor, load: Load) -> Any:
        """An on-demand model's first read: the load its construction deferred, run in the
        attempt that reads it; a pass that never uses it builds none of it."""
        parameter = load.single().binding.model_parameter_name
        try:
            reply = member._prepare_first(load)
        except Exception:
            self.poisoned = self.poisoned or member.poisoned or f"model {parameter} raised"
            raise
        if not reply.get("ok"):
            self.poisoned = self.poisoned or member.poisoned
            detail = f"model {parameter} refused: {reply.get('detail', '')}"
            raise RuntimeFailure(detail[:900], code=str(reply.get("code") or "runtime_failure"))
        member.load_reply = reply
        self.built_on_demand[parameter] = member.facts
        self.torch = self.torch or member.torch
        if member.residency is not None:
            self.residency.rows[parameter] = member.residency
        return member.models[parameter]

    def _discover_provisioned(self, release: Release) -> None:
        """Import the installed release and bind its PackageInterface. No elapsed-time limit:
        the worker or machine ends an import only on its measured lack of progress, and a slow
        first import on a busy host is still an import."""
        if not release.application or not release.package_interface:
            raise RuntimeError("executor requires the canonical installed release identity")
        path = release.package_interface
        recipes = package_interface.parse(Path(path).read_bytes(), path).recipes
        discovered = discover_installed(release.application, seconds=None, recipes=recipes)
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

    def _discard(self, request_id: str) -> None:
        """A request cancelled before its invoke (waiting for admission): what its prepare
        retained goes, so the executor returns to the pool holding nothing of it."""
        for construction in (self, *self.constructions.values()):
            with construction._prepared_lock:
                construction.prepared.pop(request_id, None)
        if self.host.memory is not None:
            # Never hold the prepared-record lock while touching a native read owner.
            self.host.memory.cancel_queued(request_id)

    def serve_request(self, command: PrepareRequest) -> dict[str, Any]:
        target = self._target(command, attempt=False)
        if isinstance(target, dict):
            return target
        return target.prepare_request(command)

    def serve_invoke(self, command: Invoke) -> dict[str, Any]:
        target = self._target(command, attempt=True)
        if isinstance(target, dict):
            return target
        requested = _attention_pin(command.attention_kernel, command.bits)
        alone = self.world > 1 or bool(requested and target._attention_reaches(requested))
        with self.in_flight.attempt(alone, command.request_id) as slot, request_path.serving():
            # Admitted only now: an attempt before it may have poisoned the generation. Its
            # record is taken once nothing can refuse it, so a refused call may come again.
            refused = target._unadmitted(command) or self._grant(target, first=slot.first)
            if not alone:
                self.in_flight.ready()
            if refused is not None:
                return refused
            with target._prepared_lock:
                prepared = target.prepared.pop(command.request_id, None)
            if prepared is None:  # trimmed by a prepare meanwhile
                return target._unadmitted(command) or {}
            reply = target.invoke(command, prepared, slot)
        self.attempts += 1
        self.poisoned = self.poisoned or target.poisoned
        reply["poisoned"] = self.poisoned
        return reply

    def _grant(self, target: Executor, *, first: bool) -> Frame | None:
        """The kernels that became ready, installed while no call is in a scope (a group none
        of whose GPUs may serve otherwise refuses); the first attempt in also opens the
        residency's attempt."""
        turn = object()
        target._granted = kernel_ready.boundary()
        with SCOPES.held(turn):
            target._upgrade_attention(turn)
            refused = target._install_ready_kernels()
            if refused is not None:
                return {**refused, "poisoned": target.poisoned}
            if first and target.residency is not None:
                target.residency.open_attempt()
        return None

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
        canceled = {
            "ok": False,
            "code": "cancelled",
            "terminal": "canceled",
            "origin": "request",
            "detail": "request preparation was canceled before entry",
        }
        if self.host.mux.cancelled(command.request_id):
            return canceled
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
        pin = _attention_pin(command.attention_kernel, command.bits)
        waits = Attribution()
        if pin and self._attention_reaches(pin):
            held = kernel_compile.held.set(
                functools.partial(_held, self.host.mux, command.request_id, waits)
            )
            try:
                self._attention_plan(pin)
            except attention.AttentionRefusal as exc:
                return {
                    "ok": False,
                    "code": exc.code,
                    "detail": str(exc)[:900],
                    "origin": "request",
                }
            except kernel_compile.Canceled as exc:
                return {
                    "ok": False,
                    "code": "cancelled",
                    "terminal": "canceled",
                    "detail": str(exc),
                    "origin": "request",
                }
            finally:
                kernel_compile.held.reset(held)
        try:
            self._check_prepared_use(registration, command.payload)
            prepared = prepare(registration, command.payload, input_metadata=command.input_metadata)
        except Exception as exc:
            outcome = classify(exc)
            return {
                "ok": False,
                "code": outcome.code,
                "detail": ends(outcome.message, 1024),
                "terminal": outcome.terminal,
                "origin": outcome.origin,
                "fields": list(outcome.fields),
            }
        with self._prepared_lock:
            # Mux records the cancel before calling _discard under this same lock. A
            # cancel before this check refuses publication; one after it removes the row.
            if self.host.mux.cancelled(command.request_id):
                return canceled
            if not self.ready or self.poisoned:
                return {
                    "ok": False,
                    "code": "poisoned_generation" if self.poisoned else "executor_not_ready",
                    "detail": self.poisoned or "executor stopped during request preparation",
                }
            self.prepared[command.request_id] = replace(
                prepared,
                stages=waits.stages,
                prefetch_key=(command.attention_kernel, command.bits),
            )
            # A prepared request never dispatched must not survive for the generation.
            for stale in list(self.prepared)[: -PREPARED_HELD or None]:
                self.prepared.pop(stale, None)
        return {
            "ok": True,
            "features": dict(prepared.features.values),
            "adjustments": [adjustment_row(row) for row in prepared.overlay.rows],
            **prepared.digests(),
        }

    def _unadmitted(self, command: Invoke) -> Frame | None:
        """Why this construction cannot run `command` now; None when it can."""
        if not self.ready:
            return {
                "ok": False,
                "code": "executor_not_ready",
                "detail": "an online attempt against a non-Ready executor: the directive "
                "prepares the executor first — serving never accepts-then-cold-loads",
            }
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        if self.surface_names.get(command.entrypoint) is None:
            return {
                "ok": False,
                "code": "unknown_entrypoint",
                "detail": f"{command.entrypoint!r} is not an entrypoint of this release",
            }
        with self._prepared_lock:
            missing = command.request_id not in self.prepared
        if missing:
            return {
                "ok": False,
                "code": "prepared_request_absent",
                "detail": f"nothing was prepared for {command.request_id!r}: an attempt "
                "runs the record its plan was priced against, and this executor holds none "
                "(a fresh epoch holds nothing at all)",
            }
        return None

    def invoke(self, command: Invoke, prepared: PreparedRequest, slot: _Slot) -> dict[str, Any]:
        """ONE attempt through the author kernel, with the §3.4 freeze point held."""
        surface = self.surface_names[command.entrypoint]
        requested = _attention_pin(command.attention_kernel, command.bits)
        # A pin is the run's: a call holding no site in its scope runs its own selection,
        # and the worker reports a run none of whose calls applied it. A pinned attempt runs
        # alone; another changes no site, so calls beside it see only upgrades.
        pin = requested if requested and self._attention_reaches(requested) else ""
        swap = None
        if pin:
            try:
                swap = self._attention_plan(pin)
            except attention.AttentionRefusal as exc:
                return {
                    "ok": False,
                    "code": exc.code,
                    "detail": str(exc)[:900],
                    "origin": "request",
                }
            finally:
                self._restore_attention()
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
        own = self.host.memory
        allocated_at_start = (
            accel.allocated(torch, self.device_kind, own) if torch is not None else 0
        )
        lease_acquired = time.perf_counter()
        started_wall = time.time()
        started_us = execution_evidence.now_us()
        served_attention: attention.Applied | None = None
        sol_counts: dict[str, int] = {}
        sol_rank_counts: dict[int, dict[str, int]] = {}
        sol_kernels: list[dict[str, Any]] = []
        served_impls: dict[str, set[str]] = {}
        # A kernel that finishes compiling mid-run serves from the next step or stage on
        # (`_upgrade_attention`): each selection the run served and from which step.
        segments: list[dict[str, Any]] = []
        later: list[attention.Applied] = []
        #: each kernel that became ready mid-run, and the step from which it served; in a
        #: group, on which GPU
        here: dict[str, str | int] = {} if self.group is None else {"gpu": self.group.gpu(0)}
        point = [{"stage": "", "step": 0, **here}]  # the last boundary
        swapped = [(name, point[0]) for name in self._granted]
        steps: dict[str, int] = {}
        stage = [""]  # the last stage this attempt itself reported
        handler = threading.get_ident()

        def followed() -> None:
            """What each follower began serving since the last boundary, at it."""
            if self.group is not None:
                began, self.group.began = self.group.began, []
                gpu = self.group.gpu
                swapped.extend((kernel, {**point[0], "gpu": gpu(rank)}) for rank, kernel in began)

        def boundary(frame: Any) -> None:
            progress(frame)
            if isinstance(frame, ProgressFrame) and frame.call_request is None and frame.stage:
                stage[0] = frame.stage
            if not isinstance(frame, ProgressFrame) or threading.get_ident() != handler:
                return
            followed()
            if frame.position is not None:
                steps[frame.stage] = frame.position
            at = point[0] = {"stage": frame.stage, "step": steps.get(frame.stage, 0), **here}
            swapped.extend((name, at) for name in kernel_ready.boundary())
            if pin or served_attention is None:
                return  # an override holds its kernel for every step
            if not segments:
                segments.append(_segment("", 0, served_attention, started_us))
            if not self._upgrade_attention(CURRENT.get()):
                return
            later.append(self._attention_selection(attention.observed(self._attention_roots())))
            row = _segment(frame.stage, steps.get(frame.stage, 0), later[-1])
            if row["observed"] != segments[-1]["observed"] or row["dense"] != segments[-1]["dense"]:
                segments.append(row)

        record: AuthorAttempt | None = None
        # Weights grants are the attempt's own: its sources and outputs end with it.
        sources: dict[str, RawWeights[Any]] = {
            name: RawWeights(manifest) for name, manifest in command.sources.items()
        }
        exchange = self._exchange(command.request_id)
        connection = _connection(command.tensorfs, command.actor, self.bearers)
        weight_outputs = (
            weights_store.Outputs(
                self._store(),
                request_id=command.request_id,
                actor=command.actor,
                code=_replay_code(command, prepared.overlay.digest),
                bounds=command.weights,
                show=functools.partial(_show_checkpoint, exchange),
            )
            if command.weights
            else None
        )
        stage_memo.ENGINE.open(exchange, spool)
        warnings: list[WarningRow] = []
        blocking: dict[str, tuple[float, int]] = {}
        own_memory: list[GpuMemory] = []
        rank_memory: list[GpuMemory] = []
        try:
            with (
                WARNINGS.attempt() as warnings,
                request_path.serving() as blocking,
                self._request(command.request_id, prepared) as own_memory,
                self._ranks_request(command.request_id) as rank_memory,
                attention.override(self._attention_snapshot(), self._attention_roots(), swap, pin)
                if pin
                else contextlib.nullcontext(
                    attention.observed(self._attention_roots())
                ) as served_attention,
                self._ranks_attention(pin, served_attention),
                attention_sol.observing(ranks=sol_rank_counts, kernels=sol_kernels) as sol_counts,
                attention_ulysses.observing() as served_impls,
            ):
                served_attention = self._attention_selection(served_attention)
                if prepared.phases:
                    self._plan_request(command.request_id, prepared.phases)
                envelope, outcome, record = run_prepared(
                    prepared,
                    Invocation(
                        request_id=command.request_id,
                        spool=spool,
                        deadline=deadline,
                        device=self._device(),
                        cancel=functools.partial(self.host.mux.cancelled, command.request_id),
                        models=self.models,
                        recipes=self.discovered.recipes,
                        progress=boundary,
                        plan=functools.partial(self._plan_request, command.request_id),
                        # cr-012: the inputs the WORKER already fetched, verified and spooled.
                        # Every value is a path inside this attempt's own spool: this process is
                        # `sandbox.refuse_network` rejects ungranted network access.
                        # Input bytes arrive only through the worker grant.
                        # That fence is Python-level (cr-042) — it bounds a buggy package, not a
                        # hostile one, and there is no netns or seccomp under it.
                        trees=_trees(command),
                        assets=_granted_inputs(command),
                        max_input_bytes=(
                            None
                            if command.max_input_bytes is None
                            else command.max_input_bytes or 64 << 20
                        ),
                        max_output_bytes=command.max_output_bytes,
                        sources=sources,
                        weights_bytes=weights_store.read_into,
                        weights_values=functools.partial(read_values_into, self.torch),
                        tensorfs_source=(
                            functools.partial(
                                weights_store.open_checkpoint,
                                self._store(),
                                connection=connection,
                            )
                            if sources
                            else None
                        ),
                        tensorfs_output=(
                            weight_outputs.open if weight_outputs is not None else None
                        ),
                        tensorfs_adopt=(
                            weight_outputs.adopt if weight_outputs is not None else None
                        ),
                        publish=lambda request: exchange(request, Published),
                        calls=self.host._call_broker(command.request_id, self.host.call_interfaces),
                    ),
                )
        except Exception as exc:
            envelope, outcome = None, classify(exc)
            unwind(exc)
            if record is None:
                record = AuthorAttempt(command.request_id, spool, 0, sink=progress)
        assert record is not None
        self.handled[command.request_id] = (outcome, stage[0], record)
        # A failure the device raised (CUDA's or NCCL's own error, not capacity), or one that
        # left it faulted: this process serves nothing more, so it is poisoned and the machine
        # ends it, freeing everything (12167 and 12293 ran on a faulted one and failed again).
        failed = outcome.terminal == "failed" and torch is not None
        if failed and (outcome.code == DEVICE_FAULT or not accel.healthy(torch, self.device_kind)):
            self.poisoned = self.poisoned or f"device fault: {outcome.message}"[:300]
        if self.rank == 0:
            for member in self.model_executors or [self]:
                self._emit_gemm(record, member)
            for feature, instead in accel.unavailable(self.device_kind) if self.models else ():
                record.emit(
                    "log", f"{feature} unavailable on {self.device_kind}", "warn", instead=instead
                )
            for message, raised in (*warnings, *tools.warnings()):
                record.emit("log", message, "warn", **raised)
            for work, (ms, times) in blocking.items():
                took = {"ms": round(ms, 1), "times": times}
                record.emit("confession", "request_path.blocking", work, **took)
                record.emit("log", "kernel work blocked the request", "warn", work=work, **took)
            followed()
            for name, where in swapped:
                record.emit("confession", "kernel.ready", name, **where)
            members = self.model_executors or [self]
            facts = next((m.device_facts for m in members if m.device_facts), None)
            if facts is not None and (unnamed := arch.inherited(facts.sm, facts.name)):
                record.emit("log", unnamed, "warn")
        stage_memo.ENGINE.close(record.emit)
        if later:
            # The override put back the run's first selection; the construction keeps its last.
            self._restore_attention()
            for held in later:
                served_attention = served_attention.joined(held) if served_attention else held
        if served_attention is not None:
            self._emit_attention(record, served_attention, "attention.selected")
            sparse_kernels = attention.sparse_names(served_attention.totals())
            if sparse_kernels:
                sparse_name = "+".join(sparse_kernels)
                record.emit(
                    "confession",
                    "attention.sol.calls",
                    sparse_name,
                    rank=0,
                    world=self.world,
                    dense_reference=_dense(sparse_kernels),
                    **sol_counts,
                )
                for rank, counts in sorted(sol_rank_counts.items()):
                    record.emit(
                        "confession",
                        "attention.sol.calls",
                        sparse_name,
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
            for row in attention.failures(served_attention.totals(), sol_kernels):
                record.emit("log", "attention kernel failed, a fallback served", "warn", **row)
        # Nothing was swapped, so the restore left the sites exactly as the entry walk read them.
        restored = (
            served_attention
            if swap is None and not pin and served_attention is not None and not later
            else attention.observed(self._attention_roots())
        )
        self._emit_attention(record, restored, "attention.restored")
        # The last attempt out drains the device, with none running.
        last = self.host.in_flight.last(slot)
        varena = [*own_memory, *rank_memory]
        memory = {"varena": msgspec.to_builtins(varena)} if varena else None
        handler_ms = (time.perf_counter() - lease_acquired) * 1000
        self._attempt_spool = self.host._attempt_spool = None
        self.host._attempt_request = ""
        if self.group is not None and self.group.broken and not self.poisoned:
            # A FOLLOWER DIED OR DIVERGED UNDER THIS ATTEMPT (cr-068): the group is one
            # executor, so this process is poisoned with it and the worker rebuilds all K.
            self.poisoned = f"group: {self.group.broken}"[:200]

        # §3.4 step 2: the handler's own stream drains BEFORE the device lease releases, so
        # everything the tail touches is immutable. Only the compute stream: the plane's
        # copy streams may still be prefetching the next pass, and that is not the
        # handler's state. The allocator keeps its segments for the next attempt; an idle
        # tenant gives room by a budget cut, never by a per-request cache flush. A
        # WEIGHTLESS generation holds no stream, so its quiescence is structural.
        d2h_started = time.perf_counter()
        if torch is not None and last:
            accel.drain(torch, self.device_kind)
        if self.residency is not None and last:
            self.residency.release()  # the last step's holds
        d2h_ms = (time.perf_counter() - d2h_started) * 1000
        released_us = execution_evidence.now_us()
        # Stock Torch's peak is absolute and process-wide. Overlapping attempts share it;
        # never add the request-start allocation again or present it as exclusive ownership.
        arena = (own_memory[0].torch_peak or 0) if own_memory else 0
        live = accel.allocated(torch, self.device_kind, own) if torch is not None else 0
        peak_vram = max(arena, live) if self.device_kind == "cuda" else live
        peak_reserved = accel.reserved(torch, self.device_kind, own) if torch is not None else 0
        residency_document = self.residency.document() if self.residency else {}
        working_peak = max(peak_vram - allocated_at_start, 0)
        if self.group is not None:
            working_peak = max(working_peak, *self.group.working_peaks.values(), 0)
        # step 3: the lease releases here, and nothing after this line touches the device.
        # DEVICE RELEASE IS THIS REPLY (cr-079): the handler registered its outputs as raw
        # host frames, so `device_lease_ms` is device time and the encode is the worker's.
        lease_ms = (time.perf_counter() - lease_acquired) * 1000
        # An attempt leaving others running has its outputs on the host; theirs are theirs.
        quiescent = self._quiescent() if torch is not None and last else True
        self.attempts += 1
        result_ref, outputs, oversize = _serialize_result(envelope, spool)
        # A failed request frees everything it allocated and never poisons the executor: bytes
        # package state still holds show as `leftover` (memory-varena §5.7).
        terminal = {
            "terminal": outcome.terminal,
            "origin": outcome.origin,
            "code": outcome.code,
            "message": ends(outcome.message, 2048),
            "fields": list(outcome.fields),
            "traceback": outcome.traceback,
        }
        # The BOUNDED observation tail and the O(1) time attribution cross the seam on every
        # path, success or fault. A failed attempt's log tail is the one triage most needs,
        # and cr-011 hoisted the record out of `invoke` precisely so it survives the failure.
        del self.handled[command.request_id]
        return {
            "ok": True,
            "residency": residency_document,
            "observations": record.ring.rows(),
            "observation_caps": record.ring.caps(),
            "attribution": record.attribution.document(),
            "position": record.position,
            "outcome": terminal,
            "execution_observation": self._execution_observation(served_attention),
            # A pin is applied where its kernel served; one that fell back everywhere is not.
            "attention_applied": bool(pin)
            and attention._pin_parts(pin)[1]
            in (served_attention.totals() if served_attention else {}),
            "execution": {
                **self._execution(
                    requested,
                    served_attention,
                    started_us,
                    released_us,
                    attention.implementations(served_impls),
                    segments if len(segments) > 1 else (),
                ),
                "cuts": self.host.cuts.pop(
                    command.request_id, {"sites": [], "levels": {}, "oom_floor": 0}
                ),
                "loops": self.host.loops.pop(command.request_id, []),
                # When this reply was built, and (`_send_reply`) when it went on the seam: a
                # request's time after its observations splits into this process's and the host's.
                "reply_built_unix_ms": round(time.time() * 1000, 1),
            },
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
                adjustment_row(row) for row in (envelope.adjustments if envelope else ())
            ],
            "ignored": list(record.ignored),
            "oversize_result_bytes": oversize,
            "memory": memory,
            "metrics": {
                "handler_ms": round(handler_ms, 3),
                "device_lease_ms": round(lease_ms, 3),
                "d2h_wait_ms": round(d2h_ms, 3),
                "peak_vram_bytes": peak_vram,
                "peak_reserved_bytes": peak_reserved,
                "activation_peak_bytes": working_peak,
                "allocated_at_start_bytes": allocated_at_start,
                "working_peak_vram_bytes": working_peak,
                "shape_cell": shape_cell(prepared.features.values),
                "rss_at_end_bytes": _rss(),
                "started_unix": started_wall,
                "gpu_count": self.world if torch is not None else 0,
            },
        }

    # ------------------------------------------------------- the per-request attention pin

    def _attention_snapshot(self) -> attention.Snapshot:
        defaults = [executor.attention_defaults for executor in self.model_executors or [self]]
        return attention.Snapshot(
            tuple(choice for held in defaults for choice in held.choices),
            next((held.dense for held in defaults if held.dense is not None), None),
        )

    def _restore_attention(self) -> None:
        self._attention_snapshot().restore()

    def _prefix(self) -> str:
        """A qualified request names its model even when it is the only one; component-only pins
        still match through attention._matches."""
        return "+".join(sorted(self.models)) + "/" if self.models else ""

    def _attention_roots(self) -> dict[str, Any]:
        roots: dict[str, Any] = {}
        executors = self.model_executors or [self]
        for executor in executors:
            if executor.backend is None:
                continue
            for name, root in executor.backend.roots.items():
                roots[executor._prefix() + name] = root
        return roots

    @staticmethod
    def _emit_gemm(record: AuthorAttempt, member: Executor) -> None:
        """`gemm.selected`: per component and leaf provider, the kernel its installed leaves serve
        now (read once a run, never per call). `gemm.fallback`: each that serves below what its
        lane wanted, and why, also said as a warning line (a machine before TensorD keeps
        `gemm.*` drops the fact)."""

        def fallback(value: str, **fields: Any) -> None:
            record.emit("confession", "gemm.fallback", value, **fields)
            record.emit("log", "a GEMM served below its lane", "warn", kernel=value, **fields)

        prefix = member._prefix()
        counted = member.backend.gemm_sites if member.backend is not None else {}
        index = member.device_facts.index if member.device_facts is not None else 0
        sm = member.device_facts.sm if member.device_facts is not None else 0
        missing = int8_linear.unavailable()
        native = {"int8": int8_linear.OP, "fp8": fp8_linear.OP, "w4a8": w4a8_linear.OP}
        for (component, leaf), sites in sorted(counted.items()):
            kernel, form = serving(leaf, index), FORMATS[leaf]
            fields: dict[str, Any] = {"component": prefix + component, "sites": sites}
            record.emit("confession", "gemm.selected", kernel, format=form, leaf=leaf, **fields)
            wanted = native.get(form, kernel)
            reason = prebuilt.lost(form, sm) or (missing[0].reason if missing else "")
            reason = reason or (w4a8_linear.why(index) if form == "w4a8" else "")
            if reason and kernel != wanted:
                fallback(kernel, wanted=wanted, served=kernel, reason=reason, **fields)
        for group in member.refused_leaves:
            fallback(
                "torch.linear",
                component=prefix + group.component,
                wanted=FORMATS.get(group.leaf, group.lane),
                served={"bfloat16": "bf16", "float16": "fp16"}.get(group.dtype, group.dtype),
                reason=group.reason,
                sites=group.tensors,
            )

    def _attention_selection(self, observed: attention.Applied) -> attention.Applied:
        """Attach preparation reasons only to the choices this request actually holds.

        A scoped pin can replace one component while other components still fall back.
        A later successful compile does not erase a reason from an earlier run segment.
        """
        scope, pin = attention._pin_parts(observed.pin)
        fallbacks: list[execution_evidence.AttentionFallback] = []
        for executor in self.model_executors or [self]:
            if executor.attention is None:
                continue
            for row in executor.attention.fallbacks:
                component = executor._prefix() + row["component"]
                if pin and attention._matches(component, scope):
                    continue
                if row["selected"] in observed.hosts.get(component, {}):
                    fallbacks.append({**row, "component": component})
        return replace(observed, fallbacks=observed.fallbacks + tuple(fallbacks))

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

    def _install_ready_kernels(self) -> Frame | None:
        """THE EXECUTION PLAN'S INSTALLS (`Plan.when_ready`), each once its kernels are built
        for every GPU of its construction: at the end of preparation and between requests, on
        every GPU at the same boundary, never waiting for a compile. Rank 0 decides and
        installs first; each follower then installs what rank 0 did (`kernels`). One that
        raises on rank 0 is dropped and its construction serves as it was; one only a
        follower cannot install leaves a divergent group, refused."""
        for executor in self.model_executors or [self]:
            launches, _ = executor._built()
            if not launches:
                continue
            if executor.group is None:
                executor._install(launches)
                continue
            refused = executor._install_in_group(launches)
            if refused is not None:
                return refused
        return None

    def _built(self, sets: Collection[str] | None = None) -> tuple[dict[str, Launch], list[str]]:
        """This GPU's waiting installs (of `sets`, else all) whose kernels are built here, by
        set name, and the sets whose build ended without them, dropped for good."""
        launches: dict[str, Launch] = {}
        failed: list[str] = []
        for row in self._installs:
            name = row.name
            if (sets is not None and name not in sets) or name in launches or name in failed:
                continue
            try:
                launch = row.ready()
            except Exception as exc:  # an upgrade is never worth a failed run
                self._refuse_install(name, "kernels_failed", f"its build failed: {exc}")
                failed.append(name)
                continue
            if launch is not None:
                launches[name] = launch
        self._drop(failed)
        return launches, failed

    def _install(self, launches: Mapping[str, Launch]) -> list[str]:
        """Every waiting install of these built sets, now: the sets whose installs all ran. One
        that raises is dropped with its set's others, and the construction serves as it was.
        Graphs record the new paths again (`LANDED`)."""
        failed: set[str] = set()
        for row in self._installs:
            if row.name in launches and row.name not in failed:
                try:
                    row.install(launches[row.name])
                except Exception as exc:  # an upgrade is never worth a failed run
                    self._refuse_install(row.name, "install_refused", f"its install raised: {exc}")
                    failed.add(row.name)
        self._drop(launches)
        LANDED[0] += bool(launches)
        return sorted(set(launches) - failed)

    def _drop(self, sets: Collection[str]) -> None:
        self._installs = [row for row in self._installs if row.name not in sets]

    def _refuse_install(self, name: str, code: str, detail: str) -> None:
        """A deferred install of set `name` that will never run: said in the log and in its
        construction's plan reports under the set's name (where a package reports its deferral,
        so the report never stays "deferred")."""
        print(f"[kernels] {name}: {detail}"[:600], file=sys.stderr)
        for row in self._installs:
            if row.name == name:
                row.reports[name] = {"applied": False, "code": code, "reason": detail[:300]}

    def _install_in_group(self, launches: dict[str, Launch]) -> Frame | None:
        """Rank 0's half: each follower names which of these sets it has built (one whose
        build failed there is dropped here too), rank 0 installs the sets every GPU has, and
        then each follower installs what rank 0 installed."""
        group = self.group
        assert group is not None
        asked = functools.partial(
            Kernels, construction=self.construction, model_key=self._group_model_key
        )
        with group.hold(range(1, group.degree)):
            try:
                replies = group.call(executor_commands.encode(asked(sets=tuple(sorted(launches)))))
                for follower, reply in zip(group.followers, replies, strict=True):
                    if not reply.get("ok"):
                        raise GpuDivergence(follower.gpu, "kernels", str(reply.get("detail")))
                    missing = set(launches) - set(reply.get("built") or ())
                    for name in sorted(set(reply.get("failed") or ()) & missing):
                        self._refuse_install(
                            name,
                            "kernels_failed",
                            f"{follower.gpu} has no build of it; the group serves without it",
                        )
                        self._drop((name,))
                    launches = {n: launch for n, launch in launches.items() if n not in missing}
                installed = self._install(launches) if launches else []
                if not installed:
                    return None
                command = asked(sets=tuple(installed), install=True)
                replies = group.call(executor_commands.encode(command))
                for follower, reply in zip(group.followers, replies, strict=True):
                    if not reply.get("ok"):
                        raise GpuDivergence(
                            follower.gpu,
                            "kernels",
                            f"{_gpu(0)} installed {installed}: {reply.get('detail')}"[:800],
                        )
            except GroupRefusal as exc:
                return self._group_refusal(exc)
        return None

    def kernels(self, command: Kernels) -> Frame:
        """A follower's half of `_install_in_group`: which of rank 0's built sets this GPU has
        built (and which failed here, for good), or install those rank 0 installed."""
        target = self._follower_target(command.construction, command.model_key)
        if target is None:
            return {"ok": False, "code": "gpu_divergence", "detail": "no such construction"}
        launches, failed = target._built(command.sets)
        if not command.install:
            return {"ok": True, "rank": self.rank, "built": sorted(launches), "failed": failed}
        missing = sorted(set(command.sets) - set(target._install(launches)))
        if missing:
            detail = f"{_gpu(self.rank)} installed none of {missing}"
            return {"ok": False, "code": "gpu_divergence", "detail": detail, "rank": self.rank}
        return {"ok": True, "rank": self.rank}

    def _follower_target(self, construction: str, model_key: str | None) -> Executor | None:
        """The generation rank 0 names on this follower: its construction, or one model of a
        many-model construction."""
        container = self.constructions.get(construction)
        members = (container.model_executors or [container]) if container is not None else []
        target = next((m for m in members if m._group_model_key == model_key), None)
        return None if self.rank == 0 else target

    def _upgrade_attention(self, call: object = None) -> bool:
        """A RUN WITHOUT AN OVERRIDE NEVER WAITS FOR A COMPILE OR A SELF-CHECK: at each step or
        stage boundary of one, and between requests, a kernel the selection passed over while it
        opened (`kernel_ready`: built and checked off the request) is ready, so the construction
        selects again and serves it from the next step on (a run
        may mix kernels, owner ruling 2026-10-09); a group's followers take rank 0's kernels
        once no call holds them. Where a selection or a follower cannot, the prepared one
        stays and is not tried again. Sites every call reads change only while no other call is
        in a scope (`SCOPES`, entered as `call`). Whether any construction selected again."""
        due = [
            executor
            for executor in self.model_executors or [self]
            if executor.attention is not None
            and executor._reselect is not None
            and executor.device_facts is not None
            and executor.attention.waiting
            and attention.compiled(executor.attention.waiting, executor.device_facts)
        ]
        if not due:
            return False
        with SCOPES.held(object() if call is None else call):
            # Another call may have selected again while this one waited for its turn.
            due = [e for e in due if e.attention is not None and e.attention.waiting]
            return any([self._reselect_attention(executor) for executor in due])

    @staticmethod
    def _reselect_attention(executor: Executor) -> bool:
        applied, again = executor.attention, executor._reselect
        assert applied is not None and again is not None
        try:
            fresh = again()
        except Exception as exc:  # an upgrade is never worth a failed run
            print(f"[attention] selecting again refused: {exc}"[:600], file=sys.stderr, flush=True)
            executor.attention_defaults.restore()
            executor.attention = replace(applied, waiting=())
            return False
        executor.attention = fresh
        if executor.group is not None:
            with executor.group.hold(range(1, executor.group.degree)):
                if executor._agree_attention(poison=False):
                    # Every GPU back on the prepared selection; one that cannot take it back
                    # leaves a divergent group, refused.
                    executor.attention_defaults.restore()
                    executor.attention = applied
                    executor._agree_attention()
                    executor.attention = replace(applied, waiting=())
                    return False
        assert executor.backend is not None
        executor.attention_defaults = attention.snapshot(dict(executor.backend.roots))
        return True

    def _agree_attention(self, *, poison: bool = True) -> dict[str, Any] | None:
        """THE GROUP SERVES ONE SELECTION, WHOLE, THAT EVERY GPU HAS READY: each site's kernel
        and Sol's dense reference. Every follower installs rank 0's through the
        `attention.install` rank 0 selected with, never waiting for a build, and answers what
        it then serves. A kernel a follower lacks, rank 0 selects again without, and one still
        compiling there waits for a later boundary (`_upgrade_attention`), as kernel sets do
        (`_install_ready_kernels`). A follower that serves anything else refuses the group
        instead of serving a mixed selection. Not `poison`: an upgrade a follower did not take
        is refused for this group alone, which stays whole."""
        assert self.group is not None and self.backend is not None and self.attention is not None
        passed: dict[str, str] = {}
        later: set[str] = set()
        while True:
            mine = attention.chosen(self.backend.roots)
            try:
                self.group.broadcast(
                    executor_commands.encode(
                        Attention(
                            construction=self.construction,
                            model_key=self._group_model_key,
                            adopt=mine,
                            serving=request_path.in_flight(),
                            kernels=kernel_ready.agreed(),
                        )
                    )
                )
                replies = self.group.collect("attention")
                for follower, reply in zip(self.group.followers, replies, strict=True):
                    self.group.began.extend((follower.rank, k) for k in reply.get("began") or ())
            except GroupRefusal as exc:
                return self._group_refusal(exc)
            lacking = {
                (name, follower.gpu, code)
                for follower, reply in zip(self.group.followers, replies, strict=True)
                for name, code in dict(reply.get("lacking") or {}).items()
            }
            if not {name for name, _, _ in lacking} - set(passed):
                break
            passed.update({name: f"{gpu}: {code}" for name, gpu, code in lacking})
            later.update(n for n, _, code in lacking if code == "attention_kernel_compiling")
            assert self._reselect is not None
            try:
                self.attention = self._reselect(passed=passed)
            except attention.AttentionRefusal as exc:
                replies = [{"ok": False, "code": exc.code, "detail": str(exc)}] * len(replies)
                break
        if later:
            waiting = tuple(sorted({*self.attention.waiting, *later}))
            self.attention = replace(self.attention, waiting=waiting)
        for follower, reply in zip(self.group.followers, replies, strict=True):
            why = _attention_divergence(mine, reply)
            if why:
                if not poison:
                    return {"ok": False, "code": "gpu_divergence", "detail": why}
                return self._group_refusal(
                    GpuDivergence(follower.gpu, "attention", f"{_gpu(0)} serves otherwise: {why}")
                )
        return None

    def _adopt_attention(
        self, command: Attention, choice: executor_commands.Choice
    ) -> dict[str, Any]:
        """A follower's half of `_agree_attention`: rank 0's whole selection installed here as
        this construction's prepared one, and what this GPU then serves."""
        target = self._follower_target(command.construction, command.model_key)
        if target is None or target.backend is None or target.device_facts is None:
            return {"ok": False, "code": "gpu_divergence", "detail": "no such construction"}
        roots = target.backend.roots
        held = attention.chosen(roots)
        try:
            applied = attention.install(
                [], roots, degree=self.world, device=target.device_facts, choice=choice
            )
        except attention.AttentionRefusal as exc:
            lacking = exc.kernels if isinstance(exc, attention.Lacking) else {}
            detail = str(exc)[:900]
            return {
                "ok": False,
                "code": exc.code,
                "detail": detail,
                "rank": self.rank,
                "lacking": lacking,
            }
        took = attention.difference(choice, held)
        if took:
            print(
                f"[attention] {_gpu(self.rank)} took {_gpu(0)}'s selection: {took}",
                file=sys.stderr,
                flush=True,
            )
        target.attention_defaults = attention.snapshot(roots)
        if target.attention is not None:
            target.attention = replace(target.attention, hosts=applied.hosts)
        serves = attention.chosen(roots)
        return {"ok": True, "rank": self.rank, "choice": msgspec.to_builtins(serves)}

    def select_attention(self, command: Attention) -> dict[str, Any]:
        """A FOLLOWER restores its prepared selection and, given rank 0's `pin` for the next
        attempt (empty is the prepared one), holds it and answers what its sites now hold."""
        if command.adopt is not UNSET:
            began = kernel_ready.boundary(command.kernels)
            with request_path.serving() if command.serving else contextlib.nullcontext():
                return {**self._adopt_attention(command, command.adopt), "began": began}
        self._attention_hold.close()
        self._producer_hold = None
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
        # One attempt's producer states live across its mirrored calls; the restore or the
        # next selection closes them.
        store: dict[str, object] = {}
        self._attention_hold.callback(close_producer_states, store)
        self._producer_hold = (command.construction, store)
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

    def request_memory(self, command: executor_commands.RequestMemory) -> RankMemory:
        """A follower's part of rank 0's attempt: its own request, begun, planned and ended
        with rank 0's (`_ranks_request`), its record in the end's reply."""
        state, rank, request_id = self.host._rank_request, self.rank, command.request_id
        refused = functools.partial(RankMemory, ok=False, rank=rank, request_id=request_id)
        if not request_id or command.construction not in self.constructions:
            return refused(code="construction_absent")
        if command.phase == "begin":
            if state is not None:
                return refused(code="request_memory_busy")
            context = self.constructions[command.construction]._request(request_id)
            rows = context.__enter__()
            self.host._rank_request = _RankRequest(command.construction, request_id, context, rows)
        elif state is None or (state.id, state.construction) != (
            request_id,
            command.construction,
        ):
            return refused(code="request_memory_absent")
        elif command.phase == "plan":
            self._plan_request(request_id, command.steps)
        else:
            self.host._rank_request = None
            state.context.__exit__(None, None, None)
            own = state.rows[0] if state.rows else None
            return RankMemory(ok=True, rank=rank, request_id=request_id, memory=own)
        return RankMemory(ok=True, rank=rank, request_id=request_id)

    def _rank_memory_command(self, command: executor_commands.RequestMemory) -> list[RankMemory]:
        assert self.group is not None
        try:
            replies = [
                msgspec.convert(reply, RankMemory)
                for reply in self.group.call(executor_commands.encode(command))
            ]
        except msgspec.ValidationError as exc:
            self.poisoned = "group request memory reply was malformed"
            raise GroupRefusal(self.poisoned, code="group_broken") from exc
        if (
            {reply.rank for reply in replies} != set(range(1, self.world))
            or len(replies) != self.world - 1
            or any(not reply.ok or reply.request_id != command.request_id for reply in replies)
        ):
            self.poisoned = "group request memory did not settle on every rank"
            raise GroupRefusal(self.poisoned, code="group_broken")
        return replies

    @contextlib.contextmanager
    def _ranks_request(self, request_id: str) -> Iterator[list[GpuMemory]]:
        """Every follower's request beside this one; their records land in the list."""
        records: list[GpuMemory] = []
        if self.group is None:
            yield records
            return
        begun = False
        try:
            self._rank_memory_command(
                executor_commands.RequestMemory(
                    construction=self.construction, request_id=request_id, phase="begin"
                )
            )
            begun = True
            yield records
        finally:
            if not self.group.broken:
                command = executor_commands.RequestMemory(
                    construction=self.construction, request_id=request_id, phase="end"
                )
                if begun:
                    replies = self._rank_memory_command(command)
                    if self.host.memory is not None and any(r.memory is None for r in replies):
                        self.poisoned = "a GPU rank omitted its own request memory record"
                        raise GroupRefusal(self.poisoned, code="group_broken")
                    records.extend(reply.memory for reply in replies if reply.memory is not None)
                else:
                    # A partial begin still closes the ranks that entered; the original
                    # refusal, not an absent state on another rank, remains the failure.
                    self.group.call(executor_commands.encode(command))

    def _plan_request(self, request_id: str, steps: Sequence[str | Collection[str]]) -> None:
        """`ctx.plan`: the attempt's request takes the steps as its plan, on every GPU."""
        request = self.host.requests.get(request_id)
        if request is not None:
            request.plan(list(steps))
        if self.residency is not None:  # a step's components keep their holds (`using`)
            self.residency.plan(steps, request=request)
        if self.group is not None and self.rank == 0:
            normalized = tuple(
                (step,) if isinstance(step, str) else tuple(sorted(step)) for step in steps
            )
            self._rank_memory_command(
                executor_commands.RequestMemory(
                    construction=self.construction,
                    request_id=request_id,
                    phase="plan",
                    steps=normalized,
                )
            )

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
            # A broken group is rebuilt whole: no follower is left to restore, and a refusal
            # here would replace the failure that broke it.
            if not self.group.broken:
                self._followers_attention()

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
                    dense_reference=_dense((kernel,)),
                )

        if name == "attention.selected":
            for row in choices.fallbacks:
                record.emit("confession", "attention.fallback", row["selected"], **row)

    def _execution_observation(self, choices: attention.Applied | None = None) -> dict[str, Any]:
        # Observed generation facts; image and boot identity are added by the worker.
        # Optional kernel/lane fields stay absent until their execution is observed.
        return {
            "runtime_version": RUNTIME_VERSION,
            "accelerator": str(self.facts.get("gpu_name") or "CPU"),
            "cuda": str(self.facts.get("cuda_version") or ""),
            "driver": str(self.facts.get("device_driver") or ""),
            "execution_lane": "",  # no generic eager/compiled observation exists yet
            "execution_contract_digest": "",
            "kernel_symbol": ",".join(sorted(choices.totals())) if choices is not None else "",
        }

    def _install_call_proxies(self, rows: tuple[CallInterface, ...]) -> None:
        """Expose managed dependency imports before importing a private workflow.

        App registrations keep their original implementations. Only public module
        exports in this caller process become typed proxies; installed files and
        the independently executing callee remain ordinary uv installations.
        """
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
                proxy = companion.__dict__[export]
                # Callable lookup (including prefetch) uses the admitted public
                # module. Generated wire types stay private to the companion.
                proxy.__module__ = module
                imported.__dict__[export] = proxy

    def _call_broker(self, request_id: str, rows: tuple[CallInterface, ...]) -> Any:
        """This attempt's broker over `rows`, bound once per executor (a served package's
        attempts share its Start's rows)."""
        if not rows:
            return None
        if self._bound_calls is None or self._bound_calls[0] is not rows:
            self._bound_calls = (rows, self._call_bindings(rows))
        exchange = self._exchange(request_id)
        return _Broker(
            request_id,
            self._bound_calls[1],
            exchange,
            lambda: self._child_events(exchange),
            functools.partial(self._yielding, request_id),
        )

    def _yielding(self, request_id: str) -> contextlib.AbstractContextManager[None]:
        """A wait on a child call: the attempt's step pages are taken before the child's."""
        request = self.requests.get(request_id)
        return request.yielding() if request is not None else contextlib.nullcontext()

    def _call_bindings(self, rows: tuple[CallInterface, ...]) -> CallBindings:
        if len(rows) > 256:
            raise CapabilityError(
                "call interface inventory exceeds its bound", code="child_interface"
            )
        bindings: dict[tuple[str, str], _CallType] = {}
        own = effect_interfaces.bindings()
        for row in rows:
            module, export = row.module, row.export
            if row.native_effect:
                if module == effect_interfaces.MODULE and export in own:
                    bindings[(module, export)] = own[export]
            elif row.builtin:
                if row.builtin != "operations":
                    raise CapabilityError("unknown Runtime builtin", code="child_interface")
                builtin_operations.validate_binding(module, export)
                describe(builtin_operations_app)
                builtin = builtin_operations_app.get(export).surface
                bindings[(module, export)] = _serving(module, export, builtin)
            elif row.self_call:
                assert self.discovered is not None
                own_surface = next(
                    (
                        surface
                        for surface in self.discovered.surfaces
                        if surface.fn.__module__ == module and surface.fn.__name__ == export
                    ),
                    None,
                )
                if own_surface is None:
                    raise CapabilityError(
                        "self call is not a registered App export", code="child_interface"
                    )
                bindings[(module, export)] = _serving(module, export, own_surface)
            else:
                imported = importlib.import_module(module)
                function = getattr(imported, export, None)
                declared = _export(function) if callable(function) else None
                if declared is not None:
                    registration = Registration(
                        name=export,
                        fn=declared.implementation,
                        invocable=True,
                        memoize=declared.memoize,
                        model_defaults=declared.defaults,
                        model_components=declared.components,
                    )
                    binding = _serving(module, export, registration.surface)
                else:
                    generated_binding = getattr(imported, "__cozy_bindings__", {}).get(export)
                    if not isinstance(generated_binding, _CallType):
                        raise CapabilityError(
                            "installed export is not invocable", code="child_interface"
                        )
                    binding = generated_binding
                prior = bindings.get((module, export))
                if prior is not None and prior != binding:
                    raise CapabilityError("ambiguous managed export", code="child_interface")
                bindings[(module, export)] = binding
        for export in own:
            key = (effect_interfaces.MODULE, export)
            bindings.setdefault(key, effect_interfaces.unserved(export))
        return bindings

    def _store(self) -> Path:
        host = self.host
        if host.store is None:
            host.store = Path(
                _SEALED.get(child_env.TENSORFS_HOME) or weights_store.DEFAULT_HOME
            ).expanduser()
        return host.store

    def _checkpoint(
        self,
        ref: str,
        connection: weights_store.Connection | None = None,
        views: Mapping[str, View] | None = None,
    ) -> Checkpoint:
        """`ref` from the store, or from the bytes TensorD sent when it is a view (`views`)."""
        with store_refusals(f"opening {ref[:23]}"):
            view = (views or {}).get(ref)
            if view is not None:
                opened = weights_store.open_view(self._store(), view.header, view.own, connection)
                return Checkpoint(opened)
            return Checkpoint(weights_store.open_checkpoint(self._store(), ref, connection))

    def _child_events(self, exchange: Exchange) -> Nudges:
        """The worker's nudges for this attempt's calls: a socket, a pipe on Windows."""
        answer = exchange(ChildEvents(), Handoff)
        if not answer.ok:
            raise RuntimeFailure(
                answer.detail or "the machine refused this attempt's call events",
                code=answer.code or "child_events_refused",
            )
        if sys.platform == "win32":
            return winpipe.Pipe(answer.descriptor)
        nudges = socket.socket(fileno=answer.descriptor)
        nudges.setblocking(True)
        return nudges

    def _exchange(self, request_id: str) -> Exchange:
        """One call's durable exchange, naming it from whatever thread its code runs on."""

        def exchange[A: Answer](request: Request, into: type[A], /) -> A:
            return self._durable(request, into, request_id)

        return exchange

    def _durable[A: Answer](
        self, request: Request, into: type[A], request_id: str | None = None
    ) -> A:
        """One DURABLE mid-attempt exchange. Blocks until the worker answers.

        This is the seam's other half of `child.Executor.call(on_request=…)`. It is a
        separate lane from progress on purpose: a durable fact must not be sheddable, and
        the lossy lane's whole contract is that it may shed. It names the call it serves (by
        default the one this thread serves, "" outside every call).
        """
        body = encode_request(request)
        channel, exchange = self.channel, body["kind"]

        def receive(frame: Frame) -> None:
            handoff = issubclass(into, Handoff)
            if frame.get("descriptor") is True:
                if not handoff:
                    channel.close()
                    raise SeamError("seam_descriptor", "unexpected capability in control reply")
                frame["descriptor"] = channel.recv_descriptor()
            elif handoff and frame.get("ok") is True:
                raise SeamError("seam_descriptor", "successful reply omitted its capability")
            if frame.pop("descriptors", 0) or frame.get("lease") is True:
                channel.close()
                raise SeamError("seam_descriptor", "unexpected descriptors in a reply")

        frame = self.host.mux.ask(
            {"event": "request", "request_id": _CALL.get() if request_id is None else request_id}
            | body,
            receive=receive,
        )
        try:
            return tolerant.read(frame, into, ("progress", "observation"))[0]
        except msgspec.ValidationError as exc:
            raise SeamError("seam_answer_malformed", f"{exchange}: {exc}") from exc

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
        cached = self.schema_digests.get(surface.name)
        if cached is None:
            cached = self.schema_digests[surface.name] = canonical.digest(
                schema.render(surface.result_type, decoded_bounds=True)
            )
        return cached

    def _device(self) -> Device:
        """This executor's device as `author.device()` names it: its rank's GPU (`cuda:0`
        alone; the worker sealed this process to its lane, cr-066), MPS, or the CPU where none
        is present. No torch in this process means no device to name (cl-013)."""
        if self.torch is None:
            return Device()
        placed = self._rank_device(self.torch)
        return Device(placed.type, placed.index)

    def _queued_prefetch(self, current: str, active: PreparedRequest) -> QueuedPrefetch | None:
        """Snapshot one exact same-construction preparation; no device work under its lock."""
        from cozy_runtime.internal.residency import QueuedPrefetch

        if self.world != 1 or self.host.in_flight.count != 1 or self.residency is None:
            return None
        placements = (
            tuple(self.residency.rows.values())
            if isinstance(self.residency, _CompositeResidency)
            else (self.residency,)
        )
        if len(placements) != 1:
            return None  # Component names must identify one actual model slot.
        with self._prepared_lock:
            pending = [
                (request_id, prepared)
                for request_id, prepared in self.prepared.items()
                if request_id != current and not self.host.mux.cancelled(request_id)
            ]
            if len(pending) != 1:
                return None
            request_id, prepared = pending[0]
            if (
                not prepared.phases
                or prepared.lora != active.lora
                or prepared.prefetch_key != active.prefetch_key
            ):
                return None
            models = placements[0].models
            names = prepared.phases[0]
            if any(name not in models for name in names):
                return None
            targets = tuple(models[name] for name in names)

        def valid() -> bool:
            with self._prepared_lock:
                return (
                    self.prepared.get(request_id) is prepared
                    and not self.host.mux.cancelled(request_id)
                    and not self.host.mux.cancelled(current)
                    and self.host.in_flight.count == 1
                    and self.ready
                    and not self.poisoned
                )

        return QueuedPrefetch(request_id, targets, valid)

    @contextlib.contextmanager
    def _request(
        self, request_id: str, prepared: PreparedRequest | None = None
    ) -> Iterator[list[GpuMemory]]:
        """The request's residency scopes and pure cuts. Tensor ownership determines lifetime;
        its recorded allocator figures are process-wide, not exact per-request allocations."""
        host, rows = self.host, list[GpuMemory]()
        if host.memory is None or host.execution is None:
            yield rows
            return
        request: MemoryRequest | None = None
        cuda = host.memory.cuda
        device = host.memory.device.index if cuda else None
        peak: int | None = None
        try:
            with (
                host.memory.request(request_id, others=self._sibling_sets()) as request,
                host.execution.serving(request),
            ):
                host.requests[request_id] = request
                if prepared is not None:
                    request.queued = functools.partial(self._queued_prefetch, request_id, prepared)
                try:
                    yield rows
                finally:
                    del host.requests[request_id]
                    peak = accel.allocator_peak(host.memory, request_id)  # read while open
                    host.cuts[request_id] = msgspec.to_builtins(host.execution.cut_rows(request))
        finally:
            if request is not None:
                gpu = _gpu_numbers(self.world)[self.rank] if cuda else -1
                rows.append(host.execution.record(request, gpu, device, self._store(), peak))

    def _sibling_sets(self) -> list[Any]:
        """The residency sets of this process's other loaded constructions: their cached weights
        are not this construction's working set."""
        memory = self.host.memory
        if memory is None:
            return []
        modules = [
            model.module
            for key, child in self.host.constructions.items()
            if key != self.construction
            for member in child.model_executors or [child]
            if member.backend is not None
            for model in member.backend.models.values()
        ]
        return [memory.models[module] for module in modules if module in memory.models]

    def _quiescent(self) -> bool:
        """No author computation may still mutate GPU or output state when a terminal is
        recorded. The proof is the device's own: every stream is complete."""
        return accel.streams_idle(self.torch, self.device_kind)

    def _check_prepared_use(self, registration: Registration, payload: object) -> None:
        """Check current callable slots against the actual member, not historical aliases."""
        declarations = registration.model_prepared_use
        slots = {slot.parameter for slot in registration.surface.slots if not slot.weights}
        checked: set[str] = set()
        for member in self.model_executors or [self]:
            names = slots.intersection(member.models)
            if not names:
                continue
            declared = names.intersection(declarations)
            use = member._prepared_use
            if declared != (names if use is not None else set()):
                raise InvalidRequest(
                    "prepared-use declaration differs from the loaded construction",
                    code="prepared_use_binding",
                )
            for name in declared:
                field = declarations[name]
                if not isinstance(payload, Mapping) or field not in payload:
                    raise InvalidRequest(
                        "prepared-use field is absent", code="prepared_use_invalid"
                    )
                if prepared_use(payload[field]) != use:
                    raise InvalidRequest(
                        "changed prepared use needs a new construction",
                        code="prepared_use_binding",
                    )
            checked.update(declared)
        if checked != set(declarations):
            raise InvalidRequest("a prepared model slot is absent", code="prepared_use_binding")

    def map(self, command: Map) -> dict[str, Any]:
        """TensorD's warming: one loaded construction's models at `command.level`, on every GPU
        of a group."""
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        child = self.constructions.get(command.construction)
        if child is None:
            return self._refusal(
                "construction_not_loaded", f"{command.construction!r} is not loaded"
            )
        group = self.group if self.rank == 0 else None
        try:
            if group is not None:
                group.broadcast(executor_commands.encode(command))
            if self.memory is not None:
                if command.level == "ram":
                    # Map is ordered after every preceding invocation. Join its remaining
                    # consumer streams before retiring owners and returning allocator storage.
                    accel.synchronize(self.torch, self.device_kind)
                for member in child.model_executors or [child]:
                    if member.backend is not None:
                        if command.level == "ram" and member.residency is not None:
                            member.residency.release()
                        for model in member.backend.models.values():
                            if command.level == "ram":
                                for _, release in graphs.DECODERS:
                                    release(model.module)
                                graphs.retire(self.torch, model.module)
                            self.memory.load(model.module, command.level)
                if command.level == "ram":
                    self.memory.trim()
            for reply in group.collect("map") if group is not None else ():
                if not reply.get("ok"):
                    raise GroupRefusal(
                        f"a follower refused to map: {reply.get('code')}: "
                        f"{reply.get('detail', '')}"[:600],
                        code="group_broken",
                    )
        except GroupRefusal as exc:
            return self._group_refusal(exc, closing=False)
        return {"ok": True}

    def _active_residency(self) -> dict[str, Any]:
        child = self.constructions.get(self.active)
        if child is None or child.residency is None:
            return {}
        document: dict[str, Any] = child.residency.document()
        return document

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

    def _device_figures(self) -> dict[str, int]:
        """The card as its allocator and driver see it now. A process with no card (a CPU
        executor, a gloo rank) has none; readers take absent as unknown."""
        if self.torch is None or self.device_kind == "mps":
            return {}
        if not accel.present(self.torch, self.device_kind):
            return {}
        memory = accel.allocation(self.torch, self.device_kind, self.host.memory)
        return {
            "device_free_bytes": memory["driver_free_bytes"],
            "device_total_bytes": memory["driver_total_bytes"],
            "allocator_bytes": memory["allocated_bytes"],
            "reserved_bytes": memory["reserved_bytes"],
        }

    def unsettled(self, target: Executor, exc: Exception, request_id: str) -> dict[str, Any]:
        """An attempt that raised outside its handler. A CUDA fault is sticky, so the tail's
        first sync raises it again: the process cannot serve another attempt, and the reply
        names the handler's own failure (else this one) and the stage it happened in. Memory
        is capacity: it ends the attempt, never the process, and its facts go with it."""
        later = classify(exc)
        first, stage, record = target.handled.pop(request_id, None) or (later, "", None)
        first = first if first.terminal == "failed" else later
        where = f" in {stage}" if stage else ""
        message = first.message + where + ("" if first is later else f"; then {later.message}")
        capacity, quiet = later.code in CAPACITY_CODES, False
        if capacity:
            with contextlib.suppress(Exception):  # a sticky fault raises here: then it is one
                # Calls still in flight own the work the streams still hold.
                quiet = (
                    self._quiescent()
                    if self.torch is not None and not self.in_flight.count
                    else True
                )
        if not quiet:
            self.poisoned = target.poisoned = (
                self.poisoned or target.poisoned or f"{first.code}{where}: {first.message}"[:300]
            )
        return {
            "ok": True,
            "outcome": {
                "terminal": "failed",
                "origin": first.origin,
                "code": first.code,
                "message": ends(message, 2048),
                "fields": list(first.fields),
                "traceback": first.traceback or later.traceback,
            },
            "observations": record.ring.rows() if record is not None else [],
            "quiescent": quiet,
            "poisoned": self.poisoned,
            "outputs": [],
            "frames": [],
        }

    def _settled(self, reply: dict[str, Any]) -> dict[str, Any]:
        """An attempt's reply carries the after-state a `Probe` would read next, so the
        worker closes its ledger without another round trip; residency and RSS are already
        in the reply."""
        reply["after"] = {**self._device_figures(), "processes": self.process_incarnations()}
        return reply

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
        return {
            "ok": True,
            "torch": True,
            **self._device_figures(),
            "residency": self._active_residency(),
            "constructions": self.residency_document()["constructions"],
            "rss_bytes": _rss(),
            "attempts": self.attempts,
            "processes": processes,
            "ready": self.ready,
            "poisoned": self.poisoned,
            **_env_document(),
        }


#: Telemetry the worker reads as optional: what a reply loses when an author's key reads as
#: a credential (`tel.log(..., token=1)`) and the seam refuses the frame.
_TELEMETRY = ("observations", "execution", "attribution")


def _send_reply(channel: Channel, reply: dict[str, Any]) -> None:
    """Put ONE reply on the seam. Telemetry never costs a reply its outcome."""
    if isinstance(execution := reply.get("execution"), dict):
        execution["reply_sent_unix_ms"] = round(time.time() * 1000, 1)
    try:
        channel.send(reply)
    except SeamError as exc:
        if exc.code != "credential_at_seam" or not any(key in reply for key in _TELEMETRY):
            raise
        dropped = {f"dropped_{key}_for_credential": 1 for key in _TELEMETRY if key in reply}
        for key in _TELEMETRY:
            reply.pop(key, None)
        reply["observation_caps"] = {**(reply.get("observation_caps") or {}), **dropped}
        channel.send(reply)


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

    Returns `(result_ref, outputs, oversize_bytes)`. The BYTES land in the attempt spool the
    worker brokers and only their digest and length ride the reply. `oversize_bytes` is
    non-zero when the canonical result exceeds the 4 MiB inline door, which cr-007 REFUSES
    rather than papering over with a receipt for bytes nobody wrote.
    """
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
    data = canonical.write_application(document)
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
        "join",
        "form",
        "load",
        "activate",
        "unload",
        "map",
        "run",
        "attention",
        "kernels",
        "request_memory",
        "vacate",
        "restore",
        "budget",
        "probe",
        "shutdown",
    }
)


def answer(executor: Executor, frame: Frame) -> Frame | None:
    """One command's reply as this GPU's process answers it; None is `shutdown`."""
    name = str(frame.get("cmd", ""))
    if executor.rank > 0 and name not in _FOLLOWER_COMMANDS:
        return {
            "ok": False,
            "code": "follower_command_unsupported",
            "detail": f"{_gpu(executor.rank)} answers {sorted(_FOLLOWER_COMMANDS)}, not {name!r}",
        }
    if name == "run":
        # A follower's call serves rank 0's request, with the kernels the group serves from it.
        with request_path.serving():
            began = kernel_ready.boundary(frame.get("kernels") or ())
            reply = executor.run(frame)
        return {**reply, "kernels": kernel_ready.pending(), "began": began}
    if name not in executor_commands.NAMES:
        return {"ok": False, "code": "unknown_command", "detail": name}
    command = executor_commands.decode(frame)
    if isinstance(command, Shutdown):
        return None
    reply = executor.handle(command)
    if name == "load" and executor.rank:  # what it opened loading: the group may serve it at once
        reply["kernels"] = kernel_ready.pending()
    return reply


#: The commands whose refusals carry the raising traceback to the worker.
_TRACED_COMMANDS = frozenset({"start", "load", "join", "form"})

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
    faulthandler.enable(all_threads=True)
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
    if not proctree.PARENT_DEATH_SURVIVES_EXEC:
        # The trampoline's exec ended its guard (darwin): this process watches its parent.
        expected = os.getppid()
        proctree.arm_parent_death()
        if os.getppid() != expected:
            return int(Exit.structural)

    _capture_seal()
    _remember_distributions()
    declared_recipes.install(run_recipe)
    kernel_cache.configure(kernel_cache.from_sealed(_SEALED))
    # cr-042: the executor's ONE network fence, installed BEFORE the seam dials so the
    # AF_UNIX exemption is exercised on every boot rather than asserted. It refuses
    # Python-level egress from package code; it is not a kernel bound and
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
    return serve(executor, channel)


def _compiling(channel: Channel, state: kernel_compile.State) -> None:
    """A wait on a compiling kernel, told to the machine as the waiting call's stage."""
    _waiting(channel, f"waiting for {state.kernel} compile", state.progress)


def _waiting(channel: Channel, what: str, fraction: float | None = None) -> None:
    """A wait, told to the machine as the waiting call's stage ("": the executor's, outside
    every call)."""
    with contextlib.suppress(SeamError):
        channel.send(
            {
                "event": "progress",
                "request_id": _CALL.get(),
                "kind": "progress",
                "stage": what,
                "stage_fraction": fraction,
                "advance": 0,
            }
        )


#: Commands of one call: served on the executor's pool, side by side with other calls.
_CALL_COMMANDS = frozenset({"prepare_request", "invoke"})


class _Order:
    """Commands in the order they came: a call's run side by side, and a process-wide one
    alone, after every command before it and before every command after it. One waiting for
    its turn is told what it waits on."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        #: ticket -> whether it runs alone and what it is, until its command is answered
        self._open: dict[int, tuple[bool, str]] = {}
        self._next = 0
        #: commands answered so far: an idle period follows each (`idle`)
        self.answered = 0

    def ticket(self, alone: bool, what: str) -> int:
        with self._changed:
            self._next += 1
            self._open[self._next] = (alone, what)
            return self._next

    def first(self, ticket: int) -> bool:
        with self._changed:
            return min(self._open) == ticket

    @contextlib.contextmanager
    def turn(self, ticket: int) -> Iterator[None]:
        with self._changed:
            alone = self._open[ticket][0]
            request_path.wait(
                self._changed,
                lambda: (
                    min(self._open) == ticket
                    if alone
                    else not any(a for t, (a, _) in self._open.items() if t < ticket)
                ),
                lambda: next(
                    (what for t, (a, what) in self._open.items() if t < ticket and (alone or a)),
                    "",
                ),
            )
        try:
            yield
        finally:
            with self._changed:
                del self._open[ticket]
                self.answered += 1
                self._changed.notify_all()

    @contextlib.contextmanager
    def idle(self, answered: int, ready: Callable[[], bool]) -> Iterator[int]:
        """Once no command is open, one was answered since `answered` and `ready()`: the order
        held as a process-wide command holds it, so a command arriving waits for the holder to
        give it back (`waiting`). Yields the answered count."""
        with self._changed:
            self._changed.wait_for(lambda: not self._open and self.answered != answered and ready())
            self._next += 1
            ticket, answered = self._next, self.answered
            self._open[ticket] = (True, "graph capture")
        try:
            yield answered
        finally:
            with self._changed:
                del self._open[ticket]
                self._changed.notify_all()

    def waiting(self) -> bool:
        """A command arrived beside the idle holder."""
        return len(self._open) > 1


def _idle(executor: Executor, order: _Order) -> None:
    """Graphs record off the request path (`graphs.capture_idle`): after a command is answered,
    while no other is open. One arriving makes the capture yield at its next region boundary. A
    device fault there poisons the executor, as one in a request does (the machine ends it)."""
    answered = 0
    while True:
        with order.idle(answered, lambda: bool(graphs.IDLE.jobs)) as answered:
            try:
                graphs.capture_idle(order.waiting)
            except Exception as exc:
                traceback.print_exc()
                if fault(exc) in (Fault.STICKY, Fault.DEVICE):
                    executor.poisoned = executor.poisoned or f"device fault at idle: {exc}"[:300]
                    return


def serve(executor: Executor, channel: Channel) -> int:
    """Answer commands until the machine shuts this executor down or its seam ends. This thread
    keeps reading, so a `bearer` or `cancel` sent during a long command (a Load streaming its
    weights) is taken at once: after the device Start every command runs on the pool in its
    turn. Before it the process stays single-threaded and forkable: Start and Fork run here,
    as does any other command that is first and not a call's."""
    kernel_compile.waiting = functools.partial(_compiling, channel)
    request_path.told = functools.partial(_waiting, channel)
    # Runtime-owned workers, none before the first is needed.
    pool = ThreadPoolExecutor(CALLS_IN_FLIGHT, "call")
    order = _Order()
    idle = False
    while True:
        if not idle and executor.started:
            # After the device Start: a live thread in an import-only executor refuses its forks.
            threading.Thread(
                target=_idle, args=(executor, order), name="graphs-idle", daemon=True
            ).start()
            idle = True
        try:
            frame = executor.mux.command()
        except SeamError:
            return int(Exit.unavailable)
        if frame is None:
            return int(Exit.ok)
        name = str(frame.get("cmd", ""))
        if executor.rank > 0:  # a follower answers its GPU 0, one command at a time
            reply = _reply(executor, frame)
            if reply is None:
                return _shut(executor, channel)
            _send_reply(channel, reply)
            continue
        call = name in _CALL_COMMANDS
        ticket = order.ticket(not call, f"run {frame.get('request_id', '')}" if call else name)
        if executor.started or call or not order.first(ticket):
            pool.submit(contextvars.Context().run, _serve, executor, frame, order.turn(ticket))
            continue
        with order.turn(ticket):
            reply = _reply(executor, frame)
        if reply is None:
            return _shut(executor, channel)
        _send_reply(channel, reply)


def _shut(executor: Executor, channel: Channel) -> int:
    if executor.group is not None:
        executor.group.close()
    channel.send({"reply": "shutdown", "ok": True})
    return int(Exit.ok)


def _serve(executor: Executor, frame: Frame, turn: contextlib.AbstractContextManager[None]) -> None:
    """One command on a pool worker, in its turn. A BaseException, or a reply the seam would
    not take, ends the process, as either did when commands ran on the main thread; so does a
    `shutdown` that waited here for the calls before it, once answered."""
    try:
        call = (
            str(frame.get("request_id", "")) if frame.get("cmd") in _CALL_COMMANDS | {"map"} else ""
        )
        _CALL.set(call)  # its wait for its turn is told as its own
        with turn:
            reply = _reply(executor, frame)
            if reply is None:
                code = _shut(executor, executor.channel)
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(code)
            if frame.get("cmd") in {"invoke", "map"} and call:
                executor.mux.forget(call)
            _send_reply(executor.channel, {**reply, "request_id": call} if call else reply)
    except BaseException as exc:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 1)


def _reply(executor: Executor, frame: Frame) -> Frame | None:
    """One command's reply, its faults included; None is `shutdown`."""
    name = str(frame.get("cmd", ""))
    if executor.group is not None and executor.group.broken and not executor.poisoned:
        # A FOLLOWER DIED BETWEEN COMMANDS (cr-068): the group is one executor, so this
        # process is poisoned before it answers anything, and the worker's next
        # command reads `poisoned_generation` and rebuilds the whole cgroup.
        executor.poisoned = f"group: {executor.group.broken}"[:200]
    try:
        reply = answer(executor, frame)
        if reply is None:
            return None
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
    return reply


if __name__ == "__main__":
    raise SystemExit(main())
