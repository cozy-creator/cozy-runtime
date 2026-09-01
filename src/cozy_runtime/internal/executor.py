"""The disposable device executor: Python, CUDA, bound instances, one GPU attempt.

This is the ONLY process in the worker that imports torch or touches a device. It is
disposable in the strong sense: `kill -9` at any instant is a supported input, and the
worker converges to a typed terminal and a fresh incarnation from what it recorded.
Nothing durable lives here — no records, no credential, no monotone wire counter.

Its whole surface is the seam (`internal/seam.py`), and its whole vocabulary is:

    hello     process facts: pid, parent, no_new_privs, oom_score_adj, the SEALED env
    describe_installed  derive one installed package descriptor into brokered scratch
    prepare   construct + fill one binding; report the construction digest and ready facts
    prepare_request  resolve ONE request (decode, normalize, preflight) and retain it, so
              the worker prices a plan against the record the handler will actually run
    invoke    run ONE PREPARED attempt through the author invocation kernel and report its facts
    run_job   run ONE @app.job to completion through the SAME kernel, with job services
    probe     device/host memory facts, for the worker's ledger reconciliation
    shutdown  cooperative exit

Cancellation is attempt-keyed: `ctx.cancelled` reads the marker the worker publishes for
this exact request. A stale cancellation therefore cannot cross into the next attempt.
The worker owns the grace clock and forcefully reclaims the whole executor scope if author
code does not cooperate.

The env seal is proved, not trusted: the sealed values are snapshotted BEFORE torch is
imported, and every reply after CUDA init carries `env_intact`. A late imposition (the
allocator already read its environment) is a mismatch there rather than a silent no-op.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cozy_runtime.author._assets import GrantedInput
from cozy_runtime.author._context import Device
from cozy_runtime.author._errors import CapabilityError, Outcome, classify
from cozy_runtime.author._invoke import (
    Invocation,
    PreparedRequest,
    attempt,
    prepare,
    run_prepared,
)
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author._services import ProgressFrame
from cozy_runtime.internal import accel, canonical, child_env, descriptor, sandbox
from cozy_runtime.internal.config import seal_snapshot
from cozy_runtime.internal.discovery import discover, discover_distribution, discover_installed
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel, SeamError, connect
from cozy_runtime.protocol import documents

#: The inline-result door (worker-protocol/01 `ResultEnvelope`): a small typed result rides
#: the terminal envelope and skips the file-store round trip. Larger than this REFUSES at
#: cr-007 — the blob-receipt branch needs a real write-and-receipt transaction, and minting
#: a receipt for bytes nobody wrote is the fabrication class this runtime exists to prevent.
INLINE_RESULT_MAX = 4 * 1024 * 1024

#: How many PREPARED requests this incarnation holds at once. A serving lane admits and
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


def _env_document() -> dict[str, Any]:
    """One coherent post-CUDA seal verdict for an executor reply."""
    changed = _env_changes()
    return {"env_intact": not changed, "env_changed": list(changed)}


def _cancelled(root: Path, request_id: str) -> bool:
    """Whether the worker's one cooperative marker names this exact attempt."""
    try:
        return (root / "executor.cancel").read_text() == request_id
    except (OSError, UnicodeError):
        return False


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


def _granted_inputs(command: Mapping[str, Any]) -> dict[str, GrantedInput]:
    """Verified worker-spooled inputs, projected identically into either execution lane."""
    return {
        input_id: GrantedInput(
            input_id=input_id,
            local=Path(row["local"]),
            media_type=str(row["media_type"]),
            digest=str(row["digest"]),
            length=int(row["length"]),
            order=int(row.get("order", 0)),
        )
        for input_id, row in (command.get("inputs") or {}).items()
    }


def residence_ceiling(*, destinations: int, allocatable: int) -> int:
    """The construction's residence ceiling, DERIVED from the card (cl-027 / cr-025).

    Nothing authors this: it is the measured allocatable capacity, capped at the artifact's
    own complete destination bytes so a card with room reports the artifact's size rather
    than its own. Whatever it comes to is what the generation's digest carries. Zero means
    the card has no room at all, which is a typed shortfall at the caller.
    """
    return min(int(destinations), max(int(allocatable), 0))


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

    def open_attempt(self, placement: str, headroom_bytes: int) -> None:
        for row in self.rows.values():
            row.open_attempt(placement, headroom_bytes)

    def document(self) -> dict[str, Any]:
        documents = {name: row.document() for name, row in sorted(self.rows.items())}
        resident = {
            f"{name}/{component}": int(size)
            for name, document in documents.items()
            for component, size in document.get("resident", {}).items()
        }
        evicted = {
            f"{name}/{component}": int(size)
            for name, document in documents.items()
            for component, size in document.get("evicted", {}).items()
        }
        return {
            "placement": next(
                (document.get("placement", "") for document in documents.values()), ""
            ),
            "headroom_bytes": max(
                (int(document.get("headroom_bytes", 0)) for document in documents.values()),
                default=0,
            ),
            "admissions": sum(int(row.get("admissions", 0)) for row in documents.values()),
            "stages": sum(int(row.get("stages", 0)) for row in documents.values()),
            "evictions": sum(int(row.get("evictions", 0)) for row in documents.values()),
            "attempt_stages": sum(int(row.get("attempt_stages", 0)) for row in documents.values()),
            "attempt_evictions": sum(
                int(row.get("attempt_evictions", 0)) for row in documents.values()
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


class Executor:
    """One incarnation's mutable state. Everything here dies with the process."""

    def __init__(self, channel: Channel, root: Path, fill_forensics_dir: str = "") -> None:
        self.channel = channel
        self.root = root
        #: cr-038: the fill plane's capture directory, taken from the sealed environment
        #: snapshot at process entry and passed down — never read ambiently.
        self.fill_forensics_dir = fill_forensics_dir
        self.torch: Any = None
        #: The BACKEND this incarnation serves. Every device operation below goes through
        #: `accel` with it, so the whole process holds exactly one backend conditional and
        #: it is this line (#447). A weightless executor never reaches a device at all.
        self.device_kind = accel.host_backend_family()
        self.registry: Any = None
        self.backend: Any = None
        self.residency: Any = None
        self.model_executors: list[Executor] = []
        self.models: dict[str, Any] = {}
        self.surface_names: dict[str, Any] = {}
        self.discovered: Any = None
        self.ready = False
        self._prepare_consumed = False
        self.poisoned = ""
        self.logical_model_digest = ""
        self.constructed_model_digest = ""
        self.facts: dict[str, Any] = {}
        self.schema_digests: dict[str, str] = {}
        #: request_id -> the record `prepare_request` resolved, run by `invoke`
        self.prepared: dict[str, PreparedRequest] = {}
        self.attempts = 0
        #: the DURABLE exchange counter (cr-009). Worker-answered, never sheddable.
        self._exchange = 0

    # ------------------------------------------------------------------ prepare

    def describe_installed(self, command: dict[str, Any]) -> dict[str, Any]:
        """Import one installed package and write its descriptor to brokered scratch."""

        distribution = str(command.get("distribution", ""))
        output = Path(str(command.get("output", "")))
        if not distribution or not output.is_absolute():
            return {"ok": False, "code": "package_prepare_path_invalid", "detail": "paths"}
        discovered = discover_distribution(distribution)
        body = descriptor.build(discovered)
        raw = descriptor.canonical_bytes(body)
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
            "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "length": len(raw),
        }

    def prepare(self, command: dict[str, Any]) -> dict[str, Any]:
        """Consume this incarnation's one construction attempt.

        The latch moves before the first operation deliberately. A preparation that raises
        after reserving or allocating state is just as terminal for this incarnation as a
        successful fill: no later command may enter the construction body and stack another
        generation on whatever the failed attempt retained.
        """
        if self._prepare_consumed:
            return {
                "ok": False,
                "code": "generation_already_prepared",
                "detail": "this executor already consumed its one preparation attempt; "
                "the worker must reuse a successful generation or replace the process "
                "before preparing again",
            }
        self._prepare_consumed = True
        try:
            reply = self._prepare_first(command)
        except Exception as exc:
            self.poisoned = self.poisoned or f"prepare raised: {type(exc).__name__}"
            raise
        if not reply.get("ok"):
            self.poisoned = self.poisoned or f"prepare refused: {reply.get('code', 'unknown')}"
        return reply

    def _prepare_first(self, command: dict[str, Any]) -> dict[str, Any]:
        """Construct and fill the incarnation's one binding."""
        if self.poisoned:
            return {
                "ok": False,
                "code": "poisoned_generation",
                "detail": f"this executor is poisoned ({self.poisoned}); a poisoned "
                "generation has no recovery edge to Ready — the worker replaces it",
            }
        if "models" in command:
            return self._prepare_many(command)
        binding = command["binding"]
        budgets = command["budgets"]
        started = time.perf_counter()
        if not binding.get("model_class"):
            return self._prepare_weightless(binding, started)

        changed = _env_changes()
        if changed:
            return {
                "ok": False,
                "code": "env_seal_broken",
                "detail": "the executor environment changed before Torch import "
                f"({', '.join(changed)}); no CUDA initialization was attempted",
            }

        import torch  # the ONE torch import in the runtime, and it is in THIS process

        self.torch = torch
        from cozy_runtime.author import ModelRegistry
        from cozy_runtime.author._loader import Artifact, Config
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
            topology_of,
        )
        from cozy_runtime.internal.planfacts import PlanFacts
        from cozy_runtime.internal.probe import qualify
        from cozy_runtime.internal.residency import ComponentResidency
        from cozy_runtime.internal.resolution import (
            ConstructionFacts,
            PlanRefusal,
            Variant,
            resolve,
        )

        if accel.initialized(torch, self.device_kind):
            return {
                "ok": False,
                "code": "cuda_initialized_before_runtime",
                "detail": "Torch CUDA was already initialized before Runtime's explicit "
                "allocator boundary; package execution is refused",
            }
        accel.initialize(torch, self.device_kind)
        accel.reset_peak(torch, self.device_kind)
        cuda_ms = (time.perf_counter() - started) * 1000
        changed = _env_changes()
        if changed:
            return {
                "ok": False,
                "code": "env_seal_broken",
                "detail": "the process environment changed between the seal and CUDA init; "
                "the allocator read its environment once and a late imposition is a silent "
                f"no-op, so this is a refusal rather than a warning ({', '.join(changed)})",
            }

        if self.discovered is None:
            self._discover_provisioned(binding)

        # The store IS the artifact identity: one snapshot digest, and TensorFS answers with
        # the header, the plan and the verified lease. The interim path needed a store root
        # AND a pre-parsed read-plan file beside it; tfs-007 deleted the second one.
        # The component SET, in the contract's own declared order. A single-component
        # binding is the same code with a one-element set: cr-008b did not add a second
        # path, it widened the one that existed. `snapshots` names ONE snapshot per
        # component, because a TensorFS read plan is complete over its header and a
        # partial read of a whole-pipeline snapshot has no representation (tfs-007).
        names: list[str] = list(binding.get("components") or [binding["component"]])
        per_component: dict[str, str] = dict(binding.get("snapshots") or {})

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
        # the executor's selected device; a repository lane is never a hardware claim.
        device_index = int(binding.get("device_index") or 0)
        device_facts = measure_device(torch, device_index)
        hardware_variant = str(binding.get("variant") or "")
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
            store=binding["store"],
            snapshot=binding["snapshot"],
            snapshots=dict(per_component),
            reference=True,
        )
        vmap = reference.snapshot_map()
        checkpoints = {
            name: Checkpoint(reference.store, vmap.get(name, reference.snapshot)) for name in names
        }
        rows = [r for name in names for r in checkpoints[name].rows(name)]
        # THE SELECTED DEVICE, not card zero (#549.9). `device_index` is the binding's, and
        # NVML is queried with it — a multi-GPU pod used to mint every record against GPU 0's
        # power cap and MIG profile whatever card the fill was about to use.
        runtime_identity = measure_runtime(torch, binding["release"])
        providers = launch_providers()
        qualified = qualify(
            torch,
            providers,
            device_facts,
            sorted({dtype_name(r.dtype) for r in rows}),
            release=binding["release"],
            runtime=runtime_identity,
        )
        objective = str(binding.get("objective") or "latency")
        model_cls = getattr(self.discovered.module, binding["model_class"])
        # AN ARTIFACT THAT DECLARES NO CONSTRUCTION CONFIG GETS THE EMPTY DOCUMENT, and `{}`
        # is that artifact's honest config identity rather than a missing one — it hashes
        # into the plan like any other, and the construction layers already say what it
        # means ("an artifact that carries none leaves every field at the release's own
        # value"). Reading `Path("")` instead turned a legitimate config-less checkpoint
        # into a file-not-found inside prepare.
        config_document = (
            json.loads(Path(binding["config"]).read_text()) if binding.get("config") else {}
        )
        # The construction half of plan identity, computed from the HEADER and the CONFIG —
        # both available before a line of the model is built. They are the same facts the
        # post-construction record carries, one plane earlier, which is the only place a plan
        # that has to be resolved BEFORE construction can read them.
        construction = ConstructionFacts(
            release=str(binding["release"]),
            model_class=f"{model_cls.__module__}:{model_cls.__qualname__}",
            config_digest=canonical.digest(config_document),
            topology_digest=canonical.digest(
                {
                    key: [list(spec.shape), spec.dtype, spec.encoding]
                    for key, spec in sorted(topology_of(rows).items())
                }
            ),
            components=tuple(names),
            custody=str(binding.get("custody", "canonical")),
            strict_keys=bool(binding.get("strict_keys", False)),
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
                capabilities=qualified[0],
                device=device_facts,
                runtime=runtime_identity,
                construction=construction,
                facts=facts,
                objective=objective,
                placement=str(binding.get("placement") or "all_resident"),
                encoded_leaves=consent,
                steps_basis=int(binding.get("steps_basis") or 20),
                dtype_name=dtype_name,
            )
        except PlanRefusal as exc:
            self.poisoned = f"plan refused: {exc.code}"
            return {
                "ok": False,
                "code": exc.code,
                "detail": str(exc).splitlines()[0][:900],
                "plan": {"walk": exc.walk, "objective": binding.get("objective")},
            }

        # The LOAD leg's liveness feed. Integral-MiB positions on the lossy lane, so the
        # worker can report measured progress for a multi-GB load without inventing a timeout.
        # `started` is published at position 0 before the first byte moves, which is what
        # makes a pre-first-unit wedge render at all.
        def position(mib: int) -> None:
            with contextlib.suppress(SeamError):
                self.channel.send(
                    {
                        "event": "progress",
                        "request_id": f"prepare:{binding['component']}",
                        "kind": "load",
                        "name": binding["component"],
                        "value": None,
                        "position": mib,
                    }
                )

        position(0)
        backend = StreamingFillBackend(
            checkpoints,
            rows,
            plan=plan,
            on_position=position,
            window_bytes=binding.get("window_bytes", 4 << 20),
            slots=binding.get("slots", 16),
            readers=binding.get("readers", 16),
            inflight=binding.get("inflight", 8),
            release=binding["release"],
            # The suite already ran, above, to resolve the plan. Running it a second time
            # here would re-measure a device fact that has not changed.
            qualified=qualified,
            forensics_dir=self.fill_forensics_dir,
        )
        # THE ARITHMETIC, BEFORE THE FIRST BYTE (#574b). The provisioned binding derives its
        # destination residency from the MCC's complete logical+derived tensor census. The
        # header independently prices logical rows and the selected route's native/decode
        # overhead. Derived buffers remain live beside that overhead, so the complete peak
        # is additive and the larger destination census wins if the two have not yet met.
        # This refusal runs before allocation instead of discovering an OOM mid-fill.
        declared = int(budgets["declared_weight_bytes"])
        envelope = complete_device_envelope(backend.device_envelope(), declared)
        memory = accel.allocation(torch, self.device_kind)
        allocatable = memory["driver_free_bytes"] + max(
            memory["reserved_bytes"] - memory["allocated_bytes"], 0
        )
        # THE RESIDENCE CEILING (cl-027 / cr-025). DERIVED here from measured capacity and
        # the artifact's complete destination bytes; the APPLIED value is what the
        # generation's digest carries. A staging generation re-reads the store after
        # construction, so its GC-safe hold must outlive the first commit; that is
        # `hold_lease`.
        destinations = max(int(envelope["destination_bytes"]), declared)
        resident_budget = residence_ceiling(destinations=destinations, allocatable=int(allocatable))
        if resident_budget <= 0:
            self.poisoned = "fill refused: device_shortfall"
            return {
                "ok": False,
                "code": "device_shortfall",
                "detail": (
                    f"{allocatable} B allocatable leaves no residence ceiling — nothing of "
                    "this construction can stay resident, so there is no plan to derive"
                ),
                "shortfall": {
                    "resource": "vram",
                    "scope": "component_generation",
                    "needed_bytes": destinations,
                    "available_bytes": int(allocatable),
                    "evidence_class": "measured",
                },
            }
        backend.resident_budget = resident_budget
        backend.hold_lease = 0 < resident_budget < destinations
        # ADMISSION prices the RESIDENT set, not every destination: a construction whose
        # total exceeds the card but whose ceiling fits prepares and stages (cr-025 done-when).
        resident_destinations = (
            min(destinations, resident_budget) if resident_budget else destinations
        )
        required = (
            resident_destinations
            + int(envelope["native_bytes"])
            + int(envelope["decode_peak_bytes"])
        )
        envelope = {
            **envelope,
            "resident_destination_bytes": resident_destinations,
            "resident_budget_bytes": resident_budget,
            "required_bytes": required,
        }
        if required > allocatable:
            self.poisoned = "fill refused: device_shortfall"
            return {
                "ok": False,
                "code": "device_shortfall",
                "detail": (
                    f"this fill will hold {required} B on the device "
                    f"({resident_destinations} B of resident destinations under a "
                    f"derived ceiling of {resident_budget} B; "
                    f"{envelope['destination_bytes']} B of logical destinations, "
                    f"{envelope['derived_destination_bytes']} B of additional MCC-derived "
                    "destinations, "
                    f"{envelope['native_bytes']} B of native encoded leaves, "
                    f"{envelope['decode_peak_bytes']} B of decode scratch at its widest "
                    f"component) against {allocatable} B allocatable - refused on the "
                    f"arithmetic, before a byte moved. The binding's MCC priced {declared} B "
                    "of logical plus derived destinations; the header independently prices "
                    "stored encoding overhead"
                ),
                "envelope": envelope,
                "declared_weight_bytes": declared,
                "shortfall": {
                    "resource": "vram",
                    "scope": "component_generation",
                    "needed_bytes": required,
                    "available_bytes": int(allocatable),
                    "evidence_class": "measured",
                },
            }
        artifact = Artifact(
            reference.snapshot,
            topology_of(rows),
            Config._verified(
                config_document,
                "unet.config",
                construction.config_digest,
            ),
            # CUSTODY is a BINDING fact, not a runtime constant: a checkpoint that came
            # through the border is canonical and a serve-time extra key is drift, while a
            # local bind of an un-bordered checkpoint tiers its extras (§1.1). Defaulting
            # to canonical keeps the strict reading for any record that does not say.
            custody=binding.get("custody", "canonical"),
            strict_keys=bool(binding.get("strict_keys", False)),
            # THE STAMPS THE CHECKPOINT ACTUALLY CARRIES (#567d). `Artifact.stamps` was
            # never once populated by the serving path: no caller passed it, the default
            # `{}` rode every binding, and `_admit` therefore refused every stamped model
            # class on every artifact — a whole admission plane that could only ever say no.
            stamps={
                str(key): tuple(str(v) for v in values)
                for key, values in (binding.get("stamps") or {}).items()
            },
        )
        seen = Observations()
        registry = ModelRegistry(
            release=binding["release"],
            backend=backend,
            substrate=lambda: serving_substrate(hardware_variant, seen),
        )
        try:
            # `expect` is INSIDE the handler because it now takes the verified read lease:
            # a corrupt artifact refuses here, before construction and before a single
            # destination is reserved, and that refusal is a typed fill refusal like any
            # other rather than an exception escaping the prepare path.
            backend.expect(
                {name: tuple(r.key for r in checkpoints[name].rows(name)) for name in names}
            )
            model = registry.acquire(binding["model_binding_path"], model_cls, artifact)
            optimization = apply_anima_execution_plan(
                model,
                application=str(binding["application"]),
                model_class=f"{model_cls.__module__}:{model_cls.__qualname__}",
            )
        except AnimaOptimizationRefusal as exc:
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
        except Exception as exc:
            # The two failures this path owns, and nothing else: a typed fill refusal, or
            # the backend's own OOM — which is asked of the accelerator boundary rather
            # than named here, so a second backend spells its allocator error once (#447).
            refusal = exc if isinstance(exc, FillRefusal) else None
            if refusal is None and not accel.is_oom(torch, exc):
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
                    "needed_bytes": int(budgets["declared_weight_bytes"]),
                    "available_bytes": int(free),
                    "evidence_class": "measured",
                }
            return reply
        self.backend = backend
        self.registry = registry
        self.models = {binding["model_parameter_name"]: model}
        # THE COMPONENT-USE CONTRACT'S RUNTIME BODY, installed on the constructed
        # generation. From here on, entering a declared scope admits and materializes its
        # set before the body runs; the package cannot observe that it happened.
        self.residency = ComponentResidency(backend=backend, torch=torch, placement="all_resident")
        object.__setattr__(model, "_cozy_residency", self.residency)
        record = registry.generations()[0].record
        # TWO IDENTITIES, because there are two questions and one digest could only answer
        # one of them (codex audit, adopted 2026-08-25).
        #
        # The LOGICAL MODEL is the reference snapshot and the topology around it.
        # cr-006 proved on real bytes that an encoding rung changes no logical keyset and
        # shares its `topology_digest`, which is what keeps `expect_construction_digest` a
        # usable fence.
        self.logical_model_digest = canonical.digest(
            {
                "format": "cozy.runtime.ModelConstructionContract/1",
                "release": binding["release"],
                "model_class": f"{model_cls.__module__}:{model_cls.__qualname__}",
                "artifact": reference.snapshot,
                "config_digest": record.config_digest,
                "factory_identity": record.factory_identity,
                "topology_digest": record.topology_digest,
                # ModelConstructionContract destination sets are canonicalized by component.
                # Construction traversal order is not identity and cannot change this fence.
                "components": sorted(record.components),
                "variant": hardware_variant,
            }
        )
        # The CONSTRUCTED GENERATION is what actually got filled, and since #549.1 it is
        # THE RESOLVED PLAN'S OWN DIGEST that says so. The old spelling hashed a delivery
        # RUNG NAME — an artifact-wide label chosen by a chooser with no per-tensor knowledge
        # — so two materially different fills of a mixed artifact could share a constructed
        # digest, and the field a reader trusted most was the one nothing executed from.
        #
        # The plan digest binds the exact snapshot map, every tensor's implementation digest
        # and route, the device identity, the runtime source digest and the construction
        # contract. `logical_model_digest` stays what it was: the question "which MODEL is
        # this" has a different answer from "which BYTES are resident", and one digest could
        # only ever answer one of them.
        self.plan = plan
        constructed_preimage = canonical.write(
            {
                "logical_model_digest": self.logical_model_digest,
                "plan_digest": plan.digest(),
                "topology_digest": record.topology_digest,
                "config_digest": record.config_digest,
                # THE APPLIED CEILING, derived from the card (cl-027).
                "resident_budget_bytes": resident_budget,
                "execution_optimization": optimization.document() if optimization else None,
            }
        )
        self.constructed_model_digest = (
            "sha256:"
            + hashlib.sha256(
                b"cozy.runtime.constructed-generation\0" + constructed_preimage
            ).hexdigest()
        )
        expected = command.get("expect_construction_digest") or ""
        if expected and expected != self.logical_model_digest:
            self.poisoned = "construction digest mismatch"
            return {
                "ok": False,
                "code": "construction_digest_mismatch",
                "detail": f"the plan prices {expected} and this executor constructed "
                f"{self.logical_model_digest}: the planner pricing variant A while the "
                "executor serves variant B is unrepresentable",
            }
        memory = accel.allocation(torch, self.device_kind)
        self.ready = True
        self.facts = {
            "cuda_init_ms": round(cuda_ms, 2),
            "prepare_ms": round((time.perf_counter() - started) * 1000, 2),
            # An all-inline checkpoint commits without opening the object stream. No stream
            # wall was measured in that case, and zero is the exact value rather than an
            # absent-stat KeyError after a successful construction.
            "fill_ms": backend.stats.get("fill_ms", 0),
            "filled": record.filled,
            "filled_bytes": record.filled_bytes,
            "device_free_bytes": memory["driver_free_bytes"],
            "device_total_bytes": memory["driver_total_bytes"],
            "allocator_bytes": memory["allocated_bytes"],
            "reserved_bytes": memory["reserved_bytes"],
            "resident": dict(backend.vram_charge),
            "parked": sorted(backend.parked),
            "resident_budget_bytes": resident_budget,
            "rss_bytes": _rss(),
            "pinned": dict(backend.pinned),
            "store": backend.stats.get("store", {}),
            "gpu_name": accel.device_identity(torch, self.device_kind)["name"],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "",
            # THE CONFESSION (§1.1). "Loudly ignored, never silently" is only true if the
            # ignoring is named where a person and a RecordOwner both see it, so the extras
            # this bind dropped and the markers it transcribed ride the bind result.
            "custody": binding.get("custody", "canonical"),
            # THE COMPONENT-USE CONTRACT, as a construction fact. The worker prices a
            # staged rung against it and cannot read it any other way: the declaration lives
            # on the package's class, and that class is only ever imported here.
            "declared_scopes": {
                method: list(names) for method, names in sorted(component_use(model_cls).items())
            },
            "ignored_extra_keys": list(record.ignored_extras),
            "transcribed_markers": list(record.transcribed_markers),
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
        }
        return {
            "ok": True,
            "ready": True,
            "model_construction_digest": self.logical_model_digest,
            "constructed_model_digest": self.constructed_model_digest,
            "entrypoints": sorted(self.surface_names),
            "plan": plan.seam_document(),
            "delivery": self.facts["delivery"],
            "facts": self.facts,
        }

    def _prepare_many(self, command: dict[str, Any]) -> dict[str, Any]:
        """Prepare one complete bounded multi-model callable in this device process."""

        rows = command.get("models")
        if (
            not isinstance(rows, list)
            or not 2 <= len(rows) <= 16
            or any(
                not isinstance(row, dict)
                or set(row) != {"binding", "budgets", "expect_construction_digest"}
                for row in rows
            )
        ):
            return {
                "ok": False,
                "code": "model_binding_set_invalid",
                "detail": "models must be a complete array of 2 through 16 closed binding rows",
            }
        paths = [str(row["binding"].get("model_binding_path", "")) for row in rows]
        parameters = [str(row["binding"].get("model_parameter_name", "")) for row in rows]
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

        children: list[tuple[str, Executor, dict[str, Any]]] = []
        for row in rows:
            binding = row["binding"]
            parameter = str(binding["model_parameter_name"])
            child = Executor(self.channel, self.root, self.fill_forensics_dir)
            child.discovered = self.discovered
            child.surface_names = dict(self.surface_names)
            reply = child._prepare_first(row)
            if not reply.get("ok"):
                self.poisoned = f"model {parameter} refused: {reply.get('code', 'unknown')}"
                return {**reply, "model_parameter": parameter}
            self.discovered = child.discovered
            self.surface_names = dict(child.surface_names)
            children.append((parameter, child, reply))

        self.model_executors = [child for _, child, _ in children]
        self.torch = children[-1][1].torch
        self.models = {}
        for parameter, child, _ in children:
            if set(child.models) != {parameter} or parameter in self.models:
                self.poisoned = "model binding parameter mismatch"
                return {
                    "ok": False,
                    "code": "model_binding_set_invalid",
                    "detail": f"prepared model object did not bind parameter {parameter!r}",
                }
            self.models[parameter] = child.models[parameter]
        residencies = {
            parameter: child.residency
            for parameter, child, _ in children
            if child.residency is not None
        }
        self.residency = _CompositeResidency(residencies)

        logical = [
            {
                "parameter": parameter,
                "digest": str(reply["model_construction_digest"]),
            }
            for parameter, _, reply in children
        ]
        constructed = [
            {"parameter": parameter, "digest": str(reply["constructed_model_digest"])}
            for parameter, _, reply in children
        ]
        self.logical_model_digest = (
            "sha256:"
            + hashlib.sha256(
                b"cozy.runtime.model-construction-set\0" + canonical.write(logical)
            ).hexdigest()
        )
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
        scopes = {
            f"{parameter}.{method}": [f"{parameter}/{component}" for component in components]
            for parameter, facts in facts_rows
            for method, components in (facts.get("declared_scopes") or {}).items()
        }
        pinned_rows = [facts.get("pinned") or {} for _, facts in facts_rows]
        routes: dict[str, int] = {}
        kernels: dict[str, int] = {}
        for _, _, reply in children:
            delivery = reply.get("delivery") or {}
            for name, count in (delivery.get("routes") or {}).items():
                routes[str(name)] = routes.get(str(name), 0) + int(count)
            for name, count in (delivery.get("kernels") or {}).items():
                kernels[str(name)] = kernels.get(str(name), 0) + int(count)
        delivery = {
            "variant": "joint",
            "label": "joint",
            "rung": "joint",
            "delivery": "mixed",
            "materialization": "joint",
            "routes": routes,
            "kernels": kernels,
            "calibrated": all(
                bool(reply.get("delivery", {}).get("calibrated")) for _, _, reply in children
            ),
            "objective": "joint",
        }
        self.facts = {
            "cuda_init_ms": sum(float(facts.get("cuda_init_ms", 0)) for _, facts in facts_rows),
            "prepare_ms": sum(float(facts.get("prepare_ms", 0)) for _, facts in facts_rows),
            "fill_ms": sum(float(facts.get("fill_ms", 0)) for _, facts in facts_rows),
            "filled": sum(int(facts.get("filled", 0)) for _, facts in facts_rows),
            "filled_bytes": sum(int(facts.get("filled_bytes", 0)) for _, facts in facts_rows),
            "device_free_bytes": int(last.get("device_free_bytes", -1)),
            "device_total_bytes": int(last.get("device_total_bytes", -1)),
            "allocator_bytes": int(last.get("allocator_bytes", -1)),
            "reserved_bytes": int(last.get("reserved_bytes", -1)),
            "resident": resident,
            "parked": parked,
            "declared_scopes": scopes,
            "rss_bytes": max(int(facts.get("rss_bytes", 0)) for _, facts in facts_rows),
            "pinned": {
                "registered_bytes": sum(int(row.get("registered_bytes", 0)) for row in pinned_rows),
                "in_flight_bytes": sum(int(row.get("in_flight_bytes", 0)) for row in pinned_rows),
            },
            "gpu_name": str(last.get("gpu_name", "")),
            "torch_version": str(last.get("torch_version", "")),
            "cuda_version": str(last.get("cuda_version", "")),
            "custody": "joint",
            "ignored_extra_keys": [
                f"{parameter}/{key}"
                for parameter, facts in facts_rows
                for key in facts.get("ignored_extra_keys", [])
            ],
            "transcribed_markers": [
                f"{parameter}/{key}"
                for parameter, facts in facts_rows
                for key in facts.get("transcribed_markers", [])
            ],
            "delivery": delivery,
            "models": {parameter: facts for parameter, facts in facts_rows},
            "cross_model_eviction": False,
        }
        self.ready = True
        return {
            "ok": True,
            "ready": True,
            "model_construction_digest": self.logical_model_digest,
            "model_construction_digests": {row["parameter"]: row["digest"] for row in logical},
            "constructed_model_digest": self.constructed_model_digest,
            "entrypoints": sorted(self.surface_names),
            "delivery": delivery,
            "facts": self.facts,
        }

    def _discover_provisioned(self, binding: Mapping[str, Any]) -> None:
        """Import the installed release or explicit development source and bind its descriptor."""

        application = str(binding.get("application") or "")
        descriptor_path = str(binding.get("package_descriptor") or "")
        if not application or not descriptor_path:
            raise RuntimeError("executor requires the canonical installed release identity")
        development_project = str(binding.get("development_project") or "")
        discovered = (
            discover(Path(development_project))
            if development_project
            else discover_installed(application)
        )
        published = descriptor.read_bytes(Path(descriptor_path).read_bytes(), descriptor_path)
        descriptor.compare(
            published,
            descriptor.build(discovered),
            ("provisioned PackageDescriptor", "installed package"),
        )
        self.discovered = discovered
        self.surface_names = {surface.name: surface for surface in discovered.surfaces}

    def _prepare_weightless(self, binding: dict[str, Any], started: float) -> dict[str, Any]:
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
            "transcribed_markers": [],
        }
        return {
            "ok": True,
            "ready": True,
            "weightless": True,
            "model_construction_digest": "",
            "constructed_model_digest": "",
            "entrypoints": sorted(self.surface_names),
            "delivery": {},
            "facts": self.facts,
        }

    # ------------------------------------------------------------------ invoke

    def prepare_request(self, command: dict[str, Any]) -> dict[str, Any]:
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
        if self.surface_names.get(command["entrypoint"]) is None:
            return {
                "ok": False,
                "code": "unknown_entrypoint",
                "detail": f"{command['entrypoint']!r} is not an entrypoint of this release",
            }
        registration = self.discovered.app.get(command["entrypoint"])
        try:
            prepared = prepare(registration, command["payload"])
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
        self.prepared[command["request_id"]] = prepared
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

    def invoke(self, command: dict[str, Any]) -> dict[str, Any]:
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
        surface = self.surface_names.get(command["entrypoint"])
        if surface is None:
            return {
                "ok": False,
                "code": "unknown_entrypoint",
                "detail": f"{command['entrypoint']!r} is not an entrypoint of this release",
            }
        prepared = self.prepared.pop(command["request_id"], None)
        if prepared is None:
            return {
                "ok": False,
                "code": "prepared_request_absent",
                "detail": f"nothing was prepared for {command['request_id']!r}: an attempt "
                "runs the record its plan was priced against, and this executor holds none "
                "(a fresh incarnation holds nothing at all)",
            }
        spool = Path(command["spool"])
        deadline = _deadline(command["deadline_s"])
        torch = self.torch

        # The lossy lane, carrying BOTH kinds of live record. A progress frame is not
        # retained anywhere: its `position` is the monotone unit count the worker's
        # silence detector reads, and that is all it is for. A RETAINED observation rides the
        # same lane so the worker holds a copy BEFORE this process can die — a ring that
        # only crosses in the reply crosses never when the process is killed, which is the
        # exact case triage exists for. A seam refusal there is dropped live output, the ONE
        # loss this lane may incur.
        progress = self._progress_sink(command["request_id"])

        # The attempt's PLACEMENT RUNG, fixed by the PlanChooser before durable acceptance
        # and applied here. The executor decides nothing: it is handed a rung and obeys it,
        # which is what keeps `all_resident` from quietly starting to swap.
        if self.residency is not None:
            self.residency.open_attempt(
                str(command.get("placement", "all_resident")),
                int(command.get("headroom_bytes", 0)),
            )
        if torch is not None:
            accel.reset_peak(torch, self.device_kind)
        allocated_at_start = accel.allocated(torch, self.device_kind) if torch is not None else 0
        lease_acquired = time.perf_counter()
        started_wall = time.time()
        envelope, outcome, record = run_prepared(
            prepared,
            Invocation(
                request_id=command["request_id"],
                spool=spool,
                deadline=deadline,
                boot_warmup=bool(command.get("boot_warmup")),
                device=self._device(),
                cancel=lambda: _cancelled(self.root, command["request_id"]),
                models=self.models,
                progress=progress,
                # cr-012: the inputs the WORKER already fetched, verified and spooled.
                # Every value is a path inside this attempt's own spool: this process is
                # never HANDED an address, and `main` installs `sandbox.refuse_network` so a
                # handler that dials one anyway is refused rather than merely unexpected.
                # That fence is Python-level (cr-042) — it bounds a buggy package, not a
                # hostile one, and there is no netns or seccomp under it.
                assets=_granted_inputs(command),
                max_input_bytes=int(command.get("max_input_bytes") or 64 << 20),
            ),
        )
        handler_ms = (time.perf_counter() - lease_acquired) * 1000

        # §3.4 step 2: the exact D2H wait BEFORE the device lease releases. Everything the
        # tail touches must already be immutable and independent of package GPU state. A
        # WEIGHTLESS generation holds no stream, so its quiescence is structural — there is
        # nothing that could still be moving.
        d2h_started = time.perf_counter()
        if torch is not None:
            accel.synchronize(torch, self.device_kind)
        d2h_ms = (time.perf_counter() - d2h_started) * 1000
        peak_vram = accel.peak_allocated(torch, self.device_kind) if torch is not None else 0
        # step 3: the lease releases here, and nothing after this line touches the device.
        lease_ms = (time.perf_counter() - lease_acquired) * 1000
        quiescent = self._quiescent() if torch is not None else True
        self.attempts += 1
        result_ref, outputs, oversize = _serialize_result(envelope, spool)
        # A typed refusal before handler entry leaves a Ready executor untouched. This is
        # an observed phase bit, not a refusal-code allowlist that drifts as validation grows.
        if outcome.terminal != "succeeded" and record.entered:
            self.poisoned = f"{outcome.terminal}/{outcome.code}"
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
            "residency": self.residency.document() if self.residency is not None else {},
            "observations": record.ring.rows(),
            "observation_caps": record.ring.caps(),
            "attribution": record.attribution.document(),
            "position": record.position,
            "outcome": {
                "terminal": outcome.terminal,
                "origin": outcome.origin,
                "code": outcome.code,
                "message": outcome.message[:2048],
                "fields": list(outcome.fields),
            },
            "quiescent": quiescent,
            **_env_document(),
            "poisoned": self.poisoned,
            "outputs": outputs,
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
            "oversize_result_bytes": oversize,
            "metrics": {
                "handler_ms": round(handler_ms, 3),
                "device_lease_ms": round(lease_ms, 3),
                "d2h_wait_ms": round(d2h_ms, 3),
                "peak_vram_bytes": peak_vram,
                "allocated_at_start_bytes": allocated_at_start,
                "rss_at_end_bytes": _rss(),
                "started_unix": started_wall,
                "gpu_count": 1 if torch is not None else 0,
            },
        }

    # ------------------------------------------------------------------ job

    def _job_models(self, surface: Any, command: Mapping[str, Any], store: Any) -> dict[str, Any]:
        """Create typed Manifest-only Model instances; construct/load no package state."""
        declared = {binding.param: binding for binding in surface.model_bindings}
        rows = command.get("models")
        if rows is None:
            rows = {}
        if not isinstance(rows, Mapping) or set(rows) != set(declared):
            carried = sorted(str(key) for key in rows) if isinstance(rows, Mapping) else []
            raise CapabilityError(
                f"job declares Model parameters {sorted(declared) or 'none'} and the worker "
                f"passed {carried if isinstance(rows, Mapping) else 'a non-object'}",
                code="job_model_binding_mismatch",
            )
        models: dict[str, Any] = {}
        for parameter in sorted(declared):
            row = rows[parameter]
            binding = declared[parameter]
            if not isinstance(row, Mapping) or row.get("class") != binding.class_key:
                raise CapabilityError(
                    f"job Model parameter {parameter!r} changed its descriptor class",
                    code="job_model_class_mismatch",
                )
            manifest = str(row.get("manifest", ""))
            try:
                length = int(row.get("length", 0))
            except (TypeError, ValueError) as exc:
                raise CapabilityError(
                    f"job Model parameter {parameter!r} has a malformed Manifest length",
                    code="job_model_manifest_invalid",
                ) from exc
            try:
                digest = documents.raw(manifest)
            except (documents.DocumentError, ValueError) as exc:
                raise CapabilityError(
                    f"job Model parameter {parameter!r} has a malformed Manifest identity",
                    code="job_model_manifest_invalid",
                ) from exc
            if manifest != documents.spell(digest) or length <= 0:
                raise CapabilityError(
                    f"job Model parameter {parameter!r} has a noncanonical Manifest ref",
                    code="job_model_manifest_invalid",
                )
            try:
                held = store.manifest(manifest)
                if len(bytes(held["manifest"])) != length:
                    raise ValueError("Manifest length changed")
            except Exception as exc:
                raise CapabilityError(
                    f"job Model parameter {parameter!r} is absent or stale in TensorFS",
                    code="job_model_manifest_unavailable",
                ) from exc
            models[parameter] = _derive_model(binding.model_class, manifest)
        return models

    def run_job(self, command: dict[str, Any]) -> dict[str, Any]:
        """ONE `@app.job`, run to completion through the SAME invocation kernel (cr-009).

        Everything structural about a job is here and it is short, which is the point: a job
        is not a second runtime, it is the one attempt kernel with a different service set
        and no serving loop underneath it. The three differences from `invoke`, out loud:

        1. **No prepared generation is required.** A job with no Model parameter never
           imports torch in this process at all — a CPU-class conversion job pays no CUDA
           init and holds no device. A job Model is a typed, derive-only exact Manifest
           capability: no Loader construction, component state, or serving residency exists.
        2. **Job services are injected**: a resumable Scratch tree keyed on the RUN, the
           durable checkpoint exchange, and the typed budget facts.
        3. **Quiescence is proven by what exists.** A torch-free job has no stream that
           could still be moving, so the proof is structural rather than a device query.
        """
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        if self.discovered is None:
            try:
                self._discover_provisioned(
                    {
                        "application": command["application"],
                        "package_descriptor": command["package_descriptor"],
                    }
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "code": "job_release_undiscoverable",
                    "detail": f"{type(exc).__name__}: {exc}"[:1024],
                }
        name = command["job"]
        surface = self.surface_names.get(name)
        if surface is None or surface.kind != "job":
            return {
                "ok": False,
                "code": "unknown_job",
                "detail": f"{name!r} is not a registered @app.job of this release (it has "
                f"{sorted(s.name for s in self.surface_names.values() if s.kind == 'job')})",
            }
        registration = self.discovered.app.get(name)
        spool = Path(command["spool"])
        scratch = Path(command["scratch"]) if command.get("scratch") else None
        if scratch is not None:
            scratch.mkdir(parents=True, exist_ok=True)

        job_models: dict[str, Any] = {}
        store = None
        if surface.model_bindings:
            from cozy_runtime.internal import fill

            store_value = str(command.get("artifact_store", ""))
            if not store_value or not Path(store_value).is_dir():
                return {
                    "ok": False,
                    "code": "job_model_store_unavailable",
                    "detail": "this model-bound job has no existing TensorFS store",
                }
            try:
                store = fill.tensorfs_module().Store.open(store_value)
            except Exception as exc:
                return {
                    "ok": False,
                    "code": "job_model_store_unavailable",
                    "detail": f"TensorFS store refused ({type(exc).__name__})",
                }
        elif surface.artifact_outputs:
            from cozy_runtime.internal import fill, pull

            store_value = str(command.get("artifact_store", ""))
            if not store_value:
                return {
                    "ok": False,
                    "code": "artifact_store_unavailable",
                    "detail": "this artifact-producing job has no verified TensorFS store",
                }
            try:
                store = pull.open_store(fill.tensorfs_module(), Path(store_value))
            except pull.PullRefusal as exc:
                return {"ok": False, "code": "artifact_store_unavailable", "detail": exc.message}
        try:
            job_models = self._job_models(surface, command, store)
        except CapabilityError as exc:
            return {"ok": False, "code": exc.code, "detail": str(exc)[:1024]}

        artifact_open = None
        if surface.artifact_outputs:
            from cozy_runtime.internal import artifact_sink

            if store is None:
                return {
                    "ok": False,
                    "code": "artifact_store_unavailable",
                    "detail": "this artifact-producing job has no verified TensorFS store",
                }
            allowed_sources = {
                model.checkpoint_ref: int(command["models"][parameter]["length"])
                for parameter, model in job_models.items()
            }

            owner_scope = str(command.get("artifact_owner_scope", ""))
            artifact_request_id = str(command.get("artifact_request_id", ""))
            invocation_spec_digest = str(command.get("invocation_spec_digest", ""))

            def bind_intent(
                output_slot: str, declaration_digest: bytes, declaration: bytes
            ) -> artifact_sink.ArtifactHostBinding:
                artifact_sink.write_exchange_file(spool, output_slot, "declaration", declaration)
                answer = self._durable(
                    "weights_intent",
                    {
                        "output_slot": output_slot,
                        "tensorfs_declaration_digest": documents.spell(declaration_digest),
                        "tensorfs_declaration_length": len(declaration),
                        "requested_writer_generation": 0,
                    },
                )
                if not answer.get("ok"):
                    raise CapabilityError(
                        str(answer.get("detail", "the artifact host refused intent")),
                        code=str(answer.get("code", "artifact_host_refused")),
                    )
                receipt_digest = str(answer.get("weights_receipt_digest", ""))
                return artifact_sink.ArtifactHostBinding(
                    transaction_id=str(answer.get("weights_transaction_id", "")),
                    writer_generation=int(answer.get("writer_generation", 0)),
                    weights_receipt_digest=(
                        documents.raw(receipt_digest) if receipt_digest else b""
                    ),
                )

            def record_receipt(receipt: Any) -> None:
                _, canonical_receipt, receipt_digest = artifact_sink.protocol_receipt(
                    receipt,
                    owner_scope=owner_scope,
                    request_id=artifact_request_id,
                    invocation_spec_digest=invocation_spec_digest,
                )
                artifact_sink.write_exchange_file(
                    spool, receipt.output_slot, "receipt", canonical_receipt
                )
                answer = self._durable(
                    "weights_receipt",
                    {
                        "output_slot": receipt.output_slot,
                        "weights_transaction_id": receipt.artifact_transaction_id,
                        "weights_receipt_digest": documents.spell(receipt_digest),
                        "artifact_receipt_length": len(canonical_receipt),
                    },
                )
                if not answer.get("ok") or answer.get("weights_receipt_digest") != documents.spell(
                    receipt_digest
                ):
                    raise CapabilityError(
                        str(answer.get("detail", "the artifact host refused receipt")),
                        code=str(answer.get("code", "artifact_host_refused")),
                    )

            artifact_host = artifact_sink.ArtifactTransactionHost(
                store=store,
                owner_scope=owner_scope,
                request_id=artifact_request_id,
                invocation_spec_digest=invocation_spec_digest,
                writer_session_id=1,
                allowed_sources=allowed_sources,
                output_bounds={
                    output.name: output.max_new_bytes for output in surface.artifact_outputs
                },
                bind_intent=bind_intent,
                record_receipt=record_receipt,
            )
            artifact_open = artifact_host.open

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
                "checkpoint",
                {
                    "operation_key": save.operation_key,
                    "logical_key": save.logical_key,
                    "content_digest": save.content_digest,
                    "length": save.length,
                    "path": str(save.path),
                },
            )
            if not answer.get("ok"):
                from cozy_runtime.author._services import CheckpointConflict

                raise CheckpointConflict(
                    f"{save.logical_key}: {answer.get('detail', 'the durable exchange refused')}",
                    code=str(answer.get("code", "checkpoint_refused")),
                )
            return CheckpointDeclaration(
                declaration_id=str(answer["receipt_id"]),
                operation_key=save.operation_key,
                logical_key=save.logical_key,
                content_digest=save.content_digest,
                length=save.length,
                replayed=bool(answer.get("replayed", False)),
            )

        budget = command.get("budget") or {}
        started = time.perf_counter()
        started_wall = time.time()
        if self.torch is not None:
            accel.reset_peak(self.torch, self.device_kind)
        envelope, outcome, record = attempt(
            registration,
            command["payload"],
            Invocation(
                request_id=command["request_id"],
                spool=spool,
                deadline=_deadline(command["deadline_s"]),
                device=self._device() if self.torch is not None else Device(),
                cancel=lambda: _cancelled(self.root, command["request_id"]),
                models=job_models,
                trees={
                    ref: (Path(root), digest)
                    for ref, (root, digest) in (command.get("trees") or {}).items()
                },
                assets=_granted_inputs(command),
                max_input_bytes=int(command.get("max_input_bytes") or 64 << 20),
                scratch=scratch,
                checkpoints=durable_checkpoint,
                artifacts=artifact_open,
                budget=BudgetFacts(
                    gpu_rate_usd_per_hour=int(budget.get("gpu_rate_micro_usd_per_hour", 0)) / 1e6,
                    gpu_count=int(budget.get("gpu_count", 0)),
                    cap_usd=int(budget.get("cap_micro_usd", 0)) / 1e6,
                    started_monotonic=time.monotonic(),
                ),
                progress=self._progress_sink(command["request_id"]),
            ),
        )
        handler_ms = (time.perf_counter() - started) * 1000
        peak_vram = 0
        if self.torch is not None:
            accel.synchronize(self.torch, self.device_kind)
            peak_vram = accel.peak_allocated(self.torch, self.device_kind)
        self.attempts += 1
        result_ref, outputs, oversize = _serialize_result(envelope, spool)
        if outcome.terminal != "succeeded" and record.entered:
            self.poisoned = f"{outcome.terminal}/{outcome.code}"
        return {
            "ok": True,
            "kind": "job",
            "shortfall": None,
            "residency": {},
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
            "result_ref": result_ref,
            "result_schema_digest": self._result_schema_digest(surface),
            "adjustments": [],
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

    def _durable(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        """One DURABLE mid-attempt exchange. Blocks until the worker answers.

        This is the seam's other half of `child.Executor.call(on_request=…)`. It is a
        separate lane from progress on purpose: a durable fact must not be sheddable, and
        the lossy lane's whole contract is that it may shed.
        """
        self._exchange += 1
        seq = self._exchange
        self.channel.send({"event": "request", "kind": kind, "seq": seq, **body})
        while True:
            frame = self.channel.recv()
            if frame is None:
                raise SeamError("seam_closed", f"the worker closed during {kind!r}")
            if frame.get("event") == "answer" and int(frame.get("seq", -1)) == seq:
                return frame
            raise SeamError(
                "seam_out_of_order",
                f"expected the answer to {kind}#{seq}, got {frame.get('event')!r}",
            )

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
                            "name": str(frame.name)[:120],
                            "value": frame.fraction,
                            "position": frame.position,
                            "step_ms": frame.step_ms,
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

    # ------------------------------------------------------------------ warm

    def warm(self, command: dict[str, Any]) -> dict[str, Any]:
        """One attempted warm case for one entrypoint, through ORDINARY dispatch.

        It runs HERE, in the disposable process, for one reason that is not convenience:
        synthesizing a payload needs the entrypoint's schema, and reading that schema in the
        worker would mean importing package code into the long-lived no-CUDA process.
        A package whose module scope touches torch would then contaminate the worker
        forever. The executor already imported it and is already disposable.
        """
        from cozy_runtime.internal import schema, warm

        surface = self.surface_names.get(command["entrypoint"])
        if surface is None:
            return {
                "ok": True,
                "outcome": "skipped",
                "reason": f"{command['entrypoint']!r} is not an entrypoint of this release",
            }
        spool = Path(command["spool"])
        payload, reason = warm.synthesize(schema.render(surface.payload_type), spool)
        if reason:
            return {"ok": True, "outcome": "skipped", "reason": reason, "payload": {}}
        # THROUGH ORDINARY DISPATCH means through both halves in order. The warm pass has no
        # worker to prepare for it, so it prepares its own request and then invokes the
        # record — the same two calls a served attempt makes across the seam.
        ready = self.prepare_request({**command, "payload": payload})
        if not ready.get("ok"):
            return {
                "ok": True,
                "outcome": "failed",
                "payload": payload,
                "reason": f"{ready.get('code')}: {str(ready.get('detail'))[:200]}",
            }
        reply = self.invoke({**command, "boot_warmup": True})
        succeeded = reply.get("ok") and reply["outcome"]["terminal"] == "succeeded"
        if succeeded:
            return {
                "ok": True,
                "outcome": "warmed",
                "reason": "",
                "payload": payload,
                "boot_warmup_observed": bool(_read_result(spool).get("warm")),
                "handler_ms": reply.get("metrics", {}).get("handler_ms", 0),
            }
        # A failed warm pass NEVER bricks the worker: it is reported and the executor is
        # left usable, because nothing observed under boot_warmup is correctness evidence.
        self.poisoned = ""
        return {
            "ok": True,
            "outcome": "failed",
            "payload": payload,
            "reason": f"{reply.get('code') or reply['outcome']['code']}: "
            f"{(reply.get('detail') or reply['outcome']['message'])[:200]}",
        }

    def _result_schema_digest(self, surface: Any) -> str:
        """The identity of the DECLARED annotated result surface — never of the value.

        A digest over the value document would change with every request and would fence
        nothing; the RecordOwner needs to know which CONTRACT the bytes satisfy.
        """
        from cozy_runtime.internal import schema

        cached = self.schema_digests.get(surface.name)
        if cached is None:
            cached = self.schema_digests[surface.name] = canonical.digest(
                schema.render(surface.result_type)
            )
        return cached

    def _device(self) -> Device:
        """No torch in this process means no device to name (cl-013)."""
        return Device(self.device_kind, 0) if self.torch is not None else Device()

    def _quiescent(self) -> bool:
        """No author computation may still mutate GPU or output state when a terminal is
        recorded. The proof is the device's own: every stream is complete."""
        return accel.streams_idle(self.torch, self.device_kind)

    def probe(self) -> dict[str, Any]:
        if self.torch is None:
            return {"ok": True, "torch": False, "rss_bytes": _rss()}
        memory = accel.allocation(self.torch, self.device_kind)
        return {
            "ok": True,
            "torch": True,
            "device_free_bytes": memory["driver_free_bytes"],
            "device_total_bytes": memory["driver_total_bytes"],
            "allocator_bytes": memory["allocated_bytes"],
            "reserved_bytes": memory["reserved_bytes"],
            "residency": self.residency.document() if self.residency else {},
            "rss_bytes": _rss(),
            "attempts": self.attempts,
            "ready": self.ready,
            "poisoned": self.poisoned,
            **_env_document(),
        }


#: The keys the minimal reply below actually carries. A reply whose own keys are a subset
#: of these loses nothing by being reduced; anything else would be silently mutilated.
_MINIMAL_KEYS = frozenset(
    {
        "reply",
        "ok",
        "code",
        "detail",
        "outcome",
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
    while True:
        try:
            channel.send(reply)
            return
        except SeamError as exc:
            if exc.code != "frame_too_large" or not rows:
                break
            rows = rows[len(rows) // 2 :]
            reply["observations"] = rows
            reply["observation_caps"] = {
                **(reply.get("observation_caps") or {}),
                "dropped_for_frame": held - len(rows),
            }
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
            "metrics": reply.get("metrics"),
            "outputs": reply.get("outputs") or [],
            "result_ref": reply.get("result_ref"),
            "result_schema_digest": reply.get("result_schema_digest"),
            "oversize_result_bytes": reply.get("oversize_result_bytes"),
            "shortfall": reply.get("shortfall"),
            "residency": reply.get("residency") or {},
            "adjustments": reply.get("adjustments") or [],
            "observations": [],
            "observation_caps": {"dropped_for_frame": held, "minimal_reply": True},
        }
    )


def _read_result(spool: Path) -> dict[str, Any]:
    """The document this process just spooled, read back in-process (the warm pass's half)."""
    path = spool / RESULT_DOCUMENT
    if not path.is_file():
        return {}
    value = canonical.parse(path.read_bytes())
    return value if isinstance(value, dict) else {}


def _serialize_result(
    envelope: Any, spool: Path
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int]:
    """Serialize the typed result EXACTLY ONCE, before its types are lost.

    msgspec encodes the whole annotated result struct — nested structs, lists and all — in
    one pass, so nothing is stringified on the way out and a nested typed value survives as
    a typed value. Assets are the one thing that cannot go on the wire as themselves: the
    hook replaces each with its MANIFEST ROW (`asset_ref`, digest, media type, size), which
    is exactly what `Asset.row()` already publishes and carries no local path.

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

    from cozy_runtime.author._assets import Asset

    if envelope is None:
        return None, [], 0

    def hook(value: object) -> object:
        if isinstance(value, Asset):
            row = value.row()
            return {
                "asset_ref": row["ref"],
                "kind": row["kind"],
                "media_type": row["media_type"],
                "size_bytes": row["size_bytes"],
                "digest": row["digest"],
            }
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cozy-runtime device executor")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)

    _capture_seal()
    # cr-042: the executor's ONE network fence, installed BEFORE the seam dials so the
    # AF_UNIX exemption is exercised on every boot rather than asserted. It refuses
    # Python-level egress from package and job code; it is not a kernel bound and
    # `sandbox.refuse_network` says exactly which adversary that leaves.
    sandbox.refuse_network(
        "the executor is handed verified files in its attempt spool and never an address "
        "(cr-012). Package code that dials is a bug, and this is where the bug stops",
        allow_unix_seam=True,
    )
    channel = connect(args.socket)
    executor = Executor(
        channel,
        Path(args.root),
        fill_forensics_dir=_SEALED.get(child_env.FILL_FORENSICS_DIR, "").strip(),
    )

    while True:
        try:
            command = channel.recv()
        except SeamError:
            return int(Exit.unavailable)
        if command is None:
            return int(Exit.ok)
        name = command.get("cmd", "")
        try:
            if name == "hello":
                reply = {
                    "ok": True,
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "pgid": os.getpgrp(),
                    "no_new_privs": _proc_status("NoNewPrivs"),
                    "oom_score_adj": Path("/proc/self/oom_score_adj").read_text().strip(),
                    "uid": os.getuid(),
                    "euid": os.geteuid(),
                    "sealed": dict(_SEALED),
                    "torch_loaded": "torch" in sys.modules,
                    "rss_bytes": _rss(),
                }
            elif name == "describe_installed":
                reply = executor.describe_installed(command)
            elif name == "prepare":
                reply = executor.prepare(command)
            elif name == "prepare_request":
                reply = executor.prepare_request(command)
            elif name == "invoke":
                reply = executor.invoke(command)
            elif name == "run_job":
                reply = executor.run_job(command)
            elif name == "warm":
                reply = executor.warm(command)
            elif name == "probe":
                reply = executor.probe()
            elif name == "shutdown":
                channel.send({"reply": name, "ok": True})
                return int(Exit.ok)
            else:
                reply = {"ok": False, "code": "unknown_command", "detail": name}
        except Exception as exc:  # the executor reports its own faults; it never hides one
            outcome: Outcome = classify(exc)
            reply = {
                "ok": False,
                "code": outcome.code,
                "detail": f"{type(exc).__name__}: {exc}"[:1024],
                "terminal": outcome.terminal,
                "origin": outcome.origin,
            }
        reply["reply"] = name
        _send_reply(channel, reply)


if __name__ == "__main__":
    raise SystemExit(main())
