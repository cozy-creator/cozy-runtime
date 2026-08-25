"""The disposable device executor: Python, CUDA, bound instances, one GPU attempt.

This is the ONLY process in the worker that imports torch or touches a device. It is
disposable in the strong sense: `kill -9` at any instant is a supported input, and the
supervisor converges to a typed terminal and a fresh incarnation from what it journaled.
Nothing durable lives here — no journal, no credential, no monotone wire counter.

Its whole surface is the seam (`internal/seam.py`), and its whole vocabulary is:

    hello     process facts: pid, parent, no_new_privs, oom_score_adj, the SEALED env
    prepare   construct + fill one binding; report the construction digest and ready facts
    prepare_request  resolve ONE request (decode, normalize, preflight) and retain it, so
              the supervisor prices a plan against the record the handler will actually run
    invoke    run ONE PREPARED attempt through the author invocation kernel and report its facts
    run_job   run ONE @app.job to completion through the SAME kernel, with job services
    probe     device/host memory facts, for the supervisor's ledger reconciliation
    shutdown  cooperative exit

Cancellation arrives out of band as SIGUSR1 (cooperative) and SIGKILL (forceful) — the
supervisor owns both clocks. `ctx.cancelled` reads the flag the handler is signalled with,
which is why a cooperative body unwinds and a non-cooperative one gets killed.

The env seal is proved, not trusted: the sealed values are snapshotted BEFORE torch is
imported, and every reply after CUDA init carries `env_intact`. A late imposition (the
allocator already read its environment) is a mismatch there rather than a silent no-op.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any

from cozy_runtime.author._assets import GrantedInput
from cozy_runtime.author._context import Device
from cozy_runtime.author._errors import Outcome, classify
from cozy_runtime.author._invoke import (
    Invocation,
    PreparedRequest,
    attempt,
    prepare,
    run_prepared,
)
from cozy_runtime.author._services import ProgressFrame
from cozy_runtime.internal import canonical
from cozy_runtime.internal.config import seal_snapshot
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.seam import RESULT_DOCUMENT, Channel, SeamError, connect

#: The inline-result door (worker-protocol/01 `ResultEnvelope`): a small typed result rides
#: the terminal envelope and skips the file-store round trip. Larger than this REFUSES at
#: cr-007 — the blob-receipt branch needs a real write-and-receipt transaction, and minting
#: a receipt for bytes nobody wrote is the fabrication class this runtime exists to prevent.
INLINE_RESULT_MAX = 4 * 1024 * 1024

#: How many PREPARED requests this incarnation holds at once. A serving lane admits and
#: runs one attempt at a time, so anything past the newest few is a request the
#: supervisor prepared and then never dispatched.
PREPARED_HELD = 8

#: Set by SIGUSR1. THE cooperative-cancel signal: `ctx.raise_if_cancelled()` reads it.
_CANCEL = False

#: The environment as it was when this process started, restricted to the sealed names.
#: Captured before any import that could initialize CUDA.
_SEALED: dict[str, str] = {}


def _sigusr1(signum: int, frame: object) -> None:
    global _CANCEL
    _CANCEL = True


def _capture_seal() -> None:
    """Snapshot the seal BEFORE anything that could initialize CUDA."""
    _SEALED.update(seal_snapshot())


def _env_intact() -> bool:
    """Whether the process environment still equals the seal CUDA initialized under."""
    return seal_snapshot() == _SEALED


def _proc_status(field: str) -> str:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(field + ":"):
                return line.split(":", 1)[1].strip()
    return ""


def _rss() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


# --------------------------------------------------------------------------- state


class Executor:
    """One incarnation's mutable state. Everything here dies with the process."""

    def __init__(self, channel: Channel, root: Path) -> None:
        self.channel = channel
        self.root = root
        self.torch: Any = None
        self.registry: Any = None
        self.backend: Any = None
        self.residency: Any = None
        self.models: dict[str, Any] = {}
        self.surface_names: dict[str, Any] = {}
        self.discovered: Any = None
        self.ready = False
        self.poisoned = ""
        self.logical_model_digest = ""
        self.constructed_generation_digest = ""
        self.facts: dict[str, Any] = {}
        self.schema_digests: dict[str, str] = {}
        #: request_id -> the record `prepare_request` resolved, run by `invoke`
        self.prepared: dict[str, PreparedRequest] = {}
        self.attempts = 0
        #: the DURABLE exchange counter (cr-009). Supervisor-answered, never sheddable.
        self._exchange = 0

    # ------------------------------------------------------------------ prepare

    def prepare(self, command: dict[str, Any]) -> dict[str, Any]:
        """Construct + fill one binding. THE cold path; serving never accepts before it."""
        if self.poisoned:
            return {
                "ok": False,
                "code": "poisoned_generation",
                "detail": f"this executor is poisoned ({self.poisoned}); a poisoned "
                "generation has no recovery edge to Ready — the supervisor replaces it",
            }
        binding = command["binding"]
        budgets = command["budgets"]
        started = time.perf_counter()
        if not binding.get("model_class"):
            return self._prepare_weightless(binding, started)

        import torch  # the ONE torch import in the runtime, and it is in THIS process

        self.torch = torch
        from cozy_runtime.author import ModelRegistry
        from cozy_runtime.author._loader import Artifact, Config
        from cozy_runtime.author._model import component_use
        from cozy_runtime.internal.delivery import (
            ROUTE_KERNEL,
            DeliveryChooser,
            DeliveryFact,
            DeliveryRefusal,
            DeliveryTable,
            Variant,
        )
        from cozy_runtime.internal.derive import Observations, serving_substrate
        from cozy_runtime.internal.encoding import Selector, launch_providers, measure_device
        from cozy_runtime.internal.fill import (
            Checkpoint,
            FillRefusal,
            ResourceLedger,
            StreamingFillBackend,
            dtype_name,
            survey_routes,
            topology_of,
        )
        from cozy_runtime.internal.probe import qualify
        from cozy_runtime.internal.staging import ComponentResidency

        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        cuda_ms = (time.perf_counter() - started) * 1000
        if not _env_intact():
            return {
                "ok": False,
                "code": "env_seal_broken",
                "detail": "the process environment changed between the seal and CUDA init; "
                "the allocator read its environment once and a late imposition is a silent "
                "no-op, so this is a refusal rather than a warning",
            }

        if self.discovered is None:
            self.discovered = discover(Path(binding["project"]))
            self.surface_names = {s.name: s for s in self.discovered.surfaces}

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

        # ------------------------------------------------------------ the delivery choice
        #
        # cr-008c. WHICH stored bytes this generation reads, decided once, here, before a
        # destination is reserved — because delivery decides what is resident and that is a
        # property of the construction, not of a request. Every candidate is surveyed from
        # its HEADER (no tensor bytes), the qualification suite runs once for the union of
        # logical dtypes, and the chooser is pure arithmetic over measured envelopes.
        offered = [
            v if isinstance(v, Variant) else Variant.read(v)
            for v in (binding.get("variants") or ())
        ] or [
            Variant(
                name=str(binding.get("variant") or "default"),
                store=binding["store"],
                snapshot=binding["snapshot"],
                snapshots=",".join(f"{k}={v}" for k, v in sorted(per_component.items())),
                reference=True,
            )
        ]
        surveyed: dict[str, tuple[dict[str, Any], list[Any]]] = {}
        for variant in offered:
            vmap = variant.snapshot_map()
            books = {
                name: Checkpoint(variant.store, vmap.get(name, variant.snapshot)) for name in names
            }
            surveyed[variant.name] = (books, [r for name in names for r in books[name].rows(name)])
        device_facts = measure_device(torch, 0)
        providers = launch_providers()
        qualified = qualify(
            torch,
            providers,
            device_facts,
            sorted({dtype_name(r.dtype) for _, rs in surveyed.values() for r in rs}),
            release=binding["release"],
            cache=Path(binding["probe_cache"]) if binding.get("probe_cache") else None,
        )
        selector = Selector(providers, qualified[0], device_facts)
        candidates = [
            (variant, *survey_routes(surveyed[variant.name][1], selector)) for variant in offered
        ]
        table_root = Path(binding.get("delivery_cache") or binding.get("probe_cache") or ".")
        table_path = DeliveryTable.path_for(table_root, device_facts)
        table = DeliveryTable.load(table_path, device_facts)
        try:
            choice = DeliveryChooser(table, device_facts).choose(
                candidates,
                objective=str(binding.get("objective") or "latency"),
                steps_basis=int(binding.get("steps_basis") or 20),
            )
        except DeliveryRefusal as exc:
            self.poisoned = f"delivery refused: {exc.code}"
            return {
                "ok": False,
                "code": exc.code,
                "detail": str(exc).splitlines()[0][:900],
                "delivery": {"walk": exc.walk, "objective": binding.get("objective")},
            }
        checkpoints, rows = surveyed[choice.variant.name]
        ledger = ResourceLedger(budgets={"vram": budgets["vram"], "pinned": budgets["pinned"]})

        # The LOAD leg's liveness feed. Integral-MiB positions on the lossy lane, so the
        # supervisor can tell a wedged multi-GB load from a slow one WITHOUT a boot timeout.
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
            ledger=ledger,
            on_position=position,
            window_bytes=binding.get("window_bytes", 4 << 20),
            slots=binding.get("slots", 16),
            readers=binding.get("readers", 16),
            inflight=binding.get("inflight", 8),
            # cr-006: the per-device qualification suite persists here, first-writer-wins.
            # It is a DEVICE fact, not a store or artifact fact, so it lives under
            # cozy_home rather than beside a checkpoint; a worker landing on a tuple this
            # rig has already probed skips the suite entirely.
            probe_cache=binding.get("probe_cache"),
            release=binding["release"],
            # The suite already ran, above, to answer the delivery choice. Running it a
            # second time here would re-measure a device fact that has not changed.
            qualified=qualified,
        )
        # A STAGING generation re-reads the store after construction, so its GC-safe hold
        # must outlive the first commit. `resident_budget` is what makes a parked component
        # possible at all: 0 means every component stays resident.
        backend.resident_budget = int(budgets.get("resident", 0))
        backend.hold_lease = backend.resident_budget > 0
        artifact = Artifact(
            choice.variant.snapshot,
            topology_of(rows),
            Config(json.loads(Path(binding["config"]).read_text()), "unet.config"),
            # CUSTODY is a BINDING fact, not a runtime constant: a checkpoint that came
            # through the border is canonical and a serve-time extra key is drift, while a
            # local bind of an un-bordered checkpoint tiers its extras (§1.1). Defaulting
            # to canonical keeps the strict reading for any record that does not say.
            custody=binding.get("custody", "canonical"),
            strict_keys=bool(binding.get("strict_keys", False)),
        )
        seen = Observations()
        registry = ModelRegistry(
            release=binding["release"],
            backend=backend,
            substrate=lambda: serving_substrate(binding["variant"], seen),
        )
        model_cls = getattr(self.discovered.module, binding["model_class"])
        try:
            # `expect` is INSIDE the handler because it now takes the verified read lease:
            # a corrupt artifact refuses here, before construction and before a single
            # destination is reserved, and that refusal is a typed fill refusal like any
            # other rather than an exception escaping the prepare path.
            backend.expect(
                {name: tuple(r.key for r in checkpoints[name].rows(name)) for name in names}
            )
            model = registry.acquire(binding["binding_path"], model_cls, artifact)
        except (FillRefusal, torch.cuda.OutOfMemoryError) as exc:
            # The generation is already rolled back, so the numbers that matter are the
            # HIGH-WATER marks beside the after state — "released 0 B" would be true and
            # useless. A device OOM here is a RESOURCE SHORTFALL, not a crash: it is the
            # ordinary outcome of a neighbour holding the card, and it carries the two
            # numbers an operator needs (cr-005).
            oom = not isinstance(exc, FillRefusal)
            if oom:
                # A device OOM must roll back like any other fill failure. cr-005's
                # transaction unwinds on a typed FillRefusal, which an allocator error is
                # not — so without this the destinations stay allocated and the refusal
                # reports "after rollback the allocator holds 2,375,322,112 B", which is an
                # honest report of a rollback that did not happen. Measured exactly that on
                # a contended card. The generation is poisoned either way and the executor
                # is replaced, but a worker holding 2.4 GiB it can never use until the
                # process dies is a worker whose ledger tells the truth about a lie.
                backend.rollback(())
            backend.release_lease()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            code = "device_shortfall" if oom else exc.code
            self.poisoned = f"fill refused: {code}"
            reply: dict[str, Any] = {
                "ok": False,
                "code": code,
                "detail": str(exc).splitlines()[0][:400],
                "state": backend.state,
                "peak_allocator_bytes": torch.cuda.max_memory_allocated(),
                "peak_ledger_vram_bytes": ledger.peak["vram"],
                "allocator_bytes": torch.cuda.memory_allocated(),
                "device_free_bytes": free,
                "device_total_bytes": total,
                "ledger": ledger.rows(),
            }
            if oom:
                reply["shortfall"] = {
                    "resource": "vram",
                    "scope": "component_generation",
                    "needed_bytes": int(budgets["vram"]),
                    "available_bytes": int(free),
                    "evidence_class": "measured",
                }
            return reply
        self.backend = backend
        self.registry = registry
        self.models = {binding["param"]: model}
        # THE COMPONENT-USE CONTRACT'S RUNTIME BODY, installed on the constructed
        # generation. From here on, entering a declared scope admits and materializes its
        # set before the body runs; the endpoint cannot observe that it happened.
        self.residency = ComponentResidency(backend=backend, torch=torch, placement="all_resident")
        object.__setattr__(model, "_cozy_residency", self.residency)
        record = registry.generations()[0].record
        # TWO IDENTITIES, because there are two questions and one digest could only answer
        # one of them (codex audit, adopted 2026-08-25).
        #
        # The LOGICAL MODEL is the REFERENCE variant's snapshot and the topology around it.
        # cr-006 proved on real bytes that an encoding rung changes no logical keyset and
        # shares its `topology_digest`, so every delivery variant of one model is one
        # logical model — which is what keeps `expect_construction_digest` a usable fence
        # for a deployment that offers several variants.
        reference = next((v for v in offered if v.reference), offered[0])
        self.logical_model_digest = canonical.digest(
            {
                "format": "cozy.runtime.ModelConstructionContract/1",
                "release": binding["release"],
                "model_class": f"{model_cls.__module__}:{model_cls.__qualname__}",
                "artifact": reference.snapshot,
                "config_digest": record.config_digest,
                "factory_identity": record.factory_identity,
                "topology_digest": record.topology_digest,
                "components": list(record.components),
                "variant": binding["variant"],
            }
        )
        # The CONSTRUCTED GENERATION is what actually got filled: the SELECTED bytes, the
        # route that decodes them, and the configuration the fill ran under. The logical
        # digest cannot answer "which bytes are resident right now" — it is identical
        # across every variant by construction — so using it as the generation's identity
        # made two generations over different artifacts indistinguishable.
        self.constructed_generation_digest = canonical.digest(
            {
                "format": "cozy.runtime.ConstructedGeneration/1",
                "logical_model_digest": self.logical_model_digest,
                "store": choice.variant.store,
                "snapshot": choice.variant.snapshot,
                "snapshots": choice.variant.snapshot_map(),
                "delivery_variant": choice.variant.name,
                "delivery": choice.rung.delivery,
                "materialization": choice.rung.materialization,
                "route": choice.rung.route,
                "kernel": ROUTE_KERNEL[choice.rung.route],
                "topology_digest": record.topology_digest,
                "config_digest": record.config_digest,
                "components": list(record.components),
                "resident_budget_bytes": int(budgets.get("resident", 0)),
                "custody": str(binding.get("custody", "canonical")),
                "strict_keys": bool(binding.get("strict_keys", False)),
            }
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
        free, total = torch.cuda.mem_get_info()
        self.ready = True
        self.facts = {
            "cuda_init_ms": round(cuda_ms, 2),
            "prepare_ms": round((time.perf_counter() - started) * 1000, 2),
            "fill_ms": backend.stats["fill_ms"],
            "filled": record.filled,
            "filled_bytes": record.filled_bytes,
            "device_free_bytes": free,
            "device_total_bytes": total,
            "allocator_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "resident": dict(backend.vram_charge),
            "parked": sorted(backend.parked),
            "rss_bytes": _rss(),
            "ledger": ledger.rows(),
            "store": backend.stats.get("store", {}),
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "",
            # THE CONFESSION (§1.1). "Loudly ignored, never silently" is only true if the
            # ignoring is named where a person and a coordinator both see it, so the extras
            # this bind dropped and the markers it transcribed ride the bind result.
            "custody": binding.get("custody", "canonical"),
            # THE COMPONENT-USE CONTRACT, as a construction fact. The supervisor prices a
            # staged rung against it and cannot read it any other way: the declaration lives
            # on the endpoint's class, and that class is only ever imported here.
            "declared_scopes": {
                method: list(names) for method, names in sorted(component_use(model_cls).items())
            },
            "ignored_extra_keys": list(record.ignored_extras),
            "transcribed_markers": list(record.transcribed_markers),
            "delivery": choice.document(),
        }
        # BANK WHAT THIS RUN MEASURED (cr-008c). The envelope the next choice reads is the
        # one this fill actually paid: its wall, its decode share, the destinations that
        # ended up resident and the decode scratch high-water the ledger saw. `step_ms` and
        # the deviation are NOT measurable here — they need a forward pass and a comparison
        # against the reference — so an existing row's values are preserved and a fresh row
        # is written UNCONFORMED (-1.0), which makes it ineligible for selection until
        # something measures them. A rung admitted by fill numbers alone is exactly what
        # "arithmetic fit admits nothing" forbids.
        prior = table.get(choice.variant.name, choice.rung.name)
        table.put(
            DeliveryFact(
                variant=choice.variant.name,
                rung=choice.rung.name,
                stored_bytes=sum(r.stored_nbytes for r in rows),
                resident_bytes=sum(backend.vram_charge.values()),
                scratch_bytes=int(ledger.peak.get("decode", 0)),
                fill_ms=int(backend.stats["fill_ms"]),
                decode_ms=int(backend.decoded["ms"]),
                step_ms=prior.step_ms if prior else -1.0,
                deviation_rel_l2=(
                    0.0 if choice.variant.reference else (prior.deviation_rel_l2 if prior else -1.0)
                ),
                samples=(prior.samples + 1) if prior else 1,
                evidence=f"prepare on {device_facts.tuple_key()} at {binding['release']}",
                encodings=tuple(sorted({r.encoded.alias for r in rows})),
            )
        )
        table.save(table_path)
        return {
            "ok": True,
            "ready": True,
            "model_construction_digest": self.logical_model_digest,
            "constructed_generation_digest": self.constructed_generation_digest,
            "entrypoints": sorted(self.surface_names),
            "delivery": choice.document(),
            "facts": self.facts,
        }

    def _prepare_weightless(self, binding: dict[str, Any], started: float) -> dict[str, Any]:
        """A binding that declares NO MODEL: discover the release, be ready, import nothing.

        The whole cold path below is a fill, and there is nothing to fill. Returning here is
        what makes a weightless endpoint servable at all — and it is the ONLY thing a
        cardless runner can ever pass, because everything past this line imports torch
        (cl-013). There is no construction, so there is no construction identity: the
        digests are empty and say so rather than naming a build that never happened.
        """
        if self.discovered is None:
            self.discovered = discover(Path(binding["project"]))
            self.surface_names = {s.name: s for s in self.discovered.surfaces}
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
            "constructed_generation_digest": "",
            "entrypoints": sorted(self.surface_names),
            "delivery": {},
            "facts": self.facts,
        }

    # ------------------------------------------------------------------ invoke

    def prepare_request(self, command: dict[str, Any]) -> dict[str, Any]:
        """Resolve ONE request through the kernel's pre-admission half, and retain it.

        The supervisor calls this BEFORE it prices a plan, so the normalized features and
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
        # A prepared request the supervisor never dispatched (a refusal after the plan, a
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
        global _CANCEL
        _CANCEL = False
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
        deadline = time.monotonic() + command["deadline_s"]
        torch = self.torch

        # The lossy lane, carrying BOTH kinds of live record. A progress frame is not
        # retained anywhere: its `position` is the monotone unit count the supervisor's
        # silence detector reads, and that is all it is for. A RETAINED observation rides the
        # same lane so the supervisor holds a copy BEFORE this process can die — a ring that
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
            torch.cuda.reset_peak_memory_stats()
        allocated_at_start = torch.cuda.memory_allocated() if torch is not None else 0
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
                cancel=lambda: _CANCEL,
                models=self.models,
                progress=progress,
                # cr-012: the inputs the SUPERVISOR already fetched, verified and spooled.
                # Every value is a path inside this attempt's own spool; this process has
                # no outbound network and never resolves an address of its own.
                assets={
                    input_id: GrantedInput(
                        input_id=input_id,
                        local=Path(row["local"]),
                        media_type=str(row["media_type"]),
                        digest=str(row["digest"]),
                        length=int(row["length"]),
                        order=int(row.get("order", 0)),
                    )
                    for input_id, row in (command.get("inputs") or {}).items()
                },
                max_input_bytes=int(command.get("max_input_bytes") or 64 << 20),
            ),
        )
        handler_ms = (time.perf_counter() - lease_acquired) * 1000

        # §3.4 step 2: the exact D2H wait BEFORE the device lease releases. Everything the
        # tail touches must already be immutable and independent of endpoint GPU state. A
        # WEIGHTLESS generation holds no stream, so its quiescence is structural — there is
        # nothing that could still be moving.
        d2h_started = time.perf_counter()
        if torch is not None:
            torch.cuda.synchronize()
        d2h_ms = (time.perf_counter() - d2h_started) * 1000
        peak_vram = torch.cuda.max_memory_allocated() if torch is not None else 0
        # step 3: the lease releases here, and nothing after this line touches the device.
        lease_ms = (time.perf_counter() - lease_acquired) * 1000
        quiescent = self._quiescent() if torch is not None else True
        self.attempts += 1
        result_ref, outputs, oversize = _serialize_result(envelope, spool)
        # Author code entered, so reuse is poisoned by any non-success — a refusal raised
        # BEFORE entry (schema, bounds, preflight) leaves a Ready executor untouched (§3.1).
        pre_entry = (
            "invalid_request",
            "unsupported_input",
            "default_resolution",
            # cr-012's typed input refusals. They fire BEFORE the entrypoint exists to the
            # request, so they leave the generation exactly as they found it — listing them
            # here is what keeps a bad input from poisoning a Ready executor.
            "asset_ungranted",
            "asset_media_type",
            "asset_too_large",
        )
        if outcome.terminal != "succeeded" and outcome.code not in pre_entry:
            self.poisoned = f"{outcome.terminal}/{outcome.code}"
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
            "env_intact": _env_intact(),
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

    def run_job(self, command: dict[str, Any]) -> dict[str, Any]:
        """ONE `@app.job`, run to completion through the SAME invocation kernel (cr-009).

        Everything structural about a job is here and it is short, which is the point: a job
        is not a second runtime, it is the one attempt kernel with a different service set
        and no serving loop underneath it. The three differences from `invoke`, out loud:

        1. **No prepared generation is required.** A job with no Model parameter never
           imports torch in this process at all — a CPU-class conversion job pays no CUDA
           init and holds no device. A job that DOES declare a Model was prepared exactly
           like a serving attempt and reuses that construction for its one attempt.
        2. **Job services are injected**: a resumable Scratch tree keyed on the RUN, the
           durable checkpoint exchange, and the typed budget facts.
        3. **Quiescence is proven by what exists.** A torch-free job has no stream that
           could still be moving, so the proof is structural rather than a device query.
        """
        global _CANCEL
        _CANCEL = False
        if self.poisoned:
            return {"ok": False, "code": "poisoned_generation", "detail": self.poisoned}
        if self.discovered is None:
            try:
                self.discovered = discover(Path(command["project"]))
            except Exception as exc:
                return {
                    "ok": False,
                    "code": "job_release_undiscoverable",
                    "detail": f"{type(exc).__name__}: {exc}"[:1024],
                }
            self.surface_names = {s.name: s for s in self.discovered.surfaces}
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

        from cozy_runtime.author._services import BudgetFacts, CheckpointReceipt, CheckpointSave

        def durable_checkpoint(save: CheckpointSave) -> CheckpointReceipt:
            """Block until the supervisor says the save is DURABLE. Never the lossy lane."""
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
            return CheckpointReceipt(
                receipt_id=str(answer["receipt_id"]),
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
            self.torch.cuda.reset_peak_memory_stats()
        envelope, outcome, record = attempt(
            registration,
            command["payload"],
            Invocation(
                request_id=command["request_id"],
                spool=spool,
                deadline=time.monotonic() + command["deadline_s"],
                device=self._device() if self.torch is not None else Device(),
                cancel=lambda: _CANCEL,
                models=self.models,
                trees={
                    ref: (Path(root), digest)
                    for ref, (root, digest) in (command.get("trees") or {}).items()
                },
                scratch=scratch,
                checkpoints=durable_checkpoint,
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
            self.torch.cuda.synchronize()
            peak_vram = self.torch.cuda.max_memory_allocated()
        self.attempts += 1
        result_ref, outputs, oversize = _serialize_result(envelope, spool)
        pre_entry = ("invalid_request", "unsupported_input", "default_resolution", "tree_ungranted")
        if outcome.terminal != "succeeded" and outcome.code not in pre_entry:
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
            "env_intact": _env_intact(),
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
        """One DURABLE mid-attempt exchange. Blocks until the supervisor answers.

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
                raise SeamError("seam_closed", f"the supervisor closed during {kind!r}")
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
        supervisor would mean importing endpoint code into the long-lived no-CUDA process.
        An endpoint whose module scope touches torch would then contaminate the supervisor
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
        # supervisor to prepare for it, so it prepares its own request and then invokes the
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
        nothing; the coordinator needs to know which CONTRACT the bytes satisfy.
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
        return Device("cuda", 0) if self.torch is not None else Device()

    def _quiescent(self) -> bool:
        """No author computation may still mutate GPU or output state when a terminal is
        journaled. The proof is the device's own: every stream is complete."""
        torch = self.torch
        return bool(torch.cuda.current_stream().query() and torch.cuda.default_stream().query())

    def probe(self) -> dict[str, Any]:
        if self.torch is None:
            return {"ok": True, "torch": False, "rss_bytes": _rss()}
        free, total = self.torch.cuda.mem_get_info()
        return {
            "ok": True,
            "torch": True,
            "device_free_bytes": free,
            "device_total_bytes": total,
            "allocator_bytes": self.torch.cuda.memory_allocated(),
            "reserved_bytes": self.torch.cuda.memory_reserved(),
            "residency": self.residency.document() if self.residency else {},
            "rss_bytes": _rss(),
            "attempts": self.attempts,
            "ready": self.ready,
            "poisoned": self.poisoned,
            "env_intact": _env_intact(),
            "ledger": self.backend.ledger.rows() if self.backend else {},
        }


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
    `frames.0`) — a stable name the coordinator can grant against. It is never a list index
    into the grant: two outputs arriving in a different order must still land in their own
    destinations, and position cannot express that.

    Returns `(result_ref, outputs, oversize_bytes)`. The BYTES land in the attempt spool and
    only their digest and length ride the reply: the public door admits 4 MiB inline and one
    control frame admits 64 KiB, so a result between the two was an executor fault instead of
    a served result until the bytes moved to the spool the supervisor already brokers.
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
    signal.signal(signal.SIGUSR1, _sigusr1)
    channel = connect(args.socket)
    executor = Executor(channel, Path(args.root))

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
                    "no_new_privs": _proc_status("NoNewPrivs"),
                    "oom_score_adj": Path("/proc/self/oom_score_adj").read_text().strip(),
                    "uid": os.getuid(),
                    "euid": os.geteuid(),
                    "sealed": dict(_SEALED),
                    "torch_loaded": "torch" in sys.modules,
                    "rss_bytes": _rss(),
                }
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
        channel.send(reply)


if __name__ == "__main__":
    raise SystemExit(main())
