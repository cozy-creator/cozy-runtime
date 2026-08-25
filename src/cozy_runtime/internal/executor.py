"""The disposable device executor: Python, CUDA, bound instances, one GPU attempt.

This is the ONLY process in the worker that imports torch or touches a device. It is
disposable in the strong sense: `kill -9` at any instant is a supported input, and the
supervisor converges to a typed terminal and a fresh incarnation from what it journaled.
Nothing durable lives here — no journal, no credential, no monotone wire counter.

Its whole surface is the seam (`internal/seam.py`), and its whole vocabulary is:

    hello     process facts: pid, parent, no_new_privs, oom_score_adj, the SEALED env
    prepare   construct + fill one binding; report the construction digest and ready facts
    invoke    run ONE attempt through the author invocation kernel and report its facts
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
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any

from cozy_runtime.author._errors import Outcome, classify
from cozy_runtime.author._invoke import Invocation, attempt
from cozy_runtime.author._services import ProgressFrame
from cozy_runtime.internal import canonical
from cozy_runtime.internal.config import seal_snapshot
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.seam import Channel, SeamError, connect

#: The inline-result door (worker-protocol/01 `ResultEnvelope`): a small typed result rides
#: the terminal envelope and skips the file-store round trip. Larger than this REFUSES at
#: cr-007 — the blob-receipt branch needs a real write-and-receipt transaction, and minting
#: a receipt for bytes nobody wrote is the fabrication class this runtime exists to prevent.
INLINE_RESULT_MAX = 4 * 1024 * 1024

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
        self.construction_digest = ""
        self.facts: dict[str, Any] = {}
        self.schema_digests: dict[str, str] = {}
        self.attempts = 0

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

        import torch  # the ONE torch import in the runtime, and it is in THIS process

        self.torch = torch
        from cozy_runtime.author import ModelRegistry
        from cozy_runtime.author._loader import Artifact, Config
        from cozy_runtime.internal.derive import Observations, serving_substrate
        from cozy_runtime.internal.fill import (
            Checkpoint,
            FillRefusal,
            ResourceLedger,
            StreamingFillBackend,
            topology_of,
        )
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
        checkpoints = {
            name: Checkpoint(binding["store"], per_component.get(name, binding["snapshot"]))
            for name in names
        }
        rows = [row for name in names for row in checkpoints[name].rows(name)]
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
        )
        # A STAGING generation re-reads the store after construction, so its GC-safe hold
        # must outlive the first commit. `resident_budget` is what makes a parked component
        # possible at all: 0 means every component stays resident.
        backend.resident_budget = int(budgets.get("resident", 0))
        backend.hold_lease = backend.resident_budget > 0
        artifact = Artifact(
            binding["snapshot"],
            topology_of(rows),
            Config(json.loads(Path(binding["config"]).read_text()), "unet.config"),
            custody="canonical",
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
            # cr-005's transaction already rolled the generation back on the way out. The
            # numbers that matter are therefore the HIGH-WATER marks (what was live when it
            # failed) beside the after state — "released 0 B" would be true and useless.
            #
            # A device OOM is a RESOURCE SHORTFALL, not an unhandled exception. It is the
            # ordinary outcome of another process holding the card (this box's 8 GiB fits
            # exactly one 4.78 GiB resident), and reporting it as a crash loses the two
            # numbers an operator needs: what was wanted and what was there. The plan
            # reserved against a DECLARED budget; the device is what actually says no. The
            # request-aware ladder that would pick a plan that fits is cr-008b's.
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
        self.construction_digest = canonical.digest(
            {
                "format": "cozy.runtime.ModelConstructionContract/1",
                "release": binding["release"],
                "model_class": f"{model_cls.__module__}:{model_cls.__qualname__}",
                "artifact": binding["snapshot"],
                "config_digest": record.config_digest,
                "factory_identity": record.factory_identity,
                "topology_digest": record.topology_digest,
                "components": list(record.components),
                "variant": binding["variant"],
            }
        )
        expected = command.get("expect_construction_digest") or ""
        if expected and expected != self.construction_digest:
            self.poisoned = "construction digest mismatch"
            return {
                "ok": False,
                "code": "construction_digest_mismatch",
                "detail": f"the plan prices {expected} and this executor constructed "
                f"{self.construction_digest}: the planner pricing variant A while the "
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
        }
        return {
            "ok": True,
            "ready": True,
            "model_construction_digest": self.construction_digest,
            "entrypoints": sorted(self.surface_names),
            "facts": self.facts,
        }

    # ------------------------------------------------------------------ invoke

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
        registration = self.discovered.app.get(command["entrypoint"])
        spool = Path(command["spool"])
        deadline = time.monotonic() + command["deadline_s"]
        torch = self.torch

        def progress(frame: Any) -> None:
            """The lossy lane, carrying BOTH kinds of live record.

            A progress frame is not retained anywhere: its `position` is the monotone unit
            count the supervisor's silence detector reads, and that is all it is for. A
            RETAINED observation rides the same lane so the supervisor holds a copy BEFORE
            this process can die — a ring that only crosses in the reply crosses never when
            the process is killed, which is the exact case triage exists for.

            A seam refusal here is dropped live output, the ONE loss this lane may incur.
            """
            with contextlib.suppress(SeamError):
                if isinstance(frame, ProgressFrame):
                    self.channel.send(
                        {
                            "event": "progress",
                            "request_id": command["request_id"],
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
                            "request_id": command["request_id"],
                            "kind": frame.kind,
                            "name": frame.name,
                            "value": frame.value,
                            "fields": dict(frame.fields),
                            "seq": frame.seq,
                            "at_unix_ms": frame.at_unix_ms,
                        }
                    )

        # The attempt's PLACEMENT RUNG, fixed by the PlanChooser before durable acceptance
        # and applied here. The executor decides nothing: it is handed a rung and obeys it,
        # which is what keeps `all_resident` from quietly starting to swap.
        if self.residency is not None:
            self.residency.placement = str(command.get("placement", "all_resident"))
            self.residency.headroom = int(command.get("headroom_bytes", 0))
            self.residency.last_shortfall = None
        torch.cuda.reset_peak_memory_stats()
        allocated_at_start = torch.cuda.memory_allocated()
        lease_acquired = time.perf_counter()
        started_wall = time.time()
        envelope, outcome, record = attempt(
            registration,
            command["payload"],
            Invocation(
                request_id=command["request_id"],
                spool=spool,
                deadline=deadline,
                boot_warmup=bool(command.get("boot_warmup")),
                device=self._device(),
                cancel=lambda: _CANCEL,
                models=self.models,
                progress=progress,
            ),
        )
        handler_ms = (time.perf_counter() - lease_acquired) * 1000

        # §3.4 step 2: the exact D2H wait BEFORE the device lease releases. Everything the
        # tail touches must already be immutable and independent of endpoint GPU state.
        d2h_started = time.perf_counter()
        torch.cuda.synchronize()
        d2h_ms = (time.perf_counter() - d2h_started) * 1000
        peak_vram = torch.cuda.max_memory_allocated()
        # step 3: the lease releases here, and nothing after this line touches the device.
        lease_ms = (time.perf_counter() - lease_acquired) * 1000
        quiescent = self._quiescent()
        self.attempts += 1
        result_document, outputs, oversize = _serialize_result(envelope)
        if oversize:
            # The seam carries CONTROL: an oversized result must not be attempted across it
            # at all. Only its SIZE crosses, and the supervisor turns that into the typed
            # `result_too_large` refusal — a frame that blows the cap would kill this
            # process and turn a contract refusal into an executor fault.
            result_document = None
            outputs = []
        # Author code entered, so reuse is poisoned by any non-success — a refusal raised
        # BEFORE entry (schema, bounds, preflight) leaves a Ready executor untouched (§3.1).
        pre_entry = ("invalid_request", "unsupported_input", "default_resolution")
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
            "result": result_document,
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
                "gpu_count": 1,
            },
        }

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
        reply = self.invoke({**command, "payload": payload, "boot_warmup": True})
        succeeded = reply.get("ok") and reply["outcome"]["terminal"] == "succeeded"
        if succeeded:
            return {
                "ok": True,
                "outcome": "warmed",
                "reason": "",
                "payload": payload,
                "boot_warmup_observed": bool((reply.get("result") or {}).get("warm")),
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

    def _device(self) -> Any:
        from cozy_runtime.author._context import Device

        return Device("cuda", 0)

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


def _serialize_result(envelope: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]], int]:
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

    Returns `(document, outputs, oversize_bytes)`; `oversize_bytes` is non-zero when the
    canonical result exceeds the inline door, which cr-007 REFUSES rather than papering over
    with a receipt for bytes nobody wrote.
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
    oversize = len(canonical.write(document))
    return document, outputs, oversize if oversize > INLINE_RESULT_MAX else 0


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
            elif name == "invoke":
                reply = executor.invoke(command)
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
