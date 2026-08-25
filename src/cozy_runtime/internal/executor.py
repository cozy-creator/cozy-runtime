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
from cozy_runtime.internal import canonical
from cozy_runtime.internal.config import impose_late, seal_snapshot
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.exits import Exit
from cozy_runtime.internal.seam import Channel, SeamError, connect

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
        self.models: dict[str, Any] = {}
        self.surface_names: dict[str, Any] = {}
        self.discovered: Any = None
        self.ready = False
        self.poisoned = ""
        self.construction_digest = ""
        self.facts: dict[str, Any] = {}
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
        from cozy_runtime.internal.derive import Observations, meta_substrate
        from cozy_runtime.internal.fill import (
            CasStore,
            FillRefusal,
            ResourceLedger,
            StreamingFillBackend,
            read_plan,
            topology,
        )

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

        plan = read_plan(binding["plan"], binding["component"])
        store = CasStore(binding["store"], verify=binding.get("verify", "record"))
        ledger = ResourceLedger(budgets={"vram": budgets["vram"], "pinned": budgets["pinned"]})
        backend = StreamingFillBackend(
            plan,
            store,
            ledger=ledger,
            window_bytes=binding.get("window_bytes", 16 << 20),
            slots=binding.get("slots", 4),
            readers=binding.get("readers", 4),
            interrupt_at=command.get("plant_interrupt_at"),
        )
        backend.expect({binding["component"]: tuple(row.key for row in plan)})
        artifact = Artifact(
            binding["snapshot"],
            topology(plan),
            Config(json.loads(Path(binding["config"]).read_text()), "unet.config"),
            custody="canonical",
        )
        seen = Observations()
        registry = ModelRegistry(
            release=binding["release"],
            backend=backend,
            substrate=lambda: meta_substrate(binding["variant"], seen),
        )
        model_cls = getattr(self.discovered.module, binding["model_class"])
        try:
            model = registry.acquire(binding["binding_path"], model_cls, artifact)
        except FillRefusal as exc:
            # cr-005's transaction already rolled the generation back on the way out. The
            # numbers that matter are therefore the HIGH-WATER marks (what was live when it
            # failed) beside the after state — "released 0 B" would be true and useless.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            self.poisoned = f"fill refused: {exc.code}"
            return {
                "ok": False,
                "code": exc.code,
                "detail": str(exc).splitlines()[0][:400],
                "state": backend.state,
                "peak_allocator_bytes": torch.cuda.max_memory_allocated(),
                "peak_ledger_vram_bytes": ledger.peak["vram"],
                "allocator_bytes": torch.cuda.memory_allocated(),
                "device_free_bytes": free,
                "device_total_bytes": total,
                "ledger": ledger.rows(),
            }
        self.backend = backend
        self.registry = registry
        self.models = {binding["param"]: model}
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
            "rss_bytes": _rss(),
            "ledger": ledger.rows(),
            "store": backend.store.stats(),
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

        def progress(event: Any) -> None:
            # The lossy lane never blocks the attempt: a seam refusal here is dropped
            # progress, which is the ONE loss this lane may incur.
            with contextlib.suppress(SeamError):
                self.channel.send(
                    {
                        "event": "progress",
                        "request_id": command["request_id"],
                        "kind": event.kind,
                        "name": str(event.name)[:120],
                        "value": event.value if isinstance(event.value, (int, float)) else None,
                    }
                )

        torch.cuda.reset_peak_memory_stats()
        lease_acquired = time.perf_counter()
        started_wall = time.time()
        envelope, outcome = attempt(
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
        # Author code entered, so reuse is poisoned by any non-success — a refusal raised
        # BEFORE entry (schema, bounds, preflight) leaves a Ready executor untouched (§3.1).
        pre_entry = ("invalid_request", "unsupported_input", "default_resolution")
        if outcome.terminal != "succeeded" and outcome.code not in pre_entry:
            self.poisoned = f"{outcome.terminal}/{outcome.code}"
        return {
            "ok": True,
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
            "outputs": [dict(row) for row in (envelope.outputs if envelope else ())],
            "result": _result_document(envelope),
            "adjustments": [
                {
                    "field": row.field,
                    "requested": str(row.requested),
                    "applied": str(row.applied),
                    "reason": f"{row.kind}: {row.reason}"[:256],
                }
                for row in (envelope.adjustments if envelope else ())
            ],
            "metrics": {
                "handler_ms": round(handler_ms, 3),
                "device_lease_ms": round(lease_ms, 3),
                "d2h_wait_ms": round(d2h_ms, 3),
                "peak_vram_bytes": peak_vram,
                "rss_at_end_bytes": _rss(),
                "started_unix": started_wall,
                "gpu_count": 1,
            },
        }

    def _device(self) -> Any:
        from cozy_runtime.author._context import Device

        return Device("cuda", 0)

    def _quiescent(self) -> bool:
        """No author computation may still mutate GPU or output state when a terminal is
        journaled. The proof is the device's own: every stream is complete."""
        torch = self.torch
        return bool(
            torch.cuda.current_stream().query() and torch.cuda.default_stream().query()
        )

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
            "rss_bytes": _rss(),
            "attempts": self.attempts,
            "ready": self.ready,
            "poisoned": self.poisoned,
            "env_intact": _env_intact(),
            "ledger": self.backend.ledger.rows() if self.backend else {},
        }


def _result_document(envelope: Any) -> dict[str, Any] | None:
    """The typed function result, canonicalized. An asset manifest is never conflated with
    the return value (§3.4 step 5), so this carries the RESULT only."""
    if envelope is None:
        return None
    from cozy_runtime.author._walker import field_values

    out: dict[str, Any] = {}
    for name, value in field_values(envelope.result).items():
        if hasattr(value, "ref"):
            out[name] = {"asset": value.ref, "digest": value.digest, "bytes": value.size_bytes}
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
        else:
            out[name] = str(value)
    return out


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="cozy-runtime device executor")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--plant", default="")
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
                if args.plant == "late-env":
                    # RED ARM: impose after CUDA init. The allocator already read its
                    # environment, so this changes nothing about the running process —
                    # which is exactly why the seal check must catch it.
                    impose_late("CUDA_VISIBLE_DEVICES", "")
                    reply["late_env_device_count"] = executor.torch.cuda.device_count()
                    reply["env_intact"] = _env_intact()
                if args.plant == "unquiesced":
                    reply["quiescent"] = False
                if args.plant == "state-seq":
                    reply["state_seq"] = 1
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
