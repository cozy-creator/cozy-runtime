#!/usr/bin/env python
"""cr-009's live verification: the @job surface, the job services and the runner, run for real.

Not a test suite (#160). Every section below drives the ACTUAL machinery — the ephemeral
local worker, the real seam, the real disposable executor, the real journal — over the
real 6.9 GB cozytensors bench store, and prints what it observed.

    corpus/.venv/bin/python scripts/job-live.py e2e       submit -> terminal -> output
    corpus/.venv/bin/python scripts/job-live.py kill      kill -9 mid-job, then converge
    corpus/.venv/bin/python scripts/job-live.py arms      the capability + separation arms
    corpus/.venv/bin/python scripts/job-live.py identity  the CLI verb == a direct run_slice
    corpus/.venv/bin/python scripts/job-live.py bench     job overhead and runner RSS
    corpus/.venv/bin/python scripts/job-live.py all

`arms` needs no store. Everything else reads the store READ-ONLY; nothing here writes into
it, and every output lands in this run's own workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "corpus" / "job"))

PROJECT = ROOT / "corpus" / "job"
STORE = Path("/tmp/cozy-sdxl4/store")
SNAPSHOTS = Path("/tmp/cozy-sdxl4/snapshots.json")
REF = "cozy/sdxl-pipeline@cr-008b"
WORK = Path("/tmp/cozy-job-live")

PASS = 0
FAIL = 0

# --------------------------------------------------------------- the service-arm probe
# Module scope on purpose: a signature is the declaration, and `get_type_hints` resolves it
# against the module — a struct defined inside a function is unresolvable, which describe
# refuses (correctly) as `unresolved_annotation`.

import msgspec  # noqa: E402

from cozy_runtime.author import (  # noqa: E402
    App,
    Budget,
    Checkpoints,
    ImageAsset,
    Scratch,
    Tree,
)
from cozy_runtime.author._assets import GrantedInput  # noqa: E402

PROBE = App()


class ProbeIn(msgspec.Struct):
    pass


class ProbeOut(msgspec.Struct):
    note: str


@PROBE.job
def no_grant(payload: ProbeIn, scratch: Scratch, budget: Budget) -> ProbeOut:
    """Every job service refuses when its capability was not GRANTED to this attempt."""
    del payload
    try:
        scratch.mktemp()
        return ProbeOut("scratch wrote without a grant")
    except Exception as exc:
        first = f"{type(exc).__name__}: {exc}"
    try:
        budget.raise_if_exhausted()
        return ProbeOut(first + " | budget answered without facts")
    except Exception as exc:
        return ProbeOut(f"{first} | {type(exc).__name__}: {exc}")


@PROBE.job
def conflicting(payload: ProbeIn, ckpt: Checkpoints) -> ProbeOut:
    """Same identity replays; same key with different bytes conflicts. Never a replacement."""
    del payload
    first = ckpt.save("state", b"one", operation_key="op/1")
    replay = ckpt.save("state", b"one", operation_key="op/1")
    try:
        ckpt.save("state", b"two", operation_key="op/1")
        return ProbeOut("a conflicting save was ACCEPTED")
    except Exception as exc:
        return ProbeOut(f"{first.receipt_id}|replayed={replay.replayed}|{type(exc).__name__}")


class OneAsset(msgspec.Struct):
    one: ImageAsset


class ListOfAssets(msgspec.Struct):
    frames: list[ImageAsset]


class OptionalAsset(msgspec.Struct):
    one: ImageAsset | None = None


class MapOfTrees(msgspec.Struct):
    trees: dict[str, Tree] | None = None


class NoAssets(msgspec.Struct):
    n: int = 1


@PROBE.job
def reads_a_list(payload: ListOfAssets) -> ProbeOut:
    """Read the BYTES of every element — the call that used to refuse `asset_bytes_unavailable`.

    The note is length + digest rather than the decoded text: since cr-012 an `ImageAsset`
    field admits only image media types, so the elements are real PNGs and a `.decode()`
    would be reading a picture as a string.
    """
    import hashlib as _hashlib

    return ProbeOut(
        "|".join(
            f"{len(body)}:{_hashlib.sha256(body).hexdigest()[:8]}"
            for body in (frame.read_bytes() for frame in payload.frames)
        )
    )


@PROBE.job
def spender(payload: ProbeIn, budget: Budget) -> ProbeOut:
    del payload
    budget.raise_if_exhausted()
    return ProbeOut("the cap did not bite")


def _tiny_png(index: int) -> bytes:
    """A real 1x1 RGB PNG, one per index, built here so the three elements differ."""
    import struct
    import zlib

    raw = bytes([0, index * 40 % 256, 0, 0])  # one filter byte + one RGB pixel

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def head(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def check(label: str, ok: bool, detail: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def payload(dwell_ms: int = 0, **over: Any) -> dict[str, Any]:
    return {
        "store": REF,
        "snapshots": json.loads(SNAPSHOTS.read_text()),
        "dwell_ms": dwell_ms,
        **over,
    }


def fresh(name: str) -> Path:
    target = WORK / name
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    (target / "home").mkdir(parents=True, exist_ok=True)
    return target


def config_for(home: Path) -> Any:
    """One RuntimeConfig per PROCESS is the rule; this harness runs several runs in one, so
    it reads once and rebases `cozy_home` explicitly — the value is passed, never ambient."""
    import dataclasses

    from cozy_runtime.internal.config import read_config

    global _CONFIG
    if _CONFIG is None:
        os.environ["COZY_HOME"] = str(home)
        _CONFIG = read_config()
    return dataclasses.replace(_CONFIG, cozy_home=home)


_CONFIG: Any = None


def job_record(name: str = "census") -> dict[str, Any]:
    from cozy_runtime.internal import descriptor as desc
    from cozy_runtime.internal.discovery import discover

    found = discover(PROJECT)
    body = desc.build(found)
    surface = next(s for s in found.surfaces if s.name == name)
    return {
        "job_descriptor_id": desc.job_descriptor_id(body, name),
        "build_id": str(body["application"]),
        "project": str(found.project),
        "job": name,
        "gpu_count": 0,
        "publishes": bool(surface.publishes),
        "emits_media": bool(surface.emits_media),
    }


def run_job(
    workspace: Path,
    *,
    request_id: str = "job-1",
    dwell_ms: int = 0,
    trees: dict[str, str] | None = None,
    hook: Any = None,
    attempt: int = 1,
    **over: Any,
) -> Any:
    """One job attempt through `internal.local.run_slice` — the SAME body the CLI wraps."""
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions

    record = job_record()
    request = LocalRequest(
        entrypoint="census",
        payload=payload(dwell_ms, **over),
        outputs=("census",),
        request_id=request_id,
        attempt=attempt,
        kind="job",
        job_descriptor_id=record["job_descriptor_id"],
        trees={REF: str(STORE)} if trees is None else trees,
    )
    return run_slice(
        config_for(workspace / "home"),
        request,
        [],
        job_records=[record],
        workspace=workspace,
        release_id=record["build_id"],
        options=WorkerOptions(
            root=workspace / "worker", release_id=record["build_id"], no_warm=True
        ),
        coordinator_hook=hook,
    )


# --------------------------------------------------------------------------- e2e


def section_e2e() -> None:
    head("e2e: submit -> run to completion -> terminal + output, transactionally")
    workspace = fresh("e2e")
    started = time.perf_counter()
    outcome = run_job(workspace)
    wall = (time.perf_counter() - started) * 1000

    check("the run produced a terminal", outcome.ended == "terminal", outcome.error)
    check(
        "terminal is SUCCEEDED",
        outcome.status == "OUTCOME_STATUS_SUCCEEDED",
        outcome.status + " " + str(outcome.terminal.get("safe_message", ""))[:160],
    )
    result = outcome.result or {}
    installed = 6_937_675_734
    check(
        "the census read the REAL store",
        int(result.get("logical_bytes", 0)) == installed,
        f"{result.get('logical_bytes')} logical bytes over {result.get('tensors')} tensors in "
        f"{result.get('components')} components (installed artifact records {installed})",
    )
    check(
        "the output is visible ONLY through the granted destination",
        set(outcome.outputs) == {"census"} and Path(outcome.outputs["census"]).is_file(),
        f"{sorted(outcome.outputs)} -> {outcome.outputs.get('census', '')}",
    )
    check(
        "the manifest names the same output, with its own digest",
        len(outcome.manifest) == 1 and outcome.manifest[0]["output_id"] == "census",
        json.dumps(outcome.manifest[:1])[:200],
    )
    check(
        "one DURABLE checkpoint receipt reached the coordinator",
        len(outcome.checkpoints) == 1 and outcome.checkpoints[0]["outcome"] == "RECORDED",
        json.dumps(outcome.checkpoints)[:220],
    )
    journal_kinds = [row.get("kind") for row in outcome.journal]
    check(
        "the journal carries accepted -> job_checkpoint -> terminal -> closed, in that order",
        journal_kinds.count("accepted") == 1
        and journal_kinds.index("job_checkpoint") > journal_kinds.index("accepted")
        and journal_kinds.index("outcome") > journal_kinds.index("job_checkpoint")
        and journal_kinds.index("closed") > journal_kinds.index("outcome"),
        " ".join(journal_kinds),
    )
    check(
        "the accepted plan is the JOB lane's, with no model construction",
        outcome.accepted.get("delivery") == "job"
        and outcome.accepted.get("placement") == "job_attempt"
        and outcome.accepted.get("model_construction_digest") == "",
        outcome.accepted.get("quantified_choice", "")[:160],
    )
    check(
        "publication is rewritten into job SCRATCH, outside the public repo grammar",
        outcome.publication_scope == "local/_job-job-1",
        outcome.publication_scope,
    )
    from cozy_runtime.internal.worker.session import RECYCLE_EXIT

    check(
        "the run-once worker exits RECYCLE, a disposition distinct from any death",
        outcome.exit_code == RECYCLE_EXIT,
        f"worker exit {outcome.exit_code} (recycle={RECYCLE_EXIT}, ok=0, internal=1)",
    )
    print(f"  wall {wall:.0f} ms; timings {json.dumps(outcome.timings)}")
    globals()["_E2E_DIGEST"] = result.get("topology_digest", "")


# --------------------------------------------------------------------------- kill


def section_kill() -> None:
    head("kill -9 mid-job: typed terminal, no partial output, re-run converges")
    if not globals().get("_E2E_DIGEST"):
        # The convergence claim needs a REFERENCE the interrupted run can be compared to.
        reference = run_job(fresh("kill-ref"), request_id="job-ref")
        globals()["_E2E_DIGEST"] = (reference.result or {}).get("topology_digest", "")
        print(f"  reference (uninterrupted) topology {globals()['_E2E_DIGEST'][:31]}…")
    workspace = fresh("kill")

    def hook(coordinator: Any, worker: Any) -> None:
        """Kill the executor's whole process GROUP once the job is genuinely mid-loop.

        Fault injection belongs to the harness, never to a plant switch inside the runtime
        (decisions #249): this waits on the job's own progress frames, so the kill lands
        between components rather than at an arbitrary moment before anything ran.
        """

        def killer() -> None:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                frames = [f for f in coordinator.progress_frames if b'"type": "progress"' in f.data]
                if len(frames) >= 2 and worker.supervision.current is not None:
                    pid = worker.supervision.current.pid
                    worker.supervision.current.kill_group()
                    print(f"  killed executor pid {pid} after {len(frames)} progress frames")
                    return
                time.sleep(0.02)
            print("  WARNING: the killer never saw enough progress to fire")

        threading.Thread(target=killer, daemon=True, name="killer").start()

    first = run_job(workspace, request_id="job-kill", dwell_ms=400, hook=hook)
    check("the killed attempt still produced a terminal", first.ended == "terminal", first.error)
    check(
        "the terminal is the TYPED abandoned one, not silence",
        first.status == "OUTCOME_STATUS_ABANDONED",
        f"{first.status}: {str(first.terminal.get('safe_message', ''))[:140]}",
    )
    granted = workspace / "outputs"
    written = sorted(p.name for p in granted.glob("*") if p.is_file()) if granted.is_dir() else []
    check(
        "NO partial output is visible at the granted destination",
        not first.outputs and not written,
        f"outputs={sorted(first.outputs)} files={written}",
    )
    scratch = workspace / "scratch" / "job-kill" / "checkpoints" / "components"
    survived = sorted(p.name for p in scratch.glob("*.json")) if scratch.is_dir() else []
    check(
        "the RUN's scratch survives the killed attempt",
        bool(survived),
        f"{len(survived)} component(s) already derived: {survived}",
    )

    second = run_job(workspace, request_id="job-kill", dwell_ms=0, attempt=2)
    check("the re-run reached a terminal", second.ended == "terminal", second.error)
    check(
        "the re-run SUCCEEDED and RESUMED from the surviving scratch",
        second.status == "OUTCOME_STATUS_SUCCEEDED"
        and int((second.result or {}).get("resumed_from", 0)) == len(survived),
        f"{second.status}, resumed_from={(second.result or {}).get('resumed_from')} "
        f"of {len(survived)} surviving component(s)",
    )
    check(
        "the re-run CONVERGES: same topology digest as the uninterrupted run",
        (second.result or {}).get("topology_digest") == globals().get("_E2E_DIGEST"),
        f"{(second.result or {}).get('topology_digest', '')[:31]}… vs "
        f"{str(globals().get('_E2E_DIGEST', ''))[:31]}…",
    )


# --------------------------------------------------------------------------- arms


_JOB_HEAD = """
import msgspec
from cozy_runtime.author import (
    Adjustments, App, Bound, Budget, Checkpoints, Context, FileAsset, Outputs,
    Preflight, Scratch, Telemetry, Tree,
)

app = App()


class In(msgspec.Struct):
    x: int = 1


class Facts(msgspec.Struct, frozen=True):
    y: int


class Out(msgspec.Struct):
    y: int
"""

#: Each arm: (name, module source, the substring the refusal must contain).
DECLARATION_ARMS: tuple[tuple[str, str, str], ...] = (
    (
        "entrypoint_requests_a_job_service",
        "@app.entrypoint\ndef e(payload: In, scratch: Scratch) -> Out:\n    return Out(1)\n",
        "job_serving_separation",
    ),
    (
        "job_requests_the_serving_adjustments_envelope",
        "@app.job\ndef j(payload: In, adj: Adjustments) -> Out:\n    return Out(1)\n",
        "job_serving_separation",
    ),
    (
        "job_declares_a_preflight_hook",
        "def pre(payload: In) -> Facts:\n    return Facts(1)\n\n\n"
        "@app.job(name='j2')\ndef j2(payload: In, facts: Preflight[Facts]) -> Out:\n"
        "    return Out(1)\n",
        "",  # asserted below: `job()` has no preflight door at all
    ),
    (
        "job_declares_a_serving_demand_envelope",
        "@app.job(demand=Bound(reason='structural', vram_bytes=1))\n"
        "def j3(payload: In) -> Out:\n    return Out(1)\n",
        "job_serving_separation",
    ),
    (
        "job_publishes_but_cannot_write",
        "@app.job(publishes=True)\ndef j4(payload: In) -> Out:\n    return Out(1)\n",
        "publication_declaration",
    ),
    (
        "entrypoint_declares_a_publication",
        "@app.entrypoint(name='e2')\ndef e2(payload: In) -> Out:\n    return Out(1)\n",
        "",  # asserted below: `entrypoint()` has no publishes door at all
    ),
)


def _declare(source: str) -> str:
    """Import one planted module in a fresh process and return what describe said."""
    plant = fresh("arms") / "plant"
    plant.mkdir(parents=True, exist_ok=True)
    (plant / "endpoint.toml").write_text('[application]\nobject = "ep:app"\n')
    (plant / "ep.py").write_text(_JOB_HEAD + "\n\n" + source)
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path;"
            "from cozy_runtime.internal.discovery import discover;"
            "discover(Path(sys.argv[1]))",
            str(plant),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    return (out.stderr.strip().splitlines() or [""])[-1]


def section_arms() -> None:
    head("a refusal AFTER a full observation ring still converges on its own message")
    # job-001's host-fit shape: a body that narrates a border and then refuses typed. The
    # ring's budget is 128 KiB and one control frame is 64 KiB, so the reply does not fit —
    # and the tail is what gives way, never the terminal (the executor used to die on its
    # own narration and the attempt reported ABANDONED instead of the refusal).
    narrated = run_job(
        fresh("narrate"), request_id="job-narrate", narrate_rows=300, refuse_after_narration=True
    )
    message = str(narrated.terminal.get("safe_message", ""))
    check(
        "the attempt converges on the ENDPOINT's refusal, not on an executor death",
        narrated.status == "OUTCOME_STATUS_REFUSED" and "host_unfit" in message,
        f"{narrated.status}: {message[:110]}",
    )
    dropped = int((narrated.triage.get("caps") or {}).get("dropped_for_frame", 0))
    check(
        "the tail it could not carry is a NUMBER, not a silence",
        dropped > 0,
        f"dropped_for_frame={dropped}, {len(narrated.triage.get('events') or [])} rows held",
    )

    head("red arms: the structural separation refuses at BUILD")
    import inspect

    from cozy_runtime.author import App

    for name, source, expect in DECLARATION_ARMS:
        if not expect:
            continue
        line = _declare(source)
        check(name, expect in line, line[:160])

    job_params = set(inspect.signature(App.job).parameters)
    check(
        "job_declares_a_preflight_hook",
        "preflight" not in job_params,
        "`app.job` has no preflight door at all — the serving pre-admission hook is not "
        f"spellable on a job (its kwargs are {sorted(job_params - {'self', 'fn'})})",
    )
    entry_params = set(inspect.signature(App.entrypoint).parameters)
    check(
        "entrypoint_declares_a_publication",
        "publishes" not in entry_params and "emits_media" not in entry_params,
        "`app.entrypoint` has no publishes/emits_media door — a publication declaration is "
        f"a job fact (its kwargs are {sorted(entry_params - {'self', 'fn'})})",
    )

    head("red arms: the capability refusals, at RUN time on the real path")
    from cozy_runtime.protocol import worker_pb2 as pb

    workspace = fresh("arms-run")
    ungranted = run_job(workspace, request_id="job-ungranted", trees={})
    cause = (ungranted.terminal.get("cause") or {}) if ungranted.terminal else {}
    check(
        "a job naming an UNGRANTED tensorfs read refuses typed",
        ungranted.status == "OUTCOME_STATUS_REFUSED"
        and int(cause.get("code", 0)) == pb.CauseCode.CAUSE_CODE_UNSUPPORTED_INPUT
        and "is not granted to this attempt" in str(ungranted.terminal.get("safe_message", "")),
        f"{ungranted.status} / cause "
        f"{pb.CauseCode.Name(int(cause.get('code', 0)))}: "
        f"{str(ungranted.terminal.get('safe_message', ''))[:130]}",
    )
    check(
        "and it wrote NOTHING",
        not ungranted.outputs,
        f"outputs={sorted(ungranted.outputs)}",
    )
    _oversize_arm()
    _container_asset_arm()
    _service_arms()


def _oversize_arm() -> None:
    """A payload over the CONTROL SEAM cap refuses BEFORE acceptance, and never hangs.

    ev-003 measured the defect (decisions #302): 73,721 B was ACCEPTED and then the dispatch
    frame raised a worker-side SeamError, so no terminal was ever produced and the run
    hung until its readiness deadline — which only fires for a run that never dispatched.
    The check moved to the acceptance boundary, where the law already says everything either
    completes or refuses typed. Armed on the JOB lane; the code is shared with serving.
    """
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.attempts import PAYLOAD_MAX_BYTES
    from cozy_runtime.internal.worker.session import WorkerOptions
    from cozy_runtime.protocol import worker_pb2 as pb

    workspace = fresh("oversize")
    record = job_record()
    fat = dict(payload())
    fat["snapshots"] = {
        **fat["snapshots"],
        **{f"pad{i:04d}": "sha256:" + "0" * 64 for i in range(1200)},
    }
    started = time.perf_counter()
    outcome = run_slice(
        config_for(workspace / "home"),
        LocalRequest(
            entrypoint="census",
            payload=fat,
            outputs=("census",),
            request_id="job-fat",
            kind="job",
            job_descriptor_id=record["job_descriptor_id"],
            trees={REF: str(STORE)},
        ),
        [],
        job_records=[record],
        workspace=workspace,
        release_id=record["build_id"],
        options=WorkerOptions(
            root=workspace / "worker", release_id=record["build_id"], no_warm=True
        ),
    )
    elapsed = time.perf_counter() - started
    cause = (outcome.terminal.get("cause") or {}) if outcome.terminal else {}
    check(
        "a payload over the control-seam cap refuses PRE-ACCEPT, and does not hang",
        outcome.status == "OUTCOME_STATUS_REFUSED"
        and int(cause.get("code", 0)) == pb.CauseCode.CAUSE_CODE_INVALID_REQUEST
        and elapsed < 60,
        f"{outcome.status} / {pb.CauseCode.Name(int(cause.get('code', 0)))} in {elapsed:.1f} s "
        f"(cap {PAYLOAD_MAX_BYTES} B): "
        f"{str(outcome.terminal.get('safe_message', ''))[:110]}",
    )
    check(
        "the refusal is journaled and nothing was ACCEPTED",
        not any(row.get("kind") == "accepted" for row in outcome.journal),
        " ".join(str(row.get("kind")) for row in outcome.journal),
    )


def _container_asset_arm() -> None:
    """An asset inside a CONTAINER declares assets, so hydration runs (ev-003, #303).

    The type walker emits a node per FIELD, so a `list[ImageAsset]` field's annotation is the
    LIST — and the old predicate asked `issubclass(list[ImageAsset], Asset)`, which is false.
    Hydration was then skipped SILENTLY and every element arrived unhydrated. Fixed rather
    than refused: a list of assets is an ordinary declaration.
    """
    from cozy_runtime.author._invoke import declares_assets, declares_trees

    rows = [
        ("ImageAsset (the case that always worked)", declares_assets(OneAsset), True),
        ("list[ImageAsset]", declares_assets(ListOfAssets), True),
        ("ImageAsset | None", declares_assets(OptionalAsset), True),
        ("dict[str, Tree] | None", declares_trees(MapOfTrees), True),
        ("a schema with neither", declares_assets(NoAssets), False),
    ]
    for label, got, want in rows:
        check(f"declares_assets: {label}", got is want, f"-> {got}")

    workspace = fresh("hydrate")
    staged = workspace / "in"
    staged.mkdir(parents=True, exist_ok=True)
    refs = []
    # cr-012: an input asset is a GRANTED READ CAPABILITY, not a location a request may name.
    # The worker fetches, verifies and spools every one BEFORE acceptance and hands the
    # executor identified files; this driver stands in for that half, because the claim under
    # test is what the KERNEL does with a list of them — hydrate every element, or none.
    grants = {}
    expected: list[str] = []
    for index in range(3):
        blob = staged / f"{index}.png"
        body = _tiny_png(index)
        blob.write_bytes(body)
        refs.append(f"file://{blob}")
        expected.append(f"{len(body)}:{hashlib.sha256(body).hexdigest()[:8]}")
        grants[f"frames.{index}"] = GrantedInput(
            input_id=f"frames.{index}",
            local=blob,
            media_type="image/png",
            digest="sha256:" + hashlib.sha256(body).hexdigest(),
            length=len(body),
            order=index,
        )
    from cozy_runtime.author import attempt
    from cozy_runtime.author._invoke import Invocation

    envelope, outcome, _ = attempt(
        PROBE.get("reads_a_list"),
        {"frames": refs},
        Invocation(request_id="hydrate", spool=workspace / "spool", deadline=1e18, assets=grants),
    )
    note = envelope.result.note if envelope else f"{outcome.code}: {outcome.message}"
    check(
        "every element of a list[ImageAsset] arrives HYDRATED at the handler",
        note == "|".join(expected),
        f"{note[:120]} (wanted {'|'.join(expected)})",
    )


def _service_arms() -> None:
    """The job services' own refusals, exercised through the ONE kernel's public boundary."""
    head("red arms: the job services refuse outside their capability")
    from cozy_runtime.author import BudgetFacts, CheckpointConflict, CheckpointReceipt, attempt
    from cozy_runtime.author._invoke import Invocation

    spool = fresh("services") / "spool"
    envelope, outcome, _ = attempt(
        PROBE.get("no_grant"), {}, Invocation(request_id="svc", spool=spool, deadline=1e18)
    )
    note = envelope.result.note if envelope else f"{outcome.code}: {outcome.message}"
    check(
        "Scratch and Budget refuse typed when the attempt was granted neither",
        "scratch_unavailable" in note and "budget_unavailable" in note,
        note[:190],
    )

    seen: list[Any] = []

    def sink(save: Any) -> Any:
        held = next((s for s in seen if s.operation_key == save.operation_key), None)
        if held is not None and held.content_digest != save.content_digest:
            raise CheckpointConflict(
                f"{save.logical_key}: {save.operation_key!r} is durable at "
                f"{held.content_digest[:19]}… and these bytes digest to "
                f"{save.content_digest[:19]}…"
            )
        replayed = held is not None
        if not replayed:
            seen.append(save)
        return CheckpointReceipt(
            receipt_id=f"ckpt-{len(seen)}",
            operation_key=save.operation_key,
            logical_key=save.logical_key,
            content_digest=save.content_digest,
            length=save.length,
            replayed=replayed,
        )

    envelope, outcome, _ = attempt(
        PROBE.get("conflicting"),
        {},
        Invocation(request_id="ckpt", spool=spool, deadline=1e18, checkpoints=sink),
    )
    note = envelope.result.note if envelope else f"{outcome.code}: {outcome.message}"
    check(
        "the same identity REPLAYS its receipt; the same key with different bytes CONFLICTS",
        "replayed=True" in note and "CheckpointConflict" in note,
        note[:190],
    )

    envelope, outcome, _ = attempt(
        PROBE.get("spender"),
        {},
        Invocation(
            request_id="budget",
            spool=spool,
            deadline=1e18,
            budget=BudgetFacts(
                gpu_rate_usd_per_hour=3600.0,
                gpu_count=1,
                cap_usd=0.0001,
                started_monotonic=time.monotonic() - 10.0,
            ),
        ),
    )
    check(
        "Budget raises on its declared cap, from the TYPED CONFIG facts (never env)",
        envelope is None and outcome.code == "budget_exhausted",
        f"{outcome.terminal}/{outcome.code}: {outcome.message[:110]}",
    )


# --------------------------------------------------------------------------- identity


def section_identity() -> None:
    head("observational identity: `cozy-runtime job` == a direct run_slice call")
    direct = run_job(fresh("id-direct"), request_id="id-1")
    check("the direct run succeeded", direct.status == "OUTCOME_STATUS_SUCCEEDED", direct.error)

    workspace = fresh("id-cli")
    (workspace / "payload.json").write_text(json.dumps(payload()))
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "cozy_runtime.cli.main",
            "--json",
            "--dir",
            str(PROJECT),
            "job",
            "census",
            "--in",
            str(workspace / "payload.json"),
            "--input",
            f"{REF}={STORE}",
            "--out",
            str(workspace / "out"),
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": f"{ROOT / 'src'}:{PROJECT}",
            "COZY_HOME": str(workspace / "home"),
        },
    )
    check("the CLI job verb exits 0", out.returncode == 0, out.stderr.strip()[-200:])
    document = json.loads(out.stdout) if out.stdout.strip().startswith("{") else {}
    cli_result = document.get("result") or {}
    check(
        "the CLI and the direct call produce the SAME typed result",
        {k: v for k, v in cli_result.items() if k not in ("census", "receipts")}
        == {k: v for k, v in (direct.result or {}).items() if k not in ("census", "receipts")},
        f"cli topology {str(cli_result.get('topology_digest'))[:31]}…",
    )
    cli_bytes = Path(document["outputs"]["census"]).read_bytes() if document.get("outputs") else b""
    direct_bytes = Path(direct.outputs["census"]).read_bytes() if direct.outputs else b"x"
    check(
        "the same RESULT BYTES land at the granted destination on both doors",
        cli_bytes == direct_bytes and bool(cli_bytes),
        f"{len(cli_bytes)} B identical",
    )
    # The receipt the runtime MINTS is the receipt the CLI RENDERS. It was journaled before
    # the terminal named it and then dropped from the document, so a producer could not read
    # its own publication from the verb that made it (job-001).
    receipt = document.get("publication") or {}
    check(
        "`job --json` renders the typed publication receipt",
        receipt.get("format") == "cozy.runtime.PublicationReceipt/1"
        and receipt.get("mode") == "job"
        and bool(receipt.get("outputs")),
        f"{receipt.get('grant_id')} · {len(receipt.get('outputs') or [])} output(s) · "
        f"{receipt.get('total_bytes')} B",
    )
    check(
        "and the scope and the durable checkpoints beside it",
        document.get("publication_scope") == receipt.get("grant_id")
        and len(document.get("checkpoints") or []) >= 1,
        f"scope {document.get('publication_scope')} · "
        f"{len(document.get('checkpoints') or [])} checkpoint receipt(s)",
    )


# --------------------------------------------------------------------------- bench


def section_bench() -> None:
    head("benchmarks: job overhead vs a bare attempt, and runner RSS")
    import msgspec

    from cozy_runtime.author import App, attempt
    from cozy_runtime.author._invoke import Invocation

    probe = App()

    class In(msgspec.Struct):
        pass

    class Out(msgspec.Struct):
        ok: bool

    @probe.job
    def nothing(payload: In) -> Out:
        del payload
        return Out(True)

    spool = fresh("bench") / "spool"
    for _ in range(20):
        attempt(probe.get("nothing"), {}, Invocation(request_id="w", spool=spool, deadline=1e18))
    started = time.perf_counter()
    runs = 200
    for _ in range(runs):
        attempt(probe.get("nothing"), {}, Invocation(request_id="b", spool=spool, deadline=1e18))
    bare_us = (time.perf_counter() - started) / runs * 1e6
    print(f"  bare attempt through the kernel:      {bare_us:8.1f} us")

    workspace = fresh("bench-run")
    started = time.perf_counter()
    outcome = run_job(workspace, request_id="bench-1")
    wall = (time.perf_counter() - started) * 1000
    body = float(outcome.terminal.get("metrics", {}).get("handler_ms", 0.0))
    overhead = wall - body
    print(f"  whole job through the real runner:    {wall:8.1f} ms")
    print(f"    of which the job BODY (executor):   {body:8.1f} ms")
    print(f"    runner overhead (spawn+accept+tail): {overhead:7.1f} ms")
    print(f"    {json.dumps(outcome.timings)}")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(f"  harness peak RSS (worker in-process): {rss / 2**20:.1f} MiB")
    check("the benchmark run itself succeeded", outcome.status == "OUTCOME_STATUS_SUCCEEDED")
    check(
        "runner overhead is a fixed cost, not a per-byte one",
        overhead < 1500,
        f"{overhead:.0f} ms for spawn + accept + journal + output transaction",
    )


# --------------------------------------------------------------------------- main

SECTIONS = {
    "e2e": section_e2e,
    "kill": section_kill,
    "arms": section_arms,
    "identity": section_identity,
    "bench": section_bench,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=[*SECTIONS, "all"], nargs="?", default="all")
    args = parser.parse_args()
    names = list(SECTIONS) if args.section == "all" else [args.section]
    if any(n in ("e2e", "kill", "bench", "identity") for n in names) and not STORE.is_dir():
        print(f"the bench store {STORE} is not present on this box", file=sys.stderr)
        return 2
    for name in names:
        SECTIONS[name]()
    print(f"\n{PASS}/{PASS + FAIL} observations green")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
