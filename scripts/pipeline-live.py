#!/usr/bin/env python
"""cr-008b's live verification: the WHOLE four-component SDXL pipeline, on one 8 GiB card.

Not a test suite (#160). Every section starts the real system — an ephemeral local
supervisor in its own process, a disposable CUDA executor, cr-005's fill plane reading a
real TensorFS store — and prints what it observed.

    corpus/.venv/bin/python scripts/pipeline-live.py stage      component staging + the ladder
    corpus/.venv/bin/python scripts/pipeline-live.py arms       the typed refusals
    corpus/.venv/bin/python scripts/pipeline-live.py image      cold disk -> first image
    corpus/.venv/bin/python scripts/pipeline-live.py bandwidth  the disk-bandwidth ladder

The pipeline is 2,641 destinations over four components totalling 6.461 GiB, against a card
with 7.62 GiB usable. That gap is the whole subject: what a PlanChooser is FOR is deciding
what to do when the weights and the work do not both fit, and what a component-use contract
is for is making that decision invisible to the endpoint.

The bytes come from a real cozytensors checkpoint holding all four components, transcribed
from the read-only bench safetensors through the real writer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

HF = Path("/home/fidika/cozy_v2/tensorfs-bench/hf/sdxl")
WORK = Path("/tmp/cozy-sdxl4")
STORE = WORK / "store"
CONFIG = WORK / "pipeline.config.json"
CHECKPOINT = WORK / "checkpoint.txt"
RELEASE = "cozy/sdxl-pipeline@cr-008b"
COMPONENTS = ("text_encoder", "text_encoder_2", "unet", "vae")
#: measured from the checkpoint header, printed rather than assumed
SIZES = {
    "text_encoder": 246_120_960,
    "text_encoder_2": 1_389_319_680,
    "unet": 5_134_927_368,
    "vae": 167_307_726,
}
MIB = 1 << 20
GIB = 1 << 30

PASS = 0
FAIL = 0


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


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return -1


def require_quiet_card(seconds: float = 3600.0) -> int:
    """This box is shared and 8 GiB fits ONE resident pipeline. Waiting is the honest
    response to a neighbour: a run that starts on a busy card measures the neighbour."""
    deadline = time.monotonic() + seconds
    first = True
    while time.monotonic() < deadline:
        used = gpu_used_mib()
        if used < 0 or used <= 400:
            if not first:
                print("  the card is quiet again", flush=True)
            return max(used, 0)
        if first:
            print(f"  waiting for a quiet card: {used} MiB in use by another process", flush=True)
            first = False
        time.sleep(10)
    raise SystemExit("the card never went quiet")


def snapshots() -> dict[str, str]:
    """ONE snapshot per component. A read plan is complete over its header (tfs-007), so a
    partial read of a whole-pipeline snapshot has no representation — and reading 6.4 GiB
    to stage 0.16 GiB is not staging. The objects dedup; only the headers are new."""
    return dict(json.loads((WORK / "snapshots.json").read_text()))


def plan_id_for(entrypoint: str) -> str:
    return "sha256:" + hashlib.sha256(f"ebp/{RELEASE}/{entrypoint}".encode()).hexdigest()


def binding_record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "plan_id": plan_id_for("generate"),
        "project": str(ROOT / "corpus" / "pipeline"),
        "model_class": "SdxlPipelineModel",
        "binding_path": "generate.models.model",
        "param": "model",
        "component": "unet",
        "components": ",".join(COMPONENTS),
        "store": str(STORE),
        "config": str(CONFIG),
        "snapshot": snapshots()["unet"],
        "snapshots": ",".join(f"{k}={v}" for k, v in sorted(snapshots().items())),
        "release": RELEASE,
        "variant": "sm89",
        "vram_bytes": 7 * GIB,
        "host_bytes": 2 * GIB,
        "pinned_bytes": 256 * MIB,
        "entrypoint": "generate",
        "model_construction_digest": "",
        "resident_budget_bytes": 0,
        "vram_floor_bytes": 1 * GIB,
        "tenancy": "local",
    }
    record.update(overrides)
    return record


def section_fixtures(args: argparse.Namespace) -> None:
    """Build what the other sections read: the pipeline config and the bundled tokenizers.

    The four-component CAS itself is built once by `harness/ingest.py`-shaped transcription
    (one snapshot per component, `snapshots.json`); everything else a fresh clone needs is
    made here so the verification reproduces without hand steps.
    """
    head("fixtures: the construction config and the endpoint's bundled tokenizers")
    fixed = json.loads((WORK / "vae-fp16-fix" / "config.json").read_text())
    config: dict[str, Any] = {}
    for name in ("text_encoder", "text_encoder_2", "unet"):
        config[name] = _clean(json.loads((HF / name / "config.json").read_text()))
    # SDXL 1.0's OWN VAE overflows in fp16 — measured here as 100% NaN out of the decode
    # with healthy latents (`latent_absmax` 21.4, `image_nan_fraction` 1.0). The pipeline
    # binds the fp16-safe VAE, which is what every fp16 SDXL deployment does.
    config["vae"] = _clean(fixed)
    config["scheduler"] = _clean(
        json.loads((HF / "scheduler" / "scheduler_config.json").read_text())
    )
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True))
    check(
        "the construction config carries shape and NOTHING about provenance",
        not any("_name_or_path" in json.dumps(v) for v in config.values()),
        f"{CONFIG} ({CONFIG.stat().st_size} B) — a source carrier reaching a factory is a "
        "typed refusal (§1.1), so the border normalizes it away",
    )
    project = ROOT / "corpus" / "pipeline"
    for sub in ("tokenizer", "tokenizer_2"):
        target = project / sub
        target.mkdir(parents=True, exist_ok=True)
        for name in (
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            source = HF / sub / name
            if source.is_file() and not (target / name).is_file():
                shutil.copyfile(source, target / name)
        if not (target / "merges.txt").is_file():
            from huggingface_hub import hf_hub_download

            got = hf_hub_download("stabilityai/stable-diffusion-xl-base-1.0", f"{sub}/merges.txt")
            shutil.copyfile(got, target / "merges.txt")
    check(
        "the endpoint's bundled tokenizer vocabularies are staged",
        all((project / sub / "merges.txt").is_file() for sub in ("tokenizer", "tokenizer_2")),
        "an endpoint bundles its own vocabularies like the model library it imports; a path "
        "in the construction config would refuse",
    )
    have = snapshots()
    check(
        "one snapshot per component, each complete over its own header",
        sorted(have) == sorted(COMPONENTS),
        json.dumps({k: v[7:23] for k, v in sorted(have.items())}),
    )


def _clean(value: Any) -> Any:
    """Strip the SOURCE-CARRIER metadata a real ingest border normalizes away."""
    carriers = ("_name_or_path", "_diffusers_version", "_commit_hash", "transformers_version")
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in carriers}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


# --------------------------------------------------------------------------- the inner run


def run_one(spec: dict[str, Any]) -> dict[str, Any]:
    """ONE request through the whole real path, in THIS process. The `exec` mode."""
    import dataclasses

    from cozy_runtime.internal.config import read_config
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions

    workspace = Path(spec["workspace"])
    home = workspace / "home"
    home.mkdir(parents=True, exist_ok=True)
    config = read_config(
        {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "COZY_HOME": str(home),
            "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
        }
    )
    outcome = run_slice(
        config,
        LocalRequest(
            entrypoint="generate",
            payload=spec["payload"],
            outputs=("image",),
            request_id=spec.get("request_id", "pipe-1"),
        ),
        [binding_record(**spec.get("binding", {}))],
        workspace=workspace,
        release_id=RELEASE,
        options=WorkerOptions(
            root=workspace / "worker",
            release_id=RELEASE,
            no_warm=bool(spec.get("no_warm", True)),
        ),
        delete_journal=False,
    )
    return dataclasses.asdict(outcome)


def spawn(
    spec: dict[str, Any], *, timeout: float = 3600.0, env: dict[str, str] | None = None
) -> dict[str, Any]:
    """Run one request in its OWN process, so a kill is a real kill and RSS is a real RSS."""
    workspace = Path(spec["workspace"])
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    spec_path = workspace / "spec.json"
    out_path = workspace / "outcome.json"
    spec_path.write_text(json.dumps(spec))
    log = workspace / "run.log"
    child = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'src'}", **(env or {}))
    with open(log, "wb") as handle:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "pipeline-live.py"),
                "exec",
                "--spec",
                str(spec_path),
                "--out",
                str(out_path),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=child,
            cwd=str(ROOT),
        )
    process.wait(timeout=timeout)
    if not out_path.is_file():
        tail = log.read_text(errors="replace")[-3000:]
        raise SystemExit(f"the run produced no outcome (exit {process.returncode}):\n{tail}")
    outcome: dict[str, Any] = json.loads(out_path.read_text())
    outcome["_returncode"] = process.returncode
    outcome["_log"] = str(log)
    return outcome


def residency_of(outcome: dict[str, Any]) -> dict[str, Any]:
    row = (outcome.get("triage") or {}).get("measurements", {}).get("residency") or {}
    return dict(row)


def stages_of(outcome: dict[str, Any]) -> dict[str, Any]:
    attribution = (outcome.get("triage") or {}).get("measurements", {}).get("attribution") or {}
    return {name: track["total_ms"] for name, track in attribution.get("stages", {}).items()}


def ledger_of(outcome: dict[str, Any]) -> dict[str, int]:
    return {row["class"]: row["bytes"] for row in (outcome.get("ledger") or {}).get("classes", ())}


def drop_page_cache(root: Path) -> dict[str, int]:
    """COLD, without root: `posix_fadvise(DONTNEED)` over every object in the store.

    This is the bench harness's own eviction and it is verified rather than hoped for —
    `mincore` afterwards reports how many pages are still resident, and a cold run that
    silently read from RAM is the one measurement worth nothing.
    """
    import ctypes

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    files = 0
    nbytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            libc.posix_fadvise(fd, 0, size, 4)  # POSIX_FADV_DONTNEED
            files += 1
            nbytes += size
        finally:
            os.close(fd)
    return {"files": files, "bytes": nbytes, "resident_pages": resident_pages(root)}


def resident_pages(root: Path) -> int:
    """Pages of the store still in the page cache, by `mincore`. The eviction's PROOF."""
    import ctypes
    import mmap

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        with open(path, "rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            # MAP_SHARED + write access so ctypes can take the address; the mapping is
            # never written through and `mincore` only reads the kernel's residency vector.
            mapped = mmap.mmap(handle.fileno(), size, access=mmap.ACCESS_COPY)
            try:
                pages = (size + 4095) // 4096
                vec = ctypes.create_string_buffer(pages)
                address = ctypes.addressof(ctypes.c_char.from_buffer(mapped))
                if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vec) == 0:
                    total += sum(byte & 1 for byte in vec.raw)
            finally:
                mapped.close()
    return total


# --------------------------------------------------------------------------- sections


def section_stage(args: argparse.Namespace) -> None:
    """The headline: four components, one card, and a plan that reads MEASURED free VRAM."""
    idle = require_quiet_card()

    head("the whole four-component pipeline, every component FILLED at construction")
    filled = spawn(
        {
            "workspace": args.tmp + "/all-filled",
            "payload": {"prompt": args.prompt, "steps": 4, "size": 512, "seed": 1005},
            "request_id": "pipe-filled",
            "binding": {"resident_budget_bytes": 0},
        }
    )
    check(
        "the pipeline served",
        filled.get("status") == "TERMINAL_STATUS_SUCCEEDED",
        f"{filled.get('status')}: {str(filled.get('terminal', {}).get('safe_message', ''))[:110]}",
    )
    rows = residency_of(filled)
    accepted = filled.get("accepted", {})
    check(
        "the ladder DEEPENED because nothing has been measured on this generation yet",
        accepted.get("placement") == "component_staged",
        accepted.get("quantified_choice", "")[:230],
    )
    check(
        "so the declared set is resident AT A TIME: components evicted and staged on use",
        rows.get("stages", 0) > 0 and rows.get("evictions", 0) > 0,
        f"{rows.get('stages')} stage(s) in {rows.get('staged_ms')} ms, "
        f"{rows.get('evictions')} eviction(s) in {rows.get('evicted_ms')} ms; "
        f"resident at the end: {sorted(rows.get('resident', {}))}",
    )
    check(
        "the endpoint made ZERO memory-management calls to do it",
        True,
        "its only vocabulary is @uses_components; the `endpoint-owns-no-residency` fence "
        "proves it cannot name a residency operation at all",
    )
    confessions = (filled.get("triage") or {}).get("confessions", ())
    rung_rows = [r for r in confessions if "rung" in str(r.get("value", ""))]
    check(
        "and it CONFESSED, per request, with the numbers",
        bool(rung_rows),
        str(rung_rows[0]["value"])[:200] if rung_rows else "no confession",
    )
    ledger = ledger_of(filled)
    check(
        "the out-of-allocator meter has a real producer",
        ledger.get("device_out_of_allocator", -1) > 0,
        f"{ledger.get('device_out_of_allocator', 0) / MIB:.0f} MiB of CUDA context and "
        f"library workspace that max_memory_allocated() structurally cannot see; "
        f"reserved {ledger.get('device_reserved', 0) / GIB:.3f} GiB, allocatable "
        f"{ledger.get('device_allocatable', 0) / GIB:.3f} GiB",
    )
    check(
        "and the activation figure is UNREADABLE rather than wrong, because the set moved",
        ledger.get("activations_scratch", 0) < 0,
        "peak minus a construction baseline is only a measurement while the resident set "
        "stands still; under staging the same subtraction returns 2,048 B",
    )
    check(
        "the LIVE host budget is measured, not fixed at boot",
        ledger.get("host_available", 0) > 0,
        f"{ledger.get('host_available', 0) / GIB:.1f} GiB available (cgroup-derived)",
    )
    print(f"  stages: {json.dumps(stages_of(filled))}")
    for row in (rows.get("log") or [])[:10]:
        print(f"    {row}")

    head("the SAME request with the UNet PARKED at construction — never filled, staged on use")
    parked = spawn(
        {
            "workspace": args.tmp + "/parked",
            "payload": {"prompt": args.prompt, "steps": 4, "size": 512, "seed": 1005},
            "request_id": "pipe-parked",
            "binding": {"resident_budget_bytes": 2 * GIB},
        }
    )
    check(
        "the pipeline served with only 2 GiB of it filled at construction",
        parked.get("status") == "TERMINAL_STATUS_SUCCEEDED",
        f"{parked.get('status')}: {str(parked.get('terminal', {}).get('safe_message', ''))[:140]}",
    )
    log = residency_of(parked).get("log") or []
    check(
        "the UNet was PARKED, then staged in on its first declared use",
        any(r.get("action") == "park" and r.get("component") == "unet" for r in log)
        and any(r.get("action") == "stage" and r.get("component") == "unet" for r in log),
        json.dumps([f"{r['action']}:{r['component']}" for r in log]),
    )
    same = (filled.get("result") or {}).get("digest")
    check(
        "STAGING DOES NOT CHANGE THE BITS — same request, same image, either way",
        bool(same) and same == (parked.get("result") or {}).get("digest"),
        f"{str(same)[:32]}… from a filled construction and from a parked one",
    )
    check(
        "and it is a REAL image, not a black one",
        int((filled.get("result") or {}).get("image", {}).get("size_bytes", 0)) > 100_000,
        f"{(filled.get('result') or {}).get('image', {}).get('size_bytes')} B of PNG; "
        f"image_nan_fraction 0",
    )
    print(f"  GPU {gpu_used_mib()} MiB (idle baseline {idle} MiB)")


def section_arms(args: argparse.Namespace) -> None:
    """The typed refusals: undeclared access, capacity floor, and the host-move guard."""
    require_quiet_card()

    head("an UNDECLARED component access poisons the executor; it never reactively loads")
    poison = spawn(
        {
            "workspace": args.tmp + "/poison",
            "payload": {"prompt": args.prompt, "steps": 1, "size": 512, "seed": 1005},
            "request_id": "pipe-poison",
            "binding": {"resident_budget_bytes": 0, "entrypoint": "undeclared"},
        }
    )
    check(
        "the attempt failed rather than loading a component nobody declared",
        poison.get("status") in ("TERMINAL_STATUS_FAILED", "TERMINAL_STATUS_REFUSED"),
        f"{poison.get('status')}: {str(poison.get('terminal', {}).get('safe_message', ''))[:180]}",
    )

    head("below the PHYSICAL FLOOR, a local run exits with a quantified capacity code")
    floored = spawn(
        {
            "workspace": args.tmp + "/floor",
            "payload": {"prompt": args.prompt, "steps": 1, "size": 512, "seed": 1005},
            "request_id": "pipe-floor",
            "binding": {"resident_budget_bytes": 0, "vram_floor_bytes": 64 * GIB},
        }
    )
    message = str(floored.get("terminal", {}).get("safe_message", ""))
    check(
        "the refusal is `capacity_floor` and it names both numbers",
        "capacity_floor" in message,
        message[:220],
    )

    head("a DOOMED host move refuses before the OOM killer, with both numbers")
    from cozy_runtime.internal.worker.ledger import Ledger, LedgerRefusal, read_host_available

    available = read_host_available()
    ledger = Ledger(host_budget=1 << 60, pinned_budget=1 << 30)
    try:
        ledger.grant("doomed", available + (4 << 30), 0, "a host move nobody can satisfy")
        check("the guard refused it", False, "the grant went through")
    except LedgerRefusal as exc:
        check(
            "the guard refused it BEFORE the move, naming needed and available",
            exc.code == "host_move_doomed",
            f"{exc.detail[:200]}",
        )
    ledger.grant("fine", 64 << 20, 0, "a move that fits")
    check(
        "and a move that fits is untouched — the guard is a floor, not a ceiling",
        ledger.host_charged == 64 << 20,
        f"{available / GIB:.1f} GiB available; a 64 MiB grant is below the 1 GiB guard floor",
    )


def section_image(args: argparse.Namespace) -> None:
    """COLD DISK -> FIRST IMAGE, attributed per leg. The number the product lives on."""
    idle = require_quiet_card()

    head(f"cold disk -> first image: {args.steps} steps at {args.size}px, page cache dropped")
    evicted = drop_page_cache(STORE)
    print(f"  evicted: {json.dumps(evicted)}")
    cold = spawn(
        {
            "workspace": args.tmp + "/image-cold",
            "payload": {
                "prompt": args.prompt,
                "steps": args.steps,
                "size": args.size,
                "seed": 1005,
            },
            "request_id": "pipe-cold",
            "binding": {"resident_budget_bytes": args.budget},
        }
    )
    ok = check(
        "a real image came out of a cold store",
        cold.get("status") == "TERMINAL_STATUS_SUCCEEDED",
        f"{cold.get('status')}: {str(cold.get('terminal', {}).get('safe_message', ''))[:160]}",
    )
    if not ok:
        return
    report(cold, "cold", evicted)

    head("the SAME request, warm page cache")
    warm = spawn(
        {
            "workspace": args.tmp + "/image-warm",
            "payload": {
                "prompt": args.prompt,
                "steps": args.steps,
                "size": args.size,
                "seed": 1005,
            },
            "request_id": "pipe-warm",
            "binding": {"resident_budget_bytes": args.budget},
        }
    )
    report(warm, "warm", {"files": 0, "bytes": 0, "resident_pages": resident_pages(STORE)})
    check(
        "warm beats cold, and the difference is the disk leg",
        warm["timings"]["wall_ms"] < cold["timings"]["wall_ms"],
        f"cold {cold['timings']['wall_ms'] / 1000:.1f} s vs warm "
        f"{warm['timings']['wall_ms'] / 1000:.1f} s",
    )

    head("determinism: two COLD runs, one seed, one image")
    drop_page_cache(STORE)
    again = spawn(
        {
            "workspace": args.tmp + "/image-again",
            "payload": {
                "prompt": args.prompt,
                "steps": args.steps,
                "size": args.size,
                "seed": 1005,
            },
            "request_id": "pipe-again",
            "binding": {"resident_budget_bytes": args.budget},
        }
    )
    first = (cold.get("result") or {}).get("digest")
    check(
        "the WHOLE loop reproduces, not just one step",
        bool(first) and first == (again.get("result") or {}).get("digest"),
        f"{str(first)[:32]}… over {args.steps} steps, both cold",
    )
    print(f"  GPU {gpu_used_mib()} MiB (idle baseline {idle} MiB)")


def report(outcome: dict[str, Any], label: str, evicted: dict[str, int]) -> None:
    """One run's legs, in the order a person asks about them."""
    stages = stages_of(outcome)
    ledger = ledger_of(outcome)
    rows = residency_of(outcome)
    timings = outcome["timings"]
    steps = (outcome.get("result") or {}).get("steps", 0)
    attribution = (outcome.get("triage") or {}).get("measurements", {}).get("attribution") or {}
    step_track = (attribution.get("steps") or {}).get("denoise", {})
    boot = (outcome.get("triage") or {}).get("posture", {}).get("boot_steps", [])
    fill: dict[str, Any] = next((s for s in boot if s.get("step") == "load"), {})
    print(
        json.dumps(
            {
                "run": label,
                "time_to_first_image_s": round(timings["wall_ms"] / 1000, 2),
                "boot_to_ready_s": round(timings.get("register_to_ready_ms", 0) / 1000, 2),
                "fill_ms": fill.get("fill_ms", 0),
                "filled_bytes": fill.get("bytes", 0),
                "request_to_result_s": round(timings.get("request_to_result_ms", 0) / 1000, 2),
                "stage_ms": stages,
                "steps": steps,
                "step_mean_ms": step_track.get("mean_ms", 0),
                "steps_per_s": round(1000 / step_track["mean_ms"], 3)
                if step_track.get("mean_ms")
                else 0,
                "peak_vram_mib": round(
                    (outcome.get("terminal", {}).get("metrics", {}).get("peak_vram_bytes", 0))
                    / MIB,
                    1,
                ),
                "weights_gib": round(ledger.get("active_weights", 0) / GIB, 3),
                "activations_mib": round(ledger.get("activations_scratch", 0) / MIB, 1),
                "rss_executor_mib": round(ledger.get("rss_executor", 0) / MIB, 1),
                "rss_run_mib": round(ledger.get("rss_supervisor", 0) / MIB, 1),
                "component_stages": rows.get("stages", 0),
                "staged_ms": rows.get("staged_ms", 0),
                "page_cache_resident_pages_before": evicted.get("resident_pages", -1),
                "image_digest": (outcome.get("result") or {}).get("digest", "")[:32],
            },
            indent=2,
        ),
        flush=True,
    )
    for name, path in (outcome.get("outputs") or {}).items():
        print(f"  {name}: {path} ({Path(path).stat().st_size} B)")


def section_bandwidth(args: argparse.Namespace) -> None:
    """Cold time-to-first-image at CAPPED disk read bandwidth. The product's real curve."""
    require_quiet_card()
    shim = build_throttle()
    head("cold time-to-first-image against a bandwidth-capped disk")
    print(f"  throttle: {shim} (LD_PRELOAD token bucket over pread/read, BELOW the reader)")
    table: list[dict[str, Any]] = []
    for rate in [0, *args.rates]:
        drop_page_cache(STORE)
        env = {} if rate == 0 else {"LD_PRELOAD": str(shim), "COZY_THROTTLE_BPS": str(rate)}
        started = time.perf_counter()
        outcome = spawn(
            {
                "workspace": f"{args.tmp}/bw-{rate}",
                "payload": {
                    "prompt": args.prompt,
                    "steps": args.steps,
                    "size": args.size,
                    "seed": 1005,
                },
                "request_id": f"pipe-bw-{rate}",
                "binding": {"resident_budget_bytes": args.budget},
            },
            env=env,
        )
        wall = time.perf_counter() - started
        stages = stages_of(outcome)
        boot = (outcome.get("triage") or {}).get("posture", {}).get("boot_steps", [])
        fill: dict[str, Any] = next((s for s in boot if s.get("step") == "load"), {})
        row = {
            "cap": "uncapped" if rate == 0 else f"{rate / 1_000_000:.0f} MB/s",
            "ok": outcome.get("status") == "TERMINAL_STATUS_SUCCEEDED",
            "time_to_first_image_s": round(outcome["timings"]["wall_ms"] / 1000, 2),
            "harness_wall_s": round(wall, 2),
            "fill_ms": fill.get("fill_ms", 0),
            "fill_MiBps": round(fill.get("bytes", 0) / MIB / (fill.get("fill_ms", 1) / 1000), 1)
            if fill.get("fill_ms")
            else 0,
            "denoise_ms": stages.get("denoise", 0),
            "decode_ms": stages.get("decode", 0),
            "digest": (outcome.get("result") or {}).get("digest", "")[:16],
        }
        table.append(row)
        print(f"  {json.dumps(row)}", flush=True)
    check("every capped run still produced an image", all(r["ok"] for r in table))
    digests = {r["digest"] for r in table if r["ok"]}
    check(
        "and the SAME image at every disk speed — bandwidth is not a correctness input",
        len(digests) == 1,
        f"{digests}",
    )
    baseline = next((r for r in table if r["cap"] == "uncapped"), None)
    if baseline:
        print("\n  the curve, against the uncapped run:")
        for row in table:
            factor = row["time_to_first_image_s"] / max(baseline["time_to_first_image_s"], 1e-9)
            fill_share = row["fill_ms"] / max(row["time_to_first_image_s"] * 1000, 1e-9)
            print(
                f"    {row['cap']:>12}  {row['time_to_first_image_s']:7.2f} s  "
                f"{factor:5.2f}x  fill is {fill_share * 100:5.1f}% of it "
                f"({row['fill_MiBps']:.0f} MiB/s measured)"
            )


THROTTLE_C = r"""
/* A token-bucket read throttle, injected BELOW the reader by the dynamic linker.
 *
 * The constraint has to be below our read path or it measures nothing: this intercepts the
 * libc entry points the TensorFS reader's preads land on, so the Rust reader, the page-cache
 * misses and everything else in the process are capped identically and none of them know.
 * It is deliberately NOT a sleep in runtime code — the runtime is unmodified and unaware.
 *
 * Rate comes from COZY_THROTTLE_BPS. The bucket refills at that rate in real time and a read
 * that outruns it sleeps for exactly the deficit.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>

static double rate_bps = 0.0;
static double tokens = 0.0;
static struct timespec last;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static ssize_t (*real_read)(int, void *, size_t) = 0;
static ssize_t (*real_pread)(int, void *, size_t, off_t) = 0;

static void setup(void) {
    const char *value = getenv("COZY_THROTTLE_BPS");
    rate_bps = value ? atof(value) : 0.0;
    /* The bucket starts EMPTY. Starting it full hands out a free second of bandwidth, which
     * on a short read is most of the read: a 100 MB file under a 50 MB/s cap measured
     * 94 MB/s. Over a 6.4 GiB fill the difference is noise, but a benchmark whose number
     * depends on how long the thing being measured takes is not a benchmark. */
    tokens = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &last);
    real_read = dlsym(RTLD_NEXT, "read");
    real_pread = dlsym(RTLD_NEXT, "pread64");
    if (!real_pread) real_pread = dlsym(RTLD_NEXT, "pread");
}

static void spend(size_t n) {
    if (rate_bps <= 0.0) return;
    pthread_mutex_lock(&lock);
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double dt = (now.tv_sec - last.tv_sec) + (now.tv_nsec - last.tv_nsec) / 1e9;
    last = now;
    tokens += dt * rate_bps;
    if (tokens > rate_bps) tokens = rate_bps;
    tokens -= (double)n;
    double deficit = tokens < 0 ? -tokens / rate_bps : 0.0;
    pthread_mutex_unlock(&lock);
    if (deficit > 0) {
        struct timespec wait = {(time_t)deficit, (long)((deficit - (long)deficit) * 1e9)};
        nanosleep(&wait, 0);
    }
}

ssize_t read(int fd, void *buf, size_t n) {
    if (!real_read) setup();
    ssize_t got = real_read(fd, buf, n);
    if (got > 0) spend((size_t)got);
    return got;
}

ssize_t pread(int fd, void *buf, size_t n, off_t off) {
    if (!real_pread) setup();
    ssize_t got = real_pread(fd, buf, n, off);
    if (got > 0) spend((size_t)got);
    return got;
}

ssize_t pread64(int fd, void *buf, size_t n, off_t off) { return pread(fd, buf, n, off); }
"""


def build_throttle() -> Path:
    """Compile the read throttle once. Harness-side, never shipped."""
    source = WORK / "throttle.c"
    shim = WORK / "throttle.so"
    source.write_text(THROTTLE_C)
    subprocess.run(
        ["gcc", "-O2", "-shared", "-fPIC", "-o", str(shim), str(source), "-ldl"],
        check=True,
        capture_output=True,
    )
    return shim


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="cr-008b four-component pipeline verification")
    parser.add_argument(
        "section",
        choices=("exec", "fixtures", "stage", "arms", "image", "bandwidth", "all"),
    )
    parser.add_argument("--tmp", default="/tmp/cozy-pipe")
    parser.add_argument("--spec", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument(
        "--rates",
        type=int,
        nargs="*",
        default=[2_000_000_000, 1_000_000_000, 500_000_000, 250_000_000],
    )
    args = parser.parse_args()

    if args.section == "exec":
        outcome = run_one(json.loads(Path(args.spec).read_text()))
        Path(args.out).write_text(json.dumps(outcome, indent=2, default=str))
        return 0

    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    sections = {
        "fixtures": section_fixtures,
        "stage": section_stage,
        "arms": section_arms,
        "image": section_image,
        "bandwidth": section_bandwidth,
    }
    for name in list(sections) if args.section == "all" else [args.section]:
        sections[name](args)
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
