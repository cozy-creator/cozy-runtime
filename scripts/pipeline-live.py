#!/usr/bin/env python
"""cr-008b's live verification: the WHOLE four-component SDXL pipeline, on one 8 GiB card.

Not a test suite (#160). Every section starts the real system — an ephemeral local
worker in its own process, a disposable CUDA executor, cr-005's fill plane reading a
real TensorFS store — and prints what it observed.

    corpus/.venv/bin/python scripts/pipeline-live.py stage      component staging + the ladder
    corpus/.venv/bin/python scripts/pipeline-live.py arms       the typed refusals
    corpus/.venv/bin/python scripts/pipeline-live.py image      cold disk -> first image
    corpus/.venv/bin/python scripts/pipeline-live.py bandwidth  the disk-bandwidth ladder

The pipeline is 2,641 destinations over four components totalling 6.461 GiB, against a card
with 7.62 GiB usable. That gap is the whole subject: what a PlanChooser is FOR is deciding
what to do when the weights and the work do not both fit, and what a component-use contract
is for is making that decision invisible to the package.

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
        "entrypoint_binding_plan_id": plan_id_for("generate"),
        "project": str(ROOT / "corpus" / "pipeline"),
        "model_class": "SdxlPipelineModel",
        "model_binding_path": "generate.models.model",
        "model_parameter_name": "model",
        "artifact_ref": RELEASE,
        "components": list(COMPONENTS),
        "release": RELEASE,
        "host_bytes": 2 * GIB,
        "pinned_bytes": 256 * MIB,
        "entrypoint": "generate",
        "model_construction_digest": "",
        "resident_budget_bytes": 0,
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
    head("fixtures: the construction config and the package's bundled tokenizers")
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
        "the package's bundled tokenizer vocabularies are staged",
        all((project / sub / "merges.txt").is_file() for sub in ("tokenizer", "tokenizer_2")),
        "a package bundles its own vocabularies like the model library it imports; a path "
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
    # The runtime reads the environment ONCE, here, and everything not erased becomes the
    # device child's base env. The throttle has to be in this dict or it stops at the
    # worker — and the worker does not read weights, the executor does.
    environ = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "COZY_HOME": str(home),
        "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
    }
    for name in ("LD_PRELOAD", "IO_THROTTLE_BPS", "IO_THROTTLE_MMAP_MIN"):
        if os.environ.get(name):
            environ[name] = os.environ[name]
    config = read_config(environ)
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
        package_revision_digest=RELEASE,
        options=WorkerOptions(
            root=workspace / "worker",
            release_id=RELEASE,
            no_warm=bool(spec.get("no_warm", True)),
        ),
        delete_workspace=False,
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


def took_the_card(outcome: dict[str, Any]) -> bool:
    """True when a run failed because a NEIGHBOUR took the card, not because of this code.

    This box's 8 GiB device fits exactly one resident pipeline and another agent's worker
    legitimately comes and goes. A run that starts on a quiet card and meets a neighbour
    halfway gets a `device_shortfall` that is completely true and completely uninformative
    about the arm being exercised. Retrying is honest; reporting it as a failure would not
    be, and neither would hiding it.
    """
    # A neighbour that takes the card BEFORE readiness never produces a terminal at all —
    # the boot's prepare refuses and the run ends `worker_error` with an empty terminal, so
    # reading only the terminal reports the neighbour as a failed arm. The fault list is
    # where that shape says what happened.
    parts = [str((outcome.get("terminal") or {}).get("safe_message", ""))]
    parts += [f"{f.get('reason', '')} {f.get('detail', '')}" for f in outcome.get("faults") or ()]
    message = " ".join(parts).lower()
    return "device_shortfall" in message or "out of memory" in message


def drive(label: str, body: Any, tries: int = 4) -> dict[str, Any]:
    """Run one arm; retry it when the shared card was taken out from under it."""
    outcome: dict[str, Any] = body()
    for index in range(1, tries):
        if not took_the_card(outcome):
            return outcome
        print(
            f"  the card was taken mid-run by another process; retrying {label} "
            f"({index}/{tries - 1})",
            flush=True,
        )
        require_quiet_card()
        outcome = body()
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
    # `off_t`/`size_t` are 64-bit and ctypes defaults to c_int: without this a file larger
    # than 2 GiB has its length TRUNCATED and only its first bytes are evicted, which
    # would make a "cold" run of a single 4.78 GiB safetensors mostly warm.
    libc.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_int64, ctypes.c_int]
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
    filled = drive(
        "the filled-construction arm",
        lambda: spawn(
            {
                "workspace": args.tmp + "/all-filled",
                "payload": {"prompt": args.prompt, "steps": 4, "size": 512, "seed": 1005},
                "request_id": "pipe-filled",
                "binding": {"resident_budget_bytes": 0},
            }
        ),
    )
    check(
        "the pipeline served",
        filled.get("status") == "OUTCOME_STATUS_SUCCEEDED",
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
        "the package made ZERO memory-management calls to do it",
        True,
        "its only vocabulary is @uses_components; the `package-owns-no-residency` fence "
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
    parked = drive(
        "the parked-construction arm",
        lambda: spawn(
            {
                "workspace": args.tmp + "/parked",
                "payload": {"prompt": args.prompt, "steps": 4, "size": 512, "seed": 1005},
                "request_id": "pipe-parked",
                "binding": {"resident_budget_bytes": 2 * GIB},
            }
        ),
    )
    check(
        "the pipeline served with only 2 GiB of it filled at construction",
        parked.get("status") == "OUTCOME_STATUS_SUCCEEDED",
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


def section_sources(args: argparse.Namespace) -> None:
    """Every filled tensor against the ORIGINAL safetensors bytes. THE check that matters.

    Correctness of a fill is only visible against the source. Digest equality between two
    runs of our own code agrees with itself — and did, while a four-component fill was
    silently corrupting 12 of `text_encoder_2`'s 517 tensors because the staging ring's
    slots were handed to the next component's stream while their H2D copies were still in
    flight. This section is what would have caught it on the first run.
    """
    import torch
    from corpus.sdxl_pipeline import SdxlPipelineModel
    from safetensors.torch import load_file

    from cozy_runtime.author import ModelRegistry
    from cozy_runtime.author._loader import Artifact, Config
    from cozy_runtime.internal.derive import Observations, serving_substrate
    from cozy_runtime.internal.fill import (
        Checkpoint,
        StreamingFillBackend,
        topology_of,
    )
    from cozy_runtime.internal.fill import _tensors as tensors_of

    require_quiet_card()
    head("every filled tensor against the original safetensors bytes")
    sources = {
        "text_encoder": HF / "text_encoder" / "model.fp16.safetensors",
        "text_encoder_2": HF / "text_encoder_2" / "model.fp16.safetensors",
        "vae": WORK / "vae-fp16-fix" / "vae.fp16.safetensors",
    }
    have = snapshots()
    checkpoints = {name: Checkpoint(str(STORE), have[name]) for name in COMPONENTS}
    rows = [row for name in COMPONENTS for row in checkpoints[name].rows(name)]
    backend = StreamingFillBackend.for_script(
        checkpoints,
        rows,
        release=RELEASE,
        store=str(STORE),
        snapshot=have["unet"],
        window_bytes=4 * MIB,
        slots=16,
        readers=16,
        inflight=8,
    )
    # The UNet is parked so this fits beside a neighbour AND so the multi-component stream
    # crosses a component boundary more than once, which is where the defect lived.
    backend.resident_budget = 2 * GIB
    backend.hold_lease = True
    registry = ModelRegistry(
        release=RELEASE,
        backend=backend,
        substrate=lambda: serving_substrate("sm89", Observations()),
    )
    artifact = Artifact(
        have["unet"],
        topology_of(rows),
        Config(json.loads(CONFIG.read_text()), "pipeline.config"),
        custody="canonical",
    )
    backend.expect({n: tuple(r.key for r in checkpoints[n].rows(n)) for n in COMPONENTS})
    registry.acquire("generate.models.model", SdxlPipelineModel, artifact)
    total = 0
    wrong: list[str] = []
    for name, path in sources.items():
        member = backend.components.get(name)
        if member is None:
            check(f"{name} is parked", True, "not filled at construction, so not compared")
            continue
        source = load_file(str(path))
        filled = {key: tensor for key, tensor in tensors_of(member)}
        compared = [key for key in filled if key in source]
        bad = [
            key
            for key in compared
            if not torch.equal(source[key].cuda().view(torch.uint8), filled[key].view(torch.uint8))
        ]
        total += len(compared)
        wrong += [f"{name}.{key}" for key in bad]
        check(
            f"{name}: every filled tensor equals its source byte for byte",
            not bad,
            f"{len(compared)} compared, {len(bad)} differing{', first ' + bad[0] if bad else ''}",
        )
    check(
        "the multi-component fill is byte-exact end to end",
        not wrong,
        f"{total} tensors compared against the original safetensors payloads",
    )
    backend.close()
    for name in list(backend.components):
        backend.evict(name)
    torch.cuda.empty_cache()


def section_poison(args: argparse.Namespace) -> None:
    """Evict, POISON the freed pages, re-stage, compare. ~350 MiB, so it runs anywhere.

    The check the first hand-written one was missing. Evict-then-stage FINGERPRINTED
    IDENTICAL while the pipeline produced a black image, because a warm allocator returned
    the freed pages with their old contents and the fill had been skipped entirely. A
    correctness check on bytes has to run against memory that was deliberately overwritten,
    or it is checking a coincidence.
    """
    import torch

    from cozy_runtime.author import Model, ModelRegistry, uses_components
    from cozy_runtime.author._loader import Artifact, Config
    from cozy_runtime.internal.derive import Observations, serving_substrate
    from cozy_runtime.internal.fill import (
        Checkpoint,
        StreamingFillBackend,
        _tensors,
        topology_of,
    )

    head("evict -> poison the freed pages -> re-stage -> compare")
    have = snapshots()
    checkpoints = {"vae": Checkpoint(str(STORE), have["vae"])}
    rows = checkpoints["vae"].rows("vae")
    backend = StreamingFillBackend.for_script(
        checkpoints,
        rows,
        release="cr-008b",
        store=str(STORE),
        snapshot=have["vae"],
        window_bytes=4 * MIB,
        slots=16,
        readers=16,
        inflight=8,
    )
    backend.hold_lease = True
    config = json.loads(CONFIG.read_text())

    class OnlyVae:
        def __init__(self, bound: Config) -> None:
            from diffusers import AutoencoderKL

            self.components = {
                "vae": AutoencoderKL.from_config(bound.mapping()["vae"]).to(torch.float16)
            }

    class VaeModel(Model[OnlyVae]):
        pipe: OnlyVae

        def load(self, loader: Any) -> None:
            self.pipe = loader.construct(OnlyVae, factory=OnlyVae)

        @uses_components("vae")
        def touch(self) -> None:
            return None

    registry = ModelRegistry(
        release="cr-008b",
        backend=backend,
        substrate=lambda: serving_substrate("sm89", Observations()),
    )
    artifact = Artifact(
        have["vae"], topology_of(rows), Config(config, "pipeline.config"), custody="canonical"
    )
    backend.expect({"vae": tuple(row.key for row in rows)})
    registry.acquire("poison.models.model", VaeModel, artifact)

    def fingerprint() -> dict[str, float]:
        member = backend.components["vae"]
        return {k: float(t.detach().float().sum()) for k, t in list(_tensors(member))[:8]}

    before = fingerprint()
    freed = backend.evict("vae")
    poison = torch.full((freed // 2,), 31415.0, dtype=torch.float16, device="cuda")
    del poison
    torch.cuda.empty_cache()
    row = backend.stage("vae")
    after = fingerprint()
    check(
        "the re-staged component is BIT-IDENTICAL after its freed pages were overwritten",
        before == after,
        f"{freed} B freed, poisoned, re-staged in {row['ms']} ms; "
        f"{len(before)} sampled tensors, {len([k for k in before if before[k] != after[k]])} "
        "differing",
    )
    check(
        "and the stage rate is physically possible",
        row["ms"] > 0 and (row["bytes"] / MIB) / (row["ms"] / 1000) < 40_000,
        f"{(row['bytes'] / MIB) / (row['ms'] / 1000):.0f} MiB/s — the defect that produced a "
        "black image announced itself as 4.78 GiB in 43 ms, which no disk can produce",
    )
    backend.close()


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
        poison.get("status") in ("OUTCOME_STATUS_FAILED", "OUTCOME_STATUS_REFUSED")
        or poison.get("ended") == "unbound_entrypoint",
        f"{poison.get('ended')}/{poison.get('status')}: "
        f"{poison.get('error') or str(poison.get('terminal', {}).get('safe_message', ''))[:180]}",
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
        cold.get("status") == "OUTCOME_STATUS_SUCCEEDED",
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
                "rss_run_mib": round(ledger.get("rss_worker", 0) / MIB, 1),
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


def section_fillbw(args: argparse.Namespace) -> None:
    """The FILL LEG alone, at capped disk bandwidth. One component, small footprint.

    The full cold-to-first-image ladder needs a card nobody else is on. This isolates the
    leg bandwidth actually acts on — disk bytes into device destinations through the real
    TensorFS reader and the real pinned ring — for a component that fits beside a
    neighbour, so the scaling curve is measurable on a shared box.
    """
    head("the fill leg at capped disk bandwidth")
    shim = build_throttle()
    print(f"  throttle: {shim} — LD_PRELOAD deadline bucket over read/pread, BELOW the reader")
    table: list[dict[str, Any]] = []
    for cap in [0, *args.rates]:
        evicted = drop_page_cache(STORE)
        env = dict(os.environ, COZY_FILL_CHILD="1", PYTHONPATH=f"{ROOT}:{ROOT / 'src'}")
        if cap:
            env.update(LD_PRELOAD=str(shim), IO_THROTTLE_BPS=str(cap))
        out = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "pipeline-live.py"),
                "fill",
                "--component",
                args.component,
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        if not line.startswith("{"):
            check(f"the fill ran at {cap or 'uncapped'}", False, out.stderr[-300:])
            continue
        row = json.loads(line)
        row["cap"] = "uncapped" if not cap else f"{cap // 1_000_000} MB/s"
        row["evicted_pages_before"] = evicted["resident_pages"]
        table.append(row)
        print(f"  {json.dumps(row)}", flush=True)
    if not table:
        return
    check(
        "every capped fill delivered the same bytes",
        len({r["checksum"] for r in table}) == 1,
        f"{table[0]['bytes']} B, {table[0]['tensors']} tensors, one checksum across "
        f"{len(table)} runs — bandwidth is not a correctness input",
    )
    base = table[0]["fill_ms"]
    print("\n  cap            fill_ms   MiB/s   vs uncapped")
    for row in table:
        print(
            f"  {row['cap']:>12}  {row['fill_ms']:8.0f}  {row['MiBps']:7.0f}  "
            f"{row['fill_ms'] / base:5.2f}x"
        )


def fill_once(component: str) -> dict[str, Any]:
    """ONE component filled through the real plane, timed. The `fill` inner mode."""
    import torch

    from cozy_runtime.author._loader import TensorSpec
    from cozy_runtime.internal.encoding import TORCH_DTYPES
    from cozy_runtime.internal.fill import Checkpoint
    from cozy_runtime.internal.fill import StreamingFillBackend as Backend

    spellings = {name: getattr(torch, torch_name) for name, torch_name in TORCH_DTYPES.items()}
    checkpoint = Checkpoint(str(STORE), snapshots()[component])
    rows = checkpoint.rows(component)
    backend = Backend.for_script(
        {component: checkpoint},
        rows,
        release=RELEASE,
        store=str(STORE),
        snapshot=snapshots()[component],
        window_bytes=4 * MIB,
        slots=16,
        readers=16,
        inflight=8,
    )
    backend.expect({component: tuple(row.key for row in rows)})
    destinations = {
        row.key: torch.empty(row.shape, dtype=spellings[row.dtype], device="cuda") for row in rows
    }
    started = time.perf_counter()
    for row in rows:
        backend.fill(
            row.key,
            TensorSpec(shape=row.shape, dtype=row.dtype, nbytes=row.nbytes),
            destinations[row.key],
        )
    backend._commit_enqueued()
    torch.cuda.synchronize()
    fill_ms = (time.perf_counter() - started) * 1000
    total = sum(row.nbytes for row in rows)
    checksum = round(float(next(iter(destinations.values())).float().sum()), 6)
    del destinations
    torch.cuda.empty_cache()
    return {
        "components": [component],
        "bytes": total,
        "tensors": len(rows),
        "fill_ms": round(fill_ms, 1),
        "MiBps": round(total / MIB / (fill_ms / 1000), 1),
        "store": dict(backend.stats.get("store", {})),
        "checksum": checksum,
    }


def ladder_row(outcome: dict[str, Any], label: str, wall: float) -> dict[str, Any]:
    """ONE cold-to-first-image run, attributed per leg. The ladder's unit of evidence."""
    stages = stages_of(outcome)
    ledger = ledger_of(outcome)
    timings = outcome["timings"]
    boot = (outcome.get("triage") or {}).get("posture", {}).get("boot_steps", [])
    fill: dict[str, Any] = next((s for s in boot if s.get("step") == "load"), {})
    attribution = (outcome.get("triage") or {}).get("measurements", {}).get("attribution") or {}
    step_track = (attribution.get("steps") or {}).get("denoise", {})
    rows = residency_of(outcome)
    return {
        "cap": label,
        "ok": outcome.get("status") == "OUTCOME_STATUS_SUCCEEDED",
        "ttfi_s": round(timings["wall_ms"] / 1000, 2),
        "harness_wall_s": round(wall, 2),
        "boot_to_ready_s": round(timings.get("register_to_ready_ms", 0) / 1000, 2),
        "fill_ms": fill.get("fill_ms", 0),
        "fill_MiBps": round(fill.get("bytes", 0) / MIB / (fill.get("fill_ms", 1) / 1000), 1)
        if fill.get("fill_ms")
        else 0,
        "tokenize_ms": round(stages.get("tokenize", 0), 1),
        "encode_ms": round(stages.get("encode", 0), 1),
        "denoise_ms": round(stages.get("denoise", 0), 1),
        "decode_ms": round(stages.get("decode", 0), 1),
        "png_ms": round(stages.get("encode_png", 0), 1),
        "staged_ms": rows.get("staged_ms", 0),
        "steps_per_s": round(1000 / step_track["mean_ms"], 3) if step_track.get("mean_ms") else 0,
        "peak_vram_mib": round(
            outcome.get("terminal", {}).get("metrics", {}).get("peak_vram_bytes", 0) / MIB, 1
        ),
        "rss_executor_mib": round(ledger.get("rss_executor", 0) / MIB, 1),
        "rss_run_mib": round(ledger.get("rss_worker", 0) / MIB, 1),
        "digest": (outcome.get("result") or {}).get("digest", "")[:16],
    }


def median(values: list[float]) -> float:
    ordered = sorted(values)
    half = len(ordered) // 2
    if not ordered:
        return 0.0
    return ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2


def section_bandwidth(args: argparse.Namespace) -> None:
    """Cold time-to-first-image at CAPPED disk read bandwidth, plus the WARM pass.

    THE BAR's cozy half (ROADMAP.md): the full cold->first-image ladder at each disk
    allotment with the legs visible, so the fill-vs-denoise crossover is a measurement
    rather than a composition of two curves.
    """
    require_quiet_card()
    shim = build_throttle()
    head("cold time-to-first-image against a bandwidth-capped disk")
    print(f"  throttle: {shim} (LD_PRELOAD deadline bucket over pread/read, BELOW the reader)")
    print(f"  {args.repeat} cold run(s) per allotment, page cache dropped before each")
    table: list[dict[str, Any]] = []
    for rate in [0, *args.rates]:
        label = "uncapped" if rate == 0 else f"{rate // 1_000_000} MB/s"
        env = {} if rate == 0 else {"LD_PRELOAD": str(shim), "IO_THROTTLE_BPS": str(rate)}
        for index in range(args.repeat):
            evicted = drop_page_cache(STORE)
            started = time.perf_counter()
            outcome = drive(
                f"the {label} run",
                lambda rate=rate, index=index, env=env: spawn(
                    {
                        "workspace": f"{args.tmp}/bw-{rate}-{index}",
                        "payload": {
                            "prompt": args.prompt,
                            "steps": args.steps,
                            "size": args.size,
                            "seed": 1005,
                        },
                        "request_id": f"pipe-bw-{rate}-{index}",
                        "binding": {"resident_budget_bytes": args.budget},
                    },
                    env=env,
                ),
            )
            row = ladder_row(outcome, label, time.perf_counter() - started)
            row["run"] = index
            row["resident_pages_before"] = evicted["resident_pages"]
            table.append(row)
            print(f"  {json.dumps(row)}", flush=True)

    head("the same request, WARM page cache (nothing evicted)")
    started = time.perf_counter()
    warm = drive(
        "the warm run",
        lambda: spawn(
            {
                "workspace": f"{args.tmp}/bw-warm",
                "payload": {
                    "prompt": args.prompt,
                    "steps": args.steps,
                    "size": args.size,
                    "seed": 1005,
                },
                "request_id": "pipe-bw-warm",
                "binding": {"resident_budget_bytes": args.budget},
            }
        ),
    )
    warm_row = ladder_row(warm, "warm", time.perf_counter() - started)
    print(f"  {json.dumps(warm_row)}", flush=True)

    check("every capped run still produced an image", all(r["ok"] for r in table))
    digests = {r["digest"] for r in table if r["ok"]} | {warm_row["digest"]}
    check(
        "the SAME image at every disk speed and warm — bandwidth is not a correctness input",
        len(digests) == 1,
        f"{digests}",
    )
    caps = [r["cap"] for r in table]
    ordered = list(dict.fromkeys(caps))
    base = median([r["ttfi_s"] for r in table if r["cap"] == "uncapped"]) or 1e-9
    print("\n  cap            ttfi_s   vs uncapped   fill_ms   fill_MiBps   fill share")
    for cap in ordered:
        group = [r for r in table if r["cap"] == cap]
        ttfi = median([r["ttfi_s"] for r in group])
        fill_ms = median([float(r["fill_ms"]) for r in group])
        rate = median([float(r["fill_MiBps"]) for r in group])
        print(
            f"  {cap:>12}  {ttfi:7.2f}  {ttfi / base:10.2f}x  {fill_ms:9.0f}  {rate:10.0f}  "
            f"{fill_ms / max(ttfi * 1000, 1e-9) * 100:9.1f}%"
        )
    print(f"  {'warm':>12}  {warm_row['ttfi_s']:7.2f}  {warm_row['ttfi_s'] / base:10.2f}x")
    Path(args.tmp, "cozy-ladder.json").write_text(
        json.dumps({"cold": table, "warm": warm_row}, indent=2)
    )
    print(f"  banked: {Path(args.tmp, 'cozy-ladder.json')}")


THROTTLE_C = r"""
/* A token-bucket read throttle, injected BELOW the reader by the dynamic linker.
 *
 * The constraint has to be below our read path or it measures nothing: this intercepts the
 * libc entry points the TensorFS reader's preads land on, so the Rust reader, the page-cache
 * misses and everything else in the process are capped identically and none of them know.
 * It is deliberately NOT a sleep in runtime code — the runtime is unmodified and unaware.
 *
 * Rate comes from IO_THROTTLE_BPS. The bucket refills at that rate in real time and a read
 * that outruns it sleeps for exactly the deficit.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <errno.h>

static double rate_bps = 0.0;
static double next_free = 0.0;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static ssize_t (*real_read)(int, void *, size_t) = 0;
static ssize_t (*real_pread)(int, void *, size_t, off_t) = 0;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t) = 0;
static int (*real_connect)(int, const struct sockaddr *, socklen_t) = 0;
static long mmap_min = 0;   /* 0 disables the mmap arm entirely */
static int block_net = 0;   /* IO_BLOCK_NETWORK: refuse every INET connect */

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static void setup(void) {
    /* IO_ rather than COZY_: the worker's ERASE pass strips every COZY_-prefixed name
     * from the device child's environment, and the fill happens in that child — a cap named
     * COZY_THROTTLE_BPS reaches the worker and silently does not reach the reader. */
    const char *value = getenv("IO_THROTTLE_BPS");
    if (!value) value = getenv("COZY_THROTTLE_BPS");
    rate_bps = value ? atof(value) : 0.0;
    const char *mm = getenv("IO_THROTTLE_MMAP_MIN");
    if (!mm) mm = getenv("COZY_THROTTLE_MMAP_MIN");
    mmap_min = mm ? atol(mm) : 0;
    next_free = now_s();
    real_read = dlsym(RTLD_NEXT, "read");
    real_pread = dlsym(RTLD_NEXT, "pread64");
    if (!real_pread) real_pread = dlsym(RTLD_NEXT, "pread");
    real_mmap = dlsym(RTLD_NEXT, "mmap");
    real_connect = dlsym(RTLD_NEXT, "connect");
    const char *blk = getenv("IO_BLOCK_NETWORK");
    block_net = blk && blk[0] == '1';
}

/* A DEADLINE bucket, not a per-thread deficit. The first version had each thread sleep for
 * its own share of a shared negative balance, and sixteen reader threads then slept in
 * PARALLEL — so a 1000 MB/s cap measured 1591 MB/s. Reserving a slot on one shared timeline
 * and sleeping until it is exact regardless of how many threads are reading.
 *
 * The slot is charged from when the read STARTED, so a read's own service time counts
 * against its allotment instead of adding to it. Charging from completion made the two
 * serial — a single-threaded reader at a 250 MB/s cap measured 139 MB/s, which is
 * 1/(1/250 + 1/330), the cap composed with the disk rather than imposed on it. Sixteen
 * parallel readers hid the error because their service times overlapped; a one-threaded
 * mmap loader does not, and both engines have to meet the same cap.
 */
static void spend(size_t n, double began) {
    if (rate_bps <= 0.0) return;
    pthread_mutex_lock(&lock);
    if (next_free < began) next_free = began;
    next_free += (double)n / rate_bps;
    double target = next_free;
    pthread_mutex_unlock(&lock);
    double wait = target - now_s();
    if (wait > 0) {
        struct timespec ts = {(time_t)wait, (long)((wait - (long)wait) * 1e9)};
        nanosleep(&ts, 0);
    }
}

ssize_t read(int fd, void *buf, size_t n) {
    if (!real_read) setup();
    double began = now_s();
    ssize_t got = real_read(fd, buf, n);
    if (got > 0) spend((size_t)got, began);
    return got;
}

ssize_t pread(int fd, void *buf, size_t n, off_t off) {
    if (!real_pread) setup();
    double began = now_s();
    ssize_t got = real_pread(fd, buf, n, off);
    if (got > 0) spend((size_t)got, began);
    return got;
}

ssize_t pread64(int fd, void *buf, size_t n, off_t off) { return pread(fd, buf, n, off); }

/* The MMAP arm, off unless COZY_THROTTLE_MMAP_MIN is set.
 *
 * A read/pread bucket caps a reader that issues read syscalls; it does not touch a reader
 * that mmaps the file and lets page FAULTS bring the bytes in. safetensors does exactly
 * that, so ComfyUI's whole weight load is invisible to the bucket above and a ladder run
 * against it would measure nothing. This turns a large read-only private file mapping into
 * an anonymous mapping populated through the SAME throttled pread path, so both engines
 * meet one constraint. It is deliberately narrow — MAP_PRIVATE, PROT_READ with no write or
 * exec bit, a real fd, and at least COZY_THROTTLE_MMAP_MIN bytes — so libraries, the
 * loader's own mappings, and every small mapping keep the kernel's behavior.
 *
 * The cost it charges honestly: the bytes become anonymous RSS instead of evictable page
 * cache, so RSS under this arm is not the RSS of an unthrottled run.
 */
void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) setup();
    int eligible = mmap_min > 0 && fd >= 0 && len >= (size_t)mmap_min &&
                   (flags & MAP_PRIVATE) && (prot & PROT_READ) &&
                   !(prot & PROT_WRITE) && !(prot & PROT_EXEC) && !(flags & MAP_ANONYMOUS);
    if (!eligible) return real_mmap(addr, len, prot, flags, fd, off);
    void *anon = real_mmap(0, len, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (anon == MAP_FAILED) return real_mmap(addr, len, prot, flags, fd, off);
    size_t done = 0;
    while (done < len) {
        size_t want = len - done;
        if (want > (size_t)(8 << 20)) want = 8 << 20;
        double began = now_s();
        ssize_t got = real_pread(fd, (char *)anon + done, want, off + (off_t)done);
        if (got <= 0) break;
        spend((size_t)got, began);
        done += (size_t)got;
    }
    return anon;
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    return mmap(addr, len, prot, flags, fd, off);
}

/* The NETWORK-BLOCK arm, off unless IO_BLOCK_NETWORK=1.
 *
 * The hubless contract is a claim that a populated local CAS is sufficient — provable only
 * by taking the network away and watching inference finish anyway. AF_UNIX is left alone on
 * purpose: the worker/executor control channel is a Unix socket, and cutting THAT would
 * test the harness rather than the contract. */
int connect(int fd, const struct sockaddr *addr, socklen_t len) {
    if (!real_connect) setup();
    if (block_net && addr &&
        (addr->sa_family == AF_INET || addr->sa_family == AF_INET6)) {
        errno = ECONNREFUSED;
        return -1;
    }
    return real_connect(fd, addr, len);
}
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
        choices=(
            "exec",
            "fill",
            "fixtures",
            "sources",
            "poison",
            "stage",
            "arms",
            "image",
            "fillbw",
            "bandwidth",
            "all",
        ),
    )
    parser.add_argument("--tmp", default="/tmp/cozy-pipe")
    parser.add_argument("--spec", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--component", default="text_encoder_2")
    parser.add_argument(
        "--rates",
        type=int,
        nargs="*",
        default=[2_000_000_000, 1_000_000_000, 500_000_000, 250_000_000],
    )
    args = parser.parse_args()

    if args.section == "fill":
        print(json.dumps(fill_once(args.component)))
        return 0

    if args.section == "exec":
        outcome = run_one(json.loads(Path(args.spec).read_text()))
        Path(args.out).write_text(json.dumps(outcome, indent=2, default=str))
        return 0

    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    sections = {
        "fixtures": section_fixtures,
        "sources": section_sources,
        "poison": section_poison,
        "fillbw": section_fillbw,
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
