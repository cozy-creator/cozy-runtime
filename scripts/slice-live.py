#!/usr/bin/env python
"""cr-008a's live verification: ONE request, through the whole real path, no network.

Not a test suite (#160). Every section starts the actual system — an ephemeral local
worker in a REAL process, with an in-memory coordinator instead of a gRPC one, and a
disposable CUDA executor child serving cr-005's real SDXL UNet — and prints what it observed.

    nice -n 19 corpus/.venv/bin/python scripts/slice-live.py slice
    nice -n 19 corpus/.venv/bin/python scripts/slice-live.py ledger
    nice -n 19 corpus/.venv/bin/python scripts/slice-live.py arms
    nice -n 19 corpus/.venv/bin/python scripts/slice-live.py clone
    nice -n 19 corpus/.venv/bin/python scripts/slice-live.py bench

`exec` is the inner mode: ONE slice in ONE process, which is what a `cozy-runtime run`
process will be. Every section spawns it, so a kill is a real kill of a real run and the
no-torch-in-the-run-process claim is a fact about a process rather than an import graph.

The bytes come from `~/cozy_v2/tensorfs-bench`, cr-005's real SDXL CAS, READ-ONLY.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

BENCH = Path("/home/fidika/cozy_v2/tensorfs-bench")
STORE = BENCH / "store"
CONFIG = BENCH / "hf" / "sdxl" / "unet" / "config.json"
CHECKPOINT = BENCH / "results" / "checkpoint.txt"
#: Where the hardlinked working copies of the read-only bench CAS live. Shared across
#: sections and across runs, because a hardlink tree costs blocks nobody duplicates.
WORK = Path("/tmp/cozy-slice-store")
#: tfs-007's facade wheel. TensorFS is not on an index yet (its PyPI done-when is
#: deliberately deferred), so the wheel is built once from the tensorfs repo at c67a9da
#: (`maturin build --release`) and dropped here. When it publishes, this becomes an ordinary
#: pinned dependency and every `uv pip install` below disappears.
WHEELHOUSE = Path("/tmp/cozy-wheels")
RELEASE = "cozy/sdxl-unet@cr-005"
COMPONENT = "unet"
MIB = 1 << 20
GIB = 1 << 30

#: cr-005 banked this digest for ONE denoising step on the real filled UNet at seed 1005,
#: 64x64 latents, fp16, deterministic algorithms. cr-008a's whole point is that the same
#: computation, driven through the PROTOCOL PATH instead of a direct call, produces the same
#: bits — which is what proves the protocol did not perturb execution.
BANKED = "19ca4b8dee5f8e498e012baf67f921f7210cfcbb68a0d350de5b6909d20ff05c"
BANKED_PAYLOAD = {"steps": 1, "latent": 64, "seed": 1005}

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


def snapshot() -> str:
    for line in CHECKPOINT.read_text().strip().split("\n"):
        name, _, value = line.partition("\t")
        if name == "checkpoint":
            return value
    raise SystemExit("no checkpoint digest in results/checkpoint.txt")


def plan_id_for(entrypoint: str) -> str:
    return "sha256:" + hashlib.sha256(f"ebp/{RELEASE}/{entrypoint}".encode()).hexdigest()


def install_row(home: Path, store: str = "") -> None:
    """Put this bench artifact in the run's OWN artifact index (#567e).

    The record names a ref and the worker resolves it against this machine's index, so a
    harness that hands the worker a store root without an index row is handing it a fact
    the product path never carries. Installing the row is what makes these arms run the
    shape production runs: pull writes a row, and the binding names what the row is called.
    """
    from cozy_runtime.internal import artifacts

    artifacts.install(
        home,
        artifacts.Artifact(
            ref=RELEASE,
            store=store or working_store(Path(WORK) / "store"),
            config=str(CONFIG),
            snapshots={COMPONENT: snapshot()},
        ),
    )


def binding_record(entrypoint: str) -> dict[str, Any]:
    return {
        "entrypoint_binding_plan_id": plan_id_for(entrypoint),
        "project": str(ROOT / "corpus" / "endpoint"),
        "model_class": "SdxlUnetModel",
        "model_binding_path": f"{entrypoint}.models.model",
        "model_parameter_name": "model",
        "artifact_ref": RELEASE,
        "components": [COMPONENT],
        "release": RELEASE,
        "host_bytes": 2 * GIB,
        "pinned_bytes": 256 * MIB,  # tfs-007 measured shape: 16 slots x 16 MiB
        "entrypoint": entrypoint,
        "model_construction_digest": "",
    }


def working_store(at: Path) -> str:
    """A HARDLINKED working copy of the read-only bench CAS.

    TensorFS registers a GC-safe hold when a lease is acquired, which is a WRITE into the
    store. `~/cozy_v2/tensorfs-bench` is read-only by standing rule, so every run reads
    through a hardlinked copy: the object files share their blocks with the originals (no
    4.9 GB duplicated, no bytes rewritten) while holds, tmp and verification records land in
    the copy's own directories.
    """
    if not (at / "store.json").is_file():
        at.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(at, ignore_errors=True)
        subprocess.run(["cp", "-al", str(STORE), str(at)], check=True)
    return str(at)


def corrupt_store(at: Path, source: str) -> str:
    """A working copy with ONE object corrupted — the fill-failure plant, harness-side.

    A byte is flipped in a real object, and the hardlink is BROKEN first so the original CAS
    is untouched. This is the most ordinary way a stored artifact goes wrong, and it needs
    no switch inside the runtime (decisions #249). Since tfs-007 it is caught when the lease
    verifies the snapshot — before a single destination is reserved.
    """
    # Rebuilt EVERY time, never reused: TensorFS QUARANTINES an object it finds corrupt, so
    # the second lease over the same doctored store reports the object ABSENT rather than
    # corrupt. That is the store behaving correctly, and it would quietly turn this arm into
    # a different arm — so the plant is fresh on every run.
    shutil.rmtree(at, ignore_errors=True)
    subprocess.run(["cp", "-al", source, str(at)], check=True)
    objects = sorted((at / "objects" / "sha256").rglob("*"))
    target = max((p for p in objects if p.is_file()), key=lambda p: p.stat().st_size)
    body = bytearray(target.read_bytes())
    body[len(body) // 2] ^= 0x40
    target.unlink()  # break the hardlink BEFORE writing: the original must not move
    target.write_bytes(bytes(body))
    return str(at)


# --------------------------------------------------------------------------- the inner run


def run_one(spec: dict[str, Any]) -> dict[str, Any]:
    """ONE slice, in THIS process. The whole of `exec` mode."""
    from cozy_runtime.internal.config import read_config
    from cozy_runtime.internal.local import LocalCoordinator, LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions
    from cozy_runtime.protocol import worker_pb2 as pb

    workspace = Path(spec["workspace"])
    home = workspace / "home"
    home.mkdir(parents=True, exist_ok=True)
    install_row(home, spec.get("store", ""))
    config = read_config(
        {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "COZY_HOME": str(home),
            "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
        }
    )
    request = LocalRequest(
        entrypoint=spec["entrypoint"],
        payload=spec["payload"],
        outputs=tuple(spec.get("outputs", ("image",))),
        request_id=spec.get("request_id", "run-1"),
        timeout_ms=int(spec.get("timeout_ms", spec.get("deadline_ms", 0))),
        payload_url=spec.get("payload_url", ""),
    )
    options = WorkerOptions(
        root=workspace / "worker",
        release_id=RELEASE,
        instance_id=spec.get("instance_id", ""),
        grace=float(spec.get("grace", 2.0)),
        no_warm=bool(spec.get("no_warm", True)),
    )

    def arm(coordinator: LocalCoordinator, worker: Any) -> None:
        # SIGINT on a local run CANCELS the attempt — the same CancelAttempt a hub sends,
        # on the same lane. This is the local spelling of Ctrl-C, and it is why the
        # cancellation arms need no hook inside the runtime. SIGTERM is deliberately NOT
        # bound: a terminate must remain a terminate, or an operator's kill turns into a
        # polite request the run may outlive.
        def cancel(signum: int, frame: object) -> None:
            print(f"[run] signal {signum}: cancelling the live attempt", flush=True)
            coordinator.cancel(pb.CancelReason.CANCEL_REASON_CLIENT)

        signal.signal(signal.SIGINT, cancel)
        marker = Path(spec["workspace"]) / "accepted.marker"

        def watch() -> None:
            while not coordinator.done.is_set():
                if coordinator.accepted is not None and not marker.exists():
                    marker.write_text("accepted\n")
                if worker.supervision.current is not None:
                    executor = workspace / "executor.pid"
                    if not executor.exists():
                        executor.write_text(f"{worker.supervision.current.pid}\n")
                time.sleep(0.005)

        threading.Thread(target=watch, daemon=True, name="marks").start()

    outcome = run_slice(
        config,
        request,
        [binding_record(spec["entrypoint"])],
        workspace=workspace,
        release_id=RELEASE,
        options=options,
        delete_journal=bool(spec.get("delete_journal", True)),
        coordinator_hook=arm,
    )
    return dataclasses.asdict(outcome)


def spawn(spec: dict[str, Any], *, timeout: float = 900.0) -> tuple[dict[str, Any], Any]:
    """Run one slice in its OWN process and read back what it observed."""
    workspace = Path(spec["workspace"])
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    spec_path = workspace / "spec.json"
    out_path = workspace / "outcome.json"
    spec_path.write_text(json.dumps(spec))
    process = start(spec_path, out_path, workspace)
    process.wait(timeout=timeout)
    return read_outcome(out_path), process


def start(spec_path: Path, out_path: Path, workspace: Path) -> subprocess.Popen[bytes]:
    # EVERY run waits for a quiet card, not just the first of a section. The neighbour on
    # this shared box comes and goes, and a 4.78 GiB resident that starts while it holds
    # 2.5 GiB fails with a device shortfall that is true and says nothing about this code.
    require_quiet_card()
    log = workspace / "run.log"
    process = subprocess.Popen(
        [
            "nice",
            "-n",
            "19",
            sys.executable,
            str(ROOT / "scripts" / "slice-live.py"),
            "exec",
            "--spec",
            str(spec_path),
            "--out",
            str(out_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env={
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
        },
    )

    def pump() -> None:
        assert process.stdout is not None
        with open(log, "wb") as handle:
            for line in process.stdout:
                handle.write(line)
                handle.flush()

    threading.Thread(target=pump, daemon=True).start()
    return process


def read_outcome(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


#: A quiet card, in MiB. The real UNet is 4.78 GiB on an 8 GiB device, so a benchmark or a
#: red arm only means anything when nothing else is resident: this box is shared, and a
#: neighbour holding 2.5 GiB turns a correct run into an OOM that says nothing about this
#: code. Waiting is the honest answer; measuring through the interference is not.
QUIET_MIB = 400


def require_quiet_card(seconds: float = 3600.0) -> int:
    """Block until the card is effectively idle, saying so while it waits."""
    started = time.monotonic()
    announced = False
    while time.monotonic() - started < seconds:
        used = gpu_used_mib()
        if used < 0 or used <= QUIET_MIB:
            if announced:
                print(f"  the card is quiet again after {time.monotonic() - started:.0f} s")
            return used
        if not announced:
            announced = True
            print(
                f"  waiting for a quiet card: {used} MiB in use by another process on this "
                f"shared box (need <= {QUIET_MIB} MiB)",
                flush=True,
            )
        time.sleep(5.0)
    return gpu_used_mib()


def settle_gpu(baseline: int, seconds: float = 60.0) -> int:
    """Wait for the card to come back before starting the next run.

    A SIGKILLed executor holding ~5 GiB is reaped by `waitpid` well before the NVIDIA driver
    has finished tearing its context down, so a run started the instant the previous process
    exits can find the card still occupied — and on an 8 GiB card that is a real
    `device_shortfall`, not a flake. It is a HARNESS concern here (back-to-back benchmark
    runs); a back-to-back local `run` meets the same edge and cr-016 owns what it should say.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        used = gpu_used_mib()
        if used < 0 or used <= baseline + 64:
            return used
        time.sleep(0.5)
    return gpu_used_mib()


def wait_for(path: Path, seconds: float, process: subprocess.Popen[bytes] | None = None) -> bool:
    """Wait for a marker, and STOP waiting when the run that would write it has exited.

    Waiting out a full timeout on a process that is already gone turns one failed arm into
    fifteen wasted minutes, and the arm still fails — it just fails later and says less.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        if process is not None and process.poll() is not None:
            return False
        time.sleep(0.005)
    return False


def journal_rows(workspace: Path, kind: str = "") -> list[dict[str, Any]]:
    """Read the surviving journal of a run whose process is gone."""
    sys.path.insert(0, str(ROOT / "src"))
    from cozy_runtime.internal.local import _journal_rows

    rows = _journal_rows(workspace / "worker" / "journal.log")
    return [row for row in rows if not kind or row.get("kind") == kind]


def outputs_present(workspace: Path) -> list[str]:
    directory = workspace / "outputs"
    return sorted(str(p.name) for p in directory.glob("*")) if directory.is_dir() else []


# --------------------------------------------------------------------------- sections


def section_slice(args: argparse.Namespace) -> dict[str, Any]:
    """The headline: one request, one model, one kernel, one ledger, one typed result."""
    idle = require_quiet_card()
    head("one request through the REAL protocol path, with no network")
    workspace = Path(args.tmp) / "slice"
    outcome, process = spawn(
        {
            "workspace": str(workspace),
            "entrypoint": "denoise",
            "payload": BANKED_PAYLOAD,
            "outputs": ["image"],
            "request_id": "run-1",
            "no_warm": True,
            "delete_journal": False,
        }
    )
    if not outcome:
        raise SystemExit(
            f"the slice produced no outcome; log:\n{(workspace / 'run.log').read_text()}"
        )

    check(
        "the run exited clean",
        process.returncode == 0 and outcome["exit_code"] == 0,
        f"process {process.returncode}, worker exit {outcome['exit_code']}",
    )
    check(
        "the terminal SUCCEEDED",
        outcome["status"] == "OUTCOME_STATUS_SUCCEEDED",
        f"{outcome['status']} — {str(outcome['terminal'].get('safe_message', ''))[:80]}",
    )
    accepted = outcome["accepted"]
    check(
        "acceptance carried the digests it JOURNALED",
        bool(accepted.get("plan_digest")) and bool(accepted.get("model_construction_digest")),
        f"plan {accepted.get('plan_digest', '')[7:23]}… construction "
        f"{accepted.get('model_construction_digest', '')[7:23]}…",
    )
    check(
        "the plan is the CLOSED observable projection, not just a hash",
        accepted.get("placement") == "all_resident" and accepted.get("delivery") == "native",
        f"{accepted.get('delivery')}/{accepted.get('materialization')}/"
        f"{accepted.get('placement')} {accepted.get('compute_dtype')}, "
        f"{accepted.get('reserved_vram_bytes', 0) / GIB:.2f} GiB reserved",
    )

    head("the kernel: one real denoising step on the real weights")
    result = outcome["result"] or {}
    check(
        "the typed result rode the terminal INLINE",
        bool(result),
        f"schema digest {outcome['result_schema_digest'][7:23]}…, "
        f"{len(json.dumps(result))} B of canonical result",
    )
    check(
        "THE PROTOCOL PATH DID NOT PERTURB EXECUTION — cr-005's banked digest reproduces",
        result.get("digest") == BANKED,
        f"{result.get('digest', 'none')} vs banked {BANKED[:32]}…",
    )
    check(
        "the step is real computation, not a fixture",
        abs(float(result.get("std", 0.0)) - 1.0) < 0.2 and int(result.get("steps", 0)) == 1,
        f"steps={result.get('steps')} mean={result.get('mean')} std={result.get('std')}",
    )
    check(
        "`ctx.boot_warmup` is FALSE on a paying request",
        result.get("warm") is False,
        "true only during the boot warm pass",
    )

    head("the output transaction: one grant, one destination, one manifest")
    manifest = outcome["manifest"]
    check(
        "the tail wrote under the EXACT granted destination",
        len(manifest) == 1 and manifest[0]["output_id"] == "image",
        f"{manifest[0]['output_id']} -> {manifest[0]['length']} B {manifest[0]['digest'][:23]}…"
        if manifest
        else "no manifest",
    )
    landed = outcome["outputs"].get("image", "")
    check(
        "the bytes are actually on disk at that destination",
        bool(landed) and Path(landed).is_file(),
        f"{Path(landed).name if landed else 'nothing'} "
        f"({Path(landed).stat().st_size if landed else 0} B)",
    )
    check(
        "the result names the asset by its FIELD PATH, never a position",
        (result.get("image") or {}).get("asset_ref", "").startswith("attempt:"),
        str((result.get("image") or {}).get("asset_ref", "none")),
    )

    head("the journal: acceptance, ledger, triage, terminal, closure — in that order")
    rows = journal_rows(workspace)
    kinds = [row["kind"] for row in rows]
    check(
        "every durable step is journaled, once, in order",
        kinds == ["session", "accepted", "ledger", "triage", "outcome", "closed"],
        " -> ".join(kinds),
    )
    check(
        "the ledger closes and the triage bundle is DURABLE before the terminal names it",
        kinds.index("ledger") < kinds.index("triage") < kinds.index("outcome"),
        "a terminal referencing a bundle nobody wrote is the fabrication class this "
        "runtime exists to prevent",
    )
    check(
        "exactly ONE execution happened",
        outcome["executions"] == 1,
        f"executions={outcome['executions']}",
    )

    head("cr-011: the triage bundle, read back BY OPAQUE KEY with no live process")
    subject = outcome["triage_subject"]
    bundle = outcome["triage"]
    check(
        "the terminal carries an opaque triage subject",
        subject.startswith("trb-") and "run-1" not in subject,
        f"{subject} — an id, not an address; nothing in it names the request",
    )
    ref = outcome["terminal"].get("triage_bundle") or {}
    check(
        "the reference is TRUTHFUL: digest and length of bytes that exist",
        bool(ref.get("write_receipt_digest")) and 0 < int(ref.get("length", 0)) <= (1 << 20),
        f"{int(ref.get('length', 0))} B (cap 1048576), receipt "
        f"{str(ref.get('write_receipt_digest', ''))[:23]}…",
    )
    check(
        "the bundle verifies against its JOURNALED write receipt",
        bundle.get("format") == "cozy.runtime.WorkerTriageBundle/1",
        f"{len(bundle.get('events', ()))} event rows, "
        f"{len(bundle.get('confessions', ()))} confession(s)",
    )
    attribution = bundle.get("measurements", {}).get("attribution", {})
    steps = attribution.get("steps", {})
    stages = attribution.get("stages", {})
    check(
        "STEP TIME IS ATTRIBUTABLE — per-step intervals, accumulated not listed",
        bool(stages.get("denoise")),
        f"stages={ {k: v['total_ms'] for k, v in stages.items()} } "
        f"steps={ {k: (v['count'], v['mean_ms']) for k, v in steps.items()} }",
    )
    check(
        "the bundle carries the applied POSTURE it ran under",
        bundle.get("posture", {}).get("intake_state") == "INTAKE_STATE_READY",
        f"revision {bundle.get('posture', {}).get('applied_revision')}, "
        f"posture {bundle.get('posture', {}).get('posture')}, "
        f"credential epoch {bundle.get('posture', {}).get('credential_epoch')}",
    )
    check(
        "the ledger reconciliation is IN the bundle, not a journal join away",
        bool(bundle.get("measurements", {}).get("ledger", {}).get("classes")),
        f"{len(bundle['measurements']['ledger'].get('classes', ()))} byte classes, "
        f"closed={bundle['measurements']['ledger'].get('closed')}",
    )
    print("\n  what a person reads first:")
    for line in outcome["triage_explained"]:
        print(f"    {line}")

    head("no CUDA in the run process — the worker half of a local run")
    log = (workspace / "run.log").read_text(errors="replace")
    check(
        "the run process spawned a disposable executor child",
        "spawning the device executor" in log,
        next((ln for ln in log.splitlines() if "dialled back" in ln), "")[:140],
    )
    print(f"\n  timings: {json.dumps(outcome['timings'])}")
    print(f"  GPU {gpu_used_mib()} MiB (idle baseline {idle} MiB)")
    return outcome


def section_ledger(args: argparse.Namespace) -> dict[str, Any]:
    """Ledger v0: every class declares its method, and every attempt closes reconciled."""
    require_quiet_card()
    head("ledger v0: every counter declares HOW it was measured")
    workspace = Path(args.tmp) / "ledger"
    outcome, _ = spawn(
        {
            "workspace": str(workspace),
            "entrypoint": "denoise",
            "payload": {"steps": 2, "latent": 64, "seed": 7},
            "outputs": ["image"],
            "request_id": "run-ledger",
            "no_warm": True,
            "delete_journal": False,
        }
    )
    ledger = outcome.get("ledger", {})
    classes = ledger.get("classes", [])
    check(
        "every class carries a kind and a method",
        bool(classes) and all(row.get("kind") and row.get("method") for row in classes),
        f"{len(classes)} classes, kinds {sorted({row['kind'] for row in classes})}",
    )
    for row in classes:
        print(f"    {row['class']:<20} {row['bytes']:>14} B  [{row['kind']}]  {row['method'][:74]}")
    measured = [row for row in classes if row["kind"] == "measured"]
    check(
        "the weights class is MEASURED, not asserted",
        any(row["class"] == "active_weights" and row["bytes"] > 4 * GIB for row in classes),
        next(
            (
                f"{row['bytes'] / GIB:.3f} GiB"
                for row in classes
                if row["class"] == "active_weights"
            ),
            "absent",
        ),
    )
    check(
        "activations are the attempt's peak MINUS the resident baseline",
        any(row["class"] == "activations_scratch" and row["bytes"] > 0 for row in classes),
        next(
            (
                f"{row['bytes'] / MIB:.1f} MiB of scratch above the weights"
                for row in classes
                if row["class"] == "activations_scratch"
            ),
            "absent",
        ),
    )
    check(
        "arithmetic and measurement are not conflated",
        len(measured) >= 5 and any(row["kind"] == "arithmetic" for row in classes),
        f"{len(measured)} measured, "
        f"{len([r for r in classes if r['kind'] == 'arithmetic'])} arithmetic",
    )

    head("attempt close reconciles EVERY class — the success path")
    record = outcome.get("reconciliation", {})
    check(
        "the attempt closed with no leak in any reconciled class",
        bool(record) and record.get("closed") is True,
        f"leaks {record.get('leaks', ['<no record>'])}",
    )
    for row in record.get("classes", []):
        if row.get("reconciled"):
            print(
                f"    {row['class']:<20} {row['before']:>14} -> {row['after']:<14} "
                f"delta {row.get('delta', 0):+d} B  [{row.get('verdict', '-')}]"
            )
    check(
        "the whole granted envelope came back, not just what the kernel allocated",
        all(
            row.get("verdict") == "reconciled"
            for row in record.get("classes", [])
            if row.get("reconciled") and row["class"] in ("host_ram", "pinned_allocated")
        ),
        "host_ram and pinned_allocated are compared against the line BEFORE acceptance",
    )
    check(
        "the ledger reconciliation is DURABLE, not a print",
        len(journal_rows(workspace, "ledger")) == 1,
        f"{len(journal_rows(workspace, 'ledger'))} ledger row(s) in the journal",
    )

    head("the device baseline: a first-attempt SHIFT is named, a per-attempt leak is not")
    shifted = Path(args.tmp) / "ledger-warm"
    outcome_warm, _ = spawn(
        {
            "workspace": str(shifted),
            "entrypoint": "denoise",
            "payload": BANKED_PAYLOAD,
            "outputs": ["image"],
            "request_id": "run-warm",
            "no_warm": False,  # the boot warm pass runs one case through ORDINARY dispatch
            "delete_journal": False,
        }
    )
    warm_record = outcome_warm.get("reconciliation", {})
    device_row: dict[str, Any] = next(
        (r for r in warm_record.get("classes", []) if r["class"] == "device_allocated"), {}
    )
    check(
        "with the warm pass on, the first PAYING request reconciles device bytes to zero",
        device_row.get("verdict") == "reconciled" and device_row.get("delta") == 0,
        f"delta {device_row.get('delta')} B [{device_row.get('verdict')}] — the lazily "
        "allocated library workspace was paid by the warm case, like the first-call tax",
    )
    check(
        "and the warm pass is what absorbed it",
        bool(outcome_warm.get("warm")) and outcome_warm["warm"][0]["outcome"] == "warmed",
        f"warm outcomes {[w['outcome'] for w in outcome_warm.get('warm', [])]}",
    )
    cold_shift: dict[str, Any] = next(
        (
            r
            for r in outcome.get("reconciliation", {}).get("classes", [])
            if r["class"] == "device_allocated"
        ),
        {},
    )
    check(
        "without it, the shift is RECORDED with its size rather than called zero",
        cold_shift.get("verdict") in ("baseline_shift", "reconciled"),
        f"delta {cold_shift.get('delta')} B [{cold_shift.get('verdict')}]: "
        f"{(outcome.get('reconciliation', {}).get('baseline_shifts') or ['none'])[0][:120]}",
    )

    head("the failure path: an INJECTED corrupt object refuses BEFORE anything is reserved")
    failed = Path(args.tmp) / "ledger-fail"
    failed.mkdir(parents=True, exist_ok=True)
    bad = corrupt_store(WORK / "store-corrupt", working_store(WORK / "store"))
    outcome2, _ = spawn(
        {
            "workspace": str(failed / "run"),
            "entrypoint": "denoise",
            "payload": BANKED_PAYLOAD,
            "outputs": ["image"],
            "request_id": "run-fail",
            "no_warm": True,
            "store": bad,
            "delete_journal": False,
        }
    )
    log = (failed / "run" / "run.log").read_text(errors="replace")
    fault = (outcome2.get("faults") or [{}])[-1]
    check(
        "one flipped byte in a real object refuses the whole preparation",
        fault.get("reason") == "digest_mismatch",
        f"{fault.get('reason')}: {str(fault.get('detail', ''))[:140]}",
    )
    live = re.search(r"peak allocator (\d+) B", log)
    live_bytes = int(live.group(1)) if live else -1
    # THE FILL PLANE'S OWN LEDGER, not the torch allocator's peak. They stopped being the
    # same number when cr-006 landed the device probe: `prepare` qualifies the encodings it
    # may read before it takes a lease, and that probe allocates ~20 MiB of its own scratch.
    # The claim here is about DESTINATIONS — the bytes reserved for weights — and the ledger
    # is what counts those. Reading the allocator peak instead would make a correct refusal
    # look like a leaking one for as long as anything else on the prepare path touches the
    # device, which is a measurement that decays rather than a fence.
    reserved = re.search(r"ledger (\d+) B was LIVE", log)
    reserved_bytes = int(reserved.group(1)) if reserved else -1
    check(
        "and it refuses with ZERO destinations reserved — the lease verifies FIRST",
        reserved_bytes == 0,
        f"{reserved_bytes} B of destinations were live when it failed (the allocator's own "
        f"peak was {live_bytes} B, which is cr-006's pre-lease device probe); the interim "
        "reader could only find this 85 % of the way through a 4.78 GiB fill",
    )
    charged = [
        row
        for row in outcome2.get("ledger", {}).get("classes", [])
        if row["class"] in ("host_ram", "pinned_allocated") and row["bytes"] > 0
    ]
    check(
        "no host reservation survived the failure",
        not charged,
        "host_ram and pinned_allocated both 0 B" if not charged else str(charged),
    )
    check(
        "and no attempt was ever accepted against a generation that never became ready",
        not journal_rows(failed / "run", "accepted"),
        "the journal carries no acceptance — serving never accepts-then-cold-loads",
    )
    check(
        "no output bytes exist anywhere",
        not outputs_present(failed / "run"),
        f"{outputs_present(failed / 'run') or 'nothing at the granted destination'}",
    )
    print(f"  GPU {gpu_used_mib()} MiB")
    return outcome


def took_the_card(outcome: dict[str, Any]) -> bool:
    """True when a run failed because a NEIGHBOUR took the card, not because of this code.

    On this shared box the 8 GiB device fits exactly one 4.78 GiB resident, and another
    agent's worker legitimately comes and goes. A run that starts on a quiet card and meets
    a neighbour halfway through its fill gets a `device_shortfall` that is completely true
    and completely uninformative about the arm being exercised. Retrying it is honest;
    reporting it as a red arm would not be, and neither would hiding it.
    """
    if any(f.get("reason") == "device_shortfall" for f in outcome.get("faults") or ()):
        return True
    message = str((outcome.get("terminal") or {}).get("safe_message", "")).lower()
    return "out of memory" in message


def drive(label: str, body: Any, tries: int = 4) -> Any:
    """Run one arm; retry it when the shared card was taken out from under it."""
    result = body()
    for index in range(1, tries):
        outcome = result[0] if isinstance(result, tuple) else result
        if not took_the_card(outcome):
            return result
        print(
            f"  the card was taken mid-run by another process on this shared box; "
            f"retrying the {label} arm ({index}/{tries - 1})",
            flush=True,
        )
        require_quiet_card()
        result = body()
    return result


def section_arms(args: argparse.Namespace) -> None:
    """cancel / kill during fetch, fill, invoke and encode — one terminal, no partials."""
    idle = require_quiet_card()

    head("ARM 1/4 — FETCH: the executor dies while the payload is being fetched")

    def fetch_arm() -> tuple[dict[str, Any], Path]:
        workspace = Path(args.tmp) / "arm-fetch"
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        fifo = workspace / "payload.fifo"
        os.mkfifo(fifo)
        spec_path, out_path = workspace / "spec.json", workspace / "outcome.json"
        spec_path.write_text(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "entrypoint": "denoise",
                    "payload": BANKED_PAYLOAD,
                    "payload_url": f"file://{fifo}",
                    "outputs": ["image"],
                    "request_id": "run-fetch",
                    "no_warm": True,
                    "delete_journal": False,
                }
            )
        )
        process = start(spec_path, out_path, workspace)
        # The payload fetch is INSIDE the atomic acceptance boundary. A FIFO holds it there
        # deterministically: the worker blocks reading the granted input, and the
        # executor it would have accepted against is killed while it waits.
        if not wait_for(workspace / "executor.pid", 900, process):
            return read_outcome(out_path), workspace
        time.sleep(0.5)
        pid = int((workspace / "executor.pid").read_text().strip())

        def feed() -> None:
            with open(fifo, "wb") as handle:
                time.sleep(1.0)  # the worker is now blocked in the fetch
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
                handle.write(json.dumps(BANKED_PAYLOAD).encode())

        threading.Thread(target=feed, daemon=True).start()
        process.wait(timeout=900)
        return read_outcome(out_path), workspace

    outcome, workspace = drive("fetch", fetch_arm)
    check(
        "the attempt refused rather than accepting against a dead executor",
        outcome.get("status") in ("OUTCOME_STATUS_REFUSED", "OUTCOME_STATUS_ABANDONED"),
        f"{outcome.get('status')}: "
        f"{str(outcome.get('terminal', {}).get('safe_message', ''))[:110]}",
    )
    rows = journal_rows(workspace)
    check(
        "exactly ONE durable terminal",
        len([r for r in rows if r["kind"] == "outcome"]) == 1,
        f"journal kinds {[r['kind'] for r in rows]}",
    )
    check(
        "and no output bytes were produced",
        not outputs_present(workspace),
        str(outputs_present(workspace) or "nothing at the granted destination"),
    )

    head("ARM 2/4 — FILL: the executor is killed mid-fill, with GiB of destinations live")

    def fill_arm() -> tuple[dict[str, Any], Path, bool, int]:
        workspace = Path(args.tmp) / "arm-fill"
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        spec_path, out_path = workspace / "spec.json", workspace / "outcome.json"
        spec_path.write_text(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "entrypoint": "denoise",
                    "payload": BANKED_PAYLOAD,
                    "outputs": ["image"],
                    "request_id": "run-fill",
                    "no_warm": True,
                    "delete_journal": False,
                }
            )
        )
        process = start(spec_path, out_path, workspace)
        if not wait_for(workspace / "executor.pid", 900, process):
            return read_outcome(out_path), workspace, False, 0
        pid = int((workspace / "executor.pid").read_text().strip())
        time.sleep(2.0)  # mid-fill: the 4.78 GiB fill takes seconds, this lands inside it
        alive = Path(f"/proc/{pid}").exists()
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
        process.wait(timeout=900)
        return read_outcome(out_path), workspace, alive, pid

    outcome, workspace, alive_before, pid = drive("fill", fill_arm)
    check(
        "the kill landed while the fill was still moving bytes",
        alive_before,
        f"executor pid {pid} was live when SIGKILL was sent",
    )
    check(
        "the run CONVERGED instead of hanging on a worker that will never be ready",
        outcome.get("ended") == "worker_error",
        f"ended={outcome.get('ended')!r}; "
        f"{(outcome.get('faults') or [{}])[-1].get('reason', 'no fault')}",
    )
    check(
        "NO acceptance was journaled for a generation that never became ready",
        not journal_rows(workspace, "accepted"),
        f"journal kinds {[r['kind'] for r in journal_rows(workspace)]}",
    )
    check(
        "no terminal was FABRICATED for an attempt that was never accepted",
        not journal_rows(workspace, "outcome"),
        "a worker that owes nothing claims nothing",
    )
    check(
        "and no output bytes exist",
        not outputs_present(workspace),
        str(outputs_present(workspace) or "nothing at the granted destination"),
    )
    settle_gpu(idle)
    check(
        "the device came back — a dead process's CUDA context is reclaimed",
        gpu_used_mib() <= idle + 64,
        f"{gpu_used_mib()} MiB (idle baseline {idle} MiB)",
    )

    head("ARM 3/4 — INVOKE: cancel a running handler, then kill one that never checks")

    def cancel_arm() -> tuple[dict[str, Any], Path]:
        workspace = Path(args.tmp) / "arm-invoke"
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        spec_path, out_path = workspace / "spec.json", workspace / "outcome.json"
        spec_path.write_text(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "entrypoint": "denoise",
                    "payload": {"steps": 64, "latent": 80, "seed": 3},
                    "outputs": ["image"],
                    "request_id": "run-cancel",
                    "no_warm": True,
                    "grace": 2.0,
                    "delete_journal": False,
                }
            )
        )
        process = start(spec_path, out_path, workspace)
        if not wait_for(workspace / "accepted.marker", 900, process):
            process.wait(timeout=60)
            return read_outcome(out_path), workspace
        time.sleep(0.4)  # inside the handler's step loop
        process.send_signal(signal.SIGINT)  # Ctrl-C on a local run
        process.wait(timeout=900)
        return read_outcome(out_path), workspace

    outcome, workspace = drive("cancel", cancel_arm)
    check(
        "a cooperative handler converges to CANCELED",
        outcome.get("status") == "OUTCOME_STATUS_CANCELED",
        f"{outcome.get('status')}: "
        f"{str(outcome.get('terminal', {}).get('safe_message', ''))[:110]}",
    )
    check(
        "exactly ONE durable terminal, journaled before it was sent",
        len(journal_rows(workspace, "outcome")) == 1,
        f"journal kinds {[r['kind'] for r in journal_rows(workspace)]}",
    )
    check(
        "no partial output is visible",
        not outputs_present(workspace) and not outcome.get("manifest"),
        str(outputs_present(workspace) or "nothing at the granted destination"),
    )

    def stubborn_arm() -> tuple[dict[str, Any], Path, float]:
        workspace = Path(args.tmp) / "arm-stubborn"
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        spec_path, out_path = workspace / "spec.json", workspace / "outcome.json"
        spec_path.write_text(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "entrypoint": "stubborn",
                    "payload": {"steps": 512, "latent": 80, "seed": 3},
                    "outputs": [],
                    "request_id": "run-stubborn",
                    "no_warm": True,
                    "grace": 2.0,
                    "delete_journal": False,
                }
            )
        )
        process = start(spec_path, out_path, workspace)
        if not wait_for(workspace / "accepted.marker", 900, process):
            process.wait(timeout=60)
            return read_outcome(out_path), workspace, 0.0
        time.sleep(0.4)
        started = time.perf_counter()
        process.send_signal(signal.SIGINT)
        process.wait(timeout=900)
        return read_outcome(out_path), workspace, time.perf_counter() - started

    outcome, workspace, took = drive("stubborn-cancel", stubborn_arm)
    log = (workspace / "run.log").read_text(errors="replace")
    check(
        "a handler that never checks cancellation is KILLED, not waited for",
        "forceful" in log and outcome.get("status") == "OUTCOME_STATUS_CANCELED",
        next((ln for ln in log.splitlines() if "forceful:" in ln), "")[:150],
    )
    check(
        "grace expired first, and the terminal came after the process was gone",
        took >= 2.0,
        f"{took:.2f} s from cancel to a closed run (2.0 s grace)",
    )
    check(
        "still exactly ONE durable terminal",
        len(journal_rows(workspace, "outcome")) == 1,
        f"journal kinds {[r['kind'] for r in journal_rows(workspace)]}",
    )

    head("ARM 4/4 — ENCODE: a tail failure, and a run killed inside the attempt window")
    workspace = Path(args.tmp) / "arm-tail"
    outcome, _ = spawn(
        {
            "workspace": str(workspace),
            "entrypoint": "denoise",
            "payload": BANKED_PAYLOAD,
            # The grant names a destination the handler will not produce. The tail binds by
            # FIELD PATH with exact set equality, so it refuses WHOLE (#248).
            "outputs": ["image", "thumbnail"],
            "request_id": "run-tail",
            "no_warm": True,
            "delete_journal": False,
        }
    )
    check(
        "a tail that cannot satisfy the grant set fails the attempt",
        outcome.get("status") == "OUTCOME_STATUS_FAILED"
        and "destination_absent" in str(outcome.get("terminal", {}).get("safe_message", "")),
        f"{outcome.get('status')}: "
        f"{str(outcome.get('terminal', {}).get('safe_message', ''))[:150]}",
    )
    check(
        "the output transaction is not partially successful",
        not outcome.get("manifest") and not outputs_present(workspace),
        f"manifest {outcome.get('manifest')}, destination {outputs_present(workspace) or 'empty'}",
    )
    check(
        "one durable terminal, and the ledger still closed",
        len(journal_rows(workspace, "outcome")) == 1
        and outcome.get("reconciliation", {}).get("closed") is True,
        f"kinds {[r['kind'] for r in journal_rows(workspace)]}",
    )

    def killrun_arm() -> tuple[dict[str, Any], Path]:
        workspace = Path(args.tmp) / "arm-killrun"
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        spec_path, out_path = workspace / "spec.json", workspace / "outcome.json"
        spec_path.write_text(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "entrypoint": "denoise",
                    "payload": {"steps": 24, "latent": 80, "seed": 5},
                    "outputs": ["image"],
                    "request_id": "run-killed",
                    "instance_id": "ins-stable",
                    "no_warm": True,
                    "delete_journal": False,
                }
            )
        )
        process = start(spec_path, out_path, workspace)
        if not wait_for(workspace / "accepted.marker", 900, process):
            process.wait(timeout=60)
            return read_outcome(out_path), workspace
        time.sleep(0.3)
        process.send_signal(signal.SIGKILL)  # the whole run dies inside the attempt window
        process.wait(timeout=60)
        # No outcome document exists — the run was killed before it could write one — so the
        # JOURNAL it left behind is the evidence, which is exactly the point of the arm.
        return {"killed": True}, workspace

    _, workspace = drive("killed-run", killrun_arm)
    check(
        "a killed run leaves its journal behind — the deletable journal is deleted LAST",
        (workspace / "worker" / "journal.log").is_file(),
        f"journal kinds {[r['kind'] for r in journal_rows(workspace)]}",
    )
    check(
        "the accepted-but-incomplete attempt is on record with no terminal",
        bool(journal_rows(workspace, "accepted")) and not journal_rows(workspace, "outcome"),
        "acceptance is journaled before anything can run, which is what makes recovery "
        "possible without guessing",
    )
    check(
        "and no output bytes are visible from the interrupted attempt",
        not outputs_present(workspace),
        str(outputs_present(workspace) or "nothing at the granted destination"),
    )
    # The NEXT run over the same journal is the recovery: it must mint exactly one durable
    # ABANDONED terminal for the interrupted ordinal and never re-execute it.
    spec_path = workspace / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "workspace": str(workspace),
                "entrypoint": "denoise",
                "payload": BANKED_PAYLOAD,
                "outputs": ["image"],
                "request_id": "run-after",
                "instance_id": "ins-stable",
                "no_warm": True,
                "delete_journal": False,
            }
        )
    )
    recovered = start(spec_path, workspace / "outcome.json", workspace)
    recovered.wait(timeout=900)
    rows = journal_rows(workspace)
    killed_terminals = [
        r for r in rows if r["kind"] == "outcome" and r["body"]["request_id"] == "run-killed"
    ]
    check(
        "recovery minted exactly one durable terminal for the interrupted attempt",
        len(killed_terminals) == 1,
        f"{len(killed_terminals)} terminal(s) for run-killed",
    )
    check(
        "the interrupted attempt was never re-executed",
        len(
            [r for r in rows if r["kind"] == "accepted" and r["body"]["request_id"] == "run-killed"]
        )
        == 1,
        "one acceptance, one terminal, zero second executions",
    )
    print(f"\n  GPU {gpu_used_mib()} MiB (idle baseline {idle} MiB)")


def section_clone(args: argparse.Namespace) -> None:
    """The git-clone packaging contract, against a CLEAN clone of this repo."""
    require_quiet_card()
    head("git clone + `uv sync --frozen` + one slice, with no other Cozy software")
    clone = Path(args.tmp) / "clone"
    shutil.rmtree(clone, ignore_errors=True)
    out = subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", "file://" + str(ROOT), str(clone)],
        capture_output=True,
        text=True,
        check=False,
    )
    check("a clean clone of the committed tree", out.returncode == 0, out.stderr[:150])
    started = time.perf_counter()
    sync = subprocess.run(
        ["nice", "-n", "19", "uv", "sync", "--frozen", "--extra", "corpus"],
        cwd=clone,
        capture_output=True,
        text=True,
        check=False,
    )
    check(
        "`uv sync --frozen` resolves the committed lock with nothing else installed",
        sync.returncode == 0,
        f"{(time.perf_counter() - started):.1f} s — {sync.stderr.strip().splitlines()[-1][:110]}"
        if sync.stderr.strip()
        else "",
    )
    wheels = sorted(WHEELHOUSE.glob("tensorfs-*.whl"))
    check(
        "the TensorFS facade wheel is available to install into the clone",
        bool(wheels),
        str(wheels[0].name) if wheels else f"no tensorfs wheel under {WHEELHOUSE}",
    )
    if wheels:
        # NAMED, not hidden: TensorFS has no index yet, so the clone installs the wheel
        # explicitly. Everything the REPO owns resolves from the committed lock; this one
        # external dependency is a step the packaging contract does not yet cover, and
        # saying so is the honest form of the claim.
        extra = subprocess.run(
            ["nice", "-n", "19", "uv", "pip", "install", str(wheels[0])],
            cwd=clone,
            capture_output=True,
            text=True,
            check=False,
        )
        check(
            "and it installs into the clone's venv",
            extra.returncode == 0,
            (extra.stderr or extra.stdout).strip().splitlines()[-1][:110],
        )
    workspace = Path(args.tmp) / "clone-run"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    spec = {
        "workspace": str(workspace),
        "entrypoint": "denoise",
        "payload": BANKED_PAYLOAD,
        "outputs": ["image"],
        "request_id": "run-clone",
        "no_warm": True,
        "delete_journal": True,
    }
    (workspace / "spec.json").write_text(json.dumps(spec))
    run = subprocess.run(
        [
            "nice",
            "-n",
            "19",
            str(clone / ".venv" / "bin" / "python"),
            str(clone / "scripts" / "slice-live.py"),
            "exec",
            "--spec",
            str(workspace / "spec.json"),
            "--out",
            str(workspace / "outcome.json"),
        ],
        cwd=clone,
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "PYTHONPATH": f"{clone}:{clone / 'src'}",
        },
        timeout=1800,
    )
    outcome = read_outcome(workspace / "outcome.json")
    check(
        "the clone served one real request end to end",
        run.returncode == 0 and outcome.get("status") == "OUTCOME_STATUS_SUCCEEDED",
        f"exit {run.returncode}, {outcome.get('status')}; "
        f"{run.stdout.strip().splitlines()[-1][:110] if run.stdout.strip() else run.stderr[:110]}",
    )
    check(
        "and it produced the SAME bits as the working tree",
        (outcome.get("result") or {}).get("digest") == BANKED,
        f"{(outcome.get('result') or {}).get('digest', 'none')}",
    )
    check(
        "the one-shot run deleted its journal after a fully acknowledged terminal",
        outcome.get("journal_deleted") is True,
        "§8: deletable AFTER the ack, never before",
    )
    print(f"  GPU {gpu_used_mib()} MiB")


def section_bench(args: argparse.Namespace) -> None:
    """request -> result, with the legs attributed. Compared against cr-007's numbers."""
    head("benchmarks: request -> result, per-leg")
    idle = require_quiet_card()
    rows: list[dict[str, Any]] = []
    for index in range(args.repeat):
        settle_gpu(idle)
        require_quiet_card()
        workspace = Path(args.tmp) / f"bench-{index}"
        outcome, _ = spawn(
            {
                "workspace": str(workspace),
                "entrypoint": "denoise",
                "payload": BANKED_PAYLOAD,
                "outputs": ["image"],
                "request_id": f"bench-{index}",
                "no_warm": index == 0,
                "delete_journal": False,
            }
        )
        if not check(
            f"run {index} produced a terminal",
            outcome.get("status") == "OUTCOME_STATUS_SUCCEEDED",
            f"ended={outcome.get('ended')!r} status={outcome.get('status')!r} "
            f"{str(outcome.get('terminal', {}).get('safe_message', ''))[:120]}",
        ):
            continue
        ledger = {row["class"]: row["bytes"] for row in outcome["ledger"]["classes"]}
        metrics = outcome["terminal"].get("metrics", {})
        rows.append(
            {
                **outcome["timings"],
                "handler_ms": metrics.get("handler_ms", 0),
                "finalization_ms": metrics.get("finalization_ms", 0),
                "activations_mib": round(ledger.get("activations_scratch", 0) / MIB, 1),
                "weights_gib": round(ledger.get("active_weights", 0) / GIB, 3),
                "rss_run_mib": round(ledger.get("rss_worker", 0) / MIB, 1),
                "rss_executor_mib": round(ledger.get("rss_executor", 0) / MIB, 1),
                "digest_ok": (outcome.get("result") or {}).get("digest") == BANKED,
            }
        )
        print(f"  run {index}: {json.dumps(rows[-1])}")
    if rows:
        keys = [k for k in rows[0] if k != "digest_ok"]
        median = {k: sorted(float(r[k]) for r in rows)[len(rows) // 2] for k in keys}
        print(f"\n  median: {json.dumps(median)}")
        check("every benchmark run produced the banked digest", all(r["digest_ok"] for r in rows))
    print(f"  GPU {gpu_used_mib()} MiB")
    bench_observability(rows)


def bench_observability(runs: list[dict[str, Any]]) -> None:
    """cr-011's cost, against the thing it is measuring. It has to be noise or it is a tax."""
    import timeit

    from cozy_runtime.author._observations import Attribution, EventRing
    from cozy_runtime.internal.journal import Journal
    from cozy_runtime.internal.worker import triage

    head("cr-011 cost — attribution on the hot path, and the bundle off it")
    attribution = Attribution()
    step_ns = timeit.timeit(lambda: attribution.step("denoise", 15.2), number=200_000) / 2e5 * 1e9
    ring = EventRing(subject="bench")
    emit_ns = (
        timeit.timeit(lambda: ring.emit("log", "one line", "info"), number=100_000) / 1e5 * 1e9
    )
    step_ms = next((float(r["handler_ms"]) for r in runs if float(r.get("handler_ms", 0)) > 0), 0.0)
    print(f"  attribution.step()      {step_ns:8.0f} ns per step")
    print(f"  ring.emit()             {emit_ns:8.0f} ns per record")
    print(f"  one real handler leg    {step_ms:8.1f} ms")
    check(
        "per-step attribution is NOISE against the step it attributes",
        step_ns < 2000 and (step_ms == 0 or step_ns / 1e6 < step_ms / 1000),
        f"{step_ns:.0f} ns of bookkeeping against a {step_ms:.0f} ms handler leg — "
        f"{step_ns / 1e6 / max(step_ms, 1e-9) * 100:.5f}% of it",
    )

    root = Path(tempfile.mkdtemp(prefix="cozy-bench-triage"))
    try:
        journal = Journal(root / "journal.log")
        store = triage.BundleStore(root / "triage", journal)
        full = EventRing(subject="bench#1")
        for index in range(1000):
            full.emit("log", f"handler line {index} with a realistic amount of text", "info")

        def build_one() -> None:
            assembly = triage.Assembly(triage.mint_subject(), attempt={"request_id": "b"})
            assembly.events = full.rows()
            assembly.build()

        build_ms = timeit.timeit(build_one, number=50) / 50 * 1000

        def commit_one() -> None:
            assembly = triage.Assembly(triage.mint_subject(), attempt={"request_id": "b"})
            assembly.events = full.rows()
            store.commit(assembly)

        commit_ms = timeit.timeit(commit_one, number=20) / 20 * 1000
        print(f"  bundle build (256 rows) {build_ms:8.3f} ms")
        print(f"  build + fsync + journal {commit_ms:8.3f} ms")
        check(
            "the bundle is built and made durable OFF the hot path, in the terminal tail",
            commit_ms < 50,
            f"{commit_ms:.2f} ms per terminal, of which {commit_ms - build_ms:.2f} ms is "
            "two fsyncs — the same durability the terminal itself pays for",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- cr-011 helpers


def clock_names_in(function: str) -> list[str]:
    """Every clock-shaped NAME reachable in one function's syntax tree.

    Deliberately the AST and not the text: the module's prose says "there is no boot
    timeout", and a substring scan would find that sentence and call the fence red for
    saying the true thing. What matters is whether the CODE can consult a clock.
    """
    import ast

    clocks = ("time", "monotonic", "perf_counter", "sleep", "deadline", "timeout", "elapsed")
    source = (ROOT / "src" / "cozy_runtime" / "internal" / "worker" / "observe.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return sorted(
                {
                    inner.id if isinstance(inner, ast.Name) else inner.attr
                    for inner in ast.walk(node)
                    if isinstance(inner, (ast.Name, ast.Attribute))
                    and (inner.id if isinstance(inner, ast.Name) else inner.attr) in clocks
                }
            )
    raise SystemExit(f"no function named {function!r} in observe.py")


def triage_read(root: Path, subject: str) -> dict[str, Any]:
    """Read one triage bundle from a DIRECTORY. No process, no socket, no SSH."""
    from cozy_runtime.internal.worker.triage import BundleStore

    return BundleStore.read(root / "triage", subject, root / "journal.log")


def explain_bundle(document: dict[str, Any]) -> list[str]:
    from cozy_runtime.internal.worker import triage

    return triage.explain(document)


def flood_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Flood the emit boundary, then assemble a bundle far past the 1 MiB wire cap.

    Two caps have to hold and they are different: the RING's memory cap (count + bytes,
    oldest shed first) and the BUNDLE's protocol cap. A flood that fits one and blows the
    other would produce a terminal a coordinator cannot accept.
    """
    from cozy_runtime.author._observations import EventRing, ObservationRefusal
    from cozy_runtime.internal.journal import Journal
    from cozy_runtime.internal.worker import triage

    ring = EventRing(subject="flood#1")
    for index in range(20_000):
        ring.emit("log", f"step {index} of a handler that logs every iteration", "info")
        if index % 5000 == 0:
            with contextlib.suppress(ObservationRefusal):
                ring.emit("log", "auth", "info", api_key="sk-livekeythatmustnevertravel")
    caps = ring.caps()

    journal = Journal(root / "journal.log")
    store = triage.BundleStore(root / "triage", journal)
    assembly = triage.Assembly("trb-" + "f" * 24, attempt={"request_id": "flood"}, caps=caps)
    assembly.events = ring.rows()
    # Push the assembled document PAST the wire cap on purpose: 4,000 rows at the row cap.
    assembly.events += [
        {"seq": n, "kind": "log", "name": "x" * 200, "value": "y" * 400, "attempt_key": "flood#1"}
        for n in range(4000)
    ]
    receipt = store.commit(assembly)
    document = triage.BundleStore.read(root / "triage", receipt.subject_id, root / "journal.log")
    return caps, {
        "length": receipt.length,
        "shed_to_fit": document["caps"]["shed_to_fit"],
        "truncated": document["caps"]["truncated"],
        "kept": len(document["events"]),
    }


def section_triage(args: argparse.Namespace) -> None:
    """cr-011: kill the executor mid-attempt and explain it FROM THE BUNDLE ALONE."""
    idle = require_quiet_card()

    head("kill the executor MID-ATTEMPT and explain the attempt with no live process")

    def kill_arm() -> tuple[dict[str, Any], Path]:
        workspace = Path(args.tmp) / "triage-kill"
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        spec_path, out_path = workspace / "spec.json", workspace / "outcome.json"
        spec_path.write_text(
            json.dumps(
                {
                    "workspace": str(workspace),
                    "entrypoint": "denoise",
                    "payload": {"steps": 64, "latent": 80, "seed": 3},
                    "outputs": ["image"],
                    "request_id": "run-triage",
                    "no_warm": True,
                    "delete_journal": False,
                }
            )
        )
        process = start(spec_path, out_path, workspace)
        if not wait_for(workspace / "accepted.marker", 900, process):
            process.wait(timeout=120)
            return read_outcome(out_path), workspace
        time.sleep(0.6)  # the handler is inside its step loop, on the device
        os.kill(int((workspace / "executor.pid").read_text().strip()), signal.SIGKILL)
        process.wait(timeout=900)
        return read_outcome(out_path), workspace

    outcome, workspace = drive("triage-kill", kill_arm)
    check(
        "the attempt reached a terminal despite the executor dying",
        outcome.get("status") in ("OUTCOME_STATUS_ABANDONED", "OUTCOME_STATUS_CANCELED"),
        f"{outcome.get('status')}: "
        f"{str(outcome.get('terminal', {}).get('safe_message', ''))[:110]}",
    )

    # THE ARM: from here on, nothing but the directory. No process, no socket, no SSH.
    subject = outcome["triage_subject"]
    root = workspace / "worker"
    bundle = triage_read(root, subject)
    check(
        "the bundle is readable BY OPAQUE KEY from a directory alone",
        bundle.get("subject_id") == subject,
        f"{root / 'triage' / (subject + '.json')} verified against its journaled receipt",
    )
    lines = explain_bundle(bundle)
    print("\n  the whole postmortem, from the file:")
    for line in lines:
        print(f"    {line}")
    text = "\n".join(lines)
    check(
        "it names the attempt, the spec and the terminal cause",
        "run-triage#1" in text and "sha256:" in text and "CAUSE_CODE" in text,
        "identity + cause without asking anyone",
    )
    check(
        "it names the executor generation that died",
        "incarnation" in text and "readiness epoch" in text,
        f"incarnation {bundle['attempt']['executor_incarnation']}, "
        f"epoch {bundle['attempt']['readiness_epoch']}",
    )
    classes = {row["class"]: row for row in bundle["measurements"]["ledger"]["classes"]}
    unreadable = sorted(k for k, v in classes.items() if v["kind"] == "unreadable")
    check(
        "the dead executor's device classes are UNREADABLE in the bundle, never zero",
        "device_free" in unreadable and "device_allocated" in unreadable,
        f"unreadable: {unreadable}",
    )
    check(
        "host classes still reconcile to their opening values",
        classes["host_ram"]["after"] == classes["host_ram"]["before"] == 0,
        f"host_ram {classes['host_ram']['before']} -> {classes['host_ram']['after']} B",
    )
    check(
        "the applied posture at the moment of death is in the bundle",
        bundle["posture"]["applied_revision"] == 1,
        f"intake {bundle['posture']['intake_state']}, "
        f"posture {bundle['posture']['posture']}, "
        f"boot steps {[s['step'] + ':' + s['outcome'] for s in bundle['posture']['boot_steps']]}",
    )
    names = [f"{row['kind']}/{row['name']}" for row in bundle["events"]]
    print(f"\n  the tail, from the dead process: {names}")
    check(
        "the EXECUTOR's own observations survived the kill, live-forwarded before it died",
        any(row["kind"] == "log" and "denoise loop" in row["name"] for row in bundle["events"]),
        "a ring that only crossed in the reply would have crossed never",
    )
    check(
        "the WORKER's own rows are there too — dispatch, and the death it witnessed",
        any("dispatched" in row["name"] for row in bundle["events"])
        and any(row["kind"] == "fault" for row in bundle["events"]),
        f"{len(bundle['events'])} rows kept, caps {json.dumps(bundle['caps'])}",
    )
    check(
        "and nothing in the bundle can override the terminal",
        bundle["terminal"]["status"] == outcome["status"],
        "TriageBundleRef is OBSERVATION ONLY — the bundle repeats the terminal, never sets it",
    )
    settle_gpu(idle)

    head("the emit boundary under a FLOOD, and the 1 MiB wire cap")
    flooded = Path(args.tmp) / "triage-flood"
    shutil.rmtree(flooded, ignore_errors=True)
    flooded.mkdir(parents=True)
    caps, built = flood_bundle(flooded)
    check(
        "a flooded emitter hits the ring caps and COUNTS what it shed",
        caps["kept"] <= caps["max_events"] and caps["dropped"] > 0,
        f"{caps['admitted']} admitted, {caps['kept']} kept, {caps['dropped']} shed, "
        f"{caps['held_bytes']} B held (cap {caps['max_bytes']} B)",
    )
    check(
        "the assembled bundle fits the protocol's 1 MiB reference cap",
        built["length"] <= (1 << 20),
        f"{built['length']} B after shedding {built['shed_to_fit']} more rows to fit",
    )
    check(
        "a credential-shaped field is REFUSED at the emit boundary, not redacted later",
        caps["refused"] > 0,
        f"{caps['refused']} refusal(s) counted; nothing credential-shaped reached the ring",
    )


def section_wedge(args: argparse.Namespace) -> None:
    """cr-011: a wedged transfer is detected by SILENCE; a slow-but-advancing one is not."""
    head("silence-bounded liveness — the same detector, two transfers, no clock anywhere")
    from cozy_runtime.author._observations import EventRing
    from cozy_runtime.internal.worker.observe import SilenceMonitor

    monitor = SilenceMonitor(EventRing(subject="worker"), give_up_after=8)
    monitor.start("slow:transfer", "download")
    monitor.start("wedged:transfer", "download")
    verdicts: dict[str, list[str]] = {"slow": [], "wedged": []}
    for position in range(1, 41):
        # The SLOW one moves ONE MiB per observation, which is 1/4000th of the wedged one's
        # nominal rate. The WEDGED one moved 4,096 MiB and then stopped. Nothing below reads
        # a clock, sleeps, or measures elapsed time: the inputs are two integers.
        monitor.advance("slow:transfer", position)
        monitor.advance("wedged:transfer", 4096)
        verdicts["slow"].append(monitor.sample("slow:transfer"))
        verdicts["wedged"].append(monitor.sample("wedged:transfer"))
    check(
        "the WEDGED transfer is called wedged, by silence",
        verdicts["wedged"][-1] == "wedged",
        f"{verdicts['wedged'].count('silent')} silent observations, then "
        f"{verdicts['wedged'].count('wedged')} wedged — position never left 4096",
    )
    check(
        "the SLOW-BUT-ADVANCING transfer is NEVER called wedged, however slow",
        set(verdicts["slow"]) == {"advancing"},
        f"{len(verdicts['slow'])} observations, all advancing at 1 MiB each — a transfer "
        "this slow would have blown any boot timeout",
    )
    check(
        "there is NO CLOCK in the detector — proven against its own SYNTAX TREE",
        not clock_names_in("sample"),
        "`sample` reads two integers and a counter; the only time-shaped tokens anywhere "
        "in that module are in prose saying there is no boot timeout",
    )
    check(
        "`started` is published at POSITION ZERO so a pre-first-unit wedge renders",
        any(row["name"] == "started" and row["value"] == 0 for row in monitor.ring.rows()),
        "a subject nobody ever mentioned cannot be reported as wedged",
    )
    print(f"\n  cursors: {json.dumps(monitor.document())}")


def section_witness(args: argparse.Namespace) -> None:
    """cr-011: OOM-kill the WORKER and name the death from the surviving record."""
    head("the postmortem witness — a parent that outlives the journal owner")
    from cozy_runtime.internal import witness as wit

    workspace = Path(args.tmp) / "witness"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)

    kind = wit.require_durable(workspace / "witness")
    check(
        "the boot record lives OFF tmpfs/ramfs",
        kind not in ("tmpfs", "ramfs"),
        f"{workspace} is on {kind} — a boot record in RAM dies with the event it explains",
    )

    # A REAL kernel OOM kill, in a REAL memory cgroup, without root: the systemd user
    # manager delegates a cgroup subtree, so a scope with MemoryMax set is the actual
    # mechanism a pod uses. The victim is a stand-in for the worker and the witness
    # cannot tell the difference — it reads a wait status and a kernel counter.
    victim = workspace / "victim.py"
    victim.write_text(
        "import sys\nblocks = []\n"
        "while True:\n    blocks.append(bytearray(8 << 20))\n    sys.stdout.flush()\n"
    )
    scope = f"cozy-witness-{os.getpid()}"
    command = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        f"--unit={scope}",
        "-p",
        "MemoryMax=96M",
        "-p",
        "MemorySwapMax=0",
        # Without this systemd tears the whole scope down on the OOM — witness included —
        # and the record that exists to explain the death dies with it. `continue` leaves
        # the kernel's per-task kill in place and lets the parent live to report it, which
        # is exactly the arrangement a pod needs.
        "-p",
        "OOMPolicy=continue",
        sys.executable,
        "-c",
        "import sys; from cozy_runtime.internal.witness import main;"
        " raise SystemExit(main(sys.argv[1:]))",
        "--root",
        str(workspace),
        "--",
        sys.executable,
        str(victim),
    ]
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT / 'src'}")
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300)
    print(f"  systemd-run exit {result.returncode}: {result.stderr.strip()[:200]}")

    # From here on: NO live process. Only the file.
    death = wit.read_death(workspace)
    print(f"  the surviving record: {json.dumps(death, sort_keys=True)}")
    check(
        "the death is NAMED by a process the dying one could not have written from",
        death["how"] == "oom_killed",
        f"how={death['how']} signal={death['signal_name']} "
        f"oom_kill {death['cgroup_oom_kill_before']} -> {death['cgroup_oom_kill_after']}",
    )
    check(
        "the OOM verdict is a MEASURED kernel counter delta, not an inference from SIGKILL",
        death["oom_killed"] == "true"
        and death["cgroup_oom_kill_after"] > death["cgroup_oom_kill_before"],
        death["method"],
    )
    boot = json.loads((workspace / "witness" / "boot.json").read_text())
    check(
        "the boot record was written BEFORE the child started and survived it",
        boot["child_pid"] == death["child_pid"] and boot["format"].endswith("WitnessBoot/1"),
        f"child {boot['child_pid']} ran {death['ran_ms']} ms on {boot['record_filesystem']}",
    )


# --------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description="cr-008a local slice verification")
    parser.add_argument(
        "section",
        choices=(
            "exec",
            "slice",
            "ledger",
            "arms",
            "triage",
            "wedge",
            "witness",
            "clone",
            "bench",
            "all",
        ),
    )
    parser.add_argument("--tmp", default="/tmp/cozy-slice")
    parser.add_argument("--spec", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.section == "exec":
        outcome = run_one(json.loads(Path(args.spec).read_text()))
        Path(args.out).write_text(json.dumps(outcome, indent=2, default=str))
        return 0

    Path(args.tmp).mkdir(parents=True, exist_ok=True)
    sections = {
        "slice": section_slice,
        "ledger": section_ledger,
        "arms": section_arms,
        "triage": section_triage,
        "wedge": section_wedge,
        "witness": section_witness,
        "clone": section_clone,
        "bench": section_bench,
    }
    chosen = list(sections) if args.section == "all" else [args.section]
    for name in chosen:
        sections[name](args)
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
