#!/usr/bin/env python3
"""cr-008c live verification: the DELIVERY axis, chosen over two real SDXL artifacts.

Not a test (tracker README #160): every leg runs the real chooser, the real prepare, the
real attempt path and the real endpoint, and prints numbers.

The two artifacts are cr-006's: the real fp16 SDXL UNet (`plain/1`) and the real
`fp8-scaled-scalar/1` carrier made from it, both installed in ONE four-component store so
the only difference between the arms is the UNet's encoding.

  variants   survey both artifacts from their HEADERS: routes, bytes, ladder eligibility
  calibrate  one real end-to-end generation per variant; banks each one's fill envelope
  conform    one denoise step per variant on identical inputs: step wall + deviation
  choose     the chooser, over the banked table, under each objective
  serve      both variants OFFERED: the choice reaches the attempt, twice, two axes
  arms       the red arms, each planted for real
  all        every section above, in order
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _quant() -> Any:
    """cr-006's driver, loaded as a module. Its artifact production, its card-quiet wait
    and its single-step harness are the SAME experiment one issue earlier; copying them
    here would let the two drift and make a deviation number mean two things."""
    spec = importlib.util.spec_from_file_location("quant_live", ROOT / "scripts" / "quant-live.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Q = _quant()

WORK = Path("/tmp/cozy-cr008c")
#: ONE cozy_home across every arm, deliberately: the probe fact and the delivery table are
#: DEVICE properties and a fresh home per run would re-probe and re-measure forever.
HOME = WORK / "home"
PLAIN_STORE = Q.PLAIN_STORE
RELEASE = "cozy/sdxl-pipeline@cr-008c"
COMPONENTS = Q.COMPONENTS
GIB = 1 << 30
MIB = 1 << 20

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


# --------------------------------------------------------------------------- variants


def snapshots() -> dict[str, str]:
    """The four-component snapshot map of the PLAIN pipeline, read from cr-008b's store."""
    return dict(json.loads(Q.PLAIN_SNAPSHOTS.read_text()))


def fp8_unet() -> str:
    """The fp8 UNet's snapshot inside the four-component store.

    `install_into` is content-addressed and additive, so calling it again when the objects
    are already there re-plans the same bytes and returns the same digest — which is why
    this is a lookup rather than a cached constant transcribed into this file.
    """
    cached = WORK / "fp8-unet.txt"
    if cached.is_file():
        return cached.read_text().strip()
    WORK.mkdir(parents=True, exist_ok=True)
    digest = "sha256:" + str(Q.install_into(PLAIN_STORE)).removeprefix("sha256:")
    cached.write_text(digest)
    return digest


def variants() -> list[dict[str, Any]]:
    """The OFFERED set: two encodings of one logical model, one store, one topology."""
    plain = snapshots()
    encoded = dict(plain, unet=fp8_unet())
    return [
        {
            "name": "fp16",
            "store": str(PLAIN_STORE),
            "snapshot": plain["unet"],
            "snapshots": ",".join(f"{k}={v}" for k, v in sorted(plain.items())),
            "reference": True,
        },
        {
            "name": "fp8",
            "store": str(PLAIN_STORE),
            "snapshot": encoded["unet"],
            "snapshots": ",".join(f"{k}={v}" for k, v in sorted(encoded.items())),
            "reference": False,
        },
    ]


def binding_record(
    offered: list[dict[str, Any]], objective: str = "latency", resident_budget: int = 0
) -> dict[str, Any]:
    """The deployment record. `variants` is the only new field, and it is ADDITIVE: a
    record that omits it offers exactly what `store`/`snapshot` already named."""
    reference = next(v for v in offered if v["reference"])
    return {
        "plan_id": Q.plan_id_for("generate"),
        "project": str(ROOT / "corpus" / "pipeline"),
        "model_class": "SdxlPipelineModel",
        "binding_path": "generate.models.model",
        "param": "model",
        "component": "unet",
        "components": ",".join(COMPONENTS),
        "store": str(PLAIN_STORE),
        "config": str(Q.PIPE_CONFIG),
        "snapshot": reference["snapshot"],
        "snapshots": reference["snapshots"],
        "release": RELEASE,
        "variant": "sm89",
        "vram_bytes": 7 * GIB,
        "host_bytes": 2 * GIB,
        "pinned_bytes": 256 * MIB,
        "entrypoint": "generate",
        "model_construction_digest": "",
        "resident_budget_bytes": resident_budget,
        "vram_floor_bytes": 1 * GIB,
        "tenancy": "local",
        "variants": offered,
        "objective": objective,
        "steps_basis": 20,
    }


def survey() -> None:
    head("variants — what each artifact's HEADER demands, and which rungs that admits")
    import torch

    from cozy_runtime.internal.delivery import DELIVERY_RUNGS, Variant, eligible_rungs
    from cozy_runtime.internal.encoding import (
        Selector,
        device_line,
        launch_providers,
        measure_device,
    )
    from cozy_runtime.internal.fill import Checkpoint, dtype_name, survey_routes
    from cozy_runtime.internal.probe import qualify

    print("  the CLOSED delivery ladder:")
    for rung in DELIVERY_RUNGS:
        mark = "built" if rung.status == "proven" else "NAMED, NOT BUILT"
        print(
            f"    {rung.name:<15}{rung.delivery}/{rung.materialization or '-':<14}"
            f"{rung.cost_class:<16}{mark}"
        )

    device = measure_device(torch, 0)
    print(f"  device: {device_line(device)}")
    providers = launch_providers()
    surveyed: dict[str, Any] = {}
    for record in variants():
        variant = Variant.read(record)
        vmap = variant.snapshot_map()
        books = {
            name: Checkpoint(variant.store, vmap.get(name, variant.snapshot)) for name in COMPONENTS
        }
        rows = [r for name in COMPONENTS for r in books[name].rows(name)]
        surveyed[variant.name] = (variant, rows)
    dtypes = sorted({dtype_name(r.dtype) for _, rows in surveyed.values() for r in rows})
    capabilities, probe = qualify(
        torch, providers, device, dtypes, release=RELEASE, cache=HOME / "probe"
    )
    selector = Selector(providers, capabilities, device)
    for name, (_variant, rows) in surveyed.items():
        routes, refusal = survey_routes(rows, selector)
        rungs = [r.name for r in eligible_rungs(routes)]
        aliases = sorted({r.encoded.alias for r in rows})
        print(
            f"  {name:<6}{len(rows):>5} tensors  {sum(r.stored_nbytes for r in rows) / GIB:>7.3f} "
            f"GiB stored  routes={routes}  encodings={aliases}"
        )
        print(f"         eligible rungs: {rungs}  qualification: {refusal or 'every encoding'}")
    check(
        "the two variants are the SAME logical model",
        [r.key for r in surveyed["fp16"][1]] == [r.key for r in surveyed["fp8"][1]],
        f"{len(surveyed['fp16'][1])} destinations, key for key",
    )
    check(
        "the encoded variant admits the DECODE rungs and the plain one does not",
        [r.name for r in eligible_rungs(survey_routes(surveyed["fp8"][1], selector)[0])]
        == ["aot_decode", "staged_decode", "jit_decode"]
        and [r.name for r in eligible_rungs(survey_routes(surveyed["fp16"][1], selector)[0])]
        == ["verbatim"],
        "route eligibility comes from the header's own per-tensor encodings",
    )
    check(
        "the probe suite is milliseconds and ran once for BOTH variants",
        probe["suite_ms"] < 5000,
        f"{probe['suite_ms']} ms over {len(probe['observations'])} observations",
    )


# ------------------------------------------------------------------------- the runs


def spawn(label: str, spec: dict[str, Any], mode: str = "exec") -> dict[str, Any]:
    """One arm in its OWN process, so peak VRAM and the CUDA context are that arm's."""
    workspace = WORK / f"run-{label}"
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    spec_path = workspace / "spec.json"
    out_path = workspace / "outcome.json"
    spec_path.write_text(json.dumps(dict(spec, workspace=str(workspace))))
    log = workspace / "run.log"
    for attempt in range(1, 5):
        Q.require_quiet_card()
        with open(log, "wb") as handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    mode,
                    str(spec_path),
                    str(out_path),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(__import__("os").environ, PYTHONPATH=f"{ROOT}:{ROOT / 'src'}"),
                cwd=str(ROOT),
            )
        process.wait(timeout=5400)
        if not out_path.is_file():
            raise SystemExit(
                f"{label}: no outcome (exit {process.returncode})\n"
                f"{log.read_text(errors='replace')[-4000:]}"
            )
        outcome: dict[str, Any] = json.loads(out_path.read_text())
        if mode != "exec" or not Q.took_the_card(outcome) or attempt == 4:
            outcome["_log"] = str(log)
            return outcome
        print(f"    a neighbour took the card during {label}; retrying ({attempt}/4)")
    raise SystemExit("unreachable")


def exec_one(spec_path: Path, out_path: Path) -> None:
    """ONE request through the WHOLE real path — supervisor, executor, endpoint."""
    import dataclasses
    import os

    from cozy_runtime.internal.config import read_config
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions

    spec = json.loads(spec_path.read_text())
    workspace = Path(spec["workspace"])
    HOME.mkdir(parents=True, exist_ok=True)
    config = read_config(
        {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "COZY_HOME": str(HOME),
            "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}",
        }
    )
    outcome = run_slice(
        config,
        LocalRequest(
            entrypoint="generate",
            payload=spec["payload"],
            outputs=("image",),
            request_id=spec["request_id"],
        ),
        [spec["binding"]],
        workspace=workspace,
        release_id=RELEASE,
        options=WorkerOptions(root=workspace / "worker", release_id=RELEASE, no_warm=True),
        delete_journal=False,
    )
    out_path.write_text(json.dumps(dataclasses.asdict(outcome), default=str))


def exec_conform(spec_path: Path, out_path: Path) -> None:
    """One denoise step per variant on IDENTICAL inputs: the step wall and the deviation.

    The UNet alone, twice in one process, because the experiment is the WEIGHTS: holding
    the inputs bit-identical across two fills is the whole comparison, and the other three
    components are byte-identical between the arms anyway.
    """
    import torch

    spec = json.loads(spec_path.read_text())
    result: dict[str, Any] = {"rows": {}}
    outputs: dict[str, Any] = {}
    for record in spec["variants"]:
        Q.require_quiet_card()
        Q.drop_page_cache(Path(record["store"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        backend, _model, _ = Q._fill(record["store"], record["snapshot"])
        unet = backend.components["unet"]
        prediction, step_ms = Q._one_forward(torch, unet)
        result["rows"][record["name"]] = {
            "step_ms": step_ms,
            "unet_fill_ms": backend.stats["fill_ms"],
            "unet_decode_ms": backend.stats["decode"]["ms"],
            "unet_stored_bytes": backend.stats["lease_bytes"],
            "unet_weights_bytes": backend.ledger.peak["vram"],
            "unet_scratch_bytes": backend.ledger.peak["decode"],
        }
        outputs[record["name"]] = prediction
        backend.close()
        for name in list(backend.components):
            backend.evict(name)
        del backend, unet
        torch.cuda.empty_cache()
    reference = next(v["name"] for v in spec["variants"] if v["reference"])
    ref = outputs[reference]
    for name, prediction in outputs.items():
        rel = float((prediction - ref).norm().item() / ref.norm().item())
        cosine = float(
            (prediction.flatten() @ ref.flatten()).item()
            / (prediction.norm().item() * ref.norm().item())
        )
        result["rows"][name]["deviation_rel_l2"] = rel
        result["rows"][name]["cosine"] = cosine
    out_path.write_text(json.dumps(result))


# ------------------------------------------------------------------------ calibrate


def table_path() -> Path:
    import torch

    from cozy_runtime.internal.delivery import DeliveryTable
    from cozy_runtime.internal.encoding import measure_device

    return DeliveryTable.path_for(HOME / "probe", measure_device(torch, 0))


def load_table() -> Any:
    import torch

    from cozy_runtime.internal.delivery import DeliveryTable
    from cozy_runtime.internal.encoding import measure_device

    device = measure_device(torch, 0)
    return DeliveryTable.load(DeliveryTable.path_for(HOME / "probe", device), device), device


def calibrate() -> None:
    head("calibrate — one real generation per variant; each banks its own fill envelope")
    payload = {
        "prompt": "an astronaut riding a horse on the moon, dramatic lighting",
        "negative_prompt": "blurry, low quality",
        "steps": 20,
        "size": 512,
        "guidance": 5.0,
        "seed": 1005,
    }
    offered = variants()
    for record in offered:
        # COLD, EVERY TIME. A fill wall banked over a warm page cache is not a cold-start
        # number, and the two variants share three of their four components — so running
        # them back to back without evicting measures the second one's luck. Measured
        # exactly that: a warm fp16 arm banked 907 ms against its own cold 5262 ms and
        # took the latency axis it loses cold.
        evicted = Q.drop_page_cache(Path(record["store"]))
        print(f"  evicted {evicted['bytes'] / GIB:.2f} GiB of page cache before {record['name']}")
        outcome = spawn(
            f"cal-{record['name']}",
            {
                # ONE variant offered, so the choice is forced and what is measured is that
                # variant's own envelope. The chooser still runs and still says why.
                "binding": binding_record([dict(record, reference=True)], objective="latency"),
                "payload": payload,
                "request_id": f"cal-{record['name']}",
            },
        )
        plan = outcome.get("accepted") or {}
        outputs = outcome.get("outputs") or {}
        print(
            f"  {record['name']:<6}{outcome.get('status', '?'):<12}"
            f"delivery={plan.get('delivery', '?')}/{plan.get('materialization') or 'verbatim'} "
            f"outputs={list(outputs)}"
        )
        check(
            f"{record['name']}: a real image through the whole attempt path",
            outcome.get("status") == "TERMINAL_STATUS_SUCCEEDED" and bool(outputs),
            str(plan.get("quantified_choice", ""))[:160],
        )
    table, _ = load_table()
    for key in sorted(table.rows):
        row = table.rows[key]
        print(
            f"  banked {row.variant:<6}{row.rung:<15}stored {row.stored_bytes / GIB:>7.3f}G  "
            f"fill {row.fill_ms:>6} ms  decode {row.decode_ms:>5} ms  "
            f"resident {row.resident_bytes / GIB:>6.3f}G  scratch {row.scratch_bytes / MIB:>6.0f}M"
        )
    check(
        "both variants banked a MEASURED fill envelope",
        len(table.rows) >= 2,
        f"{len(table.rows)} rows in {table_path().name}",
    )
    check(
        "a prepare INVENTS no step time and no deviation",
        all(row.step_ms != 0.0 for row in table.rows.values()),
        "a prepare measures a load; the step wall and the deviation stay whatever a real "
        "forward pass banked (-1.0 until one has), because a zero would win every latency "
        "comparison it was never entitled to enter",
    )


def conform() -> None:
    head("conform — one denoise step per variant: the step wall, and the deviation")
    from cozy_runtime.internal.delivery import DeliveryFact

    offered = variants()
    # BOTH ORDERS, in two processes. The first pass measured the second-filled arm 12%
    # slower and the only honest way to tell a WEIGHTS effect from a position-in-the-
    # process effect is to swap the position and look again. Whichever way it comes out,
    # the number banked is the one measured with the arm FIRST — the only position where
    # nothing else has already run on the allocator.
    forward = spawn("conform-a", {"variants": offered}, mode="exec-conform")
    backward = spawn("conform-b", {"variants": list(reversed(offered))}, mode="exec-conform")
    print("  step wall by POSITION in the process (ms):")
    for name in ("fp16", "fp8"):
        first = forward["rows"][name] if offered[0]["name"] == name else backward["rows"][name]
        second = backward["rows"][name] if offered[0]["name"] == name else forward["rows"][name]
        print(
            f"    {name:<6}first {first['step_ms']:>7.1f}   second {second['step_ms']:>7.1f}   "
            f"delta {100 * (second['step_ms'] / first['step_ms'] - 1):>+6.1f}%"
        )
    result = {
        "rows": {
            name: (forward if offered[0]["name"] == name else backward)["rows"][name]
            for name in ("fp16", "fp8")
        }
    }
    table, _device = load_table()
    for name, row in sorted(result["rows"].items()):
        print(
            f"  {name:<6}step {row['step_ms']:>7.1f} ms   unet fill {row['unet_fill_ms']:>6} ms   "
            f"decode {row['unet_decode_ms']:>4} ms   deviation {row['deviation_rel_l2']:.5f}   "
            f"cosine {row['cosine']:.6f}"
        )
        for key in [k for k in table.rows if k[0] == name]:
            prior = table.rows[key]
            table.put(
                DeliveryFact(
                    variant=prior.variant,
                    rung=prior.rung,
                    stored_bytes=prior.stored_bytes,
                    resident_bytes=prior.resident_bytes,
                    scratch_bytes=prior.scratch_bytes,
                    fill_ms=prior.fill_ms,
                    decode_ms=prior.decode_ms,
                    step_ms=row["step_ms"],
                    deviation_rel_l2=row["deviation_rel_l2"],
                    samples=prior.samples,
                    evidence=(
                        f"{prior.evidence}; step + deviation from one denoise step on "
                        f"identical inputs (delivery-live conform, SINGLE SAMPLE)"
                    ),
                    encodings=prior.encodings,
                )
            )
    table.save(table_path())
    fp8 = result["rows"]["fp8"]
    fp16 = result["rows"]["fp16"]
    check(
        "the encoded variant is a CONFORMANCE observation, not an assertion",
        fp8["deviation_rel_l2"] < 0.05 and fp8["cosine"] > 0.998,
        f"relative L2 {fp8['deviation_rel_l2']:.5f}, cosine {fp8['cosine']:.6f}",
    )
    check(
        "the reference's deviation from itself is exactly zero",
        fp16["deviation_rel_l2"] == 0.0,
        "the fidelity origin is a measured identity, not a convention",
    )
    check(
        "KERNEL-NEUTRAL: the decoded rung's step time is the plain rung's",
        abs(fp8["step_ms"] / fp16["step_ms"] - 1.0) < 0.10,
        f"{fp8['step_ms']:.1f} vs {fp16['step_ms']:.1f} ms per UNet forward, each measured "
        "in FIRST position — the same cuBLAS fp16 GEMM on both",
    )


# --------------------------------------------------------------------------- choose


def choose() -> None:
    head("choose — the chooser over the banked table, under each objective")
    import torch

    from cozy_runtime.internal.delivery import DeliveryChooser, Variant
    from cozy_runtime.internal.encoding import Selector, launch_providers
    from cozy_runtime.internal.fill import Checkpoint, dtype_name, survey_routes
    from cozy_runtime.internal.probe import qualify

    table, device = load_table()
    if not table.rows:
        raise SystemExit("no delivery table — run `calibrate` and `conform` first")
    providers = launch_providers()
    candidates = []
    all_rows = []
    for record in variants():
        variant = Variant.read(record)
        vmap = variant.snapshot_map()
        rows = [
            r
            for name in COMPONENTS
            for r in Checkpoint(variant.store, vmap.get(name, variant.snapshot)).rows(name)
        ]
        all_rows.append((variant, rows))
    capabilities, _ = qualify(
        torch,
        providers,
        device,
        sorted({dtype_name(r.dtype) for _, rows in all_rows for r in rows}),
        release=RELEASE,
        cache=HOME / "probe",
    )
    selector = Selector(providers, capabilities, device)
    candidates = [(variant, *survey_routes(rows, selector)) for variant, rows in all_rows]

    picked: dict[str, str] = {}
    for objective in ("latency", "footprint", "fidelity"):
        choice = DeliveryChooser(table, device).choose(candidates, objective=objective)
        picked[objective] = choice.variant.name
        print(f"  {objective:<10}-> {choice.variant.name} / {choice.rung.name}")
        for row in choice.walk:
            score = f"{row['score']:>14.3f}" if "score" in row else " " * 14
            print(f"      {row['variant']:<6}{row.get('rung', '-'):<15}{row['verdict']:<20}{score}")
        if choice.confession:
            print(f"      confession: {choice.confession}")
    check(
        "the LATENCY axis takes the encoded rung — it reads fewer bytes for the same step",
        picked["latency"] == "fp8",
        f"{table.get('fp8', 'aot_decode').fill_ms} ms cold fill against "
        f"{table.get('fp16', 'verbatim').fill_ms} ms",
    )
    check(
        "the FIDELITY axis takes the reference — nothing beats zero deviation",
        picked["fidelity"] == "fp16",
        "an encoded rung is admitted by a conformance observation and ordered by it",
    )
    check(
        "the FOOTPRINT axis is the DEVICE-HONEST one and does NOT take the encoded rung",
        picked["footprint"] == "fp16",
        "weight-only fp8 on sm89 is decoded to float at fill: resident bytes are equal and "
        "the decode scratch is a real extra, so the encoded rung LOSES this axis. The rung "
        "that would win it is jit_decode, which is named and not built",
    )


# ---------------------------------------------------------------------------- serve


def serve() -> None:
    head("serve — both variants OFFERED: the choice reaches the attempt, on two axes")
    offered = variants()
    payload = {
        "prompt": "an astronaut riding a horse on the moon, dramatic lighting",
        "negative_prompt": "blurry, low quality",
        "steps": 4,
        "size": 512,
        "guidance": 5.0,
        "seed": 1005,
    }
    seen: dict[str, dict[str, Any]] = {}
    for objective in ("latency", "fidelity"):
        outcome = spawn(
            f"serve-{objective}",
            {
                "binding": binding_record(offered, objective=objective),
                "payload": payload,
                "request_id": f"serve-{objective}",
            },
        )
        plan = outcome.get("accepted") or {}
        seen[objective] = {"plan": plan, "outcome": outcome}
        print(
            f"  {objective:<10}{outcome.get('status', '?'):<12}"
            f"{plan.get('delivery', '?')}/{plan.get('materialization') or 'verbatim'} "
            f"outputs={list(outcome.get('outputs') or {})}"
        )
        print(f"      {str(plan.get('quantified_choice', ''))[:220]}")
        check(
            f"{objective}: a real image, end to end, on the chosen route",
            outcome.get("status") == "TERMINAL_STATUS_SUCCEEDED" and bool(outcome.get("outputs")),
            f"exit {outcome.get('exit_code')}",
        )
    check(
        "the SAME binding on two axes serves two different artifacts",
        seen["latency"]["plan"].get("delivery") == "float"
        and seen["fidelity"]["plan"].get("delivery") == "native",
        f"latency -> {seen['latency']['plan'].get('delivery')}/"
        f"{seen['latency']['plan'].get('materialization')}, fidelity -> "
        f"{seen['fidelity']['plan'].get('delivery')}",
    )
    check(
        "the chosen VARIANT is named on the plan a coordinator holds",
        "'fp8'" in str(seen["latency"]["plan"].get("quantified_choice"))
        and "'fp16'" in str(seen["fidelity"]["plan"].get("quantified_choice")),
        "the AttemptPlan document carries `delivery_variant`, so `plan_digest` binds WHICH "
        "bytes served",
    )


def staged() -> None:
    head("staged — the CROSS: a float delivery under a staged placement is per-TRANSITION")
    offered = variants()
    encoded = [dict(v, reference=v["name"] == "fp8") for v in offered if v["name"] == "fp8"]
    outcome = spawn(
        "staged",
        {
            # 2 GiB resident ceiling: the UNet is PARKED at construction and staged in on
            # first declared use, which is cr-008b's seam. Its decode therefore happens
            # inside `stage()` rather than inside the whole-generation fill.
            "binding": binding_record(encoded, objective="latency", resident_budget=2 * GIB),
            "payload": {
                "prompt": "an astronaut riding a horse on the moon",
                "negative_prompt": "",
                "steps": 4,
                "size": 512,
                "guidance": 5.0,
                "seed": 1005,
            },
            "request_id": "staged",
        },
    )
    plan = outcome.get("accepted") or {}
    print(
        f"  status={outcome.get('status')}  plan={plan.get('delivery')}/"
        f"{plan.get('materialization')}  placement={plan.get('placement')}"
    )
    print(f"      {str(plan.get('quantified_choice', ''))[:220]}")
    check(
        "the encoded component is STAGED and the plan says so",
        plan.get("placement") == "component_staged",
        "the resident ceiling parks the UNet; cr-008b's stage() moves and decodes it",
    )
    check(
        "the materialization LABEL becomes staged_decode — the cost class changed",
        plan.get("materialization") == "staged_decode",
        "aot_decode x component_staged is per-TRANSITION, not per-generation (pgw#1505): "
        "the same decode is paid again on every stage-in",
    )
    check(
        "and the KERNEL did not change with it",
        outcome.get("status") == "TERMINAL_STATUS_SUCCEEDED",
        "a placement rung changes weight placement only; the delivery stayed float and the "
        "GEMM stayed cuBLAS fp16",
    )


# ----------------------------------------------------------------------------- arms


def arms() -> None:
    head("arms — the red arms, each planted for real")
    import dataclasses

    import torch

    from cozy_runtime.internal.delivery import (
        RUNGS_BY_NAME,
        DeliveryChooser,
        DeliveryRefusal,
        DeliveryTable,
        Variant,
    )
    from cozy_runtime.internal.encoding import Capabilities, Selector, launch_providers
    from cozy_runtime.internal.fill import Checkpoint, dtype_name, survey_routes
    from cozy_runtime.internal.probe import qualify
    from cozy_runtime.internal.worker.ledger import Ledger
    from cozy_runtime.internal.worker.plan import BindingPlan, PlanChooser, PlanRefusal

    table, device = load_table()
    providers = launch_providers()
    prepared = []
    for record in variants():
        variant = Variant.read(record)
        vmap = variant.snapshot_map()
        rows = [
            r
            for name in COMPONENTS
            for r in Checkpoint(variant.store, vmap.get(name, variant.snapshot)).rows(name)
        ]
        prepared.append((variant, rows))
    capabilities, _ = qualify(
        torch,
        providers,
        device,
        sorted({dtype_name(r.dtype) for _, rows in prepared for r in rows}),
        release=RELEASE,
        cache=HOME / "probe",
    )
    selector = Selector(providers, capabilities, device)

    # ARM 1 — an execution-oriented artifact with no qualified route on this card.
    # The plant is REAL: the fp8 artifact's own rows, surveyed against an EMPTY capability
    # table, which is exactly the state of a device whose probe minted no record.
    blind = Selector(providers, Capabilities(), device)
    encoded_variant, encoded_rows = prepared[1]
    routes, refusal = survey_routes(encoded_rows, blind)
    try:
        DeliveryChooser(DeliveryTable(device.tuple_key()), device).choose(
            [(dataclasses.replace(encoded_variant, reference=True), routes, refusal)],
            objective="latency",
        )
        check("ARM: no qualified route refuses", False, "it did not refuse")
    except DeliveryRefusal as exc:
        check(
            "ARM: an artifact with no qualified provider refuses NO VALID EXECUTION PLAN",
            exc.code == "no_qualified_delivery" and "topology is known" in exc.detail,
            exc.detail[:230],
        )

    # ARM 2 — a registered encoding this build does not execute. `mxfp8/1` is real in the
    # registry and named-not-built here, so the refusal must name the DEVICE, not the
    # registry: "unknown topology" would send a reader to the wrong question entirely.
    import tensorfs

    from cozy_runtime.internal.encoding import Encoded, EncodingRefusal, RolePart

    mx = next(d for a, d in tensorfs.seed_digests() if a == "mxfp8/1")
    try:
        selector.select(
            "unet.planted.weight",
            Encoded(mx, "mxfp8/1", (RolePart("data", "f8_e4m3fn", (16, 16), 256),)),
            "float16",
        )
        check("ARM: a named-not-built encoding refuses", False, "it did not refuse")
    except EncodingRefusal as exc:
        check(
            "ARM: a REGISTERED unbuilt encoding refuses in the CAPABILITY vocabulary",
            exc.code == "encoding_unqualified" and "sm89" in str(exc),
            str(exc)[:230],
        )

    # ARM 3 — a delivery switch as a PLACEMENT SIDE EFFECT. The plant mutates the closed
    # ladder so the staged cross lands on a rung with a DIFFERENT kernel, which is exactly
    # the §3.2 violation the fence exists for. Planted, observed, removed.
    # A ledger posed so the placement ladder REACHES the staged rung: 4 GiB resident, a
    # MEASURED 2 GiB activation envelope, and 1 GiB allocatable. all_resident is short by
    # 1 GiB; component_staged can free the UNet and raise 5 GiB, so the cross fires.
    ledger = Ledger(host_budget=8 * GIB, pinned_budget=256 * MIB)
    ledger.resident = {"unet": 4 * GIB}
    ledger.activations_peak = 2 * GIB
    ledger.device_free = 1 * GIB
    ledger.device_reserved = 0
    ledger.device_allocated = 0
    binding = BindingPlan.read(
        dict(binding_record(variants()), delivery_rung="aot_decode", delivery_variant="fp8")
    )
    chooser = PlanChooser(ledger)
    good = chooser.choose(binding, {"entrypoint": "generate", "scalars": {}})
    original = RUNGS_BY_NAME["staged_decode"]
    RUNGS_BY_NAME["staged_decode"] = type(original)(
        name=original.name,
        delivery="native",
        materialization="",
        route="native_encoded",
        cost_class=original.cost_class,
        why=original.why,
        status=original.status,
    )
    try:
        planted = chooser.choose(binding, {"entrypoint": "generate", "scalars": {}})
        if planted.placement != "component_staged":
            check(
                "ARM: the plant needs the staged cross to fire",
                False,
                f"the ladder chose {planted.placement}, so the cross never ran",
            )
        else:
            check("ARM: a delivery side effect refuses", False, "it did not refuse")
    except PlanRefusal as exc:
        check(
            "ARM: a delivery switch arriving as a PLACEMENT side effect refuses typed",
            exc.code == "delivery_side_effect",
            exc.detail[:230],
        )
    finally:
        RUNGS_BY_NAME["staged_decode"] = original
    after = chooser.choose(binding, {"entrypoint": "generate", "scalars": {}})
    check(
        "the plant is REMOVED: the same choice is served again",
        after.delivery == good.delivery and after.materialization == good.materialization,
        f"{after.delivery}/{after.materialization}",
    )

    # ARM 4 — an unmeasured rung may not be SELECTED. Empty table, both variants offered.
    empty = DeliveryChooser(DeliveryTable(device.tuple_key()), device)
    candidates = [(variant, *survey_routes(rows, selector)) for variant, rows in prepared]
    choice = empty.choose(candidates, objective="latency")
    check(
        "ARM: with NOTHING measured the ladder takes the reference and confesses",
        choice.variant.name == "fp16" and not choice.calibrated,
        choice.confession[:200],
    )

    # ARM 5 — a variant with a fill envelope but NO conformance observation is ineligible.
    from cozy_runtime.internal.delivery import DeliveryFact

    partial = DeliveryTable(device.tuple_key())
    partial.put(DeliveryFact("fp16", "verbatim", 1, 1, 0, 3500, 0, 128.0, 0.0, 1, "planted"))
    partial.put(DeliveryFact("fp8", "aot_decode", 1, 1, 0, 2000, 79, 128.0, -1.0, 1, "planted"))
    choice = DeliveryChooser(partial, device).choose(candidates, objective="latency")
    check(
        "ARM: a FASTER variant with no conformance observation is not selected",
        choice.variant.name == "fp16",
        "arithmetic fit admits nothing: 2000 ms beats 3500 ms and still loses, because an "
        "encoded rung is admitted by evidence that it is still this model",
    )

    # ARM 6 — a candidate set with no fidelity reference has no scale to measure on.
    try:
        DeliveryChooser(table, device).choose(
            [
                (Variant(v.name, v.store, v.snapshot, v.snapshots, False), r, q)
                for (v, r, q) in candidates
            ],
            objective="fidelity",
        )
        check("ARM: a set with no reference refuses", False, "it did not refuse")
    except DeliveryRefusal as exc:
        check(
            "ARM: a candidate set with no fidelity REFERENCE refuses",
            exc.code == "reference_variant_unresolved",
            exc.detail[:200],
        )

    # ARM 7 — an unknown objective. The axis has to be one something measured.
    try:
        DeliveryChooser(table, device).choose(candidates, objective="cheapest")
        check("ARM: an unmeasured objective refuses", False, "it did not refuse")
    except DeliveryRefusal as exc:
        check(
            "ARM: an objective no measurement supports refuses",
            exc.code == "unknown_objective",
            exc.detail[:160],
        )

    # ARM 8 — the jit_decode rung is NAMED and NOT BUILT, and says so with its blocker.
    jit = RUNGS_BY_NAME["jit_decode"]
    check(
        "ARM: the per-step rung refuses as unbuilt and names WHY, not silently aliases",
        jit.status == "unbuilt" and "LEAF REPLACEMENT" in jit.owner,
        jit.owner[:200],
    )


# ------------------------------------------------------------------------------ main


def main(argv: list[str]) -> int:
    if argv and argv[0] == "exec":
        exec_one(Path(argv[1]), Path(argv[2]))
        return 0
    if argv and argv[0] == "exec-conform":
        exec_conform(Path(argv[1]), Path(argv[2]))
        return 0
    sections = {
        "variants": survey,
        "calibrate": calibrate,
        "conform": conform,
        "choose": choose,
        "serve": serve,
        "staged": staged,
        "arms": arms,
    }
    names = argv or ["variants"]
    if names == ["all"]:
        names = list(sections)
    started = time.perf_counter()
    for name in names:
        if name not in sections:
            raise SystemExit(f"unknown section {name!r}; have {sorted(sections)}")
        sections[name]()
    print(f"\n{PASS} ok, {FAIL} failed in {time.perf_counter() - started:.1f} s")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
