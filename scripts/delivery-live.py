#!/usr/bin/env python3
"""cr-008c live verification: the DELIVERY axis, chosen over two real SDXL artifacts.

Not a test (tracker README #160): every leg runs the real chooser, the real prepare, the
real attempt path and the real package, and prints numbers.

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
            "snapshots": dict(sorted(plain.items())),
            "reference": True,
        },
        {
            "name": "fp8",
            "store": str(PLAIN_STORE),
            "snapshot": encoded["unet"],
            "snapshots": dict(sorted(encoded.items())),
            "reference": False,
        },
    ]


def binding_record(
    offered: list[dict[str, Any]], objective: str = "latency", resident_budget: int = 0
) -> dict[str, Any]:
    """The deployment record. `variants` is the deployment's OFFER — several artifacts for
    one logical model — and it stays in the record; what left it (#567e) are the single
    artifact's own facts, which the index row below carries instead."""
    return {
        "entrypoint_binding_digest": Q.plan_id_for("generate"),
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
        "resident_budget_bytes": resident_budget,
        "tenancy": "local",
        "variants": offered,
        "objective": objective,
        "steps_basis": 20,
    }


def install_row(offered: list[dict[str, Any]] | None = None) -> None:
    """The artifact index row the record's ref resolves against (#567e)."""
    from cozy_runtime.internal import artifacts

    reference = next(v for v in (offered or variants()) if v["reference"])
    HOME.mkdir(parents=True, exist_ok=True)
    artifacts.install(
        HOME,
        artifacts.Artifact(
            ref=RELEASE,
            store=str(PLAIN_STORE),
            config=str(Q.PIPE_CONFIG),
            snapshots=dict(reference["snapshots"]),
            vram_floor_bytes=1 * GIB,
        ),
    )


def survey() -> None:
    head("variants — what each artifact's HEADER demands, and what plan that resolves to")

    from cozy_runtime.internal.delivery import DELIVERY_RUNGS
    from cozy_runtime.internal.encoding import device_line
    from cozy_runtime.internal.fill import servability_for_script
    from cozy_runtime.internal.resolution import route_label

    print("  the delivery routes, and WHEN each one is paid (data, not a chooser):")
    for rung in DELIVERY_RUNGS:
        mark = "built" if rung.status == "proven" else "NAMED, NOT BUILT"
        print(
            f"    {rung.name:<15}{rung.delivery}/{rung.materialization or '-':<14}"
            f"{rung.cost_class:<16}{mark}"
        )

    context = resolved()
    device, probe = context["device"], context["probe"]
    print(f"  device: {device_line(device)}")
    labels: dict[str, str] = {}
    for name, rows in context["rows_for"].items():
        answer = servability_for_script(
            rows,
            release=RELEASE,
            providers=context["providers"],
            capabilities=context["capabilities"],
            device=device,
            facts_dir=str(HOME / "probe"),
        )
        routes, refusal = answer.routes, answer.refusal
        labels[name] = route_label(routes)
        aliases = sorted({r.encoded.alias for r in rows})
        print(
            f"  {name:<6}{len(rows):>5} tensors  {sum(r.stored_nbytes for r in rows) / GIB:>7.3f} "
            f"GiB stored  routes={routes}  encodings={aliases}"
        )
        print(f"         plan label: {labels[name]}  qualification: {refusal or 'every encoding'}")
    check(
        "the two variants are the SAME logical model",
        [r.key for r in context["rows_for"]["fp16"]] == [r.key for r in context["rows_for"]["fp8"]],
        f"{len(context['rows_for']['fp16'])} destinations, key for key",
    )
    check(
        "each variant's plan LABEL derives from its own per-tensor route census (#549.1)",
        labels["fp8"] == "aot_decode" and labels["fp16"] == "verbatim",
        f"fp8 -> {labels['fp8']}, fp16 -> {labels['fp16']}; a rung name used to be CHOSEN "
        "artifact-wide and is now DERIVED, so a census with more than one route is "
        "unnameable as a single rung and says `mixed(...)` instead",
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
    """ONE request through the WHOLE real path — worker, executor, package."""
    import dataclasses
    import os

    from cozy_runtime.internal.config import read_config
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions

    spec = json.loads(spec_path.read_text())
    workspace = Path(spec["workspace"])
    HOME.mkdir(parents=True, exist_ok=True)
    install_row(spec["binding"].get("variants"))
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
        package_revision_digest=RELEASE,
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
            "unet_weights_bytes": sum(backend.vram_charge.values()),
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


def plan_facts() -> Any:
    """The #549.2 store, rooted where the executor roots it. NOT the old delivery table:
    those rows were keyed by `(variant, rung)`, which bound neither the snapshot bytes nor
    the runtime build nor the provider map, and they are INVALID rather than migrated."""
    from cozy_runtime.internal.planfacts import PlanFacts

    return PlanFacts(HOME / "probe")


def resolved() -> Any:
    """Resolve every offered variant into plans, exactly as the executor does.

    ONE call, one decision. The old shape needed three (`survey_routes` per variant, then a
    `DeliveryChooser` over a rung table, then a `Selector` per tensor at fill), and the
    second of those could not see what the third would do.
    """
    import torch

    from cozy_runtime.internal import canonical
    from cozy_runtime.internal.encoding import launch_providers, measure_device, measure_runtime
    from cozy_runtime.internal.fill import Checkpoint, dtype_name, topology_of
    from cozy_runtime.internal.probe import qualify
    from cozy_runtime.internal.resolution import ConstructionFacts, Variant

    device = measure_device(torch, 0)
    runtime = measure_runtime(torch, RELEASE)
    providers = launch_providers()
    offered = [Variant.read(record) for record in variants()]
    rows_for: dict[str, list[Any]] = {}
    for variant in offered:
        vmap = variant.snapshot_map()
        rows_for[variant.name] = [
            r
            for name in COMPONENTS
            for r in Checkpoint(variant.store, vmap.get(name, variant.snapshot)).rows(name)
        ]
    capabilities, probe = qualify(
        torch,
        providers,
        device,
        sorted({dtype_name(r.dtype) for rows in rows_for.values() for r in rows}),
        release=RELEASE,
        cache=HOME / "probe",
        runtime=runtime,
    )
    reference = offered[0]
    construction = ConstructionFacts(
        release=RELEASE,
        model_class="corpus.sdxl_pipeline:Pipeline",
        config_digest=canonical.digest({"fixture": "delivery-live"}),
        topology_digest=canonical.digest(
            {
                key: [list(spec.shape), spec.dtype, spec.encoding]
                for key, spec in sorted(topology_of(rows_for[reference.name]).items())
            }
        ),
        components=tuple(COMPONENTS),
    )
    return {
        "device": device,
        "runtime": runtime,
        "providers": providers,
        "capabilities": capabilities,
        "probe": probe,
        "offered": offered,
        "rows_for": rows_for,
        "construction": construction,
        "dtype_name": dtype_name,
    }


def plan_for(name: str, objective: str = "latency") -> Any:
    """The plan ONE offered variant resolves to, alone. What `calibrate` banks against.

    Resolving the variant on its own forces the choice, so what comes back is that variant's
    own plan and its own digest — which is the key its measurements belong under.
    """
    import dataclasses

    from cozy_runtime.internal.resolution import resolve

    context = resolved()
    only = next(
        dataclasses.replace(v, reference=True) for v in context["offered"] if v.name == name
    )
    return resolve(
        variants=[only],
        rows_for={name: context["rows_for"][name]},
        providers=context["providers"],
        capabilities=context["capabilities"],
        device=context["device"],
        runtime=context["runtime"],
        construction=context["construction"],
        facts=plan_facts(),
        objective=objective,
        dtype_name=context["dtype_name"],
    )


def banked_plans() -> dict[str, Any]:
    """Every plan digest this rig has banked a fact for, folded."""
    store = plan_facts()
    return {digest: store.get(digest) for digest in store.digests()}


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
            outcome.get("status") == "OUTCOME_STATUS_SUCCEEDED" and bool(outputs),
            str(plan.get("quantified_choice", ""))[:160],
        )
    banked = banked_plans()
    for digest, fact in sorted(banked.items()):
        print(
            f"  banked {digest[:19]}… stored {fact.stored_bytes / GIB:>7.3f}G  "
            f"fill {fact.fill_ms:>6} ms  decode {fact.decode_ms:>5} ms  "
            f"resident {fact.resident_bytes / GIB:>6.3f}G  "
            f"scratch {fact.scratch_bytes / MIB:>6.0f}M  samples {fact.samples}"
        )
    check(
        "both variants banked a MEASURED fill envelope, keyed by PLAN DIGEST (#549.2)",
        len(banked) >= 2,
        f"{len(banked)} plan directories — a `(variant, rung)` key bound neither the "
        "snapshot bytes nor the runtime build, so a re-pointed variant name reused "
        "measurements taken over unrelated bytes",
    )
    check(
        "a prepare INVENTS no step time and no deviation",
        all(fact.step_ms != 0.0 for fact in banked.values()),
        "a prepare measures a load; the step wall and the deviation stay whatever a real "
        "forward pass banked (-1.0 until one has), because a zero would win every latency "
        "comparison it was never entitled to enter",
    )


def conform() -> None:
    head("conform — one denoise step per variant: the step wall, and the deviation")
    from cozy_runtime.internal.planfacts import RunFact, run_id

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
    store = plan_facts()
    for name, row in sorted(result["rows"].items()):
        print(
            f"  {name:<6}step {row['step_ms']:>7.1f} ms   unet fill {row['unet_fill_ms']:>6} ms   "
            f"decode {row['unet_decode_ms']:>4} ms   deviation {row['deviation_rel_l2']:.5f}   "
            f"cosine {row['cosine']:.6f}"
        )
        # A SECOND, IMMUTABLE FACT FILE against the same plan digest — never an edit of the
        # prepare's. The prepare measured a fill and this measured a step; they are two
        # observations of one plan, and the fold takes each field from whichever runs
        # measured it. The old shape read the prepare's row, copied six fields out of it and
        # wrote the merged result back, which is exactly the lost-update the per-run files
        # remove (#549.2).
        digest = plan_for(name).digest()
        store.bank(
            RunFact(
                plan_digest=digest,
                run_id=run_id(digest, "delivery-live-conform", RELEASE),
                step_ms=row["step_ms"],
                reference_deviation_rel_l2=row["deviation_rel_l2"],
                evidence="step + deviation from one denoise step on identical inputs "
                "(delivery-live conform, SINGLE SAMPLE)",
                at_ns=time.time_ns(),
            )
        )
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
    head("choose — ONE plan resolution over the banked facts, under each objective")
    from cozy_runtime.internal.resolution import resolve

    context = resolved()
    facts = plan_facts()
    if not banked_plans():
        raise SystemExit("no banked plan facts — run `calibrate` and `conform` first")

    picked: dict[str, Any] = {}
    for objective in ("latency", "footprint", "kernel_numeric_error"):
        plan = resolve(
            variants=context["offered"],
            rows_for=context["rows_for"],
            providers=context["providers"],
            capabilities=context["capabilities"],
            device=context["device"],
            runtime=context["runtime"],
            construction=context["construction"],
            facts=facts,
            objective=objective,
            dtype_name=context["dtype_name"],
        )
        picked[objective] = plan
        print(f"  {objective:<22}-> {plan.summary()}")
        for row in plan.walk:
            score = f"{row['score']:>14.3f}" if "score" in row else " " * 14
            print(
                f"      {row.get('variant', '-'):<6}{row.get('policy', '-'):<14}"
                f"{row['verdict']:<20}{score}  "
                + str(row.get("provenance") or row.get("why", ""))[:70]
            )
        if plan.confession:
            print(f"      confession: {plan.confession[:180]}")
    check(
        "the LATENCY axis takes the encoded variant — it reads fewer bytes for the same step",
        picked["latency"].variant == "fp8",
        f"latency is scored from a WHOLE-PLAN fill+step wall, which is the measurement the "
        f"old per-provider admission had nothing of — and refused for (#549.3). "
        f"{picked['latency'].summary()}",
    )
    check(
        "the KERNEL_NUMERIC_ERROR axis takes the reference — nothing beats zero deviation",
        picked["kernel_numeric_error"].variant == "fp16",
        "renamed from `fidelity` (#549.11): it is a kernel diagnostic over one GEMM, and "
        "model quality is a paired video+audio eval the eval plane owns",
    )
    check(
        "the FOOTPRINT axis is the DEVICE-HONEST one and does NOT take the encoded variant",
        picked["footprint"].variant == "fp16",
        "weight-only fp8 on sm89 is decoded to float at fill: resident bytes are equal and "
        "the decode scratch is a real extra, so the encoded plan LOSES this axis. The route "
        "that would win it is the encoded GEMM, whose leaf this card does not qualify for "
        "these bytes",
    )
    check(
        "every chosen plan's LABEL derives from its own route census, not from a rung name",
        all(
            plan.label() == ("aot_decode" if plan.variant == "fp8" else "verbatim")
            for plan in picked.values()
        ),
        ", ".join(f"{o}:{p.label()}" for o, p in sorted(picked.items())),
    )
    check(
        "the plan DIGEST is what enters generation identity, and it moves with the BYTES",
        picked["latency"].digest() != picked["footprint"].digest()
        and picked["footprint"].digest() == picked["kernel_numeric_error"].digest(),
        "two objectives that landed on the same variant, same policy and same providers "
        "resolve the SAME plan; the one that landed elsewhere does not. A rung name could "
        "not tell those apart",
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
    for objective in ("latency", "kernel_numeric_error"):
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
            outcome.get("status") == "OUTCOME_STATUS_SUCCEEDED" and bool(outcome.get("outputs")),
            f"exit {outcome.get('exit_code')}",
        )
    check(
        "the SAME binding on two axes serves two different artifacts",
        seen["latency"]["plan"].get("delivery") == "float"
        and seen["kernel_numeric_error"]["plan"].get("delivery") == "native",
        f"latency -> {seen['latency']['plan'].get('delivery')}/"
        f"{seen['latency']['plan'].get('materialization')}, fidelity -> "
        f"{seen['kernel_numeric_error']['plan'].get('delivery')}",
    )
    check(
        "the chosen VARIANT is named on the plan a RecordOwner holds",
        "'fp8'" in str(seen["latency"]["plan"].get("quantified_choice"))
        and "'fp16'" in str(seen["kernel_numeric_error"]["plan"].get("quantified_choice")),
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
        outcome.get("status") == "OUTCOME_STATUS_SUCCEEDED",
        "a placement rung changes weight placement only; the delivery stayed float and the "
        "GEMM stayed cuBLAS fp16",
    )


# ----------------------------------------------------------------------------- arms


def arms() -> None:
    head("arms — the red arms, each planted for real")
    import dataclasses

    from cozy_runtime.internal.delivery import ROUTE_KERNEL, RUNGS_BY_NAME
    from cozy_runtime.internal.encoding import Capabilities, Selector
    from cozy_runtime.internal.planfacts import PlanFacts, RunFact, run_id
    from cozy_runtime.internal.resolution import PlanRefusal as ResolutionRefusal
    from cozy_runtime.internal.resolution import Variant, resolve
    from cozy_runtime.internal.worker.ledger import Ledger
    from cozy_runtime.internal.worker.plan import (
        DeclaredBinding,
        PlanChooser,
        PlanRefusal,
        PreparedModel,
        PreparedRequest,
    )

    context = resolved()
    device = context["device"]
    facts = plan_facts()

    def resolve_with(**changes: Any) -> Any:
        base: dict[str, Any] = {
            "variants": context["offered"],
            "rows_for": context["rows_for"],
            "providers": context["providers"],
            "capabilities": context["capabilities"],
            "device": device,
            "runtime": context["runtime"],
            "construction": context["construction"],
            "facts": facts,
            "objective": "latency",
            "dtype_name": context["dtype_name"],
        }
        base.update(changes)
        return resolve(**base)

    # ARM 1 — an execution-oriented artifact with no qualified route on this card.
    # The plant is REAL: the fp8 artifact's own rows, resolved against an EMPTY capability
    # table, which is exactly the state of a device whose probe minted no record.
    encoded = next(v for v in context["offered"] if v.name == "fp8")
    try:
        resolve_with(
            variants=[dataclasses.replace(encoded, reference=True)],
            rows_for={"fp8": context["rows_for"]["fp8"]},
            capabilities=Capabilities(),
        )
        check("ARM: no qualified route refuses", False, "it did not refuse")
    except ResolutionRefusal as exc:
        check(
            "ARM: an artifact with no qualified provider refuses NO EXECUTABLE PLAN",
            exc.code == "no_qualified_plan" and "topology is known" in exc.detail,
            exc.detail[:230],
        )

    # ARM 2 — REARMED BY #549.4, and it is the arm that had gone stale. It used to plant an
    # `mxfp8/1` digest and expect a named-not-built CAPABILITY refusal; `mxfp8/1` acquired
    # two real providers in #547, so the plant stopped reaching the branch it was aimed at
    # and the arm has been red ever since (`role_mismatch`, not `encoding_unqualified`).
    #
    # The plant that reaches the same QUESTION now is a spec digest the registry carries and
    # NO provider in this build reviewed: `svdq-microscale/1`. Under the old alias-keyed
    # launch table a new digest under a REVIEWED alias inherited its sibling's decoder, which
    # is the substitution #549.4 forbids; here it acquires nothing and says which of the two
    # things went wrong.
    import tensorfs

    from cozy_runtime.internal.encoding import Encoded, EncodingRefusal, RolePart

    selector = Selector(context["providers"], context["capabilities"], device)
    unreviewed = next(d for a, d in tensorfs.seed_digests() if a == "svdq-microscale/1")
    try:
        selector.select(
            "unet.planted.weight",
            Encoded(
                unreviewed, "svdq-microscale/1", (RolePart("data", "f8_e4m3fn", (16, 16), 256),)
            ),
            "float16",
        )
        check("ARM: an unreviewed spec digest refuses", False, "it did not refuse")
    except EncodingRefusal as exc:
        check(
            "ARM: a REGISTERED, NAMED-NOT-BUILT encoding refuses in the CAPABILITY vocabulary",
            exc.code == "encoding_unqualified" and "sm100" in str(exc),
            str(exc)[:230],
        )

    # ARM 2b — #549.4's own arm: a digest under an alias this build DOES serve, that no
    # provider reviewed. Planted by lying to the selector's provider map, which is exactly
    # the state a registry that grew a new `mxfp8/1` spec would put this build in.
    planted_digest = "sha256:" + "de" * 32
    naive = Selector(context["providers"], context["capabilities"], device)
    naive.aliases = dict(naive.aliases)
    naive.aliases[planted_digest] = "mxfp8/1"
    try:
        naive.select(
            "unet.planted.weight",
            Encoded(
                planted_digest,
                "mxfp8/1",
                (
                    RolePart("data", "f8_e4m3fn", (16, 64), 1024),
                    RolePart("scale", "u8", (16, 2), 32),
                ),
            ),
            "float16",
        )
        check("ARM: a new digest under a served alias refuses", False, "it did not refuse")
    except EncodingRefusal as exc:
        check(
            "ARM: a NEW digest filed under a SERVED alias acquires NO provider (#549.4)",
            exc.code == "spec_unreviewed" and "claims DIGESTS" in str(exc),
            str(exc)[:260],
        )

    # ARM 3 — a delivery switch as a PLACEMENT SIDE EFFECT. The plant mutates the route
    # table so the staged cross lands on a rung with a DIFFERENT kernel, which is exactly
    # the §3.2 violation the fence exists for. Planted, observed, removed.
    #
    # THE LEDGER POSE, CORRECTED. This arm was the second of delivery-live's two
    # pre-existing reds and it was a STALE ASSERTION, not a machinery failure: the pose
    # reasoned from `activations_peak = 2 GiB` against `device_free = 1 GiB` and expected
    # `all_resident` to come up short — but the activation coefficient BOUNDS the charge at
    # 1 GiB, so all_resident needed exactly 1 GiB against exactly 1 GiB allocatable and fit.
    # The ladder was right and the arm's arithmetic was not. 512 MiB free is short of the
    # bound charge, so `all_resident` misses, `component_staged` may free the 4 GiB resident
    # UNet and fits, and the cross the fence exists for actually runs.
    ledger = Ledger(host_budget=8 * GIB, pinned_budget=256 * MIB)
    ledger.resident = {"unet": 4 * GIB}
    ledger.activations_peak = 2 * GIB
    ledger.device_free = 512 * MIB
    ledger.device_reserved = 0
    ledger.device_allocated = 0
    offered = tuple(Variant.read(row) for row in variants())
    reference = next(row for row in offered if row.reference)
    declared = DeclaredBinding(
        entrypoint_binding_digest=Q.plan_id_for("generate"),
        entrypoint="generate",
        model_class="SdxlPipelineModel",
        model_binding_path="generate.models.model",
        model_parameter_name="model",
        release=RELEASE,
        store=str(PLAIN_STORE),
        variant="fp16",
        reference_snapshot=reference.snapshot,
        components=COMPONENTS,
        snapshots=reference.snapshot_map(),
        offered=offered,
        model_weight_bytes=4 * GIB,
        host_bytes=2 * GIB,
        pinned_bytes=256 * MIB,
    )
    generation = PreparedModel(delivery_rung="aot_decode", delivery_variant="fp8")
    chooser = PlanChooser(ledger)
    good = chooser.choose(declared, generation, PreparedRequest.unresolved("generate", {}))
    original = RUNGS_BY_NAME["staged_decode"]
    RUNGS_BY_NAME["staged_decode"] = type(original)(
        name=original.name,
        delivery="native",
        materialization="",
        route="encoded_gemm",
        cost_class=original.cost_class,
        why=original.why,
        status=original.status,
    )
    try:
        planted = chooser.choose(declared, generation, PreparedRequest.unresolved("generate", {}))
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
    after = chooser.choose(declared, generation, PreparedRequest.unresolved("generate", {}))
    check(
        "the plant is REMOVED: the same choice is served again",
        after.delivery == good.delivery and after.materialization == good.materialization,
        f"{after.delivery}/{after.materialization}",
    )

    # ARM 4 — an unmeasured plan may not be SELECTED. Empty fact store, both variants.
    plan = resolve_with(facts=PlanFacts(None))
    check(
        "ARM: with NOTHING measured resolution takes the reference FLOOR and CONFESSES",
        plan.variant == "fp16" and plan.measurements is None and "UNCALIBRATED" in plan.confession,
        plan.confession[:200],
    )

    # ARM 5 — #549.3's OWN ARM, and the one the old design could not pass. Two qualified
    # implementations for one tensor plus the DEFAULT objective used to refuse
    # `objective_unranked` at admission — one layer above the store that holds the latency
    # measurements. It now resolves, because the objective ranks WHOLE PLANS.
    check(
        "ARM: the DEFAULT objective + a multi-provider encoding no longer REFUSES (#549.3)",
        plan.objective == "latency" and bool(plan.tensors),
        f"{plan.summary()} — an axis with no per-record number is not a dead end, it is a "
        "confession; refusing here is what the two-chooser split forced",
    )

    # ARM 6 — a faster plan with NO conformance observation still loses to the reference.
    partial = PlanFacts(WORK / "arm-partial")
    shutil.rmtree(WORK / "arm-partial", ignore_errors=True)
    fp16_digest = plan_for("fp16").digest()
    fp8_digest = plan_for("fp8").digest()
    partial.bank(
        RunFact(
            plan_digest=fp16_digest,
            run_id=run_id(fp16_digest, "arm", RELEASE),
            resident_bytes=1,
            scratch_bytes=0,
            fill_ms=3500,
            step_ms=128.0,
            reference_deviation_rel_l2=0.0,
            evidence="planted",
        )
    )
    partial.bank(
        RunFact(
            plan_digest=fp8_digest,
            run_id=run_id(fp8_digest, "arm", RELEASE),
            resident_bytes=1,
            scratch_bytes=0,
            fill_ms=2000,
            step_ms=-1.0,
            evidence="planted",
        )
    )
    picked = resolve_with(facts=partial)
    check(
        "ARM: a FASTER plan with no measured STEP WALL is not selected",
        picked.variant == "fp16",
        "arithmetic fit admits nothing: a 2000 ms fill beats 3500 ms and still loses, "
        "because latency is fill PLUS step and nothing measured this plan's step",
    )

    # ARM 7 — a candidate set with no deviation reference has no scale to measure on.
    try:
        resolve_with(
            variants=[
                Variant(v.name, v.store, v.snapshot, v.snapshots, False) for v in context["offered"]
            ],
            objective="kernel_numeric_error",
        )
        check("ARM: a set with no reference refuses", False, "it did not refuse")
    except ResolutionRefusal as exc:
        check(
            "ARM: a candidate set with no deviation REFERENCE refuses",
            exc.code == "reference_variant_unresolved",
            exc.detail[:200],
        )

    # ARM 8 — an unknown objective. The axis has to be one something measured.
    try:
        resolve_with(objective="cheapest")
        check("ARM: an unmeasured objective refuses", False, "it did not refuse")
    except ResolutionRefusal as exc:
        check(
            "ARM: an objective no measurement supports refuses",
            exc.code == "unknown_objective",
            exc.detail[:160],
        )

    # ARM 9 — the two encoded routes, and the line between them. `encoded_gemm` is PROVEN
    # since the leaf-replacement lane; `jit_decode` is still unbuilt, and its blocker is no
    # longer "the machinery does not exist" — it is that a per-step DECODE is a different
    # thing from a per-step encoded KERNEL, and it refuses rather than aliasing itself onto
    # either neighbour.
    jit = RUNGS_BY_NAME["jit_decode"]
    native = RUNGS_BY_NAME["encoded_gemm"]
    check(
        "ARM: the per-step rung refuses as unbuilt and names WHY, not silently aliases",
        jit.status == "unbuilt" and "dequantize a full" in jit.owner,
        jit.owner[:200],
    )
    check(
        "the ENCODED-GEMM route is proven, and it is the only one that lowers steady-state VRAM",
        native.status == "proven" and ROUTE_KERNEL[native.route] == "encoded_gemm",
        f"{native.name}: {native.delivery}/{native.materialization or '-'} "
        f"cost_class={native.cost_class}, kernel={ROUTE_KERNEL[native.route]}",
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
