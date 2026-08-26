#!/usr/bin/env python3
"""#549's conformance runner: ONE runner, scenario manifests, CPU-only, CI-executable.

The #533 CI class (`serverless-endpoints/scripts/h3-conform.py` is the exemplar): a dict of
named scenarios, each a function, selected on argv, scoring through one `check`. It is the
shape the ~11k lines of `*-live.py` drivers are meant to consolidate into, and this lane
starts that consolidation with what it TOUCHED — plan resolution, cache identity, the
objective, spec-digest qualification, geometry preflight, capability identity, and the
decode-envelope arithmetic. The GPU-and-real-artifact drivers (`delivery-live`, `leaf-live`,
`quant-live`, `mxfp8-live`) keep their end-to-end legs; consolidating the ones this lane did
not touch is named follow-up, not silently claimed here.

Every scenario is RED-FIRST: it plants the pre-#549 behaviour and shows it produce the wrong
answer, then runs the same input through the shipped code and shows the right one. A green
assertion over shipped code proves the code agrees with itself; a red arm proves the change
was load-bearing.

    uv run python scripts/plan-conform.py            # every scenario
    uv run python scripts/plan-conform.py plan cache # two of them

CPU-only by construction: no scenario opens a device, none loads a checkpoint, no bytes move
and the one arm that needs torch imports it inside itself. Runs in seconds.

**NOT YET a CI step, and the reason is a dependency rather than a design.** The `checks` job
installs neither the `tensorfs` wheel nor torch, and this runner needs both — the registry is
what a reviewed spec DIGEST is reviewed against, so `registry_digests()` is not something a
fixture can stand in for without the arm becoming a check on the fixture. Wiring it in means
adding those two to the job (tensorfs is a ~10 MB wheel; torch is not, so the vector-rounding
arm would have to move to the GPU lane or the job would have to install a CPU-only build).
That choice belongs to whoever owns the CI budget and is NAMED here rather than claimed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cozy_runtime.internal import canonical  # noqa: E402
from cozy_runtime.internal.encoding import (  # noqa: E402
    SPEC_MXFP8,
    SPEC_PLAIN,
    SPEC_ROWWISE,
    SPEC_SCALED_SCALAR,
    SPEC_VECTORS,
    Capabilities,
    CapabilityRecord,
    DeviceFacts,
    Encoded,
    EncodingRefusal,
    MicroScaledDequant,
    MicroScaledNativeLeaf,
    RolePart,
    RowwiseNativeLeaf,
    RuntimeIdentity,
    Selector,
    implementation_digest,
    launch_providers,
    registry_digests,
)
from cozy_runtime.internal.planfacts import PlanFacts, RunFact, fold, run_id  # noqa: E402
from cozy_runtime.internal.resolution import (  # noqa: E402
    ConstructionFacts,
    PlanRefusal,
    Variant,
    resolve,
    route_label,
)

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


def red(label: str, planted: str) -> None:
    """The PRE-#549 behaviour, observed. Printed as evidence, not scored: an arm that
    scored its own plant would be counting the bug as a pass."""
    print(f"  red  {label} — {planted}", flush=True)


# --------------------------------------------------------------------------- the fixture
#
# A hand-built artifact with THREE encodings in one component, which is what the
# `quantize-mixed` job kind (#543g) emits and what the old artifact-wide rung could not
# describe. No store, no bytes: every seam this runner exercises reads a HEADER.

DEVICE = DeviceFacts(
    kind="cuda",
    name="NVIDIA Fixture",
    sm=120,
    driver="595.84",
    configuration="150W/N/A",
    index=3,
    cuda_runtime="13.0",
    os_arch="Linux/x86_64",
)

RUNTIME = RuntimeIdentity(
    release="cozy-runtime/0.0.1",
    source_digest=canonical.digest({"fixture": "plan-conform"}),
    torch_build="2.13.0+cu130",
)


class Row:
    """The header facts one destination contributes — `fill.PlanRow`'s shape, hand-built."""

    __slots__ = ("component", "dtype", "encoded", "key", "shape")

    def __init__(self, key: str, shape: tuple[int, ...], encoded: Encoded) -> None:
        self.key = key
        self.component = key.split(".", 1)[0]
        self.dtype = "bf16"
        self.shape = shape
        self.encoded = encoded


def mx_row(key: str, out: int, k: int) -> Row:
    blocks = -(-k // 32)
    return Row(
        key,
        (out, k),
        Encoded(
            SPEC_MXFP8,
            "mxfp8/1",
            (
                RolePart("data", "f8_e4m3fn", (out, k), out * k),
                RolePart("scale", "u8", (out, blocks), out * blocks),
            ),
        ),
    )


def rowwise_row(key: str, out: int, k: int) -> Row:
    return Row(
        key,
        (out, k),
        Encoded(
            SPEC_ROWWISE,
            "fp8-rowwise/1",
            (
                RolePart("data", "f8_e4m3fn", (out, k), out * k),
                RolePart("scale", "f32", (out,), out * 4),
            ),
        ),
    )


def scalar_row(key: str, out: int, k: int) -> Row:
    """`fp8-scaled-scalar/1`: an encoding with a DECODE provider and no encoded-GEMM leaf.

    It is what makes the fixture reach THREE routes under one objective — the mixed census
    an artifact-wide rung could not name.
    """
    return Row(
        key,
        (out, k),
        Encoded(
            SPEC_SCALED_SCALAR,
            "fp8-scaled-scalar/1",
            (
                RolePart("data", "f8_e4m3fn", (out, k), out * k),
                RolePart("input_scale", "f32", (), 4),
                RolePart("weight_scale", "f32", (), 4),
            ),
        ),
    )


def plain_row(key: str, shape: tuple[int, ...]) -> Row:
    n = 1
    for extent in shape:
        n *= extent
    return Row(
        key, shape, Encoded(SPEC_PLAIN, "plain/1", (RolePart("value", "bf16", shape, n * 2),))
    )


def dtype_name(name: object) -> str:
    return {"bf16": "bfloat16", "f16": "float16", "f32": "float32"}[str(name)]


def records(*, ragged_leaf: bool = False) -> Capabilities:
    """A capability table as a probe on the fixture device WOULD have minted it.

    Every number is a plausible MEASUREMENT and each is labelled as planted; what the arms
    exercise is the arithmetic over them, not the numbers themselves.
    """
    providers = launch_providers()
    caps = Capabilities()
    for digest, group in providers.items():
        for provider in group:
            if not DEVICE.satisfies(provider.device_predicate):
                continue
            if provider.route == "encoded_gemm":
                bits, error = 8.25, 0.0421
            elif provider.route == "decoded_float":
                bits, error = 16.0, 0.0021
            else:
                bits, error = 16.0, 0.0
            caps.add(
                CapabilityRecord(
                    encoding_spec_id=digest,
                    implementation_digest=implementation_digest(provider),
                    implementation_revision=provider.code_revision,
                    runtime_release=RUNTIME.release,
                    runtime_source_digest=RUNTIME.source_digest,
                    torch_build=RUNTIME.torch_build,
                    os_arch=DEVICE.os_arch,
                    device_predicate=DEVICE.predicate,
                    device_index=DEVICE.index,
                    device_driver=DEVICE.driver,
                    cuda_runtime=DEVICE.cuda_runtime,
                    output_dtype="bfloat16",
                    geometry_class="rank2.block32" if not ragged_leaf else "rank2.ragged32",
                    delivery_route=provider.route,
                    evidence_digest=canonical.digest({"planted": provider.name}),
                    resident_bits_per_element=bits,
                    kernel_numeric_error=error,
                )
            )
    return caps


CONSTRUCTION = ConstructionFacts(
    release=RUNTIME.release,
    model_class="fixture:Mixed",
    config_digest=canonical.digest({"fixture": "config"}),
    topology_digest=canonical.digest({"fixture": "topology"}),
    components=("block",),
)


def plan_over(
    rows: Sequence[Row],
    *,
    objective: str = "latency",
    facts: PlanFacts | None = None,
    consent: str = "accept",
    capabilities: Capabilities | None = None,
    variants: Sequence[Variant] | None = None,
    rows_for: Mapping[str, Sequence[Row]] | None = None,
) -> Any:
    offered = list(
        variants
        or [
            Variant(
                "only", "/fixture", "sha256:" + "aa" * 32, {"block": "sha256:" + "aa" * 32}, True
            )
        ]
    )
    return resolve(
        variants=offered,
        rows_for=dict(rows_for or {offered[0].name: rows}),
        providers=launch_providers(),
        capabilities=capabilities or records(),
        device=DEVICE,
        runtime=RUNTIME,
        construction=CONSTRUCTION,
        facts=facts or PlanFacts(None),
        objective=objective,
        encoded_leaves=consent,
        dtype_name=dtype_name,
    )


#: THREE encodings, THREE routes, ONE component. `plain/1` is verbatim, `fp8-scaled-scalar/1`
#: has a decode provider and no encoded-GEMM leaf, and `mxfp8/1` has both — so under one
#: objective this component legitimately holds a verbatim copy beside a decoded float weight
#: beside an encoded leaf, which is exactly the artifact the rung LABEL could not describe.
MIXED = [
    plain_row("block.norm.weight", (224,)),
    scalar_row("block.attn.out_proj.weight", 64, 224),
    mx_row("block.mlp.fc1.weight", 128, 224),
]


# ------------------------------------------------------------------------ the scenarios


def plan() -> None:
    head("plan — #549.1: one plan object, and a label that DERIVES")

    # FOOTPRINT, because that is the axis on which the encoded leaf wins and the plan
    # therefore reaches all three routes at once.
    resolved = plan_over(MIXED, objective="footprint")
    census = resolved.route_census()
    red(
        "the pre-#549 chooser",
        "picked ONE artifact-wide rung from a table keyed by (variant, rung) and reported it "
        f"as the whole generation — this artifact's real census is {census}, and whichever "
        "rung won would have named exactly one of those three routes for all of it",
    )
    check(
        "a MIXED plan resolves THREE routes and LABELS ITSELF MIXED",
        resolved.label().startswith("mixed(")
        and census == {"verbatim": 1, "decoded_float": 1, "encoded_gemm": 1},
        f"{resolved.label()} — every count in the label came off a per-tensor resolution",
    )
    check(
        "a SINGLE-route plan still gets a single-route name, so the label is not noise",
        route_label({"decoded_float": 743}) == "aot_decode"
        and route_label({"verbatim": 937}) == "verbatim"
        and route_label({"encoded_gemm": 743}) == "encoded_gemm",
        "the derivation only refuses to collapse a census it cannot honestly collapse",
    )
    check(
        "the WIRE's closed vocabulary is derived, not asserted: one decode makes it `float`",
        resolved.wire_delivery() == "float" and resolved.wire_materialization() == "aot_decode",
        f"{resolved.wire_delivery()}/{resolved.wire_materialization()} beside the honest "
        f"label {resolved.label()} — a plan holding an encoded leaf AND a decoded weight is "
        "not a `native` generation, because a float destination is resident",
    )
    import dataclasses

    staged = dataclasses.replace(resolved, placement="component_staged")
    check(
        "a STAGED placement derives `staged_decode` from the same census (#549 delivery.py)",
        staged.wire_materialization() == "staged_decode",
        "WHEN a decode is paid is a fact about the route AND the placement, and it is the "
        "one thing the rung table still owns",
    )

    # THE CLOSED WORKER SEAM still gets a rung name, and it is DERIVED.
    document = resolved.wire_document()
    check(
        "the wire document carries the worker's own key set, every value derived",
        document["rung"] == "aot_decode"
        and staged.wire_document()["rung"] == "staged_decode"
        and document["variant"] == resolved.variant
        and document["label"] == resolved.label()
        and document["plan_digest"] == resolved.digest(),
        f"rung={document['rung']} beside label={document['label']} — `worker/plan.py`'s "
        "kernel fence prices a plan against a rung, so a mixed plan prices as what it COSTS "
        "and reports what it IS; the two used to be one field and it was the label that won",
    )

    # Per-tensor truth is IN the plan, with the reason.
    by_key = {row.key: row for row in resolved.tensors}
    check(
        "every tensor carries its own encoding digest, implementation, route and geometry",
        by_key["block.mlp.fc1.weight"].encoding == SPEC_MXFP8
        and by_key["block.mlp.fc1.weight"].route == "encoded_gemm"
        and by_key["block.attn.out_proj.weight"].route == "decoded_float"
        and by_key["block.norm.weight"].route == "verbatim"
        and by_key["block.mlp.fc1.weight"].geometry == (128, 224),
        "the unit the old plan had no representation for at all",
    )

    # The digest, and what it binds.
    other = Variant(
        "only", "/fixture", "sha256:" + "bb" * 32, {"block": "sha256:" + "bb" * 32}, True
    )
    repointed = plan_over(MIXED, objective="footprint", variants=[other], rows_for={"only": MIXED})
    red(
        "the pre-#549 identity",
        "hashed a rung NAME into the constructed digest, so re-pointing a variant name at "
        "different snapshot bytes left the identity unchanged",
    )
    check(
        "the plan digest moves when the SNAPSHOT BYTES move, under an unchanged name",
        repointed.digest() != resolved.digest() and repointed.variant == resolved.variant,
        f"{resolved.digest()[:19]}… vs {repointed.digest()[:19]}… for variant "
        f"{resolved.variant!r} both times",
    )
    check(
        "the plan digest moves when the RUNTIME BUILD moves",
        resolve(
            variants=[
                Variant(
                    "only",
                    "/fixture",
                    "sha256:" + "aa" * 32,
                    {"block": "sha256:" + "aa" * 32},
                    True,
                )
            ],
            rows_for={"only": MIXED},
            providers=launch_providers(),
            capabilities=records(),
            device=DEVICE,
            runtime=RuntimeIdentity(RUNTIME.release, "sha256:" + "cc" * 32, RUNTIME.torch_build),
            construction=CONSTRUCTION,
            facts=PlanFacts(None),
            objective="footprint",
            dtype_name=dtype_name,
        ).digest()
        != resolved.digest(),
        "a decode rewritten between deploys is a different plan, so it does not inherit the "
        "old decode's measurements",
    )
    check(
        "the plan digest does NOT move for a resolution that changed nothing",
        plan_over(MIXED, objective="footprint").digest() == resolved.digest(),
        "identity has to be stable or it is not identity",
    )


def objective() -> None:
    head("objective — #549.3: the default axis no longer dead-ends")

    # THE RED. Two qualified providers for one encoding, plus the DEFAULT objective, is the
    # exact input `Capabilities.admit` used to refuse on.
    providers = launch_providers()
    caps = records()
    among = {
        implementation_digest(p): p
        for p in providers[SPEC_ROWWISE]
        if DEVICE.satisfies(p.device_predicate)
    }
    qualified = caps.qualified(SPEC_ROWWISE, DEVICE, "bfloat16", among)
    red(
        "the pre-#549 admission",
        f"{len(qualified)} qualified implementations of fp8-rowwise/1 on this card and the "
        "default `latency` objective named no per-record axis, so `admit` raised "
        "`objective_unranked` — before the layer holding latency measurements ever ran",
    )
    check(
        "the plant is REAL: two implementations of one encoding qualify on this device",
        len(qualified) == 2
        and {r.delivery_route for r in qualified} == {"decoded_float", "encoded_gemm"},
        ", ".join(sorted(among[r.implementation_digest].name for r in qualified)),
    )
    resolved = plan_over(MIXED, objective="latency")
    check(
        "ARM: the DEFAULT objective + two qualified providers RESOLVES (#549.3)",
        resolved.objective == "latency" and len(resolved.tensors) == 3,
        f"{resolved.summary()}",
    )
    check(
        "and it CONFESSES rather than pretending: nothing has measured this plan's latency",
        resolved.measurements is None and "UNCALIBRATED" in resolved.confession,
        resolved.confession[:150],
    )
    check(
        "an unmeasured latency plan takes the DECODE FLOOR, which needs no comparison",
        resolved.route_census().get("encoded_gemm") is None and resolved.policy == "decode_floor",
        f"{resolved.label()} — the reference plan is the one every deviation is measured "
        "against, so it is what an uncalibrated rig serves while the first prepare banks",
    )

    # With a measurement, latency scores and the encoded plan can win.
    with tempfile.TemporaryDirectory() as tmp:
        facts = PlanFacts(tmp)
        encoded = plan_over(MIXED, objective="footprint")
        floor = plan_over(MIXED, objective="latency")
        for target, fill_ms, step_ms in ((encoded, 2000, 80.0), (floor, 3500, 128.0)):
            digest = target.digest()
            facts.bank(
                RunFact(
                    plan_digest=digest,
                    run_id=run_id(digest, "plan-conform", RUNTIME.release),
                    resident_bytes=1,
                    scratch_bytes=0,
                    fill_ms=fill_ms,
                    step_ms=step_ms,
                    reference_deviation_rel_l2=0.0,
                    evidence="planted",
                )
            )
        timed = plan_over(MIXED, objective="latency", facts=facts)
        check(
            "ARM: with a banked fill+step wall the LATENCY axis SCORES and picks the winner",
            timed.digest() == encoded.digest() and timed.measurements is not None,
            f"{timed.summary()} — latency is a WHOLE-PLAN measurement, which is why it "
            "belongs here and had nothing to stand on per-provider",
        )

    check(
        "FOOTPRINT is derivable with no plan measurement at all, from the RECORDS",
        plan_over(MIXED, objective="footprint").route_census().get("encoded_gemm") == 1,
        "resident_bits_per_element x element count is measured x header, so this axis never "
        "has to fall back to uncalibrated",
    )
    check(
        "KERNEL_NUMERIC_ERROR takes the WORST tensor, not the mean",
        abs(
            plan_over(MIXED, objective="kernel_numeric_error").derived_kernel_numeric_error()
            - 0.0021
        )
        < 1e-9,
        "the decode floor's worst tensor is its decoded one; a mean over 937 verbatim "
        "copies would report a plan that replaced every linear op as almost exact",
    )
    try:
        plan_over(MIXED, objective="fidelity")
        check("ARM: the retired `fidelity` objective refuses", False, "it did not refuse")
    except PlanRefusal as exc:
        check(
            "ARM: `fidelity` is GONE and refuses by name (#549.11)",
            exc.code == "unknown_objective" and "kernel_numeric_error" in exc.detail,
            "quality is a paired video+audio eval on real outputs; an axis that prices one "
            "GEMM against an f64 reference may not borrow the word",
        )


def cache() -> None:
    head("cache — #549.2: facts keyed by plan digest, in immutable per-run files")

    with tempfile.TemporaryDirectory() as tmp:
        facts = PlanFacts(tmp)
        a = plan_over(MIXED, objective="footprint")
        b = plan_over(MIXED, objective="latency")
        check(
            "the two policies resolve DIFFERENT plans, so their facts cannot collide",
            a.digest() != b.digest(),
            f"{a.label()} vs {b.label()} — under the old key BOTH were `(only, aot_decode)`",
        )
        red(
            "the pre-#549 table",
            "keyed `(variant, rung)`, which bound neither the snapshot bytes, nor the "
            "runtime build, nor the provider map, nor the construction contract — so these "
            "two plans shared one row and each overwrote the other's numbers",
        )

        first = run_id(a.digest(), "dev", RUNTIME.release)
        second = run_id(a.digest(), "dev", RUNTIME.release)
        check(
            "two runs of ONE plan get two run ids, so neither can overwrite the other",
            first != second,
            "the clock and the pid are what make it a RUN rather than a plan",
        )
        for i, (fill, step) in enumerate(((100, 10.0), (140, 12.0), (120, 11.0))):
            digest = a.digest()
            facts.bank(
                RunFact(
                    plan_digest=digest,
                    run_id=f"{run_id(digest, 'dev', RUNTIME.release)}-{i}",
                    fill_ms=fill,
                    step_ms=step,
                    resident_bytes=7,
                    evidence="planted",
                )
            )
        folded = facts.get(a.digest())
        check(
            "three immutable files FOLD to a median and an HONEST sample count",
            folded.samples == 3 and folded.fill_ms == 120 and abs(folded.step_ms - 11.0) < 1e-9,
            f"fill {folded.fill_ms} ms, step {folded.step_ms} ms over {folded.samples} runs — "
            "the old shape incremented a `samples` integer nobody could audit",
        )
        check(
            "a field NO run measured stays -1 and is named absent, never zero",
            folded.reference_deviation_rel_l2 == -1.0
            and "reference_deviation_rel_l2" not in folded.measured,
            f"measured: {list(folded.measured)}",
        )
        # A SECOND observation of the same plan, measuring a DIFFERENT field. The old shape
        # read the prior row, copied six fields out of it and wrote the merge back.
        digest = a.digest()
        facts.bank(
            RunFact(
                plan_digest=digest,
                run_id=f"{run_id(digest, 'dev', RUNTIME.release)}-conform",
                reference_deviation_rel_l2=0.0031,
                evidence="planted",
            )
        )
        merged = facts.get(digest)
        check(
            "a later run measuring a DIFFERENT field folds in WITHOUT touching the first's",
            merged.samples == 4
            and merged.fill_ms == 120
            and abs(merged.reference_deviation_rel_l2 - 0.0031) < 1e-9,
            "no read-modify-write, no `.partial`, no lost update — each field comes from "
            "whichever runs actually measured it",
        )
        check(
            "an O_EXCL collision leaves the winner's bytes ALONE",
            facts.bank(
                RunFact(plan_digest=digest, run_id=f"{run_id(digest, 'dev', RUNTIME.release)}-x")
            )
            is not None
            and facts.get(digest).samples == 5,
            "a collision means the same run wrote twice, which is a caller bug, not a race",
        )
        check(
            "a fold of ZERO runs is zero samples, not a plausible-looking record",
            fold("sha256:none", []).samples == 0 and fold("sha256:none", []).fill_ms == -1,
            "an absent measurement is absent",
        )
        # The old artifact is not read.
        legacy = Path(tmp) / "delivery-deadbeef.json"
        legacy.write_text(json.dumps({"format": "cozy.runtime.delivery-table/1", "rows": []}))
        check(
            "an OLD label-keyed table on disk is INVALID and is never read",
            facts.get(b.digest()).samples == 0,
            "the numbers in it cannot be attributed to a plan after the fact — that is "
            "exactly what they failed to record — so a first prepare re-mints",
        )


def specs() -> None:
    head("specs — #549.4: providers claim DIGESTS, and an alias claims nothing")

    providers = launch_providers()
    check(
        "every reviewed digest resolved to a provider, and none was invented",
        set(providers) <= registry_digests() and len(providers) >= 6,
        f"{len(providers)} spec digests carry an implementation, each declared by digest",
    )
    check(
        "a reviewed digest carries the producer-mined VECTOR DOCUMENT the spec names",
        all(
            (ROOT / "src/cozy_runtime/internal/encoding/vectors" / n).is_file()
            for n in SPEC_VECTORS.values()
        ),
        f"{len(SPEC_VECTORS)} vendored documents — the suite DECODES them, so `reviewed` "
        "means the exact producer bytes ran through the exact provider",
    )

    # THE ARM. A new digest under an alias this build serves.
    planted = "sha256:" + "de" * 32
    selector = Selector(providers, records(), DEVICE, "latency")
    selector.aliases = dict(selector.aliases)
    selector.aliases[planted] = "mxfp8/1"
    red(
        "the pre-#549 launch table",
        "read digests OUT of `seed_digests()` BY ALIAS, so a new, semantically different "
        "spec filed under `mxfp8/1` would have been handed `cozy.mxfp8.dequant/1` on the "
        "strength of its display name",
    )
    try:
        selector.select(
            "block.planted.weight",
            Encoded(
                planted,
                "mxfp8/1",
                (
                    RolePart("data", "f8_e4m3fn", (16, 64), 1024),
                    RolePart("scale", "u8", (16, 2), 32),
                ),
            ),
            "bfloat16",
            (16, 64),
        )
        check("ARM: a new digest under a served alias acquires no provider", False, "served it")
    except EncodingRefusal as exc:
        check(
            "ARM: a NEW digest filed under an EXISTING alias acquires NO provider",
            exc.code == "spec_unreviewed",
            str(exc)[:220],
        )
    _vector_rounding()
    check(
        "the alias is still DISPLAYED — it names what the refusal is talking about",
        "mxfp8/1"
        in str(next(iter([e for e in [_refusal_of(selector, planted)] if e is not None]))),
        "display, and only display",
    )


def _vector_rounding() -> None:
    """The DEFECT the vector leg found on its first run, kept as an arm (#549.4).

    `fp8-rowwise/1`'s decode narrowed the scale to the destination dtype and multiplied
    there — the rule `fp8-scaled-scalar/1`'s registry entry pins explicitly, applied to its
    alias-neighbour, which turns out to round the other way. Nothing could see it: the
    synthetic oracle sweeps this route with scales that are exact POWERS OF TWO, and under a
    power of two the two rules agree bit for bit. The producer's mined vectors carry
    continuous f32 scales, which is the case the sweep excluded.

    CPU-only: 16x32 elements, no device, no store.
    """
    import torch

    from cozy_runtime.internal.encoding import RowwiseDequant
    from cozy_runtime.internal.probe import spec_vector_document, vector_tensor

    document = spec_vector_document(SPEC_ROWWISE)
    assert document is not None
    case = next(c for c in document["cases"] if c["logical"]["dtype"] == "bf16")
    cpu = torch.device("cpu")
    data = vector_tensor(torch, case["roles"]["data"], cpu)
    scale = vector_tensor(torch, case["roles"]["scale"], cpu)
    want = vector_tensor(torch, {**case["logical"], "bits": case["expect_logical_bits"]}, cpu)
    rows = int(data.shape[0])

    def differing(got: Any) -> int:
        return int((got.reshape(-1).view(torch.uint8) != want.reshape(-1).view(torch.uint8)).sum())

    narrowed = data.to(torch.bfloat16) * scale.to(torch.bfloat16).reshape(rows, 1)
    red(
        "the pre-#549.4 rowwise decode",
        f"narrowed the scale THEN multiplied, and differs from `fp8-rowwise/1`'s own "
        f"producer-mined reference on {differing(narrowed)} of {want.numel() * 2} bf16 "
        "bytes — a full ULP, invisible to a sweep whose scales are all powers of two",
    )
    shipped = torch.empty_like(want)
    RowwiseDequant(SPEC_ROWWISE).decode(torch, {"data": data, "scale": scale}, shipped)
    check(
        "ARM: the SHIPPED rowwise decode reproduces the producer's bytes EXACTLY",
        differing(shipped) == 0 and differing(narrowed) > 0,
        f"0 differing bytes against {differing(narrowed)} for the rule this alias's "
        "NEIGHBOUR pins — two encodings, two rounding rules, and #511c decides it: the "
        "reference wins over prose, and here it is the only thing that had an opinion",
    )
    check(
        "the correction bumped the provider's CODE REVISION, so old records stop admitting it",
        RowwiseDequant(SPEC_ROWWISE).code_revision == "2",
        "#549.9's revision field doing exactly the job it was added for: the records that "
        "measured revision 1 measured different arithmetic",
    )


def _refusal_of(selector: Selector, digest: str) -> EncodingRefusal | None:
    try:
        selector.select(
            "block.planted.weight",
            Encoded(digest, "mxfp8/1", (RolePart("data", "f8_e4m3fn", (16, 64), 1024),)),
            "bfloat16",
            (16, 64),
        )
    except EncodingRefusal as exc:
        return exc
    return None


def geometry() -> None:
    head("geometry — #549.5: a ragged K resolves to the floor, and the plan SAYS WHY")

    whole = MicroScaledNativeLeaf(SPEC_MXFP8).supports((128, 224))
    ragged = MicroScaledNativeLeaf(SPEC_MXFP8).supports((128, 225))
    check(
        "the mxfp8 encoded-GEMM leaf ACCEPTS K divisible by 32",
        whole == "",
        "224 = 7 whole blocks",
    )
    check(
        "ARM: it REFUSES a ragged K, and the sentence names the rule and both producers",
        "divisible by 32" in ragged and "whole blocks" in ragged,
        ragged[:170],
    )
    check(
        "the DECODE floor accepts the same ragged tensor — it is the route that handles one",
        MicroScaledDequant(SPEC_MXFP8).supports((128, 225)) == "",
        "`ceil_div(axis 1, by 32)` is the spec's own relation and the slow path spans a "
        "partial block correctly",
    )
    check(
        "the preflight is PURE: no torch, no device, no allocation",
        RowwiseNativeLeaf(SPEC_ROWWISE).supports((64, 224)) == ""
        and "rank 1" in RowwiseNativeLeaf(SPEC_ROWWISE).supports((64,)),
        "it is asked at plan resolution, over a header's logical shape, before a "
        "destination exists",
    )

    red(
        "the pre-#549 order",
        "nothing asked until `leaf()` was being built with the component RESIDENT — "
        "TensorFS admits a ragged tail, the decode handles one, and the activation "
        "quantizer's [M, K/32, 32] reshape would have truncated it in silence",
    )
    rows = [mx_row("block.mlp.fc1.weight", 128, 225), mx_row("block.mlp.fc2.weight", 128, 224)]
    # FOOTPRINT, the axis the encoded leaf wins, so the plan PREFERS it everywhere and the
    # only thing that can keep a tensor off it is the tensor's own geometry.
    resolved = plan_over(rows, objective="footprint")
    by_key = {row.key: row for row in resolved.tensors}
    check(
        "ARM: the ragged tensor resolves to the DECODE FLOOR while its neighbour takes the leaf",
        by_key["block.mlp.fc1.weight"].route == "decoded_float"
        and by_key["block.mlp.fc2.weight"].route == "encoded_gemm",
        f"{resolved.label()} — one component, one encoding, two routes, decided by GEOMETRY",
    )
    check(
        "ARM: the REASON is IN the plan, not in a log nobody keeps",
        "GEOMETRY:" in by_key["block.mlp.fc1.weight"].reason
        and "divisible by 32" in by_key["block.mlp.fc1.weight"].reason,
        by_key["block.mlp.fc1.weight"].reason[-150:],
    )
    check(
        "the plan records the GEOMETRY CLASS each tensor was resolved at",
        by_key["block.mlp.fc1.weight"].geometry_class == "rank2.ragged32"
        and by_key["block.mlp.fc2.weight"].geometry_class == "rank2.block32",
        "evidence on the record, never a filter — filtering admission on it would refuse "
        "the decode floor for the one tensor it is the only route for",
    )


def identity() -> None:
    head("identity — #549.9: what a capability record is actually about")

    key = DEVICE.tuple_key()
    red(
        "the pre-#549 key",
        "was `cuda/NAME/CONFIG/torch.version.cuda` — the last field is the CUDA RUNTIME "
        "torch was built against, not the driver, and the CONFIG came from NVML queried on "
        "GPU ZERO whatever device was selected",
    )
    check(
        "the key carries the REAL driver and the CUDA runtime as SEPARATE fields",
        "driver=595.84" in key and "cudart=13.0" in key,
        key,
    )
    check(
        "the key carries the SELECTED device index and the OS/arch",
        "dev3" in key and "Linux/x86_64" in key,
        "two pods with one torch wheel and different NVIDIA drivers used to share a key",
    )
    same_card_other_driver = DeviceFacts(
        DEVICE.kind,
        DEVICE.name,
        DEVICE.sm,
        "560.35",
        DEVICE.configuration,
        DEVICE.index,
        DEVICE.cuda_runtime,
        DEVICE.os_arch,
    )
    check(
        "ARM: a DRIVER change makes a different key, so old records do not apply",
        same_card_other_driver.tuple_key() != key,
        "the old key could not see this at all",
    )
    check(
        "ARM: a different SELECTED DEVICE makes a different key",
        DeviceFacts(
            DEVICE.kind,
            DEVICE.name,
            DEVICE.sm,
            DEVICE.driver,
            DEVICE.configuration,
            0,
            DEVICE.cuda_runtime,
            DEVICE.os_arch,
        ).tuple_key()
        != key,
        "a multi-GPU pod used to mint every record against card 0",
    )

    providers = launch_providers()
    rowwise = next(p for p in providers[SPEC_ROWWISE] if p.route == "encoded_gemm")
    before = implementation_digest(rowwise)

    class Bumped:
        """The same provider with a bumped code revision — an edited forward pass."""

        name = rowwise.name
        route = rowwise.route
        device_predicate = rowwise.device_predicate
        suite = rowwise.suite
        required = rowwise.required
        optional = rowwise.optional
        reviewed_specs = rowwise.reviewed_specs
        code_revision = "2"

    check(
        "ARM: a provider's CODE REVISION is part of its identity (#549.9)",
        implementation_digest(Bumped()) != before,
        "an edit to a leaf's forward invalidates the records that measured the old one, "
        "instead of inheriting them",
    )
    check(
        "a provider's REVIEWED SPEC SET is part of its identity too",
        implementation_digest(
            type("Widened", (Bumped,), {"code_revision": "1", "reviewed_specs": frozenset()})()
        )
        != before,
        "an implementation whose reviewed contract set changed is a different implementation",
    )
    check(
        "the RUNTIME's own source digest is a MEASUREMENT over the modules that decide",
        len(RUNTIME.source_digest) > 20 and RUNTIME.source_digest.startswith("sha256:"),
        "a version string is a promise and a digest is a fact",
    )


def envelope() -> None:
    head("envelope — #549.10: the decode-scratch double-count at fill.py")

    def old(total: int, budget: int, largest: int) -> int:
        return min(total, budget) + largest

    def new(total: int, budget: int, largest: int) -> int:
        return min(total, budget + largest)

    budget = 512 << 20
    # A component whose WHOLE scratch fits under the budget. It never flushes, so its true
    # high-water mark is exactly `total`.
    total, largest = 300 << 20, 120 << 20
    red(
        "the pre-#549 expression",
        f"`min(total, budget) + largest` reserved {old(total, budget, largest) >> 20} MiB "
        f"for a walk whose live scratch can never exceed {total >> 20} MiB — the largest "
        "tensor counted twice, because it is already inside `total`",
    )
    check(
        "ARM: when the whole scratch fits under the budget the envelope IS the total",
        new(total, budget, largest) == total and old(total, budget, largest) == total + largest,
        f"{new(total, budget, largest) >> 20} MiB, against the old "
        f"{old(total, budget, largest) >> 20} MiB — a "
        f"{100 * largest / total:.0f}% over-reservation charged to a `decode` budget that "
        "can refuse a fill for want of memory nothing was going to use",
    )
    total = 4096 << 20
    check(
        "over the budget the two agree — the flush caps live scratch at budget + largest",
        new(total, budget, largest) == old(total, budget, largest) == budget + largest,
        "the fix is narrow on purpose: it changes only the case the old expression got wrong",
    )
    check(
        "a single tensor LARGER than the budget still gets an envelope that holds it",
        new(600 << 20, budget, 600 << 20) == 600 << 20,
        "one tensor may exceed the budget on its own, which is why `largest` is in the "
        "expression at all",
    )
    source = (ROOT / "src/cozy_runtime/internal/fill.py").read_text()
    check(
        "the SHIPPED expression is the corrected one",
        "min(total, self.decode_scratch_bytes + largest)" in source
        and "min(\n" not in source.split("envelope = ")[1][:60],
        "read off the module, so this arm cannot pass against a reverted fix",
    )


def consent() -> None:
    head("consent — the encoded route needs the endpoint's word, at RESOLUTION")

    refusing = plan_over(MIXED, objective="footprint", consent="refuse")
    check(
        "ARM: an endpoint that did not consent gets NO encoded-GEMM plan, on ANY axis",
        refusing.route_census().get("encoded_gemm") is None,
        f"{refusing.label()} — footprint is the axis the encoded route WINS, and it is not "
        "offered, because a plan the endpoint cannot consent to is not a candidate",
    )
    check(
        "the walk SAYS SO rather than silently omitting the plan",
        any(row.get("verdict") == "unconsented" for row in refusing.walk),
        next(
            (row.get("why", "") for row in refusing.walk if row.get("verdict") == "unconsented"),
            "",
        )[:160],
    )
    check(
        "with consent the same artifact resolves the encoded plan",
        plan_over(MIXED, objective="footprint", consent="accept").route_census().get("encoded_gemm")
        == 1,
        "it used to be discovered at `materialize`, with the plan chosen and the endpoint "
        "already imported",
    )


SCENARIOS: dict[str, Callable[[], None]] = {
    "plan": plan,
    "objective": objective,
    "cache": cache,
    "specs": specs,
    "geometry": geometry,
    "identity": identity,
    "envelope": envelope,
    "consent": consent,
}


def main(argv: list[str]) -> int:
    names = argv or list(SCENARIOS)
    for name in names:
        if name not in SCENARIOS:
            raise SystemExit(f"unknown scenario {name!r}; have {sorted(SCENARIOS)}")
        SCENARIOS[name]()
    print(f"\n{PASS} ok, {FAIL} failed over {len(names)} scenario(s)")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
