"""The leaf speedup the probe banks is READ where the route is recorded (cr-135).

`probe._leaf_bench` has measured every qualified native leaf against the bf16 GEMM it
replaces since #517b, and banked `speedup_x` under COZY_HOME. Nothing read it: not the
capability record, not the plan, not the delivery line, not the boot tail. On the box's own
sm89 card the recorded probe below banks the rowwise fp8 leaf at **0.61x and 0.77x** — slower
than bf16 — and the same card's delivery lines served `routes encoded_gemm x453` with no word
of it. The fp8 path degrading silently (a ComfyUI user lost 2.17x for weeks to exactly that)
would have left every record byte-identical.

The input is the RECORDED REAL PROBE `tests/observations/probe-cuda-sm89.json`, replayed
through the real minting, the real resolution and the real worker renderer. It was written by
runtime 0.0.36 under that build's spec digests, so `rekeyed()` moves its observations onto the
digests THIS build reviews — the card's measurements are untouched; a record admits nothing
this build does not carry. The CUDA arm at the bottom measures the card it runs on.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.internal import probe
from cozy_runtime.internal.encoding import (
    SPEC_ROWWISE,
    SPEC_ROWWISE_KEEPDIM,
    Encoded,
    RolePart,
    implementation_digest,
    launch_providers,
)
from cozy_runtime.internal.fill import dtype_name
from cozy_runtime.internal.planfacts import PlanFacts
from cozy_runtime.internal.probe import NativeEvidence, Observation, QualificationResult
from cozy_runtime.internal.resolution import ConstructionFacts, ResolvedModelPlan, Variant, resolve
from cozy_runtime.internal.worker.session import _Delivery, _delivery_observation

OBSERVATION = Path(__file__).resolve().parent / "observations" / "probe-cuda-sm89.json"

#: The two `fp8-rowwise/1` spec digests runtime 0.0.36 reviewed, and what this build calls them.
RECORDED_SPECS = {
    "sha256:8c86b26daec5bcd287401d9873b70aa3594855f97ac4d01cf75cc2bafd5d50b5": SPEC_ROWWISE,
    "sha256:91ddc2a73ba2778b344afc5e78b4addee9174a015cfff02f17cfbb629ad5ef78": (
        SPEC_ROWWISE_KEEPDIM
    ),
}

LEAF = "cozy.fp8-rowwise.native-leaf/1"


def _narrated(wire: dict[str, object]) -> str:
    return _delivery_observation(msgspec.convert(wire, _Delivery))


def recorded() -> QualificationResult:
    return msgspec.convert(json.loads(OBSERVATION.read_bytes()), QualificationResult)


def rekeyed() -> QualificationResult:
    """The recorded rowwise observations on this build's digests, measurements verbatim."""
    result = recorded()
    by_name = {provider.name: provider for provider in launch_providers()[SPEC_ROWWISE]}
    return msgspec.structs.replace(
        result,
        observations=[
            msgspec.structs.replace(
                o,
                encoding_spec_id=RECORDED_SPECS[o.encoding_spec_id],
                implementation_digest=implementation_digest(by_name[o.implementation]),
            )
            for o in result.observations
            if o.encoding_spec_id in RECORDED_SPECS and o.implementation in by_name
        ],
    )


def benched(result: QualificationResult, speedup: float | None) -> QualificationResult:
    """`result` with every leaf bench's speedup moved to `speedup`, or the bench removed."""

    def move(o: Observation) -> Observation:
        evidence = o.evidence
        if not isinstance(evidence, NativeEvidence) or evidence.micro_bench is None:
            return o
        bench = (
            None
            if speedup is None
            else msgspec.structs.replace(evidence.micro_bench, speedup_x=speedup)
        )
        return msgspec.structs.replace(
            o, evidence=msgspec.structs.replace(evidence, micro_bench=bench)
        )

    return msgspec.structs.replace(result, observations=[move(o) for o in result.observations])


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    """`fill.PlanRow`-shaped, which is the input `resolve()` documents."""

    key: str
    name: str
    component: str
    dtype: str
    shape: tuple[int, ...]
    encoded: Encoded


def _resolve(result: QualificationResult, consent: str) -> ResolvedModelPlan:
    """One rowwise-encoded bf16 linear at the probe's own shape, through the real resolver."""
    encoded = Encoded(
        encoding=SPEC_ROWWISE,
        alias="fp8-rowwise/1",
        parts=(
            RolePart("data", "f8_e4m3fn", (2048, 2048), 2048 * 2048),
            RolePart("scale", "f32", (2048,), 2048 * 4),
        ),
    )
    row = Row("dit.to_q.weight", "to_q.weight", "dit", "bf16", (2048, 2048), encoded)
    return resolve(
        variants=[
            Variant(
                name="fp8",
                store="store",
                snapshot="sha256:aa",
                snapshots={"dit": "sha256:aa"},
                reference=True,
            )
        ],
        rows_for={"fp8": [row]},
        providers=launch_providers(),
        capabilities=probe._table(result),
        device=result.device,
        runtime=result.runtime,
        construction=ConstructionFacts(release="test", model_class="pkg:Dit", components=("dit",)),
        facts=PlanFacts(),
        objective="latency",
        encoded_leaves=consent,
        dtype_name=dtype_name,
    )


# ------------------------------------------------------------------------- the record


def test_the_probe_mints_the_leaf_speedup_onto_the_record_and_nothing_else() -> None:
    """Every passed encoded-GEMM observation carries its bench; every decode route carries
    NOT MEASURED, spelled -1.0 exactly as the two axes beside it spell it."""
    table = probe._table(rekeyed())
    leaves = [r for r in table.records if r.delivery_route == "encoded_gemm"]
    assert {r.leaf_speedup_x for r in leaves} == {0.6122, 0.7683}, leaves
    assert leaves and all(r.document()["leaf_speedup_x"] == r.leaf_speedup_x for r in leaves)
    floors = [r for r in table.records if r.delivery_route != "encoded_gemm"]
    assert floors and all(r.leaf_speedup_x == -1.0 for r in floors)


def test_the_qualification_document_names_the_worst_leaf_speedup_per_implementation() -> None:
    """One implementation measured under two spec digests reports the WORSE of the two: the
    number the boot tail prints is the one an operator would want to be alarmed by."""
    result = recorded()
    document = probe.Qualification(probe._table(result), result, "key", measured=False).document()
    assert document["leaf_speedup_x"] == {f"{LEAF}@bfloat16": 0.6122}
    # A document with no passed leaf carries an empty map, not a missing key.
    floors_only = msgspec.structs.replace(
        result,
        observations=[o for o in result.observations if o.delivery_route != "encoded_gemm"],
    )
    assert probe.leaf_speedups(floors_only) == {}


# --------------------------------------------------------------------------- the plan


def test_resolution_carries_the_leaf_speedup_onto_the_plan_the_walk_and_the_line() -> None:
    """The reference variant stores rowwise bytes, the package accepts leaves, nothing is
    banked: resolution serves the encoded lane uncalibrated, exactly as before — and now the
    plan, its walk, its confession and the worker's delivery line all say the leaf measured
    0.61x its float GEMM on this card."""
    plan = _resolve(rekeyed(), "accept")
    assert plan.policy == "encoded_gemm" and plan.route_census() == {"encoded_gemm": 1}
    assert plan.derived_leaf_speedup_x() == 0.6122
    assert plan.tensors[0].leaf_speedup_x == 0.6122
    assert "leaf_speedup_x" not in plan.tensors[0].identity()
    assert plan.tensors[0].document()["leaf_speedup_x"] == 0.6122
    assert plan.document()["derived_leaf_speedup_x"] == 0.6122

    chosen = next(step for step in plan.walk if step.verdict == "chosen_uncalibrated")
    assert chosen.leaf_speedup_x == 0.6122
    floor = next(step for step in plan.walk if step.policy == "decode_floor")
    assert floor.leaf_speedup_x is None
    assert "leaf_speedup_x" not in msgspec.to_builtins(floor)

    assert "measured these leaves at 0.61x their float GEMM" in plan.confession
    assert "SLOWER than decoding" in plan.confession
    assert "served for the bytes it keeps resident, not for speed" in plan.confession

    wire = plan.wire_document()
    assert wire["leaf_speedup_x"] == 0.6122
    line = _narrated(wire)
    assert (
        "routes encoded_gemm x1; kernels encoded_gemm x1; leaf 0.61x its float GEMM; "
        "uncalibrated) chosen on the latency axis"
    ) in line, line
    assert "SLOWER than decoding" in line


def test_a_leaf_that_wins_is_reported_without_the_loss_verdict() -> None:
    plan = _resolve(benched(rekeyed(), 2.03), "accept")
    assert plan.derived_leaf_speedup_x() == 2.03
    assert "measured these leaves at 2.03x their float GEMM" in plan.confession
    assert "SLOWER" not in plan.confession
    assert "leaf 2.03x its float GEMM" in _narrated(plan.wire_document())


def test_the_decode_floor_carries_no_leaf_speedup_and_the_line_says_nothing() -> None:
    """A package that refuses leaves resolves to the floor; there is no leaf to report and
    no number is invented for the line."""
    plan = _resolve(rekeyed(), "refuse")
    assert plan.route_census() == {"decoded_float": 1}
    assert plan.derived_leaf_speedup_x() == -1.0
    wire = plan.wire_document()
    assert wire["leaf_speedup_x"] == -1.0
    assert "leaf " not in _narrated(wire)
    assert "float GEMM" not in plan.confession


def test_a_leaf_with_no_bench_reads_as_not_measured() -> None:
    """The same discipline as the two axes beside it: no bench is -1.0, which is UNREPORTED
    rather than a plausible number."""
    result = benched(rekeyed(), None)
    plan = _resolve(result, "accept")
    assert plan.route_census() == {"encoded_gemm": 1}
    assert plan.derived_leaf_speedup_x() == -1.0
    assert "float GEMM" not in plan.confession
    assert "leaf " not in _narrated(plan.wire_document())
    assert probe.leaf_speedups(result) == {}


# ------------------------------------------------------------------------ the CUDA arm


def _cuda_skip() -> str:
    if importlib.util.find_spec("torch") is None:
        return "torch is not installed"
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 89:
        return "the rowwise leaf's floor is sm89"
    return ""


def test_the_leaf_speedup_this_card_measures_is_read_on_its_own_record() -> None:
    """MEASURED HERE, on the card this runs on, through the real suite, the real resolution
    and the real renderer: the number the boot record prints is the number the probe took."""
    why = _cuda_skip()
    if why:
        pytest.skip(why)
    child = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "testdata/leaf_speedup_proof.py")],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    out = json.loads(child.stdout.splitlines()[-1])
    banked = out["qualification"]["leaf_speedup_x"]
    assert out["qualification"]["source"] == "measured"
    leaf = banked.get(f"{LEAF}@bfloat16")
    if leaf is None:
        # The leaf's numerics failed on this card: it minted no record, the floor serves,
        # and no speed is invented for a route that is not there.
        assert out["wire"]["routes"] == {"decoded_float": 1}, out
        assert out["wire"]["leaf_speedup_x"] == -1.0
        return
    assert 0.0 < leaf < 100.0, banked
    assert out["wire"]["routes"] == {"encoded_gemm": 1}, out
    served = out["wire"]["leaf_speedup_x"]
    assert served >= leaf and served in out["leaf_records"], out
    assert f"leaf {served:.2f}x its float GEMM" in out["line"], out["line"]
    assert f"measured these leaves at {served:.2f}x their float GEMM" in out["confession"]
    assert ("SLOWER than decoding" in out["confession"]) == (served < 1.0)
