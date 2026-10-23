"""Kernel selection is a rule of the card's tensor cores (xs-065, owner 2026-10-22).

A real TensorFS checkpoint holds each lane's weights as the real producers encode them; its
header resolves through the real resolver. Each lane takes its native leaf on every card whose
`arch` row has the leaf's formats — named rows, a compute capability no row names (it inherits
one), and a faked future row — and decodes only where the tensor cores are missing or its kernels
write no such destination. Nothing is measured, no table is read, and the serving path cannot
import the conformance suite.
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import tensorfs

from cozy_runtime.author.kernels import INT8_CONVROT, ROWWISE_FP8
from cozy_runtime.derive import svdq
from cozy_runtime.derive.quantization import _ENCODINGS
from cozy_runtime.internal import arch
from cozy_runtime.internal.encoding import (
    DeviceFacts,
    LeafProvider,
    RuntimeIdentity,
    launch_providers,
)
from cozy_runtime.internal.encoding.leaves import MXFP8, NVFP4, SVDQ, W4A8
from cozy_runtime.internal.fill import Checkpoint, PlanRow, dtype_name
from cozy_runtime.internal.probe import vector_cases
from cozy_runtime.internal.resolution import (
    ConstructionFacts,
    PlanRefusal,
    ResolvedModelPlan,
    Variant,
    resolve,
)
from store_fixtures import home

LEAVES = {
    "int8.convrot": INT8_CONVROT, "fp8.rowwise": ROWWISE_FP8, "nvfp4": NVFP4, "mxfp8": MXFP8,
    "w4a8.convrot": W4A8, "svdq.nvfp4": SVDQ,
}  # fmt: skip
ALL = frozenset(LEAVES)
INT8 = frozenset({"int8.convrot", "w4a8.convrot"})
EIGHT_AND_FOUR = INT8 | {"nvfp4"}
#: The lanes on the FP4 tensor cores' Triton kernels: tcgen05 or sm120's block-scaled MMA.
FP4 = frozenset({"svdq.nvfp4"})
NOT_MX = ALL - {"mxfp8"} - FP4

#: sm -> the lanes that run on their native leaf. fp8 needs sm89+, block-scaled MX sm100+, int8
#: (and w4a8, whose codes multiply as int8) is absent from sm103, nothing runs natively below the
#: MMA families; 88 and 110 are named by no row (110 holds tcgen05, as its cc could).
NATIVE = {
    75: frozenset(),
    80: EIGHT_AND_FOUR,
    86: EIGHT_AND_FOUR,
    87: EIGHT_AND_FOUR,
    88: EIGHT_AND_FOUR,
    89: NOT_MX,
    90: NOT_MX,
    100: ALL,
    103: ALL - INT8,
    110: ALL,
    120: ALL,
    121: ALL,
}
#: The lanes whose kernels write no float32 (`_scaled_mm`'s fp8 scalings, svdq's half epilogue):
#: they decode there.
HALF_ONLY = frozenset({"fp8.rowwise", "mxfp8", "svdq.nvfp4"})
#: What a decoded lane decodes with: the decode cores (Triton on CUDA) where they serve it.
DECODER = {
    "int8.convrot": "cozy.int8-convrot.dequant/1",
    "fp8.rowwise": "cozy.fp8-rowwise.dequant/1",
    "nvfp4": "cozy.nvfp4.cores/1",
    "mxfp8": "cozy.mxfp8.cores/1",
    "w4a8.convrot": "tensorfs.w4a8.convrot.decode/1",
    "svdq.nvfp4": "cozy.svdq.nvfp4.cores/1",
}


class Rows(NamedTuple):
    rows: list[PlanRow]
    dtype: str


@pytest.fixture(params=["BF16", "F16", "F32"])
def lanes(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rows:
    """One weight per lane into the fixture's dtype: the producer `cozy model quantize` runs
    for int8, fp8, nvfp4, w4a8 and svdq-nvfp4 (fitted to random inputs), and the spec's
    producer-mined mxfp8 case."""
    torch = pytest.importorskip("torch")
    writer = tensorfs.Writer(cache_dir=home(tmp_path, monkeypatch))
    model = writer.component("model", cls=("", "", ""), config={})
    rng = np.random.default_rng(65)
    values = rng.standard_normal((256, 512)).astype(np.float32)
    seen = torch.from_numpy(rng.standard_normal((64, 512)).astype(np.float32))
    for scheme, encoder in _ENCODINGS.items():
        rows, cols = values.shape
        if encoder.calibrated:
            stats = svdq.Stats(seen.abs().amax(0), seen.to(torch.bfloat16))
            fitted = svdq.fit(torch, scheme, values, stats, "cpu")
            payload, raws = fitted.payload, fitted.companions
        else:
            encoded = encoder.encode(scheme, values, float(np.abs(values).max()))
            payload, raws = encoded.payload, tuple(c.raw for c in encoded.companions)
        parts = {"weight": (encoder.weight_dtype, [rows, cols // encoder.pack], payload)}
        for (role, dtype, shape), raw in zip(encoder.companions(rows, cols), raws, strict=True):
            parts[role] = (dtype, list(shape), raw)
        model.put_encoded(
            f"{scheme}.weight", dtype=request.param, shape=[rows, cols], scheme=scheme,
            parts=parts, params=encoder.params,
        )  # fmt: skip
    mx = next(c for c in vector_cases() if c.encoding.scheme == "mxfp8" and c.key == "4x64-bf16")
    model.put_encoded(
        "mxfp8.weight", dtype=request.param, shape=mx.shape, scheme="mxfp8",
        parts={p.role: (p.dtype, p.shape, base64.b64decode(p.bits)) for p in mx.parts},
        params=mx.encoding.params,
    )  # fmt: skip
    return Rows(Checkpoint(writer.commit()).rows("model"), request.param)


def card(sm: int) -> DeviceFacts:
    return DeviceFacts("cuda", f"a card of cc {sm}", sm, "580.95.05/api13000", "300W/N/A")


def plan(rows: list[PlanRow], device: DeviceFacts, consent: str = "accept") -> ResolvedModelPlan:
    return resolve(
        variant=Variant("lanes", "store", rows[0].key),
        rows=rows,
        providers=launch_providers(),
        device=device,
        runtime=RuntimeIdentity("t/1", "2.14.0"),
        construction=ConstructionFacts("t/1", "t:Model", ("model",)),
        encoded_leaves=consent,
        dtype_name=dtype_name,
    )


def native(served: ResolvedModelPlan) -> frozenset[str]:
    """The lanes served on their native leaf; each other one serves its decoder."""
    lanes = {row.encoding: row for row in served.tensors}
    for scheme, row in lanes.items():
        if row.route == "encoded_gemm":
            assert row.implementation == LEAVES[scheme]
        else:
            assert (row.route, row.implementation) == ("decoded_float", DECODER[scheme]), row
    return frozenset(s for s, row in lanes.items() if row.route == "encoded_gemm")


def expected(sm: int, dtype: str) -> frozenset[str]:
    return NATIVE[sm] - HALF_ONLY if dtype == "F32" else NATIVE[sm]


@pytest.mark.parametrize("sm", sorted(NATIVE))
def test_each_lane_runs_natively_wherever_its_tensor_cores_exist(lanes: Rows, sm: int) -> None:
    served = plan(lanes.rows, card(sm))
    assert native(served) == expected(sm, lanes.dtype)
    # A lane this card does not serve natively says so once; a package that declines leaves
    # decodes everything and says nothing.
    assert {group.lane for group in served.refused_leaves()} == ALL - expected(sm, lanes.dtype)
    declined = plan(lanes.rows, card(sm), consent="refuse")
    assert native(declined) == frozenset() and declined.refused_leaves() == ()


def test_a_row_no_one_has_measured_runs_every_lane_its_formats_allow(
    lanes: Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A future card with every format serves every lane natively; one without int8 tensor
    cores decodes int8 and w4a8 alone. No table, verdict or benchmark names either."""
    monkeypatch.setitem(arch.ROWS, 130, arch.Row("Future", arch.TCGEN05, True, True, True))
    monkeypatch.setitem(arch.ROWS, 131, arch.Row("No int8", arch.TCGEN05, False, True, True))
    every = ALL - (HALF_ONLY if lanes.dtype == "F32" else frozenset())
    assert native(plan(lanes.rows, card(130))) == every
    assert native(plan(lanes.rows, card(131))) == every - INT8
    refused = {g.lane: g.reason for g in plan(lanes.rows, card(131)).refused_leaves()}
    assert all("has no int8 tensor cores" in refused[lane] for lane in INT8), refused


def test_an_unnamed_card_admits_by_its_capability_not_by_what_a_process_learned(
    lanes: Rows, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sm110's lanes are the same on every Load, even after its 8-bit op settled on mma.sync."""
    monkeypatch.setattr(arch, "_LEARNED", {110: arch.MMA_SYNC})
    assert native(plan(lanes.rows, card(110))) == expected(110, lanes.dtype)


def test_off_cuda_every_lane_decodes_and_mps_refuses_float8(lanes: Rows) -> None:
    assert native(plan(lanes.rows, DeviceFacts("cpu", "cpu", 0, "", "default"))) == frozenset()
    mps = DeviceFacts("mps", "Apple M3", 0, "api", "unified")
    int8 = [r for r in lanes.rows if r.encoded.scheme == "int8.convrot"]
    assert plan(int8, mps).route_census() == {"decoded_float": 1}
    with pytest.raises(PlanRefusal, match="MPS has no float8"):
        plan(lanes.rows, mps)


def test_every_leaf_declares_its_formats_and_destinations() -> None:
    """By construction: a new lane is admitted by declaring what its kernels multiply in and
    write, so there is no list for it to be missing from."""
    leaves = [p for ps in launch_providers().values() for p in ps if isinstance(p, LeafProvider)]
    assert leaves
    known = arch.ROWS[100].formats() | arch.ROWS[120].formats()
    for leaf in leaves:
        assert isinstance(leaf.formats, frozenset) and leaf.formats <= known, leaf.name
        assert leaf.out_dtypes and leaf.out_dtypes <= {"bfloat16", "float16", "float32"}


def test_serving_never_imports_the_conformance_suite() -> None:
    """The executor and the fill plane resolve and load without the suite: nothing measures
    a card to serve it."""
    code = (
        "import sys; import cozy_runtime.internal.executor, cozy_runtime.internal.weights; "
        "sys.exit('cozy_runtime.internal.probe' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
