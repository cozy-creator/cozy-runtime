"""Kernel selection is a rule of the card's tensor cores (xs-065, owner 2026-10-22).

A real TensorFS checkpoint holds each lane's weights as the real producers encode them; its
header resolves through the real resolver. Each lane takes its native leaf on every card whose
`arch` row has the leaf's formats — named rows, a compute capability no row names (it inherits
one), and a faked future row — and decodes only where the tensor cores are missing. Nothing is
measured, no table is read, and the serving path cannot import the conformance suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorfs

from cozy_runtime.author.kernels import INT8_CONVROT, ROWWISE_FP8
from cozy_runtime.derive.quantization import _ENCODINGS
from cozy_runtime.internal import arch
from cozy_runtime.internal.encoding import (
    DeviceFacts,
    LeafProvider,
    RuntimeIdentity,
    launch_providers,
)
from cozy_runtime.internal.encoding.leaves import NVFP4
from cozy_runtime.internal.fill import Checkpoint, PlanRow, dtype_name
from cozy_runtime.internal.resolution import (
    ConstructionFacts,
    PlanRefusal,
    ResolvedModelPlan,
    Variant,
    resolve,
)
from store_fixtures import home

LEAVES = {"int8.convrot": INT8_CONVROT, "fp8.rowwise": ROWWISE_FP8, "nvfp4": NVFP4}
ALL = frozenset(LEAVES)
EIGHT_AND_FOUR = frozenset({"int8.convrot", "nvfp4"})

#: sm -> the lanes that run on their native leaf. fp8 needs sm89+, int8 is absent from sm103,
#: nothing runs natively below the MMA families; 88 and 110 are named by no row.
NATIVE = {
    75: frozenset(),
    80: EIGHT_AND_FOUR,
    86: EIGHT_AND_FOUR,
    87: EIGHT_AND_FOUR,
    88: EIGHT_AND_FOUR,
    89: ALL,
    90: ALL,
    100: ALL,
    103: frozenset({"fp8.rowwise", "nvfp4"}),
    110: ALL,
    120: ALL,
    121: ALL,
}


@pytest.fixture
def rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[PlanRow]:
    """One weight per lane, encoded by the producer `cozy model quantize` runs."""
    writer = tensorfs.Writer(cache_dir=home(tmp_path, monkeypatch))
    model = writer.component("model", cls=("", "", ""), config={})
    values = np.random.default_rng(65).standard_normal((256, 512)).astype(np.float32)
    for scheme, encoder in _ENCODINGS.items():
        rows_, cols = values.shape
        encoded = encoder.encode(scheme, values, float(np.abs(values).max()))
        parts = {"weight": (encoder.weight_dtype, [rows_, cols], encoded.payload)}
        for (role, dtype, shape), companion in zip(
            encoder.companions(rows_, cols), encoded.companions, strict=True
        ):
            parts[role] = (dtype, list(shape), companion.raw)
        model.put_encoded(
            f"{scheme}.weight", dtype="BF16", shape=[rows_, cols], scheme=scheme,
            parts=parts, params=encoder.params,
        )  # fmt: skip
    return Checkpoint(writer.commit()).rows("model")


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
    """The lanes served on their native leaf; each decoded one serves the decode floor."""
    lanes = {row.encoding: row for row in served.tensors}
    for scheme, row in lanes.items():
        if row.route == "encoded_gemm":
            assert row.implementation == LEAVES[scheme]
        else:
            assert row.route == "decoded_float", row
    return frozenset(s for s, row in lanes.items() if row.route == "encoded_gemm")


@pytest.mark.parametrize("sm", sorted(NATIVE))
def test_each_lane_runs_natively_wherever_its_tensor_cores_exist(
    rows: list[PlanRow], sm: int
) -> None:
    served = plan(rows, card(sm))
    assert native(served) == NATIVE[sm]
    # A card missing a lane's tensor cores says so once per lane; a package that declines
    # leaves decodes everything and says nothing.
    assert {group.lane for group in served.refused_leaves()} <= ALL - NATIVE[sm]
    declined = plan(rows, card(sm), consent="refuse")
    assert native(declined) == frozenset() and declined.refused_leaves() == ()


def test_a_row_no_one_has_measured_runs_every_lane_its_formats_allow(
    rows: list[PlanRow], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A future card with all three formats serves every lane natively; one without int8
    tensor cores decodes int8 alone. No table, verdict or benchmark names either."""
    future = arch.Row("Future", arch.TCGEN05, int8=True, fp8=True, fp4=True)
    no_int8 = arch.Row("Future without int8", arch.TCGEN05, fp8=True, fp4=True)
    monkeypatch.setitem(arch.ROWS, 130, future)
    monkeypatch.setitem(arch.ROWS, 131, no_int8)
    assert native(plan(rows, card(130))) == ALL
    assert native(plan(rows, card(131))) == frozenset({"fp8.rowwise", "nvfp4"})
    refused = plan(rows, card(131)).refused_leaves()
    assert [(g.lane, g.leaf) for g in refused] == [("int8.convrot", INT8_CONVROT)]
    assert "has no int8 tensor cores" in refused[0].reason


def test_off_cuda_every_lane_decodes_and_mps_refuses_float8(rows: list[PlanRow]) -> None:
    assert native(plan(rows, DeviceFacts("cpu", "cpu", 0, "", "default"))) == frozenset()
    mps = DeviceFacts("mps", "Apple M3", 0, "api", "unified")
    int8 = [r for r in rows if r.encoded.scheme == "int8.convrot"]
    assert plan(int8, mps).route_census() == {"decoded_float": 1}
    with pytest.raises(PlanRefusal, match="MPS has no float8"):
        plan(rows, mps)


def test_every_leaf_declares_its_formats() -> None:
    """By construction: a new lane is admitted by declaring what its kernels multiply in, so
    there is no list for it to be missing from."""
    leaves = [p for ps in launch_providers().values() for p in ps if isinstance(p, LeafProvider)]
    assert leaves
    known = arch.ROWS[100].formats() | arch.ROWS[120].formats()
    for leaf in leaves:
        assert isinstance(leaf.formats, frozenset) and leaf.formats <= known, leaf.name


def test_serving_never_imports_the_conformance_suite() -> None:
    """The executor and the fill plane resolve and load without the suite: nothing measures
    a card to serve it."""
    code = (
        "import sys; import cozy_runtime.internal.executor, cozy_runtime.internal.weights; "
        "sys.exit('cozy_runtime.internal.probe' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
