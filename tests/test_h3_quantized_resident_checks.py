"""Actual H3 finite checks over standard encoded weights and their lifetime."""

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.author import Context, OutputError, Telemetry
from cozy_runtime.author._loader import census
from cozy_runtime.author._services import Attempt
from cozy_runtime.internal.base_weights import BaseWeights
from cozy_runtime.internal.encoding.torchao_rowwise import bind_rowwise_weight

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("diffusers")


def test_h3_checks_physical_fp8_parts_caches_and_rescans_after_refill(tmp_path: Path) -> None:
    from cozy_runtime.models.minimax_h3.official import NumericalChecks, ResidentWeights

    base = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    root = torch.nn.Sequential(base)
    owned = BaseWeights("dit", root, census(SimpleNamespace(components={"dit": root})))
    events: list[Any] = []
    telemetry = Telemetry(
        Attempt("resident", tmp_path, sink=events.append), Context("resident", float("inf"))
    )
    resident = ResidentWeights()
    checks = NumericalChecks(telemetry, resident)
    codes = torch.ones(32, 16, dtype=torch.float8_e4m3fn)
    scales = torch.ones(32, 1, dtype=torch.float32)
    bind_rowwise_weight(base, codes, scales)
    checks.component("dit", root)
    assert events[-1].fields["tensors"] == 2
    assert events[-1].fields["elements"] == codes.numel() + scales.numel()
    assert resident.verified(base.weight)
    checks.component("dit", root)
    assert events[-1].fields["status"] == "verified_fill"
    checks.tensors("direct", [("weight", base.weight)], required=True)
    assert events[-1].fields["elements"] == codes.numel() + scales.numel()

    # Mutation of either physical part invalidates its cached verdict.
    base.weight.scale[0] = float("inf")
    assert not resident.verified(base.weight)
    with pytest.raises(OutputError, match="non-finite"):
        checks.component("dit", root)
    base.weight.scale[0] = 1
    checks.component("dit", root)
    base.weight.qdata[0, 0] = float("nan")
    with pytest.raises(OutputError, match="non-finite"):
        checks.component("dit", root)
    base.weight.qdata[0, 0] = 1
    checks.component("dit", root)

    held_codes = weakref.ref(codes)
    owned.evict()
    checks.component("dit", root)
    assert events[-1].fields["status"] == "no_resident_float_values"
    assert events[-1].fields["parked_tensors"] == 2
    del codes, scales
    gc.collect()
    assert held_codes() is None
    assert not resident._verdicts  # Scanning retains no storage ownership.

    owned.reserve("cpu")
    codes = torch.ones(32, 16, dtype=torch.float8_e4m3fn)
    scales = torch.full((32, 1), 2.0, dtype=torch.float32)
    bind_rowwise_weight(base, codes, scales)
    assert not resident.verified(base.weight)
    checks.component("dit", root)
    assert events[-1].fields["status"] == "finite_resident"
    assert events[-1].fields["absmax"] == 2.0
    assert resident.verified(base.weight)
