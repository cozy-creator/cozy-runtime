"""Device qualification: the recorded real probe is evidence, and the decode suite qualifies a
provider against the conformance cases of its schemes.

`tests/observations/probe-cuda-sm89.json` is the exact result `probe.qualify` returned on an
NVIDIA RTX 4070 Laptop (sm89, driver 595.84, torch 2.13.0+cu130): 27 observations, 19 passing.
"""

from __future__ import annotations

import json
from pathlib import Path

import msgspec
import pytest

from cozy_runtime.internal.encoding import RowwiseDequant, launch_providers
from cozy_runtime.internal.probe import QualificationResult, _spec_vectors, vector_cases

OBSERVATION = Path(__file__).resolve().parent / "observations" / "probe-cuda-sm89.json"


def test_the_recorded_observation_is_a_real_measured_probe() -> None:
    result = msgspec.convert(json.loads(OBSERVATION.read_bytes()), QualificationResult)
    assert (result.device_predicate, result.device.sm, result.device.kind) == (
        "cuda.sm89",
        89,
        "cuda",
    )
    assert result.runtime.source_digest.startswith("sha256:")
    assert result.runtime.torch_build.startswith("2.")
    assert len(result.observations) == 27
    assert sum(1 for o in result.observations if o.evidence.passed) == 19


def test_the_vendored_cases_cover_every_decoding_scheme() -> None:
    decoding = {
        s for s, ps in launch_providers().items() if any(p.route == "decoded_float" for p in ps)
    }
    assert decoding == {case.encoding.scheme for case in vector_cases() if not case.refuse}


def test_the_conformance_cases_run_against_the_shipped_decoder() -> None:
    torch = pytest.importorskip("torch")
    report = _spec_vectors(
        torch,
        RowwiseDequant(),
        torch.device("cpu"),
    )
    assert report.passed and not report.skipped and report.cases, report
