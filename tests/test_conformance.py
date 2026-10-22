"""The conformance suite checks a provider against the conformance cases of its schemes."""

from __future__ import annotations

import pytest

from cozy_runtime.internal.encoding import RowwiseDequant, launch_providers, measure_device
from cozy_runtime.internal.probe import _spec_vectors, conform, vector_cases


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


def test_the_suite_passes_this_cards_decoders_and_observes_nothing_below_a_floor() -> None:
    """`conform`, as `cli.conformance` runs it, here over the int8 lane on the CPU: its decoder
    passes its oracle and vectors; its CUDA leaf is below this device's floor."""
    torch = pytest.importorskip("torch")
    providers = {"int8.convrot": launch_providers()["int8.convrot"]}
    report = conform(torch, providers, measure_device(torch, 0), ["bfloat16"], release="t/1")
    kinds = {o.implementation: type(o.evidence).__name__ for o in report.observations}
    assert kinds == {
        "cozy.int8-convrot.dequant/1": "DecodeEvidence",
        "cozy.int8-convrot256.native-leaf/1": "UnmeasuredEvidence",
    }
    assert report.passed and report.device.kind == "cpu"
