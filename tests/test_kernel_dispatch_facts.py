"""Shipped offline kernel facts, keyed by the kernel's own identity, consumed without a probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import int8_linear, prebuilt, verify  # noqa: E402

KERNEL = verify.Kernel("prebuilt-int8-mmasync", "sha256:build", "sha256:abi")
RULES = (
    verify.Rule(0, 4, "square", verify.GEMV, 128, 256),
    verify.Rule(1024, 2048, "square", 4, 128, 256),
    verify.Rule(2048, 4096, "up", verify.CUBLASLT, 128, 256),
)


def forbidden() -> Any:
    raise AssertionError("consumer invoked the development oracle")


def cpu_probe() -> verify.Verdict:
    x = torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32
    weight = torch.arange(48, dtype=torch.float32).reshape(6, 8) / 48
    error = verify.error(torch, x @ weight.T, x.double() @ weight.double().T)
    return verify.within({"float32 CPU matmul": error}, 1e-6)


def test_a_published_verdict_and_its_rules_bind_to_the_kernel_alone(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    assert not verify.admitted(KERNEL, 89, forbidden, tables=missing).ok
    assert verify.shipped(KERNEL, 89, tables=missing) is None and not missing.exists()
    with verify.measuring() as records:
        measured = verify.admitted(KERNEL, 89, cpu_probe)
    assert measured.ok
    path = verify.publish(tmp_path, 89, records, {KERNEL: RULES}, {"card": "cpu"}, {"raw": [1]})
    assert path.name == "kernels-sm89.json"
    before = path.read_bytes()
    assert verify.admitted(KERNEL, 89, forbidden, tables=tmp_path) == measured
    assert verify.rules(KERNEL, 89, tables=tmp_path) == RULES
    # Any other build, interface or card is another kernel: no fact transfers to it.
    for other in (
        verify.Kernel(KERNEL.name, "sha256:rebuilt", KERNEL.abi),
        verify.Kernel(KERNEL.name, KERNEL.build, "sha256:new-signature"),
        verify.Kernel("prebuilt-int8-wgmma", KERNEL.build, KERNEL.abi),
    ):
        assert verify.shipped(other, 89, tables=tmp_path) is None
        assert verify.rules(other, 89, tables=tmp_path) == ()
    assert verify.shipped(KERNEL, 90, tables=tmp_path) is None
    assert path.read_bytes() == before


def test_a_refused_or_malformed_table_ships_no_rules(tmp_path: Path) -> None:
    with verify.measuring() as records:
        verify.measured(KERNEL, 86, lambda: verify.within({"tile 3": 0.7}, 1e-2))
    path = verify.publish(tmp_path, 86, records, {KERNEL: RULES}, {}, None)
    assert verify.shipped(KERNEL, 86, tables=tmp_path) == verify.Verdict(
        False, "rel L2 7.00e-01 at tile 3 (budget 1.00e-02, 1 cases)"
    )
    assert verify.rules(KERNEL, 86, tables=tmp_path) == ()
    document = json.loads(path.read_bytes())
    document["schema"] = 1
    path.write_text(json.dumps(document))
    assert verify.shipped(KERNEL, 86, tables=tmp_path) is None


def test_admission_takes_its_cards_rules_and_a_refusal_never_drops_the_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = prebuilt.gemm_kernel("int8", "mma.sync", 89)
    other = verify.Rule(0, 4, "square", 3, 64, 128)  # measured on a card of another SM bucket
    with verify.measuring() as records:
        verify.measured(identity, 89, cpu_probe)
    verify.publish(tmp_path, 89, records, {identity: (*RULES, other)}, {}, None)
    monkeypatch.setattr(verify, "TABLES", tmp_path)
    for sms, rules in ((142, RULES), (128, (other,)), (24, ())):
        ops = prebuilt.Prebuilt(torch, 0, "mma.sync", False, sms)
        assert prebuilt.admission(torch, ops, prebuilt.INT8, "mma.sync", 89).ok
        assert ops.rules["int8"] == rules
    with verify.measuring() as records:
        verify.measured(identity, 89, lambda: verify.within({"tile 3": 0.7}, 1e-2))
    verify.publish(tmp_path, 89, records, {identity: RULES}, {}, None)
    ops = prebuilt.Prebuilt(torch, 0, "mma.sync", False, 142)
    try:
        verdict = prebuilt.admission(torch, ops, prebuilt.INT8, "mma.sync", 89)
        assert (
            verdict.ok
            and "refused verdict" in verdict.detail
            and "family default" in verdict.detail
        )
        assert "int8" not in ops.rules
    finally:
        prebuilt.DEFAULTED.pop((89, "int8"), None)


def test_a_raising_probe_is_a_refusal_and_consumers_never_probe() -> None:
    def raises() -> verify.Verdict:
        raise RuntimeError("8-bit linear: the GEMM: cudaErrorInvalidKernelImage")

    with verify.measuring() as records:
        refused = verify.measured(KERNEL, 90, raises)
    assert refused == verify.Verdict(
        False, "RuntimeError: 8-bit linear: the GEMM: cudaErrorInvalidKernelImage"
    )
    assert records == {(KERNEL, 90): refused}
    assert not verify.within({}, 1e-2).ok
    with pytest.raises(RuntimeError, match="a consumer never probes"):
        verify.measured(KERNEL, 90, forbidden)


def test_rules_choose_by_shape_class_and_bucket_else_the_family_default() -> None:
    ops = prebuilt.Prebuilt(torch, 0, "mma.sync", False, 142, rules={"int8": RULES})
    assert ops.choice(prebuilt.INT8, 1, 4096, 4096) == verify.GEMV
    assert ops.choice(prebuilt.INT8, 2048, 1280, 1280) == 4
    assert ops.choice(prebuilt.INT8, 1025, 1280, 1280) == 4
    assert ops.choice(prebuilt.INT8, 1024, 1280, 1280) is None  # no bucket: family default
    assert ops.choice(prebuilt.INT8, 4096, 10240, 1280) == verify.CUBLASLT
    assert ops.choice(prebuilt.INT8, 4096, 1280, 5120) is None  # a down projection
    assert ops.choice(prebuilt.FP8, 2048, 1280, 1280) is None  # another format's rules
    assert not torch.cuda.is_initialized()


def test_unqualified_optional_kitchen_never_loads_during_consumer_dispatch() -> None:
    before = dict(int8_linear._KITCHEN)
    assert not verify.in_oracle()
    assert int8_linear.kitchen(torch, 0) is None
    assert before == int8_linear._KITCHEN and not torch.cuda.is_initialized()
