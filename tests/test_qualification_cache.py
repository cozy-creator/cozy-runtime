"""Offline qualification files and deterministic reference admission, using real CPU decoders."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal import probe, qualification  # noqa: E402
from cozy_runtime.internal.encoding import (  # noqa: E402
    launch_providers,
    measure_device,
    measure_runtime,
)
from cozy_runtime.internal.encoding.selection import reference_decoder  # noqa: E402


@pytest.fixture(scope="module")
def offline() -> tuple[Any, Any, Any, Any]:
    if torch.cuda.is_available():
        pytest.skip("CPU decoder proof: run with CUDA_VISIBLE_DEVICES= to leave GPUs untouched")
    device, runtime = measure_device(torch, 0), measure_runtime(torch, "offline-test")
    providers = {k: v for k, v in launch_providers().items() if k in ("plain", "fp8.rowwise")}
    # This is the developer's real offline oracle. Its measurements are never forged or rekeyed.
    _, table = probe.qualify(
        torch, providers, device, ("float32",), release="offline-test", runtime=runtime
    )
    return providers, device, runtime, table


def load(offline: Any, directory: Path, **options: Any) -> qualification.Qualification:
    providers, device, runtime, _ = offline
    visited: list[str] = []
    forbidden = {"qualify", "_measured", "_evidence", "_leaf_bench", "_micro_bench"}

    def calls(frame: Any, event: str, arg: Any) -> None:
        if (
            event == "call"
            and frame.f_code.co_filename == probe.__file__
            and frame.f_code.co_name in forbidden
        ):
            visited.append(frame.f_code.co_name)

    before = sys.getprofile()
    sys.setprofile(calls)
    try:
        result = qualification.qualified(
            torch,
            providers,
            device,
            options.get("dtypes", ("float32",)),
            runtime=options.get("runtime", runtime),
            release="offline-test",
            tables=directory,
        )
    finally:
        sys.setprofile(before)
    assert not visited, f"consumer ran the offline oracle: {visited}"
    return result


def ship(directory: Path, table: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{qualification.table_name(table.device, table.runtime)}.json"
    qualification.write_table(path, table)
    return path


def test_offline_oracle_output_is_read_without_consumer_measurement(
    offline: Any, tmp_path: Path
) -> None:
    table = offline[3]
    path = ship(tmp_path, table)
    original = path.read_bytes()
    result = load(offline, tmp_path)
    assert result.source == "shipped" and result.result.suite_ms == table.suite_ms
    assert result.capabilities.records
    assert path.read_bytes() == original and list(tmp_path.iterdir()) == [path]
    assert any(isinstance(o.evidence, probe.DecodeEvidence) for o in result.result.observations)


def test_missing_table_admits_declared_decoders_without_inventing_measurements(
    offline: Any, tmp_path: Path
) -> None:
    directory = tmp_path / "absent"
    result = load(offline, directory)
    assert result.source == "supported" and result.result.suite_ms == 0
    assert not directory.exists(), "inference must not create a measurement cache or lock"
    decoded = [
        o for o in result.result.observations if isinstance(o.evidence, probe.ReferenceEvidence)
    ]
    assert decoded and all(o.delivery_route == "decoded_float" for o in decoded)
    assert all(o.geometry_class == "declared-format" for o in decoded)
    assert all(o.evidence.route_axes.deviation_rel < 0 for o in decoded)
    assert all(
        r.leaf_speedup_x < 0 and r.kernel_numeric_error < 0 for r in result.capabilities.records
    )
    assert all(r.delivery_route != "encoded_gemm" for r in result.capabilities.records)
    assert any(
        o.delivery_route == "verbatim" and o.evidence.passed for o in result.result.observations
    )


def test_stale_and_partial_tables_keep_only_their_actual_offline_facts(
    offline: Any, tmp_path: Path
) -> None:
    table = offline[3]
    ship(tmp_path, table)
    other_torch = dataclasses.replace(offline[2], torch_build="2.99.0+cu130")
    assert load(offline, tmp_path, runtime=other_torch).source == "supported"
    # Another Runtime build with the same implementations keeps every fact.
    rebuilt = dataclasses.replace(offline[2], commit="another-commit")
    assert load(offline, tmp_path, runtime=rebuilt).source == "shipped"
    # A valid partial table retains its facts; another destination uses the declared decoder.
    result = load(offline, tmp_path, dtypes=("float32", "float16"))
    assert result.source == "shipped"
    assert any(
        o.output_dtype == "float32" and isinstance(o.evidence, probe.DecodeEvidence)
        for o in result.result.observations
    )
    assert any(
        o.output_dtype == "float16" and isinstance(o.evidence, probe.ReferenceEvidence)
        for o in result.result.observations
    )


def test_an_offline_decoder_failure_is_not_overridden_by_reference_admission(
    offline: Any, tmp_path: Path
) -> None:
    table = offline[3]
    # A real format refusal stored by an offline run must remain a refusal at consumption.
    observed = next(o for o in table.observations if o.delivery_route == "decoded_float")
    provider = next(
        p for ps in offline[0].values() for p in ps if p.name == observed.implementation
    )
    why = provider.supports((3,), {})
    assert why
    refused = msgspec.structs.replace(observed, evidence=probe.UnmeasuredEvidence(note=why))
    ship(tmp_path, msgspec.structs.replace(table, observations=[refused]))
    result = load(offline, tmp_path)
    row = next(o for o in result.result.observations if o.implementation == provider.name)
    assert not row.evidence.passed and row.evidence.note == why


def test_declared_fallback_decoders_pass_the_real_format_vectors(offline: Any) -> None:
    device = offline[1]
    supported = [
        p
        for group in launch_providers().values()
        for p in group
        if reference_decoder(p, device, "float32")
    ]
    assert supported
    for provider in supported:
        report = probe._spec_vectors(torch, provider, torch.device("cpu"))
        assert report.passed and not report.skipped, (provider.name, report)
        assert not reference_decoder(provider, device, "int64")
    unknown = dataclasses.replace(supported[0], encoding="unknown.future-format")
    assert not reference_decoder(unknown, device, "float32")
