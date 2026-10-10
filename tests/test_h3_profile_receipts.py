"""Portable capture and phase receipts must match the qualified diagnostic cohort."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def helper(name: str) -> ModuleType:
    path = Path(__file__).parents[1] / "research/h3-request-profile" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"h3_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wheels(directory: Path, h3_version: str) -> dict[str, Path]:
    result = {}
    for name, version in (
        ("minimax-h3", h3_version),
        ("qwen-image-2", "0.3.0"),
        ("diffusers", "0.41.0.dev0"),
    ):
        path = directory / f"{name}-{version}.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                f"{name.replace('-', '_')}-{version}.dist-info/METADATA",
                f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n",
            )
        result[name] = path
    return result


def test_capture_requires_current_h3_and_retains_explicit_wheel_sources(tmp_path: Path) -> None:
    destination = tmp_path / "current"
    sources = wheels(tmp_path, "1.26.3")
    helper("prepare_capture").prepare(destination, sources)
    project = (destination / "pyproject.toml").read_text()
    provenance = json.loads((destination / "captured-wheel-provenance.json").read_text())
    assert '"minimax-h3==1.26.3"' in project
    assert provenance["minimax-h3"]["version"] == "1.26.3"
    for name, path in sources.items():
        assert name + " = {path=" + json.dumps(str(path.resolve())) + "}" in project
        assert len(provenance[name]["sha256"]) == 64


def test_capture_refuses_old_h3_before_creating_snapshot(tmp_path: Path) -> None:
    destination = tmp_path / "old"
    with pytest.raises(ValueError, match="benchmark package cohort"):
        helper("prepare_capture").prepare(destination, wheels(tmp_path, "1.26.2"))
    assert not destination.exists()


def test_unprofiled_receipt_keeps_handler_timing_without_inventing_trace_steps() -> None:
    summary = {
        "profile_enabled": False,
        "trace_collection": "disabled; unprofiled full request",
        "render_call": {"wall_ms": 1234, "plane_delta": {"h2d_bytes": 42}},
    }
    result = helper("phase_receipt").receipt({"number": 5089}, summary)
    assert result["profile_enabled"] is False
    assert result["bounded_kernel_trace_steps"] == []
    assert result["full_request_trace"] is False
    assert result["render_call"] is summary["render_call"]
    assert result["trace_collection"] == summary["trace_collection"]


@pytest.mark.parametrize("flag", [{}, {"profile_enabled": True}])
def test_missing_profiled_steps_still_refuses_instead_of_becoming_unprofiled(
    flag: dict[str, object],
) -> None:
    with pytest.raises(KeyError, match="profiled_step_indices"):
        helper("phase_receipt").receipt({}, {"render_call": {}, **flag})


@pytest.mark.parametrize(
    "fields",
    [
        {"profile_enabled": 0},
        {"profile_enabled": False, "profiled_step_indices": [0]},
        {"profile_enabled": False, "full_request_trace": True},
        {"profiled_step_indices": []},
        {"profiled_step_indices": [True]},
    ],
)
def test_contradictory_or_malformed_trace_receipts_refuse(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        helper("phase_receipt").receipt({}, {"render_call": {}, **fields})
