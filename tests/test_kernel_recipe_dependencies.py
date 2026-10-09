"""Pinned wheel selection and real compile-manager dependency scheduling; no GPU work."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest

from cozy_runtime.internal import (
    attention,
    attention_upstream,
    kernel_cache,
    kernel_compile,
    kernel_sources,
)
from cozy_runtime.internal.encoding import DeviceFacts


def test_reviewed_kitchen_wheel_is_exact_and_reuses_its_existing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [kernel_cache.Store(tmp_path / "u0")])
    sources = kernel_sources._pinned_sources()
    wheel = sources[kernel_sources.KITCHEN_WHEEL]
    if not kernel_sources._compatible_kitchen_wheel(wheel):
        pytest.skip("The reviewed native wheel is for compatible Linux CPython ABI tags")
    job = kernel_sources.job("comfy-kitchen", 120)
    assert job.kind == "python" and job.spec["build_deps"] == {}
    assert json.loads(job.key.inputs)["sources"] == {
        "comfy-kitchen": "03cca0d10c314aaa13bff8f9422d03974221a344162b23d4f10d27eb2a35d0e8"
    }
    if sys.version_info[:2] == (3, 12):
        assert job.key.digest == "2c34122edcea8f38c75aed0575618830535fded8e0cafd06f8b45a8abe46cc56"
    source = job.spec["sources"]["comfy-kitchen"]
    assert source["version"] == "0.2.37" and source["origin"].startswith(
        "https://files.pythonhosted.org/"
    )
    python_job = kernel_sources.job(kernel_sources.PYTHON, 120)
    assert not {"nanobind", kernel_sources.KITCHEN_WHEEL} & set(python_job.spec["sources"])


@pytest.mark.parametrize(
    "filename",
    [
        "comfy_kitchen-0.2.37-py3-none-any.whl",
        "comfy_kitchen-0.2.37-cp312-abi3-win_amd64.whl",
        "comfy_kitchen-0.2.37-cp999-abi3-manylinux_2_28_x86_64.whl",
        "other_package-0.2.37-cp312-abi3-manylinux_2_28_x86_64.whl",
    ],
)
def test_unqualified_wheel_tags_do_not_select_the_native_artifact(filename: str) -> None:
    source = kernel_sources.Source(kernel_sources.KITCHEN_WHEEL, "0.2.37", filename, "0" * 64)
    assert not kernel_sources._compatible_kitchen_wheel(source)


def test_source_fallback_keys_its_required_build_tool_and_refuses_its_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "sources"
    root.mkdir()
    rows = {
        "comfy-kitchen": {"version": "0.2.37", "file": "kitchen.tar.gz", "sha256": "1" * 64},
        "nanobind": {"version": "3.1.0", "file": "nanobind.whl", "sha256": "2" * 64},
    }
    index = root / "index.json"
    index.write_text(json.dumps({"sources": rows}))
    monkeypatch.setattr(kernel_sources, "torch_build", lambda: ("2.14.0", "13.0"))
    monkeypatch.setattr(kernel_sources, "compiler", lambda: "declared-compiler")
    kernel_sources.registry.cache_clear()
    first = kernel_sources.job("comfy-kitchen", 120, root)
    assert first.kind == "setup" and "nanobind" in first.spec["build_deps"]
    assert json.loads(first.key.inputs)["build_deps"] == {"nanobind": "2" * 64}
    rows["nanobind"]["sha256"] = "3" * 64
    index.write_text(json.dumps({"sources": rows}))
    kernel_sources.registry.cache_clear()
    assert kernel_sources.job("comfy-kitchen", 120, root).key != first.key
    del rows["nanobind"]
    index.write_text(json.dumps({"sources": rows}))
    kernel_sources.registry.cache_clear()
    with pytest.raises(kernel_sources.Absent, match="nanobind"):
        kernel_sources.job("comfy-kitchen", 120, root)


def _dependency(tmp_path: Path, name: str) -> tuple[attention.Candidate, kernel_compile.Job]:
    module = "dependency_" + name + "_" + tmp_path.name.replace("-", "_")
    wheel = tmp_path / f"{module}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{module}/__init__.py", "VALUE = 42\n")
        archive.writestr(f"{module}-1.0.dist-info/METADATA", f"Name: {module}\nVersion: 1.0\n")
    source = kernel_sources.Source(
        name, "1.0", str(wheel), hashlib.sha256(wheel.read_bytes()).hexdigest()
    )
    job = kernel_compile.Job(
        "python",
        kernel_cache.Key.of(name, {"source": source.sha256}),
        {"sources": {name: asdict(source)}},
    )
    candidate = attention.Candidate(
        name,
        "_cozy_" + name,
        attention_upstream.SDPA,
        0,
        frozenset(),
        0,
        artifact=name,
        import_name=module,
    )
    return candidate, job


@pytest.mark.parametrize("wait", [False, True])
def test_independent_dependencies_are_all_requested_before_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wait: bool
) -> None:
    first, first_job = _dependency(tmp_path, "primary")
    second, second_job = _dependency(tmp_path, "companion")
    jobs = {first.name: first_job, second.name: second_job}
    store = kernel_cache.Store(tmp_path / "u0")
    monkeypatch.setattr(kernel_cache, "_MACHINE_STORE", [store])
    monkeypatch.setattr(attention, "_job", lambda candidate, device: jobs[candidate.name])
    monkeypatch.setattr(attention, "_STATES", {})
    monkeypatch.setattr(sys, "path", list(sys.path))
    device = DeviceFacts("cpu", "CPU", 0, "", "")
    if not wait:
        # Actual held build locks keep both entries pending without spawning a GPU or compiler.
        locks = [store.claim(job.key) for job in jobs.values()]
        assert all(lock is not None for lock in locks)
        try:
            with pytest.raises(attention.AttentionRefusal, match="compiling"):
                attention._compile_dependencies((first, second), device, wait=False)
            assert set(attention._STATES) == {first.name, second.name}
        finally:
            for lock in locks:
                assert lock is not None
                os.close(lock)
    else:
        # Real isolated builder processes unpack and publish both independent wheel artifacts.
        ready = attention._compile_dependencies((first, second), device, wait=True)
        assert set(ready) == {first.name, second.name}
        assert all(store.entry(job.key) is not None for job in jobs.values())
        assert all(kernel_compile.status(store, job).state == "ready" for job in jobs.values())
