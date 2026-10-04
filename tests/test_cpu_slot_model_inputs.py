"""A script declares its accelerator the same way to the static and the imported reader, and a
CPU script that reaches for CUDA is told which declaration it is missing."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pytest

from cozy_runtime.author import Invocation, attempt, describe, script_app
from cozy_runtime.internal import package_interface, static_interface
from cozy_runtime.internal.discovery import discover


def _script(tmp_path: Path, name: str, metadata: str, body: str) -> Path:
    project = tmp_path / name
    project.mkdir()
    (project / "pyproject.toml").write_text(f'[project]\nname="{name}"\nversion="0.1.0"\n')
    (project / "package.toml").write_text(f'[application]\nobject="{name}_entry:app"\n')
    (project / f"{name}_entry.py").write_text(
        f'from cozy_runtime.author import script_app\napp = script_app("{name}")\n'
    )
    (project / f"{name}.py").write_text(metadata + body)
    return project


@pytest.mark.parametrize(
    ("metadata", "declared"),
    [("", False), ("# /// script\n# [tool.cozy]\n# accelerator = true\n# ///\n", True)],
)
def test_a_script_declares_its_accelerator_in_both_readers(
    tmp_path: Path, metadata: str, declared: bool
) -> None:
    name = "accel_script_" + str(declared).lower()
    project = _script(tmp_path, name, metadata, "def main():\n    pass\n")
    sys.modules.pop(name, None)
    static = static_interface.build(project)
    imported = package_interface.build(discover(project))
    assert package_interface.canonical_bytes(static) == package_interface.canonical_bytes(imported)
    assert static["jobs"][0]["accelerator"] is declared


def test_a_cpu_script_that_uses_cuda_names_the_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # torch's own words when a process with no visible device asks for CUDA.
    (tmp_path / "cuda_script.py").write_text(
        "def main():\n    raise RuntimeError('No CUDA GPUs are available')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    app = script_app("cuda_script")
    describe(app)
    _, outcome, _ = attempt(
        app.get("main"), {}, Invocation("cuda", tmp_path / "out", time.monotonic() + 5)
    )
    assert outcome.terminal == "failed" and outcome.code == "job_needs_accelerator", outcome
    assert "accelerator = true" in outcome.message and "[tool.cozy]" in outcome.message
