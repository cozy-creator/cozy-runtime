"""An ordinary job-only package must install without a fake serving entrypoint."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import package_interface
from cozy_runtime.internal.discovery import discover
from cozy_runtime.internal.local import development_placement_documents
from cozy_runtime.protocol import documents


def test_job_only_project_produces_a_development_placement(tmp_path: Path) -> None:
    (tmp_path / "package.toml").write_text('[application]\nobject = "only_job:app"\n')
    (tmp_path / "only_job.py").write_text(
        "import msgspec\n"
        "from cozy_runtime.author import App\n"
        "app = App()\n"
        "class Request(msgspec.Struct):\n    value: int\n"
        "class Result(msgspec.Struct):\n    value: int\n"
        "@app.job\n"
        "def run(payload: Request) -> Result:\n    return Result(payload.value + 1)\n"
    )
    found = discover(tmp_path)
    interface = package_interface.build(found)
    assert not interface["entrypoints"]
    assert len(interface["jobs"]) == 1
    result = development_placement_documents(
        interface,
        package="local/only-job",
        release="0.0.0",
        source_digest="sha256:" + "a" * 64,
        project=tmp_path,
    )
    assert not result.placement.entrypoints
    assert not result.placement.models
    assert result.placement.development.package == "local/only-job"
    assert result.package_interface == package_interface.canonical_bytes(interface)
    assert documents.body(result.placement)["entrypoints"] == []


def test_empty_interface_still_refuses() -> None:
    with pytest.raises(RuntimeError, match="at least one visible"):
        development_placement_documents(
            {"entrypoints": [], "jobs": []},
            package="local/empty",
            release="0.0.0",
            source_digest="sha256:" + "a" * 64,
        )


@pytest.mark.parametrize("kind", ["job", "entrypoint", "modeled"])
def test_metadata_placement_never_imports_source(
    tmp_path: Path, kind: str
) -> None:
    module = f"never_import_placement_{kind}"
    marker = tmp_path / "imported"
    (tmp_path / "package.toml").write_text(f'[application]\nobject = "{module}:app"\n')
    (tmp_path / f"{module}.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "import unavailable_package_dependency\n"
        "import msgspec\nfrom cozy_runtime.author import App, Model, uses_components\n"
        "app = App()\n"
        "class ExampleModel(Model[object]):\n"
        "    def load(self, loader):\n"
        "        raise RuntimeError('constructor ran during metadata')\n"
        "    @uses_components('example')\n"
        "    def predict(self):\n        return None\n"
        "class Request(msgspec.Struct):\n    value: int\n"
        "class Result(msgspec.Struct):\n    value: int\n"
        f"@app.{'entrypoint' if kind == 'modeled' else kind}\n"
        f"def run(payload: Request{', model: ExampleModel' if kind == 'modeled' else ''})"
        " -> Result:\n    return Result(payload.value)\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cozy_runtime.cli.main",
            "--json",
            "--dir",
            str(tmp_path),
            "development-placement",
            "--package",
            "local/no-import",
            "--release",
            "0.0.0",
            "--source-digest",
            "sha256:" + "a" * 64,
            "--tensorfs-root",
            str(tmp_path / "store"),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "COZY_HOME": str(tmp_path / "home")},
    )
    if kind == "modeled":
        assert result.returncode != 0
        assert "worker_preparation_required" in result.stdout + result.stderr
        assert not marker.exists()
        return
    assert result.returncode == 0, result.stderr
    body = json.loads(result.stdout)
    interface = json.loads(base64.b64decode(body["package_interface"]["canonical_bytes_base64"]))
    assert [row["name"] for row in interface["jobs" if kind == "job" else "entrypoints"]] == ["run"]
    assert not marker.exists()
