"""A private package installs its own exact dependency closure with ordinary uv."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import subprocess
import zipfile
from pathlib import Path

import pytest

from conftest import image_python
from cozy_runtime.internal import package_installation
from cozy_runtime.internal.package_environment import EnvironmentRefusal


def _wheel(
    root: Path,
    name: str,
    requirements: tuple[str, ...],
    code: bytes,
    *,
    resources: dict[str, bytes] | None = None,
    requires_python: str = "",
    version: str = "1.0",
) -> Path:
    info = f"{name}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.3\nName: {name}\nVersion: {version}\n"
    if requires_python:
        metadata += f"Requires-Python: {requires_python}\n"
    metadata += "".join(f"Requires-Dist: {value}\n" for value in requirements)
    members = {
        f"{name}.py": code,
        f"{info}/METADATA": (metadata + "\n").encode(),
        f"{info}/WHEEL": (b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
    }
    members.update(resources or {})
    rows = [
        (
            path,
            "sha256="
            + base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode(),
            str(len(body)),
        )
        for path, body in members.items()
    ]
    record = io.StringIO()
    csv.writer(record).writerows([*rows, (f"{info}/RECORD", "", "")])
    members[f"{info}/RECORD"] = record.getvalue().encode()
    path = root / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, body in members.items():
            archive.writestr(member, body)
    return path


def test_conflicting_private_closures_ignore_base_and_ambient_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = image_python()
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "ambient_sentinel.py").write_text("VALUE=999")
    monkeypatch.setenv("PYTHONPATH", str(poison))
    monkeypatch.setenv("UV_CONSTRAINT", str(poison / "missing-constraints"))
    root = tmp_path / "installs"
    held = []
    for version in ("1.23.0", "1.16.1"):
        wheel_root = tmp_path / version
        wheel_root.mkdir()
        dependency = _wheel(
            wheel_root, "huggingface_hub", (), f"VALUE={version!r}\n".encode(), version=version
        )
        helper = _wheel(
            wheel_root, "diffusers", (f"huggingface-hub=={version}",), b"import huggingface_hub\n"
        )
        project = _wheel(wheel_root, "model_project", ("diffusers==1.0",), b"import diffusers\n")
        installed = package_installation.install(
            root,
            package="local/model-project",
            release="1.0",
            python=python,
            wheels=(project, helper, dependency),
        )
        result = subprocess.run(
            [
                str(installed.python),
                "-I",
                "-c",
                "import huggingface_hub,importlib.util; print(huggingface_hub.VALUE); "
                "assert importlib.util.find_spec('ambient_sentinel') is None; "
                "assert importlib.util.find_spec('grpc') is None",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip() == version
        assert "include-system-site-packages = false" in (
            (installed.generation / "pyvenv.cfg").read_text()
        )
        held.append(installed)
    assert held[0].generation != held[1].generation
    for installed in held:
        replay = package_installation.install(
            root,
            package="local/model-project",
            release="1.0",
            python=python,
            installation_id=installed.installation_id,
        )
        assert replay.reused and replay.generation == installed.generation
    with pytest.raises(EnvironmentRefusal, match="package_installation_conflict"):
        package_installation.install(
            root,
            package="local/other",
            release="1.0",
            python=python,
            installation_id=held[0].installation_id,
        )
