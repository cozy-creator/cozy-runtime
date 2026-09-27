from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import _image_wheel_cache as seed
from cozy_runtime.internal import package_installation
from test_publish_derivation import _SimpleIndex
from test_unpublished_exact_dependencies import _wheel


def test_private_install_links_the_image_seed_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    root = tmp_path / "image"
    monkeypatch.setattr(seed, "ROOT", root)
    dependency = _wheel(tmp_path, "seed_dependency", (), b"VALUE=91\n")
    digest = hashlib.sha256(dependency.read_bytes()).hexdigest()
    row = f"seed-dependency==1.0 --hash=sha256:{digest}"
    index = _SimpleIndex(tmp_path / "index", [dependency])
    try:
        monkeypatch.setattr(seed, "INDEXES", (index.url,))
        requirements = tmp_path / "seed.txt"
        requirements.write_text(row + "\n")
        baseline = tmp_path / "baseline"
        subprocess.run(
            [uv, "venv", "--python", sys.executable, str(baseline)], check=True, capture_output=True
        )
        command = seed.command(uv, baseline / "bin/python", requirements)
        command.remove("--offline")
        subprocess.run(command, check=True, capture_output=True)
        monkeypatch.setattr(package_installation, "IMAGE_UV_CACHE", root / "uv-cache")
        installed = package_installation.install(
            tmp_path / "installs",
            package="proof/seed-dependency",
            release="1.0",
            python=Path(sys.executable),
            requirements=f"--index-url {index.url}\n{row}\n".encode(),
        )
    finally:
        index.close()
    module = installed.site_packages / "seed_dependency.py"
    assert module.is_symlink() and module.resolve().is_relative_to(root / "uv-cache")
    value = subprocess.run(
        [str(installed.python), "-I", "-c", "import seed_dependency; print(seed_dependency.VALUE)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert value.stdout.strip() == "91"
