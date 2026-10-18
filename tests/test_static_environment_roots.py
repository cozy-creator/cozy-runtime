"""The static reader finds a venv's site-packages in either OS's layout, reading only files."""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_runtime.internal.static_interface import environment_roots


@pytest.mark.parametrize(
    ("python", "site"),
    [("bin/python", "lib/python3.12/site-packages"), ("Scripts/python.exe", "Lib/site-packages")],
)
def test_a_venvs_site_packages_is_its_root_on_every_os(
    tmp_path: Path, python: str, site: str
) -> None:
    venv = tmp_path / "env"
    (venv / site).mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    interpreter = venv / python
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.touch()
    assert environment_roots(interpreter, project=tmp_path) == ((venv / site).resolve(),)
