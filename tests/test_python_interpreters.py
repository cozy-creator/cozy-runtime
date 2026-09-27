"""One worker installs isolated package closures on each installed supported CPython."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import package_installation, python_interpreters
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal, _select_python
from test_unpublished_exact_dependencies import _wheel


def test_unsupported_range_refuses_before_install() -> None:
    with pytest.raises(PreparationRefusal, match="supported window"):
        _select_python(Path(sys.executable), ">=3.11,<3.12", "")
    with pytest.raises(PreparationRefusal, match="supported window"):
        _select_python(Path(sys.executable), ">=3.15", "")


def test_installed_window_selects_lowest_compatible() -> None:
    choices = python_interpreters.available()
    assert choices
    assert python_interpreters.select(">=3.12").version == choices[0].version
    for choice in choices:
        selected = python_interpreters.select(">=3.12", choice.version)
        assert selected.version == choice.version
        assert selected.abi == choice.abi


def test_same_worker_materializes_multiple_python_closures(tmp_path: Path) -> None:
    choices = python_interpreters.available()
    minors = {".".join(item.version.split(".")[:2]) for item in choices}
    if len(minors) < 2:
        pytest.skip("install multiple supported CPythons to exercise cross-interpreter execution")
    held = []
    for choice in choices:
        minor = ".".join(choice.version.split(".")[:2])
        wheels = tmp_path / minor
        wheels.mkdir(exist_ok=True)
        dependency = _wheel(wheels, "private_helper", (), f"VALUE={minor!r}\n".encode())
        project = _wheel(
            wheels,
            "private_client",
            ("private-helper==1.0",),
            b"from private_helper import VALUE\n",
            requires_python=f"=={minor}.*",
        )
        selected, _ = _select_python(Path(sys.executable), f"=={minor}.*", choice.version)
        installed = package_installation.install(
            tmp_path / "worker",
            package="local/private-client",
            release="1.0",
            python=selected,
            wheels=(project, dependency),
        )
        actual = subprocess.run(
            [
                str(installed.python),
                "-I",
                "-c",
                "import json,platform,private_client; "
                "print(json.dumps([platform.python_version(),private_client.VALUE]))",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        assert json.loads(actual.stdout) == [choice.version, minor]
        held.append(installed)
    assert len({item.generation for item in held}) == len(held)


def test_configured_inventory_is_finite_and_empty_inventory_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COZY_PACKAGE_PYTHONS", str(Path(sys.executable)))
    assert len(python_interpreters.available()) == 1
    monkeypatch.setenv("COZY_PACKAGE_PYTHONS", "")
    with pytest.raises(python_interpreters.InterpreterRefusal):
        python_interpreters.available()
