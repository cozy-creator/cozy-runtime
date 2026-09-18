"""One worker installs isolated package closures on each installed supported CPython."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import canonical, package_environment, python_interpreters
from cozy_runtime.internal.worker.package_prepare import PreparationRefusal, _select_python
from test_unpublished_exact_dependencies import _material, _wheel


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
        selected, base = _select_python(Path(sys.executable), f"=={minor}.*", choice.version)
        base = base or package_environment.observe_base(selected)
        source = _material(project, (dependency,), base)
        installed = package_environment.materialize(
            source, tmp_path / "worker", python=selected, base=base, exact_dependencies=True
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
        identity = canonical.parse_canonical(
            (installed.generation / package_environment.CONTENT_IDENTITY_NAME).read_bytes()
        )
        assert identity["interpreter"]["python_full_version"] == choice.version
        assert identity["interpreter"]["python_abi"] == choice.abi
        held.append(installed)
    assert len({item.generation for item in held}) == len(held)


def test_dependency_snapshot_cannot_cross_python_abi() -> None:
    choices = python_interpreters.available()
    if len({item.abi for item in choices}) < 2:
        pytest.skip("two installed Python ABIs required")
    with pytest.raises(package_environment.EnvironmentRefusal, match="dependency_python_mismatch"):
        package_environment._require_same_interpreter(choices[0].executable, choices[-1].executable)
