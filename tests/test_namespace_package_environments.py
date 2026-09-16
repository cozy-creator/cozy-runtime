"""Disjoint namespace wheels install together; executable ownership stays exclusive."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cozy_runtime.internal import package_environment as environment
from test_unpublished_exact_dependencies import _material, _wheel


def test_disjoint_namespace_packages_import_and_reopen(tmp_path: Path) -> None:
    base = environment.observe_base()
    bindings = _wheel(
        tmp_path,
        "cuda_bindings",
        (),
        b"",
        resources={"cuda/bindings/__init__.py": b"VALUE=19\n"},
    )
    pathfinder = _wheel(
        tmp_path,
        "cuda_pathfinder",
        (),
        b"",
        resources={"cuda_pathfinder-1.0.data/purelib/cuda/pathfinder/__init__.py": b"VALUE=23\n"},
    )
    project = _wheel(
        tmp_path,
        "namespace_client",
        ("cuda-bindings==1.0", "cuda-pathfinder==1.0"),
        b"from cuda.bindings import VALUE as A\n"
        b"from cuda.pathfinder import VALUE as B\nVALUE=A+B\n",
    )
    source = _material(project, (bindings, pathfinder), base)
    installed = environment.materialize(
        source, tmp_path / "installed", base=base, exact_dependencies=True
    )
    subprocess.run(
        [
            str(installed.python),
            "-I",
            "-c",
            "import cuda,namespace_client; assert cuda.__file__ is None; "
            "assert namespace_client.VALUE == 42",
        ],
        check=True,
        capture_output=True,
    )
    reopened = environment.materialize(
        source, tmp_path / "installed", base=base, exact_dependencies=True
    )
    assert reopened.reused and reopened.generation == installed.generation


@pytest.mark.parametrize(
    ("first", "second", "code"),
    [
        ({"shared/part.py": b"VALUE=1"}, {"shared/part.py": b"VALUE=2"}, "resource_collision"),
        (
            {"shared/part.py": b"VALUE=1"},
            {"second-1.0.data/purelib/shared/part.py": b"VALUE=2"},
            "resource_collision",
        ),
        ({"shared.py": b"VALUE=1"}, {"shared/part.py": b"VALUE=2"}, "import_root_duplicate"),
        (
            {"shared/__init__.py": b"VALUE=1"},
            {"shared/part.py": b"VALUE=2"},
            "import_root_duplicate",
        ),
        (
            {"shared/nested/__init__.py": b"VALUE=1"},
            {"shared/nested/part.py": b"VALUE=2"},
            "import_root_duplicate",
        ),
        (
            {"shared/part.py": b"VALUE=1"},
            {"second-1.0.data/data/lib/python3.12/site-packages/shared/part.py": b"VALUE=2"},
            "resource_collision",
        ),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_namespace_sharing_refuses_file_and_module_shadowing(
    tmp_path: Path, first: dict[str, bytes], second: dict[str, bytes], code: str, reverse: bool
) -> None:
    base = environment.observe_base()
    if reverse:
        first, second = second, first
    first_wheel = _wheel(tmp_path, "first", (), b"", resources=first)
    second_wheel = _wheel(tmp_path, "second", (), b"", resources=second)
    project = _wheel(tmp_path, "client", ("first==1.0", "second==1.0"), b"VALUE=1\n")
    dependencies = (first_wheel, second_wheel)
    source = _material(project, dependencies, base)
    with pytest.raises(environment.EnvironmentRefusal, match=code):
        environment.materialize(source, tmp_path / "refused", base=base, exact_dependencies=True)
    assert not (tmp_path / "refused").exists()
