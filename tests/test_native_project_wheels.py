"""Precompiled project variants execute through the ordinary private installers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import package_installation
from cozy_runtime.internal.package_environment import EnvironmentRefusal
from test_publish_derivation import _SimpleIndex
from test_unpublished_exact_dependencies import _wheel

_SOURCE = """
#include <Python.h>
static PyObject *answer(PyObject *self, PyObject *args) {
    return PyLong_FromLong(42);
}
static PyMethodDef methods[] = {{"answer", answer, METH_NOARGS, "compiled proof"}, {NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native_project", NULL, -1, methods};
PyMODINIT_FUNC PyInit__native_project(void) { return PyModule_Create(&module); }
"""


def _compile(root: Path, python: Path, *, stable: bool = False) -> Path:
    root.mkdir()
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import json,sysconfig,sys; print(json.dumps([sysconfig.get_path('include'), "
            "sysconfig.get_config_var('EXT_SUFFIX'), sys.implementation.cache_tag]))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    include, extension, cache_tag = json.loads(result.stdout)
    abi = cache_tag.replace("cpython-", "cp")
    tag = f"{abi}-{abi}-linux_x86_64"
    if stable:
        extension, tag = ".abi3.so", "cp312-abi3-linux_x86_64"
    source = root / "native.c"
    source.write_text(_SOURCE)
    output = root / ("_native_project" + extension)
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-O2",
            "-I",
            include,
            *(["-DPy_LIMITED_API=0x030C0000"] if stable else []),
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = _wheel(
        root,
        "native_project",
        (),
        b"from _native_project import answer\n",
        resources={
            output.name: output.read_bytes(),
            "native_project-1.0.dist-info/WHEEL": (
                "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n" + f"Tag: {tag}\n"
            ).encode(),
        },
    )
    return wheel.rename(wheel.with_name(f"native_project-1.0-{tag}.whl"))


def _invoke(installed: package_installation.InstalledEnvironment) -> None:
    result = subprocess.run(
        [
            str(installed.python),
            "-I",
            "-c",
            "import native_project; print(native_project.answer())",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "42"
    assert len(list(installed.site_packages.glob("_native_project*.so"))) == 1


def test_compiled_project_installs_from_local_wheel_and_exact_lock(tmp_path: Path) -> None:
    """Development placements and published workers install the same native bytes with uv."""
    python = Path(sys.executable)
    project = _compile(tmp_path / "compiled", python)
    _invoke(
        package_installation.install(
            tmp_path / "local",
            package="local/native-project",
            release="1.0",
            python=python,
            wheels=(project,),
        )
    )
    digest = hashlib.sha256(project.read_bytes()).hexdigest()
    index = _SimpleIndex(tmp_path / "index", [project])
    try:
        published = package_installation.install(
            tmp_path / "published",
            package="proof/native-project",
            release="1.0",
            python=python,
            requirements=(
                f"--index-url {index.url}\nnative-project @ "
                f"{index.url}native-project/{project.name} --hash=sha256:{digest}\n"
            ).encode(),
        )
    finally:
        index.close()
    _invoke(published)


@pytest.fixture(scope="module")
def stable_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _compile(
        tmp_path_factory.mktemp("stable-wheel") / "compiled", Path(sys.executable), stable=True
    )


@pytest.mark.parametrize("minor", [12, 13, 14])
@pytest.mark.parametrize("stable", [False, True], ids=["interpreter-abi", "shared-abi3"])
def test_compiled_project_on_available_supported_interpreters(
    tmp_path: Path,
    minor: int,
    stable: bool,
    stable_project: Path,
) -> None:
    result = subprocess.run(
        ["uv", "python", "find", "--no-python-downloads", f"3.{minor}"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        pytest.skip(f"CPython 3.{minor} is not installed")
    python = Path(result.stdout.strip())
    project = stable_project if stable else _compile(tmp_path / "compiled", python)
    installed = package_installation.install(
        tmp_path / "installed",
        package="local/native-project",
        release="1.0",
        python=python,
        wheels=(project,),
    )
    _invoke(installed)


@pytest.mark.parametrize(
    "tag",
    [
        "cp399-cp399-linux_x86_64",
        "cp312-cp312-manylinux_2_17_aarch64",
        "cp312-cp312-win_amd64",
    ],
)
def test_incompatible_native_variant_refuses_at_install(tmp_path: Path, tag: str) -> None:
    compiled = _compile(tmp_path / "compiled", Path(sys.executable))
    variant = compiled.rename(compiled.with_name(f"native_project-1.0-{tag}.whl"))
    with pytest.raises(EnvironmentRefusal, match="package_installation_uv_failed"):
        package_installation.install(
            tmp_path / "installed",
            package="local/native-project",
            release="1.0",
            python=Path(sys.executable),
            wheels=(variant,),
        )
