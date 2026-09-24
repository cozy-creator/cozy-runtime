"""Precompiled project variants execute through the ordinary private installers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.tags import Tag, sys_tags

from cozy_runtime.internal import canonical, package_environment, placement_materialization
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


def _invoke(installed: package_environment.InstalledEnvironment) -> None:
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


def test_compiled_project_local_placement_and_published_exact_lock(tmp_path: Path) -> None:
    """Both development placements and published workers install the same native bytes."""
    python = Path(sys.executable)
    base = package_environment.observe_base(python)
    project = _compile(tmp_path / "compiled", python)
    fact, verified = package_environment.measure_wheel(project, "project", base)
    document: dict[str, object] = {"local_wheels": []}
    digest = (
        "sha256:"
        + hashlib.sha256(
            canonical.write(
                {
                    **document,
                    "format": placement_materialization.ENVIRONMENT_FORMAT,
                }
            )
        ).hexdigest()
    )
    source = placement_materialization.package_environment_input(
        package_name="proof/native-project",
        release="1.0",
        project_wheel={
            "ref": {"digest": fact.digest, "length": fact.length},
            "distribution": fact.distribution,
            "version": fact.version,
            "filename": fact.filename,
            "tags": list(fact.tags),
            "import_roots": list(fact.import_roots),
        },
        environment=document,
        environment_digest=digest,
        wheel_paths={fact.digest: project},
        verified_wheels={fact.digest: verified},
    )
    local = package_environment.materialize(
        source,
        tmp_path / "local",
        base=base,
        exact_dependencies=True,
    )
    _invoke(local)
    index = _SimpleIndex(tmp_path / "index", [project])
    try:
        locked = package_environment.read_locked_requirements(
            (
                f"--index-url {index.url}\nnative-project @ "
                f"{index.url}native-project/{project.name} --hash={fact.digest}\n"
            ).encode()
        )
        # The pod's bounded path fetches and inspects selected exact artifacts,
        # then calls the same offline uv installer as development placements.
        staging = tmp_path / "pod"
        package_environment._install_locked_bounded(
            package_environment._uv_binary(),
            python,
            staging,
            locked.rows,
        )
        result = subprocess.run(
            [
                str(staging / "bin/python"),
                "-I",
                "-c",
                "import native_project; print(native_project.answer())",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "42"
        published = package_environment.materialize_locked(
            locked,
            locked.rows,
            tmp_path / "published",
            environment_digest=digest,
            base=base,
        )
        _invoke(published)
    finally:
        index.close()


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
    # The managed bare interpreter can coexist with the controller's venv.
    base = package_environment.observe_base(python)
    project = stable_project if stable else _compile(tmp_path / "compiled", python)
    from test_unpublished_exact_dependencies import _material

    installed = package_environment.materialize(
        _material(project, (), base),
        tmp_path / "installed",
        python=python,
        base=base,
        exact_dependencies=True,
    )
    _invoke(installed)


@pytest.mark.parametrize(
    "tag",
    [
        "cp314-cp314-linux_x86_64",
        "cp313-abi3-linux_x86_64",
        "cp312-cp312-manylinux_2_99_x86_64",
        "cp312-cp312-manylinux_2_17_aarch64",
        "cp312-cp312-win_amd64",
        "cp312-cp312-any",
    ],
)
def test_native_variants_require_actual_python_abi_and_platform(tmp_path: Path, tag: str) -> None:
    base = dataclasses.replace(
        package_environment.observe_base(),
        python_abi="cp312",
        python_full_version="3.12.12",
    )
    wheel = package_environment.Wheel(
        "native-project",
        "1.0",
        f"native_project-1.0-{tag}.whl",
        "sha256:" + "a" * 64,
        1,
        (tag,),
        "project",
    )
    with pytest.raises(package_environment.EnvironmentRefusal, match="platform_incompatible"):
        package_environment._require_compatible_tags(wheel, base)
    assert not (tmp_path / "installed").exists()


def test_actual_supported_tags_match_packaging() -> None:
    base = package_environment.observe_base()
    assert package_environment._compatible_tags(base) == frozenset(sys_tags())
    assert Tag("cp312", "abi3", "linux_x86_64") in package_environment._compatible_tags(base)


@pytest.mark.parametrize("filename", ["payload.so", "innocent.dat"])
def test_native_project_cannot_be_mislabeled_as_a_pure_wheel(tmp_path: Path, filename: str) -> None:
    project = _wheel(
        tmp_path,
        "native_project",
        (),
        b"VALUE=1\n",
        resources={filename: b"\x7fELFpayload"},
    )
    with pytest.raises(package_environment.EnvironmentRefusal, match="tag_not_concrete"):
        package_environment.measure_wheel(project, "project", package_environment.observe_base())
