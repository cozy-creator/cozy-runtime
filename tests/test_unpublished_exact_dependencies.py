"""Captured wheel revisions cannot borrow a different numerical base or omit a dependency."""

from __future__ import annotations

import base64
import csv
import dataclasses
import hashlib
import io
import platform
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from cozy_runtime.internal import package_environment as environment
from cozy_runtime.internal.base_observation import Observation


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


def _material(
    project: Path, dependencies: tuple[Path, ...], base: Observation
) -> environment.Input:
    rows = [
        (*environment.measure_wheel(path, "project" if index == 0 else "custom", base), path)
        for index, path in enumerate((project, *dependencies))
    ]
    return environment.Input(
        "sha256:" + hashlib.sha256("".join(row[0].digest for row in rows).encode()).hexdigest(),
        rows[0][0].digest,
        tuple(row[0] for row in rows),
        tuple(
            environment.BlobGrant(row.digest, row.length, path, verified)
            for row, verified, path in rows
        ),
    )


@pytest.mark.parametrize("marker", ["python_full_version", "implementation_version"])
def test_patch_markers_require_the_dependency_on_the_actual_interpreter(
    tmp_path: Path, marker: str
) -> None:
    base = environment.observe_base(Path(sys.executable))
    version = platform.python_version()
    project = _wheel(
        tmp_path, "patch_client", (f"patch_helper==1.0; {marker} >= '{version}'",), b"VALUE=42\n"
    )
    with pytest.raises(environment.EnvironmentRefusal, match="dependency_missing"):
        environment.materialize(
            _material(project, (), base), tmp_path / "missing", base=base, exact_dependencies=True
        )
    assert not (tmp_path / "missing").exists()


def test_actual_patch_requires_python_and_inactive_marker_install(tmp_path: Path) -> None:
    base = environment.observe_base(Path(sys.executable))
    version = platform.python_version()
    project = _wheel(
        tmp_path,
        "patch_client",
        (f"patch_helper==1.0; python_full_version < '{version}'",),
        b"VALUE=42\n",
        requires_python=f">={version}",
    )
    installed = environment.materialize(
        _material(project, (), base), tmp_path / "installed", base=base, exact_dependencies=True
    )
    result = subprocess.run(
        [str(installed.python), "-I", "-c", "import patch_client; print(patch_client.VALUE)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "42"


@pytest.mark.parametrize("distribution", ["numpy", "setuptools"])
def test_unpublished_missing_dependency_cannot_borrow_base(
    tmp_path: Path, distribution: str
) -> None:
    base = environment.observe_base(Path(sys.executable))
    base = dataclasses.replace(
        base,
        distributions=(
            *(row for row in base.distributions if row[0] != distribution),
            (distribution, "2.5.2"),
        ),
    )
    project = _wheel(tmp_path, "private_pin", (f"{distribution}==2.5.3",), b"VALUE=1\n")
    material = _material(project, (), base)
    root = tmp_path / "refused"
    with pytest.raises(environment.EnvironmentRefusal) as caught:
        environment.materialize(material, root, base=base, exact_dependencies=True)
    assert caught.value.code == "package_environment_dependency_missing"
    assert not root.exists()


@pytest.mark.parametrize("damage", ["bytes", "symlink"])
def test_dependency_share_resource_installs_and_reuse_verifies_bytes(
    tmp_path: Path, damage: str
) -> None:
    base = environment.observe_base(Path(sys.executable))
    dependency = _wheel(
        tmp_path,
        "documented",
        (),
        b"VALUE=641\n",
        resources={"documented-1.0.data/data/share/man/man1/example.1": b"manual bytes\n"},
    )
    project = _wheel(tmp_path, "document_user", ("documented==1.0",), b"import documented\n")
    material = _material(project, (dependency,), base)
    installed = environment.materialize(
        material, tmp_path / "installed", base=base, exact_dependencies=True
    )
    resource = installed.generation / "share/man/man1/example.1"
    assert resource.read_bytes() == b"manual bytes\n"
    assert not (installed.site_packages / "share").exists()
    assert environment.materialize(
        material, tmp_path / "installed", base=base, exact_dependencies=True
    ).reused
    if damage == "bytes":
        resource.chmod(0o644)
        resource.write_bytes(b"changed text\n")
    else:
        other = tmp_path / "outside-manual"
        other.write_bytes(b"manual bytes\n")
        resource.parent.chmod(0o755)
        resource.unlink()
        resource.symlink_to(other)
    with pytest.raises(environment.EnvironmentRefusal, match="installed_resource_changed"):
        environment.materialize(
            material, tmp_path / "installed", base=base, exact_dependencies=True
        )


def test_native_resources_still_require_native_package_tags(tmp_path: Path) -> None:
    base = environment.observe_base(Path(sys.executable))
    resource = "resource-1.0.data/data/share/man/manual"
    for owner in ("project", "custom"):
        content = b"\x7fELFpretend native"
        wheel = _wheel(tmp_path, "resource", (), b"VALUE=1\n", resources={resource: content})
        with pytest.raises(environment.EnvironmentRefusal):
            environment.measure_wheel(wheel, owner, base)


@pytest.mark.parametrize(
    "member",
    [
        "resource-1.0.data/data/share/../../outside",
        "resource-1.0.data/scripts/../../outside",
        "resource-1.0.data/unknown/api.h",
    ],
)
def test_dependency_share_resource_keeps_closed_install_schemes(
    tmp_path: Path, member: str
) -> None:
    base = environment.observe_base(Path(sys.executable))
    wheel = _wheel(tmp_path, "resource", (), b"VALUE=1\n", resources={member: b"resource\n"})
    with pytest.raises(environment.EnvironmentRefusal):
        environment.measure_wheel(wheel, "custom", base)


@pytest.mark.parametrize("other", ["share/man/manual", "share/man/manual/page"])
def test_dependency_share_resource_collisions_refuse_before_install(
    tmp_path: Path, other: str
) -> None:
    base = environment.observe_base(Path(sys.executable))
    a = _wheel(
        tmp_path,
        "resource_a",
        (),
        b"VALUE=1\n",
        resources={"resource_a-1.0.data/data/share/man/manual": b"one"},
    )
    b = _wheel(
        tmp_path,
        "resource_b",
        (),
        b"VALUE=2\n",
        resources={f"resource_b-1.0.data/data/{other}": b"two"},
    )
    project = _wheel(
        tmp_path, "resource_user", ("resource_a==1.0", "resource_b==1.0"), b"VALUE=3\n"
    )
    with pytest.raises(environment.EnvironmentRefusal, match="resource_collision"):
        environment.materialize(
            _material(project, (a, b), base),
            tmp_path / "refused",
            base=base,
            exact_dependencies=True,
        )
    assert not (tmp_path / "refused").exists()


def test_unpublished_missing_wheel_refuses_then_real_install_and_reopen_verify(
    tmp_path: Path,
) -> None:
    base = environment.observe_base(Path(sys.executable))
    helper = _wheel(tmp_path, "closure_helper", (), b"VALUE=641\n")
    project = _wheel(
        tmp_path,
        "closure_client",
        ("closure_helper==1.0",),
        b"import closure_helper\nVALUE=closure_helper.VALUE+1\n",
    )
    missing = _material(project, (), base)
    with pytest.raises(environment.EnvironmentRefusal) as caught:
        environment.materialize(missing, tmp_path / "missing", base=base, exact_dependencies=True)
    assert caught.value.code == "package_environment_dependency_missing"
    assert not (tmp_path / "missing").exists()
    material = _material(project, (helper,), base)
    root = tmp_path / "installed"
    installed = environment.materialize(material, root, base=base, exact_dependencies=True)
    result = subprocess.run(
        [str(installed.python), "-I", "-c", "import closure_client;print(closure_client.VALUE)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "642"
    reused = environment.materialize(material, root, base=base, exact_dependencies=True)
    assert reused.reused and reused.content_digest == installed.content_digest
    metadata = reused.site_packages / "closure_helper-1.0.dist-info/METADATA"
    metadata.chmod(0o644)
    metadata.write_bytes(metadata.read_bytes().replace(b"Version: 1.0", b"Version: 2.0"))
    with pytest.raises(environment.EnvironmentRefusal) as changed:
        environment.materialize(material, root, base=base, exact_dependencies=True)
    assert changed.value.code == "package_environment_installed_dependency_changed"


def test_prebuilt_dependency_source_resources_are_installed_without_building(
    tmp_path: Path,
) -> None:
    base = environment.observe_base(Path(sys.executable))
    resources = {
        "resource_helper/data/wrapper.cc": b"not compilable C++\n",
        "resource_helper/data/include.h": b"not compilable C\n",
        "resource_helper/data/setup.py": b"raise RuntimeError('build must never run')\n",
        "resource_helper/data/statistics.dat": b"1.2500000  2.7500000\n" * 30_000,
    }
    helper = _wheel(tmp_path, "resource_helper", (), b"VALUE=37\n", resources=resources)
    project = _wheel(
        tmp_path, "resource_client", ("resource_helper==1.0",), b"import resource_helper\n"
    )
    installed = environment.materialize(
        _material(project, (helper,), base),
        tmp_path / "installed",
        base=base,
        exact_dependencies=True,
    )
    for member, content in resources.items():
        assert (installed.site_packages / member).read_bytes() == content
    result = subprocess.run(
        [str(installed.python), "-I", "-c", "import resource_helper;print(resource_helper.VALUE)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "37"


@pytest.mark.parametrize(
    "member", ["kernel.c", "include.h", "nested/setup.py", "nested/pyproject.toml"]
)
def test_project_native_build_resources_still_refuse(tmp_path: Path, member: str) -> None:
    base = environment.observe_base(Path(sys.executable))
    project = _wheel(tmp_path, "unsafe_project", (), b"VALUE=1\n", resources={member: b"data\n"})
    with pytest.raises(environment.EnvironmentRefusal) as caught:
        environment.measure_wheel(project, "project", base)
    assert caught.value.code == "package_environment_native_build_input_forbidden"


@pytest.mark.parametrize(
    "fault", ["total-expansion", "lying-size", "truncated", "oversized-metadata"]
)
def test_prebuilt_dependency_zip_bounds_remain_enforced(tmp_path: Path, fault: str) -> None:
    base = environment.observe_base(Path(sys.executable))
    wheel = _wheel(
        tmp_path, "bounded", (), b"VALUE=1\n", resources={"one.dat": b"a\n", "two.dat": b"b\n"}
    )
    raw = bytearray(wheel.read_bytes())
    if fault == "truncated":
        del raw[-10:]
    else:
        offset = raw.index(b"PK\x01\x02")
        while raw[offset : offset + 4] == b"PK\x01\x02":
            name_len, extra_len, comment_len = struct.unpack_from("<HHH", raw, offset + 28)
            name = raw[offset + 46 : offset + 46 + name_len].decode()
            if fault == "oversized-metadata" and name.endswith(".dist-info/METADATA"):
                struct.pack_into("<I", raw, offset + 24, environment.MAX_METADATA_BYTES + 1)
            elif fault != "oversized-metadata" and name in {"one.dat", "two.dat"}:
                struct.pack_into(
                    "<I", raw, offset + 24, (1 << 32) - 2 if fault == "total-expansion" else 3
                )
            offset += 46 + name_len + extra_len + comment_len
    wheel.write_bytes(raw)
    with pytest.raises(environment.EnvironmentRefusal) as caught:
        environment.measure_wheel(wheel, "custom", base)
    if fault == "total-expansion":
        assert caught.value.code == "package_environment_wheel_expansion_cap"
    elif fault == "oversized-metadata":
        assert caught.value.code == "package_environment_wheel_metadata_oversized"
    else:
        assert caught.value.code in {
            "package_environment_wheel_invalid",
            "package_environment_wheel_record_mismatch",
        }


def test_standard_wheel_schemes_survive_install_rename_and_snapshot(tmp_path: Path) -> None:
    from cozy_runtime.internal import _installed_dependencies as dependencies
    from test_installed_dependencies import _venv

    base = environment.observe_base()
    wheel = _wheel(
        tmp_path,
        "scheme_payload",
        (),
        b"VALUE=73\n",
        resources={
            "scheme_payload-1.0.data/scripts/ordinary-script": (
                b"#!python\nfrom scheme_payload import VALUE; print(VALUE)\n"
            ),
            "scheme_payload-1.0.data/scripts/shell-script": b"#!/bin/sh\nprintf 'shell-ok\\n'\n",
            "scheme_payload-1.0.data/headers/api.h": b"// API\n",
            "scheme_payload-1.0.data/data/lib/data-file": b"data bytes\n",
        },
    )
    project = _wheel(tmp_path, "scheme_client", ("scheme-payload==1.0",), b"VALUE=1\n")
    source = _material(project, (wheel,), base)
    installed = environment.materialize(
        source, tmp_path / "installed", base=base, exact_dependencies=True
    )
    assert (
        installed.generation / "include/site/python3.12/scheme-payload/api.h"
    ).read_bytes() == b"// API\n"
    assert (installed.generation / "lib/data-file").read_bytes() == b"data bytes\n"
    target = tmp_path / "captured"
    _venv(target, executable=Path(sys.executable))
    manifest = dependencies.snapshot(installed.python, target, set())
    dependencies.verify_files(target, manifest)
    for prefix in (installed.generation, target):
        for name, output in (("ordinary-script", "73"), ("shell-script", "shell-ok")):
            result = subprocess.run([str(prefix / "bin" / name)], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == output
    assert environment.materialize(
        source, tmp_path / "installed", base=base, exact_dependencies=True
    ).reused
