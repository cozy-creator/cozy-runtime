"""Private wheel revisions cannot borrow a different numerical base or omit a dependency."""

from __future__ import annotations

import base64
import csv
import dataclasses
import hashlib
import io
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
) -> Path:
    info = f"{name}-1.0.dist-info"
    metadata = f"Metadata-Version: 2.3\nName: {name}\nVersion: 1.0\n"
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
    path = root / f"{name}-1.0-py3-none-any.whl"
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


def test_private_refuses_divergent_base_before_install(tmp_path: Path) -> None:
    base = environment.observe_base(Path(sys.executable))
    base = dataclasses.replace(
        base,
        distributions=(
            *(row for row in base.distributions if row[0] != "numpy"),
            ("numpy", "2.5.2"),
        ),
    )
    project = _wheel(tmp_path, "private_pin", ("numpy==2.5.3",), b"VALUE=1\n")
    material = _material(project, (), base)
    root = tmp_path / "refused"
    with pytest.raises(environment.EnvironmentRefusal) as caught:
        environment.materialize(material, root, base=base, exact_dependencies=True)
    assert caught.value.code == "package_environment_dependency_base_conflict"
    assert not root.exists()


def test_private_missing_wheel_refuses_then_real_install_and_reopen_verify(tmp_path: Path) -> None:
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
