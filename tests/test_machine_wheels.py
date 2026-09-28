"""A machine running a local Runtime build gives packages that build, from the wheels the Host
kept beside its interpreter, or refuses; it never substitutes a published Runtime.

Real: uv, a venv, and wheel files. The machine's Runtime is a local-version `cozy-runtime`
wheel; the package is a wheel that requires `cozy-runtime>=0.18.0`.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from cozy_runtime.internal import host_paths, package_installation
from cozy_runtime.internal.package_environment import EnvironmentRefusal

LOCAL = "0.18.77+probe"


def wheel(directory: Path, name: str, version: str, requires: tuple[str, ...] = ()) -> Path:
    module = name.replace("-", "_")
    info = f"{module}-{version}.dist-info"
    files = {
        f"{module}/__init__.py": f"VERSION = {version!r}\n",
        f"{info}/METADATA": "Metadata-Version: 2.1\n"
        f"Name: {name}\nVersion: {version}\n"
        + "".join(f"Requires-Dist: {row}\n" for row in requires),
        f"{info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    rows = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=")
        rows.append(f"{path},sha256={digest.decode()},{len(text.encode())}")
    rows.append(f"{info}/RECORD,,")
    target = directory / f"{module}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(target, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
        archive.writestr(f"{info}/RECORD", "\n".join(rows) + "\n")
    return target


def uv(*argv: str) -> None:
    done = subprocess.run(["uv", *argv], capture_output=True, text=True, check=False)
    if done.returncode:
        raise EnvironmentRefusal("package_installation_failed", done.stderr[-800:])


@pytest.fixture
def package(tmp_path: Path) -> tuple[Path, Path]:
    venv = tmp_path / "venv"
    uv("venv", "--no-config", "--python", sys.executable, str(venv))
    python = venv / "bin" / "python"
    probe = wheel(tmp_path, "bound-probe", "1.0.0", ("cozy-runtime>=0.18.0",))
    uv(
        "pip",
        "install",
        "--no-config",
        "--offline",
        "--no-deps",
        "--python",
        str(python),
        str(probe),
    )
    (site,) = venv.glob("lib/python*/site-packages")
    return python, site


def select(python: Path, site: Path, wheels: Path) -> None:
    package_installation._select_sdk(
        site,
        [],
        site.parent / "constraints.txt",
        lambda rows: uv(
            "pip", "install", "--no-config", "--offline", "--python", str(python), *rows
        ),
        lambda: pytest.fail("a local build never falls back to the lock"),
        {"cozy-runtime": LOCAL},
        wheels,
    )


def test_a_local_build_installs_from_the_machines_wheels(
    package: tuple[Path, Path], tmp_path: Path
) -> None:
    python, site = package
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel(wheels, "cozy-runtime", LOCAL)
    select(python, site, wheels)
    installed = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions(path=[str(site)])
    }
    assert installed["cozy-runtime"] == LOCAL


def test_the_machine_wheels_live_in_the_machine_layout(tmp_path: Path) -> None:
    layout = host_paths.Layout(tmp_path / "machine")
    assert package_installation.machine_wheels(layout.install_root) == layout.machine_wheels
    assert layout.machine_wheels == tmp_path / "machine/opt/cozy/wheels"
    assert host_paths.Layout().machine_wheels == Path("/opt/cozy/wheels")
    assert package_installation.machine_wheels(tmp_path / "installs") is None


def test_a_local_build_without_its_wheel_refuses_rather_than_substitute(
    package: tuple[Path, Path], tmp_path: Path
) -> None:
    python, site = package
    with pytest.raises(EnvironmentRefusal) as refused:
        select(python, site, tmp_path / "absent")
    assert refused.value.code == "package_runtime_unavailable"
    assert LOCAL in refused.value.detail and "cozy-runtime>=0.18.0" in refused.value.detail
