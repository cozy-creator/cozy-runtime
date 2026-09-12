"""The Runtime carrier authorizes one existing builtin, never foreign code."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from cozy_runtime.author import ConformanceError
from cozy_runtime.internal import builtin_operations, package_environment


def carrier(path: Path, *, change: str = "") -> Path:
    version = "0.16.8"
    root = f"cozy_runtime_operations-{version}.dist-info/"
    metadata = (
        "Metadata-Version: 2.4\nName: cozy-runtime-operations\nVersion: 0.16.8\n"
        "Requires-Python: >=3.12,<3.13\nRequires-Dist: cozy-runtime>=0.16.8,<1\n"
        "Requires-Dist: numpy>=1.26\nCozy-Builtin: operations\nCozy-Builtin-Interface: "
        + builtin_operations.digest()
        + "\n"
    ).encode()
    files = {
        root + "METADATA": metadata,
        root + "WHEEL": builtin_operations.WHEEL,
        root + "entry_points.txt": builtin_operations.ENTRY_POINTS,
    }
    if change == "code":
        files["untrusted.py"] = b"raise AssertionError('must not execute')\n"
    if change == "foreign-app":
        files[root + "entry_points.txt"] = b"[cozy.application]\ndefault=foreign:app\n"
    if change == "stale-interface":
        files[root + "METADATA"] = metadata.replace(
            builtin_operations.digest().encode(), ("sha256:" + "00" * 32).encode()
        )
    if change == "exact-numpy":
        files[root + "METADATA"] = metadata.replace(b"numpy>=1.26", b"numpy==2.5.2")
    if change == "undeclared":
        files[root + "METADATA"] += b"Untrusted-Capability: launch\n"
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    for name, body in sorted(files.items()):
        writer.writerow(
            (
                name,
                "sha256="
                + base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("="),
                str(len(body)),
            )
        )
    writer.writerow((root + "RECORD", "", ""))
    files[root + "RECORD"] = output.getvalue().encode()
    path.mkdir(parents=True, exist_ok=True)
    wheel = path / f"cozy_runtime_operations-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return wheel


def test_metadata_only_carrier_passes_ordinary_wheel_admission(tmp_path: Path) -> None:
    path = carrier(tmp_path)
    builtin_operations.validate_carrier(path, "0.16.8")
    base = package_environment.observe_base()
    measured, _ = package_environment.measure_wheel(path, "project", base)
    assert measured.import_roots == () and measured.distribution == builtin_operations.NAME
    root = tmp_path / "installed"
    with zipfile.ZipFile(path) as archive:
        archive.extractall(root)
    builtin_operations.validate_installed(root / "cozy_runtime_operations-0.16.8.dist-info")


@pytest.mark.parametrize(
    "change", ["code", "foreign-app", "stale-interface", "exact-numpy", "undeclared"]
)
def test_carrier_refuses_executable_or_changed_authority(tmp_path: Path, change: str) -> None:
    path = carrier(tmp_path, change=change)
    with pytest.raises(ConformanceError, match="builtin_carrier"):
        builtin_operations.validate_carrier(path, "0.16.8")
