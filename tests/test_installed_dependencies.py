"""Real file snapshots refuse mutable hooks and survive loss of their source venv."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import _installed_dependencies as dependencies
from cozy_runtime.internal import canonical, package_environment


def _venv(path: Path, *, executable: Path | None = None) -> Path:
    (path / "bin").mkdir(parents=True)
    (path / "pyvenv.cfg").write_text("version_info = 3.12\n")
    if executable is None:
        # Starting the source interpreter would leave observable evidence and fail.
        (path / "bin/python").write_text('#!/bin/sh\ntouch "${0%/*}/executed"\nexit 77\n')
        (path / "bin/python").chmod(0o755)
    else:
        (path / "bin/python").symlink_to(executable)
    site = path / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    return site


def _distribution(
    site: Path, name: str, members: dict[str, bytes], *, editable: bool = False
) -> Path:
    info = site / f"{name}-1.0.dist-info"
    content = {
        f"{info.name}/METADATA": f"Metadata-Version: 2.3\nName: {name}\nVersion: 1.0\n".encode(),
        **members,
    }
    if editable:
        content[f"{info.name}/direct_url.json"] = json.dumps(
            {"url": "file:///absent-source", "dir_info": {"editable": True}}
        ).encode()
    rows = []
    for relative, body in content.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode()
        rows.append((relative, "sha256=" + digest, str(len(body))))
    record = info / "RECORD"
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows([*rows, (f"{info.name}/RECORD", "", "")])
    return info


def test_snapshot_excludes_editables_and_hooks_and_survives_source_removal(tmp_path: Path) -> None:
    source, staging = tmp_path / "source", tmp_path / "staging"
    source_site = _venv(source)
    target_site = _venv(staging, executable=Path(sys.executable))
    _distribution(
        source_site,
        "dependency",
        {
            "dependency/data.bin": b"accepted immutable bytes",
            "dangerous.pth": b"import os; raise RuntimeError('mutable hook')\n",
            "dependency-1.0.dist-info/entry_points.txt": (
                b"[console_scripts]\nsnapshot-proof = json:main\n"
            ),
            "../../../bin/snapshot-proof": b"#!/absent/source/bin/python\n",
        },
    )
    _distribution(source_site, "captured", {"captured.pth": b"/absent-source\n"}, editable=True)
    (source_site / "unrecorded.py").write_text("raise RuntimeError('not a distribution member')")
    manifest = dependencies.snapshot(source / "bin/python", staging, {"captured"})
    assert not (source / "bin/executed").exists()
    assert not list(target_site.glob("*.pth"))
    assert not (target_site / "captured-1.0.dist-info").exists()
    assert not (target_site / "unrecorded.py").exists()
    assert (target_site / "dependency/data.bin").stat().st_ino != (
        source_site / "dependency/data.bin"
    ).stat().st_ino
    (source_site / "dependency/data.bin").write_bytes(b"source changed")
    dependencies.verify_files(staging, manifest)
    shutil.rmtree(source)
    final = tmp_path / "published"
    staging.rename(final)
    result = subprocess.run(
        [str(final / "bin/snapshot-proof"), "--help"], capture_output=True, text=True
    )
    # The launcher reaches this environment's interpreter after publication; json:main
    # is intentionally absent, so the concrete failure proves the entrypoint import ran.
    assert "has no attribute 'main'" in result.stderr
    assert "not found" not in result.stderr
    assert (final / "lib/python3.12/site-packages/dependency/data.bin").read_bytes() == (
        b"accepted immutable bytes"
    )


@pytest.mark.parametrize("failure", ["hash", "editable", "symlink", "collision"])
def test_snapshot_refuses_unverifiable_dependency_files(tmp_path: Path, failure: str) -> None:
    source, staging = tmp_path / "source", tmp_path / "staging"
    source_site = _venv(source)
    target_site = _venv(staging)
    _distribution(
        source_site, "dependency", {"dependency.py": b"value=1\n"}, editable=failure == "editable"
    )
    if failure == "hash":
        (source_site / "dependency.py").write_bytes(b"value=2\n")
    elif failure == "symlink":
        (source_site / "dependency.py").unlink()
        (source_site / "dependency.py").symlink_to(__file__)
    elif failure == "collision":
        (target_site / "dependency.py").write_bytes(b"different distribution\n")
    with pytest.raises(package_environment.EnvironmentRefusal) as caught:
        dependencies.snapshot(source / "bin/python", staging, set())
    assert (
        caught.value.code
        == {
            "hash": "package_environment_dependency_member_changed",
            "editable": "package_environment_dependency_editable",
            "symlink": "package_environment_dependency_member_unsupported",
            "collision": "package_environment_dependency_member_collision",
        }[failure]
    )
    assert not (source / "bin/executed").exists()


def test_manifest_binds_bytes_and_detects_wheel_overwrite(tmp_path: Path) -> None:
    source, staging = tmp_path / "source", tmp_path / "staging"
    source_site = _venv(source)
    target_site = _venv(staging)
    _distribution(source_site, "dependency", {"dependency.py": b"value=1\n"})
    manifest = dependencies.snapshot(source / "bin/python", staging, set())
    (staging / dependencies.MANIFEST_NAME).write_bytes(manifest)
    ref = {"digest": "sha256:" + hashlib.sha256(manifest).hexdigest(), "length": len(manifest)}
    assert dependencies.read_manifest(staging, ref) == canonical.parse_canonical(manifest)
    (target_site / "dependency.py").write_bytes(b"value=2\n")
    with pytest.raises(package_environment.EnvironmentRefusal, match="member_collision"):
        dependencies.verify_files(staging, manifest)
    (staging / dependencies.MANIFEST_NAME).write_bytes(manifest.replace(b"1.0", b"2.0"))
    with pytest.raises(package_environment.EnvironmentRefusal, match="snapshot_invalid"):
        dependencies.read_manifest(staging, ref)
