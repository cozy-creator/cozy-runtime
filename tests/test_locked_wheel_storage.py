from __future__ import annotations

import hashlib
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from cozy_runtime.internal import package_environment as environment
from cozy_runtime.internal import storage_admission as storage
from test_private_exact_dependencies import _wheel


def test_standard_direct_requirements_preserve_exact_wheel_and_legacy_pins():
    direct = b"proof @ https://files.example/proof-1.0-py3-none-any.whl --hash=sha256:" + b"a" * 64
    selected = environment.read_locked_requirements(direct).rows[0]
    assert (selected.name, selected.version, selected.hashes) == (
        "proof",
        "1.0",
        ("sha256:" + "a" * 64,),
    )
    assert selected.url == "https://files.example/proof-1.0-py3-none-any.whl"
    legacy = environment.read_locked_requirements(b"proof==1.0 --hash=sha256:" + b"a" * 64).rows[0]
    assert legacy.version == "1.0" and not legacy.url


@pytest.mark.parametrize(
    "target",
    [
        "https://user:secret@example/proof-1.0-py3-none-any.whl",
        "file:///proof-1.0-py3-none-any.whl",
        "https://example/different-1.0-py3-none-any.whl",
        "https://example/source.tar.gz",
    ],
)
def test_direct_lock_rejects_credentials_other_artifacts_and_wrong_distribution(target):
    with pytest.raises(environment.EnvironmentRefusal):
        environment.read_locked_requirements(f"proof @ {target} --hash=sha256:{'a' * 64}".encode())


def wheel_fixture(tmp_path: Path):
    candidate = _wheel(tmp_path, "bounded_proof", (), b"VALUE = 73\n")
    digest = "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()
    raw = f"bounded-proof @ https://files.example/{candidate.name} --hash={digest}".encode()
    row = environment.read_locked_requirements(raw).rows[0]
    operation = "locked-" + hashlib.sha256(row.text.encode()).hexdigest()
    root = tmp_path / "installs"
    wheels = root / ".stage" / operation / "wheels"
    wheels.mkdir(parents=True)
    target = wheels / candidate.name
    candidate.rename(target)
    return root, target, row


def test_refused_expansion_does_not_start_uv_or_create_environment(tmp_path, monkeypatch):
    root, wheel, row = wheel_fixture(tmp_path)
    staging = root / "generation"
    monkeypatch.setattr(storage, "fetch_wheel", lambda *args: (wheel, wheel.stat().st_size))

    @contextmanager
    def refuse(*bounds):
        assert bounds[0].bytes > wheel.stat().st_size
        assert bounds[0].inodes > 4
        raise storage.StorageRefusal("disk full before expansion")
        yield

    monkeypatch.setattr(storage, "admit", refuse)
    before = set(root.rglob("*"))
    with pytest.raises(storage.StorageRefusal):
        environment._install_locked_bounded(
            environment._uv_binary(), Path(sys.executable), staging, [row]
        )
    assert set(root.rglob("*")) == before
    assert not staging.exists()


def test_exact_downloaded_wheel_installs_with_uv_offline(tmp_path, monkeypatch):
    root, wheel, row = wheel_fixture(tmp_path)
    staging = root / "generation"
    monkeypatch.setattr(storage, "fetch_wheel", lambda *args: (wheel, wheel.stat().st_size))
    admitted = []

    @contextmanager
    def allow(*bounds):
        admitted.extend(bounds)
        yield

    monkeypatch.setattr(storage, "admit", allow)
    environment._install_locked_bounded(
        environment._uv_binary(), Path(sys.executable), staging, [row]
    )
    assert len(admitted) == 1
    result = subprocess.run(
        [
            str(staging / "bin/python"),
            "-B",
            "-c",
            "import bounded_proof; print(bounded_proof.VALUE)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "73"
    assert not list(root.glob(".uv-cache-*"))
