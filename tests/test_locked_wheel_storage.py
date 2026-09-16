from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

import pytest

from cozy_runtime.internal import package_environment as environment
from cozy_runtime.internal import storage_admission as storage
from test_unpublished_exact_dependencies import _wheel


def test_standard_direct_requirements_preserve_exact_wheel_and_legacy_pins() -> None:
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
def test_direct_lock_rejects_credentials_other_artifacts_and_wrong_distribution(
    target: str,
) -> None:
    with pytest.raises(environment.EnvironmentRefusal):
        environment.read_locked_requirements(f"proof @ {target} --hash=sha256:{'a' * 64}".encode())


def wheel_fixture(tmp_path: Path) -> tuple[Path, Path, environment.LockedRow]:
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


def test_refused_expansion_does_not_start_uv_or_create_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, wheel, row = wheel_fixture(tmp_path)
    staging = root / "generation"
    monkeypatch.setattr(storage, "enabled", lambda: True)
    monkeypatch.setattr(storage, "fetch_wheel", lambda *args: (wheel, wheel.stat().st_size))

    def refuse(*bounds: storage.Write) -> NoReturn:
        assert bounds[0].bytes > wheel.stat().st_size
        assert bounds[0].inodes > 4
        raise storage.StorageRefusal("disk full before expansion")

    monkeypatch.setattr(storage, "admit", refuse)
    before = set(root.rglob("*"))
    with pytest.raises(storage.StorageRefusal):
        environment._install_locked_bounded(
            environment._uv_binary(), Path(sys.executable), staging, [row]
        )
    assert set(root.rglob("*")) == before
    assert not staging.exists()


def test_exact_downloaded_wheel_installs_with_uv_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, wheel, row = wheel_fixture(tmp_path)
    staging = root / "generation"
    monkeypatch.setattr(storage, "enabled", lambda: True)
    monkeypatch.setattr(storage, "fetch_wheel", lambda *args: (wheel, wheel.stat().st_size))

    original_hash = environment._hash_file
    checked: list[Path] = []

    def verify_new_archive(path: Path) -> str:
        assert path != wheel, "Runtime rehashed the supervisor-verified acquisition"
        checked.append(path)
        return original_hash(path)

    monkeypatch.setattr(environment, "_hash_file", verify_new_archive)
    admitted: list[storage.Write] = []

    @contextmanager
    def allow(*bounds: storage.Write) -> Iterator[None]:
        admitted.extend(bounds)
        yield

    monkeypatch.setattr(storage, "admit", allow)
    environment._install_locked_bounded(
        environment._uv_binary(), Path(sys.executable), staging, [row]
    )
    assert len(admitted) == 2
    assert admitted[1].path == root / "wheel-cache"
    assert admitted[1].bytes >= wheel.stat().st_size
    assert checked and all(path.is_relative_to(root / "wheel-cache") for path in checked)
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


def test_complete_locked_install_admits_metadata_and_releases_wheel_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, wheel, row = wheel_fixture(tmp_path)
    locked = environment.read_locked_requirements(
        ("--index-url https://pypi.org/simple\n" + row.text + "\n").encode()
    )
    monkeypatch.setattr(storage, "enabled", lambda: True)
    fetched: list[tuple[str, ...]] = []
    released: list[str] = []
    held: list[storage.Write] = []

    def fetch(*args: str) -> tuple[Path, int]:
        assert not held
        fetched.append(args)
        return wheel, wheel.stat().st_size

    @contextmanager
    def admit(*bounds: storage.Write) -> Iterator[None]:
        assert not held, "metadata/installer acquired nested host exclusion"
        held.extend(bounds)
        try:
            yield
        finally:
            held.clear()

    def release(operation: str) -> None:
        assert not held
        assert list((root / "receipts").glob("*.json")), (
            "released before immutable install completed"
        )
        released.append(operation)
        shutil.rmtree(root / ".stage" / operation, ignore_errors=True)

    monkeypatch.setattr(storage, "admit", admit)
    monkeypatch.setattr(storage, "fetch_wheel", fetch)
    monkeypatch.setattr(storage, "release_wheels", release)
    installed = environment.materialize_locked(
        locked,
        locked.rows,
        root,
        environment_digest="sha256:" + "73" * 32,
        python=Path(sys.executable),
    )
    assert len(fetched) == 1 and len(released) == 1
    assert not wheel.exists()
    replayed = environment.materialize_locked(
        locked,
        locked.rows,
        root,
        environment_digest="sha256:" + "73" * 32,
        python=Path(sys.executable),
    )
    assert replayed.generation == installed.generation and replayed.reused
    assert len(fetched) == 1


def test_locked_sdk_snapshot_survives_sdk_and_index_loss(tmp_path: Path) -> None:
    from cozy_runtime.internal import canonical
    from test_publish_derivation import _SimpleIndex

    dependency = _wheel(tmp_path, "sdk_dependency", (), b"VALUE = 73\n")
    project = _wheel(tmp_path, "sdk_project", ("sdk-dependency==1.0",), b"VALUE = 0\n")
    sdk = tmp_path / "sdk"
    uv = str(environment._uv_binary())
    subprocess.run([uv, "venv", "--python", sys.executable, str(sdk)], check=True)
    python = sdk / "bin/python"
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), str(dependency), str(project)], check=True
    )
    # The mutable SDK's project bytes are superseded by the exact exported wheel.
    project = _wheel(
        tmp_path, "sdk_project", ("sdk-dependency==1.0",), b"from sdk_dependency import VALUE\n"
    )
    index = _SimpleIndex(tmp_path / "index", [project])
    try:
        locked = environment.read_locked_requirements(
            (
                f"--index-url {index.url}\nsdk-project==1.0 "
                f"--hash=sha256:{hashlib.sha256(project.read_bytes()).hexdigest()}\n"
            ).encode()
        )
        digest = "sha256:" + "31" * 32
        root = tmp_path / "installs"
        installed = environment.materialize_locked(
            locked,
            locked.rows,
            root,
            environment_digest=digest,
            python=Path(sys.executable),
            dependency_python=python,
        )
    finally:
        index.close()
    receipt = environment.parse_environment_receipt(installed.receipt_bytes)
    assert receipt["environment_digest"] == digest
    assert receipt["locked_requirements"] == {"digest": locked.digest, "length": locked.length}
    identity = canonical.parse_canonical(
        (installed.generation / environment.CONTENT_IDENTITY_NAME).read_bytes()
    )
    assert identity["dependency_snapshot"]["digest"].startswith("sha256:")
    shutil.rmtree(sdk)
    project.unlink()
    reopened = environment.open_generation(
        root,
        receipt_digest=installed.receipt_digest,
        environment_digest=digest,
        python=Path(sys.executable),
    )
    result = subprocess.run(
        [str(reopened.python), "-I", "-c", "import sdk_project; print(sdk_project.VALUE)"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "73"
