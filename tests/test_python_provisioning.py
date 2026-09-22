"""One provisioner serves local capture and remote package preparation."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.internal import image_inventory, package_environment, storage_admission
from cozy_runtime.internal import python_interpreters as p
from cozy_runtime.internal.worker import package_prepare


def _candidate(version: str, path: Path = Path("/installed/python")) -> p.Interpreter:
    return p.Interpreter(path, version, "cp" + "".join(version.split(".")[:2]))


def _catalogue(*versions: str) -> str:
    return json.dumps(
        [
            {"implementation": "cpython", "variant": "default", "version": version}
            for version in versions
        ]
    )


def test_inventory_is_read_only_and_capability_is_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(p, "available", lambda **kw: (_candidate("3.12.12"),))
    monkeypatch.setattr(shutil, "which", lambda name: "/uv")
    root = tmp_path / "absent"
    document = p.document(root=root)
    rows = document["interpreters"]
    assert isinstance(rows, list)
    assert [i["version"] for i in rows] == ["3.12.12"]
    assert document["provisionable_minors"] == ["3.12", "3.13", "3.14"]
    assert not root.exists()
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert p.document(root=root)["provisionable_minors"] == []


def test_installed_match_wins_before_any_download_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = _candidate("3.13.2")
    monkeypatch.setattr(p, "available", lambda *a, **kw: (installed,))
    monkeypatch.setattr(p, "_download_version", lambda *a: pytest.fail("must reuse installed"))
    root = tmp_path / "absent"
    assert p.ensure(">=3.13", root=root) == installed
    assert not root.exists()


@pytest.mark.parametrize(
    ("requires", "version", "expected"),
    [
        (">=3.13,<3.13.5", "", "3.13.4"),
        (">=3.12", "3.13.2", "3.13.2"),
        (">=3.12", "3.13", "3.13.8"),
        (">=3.12", "", "3.12.12"),
    ],
)
def test_catalogue_selects_matching_patch_and_lowest_minor(
    monkeypatch: pytest.MonkeyPatch, requires: str, version: str, expected: str
) -> None:
    monkeypatch.setattr(
        p,
        "_run_uv",
        lambda *a, **kw: _catalogue(
            "3.11.12", "3.12.12", "3.13.2", "3.13.4", "3.13.8", "3.14.7", "3.15.0rc1"
        ),
    )
    assert p._download_version("/uv", p._constraint(requires, version)) == expected


@pytest.mark.parametrize("version", ["3.13.1+freethreaded", "3.15", "3.11.9"])
def test_unsupported_versions_never_create_install_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.setattr(p, "available", lambda *a, **kw: ())
    monkeypatch.setattr(shutil, "which", lambda _: "/uv")
    monkeypatch.setattr(p, "_run_uv", lambda *a, **kw: _catalogue("3.12.12", "3.13.8"))
    root = tmp_path / "absent"
    with pytest.raises(p.InterpreterRefusal):
        p.ensure(">=3.12", version, root=root)
    assert not root.exists()


def test_free_threaded_download_is_not_a_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        p,
        "_run_uv",
        lambda *a, **kw: json.dumps(
            [
                {"implementation": "cpython", "variant": "freethreaded", "version": "3.13.8"},
                {"implementation": "pypy", "variant": "default", "version": "3.13.8"},
            ]
        ),
    )
    with pytest.raises(p.InterpreterRefusal, match="no matching"):
        p._download_version("/uv", p._constraint(">=3.13", ""))


def test_concurrent_requests_share_install_and_keep_owned_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed: list[p.Interpreter] = []
    calls = []
    barrier = threading.Barrier(2)
    root = tmp_path / "python"
    monkeypatch.setattr(p, "available", lambda *a, **kw: tuple(installed))
    monkeypatch.setattr(shutil, "which", lambda name: "/uv")

    def download(*args: Any) -> str:
        barrier.wait(timeout=5)
        return "3.13.8"

    monkeypatch.setattr(p, "_download_version", download)

    def install(command: list[str], *, env: dict[str, str], cancel: Any) -> str:
        calls.append(command)
        assert env["UV_PYTHON_INSTALL_DIR"] == str(root / "managed")
        assert env["UV_CACHE_DIR"] == str(root / "cache")
        assert env["TMPDIR"] == str(root)
        assert "UV_PYTHON_MIRROR" not in env
        assert "--no-bin" in command
        (root / "managed").mkdir()
        time.sleep(0.1)
        installed.append(_candidate("3.13.8", root / "managed/python"))
        return ""

    monkeypatch.setattr(p, "_run_uv", install)
    monkeypatch.setenv("UV_PYTHON_MIRROR", "https://untrusted.invalid")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: p.ensure(">=3.13", root=root), range(2)))
    assert len(calls) == 1
    assert results[0] == results[1]
    assert root.stat().st_mode & 0o777 == 0o711


def test_failed_install_never_reports_an_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(p, "available", lambda *a, **kw: ())
    monkeypatch.setattr(shutil, "which", lambda name: "/uv")
    monkeypatch.setattr(p, "_download_version", lambda *a: "3.13.8")

    def fail(*a: Any, **kw: Any) -> str:
        raise p.InterpreterRefusal("download failed")

    monkeypatch.setattr(p, "_run_uv", fail)
    with pytest.raises(p.InterpreterRefusal, match="download failed"):
        p.ensure(">=3.13", root=tmp_path / "python")


def test_cancel_before_install_does_not_write(tmp_path: Path) -> None:
    root = tmp_path / "absent"
    with pytest.raises(p.InterpreterRefusal, match="cancelled"):
        p.ensure(">=3.13", root=root, cancel=lambda: True)
    assert not root.exists()


def test_cancellation_reaps_active_child(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"
    started = time.monotonic()

    def cancel() -> bool:
        return pid_file.exists() or time.monotonic() - started > 10

    with pytest.raises(p.InterpreterRefusal, match="cancelled"):
        p._run_uv(
            [
                sys.executable,
                "-c",
                "import os,time,pathlib; pathlib.Path("
                + repr(str(pid_file))
                + ").write_text(str(os.getpid())); time.sleep(30)",
            ],
            env=dict(os.environ),
            cancel=cancel,
        )
    assert pid_file.exists(), "child must start before cancellation"
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_remote_selection_uses_shared_ensure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = _candidate("3.13.8")
    calls = []

    def ensure(requires: str, version: str, **kw: Any) -> p.Interpreter:
        calls.append((requires, version, kw["root"]))
        return selected

    monkeypatch.setattr(p, "ensure", ensure)
    observed = package_environment.observe_base()
    monkeypatch.setattr(package_environment, "observe_base", lambda path: observed)
    assert package_prepare._select_python(
        Path("/base/python"), ">=3.13", "3.13.8", root=tmp_path / "python"
    ) == (selected.executable, observed)
    assert calls == [(">=3.13", "3.13.8", tmp_path / "python")]


def test_image_capability_is_not_installed_inventory() -> None:
    body = dict(
        format=image_inventory.INVENTORY_FORMAT,
        profile="cpu",
        python="3.12.12",
        distributions=[dict(name="cozy-runtime", version="0.18.13")],
        interpreters=[dict(version="3.12.12", abi="cp312")],
        provisionable_minors=["3.12", "3.13", "3.14"],
    )
    value = image_inventory.read_bytes(json.dumps(body).encode())
    assert value.interpreters == (("3.12.12", "cp312"),)
    assert value.provisionable_minors == ("3.12", "3.13", "3.14")
    body["provisionable_minors"] = ["3.13", "3.13"]
    with pytest.raises(image_inventory.ImageInventoryRefusal):
        image_inventory.read_bytes(json.dumps(body).encode())


def test_storage_refusal_precedes_download_and_directory_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cozy_runtime.internal import storage_admission

    monkeypatch.setattr(p, "available", lambda *a, **kw: ())
    monkeypatch.setattr(shutil, "which", lambda _: "/uv")
    monkeypatch.setattr(p, "_download_version", lambda *a: "3.13.8")
    monkeypatch.setattr(p, "_run_uv", lambda *a, **kw: pytest.fail("download cannot start"))

    def deny(*writes: storage_admission.Write) -> Any:
        assert writes[0].bytes >= 1 << 30
        raise storage_admission.StorageRefusal("insufficient storage")

    monkeypatch.setattr(storage_admission, "admit", deny)
    root = tmp_path / "absent"
    with pytest.raises(storage_admission.StorageRefusal):
        p.ensure(">=3.13", root=root)
    assert not root.exists()


def test_cli_ensure_calls_shared_provisioner_without_an_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cozy_runtime.cli import main

    calls = []

    def ensure(requires: str, version: str, **kw: Any) -> p.Interpreter:
        calls.append((requires, version, kw["root"]))
        return _candidate("3.13.8")

    monkeypatch.setattr(p, "ensure", ensure)
    # The real command parser and frozen default home are used; ensure is the only seam.
    assert main.main(["--json", "python-ensure", ">=3.13", "3.13.8"]) == 0
    assert calls == [(">=3.13", "3.13.8", Path.home() / ".cozy" / "python")]


def test_installed_system_interpreter_is_preferred_to_provisioning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    monkeypatch.delenv("COZY_PACKAGE_PYTHONS", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/uv")
    system = _candidate("3.13.8", Path("/system/python3.13"))
    control = _candidate("3.12.12", Path("/control/python"))
    monkeypatch.setattr(p, "probe", lambda path: system if path == system.executable else control)

    def discover(command: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        if "list" in command:
            return subprocess.CompletedProcess(command, 0, "[]", "")
        assert "find" in command and "--no-python-downloads" in command
        found = command[-1] == "3.13"
        return subprocess.CompletedProcess(command, 0 if found else 1, str(system.executable), "")

    monkeypatch.setattr(subprocess, "run", discover)
    monkeypatch.setattr(p, "_download_version", lambda *a: pytest.fail("installed Python must win"))
    root = tmp_path / "absent"
    assert p.ensure(">=3.13", root=root, default=control.executable) == system
    assert not root.exists()


def test_private_preparation_observes_cancellation_before_wheel_access(tmp_path: Path) -> None:
    from cozy_runtime.protocol import worker_pb2 as pb

    root = tmp_path / "absent"
    request = pb.PrepareLocalPackageRequest(
        operation_id="cancelled",
        install_root=str(root),
        python_requires=">=3.13",
        python_version="3.13.8",
        package=pb.DevelopmentPackage(
            package="local/cancelled",
            release="1.0",
            source_digest=b"s" * 32,
            local_revision_digest=b"r" * 32,
        ),
        wheels=[pb.LocalPackageWheel(filename="cancelled-1.0-py3-none-any.whl")],
    )
    with pytest.raises(package_prepare.PreparationRefusal, match="cancelled"):
        package_prepare.prepare_local_package(
            request,
            artifact_cache=tmp_path / "artifacts",
            install_root=root,
            python=Path(sys.executable),
            describe=lambda *a: b"",
            verified=lambda *a: None,
            cancel=lambda: True,
        )
    assert not root.exists()
