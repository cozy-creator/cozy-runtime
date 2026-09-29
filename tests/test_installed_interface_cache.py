"""Owned installation replays reuse metadata without reimporting package code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from cozy_runtime.internal import discovery, package_installation, package_interface
from cozy_runtime.internal.worker import package_prepare
from cozy_runtime.internal.worker.session import Worker
from cozy_runtime.protocol import worker_pb2 as pb

RAW = b'{"format":"cozy.package.interface/1","application":"tiny:app","entrypoints":[],"jobs":[]}'


def installed(root: Path, identifier: str = "one") -> package_installation.InstalledEnvironment:
    directory = root / "installations" / identifier
    generation = directory / "venv"
    (generation / "bin").mkdir(parents=True, exist_ok=True)
    (generation / "bin/python").touch()
    (generation / "lib/python3.12/site-packages").mkdir(parents=True, exist_ok=True)
    (directory / "source").mkdir(exist_ok=True)
    # The production record writer atomically replaces the record even if the path and
    # contents of the environment happen to match a previous SDK incarnation.
    package_installation._write_record(
        directory,
        generation,
        {
            "package": "test/tiny",
            "release": "1.0",
            "machine": package_installation.machine_sdk(),
        },
    )
    return package_installation.open_installation(root, identifier)


def worker(describe):
    value = SimpleNamespace(
        preparation_slot=threading.Lock(), _installed_interfaces={}, _describe_in_slot=describe
    )
    value._describe_installed = MethodType(Worker._describe_installed, value)
    return value


def test_private_replay_imports_real_cpu_app_once(tmp_path: Path) -> None:
    env = installed(tmp_path)
    counter = tmp_path / "imports"
    (env.site_packages / "tiny.py").write_text(
        "from cozy_runtime.author import App\n"
        f"from pathlib import Path\nPath({str(counter)!r}).open('a').write('imported\\n')\n"
        "app = App()\n"
    )
    calls = []

    def describe(environment, distribution, *, application):
        calls.append(environment.incarnation)
        return subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from cozy_runtime.internal.discovery import discover_installed; "
                "from cozy_runtime.internal import package_interface; import sys; "
                "sys.stdout.buffer.write(package_interface.canonical_bytes("
                "package_interface.build(discover_installed('tiny:app', seconds=None))))",
            ],
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join(
                    [str(environment.site_packages), str(Path(__file__).parents[1] / "src")]
                ),
            },
        )

    held = worker(describe)
    request = pb.PrepareLocalPackageRequest(
        install_root=str(tmp_path),
        operation_id="request",
        files=[pb.LocalPackageFile(filename="source.tar")],
        package=pb.DevelopmentPackage(package="test/tiny", release="1.0", installation_id="one"),
    )
    for _ in range(2):
        result = package_prepare.prepare_local_package(
            request,
            artifact_cache=tmp_path / "artifacts",
            install_root=tmp_path,
            python=Path(sys.executable),
            describe=held._describe_installed,
            verified=lambda *_: None,
        )
        assert (
            package_interface.parse(result.installed_package.package_interface).application
            == "tiny:app"
        )
    assert len(calls) == 1
    assert counter.read_text() == "imported\n"


def test_new_source_and_rebuild_same_path_miss(tmp_path: Path) -> None:
    calls = []
    held = worker(lambda *a, **k: calls.append(a[0]) or RAW)
    first = installed(tmp_path)
    held._describe_installed(first, "tiny")
    held._describe_installed(package_installation.open_installation(tmp_path, "one"), "tiny")
    rebuilt = installed(tmp_path)
    assert rebuilt.generation == first.generation
    assert rebuilt.incarnation != first.incarnation
    held._describe_installed(rebuilt, "tiny")
    held._describe_installed(installed(tmp_path, "new-source"), "tiny")
    assert len(calls) == 3


def test_failures_invalid_documents_and_mutable_environments_are_not_cached(tmp_path: Path) -> None:
    answers = iter([ValueError("failed"), b"{}", RAW, RAW])
    calls = []

    def describe(*args, **kwargs):
        calls.append(1)
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    held = worker(describe)
    env = installed(tmp_path)
    with pytest.raises(ValueError, match="failed"):
        held._describe_installed(env, "tiny")
    with pytest.raises(package_interface.StalePackageInterface):
        held._describe_installed(env, "tiny")
    for _ in range(2):
        held._describe_installed(replace(env, incarnation=None), "tiny")
    assert len(calls) == 4
    assert held._installed_interfaces == {}


def test_distribution_application_and_concurrent_replays(tmp_path: Path) -> None:
    calls = []
    held = worker(lambda *a, **k: calls.append((a[1], k["application"])) or RAW)
    env = installed(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert (
            list(pool.map(lambda _: held._describe_installed(env, "tiny"), range(8))) == [RAW] * 8
        )
    held._describe_installed(env, "other")
    held._describe_installed(env, "tiny", application="tiny:alternate")
    assert calls == [("tiny", ""), ("other", ""), ("tiny", "tiny:alternate")]


def test_adopted_external_record_has_no_cache_incarnation(tmp_path: Path) -> None:
    env = installed(tmp_path)
    record_path = tmp_path / "installations/one/installation.json"
    record = json.loads(record_path.read_bytes())
    record["generation"] = str(tmp_path / "external")
    record_path.write_text(json.dumps(record))
    assert package_installation.open_installation(tmp_path, "one").incarnation is None
    assert env.incarnation is not None


def test_supervised_describe_does_not_arm_fixed_alarm(tmp_path: Path, monkeypatch) -> None:
    from cozy_runtime.internal.executor import Executor
    from cozy_runtime.internal.executor_commands import DescribeInstalled

    monkeypatch.setattr(discovery.signal, "setitimer", lambda *a: pytest.fail("fixed import alarm"))
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "supervised_tiny.py").write_text(
        "from cozy_runtime.author import App\napp = App()\n"
    )
    executor = Executor.__new__(Executor)
    assert executor.describe_installed(
        DescribeInstalled(
            distribution="unused",
            application="supervised_tiny:app",
            output=str(tmp_path / "interface.json"),
        )
    )["ok"]


def test_explicit_builder_deadline_remains(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(discovery.signal, "setitimer", lambda *a: calls.append(a))
    with discovery._wall_clock(60, "tiny:app"):
        pass
    assert [seconds for _, seconds in calls] == [60, 0]


def test_sdk_rollback_rebuilds_same_generation_and_misses(tmp_path: Path, monkeypatch) -> None:
    sdk = {"cozy-runtime": "first"}
    monkeypatch.setattr(package_installation, "machine_sdk", lambda: dict(sdk))
    env = installed(tmp_path)
    calls = []
    held = worker(lambda *a, **k: calls.append(a[0].incarnation) or RAW)

    def build(directory, generation, python, **kwargs):
        (generation / "bin").mkdir(parents=True)
        (generation / "bin/python").touch()
        (generation / "lib/python3.12/site-packages").mkdir(parents=True)
        return dict(sdk)

    monkeypatch.setattr(package_installation, "_build", build)
    generations = []
    for version in ("second", "third", "second"):
        sdk["cozy-runtime"] = version
        env = package_installation.refresh(tmp_path, env.installation_id)
        generations.append(env.generation)
        held._describe_installed(env, "tiny")
    assert generations[0] == generations[2]
    assert len(set(calls)) == 3


def test_repair_replaces_record_even_with_same_generation(tmp_path: Path, monkeypatch) -> None:
    env = installed(tmp_path)
    calls = []
    held = worker(lambda *a, **k: calls.append(a[0].incarnation) or RAW)
    held._describe_installed(env, "tiny")
    # A missing record makes installation creation repair the retained uv state.
    (tmp_path / "installations/one/installation.json").unlink()
    monkeypatch.setattr(package_installation, "_build", lambda *a, **k: {})
    repaired = package_installation.install(
        tmp_path,
        package=env.package,
        release=env.release,
        python=Path(sys.executable),
        installation_id=env.installation_id,
    )
    assert repaired.generation == env.generation
    assert repaired.incarnation is not None
    held._describe_installed(repaired, "tiny")
    assert len(set(calls)) == 2
