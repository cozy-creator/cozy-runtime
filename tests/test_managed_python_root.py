"""Local serve can share the CLI's interpreter root without sharing worker state."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import python_interpreters


def _configuration(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json,sys
from cozy_runtime.internal.config import read_config,ConfigError
try:
    value=read_config(json.load(sys.stdin))
except ConfigError as exc:
    print(exc.message)
    raise SystemExit(2)
print(json.dumps({'home':str(value.cozy_home),'python':str(value.managed_python_root),
                  'child':dict(value.child_base_env)}))
""",
        ],
        input=json.dumps(environment),
        text=True,
        capture_output=True,
    )


def test_worker_home_and_reported_python_root_remain_independent(tmp_path: Path) -> None:
    owner = tmp_path / "owner"
    cli = _configuration({"COZY_HOME": str(owner)})
    assert cli.returncode == 0, cli.stderr
    reported = json.loads(cli.stdout)["python"]
    worker_home = owner / "runtime"
    worker = _configuration({"COZY_HOME": str(worker_home), "COZY_PYTHON_ROOT": reported})
    assert worker.returncode == 0, worker.stderr
    assert json.loads(worker.stdout) == {"home": str(worker_home), "python": reported, "child": {}}
    assert not owner.exists(), "configuration and discovery must not create either store"


def test_relative_managed_root_is_refused_before_creation() -> None:
    result = _configuration({"COZY_HOME": "/unused", "COZY_PYTHON_ROOT": "relative/python"})
    assert result.returncode == 2
    assert "COZY_PYTHON_ROOT must be an absolute path" in result.stdout


def test_inventory_reports_exact_root_without_creating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "shared-owned-interpreters"
    monkeypatch.setattr(python_interpreters, "available", lambda **kw: ())
    document = python_interpreters.document(root=root)
    assert document["managed_root"] == str(root)
    assert document["interpreters"] == []
    assert not root.exists()
