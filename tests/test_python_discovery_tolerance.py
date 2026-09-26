"""An unusable unrelated interpreter must not poison valid package selection."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import python_interpreters


def test_stale_managed_python_does_not_block_selected_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX failed-executable fixture")
    real = Path(sys.executable)
    actual = python_interpreters.probe(real)
    monkeypatch.setenv("COZY_PACKAGE_PYTHONS", str(real))
    stale = tmp_path / "managed/cpython-3.14.0-linux-x86_64/bin/python3.14"
    stale.parent.mkdir(parents=True)
    stale.write_text("#!/bin/sh\nexit 7\n")
    stale.chmod(0o700)
    with pytest.raises(python_interpreters.InterpreterRefusal):
        python_interpreters.probe(stale)
    selected = python_interpreters.select(">=3.12", actual.version, root=tmp_path)
    assert selected.version == actual.version and selected.executable == real
    inventory = python_interpreters.document(root=tmp_path)
    assert inventory["interpreters"] == [actual.document()]


def test_broken_configured_candidate_isolated_but_requested_abi_still_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "removed/bin/python3.14"
    real = Path(sys.executable)
    actual = python_interpreters.probe(real)
    monkeypatch.setenv("COZY_PACKAGE_PYTHONS", os.pathsep.join((str(missing), str(real))))
    assert python_interpreters.select(">=3.12", actual.version).abi == actual.abi
    unsupported = "3.14" if not actual.version.startswith("3.14.") else "3.13"
    with pytest.raises(python_interpreters.InterpreterRefusal, match="installed:"):
        python_interpreters.select("==" + unsupported + ".*")
    monkeypatch.setenv("COZY_PACKAGE_PYTHONS", str(missing))
    with pytest.raises(python_interpreters.InterpreterRefusal, match="installed: none"):
        python_interpreters.select(">=3.12")
