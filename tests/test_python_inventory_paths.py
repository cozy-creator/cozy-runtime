"""Real uv inventory may print home-relative paths when invoked from the user's home."""

import shutil
from pathlib import Path

import pytest

from cozy_runtime.internal import python_interpreters


def test_managed_inventory_is_absolute_from_home(monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv is not installed")
    monkeypatch.delenv("COZY_PACKAGE_PYTHONS", raising=False)
    monkeypatch.chdir(Path.home())
    choices = python_interpreters.available()
    assert choices
    assert all(choice.executable.is_absolute() for choice in choices)
    assert all(choice.executable.is_file() for choice in choices)
    selected = python_interpreters.select(">=3.12", choices[0].version)
    assert selected.version == choices[0].version
