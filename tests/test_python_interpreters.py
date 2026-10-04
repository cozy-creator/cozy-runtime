"""Interpreter selection: the lowest compatible installed CPython, from a finite inventory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cozy_runtime.internal import python_interpreters


def test_installed_window_selects_lowest_compatible() -> None:
    choices = python_interpreters.available()
    assert choices
    assert python_interpreters.select(">=3.12").version == choices[0].version
    for choice in choices:
        selected = python_interpreters.select(">=3.12", choice.version)
        assert selected.version == choice.version
        assert selected.abi == choice.abi


def test_configured_inventory_is_finite_and_empty_inventory_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COZY_PACKAGE_PYTHONS", str(Path(sys.executable)))
    assert len(python_interpreters.available()) == 1
    monkeypatch.setenv("COZY_PACKAGE_PYTHONS", "")
    with pytest.raises(python_interpreters.InterpreterRefusal):
        python_interpreters.available()
