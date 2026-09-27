"""Real shared derivations bind native numerical builds without importing caller packages."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable
from pathlib import Path

import pytest

from cozy_runtime.author._calls import _export
from cozy_runtime.derive.operations import prepare_model, quantize
from cozy_runtime.internal.memo_implementation import describe


def identity(fn: Callable[..., object]) -> dict[str, str]:
    exported = _export(fn)
    assert exported is not None
    return describe(exported.implementation)


@pytest.mark.parametrize("distribution_name", ["tensorfs", "numpy", "msgspec"])
def test_derivation_native_version_invalidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, distribution_name: str
) -> None:
    functions = (quantize,) if distribution_name == "numpy" else (quantize, prepare_model)
    original = [identity(fn) for fn in functions]
    assert all(row.get("operation_identity") for row in original), original
    installed = importlib.metadata.distribution(distribution_name)
    metadata = installed.read_text("METADATA")
    wheel = installed.read_text("WHEEL")
    assert metadata is not None and wheel is not None
    shadow = tmp_path / f"{distribution_name}-999.0.dist-info"
    shadow.mkdir()
    (shadow / "METADATA").write_text(
        metadata.replace(f"Version: {installed.version}\n", "Version: 999.0\n", 1)
    )
    (shadow / "WHEEL").write_text(wheel)
    monkeypatch.syspath_prepend(str(tmp_path))
    changed = [identity(fn) for fn in functions]
    assert all(row.get("operation_identity") for row in changed), changed
    assert all(first != second for first, second in zip(original, changed, strict=True))
