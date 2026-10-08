"""Approved artifacts are data: refuse changed bytes, ambiguous names and extra files."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "release_assets", Path(__file__).resolve().parents[1] / "checks" / "release_assets.py"
)
assert SPEC is not None and SPEC.loader is not None
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


def test_approved_bytes_and_only_visible_release_files(tmp_path: Path) -> None:
    path = tmp_path / "cozy_runtime-1.2.3-py3-none-any.whl"
    path.write_bytes(b"artifact bytes, not executable test code")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sums = f"{digest} {path.name}"
    (tmp_path / ".gitignore").write_text("*\n")
    assert assets.verify(tmp_path, sums) == f"{digest}  {path.name}"
    path.write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="digest differs"):
        assets.verify(tmp_path, sums)


@pytest.mark.parametrize("change", ["missing", "extra", "symlink", "directory"])
def test_missing_extra_or_nonordinary_artifacts_refuse(tmp_path: Path, change: str) -> None:
    path = tmp_path / "artifact.whl"
    path.write_bytes(b"artifact")
    sums = f"{hashlib.sha256(path.read_bytes()).hexdigest()} {path.name}"
    if change == "extra":
        (tmp_path / "extra.whl").write_bytes(b"extra")
    else:
        path.unlink()
        if change == "symlink":
            target = tmp_path / ".target"
            target.write_bytes(b"artifact")
            path.symlink_to(target)
        elif change == "directory":
            path.mkdir()
    with pytest.raises(ValueError, match=r"files differ|not an ordinary file"):
        assets.verify(tmp_path, sums)


@pytest.mark.parametrize(
    "sums",
    [
        "",
        "0" * 64,
        "bad artifact.whl",
        f"{'0' * 64} ../artifact.whl",
        f"{'0' * 64} .hidden.whl",
        f"{'0' * 64} artifact.whl {'1' * 64} artifact.whl",
    ],
)
def test_asset_selection_requires_unique_safe_names_and_hashes(sums: str) -> None:
    with pytest.raises(ValueError):
        assets.selected(sums)
