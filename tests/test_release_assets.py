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


def release_files(root: Path, version: str = "1.2.3") -> dict[str, str]:
    names = [
        f"cozy_runtime-{version}-py3-none-any.whl",
        f"cozy_runtime-{version}-cp312-abi3-manylinux_2_28_x86_64.whl",
        f"cozy_runtime-{version}.tar.gz",
    ]
    files = {}
    for name in names:
        path = root / name
        path.write_bytes(b"artifact bytes, not executable test code")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def sums_for(files: dict[str, str]) -> str:
    return "\n".join(f"{digest}  {name}" for name, digest in sorted(files.items()))


def test_approved_bytes_and_only_visible_release_files(tmp_path: Path) -> None:
    files = release_files(tmp_path)
    sums = sums_for(files)
    (tmp_path / ".gitignore").write_text("*\n")
    assert assets.verify(tmp_path, sums, "1.2.3") == sums
    (tmp_path / next(iter(files))).write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="digest differs"):
        assets.verify(tmp_path, sums, "1.2.3")


@pytest.mark.parametrize("change", ["missing", "extra", "symlink", "directory"])
def test_missing_extra_or_nonordinary_artifacts_refuse(tmp_path: Path, change: str) -> None:
    files = release_files(tmp_path)
    sums = sums_for(files)
    path = tmp_path / next(iter(files))
    if change == "extra":
        (tmp_path / "extra.whl").write_bytes(b"extra")
    else:
        path.unlink()
        if change == "symlink":
            target = tmp_path / ".target"
            target.write_bytes(b"artifact bytes, not executable test code")
            path.symlink_to(target)
        elif change == "directory":
            path.mkdir()
    with pytest.raises(ValueError, match=r"files differ|not an ordinary file"):
        assets.verify(tmp_path, sums, "1.2.3")


def test_valid_hashes_of_another_version_cannot_publish_under_selected_tag(tmp_path: Path) -> None:
    files = release_files(tmp_path, "1.2.4")
    with pytest.raises(ValueError, match=r"not the complete 1\.2\.3 release"):
        assets.verify(tmp_path, sums_for(files), "1.2.3")


@pytest.mark.parametrize(
    "omit", ["py3-none-any.whl", "cp312-abi3-manylinux_2_28_x86_64.whl", "tar.gz"]
)
def test_hash_approval_cannot_omit_a_required_distribution(tmp_path: Path, omit: str) -> None:
    files = release_files(tmp_path)
    missing = next(name for name in files if name.endswith(omit))
    (tmp_path / missing).unlink()
    del files[missing]
    with pytest.raises(ValueError, match=r"not the complete 1\.2\.3 release"):
        assets.verify(tmp_path, sums_for(files), "1.2.3")


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
