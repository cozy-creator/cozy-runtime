"""Runtime operation rows go only to an executor whose installed Runtime supports them."""

from __future__ import annotations

from pathlib import Path

import pytest

from cozy_runtime.internal import builtin_operations


@pytest.mark.parametrize("version,expected", [("0.16.5", False), ("0.16.8", True)])
def test_builtin_inventory_tracks_selected_executor_not_parent_runtime(
    tmp_path: Path, version: str, expected: bool
) -> None:
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    metadata = environment / f"lib/python3.12/site-packages/cozy_runtime-{version}.dist-info"
    metadata.mkdir(parents=True)
    (metadata / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: cozy-runtime\nVersion: {version}\n"
    )
    assert builtin_operations.supports_environment(str(environment / "bin/python")) is expected
