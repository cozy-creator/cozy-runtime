"""The prepare cache root is not the install root."""

from __future__ import annotations

from cozy_runtime.internal import host_paths


def test_the_cache_root_is_not_the_install_root() -> None:
    """`artifact_cache` holds content-addressed CONTROL DOCUMENTS; the install root holds
    package environments. Collapsing them would scatter digest-named files among installs."""

    layout = host_paths.Layout()
    assert layout.prepare_cache != layout.install_root
    assert layout.install_root not in layout.prepare_cache.parents
