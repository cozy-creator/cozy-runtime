"""Deterministic checks for actual protected package-base observation."""

from __future__ import annotations

import sys
from pathlib import Path

from cozy_runtime.internal import base_observation


def main() -> None:
    observed = base_observation.current()
    assert observed is base_observation.current()
    assert observed == base_observation.from_python(Path(sys.executable))
    assert observed.python_abi == "cp312"
    assert observed.os_arch == "linux/amd64"
    assert observed.libc.startswith("glibc2.")
    assert observed.uv_version
    assert "cozy-runtime" in observed.distribution_versions
    assert "msgspec" in observed.distribution_versions
    assert "python-stdlib" in observed.root_providers["json"]
    assert "cozy-runtime" in observed.root_providers["cozy_runtime"]
    assert "msgspec" in observed.root_providers["msgspec"]


if __name__ == "__main__":
    main()
