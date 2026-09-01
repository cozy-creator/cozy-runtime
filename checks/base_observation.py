"""Deterministic checks for actual protected package-base observation."""

from __future__ import annotations

import sys
from pathlib import Path

from cozy_runtime.internal import base_observation


def expect_torch_refusal(metadata: str, imported: str, cuda: str | None) -> None:
    try:
        base_observation._torch_identity(metadata, imported, cuda)
    except base_observation.ObservationRefusal:
        return
    raise AssertionError(f"invalid Torch identity was admitted: {metadata}/{imported}/{cuda}")


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
    assert "packaging" in observed.distribution_versions
    assert "python-stdlib" in observed.root_providers["json"]
    assert "cozy-runtime" in observed.root_providers["cozy_runtime"]
    assert "msgspec" in observed.root_providers["msgspec"]

    # Exact official Torch spellings exercised by Cozy's CPU wheel and CUDA base image.
    assert base_observation._torch_identity("2.13.0+cpu", "2.13.0+cpu", None) == (
        "none",
        "cpu",
        "2.13.0",
        "none",
    )
    assert base_observation._torch_identity("2.13.0", "2.13.0+cu130", "13.0") == (
        "cuda",
        "cu130",
        "2.13.0",
        "13.0",
    )
    assert base_observation._torch_identity("2.13.0+cu130", "2.13.0+cu130", "13.0") == (
        "cuda",
        "cu130",
        "2.13.0",
        "13.0",
    )

    expect_torch_refusal("2.13.0", "2.12.0+cu130", "13.0")
    expect_torch_refusal("2.13.0", "2.13.0+cu126", "13.0")
    expect_torch_refusal("2.13.0", "2.13.0+cpu", "13.0")
    expect_torch_refusal("2.13.0", "2.13.0+cu130", None)
    expect_torch_refusal("2.13.0+cpu", "2.13.0+cu130", "13.0")
    expect_torch_refusal("2.13.0", "2.13.0+cu130", "13")


if __name__ == "__main__":
    main()
