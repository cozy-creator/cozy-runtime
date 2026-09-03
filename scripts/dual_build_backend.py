"""PEP 517 backend: hatchling's pure wheel, unless the build asks for the kernels.

`COZY_RUNTIME_BUILD_KERNELS=1` is a BUILD-artifact selector (which wheel file this
invocation produces), not runtime configuration: it swaps the delegate to
scikit-build-core, which compiles `native/` via the root CMakeLists.txt into
`cozy_runtime/_kernels/_C.abi3.so` and tags the wheel cp312-abi3. Every other path —
sdist, editable, and the default wheel — is hatchling exactly as before, with no
CMake, nvcc, or nanobind anywhere near the build.
"""

from __future__ import annotations

import os
from typing import Any

from hatchling import build as _hatchling

_KERNEL_REQUIRES = [
    "scikit-build-core==1.0.3",
    "nanobind==3.0.1",
    "cmake>=3.26",
    "ninja>=1.11",
]


def _kernels_requested() -> bool:
    return os.environ.get("COZY_RUNTIME_BUILD_KERNELS") == "1"


def get_requires_for_build_wheel(config_settings: dict[str, Any] | None = None) -> list[str]:
    if _kernels_requested():
        return list(_KERNEL_REQUIRES)
    return _hatchling.get_requires_for_build_wheel(config_settings)


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    if _kernels_requested():
        from scikit_build_core import build as _skbuild

        return _skbuild.build_wheel(wheel_directory, config_settings, metadata_directory=None)
    return _hatchling.build_wheel(wheel_directory, config_settings, metadata_directory)


def get_requires_for_build_sdist(config_settings: dict[str, Any] | None = None) -> list[str]:
    return _hatchling.get_requires_for_build_sdist(config_settings)


def build_sdist(sdist_directory: str, config_settings: dict[str, Any] | None = None) -> str:
    return _hatchling.build_sdist(sdist_directory, config_settings)


def get_requires_for_build_editable(config_settings: dict[str, Any] | None = None) -> list[str]:
    return _hatchling.get_requires_for_build_editable(config_settings)


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    return _hatchling.build_editable(wheel_directory, config_settings, metadata_directory)
