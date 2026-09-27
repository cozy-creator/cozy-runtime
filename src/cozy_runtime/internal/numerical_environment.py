"""Cheap numerical execution policy, independent of SDK and package inventories.

Callee implementation declarations own Python helpers, data, and installed native
build identities. This salt carries only the execution device and imposed numeric
policy; it must never inspect distributions, RECORD files, or native binaries.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from cozy_runtime import canonical_json
from cozy_runtime.internal.hostfacts import HostFacts


class NumericalEnvironmentRefusal(Exception):
    pass


_NUMERICAL_SETTINGS = frozenset(
    {
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_SCHEDULE",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "MKL_CBWR",
        "OPENBLAS_NUM_THREADS",
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "PYTHONHASHSEED",
    }
)


def fingerprint(
    device: HostFacts,
    *,
    threads: int,
    inherited: Mapping[str, str],
) -> bytes:
    """Bind device architecture and numerical policy; callee declarations bind libraries.

    The host driver is not a numerical fact: kernels come from the environment's own
    libraries, so a provider's driver update must not orphan interrupted derived work.
    """
    if device.backend == "cuda" and device.gpu_sm <= 0:
        raise NumericalEnvironmentRefusal("numerical device identity is unavailable")
    facts = {
        "device": {
            "backend": device.backend or "cpu",
            "architecture": device.gpu_sm,
        },
        "settings": {
            "threads": threads,
            "environment": {
                key: value for key, value in inherited.items() if key in _NUMERICAL_SETTINGS
            },
        },
    }
    return hashlib.sha256(
        b"cozy.runtime.numerical-environment/4\0" + canonical_json.encode(facts)
    ).digest()
