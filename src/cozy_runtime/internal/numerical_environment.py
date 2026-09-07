"""Measured numerical platform identity, independent of a caller's code and custody."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Mapping

from cozy_runtime import canonical_json
from cozy_runtime.internal.base_observation import Observation, protected_distribution
from cozy_runtime.internal.hostfacts import HostFacts


class NumericalEnvironmentRefusal(Exception):
    pass


def distribution_records(base: Observation) -> dict[str, dict[str, str]]:
    """Installed wheels bind code through their exact RECORD, including native libraries.

    Private platform updates install new wheels before the worker starts. Editable
    platform installations have no immutable RECORD authority and cannot claim reuse.
    Caller libraries belong to the captured package closure, outside this identity.
    """
    records: dict[str, dict[str, str]] = {}
    for name, version in base.distributions:
        if not protected_distribution(name):
            continue
        distribution = importlib.metadata.distribution(name)
        record = distribution.read_text("RECORD")
        source = distribution.read_text("direct_url.json")
        if (
            distribution.version != version
            or not record
            or len(record.encode()) > 32 << 20
            or "__editable__" in record
            or (source and json.loads(source).get("dir_info", {}).get("editable"))
        ):
            raise NumericalEnvironmentRefusal("numerical platform has no immutable installed wheel")
        records[name] = {
            "version": version,
            "record_digest": hashlib.sha256(record.encode()).hexdigest(),
        }
    if "cozy-runtime" not in records:
        raise NumericalEnvironmentRefusal("Runtime has no installed wheel identity")
    return records


def fingerprint(
    base: Observation,
    device: HostFacts,
    *,
    threads: int,
    alloc_conf: str,
    inherited: Mapping[str, str],
) -> bytes:
    """Hash installed numerical code, measured hardware behavior, and imposed settings."""
    if device.backend == "cuda" and (device.gpu_sm <= 0 or not device.driver_version):
        raise NumericalEnvironmentRefusal("numerical device identity is unavailable")
    # Runtime's sealed child environment normally erases these. Include any
    # explicitly inherited numerical knobs, excluding access and machine identity.
    numerical = {
        key: value
        for key, value in inherited.items()
        if key.startswith(("OMP_", "MKL_", "OPENBLAS_"))
        or key
        in {
            "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE",
            "PYTHONHASHSEED",
            "CUDA_LAUNCH_BLOCKING",
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_ALLOC_CONF",
        }
    }
    facts = {
        "interpreter": {**base.interpreter_identity, "version": platform.python_version()},
        "distributions": distribution_records(base),
        "torch_build": base.accelerator_build,
        "cuda_runtime": base.cuda_version,
        "device": {
            "backend": device.backend or "cpu",
            "architecture": device.gpu_sm,
            "driver": device.driver_version,
        },
        "settings": {"threads": threads, "alloc_conf": alloc_conf, "environment": numerical},
    }
    return hashlib.sha256(
        b"cozy.runtime.numerical-environment/1\0" + canonical_json.encode(facts)
    ).digest()
