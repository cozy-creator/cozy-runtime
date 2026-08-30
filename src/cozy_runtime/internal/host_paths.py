"""Fixed pod paths shared with worker-protocol and the base worker image build."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Final

_VECTOR = files("cozy_runtime.vectors").joinpath("runtime-worker-host-paths.json")
_EXPECTED = {
    "artifact_cache",
    "artifact_store",
    "boot_data",
    "boot_id",
    "bootstrap_state",
    "environment_root",
    "media_root",
    "readiness_envelope",
    "readiness_payload",
    "tls_certificate",
    "tls_private_key",
    "worker_control_address",
    "worker_state",
}


def _load() -> MappingProxyType[str, Path]:
    value = json.loads(_VECTOR.read_text())
    if not isinstance(value, dict) or set(value) != _EXPECTED:
        raise RuntimeError("runtime-worker-host-paths vector has the wrong field set")
    paths: dict[str, Path] = {}
    for name, raw in value.items():
        path = Path(raw) if isinstance(raw, str) else Path()
        if not isinstance(raw, str) or not path.is_absolute() or str(path) != raw:
            raise RuntimeError(f"runtime-worker-host-paths.{name} is not absolute and normalized")
        paths[name] = path
    return MappingProxyType(paths)


PATHS: Final = _load()

ARTIFACT_CACHE: Final = PATHS["artifact_cache"]
ARTIFACT_STORE: Final = PATHS["artifact_store"]
BOOT_DATA: Final = PATHS["boot_data"]
BOOT_ID: Final = PATHS["boot_id"]
BOOTSTRAP_STATE: Final = PATHS["bootstrap_state"]
ENVIRONMENT_ROOT: Final = PATHS["environment_root"]
MEDIA_ROOT: Final = PATHS["media_root"]
READINESS_ENVELOPE: Final = PATHS["readiness_envelope"]
READINESS_PAYLOAD: Final = PATHS["readiness_payload"]
TLS_CERTIFICATE: Final = PATHS["tls_certificate"]
TLS_PRIVATE_KEY: Final = PATHS["tls_private_key"]
WORKER_CONTROL_ADDRESS: Final = PATHS["worker_control_address"]
WORKER_STATE: Final = PATHS["worker_state"]
