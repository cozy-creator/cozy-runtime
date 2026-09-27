"""Assemble the package constructor document from inline CozyTensors configs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from cozy_runtime.internal import canonical


class ModelConfigRefusal(Exception):
    pass


#: Inline header config names that are NOT construction configs and are SKIPPED here (cr-121).
#: `execution` is the RETIRED lane execution contract (cr-124 deleted its reader). Checkpoints
#: minted while it existed still carry the config in their headers, and it must stay skipped
#: rather than deleted: folded into the constructor document its `weights` key reads as a
#: source carrier and the §1.1 ban refuses the whole lane, which is exactly the 42-second
#: prepare death cr-121 was opened for. Inert, not fatal.
RESERVED = frozenset({"execution"})


def tensor_dtypes(header: Mapping[str, Any]) -> dict[str, str]:
    """Project the already validated native header for weightless construction.

    Ephemeral constructor input only: no duplicated config field, geometry calculation,
    schema document, or dtype policy. TensorFS remains the metadata authority.
    """
    return {
        f"{component}.{key}": row["logical"]["logical_dtype"]
        for component, tensors in header["components"].items()
        for key, row in tensors.items()
    }


def construction_config(configs: object) -> bytes:
    """Return one exact config document; multiple component configs become one mapping."""
    if not isinstance(configs, Mapping):
        raise ModelConfigRefusal("the CozyTensors header declares no inline construction configs")
    decoded: dict[str, object] = {}
    held: list[bytes] = []
    for name, value in sorted(configs.items(), key=lambda row: str(row[0])):
        if not isinstance(name, str) or not name or not isinstance(value, bytes):
            raise ModelConfigRefusal("the CozyTensors header carries a malformed inline config")
        if name in RESERVED:
            continue
        try:
            document = json.loads(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelConfigRefusal(f"config {name!r} is not JSON") from exc
        if not isinstance(document, dict):
            raise ModelConfigRefusal(f"config {name!r} is not an object")
        try:
            normalized = canonical.write(document)  # any writer's spelling, one identity
        except canonical.CanonicalError as exc:
            raise ModelConfigRefusal(f"config {name!r} has no canonical JSON form: {exc}") from exc
        decoded[name] = document
        held.append(normalized)
    if not held:
        raise ModelConfigRefusal("the CozyTensors header declares no inline construction configs")
    if len(held) == 1:
        return held[0]
    return canonical.write(decoded)
