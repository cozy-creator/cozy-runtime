"""Assemble the package constructor document from inline CozyTensors configs."""

from __future__ import annotations

import json
from collections.abc import Mapping

from cozy_runtime.internal import canonical, execution_contract


class ModelConfigRefusal(Exception):
    pass


#: Inline header config names that are NOT construction configs (cr-121). `execution` is the
#: lane's execution contract (cr-109, attention-quantization.md §2): a closed msgspec schema
#: with its own reader, `execution_contract.read`, over the header's own bytes. It never joins
#: the constructor document and never reaches a factory — where its binding `weights.route`
#: would read as a source carrier and the §1.1 ban would refuse the whole lane.
RESERVED = frozenset({execution_contract.CONFIG_NAME})


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
        raw = value
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelConfigRefusal(f"config {name!r} is not JSON") from exc
        if not isinstance(document, dict):
            raise ModelConfigRefusal(f"config {name!r} is not an object")
        try:
            normalized = canonical.write(document)
        except canonical.CanonicalError as exc:
            raise ModelConfigRefusal(f"config {name!r} is not canonical JSON: {exc}") from exc
        if normalized != raw:
            raise ModelConfigRefusal(f"config {name!r} is not canonical JSON")
        decoded[name] = document
        held.append(raw)
    if not held:
        raise ModelConfigRefusal("the CozyTensors header declares no inline construction configs")
    if len(held) == 1:
        return held[0]
    return canonical.write(decoded)
