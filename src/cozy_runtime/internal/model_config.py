"""Assemble the package constructor document from a checkpoint's config files."""

from __future__ import annotations

import json
from collections.abc import Mapping

from cozy_runtime.internal import canonical


class ModelConfigRefusal(Exception):
    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.code = code


#: The manifest file holding a checkpoint's prepared adapter graph (`lora_contract`); the
#: Runtime's, never a construction config.
ADAPTERS = "model_adapters.json"


def construction_config(configs: Mapping[str, bytes]) -> bytes:
    """One exact config document: a single component's, or one mapping of them by name."""
    decoded: dict[str, object] = {}
    held: list[bytes] = []
    for name, value in sorted(configs.items()):
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
        raise ModelConfigRefusal("the checkpoint carries no construction config")
    if len(held) == 1:
        return held[0]
    return canonical.write(decoded)
