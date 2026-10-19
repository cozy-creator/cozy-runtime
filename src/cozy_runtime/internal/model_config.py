"""Assemble the package constructor document from a checkpoint's component configs."""

from __future__ import annotations

from collections.abc import Mapping

from cozy_runtime.internal import canonical


class ModelConfigRefusal(Exception):
    def __init__(self, message: str, code: str = "") -> None:
        super().__init__(message)
        self.code = code


#: The asset holding an adapted component's prepared adapter graph (`lora_contract`); the
#: Runtime's, never a construction config.
ADAPTERS = "model_adapters.json"


def construction_config(configs: Mapping[str, Mapping[str, object]]) -> bytes:
    """One exact config document: a single component's, or one mapping of them by name."""
    held: dict[str, bytes] = {}
    for name, document in sorted(configs.items()):
        try:
            held[name] = canonical.write(dict(document))  # any writer's spelling, one identity
        except canonical.CanonicalError as exc:
            raise ModelConfigRefusal(f"config {name!r} has no canonical JSON form: {exc}") from exc
    if not held:
        raise ModelConfigRefusal("the checkpoint carries no construction config")
    if len(held) == 1:
        return next(iter(held.values()))
    return canonical.write({name: dict(configs[name]) for name in held})
