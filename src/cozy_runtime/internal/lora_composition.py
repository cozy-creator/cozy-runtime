"""Prepare immutable LoRA component graphs using native TensorFS part grafts.

Base weights and factor payloads stay byte-identical. The prepared graph adds ordinary
PEFT destinations, so the existing census, fill, residency and group admission own them.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace

import msgspec
import numpy as np
from tensorfs.derived import Config, Derivation, Part, PartSource, SourceCapability, Target, Tensor

from cozy_runtime import canonical_json
from cozy_runtime.author import CapabilityError
from cozy_runtime.internal.encoding import SPEC_PLAIN

CONFIG = "model_adapters"
FORMAT = "cozy.model.lora/1"
_VALIDATION_BYTES = 1 << 20
_DTYPES = {"f16": ("<f2", 2), "bf16": ("<u2", 2), "f32": ("<f4", 4)}
_SUFFIXES = {".lora_A.weight": "a", ".lora_B.weight": "b", ".alpha": "alpha"}


class Linear(msgspec.Struct, frozen=True):
    """One ordinary PEFT layer's construction geometry and ordered scaling."""

    component: str
    target: str
    adapter: str
    rank: int
    alpha: float
    strength: float
    dtype: str


class Graph(msgspec.Struct, frozen=True):
    format: str
    layers: tuple[Linear, ...]


@dataclass(frozen=True, slots=True)
class Selection:
    source: SourceCapability
    component: str
    source_component: str = "adapter"
    strength: float = 1.0


@dataclass(frozen=True, slots=True)
class Prepared:
    declaration: Derivation
    graph: Graph

    @property
    def config(self) -> bytes:
        return canonical_json.encode(msgspec.to_builtins(self.graph))


def _refuse(code: str, detail: str) -> CapabilityError:
    return CapabilityError(detail, code="adapter_" + code)


def _graft(tensor: Tensor, source: str, component: str, key: str) -> Tensor:
    return replace(
        tensor,
        parts={
            role: Part(part.dtype, part.shape, PartSource(source, component, key, role))
            for role, part in tensor.parts.items()
        },
    )


def _validate(source: SourceCapability, component: str, key: str, tensor: Tensor) -> float | None:
    """Validate every factor, including zero-strength inputs, with bounded CPU scratch."""
    part = tensor.parts.get("value")
    dtype = tensor.logical_dtype
    if (
        dtype not in _DTYPES
        or tensor.encoding != SPEC_PLAIN
        or set(tensor.parts) != {"value"}
        or part is None
        or part.dtype != dtype
        or tuple(part.shape) != tuple(tensor.shape)
    ):
        raise _refuse("tensor", f"{component}.{key}: plain f16, bf16 or f32 factors required")
    numpy_dtype, width = _DTYPES[dtype]
    length = math.prod(tensor.shape) * width
    scalar: float | None = None
    for offset in range(0, length, _VALIDATION_BYTES):
        source.check()
        data = bytearray(min(_VALIDATION_BYTES, length - offset))
        source.read_part_into(component, key, "value", offset, data)
        values = np.frombuffer(data, dtype=numpy_dtype)
        finite = (values & 0x7F80) != 0x7F80 if dtype == "bf16" else np.isfinite(values)
        if not bool(np.all(finite)):
            raise _refuse("nonfinite", f"{component}.{key}: factor contains NaN or infinity")
        if not tensor.shape:
            scalar = (
                struct.unpack("<f", bytes(2) + data)[0] if dtype == "bf16" else float(values[0])
            )
    return scalar


def prepare(base: SourceCapability, selections: Sequence[Selection]) -> Prepared:
    """Produce a zero-copy native derivation, refusing unsupported keys before writing.

    A/B use the canonical PEFT orientation: A=(rank,input), B=(output,rank), with
    alpha/rank and the caller's independent strength applied by PEFT exactly once.
    """
    structure = base.inspect()
    if CONFIG in structure.configs:
        raise _refuse("nested", "select the original base and one complete ordered adapter stack")
    sources = {"base": structure.source}
    additions: dict[str, dict[str, Tensor]] = {name: {} for name in structure.components}
    drops: dict[str, set[str]] = {name: set() for name in structure.components}
    layers: list[Linear] = []
    for index, selection in enumerate(selections):
        if not math.isfinite(selection.strength):
            raise _refuse("scale", "adapter strength must be finite")
        if selection.component not in structure.components:
            raise _refuse("component", f"base has no component {selection.component!r}")
        inspection = selection.source.inspect()
        if set(inspection.components) != {selection.source_component}:
            raise _refuse(
                "components", "an adapter must contain only its selected factor component"
            )
        if set(inspection.configs) - {"normalization"}:
            raise _refuse("contract", "adapter introduces a separate inference configuration")
        tensors = inspection.components[selection.source_component]
        if not tensors:
            raise _refuse("empty", "adapter contains no factors")
        pairs: dict[str, dict[str, tuple[str, Tensor]]] = {}
        scalars: dict[str, float] = {}
        for key, tensor in tensors.items():
            suffix = next((ending for ending in _SUFFIXES if key.endswith(ending)), None)
            if suffix is None:
                raise _refuse("key", f"unsupported adapter tensor {key!r}")
            path, role = key[: -len(suffix)], _SUFFIXES[suffix]
            if not path:
                raise _refuse("target", "an adapter target must name a child linear layer")
            pairs.setdefault(path, {})[role] = key, tensor
            value = _validate(selection.source, selection.source_component, key, tensor)
            if role == "alpha":
                if value is None:
                    raise _refuse("alpha", f"{path}: alpha must be a scalar")
                scalars[path] = value
        source_name = f"adapter_{index}"
        component = selection.component
        base_tensors = structure.components[component]
        for path, pair in sorted(pairs.items()):
            if not {"a", "b"} <= pair.keys():
                raise _refuse("pair", f"{component}.{path}: incomplete A/B pair")
            a_key, a = pair["a"]
            b_key, b = pair["b"]
            weight = base_tensors.get(path + ".weight")
            if (
                weight is None
                or len(weight.shape) != 2
                or weight.logical_dtype not in _DTYPES
                or len(a.shape) != 2
                or len(b.shape) != 2
                or a.shape[0] <= 0
                or a.shape[0] != b.shape[1]
                or (b.shape[0], a.shape[1]) != tuple(weight.shape)
                or a.logical_dtype != b.logical_dtype
            ):
                raise _refuse(
                    "shape", f"{component}.{path}: factors do not fit the base projection"
                )
            rank = a.shape[0]
            alpha = scalars.get(path, float(rank))
            if not math.isfinite(alpha / rank * selection.strength):
                raise _refuse("scale", f"{component}.{path}: effective scale is not finite")
            if selection.strength == 0:
                continue
            sources[source_name] = inspection.source
            for key in (path + ".weight", path + ".bias"):
                if key in base_tensors:
                    drops[component].add(key)
                    name = path + ".base_layer." + key.rpartition(".")[2]
                    additions[component][name] = _graft(base_tensors[key], "base", component, key)
            for role, key, tensor in (("A", a_key, a), ("B", b_key, b)):
                name = f"{path}.lora_{role}.{source_name}.weight"
                additions[component][name] = _graft(
                    tensor, source_name, selection.source_component, key
                )
            layers.append(
                Linear(
                    component, path, source_name, rank, alpha, selection.strength, a.logical_dtype
                )
            )
    targets = {
        component: Target("base", component, tuple(sorted(drops[component])), additions[component])
        for component in structure.components
    }
    configs = {name: Config("copy", "base", name) for name in structure.configs}
    configs[CONFIG] = Config("add")
    order = [
        (component, key)
        for component, tensors in structure.components.items()
        for key in (
            *[name for name in tensors if name not in drops[component]],
            *additions[component],
        )
    ]
    return Prepared(Derivation(sources, targets, configs, order), Graph(FORMAT, tuple(layers)))
