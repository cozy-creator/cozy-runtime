"""Prepare immutable LoRA component graphs using native TensorFS part grafts.

Base weights and factor payloads stay byte-identical. The prepared graph adds ordinary
PEFT destinations, so the existing census, fill, residency and group admission own them.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import msgspec
import numpy as np
from tensorfs.derived import (
    Config,
    Derivation,
    Part,
    PartSource,
    SourceCapability,
    SourceInspection,
    Target,
    Tensor,
)

from cozy_runtime import canonical_json
from cozy_runtime.author import AdapterRef, Artifact, CapabilityError, Config as ModelConfig, Model
from cozy_runtime.internal.encoding import SPEC_PLAIN
from cozy_runtime.internal.lora_contract import (
    CAPABILITY as CAPABILITY,
    FORMAT as FORMAT,
    GRAPH_CONFIG as GRAPH_CONFIG,
    Graph as Graph,
    Linear,
    read as read,
)

_VALIDATION_BYTES = 1 << 20
_DTYPES = {"f16": ("<f2", 2), "bf16": ("<u2", 2), "f32": ("<f4", 4)}
_SUFFIXES = {".lora_A.weight": "a", ".lora_B.weight": "b", ".alpha": "alpha"}


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


def _component(base: SourceInspection, adapter: SourceInspection, selected: Selection) -> str:
    if selected.component:
        return selected.component
    factors = adapter.components.get(selected.source_component, {})
    pairs = {
        key.removesuffix(".lora_A.weight"): tensor
        for key, tensor in factors.items()
        if key.endswith(".lora_A.weight")
    }
    matches: list[str] = []
    for name, tensors in base.components.items():
        if not pairs:
            break
        for path, a in pairs.items():
            b, weight = factors.get(path + ".lora_B.weight"), tensors.get(path + ".weight")
            if (
                b is None
                or weight is None
                or len(a.shape) != 2
                or len(b.shape) != 2
                or (b.shape[0], a.shape[1]) != tuple(weight.shape)
            ):
                break
        else:
            matches.append(name)
    if len(matches) != 1:
        raise _refuse(
            "component",
            "select one adapter component; compatible choices: " + (", ".join(matches) or "none"),
        )
    return matches[0]


def prepare(base: SourceCapability, selections: Sequence[Selection]) -> Prepared:
    """Produce a zero-copy native derivation, refusing unsupported keys before writing.

    A/B use the canonical PEFT orientation: A=(rank,input), B=(output,rank), with
    alpha/rank and the caller's independent strength applied by PEFT exactly once.
    """
    structure = base.inspect()
    if GRAPH_CONFIG in structure.configs:
        raise _refuse("nested", "select the original base and one complete ordered adapter stack")
    sources = {"base": structure.source}
    additions: dict[str, dict[str, Tensor]] = {name: {} for name in structure.components}
    drops: dict[str, set[str]] = {name: set() for name in structure.components}
    layers: list[Linear] = []
    references: list[AdapterRef] = []
    for index, selection in enumerate(selections):
        if not math.isfinite(selection.strength):
            raise _refuse("scale", "adapter strength must be finite")
        inspection = selection.source.inspect()
        component = _component(structure, inspection, selection)
        if component not in structure.components:
            raise _refuse("component", f"base has no component {component!r}")
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
        references.append(
            AdapterRef(
                selection.source.manifest,
                selection.strength,
                "lora",
                component,
                selection.source_component,
                "",
            )
        )
        source_name = f"adapter_{index}"
        base_tensors = structure.components[component]
        for path, pair in sorted(pairs.items()):
            if not {"a", "b"} <= pair.keys():
                raise _refuse("pair", f"{component}.{path}: incomplete A/B pair")
            a_key, a = pair["a"]
            b_key, b = pair["b"]
            weight = base_tensors.get(path + ".weight")
            if weight is not None and len(weight.shape) != 2:
                raise _refuse(
                    "target_type",
                    f"{component}.{path}: only ordinary linear LoRA targets are supported",
                )
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
    configs[GRAPH_CONFIG] = Config("add")
    order = [
        (component, key)
        for component, tensors in structure.components.items()
        for key in (
            *[name for name in tensors if name not in drops[component]],
            *additions[component],
        )
    ]
    return Prepared(
        Derivation(sources, targets, configs, order),
        Graph(FORMAT, tuple(layers), tuple(references)),
    )


def composer(graph: Graph) -> Callable[[object], None]:
    """Construct PEFT layers before the ordinary census and fill.

    Imports happen before the caller enters its weightless construction substrate.
    PEFT owns the layer, its keys and scales; each update accumulates into the base output
    in place at the activation's precision, never copying the input to the factor dtype
    (#1077). No state is merged into the base, and this transform allocates no weight values.
    """
    if not graph.layers:
        return lambda _obj: None
    try:
        import torch
        from peft import LoraConfig
        from peft.tuners.lora.layer import Linear as PeftLinear
        from torch import nn
    except ImportError as exc:
        raise _refuse(
            "backend_unavailable",
            "prepared LoRA graphs require cozy-runtime[adapters] in the package dependencies; "
            "add the extra and relock the package",
        ) from exc

    # Private construction recipe only; rank and alpha are set independently for each term.
    config = LoraConfig(init_lora_weights=False, inference_mode=True, lora_dropout=0.0)

    class _Linear(PeftLinear):  # type: ignore[misc]
        def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            # Base hooks (Turbo's overlay) run inside `base_layer`. Extra memory: [rows, rank].
            result = self.base_layer(x, *args, **kwargs).contiguous()
            flat, rows = result.view(-1, result.shape[-1]), x.reshape(-1, x.shape[-1])
            for name in self.active_adapters:
                low = nn.functional.linear(rows, self.lora_A[name].weight.to(rows.dtype))
                flat.addmm_(
                    low.to(flat.dtype),
                    self.lora_B[name].weight.to(flat.dtype).t(),
                    alpha=self.scaling[name],
                )
            return result

    def apply(obj: object) -> None:
        members = getattr(obj, "components", None)
        if not isinstance(members, Mapping):
            raise _refuse("component", "model construction exposes no component mapping")
        groups: dict[tuple[str, str], list[Linear]] = {}
        for row in graph.layers:
            groups.setdefault((row.component, row.target), []).append(row)
        seen: set[int] = set()
        for (component, path), rows in groups.items():
            root = members.get(component)
            if not isinstance(root, nn.Module):
                raise _refuse("component", f"{component!r} is not a constructed module")
            parent_path, _, name = path.rpartition(".")
            try:
                parent = root.get_submodule(parent_path)
                base = parent.get_submodule(name)
            except AttributeError as exc:
                raise _refuse(
                    "target", f"constructed model has no target {component}.{path}"
                ) from exc
            if not isinstance(base, nn.Linear):
                raise _refuse("target_type", f"{component}.{path} is not an ordinary linear layer")
            if id(base) in seen:
                raise _refuse("alias", f"{component}.{path} aliases another adapted layer")
            seen.add(id(base))
            if len({row.adapter for row in rows}) != len(rows):
                raise _refuse("graph", f"{component}.{path} repeats one adapter term")
            first, *others = rows
            layer = _Linear(
                base, first.adapter, config=config, r=first.rank, lora_alpha=first.alpha
            )
            for row in others:
                layer.update_layer(row.adapter, row.rank, lora_alpha=row.alpha, config=config)
            for row in rows:
                dtype = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[
                    row.dtype
                ]
                layer.lora_A[row.adapter].to(dtype=dtype)
                layer.lora_B[row.adapter].to(dtype=dtype)
                layer.set_scale(row.adapter, row.strength)
            layer.set_adapter([row.adapter for row in rows], inference_mode=True)
            layer.eval()
            parent.set_submodule(name, layer)

    return apply


def bind(
    artifact: Artifact,
    model: type[Model[object]],
    data: bytes,
    *,
    prepared_snapshot: str = "",
) -> Artifact:
    """Bind one prepared view to a model's opt-in before weightless construction."""
    if not data:
        return artifact
    graph = read(data)
    model.validate_adapter_stack(graph.adapters)
    given = artifact.config._tensor_dtypes
    dtypes = (
        dict(given)
        if given is not None
        else {key: spec.dtype for key, spec in artifact.tensor_schema.items()}
    )
    for row in graph.layers:
        prefix = row.component + "." + row.target
        for role in ("weight", "bias"):
            key = prefix + ".base_layer." + role
            if key in dtypes:
                dtypes[prefix + "." + role] = dtypes.pop(key)
        for role in ("A", "B"):
            dtypes.pop(prefix + f".lora_{role}.{row.adapter}.weight", None)
    return replace(
        artifact,
        snapshot=prepared_snapshot or artifact.snapshot,
        variant=prepared_snapshot or artifact.variant,
        _selection_snapshot=artifact._selection_snapshot or artifact.snapshot,
        config=ModelConfig(artifact.config.mapping(), tensor_dtypes=dtypes),
        _prepare=composer(graph),
        _adapters=graph.adapters,
    )
