"""Bind a prepared LoRA view (TensorD composes it: `adapter.rs`) to a construction.

The view's base weights are `<path>.base_layer.*` and its factors `<path>.lora_{A,B}.<adapter>`
(a DoRA's magnitude `<path>.lora_magnitude_vector.<adapter>`); its `model_adapters.json` graph
names, by stored names, the PEFT layers (linear and 2-D convolution) this module builds before the
ordinary census, fill and residency own them.

A DoRA term is PEFT's, `D = m * V / ||V||` per output channel with `V = W + s * B @ A`
(s = alpha / r), at strength t: `W' = W + t * (D - W)` (`lora_bake.dora_weight`), the identity at 0
and PEFT's own weight at 1. The weight plane (`weights._Bakes`) computes it from the stored bytes as
they fill: baked, or for a per-step forward each row of `W` scaled by `1 + t * (m / ||V|| - 1)` and
`B` by `t * s * m / ||V||` (scale 1 here).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import msgspec

from cozy_runtime.author import Config as ModelConfig, Model
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._loader import Artifact
from cozy_runtime.author._loader import TENSORFS_DTYPES, TORCH_DTYPES, canonical_dtype
from cozy_runtime.internal.lora_contract import (
    CAPABILITY as CAPABILITY,
    FORMAT as FORMAT,
    GRAPH_CONFIG as GRAPH_CONFIG,
    Graph as Graph,
    Layer,
    read as read,
)


def _refuse(code: str, detail: str) -> CapabilityError:
    return CapabilityError(detail, code="adapter_" + code)


def composer(graph: Graph, magnitudes: Mapping[str, str]) -> Callable[[object], None]:
    """Construct PEFT layers before the ordinary census and fill.

    Imports happen before the caller enters its weightless construction substrate.
    PEFT owns the layer, its keys and scales; each update accumulates into the base output
    in place at the activation's precision, never copying the input to the factor dtype
    (#1077). No state is merged into the base, and this transform allocates no weight values.
    `magnitudes` is each DoRA magnitude's stored dtype, by its key.
    """
    if not graph.layers:
        return lambda _obj: None
    try:
        import torch
        from peft import LoraConfig
        from peft.tuners.lora.layer import Conv2d as PeftConv2d, Linear as PeftLinear
        from torch import nn

        from cozy_runtime.accel import lora
    except ImportError as exc:
        raise _refuse(
            "backend_unavailable",
            "prepared LoRA graphs require cozy-runtime[adapters] in the package dependencies; "
            "add the extra and relock the package",
        ) from exc

    # Private construction recipe only; rank and alpha are set independently for each term.
    config = LoraConfig(init_lora_weights=False, inference_mode=True, lora_dropout=0.0)

    class _Magnitude(nn.Module):
        """A DoRA layer's magnitude, `m` `[out]`."""

        def __init__(self, out: int, dtype: torch.dtype) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out, dtype=dtype), requires_grad=False)

    class _Linear(PeftLinear):  # type: ignore[misc]
        #: the weight plane baked the update into the base weight's resident bytes
        _cozy_baked = False
        #: each DoRA adapter's (s, t): its factors' scale alpha / r and its strength
        _cozy_dora: dict[str, tuple[float, float]]

        def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            # Base hooks (Turbo's overlay) run inside `base_layer`.
            result = self.base_layer(x, *args, **kwargs).contiguous()
            if not self._cozy_baked:
                for name in self.active_adapters:
                    lora.accumulate(
                        x, result, self.lora_A[name].weight, self.lora_B[name].weight,
                        self.scaling[name],
                    )
            return result

    class _Conv2d(PeftConv2d):  # type: ignore[misc]
        _cozy_baked = False

        def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            # Extra memory: a rank-channel activation and one update.
            result = self.base_layer(x, *args, **kwargs)
            if self._cozy_baked:
                return result
            for name in self.active_adapters:
                down, up = self.lora_A[name], self.lora_B[name]
                low = up(down(x.to(down.weight.dtype)))
                result.add_(low.to(result.dtype), alpha=self.scaling[name])
            return result

    def apply(obj: object) -> None:
        members = getattr(obj, "components", None)
        if not isinstance(members, Mapping):
            raise _refuse("component", "model construction exposes no component mapping")
        groups: dict[tuple[str, str], list[Layer]] = {}
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
            if not isinstance(base, (nn.Linear, nn.Conv2d)):
                raise _refuse(
                    "target_type", f"{component}.{path} is neither a linear nor a 2-D convolution"
                )
            if id(base) in seen:
                raise _refuse("alias", f"{component}.{path} aliases another adapted layer")
            seen.add(id(base))
            if len({row.adapter for row in rows}) != len(rows):
                raise _refuse("graph", f"{component}.{path} repeats one adapter term")
            first, *others = rows
            kind = _Linear if isinstance(base, nn.Linear) else _Conv2d
            if kind is _Conv2d and any(row.dora for row in rows):
                raise _refuse(
                    "target_type", f"{component}.{path}: a DoRA adapts a linear, not a convolution"
                )
            layer = kind(base, first.adapter, config=config, r=first.rank, lora_alpha=first.alpha)
            layer._cozy_dora = {}
            for row in others:
                layer.update_layer(row.adapter, row.rank, lora_alpha=row.alpha, config=config)
            for row in rows:
                dtype = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}[
                    row.dtype
                ]
                layer.lora_A[row.adapter].to(dtype=dtype)
                layer.lora_B[row.adapter].to(dtype=dtype)
                layer.set_scale(row.adapter, row.strength)
                if row.dora:
                    key = f"{component}.{path}.lora_magnitude_vector.{row.adapter}.weight"
                    width = getattr(torch, TORCH_DTYPES[magnitudes[key]])
                    layer.lora_magnitude_vector[row.adapter] = _Magnitude(base.out_features, width)
                    layer._cozy_dora[row.adapter] = (row.alpha / row.rank, row.strength)
                    layer.scaling[row.adapter] = 1.0
            layer.set_adapter([row.adapter for row in rows], inference_mode=True)
            layer.eval()
            parent.set_submodule(name, layer)

    return apply


def bind(
    artifact: Artifact,
    model: type[Model[object]],
    data: bytes,
    binding: Callable[[str, str], str],
    *,
    prepared_snapshot: str = "",
) -> Artifact:
    """Bind one prepared view to a model's opt-in before weightless construction. `binding`
    is a component's stored name -> the name its module tree uses (`fill.Checkpoint.binding`):
    the graph's targets are stored names (SDXL's `text_model.` that `CLIPTextModel` drops)."""
    if not data:
        return artifact
    graph = read(data)
    layers = tuple(
        msgspec.structs.replace(
            row, target=binding(row.component, row.target + ".weight").removesuffix(".weight")
        )
        for row in graph.layers
    )
    graph = msgspec.structs.replace(graph, layers=layers)
    model.validate_adapter_stack(graph.adapters)
    given = artifact.config._tensor_dtypes
    dtypes = (
        dict(given)
        if given is not None
        else {key: spec.dtype for key, spec in artifact.tensor_schema.items()}
    )
    magnitudes: dict[str, str] = {}
    for row in graph.layers:
        prefix = row.component + "." + row.target
        for role in ("weight", "bias"):
            key = prefix + ".base_layer." + role
            if key in dtypes:
                dtypes[prefix + "." + role] = dtypes.pop(key)
        for role in ("A", "B"):
            dtypes.pop(prefix + f".lora_{role}.{row.adapter}.weight", None)
        if row.dora:
            key = prefix + f".lora_magnitude_vector.{row.adapter}.weight"
            if key not in dtypes:
                raise _refuse("graph", f"{key}: a DoRA layer the view stores no magnitude of")
            stored = str(dtypes.pop(key))
            magnitudes[key] = canonical_dtype(TENSORFS_DTYPES.get(stored, stored))
    return replace(
        artifact,
        snapshot=prepared_snapshot or artifact.snapshot,
        variant=prepared_snapshot or artifact.variant,
        _selection_snapshot=artifact._selection_snapshot or artifact.snapshot,
        config=ModelConfig(artifact.config.mapping(), tensor_dtypes=dtypes),
        _prepare=composer(graph, magnitudes),
        _adapters=graph.adapters,
    )
