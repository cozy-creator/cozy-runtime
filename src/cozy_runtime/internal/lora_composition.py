"""Bind a prepared LoRA view (TensorD composes it: `adapter.rs`) to a construction.

The view's base weights are `<path>.base_layer.*` and its factors `<path>.lora_{A,B}.<adapter>`
(a DoRA's magnitude `<path>.lora_magnitude_vector.<adapter>`); its `model_adapters.json` graph
names, by stored names, the PEFT layers (linear and 2-D convolution) this module builds before the
ordinary census, fill and residency own them.

A LoKr layer (LyCORIS' Kronecker adapter) adds `s * x @ kron(W1, W2).T` each step, never the
product: `x` as `[.., b, d]`, then `W1 · x · W2ᵀ` (`[.., a, c]`), its target's rows. It isn't baked
(the update is dense, and costs about `1/a` of its linear's GEMM).

A DoRA term is PEFT's, `D = m * V / ||V||` per output channel with `V = W + s * B @ A`
(s = alpha / r), at strength t: `W' = W + t * (D - W)` (`lora_bake.dora_weight`), the identity at 0
and PEFT's own weight at 1. The weight plane (`weights._Bakes`) computes it from the stored bytes as
they fill: baked, or for a per-step forward each row of `W` scaled by `1 + t * (m / ||V|| - 1)` and
`B` by `t * s * m / ||V||` (scale 1 here).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import cast

import msgspec

from cozy_runtime.author import Config as ModelConfig
from cozy_runtime.author import Model
from cozy_runtime.author._errors import CapabilityError
from cozy_runtime.author._loader import TENSORFS_DTYPES, TORCH_DTYPES, Artifact, canonical_dtype
from cozy_runtime.internal.lora_contract import (
    CAPABILITY as CAPABILITY,
)
from cozy_runtime.internal.lora_contract import (
    FORMAT as FORMAT,
)
from cozy_runtime.internal.lora_contract import (
    GRAPH_CONFIG as GRAPH_CONFIG,
)
from cozy_runtime.internal.lora_contract import (
    Graph as Graph,
)
from cozy_runtime.internal.lora_contract import (
    KRON_ROLES,
    Layer,
)
from cozy_runtime.internal.lora_contract import (
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
        from peft.tuners.lora.layer import Conv2d as PeftConv2d
        from peft.tuners.lora.layer import Linear as PeftLinear
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

    class _Linear(PeftLinear):
        #: the weight plane baked the update into the base weight's resident bytes
        _cozy_baked = False
        #: each DoRA adapter's (s, t): its factors' scale alpha / r and its strength
        _cozy_dora: dict[str, tuple[float, float]]

        def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            # Base hooks (Turbo's overlay) run inside `base_layer`.
            result: torch.Tensor = self.base_layer(x, *args, **kwargs).contiguous()
            if not self._cozy_baked:
                for name in self.active_adapters:
                    # peft keeps each factor of a linear as an nn.Linear in an nn.ModuleDict.
                    down = cast("nn.Linear", self.lora_A[name])
                    up = cast("nn.Linear", self.lora_B[name])
                    lora.accumulate(x, result, down.weight, up.weight, self.scaling[name])
            return result

    _DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}

    class _Factor(nn.Module):
        """One LoKr factor, `[rows, cols]`."""

        def __init__(self, shape: tuple[int, int], dtype: torch.dtype) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)

    class _Kron(nn.Module):
        """A linear with LoKr adapters: `base_layer(x)` plus each adapter's scale times
        `x @ kron(W1, W2).T` (its `rows`), W1 and W2 each `lokr_<role>[adapter]`, whole or a
        pair."""

        #: what `attention._base` unwraps to read the base's output dtype
        _cozy_kron = True

        def __init__(self, base: nn.Linear, rows: list[Layer]) -> None:
            super().__init__()
            self.base_layer = base
            self.in_features, self.out_features = base.in_features, base.out_features
            for role in KRON_ROLES:
                self.add_module(f"lokr_{role}", nn.ModuleDict())
            #: each adapter's scale (alpha / rank * strength) and rows
            self.terms: dict[str, tuple[float, tuple[tuple[int, int], ...]]] = {}
            for row in rows:
                kron = row.kron
                assert kron is not None  # every row of a _Kron is a LoKr's
                dtype = _DTYPES[row.dtype]
                for role, shape in kron.factors.items():
                    factors = cast("nn.ModuleDict", getattr(self, f"lokr_{role}"))
                    factors[row.adapter] = _Factor(shape, dtype)
                self.terms[row.adapter] = (row.alpha / row.rank * row.strength, kron.rows)

        def _factor(self, w: str, adapter: str) -> torch.Tensor:
            whole = cast("nn.ModuleDict", getattr(self, f"lokr_{w}"))
            if adapter in whole:
                return cast("torch.Tensor", whole[adapter].weight)
            pair = (cast("nn.ModuleDict", getattr(self, f"lokr_{w}_{p}"))[adapter] for p in "ab")
            a, b = (cast("torch.Tensor", factor.weight) for factor in pair)
            return a @ b

        def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            result: torch.Tensor = self.base_layer(x, *args, **kwargs)
            for adapter, (scale, rows) in self.terms.items():
                w1 = self._factor("w1", adapter).to(x.dtype)
                w2 = self._factor("w2", adapter).to(x.dtype)
                (a, b), (c, d) = w1.shape, w2.shape
                # x @ kron(W1, W2).T: row p*c + q of the product is W1[p, :] (x) W2[q, :].
                y = torch.matmul(w1, x.reshape(-1, b, d) @ w2.t()).reshape(*x.shape[:-1], a * c)
                if rows:
                    y = torch.cat([y[..., start:stop] for start, stop in rows], -1)
                result = result + (y * scale).to(result.dtype)
            return result

    class _Conv2d(PeftConv2d):
        _cozy_baked = False

        def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            # Extra memory: a rank-channel activation and one update.
            result: torch.Tensor = self.base_layer(x, *args, **kwargs)
            if self._cozy_baked:
                return result
            for name in self.active_adapters:
                # peft keeps each factor of a convolution as an nn.Conv2d in an nn.ModuleDict.
                down = cast("nn.Conv2d", self.lora_A[name])
                up = cast("nn.Conv2d", self.lora_B[name])
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
            kron = [row.kron is not None for row in rows]
            if any(kron) and not all(kron):
                raise _refuse("graph", f"{component}.{path}: LoRA and LoKr adapters on one layer")
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
            if all(kron):
                if not isinstance(base, nn.Linear):
                    why = f"{component}.{path}: a LoKr adapts a linear, not a convolution"
                    raise _refuse("target_type", why)
                parent.set_submodule(name, _Kron(base, rows))
                continue
            first, *others = rows
            kind = _Linear if isinstance(base, nn.Linear) else _Conv2d
            if kind is _Conv2d and any(row.dora for row in rows):
                raise _refuse(
                    "target_type", f"{component}.{path}: a DoRA adapts a linear, not a convolution"
                )
            # peft annotates `lora_alpha` int; it scales by alpha / r, and a float alpha is ours.
            alpha = first.alpha
            layer = kind(base, first.adapter, config=config, r=first.rank, lora_alpha=alpha)  # type: ignore[arg-type]
            layer._cozy_dora = {}
            for row in others:
                layer.update_layer(row.adapter, row.rank, lora_alpha=row.alpha, config=config)  # type: ignore[arg-type]
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
                    # A DoRA adapts a linear: a convolution's was refused above.
                    out = cast("nn.Linear", base).out_features
                    layer.lora_magnitude_vector[row.adapter] = _Magnitude(out, width)
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
    stated = artifact.config._tensor_shapes
    shapes = (
        dict(stated)
        if stated is not None
        else {key: spec.shape for key, spec in artifact.tensor_schema.items()}
    )
    magnitudes: dict[str, str] = {}
    for row in graph.layers:
        prefix = row.component + "." + row.target
        for role in ("weight", "bias"):
            key = prefix + ".base_layer." + role
            if key in dtypes:
                dtypes[prefix + "." + role] = dtypes.pop(key)
            if key in shapes:
                shapes[prefix + "." + role] = shapes.pop(key)
        for role in ("A", "B"):
            dtypes.pop(prefix + f".lora_{role}.{row.adapter}.weight", None)
            shapes.pop(prefix + f".lora_{role}.{row.adapter}.weight", None)
        for role in KRON_ROLES if row.kron is not None else ():
            dtypes.pop(prefix + f".lokr_{role}.{row.adapter}.weight", None)
            shapes.pop(prefix + f".lokr_{role}.{row.adapter}.weight", None)
        if row.dora:
            key = prefix + f".lora_magnitude_vector.{row.adapter}.weight"
            if key not in dtypes:
                raise _refuse("graph", f"{key}: a DoRA layer the view stores no magnitude of")
            stored = str(dtypes.pop(key))
            shapes.pop(key, None)
            magnitudes[key] = canonical_dtype(TENSORFS_DTYPES.get(stored, stored))
    return replace(
        artifact,
        snapshot=prepared_snapshot or artifact.snapshot,
        variant=prepared_snapshot or artifact.variant,
        _selection_snapshot=artifact._selection_snapshot or artifact.snapshot,
        config=ModelConfig(artifact.config.mapping(), tensor_dtypes=dtypes, tensor_shapes=shapes),
        _prepare=composer(graph, magnitudes),
        _adapters=graph.adapters,
    )
