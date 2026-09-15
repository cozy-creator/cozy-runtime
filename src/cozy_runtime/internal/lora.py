"""Request-owned adapters executed by upstream PEFT Linear layers.

This module owns validation, selection, memory admission and lifetime. PEFT owns
all LoRA arithmetic. It does not merge into or recover a mutable base weight.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType, MethodType
from typing import Any

from cozy_runtime.author._context import AdapterRef
from cozy_runtime.author._errors import InvalidRequest

_HOOK_REGISTRIES = (
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
)


class AdapterRefusal(InvalidRequest):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail, code=code)


@dataclass(frozen=True, slots=True)
class AdapterInput:
    ref: str
    component: str
    base_snapshot: str
    tensors: Mapping[str, Any]
    scale: float = 1.0
    kind: str = "lora"


@dataclass(slots=True)
class _Attempt:
    terms: Mapping[tuple[str, str], tuple[str, ...]]
    scales: Mapping[str, float]
    check_cancelled: Callable[[], None]
    check_capacity: Callable[[int], None] | None
    closed: bool = False
    calls: int = 0
    factor_bytes: int = 0
    converted_factors: int = 0


@dataclass(frozen=True, slots=True)
class _Gate:
    binding: LinearAdapters
    site: tuple[str, str]

    def __call__(self, module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.binding._before(self.site, module, args, kwargs)
        # Keep the ordinary nn.Linear keyword contract after PEFT changes the
        # Python signature to x. Authored pre-hooks have already observed input.
        if not args and "input" in kwargs and "x" not in kwargs:
            return args, {"x" if key == "input" else key: value for key, value in kwargs.items()}
        return None


@dataclass(frozen=True, slots=True)
class _Observed:
    binding: LinearAdapters
    site: tuple[str, str]

    def __call__(self, module: Any, args: tuple[Any, ...], output: Any) -> None:
        attempt = self.binding._active.get()
        if attempt is not None and attempt.terms.get(self.site):
            attempt.calls += 1


def _peft_forward(self: Any, *args: Any, **kwargs: Any) -> Any:
    """An entered PEFT execution owns any failure from its base/A/B operations.

    Pre-hook admission errors happen before this function; user post-hook errors
    happen after it. A native inner failure may retain factor operands in its
    traceback, so the generation must be retired rather than reused.
    """
    gate = next(hook for hook in self._forward_pre_hooks.values() if isinstance(hook, _Gate))
    attempt = gate.binding._active.get()
    try:
        return type(self).forward(self, *args, **kwargs)
    except BaseException:
        if attempt is not None and attempt.terms.get(gate.site):
            gate.binding._poisoned = True
        raise


def prequantized_input_allowed(module: Any) -> bool:
    """Adapted PEFT calls need their original floating input for A/B projections."""
    import torch

    if (
        torch.nn.modules.module._global_forward_hooks
        or torch.nn.modules.module._global_forward_pre_hooks
    ):
        return False
    gates = [hook for hook in module._forward_pre_hooks.values() if isinstance(hook, _Gate)]
    if not gates:
        # An upstream/custom adapter wrapper may have active residuals without
        # our request gate. Only an unwrapped base is safe by hook absence alone.
        return (
            not hasattr(module, "base_layer")
            and not module._forward_pre_hooks
            and not module._forward_hooks
        )
    if len(gates) != 1 or len(module._forward_pre_hooks) != 1:
        return False
    gate = gates[0]
    if any(not isinstance(hook, _Observed) for hook in module._forward_hooks.values()):
        return False
    attempt = gate.binding._active.get()
    if attempt is None:
        return not gate.binding._lock.locked()
    return not attempt.closed and not attempt.terms.get(gate.site)


class LinearAdapters:
    """Native PEFT wrappers with exclusive request selection and owned factors."""

    def __init__(
        self,
        components: Mapping[str, Any],
        snapshots: Mapping[str, str],
        targets: Mapping[str, Sequence[str]],
    ) -> None:
        import torch

        try:
            from peft import LoraConfig
            from peft.tuners.lora.layer import Linear
            from peft.tuners.lora.torchao import dispatch_torchao
        except ImportError as exc:
            raise AdapterRefusal(
                "adapter_backend_unavailable", "native LoRA requires the qualified PEFT deployment"
            ) from exc

        from cozy_runtime.internal.encoding.torchao_rowwise import _rowwise_forward

        self._torch = torch
        self._snapshots = dict(snapshots)
        self._sites: dict[tuple[str, str], tuple[int, int, Any]] = {}
        self._layers: dict[tuple[str, str], Any] = {}
        self._owners: dict[tuple[str, str], tuple[Any, str, Any]] = {}
        self._handles: list[Any] = []
        self._lock = threading.Lock()
        self._active: ContextVar[_Attempt | None] = ContextVar("linear_adapters", default=None)
        self._closed = self._poisoned = False
        self._config = LoraConfig(
            r=1, lora_alpha=1, init_lora_weights=False, inference_mode=True, lora_dropout=0.0
        )
        modules = []
        seen: set[int] = set()
        for component, paths in targets.items():
            if component not in components or not snapshots.get(component):
                raise AdapterRefusal("adapter_component", f"unknown component {component!r}")
            for path in paths:
                site = component, path
                if site in self._sites:
                    raise AdapterRefusal("adapter_target_duplicate", f"duplicate target {site}")
                if not path:
                    raise AdapterRefusal(
                        "adapter_target_unsupported", "root Linear requires an owner"
                    )
                try:
                    parent_path, _, attribute = path.rpartition(".")
                    parent = components[component].get_submodule(parent_path)
                    base = getattr(parent, attribute)
                except (AttributeError, KeyError) as exc:
                    raise AdapterRefusal(
                        "adapter_target_missing", f"unknown target {site}"
                    ) from exc
                if type(base) is not torch.nn.Linear:
                    raise AdapterRefusal(
                        "adapter_target_unsupported", f"{site}: standard Linear required"
                    )
                forward = base.__dict__.get("forward")
                if (
                    forward is not None
                    and getattr(forward, "__func__", None) is not _rowwise_forward
                ) or (
                    torch.nn.modules.module._global_forward_hooks
                    or torch.nn.modules.module._global_forward_pre_hooks
                ):
                    raise AdapterRefusal(
                        "adapter_forward_unsupported", f"{site}: authored/global forward"
                    )
                if id(base) in seen:
                    raise AdapterRefusal("adapter_target_alias", f"{site}: aliased linear target")
                if base.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise AdapterRefusal(
                        "adapter_compute_dtype", f"{site}: unsupported compute dtype"
                    )
                seen.add(id(base))
                self._sites[site] = base.in_features, base.out_features, base.weight.dtype
                modules.append((site, parent, attribute, base))
        try:
            for site, parent, attribute, base in modules:
                # The unused placeholder has no physical allocation and never runs.
                # Actual ranks and copied factors are owned by request(), below.
                with torch.device("meta"):
                    layer = dispatch_torchao(
                        base, "_inactive", config=self._config, r=1, lora_alpha=1
                    )
                    if layer is None:
                        layer = Linear(base, "_inactive", config=self._config, r=1, lora_alpha=1)
                layer.set_adapter([])
                layer.eval()
                # If a custom parent refuses replacement, the base's hooks have
                # not moved yet and its existing handles remain valid.
                setattr(parent, attribute, layer)
                self._layers[site] = layer
                self._owners[site] = parent, attribute, base
                for name in _HOOK_REGISTRIES:
                    registry = getattr(base, name)
                    setattr(layer, name, registry)
                    setattr(base, name, type(registry)())
                self._handles.append(
                    layer.register_forward_pre_hook(_Gate(self, site), with_kwargs=True)
                )
                self._handles.append(
                    layer.register_forward_hook(_Observed(self, site), prepend=True)
                )
                layer.forward = MethodType(_peft_forward, layer)
        except BaseException:
            self.close()
            raise

    def _before(
        self, site: tuple[str, str], layer: Any, args: tuple[Any, ...], kwargs: Any
    ) -> None:
        self.check_ready()
        if len(args) + len(kwargs) != 1 or len(args) > 1 or set(kwargs) - {"input", "x"}:
            raise AdapterRefusal(
                "adapter_arguments",
                "expected a single Linear input; PEFT selection/variant kwargs are unsupported",
            )
        attempt = self._active.get()
        if attempt is None:
            if self._lock.locked():
                raise AdapterRefusal(
                    "adapter_context_missing", "call escaped its active adapter context"
                )
            return
        if attempt.closed:
            raise AdapterRefusal("adapter_attempt_closed", "adapter context outlived its request")
        attempt.check_cancelled()
        names = attempt.terms.get(site, ())
        if tuple(layer.active_adapters) != names or layer.merged or layer.disable_adapters:
            raise AdapterRefusal(
                "adapter_state_changed", f"{site}: request adapter selection changed"
            )
        if not names:
            return
        if any(layer.scaling[name] != attempt.scales[name] for name in names):
            raise AdapterRefusal(
                "adapter_scale_changed", f"{site}: request adapter strengths changed"
            )
        if (
            self._torch.nn.modules.module._global_forward_hooks
            or self._torch.nn.modules.module._global_forward_pre_hooks
            or any(getattr(layer.base_layer, name) for name in _HOOK_REGISTRIES)
        ):
            raise AdapterRefusal(
                "adapter_forward_unsupported", f"{site}: inner/global hooks changed"
            )
        x = args[0] if args else kwargs.get("input", kwargs.get("x"))
        in_features, out_features, dtype = self._sites[site]
        if (
            type(x) is not self._torch.Tensor
            or x.ndim < 1
            or x.shape[-1] != in_features
            or x.dtype != dtype
        ):
            raise AdapterRefusal("adapter_input", f"{site}: original floating activation required")
        if next(reversed(layer._forward_pre_hooks.values()), None) != _Gate(self, site):
            raise AdapterRefusal(
                "adapter_forward_unsupported", f"{site}: input changed after admission"
            )
        if any(
            parameter.device != x.device
            for name in names
            for parameter in layer.lora_A[name].parameters()
        ):
            raise AdapterRefusal("adapter_device", f"{site}: factors are on another device")
        rank = max(layer.r[name] for name in names)
        # Before base forward, include its output plus PEFT's two additional
        # output temporaries and low-rank intermediate, for the actual row count.
        rows = x.numel() // in_features
        needed = rows * (3 * out_features + rank) * x.element_size()
        from cozy_runtime.internal.encoding.torchao_rowwise import _rowwise_forward

        if getattr(layer.base_layer.forward, "__func__", None) is _rowwise_forward:
            # The existing quantizer can fall back from Triton to eager FP32
            # division/clamp. Price that compatible path too, including a possible
            # floating reshape copy; GPU qualification may justify a tighter bound.
            base_workspace = rows * (
                in_features * (8 + x.element_size()) + out_features * x.element_size() + 12
            )
            needed = max(needed, base_workspace)
        if attempt.check_capacity is not None:
            attempt.check_capacity(needed)

    @contextmanager
    def request(
        self,
        adapters: Sequence[AdapterInput],
        *,
        device: Any,
        max_device_bytes: int,
        check_cancelled: Callable[[], None],
        check_capacity: Callable[[int], None] | None = None,
    ) -> Iterator[tuple[AdapterRef, ...]]:
        if adapters and check_capacity is None and self._torch.device(device).type == "cuda":
            raise AdapterRefusal("adapter_budget", "CUDA PEFT requires temporary memory admission")
        if not self._lock.acquire(blocking=False):
            raise AdapterRefusal("adapter_concurrent", "this model already has an active request")
        terms: dict[tuple[str, str], tuple[str, ...]] = {}
        scales: dict[str, float] = {}
        loaded: list[tuple[Any, str]] = []
        attempt = _Attempt(
            MappingProxyType(terms), MappingProxyType(scales), check_cancelled, check_capacity
        )
        token = None
        ca = cb = None
        try:
            self.check_ready()
            check_cancelled()
            rows, nbytes = self._validate(adapters, max_device_bytes)
            attempt.factor_bytes = nbytes
            if check_capacity is not None:
                check_capacity(nbytes)
            for index, (site, a, b, scale) in enumerate(rows):
                check_cancelled()
                layer = self._layers[site]
                dtype = self._sites[site][2]
                if a.dtype != dtype:
                    attempt.converted_factors += 2
                name = f"request_{index}"
                loaded.append((layer, name))
                with self._torch.device("meta"):
                    layer.update_layer(
                        name, r=a.shape[0], lora_alpha=a.shape[0], config=self._config
                    )
                # Convert during preparation, never inside each projection. Sources
                # and request Parameters are independent even in CPU qualification.
                ca, cb = (value.to(dtype=dtype, copy=True) for value in (a, b))
                if not all(bool(self._torch.isfinite(value).all()) for value in (ca, cb)):
                    raise AdapterRefusal("adapter_dtype_range", f"{site}: factors overflow {dtype}")
                for target, value in ((layer.lora_A[name], ca), (layer.lora_B[name], cb)):
                    target.load_state_dict(
                        {"weight": value.to(device=device, copy=True)}, assign=True
                    )
                layer.set_scale(name, scale)
                scales[name] = layer.scaling[name]
                terms[site] = (*terms.get(site, ()), name)
            for site, layer in self._layers.items():
                layer.set_adapter(list(terms.get(site, ())), inference_mode=True)
            token = self._active.set(attempt)
            with self._torch.no_grad():
                yield tuple(AdapterRef(a.ref, a.scale, a.kind) for a in adapters)
        finally:
            attempt.closed = True
            if token is not None:
                self._active.reset(token)
            try:
                if loaded and self._torch.device(device).type == "cuda":
                    self._torch.cuda.synchronize(device)
            except BaseException:
                self._poisoned = True
                raise
            finally:
                try:
                    for layer in self._layers.values():
                        layer.set_adapter([])
                    for layer, name in loaded:
                        for table in (layer.lora_A, layer.lora_B):
                            if name in table:
                                table[name].to_empty(device="meta")
                        layer.delete_adapter(name)
                except BaseException:
                    self._poisoned = True
                    raise
                finally:
                    terms.clear()
                    scales.clear()
                    loaded.clear()
                    ca = cb = None
                    self._lock.release()

    def observed_calls(self) -> int:
        attempt = self._active.get()
        return attempt.calls if attempt is not None else 0

    def preparation_facts(self) -> dict[str, int | bool]:
        """Small request facts; a compatibility cast is explicit degraded preparation."""
        attempt = self._active.get()
        converted = attempt.converted_factors if attempt is not None else 0
        return {
            "factor_device_bytes": attempt.factor_bytes if attempt is not None else 0,
            "factor_tensors_converted": converted,
            "degraded_mode": converted > 0,
        }

    def check_ready(self) -> None:
        if self._closed or self._poisoned:
            raise AdapterRefusal("adapter_binding_closed", "adapter binding is closed or poisoned")

    def close(self) -> None:
        if not self._lock.acquire(blocking=False):
            raise AdapterRefusal("adapter_concurrent", "cannot close an active adapter binding")
        try:
            if self._closed:
                return
            # Validate every owner before changing any tree or hook registry.
            for site, (parent, attribute, base) in self._owners.items():
                layer = self._layers[site]
                if getattr(parent, attribute, None) is not layer:
                    raise AdapterRefusal(
                        "adapter_target_changed", f"{site}: adapter target was replaced"
                    )
                if any(getattr(base, name) for name in _HOOK_REGISTRIES):
                    raise AdapterRefusal(
                        "adapter_hooks_changed", f"{site}: base hooks changed while owned"
                    )
            for site, (parent, attribute, base) in self._owners.items():
                layer = self._layers[site]
                for name in _HOOK_REGISTRIES:
                    setattr(base, name, getattr(layer, name))
                setattr(parent, attribute, base)
            # Handles still address the same registries, now restored on each
            # base. Retain the gates until every restoration has succeeded.
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._closed = True
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._lock.release()

    def _validate(
        self, adapters: Sequence[AdapterInput], max_device_bytes: int
    ) -> tuple[list[tuple[tuple[str, str], Any, Any, float]], int]:
        torch = self._torch
        rows: list[tuple[tuple[str, str], Any, Any, float]] = []
        nbytes = 0
        if max_device_bytes < 0:
            raise AdapterRefusal("adapter_budget", "adapter byte budget must be nonnegative")
        for adapter in adapters:
            if not adapter.ref or not adapter.kind or not math.isfinite(adapter.scale):
                raise AdapterRefusal(
                    "adapter_identity", "adapter identity, kind and finite scale required"
                )
            if self._snapshots.get(adapter.component) != adapter.base_snapshot:
                raise AdapterRefusal(
                    "adapter_base_mismatch", f"{adapter.ref}: selected base changed"
                )
            pairs: dict[str, dict[str, Any]] = {}
            for key, value in adapter.tensors.items():
                for suffix, role in (
                    (".lora_A.weight", "a"),
                    (".lora_B.weight", "b"),
                    (".alpha", "alpha"),
                ):
                    if key.endswith(suffix):
                        pairs.setdefault(key[: -len(suffix)], {})[role] = value
                        break
                else:
                    raise AdapterRefusal("adapter_key", f"{adapter.ref}: unknown tensor {key!r}")
            if not pairs:
                raise AdapterRefusal("adapter_empty", f"{adapter.ref}: adapter has no factors")
            for target, pair in pairs.items():
                site = adapter.component, target
                if site not in self._sites:
                    raise AdapterRefusal(
                        "adapter_target_missing", f"{adapter.ref}: unknown target {site}"
                    )
                if not {"a", "b"} <= pair.keys():
                    raise AdapterRefusal(
                        "adapter_pair", f"{adapter.ref}: incomplete factors for {site}"
                    )
                a, b = pair["a"], pair["b"]
                if any(
                    not isinstance(t, torch.Tensor)
                    or t.layout != torch.strided
                    or t.device.type != "cpu"
                    or t.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                    or t.requires_grad
                    for t in pair.values()
                ):
                    raise AdapterRefusal(
                        "adapter_tensor", f"{site}: expected dense floating CPU tensors"
                    )
                in_features, out_features, dtype = self._sites[site]
                if (
                    a.ndim != 2
                    or b.ndim != 2
                    or a.shape[0] <= 0
                    or a.shape[1] != in_features
                    or b.shape != (out_features, a.shape[0])
                    or a.dtype != b.dtype
                ):
                    raise AdapterRefusal(
                        "adapter_shape", f"{site}: incompatible A/B shape or dtype"
                    )
                alpha = pair.get("alpha")
                if alpha is not None and alpha.ndim != 0:
                    raise AdapterRefusal("adapter_alpha", f"{site}: alpha must be a scalar")
                scale = adapter.scale * (1.0 if alpha is None else float(alpha) / a.shape[0])
                if not math.isfinite(scale):
                    raise AdapterRefusal("adapter_scale", f"{site}: effective scale is not finite")
                if scale != 0:
                    nbytes += (a.numel() + b.numel()) * (torch.finfo(dtype).bits // 8)
                    if nbytes > max_device_bytes:
                        raise AdapterRefusal(
                            "adapter_budget", f"adapter factors need {nbytes} device bytes"
                        )
                # Do not scan a rejected over-budget factor or allocate its finite mask.
                if any(not bool(torch.isfinite(t).all()) for t in pair.values()):
                    raise AdapterRefusal("adapter_nonfinite", f"{site}: nonfinite adapter tensor")
                # Zero scale still validates every tensor, but owns no device factors.
                if scale != 0:
                    rows.append((site, a, b, scale))
        return rows, nbytes
