"""Unmerged linear LoRA residuals at the Runtime-owned leaf boundary.

This is the execution primitive, not an adapter resolver or download path. Its caller
must provide verified tensors from exact snapshots and the admission byte budget. Hooks
are installed once, before fusion, and survive encoded-leaf replacement through the
fill backend's existing hook-registry transfer. Only request-local tensors change.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from cozy_runtime.author._context import AdapterRef
from cozy_runtime.author._errors import InvalidRequest

TILE_ROWS = 4096


class AdapterRefusal(InvalidRequest):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail, code=code)


@dataclass(frozen=True, slots=True)
class AdapterInput:
    """One resolved adapter, directed to one exact model-slot component.

    ``base_snapshot`` is the selected binding's identity, not publisher provenance.
    Tensors are canonical, component-relative keys; no runtime dialect guessing occurs.
    """

    ref: str
    component: str
    base_snapshot: str
    tensors: Mapping[str, Any]
    scale: float = 1.0
    kind: str = "lora"


@dataclass(slots=True)
class _Term:
    a: Any
    b: Any
    scale: float


@dataclass(slots=True)
class _Attempt:
    terms: Mapping[tuple[str, str], tuple[_Term, ...]]
    check_cancelled: Callable[[], None]
    check_capacity: Callable[[int], None] | None = None
    closed: bool = False
    calls: int = 0


@dataclass(frozen=True, slots=True)
class _Hook:
    binding: LinearAdapters
    site: tuple[str, str]

    def __call__(
        self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any
    ) -> Any:
        return self.binding._apply(self.site, module, args, kwargs, output)


def prequantized_input_allowed(module: Any) -> bool:
    """Inactive LoRA hooks are exact no-ops and need no floating-point input.

    Other semantic hooks, missing active contexts and nonzero factors retain the
    original Tensor. This decision is request-local; it mutates no fusion flags.
    """
    if module._forward_pre_hooks:
        return False
    for hook in module._forward_hooks.values():
        if not isinstance(hook, _Hook):
            return False
        attempt = hook.binding._active.get()
        if attempt is None:
            if hook.binding._lock.locked():
                return False
        elif attempt.closed or attempt.terms.get(hook.site):
            return False
    return True


class LinearAdapters:
    """Construction-lifetime hooks with exclusive, attempt-lifetime factor allocations.

    ``targets`` names the admitted canonical module paths, per component. Installing
    only those hooks preserves fused activation quantization at unrelated leaves.
    The binding must be recreated when this target set changes; never install hooks
    during an attempt. Nonlinear/custom forward implementations refuse explicitly.
    """

    def __init__(
        self,
        components: Mapping[str, Any],
        snapshots: Mapping[str, str],
        targets: Mapping[str, Sequence[str]],
    ) -> None:
        import torch

        from cozy_runtime.internal.encoding.leaves import (
            MicroScaledNativeLeaf,
            RowwiseNativeLeaf,
            _leaf_class,
        )

        self._torch = torch
        self._snapshots = dict(snapshots)
        self._sites: dict[tuple[str, str], tuple[int, int, Any]] = {}
        self._handles: list[Any] = []
        self._lock = threading.Lock()
        self._active: ContextVar[_Attempt | None] = ContextVar("linear_adapters", default=None)
        self._closed = False
        self._poisoned = False
        # Resolve the complete target set before mutating any hook registry.
        modules: list[tuple[tuple[str, str], Any]] = []
        seen_modules: set[int] = set()
        for component, paths in targets.items():
            if component not in components or not snapshots.get(component):
                raise AdapterRefusal("adapter_component", f"unknown component {component!r}")
            for path in paths:
                site = component, path
                if site in self._sites:
                    raise AdapterRefusal("adapter_target_duplicate", f"duplicate target {site}")
                try:
                    module = components[component].get_submodule(path)
                except (AttributeError, KeyError) as exc:
                    raise AdapterRefusal(
                        "adapter_target_missing", f"unknown target {site}"
                    ) from exc
                if type(module) is torch.nn.Linear:
                    dtype = module.weight.dtype
                elif type(module) is _leaf_class(torch):
                    dtype = module.out_dtype
                else:
                    raise AdapterRefusal(
                        "adapter_target_unsupported", f"{site} is not a linear leaf"
                    )
                if "forward" in module.__dict__ or torch.nn.modules.module._global_forward_hooks:
                    raise AdapterRefusal(
                        "adapter_forward_unsupported", f"{site}: wrapped/global-hook forward"
                    )
                if type(module) is _leaf_class(torch) and module._cozy_forward not in (
                    RowwiseNativeLeaf._forward,
                    MicroScaledNativeLeaf._forward,
                ):
                    raise AdapterRefusal(
                        "adapter_forward_unsupported", f"{site}: unknown encoded forward"
                    )
                if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise AdapterRefusal("adapter_compute_dtype", f"{site}: unsupported {dtype}")
                if id(module) in seen_modules:
                    raise AdapterRefusal("adapter_target_alias", f"{site}: aliased linear target")
                if any(isinstance(hook, _Hook) for hook in module._forward_hooks.values()):
                    raise AdapterRefusal(
                        "adapter_target_owned", f"{site}: leaf already has an adapter binding"
                    )
                seen_modules.add(id(module))
                self._sites[site] = module.in_features, module.out_features, dtype
                modules.append((site, module))
        try:
            for site, module in modules:
                self._handles.append(
                    module.register_forward_hook(_Hook(self, site), with_kwargs=True, prepend=True)
                )
        except BaseException:
            self.close()
            raise

    def _apply(
        self,
        site: tuple[str, str],
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> Any:
        self.check_ready()
        attempt = self._active.get()
        if attempt is None:
            if self._lock.locked():
                raise AdapterRefusal(
                    "adapter_context_missing", "model called outside its active adapter context"
                )
            return output
        if attempt.closed:
            raise AdapterRefusal("adapter_attempt_closed", "adapter context outlived its attempt")
        attempt.check_cancelled()
        terms = attempt.terms.get(site, ())
        if not terms:
            return output
        x = args[0] if args else kwargs.get("input", kwargs.get("x"))
        torch = self._torch
        if not isinstance(x, torch.Tensor) or not isinstance(output, torch.Tensor):
            raise AdapterRefusal("adapter_input", f"{site} did not receive an ordinary Tensor")
        in_features, out_features, dtype = self._sites[site]
        if (
            x.ndim < 1
            or x.shape[-1] != in_features
            or output.shape != (*x.shape[:-1], out_features)
            or x.dtype != dtype
            or output.dtype != dtype
            or x.device != output.device
        ):
            raise AdapterRefusal("adapter_input", f"{site}: input/output geometry or dtype changed")
        first = next(iter(module._forward_hooks.values()), None)
        if (
            not isinstance(first, _Hook)
            or first.binding is not self
            or torch.nn.modules.module._global_forward_hooks
            or "forward" in module.__dict__
        ):
            raise AdapterRefusal(
                "adapter_forward_unsupported",
                f"{site}: LoRA must observe the fresh linear result first",
            )
        if not x.is_contiguous() or not output.is_contiguous():
            raise AdapterRefusal(
                "adapter_input_layout", f"{site}: contiguous linear input/output required"
            )
        address = output.untyped_storage().data_ptr()
        if address == x.untyped_storage().data_ptr() or any(
            address == value.untyped_storage().data_ptr()
            for value in (*module.parameters(recurse=False), *module.buffers(recurse=False))
        ):
            raise AdapterRefusal(
                "adapter_output_alias", f"{site}: output aliases input or base storage"
            )
        # Both supported implementations return fresh output storage. This first local
        # post-hook may add to it; parameters, buffers and inputs remain immutable.
        inputs = x.view(-1, in_features)
        result = output.view(-1, out_features)
        linear = torch.nn.functional.linear
        if x.device.type == "cuda" and torch.get_float32_matmul_precision() != "highest":
            raise AdapterRefusal(
                "adapter_precision", "FP32 LoRA requires IEEE float32 matmul precision"
            )
        for term in terms:
            if term.a.device != x.device:
                raise AdapterRefusal("adapter_device", f"{site}: factors on another device")
        rank = max(term.a.shape[0] for term in terms)
        scratch = TILE_ROWS * (in_features + rank + 2 * out_features) * 4
        if attempt.check_capacity is not None:
            attempt.check_capacity(scratch)
        tile = accumulator = hidden = delta = None
        try:
            # FP16/BF16 -> FP32 is lossless. The ordered FP32 accumulator rounds back
            # to the base output only once, after every adapter, like an FP32 PEFT stack.
            with torch.autocast(device_type=x.device.type, enabled=False):
                if inputs.shape[0]:
                    tile = torch.empty(
                        (TILE_ROWS, in_features), dtype=torch.float32, device=x.device
                    )
                    accumulator = torch.empty(
                        (TILE_ROWS, out_features), dtype=torch.float32, device=x.device
                    )
                for start in range(0, inputs.shape[0], TILE_ROWS):
                    assert tile is not None and accumulator is not None
                    attempt.check_cancelled()
                    stop = min(start + TILE_ROWS, inputs.shape[0])
                    # Every valid row is overwritten below. Only padding needs a
                    # defined zero; keep the same GEMM geometry and rounding path.
                    if stop - start < TILE_ROWS:
                        tile[stop - start :].zero_()
                        accumulator[stop - start :].zero_()
                    tile[: stop - start].copy_(inputs[start:stop])
                    accumulator[: stop - start].copy_(result[start:stop])
                    for term in terms:
                        hidden = linear(tile, term.a)
                        delta = linear(hidden, term.b)
                        delta.mul_(term.scale)
                        accumulator.add_(delta)
                        hidden = delta = None
                    result[start:stop].copy_(accumulator[: stop - start])
        finally:
            tile = accumulator = hidden = delta = None
        attempt.calls += 1
        return output

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
                in_features, out_features, _dtype = self._sites[site]
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
                    nbytes += (a.numel() + b.numel()) * 4
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
        """Own adapter allocations until completion/cancellation; leave base bytes alone.

        The admission caller separately budgets intermediate activation scratch. Cleanup
        synchronizes the target device before releasing factors. Failure poisons this
        binding so its executor cannot reuse it accidentally.
        """
        if not self._lock.acquire(blocking=False):
            raise AdapterRefusal("adapter_concurrent", "this model already has an active attempt")
        terms: dict[tuple[str, str], tuple[_Term, ...]] = {}
        attempt = _Attempt(terms, check_cancelled, check_capacity)
        token = None
        allocated = False
        ca = cb = None
        try:
            self.check_ready()
            check_cancelled()
            rows, nbytes = self._validate(adapters, max_device_bytes)
            if check_capacity is not None:
                check_capacity(nbytes)
            for site, a, b, scale in rows:
                check_cancelled()
                dtype = self._torch.float32
                allocated = True
                # copy=True is essential even on CPU: source storage must never alias an
                # attempt, and the attempt must not retain an uploader-owned tensor.
                ca = a.to(device=device, dtype=dtype, copy=True)
                cb = b.to(device=device, dtype=dtype, copy=True)
                # Source values were checked on CPU. All supported source dtypes
                # convert losslessly to FP32, so no full-factor GPU finite mask is needed.
                terms[site] = (*terms.get(site, ()), _Term(ca, cb, scale))
            attempt.terms = MappingProxyType(terms)
            token = self._active.set(attempt)
            with self._torch.no_grad():
                yield tuple(AdapterRef(a.ref, a.scale, a.kind) for a in adapters)
        finally:
            attempt.closed = True
            if token is not None:
                self._active.reset(token)
            try:
                if allocated and self._torch.device(device).type == "cuda":
                    self._torch.cuda.synchronize(device)
            except BaseException:
                self._poisoned = True
                raise
            finally:
                # A retained exception traceback or copied context can still reference
                # a term. Clear its device references as well as the lookup container.
                for stack in terms.values():
                    for term in stack:
                        term.a = term.b = None
                terms.clear()
                ca = cb = None
                self._lock.release()

    def observed_calls(self) -> int:
        attempt = self._active.get()
        return attempt.calls if attempt is not None else 0

    def check_ready(self) -> None:
        if self._closed or self._poisoned:
            raise AdapterRefusal("adapter_binding_closed", "adapter binding is closed or poisoned")

    def close(self) -> None:
        """Construction teardown; never change hook registries during an attempt."""
        if not self._lock.acquire(blocking=False):
            raise AdapterRefusal("adapter_concurrent", "cannot close an active adapter binding")
        try:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._closed = True
        finally:
            self._lock.release()
