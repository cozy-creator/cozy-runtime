"""Adopt existing rowwise FP8 bytes through the upstream Tensor interface.

The activation recipe and separate bias addition preserve the existing Runtime
route. TorchAO owns quantized Tensor dispatch and the GEMM; PEFT can wrap the real
Linear and retain its original floating input. The provider oracle must qualify
this binding on the execution device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType
from typing import Any

from tensorfs.derived import Part, Tensor

from cozy_runtime.author._loader import canonical_dtype
from cozy_runtime.internal.encoding.leaves import (
    PreQuantized,
    RowwiseNativeLeaf,
    quantize_activation_rowwise,
)


def _activation_tensor(payload: Any, scale: Any, *, dtype: Any) -> Any:
    """Wrap ephemeral activations; stored weights use TensorFS.from_parts."""
    import torch
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization.quantize_.common import KernelPreference
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

    if (
        payload.ndim < 2
        or payload.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
        or payload.device != scale.device
        or not payload.is_contiguous()
        or not scale.is_contiguous()
        or scale.numel() != math.prod(payload.shape[:-1])
        or dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise ValueError("rowwise FP8 requires contiguous E4M3 codes and FP32 row scales")
    return Float8Tensor(
        payload,
        scale.view(*payload.shape[:-1], 1),
        block_size=[1] * (payload.ndim - 1) + [payload.shape[-1]],
        dtype=dtype,
        mm_config=Float8MMConfig(use_fast_accum=True),
        act_quant_kwargs=None,
        kernel_preference=KernelPreference.TORCH,
    )


def rowwise_input(value: Any, *, dtype: Any) -> Any:
    """Keep the established quantizer, or adopt an already encoded activation."""
    import torch

    if isinstance(value, PreQuantized):
        shape = value.shape
        payload, scale = value.payload, value.scale
        if not shape or tuple(payload.shape) != (math.prod(shape[:-1]), shape[-1]):
            raise ValueError("prequantized activation geometry differs from its logical shape")
    else:
        if (
            type(value) is not torch.Tensor
            or value.ndim < 1
            or value.dtype != dtype
            or value.shape[-1] == 0
        ):
            raise ValueError("rowwise FP8 expects an ordinary floating activation")
        shape = tuple(value.shape)
        payload, scale = quantize_activation_rowwise(torch, value)
    return _activation_tensor(
        payload.view(shape if len(shape) > 1 else (1, shape[0])), scale, dtype=dtype
    )


def _rowwise_forward(self: Any, input: Any) -> Any:
    import torch

    if self.weight.device.type != "cuda":
        raise RuntimeError("rowwise FP8 requires CUDA W8A8; no weight-only fallback is enabled")
    quantized = rowwise_input(input, dtype=self.weight.dtype)
    # Native nn.Linear would fuse bias into _scaled_mm's epilogue. The existing
    # route rounds GEMM to the logical dtype before adding bias; retain that order.
    result = torch.nn.functional.linear(quantized, self.weight, None)
    result = result.reshape(*input.shape[:-1], self.out_features)
    return result if self.bias is None else result + self.bias


def bind_rowwise_weight(module: Any, payload: Any, scale: Any, *, spec: Tensor) -> Any:
    """Replace only a Linear's weight; return its former Parameter to its owner.

    The receiver-bound method contains no closure over a model or Parameter, so
    deepcopy uses the copied module's weights. Existing hooks and their handles
    stay on the original module. Lifecycle ownership belongs to the fill plane.
    """
    import torch
    from tensorfs import from_parts
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization.quantize_.common import KernelPreference

    if type(module) is not torch.nn.Linear:
        raise TypeError("standard rowwise binding requires an ordinary nn.Linear")
    forward = module.__dict__.get("forward")
    if forward is not None and getattr(forward, "__func__", None) is not _rowwise_forward:
        raise ValueError("standard rowwise binding cannot replace an authored forward")
    previous = module.weight
    if tuple(spec.shape) != (module.out_features, module.in_features):
        raise ValueError("stored weight shape differs from the Linear geometry")
    if spec.logical_dtype != canonical_dtype(previous.dtype):
        raise ValueError("stored logical dtype differs from the Linear weight")
    weight = from_parts(
        spec,
        {"data": payload, "scale": scale},
        mm_config=Float8MMConfig(use_fast_accum=True),
        kernel_preference=KernelPreference.TORCH,
        act_quant_kwargs=None,
    )
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    module.forward = MethodType(_rowwise_forward, module)
    return previous


@dataclass(frozen=True, slots=True)
class TorchaoRowwiseLeaf(RowwiseNativeLeaf):
    """Standard weight provider, admitted only by its native device probe."""

    @property
    def name(self) -> str:
        return "torchao.fp8-rowwise.native/1"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm89+"

    def leaf(self, torch: Any, parts: Any, replaced: Any, out_dtype: Any) -> Any:
        module = torch.nn.Linear(
            replaced.in_features,
            replaced.out_features,
            bias=False,
            dtype=out_dtype,
            device="meta",
        )
        if replaced.bias is not None:
            module.bias = torch.nn.Parameter(replaced.bias, requires_grad=False)
        # Provider probes already supply their exact encoding and physical roles.
        spec = Tensor(
            canonical_dtype(out_dtype),
            (replaced.out_features, replaced.in_features),
            self.encoding,
            {
                role: Part(canonical_dtype(value.dtype), tuple(value.shape))
                for role, value in parts.items()
            },
        )
        bind_rowwise_weight(module, parts["data"], parts["scale"], spec=spec)
        return module
