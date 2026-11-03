"""H3's stored INT8 projection, with queried dispatch and supplied Rows storage.

Codes/scales stay encoded. The caller owns activation codes, row scales, workspace,
and the FP16 bias cast that the ordinary Prebuilt path performs before its epilogue.
No allocation, capacity decision, kernel substitution or retry occurs in execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from cozy_runtime import _kernels
from cozy_runtime.author.kernels import _LINEAR_SPLIT
from cozy_runtime.internal.execution.native_linear_split import _Dispatch, launch_projection
from cozy_runtime.internal.execution.native_residual_ffn import _leaf
from cozy_runtime.rows import rows


def eligible(module: Any, channels: int, width: int) -> bool:
    return bool(
        _leaf(module, torch.float16)
        and not module.training
        and module.in_features == width and module.out_features == channels
        and not module._backward_hooks and not module._backward_pre_hooks
        and rows(module) is not None
    )


def query(library: Any, device: int, count: int, width: int, channels: int) -> _Dispatch | None:
    info = _kernels._gemm.int8_plan(device, True, count, channels, width, channels)
    if info["status"] == _kernels._gemm.NOT_SUPPORTED:
        return None
    library._check(info["status"], "the VAE INT8 projection plan")
    if info["kind"] not in (0, 1):
        return None
    return _Dispatch(count, info["kind"], info["config"], info["workspace_bytes"])


def validate(
    dispatch: _Dispatch, data: Tensor, scale: Tensor, bias: Tensor,
    x: Tensor, out: Tensor, q: Tensor | None, scales: Tensor | None,
    workspace: Tensor | None,
) -> None:
    count, width, channels = x.numel() // x.shape[-1], x.shape[-1], data.shape[0]
    if (dispatch.rows != count or dispatch.kind not in (0, 1)
            or tuple(data.shape) != (channels, width) or data.dtype != torch.int8
            or tuple(scale.shape) != (channels,) or scale.dtype != torch.float32
            or tuple(bias.shape) != (channels,) or bias.dtype not in (torch.float16, torch.float32)
            or x.dtype != torch.float16 or out.dtype != torch.float16
            or out.numel() != count * channels or out.shape[-1] != channels
            or channels % 8 or width % 256 or width > _kernels._gemm.MAX_K
            or not 0 < count <= 2**31 - 1):
        raise ValueError("VAE INT8 operands differ from their queried geometry")
    operands = [data, scale, bias, x, out]
    if dispatch.kind == 0:
        if (q is None or q.dtype != torch.int8 or q.numel() < count * width
                or scales is None or scales.dtype != torch.float32 or scales.numel() < count):
            raise ValueError("VAE INT8 projection lacks its declared quantization storage")
        operands.extend((q, scales))
    if dispatch.workspace:
        if workspace is None or workspace.dtype != torch.uint8 or workspace.numel() < dispatch.workspace:
            raise ValueError("VAE INT8 projection lacks its queried workspace")
        operands.append(workspace)
    if any(not t.is_contiguous() or t.device != x.device or not t.data_ptr()
           or t.data_ptr() % 16 for t in operands):
        raise ValueError("VAE INT8 supplied operands must be contiguous, aligned and colocated")


@dataclass
class _Window:
    module: Any
    invocation: _Invocation

    def run(self, x: Any, operand: Any, out: Any) -> Tensor:
        return self.invocation.run(x, operand, out)


@dataclass
class _Invocation:
    module: Any
    library: Any
    data: Tensor
    scale: Tensor
    source_bias: Tensor
    bias: Tensor
    x: Tensor
    target: Tensor
    dispatch: _Dispatch
    q: Tensor | None
    scales: Tensor | None
    workspace: Tensor | None
    stream: int
    calls: int = 0

    def rows(self, window: Any, x: Any, start: int, stop: int, out: Any) -> Tensor:
        if (x is not self.x or out is not self.target or start != 0 or stop != self.data.shape[0]
                or window.data.data_ptr() != self.data.data_ptr()
                or window.scale.data_ptr() != self.scale.data_ptr()
                or window.bias.data_ptr() != self.source_bias.data_ptr()):
            raise RuntimeError("VAE supplied Rows binding changed its source or operands")
        token = _LINEAR_SPLIT.set(_Window(window, self))
        try:
            return window.linear(x, None, out)
        finally:
            _LINEAR_SPLIT.reset(token)

    def run(self, x: Any, operand: Any, out: Any) -> Tensor:
        self.calls += 1
        if (self.calls != 1 or x is not self.x or operand is not None or out is not self.target
                or int(torch.cuda.current_stream(x.device).cuda_stream) != self.stream):
            raise RuntimeError("VAE INT8 projection changed its one-shot invocation or stream")

        def scratch(size: int) -> Tensor:
            if self.workspace is None or size > self.workspace.numel():
                raise RuntimeError("VAE INT8 projection requested undeclared workspace")
            return self.workspace

        launch_projection(self.library, self.data, self.scale, self.bias, x, out,
                          self.dispatch, self.q, self.scales, scratch, self.stream)
        return out


def execute(
    module: Any, library: Any, dispatch: _Dispatch, data: Tensor, scale: Tensor,
    bias: Tensor, x: Tensor, out: Tensor, q: Tensor | None, scales: Tensor | None,
    workspace: Tensor | None, bias_bank: Tensor, stream: int,
) -> None:
    call = rows(module)
    if call is None or _LINEAR_SPLIT.get() is not None:
        raise RuntimeError("VAE encoded Rows eligibility changed before execution")
    cast = bias_bank[:bias.numel()]
    cast.copy_(bias)  # Same FP16 roundpoint as Prebuilt._weight's bias.to(out_dtype).
    invocation = _Invocation(module, library, data, scale, bias, cast, x, out,
                             dispatch, q, scales, workspace, stream)
    token = _LINEAR_SPLIT.set(invocation)
    try:
        result = call(x, 0, data.shape[0], out=out)
        if result is not out or invocation.calls != 1:
            raise RuntimeError("VAE supplied Rows did not execute exactly once into its output")
    finally:
        _LINEAR_SPLIT.reset(token)
