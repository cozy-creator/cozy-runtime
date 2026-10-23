"""The `w4a8.convrot` encoded GEMM (xs-062, design `four-bit-lanes.md`): 4-bit codes in
int8.convrot's rotated basis, unpacked to int8 in registers and multiplied on the int8 tensor cores
with int8.convrot's activation (`_gemm.w4a8_gemm`). The activation is the int8 leaves' operand,
quantized by their op and shared with them.

The GEMM serves once its probe passes on the card (`prebuilt.probe_w4a8`). Where it does not, the
codes are unpacked by torch ops and the int8 lane's GEMM serves, and every run says so once.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import sys
from functools import partial
from types import ModuleType
from typing import cast

from cozy_runtime.author._model import Building
from cozy_runtime.author.kernels import INT8_CONVROT, PreQuantized
from cozy_runtime.internal import int8_linear, kernel_ready, prebuilt, verify
from cozy_runtime.internal.encoding.formats import Tensor

OP = "cozy-w4a8"
#: The activation operand it takes: int8.convrot's.
OPERAND = INT8_CONVROT


def native(torch: ModuleType, index: int) -> prebuilt.Prebuilt | None:
    """The 8-bit library on this card once its w4a8 GEMM verifies here, off any request thread
    (`kernel_ready`); None until then, and for good where the GEMM fails its probe (said once)."""
    name = f"{OP} cuda:{index}"
    if int8_linear.native(torch, index) is None:
        return None
    if torch.cuda.is_current_stream_capturing():
        ops = kernel_ready.ready(name)
        if ops is None and kernel_ready.failed(name) is None:
            raise Building(
                OP
            )  # its probe would end the graph being recorded: record once it serves
        return cast("prebuilt.Prebuilt | None", ops)
    return kernel_ready.ready(name, partial(_verified, torch, index))


def _verified(torch: ModuleType, index: int, wait: bool) -> prebuilt.Prebuilt:
    ops = int8_linear.library(index)
    major, minor = torch.cuda.get_device_capability(index)
    verdict = verify.verified(
        "prebuilt-w4a8", prebuilt.artifact(), major * 10 + minor,
        partial(prebuilt.probe_w4a8, torch, ops),
    )  # fmt: skip
    if not verdict.ok:
        print(f"[w4a8] no w4a8 GEMM on cuda:{index}: {verdict.detail}", file=sys.stderr)
        raise RuntimeError(verdict.detail)
    print(f"[w4a8] verified on sm{major}{minor}: {verdict.detail}", file=sys.stderr)
    return ops


def why(index: int) -> str:
    """Why this card's w4a8 leaves serve a slower path, or ""."""
    failed = kernel_ready.failed(f"{OP} cuda:{index}")
    return "" if failed is None else str(failed)


def serving(index: int) -> str:
    """The kernel this card's w4a8 leaves serve now: a run's `gemm.selected`."""
    if kernel_ready.ready(f"{OP} cuda:{index}") is not None:
        return OP
    return int8_linear.serving(index)


def linear(
    torch: ModuleType, x: Tensor, w: prebuilt.W4a8, bias: Tensor | None, out_dtype: object
) -> Tensor:
    """`x @ W.T + bias` for one w4a8 weight."""
    index = x.get_device()
    ops = native(torch, index) if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) else None
    if ops is not None:
        try:
            return ops.w4a8_linear(x, w, bias, out_dtype)
        except Exception as exc:
            int8_linear.failed(OP, index, exc)
    return int8_linear.linear(torch, x, prebuilt.unpack(torch, w), w.scale, bias, out_dtype)


def gemm(
    torch: ModuleType,
    operand: PreQuantized,
    w: prebuilt.W4a8,
    bias: Tensor | None,
    out_dtype: object,
) -> Tensor:
    """The linear of an int8.convrot operand (`int8_linear.quantize`, a producer's prologue)."""
    index = operand.payload.get_device()
    ops = native(torch, index) if out_dtype in (torch.bfloat16, torch.float16) else None
    if ops is not None:
        try:
            out = ops.w4a8_gemm(operand.payload, operand.scale, w, bias, out_dtype)
            return out.view(*operand.shape[:-1], -1)
        except Exception as exc:
            int8_linear.failed(OP, index, exc)
    return int8_linear.gemm(torch, operand, prebuilt.unpack(torch, w), w.scale, bias, out_dtype)
