"""The `fp8-rowwise/1` encoded GEMM: `torch.ops.cozy.fp8_linear`, one torch op.

It lives in the Runtime's 8-bit op library (`int8_op/`), which the machine compiles on the
card at boot (`int8_linear.native`). One kernel quantizes each activation row to E4M3 against
its own amax (`eager_quantize`'s law, bit for bit). The GEMM then applies both row scales and
the bias in its epilogue: wgmma on Hopper, tcgen05 on B200 and B300, the TMA mma on RTX 50, and
cuBLASLt's `_scaled_mm` on other cards. Until the library lands, and for what it does not take
(f32, a CPU probe), the same law runs in torch ops and `_scaled_mm`.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import sys
from types import ModuleType

from cozy_runtime.author.kernels import PreQuantized
from cozy_runtime.internal import int8_linear
from cozy_runtime.internal.encoding.formats import E4M3_MAX, Tensor

#: The activation scale's floor, as `ROW_SCALE_FLOOR` is the weights': a row of exact zeros
#: has amax 0, and dividing by it poisons a whole output row with NaN.
SCALE_FLOOR = 1e-12

#: The op's name in a run's warnings.
OP = "cozy-fp8"
#: Device index -> the loaded op, `(x, data, scale, bias, out_dtype) -> out`.
_OPS: dict[int, int8_linear.Op] = {}


def native(torch: ModuleType, index: int) -> int8_linear.Op | None:
    """`torch.ops.cozy.fp8_linear` on this card, or None while the library builds."""
    op = _OPS.get(index)
    if op is None and int8_linear.native(torch, index) is not None:
        op = _OPS[index] = torch.ops.cozy.fp8_linear.default
        major, minor = torch.cuda.get_device_capability(index)
        print(f"[fp8] cozy fp8 op serves sm{major * 10 + minor}", file=sys.stderr)
    return op


def _op(torch: ModuleType, x: Tensor, *dtypes: object) -> int8_linear.Op | None:
    """The op, where it serves: a CUDA operand on a card whose library loaded, and bf16 or fp16
    for each of `dtypes`."""
    if not x.is_cuda or any(dtype not in (torch.bfloat16, torch.float16) for dtype in dtypes):
        return None
    return native(torch, x.get_device())


def eager_quantize(torch: ModuleType, x: Tensor) -> tuple[Tensor, Tensor]:
    """`[..., K]` float -> `[M, K]` E4M3 payload and `[M, 1]` f32 per-row scale: amax over K,
    `.float()` before the divide, the literal 448, a floor, and the clamp before the cast
    because the cast does not saturate. A per-row scale is what pairs with the weight's
    per-output-row one: they are the product's two outer dimensions, neither the reduction."""
    flat = x.reshape(-1, x.shape[-1])
    scale = (flat.abs().amax(dim=-1, keepdim=True).float() / E4M3_MAX).clamp(min=SCALE_FLOOR)
    return (flat.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


def quantize(torch: ModuleType, x: Tensor) -> PreQuantized:
    """`x` as the leaf's operand, shared by the fp8 leaves that read it (`quantize_once`)."""
    if _op(torch, x, x.dtype) is not None:
        try:
            payload, scale = torch.ops.cozy.fp8_quantize.default(x)
            return PreQuantized(payload, scale, tuple(x.shape))
        except Exception as exc:  # torch ops serve this call
            int8_linear.failed(OP, x.get_device(), exc)
    payload, scale = eager_quantize(torch, x)
    return PreQuantized(payload, scale, tuple(x.shape))


def gemm(
    torch: ModuleType,
    operand: PreQuantized,
    data: Tensor,
    scale: Tensor,
    bias: Tensor | None,
    out_dtype: object,
) -> Tensor:
    """The linear of a quantized operand: the op's GEMM, both scales and the bias in its
    epilogue, or `_scaled_mm` (fast accumulation, #517b) and the bias after it."""
    if _op(torch, operand.payload, out_dtype) is not None:
        try:
            out = torch.ops.cozy.fp8_gemm.default(
                operand.payload, operand.scale, data, scale, bias, out_dtype
            )
            return out.view(*operand.shape[:-1], -1)
        except Exception as exc:  # `_scaled_mm` serves this call
            int8_linear.failed(OP, operand.payload.get_device(), exc)
    # `payload.t()` of a contiguous `[out, in]` is the column-major operand `_scaled_mm`
    # wants; its `scale_b` is per output row, rank-1 and rank-2 scales alike.
    out = torch._scaled_mm(
        operand.payload,
        data.t(),
        scale_a=operand.scale.reshape(-1, 1),
        scale_b=scale.to(torch.float32).reshape(1, -1),
        out_dtype=out_dtype,
        use_fast_accum=True,
    )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out.view(*operand.shape[:-1], -1)


def linear(
    torch: ModuleType,
    x: Tensor,
    data: Tensor,
    scale: Tensor,
    bias: Tensor | None,
    out_dtype: object,
) -> Tensor:
    """`x @ W.T + bias` for one `fp8-rowwise/1` weight: `data` E4M3 `[N, K]`, `scale` f32."""
    op = _op(torch, x, x.dtype, out_dtype)
    if op is not None:
        try:
            return op(x, data, scale, bias, out_dtype)
        except Exception as exc:  # the op's two halves, else torch ops, serve this call
            int8_linear.failed(OP, x.get_device(), exc)
    return gemm(torch, quantize(torch, x), data, scale, bias, out_dtype)
