"""The `nvfp4-w4a4/1` encoded GEMM: comfy-kitchen's NVFP4 quantizer and cuBLASLt FP4 GEMM, the pair
ComfyUI serves BFL's nvfp4 checkpoints with, on Blackwell's FP4 tensor cores.

Each activation is scaled by its own tensor amax (`amax / (448 * 6)`, the producer's rule for
the weight's second level), quantized per 16-block by the kitchen kernel and multiplied against
the stored weight; the GEMM applies both second-level scales as its alpha. Until the kitchen
build lands the weight is decoded and multiplied in the activation dtype, so no run waits for a
compile.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

from types import ModuleType

from cozy_runtime.author.kernels import readable
from cozy_runtime.internal import int8_linear
from cozy_runtime.internal.encoding.formats import NVFP4_BLOCK, Tensor, nvfp4_values

#: The largest magnitude of the two levels: E4M3's 448 times E2M1's 6.
LEVELS_MAX = 448.0 * 6.0
#: An all-zero activation has amax 0; its scale must stay a real number.
SCALE_FLOOR = 1e-30


def matmul(
    torch: ModuleType,
    x: Tensor,
    weight: Tensor,
    blocked_scale: Tensor,
    linear_scale: object,
    scale_2: Tensor,
    out_dtype: Tensor,
) -> Tensor:
    """`x @ W.T` for one stored `nvfp4-w4a4/1` weight (`weight` u8 `[N, K/2]`, `blocked_scale` its
    E4M3 block scales in cuBLASLt's 128x4 layout, `scale_2` the f32 `[1]` second level).

    `linear_scale` is a zero-argument callable returning the `[N, K/16]` row-major scales; only the
    eager path asks for it."""
    rows_out = int(weight.shape[0])
    flat = x.reshape(-1, x.shape[-1])
    half = x.dtype in (torch.bfloat16, torch.float16)
    module = int8_linear.kitchen(torch, x.device.index or 0) if x.is_cuda and half else None
    if module is None:
        decoded = nvfp4_values(torch, weight, linear_scale(), scale_2).to(x.dtype)  # type: ignore[operator]
        return torch.nn.functional.linear(x, decoded).to(out_dtype)
    rows = int(flat.shape[0])
    flat = flat.contiguous()
    amax = flat.abs().amax().float()
    tensor_scale = (amax / LEVELS_MAX).clamp(min=SCALE_FLOOR).reshape(1)
    with readable(weight, blocked_scale, scale_2) as (weight, blocked_scale, scale_2):
        payload, block = module.quantize_nvfp4(flat, tensor_scale, 0.0, rows % NVFP4_BLOCK != 0)
        out = module.scaled_mm_nvfp4(
            payload, weight, tensor_scale, scale_2, block, blocked_scale, out_dtype=out_dtype
        )
    return out[:rows].reshape(*x.shape[:-1], rows_out)
