"""The `int8-convrot256/1` encoded GEMM: ComfyUI's CUTLASS W8A8 kernel, eager int8 until built.

`comfy_kitchen`'s `int8_linear` (the Runtime's reviewed `comfy-kitchen` kernel artifact, the
binary ComfyUI serves its int8 checkpoints with) rotates each activation row by the regular
Hadamard per 256 channels, quantizes it per token in the same kernel, and runs a CUTLASS int8
GEMM whose epilogue applies both scales. On an RTX 4090 at Qwen-Image-2.1's shapes it runs at
cuBLAS's bare int8 rate with the dequantization fused, 2.5-3x the bf16 GEMM.

The artifact's build starts when the first leaf is built; until it is ready the same arithmetic
runs eagerly on torch's int8 GEMM, so no run waits for a compile, and the leaf switches over at
its next call once the build lands.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import site
import sys
import time
from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from types import ModuleType

from cozy_runtime.author.kernels import readable
from cozy_runtime.internal import kernel_cache, kernel_compile, kernel_sources
from cozy_runtime.internal.encoding.formats import CONVROT_GROUP, Tensor, hadamard256

ARTIFACT = "comfy-kitchen"
#: The cards the artifact's binary carries device code for (`attention_upstream.KITCHEN_SMS`).
KITCHEN_SMS = frozenset({80, 86, 87, 89, 90, 100, 103, 110, 120, 121})
#: How often a leaf asks whether a running build has landed: resolution, never a verdict.
POLL_SECONDS = 1.0


@dataclass(slots=True)
class _Kitchen:
    module: ModuleType | None = None
    asked: float = float("-inf")
    final: str = ""


_STATE: dict[int, _Kitchen] = {}


def kitchen(torch: ModuleType, index: int) -> ModuleType | None:
    """`comfy_kitchen`'s CUDA backend on this card, or None while its build runs (it starts on
    the first ask) and for good when this card or machine cannot have it."""
    state = _STATE.setdefault(index, _Kitchen())
    now = time.monotonic()
    if state.module is not None or state.final or now - state.asked < POLL_SECONDS:
        return state.module
    state.asked = now
    major, minor = torch.cuda.get_device_capability(index)
    sm = major * 10 + minor
    store = kernel_cache.configured()
    found = util.find_spec("comfy_kitchen")
    origin = str(found.origin or "") if found is not None else ""
    from_store = store is not None and origin.startswith(
        tuple(str(root) for root in (store.own, store.trusted) if root is not None)
    )
    try:
        if sm not in KITCHEN_SMS:
            raise kernel_sources.Absent(f"comfy-kitchen carries no device code for sm{sm}")
        if found is None or from_store:
            if store is None:
                raise kernel_sources.Absent("no machine kernel store to build comfy-kitchen into")
            built = kernel_compile.submit(store, kernel_sources.job(ARTIFACT, sm))
            if built.state in ("compiling", "pending"):
                return None
            if built.state != "ready":
                raise kernel_sources.Absent(f"comfy-kitchen sm{sm}: {built.line()}")
            site.addsitedir(str(Path(built.path) / kernel_cache.SITE))
        state.module = import_module("comfy_kitchen.backends.cuda")
    except Exception as exc:  # the eager path serves this process; said once
        state.final = f"{type(exc).__name__}: {exc}"[:300]
        print(f"[int8] eager int8 GEMM for this executor: {state.final}", file=sys.stderr)
    return state.module


def matmul(torch: ModuleType, x: Tensor, data: Tensor, scale: Tensor, out_dtype: Tensor) -> Tensor:
    """`x @ W.T` for one `int8-convrot256/1` weight (`data` int8 `[N, K]`, `scale` f32 `[N]`)."""
    half = x.dtype in (torch.bfloat16, torch.float16)
    module = kitchen(torch, x.device.index or 0) if x.is_cuda and half else None
    if module is not None:
        with readable(data, scale) as (data, scale):
            return module.int8_linear(
                x, data, scale, None, out_dtype, convrot=True, convrot_groupsize=CONVROT_GROUP
            )
    return eager(torch, x, data, scale, out_dtype)


def eager(torch: ModuleType, x: Tensor, data: Tensor, scale: Tensor, out_dtype: Tensor) -> Tensor:
    """The same W8A8 arithmetic in torch ops: rotate, per-token int8, int8 GEMM, both scales."""
    rotated = hadamard256(torch, x.reshape(-1, x.shape[-1]).float()).mul_(1.0 / 16)
    row_scale = (rotated.abs().amax(-1, keepdim=True) / 127.0).clamp_(min=1e-12)
    quantized = torch.round(rotated / row_scale).clamp_(-127, 127).to(torch.int8)
    rows = int(quantized.shape[0])
    if quantized.is_cuda:
        # cuBLASLt's int8 GEMM takes more than 16 rows; the padding rows are zeros.
        padded = torch.nn.functional.pad(quantized, (0, 0, 0, max(0, 17 - rows)))
        acc = torch._int_mm(padded, data.t())[:rows]
    else:
        acc = quantized.to(torch.int32) @ data.to(torch.int32).t()
    out = acc.float().mul_(row_scale).mul_(scale.float().reshape(1, -1))
    return out.to(out_dtype).reshape(*x.shape[:-1], -1)
