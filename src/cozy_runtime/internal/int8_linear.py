"""The `int8-convrot256/1` encoded GEMM: the Runtime's int8 op, compiled on the card at boot.

`torch.ops.cozy.int8_linear` (`int8_op/`) is one torch op. It rotates each activation row by the
regular Hadamard per 256 channels and quantizes it per token in one kernel, then runs an int8
GEMM whose epilogue applies both scales and the bias: wgmma on Hopper, tcgen05 on datacenter
Blackwell, mma.sync on every other card, and a single GEMV launch for up to 4 rows. One Python
call, so a leaf's host time is a bf16 `F.linear`'s. A weight stand-in reaches it through torch
dispatch, which binds its bytes for the op.

The op is built by the machine's compile manager at boot (`machine_kernels`), beside the weight
download. Until it lands a leaf serves through comfy-kitchen's reviewed binary, else eagerly
on torch's int8 GEMM, so no run waits for a compile, and it switches at its next call.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import site
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module, util
from pathlib import Path
from types import ModuleType
from typing import Any

from cozy_runtime.author.kernels import readable
from cozy_runtime.internal import kernel_cache, kernel_compile, kernel_sources
from cozy_runtime.internal.encoding.formats import CONVROT_GROUP, Tensor, hadamard256

ARTIFACT = "comfy-kitchen"
OP = "cozy-int8"
#: The cards the artifact's binary carries device code for (`attention_upstream.KITCHEN_SMS`).
KITCHEN_SMS = frozenset({80, 86, 87, 89, 90, 100, 103, 110, 120, 121})
#: How often a leaf asks whether a running build has landed: resolution, never a verdict.
POLL_SECONDS = 1.0


@dataclass(slots=True)
class _Build:
    served: Any = None
    asked: float = float("-inf")
    final: str = ""


_KITCHEN: dict[int, _Build] = {}
_NATIVE: dict[int, _Build] = {}
#: Device index -> the loaded op, and the activation dtypes it takes: every call reads these.
_OPS: dict[int, Any] = {}
_HALF: list[Any] = []
#: The op library this process loaded; its ops are registered once per process.
_LOADED: list[str] = []


def _built(
    torch: ModuleType,
    index: int,
    states: dict[int, _Build],
    name: str,
    opened: Callable[[int], Any],
) -> Any:
    """`opened(sm)` on this card once its build lands, None while it runs (the first ask starts
    it) and for good when this card or machine cannot have it, said once."""
    state = states.setdefault(index, _Build())
    now = time.monotonic()
    if state.served is not None or state.final or now - state.asked < POLL_SECONDS:
        return state.served
    state.asked = now
    major, minor = torch.cuda.get_device_capability(index)
    sm = major * 10 + minor
    try:
        state.served = opened(sm)
        if state.served is not None:
            print(f"[int8] {name} serves sm{sm}", file=sys.stderr)
    except Exception as exc:  # the next path serves this process; said once
        state.final = f"{type(exc).__name__}: {exc}"[:300]
        print(f"[int8] no {name} for this executor: {state.final}", file=sys.stderr)
    return state.served


def _store() -> kernel_cache.Store:
    store = kernel_cache.configured()
    if store is None:
        raise kernel_sources.Absent("no machine kernel store to build into")
    return store


def _compiled(name: str, sm: int) -> Path | None:
    """The built artifact's tree, None while it compiles."""
    built = kernel_compile.submit(_store(), kernel_sources.job(name, sm))
    if built.state in ("compiling", "pending"):
        return None
    if built.state != "ready":
        raise kernel_sources.Absent(f"{name} sm{sm}: {built.line()}")
    return Path(built.path) / kernel_cache.SITE


def native(torch: ModuleType, index: int) -> Any:
    """`torch.ops.cozy.int8_linear` on this card, or None while its build runs."""

    def opened(sm: int) -> Any:
        if sm < 80:
            raise kernel_sources.Absent(f"no int8 tensor-core GEMM for sm{sm}")
        tree = _compiled(OP, sm)
        if tree is None:
            return None
        [library] = tree.glob("cozy_int8*.so")
        if not _LOADED:
            torch.ops.load_library(str(library))
            _LOADED.append(str(library))
        elif _LOADED[0] != str(library):
            raise kernel_sources.Absent(f"this process serves another card's op ({_LOADED[0]})")
        _HALF[:] = (torch.bfloat16, torch.float16)
        _OPS[index] = torch.ops.cozy.int8_linear.default
        return _OPS[index]

    return _built(torch, index, _NATIVE, "cozy int8 op", opened)


def kitchen(torch: ModuleType, index: int) -> ModuleType | None:
    """`comfy_kitchen`'s CUDA backend on this card, or None while its build runs (it starts on
    the first ask) and for good when this card or machine cannot have it."""

    def opened(sm: int) -> ModuleType | None:
        if sm not in KITCHEN_SMS:
            raise kernel_sources.Absent(f"comfy-kitchen carries no device code for sm{sm}")
        found = util.find_spec("comfy_kitchen")
        store = kernel_cache.configured()
        origin = str(found.origin or "") if found is not None else ""
        roots = () if store is None else (store.own, store.trusted)
        if found is None or origin.startswith(tuple(str(root) for root in roots if root)):
            tree = _compiled(ARTIFACT, sm)
            if tree is None:
                return None
            site.addsitedir(str(tree))
        return import_module("comfy_kitchen.backends.cuda")

    module: ModuleType | None = _built(
        torch, index, _KITCHEN, "comfy-kitchen CUTLASS int8 GEMM", opened
    )
    return module


def linear(
    torch: ModuleType, x: Tensor, data: Tensor, scale: Tensor, bias: Tensor | None, out_dtype: Any
) -> Tensor:
    """`x @ W.T + bias` for one `int8-convrot256/1` weight: `data` int8 `[N, K]`, `scale` f32."""
    op = _OPS.get(x.get_device())
    if op is not None and x.dtype in _HALF:
        return op(x, data, scale, bias, out_dtype)
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
        index = x.device.index or 0
        op = native(torch, index)
        if op is not None:
            return op(x, data, scale, bias, out_dtype)
        module = kitchen(torch, index)
        if module is not None:
            with readable(data, scale, bias) as (data, scale, bias):
                return module.int8_linear(
                    x, data, scale, bias, out_dtype, convrot=True, convrot_groupsize=CONVROT_GROUP
                )
    out = eager(torch, x, data, scale, out_dtype)
    return out if bias is None else out + bias.to(out.dtype)


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
