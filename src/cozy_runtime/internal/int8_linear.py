"""The `int8-convrot256/1` encoded GEMM: the Runtime wheel's prebuilt 8-bit op (`prebuilt`).

The op rotates each activation row by the regular Hadamard per 256 channels and quantizes it per
token in one kernel, then runs an int8 GEMM whose epilogue applies both scales and the bias:
wgmma on Hopper, tcgen05 on datacenter Blackwell, mma.sync on every other card, and a single GEMV
launch for up to 4 rows. A weight stand-in reaches it through torch dispatch, which binds its
bytes for the op. It serves once it passes its probe on the card (`verify`); where it does not,
a leaf serves through comfy-kitchen's reviewed binary, else eagerly on torch's int8 GEMM, and
every run says so once.

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
from typing import cast

from cozy_runtime.author._errors import is_device_oom
from cozy_runtime.author._model import LANDED, WARNINGS, Building
from cozy_runtime.author.kernels import INT8_CONVROT, PreQuantized, readable
from cozy_runtime.internal import (
    arch,
    kernel_cache,
    kernel_compile,
    kernel_sources,
    prebuilt,
    tools,
)
from cozy_runtime.internal.encoding.formats import CONVROT_GROUP, Tensor, hadamard256

ARTIFACT = "comfy-kitchen"
OP = "cozy-int8"
#: The kernel a leaf serves when neither the op nor comfy-kitchen does (`gemm.selected`).
EAGER = "torch._int_mm"
#: How often a leaf asks whether comfy-kitchen's build has landed: resolution, never a verdict.
POLL_SECONDS = 1.0
#: The op's one-launch GEMV takes this many rows (`kGemvRows`): it rotates them itself.
GEMV_ROWS = 4


@dataclass(slots=True)
class _Build:
    served: object = None
    asked: float = float("-inf")
    final: str = ""


#: The loaded op: `(x, data, scale, bias, out_dtype) -> out`.
type Op = Callable[..., Tensor]

#: Device index -> the verified library serving its 8-bit ops.
_LIBS: dict[int, prebuilt.Prebuilt] = {}

_KITCHEN: dict[int, _Build] = {}
_NATIVE: dict[int, _Build] = {}
#: Device index -> the loaded op, and the activation dtypes it takes: every call reads these.
_OPS: dict[int, Op] = {}
_HALF: list[object] = []


def _built(
    torch: ModuleType,
    index: int,
    states: dict[int, _Build],
    name: str,
    opened: Callable[[int], object],
) -> object:
    """`opened(sm)` on this card once its build lands, None while it runs (the first ask starts
    it) and for good when this card or machine cannot have it, said once."""
    state = states.setdefault(index, _Build())
    now = time.monotonic()
    if (
        state.served is not None
        or state.final
        or now - state.asked < POLL_SECONDS
        or torch.cuda.is_current_stream_capturing()  # a load would end the graph being recorded
    ):
        return state.served
    state.asked = now
    major, minor = torch.cuda.get_device_capability(index)
    sm = major * 10 + minor
    try:
        state.served = opened(sm)
        if state.served is not None:
            LANDED[0] += 1
            print(f"[int8] {name} serves sm{sm}", file=sys.stderr)
    except Exception as exc:  # the next path serves this process; said once
        state.final = f"{type(exc).__name__}: {exc}"[:300]
        print(f"[int8] no {name} for this executor: {state.final}", file=sys.stderr)
    return state.served


@dataclass(frozen=True, slots=True)
class Unavailable:
    """A card the 8-bit op does not serve for this executor (no wheel for this platform, or its
    kernels failed their probe here): its int8 and fp8 leaves serve slower paths (comfy-kitchen,
    cuBLASLt, torch ops) for the executor's life."""

    op: str
    device: int
    reason: str


def unavailable() -> tuple[Unavailable, ...]:
    """Each card a leaf asked for the op on that it does not serve; every run says so."""
    return tuple(Unavailable(OP, i, b.final) for i, b in sorted(_NATIVE.items()) if b.final)


def failed(op: str, device: int, exc: Exception) -> None:
    """`op` raised on this card: its caller serves a slower path for this one call, and the run
    says so, once per reason. A device OOM raises: it is the memory manager's to answer."""
    if is_device_oom(exc):
        raise exc
    reason = f"{type(exc).__name__}: {exc}"[:300]
    fields: dict[str, str | bool | int | float | None] = {
        "op": op,
        "device": device,
        "reason": reason,
    }
    row = ("encoded GEMM failed, a slower path served", fields)
    if row not in WARNINGS:
        WARNINGS.append(row)
        print(
            f"[int8] {op} failed on cuda:{device}, a slower path serves: {reason}", file=sys.stderr
        )


def library(index: int) -> prebuilt.Prebuilt:
    """The library `native` verified for this card."""
    return _LIBS[index]


def _store() -> kernel_cache.Store:
    store = kernel_cache.configured()
    if store is None:
        raise kernel_sources.Absent("no machine kernel store to build into")
    return store


def _kitchen_tree(sm: int) -> Path | None:
    """comfy-kitchen's reviewed wheel unpacked for this card (`kernel_sources`), None while the
    machine's compile manager places it."""
    try:
        job = kernel_sources.job(ARTIFACT, sm)
    except kernel_sources.Absent as exc:
        for tool in exc.tools:
            tools.missing(tool, "kernel_prebuilt_missing")
        raise
    built = kernel_compile.submit(_store(), job)
    if built.state in ("compiling", "pending"):
        return None
    if built.state != "ready":
        raise kernel_sources.Absent(f"{ARTIFACT} sm{sm}: {built.line()}")
    return Path(built.path) / kernel_cache.SITE


def native(torch: ModuleType, index: int) -> Op | None:
    """The prebuilt 8-bit op's int8 linear on this card, once it verifies here (`prebuilt`);
    None for good where it does not, said once."""

    def opened(sm: int) -> Op:
        ops, detail = prebuilt.library(torch, index, sm)
        if ops is None:
            raise kernel_sources.Absent(detail)
        print(f"[int8] prebuilt kernels verified on sm{sm}: {detail}", file=sys.stderr)
        _HALF[:] = (torch.bfloat16, torch.float16)
        _LIBS[index] = ops
        _OPS[index] = ops.int8_linear
        return _OPS[index]

    return cast("Op | None", _built(torch, index, _NATIVE, "cozy int8 op", opened))


def kitchen(torch: ModuleType, index: int) -> ModuleType | None:
    """`comfy_kitchen`'s CUDA backend on this card, or None while its build runs (it starts on
    the first ask) and for good when this card or machine cannot have it."""

    def opened(sm: int) -> ModuleType | None:
        if not arch.row(sm).family:
            raise kernel_sources.Absent(f"comfy-kitchen carries no device code for sm{sm}")
        found = util.find_spec("comfy_kitchen")
        store = kernel_cache.configured()
        origin = str(found.origin or "") if found is not None else ""
        roots = () if store is None else (store.own, store.trusted)
        if found is None or origin.startswith(tuple(str(root) for root in roots if root)):
            tree = _kitchen_tree(sm)
            if tree is None:
                return None
            site.addsitedir(str(tree))
        return import_module("comfy_kitchen.backends.cuda")

    return cast(
        "ModuleType | None",
        _built(torch, index, _KITCHEN, "comfy-kitchen CUTLASS int8 GEMM", opened),
    )


def linear(
    torch: ModuleType,
    x: Tensor,
    data: Tensor,
    scale: Tensor,
    bias: Tensor | None,
    out_dtype: object,
) -> Tensor:
    """`x @ W.T + bias` for one `int8-convrot256/1` weight: `data` int8 `[N, K]`, `scale` f32."""
    index = x.get_device()
    op = _OPS.get(index)
    try:
        if op is not None and x.dtype in _HALF:
            return op(x, data, scale, bias, out_dtype)
        if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
            op = native(torch, index)
            if op is not None:
                return op(x, data, scale, bias, out_dtype)
            if torch.cuda.is_current_stream_capturing() and not _NATIVE[index].final:
                # comfy-kitchen's GEMM ends a capture (run 10264) and a recording of the eager law
                # replays 5x slower than kitchen as written (run 10287): record once the op lands.
                # An op refused for good never lands: the component runs as written.
                raise Building(OP)
            module = kitchen(torch, index)
            if module is not None:
                with readable(data, scale, bias) as (d, s, b):
                    return module.int8_linear(
                        x, d, s, b, out_dtype, convrot=True, convrot_groupsize=CONVROT_GROUP
                    )
    except Building:
        raise
    except Exception as exc:  # eager serves this call
        failed(OP if op is not None else ARTIFACT, index, exc)
    out = eager(torch, x, data, scale, out_dtype)
    return out if bias is None else out + bias.to(out.dtype)


def serving(index: int) -> str:
    """The kernel this card's int8 leaves serve now: a run's `gemm.selected`, read once a run."""
    if _OPS.get(index) is not None:
        return OP
    built = _KITCHEN.get(index)
    return ARTIFACT if built is not None and built.served is not None else EAGER


def quantize(torch: ModuleType, x: Tensor) -> PreQuantized | None:
    """`x` rotated and quantized once for several int8 leaves (`kernels.quantize_once`), or None
    where each leaf's own call serves it: no op on this card yet, or the GEMV's few rows."""
    if not x.is_cuda or x.dtype not in _HALF or x.numel() <= GEMV_ROWS * x.shape[-1]:
        return None
    if _OPS.get(x.get_device()) is None:
        return None
    try:
        payload, scale = _LIBS[x.get_device()].int8_quantize(x)
    except Exception as exc:
        failed(OP, x.get_device(), exc)
        return None
    return PreQuantized(payload, scale, tuple(x.shape), INT8_CONVROT)


#: The quantize kernel's prologue codes (`cozy_int8::Prologue`).
NORMS = {None: 0, "layer": 1, "rms": 2}
ACTIVATIONS = {None: 0, "gelu": 1, "gelu_tanh": 2}


def prologue(
    torch: ModuleType,
    x: Tensor,
    consumers: tuple[object, ...],
    *,
    norm: str | None,
    eps: float,
    scale: Tensor | None,
    shift: Tensor | None,
    activation: str | None,
) -> PreQuantized | None:
    """`activation(norm(x) * (1 + scale) + shift)`, rotated and quantized in one pass, when each
    of `consumers` is an un-hooked int8 leaf of x's width the op serves on this card; else None."""
    if not x.is_cuda or x.dtype not in _HALF or _OPS.get(x.get_device()) is None:
        return None
    if any(t is not None and t.dtype != x.dtype for t in (scale, shift)):
        return None  # eager promotes the rows to the table's dtype
    hooks = torch.nn.modules.module
    if hooks._global_forward_hooks or hooks._global_forward_pre_hooks:
        return None
    for module in consumers:
        if (
            getattr(module, "provider", None) != INT8_CONVROT
            or getattr(module, "in_features", None) != x.shape[-1]
            or module._forward_hooks  # type: ignore[attr-defined]
            or module._forward_pre_hooks  # type: ignore[attr-defined]
        ):
            return None
    try:
        payload, row_scale = _LIBS[x.get_device()].int8_quantize(
            x, scale, shift, NORMS[norm], eps, ACTIVATIONS[activation]
        )
    except Exception as exc:
        failed(OP, x.get_device(), exc)
        return None
    return PreQuantized(payload, row_scale, tuple(x.shape), INT8_CONVROT)


def gemm(
    torch: ModuleType,
    operand: PreQuantized,
    data: Tensor,
    scale: Tensor,
    bias: Tensor | None,
    out_dtype: object,
) -> Tensor:
    """The linear of a `quantize`d operand: the op's GEMM, both scales and the bias in its
    epilogue."""
    try:
        out = _LIBS[operand.payload.get_device()].int8_gemm(
            operand.payload, operand.scale, data, scale, bias, out_dtype
        )
    except Exception as exc:  # torch's int8 GEMM serves this call
        failed(OP, operand.payload.get_device(), exc)
        out = _product(torch, operand.payload, operand.scale, data, scale, out_dtype)
        out = out if bias is None else out + bias.to(out.dtype)
    return out.view(*operand.shape[:-1], -1)


def eager(torch: ModuleType, x: Tensor, data: Tensor, scale: Tensor, out_dtype: Tensor) -> Tensor:
    """The same W8A8 arithmetic in torch ops: rotate, per-token int8, int8 GEMM, both scales."""
    rotated = hadamard256(torch, x.reshape(-1, x.shape[-1]).float()).mul_(1.0 / 16)
    row_scale = (rotated.abs().amax(-1, keepdim=True) / 127.0).clamp_(min=1e-12)
    quantized = torch.round(rotated / row_scale).clamp_(-127, 127).to(torch.int8)
    out = _product(torch, quantized, row_scale, data, scale, out_dtype)
    return out.reshape(*x.shape[:-1], -1)


def _product(
    torch: ModuleType,
    quantized: Tensor,
    row_scale: Tensor,
    data: Tensor,
    scale: Tensor,
    out_dtype: Tensor,
) -> Tensor:
    """`[M, K]` int8 rows times the weight in torch's int8 GEMM, both scales applied."""
    rows = int(quantized.shape[0])
    if quantized.is_cuda:
        # cuBLASLt's int8 GEMM takes more than 16 rows; the padding rows are zeros.
        padded = torch.nn.functional.pad(quantized, (0, 0, 0, max(0, 17 - rows)))
        acc = torch._int_mm(padded, data.t())[:rows]
    else:
        acc = quantized.to(torch.int32) @ data.to(torch.int32).t()
    out = acc.float().mul_(row_scale.reshape(-1, 1)).mul_(scale.float().reshape(1, -1))
    return out.to(out_dtype)
