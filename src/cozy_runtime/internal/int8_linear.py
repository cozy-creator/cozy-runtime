"""The `int8-convrot256/1` encoded GEMM: the Runtime wheel's prebuilt 8-bit op (`prebuilt`).

The op rotates each activation row by the regular Hadamard per 256 channels and quantizes it per
token in one kernel, then runs an int8 GEMM whose epilogue applies both scales and the bias:
wgmma on Hopper, tcgen05 on datacenter Blackwell, mma.sync on every other card, and a single GEMV
launch for up to 4 rows. Its leaf hands it bound weights (`kernels.readable`). It serves once
it passes its probe on the card (`verify`); where it does not, a leaf serves through
comfy-kitchen's reviewed binary, else eagerly on torch's int8 GEMM, and every run says so once.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import site
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import import_module, util
from pathlib import Path
from types import ModuleType
from typing import cast

from cozy_runtime.author._errors import Fault, fault
from cozy_runtime.author._model import WARNINGS, Building
from cozy_runtime.author.kernels import INT8_CONVROT, PreQuantized
from cozy_runtime.internal import (
    arch,
    kernel_cache,
    kernel_compile,
    kernel_ready,
    kernel_sources,
    prebuilt,
    tools,
)
from cozy_runtime.internal.encoding.formats import CONVROT_GROUP, Tensor, hadamard256

ARTIFACT = "comfy-kitchen"
OP = "cozy-int8"
#: The kernel a leaf serves when neither the op nor comfy-kitchen does (`gemm.selected`).
EAGER = "torch._int_mm"
#: The op's one-launch GEMV takes this many rows (`kGemvRows`): it rotates them itself.
GEMV_ROWS = 4
#: What each card opens, by its `kernel_ready` name's head: the verified op, comfy-kitchen.
NATIVE, KITCHEN = "cozy int8 op", "comfy-kitchen CUTLASS int8 GEMM"

#: The loaded op: `(x, data, scale, bias, out_dtype) -> out`.
type Op = Callable[..., Tensor]

#: Device index -> the verified library serving its 8-bit ops.
_LIBS: dict[int, prebuilt.Prebuilt] = {}
#: Device index -> the loaded op, and the activation dtypes it takes: every call reads these.
_OPS: dict[int, Op] = {}
_HALF: list[object] = []


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
    return tuple(
        Unavailable(OP, int(name.rpartition(":")[2]), f"{type(why).__name__}: {why}"[:300])
        for name, why in sorted(kernel_ready.failures(f"{NATIVE} cuda:").items())
    )


def failed(op: str, device: int, exc: Exception) -> None:
    """`op` raised on this card: its caller serves a slower path for this one call, and the run
    says so, once per reason. A device OOM raises: it is the memory manager's to answer."""
    if fault(exc) in (Fault.OOM, Fault.STICKY, Fault.DEVICE):
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


def _kitchen_tree(sm: int, wait: bool) -> Path | None:
    """comfy-kitchen's reviewed wheel unpacked for this card (`kernel_sources`), None while the
    machine's compile manager places it and `wait` is not set."""
    try:
        job = kernel_sources.job(ARTIFACT, sm)
    except kernel_sources.Absent as exc:
        for tool in exc.tools:
            tools.missing(tool, "kernel_prebuilt_missing")
        raise
    built = (kernel_compile.wait if wait else kernel_compile.submit)(_store(), job)
    if built.state in ("compiling", "pending"):
        return None
    if built.state != "ready":
        raise kernel_sources.Absent(f"{ARTIFACT} sm{sm}: {built.line()}")
    return Path(built.path) / kernel_cache.SITE


def native(torch: ModuleType, index: int) -> Op | None:
    """The prebuilt 8-bit op's int8 linear on this card, once it verifies here (`prebuilt`), off
    any request thread; None until then, and for good where it does not, said once."""
    op = _OPS.get(index)
    if op is not None or torch.cuda.is_current_stream_capturing():  # an open ends a recording
        return op
    ops = kernel_ready.ready(f"{NATIVE} cuda:{index}", partial(_library, torch, index))
    if ops is not None:  # it serves this card from here on
        _HALF[:] = (torch.bfloat16, torch.float16)
        _LIBS[index] = ops
        op = _OPS[index] = ops.int8_linear
    return op


def _library(torch: ModuleType, index: int, wait: bool) -> prebuilt.Prebuilt:
    """`native`'s opener: the library once each format it serves passes its probe here."""
    sm = _sm(torch, index)
    ops, detail = prebuilt.library(torch, index, sm)
    if ops is None:
        print(f"[int8] no {NATIVE} for this executor: {detail}"[:400], file=sys.stderr)
        raise kernel_sources.Absent(detail)
    print(f"[int8] prebuilt kernels verified on sm{sm}: {detail}", file=sys.stderr)
    return ops


def kitchen(torch: ModuleType, index: int) -> ModuleType | None:
    """`comfy_kitchen`'s CUDA backend on this card, or None while its build runs (it starts on
    the first ask) and for good when this card or machine cannot have it."""
    if torch.cuda.is_current_stream_capturing():  # an open would end the graph being recorded
        return cast("ModuleType | None", kernel_ready.ready(f"{KITCHEN} cuda:{index}"))
    return kernel_ready.ready(f"{KITCHEN} cuda:{index}", partial(_kitchen, torch, index))


def _kitchen(torch: ModuleType, index: int, wait: bool) -> ModuleType | None:
    """`kitchen`'s opener: the backend imported, its build waited for when `wait`; said once."""
    sm = _sm(torch, index)
    try:
        if not arch.row(sm).family:
            raise kernel_sources.Absent(f"comfy-kitchen carries no device code for sm{sm}")
        found = util.find_spec("comfy_kitchen")
        store = kernel_cache.configured()
        origin = str(found.origin or "") if found is not None else ""
        roots = () if store is None else (store.own, store.trusted)
        if found is None or origin.startswith(tuple(str(root) for root in roots if root)):
            tree = _kitchen_tree(sm, wait)
            if tree is None:
                return None
            site.addsitedir(str(tree))
        module = import_module("comfy_kitchen.backends.cuda")
    except Exception as exc:  # the next path serves this process
        print(f"[int8] no {KITCHEN} for this executor: {type(exc).__name__}: {exc}"[:400],
              file=sys.stderr)  # fmt: skip
        raise
    print(f"[int8] {KITCHEN} serves sm{sm}", file=sys.stderr)
    return module


def _sm(torch: ModuleType, index: int) -> int:
    major, minor = torch.cuda.get_device_capability(index)
    return int(major * 10 + minor)


def kitchen_serves(index: int) -> bool:
    """Whether comfy-kitchen's CUDA backend serves this card now."""
    return kernel_ready.ready(f"{KITCHEN} cuda:{index}") is not None


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
            if torch.cuda.is_current_stream_capturing() and not unserved(index):
                # comfy-kitchen's GEMM ends a capture (run 10264) and a recording of the eager law
                # replays 5x slower than kitchen as written (run 10287): record once the op lands.
                # An op refused for good never lands: the component runs as written.
                raise Building(OP)
            module = kitchen(torch, index)
            if module is not None:
                return module.int8_linear(
                    x, data, scale, bias, out_dtype, convrot=True, convrot_groupsize=CONVROT_GROUP
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
    return ARTIFACT if kitchen_serves(index) else EAGER


def unserved(index: int) -> bool:
    """Whether the op failed to open on this card: it never serves this process."""
    return kernel_ready.failed(f"{NATIVE} cuda:{index}") is not None


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
    of `consumers` is an un-hooked leaf of x's width taking int8.convrot's operand and the op
    serves this card; else None."""
    if not x.is_cuda or x.dtype not in _HALF or _OPS.get(x.get_device()) is None:
        return None
    if any(t is not None and t.dtype != x.dtype for t in (scale, shift)):
        return None  # eager promotes the rows to the table's dtype
    hooks = torch.nn.modules.module
    if hooks._global_forward_hooks or hooks._global_forward_pre_hooks:
        return None
    for module in consumers:
        if (
            getattr(module, "operand", None) != INT8_CONVROT
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
