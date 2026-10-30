"""The `int8-convrot256/1` encoded GEMM: the Runtime wheel's prebuilt 8-bit op (`prebuilt`).

The op rotates each activation row by the regular Hadamard per 256 channels and quantizes it per
token in one kernel, then runs an int8 GEMM whose epilogue applies both scales and the bias:
wgmma on Hopper, tcgen05 on datacenter Blackwell, mma.sync on every other card, and a single GEMV
launch for up to 4 rows. Its leaf hands it bound weights (`kernels.readable`). It serves once
it passes its probe on the card (`verify`); where it does not, a leaf serves through
comfy-kitchen's reviewed binary once that passes its own probe, else eagerly on torch's int8
GEMM, and every run says so once. Each opens through `kernel_ready.serve`.

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
    verify,
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
_SM: dict[int, int] = {}


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


def _kitchen_tree(sm: int, wait: bool) -> tuple[Path, str] | None:
    """comfy-kitchen's reviewed wheel unpacked for this card (`kernel_sources`) and its build's
    key, None while the machine's compile manager places it and `wait` is not set."""
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
    return Path(built.path) / kernel_cache.SITE, job.key.digest


def native(torch: ModuleType, index: int) -> Op | None:
    """The prebuilt 8-bit op's int8 linear on this card, once it verifies here (`prebuilt`), off
    any request thread; None until then, and for good where it does not, said once. Raises
    `Building` while a graph records before it serves (`kernel_ready.serve`)."""
    op = _OPS.get(index)
    if op is not None:
        return op
    capturing = torch.cuda.is_current_stream_capturing()
    ops = kernel_ready.serve(f"{NATIVE} cuda:{index}", partial(_library, torch, index), capturing)
    if ops is not None:  # it serves this card from here on
        _HALF[:] = (torch.bfloat16, torch.float16)
        _LIBS[index] = ops
        op = _OPS[index] = ops.int8_linear
    return op


def prime(torch: ModuleType, index: int) -> None:
    """Opens the op on this card on its own thread at device init, beside the weight fill: its
    first ask (`native`) waits for it rather than serve a fallback (`kernel_ready.prime`)."""
    kernel_ready.prime(f"{NATIVE} cuda:{index}", partial(_library, torch, index))


def _library(torch: ModuleType, index: int, wait: bool) -> prebuilt.Prebuilt:
    """`native`'s opener: the library once each format it serves passes its probe here."""
    sm = capability(torch, index)
    ops, detail = prebuilt.library(torch, index, sm)
    if ops is None:
        print(f"[kernels] {NATIVE} refused on sm{sm}: {detail}"[:400], file=sys.stderr)
        raise kernel_sources.Absent(detail)
    print(f"[kernels] {NATIVE} verified on sm{sm}: {detail}"[:400], file=sys.stderr)
    return ops


def kitchen(torch: ModuleType, index: int) -> ModuleType | None:
    """`comfy_kitchen`'s CUDA backend on this card once its int8 GEMM passes its probe here, or
    None while its build runs (it starts on the first ask) and for good when this card or
    machine cannot have it (`kernel_ready.serve`)."""
    capturing = torch.cuda.is_current_stream_capturing()
    return kernel_ready.serve(f"{KITCHEN} cuda:{index}", partial(_kitchen, torch, index), capturing)


def kitchen_module(torch: ModuleType, index: int, wait: bool) -> tuple[ModuleType, str] | None:
    """comfy-kitchen's CUDA backend imported and what identifies its bytes, its build waited
    for when `wait`: the start of each of its GEMMs' openers."""
    sm = capability(torch, index)
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
            site.addsitedir(str(tree[0]))
            origin = tree[1]
        return import_module("comfy_kitchen.backends.cuda"), origin
    except Exception as exc:  # the next path serves this process
        print(f"[kernels] no {ARTIFACT} on sm{sm}: {type(exc).__name__}: {exc}"[:400],
              file=sys.stderr)  # fmt: skip
        raise


def _kitchen(torch: ModuleType, index: int, wait: bool) -> ModuleType | None:
    """`kitchen`'s opener: the backend, once its int8 GEMM agrees with the exact law here."""
    found = kitchen_module(torch, index, wait)
    if found is None:
        return None
    module, artifact = found
    sm, probe = capability(torch, index), partial(_probe_kitchen, torch, index)
    return kernel_ready.verified(module, "comfy-kitchen-int8", artifact, sm, probe)


def _probe_kitchen(torch: ModuleType, index: int, module: ModuleType) -> verify.Verdict:
    """comfy-kitchen's int8.convrot linear against the unquantized product in float64 (the
    rotated rows times the stored rotated weight): it rounds the activation its own way, so it
    may lose `verify.SLACK` times what the exact law does (`prebuilt.law`). bf16 and fp16, a few
    rows and a GEMM's, a ragged N."""
    device = torch.device("cuda", index)
    generator = torch.Generator(device=device).manual_seed(20261025)
    out_features, width = 520, 1024
    data = torch.randint(-127, 128, (out_features, width), generator=generator, device=device,
                         dtype=torch.int8)  # fmt: skip
    scale = torch.rand(out_features, generator=generator, device=device, dtype=torch.float32).add_(0.5).mul_(1e-3)
    weight = data.double() * scale.double()[:, None]
    errors: dict[str, tuple[float, float]] = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        bias = torch.randn(out_features, generator=generator, device=device, dtype=torch.float32).to(dtype)
        for m in (3, 300):
            x = torch.randn(m, width, generator=generator, device=device, dtype=torch.float32).to(dtype)
            got = module.int8_linear(
                x, data, scale, bias, dtype, convrot=True, convrot_groupsize=CONVROT_GROUP
            )
            law = prebuilt.law(torch, prebuilt.INT8, x.double(), data, scale, bias)
            exact = hadamard256(torch, x.double()) / 16 @ weight.t() + bias.double()
            case = f"{name} M={m}"
            errors[case] = (verify.error(torch, got, exact), verify.error(torch, law, exact))
    return verify.against_law(errors)


def capability(torch: ModuleType, index: int) -> int:
    """cuda:`index`'s compute capability, e.g. 89."""
    if index not in _SM:
        major, minor = torch.cuda.get_device_capability(index)
        _SM[index] = int(major * 10 + minor)
    return _SM[index]


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
            # While a graph records, both raise `Building` until they serve: comfy-kitchen's GEMM
            # ends a capture (run 10264) and a recording of the eager law replays 5x slower than
            # kitchen as written (run 10287).
            op = native(torch, index)
            if op is not None:
                return op(x, data, scale, bias, out_dtype)
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
ACTIVATIONS = {None: 0, "gelu": 1, "gelu_tanh": 2, "situ": 3, "swiglu": 4}
#: The gated activations: x holds [hidden | gate] halves, and the operand is half its width.
GATED = frozenset({"situ"})


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
    up: Tensor | None = None,
) -> PreQuantized | None:
    """`activation(norm(x) * (1 + scale) + shift)`, rotated and quantized in one pass, when each
    of `consumers` is an un-hooked leaf of the rows' width taking int8.convrot's operand and the
    op serves this card; else None. A gated activation (`GATED`) reads x's [hidden | gate] halves
    alone: the rows are half x's width. `swiglu` reads x as the gate rows and `up` as the up rows
    of the same shape, alone: `silu(x) * up` as an eager SwiGLU rounds it."""
    gated = activation in GATED
    if gated and (norm is not None or scale is not None or shift is not None or x.shape[-1] % 2):
        raise ValueError(f"int8_rows: {activation} takes [hidden | gate] rows, no norm or tables")
    if (activation == "swiglu") != (up is not None) or (
        up is not None and (norm is not None or scale is not None or shift is not None)
    ):
        raise ValueError("int8_rows: swiglu takes gate and up rows, no norm or tables")
    if up is not None and (up.shape != x.shape or up.dtype != x.dtype or not up.is_cuda):
        return None
    width = x.shape[-1] // 2 if gated else x.shape[-1]
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
            or getattr(module, "in_features", None) != width
            or module._forward_hooks  # type: ignore[attr-defined]
            or module._forward_pre_hooks  # type: ignore[attr-defined]
        ):
            return None
    try:
        payload, row_scale = _LIBS[x.get_device()].int8_quantize(
            x, scale, shift, NORMS[norm], eps, ACTIVATIONS[activation], up
        )
    except Exception as exc:
        failed(OP, x.get_device(), exc)
        return None
    return PreQuantized(payload, row_scale, (*x.shape[:-1], width), INT8_CONVROT)


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
    quantized, row_scale = quantize_rows(torch, x)
    out = _product(torch, quantized, row_scale, data, scale, out_dtype)
    return out.reshape(*x.shape[:-1], -1)


def quantize_rows(torch: ModuleType, x: Tensor) -> tuple[Tensor, Tensor]:
    """`x`'s rows rotated (H/16 per 256 channels) and quantized per row in torch ops: int8
    `[M, K]` and their f32 scales `[M, 1]`, the operand `eager` multiplies."""
    rotated = hadamard256(torch, x.reshape(-1, x.shape[-1]).float()).mul_(1.0 / 16)
    row_scale = (rotated.abs().amax(-1, keepdim=True) / 127.0).clamp_(min=1e-12)
    quantized = torch.round(rotated / row_scale).clamp_(-127, 127).to(torch.int8)
    return quantized, row_scale


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
