"""The prebuilt kernels of the `cozy_kernels` wheel: one wheel per OS and CPU with every card's
cubins inside (`kernels/`), each verified on the device before it serves (`verify`). `_gemm` is
the 8-bit linears; `_fa3` is FlashAttention-3's Hopper forward (`FlashAttention3`).

`Prebuilt` answers the six calls of the op a machine can compile (`int8_op/op.cpp`):
`*_linear` quantizes, multiplies and applies both scales and the bias (int8's few rows in one
GEMV launch), `*_quantize` optionally folds the producer's norm, modulation and activation into
the quantization, and `*_gemm` multiplies a quantized operand. It allocates with torch and
launches on torch's current stream. A GEMM tile that cannot take a shape hands that call to
cuBLASLt (`_int_mm`, `_scaled_mm`), said once; a card the wheel has no fp8 GEMM for (Ada) is
cuBLASLt's by design. `torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from importlib import metadata
from types import ModuleType

from cozy_runtime.internal import arch, verify
from cozy_runtime.internal.encoding.formats import CONVROT_GROUP, E4M3_MAX, Tensor, hadamard256

try:
    from cozy_kernels import FAMILIES, GEMM, _gemm

    ABSENT = ""
except ImportError as absent:  # no cozy-kernels wheel for this OS and CPU
    ABSENT = f"no cozy-kernels wheel is installed ({absent})"
try:
    from cozy_kernels import _fa3

    NO_FA3 = ""
except ImportError as absent:  # Linux aarch64's wheel carries no FA3 (cr-200)
    NO_FA3 = f"the cozy-kernels wheel carries no FlashAttention-3 ({absent})"

#: The verdicts' kernel names, one per format: each verifies on its own.
KERNELS = {"int8": "cozy-kernels-int8", "fp8": "cozy-kernels-fp8"}
#: The probe's budget for the relative L2 error against the exact law (the op tests' bound).
BUDGET = 1e-2
#: Attention's, against fp32 attention: an exact kernel's bf16 output, and a quantized one's
#: (8 or 4 bits): far below the O(1) of garbage, above the measured quantization error.
ATTENTION_BUDGET = {True: 2e-2, False: 0.25}
_SAID: set[str] = set()


@dataclass(frozen=True, slots=True)
class _Format:
    name: str
    fp8: bool
    #: the activation width each row must be a multiple of
    multiple: int


INT8, FP8 = _Format("int8", False, CONVROT_GROUP), _Format("fp8", True, 16)


@dataclass(slots=True)
class Prebuilt:
    """The six ops on one card, for its tensor-core family."""

    torch: ModuleType
    index: int
    family: str
    #: whether the wheel carries a CUTLASS fp8 GEMM for this card
    cutlass_fp8: bool
    _scratch: Callable[[int], Tensor] = field(init=False)

    def __post_init__(self) -> None:
        device = self.torch.device("cuda", self.index)
        self._scratch = lambda size: self.torch.empty(size, dtype=self.torch.uint8, device=device)

    def int8_linear(
        self, x: Tensor, data: Tensor, scale: Tensor, bias: Tensor | None, out_dtype: object
    ) -> Tensor:
        flat = self._rows(INT8, x)
        rows, width = flat.shape
        bias = self._weight(INT8, data, scale, bias, width, out_dtype)
        shape = (*x.shape[:-1], data.shape[0])
        if rows <= _gemm.GEMV_ROWS:
            out = self.torch.empty(shape, dtype=out_dtype, device=x.device)
            code = 0 if rows == 0 else _gemm.gemv(
                self.index, flat.data_ptr(), self._half(flat.dtype), data.data_ptr(),
                scale.data_ptr(), _ptr(bias), out.data_ptr(), self._half(out_dtype), rows,
                data.shape[0], width, self._stream(),
            )  # fmt: skip
            if code != _gemm.NOT_SUPPORTED:
                self._check(code, "the GEMV")
                return out
        q, scales = self.int8_quantize(flat)
        return self.int8_gemm(q, scales, data, scale, bias, out_dtype).view(shape)

    def fp8_linear(
        self, x: Tensor, data: Tensor, scale: Tensor, bias: Tensor | None, out_dtype: object
    ) -> Tensor:
        q, scales = self.fp8_quantize(x)
        shape = (*x.shape[:-1], data.shape[0])
        return self.fp8_gemm(q, scales, data, scale, bias, out_dtype).view(shape)

    def int8_quantize(
        self,
        x: Tensor,
        scale: Tensor | None = None,
        shift: Tensor | None = None,
        norm: int = 0,
        eps: float = 0.0,
        activation: int = 0,
    ) -> tuple[Tensor, Tensor]:
        return self._quantize(INT8, x, scale, shift, norm, eps, activation)

    def fp8_quantize(
        self,
        x: Tensor,
        scale: Tensor | None = None,
        shift: Tensor | None = None,
        norm: int = 0,
        eps: float = 0.0,
        activation: int = 0,
    ) -> tuple[Tensor, Tensor]:
        return self._quantize(FP8, x, scale, shift, norm, eps, activation)

    def int8_gemm(
        self,
        q: Tensor,
        scales: Tensor,
        data: Tensor,
        scale: Tensor,
        bias: Tensor | None,
        out_dtype: object,
        config: int = -1,
    ) -> Tensor:
        return self._gemm(INT8, q, scales, data, scale, bias, out_dtype, config)

    def fp8_gemm(
        self,
        q: Tensor,
        scales: Tensor,
        data: Tensor,
        scale: Tensor,
        bias: Tensor | None,
        out_dtype: object,
        config: int = -1,
    ) -> Tensor:
        return self._gemm(FP8, q, scales, data, scale, bias, out_dtype, config)

    def _quantize(
        self,
        f: _Format,
        x: Tensor,
        scale: Tensor | None,
        shift: Tensor | None,
        norm: int,
        eps: float,
        activation: int,
    ) -> tuple[Tensor, Tensor]:
        torch = self.torch
        flat = self._rows(f, x)
        rows, width = flat.shape
        q = torch.empty(rows, width, dtype=torch.float8_e4m3fn if f.fp8 else torch.int8,
                        device=x.device)  # fmt: skip
        scales = torch.empty(rows, dtype=torch.float32, device=x.device)
        tables = [self._table(t, flat) for t in (scale, shift)]
        entries = next((t.shape[0] for t in tables if t is not None), rows) or 1
        if any(t is not None and t.shape[0] != entries for t in tables) or rows % entries:
            raise ValueError(f"{f.name}_quantize: modulation rows do not divide {rows} rows")
        code = _gemm.quantize(
            self.index, f.fp8, flat.data_ptr(), self._half(flat.dtype), q.data_ptr(),
            scales.data_ptr(), rows, width, _ptr(tables[0]), _ptr(tables[1]),
            _stride(tables[0]), _stride(tables[1]), max(rows // entries, 1), norm, eps,
            activation, self._stream(),
        )  # fmt: skip
        self._check(code, "the quantization")
        return q, scales

    def _gemm(
        self,
        f: _Format,
        q: Tensor,
        scales: Tensor,
        data: Tensor,
        scale: Tensor,
        bias: Tensor | None,
        out_dtype: object,
        config: int,
    ) -> Tensor:
        torch = self.torch
        want = torch.float8_e4m3fn if f.fp8 else torch.int8
        if q.dtype != want or q.dim() != 2 or not q.is_contiguous() or q.data_ptr() % 16:
            raise ValueError(f"{f.name}_gemm: q must be a contiguous, aligned {want} [M, K]")
        rows, width = q.shape
        if scales.dtype != torch.float32 or scales.numel() != rows or not scales.is_contiguous():
            raise ValueError(f"{f.name}_gemm: scales must be contiguous float32 with {rows} values")
        bias = self._weight(f, data, scale, bias, width, out_dtype)
        out = torch.empty(rows, data.shape[0], dtype=out_dtype, device=q.device)
        if rows == 0:
            return out
        code = _gemm.NOT_SUPPORTED
        if not f.fp8 or self.cutlass_fp8:
            code = _gemm.gemm(
                self.index, FAMILIES.index(self.family), f.fp8, q.data_ptr(), scales.data_ptr(),
                data.data_ptr(), scale.data_ptr(), _ptr(bias), out.data_ptr(),
                self._half(out_dtype), rows, data.shape[0], width, config,
                None if f.fp8 else self._scratch, self._stream(),
            )  # fmt: skip
        if code == 0:
            return out
        if code != _gemm.NOT_SUPPORTED:
            self._check(code, "the GEMM")
        if config >= 0:
            raise ValueError(f"{f.name}_gemm: config {config} cannot take {rows}x{width}")
        if f.fp8 and not self.cutlass_fp8:
            pass  # cuBLASLt is this card's fp8 GEMM
        elif f.name not in _SAID:
            _SAID.add(f.name)
            print(f"[{f.name}] the prebuilt GEMM did not run (M={rows} N={data.shape[0]} "
                  f"K={width}); cuBLASLt serves such calls", file=sys.stderr)  # fmt: skip
        if f.fp8:
            product = torch._scaled_mm(
                q, data.t(), scale_a=scales.view(-1, 1), scale_b=scale.view(1, -1),
                out_dtype=out_dtype, use_fast_accum=True,
            )  # fmt: skip
        else:
            padded = torch.nn.functional.pad(q, (0, 0, 0, max(0, 17 - rows)))
            acc = torch._int_mm(padded, data.t())[:rows].float()
            product = acc.mul_(scales.view(-1, 1)).mul_(scale.view(1, -1)).to(out_dtype)
        return product if bias is None else product.add_(bias)

    def _rows(self, f: _Format, x: Tensor) -> Tensor:
        """The activation as contiguous, 16-byte aligned `[M, K]` rows."""
        width = x.shape[-1]
        if not x.is_cuda or x.get_device() != self.index:
            raise ValueError(f"{f.name}: x must be on cuda:{self.index}")
        if width % f.multiple or width > _gemm.MAX_K:
            raise ValueError(f"{f.name}: K must be a multiple of {f.multiple} up to "
                             f"{_gemm.MAX_K}, not {width}")  # fmt: skip
        flat = x.reshape(-1, width).contiguous()
        return flat if flat.data_ptr() % 16 == 0 else flat.clone()

    def _weight(
        self,
        f: _Format,
        data: Tensor,
        scale: Tensor,
        bias: Tensor | None,
        width: int,
        out_dtype: object,
    ) -> Tensor | None:
        torch = self.torch
        want = torch.float8_e4m3fn if f.fp8 else torch.int8
        if data.dtype != want or data.dim() != 2 or data.shape[1] != width:
            raise ValueError(f"{f.name}: data must be {want} [N, {width}]")
        if not data.is_contiguous() or data.data_ptr() % 16 or data.shape[0] % 8:
            raise ValueError(f"{f.name}: data must be contiguous, aligned, N a multiple of 8")
        if scale.dtype != torch.float32 or scale.numel() != data.shape[0]:
            raise ValueError(f"{f.name}: scale must be float32 with {data.shape[0]} values")
        if not scale.is_contiguous():
            raise ValueError(f"{f.name}: scale must be contiguous")
        on = (data, scale) if bias is None else (data, scale, bias)
        if any(t.get_device() != self.index for t in on):
            raise ValueError(f"{f.name}: the weight's tensors must be on cuda:{self.index}")
        self._half(out_dtype)
        if bias is None:
            return None
        if bias.numel() != data.shape[0]:
            raise ValueError(f"{f.name}: bias must have {data.shape[0]} values")
        return bias.to(out_dtype).contiguous()

    def _table(self, table: Tensor | None, flat: Tensor) -> Tensor | None:
        """A modulation table as `[E, K]` rows read 16 bytes at a time."""
        if table is None:
            return None
        if table.device != flat.device or table.dtype != flat.dtype:
            raise ValueError("a modulation table must match the activation's device and dtype")
        found = table.reshape(-1, flat.shape[1])
        if found.stride(1) != 1 or found.stride(0) % 8 or found.data_ptr() % 16:
            found = found.contiguous()
        return found if found.data_ptr() % 16 == 0 else found.clone()

    def _half(self, dtype: object) -> bool:
        if dtype not in (self.torch.bfloat16, self.torch.float16):
            raise ValueError(f"8-bit linear: activations and outputs are bf16 or fp16, not {dtype}")
        return bool(dtype == self.torch.float16)

    def _stream(self) -> int:
        return int(self.torch.cuda.current_stream(self.index).cuda_stream)

    def _check(self, code: int, what: str) -> None:
        if code == 0:
            return
        detail = f"8-bit linear: {what}: {_gemm.error_name(code)}"
        if code == _gemm.OUT_OF_MEMORY:
            raise self.torch.cuda.OutOfMemoryError(detail)
        raise RuntimeError(detail)


def _ptr(t: Tensor | None) -> int:
    return 0 if t is None else int(t.data_ptr())


def _stride(t: Tensor | None) -> int:
    return 0 if t is None else int(t.stride(0))


def artifact(module: str = "_gemm") -> str:
    """An installed module's digest, as its wheel's RECORD has it: nothing is hashed."""
    try:
        files = metadata.distribution("cozy-kernels").files or []
    except metadata.PackageNotFoundError:
        return ""
    found = next((f for f in files if f.name.startswith(f"{module}.") and f.hash), None)
    return f"{found.hash.mode}:{found.hash.value}" if found and found.hash else ""


def covers(sm: int) -> bool:
    """Whether the installed wheel carries this card's 8-bit GEMMs: a machine then builds none at
    boot, and builds them only if they fail their probe here (`int8_linear.native`)."""
    family = arch.row(sm).family
    return not ABSENT and any(sm in GEMM[f].get(family, ()) for f in GEMM)


def library(torch: ModuleType, index: int, sm: int) -> tuple[Prebuilt | None, str]:
    """The prebuilt ops on cuda:`index` (capability `sm`) once each format its row runs passes
    its probe, or None and why. A card no row names tries each family it could hold, newest
    first (`arch`, §2.6): the first whose kernels verify is the card's."""
    if ABSENT:
        return None, ABSENT
    if not arch.families(sm):
        return None, f"sm{sm} has no tensor-core family"
    digest = artifact()
    reasons: list[str] = []
    for family in arch.families(sm):
        row = arch.row(sm, family)
        ops = Prebuilt(torch, index, family, sm in GEMM["fp8"].get(family, ()))
        formats = [f for f in (INT8, FP8) if getattr(row, f.name)]
        verdicts = {
            f.name: verify.verified(
                f"{KERNELS[f.name]}-{family.replace('.', '')}", digest, sm,
                partial(probe, torch, ops, f),
            )
            for f in formats
        }  # fmt: skip
        failed = [f"{name}: {v.detail}" for name, v in verdicts.items() if not v.ok]
        if not failed:
            if row.inherited:
                arch.learn(sm, family)
            return ops, "; ".join(f"{name} {v.detail}" for name, v in verdicts.items())
        reasons.append(f"{family}: {'; '.join(failed)}")
    return None, f"cozy-kernels failed verification on sm{sm}: {' | '.join(reasons)}"


def probe(torch: ModuleType, ops: Prebuilt, f: _Format) -> verify.Verdict:
    """`f`'s kernels on the card against the exact law in float64: the linear's GEMV rows and its
    tiled GEMM in bf16 and fp16, every GEMM tile the card's table holds, and one quantization
    with a producer's RMS norm, modulation and tanh GELU."""
    device = torch.device("cuda", ops.index)
    generator = torch.Generator(device=device).manual_seed(20261017)

    def randn(*shape: int, scale: float = 1.0) -> Tensor:
        return torch.randn(*shape, generator=generator, device=device) * scale

    rows, out_features, width = 300, 512, 1024
    if f.fp8:
        data, scale = _fp8_law(torch, randn(out_features, width, scale=0.02))
        scale = scale.view(-1)
    else:
        data = torch.randint(-127, 128, (out_features, width), generator=generator,
                             device=device, dtype=torch.int8)  # fmt: skip
        scale = randn(out_features).abs_().add_(0.5).mul_(1e-3)
    errors: dict[str, float] = {}
    for dtype in (torch.bfloat16, torch.float16):
        bias = randn(out_features).to(dtype)
        name = "bf16" if dtype == torch.bfloat16 else "fp16"
        for m in (3, rows):
            x = randn(m, width).to(dtype)
            want = _law(torch, f, x.double(), data, scale, bias)
            got = (ops.fp8_linear if f.fp8 else ops.int8_linear)(x, data, scale, bias, dtype)
            errors[f"{name} linear M={m}"] = verify.error(torch, got, want)
        q, scales = (ops.fp8_quantize if f.fp8 else ops.int8_quantize)(x)
        for config in range(8):
            try:
                got = (ops.fp8_gemm if f.fp8 else ops.int8_gemm)(
                    q, scales, data, scale, bias, dtype, config
                )
            except ValueError:  # a tile this card's table does not hold, or this shape
                continue
            errors[f"{name} tile {config}"] = verify.error(torch, got, want)
    x = randn(64, width).to(torch.bfloat16)
    shift, modulation = randn(2, width, scale=0.1), randn(2, width, scale=0.1)
    shift, modulation = shift.to(torch.bfloat16), modulation.to(torch.bfloat16)
    folded = (ops.fp8_quantize if f.fp8 else ops.int8_quantize)(x, modulation, shift, 2, 1e-6, 2)
    got = (ops.fp8_gemm if f.fp8 else ops.int8_gemm)(*folded, data, scale, None, torch.bfloat16)
    produced = _prologue(torch, x.double(), modulation.double(), shift.double(), 1e-6)
    errors["prologue rms+modulation+gelu_tanh"] = verify.error(
        torch, got, _law(torch, f, produced, data, scale, None)
    )
    return verify.within(errors, BUDGET)


def _fp8_law(torch: ModuleType, x: Tensor) -> tuple[Tensor, Tensor]:
    """Per-row E4M3 and its scale (`fp8_linear.eager_quantize`'s law)."""
    scale = (x.abs().amax(-1, keepdim=True).float() / E4M3_MAX).clamp(min=1e-12)
    return (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


def _law(
    torch: ModuleType, f: _Format, x: Tensor, data: Tensor, scale: Tensor, bias: Tensor | None
) -> Tensor:
    """The quantized product in float64: what the kernels compute, up to their output's rounding."""
    if f.fp8:
        q, row = _fp8_law(torch, x)
        q = q.double()
    else:
        rotated = hadamard256(torch, x) / 16
        row = (rotated.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-12)
        q = torch.round(rotated / row).clamp(-127, 127)
    out = (q @ data.double().t()) * row.double() * scale.double().view(1, -1)
    return out if bias is None else out + bias.double()


def _prologue(
    torch: ModuleType, x: Tensor, modulation: Tensor, shift: Tensor, eps: float
) -> Tensor:
    """`gelu_tanh(rms_norm(x) * (1 + modulation) + shift)`, a table row per half of the rows."""
    entry = torch.arange(x.shape[0], device=x.device) // (x.shape[0] // modulation.shape[0])
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    y = y * (1 + modulation[entry]) + shift[entry]
    return 0.5 * y * (1 + torch.tanh(0.79788456 * (y + 0.044715 * y**3)))


@dataclass(frozen=True, slots=True)
class FlashAttention3:
    """`flash_attn_interface.flash_attn_func`'s dense forward over `_fa3`, on one Hopper card:
    `(b, s, h, d)` bf16 or fp16, head dims 64 and 128, K/V heads dividing Q heads, one split."""

    torch: ModuleType
    index: int

    def flash_attn_func(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_splits: int = 1,
    ) -> tuple[Tensor, Tensor]:
        torch = self.torch
        q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))
        batch, seqlen_q, heads, head_dim = q.shape
        if num_splits != 1 or q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("flash-attn3 prebuilt: one split of bf16 or fp16")
        out = torch.empty_like(q)
        lse = torch.empty(batch, heads, seqlen_q, dtype=torch.float32, device=q.device)
        semaphore = torch.zeros(1, dtype=torch.int32, device=q.device) if causal else None
        scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
        code = _fa3.fwd(
            self.index, q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
            lse.data_ptr(), q.dtype == torch.bfloat16, batch, seqlen_q, k.shape[1], heads,
            k.shape[2], head_dim, _strides(q), _strides(k), _strides(v), _strides(out),
            float(scale), causal, _ptr(semaphore),
            int(torch.cuda.current_stream(self.index).cuda_stream),
        )  # fmt: skip
        if code:
            raise ValueError(f"flash-attn3 prebuilt: {_fa3.error_name(code)}")
        return out, lse


def _strides(t: Tensor) -> tuple[int, int, int]:
    return int(t.stride(0)), int(t.stride(1)), int(t.stride(2))


def fa3(torch: ModuleType, index: int, sm: int) -> tuple[FlashAttention3 | None, str]:
    """FlashAttention-3 from the wheel on cuda:`index` once it passes its probe, or None and why."""
    if ABSENT or NO_FA3:
        return None, ABSENT or NO_FA3
    if sm != 90:
        return None, f"the wheel's FlashAttention-3 runs on sm90, not sm{sm}"
    kernel = FlashAttention3(torch, index)

    def call(q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
        return kernel.flash_attn_func(q, k, v, causal=causal)[0]

    pairs = [(d, n) for d in (64, 128) for n in ("bfloat16", "float16")]
    probe = partial(attention_probe, torch, index, call, True, pairs, True)
    verdict = verify.verified("cozy-kernels-fa3", artifact("_fa3"), sm, probe)
    return (kernel if verdict.ok else None), verdict.detail


def attention_probe(
    torch: ModuleType,
    index: int,
    call: Callable[[Tensor, Tensor, Tensor, bool], Tensor],
    exact: bool,
    pairs: Sequence[tuple[int, str]],
    causal: bool,
) -> verify.Verdict:
    """An attention kernel `call(q, k, v, causal) -> out` on `(b, s, h, d)` against attention in
    float64: each (head dim, dtype) it serves, with a query longer than its keys, and causal at
    equal lengths (flash-attention aligns the mask bottom-right) where `causal`."""
    device = torch.device("cuda", index)
    generator = torch.Generator(device=device).manual_seed(20261017)
    errors: dict[str, float] = {}
    for head_dim, name in pairs:
        for masked, lengths in ((False, (300, 257)), *(((True, (257, 257)),) if causal else ())):
            q, k, v = (
                torch.randn(1, s, 2, head_dim, generator=generator, device=device)
                for s in (lengths[0], lengths[1], lengths[1])
            )
            want = torch.nn.functional.scaled_dot_product_attention(
                *(t.transpose(1, 2).double() for t in (q, k, v)), is_causal=masked
            ).transpose(1, 2)
            dtype = getattr(torch, name)
            got = call(q.to(dtype), k.to(dtype), v.to(dtype), masked)
            case = f"d={head_dim} {name}{' causal' if masked else ''}"
            errors[case] = verify.error(torch, got, want)
    return verify.within(errors, ATTENTION_BUDGET[exact])
