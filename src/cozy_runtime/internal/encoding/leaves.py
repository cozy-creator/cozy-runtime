"""cr-006 seam 2 — LEAVES: the providers whose answer is a replaced MODULE, not a tensor.

Seam one's providers write a float destination the contract's dtype named. These do not:
the stored payload and its scales ARE the resident weight, and what changes is the module
that multiplies by them. That is the whole of the `encoded_gemm` route (#515b, #517e), and
it is the only route on which an encoded lane is also a different kernel.

Every leaf here is **w8a8**: fp8 and int8 tensor cores need BOTH operands encoded, so the
activation is quantized at dispatch. That is a real, measured, accepted loss (#517d: 16.72 dB
against the dequant lane's 20.82 dB on the same bytes) bought for the residency a
decoded-float route structurally cannot offer (0.5003x, measured on real H3 block bytes,
#547a). Which side of that trade a deployment wants is what the resolved plan's OBJECTIVE
decides — never registration order, and never a ladder rung.

**Geometry is preflighted here, purely** (#549.5): a shape a leaf's `supports` refuses
resolves to the decode floor at PLAN time with the reason in the plan, instead of reaching
`leaf()` with the component already resident.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from cozy_runtime.author.kernels import PreQuantized, readable
from cozy_runtime.internal import int8_linear, nvfp4_linear
from cozy_runtime.internal.encoding.formats import (
    E4M3_MAX,
    NVFP4_BLOCK,
    SPEC_INT8_CONVROT256,
    SPEC_NVFP4,
    Int8ConvRotDequant,
    Nvfp4Dequant,
    RolePart,
    Tensor,
    _rank2,
)

#: The activation scale's floor, and the same policy `ROW_SCALE_FLOOR` states for weights:
#: a row of exact zeros has amax 0, and dividing by it poisons a whole output row with NaN.
ACT_SCALE_FLOOR = 1e-12


def _triton_rowwise_kernel() -> Any:
    """The FUSED spelling of `quantize_activation_rowwise`, compiled once, lazily.

    cr-086 measured #515f's warning coming true on sm89 at the real Anima DiT shapes: the
    unfused six-pass quantize costs 1.2-5.1 ms around a GEMM that costs 1.0-6.6 ms, so the
    whole w8a8 leaf ran at 0.60-1.14x the bf16 linear it replaced while the fp8 GEMM alone
    ran at 1.6-2.0x. One Triton kernel — a per-row amax reduction and the scaled cast in a
    single launch — does the same arithmetic (same divide, same clamp, same RTNE cast) in
    0.03-0.54 ms and turns the leaf into the speedup the tensor cores promise.

    Same divide (`div_rn`, IEEE round-nearest — triton's bare `/` is an approximate
    reciprocal and measurably drifts), same clamp, same floor. The one honest difference:
    triton's f32->e4m3 downcast double-rounds through f16, so values landing exactly on a
    representational tie pick the other neighbour (~0.25% of elements, one fp8 code) —
    inside the probe's deviation bound, and the probe measures whichever path actually runs.

    Returns the compiled kernel, or None where triton is absent or refuses this device;
    the eager spelling below stays the executable floor.
    """
    cached = getattr(_triton_rowwise_kernel, "_cached", "unset")
    if cached != "unset":
        return cached
    try:
        import triton
        import triton.language as tl

        @triton.jit  # type: ignore[untyped-decorator]
        def _kernel(  # type: ignore[no-untyped-def]
            x_ptr, out_ptr, scale_ptr, columns, stride_x, stride_out, BLOCK: tl.constexpr
        ):
            # int64: `row * stride` passes 2**31 elements at H3's 15 s + 56-frame continuation
            # (~155k rows into the 14336-wide FFN out projection) and wrapped to a negative offset.
            row = tl.program_id(0).to(tl.int64)
            offsets = tl.arange(0, BLOCK)
            amax = tl.zeros((BLOCK,), dtype=tl.float32)
            for start in range(0, columns, BLOCK):
                index = start + offsets
                value = tl.load(x_ptr + row * stride_x + index, mask=index < columns, other=0.0).to(
                    tl.float32
                )
                amax = tl.maximum(amax, tl.abs(value))
            scale = tl.maximum(tl.max(amax, axis=0) / 448.0, 1e-12)
            tl.store(scale_ptr + row, scale)
            for start in range(0, columns, BLOCK):
                index = start + offsets
                value = tl.load(x_ptr + row * stride_x + index, mask=index < columns, other=0.0).to(
                    tl.float32
                )
                quantized = tl.clamp(tl.div_rn(value, scale), -448.0, 448.0)
                tl.store(
                    out_ptr + row * stride_out + index,
                    quantized.to(tl.float8e4nv, fp_downcast_rounding="rtne"),
                    mask=index < columns,
                )

    except Exception:
        _kernel = None
    _triton_rowwise_kernel._cached = _kernel  # type: ignore[attr-defined]
    return _kernel


def quantize_activation_rowwise(torch: Any, x: Any) -> tuple[Any, Any]:
    """`[..., K]` float -> `[M, K]` E4M3 payload + `[M, 1]` f32 per-TOKEN scale.

    v1's proven recipe (`gen_worker/models/w8a8.py::_quant_src`), transcribed: amax over the
    reduction axis, `.float()` BEFORE the divide so the scale is f32 whatever the activation
    is, the literal 448.0 rather than `finfo.max`, a floor, and the clamp BEFORE the cast
    because the cast does not saturate. One scale per ROW of the `[M, K]` operand is what
    pairs with a per-output-row weight scale: the two scale vectors are the two OUTER
    dimensions of the product and neither is the reduction axis.

    #515f named the unfused cost honestly — six elementwise passes over `[M, K]` around a
    GEMM costing `M*K*N`, disappearing only "when the quantize is fused into one kernel" —
    and `_triton_rowwise_kernel` is that fusion (cr-086): one launch, deviation noted on
    its docstring. A CUDA operand takes it when triton built; everything else (CPU probes,
    a venv without triton, a launch failure) runs the eager passes below and stays correct.
    """
    if isinstance(x, PreQuantized):
        return x.payload, x.scale
    flat = x.reshape(-1, x.shape[-1])
    if flat.is_cuda and flat.numel() and flat.stride(-1) == 1:
        kernel = _triton_rowwise_kernel()
        if kernel is not None:
            rows, columns = int(flat.shape[0]), int(flat.shape[1])
            out = torch.empty(rows, columns, dtype=torch.float8_e4m3fn, device=flat.device)
            scale = torch.empty(rows, 1, dtype=torch.float32, device=flat.device)
            try:
                with readable(flat) as (flat,):
                    kernel[(rows,)](
                        flat, out, scale, columns, flat.stride(0), columns, BLOCK=1024, num_warps=8
                    )
                return out, scale
            except Exception:
                # One honest disable: a device triton cannot serve falls to eager for the
                # life of the process rather than re-raising per forward.
                _triton_rowwise_kernel._cached = None  # type: ignore[attr-defined]
    amax = flat.abs().amax(dim=-1, keepdim=True).float()
    scale = (amax / E4M3_MAX).clamp(min=ACT_SCALE_FLOOR)
    return (flat.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), scale


def _blocked_shape(rows: int, cols: int) -> tuple[int, int]:
    return -(-rows // 128) * 128, -(-cols // 4) * 4


def to_blocked(torch: Any, scales: Any) -> Any:
    """`[rows, cols]` block-scale bytes -> cuBLASLt's 128x4 blocked ("swizzled") layout.

    A silent-wrong-output hazard rather than an optimisation (#515d): a block-scaled GEMM checks
    the scale grid's ELEMENT COUNT, not its LAYOUT, so a plain row-major grid is accepted where
    the tiled one belongs and returns wrong numbers almost everywhere. Nothing raises; the probe
    oracle is what catches a builder who skips this.
    """
    rows, cols = int(scales.shape[0]), int(scales.shape[1])
    padded_rows, padded_cols = _blocked_shape(rows, cols)
    row_blocks, col_blocks = padded_rows // 128, padded_cols // 4
    padded = torch.zeros(padded_rows, padded_cols, dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    tiles = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def from_blocked(torch: ModuleType, blocked: Tensor, rows: int, cols: int) -> Tensor:
    """`to_blocked`'s inverse: the `[rows, cols]` row-major grid of a blocked scale layout."""
    padded_rows, padded_cols = _blocked_shape(rows, cols)
    row_blocks, col_blocks = padded_rows // 128, padded_cols // 4
    tiles = blocked.reshape(-1, 32, 4, 4).transpose(1, 2).reshape(row_blocks, col_blocks, 128, 4)
    return tiles.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)[:rows, :cols]


def _leaf_class(torch: Any) -> Any:
    """The `nn.Module` subclass every encoded leaf is an instance of, built ONCE, lazily.

    Built inside a function because this module imports no torch — the whole runtime's
    torch-free-import fence — and because `nn.Module.__setattr__` REFUSES any child that is
    not a Module, which is precisely why leaf replacement is machinery and not a swap: the
    payload cannot simply be assigned where the float weight was.

    The roles are BUFFERS, never Parameters. An fp8 Parameter is hostile to every
    `.to(dtype)` that walks a module (v1's `_Fp8ScaledLinear` carries the same rule for the
    same reason), and a buffer is what `fill.py`'s `_tensors` census, its VRAM charge and
    its eviction already walk — so the resident bytes stay ledgered by the machinery that
    was already counting them.
    """
    cached = getattr(_leaf_class, "_cached", None)
    if cached is not None:
        return cached
    from torch import nn

    class EncodedLinear(nn.Module):  # type: ignore[misc]
        """One linear op whose WEIGHT IS THE ENCODED ROLE SET, computed by an encoded GEMM.

        It stands exactly where the module it replaced stood, takes exactly that module's
        input and returns exactly that module's output dtype and shape. What it does not
        have is a `.weight`: a package that reaches for one is why author CONSENT is a
        precondition of this route rather than a courtesy.
        """

        def __init__(
            self,
            forward_fn: Any,
            roles: Mapping[str, Any],
            bias: Any,
            out_dtype: Any,
            provider: str,
            encoding: str,
            in_features: int,
            out_features: int,
        ) -> None:
            super().__init__()
            # Upstream projections inspect the first floating parameter/buffer
            # before casting activations. Encoded storage is not that logical
            # dtype; preserve the replaced projection's compute contract without
            # adding a parameter, persisted weight, or resident payload bytes.
            self.register_buffer(
                "_logical_dtype",
                next(iter(roles.values())).new_empty(0, dtype=out_dtype),
                persistent=False,
            )
            self._cozy_forward = forward_fn
            self._cozy_roles = tuple(sorted(roles))
            for role, tensor in sorted(roles.items()):
                self.register_buffer(role, tensor)
            self.register_buffer("bias", bias)
            self.out_dtype = out_dtype
            self.provider = provider
            self.encoding = encoding
            self.in_features = in_features
            self.out_features = out_features

        def roles(self) -> dict[str, Any]:
            return {role: getattr(self, role) for role in self._cozy_roles}

        def forward(self, x: Any, *, prequantized: PreQuantized | None = None) -> Any:
            # Semantic hooks still see the original positional float operand.
            if prequantized is None and torch.is_autocast_enabled(x.device.type):
                # As the replaced `nn.Linear` does under autocast: compute from, and return,
                # the autocast dtype, whatever the stored weight's logical dtype is.
                dtype = torch.get_autocast_dtype(x.device.type)
                out = self._cozy_forward(self, x.to(dtype), dtype)
            else:
                operand = x if prequantized is None else prequantized
                out = self._cozy_forward(self, operand, self.out_dtype)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"provider={self.provider!r}, roles={list(self._cozy_roles)}, "
                f"out_dtype={self.out_dtype}"
            )

    _leaf_class._cached = EncodedLinear  # type: ignore[attr-defined]
    return EncodedLinear


def _leaf_geometry(replaced: Any, data: Any) -> tuple[int, int, Any]:
    """`(in_features, out_features, bias)` of the module being stood in for.

    Read off the REPLACED module rather than inferred from the payload, because the payload
    is `[out, in]` for a Linear and the runtime must not assume every consumer of an encoded
    weight is one. A module whose declared geometry disagrees with the stored payload is a
    refusal in `fill.py`, not a reshape here.
    """
    out_features, in_features = int(data.shape[0]), int(data.shape[1])
    bias = getattr(replaced, "bias", None)
    return in_features, out_features, bias


@dataclass(frozen=True, slots=True)
class RowwiseNativeLeaf:
    """`fp8-rowwise/1` ENCODED GEMM: the tensor cores consume the stored E4M3 payload directly.

    #514b's engineering fact decides the shape: fp8 tensor cores need BOTH operands fp8, so
    the serving form is necessarily **w8a8** — the weight is stored quantized and the
    ACTIVATION is quantized at dispatch, per token, into the scale geometry that pairs with
    the weight's per-output-row one. That is not weight-only quantization and this docstring
    does not pretend it is: #517d measured the derived-w8a8 lane at 16.72 dB against the
    dequant-bf16 lane's 20.82 dB on identical bytes, of which 1.48-1.63 dB is the activation
    quantization alone. What it buys is the other half of the same measurement: 0.5003x the
    resident VRAM, because the payload IS the weight.

    **`use_fast_accum=True` is not a tuning knob here, it is the difference between winning
    and losing** (#517b): measured on an L40S, the default accumulate path runs the fp8 GEMM
    at 0.52x bf16 and the fast one at 2.03x, for -0.007 to -0.018 dB. v1 never passes the
    flag and banked its own lane losing 0.81-0.93x to bf16 on the same shapes.

    **The device predicate is a PROFITABILITY floor stated honestly.** #517c measured rowwise
    `_scaled_mm` WORKING on sm89, which makes v1's `W8A8_ROWWISE_MIN_SM=90` a profitability
    floor mislabelled as a capability floor. The floor here is `cuda.sm89+` because that is
    where the kernel exists; whether it is PROFITABLE on a given card is what the probe
    measures, and a card no probe ran on mints no record whatever this floor says.
    """

    encoding: str

    @property
    def name(self) -> str:
        return "cozy.fp8-rowwise.native-leaf/1"

    @property
    def route(self) -> str:
        return "encoded_gemm"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm89+"

    @property
    def reviewed_specs(self) -> frozenset[str]:
        from cozy_runtime.internal.encoding.formats import SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM

        return frozenset({SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "fp8-rowwise-native"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"data", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    @property
    def output_dtype_rule(self) -> str:
        return "contract"

    def supports(self, shape: Sequence[int]) -> str:
        """Rank 2, and NOTHING ELSE IS CLAIMED (#549.5).

        A per-output-row scale has no block structure, so there is no divisibility law to
        state here, unlike a block-scaled format. `torch._scaled_mm` carries alignment
        preferences of its own on some arches; NOTHING IN THIS FLEET HAS MEASURED THEM, and
        a preflight that refused on an unmeasured guess would be exactly the hand-written
        floor #511b forbids. A shape the kernel actually rejects raises inside the probe's
        `_native_agreement`, mints no record, and falls to the decode floor — measured, not
        guessed.
        """
        return _rank2(shape)

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        raise AssertionError(
            "an encoded_gemm provider REPLACES the leaf; it never writes a float "
            "destination — see LeafProvider.leaf"
        )

    def resident_bytes(self, parts: Mapping[str, RolePart]) -> int:
        return sum(p.nbytes for p in parts.values())

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float:
        return 8.0 * self.resident_bytes(parts) / max(numel, 1)

    def fill_scratch_bytes(self, parts: Mapping[str, RolePart]) -> int:
        return 0

    @staticmethod
    def _forward(leaf: Any, x: Any, out_dtype: Any) -> Any:
        import torch

        payload, scale = leaf.data, leaf.scale
        xq, xs = quantize_activation_rowwise(torch, x)
        # `_scaled_mm` wants the second operand COLUMN-major, which `payload.t()` already is
        # for a contiguous `[out, in]`. `scale_b` is per COLUMN of that operand, which is per
        # OUTPUT row of the weight — and `reshape(1, -1)` is what serves the alias's rank-1
        # `[out]` and rank-2 `[out, 1]` variants through one spelling.
        out = torch._scaled_mm(
            xq,
            payload.t(),
            scale_a=xs,
            scale_b=scale.to(torch.float32).reshape(1, -1),
            out_dtype=out_dtype,
            use_fast_accum=True,
        )
        return out.reshape(*x.shape[:-1], -1)

    def leaf(self, torch: Any, parts: Mapping[str, Any], replaced: Any, out_dtype: Any) -> Any:
        data = parts["data"]
        in_features, out_features, bias = _leaf_geometry(replaced, data)
        return _leaf_class(torch)(
            RowwiseNativeLeaf._forward,
            {"data": data, "scale": parts["scale"]},
            bias,
            out_dtype,
            self.name,
            self.encoding,
            in_features,
            out_features,
        )


@dataclass(frozen=True, slots=True)
class Int8ConvRotNativeLeaf:
    """`int8-convrot256/1` ENCODED GEMM: W8A8 on the int8 tensor cores (`int8_linear`).

    The weight rows were rotated by the orthonormal Hadamard per 256 input channels before
    they were quantized, so each activation row is rotated by the same H at dispatch and
    quantized per token; the rotation cancels in the product. On Qwen-Image-2.1's real
    activations that rotation cuts the per-linear error of W8A8 from 1.7-4.4% to 0.5-1.4%
    (weight-only int8: 0.5-1.0%). On consumer cards int8 runs at 2x their fp8 rate with fp32
    accumulation and 3-4x their bf16 rate, so this is the fast lane there.
    """

    encoding: str

    @property
    def name(self) -> str:
        return "cozy.int8-convrot256.native-leaf/1"

    @property
    def route(self) -> str:
        return "encoded_gemm"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm80+"

    @property
    def reviewed_specs(self) -> frozenset[str]:
        return frozenset({SPEC_INT8_CONVROT256})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "int8-convrot256-native"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"data", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    @property
    def output_dtype_rule(self) -> str:
        return "contract"

    def supports(self, shape: Sequence[int]) -> str:
        """The decode's group law, and cuBLASLt's 8-aligned N for the eager int8 GEMM."""
        why = Int8ConvRotDequant("").supports(shape)
        if why or int(shape[0]) % 8 == 0:
            return why
        return f"the int8 GEMM needs N divisible by 8 and this tensor has {shape[0]} rows"

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        raise AssertionError(
            "an encoded_gemm provider REPLACES the leaf; it never writes a float "
            "destination — see LeafProvider.leaf"
        )

    def resident_bytes(self, parts: Mapping[str, RolePart]) -> int:
        return sum(p.nbytes for p in parts.values())

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float:
        return 8.0 * self.resident_bytes(parts) / max(numel, 1)

    def fill_scratch_bytes(self, parts: Mapping[str, RolePart]) -> int:
        return 0

    @staticmethod
    def _forward(leaf: Tensor, x: Tensor, out_dtype: Tensor) -> Tensor:
        import torch

        return int8_linear.matmul(torch, x, leaf.data, leaf.scale, out_dtype)

    def leaf(
        self, torch: ModuleType, parts: Mapping[str, Tensor], replaced: Tensor, out_dtype: Tensor
    ) -> Tensor:
        if torch.cuda.is_available():
            # Starts the kernel's build beside the weight fill; the first calls run eager.
            int8_linear.kitchen(torch, torch.cuda.current_device())
        data = parts["data"]
        in_features, out_features, bias = _leaf_geometry(replaced, data)
        return _leaf_class(torch)(
            Int8ConvRotNativeLeaf._forward,
            {"data": data, "scale": parts["scale"]},
            bias,
            out_dtype,
            self.name,
            self.encoding,
            in_features,
            out_features,
        )


@dataclass(frozen=True, slots=True)
class Nvfp4NativeLeaf:
    """`nvfp4-w4a4/1` ENCODED GEMM: the stored E2M1 payload and E4M3 block scales stay resident
    (0.56 bytes per weight) and are multiplied as they are (`nvfp4_linear`).

    On Blackwell a product of more than 16 rows is W4A4 on the FP4 tensor cores: each activation is
    quantized at dispatch against its own tensor amax, and on real FLUX.3 Klein activations that is
    5-10% from bf16 per linear (int8-convrot: 0.5-1.2%) at 1.3-1.5x the int8 leaf's rate on an RTX
    5090. Every other product, and every one before Blackwell, is W4A16: Triton decodes the weight
    in registers onto the bf16 tensor cores, which is the bandwidth-bound decode loop of an LM.
    """

    encoding: str

    @property
    def name(self) -> str:
        return "cozy.nvfp4.native-leaf/1"

    @property
    def route(self) -> str:
        return "encoded_gemm"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm80+"

    @property
    def reviewed_specs(self) -> frozenset[str]:
        return frozenset({SPEC_NVFP4})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "nvfp4-native"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "weight_scale", "weight_scale_2"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    @property
    def output_dtype_rule(self) -> str:
        return "contract"

    def supports(self, shape: Sequence[int]) -> str:
        """The decode's block law, and cuBLASLt's FP4 GEMM: K divisible by 32, N by 8."""
        why = Nvfp4Dequant("").supports(shape)
        if why:
            return why
        if int(shape[1]) % 32 or int(shape[0]) % 8:
            return f"the FP4 GEMM needs K % 32 == 0 and N % 8 == 0; this tensor is {list(shape)}"
        return ""

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        raise AssertionError(
            "an encoded_gemm provider REPLACES the leaf; it never writes a float "
            "destination — see LeafProvider.leaf"
        )

    def resident_bytes(self, parts: Mapping[str, RolePart]) -> int:
        rows, cols = _blocked_shape(*parts["weight_scale"].shape)
        return parts["weight"].nbytes + rows * cols + 4

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float:
        return 8.0 * self.resident_bytes(parts) / max(numel, 1)

    def fill_scratch_bytes(self, parts: Mapping[str, RolePart]) -> int:
        # As the MX leaf's: to_blocked's padded grid, reshape copy and final blocked copy.
        rows, cols = _blocked_shape(*parts["weight_scale"].shape)
        return 3 * rows * cols

    @staticmethod
    def _forward(leaf: Tensor, x: Tensor) -> Tensor:
        import torch

        rows, cols = leaf.out_features, leaf.in_features // NVFP4_BLOCK
        return nvfp4_linear.matmul(
            torch,
            x,
            leaf.weight,
            leaf.weight_scale,
            lambda: from_blocked(torch, leaf.weight_scale, rows, cols),
            leaf.weight_scale_2,
            leaf.out_dtype,
        )

    def leaf(
        self, torch: ModuleType, parts: Mapping[str, Tensor], replaced: Tensor, out_dtype: Tensor
    ) -> Tensor:
        if torch.cuda.is_available():
            # Starts the kernel's build beside the weight fill; the first calls run eager.
            int8_linear.kitchen(torch, torch.cuda.current_device())
        weight = parts["weight"]
        out_features, in_features = int(weight.shape[0]), 2 * int(weight.shape[1])
        rows, cols = _blocked_shape(out_features, in_features // NVFP4_BLOCK)
        # Swizzled once, at fill: the weight's grid never changes.
        blocked = to_blocked(torch, parts["weight_scale"].view(torch.uint8))
        blocked = blocked.view(rows, cols).view(torch.float8_e4m3fn)
        return _leaf_class(torch)(
            Nvfp4NativeLeaf._forward,
            {
                "weight": weight,
                "weight_scale": blocked,
                "weight_scale_2": parts["weight_scale_2"].to(torch.float32).reshape(1),
            },
            getattr(replaced, "bias", None),
            out_dtype,
            self.name,
            self.encoding,
            in_features,
            out_features,
        )
