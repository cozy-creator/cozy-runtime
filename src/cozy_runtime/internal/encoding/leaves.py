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

from cozy_runtime.author.kernels import (
    INT8_CONVROT,
    ROWWISE_FP8,
    PreQuantized,
    shared_operand,
)
from cozy_runtime.internal import fp8_linear, int8_linear, nvfp4_linear
from cozy_runtime.internal.encoding.formats import (
    NVFP4_BLOCK,
    SPEC_INT8_CONVROT256,
    SPEC_NVFP4,
    Int8ConvRotDequant,
    Nvfp4Dequant,
    RolePart,
    Tensor,
    _rank2,
)


def _blocked_shape(rows: int, cols: int) -> tuple[int, int]:
    return -(-rows // 128) * 128, -(-cols // 4) * 4


def to_blocked(torch: ModuleType, scales: Tensor) -> Tensor:
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


#: The 8-bit leaves' ops, by provider: each module's `quantize`, `gemm` and `linear`.
_NATIVE_OPS: dict[str, ModuleType] = {INT8_CONVROT: int8_linear, ROWWISE_FP8: fp8_linear}


def _native_leaf_class(torch: ModuleType) -> type:
    """`EncodedLinear` whose forward is one call of its provider's op (`int8_linear`,
    `fp8_linear`), the bias in the GEMM's epilogue."""
    cached: type | None = getattr(_native_leaf_class, "_cached", None)
    if cached is not None:
        return cached

    class NativeLinear(_leaf_class(torch)):  # type: ignore[misc]
        def forward(
            self, x: Tensor | PreQuantized, *, prequantized: PreQuantized | None = None
        ) -> Tensor:
            ops = _NATIVE_OPS[self.provider]
            out_dtype = self.out_dtype
            if isinstance(x, PreQuantized):
                # A producer's prologue (`kernels.int8_rows`) hands the operand in place of x.
                if torch.is_autocast_enabled(x.payload.device.type):
                    out_dtype = torch.get_autocast_dtype(x.payload.device.type)
                return ops.gemm(torch, x, self.data, self.scale, self.bias, out_dtype)
            cast = x
            if torch.is_autocast_enabled(x.device.type):
                # As the replaced `nn.Linear` does: from and to the autocast dtype.
                out_dtype = torch.get_autocast_dtype(x.device.type)
                cast = x.to(out_dtype)
            if prequantized is None or prequantized.provider != self.provider:
                prequantized = shared_operand(x, self.provider, lambda: ops.quantize(torch, cast))
            if prequantized is not None:
                return ops.gemm(torch, prequantized, self.data, self.scale, self.bias, out_dtype)
            return ops.linear(torch, cast, self.data, self.scale, self.bias, out_dtype)

        def int8_prologue(
            self,
            x: Tensor,
            consumers: tuple[object, ...],
            norm: str | None,
            eps: float,
            scale: Tensor | None,
            shift: Tensor | None,
            activation: str | None,
        ) -> PreQuantized | None:
            """`kernels.int8_rows` for consumers led by this leaf."""
            return int8_linear.prologue(
                torch,
                x,
                consumers,
                norm=norm,
                eps=eps,
                scale=scale,
                shift=shift,
                activation=activation,
            )

    _native_leaf_class._cached = NativeLinear  # type: ignore[attr-defined]
    return NativeLinear


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
    """`fp8-rowwise/1` ENCODED GEMM: the tensor cores consume the stored E4M3 payload directly
    (`fp8_linear`).

    fp8 tensor cores need BOTH operands fp8, so this is w8a8 (#514b): the activation is
    quantized per token at dispatch, into the scale geometry that pairs with the weight's
    per-output-row one. #517d measured that at 16.72 dB against the dequant-bf16 lane's 20.82
    dB on identical bytes, 1.48-1.63 dB of it the activation quantization; it buys 0.5003x the
    resident VRAM, because the payload IS the weight. The accumulation is the fast one, as
    `_scaled_mm(use_fast_accum=True)`'s: on an L40S the default ran 0.52x bf16 and the fast
    one 2.03x, for -0.007 to -0.018 dB (#517b).

    The floor `cuda.sm89+` is where an fp8 tensor-core GEMM exists; whether it pays on a card
    is what the probe measures.
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
        """Rank 2, and the fp8 GEMMs' 16-byte rows: K and N divisible by 16."""
        why = _rank2(shape)
        if why or (int(shape[0]) % 16 == 0 and int(shape[1]) % 16 == 0):
            return why
        return f"the fp8 GEMM needs K and N divisible by 16; this tensor is {list(shape)}"

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

    def leaf(self, torch: Any, parts: Mapping[str, Any], replaced: Any, out_dtype: Any) -> Any:
        data = parts["data"]
        in_features, out_features, bias = _leaf_geometry(replaced, data)
        return _native_leaf_class(torch)(
            None,
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

    def leaf(
        self, torch: ModuleType, parts: Mapping[str, Tensor], replaced: Tensor, out_dtype: Tensor
    ) -> Tensor:
        if (
            torch.cuda.is_available()
            and int8_linear.native(torch, torch.cuda.current_device()) is None
        ):
            # The op is built at boot; until it lands the reviewed comfy-kitchen binary serves
            # (its build starts here, beside the weight fill), else the eager int8 GEMM.
            int8_linear.kitchen(torch, torch.cuda.current_device())
        data = parts["data"]
        in_features, out_features, bias = _leaf_geometry(replaced, data)
        return _native_leaf_class(torch)(
            None,
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
    def _forward(leaf: Tensor, x: Tensor, out_dtype: Tensor) -> Tensor:
        import torch

        rows, cols = leaf.out_features, leaf.in_features // NVFP4_BLOCK
        return nvfp4_linear.matmul(
            torch,
            x,
            leaf.weight,
            leaf.weight_scale,
            lambda: from_blocked(torch, leaf.weight_scale, rows, cols),
            leaf.weight_scale_2,
            out_dtype,
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
