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

import functools
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, ClassVar, Protocol

from cozy_runtime.author._context import device as placed_device
from cozy_runtime.author.kernels import (
    INT8_CONVROT,
    ROWWISE_FP8,
    PreQuantized,
    read,
    readable,
    shared_operand,
)
from cozy_runtime.internal import (
    fp8_linear,
    int8_linear,
    nvfp4_linear,
    prebuilt,
    svdq_linear,
    w4a8_linear,
)
from cozy_runtime.internal.encoding.formats import (
    CONVROT_GROUP,
    E4M3_MAX,
    MX_BLOCK,
    NVFP4_BLOCK,
    Int8ConvRotDequant,
    MicroScaledDequant,
    Nvfp4Dequant,
    Params,
    RolePart,
    Tensor,
    _blocked_shape,
    _params,
    _rank2,
    e8m0_multipliers,
    from_blocked,
    refuse,
    to_blocked,
)

NVFP4 = "cozy.nvfp4.native-leaf/1"
MXFP8 = "cozy.mxfp8.native-leaf/2"
W4A8 = "cozy.w4a8-convrot256.native-leaf/1"
SVDQ = "cozy.svdq-nvfp4.native-leaf/1"


class _Serving(Protocol):
    def serving(self, index: int) -> str: ...


class _ScaledMm:
    """torch's block-scaled GEMM: one stock kernel, nothing to build."""

    @staticmethod
    def serving(index: int) -> str:
        return "torch._scaled_mm"


#: Each leaf provider's lane format, as a run's `gemm.selected` names it.
FORMATS = {
    INT8_CONVROT: "int8",
    ROWWISE_FP8: "fp8",
    NVFP4: "nvfp4",
    MXFP8: "mxfp8",
    W4A8: "w4a8",
    SVDQ: "svdq-nvfp4",
}
_MODULES: dict[str, _Serving] = {
    INT8_CONVROT: int8_linear,
    ROWWISE_FP8: fp8_linear,
    NVFP4: nvfp4_linear,
    MXFP8: _ScaledMm,
    W4A8: w4a8_linear,
    SVDQ: svdq_linear,
}
#: The activation operand each native leaf takes, by provider: a w4a8 leaf takes int8.convrot's,
#: so it shares one quantization and a producer's prologue with the int8 leaves of its input.
OPERANDS = {INT8_CONVROT: INT8_CONVROT, ROWWISE_FP8: ROWWISE_FP8, W4A8: INT8_CONVROT}


def serving(provider: str, index: int) -> str:
    """The kernel `provider`'s leaves on this card serve now: read once a run, never per call."""
    return str(_MODULES[provider].serving(index))


def sites(leaves: Mapping[str, tuple[Any, ...]]) -> Counter[tuple[str, str]]:
    """`(component, provider)` -> how many installed leaves: counted once, when they install."""
    return Counter((key.partition(".")[0], entry[0].provider) for key, entry in leaves.items())


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

        #: A role's buffer where it is not the role's own name.
        BUFFERS: ClassVar[dict[str, str]] = {}

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
                self.register_buffer(self.BUFFERS.get(role, role), tensor)
            self.register_buffer("bias", bias)
            self.out_dtype = out_dtype
            self.provider = provider
            self.encoding = encoding
            self.in_features = in_features
            self.out_features = out_features

        def roles(self) -> dict[str, Any]:
            return {role: getattr(self, self.BUFFERS.get(role, role)) for role in self._cozy_roles}

        def forward(self, x: Any, *, prequantized: PreQuantized | None = None) -> Any:
            # Semantic hooks still see the original positional float operand.
            return self.linear(x, prequantized)

        def linear(self, x: Tensor, prequantized: PreQuantized | None) -> Tensor:
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
        BUFFERS: ClassVar[dict[str, str]] = {"weight": "data"}  # no `.weight` on this leaf

        @property
        def operand(self) -> str:
            return OPERANDS[self.provider]

        def linear(self, x: Tensor | PreQuantized, prequantized: PreQuantized | None) -> Tensor:
            # The ops launch by pointer: a leaf run outside its model's forward (Klein's
            # modulation heads) has its weight read in (cr-208).
            ops = _NATIVE_OPS[self.provider]
            out_dtype = self.out_dtype
            data, scale, bias = read(self.data), read(self.scale), read(self.bias)
            if isinstance(x, PreQuantized):
                # A producer's prologue (`kernels.int8_rows`) hands the operand in place of x.
                if torch.is_autocast_enabled(x.payload.device.type):
                    out_dtype = torch.get_autocast_dtype(x.payload.device.type)
                return ops.gemm(torch, x, data, scale, bias, out_dtype)
            cast = x
            if torch.is_autocast_enabled(x.device.type):
                # As the replaced `nn.Linear` does: from and to the autocast dtype.
                out_dtype = torch.get_autocast_dtype(x.device.type)
                cast = x.to(out_dtype)
            few = ops is int8_linear and x.numel() <= int8_linear.GEMV_ROWS * x.shape[-1]
            if prequantized is None and few:
                # A decode step's row or few: the op's one GEMV launch, which no shared operand
                # serves (`int8_linear.quantize` declines so few rows).
                return ops.linear(torch, cast, data, scale, bias, out_dtype)
            if prequantized is None or prequantized.provider != self.provider:
                prequantized = shared_operand(x, self.provider, lambda: ops.quantize(torch, cast))
            if prequantized is not None:
                return ops.gemm(torch, prequantized, data, scale, bias, out_dtype)
            return ops.linear(torch, cast, data, scale, bias, out_dtype)

        def int8_prologue(
            self,
            x: Tensor,
            consumers: tuple[object, ...],
            norm: str | None,
            eps: float,
            scale: Tensor | None,
            shift: Tensor | None,
            activation: str | None,
            up: Tensor | None = None,
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
                up=up,
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
    """`fp8.rowwise` ENCODED GEMM: the tensor cores consume the stored E4M3 payload directly
    (`fp8_linear`).

    fp8 tensor cores need BOTH operands fp8, so this is w8a8 (#514b): the activation is
    quantized per token at dispatch, into the scale geometry that pairs with the weight's
    per-output-row one. #517d measured that at 16.72 dB against the dequant-bf16 lane's 20.82
    dB on identical bytes, 1.48-1.63 dB of it the activation quantization; it buys 0.5003x the
    resident VRAM, because the payload IS the weight. The accumulation is the fast one, as
    `_scaled_mm(use_fast_accum=True)`'s: on an L40S the default ran 0.52x bf16 and the fast
    one 2.03x, for -0.007 to -0.018 dB (#517b).

    The floor `cuda.sm89+` is where an fp8 tensor-core GEMM exists (`formats`).
    """

    encoding: str = "fp8.rowwise"
    params: Params = ()

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
    def schemes(self) -> frozenset[str]:
        return frozenset({"fp8.rowwise"})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "fp8-rowwise-native"

    @property
    def formats(self) -> frozenset[str]:
        return frozenset({"fp8"})

    @property
    def out_dtypes(self) -> frozenset[str]:
        """`_scaled_mm`'s row-wise scaling writes half precision only."""
        return frozenset({"bfloat16", "float16"})

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    @property
    def output_dtype_rule(self) -> str:
        return "contract"

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        """Rank 2, a scale per row, and the fp8 GEMMs' 16-byte rows: K and N divisible by 16."""
        why = _rank2(shape) or _params(params, {})
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
        data = parts["weight"]
        in_features, out_features, bias = _leaf_geometry(replaced, data)
        return _native_leaf_class(torch)(
            None,
            {"weight": data, "scale": parts["scale"].to(torch.float32)},  # the kernels read f32
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

    encoding: str = "int8.convrot"
    params: Params = ()

    @property
    def name(self) -> str:
        return INT8_CONVROT

    @property
    def route(self) -> str:
        return "encoded_gemm"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm80+"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({"int8.convrot"})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "int8-convrot256-native"

    @property
    def formats(self) -> frozenset[str]:
        return frozenset({"int8"})

    @property
    def out_dtypes(self) -> frozenset[str]:
        """Half through the 8-bit op; float32 through torch's int8 GEMM."""
        return frozenset({"bfloat16", "float16", "float32"})

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    @property
    def output_dtype_rule(self) -> str:
        return "contract"

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        """The decode's group law, and cuBLASLt's 8-aligned N for the eager int8 GEMM."""
        why = _rank2(shape) or Int8ConvRotDequant().supports(shape, params)
        if not why and params.get("rotation_size") != CONVROT_GROUP:
            why = f"the int8 kernel rotates {CONVROT_GROUP}-groups; this weight's are {params.get('rotation_size')}"
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
        placed = placed_device()
        if placed.type == "cuda" and int8_linear.native(torch, placed.index or 0) is None:
            # The op is built at boot; until it lands the reviewed comfy-kitchen binary serves
            # (its build starts here, beside the weight fill), else the eager int8 GEMM.
            int8_linear.kitchen(torch, torch.cuda.current_device())
        data = parts["weight"]
        in_features, out_features, bias = _leaf_geometry(replaced, data)
        return _native_leaf_class(torch)(
            None,
            {"weight": data, "scale": parts["scale"].to(torch.float32)},  # the kernels read f32
            bias,
            out_dtype,
            self.name,
            self.encoding,
            in_features,
            out_features,
        )


def _w4a8_leaf_class(torch: ModuleType) -> type:
    """`NativeLinear` over w4a8's roles: its GEMM (`w4a8_linear`) on int8.convrot's operand."""
    cached: type | None = getattr(_w4a8_leaf_class, "_cached", None)
    if cached is not None:
        return cached

    class W4a8Linear(_native_leaf_class(torch)):  # type: ignore[misc]
        def weights(self) -> prebuilt.W4a8:
            return prebuilt.W4a8(
                self.data, self.group_scale, self.scale, self._buffers.get("codebook")
            )

        def linear(self, x: Tensor | PreQuantized, prequantized: PreQuantized | None) -> Tensor:
            out_dtype = self.out_dtype
            if isinstance(x, PreQuantized):
                if torch.is_autocast_enabled(x.payload.device.type):
                    out_dtype = torch.get_autocast_dtype(x.payload.device.type)
                return w4a8_linear.gemm(torch, x, self.weights(), self.bias, out_dtype)
            cast = x
            if torch.is_autocast_enabled(x.device.type):
                out_dtype = torch.get_autocast_dtype(x.device.type)
                cast = x.to(out_dtype)
            if prequantized is None or prequantized.provider != INT8_CONVROT:
                prequantized = shared_operand(
                    x, INT8_CONVROT, lambda: int8_linear.quantize(torch, cast)
                )
            if prequantized is not None:
                return w4a8_linear.gemm(torch, prequantized, self.weights(), self.bias, out_dtype)
            return w4a8_linear.linear(torch, cast, self.weights(), self.bias, out_dtype)

    _w4a8_leaf_class._cached = W4a8Linear  # type: ignore[attr-defined]
    return W4a8Linear


@dataclass(frozen=True, slots=True)
class W4a8NativeLeaf:
    """`w4a8.convrot` ENCODED GEMM (xs-062): 4-bit codes in int8.convrot's rotated basis,
    E4M3 group scales per 16 and a row scale stay resident (4.5 bits a weight); each code is
    unpacked to its int8 level in registers and multiplied with int8.convrot's activation on the
    int8 tensor cores (`w4a8_linear`). So it is W4A8 with int8.convrot's epilogue, at its compute
    rate, for half its weight bytes: the 4-bit lane before Blackwell.

    It takes the layout our producer writes (group 16, rotation 256, E4M3 group scales, a codebook
    or the uniform grid). Kitchen's own `comfy.asym_w4a8_int8` decodes.
    """

    encoding: str = "w4a8.convrot"
    params: Params = ()

    @property
    def name(self) -> str:
        return W4A8

    @property
    def route(self) -> str:
        return "encoded_gemm"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm80+"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({"w4a8.convrot"})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "w4a8-native"

    @property
    def formats(self) -> frozenset[str]:
        return frozenset({"int8"})

    @property
    def out_dtypes(self) -> frozenset[str]:
        """Half through the w4a8 op; float32 through torch's int8 GEMM on the unpacked levels."""
        return frozenset({"bfloat16", "float16", "float32"})

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "group_scale", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset({"codebook"})

    @property
    def output_dtype_rule(self) -> str:
        return "contract"

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        """Rank 2; group 16 and int8.convrot's 256-blocks; N divisible by 8."""
        why = _rank2(shape)
        wanted = {"group_size": 16, "rotation": "kron_regular_h4", "rotation_size": CONVROT_GROUP}
        if not why and dict(params) != wanted:
            why = f"the w4a8 GEMM reads {wanted}; this weight's are {dict(params)}"
        if not why and (int(shape[1]) % CONVROT_GROUP or int(shape[0]) % 8):
            why = f"the w4a8 GEMM needs K % {CONVROT_GROUP} and N % 8 == 0; this is {list(shape)}"
        return why

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
        if (placed := placed_device()).type == "cuda":
            # The 8-bit op is built at boot; asking starts its load beside the weight fill.
            int8_linear.native(torch, placed.index or 0)
        codes = parts["weight"]
        roles = {
            "weight": codes,
            "group_scale": parts["group_scale"],
            "scale": parts["scale"].to(torch.float32),
        }
        if "codebook" in parts:
            roles["codebook"] = parts["codebook"].to(torch.float32)
        return _w4a8_leaf_class(torch)(
            None,
            roles,
            getattr(replaced, "bias", None),
            out_dtype,
            self.name,
            self.encoding,
            2 * int(codes.shape[1]),
            int(codes.shape[0]),
        )


def _nvfp4_leaf_class(torch: ModuleType) -> type:
    """`EncodedLinear` keyed by nvfp4's stored roles, held in the buffers its kernels read."""
    cached: type | None = getattr(_nvfp4_leaf_class, "_cached", None)
    if cached is not None:
        return cached

    class Nvfp4Linear(_leaf_class(torch)):  # type: ignore[misc]
        BUFFERS: ClassVar[dict[str, str]] = {"scale": "weight_scale", "scale_2": "weight_scale_2"}

    _nvfp4_leaf_class._cached = Nvfp4Linear  # type: ignore[attr-defined]
    return Nvfp4Linear


@dataclass(frozen=True, slots=True)
class Nvfp4NativeLeaf:
    """`nvfp4-w4a4/1` ENCODED GEMM: the stored E2M1 payload and E4M3 block scales stay resident
    (0.56 bytes per weight) and are multiplied as they are (`nvfp4_linear`).

    On Blackwell a product of more than 16 rows is W4A4 on the FP4 tensor cores: each activation is
    quantized at dispatch against its own tensor amax, and on real FLUX.3 Klein activations that is
    5-10% from bf16 per linear (int8.convrot: 0.5-1.2%) at 1.3-1.5x the int8 leaf's rate on an RTX
    5090. Eligible small BF16 products use packed W4A16 GEMV. Other products materialize a
    floating weight for the matmul; this leaf does not retain that decoded temporary across calls.
    """

    encoding: str = "nvfp4"
    params: Params = ()

    @property
    def name(self) -> str:
        return NVFP4

    @property
    def route(self) -> str:
        return "encoded_gemm"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm80+"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({"nvfp4"})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "nvfp4-native"

    @property
    def formats(self) -> frozenset[str]:
        """None: W4A16 runs on every MMA card, W4A4 where the row has fp4 (`nvfp4_linear`)."""
        return frozenset()

    @property
    def out_dtypes(self) -> frozenset[str]:
        """Half through its kernels; float32 through the eager decode."""
        return frozenset({"bfloat16", "float16", "float32"})

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "scale", "scale_2"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset({"input_scale"})

    @property
    def output_dtype_rule(self) -> str:
        return "contract"

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        """The decode's block law, and cuBLASLt's FP4 GEMM: K divisible by 32, N by 8."""
        why = _rank2(shape) or Nvfp4Dequant().supports(shape, params)
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
        rows, cols = _blocked_shape(*parts["scale"].shape)
        return parts["weight"].nbytes + rows * cols + 4

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float:
        return 8.0 * self.resident_bytes(parts) / max(numel, 1)

    def fill_scratch_bytes(self, parts: Mapping[str, RolePart]) -> int:
        # As the MX leaf's: to_blocked's padded grid, reshape copy and final blocked copy.
        rows, cols = _blocked_shape(*parts["scale"].shape)
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
        if (placed := placed_device()).type == "cuda":
            # Starts the kernel's build beside the weight fill; the first calls run eager.
            int8_linear.kitchen(torch, placed.index or 0)
        params = dict(self.params)
        weight = parts["weight"].view(torch.uint8)
        if params.get("nibble_order") == "high_first":
            weight = (weight << 4) | (weight >> 4)  # the kernels read element 2k low
        out_features, in_features = int(weight.shape[0]), 2 * int(weight.shape[1])
        rows, cols = _blocked_shape(out_features, in_features // NVFP4_BLOCK)
        if params.get("scale_layout") == "cublas_128x4":
            blocked = parts["scale"].view(torch.uint8).reshape(rows, cols)  # the kernels' tiles
        else:
            # Swizzled once, at fill: the weight's grid never changes.
            blocked = to_blocked(torch, parts["scale"].view(torch.uint8)).view(rows, cols)
        blocked = blocked.view(torch.float8_e4m3fn)
        second = parts["scale_2"].to(torch.float32).reshape(1)
        if params.get("global_scale") == "divide":
            second = 1.0 / second  # the kernels multiply: one f32 reciprocal, a W4A4 rounding
        return _nvfp4_leaf_class(torch)(
            Nvfp4NativeLeaf._forward,
            {"weight": weight, "scale": blocked, "scale_2": second},
            getattr(replaced, "bias", None),
            out_dtype,
            self.name,
            self.encoding,
            in_features,
            out_features,
        )


def _svdq_leaf_class(torch: ModuleType) -> type:
    """`EncodedLinear` over svdq.nvfp4's roles: both kernels (`svdq_linear`), the bias in the
    GEMM's epilogue."""
    cached: type | None = getattr(_svdq_leaf_class, "_cached", None)
    if cached is not None:
        return cached

    class SvdqLinear(_leaf_class(torch)):  # type: ignore[misc]
        #: A baked request's LoRAs (`weights._Bakes.prepare`): what returns each one's
        #: `(identity, down [r, K], up [N, r], scale)` as the forward runs; None for none.
        lora: Callable[[], list[tuple[object, Tensor, Tensor, float]]] | None = None
        #: The branch with the LoRAs as more ranks (else their own term, past `RANKS`), for the
        #: factors and branch bound at these addresses.
        _merged: tuple[tuple[object, ...], dict[str, Tensor] | None, tuple[Tensor, Tensor]] | None
        _merged = None
        #: Whether the weight is in the prebuilt GEMM's tiles (`svdq_linear.tiled`).
        tiled = False

        def adopt(
            self, lora: Callable[[], list[tuple[object, Tensor, Tensor, float]]] | None
        ) -> None:
            self.lora = lora

        def linear(self, x: Tensor, prequantized: PreQuantized | None) -> Tensor:
            out_dtype = self.out_dtype
            if torch.is_autocast_enabled(x.device.type):
                out_dtype = torch.get_autocast_dtype(x.device.type)
                x = x.to(out_dtype)
            roles = self.roles()
            deltas = self.lora() if self.lora is not None else []
            reads = [*roles.values(), self.bias, *(t for d in deltas for t in d[1:3])]
            with readable(*reads) as held:
                count = len(roles)
                roles, bias = dict(zip(roles, held[:count], strict=True)), held[count]
                apart = None
                if deltas:
                    bound = held[count + 1 :]
                    at = (
                        roles["lowrank_down"].data_ptr(), roles["lowrank_up"].data_ptr(),
                        *((d[0], d[3]) for d in deltas), *(t.data_ptr() for t in bound),
                    )  # fmt: skip
                    if self._merged is None or self._merged[0] != at:
                        down = torch.cat(bound[0::2])
                        ups = zip(bound[1::2], deltas, strict=True)
                        up = torch.cat([u * d[3] for u, d in ups], dim=1)
                        merged = svdq_linear.merged(torch, roles, down, up)
                        self._merged = (at, merged, (down, up))
                    merged, apart = self._merged[1], self._merged[2]
                    if merged is not None:
                        roles.update(merged)
                        apart = None
                y = svdq_linear.linear(torch, x, roles, bias, out_dtype, self.tiled)
            if apart is not None:  # more ranks than the kernels take: the LoRAs' own term
                down, up = apart
                y = y + ((x.to(down.dtype) @ down.t()) @ up.t()).to(y.dtype)
            return y

    _svdq_leaf_class._cached = SvdqLinear  # type: ignore[attr-defined]
    return SvdqLinear


@dataclass(frozen=True, slots=True)
class SvdqNvfp4NativeLeaf:
    """`svdq.nvfp4` ENCODED GEMM (xs-066): SVDQuant's nvfp4 residual, its smoothing and its 16-bit
    low-rank branch stay resident and are multiplied as they are: the activation is smoothed,
    quantized and down-projected in one pass, and the FP4 GEMM adds the up-projection in its
    epilogue (`svdq_linear`). Blackwell only; elsewhere the cores decode it."""

    encoding: str = "svdq.nvfp4"
    params: Params = ()

    @property
    def name(self) -> str:
        return SVDQ

    @property
    def route(self) -> str:
        return "encoded_gemm"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm100+"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({"svdq.nvfp4"})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "svdq-nvfp4-native"

    @property
    def formats(self) -> frozenset[str]:
        return frozenset({"fp4"})

    @property
    def out_dtypes(self) -> frozenset[str]:
        return frozenset({"bfloat16", "float16"})

    @property
    def required(self) -> frozenset[str]:
        return frozenset(
            {"weight", "scale", "scale_2", "pre_quant_scale", "lowrank_down", "lowrank_up"}
        )

    @property
    def optional(self) -> frozenset[str]:
        return frozenset({"input_scale", "channel_scale"})

    @property
    def output_dtype_rule(self) -> str:
        return "contract"

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        """One matrix, K a whole number of the GEMM's K steps, N of 16."""
        why = _rank2(shape) or Nvfp4Dequant().supports(shape, params)
        if not why and (int(shape[1]) % svdq_linear.K_ALIGN or int(shape[0]) % 16):
            why = (
                f"the svdq GEMM needs K % {svdq_linear.K_ALIGN} == 0 and N % 16 == 0; this tensor "
                f"is {list(shape)}"
            )
        return why

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
        """The roles as stored where the kernels read them so (every role of a canonical tensor);
        a high-first nibble order, the cuBLAS scale tiles, a divided second level, a rank no
        kernel is built for and 16-bit factors other than bf16 are made once, at the bind."""
        params = dict(self.params)
        down, up = parts["lowrank_down"], parts["lowrank_up"]
        if int(down.shape[0]) > svdq_linear.RANKS[-1]:
            raise refuse(
                "geometry_unsupported",
                f"a rank-{int(down.shape[0])} branch: the svdq kernels take up to "
                f"{svdq_linear.RANKS[-1]}; it decodes instead",
            )
        weight = parts["weight"]
        if weight.dtype != torch.uint8:
            weight = weight.view(torch.uint8)
        if params.get("nibble_order") == "high_first":
            weight = (weight << 4) | (weight >> 4)  # the kernels read element 2k low
        out_features, in_features = int(weight.shape[0]), 2 * int(weight.shape[1])
        scale = parts["scale"]
        if params.get("scale_layout") == "cublas_128x4":
            blocked = from_blocked(
                torch, scale.view(torch.uint8), out_features, in_features // NVFP4_BLOCK
            )
            scale = blocked.contiguous().view(torch.float8_e4m3fn)
        second = parts["scale_2"].to(torch.float32)
        if params.get("global_scale") == "divide":
            second = 1.0 / second
        rank = int(down.shape[0])
        if svdq_linear.rank(rank) != rank or torch.bfloat16 not in (down.dtype, up.dtype):
            down, up = svdq_linear.padded(torch, down, up)
        elif down.dtype != up.dtype:
            down, up = down.to(torch.bfloat16), up.to(torch.bfloat16)
        roles = {
            "weight": weight,
            "scale": scale,
            "scale_2": second,
            "pre_quant_scale": parts["pre_quant_scale"].to(torch.float32),
            "lowrank_down": down,
            "lowrank_up": up,
        }
        placed = placed_device()
        tiled = placed.type == "cuda" and svdq_linear.tiled(torch, placed.index or 0, out_features)
        if tiled:  # the prebuilt kernels' tiles (RTX 50); asking starts their probe beside the fill
            svdq_linear.native(torch, placed.index or 0)
            roles["weight"], roles["scale"] = svdq_linear.tiles(torch, weight, scale)
        for role in ("channel_scale", "input_scale"):
            if role in parts:
                roles[role] = parts[role].to(torch.float32)
        leaf = _svdq_leaf_class(torch)(
            None,
            roles,
            getattr(replaced, "bias", None),
            out_dtype,
            self.name,
            self.encoding,
            in_features,
            out_features,
        )
        leaf.tiled = tiled
        return leaf


def quantize_activation_mx(torch: ModuleType, x: Tensor) -> tuple[Tensor, Tensor]:
    """`[..., K]` float -> `[M, K]` E4M3 and `[M, K/32]` E8M0 bytes. Each block's exponent is
    `ceil(log2(amax / 448))`, so its largest element lands at or under 448 and nothing
    saturates. `K % 32 == 0` is the caller's (`MicroScaledNativeLeaf.supports`)."""
    flat = x.reshape(-1, x.shape[-1])
    rows, cols = int(flat.shape[0]), int(flat.shape[1])
    work = flat.float().view(rows, cols // MX_BLOCK, MX_BLOCK)
    amax = work.abs().amax(dim=-1)
    exponent = torch.where(
        amax > 0, torch.ceil(torch.log2(amax / E4M3_MAX)), torch.full_like(amax, -127.0)
    )
    byte = (exponent + 127).clamp(0.0, 254.0).to(torch.uint8)
    payload = (work / e8m0_multipliers(torch, byte).unsqueeze(-1)).view(rows, cols)
    return payload.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn), byte


@dataclass(frozen=True, slots=True)
class MicroScaledNativeLeaf:
    """`mxfp8` ENCODED GEMM on Blackwell's MX tensor cores (`torch._scaled_mm`, 1x32 block
    scales): the E4M3 payload and its E8M0 grid stay resident (8.3 bits a weight) and each
    activation is quantized per 32-block at dispatch, so it is W8A8 — a measured, accepted
    loss against the decoded lane (#515a: 27.5 dB against 30.4 dB) for the residency. The
    scale grid must be swizzled (`to_blocked`): torch checks its size, not its layout."""

    encoding: str = "mxfp8"
    params: Params = ()

    @property
    def name(self) -> str:
        return MXFP8

    @property
    def route(self) -> str:
        return "encoded_gemm"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm100+"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({"mxfp8"})

    @property
    def code_revision(self) -> str:
        return "2"

    @property
    def suite(self) -> str:
        return "mxfp8-native"

    @property
    def formats(self) -> frozenset[str]:
        return frozenset({"mxfp8"})

    @property
    def out_dtypes(self) -> frozenset[str]:
        """`_scaled_mm`'s MX scaling, at half precision."""
        return frozenset({"bfloat16", "float16"})

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    @property
    def output_dtype_rule(self) -> str:
        return "contract"

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        """The decode's law, and whole blocks: the MMA reads 32-element blocks, so a short last
        block decodes instead."""
        why = _rank2(shape) or MicroScaledDequant().supports(shape, params)
        if why:
            return why
        if int(shape[1]) % MX_BLOCK:
            return (
                f"the MX GEMM reads whole {MX_BLOCK}-element blocks; this tensor is {list(shape)}"
            )
        return ""

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        raise AssertionError("an encoded_gemm provider replaces the linear; it never decodes")

    def resident_bytes(self, parts: Mapping[str, RolePart]) -> int:
        rows, cols = _blocked_shape(*parts["scale"].shape)
        return parts["weight"].nbytes + rows * cols

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float:
        return 8.0 * self.resident_bytes(parts) / max(numel, 1)

    def fill_scratch_bytes(self, parts: Mapping[str, RolePart]) -> int:
        rows, cols = _blocked_shape(*parts["scale"].shape)
        return 3 * rows * cols  # to_blocked's padded grid, its reshape copy and the result

    @staticmethod
    def _forward(torch: ModuleType, leaf: Tensor, x: Tensor, out_dtype: Tensor) -> Tensor:
        payload, scales = quantize_activation_mx(torch, x)
        out = torch._scaled_mm(
            payload,
            leaf.weight.t(),
            scale_a=to_blocked(torch, scales).view(torch.float8_e8m0fnu),
            scale_b=leaf.scale,
            out_dtype=out_dtype,
        )
        return out.reshape(*x.shape[:-1], -1)

    def leaf(
        self, torch: ModuleType, parts: Mapping[str, Tensor], replaced: Tensor, out_dtype: Tensor
    ) -> Tensor:
        weight = parts["weight"]
        if weight.dtype != torch.float8_e4m3fn:
            raise refuse(
                "geometry_unsupported",
                f"the MX GEMM takes E4M3 weights and this one is {weight.dtype}: it decodes instead",
            )
        in_features, out_features, bias = _leaf_geometry(replaced, weight)
        scale = parts["scale"].view(torch.uint8)
        if dict(self.params).get("scale_layout") == "cublas_128x4":
            blocked = scale.reshape(-1)  # already the kernels' tiles
        else:
            blocked = to_blocked(torch, scale)  # swizzled once, at fill
        blocked = blocked.view(torch.float8_e8m0fnu)
        return _leaf_class(torch)(
            functools.partial(MicroScaledNativeLeaf._forward, torch),
            {"weight": weight, "scale": blocked},
            bias,
            out_dtype,
            self.name,
            self.encoding,
            in_features,
            out_features,
        )
