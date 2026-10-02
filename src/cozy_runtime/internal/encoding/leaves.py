"""cr-006 seam 2 — LEAVES: the providers whose answer is a replaced MODULE, not a tensor.

Seam one's providers write a float destination the contract's dtype named. These do not:
the stored payload and its scales ARE the resident weight, and what changes is the module
that multiplies by them. That is the whole of the `encoded_gemm` route (#515b, #517e), and
it is the only route on which an encoded lane is also a different kernel.

Both leaves here are **w8a8**: fp8/mx tensor cores need BOTH operands encoded, so the
activation is quantized at dispatch. That is a real, measured, accepted loss (#515a: 27.5 dB
against the dequant lane's 30.4 dB on the same bytes; #517d: 16.72 vs 20.82 on a harder
subject) bought for the residency a decoded-float route structurally cannot offer (0.5003x,
measured on real H3 block bytes, #547a). Which side of that trade a deployment wants is what
the resolved plan's OBJECTIVE decides — never registration order, and never a ladder rung.

**Geometry is preflighted here, purely, and it is the #549.5 ruling's home.** `mxfp8/1`
admits a ragged tail by spec; these tensor cores do not. `MicroScaledNativeLeaf.supports`
refuses `K % 32 != 0` — matching both producers, which emit whole blocks — so a ragged
tensor resolves to the decode floor at PLAN time with the reason in the plan, instead of
reaching `leaf()` with the component already resident.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cozy_runtime.internal.accel import readable
from cozy_runtime.internal.encoding.formats import (
    E4M3_MAX,
    E8M0_BIAS,
    MX_BLOCK,
    MicroScaledDequant,
    RolePart,
    _rank2,
    refuse,
)

#: The activation scale's floor, and the same policy `ROW_SCALE_FLOOR` states for weights:
#: a row of exact zeros has amax 0, and dividing by it poisons a whole output row with NaN.
ACT_SCALE_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class PreQuantized:
    """An activation already in the rowwise leaf's operand geometry (h3a-015).

    A fused glue kernel that holds the row in registers quantizes it there instead of
    writing bf16 for the leaf to read back; the leaf then skips `quantize_activation_rowwise`
    and feeds the payload and per-row scale straight to the GEMM. `shape` is the bf16
    activation's, so the leaf's output reshape is the same expression either way.
    """

    payload: Any
    scale: Any
    shape: tuple[int, ...]


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
            row = tl.program_id(0)
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

    NOT a probe-only helper, despite the family resemblance to `instruments.py`'s two: this
    one runs inside every forward pass of a rowwise encoded leaf.
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


def quantize_activation_mx(torch: Any, x: Any) -> tuple[Any, Any]:
    """`[..., K]` float -> `[M, K]` E4M3 payload + `[M, K/32]` E8M0 byte grid.

    The exponent rule is #517's CEILING one, not the shipped `floor(log2 amax) - 8`. That
    rule puts a block's maximum at `256*m` for `m` in `[1, 2)`, which exceeds E4M3's 448
    whenever `m > 1.75` — so the block's own LARGEST element clamps, and #517 measured
    884,220 saturated elements on `qkv_proj` alone and +1.077 dB recovered by fixing it.
    The producing JOB owns the same fix for the stored weights (#513/#517); the ACTIVATION
    quantizer is the runtime's own, so it is simply correct here: `e = ceil(log2(amax/448))`
    makes `amax / 2**e <= 448` by construction and saturates nothing.

    `K % 32 == 0` is a PRECONDITION, and it is the caller's to hold — which is exactly what
    `MicroScaledNativeLeaf.supports` now guarantees at plan resolution (#549.5). The reshape
    below would otherwise truncate the tail silently.
    """
    flat = x.reshape(-1, x.shape[-1])
    rows, cols = int(flat.shape[0]), int(flat.shape[1])
    work = flat.float().view(rows, cols // MX_BLOCK, MX_BLOCK)
    amax = work.abs().amax(dim=-1)
    exponent = torch.where(
        amax > 0, torch.ceil(torch.log2(amax / E4M3_MAX)), torch.full_like(amax, -127.0)
    )
    byte = (exponent + E8M0_BIAS).clamp(0.0, 254.0)
    mult = MicroScaledDequant._multiplier(torch, byte.to(torch.uint8), torch.float32)
    payload = (work / mult.unsqueeze(-1)).view(rows, cols)
    quant = payload.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return quant, byte.to(torch.uint8)


def _blocked_shape(rows: int, cols: int) -> tuple[int, int]:
    return -(-rows // 128) * 128, -(-cols // 4) * 4


def to_blocked(torch: Any, scales: Any) -> Any:
    """`[rows, cols]` E8M0 bytes -> cuBLASLt's 128x4 blocked ("swizzled") scale layout.

    #515d IS THIS FUNCTION'S REASON TO EXIST, and it is a silent-wrong-output hazard rather
    than an optimisation: torch validates the MX scale grids' ELEMENT COUNT, not their
    LAYOUT, so a plain row-major `[rows, cols/32]` grid is ACCEPTED in place of the tiled
    one the MMA actually reads — and returns wrong numbers at 62,453 of 64,485 finite
    positions. Nothing raises. The probe oracle is what catches a builder who skips this.
    """
    rows, cols = int(scales.shape[0]), int(scales.shape[1])
    padded_rows, padded_cols = _blocked_shape(rows, cols)
    row_blocks, col_blocks = padded_rows // 128, padded_cols // 4
    padded = torch.zeros(padded_rows, padded_cols, dtype=scales.dtype, device=scales.device)
    padded[:rows, :cols] = scales
    tiles = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return tiles.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


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

        def forward(self, x: Any) -> Any:
            out = self._cozy_forward(self, x)
            if self.bias is not None:
                out = out + self.bias
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
        state here the way `mxfp8/1` has one. `torch._scaled_mm` carries alignment
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
    def _forward(leaf: Any, x: Any) -> Any:
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
            out_dtype=leaf.out_dtype,
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
class MicroScaledNativeLeaf:
    """`mxfp8/1` ENCODED GEMM: block-scaled MXFP8 straight into the tensor cores (#515).

    #515 measured every premise of this provider on real silicon before a line of it
    existed: on sm120 stock `torch._scaled_mm`'s Blockwise-1x32 recipe executes MXFP8
    natively, its arithmetic is BIT-IDENTICAL to `cozy.mxfp8.dequant/1` over 64,770 of
    64,770 non-NaN pairs of the format domain, it runs the real DiT shapes at 2.45-2.75x
    the bf16 GEMM, and it holds the weights at a MEASURED 8.294 bits per element — 18.60
    against 35.96 GiB for the curve DiT.

    Two facts a reader should not have to rediscover. It is **w8a8-mx**: both GEMM operands
    must be mxfp8, so activations are quantized per 32-element block at dispatch, and #515a
    measured the whole route at 27.5 dB against the dequant lane's 30.4 dB on the same
    bytes — the kernel contributes ~nothing (55.6 dB against f32), the activation
    quantization is the cost. And the scale grid must be SWIZZLED: see `to_blocked`.

    The predicate is `cuda.sm120` EXACTLY, not `cuda.sm100+`. No B200 ran, so sm100 gets no
    floor — #511b's fail-closed-by-measurement rule forbids the hand-written one that would
    read so plausibly here.
    """

    encoding: str

    @property
    def name(self) -> str:
        return "cozy.mxfp8.native-leaf/1"

    @property
    def route(self) -> str:
        return "encoded_gemm"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm120"

    @property
    def reviewed_specs(self) -> frozenset[str]:
        from cozy_runtime.internal.encoding.formats import SPEC_MXFP8

        return frozenset({SPEC_MXFP8})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "mxfp8-native"

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
        """#549.5's RULING, implemented: `K % 32 != 0` is refused, before allocation.

        `mxfp8/1`'s scale relation is `ceil_div(axis 1, by 32)`, so the SPEC admits a last
        block shorter than 32 and TensorFS ingests one. The tensor cores do not: the MMA
        reads whole 32-element blocks against whole scale entries, and the activation
        quantizer this leaf pairs with (`quantize_activation_mx`) reshapes `[M, K]` into
        `[M, K/32, 32]`, which silently truncates a ragged tail.

        The ruling is EXACT-32 divisibility, matching both producers — ours emits whole
        blocks and every wild MXFP8 carrier mined so far does too — rather than a padding
        semantics this build does not implement. A provider that DECLARES padding semantics
        may relax this; inferring them would be inventing a contract on the artifact's
        behalf.

        A ragged tensor therefore resolves to `cozy.mxfp8.dequant/1`, whose slow path spans
        a partial block correctly, and the plan records THIS SENTENCE as the reason.
        """
        rank = _rank2(shape)
        if rank:
            return rank
        if int(shape[1]) % MX_BLOCK:
            return (
                f"mxfp8 encoded-GEMM requires K divisible by {MX_BLOCK} and this tensor is "
                f"[{shape[0]}, {shape[1]}] — {int(shape[1]) % MX_BLOCK} elements past the "
                f"last whole block. The SPEC admits a ragged tail (`ceil_div(axis 1, by "
                f"{MX_BLOCK})`) and the tensor cores do not: the MMA reads whole blocks, and "
                "the activation quantizer that pairs with this leaf reshapes [M, K] into "
                f"[M, K/{MX_BLOCK}, {MX_BLOCK}], which would truncate the tail in silence. "
                "Both known producers emit whole blocks. This build declares no padding "
                "semantics, and inferring them would invent a contract the artifact never "
                "made — so the decode floor serves this tensor"
            )
        return ""

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        raise AssertionError(
            "an encoded_gemm provider REPLACES the leaf; it never writes a float "
            "destination — see LeafProvider.leaf"
        )

    def resident_bytes(self, parts: Mapping[str, RolePart]) -> int:
        rows, cols = _blocked_shape(*parts["scale"].shape)
        return parts["data"].nbytes + rows * cols

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float:
        return 8.0 * self.resident_bytes(parts) / max(numel, 1)

    def fill_scratch_bytes(self, parts: Mapping[str, RolePart]) -> int:
        # to_blocked can overlap the padded grid, reshape copy, and final blocked copy.
        # The source grid is already part of the fill payload. Charging all three for
        # each pending leaf also bounds outputs retained until the component fence.
        rows, cols = _blocked_shape(*parts["scale"].shape)
        return 3 * rows * cols

    @staticmethod
    def _forward(leaf: Any, x: Any) -> Any:
        import torch

        xq, xbytes = quantize_activation_mx(torch, x)
        out = torch._scaled_mm(
            xq,
            leaf.data.t(),
            scale_a=to_blocked(torch, xbytes).view(torch.float8_e8m0fnu),
            scale_b=leaf.scale,
            out_dtype=leaf.out_dtype,
        )
        return out.reshape(*x.shape[:-1], -1)

    def leaf(self, torch: Any, parts: Mapping[str, Any], replaced: Any, out_dtype: Any) -> Any:
        if not hasattr(torch, "float8_e8m0fnu"):
            raise refuse(
                "encoding_unqualified",
                "this torch build carries no `float8_e8m0fnu` dtype, so it cannot express "
                "an MX block-scale operand at all — the mxfp8 encoded-GEMM leaf has no way "
                "to hand the kernel its scales, and refusing beats handing it a u8 grid",
            )
        data = parts["data"]
        in_features, out_features, bias = _leaf_geometry(replaced, data)
        # The SWIZZLE IS PAID ONCE, at fill, and its result is what stays resident. Paying
        # it per forward would put a `[out, in/32]` permute in front of every GEMM, which is
        # the same class of unfused elementwise overhead #515f names on the activation side
        # — except here it is avoidable, because the weight's grid never changes.
        blocked = to_blocked(torch, parts["scale"]).view(torch.float8_e8m0fnu)
        return _leaf_class(torch)(
            MicroScaledNativeLeaf._forward,
            {"data": data, "scale": blocked},
            bias,
            out_dtype,
            self.name,
            self.encoding,
            in_features,
            out_features,
        )
