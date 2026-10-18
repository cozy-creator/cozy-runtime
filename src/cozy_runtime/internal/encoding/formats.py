"""Stored-encoding contracts and the providers that decode them to a float tensor.

Selection is by the header's scheme name (cozytensors/2 §6); the params travel with the tensor
and a provider's `supports` judges both geometry and params before anything is allocated.
`leaves.py` holds the providers that replace a module instead; `selection.py` admits them.
Capability is fail-closed and device-keyed: a provider serves a device only where a probe
minted a record (`probe.py`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, runtime_checkable

from cozy_runtime.author._loader import TORCH_DTYPES as _TORCH_DTYPES

# `Any` below is a torch module, tensor, dtype or nn.Module: torch is absent from the check venv.

REFUSALS = frozenset(
    {
        "unknown_encoding",  # a scheme no provider here decodes
        "encoding_unqualified",  # no capability record for this device
        "role_mismatch",  # the stored parts are not a provider's parts
        "geometry_unsupported",  # every candidate refuses this tensor's shape or params
    }
)


class EncodingRefusal(Exception):
    """A typed encoding refusal. Every one fires at BIND, before a byte moves."""

    def __init__(self, message: str, *, code: str, fields: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.fields = tuple(fields)


def refuse(code: str, message: str, fields: Sequence[str] = ()) -> EncodingRefusal:
    if code not in REFUSALS:
        raise AssertionError(f"{code} is not a declared encoding refusal")
    return EncodingRefusal(message, code=code, fields=fields)


#: A plain tensor's scheme: one `value` part holding the logical bytes.
PLAIN = "plain"

#: Params as a provider holds them: hashable, sorted by name.
type Params = tuple[tuple[str, object], ...]


def params_of(params: Mapping[str, object]) -> Params:
    return tuple(sorted(params.items()))


#: Runtime dtype name -> torch dtype name (`author/_loader.py` owns it).
TORCH_DTYPES = _TORCH_DTYPES


# --------------------------------------------------------------------------- the parts


@dataclass(frozen=True, slots=True)
class RolePart:
    """One stored part of a tensor, as the header declares it, in Runtime dtype names."""

    role: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    @property
    def numel(self) -> int:
        n = 1
        for extent in self.shape:
            n *= extent
        return n


@dataclass(frozen=True, slots=True)
class Encoded:
    """A tensor's stored form: its scheme (`PLAIN` when plain), params and parts."""

    scheme: str
    params: Params
    parts: tuple[RolePart, ...]

    @property
    def roles(self) -> frozenset[str]:
        return frozenset(p.role for p in self.parts)

    @property
    def stored_nbytes(self) -> int:
        return sum(p.nbytes for p in self.parts)

    def part(self, role: str) -> RolePart:
        for p in self.parts:
            if p.role == role:
                return p
        raise refuse(
            "role_mismatch",
            f"this tensor stores roles {sorted(self.roles)} and no {role!r}",
            [role],
        )


#: Logical dtypes a decode writes (spec §6).
DECODED = frozenset({"f32", "bf16", "f16"})
_F8 = frozenset({"f8_e4m3fn", "f8_e5m2"})

#: mxfp8's block: one E8M0 scale per this many elements along the last axis.
MX_BLOCK = 32
#: E8M0's one non-numeric byte; any other `b` is exactly `2**(b - 127)`.
E8M0_NAN = 255


def part_rule(encoded: Encoded, shape: Sequence[int], dtype: str) -> str:
    """ "" when an encoded tensor's parts keep its scheme's dtype and shape rules (spec §6),
    else why not. Roles and params are judged by the providers; an unknown scheme is not."""
    if encoded.scheme == PLAIN:
        return ""
    if dtype not in DECODED:
        return f"{encoded.scheme} decodes to f32, bf16 or f16, not {dtype}"
    parts = {p.role: p for p in encoded.parts}
    rows = int(shape[0]) if shape else 0
    cols = int(shape[1]) if len(shape) > 1 else 0
    axis = dict(encoded.params).get("axis")
    rules: dict[str, tuple[frozenset[str], int | tuple[int, ...]]] = {
        "fp8-rowwise": {
            "weight": (_F8, tuple(shape)),
            "scale": (frozenset({"f32"}), cols if axis == 1 else rows),
        },
        "int8-rowwise": {
            "weight": (frozenset({"i8"}), tuple(shape)),
            "scale": (frozenset({"f32"}), cols if axis == 1 else rows),
        },
        "mxfp8": {
            "weight": (_F8, tuple(shape)),
            "scale": (frozenset({"f8_e8m0"}), (rows, -(-cols // MX_BLOCK))),
        },
        "fp8-tensorwise": {
            "weight": (_F8, tuple(shape)),
            "scale": (frozenset({"f32"}), 1),
            "input_scale": (frozenset({"f32"}), 1),
        },
        "int8-convrot": {
            "weight": (frozenset({"i8"}), tuple(shape)),
            "scale": (frozenset({"f32"}), rows),
        },
        "nvfp4": {
            "weight": (frozenset({"u8"}), (rows, cols // 2)),
            "scale": (frozenset({"f8_e4m3fn"}), (rows, cols // NVFP4_BLOCK)),
            "scale_2": (frozenset({"f32"}), 1),
            "input_scale": (frozenset({"f32"}), 1),
        },
    }.get(encoded.scheme, {})  # type: ignore[assignment]
    for role, (dtypes, want) in rules.items():
        part = parts.get(role)
        if part is None:
            continue
        if part.dtype not in dtypes:
            return f"{encoded.scheme} part {role} is {part.dtype}, not one of {sorted(dtypes)}"
        got: int | tuple[int, ...] = part.numel if isinstance(want, int) else part.shape
        if got != want:
            return f"{encoded.scheme} part {role} has shape {part.shape}, expected {want}"
    return ""


# ------------------------------------------------------------------------ the providers

#: A torch tensor, spelled once: torch is absent from the check venv.
type Tensor = Any


class Provider(Protocol):
    """One implementation that turns one scheme's stored parts into one destination.

    A provider instance is bound to a tensor's params (`params`); `decode` reads them there.
    An `encoded_gemm` provider replaces the module instead (`LeafProvider`).
    """

    @property
    def encoding(self) -> str:
        """The scheme this instance serves."""
        ...

    @property
    def params(self) -> Params: ...

    @property
    def name(self) -> str: ...

    @property
    def route(self) -> str:
        """`verbatim`, `decoded_float` or `encoded_gemm`."""
        ...

    @property
    def device_predicate(self) -> str: ...

    @property
    def schemes(self) -> frozenset[str]:
        """The schemes this implementation decodes."""
        ...

    @property
    def code_revision(self) -> str:
        """Bumped when the arithmetic, operand layout or kernel call changes; it joins the
        capability-record key, so records of the old code stop admitting the new."""
        ...

    @property
    def suite(self) -> str:
        """The probe oracle that qualifies it."""
        ...

    @property
    def required(self) -> frozenset[str]: ...

    @property
    def optional(self) -> frozenset[str]: ...

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        """Pure preflight of the logical shape and the scheme's params: "" or why not."""
        ...

    def decode(
        self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor
    ) -> None: ...


@runtime_checkable
class LeafProvider(Provider, Protocol):
    """A module-replacing provider: the stored parts stay resident as the weight and the
    leaf (a real `nn.Module`) multiplies by them."""

    @property
    def output_dtype_rule(self) -> str:
        """`contract`: the leaf emits the destination dtype the contract named."""
        ...

    def leaf(self, torch: Any, parts: Mapping[str, Any], replaced: Any, out_dtype: Any) -> Any: ...

    def resident_bytes(self, parts: Mapping[str, RolePart]) -> int: ...

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float: ...

    def fill_scratch_bytes(self, parts: Mapping[str, RolePart]) -> int: ...


def _rank2(shape: Sequence[int]) -> str:
    if len(shape) != 2:
        return f"a scaled GEMM has two outer dimensions and this tensor is rank {len(shape)} {tuple(shape)}"
    return ""


def _params(params: Mapping[str, object], allowed: Mapping[str, Sequence[object]]) -> str:
    """ "" when `params` names exactly `allowed`'s keys, each with an allowed value."""
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        return f"unknown params {unknown}"
    for name, values in allowed.items():
        if params.get(name) not in values:
            return f"param {name}={params.get(name)!r}, expected one of {list(values)}"
    return ""


@dataclass(frozen=True, slots=True)
class Verbatim:
    """A plain tensor: the stored bytes are the logical bytes."""

    encoding: str = PLAIN
    params: Params = ()

    @property
    def name(self) -> str:
        return "cozy.plain.verbatim/1"

    @property
    def route(self) -> str:
        return "verbatim"

    @property
    def device_predicate(self) -> str:
        return "any"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({PLAIN})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "verbatim"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"value"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        return _params(params, {})

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        raise AssertionError("a verbatim part is streamed straight into its destination")


@dataclass(frozen=True, slots=True)
class TensorwiseDequant:
    """`fp8-tensorwise` -> float: `weight × s`, `s` the scale rounded to the destination dtype.

    `input_scale` is the activation scale of an fp8 matmul and never enters the weight. The
    multiply in the destination dtype is the spec's rounding (and halves the scratch).
    """

    encoding: str = "fp8-tensorwise"
    params: Params = ()

    @property
    def name(self) -> str:
        return "cozy.fp8-tensorwise.dequant/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        return "any"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({"fp8-tensorwise"})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "fp8-tensorwise-decode"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset({"input_scale"})

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        return _params(params, {})

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        destination.copy_(parts["weight"].view(destination.shape).to(destination.dtype))
        destination.mul_(parts["scale"].to(destination.dtype).reshape(()))


#: E4M3's largest finite magnitude. The cast to `float8_e4m3fn` does not saturate (449 becomes
#: NaN), so every path that rounds into e4m3 clamps first.
E4M3_MAX = 448.0

#: The floor under a derived row scale: a zero row must not divide by zero.
ROW_SCALE_FLOOR = 1e-12

#: The largest f32 transient a chunked decode holds at once: one region's grain.
_CHUNK_BYTES = 16 << 20


@dataclass(frozen=True, slots=True)
class RowwiseDequant:
    """`fp8-rowwise` and `int8-rowwise` -> float: `weight[i, j] × scale[axis == 0 ? i : j]`,
    formed in f32 and rounded once into the destination (the producer's rule; narrowing the
    scale first is a full ULP off). Row chunks bound the f32 transient."""

    encoding: str = "fp8-rowwise"
    params: Params = ()

    @property
    def name(self) -> str:
        return f"cozy.{self.encoding}.dequant/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        return "any"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({"fp8-rowwise", "int8-rowwise"})

    @property
    def code_revision(self) -> str:
        return "3"

    @property
    def suite(self) -> str:
        return f"{self.encoding}-decode"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        return _rank2(shape) or _params(params, {"axis": (0, 1)})

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        rows, cols = int(destination.shape[0]), int(destination.shape[1])
        payload = parts["weight"].view(rows, cols)
        per_column = dict(self.params).get("axis") == 1
        scale = parts["scale"].to(torch.float32).reshape((1, cols) if per_column else (rows, 1))
        stride = max(1, _CHUNK_BYTES // max(cols * 4, 1))
        for start in range(0, rows, stride):
            stop = min(start + stride, rows)
            block = payload[start:stop].to(torch.float32)
            block.mul_(scale if per_column else scale[start:stop])
            destination[start:stop].copy_(block)


def e8m0_multipliers(torch: ModuleType, scale: Tensor) -> Tensor:
    """E8M0 bytes -> their f32 values by bit construction: `b` is the f32 exponent field
    (`b << 23`), `0` the subnormal `2**-127` and `255` NaN. No `exp2`, so no backend math."""
    raw = scale.view(torch.uint8)
    bits = raw.to(torch.int32) << 23
    bits = torch.where(raw == 0, torch.full_like(bits, 1 << 22), bits)
    mult = bits.view(torch.float32)
    return torch.where(raw == E8M0_NAN, torch.full_like(mult, float("nan")), mult)


@dataclass(frozen=True, slots=True)
class MicroScaledDequant:
    """`mxfp8` -> float: `weight[i, j] × 2**(scale[i, j // 32] - 127)`, formed in f32 (exact:
    an E4M3/E5M2 element times a power of two stays inside f32) and rounded once into the
    destination. A short last block (`ceil(C / 32)` scales) decodes like any other."""

    encoding: str = "mxfp8"
    params: Params = ()

    @property
    def name(self) -> str:
        return "cozy.mxfp8.dequant/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        return "any"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({"mxfp8"})

    @property
    def code_revision(self) -> str:
        return "2"

    @property
    def suite(self) -> str:
        return "mxfp8-decode"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        return _rank2(shape) or _params(params, {"block": (MX_BLOCK,)})

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        rows, cols = int(destination.shape[0]), int(destination.shape[1])
        payload = parts["weight"].view(rows, cols)
        mult = e8m0_multipliers(torch, parts["scale"]).reshape(rows, -1)
        stride = max(1, _CHUNK_BYTES // max(cols * 8, 1))
        for start in range(0, rows, stride):
            stop = min(start + stride, rows)
            block = payload[start:stop].to(torch.float32)
            block.mul_(mult[start:stop].repeat_interleave(MX_BLOCK, dim=1)[:, :cols])
            destination[start:stop].copy_(block)


#: ConvRot's group: the regular Hadamard H4⊗H4⊗H4⊗H4 rotates each 256 input channels.
CONVROT_GROUP = 256


def hadamard256(torch: ModuleType, values: Tensor) -> Tensor:
    """`[..., K]` -> each 256-group times the unnormalized regular Hadamard (entries ±1), in
    four 4-point stages of additions: integers stay exact. H is symmetric, H·H = 256·I."""
    shape = values.shape
    work = values.reshape(-1, 4, 4, 4, 4)
    for axis in range(1, 5):
        a, b, c, d = work.unbind(axis)
        work = torch.stack((a + b + c - d, a + b - c + d, a - b + c + d, b + c + d - a), axis)
    return work.reshape(shape)


@dataclass(frozen=True, slots=True)
class Int8ConvRotDequant:
    """`int8-convrot` -> float: `(y[i] × scale[i]) × 2^-4`, `y` the exact int32 Hadamard sum of
    each 256-group of int8 weights, cast once into the destination. The Runtime decodes
    `block: 256` only."""

    encoding: str = "int8-convrot"
    params: Params = ()

    @property
    def name(self) -> str:
        return "cozy.int8-convrot.dequant/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        return "any"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({"int8-convrot"})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "int8-convrot-decode"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        why = _rank2(shape) or _params(
            params, {"block": (CONVROT_GROUP,), "rotation": ("hadamard",)}
        )
        if why or int(shape[1]) % CONVROT_GROUP == 0:
            return why
        return f"int8-convrot rotates whole {CONVROT_GROUP}-column groups; this tensor has {shape[1]} columns"

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        rows, cols = int(destination.shape[0]), int(destination.shape[1])
        payload = parts["weight"].view(rows, cols)
        scale = parts["scale"].to(torch.float32).reshape(rows, 1)
        stride = max(1, _CHUNK_BYTES // max(cols * 4, 1))
        for start in range(0, rows, stride):
            stop = min(start + stride, rows)
            block = hadamard256(torch, payload[start:stop].to(torch.int32)).to(torch.float32)
            block.mul_(scale[start:stop]).mul_(0.0625)
            destination[start:stop].copy_(block)


#: NVFP4's block: one E4M3 scale per 16 elements along the input axis.
NVFP4_BLOCK = 16

#: E2M1 by nibble: bit 3 is the sign, bits 0-2 index the magnitudes.
E2M1_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip


def nvfp4_values(torch: ModuleType, weight: Tensor, scale: Tensor, scale_2: Tensor) -> Tensor:
    """`nvfp4` rows -> f32: `(e2m1 × e4m3 scale) × scale_2`. `weight` is u8 `[R, C/2]`,
    element 2k in the low nibble."""
    rows, cols = int(weight.shape[0]), int(weight.shape[1]) * 2
    codes = torch.stack((weight & 0x0F, weight >> 4), dim=-1).reshape(rows, cols)
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=weight.device)
    blocks = lut[codes.long()].view(rows, cols // NVFP4_BLOCK, NVFP4_BLOCK)
    blocks.mul_(scale.to(torch.float32).view(rows, cols // NVFP4_BLOCK, 1))
    return blocks.view(rows, cols).mul_(scale_2.to(torch.float32).reshape(()))


@dataclass(frozen=True, slots=True)
class Nvfp4Dequant:
    """`nvfp4` -> float on every device; row chunks bound the f32 transient. `input_scale` is
    the activation scale and never enters the weight."""

    encoding: str = "nvfp4"
    params: Params = ()

    @property
    def name(self) -> str:
        return "cozy.nvfp4.dequant/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        return "any"

    @property
    def schemes(self) -> frozenset[str]:
        return frozenset({"nvfp4"})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "nvfp4-decode"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"weight", "scale", "scale_2"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset({"input_scale"})

    def supports(self, shape: Sequence[int], params: Mapping[str, object]) -> str:
        why = _rank2(shape) or _params(params, {"block": (NVFP4_BLOCK,)})
        if why or int(shape[1]) % NVFP4_BLOCK == 0:
            return why
        return f"nvfp4 scales whole {NVFP4_BLOCK}-column blocks; this tensor has {shape[1]} columns"

    def decode(self, torch: ModuleType, parts: Mapping[str, Tensor], destination: Tensor) -> None:
        rows, cols = int(destination.shape[0]), int(destination.shape[1])
        weight = parts["weight"].view(torch.uint8).view(rows, cols // 2)
        scale = parts["scale"].view(rows, cols // NVFP4_BLOCK)
        stride = max(1, _CHUNK_BYTES // max(cols * 8, 1))
        for start in range(0, rows, stride):
            stop = min(start + stride, rows)
            destination[start:stop].copy_(
                nvfp4_values(torch, weight[start:stop], scale[start:stop], parts["scale_2"])
            )
