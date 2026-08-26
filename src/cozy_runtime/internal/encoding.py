"""cr-006 — the selected-encoding compute plane: which module executes which stored bytes.

cr-005 moves bytes that already ARE the destination's dtype. This module is the other
case: a checkpoint whose per-tensor encoding says the stored bytes are a CODE for the
logical tensor, and something has to turn the one into the other.

Three rules hold the shape, and each is a refusal rather than a convention:

**Selection is by DIGEST, never by name.** The header assigns every tensor an
`EncodingSpec` object digest. That digest — not the alias, not the key, not the dtype —
picks the provider. Aliases (`fp8-scaled-scalar/1`) exist for display and are read from
`tensorfs.seed_digests()` so a digest can never be transcribed wrong into this file.
Two physically different role sets under one alias are two digests and two providers,
which is exactly what the H3 production artifact carries (150 modules with an
`input_scale`, 50 `mlp.fc2` without).

**The decode target is the CONTRACT's dtype, and the contract cannot express an
encoding.** `ModelConstructionContract` names destinations and nothing else — no parts,
no scale_of, no dequantize (contract.py's `FORBIDDEN_VOCABULARY` makes that an import-time
proof). So the decode target is the destination's own dtype, the header's logical dtype
must equal it, and a disagreement is cr-005's `dtype_mismatch` unchanged. Nothing about
quantization reaches the author surface or the contract.

**Capability is fail-closed and device-keyed.** A provider serves an (encoding, device)
pair only when a CapabilityRecord exists for it, and a record cannot be written without
the numerics evidence that minted it (`probe.py`). An unrecognised device is the ABSENCE
of evidence, never permission — v1's `DTYPE_MIN_SM` fell OPEN on an unknown card and
served plausible wrong output.

**The device-honesty split, stated once.** On sm89 (this repo's development card) fp8
E4M3 is a STORAGE dtype: the tensor cores do not consume it, `torch._scaled_mm` needs
sm90. So the launch tier's fp8 route is `decoded_float` — dequantize at fill, compute in
the destination's float dtype, which is what the consumer-card ecosystem does and what
weight-only quantization means. It buys bytes read and load time, NOT VRAM: the resident
weights are float either way. The sm90+ scaled-mm route and the SM100 mxfp8 native route
are NAMED here as seams with their owning milestone; they are not built, and an artifact
that needs one refuses typed naming the device fact rather than silently taking the
float route it was not qualified for.

MXFP8 (#469a's preferred H3 DiT lane) takes that same decoded-float route and is BUILT:
E4M3 elements with one E8M0 power-of-two scale per 32-element block, dequantized into the
destination's dtype at fill. Producing an mxfp8 artifact never made one servable (#469a,
#501f); what makes it servable is a provider here plus a capability record minted by
measured numerics, and nothing else.

Its NATIVE door is no longer a hypothesis. `scripts/mxfp8-native-probe.py` rented an
RTX 5090 and measured it: on sm120 the block-scaled MXFP8 GEMM is real, its arithmetic is
bit-identical to `cozy.mxfp8.dequant/1` over the whole format domain, it runs ~2.7x the
bf16 GEMM, and it is the only route that makes the weights RESIDENT at 8.25 bits. The door
stays UNBUILT anyway, and for a reason that is not about silicon: the native route
replaces the LEAF, which is `delivery.py`'s `native_encoded` rung, not a provider here.
The seam entry carries the measured matrix, the two hazards a builder needs, and what is
still unmeasured (sm100). No record is minted for a route with no implementation — the
capability table's whole worth is that a record means something ran.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from cozy_runtime.internal import accel, canonical

#: This module's own refusal codes, added to cr-005's matrix. Closed, like that one.
REFUSALS = frozenset(
    {
        "unknown_encoding",  # the header cites a spec digest no provider claims
        "encoding_unqualified",  # a claimed encoding with no capability record for THIS device
        "role_mismatch",  # the delivered parts are not the roles the provider decodes
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


# ------------------------------------------------------------------------ the registry

#: TensorFS dtype name -> torch dtype name, for the ROLE carriers. cr-005's `TORCH_DTYPES`
#: is the same fact for logical tensors; this module imports that one rather than keeping
#: a second copy — see `torch_dtype()` below.


def aliases() -> dict[str, str]:
    """spec digest -> platform alias, read from the COMPILED registry.

    The digest is identity and the alias is display, so this direction is the safe one:
    a provider registers against a digest it looked up by alias at import, and a registry
    change that moves a digest breaks selection loudly instead of decoding wrong bytes.
    """
    import tensorfs

    return {digest: alias for alias, digest in tensorfs.seed_digests()}


def digests_of(alias: str) -> tuple[str, ...]:
    """Every spec digest the platform registry files under one alias, in registry order."""
    import tensorfs

    return tuple(digest for name, digest in tensorfs.seed_digests() if name == alias)


# --------------------------------------------------------------------------- the parts


@dataclass(frozen=True, slots=True)
class RolePart:
    """One stored role of one encoded tensor, exactly as the header declares it."""

    role: str
    dtype: str
    """The TensorFS carrier dtype name (`f8_e4m3fn`, `f32`)."""
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
    """The per-tensor encoding assignment: the digest, and the roles it stores."""

    encoding: str
    """`sha256:...` of the EncodingSpec object. THE selection key."""
    alias: str
    """Display only. Never resolved through."""
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


# ------------------------------------------------------------------------- the device


@dataclass(frozen=True, slots=True)
class DeviceFacts:
    """What a capability record is keyed against.

    `configuration` is the NVML power cap plus MIG profile (§3.2): two "same model" GPUs
    at different power caps are not the same device, and first-writer-wins must never let
    a power-starved pod mint the fleet's coefficients.
    """

    kind: str
    name: str
    sm: int
    driver: str
    configuration: str

    @property
    def predicate(self) -> str:
        """The device CLASS a capability record names: `cuda.sm89`, `cpu`."""
        return "cpu" if self.kind != "cuda" else f"cuda.sm{self.sm}"

    def satisfies(self, predicate: str) -> bool:
        """Does this device meet a provider's declared floor?

        `cuda.sm90+` admits sm90 and up; a bare `cuda.sm89` is exact; `any` admits all.
        The floor lives on the PROVIDER (a fact about an implementation), never on the
        spec (an immutable byte contract that hardware has no business inside).
        """
        if predicate == "any":
            return True
        if predicate == "cpu":
            return self.kind == "cpu"
        if not predicate.startswith("cuda.sm") or self.kind != "cuda":
            return False
        floor = predicate.removeprefix("cuda.sm")
        if floor.endswith("+"):
            return self.sm >= int(floor[:-1])
        return self.sm == int(floor)

    def tuple_key(self) -> str:
        """The probe cache key (§3.2): accelerator model, device configuration, driver."""
        return f"{self.kind}/{self.name}/{self.configuration}/{self.driver}"


def measure_device(torch: Any, index: int = 0) -> DeviceFacts:
    """Read the device once, as facts a probe result may be keyed by.

    Everything here is STABLE for the life of the machine: the power CAP (not the draw —
    a key that moves every read is not a cache key), the MIG profile, the driver's CUDA
    version. An unreadable knob is SPELLED into the key rather than defaulted, so a probe
    minted while NVML was mute cannot be reused as one minted while it answered.
    """
    kind = accel.host_backend_family()
    if not accel.present(torch, kind):
        return DeviceFacts("cpu", "cpu", 0, "", "default")
    identity = accel.device_identity(torch, kind, index)
    return DeviceFacts(
        kind=kind,
        name=str(identity["name"]),
        sm=int(identity["sm"]),
        driver=str(identity["driver"]),
        configuration=f"{_nvml('power.limit')}W/{_nvml('mig.mode.current')}",
    )


def _nvml(query: str) -> str:
    """One bounded NVML read. `N/A` and a failure are both SPELLED, never a number."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unreadable"
    return (out.splitlines() or ["unreadable"])[0].strip().replace(" ", "") or "unreadable"


# ------------------------------------------------------------------------ the providers


class Provider(Protocol):
    """One implementation that turns one encoding's stored roles into one destination.

    Providers present a PLAIN compute-dtype tensor at the leaf boundary and need no adapter
    awareness (cr-006's scope note): the SDK side-branch wraps the leaf, cr-010.
    """

    @property
    def name(self) -> str: ...

    @property
    def route(self) -> str:
        """`verbatim` or `decoded_float` — the delivery route (§3.2)."""
        ...

    @property
    def device_predicate(self) -> str: ...

    @property
    def required(self) -> frozenset[str]:
        """Roles this provider MUST be handed. A header short of one refuses."""
        ...

    @property
    def optional(self) -> frozenset[str]:
        """Roles it tolerates. A header carrying anything else refuses."""
        ...

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class Verbatim:
    """`plain/1`: the stored bytes ARE the logical bytes. There is nothing to decode.

    Entry zero of the registry is a real provider rather than an absent form, so the
    selection path has no "no encoding" branch to get wrong.
    """

    encoding: str

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
    def required(self) -> frozenset[str]:
        return frozenset({"value"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        raise AssertionError("a verbatim role is streamed straight into its destination")


@dataclass(frozen=True, slots=True)
class ScaledScalarDequant:
    """`fp8-scaled-scalar/1` -> float, weight-only: `value = data * weight_scale`.

    That equation is the spec's own, transcribed from the registry entry and from nothing
    else. `input_scale`, where the artifact carries it, is the ACTIVATION scale of a
    native fp8 matmul — it is deliberately NOT applied to the weight, because applying it
    would be a different (and wrong) tensor. On this route activations stay in the
    destination's float dtype, which is strictly more precise than the native path the
    scale was minted for; the two spec variants therefore decode identically and are still
    two providers, because a delivered role set that does not match refuses rather than
    being tolerated.

    Numerically this is exact up to the destination dtype: `data` is an e4m3 value and
    `weight_scale` an f32 scalar, so the product is representable and the only error is
    the one the QUANTIZER already committed. The decode adds no error of its own, which is
    what the probe's numerics self-check measures against an f64 reference.
    """

    encoding: str

    @property
    def name(self) -> str:
        return "cozy.fp8-scaled-scalar.dequant/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        # A CAST, not a matmul: e4m3 -> float is available wherever torch has the dtype,
        # which is every CUDA card torch builds for and the CPU. The sm90 floor belongs to
        # the NATIVE route (`torch._scaled_mm`), which is a different provider and is not
        # built here — see NAMED_SEAMS.
        return "any"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"data", "weight_scale"})

    @property
    def optional(self) -> frozenset[str]:
        # `input_scale` is the ACTIVATION scale of the native fp8 matmul. The two spec
        # digests under this alias differ by exactly its presence, and WHICH digest a
        # tensor cites is checked by TensorFS against the header's own closure (its fact,
        # its owner) — this provider serves both because the value law is identical.
        return frozenset({"input_scale"})

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        data = parts["data"]
        weight_scale = parts["weight_scale"]
        # The scale is a rank-0 f32. Multiplying in the DESTINATION dtype rather than in
        # f32-then-cast is deliberate: the destination is the compute contract, and an
        # intermediate f32 tensor of the full weight would double this route's scratch.
        destination.copy_(data.view(destination.shape).to(destination.dtype))
        destination.mul_(weight_scale.to(destination.dtype).reshape(()))


#: The OCP microscaling block: one shared power-of-two scale per this many elements along
#: the LAST axis. Transcribed from the registry's `mxfp8/1` spec relation
#: (`ceil_div(axis 1, by 32)`) and from nothing else; the provider below asserts the
#: delivered scale grid against it rather than against a number spelled twice.
MX_BLOCK = 32

#: E8M0's bias: a scale byte `b` denotes exactly `2**(b - 127)`.
E8M0_BIAS = 127

#: E8M0's ONE non-numeric code point. The registry entry's prose says only "scale byte b
#: means 2^(b-127)", which at `b = 255` reads as `2**128`; the encoding's own NORMATIVE
#: decoder — the conform corpus this spec is qualified against — returns NaN there, which
#: is OCP's reading. Prose and reference disagree and the REFERENCE wins, because the
#: corpus is what a decoder is checked against bit-for-bit. `2**128` would also be a
#: plausible finite-looking overflow, and serving a poisoned block as `inf` where the
#: contract says NaN is the silent-wrong-output class this whole module exists to prevent.
E8M0_NAN = 255


@dataclass(frozen=True, slots=True)
class MicroScaledDequant:
    """`mxfp8/1` -> float, weight-only: `value = data * 2**(scale - 127)` per 32-block.

    The registry's spec is the whole contract and is transcribed from it: E4M3 elements in
    a `data` role shaped exactly like the logical tensor, and one E8M0 exponent byte per
    32-element block along the LAST axis in a `scale` role shaped `[rows, ceil(cols/32)]`.
    Rank is 2 because the spec says `logical_rank: 2`.

    **Why DECODED FLOAT and not native.** MXFP8's whole point on Blackwell is that the
    tensor cores consume E4M3 elements and E8M0 block scales DIRECTLY. That route is real
    and, since `scripts/mxfp8-native-probe.py`, MEASURED: on an RTX 5090 it needs no new
    kernel at all — `torch._scaled_mm`'s Blockwise-1x32 recipe already ships it, and it
    agrees with the decode below bit for bit across the format domain. What it needs is a
    LEAF that holds the payload instead of a float destination, which is a different plane's
    machinery (`delivery.py`'s `native_encoded` rung), so the route that is servable TODAY
    is still this one: dequantize each block into the destination's own float dtype at fill,
    compute in float. It buys bytes read and load time, NOT VRAM — the resident weights are
    float either way, exactly as the fp8 route's docstring says of itself, and the native
    route's measured 8.294 resident bits per element is precisely what this one cannot offer.
    It is also the more ACCURATE of the two on the same bytes, because it never quantizes an
    activation: 0.0300 relative Frobenius against bf16 where native w8a8-mx measured 0.0421.
    `NAMED_SEAMS` carries that whole matrix with the device class it was measured on.

    **Why the multiply lands in the DESTINATION dtype.** The alternative — promote the
    PAYLOAD to f32, multiply, cast down — allocates a second full-width copy of every
    weight, which this plane prices as decode scratch and would then be carrying
    unledgered. It is also unnecessary: an e4m3 element is exact in bf16 and f32 (4
    significant bits into 8 or 24), the block multiplier is an exact power of two, so the
    product carries ONE rounding and the measured error over the whole format domain is
    ZERO. The multiplier grid itself is 1/32 of the weight, so `_multiplier` builds it in
    f32 and narrows once — negligible bytes, one exact code path.

    What f16 cannot do is hold the multiplier: `2**(b-127)` leaves f16's range below
    b = 103, so a block whose PRODUCT would be representable decodes as zero. That is not
    papered over — `probe.py` measures the full domain per output dtype, f16 mints no
    record, and an f16 mxfp8 destination refuses `encoding_unqualified` rather than being
    served silently wrong. Widening f16 needs a capability record that can carry a qualified
    exponent RANGE, which the tfs-013 key cannot express today.
    """

    encoding: str

    @property
    def name(self) -> str:
        return "cozy.mxfp8.dequant/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        # A cast plus a broadcast multiply — no matmul, no block-scale kernel. Available
        # wherever torch carries the e4m3 dtype, which is the same argument the scalar fp8
        # route makes. The sm100 floor belongs to the NATIVE route in NAMED_SEAMS.
        return "any"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"data", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    @staticmethod
    def _multiplier(torch: Any, scale: Any, dtype: Any) -> Any:
        """One E8M0 byte grid -> the block multipliers, BY BIT CONSTRUCTION.

        An E8M0 byte IS an IEEE binary32 exponent field: `b` in 1..254 denotes exactly the
        f32 whose bit pattern is `b << 23`, mantissa zero. So the multiplier is built, not
        computed — no `exp2`, no `pow`, nothing whose accuracy is a property of the backend's
        math library. That is the whole reason this is written the hard way: measured on an
        L40S, `torch.exp2` over the same domain is correct in bf16 and off by up to 2.3 ULP
        in f32 for the smallest scale bytes, which failed 46 of 65,536 pairs against a
        two-rounding budget. Loosening the budget to admit it would have been the wrong
        repair — the decode's error should be ZERO, and a transcendental in the path is what
        made it non-zero. It is also the only spelling that cannot drift between a CPU
        qualification and the CUDA decode it qualifies.

        Two code points are not exponents and are placed by hand: `b = 0` denotes `2**-127`,
        one binade below f32's smallest NORMAL, so it is the subnormal `1 << 22`; and
        `b = 255` is E8M0's NaN, which `b << 23` would spell as a signed infinity.
        """
        bits = scale.to(torch.int32) << 23
        bits = torch.where(scale == 0, torch.full_like(bits, 1 << 22), bits)
        # f32 first, in one code path, because its exponent range covers the whole byte
        # domain exactly; narrowing to the destination is then a single exact conversion
        # wherever the destination can hold the value, and 0 or inf where it cannot — which
        # is exactly what the probe measures per output dtype.
        mult = bits.view(torch.float32).to(dtype)
        return torch.where(scale == E8M0_NAN, torch.full_like(mult, float("nan")), mult)

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        data = parts["data"]
        scale = parts["scale"]
        dtype = destination.dtype
        rows, cols = int(destination.shape[0]), int(destination.shape[1])
        blocks = int(scale.shape[-1])
        # The block multiplier: one value per block, so this tensor is 1/32 of the weight
        # and forming it costs nothing worth accounting. `.to(dtype)` on a u8 in 0..255 is
        # exact in every float dtype; the subtraction and the exp2 are exact wherever the
        # result is representable, which is what the probe proves rather than assumes.
        mult = self._multiplier(torch, scale, dtype)
        if cols == blocks * MX_BLOCK and destination.is_contiguous():
            # THE PATH EVERY REAL ARTIFACT TAKES. A 3-D view over both sides broadcasts the
            # block scale with no intermediate at all: no repeat, no f32 copy, one kernel.
            view = destination.view(rows, blocks, MX_BLOCK)
            view.copy_(data.view(rows, blocks, MX_BLOCK).to(dtype))
            view.mul_(mult.unsqueeze(-1))
            return
        # The RAGGED tail (`ceil_div` admits a last block shorter than 32) and the
        # non-contiguous destination, both handled by widening the multiplier instead of
        # viewing the weight. It costs one destination-sized temporary and is written as the
        # slow path on purpose: a 3-D view cannot span a partial block or a strided slice,
        # and quietly taking the fast path anyway would mis-stride the scales.
        destination.copy_(data.view(destination.shape).to(dtype))
        destination.mul_(mult.repeat_interleave(MX_BLOCK, dim=-1)[..., :cols])


#: E4M3's largest finite magnitude. The cast to `float8_e4m3fn` DOES NOT SATURATE — an
#: unclamped 449 becomes NaN, not 448 — so every path that rounds into e4m3 clamps FIRST.
E4M3_MAX = 448.0

#: How far past 448 a value may sit and still be explained by the three roundings between
#: `rowmax|V|` and `payload * mult / s`. Beyond it the row scale genuinely failed to cover
#: its row, which is a different fact and is counted separately.
SATURATION_MARGIN = 1.0 + 2.0**-10

#: The floor under a derived row scale. A row of exact zeros has amax 0, and a zero scale
#: is a division by zero that would poison an entire output row with NaN. Spelled as a
#: constant rather than an inline literal because it is a POLICY (v1 carries the same floor
#: on the same quantity) and a reader must be able to find it.
ROW_SCALE_FLOOR = 1e-12


@dataclass(frozen=True, slots=True)
class RowwiseDerivation:
    """What `derive_rowwise` produced, and what it cost. Returned rather than logged: the
    error IS the product here, because the derivation is a second lossy quantization and a
    caller that cannot see its cost cannot gate on it."""

    data: Any
    """E4M3 payload, the logical shape. `fp8-rowwise/1`'s `data` role."""
    scale: Any
    """f32, one CONTINUOUS scale per output row. `fp8-rowwise/1`'s `scale` role."""
    clamped: int
    """Elements the 448 clamp caught AT ALL, boundary rounding included."""
    saturated: int
    """Elements that overshot 448 by more than rounding can explain — the ones that mean
    the row scale genuinely did not cover its row. `clamped` minus this is benign: the row
    maximum is SUPPOSED to land on 448, three roundings can put it a few ULP over, and
    clamping it back to 448 yields the very value the cast would have produced anyway."""
    max_rescaled: float
    """The largest magnitude offered to the cast, before clamping. The number that lets a
    reader tell 448.00003 (rounding) from 900 (lost range) instead of guessing."""
    zero_rows: int
    """Rows whose amax was 0, held off the divide by `ROW_SCALE_FLOOR`."""


def derive_rowwise(
    torch: Any, data: Any, scale: Any, *, block: int = MX_BLOCK
) -> RowwiseDerivation:
    """`mxfp8/1` roles -> `fp8-rowwise/1` roles. A SECOND LOSSY QUANTIZATION (#514a).

    Hopper's fp8 tensor cores take one scale per row, not one per 32-element block, so
    serving mxfp8 there means collapsing the block scales into a continuous row scale and
    RE-ROUNDING the payload. Both halves of that lose information and neither is hidden:

    * the row scale is `rowmax|V| / 448`, so the row's largest magnitude lands exactly on
      E4M3's largest. That is v1's rule for the same quantity, and it makes the derived
      form's dynamic range the ROW's rather than the BLOCK's — which is precisely the
      information mxfp8 had and `fp8-rowwise/1` cannot carry.
    * each block's payload is then rescaled by `mult[r,b] / s[r]` and re-rounded to E4M3,
      ROUND-TO-NEAREST-EVEN (torch's `.to()` on this dtype), clamped to +/-448 BEFORE the
      cast because that cast does not saturate. A block far below its row's maximum loses
      mantissa bits here; a block at the maximum loses none.

    The ratio is NOT a power of two — `mult` is, `s[r]` is continuous — so the rescale is a
    real multiply and a real rounding, not an exponent shift. That is the whole source of
    the double-quantization penalty, and `scripts/mxfp8-live.py` measures it against a
    DIRECT bf16 -> fp8-rowwise quantization of the same weights rather than asserting it.

    **Structured blockwise on purpose.** `rowmax|V|` is computed as
    `max_b(blockmax|payload| * mult[r,b])` — in the PAYLOAD domain, never by materializing
    the dequantized weight — so the derivation's working set is the payload plus two grids
    of 1/32 its size. A serving path that had to build the bf16 tensor first to produce the
    fp8 one would defeat the entire point of the lane.
    """
    rows, cols = int(data.shape[0]), int(data.shape[1])
    blocks = int(scale.shape[-1])
    if cols != blocks * block:
        raise AssertionError(
            f"a ragged mxfp8 tail ({cols} cols, {blocks} blocks) has no rowwise derivation"
        )
    mult = MicroScaledDequant._multiplier(torch, scale, torch.float32)
    payload = data.view(rows, blocks, block).to(torch.float32)
    # The row's true maximum magnitude, exactly, without ever forming the row.
    #
    # E8M0's NaN byte is EXCLUDED from the maximum rather than allowed to propagate into
    # it. A NaN block scale means THAT BLOCK is not a number; letting it set the row scale
    # turns 32 poisoned elements into a poisoned ROW, which is a strictly larger claim than
    # the stored bytes make. The NaN still reaches the output — `ratio` carries it, so the
    # block's own 32 payload elements cast to E4M3's NaN — it just stops there. The
    # producing job clamps to 254 and never emits the byte; a derivation that is only
    # correct on its own producer's output is not a derivation.
    block_peak = payload.abs().amax(dim=-1) * mult
    amax = torch.nan_to_num(block_peak, nan=0.0).amax(dim=-1)
    zero_rows = int((amax == 0).sum())
    row_scale = (amax / E4M3_MAX).clamp(min=ROW_SCALE_FLOOR)
    ratio = mult / row_scale.unsqueeze(-1)
    rescaled = payload * ratio.unsqueeze(-1)
    magnitude = rescaled.abs()
    # nan-safe: a NaN block must not erase the peak a reader uses to tell rounding-boundary
    # clamping from genuinely lost range.
    peak = float(torch.nan_to_num(magnitude, nan=0.0).max())
    clamped = int((magnitude > E4M3_MAX).sum())
    saturated = int((magnitude > E4M3_MAX * SATURATION_MARGIN).sum())
    derived = rescaled.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(rows, cols)
    return RowwiseDerivation(
        derived, row_scale.to(torch.float32), clamped, saturated, peak, zero_rows
    )


def quantize_rowwise(torch: Any, weight: Any) -> RowwiseDerivation:
    """bf16 -> `fp8-rowwise/1` DIRECTLY, in one quantization. The derivation's control.

    Same rule, same clamp, same floor — the only difference is that this one sees the
    original weight and `derive_rowwise` sees a weight that has already been through mxfp8.
    Subtracting the two is what turns "a second quantization is lossy" from a statement
    into a number.
    """
    work = weight.to(torch.float32)
    amax = work.abs().amax(dim=-1)
    zero_rows = int((amax == 0).sum())
    row_scale = (amax / E4M3_MAX).clamp(min=ROW_SCALE_FLOOR)
    rescaled = work / row_scale.unsqueeze(-1)
    magnitude = rescaled.abs()
    return RowwiseDerivation(
        rescaled.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn),
        row_scale.to(torch.float32),
        int((magnitude > E4M3_MAX).sum()),
        int((magnitude > E4M3_MAX * SATURATION_MARGIN).sum()),
        float(magnitude.max()),
        zero_rows,
    )


#: Encodings this runtime RECOGNISES but does not execute, with the device class that
#: would and the milestone that owns building it. Data, not machinery: there is no stub
#: provider object, no empty package and no dormant enum case — the refusal below reads
#: this table to name the alternative, which is the only consumer it has.
NAMED_SEAMS: dict[str, tuple[str, str]] = {
    "fp8-scaled-scalar/1": (
        "cuda.sm90+",
        "native fp8 tensor-core execution via torch._scaled_mm — cr-008c's native-provider "
        "leg; on sm89 the qualified route is decoded_float, which this runtime serves",
    ),
    "fp8-rowwise/1": (
        "cuda.sm90+",
        "per-output-row fp8; the decoded_float provider is a small generalisation of the "
        "scalar one and lands with its first consuming lane (cr-008c)",
    ),
    "mxfp8/1": (
        "cuda.sm120",
        "MXFP8 NATIVE — tensor cores consume E4M3 elements + E8M0 block scales directly. "
        "MEASURED on an RTX 5090 (sm120, torch 2.9.1+cu130, driver 580.126.20), "
        "`scripts/mxfp8-native-probe.py`: `torch._scaled_mm`'s Blockwise-1x32 recipe DOES "
        "execute it, and its arithmetic is BIT-IDENTICAL to `cozy.mxfp8.dequant/1` over all "
        "64,770 non-NaN pairs of the format domain (the 766 NaN-bearing pairs are excluded "
        "because `0 * NaN` is NaN, which poisons a dot product on BOTH routes — a matmul "
        "fact, not an encoding one). It runs the real DiT block's GEMMs at 628-674 TFLOP/s "
        "against bf16's 227-243, and holds the weights at a MEASURED 8.294 bits resident "
        "(18.60 GiB vs 35.96 GiB for the curve DiT's GEMM weights) — the VRAM claim #511's "
        "open owner question says the decoded-float route cannot make. "
        "STILL UNBUILT, and the missing piece is NOT this module: a native route needs the "
        "LEAF to hold the payload plus its blocked scale and to replace the linear op, "
        "where `Provider.decode` writes a float destination the contract's dtype named. "
        'That is `delivery.py`\'s `native_encoded` rung, still `status="unbuilt"` for the '
        "reason it already gives — LEAF REPLACEMENT — though its `owner` line's \"this card "
        "is sm89\" is now stale and belongs to that file's lane to correct. "
        "Two facts a builder needs before starting. (1) The route "
        "is w8a8-mx, not weight-only: BOTH operands must be mxfp8, so it quantizes "
        "ACTIVATIONS, measured at 0.0421 relative Frobenius against bf16 where the "
        "weight-only lane sits at 0.0300 — 2.9 dB lossier, and the kernel itself accounts "
        "for almost none of it (0.00166 against the same operands in f32). (2) torch "
        "validates the scale grids' ELEMENT COUNT, not their LAYOUT: a row-major "
        "`[rows, cols/32]` grid is ACCEPTED in place of the 128x4-tiled one the MMA reads "
        "and returns wrong numbers at 62,453 of 64,485 finite positions. That is this "
        "module's silent-wrong-output class living one layer down, and it is why a native "
        "provider must qualify through `probe.py` on the arch it claims. sm100 (B200) is "
        "UNMEASURED — no card was rented, so it gets no record and no floor; the "
        "predicate above is `cuda.sm120` exactly, not `cuda.sm100+`, because a hand-written "
        "floor is what #511b's fail-closed-by-measurement rule forbids",
    ),
    "nvfp4-w4a4/1": (
        "cuda.sm100+",
        "the v1 SVDQuant/W4A4 native stack, ported as providers with its layout contracts "
        "already registered as encoding rules; sequences with its consuming lane's tier",
    ),
    "svdq-microscale/1": (
        "cuda.sm100+",
        "svdq micro-scale, same port as nvfp4-w4a4/1; its wscale geometry is PROVISIONAL "
        "pending producer-mined confirmation (tfs-008)",
    ),
}


def launch_providers() -> dict[str, Any]:
    """The LAUNCH TIER provider set, keyed by spec digest.

    Read from the compiled registry at call time so a digest is never transcribed. The set
    is deliberately small: `plain/1` plus the encoding proto-001 SELECTED. Everything else
    the registry knows is in `NAMED_SEAMS` and refuses naming its device class.
    """
    out: dict[str, Any] = {}
    for digest in digests_of("plain/1"):
        out[digest] = Verbatim(digest)
    # BOTH role sets one real fp8_scaled artifact carries. They share one value law, so
    # they share one implementation; which digest a given tensor may cite is TensorFS's
    # fact, enforced when the header is parsed against its own encoding closure.
    for digest in digests_of("fp8-scaled-scalar/1"):
        out[digest] = ScaledScalarDequant(digest)
    # #469a's PREFERRED H3 DiT serving lane. One digest under this alias — the block
    # geometry IS the spec, so a producer variation would be a different spec and a
    # different provider, not a tolerated role set.
    for digest in digests_of("mxfp8/1"):
        out[digest] = MicroScaledDequant(digest)
    return out


# ------------------------------------------------------------------- capability records


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """The tfs-013 key, kept whole: an (encoding, implementation, device) qualification.

    A record cannot exist without `evidence_digest`, which is the digest of the probe
    observation that minted it — so "qualified" always means "something ran and was
    measured", never "someone believed".
    """

    encoding_spec_id: str
    implementation_digest: str
    runtime_release: str
    device_predicate: str
    output_dtype: str
    delivery_route: str
    evidence_digest: str


class Capabilities:
    """The fail-closed admission table. No default branch, no "unknown means allowed"."""

    def __init__(self, records: Sequence[CapabilityRecord] = ()) -> None:
        self.records = list(records)

    def add(self, record: CapabilityRecord) -> None:
        self.records.append(record)

    def admit(
        self, encoding: str, device: DeviceFacts, output_dtype: str, alias: str
    ) -> CapabilityRecord:
        for record in self.records:
            if (
                record.encoding_spec_id == encoding
                and record.output_dtype == output_dtype
                and device.satisfies(record.device_predicate)
            ):
                return record
        elsewhere = sorted(
            {r.device_predicate for r in self.records if r.encoding_spec_id == encoding}
        )
        seam = NAMED_SEAMS.get(alias)
        alternative = f" The route that WOULD serve it is {seam[0]}: {seam[1]}." if seam else ""
        raise refuse(
            "encoding_unqualified",
            f"{alias} ({encoding[:23]}…) has no qualified provider on "
            f"{device_line(device)} for output dtype {output_dtype}. Qualified device "
            f"predicates for this encoding: {elsewhere or 'none'}.{alternative} An "
            "unrecognised device is the ABSENCE of evidence, never permission.",
            [encoding],
        )


def device_line(device: DeviceFacts) -> str:
    """The device FACT, spelled the same way in every capability refusal. A refusal that
    says only "unsupported" sends a reader to the wrong question."""
    return (
        f"{device.predicate} ({device.name}, sm{device.sm}, driver cuda {device.driver}, "
        f"config {device.configuration})"
    )


# ------------------------------------------------------------------------- the selector


@dataclass(frozen=True, slots=True)
class Selection:
    """What the fill plane needs to move and decode one tensor."""

    provider: Any
    record: CapabilityRecord
    encoded: Encoded


class Selector:
    """Per-tensor provider selection, by digest, fail-closed, priced at O(1) per key."""

    def __init__(
        self, providers: Mapping[str, Any], capabilities: Capabilities, device: DeviceFacts
    ) -> None:
        self.providers = dict(providers)
        self.capabilities = capabilities
        self.device = device
        self.aliases = aliases()

    def select(self, key: str, encoded: Encoded, output_dtype: str) -> Selection:
        provider = self.providers.get(encoded.encoding)
        if provider is None:
            alias = self.aliases.get(encoded.encoding)
            if alias is None:
                raise refuse(
                    "unknown_encoding",
                    f"{key}: the checkpoint cites encoding {encoded.encoding} and this "
                    "runtime's registry does not contain that spec at all — the pinned "
                    "registry plus reader set IS this runtime's encoding capability, and "
                    "an unregistered encoding refuses before generation rather than being "
                    "guessed from the key or the carrier dtypes",
                    [key],
                )
            seam = NAMED_SEAMS.get(alias)
            if seam is None:
                raise refuse(
                    "unknown_encoding",
                    f"{key}: {alias} is a REGISTERED encoding with no provider in this "
                    "runtime build and no named seam — the registry knows the bytes and "
                    "nothing here executes them",
                    [key],
                )
            # A registered encoding whose provider is a NAMED, UNBUILT seam is a DEVICE
            # CAPABILITY statement, not an unknown one, so it refuses in the capability
            # vocabulary and names the card it is standing on. The alternative reading —
            # "unknown encoding" — would send a reader looking for a registry problem
            # that does not exist.
            raise refuse(
                "encoding_unqualified",
                f"{key}: {alias} is registered and this runtime has no provider for it on "
                f"{device_line(self.device)}. The route that WOULD serve it is {seam[0]}: "
                f"{seam[1]}. Named and not built — never silently served by a route it was "
                "not qualified for.",
                [key],
            )
        missing = sorted(provider.required - encoded.roles)
        surplus = sorted(encoded.roles - provider.required - provider.optional)
        if missing or surplus:
            raise refuse(
                "role_mismatch",
                f"{key}: the header stores roles {sorted(encoded.roles)}; the provider "
                f"selected by this tensor's own digest requires {sorted(provider.required)} "
                f"and tolerates {sorted(provider.optional)} — missing {missing}, "
                f"unhandled {surplus}. A role a decoder cannot account for is a different "
                "physical contract, never a tolerated variant",
                [key],
            )
        record = self.capabilities.admit(
            encoded.encoding, self.device, output_dtype, self.aliases.get(encoded.encoding, "?")
        )
        return Selection(provider, record, encoded)


# ------------------------------------------------------------------------------ digests


def implementation_digest(provider: Any) -> str:
    """The provider's identity: its name, route, device floor and exact role set."""
    return canonical.digest(
        {
            "name": provider.name,
            "route": provider.route,
            "device_predicate": provider.device_predicate,
            "required": sorted(provider.required),
            "optional": sorted(provider.optional),
        }
    )
