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

**The device-honesty split, stated once, and now split BOTH WAYS.** The original reading
was that on sm89 fp8 E4M3 is a STORAGE dtype and the only servable route is
`decoded_float` — dequantize at fill, compute in the destination's float dtype. That
remains the FLOOR and it remains what an unmeasured device gets. What changed is that it
is no longer the only route this module builds: #517c measured rowwise `torch._scaled_mm`
WORKING on sm89, which makes v1's `W8A8_ROWWISE_MIN_SM=90` a profitability floor that had
been mislabelled a capability floor for two years.

So an encoding here can carry TWO providers on one device:

* a **decode** provider (`route="decoded_float"`) — the weights become float at fill. It
  buys bytes read and load time, NOT VRAM, and it is the more accurate of the two because
  it never quantizes an activation.
* a **native LEAF** provider (`route="native_encoded"`) — the stored roles ARE the resident
  weight and the module that multiplies by them is REPLACED. It buys ~half the VRAM and
  the fp8 GEMM, and it costs fidelity, because native fp8/mx tensor cores need BOTH
  operands encoded and the activation is therefore quantized at dispatch (w8a8).

**Those two order OPPOSITELY** (#517d measured 20.82 dB at 1.0x VRAM against 16.72 dB at
0.5003x), which is why selection among them is a stated OBJECTIVE ranked over measured
records — `Capabilities.admit` — and never registration order. Both defects #517e named
are fixed here: `launch_providers` is digest -> ORDERED CANDIDATES, and admission RANKS.

MXFP8 (#469a's preferred H3 DiT lane) has both. The decode route was built first and is
unchanged. Its native door was measured before it was built: `scripts/mxfp8-native-probe.py`
rented an RTX 5090 and found that on sm120 the block-scaled MXFP8 GEMM is real, that its
arithmetic is bit-identical to `cozy.mxfp8.dequant/1` over the whole format domain, that it
runs ~2.7x the bf16 GEMM, and that it is the only route that makes the weights RESIDENT at
8.294 bits. What that lane could not do was FILL through it — the native route replaces the
LEAF, where `Provider.decode` writes a float destination the contract's dtype named. That
is the half this module's `LeafProvider` and `fill.py`'s leaf-replacement leg are.

A record is still minted only where a probe MEASURED (#511b). The native leaf's oracle runs
the real kernel on the real arch and compares it to the decode reference, so a card whose
suite never ran mints nothing and refuses to the floor — which is exactly what an
sm89 build does with `MicroScaledNativeLeaf` (`cuda.sm120`, and no B200 ever ran either).
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
        # Two or more providers of one encoding are qualified on this device and the
        # deployment's stated objective names no axis that was MEASURED on all of them.
        # Producible only since #517e's digest -> ORDERED CANDIDATES: with one candidate
        # there is nothing to rank, and the objective decides nothing.
        "objective_unranked",
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

    DECODING providers present a PLAIN compute-dtype tensor at the leaf boundary and need
    no adapter awareness (cr-006's scope note): the SDK side-branch wraps the leaf, cr-010.
    A `native_encoded` provider does not decode at all — it REPLACES the leaf, and its
    surface is `LeafProvider` below.
    """

    @property
    def name(self) -> str: ...

    @property
    def route(self) -> str:
        """`verbatim`, `decoded_float` or `native_encoded` — the delivery route (§3.2)."""
        ...

    @property
    def device_predicate(self) -> str: ...

    @property
    def suite(self) -> str:
        """WHICH probe oracle qualifies this implementation.

        Keyed on the DECLARATION and never on the class, so a second implementation of one
        encoding is checked by the same oracle as the first — which is the only way two
        implementations' records mean the same thing. It replaced probe.py's role-set key
        the moment two providers of one encoding declared the SAME role set and different
        routes (`mxfp8/1`'s decode and its native leaf both take `{data, scale}`), where
        a role-set key silently hands one provider the other's oracle.
        """
        ...

    @property
    def required(self) -> frozenset[str]:
        """Roles this provider MUST be handed. A header short of one refuses."""
        ...

    @property
    def optional(self) -> frozenset[str]:
        """Roles it tolerates. A header carrying anything else refuses."""
        ...

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None: ...


class LeafProvider(Protocol):
    """A MODULE-REPLACING provider: the stored roles stay stored and the leaf changes.

    `Provider.decode` writes a float destination the contract's dtype named, which is the
    whole reason the `native_encoded` rung stayed unbuilt through two lanes: on the native
    route there IS no float destination — the payload and its scales are the resident
    weight, and what has to change is the module that multiplies by them.

    `leaf()` returns a real `torch.nn.Module`, and it must be one: `nn.Module.__setattr__`
    refuses to accept anything else as a child, which is what makes leaf replacement
    machinery rather than a one-line swap (proved by the `_fp8_leaf` prototype in
    `scripts/mxfp8-pod.py`, which this generalises).

    `resident_bits_per_element` is the FOOTPRINT axis of #515a's split, computed from the
    role bytes that stay resident rather than declared: on the native route those bytes ARE
    the weight, which is the residency a decoded-float route structurally cannot offer.
    """

    @property
    def output_dtype_rule(self) -> str:
        """How the leaf's forward picks its output dtype. `contract` is the only rule this
        build has: the leaf emits the destination dtype the contract named, so a downstream
        module sees exactly what the float path would have handed it."""
        ...

    def leaf(self, torch: Any, parts: Mapping[str, Any], replaced: Any, out_dtype: Any) -> Any: ...

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float: ...


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
    def suite(self) -> str:
        return "verbatim"

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
    def suite(self) -> str:
        return "fp8-scalar-decode"

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
        # route makes. The sm120 floor belongs to the NATIVE LEAF, which is a different
        # provider under this same digest — see `MicroScaledNativeLeaf`.
        return "any"

    @property
    def suite(self) -> str:
        return "mxfp8-decode"

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


@dataclass(frozen=True, slots=True)
class RowwiseDequant:
    """`fp8-rowwise/1` -> float, weight-only: `value[i,j] = data[i,j] * scale[i]`.

    The spec's own equation, transcribed from the registry entry. It serves BOTH physical
    variants the alias files — rank-1 `[out]` and rank-2 `[out, 1]` — because the value law
    is identical and the only difference is a trailing unit axis, which `reshape(rows, 1)`
    resolves for both. That is the same argument `ScaledScalarDequant` makes about its two
    digests, and the same guard applies: a role set that does not match still refuses.

    This is the DEQUANT FLOOR of #518's Hopper/Ada artifact — the route that serves the
    fp8-rowwise bytes on a device where nothing measured the native leaf. It exists so that
    "the chooser picks the leaf where it was measured and the floor where it was not" is a
    statement about two real providers rather than about a provider and an absence.
    """

    encoding: str

    @property
    def name(self) -> str:
        return "cozy.fp8-rowwise.dequant/1"

    @property
    def route(self) -> str:
        return "decoded_float"

    @property
    def device_predicate(self) -> str:
        # A cast and a broadcast multiply, exactly like the other two decode routes.
        return "any"

    @property
    def suite(self) -> str:
        return "fp8-rowwise-decode"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"data", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        data = parts["data"]
        scale = parts["scale"]
        rows = int(destination.shape[0])
        destination.copy_(data.view(destination.shape).to(destination.dtype))
        destination.mul_(scale.to(destination.dtype).reshape(rows, 1))


# ------------------------------------------------------------------- the encoded leaf

#: The activation scale's floor, and the same policy `ROW_SCALE_FLOOR` states for weights:
#: a row of exact zeros has amax 0, and dividing by it poisons a whole output row with NaN.
ACT_SCALE_FLOOR = 1e-12


def quantize_activation_rowwise(torch: Any, x: Any) -> tuple[Any, Any]:
    """`[..., K]` float -> `[M, K]` E4M3 payload + `[M, 1]` f32 per-TOKEN scale.

    v1's proven recipe (`gen_worker/models/w8a8.py::_quant_src`), transcribed: amax over the
    reduction axis, `.float()` BEFORE the divide so the scale is f32 whatever the activation
    is, the literal 448.0 rather than `finfo.max`, a floor, and the clamp BEFORE the cast
    because the cast does not saturate. One scale per ROW of the `[M, K]` operand is what
    pairs with a per-output-row weight scale: the two scale vectors are the two OUTER
    dimensions of the product and neither is the reduction axis.

    #515f is banked against this spelling and not hidden by it: unfused, this is six
    elementwise passes over `[M, K]` around a GEMM that costs `M*K*N`, so the overhead
    fraction falls as 1/N and only disappears when the quantize is fused into one kernel.
    """
    flat = x.reshape(-1, x.shape[-1])
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


def to_blocked(torch: Any, scales: Any) -> Any:
    """`[rows, cols]` E8M0 bytes -> cuBLASLt's 128x4 blocked ("swizzled") scale layout.

    #515d IS THIS FUNCTION'S REASON TO EXIST, and it is a silent-wrong-output hazard rather
    than an optimisation: torch validates the MX scale grids' ELEMENT COUNT, not their
    LAYOUT, so a plain row-major `[rows, cols/32]` grid is ACCEPTED in place of the tiled
    one the MMA actually reads — and returns wrong numbers at 62,453 of 64,485 finite
    positions. Nothing raises. The probe oracle is what catches a builder who skips this.
    """
    rows, cols = int(scales.shape[0]), int(scales.shape[1])
    row_blocks = -(-rows // 128)
    col_blocks = -(-cols // 4)
    padded = torch.zeros(row_blocks * 128, col_blocks * 4, dtype=scales.dtype, device=scales.device)
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
        """One linear op whose WEIGHT IS THE ENCODED ROLE SET, computed natively.

        It stands exactly where the module it replaced stood, takes exactly that module's
        input and returns exactly that module's output dtype and shape. What it does not
        have is a `.weight`: an endpoint that reaches for one is why author CONSENT is a
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
    """`fp8-rowwise/1` NATIVE: the tensor cores consume the stored E4M3 payload directly.

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
        return "native_encoded"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm89+"

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

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        raise AssertionError(
            "a native_encoded provider REPLACES the leaf; it never writes a float "
            "destination — see LeafProvider.leaf"
        )

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float:
        return 8.0 * sum(p.nbytes for p in parts.values()) / max(numel, 1)

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
    """`mxfp8/1` NATIVE: block-scaled MXFP8 straight into the tensor cores (#515).

    #515 measured every premise of this provider on real silicon before a line of it
    existed: on sm120 stock `torch._scaled_mm`'s Blockwise-1x32 recipe executes MXFP8
    natively, its arithmetic is BIT-IDENTICAL to `cozy.mxfp8.dequant/1` over 64,770 of
    64,770 non-NaN pairs of the format domain, it runs the real DiT shapes at 2.45-2.75x
    the bf16 GEMM, and it holds the weights at a MEASURED 8.294 bits per element — 18.60
    against 35.96 GiB for the curve DiT. What was missing was OUR fill plane, which is what
    this provider and `fill.py`'s leaf-replacement leg are.

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
        return "native_encoded"

    @property
    def device_predicate(self) -> str:
        return "cuda.sm120"

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

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        raise AssertionError(
            "a native_encoded provider REPLACES the leaf; it never writes a float "
            "destination — see LeafProvider.leaf"
        )

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float:
        return 8.0 * sum(p.nbytes for p in parts.values()) / max(numel, 1)

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
                "an MX block-scale operand at all — the native mxfp8 leaf has no way to "
                "hand the kernel its scales, and refusing beats handing it a u8 grid",
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
    # `fp8-rowwise/1` and `mxfp8/1` LEFT THIS TABLE when their native leaves were built
    # (`RowwiseNativeLeaf`, `MicroScaledNativeLeaf`). A seam entry means "this runtime
    # recognises the bytes and executes none of them"; both now have TWO providers each, a
    # decode floor and a native leaf, and the honest answer for a device that serves neither
    # is the candidate list `Capabilities.admit` prints — which names the implementations
    # that exist and what each one measured — not a sentence about a door nobody walked.
    # sm100 (B200) stays UNMEASURED: no card was rented, so it gets no record and no floor.
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


def launch_providers() -> dict[str, tuple[Any, ...]]:
    """The LAUNCH TIER provider set: spec digest -> the ORDERED CANDIDATES for that digest.

    Read from the compiled registry at call time so a digest is never transcribed.

    **This used to be digest -> ONE provider**, which was true exactly as long as no
    encoding had a second implementation, and #517e named it as one of the two latent
    defects that fire the moment one does. Both fired together here: `fp8-rowwise/1` and
    `mxfp8/1` each now carry a DECODE FLOOR and a NATIVE LEAF, which are two providers of
    one encoding, on one device, with opposite standings on footprint and fidelity.

    The order in this function is REGISTRATION order and it decides NOTHING. Selection
    among candidates is `Capabilities.admit`'s ranking under the deployment's stated
    objective, over records that exist only where a probe measured — which is the other
    half of #517e, and the reason a first-match read of this dict would have been an
    order-dependence bug rather than a policy.
    """
    out: dict[str, list[Any]] = {}

    def offer(alias: str, *builders: Any) -> None:
        for digest in digests_of(alias):
            out.setdefault(digest, []).extend(build(digest) for build in builders)

    offer("plain/1", Verbatim)
    # BOTH role sets one real fp8_scaled artifact carries. They share one value law, so
    # they share one implementation; which digest a given tensor may cite is TensorFS's
    # fact, enforced when the header is parsed against its own encoding closure.
    offer("fp8-scaled-scalar/1", ScaledScalarDequant)
    # #518's Hopper/Ada artifact, and the first encoding to carry two routes. BOTH physical
    # variants of the alias (rank-1 `[out]` and rank-2 `[out, 1]` scales) get both.
    offer("fp8-rowwise/1", RowwiseDequant, RowwiseNativeLeaf)
    # #469a's H3 DiT serving lane. One digest under this alias — the block geometry IS the
    # spec, so a producer variation would be a different spec and a different provider, not
    # a tolerated role set. Two ROUTES under that one digest since #515.
    offer("mxfp8/1", MicroScaledDequant, MicroScaledNativeLeaf)
    return {digest: tuple(candidates) for digest, candidates in out.items()}


# ------------------------------------------------------------------- capability records


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """The tfs-013 key, kept whole: an (encoding, implementation, device) qualification.

    A record cannot exist without `evidence_digest`, which is the digest of the probe
    observation that minted it — so "qualified" always means "something ran and was
    measured", never "someone believed".

    The two MEASURED axes below are what turned this from a membership test into an
    ORDERING (#517e). They are stored on the record and not derived at admission because
    they are properties of a probe run: a number the table carries is a number something
    produced, and one nothing produced is spelled -1.0 rather than defaulted to a value
    that would win or lose a comparison it was never entitled to enter.
    """

    encoding_spec_id: str
    implementation_digest: str
    runtime_release: str
    device_predicate: str
    output_dtype: str
    delivery_route: str
    evidence_digest: str
    resident_bits_per_element: float = -1.0
    """FOOTPRINT axis. The bits this route leaves resident per logical element, measured on
    the probe's own tensor: the destination dtype's width on a decode route (the weights are
    float either way), the role bytes on a native one. -1.0 means NOT MEASURED."""
    route_deviation_rel: float = -1.0
    """FIDELITY axis. Relative Frobenius of the ROUTE's own GEMM output against an f32
    reference over the exactly-dequantized weights, on identical activations — the one
    quantity that is comparable across a decode route and a native one, because it prices
    the whole route rather than one of its halves. -1.0 means NOT MEASURED."""

    def axis(self, objective: str) -> float:
        """The ranking key for one objective. Lower is better; -1.0 means UNRANKABLE."""
        if objective == "footprint":
            return self.resident_bits_per_element
        if objective == "fidelity":
            return self.route_deviation_rel
        return -1.0


#: Which MEASURED field each delivery objective ranks providers on. `latency` is absent on
#: purpose and its absence is a refusal, not a fallback: nothing has measured a quantity
#: that is comparable between "a decode paid once at fill" and "a GEMM paid every step", so
#: a latency-stated deployment with two qualified providers is told to state an axis
#: something measured rather than handed whichever provider was registered first.
OBJECTIVE_AXIS: dict[str, str] = {
    "footprint": "resident_bits_per_element",
    "fidelity": "route_deviation_rel",
}


class Capabilities:
    """The fail-closed admission table. No default branch, no "unknown means allowed"."""

    def __init__(self, records: Sequence[CapabilityRecord] = ()) -> None:
        self.records = list(records)

    def add(self, record: CapabilityRecord) -> None:
        self.records.append(record)

    def admit(
        self,
        encoding: str,
        device: DeviceFacts,
        output_dtype: str,
        alias: str,
        *,
        objective: str,
        among: Mapping[str, Any],
    ) -> CapabilityRecord:
        """The qualified record for this (encoding, device, dtype), RANKED (#517e).

        `among` is implementation digest -> provider, for exactly the candidates whose
        declared role set the header satisfies. A record for an implementation this build
        does not carry admits nothing, and neither does a provider whose declared device
        floor this card misses — the two filters are separate because they answer different
        questions ("was this measured here" and "does this implementation claim this card"),
        and collapsing them would let a stale record select a provider that is gone.
        """
        qualified = [
            record
            for record in self.records
            if record.encoding_spec_id == encoding
            and record.output_dtype == output_dtype
            and device.satisfies(record.device_predicate)
            and record.implementation_digest in among
            and device.satisfies(among[record.implementation_digest].device_predicate)
        ]
        if not qualified:
            raise refuse(
                "encoding_unqualified",
                self._why(encoding, device, output_dtype, alias, among),
                [encoding],
            )
        if len(qualified) == 1:
            # ONE candidate: there is nothing to rank and the objective decides nothing.
            # Stated so the objective's own refusal below cannot fire on the single-provider
            # path every encoding took before this lane.
            return qualified[0]
        axis = OBJECTIVE_AXIS.get(objective)
        unranked = [r for r in qualified if r.axis(objective) < 0.0]
        if axis is None or unranked:
            raise refuse(
                "objective_unranked",
                f"{alias} ({encoding[:23]}…) has {len(qualified)} qualified providers on "
                f"{device_line(device)} and the deployment's {objective!r} objective "
                + (
                    f"names no measured axis (measured axes: {sorted(OBJECTIVE_AXIS)})"
                    if axis is None
                    else f"has no {axis} measurement on "
                    + ", ".join(sorted(among[r.implementation_digest].name for r in unranked))
                )
                + ". A number nobody measured may not select a provider, and neither may "
                "registration order — the two routes order OPPOSITELY on footprint and "
                "fidelity (#517d), so first-match is a coin flip wearing a ladder's clothes. "
                "Candidates: "
                + " | ".join(
                    f"{among[r.implementation_digest].name} [{r.delivery_route}] "
                    f"footprint={r.resident_bits_per_element:.3f} b/elt "
                    f"fidelity={r.route_deviation_rel:.6f} rel"
                    for r in qualified
                ),
                [encoding],
            )
        # Lower is better on both axes. The implementation NAME breaks an exact tie, so the
        # answer is a function of the measurements and not of dict iteration order.
        return min(
            qualified, key=lambda r: (r.axis(objective), among[r.implementation_digest].name)
        )

    def _why(
        self,
        encoding: str,
        device: DeviceFacts,
        output_dtype: str,
        alias: str,
        among: Mapping[str, Any],
    ) -> str:
        elsewhere = sorted(
            {r.device_predicate for r in self.records if r.encoding_spec_id == encoding}
        )
        seam = NAMED_SEAMS.get(alias)
        alternative = f" The route that WOULD serve it is {seam[0]}: {seam[1]}." if seam else ""
        # WHICH implementations exist and what each one's floor is. It replaces the seam
        # sentence for the encodings that now carry real providers: "no qualified provider"
        # is the same words whether nothing was built or nothing was measured, and those are
        # different problems with different remedies.
        built = " | ".join(
            f"{p.name} [{p.route}] floor {p.device_predicate}"
            f"{'' if device.satisfies(p.device_predicate) else ' — this card misses that floor'}"
            for p in sorted(among.values(), key=lambda p: str(p.name))
        )
        return (
            f"{alias} ({encoding[:23]}…) has no qualified provider on "
            f"{device_line(device)} for output dtype {output_dtype}. Qualified device "
            f"predicates for this encoding: {elsewhere or 'none'}. Implementations this "
            f"build carries: {built or 'none'}.{alternative} An unrecognised device is the "
            "ABSENCE of evidence, never permission."
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
    """Per-tensor provider selection, by digest, fail-closed, priced at O(1) per key.

    `objective` is the DEPLOYMENT's stated axis, carried down from the binding plan through
    the same field `DeliveryChooser` reads. It is a constructor argument rather than a
    per-call one because delivery is decided ONCE per generation (delivery.py's fourth
    rule): a per-tensor objective would let one component's leaves and another's disagree
    about what this deployment is for.
    """

    def __init__(
        self,
        providers: Mapping[str, Any],
        capabilities: Capabilities,
        device: DeviceFacts,
        objective: str = "fidelity",
    ) -> None:
        #: digest -> ORDERED CANDIDATES. A single provider is accepted and wrapped, so a
        #: caller holding a pre-#517e map still selects — through the ranked path, with one
        #: candidate, which is where that map was always right.
        self.providers = {
            digest: tuple(value) if isinstance(value, tuple | list) else (value,)
            for digest, value in providers.items()
        }
        self.capabilities = capabilities
        self.device = device
        self.objective = objective
        self.aliases = aliases()

    def select(self, key: str, encoded: Encoded, output_dtype: str) -> Selection:
        candidates = self.providers.get(encoded.encoding)
        if not candidates:
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
        # THE ROLE FILTER, applied to every candidate before capability. A candidate whose
        # declared role set the header does not satisfy is not a rejected provider, it is
        # not a provider OF THESE BYTES at all — so it must not appear in the ranking, and
        # its absence must not read as "unqualified on this device".
        among: dict[str, Any] = {}
        mismatched: list[tuple[Any, list[str], list[str]]] = []
        for candidate in candidates:
            missing = sorted(candidate.required - encoded.roles)
            surplus = sorted(encoded.roles - candidate.required - candidate.optional)
            if missing or surplus:
                mismatched.append((candidate, missing, surplus))
                continue
            among[implementation_digest(candidate)] = candidate
        if not among:
            raise refuse(
                "role_mismatch",
                f"{key}: the header stores roles {sorted(encoded.roles)} and no provider "
                f"registered for this tensor's own digest accounts for them — "
                + " | ".join(
                    f"{c.name} requires {sorted(c.required)}, tolerates {sorted(c.optional)} "
                    f"(missing {m}, unhandled {s})"
                    for c, m, s in mismatched
                )
                + ". A role a decoder cannot account for is a different physical contract, "
                "never a tolerated variant",
                [key],
            )
        record = self.capabilities.admit(
            encoded.encoding,
            self.device,
            output_dtype,
            self.aliases.get(encoded.encoding, "?"),
            objective=self.objective,
            among=among,
        )
        return Selection(among[record.implementation_digest], record, encoded)


# ------------------------------------------------------------------------------ digests


def implementation_digest(provider: Any) -> str:
    """The provider's identity: its name, route, device floor, oracle and exact role set.

    `suite` joins the key because it is part of what a record MEANS: two implementations
    checked by different oracles are not interchangeable, and a record whose evidence came
    from one must not admit the other. It is also the field that keeps the two providers of
    one encoding apart when everything else about their declaration matches.
    """
    return canonical.digest(
        {
            "name": provider.name,
            "route": provider.route,
            "device_predicate": provider.device_predicate,
            "suite": provider.suite,
            "required": sorted(provider.required),
            "optional": sorted(provider.optional),
        }
    )
