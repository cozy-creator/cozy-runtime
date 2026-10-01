"""cr-006 seam 1 — FORMATS: the stored-byte contracts and the providers that DECODE them.

This is the first of `encoding`'s three seams (#549.10). It owns the refusal vocabulary, the
per-tensor role parts a header declares, and every provider whose answer is a FLOAT TENSOR.
Seam two (`leaves.py`) owns the providers whose answer is a replaced module; seam three
(`selection.py`) owns capability records, admission and the launch table. Nothing here
imports either of the others, which is what makes the split a seam rather than a filename.

Three rules hold the shape, and each is a refusal rather than a convention:

**Selection is by DIGEST, never by name.** The header assigns every tensor an
`EncodingSpec` object digest. That digest — not the alias, not the key, not the dtype —
picks the provider. Two physically different role sets under one alias are two digests and
two providers, which is exactly what the H3 production artifact carries (150 modules with
an `input_scale`, 50 `mlp.fc2` without).

**A provider claims DIGESTS it was reviewed against, and an alias claims nothing** (#549.4).
`reviewed_specs` is a frozenset of exact spec digests, spelled here, and `launch_providers`
offers a provider for a digest only if that digest is in the set AND still in the registry.
The old spelling read digests OUT of `tensorfs.seed_digests()` by ALIAS, which meant a new,
semantically different spec filed under `mxfp8/1` would have INHERITED this file's decoder
the moment the registry grew it. Transcribing the digests is the cost of making that
impossible: a registry change that moves a digest now breaks selection loudly at launch
(`spec_unreviewed`) instead of decoding unreviewed bytes. Alias is display, and only display.

**The decode target is the CENSUS's dtype, and the census cannot express an encoding.**
A derived `LogicalTensor` names a destination and nothing else — no parts, no scale_of, no
dequantize (derive.py's `_ROW_FIELDS` equality makes that an import-time proof). So the
decode target is the destination's own dtype, the header's logical dtype must equal it, and
a disagreement is cr-005's `dtype_mismatch` unchanged.

**Capability is fail-closed and device-keyed.** A provider serves an (encoding, device) pair
only when a CapabilityRecord exists for it, and a record cannot be written without the
numerics evidence that minted it (`probe.py`). An unrecognised device is the ABSENCE of
evidence, never permission — v1's `DTYPE_MIN_SM` fell OPEN on an unknown card and served
plausible wrong output.

**Geometry is preflighted, PURELY, before anything is allocated** (#549.5). Every provider
answers `supports(shape)` with "" or with the sentence saying why not. TensorFS admits a
ragged MXFP8 tail, this module's decode handles one, and the native MX leaf cannot — the
tensor cores read whole 32-blocks. Before #549.5 nothing asked until the leaf was being
built with the component resident. Now plan resolution asks first, and a tensor whose
geometry the leaf refuses resolves to the decode floor with the reason recorded IN the plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from cozy_runtime.author._loader import TORCH_DTYPES as _TORCH_DTYPES

# `Any` below is a torch module, tensor, dtype or nn.Module: torch is absent from the check venv.

#: This module's own refusal codes, added to cr-005's matrix. Closed, like that one.
REFUSALS = frozenset(
    {
        "unknown_encoding",  # the header cites a spec digest no provider claims
        "encoding_unqualified",  # a claimed encoding with no capability record for THIS device
        "role_mismatch",  # the delivered parts are not the roles the provider decodes
        # A spec digest the registry carries and NO provider in this build declares in its
        # `reviewed_specs`. #549.4's arm: a new digest filed under an existing alias acquires
        # no provider, where the old alias-keyed launch table handed it the sibling's decoder.
        "spec_unreviewed",
        # Every candidate for this tensor refuses its GEOMETRY (#549.5). Distinct from
        # `encoding_unqualified` because the remedy is different: the device is fine and the
        # tensor's own shape is what no implementation here can consume.
        "geometry_unsupported",
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


def aliases() -> dict[str, str]:
    """spec digest -> platform alias, read from the COMPILED registry.

    DISPLAY ONLY since #549.4. It used to be half of selection — a provider was offered for
    every digest an alias carried — and that is precisely the defect: an alias is a human
    label the registry may file two, four or five physically different specs under, and
    `nvfp4-w4a4/1` already files four. Now nothing resolves THROUGH this map; it names what
    a refusal is talking about.
    """
    import tensorfs

    return {digest: alias for alias, digest in tensorfs.seed_digests()}


def registry_digests() -> frozenset[str]:
    """Every spec digest the compiled registry carries, in any alias."""
    import tensorfs

    return frozenset(digest for _, digest in tensorfs.seed_digests())


def digests_of(alias: str) -> tuple[str, ...]:
    """Every spec digest the platform registry files under one alias, in registry order.

    Kept for DISPLAY and for the arms that need to talk about an alias's digest set. It is
    no longer how a provider acquires a digest — see `reviewed_specs`.
    """
    import tensorfs

    return tuple(digest for name, digest in tensorfs.seed_digests() if name == alias)


# --------------------------------------------------------- the reviewed spec digests
#
# EXACT spec object digests, transcribed from `tensorfs.seed_digests()` at review time and
# pinned here on purpose (#549.4). Each one names the physical role contract a provider in
# this file was read against, and each has a PRODUCER-MINED VECTOR DOCUMENT vendored beside it in
# `encoding/vectors/` that the qualification suite RUNS (probe.py's `_spec_vectors`) — so
# "reviewed" means the exact vector object this spec digest names was decoded bit-for-bit,
# not that somebody read the prose.
#
# A digest that leaves the registry, or a registry that files a NEW digest under one of
# these aliases, changes this file's answer LOUDLY: the first case refuses `spec_unreviewed`
# at launch, the second acquires no provider at all. Both are the intended behaviour; the
# alias-keyed spelling had neither.

#: `plain/1` — one `value` role carrying the logical bytes verbatim.
SPEC_PLAIN = "sha256:1fb882a7e46d0aff520f9d8a28cefd643954c19371737443101ba3c5fcc3613f"

#: `fp8-rowwise/1`, rank-1 `[out]` scale. v1's `fp8-w8a8` writer.
SPEC_ROWWISE = "sha256:c4be0120fb4548306b134f6ee07eb2545a363bc140a005af1ef6543c790cf890"

#: `fp8-rowwise/1`, rank-2 `[out, 1]` scale — a KEPT trailing unit axis, physically distinct.
SPEC_ROWWISE_KEEPDIM = "sha256:6773dfe384aeb380d8d32de9f0b384a59a3294a96da4f3f1c8b956e194e3ff2e"

#: `fp8-scaled-scalar/1` WITH the rank-0 activation `input_scale` (H3's 150 modules).
#:
#: TensorFS 0.3 removed prose from EncodingSpec identity. Roles and the exact vendored vector
#: bytes remain unchanged; only the reviewed spec identity moves.
SPEC_SCALED_SCALAR = "sha256:762e1d49dd11df1d9e99893213f232cf72797bf879aa6f07bc138ec86c04bac3"

#: `fp8-scaled-scalar/1` weight-scale only (H3's 50 `mlp.fc2`). One artifact, two role sets.
#: Roles (`data`, `weight_scale`) and vendored vector bytes remain unchanged under 0.3.
SPEC_SCALED_SCALAR_WEIGHT_ONLY = (
    "sha256:7bede606ac5329020c2653eb04663b0d94d6c778e60234479509eddcce38240c"
)

#: `mxfp8/1` — E4M3 elements, one E8M0 byte per 32-element block along the last axis.
SPEC_MXFP8 = "sha256:7e9b1ad8f2e5ddd236a4d4303042d632a96eadef4f25d44eb0fb63124cec7cfd"

#: Reviewed digest -> (vendored fixture filename, exact vector-byte digest) from that nested spec.
#: Qualification checks both before executing a case. `plain/1`'s route decodes nothing, so it
#: has no entry — its evidence is the fill plane's byte-identity proof.
SPEC_VECTORS: dict[str, tuple[str, str]] = {
    SPEC_ROWWISE: (
        "fp8-rowwise.json",
        "sha256:e7a1264e4b60ac9f800736b23b3e3c50bb3e6ed506bdc30a2769abfc850ef2ba",
    ),
    SPEC_ROWWISE_KEEPDIM: (
        "fp8-rowwise-keepdim.json",
        "sha256:55e4b815d169c5b6959decdb54a89a7af94c26f9f4ac7339c25d6b365582dd27",
    ),
    SPEC_SCALED_SCALAR: (
        "fp8-scaled-scalar.json",
        "sha256:ed0ee586675b05cfff658e61a8f46712e679be3cfac350b878c5e0b69e5732d4",
    ),
    SPEC_SCALED_SCALAR_WEIGHT_ONLY: (
        "fp8-scaled-scalar-weight-only.json",
        "sha256:2326cbd4a1ccaac484792bba04f07e19bc59d61fd913cbe5a4a9a66e469a9374",
    ),
    SPEC_MXFP8: (
        "mxfp8.json",
        "sha256:9950f32248a14b37267c17d8d8e1fd9c8a281c8291183ee390908e2f00326a22",
    ),
}


#: TensorFS dtype name -> torch dtype name. Owned by `author/_loader.py` since cr-087 —
#: the census is where a torch dtype first becomes a runtime fact and the author surface
#: may import nothing deeper — and re-exported here for the planes #549.10 named: the fill
#: plane, `probe.py`'s role tensors, and the vendored vector documents, all of which read
#: TensorFS dtype names.
TORCH_DTYPES = _TORCH_DTYPES


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


# ------------------------------------------------------------------------ the providers


class Provider(Protocol):
    """One implementation that turns one encoding's stored roles into one destination.

    DECODING providers present a PLAIN compute-dtype tensor at the leaf boundary and need
    no adapter awareness (cr-006's scope note): the SDK side-branch wraps the leaf, cr-010.
    An `encoded_gemm` provider does not decode at all — it REPLACES the leaf, and its
    surface is `LeafProvider` in seam two.
    """

    @property
    def encoding(self) -> str:
        """The spec digest this instance serves: one instance per reviewed digest."""
        ...

    @property
    def name(self) -> str: ...

    @property
    def route(self) -> str:
        """`verbatim`, `decoded_float` or `encoded_gemm` — the delivery route (§3.2).

        RENAMED from `native_encoded` (#549.11): "native" named a device generation and the
        thing that is actually different about this route is that the GEMM consumes the
        stored elements. `encoded_gemm` says that and nothing else.
        """
        ...

    @property
    def device_predicate(self) -> str: ...

    @property
    def reviewed_specs(self) -> frozenset[str]:
        """The EXACT spec digests this implementation was reviewed against (#549.4)."""
        ...

    @property
    def code_revision(self) -> str:
        """This implementation's own ABI/code revision, bumped when its arithmetic, its
        operand layout or its kernel call changes (#549.9). It joins `implementation_digest`
        and therefore the capability-record key, so an edit to a forward pass invalidates the
        records that measured the OLD one instead of inheriting them."""
        ...

    @property
    def suite(self) -> str:
        """WHICH probe oracle qualifies this implementation.

        Keyed on the DECLARATION and never on the class, so a second implementation of one
        encoding is checked by the same oracle as the first — which is the only way two
        implementations' records mean the same thing.
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

    def supports(self, shape: Sequence[int]) -> str:
        """PURE geometry preflight (#549.5): "" if servable, else the reason it is not.

        No torch, no device, no allocation — it is asked at plan resolution, over a header's
        logical shape, before a destination exists. A provider that has no geometry
        constraint answers "" and says so in its own body rather than leaving the method off.
        """
        ...

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None: ...


@runtime_checkable
class LeafProvider(Provider, Protocol):
    """A MODULE-REPLACING provider: the stored roles stay stored and the leaf changes.

    `Provider.decode` writes a float destination the contract's dtype named, which is the
    whole reason the `encoded_gemm` rung stayed unbuilt through two lanes: on that route
    there IS no float destination — the payload and its scales are the resident weight, and
    what has to change is the module that multiplies by them.

    `leaf()` returns a real `torch.nn.Module`, and it must be one: `nn.Module.__setattr__`
    refuses to accept anything else as a child, which is what makes leaf replacement
    machinery rather than a one-line swap.

    `resident_bits_per_element` is the FOOTPRINT axis of #515a's split, computed from the
    role bytes that stay resident rather than declared: on the encoded route those bytes ARE
    the weight, which is the residency a decoded-float route structurally cannot offer.
    """

    @property
    def output_dtype_rule(self) -> str:
        """How the leaf's forward picks its output dtype. `contract` is the only rule this
        build has: the leaf emits the destination dtype the contract named, so a downstream
        module sees exactly what the float path would have handed it."""
        ...

    def leaf(self, torch: Any, parts: Mapping[str, Any], replaced: Any, out_dtype: Any) -> Any: ...

    def resident_bytes(self, parts: Mapping[str, RolePart]) -> int: ...

    def resident_bits_per_element(self, parts: Mapping[str, RolePart], numel: int) -> float: ...

    def fill_scratch_bytes(self, parts: Mapping[str, RolePart]) -> int: ...


def _rank2(shape: Sequence[int]) -> str:
    """The one geometry law every fp8/mx GEMM route shares: two outer dimensions."""
    if len(shape) != 2:
        return (
            f"a scaled GEMM has two outer dimensions and this tensor is rank {len(shape)} "
            f"{tuple(shape)}; a reshape here would be a different model"
        )
    return ""


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
    def reviewed_specs(self) -> frozenset[str]:
        return frozenset({SPEC_PLAIN})

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

    def supports(self, shape: Sequence[int]) -> str:
        # Any rank, any extent: a verbatim copy has no arithmetic to have a geometry.
        return ""

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        raise AssertionError("a verbatim role is streamed straight into its destination")


@dataclass(frozen=True, slots=True)
class ScaledScalarDequant:
    """`fp8-scaled-scalar/1` -> float, weight-only: `value = data * weight_scale`.

    That equation is the spec's own, transcribed from the registry entry and from nothing
    else. `input_scale`, where the artifact carries it, is the ACTIVATION scale of a
    native fp8 matmul — it is deliberately NOT applied to the weight, because applying it
    would be a different (and wrong) tensor. On this route activations stay in the
    destination's float dtype, which is strictly more precise than the encoded path the
    scale was minted for; the two spec variants therefore decode identically and are still
    two REVIEWED DIGESTS, because a delivered role set that does not match refuses rather
    than being tolerated.

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
        # the ENCODED-GEMM route (`torch._scaled_mm`), which is a different provider and is
        # not built here — see NAMED_SEAMS.
        return "any"

    @property
    def reviewed_specs(self) -> frozenset[str]:
        # BOTH role sets one real fp8_scaled artifact carries, reviewed as two digests. The
        # value law is identical and the physical contracts are not, which is exactly why
        # they are two entries here rather than one alias lookup.
        return frozenset({SPEC_SCALED_SCALAR, SPEC_SCALED_SCALAR_WEIGHT_ONLY})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "fp8-scalar-decode"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"data", "weight_scale"})

    @property
    def optional(self) -> frozenset[str]:
        # `input_scale` is the ACTIVATION scale of the native fp8 matmul. The two reviewed
        # digests differ by exactly its presence, and WHICH digest a tensor cites is checked
        # by TensorFS against the header's own closure (its fact, its owner).
        return frozenset({"input_scale"})

    def supports(self, shape: Sequence[int]) -> str:
        # A per-tensor scalar scale broadcasts over any shape; the spec pins rank 2 and the
        # header is checked against the spec by TensorFS, so this route adds nothing.
        return ""

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
#: (`ceil_div(axis 1, by 32)`) and from nothing else.
MX_BLOCK = 32

#: E8M0's bias: a scale byte `b` denotes exactly `2**(b - 127)`.
E8M0_BIAS = 127

#: E8M0's ONE non-numeric code point. The registry entry's prose says only "scale byte b
#: means 2^(b-127)", which at `b = 255` reads as `2**128`; the encoding's own NORMATIVE
#: decoder — the conform corpus this spec is qualified against — returns NaN there, which
#: is OCP's reading. Prose and reference disagree and the REFERENCE wins (#511c).
E8M0_NAN = 255

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

#: The ceiling on the f32 transient a rounding-correct rowwise decode is allowed to hold at
#: once. It is NOT a tuning knob and NOT arbitrary: it is the weight plane's region grain
#: (`weights.REGION_LIMIT`, 16 MiB), so this route's largest unledgered allocation is the
#: same order as one streamed region. A decode that held a full-width f32 copy of a 4 GiB
#: weight would be a second residency the ledger never granted.
_CHUNK_BYTES = 16 << 20


@dataclass(frozen=True, slots=True)
class MicroScaledDequant:
    """`mxfp8/1` -> float, weight-only: `value = data * 2**(scale - 127)` per 32-block.

    The registry's spec is the whole contract and is transcribed from it: E4M3 elements in
    a `data` role shaped exactly like the logical tensor, and one E8M0 exponent byte per
    32-element block along the LAST axis in a `scale` role shaped `[rows, ceil(cols/32)]`.
    Rank is 2 because the spec says `logical_rank: 2`.

    **This route is THE RAGGED-TAIL FLOOR** (#549.5). `ceil_div` admits a last block shorter
    than 32, TensorFS accepts one, and the encoded-GEMM leaf cannot consume one — so the
    slow path below is not a curiosity, it is the qualified route a ragged tensor resolves
    to, by plan, with the reason recorded. See `MicroScaledNativeLeaf.supports`.

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
    served silently wrong.
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
        # wherever torch carries the e4m3 dtype. The sm120 floor belongs to the ENCODED-GEMM
        # LEAF, which is a different provider under this same digest.
        return "any"

    @property
    def reviewed_specs(self) -> frozenset[str]:
        return frozenset({SPEC_MXFP8})

    @property
    def code_revision(self) -> str:
        return "1"

    @property
    def suite(self) -> str:
        return "mxfp8-decode"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"data", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    def supports(self, shape: Sequence[int]) -> str:
        # Rank 2 is the spec's own `logical_rank`. RAGGED IS FINE HERE and that is the whole
        # point of stating it: this route is what a ragged tensor falls to.
        return _rank2(shape)

    @staticmethod
    def _multiplier(torch: Any, scale: Any, dtype: Any) -> Any:
        """One E8M0 byte grid -> the block multipliers, BY BIT CONSTRUCTION.

        An E8M0 byte IS an IEEE binary32 exponent field: `b` in 1..254 denotes exactly the
        f32 whose bit pattern is `b << 23`, mantissa zero. So the multiplier is built, not
        computed — no `exp2`, no `pow`, nothing whose accuracy is a property of the backend's
        math library. Measured on an L40S, `torch.exp2` over the same domain is correct in
        bf16 and off by up to 2.3 ULP in f32 for the smallest scale bytes, which failed 46 of
        65,536 pairs against a two-rounding budget. Loosening the budget to admit it would
        have been the wrong repair — the decode's error should be ZERO.

        Two code points are not exponents and are placed by hand: `b = 0` denotes `2**-127`,
        one binade below f32's smallest NORMAL, so it is the subnormal `1 << 22`; and
        `b = 255` is E8M0's NaN, which `b << 23` would spell as a signed infinity.
        """
        bits = scale.to(torch.int32) << 23
        bits = torch.where(scale == 0, torch.full_like(bits, 1 << 22), bits)
        # f32 first, in one code path, because its exponent range covers the whole byte
        # domain exactly; narrowing to the destination is then a single exact conversion
        # wherever the destination can hold the value, and 0 or inf where it cannot.
        mult = bits.view(torch.float32).to(dtype)
        return torch.where(scale == E8M0_NAN, torch.full_like(mult, float("nan")), mult)

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        data = parts["data"]
        scale = parts["scale"]
        dtype = destination.dtype
        rows, cols = int(destination.shape[0]), int(destination.shape[1])
        blocks = int(scale.shape[-1])
        if dtype == torch.float16:
            # f16's exponent range cannot hold every block scale (2**-31 narrows to 0), so
            # the product is formed in f32, where it is exact, and rounded once.
            wide = data.view(destination.shape).to(torch.float32)
            mult = self._multiplier(torch, scale, torch.float32)
            destination.copy_(wide * mult.repeat_interleave(MX_BLOCK, dim=-1)[..., :cols])
            return
        mult = self._multiplier(torch, scale, dtype)
        if cols == blocks * MX_BLOCK and destination.is_contiguous():
            # THE PATH EVERY WHOLE-BLOCK ARTIFACT TAKES. A 3-D view over both sides
            # broadcasts the block scale with no intermediate at all: no repeat, no f32
            # copy, one kernel.
            view = destination.view(rows, blocks, MX_BLOCK)
            view.copy_(data.view(rows, blocks, MX_BLOCK).to(dtype))
            view.mul_(mult.unsqueeze(-1))
            return
        # The RAGGED tail and the non-contiguous destination, both handled by widening the
        # multiplier instead of viewing the weight. It costs one destination-sized temporary
        # and is written as the slow path on purpose: a 3-D view cannot span a partial block
        # or a strided slice, and quietly taking the fast path anyway would mis-stride the
        # scales.
        destination.copy_(data.view(destination.shape).to(dtype))
        destination.mul_(mult.repeat_interleave(MX_BLOCK, dim=-1)[..., :cols])


@dataclass(frozen=True, slots=True)
class RowwiseDequant:
    """`fp8-rowwise/1` -> float, weight-only: `value[i,j] = data[i,j] * scale[i]`.

    The spec's own equation, transcribed from the registry entry. It serves BOTH physical
    variants — rank-1 `[out]` and rank-2 `[out, 1]` — because the value law is identical and
    the only difference is a trailing unit axis, which `reshape(rows, 1)` resolves for both.
    They are two REVIEWED DIGESTS, not one tolerated shape.

    This is the DEQUANT FLOOR of #518's Hopper/Ada artifact — the route that serves the
    fp8-rowwise bytes on a device where nothing measured the encoded-GEMM leaf. It exists so
    that "the chooser picks the leaf where it was measured and the floor where it was not"
    is a statement about two real providers rather than about a provider and an absence.

    **THE ROUNDING RULE IS THE PRODUCER'S, AND IT IS NOT THIS ALIAS'S NEIGHBOUR'S** (found by
    #549.4's vector leg, revision 2). The product is formed in f32 and rounded ONCE into the
    destination; the scale is NOT narrowed first. That is the opposite of
    `fp8-scaled-scalar/1`, whose registry entry pins narrow-then-multiply explicitly, and the
    difference is a full ULP: measured against `fp8-rowwise/1`'s own producer-mined vectors,
    narrowing first differs from the reference on 86 of 1,024 bf16 bytes and rounding once
    matches it exactly, on both the rank-1 and the rank-2 digest.

    Nothing could see that before. `probe.py::_numerics_rowwise` sweeps this route against
    scales that are exact POWERS OF TWO — its own docstring says so — and under a power of
    two the two rules agree bit for bit, so a domain sweep of 65,536 pairs proved the scale's
    ROUTING and could not reach its ROUNDING. The producer's mined vectors carry continuous
    f32 scales, which is exactly the case the synthetic oracle excluded, and #511c's rule
    decides it: the reference wins over prose, and here the reference is the only thing that
    had an opinion at all.

    `code_revision` is `2` because of this. The records that measured revision 1 measured
    different arithmetic, and #549.9's revision field is what makes them stop admitting it
    rather than silently carrying a verdict about a decoder that no longer exists.
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
    def reviewed_specs(self) -> frozenset[str]:
        return frozenset({SPEC_ROWWISE, SPEC_ROWWISE_KEEPDIM})

    @property
    def code_revision(self) -> str:
        # 2: the rounding rule corrected to the producer's (see the class docstring).
        return "2"

    @property
    def suite(self) -> str:
        return "fp8-rowwise-decode"

    @property
    def required(self) -> frozenset[str]:
        return frozenset({"data", "scale"})

    @property
    def optional(self) -> frozenset[str]:
        return frozenset()

    def supports(self, shape: Sequence[int]) -> str:
        # Rank 2 is the spec's own. A row scale has no block structure, so no K constraint.
        return _rank2(shape)

    def decode(self, torch: Any, parts: Mapping[str, Any], destination: Any) -> None:
        data = parts["data"]
        scale = parts["scale"]
        rows = int(destination.shape[0])
        payload = data.view(destination.shape)
        rowscale = scale.to(torch.float32).reshape(rows, 1)
        if destination.dtype == torch.float32:
            # Already the working dtype: one rounding by construction, no temporary.
            destination.copy_(payload.to(torch.float32))
            destination.mul_(rowscale)
            return
        # ONE ROUNDING, IN ROW CHUNKS. The product has to be formed in f32 (see the class
        # docstring), and forming it whole would allocate a full-width f32 copy of every
        # weight — the second full copy `MicroScaledDequant` refuses to make, and one the
        # ledger never granted because it is an allocator transient rather than decode
        # scratch. Chunking by ROWS bounds that transient without changing a single
        # arithmetic step: each row's scale applies to its own row and nothing else, so a
        # row-block boundary is not a boundary the arithmetic can see.
        stride = max(1, _CHUNK_BYTES // max(int(destination.shape[1]) * 4, 1))
        for start in range(0, rows, stride):
            stop = min(start + stride, rows)
            block = payload[start:stop].to(torch.float32)
            block.mul_(rowscale[start:stop])
            destination[start:stop].copy_(block)
