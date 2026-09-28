"""cr-006 — the per-device qualification suite that MINTS capability records.

§3.2's cost placement, made concrete: this runs ONCE per (accelerator model, device
configuration, driver, runtime build, encoding rule) tuple, at prepare time, and its
result is a persisted typed fact. A worker landing on a known tuple skips the suite
entirely; a local rig probes once and persists. Per request there is only arithmetic.

Four things happen per (encoding, IMPLEMENTATION, output dtype), in this order, and a
failure at any one leaves NO record — which is what "fail-closed" means when the table is
minted rather than declared. The middle term is new: since #517e an encoding carries
ORDERED CANDIDATES rather than one provider, and each candidate is qualified separately,
because "mxfp8 works here" is not a fact — "this implementation of mxfp8 works here" is.

1. **numerics self-check** against an INDEPENDENT reference. The e4m3 and e8m0 decoders
   below are written from the OCP number formats by integer arithmetic, not by asking
   torch to cast — a self-check that uses the implementation as its own oracle checks
   nothing. The domain is the WHOLE format: all 256 e4m3 bit patterns for a per-tensor
   scale, and all 65,536 (e4m3 element, e8m0 block scale) pairs for a microscaled one,
   NaN included (as a NaN, not a number) and overflow included (as an infinity, not a
   plausible finite number). The suite a provider gets is keyed on the ORACLE NAME it
   declares (`Provider.suite`), so an implementation with no oracle qualifies nothing
   rather than passing by default.
   A NATIVE LEAF's oracle is a different shape and has to be: its stored roles never become
   a float tensor, so there is nothing per-element to check. `_native_agreement` runs the
   leaf's real kernel on the real arch against an f64 matmul over the exactly-dequantized
   weights — which is the only check that can see #515d, where torch validates an MX scale
   grid's element COUNT and not its LAYOUT and returns wrong numbers without raising.
2. **micro-benchmark** at a realistic tensor size, so the plan table has a measured
   decode rate rather than an assumed one — or, for a leaf, a measured forward beside the
   float GEMM it replaces.
3. **route axes** — the two MEASURED numbers `Capabilities.admit` RANKS candidates on
   (#517e): resident bits per element, and the whole route's deviation from an f64
   reference on identical inputs. Both routes are scored in ONE frame, because #517d's
   finding is that fidelity and footprint order OPPOSITELY and a chooser therefore needs
   an objective with numbers on both sides rather than a ladder position.
4. **record** — the tfs-013 key with the evidence digest of exactly this observation.

The suite QUALIFIES a fixed kernel set; it does not search one. Its seconds are returned as
an observation.

**The result is KEPT, and this module's first paragraph is now true (cr-103).** It said the
suite runs once per tuple and persists; the last paragraph said "a new Runtime process
probes again", and the code did the second thing — 3.2-6.1 s of every model-bearing
prepare, re-measuring facts about a card and a build that had not moved. `qualified` stores
one document per `qualification_key` and reuses it when every term of that key matches, so
the second model on a warm pod pays none of it. It is a CACHE OF A MEASUREMENT, never a
substitute for one: the key carries the whole device identity (`DeviceFacts` — "every field
is stable for the life of the machine, which is what lets a probe result be keyed by it"),
the runtime identity (whose `source_digest` covers this very file, so an edit to a decode
or to the suite invalidates every document that measured the old one), the exact provider
implementation digests, and the output dtypes the suite was run over. Anything a stored
document could be wrong about is a term of the key, and a stored document that does not
match its own key is discarded.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec

from cozy_runtime.author._numeric import within, worst
from cozy_runtime.internal import accel, canonical
from cozy_runtime.internal.encoding import (
    MX_BLOCK,
    SPEC_VECTORS,
    TORCH_DTYPES,
    Capabilities,
    CapabilityRecord,
    DeviceFacts,
    LeafProvider,
    Provider,
    RolePart,
    RuntimeIdentity,
    geometry_class,
    implementation_digest,
    measure_runtime,
)
from cozy_runtime.internal.instruments import quantize_rowwise

# `torch`, `device`, `dtype` and tensor values are `Any`: the check venv carries no torch.

#: One decode of this many elements is the micro-benchmark's unit — a realistic attention
#: projection, not a toy and not a search space.
BENCH_SHAPE = (2048, 2048)
BENCH_REPS = 8

#: The token count the ROUTE AXES measure one GEMM at, against `BENCH_SHAPE` weights. Small
#: on purpose: the axes are a fidelity and a footprint number, not a throughput one, and a
#: qualification suite that spent a second per (provider, dtype) on a matmul nobody reads
#: would be paying §3.2's prepare-time budget for nothing.
FIDELITY_TOKENS = 256

#: Where a native route's GEMM stops being lossy and starts being wrong. #515a measured the
#: real thing at 0.0421 relative (27.5 dB) and #517d at 0.146 (16.7 dB) on a harder subject;
#: a mis-strided scale or an unswizzled grid measures past 0.5. The line is drawn in the gap
#: and it is a SANITY bound, not a quality one — quality is the ranked axis, not the gate.
NATIVE_DEVIATION_BOUND = 0.30

# ----------------------------------------------------------------- the result records


class RouteAxes(msgspec.Struct, frozen=True):
    """The two MEASURED axes `Capabilities.admit` ranks on (#517e). -1.0 is NOT MEASURED,
    which makes a record UNRANKABLE rather than plausible: a zero would win every comparison
    it was never entitled to enter."""

    deviation_rel: float = -1.0
    resident_bits_per_element: float = -1.0


class Sweep(msgspec.Struct, frozen=True, omit_defaults=True):
    """A decode oracle's verdict: `patterns` for the one-scale sweep; `pairs`, per-band counts
    and the first wrong pair for a whole-domain one."""

    max_abs_error: float
    max_rel_error: float
    tolerance: float
    nan_preserved: bool
    passed: bool
    patterns: int = 0
    pairs: int = 0
    subnormal_tolerance: float = 0.0
    domain: dict[str, int] | None = None
    wrong: dict[str, int] | None = None
    first_wrong: str | None = None


class NativeAgreement(
    msgspec.Struct, frozen=True, omit_defaults=True, tag_field="kind", tag="native"
):
    """A native leaf's GEMM against the f64 reference, or its kernel's refusal verbatim."""

    passed: bool
    shape: list[int] = []
    deviation_rel: float = -1.0
    tolerance: float | None = None
    finite: bool | None = None
    out_dtype: str = ""
    out_dtype_is_contract: bool | None = None
    kernel: str = ""
    note: str = ""


class DecodeBench(msgspec.Struct, frozen=True):
    shape: list[int]
    reps: int
    decode_ns: int
    gb_per_s: float


class LeafBench(msgspec.Struct, frozen=True):
    shape: list[int]
    reps: int
    leaf_ns: int
    float_gemm_ns: int
    speedup_x: float


class VectorRef(msgspec.Struct, frozen=True):
    filename: str
    sha256: str


class VectorCheck(msgspec.Struct, frozen=True, omit_defaults=True):
    """One mined case: its bytes compared, skipped for geometry, or refused for its roles."""

    name: str
    passed: bool
    skipped: bool = False
    why: str = ""
    elements: int | None = None
    differing_bytes: int | None = None


class SpecVectors(
    msgspec.Struct, frozen=True, omit_defaults=True, tag_field="kind", tag="spec-vectors"
):
    """The reviewed spec's own vectors (#549.4). SKIPPED neither passes nor fails a record."""

    passed: bool
    skipped: bool = False
    note: str = ""
    document: VectorRef | None = None
    cases: list[VectorCheck] = []


class VerbatimEvidence(msgspec.Struct, frozen=True, tag_field="kind", tag="verbatim"):
    note: str
    passed: bool = True
    route_axes: RouteAxes = RouteAxes()


class UnmeasuredEvidence(msgspec.Struct, frozen=True, tag_field="kind", tag="unmeasured"):
    note: str
    passed: bool = False
    route_axes: RouteAxes = RouteAxes()


class NativeEvidence(
    msgspec.Struct, frozen=True, omit_defaults=True, tag_field="kind", tag="native"
):
    numerics: NativeAgreement
    passed: bool
    route_axes: RouteAxes
    micro_bench: LeafBench | None = None


class DecodeEvidence(msgspec.Struct, frozen=True, tag_field="kind", tag="decode"):
    numerics: Sweep
    spec_vectors: SpecVectors
    micro_bench: DecodeBench
    passed: bool
    route_axes: RouteAxes = RouteAxes()


Evidence = VerbatimEvidence | UnmeasuredEvidence | NativeEvidence | DecodeEvidence


class Observation(msgspec.Struct, frozen=True):
    encoding_spec_id: str
    implementation: str
    implementation_digest: str
    implementation_revision: str
    device_predicate: str
    output_dtype: str
    geometry_class: str
    """WHAT SHAPE THIS WAS MEASURED AT (#549.9). Evidence, not a filter: `Provider.supports`
    gates a tensor's geometry, and filtering admission on this string too would refuse the
    decode floor for the ragged tensor it is the only route for."""
    delivery_route: str
    evidence: Evidence


class QualificationResult(msgspec.Struct, frozen=True):
    """One suite run: what it ran on and everything it observed. What `qualified` keeps."""

    tuple_key: str = msgspec.field(name="tuple")
    device: DeviceFacts
    device_predicate: str
    runtime_release: str
    runtime: RuntimeIdentity
    suite_ms: int
    observations: list[Observation]


def e4m3_value(bits: int) -> float:
    """One E4M3FN byte -> its exact value, by integer arithmetic over the OCP format.

    E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa; no infinities; the single NaN is
    S.1111.111. Written here so the self-check has an oracle the implementation did not
    supply.
    """
    sign = -1.0 if bits & 0x80 else 1.0
    exponent = (bits >> 3) & 0x0F
    mantissa = bits & 0x07
    if exponent == 0x0F and mantissa == 0x07:
        return math.nan
    if exponent == 0:
        return sign * (mantissa / 8.0) * 2.0**-6
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def e8m0_value(bits: int) -> float:
    """One E8M0 scale byte -> its exact value, by integer arithmetic over the OCP format.

    E8M0 is a bare biased exponent: no sign, no mantissa, `b` denotes `2**(b - 127)`, and
    `0xFF` is the format's single NaN. `math.ldexp` forms the power exactly in a Python
    double for the whole 0..254 range (`2**-127` and `2**127` both land exactly), so this
    oracle never asks torch what a power of two is.
    """
    if bits == 0xFF:
        return math.nan
    return math.ldexp(1.0, bits - 127)


def _eps(torch: Any, dtype: Any) -> float:
    return float(torch.finfo(dtype).eps)


def _numerics_mx(torch: Any, provider: Provider, device: Any, dtype: Any) -> Sweep:
    """Decode the WHOLE (e4m3 element x e8m0 block scale) cross product — 65,536 pairs.

    The fp8 suite below checks 256 element patterns at ONE scale, which is the whole domain
    when the scale is a single f32 the artifact carries. MXFP8's scale is itself a format
    with 256 code points, so the honest domain is the product of the two, and the layout
    that covers it exactly is 256 rows of 256 columns: row `r` carries scale byte `r` in
    all 8 of its 32-element blocks, and every row carries all 256 element patterns.

    Four verdicts per pair, because "correct" is not one question when a product spans
    2**-136 to 2**136 and the destination dtype holds a window of that:

    * NORMAL — the exact product is a finite normal of the dtype. Demand it within two
      roundings RELATIVE, the same budget the fp8 route is held to.
    * SUBNORMAL — finite, below the dtype's smallest normal. Relative error is the wrong
      metric here and would fail a correct decoder: in the subnormal band the ULP is
      CONSTANT at `smallest_subnormal`, so the budget is two roundings ABSOLUTE. Derived
      from the format, not chosen to make anything pass.
    * OVERFLOW / UNDERFLOW — the exact product leaves the dtype's finite range, so the
      arithmetic's own answers are a signed infinity and a zero respectively. A plausible
      finite number in either place would be a wrong number.
    * NAN — either operand is its format's NaN, and NaN must survive as NaN.

    `passed` requires all of them, so a dtype whose block multiplier underflows before the
    product does — f16 flushes `2**(b-127)` to zero below `b = 103`, zeroing blocks whose
    products it could have held — fails on the NORMAL pairs it silently zeroes, mints no
    record, and refuses at bind instead of serving zeros.
    """
    rows = cols = 256
    blocks = cols // 32
    data = (
        torch.frombuffer(bytearray(bytes(range(256)) * rows), dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .reshape(rows, cols)
        .to(device)
    )
    scale = torch.arange(rows, dtype=torch.uint8).reshape(rows, 1).repeat(1, blocks).to(device)
    destination = torch.empty(rows, cols, dtype=dtype, device=device)
    provider.decode(torch, {"data": data, "scale": scale}, destination)
    got = destination.double().cpu().reshape(-1).tolist()
    return _domain_sweep(
        torch, dtype, got, [(e8m0_value(r), f"scale {r:#04x}") for r in range(rows)]
    )


def _numerics(torch: Any, provider: Provider, device: Any, dtype: Any) -> Sweep:
    """Decode all 256 e4m3 patterns at a known scale and compare to the oracle.

    The tolerance is DERIVED, not chosen: the route casts the element to the destination
    dtype and multiplies by the scale in that dtype, so two roundings are the whole error
    budget. A decoder that adds anything of its own fails here.
    """
    scale_value = 0.013671875  # exactly representable in f16/bf16/f32: 7 * 2^-9
    patterns = bytes(range(256))
    data = torch.frombuffer(bytearray(patterns), dtype=torch.uint8).view(torch.float8_e4m3fn)
    data = data.reshape(16, 16).to(device)
    weight_scale = torch.tensor(scale_value, dtype=torch.float32, device=device)
    destination = torch.empty(16, 16, dtype=dtype, device=device)
    provider.decode(torch, {"data": data, "weight_scale": weight_scale}, destination)
    got = destination.reshape(-1).double().cpu().tolist()
    return _grade_scaled(_eps(torch, dtype), got, scale_value)


def _grade_scaled(eps: float, got: Sequence[float], scale_value: float) -> Sweep:
    """Grade the 256 decoded patterns: NaN stays NaN, zero stays exactly zero, and every
    other value is within two roundings relative. A non-finite error never passes."""
    worst_rel, worst_abs, nan_ok = 0.0, 0.0, True
    for byte, have in enumerate(got):
        want = e4m3_value(byte) * scale_value
        if math.isnan(want):
            nan_ok = nan_ok and math.isnan(have)
            continue
        error = abs(have - want)
        worst_abs = worst(worst_abs, error)
        worst_rel = worst(
            worst_rel, error / abs(want) if want else (0.0 if error == 0 else math.inf)
        )
    bound = 2.0 * eps
    return Sweep(
        patterns=len(got),
        max_abs_error=worst_abs,
        max_rel_error=worst_rel,
        tolerance=bound,
        nan_preserved=nan_ok,
        passed=within(worst_rel, bound) and nan_ok,
    )


def _numerics_rowwise(torch: Any, provider: Provider, device: Any, dtype: Any) -> Sweep:
    """Decode all 256 e4m3 patterns against 256 DIFFERENT row scales — 65,536 pairs.

    `fp8-rowwise/1`'s scale is a CONTINUOUS f32 per output row, not a format with 256 code
    points, so there is no finite cross-product to exhaust the way `mxfp8/1` has. The
    honest substitute is the one that would catch the failure this route actually has: a
    row scale is applied per ROW, so the domain that matters is (every element pattern) x
    (a scale spread across the f32 exponent range), and a decoder that mis-strides the
    scale — reads it down the columns, or broadcasts a rank-2 `[out, 1]` as `[1, out]` —
    gets the wrong answer on every row but the diagonal.

    The scales are exact powers of two so the product carries ONE rounding, which is the
    same two-rounding budget the other decode oracles hold their providers to. That is a
    deliberate narrowing and it is stated: it checks the ROUTING of the scale exactly and
    the ROUNDING of an arbitrary f32 multiply not at all.
    """
    rows = cols = 256
    data = (
        torch.frombuffer(bytearray(bytes(range(256)) * rows), dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .reshape(rows, cols)
        .to(device)
    )
    exponents = [(r % 60) - 30 for r in range(rows)]
    scale = torch.tensor(
        [math.ldexp(1.0, e) for e in exponents], dtype=torch.float32, device=device
    )
    destination = torch.empty(rows, cols, dtype=dtype, device=device)
    provider.decode(torch, {"data": data, "scale": scale}, destination)
    got = destination.double().cpu().reshape(-1).tolist()
    return _domain_sweep(
        torch, dtype, got, [(math.ldexp(1.0, e), f"row scale 2**{e}") for e in exponents]
    )


def _domain_sweep(
    torch: Any, dtype: Any, got: Sequence[float], scales: Sequence[tuple[float, str]]
) -> Sweep:
    """Grade a decode of all 256 e4m3 elements under each row's `(scale, label)` into the
    five bands `_numerics_mx` names. `got` is row-major, one row per scale."""
    info = torch.finfo(dtype)
    finite_max, smallest_normal = float(info.max), float(info.tiny)
    ulp_sub = smallest_normal * _eps(torch, dtype)
    rel_bound, abs_bound = 2.0 * _eps(torch, dtype), 2.0 * ulp_sub
    elements = [e4m3_value(b) for b in range(256)]
    counts = dict.fromkeys(("normal", "subnormal", "overflow", "underflow", "nan"), 0)
    wrong = dict.fromkeys(counts, 0)
    worst_rel = worst_abs = 0.0
    first_bad = ""

    def note(band: str, why: str) -> None:
        nonlocal first_bad
        wrong[band] += 1
        first_bad = first_bad or why

    for r, (s, label) in enumerate(scales):
        for c, element in enumerate(elements):
            want = element * s
            have = got[r * len(elements) + c]
            where = f"element {c:#04x} x {label}"
            if math.isnan(want):
                counts["nan"] += 1
                if not math.isnan(have):
                    note("nan", f"{where}: {have} != nan")
            elif abs(want) > finite_max:
                counts["overflow"] += 1
                if not math.isinf(have) or (have > 0) != (want > 0):
                    note("overflow", f"{where}: {have} != inf")
            elif want != 0.0 and abs(want) < ulp_sub / 2.0:
                counts["underflow"] += 1
                if have != 0.0:
                    note("underflow", f"{where}: {have} != 0 (exact {want:.3e} is below half ULP)")
            elif abs(want) < smallest_normal:
                counts["subnormal"] += 1
                error = abs(have - want)
                worst_abs = worst(worst_abs, error)
                if not within(error, abs_bound):
                    note(
                        "subnormal",
                        f"{where}: {have} != {want} (abs {error:.3e} > {abs_bound:.3e})",
                    )
            else:
                counts["normal"] += 1
                error = abs(have - want)
                worst_abs = worst(worst_abs, error)
                relative = error / abs(want)
                worst_rel = worst(worst_rel, relative)
                if not within(relative, rel_bound):
                    note(
                        "normal",
                        f"{where}: {have} != {want} (rel {relative:.3e} > {rel_bound:.3e})",
                    )
    return Sweep(
        pairs=len(got),
        domain=counts,
        max_abs_error=worst_abs,
        max_rel_error=worst_rel,
        tolerance=rel_bound,
        subnormal_tolerance=abs_bound,
        nan_preserved=wrong["nan"] == 0,
        wrong=wrong,
        first_wrong=first_bad,
        passed=not any(wrong.values()),
    )


# ------------------------------------------------------------------- the native oracles


def _native_agreement(
    torch: Any, provider: LeafProvider, device: Any, dtype: Any
) -> NativeAgreement:
    """A NATIVE LEAF is checked by running its kernel and comparing it to arithmetic.

    A decode provider is checked against a format oracle because its output IS a number per
    stored element. A native leaf has no such output: its stored roles never become a float
    tensor at all, and what it produces is a GEMM. So the oracle is the GEMM — the leaf's
    own forward against an f64 reference matmul over the EXACTLY dequantized weights and
    the un-quantized activation, on identical inputs.

    Two failures this catches and nothing else in the runtime would:

    * **the kernel does not exist here.** `torch._scaled_mm` raising on this arch, this
      torch build or this dtype is recorded VERBATIM and mints no record — which is the
      whole of "a device with no measured suite mints nothing" for the native routes.
    * **#515d's silent-wrong-layout.** torch validates the MX scale grids' element COUNT and
      not their LAYOUT, so a row-major grid where the 128x4-tiled one belongs is ACCEPTED
      and returns wrong numbers at 62,453 of 64,485 finite positions. Nothing raises. A
      leaf that got the swizzle wrong fails HERE, with a deviation three orders of
      magnitude past the budget, and mints nothing.

    The budget is deliberately LOOSE — this is a w8a8 route and the activation quantization
    is a real, measured, accepted loss (#515a: 27.5 dB against the dequant lane's 30.4 dB).
    What the bound rejects is not imprecision, it is nonsense: the pass line sits where a
    mis-strided scale, a transposed operand or a wrong-layout grid lands, all of which
    measure worse than 0.5 relative, and no correct w8a8 route measures worse than 0.15.
    """
    weight, roles = _weight_roles(torch, provider, device)
    x = _activation(torch, device, dtype)
    reference = _reference_gemm(torch, x, weight)
    replaced = _StandIn(int(weight.shape[1]), int(weight.shape[0]))
    try:
        got = provider.leaf(torch, roles, replaced, dtype)(x)
    except Exception as exc:  # the kernel, the dtype or the arch says no — verbatim
        return NativeAgreement(
            passed=False,
            kernel=f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
            note="the native kernel refused on this device; no record is minted, and the "
            "encoding falls to whichever decode provider this card DID qualify",
        )
    deviation = _relative(torch, got, reference)
    finite = bool(torch.isfinite(got).all())
    contract = str(got.dtype) == str(x.dtype)
    return NativeAgreement(
        passed=within(deviation, NATIVE_DEVIATION_BOUND) and finite and contract,
        shape=[int(x.shape[0]), *[int(n) for n in weight.shape]],
        deviation_rel=deviation,
        tolerance=NATIVE_DEVIATION_BOUND,
        finite=finite,
        out_dtype=str(got.dtype).removeprefix("torch."),
        out_dtype_is_contract=contract,
    )


class _StandIn:
    """The module a leaf stands in for, during qualification only.

    A leaf reads its geometry off the module it REPLACES rather than off the payload, so
    the oracle has to hand it one. This is the smallest thing that answers those questions
    honestly: two integers and no bias. It is not an `nn.Module` and does not need to be —
    nothing installs it.
    """

    __slots__ = ("bias", "in_features", "out_features")

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None


def _weight_roles(torch: Any, provider: Provider, device: Any) -> tuple[Any, dict[str, Any]]:
    """A deterministic `BENCH_SHAPE` weight and the stored roles of it, per suite.

    SEEDED, so two probes of one device measure the same tensor and their records are
    comparable; and quantized by the encoding's OWN equation, so what the routes are then
    compared on is a real carrier rather than random bytes that no producer would emit.
    """
    rows, cols = BENCH_SHAPE
    generator = torch.Generator(device="cpu").manual_seed(0x511B)
    weight = torch.randn(rows, cols, generator=generator, dtype=torch.float32).to(device) * 0.05
    if provider.suite in ("fp8-rowwise-decode", "fp8-rowwise-native"):
        derived = quantize_rowwise(torch, weight)
        return weight, {"data": derived.data, "scale": derived.scale}
    if provider.suite in ("mxfp8-decode", "mxfp8-native"):
        work = weight.float().view(rows, cols // MX_BLOCK, MX_BLOCK)
        amax = work.abs().amax(dim=-1)
        # The CEILING rule (#517): `floor(log2 amax) - 8` saturates a block's own largest
        # element whenever its mantissa exceeds 1.75, which would make the oracle measure
        # the shipped quantizer's defect instead of the route's arithmetic.
        exponent = torch.where(
            amax > 0, torch.ceil(torch.log2(amax / 448.0)), torch.full_like(amax, -127.0)
        )
        byte = (exponent + 127.0).clamp(0.0, 254.0)
        mult = torch.pow(2.0, byte - 127.0)
        payload = (work / mult.unsqueeze(-1)).view(rows, cols)
        return weight, {
            "data": payload.clamp(-448.0, 448.0).to(torch.float8_e4m3fn),
            "scale": byte.to(torch.uint8),
        }
    if provider.suite == "fp8-scalar-decode":
        amax = weight.abs().max().clamp(min=1e-12)
        scale = (amax / 448.0).to(torch.float32)
        payload = (weight / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return weight, {"data": payload, "weight_scale": scale.reshape(())}
    raise AssertionError(f"no operand builder for suite {provider.suite!r}")


def _exact_weight(torch: Any, provider: Provider, roles: Mapping[str, Any]) -> Any:
    """The EXACT float64 value of the stored roles. The reference both routes are scored on.

    Written here in f64 rather than delegated to the decode provider on purpose: a reference
    that calls one of the two implementations under comparison is that implementation's
    self-check, not a reference. The arithmetic is the spec's and it is exact — an e4m3
    element and a power-of-two or f32 scale multiply without rounding in f64 — so what the
    deviation below measures is entirely the route's own.
    """
    data = roles["data"].double()
    if provider.suite in ("fp8-rowwise-decode", "fp8-rowwise-native"):
        return data * roles["scale"].double().reshape(int(data.shape[0]), 1)
    if provider.suite in ("mxfp8-decode", "mxfp8-native"):
        rows, cols = int(data.shape[0]), int(data.shape[1])
        mult = torch.pow(2.0, roles["scale"].double() - 127.0)
        return (data.view(rows, cols // MX_BLOCK, MX_BLOCK) * mult.unsqueeze(-1)).view(rows, cols)
    return data * roles["weight_scale"].double()


def _activation(torch: Any, device: Any, dtype: Any) -> Any:
    """The same seeded activation for every route, so a deviation is about the route."""
    generator = torch.Generator(device="cpu").manual_seed(0x517E)
    x = torch.randn(FIDELITY_TOKENS, BENCH_SHAPE[1], generator=generator, dtype=torch.float32)
    return (x * 0.5).to(device=device, dtype=dtype)


def _reference_gemm(torch: Any, x: Any, weight: Any) -> Any:
    """`x @ W.t()` in f64. f32 would be measured through TF32 on every card that has it,
    which is a ~1e-3 error sitting under numbers this suite reports to 1e-6."""
    return (x.double() @ weight.double().t()).double()


def _relative(torch: Any, got: Any, reference: Any) -> float:
    delta = (got.double() - reference).norm()
    return float(delta / reference.norm().clamp(min=1e-30))


def _decode_axes(torch: Any, provider: Provider, device: Any, dtype: Any) -> RouteAxes:
    """The two MEASURED axes #517e's ranking orders providers on, in ONE comparable frame.

    Both routes are scored against the same reference — an f64 matmul over the exact value
    of the same stored roles — so `deviation_rel` prices the WHOLE route on each side: the
    decode route's rounding into the destination dtype and the float GEMM that follows, and
    the native route's activation quantization and its encoded GEMM. That comparability is
    the point. A fidelity number that measured a decode's element error against a native
    route's GEMM error would order the two on a scale neither shares.

    Footprint is the other side of the same fact and is read the same way: what stays
    RESIDENT per logical element once the fill is over. On a decode route that is the
    destination dtype's own width, because the weights are float either way; on a native
    route it is the role bytes, because the roles are the weight (`_native_axes`).
    """
    _, roles = _weight_roles(torch, provider, device)
    destination = torch.empty(*BENCH_SHAPE, dtype=dtype, device=device)
    provider.decode(torch, roles, destination)
    x = _activation(torch, device, dtype)
    reference = _reference_gemm(torch, x, _exact_weight(torch, provider, roles))
    # The weights are FLOAT after this route's fill, which is the whole of "it buys bytes
    # read and load time, NOT VRAM". Reading it off the destination rather than asserting
    # 16 keeps it true for an f32 destination too.
    return RouteAxes(
        _relative(torch, x @ destination.t(), reference), 8.0 * destination.element_size()
    )


#: torch dtype name -> TensorFS carrier name, for the role parts a leaf prices.
_CARRIER = {name: carrier for carrier, name in TORCH_DTYPES.items()}


def _native_axes(torch: Any, provider: LeafProvider, device: Any, deviation: float) -> RouteAxes:
    """A native route's axes: its measured GEMM deviation, and the role bytes the leaf keeps
    resident, priced by the provider from the probe's real tensors (MXFP8 pads its scale grid
    to the blocked layout, so a raw byte count would under-price it)."""
    weight, roles = _weight_roles(torch, provider, device)
    parts = {
        role: RolePart(
            role,
            _CARRIER[str(t.dtype).removeprefix("torch.")],
            tuple(int(n) for n in t.shape),
            int(t.numel() * t.element_size()),
        )
        for role, t in roles.items()
    }
    return RouteAxes(deviation, provider.resident_bits_per_element(parts, int(weight.numel())))


# ------------------------------------------------------- the reviewed spec's own vectors


#: Where the vendored vector documents live. They ship INSIDE the package, so an installed
#: wheel qualifies exactly as the repo does; a suite whose subject is only present in a
#: source checkout qualifies nothing in production.
VECTOR_ROOT = Path(__file__).resolve().parent / "encoding" / "vectors"


class VectorBlob(msgspec.Struct, frozen=True):
    dtype: str
    shape: list[int]
    bits: str


class MinedLogical(msgspec.Struct, frozen=True):
    dtype: str
    shape: list[int]


class MinedCase(msgspec.Struct, frozen=True):
    name: str
    logical: MinedLogical
    roles: dict[str, VectorBlob]
    expect_logical_bits: str


class VectorDocument(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A producer-mined vector document, as TensorFS vendors it."""

    cases: list[MinedCase]


def vector_tensor(torch: Any, blob: VectorBlob, device: Any) -> Any:
    """One vector role/logical field -> the tensor it encodes. PUBLIC: the conformance
    runner builds the same tensors to compare a rounding rule against the reference.

    `bits` is base64 of the LITTLE-ENDIAN raw bytes, exactly as the carrier stores them, so
    this is a `frombuffer` and a `view` and never a parse. A dtype the runtime has no torch
    binding for is a refusal rather than a guess, for the same reason the fill plane refuses
    one: landing the wrong number of bytes is worse than not landing them.
    """
    torch_name = TORCH_DTYPES.get(blob.dtype)
    if torch_name is None:
        raise AssertionError(
            f"the vendored vectors cite carrier dtype {blob.dtype!r} and this "
            "runtime has no torch binding for it"
        )
    flat = torch.frombuffer(bytearray(base64.b64decode(blob.bits)), dtype=torch.uint8)
    return flat.view(getattr(torch, torch_name)).reshape(blob.shape).to(device)


def spec_vector_document(spec_digest: str) -> VectorDocument | None:
    """The vendored vector document this SPEC DIGEST names, or None if it names none.

    #549.4's substance. A provider declares reviewed spec DIGESTS; each reviewed digest maps
    to the producer-mined vector document the registry's own seed attaches to that spec, and
    this is what the qualification suite decodes. `plain/1` has no entry: its route decodes
    nothing and its evidence is the fill plane's byte-identity proof.

    The reviewed table pins both the nested spec digest and its vector-byte digest. A copied
    fixture that drifts therefore stops before JSON parsing or provider execution. Pinned
    bytes this reader cannot decode raise `msgspec.ValidationError`: that is the reader
    drifting from the bytes, and it must fail rather than skip.
    """
    pinned = SPEC_VECTORS.get(spec_digest)
    if pinned is None:
        return None
    name, expected_digest = pinned
    try:
        raw = (VECTOR_ROOT / name).read_bytes()
    except OSError:
        return None
    if "sha256:" + hashlib.sha256(raw).hexdigest() != expected_digest:
        return None
    return msgspec.json.decode(raw, type=VectorDocument)


def case_dtype(case: MinedCase) -> str | None:
    """The TORCH dtype name one mined case's logical value carries, or None.

    The key is `dtype`. These documents are TensorFS's producer-mined vector artifacts,
    pinned BY DIGEST in `SPEC_VECTORS`, and their `logical` object is the same shape
    `vector_tensor` reads one line below — it is NOT the CozyTensors header's `logical`,
    whose dtype field TensorFS 0.3 renamed to `logical_dtype`. Reading the header's name
    here matched nothing, and a document that matches nothing reports SKIPPED, which is a
    pass locally and an absence everywhere else.
    """
    return TORCH_DTYPES.get(case.logical.dtype)


def _spec_vectors(torch: Any, provider: Provider, device: Any) -> SpecVectors:
    """Decode the REVIEWED SPEC's own producer-mined cases, BIT-FOR-BIT (#549.4).

    The synthetic oracles above sweep a format's whole domain against arithmetic written from
    the OCP spec — they prove the decoder's numbers. This proves something they cannot: that
    the decoder reads the exact ROLE LAYOUT a real producer emits, at the exact rank and
    stride the reviewed spec digest pins, on bytes that came out of the producer rather than
    out of this repo.

    The comparison is on RAW BITS, not on a tolerance. These cases carry the producer's own
    expected logical output for the producer's own carrier; a decoder that is one ULP away
    from it has made a different choice than the producer did (`fp8-scaled-scalar/1`'s doc
    calls that out explicitly: narrowing the scale before the multiply differs from
    multiplying in f32 by a full ulp on 132 of 512 bf16 elements). A tolerance here would
    admit exactly the drift the vectors were mined to catch.

    Every mined case runs at the producer's own logical dtype, whatever output dtype the fill
    asked about: the vectors prove the layout, which no output dtype changes, and the
    synthetic oracle proves the arithmetic at the requested dtype. Filtering cases by the
    requested dtype skipped the whole check for a float16-only fill (SDXL) because the
    fp8-rowwise and mxfp8 documents mine f32 and bf16 cases only, `_reportable` dropped the
    skip, and `tensorfs.fit` refused `encoding_unqualified` on every card (run 1312).

    A spec with no vendored document is reported as SKIPPED and neither passes nor fails: it
    must not mint a record on the strength of a check that did not run.

    A document whose cases declare NO dtype this runtime reads is neither of those things —
    it is a reader that stopped matching the bytes, and it FAILS. Skipping there is how the
    observed-capability document quietly emptied itself: `_reportable` drops a skipped
    check, so every accelerator lost every non-verbatim record and `tensorfs.fit` refused
    `encoding_unqualified` on a card whose decode had actually passed.
    """
    spec = f"{provider.encoding[:30]}…"
    try:
        document = spec_vector_document(provider.encoding)
    except msgspec.ValidationError as exc:
        return SpecVectors(passed=False, note=f"the vendored document for spec {spec}: {exc}")
    if document is None:
        return SpecVectors(
            passed=True, skipped=True, note=f"no vendored vector document for spec {spec}"
        )
    if document.cases and not any(case_dtype(case) for case in document.cases):
        return SpecVectors(
            passed=False,
            note=f"the vendored document for spec {spec} declares no logical dtype this "
            f"runtime reads — its cases carry {sorted({c.logical.dtype for c in document.cases})}"
            ", and the reader and the mined bytes have to agree before a case can run",
        )
    cases: list[VectorCheck] = []
    for case in document.cases:
        logical_dtype = case_dtype(case)
        if logical_dtype is None:
            continue
        roles = {role: vector_tensor(torch, blob, device) for role, blob in case.roles.items()}
        missing = sorted(provider.required - set(roles))
        surplus = sorted(set(roles) - provider.required - provider.optional)
        if missing or surplus:
            why = (
                f"the mined case stores roles {sorted(roles)} and this provider requires "
                f"{sorted(provider.required)} (missing {missing}, unhandled {surplus})"
            )
            cases.append(VectorCheck(case.name, passed=False, why=why))
            continue
        shape = case.logical.shape
        why = provider.supports(shape)
        if why:
            cases.append(VectorCheck(case.name, passed=True, skipped=True, why=why))
            continue
        destination = torch.empty(*shape, dtype=getattr(torch, logical_dtype), device=device)
        provider.decode(torch, roles, destination)
        want = vector_tensor(
            torch, VectorBlob(case.logical.dtype, shape, case.expect_logical_bits), device
        )
        got_bits = destination.reshape(-1).view(torch.uint8).cpu()
        want_bits = want.reshape(-1).view(torch.uint8).cpu()
        differing = int((got_bits != want_bits).sum())
        cases.append(
            VectorCheck(
                case.name,
                passed=differing == 0,
                elements=int(want.numel()),
                differing_bytes=differing,
            )
        )
    if not cases:
        return SpecVectors(passed=True, skipped=True, note="the vendored document carries no case")
    filename, sha256 = SPEC_VECTORS[provider.encoding]
    return SpecVectors(
        passed=all(c.passed for c in cases), document=VectorRef(filename, sha256), cases=cases
    )


def _micro_bench(torch: Any, provider: Provider, device: Any, dtype: Any) -> DecodeBench:
    rows, cols = BENCH_SHAPE
    destination = torch.empty(rows, cols, dtype=dtype, device=device)
    _, parts = _weight_roles(torch, provider, device)
    provider.decode(torch, parts, destination)
    # The fence goes through the accelerator boundary (#447): timing an MPS queue without
    # its own synchronize would mint a fictional decode rate, so mps refuses typed there.
    accel.synchronize(torch, device.type)
    started = time.perf_counter_ns()
    for _ in range(BENCH_REPS):
        provider.decode(torch, parts, destination)
    accel.synchronize(torch, device.type)
    took = (time.perf_counter_ns() - started) / BENCH_REPS
    # Bytes the decode actually touches: every stored role IN, the logical tensor OUT. Summed
    # over the parts rather than assuming one byte per element, because mxfp8's second role
    # is real traffic (1/32 of the payload) and a rate that ignores it flatters this route.
    moved = rows * cols * destination.element_size() + sum(
        int(t.numel() * t.element_size()) for t in parts.values()
    )
    return DecodeBench(list(BENCH_SHAPE), BENCH_REPS, int(took), round(moved / took, 3))


def _leaf_bench(torch: Any, provider: LeafProvider, device: Any, dtype: Any) -> LeafBench:
    """One native leaf's FORWARD, timed, against the same shape's bf16 GEMM.

    The decode bench times a decode because that is the whole of what a decode route costs
    at fill. A native leaf costs nothing at fill and everything per step, so the number that
    means the same thing is its forward — and it is reported BESIDE the float GEMM it
    replaces, because "2.03x" and "0.52x" are the same kernel with and without
    `use_fast_accum` (#517b) and a bare microsecond count hides which one ran.

    The activation quantization is INSIDE the timed region on purpose. #515f's finding is
    that an unfused eager quantizer costs 0.11-2.80 ms and can eat the GEMM win on small
    shapes; a benchmark that timed the bare GEMM would publish a speedup this build does
    not have.
    """
    weight, roles = _weight_roles(torch, provider, device)
    x = _activation(torch, device, dtype)
    leaf = provider.leaf(torch, roles, _StandIn(int(weight.shape[1]), int(weight.shape[0])), dtype)
    float_weight = weight.to(dtype)

    def timed(fn: Callable[[], object]) -> float:
        fn()
        accel.synchronize(torch, device.type)
        started = time.perf_counter_ns()
        for _ in range(BENCH_REPS):
            fn()
        accel.synchronize(torch, device.type)
        return (time.perf_counter_ns() - started) / BENCH_REPS

    leaf_ns = timed(lambda: leaf(x))
    float_ns = timed(lambda: x @ float_weight.t())
    return LeafBench(
        shape=[FIDELITY_TOKENS, *[int(n) for n in weight.shape]],
        reps=BENCH_REPS,
        leaf_ns=int(leaf_ns),
        float_gemm_ns=int(float_ns),
        speedup_x=round(float_ns / max(leaf_ns, 1.0), 4),
    )


#: Which self-check a provider gets, keyed by the ORACLE IT DECLARES (`Provider.suite`) and
#: never by class or role set, so a second implementation of one encoding is measured by the
#: same oracle as the first. A suite name in neither table qualifies NOTHING: minting a record
#: for a decoder no oracle checked is exactly the fabricated evidence the table forbids.
#: (Keyed on the ROLE SET, `mxfp8/1`'s decode and native leaf — both `{data, scale}` — would
#: have shared the decode oracle, which never runs the kernel and cannot see #515d.)
DECODE_SUITES: dict[str, Callable[[Any, Provider, Any, Any], Sweep]] = {
    "fp8-scalar-decode": _numerics,
    "mxfp8-decode": _numerics_mx,
    "fp8-rowwise-decode": _numerics_rowwise,
}

#: Native leaves' oracle is their own GEMM against arithmetic (`_native_agreement`).
NATIVE_SUITES = frozenset({"fp8-rowwise-native", "mxfp8-native"})


def qualify(
    torch: Any,
    providers: Mapping[str, tuple[Provider, ...]],
    device: DeviceFacts,
    output_dtypes: Sequence[str],
    *,
    release: str,
    runtime: RuntimeIdentity | None = None,
) -> tuple[Capabilities, QualificationResult]:
    """Run the suite and return this process's admission table.

    The evidence for a route that decodes nothing (`plain/1`) is the fill plane's own
    byte-identity proof, so its record carries the trivial observation rather than a
    fabricated one — a record with no evidence is the thing this design forbids, and a
    record whose evidence says "verbatim" is honest.

    The BACKEND is gated first, through `accel` (#496c/#503d: every backend question lives
    there). This is that gate's first consumer, and the reason it needs one: without it a
    backend with no measured suite would run these vectors anyway and mint records whose
    device predicate claims a card the suite never ran on. Refusing here is #501f's runtime
    twin — the capability answer is typed and it happens BEFORE any fill, not after a
    manifest has been believed.

    `runtime` is #549.9's other half. A capability record is a claim about an IMPLEMENTATION
    on a device, and only the device half used to have an identity. The runtime's source
    digest and torch build ride every record. Defaulted rather than required so a caller that
    has torch and a release can still probe; the default MEASURES, it does not fabricate.
    """
    accel.qualification_gate(device.kind)
    if runtime is None:
        runtime = measure_runtime(torch, release)
    target = torch.device("cuda" if device.kind == "cuda" else "cpu")
    started = time.perf_counter_ns()
    observations: list[Observation] = []
    for encoding in sorted(providers):
        for provider in providers[encoding]:
            for name in output_dtypes:
                dtype = getattr(torch, name)
                if provider.route != "verbatim" and not dtype.is_floating_point:
                    # A decoded-float route has nothing to say about an integer destination,
                    # and neither has a native leaf: `_scaled_mm` emits floats. DERIVED from
                    # the dtype rather than listed — the registry's fp8 specs declare float
                    # logical dtypes and a checkpoint's index buffers are plain/1 anyway, so
                    # this skip removes a record that could never be selected rather than
                    # hiding one that could.
                    continue
                observations.append(
                    Observation(
                        encoding_spec_id=encoding,
                        implementation=provider.name,
                        implementation_digest=implementation_digest(provider),
                        implementation_revision=provider.code_revision,
                        device_predicate=device.predicate,
                        output_dtype=name,
                        geometry_class=geometry_class(BENCH_SHAPE),
                        delivery_route=provider.route,
                        evidence=_evidence(torch, provider, device, target, dtype),
                    )
                )
    result = QualificationResult(
        tuple_key=device.tuple_key(),
        device=device,
        device_predicate=device.predicate,
        runtime_release=release,
        runtime=runtime,
        suite_ms=(time.perf_counter_ns() - started) // 1_000_000,
        observations=observations,
    )
    return _table(result), result


def _evidence(
    torch: Any, provider: Provider, device: DeviceFacts, target: Any, dtype: Any
) -> Evidence:
    """One (implementation, output dtype)'s evidence. Only a PASSED one mints a record."""
    if provider.route == "verbatim":
        return VerbatimEvidence(
            note="no decode: the stored bytes ARE the logical bytes, and the fill plane's "
            "source-comparison proves them byte for byte"
        )
    if not device.satisfies(provider.device_predicate):
        # THE PROVIDER'S OWN FLOOR, honoured before its oracle runs. Qualifying a leaf on a
        # card whose kernel it does not claim would either fail confusingly or — worse —
        # pass, and mint a record whose predicate says this card because `device.predicate`
        # is where records get theirs.
        return UnmeasuredEvidence(
            note=f"{provider.name} declares the floor {provider.device_predicate} and this "
            f"device is {device.predicate}; its suite is not run and it mints no record"
        )
    if isinstance(provider, LeafProvider) and provider.suite in NATIVE_SUITES:
        agreement = _native_agreement(torch, provider, target, dtype)
        if not agreement.passed:
            return NativeEvidence(numerics=agreement, passed=False, route_axes=RouteAxes())
        return NativeEvidence(
            numerics=agreement,
            passed=True,
            micro_bench=_leaf_bench(torch, provider, target, dtype),
            route_axes=_native_axes(torch, provider, target, agreement.deviation_rel),
        )
    suite = DECODE_SUITES.get(provider.suite)
    if suite is None:
        # An implementation no oracle covers. NOT an error and NOT a pass: it mints no
        # record, so the encoding refuses at bind naming this device — the same answer an
        # unqualified card gets, for the same reason.
        return UnmeasuredEvidence(
            note=f"no oracle named {provider.suite!r}; an unchecked implementation is not a "
            "qualified one"
        )
    sweep = suite(torch, provider, target, dtype)
    # #549.4: the REVIEWED SPEC's own producer-mined cases, decoded bit-for-bit. The
    # synthetic sweep proves the arithmetic; this proves the provider reads the layout a real
    # producer emits at the rank and stride THIS digest pins. Both must pass to mint.
    vectors = _spec_vectors(torch, provider, target)
    passed = sweep.passed and vectors.passed
    return DecodeEvidence(
        numerics=sweep,
        spec_vectors=vectors,
        micro_bench=_micro_bench(torch, provider, target, dtype),
        passed=passed,
        route_axes=_decode_axes(torch, provider, target, dtype) if passed else RouteAxes(),
    )


def _table(result: QualificationResult) -> Capabilities:
    """Observations -> the admission table. A FAILED observation mints no record."""
    return Capabilities(
        [
            CapabilityRecord(
                encoding_spec_id=observation.encoding_spec_id,
                implementation_digest=observation.implementation_digest,
                runtime_release=result.runtime_release,
                device_predicate=observation.device_predicate,
                output_dtype=observation.output_dtype,
                delivery_route=observation.delivery_route,
                evidence_digest=canonical.digest(msgspec.to_builtins(observation.evidence)),
                resident_bits_per_element=observation.evidence.route_axes.resident_bits_per_element,
                kernel_numeric_error=observation.evidence.route_axes.deviation_rel,
                leaf_speedup_x=_leaf_speedup(observation.evidence),
                # #549.9's identity, carried onto every record it minted.
                runtime_source_digest=result.runtime.source_digest,
                torch_build=result.runtime.torch_build,
                os_arch=result.device.os_arch,
                device_index=result.device.index,
                device_driver=result.device.driver,
                cuda_runtime=result.device.cuda_runtime,
                geometry_class=observation.geometry_class,
                implementation_revision=observation.implementation_revision,
            )
            for observation in result.observations
            if observation.evidence.passed
        ]
    )


def _leaf_speedup(evidence: Evidence) -> float:
    """`speedup_x` off a native observation's leaf bench; -1.0 for everything else."""
    if isinstance(evidence, NativeEvidence) and evidence.micro_bench is not None:
        return evidence.micro_bench.speedup_x
    return -1.0


def leaf_speedups(result: QualificationResult) -> dict[str, float]:
    """`<implementation>@<output dtype>` -> the WORST `speedup_x` a passed leaf observation
    banked for it, across every spec digest it was measured under. What the boot record
    prints, so a leaf this card measured losing to its float GEMM is read where the route
    census is read."""
    worst: dict[str, float] = {}
    for observation in result.observations:
        speedup = _leaf_speedup(observation.evidence)
        if not observation.evidence.passed or speedup < 0.0:
            continue
        name = f"{observation.implementation}@{observation.output_dtype}"
        worst[name] = min(worst.get(name, speedup), speedup)
    return dict(sorted(worst.items()))


#: Bumped when a stored document's MEANING changes in a way no other term of the key
#: catches. Every other term is derived from what was measured, so this exists for the one
#: case they cannot cover: a change to what this file does with a document it reads back.
QUALIFICATION_FORMAT = 1

#: How many stored qualifications one cache directory keeps. A document is ~11-33 KB and a
#: key changes only when the card, the build, the providers or the dtypes do, so this is
#: not a working-set bound — it is a floor under unbounded growth on a rig that upgrades
#: its driver every week.
QUALIFICATION_KEEP = 64


@dataclass(frozen=True, slots=True)
class Qualification:
    """One admission table, and WHERE IT CAME FROM.

    The provenance is not decoration. A cache whose firing nobody can see is a cache
    nobody can prove fired, and the whole claim of this change is that it fires on the
    second prepare — so `source` rides the prepare facts and the worker prints it.
    """

    capabilities: Capabilities
    result: QualificationResult
    key: str
    measured: bool
    note: str = ""

    def document(self) -> dict[str, str | int | dict[str, float]]:
        return {
            "key": self.key,
            "source": "measured" if self.measured else "reused",
            "suite_ms": self.result.suite_ms,
            "observations": len(self.result.observations),
            "leaf_speedup_x": leaf_speedups(self.result),
            "note": self.note,
        }


class StoredQualification(msgspec.Struct, frozen=True):
    """One cache file: a result, filed under the key it was measured for."""

    key: str
    result: QualificationResult


def qualification_key(
    providers: Mapping[str, tuple[Provider, ...]],
    device: DeviceFacts,
    output_dtypes: Sequence[str],
    *,
    runtime: RuntimeIdentity,
) -> str:
    """Everything a qualification result depends on, and nothing that it does not.

    * **the device**, whole. `DeviceFacts` exists to be the thing a probe result is keyed
      by — card, index, NVML configuration (power cap and MIG profile), the REAL driver,
      the CUDA runtime torch was built against, and the OS/arch. Two "same model" cards at
      different power caps are not the same measurement subject.
    * **the runtime**, whole. `source_digest` is over `encoding/`, `instruments.py`,
      `probe.py` and `resolution.py` as installed, so editing a decode, a leaf forward or
      this suite changes the key. `torch_build` and `release` ride with it.
    * **the providers**, by `implementation_digest` — name, route, floor, oracle, roles,
      code revision and reviewed spec set. An implementation whose arithmetic changed is a
      different implementation and gets a different key.
    * **the output dtypes**, because the suite runs per dtype and a document measured over
      `["bfloat16"]` has nothing to say about a `float16` destination. The match is exact
      rather than by superset: reusing a wider document would mint records for dtypes this
      construction never asked about, and a narrower one would silently qualify nothing.

    NOT in the key: free VRAM, host load, the clock, the manifest, the model, the package.
    None of them can change what the suite measures, and putting any of them in would make
    the cache miss forever, which is the failure mode that looks exactly like working.
    """
    return canonical.digest(
        {
            "format": QUALIFICATION_FORMAT,
            "device": device.document(),
            "runtime": runtime.document(),
            "providers": {
                spec: sorted(implementation_digest(one) for one in group)
                for spec, group in providers.items()
            },
            "output_dtypes": list(output_dtypes),
        }
    )


def qualified(
    torch: Any,
    providers: Mapping[str, tuple[Provider, ...]],
    device: DeviceFacts,
    output_dtypes: Sequence[str],
    *,
    release: str,
    runtime: RuntimeIdentity | None = None,
    cache: str = "",
) -> Qualification:
    """The admission table for these inputs, measured once per key and kept.

    `cache` is a DIRECTORY, not a switch: unset means nowhere to keep a result, and the
    served code path is identical either way — the suite runs, the same table comes back,
    and nothing is stored. Every filesystem failure is the same outcome as an empty cache,
    named in `note` and never raised: a rig whose cache directory is unwritable must still
    serve, and it must not serve a table it did not obtain.

    A stored document is USED only when it carries its own key and that key is the one
    these inputs produce. The table is then rebuilt from the stored observations by the
    same `_table` the measured path uses, so a reused qualification and a measured one are
    the same function of the same evidence.
    """
    if runtime is None:
        runtime = measure_runtime(torch, release)
    key = qualification_key(providers, device, output_dtypes, runtime=runtime)
    stored, note = _read_qualification(cache, key)
    if stored is not None:
        return Qualification(_table(stored), stored, key, measured=False, note=note)
    capabilities, result = qualify(
        torch, providers, device, output_dtypes, release=release, runtime=runtime
    )
    # A MISS ALWAYS WRITES, including a miss caused by a document that was there and was
    # not usable. Reusing the read's reason as the write's would leave a corrupt or
    # mis-filed document in place forever, so a rig that hit one once would re-measure on
    # every prepare for the life of the key with nothing said about why.
    kept = _write_qualification(cache, key, result)
    return Qualification(
        capabilities,
        result,
        key,
        measured=True,
        note="; ".join(dict.fromkeys(part for part in (note, kept) if part)),
    )


def _read_qualification(cache: str, key: str) -> tuple[QualificationResult | None, str]:
    """The stored result for this key, or nothing and the reason there is nothing."""
    if not cache:
        return None, "no cache directory"
    try:
        # stdlib JSON both ways: a failed native route can measure a NaN or infinite
        # deviation, which `msgspec.json` would write as null and then refuse to read back.
        raw = json.loads((Path(cache) / f"{key}.json").read_bytes())
        stored = msgspec.convert(raw, StoredQualification)
    except FileNotFoundError:
        return None, ""
    except (OSError, ValueError, msgspec.ValidationError) as exc:
        return None, f"unreadable: {type(exc).__name__}: {exc}"
    if stored.key != key:
        # A document that does not carry the key it is filed under is not evidence about
        # these inputs, whatever it is evidence about.
        return None, "stored document does not carry its own key"
    return stored.result, ""


def _write_qualification(cache: str, key: str, result: QualificationResult) -> str:
    """Keep this measurement for the next process. Returns why it was not kept, or ""."""
    if not cache:
        return "no cache directory"
    directory = Path(cache)
    temporary = directory / f".{key}.{os.getpid()}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(msgspec.to_builtins(StoredQualification(key, result))))
        temporary.chmod(0o600)
        os.replace(temporary, directory / f"{key}.json")
        _prune_qualifications(directory)
    except (OSError, ValueError) as exc:
        with contextlib.suppress(OSError):
            temporary.unlink()
        return f"not kept: {type(exc).__name__}"
    return ""


def _prune_qualifications(directory: Path) -> None:
    """Keep the newest `QUALIFICATION_KEEP` documents. Best effort, never fatal."""
    with contextlib.suppress(OSError):
        stored = sorted(directory.glob("*.json"), key=lambda one: one.stat().st_mtime)
        for old in stored[: max(len(stored) - QUALIFICATION_KEEP, 0)]:
            with contextlib.suppress(OSError):
                old.unlink()


class UnreportableObservation(Exception):
    """A record would have crossed the process boundary carrying a byte TensorFS refuses.

    Loud rather than dropped: a silently omitted record reads to the hub as "this card
    cannot run it", which is the wrong answer with the wrong remedy.
    """


def observed_capability_records(result: QualificationResult) -> dict[str, list[dict[str, str]]]:
    """This probe result, projected into TensorFS's OBSERVED capability-records document.

    The sibling of `_table`, over the same observation and for the other consumer. `_table`
    builds the admission table this process fills against; this builds the document another
    process joins against, because TensorFS owns the admission RULE and holds no card while
    this process holds one and owns the OBSERVATION.

    The document states what was qualified and WHERE, and deliberately not which vector set
    qualified it: the spec pins that, and an observer free to name it could qualify a spec
    against bytes the spec never carried. TensorFS binds the vector set itself and refuses a
    record whose spec has none.

    One record per (encoding, device), because that pair is what the rule admits on. The
    suite mints one per output dtype and implementation, so several collapse into one and
    every implementation that earned it is named.

    REPRODUCIBLE for a given card class, runtime build and provider set. The observation's
    `evidence_digest` is deliberately absent: it covers the micro-benchmark, so it moves on
    every run, and a registered artifact that churns without a change of meaning is one
    nobody can diff. What was qualified is carried WHOLE by `implementation_digest`, which is
    over the implementation's name, route, floor, oracle, roles, revision and reviewed specs
    and over no timing at all.
    """
    records: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    for observation in result.observations:
        if not observation.evidence.passed or not _reportable(observation.evidence):
            continue
        implementations, dtypes = records.setdefault(
            (observation.encoding_spec_id, observation.device_predicate), (set(), set())
        )
        implementations.add(
            f"{observation.implementation}[{observation.delivery_route}]"
            f"@{observation.implementation_digest}"
        )
        dtypes.add(observation.output_dtype)
    out = [
        {
            "device": device,
            "encoding": encoding,
            "implementation": " ".join(sorted(implementations)),
            "note": "qualified by the cozy-runtime probe suite for output dtypes "
            + ",".join(sorted(dtypes)),
        }
        for (encoding, device), (implementations, dtypes) in sorted(records.items())
    ]
    # TensorFS's canonical layer is printable-ASCII only. A stray character would refuse the
    # document at the border with an error that points at bytes, not at the provider.
    for index, record in enumerate(out):
        for field, value in record.items():
            if not value.isascii() or not value.isprintable():
                raise UnreportableObservation(
                    f".records[{index}].{field} carries a non-printable-ASCII character: {value!r}"
                )
    return {"records": out}


def _reportable(evidence: Evidence) -> bool:
    """May this passing observation be REPORTED, not merely believed here?

    A `verbatim` route decodes nothing and its evidence is the fill plane's byte-identity
    proof, which is evidence anywhere. Any other route must have actually RUN the spec's
    producer-mined vectors: `_spec_vectors` reports a spec with no vendored document as
    SKIPPED and passes it, which is right for a local table built beside the synthetic
    oracle that did run, and is not a claim to hand another component. It is also the fence
    that keeps a spec TensorFS carries vectorless out of the document, where it would refuse
    the whole thing rather than one record.
    """
    if isinstance(evidence, VerbatimEvidence):
        return True
    return isinstance(evidence, DecodeEvidence) and not evidence.spec_vectors.skipped
