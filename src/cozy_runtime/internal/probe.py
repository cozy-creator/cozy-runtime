"""cr-006 — the per-device qualification suite that MINTS capability records.

§3.2's cost placement, made concrete: this runs ONCE per (card architecture, torch minor,
Runtime source), at a release gate, and its result ships as a typed table (`qualified`). Per
request there is only arithmetic.

Four things happen per (encoding, IMPLEMENTATION, output dtype), in this order, and a
failure at any one leaves NO record — which is what "fail-closed" means when the table is
minted rather than declared. The middle term is new: since #517e an encoding carries
ORDERED CANDIDATES rather than one provider, and each candidate is qualified separately,
because "fp8.rowwise works here" is not a fact — "this implementation of it works here" is.

1. **numerics self-check** against an INDEPENDENT reference. The e4m3 decoder below is
   written from the OCP number format by integer arithmetic, not by asking torch to cast —
   a self-check that uses the implementation as its own oracle checks nothing. The domain
   is the WHOLE format: all 256 e4m3 bit patterns, at a per-tensor scale or against 256
   row scales, NaN included (as a NaN, not a number) and overflow included (as an
   infinity, not a plausible finite number). The suite a provider gets is keyed on the
   ORACLE NAME it declares (`Provider.suite`), so an implementation with no oracle
   qualifies nothing rather than passing by default.
   A NATIVE LEAF's oracle is a different shape and has to be: its stored roles never become
   a float tensor, so there is nothing per-element to check. `_native_agreement` runs the
   leaf's real kernel on the real arch against an f64 matmul over the exactly-dequantized
   weights — which is the only check that can see a kernel that returns wrong numbers
   without raising.
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

**The result is release data** (`qualification`): one table per card architecture, torch
minor and Runtime source, measured by the release gates and shipped in the wheel.
"""

from __future__ import annotations

import base64
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import msgspec
from tensorfs import UnsupportedError

from cozy_runtime.author._loader import TENSORFS_DTYPES
from cozy_runtime.author._numeric import within, worst
from cozy_runtime.derive.microscale import encode_nvfp4
from cozy_runtime.internal import accel, canonical
from cozy_runtime.internal.decode_cores import Cores
from cozy_runtime.internal.decode_schemes import run, sample
from cozy_runtime.internal.encoding import (
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
from cozy_runtime.internal.encoding.adapters import CoreDequant
from cozy_runtime.internal.encoding.formats import (
    E2M1_VALUES,
    E8M0_NAN,
    MX_BLOCK,
    NVFP4_BLOCK,
    Tensor,
    e8m0_multipliers,
    hadamard256,
    params_of,
)

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

#: Where a native route's GEMM stops being lossy and starts being wrong. #517d measured the
#: real thing at 0.146 relative (16.7 dB); a mis-strided scale or a transposed operand
#: measures past 0.5. The line is drawn in the gap and it is a SANITY bound, not a quality
#: one — quality is the ranked axis, not the gate.
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
    """The conformance cases of a provider's schemes. SKIPPED neither passes nor fails."""

    passed: bool
    skipped: bool = False
    note: str = ""
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
    scheme: str
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


def _eps(torch: Any, dtype: Any) -> float:
    return float(torch.finfo(dtype).eps)


def _numerics(torch: Any, provider: Provider, device: Any, dtype: Any) -> Sweep:
    """All 256 E4M3 patterns under one f32 scale: the decode is `round_L(weight * scale)`, the
    scale entering exactly, so it must equal the f64 product cast once into the dtype."""
    data = torch.frombuffer(bytearray(bytes(range(256))), dtype=torch.uint8)
    data = data.view(torch.float8_e4m3fn).reshape(16, 16)
    scale = torch.tensor(0.0123, dtype=torch.float32)
    destination = torch.empty(16, 16, dtype=dtype, device=device)
    provider.decode(torch, {"weight": data.to(device), "scale": scale.to(device)}, destination)
    return _exact_sweep(torch, destination.cpu(), data.double() * scale.double(), dtype)


def _numerics_rowwise(torch: Any, provider: Provider, device: Any, dtype: Any) -> Sweep:
    """Decode all 256 e4m3 patterns against 256 DIFFERENT row scales — 65,536 pairs.

    `fp8.rowwise`'s scale is a CONTINUOUS f32 per output row, not a format with 256 code
    points, so there is no finite cross-product of element and scale codes to exhaust. The
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
    provider.decode(torch, {"weight": _across(provider, data), "scale": scale}, destination)
    got = _across(provider, destination).cpu().double().reshape(-1).tolist()
    return _domain_sweep(
        torch, dtype, got, [(math.ldexp(1.0, e), f"row scale 2**{e}") for e in exponents]
    )


def _numerics_int8_convrot(
    torch: ModuleType, provider: Provider, device: Tensor, dtype: Tensor
) -> Sweep:
    """Every int8 value in every Hadamard position, against 256 power-of-two row scales.

    Row `r` holds all 255 values -127..127 rotated by `r`, so every value meets every row of
    H, and a decoder that skips the rotation, rotates the wrong axis or mis-strides the scale
    is wrong almost everywhere. With power-of-two scales the integer Hadamard sum times the
    scale is exact in f32, so the destination's one RNE cast is the whole budget: the decode
    must equal the f64 truth cast once into the dtype, bit for bit.
    """
    rows = cols = 256
    columns = torch.arange(cols).reshape(1, cols)
    data = ((columns + torch.arange(rows).reshape(rows, 1)) % 255 - 127).to(torch.int8)
    exponents = torch.tensor([(r % 60) - 30 for r in range(rows)], dtype=torch.float64)
    scale = torch.pow(2.0, exponents)
    truth = hadamard256(torch, data.double()) * scale.reshape(rows, 1) / 16
    destination = torch.empty(rows, cols, dtype=dtype, device=device)
    provider.decode(
        torch, {"weight": data.to(device), "scale": scale.float().to(device)}, destination
    )
    got = destination.cpu().double()
    finite = torch.isfinite(got)
    error = (got - truth).abs()[finite]
    relative = error / truth.abs()[finite].clamp(min=1e-300)
    exact = bool((destination.cpu() == truth.to(dtype)).all())
    return Sweep(
        patterns=rows * cols,
        max_abs_error=float(error.max()),
        max_rel_error=float(relative.max()),
        tolerance=_eps(torch, dtype),
        nan_preserved=True,
        passed=exact,
    )


def _exact_sweep(torch: ModuleType, got: Tensor, truth: Tensor, dtype: Tensor) -> Sweep:
    """`got` against the f64 `truth` cast once into `dtype`, bit for bit; NaN as NaN."""
    want = truth.to(dtype)
    nan = torch.isnan(truth)
    same = bool((torch.isnan(got) == nan).all()) and bool((got == want)[~nan].all())
    finite = torch.isfinite(got.double()) & torch.isfinite(truth)
    error = (got.double() - truth).abs()[finite]
    relative = error / truth.abs()[finite].clamp(min=1e-300)
    return Sweep(
        patterns=int(truth.numel()),
        max_abs_error=float(error.max()) if error.numel() else 0.0,
        max_rel_error=float(relative.max()) if relative.numel() else 0.0,
        tolerance=_eps(torch, dtype),
        nan_preserved=bool((torch.isnan(got) == nan).all()),
        passed=same,
    )


def _numerics_int8_rowwise(
    torch: ModuleType, provider: Provider, device: Tensor, dtype: Tensor
) -> Sweep:
    """Every int8 value against 256 power-of-two row scales: row `r` holds -128..127 shifted by
    `r`, so a decoder that mis-strides the scale is wrong on every row but one. The product is
    exact in f32, so the decode must equal the f64 truth cast once into the dtype."""
    rows = cols = 256
    columns = torch.arange(cols).reshape(1, cols)
    data = ((columns + torch.arange(rows).reshape(rows, 1)) % 256 - 128).to(torch.int8)
    scale = torch.pow(2.0, torch.tensor([(r % 60) - 30 for r in range(rows)], dtype=torch.float64))
    truth = data.double() * scale.reshape(rows, 1)
    destination = torch.empty(rows, cols, dtype=dtype, device=device)
    weight = _across(provider, data).to(device)
    provider.decode(torch, {"weight": weight, "scale": scale.float().to(device)}, destination)
    return _exact_sweep(torch, _across(provider, destination).cpu(), truth, dtype)


def _across(provider: Provider, rows: Tensor) -> Tensor:
    """A columnwise provider's view of a rowwise fixture: the transpose, its scale per column,
    so one oracle checks both directions (and a decoder that reads the wrong axis fails it)."""
    return rows.t().contiguous() if provider.encoding.endswith(".columnwise") else rows


def _numerics_int8_tensorwise(
    torch: ModuleType, provider: Provider, device: Tensor, dtype: Tensor
) -> Sweep:
    """Every int8 value under one f32 scale: the decode is `round_L(weight * scale)`, the scale
    entering exactly, so it must equal the f64 product cast once into the dtype."""
    data = (torch.arange(256) - 128).to(torch.int8).reshape(16, 16)
    scale = torch.tensor(0.0123, dtype=torch.float32)
    destination = torch.empty(16, 16, dtype=dtype, device=device)
    provider.decode(torch, {"weight": data.to(device), "scale": scale.to(device)}, destination)
    return _exact_sweep(torch, destination.cpu(), data.double() * scale.double(), dtype)


def _comfy_fixture(
    torch: ModuleType, provider: Provider, device: Tensor, shape: tuple[int, int]
) -> tuple[Tensor, dict[str, Tensor]]:
    """A valid constant-q carrier with an independent exact logical reference.

    Each default level becomes q8=1. Every H4 row sums to 2, so its normalized
    Kronecker powers map all-ones to all-ones. The result is each F32 channel
    scale, repeated across its row; no decoder is used to construct the oracle.
    General codes, corrections, F8 metadata and stacks remain the mined vectors'.
    """
    h4 = ((1, 1, 1, -1), (1, 1, -1, 1), (1, -1, 1, 1), (-1, 1, 1, 1))
    assert tuple(sum(row) for row in h4) == (2, 2, 2, 2)
    assert all(
        sum(a * b for a, b in zip(left, right, strict=True)) == (4 if i == j else 0)
        for i, left in enumerate(h4)
        for j, right in enumerate(h4)
    )
    params = dict(provider.params)
    if provider.supports(shape, params):
        raise AssertionError("the Comfy synthetic fixture must fit its bound descriptor")
    rotation = params.get("rotation_size")
    if not isinstance(rotation, int):
        raise AssertionError("rotation_size must be an integer")
    exponent = (rotation.bit_length() - 1) // 2
    assert 4**exponent == rotation and 2**exponent * 2.0**-exponent == 1.0
    rows, columns = shape
    channel = torch.linspace(0.5, 1.5, rows, dtype=torch.float32, device=device)
    truth = channel.reshape(rows, 1).expand(rows, columns)
    if provider.encoding == "int4.convrot":
        # Signed nibble 1 decodes to 1.
        return truth, {
            "weight": torch.full((rows, columns // 2), 0x11, dtype=torch.int8, device=device),
            "scale": channel,
        }
    group = params.get("group_size")
    if not isinstance(group, int) or type(group) is not int:
        raise AssertionError("the Comfy probe must bind its declared group_size")
    if provider.encoding == "comfy.asym_w4a8_int8":
        # Unsigned nibble 9 addresses default level 9 - 8 = 1.
        weight = torch.full((rows, columns // 2), -103, dtype=torch.int8, device=device)
    elif provider.encoding == "comfy.w6a8_int8":
        # Split planes: low nibble1 | (high2 << 4) = 33; level33 - 32 = 1.
        weight = torch.cat(
            (
                torch.full((rows, columns // 2), 0x11, dtype=torch.int8, device=device),
                torch.full((rows, columns // 4), -86, dtype=torch.int8, device=device),
            ),
            dim=1,
        )
    else:
        raise AssertionError(f"no Comfy fixture for {provider.encoding}")
    return truth, {
        "weight": weight,
        "group_scale": torch.ones((rows, columns // group), dtype=torch.float32, device=device),
        "scale": channel,
    }


def _numerics_comfy(torch: ModuleType, provider: Provider, device: Tensor, dtype: Tensor) -> Sweep:
    truth, parts = _comfy_fixture(torch, provider, device, (16, 256))
    destination = torch.empty((16, 256), dtype=dtype, device=device)
    provider.decode(torch, parts, destination)
    return _exact_sweep(torch, destination.cpu(), truth.cpu().double(), dtype)


def _numerics_mxfp8(torch: ModuleType, provider: Provider, device: Tensor, dtype: Tensor) -> Sweep:
    """All 256 E4M3 patterns under all 256 E8M0 bytes (row `r` scales every block by byte `r`):
    65,536 pairs, NaN from either operand included. E4M3 times a power of two is exact in f32,
    so the decode must equal the f64 truth cast once into the dtype."""
    rows = cols = 256
    patterns = torch.frombuffer(bytearray(bytes(range(256)) * rows), dtype=torch.uint8)
    data = patterns.view(torch.float8_e4m3fn).reshape(rows, cols)
    byte = torch.arange(rows, dtype=torch.uint8).reshape(rows, 1).expand(rows, cols // MX_BLOCK)
    scale = byte.contiguous().view(torch.float8_e8m0fnu)
    mult = torch.pow(2.0, torch.arange(rows, dtype=torch.float64) - 127).reshape(rows, 1)
    mult[E8M0_NAN] = float("nan")
    truth = data.double() * mult
    destination = torch.empty(rows, cols, dtype=dtype, device=device)
    provider.decode(torch, {"weight": data.to(device), "scale": scale.to(device)}, destination)
    return _exact_sweep(torch, destination.cpu(), truth, dtype)


def _numerics_nvfp4(torch: ModuleType, provider: Provider, device: Tensor, dtype: Tensor) -> Sweep:
    """Every byte pattern (both nibbles of every E2M1 value) against every finite positive E4M3
    block scale, at a power-of-two second level.

    Row `r` stores all 256 bytes under block scale code `r + 1`, so a decoder that reads the
    nibbles the wrong way round, mis-strides the block scales or drops the second level is wrong
    almost everywhere. An E2M1 value times an E4M3 scale times a power of two is exact in f32, so
    the destination's one RNE cast is the whole budget: bit for bit against the f64 truth.
    """
    rows, cols = 126, 512
    weight = torch.arange(256, dtype=torch.int32).repeat(rows, 1).to(torch.uint8)
    codes = torch.arange(1, rows + 1, dtype=torch.int32).to(torch.uint8)
    weight_scale = codes.reshape(rows, 1).repeat(1, cols // NVFP4_BLOCK).view(torch.float8_e4m3fn)
    scale_2 = torch.tensor(0.125, dtype=torch.float32)
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float64)
    nibbles = torch.stack((weight & 0x0F, weight >> 4), dim=-1).reshape(rows, cols).long()
    blocks = weight_scale.double().repeat_interleave(NVFP4_BLOCK, dim=1)
    truth = lut[nibbles] * blocks * 0.125
    destination = torch.empty(rows, cols, dtype=dtype, device=device)
    provider.decode(
        torch,
        {
            "weight": weight.to(device),
            "scale": weight_scale.to(device),
            "scale_2": scale_2.to(device),
        },
        destination,
    )
    got = destination.cpu().double()
    error = (got - truth).abs()
    relative = error / truth.abs().clamp(min=1e-300)
    exact = bool((destination.cpu() == truth.to(dtype)).all())
    return Sweep(
        patterns=rows * cols,
        max_abs_error=float(error.max()),
        max_rel_error=float(relative.max()),
        tolerance=_eps(torch, dtype),
        nan_preserved=True,
        passed=exact,
    )


def _domain_sweep(
    torch: Any, dtype: Any, got: Sequence[float], scales: Sequence[tuple[float, str]]
) -> Sweep:
    """Grade a decode of all 256 e4m3 elements under each row's `(scale, label)`. `got` is
    row-major, one row per scale.

    Five verdicts per pair, because "correct" is not one question when a product spans a
    wider range than the destination dtype holds:

    * NORMAL — the exact product is a finite normal of the dtype: within two roundings
      RELATIVE.
    * SUBNORMAL — finite, below the dtype's smallest normal, where the ULP is CONSTANT: two
      roundings ABSOLUTE. Relative error would fail a correct decoder here.
    * OVERFLOW / UNDERFLOW — the exact product leaves the dtype's finite range, so the
      answers are a signed infinity and a zero; a plausible finite number would be wrong.
    * NAN — an operand is its format's NaN, and NaN must survive as NaN.
    """
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
    * **a silently wrong kernel.** A mis-strided scale or a transposed operand returns wrong
      numbers without raising. It fails HERE, far past the budget, and mints nothing.

    The budget is deliberately LOOSE — this is a w8a8 route and the activation quantization
    is a real, measured, accepted loss (#517d: 16.72 dB against the dequant lane's 20.82 dB).
    What the bound rejects is not imprecision, it is nonsense: the pass line sits where a
    mis-strided scale or a transposed operand lands, both of which measure worse than 0.5
    relative, and no correct w8a8 route measures worse than 0.15.
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
    if provider.suite == "comfy-decode":
        return _comfy_fixture(torch, provider, device, BENCH_SHAPE)
    rows, cols = BENCH_SHAPE
    generator = torch.Generator(device="cpu").manual_seed(0x511B)
    if isinstance(provider, CoreDequant):
        roles = sample(torch, provider.adapter, BENCH_SHAPE, generator)
        roles = {role: part.to(device) for role, part in roles.items()}
        return _exact_weight(torch, provider, roles), roles
    weight = torch.randn(rows, cols, generator=generator, dtype=torch.float32).to(device) * 0.05
    if provider.suite in ("fp8-rowwise-decode", "fp8-rowwise-native"):
        scale = (weight.abs().amax(dim=-1) / 448.0).clamp(min=1e-12)
        payload = (weight / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
        return weight, {"weight": payload.to(torch.float8_e4m3fn), "scale": scale}
    if provider.suite == "fp8-columnwise-decode":
        scale = (weight.abs().amax(dim=0) / 448.0).clamp(min=1e-12)
        payload = (weight / scale.unsqueeze(0)).clamp(-448.0, 448.0)
        return weight, {"weight": payload.to(torch.float8_e4m3fn), "scale": scale}
    if provider.suite in ("int8-convrot-decode", "int8-convrot256-native"):
        rotated = hadamard256(torch, weight.double()) / 16
        scale = (rotated.abs().amax(dim=-1) / 127.0).clamp(min=1e-12).float()
        payload = torch.round(rotated / scale.double().unsqueeze(-1)).clamp(-127, 127)
        return weight, {"weight": payload.to(torch.int8), "scale": scale}
    if provider.suite in ("nvfp4-decode", "nvfp4-native"):
        encoded = encode_nvfp4("probe", weight.cpu().numpy())
        scale, scale_2 = encoded.companions
        return weight, {
            "weight": torch.frombuffer(bytearray(encoded.payload), dtype=torch.uint8)
            .reshape(rows, cols // 2)
            .to(device),
            "scale": torch.frombuffer(bytearray(scale.raw), dtype=torch.uint8)
            .reshape(rows, cols // NVFP4_BLOCK)
            .view(torch.float8_e4m3fn)
            .to(device),
            "scale_2": torch.frombuffer(bytearray(scale_2.raw), dtype=torch.float32)
            .reshape(())
            .to(device),
        }
    if provider.suite == "int8-rowwise-decode":
        scale = (weight.abs().amax(dim=-1) / 127.0).clamp(min=1e-12)
        payload = torch.round(weight / scale.unsqueeze(-1)).clamp(-127, 127)
        return weight, {"weight": payload.to(torch.int8), "scale": scale}
    if provider.suite == "int8-columnwise-decode":
        scale = (weight.abs().amax(dim=0) / 127.0).clamp(min=1e-12)
        payload = torch.round(weight / scale.unsqueeze(0)).clamp(-127, 127)
        return weight, {"weight": payload.to(torch.int8), "scale": scale}
    if provider.suite in ("mxfp8-decode", "mxfp8-native"):
        blocks = weight.reshape(rows, cols // MX_BLOCK, MX_BLOCK)
        amax = blocks.abs().amax(dim=-1)
        exponent = torch.where(
            amax > 0, torch.ceil(torch.log2(amax / 448.0)), torch.full_like(amax, -127.0)
        )
        byte = (exponent + 127).clamp(0.0, 254.0).to(torch.uint8)
        payload = blocks / e8m0_multipliers(torch, byte).unsqueeze(-1)
        return weight, {
            "weight": payload.reshape(rows, cols).clamp(-448.0, 448.0).to(torch.float8_e4m3fn),
            "scale": byte.view(torch.float8_e8m0fnu),
        }
    if provider.suite == "int8-tensorwise-decode":
        scale = (weight.abs().max() / 127.0).clamp(min=1e-12).to(torch.float32)
        payload = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        return weight, {"weight": payload, "scale": scale.reshape(())}
    if provider.suite == "fp8-tensorwise-decode":
        amax = weight.abs().max().clamp(min=1e-12)
        scale = (amax / 448.0).to(torch.float32)
        payload = (weight / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return weight, {"weight": payload, "scale": scale.reshape(())}
    raise AssertionError(f"no operand builder for suite {provider.suite!r}")


def _exact_weight(torch: Any, provider: Provider, roles: Mapping[str, Any]) -> Any:
    """The EXACT float64 value of the stored roles. The reference both routes are scored on.

    Written here in f64 rather than delegated to the decode provider on purpose: a reference
    that calls one of the two implementations under comparison is that implementation's
    self-check, not a reference. The arithmetic is the spec's and it is exact — an e4m3
    element and a power-of-two or f32 scale multiply without rounding in f64 — so what the
    deviation below measures is entirely the route's own.
    """
    if isinstance(provider, CoreDequant):
        return (
            _cores_reference(torch, provider, roles, BENCH_SHAPE)
            .double()
            .to(next(iter(roles.values())).device)
        )
    if provider.suite == "comfy-decode":
        # Only the constant-q fixture above enters this benchmark/route oracle.
        # Its normalized rotation is identity on all-ones; the independently
        # stored channel scale is the exact value, not another decoder's output.
        rows, packed = (int(n) for n in roles["weight"].shape)
        columns = packed * 4 // 3 if provider.encoding == "comfy.w6a8_int8" else packed * 2
        return _f64(roles["scale"]).reshape(rows, 1).expand(rows, columns)
    if provider.suite in ("nvfp4-decode", "nvfp4-native"):
        packed = roles["weight"].cpu()
        rows, cols = int(packed.shape[0]), 2 * int(packed.shape[1])
        nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(rows, cols).long()
        lut = torch.tensor(E2M1_VALUES, dtype=torch.float64)
        blocks = roles["scale"].cpu().double().repeat_interleave(NVFP4_BLOCK, dim=1)
        exact = lut[nibbles] * blocks * roles["scale_2"].cpu().double()
        return exact.to(roles["weight"].device)
    data = _f64(roles["weight"])
    if provider.suite in ("fp8-rowwise-decode", "fp8-rowwise-native", "int8-rowwise-decode"):
        return data * _f64(roles["scale"]).reshape(int(data.shape[0]), 1)
    if provider.suite in ("fp8-columnwise-decode", "int8-columnwise-decode"):
        return data * _f64(roles["scale"]).reshape(1, int(data.shape[1]))
    if provider.suite in ("mxfp8-decode", "mxfp8-native"):
        mult = _f64(e8m0_multipliers(torch, roles["scale"]))
        return data * mult.repeat_interleave(MX_BLOCK, dim=1)[:, : int(data.shape[1])]
    if provider.suite in ("int8-convrot-decode", "int8-convrot256-native"):
        return hadamard256(torch, data) * _f64(roles["scale"]).reshape(-1, 1) / 16
    return data * _f64(roles["scale"])


def _f64(tensor: Tensor) -> Tensor:
    """`tensor` in float64, where the device has it; MPS has none, so the host's."""
    return (tensor.cpu() if tensor.device.type == "mps" else tensor).double()


def _activation(torch: Any, device: Any, dtype: Any) -> Any:
    """The same seeded activation for every route, so a deviation is about the route."""
    generator = torch.Generator(device="cpu").manual_seed(0x517E)
    x = torch.randn(FIDELITY_TOKENS, BENCH_SHAPE[1], generator=generator, dtype=torch.float32)
    return (x * 0.5).to(device=device, dtype=dtype)


def _reference_gemm(torch: Any, x: Any, weight: Any) -> Any:
    """`x @ W.t()` in f64. f32 would be measured through TF32 on every card that has it,
    which is a ~1e-3 error sitting under numbers this suite reports to 1e-6."""
    return (_f64(x) @ _f64(weight).t()).double()


def _relative(torch: Any, got: Any, reference: Any) -> float:
    delta = (_f64(got) - reference).norm()
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
    resident, priced by the provider from the probe's real tensors."""
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


# ------------------------------------------------------------- the conformance vectors

#: The cozytensors/1 conformance decode cases, one file per scheme family (tensorfs
#: `conformance/vectors/cases/<family>.json`). They ship in the package, so an installed wheel
#: qualifies as the repo does.
VECTORS = Path(__file__).resolve().parent / "encoding" / "vectors" / "cases"


class VectorPart(msgspec.Struct, frozen=True):
    role: str
    dtype: str
    shape: list[int]
    bits: str


class VectorEncoding(msgspec.Struct, frozen=True):
    scheme: str
    params: dict[str, bool | int | float | str | list[int]] = {}


class VectorCase(msgspec.Struct, frozen=True):
    component: str
    key: str
    dtype: str
    shape: list[int]
    encoding: VectorEncoding
    #: in the header's order
    parts: list[VectorPart]
    expect: str = ""
    refuse: str = ""


def vector_cases() -> list[VectorCase]:
    """The conformance cases with a scheme (the plain ones are TensorFS's read cases)."""
    out = []
    for path in sorted(VECTORS.glob("*.json")):
        rows = msgspec.json.decode(path.read_bytes(), type=dict[str, list[dict[str, object]]])
        out += [msgspec.convert(row, VectorCase) for row in rows["cases"] if "encoding" in row]
    return out


def vector_tensor(torch: ModuleType, part: VectorPart, device: Tensor) -> Tensor:
    """A vector part's tensor: base64 little-endian bytes, viewed in its dtype. An `F4` part
    is its packed bytes, u8 `[R, C/2]`, as the fill plane hands it to a decoder."""
    flat = torch.frombuffer(bytearray(base64.b64decode(part.bits)), dtype=torch.uint8)
    if part.dtype == "F4":
        return flat.reshape(*part.shape[:-1], part.shape[-1] // 2).to(device)
    name = TORCH_DTYPES[TENSORFS_DTYPES[part.dtype]]
    return flat.view(getattr(torch, name)).reshape(part.shape).to(device)


def decode_case(torch: ModuleType, provider: Provider, case: VectorCase, device: Tensor) -> Tensor:
    """`case` decoded by `provider` bound to its params: the logical tensor, or None when the
    provider refuses the case's shape or params."""
    bound = replace(provider, params=params_of(case.encoding.params))  # type: ignore[type-var]
    if bound.supports(case.shape, case.encoding.params):
        return None
    roles = {part.role: vector_tensor(torch, part, device) for part in case.parts}
    dtype = getattr(torch, TORCH_DTYPES[TENSORFS_DTYPES[case.dtype]])
    destination = torch.empty(*case.shape, dtype=dtype, device=device)
    bound.decode(torch, roles, destination)
    return destination


def same_bits(torch: ModuleType, got: Tensor, expect: str) -> int:
    """Differing elements of `got` against the expected bytes; NaN equals NaN (the spec
    canonicalizes NaN bits, a decode need not)."""
    want = torch.frombuffer(bytearray(base64.b64decode(expect)), dtype=torch.uint8)
    want = want.view(got.dtype).reshape(got.shape)
    got = got.cpu()
    differ = got.view(torch.uint8).reshape(*got.shape, -1) != want.view(torch.uint8).reshape(
        *got.shape, -1
    )
    differ = differ.any(dim=-1) & ~(got.isnan() & want.isnan())
    return int(differ.sum())


def _spec_vectors(torch: ModuleType, provider: Provider, device: Tensor) -> SpecVectors:
    """The conformance cases of the provider's schemes, decoded bit for bit (NaN as NaN).

    The synthetic sweeps prove the arithmetic; these prove the provider reads the layout a real
    producer emits. A case whose shape or params the provider refuses is skipped; a case whose
    parts the provider does not account for fails."""
    cases: list[VectorCheck] = []
    for case in vector_cases():
        if case.encoding.scheme != provider.encoding or case.refuse:
            continue
        roles = {part.role for part in case.parts}
        missing = sorted(provider.required - roles)
        surplus = sorted(roles - provider.required - provider.optional)
        if surplus and not missing:  # the selector never hands this provider such a tensor
            cases.append(
                VectorCheck(case.key, passed=True, skipped=True, why=f"takes no {surplus}")
            )
            continue
        if missing or surplus:
            why = f"parts {sorted(roles)}: missing {missing}, unhandled {surplus}"
            cases.append(VectorCheck(case.key, passed=False, why=why))
            continue
        got = decode_case(torch, provider, case, device)
        if got is None:
            cases.append(VectorCheck(case.key, passed=True, skipped=True, why="refused"))
            continue
        differing = same_bits(torch, got, case.expect)
        cases.append(
            VectorCheck(
                case.key,
                passed=differing == 0,
                elements=int(got.numel()),
                differing_bytes=differing,
            )
        )
    if not cases:
        return SpecVectors(passed=True, skipped=True, note="no conformance case for these schemes")
    return SpecVectors(passed=all(c.passed for c in cases), cases=cases)


def _micro_bench(torch: Any, provider: Provider, device: Any, dtype: Any) -> DecodeBench:
    rows, cols = BENCH_SHAPE
    destination = torch.empty(rows, cols, dtype=dtype, device=device)
    _, parts = _weight_roles(torch, provider, device)
    provider.decode(torch, parts, destination)
    # The fence goes through the accelerator boundary (#447): a queue timed without its own
    # synchronize would mint a fictional decode rate.
    accel.synchronize(torch, device.type)
    started = time.perf_counter_ns()
    for _ in range(BENCH_REPS):
        provider.decode(torch, parts, destination)
    accel.synchronize(torch, device.type)
    took = (time.perf_counter_ns() - started) / BENCH_REPS
    # Bytes the decode actually touches: every stored role IN, the logical tensor OUT. Summed
    # over the parts rather than assuming one byte per element, because a scale role is real
    # traffic and a rate that ignores it flatters this route.
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


def _cores_reference(
    torch: ModuleType, provider: CoreDequant, roles: Mapping[str, Tensor], shape: tuple[int, ...]
) -> Tensor:
    """The spec's f32 value of `roles` by the cores' eager reference, on the host: a second
    implementation of the arithmetic the Triton kernels run."""
    reference = torch.empty(*shape, dtype=torch.float32)
    host = {role: part.cpu() for role, part in roles.items()}
    run(provider.adapter, Cores(torch, kernel=False), host, dict(provider.params), reference)
    return reference


def _numerics_cores(torch: ModuleType, provider: Provider, device: Tensor, dtype: Tensor) -> Sweep:
    """The row's random parts at its probe params (every byte pattern), decoded on the device
    and by the eager reference: bit for bit after the one cast."""
    assert isinstance(provider, CoreDequant)
    provider = replace(provider, params=params_of(provider.adapter.probe_params))
    shape = (384, 512)
    roles = sample(
        torch, provider.adapter, shape, torch.Generator(device="cpu").manual_seed(0x0C0E)
    )
    destination = torch.empty(*shape, dtype=dtype, device=device)
    provider.decode(torch, {role: part.to(device) for role, part in roles.items()}, destination)
    truth = _cores_reference(torch, provider, roles, shape).double()
    return _exact_sweep(torch, destination.cpu(), truth, dtype)


#: Which self-check a provider gets, keyed by the ORACLE IT DECLARES (`Provider.suite`) and
#: never by class or role set, so a second implementation of one encoding is measured by the
#: same oracle as the first. A suite name in neither table qualifies NOTHING: minting a record
#: for a decoder no oracle checked is exactly the fabricated evidence the table forbids.
DECODE_SUITES: dict[str, Callable[[Any, Provider, Any, Any], Sweep]] = {
    "comfy-decode": _numerics_comfy,
    "fp8-tensorwise-decode": _numerics,
    "fp8-rowwise-decode": _numerics_rowwise,
    "fp8-columnwise-decode": _numerics_rowwise,
    "int8-convrot-decode": _numerics_int8_convrot,
    "nvfp4-decode": _numerics_nvfp4,
    "int8-rowwise-decode": _numerics_int8_rowwise,
    "int8-columnwise-decode": _numerics_int8_rowwise,
    "int8-tensorwise-decode": _numerics_int8_tensorwise,
    "mxfp8-decode": _numerics_mxfp8,
    "cores-decode": _numerics_cores,
}

#: Native leaves' oracle is their own GEMM against arithmetic (`_native_agreement`).
NATIVE_SUITES = frozenset(
    {"fp8-rowwise-native", "int8-convrot256-native", "nvfp4-native", "mxfp8-native"}
)


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
    target = torch.device(device.kind)
    started = time.perf_counter_ns()
    observations = [
        make_observation(
            encoding,
            provider,
            device,
            name,
            settled(provider, device) or _measured(torch, provider, target, getattr(torch, name)),
        )
        for encoding, provider, name in cells(providers, output_dtypes)
    ]
    result = QualificationResult(
        tuple_key=device.tuple_key(),
        device=device,
        device_predicate=device.predicate,
        runtime_release=release,
        runtime=runtime,
        suite_ms=(time.perf_counter_ns() - started) // 1_000_000,
        observations=observations,
    )
    return admission_table(result), result


#: The params a suite's synthetic operands are written with: the producers' own (ModelOpt's
#: layouts). The spec's conformance cases cover the other layouts.
PROBE_PARAMS: dict[str, dict[str, object]] = {
    "comfy.asym_w4a8_int8": {"group_size": 16, "rotation": "kron_regular_h4", "rotation_size": 256},
    "comfy.w6a8_int8": {"group_size": 32, "rotation": "kron_regular_h4", "rotation_size": 256},
    "int4.convrot": {"rotation": "kron_regular_h4", "rotation_size": 256},
    "fp8.tensorwise": {},
    "int8.tensorwise": {},
    "fp8.rowwise": {},
    "int8.rowwise": {},
    "fp8.columnwise": {},
    "int8.columnwise": {},
    "int8.convrot": {"rotation": "kron_regular_h4", "rotation_size": 256},
    "mxfp8": {"block_size": MX_BLOCK, "scale_layout": "row_major"},
    "nvfp4": {
        "block_size": NVFP4_BLOCK,
        "global_scale": "multiply",
        "nibble_order": "low_first",
        "scale_layout": "row_major",
    },
}


def _measured(torch: ModuleType, provider: Provider, target: Tensor, dtype: Tensor) -> Evidence:
    """`_evidence`, where a device that lacks a dtype the decode reads (MPS has no float8)
    mints no record for it: the encoding refuses at bind naming why."""
    # Qualification considers every provider even for a plain F64/FP8 model.
    # Those are valid verbatim outputs, not logical Comfy decode destinations.
    if provider.suite == "comfy-decode" and dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        return UnmeasuredEvidence(note="Comfy logical outputs are F32, F16 or BF16")
    if isinstance(provider, CoreDequant):
        params = params_of(provider.adapter.probe_params)
    else:
        params = params_of(PROBE_PARAMS.get(provider.encoding, {}))
    provider = replace(provider, params=params)  # type: ignore[type-var]
    try:
        return _evidence(torch, provider, target, dtype)
    except (TypeError, RuntimeError, NotImplementedError, UnsupportedError) as exc:
        if target.type == "cuda":
            raise
        return UnmeasuredEvidence(note=f"{type(exc).__name__}: {exc}"[:300])


def _evidence(torch: Any, provider: Provider, target: Any, dtype: Any) -> Evidence:
    """One measured (implementation, output dtype)'s evidence. Only a PASSED one mints a record."""
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


def settled(provider: Provider, device: DeviceFacts) -> Evidence | None:
    """Evidence nothing has to measure: a verbatim route, or a card below the provider's floor."""
    if provider.route == "verbatim":
        return VerbatimEvidence(
            note="no decode: the stored bytes ARE the logical bytes, and the fill plane's "
            "source-comparison proves them byte for byte"
        )
    if not device.satisfies(provider.device_predicate):
        # Qualifying a leaf on a card whose kernel it does not claim would mint a record whose
        # predicate says this card.
        return UnmeasuredEvidence(
            note=f"its kernel needs {provider.device_predicate} and this device is "
            f"{device.predicate}"
        )
    return None


def cells(
    providers: Mapping[str, tuple[Provider, ...]], output_dtypes: Sequence[str]
) -> Iterator[tuple[str, Provider, str]]:
    """Every (encoding, implementation, output dtype) the suite answers. A decoded or native
    route has nothing to say about an integer destination."""
    for encoding in sorted(providers):
        for provider in providers[encoding]:
            for name in output_dtypes:
                if provider.route == "verbatim" or name.startswith(("float", "bfloat")):
                    yield encoding, provider, name


def make_observation(
    encoding: str, provider: Provider, device: DeviceFacts, dtype: str, evidence: Evidence
) -> Observation:
    return Observation(
        scheme=encoding,
        implementation=provider.name,
        implementation_digest=implementation_digest(provider),
        implementation_revision=provider.code_revision,
        device_predicate=device.predicate,
        output_dtype=dtype,
        geometry_class=geometry_class(BENCH_SHAPE),
        delivery_route=provider.route,
        evidence=evidence,
    )


def admission_table(result: QualificationResult) -> Capabilities:
    """Observations -> the admission table. A FAILED observation mints no record; its reason
    is kept, so a tensor served by another route can say why."""
    return Capabilities(
        [
            CapabilityRecord(
                scheme=observation.scheme,
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
        ],
        {
            (o.scheme, o.implementation_digest, o.output_dtype): _refusal(o.evidence)
            for o in result.observations
            if not o.evidence.passed
        },
    )


def _refusal(evidence: Evidence) -> str:
    """Why a failed observation minted no record, in one clause."""
    match evidence:
        case NativeEvidence(numerics=numerics):
            return numerics.kernel or (
                f"its GEMM measured {numerics.deviation_rel:.3g} relative error against the "
                f"{numerics.tolerance} bound (finite {numerics.finite}, out {numerics.out_dtype})"
            )
        case DecodeEvidence():
            return "its decode failed the format oracle or the spec's vectors"
        case _:
            return evidence.note


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


class UnreportableObservation(Exception):
    """A record would have crossed the process boundary carrying a byte TensorFS refuses.

    Loud rather than dropped: a silently omitted record reads to the hub as "this card
    cannot run it", which is the wrong answer with the wrong remedy.
    """
