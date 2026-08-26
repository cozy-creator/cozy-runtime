"""cr-006 — the per-device qualification suite that MINTS capability records.

§3.2's cost placement, made concrete: this runs ONCE per (accelerator model, device
configuration, driver, runtime build, encoding rule) tuple, at prepare time, and its
result is a persisted typed fact. A worker landing on a known tuple skips the suite
entirely; a local rig probes once and persists. Per request there is only arithmetic.

Three things happen per (encoding, output dtype), in this order, and a failure at any
one leaves NO record — which is what "fail-closed" means when the table is minted rather
than declared:

1. **numerics self-check** against an INDEPENDENT reference. The e4m3 and e8m0 decoders
   below are written from the OCP number formats by integer arithmetic, not by asking
   torch to cast — a self-check that uses the implementation as its own oracle checks
   nothing. The domain is the WHOLE format: all 256 e4m3 bit patterns for a per-tensor
   scale, and all 65,536 (e4m3 element, e8m0 block scale) pairs for a microscaled one,
   NaN included (as a NaN, not a number) and overflow included (as an infinity, not a
   plausible finite number). The suite a provider gets is keyed on the ROLE SET it
   declares, so an encoding with no oracle qualifies nothing rather than passing by
   default.
2. **micro-benchmark** at a realistic tensor size, so the plan table has a measured
   decode rate rather than an assumed one.
3. **record** — the tfs-013 key with the evidence digest of exactly this observation.

The suite QUALIFIES a fixed kernel set; it does not search one. Its seconds are an
acceptance fact and are returned so the caller can bank them (cr-006's checklist): any
growth toward per-shape search is out of scope by construction, because there is no
shape loop here to grow.

First-writer-wins persistence (the tcg#84 pattern): the cache file is created with
`O_EXCL`, and a loser reads the winner's bytes rather than overwriting them.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cozy_runtime.internal import accel, canonical
from cozy_runtime.internal.encoding import (
    Capabilities,
    CapabilityRecord,
    DeviceFacts,
    implementation_digest,
)

#: One decode of this many elements is the micro-benchmark's unit — a realistic attention
#: projection, not a toy and not a search space.
BENCH_SHAPE = (2048, 2048)
BENCH_REPS = 8


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


def _numerics_mx(torch: Any, provider: Any, device: Any, dtype: Any) -> dict[str, Any]:
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

    info = torch.finfo(dtype)
    finite_max = float(info.max)
    smallest_normal = float(info.tiny)
    ulp_sub = float(getattr(info, "smallest_normal", info.tiny)) * _eps(torch, dtype)
    rel_bound = 2.0 * _eps(torch, dtype)
    abs_bound = 2.0 * ulp_sub
    elements = [e4m3_value(b) for b in range(256)]
    scales = [e8m0_value(b) for b in range(256)]
    bands = ("normal", "subnormal", "overflow", "underflow", "nan")
    counts = dict.fromkeys(bands, 0)
    wrong = dict.fromkeys(bands, 0)
    worst_rel = worst_abs = 0.0
    first_bad = ""

    def note(band: str, why: str) -> None:
        nonlocal first_bad
        wrong[band] += 1
        first_bad = first_bad or why

    for r in range(rows):
        s = scales[r]
        for c in range(cols):
            want = elements[c] * s
            have = got[r * cols + c]
            where = f"element {c:#04x} x scale {r:#04x}"
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
                worst_abs = max(worst_abs, error)
                if error > abs_bound:
                    note(
                        "subnormal",
                        f"{where}: {have} != {want} (abs {error:.3e} > {abs_bound:.3e})",
                    )
            else:
                counts["normal"] += 1
                error = abs(have - want)
                worst_abs = max(worst_abs, error)
                relative = error / abs(want)
                worst_rel = max(worst_rel, relative)
                if relative > rel_bound:
                    note(
                        "normal",
                        f"{where}: {have} != {want} (rel {relative:.3e} > {rel_bound:.3e})",
                    )
    return {
        "pairs": rows * cols,
        "domain": counts,
        "max_abs_error": worst_abs,
        "max_rel_error": worst_rel,
        "tolerance": rel_bound,
        "subnormal_tolerance": abs_bound,
        "nan_preserved": wrong["nan"] == 0,
        "wrong": wrong,
        "first_wrong": first_bad,
        "passed": not any(wrong.values()),
    }


def _numerics(torch: Any, provider: Any, device: Any, dtype: Any) -> dict[str, Any]:
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

    worst_rel, worst_abs, nan_ok = 0.0, 0.0, True
    for index, byte in enumerate(patterns):
        want = e4m3_value(byte) * scale_value
        have = got[index]
        if math.isnan(want):
            nan_ok = nan_ok and math.isnan(have)
            continue
        error = abs(have - want)
        worst_abs = max(worst_abs, error)
        if want != 0.0:
            worst_rel = max(worst_rel, error / abs(want))
    bound = 2.0 * _eps(torch, dtype)
    return {
        "patterns": 256,
        "max_abs_error": worst_abs,
        "max_rel_error": worst_rel,
        "tolerance": bound,
        "nan_preserved": nan_ok,
        "passed": bool(worst_rel <= bound and nan_ok),
    }


#: Which self-check a provider gets, keyed by the ROLE SET it declares. Keyed on the
#: declaration and never on the class, so a second implementation of one encoding is
#: measured by the same oracle as the first — which is the only way two implementations'
#: records mean the same thing. A role set with no suite qualifies NOTHING (see `qualify`):
#: minting a record for a decoder no oracle checked is exactly the fabricated evidence the
#: capability table exists to forbid.
SUITES: dict[frozenset[str], Any] = {}


def _bench_parts(torch: Any, provider: Any, device: Any) -> dict[str, Any]:
    """A realistic role set at `BENCH_SHAPE`, built from the provider's own declaration."""
    rows, cols = BENCH_SHAPE
    data = (
        torch.randint(0, 240, (rows, cols), dtype=torch.uint8, device=device)
        .view(torch.float8_e4m3fn)
        .reshape(rows, cols)
    )
    if "scale" in provider.required:
        return {
            "data": data,
            # 0..254: the producing quantizer's own clamp, so the bench measures the decode
            # and not a tensor of NaN blocks.
            "scale": torch.randint(0, 255, (rows, cols // 32), dtype=torch.uint8, device=device),
        }
    return {"data": data, "weight_scale": torch.tensor(0.01, dtype=torch.float32, device=device)}


def _micro_bench(torch: Any, provider: Any, device: Any, dtype: Any) -> dict[str, Any]:
    rows, cols = BENCH_SHAPE
    destination = torch.empty(rows, cols, dtype=dtype, device=device)
    parts = _bench_parts(torch, provider, device)
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
    return {
        "shape": list(BENCH_SHAPE),
        "reps": BENCH_REPS,
        "decode_ns": int(took),
        "gb_per_s": round(moved / took, 3),
    }


#: Populated once, after both suites exist. The key is a provider's declared `required`
#: role set; a provider whose role set is absent qualifies nothing.
SUITES[frozenset({"data", "weight_scale"})] = _numerics
SUITES[frozenset({"data", "scale"})] = _numerics_mx


def qualify(
    torch: Any,
    providers: Mapping[str, Any],
    device: DeviceFacts,
    output_dtypes: Sequence[str],
    *,
    release: str,
    cache: Path | str | None = None,
) -> tuple[Capabilities, dict[str, Any]]:
    """Run the suite (or read the persisted fact) and return the admission table.

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
    """
    accel.qualification_gate(device.kind)
    key = canonical.digest(
        {
            "tuple": device.tuple_key(),
            "release": release,
            "encodings": sorted(providers),
            "dtypes": sorted(output_dtypes),
        }
    )
    path = Path(cache) / f"{key.removeprefix('sha256:')[:32]}.json" if cache else None
    cached = _read(path)
    if cached is not None:
        return _table(cached), cached

    started = time.perf_counter_ns()
    observations: list[dict[str, Any]] = []
    for encoding, provider in sorted(providers.items()):
        for name in output_dtypes:
            dtype = getattr(torch, name)
            target = torch.device("cuda" if device.kind == "cuda" else "cpu")
            if provider.route == "verbatim":
                evidence: dict[str, Any] = {
                    "kind": "verbatim",
                    "note": "no decode: the stored bytes ARE the logical bytes, and the "
                    "fill plane's source-comparison proves them byte for byte",
                    "passed": True,
                }
            elif not dtype.is_floating_point:
                # A decoded-float route has nothing to say about an integer destination.
                # DERIVED from the dtype rather than listed: the registry's fp8 specs
                # declare float logical dtypes, and a checkpoint's index buffers are
                # plain/1 anyway — so this skip removes a record that could never be
                # selected, it does not hide one that could.
                continue
            else:
                suite = SUITES.get(frozenset(provider.required))
                if suite is None:
                    # A decoder no oracle covers. NOT an error and NOT a pass: it mints no
                    # record, so the encoding refuses at bind naming this device — the same
                    # answer an unqualified card gets, for the same reason.
                    observations.append(
                        {
                            "encoding_spec_id": encoding,
                            "implementation": provider.name,
                            "implementation_digest": implementation_digest(provider),
                            "device_predicate": device.predicate,
                            "output_dtype": name,
                            "delivery_route": provider.route,
                            "evidence": {
                                "kind": "unmeasured",
                                "note": f"no numerics oracle for role set "
                                f"{sorted(provider.required)}; an unchecked decoder is not "
                                "a qualified one",
                                "passed": False,
                            },
                        }
                    )
                    continue
                evidence = {"kind": "decode"}
                evidence["numerics"] = suite(torch, provider, target, dtype)
                evidence["micro_bench"] = _micro_bench(torch, provider, target, dtype)
                evidence["passed"] = bool(evidence["numerics"]["passed"])
            observations.append(
                {
                    "encoding_spec_id": encoding,
                    "implementation": provider.name,
                    "implementation_digest": implementation_digest(provider),
                    "device_predicate": device.predicate,
                    "output_dtype": name,
                    "delivery_route": provider.route,
                    "evidence": evidence,
                }
            )
    result = {
        "format": "cozy.runtime.probe/1",
        "tuple": device.tuple_key(),
        "device_predicate": device.predicate,
        "runtime_release": release,
        "suite_ms": (time.perf_counter_ns() - started) // 1_000_000,
        "observations": observations,
    }
    _write(path, result)
    return _table(result), result


def _table(result: Mapping[str, Any]) -> Capabilities:
    """Observations -> the admission table. A FAILED observation mints no record."""
    caps = Capabilities()
    for observation in result["observations"]:
        if not observation["evidence"]["passed"]:
            continue
        caps.add(
            CapabilityRecord(
                encoding_spec_id=observation["encoding_spec_id"],
                implementation_digest=observation["implementation_digest"],
                runtime_release=str(result["runtime_release"]),
                device_predicate=observation["device_predicate"],
                output_dtype=observation["output_dtype"],
                delivery_route=observation["delivery_route"],
                evidence_digest=canonical.digest(observation["evidence"]),
            )
        )
    return caps


def _read(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("format") == "cozy.runtime.probe/1" else None


def _write(path: Path | None, result: Mapping[str, Any]) -> None:
    """FIRST WRITER WINS. A loser leaves the winner's bytes alone and moves on."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(result, sort_keys=True, indent=1).encode()
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return
    with os.fdopen(fd, "wb") as handle:
        handle.write(body)
