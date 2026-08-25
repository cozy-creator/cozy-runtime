"""cr-006 — the per-device qualification suite that MINTS capability records.

§3.2's cost placement, made concrete: this runs ONCE per (accelerator model, device
configuration, driver, runtime build, encoding rule) tuple, at prepare time, and its
result is a persisted typed fact. A worker landing on a known tuple skips the suite
entirely; a local rig probes once and persists. Per request there is only arithmetic.

Three things happen per (encoding, output dtype), in this order, and a failure at any
one leaves NO record — which is what "fail-closed" means when the table is minted rather
than declared:

1. **numerics self-check** against an INDEPENDENT reference. The e4m3 decoder below is
   written from the OCP number format by integer arithmetic, not by asking torch to cast
   — a self-check that uses the implementation as its own oracle checks nothing. Every
   one of the 256 e4m3 bit patterns is decoded, NaN included (as a NaN, not a number).
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

from cozy_runtime.internal.encoding import (
    Capabilities,
    CapabilityRecord,
    DeviceFacts,
    canonical_digest,
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


def _eps(torch: Any, dtype: Any) -> float:
    return float(torch.finfo(dtype).eps)


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


def _micro_bench(torch: Any, provider: Any, device: Any, dtype: Any) -> dict[str, Any]:
    rows, cols = BENCH_SHAPE
    data = (
        torch.randint(0, 240, (rows, cols), dtype=torch.uint8, device=device)
        .view(torch.float8_e4m3fn)
        .reshape(rows, cols)
    )
    weight_scale = torch.tensor(0.01, dtype=torch.float32, device=device)
    destination = torch.empty(rows, cols, dtype=dtype, device=device)
    parts = {"data": data, "weight_scale": weight_scale}
    provider.decode(torch, parts, destination)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter_ns()
    for _ in range(BENCH_REPS):
        provider.decode(torch, parts, destination)
    if device.type == "cuda":
        torch.cuda.synchronize()
    took = (time.perf_counter_ns() - started) / BENCH_REPS
    moved = rows * cols * (1 + destination.element_size())
    return {
        "shape": list(BENCH_SHAPE),
        "reps": BENCH_REPS,
        "decode_ns": int(took),
        "gb_per_s": round(moved / took, 3),
    }


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
    """
    key = canonical_digest(
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
                evidence = {"kind": "decode"}
                evidence["numerics"] = _numerics(torch, provider, target, dtype)
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
                evidence_digest=canonical_digest(observation["evidence"]),
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
