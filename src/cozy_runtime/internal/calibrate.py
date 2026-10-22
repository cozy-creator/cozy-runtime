"""The development oracle's GEMM calibration (`cli.qualify`); no consumer imports it.

For each 8-bit format this card's row runs, every legal candidate of its family runs at the
shipped packages' GEMM shapes (`SHAPES`, written by `scripts/gemm-census.py`): each tile of the
family's table, the compiled family default, cuBLASLt, and the GEMV for a few int8 rows.

- **Numerics first.** A candidate's timing counts only after its bf16 and fp16 outputs match
  the exact float64 product of the same quantized operands. The budget is twice that output
  dtype's own rounding error of the exact product, at every timed shape. A non-finite output
  fails before any ratio.
- **Timing.** Each number is a CUDA-event median of `REPS` launches queued behind a sleep, so the
  host never lets the device idle between them. Cold (the first launch at a shape) is reported
  apart from warm. Weights and activations rotate through a pool over twice the L2.
- **Plausibility.** A rate above the card's 8-bit tensor peak or its DRAM peak is refused as
  impossible.
- **Rules.** A workload rule (`verify.Rule`) ships where one candidate beats the family default
  over a (shape class, M bucket) cell by more than the measured spread.
"""

from __future__ import annotations

import math
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import msgspec

from cozy_runtime import _kernels
from cozy_runtime.internal import arch, prebuilt, verify
from cozy_runtime.internal.encoding.formats import Tensor, hadamard256

#: The packages' GEMMs: one row per (model, lane, M, N, K) with its layer count.
SHAPES = Path(__file__).parent / "qualification_tables" / "gemm-shapes.json"
#: Warm launches per candidate and shape: the median's standard error is then under a quarter
#: of one launch's spread.
REPS = 32
#: One reference row per this many rows, plus the last: every M tile of every family's table
#: (the smallest is 16 rows, the widest 256) is sampled at its edge and inside large ones.
ROW_STEP = 128
#: Float64 weight bytes per reference chunk, so the exact product fits beside the pools on a
#: 24 GB card.
REFERENCE_BYTES = 1 << 28
#: Dense 8-bit tensor-core operations per clock per SM, from NVIDIA's architecture whitepapers:
#: the ceiling no measured rate may exceed.
OPS_PER_CLOCK = {80: 4096, 86: 2048, 89: 2048, 90: 8192, 100: 16384, 103: 16384, 120: 2048}
#: Twice the widest tile any family's table holds (256): every tile takes it whole.
TILE_PROBE = 512
DEFAULT = "default"
#: Out dtypes the lanes write (SDXL fp16, the others bf16); timing uses the first.
DTYPES = ("bfloat16", "float16")


class Shape(msgspec.Struct, frozen=True):
    model: str
    lane: str
    format: str
    m: int
    n: int
    k: int
    layers: int


class Candidate(msgspec.Struct):
    choice: int | str
    ok: bool
    #: out dtype -> relative L2 against the exact product, and its budget
    rel_l2: dict[str, float] = {}
    budget: dict[str, float] = {}
    #: the first launch at this shape: device and host microseconds
    cold_us: float = 0.0
    host_first_us: float = 0.0
    #: warm device microseconds: min, p25, median, p75, max; and every launch, in order
    warm_us: tuple[float, ...] = ()
    samples_us: list[float] = []
    tops: float = 0.0
    dram_gbps: float = 0.0
    #: a warm rate above the card's peak: a measurement the numeric check already passed but
    #: no rule may use (operands found cached, or a timer fault)
    impossible: bool = False
    note: str = ""


class Measured(msgspec.Struct):
    format: str
    m: int
    n: int
    k: int
    shape: str
    models: list[str]
    pool: int
    #: the activation quantizer against its law: largest code difference, codes that differ
    quantize_max_step: float
    quantize_mismatched: int
    quantize_us: float
    bf16_us: float
    candidates: list[Candidate]


@dataclass(frozen=True, slots=True)
class Card:
    clock_hz: float
    l2_bytes: int
    peak_ops: float
    peak_bytes_per_s: float

    def facts(self) -> dict[str, float | int]:
        return {
            "clock_hz": self.clock_hz,
            "l2_bytes": self.l2_bytes,
            "peak_8bit_tops": self.peak_ops / 1e12,
            "peak_dram_gbps": self.peak_bytes_per_s / 1e9,
        }


@dataclass(frozen=True, slots=True)
class Calibration:
    verdict: verify.Verdict
    rules: tuple[verify.Rule, ...]
    evidence: list[Measured]


def card(torch: ModuleType, index: int, sm: int) -> Card:
    props = torch.cuda.get_device_properties(index)
    clock = props.clock_rate * 1e3
    return Card(
        clock_hz=clock,
        l2_bytes=props.L2_cache_size,
        peak_ops=OPS_PER_CLOCK[sm] * props.multi_processor_count * clock,
        # Double data rate over the bus width (CUDA's deviceQuery arithmetic).
        peak_bytes_per_s=2 * props.memory_clock_rate * 1e3 * props.memory_bus_width / 8,
    )


def clocks(index: int) -> str:
    """The card's clocks, power and throttle state, as nvidia-smi reports them now."""
    query = (
        "name,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,power.limit,power.max_limit,"
        "power.draw,temperature.gpu,clocks_event_reasons.active"
    )
    done = subprocess.run(
        ["nvidia-smi", "-i", str(index), f"--query-gpu={query}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return done.stdout.strip() or done.stderr.strip()


def run(
    torch: ModuleType,
    ops: prebuilt.Prebuilt,
    sm: int,
    progress: Callable[[Measured], object] = lambda measured: None,
) -> dict[str, Calibration]:
    """Every format of this card's row, calibrated at its packages' shapes; `progress` hears
    each shape as it is measured."""
    shapes = msgspec.json.decode(SHAPES.read_bytes(), type=list[Shape])
    measured = card(torch, ops.index, sm)
    found = {}
    for f in (prebuilt.INT8, prebuilt.FP8):
        if not getattr(arch.row(sm), f.name):
            continue
        cells: dict[tuple[int, int, int], list[str]] = defaultdict(list)
        for one in shapes:
            if one.format == f.name:
                cells[one.m, one.n, one.k].append(f"{one.model}/{one.lane} x{one.layers}")
        evidence = []
        for (m, n, k), models in sorted(cells.items()):
            evidence.append(_shape(torch, ops, f, measured, m, n, k, models))
            progress(evidence[-1])
        found[f.name] = Calibration(_verdict(evidence), _rules(evidence), evidence)
    return found


def _tiles(ops: prebuilt.Prebuilt, f: prebuilt._Format) -> list[int]:
    """The family table's tiles: configs until the first the GEMM refuses, at a shape of full
    tiles for every table (`TILE_PROBE`)."""
    torch, device = ops.torch, ops.torch.device("cuda", ops.index)
    if f.fp8 and not ops.cutlass_fp8:
        return []
    x = torch.randn(TILE_PROBE, TILE_PROBE, device=device, dtype=torch.bfloat16)
    q, scales = (ops.fp8_quantize if f.fp8 else ops.int8_quantize)(x)
    want = torch.float8_e4m3fn if f.fp8 else torch.int8
    data = torch.zeros(TILE_PROBE, TILE_PROBE, device=device).to(want)
    scale = torch.ones(TILE_PROBE, device=device)
    tiles: list[int] = []
    while True:
        try:
            ops._gemm(f, q, scales, data, scale, None, torch.bfloat16, len(tiles))
        except ValueError:  # past the table's end
            return tiles
        except Exception:  # in the table but failing here: its candidate records why
            pass
        tiles.append(len(tiles))


@dataclass(slots=True)
class _Set:
    x: Tensor
    q: Tensor
    scales: Tensor
    data: Tensor
    scale: Tensor
    bias: Tensor


def _pool(
    torch: ModuleType, ops: prebuilt.Prebuilt, f: prebuilt._Format, card: Card, m: int, n: int,
    k: int,
) -> list[_Set]:  # fmt: skip
    """Distinct operand sets totalling over twice the L2 (at least two), within a quarter of the
    free memory: no launch finds its operands cached by the previous one."""
    each = n * k + 6 * n + 3 * m * k + 4 * m
    free, _ = torch.cuda.mem_get_info(ops.index)
    count = min(max(2, math.ceil(2 * card.l2_bytes / each)), max(2, free // 4 // each))
    device = torch.device("cuda", ops.index)
    sets = []
    for _ in range(count):
        x = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        if f.fp8:
            data, scale = prebuilt._fp8_law(torch, torch.randn(n, k, device=device) * 0.02)
            scale = scale.view(-1)
        else:
            data = torch.randint(-127, 128, (n, k), device=device, dtype=torch.int8)
            scale = torch.rand(n, device=device).add_(0.5).mul_(1e-3)
        q, scales = (ops.fp8_quantize if f.fp8 else ops.int8_quantize)(x)
        bias = torch.randn(n, device=device).to(torch.bfloat16)
        sets.append(_Set(x, q, scales, data, scale, bias))
    return sets


def _exact(torch: ModuleType, one: _Set, rows: Tensor) -> Tensor:
    """The exact product of the quantized operands at `rows`, in float64 chunks of columns."""
    q = one.q[rows].double()
    width = max(1, REFERENCE_BYTES // (8 * one.data.shape[1]))
    parts = [
        q @ one.data[start : start + width].double().t()
        for start in range(0, one.data.shape[0], width)
    ]
    product = torch.cat(parts, dim=1) * one.scales[rows].double().view(-1, 1)
    return product * one.scale.double().view(1, -1) + one.bias.double().view(1, -1)


def _quantize_law(
    torch: ModuleType, f: prebuilt._Format, one: _Set, rows: Tensor
) -> tuple[float, int]:
    """The native quantizer's codes against its law at `rows`: the largest difference in
    steps (one is a tie rounded the other way; an E4M3 step is one ulp of the law's value)
    and how many codes differ."""
    x = one.x[rows]
    if f.fp8:
        law, _ = prebuilt._fp8_law(torch, x)
        want, got = law.float(), one.q[rows].float()
        exponent = torch.floor(torch.log2(want.abs().clamp(min=2.0**-6)))
        steps = (got - want).abs() / torch.exp2(exponent - 3)
    else:
        rotated = hadamard256(torch, x.double()) / 16
        row = (rotated.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-12)
        want = torch.round(rotated / row).clamp(-127, 127)
        steps = (one.q[rows].double() - want).abs()
    return float(steps.max()), int((steps > 0).sum())


def _timed(
    torch: ModuleType, launch: Callable[[int], object], reps: int, card: Card
) -> list[float]:
    """Device microseconds of `reps` launches, queued behind a sleep: the batch reruns with a
    doubled sleep until the sleep is still running once the host has queued it all, so no host
    gap enters a measurement. `launch` gets a running index, never repeated across reruns."""
    cycles, index = 0, 0
    while True:
        torch.cuda.synchronize()
        marks = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(reps)
        ]
        slept = torch.cuda.Event()
        started = time.perf_counter()
        torch.cuda._sleep(cycles)
        slept.record()
        for begin, end in marks:
            begin.record()
            launch(index)
            end.record()
            index += 1
        queued = time.perf_counter() - started
        ahead = not slept.query()
        torch.cuda.synchronize()
        if ahead:
            return [begin.elapsed_time(end) * 1e3 for begin, end in marks]
        cycles = max(2 * cycles, int(2 * queued * card.clock_hz))


def _shape(
    torch: ModuleType, ops: prebuilt.Prebuilt, f: prebuilt._Format, card: Card, m: int,
    n: int, k: int, models: list[str],
) -> Measured:  # fmt: skip
    sets = _pool(torch, ops, f, card, m, n, k)
    rows = torch.tensor(sorted({*range(0, m, ROW_STEP), m - 1}), device=sets[0].x.device)
    exact = _exact(torch, sets[0], rows)
    budget = {
        name: 2 * verify.error(torch, exact.to(getattr(torch, name)), exact) for name in DTYPES
    }
    gemm = ops.fp8_gemm if f.fp8 else ops.int8_gemm
    choices: list[int | str] = [DEFAULT, *_tiles(ops, f)]
    if not f.fp8 or ops.cutlass_fp8:
        choices.append(verify.CUBLASLT)
    if not f.fp8 and m <= _kernels._gemm.GEMV_ROWS:
        choices.append(verify.GEMV)

    def call(choice: int | str, one: _Set, dtype: str) -> Tensor:
        out = getattr(torch, dtype)
        if choice == verify.GEMV:
            found = ops.int8_gemv(one.x, one.data, one.scale, one.bias, out)
            if found is None:
                raise ValueError("the GEMV does not take this shape")
            return found
        config = -1 if choice == DEFAULT else choice
        return gemm(one.q, one.scales, one.data, one.scale, one.bias, out, config)

    candidates = [_candidate(torch, card, call, choice, sets, rows, exact, budget, m, n, k)
                  for choice in choices]  # fmt: skip
    quantize = ops.fp8_quantize if f.fp8 else ops.int8_quantize
    step, mismatched = _quantize_law(torch, f, sets[0], rows)
    quantize_us = statistics.median(
        _timed(torch, lambda i: quantize(sets[i % len(sets)].x), REPS, card)
    )
    weights = [one.data.to(torch.bfloat16) for one in sets[: max(2, len(sets) // 2)]]
    bf16_us = statistics.median(
        _timed(
            torch,
            lambda i: torch.nn.functional.linear(
                sets[i % len(sets)].x, weights[i % len(weights)], sets[i % len(sets)].bias
            ),
            REPS,
            card,
        )
    )
    return Measured(
        f.name, m, n, k, verify.shape_class(n, k), sorted(models), len(sets), step, mismatched,
        quantize_us, bf16_us, candidates,
    )  # fmt: skip


def _candidate(
    torch: ModuleType, card: Card, call: Callable[[int | str, _Set, str], Tensor],
    choice: int | str, sets: Sequence[_Set], rows: Tensor, exact: Tensor,
    budget: dict[str, float], m: int, n: int, k: int,
) -> Candidate:  # fmt: skip
    """One candidate at one shape: a cold launch (operands evicted by the pool since their
    last use), its bf16 and fp16 outputs checked, and only if both pass its warm median and
    rates."""
    host: list[float] = []

    def cold(index: int) -> None:
        started = time.perf_counter()
        call(choice, sets[(1 + index) % len(sets)], DTYPES[0])
        host.append((time.perf_counter() - started) * 1e6)

    try:
        (cold_us,) = _timed(torch, cold, 1, card)
        outputs = [call(choice, sets[0], dtype)[rows] for dtype in DTYPES]
    except ValueError as refused:  # this tile or the GEMV does not take the shape
        return Candidate(choice, ok=True, note=f"not supported: {refused}"[:200])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return Candidate(choice, ok=True, note="out of memory on this card")
    except Exception as exc:  # a kernel that raises is wrong, never skipped
        return Candidate(choice, ok=False, note=f"{type(exc).__name__}: {exc}"[:200])
    errors = {
        name: verify.error(torch, out, exact) for name, out in zip(DTYPES, outputs, strict=True)
    }
    ok = all(errors[name] <= budget[name] for name in DTYPES)
    # JSON holds no infinity: a non-finite output's error is stored as -1 and says so.
    stored = {name: error if math.isfinite(error) else -1.0 for name, error in errors.items()}
    found = Candidate(choice, ok, stored, budget, cold_us, host[0])
    if not ok:
        finite = all(math.isfinite(error) for error in errors.values())
        found.note = f"{'' if finite else 'non-finite output; '}numeric check failed: untimed"
        return found
    found.samples_us = [
        round(sample, 3)
        for sample in _timed(
            torch, lambda i: call(choice, sets[i % len(sets)], DTYPES[0]), REPS, card
        )
    ]
    samples = sorted(found.samples_us)
    median = statistics.median(samples)
    quartiles = statistics.quantiles(samples, n=4)
    found.warm_us = (samples[0], quartiles[0], median, quartiles[2], samples[-1])
    found.tops = 2 * m * n * k / (median * 1e-6) / 1e12
    found.dram_gbps = (m * k + n * k + 2 * m * n) / (median * 1e-6) / 1e9
    if found.tops * 1e12 > card.peak_ops or found.dram_gbps * 1e9 > card.peak_bytes_per_s:
        found.impossible = True
        found.note = "faster than the card's peak: excluded from rules"
    return found


def _verdict(evidence: Sequence[Measured]) -> verify.Verdict:
    """Every candidate that ran passed its numeric check at every shape; the detail names the
    worst error against its budget and counts timings above the card's peak."""
    ran = [(one, c) for one in evidence for c in one.candidates if c.rel_l2]
    failed = [(one, c) for one in evidence for c in one.candidates if not c.ok]
    if not ran:
        return verify.Verdict(False, "no candidate ran")
    one, worst = max(ran, key=lambda pair: max(_ratio(pair[1], d) for d in DTYPES))
    dtype = max(DTYPES, key=lambda d: _ratio(worst, d))
    detail = (
        f"{len(evidence)} shapes, {len(ran)} candidate runs; worst {worst.choice} at "
        f"{one.m}x{one.n}x{one.k} {dtype}: rel L2 {worst.rel_l2[dtype]:.2e} "
        f"(budget {worst.budget[dtype]:.2e})"
    )
    impossible = sum(c.impossible for one in evidence for c in one.candidates)
    if impossible:
        detail += f"; {impossible} timings above the card's peak, excluded from rules"
    if failed:
        one, bad = failed[0]
        return verify.Verdict(
            False, f"{len(failed)} failed, first {bad.choice} at {one.m}x{one.n}x{one.k}: "
            f"{bad.note}; {detail}",
        )  # fmt: skip
    return verify.Verdict(True, detail)


def _ratio(candidate: Candidate, dtype: str) -> float:
    budget = candidate.budget[dtype]
    return candidate.rel_l2[dtype] / budget if budget else math.inf


def _rules(evidence: Sequence[Measured]) -> tuple[verify.Rule, ...]:
    """Per (shape class, M bucket (2^(b-1), 2^b]): the candidate with the least geometric-mean
    time over the cell's shapes, where it beats the family default by more than the largest
    relative interquartile spread either shows there."""
    cells: dict[tuple[str, int], list[Measured]] = defaultdict(list)
    for one in evidence:
        cells[one.shape, math.ceil(math.log2(one.m))].append(one)
    rules = []
    for (kind, bucket), group in sorted(cells.items()):
        timed = [
            {c.choice: c for c in one.candidates if c.ok and c.warm_us and not c.impossible}
            for one in group
        ]
        if not all(DEFAULT in found for found in timed):
            continue
        shared = set.intersection(*(set(found) for found in timed)) - {DEFAULT}

        if not shared:
            continue
        means = {
            choice: statistics.geometric_mean(found[choice].warm_us[2] for found in timed)
            for choice in (*shared, DEFAULT)
        }
        best = min(shared, key=lambda choice: (means[choice], str(choice)))
        spread = max(
            (found[choice].warm_us[3] - found[choice].warm_us[1]) / found[choice].warm_us[2]
            for found in timed
            for choice in (DEFAULT, best)
        )
        if means[DEFAULT] / means[best] > 1 + spread:
            low = 0 if bucket <= 0 else 2 ** (bucket - 1)
            rules.append(verify.Rule(low, 2**bucket, kind, best))
    return tuple(rules)
