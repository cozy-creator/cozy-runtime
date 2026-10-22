"""The development oracle's GEMM calibration (`cli.qualify`); no consumer imports it.

For each 8-bit format this card's row runs, every legal candidate of its family runs at the
shipped packages' GEMM shapes (`SHAPES`, written by `scripts/gemm-census.py`): each tile of the
family's table, the compiled family default, cuBLASLt, and the GEMV for a few int8 rows.

- **Numerics first.** A candidate's timing counts only after its bf16 and fp16 outputs match
  the exact float64 product of the same quantized operands, at one row in every 16-row block.
  The budget is twice the arm's error model: the output dtype's own rounding of that product,
  plus, for fp8, an accumulator of `FP8_ACCUMULATOR_BITS` random-walking over K, plus, for the
  GEMV (which quantizes its rows itself), every code at a rounding tie flipping. A non-finite
  output fails before any ratio. The activation quantizer is held to its law: a code may differ
  only at a tie.
- **Timing.** After its checked first call, each candidate runs interleaved with the others in
  a fresh random order each round: one cold round (each one's first timed launch, on operands
  the pool has since evicted), then `REPS` warm rounds, the quantizer and bf16 baselines among
  them. A round is queued behind a sleep, so the host never lets the device idle inside it.
  Every launch takes the pool's least-recently-used operands; the pool holds over twice the L2.
- **Plausibility.** A rate above the card's 8-bit tensor peak or its DRAM peak is flagged and
  excluded from rules.
- **Rules.** A workload rule (`verify.Rule`) holds for this card's SM-count bucket and a
  (shape class, M bucket) cell. It ships where one candidate regresses no shape of the cell
  beyond that shape's measured spread, and beats the family default over the cell's
  layer-weighted time by more than the cell's spread.

A failed candidate is excluded alone. The verdict fails only on the family default or the
quantizer: what serves without a rule.
"""

from __future__ import annotations

import math
import random
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType

import msgspec

from cozy_runtime import _kernels
from cozy_runtime.internal import arch, prebuilt, verify
from cozy_runtime.internal.encoding.formats import Tensor, hadamard256

#: The packages' GEMMs: one row per (model, lane, M, N, K) with its layer count.
SHAPES = Path(__file__).parent / "qualification_tables" / "gemm-shapes.json"
#: Warm rounds per shape: the median's standard error is then under a quarter of one launch's
#: spread.
REPS = 32
#: The smallest M tile any family's table holds (mma.sync's 16x128): one reference row in each
#: block of this many rows, at a varying offset, plus the last row.
ROW_BLOCK = 16
#: Float64 weight bytes per reference chunk, so the exact product fits beside the pools on a
#: 24 GB card.
REFERENCE_BYTES = 1 << 28
#: Dense 8-bit tensor-core operations per clock per SM, from NVIDIA's architecture whitepapers:
#: the ceiling no measured rate may exceed.
OPS_PER_CLOCK = {80: 4096, 86: 2048, 89: 2048, 90: 8192, 100: 16384, 103: 16384, 120: 2048}
#: Twice the widest tile any family's table holds (256): every tile takes it whole.
TILE_PROBE = 512
#: Mantissa bits an fp8 tensor-core accumulator keeps under fast accumulation (DeepSeek-V3,
#: section 3.3.2, measured about 14 on Hopper). int8 accumulates exactly in int32.
FP8_ACCUMULATOR_BITS = 14
#: The worst fp32 error of a rotated int8 code, in code units: a 256-term sum within +-127.
INT8_TIE = 256 * 127 * 2.0**-24
#: The worst fp32 error of an fp8 code relative to its value: one division, the scale's rounding.
FP8_TIE = 2.0**-22
DEFAULT = "default"
#: The quantizer and the bf16 GEMM the leaf replaces, timed in the same rounds.
BASELINES = ("quantize", "bf16")
#: Out dtypes the lanes write (SDXL fp16, the others bf16); timing uses the first.
DTYPES = ("bfloat16", "float16")

type Choice = int | str


class Shape(msgspec.Struct, frozen=True):
    model: str
    lane: str
    format: str
    m: int
    n: int
    k: int
    layers: int


class Candidate(msgspec.Struct):
    choice: Choice
    ok: bool
    #: out dtype -> relative L2 against the exact product, and its budget
    rel_l2: dict[str, float] = {}
    budget: dict[str, float] = {}
    #: the first timed launch, on evicted operands (device us); the first call's host us
    cold_us: float = 0.0
    host_first_us: float = 0.0
    #: warm device microseconds: min, p25, median, p75, max; and every round's, in order
    warm_us: tuple[float, ...] = ()
    samples_us: list[float] = []
    tops: float = 0.0
    dram_gbps: float = 0.0
    #: a warm rate above the card's peak: numerics passed, but no rule may use it
    impossible: bool = False
    note: str = ""


class Quantizer(msgspec.Struct):
    """The activation quantizer against its law at the reference rows."""

    max_step: float
    mismatched: int
    #: differing codes not at a rounding tie: a bug
    unexplained: int


class Measured(msgspec.Struct):
    format: str
    m: int
    n: int
    k: int
    shape: str
    layers: int
    models: list[str]
    pool: int
    #: the seed of every round's candidate order
    seed: str
    quantizer: Quantizer
    quantize_us: float
    bf16_us: float
    candidates: list[Candidate]


@dataclass(frozen=True, slots=True)
class Card:
    clock_hz: float
    l2_bytes: int
    sms: int
    peak_ops: float
    peak_bytes_per_s: float

    def facts(self) -> dict[str, float | int]:
        return {
            "clock_hz": self.clock_hz,
            "l2_bytes": self.l2_bytes,
            "sms": self.sms,
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
        sms=props.multi_processor_count,
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


def shapes(sm: int) -> int:
    """How many GEMM shapes `run` measures on this card."""
    found = msgspec.json.decode(SHAPES.read_bytes(), type=list[Shape])
    formats = {f.name for f in (prebuilt.INT8, prebuilt.FP8) if getattr(arch.row(sm), f.name)}
    return len({(one.format, one.m, one.n, one.k) for one in found if one.format in formats})


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
        cells: dict[tuple[int, int, int], list[Shape]] = defaultdict(list)
        for one in shapes:
            if one.format == f.name:
                cells[one.m, one.n, one.k].append(one)
        evidence = []
        for (m, n, k), rows in sorted(cells.items()):
            evidence.append(_shape(torch, ops, f, measured, m, n, k, rows))
            progress(evidence[-1])
        found[f.name] = Calibration(_verdict(evidence), _rules(evidence, measured.sms), evidence)
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


@dataclass(slots=True)
class _Pool:
    """Operand sets used in turn across every launch at a shape: between two uses of one set,
    all the others (over twice the L2) were used."""

    sets: list[_Set]
    turn: int = 0
    #: device-clock cycles of sleep that last kept a whole round at this shape queued
    sleep: int = 0

    def next(self) -> _Set:
        self.turn += 1
        return self.sets[self.turn % len(self.sets)]


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


def _rows(m: int, rng: random.Random) -> list[int]:
    """One row in every `ROW_BLOCK`-row block, at a random offset, and the last row."""
    blocks = range(0, m, ROW_BLOCK)
    return sorted({*(start + rng.randrange(min(ROW_BLOCK, m - start)) for start in blocks), m - 1})


def _picked(torch: ModuleType, codes: Tensor, rows: Tensor) -> Tensor:
    """`codes[rows]` for 8-bit codes, indexed as bytes (torch indexes no float8 dtype)."""
    return codes.view(torch.uint8)[rows].view(codes.dtype)


def _exact(torch: ModuleType, one: _Set, rows: Tensor) -> Tensor:
    """The exact product of the quantized operands at `rows`, in float64 chunks of columns."""
    q = _picked(torch, one.q, rows).double()
    width = max(1, REFERENCE_BYTES // (8 * one.data.shape[1]))
    parts = [
        q @ one.data[start : start + width].double().t()
        for start in range(0, one.data.shape[0], width)
    ]
    product = torch.cat(parts, dim=1) * one.scales[rows].double().view(-1, 1)
    return product * one.scale.double().view(1, -1) + one.bias.double().view(1, -1)


def _law(
    torch: ModuleType, f: prebuilt._Format, one: _Set, rows: Tensor
) -> tuple[Quantizer, Tensor]:
    """The native quantizer's codes against its law at `rows`, and where the law sits at a
    rounding tie (within the fp32 error of the codes' arithmetic): only there may a code differ,
    by one step."""
    x, got = one.x[rows], _picked(torch, one.q, rows).double()
    if f.fp8:
        law, scale = prebuilt._fp8_law(torch, x)
        value = x.double() / scale.double()
        want = law.double()
        step = torch.exp2(torch.floor(torch.log2(want.abs().clamp(min=2.0**-6))) - 3)
        ties = (value / step - torch.floor(value / step) - 0.5).abs() * step <= (
            FP8_TIE * value.abs()
        )
    else:
        rotated = hadamard256(torch, x.double()) / 16
        value = rotated / (rotated.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-12)
        want, step = torch.round(value).clamp(-127, 127), torch.ones_like(value)
        ties = (value - torch.floor(value) - 0.5).abs() <= INT8_TIE
    steps = (got - want).abs() / step
    differs = steps > 0
    unexplained = int((differs & ~ties).sum() + (steps > 1).sum())
    return Quantizer(float(steps.max()), int(differs.sum()), unexplained), ties


def _timed(
    torch: ModuleType, launches: Sequence[Callable[[], bool]], card: Card, pool: _Pool
) -> list[float] | None:
    """Device microseconds of each launch, the round queued behind a sleep: it reruns with a
    longer sleep until the sleep is still running once the host has queued it all, so no host
    gap enters a measurement. None, at once, when a launch did not run (it may have waited on
    the device, so no sleep could stay ahead). The sleep that sufficed starts the shape's next
    round."""
    while True:
        torch.cuda.synchronize()
        marks = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in launches
        ]
        slept = torch.cuda.Event()
        started = time.perf_counter()
        torch.cuda._sleep(pool.sleep)
        slept.record()
        ran = True
        for launch, (begin, end) in zip(launches, marks, strict=True):
            begin.record()
            ran = launch() and ran
            end.record()
        queued = time.perf_counter() - started
        ahead = not slept.query()
        torch.cuda.synchronize()
        if not ran:
            return None
        if ahead:
            return [begin.elapsed_time(end) * 1e3 for begin, end in marks]
        pool.sleep = max(2 * pool.sleep, int(2 * queued * card.clock_hz))


def _measure(
    torch: ModuleType, card: Card, pool: _Pool, calls: dict[Choice, Callable[[_Set, str], object]],
    choices: list[Choice], rng: random.Random, rounds: int, left: dict[Choice, str],
) -> dict[Choice, list[float]]:  # fmt: skip
    """`rounds` interleaved rounds of `choices`, each in a fresh random order. A candidate that
    stops running leaves `choices` with its reason in `left`, and its round runs again
    without it."""
    samples: dict[Choice, list[float]] = defaultdict(list)

    def launch(choice: Choice) -> Callable[[], bool]:
        def one() -> bool:
            try:
                calls[choice](pool.next(), DTYPES[0])
            except ValueError as refused:  # this tile or the GEMV does not take the shape
                left[choice] = f"not supported: {refused}"[:200]
            except torch.cuda.OutOfMemoryError:
                left[choice] = "out of memory on this card"
            except Exception as exc:  # a kernel that raises is wrong, never skipped
                left[choice] = f"failed: {type(exc).__name__}: {exc}"[:200]
            return choice not in left

        return one

    done = 0
    while done < rounds:
        order = rng.sample(choices, len(choices))
        found = _timed(torch, [launch(c) for c in order], card, pool)
        if found is None:
            choices[:] = [c for c in choices if c not in left]
            continue
        for choice, us in zip(order, found, strict=True):
            samples[choice].append(round(us, 3))
        done += 1
    return samples


def _shape(
    torch: ModuleType, ops: prebuilt.Prebuilt, f: prebuilt._Format, card: Card, m: int,
    n: int, k: int, rows_of: Sequence[Shape],
) -> Measured:  # fmt: skip
    seed = f"{f.name} {m}x{n}x{k}"
    rng = random.Random(seed)
    pool = _Pool(_pool(torch, ops, f, card, m, n, k))
    first = pool.sets[0]
    rows = torch.tensor(_rows(m, rng), device=first.x.device)
    exact = _exact(torch, first, rows)
    accumulator = 2.0**-FP8_ACCUMULATOR_BITS * math.sqrt(k) if f.fp8 else 0.0
    budget = {
        name: 2 * (verify.error(torch, exact.to(getattr(torch, name)), exact) + accumulator)
        for name in DTYPES
    }
    quantizer, ties = _law(torch, f, first, rows)
    gemm = ops.fp8_gemm if f.fp8 else ops.int8_gemm
    calls: dict[Choice, Callable[[_Set, str], object]] = {}
    tiles: list[Choice] = [DEFAULT, *_tiles(ops, f)]
    if ops.cutlass_fp8 or not f.fp8:  # else cuBLASLt is the default itself
        tiles.append(verify.CUBLASLT)
    for tile in tiles:
        calls[tile] = partial(_gemm, gemm, -1 if tile == DEFAULT else tile)
    budgets = dict.fromkeys(calls, budget)
    if not f.fp8 and m <= _kernels._gemm.GEMV_ROWS:
        calls[verify.GEMV] = partial(_gemv, ops)
        # The GEMV quantizes its rows itself: at a tie its code may be the other one.
        flips = ties.double() @ first.data.double().abs().t()
        flips = flips * first.scales[rows].double()[:, None] * first.scale.double()
        budgets[verify.GEMV] = {
            name: value + 2 * float(flips.norm() / exact.norm()) for name, value in budget.items()
        }
    # The first call of each: refused, raised, or checked in both dtypes before any timing.
    candidates = {c: _checked(torch, c, calls[c], first, rows, exact, budgets[c]) for c in calls}
    alive = [c for c, found in candidates.items() if found.ok and not found.note]
    left: dict[Choice, str] = {}
    # Cold: each one's first timed launch here, on operands the pool has since evicted.
    cold = _measure(torch, card, pool, calls, alive, rng, 1, left)
    quantize = ops.fp8_quantize if f.fp8 else ops.int8_quantize
    weights = [one.data.to(torch.bfloat16) for one in pool.sets[: max(2, len(pool.sets) // 2)]]
    everything: dict[Choice, Callable[[_Set, str], object]] = {
        **calls,
        "quantize": lambda one, dtype: quantize(one.x),
        "bf16": lambda one, dtype: torch.nn.functional.linear(
            one.x, weights[pool.turn % len(weights)], one.bias
        ),
    }
    warm = _measure(torch, card, pool, everything, [*alive, *BASELINES], rng, REPS, left)
    if any(baseline in left for baseline in BASELINES):
        raise RuntimeError(f"a baseline did not run: {left}")
    for choice, found in candidates.items():
        if choice in left:
            found.ok, found.note = (
                left[choice].startswith(("not supported", "out of")),
                left[choice],
            )
        elif choice in alive:
            found.cold_us = cold[choice][0]
            _rates(found, warm[choice], card, m, n, k)
    return Measured(
        f.name, m, n, k, verify.shape_class(n, k), sum(one.layers for one in rows_of),
        sorted(f"{one.model}/{one.lane} x{one.layers}" for one in rows_of), len(pool.sets), seed,
        quantizer, statistics.median(warm["quantize"]), statistics.median(warm["bf16"]),
        list(candidates.values()),
    )  # fmt: skip


def _gemm(gemm: Callable[..., Tensor], config: Choice, one: _Set, dtype: str) -> Tensor:
    out = getattr(gemm.__self__.torch, dtype)  # type: ignore[attr-defined]
    return gemm(one.q, one.scales, one.data, one.scale, one.bias, out, config)


def _gemv(ops: prebuilt.Prebuilt, one: _Set, dtype: str) -> Tensor:
    found = ops.int8_gemv(one.x, one.data, one.scale, one.bias, getattr(ops.torch, dtype))
    if found is None:
        raise ValueError("the GEMV does not take this shape")
    return found


def _checked(
    torch: ModuleType, choice: Choice, call: Callable[[_Set, str], object], first: _Set,
    rows: Tensor, exact: Tensor, budget: dict[str, float],
) -> Candidate:  # fmt: skip
    """A candidate's first calls at a shape, on the reference operands: refused (not taken,
    ok), raised (failed), or its bf16 and fp16 outputs checked against the exact product."""
    found = Candidate(choice, False, {}, budget)
    started = time.perf_counter()
    try:
        outputs = [call(first, DTYPES[0])]
        found.host_first_us = (time.perf_counter() - started) * 1e6
        outputs.append(call(first, DTYPES[1]))
    except ValueError as refused:  # this tile or the GEMV does not take the shape
        return Candidate(choice, True, note=f"not supported: {refused}"[:200])
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return Candidate(choice, True, note="out of memory on this card")
    except Exception as exc:  # a kernel that raises is wrong, never skipped
        found.note = f"failed: {type(exc).__name__}: {exc}"[:200]
        return found
    errors = {
        name: verify.error(torch, out[rows], exact)  # type: ignore[index]
        for name, out in zip(DTYPES, outputs, strict=True)
    }
    found.ok = all(errors[name] <= budget[name] for name in DTYPES)
    # JSON holds no infinity: a non-finite output's error is stored as -1 and says so.
    found.rel_l2 = {name: e if math.isfinite(e) else -1.0 for name, e in errors.items()}
    if not found.ok:
        finite = all(math.isfinite(e) for e in errors.values())
        found.note = f"{'' if finite else 'non-finite output; '}numeric check failed: untimed"
    return found


def _rates(found: Candidate, samples: list[float], card: Card, m: int, n: int, k: int) -> None:
    found.samples_us = samples
    ordered = sorted(samples)
    median = statistics.median(ordered)
    quartiles = statistics.quantiles(ordered, n=4)
    found.warm_us = (ordered[0], quartiles[0], median, quartiles[2], ordered[-1])
    found.tops = 2 * m * n * k / (median * 1e-6) / 1e12
    found.dram_gbps = (m * k + n * k + 2 * m * n) / (median * 1e-6) / 1e9
    if found.tops * 1e12 > card.peak_ops or found.dram_gbps * 1e9 > card.peak_bytes_per_s:
        found.impossible = True
        found.note = "faster than the card's peak: excluded from rules"


def _verdict(evidence: Sequence[Measured]) -> verify.Verdict:
    """What serves without a rule passed everywhere: the family default's numeric check at every
    shape, and the quantizer's law. Other failed candidates are excluded alone and counted."""
    defaults = [(one, c) for one in evidence for c in one.candidates if c.choice == DEFAULT]
    if not defaults:
        return verify.Verdict(False, "no shape ran")
    one, worst = max(defaults, key=lambda pair: max(_ratio(pair[1], d) for d in DTYPES))
    dtype = max(DTYPES, key=lambda d: _ratio(worst, d))
    failed = [(one, c) for one in evidence for c in one.candidates if not c.ok]
    detail = (
        f"{len(evidence)} shapes, {sum(len(one.candidates) for one in evidence)} candidate runs; "
        f"default worst at {one.m}x{one.n}x{one.k} {dtype}: "
        f"rel L2 {worst.rel_l2.get(dtype, -1):.2e} (budget {worst.budget.get(dtype, 0):.2e}); "
        f"{len(failed)} other candidates excluded; "
        f"quantizer max step {max(o.quantizer.max_step for o in evidence):g}"
    )
    impossible = sum(c.impossible for one in evidence for c in one.candidates)
    if impossible:
        detail += f"; {impossible} timings above the card's peak, excluded from rules"
    bad = [(o, c) for o, c in defaults if not c.ok]
    if bad:
        o, c = bad[0]
        return verify.Verdict(False, f"the default failed at {o.m}x{o.n}x{o.k}: {c.note}; {detail}")
    unexplained = [o for o in evidence if o.quantizer.unexplained]
    if unexplained:
        o = unexplained[0]
        return verify.Verdict(
            False, f"the quantizer broke its law at {o.m}x{o.n}x{o.k} "
            f"({o.quantizer.unexplained} codes); {detail}",
        )  # fmt: skip
    return verify.Verdict(True, detail)


def _ratio(candidate: Candidate, dtype: str) -> float:
    budget = candidate.budget.get(dtype, 0.0)
    error = candidate.rel_l2.get(dtype, math.inf)
    return math.inf if error < 0 or not budget else error / budget


def _spread(candidate: Candidate) -> float:
    """The relative interquartile spread of a candidate's warm samples."""
    return (candidate.warm_us[3] - candidate.warm_us[1]) / candidate.warm_us[2]


def _rules(evidence: Sequence[Measured], sms: int) -> tuple[verify.Rule, ...]:
    """Per (shape class, M bucket (2^(b-1), 2^b]), for this card's SM-count bucket, the cell's
    `_best` candidate where it has one."""
    cells: dict[tuple[str, int], list[Measured]] = defaultdict(list)
    for one in evidence:
        cells[one.shape, math.ceil(math.log2(one.m))].append(one)
    power = math.ceil(math.log2(sms))
    rules = []
    for (kind, bucket), group in sorted(cells.items()):
        best = _best(group)
        if best is not None:
            low = 0 if bucket <= 0 else 2 ** (bucket - 1)
            rules.append(verify.Rule(low, 2**bucket, kind, best, 2 ** (power - 1), 2**power))
    return tuple(rules)


def _best(group: Sequence[Measured]) -> Choice | None:
    """Of the candidates timed at every shape of a cell and regressing none beyond that shape's
    spread, the one with the least layer-weighted time, where it beats the family default's by
    more than the cell's largest spread."""
    timed = [
        {c.choice: c for c in one.candidates if c.ok and c.warm_us and not c.impossible}
        for one in group
    ]
    if not all(DEFAULT in found for found in timed):
        return None
    held = [
        choice
        for choice in timed[0]
        if choice != DEFAULT
        and all(
            choice in found
            and found[choice].warm_us[2]
            <= found[DEFAULT].warm_us[2]
            * (1 + max(_spread(found[choice]), _spread(found[DEFAULT])))
            for found in timed
        )
    ]
    weighted = {
        choice: sum(
            one.layers * found[choice].warm_us[2] for one, found in zip(group, timed, strict=True)
        )
        for choice in (*held, DEFAULT)
    }
    if not held:
        return None
    best = min(held, key=lambda choice: (weighted[choice], str(choice)))
    spread = max(_spread(found[c]) for found in timed for c in (DEFAULT, best))
    return best if weighted[DEFAULT] / weighted[best] > 1 + spread else None
