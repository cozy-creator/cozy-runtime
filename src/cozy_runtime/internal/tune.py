"""Local tuning of the prebuilt 8-bit GEMMs on this card (xs-065 change 2, after magnitude).

A GEMM's launch config is its kernel's rule (`choose()` in `kernels/csrc/gemm`) until this card
has measured a better one for that exact shape. Each shape a served call multiplies at is noted
(`config`); while its executor is idle, the tuner times the shape on its family's configs. A
config replaces the rule only when its output is bit-identical to the rule's on the same
operands (int8 accumulates exactly; an fp8 config that sums in another order shows here and is
refused), its measured rate is one the card can reach, and its median beats the rule's by more
than both spreads. Every measurement is kept in one file on this machine, keyed by the kernel
binary, the card and the driver. A missing, stale or unreadable file is the rule; nothing is
uploaded, and nothing here decides a lane or a route.

A run never waits on tuning beyond one round: the tuner holds the device for a round at a time
(milliseconds) and stops at a round boundary when a command starts (`command`). A call reads
the table as it stands: eager, its next call; a CUDA graph, its next capture.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random
import statistics
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import msgspec

from cozy_runtime.internal import accel, filelock, kernel_cache
from cozy_runtime.internal.encoding.formats import Tensor

if TYPE_CHECKING:
    from cozy_runtime.internal.prebuilt import Prebuilt

#: One GEMM shape as a call multiplies it: format, fp16 output (else bf16), M, N, K.
type Key = tuple[str, bool, int, int, int]

#: Wall seconds of tuning one executor spends (magnitude's first-load allowance). What is left
#: waits for the next executor on this machine, which starts from what this one kept.
BUDGET_S = 60.0
#: Timed rounds per shape, after one untimed: each candidate once a round, in a fresh order.
ROUNDS = 15
#: Past the largest family table (mma.sync's 7): configs past a table's end refuse.
CONFIGS = 12
#: Device cycles a round starts queued behind, so no host gap enters a launch's time.
SLEEP_CYCLES = 1 << 20
#: Tables kept on a machine: a floor under growth across kernel builds and drivers.
KEEP = 64
#: Dense 8-bit tensor-core operations per clock per SM, from NVIDIA's architecture
#: whitepapers: a rate above the card's ceiling is a broken measurement, never a winner.
OPS_PER_CLOCK = {80: 4096, 86: 2048, 87: 2048, 89: 2048, 90: 8192, 100: 16384, 103: 16384}
#: An sm OPS_PER_CLOCK does not name (RTX 50's block-scaled mma.sync, a new cc).
OPS_FALLBACK = 2048


def name(key: Key) -> str:
    fmt, fp16, m, n, k = key
    return f"{fmt}|{'f16' if fp16 else 'bf16'}|{m}|{n}|{k}"


def parse(text: str) -> Key:
    fmt, dtype, m, n, k = text.split("|")
    return fmt, dtype == "f16", int(m), int(n), int(k)


class Unit(msgspec.Struct, frozen=True, omit_defaults=True):
    """One shape as this card measured it: the config that serves it (None: the rule) and every
    number behind that, microseconds as median and interquartile spread."""

    config: int | None = None
    note: str = ""
    rule_us: float = 0.0
    rule_spread_us: float = 0.0
    configs: dict[str, tuple[float, float]] = {}
    implausible: list[int] = []
    refused: dict[str, str] = {}
    sets: int = 0
    rounds: int = 0

    def gain(self) -> float:
        """The rule's median over the served config's; 1.0 where the rule serves."""
        if self.config is None:
            return 1.0
        return self.rule_us / self.configs[str(self.config)][0]


class Identity(msgspec.Struct, frozen=True):
    """What a table answers for: the kernel binary, the card and its driver."""

    kernel: str
    family: str
    sm: int
    card: str
    driver: str
    schema: int = 1


class Document(msgspec.Struct):
    identity: Identity
    units: dict[str, Unit] = {}


class Table:
    """One card's tuned configs and every measurement behind them, as this machine took them."""

    def __init__(self, identity: Identity, path: Path | None) -> None:
        self.identity, self.path = identity, path
        self.units: dict[str, Unit] = {}
        self.choices: dict[Key, int] = {}

    def merge(self, units: dict[str, Unit]) -> None:
        self.units.update(units)
        self.choices = {
            parse(key): unit.config for key, unit in self.units.items() if unit.config is not None
        }


def identity(ops: Prebuilt, kernel: str) -> Identity:
    torch = ops.torch
    major, minor = torch.cuda.get_device_capability(ops.index)
    card = str(torch.cuda.get_device_properties(ops.index).name)
    return Identity(kernel, ops.family, major * 10 + minor, card, accel.driver_version())


def _path(found: Identity) -> Path | None:
    store = kernel_cache.configured()
    if store is None:
        return None
    digest = hashlib.sha256(msgspec.json.encode(found)).hexdigest()[:32]
    return store.own / "tuned" / f"{digest}.json"


def _read(path: Path | None, want: Identity) -> dict[str, Unit]:
    """The units a table file holds for exactly `want`; none where it is absent, stale or torn."""
    if path is None:
        return {}
    try:
        document = msgspec.json.decode(path.read_bytes(), type=Document)
    except (OSError, msgspec.DecodeError, msgspec.ValidationError):
        return {}
    return document.units if document.identity == want else {}


def _keep(table: Table) -> None:
    """The table's units into its file, merged with what another executor of this card kept
    meanwhile. A store that cannot take it tunes again next process."""
    path = table.path
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("a+") as lock:
            filelock.lock(lock)
            table.merge({**_read(path, table.identity), **table.units})
            temporary = path.with_name(f".{path.name}.{os.getpid()}")
            temporary.write_bytes(msgspec.json.encode(Document(table.identity, table.units)))
            os.replace(temporary, path)
        kept = sorted(path.parent.glob("*.json"), key=lambda one: one.stat().st_mtime)
        for old in kept[: max(len(kept) - KEEP, 0)]:
            old.unlink(missing_ok=True)
    except OSError:
        pass


# ------------------------------------------------------------------- the process's state

#: Device index -> its library and table, once the library serves there (`attach`).
_OPS: dict[int, Prebuilt] = {}
_TABLES: dict[int, Table] = {}
#: Device index -> shapes its calls ran that the table has not measured, and how many calls.
_PENDING: dict[int, dict[Key, int]] = {}
#: Commands running now; the tuner works only at zero (`command`).
_STATE = threading.Condition()
_BUSY = [0]
_SPENT = [0.0]
#: Held for one tuning round; a command starting waits only for the round in flight.
_DEVICE = threading.Lock()
_THREAD: list[threading.Thread] = []


def attach(ops: Prebuilt, kernel: str) -> None:
    """The library serving device `ops.index`, and the table this machine kept for it."""
    found = identity(ops, kernel)
    table = Table(found, _path(found))
    table.merge(_read(table.path, found))
    _OPS[ops.index], _TABLES[ops.index] = ops, table


def config(index: int, key: Key) -> int:
    """The tuned config for `key` on device `index`, else -1 (the rule): a dict lookup a call,
    and a count while the shape waits to be measured."""
    table = _TABLES.get(index)
    if table is None:
        return -1
    found = table.choices.get(key)
    if found is not None:
        return found
    if name(key) not in table.units:
        pending = _PENDING.setdefault(index, {})
        pending[key] = pending.get(key, 0) + 1
    return -1


@contextlib.contextmanager
def command() -> Iterator[None]:
    """One executor command: tuning yields until it ends. It waits only for the round in flight,
    and once the executor is idle again the tuner measures what its calls ran."""
    with _STATE:
        _BUSY[0] += 1
    with _DEVICE:
        pass
    try:
        yield
    finally:
        with _STATE:
            _BUSY[0] -= 1
            _STATE.notify_all()
            if not _THREAD and _PENDING:
                _THREAD.append(threading.Thread(target=_loop, name="cozy-tune", daemon=True))
                _THREAD[0].start()


def _busy() -> bool:
    return _BUSY[0] > 0


def _next() -> tuple[int, Key] | None:
    """The waiting shape worth most: its calls times its multiply-adds."""
    best: tuple[int, int, Key] | None = None
    for index, pending in list(_PENDING.items()):
        for key, calls in list(pending.items()):
            weight = calls * key[2] * key[3] * key[4]
            if best is None or weight > best[0]:
                best = (weight, index, key)
    return None if best is None else (best[1], best[2])


def _loop() -> None:
    while _SPENT[0] < BUDGET_S:
        with _STATE:
            _STATE.wait_for(lambda: not _busy() and _next() is not None)
            found = _next()
        if found is None:
            continue
        index, key = found
        started = time.monotonic()
        try:
            unit = _unit(_OPS[index], key)
        except Exception as exc:  # a shape the tuner cannot measure keeps the rule
            unit = Unit(note=f"{type(exc).__name__}: {exc}"[:300])
        _SPENT[0] += time.monotonic() - started
        if unit is None:  # a command started: the shape waits for the next idle
            continue
        table = _TABLES[index]
        _PENDING[index].pop(key, None)
        table.merge({name(key): unit})
        _keep(table)


# ---------------------------------------------------------------------------- measuring


class _Set(msgspec.Struct):
    q: Tensor
    scales: Tensor
    data: Tensor
    scale: Tensor
    bias: Tensor


type _Call = Callable[[_Set, int, bool], Tensor]


def _unit(ops: Prebuilt, key: Key) -> Unit | None:
    """One shape measured on this card: the rule and each config bit-identical to it, timed in
    interleaved rounds. None when a command started (nothing kept)."""
    fmt, fp16, m, n, k = key
    torch = ops.torch
    if fmt == "fp8" and not ops.cutlass_fp8:
        return Unit(note="cuBLASLt multiplies fp8 on this card: nothing to tune")
    dtype = torch.float16 if fp16 else torch.bfloat16
    props = torch.cuda.get_device_properties(ops.index)
    each = m * k + 4 * m + n * k + 6 * n + 4 * m * n
    free, _ = torch.cuda.mem_get_info(ops.index)
    if 2 * each > free // 4:
        return Unit(note=f"no room: two operand sets are {2 * each} B and {free} B are free")
    count = min(max(2, math.ceil(2 * props.L2_cache_size / each)), max(2, free // 4 // each))
    gemm = ops.fp8_gemm if fmt == "fp8" else ops.int8_gemm

    def call(one: _Set, choice: int, bias: bool = True) -> Tensor:
        return gemm(one.q, one.scales, one.data, one.scale, one.bias if bias else None,
                    dtype, choice)  # fmt: skip

    stream = torch.cuda.Stream(ops.index)
    sets: list[_Set] = []
    with torch.cuda.device(ops.index), torch.cuda.stream(stream):
        for seed in range(count):
            with _DEVICE:
                if _busy():
                    return None
                sets.append(_operands(ops, fmt, m, n, k, dtype, seed))
                stream.synchronize()
        identical: list[int] = []
        refused: dict[str, str] = {}
        with _DEVICE:
            if _busy():
                return None
            rule = [call(sets[0], -1), call(sets[1], -1), call(sets[0], -1, False)]
            for choice in range(CONFIGS):
                try:
                    got = [
                        call(sets[0], choice),
                        call(sets[1], choice),
                        call(sets[0], choice, False),
                    ]
                except ValueError as exc:  # this config does not take the shape, or past the end
                    refused[str(choice)] = str(exc)[:120]
                    continue
                if all(torch.equal(a, b) for a, b in zip(got, rule, strict=True)):
                    identical.append(choice)
                else:
                    refused[str(choice)] = "its output differs from the rule's"
            stream.synchronize()
        times = _rounds(torch, stream, sets, [-1, *identical], call)
    if times is None:
        return None
    major, minor = torch.cuda.get_device_capability(ops.index)
    per_clock = OPS_PER_CLOCK.get(major * 10 + minor, OPS_FALLBACK)
    peak = per_clock * props.multi_processor_count * props.clock_rate * 1e3
    return verdict(times, 2 * m * n * k / peak * 1e6, refused, len(sets))


def verdict(
    times: dict[int, list[float]], floor_us: float, refused: dict[str, str], sets: int
) -> Unit:
    """`times` (config -> device microseconds, -1 the rule's) as a unit: the fastest config
    whose median beats the rule's by more than both spreads and by 1%, none faster than the
    card can multiply (`floor_us`, the shape at the card's peak); else the rule."""
    stats = {c: (statistics.median(t), _spread(t)) for c, t in times.items()}
    rule_us, rule_spread = stats.pop(-1)
    plausible = {c: s for c, s in stats.items() if s[0] >= floor_us / 1.05}
    best = min(plausible, key=lambda c: plausible[c][0], default=None)
    wins = (
        best is not None
        and rule_us - plausible[best][0] > max(rule_spread, plausible[best][1])
        and plausible[best][0] < 0.99 * rule_us
    )
    return Unit(
        config=best if wins else None,
        rule_us=round(rule_us, 3),
        rule_spread_us=round(rule_spread, 3),
        configs={str(c): (round(s[0], 3), round(s[1], 3)) for c, s in sorted(stats.items())},
        implausible=sorted(set(stats) - set(plausible)),
        refused=refused,
        sets=sets,
        rounds=min((len(t) for t in times.values()), default=0),
    )


def _operands(ops: Prebuilt, fmt: str, m: int, n: int, k: int, dtype: object, seed: int) -> _Set:
    """One operand set as a served call has it: activations quantized by the op's own kernel,
    a weight in the format's law, row scales and a bias."""
    torch = ops.torch
    device = torch.device("cuda", ops.index)
    generator = torch.Generator(device=device).manual_seed(20261022 + seed)
    x = torch.randn(m, k, generator=generator, device=device).to(dtype)
    if fmt == "fp8":
        weight = torch.randn(n, k, generator=generator, device=device) * 0.02
        scale = (weight.abs().amax(-1).float() / 448.0).clamp(min=1e-12)
        data = (weight / scale.view(-1, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        q, scales = ops.fp8_quantize(x)
    else:
        data = torch.randint(-127, 128, (n, k), generator=generator, device=device,
                             dtype=torch.int8)  # fmt: skip
        scale = torch.rand(n, generator=generator, device=device).add_(0.5).mul_(1e-3)
        q, scales = ops.int8_quantize(x)
    bias = torch.randn(n, generator=generator, device=device).to(dtype)
    return _Set(q, scales, data, scale.contiguous(), bias)


def _rounds(
    torch: ModuleType, stream: object, sets: list[_Set], choices: list[int], call: _Call
) -> dict[int, list[float]] | None:
    """Device microseconds of each choice: one untimed round, then `ROUNDS`, each queued behind
    a sleep so no host gap enters a launch, in a fresh order, every launch on the least recently
    used operand set (the sets hold over twice the L2). None when a command started."""
    times: dict[int, list[float]] = {c: [] for c in choices}
    rng = random.Random(len(sets))
    turn, sleep = 0, SLEEP_CYCLES
    for round_ in range(ROUNDS + 1):
        order = rng.sample(choices, len(choices))
        while True:
            with _DEVICE:
                if _busy():
                    return None
                marks = []
                slept = torch.cuda.Event()
                torch.cuda._sleep(sleep)
                slept.record(stream)
                for choice in order:
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    turn += 1
                    begin.record(stream)
                    call(sets[turn % len(sets)], choice, True)
                    end.record(stream)
                    marks.append((choice, begin, end))
                ahead = not slept.query()
                stream.synchronize()  # type: ignore[attr-defined]
            if ahead:
                break
            sleep *= 2
        if round_:
            for choice, begin, end in marks:
                times[choice].append(begin.elapsed_time(end) * 1e3)
    return times


def _spread(samples: list[float]) -> float:
    """The interquartile range: what this shape's timing moves by on its own."""
    ordered = sorted(samples)
    return ordered[(3 * len(ordered)) // 4] - ordered[len(ordered) // 4]


def summary(index: int) -> dict[str, int | float]:
    """The run's account of tuning on device `index`: shapes measured, faster than the rule,
    waiting, and the median gain of those faster (rule time over tuned)."""
    table = _TABLES.get(index)
    if table is None:
        return {}
    gains = [unit.gain() for unit in table.units.values() if unit.config is not None]
    return {
        "measured": len(table.units),
        "faster": len(gains),
        "waiting": len(_PENDING.get(index, {})),
        "median_gain": round(statistics.median(gains), 3) if gains else 1.0,
    }
