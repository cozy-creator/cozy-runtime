"""The engine's step launcher on a GPU (`_kernels._engine`, xs-213; the varena suite's `engine-loop` case).

- `loop`: a tiny captured step (advance a device counter `n`, set `go = n < limit`, report `n`) as one
  Segment launch of `STEPS` steps, against the same graph replayed from Python `STEPS` times; then a stop
  the host cannot know (a limit drawn on the device) and a ring the device laps.
- `polled`: a slower segment whose ring the host reads at 10 Hz while the device writes it.
- `attention`, `ggml`: the engine's paged attention and a ggml matmul captured into a body, each step's
  output reported through the ring and held against the same steps run eagerly.
- `copies`: whether the body's host-to-card copy can be moved, resized or disabled through the Segment's
  executable between launches (`Segment.set_copy`, `enable_copy`). It reports what the driver does.

It reports numbers and fails only on a wrong result (or a launcher error).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING, cast

import tensorfs.varena.torch as vt
import torch
from tensorfs import _ext

from cozy_runtime._kernels import NO_ENGINE
from cozy_runtime.engine.attention import HostTier, paged_attention
from cozy_runtime.engine.kv import KvLayout
from cozy_runtime.internal import ggml_linear

if TYPE_CHECKING:
    from cozy_runtime._kernels import _engine

type JSON = None | bool | int | float | str | Sequence[JSON] | Mapping[str, JSON]

STEPS = 10_000
#: More steps than the loop's ring holds: the device laps it.
LAPPED = 40_000
LOOP_SLOTS = 16_384
#: The polled segment: about 100 µs a step (`torch.cuda._sleep` cycles), a ring of 0.4 s or more.
SLOW_CYCLES = 200_000
POLLED_SLOTS = 4096
POLL_S = 0.1
#: Steps of each kernel held against eager.
ITERATIONS = 256
#: What a step's output may differ from eager's by, relative to eager's largest value: the same kernels on
#: the same inputs; consecutive steps' inputs differ far more.
BUDGET = 1e-3
RESERVE = 64 << 20


def _slot(record: int) -> int:
    from cozy_runtime._kernels import _engine

    return _engine.HEADER + -(-record // 16) * 16


class _Ring:
    """Pinned, device-mapped host slots a step reports into, its device counter, and their reader."""

    def __init__(self, slots: int, record: int) -> None:
        from cozy_runtime._kernels import _engine

        self.slots, self.slot_bytes = slots, _slot(record)
        self.host = torch.zeros(slots * self.slot_bytes, dtype=torch.uint8).pin_memory()
        self.counter = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.reader: _engine.Reader = _engine.Reader(self.host.data_ptr(), slots, self.slot_bytes)

    def report(self, src: torch.Tensor) -> None:
        from cozy_runtime._kernels import _engine

        _engine.report(
            self.host.data_ptr(), self.slots, self.slot_bytes, self.counter.data_ptr(), src.data_ptr(),
            src.nbytes, torch.cuda.current_stream().cuda_stream,
        )  # fmt: skip

    def drain(self) -> None:
        """Drop what eager warm-up steps reported, once they ran."""
        torch.cuda.synchronize()
        self.reader.poll()


class _Counted:
    """A step's device counter `n`, advanced first, and its go flag: `n < limit`, both on the device."""

    def __init__(self, limit: int) -> None:
        self.n = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.limit = torch.full((1,), limit, dtype=torch.int64, device="cuda")
        self.below = torch.zeros(1, dtype=torch.bool, device="cuda")
        self.go = torch.zeros(1, dtype=torch.int32, device="cuda")

    def tick(self) -> None:
        self.n.add_(1)
        torch.lt(self.n, self.limit, out=self.below)
        self.go.copy_(self.below)


def _capture(step: Callable[[], None]) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with vt.graph(graph, reserve_bytes=RESERVE):
        step()
    return graph


@contextmanager
def _launching(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> Iterator[None]:
    """What `vt.replay` does around a replay, around launches of an executable holding `graph`'s clone:
    its varena pool mapped, and retained until `stream` passes them."""
    owner = graph._varena_owner  # type: ignore[attr-defined]  # vt.graph marks the graphs it captured
    _, device, pool = cast("tuple[int, int, tuple[int, int]]", owner)
    vt.prepare_pool(pool, 0, device)
    _ext._varena_working_pool_retain(device, pool)
    try:
        yield
    finally:
        _ext._varena_working_pool_record_stream(device, pool, stream.cuda_stream, False)
        _ext._varena_working_pool_release(device, pool)


def _segment(graph: torch.cuda.CUDAGraph, counted: _Counted) -> _engine.Segment:
    from cozy_runtime._kernels import _engine

    return _engine.Segment(graph.raw_cuda_graph(), counted.go.data_ptr(), 0)


def _run(segment: _engine.Segment, graph: torch.cuda.CUDAGraph) -> tuple[float, float]:
    """One launch and one wait for it: the GPU's seconds and the host's."""
    stream = torch.cuda.current_stream()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    done = torch.cuda.Event(blocking=True)
    began = time.perf_counter()
    with _launching(graph, stream):
        begin.record(stream)
        segment.launch(stream.cuda_stream)
        end.record(stream)
    done.record(stream)
    done.synchronize()
    return begin.elapsed_time(end) / 1e3, time.perf_counter() - began


def _ints(records: list[bytes]) -> list[int]:
    return [int.from_bytes(record, "little", signed=True) for record in records]


def _counts(records: list[bytes], first: int, last: int, wrong: list[str], arm: str) -> bool:
    """Whether the records are `n` = first..last, each once, in order."""
    got = _ints(records)
    if got == list(range(first, last + 1)):
        return True
    wrong.append(f"loop {arm}: {len(got)} records, not n = {first}..{last} ({got[:3]}..{got[-3:]})")
    return False


def _loop(wrong: list[str]) -> dict[str, JSON]:
    stream = torch.cuda.current_stream()
    counted = _Counted(STEPS)
    ring = _Ring(LOOP_SLOTS, 8)

    def step() -> None:
        counted.tick()
        ring.report(counted.n)

    step()  # eager: the kernels load before capture
    ring.drain()
    graph = _capture(step)
    segment = _segment(graph, counted)
    row: dict[str, JSON] = {"steps": STEPS, "body_copies": len(segment.copies)}
    try:
        counted.limit.fill_(10)
        counted.n.zero_()
        _run(segment, graph)  # the executable's first launch uploads it
        _counts(ring.reader.poll(), 1, 10, wrong, "first launch")

        counted.limit.fill_(STEPS)
        counted.n.zero_()
        gpu_s, host_s = _run(segment, graph)
        row["segment"] = {
            "us_step": 1e6 * gpu_s / STEPS, "host_us_step": 1e6 * host_s / STEPS,
            "host_launches": 1, "host_waits": 1,
            "ran": _counts(ring.reader.poll(), 1, STEPS, wrong, "segment"),
        }  # fmt: skip

        graph.instantiate()
        counted.n.zero_()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        done = torch.cuda.Event(blocking=True)
        began = time.perf_counter()
        with _launching(graph, stream):
            begin.record(stream)
            for _ in range(STEPS):
                graph.replay()
            end.record(stream)
        done.record(stream)
        done.synchronize()
        host_s = time.perf_counter() - began
        row["python_replay"] = {
            "us_step": 1e3 * begin.elapsed_time(end) / STEPS, "host_us_step": 1e6 * host_s / STEPS,
            "host_launches": STEPS, "host_waits": 1,
            "ran": _counts(ring.reader.poll(), 1, STEPS, wrong, "python replay"),
        }  # fmt: skip

        # A stop the host cannot know: the limit is drawn on the device, read only after the run.
        drawn = torch.Generator(device="cuda").manual_seed(213)
        torch.randint(1000, STEPS, (1,), generator=drawn, out=counted.limit)
        counted.n.zero_()
        _run(segment, graph)
        records = ring.reader.poll()
        limit = int(counted.limit.item())
        row["device_stop"] = {"limit": limit, "steps": len(records),
                              "ran": _counts(records, 1, limit, wrong, "device stop")}  # fmt: skip

        counted.limit.fill_(LAPPED)
        counted.n.zero_()
        _run(segment, graph)
        lapped: dict[str, JSON] = {"steps": LAPPED, "slots": LOOP_SLOTS}
        try:
            records = ring.reader.poll()
            wrong.append(f"loop lapped: the first poll returned {len(records)} records, not a gap")
        except OverflowError as gap:
            lapped["gap"] = str(gap)
            rest = ring.reader.poll()
            lapped["after"] = len(rest)
            lapped["ran"] = _counts(rest, LAPPED - LOOP_SLOTS + 1, LAPPED, wrong, "after the gap")
        row["lapped"] = lapped
    finally:
        segment.close()
        graph.reset()
    return row


def _polled(wrong: list[str]) -> dict[str, JSON]:
    """A slower segment the host reads every `POLL_S` while the device writes the ring round and round."""
    stream = torch.cuda.current_stream()
    counted = _Counted(STEPS)
    ring = _Ring(POLLED_SLOTS, 8)

    def step() -> None:
        counted.tick()
        torch.cuda._sleep(SLOW_CYCLES)
        ring.report(counted.n)

    step()
    ring.drain()
    graph = _capture(step)
    segment = _segment(graph, counted)
    try:
        counted.n.zero_()
        done = torch.cuda.Event(blocking=False)
        records: list[bytes] = []
        wakeups = 0
        began = time.perf_counter()
        with _launching(graph, stream):
            segment.launch(stream.cuda_stream)
        done.record(stream)
        while True:
            time.sleep(POLL_S)
            wakeups += 1
            finished = done.query()
            records += ring.reader.poll()
            if finished or time.perf_counter() - began > 120:
                break
        spent = time.perf_counter() - began
        done.synchronize()
        return {
            "steps": STEPS, "slots": POLLED_SLOTS, "s": spent, "us_step": 1e6 * spent / STEPS,
            "wakeups": wakeups, "wakeups_s": wakeups / spent, "records_wakeup": len(records) / wakeups,
            "ran": _counts(records, 1, STEPS, wrong, "polled"),
        }  # fmt: skip
    finally:
        segment.close()
        graph.reset()


def _against_eager(
    name: str, compute: Callable[[torch.Tensor], torch.Tensor], wrong: list[str]
) -> dict[str, JSON]:
    """`compute(n)` as a step's work: each of `ITERATIONS` steps' output through the ring, against the
    same steps run eagerly."""
    counted = _Counted(ITERATIONS)
    eager = []
    for _ in range(ITERATIONS):
        counted.tick()
        eager.append(compute(counted.n).clone())
    shape, dtype = eager[0].shape, eager[0].dtype
    ring = _Ring(ITERATIONS, eager[0].nbytes)
    ring.report(eager[0])
    ring.drain()

    def step() -> None:
        counted.tick()
        ring.report(compute(counted.n))

    graph = _capture(step)
    segment = _segment(graph, counted)
    try:
        counted.n.zero_()
        gpu_s, _ = _run(segment, graph)
        records = ring.reader.poll()
    finally:
        segment.close()
        graph.reset()
    row: dict[str, JSON] = {"iterations": ITERATIONS, "steps": len(records), "us_step": 1e6 * gpu_s / ITERATIONS}
    if len(records) != ITERATIONS:
        wrong.append(f"{name}: {len(records)} steps, not {ITERATIONS}")
        return row
    bitwise, worst = 0, 0.0
    for record, want in zip(records, eager, strict=True):
        got = torch.frombuffer(bytearray(record), dtype=dtype).view(shape)
        want = want.cpu()
        bitwise += int(torch.equal(got, want))
        worst = max(worst, float((got.float() - want.float()).abs().max() / want.float().abs().max()))
    row |= {"bitwise": bitwise, "worst_error": worst}
    if worst > BUDGET:
        wrong.append(f"{name}: a step is off eager by {worst} of its largest value")
    return row


def _attention(wrong: list[str]) -> dict[str, JSON]:
    """Paged decode attention on a card-only K/V: 8 heads over 2 K/V heads of 128, a row of 1024 tokens and an
    empty one (a free slot in its bucket), whose output must be zero."""
    layout = KvLayout(attn_layers=2, kv_heads=2, head_dim=128, dtype=torch.bfloat16)
    heads, tokens, parts = 8, 1024, 8
    blocks = tokens // layout.block_tokens
    tile = (2, layout.block_tokens, layout.kv_heads, layout.head_dim)
    made = torch.Generator(device="cuda").manual_seed(213)
    card = torch.randn((blocks, layout.attn_layers, *tile), generator=made, device="cuda")
    card = card.mul_(0.25).to(layout.dtype)
    host = HostTier(torch.zeros((1, 1, *tile), dtype=layout.dtype, device="cuda"), 0)  # never read
    table = torch.arange(blocks, dtype=torch.int32, device="cuda")[None].repeat(2, 1).contiguous()
    lengths = torch.tensor([tokens, 0], dtype=torch.int32, device="cuda")
    base = torch.randn((2, heads, layout.head_dim), generator=made, device="cuda").mul_(4).to(layout.dtype)
    q = torch.empty_like(base)

    def compute(n: torch.Tensor) -> torch.Tensor:
        q.copy_(base + n * 0.05)
        return paged_attention(q, card, host, table, lengths, layout, 1, parts, layout.head_dim**-0.5)

    row = _against_eager("attention", compute, wrong)
    empty = compute(torch.zeros((), device="cuda"))[1]
    row["empty_row_zero"] = bool(torch.equal(empty, torch.zeros_like(empty)))
    if not row["empty_row_zero"]:
        wrong.append(f"attention: an empty row's output is not zero ({int(empty.isnan().sum())} NaN)")
    return row


def _ggml(wrong: list[str]) -> dict[str, JSON]:
    """One q8_0 linear, 4096 to 512, one row: MMVQ."""
    deadline = time.monotonic() + 120
    while (ops := ggml_linear.native(torch, 0)) is None:
        if why := ggml_linear.why(0) or ggml_linear.carried(ggml_linear.capability(torch, 0)):
            return {"served": False, "why": why}
        if time.monotonic() > deadline:
            return {"served": False, "why": "the card's ggml kernels were not verified in 120 s"}
        time.sleep(0.1)
    native: ModuleType = ops
    scheme, n_out, k = "ggml.q8_0", 512, 4096
    made = torch.Generator(device="cpu").manual_seed(213)
    blocks = ggml_linear.encode(torch, scheme, torch.randn((n_out, k), generator=made).mul_(0.02)).to("cuda")
    base = torch.randn((1, k), generator=made).to("cuda", torch.bfloat16)
    x = torch.empty_like(base)

    def compute(n: torch.Tensor) -> torch.Tensor:
        x.copy_(base + n * 0.05)
        return ggml_linear.matmul(native, torch, x, blocks, scheme, n_out)

    row = _against_eager("ggml", compute, wrong)
    return {"served": True, "kernel": ggml_linear.kernel(torch, 0, scheme, k, n_out, 1, blocks), **row}


def _copies(wrong: list[str]) -> dict[str, JSON]:
    """A step whose first op copies pinned host bytes to the card: that copy, as the body holds it,
    moved to another source, resized, disabled and enabled through the Segment between launches."""
    nbytes = 4096
    sources = [torch.full((nbytes,), value, dtype=torch.uint8).pin_memory() for value in (1, 2, 3)]
    window = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
    counted = _Counted(1)
    ring = _Ring(8, nbytes)

    def step() -> None:
        counted.tick()
        window.copy_(sources[0], non_blocking=True)
        ring.report(window)

    step()
    ring.drain()
    graph = _capture(step)
    segment = _segment(graph, counted)
    row: dict[str, JSON] = {"body_copies": [list(copy) for copy in segment.copies]}

    def once(before: int) -> bytes:
        counted.n.zero_()
        window.fill_(before)
        _run(segment, graph)
        [record] = ring.reader.poll()
        return record

    def tried(name: str, change: Callable[[], None], before: int, want: bytes) -> None:
        try:
            change()
        except RuntimeError as refused:
            row[name] = {"accepted": False, "error": str(refused)}
            return
        got = once(before)
        row[name] = {"accepted": True, "took_effect": got == want, "first_bytes": list(got[:2]),
                     "last_bytes": list(got[-2:])}  # fmt: skip

    try:
        found = [i for i, (dst, _, _) in enumerate(segment.copies) if dst == window.data_ptr()]
        if not found:
            row["found"] = False
            return row
        index = found[0]
        as_captured = once(0)
        row["as_captured"] = as_captured == bytes([1]) * nbytes
        if not row["as_captured"]:
            wrong.append(f"copies: the captured copy gave {list(as_captured[:4])}, not its source's bytes")
        half = nbytes // 2
        tried("moved", lambda: segment.set_copy(index, window.data_ptr(), sources[1].data_ptr(), nbytes), 0,
              bytes([2]) * nbytes)  # fmt: skip
        tried("resized", lambda: segment.set_copy(index, window.data_ptr(), sources[2].data_ptr(), half), 0,
              bytes([3]) * half + bytes(half))  # fmt: skip
        tried("disabled", lambda: segment.enable_copy(index, False), 7, bytes([7]) * nbytes)
        tried("enabled", lambda: segment.enable_copy(index, True), 0, bytes([3]) * half + bytes(half))
    finally:
        segment.close()
        graph.reset()
    return row


def _replayed(wrong: list[str]) -> dict[str, JSON]:
    """The engine's host-replayed path (`Decoder(launcher=False)`): a counted step captured once and
    replayed from the host `ITERATIONS` times, each output copied out on the card, against eager: no
    WHILE node, no host ring."""
    counted = _Counted(ITERATIONS)
    base = torch.arange(4096, dtype=torch.float32, device="cuda")
    eager = []
    for _ in range(ITERATIONS):
        counted.tick()
        eager.append((base * counted.n.float() + 1.0).clone())
    out = torch.empty_like(base)

    def step() -> None:
        counted.tick()
        out.copy_(base * counted.n.float() + 1.0)

    counted.n.zero_()
    graph = _capture(step)
    counted.n.zero_()
    made = torch.empty((ITERATIONS, *base.shape), dtype=base.dtype, device="cuda")
    for k in range(ITERATIONS):
        graph.replay()
        made[k].copy_(out)
    torch.cuda.synchronize()
    bitwise = sum(int(torch.equal(got, want)) for got, want in zip(made, eager, strict=True))
    if bitwise != ITERATIONS:
        wrong.append(f"replayed: {bitwise} of {ITERATIONS} steps equal eager")
    return {"iterations": ITERATIONS, "bitwise": bitwise}


def measure(only: str = "") -> dict[str, JSON]:
    """Every section, or `only` that one (the varena suite's `engine-loop-<section>`, each its own process,
    so one that crashes names itself)."""
    if NO_ENGINE:
        raise AssertionError(NO_ENGINE)
    major, minor = torch.cuda.get_device_capability(0)
    row: dict[str, JSON] = {"card": torch.cuda.get_device_name(0), "sm": major * 10 + minor,
                            "cuda": torch.version.cuda}  # fmt: skip
    wrong: list[str] = []
    sections: dict[str, Callable[[list[str]], dict[str, JSON]]] = {
        "loop": _loop, "polled": _polled, "attention": _attention, "ggml": _ggml, "copies": _copies,
        "replayed": _replayed,
    }  # fmt: skip
    if only:
        sections = {only: sections[only]}
    for name, section in sections.items():
        try:
            row[name] = section(wrong)
        except Exception as failed:
            row[name] = {"error": f"{type(failed).__name__}: {failed}"}
            wrong.append(f"{name}: {type(failed).__name__}: {failed}")
    torch.cuda.synchronize()
    row["wrong"] = wrong
    if wrong:
        raise AssertionError(json.dumps(row))
    return row
