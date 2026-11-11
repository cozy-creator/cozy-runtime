"""The attention table's measurements (xs-171; `cozy_runtime.bench`'s built-in calls): every attention kernel this card
runs, through the path a model's call takes (`attention.pinned`, then the kernel's backend call), timed at the grid's
shapes and judged against FP32 attention. Nothing here ranks or judges: `scripts/attention_table.py` does, from what the
release gate's `suites/attention/collect.py` gathers.

- `inventory`: the card's hardware class, its kernels, and the sets that can share one process.
- `speed`: a set's kernels over the grid, interleaved, three rounds; SDPA joins every set as the anchor.
- `quality`: one kernel's error per head against FP32 attention on every site.

The pod's clock, temperature and throttling are TensorD's NVML reads of the call, in its record. Q/K/V are synthetic
(`rng.counter_noise`: seeded random, outlier rows, and block-routable for sparse kernels) until the corpus of captured
activations exists."""

from __future__ import annotations

import contextlib
import statistics
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NotRequired, TypedDict

from cozy_runtime import _build_provenance, rng
from cozy_runtime.accel.attention import AttentionLayout, attention_scope
from cozy_runtime.internal import (
    arch,
    attention,
    attention_sol,
    kernel_cache,
    kernel_ready,
    kernel_sources,
)
from cozy_runtime.internal.attention_ulysses import Options

if TYPE_CHECKING:
    import torch

    from cozy_runtime.author import Context
    from cozy_runtime.internal.encoding.selection import DeviceFacts

#: H3's text and audio prefix: the rows a sparse step keeps dense.
_PREFIX = 1259
#: A sparse step's layout: past H3's dense steps.
_SPARSE_STEP = 1
#: Bytes overwritten before each call whose operands fit in L2: in a model a GEMM runs between attentions.
_FLUSH_BYTES = 256 << 20
_ROUNDS, _MIN_SAMPLES, _MAX_SAMPLES, _MIN_MS = 3, 3, 200, 200.0
_OPTIONS = Options(attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False)


@dataclass(frozen=True, slots=True)
class Point:
    """One shape: [batch, lq, heads, dim] queries against `lk` keys and values."""

    batch: int
    lq: int
    lk: int
    heads: int
    dim: int
    dtype: str = "bfloat16"
    #: dense: random operands; sparse: a sparse step's layout over block-routable operands
    layout: str = "dense"

    @property
    def id(self) -> str:
        cross = f"x{self.lk}" if self.lk != self.lq else ""
        batch = f"b{self.batch}-" if self.batch != 1 else ""
        dtype = "" if self.dtype == "bfloat16" else f"-{self.dtype}"
        return f"{self.layout}-d{self.dim}-{batch}{self.lq}{cross}-h{self.heads}{dtype}"


def _self(lq: int, heads: int, dim: int = 128, batch: int = 1, layout: str = "dense") -> Point:
    return Point(batch, lq, lq, heads, dim, layout=layout)


#: measurement-design §1.3: our packages' real lengths; heads the model's own where one sets them, else 24.
GRID: tuple[Point, ...] = (
    *(_self(n, 24) for n in (512, 1024, 2048)),
    _self(4608, 24), _self(4608, 48),           # 1 MP DiTs (Klein, FLUX.2)
    _self(7168, 24), _self(8704, 24),           # Qwen-Image 2512, Klein's edit
    _self(9216, 16),                            # Anima
    Point(1, 9216, 512, 16, 128),               # Anima's cross-attention onto its text
    _self(16640, 24), _self(16640, 32), _self(16640, 48),  # 4 MP DiTs
    _self(32768, 24),
    _self(38400, 56),                           # H3 at 5 s
    *(_self(111598, h) for h in (56, 28, 14, 7)),  # H3 15 s on 1, 2, 4 and 8 GPUs
    _self(387000, 8),                           # Regenerate at 1440p, at 8 heads: time is linear in heads there
    _self(1024, 20, 64, 2), _self(4096, 10, 64, 2), _self(16384, 10, 64, 8), _self(16384, 20, 64, 2),  # SDXL
    *(Point(2, n, 77, h, 64) for n, h in ((1024, 20), (4096, 10), (16384, 10))),  # SDXL onto 77 text tokens
    *(_self(111598, h, layout="sparse") for h in (56, 28, 14, 7)),  # H3's sparse steps
    _self(38400, 56, layout="sparse"),
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class Site:
    """One quality case: operands at a shape, judged per head against FP32 attention."""

    name: str
    point: Point
    #: random, outliers (a few rows far past the rest, as H3's key gain has) or routable (sparse)
    kind: str
    #: the attention level the site runs at: a kernel is judged on every site at or below its own level
    level: int = 4


#: Synthetic until the corpus exists: four heads each, at level 4, so every kernel is judged on every site.
SITES: tuple[Site, ...] = (
    Site("random-d128-4608", _self(4608, 4), "random"),
    Site("random-d128-16640", _self(16640, 4), "random"),
    Site("random-d128-111598", _self(111598, 4), "random"),
    Site("outliers-d128-16640", _self(16640, 4), "outliers"),
    Site("random-d64-4096", _self(4096, 4, 64), "random"),
    Site("outliers-d64-4096", _self(4096, 4, 64), "outliers"),
    Site("cross-d128-9216x512", Point(1, 9216, 512, 4, 128), "random"),
    Site("routable-d128-38400", _self(38400, 4, layout="sparse"), "routable"),
    Site("routable-d128-111598", _self(111598, 4, layout="sparse"), "routable"),
)


class Inventory(TypedDict):
    klass: str
    card: str
    sm: int
    row: str
    half_accumulate: bool
    runtime: str
    commit: str
    torch: str
    cuda: str
    driver: str
    kernels: list[str]
    sets: list[list[str]]


class Timed(TypedDict):
    """One kernel at one point: ms per call (median, p10, p90 over every call) and the spread of its three round
    medians over their median."""

    ms: list[float]
    spread: float


class PointResult(TypedDict):
    id: str
    batch: int
    lq: int
    lk: int
    heads: int
    dim: int
    dtype: str
    layout: str
    data: str
    times: dict[str, Timed]
    #: kernel -> why its row refuses this call: a model's call it refuses runs on SDPA
    refused: dict[str, str]
    #: kernel -> how it failed a call its row admits
    failed: dict[str, str]


class Speed(TypedDict):
    inventory: Inventory
    set: list[str]
    absent: dict[str, str]
    points: list[PointResult]


class Error(TypedDict):
    """One kernel on one site: worst-head and mean relative L2 against FP32 attention, the mean cosine, and the best
    answer a kernel can give without computing. A sparse kernel's `ref16` is its error against the class's 16-bit
    sparse kernel on the same route, and `ref16_fp32` that kernel's own against FP32."""

    site: str
    level: int
    kind: str
    finite: bool
    worst: float
    mean: float
    cosine: float
    trivial: float
    ref16: NotRequired[float]
    ref16_fp32: NotRequired[float]
    sparse_calls: NotRequired[int]


class Quality(TypedDict):
    inventory: Inventory
    kernel: str
    level: int
    reference: str
    corpus: str
    errors: list[Error]
    skipped: dict[str, str]


class Fault(RuntimeError):
    """A kernel left this process's CUDA context unusable: the set runs again without it (the suite's cells)."""


# ---- the card


def _device() -> DeviceFacts:
    import torch

    from cozy_runtime.internal.encoding.selection import measure_device

    return measure_device(torch, 0)


def _compatible(a: attention.Candidate, b: attention.Candidate) -> bool:
    """Whether two kernels can share a process: it holds one build of each module (`attention._build`)."""
    if not a.import_name or a.import_name != b.import_name or a.artifact == b.artifact:
        return True
    recipes = kernel_sources.RECIPES
    return any(x.artifact in recipes and y.artifact in recipes[x.artifact].serves for x, y in ((a, b), (b, a)))


def sets(names: list[str]) -> list[list[str]]:
    found: list[list[str]] = []
    for name in names:
        candidate = attention.BY_NAME[name]
        home = next((s for s in found if all(_compatible(candidate, attention.BY_NAME[o]) for o in s)), None)
        if home is None:
            found.append([name])
        else:
            home.append(name)
    return found


def inventory() -> Inventory:
    import torch

    import cozy_runtime

    device = _device()
    row = attention.device_row(device)
    names = [c.name for c in attention.KERNELS if not attention._off_row(c, row)]
    return {
        "klass": arch.klass(row), "card": device.name, "sm": device.sm, "row": row.name,
        "half_accumulate": row.half_accumulate,
        "runtime": cozy_runtime.__version__, "commit": _build_provenance.COMMIT, "torch": torch.__version__,
        "cuda": str(torch.version.cuda), "driver": device.driver, "kernels": names, "sets": sets(names),
    }  # fmt: skip


# ---- operands


def _operands(point: Point, kind: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q, K and V at POINT from `rng.counter_noise` (the same on any card), a head at a time so the largest point never
    holds an FP32 copy."""
    import torch

    device = torch.device("cuda", 0)
    dtype = getattr(torch, point.dtype)
    q, k, v = (
        torch.empty(point.batch, n, point.heads, point.dim, device=device, dtype=dtype)
        for n in (point.lq, point.lk, point.lk)
    )
    for h in range(point.heads):
        for b in range(point.batch):
            step = h * point.batch + b

            def randn(rows: int, stream: int) -> torch.Tensor:
                return rng.counter_noise((rows, point.dim), seed, step, stream=stream, device=device)

            if kind == "routable":
                # Each query row belongs to one block of keys: a kernel that routes it wrong misses by O(1).
                blocks = -(-point.lk // 64)
                centers, values = randn(blocks, 0) * 2.0, randn(blocks, 1)
                home = torch.arange(point.lk, device=device) // 64
                goes = rng.counter_noise((blocks,), seed, step, stream=2, device=device).argsort()[home]
                q[b, :, h] = (centers[goes][: point.lq] + 0.1 * randn(point.lq, 3)).to(dtype)
                k[b, :, h] = (centers[home] + 0.1 * randn(point.lk, 4)).to(dtype)
                v[b, :, h] = (values[home] + 0.05 * randn(point.lk, 5)).to(dtype)
            else:
                for stream, t in enumerate((q, k, v)):
                    t[b, :, h] = randn(t.shape[1], stream).to(dtype)
    if kind == "outliers":
        for t in (q, k, v):
            t[:, ::97] *= 64.0
    return q, k, v


# ---- calling a kernel as a model does


@dataclass(slots=True)
class Opened:
    """A kernel opened on this card, called through its backend's call (`attention_ulysses.function`)."""

    ready: attention.Ready

    @property
    def sparse(self) -> bool:
        return self.ready.candidate.sparse

    def refusal(self, q: torch.Tensor, k: torch.Tensor) -> str:
        return self.ready.candidate.refusal(q, k, _OPTIONS)

    @contextlib.contextmanager
    def step(self, point: Point) -> Iterator[Callable[[], int]]:
        """A sparse kernel's call at a sparse point is a sparse step of a Sol site, whose sparse calls are counted."""
        if not (self.sparse and point.layout == "sparse"):
            yield lambda: 0
            return
        layout = AttentionLayout(live_tokens=point.lq, protected_prefix=_PREFIX, step=_SPARSE_STEP,
                                 dense_until_step=_SPARSE_STEP)  # fmt: skip
        site = attention_sol._SITE.set(attention_sol.Site("attention-bench", "attn"))
        try:
            with attention_scope(layout), attention_sol.observing() as counts:
                yield lambda: int(counts["sparse"])
        finally:
            attention_sol._SITE.reset(site)

    def prepare(self, q: torch.Tensor) -> None:
        """What a model's first call at a new length waits for off the request path: Sol's object for that length,
        and upstream's preprocess, on every route."""
        if not (self.sparse and self.ready.candidate.import_name == "sol_attn"):
            return
        store, index = kernel_cache.configured(), q.device.index or 0
        if store is None:
            return
        card = attention_sol._capability(q.device)
        if card is not None:
            found = attention_sol._job_for(q.shape[1], card)
            kernel_ready.wait(f"{attention_sol.KERNEL} {found.key.digest[:16]}",
                              partial(attention_sol._open, store, found, card))  # fmt: skip
        kernel_ready.wait(f"{attention_sol.KERNEL} preprocess cuda:{index}", partial(attention_sol._warm, index))

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.ready.function(q, k, v)


def _open(name: str, device: DeviceFacts) -> Opened:
    ready = attention.pinned(name, device)[0]
    if ready.candidate.sparse:
        attention.bind_dense(device)  # its undeclared dense rows, as a selection with a sparse site binds them
    return Opened(ready)


def _alive() -> bool:
    import torch

    try:
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - any error here is a context that no longer works
        return False
    return True


# ---- speed


def _timed(call: Callable[[], object], flush: torch.Tensor | None) -> float:
    """One call's GPU time in ms, by CUDA events, after an L2 flush when one is given."""
    import torch

    if flush is not None:
        flush.fill_(1)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    call()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end))


def _warm(call: Callable[[], object]) -> None:
    """Until two consecutive calls agree within 5%: compiles and first-call setup are not timed."""
    last = _timed(call, None)
    for _ in range(19):
        now = _timed(call, None)
        if abs(now - last) <= 0.05 * max(now, last):
            return
        last = now


def _quantiles(values: list[float]) -> list[float]:
    ordered = sorted(values)

    def at(p: float) -> float:
        return ordered[round(p * (len(ordered) - 1))]

    return [round(statistics.median(ordered), 5), round(at(0.1), 5), round(at(0.9), 5)]


def speed(names: list[str], points: list[Point], seed: int, ctx: Context) -> Speed:
    import torch

    device = _device()
    opened: dict[str, Opened] = {}
    absent: dict[str, str] = {}
    for name in dict.fromkeys(["sdpa", *names]):
        try:
            opened[name] = _open(name, device)
        except Exception as exc:  # noqa: BLE001 - a kernel that does not open is that kernel's finding
            absent[name] = f"{type(exc).__name__}: {exc}"[:600]
    flush = torch.empty(_FLUSH_BYTES, dtype=torch.uint8, device="cuda:0")
    l2 = int(torch.cuda.get_device_properties(0).L2_cache_size)
    results: list[PointResult] = []
    for at, point in enumerate(points):
        kind = "routable" if point.layout == "sparse" else "random"
        q, k, v = _operands(point, kind, seed + at)
        fits = sum(t.numel() * t.element_size() for t in (q, k, v)) <= l2
        refused: dict[str, str] = {}
        failed: dict[str, str] = {}
        live: list[str] = []
        for name, kernel in opened.items():
            if kernel.sparse != (point.layout == "sparse") and name != "sdpa":
                # A sparse kernel serves sparse steps only; a dense one at a sparse point is its dense twin's time,
                # which the generator scales by SDPA's ratio between the two.
                continue
            why = kernel.refusal(q, k)
            if why:
                refused[name] = why
                continue
            try:
                kernel.prepare(q)
                with kernel.step(point) as sparse_calls:
                    _warm(partial(kernel, q, k, v))
                    if kernel.sparse and not sparse_calls():
                        failed[name] = "its sparse path did not run: a dense reference served"
                        continue
            except Exception as exc:  # noqa: BLE001 - a kernel that raises here is that kernel's finding
                if not _alive():
                    raise Fault(f"fault in {name} at {point.id}: {type(exc).__name__}: {exc}"[:1500]) from exc
                failed[name] = f"{type(exc).__name__}: {exc}"[:600]
                continue
            live.append(name)
        samples: dict[str, list[list[float]]] = {name: [] for name in live}
        for turn in range(_ROUNDS if live else 0):
            # Interleaved, each round starting one kernel later: no kernel always runs first after a new point.
            for name in live[turn % len(live) :] + live[: turn % len(live)]:
                kernel = opened[name]
                got: list[float] = []
                with kernel.step(point):
                    while len(got) < _MAX_SAMPLES and (len(got) < _MIN_SAMPLES or sum(got) < _MIN_MS):
                        got.append(_timed(partial(kernel, q, k, v), flush if fits else None))
                samples[name].append(got)
        times: dict[str, Timed] = {}
        for name, rounds in samples.items():
            medians = [statistics.median(r) for r in rounds]
            middle = statistics.median(medians)
            times[name] = {
                "ms": _quantiles([s for r in rounds for s in r]),
                "spread": round((max(medians) - min(medians)) / middle, 5) if middle > 0 else 0.0,
            }
        results.append({
            "id": point.id, "batch": point.batch, "lq": point.lq, "lk": point.lk, "heads": point.heads,
            "dim": point.dim, "dtype": point.dtype, "layout": point.layout, "data": f"synthetic:{kind}",
            "times": times, "refused": refused, "failed": failed,
        })  # fmt: skip
        del q, k, v
        ctx.progress(at, len(points))
    return {"inventory": inventory(), "set": list(opened), "absent": absent, "points": results}


# ---- quality


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """FP32 attention, a head at a time: torch serves FP32 SDPA with its memory-efficient kernel, so no L x L matrix."""
    import torch

    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    for h in range(q.shape[2]):
        qh, kh, vh = (t[:, :, h : h + 1].transpose(1, 2).float() for t in (q, k, v))
        out[:, :, h : h + 1] = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh).transpose(1, 2)
    return out


def _per_head(got: torch.Tensor, want: torch.Tensor) -> list[float]:
    """Relative L2 per head over the batch, tokens and head dimension: ||O - O_ref|| / ||O_ref||."""
    diff = (got.float() - want.float()).pow(2).sum(dim=(0, 1, 3)).sqrt()
    norm = want.float().pow(2).sum(dim=(0, 1, 3)).sqrt().clamp_min(1e-30)
    return [float(x) for x in diff / norm]


def _cosine(got: torch.Tensor, want: torch.Tensor) -> float:
    import torch

    a, b = got.float().flatten(), want.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))


def _trivial(v: torch.Tensor, want: torch.Tensor) -> float:
    """The best answer a kernel can give without computing: V itself (equal lengths), V's mean, or zero."""
    found = [1.0, max(_per_head(v.float().mean(1, keepdim=True).expand_as(want), want))]
    if v.shape[1] == want.shape[1]:
        found.append(max(_per_head(v, want)))
    return min(found)


def _sparse_reference(device: DeviceFacts) -> str:
    """The class's 16-bit sparse kernel, which a quantized sparse kernel on the same route is held against."""
    row = attention.device_row(device)
    found = [c.name for c in attention.KERNELS if c.sparse and c.exact and not attention._off_row(c, row)]
    return found[0] if found else ""


def quality(name: str, seed: int, ctx: Context) -> Quality:
    import torch

    device = _device()
    kernel = _open(name, device)
    reference = _sparse_reference(device) if kernel.sparse and not kernel.ready.candidate.exact else ""
    ref16 = _open(reference, device) if reference else None
    errors: list[Error] = []
    skipped: dict[str, str] = {}
    sites = [s for s in SITES if (s.point.layout == "sparse") == kernel.sparse]
    for at, site in enumerate(sites):
        q, k, v = _operands(site.point, site.kind, seed + at)
        why = kernel.refusal(q, k)
        if why:
            skipped[site.name] = why
            continue
        want = _reference(q, k, v)
        kernel.prepare(q)
        with kernel.step(site.point) as sparse_calls:
            got = kernel(q, k, v)
            calls = sparse_calls()
        heads = _per_head(got, want)
        error: Error = {
            "site": site.name, "level": site.level, "kind": site.kind, "finite": bool(torch.isfinite(got).all()),
            "worst": round(max(heads), 6), "mean": round(sum(heads) / len(heads), 6),
            "cosine": round(_cosine(got, want), 6), "trivial": round(_trivial(v, want), 6),
        }  # fmt: skip
        if kernel.sparse:
            error["sparse_calls"] = calls
        if ref16 is not None:
            ref16.prepare(q)
            with ref16.step(site.point):
                base = ref16(q, k, v)
            error["ref16"] = round(max(_per_head(got, base)), 6)
            error["ref16_fp32"] = round(max(_per_head(base, want)), 6)
        errors.append(error)
        del q, k, v, want, got
        ctx.progress(at, len(sites))
    return {
        "inventory": inventory(), "kernel": name, "level": kernel.ready.candidate.level,
        "reference": reference, "corpus": "synthetic", "errors": errors, "skipped": skipped,
    }  # fmt: skip


# ---- running off the request path


def off_request[T](work: Callable[[], T]) -> T:
    """WORK on a thread of its own, as `kernel_ready`'s background openers run: a request never compiles or probes,
    so the kernels this opens are not the request path blocking. Its exception is raised here."""
    import torch

    found: list[T] = []
    failed: list[BaseException] = []

    def body() -> None:
        try:
            with torch.no_grad(), kernel_ready.EXECUTION[0]():
                found.append(work())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            failed.append(exc)

    thread = threading.Thread(target=body, name="attention-bench")
    thread.start()
    thread.join()
    if failed:
        raise failed[0]
    return found[0]


