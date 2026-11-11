"""The GEMM measurements (xs-177; `cozy_runtime.bench`'s built-in calls): every kernel this card can serve a lane's
linear with, timed at our packages' real shapes beside the pick the Runtime's rule makes today (`int8_linear`,
`fp8_linear`, `nvfp4_linear`, `prebuilt`). Nothing here ranks: the release gate's `suites/gemm` reads the JSON.

- `speed`: per shape, every candidate of every lane in one interleaved set, three rounds. A `gemm:` candidate multiplies
  an operand already quantized (as a block's shared quantize hands it); a `linear:` one starts from the bf16 rows.
  Each tile's output is held to the rule's, bit for bit, on its first and last rows (`same_bits`).
- `tiles`: two of the Runtime's Triton kernels (nvfp4's weight decode and its GEMV) over tile settings that leave every
  output bit as it is: does one card of a class want another tile than its sibling?
Operands are random and seeded; the pod's clocks are TensorD's NVML reads of the call, in its record."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import ModuleType
from typing import TypedDict

import torch

import cozy_runtime
from cozy_runtime import _build_provenance
from cozy_runtime.author import Context
from cozy_runtime.accel.kernels import Launch
from cozy_runtime.internal import arch, attention, int8_linear, nvfp4_linear, prebuilt, triton_kernels
from cozy_runtime.internal.bench.attention import _device, _quantiles, _timed, _warm
from cozy_runtime.internal.encoding.formats import CONVROT_GROUP

BF16 = torch.bfloat16
_ROUNDS, _MIN_SAMPLES, _MAX_SAMPLES, _MIN_MS = 3, 3, 100, 60.0
_FLUSH_BYTES = 256 << 20
#: Rows held bit for bit at each end of an output: a tile's sums are its own, whatever the row.
_EDGE = 2048
LANES = ("bf16", "int8", "fp8", "nvfp4", "w4a8")


@dataclass(frozen=True, slots=True)
class Shape:
    """One linear of a model: `[m, k] @ [n, k].T`, `calls` times in one request of its bench cell."""

    model: str
    site: str
    m: int
    n: int
    k: int
    calls: int

    @property
    def id(self) -> str:
        return f"{self.model}-{self.site}"


#: The bench cells' requests (release-gate suites/bench/cells.tsv). H3 Turbo 15 s: 50 blocks x 8 steps on one GPU
#: (111,598 rows) and on one of four (27,900). Qwen-Image 2.1 at 4 MP: 32 blocks x 40 steps x 2 (CFG). Klein: 33
#: blocks x 16 steps, text to image (4,608 rows) and an edit (8,704). SDXL 1024², CFG as batch 2, 20 steps: 10
#: transformer layers at 64² (six C x C linears each, cross-attention's q and out included), 60 at 32².
SHAPES: tuple[Shape, ...] = (
    *(
        Shape(model, site, m, n, k, 400 * c)
        for model, m in (("h3", 111598), ("h3-g4", 27900))
        for site, n, k, c in (("qkv", 7168, 5376, 3), ("out", 5376, 7168, 1), ("up", 28672, 5376, 1), ("down", 5376, 14336, 1))
    ),
    *(Shape("qwen4mp", site, 16640, n, k, 2560 * c) for site, n, k, c in (("qkvo", 4096, 4096, 4), ("up", 12288, 4096, 2), ("down", 4096, 12288, 1))),
    *(
        Shape(model, site, m, n, k, 528 * c)
        for model, m in (("klein", 4608), ("klein-edit", 8704))
        for site, n, k, c in (("qkvo", 3072, 3072, 4), ("up", 18432, 3072, 1), ("down", 3072, 9216, 1))
    ),
    Shape("sdxl64", "attn", 8192, 640, 640, 1200),
    Shape("sdxl64", "up", 8192, 5120, 640, 200),
    Shape("sdxl64", "down", 8192, 640, 2560, 200),
    Shape("sdxl32", "attn", 2048, 1280, 1280, 7200),
    Shape("sdxl32", "up", 2048, 10240, 1280, 1200),
    Shape("sdxl32", "down", 2048, 1280, 5120, 1200),
)  # fmt: skip

Call = Callable[[], object]


class Timed(TypedDict):
    """ms per call (median, p10, p90 over every call) and the spread of the three round medians over their median."""

    ms: list[float]
    spread: float


class Inventory(TypedDict):
    card: str
    sm: int
    klass: str
    sms: int
    l2: int
    runtime: str
    commit: str
    torch: str
    cuda: str
    driver: str
    family: str
    cutlass_fp8: bool
    library: str
    kitchen: str
    triton: str
    fp4: bool


class ShapeResult(TypedDict):
    id: str
    model: str
    site: str
    m: int
    n: int
    k: int
    calls: int
    #: lane -> {"gemm": candidate, "linear": candidate}: what the Runtime's rule serves today
    rules: dict[str, dict[str, str]]
    times: dict[str, Timed]
    #: candidate -> whether its output equals the rule's (tiles of one lane)
    same_bits: dict[str, bool]
    refused: dict[str, str]


class Speed(TypedDict):
    inventory: Inventory
    shapes: list[ShapeResult]


# ---- the card


@dataclass(slots=True)
class Card:
    ops: prebuilt.Prebuilt | None
    kitchen: ModuleType | None
    launch: Launch | None
    row: arch.Row
    inventory: Inventory


def _card() -> Card:
    device = _device()
    row = arch.row(device.sm)
    ops, why = prebuilt.library(torch, 0, device.sm)
    kitchen, kitchen_why = None, ""
    try:
        found = int8_linear.kitchen_module(torch, 0, True)
        kitchen = None if found is None else found[0]
        kitchen_why = "" if found is not None else "no build"
    except Exception as exc:  # noqa: BLE001 - a missing backend is this card's finding
        kitchen_why = f"{type(exc).__name__}: {exc}"[:300]
    launch: Launch | None = None
    triton_why = "serves"
    try:
        launch = triton_kernels.ready(nvfp4_linear.KERNELS, True, device="cuda:0")
    except Exception as exc:  # noqa: BLE001 - no Triton here: the decode candidates leave
        triton_why = f"{type(exc).__name__}: {exc}"[:300]
    props = torch.cuda.get_device_properties(0)
    inventory: Inventory = {
        "card": device.name, "sm": device.sm,
        "klass": arch.klass(attention.device_row(device)),
        "sms": int(props.multi_processor_count), "l2": int(props.L2_cache_size),
        "runtime": cozy_runtime.__version__, "commit": _build_provenance.COMMIT, "torch": torch.__version__,
        "cuda": str(torch.version.cuda), "driver": device.driver, "family": row.family,
        "cutlass_fp8": bool(ops is not None and ops.cutlass_fp8), "library": "serves" if ops is not None else why,
        "kitchen": "serves" if kitchen is not None else kitchen_why, "triton": triton_why, "fp4": row.fp4,
    }  # fmt: skip
    return Card(ops, kitchen, launch, row, inventory)


# ---- each lane's candidates


def _rand(g: torch.Generator, *shape: int, scale: float = 1.0, add: float = 0.0) -> torch.Tensor:
    return torch.rand(*shape, generator=g, device="cuda:0", dtype=torch.float32).mul_(scale).add_(add)


def _bf16(x: torch.Tensor, s: Shape, g: torch.Generator, card: Card) -> tuple[dict[str, Call], dict[str, str]]:
    w = torch.randn(s.n, s.k, generator=g, device="cuda:0", dtype=torch.float32).mul_(0.02).to(BF16)
    default = torch.backends.cuda.preferred_blas_library()

    def under(name: str) -> torch.Tensor:
        torch.backends.cuda.preferred_blas_library(name)
        try:
            return torch.nn.functional.linear(x, w)
        finally:
            torch.backends.cuda.preferred_blas_library(default)

    rule = f"bf16-gemm:{default.name.lower()}"
    calls: dict[str, Call] = {rule: partial(torch.nn.functional.linear, x, w)}
    calls |= {f"bf16-gemm:{name}": partial(under, name) for name in ("cublas", "cublaslt") if f"bf16-gemm:{name}" != rule}
    return calls, {"gemm": rule, "linear": rule}


def _tiles(name: str, gemm: Callable[..., torch.Tensor], *args: object) -> dict[str, Call]:
    """`gemm(*args, config)` for every config a family's table could hold: one it does not raises at its first call."""
    return {f"{name}:c{c}": partial(gemm, *args, c) for c in range(8)}


def _tile(name: str) -> bool:
    """Whether a candidate is one explicit tile (`...:c3`)."""
    tail = name.rpartition(":")[2]
    return tail[:1] == "c" and tail[1:].isdigit()


def _int8(x: torch.Tensor, s: Shape, g: torch.Generator, card: Card) -> tuple[dict[str, Call], dict[str, str]]:
    ops = card.ops
    data = torch.randint(-127, 128, (s.n, s.k), generator=g, device="cuda:0", dtype=torch.int8)
    scale = _rand(g, s.n, scale=1e-3, add=5e-4)
    calls: dict[str, Call] = {}
    if ops is not None:
        q, qs = ops.int8_quantize(x)
        calls["quantize:int8"] = partial(ops.int8_quantize, x)
        calls["int8-gemm:prebuilt"] = partial(ops.int8_gemm, q, qs, data, scale, None, BF16, -1)
        calls |= _tiles("int8-gemm", ops.int8_gemm, q, qs, data, scale, None, BF16)
        calls["int8-gemm:cublaslt"] = partial(int8_linear._product, torch, q, qs, data, scale, BF16)
        calls["int8-linear:prebuilt"] = partial(ops.int8_linear, x, data, scale, None, BF16)
    if card.kitchen is not None:
        kitchen = card.kitchen
        calls["int8-linear:kitchen"] = lambda: kitchen.int8_linear(
            x, data, scale, None, BF16, convrot=True, convrot_groupsize=CONVROT_GROUP
        )
    gemm = "int8-gemm:prebuilt" if ops is not None else ""
    linear = "int8-linear:prebuilt" if ops is not None else "int8-linear:kitchen" if card.kitchen else ""
    return calls, {"gemm": gemm, "linear": linear}


def _fp8(x: torch.Tensor, s: Shape, g: torch.Generator, card: Card) -> tuple[dict[str, Call], dict[str, str]]:
    ops = card.ops
    raw = torch.randint(0, 256, (s.n, s.k), generator=g, device="cuda:0", dtype=torch.uint8)
    data = (raw & 0xF7).view(torch.float8_e4m3fn)  # signed, no NaN
    scale = _rand(g, s.n, add=0.5)
    if ops is not None:
        q, qs = ops.fp8_quantize(x)
    else:
        from cozy_runtime.internal import fp8_linear

        q, qs = fp8_linear.eager_quantize(torch, x)

    def scaled_mm() -> torch.Tensor:
        return torch._scaled_mm(q, data.t(), scale_a=qs.reshape(-1, 1), scale_b=scale.reshape(1, -1),
                                out_dtype=BF16, use_fast_accum=True)  # fmt: skip

    calls: dict[str, Call] = {"fp8-gemm:cublaslt": scaled_mm}
    if ops is not None:
        calls["quantize:fp8"] = partial(ops.fp8_quantize, x)
    if ops is not None and ops.cutlass_fp8:
        calls["fp8-gemm:prebuilt"] = partial(ops.fp8_gemm, q, qs, data, scale, None, BF16, -1)
        calls |= _tiles("fp8-gemm", ops.fp8_gemm, q, qs, data, scale, None, BF16)
        calls["fp8-linear:prebuilt"] = partial(ops.fp8_linear, x, data, scale, None, BF16)
        return calls, {"gemm": "fp8-gemm:prebuilt", "linear": "fp8-linear:prebuilt"}
    if ops is not None:

        def linear() -> torch.Tensor:
            p, ps = ops.fp8_quantize(x)
            return torch._scaled_mm(p, data.t(), scale_a=ps.reshape(-1, 1), scale_b=scale.reshape(1, -1),
                                    out_dtype=BF16, use_fast_accum=True)  # fmt: skip

        calls["fp8-linear:cublaslt"] = linear
    return calls, {"gemm": "fp8-gemm:cublaslt", "linear": "fp8-linear:cublaslt" if ops is not None else ""}


def _nvfp4(x: torch.Tensor, s: Shape, g: torch.Generator, card: Card) -> tuple[dict[str, Call], dict[str, str]]:
    weight, _, blocked, second = nvfp4_linear._probe_weight(torch, g, "cuda:0", s.n, s.k)
    calls: dict[str, Call] = {}
    launch = card.launch
    if launch is not None:
        decode = partial(nvfp4_linear._decoded, torch, launch, weight, blocked, second)
        calls["nvfp4-decode"] = decode
        calls["nvfp4-linear:w4a16"] = lambda: torch.nn.functional.linear(x, decode())
    kitchen = card.kitchen
    if card.row.fp4 and kitchen is not None:
        payload, block, tensor_scale = nvfp4_linear._quantized(torch, kitchen, x)
        alpha = tensor_scale * second
        calls["quantize:nvfp4"] = partial(nvfp4_linear._quantized, torch, kitchen, x)
        calls["nvfp4-gemm:kitchen"] = lambda: kitchen.scaled_mm_nvfp4(
            payload, weight, tensor_scale, second, block, blocked, out_dtype=BF16
        )
        fp4 = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4 is not None:

            def scaled_mm() -> torch.Tensor:
                out = torch._scaled_mm(payload.view(fp4), weight.view(fp4).t(),
                                       scale_a=block.view(torch.float8_e4m3fn), scale_b=blocked, out_dtype=BF16)  # fmt: skip
                return out.mul_(alpha)

            calls["nvfp4-gemm:cublaslt"] = scaled_mm

        def w4a4() -> torch.Tensor:
            p, b, t = nvfp4_linear._quantized(torch, kitchen, x)
            return kitchen.scaled_mm_nvfp4(p, weight, t, second, b, blocked, out_dtype=BF16)

        calls["nvfp4-linear:kitchen"] = w4a4
        return calls, {"gemm": "nvfp4-gemm:kitchen", "linear": "nvfp4-linear:kitchen"}
    return calls, {"gemm": "", "linear": "nvfp4-linear:w4a16" if launch is not None else ""}


def _w4a8(x: torch.Tensor, s: Shape, g: torch.Generator, card: Card) -> tuple[dict[str, Call], dict[str, str]]:
    ops = card.ops
    if ops is None:
        return {}, {"gemm": "", "linear": ""}
    codes = torch.randint(0, 256, (s.n, s.k // 2), generator=g, device="cuda:0", dtype=torch.uint8)
    group = _rand(g, s.n, s.k // 16, scale=4.0, add=0.25).to(torch.float8_e4m3fn)
    w = prebuilt.W4a8(codes, group, _rand(g, s.n, scale=1e-3, add=5e-4), None)
    q, qs = ops.int8_quantize(x)
    calls: dict[str, Call] = {"w4a8-gemm:prebuilt": partial(ops.w4a8_gemm, q, qs, w, None, BF16, -1)}
    calls |= _tiles("w4a8-gemm", ops.w4a8_gemm, q, qs, w, None, BF16)
    calls["w4a8-linear:prebuilt"] = partial(ops.w4a8_linear, x, w, None, BF16)
    return calls, {"gemm": "w4a8-gemm:prebuilt", "linear": "w4a8-linear:prebuilt"}


BUILDERS: dict[str, Callable[[torch.Tensor, Shape, torch.Generator, Card], tuple[dict[str, Call], dict[str, str]]]] = {
    "bf16": _bf16, "int8": _int8, "fp8": _fp8, "nvfp4": _nvfp4, "w4a8": _w4a8,
}  # fmt: skip


def _admits(lane: str, s: Shape, card: Card) -> str:
    """Why `lane` cannot take this shape on this card, or ''."""
    if lane in ("int8", "w4a8") and (s.k % CONVROT_GROUP or not card.row.int8):
        return f"K {s.k} is not whole {CONVROT_GROUP}-wide ConvRot groups" if card.row.int8 else "no int8 tensor cores"
    if lane == "fp8" and not card.row.fp8:
        return "no fp8 tensor cores"
    if lane == "nvfp4" and s.k % 16:
        return f"K {s.k} is not whole 16-blocks"
    return ""


# ---- timing


def _edges(out: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not isinstance(out, torch.Tensor) or out.dim() != 2:
        return None
    return out[:_EDGE].clone(), out[-_EDGE:].clone()


def _interleaved(calls: dict[str, Call], flush: torch.Tensor | None) -> dict[str, Timed]:
    """Each call warmed, then three rounds, each starting one call later, at least `_MIN_MS` per call a round."""
    for call in calls.values():
        _warm(call)
    names = list(calls)
    samples: dict[str, list[list[float]]] = {name: [] for name in names}
    for turn in range(_ROUNDS):
        for name in names[turn % len(names) :] + names[: turn % len(names)]:
            got: list[float] = []
            while len(got) < _MAX_SAMPLES and (len(got) < _MIN_SAMPLES or sum(got) < _MIN_MS):
                got.append(_timed(calls[name], flush))
            samples[name].append(got)
    times: dict[str, Timed] = {}
    for name, rounds in samples.items():
        medians = [statistics.median(r) for r in rounds]
        middle = statistics.median(medians)
        spread = round((max(medians) - min(medians)) / middle, 5) if middle > 0 else 0.0
        times[name] = {"ms": _quantiles([v for r in rounds for v in r]), "spread": spread}
    return times


def speed(lanes: list[str], wanted: list[str], seed: int, ctx: Context) -> Speed:
    card = _card()
    flush = torch.empty(_FLUSH_BYTES, dtype=torch.uint8, device="cuda:0")
    chosen = [s for s in SHAPES if not wanted or s.id in wanted or s.model in wanted]
    results: list[ShapeResult] = []
    for at, s in enumerate(chosen):
        g = torch.Generator(device="cuda:0").manual_seed(seed + at)
        x = torch.randn(s.m, s.k, generator=g, device="cuda:0", dtype=torch.float32).to(BF16)
        calls: dict[str, Call] = {}
        rules: dict[str, dict[str, str]] = {}
        refused: dict[str, str] = {}
        for lane in lanes or LANES:
            why = _admits(lane, s, card)
            if why:
                refused[lane] = why
                continue
            try:
                found, rules[lane] = BUILDERS[lane](x, s, g, card)
            except Exception as exc:  # noqa: BLE001 - a lane that cannot build here is this card's finding
                refused[lane] = f"{type(exc).__name__}: {exc}"[:400]
                continue
            calls |= found
        # The first call of each: a tile the card's table does not hold, or a shape it refuses, leaves the set.
        live: dict[str, Call] = {}
        reference: dict[str, tuple[torch.Tensor, torch.Tensor] | None] = {}
        same: dict[str, bool] = {}
        for name, call in calls.items():
            try:
                out = call()
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001 - recorded, and the set goes on without it
                refused[name] = f"{type(exc).__name__}: {exc}"[:300]
                continue
            live[name] = call
            lane = name.split("-", 1)[0]
            rule = rules.get(lane, {}).get("gemm", "")
            if name == rule:
                reference[lane] = _edges(out)
            elif _tile(name) and reference.get(lane) is not None:
                got, want = _edges(out), reference[lane]
                same[name] = got is not None and want is not None and all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
            del out
        weight_bytes = s.n * s.k * 2
        fits = weight_bytes <= int(torch.cuda.get_device_properties(0).L2_cache_size)
        times = _interleaved(live, flush if fits else None)
        results.append({
            "id": s.id, "model": s.model, "site": s.site, "m": s.m, "n": s.n, "k": s.k, "calls": s.calls,
            "rules": rules, "times": times, "same_bits": same, "refused": refused,
        })  # fmt: skip
        del x, calls, live, reference
        torch.cuda.empty_cache()
        ctx.progress(at, len(chosen))
    return {"inventory": card.inventory, "shapes": results}


# ---- Triton tiles that keep every bit


class TileResult(TypedDict):
    kernel: str
    n: int
    k: int
    m: int
    default: str
    times: dict[str, Timed]
    same_bits: dict[str, bool]
    refused: dict[str, str]


class Tiles(TypedDict):
    inventory: Inventory
    kernels: list[TileResult]


#: nvfp4 weights (N, K) of the shapes above.
TILE_WEIGHTS = ((28672, 5376), (5376, 14336), (7168, 5376), (12288, 4096), (4096, 12288), (18432, 3072), (3072, 9216))


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def tiles(seed: int, ctx: Context) -> Tiles:
    """nvfp4's `_dequant` over BLOCK_N x BLOCK_K x warps (an elementwise decode: every tile writes the same bits), and
    its `_gemv` over BLOCK_N x warps at 1 and 16 rows (each row's K order is BLOCK_W's, held at the Runtime's 32)."""
    card = _card()
    flush = torch.empty(_FLUSH_BYTES, dtype=torch.uint8, device="cuda:0")
    found: list[TileResult] = []
    total = len(TILE_WEIGHTS) * 3
    for at, (n, k) in enumerate(TILE_WEIGHTS):
        g = torch.Generator(device="cuda:0").manual_seed(seed + at)
        weight, _, blocked, second = nvfp4_linear._probe_weight(torch, g, "cuda:0", n, k)
        scales, col_tiles = blocked.view(torch.uint8), int(blocked.shape[1]) // 4
        decoded = torch.empty(n, k, dtype=BF16, device="cuda:0")

        def dequant(bn: int, bk: int, warps: int) -> torch.Tensor:
            nvfp4_linear._dequant[(_cdiv(n, bn), _cdiv(k, bk))](
                weight, scales, second, decoded, n, k, col_tiles, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps
            )
            return decoded

        configs = {f"n{bn}-k{bk}-w{w}": partial(dequant, bn, bk, w) for bn in (32, 64, 128) for bk in (64, 128, 256) for w in (2, 4, 8)}
        found.append(_tile_set("dequant", n, k, 0, "n64-k128-w4", configs, flush))
        ctx.progress(at * 3, total)
        words = weight.view(torch.int32)
        for i, m in enumerate((1, 16)):
            x = torch.randn(m, k, generator=g, device="cuda:0", dtype=torch.float32).to(BF16)
            out = torch.empty(m, n, dtype=BF16, device="cuda:0")
            span = _cdiv(k, 8 * nvfp4_linear.GEMV_W) * 8 * nvfp4_linear.GEMV_W

            def gemv(bn: int, warps: int, x: torch.Tensor = x, out: torch.Tensor = out, m: int = m) -> torch.Tensor:
                nvfp4_linear._gemv[(_cdiv(n, bn), 1, 1)](
                    x, words, scales, second, out, m, n, k, col_tiles, span,
                    BLOCK_N=bn, BLOCK_W=nvfp4_linear.GEMV_W, SPLIT=0, num_warps=warps,
                )  # fmt: skip
                return out

            configs = {f"n{bn}-w{w}": partial(gemv, bn, w) for bn in (32, 64, 128) for w in (4, 8)}
            found.append(_tile_set("gemv", n, k, m, f"n{nvfp4_linear.GEMV_N}-w8", configs, flush))
            ctx.progress(at * 3 + 1 + i, total)
        del weight, blocked, decoded
        torch.cuda.empty_cache()
    return {"inventory": card.inventory, "kernels": found}


def _tile_set(kernel: str, n: int, k: int, m: int, default: str, configs: dict[str, Call], flush: torch.Tensor) -> TileResult:
    live: dict[str, Call] = {}
    refused: dict[str, str] = {}
    same: dict[str, bool] = {}
    want: torch.Tensor | None = None
    for name in [default, *(c for c in configs if c != default)]:
        try:
            got = configs[name]()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - a tile this kernel cannot take here is recorded
            refused[name] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        live[name] = configs[name]
        if isinstance(got, torch.Tensor):
            if want is None:
                want = got.clone()
            else:
                same[name] = torch.equal(got, want)
    return {"kernel": kernel, "n": n, "k": k, "m": m, "default": default,
            "times": _interleaved(live, flush), "same_bits": same, "refused": refused}  # fmt: skip
