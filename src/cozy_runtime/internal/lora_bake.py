"""Bake a LoRA into stored weights, in place, and the two-GEMM per-step update.

A bake is decode -> add `alpha * up @ down` -> round back to the stored representation, one
row chunk at a time, so nothing larger than a chunk exists twice. Plain floats round once to
their dtype. `fp8-rowwise/1` and `mxfp8/1` re-encode E4M3; the rounding is chosen because a
delta below half a grid step vanishes under nearest rounding. `Stats` is what a quality gate
reads: how much of the delta survived and how much rounding error the bake added.

On a GPU, `bake` is one fused Triton kernel per weight (`_bake_kernel`): it reads each stored
byte once, adds the rank-r product computed in its tile, rounds and writes the byte back, with
counter-based noise (one stream per weight, by element index). The kernel is compiled and
checked in a child process (`warm`); this process only loads it from Triton's cache. The eager
row-chunk functions are its reference on CPU, what `Stats` measures, and what a GPU bakes with
when the child failed.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from types import ModuleType
from typing import IO, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from torch import Generator, Tensor, device as Device

from cozy_runtime.internal.encoding import E4M3_MAX, E8M0_BIAS, MX_BLOCK

try:
    import triton
    import triton.language as tl
except ImportError:  # a host without Triton bakes eagerly
    triton = None

Strategy = Literal["bake", "per_step"]

#: Names this module's bake to other executors: the same factors and scales baked by the same
#: rule are the same bytes (one seed per weight). Changed whenever the bytes a bake writes do.
VARIANT = "cozy.lora-bake/2"
Rounding = Literal["nearest", "stochastic", "feedback"]
ROUNDINGS: tuple[Rounding, ...] = ("nearest", "stochastic", "feedback")
#: A rowwise row's scale after a bake: unchanged, what its new maximum needs, or that moved
#: off the old grid so every element rounds afresh.
Grid = Literal["keep", "refit", "regrid"]
GRIDS: tuple[Grid, ...] = ("keep", "refit", "regrid")
#: Elements decoded at once (8 MiB of f32): a bake's working memory stays well below a
#: block's activations, so it runs inside the room a stage already has.
CHUNK_ELEMENTS = 1 << 21
#: Columns an error-feedback run carries its rounding error across (an mxfp8 block).
FEEDBACK_RUN = MX_BLOCK
ROW_SCALE_FLOOR = 1e-12
#: Halfway to the step above 448 that E4M3 lacks: past it a value is clamped, not rounded.
SATURATION = 464.0


@dataclass(slots=True)
class Stats:
    """Sums over every baked element, in the weights' own units."""

    layers: int = 0
    elements: int = 0
    #: |W + delta|^2, |delta|^2, |stored - (W + delta)|^2 and <stored - W, delta>
    weight_energy: float = 0.0
    delta_energy: float = 0.0
    error_energy: float = 0.0
    kept: float = 0.0
    #: the delta's action on its own inputs, |delta @ down.T|^2, and the rounding error's there
    effect_energy: float = 0.0
    effect_error: float = 0.0
    saturated: int = 0
    seconds: float = 0.0

    @property
    def retention(self) -> float:
        """The delta's own direction in what was stored: 1.0 keeps all of it, 0.0 none."""
        return self.kept / self.delta_energy if self.delta_energy else 1.0

    @property
    def sqnr_db(self) -> float:
        """Baked weight against the rounding error the bake added."""
        return _db(self.weight_energy, self.error_energy)

    @property
    def effect_snr_db(self) -> float:
        """What the LoRA does to the inputs it reads, against what rounding did to them."""
        return _db(self.effect_energy, self.effect_error)

    def add(self, other: Stats) -> None:
        for name in self.__slots__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def document(self) -> dict[str, float | int]:
        return {
            "layers": self.layers,
            "elements": self.elements,
            "retention": self.retention,
            "sqnr_db": self.sqnr_db,
            "effect_snr_db": self.effect_snr_db,
            "delta_to_weight_db": _db(self.delta_energy, self.weight_energy),
            "saturated": self.saturated,
            "seconds": self.seconds,
        }


def _db(signal: float, noise: float) -> float:
    if noise <= 0:
        return math.inf
    return 10 * math.log10(signal / noise) if signal > 0 else -math.inf


def _magnitudes(torch: ModuleType, device: Device) -> Tensor:
    """E4M3's 127 magnitudes by code: man * 2^-9 below exponent 1, else (1 + man/8) * 2^(exp-7)."""
    code = torch.arange(127, device=device)
    exponent, mantissa = code >> 3, (code & 7).to(torch.float32)
    normal = torch.ldexp(1 + mantissa / 8, exponent - 7)
    return torch.where(exponent == 0, mantissa * 2.0**-9, normal)


def decode_e4m3(torch: ModuleType, data: Tensor) -> Tensor:
    decoded = data.view(torch.uint8).view(torch.float8_e4m3fn).float()
    # This codec saturates the two reserved NaN codes to their signed maxima.
    # Keep their original signs (and signed zero) after the native conversion.
    fixed = torch.nan_to_num(decoded, nan=E4M3_MAX)
    return fixed.copysign_(decoded)


def _floor_code(torch: ModuleType, mags: Tensor, magnitude: Tensor) -> tuple[Tensor, Tensor]:
    """The code at or below each magnitude, and how far towards the next one it sits (0..1)."""
    low = (torch.searchsorted(mags, magnitude, right=True) - 1).clamp(0, 126)
    high = (low + 1).clamp(max=126)
    gap = mags[high] - mags[low]
    return low, torch.where(gap > 0, (magnitude - mags[low]) / gap, torch.zeros_like(gap))


def _nearest(torch: ModuleType, mags: Tensor, values: Tensor) -> Tensor:
    """Signed values -> E4M3 bytes, ties to the even code, saturating at 448."""
    low, fraction = _floor_code(torch, mags, values.abs().clamp(max=E4M3_MAX))
    up = (fraction > 0.5) | ((fraction == 0.5) & (low % 2 == 1))
    return ((low + up) | ((values < 0) << 7)).to(torch.uint8)


def encode_e4m3(
    torch: ModuleType,
    values: Tensor,
    rounding: Rounding = "nearest",
    generator: Generator | None = None,
    *,
    magnitudes: Tensor | None = None,
) -> Tensor:
    """`[rows, cols]` f32 in payload units -> E4M3 bytes.

    `stochastic` rounds up with the probability of the distance covered, so the expected
    stored value is the target. `feedback` rounds to nearest and carries each element's
    error into the next one of its `FEEDBACK_RUN`, so a run's sum is kept without a random
    draw (the same bytes on every bake).
    """
    mags = _magnitudes(torch, values.device) if magnitudes is None else magnitudes
    if rounding == "nearest":
        return _nearest(torch, mags, values)
    if rounding == "stochastic":
        low, fraction = _floor_code(torch, mags, values.abs().clamp(max=E4M3_MAX))
        draw = torch.rand(fraction.shape, device=values.device, generator=generator)
        return ((low + (draw < fraction)) | ((values < 0) << 7)).to(torch.uint8)
    rows, cols = values.shape
    pad = -cols % FEEDBACK_RUN
    work = torch.nn.functional.pad(values, (0, pad)) if pad else values
    runs = work.reshape(rows, -1, FEEDBACK_RUN)
    out = torch.empty(runs.shape, dtype=torch.uint8, device=values.device)
    carry = torch.zeros(runs.shape[:2], dtype=values.dtype, device=values.device)
    for k in range(FEEDBACK_RUN):
        want = runs[..., k] + carry
        code = _nearest(torch, mags, want)
        out[..., k] = code
        stored = mags[(code & 0x7F).long()]
        carry = want - torch.where(code >= 0x80, -stored, stored)
    return out.reshape(rows, -1)[:, :cols]


def _chunks(rows: int, cols: int) -> list[tuple[int, int]]:
    step = max(1, CHUNK_ELEMENTS // max(cols, 1))
    return [(start, min(start + step, rows)) for start in range(0, rows, step)]


def _delta(up: Tensor, down_fp32: Tensor, alpha: float, start: int, stop: int) -> Tensor:
    return (up[start:stop].float() @ down_fp32) * alpha


def _account(
    stats: Stats | None,
    base: Tensor,
    delta: Tensor,
    stored: Tensor,
    saturated: Tensor,
    down: Tensor,
) -> None:
    if stats is None:
        return
    target = base + delta
    basis = down.float().t()
    stats.effect_energy += float((delta @ basis).double().square().sum())
    stats.effect_error += float(((stored - target) @ basis).double().square().sum())
    stats.elements += target.numel()
    stats.weight_energy += float(target.double().square().sum())
    stats.delta_energy += float(delta.double().square().sum())
    stats.error_energy += float((stored - target).double().square().sum())
    stats.kept += float(((stored - base).double() * delta.double()).sum())
    stats.saturated += int(saturated)


def bake_plain(
    torch: ModuleType,
    weight: Tensor,
    down: Tensor,
    up: Tensor,
    alpha: float,
    *,
    rounding: Rounding = "nearest",
    generator: Generator | None = None,
    stats: Stats | None = None,
) -> None:
    """`weight += alpha * up @ down`, computed in f32 and rounded once to its dtype.

    A bf16 weight has 8 significant bits, so a small delta vanishes under nearest rounding
    there too; `stochastic` keeps it in expectation. Other dtypes round to nearest.
    """
    began = time.perf_counter()
    # The immutable rank-by-input factor is shared by every row chunk of this bake.
    # Keep only this call's FP32 copy, never a cache across weights or invocations.
    down_fp32 = down.float()
    for start, stop in _chunks(*weight.shape):
        base = weight[start:stop].to(torch.float32, copy=True)
        delta = _delta(up, down_fp32, alpha, start, stop)
        target = base + delta
        if rounding == "stochastic" and weight.dtype == torch.bfloat16:
            target = _bf16_either(torch, target, generator)
        weight[start:stop].copy_(target)
        if stats is not None:
            _account(stats, base, delta, weight[start:stop].float(), 0, down_fp32)
    _finish(torch, weight, stats, began)


def _bf16_either(torch: ModuleType, values: Tensor, generator: Generator | None) -> Tensor:
    """f32 -> one of its two bf16 neighbours (as f32), the farther one with the probability
    of the distance covered."""
    toward_zero = values.contiguous().view(torch.int32) & -65536
    low, high = toward_zero.view(torch.float32), (toward_zero + 65536).view(torch.float32)
    gap = high - low
    fraction = torch.where(torch.isfinite(gap) & (gap != 0), (values - low) / gap, 0.0)
    draw = torch.rand(fraction.shape, device=values.device, generator=generator)
    return torch.where(draw < fraction, high, low)


def bake_rowwise(
    torch: ModuleType,
    data: Tensor,
    scale: Tensor,
    down: Tensor,
    up: Tensor,
    alpha: float,
    *,
    rounding: Rounding = "nearest",
    grid: Grid = "refit",
    generator: Generator | None = None,
    stats: Stats | None = None,
) -> None:
    """Bake into `fp8-rowwise/1`: `data` `[out, in]` E4M3, `scale` one f32 per output row.

    The stored base sits exactly on its row's grid, so with `keep` (or a `refit` that barely
    moves the scale) nearest rounding returns every element whose delta is under half a
    step to where it was. `regrid` raises the refitted scale by 1/32 to 3/32, which moves
    each element 0.25 to 1.5 steps against the grid: nearest rounding is then unbiased, for
    one more quantization's worth of error. Only `keep` can saturate.
    """
    began = time.perf_counter()
    raw, scales = data.view(torch.uint8), scale.view(-1)
    down_fp32 = down.float()
    magnitudes = _magnitudes(torch, data.device)
    for start, stop in _chunks(*data.shape):
        old = scales[start:stop].float().unsqueeze(1)
        base = decode_e4m3(torch, raw[start:stop]) * old
        delta = _delta(up, down_fp32, alpha, start, stop)
        target = base + delta
        new = old
        if grid != "keep":
            new = (target.abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp(min=ROW_SCALE_FLOOR)
        if grid == "regrid":
            row = torch.arange(start, stop, device=new.device, dtype=torch.float32)
            new = new * (1 + (1 + 2 * torch.frac(row * 0.6180339887)).unsqueeze(1) / 32)
        payload = target / new
        code = encode_e4m3(torch, payload, rounding, generator, magnitudes=magnitudes)
        raw[start:stop].copy_(code)
        scales[start:stop].copy_(new.view(-1))
        if stats is not None:
            saturated = (payload.abs() > SATURATION).sum()
            _account(stats, base, delta, decode_e4m3(torch, code) * new, saturated, down_fp32)
    _finish(torch, data, stats, began)


def bake_mx(
    torch: ModuleType,
    data: Tensor,
    scale: Tensor,
    down: Tensor,
    up: Tensor,
    alpha: float,
    *,
    rounding: Rounding = "nearest",
    refit: bool = True,
    generator: Generator | None = None,
    stats: Stats | None = None,
) -> None:
    """Bake into `mxfp8/1`: `data` `[out, in]` E4M3, `scale` `[out, in/32]` E8M0 bytes
    (2^(b-127) per block of 32 columns), the stored row-major grid.

    `refit` chooses each block's exponent again as the encoder does: the floor exponent or
    one above it, whichever stores the block with less squared error under nearest
    rounding. A block whose delta is small keeps the exponent it sits on.
    """
    began = time.perf_counter()
    raw, grid = data.view(torch.uint8), scale.view(torch.uint8)
    cols = int(data.shape[1])
    down_fp32 = down.float()
    magnitudes = _magnitudes(torch, data.device)
    for start, stop in _chunks(*data.shape):
        rows = stop - start
        old = _exp2(torch, grid[start:stop].to(torch.int32) - E8M0_BIAS).unsqueeze(-1)
        base = decode_e4m3(torch, raw[start:stop]).view(rows, -1, MX_BLOCK) * old
        delta = _delta(up, down_fp32, alpha, start, stop).view(rows, -1, MX_BLOCK)
        target = base + delta
        byte = grid[start:stop].to(torch.int32)
        if refit:
            amax = target.abs().amax(dim=-1)
            lowest = torch.full_like(amax, -127.0, dtype=torch.float64)
            floor = torch.where(amax > 0, torch.floor(torch.log2(amax.double())) - 8, lowest)
            low = (floor + E8M0_BIAS).clamp(0, 254).to(torch.int32)
            high = (low + 1).clamp(max=254)
            byte = torch.where(
                _mx_error(torch, target, high, magnitudes)
                < _mx_error(torch, target, low, magnitudes),
                high,
                low,
            )
        mult = _exp2(torch, byte - E8M0_BIAS).unsqueeze(-1)
        payload = (target / mult).view(rows, cols)
        code = encode_e4m3(torch, payload, rounding, generator, magnitudes=magnitudes)
        if stats is not None:
            stored = decode_e4m3(torch, code).view(rows, -1, MX_BLOCK) * mult
        raw[start:stop].copy_(code)
        grid[start:stop].copy_(byte.to(torch.uint8))
        if stats is not None:
            _account(
                stats,
                base.view(rows, cols),
                delta.view(rows, cols),
                stored.view(rows, cols),
                (payload.abs() > SATURATION).sum(),
                down_fp32,
            )
    _finish(torch, data, stats, began)


def _mx_error(torch: ModuleType, target: Tensor, byte: Tensor, magnitudes: Tensor) -> Tensor:
    """Each block's squared error stored to nearest under exponent byte `byte`."""
    mult = _exp2(torch, byte - E8M0_BIAS).unsqueeze(-1)
    code = _nearest(torch, magnitudes, target / mult)
    return (decode_e4m3(torch, code) * mult - target).square().sum(dim=-1)


def _exp2(torch: ModuleType, exponent: Tensor) -> Tensor:
    return torch.ldexp(torch.ones_like(exponent, dtype=torch.float32), exponent)


def _finish(torch: ModuleType, tensor: Tensor, stats: Stats | None, began: float) -> None:
    if stats is None:
        return
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    stats.layers += 1
    stats.seconds += time.perf_counter() - began


def bakeable(roles: dict[str, str]) -> bool:
    """Whether a stored weight (role -> TensorFS dtype) takes `bake`."""
    if set(roles) == {"value"}:
        return roles["value"] in ("bf16", "f16", "f32")
    return (
        set(roles) == {"data", "scale"}
        and roles["data"] == "f8_e4m3fn"
        and roles["scale"] in ("f32", "u8")
    )


def bake(
    torch: ModuleType,
    stored: dict[str, Tensor],
    down: Tensor,
    up: Tensor,
    *,
    seed: int,
    stats: Stats | None = None,
) -> None:
    """Bake `up @ down` into one stored weight by its roles: plain floats, `fp8-rowwise/1`
    (E4M3 data, f32 row scale) or `mxfp8/1` (E4M3 data, E8M0 block grid). bf16 and E4M3 round
    stochastically, scales kept: nearest returns a small delta to where it was, and loses
    most of a LoRA's effect on fp8 (MEASURE.md, TensorD #26). `seed` names the weight's
    noise. A GPU bakes fused unless `stats` are asked for (the eager reference measures them)."""
    device = next(iter(stored.values())).device
    if stats is None and device.type == "cuda" and _ready(device.index or 0):
        fused(torch, stored, down, up, seed)
        return
    generator = torch.Generator(device=device).manual_seed(seed & SEED_MASK)
    if set(stored) == {"value"}:
        weight = stored["value"]
        rounding: Rounding = "stochastic" if weight.dtype == torch.bfloat16 else "nearest"
        bake_plain(
            torch, weight, down, up, 1.0, rounding=rounding, generator=generator, stats=stats
        )
    elif stored["scale"].dtype == torch.float32:
        bake_rowwise(
            torch,
            stored["data"],
            stored["scale"],
            down,
            up,
            1.0,
            rounding="stochastic",
            grid="keep",
            generator=generator,
            stats=stats,
        )
    else:
        bake_mx(
            torch,
            stored["data"],
            stored["scale"],
            down,
            up,
            1.0,
            rounding="stochastic",
            refit=False,
            generator=generator,
            stats=stats,
        )


def accumulate(
    torch: ModuleType, x: Tensor, out: Tensor, down: Tensor, up: Tensor, alpha: float
) -> None:
    """`out += alpha * (x @ down.T) @ up.T` in two GEMMs: the up projection adds into the
    base output, so no `[rows, out]` update is ever allocated."""
    flat, rows = out.view(-1, out.shape[-1]), x.reshape(-1, x.shape[-1])
    low = torch.nn.functional.linear(rows, down.to(rows.dtype))
    flat.addmm_(low.to(flat.dtype), up.to(flat.dtype).t(), alpha=alpha)


# ----------------------------------------------------------------------------- fused bake

#: Stored forms the fused kernel bakes (`bakeable`): bf16 (stochastic), f16 and f32 (nearest),
#: `fp8-rowwise/1` and `mxfp8/1` (E4M3, stochastic, scales kept).
BF16, F16, F32, ROWWISE, MX = range(5)
#: A seed is 31 bits: one Triton signature for every weight.
SEED_MASK = 0x7FFFFFFF
_TILE = 64
_INNER = 16
#: Why this process bakes eagerly on its GPU, or None; whether a GPU here bakes (`warm`); whether
#: its kernels are built and checked; the child building them and the file holding its output.
FUSED_ERROR: str | None = None if triton is not None else "triton does not import"
_TRIED = _BUILT = False
_BUILD: tuple[subprocess.Popen[bytes], IO[bytes]] | None = None
_CHECK = (
    "import json, sys; boot = json.loads(sys.argv[1]); sys.path[:] = boot['path']\n"
    "import torch\n"
    "from cozy_runtime.internal import lora_bake\n"
    "lora_bake.check(torch, torch.device('cuda', boot['device']))\n"
)


def fused(
    torch: ModuleType, stored: dict[str, Tensor], down: Tensor, up: Tensor, seed: int
) -> None:
    """`bake`, one kernel launch: each tile adds its `up @ down` (f32) to the decoded stored
    values and rounds them back in place."""
    if "value" in stored:
        data = scale = stored["value"]
        kind = {torch.bfloat16: BF16, torch.float16: F16, torch.float32: F32}[data.dtype]
    else:
        data, scale = stored["data"].view(torch.uint8), stored["scale"]
        kind = ROWWISE if scale.dtype == torch.float32 else MX
        scale = scale.reshape(-1) if kind == ROWWISE else scale.view(torch.uint8)
    if not data.is_contiguous():
        raise ValueError("a fused bake writes a contiguous [out, in] weight")
    rows, cols = data.shape
    factors = (up.to(torch.float32).contiguous(), down.to(torch.float32).contiguous())
    grid = (triton.cdiv(rows, _TILE), triton.cdiv(cols, _TILE))
    _bake_kernel[grid](
        data, scale, *factors, rows, cols, down.shape[0], seed & SEED_MASK,
        FORM=kind, TILE=_TILE, INNER=_INNER,
    )  # fmt: skip


def warm(device: int) -> None:
    """Start building every fused form for GPU `device`, and checking its numbers (`check`), in
    a child process under this one's interpreter and import path, once per process at load.
    This process then only loads the kernels from Triton's cache: it never runs a compiler, so
    one that crashes ends the child, and this process bakes eagerly with its reason."""
    global _BUILD, _TRIED
    _TRIED = True
    if FUSED_ERROR is None and not _BUILT and _BUILD is None:
        log = tempfile.TemporaryFile()  # noqa: SIM115 - read and closed with the build (`_ready`)
        boot = json.dumps({"path": sys.path, "device": device})
        command = [sys.executable, "-I", "-c", _CHECK, boot]
        _BUILD = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log), log


def _ready(device: int, wait: bool = True) -> bool:
    """Whether this process bakes fused: the build `warm` started (now, when none did) ended
    well. Waits for it unless `wait` is False."""
    global FUSED_ERROR, _BUILT, _BUILD
    if FUSED_ERROR is None and not _BUILT:
        warm(device)
        assert _BUILD is not None
        child, log = _BUILD
        if child.poll() is None and not wait:
            return False
        code = child.wait()
        log.seek(0)
        said = log.read().decode(errors="replace").strip().splitlines()
        log.close()
        _BUILD, _BUILT = None, code == 0
        if code:
            how = f"killed by signal {-code}" if code < 0 else f"exit {code}"
            FUSED_ERROR = f"bake kernel build {how}" + (f": {said[-1]}" if said else "")[:300]
    return FUSED_ERROR is None


def mode() -> dict[str, str]:
    """How this process bakes on its GPU, once a build has ended (`modes.lora_bake`)."""
    if _BUILD is not None:
        _ready(0, wait=False)
    if _BUILT:
        return {"lora_bake": "fused"}
    return {"lora_bake": f"eager ({FUSED_ERROR})"} if FUSED_ERROR and _TRIED else {}


def check(torch: ModuleType, device: Device) -> None:
    """Bake every form on `device` where the result is exact (0.25 + 24 * +-1/32 is 1.0 or
    -0.5, over partial tiles and two inner steps): one stored value off it is a broken kernel."""
    rows, cols, rank = 70, 96, 24
    up = torch.full((rows, rank), 1 / 32, device=device)
    up[1::2] = -1 / 32
    down = torch.ones((rank, cols), device=device)
    want = torch.tensor([1.0, -0.5], device=device).repeat(rows // 2)[:, None].expand(rows, cols)
    for kind in range(5):
        if kind in (ROWWISE, MX):
            data = torch.full((rows, cols), 0.5, device=device).to(torch.float8_e4m3fn)
            scale = torch.full((rows,), 0.5, device=device)
            if kind == MX:  # an exponent byte 2^-1 per block of 32 columns
                scale = torch.full((rows, cols // 32), E8M0_BIAS - 1, device=device).byte()
            stored = {"data": data, "scale": scale}
        else:
            dtype = (torch.bfloat16, torch.float16, torch.float32)[kind]
            stored = {"value": torch.full((rows, cols), 0.25, dtype=dtype, device=device)}
        fused(torch, stored, down, up, kind)
        got = stored["value"].float() if "value" in stored else decode_e4m3(torch, data) * 0.5
        if not torch.equal(got, want):
            wrong = int((got != want).sum())
            raise RuntimeError(f"the fused bake of form {kind} stored {wrong} values off the grid")


#: Never specialized on a value or an address's alignment: one compiled kernel per form serves
#: every weight wherever it lies.
_VALUES, _ADDRESSES = ["rows", "cols", "rank", "seed"], ["data", "scale", "up", "down"]

if triton is not None:

    @triton.jit
    def _decode_e4m3(code):  # type: ignore[no-untyped-def]
        """E4M3 bytes (int32) -> f32; the two reserved codes saturate to their signed maximum."""
        field = (code >> 3) & 15
        mantissa = (code & 7).to(tl.float32)
        normal = (mantissa + 8.0) * ((field + 117) << 23).to(tl.float32, bitcast=True)
        magnitude = tl.where(field == 0, mantissa * 0.001953125, normal)
        magnitude = tl.where((code & 127) == 127, 448.0, magnitude)
        return tl.where(code > 127, -magnitude, magnitude)

    @triton.jit
    def _stochastic_e4m3(payload, noise):  # type: ignore[no-untyped-def]
        """f32 payload -> E4M3 bytes (int32), saturating at 448. 20 random bits are added below
        the three kept mantissa bits and the rest truncated: a value rounds up with the
        probability of the distance it covered. Below 2^-6 the grid steps by 2^-9, the normal
        grid of [2^-6, 2^-5) lowered by 2^-6: such a value is lifted into it and lowered back."""
        magnitude = tl.minimum(tl.abs(payload), 448.0)
        small = magnitude < 0.015625
        lifted = tl.where(small, magnitude + 0.015625, magnitude)
        kept = ((lifted.to(tl.uint32, bitcast=True) + (noise >> 12)) >> 20).to(tl.int32)
        code = kept - tl.where(small, 968, 960)  # (f32 exponent - 120) << 3 | mantissa
        return tl.where(payload < 0, code | 128, code)

    @triton.jit(  # type: ignore[untyped-decorator]
        do_not_specialize=_VALUES, do_not_specialize_on_alignment=_ADDRESSES
    )
    def _bake_kernel(  # type: ignore[no-untyped-def]
        data, scale, up, down, rows, cols, rank, seed,
        FORM: tl.constexpr, TILE: tl.constexpr, INNER: tl.constexpr,
    ):  # fmt: skip
        row = (tl.program_id(0) * TILE + tl.arange(0, TILE)).to(tl.int64)
        col = (tl.program_id(1) * TILE + tl.arange(0, TILE)).to(tl.int64)
        delta = tl.zeros((TILE, TILE), dtype=tl.float32)
        for start in range(0, rank, INNER):
            inner = start + tl.arange(0, INNER)
            a = tl.load(
                up + row[:, None] * rank + inner[None, :],
                mask=(row[:, None] < rows) & (inner[None, :] < rank),
                other=0.0,
            )
            b = tl.load(
                down + inner[:, None] * cols + col[None, :],
                mask=(inner[:, None] < rank) & (col[None, :] < cols),
                other=0.0,
            )
            delta += tl.dot(a, b, input_precision="ieee")
        at = row[:, None] * cols + col[None, :]
        inside = (row[:, None] < rows) & (col[None, :] < cols)
        noise = tl.randint(seed, at.to(tl.uint32))
        if FORM >= 3:  # E4M3: ROWWISE (3) or MX (4)
            if FORM == 3:
                unit = tl.load(scale + row, mask=row < rows, other=1.0)[:, None]
            else:
                byte = tl.load(scale + row[:, None] * (cols // 32) + col[None, :] // 32, inside, 1)
                unit = (tl.maximum(byte.to(tl.int32), 1) << 23).to(tl.float32, bitcast=True)
            code = tl.load(data + at, mask=inside, other=0).to(tl.int32)
            payload = _decode_e4m3(code) + delta / unit
            tl.store(data + at, _stochastic_e4m3(payload, noise).to(tl.uint8), mask=inside)
        else:
            value = tl.load(data + at, mask=inside, other=0.0).to(tl.float32) + delta
            if FORM == 0:  # BF16
                kept = (value.to(tl.uint32, bitcast=True) + (noise >> 16)) >> 16 << 16
                value = kept.to(tl.float32, bitcast=True)
            tl.store(data + at, value.to(data.dtype.element_ty), mask=inside)
