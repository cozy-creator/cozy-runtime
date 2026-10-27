"""Bake a LoRA into stored weights, in place, and the two-GEMM per-step update.

A bake is decode -> add `alpha * up @ down` -> round back to the stored representation, one
row chunk at a time, so nothing larger than a chunk exists twice. Plain floats round once to
their dtype. `fp8.rowwise` re-encodes E4M3, `int8-convrot256/1` its int8 grid in the
rotated basis, `nvfp4` its E2M1 values under refitted block scales; the rounding is chosen
because a delta below half a grid step vanishes under nearest rounding. `Stats` is what a
quality gate reads: how much of the delta survived and how much rounding error the bake added.

On a GPU, `bake` is one fused Triton kernel per weight (`_bake_kernel`): it reads each stored
byte once, adds the rank-r product computed in its tile, rounds and writes the byte back, with
counter-based noise (one stream per weight, by element index). Its forms are a Runtime kernel
set (`KERNELS`): built when the executor starts, beside the weight download, then loaded and
checked at the model's load (`warm`), so a cold executor's first request bakes. No request waits
for them: until they serve, a LoRA serves per step. The eager row-chunk functions are their
reference on CPU, what `Stats` measures, and what a GPU bakes with where the set cannot serve.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from torch import Generator, Tensor
    from torch import device as Device

from cozy_runtime.author.kernels import Launch, TritonEntry, TritonKernels
from cozy_runtime.internal import kernel_ready, verify
from cozy_runtime.internal.encoding import E4M3_MAX
from cozy_runtime.internal.encoding.formats import E2M1_VALUES, NVFP4_BLOCK, hadamard256

try:
    import triton
    import triton.language as tl
except ImportError:  # a host without Triton bakes eagerly
    triton = None

Strategy = Literal["bake", "per_step"]
#: How a bake runs on a GPU: its fused kernels, or the eager reference.
How = Literal["fused", "eager"]

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
#: Columns an error-feedback run carries its rounding error across.
FEEDBACK_RUN = 32
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
    """Bake into `fp8.rowwise`: `data` `[out, in]` E4M3, `scale` one f32 per output row.

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


def rotate_factor(torch: ModuleType, down: Tensor) -> Tensor:
    """`down` (`[rank, in]`) in an `int8-convrot256/1` weight's basis: each 256 input channels
    times the orthonormal Hadamard, so `up @ rotated` is the delta of the stored rotated rows."""
    return hadamard256(torch, down.float()).mul_(1.0 / 16)


def bake_int8_convrot(
    torch: ModuleType,
    data: Tensor,
    scale: Tensor,
    down: Tensor,
    up: Tensor,
    alpha: float,
    *,
    rounding: Rounding = "nearest",
    generator: Generator | None = None,
    stats: Stats | None = None,
) -> None:
    """Bake into `int8-convrot256/1`: `data` `[out, in]` int8 of the rotated rows, `scale` one
    f32 per output row, kept. The rotation is orthonormal, so the delta is rotated (`down`'s
    columns) and added on the stored grid; every energy `Stats` sums is the unrotated one.
    `stochastic` rounds up with the probability of the distance covered; beyond 127 clamps."""
    began = time.perf_counter()
    scales = scale.view(-1)
    rotated = rotate_factor(torch, down)
    for start, stop in _chunks(*data.shape):
        unit = scales[start:stop].float().unsqueeze(1)
        code = data[start:stop].float()
        delta = _delta(up, rotated, alpha, start, stop)
        payload = code + delta / unit
        if rounding == "stochastic":
            draw = torch.rand(payload.shape, device=payload.device, generator=generator)
            stored = torch.floor(payload + draw)
        else:
            stored = torch.round(payload)
        stored = stored.clamp_(-127, 127)
        data[start:stop].copy_(stored.to(torch.int8))
        if stats is not None:
            saturated = (payload.abs() > 127.5).sum()
            _account(stats, code * unit, delta, stored * unit, saturated, rotated)
    _finish(torch, data, stats, began)


#: The canonical stored `nvfp4` a bake writes (`bakeable`): element 2k in the low nibble,
#: row-major E4M3 block scales, the f32 second level multiplied.
NVFP4_PARAMS = {
    "block_size": NVFP4_BLOCK,
    "global_scale": "multiply",
    "nibble_order": "low_first",
    "scale_layout": "row_major",
}
#: The largest product of nvfp4's two levels: E4M3's 448 times E2M1's 6.
NVFP4_LEVELS = E4M3_MAX * 6.0
#: A kept block saturates up to 7/6 of its largest value, halfway to the 8 E2M1 lacks: a block
#: quantized with a clipping scale (`encode_nvfp4`'s min-SSE rule) has its top element at 6,
#: and any delta that lifts it would otherwise regrid all 16 of its elements.
E2M1_OUTGROWN = 7.0 / 6.0


def _nvfp4_decode(torch: ModuleType, raw: Tensor, blocks: Tensor, second: float) -> Tensor:
    """Packed u8 `[rows, cols/2]` and E4M3 block bytes `[rows, cols/16]` -> f32 `[rows, cols]`."""
    rows = raw.shape[0]
    codes = torch.stack((raw & 0x0F, raw >> 4), dim=-1).reshape(rows, -1).long()
    values = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=raw.device)[codes]
    scale = decode_e4m3(torch, blocks) * second
    return (values.view(rows, -1, NVFP4_BLOCK) * scale.unsqueeze(-1)).view(rows, -1)


def _e2m1(
    torch: ModuleType,
    payload: Tensor,
    negative: Tensor,
    rounding: Rounding,
    generator: Generator | None,
) -> Tensor:
    """f32 in block units -> E2M1 codes (sign `negative` in bit 3), saturating at 6.
    `stochastic` rounds up with the probability of the distance covered; `nearest` ties to the
    even code."""
    grid = torch.tensor(E2M1_VALUES[:8], dtype=torch.float32, device=payload.device)
    magnitude = payload.abs().clamp(max=6.0)
    low = (torch.searchsorted(grid, magnitude, right=True) - 1).clamp(0, 7)
    high = (low + 1).clamp(max=7)
    gap = grid[high] - grid[low]
    fraction = torch.where(gap > 0, (magnitude - grid[low]) / gap, torch.zeros_like(gap))
    if rounding == "stochastic":
        up = torch.rand(fraction.shape, device=payload.device, generator=generator) < fraction
    else:
        up = (fraction > 0.5) | ((fraction == 0.5) & (low % 2 == 1))
    return (low + up) | (negative.long() << 3)


def bake_nvfp4(
    torch: ModuleType,
    weight: Tensor,
    scale: Tensor,
    scale_2: Tensor,
    down: Tensor,
    up: Tensor,
    alpha: float,
    *,
    rounding: Rounding = "stochastic",
    grid: Grid = "keep",
    generator: Generator | None = None,
    stats: Stats | None = None,
) -> None:
    """Bake into canonical `nvfp4` (`NVFP4_PARAMS`): `weight` u8 `[out, in/2]`, `scale` E4M3
    `[out, in/16]`, `scale_2` the f32 second level, written in place.

    `keep` keeps the second level and refits a block's E4M3 scale only where its baked values
    outgrow it past `E2M1_OUTGROWN` (to the smallest scale that holds them), so every other
    element rounds on the grid it was quantized to and a value just past 6 saturates. `refit`
    sets the second level from the baked tensor's amax and every block's scale from its baked
    amax, saturation-free, so every element rounds afresh (H3's turbo LoRA: retention 1.00 but
    18.2 dB rounding SNR, against `keep`'s 0.945 and 30.6 dB)."""
    began = time.perf_counter()
    raw, blocks = weight.view(torch.uint8), scale.view(torch.uint8)
    rows, cols = raw.shape[0], 2 * raw.shape[1]
    down_fp32 = down.float()
    magnitudes = _magnitudes(torch, raw.device)
    stored_second = second = float(scale_2.float().reshape(-1)[0])
    if grid != "keep":
        amax = 0.0
        for start, stop in _chunks(rows, cols):
            base = _nvfp4_decode(torch, raw[start:stop], blocks[start:stop], stored_second)
            target = base + _delta(up, down_fp32, alpha, start, stop)
            amax = max(amax, float(target.abs().amax()))
        second = amax / NVFP4_LEVELS if amax > 0 else 1.0
    for start, stop in _chunks(rows, cols):
        base = _nvfp4_decode(torch, raw[start:stop], blocks[start:stop], stored_second)
        delta = _delta(up, down_fp32, alpha, start, stop)
        target = (base + delta).view(stop - start, -1, NVFP4_BLOCK)
        need = target.abs().amax(dim=-1) / (6.0 * second)
        fits = torch.searchsorted(magnitudes, need.contiguous()).clamp(max=126)
        code = blocks[start:stop].long() & 0x7F
        # `keep`: past halfway to the 8 E2M1 lacks (`E2M1_OUTGROWN`), the block's scale grows
        grown = need * 6 > magnitudes[code] * 7
        code = torch.where(grown, fits, code) if grid == "keep" else fits
        unit = (magnitudes[code] * second).unsqueeze(-1)
        payload = torch.where(unit > 0, target / torch.where(unit > 0, unit, 1.0), 0.0)
        # A zero keeps its stored sign: a zero delta writes back every byte it read.
        before = base.view_as(target)
        negative = torch.where(target == 0, torch.signbit(before), torch.signbit(target))
        nibbles = _e2m1(torch, payload, negative, rounding, generator).view(stop - start, -1)
        raw[start:stop].copy_((nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).to(torch.uint8))
        if stats is not None:
            stored = _nvfp4_decode(torch, raw[start:stop], code.to(torch.uint8), second)
            saturated = (payload.abs() > 6.0).sum()
            _account(stats, base, delta, stored, saturated, down_fp32)
        blocks[start:stop].copy_(code.to(torch.uint8))
    if second != stored_second:
        scale_2.copy_(torch.full_like(scale_2, second))
    _finish(torch, weight, stats, began)


def _finish(torch: ModuleType, tensor: Tensor, stats: Stats | None, began: float) -> None:
    if stats is None:
        return
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    stats.layers += 1
    stats.seconds += time.perf_counter() - began


def bakeable(scheme: str, params: Mapping[str, object], roles: Mapping[str, str]) -> bool:
    """Whether a stored weight (its scheme, params and role -> dtype) takes `bake`: a plain
    float, a per-row `fp8.rowwise` E4M3, an `int8.convrot` weight or a canonical `nvfp4` one
    (its activation `input_scale` is not the weight's). `int8.rowwise` stores the same parts
    as `int8.convrot` and is not rotated, so the scheme decides, not the parts."""
    if scheme == "plain":
        return dict(roles) in ({"value": "bf16"}, {"value": "f16"}, {"value": "f32"})
    if scheme == "fp8.rowwise":
        return dict(roles) == {"weight": "f8_e4m3fn", "scale": "f32"}
    if scheme == "nvfp4":
        kept = {role: dtype for role, dtype in roles.items() if role != "input_scale"}
        return kept == {"weight": "u8", "scale": "f8_e4m3fn", "scale_2": "f32"} and all(
            params.get(name, value) == value for name, value in NVFP4_PARAMS.items()
        )
    return scheme == "int8.convrot" and dict(roles) == {"weight": "i8", "scale": "f32"}


def bake(
    torch: ModuleType,
    stored: dict[str, Tensor],
    down: Tensor,
    up: Tensor,
    *,
    seed: int,
    stats: Stats | None = None,
    how: How | None = "eager",
) -> None:
    """Bake `up @ down` into one stored weight by its roles: plain floats, `fp8.rowwise`
    (E4M3 data, f32 row scale), `int8-convrot256/1` (int8 data, f32 row scale) or `nvfp4`
    (`bake_nvfp4`). bf16, E4M3, int8 and E2M1 round stochastically, scales kept: nearest
    returns a small delta to where it was, and loses most of a LoRA's effect on fp8
    (MEASURE.md, TensorD #26). `seed` names the weight's noise. A GPU bakes fused when `how`
    says its kernels serve (`serving`), unless `stats` are asked for (the eager reference
    measures them). A convolution's weight and factors bake as matrices, each kernel tap one
    more input column."""
    if down.dim() > 2:
        stored = {role: t.view(t.shape[0], -1) if t.dim() > 2 else t for role, t in stored.items()}
        down, up = down.flatten(1), up.flatten(1)
    device = next(iter(stored.values())).device
    if stats is None and device.type == "cuda" and how == "fused":
        fused(torch, _launch(device.index or 0), stored, down, up, seed)
        return
    generator = torch.Generator(device=device).manual_seed(seed & SEED_MASK)
    if "scale_2" in stored:
        bake_nvfp4(
            torch,
            stored["weight"],
            stored["scale"],
            stored["scale_2"],
            down,
            up,
            1.0,
            generator=generator,
            stats=stats,
        )
    elif set(stored) == {"value"}:
        weight = stored["value"]
        rounding: Rounding = "stochastic" if weight.dtype == torch.bfloat16 else "nearest"
        bake_plain(
            torch, weight, down, up, 1.0, rounding=rounding, generator=generator, stats=stats
        )
    elif stored["weight"].dtype == torch.int8:
        bake_int8_convrot(
            torch,
            stored["weight"],
            stored["scale"],
            down,
            up,
            1.0,
            rounding="stochastic",
            generator=generator,
            stats=stats,
        )
    else:
        bake_rowwise(
            torch,
            stored["weight"],
            stored["scale"],
            down,
            up,
            1.0,
            rounding="stochastic",
            grid="keep",
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
#: `fp8.rowwise` (E4M3, stochastic, scale kept), `int8-convrot256/1` (int8 of the rotated
#: rows, stochastic, scale kept; `fused` rotates `down`), and `nvfp4` (its own kernel, as
#: `bake_nvfp4` with `keep`).
BF16, F16, F32, ROWWISE, INT8, NVFP4 = range(6)
#: A seed is 31 bits: one Triton signature for every weight.
SEED_MASK = 0x7FFFFFFF
_TILE = 64
_INNER = 16


def fused(
    torch: ModuleType,
    launch: Launch,
    stored: dict[str, Tensor],
    down: Tensor,
    up: Tensor,
    seed: int,
) -> None:
    """`bake`, one kernel launch: each tile adds its `up @ down` (f32) to the decoded stored
    values and rounds them back in place."""
    factors = (up.to(torch.float32).contiguous(), down.to(torch.float32).contiguous())
    if "scale_2" in stored:
        raw = stored["weight"].view(torch.uint8)
        rows, cols = raw.shape[0], 2 * raw.shape[1]
        second = stored["scale_2"].to(torch.float32).reshape(1)
        grid = (triton.cdiv(rows, _TILE), triton.cdiv(cols, _TILE), 1)
        launch(
            _bake_nvfp4_kernel, grid,
            raw, stored["scale"].view(torch.uint8), second, *factors, rows, cols, down.shape[0],
            seed & SEED_MASK, _TILE, _INNER,
        )  # fmt: skip
        return
    if "value" in stored:
        data = scale = stored["value"]
        kind = {torch.bfloat16: BF16, torch.float16: F16, torch.float32: F32}[data.dtype]
    elif stored["weight"].dtype == torch.int8:
        data, scale, kind = stored["weight"], stored["scale"].reshape(-1), INT8
        down = rotate_factor(torch, down)
    else:
        data, scale, kind = stored["weight"].view(torch.uint8), stored["scale"].reshape(-1), ROWWISE
    if not data.is_contiguous():
        raise ValueError("a fused bake writes a contiguous [out, in] weight")
    rows, cols = data.shape
    if kind == INT8:
        factors = (factors[0], down.to(torch.float32).contiguous())
    grid = (triton.cdiv(rows, _TILE), triton.cdiv(cols, _TILE), 1)
    launch(
        _bake_kernel, grid,
        data, scale, *factors, rows, cols, down.shape[0], seed & SEED_MASK, kind, _TILE, _INNER,
    )  # fmt: skip


def warm(torch: ModuleType, device: int) -> None:
    """Open the bake kernels for GPU `device` at load. They build when the executor starts
    (`KERNELS`, beside the weight download), so a load that follows the build serves them from
    its first request; nothing waits for them, and until they serve a LoRA serves per step."""
    serving(torch, device)


def serving(torch: ModuleType, device: int) -> How | None:
    """How a bake on GPU `device` runs now: `fused` once its kernels are built, loaded here and
    pass `probe` (`kernel_ready`), `eager` (the reference) where they cannot, None while they
    build."""
    if triton is None or not kernel_ready.TRITON:
        return "eager"
    gpu = torch.device("cuda", device)
    opener = partial(kernel_ready.TRITON[0], KERNELS, partial(probe, torch, gpu), device=gpu)
    name = f"{KERNELS.name} cuda:{device}"
    if kernel_ready.ready(name, opener) is not None:
        return "fused"
    return "eager" if kernel_ready.failed(name) else None


def _launch(device: int) -> Launch:
    """GPU `device`'s bake launcher: the attempt chose `fused` once it served."""
    launch = kernel_ready.ready(f"{KERNELS.name} cuda:{device}")
    if launch is None:
        raise RuntimeError("a fused bake was chosen before its kernels served")
    return cast("Launch", launch)


def recipe(fused: bool) -> str:
    """What names the bytes a bake writes to other executors: `VARIANT`, the path it bakes by
    (fused, else the eager reference) and what that path's numerics come from."""
    return f"{VARIANT} fused triton {triton.__version__}" if fused else f"{VARIANT} eager"


def probe(torch: ModuleType, device: Device, launch: Launch) -> verify.Verdict:
    """Bake every form on `device` where the result is exact (0.25 + 24 * +-1/32 is 1.0 or
    -0.5, over partial tiles and two inner steps): one stored value off it is a broken kernel."""
    rows, cols, rank = 70, 96, 24
    up = torch.full((rows, rank), 1 / 32, device=device)
    up[1::2] = -1 / 32
    down = torch.ones((rank, cols), device=device)
    want = torch.tensor([1.0, -0.5], device=device).repeat(rows // 2)[:, None].expand(rows, cols)
    wrong: dict[int, int] = {}
    for kind in (BF16, F16, F32, ROWWISE):
        if kind == ROWWISE:
            data = torch.full((rows, cols), 0.5, device=device).to(torch.float8_e4m3fn)
            stored = {"weight": data, "scale": torch.full((rows,), 0.5, device=device)}
        else:
            dtype = (torch.bfloat16, torch.float16, torch.float32)[kind]
            stored = {"value": torch.full((rows, cols), 0.25, dtype=dtype, device=device)}
        fused(torch, launch, stored, down, up, kind)
        got = stored["value"].float() if "value" in stored else decode_e4m3(torch, data) * 0.5
        wrong[kind] = int((got != want).sum())
    # nvfp4: 0.25 (E2M1 0.5 under a 0.5 block scale) + 24 * +-1/32 is 1.0 or -0.5, both on that
    # block's kept grid, over a partial column tile.
    stored = {
        "weight": torch.full((rows, cols // 2), 0x11, dtype=torch.uint8, device=device),
        "scale": torch.full((rows, cols // 16), 0.5, device=device).to(torch.float8_e4m3fn),
        "scale_2": torch.ones(1, device=device),
    }
    fused(torch, launch, stored, down, up, NVFP4)
    got = _nvfp4_decode(torch, stored["weight"], stored["scale"].view(torch.uint8), 1.0)
    wrong[NVFP4] = int((got != want).sum())
    # int8 of rotated rows: 2 + 24 * +-1/48 / 0.5 is 3 or 1 (the regular Hadamard keeps a
    # constant row), over partial row tiles and two inner steps.
    data = torch.full((rows, 256), 2, dtype=torch.int8, device=device)
    stored = {"weight": data, "scale": torch.full((rows,), 0.5, device=device)}
    fused(torch, launch, stored, torch.ones((rank, 256), device=device), up * (32 / 48), INT8)
    want = torch.tensor([3, 1], device=device).repeat(rows // 2)[:, None].expand(rows, 256)
    wrong[INT8] = int((data.long() != want).sum())
    off = {kind: n for kind, n in wrong.items() if n}
    detail = f"form(s) {off} stored values off the grid" if off else f"{len(wrong)} forms exact"
    return verify.Verdict(not off, detail)


#: Never specialized on a value or an address's alignment: one compiled kernel per form serves
#: every weight wherever it lies.
_VALUES, _ADDRESSES = ["rows", "cols", "rank", "seed"], ["data", "scale", "up", "down"]
_NVFP4_ADDRESSES = ["data", "blocks", "second", "up", "down"]

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

    @triton.jit
    def _e4m3_at_least(need):  # type: ignore[no-untyped-def]
        """The smallest E4M3 code (int32, at most 126) whose magnitude holds `need` >= 0: its f32
        bits rounded up to three mantissa bits, or below 2^-6 a count of 2^-9."""
        bits = (need.to(tl.uint32, bitcast=True) + 0xFFFFF) & 0xFFF00000
        normal = (((bits >> 23).to(tl.int32) - 120) << 3) | ((bits >> 20) & 7).to(tl.int32)
        code = tl.where(need < 0.015625, tl.ceil(need * 512.0).to(tl.int32), normal)
        return tl.minimum(tl.maximum(code, 0), 126)

    @triton.jit
    def _e2m1_value(code):  # type: ignore[no-untyped-def]
        """E2M1 codes (int32, sign in bit 3) -> f32."""
        index = (code & 7).to(tl.float32)
        magnitude = tl.where(
            index < 4, index * 0.5, tl.where(index < 6, index - 2.0, tl.where(index < 7, 4.0, 6.0))
        )
        return tl.where(code >= 8, -magnitude, magnitude)

    @triton.jit
    def _stochastic_e2m1(payload, draw):  # type: ignore[no-untyped-def]
        """|payload| (block units, saturating at 6) -> its E2M1 magnitude code, the upper neighbour
        with the probability of the distance covered."""
        magnitude = tl.minimum(tl.abs(payload), 6.0)
        step = tl.where(magnitude < 2.0, 0.5, tl.where(magnitude < 4.0, 1.0, 2.0))
        low = tl.minimum(tl.floor(magnitude / step) * step, 6.0)
        index = tl.where(
            low < 2.0, low * 2.0, tl.where(low < 4.0, low + 2.0, tl.where(low < 6.0, 6.0, 7.0))
        )
        return (index + tl.where(draw < (magnitude - low) / step, 1.0, 0.0)).to(tl.int32)

    @triton.jit(  # type: ignore[untyped-decorator]
        do_not_specialize=_VALUES, do_not_specialize_on_alignment=_NVFP4_ADDRESSES
    )
    def _bake_nvfp4_kernel(  # type: ignore[no-untyped-def]
        data, blocks, second, up, down, rows, cols, rank, seed,
        TILE: tl.constexpr, INNER: tl.constexpr,
    ):  # fmt: skip
        """`bake_nvfp4` with `keep`, one tile of 16-element blocks at a time."""
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
        inside = (row[:, None] < rows) & (col[None, :] < cols)
        width, count = cols // 2, cols // 16
        byte = tl.load(data + row[:, None] * width + col[None, :] // 2, mask=inside, other=0)
        code = tl.where(col[None, :] % 2 == 0, byte & 15, byte >> 4).to(tl.int32)
        scaled = tl.load(blocks + row[:, None] * count + col[None, :] // 16, mask=inside, other=0)
        scaled = scaled.to(tl.int32)
        level = tl.load(second)
        target = _e2m1_value(code) * (_decode_e4m3(scaled) * level) + delta
        need = tl.max(tl.reshape(tl.abs(target), (TILE, TILE // 16, 16)), axis=2) / (6.0 * level)
        kept = tl.max(tl.reshape(scaled, (TILE, TILE // 16, 16)), axis=2)
        grown = need * 6.0 > _decode_e4m3(kept) * 7.0  # `E2M1_OUTGROWN`
        block = tl.where(grown, _e4m3_at_least(need), kept)
        unit = (_decode_e4m3(block) * level)[:, :, None]
        unit = tl.reshape(tl.broadcast_to(unit, (TILE, TILE // 16, 16)), (TILE, TILE))
        payload = target / tl.where(unit > 0, unit, 1.0)
        at = row[:, None] * cols + col[None, :]
        nibble = _stochastic_e2m1(payload, tl.rand(seed, at.to(tl.uint32)))
        negative = tl.where(target == 0, code >= 8, target < 0)  # a zero keeps its stored sign
        nibble = nibble | tl.where(negative, 8, 0)
        low, high = tl.split(tl.reshape(nibble, (TILE, TILE // 2, 2)))
        pair = tl.program_id(1) * (TILE // 2) + tl.arange(0, TILE // 2).to(tl.int64)
        tl.store(
            data + row[:, None] * width + pair[None, :],
            (low | (high << 4)).to(tl.uint8),
            mask=(row[:, None] < rows) & (pair[None, :] < width),
        )
        which = tl.program_id(1) * (TILE // 16) + tl.arange(0, TILE // 16).to(tl.int64)
        tl.store(
            blocks + row[:, None] * count + which[None, :],
            block.to(tl.uint8),
            mask=(row[:, None] < rows) & (which[None, :] < count),
        )

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
        if FORM == 4:  # INT8: the rotated rows' int8 grid, one f32 scale per row
            unit = tl.load(scale + row, mask=row < rows, other=1.0)[:, None]
            code = tl.load(data + at, mask=inside, other=0).to(tl.float32)
            draw = tl.rand(seed, at.to(tl.uint32))
            stored = tl.floor(code + delta / unit + draw)
            stored = tl.minimum(tl.maximum(stored, -127.0), 127.0)
            tl.store(data + at, stored.to(tl.int8), mask=inside)
        elif FORM == 3:  # ROWWISE: E4M3, one f32 scale per row
            unit = tl.load(scale + row, mask=row < rows, other=1.0)[:, None]
            code = tl.load(data + at, mask=inside, other=0).to(tl.int32)
            payload = _decode_e4m3(code) + delta / unit
            tl.store(data + at, _stochastic_e4m3(payload, noise).to(tl.uint8), mask=inside)
        else:
            value = tl.load(data + at, mask=inside, other=0.0).to(tl.float32) + delta
            if FORM == 0:  # BF16
                kept = (value.to(tl.uint32, bitcast=True) + (noise >> 16)) >> 16 << 16
                value = kept.to(tl.float32, bitcast=True)
            tl.store(data + at, value.to(data.dtype.element_ty), mask=inside)


def _entries() -> list[TritonEntry]:
    """One compiled bake per stored form, never specialized on a value or an address."""
    if triton is None:
        return []
    sizes = {"rows": "i32", "cols": "i32", "rank": "i32", "seed": "i32"}
    tiles = {"TILE": "constexpr", "INNER": "constexpr"}
    stored = {BF16: "*bf16", F16: "*fp16", F32: "*fp32", ROWWISE: "*u8", INT8: "*i8"}
    entries = [
        TritonEntry(
            _bake_kernel,
            {
                "data": data,
                "scale": data if form in (BF16, F16, F32) else "*fp32",
                "up": "*fp32",
                "down": "*fp32",
                **sizes,
                "FORM": "constexpr",
                **tiles,
            },
            {"FORM": form, "TILE": _TILE, "INNER": _INNER},
        )
        for form, data in stored.items()
    ]
    nvfp4 = {"data": "*u8", "blocks": "*u8", "second": "*fp32", "up": "*fp32", "down": "*fp32"}
    tile = {"TILE": _TILE, "INNER": _INNER}
    entries.append(TritonEntry(_bake_nvfp4_kernel, {**nvfp4, **sizes, **tiles}, tile))
    return entries


#: Built on every executor's GPUs when it starts (`author.kernels`): a request never compiles.
KERNELS = TritonKernels("cozy-lora-bake", _entries())
