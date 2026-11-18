"""int8-convrot convolution (cr-327): `_kernels._conv`'s NDHWC conv3d on the int8 tensor cores and the passes that
feed it (`kernels/csrc/conv/conv8.cuh`).

An int8.convrot conv weight is stored as its filter with the input channels last, `[O, kt, kh, kw, I]`
(`formats.conv_declared`): a row for each (output channel, tap), rotated over the input channels by the regular H256 and
scaled per row. Activations are rotated the same way and quantized with one scale per voxel (`quantize`), a GroupNorm's
normalization, an affine and SiLU in the same pass where a block has them. A voxel's scale differs between the taps an
output sums, so each tap's int32 sum is promoted to f32 with its voxel and weight scales: the conv is exact per tap.
Activations are NDHWC (`channels_last_3d`), 512 channels a voxel for the passes. Every call opens the kernels through
`kernel_ready` once they pass their probe on the card (`kernels`).

A pass and its conv run in bands of output frames at the level native selection takes (`cut`), as a 16-bit conv runs
in bands: a band quantizes the input frames its taps read, which slide to the next band (each quantized once), and
writes its frames of the output, which may be the dead input itself. A voxel's scale is its own and the norm's
statistics come from the whole sample first, so a band is the whole's arithmetic: the probe checks it bit for bit."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING

from cozy_runtime import _kernels
from cozy_runtime.internal import arch, kernel_ready, prebuilt, verify

if TYPE_CHECKING:
    import torch
    from torch import Tensor
else:
    try:
        import torch
    except ImportError:  # pragma: no cover - the base package is torch-free
        torch = None

#: The channels a voxel of the passes holds (`conv8_quant`, `conv8_temporal`): a warp's 16 a lane.
CHANNELS = 512
#: The depthwise temporal conv `temporal` runs: its taps.
TEMPORAL_TAPS = 5
#: The rel L2 a probe case may reach against float64: the conv's bf16 output rounding (~2e-3); int8 quantization of a
#: rotated voxel (~1e-2); f32 sums for the statistics. A wrong rotation, tap or scale is O(1).
BUDGET = {"conv": 6e-3, "quantize": 2.5e-2, "stats": 1e-5}


def config(sm: int) -> int | None:
    """The int8 conv's tile config on a card (`conv8.cu`), by tensor-core family: mma.sync cards (sm80, sm86, sm89,
    sm120, sm121) take 128x128 tiles of eight 64x32 warps, K steps of 128 bytes in three stages (99 KB). None where
    the family's 16-bit tensor cores beat int8 mma.sync, wgmma and tcgen05 (sm90, sm100, sm103): an int8 filter
    convolves decoded on torch there. Measured (cr-327), a 512-channel 3x3x3 conv at 22x90x160: RTX 5090 565-587 TOPS
    against cuDNN bf16's 224-235 TFLOPS (comfy-kitchen's fp16-accumulate conv 355-381); RTX 5060 Ti 130-132 against
    49; H100 SXM 448-457 against 670-747."""
    return None if sm in (90, 100, 103) else 0


def kernels(index: int) -> ModuleType | None:
    """`_kernels._conv`'s int8 conv once it passed its probe on cuda:`index`, else None: opening (off the request
    thread), failed, or absent from this wheel, said once (`kernel_ready`)."""
    capturing = torch.cuda.is_current_stream_capturing()
    return kernel_ready.serve(f"cozy conv8 cuda:{index}", partial(_open, index), capturing)


def prime(index: int) -> None:
    """Opens the int8 conv on cuda:`index` at device init, beside the weight fill (`kernel_ready.prime`), so a first
    call serves it rather than the decoded filter."""
    kernel_ready.prime(f"cozy conv8 cuda:{index}", partial(_open, index))


def serving(index: int) -> str:
    """The kernel an int8 conv leaf on cuda:`index` serves now: read once a run (`gemm.selected`)."""
    return "cozy-conv8" if kernel_ready.opened(f"cozy conv8 cuda:{index}") is not None else "torch.conv3d"


def _open(index: int, wait: bool) -> ModuleType:
    missing = _kernels.ABSENT or getattr(_kernels, "NO_CONV", "")
    if missing:
        raise kernel_ready.Refused(missing)
    major, minor = torch.cuda.get_device_capability(index)
    sm = major * 10 + minor
    if "int8" not in arch.hardware(sm).formats():
        raise kernel_ready.Refused(f"sm{sm} has no int8 tensor cores")
    if config(sm) is None:
        raise kernel_ready.Refused(f"sm{sm}'s 16-bit tensor cores beat its int8 mma.sync for a conv")
    module = _kernels._conv
    if not hasattr(module, "conv8"):
        raise kernel_ready.Refused("this cozy-runtime wheel's conv kernels have no int8 conv")
    return kernel_ready.verified(module, "prebuilt-conv8", prebuilt.artifact("_conv"), sm, partial(_probe, index))


def _sms(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _stream(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def _check(module: ModuleType, what: str, code: int) -> None:
    if code:
        raise RuntimeError(f"cozy conv8: {what}: {module.error_name(code) if code > 0 else code}")


def stats(module: ModuleType, x: Tensor, groups: int, eps: float) -> tuple[Tensor, Tensor]:
    """A GroupNorm's (mean, rstd) of NDHWC `x` [N, ..., C] over each sample's voxels and each group's channels, f32
    [N, groups]."""
    n, c = int(x.shape[0]), int(x.shape[-1])
    voxels = x.numel() // (n * c)
    acc = torch.zeros((n, groups, 2), dtype=torch.float64, device=x.device)
    code = module.conv8_stats(x.data_ptr(), acc.data_ptr(), n, voxels, c, groups, x.dtype == torch.float16,
                              _sms(x.device), _stream(x.device))  # fmt: skip
    _check(module, "stats", code)
    count = voxels * (c // groups)
    mean = acc[..., 0] / count
    var = (acc[..., 1] / count - mean * mean).clamp_min_(0)
    return mean.float(), (var + eps).rsqrt_().float()


def affine(
    mean: Tensor,
    rstd: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    scale: Tensor | None = None,
    shift: Tensor | None = None,
    channels: int = CHANNELS,
) -> tuple[Tensor, Tensor]:
    """`(a, b)`, f32 [N, C], with `a x + b` = GroupNorm(x) by (`mean`, `rstd`) and its affine (`weight`, `bias`), then
    `* (1 + scale) + shift` where a block modulates it (`scale`, `shift` [N, C])."""
    n, groups = (int(v) for v in mean.shape)
    per = channels // groups
    a = rstd.repeat_interleave(per, 1)
    b = -mean.repeat_interleave(per, 1) * a
    if weight is not None:
        gamma = weight.float()
        a, b = a * gamma, b * gamma
    if bias is not None:
        b = b + bias.float()
    if scale is not None:
        factor = 1 + scale.float()
        a, b = a * factor, b * factor
    if shift is not None:
        b = b + shift.float()
    return a.expand(n, channels).contiguous(), b.expand(n, channels).contiguous()


def _codes(x: Tensor, out: tuple[Tensor, Tensor] | None) -> tuple[Tensor, Tensor]:
    """`out`, or new int8 codes like `x` [N, ..., 512] and f32 scales [N, ...]."""
    if out is None:
        return (torch.empty(x.shape, dtype=torch.int8, device=x.device),
                torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device))  # fmt: skip
    q, qs = out
    if q.shape != x.shape or qs.shape != x.shape[:-1] or not (q.is_contiguous() and qs.is_contiguous()):
        raise ValueError(f"cozy conv8: codes {list(q.shape)} and scales {list(qs.shape)} for {list(x.shape)}")
    return q, qs


def quantize(
    module: ModuleType, x: Tensor, a: Tensor, b: Tensor, *, silu: bool, out: tuple[Tensor, Tensor] | None = None
) -> tuple[Tensor, Tensor]:
    """NDHWC `x` [N, ..., 512] (fp16 or bf16) as `silu(a x + b)` (or `a x + b`), rotated and quantized a voxel: int8
    codes like `x` and their f32 scales [N, ...], into `out` where given."""
    q, qs = _codes(x, out)
    voxels = x.numel() // CHANNELS
    code = module.conv8_quant(x.data_ptr(), a.data_ptr(), b.data_ptr(), q.data_ptr(), qs.data_ptr(), voxels,
                              voxels // int(x.shape[0]), silu, x.dtype == torch.float16, _sms(x.device),
                              _stream(x.device))  # fmt: skip
    _check(module, "quantize", code)
    return q, qs


def temporal(
    module: ModuleType,
    x: Tensor,
    a: Tensor,
    b: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    out: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """`quantize` of the depthwise temporal conv of `silu(a x + b)`: `weight` [512, 5] f32 (`TEMPORAL_TAPS`, zero
    padding 2), `bias` [512] f32; `x` NDHWC [N, T, H, W, 512]."""
    n, t = int(x.shape[0]), int(x.shape[1])
    hw = x.numel() // (n * t * CHANNELS)
    q, qs = _codes(x, out)
    code = module.conv8_temporal(x.data_ptr(), a.data_ptr(), b.data_ptr(), weight.data_ptr(),
                                 0 if bias is None else bias.data_ptr(), q.data_ptr(), qs.data_ptr(), n, t, hw,
                                 x.dtype == torch.float16, _sms(x.device), _stream(x.device))  # fmt: skip
    _check(module, "temporal", code)
    return q, qs


@dataclass(frozen=True, slots=True)
class Filter:
    """An int8 conv leaf's filter as `conv` and `cut` take it: `weight` int8 [K, kt, kh, kw, C], `scale` f32 [K,
    taps], `bias` f32 [K] or None, zero `padding` (frames, rows, columns), stride 1."""

    weight: Tensor
    scale: Tensor
    bias: Tensor | None
    padding: tuple[int, int, int]

    def out(self, x: Tensor) -> tuple[int, int, int, int, int]:
        """The output's NDHWC shape for NDHWC `x`."""
        n, d, h, w, _ = (int(v) for v in x.shape)
        k, t, r, s = (int(v) for v in self.weight.shape[:4])
        pt, pr, ps = self.padding
        return n, d + 2 * pt - t + 1, h + 2 * pr - r + 1, w + 2 * ps - s + 1, k


def conv(
    module: ModuleType,
    q: Tensor,
    qs: Tensor,
    f: Filter,
    residual: Tensor | None,
    dtype: torch.dtype,
    *,
    front: int | None = None,
    frames: int | None = None,
    out: Tensor | None = None,
) -> Tensor | None:
    """`conv3d(x, w) + bias + residual`, stride 1, zero padding: `q` int8 NDHWC [N, D, H, W, C] with `qs` [N, D, H,
    W], `f`'s filter, `residual` NDHWC like the output (`dtype`, fp16 or bf16). The output NDHWC [N, Z, P, Q, K] (into
    `out` where given); None where the kernels do not serve the shape. A band of output frames passes its `frames` and
    its `front` padding (the whole's, less the frames before its input): the kernel skips taps outside `q`."""
    n, d, h, w, c = (int(v) for v in q.shape)
    k, t, r, s = (int(v) for v in f.weight.shape[:4])
    pt, pr, ps = f.padding
    front = pt if front is None else front
    z, p, qq = d + 2 * pt - t + 1 if frames is None else frames, h + 2 * pr - r + 1, w + 2 * ps - s + 1
    major, minor = torch.cuda.get_device_capability(q.device)
    tiles = config(major * 10 + minor)
    if tiles is None:
        return None
    if out is None:
        out = torch.empty((n, z, p, qq, k), dtype=dtype, device=q.device)
    elif tuple(out.shape) != (n, z, p, qq, k) or out.dtype != dtype or not out.is_contiguous():
        raise ValueError(f"cozy conv8: an output {list(out.shape)} {out.dtype} for [{n}, {z}, {p}, {qq}, {k}] {dtype}")
    code = module.conv8(q.data_ptr(), qs.data_ptr(), f.weight.data_ptr(), f.scale.data_ptr(),
                        0 if f.bias is None else f.bias.data_ptr(), 0 if residual is None else residual.data_ptr(),
                        out.data_ptr(), n, d, h, w, c, k, t, r, s, front, pr, ps, z, p, qq, True,
                        dtype == torch.float16, tiles, _stream(q.device))  # fmt: skip
    if code == -1:
        return None
    _check(module, "conv", code)
    return out


#: The temporal pass's halo: the input frames each side an output frame reads.
_HALO = TEMPORAL_TAPS // 2

type Mix = tuple[Tensor, Tensor | None]


def _over(
    x: Tensor, f: Filter, dtype: torch.dtype, mix: Mix | None, residual: Tensor | None, into: Tensor | None
) -> bool:
    """Whether the output can be written into `into`: its shape, not the residual's memory, and where `into` is `x`
    itself, a band's output frames are input frames every later band has quantized (the taps read no further back
    than they reach forward; the temporal pass reads raw frames, so never)."""
    if into is None or tuple(into.shape) != f.out(x) or into.dtype != dtype or not into.is_contiguous():
        return False
    memory = into.untyped_storage().data_ptr()
    if residual is not None and residual.untyped_storage().data_ptr() == memory:
        return False
    if memory != x.untyped_storage().data_ptr():
        return True
    t, pt = int(f.weight.shape[1]), f.padding[0]
    return mix is None and pt <= t - 1 - pt


def _banded(f: Filter) -> bool:
    """A band's input frames start at or before its output's (front padding within the filter)."""
    return f.padding[0] <= int(f.weight.shape[1]) - 1


def need(
    x: Tensor, f: Filter, dtype: torch.dtype, mix: Mix | None, residual: Tensor | None, into: Tensor | None
) -> tuple[int, Callable[[int], tuple[int, ...]]]:
    """`cut`'s finest level and each level's allocations: the new output (none where it goes over `into`), and the
    codes and scales a band holds (the whole input's at level 0)."""
    n, d = int(x.shape[0]), int(x.shape[1])
    shape = f.out(x)
    z, plane = shape[1], int(x.shape[2]) * int(x.shape[3])
    new = 0 if _over(x, f, dtype, mix, residual, into) else math.prod(shape) * dtype.itemsize
    reach = int(f.weight.shape[1]) - 1 + (2 * _HALO if mix is not None else 0)

    def at(level: int) -> tuple[int, ...]:
        frames = n * d if not level else min(d, -(-z // (1 << level)) + reach)
        return new, frames * plane * (CHANNELS + 4)

    return (max(z - 1, 0).bit_length() if _banded(f) else 0), at


def frames(
    module: ModuleType,
    x: Tensor,
    a: Tensor,
    b: Tensor,
    f: Filter,
    dtype: torch.dtype,
    level: int,
    *,
    silu: bool = True,
    mix: Mix | None = None,
    residual: Tensor | None = None,
    into: Tensor | None = None,
) -> Tensor | None:
    """The conv of NDHWC `x` [N, D, H, W, 512] after its pass, `silu(a x + b)` per voxel or, with `mix`, the
    temporal one (`temporal`), plus `residual`: whole at level 0, else in 2^level bands of each sample's output frames.
    Each band's codes hold the input frames its taps read; the frames the next band also reads move down, so each
    frame quantizes once (the temporal pass computes its own halo's codes again, and drops them). The output goes into
    `into` where its shape allows (`_over`), else a new tensor; everything a band holds is made before the first band
    writes, so a refusal leaves `into` as it was. None where the kernels do not serve the shape."""
    from cozy_runtime.internal.execution import catch

    n, d = int(x.shape[0]), int(x.shape[1])
    shape = f.out(x)
    z, t, pt = shape[1], int(f.weight.shape[1]), f.padding[0]
    over = into is not None and _over(x, f, dtype, mix, residual, into)
    out = into if into is not None and over else torch.empty(shape, dtype=dtype, device=x.device)
    if not level or not _banded(f):
        q, qs = temporal(module, x, a, b, *mix) if mix is not None else quantize(module, x, a, b, silu=silu)
        return conv(module, q, qs, f, residual, dtype, out=out)
    size = -(-z // (1 << level))
    span = min(d, size + t - 1 + (2 * _HALO if mix is not None else 0))
    q = torch.empty((1, span, *x.shape[2:]), dtype=torch.int8, device=x.device)
    qs = torch.empty((1, span, *x.shape[2:4]), dtype=torch.float32, device=x.device)
    for i in range(n):
        held = (0, 0)
        for z0 in range(0, z, size):
            catch.checkpoint()
            z1 = min(z, z0 + size)
            lo, hi = max(0, z0 - pt), min(d, z1 + t - 1 - pt)
            if mix is None:
                keep, shift = max(0, held[1] - lo), lo - held[0]
                for j in range(keep if shift else 0):  # upward: a frame moves before a later one lands on it
                    q[:, j].copy_(q[:, shift + j])
                    qs[:, j].copy_(qs[:, shift + j])
                if lo + keep < hi:
                    window = (q[:, keep : hi - lo], qs[:, keep : hi - lo])
                    quantize(module, x[i : i + 1, lo + keep : hi], a[i : i + 1], b[i : i + 1], silu=silu, out=window)
                codes = q[:, : hi - lo], qs[:, : hi - lo]
            else:
                e0, e1 = max(0, lo - _HALO), min(d, hi + _HALO)
                temporal(module, x[i : i + 1, e0:e1], a[i : i + 1], b[i : i + 1], *mix,
                         out=(q[:, : e1 - e0], qs[:, : e1 - e0]))  # fmt: skip
                codes = q[:, lo - e0 : hi - e0], qs[:, lo - e0 : hi - e0]
            held = (lo, hi)
            band = None if residual is None else residual[i : i + 1, z0:z1]
            found = conv(module, *codes, f, band, dtype, front=pt - z0 + lo, frames=z1 - z0, out=out[i : i + 1, z0:z1])
            if found is None:
                if i or z0:
                    raise RuntimeError("cozy conv8: a band of a conv the kernels served was refused")
                return None
    return out


def cut(
    name: str | None,
    module: ModuleType,
    x: Tensor,
    a: Tensor,
    b: Tensor,
    f: Filter,
    dtype: torch.dtype,
    *,
    silu: bool = True,
    mix: Mix | None = None,
    residual: Tensor | None = None,
    into: Tensor | None = None,
) -> Tensor | None:
    """`frames` at the level native selection takes for site `name` (a known cut's name under the current block where
    None), as a 16-bit conv's bands are (`execution.cuts`); None where the kernels do not serve this card or shape."""
    from cozy_runtime.internal.execution import catch

    major, minor = torch.cuda.get_device_capability(x.device)
    if config(major * 10 + minor) is None:
        return None
    finest, per = need(x, f, dtype, mix, residual, into)
    run = partial(frames, module, x, a, b, f, dtype, silu=silu, mix=mix, residual=residual, into=into)
    site = name or catch.site("conv8:frames") or "conv8:frames"
    found: Tensor | None = catch.cut(site, finest, per, run)
    return found


def rotate(x: Tensor) -> Tensor:
    """The orthonormal regular H256 (H4 = J - 2 anti-identity, four times Kronecker, / 16) over each 256 channels of
    the last axis, in f64 or f32: the probe's reference, and the weights' decode."""
    m = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=x.dtype, device=x.device)
    shape = x.shape
    y = x.reshape(*shape[:-1], shape[-1] // 256, 4, 4, 4, 4)
    y = torch.einsum("...abcd,ia->...ibcd", y, m)
    y = torch.einsum("...abcd,ib->...aicd", y, m)
    y = torch.einsum("...abcd,ic->...abid", y, m)
    y = torch.einsum("...abcd,id->...abci", y, m)
    return y.reshape(shape) / 16


def _bands(module: ModuleType, g: torch.Generator, device: torch.device) -> verify.Verdict:
    """`frames` in bands against the whole, bit for bit, at every level of 7 frames: a 3x3x3 conv after the norm pass
    (two samples, each its own affine) with a residual, the same over its own input, and the temporal pass's 1x1x1
    conv."""

    def rand(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32)

    def filt(kt: int, pad: int) -> Filter:
        w = torch.randint(-127, 128, (CHANNELS, kt, kt, kt, CHANNELS), generator=g, device=device, dtype=torch.int8)
        return Filter(w, rand(CHANNELS, kt**3).abs_().mul_(1e-3).add_(1e-4), rand(CHANNELS), (pad,) * 3)

    x = (rand(2, 7, 5, 6, CHANNELS) * 2).bfloat16()
    a, b = rand(2, CHANNELS).abs_().add_(0.5), rand(2, CHANNELS)
    spatial, pointwise = filt(3, 1), filt(1, 0)
    mix = (rand(CHANNELS, TEMPORAL_TAPS) * 0.3, rand(CHANNELS))

    def over(level: int) -> Tensor | None:
        h = x.clone()
        return frames(module, h, a, b, spatial, torch.bfloat16, level, into=h)

    cases: dict[str, Callable[[int], Tensor | None]] = {
        "3x3x3": lambda level: frames(module, x, a, b, spatial, torch.bfloat16, level, residual=x),
        "3x3x3 over its input": over,
        "temporal 1x1x1": lambda level: frames(module, x, a, b, pointwise, torch.bfloat16, level, mix=mix, residual=x),
    }
    differ = []
    for case, run in cases.items():
        whole = run(0)
        for level in (1, 2, 3):
            got = run(level)
            if whole is None or got is None or not torch.equal(got, whole):
                differ.append(f"{case} at level {level}")
    if differ:
        return verify.Verdict(False, f"banded != whole: {', '.join(differ)}")
    return verify.Verdict(True, f"banded == whole bit for bit ({3 * len(cases)} cases)")


def _probe(index: int, module: ModuleType) -> verify.Verdict:
    """Every entry against float64 on this card: a 3x3x3 and a 1x1x1 conv of int8 operands with a bias and a residual
    (per-tap weight scales, padding at every border), the statistics, the quantize pass (its rotation by the
    reference's), and the temporal pass; and each in bands of frames against the whole, bit for bit (`_bands`)."""
    device = torch.device("cuda", index)
    g = torch.Generator(device=device).manual_seed(20261118)

    def rand(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=g, device=device, dtype=torch.float32)

    errors: dict[str, dict[str, float]] = {"conv": {}, "quantize": {}, "stats": {}}
    for kt, pad in ((3, 1), (1, 0)):
        q = torch.randint(-127, 128, (1, 3, 9, 11, CHANNELS), generator=g, device=device, dtype=torch.int8)
        qs = rand(1, 3, 9, 11).abs_().add_(0.1)
        w = torch.randint(-127, 128, (128, kt, kt, kt, CHANNELS), generator=g, device=device, dtype=torch.int8)
        ws = rand(128, kt**3).abs_().mul_(1e-3).add_(1e-4)
        bias = rand(128)
        x = (q.double() * qs.double()[..., None]).permute(0, 4, 1, 2, 3)
        filt = (w.double() * ws.double().view(128, kt, kt, kt, 1)).permute(0, 4, 1, 2, 3)
        want = torch.nn.functional.conv3d(x, filt, bias.double(), padding=pad).permute(0, 2, 3, 4, 1)
        residual = rand(*want.shape).bfloat16()
        got = conv(module, q, qs, Filter(w, ws, bias, (pad,) * 3), residual, torch.bfloat16)
        errors["conv"][f"{kt}x{kt}x{kt}"] = (
            float("inf") if got is None else verify.error(torch, got, want + residual.double())
        )
    x = (rand(2, 3, 5, 7, CHANNELS) * 3 + 1).bfloat16()
    mean, rstd = stats(module, x, 32, 1e-6)
    groups = x.double().reshape(2, -1, 32, CHANNELS // 32).transpose(1, 2).reshape(2, 32, -1)
    errors["stats"]["mean"] = verify.error(torch, mean, groups.mean(-1))
    errors["stats"]["rstd"] = verify.error(torch, rstd, (groups.var(-1, unbiased=False) + 1e-6).rsqrt())
    a, b = rand(2, CHANNELS).abs_().add_(0.5), rand(2, CHANNELS)
    q, qs = quantize(module, x, a, b, silu=True)
    want = rotate(torch.nn.functional.silu(x.double() * a.double()[:, None, None, None] + b.double()[:, None, None, None]))
    errors["quantize"]["quantize"] = verify.error(torch, q.double() * qs.double()[..., None], want)
    dw, db = rand(CHANNELS, TEMPORAL_TAPS) * 0.3, rand(CHANNELS)
    q, qs = temporal(module, x, a, b, dw, db)
    act = torch.nn.functional.silu(x.double() * a.double()[:, None, None, None] + b.double()[:, None, None, None])
    mixed = torch.nn.functional.conv3d(
        act.permute(0, 4, 1, 2, 3), dw.double().view(CHANNELS, 1, TEMPORAL_TAPS, 1, 1), db.double(),
        padding=(TEMPORAL_TAPS // 2, 0, 0), groups=CHANNELS,
    ).permute(0, 2, 3, 4, 1)  # fmt: skip
    errors["quantize"]["temporal"] = verify.error(torch, q.double() * qs.double()[..., None], rotate(mixed))
    verdicts = [verify.within(found, BUDGET[kind]) for kind, found in errors.items()]
    verdicts.append(_bands(module, g, device))
    return verify.Verdict(all(v.ok for v in verdicts), "; ".join(v.detail for v in verdicts))
