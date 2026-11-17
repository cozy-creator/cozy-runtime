"""int8-convrot convolution (cr-327): `_kernels._conv`'s NDHWC conv3d on the int8 tensor cores and the passes that
feed it (`kernels/csrc/conv/conv8.cuh`).

An int8.convrot conv weight is stored as its filter with the input channels last, `[O, kt, kh, kw, I]`
(`formats.conv_declared`): a row for each (output channel, tap), rotated over the input channels by the regular H256 and
scaled per row. Activations are rotated the same way and quantized with one scale per voxel (`quantize`), a GroupNorm's
normalization, an affine and SiLU in the same pass where a block has them. A voxel's scale differs between the taps an
output sums, so each tap's int32 sum is promoted to f32 with its voxel and weight scales: the conv is exact per tap.
Activations are NDHWC (`channels_last_3d`), 512 channels a voxel for the passes. Every call opens the kernels through
`kernel_ready` once they pass their probe on the card (`kernels`)."""

from __future__ import annotations

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


def quantize(module: ModuleType, x: Tensor, a: Tensor, b: Tensor, *, silu: bool) -> tuple[Tensor, Tensor]:
    """NDHWC `x` [N, ..., 512] (fp16 or bf16) as `silu(a x + b)` (or `a x + b`), rotated and quantized a voxel: int8
    codes like `x` and their f32 scales [N, ...]."""
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    qs = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    voxels = x.numel() // CHANNELS
    code = module.conv8_quant(x.data_ptr(), a.data_ptr(), b.data_ptr(), q.data_ptr(), qs.data_ptr(), voxels,
                              voxels // int(x.shape[0]), silu, x.dtype == torch.float16, _sms(x.device),
                              _stream(x.device))  # fmt: skip
    _check(module, "quantize", code)
    return q, qs


def temporal(
    module: ModuleType, x: Tensor, a: Tensor, b: Tensor, weight: Tensor, bias: Tensor | None
) -> tuple[Tensor, Tensor]:
    """`quantize` of the depthwise temporal conv of `silu(a x + b)`: `weight` [512, 5] f32 (`TEMPORAL_TAPS`, zero
    padding 2), `bias` [512] f32; `x` NDHWC [N, T, H, W, 512]."""
    n, t = int(x.shape[0]), int(x.shape[1])
    hw = x.numel() // (n * t * CHANNELS)
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    qs = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    code = module.conv8_temporal(x.data_ptr(), a.data_ptr(), b.data_ptr(), weight.data_ptr(),
                                 0 if bias is None else bias.data_ptr(), q.data_ptr(), qs.data_ptr(), n, t, hw,
                                 x.dtype == torch.float16, _sms(x.device), _stream(x.device))  # fmt: skip
    _check(module, "temporal", code)
    return q, qs


def conv(
    module: ModuleType,
    q: Tensor,
    qs: Tensor,
    weight: Tensor,
    scale: Tensor,
    bias: Tensor | None,
    residual: Tensor | None,
    padding: tuple[int, int, int],
    dtype: torch.dtype,
) -> Tensor | None:
    """`conv3d(x, w) + bias + residual`, stride 1, zero `padding`: `q` int8 NDHWC [N, D, H, W, C] with `qs` [N, D, H,
    W], `weight` int8 [K, kt, kh, kw, C] with `scale` f32 [K, taps], `bias` f32 [K], `residual` NDHWC like the output
    (`dtype`, fp16 or bf16). The output NDHWC [N, Z, P, Q, K]; None where the kernels do not serve the shape."""
    n, d, h, w, c = (int(v) for v in q.shape)
    k, t, r, s = (int(v) for v in weight.shape[:4])
    pt, pr, ps = padding
    z, p, qq = d + 2 * pt - t + 1, h + 2 * pr - r + 1, w + 2 * ps - s + 1
    out = torch.empty((n, z, p, qq, k), dtype=dtype, device=q.device)
    major, minor = torch.cuda.get_device_capability(q.device)
    tiles = config(major * 10 + minor)
    if tiles is None:
        return None
    code = module.conv8(q.data_ptr(), qs.data_ptr(), weight.data_ptr(), scale.data_ptr(),
                        0 if bias is None else bias.data_ptr(), 0 if residual is None else residual.data_ptr(),
                        out.data_ptr(), n, d, h, w, c, k, t, r, s, pt, pr, ps, z, p, qq, True, dtype == torch.float16,
                        tiles, _stream(q.device))  # fmt: skip
    if code == -1:
        return None
    _check(module, "conv", code)
    return out


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


def _probe(index: int, module: ModuleType) -> verify.Verdict:
    """Every entry against float64 on this card: a 3x3x3 and a 1x1x1 conv of int8 operands with a bias and a residual
    (per-tap weight scales, padding at every border), the statistics, the quantize pass (its rotation by the
    reference's), and the temporal pass."""
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
        got = conv(module, q, qs, w, ws, bias, residual, (pad,) * 3, torch.bfloat16)
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
    return verify.Verdict(all(v.ok for v in verdicts), "; ".join(v.detail for v in verdicts))
