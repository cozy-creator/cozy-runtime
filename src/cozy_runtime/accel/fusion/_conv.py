"""A causal-conv video encoder's kernels (`_kernels._conv`, comfy-kitchen #167): one pass that
normalizes each frame (GroupNorm), applies SiLU and pads for the convolution, and an NDHWC conv3d
accumulating in fp16 with its bias and a residual in its epilogue. Activations are fp16
`channels_last_3d` from one to the next. A shape either kernel declines runs on torch, and so does
every call until the kernels pass their probe on this card (`kernel_ready`)."""

from __future__ import annotations

import sys
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, Literal

from cozy_runtime import _kernels
from cozy_runtime.accel.kernels import readable
from cozy_runtime.internal import kernel_ready, prebuilt, verify

if TYPE_CHECKING:
    import torch
    from torch import Tensor
else:
    try:
        import torch
    except ImportError:  # pragma: no cover - the base package is torch-free
        torch = None

#: A conv's padding: (left, right, top, bottom) pixels, the zero frames before, and what fills
#: the spatial border.
type Padding = tuple[tuple[int, int, int, int], int, Literal["reflect", "zeros"]]
#: A norm before a conv: (weight, bias, groups, eps), its statistics each frame's own.
type Norm = tuple[Tensor, Tensor, int, float]

#: The rel L2 a probe case may reach against float64: fp16 rounding (~5e-4), and fp16
#: accumulation over the probe's 432-deep reduction for the conv (~1e-2); garbage is O(1).
BUDGET = {"norm": 2e-3, "conv": 2e-2}


def kernels(index: int) -> ModuleType | None:
    """`_kernels._conv` once it passed its probe on cuda:`index`, else None: opening (off the
    request thread), failed, or absent from this wheel, said once (`kernel_ready`)."""
    capturing = torch.cuda.is_current_stream_capturing()
    return kernel_ready.serve(f"cozy conv cuda:{index}", partial(_open, index), capturing)


def _open(index: int, wait: bool) -> ModuleType:
    missing = _kernels.ABSENT or getattr(_kernels, "NO_CONV", "")
    if missing:
        raise kernel_ready.Refused(missing)
    major, minor = torch.cuda.get_device_capability(index)
    sm = major * 10 + minor
    if sm < 80:
        raise kernel_ready.Refused(f"the conv kernels run on sm80+, not sm{sm}")
    module = _kernels._conv
    artifact = prebuilt.artifact("_conv")
    return kernel_ready.verified(module, "prebuilt-conv", artifact, sm, partial(_probe, index))


def norm_pad(
    module: ModuleType, x: Tensor, norm: Norm | None, padding: Padding, *, silu: bool
) -> Tensor:
    """`x` `[N, C, T, H, W]` normalized per frame by `norm` (SiLU after it when `silu`) and
    padded, as `channels_last_3d` `[N, C, T + front, H + top + bottom, W + left + right]`; torch
    where the kernel declines."""
    (left, right, top, bottom), front, fill = padding
    n, c, t, h, w = (int(v) for v in x.shape)
    served = (
        c % 8 == 0
        and c <= 2048
        and n * (t + front) <= 65535
        and (fill == "zeros" or (max(left, right) < w and max(top, bottom) < h))
        and (norm is None or (c % norm[2] == 0 and norm[2] <= 1024))
    )
    if not served:
        padded = _eager_norm_pad(x, norm, padding, silu=silu)
        return padded.contiguous(memory_format=torch.channels_last_3d)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    out = torch.empty(
        (n, c, t + front, h + top + bottom, w + left + right),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )
    gamma: Tensor | None = None
    beta: Tensor | None = None
    workspace = out  # read only with a norm
    groups, eps = 1, 0.0
    if norm is not None:
        weight, bias, groups, eps = norm
        gamma, beta = weight.to(x.dtype).contiguous(), bias.to(x.dtype).contiguous()
        chunks = -(-(h * w) // 1024)
        size = 2 * n * t * (chunks * c + groups)
        workspace = torch.empty(size, dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    with readable(gamma, beta) as (gamma, beta):
        code = module.norm_pad(
            x.data_ptr(), 0 if gamma is None else gamma.data_ptr(),
            0 if beta is None else beta.data_ptr(), out.data_ptr(), workspace.data_ptr(),
            n, c, t, h, w, groups, float(eps), left, right, top, bottom, front, silu,
            fill == "zeros", w * c, x.dtype == torch.bfloat16, stream,
        )  # fmt: skip
    if code:
        raise RuntimeError(f"cozy conv: norm_pad: {module.error_name(code)}")
    return out


def conv3d(
    module: ModuleType | None,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    residual: Tensor | None,
    stride: tuple[int, int, int],
    *,
    last_tap: bool = False,
) -> Tensor:
    """`conv3d(x, weight) + bias + residual`, unpadded, `channels_last_3d` in and out: the
    kernel accumulating in fp16 (its deepest reductions in fp32) where `module` serves the shape,
    else torch's convolution. `last_tap`: `x` is one frame, convolved with the filter's last
    temporal tap alone (a 2-D convolution)."""
    with readable(weight, bias) as (weight, bias):
        if last_tap:
            weight = weight[:, :, -1:]
        weight = weight.contiguous(memory_format=torch.channels_last_3d)
        n, c, d, h, w = (int(v) for v in x.shape)
        k, _, kt, kr, ks = (int(v) for v in weight.shape)
        sd, sh, sw = stride
        z, p, q = (d - kt) // sd + 1, (h - kr) // sh + 1, (w - ks) // sw + 1
        half = torch.float16
        if (
            module is not None
            and x.dtype == half
            and weight.dtype == half
            and (c % 8 == 0 or c < 8)
            and k % 8 == 0
            and (bias is None or bias.dtype == half)
            and (
                residual is None
                or (residual.dtype == half and tuple(residual.shape) == (n, k, z, p, q))
            )
        ):
            if c < 8:  # a picture's channels, zero-padded to the kernel's 8-channel vectors
                x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 0, 0, 8 - c))
                weight = torch.nn.functional.pad(weight, (0, 0, 0, 0, 0, 0, 0, 8 - c))
                c = 8
            x = x.contiguous(memory_format=torch.channels_last_3d)
            weight = weight.contiguous(memory_format=torch.channels_last_3d)
            zeros = torch.zeros(k, dtype=half, device=x.device)
            rows = zeros if residual is None else residual.contiguous(
                memory_format=torch.channels_last_3d
            )
            vector = zeros if bias is None else bias.contiguous()
            out = torch.empty(
                (n, k, z, p, q), dtype=half, device=x.device, memory_format=torch.channels_last_3d
            )
            code = module.conv(
                x.data_ptr(), weight.data_ptr(), vector.data_ptr(), rows.data_ptr(),
                residual is not None, out.data_ptr(), n, d, h, w, c, k, kt, kr, ks, z, p, q,
                sd, sh, sw, -1, (0, 0, 0, 0), (0, 0, 0, 0),
                torch.cuda.current_stream(x.device).cuda_stream,
            )  # fmt: skip
            if code == 0:
                return out
            if code != -1:
                raise RuntimeError(f"cozy conv: conv3d: {module.error_name(code)}")
        x = x.contiguous(memory_format=torch.channels_last_3d)
        if last_tap:
            out = torch.nn.functional.conv2d(x[:, :, 0], weight[:, :, 0], bias, stride[1:])
            out = out.unsqueeze(2)
        else:
            out = torch.nn.functional.conv3d(x, weight, bias, stride)
        out = out.contiguous(memory_format=torch.channels_last_3d)
    return out if residual is None else out.add_(residual)


def _eager_norm_pad(x: Tensor, norm: Norm | None, padding: Padding, *, silu: bool) -> Tensor:
    """`norm_pad` on torch: the probe's reference, and every shape the kernel declines."""
    (left, right, top, bottom), front, fill = padding
    if norm is not None:
        weight, bias, groups, eps = norm
        n, c, t, h, w = x.shape
        frames = x.transpose(1, 2).reshape(n * t, c, h, w)
        frames = torch.nn.functional.group_norm(
            frames, groups, weight.to(x.dtype), bias.to(x.dtype), eps
        )
        x = frames.reshape(n, t, c, h, w).transpose(1, 2)
    if silu:
        x = torch.nn.functional.silu(x)
    if any((left, right, top, bottom)):
        mode = "constant" if fill == "zeros" else "reflect"
        x = torch.nn.functional.pad(x, (left, right, top, bottom, 0, 0), mode=mode)
    if front:
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, front, 0))
    return x


def _probe(index: int, module: ModuleType) -> verify.Verdict:
    """Both kernels against float64 on this card: frames normalized, activated and padded,
    reflecting and with zeros; then each conv tile config (fp16 accumulation in 128x256 and
    128x128 tiles, fp32's 64x64) with a bias and a residual, strided too."""
    device = torch.device("cuda", index)
    generator = torch.Generator(device=device).manual_seed(20261117)

    def rand(*shape: int, scale: float = 1.0) -> Tensor:
        drawn = torch.randn(*shape, generator=generator, device=device, dtype=torch.float32)
        return drawn.mul_(scale)

    errors: dict[str, float] = {}
    x = rand(2, 16, 3, 10, 12).add_(0.5)
    gamma, beta = rand(16, scale=0.5).add_(1.0), rand(16, scale=0.1)
    fills: tuple[Literal["reflect", "zeros"], ...] = ("reflect", "zeros")
    for fill in fills:
        padding: Padding = ((1, 1, 1, 1), 2, fill)
        exact = (gamma.double(), beta.double(), 4, 1e-6)
        want = _eager_norm_pad(x.double(), exact, padding, silu=True)
        got = norm_pad(module, x.half(), (gamma.half(), beta.half(), 4, 1e-6), padding, silu=True)
        errors[f"norm {fill}"] = verify.error(torch, got, want)
    padded = rand(1, 16, 5, 34, 36).half().contiguous(memory_format=torch.channels_last_3d)
    weight, bias = rand(16, 16, 3, 3, 3, scale=0.1).half(), rand(16, scale=0.1).half()
    filters = weight.contiguous(memory_format=torch.channels_last_3d)
    stream = torch.cuda.current_stream(device).cuda_stream
    n, c, d, h, w = (int(v) for v in padded.shape)
    for stride in ((1, 1, 1), (2, 2, 2)):
        want = torch.nn.functional.conv3d(padded.double(), weight.double(), bias.double(), stride)
        residual = rand(*want.shape).half().contiguous(memory_format=torch.channels_last_3d)
        want = want + residual.double()
        z, p, q = (int(v) for v in want.shape[2:])
        for config in (0, 1, 2):
            out = torch.empty_like(residual)
            code = module.conv(
                padded.data_ptr(), filters.data_ptr(), bias.data_ptr(), residual.data_ptr(),
                True, out.data_ptr(), n, d, h, w, c, 16, 3, 3, 3, z, p, q, *stride, config,
                (0, 0, 0, 0), (0, 0, 0, 0), stream,
            )  # fmt: skip
            case = f"conv config {config} stride {stride[0]}"
            errors[case] = verify.error(torch, out, want) if code == 0 else float("inf")
            if code:
                print(f"[kernels] cozy conv: {case} returned {code}", file=sys.stderr)
    norms = {case: e for case, e in errors.items() if case.startswith("norm")}
    convs = {case: e for case, e in errors.items() if case.startswith("conv")}
    verdicts = (verify.within(norms, BUDGET["norm"]), verify.within(convs, BUDGET["conv"]))
    return verify.Verdict(all(v.ok for v in verdicts), "; ".join(v.detail for v in verdicts))
