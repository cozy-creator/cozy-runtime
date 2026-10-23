"""The `svdq.nvfp4` linear (xs-066): SVDQuant on Blackwell's FP4 tensor cores, v1's engine.

`y = (q(x · p) · Wᵀ) · g_x · g_w · c + (x · downᵀ) · upᵀ + bias`.
- `W` is the stored nvfp4 residual: E2M1 codes and E4M3 scales per 16 columns, second level `g_w`.
- `p` is the smoothing (`pre_quant_scale`); `c` (`channel_scale`) and the bias are optional.
- `q` rounds `x · p` to bf16, then to nvfp4 at the activation's second level `g_x`. That is
  `input_scale` (calibrated), else the call's `|x| · p` max / 2688. Each 16-block's scale is
  `amax / (6 g_x)` rounded to E4M3 within [2⁻⁹, 448], each element the nearest E2M1, ties even.

Two Triton kernels, built when the executor starts (`KERNELS`):
- `_quantize` reads `x` once: it writes the nvfp4 operand and the down-projection `x · downᵀ`;
- `_gemm` multiplies on the FP4 tensor cores (`tl.dot_scaled`); its epilogue adds the
  up-projection, both second levels, `c` and the bias.
Until the build lands, and off Blackwell, `eager` serves the same law with torch ops.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import TYPE_CHECKING

from cozy_runtime.author._model import Building
from cozy_runtime.author.kernels import KernelsUnavailable, Launch, TritonEntry, TritonKernels
from cozy_runtime.internal import int8_linear
from cozy_runtime.internal.encoding.formats import E2M1_VALUES, NVFP4_BLOCK, Tensor

if TYPE_CHECKING:
    from torch import device as Device

#: E4M3's largest finite value times E2M1's: the second level that maps a tensor's max to the top.
LEVELS_MAX = 448.0 * 6.0
#: E4M3's smallest subnormal: a block scale never rounds below it (an all-zero block encodes 0).
SCALE_FLOOR = 2.0**-9
#: The ranks the kernels are built for: a branch is padded with zero rows to the next one.
RANKS = (32, 64, 128)
#: The tiles: rows quantized per program, and the GEMM's output tile, K step, warps and stages
#: (v1's best of its twelve on an sm120 card: 5-12% over 128x128x128 at 3 stages).
QUANT_ROWS, QUANT_K = 32, 128
TILE_M, TILE_N, TILE_K, GROUP_M, GEMM_WARPS, GEMM_STAGES = 128, 128, 256, 8, 8, 2
#: The column multiple the kernels need (one K step).
K_ALIGN = TILE_K
#: What a run's `gemm.selected` names.
TRITON, EAGER = "cozy-svdq-nvfp4", "torch-eager"

try:
    import triton
    import triton.language as tl
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    TRITON_ABSENT = f"triton does not import ({_absent})"
else:
    TRITON_ABSENT = ""

if not TRITON_ABSENT:

    @triton.jit
    def _e2m1(v):  # type: ignore[no-untyped-def]
        """The nearest E2M1 code of `v` (sign in bit 3), ties to the even code."""
        a = tl.abs(v)
        code = (
            (a > 0.25).to(tl.uint8)
            + (a >= 0.75).to(tl.uint8)
            + (a > 1.25).to(tl.uint8)
            + (a >= 1.75).to(tl.uint8)
            + (a > 2.5).to(tl.uint8)
            + (a >= 3.5).to(tl.uint8)
            + (a > 5.0).to(tl.uint8)
        )
        return code + (v < 0).to(tl.uint8) * 8

    @triton.jit  # type: ignore[untyped-decorator]
    def _quantize(  # type: ignore[no-untyped-def]
        x_ptr,
        p_ptr,
        g_ptr,
        down_ptr,
        q_ptr,
        s_ptr,
        la_ptr,
        M,
        K,
        R: tl.constexpr,
        BM: tl.constexpr,
        BK: tl.constexpr,
        HALF: tl.constexpr,
    ):
        """`BM` rows of `x`: the nvfp4 operand of `x · p` (codes `[M, K/2]`, E4M3 scales
        `[M, K/16]`) and `x · downᵀ` `[M, R]` (bf16), reading each element once."""
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        ok = rm < M
        rows = rm[:, None].to(tl.int64)
        rk = tl.arange(0, BK)
        rr = tl.arange(0, R)
        g = tl.load(g_ptr)
        la = tl.zeros((BM, R), dtype=tl.float32)
        for k0 in range(0, K, BK):
            x = tl.load(x_ptr + rows * K + (k0 + rk)[None, :], mask=ok[:, None], other=0.0)
            down = tl.load(down_ptr + rr[:, None] * K + (k0 + rk)[None, :])
            la = tl.dot(x, tl.trans(down).to(x.dtype), la)
            p = tl.load(p_ptr + k0 + rk)
            xs = (x.to(tl.float32) * p[None, :]).to(tl.bfloat16).to(tl.float32)
            blocks = tl.reshape(xs, (BM, BK // 16, 16))
            amax = tl.max(tl.abs(blocks), axis=2)
            scale = tl.minimum(tl.maximum(tl.math.div_rn(amax, 6.0 * g), 0.001953125), 448.0)
            scale8 = scale.to(tl.float8e4nv)
            unit = scale8.to(tl.float32) * g
            codes = _e2m1(tl.math.div_rn(blocks, unit[:, :, None]))
            lo, hi = tl.split(tl.reshape(codes, (BM, BK // 2, 2)))
            kp = k0 // 2 + tl.arange(0, BK // 2)
            tl.store(q_ptr + rows * (K // 2) + kp[None, :], lo | (hi << 4), mask=ok[:, None])
            ks = k0 // 16 + tl.arange(0, BK // 16)
            tl.store(s_ptr + rows * (K // 16) + ks[None, :], scale8, mask=ok[:, None])
        tl.store(la_ptr + rows * R + rr[None, :], la.to(tl.bfloat16), mask=ok[:, None])

    @triton.jit  # type: ignore[untyped-decorator]
    def _gemm(  # type: ignore[no-untyped-def]
        q_ptr,
        sx_ptr,
        w_ptr,
        sw_ptr,
        gx_ptr,
        gw_ptr,
        c_ptr,
        la_ptr,
        up_ptr,
        bias_ptr,
        out_ptr,
        M,
        N,
        K,
        HAS_C,
        HAS_BIAS,
        R: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP: tl.constexpr,
        HALF: tl.constexpr,
    ):
        """One `[BM, BN]` output tile: the FP4 product, then `x g_x g_w c`, the up-projection
        and the bias, stored once; `c` and the bias are read only when `HAS_C` and `HAS_BIAS`.
        Tiles walk N in bands of `GROUP` row tiles (L2 reuse)."""
        pid = tl.program_id(0)
        tiles_m, tiles_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
        band = pid // (GROUP * tiles_n)
        first = band * GROUP
        height = tl.minimum(tiles_m - first, GROUP)
        pm = first + (pid % (GROUP * tiles_n)) % height
        pn = (pid % (GROUP * tiles_n)) // height
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        m_ok, n_ok = rm < M, rn < N
        rows, cols = rm[:, None].to(tl.int64), rn[:, None].to(tl.int64)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kp = k0 // 2 + tl.arange(0, BK // 2)
            ks = k0 // 16 + tl.arange(0, BK // 16)
            qa = tl.load(q_ptr + rows * (K // 2) + kp[None, :], mask=m_ok[:, None], other=0)
            sa = tl.load(sx_ptr + rows * (K // 16) + ks[None, :], mask=m_ok[:, None], other=0)
            qb = tl.load(w_ptr + cols * (K // 2) + kp[None, :], mask=n_ok[:, None], other=0)
            sb = tl.load(sw_ptr + cols * (K // 16) + ks[None, :], mask=n_ok[:, None], other=0)
            acc = tl.dot_scaled(
                qa, sa.to(tl.float8e4nv, bitcast=True), "e2m1",
                tl.trans(qb), sb.to(tl.float8e4nv, bitcast=True), "e2m1", acc,
            )  # fmt: skip
        c = tl.load(c_ptr + rn, mask=n_ok & (HAS_C != 0), other=1.0).to(tl.float32)
        y = acc * (tl.load(gx_ptr) * tl.load(gw_ptr)) * c[None, :]
        rr = tl.arange(0, R)
        la = tl.load(la_ptr + rows * R + rr[None, :], mask=m_ok[:, None], other=0.0)
        up = tl.load(up_ptr + cols * R + rr[None, :], mask=n_ok[:, None], other=0.0)
        y = tl.dot(la, tl.trans(up), y)
        y += tl.load(bias_ptr + rn, mask=n_ok & (HAS_BIAS != 0), other=0.0).to(tl.float32)[None, :]
        tl.store(
            out_ptr + rows * N + rn[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=m_ok[:, None] & n_ok[None, :],
        )


def _signature(kernel: object, **types: str) -> dict[str, str]:
    """Triton's signature of a kernel: constexprs by their annotations, every other pointer bf16
    or as named, the sizes i32."""
    out = {}
    for p in kernel.params:  # type: ignore[attr-defined]
        if p.is_constexpr:
            out[p.name] = "constexpr"
        elif p.name.endswith("_ptr"):
            out[p.name] = types.get(p.name, "*bf16")
        else:
            out[p.name] = "i32"
    return out


def _entries() -> list[TritonEntry]:
    """Every specialization `linear` launches: bf16 or fp16 activations (`HALF` 0 or 1), each
    built rank."""
    if TRITON_ABSENT:
        return []
    entries = []
    for code, half in enumerate(("bf16", "fp16")):
        for r in RANKS:
            entries.append(
                TritonEntry(
                    _quantize,
                    _signature(
                        _quantize,
                        x_ptr=f"*{half}",
                        p_ptr="*fp32",
                        g_ptr="*fp32",
                        q_ptr="*u8",
                        s_ptr="*fp8e4nv",
                    ),
                    {"R": r, "BM": QUANT_ROWS, "BK": QUANT_K, "HALF": code},
                    aligned=("x_ptr", "p_ptr", "down_ptr", "q_ptr", "s_ptr", "la_ptr", "K"),
                    num_warps=4,
                )
            )
            fp32 = {name: "*fp32" for name in ("gx_ptr", "gw_ptr", "c_ptr")}
            u8 = {name: "*u8" for name in ("q_ptr", "sx_ptr", "w_ptr", "sw_ptr")}
            entries.append(
                TritonEntry(
                    _gemm,
                    _signature(_gemm, out_ptr=f"*{half}", bias_ptr=f"*{half}", **fp32, **u8),
                    {
                        "R": r,
                        "BM": TILE_M,
                        "BN": TILE_N,
                        "BK": TILE_K,
                        "GROUP": GROUP_M,
                        "HALF": code,
                    },
                    aligned=(*u8, "la_ptr", "up_ptr", "out_ptr", "K", "N"),
                    num_warps=GEMM_WARPS,
                    num_stages=GEMM_STAGES,
                )
            )
    return entries


#: Built on every executor's GPUs when it starts (`author.kernels`): a request never compiles.
KERNELS = TritonKernels("cozy-svdq-nvfp4", _entries())
#: How often a call asks whether the start's build has landed: resolution, never a verdict.
POLL_SECONDS = 1.0
_LAUNCH: dict[int, tuple[float, Launch | None]] = {}
#: `triton_kernels.ready`, bound by `triton_kernels` itself when it is imported (past the
#: executor's device boundary): this module imports neither Triton's runtime nor torch.
READY: list[Callable[..., Launch | None]] = []
#: Whether each GPU has FP4 tensor cores (Blackwell): only there do the kernels serve.
_FP4: dict[int, bool] = {}
#: E2M1's values per device, made outside any graph capture (a host copy would end one).
_TABLES: dict[object, Tensor] = {}


def rank(r: int) -> int:
    """The built rank a branch of rank `r` is padded to."""
    for built in RANKS:
        if r <= built:
            return built
    raise ValueError(f"a low-rank branch of rank {r} exceeds the kernels' {RANKS[-1]}")


def padded(torch: ModuleType, down: Tensor, up: Tensor) -> tuple[Tensor, Tensor]:
    """A branch `(down [r, K], up [N, r])` in bf16, zero-padded to its built rank (`rank`)."""
    r, wide = int(down.shape[0]), rank(int(down.shape[0]))
    down = torch.cat(
        [down.to(torch.bfloat16), down.new_zeros(wide - r, down.shape[1], dtype=torch.bfloat16)]
    )
    up = torch.cat(
        [up.to(torch.bfloat16), up.new_zeros(up.shape[0], wide - r, dtype=torch.bfloat16)], 1
    )
    return down.contiguous(), up.contiguous()


def merged(
    torch: ModuleType, roles: Mapping[str, Tensor], down: Tensor, up: Tensor
) -> dict[str, Tensor] | None:
    """The low-rank roles with a LoRA's factors as more ranks: `x · downᵀ · upᵀ` gains
    `x · Aᵀ · (s B)ᵀ` with no rounding of the weight. None when the ranks together exceed the
    kernels' largest: the caller adds the LoRA's term itself."""
    if int(roles["lowrank_down"].shape[0]) + int(down.shape[0]) > RANKS[-1]:
        return None
    both = torch.cat([roles["lowrank_down"], down.to(torch.bfloat16)])
    wide = torch.cat([roles["lowrank_up"], up.to(torch.bfloat16)], dim=1)
    both, wide = padded(torch, both, wide)
    return {"lowrank_down": both, "lowrank_up": wide}


def _launcher(torch: ModuleType, device: Device) -> Launch | None:
    """This GPU's launcher once the start's build has landed and agrees with the law here
    (`verify`), else None (asked again after `POLL_SECONDS`): the caller then runs `eager`."""
    index = device.index or 0
    if index not in _FP4:
        _FP4[index] = int(torch.cuda.get_device_capability(index)[0]) >= 10
    if not _FP4[index] or TRITON_ABSENT:
        return None
    asked, launch = _LAUNCH.get(index, (float("-inf"), None))
    now = time.monotonic()
    if launch is None and now - asked >= POLL_SECONDS:
        try:
            launch = READY[0](KERNELS, device=device) if READY else None
            if launch is not None:
                verify(torch, device, launch)
        except KernelsUnavailable as exc:  # absent, failed or corrupt for good: the run says so
            int8_linear.failed(TRITON, index, exc)
            _FP4[index] = False
            launch = None
        except Exception as exc:  # broken kernels never serve: the law does, and the run says so
            int8_linear.failed(TRITON, index, exc)
            _FP4[index] = False
            launch = None
        _LAUNCH[index] = (now, launch)
    return launch


def verify(torch: ModuleType, device: Device, launch: Launch) -> None:
    """Both kernels against the law on ragged rows and columns, each dtype and built rank, with
    and without `c` and the bias: the operand and its scales bit for bit; the down-projection and
    the output against the law's exact (f64) products, within their bf16 or fp16 rounding. Raises
    on a miss."""
    generator = torch.Generator(device=device).manual_seed(0)
    rows, n, k = 200, 144, 256

    def draw(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, device=device)

    def far(got: Tensor, want: Tensor) -> float:
        return float((got.double() - want).norm() / want.norm())

    for r in RANKS:
        roles = {
            "weight": torch.randint(0, 256, (n, k // 2), generator=generator, device=device).to(
                torch.uint8
            ),
            "scale": (draw(n, k // NVFP4_BLOCK).abs() + 0.5).to(torch.float8_e4m3fn),
            "scale_2": torch.full((1,), 0.01, device=device),
            "pre_quant_scale": draw(k).abs() + 0.5,
            "lowrank_down": (draw(r, k) * 0.05).to(torch.bfloat16),
            "lowrank_up": (draw(n, r) * 0.05).to(torch.bfloat16),
        }
        for half, extras in ((torch.bfloat16, True), (torch.float16, False)):
            x = draw(rows, k).to(half)
            bias = draw(n).to(half) if extras else None
            if extras:
                roles = {**roles, "channel_scale": draw(n).abs() + 0.5}
            else:
                roles = {role: t for role, t in roles.items() if role != "channel_scale"}
            q, sx, low, g = _quantized(torch, launch, x, roles)
            codes, scales = quantize(torch, _smoothed(torch, x, roles), g)
            if not (
                torch.equal(q, codes[:, 0::2] | (codes[:, 1::2] << 4))
                and torch.equal(sx.view(torch.uint8), scales.view(torch.uint8))
            ):
                raise RuntimeError(f"the svdq quantize kernel is not the law (rank {r}, {half})")
            down = x.double() @ roles["lowrank_down"].to(half).double().t()
            if not far(low, down) < 5e-3:
                raise RuntimeError(f"the svdq down-projection is {far(low, down):.3g} off ({half})")
            packed = roles["weight"]
            nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).view(n, k)
            weight = _values(torch, nibbles, roles["scale"], roles["scale_2"]).double()
            want = (_values(torch, codes, scales, g).double() @ weight.t()) * (
                roles["channel_scale"].double() if extras else 1.0
            )
            want = want + low.double() @ roles["lowrank_up"].double().t()
            if bias is not None:
                want = want + bias.double()
            got = _multiplied(torch, launch, q, sx, low, g, roles, bias, half)
            if not far(got, want) < 5e-3:
                raise RuntimeError(f"the svdq GEMM is {far(got, want):.3g} from the law ({half})")


def serving(index: int) -> str:
    """The kernel this card's svdq leaves serve now: a run's `gemm.selected`."""
    return TRITON if _LAUNCH.get(index, (0.0, None))[1] is not None else EAGER


def activation_scale(torch: ModuleType, flat: Tensor, roles: Mapping[str, Tensor]) -> Tensor:
    """`g_x`: the stored `input_scale`, else this call's `max_j(max_i |x_ij| · p_j)` over
    `LEVELS_MAX` (exactly the max of `|x · p|`: `p` is positive)."""
    stored = roles.get("input_scale")
    if stored is not None:
        return stored.float().reshape(1)
    amax = (flat.abs().amax(0).float() * roles["pre_quant_scale"]).amax()
    return (amax / LEVELS_MAX).clamp(min=1e-30).reshape(1)


def _smoothed(torch: ModuleType, x: Tensor, roles: Mapping[str, Tensor]) -> Tensor:
    """`bf16(x · p)` in f32 `[M, K]`: the operand the activation's nvfp4 is taken of."""
    flat = x.reshape(-1, int(roles["pre_quant_scale"].numel()))
    return (flat.float() * roles["pre_quant_scale"]).to(torch.bfloat16).float()


def _down(torch: ModuleType, x: Tensor, down: Tensor) -> Tensor:
    """`x · downᵀ` from `x` in its half dtype (bf16 for f32), accumulated in f32, then bf16."""
    flat = x.reshape(-1, int(down.shape[1]))
    half = flat.dtype if flat.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    return (flat.to(half) @ down.to(half).t()).to(torch.bfloat16)


def quantize(torch: ModuleType, xs: Tensor, g: Tensor) -> tuple[Tensor, Tensor]:
    """`xs` `[M, K]` (f32, already `bf16(x · p)`) -> nvfp4 codes `[M, K]` (u8, sign in bit 3)
    and E4M3 scales `[M, K/16]` at second level `g`: the kernels' law, in torch ops."""
    rows, cols = int(xs.shape[0]), int(xs.shape[1])
    blocks = xs.view(rows, cols // NVFP4_BLOCK, NVFP4_BLOCK)
    amax = blocks.abs().amax(-1)
    scale = (amax / (6.0 * g)).clamp(SCALE_FLOOR, 448.0).to(torch.float8_e4m3fn)
    a = (blocks / (scale.float() * g).unsqueeze(-1)).abs()
    code = sum(
        ((a >= t) if odd else (a > t)).to(torch.uint8)
        for t, odd in ((0.25, 0), (0.75, 1), (1.25, 0), (1.75, 1), (2.5, 0), (3.5, 1), (5.0, 0))
    )
    code = code + (blocks < 0).to(torch.uint8) * 8
    return code.view(rows, cols), scale


def _table(torch: ModuleType, device: object) -> Tensor:
    if device not in _TABLES:
        _TABLES[device] = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
    return _TABLES[device]


def _values(torch: ModuleType, codes: Tensor, scales: Tensor, g: Tensor) -> Tensor:
    """nvfp4 codes `[R, C]` (u8, sign in bit 3) with E4M3 scales `[R, C/16]` at second level `g`,
    in f32: `(E2M1 · scale) · g`."""
    blocks = _table(torch, codes.device)[codes.int()].view(codes.shape[0], -1, NVFP4_BLOCK)
    return (blocks * scales.float().unsqueeze(-1) * g.float().reshape(())).view(codes.shape)


def fake(torch: ModuleType, xs: Tensor, g: Tensor) -> Tensor:
    """`xs` through nvfp4 at second level `g` (`quantize`), back to f32."""
    return _values(torch, *quantize(torch, xs, g), g)


def eager(
    torch: ModuleType,
    x: Tensor,
    roles: Mapping[str, Tensor],
    bias: Tensor | None,
    out_dtype: object,
) -> Tensor:
    """The law in torch ops: what the kernels compute, on any device."""
    cols = int(roles["pre_quant_scale"].numel())
    flat = x.reshape(-1, cols)
    g = activation_scale(torch, flat, roles)
    packed = roles["weight"]
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).view(packed.shape[0], -1)
    weight = _values(torch, codes, roles["scale"], roles["scale_2"])
    y = fake(torch, _smoothed(torch, x, roles), g) @ weight.t()
    if "channel_scale" in roles:
        y = y * roles["channel_scale"].float()
    y = y + _down(torch, x, roles["lowrank_down"]).float() @ roles["lowrank_up"].float().t()
    if bias is not None:
        y = y + bias.float()
    return y.to(out_dtype).view(*x.shape[:-1], -1)


def linear(
    torch: ModuleType,
    x: Tensor,
    roles: Mapping[str, Tensor],
    bias: Tensor | None,
    out_dtype: object,
) -> Tensor:
    """`x @ Wᵀ + bias` for one leaf's roles (`SvdqNvfp4NativeLeaf`): `weight` u8 `[N, K/2]`,
    `scale` E4M3 `[N, K/16]`, `scale_2` f32, `pre_quant_scale` f32 `[K]`, `lowrank_down` bf16
    `[R, K]` and `lowrank_up` bf16 `[N, R]` (R one of `RANKS`), optional `channel_scale` `[N]` and
    `input_scale`. The kernels serve a half `x` written in its own dtype; the law the rest."""
    launch = None
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and out_dtype == x.dtype:
        launch = _launcher(torch, x.device)
        if (
            launch is None
            and _FP4.get(x.device.index or 0)
            and torch.cuda.is_current_stream_capturing()
        ):
            raise Building(TRITON)  # recording the law now would keep it: record once they land
    if launch is None:
        return eager(torch, x, roles, bias, out_dtype)
    try:
        q, sx, low, g = _quantized(torch, launch, x, roles)
        out = _multiplied(torch, launch, q, sx, low, g, roles, bias, x.dtype)
    except Exception as exc:  # a launch that raises: the law serves this call, and the run says so
        int8_linear.failed(TRITON, x.device.index or 0, exc)
        return eager(torch, x, roles, bias, out_dtype)
    return out.view(*x.shape[:-1], -1)


def _quantized(
    torch: ModuleType, launch: Launch, x: Tensor, roles: Mapping[str, Tensor]
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """`_quantize`: the packed codes, E4M3 scales and down-projection of `x`, and its `g_x`."""
    cols, r = int(roles["pre_quant_scale"].numel()), int(roles["lowrank_down"].shape[0])
    flat = x.reshape(-1, cols).contiguous()
    rows = int(flat.shape[0])
    g = activation_scale(torch, flat, roles).contiguous()
    q = torch.empty(rows, cols // 2, dtype=torch.uint8, device=x.device)
    sx = torch.empty(rows, cols // NVFP4_BLOCK, dtype=torch.float8_e4m3fn, device=x.device)
    low = torch.empty(rows, r, dtype=torch.bfloat16, device=x.device)
    launch(
        _quantize, (-(-rows // QUANT_ROWS), 1, 1),
        flat, roles["pre_quant_scale"], g, roles["lowrank_down"], q, sx, low, rows, cols,
        r, QUANT_ROWS, QUANT_K, int(flat.dtype == torch.float16),
    )  # fmt: skip
    return q, sx, low, g


def _multiplied(
    torch: ModuleType,
    launch: Launch,
    q: Tensor,
    sx: Tensor,
    low: Tensor,
    g: Tensor,
    roles: Mapping[str, Tensor],
    bias: Tensor | None,
    half: object,
) -> Tensor:
    """`_gemm`: `[M, N]` in `half`, from `_quantized`'s operands."""
    rows, cols = int(q.shape[0]), 2 * int(q.shape[1])
    n, r = int(roles["lowrank_up"].shape[0]), int(roles["lowrank_up"].shape[1])
    c = roles.get("channel_scale")
    out = torch.empty(rows, n, dtype=half, device=q.device)
    launch(
        _gemm, (-(-rows // TILE_M) * -(-n // TILE_N), 1, 1),
        q, sx.view(torch.uint8), roles["weight"], roles["scale"].view(torch.uint8), g,
        roles["scale_2"], g if c is None else c, low, roles["lowrank_up"],
        out if bias is None else bias.to(half), out, rows, n, cols, int(c is not None),
        int(bias is not None), r, TILE_M, TILE_N, TILE_K, GROUP_M, int(half == torch.float16),
    )  # fmt: skip
    return out
