"""The `svdq.nvfp4` linear (xs-066): SVDQuant on Blackwell's FP4 tensor cores, v1's engine.

`y = (q(x · p) · Wᵀ) · g_x · g_w · c + (x · downᵀ) · upᵀ + bias`.
- `W` is the stored nvfp4 residual: E2M1 codes and E4M3 scales per 16 columns, second level `g_w`.
- `p` is the smoothing (`pre_quant_scale`); `c` (`channel_scale`) and the bias are optional.
- `q` rounds `x · p` to bf16, then to nvfp4 at the activation's second level `g_x`. That is
  `input_scale` (calibrated), else the call's `|x| · p` max / 2688. Each 16-block's scale is
  `amax / (6 g_x)` rounded to E4M3 within [2⁻⁹, 448], each element the nearest E2M1, ties even.

On a card whose row is block-scaled FP4 mma.sync (RTX 50, DGX Spark: `arch`), the wheel's prebuilt
kernels serve (`_kernels._gemm.svdq_*`, `kernels/csrc/gemm/svdq.cu`): a leaf there holds its weight
in their tiles from the bind (`tiled`, `tiles`), and they serve once they pass `probe_native` on
the card. Until then, and on a card where they never do, a call untiles the weight and the Triton
kernels below serve, else the law, and the run says so.

Elsewhere on Blackwell, two Triton kernels built when the executor starts (`KERNELS`):
- `_quantize` reads `x` once: it writes the nvfp4 operand and the down-projection `x · downᵀ`;
- `_gemm` multiplies on the FP4 tensor cores (`tl.dot_scaled`); its epilogue adds the
  up-projection, both second levels, `c` and the bias.
They serve once they load and pass `probe` on a card whose row has FP4 tensor cores (`arch`,
`kernel_ready.serve`); until then, and elsewhere, `eager` serves the same law with torch ops.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING, cast

from cozy_runtime import _kernels
from cozy_runtime.author.kernels import Launch, TritonEntry, TritonKernels
from cozy_runtime.internal import arch, int8_linear, kernel_ready, prebuilt, verify
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
TRITON, EAGER, NATIVE = "cozy-svdq-nvfp4", "torch-eager", "cozy-svdq-nvfp4-mma"
#: The rows the prebuilt kernels' operand is padded to, and the columns their weight tiles span.
NATIVE_ROWS, NATIVE_COLS = 256, 128
#: The cards the prebuilt kernels serve (`native`): every call reads it.
_SERVES: set[int] = set()
#: The probe's budget for the down-projection and the output, against the law's exact products:
#: their bf16 or fp16 rounding.
BUDGET = 5e-3

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


def _launcher(torch: ModuleType, device: Device, capturing: bool) -> Launch | None:
    """This GPU's launcher once the start's build loads and passes `probe` here, on a card whose
    row has FP4 tensor cores (`kernel_ready.serve`); else None: the caller then runs `eager`."""
    index = device.index or 0
    if TRITON_ABSENT or not kernel_ready.TRITON:
        return None
    if not arch.row(int8_linear.capability(torch, index)).fp4:
        return None
    opener = partial(_opened, torch, device)
    return kernel_ready.serve(f"{KERNELS.name} cuda:{index}", opener, capturing)


def _opened(torch: ModuleType, device: Device, wait: bool) -> Launch | None:
    try:
        return kernel_ready.TRITON[0](KERNELS, partial(probe, torch, device), wait, device=device)
    except Exception as exc:  # it never serves here: the law does, and the run says so
        int8_linear.failed(TRITON, device.index or 0, exc)
        raise


def probe(torch: ModuleType, device: Device, launch: Launch) -> verify.Verdict:
    """The Triton build's verdict here (`checked`), on ragged rows and columns."""

    def quantized(x: Tensor, roles: Mapping[str, Tensor]) -> tuple[object, Tensor, Tensor, Tensor]:
        q, sx, low, g = _quantized(torch, launch, x, roles)
        return (q, sx, low, g), q, sx, low

    def multiplied(
        made: object, roles: Mapping[str, Tensor], bias: Tensor | None, half: object
    ) -> Tensor:
        q, sx, low, g = cast("tuple[Tensor, Tensor, Tensor, Tensor]", made)
        return _multiplied(torch, launch, q, sx, low, g, roles, bias, half)

    return checked(torch, device, quantized, multiplied, (200, 144, 256))


def probe_native(torch: ModuleType, device: Device, module: object) -> verify.Verdict:
    """The prebuilt kernels' verdict here (`checked`): two row tiles, the second ragged, two
    column tiles and an odd number of k-steps."""

    def quantized(x: Tensor, roles: Mapping[str, Tensor]) -> tuple[object, Tensor, Tensor, Tensor]:
        made = _native_quantized(torch, x, roles)
        q, sx = operand(torch, made[0], made[1])
        rows = made[4]
        return made, q[:rows], sx[:rows].view(torch.float8_e4m3fn), made[2][:rows]

    def multiplied(
        made: object, roles: Mapping[str, Tensor], bias: Tensor | None, half: object
    ) -> Tensor:
        weight, scale = tiles(torch, roles["weight"], roles["scale"])
        tiled = {**roles, "weight": weight, "scale": scale}
        operands = cast("tuple[Tensor, Tensor, Tensor, Tensor, int]", made)
        return _native_multiplied(torch, operands, tiled, bias, half)

    return checked(torch, device, quantized, multiplied, (300, 256, 320))


def checked(
    torch: ModuleType,
    device: Device,
    quantized: Callable[[Tensor, Mapping[str, Tensor]], tuple[object, Tensor, Tensor, Tensor]],
    multiplied: Callable[[object, Mapping[str, Tensor], Tensor | None, object], Tensor],
    shape: tuple[int, int, int],
) -> verify.Verdict:
    """One build's two kernels against the law, each dtype and built rank, with and without `c`
    and the bias: the operand and its scales bit for bit; the down-projection and the output
    against the law's exact (f64) products, within their bf16 or fp16 rounding.
    `quantized(x, roles)` returns what `multiplied(it, roles, bias, half)` takes, then the codes
    `[M, K/2]`, E4M3 scales `[M, K/16]` and down-projection in the law's layout. `shape` is
    (M, N, K)."""
    generator = torch.Generator(device=device).manual_seed(0)
    rows, n, k = shape

    def draw(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32)

    errors: dict[str, float] = {}
    for r in RANKS:
        roles = {
            "weight": torch.randint(0, 256, (n, k // 2), generator=generator, device=device).to(
                torch.uint8
            ),
            "scale": (draw(n, k // NVFP4_BLOCK).abs() + 0.5).to(torch.float8_e4m3fn),
            "scale_2": torch.full((1,), 0.01, device=device, dtype=torch.float32),
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
            case = f"rank {r} {'bf16' if extras else 'fp16'}"
            made, q, sx, low = quantized(x, roles)
            g = activation_scale(torch, x.reshape(-1, k), roles)
            codes, scales = quantize(torch, _smoothed(torch, x, roles), g)
            exact = torch.equal(q, codes[:, 0::2] | (codes[:, 1::2] << 4)) and torch.equal(
                sx.view(torch.uint8), scales.view(torch.uint8)
            )
            errors[f"{case} quantize (bitwise)"] = 0.0 if exact else math.inf
            down = x.double() @ roles["lowrank_down"].to(half).double().t()
            errors[f"{case} down-projection"] = verify.error(torch, low, down)
            packed = roles["weight"]
            nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).view(n, k)
            weight = _values(torch, nibbles, roles["scale"], roles["scale_2"]).double()
            want = (_values(torch, codes, scales, g).double() @ weight.t()) * (
                roles["channel_scale"].double() if extras else 1.0
            )
            want = want + low.double() @ roles["lowrank_up"].double().t()
            if bias is not None:
                want = want + bias.double()
            got = multiplied(made, roles, bias, half)
            errors[f"{case} gemm"] = verify.error(torch, got, want)
    return verify.within(errors, BUDGET)


def serving(index: int) -> str:
    """The kernel this card's svdq leaves serve now: a run's `gemm.selected`."""
    if kernel_ready.ready(f"{NATIVE} cuda:{index}") is not None:
        return NATIVE
    return TRITON if kernel_ready.ready(f"{KERNELS.name} cuda:{index}") is not None else EAGER


def tiled(torch: ModuleType, index: int, n: int) -> bool:
    """Whether a leaf of `n` output columns on cuda:`index` holds its weight in the prebuilt
    GEMM's tiles: the card's row is block-scaled FP4 mma.sync (`arch`), the wheel carries the
    kernels and `n` is whole column tiles. Decided at the bind, by rule, whatever the verdict."""
    if _kernels.ABSENT or n % NATIVE_COLS:
        return False
    row = arch.row(int8_linear.capability(torch, index))
    return row.family == arch.MMA_SYNC and row.fp4


def native(torch: ModuleType, index: int) -> bool:
    """Whether the prebuilt kernels serve cuda:`index` now: once they pass `probe_native` here,
    off any request thread (`kernel_ready.serve`); a leaf's bind starts that beside its fill.
    Raises `Building` while a graph records before they serve."""
    if index in _SERVES:
        return True
    opener = partial(_native_opened, torch, index)
    capturing = torch.cuda.is_current_stream_capturing()
    if kernel_ready.serve(f"{NATIVE} cuda:{index}", opener, capturing) is None:
        return False
    _SERVES.add(index)  # it serves this card from here on
    return True


def _native_opened(torch: ModuleType, index: int, wait: bool) -> object:
    sm = int8_linear.capability(torch, index)
    probe = partial(probe_native, torch, torch.device("cuda", index))
    try:
        return kernel_ready.verified(_kernels._gemm, NATIVE, prebuilt.artifact(), sm, probe)
    except Exception as exc:  # they never serve here: Triton or the law does, and the run says so
        int8_linear.failed(NATIVE, index, exc)
        raise


def tiles(torch: ModuleType, weight: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    """The weight's codes `[N, K/2]` (u8, element 2k low) and E4M3 scales `[N, K/16]` as the
    prebuilt GEMM reads them (Nunchaku's fragment layouts): same shapes and bytes, reordered per
    128 rows and 64 columns into eight 16-row MMA tiles of 32 lanes' 16 bytes, then 32 lanes'
    four scale words. N % 128 == 0 and K % 64 == 0."""
    n, k = int(weight.shape[0]), 2 * int(weight.shape[1])
    words = weight.contiguous().view(torch.int32).view(n // 128, 8, 2, 8, k // 64, 2, 4)
    words = words.permute(0, 4, 1, 3, 6, 2, 5).contiguous()  # (n/128, k/64, n16, g, t, n8, k32)
    scales = scale.contiguous().view(torch.uint8).view(torch.int32)
    scales = scales.view(n // 128, 4, 4, 8, k // 64).permute(0, 4, 3, 2, 1).contiguous()
    return words.view(torch.uint8).view(n, k // 2), scales.view(torch.float8_e4m3fn).view(n, -1)


def untiled(torch: ModuleType, roles: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """`roles` with the weight and its scales back in their stored layout (`tiles` undone)."""
    n, k = int(roles["weight"].shape[0]), 2 * int(roles["weight"].shape[1])
    words = roles["weight"].view(torch.int32).view(n // 128, k // 64, 8, 8, 4, 2, 2)
    words = words.permute(0, 2, 5, 3, 1, 6, 4).contiguous()
    scales = roles["scale"].view(torch.uint8).view(torch.int32).view(n // 128, k // 64, 8, 4, 4)
    scales = scales.permute(0, 4, 3, 2, 1).contiguous()
    weight = words.view(torch.uint8).view(n, k // 2)
    return {**roles, "weight": weight, "scale": scales.view(torch.float8_e4m3fn).view(n, -1)}


def operand(torch: ModuleType, act: Tensor, ascales: Tensor) -> tuple[Tensor, Tensor]:
    """The prebuilt quantize's operand `[R, K/2]` and scales `[R, K/16]` (R a multiple of
    `NATIVE_ROWS`) in the law's layout: codes with element 2k low, E4M3 bytes."""
    rows, k = int(act.shape[0]), 2 * int(act.shape[1])
    words = act.view(torch.int32).view(rows // 256, k // 64, 8, 2, 8, 4, 2, 2)
    words = words.permute(0, 2, 3, 7, 4, 1, 6, 5).contiguous()  # rows then k-chunks, natural
    scales = ascales.view(torch.int32).view(rows // 256, k // 64, 8, 8, 2, 2)
    scales = scales.permute(0, 2, 4, 5, 3, 1).contiguous()
    return words.view(torch.uint8).view(rows, k // 2), scales.view(torch.uint8).view(rows, k // 16)


def _aligned(t: Tensor) -> Tensor:
    """`t` contiguous at a 16-byte address: the prebuilt kernels load 16 bytes at a time."""
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def _native_quantized(
    torch: ModuleType, x: Tensor, roles: Mapping[str, Tensor]
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    """`svdq_quantize`: `x`'s operand in the GEMM's tiles, its scales and its down-projection
    (rows padded to `NATIVE_ROWS`), its `g_x`, and its row count."""
    cols, r = int(roles["pre_quant_scale"].numel()), int(roles["lowrank_down"].shape[0])
    flat = _aligned(x.reshape(-1, cols))
    rows = int(flat.shape[0])
    padded_rows = -(-rows // NATIVE_ROWS) * NATIVE_ROWS
    g = activation_scale(torch, flat, roles).contiguous()
    act = torch.empty(padded_rows, cols // 2, dtype=torch.uint8, device=x.device)
    ascales = torch.empty(padded_rows, cols // NVFP4_BLOCK, dtype=torch.uint8, device=x.device)
    low = torch.empty(padded_rows, r, dtype=torch.bfloat16, device=x.device)
    index = x.device.index or 0
    code = _kernels._gemm.svdq_quantize(
        index, flat.data_ptr(), flat.dtype == torch.float16, roles["pre_quant_scale"].data_ptr(),
        g.data_ptr(), roles["lowrank_down"].data_ptr(), act.data_ptr(), ascales.data_ptr(),
        low.data_ptr(), rows, cols, r, int(torch.cuda.current_stream(index).cuda_stream),
    )  # fmt: skip
    _check(torch, code, "quantize")
    return act, ascales, low, g, rows


def _native_multiplied(
    torch: ModuleType,
    quantized: tuple[Tensor, Tensor, Tensor, Tensor, int],
    roles: Mapping[str, Tensor],
    bias: Tensor | None,
    half: object,
) -> Tensor:
    """`svdq_gemm`: `[M, N]` in `half`, from `_native_quantized`'s operands and a tiled weight."""
    act, ascales, low, g, rows = quantized
    n, r = int(roles["lowrank_up"].shape[0]), int(roles["lowrank_up"].shape[1])
    c = roles.get("channel_scale")
    out = torch.empty(rows, n, dtype=half, device=act.device)
    if rows == 0:
        return out
    bias = None if bias is None else _aligned(bias.to(half))
    index = act.device.index or 0
    code = _kernels._gemm.svdq_gemm(
        index, act.data_ptr(), ascales.data_ptr(), roles["weight"].data_ptr(),
        roles["scale"].data_ptr(), g.data_ptr(), roles["scale_2"].data_ptr(),
        0 if c is None else c.data_ptr(), low.data_ptr(), roles["lowrank_up"].data_ptr(),
        0 if bias is None else bias.data_ptr(), out.data_ptr(), half == torch.float16, rows, n,
        2 * int(act.shape[1]), r, int(torch.cuda.current_stream(index).cuda_stream),
    )  # fmt: skip
    _check(torch, code, "GEMM")
    return out


def _check(torch: ModuleType, code: int, what: str) -> None:
    if code == 0:
        return
    detail = f"svdq {what}: {_kernels._gemm.error_name(code)}"
    if code == _kernels._gemm.OUT_OF_MEMORY:
        raise torch.cuda.OutOfMemoryError(detail)
    raise RuntimeError(detail)


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
    tiled: bool = False,
) -> Tensor:
    """`x @ Wᵀ + bias` for one leaf's roles (`SvdqNvfp4NativeLeaf`): `weight` u8 `[N, K/2]`,
    `scale` E4M3 `[N, K/16]`, `scale_2` f32, `pre_quant_scale` f32 `[K]`, `lowrank_down` bf16
    `[R, K]` and `lowrank_up` bf16 `[N, R]` (R one of `RANKS`), optional `channel_scale` `[N]` and
    `input_scale`. The kernels serve a half `x` written in its own dtype; the law the rest. A
    `tiled` weight (`tiles`) is the prebuilt kernels' once they serve the card."""
    half = x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and out_dtype == x.dtype
    if tiled:
        index = x.device.index or 0
        if half and native(torch, index):
            try:
                made = _native_quantized(torch, x, roles)
                return _native_multiplied(torch, made, roles, bias, x.dtype).view(*x.shape[:-1], -1)
            except Exception as exc:  # Triton or the law serves this call, and the run says so
                int8_linear.failed(NATIVE, index, exc)
        roles = untiled(torch, roles)
    launch = None
    if half:
        launch = _launcher(torch, x.device, torch.cuda.is_current_stream_capturing())
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
