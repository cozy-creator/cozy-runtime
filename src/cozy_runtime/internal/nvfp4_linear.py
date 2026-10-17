"""The `nvfp4-w4a4/1` encoded linear: one leaf from sm80 up, three ways to multiply.

* **W4A4, Blackwell (sm100+), more than `DECODE_ROWS` rows.** comfy-kitchen's NVFP4 quantizer and
  cuBLASLt FP4 GEMM, the pair ComfyUI serves BFL's nvfp4 checkpoints with. Each activation is
  scaled by its own tensor amax (`amax / (448 * 6)`, the producer's rule for the weight's second
  level) and quantized per 16-block; the GEMM applies both second-levels as its alpha.
* **W4A16, everything else on CUDA.** At decode-sized M a split-K Triton GEMV decodes the stored
  nibbles and E4M3 block scales in registers onto the bf16/fp16 tensor cores, reading 0.56 bytes
  per weight where bf16 reads 2, which is what an LM's token loop is bound by. Larger M decodes the
  weight once into a transient and takes the bf16 GEMM.
* **Eager decode** where neither exists (CPU, no Triton, fp16, the executor start's build still
  running; the kitchen build still running at large M on Blackwell takes W4A16 instead), so no
  run waits for a compile. The Triton kernels build when the executor starts (`KERNELS`). A W4A4
  call that raises is served by W4A16, and the run says so (`int8_linear.failed`).

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING

from cozy_runtime.author.kernels import (
    KernelsUnavailable,
    Launch,
    TritonEntry,
    TritonKernels,
    readable,
)
from cozy_runtime.internal import int8_linear, triton_kernels
from cozy_runtime.internal.encoding.formats import NVFP4_BLOCK, Tensor, nvfp4_values

if TYPE_CHECKING:
    from torch import device as Device

#: The largest magnitude of the two levels: E4M3's 448 times E2M1's 6.
LEVELS_MAX = 448.0 * 6.0
#: An all-zero activation has amax 0; its scale must stay a real number.
SCALE_FLOOR = 1e-30
#: Up to this many rows a product is bandwidth-bound: W4A16 reads the fewest bytes on any card.
DECODE_ROWS = 16
#: The GEMV's tile: 64 output rows by 32 int32 words (256 weights) of each.
GEMV_N, GEMV_W = 64, 32
#: The kernels a leaf serves besides comfy-kitchen's W4A4 (`gemm.selected`).
DECODE, EAGER = "cozy-nvfp4-decode", "torch-decode"

_MAJOR: dict[int, int] = {}


try:
    import triton
    import triton.language as tl
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    TRITON_ABSENT = f"triton does not import ({_absent})"
else:
    TRITON_ABSENT = ""

if not TRITON_ABSENT:

    @triton.jit  # type: ignore[untyped-decorator]
    def _e2m1(code):  # type: ignore[no-untyped-def]
        """E2M1 nibbles -> f32, built as bits: 1.m x 2^(e-1), the e=0 code being 0 or 0.5."""
        exponent = (code >> 1) & 3
        mantissa = code & 1
        bits = tl.where(
            exponent == 0, mantissa * 0x3F000000, ((exponent + 126) << 23) | (mantissa << 22)
        )
        return (bits | ((code & 8) << 28)).to(tl.float32, bitcast=True)

    @triton.jit  # type: ignore[untyped-decorator]
    def _scales(s_ptr, rn, k0, N, K, col_tiles, BLOCK_K: tl.constexpr):  # type: ignore[no-untyped-def]
        """`[BLOCK_N, BLOCK_K/16]` f32 block scales; they sit in cuBLASLt's 128x4 tiles."""
        c = k0 // 16 + tl.arange(0, BLOCK_K // 16)
        tile = (rn[:, None] // 128) * col_tiles + c[None, :] // 4
        offset = (
            (tile * 32 + rn[:, None] % 32) * 16 + (rn[:, None] % 128) // 32 * 4 + c[None, :] % 4
        )
        ok = (rn[:, None] < N) & (c[None, :] < K // 16)
        byte = tl.load(s_ptr + offset, mask=ok, other=0).to(tl.int32)
        exponent = (byte >> 3) & 0xF
        mantissa = byte & 7
        normal = (((exponent + 120) << 23) | (mantissa << 20)).to(tl.float32, bitcast=True)
        return tl.where(exponent == 0, mantissa.to(tl.float32) * 0.001953125, normal)

    @triton.jit  # type: ignore[untyped-decorator]
    def _packed(w_ptr, rn, k0, N, K, BLOCK_K: tl.constexpr):  # type: ignore[no-untyped-def]
        """`[BLOCK_N, BLOCK_K/2]` packed bytes; element 2k is the LOW nibble (ModelOpt's)."""
        rj = tl.arange(0, BLOCK_K // 2)
        return tl.load(
            w_ptr + rn[:, None].to(tl.int64) * (K // 2) + (k0 // 2 + rj)[None, :],
            mask=(rn[:, None] < N) & ((k0 // 2 + rj)[None, :] < K // 2),
            other=0,
        ).to(tl.int32)

    @triton.jit  # type: ignore[untyped-decorator]
    def _decode(  # type: ignore[no-untyped-def]
        w_ptr, s_ptr, rn, k0, N, K, col_tiles, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """`[BLOCK_N, BLOCK_K]` f32 weights at (rn, k0), the second level not yet applied."""
        packed = _packed(w_ptr, rn, k0, N, K, BLOCK_K)
        value = tl.join(_e2m1(packed & 0xF), _e2m1(packed >> 4)).reshape(BLOCK_N, BLOCK_K)
        scale = _scales(s_ptr, rn, k0, N, K, col_tiles, BLOCK_K)
        return (value.reshape(BLOCK_N, BLOCK_K // 16, 16) * scale[:, :, None]).reshape(
            BLOCK_N, BLOCK_K
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def _nibble(words, i: tl.constexpr):  # type: ignore[no-untyped-def]
        """Element 8w + i of each int32 word as f32, 2^-14 of its value: the E2M1 code's sign,
        exponent and mantissa moved into f16's fields (the caller scales back by 2^14)."""
        code = (words >> (4 * i)) & 0xF
        bits = ((code & 7) << 9) | ((code & 8) << 12)
        return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)

    @triton.jit  # type: ignore[untyped-decorator]
    def _weights(words, i: tl.constexpr, scale):  # type: ignore[no-untyped-def]
        """Nibble i of each word, scaled, in bf16: E2M1 times E4M3 is exact there."""
        return (_nibble(words, i) * scale).to(tl.bfloat16)

    @triton.jit  # type: ignore[untyped-decorator]
    def _gemv(  # type: ignore[no-untyped-def]
        x_ptr,
        w_ptr,
        s_ptr,
        two_ptr,
        out_ptr,
        M,
        N,
        K,
        col_tiles,
        span,
        BLOCK_N: tl.constexpr,
        BLOCK_W: tl.constexpr,
        SPLIT: tl.constexpr,
    ):
        """Up to 16 rows (a decoded token, or a step's drafted ones) on the tensor cores: each
        int32 word of 8 weights is decoded once for every row, so the weight is read once, and a
        row's sums never depend on the rows beside it, so a drafted token keeps decode's bits
        (a one-row FMA kernel is 7% faster on a 4090, but its sums are not these). SPLIT writes
        f32 partials per K span."""
        rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.arange(0, 16)
        rw = tl.arange(0, BLOCK_W)
        rk = tl.arange(0, 8 * BLOCK_W)
        start = tl.program_id(1) * span
        words_per_row = K // 8
        acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for w0 in range(start // 8, tl.minimum(start + span, K) // 8, BLOCK_W):
            cols = w0 + rw
            words = tl.load(
                w_ptr + rn[:, None].to(tl.int64) * words_per_row + cols[None, :],
                mask=(rn[:, None] < N) & (cols[None, :] < words_per_row),
                other=0,
            )
            # The nibbles' 2^-14 goes back with the scale; two words share a 16-block's.
            scale = _scales(s_ptr, rn, 8 * w0, N, K, col_tiles, 8 * BLOCK_W) * 16384.0
            scale = tl.reshape(
                tl.broadcast_to(scale[:, :, None], (BLOCK_N, BLOCK_W // 2, 2)), (BLOCK_N, BLOCK_W)
            )
            # Nibble i of word w is element 8w + i: joined innermost, the tile is in K order.
            even = tl.join(
                tl.join(_weights(words, 0, scale), _weights(words, 4, scale)),
                tl.join(_weights(words, 2, scale), _weights(words, 6, scale)),
            )
            odd = tl.join(
                tl.join(_weights(words, 1, scale), _weights(words, 5, scale)),
                tl.join(_weights(words, 3, scale), _weights(words, 7, scale)),
            )
            weight = tl.reshape(tl.join(even, odd), (BLOCK_N, 8 * BLOCK_W))
            k = 8 * w0 + rk
            x = tl.load(
                x_ptr + rm[:, None] * K + k[None, :],
                mask=(rm[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(x, tl.trans(weight))
        where = rm[:, None] * N + rn[None, :]
        ok = (rm[:, None] < M) & (rn[None, :] < N)
        if SPLIT:
            tl.store(out_ptr + tl.program_id(1) * M * N + where, acc, mask=ok)
        else:
            acc *= tl.load(two_ptr)
            tl.store(out_ptr + where, acc.to(out_ptr.dtype.element_ty), mask=ok)

    @triton.jit  # type: ignore[untyped-decorator]
    def _gather(part_ptr, two_ptr, out_ptr, N, SPLIT: tl.constexpr, BLOCK: tl.constexpr):  # type: ignore[no-untyped-def]
        """`_gemv`'s K-span partials summed, scaled by the second level, cast."""
        rn = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for split in range(SPLIT):
            acc += tl.load(part_ptr + split * N + rn, mask=rn < N, other=0.0)
        acc *= tl.load(two_ptr)
        tl.store(out_ptr + rn, acc.to(out_ptr.dtype.element_ty), mask=rn < N)

    @triton.jit  # type: ignore[untyped-decorator]
    def _dequant(  # type: ignore[no-untyped-def]
        w_ptr,
        s_ptr,
        two_ptr,
        out_ptr,
        N,
        K,
        col_tiles,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        k0 = tl.program_id(1) * BLOCK_K
        rk = tl.arange(0, BLOCK_K)
        weight = _decode(w_ptr, s_ptr, rn, k0, N, K, col_tiles, BLOCK_N, BLOCK_K) * tl.load(two_ptr)
        tl.store(
            out_ptr + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :],
            weight.to(out_ptr.dtype.element_ty),
            mask=(rn[:, None] < N) & ((k0 + rk)[None, :] < K),
        )


def _entries() -> list[TritonEntry]:
    """The specializations `matmul` launches: bf16 activations, the tiles above."""
    if TRITON_ABSENT:
        return []
    weight = {"w_ptr": "*u8", "s_ptr": "*u8", "two_ptr": "*fp32", "x_ptr": "*bf16"}

    def signature(kernel: triton.JITFunction, **types: str) -> dict[str, str]:
        # Constexprs by the kernel's own annotations: N and K are runtime sizes.
        return {
            p.name: types.get(p.name)
            or weight.get(p.name)
            or ("constexpr" if p.is_constexpr else "*bf16" if p.name.endswith("_ptr") else "i32")
            for p in kernel.params
        }

    entries = [
        TritonEntry(
            _gemv,
            signature(_gemv, w_ptr="*i32", out_ptr="*fp32" if split else "*bf16"),
            {"BLOCK_N": GEMV_N, "BLOCK_W": GEMV_W, "SPLIT": split},
            aligned=("x_ptr", "w_ptr", "out_ptr", "K", "span"),
            num_warps=8,
        )
        for split in (0, 1)
    ]
    entries += [
        TritonEntry(
            _gather,
            signature(_gather, part_ptr="*fp32"),
            {"SPLIT": split, "BLOCK": 1024},
            aligned=("part_ptr", "out_ptr"),
            num_warps=4,
        )
        for split in (2, 4)
    ]
    entries.append(
        TritonEntry(_dequant, signature(_dequant), {"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4)
    )
    return entries


#: Built on every executor's GPUs when it starts (`author.kernels`): a request never compiles.
KERNELS = TritonKernels("cozy-nvfp4", _entries())
#: How often a call asks whether the start's build has landed: resolution, never a verdict.
POLL_SECONDS = 1.0
_LAUNCH: dict[int, tuple[float, Launch | None]] = {}


def _launcher(device: Device) -> Launch | None:
    """This GPU's launcher once the executor start's build is loaded, else None (asked again
    after `POLL_SECONDS`): the caller then decodes eagerly."""
    index = device.index or 0
    asked, launch = _LAUNCH.get(index, (float("-inf"), None))
    now = time.monotonic()
    if launch is None and now - asked >= POLL_SECONDS:
        try:
            launch = triton_kernels.ready(KERNELS, device=device)
        except KernelsUnavailable:
            launch = None
        _LAUNCH[index] = (now, launch)
    return launch


def _major(torch: ModuleType, index: int) -> int:
    if index not in _MAJOR:
        _MAJOR[index] = int(torch.cuda.get_device_capability(index)[0])
    return _MAJOR[index]


def matmul(
    torch: ModuleType,
    x: Tensor,
    weight: Tensor,
    blocked_scale: Tensor,
    linear_scale: Callable[[], Tensor],
    scale_2: Tensor,
    out_dtype: Tensor,
) -> Tensor:
    """`x @ W.T` for one stored `nvfp4-w4a4/1` weight (`weight` u8 `[N, K/2]`, `blocked_scale` its
    E4M3 block scales in cuBLASLt's 128x4 layout, `scale_2` the f32 `[1]` second level).

    `linear_scale` returns the `[N, K/16]` row-major scales; only the eager path asks for it."""
    rows_out, cols = int(weight.shape[0]), 2 * int(weight.shape[1])
    flat = x.reshape(-1, cols).contiguous()
    rows = int(flat.shape[0])
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
        index = x.device.index or 0
        module = (
            int8_linear.kitchen(torch, index)
            if rows > DECODE_ROWS and _major(torch, index) >= 10
            else None
        )
        if module is not None:
            amax = flat.abs().amax().float()
            tensor_scale = (amax / LEVELS_MAX).clamp(min=SCALE_FLOOR).reshape(1)
            try:
                with readable(weight, blocked_scale, scale_2) as (w, scales, second):
                    # hi_first=False: element 2k in the LOW nibble, the stored weight's order.
                    # Kitchen's default packs the other way; a K-wise dot only needs both agree.
                    payload, block = module.quantize_nvfp4(
                        flat, tensor_scale, 0.0, rows % NVFP4_BLOCK != 0, False
                    )
                    out = module.scaled_mm_nvfp4(
                        payload, w, tensor_scale, second, block, scales, out_dtype=out_dtype
                    )
                return out[:rows].reshape(*x.shape[:-1], rows_out)
            except Exception as exc:  # W4A16 serves this call
                int8_linear.failed("comfy-kitchen nvfp4", index, exc)
        launch = (
            _launcher(x.device)
            if x.dtype == torch.bfloat16 and out_dtype == torch.bfloat16
            else None
        )
        if launch is not None:
            with readable(weight, blocked_scale, scale_2) as (weight, blocked_scale, scale_2):
                scales, tiles = blocked_scale.view(torch.uint8), int(blocked_scale.shape[1]) // 4
                if rows <= DECODE_ROWS and weight.data_ptr() % 16 == 0:
                    # Bandwidth-bound: split over K while a few programs per row block would
                    # leave SMs idle (a 3090: 213-335 GB/s for a program per 32 rows, 520-795 so).
                    flat = flat if flat.data_ptr() % 16 == 0 else flat.clone()
                    split = 4 if rows_out <= 8192 else 2 if rows_out <= 16384 else 1
                    span = -(-cols // (split * 8 * GEMV_W)) * 8 * GEMV_W
                    out = torch.empty(rows, rows_out, dtype=out_dtype, device=x.device)
                    part = (
                        torch.empty(split, rows, rows_out, dtype=torch.float32, device=x.device)
                        if split > 1
                        else out
                    )
                    launch(
                        _gemv,
                        (-(-rows_out // GEMV_N), split, 1),
                        flat,
                        weight.view(torch.int32),
                        scales,
                        scale_2,
                        part,
                        rows,
                        rows_out,
                        cols,
                        tiles,
                        span,
                        GEMV_N,
                        GEMV_W,
                        int(split > 1),
                    )
                    if split > 1:
                        launch(
                            _gather,
                            (-(-(rows * rows_out) // 1024), 1, 1),
                            part,
                            scale_2,
                            out,
                            rows * rows_out,
                            split,
                            1024,
                        )
                    return out.reshape(*x.shape[:-1], rows_out)
                decoded = torch.empty(rows_out, cols, dtype=x.dtype, device=x.device)
                launch(
                    _dequant,
                    (-(-rows_out // 64), -(-cols // 128), 1),
                    weight,
                    scales,
                    scale_2,
                    decoded,
                    rows_out,
                    cols,
                    tiles,
                    64,
                    128,
                )
            out = torch.nn.functional.linear(flat, decoded).to(out_dtype)
            return out.reshape(*x.shape[:-1], rows_out)
    decoded = nvfp4_values(torch, weight, linear_scale(), scale_2).to(x.dtype)
    return torch.nn.functional.linear(x, decoded).to(out_dtype)


def serving(index: int) -> str:
    """The kernel this card's nvfp4 leaves serve a product of more than `DECODE_ROWS` rows with
    now (fewer rows take the GEMV): a run's `gemm.selected`, read once a run."""
    kitchen = int8_linear._KITCHEN.get(index)
    if _MAJOR.get(index, 0) >= 10 and kitchen is not None and kitchen.served is not None:
        return int8_linear.ARTIFACT
    return DECODE if _LAUNCH.get(index, (0.0, None))[1] is not None else EAGER
