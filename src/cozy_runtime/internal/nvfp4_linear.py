"""The `nvfp4-w4a4/1` encoded linear: one leaf from sm80 up, three ways to multiply.

* **W4A4, Blackwell (sm100+), more than `DECODE_ROWS` rows.** comfy-kitchen's NVFP4 quantizer and
  cuBLASLt FP4 GEMM, the pair ComfyUI serves BFL's nvfp4 checkpoints with. Each activation is
  scaled by its own tensor amax (`amax / (448 * 6)`, the producer's rule for the weight's second
  level) and quantized per 16-block; the GEMM applies both second-levels as its alpha.
* **W4A16, everything else on CUDA.** At decode-sized M a split-K Triton GEMV decodes the stored
  nibbles and E4M3 block scales in registers onto the bf16/fp16 tensor cores, reading 0.56 bytes
  per weight where bf16 reads 2, which is what an LM's token loop is bound by. Larger M decodes the
  weight once into a transient and takes the bf16 GEMM.
* **Eager decode** where neither exists (CPU, no Triton, the kitchen build still running at large
  M on Blackwell takes W4A16 instead), so no run waits for a compile.

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

from cozy_runtime.author.kernels import readable
from cozy_runtime.internal import int8_linear
from cozy_runtime.internal.encoding.formats import NVFP4_BLOCK, Tensor, nvfp4_values

#: The largest magnitude of the two levels: E4M3's 448 times E2M1's 6.
LEVELS_MAX = 448.0 * 6.0
#: An all-zero activation has amax 0; its scale must stay a real number.
SCALE_FLOOR = 1e-30
#: Up to this many rows a product is bandwidth-bound: W4A16 reads the fewest bytes on any card.
DECODE_ROWS = 16
#: The GEMV's tile: 64 output rows by 256 input columns (128 bytes of each packed row).
GEMV_N, GEMV_K = 64, 256

_MAJOR: dict[int, int] = {}


def _kernels() -> tuple[object, object, object] | None:
    """The Triton W4A16 kernels, compiled once, lazily: split-K GEMVs for decode-sized M (one
    row, and up to 16 on the tensor cores) and a dequantizer for the rest. None without Triton."""
    cached = getattr(_kernels, "_cached", "unset")
    if cached != "unset":
        return cached  # type: ignore[return-value]
    try:
        import triton
        import triton.language as tl

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
        def _gemv(  # type: ignore[no-untyped-def]
            x_ptr,
            w_ptr,
            s_ptr,
            two_ptr,
            out_ptr,
            M,
            N,
            K,
            stride_x,
            col_tiles,
            span,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            SPLIT: tl.constexpr,
        ):
            rm = tl.arange(0, BLOCK_M)
            rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
            rk = tl.arange(0, BLOCK_K)
            start = tl.program_id(1) * span
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(start, tl.minimum(start + span, K), BLOCK_K):
                x = tl.load(
                    x_ptr + rm[:, None] * stride_x + (k0 + rk)[None, :],
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K),
                    other=0.0,
                )
                weight = _decode(w_ptr, s_ptr, rn, k0, N, K, col_tiles, BLOCK_N, BLOCK_K)
                acc += tl.dot(x, tl.trans(weight.to(x.dtype)))
            acc *= tl.load(two_ptr)
            where = out_ptr + rm[:, None] * N + rn[None, :]
            ok = (rm[:, None] < M) & (rn[None, :] < N)
            if SPLIT:
                tl.atomic_add(where, acc, mask=ok)
            else:
                tl.store(where, acc.to(out_ptr.dtype.element_ty), mask=ok)

        @triton.jit  # type: ignore[untyped-decorator]
        def _gemv1(  # type: ignore[no-untyped-def]
            x_ptr,
            w_ptr,
            s_ptr,
            two_ptr,
            out_ptr,
            N,
            K,
            col_tiles,
            span,
            BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr,
            SPLIT: tl.constexpr,
        ):
            """One row: even and odd columns against the low and high nibbles, reduced per 16-block,
            then scaled once per block. No interleave and no tensor-core layout round trip."""
            rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
            rj = tl.arange(0, BLOCK_K // 2)
            start = tl.program_id(1) * span
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k0 in range(start, tl.minimum(start + span, K), BLOCK_K):
                even = k0 + 2 * rj
                x_even = tl.load(x_ptr + even, mask=even < K, other=0.0).to(tl.float32)
                x_odd = tl.load(x_ptr + even + 1, mask=even + 1 < K, other=0.0).to(tl.float32)
                packed = _packed(w_ptr, rn, k0, N, K, BLOCK_K)
                part = _e2m1(packed & 0xF) * x_even[None, :] + _e2m1(packed >> 4) * x_odd[None, :]
                blocks = tl.sum(part.reshape(BLOCK_N, BLOCK_K // 16, 8), axis=2)
                acc += tl.sum(blocks * _scales(s_ptr, rn, k0, N, K, col_tiles, BLOCK_K), axis=1)
            acc *= tl.load(two_ptr)
            if SPLIT:
                tl.atomic_add(out_ptr + rn, acc, mask=rn < N)
            else:
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
            weight = _decode(w_ptr, s_ptr, rn, k0, N, K, col_tiles, BLOCK_N, BLOCK_K) * tl.load(
                two_ptr
            )
            tl.store(
                out_ptr + rn[:, None].to(tl.int64) * K + (k0 + rk)[None, :],
                weight.to(out_ptr.dtype.element_ty),
                mask=(rn[:, None] < N) & ((k0 + rk)[None, :] < K),
            )

        kernels: tuple[object, object, object] | None = (_gemv, _gemv1, _dequant)
    except Exception:
        kernels = None
    _kernels._cached = kernels  # type: ignore[attr-defined]
    return kernels


_SMS: dict[int, int] = {}


def _sms(torch: ModuleType, index: int) -> int:
    if index not in _SMS:
        _SMS[index] = int(torch.cuda.get_device_properties(index).multi_processor_count)
    return _SMS[index]


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
            with readable(weight, blocked_scale, scale_2) as (weight, blocked_scale, scale_2):
                # hi_first=False: element 2k in the LOW nibble, the stored weight's order.
                # Kitchen's default packs the other way; a K-wise dot only needs both to agree.
                payload, block = module.quantize_nvfp4(
                    flat, tensor_scale, 0.0, rows % NVFP4_BLOCK != 0, False
                )
                out = module.scaled_mm_nvfp4(
                    payload,
                    weight,
                    tensor_scale,
                    scale_2,
                    block,
                    blocked_scale,
                    out_dtype=out_dtype,
                )
            return out[:rows].reshape(*x.shape[:-1], rows_out)
        kernels = _kernels()
        if kernels is not None:
            gemv, gemv1, dequant = kernels
            with readable(weight, blocked_scale, scale_2) as (weight, blocked_scale, scale_2):
                scales, tiles = blocked_scale.view(torch.uint8), int(blocked_scale.shape[1]) // 4
                if rows == 1:
                    # One decoded token: the launch, not the kernel, is the cost (a split's
                    # f32 scratch and cast are two more launches), so one program per 32 rows.
                    out = torch.empty(1, rows_out, dtype=out_dtype, device=x.device)
                    gemv1[(-(-rows_out // 32), 1)](  # type: ignore[index]
                        flat,
                        weight,
                        scales,
                        scale_2,
                        out,
                        rows_out,
                        cols,
                        tiles,
                        cols,
                        BLOCK_N=32,
                        BLOCK_K=GEMV_K,
                        SPLIT=False,
                        num_warps=8,
                    )
                    return out.reshape(*x.shape[:-1], rows_out)
                if rows <= DECODE_ROWS:
                    # Split K until the card has ~4 programs per SM: a decode GEMV is bandwidth.
                    columns = -(-rows_out // GEMV_N)
                    split = max(1, min(cols // GEMV_K, -(-4 * _sms(torch, index) // columns)))
                    span = -(-cols // (split * GEMV_K)) * GEMV_K
                    acc = (
                        torch.zeros(rows, rows_out, dtype=torch.float32, device=x.device)
                        if split > 1
                        else torch.empty(rows, rows_out, dtype=out_dtype, device=x.device)
                    )
                    grid = (columns, -(-cols // span))
                    gemv[grid](  # type: ignore[index]
                        flat,
                        weight,
                        scales,
                        scale_2,
                        acc,
                        rows,
                        rows_out,
                        cols,
                        flat.stride(0),
                        tiles,
                        span,
                        BLOCK_M=16,
                        BLOCK_N=GEMV_N,
                        BLOCK_K=GEMV_K,
                        SPLIT=split > 1,
                        num_warps=4,
                    )
                    return acc.to(out_dtype).reshape(*x.shape[:-1], rows_out)
                decoded = torch.empty(rows_out, cols, dtype=x.dtype, device=x.device)
                dequant[(-(-rows_out // 64), -(-cols // 128))](  # type: ignore[index]
                    weight,
                    scales,
                    scale_2,
                    decoded,
                    rows_out,
                    cols,
                    tiles,
                    BLOCK_N=64,
                    BLOCK_K=128,
                    num_warps=4,
                )
            out = torch.nn.functional.linear(flat, decoded).to(out_dtype)
            return out.reshape(*x.shape[:-1], rows_out)
    decoded = nvfp4_values(torch, weight, linear_scale(), scale_2).to(x.dtype)
    return torch.nn.functional.linear(x, decoded).to(out_dtype)
