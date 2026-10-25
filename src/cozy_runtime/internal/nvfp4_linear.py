"""The `nvfp4-w4a4/1` encoded linear: one leaf from sm80 up, three ways to multiply.

* **W4A4, a card whose row has FP4 tensor cores (`arch`), more than `DECODE_ROWS` rows.**
  comfy-kitchen's NVFP4 quantizer and cuBLASLt FP4 GEMM, the pair ComfyUI serves BFL's nvfp4
  checkpoints with. Each activation is scaled by its own tensor amax (`amax / (448 * 6)`, the
  producer's rule for the weight's second level) and quantized per 16-block; the GEMM applies
  both second-levels as its alpha.
* **W4A16, everything else on CUDA.** At decode-sized M a split-K Triton GEMV decodes the stored
  nibbles and E4M3 block scales in registers onto the bf16/fp16 tensor cores, reading 0.56 bytes
  per weight where bf16 reads 2, which is what an LM's token loop is bound by. Larger M decodes the
  weight once into a transient and takes the bf16 GEMM.
* **Eager decode** where neither exists (CPU, no Triton, fp16, the executor start's build still
  running; the kitchen build still running at large M on an FP4 card takes W4A16 instead), so no
  run waits for a compile. The Triton kernels build when the executor starts (`KERNELS`). Each
  path serves once its probe agrees with the exact law on the card (`kernel_ready.serve`). A
  W4A4 call that raises is served by W4A16, and the run says so (`int8_linear.failed`).

`torch` is the torch module: this module imports no torch.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial
from types import ModuleType
from typing import TYPE_CHECKING

from cozy_runtime.author.kernels import (
    Launch,
    TritonEntry,
    TritonKernels,
    readable,
)
from cozy_runtime.internal import arch, int8_linear, kernel_ready, prebuilt, svdq_linear, verify
from cozy_runtime.internal.encoding.formats import (
    NVFP4_BLOCK,
    Tensor,
    from_blocked,
    nvfp4_values,
    to_blocked,
)

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
#: comfy-kitchen's W4A4 by its `kernel_ready` name's head.
KITCHEN = "comfy-kitchen nvfp4 W4A4"


try:
    import triton
    import triton.language as tl
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    TRITON_ABSENT = f"triton does not import ({_absent})"
else:
    TRITON_ABSENT = ""

if not TRITON_ABSENT:

    @triton.jit
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
        # Constexprs by the kernel's own annotations: N and K are runtime sizes. Read off its
        # Python function, which a JITFunction and the interpreter's InterpretedFunction both
        # keep as `fn` (the interpreter's has no `params`).
        def constexpr(annotation: object) -> bool:
            return annotation is tl.constexpr or str(annotation).endswith("constexpr")

        return {
            name: types.get(name)
            or weight.get(name)
            or (
                "constexpr"
                if constexpr(p.annotation)
                else "*bf16"
                if name.endswith("_ptr")
                else "i32"
            )
            for name, p in inspect.signature(kernel.fn).parameters.items()
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


def _launcher(torch: ModuleType, device: Device, capturing: bool) -> Launch | None:
    """This GPU's launcher once the executor start's build loads and passes `probe` here
    (`kernel_ready.serve`), else None: the caller then decodes eagerly."""
    if TRITON_ABSENT or not kernel_ready.TRITON:
        return None
    opener = partial(kernel_ready.TRITON[0], KERNELS, partial(probe, torch, device), device=device)
    return kernel_ready.serve(f"{KERNELS.name} cuda:{device.index or 0}", opener, capturing)


def kitchen(torch: ModuleType, index: int, capturing: bool) -> ModuleType | None:
    """comfy-kitchen's backend once its W4A4 passes its probe on this card (`kernel_ready.serve`):
    asked only where the card's row has FP4 tensor cores."""
    name = f"{KITCHEN} cuda:{index}"
    return kernel_ready.serve(name, partial(_kitchen, torch, index), capturing)


def _kitchen(torch: ModuleType, index: int, wait: bool) -> ModuleType | None:
    found = int8_linear.kitchen_module(torch, index, wait)
    if found is None:
        return None
    module, artifact = found
    sm, check = int8_linear.capability(torch, index), partial(_probe_kitchen, torch, index)
    return kernel_ready.verified(module, "comfy-kitchen-nvfp4", artifact, sm, check)


def fp4(torch: ModuleType, index: int) -> bool:
    """Whether cuda:`index`'s row has FP4 tensor cores: only there does a W4A4 product serve."""
    return arch.row(int8_linear.capability(torch, index)).fp4


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
        capturing = torch.cuda.is_current_stream_capturing()
        module = (
            kitchen(torch, index, capturing) if rows > DECODE_ROWS and fp4(torch, index) else None
        )
        if module is not None:
            try:
                with readable(weight, blocked_scale, scale_2) as (w, scales, second):
                    payload, block, tensor_scale = _quantized(torch, module, flat)
                    out = module.scaled_mm_nvfp4(
                        payload, w, tensor_scale, second, block, scales, out_dtype=out_dtype
                    )
                return out[:rows].reshape(*x.shape[:-1], rows_out)
            except Exception as exc:  # W4A16 serves this call
                int8_linear.failed("comfy-kitchen nvfp4", index, exc)
        launch = (
            _launcher(torch, x.device, capturing)
            if x.dtype == torch.bfloat16 and out_dtype == torch.bfloat16
            else None
        )
        if launch is not None:
            with readable(weight, blocked_scale, scale_2) as (weight, blocked_scale, scale_2):
                if rows <= DECODE_ROWS and weight.data_ptr() % 16 == 0:
                    # Bandwidth-bound: split over K while a few programs per row block would
                    # leave SMs idle (a 3090: 213-335 GB/s for a program per 32 rows, 520-795 so).
                    flat = flat if flat.data_ptr() % 16 == 0 else flat.clone()
                    split = 4 if rows_out <= 8192 else 2 if rows_out <= 16384 else 1
                    out = _rows(torch, launch, flat, weight, blocked_scale, scale_2, split)
                    return out.reshape(*x.shape[:-1], rows_out)
                decoded = _decoded(torch, launch, weight, blocked_scale, scale_2)
            out = torch.nn.functional.linear(flat, decoded).to(out_dtype)
            return out.reshape(*x.shape[:-1], rows_out)
    decoded = nvfp4_values(torch, weight, linear_scale(), scale_2).to(x.dtype)
    return torch.nn.functional.linear(x, decoded).to(out_dtype)


def _quantized(torch: ModuleType, module: ModuleType, flat: Tensor) -> tuple[Tensor, ...]:
    """comfy-kitchen's NVFP4 operand of `flat` under its own tensor amax: packed codes, block
    scales in cuBLASLt's tiles, the second level."""
    amax = torch.linalg.vector_norm(flat, float("inf")).float()  # one pass, exact
    tensor_scale = (amax / LEVELS_MAX).clamp(min=SCALE_FLOOR).reshape(1)
    # hi_first=False: element 2k in the LOW nibble, the stored weight's order. Kitchen's default
    # packs the other way; a K-wise dot only needs both agree.
    rows = int(flat.shape[0])
    payload, block = module.quantize_nvfp4(flat, tensor_scale, 0.0, rows % NVFP4_BLOCK != 0, False)
    return payload, block, tensor_scale


def _rows(
    torch: ModuleType,
    launch: Launch,
    flat: Tensor,
    weight: Tensor,
    blocked_scale: Tensor,
    scale_2: Tensor,
    split: int,
) -> Tensor:
    """`flat @ W.T` (bf16, up to `DECODE_ROWS` rows) on the GEMV, K split over `split` spans."""
    rows, cols, rows_out = int(flat.shape[0]), int(flat.shape[1]), int(weight.shape[0])
    scales, tiles = blocked_scale.view(torch.uint8), int(blocked_scale.shape[1]) // 4
    span = -(-cols // (split * 8 * GEMV_W)) * 8 * GEMV_W
    out = torch.empty(rows, rows_out, dtype=torch.bfloat16, device=flat.device)
    part = (
        torch.empty(split, rows, rows_out, dtype=torch.float32, device=flat.device)
        if split > 1
        else out
    )
    launch(
        _gemv, (-(-rows_out // GEMV_N), split, 1),
        flat, weight.view(torch.int32), scales, scale_2, part, rows, rows_out, cols, tiles, span,
        GEMV_N, GEMV_W, int(split > 1),
    )  # fmt: skip
    if split > 1:
        launch(
            _gather, (-(-(rows * rows_out) // 1024), 1, 1),
            part, scale_2, out, rows * rows_out, split, 1024,
        )  # fmt: skip
    return out


def _decoded(
    torch: ModuleType, launch: Launch, weight: Tensor, blocked_scale: Tensor, scale_2: Tensor
) -> Tensor:
    """The stored weight decoded to bf16 `[N, K]` in one launch."""
    rows_out, cols = int(weight.shape[0]), 2 * int(weight.shape[1])
    scales, tiles = blocked_scale.view(torch.uint8), int(blocked_scale.shape[1]) // 4
    decoded = torch.empty(rows_out, cols, dtype=torch.bfloat16, device=weight.device)
    launch(
        _dequant, (-(-rows_out // 64), -(-cols // 128), 1),
        weight, scales, scale_2, decoded, rows_out, cols, tiles, 64, 128,
    )  # fmt: skip
    return decoded


def _probe_weight(
    torch: ModuleType, generator: object, device: object, rows_out: int, cols: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A random stored weight: packed codes, its E4M3 block scales row-major and in cuBLASLt's
    tiles, an f32 second level."""
    weight = torch.randint(0, 256, (rows_out, cols // 2), generator=generator, device=device,
                           dtype=torch.uint8)  # fmt: skip
    linear = torch.rand(rows_out, cols // NVFP4_BLOCK, generator=generator, device=device)
    linear = (linear * 4 + 0.25).to(torch.float8_e4m3fn)
    padded = -(-(cols // NVFP4_BLOCK) // 4) * 4
    blocked = to_blocked(torch, linear.view(torch.uint8)).view(-1, padded)
    return weight, linear, blocked.view(torch.float8_e4m3fn), torch.full((1,), 0.02, device=device)


def probe(torch: ModuleType, device: Device, launch: Launch) -> verify.Verdict:
    """The decode kernels against the exact law in float64: the GEMV at 1, 3 and `DECODE_ROWS`
    rows over each K split, and the decoded weight, with N not a multiple of a tile."""
    generator = torch.Generator(device=device).manual_seed(20261025)
    rows_out, cols = 200, 1024
    weight, linear, blocked, second = _probe_weight(torch, generator, device, rows_out, cols)
    exact = nvfp4_values(torch, weight, linear, second).double()
    errors: dict[str, float] = {}
    for m in (1, 3, DECODE_ROWS):
        x = torch.randn(m, cols, generator=generator, device=device).to(torch.bfloat16)
        want = x.double() @ exact.t()
        for split in (1, 2, 4):
            got = _rows(torch, launch, x, weight, blocked, second, split)
            errors[f"gemv M={m} split={split}"] = verify.error(torch, got, want)
    errors["dequant"] = verify.error(torch, _decoded(torch, launch, weight, blocked, second), exact)
    return verify.within(errors, prebuilt.BUDGET)


def _probe_kitchen(torch: ModuleType, index: int, module: ModuleType) -> verify.Verdict:
    """comfy-kitchen's W4A4 on this card: its GEMM against the exact (float64) product of the
    operands it was handed, and its quantizer, which rounds its own way, within `verify.SLACK` of
    the exact law's loss (`svdq_linear.fake`). bf16 and fp16, rows not a multiple of 16."""
    device = torch.device("cuda", index)
    generator = torch.Generator(device=device).manual_seed(20261025)
    rows_out, cols = 264, 1024
    weight, linear, blocked, second = _probe_weight(torch, generator, device, rows_out, cols)
    exact = nvfp4_values(torch, weight, linear, second).double()
    gemm: dict[str, float] = {}
    quantizer: dict[str, tuple[float, float]] = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        for m in (40, 300):
            x = torch.randn(m, cols, generator=generator, device=device).to(dtype)
            payload, block, tensor_scale = _quantized(torch, module, x)
            got = module.scaled_mm_nvfp4(
                payload, weight, tensor_scale, second, block, blocked, out_dtype=dtype
            )[:m]
            scales = from_blocked(torch, block.view(torch.uint8), int(payload.shape[0]), cols // 16)
            operand = nvfp4_values(torch, payload, scales.view(torch.float8_e4m3fn), tensor_scale)
            operand = operand[:m].double()
            case, law = f"{name} M={m}", svdq_linear.fake(torch, x.float(), tensor_scale)
            gemm[case] = verify.error(torch, got, operand @ exact.t())
            quantizer[case] = (verify.error(torch, operand, x), verify.error(torch, law, x))
    product, rounding = verify.within(gemm, prebuilt.BUDGET), verify.against_law(quantizer)
    detail = f"GEMM {product.detail}; quantizer {rounding.detail}"
    return verify.Verdict(product.ok and rounding.ok, detail)


def serving(index: int) -> str:
    """The kernel this card's nvfp4 leaves serve a product of more than `DECODE_ROWS` rows with
    now (fewer rows take the GEMV): a run's `gemm.selected`, read once a run."""
    if kernel_ready.ready(f"{KITCHEN} cuda:{index}") is not None:
        return int8_linear.ARTIFACT
    return DECODE if kernel_ready.ready(f"{KERNELS.name} cuda:{index}") is not None else EAGER
