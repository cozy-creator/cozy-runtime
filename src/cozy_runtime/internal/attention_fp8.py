"""Per-(batch, head) e4m3 quantisation for FlashAttention-3's fp8 forward (cr-136).

Amax over (sequence, head dim) per head maps onto 440 rather than e4m3's 448 (e4m3fn has no
inf, so a product past 448 becomes NaN in torch's cast), then the scaled cast. On CUDA that
is two Triton launches (~5 bytes per element): h3a-013 measured the eager form at 5.6 ms per
call against a 42 ms kernel and the fused form at 3.4 ms. Everything else (CPU, no Triton or
C compiler, a head dimension that is not a power of two the tile takes) runs the eager passes.
Every reduction stays within one head, so Ulysses' head split leaves each head's scale exactly
what one GPU computes.

The kernels compile on the machine's compile manager (`KERNELS`, `triton_kernels`), never in
this process; a kernel that launches them serves once they are loaded here (`serve`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from cozy_runtime.author.kernels import Launch, TritonEntry, TritonKernels

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - a CPU-only or non-Linux environment
    triton = None

#: The largest finite e4m3fn magnitude, and the value a head's amax is mapped onto instead.
E4M3_MAX = 448.0
E4M3_TARGET = 440.0
#: Below this a head's amax is treated as this, so an all-zero head quantises to zeros
#: instead of dividing by zero. Shared by the fused and the eager path so their descales agree.
SCALE_FLOOR = 1e-12
#: Rows of one head a fused program handles per launch; the head dimension is the tile width.
CHUNK = 64
#: The head dimensions the fused path serves: the powers of two the tile takes.
DIMS = (16, 32, 64, 128, 256)
#: Without its kernels (no Triton or C compiler here) it quantises eagerly, so a kernel that
#: launches it still serves.
EAGER_FLOOR = True

if triton is not None:
    # Each program owns `CHUNK` rows of ONE head: a (CHUNK, dim) tile whose rows are `dim`
    # contiguous elements `heads * dim` apart, which is what (batch, seq, heads, dim) looks like
    # from inside a head. The amax pass folds into a (batch, heads) buffer with `atomic_max`,
    # exact for non-negative f32 because their bit patterns order like their values. Same divide
    # (`div_rn`), clamp and RTNE downcast as cr-086; the one honest difference from the eager
    # path is that triton's f32->e4m3 cast double-rounds through f16, so an element exactly on a
    # representational tie may pick the other neighbour. `HALF` (fp16 input) keys the entry.

    @triton.jit  # type: ignore[untyped-decorator]
    def _amax(  # type: ignore[no-untyped-def]
        x_ptr, amax_ptr, seq, heads, CHUNK: tl.constexpr, DIM: tl.constexpr, HALF: tl.constexpr
    ):
        # int64 element offsets: a (seq, heads, dim) tensor may pass 2**31 elements.
        head = tl.program_id(0).to(tl.int64)
        rows = tl.program_id(1).to(tl.int64) * CHUNK + tl.arange(0, CHUNK)
        cols = tl.arange(0, DIM)
        batch = head // heads
        offsets = ((batch * seq + rows)[:, None] * heads + head % heads) * DIM + cols[None, :]
        value = tl.load(x_ptr + offsets, mask=rows[:, None] < seq, other=0.0).to(tl.float32)
        tl.atomic_max(amax_ptr + head, tl.max(tl.abs(value)))

    @triton.jit  # type: ignore[untyped-decorator]
    def _cast(  # type: ignore[no-untyped-def]
        x_ptr,
        out_ptr,
        amax_ptr,
        seq,
        heads,
        CHUNK: tl.constexpr,
        DIM: tl.constexpr,
        HALF: tl.constexpr,
    ):
        head = tl.program_id(0).to(tl.int64)
        rows = tl.program_id(1).to(tl.int64) * CHUNK + tl.arange(0, CHUNK)
        cols = tl.arange(0, DIM)
        batch = head // heads
        offsets = ((batch * seq + rows)[:, None] * heads + head % heads) * DIM + cols[None, :]
        mask = rows[:, None] < seq
        value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.load(amax_ptr + head) / 440.0, 1e-12)
        quantized = tl.clamp(tl.div_rn(value, scale), -448.0, 448.0)
        tl.store(
            out_ptr + offsets,
            quantized.to(tl.float8e4nv, fp_downcast_rounding="rtne"),
            mask=mask,
        )


def _entries() -> list[TritonEntry]:
    """Each (dtype, head dimension) a model's attention takes; lengths are runtime sizes."""
    if triton is None:
        return []
    found = []
    for code, half in enumerate(("bf16", "fp16")):
        for dim in DIMS:
            constexprs = {"CHUNK": CHUNK, "DIM": dim, "HALF": code}
            pointers = {"x_ptr": f"*{half}", "amax_ptr": "*fp32"}
            sizes = dict(
                seq="i32", heads="i32", CHUNK="constexpr", DIM="constexpr", HALF="constexpr"
            )
            found.append(
                TritonEntry(
                    _amax, {**pointers, **sizes}, constexprs, ("x_ptr", "amax_ptr"), num_warps=4
                )
            )
            found.append(
                TritonEntry(
                    _cast,
                    {**pointers, "out_ptr": "*fp8e4nv", **sizes},
                    constexprs,
                    ("x_ptr", "out_ptr", "amax_ptr"),
                    num_warps=4,
                )
            )
    return found


KERNELS = TritonKernels("cozy-attention-e4m3", _entries())
#: Device index -> its loaded launcher (`serve`).
_LAUNCH: dict[int, Launch] = {}


def serve(index: int, launch: Launch) -> None:
    """GPU `index`'s launcher of `KERNELS`, loaded by the attention kernel that needs it."""
    _LAUNCH[index] = launch


def quantise(tensor: Tensor) -> tuple[Tensor, Tensor]:
    """(batch, seq, heads, dim) bf16/fp16 -> e4m3 of the same shape and a (batch, heads) fp32
    descale, the form FA3's forward takes (`CHECK_SHAPE(q_descale, batch_size, num_heads_k)`).

    The fused path serves a contiguous CUDA tensor of a built head dimension on a GPU whose
    launcher is served (one not 16-byte aligned is copied first); everything else takes
    `quantise_eager`."""
    import torch

    batch, seq, heads, dim = tensor.shape
    launch = _LAUNCH.get(tensor.device.index or 0) if tensor.is_cuda else None
    if (
        launch is None
        or dim not in DIMS
        or tensor.dtype not in (torch.bfloat16, torch.float16)
        or not tensor.is_contiguous()
        or not tensor.numel()
    ):
        return quantise_eager(tensor)
    tensor = tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()
    amax = torch.zeros((batch, heads), dtype=torch.float32, device=tensor.device)
    out = torch.empty(tensor.shape, dtype=torch.float8_e4m3fn, device=tensor.device)
    grid, half = (batch * heads, -(-seq // CHUNK), 1), int(tensor.dtype == torch.float16)
    launch(_amax, grid, tensor, amax, seq, heads, CHUNK, dim, half)
    launch(_cast, grid, tensor, out, amax, seq, heads, CHUNK, dim, half)
    return out, (amax / E4M3_TARGET).clamp_(min=SCALE_FLOOR)


def quantise_eager(tensor: Tensor) -> tuple[Tensor, Tensor]:
    """The same arithmetic in torch passes: the executable floor, and what a CPU probe runs."""
    import torch

    amax = tensor.abs().amax(dim=(1, 3)).float()
    descale = (amax / E4M3_TARGET).clamp_(min=SCALE_FLOOR)
    scaled = (tensor.float() / descale[:, None, :, None]).clamp_(-E4M3_MAX, E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn), descale.contiguous()
