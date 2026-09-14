"""FlashAttention-3's fp8-e4m3 forward, spelled as a diffusers attention backend (cr-136).

diffusers 0.40.0's `_flash_3_hub` hands the kernel `q_descale=k_descale=v_descale=None` and
constrains q/k/v to bf16/fp16, so FA3's 8-bit path — the fastest 8-bit arm h3a-013 measured
on an H100, from the SAME baked `.so` rank 1 already loads — was unreachable through any
`AttentionBackendName`. This module adds ONE backend to diffusers' registry, `_flash_3_hub_fp8`,
whose body quantises q/k/v per call (per-(batch, head) amax to e4m3, the descales handed to the
kernel, output bf16 from the kernel itself) and calls that same `flash_attn_func`. Nothing else
in diffusers is touched, and the bf16 entry's Hub slot is left alone.

THE MEMBER IS SYNTHESISED, because diffusers' enum is closed. `dispatch_attention_fn` resolves
a backend with `AttentionBackendName(backend)`, which returns a member by identity and looks
any other value up in `_value2member_map_`; both the registry dicts and that map are keyed
here with one instance built through `str.__new__`, so a processor pinned to it dispatches
exactly as one pinned to `_flash_3_hub` does. It is one instance per process, made once.

QUANTISATION IS PER CALL AND CHARGED TO THE KERNEL: amax over (sequence, head dim) per head,
a scale that maps that amax onto 440 rather than e4m3's 448 (e4m3fn has no inf — a product
that lands past 448 becomes NaN in torch's cast, not a saturated value), then the scaled
cast. On CUDA that is two Triton launches reading the tensor twice and writing it once
(~5 bytes per element), the same shape as cr-086's rowwise activation quantiser and for the
same reason: h3a-013 pass 3 measured the eager torch form at 5.6 ms per call at the 5 s cell
against a 42 ms kernel, and the fused form at 3.4 ms — the difference between 1.19x and
1.24x over rank 1. Everything else (a CPU probe, a venv without triton, a head dimension the
kernel is not built for) runs the eager passes and stays correct.
"""

from __future__ import annotations

from typing import Any

#: The backend's value in diffusers' vocabulary. Not a diffusers name; ours, and stable.
BACKEND = "_flash_3_hub_fp8"
MEMBER_NAME = "_FLASH_3_HUB_FP8"
#: The largest finite e4m3fn magnitude, and the value a head's amax is mapped onto instead.
E4M3_MAX = 448.0
E4M3_TARGET = 440.0
#: Below this a head's amax is treated as this, so an all-zero head quantises to zeros
#: instead of dividing by zero. Shared by the fused and the eager path so their descales agree.
SCALE_FLOOR = 1e-12
#: Rows of one head a fused program handles per launch; the head dimension is the tile width.
CHUNK = 64

_kernel: Any = None


class Unbound(RuntimeError):
    """The backend was dispatched before `bind` handed it a kernel. A programming error, not
    a refusal: resolution binds before any site is pinned."""


def member() -> Any:
    """The `AttentionBackendName` member for this backend, registered on first use.

    Registration is what makes the member REAL to diffusers: `_backends` holds the function,
    `_constraints` the same checks `_flash_3_hub` declares, and `_value2member_map_` lets
    `AttentionBackendName(BACKEND)` find it. Idempotent, and it does not need the kernel —
    a member that dispatches before `bind` raises `Unbound` rather than computing nothing.
    """
    from diffusers.models.attention_dispatch import (
        _check_device,
        _check_qkv_dtype_bf16_or_fp16,
        _check_shape,
    )

    from cozy_runtime.internal.attention_registry import register

    return register(
        BACKEND,
        _flash_attention_3_hub_fp8,
        constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
        supports_context_parallel=False,
    )


def bind(kernel: Any) -> Any:
    """Hand the backend the loaded snapshot's `flash_attn_func`; returns the member."""
    global _kernel
    _kernel = kernel
    return member()


def _triton_kernels() -> Any:
    """The fused per-head quantiser: an amax launch and a cast launch, or None where triton is
    absent or refuses. Cached once per process, like cr-086's rowwise kernel.

    Each program owns `CHUNK` rows of ONE head — a (CHUNK, dim) tile whose rows are `dim`
    contiguous elements `heads * dim` apart, which is what (batch, seq, heads, dim) looks like
    from inside a head. The amax pass folds into a (batch, heads) buffer with `atomic_max`,
    which is exact for non-negative f32 because their bit patterns order like their values.
    Same divide (`div_rn`), same clamp and the same RTNE downcast as cr-086; the one honest
    difference from the eager path is that triton's f32->e4m3 cast double-rounds through f16,
    so an element sitting exactly on a representational tie may pick the other neighbour.
    """
    cached = getattr(_triton_kernels, "_cached", "unset")
    if cached != "unset":
        return cached
    try:
        import triton
        import triton.language as tl

        @triton.jit  # type: ignore[untyped-decorator]
        def _amax(x_ptr, amax_ptr, seq, heads, CHUNK: tl.constexpr, DIM: tl.constexpr):  # type: ignore[no-untyped-def]
            head = tl.program_id(0)
            rows = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
            cols = tl.arange(0, DIM)
            batch = head // heads
            offsets = ((batch * seq + rows)[:, None] * heads + head % heads) * DIM + cols[None, :]
            value = tl.load(x_ptr + offsets, mask=rows[:, None] < seq, other=0.0).to(tl.float32)
            tl.atomic_max(amax_ptr + head, tl.max(tl.abs(value)))

        @triton.jit  # type: ignore[untyped-decorator]
        def _cast(x_ptr, out_ptr, amax_ptr, seq, heads, CHUNK: tl.constexpr, DIM: tl.constexpr):  # type: ignore[no-untyped-def]
            head = tl.program_id(0)
            rows = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
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

        kernels: Any = (_amax, _cast, triton.cdiv)
    except Exception:
        kernels = None
    _triton_kernels._cached = kernels  # type: ignore[attr-defined]
    return kernels


def quantise(tensor: Any) -> tuple[Any, Any]:
    """(batch, seq, heads, dim) bf16/fp16 -> e4m3 of the same shape and a (batch, heads) fp32
    descale, the form FA3's forward takes (`CHECK_SHAPE(q_descale, batch_size, num_heads_k)`).

    The fused path serves a contiguous CUDA tensor whose head dimension is a power of two the
    tile can be built for; everything else takes `quantise_eager`.
    """
    import torch

    batch, seq, heads, dim = tensor.shape
    kernels = _triton_kernels() if tensor.is_cuda and tensor.is_contiguous() else None
    if kernels is None or dim & (dim - 1) or not (16 <= dim <= 256) or not tensor.numel():
        return quantise_eager(tensor)
    amax_kernel, cast_kernel, cdiv = kernels
    amax = torch.zeros((batch, heads), dtype=torch.float32, device=tensor.device)
    out = torch.empty(tensor.shape, dtype=torch.float8_e4m3fn, device=tensor.device)
    grid = (batch * heads, cdiv(seq, CHUNK))
    amax_kernel[grid](tensor, amax, seq, heads, CHUNK=CHUNK, DIM=dim)
    cast_kernel[grid](tensor, out, amax, seq, heads, CHUNK=CHUNK, DIM=dim)
    return out, (amax / E4M3_TARGET).clamp_(min=SCALE_FLOOR)


def quantise_eager(tensor: Any) -> tuple[Any, Any]:
    """The same arithmetic in torch passes: the executable floor, and what a CPU probe runs."""
    import torch

    amax = tensor.abs().amax(dim=(1, 3)).float()
    descale = (amax / E4M3_TARGET).clamp_(min=SCALE_FLOOR)
    scaled = (tensor.float() / descale[:, None, :, None]).clamp_(-E4M3_MAX, E4M3_MAX)
    return scaled.to(torch.float8_e4m3fn), descale.contiguous()


def _flash_attention_3_hub_fp8(
    query: Any,
    key: Any,
    value: Any,
    attn_mask: Any = None,
    scale: float | None = None,
    is_causal: bool = False,
    _parallel_config: Any = None,
) -> Any:
    if attn_mask is not None:
        raise ValueError("`attn_mask` is not supported for flash-attn 3.")
    if _parallel_config is not None:
        raise ValueError("flash-attn3-fp8 does not serve context parallelism")
    if _kernel is None:
        raise Unbound("flash-attn3-fp8 was dispatched before its kernel was bound")
    q8, q_descale = quantise(query)
    k8, k_descale = quantise(key)
    v8, v_descale = quantise(value)
    out = _kernel(
        q8,
        k8,
        v8,
        softmax_scale=scale,
        causal=is_causal,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    return out[0] if isinstance(out, (tuple, list)) else out
