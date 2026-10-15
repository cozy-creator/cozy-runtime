"""The Runtime's fused DiT elementwise kernels: the chains a single-stream adaLN block runs
between its GEMMs, one pass each (`cozy_runtime.fusion` declares where they apply).

Each kernel rounds to bf16 wherever eager bf16 arithmetic rounds (`_rb`, in the integer domain
so the compiler cannot fold the round trip), so a fused block reproduces eager up to reduction
order. Widths are constexprs: a layout's set compiles only the entries it declares.

Adapted from MiniMax-H3's glue (itself from NVlabs/Sana `sol-engine`, Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cozy_runtime.author.kernels import Launch, TritonEntry

try:
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    TRITON_ABSENT = f"torch and triton do not import ({_absent})"
else:
    TRITON_ABSENT = ""

#: `NORM` values; `NONE` is the gated residual alone.
NONE, LAYER, RMS = 0, 1, 2
ROPE_HEADS = 8
SWIGLU_BLOCK = 1024
_ROW_WARPS = 8
_ROPE_WARPS = 4
_SWIGLU_WARPS = 4

if not TRITON_ABSENT:

    @triton.jit
    def _rb(x):  # type: ignore[no-untyped-def]
        bits = x.to(tl.int32, bitcast=True)
        bits = bits + 0x7FFF + ((bits >> 16) & 1)
        return (bits & -65536).to(tl.float32, bitcast=True)

    @triton.jit  # type: ignore[untyped-decorator]
    def norm_modulate_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        branch_ptr,
        gate_ptr,
        hidden_ptr,
        out_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        index_ptr,
        n_cols,
        n_index,
        stride_x,
        stride_table,
        eps,
        NORM: tl.constexpr,
        WEIGHT: tl.constexpr,
        SHIFT: tl.constexpr,
        RESIDUAL: tl.constexpr,
        TANH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One row per program. RESIDUAL: `hidden = x + act(gate) * branch` first (written);
        # then `norm(hidden) * (1 + scale) [+ shift]`. Table rows are per token (`index`).
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offset = row * stride_x + cols
        table = tl.load(index_ptr + row % n_index).to(tl.int64) * stride_table + cols
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if RESIDUAL:
            branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            gate = tl.load(gate_ptr + table, mask=mask, other=0.0).to(tl.float32)
            if TANH:
                gate = _rb(libdevice.tanh(gate))
            x = _rb(x + _rb(gate * branch))
            tl.store(hidden_ptr + offset, x.to(tl.bfloat16), mask=mask)
        if NORM != 0:
            if NORM == 1:
                mean = tl.sum(x, axis=0) / n_cols
                x = tl.where(mask, x - mean, 0.0)
            variance = tl.sum(x * x, axis=0) / n_cols
            normed = _rb(x * tl.math.rsqrt(variance + eps))
            if WEIGHT:
                weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
                normed = _rb(normed * weight)
            scale = tl.load(scale_ptr + table, mask=mask, other=0.0).to(tl.float32)
            out = _rb(normed * _rb(1.0 + scale))
            if SHIFT:
                out = _rb(out + tl.load(shift_ptr + table, mask=mask, other=0.0).to(tl.float32))
            tl.store(out_ptr + offset, out.to(tl.bfloat16), mask=mask)

    @triton.jit  # type: ignore[untyped-decorator]
    def swiglu_kernel(  # type: ignore[no-untyped-def]
        gate_ptr,
        up_ptr,
        out_ptr,
        n_cols,
        stride_gate,
        stride_up,
        stride_out,
        BLOCK: tl.constexpr,
    ):
        # `silu(gate) * up`, torch's silu: `x / (1 + exp(-x))` in fp32, IEEE divide.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < n_cols
        gate = tl.load(gate_ptr + row * stride_gate + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + row * stride_up + cols, mask=mask, other=0.0).to(tl.float32)
        activated = _rb(tl.div_rn(gate, 1.0 + libdevice.exp(-gate)))
        tl.store(out_ptr + row * stride_out + cols, _rb(activated * up).to(tl.bfloat16), mask=mask)

    @triton.jit  # type: ignore[untyped-decorator]
    def head_norm_rope_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        out_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        seq,
        heads,
        stride_token,
        stride_channel,
        eps,
        HEAD_DIM: tl.constexpr,
        ROTARY: tl.constexpr,
        HEADS: tl.constexpr,
        INTERLEAVED: tl.constexpr,
    ):
        # One program per (row, HEADS heads) of `[rows, heads, HEAD_DIM]`: per-head RMSNorm with
        # weight, then RoPE on the first ROTARY channels. INTERLEAVED pairs channel 2i with
        # 2i+1 (table column i); otherwise channel j pairs with j +/- ROTARY/2 (column j).
        # The partner is re-read and re-normalized rather than shuffled across lanes.
        row = tl.program_id(0).to(tl.int64)
        token = row % seq
        head = tl.program_id(1) * HEADS + tl.arange(0, HEADS)[:, None]
        chan = tl.arange(0, HEAD_DIM)[None, :]
        mask = head < heads
        base = (row * heads + head) * HEAD_DIM
        x = tl.load(x_ptr + base + chan, mask=mask, other=0.0).to(tl.float32)
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1)[:, None] / HEAD_DIM + eps)
        weight = tl.load(weight_ptr + chan).to(tl.float32)
        normed = _rb(_rb(x * inv) * weight)
        rotating = chan < ROTARY
        if INTERLEAVED:
            low = chan % 2 == 0
            partner = chan ^ 1
            column = chan // 2
        else:
            low = chan < ROTARY // 2
            partner = tl.where(low, chan + ROTARY // 2, chan - ROTARY // 2)
            column = chan
        paired = tl.load(x_ptr + base + partner, mask=mask & rotating, other=0.0).to(tl.float32)
        paired_weight = tl.load(weight_ptr + partner, mask=rotating, other=0.0).to(tl.float32)
        paired = _rb(_rb(paired * inv) * paired_weight)
        paired = tl.where(low, -paired, paired)
        table = token * stride_token + column * stride_channel
        cos = tl.load(cos_ptr + table, mask=rotating, other=1.0)
        sin = tl.load(sin_ptr + table, mask=rotating, other=0.0)
        out = tl.where(rotating, _rb(normed * cos + paired * sin), normed)
        tl.store(out_ptr + base + chan, out.to(tl.bfloat16), mask=mask)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _signature(kernel: Any, **types: str) -> dict[str, str]:
    """Triton's signature of `kernel`: UPPERCASE parameters are constexprs, `*_ptr` bf16
    pointers and the rest i32, except as `types` says."""
    return {
        name: types.get(name)
        or ("constexpr" if name.isupper() else "*bf16" if name.endswith("_ptr") else "i32")
        for name in kernel.arg_names
    }


def row_entry(
    width: int, norm: int, *, weight: bool, shift: bool, residual: bool, tanh: bool
) -> TritonEntry:
    constexprs = {
        "NORM": norm,
        "WEIGHT": int(weight and norm != NONE),
        "SHIFT": int(shift and norm != NONE),
        "RESIDUAL": int(residual),
        "TANH": int(tanh and residual),
        "BLOCK": _next_pow2(width),
    }
    signature = _signature(norm_modulate_kernel, index_ptr="*i64", eps="fp32")
    # Row operands every launch aligns (`_aligned`); weights and per-token indexes may not be.
    aligned = tuple(
        n
        for n, t in signature.items()
        if t in ("*bf16", "i32") and n not in ("weight_ptr", "n_index")
    )
    return TritonEntry(norm_modulate_kernel, signature, constexprs, aligned, _ROW_WARPS)


def swiglu_entry() -> TritonEntry:
    signature = _signature(swiglu_kernel)
    aligned = tuple(n for n, t in signature.items() if t != "constexpr")
    return TritonEntry(swiglu_kernel, signature, {"BLOCK": SWIGLU_BLOCK}, aligned, _SWIGLU_WARPS)


def rope_entry(head_dim: int, rotary: int, *, interleaved: bool) -> TritonEntry:
    constexprs = {
        "HEAD_DIM": head_dim,
        "ROTARY": rotary,
        "HEADS": ROPE_HEADS,
        "INTERLEAVED": int(interleaved),
    }
    signature = _signature(head_norm_rope_kernel, cos_ptr="*fp32", sin_ptr="*fp32", eps="fp32")
    return TritonEntry(
        head_norm_rope_kernel, signature, constexprs, ("x_ptr", "out_ptr"), _ROPE_WARPS
    )


# ---------------------------------------------------------------------------- launches


def _aligned(t: Any) -> Any:
    """A row operand as the entries assume it (`tt.divisibility` 16): unit column stride, a
    16-byte aligned base and a row stride divisible by 16."""
    if t.stride(-1) != 1 or t.data_ptr() % 16 or (t.dim() > 1 and t.stride(-2) % 16):
        return t.contiguous()
    return t


def _rows(x: Any, width: int) -> Any:
    return _aligned(x.reshape(-1, width))


def _tables(*tables: Any) -> list[Any]:
    """Modulation tables as one kernel addresses them: aligned, sharing one row stride."""
    found = [_aligned(t) for t in tables if t is not None]
    if len({t.stride(0) for t in found}) > 1:
        found = [t.contiguous() for t in found]
    return found


def norm_modulate(
    launch: Launch,
    entry: TritonEntry,
    x: Any,
    weight: Any,
    scale: Any,
    shift: Any,
    index: Any,
    eps: float,
    *,
    branch: Any = None,
    gate: Any = None,
) -> tuple[Any, Any]:
    """`entry`'s chain over rows of `x`: `(hidden, out)`. `hidden` is `x + act(gate) * branch`
    (or `x` without a residual); `out` the modulated norm (None for a residual alone).
    Tables are `[rows, width]` views (any row stride) read at `index[token]`."""
    width = int(x.shape[-1])
    rows = _rows(x, width)
    n = int(rows.shape[0])
    residual = branch is not None
    hidden = torch.empty_like(rows) if residual else rows
    normed = entry.constexprs["NORM"] != NONE
    out = torch.empty_like(rows) if normed else hidden
    tables = iter(_tables(gate, scale, shift))
    gate, scale, shift = (next(tables) if t is not None else None for t in (gate, scale, shift))
    first = scale if scale is not None else gate
    launch(
        norm_modulate_kernel,
        (n, 1, 1),
        rows,
        _rows(branch, width) if residual else rows,
        gate if gate is not None else first,
        hidden,
        out,
        weight if weight is not None else first,
        first,
        shift if shift is not None else first,
        index,
        width,
        int(index.numel()),
        int(rows.stride(0)),
        int(first.stride(0)),
        float(eps),
        *(entry.constexprs[k] for k in ("NORM", "WEIGHT", "SHIFT", "RESIDUAL", "TANH", "BLOCK")),
    )
    return hidden.view_as(x), out.view_as(x) if normed else None


def swiglu(launch: Launch, gate: Any, up: Any) -> Any:
    """`silu(gate) * up`, one read of each and one write."""
    width = int(gate.shape[-1])
    gate_rows, up_rows = _rows(gate, width), _rows(up, width)
    n = int(gate_rows.shape[0])
    out = torch.empty(n, width, dtype=gate.dtype, device=gate.device)
    launch(
        swiglu_kernel,
        (n, triton.cdiv(width, SWIGLU_BLOCK), 1),
        gate_rows,
        up_rows,
        out,
        width,
        int(gate_rows.stride(0)),
        int(up_rows.stride(0)),
        width,
        SWIGLU_BLOCK,
    )
    return out.view_as(gate)


def head_norm_rope(
    launch: Launch, entry: TritonEntry, x: Any, weight: Any, eps: float, tables: Any
) -> Any:
    """Per-head RMSNorm of `[batch, seq, heads, head_dim]` `x`, then `entry`'s RoPE by
    `tables` (`rope_tables`)."""
    batch, seq, heads, head_dim = (int(d) for d in x.shape)
    flat = _aligned(x.reshape(batch * seq, heads * head_dim))
    out = torch.empty_like(flat)
    cos, sin, stride_token, stride_channel = tables
    c = entry.constexprs
    launch(
        head_norm_rope_kernel,
        (batch * seq, triton.cdiv(heads, ROPE_HEADS), 1),
        flat,
        out,
        weight,
        cos,
        sin,
        seq,
        heads,
        stride_token,
        stride_channel,
        float(eps),
        c["HEAD_DIM"],
        c["ROTARY"],
        c["HEADS"],
        c["INTERLEAVED"],
    )
    return out.view(batch, seq, heads, head_dim)


def rope_tables(value: Any, *, interleaved: bool) -> tuple[Any, Any, int, int]:
    """A model's rotary operand as the kernel reads it: `(cos, sin, token stride, column
    stride)` in fp32. Interleaved: complex `[seq, rotary/2]` or a `(cos, sin)` pair of
    `[seq, rotary/2]`; split-half: a `(cos, sin)` pair of `[seq, rotary]`."""
    if isinstance(value, torch.Tensor) and value.is_complex():
        pairs = torch.view_as_real(value.to(torch.complex64))
        pairs = pairs if pairs.stride(-1) == 1 and pairs.stride(-2) == 2 else pairs.contiguous()
        return pairs[..., 0], pairs[..., 1], int(pairs.stride(-3)), 2
    cos, sin = (t.float().contiguous() for t in value)
    if interleaved and cos.shape != sin.shape:
        raise ValueError("rotary cos and sin differ in shape")
    return cos, sin, int(cos.stride(-2)), 1


@dataclass(frozen=True, slots=True)
class Entries:
    """The entries one declared block layout launches."""

    first: TritonEntry
    """norm1, modulated."""
    second: TritonEntry
    """The attention's gated residual, then norm2, modulated."""
    residual: TritonEntry
    """The MLP's gated residual."""
    swiglu: TritonEntry
    rope: TritonEntry | None

    def all(self) -> tuple[TritonEntry, ...]:
        found = (self.first, self.second, self.residual, self.swiglu, self.rope)
        return tuple(e for e in found if e is not None)


def entries(
    width: int,
    head_dim: int,
    rotary: int | None,
    *,
    norm: int,
    weight: bool,
    shift: bool,
    tanh: bool,
    interleaved: bool,
) -> Entries:
    def row(norm: int, residual: bool) -> TritonEntry:
        return row_entry(width, norm, weight=weight, shift=shift, residual=residual, tanh=tanh)

    return Entries(
        row(norm, False),
        row(norm, True),
        row(NONE, True),
        swiglu_entry(),
        None if rotary is None else rope_entry(head_dim, rotary, interleaved=interleaved),
    )
