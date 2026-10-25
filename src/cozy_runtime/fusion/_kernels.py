"""The Runtime's fused DiT elementwise kernels: the chains a single-stream block runs between its
GEMMs, one pass each (`cozy_runtime.fusion` declares where they apply).

Each kernel rounds wherever eager rounds: bf16 rows round every step to bf16 (`_rb`, in the
integer domain so the compiler cannot fold the round trip); fp16 rows (a block under fp16
autocast over an fp32 residual stream) keep the residual arithmetic in fp32, an IEEE multiply
then add, and round a norm once to fp16 (`_rh`). A q/k norm and RoPE the model computes in fp32
round once. So a fused block reproduces eager up to reduction order. Widths are constexprs: a
layout's set compiles only the entries it declares.

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
#: `ROUND` values: keep fp32, round to bf16, round to fp16.
FP32, BF16, FP16 = 0, 1, 2
ROPE_HEADS = 8
SWIGLU_BLOCK = 1024
_ROPE_WARPS = 4
_SWIGLU_WARPS = 4
#: Each kernel's constexprs, in its parameter order.
ROW_CONSTEXPRS = ("NORM", "WEIGHT", "ONCE", "MODULATE", "BRANCHES", "TANH", "F16", "X16", "BLOCK")
ROPE_CONSTEXPRS = (
    "HEAD_DIM", "ROTARY", "HEADS", "INTERLEAVED", "WEIGHT", "ONCE", "ROUND", "PRODUCT", "X32"
)  # fmt: skip

if not TRITON_ABSENT:

    @triton.jit
    def _rb(x):  # type: ignore[no-untyped-def]
        bits = x.to(tl.int32, bitcast=True)
        bits = bits + 0x7FFF + ((bits >> 16) & 1)
        return (bits & -65536).to(tl.float32, bitcast=True)

    @triton.jit
    def _rh(x):  # type: ignore[no-untyped-def]
        return tl.inline_asm_elementwise(
            "{ .reg .b16 h; cvt.rn.f16.f32 h, $1; cvt.f32.f16 $0, h; }",
            "=f,f",
            [x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit  # type: ignore[untyped-decorator]
    def _round(x, ROUND: tl.constexpr):  # type: ignore[no-untyped-def]
        if ROUND == 1:
            x = _rb(x)
        elif ROUND == 2:
            x = _rh(x)
        return x

    @triton.jit  # type: ignore[untyped-decorator]
    def _normed(x, inv, weight, WEIGHT: tl.constexpr, ONCE: tl.constexpr, ROUND: tl.constexpr):  # type: ignore[no-untyped-def]
        # `x * inv [* weight]` as the model's norm rounds it: Diffusers' RMSNorm rounds the
        # normalized row, then its product with the weight; torch's (ONCE) only the product.
        x = x * inv
        if WEIGHT:
            if ONCE == 0:
                x = _round(x, ROUND)
            x = x * weight
        return _round(x, ROUND)

    @triton.jit  # type: ignore[untyped-decorator]
    def norm_modulate_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        branch_ptr,
        other_ptr,
        gate_ptr,
        hidden_ptr,
        out_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        index_ptr,
        n_cols,
        n_index,
        stride_table,
        eps,
        NORM: tl.constexpr,
        WEIGHT: tl.constexpr,
        ONCE: tl.constexpr,
        MODULATE: tl.constexpr,
        BRANCHES: tl.constexpr,
        TANH: tl.constexpr,
        F16: tl.constexpr,
        X16: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # One row per program. BRANCHES: `hidden = x + act(gate) * (branch [+ other])` first
        # (written); then `norm(hidden) [* (1 + scale) [+ shift]]` (MODULATE 1, 2). Table rows
        # are per token (`index`). F16: fp16 rows over an fp32 residual stream (X16: x is fp16).
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        offset = row * n_cols + cols
        table = tl.load(index_ptr + row % n_index).to(tl.int64) * stride_table + cols
        x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
        if BRANCHES > 0:
            branch = tl.load(branch_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            if BRANCHES == 2:
                other = tl.load(other_ptr + offset, mask=mask, other=0.0).to(tl.float32)
                branch = _round(branch + other, 1 - F16)
            gate = tl.load(gate_ptr + table, mask=mask, other=0.0).to(tl.float32)
            if TANH:
                gate = _round(libdevice.tanh(gate), 1 - F16)
            product = _round(libdevice.mul_rn(gate, branch), 1 - F16)
            x = _round(libdevice.add_rn(x, product), 1 - F16)
            tl.store(hidden_ptr + offset, x.to(hidden_ptr.dtype.element_ty), mask=mask)
        if NORM != 0:
            if NORM == 1:
                mean = tl.sum(x, axis=0) / n_cols
                x = tl.where(mask, x - mean, 0.0)
            inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
            if WEIGHT:
                weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            else:
                weight = inv
            out = _normed(x, inv, weight, WEIGHT, ONCE, 1 + F16)
            if MODULATE > 0:
                scale = tl.load(scale_ptr + table, mask=mask, other=0.0).to(tl.float32)
                out = _round(out * _round(1.0 + scale, 1 + F16), 1 + F16)
            if MODULATE == 2:
                shift = tl.load(shift_ptr + table, mask=mask, other=0.0).to(tl.float32)
                out = _round(out + shift, 1 + F16)
            tl.store(out_ptr + offset, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit  # type: ignore[untyped-decorator]
    def swiglu_kernel(  # type: ignore[no-untyped-def]
        gate_ptr,
        up_ptr,
        out_ptr,
        n_cols,
        stride_gate,
        stride_up,
        stride_out,
        F16: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # `silu(gate) * up`, torch's silu: `x / (1 + exp(-x))` in fp32, IEEE divide.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < n_cols
        gate = tl.load(gate_ptr + row * stride_gate + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + row * stride_up + cols, mask=mask, other=0.0).to(tl.float32)
        activated = _round(tl.div_rn(gate, 1.0 + libdevice.exp(-gate)), 1 + F16)
        out = _round(activated * up, 1 + F16)
        tl.store(out_ptr + row * stride_out + cols, out.to(out_ptr.dtype.element_ty), mask=mask)

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
        WEIGHT: tl.constexpr,
        ONCE: tl.constexpr,
        ROUND: tl.constexpr,
        PRODUCT: tl.constexpr,
        X32: tl.constexpr,
    ):
        # One program per (row, HEADS heads) of `[rows, heads, HEAD_DIM]`: per-head RMSNorm [with
        # weight], then RoPE on the first ROTARY channels. INTERLEAVED pairs channel 2i with
        # 2i+1 (table column i); otherwise channel j pairs with j +/- ROTARY/2 (column j).
        # The partner is re-read and re-normalized rather than shuffled across lanes. ROUND: the
        # row dtype each step rounds to (FP32: the model computes this in fp32 and rounds once,
        # at the end). PRODUCT: the tables are cast to the row dtype and each product rounds, as
        # eager `x * cos + rotated * sin` in that dtype does; else the sum rounds once. X32: x is
        # fp32 (a processor that normalizes in fp32).
        row = tl.program_id(0).to(tl.int64)
        token = row % seq
        head = tl.program_id(1) * HEADS + tl.arange(0, HEADS)[:, None]
        chan = tl.arange(0, HEAD_DIM)[None, :]
        mask = head < heads
        base = (row * heads + head) * HEAD_DIM
        x = tl.load(x_ptr + base + chan, mask=mask, other=0.0).to(tl.float32)
        inv = tl.math.rsqrt(tl.sum(x * x, axis=1)[:, None] / HEAD_DIM + eps)
        weight = tl.load(weight_ptr + chan).to(tl.float32) if WEIGHT else inv
        normed = _normed(x, inv, weight, WEIGHT, ONCE, ROUND)
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
        if WEIGHT:
            weight = tl.load(weight_ptr + partner, mask=rotating, other=0.0).to(tl.float32)
        paired = _normed(paired, inv, weight, WEIGHT, ONCE, ROUND)
        paired = tl.where(low, -paired, paired)
        table = token * stride_token + column * stride_channel
        cos = tl.load(cos_ptr + table, mask=rotating, other=1.0)
        sin = tl.load(sin_ptr + table, mask=rotating, other=0.0)
        if PRODUCT:
            cos, sin = _round(cos, ROUND), _round(sin, ROUND)
            turned = _round(_round(normed * cos, ROUND) + _round(paired * sin, ROUND), ROUND)
        else:
            turned = _round(normed * cos + paired * sin, ROUND)
        out = tl.where(rotating, turned, normed)
        tl.store(out_ptr + base + chan, out.to(out_ptr.dtype.element_ty), mask=mask)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _entry(
    kernel: Any, constexprs: dict[str, int], warps: int, aligned: tuple[str, ...], **types: str
) -> TritonEntry:
    """`kernel`'s entry: UPPERCASE parameters are constexprs, the rest i32 unless `types` says."""
    signature = {
        name: "constexpr" if name.isupper() else types.get(name, "i32") for name in kernel.arg_names
    }
    return TritonEntry(kernel, signature, constexprs, aligned, warps)


# ---------------------------------------------------------------------------- launches


def _aligned(t: Any) -> Any:
    """A row operand as the entries assume it (`tt.divisibility` 16): unit column stride, a
    16-byte aligned base and a row stride divisible by 16."""
    if t.stride(-1) != 1 or (t.dim() > 1 and t.stride(-2) % 16):
        t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def _rows(x: Any, width: int) -> Any:
    """`[rows, width]` with any aligned row stride."""
    return _aligned(x.reshape(-1, width))


def _dense(x: Any, width: int) -> Any:
    """`[rows, width]`, contiguous: the row kernel addresses every operand at `row * width`."""
    rows = x.reshape(-1, width)
    return _aligned(rows if rows.is_contiguous() else rows.contiguous())


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
    other: Any = None,
    gate: Any = None,
) -> tuple[Any, Any]:
    """`entry`'s chain over rows of `x`: `(hidden, out)`. `hidden` is `x + act(gate) * (branch
    [+ other])` (or `x` without a residual); `out` the [modulated] norm (None for a residual
    alone). Tables are `[rows, width]` views (any row stride) read at `index[token]`."""
    width = int(x.shape[-1])
    rows = _dense(x, width)
    n = int(rows.shape[0])
    c = {**entry.constexprs, "X16": int(entry.constexprs["F16"] and x.dtype is torch.float16)}
    residual, row = (torch.float32, torch.float16) if c["F16"] else (rows.dtype, rows.dtype)
    hidden = rows.new_empty(rows.shape, dtype=residual) if c["BRANCHES"] else rows
    normed = c["NORM"] != NONE
    out = rows.new_empty(rows.shape, dtype=row) if normed else hidden
    branch = _dense(branch, width) if c["BRANCHES"] else rows
    other = _dense(other, width) if c["BRANCHES"] == 2 else branch
    tables = iter(_tables(gate, scale, shift))
    gate, scale, shift = (next(tables) if t is not None else None for t in (gate, scale, shift))
    first = next((t for t in (scale, gate) if t is not None), rows)
    launch(
        norm_modulate_kernel,
        (n, 1, 1),
        rows,
        branch,
        other,
        gate if gate is not None else first,
        hidden,
        out,
        weight if weight is not None else first,
        first,
        shift if shift is not None else first,
        index,
        width,
        int(index.numel()),
        int(first.stride(0)),
        float(eps),
        *(c[k] for k in ROW_CONSTEXPRS),
    )
    return hidden.view_as(x), out.view_as(x) if normed else None


def swiglu(launch: Launch, entry: TritonEntry, gate: Any, up: Any) -> Any:
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
        entry.constexprs["F16"],
        SWIGLU_BLOCK,
    )
    return out.view_as(gate)


def head_norm_rope(
    launch: Launch,
    entry: TritonEntry,
    x: Any,
    weight: Any,
    eps: float,
    tables: Any,
    dtype: Any,
) -> Any:
    """Per-head RMSNorm of `[batch, seq, heads, head_dim]` `x`, then `entry`'s RoPE by
    `tables` (`rope_tables`), in `dtype`."""
    batch, seq, heads, head_dim = (int(d) for d in x.shape)
    flat = _aligned(x.reshape(batch * seq, heads * head_dim))
    out = torch.empty(flat.shape, dtype=dtype, device=flat.device)
    cos, sin, stride_token, stride_channel = tables
    c = {**entry.constexprs, "X32": int(x.dtype is torch.float32)}
    launch(
        head_norm_rope_kernel,
        (batch * seq, triton.cdiv(heads, ROPE_HEADS), 1),
        flat,
        out,
        weight if weight is not None else flat,
        cos,
        sin,
        seq,
        heads,
        stride_token,
        stride_channel,
        float(eps),
        *(c[k] for k in ROPE_CONSTEXPRS),
    )
    return out.view(batch, seq, heads, head_dim)


def rope_tables(value: Any, *, interleaved: bool, rotary: int) -> tuple[Any, Any, int, int]:
    """A model's rotary operand as the kernel reads it: `(cos, sin, token stride, column
    stride)` in fp32. Interleaved: complex `[seq, rotary/2]`, or a `(cos, sin)` pair of
    `[seq, rotary/2]` or of `[seq, rotary]` with each value repeated for its pair (Diffusers'
    `repeat_interleave_real`); split-half: a `(cos, sin)` pair of `[seq, rotary]`."""
    if isinstance(value, torch.Tensor) and value.is_complex():
        pairs = torch.view_as_real(value.to(torch.complex64))
        pairs = pairs if pairs.stride(-1) == 1 and pairs.stride(-2) == 2 else pairs.contiguous()
        return pairs[..., 0], pairs[..., 1], int(pairs.stride(-3)), 2
    cos, sin = (t.float().contiguous() for t in value)
    if cos.shape != sin.shape:
        raise ValueError("rotary cos and sin differ in shape")
    if not interleaved and int(cos.shape[-1]) != rotary:
        raise ValueError(f"split-half rotary tables are {cos.shape[-1]} wide, not {rotary}")
    repeated = interleaved and int(cos.shape[-1]) == rotary
    return cos, sin, int(cos.stride(-2)), 2 if repeated else 1


@dataclass(frozen=True, slots=True)
class Entries:
    """The entries one declared block layout launches."""

    first: TritonEntry
    """The first norm [, modulated]."""
    residual: TritonEntry
    """The last gated residual: the MLP's, or a parallel block's attention + MLP."""
    swiglu: TritonEntry
    second: TritonEntry | None
    """Sequential: the attention's gated residual, then the second norm [, modulated]."""
    rope: TritonEntry | None
    variants: tuple[TritonEntry, ...] = ()
    """The same entries for the other operand dtype a launch may pass (`X16`, `X32`)."""

    def all(self) -> tuple[TritonEntry, ...]:
        found = (self.first, self.second, self.residual, self.swiglu, self.rope, *self.variants)
        return tuple(e for e in found if e is not None)


def entries(
    width: int,
    head_dim: int,
    rotary: int | None,
    *,
    norm: int,
    weight: bool,
    once: bool,
    modulate: int,
    tanh: bool,
    f16: bool,
    interleaved: bool,
    round: int,
    product: bool,
    qk_weight: bool,
    parallel: bool,
    composed: bool,
) -> Entries:
    """A layout's entries. fp16 rows: x may be fp16 (the stream's first block) or fp32, and an
    attention module's q/k norm operand fp16 or fp32 (a processor that normalizes in fp32)."""
    block = _next_pow2(width)
    row, wide = ("*fp16", "*fp32") if f16 else ("*bf16", "*bf16")

    def rows(x16: bool) -> list[TritonEntry]:
        def entry(norm: int, branches: int) -> TritonEntry:
            normed = norm != NONE
            constexprs = {
                "NORM": norm,
                "WEIGHT": int(weight and normed),
                "ONCE": int(once and weight and normed),
                "MODULATE": modulate if normed else 0,
                "BRANCHES": branches,
                "TANH": int(tanh and branches > 0),
                "F16": int(f16),
                "X16": int(x16),
                "BLOCK": block,
            }
            # Row operands every launch aligns (`_dense`); weights and per-token indexes may
            # not be. The row reduction's order follows the warps: wide rows take more.
            aligned = ("x_ptr", "branch_ptr", "other_ptr", "gate_ptr", "hidden_ptr", "out_ptr")
            return _entry(
                norm_modulate_kernel,
                constexprs,
                16 if block >= 8192 else 8,
                (*aligned, "scale_ptr", "shift_ptr", "n_cols", "stride_table"),
                x_ptr="*fp32" if f16 and not x16 else row,
                branch_ptr=row,
                other_ptr=row,
                out_ptr=row if normed else wide,
                **dict.fromkeys(
                    ("gate_ptr", "hidden_ptr", "weight_ptr", "scale_ptr", "shift_ptr"), wide
                ),
                index_ptr="*i64",
                eps="fp32",
            )

        return [entry(norm, 0), entry(NONE, 2 if parallel else 1)] + (
            [] if parallel else [entry(norm, 1)]
        )

    def rope(x32: bool) -> TritonEntry | None:
        if rotary is None:
            return None
        constexprs = {
            "HEAD_DIM": head_dim,
            "ROTARY": rotary,
            "HEADS": ROPE_HEADS,
            "INTERLEAVED": int(interleaved),
            "WEIGHT": int(qk_weight),
            "ONCE": int(once and qk_weight),
            "ROUND": round,
            "PRODUCT": int(product),
            "X32": int(x32),
        }
        return _entry(
            head_norm_rope_kernel,
            constexprs,
            _ROPE_WARPS,
            ("x_ptr", "out_ptr"),
            x_ptr="*fp32" if x32 else row,
            out_ptr=row,
            weight_ptr=wide,
            cos_ptr="*fp32",
            sin_ptr="*fp32",
            eps="fp32",
        )

    swiglu = ("gate_ptr", "up_ptr", "out_ptr", "n_cols", "stride_gate", "stride_up", "stride_out")
    first, residual, *second = rows(False)
    variants = [*rows(True), *([] if composed else [rope(True)])] if f16 else []
    return Entries(
        first,
        residual,
        _entry(
            swiglu_kernel,
            {"F16": int(f16), "BLOCK": SWIGLU_BLOCK},
            _SWIGLU_WARPS,
            swiglu,
            **dict.fromkeys(swiglu[:3], row),
        ),
        second[0] if second else None,
        rope(False),
        tuple(e for e in variants if e is not None),
    )
