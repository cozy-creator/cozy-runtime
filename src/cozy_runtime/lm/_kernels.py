"""One-token kernels for a gated-delta-net layer: its causal conv step and its recurrence, each one
launch where Transformers' reference path (when `causal_conv1d` and `flash-linear-attention` are
absent) runs a few dozen small ops per layer per token.

They follow the reference's arithmetic in f32 and round where it rounds, so they differ from it
only in reduction order.
"""

from __future__ import annotations

from typing import Any

from cozy_runtime.author.kernels import TritonEntry

try:
    import triton
    import triton.language as tl
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    TRITON_ABSENT = f"triton does not import ({_absent})"
else:
    TRITON_ABSENT = ""

#: The geometry these entries compile for: Qwen3-Next and Qwen3.5's (a layer of another shape
#: keeps Transformers' path).
CONV, CONV_BLOCK = 4, 256
KEY_DIM, VALUE_DIM, VALUE_BLOCK = 128, 128, 32

if not TRITON_ABSENT:

    @triton.jit  # type: ignore[untyped-decorator]
    def conv_step_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        state_ptr,
        weight_ptr,
        out_ptr,
        channels,
        x_stride,
        CONV: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """`causal_conv1d_update` for one token, SiLU, bf16 weights: the window shifts left by
        this token, the depthwise dot rounds to bf16 (the conv's output) and SiLU rounds again."""
        b = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = c < channels
        x = tl.load(x_ptr + b * x_stride + c, mask=ok, other=0.0)
        window = state_ptr + (b * channels + c) * CONV
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for j in tl.static_range(CONV - 1):
            held = tl.load(window + j + 1, mask=ok, other=0.0)
            weight = tl.load(weight_ptr + c * CONV + j, mask=ok, other=0.0)
            acc += weight.to(tl.float32) * held.to(tl.float32)
            tl.store(window + j, held, mask=ok)
        weight = tl.load(weight_ptr + c * CONV + CONV - 1, mask=ok, other=0.0)
        acc += weight.to(tl.float32) * x.to(tl.float32)
        tl.store(window + CONV - 1, x, mask=ok)
        y = acc.to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + b * channels + c, (y * tl.sigmoid(y)).to(tl.bfloat16), mask=ok)

    @triton.jit  # type: ignore[untyped-decorator]
    def recurrent_step_kernel(  # type: ignore[no-untyped-def]
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        state_ptr,
        out_ptr,
        heads,
        q_stride_b,
        q_stride_h,
        k_stride_b,
        k_stride_h,
        v_stride_b,
        v_stride_h,
        scale,
        DK: tl.constexpr,
        DV: tl.constexpr,
        BV: tl.constexpr,
    ):
        """The gated delta rule for one token of one head and `BV` value columns, the state
        updated in place: q and k L2-normalised (q then divided by `scale`), the state decayed
        by exp(g), corrected toward v by beta, then read by q."""
        bh = tl.program_id(0)
        b = bh // heads
        h = bh % heads
        rk = tl.arange(0, DK)
        rv = tl.program_id(1) * BV + tl.arange(0, BV)
        q = tl.load(q_ptr + b * q_stride_b + h * q_stride_h + rk).to(tl.float32)
        k = tl.load(k_ptr + b * k_stride_b + h * k_stride_h + rk).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) / scale
        k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        v = tl.load(v_ptr + b * v_stride_b + h * v_stride_h + rv).to(tl.float32)
        g = tl.load(g_ptr + bh)
        beta = tl.load(beta_ptr + bh).to(tl.float32)
        where = state_ptr + (bh * DK + rk[:, None]) * DV + rv[None, :]
        state = tl.load(where) * tl.exp(g)
        delta = (v - tl.sum(state * k[:, None], axis=0)) * beta
        state = state + k[:, None] * delta[None, :]
        tl.store(where, state)
        tl.store(out_ptr + bh * DV + rv, tl.sum(state * q[:, None], axis=0).to(tl.bfloat16))


def _signature(kernel: Any, **types: str) -> dict[str, str]:
    return {
        name: types.get(name)
        or ("constexpr" if name.isupper() else "*bf16" if name.endswith("_ptr") else "i32")
        for name in kernel.arg_names
    }


def entries() -> list[TritonEntry]:
    if TRITON_ABSENT:
        return []
    conv = TritonEntry(
        conv_step_kernel,
        _signature(conv_step_kernel),
        {"CONV": CONV, "BLOCK": CONV_BLOCK},
        num_warps=4,
    )
    recurrent = TritonEntry(
        recurrent_step_kernel,
        _signature(recurrent_step_kernel, g_ptr="*fp32", state_ptr="*fp32", scale="fp32"),
        {"DK": KEY_DIM, "DV": VALUE_DIM, "BV": VALUE_BLOCK},
        num_warps=4,
    )
    return [conv, recurrent]
