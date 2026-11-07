"""Kernels for a gated-delta-net layer's causal conv step and its recurrence, one launch where
Transformers' reference path (when `causal_conv1d` and `flash-linear-attention` are absent) runs
a few dozen small ops per layer per token; a verify pass's few drafted tokens take one launch
for all.

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
    def conv_steps_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        weight_ptr,
        out_ptr,
        channels,
        tokens,
        x_stride_b,
        x_stride_c,
        out_stride_b,
        out_stride_t,
        CONV: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """`conv_step_kernel` for each of `tokens` tokens after a `CONV`-wide window, x being the
        window then the tokens (`[B, C, CONV + T]`, contiguous over T): the same arithmetic per
        token, one launch for all, the window left as it was."""
        b = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = c < channels
        row = x_ptr + b * x_stride_b + c * x_stride_c
        for t in range(tokens):
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for j in tl.static_range(CONV):
                held = tl.load(row + t + 1 + j, mask=ok, other=0.0)
                weight = tl.load(weight_ptr + c * CONV + j, mask=ok, other=0.0)
                acc += weight.to(tl.float32) * held.to(tl.float32)
            y = acc.to(tl.bfloat16).to(tl.float32)
            at = out_ptr + b * out_stride_b + t * out_stride_t + c
            tl.store(at, (y * tl.sigmoid(y)).to(tl.bfloat16), mask=ok)

    @triton.jit  # type: ignore[untyped-decorator]
    def recurrent_steps_kernel(  # type: ignore[no-untyped-def]
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        state_ptr,
        states_ptr,
        out_ptr,
        heads,
        tokens,
        q_stride_b,
        q_stride_t,
        q_stride_h,
        k_stride_b,
        k_stride_t,
        k_stride_h,
        v_stride_b,
        v_stride_t,
        v_stride_h,
        scale,
        DK: tl.constexpr,
        DV: tl.constexpr,
        BV: tl.constexpr,
    ):
        """The gated delta rule for `tokens` tokens in turn, of one head and `BV` value columns:
        q and k L2-normalised (q then divided by `scale`), the state decayed by exp(g), corrected
        toward v by beta, then read by q. The state after each token goes to `states`
        (`[T, B, H, DK, DV]`; `state` itself for one decoded token). g and beta are `[B, T, H]`,
        out `[B, T, H, DV]`. Decoding and verifying both run it, so a token's bits do not depend
        on how many come with it."""
        bh = tl.program_id(0)
        b = bh // heads
        h = bh % heads
        rk = tl.arange(0, DK)
        rv = tl.program_id(1) * BV + tl.arange(0, BV)
        where = (bh * DK + rk[:, None]) * DV + rv[None, :]
        state = tl.load(state_ptr + where)
        every = tl.num_programs(0) * DK * DV
        for t in range(tokens):
            q = tl.load(q_ptr + b * q_stride_b + t * q_stride_t + h * q_stride_h + rk)
            k = tl.load(k_ptr + b * k_stride_b + t * k_stride_t + h * k_stride_h + rk)
            q = q.to(tl.float32)
            k = k.to(tl.float32)
            q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) / scale
            k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
            v = tl.load(v_ptr + b * v_stride_b + t * v_stride_t + h * v_stride_h + rv)
            v = v.to(tl.float32)
            at = (b * tokens + t) * heads + h
            g = tl.load(g_ptr + at)
            beta = tl.load(beta_ptr + at).to(tl.float32)
            state = state * tl.exp(g)
            delta = (v - tl.sum(state * k[:, None], axis=0)) * beta
            state = state + k[:, None] * delta[None, :]
            tl.store(states_ptr + t * every + where, state)
            tl.store(out_ptr + at * DV + rv, tl.sum(state * q[:, None], axis=0).to(tl.bfloat16))


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
    conv_steps = TritonEntry(
        conv_steps_kernel,
        _signature(conv_steps_kernel),
        {"CONV": CONV, "BLOCK": CONV_BLOCK},
        num_warps=4,
    )
    recurrent_steps = TritonEntry(
        recurrent_steps_kernel,
        _signature(
            recurrent_steps_kernel,
            g_ptr="*fp32",
            state_ptr="*fp32",
            states_ptr="*fp32",
            scale="fp32",
        ),
        {"DK": KEY_DIM, "DV": VALUE_DIM, "BV": VALUE_BLOCK},
        num_warps=4,
    )
    return [conv, conv_steps, recurrent_steps]
