"""The paged decode attention's kernel (cr-336, bring-up form): one launch a layer reads a row's
K and V through its block table, from the card or from blocks off it, and leaves each part of the
row's softmax for `attention.paged_attention` to join.

It reads every length and table entry from device buffers, so one compiled kernel serves any
context length. Which tier a block is on, and where its bytes are, is decided in one place
(`resolve` below, marked): entries below `card_blocks` are on the card, block-major across layers;
the rest are read from `host_ptr`, layer-major, whatever backs it (host memory the device
addresses, or a window on the card the blocks were copied to).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError as _absent:  # pragma: no cover - a CPU-only or non-Linux environment
    if TYPE_CHECKING:
        raise
    TRITON_ABSENT = f"triton does not import ({_absent})"  # type: ignore[unreachable]  # absent Triton
else:
    TRITON_ABSENT = ""

if not TRITON_ABSENT:

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def paged_partial_kernel(  # type: ignore[no-untyped-def]
        q_ptr,
        card_ptr,
        host_ptr,
        table_ptr,
        lengths_ptr,
        part_m_ptr,
        part_l_ptr,
        part_acc_ptr,
        heads,
        max_blocks,
        card_blocks,
        layer,
        host_layer,
        host_blocks,
        scale,
        L: tl.constexpr,
        T: tl.constexpr,
        KVH: tl.constexpr,
        G: tl.constexpr,
        D: tl.constexpr,
        P: tl.constexpr,
    ):
        """One part (`P` a row and K/V head) of one query a head over a row's cached tokens: the
        `G` query heads that share K/V head `kvh` read each block once. Per part it leaves the
        running maximum, the sum of exp(score - maximum) and that-weighted sum of V
        (`part_m`, `part_l` `[rows * KVH, P, G]`, `part_acc` `[rows * KVH, P, G, D]`, f32). q is
        `[rows, heads, D]`; the card tier `[block][layer][K|V][T][KVH][D]`; the host tier
        `[layer][block][K|V][T][KVH][D]` with `host_blocks` blocks a layer."""
        rk = tl.program_id(0)
        p = tl.program_id(1)
        row = rk // KVH
        kvh = rk % KVH
        g = tl.arange(0, G)
        t = tl.arange(0, T)
        d = tl.arange(0, D)
        length = tl.load(lengths_ptr + row)
        blocks = (length + T - 1) // T
        per = (blocks + P - 1) // P
        first = p * per
        count = tl.maximum(tl.minimum(first + per, blocks) - first, 0)
        q = tl.load(q_ptr + ((row * heads + kvh * G + g)[:, None]) * D + d[None, :])
        q = q.to(tl.float32) * scale
        inner = (t[:, None] * KVH + kvh) * D + d[None, :]
        every = inner >= 0
        tile = T * KVH * D
        m = tl.full([G], float("-inf"), tl.float32)
        l = tl.zeros([G], dtype=tl.float32)
        acc = tl.zeros([G, D], dtype=tl.float32)
        for j in range(count):
            i = first + j
            # resolve: the one place that knows where block `entry` of this layer is read from.
            entry = tl.load(table_ptr + row * max_blocks + i).to(tl.int64)
            on_host = entry >= card_blocks
            card_at = (entry * L + layer) * 2 * tile
            host_at = (host_layer * host_blocks + (entry - card_blocks)) * 2 * tile
            k_card = tl.load(card_ptr + card_at + inner, mask=every & (entry < card_blocks), other=0.0)
            k_host = tl.load(host_ptr + host_at + inner, mask=every & on_host, other=0.0)
            v_card = tl.load(card_ptr + card_at + tile + inner, mask=every & (entry < card_blocks), other=0.0)
            v_host = tl.load(host_ptr + host_at + tile + inner, mask=every & on_host, other=0.0)
            k = tl.where(on_host, k_host, k_card).to(tl.float32)
            v = tl.where(on_host, v_host, v_card).to(tl.float32)
            s = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
            s = tl.where(((i * T + t) < length)[None, :], s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            w = tl.exp(s - m_safe[:, None])
            alpha = tl.exp(m - m_safe)
            acc = acc * alpha[:, None] + tl.sum(w[:, :, None] * v[None, :, :], axis=1)
            l = l * alpha + tl.sum(w, axis=1)
            m = m_new
        at = (rk * P + p) * G + g
        tl.store(part_m_ptr + at, m)
        tl.store(part_l_ptr + at, l)
        tl.store(part_acc_ptr + at[:, None] * D + d[None, :], acc)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def write_step_kernel(  # type: ignore[no-untyped-def]
        k_ptr,
        v_ptr,
        card_ptr,
        slots_ptr,
        lengths_ptr,
        layer,
        slot_stride,
        L: tl.constexpr,
        S: tl.constexpr,
        T: tl.constexpr,
        KVH: tl.constexpr,
        D: tl.constexpr,
    ):
        """A step's own K and V (`[rows, S, KVH, D]`) for `layer`, into the card tier at each
        query's token slot (`slot_mapping`, a row every `slot_stride`: block entry * T + offset). A row that
        isn't decoding (length -1) or a query with slot -1 writes nothing. A running row's tail
        blocks are on the card (cr-336), so no entry here is off it."""
        row = tl.program_id(0)
        s = tl.program_id(1)
        slot = tl.load(slots_ptr + row * slot_stride + s)
        if (tl.load(lengths_ptr + row) >= 0) & (slot >= 0):
            entry = slot // T
            at = (entry * L + layer) * 2 * T * KVH * D + (slot % T) * KVH * D
            hd = tl.arange(0, KVH * D)
            src = (row * S + s) * KVH * D + hd
            dtype = card_ptr.dtype.element_ty  # an f32 product is rounded to the cache's dtype here
            tl.store(card_ptr + at + hd, tl.load(k_ptr + src).to(dtype))
            tl.store(card_ptr + at + T * KVH * D + hd, tl.load(v_ptr + src).to(dtype))

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def paged_step_kernel(  # type: ignore[no-untyped-def]
        q_ptr,
        card_ptr,
        host_ptr,
        table_ptr,
        lengths_ptr,
        part_m_ptr,
        part_l_ptr,
        part_acc_ptr,
        heads,
        max_blocks,
        card_blocks,
        layer,
        host_layer,
        host_blocks,
        first_seen,
        scale,
        L: tl.constexpr,
        S: tl.constexpr,
        T: tl.constexpr,
        KVH: tl.constexpr,
        G: tl.constexpr,
        Q: tl.constexpr,
        D: tl.constexpr,
        P: tl.constexpr,
    ):
        """As `paged_partial_kernel`, for a step of `S` queries a row (its token and its drafts),
        causal within the span: query `s` sees positions `[first_seen, length + s]`, where
        `length` is where the step's first query sits and its own K/V is already written (the
        draft head's layer has no K/V at position 0: it sees from 1). The `S * G`
        query heads sharing K/V head `kvh` read each block once (`Q` is that, rounded up to a
        power of two). q is `[rows, S, heads, D]`; parts `[rows * KVH, P, S * G]`."""
        rk = tl.program_id(0)
        p = tl.program_id(1)
        row = rk // KVH
        kvh = rk % KVH
        qi = tl.arange(0, Q)
        live = qi < S * G
        s_of = qi // G
        g_of = qi % G
        t = tl.arange(0, T)
        d = tl.arange(0, D)
        length = tl.load(lengths_ptr + row)
        reach = tl.maximum(length + S, 0)  # a row not decoding (-1) reads nothing
        blocks = tl.where(length >= 0, (reach + T - 1) // T, 0)
        per = (blocks + P - 1) // P
        first = p * per
        count = tl.maximum(tl.minimum(first + per, blocks) - first, 0)
        q_at = ((row * S + s_of) * heads + kvh * G + g_of)[:, None] * D + d[None, :]
        q = tl.load(q_ptr + q_at, mask=live[:, None], other=0.0).to(tl.float32) * scale
        sees = length + s_of  # the last position each query sees
        inner = (t[:, None] * KVH + kvh) * D + d[None, :]
        every = inner >= 0
        tile = T * KVH * D
        m = tl.full([Q], float("-inf"), tl.float32)
        l = tl.zeros([Q], dtype=tl.float32)
        acc = tl.zeros([Q, D], dtype=tl.float32)
        for j in range(count):
            i = first + j
            # resolve: the one place that knows where block `entry` of this layer is read from.
            entry = tl.load(table_ptr + row * max_blocks + i).to(tl.int64)
            on_card = entry < card_blocks
            card_at = (entry * L + layer) * 2 * tile
            host_at = (host_layer * host_blocks + (entry - card_blocks)) * 2 * tile
            k_card = tl.load(card_ptr + card_at + inner, mask=every & on_card, other=0.0)
            k_host = tl.load(host_ptr + host_at + inner, mask=every & ~on_card, other=0.0)
            v_card = tl.load(card_ptr + card_at + tile + inner, mask=every & on_card, other=0.0)
            v_host = tl.load(host_ptr + host_at + tile + inner, mask=every & ~on_card, other=0.0)
            k = tl.where(on_card, k_card, k_host).to(tl.float32)
            v = tl.where(on_card, v_card, v_host).to(tl.float32)
            sc = tl.sum(q[:, None, :] * k[None, :, :], axis=2)
            pos = (i * T + t)[None, :]
            seen = (pos >= first_seen) & (pos <= sees[:, None])
            sc = tl.where(seen & live[:, None], sc, float("-inf"))
            m_new = tl.maximum(m, tl.max(sc, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            w = tl.exp(sc - m_safe[:, None])
            alpha = tl.exp(m - m_safe)
            acc = acc * alpha[:, None] + tl.sum(w[:, :, None] * v[None, :, :], axis=1)
            l = l * alpha + tl.sum(w, axis=1)
            m = m_new
        at = (rk * P + p) * (S * G) + qi
        tl.store(part_m_ptr + at, m, mask=live)
        tl.store(part_l_ptr + at, l, mask=live)
        tl.store(part_acc_ptr + at[:, None] * D + d[None, :], acc, mask=live[:, None])

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def conv_span_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        weight_ptr,
        slots_ptr,
        state_ptr,
        scratch_ptr,
        out_ptr,
        channels,
        slot_stride,
        scratch_stride,
        scratch_step,
        S: tl.constexpr,
        W: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """A gated-delta layer's causal conv and SiLU for a row's `S` tokens (`x`, `[rows, S, C]`),
        from the `W` inputs before them held in the row's state slot (`state`, `[C, W]` a slot
        every `slot_stride`, the slot `slots[row]`; -1: a row that isn't decoding). One token:
        its window goes back to the slot. A span: the window after each token goes to the row's
        scratch (`scratch_stride` a row, `scratch_step` a token) for the step to keep one. The dot
        rounds to bf16 (the conv's output) and SiLU rounds again, as `lm`'s kernels do."""
        row = tl.program_id(0)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = c < channels
        slot = tl.load(slots_ptr + row)
        if slot >= 0:
            window = state_ptr + slot.to(tl.int64) * slot_stride + c * W
            w0 = tl.load(window + 0, mask=ok, other=0.0).to(tl.float32)
            w1 = tl.load(window + 1, mask=ok, other=0.0).to(tl.float32)
            w2 = tl.load(window + 2, mask=ok, other=0.0).to(tl.float32)
            k0 = tl.load(weight_ptr + c * (W + 1) + 0, mask=ok, other=0.0).to(tl.float32)
            k1 = tl.load(weight_ptr + c * (W + 1) + 1, mask=ok, other=0.0).to(tl.float32)
            k2 = tl.load(weight_ptr + c * (W + 1) + 2, mask=ok, other=0.0).to(tl.float32)
            k3 = tl.load(weight_ptr + c * (W + 1) + 3, mask=ok, other=0.0).to(tl.float32)
            for t in tl.static_range(S):
                x = tl.load(x_ptr + (row * S + t) * channels + c, mask=ok, other=0.0)
                x = x.to(out_ptr.dtype.element_ty).to(tl.float32)  # an f32 product, rounded as the model's linear
                y = (k0 * w0 + k1 * w1 + k2 * w2 + k3 * x).to(tl.bfloat16).to(tl.float32)
                tl.store(out_ptr + (row * S + t) * channels + c, (y * tl.sigmoid(y)).to(tl.bfloat16), mask=ok)
                w0, w1, w2 = w1, w2, x
                if S > 1:
                    kept = scratch_ptr + row * scratch_stride + t * scratch_step + c * W
                    tl.store(kept + 0, w0.to(tl.bfloat16), mask=ok)
                    tl.store(kept + 1, w1.to(tl.bfloat16), mask=ok)
                    tl.store(kept + 2, w2.to(tl.bfloat16), mask=ok)
            if S == 1:
                tl.store(window + 0, w0.to(tl.bfloat16), mask=ok)
                tl.store(window + 1, w1.to(tl.bfloat16), mask=ok)
                tl.store(window + 2, w2.to(tl.bfloat16), mask=ok)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def delta_span_kernel(  # type: ignore[no-untyped-def]
        q_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        a_log_ptr,
        dt_bias_ptr,
        slots_ptr,
        state_ptr,
        scratch_ptr,
        out_ptr,
        heads,
        group,
        qk_stride,
        v_stride,
        slot_stride,
        scratch_stride,
        scratch_step,
        scale,
        S: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        BV: tl.constexpr,
        GIVEN: tl.constexpr,
    ):
        """The gated delta rule for a row's `S` tokens in turn, of one value head and `BV` value
        columns, from the state in the row's slot (`[heads, DK, DV]` f32, a slot every
        `slot_stride`; -1: a row that isn't decoding). q and k are `[rows, S, k_heads, DK]` rows
        `qk_stride` apart (a value head reads key head `h // group`), v `[rows, S, heads, DV]`, a
        and b `[rows, S, heads]` (the layer's `in_proj_a` and `in_proj_b`); q and k L2-normalised
        (q then divided by `scale`), the state decayed by exp(g), corrected toward v by beta, then
        read by q, as `lm`'s kernel computes it. g and beta are Transformers' own, computed here:
        beta = sigmoid(b) rounded to b's dtype, g = -exp(A_log) * softplus(a + dt_bias) in f32.
        `GIVEN`: g (f32) and beta are given in a's and b's places, computed by torch (the reference).
        One token: the state goes back to the slot. A span: the state after each token goes to
        the row's scratch for the step to keep one."""
        bh = tl.program_id(0)
        row = bh // heads
        h = bh % heads
        kh = h // group
        rk = tl.arange(0, DK)
        rv = tl.program_id(1) * BV + tl.arange(0, BV)
        where = (h * DK + rk[:, None]) * DV + rv[None, :]
        slot = tl.load(slots_ptr + row)
        if slot >= 0:
            home = state_ptr + slot.to(tl.int64) * slot_stride
            state = tl.load(home + where)
            decay = -tl.exp(tl.load(a_log_ptr + h).to(tl.float32))
            bias = tl.load(dt_bias_ptr + h).to(tl.float32)
            for t in tl.static_range(S):
                at = row * S + t
                q = tl.load(q_ptr + at * qk_stride + kh * DK + rk).to(tl.float32)
                k = tl.load(k_ptr + at * qk_stride + kh * DK + rk).to(tl.float32)
                q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) / scale
                k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
                v = tl.load(v_ptr + at * v_stride + h * DV + rv).to(tl.float32)
                if GIVEN:
                    g = tl.load(a_ptr + at * heads + h).to(tl.float32)
                    beta = tl.load(b_ptr + at * heads + h).to(tl.float32)
                else:
                    dtype = out_ptr.dtype.element_ty  # a and b rounded to it first, as the model's linears
                    raw = tl.load(b_ptr + at * heads + h).to(dtype)
                    beta = tl.sigmoid(raw.to(tl.float32)).to(dtype).to(tl.float32)
                    a = tl.load(a_ptr + at * heads + h).to(dtype).to(tl.float32) + bias
                    g = decay * tl.where(a > 20.0, a, libdevice.log1p(tl.exp(a)))  # softplus, torch's threshold
                state = state * tl.exp(g)
                delta = (v - tl.sum(state * k[:, None], axis=0)) * beta
                state = state + k[:, None] * delta[None, :]
                if S > 1:
                    tl.store(scratch_ptr + row * scratch_stride + t * scratch_step + where, state)
                tl.store(out_ptr + (at * heads + h) * DV + rv, tl.sum(state * q[:, None], axis=0).to(tl.bfloat16))
            if S == 1:
                tl.store(home + where, state)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def keep_kernel(  # type: ignore[no-untyped-def]
        scratch_ptr,
        slots_ptr,
        kept_ptr,
        state_ptr,
        words,
        slot_stride,
        scratch_stride,
        scratch_step,
        BLOCK: tl.constexpr,
    ):
        """Each decoding row's kept candidate (`kept[row]`: the state after the last token the
        step keeps, an index into its span) from its scratch into its slot, `words` int32 words
        of it (both passed as bytes, every offset a whole word); a row that isn't decoding (slot -1) or kept nothing (-1) is left as it is."""
        row = tl.program_id(0)
        w = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = w < words
        slot = tl.load(slots_ptr + row)
        kept = tl.load(kept_ptr + row)
        if (slot >= 0) & (kept >= 0):
            words_in = scratch_ptr.to(tl.pointer_type(tl.int32))
            words_out = state_ptr.to(tl.pointer_type(tl.int32))
            src = words_in + row * scratch_stride + kept.to(tl.int64) * scratch_step
            tl.store(words_out + slot.to(tl.int64) * slot_stride + w, tl.load(src + w, mask=ok), mask=ok)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def paged_join_kernel(  # type: ignore[no-untyped-def]
        part_m_ptr,
        part_l_ptr,
        part_acc_ptr,
        gate_ptr,
        out_ptr,
        heads,
        gate_stride,
        gate_head,
        S: tl.constexpr,
        KVH: tl.constexpr,
        G: tl.constexpr,
        D: tl.constexpr,
        P: tl.constexpr,
        GATED: tl.constexpr,
    ):
        """`paged_step_kernel`'s parts of one query head joined (each weighted by exp(its maximum
        - the row's)), rounded to the output's dtype as attention's output is; then, `GATED`, times
        the sigmoid of the head's gate (`[rows, S, heads, D]`, a query every `gate_stride`, a head
        every `gate_head`), each
        rounded as Transformers' Qwen3.5 rounds them. out is `[rows, S, heads, D]`."""
        rk = tl.program_id(0)
        qi = tl.program_id(1)
        row = rk // KVH
        kvh = rk % KVH
        s = qi // G
        h = kvh * G + qi % G
        p = tl.arange(0, P)
        d = tl.arange(0, D)
        at = (rk * P + p) * (S * G) + qi
        m = tl.load(part_m_ptr + at)
        best = tl.max(m, axis=0)
        weight = tl.exp(m - tl.where(best == float("-inf"), 0.0, best))
        total = tl.maximum(tl.sum(tl.load(part_l_ptr + at) * weight, axis=0), 1e-30)
        acc = tl.load(part_acc_ptr + at[:, None] * D + d[None, :])
        joined = tl.sum(acc * weight[:, None], axis=0) / total
        y = joined.to(out_ptr.dtype.element_ty)
        if GATED:
            raw = tl.load(gate_ptr + (row * S + s).to(tl.int64) * gate_stride + h * gate_head + d)
            raw = raw.to(out_ptr.dtype.element_ty)
            sig = tl.sigmoid(raw.to(tl.float32)).to(raw.dtype)
            y = (y.to(tl.float32) * sig.to(tl.float32)).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + ((row * S + s).to(tl.int64) * heads + h) * D + d, y)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def _normed_rotated(src, w_ptr, pos, inv_ptr, eps, out_ptr, D: tl.constexpr, R: tl.constexpr):  # type: ignore[no-untyped-def]
        """`Qwen3_5RMSNorm` of one head (`D` at `src`, rounded to `out_ptr`'s dtype first: an f32
        product is the model's linear rounded), rounded to that dtype, then the split-half RoPE of
        its first `R` at text position `pos`: each product and the sum rounded to that dtype, cos
        and sin of position * inverse frequency rounded to it first, as Transformers'
        `apply_rotary_pos_emb` computes them in the model's dtype."""
        c = tl.arange(0, D)
        dtype = out_ptr.dtype.element_ty
        x = tl.load(src + c).to(dtype).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        n = (x * rms * (1.0 + tl.load(w_ptr + c).to(tl.float32))).to(dtype)
        rotary = c < R
        half = R // 2
        partner = tl.where(c < half, c + half, c - half)
        xp = tl.load(src + partner, mask=rotary, other=0.0).to(dtype).to(tl.float32)
        wp = tl.load(w_ptr + partner, mask=rotary, other=0.0).to(tl.float32)
        np_ = (xp * rms * (1.0 + wp)).to(dtype).to(tl.float32)
        turned = tl.where(c < half, -np_, np_)
        f = pos * tl.load(inv_ptr + c % half, mask=rotary, other=0.0)
        cos = libdevice.cos(f).to(dtype).to(tl.float32)
        sin = libdevice.sin(f).to(dtype).to(tl.float32)
        a = (n.to(tl.float32) * cos).to(dtype).to(tl.float32)
        b = (turned * sin).to(dtype).to(tl.float32)
        return tl.where(rotary, (a + b).to(dtype), n)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def qk_norm_rope_kernel(  # type: ignore[no-untyped-def]
        q_ptr,
        k_ptr,
        q_w_ptr,
        k_w_ptr,
        positions_ptr,
        inv_ptr,
        q_out_ptr,
        k_out_ptr,
        eps,
        heads,
        kv_heads,
        q_stride,
        D: tl.constexpr,
        R: tl.constexpr,
    ):
        """Qwen3.5 attention's q and k, one launch for both: each head's `q_norm` or `k_norm` and
        RoPE (`_normed_rotated`) for a row's query `program_id(0)` (`positions` its text position).
        q is read from `q_proj`'s output, a head's query in the first `D` of every `q_stride`
        (its gate after it); k from `k_proj`'s. Out: contiguous `[rows, S, heads | kv_heads, D]`."""
        rs = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1)
        c = tl.arange(0, D)
        pos = tl.load(positions_ptr + rs).to(tl.float32)
        if h < heads:
            y = _normed_rotated(q_ptr + (rs * heads + h) * q_stride, q_w_ptr, pos, inv_ptr, eps, q_out_ptr, D, R)
            tl.store(q_out_ptr + (rs * heads + h) * D + c, y)
        else:
            kh = h - heads
            y = _normed_rotated(k_ptr + (rs * kv_heads + kh) * D, k_w_ptr, pos, inv_ptr, eps, k_out_ptr, D, R)
            tl.store(k_out_ptr + (rs * kv_heads + kh) * D + c, y)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def silu_mul_kernel(  # type: ignore[no-untyped-def]
        gate_ptr,
        up_ptr,
        out_ptr,
        width,
        BLOCK: tl.constexpr,
    ):
        """An MLP's SiLU(gate) * up, one launch: gate and up rounded to the output's dtype, SiLU
        rounded to it, the product to it, as Transformers computes `act_fn(gate_proj(x)) * up_proj(x)`."""
        row = tl.program_id(0).to(tl.int64)
        c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        ok = c < width
        dtype = out_ptr.dtype.element_ty  # f32 products are rounded to it first, as the model's linears
        g = tl.load(gate_ptr + row * width + c, mask=ok, other=0.0).to(dtype).to(tl.float32)
        act = (g / (1.0 + tl.exp(-g))).to(dtype).to(tl.float32)
        up = tl.load(up_ptr + row * width + c, mask=ok, other=0.0).to(dtype).to(tl.float32)
        tl.store(out_ptr + row * width + c, (act * up).to(out_ptr.dtype.element_ty), mask=ok)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def add_norm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        h_ptr,
        w_ptr,
        sum_ptr,
        out_ptr,
        width,
        eps,
        BLOCK: tl.constexpr,
    ):
        """A residual add and the next `Qwen3_5RMSNorm`, one launch for every row: x + h rounded
        to x's dtype (the residual stream, into `sum`), then its norm as `rms_norm_kernel`'s."""
        row = tl.program_id(0).to(tl.int64)
        c = tl.arange(0, BLOCK)
        ok = c < width
        raw = tl.load(x_ptr + row * width + c, mask=ok, other=0.0)
        h = tl.load(h_ptr + row * width + c, mask=ok, other=0.0).to(raw.dtype)  # an f32 product, rounded
        y = (raw.to(tl.float32) + h.to(tl.float32)).to(raw.dtype)
        tl.store(sum_ptr + row * width + c, y, mask=ok)
        x = y.to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=0) / width + eps)
        w = tl.load(w_ptr + c, mask=ok, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * width + c, (x * rms * (1.0 + w)).to(out_ptr.dtype.element_ty), mask=ok)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def slots_kernel(  # type: ignore[no-untyped-def]
        start_ptr,
        live_ptr,
        table_ptr,
        out_ptr,
        offset,
        max_blocks,
        out_stride,
        T: tl.constexpr,
        COUNT: tl.constexpr,
        WIDTH: tl.constexpr,
    ):
        """Each row's token slot (block entry * T + offset in the block) at positions `start[row] +
        offset + [0, COUNT)`, into `out` (a row every `out_stride`): -1 where the row isn't live or
        the block isn't assigned. `WIDTH` is COUNT rounded up to a power of two."""
        row = tl.program_id(0)
        i = tl.arange(0, WIDTH)
        ok = i < COUNT
        at = (tl.load(start_ptr + row) + offset + i).to(tl.int64)
        index = tl.minimum(tl.maximum(at // T, 0), max_blocks - 1)
        entry = tl.load(table_ptr + row * max_blocks + index, mask=ok, other=-1).to(tl.int64)
        live = tl.load(live_ptr + row) != 0
        slot = tl.where(live & (entry >= 0), entry * T + at % T, -1)
        tl.store(out_ptr + row * out_stride + i, slot, mask=ok)

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def gated_norm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        gate_ptr,
        w_ptr,
        out_ptr,
        eps,
        D: tl.constexpr,
        WEIGHT_ROUNDS: tl.constexpr,
    ):
        """`Qwen3_5RMSNormGated` for one row of `D`, one launch for every row: the norm in f32,
        rounded to x's dtype; times the weight (rounded again where the weight is x's dtype,
        `WEIGHT_ROUNDS`); times SiLU of the gate in f32; rounded once to the output's dtype."""
        row = tl.program_id(0).to(tl.int64)
        c = tl.arange(0, D)
        raw = tl.load(x_ptr + row * D + c)
        x = raw.to(tl.float32)
        y = (x * tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)).to(raw.dtype).to(tl.float32)
        y = tl.load(w_ptr + c).to(tl.float32) * y
        if WEIGHT_ROUNDS:
            y = y.to(raw.dtype).to(tl.float32)
        z = tl.load(gate_ptr + row * D + c).to(raw.dtype).to(tl.float32)  # an f32 product, rounded
        tl.store(out_ptr + row * D + c, (y * (z * tl.sigmoid(z))).to(out_ptr.dtype.element_ty))

    @triton.jit  # type: ignore[untyped-decorator]  # Triton is unannotated
    def rms_norm_kernel(  # type: ignore[no-untyped-def]
        x_ptr,
        w_ptr,
        out_ptr,
        width,
        eps,
        BLOCK: tl.constexpr,
    ):
        """`Qwen3_5RMSNorm` for one row of `width`, one launch for every row: in f32, times 1 + the
        weight, rounded once to the output's dtype, as Transformers rounds it."""
        row = tl.program_id(0)
        c = tl.arange(0, BLOCK)
        ok = c < width
        x = tl.load(x_ptr + row * width + c, mask=ok, other=0.0).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=0) / width + eps)
        w = tl.load(w_ptr + c, mask=ok, other=0.0).to(tl.float32)
        tl.store(out_ptr + row * width + c, (x * rms * (1.0 + w)).to(out_ptr.dtype.element_ty), mask=ok)
