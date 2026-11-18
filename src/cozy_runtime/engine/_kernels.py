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
            tl.store(card_ptr + at + hd, tl.load(k_ptr + src))
            tl.store(card_ptr + at + T * KVH * D + hd, tl.load(v_ptr + src))

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
                x = tl.load(x_ptr + (row * S + t) * channels + c, mask=ok, other=0.0).to(tl.float32)
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
        g_ptr,
        beta_ptr,
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
    ):
        """The gated delta rule for a row's `S` tokens in turn, of one value head and `BV` value
        columns, from the state in the row's slot (`[heads, DK, DV]` f32, a slot every
        `slot_stride`; -1: a row that isn't decoding). q and k are `[rows, S, k_heads, DK]` rows
        `qk_stride` apart (a value head reads key head `h // group`), v `[rows, S, heads, DV]`, g
        and beta `[rows, S, heads]`; q and k L2-normalised (q then divided by `scale`), the state
        decayed by exp(g), corrected toward v by beta, then read by q, as `lm`'s kernel computes
        it. One token: the state goes back to the slot. A span: the state after each token goes
        to the row's scratch for the step to keep one."""
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
            for t in tl.static_range(S):
                at = row * S + t
                q = tl.load(q_ptr + at * qk_stride + kh * DK + rk).to(tl.float32)
                k = tl.load(k_ptr + at * qk_stride + kh * DK + rk).to(tl.float32)
                q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) / scale
                k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
                v = tl.load(v_ptr + at * v_stride + h * DV + rv).to(tl.float32)
                g = tl.load(g_ptr + at * heads + h)
                beta = tl.load(beta_ptr + at * heads + h).to(tl.float32)
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
