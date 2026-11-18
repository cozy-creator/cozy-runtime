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
