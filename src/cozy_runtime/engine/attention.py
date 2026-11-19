"""Paged decode attention (cr-336), bring-up form: one query a head over a row's K/V, read
through its block table. It attends only: writing the step's own K/V through `slot_mapping`, the
verify span and registration with Transformers' `AttentionInterface` come with the engine core.

Where a block off the card is read from is the caller's `HostTier`, not the kernel's: host memory
the device addresses in place, or a window on the card the blocks were copied to. How those are
served is open pending measurement (cr-336's tracker file); nothing else here knows which.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from cozy_runtime.engine import _kernels

if TYPE_CHECKING:
    from torch import Tensor

    from cozy_runtime.engine.kv import KvLayout


@dataclass(frozen=True)
class HostTier:
    """Where blocks at or above `card_blocks` are read from: `blocks` is
    `[layers, host_blocks, 2, block_tokens, kv_heads, head_dim]`, and `layer` the index of the
    attention layer in it (0 where `blocks` holds one layer: a window)."""

    blocks: Tensor
    layer: int


def paged_attention(
    q: Tensor,
    card: Tensor,
    host: HostTier,
    block_table: Tensor,
    lengths: Tensor,
    layout: KvLayout,
    layer: int,
    parts: int,
    scale: float,
) -> Tensor:
    """Each row's `q` (`[rows, heads, head_dim]`) attended over its cached tokens of `layer`:
    `card` is the card tier (`[card_blocks, layers, 2, block_tokens, kv_heads, head_dim]`),
    `block_table` and `lengths` the manager's device tables. A row is read in `parts` parts at
    once, a fixed number whatever its length. Returns `[rows, heads, head_dim]`, q's dtype."""
    rows, heads, dim = (int(n) for n in q.shape)
    kvh = layout.kv_heads
    group = heads // kvh
    fixed = (rows * kvh, parts, group)
    part_m = torch.empty(fixed, dtype=torch.float32, device=q.device)
    part_l = torch.empty(fixed, dtype=torch.float32, device=q.device)
    part_acc = torch.empty((*fixed, dim), dtype=torch.float32, device=q.device)
    _kernels.paged_partial_kernel[(rows * kvh, parts)](
        q, card, host.blocks, block_table, lengths, part_m, part_l, part_acc,
        heads, int(block_table.shape[1]), int(card.shape[0]), layer, host.layer,
        int(host.blocks.shape[1]), scale,
        L=layout.attn_layers, T=layout.block_tokens, KVH=kvh, G=group, D=dim, P=parts,
        num_warps=4,
    )  # fmt: skip
    # Join the parts: each is weighted by exp(its maximum - the row's).
    weight = torch.exp(part_m - part_m.amax(1, keepdim=True))
    total = (part_l * weight).sum(1)
    out: Tensor = (part_acc * weight[..., None]).sum(1) / total[..., None]
    return out.view(rows, heads, dim).to(q.dtype)
