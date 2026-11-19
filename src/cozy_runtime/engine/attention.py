"""Paged decode attention (cr-336): a step's queries over each row's K/V, read through its block
table (`paged_step`: the step's own K/V written through `slot_mapping` first, the verify span
causal within it, a sliding layer's window). `register` serves it as a Transformers attention
implementation (`NAME`), for a caller that runs a model's own forward over the engine's K/V
(`lm`): the step it sets in `PAGED` attends.

Where a block off the card is read from is the caller's `HostTier`, not the kernel's: host memory
the device addresses in place, or a window on the card the blocks were copied to. How those are
served is open pending measurement (cr-336's tracker file); nothing else here knows which.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from cozy_runtime.engine import _kernels
from cozy_runtime.engine.launch import JIT, Run

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

    from cozy_runtime.engine.kv import KvLayout

#: The attention implementation `register` adds to Transformers' `AttentionInterface`.
NAME = "cozy_paged"


class Paged(Protocol):
    """A step's attention over the engine's K/V, as `register`'s implementation calls it: `query`
    `[rows, heads, S, head_dim]`, the step's own `key` and `value` `[rows, kv_heads, S, head_dim]`
    (a cache layer of the caller's returns them as given); returns `[rows, S, heads, head_dim]`."""

    def __call__(
        self, module: Module, query: Tensor, key: Tensor, value: Tensor, scaling: float
    ) -> Tensor: ...


#: The step a model's forward attends through while its config names `NAME` (the caller's).
PAGED: ContextVar[Paged | None] = ContextVar("cozy_paged_step", default=None)


def register() -> str:
    """`NAME` in Transformers' `AttentionInterface`, once a process, and returns it: a model whose
    config names it attends through `PAGED`'s step. Transformers makes no mask for an
    implementation it has no mask function for, and this one needs none (lengths are device data)."""
    from transformers import AttentionInterface

    if NAME not in AttentionInterface._global_mapping:
        AttentionInterface.register(NAME, _attend)
    return NAME


def _attend(  # Transformers' attention-function shape
    module: Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: object,
) -> tuple[Tensor, None]:
    step = PAGED.get()
    if step is None:
        raise RuntimeError(f"{NAME} attention ran outside a paged step")
    scale = float(scaling) if scaling is not None else float(query.shape[-1]) ** -0.5
    return step(module, query, key, value, scale), None


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
    once, a fixed number whatever its length; a row of no tokens attends to nothing, and its
    output is zero, as `paged_step`'s. Returns `[rows, heads, head_dim]`, q's dtype."""
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
    # Join the parts: each is weighted by exp(its maximum - the row's). An empty row's maximum is
    # -inf and its total 0: guarded as `paged_step`'s join is, so it is zero, never NaN.
    best = part_m.amax(1, keepdim=True)
    weight = torch.exp(part_m - torch.where(best.isinf(), 0.0, best))
    total = (part_l * weight).sum(1)
    out: Tensor = (part_acc * weight[..., None]).sum(1) / total.clamp(min=1e-30)[..., None]
    return out.view(rows, heads, dim).to(q.dtype)


def paged_step(
    q: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    card: Tensor,
    host: HostTier,
    block_table: Tensor,
    lengths: Tensor,
    slot_mapping: Tensor,
    layout: KvLayout,
    layer: int,
    parts: int,
    scale: float,
    first_seen: int = 0,
    gate: Tensor | None = None,
    window: int = 0,
    fused: bool = True,
    run: Run = JIT,
) -> Tensor:
    """A decode step's attention for `layer`: the step's own K and V (`k_new`, `v_new`,
    `[rows, S, kv_heads, head_dim]`) written at each query's token slot (`slot_mapping`'s first
    `S` columns), then each row's `S` queries (`q`, `[rows, S, heads, head_dim]`: its token and
    its drafts) attended over its cached tokens and the span, causal within it. `lengths` are the
    rows' tokens before this step (where its first query sits); a row at -1 writes and reads
    nothing (its output is zero). Positions below `first_seen` are never read, nor, with a `window`
    above 0 (a sliding layer's), a position `window` or more behind its query. `gate`, an output
    gate (`[rows, S, heads, head_dim]`, its last dimension contiguous; Qwen3.5's), multiplies each
    head's output by its sigmoid. `fused`: the parts joined (and gated) in one launch, else by
    torch's ops (the reference). `run` launches the kernels (the model's set, `engine.kernels`).
    Returns `[rows, S, heads, head_dim]` in q's dtype. Every length, slot and table entry is read
    on the device, so the step's graph holds only addresses."""
    rows, span, heads, dim = (int(n) for n in q.shape)
    kvh = layout.kv_heads
    group = heads // kvh
    tokens = layout.block_tokens
    run(_kernels.write_step_kernel, (rows, span),
        k_new.contiguous(), v_new.contiguous(), card, slot_mapping, lengths, layer,
        int(slot_mapping.stride(0)),
        layout.attn_layers, span, tokens, kvh, dim, num_warps=4,
    )  # fmt: skip
    fixed = (rows * kvh, parts, span * group)
    part_m = torch.empty(fixed, dtype=torch.float32, device=q.device)
    part_l = torch.empty(fixed, dtype=torch.float32, device=q.device)
    part_acc = torch.empty((*fixed, dim), dtype=torch.float32, device=q.device)
    run(_kernels.paged_step_kernel, (rows * kvh, parts),
        q.contiguous(), card, host.blocks, block_table, lengths, part_m, part_l, part_acc,
        heads, int(block_table.shape[1]), int(card.shape[0]), layer, host.layer,
        int(host.blocks.shape[1]), first_seen, window, scale,
        layout.attn_layers, span, tokens, kvh, group, _power_of_two(span * group), dim, parts, num_warps=4,
    )  # fmt: skip
    if not fused:
        best = part_m.amax(1, keepdim=True)
        weight = torch.exp(part_m - torch.where(best.isinf(), 0.0, best))
        total = (part_l * weight).sum(1)
        joined = (part_acc * weight[..., None]).sum(1) / total.clamp(min=1e-30)[..., None]
        # [rows * kvh, span * group, dim] -> [rows, span, kvh * group, dim]
        flat = joined.view(rows, kvh, span, group, dim).permute(0, 2, 1, 3, 4).reshape(rows, span, heads, dim).to(q.dtype)
        return flat if gate is None else flat * torch.sigmoid(gate)
    out = torch.empty((rows, span, heads, dim), dtype=q.dtype, device=q.device)
    if gate is not None and (gate.shape != out.shape or gate.stride(-1) != 1 or gate.stride(0) != span * gate.stride(1)):
        gate = gate.contiguous()
    run(_kernels.paged_join_kernel, (rows * kvh, span * group),
        part_m, part_l, part_acc, gate if gate is not None else out, out, heads,
        int(gate.stride(1)) if gate is not None else 0, int(gate.stride(2)) if gate is not None else 0,
        span, kvh, group, dim, parts, int(gate is not None), num_warps=4,
    )  # fmt: skip
    return out


def _power_of_two(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()
