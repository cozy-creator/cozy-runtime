"""Hand-ports of what Inductor decided, each installable on its own so it can be measured.

Every port is a Python-callable replacement installed AFTER the module tree is built.
Parameters, buffers, state-dict names and storage addresses are invariants, exactly as
`cozy_runtime.internal.anima_optimization` requires.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------- P1: no repeat_interleave


def install_no_repeat_interleave(model: Any) -> int:
    """Inductor emitted NO copy between the QK-norm/RoPE kernel and flash attention.

    Diffusers writes `key.repeat_interleave(query.size(3) // key.size(3), dim=3)`. Anima is
    MHA, not GQA: that ratio is 1, and `repeat_interleave(x, 1)` is still a full
    materializing copy (verified: new storage, contiguous output). Per self-attention it
    costs two 16 MiB copies AND rewrites Q/K/V out of the (B, S, H, D)-physical layout that
    SDPA's own output already uses.
    """
    from diffusers.models.transformers.transformer_cosmos import CosmosAttnProcessor2_0

    patched = 0
    for module in model.modules():
        processor = getattr(module, "processor", None)
        if type(processor) is CosmosAttnProcessor2_0:
            module.set_processor(_NoRepeatProcessor())
            patched += 1
        elif type(processor).__name__ == "_FusedSelfAttention":
            processor.skip_repeat_interleave = True
            patched += 1
    return patched


class _NoRepeatProcessor:
    """CosmosAttnProcessor2_0 with the ratio-1 repeat_interleave removed."""

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: Any = None,
    ) -> torch.Tensor:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.embeddings import apply_rotary_emb

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = attn.to_k(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = attn.to_v(encoder_hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
        if query.size(3) != key.size(3):
            raise ValueError("this port covers MHA only")
        hidden_states = dispatch_attention_fn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden_states))


# ------------------------------------------------------- P2: fused AdaLN-Zero residual kernel


@triton.jit
def _gate_add_adaln_kernel(
    resid_ptr,
    branch_ptr,
    mod_prev_ptr,
    mod_next_ptr,
    out_ptr,
    n_cols: tl.constexpr,
    eps,
    HAS_BRANCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One row: `h += gate*branch`, then `layer_norm(h)*(1+scale) + shift`.

    `mod_*` are the (3C,) chunked AdaLN vectors: shift at 0, scale at C, gate at 2C.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = row.to(tl.int64) * n_cols + cols

    h = tl.load(resid_ptr + base, mask=mask, other=0.0).to(tl.float32)
    if HAS_BRANCH:
        b = tl.load(branch_ptr + base, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(mod_prev_ptr + 2 * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        h = h + g * b
        tl.store(resid_ptr + base, h, mask=mask)

    mean = tl.sum(h, axis=0) / n_cols
    centered = tl.where(mask, h - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    shift = tl.load(mod_next_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(mod_next_ptr + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base, centered * rstd * (1.0 + scale) + shift, mask=mask)


def gate_add_adaln(
    resid: torch.Tensor,
    branch: torch.Tensor | None,
    mod_prev: torch.Tensor | None,
    mod_next: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    n_cols = resid.shape[-1]
    rows = resid.numel() // n_cols
    out = torch.empty_like(resid)
    block = triton.next_power_of_2(n_cols)
    _gate_add_adaln_kernel[(rows,)](
        resid,
        branch if branch is not None else resid,
        mod_prev if mod_prev is not None else mod_next,
        mod_next,
        out,
        n_cols,
        eps,
        HAS_BRANCH=branch is not None,
        BLOCK=block,
        num_warps=8,
    )
    return out


# --------------------------------------------- P3: one batched GEMM for every block's AdaLN LoRA


class _BatchedAdaLn:
    """Every AdaLN modulation vector for the whole model, in two GEMMs instead of 168.

    `embedded_timestep` and `temb` are the same tensor for all 3*L norms, so
    `linear_2(linear_1(silu(t))) + temb` differs only in the weights. Stacking
    `linear_1` into one (L*3*R, C) matrix and `linear_2` into one batched (L*3, 3C, R)
    turns 3*L*2 = 168 launch-bound batch-1 GEMMs into 2 launches. Inductor did NOT do this
    (it kept one extern mm per norm); the capture is what made the shape obvious.

    COST, stated: the stacked copies DUPLICATE those weights in VRAM — ~350 MiB on the real
    28-block model (88 MiB of `linear_1` plus 264 MiB of `linear_2`), about 9% on top of a
    3.9 GiB transformer. That is the price of measuring it as a post-fill patch. It is not
    the price of shipping it: the runtime owns the byte plane, so the fill could place these
    rows stacked in the first place and the duplicate disappears.
    """

    def __init__(self, model: Any) -> None:
        self.norms = [
            norm
            for block in model.transformer_blocks
            for norm in (block.norm1, block.norm2, block.norm3)
        ]
        self.w1 = torch.cat([n.linear_1.weight for n in self.norms], dim=0).contiguous()
        self.w2 = torch.stack([n.linear_2.weight for n in self.norms], dim=0).contiguous()
        self.rank = self.norms[0].linear_1.weight.shape[0]
        self.count = len(self.norms)
        self.cache: torch.Tensor | None = None

    def compute(self, embedded_timestep: torch.Tensor, temb: torch.Tensor | None) -> torch.Tensor:
        activated = torch.nn.functional.silu(embedded_timestep)
        lowrank = activated @ self.w1.t()  # (1, count*rank)
        lowrank = lowrank.view(self.count, 1, self.rank)
        out = torch.bmm(lowrank, self.w2.transpose(1, 2)).squeeze(1)  # (count, 3C)
        if temb is not None:
            out = out + temb
        return out


# ------------------------------------------------------------ combined block plan for P2 and P3


def install_fused_adaln(model: Any, *, batched: bool = True, triton_norm: bool = True) -> None:
    """Restructure the block chain the way Inductor's schedule did.

    The gate-add that closes block i's feed-forward is carried forward and fused into the
    AdaLN of block i+1 (and, for the last block, into `norm_out`), so all 3*L residual
    updates ride inside a normalization kernel and none is a standalone `addcmul` pass.
    """
    state = _BatchedAdaLn(model) if batched else None
    pending: dict[str, Any] = {"branch": None, "mod": None}
    blocks = list(model.transformer_blocks)
    for index, block in enumerate(blocks):
        block.forward = MethodType(_make_block_forward(index, state, triton_norm, pending), block)
    model.norm_out.forward = MethodType(
        _make_norm_out_forward(triton_norm, pending), model.norm_out
    )
    original_forward = model.forward

    def forward(_self: Any, *args: Any, **kwargs: Any) -> Any:
        pending["branch"] = None
        pending["mod"] = None
        if state is not None:
            state.cache = None
        try:
            return original_forward(*args, **kwargs)
        finally:
            pending["branch"] = None
            pending["mod"] = None

    model.forward = MethodType(forward, model)


def _make_block_forward(
    index: int, batched: Any, triton_norm: bool, pending: dict[str, Any]
) -> Any:
    def forward(
        block: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Any,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
        image_rotary_emb: Any = None,
        extra_pos_emb: Any = None,
        attention_mask: Any = None,
        controlnet_residual: Any = None,
        latents: Any = None,
        block_idx: int | None = None,
    ) -> torch.Tensor:
        if batched is not None:
            if batched.cache is None:
                batched.cache = batched.compute(embedded_timestep, temb)
            mod1, mod2, mod3 = (batched.cache[3 * index + offset] for offset in range(3))
        else:
            mod1 = _modulation(block.norm1, embedded_timestep, temb)
            mod2 = _modulation(block.norm2, embedded_timestep, temb)
            mod3 = _modulation(block.norm3, embedded_timestep, temb)

        if not triton_norm:
            normed, gate = _apply_modulation(block.norm1, hidden_states, mod1)
            output = block.attn1(normed, image_rotary_emb=image_rotary_emb)
            hidden_states = torch.addcmul(hidden_states, gate, output)
            normed, gate = _apply_modulation(block.norm2, hidden_states, mod2)
            output = block.attn2(
                normed, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
            )
            hidden_states = torch.addcmul(hidden_states, gate, output)
            normed, gate = _apply_modulation(block.norm3, hidden_states, mod3)
            return torch.addcmul(hidden_states, gate, block.ff(normed))

        normed = gate_add_adaln(hidden_states, pending["branch"], pending["mod"], mod1)
        output = block.attn1(normed, image_rotary_emb=image_rotary_emb)
        normed = gate_add_adaln(hidden_states, output, mod1, mod2)
        output = block.attn2(
            normed, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )
        normed = gate_add_adaln(hidden_states, output, mod2, mod3)
        pending["branch"] = block.ff(normed)
        pending["mod"] = mod3
        return hidden_states

    return forward


def _make_norm_out_forward(triton_norm: bool, pending: dict[str, Any]) -> Any:
    """CosmosAdaLayerNorm, flushing the last block's carried feed-forward gate-add."""

    def forward(
        norm: Any,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod = norm.linear_2(norm.linear_1(norm.activation(embedded_timestep)))
        if temb is not None:
            mod = mod + temb[..., : 2 * norm.embedding_dim]
        if triton_norm and pending["branch"] is not None:
            # `mod` here is the 2C (shift, scale) of CosmosAdaLayerNorm; the kernel reads
            # shift at 0 and scale at C, so a 2C vector is the same layout minus the gate.
            out = gate_add_adaln(hidden_states, pending["branch"], pending["mod"], mod)
            pending["branch"] = None
            return out
        shift, scale = mod.chunk(2, dim=-1)
        normed = norm.norm(hidden_states)
        if mod.ndim == 2:
            shift, scale = (x.unsqueeze(1) for x in (shift, scale))
        return normed * (1 + scale) + shift

    return forward


def _modulation(norm: Any, embedded_timestep: torch.Tensor, temb: torch.Tensor | None) -> Any:
    value = norm.linear_2(norm.linear_1(norm.activation(embedded_timestep)))
    return value if temb is None else value + temb


def _apply_modulation(norm: Any, hidden_states: torch.Tensor, mod: torch.Tensor) -> Any:
    shift, scale, gate = mod.chunk(3, dim=-1)
    if mod.ndim == 2:
        shift, scale, gate = (x.unsqueeze(1) for x in (shift, scale, gate))
    return norm.norm(hidden_states) * (1 + scale) + shift, gate
