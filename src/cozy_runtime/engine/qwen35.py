"""Qwen3.5's forward for the engine's decode step (xs-213 phase 2): a row's token and its drafts
through the language layers, the multi-token-prediction head and the output head, over the K/V
blocks and state slots of `engine.kv`.

It reads the weights of the model as loaded (its modules, whatever lane serves them: a linear is
called as the module it is) and computes each op as Transformers' Qwen3.5 does, rounding where it
rounds, so a greedy decode keeps its tokens. Attention is `engine.attention.paged_step` and the
gated-delta layers are `engine._kernels`' span kernels: every length, slot and table entry is read
on the device, so a step's graph holds only addresses. Nothing here allocates outside the step's
own working memory, and nothing imports `cozy_runtime.lm`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.nn.functional as F

from cozy_runtime.engine import _kernels
from cozy_runtime.engine.attention import HostTier, paged_step
from cozy_runtime.engine.kv import KvLayout, StateSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from cozy_runtime.engine.kv import KvManager


class _Linear(Protocol):
    """A linear as loaded: whatever lane serves it, called as the module it is."""

    def __call__(self, x: Tensor, /) -> Tensor: ...


class _Norm(Protocol):
    weight: Tensor


class _GatedNorm(Protocol):
    weight: Tensor
    variance_epsilon: float


class _Conv(Protocol):
    weight: Tensor


class _Attention(Protocol):
    q_proj: _Linear
    k_proj: _Linear
    v_proj: _Linear
    o_proj: _Linear
    q_norm: _Norm
    k_norm: _Norm


class _Delta(Protocol):
    in_proj_qkv: _Linear
    in_proj_z: _Linear
    in_proj_b: _Linear
    in_proj_a: _Linear
    out_proj: _Linear
    conv1d: _Conv
    A_log: Tensor
    dt_bias: Tensor
    norm: _GatedNorm


class _Mlp(Protocol):
    gate_proj: _Linear
    up_proj: _Linear
    down_proj: _Linear
    act_fn: _Linear


class _Layer(Protocol):
    """`Qwen3_5DecoderLayer`, as the step reads it: an attention layer has `self_attn`, a
    gated-delta one `linear_attn`."""

    input_layernorm: _Norm
    post_attention_layernorm: _Norm
    mlp: _Mlp
    self_attn: _Attention
    linear_attn: _Delta


class _Mtp(Protocol):
    """The checkpoint's multi-token-prediction head (`model.mtp`, `lm.Draft`'s modules)."""

    fc: _Linear
    pre_fc_norm_embedding: _Norm
    pre_fc_norm_hidden: _Norm
    layers: Sequence[_Layer]
    norm: _Norm

#: A row's split-KV parts at once: fixed, so a step's graph serves every length.
PARTS = 32
#: The conv's channels a program takes.
CONV_BLOCK = 256
#: Value columns a delta-rule program takes.
VALUE_BLOCK = 32


@dataclass(frozen=True)
class Text:
    """What the step reads of a Qwen3.5 text config."""

    hidden: int
    kinds: tuple[str, ...]
    heads: int
    kv_heads: int
    head_dim: int
    rotary: int
    theta: float
    eps: float
    k_heads: int
    v_heads: int
    dk: int
    dv: int
    conv: int

    @property
    def attention(self) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.kinds) if kind == "full_attention")

    @property
    def delta(self) -> tuple[int, ...]:
        return tuple(i for i, kind in enumerate(self.kinds) if kind == "linear_attention")

    @property
    def conv_channels(self) -> int:
        return 2 * self.k_heads * self.dk + self.v_heads * self.dv


class _TextConfig(Protocol):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    layer_types: Sequence[str]
    rms_norm_eps: float
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int


class _Config(Protocol):
    def get_text_config(self, decoder: bool = ...) -> _TextConfig: ...


class _Configured(Protocol):
    config: _Config


def text_of(model: torch.nn.Module) -> Text:
    config = cast("_Configured", model).config.get_text_config(decoder=True)
    rope = dict(getattr(config, "rope_parameters", None) or {})
    head_dim = int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)
    return Text(
        hidden=int(config.hidden_size),
        kinds=tuple(config.layer_types),
        heads=int(config.num_attention_heads),
        kv_heads=int(config.num_key_value_heads),
        head_dim=head_dim,
        rotary=int(head_dim * float(rope.get("partial_rotary_factor", 1.0))),
        theta=float(rope["rope_theta"]),
        eps=float(config.rms_norm_eps),
        k_heads=int(config.linear_num_key_heads),
        v_heads=int(config.linear_num_value_heads),
        dk=int(config.linear_key_head_dim),
        dv=int(config.linear_value_head_dim),
        conv=int(config.linear_conv_kernel_dim),
    )


def layout_of(text: Text, depth: int, drafting: bool) -> KvLayout:
    """The K/V and state the step keeps: every attention layer's K/V, and the head's own layer
    last where it drafts; a gated-delta layer's conv window and recurrent state."""
    layers = len(text.attention) + (1 if drafting else 0)
    delta = len(text.delta)
    return KvLayout(
        attn_layers=layers,
        kv_heads=text.kv_heads,
        head_dim=text.head_dim,
        dtype=torch.bfloat16,
        states=(
            StateSpec((delta, text.conv_channels, text.conv - 1), torch.bfloat16),
            StateSpec((delta, text.v_heads, text.dk, text.dv), torch.float32),
        ),
        depth=depth,
    )


@dataclass(frozen=True)
class Tables:
    """Where a step's attention writes and what it sees, per row: `lengths` the position of
    its first query (-1: not decoding), `slots` each query's token slot, `first_seen` the
    lowest position it reads. The body's are the manager's; the draft head's sit a position
    ahead and see from 1."""

    lengths: Tensor
    slots: Tensor
    first_seen: int = 0


#: The widest row the fused norm takes in one program.
NORM_WIDEST = 8192
#: An MLP's intermediate columns a SiLU-and-multiply program takes.
MLP_BLOCK = 1024


def _norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """`Qwen3_5RMSNorm`: in f32, times 1 + weight, back to x's dtype: one launch for every row on CUDA
    (`_kernels.rms_norm_kernel`), else torch's seven ops."""
    width = int(x.shape[-1])
    if x.is_cuda and not _kernels.TRITON_ABSENT and width <= NORM_WIDEST:
        rows = x.reshape(-1, width).contiguous()
        out = torch.empty_like(rows)
        block = 1 << max(width - 1, 0).bit_length()
        _kernels.rms_norm_kernel[(int(rows.shape[0]),)](
            rows, weight, out, width, eps, BLOCK=block, num_warps=8 if block >= 2048 else 4
        )
        return out.view(x.shape)
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + weight.float())).type_as(x)


def _gated(x: Tensor, gate: Tensor, weight: Tensor, eps: float) -> Tensor:
    """`Qwen3_5RMSNormGated`: the norm before the gate, SiLU of the gate in f32: one launch for
    every row on CUDA (`_kernels.gated_norm_kernel`), else torch's ops."""
    width = int(x.shape[-1])
    if x.is_cuda and not _kernels.TRITON_ABSENT and (width & (width - 1)) == 0 and width <= NORM_WIDEST:
        rows = x.reshape(-1, width).contiguous()
        out = torch.empty_like(rows)
        _kernels.gated_norm_kernel[(int(rows.shape[0]),)](
            rows, gate.reshape(-1, width).contiguous(), weight, out, eps,
            D=width, WEIGHT_ROUNDS=weight.dtype == x.dtype, num_warps=1 if width <= 256 else 4,
        )  # fmt: skip
        return out.view(x.shape)
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    y = weight * y.to(dtype)
    return (y * F.silu(gate.float())).to(dtype)


def _add_norm(x: Tensor, h: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    """The residual stream `x + h` and its `Qwen3_5RMSNorm`: one launch for every row on CUDA
    (`_kernels.add_norm_kernel`), else torch's add and `_norm`."""
    width = int(x.shape[-1])
    if x.is_cuda and not _kernels.TRITON_ABSENT and width <= NORM_WIDEST:
        rows = x.reshape(-1, width).contiguous()
        total, out = torch.empty_like(rows), torch.empty_like(rows)
        block = 1 << max(width - 1, 0).bit_length()
        _kernels.add_norm_kernel[(int(rows.shape[0]),)](
            rows, h.reshape(-1, width).contiguous(), weight, total, out, width, eps,
            BLOCK=block, num_warps=8 if block >= 2048 else 4,
        )  # fmt: skip
        return total.view(x.shape), out.view(x.shape)
    total = x + h
    return total, _norm(total, weight, eps)


def _silu_mul(gate: Tensor, up: Tensor) -> Tensor:
    """SiLU(gate) * up in one launch (`_kernels.silu_mul_kernel`)."""
    width = int(gate.shape[-1])
    out = torch.empty_like(gate)
    rows = gate.numel() // width
    _kernels.silu_mul_kernel[(rows, -(-width // MLP_BLOCK))](
        gate.contiguous(), up.contiguous(), out, width, BLOCK=MLP_BLOCK, num_warps=4
    )
    return out


class Qwen35:
    """One loaded Qwen3.5 model's step, over `kv`'s blocks and slots. `positions` hold each
    row's text position offset (Qwen-VL's rope delta), as `[rows]` int32."""

    def __init__(self, model: torch.nn.Module, kv: KvManager, host: HostTier) -> None:
        self.model, self.kv, self.host = model, kv, host
        self.text = text_of(model)
        root = model.get_decoder()  # type: ignore[operator]  # a Transformers model's
        self.layers = [cast("_Layer", layer) for layer in root.layers]
        self.final = cast("_Norm", root.norm)
        self.embed = cast("_Linear", model.get_input_embeddings())  # type: ignore[operator]  # as above
        self.head = cast("_Linear", model.get_output_embeddings())  # type: ignore[operator]  # as above
        self.mtp = cast("_Mtp | None", getattr(model, "mtp", None))
        text = self.text
        inv = 1.0 / (text.theta ** (torch.arange(0, text.rotary, 2, dtype=torch.float) / text.rotary))
        self.inv_freq = inv.to(kv.device)
        #: an attention layer's index among the K/V's layers; the head's own is last
        self.slot_of = {layer: i for i, layer in enumerate(text.attention)}
        self.delta_of = {layer: i for i, layer in enumerate(text.delta)}
        mlps = [layer.mlp for layer in self.layers] + ([layer.mlp for layer in self.mtp.layers] if self.mtp else [])
        #: every MLP's activation is SiLU, so SiLU(gate) * up is one launch
        self.silu = all(type(m.act_fn).__name__ in ("SiLU", "SiLUActivation") for m in mlps)

    # ------------------------------------------------------------------ pieces

    def _attention(
        self,
        block: _Attention,
        x: Tensor,
        positions: Tensor,
        kv_layer: int,
        tables: Tables,
        eps: float,
    ) -> Tensor:
        """Qwen3.5's gated attention at text `positions` (`[rows, S]`, contiguous): q's and k's
        norms and RoPE in one launch, the step's attention, its output gate in the join."""
        rows, span, _ = x.shape
        text, dim = self.text, self.text.head_dim
        projected = block.q_proj(x).view(rows, span, -1, dim * 2)
        heads = int(projected.shape[2])
        k = block.k_proj(x).contiguous()
        v = block.v_proj(x).view(rows, span, -1, dim)
        q_out = torch.empty((rows, span, heads, dim), dtype=projected.dtype, device=x.device)
        k_out = torch.empty((rows, span, text.kv_heads, dim), dtype=k.dtype, device=x.device)
        _kernels.qk_norm_rope_kernel[(rows * span, heads + text.kv_heads)](
            projected.contiguous(), k, block.q_norm.weight, block.k_norm.weight, positions, self.inv_freq,
            q_out, k_out, eps, heads, text.kv_heads, 2 * dim, D=dim, R=text.rotary, num_warps=2,
        )  # fmt: skip
        kv = self.kv
        out = paged_step(
            q_out, k_out, v, kv.blocks, self.host, kv.block_table[:rows], tables.lengths, tables.slots,
            kv.layout, kv_layer, PARTS, dim**-0.5, tables.first_seen, gate=projected[..., dim:],
        )  # fmt: skip
        return block.o_proj(out.reshape(rows, span, -1))

    def _mlp(self, m: _Mlp, h: Tensor) -> Tensor:
        gate, up = m.gate_proj(h), m.up_proj(h)
        return m.down_proj(_silu_mul(gate, up) if self.silu and h.is_cuda else m.act_fn(gate) * up)

    def _delta(self, block: _Delta, x: Tensor, index: int) -> Tensor:
        rows, span, _ = x.shape
        text, kv = self.text, self.kv
        mixed = block.in_proj_qkv(x).contiguous()
        z = block.in_proj_z(x).reshape(rows, span, text.v_heads, text.dv)
        b = block.in_proj_b(x)
        a = block.in_proj_a(x)
        channels = text.conv_channels
        conv = torch.empty_like(mixed)
        # Every pointer a kernel is given must lie on a mapped page (Triton's launcher asks the
        # driver of each): one token writes no scratch, so it is given the state again.
        windows, kept_windows = kv.state(0), kv.state_scratch[0]
        window_scratch = kept_windows[:, :, index] if span > 1 else windows[:, index]
        _kernels.conv_span_kernel[(rows, -(-channels // CONV_BLOCK))](
            mixed, block.conv1d.weight, kv.state_slot[:rows], windows[:, index],
            window_scratch, conv, channels, int(windows.stride(0)),
            int(kept_windows.stride(0)), int(kept_windows.stride(1)),
            S=span, W=text.conv - 1, BLOCK=CONV_BLOCK, num_warps=4,
        )  # fmt: skip
        keys = text.k_heads * text.dk
        q, k, v = conv[..., :keys], conv[..., keys : 2 * keys], conv[..., 2 * keys :]
        out = torch.empty((rows, span, text.v_heads, text.dv), dtype=x.dtype, device=x.device)
        states, kept_states = kv.state(1), kv.state_scratch[1]
        state_scratch = kept_states[:, :, index] if span > 1 else states[:, index]
        _kernels.delta_span_kernel[(rows * text.v_heads, text.dv // VALUE_BLOCK)](
            q, k, v, a.contiguous(), b.contiguous(), block.A_log, block.dt_bias, kv.state_slot[:rows],
            states[:, index], state_scratch,
            out, text.v_heads, text.v_heads // text.k_heads, int(conv.stride(1)),
            int(conv.stride(1)), int(states.stride(0)), int(kept_states.stride(0)),
            int(kept_states.stride(1)), float(text.dk) ** 0.5,
            S=span, DK=text.dk, DV=text.dv, BV=VALUE_BLOCK, num_warps=4,
        )  # fmt: skip
        normed = _gated(
            out.reshape(-1, text.dv), z.reshape(-1, text.dv), block.norm.weight, block.norm.variance_epsilon
        )
        return block.out_proj(normed.reshape(rows, span, -1))

    # ------------------------------------------------------------------ what a step calls

    def body(self, ids: Tensor, positions: Tensor, tables: Tables) -> Tensor:
        """The final-normed hidden state `[rows, S, hidden]` of each row's `S` tokens (`ids`)
        at text `positions` `[rows, S]`; their K/V written at `tables`' slots and their states
        into the slots (one token) or the scratch (a span)."""
        x = self.embed(ids)
        eps, layers = self.text.eps, self.layers
        positions = positions.contiguous()
        h = _norm(x, layers[0].input_layernorm.weight, eps)
        for i, layer in enumerate(layers):
            if i in self.slot_of:
                mixed = self._attention(layer.self_attn, h, positions, self.slot_of[i], tables, eps)
            else:
                mixed = self._delta(layer.linear_attn, h, self.delta_of[i])
            x, h = _add_norm(x, mixed, layer.post_attention_layernorm.weight, eps)
            after = layers[i + 1].input_layernorm.weight if i + 1 < len(layers) else self.final.weight
            x, h = _add_norm(x, self._mlp(layer.mlp, h), after, eps)
        return h

    def draft(self, ids: Tensor, hidden: Tensor, positions: Tensor, tables: Tables) -> Tensor:
        """The head's state `[rows, S, hidden]` for tokens `ids` at `positions`, each from the
        state before it (`hidden`): `lm.Draft`'s arithmetic, its K/V in the last K/V layer."""
        mtp = self.mtp
        if mtp is None:
            raise RuntimeError("this checkpoint has no multi-token-prediction head")
        eps = self.text.eps
        e = _norm(self.embed(ids), mtp.pre_fc_norm_embedding.weight, eps)
        h = _norm(hidden, mtp.pre_fc_norm_hidden.weight, eps)
        x = mtp.fc(torch.cat([e, h], -1))
        layer = mtp.layers[0]
        a = _norm(x, layer.input_layernorm.weight, eps)
        mixed = self._attention(layer.self_attn, a, positions.contiguous(), self.kv.layout.attn_layers - 1, tables, eps)
        x, a = _add_norm(x, mixed, layer.post_attention_layernorm.weight, eps)
        _, out = _add_norm(x, self._mlp(layer.mlp, a), mtp.norm.weight, eps)
        return out

    def logits(self, hidden: Tensor) -> Tensor:
        return self.head(hidden).float()

    def keep(self, kept: Tensor) -> None:
        """Each decoding row's state after the last token its span kept (`kept`, `[rows]`, an
        index into the span; -1 keeps nothing), from the scratch into its slot."""
        kv = self.kv
        rows = int(kept.shape[0])
        for state, scratch in zip((kv.state(0), kv.state(1)), kv.state_scratch, strict=True):
            words = scratch[0, 0].numel() * scratch.element_size() // 4
            _kernels.keep_kernel[(rows, -(-words // 1024))](
                scratch.view(torch.uint8), kv.state_slot[:rows], kept, state.view(torch.uint8), words,
                int(state.stride(0) * state.element_size() // 4),
                int(scratch.stride(0) * scratch.element_size() // 4),
                int(scratch.stride(1) * scratch.element_size() // 4), BLOCK=1024, num_warps=4,
            )  # fmt: skip
