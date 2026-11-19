"""engine-parts (xs-213, a bisect; proof only): engine-step's small random Qwen3.5 and four rows, plain greedy,
each arm held against Transformers' teacher-forced forward at its clear positions, as engine-step holds them:
- `hf-decode`: Transformers' own cached greedy decode of each row (its recurrent gated-delta rule and conv
  update, one token a forward);
- `engine`: the engine's step as it is;
- `attn-ref`: the step with its paged attention (write and read) done by torch's ops on the same blocks;
- `delta-ref`: the step with each gated-delta layer done by Transformers' own torch functions
  (`causal_conv1d_update`, `torch_recurrent_gated_delta_rule`) on the same state slots;
- `both-ref`: both;
- `engine-alone`, `both-ref-alone`: each row decoded alone (a bucket of one).
The reference arms run eagerly (no graph). Each row reports its first clear position an arm misses, the lead
there, the two tokens, and where it first differs from `hf-decode`. It judges nothing: it reports."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from cozy_runtime.engine.attention import HostTier
    from cozy_runtime.engine.kv import KvLayout
    from cozy_runtime.engine.qwen35 import Qwen35

type JSON = None | bool | int | float | str | Sequence[JSON] | Mapping[str, JSON]


class _Linear(Protocol):
    def __call__(self, x: torch.Tensor, /) -> torch.Tensor: ...


class _Conv(Protocol):
    weight: torch.Tensor
    bias: torch.Tensor | None


class _Gated(Protocol):
    def __call__(self, x: torch.Tensor, gate: torch.Tensor, /) -> torch.Tensor: ...


class _Delta(Protocol):
    """A Transformers Qwen3.5 gated-delta layer, as this reference reads it."""

    in_proj_qkv: _Linear
    in_proj_z: _Linear
    in_proj_b: _Linear
    in_proj_a: _Linear
    out_proj: _Linear
    conv1d: _Conv
    norm: _Gated
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    activation: str


def _paged_reference(
    q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor, card: torch.Tensor, host: HostTier,
    block_table: torch.Tensor, lengths: torch.Tensor, slot_mapping: torch.Tensor, layout: KvLayout, layer: int,
    parts: int, scale: float, first_seen: int = 0, gate: torch.Tensor | None = None, fused: bool = True,
) -> torch.Tensor:  # fmt: skip
    """`attention.paged_step`'s contract in torch's ops, the card tier only, eagerly."""
    rows, span, heads, dim = (int(n) for n in q.shape)
    tokens, group = layout.block_tokens, heads // layout.kv_heads
    out = torch.zeros_like(q)
    for r, n in enumerate(lengths[:rows].tolist()):
        if n < 0:
            continue
        for i, slot in enumerate(slot_mapping[r, :span].tolist()):
            if slot >= 0:
                card[slot // tokens, layer, 0, slot % tokens] = k_new[r, i]
                card[slot // tokens, layer, 1, slot % tokens] = v_new[r, i]
        total = n + span
        held = block_table[r, : -(-total // tokens)].tolist()
        keys = card[held, layer, 0].reshape(-1, layout.kv_heads, dim)[:total].float().repeat_interleave(group, dim=1)
        values = card[held, layer, 1].reshape(-1, layout.kv_heads, dim)[:total].float().repeat_interleave(group, dim=1)
        scores = torch.einsum("shd,thd->hst", q[r].float(), keys) * scale
        at = torch.arange(total, device=q.device)
        seen = (at[None, :] <= n + torch.arange(span, device=q.device)[:, None]) & (at[None, :] >= first_seen)
        scores = scores.masked_fill(~seen[None], float("-inf"))
        out[r] = torch.einsum("hst,thd->shd", scores.softmax(-1), values).to(q.dtype)
    return out if gate is None else out * torch.sigmoid(gate)


def _delta_reference(self: Qwen35, block: torch.nn.Module, x: torch.Tensor, index: int) -> torch.Tensor:
    """A plain step's gated-delta layer by Transformers' own functions on the engine's state slots."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    rows, span, _ = x.shape
    if span != 1:
        raise ValueError("delta-ref serves plain steps only")
    text, kv = self.text, self.kv
    b_ = cast("_Delta", block)
    mixed = b_.in_proj_qkv(x).transpose(1, 2)
    z = b_.in_proj_z(x).reshape(rows, span, text.v_heads, text.dv)
    b, a = b_.in_proj_b(x), b_.in_proj_a(x)
    windows, states = kv.state(0), kv.state(1)
    keys = text.k_heads * text.dk
    out = torch.zeros((rows, span, text.v_heads, text.dv), dtype=x.dtype, device=x.device)
    for r, slot in enumerate(kv.state_slot[:rows].tolist()):
        if slot < 0:
            continue
        window = windows[slot, index][None].clone()
        conv = qwen.causal_conv1d_update(mixed[r : r + 1], window, b_.conv1d.weight.squeeze(1), b_.conv1d.bias,
                                         b_.activation).transpose(1, 2)  # fmt: skip
        windows[slot, index].copy_(window[0])
        q = conv[..., :keys].reshape(1, 1, text.k_heads, text.dk)
        k = conv[..., keys : 2 * keys].reshape(1, 1, text.k_heads, text.dk)
        v = conv[..., 2 * keys :].reshape(1, 1, text.v_heads, text.dv)
        rep = text.v_heads // text.k_heads
        if rep > 1:
            q, k = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        beta = b[r : r + 1].sigmoid()
        g = -b_.A_log.float().exp() * F.softplus(a[r : r + 1].float() + b_.dt_bias)
        made, final = qwen.torch_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=states[slot, index][None], output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )  # fmt: skip
        states[slot, index].copy_(final[0])
        out[r] = made[0]
    normed = b_.norm(out.reshape(-1, text.dv), z.reshape(-1, text.dv))
    return b_.out_proj(normed.reshape(rows, span, -1))


@contextmanager
def _swapped(attention: bool, delta: bool) -> Iterator[None]:
    from cozy_runtime.engine import decoder, qwen35

    def eager(self: object, step: object, made: object, *, keep_graph: bool = False) -> None:
        return None

    names = ((qwen35, "paged_step"), (qwen35.Qwen35, "_delta"), (decoder.Decoder, "_capture"))
    saved = [getattr(owner, name) for owner, name in names]
    try:
        if attention:
            setattr(qwen35, "paged_step", _paged_reference)  # the same contract
        if delta:
            setattr(qwen35.Qwen35, "_delta", _delta_reference)  # the same contract
        if attention or delta:
            setattr(decoder.Decoder, "_capture", eager)
        yield
    finally:
        for (owner, name), value in zip(names, saved, strict=True):
            setattr(owner, name, value)


def measure() -> dict[str, JSON]:
    from cozy_runtime.engine.decoder import Decoder, Request
    from cozy_runtime.internal.bench.engine_step import MARGIN, NEW, _forced, _model

    model = _model()
    del model.mtp
    g = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, 4096, (n,), generator=g).tolist() for n in (37, 5, 120, 64)]
    runs: dict[str, list[list[int]]] = {}
    errors: dict[str, str] = {}
    with torch.inference_mode():
        runs["hf-decode"] = []
        for prompt in prompts:
            ids = torch.tensor([prompt], device="cuda")
            made = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=NEW,  # type: ignore[operator]
                                  min_new_tokens=NEW, do_sample=False)  # fmt: skip
            runs["hf-decode"].append(made[0, len(prompt) :].tolist())
    for name, attention, delta in (("engine", False, False), ("attn-ref", True, False), ("delta-ref", False, True),
                                   ("both-ref", True, True)):  # fmt: skip
        with _swapped(attention, delta):
            decoder = Decoder(model, max_rows=4, max_tokens=1024)
            try:
                runs[name] = decoder(Request(prompts=prompts, max_new=NEW, stops=[])).tokens
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"[:600]
    # Each row alone, a bucket of one: a fault in how rows share a step shows as a bucket's rows wrong and these right.
    alone = Decoder(model, max_rows=4, max_tokens=1024)
    runs["engine-alone"] = [alone(Request(prompts=[p], max_new=NEW, stops=[])).tokens[0] for p in prompts]
    with _swapped(True, True):
        refs = Decoder(model, max_rows=4, max_tokens=1024)
        runs["both-ref-alone"] = [refs(Request(prompts=[p], max_new=NEW, stops=[])).tokens[0] for p in prompts]
    rows: dict[str, JSON] = {}
    for i, prompt in enumerate(prompts):
        made = runs["hf-decode"][i]
        row: dict[str, JSON] = {"prompt": len(prompt)}
        for name, tokens in runs.items():
            mine = tokens[i]
            choice, lead = _forced(model, list(prompt) + mine[:-1])
            clear = [t for t in range(len(mine)) if lead[len(prompt) - 1 + t] >= MARGIN]
            wrong = next((t for t in clear if choice[len(prompt) - 1 + t] != mine[t]), None)
            at = len(prompt) - 1 + wrong if wrong is not None else None
            row[name] = {
                "clear": len(clear), "first_wrong": wrong,
                "lead": None if at is None else round(lead[at], 3),
                "made": None if wrong is None else mine[wrong], "forced": None if at is None else choice[at],
                "vs_hf_decode": next((t for t, (x, y) in enumerate(zip(mine, made, strict=False)) if x != y), None),
            }  # fmt: skip
        rows[f"row{i}"] = row
    return {"gpu": torch.cuda.get_device_name(), "errors": cast("JSON", errors), "rows": rows}
