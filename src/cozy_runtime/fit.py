"""What a decode of a model holds on its card, from its config and stored tensors (xs-213, design
inference-engine.md §4.4, §4.5, §6.4 "By fit"): the numbers a lane's header carries as
`metadata.decode` (cozytensors.md §4), which the Hub's lane rule reads, and the K/V and state
layout the engine declares, so the two can't disagree.

`decode` is `{component: {bytes, kv, state, step}}`:
- `bytes`: stored bytes a decode step reads: the language layers, token table, head and draft
  (MTP) layer; not a vision tower, which only a prefill with pictures reads.
- `kv`: K and V bytes one token adds at 16 bits, every attention layer's, the draft layer's too.
- `state`: one row's recurrent state (each gated-delta layer's conv and recurrent state).
- `step`: an estimate of one row's working bytes beside those (a prefill chunk's widest
  activations, a drafted step's logits and candidate states), until the function's measured
  peak on the card replaces it.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

from tensorfs import raw

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorfs import Checkpoint

#: The header metadata key (cozytensors.md §4).
KEY = "decode"
#: Tokens a draft head proposes a step (`lm.DEPTH`); a drafted step reads its token and these.
DEPTH = 2
#: The most prompt tokens a prefill step computes at once (`lm.PREFILL_TOKENS`).
CHUNK = 4096
#: The draft head's tensors, kept beside the model's own (`vlm.construct`).
DRAFT = "mtp."
#: Each state tensor's alignment in a row's slot (`engine.kv`).
_ALIGN = 256
_BYTES = {"bfloat16": 2, "float32": 4}
_DELTA_NET = (
    "conv_kernel_dim",
    "num_key_heads",
    "key_head_dim",
    "num_value_heads",
    "value_head_dim",
)


@dataclass(frozen=True, slots=True)
class Layout:
    """A decoder's K/V and recurrent state for one row, from its config. The engine's `KvLayout`
    is `(attn_layers=attention + drafts, kv_heads, head_dim, torch.bfloat16, states)`."""

    #: Layers that keep K/V: the model's, then its draft head's.
    attention: int
    drafts: int
    kv_heads: int
    head_dim: int
    #: Each gated-delta layer's conv state, then its recurrent state: (shape, torch dtype name).
    states: tuple[tuple[tuple[int, ...], str], ...]

    @property
    def kv(self) -> int:
        """K and V bytes a token, at 16 bits."""
        return (self.attention + self.drafts) * 2 * self.kv_heads * self.head_dim * 2

    @property
    def state(self) -> int:
        """One row's state tensors, each aligned."""
        return sum(
            -(-prod(shape) * _BYTES[dtype] // _ALIGN) * _ALIGN for shape, dtype in self.states
        )


def layout(config: Mapping[str, object], drafts: bool) -> Layout | None:
    """`config`'s decoder (its `text_config`, else itself) as `Layout`; `drafts`: its checkpoint
    keeps a draft head. None when its layers are not attention and gated-delta layers."""
    text = _text(config)
    layers, heads = _int(text, "num_hidden_layers"), _int(text, "num_attention_heads")
    if not layers or not heads:
        return None
    types = text.get("layer_types")
    if not isinstance(types, list):
        every = _int(text, "full_attention_interval") or 1
        types = [
            "full_attention" if (i + 1) % every == 0 else "linear_attention" for i in range(layers)
        ]
    attention, states = 0, list[tuple[tuple[int, ...], str]]()
    for kind in types:
        if kind in ("full_attention", "sliding_attention"):
            attention += 1
            continue
        kernel, k_heads, k_dim, v_heads, v_dim = (_int(text, f"linear_{n}") for n in _DELTA_NET)
        if kind != "linear_attention" or not all((kernel, k_heads, k_dim, v_heads, v_dim)):
            return None
        states += [
            ((2 * k_heads * k_dim + v_heads * v_dim, kernel), "bfloat16"),
            ((v_heads, k_dim, v_dim), "float32"),
        ]
    return Layout(
        attention=attention,
        drafts=_int(text, "mtp_num_hidden_layers") if drafts else 0,
        kv_heads=_int(text, "num_key_value_heads") or heads,
        head_dim=_int(text, "head_dim") or _int(text, "hidden_size") // heads,
        states=tuple(states),
    )


def numbers(checkpoint: Checkpoint, component: str) -> dict[str, int] | None:
    """`component`'s `decode` numbers in `checkpoint`; None when its config is no decoder
    `layout` reads."""
    held = checkpoint.components[component]
    config = dict(held.config)
    stored = {
        key: sum(p.length for p in raw.parts(checkpoint, component, key).values())
        for key in held.tensors
    }
    shape = layout(config, f"{DRAFT}fc.weight" in stored)
    if shape is None:
        return None
    text = _text(config)
    unread = tuple(f"{path}." for path in _unread(held.cls.name, config))
    routed = _int(text, "moe_intermediate_size") * _int(text, "num_experts_per_tok")
    wide = max(
        _int(text, "intermediate_size"), routed + _int(text, "shared_expert_intermediate_size")
    )
    span = DEPTH + 1 if shape.drafts else 1
    step = CHUNK * (_int(text, "hidden_size") + 2 * wide) * 2 + span * _int(text, "vocab_size") * 4
    return {
        "bytes": sum(n for key, n in stored.items() if not key.startswith(unread)),
        "kv": shape.kv,
        "state": shape.state,
        "step": step + (span * shape.state if shape.drafts else 0),
    }


def _text(config: Mapping[str, object]) -> Mapping[str, object]:
    text = config.get("text_config", config)
    return text if isinstance(text, dict) else {}


def _int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _unread(name: str, config: Mapping[str, object]) -> list[str]:
    """The module paths of Transformers' class `name`, built from `config` on the meta device,
    that a decode step never reads: all outside its decoder and head (`get_decoder`,
    `get_output_embeddings`), as `lm._decoding` reads them. A class this can't build has none:
    every tensor counts as read, which overcounts and never undercounts."""
    with contextlib.suppress(Exception):
        import torch
        import transformers
        from transformers.initialization import no_init_weights

        with torch.device("meta"), no_init_weights():
            model = getattr(transformers, name)._from_config(
                transformers.AutoConfig.for_model(**config)
            )
        reads = {
            id(m) for m in (model.get_decoder(), model.get_output_embeddings()) if m is not None
        }
        found: list[str] = []

        def walk(module: torch.nn.Module, path: str) -> None:
            for child, sub in module.named_children():
                at = f"{path}.{child}" if path else child
                if id(sub) in reads:
                    continue
                if any(id(m) in reads for m in sub.modules()):
                    walk(sub, at)
                else:
                    found.append(at)

        walk(model, "")
        return found
    return []
