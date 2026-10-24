"""An attention call a package writes itself, as one site the Runtime manages: the Runtime picks
its kernel (FA3, Sage, Sol or SDPA) per site and honours `--attention-kernel`, as on Diffusers'
own attention modules. It holds no parameters, so adding one changes no state dict."""

from __future__ import annotations

from typing import Any

import torch
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn


class Processor:
    _attention_backend: Any = None
    _parallel_config: Any = None

    def __call__(
        self, attn: Attention, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return dispatch_attention_fn(
            q, k, v, backend=self._attention_backend, parallel_config=self._parallel_config
        )


class Attention(torch.nn.Module, AttentionModuleMixin):
    """Non-causal, unmasked attention over `(batch, tokens, heads, head_dim)` queries, keys and
    values, in `dtype`; keys and values may be longer than queries (a cached prefix)."""

    _default_processor_cls = Processor
    _available_processors = [Processor]  # noqa: RUF012 (Diffusers' class attribute)

    def __init__(self, heads: int, head_dim: int, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.heads, self.head_dim, self.activation_dtype = heads, head_dim, dtype
        self.set_processor(Processor())

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dtype = self.activation_dtype
        return self.processor(self, q.to(dtype), k.to(dtype), v.to(dtype))
