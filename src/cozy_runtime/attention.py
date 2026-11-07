"""An attention call a package writes itself, as one site the Runtime manages: the Runtime picks
its kernel (FA3, Sage, Sol or SDPA) per site and honours `--attention-kernel`, as on Diffusers'
own attention modules. It imports no Diffusers, and holds no parameters, so adding one changes
no state dict."""

from __future__ import annotations

import torch

from cozy_runtime.internal.attention_site import Site


class Processor(Site):
    def __call__(
        self,
        attn: Attention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if mask is not None or bias is not None:
            mask, causal = _joined(q, k, mask, bias, causal), False
        return self.dispatch(q, k, v, attn_mask=mask, is_causal=causal)


def _joined(
    q: torch.Tensor,
    k: torch.Tensor,
    mask: torch.Tensor | None,
    bias: torch.Tensor | None,
    causal: bool,
) -> torch.Tensor | None:
    """One 4D SDPA mask: what `mask` and `bias` allow and add, and the causal triangle."""
    allowed: torch.Tensor | None = None
    added: torch.Tensor | None = None
    for term in (mask, bias):
        if term is None:
            continue
        term = term.reshape((1,) * (4 - term.ndim) + tuple(term.shape))
        if term.dtype == torch.bool:
            allowed = term if allowed is None else allowed & term
        else:
            term = term.to(q.dtype)
            added = term if added is None else added + term
    if causal:
        length, keys = q.shape[1], k.shape[1]
        triangle = torch.ones(length, keys, dtype=torch.bool, device=q.device).tril(keys - length)
        allowed = triangle[None, None] if allowed is None else allowed & triangle
    if added is None or allowed is None:
        return added if allowed is None else allowed
    return torch.where(allowed, added, float("-inf"))


class Attention(torch.nn.Module):
    """Attention over `(batch, tokens, heads, head_dim)` queries, keys and values, in `dtype`;
    keys and values may be longer than queries (a cached prefix).

    - `mask`: bool (True attends) or float added to the scores, broadcastable to
      `(batch, heads, query tokens, key tokens)` by torch's rule.
    - `bias`: float added to the scores, broadcastable the same way (a relative position bias).
    - `causal`: the module's or the call's; either makes the call causal. Each query attends
      to the keys up to its own position, aligned bottom-right when keys are longer: the
      queries are the keys' last tokens.

    The Runtime picks each call's kernel from its site's: one that cannot serve the call (a
    mask or bias where it serves none, causal at unequal lengths on Sage or cuDNN, causal at
    all on the quantized and sparse kernels) leaves that call to SDPA, and the run's record
    counts it. With a mask or bias, causal is folded into that mask."""

    def __init__(
        self, heads: int, head_dim: int, dtype: torch.dtype = torch.bfloat16, causal: bool = False
    ) -> None:
        super().__init__()
        self.heads, self.head_dim, self.activation_dtype = heads, head_dim, dtype
        self.causal, self.processor = causal, Processor()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        dtype = self.activation_dtype
        return self.processor(
            self, q.to(dtype), k.to(dtype), v.to(dtype), mask, bias, self.causal or causal
        )
