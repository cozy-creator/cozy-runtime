"""A Runtime-owned attention site (`cozy_runtime.attention.Attention`), with no Diffusers.

Selection treats its processor as a Diffusers one (`attention.install`): a kernel per site,
`--attention-kernel`, a request's pin. It sets `_attention_backend` to the kernel's backend
name, whose call (`attention_ulysses.function`) each of the site's calls reaches. Unset (no
Runtime selection, outside an executor), the site runs torch's SDPA.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers.models.attention_dispatch import AttentionBackendName
    from torch import Tensor

from cozy_runtime.internal import attention_ulysses
from cozy_runtime.internal import attention_upstream as upstream


class Site:
    """A Runtime site's processor."""

    #: the selected kernel's backend (`attention.install`), its name or Diffusers' member
    _attention_backend: str | AttentionBackendName | None = None
    #: Diffusers' context-parallel config, which a backend call passes through
    _parallel_config: object = None

    def dispatch(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Attention over `(batch, tokens, heads, head_dim)` on the site's kernel."""
        backend = self._attention_backend
        if backend is None:
            return upstream.sdpa(query, key, value, attn_mask=attn_mask, is_causal=is_causal)
        call = attention_ulysses.opened(getattr(backend, "value", backend))
        return call(
            query,
            key,
            value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            _parallel_config=self._parallel_config,
        )
