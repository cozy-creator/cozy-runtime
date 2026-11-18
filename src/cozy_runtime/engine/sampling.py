"""A step's token choice on the device (xs-213 phase 2): each row's argmax, or its draw from the
`top_k` and `top_p` cut of its scores over its temperature, with Gumbel noise hashed from (seed,
the token's index in the row's output, the vocabulary entry). The same arithmetic as `lm`'s, so a
seeded call draws the same tokens on either engine; nothing reads a value back to the host.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

_LOW32 = 0xFFFFFFFF


def _times(x: Tensor, constant: int) -> Tensor:
    """`x * constant` mod 2**32 for 32-bit values held in int64, with no int64 overflow."""
    return (x * (constant & 0xFFFF) + (((x * (constant >> 16)) & 0xFFFF) << 16)) & _LOW32


def _mix(x: Tensor) -> Tensor:
    """A 32-bit integer hash (Wellons' lowbias32): the same bits on every device."""
    x = x ^ (x >> 16)
    x = _times(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _times(x, 0x846CA68B)
    return x ^ (x >> 16)


def _draw(scores: Tensor, temperature: Tensor, top_p: Tensor, seed: Tensor, index: Tensor) -> Tensor:
    """Each row's sampled token from `scores` (`[rows, ..., vocab]`, f32); `temperature`, `top_p`
    and `seed` per row, `index` the token's position in the row's output (scores' shape without
    the vocabulary)."""
    lead = (-1, *[1] * (scores.dim() - 1))
    z = scores / temperature.clamp(min=1e-6).view(lead)
    ordered, order = z.sort(-1, descending=True)
    probabilities = ordered.softmax(-1)
    cut = (probabilities.cumsum(-1) - probabilities >= top_p.view(lead)) & (top_p.view(lead) < 1)
    outside = torch.zeros_like(cut).scatter_(-1, order, cut)
    base = _mix(_mix(seed & _LOW32) ^ ((seed >> 32) & _LOW32)).view(lead)
    key = _mix(base ^ _mix(index[..., None] & _LOW32))
    entry = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.long)
    bits = _mix(key ^ _mix(entry))
    # Below 1: float32 rounds the top hash value to 1, whose noise is +inf.
    uniform = (((bits >> 8).float() + 0.5) * (1.0 / (1 << 24))).clamp_(max=1 - 2**-24)
    noise = -torch.log(-torch.log(uniform))
    return (z + noise).masked_fill_(outside, float("-inf")).argmax(-1)


@dataclass(frozen=True)
class Sampling:
    """Each row's decoding, `[max_rows]` device tensors a step's graph holds. `kept` is the most
    logits a sampling row keeps (`top_k`, 0 for none): a step's constant, so its graph takes the
    `topk` of that many; `sampled`, whether any row samples (else every pick is an argmax)."""

    temperature: Tensor
    top_p: Tensor
    top_k: Tensor
    penalty: Tensor
    seed: Tensor
    kept: int
    sampled: bool

    def pick(self, scores: Tensor, rows: int, index: Tensor) -> Tensor:
        """Each of `rows` rows' token from `scores` (`[rows, ..., vocab]`, f32, the presence
        penalty already taken), at output positions `index`."""
        best = scores.argmax(-1)
        if not self.sampled:
            return best
        temperature = self.temperature[:rows]
        drawn = _draw(scores, temperature, self.top_p[:rows], self.seed[:rows], index)
        return torch.where((temperature > 0).view(-1, *[1] * (best.dim() - 1)), drawn, best)

    def keep(self, scores: Tensor, rows: int) -> Tensor:
        """`scores` with each sampling row's logits below its `top_k`th largest at -inf."""
        if not self.kept:
            return scores
        lead = (-1, *[1] * (scores.dim() - 1))
        k = self.top_k[:rows]
        at = (k - 1).clamp(0, self.kept - 1).view(lead).expand(*scores.shape[:-1], 1)
        kth = scores.topk(self.kept, dim=-1).values.gather(-1, at)
        return scores.masked_fill((k > 0).view(lead) & (scores < kth), float("-inf"))
