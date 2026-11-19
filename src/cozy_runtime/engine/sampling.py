"""A step's token choice on the device (xs-213 phase 2), in vLLM's order: the repetition,
frequency and presence penalties; then each sampling row's temperature, top-k, top-p and min-p cut;
then its draw: the argmax of the kept scores plus Gumbel noise hashed from (seed, the token's index
in the row's output, the vocabulary entry). The noise is the same bits on every device, and a drafted
token is kept only where it is the model's own draw at its index with that noise, so drafting samples
the same distribution as an undrafted call. The scores it is added to are not bit for bit the same
between calls (a verify pass against a one-token step, kernels' reduction order), so where two
entries nearly tie the draw can turn: the same seed gives the same text only up to such ties. A
greedy row (temperature 0) takes the argmax of its penalized scores. Every setting is per row;
nothing is read back to the host.

Where a row has a top-k, its top-p nucleus lies inside those k entries (top-p cuts the top-k
filtered distribution), so the cut and the draw run over the k kept entries alone: exact, and no
sort of the vocabulary. Only a sampling row with top-p and no top-k sorts it.
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


def _noise(seed: Tensor, index: Tensor, entry: Tensor) -> Tensor:
    """Gumbel noise for vocabulary `entry` (`[..., k]`) of the token at output `index` (`[...]`)
    of a row seeded `seed` (`[rows]`, broadcast over `index`'s leading dimension)."""
    lead = (-1, *[1] * (index.dim() - 1))
    base = _mix(_mix(seed & _LOW32) ^ ((seed >> 32) & _LOW32)).view(lead)
    key = _mix(base ^ _mix(index & _LOW32))
    bits = _mix(key[..., None] ^ _mix(entry))
    # Below 1: float32 rounds the top hash value to 1, whose noise is +inf.
    uniform = (((bits >> 8).float() + 0.5) * (1.0 / (1 << 24))).clamp_(max=1 - 2**-24)
    return -torch.log(-torch.log(uniform))


@dataclass(frozen=True)
class Sampling:
    """Each row's decoding, `[max_rows]` device tensors a step's graph holds, and a call's
    constants: `kept`, the most entries a row's top-k keeps (0: no row cuts by top-k); `sampled`,
    whether any row samples; `penalized`, whether any row has a penalty."""

    temperature: Tensor
    top_p: Tensor
    top_k: Tensor
    min_p: Tensor
    presence: Tensor
    frequency: Tensor
    repetition: Tensor
    seed: Tensor
    kept: int
    sampled: bool
    penalized: bool

    def penalties(self, logits: Tensor, rows: int, counts: Tensor, prompt: Tensor) -> Tensor:
        """`logits` (`[rows, ..., vocab]`, f32) after each row's penalties: repetition over the
        tokens in its prompt (`prompt`, bool) or output, then frequency (`counts`: each token's
        count in the output so far, broadcast as `logits`) and presence."""
        if not self.penalized:
            return logits
        lead = (-1, *[1] * (logits.dim() - 1))
        repetition = self.repetition[:rows].view(lead)
        seen = (counts > 0) | prompt
        logits = torch.where(seen, torch.where(logits > 0, logits / repetition, logits * repetition), logits)
        return logits - self.frequency[:rows].view(lead) * counts - self.presence[:rows].view(lead) * (counts > 0).float()

    def pick(self, scores: Tensor, rows: int, index: Tensor) -> Tensor:
        """Each of `rows` rows' token from penalized `scores` (`[rows, ..., vocab]`, f32) at
        output positions `index` (scores' shape without the vocabulary)."""
        best = scores.argmax(-1)
        if not self.sampled:
            return best
        lead = (-1, *[1] * (scores.dim() - 1))
        temperature = self.temperature[:rows]
        z = scores / temperature.clamp(min=1e-6).view(lead)
        if self.kept:
            values, entries = z.topk(self.kept, dim=-1)
            k = self.top_k[:rows].view(lead)
            rank = torch.arange(self.kept, device=z.device)
            outside = (k > 0) & (rank >= k)
            values = values.masked_fill(outside, float("-inf"))
        else:
            values, entries = z.sort(-1, descending=True)
        probabilities = values.softmax(-1)
        top_p = self.top_p[:rows].view(lead)
        outside = (probabilities.cumsum(-1) - probabilities >= top_p) & (top_p < 1)
        min_p = self.min_p[:rows].view(lead)
        outside |= probabilities < min_p * probabilities[..., :1]
        drawn_at = (values + _noise(self.seed[:rows], index, entries)).masked_fill(outside, float("-inf")).argmax(-1)
        drawn = entries.gather(-1, drawn_at[..., None])[..., 0]
        return torch.where((temperature > 0).view(-1, *[1] * (best.dim() - 1)), drawn, best)
