"""A language model's next token from its logits, picked as vLLM 0.19.1's sampler picks it
(`vllm/v1/sample/sampler.py`), each setting as its `SamplingParams` defines it; one implementation
for every decoder in the Runtime (`lm`, and the engine's).

After the caller's bans (each banned logit at -inf): the penalties (`penalized`), then a greedy
row's argmax; a sampling row's temperature, min-p, top-k and top-p (`warped`) and a Gumbel-max
draw over what they keep: the argmax of the kept scores plus Gumbel noise hashed from (seed, the
token's index in the row's output, the vocabulary entry). From the same scores the same seed
draws the same token on any device, in any batch, drafted or not; a GPU's scores may differ
slightly run to run, which can turn a near tie. Every setting is a per-row device tensor, nothing is
read back to the host, and every function runs inside a CUDA graph.

vLLM's order, not Transformers' (which cuts min-p after top-k and top-p): the recipes that set
min-p beside top-p (Qwen's enhancers) run on vLLM, and every Transformers recipe we ship has min-p
0, where the two agree.
"""

from __future__ import annotations

import torch
from torch import Tensor

_LOW32 = 0xFFFFFFFF


def penalized(
    scores: Tensor,
    counts: Tensor,
    prompted: Tensor,
    presence: Tensor,
    frequency: Tensor,
    repetition: Tensor,
) -> Tensor:
    """`scores` (`[rows, ..., vocab]`, float32) penalized as vLLM's `apply_penalties`
    (`vllm/model_executor/layers/utils.py:51-89`, `vllm/_custom_ops.py:445-458`): the repetition
    penalty divides a positive logit and multiplies a negative one of each token in the row's
    prompt (`prompted`, bool) or output, then the frequency penalty is subtracted once per time a
    token was generated (`counts`), then the presence penalty once from each generated token.
    `counts` and `prompted` broadcast as `scores`; the penalties are `[rows]`. A neutral penalty
    (0, 0, 1) leaves every score as it is, bit for bit."""
    lead = (-1, *[1] * (scores.dim() - 1))
    seen = counts > 0
    penalties = torch.where(prompted | seen, repetition.view(lead), 1.0)
    scores = scores * torch.where(scores > 0, 1.0 / penalties, penalties)
    return scores - frequency.view(lead) * counts - presence.view(lead) * seen


def warped(
    scores: Tensor, temperature: Tensor, top_p: Tensor, top_k: Tensor, min_p: Tensor
) -> Tensor:
    """What a sampling row draws from: penalized `scores` (`[rows, ..., vocab]`, float32) over
    `temperature`, with every token outside min-p, then top-k, then top-p at -inf, in vLLM's
    `Sampler.sample` order (`sampler.py:177` temperature, `:183-184` min-p, `:187` top-k and
    top-p) and with its operations: min-p drops each probability below `min_p` times the row's
    largest (`MinPLogitsProcessor.apply`, `builtin.py:101-115`); top-k each logit below the
    `top_k`th largest, ties kept, and top-p, over what is left renormalized and summed from the
    least likely, each token whose sum is at most 1 - `top_p`, the likeliest always kept
    (`apply_top_k_top_p_pytorch`, `topk_topp_sampler.py:258-299`). Settings are `[rows]`; `top_k`
    0, `top_p` 1 and `min_p` 0 keep everything."""
    lead = (-1, *[1] * (scores.dim() - 1))
    vocab = int(scores.shape[-1])
    z = scores / temperature.clamp(min=1e-6).view(lead)
    probabilities = z.softmax(-1)
    largest = probabilities.amax(-1, keepdim=True)
    z = z.masked_fill(probabilities < largest * min_p.view(lead), float("-inf"))
    ordered, order = z.sort(-1, descending=False)
    k = torch.where(top_k > 0, top_k, vocab).clamp(max=vocab).view(lead)
    kth = ordered.gather(-1, (vocab - k).expand(*z.shape[:-1], 1))
    ordered = ordered.masked_fill(ordered < kth, float("-inf"))
    below = ordered.softmax(-1).cumsum(-1) <= 1 - top_p.view(lead)
    below[..., -1] = False
    ordered = ordered.masked_fill(below & (top_p.view(lead) < 1), float("-inf"))
    return z.scatter(-1, order, ordered)


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


def noise(seed: Tensor, index: Tensor, entry: Tensor) -> Tensor:
    """Gumbel noise for vocabulary `entry` (`[vocab]`, or `[..., n]` shaped as `index` plus one)
    of the token at output `index` (`[rows, ...]`) of a row seeded `seed` (`[rows]`)."""
    base = _mix(_mix(seed & _LOW32) ^ ((seed >> 32) & _LOW32)).view(-1, *[1] * (index.dim() - 1))
    key = _mix(base ^ _mix(index & _LOW32))
    bits = _mix(key[..., None] ^ _mix(entry))
    # Below 1: float32 rounds the top hash value to 1, whose noise is +inf (that entry would win
    # at any score, or give NaN, which argmax takes, on a score at -inf).
    uniform = (((bits >> 8).float() + 0.5) * (1.0 / (1 << 24))).clamp_(max=1 - 2**-24)
    return -torch.log(-torch.log(uniform))


def pick(
    scores: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    top_k: Tensor,
    min_p: Tensor,
    seed: Tensor,
    index: Tensor,
    sampled: bool = True,
) -> Tensor:
    """Each row's token from its penalized `scores` (`[rows, ..., vocab]`, float32) for output
    position `index` (`scores`' shape without the vocabulary): the argmax where its temperature
    is 0 (vLLM's greedy rows ignore top-p, top-k and min-p), else its draw from `warped`. Without
    `sampled` (no row samples) every row takes the argmax."""
    best = scores.argmax(-1)
    if not sampled:
        return best
    kept = warped(scores, temperature, top_p, top_k, min_p)
    entry = torch.arange(scores.shape[-1], device=scores.device, dtype=torch.long)
    drawn = (kept + noise(seed, index, entry)).argmax(-1)
    return torch.where((temperature > 0).view(-1, *[1] * (best.dim() - 1)), drawn, best)


def prompted(ids: Tensor, mask: Tensor, vocab: int) -> Tensor:
    """Each row's prompt (`ids`, `[rows, prompt]`, its padding out where `mask` is 0) as
    `[rows, vocab]` bits: the tokens the repetition penalty reads beside the output (vLLM's
    `prompt_mask`; its prompts have no padding)."""
    at = torch.where(mask.bool() & (ids < vocab), ids, vocab)
    hit = torch.zeros((int(ids.shape[0]), vocab + 1), dtype=torch.float32, device=ids.device)
    return hit.scatter_(1, at, 1.0)[:, :vocab] > 0
