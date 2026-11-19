"""A step's token choice on the device (xs-213 phase 2): the Runtime's one sampler
(`cozy_runtime.sampling`, vLLM 0.19.1's order, as `lm` samples) over each row's settings, held as
`[max_rows]` device tensors a step's graph reads; nothing is read back to the host.

A drafted token is kept only where it is the model's own draw at its index with the same noise, so
drafting samples the same distribution as an undrafted call. The scores the noise is added to are
not bit for bit the same between calls (a verify pass against a one-token step, kernels' reduction
order), so where two entries nearly tie the draw can turn.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from cozy_runtime import sampling


@dataclass(frozen=True)
class Sampling:
    """Each row's decoding, `[max_rows]` device tensors a step's graph holds, and a call's
    constants: `sampled`, whether any row samples; `penalized`, whether any row has a penalty."""

    temperature: Tensor
    top_p: Tensor
    top_k: Tensor
    min_p: Tensor
    presence: Tensor
    frequency: Tensor
    repetition: Tensor
    seed: Tensor
    sampled: bool
    penalized: bool

    def penalties(self, logits: Tensor, rows: int, counts: Tensor, prompt: Tensor) -> Tensor:
        """`logits` (`[rows, ..., vocab]`, f32) after each row's penalties (`sampling.penalized`):
        `prompt` (bool) and `counts` (each token's count in the output so far) broadcast as `logits`."""
        if not self.penalized:
            return logits
        return sampling.penalized(
            logits, counts, prompt, self.presence[:rows], self.frequency[:rows], self.repetition[:rows]
        )

    def pick(self, scores: Tensor, rows: int, index: Tensor) -> Tensor:
        """Each of `rows` rows' token from penalized `scores` (`[rows, ..., vocab]`, f32) at
        output positions `index` (scores' shape without the vocabulary): `sampling.pick`."""
        return sampling.pick(
            scores, self.temperature[:rows], self.top_p[:rows], self.top_k[:rows], self.min_p[:rows],
            self.seed[:rows], index, self.sampled,
        )  # fmt: skip
