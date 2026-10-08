"""CPU numerical oracle for the experimental Sol tile arithmetic.

GPU admission imports this oracle. The optional interpreter check is diagnostic:
Triton3.8's CPU interpreter does not faithfully implement BF16 dot or CUDA FP8
rounding, so its random-case discrepancy must not become a relaxed GPU tolerance.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def reference(
    q: Any, k8: Any, v8: Any, q8: Any, scales: Any, kc: Any, vc: Any, threshold: Any, sink: int
) -> Any:
    result = torch.empty_like(q)
    tokens, heads, dim = q.shape[1:]
    blocks = math.ceil(tokens / 64)
    factor = dim**-0.5 * math.log2(math.e)
    for h in range(heads):
        for qb in range(blocks):
            lo, hi = qb * 64, min(tokens, (qb + 1) * 64)
            query = q[0, lo:hi, h].float()
            score = query @ kc[0, :, h].float().T * factor
            exact = score.mean(0) > threshold[0, qb, h]
            indices = torch.arange(blocks)
            exact |= (indices - qb).abs() <= 1
            exact |= indices < math.ceil(sink / 64)
            maximum = torch.full((hi - lo,), -torch.inf)
            denominator = torch.zeros(hi - lo)
            output = torch.zeros(hi - lo, dim)
            for group_start in range(0, blocks, 32):
                group_stop = min(blocks, group_start + 32)
                selected_exact = exact[group_start:group_stop]
                summary = ~selected_exact
                if summary.any():
                    selected = score[:, group_start:group_stop][:, summary]
                    new_max = torch.maximum(maximum, selected.max(1).values)
                    alpha = torch.exp2(maximum - new_max)
                    probability = torch.exp2(selected - new_max[:, None])
                    output = (
                        output * alpha[:, None]
                        + probability.bfloat16().float()
                        @ vc[0, group_start:group_stop, h][summary].float()
                    )
                    lengths = torch.tensor(
                        [
                            min(64, tokens - int(i) * 64)
                            for i in indices[group_start:group_stop][summary]
                        ]
                    )
                    denominator = denominator * alpha + (probability * lengths).sum(1)
                    maximum = new_max
                for kb in indices[group_start:group_stop][selected_exact].tolist():
                    start, stop = kb * 64, min(tokens, (kb + 1) * 64)
                    scores = (
                        (q8[0, lo:hi, h].float() @ k8[0, start:stop, h].float().T).half().float()
                    )
                    scores *= scales[0][0, qb, h] * scales[1][0, kb, h] * factor
                    new_max = torch.maximum(maximum, scores.max(1).values)
                    alpha = torch.exp2(maximum - new_max)
                    probability = torch.exp2(scores - new_max[:, None])
                    denominator = denominator * alpha + probability.sum(1)
                    p_scale = probability.max(1).values.clamp_min(1e-6) / 16
                    p8 = (probability / p_scale[:, None]).clamp(max=16).to(torch.float8_e4m3fn)
                    partial = (p8.float() @ v8[0, start:stop, h].float()).half().float()
                    output = output * alpha[:, None] + partial * (
                        p_scale[:, None] * scales[2][0, kb, h]
                    )
                    maximum = new_max
            result[0, lo:hi, h] = (output / denominator[:, None]).bfloat16()
    return result


def quantize_cpu(tensor: Any) -> tuple[Any, Any]:
    parts, scales = [], []
    for block in tensor.split(64, dim=1):
        scale = (block.float().abs().amax(dim=(1, 3)) / 16).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        parts.append(
            (block.float() / scale[:, None, :, None]).clamp(-16, 16).to(torch.float8_e4m3fn)
        )
        scales.append(scale)
    return torch.cat(parts, dim=1), torch.stack(scales, dim=1)
