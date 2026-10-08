"""CPU numerical oracle for the experimental Sol tile arithmetic.

GPU admission imports this oracle. The optional interpreter check is diagnostic:
Triton3.8's CPU interpreter does not faithfully implement BF16 dot or CUDA FP8
rounding, so its random-case discrepancy must not become a relaxed GPU tolerance.
"""

from __future__ import annotations

import json
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


def check_interpreter(case: str, threshold_value: float, sink: int) -> dict[str, Any]:
    from cozy_runtime.internal.attention_fp8_sol_kernel import forward_kernel, quantize_blocks

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(617)
    tokens, heads, dim = 257, 2, 128
    tensors = [torch.randn(1, tokens, heads, dim, generator=generator).bfloat16() for _ in range(3)]
    if case == "extrema":
        for tensor in tensors:
            tensor.fill_(1)
    elif case == "zeros":
        for tensor in tensors:
            tensor.zero_()
    q, k, v = tensors
    before = [t.clone() for t in tensors]
    blocks = math.ceil(tokens / 64)
    codes = [torch.empty_like(t, dtype=torch.float8_e4m3fn) for t in tensors]
    scales = [torch.empty(1, blocks, heads) for _ in tensors]
    for x, y, sc in zip(tensors, codes, scales, strict=True):
        quantize_blocks[(blocks, heads, 1)](x, y, sc, tokens, heads, dim, num_warps=4)
        assert torch.isfinite(y.float()).all() and y.float().abs().max() <= 16
        assert torch.isfinite(sc).all() and (sc > 0).all()
    kc = torch.stack([part.float().mean(1) for part in k.split(64, dim=1)], dim=1).bfloat16()
    vc = torch.stack([part.float().sum(1) for part in v.split(64, dim=1)], dim=1).bfloat16()
    threshold = torch.full((1, blocks, heads), threshold_value)
    output = torch.full_like(q, float("nan"))
    forward_kernel[(1, blocks, heads)](
        codes[0],
        q,
        codes[1],
        codes[2],
        *scales,
        kc,
        vc,
        threshold,
        output,
        dim**-0.5,
        tokens,
        blocks,
        0,
        math.ceil(sink / 64),
        HAS_SINK=sink > 0,
        H=heads,
        D=dim,
        NT=blocks,
        BV=128,
        BLOCK_SIZE=64,
        GROUP_SIZE=32,
        num_warps=4,
        num_stages=1,
    )
    expected = reference(q, codes[1], codes[2], codes[0], scales, kc, vc, threshold, sink)
    error = (output.float() - expected.float()).abs().max().item()
    assert torch.isfinite(output).all(), case
    assert torch.allclose(output, expected, rtol=0.025, atol=0.015), (case, error)
    assert all(torch.equal(a, b) for a, b in zip(tensors, before, strict=True))
    if case == "extrema":
        assert torch.equal(output, v)
    return {"case": case, "maximum_absolute_error": error, "finite": True}


if __name__ == "__main__":
    rows = [check_interpreter("extrema", -1e6, 257), check_interpreter("zeros", 1e6, 0)]
    print(json.dumps(rows, indent=2))
