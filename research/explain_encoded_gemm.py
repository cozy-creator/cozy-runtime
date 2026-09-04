"""Does the OTHER hand-written kernel — the `encoded_gemm` fp8 leaf — survive tracing?

`RowwiseNativeLeaf._forward` runs `quantize_activation_rowwise` (a raw `triton.jit` launch
with an eager fallback) and then `torch._scaled_mm`. Both are ordinary traceable things in
principle; this asks Dynamo directly, on a real replaced leaf, so the answer is measured
rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))

from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf, _leaf_class

IN, OUT, TOKENS = 2048, 2048, 512


def build() -> Any:
    replaced = torch.nn.Linear(IN, OUT, bias=False, device="cuda", dtype=torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(0)
    data = (
        torch.randn(OUT, IN, device="cuda", dtype=torch.float32, generator=generator)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    scale = torch.rand(OUT, device="cuda", dtype=torch.float32, generator=generator) + 0.5
    provider = RowwiseNativeLeaf(encoding="cozy.fp8-rowwise/1")
    return provider.leaf(
        torch, {"data": data, "scale": scale}, replaced, torch.bfloat16
    ), _leaf_class(torch)


def main() -> None:
    leaf, _ = build()
    x = torch.randn(1, TOKENS, IN, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        eager = leaf(x)
    print(f"eager output {tuple(eager.shape)} {eager.dtype} stride {tuple(eager.stride())}")

    with torch.no_grad():
        explanation = torch._dynamo.explain(leaf)(x)
    print(
        f"graph_count={explanation.graph_count} breaks={explanation.graph_break_count} "
        f"ops={explanation.op_count}"
    )
    for index, entry in enumerate(explanation.break_reasons):
        print(f"--- break {index}: {entry.reason}")
        for frame in entry.user_stack[-3:]:
            print(f"      {frame.filename}:{frame.lineno}  {frame.name}")
            if frame.line:
                print(f"        | {frame.line.strip()}")

    torch._dynamo.reset()
    try:
        with torch.no_grad():
            compiled = torch.compile(leaf, fullgraph=True)(x)
        diff = (compiled.float() - eager.float()).abs()
        print(f"fullgraph compile OK, max|Δ| {diff.max().item():.3e}")
    except Exception as exc:  # the point is to REPORT the refusal, not to pass it on
        print(f"fullgraph compile REFUSED: {type(exc).__name__}: {str(exc)[:300]}")


if __name__ == "__main__":
    main()
