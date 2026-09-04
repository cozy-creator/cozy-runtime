"""Regional compilation vs whole-model compile vs eager, on ONE model.

`nn.Module.compile()` mutates the block IN PLACE (it installs `_compiled_call_impl`), so a
naive A/B that compiles first and then times "eager" is timing the compiled model twice.
The eager arm here clears that attribute and the regional arm restores it, both outside the
timed region, which is the only way to interleave the two on one weight set.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))
import ab
import anima_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--latent-hw", type=int, default=128)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--inner", type=int, default=2)
    ap.add_argument("--whole", action="store_true", help="also compile the whole model")
    args = ap.parse_args()

    print(ab.device_note())
    tokens = (args.latent_hw // 2) ** 2
    print(f"anima CosmosTransformer3DModel, num_layers={args.layers}, {tokens} tokens, bf16, warm")

    model = anima_model.build(num_layers=args.layers)
    kwargs = anima_model.inputs(latent_hw=args.latent_hw)

    def call() -> None:
        with torch.no_grad():
            model(**kwargs)

    with torch.no_grad():
        reference = model(**kwargs)[0].float().clone()

    blocks = list(model.transformer_blocks)
    model._repeated_blocks = ["CosmosTransformerBlock"]
    start = time.perf_counter()
    model.compile_repeated_blocks(fullgraph=True)
    with torch.no_grad():
        out = model(**kwargs)[0]
    torch.cuda.synchronize()
    regional_cold = time.perf_counter() - start
    diff = (out.float() - reference).abs()
    print(
        f"  regional: cold compile {regional_cold:.1f} s   max|Δ| {diff.max().item():.3e}  "
        f"mean|Δ|/mean|ref| {(diff.mean() / reference.abs().mean()).item():.3e}"
    )
    del out
    impls = [b._compiled_call_impl for b in blocks]

    def to_eager() -> None:
        for block in blocks:
            block._compiled_call_impl = None

    def to_regional() -> None:
        for block, impl in zip(blocks, impls, strict=True):
            block._compiled_call_impl = impl

    arms: list[tuple[str, Any, Any]] = [
        ("eager", call, to_eager),
        ("compile_repeated_blocks", call, to_regional),
    ]

    if args.whole:
        to_eager()
        torch._dynamo.reset()
        start = time.perf_counter()
        whole = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            out = whole(**kwargs)[0]
        torch.cuda.synchronize()
        whole_cold = time.perf_counter() - start
        diff = (out.float() - reference).abs()
        print(
            f"  whole:    cold compile {whole_cold:.1f} s   max|Δ| {diff.max().item():.3e}  "
            f"mean|Δ|/mean|ref| {(diff.mean() / reference.abs().mean()).item():.3e}"
        )
        del out

        def whole_call() -> None:
            with torch.no_grad():
                whole(**kwargs)

        arms.append(("torch.compile(whole)", whole_call, to_eager))

    torch.cuda.empty_cache()
    results = ab.interleaved(arms, reps=args.reps, inner=args.inner)
    ab.report(
        f"anima transformer forward ({args.layers} blocks, {tokens} tokens)", results, "eager"
    )
    print(f"  peak CUDA memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
