"""Question 2d: which graph breaks does Dynamo hit on the Anima transformer, and why.

Runs `torch._dynamo.explain` over the stock diffusers module and (optionally) over the
shipping fused Anima plan, printing every break with its reason and user stack.
"""

from __future__ import annotations

import argparse
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import anima_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--latent-hw", type=int, default=128)
    ap.add_argument("--plan", choices=["raw", "op"], default=None)
    args = ap.parse_args()

    model = anima_model.build(num_layers=args.layers)
    if args.plan:
        anima_model.apply_fused_plan(model, via=args.plan, strict=args.layers == 28)
    kwargs = anima_model.inputs(latent_hw=args.latent_hw)

    with torch.no_grad():
        explanation = torch._dynamo.explain(model)(**kwargs)

    print(f"graph_count           = {explanation.graph_count}")
    print(f"graph_break_count     = {explanation.graph_break_count}")
    print(f"op_count              = {explanation.op_count}")
    print()
    for i, (reason, stack) in enumerate(
        zip(
            [b.reason for b in explanation.break_reasons],
            [b.user_stack for b in explanation.break_reasons],
            strict=True,
        )
    ):
        print(f"--- break {i}: {reason}")
        for frame in stack[-6:]:
            print(f"      {frame.filename}:{frame.lineno}  {frame.name}")
            if frame.line:
                print(f"        | {frame.line.strip()}")
        print()


if __name__ == "__main__":
    main()
