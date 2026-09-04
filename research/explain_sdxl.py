"""Graph breaks in the SDXL UNet at the shape the package actually denoises."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import sdxl_model


def main() -> None:
    model = sdxl_model.build()
    kwargs = sdxl_model.inputs(batch=2)
    with torch.no_grad():
        explanation = torch._dynamo.explain(model)(**kwargs)
    print(f"graph_count       = {explanation.graph_count}")
    print(f"graph_break_count = {explanation.graph_break_count}")
    print(f"op_count          = {explanation.op_count}")
    for index, entry in enumerate(explanation.break_reasons):
        print(f"--- break {index}: {entry.reason}")
        for frame in entry.user_stack[-4:]:
            print(f"      {frame.filename}:{frame.lineno}  {frame.name}")
            if frame.line:
                print(f"        | {frame.line.strip()}")


if __name__ == "__main__":
    main()
