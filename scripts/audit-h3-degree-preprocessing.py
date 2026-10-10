#!/usr/bin/env python3
"""Check head/row partition invariance on CPU; this is not a GPU video determinism test.

Pass the pinned Sage3 sageattn3/api.py used by Runtime. Only preprocess_qkv is loaded,
with its global-mean branch. The Runtime rowwise quantizer is likewise loaded from
this checkout without importing the serving runtime or initializing a CUDA context.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


def function_from(path: Path, name: str, namespace: dict[str, Any]) -> Any:
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        node,
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"
        ),
        namespace,
    )
    return namespace[name]


def compare(torch: Any, actual: Any, reference: Any) -> dict[str, Any]:
    raw_equal = torch.equal(
        actual.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)
    )
    delta = actual.float() - reference.float()
    return {
        "bytes_equal": raw_equal,
        "different_elements": int((delta != 0).sum()),
        "max_absolute_difference": float(delta.abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sage-api", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    runtime = Path(__file__).resolve().parents[1]
    leaf = runtime / "src/cozy_runtime/internal/encoding/leaves.py"
    sage = function_from(args.sage_api, "preprocess_qkv", {"torch": torch, "F": functional})
    quantize = function_from(
        leaf,
        "quantize_activation_rowwise",
        {
            "PreQuantized": type("UnusedPreQuantized", (), {}),
            "E4M3_MAX": 448.0,
            "ACT_SCALE_FLOOR": 1e-12,
        },
    )
    cases = []
    for dtype in (torch.bfloat16, torch.float32):
        for tokens in (129, 1031):
            random = torch.Generator(device="cpu").manual_seed(581204)
            tensors = [
                torch.randn(1, 56, tokens, 128, generator=random, device="cpu", dtype=dtype)
                for _ in range(3)
            ]
            expected = sage(*(x.clone() for x in tensors), per_block_mean=False)
            for degree in (2, 4, 7, 8):
                outputs = [
                    sage(
                        *(x.tensor_split(degree, dim=1)[rank].clone() for x in tensors),
                        per_block_mean=False,
                    )
                    for rank in range(degree)
                ]
                result = {
                    label: compare(
                        torch, torch.cat([row[index] for row in outputs], dim=1), expected[index]
                    )
                    for index, label in enumerate(
                        ("q_centered", "k_centered", "v_padded", "delta_s")
                    )
                }
                cases.append(
                    {
                        "operation": "Sage3 global-mean preprocessing",
                        "dtype": str(dtype),
                        "tokens": tokens,
                        "degree": degree,
                        "outputs": result,
                    }
                )
            activations = torch.randn(1, tokens, 5376, generator=random, device="cpu", dtype=dtype)
            reference = quantize(torch, activations)
            for degree in (2, 4, 7, 8):
                pieces = [quantize(torch, x) for x in activations.tensor_split(degree, dim=1)]
                result = {
                    label: compare(
                        torch, torch.cat([x[index] for x in pieces], dim=0), reference[index]
                    )
                    for index, label in enumerate(("fp8_payload", "row_scale"))
                }
                cases.append(
                    {
                        "operation": "Runtime eager rowwise activation quantization",
                        "dtype": str(dtype),
                        "tokens": tokens,
                        "degree": degree,
                        "outputs": result,
                    }
                )
    report = {
        "cpu_only": True,
        "torch": torch.__version__,
        "cases": cases,
        "all_bytes_equal": all(x["bytes_equal"] for c in cases for x in c["outputs"].values()),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.sage_api, leaf)
        },
        "limits": [
            "Tests preprocessing with fixed identical input tensors, not denoising, "
            "GPU GEMMs or NVFP4 quantizers.",
            "The CPU eager FP8 cast does not qualify the CUDA Triton cast; "
            "GPU implementations must be compared separately.",
            "Passing does not establish end-to-end video/audio equality or identify "
            "the retained 4-vs-8 video divergence.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "all_bytes_equal": report["all_bytes_equal"]}))


if __name__ == "__main__":
    main()
