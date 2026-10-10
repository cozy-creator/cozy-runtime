"""Compile SM120 kernels without loading them or initializing a CUDA context."""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
source = Path(__file__).parent / "sage3_fixed_probe/kernels.py"
spec = importlib.util.spec_from_file_location("fixed_kernels", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
rows = []
for n in (129, 109225):
    tiles = triton.cdiv(n, 256)
    padded = triton.cdiv(n, 128) * 128
    recipes = [
        (
            m.partial_mean,
            {"X": "*bf16", "P": "*fp32", "stride_t": "i32"},
            {"N": n, "D": 128, "TILES": tiles, "BT": 256, "BC": 32},
            8,
        )
    ]
    for denominator in (n, padded):
        recipes.append(
            (
                m.finish_mean,
                {"P": "*fp32", "M": "*bf16"},
                {
                    "D": 128,
                    "TILES": tiles,
                    "DENOMINATOR": denominator,
                    "REDUCE": triton.next_power_of_2(tiles),
                    "BC": 32,
                },
                8,
            )
        )
    recipes.append(
        (
            m.center_and_correct,
            {
                **{x: "*bf16" for x in ("Q", "K", "V", "QM", "KM", "QO", "KO", "VO")},
                "DELTA": "*fp32",
                "stride_t": "i32",
            },
            {"N": n, "PADDED": padded, "D": 128, "BT": 16},
            4,
        )
    )
    for function, signature, constants, warps in recipes:
        started = time.monotonic()
        binary = triton.compile(
            ASTSource(function, signature, constexprs=constants),
            target=GPUTarget("cuda", 120, 32),
            options={"num_warps": warps, "enable_fp_fusion": False},
        )
        rows.append(
            {
                "kernel": function.__name__,
                "tokens": n,
                "constants": constants,
                "signature": signature,
                "warps": warps,
                "compiled_seconds": time.monotonic() - started,
                "hash": binary.metadata.hash,
                "shared_bytes": binary.metadata.shared,
                "ptx_sha256": hashlib.sha256(binary.asm["ptx"].encode()).hexdigest(),
                "binary_sha256": hashlib.sha256(binary.asm["cubin"]).hexdigest(),
            }
        )
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(
    json.dumps(
        {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "triton": triton.__version__,
            "target": "sm120",
            "GPU_context_or_launch": False,
            "kernels": rows,
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps({"compiled": len(rows), "gpu_used": False}))
