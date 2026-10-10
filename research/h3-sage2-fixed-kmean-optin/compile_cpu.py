"""Compile the fixed K-mean kernels for SM90 without opening a CUDA context."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
source = (
    Path(__file__).resolve().parents[2] / "src/cozy_runtime/internal/attention_sage2_fixed_mean.py"
)
spec = importlib.util.spec_from_file_location("mean_kernels", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
rows = []
for n in (129, 109225):
    tiles = triton.cdiv(n, 256)
    for fn, signature, constants in [
        (
            m.partial_mean,
            {"X": "*bf16", "P": "*fp32", "stride_t": "i32", "stride_h": "i32"},
            {"N": n, "D": 128, "TILES": tiles, "BT": 256, "BC": 32},
        ),
        (
            m.finish_mean,
            {"P": "*fp32", "M": "*bf16"},
            {
                "D": 128,
                "TILES": tiles,
                "DENOMINATOR": n,
                "REDUCE": triton.next_power_of_2(tiles),
                "BC": 32,
            },
        ),
    ]:
        binary = triton.compile(
            ASTSource(fn, signature, constexprs=constants),
            target=GPUTarget("cuda", 90, 32),
            options={"num_warps": 8, "enable_fp_fusion": False},
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        ir_path = args.out.parent / f"{fn.__name__}-N{n}.ttir"
        ir_path.write_text(binary.asm["ttir"])
        rows.append(
            {
                "name": fn.__name__,
                "tokens": n,
                "constants": constants,
                "shared_bytes": binary.metadata.shared,
                "ttir_path": str(ir_path),
                "ptx_sha256": hashlib.sha256(binary.asm["ptx"].encode()).hexdigest(),
                "cubin_sha256": hashlib.sha256(binary.asm["cubin"]).hexdigest(),
            }
        )
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(
    json.dumps(
        {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "target": "SM90",
            "GPU_context_or_launch": False,
            "rows": rows,
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps({"compiled": len(rows), "gpu_used": False}))
