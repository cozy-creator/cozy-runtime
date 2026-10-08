"""Explicit rented-SM120 tiny-kernel admission; never run as a local/default test.

Requires this private Runtime plus Runtime's resolved sol_attn kernel site on sys.path.
It does not time a model or substitute for the full 15-second benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib import import_module
from pathlib import Path

import torch

reference = import_module("fp8_sol_cpu").reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--kernel-site", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.extend(args.kernel_site)
    from cozy_runtime.internal import attention_fp8_sol, attention_fp8_sol_kernel, attention_sol
    prepare = import_module("sol_attn.preprocess").prepare

    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    assert torch.cuda.get_device_capability(device) == (12, 0), "requires rented SM120"
    attention_sol._pin_preprocess()
    rows = []
    for name, tokens, heads, sink, threshold_value in [
        ("extrema", 65, 1, 65, -1e6),
        ("zeros", 65, 1, 0, 1e6),
        ("all_exact", 257, 2, 0, -1e6),
        ("mixed_tail", 257, 2, 65, 1e6),
        ("group_tail", 2113, 1, 65, 1e6),
        ("actual_threshold", 257, 14, 65, None),
    ]:
        generator = torch.Generator().manual_seed(617)
        tensors = [
            torch.randn(1, tokens, heads, 128, generator=generator).bfloat16().to(device)
            for _ in range(3)
        ]
        if name in {"zeros", "extrema"}:
            for tensor in tensors:
                tensor.fill_(0 if name == "zeros" else 1)
        before = [x.clone() for x in tensors]
        blocks = math.ceil(tokens / 64)
        codes = [torch.empty_like(x, dtype=torch.float8_e4m3fn) for x in tensors]
        scales = [torch.empty(1, blocks, heads, device=device) for _ in tensors]
        for source, code, scale in zip(tensors, codes, scales, strict=True):
            attention_fp8_sol_kernel.quantize_blocks[(blocks, heads, 1)](
                source, code, scale, tokens, heads, 128, num_warps=4
            )
        assert all(torch.isfinite(x.float()).all() and x.float().abs().max() <= 16 for x in codes)
        kc, vc, threshold = prepare(*tensors, tau=1.0, scale=128**-0.5, thresh_type="exact")
        if threshold_value is not None:
            threshold.fill_(threshold_value)
        output = torch.empty_like(tensors[0])
        compiled = attention_fp8_sol_kernel.forward_kernel[(1, blocks, heads)](
            codes[0],
            tensors[0],
            codes[1],
            codes[2],
            *scales,
            kc,
            vc,
            threshold,
            output,
            128**-0.5,
            tokens,
            blocks,
            0,
            math.ceil(sink / 64),
            HAS_SINK=sink > 0,
            H=heads,
            D=128,
            NT=blocks,
            BV=128,
            BLOCK_SIZE=64,
            GROUP_SIZE=32,
            num_warps=4,
            num_stages=1,
        )
        ptx = compiled.asm["ptx"]
        assert "f16.e4m3.e4m3.f16" in ptx and "f32.e4m3.e4m3.f32" not in ptx
        host_output = output.cpu()
        expected = reference(
            tensors[0].cpu(),
            codes[1].cpu(),
            codes[2].cpu(),
            codes[0].cpu(),
            [x.cpu() for x in scales],
            kc.cpu(),
            vc.cpu(),
            threshold.cpu(),
            sink,
        )
        difference = host_output.float() - expected.float()
        relative = float(difference.norm() / expected.float().norm().clamp_min(1e-8))
        maximum = float(difference.abs().max())
        assert torch.isfinite(host_output).all(), name
        assert relative < 0.03 and maximum < 0.03, (name, relative, maximum)
        assert all(torch.equal(a, b) for a, b in zip(tensors, before, strict=True))
        if name in {"zeros", "extrema"}:
            assert torch.equal(output, tensors[2])
        if name == "actual_threshold":
            q, k, v = tensors
            assert torch.equal(output, attention_fp8_sol.sparse(q, k, v, 128**-0.5, sink))
        rows.append(
            {
                "case": name,
                "tokens": tokens,
                "heads": heads,
                "relative_l2_to_cpu_oracle": relative,
                "maximum_absolute_error": maximum,
                "fp16_mma": True,
                "finite": True,
            }
        )
        args.out.write_text(
            json.dumps({"gpu": torch.cuda.get_device_name(device), "cases": rows}, indent=2) + "\n"
        )
    print(args.out.read_text())


if __name__ == "__main__":
    main()
