"""Actual Triton interpreter on CPU tensors; never opens the local CUDA device."""

import ast
import importlib.util
import os
import unittest
from pathlib import Path

if os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError('Run CPU controls with TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=""')
import torch

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "fixed_kernels", ROOT / "sage3_fixed_probe/kernels.py"
)
kernels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernels)


def launch(values):
    q, k, v = values
    _, n, h, d = q.shape
    padded = ((n + 127) // 128) * 128
    tiles = (n + 255) // 256
    means = []
    for value, denominator in ((q, padded), (k, n)):
        partial = torch.empty(h, tiles, d, dtype=torch.float32)
        mean = torch.empty(1, h, 1, d, dtype=torch.bfloat16)
        kernels.partial_mean[(h, d // 32, tiles)](
            value,
            partial,
            value.stride(1),
            n,
            d,
            tiles,
            256,
            32,
            num_warps=8,
            enable_fp_fusion=False,
        )
        kernels.finish_mean[(h, d // 32)](
            partial,
            mean,
            d,
            tiles,
            denominator,
            1 << (tiles - 1).bit_length(),
            32,
            num_warps=8,
            enable_fp_fusion=False,
        )
        means.append(mean)
    out = [torch.empty(1, h, padded, d, dtype=torch.bfloat16) for _ in range(3)]
    delta = torch.empty(1, h, 1, padded, dtype=torch.float32)
    kernels.center_and_correct[(h, (padded + 15) // 16)](
        q,
        k,
        v,
        *means,
        *out,
        delta,
        q.stride(1),
        n,
        padded,
        d,
        16,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return {"qm": means[0], "km": means[1], "q": out[0], "k": out[1], "v": out[2], "delta": delta}


class Controls(unittest.TestCase):
    def test_live_k_padded_q_denominators_and_tail_rounding(self):
        for n in (1, 129, 256, 257):
            values = [torch.full((1, n, 2, 128), v, dtype=torch.bfloat16) for v in (2, 3, 4)]
            result = launch(values)
            padded = (n + 127) // 128 * 128
            qm = torch.tensor(2 * n / padded, dtype=torch.bfloat16)
            self.assertTrue(torch.all(result["qm"] == qm))
            self.assertTrue(torch.all(result["km"] == 3))
            self.assertTrue(torch.all(result["k"] == 0))
            self.assertTrue(
                torch.all(result["q"][:, :, :n] == (torch.tensor(2, dtype=torch.bfloat16) - qm))
            )
            self.assertTrue(torch.all(result["q"][:, :, n:] == -qm))
            self.assertTrue(torch.all(result["v"][:, :, :n] == 4))
            self.assertTrue(torch.all(result["v"][:, :, n:] == 0))
            self.assertTrue(torch.all(result["delta"] == 0))

    def test_actual_interpreter_is_exact_across_all_head_partitions(self):
        g = torch.Generator().manual_seed(581204)
        values = [torch.randn(1, 129, 56, 128, dtype=torch.bfloat16, generator=g) for _ in range(3)]
        untouched = [x.clone() for x in values]
        full = launch(values)
        for degree in (2, 4, 7, 8):
            width = 56 // degree
            parts = {name: [] for name in full}
            for start in range(0, 56, width):
                result = launch(
                    [
                        x[:, :, start : start + width].clone(memory_format=torch.contiguous_format)
                        for x in values
                    ]
                )
                for name, x in result.items():
                    parts[name].append(x)
            for name, parts_for_name in parts.items():
                actual = torch.cat(parts_for_name, dim=1)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), full[name].view(torch.uint8)),
                    (degree, name),
                )
        for a, b in zip(values, untouched, strict=True):
            self.assertTrue(torch.equal(a, b))

    def test_kernel_geometry_never_specializes_on_head_count(self):
        tree = ast.parse((ROOT / "sage3_fixed_probe/kernels.py").read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("partial_mean", "finish_mean", "center_and_correct"):
            self.assertNotIn("heads", [a.arg for a in functions[name].args.args])
            self.assertNotIn("H", [a.arg for a in functions[name].args.args])
        for name in ("partial_mean", "center_and_correct"):
            decorators = ast.unparse(functions[name].decorator_list[0])
            self.assertIn("do_not_specialize=['stride_t']", decorators)
            self.assertIn("do_not_specialize_on_alignment=['stride_t']", decorators)

    def test_public_replay_follows_the_actual_upstream_arithmetic(self):
        spec = importlib.util.spec_from_file_location(
            "fixed_stages", ROOT / "sage3_fixed_probe/stages.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Independent upstream expression, CPU only. Exact GPU API bytes are checked at run time.
        values = [torch.randn(1, 129, 2, 128, dtype=torch.bfloat16) for _ in range(3)]
        q, k, v = (x.transpose(1, 2) for x in values)
        k = k.clone(memory_format=torch.contiguous_format)
        k -= k.mean(dim=-2, keepdim=True)
        q, k, v = [
            torch.nn.functional.pad(x, (0, 0, 0, 127), value=0).contiguous() for x in (q, k, v)
        ]
        qm = q.mean(dim=-2, keepdim=True)
        q = q - qm
        delta = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32).contiguous()
        actual = module.upstream_preprocess(*values)
        for name, value in zip(("q", "k", "v", "delta"), (q, k, v, delta), strict=True):
            self.assertTrue(torch.equal(value, actual[name]), name)


if __name__ == "__main__":
    unittest.main()
