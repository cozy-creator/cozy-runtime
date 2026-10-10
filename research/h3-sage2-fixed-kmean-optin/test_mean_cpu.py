"""Actual Triton mean kernels on CPU; source contract and layout controls."""

import ast
import os
import unittest
from pathlib import Path

if os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError('Use TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=""')
import torch

from cozy_runtime.internal import attention_sage2_fixed_mean as k

ROOT = Path(__file__).resolve().parents[2]


def launch(value):
    _, n, h, d = value.shape
    tiles = (n + 255) // 256
    partial = torch.empty(h, tiles, d, dtype=torch.float32)
    result = torch.empty(1, 1, h, d, dtype=torch.bfloat16)
    k.partial_mean[(h, 4, tiles)](
        value,
        partial,
        value.stride(1),
        value.stride(2),
        n,
        d,
        tiles,
        256,
        32,
        num_warps=8,
        enable_fp_fusion=False,
    )
    k.finish_mean[(h, 4)](
        partial,
        result,
        d,
        tiles,
        n,
        1 << (tiles - 1).bit_length(),
        32,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return result


class Controls(unittest.TestCase):
    def test_actual_kernel_partition_identity_and_custody(self):
        gen = torch.Generator().manual_seed(581204)
        for n in (129, 257):
            value = torch.randn(1, n, 56, 128, dtype=torch.bfloat16, generator=gen)
            copy = value.clone()
            full = launch(value)
            for degree in (2, 4, 7, 8):
                width = 56 // degree
                parts = [
                    launch(value[:, :, start : start + width].contiguous())
                    for start in range(0, 56, width)
                ]
                self.assertTrue(
                    torch.equal(torch.cat(parts, dim=2).view(torch.uint8), full.view(torch.uint8)),
                    (n, degree),
                )
            self.assertTrue(torch.equal(value, copy))

    def test_denominator_uses_live_tokens(self):
        for n in (1, 129, 256, 257):
            value = torch.full((1, n, 2, 128), 3, dtype=torch.bfloat16)
            self.assertTrue(torch.all(launch(value) == 3), n)

    def test_no_head_dependent_reduction_specialization(self):
        tree = ast.parse(
            (ROOT / "src/cozy_runtime/internal/attention_sage2_fixed_mean.py").read_text()
        )
        fs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("partial_mean", "finish_mean"):
            self.assertNotIn("heads", [a.arg for a in fs[name].args.args])
        text = ast.unparse(fs["partial_mean"].decorator_list[0])
        self.assertIn("do_not_specialize=['stride_t', 'stride_h']", text)
        self.assertIn("do_not_specialize_on_alignment=['stride_t', 'stride_h']", text)

    def test_large_packed_address_widens_before_stride_multiplication(self):
        # No giant allocation: the same arithmetic reaches past signed int32
        # for an H3 packed QKV row. Kernel IR is separately compiled and retained.
        tokens = torch.tensor([0, 109224], dtype=torch.int64)
        stride = 21504
        offset = tokens * stride + 55 * 128 + 127
        self.assertGreater(int(offset[-1]), 2**31 - 1)
        self.assertEqual(int(offset[-1]), 109224 * 21504 + 55 * 128 + 127)
        tree = ast.parse(
            (ROOT / "src/cozy_runtime/internal/attention_sage2_fixed_mean.py").read_text()
        )
        partial = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "partial_mean"
        )
        assigned = {
            target.id: node.value
            for node in partial.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for name in ("token", "head"):
            self.assertTrue(ast.unparse(assigned[name]).endswith(".to(tl.int64)"))

    def test_strided_packed_key_has_same_means(self):
        generator = torch.Generator().manual_seed(581204)
        packed = torch.randn(1, 257, 3, 56, 128, dtype=torch.bfloat16, generator=generator)
        key = packed[:, :, 1]
        self.assertFalse(key.is_contiguous())
        self.assertTrue(torch.equal(launch(key), launch(key.contiguous())))
        for degree in (2, 4, 7, 8):
            width = 56 // degree
            actual = torch.cat(
                [launch(key[:, :, i : i + width]) for i in range(0, 56, width)], dim=2
            )
            self.assertTrue(torch.equal(actual, launch(key)))


if __name__ == "__main__":
    unittest.main()
