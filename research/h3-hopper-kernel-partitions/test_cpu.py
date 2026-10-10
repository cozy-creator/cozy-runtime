"""Actual Triton mean kernels on CPU; source contract and layout controls."""

import ast
import importlib.util
import os
import unittest
from pathlib import Path

if os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError('Use TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=""')
import torch

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "fixed_mean_kernels", ROOT / "sage2_fixed_kmean/kernels.py"
)
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)


def launch(value):
    _, n, h, d = value.shape
    tiles = (n + 255) // 256
    partial = torch.empty(h, tiles, d, dtype=torch.float32)
    result = torch.empty(1, 1, h, d, dtype=torch.bfloat16)
    k.partial_mean[(h, 4, tiles)](
        value, partial, value.stride(1), n, d, tiles, 256, 32, num_warps=8, enable_fp_fusion=False
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
        tree = ast.parse((ROOT / "sage2_fixed_kmean/kernels.py").read_text())
        fs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("partial_mean", "finish_mean"):
            self.assertNotIn("heads", [a.arg for a in fs[name].args.args])
        text = ast.unparse(fs["partial_mean"].decorator_list[0])
        self.assertIn("do_not_specialize=['stride_t']", text)
        self.assertIn("do_not_specialize_on_alignment=['stride_t']", text)

    def test_large_packed_address_widens_before_stride_multiplication(self):
        # No giant allocation: the same arithmetic reaches past signed int32
        # for an H3 packed QKV row. Kernel IR is separately compiled and retained.
        tokens = torch.tensor([0, 109224], dtype=torch.int64)
        stride = 21504
        offset = tokens * stride + 55 * 128 + 127
        self.assertGreater(int(offset[-1]), 2**31 - 1)
        self.assertEqual(int(offset[-1]), 109224 * 21504 + 55 * 128 + 127)
        tree = ast.parse((ROOT / "sage2_fixed_kmean/kernels.py").read_text())
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

    def test_replay_calls_exact_pinned_helpers_and_sm90_entry(self):
        # These are the selected statements from the recorded upstream core function.
        tree = ast.parse((ROOT / "sage2_fixed_kmean/adapter.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stages")
        calls = [
            n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]
        quant = next(n for n in calls if n.func.attr == "per_thread_int8_triton")
        self.assertEqual([ast.unparse(a) for a in quant.args], ["q", "k", "km"])
        self.assertEqual(
            {x.arg: ast.literal_eval(x.value) for x in quant.keywords},
            {"tensor_layout": "NHD", "BLKQ": 64, "WARPQ": 16, "BLKK": 128, "WARPK": 128},
        )
        pv = next(n for n in calls if n.func.attr == "per_channel_fp8")
        self.assertEqual(
            {x.arg: ast.literal_eval(x.value) for x in pv.keywords},
            {"tensor_layout": "NHD", "smooth_v": False},
        )
        entry = next(
            n for n in calls if n.func.attr == "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf"
        )
        self.assertEqual(
            [ast.unparse(a) for a in entry.args],
            ["qi", "ki", "vf", "o", "qs", "ks", "vs", "0", "0", "3", "128 ** (-0.5)", "0"],
        )


if __name__ == "__main__":
    unittest.main()
