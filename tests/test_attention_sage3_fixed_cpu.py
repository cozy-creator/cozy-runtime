"""Run separately with Triton interpreter and CPU tensors; never opens local CUDA."""

from __future__ import annotations

import os
import unittest
from typing import Any

if os.environ.get("TRITON_INTERPRET") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise unittest.SkipTest("Explicit CPU interpreter invocation required")

import torch

from cozy_runtime.internal import attention_sage3_fixed as candidate


def run_cpu(values: Any, fixed: bool) -> dict[str, Any]:
    q, k, v = values
    _, n, h, d = q.shape
    padded, tiles = (n + 127) // 128 * 128, (n + 255) // 256
    partial_mean, finish_mean, center = candidate._kernels()
    means = []
    for x, denominator in ((q, padded), (k, n)):
        partial = torch.empty(h, tiles, d, dtype=torch.float32)
        mean = torch.empty(1, h, 1, d, dtype=torch.bfloat16)
        partial_mean[(h, 4, tiles)](
            x,
            partial,
            x.stride(1),
            x.stride(2),
            n,
            d,
            tiles,
            256,
            32,
            num_warps=8,
            enable_fp_fusion=False,
        )
        finish_mean[(h, 4)](
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
    delta = torch.full((1, h, 1, padded), float("nan"), dtype=torch.float32)
    center[(h, padded // 16)](
        q,
        k,
        v,
        *means,
        *out,
        delta,
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        v.stride(1),
        v.stride(2),
        n,
        padded,
        d,
        16,
        fixed,
        num_warps=4,
        enable_fp_fusion=False,
    )
    if not fixed:
        # The kernel must skip the custom correction entirely, not compute and discard it.
        assert torch.isnan(delta).all()
        delta = torch.matmul(means[0], out[1].transpose(-2, -1)).to(torch.float32)
    return {"qm": means[0], "km": means[1], "q": out[0], "k": out[1], "v": out[2], "delta": delta}


class Controls(unittest.TestCase):
    def test_packed_and_transposed_operands_match_contiguous_without_mutation(self) -> None:
        generator = torch.Generator().manual_seed(581204)
        original = [
            torch.randn(1, 129, 14, 128, dtype=torch.bfloat16, generator=generator)
            for _ in range(3)
        ]
        packed = torch.cat(original, dim=2)
        views = list(packed.chunk(3, dim=2))
        mixed = [
            views[0].contiguous(),
            views[1].transpose(1, 2).contiguous().transpose(1, 2),
            views[2],
        ]
        before = packed.clone()
        for fixed in (False, True):
            reference = run_cpu(original, fixed)
            for operands in (views, mixed):
                actual = run_cpu(operands, fixed)
                for name in reference:
                    self.assertTrue(
                        torch.equal(
                            actual[name].view(torch.uint8), reference[name].view(torch.uint8)
                        ),
                        name,
                    )
        self.assertTrue(torch.equal(packed, before))

    def test_strided_head_partitions_preserve_every_intermediate(self) -> None:
        generator = torch.Generator().manual_seed(581205)
        packed = torch.randn(1, 129, 3 * 56, 128, dtype=torch.bfloat16, generator=generator)
        views = list(packed.chunk(3, dim=2))
        for fixed in (False, True):
            reference = run_cpu(views, fixed)
            for degree in (2, 4, 7, 8):
                width = 56 // degree
                pieces = [
                    run_cpu([x[:, :, start : start + width] for x in views], fixed)
                    for start in range(0, 56, width)
                ]
                for name in reference:
                    actual = torch.cat([x[name] for x in pieces], dim=1)
                    self.assertTrue(
                        torch.equal(actual.view(torch.uint8), reference[name].view(torch.uint8)),
                        (fixed, degree, name),
                    )

    def test_mean_denominators_and_padding_keep_the_original_recipe(self) -> None:
        values = [torch.full((1, 129, 2, 128), value, dtype=torch.bfloat16) for value in (2, 3, 4)]
        for fixed in (False, True):
            actual = run_cpu(values, fixed)
            self.assertTrue(
                torch.all(actual["qm"] == torch.tensor(2 * 129 / 256, dtype=torch.bfloat16))
            )
            self.assertTrue(torch.all(actual["km"] == 3))
            self.assertTrue(torch.all(actual["q"][:, :, 129:] == -actual["qm"]))
            self.assertTrue(torch.all(actual["k"] == 0))
            self.assertTrue(torch.all(actual["v"][:, :, 129:] == 0))
            self.assertTrue(torch.all(actual["delta"] == 0))


if __name__ == "__main__":
    unittest.main()
