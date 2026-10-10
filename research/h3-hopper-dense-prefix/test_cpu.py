"""Actual CPU SDPA layout control; not CUDA backend qualification."""

import unittest

import torch
from dense_prefix.metrics import compared
from dense_prefix.probe import OPTIONS, H, canonical, fresh, guarded

from cozy_runtime.internal import attention_upstream as upstream


class Controls(unittest.TestCase):
    def test_actual_sdpa_short_and_equal_query_layouts(self):
        generator = torch.Generator().manual_seed(581204)
        for queries in (17, 129):
            q = torch.randn(1, queries, H, 128, dtype=torch.bfloat16, generator=generator)
            k = torch.randn(1, 129, H, 128, dtype=torch.bfloat16, generator=generator)
            v = torch.randn(1, 129, H, 128, dtype=torch.bfloat16, generator=generator)
            values = [q, k, v]
            before = [a.clone() for a in values]
            expected = canonical(upstream.SDPA.forward(*values, **OPTIONS))
            for groups in (2, 4, 7, 8):
                width = H // groups
                actual = torch.cat(
                    [
                        canonical(
                            upstream.SDPA.forward(
                                *fresh(torch, values, start, start + width), **OPTIONS
                            )
                        )
                        for start in range(0, H, width)
                    ]
                )
                result = compared(torch, actual, expected)
                self.assertTrue(result["bytes_equal"], (queries, groups, result))
                self.assertEqual(result["nonfinite_elements"], 0)
            for a, b in zip(values, before, strict=True):
                self.assertTrue(torch.equal(a, b))

    def test_actual_math_dispatch_is_refused_without_switching(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        values = [torch.ones(1, 3, H, 128, dtype=torch.bfloat16) for _ in range(3)]
        calls = []
        with sdpa_kernel(SDPBackend.MATH):
            self.assertEqual(upstream.SDPA.impl(*values, OPTIONS), "math")
            with self.assertRaisesRegex(RuntimeError, "refusing large score"):
                guarded(upstream.SDPA, values, calls)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
