"""Actual Runtime scope/prefix execution on CPU; no claim of executing a CUDA kernel."""

import unittest

import torch
from sage2_fixed_kmean.metrics import compared
from sage2_fixed_kmean.other_kernels import sparse_scope

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout
from cozy_runtime.internal import attention_sol as sol


class PolicyControls(unittest.TestCase):
    def test_real_scope_restores_dense_and_site_after_exception(self):
        dense, identity, site, layout = (
            sol._DENSE,
            sol._DENSE_IDENTITY,
            sol._SITE.get(),
            _ACTIVE_LAYOUT.get(),
        )

        def replacement(**kwargs):
            return kwargs["query"].clone()

        with (
            self.assertRaisesRegex(RuntimeError, "intentional"),
            sparse_scope(sol, 19, 3, 4, replacement),
        ):
            self.assertIs(sol._DENSE, replacement)
            self.assertEqual(_ACTIVE_LAYOUT.get().step, 4)
            self.assertEqual(sol._SITE.get().path, "blocks.2.attn")
            raise RuntimeError("intentional")
        self.assertIs(sol._DENSE, dense)
        self.assertEqual(sol._DENSE_IDENTITY, identity)
        self.assertIs(sol._SITE.get(), site)
        self.assertIs(_ACTIVE_LAYOUT.get(), layout)

    def test_real_execute_preserves_dense_prefix_at_both_sparse_steps(self):
        q = (
            torch.arange(19 * 4 * 128, dtype=torch.float32)
            .reshape(1, 19, 4, 128)
            .to(torch.bfloat16)
        )
        k = q + 2
        v = q + 4
        custody = [x.clone() for x in (q, k, v)]

        def sparse(q, k, v, scale, sink):
            self.assertEqual(sink, 3)
            return v.clone()

        def dense(*, query, key, value, scale):
            self.assertEqual(key.shape, k.shape)
            return query.clone()

        for step in (4, 5):
            with sol.observing() as counts:
                out = sol._execute(
                    q,
                    k,
                    v,
                    128**-0.5,
                    AttentionLayout(19, 3, step, 4),
                    sol.Site("diagnostic", "blocks.2.attn"),
                    sparse=sparse,
                    dense=dense,
                )
            self.assertEqual(
                counts, {"sparse": 1, "dense_step": 0, "dense_path": 0, "dense_prefix": 1}
            )
            self.assertTrue(torch.equal(out[:, :3], q[:, :3]))
            self.assertTrue(torch.equal(out[:, 3:], v[:, 3:]))
            for before, after in zip(custody, (q, k, v), strict=True):
                self.assertTrue(torch.equal(before, after))

    def test_metrics_distinguish_partition_drift_from_nonfinite(self):
        a = torch.ones(2, 3, 4, dtype=torch.bfloat16)
        b = a.clone()
        b[1, 2, 3] += 1
        metric = compared(torch, a, b)
        self.assertEqual(metric["different_elements"], 1)
        self.assertEqual(metric["first_different_element"], 23)
        self.assertEqual(metric["max_absolute_difference"], 1)
        self.assertEqual(metric["nonfinite_elements"], 0)
        b[0, 0, 0] = float("nan")
        self.assertEqual(compared(torch, a, b)["nonfinite_elements"], 1)


if __name__ == "__main__":
    unittest.main()
