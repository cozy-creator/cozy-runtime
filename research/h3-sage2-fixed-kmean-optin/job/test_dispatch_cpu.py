"""Exercise the actual Diffusers registered dispatch used by the job, on CPU SDPA."""

from __future__ import annotations

import unittest

import torch
from registered_probe.probe import OPTIONS, canonical, dispatch, head_groups

from cozy_runtime.internal import attention
from cozy_runtime.internal.encoding import DeviceFacts


class DispatchControl(unittest.TestCase):
    def test_actual_registered_dispatch_and_group_report_with_rectangular_queries(self):
        ready = attention.pinned("sdpa", DeviceFacts("cpu", "CPU", 0, "", "CPU registry test"))[0]
        generator = torch.Generator().manual_seed(581204)
        values = [
            torch.randn(1, n, 56, 128, dtype=torch.bfloat16, generator=generator)
            for n in (17, 129, 129)
        ]
        actual = dispatch(ready, values)
        expected = ready.candidate.local.forward(*values, **OPTIONS)
        self.assertTrue(torch.equal(actual, expected))
        result = head_groups(torch, values, lambda v: {"output": canonical(dispatch(ready, v))})
        self.assertTrue(result["passed"])
        self.assertEqual([r["local_head_groups"] for r in result["partitions"]], [2, 4, 7, 8])


if __name__ == "__main__":
    unittest.main()
