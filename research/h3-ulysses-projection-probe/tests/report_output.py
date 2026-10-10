"""Real Diffusers CP pre/post-forward hooks with the probe's report byte carrier."""

import unittest

import torch
from diffusers.hooks.context_parallel import apply_context_parallel
from diffusers.models._modeling_parallel import ContextParallelConfig
from diffusers.models.modeling_utils import ModelMixin
from projection_probe.model import read_report, report_tensor


class ReportComponent(ModelMixin):
    def forward(self, report):
        return report_tensor(report)


class ReportOutputTest(unittest.TestCase):
    def test_real_empty_cp_plan_round_trips_the_report_tensor(self):
        module = torch.nn.Identity()
        apply_context_parallel(module, ContextParallelConfig(ulysses_degree=2), {"": {}})
        report = {"candidate_partition_exact": True, "ranks": [{"rank": 0}, {"rank": 1}]}
        # This is the actual failure in5245, from the installed Diffusers post-forward hook.
        with self.assertRaisesRegex(ValueError, "Expected output to be a tensor"):
            module(report)
        encoded_component = ReportComponent()
        apply_context_parallel(encoded_component, ContextParallelConfig(ulysses_degree=2), {"": {}})
        encoded = encoded_component(report)
        self.assertEqual(encoded.dtype, torch.uint8)
        self.assertEqual(encoded.device.type, "cpu")
        self.assertEqual(read_report(encoded), report)

    def test_refuses_nonfinite_or_wrong_carrier(self):
        with self.assertRaises(ValueError):
            report_tensor({"bad": float("nan")})
        with self.assertRaisesRegex(ValueError, "CPU byte tensor"):
            read_report(torch.ones(3))


if __name__ == "__main__":
    unittest.main()
