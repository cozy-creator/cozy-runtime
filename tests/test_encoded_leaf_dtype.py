"""Encoded storage must not change an upstream projection's activation dtype."""

from __future__ import annotations

import unittest

try:
    import torch
    from diffusers.models.modeling_utils import get_parameter_dtype
except ImportError as exc:
    raise unittest.SkipTest("the Torch/Diffusers integration environment is required") from exc

from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf, quantize_activation_rowwise


class EncodedLeafDtypeTest(unittest.TestCase):
    def test_logical_dtype_survives_encoded_storage_and_restage(self) -> None:
        payload = torch.tensor([[1, -2, 3, -4], [-4, 3, -2, 1]]).to(torch.float8_e4m3fn)
        scales = torch.tensor([0.125, 3.5], dtype=torch.float32)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                original = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
                leaf = RowwiseNativeLeaf("fp8-rowwise/1").leaf(
                    torch, {"data": payload, "scale": scales}, original, dtype
                )
                self.assertEqual(get_parameter_dtype(leaf), dtype)
                self.assertIs(leaf.roles()["data"], payload)
                self.assertIs(leaf.roles()["scale"], scales)
                self.assertEqual(set(leaf.state_dict()), {"data", "scale"})
                self.assertEqual(list(leaf.parameters()), [])
                self.assertEqual(
                    sum(t.numel() * t.element_size() for t in leaf.buffers()),
                    payload.numel() * payload.element_size()
                    + scales.numel() * scales.element_size(),
                )
                leaf.to_empty(device="meta")
                self.assertEqual(get_parameter_dtype(leaf), dtype)
                leaf.to_empty(device="cpu")
                leaf.data.copy_(payload)
                leaf.scale.copy_(scales)
                self.assertEqual(get_parameter_dtype(leaf), dtype)
                torch.testing.assert_close(leaf.data.float(), payload.float())
                torch.testing.assert_close(leaf.scale, scales)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a real CUDA FP8 device")
    def test_upstream_cast_preserves_nonunit_scaled_gpu_forward(self) -> None:
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("rowwise FP8 proof requires Hopper or newer")
        torch.cuda.set_per_process_memory_fraction(0.02)
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(917)
        payload = torch.randn(32, 32, generator=generator, device=device).to(torch.float8_e4m3fn)
        scales = torch.linspace(0.125, 3.5, 32, device=device)
        logical = torch.randn(16, 32, generator=generator, device=device)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                original = torch.nn.Linear(32, 32, bias=False, dtype=dtype, device=device)
                leaf = RowwiseNativeLeaf("fp8-rowwise/1").leaf(
                    torch, {"data": payload, "scale": scales}, original, dtype
                )
                self.assertEqual(get_parameter_dtype(leaf), dtype)
                activation = logical.to(get_parameter_dtype(leaf))
                quantized, activation_scale = quantize_activation_rowwise(torch, activation)
                expected = (quantized.float() * activation_scale) @ (
                    payload.float() * scales[:, None]
                ).t()
                torch.testing.assert_close(
                    leaf(activation).float(), expected.to(dtype).float(), rtol=0.015, atol=0.025
                )
                self.assertIs(leaf.roles()["data"], payload)
                self.assertIs(leaf.roles()["scale"], scales)


if __name__ == "__main__":
    unittest.main()
