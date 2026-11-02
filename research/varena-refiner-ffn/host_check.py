"""Real Diffusers refiner construction and stock CPU/ATen contract checks; no CUDA calls."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import diffusers
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TokenRefinerBlock
from torch import nn

from cozy_runtime.author.kernels import INT8_CONVROT
from cozy_runtime.internal import execution
from cozy_runtime.internal.encoding.leaves import Int8ConvRotNativeLeaf
from cozy_runtime.internal.execution.native_refiner_ffn import (
    _activate,
    _weight,
    declare,
    split_ffn,
)


def block() -> MiniMaxH3TokenRefinerBlock:
    return MiniMaxH3TokenRefinerBlock(256, 4, 64, 512, 1e-6, 1e-6).eval()


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(13829)
    assert not torch.cuda.is_initialized()
    refiner = block()
    with torch.device("meta"):
        full_refiner = MiniMaxH3TokenRefinerBlock(5376, 56, 128, 14336, 1e-6, 1e-6).eval()
    full_contract = declare(full_refiner.ff)
    assert full_contract is not None
    assert full_contract.up.weight.shape == (28672, 5376)
    assert full_contract.down.weight.shape == (5376, 14336)
    ff = refiner.ff
    contract = declare(ff)
    assert contract is not None and contract.unchanged()
    with torch.inference_mode():
        x = torch.randn(1, 17, 256)
        expected = ff(x)
        assert split_ffn(contract, x) is None
        execution.install(refiner)
        assert contract.unchanged()
        actual = ff(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert "_cozy_ffn_cut_forward" in vars(ff.net[0])

        activation_cases = []
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for rows in (1, 3, 5, 17):
                packed = (torch.randn(1, rows, 1024) * 8).to(dtype)
                value, gate = packed.chunk(2, dim=-1)
                expected = value * nn.functional.silu(gate)
                silu, activated = torch.empty_like(expected), torch.empty_like(expected)
                pointers = silu.data_ptr(), activated.data_ptr()
                _activate(packed, silu, activated)
                assert pointers == (silu.data_ptr(), activated.data_ptr())
                torch.testing.assert_close(activated, expected, rtol=0, atol=0)
                activation_cases.append({"dtype": str(dtype), "rows": rows, "exact": True})

        # The validator reads actual encoded leaves/tensors, not substitute provider objects.
        native_ff = block().ff
        for parent, attribute in ((native_ff.net[0], "proj"), (native_ff.net, "2")):
            original = getattr(parent, attribute)
            native = Int8ConvRotNativeLeaf().leaf(
                torch,
                {"weight": torch.randint(-31, 32, original.weight.shape, dtype=torch.int8),
                 "scale": torch.full((original.out_features,), .001)},
                original,
                torch.bfloat16,
            ).eval()
            setattr(parent, attribute, native)
        native_contract = declare(native_ff)
        assert native_contract is not None
        bx = x.to(torch.bfloat16)
        up = native_contract.up
        assert _weight(up, bx, 256, 1024) is not None
        up.provider = "unknown-provider"
        assert _weight(up, bx, 256, 1024) is None
        up.provider = INT8_CONVROT
        up.bias = torch.zeros(1024, dtype=torch.bfloat16)
        assert _weight(up, bx, 256, 1024) is None
        up.bias = None
        up.data = up.data.t().contiguous().t()
        assert _weight(up, bx, 256, 1024) is None

    # Structural mutation and custom hooks cannot bypass the normal module semantics.
    ff.net[1].p = .1
    assert not contract.unchanged()
    ff.net[1].p = 0.0
    handle = ff.net[0].register_forward_hook(lambda module, args, result: result)
    assert not contract.unchanged()
    handle.remove()
    assert contract.unchanged()
    ff.net[0].forward = lambda x: x
    assert not contract.unchanged()
    assert not torch.cuda.is_initialized()
    mapped = Path("/proc/self/maps").read_text().splitlines()
    assert not any("libcuda.so" in line for line in mapped)
    source = Path(__file__).parents[2] / "src/cozy_runtime/internal/execution/native_refiner_ffn.py"
    print(json.dumps({
        "passed": True, "scope": "real CPU refiner/leaf/ATen contract; no GPU qualification",
        "torch": str(torch.__version__), "diffusers": diffusers.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cpu_wrapper_exact": True, "actual_native_leaf_refusals": True,
        "actual_h3_shape": {"hidden": 5376, "inner": 14336, "meta_only": True},
        "activation_cases": activation_cases, "cuda_initialized": False, "libcuda_mapped": False,
    }, indent=2))


if __name__ == "__main__":
    main()
