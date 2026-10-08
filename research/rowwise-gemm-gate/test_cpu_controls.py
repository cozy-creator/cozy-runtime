"""Verify that the GPU admission fixture exercises real leaf hooks and reuse."""

import importlib.util
from pathlib import Path

import torch

from cozy_runtime.internal.encoding import leaves, rowwise_gemm


def _fixture():
    path = Path(__file__).with_name("rowwise_gemm_gate") / "compare.py"
    spec = importlib.util.spec_from_file_location("bounded_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_uses_original_leaf_with_both_prequantized_and_ordinary_inputs(monkeypatch):
    fixture = _fixture()
    generator = torch.Generator().manual_seed(16)
    x = torch.randn((7, 32), generator=generator, dtype=torch.bfloat16)
    w = torch.randn((16, 32), generator=generator, dtype=torch.bfloat16)
    aq, a_scale = leaves.quantize_activation_rowwise(torch, x)
    bq, b_scale = leaves.quantize_activation_rowwise(torch, w)

    def mm(control, a, b, sa, sb, dtype):
        return (((a.float() @ b.float()) * sa) * sb).to(dtype)

    monkeypatch.setattr(rowwise_gemm, "mm", mm)
    native = mm(torch, aq, bq.t(), a_scale, b_scale.reshape(1, -1), torch.bfloat16)
    previous_quantizer, previous_rescale = leaves.quantize_activation_rowwise, rowwise_gemm.rescale
    result = fixture._leaf_check(torch, leaves, rowwise_gemm, x, aq, a_scale, bq, b_scale, native)
    assert result["passed"], result
    assert leaves.quantize_activation_rowwise is previous_quantizer
    assert rowwise_gemm.rescale is previous_rescale


def test_selected_real_matrix_cases_cross_a_tile_and_have_a_nonempty_partial_tail():
    for m, n, _k in _fixture().SHAPES.values():
        tile = rowwise_gemm.tile_rows(n)
        assert m > tile and 0 < m % tile < tile
        assert tile * n * 4 <= rowwise_gemm.MAX_FP32_BYTES
