"""Verify that the GPU admission fixture exercises real leaf hooks and reuse."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

from cozy_runtime.internal.encoding import leaves, rowwise_gemm


def _fixture():
    path = Path(__file__).with_name("rowwise_gemm_gate") / "compare.py"
    namespace = ModuleType("rowwise_gate_fixture")
    namespace.__path__ = [str(path.parent)]
    sys.modules[namespace.__name__] = namespace
    spec = importlib.util.spec_from_file_location("rowwise_gate_fixture.compare", path)
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


def test_full_sequence_comparison_includes_tail_rows(monkeypatch):
    namespace = ModuleType("rowwise_gate_test")
    namespace.__path__ = [str(Path(__file__).with_name("rowwise_gemm_gate"))]
    monkeypatch.setitem(sys.modules, "rowwise_gate_test", namespace)
    path = Path(namespace.__path__[0]) / "full_sequence.py"
    spec = importlib.util.spec_from_file_location("rowwise_gate_test.full_sequence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = torch.zeros((129, 16), dtype=torch.bfloat16)
    same = module._compare_outputs(torch, original, original.clone())
    assert same["bytes_equal"] and same["finite"] and same["different_elements"] == 0
    assert same["max_abs"] == same["relative_l2"] == same["bf16_ulp_max"] == 0
    changed = original.clone()
    changed[128, -1] = 2
    actual = module._compare_outputs(torch, changed, original, tile_rows=128)
    assert not actual["bytes_equal"] and actual["finite"]
    assert actual["different_elements"] == actual["tail_different_elements"] == 1
    assert actual["full_tile_different_elements"] == 0 and actual["max_abs"] == 2


def test_bf16_ulp_metrics_keep_signed_zero_distinct_from_numeric_error():
    fixture = _fixture()
    expected = (
        torch.tensor([0x3F80, -16512, 0], dtype=torch.int16).view(torch.bfloat16).reshape(1, 3)
    )
    actual = (
        torch.tensor([0x3F81, -16511, -32768], dtype=torch.int16).view(torch.bfloat16).reshape(1, 3)
    )
    result = fixture._difference(torch, actual, expected)
    assert not result["bytes_equal"] and result["different_elements"] == 2
    assert result["bf16_ulp_max"] == 1
    assert result["bf16_ulp_counts"] == {"1": 2, "2": 0, "greater_than_2": 0}
    assert result["relative_l2"] == 1 / 128
