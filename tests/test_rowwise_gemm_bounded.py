"""CPU dispatch/resource/hook controls; GPU arithmetic is a separate ordinary-CLI gate."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.internal.encoding import leaves, rowwise_gemm

torch = pytest.importorskip("torch")


def test_fp32_tile_budget_is_independent_of_full_sequence_length() -> None:
    assert rowwise_gemm.tile_rows(7168) == 4096
    assert rowwise_gemm.tile_rows(28672) == 1024
    for columns in (1, 16, 5376, 7168, 14336, 28672, rowwise_gemm.MAX_FP32_BYTES // 4):
        rows = rowwise_gemm.tile_rows(columns)
        assert 0 < rows <= 4096 and rows & (rows - 1) == 0
        assert rows * columns * 4 <= rowwise_gemm.MAX_FP32_BYTES
    assert rowwise_gemm.tile_rows(rowwise_gemm.MAX_FP32_BYTES // 4 + 1) == 0
    assert rowwise_gemm.tile_rows(0) == 0


@pytest.mark.parametrize("sm", [(8, 9), (9, 0), (10, 0), (12, 0)])
def test_experiment_does_not_replace_other_architectures_or_dtypes(sm: tuple[int, int]) -> None:
    control = SimpleNamespace(
        bfloat16="bf16", cuda=SimpleNamespace(get_device_capability=lambda _: sm)
    )
    operand = SimpleNamespace(is_cuda=True, shape=(17, 32), device="cuda")
    assert rowwise_gemm.eligible(control, operand, 64, "bf16") == (sm == (12, 0))
    assert not rowwise_gemm.eligible(control, operand, 64, "fp16")
    operand.is_cuda = False
    assert not rowwise_gemm.eligible(control, operand, 64, "bf16")


def test_chunked_rows_keep_order_scales_and_only_one_fp32_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(12)
    a = torch.randn((4103, 32), generator=generator).to(torch.float8_e4m3fn)
    b = torch.randn((64, 32), generator=generator).to(torch.float8_e4m3fn).t()
    scales_a = torch.rand((4103, 2), generator=generator)[:, :1]
    scales_b = torch.rand((1, 128), generator=generator)[:, ::2]
    held: list[weakref.ReferenceType[Any]] = []
    shapes: list[tuple[int, int]] = []

    def gemm(left: Any, right: Any, **options: Any) -> Any:
        assert all(ref() is None for ref in held), "previous FP32 tile overlaps next allocation"
        assert options["out_dtype"] == torch.float32 and options["use_fast_accum"] is False
        assert options["bias"] is None
        assert options["scale_a"].item() == options["scale_b"].item() == 1
        result = left.float() @ right.float()
        held.append(weakref.ref(result))
        shapes.append((int(result.shape[0]), int(result.shape[1])))
        return result

    def rescale(accum: Any, row: Any, col: Any, output: Any) -> None:
        output.copy_((accum * row) * col)

    monkeypatch.setattr(torch, "_scaled_mm", gemm)
    monkeypatch.setattr(rowwise_gemm, "rescale", rescale)
    got = rowwise_gemm.scalar_rescaled(torch, a, b, scales_a, scales_b)
    expected = (((a.float() @ b.float()) * scales_a) * scales_b).to(torch.bfloat16)
    assert torch.equal(got, expected)
    assert shapes == [(4096, 64), (7, 64)] and got.is_contiguous()
    gc.collect()
    assert all(ref() is None for ref in held)


def test_prequantized_leaf_still_runs_original_hooks_and_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.arange(12, dtype=torch.bfloat16).reshape(2, 6)
    payload = original.to(torch.float8_e4m3fn)
    scales = torch.ones((2, 1), dtype=torch.float32)
    weight = torch.ones((3, 6), dtype=torch.float8_e4m3fn)
    weight_scales = torch.ones(3, dtype=torch.float32)
    replaced = torch.nn.Linear(6, 3, bias=True, dtype=torch.bfloat16)
    replaced.bias.data.fill_(2)
    leaf = leaves.RowwiseNativeLeaf("fixture").leaf(
        torch, {"data": weight, "scale": weight_scales}, replaced, torch.bfloat16
    )
    prequantized = leaves.PreQuantized(payload, scales, tuple(original.shape))
    seen: list[str] = []

    def quantize(_torch: Any, operand: Any) -> tuple[Any, Any]:
        assert operand is prequantized
        return operand.payload, operand.scale

    def gemm(_torch: Any, a: Any, b: Any, sa: Any, sb: Any, dtype: Any) -> Any:
        assert a is payload and sa is scales
        assert b.data_ptr() == weight.data_ptr()
        return (((a.float() @ b.float()) * sa) * sb).to(dtype)

    def before(_module: Any, args: Any) -> None:
        assert args[0] is original
        seen.append("before")

    def after(_module: Any, args: Any, output: Any) -> None:
        assert args[0] is original
        seen.append("after")
        output.add_(1)

    monkeypatch.setattr(leaves, "quantize_activation_rowwise", quantize)
    monkeypatch.setattr(rowwise_gemm, "mm", gemm)
    leaf.register_forward_pre_hook(before)
    leaf.register_forward_hook(after)
    result = leaf(original, prequantized=prequantized)
    expected = original.sum(-1, keepdim=True).expand(2, 3) + 3
    assert torch.equal(result, expected) and seen == ["before", "after"]
    assert leaf.data is weight and leaf.scale is weight_scales


def test_unexpected_gemm_error_is_not_hidden_by_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("visible GEMM failure")

    monkeypatch.setattr(torch, "_scaled_mm", fail)
    with pytest.raises(RuntimeError, match="visible GEMM failure"):
        rowwise_gemm.scalar_rescaled(
            torch,
            torch.zeros((1, 16)),
            torch.zeros((16, 16)),
            torch.ones((1, 1)),
            torch.ones((1, 16)),
        )
