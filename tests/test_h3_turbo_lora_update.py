"""H3 turbo's per-step LoRA update (`LoRAFactors.accumulate`): one rounding into the base output,
in fixed row chunks, with the same bits whatever the input's memory layout. CPU."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from cozy_runtime.models.minimax_h3 import turbo  # noqa: E402


class Gemms(TorchDispatchMode):  # type: ignore[misc]
    """The strides of every GEMM's left operand, as the kernels see them."""

    def __init__(self) -> None:
        super().__init__()
        self.strides: set[tuple[int, int]] = set()

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if func in (torch.ops.aten.addmm.default, torch.ops.aten.mm.default):
            left = args[1] if func == torch.ops.aten.addmm.default else args[0]
            self.strides.add((int(left.stride(0)), int(left.stride(1))))
        return func(*args, **(kwargs or {}))


def factors(width: int, out: int, rank: int) -> Any:
    lora = turbo.LoRAFactors(width, out, rank, 1.0)
    with torch.no_grad():
        lora.lora_down.copy_(torch.randn(rank, width))
        lora.lora_up.copy_(torch.randn(out, rank) * 0.1)
    return lora


def test_the_update_adds_in_one_rounding_over_chunks_and_a_padded_tail() -> None:
    torch.manual_seed(3)
    rows, width, out, rank = turbo._LORA_ROW_CHUNK + 777, 64, 48, 8
    lora = factors(width, out, rank)
    x = torch.randn(1, rows, width).bfloat16()
    base = torch.randn(1, rows, out).bfloat16()
    low = torch.nn.functional.linear(x, lora.lora_down.bfloat16())
    once = (base.float() + low.float() @ lora.lora_up.float().t()).bfloat16()
    got = base.clone()
    with torch.no_grad():
        lora.accumulate(x, got)
    # within one bf16 rounding of the fp32 sum (the GEMM's own accumulation order aside)
    assert torch.allclose(got.float(), once.float(), rtol=2**-7, atol=2**-7)


def test_a_strided_input_gives_the_same_bits_through_row_major_gemms() -> None:
    torch.manual_seed(5)
    rows, width, out, rank = turbo._LORA_ROW_CHUNK + 123, 64, 48, 8
    lora = factors(width, out, rank)
    x = torch.randn(1, rows, width).bfloat16()
    strided = x.transpose(1, 2).contiguous().transpose(1, 2)
    assert not strided.is_contiguous()
    base = torch.randn(1, rows, out).bfloat16()
    expected, got = base.clone(), base.clone()
    with torch.no_grad():
        lora.accumulate(x, expected)
        with Gemms() as gemms:
            lora.accumulate(strided, got)
    assert torch.equal(got, expected)
    assert gemms.strides <= {(width, 1), (rank, 1)}
