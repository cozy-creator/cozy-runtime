"""`fp8-rowwise/1`'s native op, built as a pod builds it, against the eager law."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 9),
    reason="the fp8 op runs on an sm89+ card",
)


def test_fp8_op_verifies_on_the_card_and_keeps_the_law(tmp_path: Path) -> None:
    """The Runtime wheel's prebuilt 8-bit op library, verified on this card. Its quantize kernel is
    `eager_quantize` bit for bit (a zero row, K not a multiple of 256), and every GEMM tile of
    this card's table, with a bias, in fp16 and bf16, from an unaligned view, agrees with the
    f64 product of the quantized operands."""
    from cozy_runtime.internal import fp8_linear, int8_linear, kernel_cache

    kernel_cache.configure(kernel_cache.Store(tmp_path / "kernels"))
    assert fp8_linear.native(torch, 0) is not None
    generator = torch.Generator().manual_seed(13)
    for rows, width in ((1, 16), (5, 4096), (300, 1040), (64, 32768)):
        x = (torch.randn(rows, width, generator=generator) * 4).to("cuda", torch.bfloat16)
        x[0] = 0
        operand = fp8_linear.quantize(torch, x)
        payload, scale = fp8_linear.eager_quantize(torch, x)
        assert torch.equal(operand.payload.view(torch.uint8), payload.view(torch.uint8))
        assert torch.equal(operand.scale, scale.view(-1))
    for rows, out_features, in_features in (
        (1, 1024, 4096),
        (77, 1280, 2048),
        (300, 4096, 1040),
        (4352, 3072, 3072),
    ):
        weight = torch.randn(out_features, in_features, generator=generator) * 0.02
        data, weight_scale = fp8_linear.eager_quantize(torch, weight.cuda())
        weight_scale = weight_scale.view(-1)
        exact = data.double() * weight_scale.double().view(-1, 1)
        for dtype in (torch.bfloat16, torch.float16):
            x = torch.randn(1, rows + 1, in_features, generator=generator).to("cuda", dtype)[:, 1:]
            bias = torch.randn(out_features, generator=generator).to("cuda", dtype)
            q, s = fp8_linear.eager_quantize(torch, x)
            want = (q.double() * s.double()) @ exact.t() + bias.double()
            got = fp8_linear.linear(torch, x, data, weight_scale, bias, dtype)
            assert fp8_linear.serving(0) == fp8_linear.OP
            assert got.shape == (1, rows, out_features) and got.dtype == dtype
            assert _error(got.reshape(rows, -1), want) < 0.01, (rows, out_features, dtype)
            operand = fp8_linear.quantize(torch, x.reshape(rows, -1))
            for config in range(4):
                try:
                    tile = int8_linear.library(0).fp8_gemm(
                        operand.payload, operand.scale, data, weight_scale, bias, dtype, config
                    )
                except ValueError as exc:
                    assert "cannot take" in str(exc), exc
                    continue
                assert _error(tile, want) < 0.01, (rows, out_features, dtype, config)


def test_rows_past_two_billion_elements_quantize_where_they_are() -> None:
    """H3's 15 s segment with 56 context frames puts ~155k rows into the 14336-wide FFN out
    projection: past 2**31 elements, where 32-bit row offsets wrapped (runs 5313, 5318)."""
    from cozy_runtime.internal import fp8_linear

    if fp8_linear.native(torch, 0) is None or torch.cuda.mem_get_info()[0] < 7 << 30:
        pytest.skip("needs the built op and 7 GiB free")
    columns = 14336
    x = torch.empty(2**31 // columns + 2, columns, dtype=torch.bfloat16, device="cuda")
    x[:-4].fill_(1.0)
    x[-4:] = (torch.randn(4, columns, generator=torch.Generator().manual_seed(7)) * 3).to(x.dtype)
    operand = fp8_linear.quantize(torch, x)
    payload, scale = fp8_linear.eager_quantize(torch, x[-4:])
    assert torch.equal(operand.payload[-4:].view(torch.uint8), payload.view(torch.uint8))
    assert torch.equal(operand.scale[-4:], scale.view(-1))
    assert bool((operand.scale[:2] == 1.0 / 448).all())
    assert bool((operand.payload[:2].float() == 448.0).all())


def _error(got: Any, want: Any) -> float:
    return float((got.double() - want).norm() / want.norm())
