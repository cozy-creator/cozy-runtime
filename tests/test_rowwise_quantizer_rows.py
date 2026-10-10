"""The fused rowwise quantizer past 2**31 elements, on a real GPU.

H3's 15 s segment with 56 context frames packs ~155k rows into the 14336-wide FFN out
projection. The kernel's int32 `row * stride` wrapped there: loads and stores landed GiBs
below their tensors, silently or as an illegal address (runs 5313, 5318, 5319).
"""

from __future__ import annotations

import pytest

from cozy_runtime.internal.encoding import leaves

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

COLUMNS = 14336
ROWS = 2**31 // COLUMNS + 2  # the last two rows start past 2**31 elements


def test_rows_past_two_billion_elements_quantize_where_they_are() -> None:
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 7 << 30:
        pytest.skip("needs a CUDA device with 7 GiB free")
    assert leaves._triton_rowwise_kernel() is not None
    x = torch.empty(ROWS, COLUMNS, dtype=torch.bfloat16, device="cuda")
    x[:-4].fill_(1.0)
    tail = torch.randn(4, COLUMNS, generator=torch.Generator().manual_seed(7)) * 3
    x[-4:] = tail.to(x.dtype)
    payload, scale = leaves.quantize_activation_rowwise(torch, x)
    torch.cuda.synchronize()
    assert leaves._triton_rowwise_kernel() is not None  # the fused path served, no fallback
    expected_payload, expected_scale = leaves.quantize_activation_rowwise(torch, x[-4:].cpu())
    assert torch.allclose(scale[-4:].cpu(), expected_scale, rtol=1e-6, atol=0)
    # Triton's f32->e4m3 cast double-rounds through f16: a tie may take the other neighbour.
    differs = payload[-4:].cpu().float() != expected_payload.float()
    assert differs.float().mean().item() < 0.01
    assert torch.allclose(scale[:2].cpu(), torch.full((2, 1), 1.0 / 448), rtol=1e-6, atol=0)
    assert bool((payload[:2].float() == 448.0).all())
