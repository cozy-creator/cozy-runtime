"""CPU range/oracle contracts; actual CUDA output requires separate remote admission."""

from __future__ import annotations

import math

import pytest

from cozy_runtime.internal import attention, attention_fp8_sol, attention_sol
from cozy_runtime.internal.encoding import DeviceFacts
from testdata.fp8_sol_cpu import quantize_cpu, reference

torch = pytest.importorskip("torch")


def test_fp16_tile_ranges_are_finite_even_with_all_products_at_the_bound() -> None:
    maximum = torch.finfo(torch.float16).max
    assert 128 * 16**2 == 32768 < maximum
    assert 64 * 16**2 == 16384 < maximum
    # Emulate successive K32 partial accumulations at their positive/negative bound.
    for extent in (64, 128):
        for sign in (-1, 1):
            running = torch.tensor(0, dtype=torch.float16)
            for _ in range(extent // 32):
                running = (running.float() + sign * 32 * 16**2).half()
                assert torch.isfinite(running)
            assert float(running) == sign * extent * 16**2


@pytest.mark.parametrize("case", ["zeros", "extrema", "random"])
def test_cpu_oracle_all_exact_preserves_constant_values_and_bounds_error(case: str) -> None:
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(617)
    tensors = [torch.randn(1, 257, 2, 128, generator=generator).bfloat16() for _ in range(3)]
    if case in {"zeros", "extrema"}:
        for tensor in tensors:
            tensor.fill_(0 if case == "zeros" else 1)
    q, k, v = tensors
    quantized = [quantize_cpu(tensor) for tensor in tensors]
    codes, scales = zip(*quantized, strict=True)
    assert all(
        torch.isfinite(code.float()).all() and code.float().abs().max() <= 16 for code in codes
    )
    blocks = math.ceil(q.shape[1] / 64)
    kc = torch.stack([x.float().mean(1) for x in k.split(64, dim=1)], dim=1).bfloat16()
    vc = torch.stack([x.float().sum(1) for x in v.split(64, dim=1)], dim=1).bfloat16()
    threshold = torch.full((1, blocks, 2), -1e6)
    output = reference(q, codes[1], codes[2], codes[0], scales, kc, vc, threshold, 0)
    assert torch.isfinite(output).all()
    if case != "random":
        assert torch.equal(output, v)
    else:
        full = torch.nn.functional.scaled_dot_product_attention(
            q.float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)
        ).transpose(1, 2)
        relative = (output.float() - full).norm() / full.norm()
        assert relative < 0.08, relative


def test_experimental_pins_are_explicit_head_local_and_share_sage_dense_reference() -> None:
    for name in ("sol-fp8-fp16", "kitchen-sol-sage"):
        candidate = attention.BY_NAME[name]
        assert candidate in attention.PINNABLE and candidate not in attention.CANDIDATES
        assert candidate.local.head_local and name in attention_sol.NAMES
        assert attention_sol.dense_identity(name) == "sageattention"
        with pytest.raises(attention.AttentionRefusal):
            attention.pinned(name, DeviceFacts("cpu", "CPU", 0, "", "", 0))
    q = torch.zeros(1, 65, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="SM120 experiment"):
        attention_fp8_sol.sparse(q, q, q, None, 0)
