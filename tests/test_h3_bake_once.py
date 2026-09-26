"""CPU proof for the deliberately narrow H3 bake-once composition prototype.

These tests do not qualify H3 quality or encoded weights.  They pin the two safety
properties a production composer must retain: adapter work is paid during preparation,
and composing one request cannot mutate the pristine base or another request's model.
"""

from copy import deepcopy
from typing import Any

import pytest


def _fixture() -> tuple[Any, Any]:
    torch = pytest.importorskip("torch")
    from cozy_runtime.models.minimax_h3.turbo import LoRAFactors

    base = torch.nn.Linear(4, 3, dtype=torch.bfloat16)
    factors = LoRAFactors(4, 3, rank=2, scale=0.5)
    with torch.no_grad():
        base.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4).to(torch.bfloat16))
        assert base.bias is not None
        base.bias.copy_(torch.arange(3, dtype=torch.float32).to(torch.bfloat16))
        factors.lora_down.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4))
        factors.lora_up.copy_(torch.arange(6, dtype=torch.float32).reshape(3, 2))
    return base, factors


def test_bake_is_one_time_and_does_not_call_adapter_forward_path(monkeypatch: Any) -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.models.minimax_h3.turbo import bake_lora_linear

    base, factors = _fixture()
    original = deepcopy(base.state_dict())
    baked = bake_lora_linear(base, factors)
    expected_weight = (
        base.weight.detach().to(torch.float32)
        + factors.scale
        * torch.mm(
            factors.lora_up.detach().to(torch.float32),
            factors.lora_down.detach().to(torch.float32),
        )
    ).to(torch.bfloat16)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("a baked request must not run LoRAFactors.accumulate")

    monkeypatch.setattr(factors, "accumulate", fail)
    monkeypatch.setattr(torch, "mm", fail)
    value = torch.arange(20, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16)
    actual = baked(value)
    assert actual.shape == (5, 3)
    expected = torch.nn.functional.linear(value, expected_weight, base.bias)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The base remains pristine after composition and inference.
    for name, expected in original.items():
        torch.testing.assert_close(base.state_dict()[name], expected, rtol=0, atol=0)


def test_request_compositions_are_isolated_from_each_other() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.models.minimax_h3.turbo import bake_lora_linear

    base, factors_a = _fixture()
    pristine = base.weight.detach().clone()
    factors_b = deepcopy(factors_a)
    with torch.no_grad():
        factors_b.lora_up.add_(1)
    baked_a = bake_lora_linear(base, factors_a)
    baked_b = bake_lora_linear(base, factors_b)
    before_b = baked_b.weight.detach().clone()
    with torch.no_grad():
        baked_a.weight.add_(1)
    assert not torch.equal(baked_a.weight, baked_b.weight)
    torch.testing.assert_close(baked_b.weight, before_b, rtol=0, atol=0)
    # Neither request changes the shared source model.
    torch.testing.assert_close(base.weight, pristine, rtol=0, atol=0)


def test_bake_refuses_encoded_base_dtype() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.author import ConformanceError
    from cozy_runtime.models.minimax_h3.turbo import bake_lora_linear

    base, factors = _fixture()
    # FP8 storage is exactly the case where a provider-owned conversion contract is needed.
    base.weight = torch.nn.Parameter(base.weight.detach().to(torch.float8_e4m3fn))
    with pytest.raises(ConformanceError, match="refuses encoded base dtype"):
        bake_lora_linear(base, factors)
