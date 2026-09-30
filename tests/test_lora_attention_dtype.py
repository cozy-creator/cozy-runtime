"""PEFT preserves the active base projection's result precision after encoded fill."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("factor_dtype", ["float32", "float16"])
def test_peft_attention_contract_tracks_actual_encoded_replacement(factor_dtype: str) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig
    from peft.tuners.lora.layer import Linear as PeftLinear

    from cozy_runtime.internal import attention
    from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf

    torch.manual_seed(292)
    base = torch.nn.Linear(4, 4, bias=False, dtype=torch.float32)
    adapted = PeftLinear(
        base,
        "adapter_0",
        config=LoraConfig(init_lora_weights=False, inference_mode=True, lora_dropout=0.0),
        r=2,
        lora_alpha=2,
    ).eval()
    for layer in (adapted.lora_A["adapter_0"], adapted.lora_B["adapter_0"]):
        layer.to(dtype=getattr(torch, factor_dtype))
        with torch.no_grad():
            layer.weight.fill_(0.25)
    site = SimpleNamespace(to_q=adapted, to_k=adapted, to_v=adapted)
    before = adapted(torch.ones(2, 4, dtype=torch.float32))
    assert before.dtype == torch.float32
    assert attention._dtype(site) == "float32"

    payload = torch.full((4, 4), 4, dtype=torch.float8_e4m3fn)
    scale = torch.full((4,), 0.25, dtype=torch.float32)
    native = RowwiseNativeLeaf("fp8-rowwise/1").leaf(
        torch, {"data": payload, "scale": scale}, base, torch.bfloat16
    )
    adapted.base_layer = native
    operands = torch.ones(2, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        plain, output = native(operands), adapted(operands)
        extra = adapted.lora_B["adapter_0"](
            adapted.lora_A["adapter_0"](operands.to(getattr(torch, factor_dtype)))
        )
    assert output.dtype == plain.dtype == torch.bfloat16
    torch.testing.assert_close(output, (plain + extra).to(plain.dtype), rtol=0, atol=0)
    assert attention._dtype(site) == "bfloat16"
    sol = next(candidate for candidate in attention.PINNABLE if candidate.name == "sol-attn")
    assert attention._admits(sol, attention._dtype(site), 128)
    assert not attention._admits(sol, "", 128)
    assert adapted.lora_A["adapter_0"].weight.dtype == getattr(torch, factor_dtype)
    assert adapted.lora_B["adapter_0"].weight.dtype == getattr(torch, factor_dtype)
    assert not torch.cuda.is_initialized()


def test_custom_wrapper_cannot_claim_peft_output_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.internal import attention

    class Custom:
        # Even upstream module spelling plus the same method is insufficient authority.
        __module__ = "peft.tuners.lora.layer"

        def get_base_layer(self) -> object:
            return torch.nn.Linear(4, 4, dtype=torch.bfloat16)

    assert attention._projection_dtype(Custom()) == ""
    # Unavailable optional exports must not break an unadapted author's projection.
    monkeypatch.setitem(sys.modules, "peft.tuners.lora.layer", ModuleType("unavailable_peft"))
    assert attention._projection_dtype(Custom()) == ""
    sdpa = next(candidate for candidate in attention.CANDIDATES if candidate.name == "sdpa")
    assert attention._admits(sdpa, attention._projection_dtype(Custom()), 128)
