"""Actual upstream tensor adoption and guarded native W8A8 qualification."""

import copy

import pytest

from cozy_runtime.internal.encoding.leaves import (
    PreQuantized,
    RowwiseNativeLeaf,
    quantize_activation_rowwise,
)
from cozy_runtime.internal.encoding.torchao_rowwise import (
    bind_rowwise_weight,
    rowwise_input,
    rowwise_tensor,
)

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")


def test_activation_adoption_preserves_codes_scales_shape_and_zero_recipe() -> None:
    value = torch.arange(48, dtype=torch.bfloat16).reshape(2, 3, 8) / 11
    value[0, 0].zero_()
    payload, scale = quantize_activation_rowwise(torch, value)
    prequantized = rowwise_input(
        PreQuantized(payload, scale, tuple(value.shape)), dtype=value.dtype
    )
    ordinary = rowwise_input(value, dtype=value.dtype)
    assert prequantized.qdata.data_ptr() == payload.data_ptr()
    assert prequantized.scale.data_ptr() == scale.data_ptr()
    assert prequantized.shape == value.shape
    assert prequantized.block_size == [1, 1, 8]
    assert torch.equal(ordinary.qdata.view(torch.uint8), prequantized.qdata.view(torch.uint8))
    assert torch.equal(ordinary.scale, prequantized.scale)
    assert scale[0].item() > 0
    assert torch.isfinite(ordinary.dequantize()).all()


def test_binding_keeps_real_linear_bias_hooks_and_deepcopy_receiver() -> None:
    module = torch.nn.Linear(16, 32, dtype=torch.bfloat16)
    bias = module.bias
    previous = module.weight
    handle = module.register_forward_pre_hook(lambda _module, _args: None)
    hooks = module._forward_pre_hooks
    payload = torch.ones(32, 16, dtype=torch.float8_e4m3fn)
    scale = torch.ones(32, 1, dtype=torch.float32)
    assert bind_rowwise_weight(module, payload, scale) is previous
    assert type(module) is torch.nn.Linear
    assert module.bias is bias
    assert module._forward_pre_hooks is hooks
    assert module.weight.qdata.data_ptr() == payload.data_ptr()
    assert module.weight.scale.data_ptr() == scale.data_ptr()
    assert list(module.state_dict()) == ["weight", "bias"]
    copied = copy.deepcopy(module)
    assert copied.forward.__self__ is copied
    assert copied.weight.qdata.data_ptr() != payload.data_ptr()
    assert torch.equal(copied.weight.dequantize(), module.weight.dequantize())
    handle.remove()
    assert not module._forward_pre_hooks
    with pytest.raises(RuntimeError, match="no weight-only fallback"):
        module(torch.ones(2, 16, dtype=torch.bfloat16))


def test_mismatched_parts_and_authored_forward_refuse_before_weight_mutation() -> None:
    module = torch.nn.Linear(16, 32, dtype=torch.bfloat16)
    original = module.weight
    payload = torch.ones(32, 16, dtype=torch.float8_e4m3fn)
    scale = torch.ones(32, 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="row scales"):
        rowwise_tensor(payload, scale.to(torch.bfloat16), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="logical shape"):
        rowwise_input(PreQuantized(payload, scale, (16, 32)), dtype=torch.bfloat16)
    module.forward = lambda input: input
    with pytest.raises(ValueError, match="authored forward"):
        bind_rowwise_weight(module, payload, scale)
    assert module.weight is original


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native W8A8 requires an owned GPU")
def test_native_torchao_preserves_existing_operands_bias_and_peft_input() -> None:
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(731)
    module = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda").eval()
    payload = torch.randn(256, 128, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand(256, 1, device="cuda", dtype=torch.float32) / 100
    value = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
    reference = RowwiseNativeLeaf("rowwise").leaf(
        torch, {"data": payload, "scale": scale}, module, module.weight.dtype
    )
    bind_rowwise_weight(module, payload, scale)
    with torch.no_grad():
        expected = reference(value)
        assert torch.equal(module(value), expected)
        encoded, scales = quantize_activation_rowwise(torch, value)
        assert torch.equal(module(PreQuantized(encoded, scales, tuple(value.shape))), expected)
        model = get_peft_model(
            torch.nn.Sequential(module),
            LoraConfig(r=8, lora_alpha=8, target_modules=["0"], init_lora_weights=False),
            autocast_adapter_dtype=False,
        ).eval()
        layer = model.base_model.model[0]
        assert type(layer).__name__ == "TorchaoLoraLinear"
        layer.set_scale("default", 0.3)
        assert layer.lora_A["default"].weight.dtype == torch.bfloat16
        expected = expected + layer.lora_B["default"](layer.lora_A["default"](value)) * 0.3
        actual = model(value)
        assert torch.equal(actual, expected)
        assert torch.equal(model(value), actual)
