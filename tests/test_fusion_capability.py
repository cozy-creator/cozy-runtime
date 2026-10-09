"""Supported upstream evolution uses real modules; all these controls are CPU only."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cozy_runtime.internal import fusion_install as install

try:
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.activations import SwiGLU
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3AttnProcessor,
        MiniMaxH3RotaryPosEmbed,
        MiniMaxH3TransformerBlock,
    )
except ImportError:
    pytest.skip("Torch and Diffusers are required", allow_module_level=True)


def transformer() -> Any:
    torch.manual_seed(76182)
    return (
        MiniMaxH3Transformer3DModel(
            num_layers=2,
            num_refiner_layers=1,
            hidden_size=256,
            ffn_dim=512,
            num_attention_heads=2,
            attention_head_dim=128,
            time_embed_hidden_dim=256,
            time_embed_dim=128,
        )
        .eval()
        .bfloat16()
    )


def test_actual_dependency_supports_fusion_contract() -> None:
    install._require_capabilities()
    model = transformer()
    before = install._tensor_census(model)
    install._validate_transformer(model, "dit")
    assert before == install._tensor_census(model)
    assert all(type(block) is MiniMaxH3TransformerBlock for block in model.transformer_blocks)


def test_annotations_and_extra_optional_inputs_do_not_define_compatibility() -> None:
    def original(self: object, hidden_states: object) -> object:
        return hidden_states

    def annotated(self: object, hidden_states: object, *, debug: bool = False) -> object:
        return hidden_states

    for function in (original, annotated):
        install._require_call(function, ("self", "hidden_states"))


def test_changed_input_contract_is_refused() -> None:
    def incompatible(self: object, mask: object, hidden_states: object) -> object:
        return hidden_states

    with pytest.raises(install.FusionRefusal, match="incompatible") as refused:
        install._require_call(incompatible, ("self", "hidden_states"))
    assert refused.value.code == "fusion_contract_unsupported"


@pytest.mark.parametrize(
    "change",
    [
        "dropout",
        "activation",
        "norm",
        "forward",
        "hook",
        "norm_hook",
        "activation_hook",
        "dropout_forward",
        "activation_forward",
        "attention_hook",
        "swiglu_hook",
        "swiglu_post_hook",
    ],
)
def test_unsupported_later_block_leaves_every_block_unmodified(change: str) -> None:
    model = transformer()
    first, second = model.transformer_blocks
    if change == "dropout":
        second.ff.net[1].p = 0.25
    elif change == "activation":
        second.ff.net[0].activation = torch.nn.GELU()
    elif change == "norm":
        second.norm1 = torch.nn.LayerNorm(256)
    elif change == "forward":
        second.forward = lambda *args: args[0]
    elif change == "dropout_forward":
        second.ff.net[1].forward = lambda value: value + 1
    elif change == "activation_forward":
        second.ff.net[0].activation.forward = lambda value: value + 1
    elif change == "attention_hook":
        second.attn.register_forward_pre_hook(lambda module, args: (args[0] + 1, *args[1:]))
    elif change == "swiglu_hook":
        second.ff.net[0].register_forward_pre_hook(lambda module, args: (args[0] + 1,))
    elif change == "swiglu_post_hook":
        second.ff.net[0].register_forward_hook(lambda module, args, output: output + 1)
    elif change == "norm_hook":
        second.norm1.register_forward_pre_hook(lambda module, args: (args[0] + 1,))
    elif change == "activation_hook":
        second.ff.net[0].activation.register_forward_hook(lambda module, args, output: output + 1)
    else:
        second.ff.register_forward_hook(lambda module, args, output: output + 1)
    with pytest.raises(install.FusionRefusal) as refused:
        install._validate_transformer(model, "dit")
    assert refused.value.code == "fusion_block_shape"
    assert type(first) is MiniMaxH3TransformerBlock
    assert type(first.ff.net[0]) is SwiGLU
    assert type(first.attn.processor) is MiniMaxH3AttnProcessor


def test_real_upstream_block_matches_the_fused_composition_contract() -> None:
    block = transformer().transformer_blocks[0]
    torch.manual_seed(1762)
    x = torch.randn(1, 17, 256).bfloat16()
    temb = torch.randn(3, 128).bfloat16()
    index = torch.arange(17) % 3
    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=16)(torch.randint(0, 9, (17, 3)))
    with torch.inference_mode():
        expected = block(x, temb, index, rope)
        shift1, scale1, gate1, shift2, scale2, gate2 = block.adaln_proj(temb)
        normalized = block.norm1(x) * (1.0 + scale1[index]) + shift1[index]
        attended = block.attn(normalized, rope)
        residual = x + gate1[index] * attended
        normalized = block.norm2(residual) * (1.0 + scale2[index]) + shift2[index]
        projected, gate = block.ff.net[0].proj(normalized).chunk(2, dim=-1)
        activation = projected * torch.nn.functional.silu(gate)
        actual = residual + gate2[index] * block.ff.net[2](activation)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unapplied_report_is_visible_without_gpu(caplog: pytest.LogCaptureFixture) -> None:
    holder = SimpleNamespace()
    result = install.apply_fusion_plan(holder, torch=torch, device=torch.device("cpu"))
    assert holder._cozy_fusion_report == result.document()
    assert result.code == "fusion_device_incapable"
    assert "fusion_device_incapable" in caplog.text
