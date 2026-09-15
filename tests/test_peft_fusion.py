"""Real H3 fusion-control compatibility with standard weights and native PEFT."""


import pytest

from cozy_runtime.internal.encoding.torchao_rowwise import bind_rowwise_weight
from cozy_runtime.internal.lora import AdapterInput, LinearAdapters

torch = pytest.importorskip("torch")
pytest.importorskip("torchao")
pytest.importorskip("peft")
diffusers = pytest.importorskip("diffusers")


def test_fusion_installer_preserves_storage_and_only_quantizes_inactive_peft_inputs() -> None:
    from cozy_runtime.internal.fusion_install import (
        _install_transformer,
        _rowwise_leaf,
        _tensor_census,
    )

    transformer = diffusers.MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=1,
        num_refiner_layers=0,
        ffn_dim=64,
        text_dim=32,
        freq_dim=16,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    ).to(torch.bfloat16)
    for module in transformer.modules():
        if type(module) is torch.nn.Linear:
            bind_rowwise_weight(
                module,
                torch.ones_like(module.weight, dtype=torch.float8_e4m3fn),
                torch.ones(module.out_features, 1, dtype=torch.float32),
            )
    target = "transformer_blocks.0.attn.to_q"
    original = transformer.get_submodule(target)
    binding = LinearAdapters({"dit": transformer}, {"dit": "base"}, {"dit": [target]})
    layer = transformer.get_submodule(target)
    before = _tensor_census(transformer)
    substitutions, fp8_sites = _install_transformer(transformer, "dit")
    assert substitutions and fp8_sites == 3
    assert _tensor_census(transformer) == before
    assert _rowwise_leaf(layer)
    adapter = AdapterInput(
        "one",
        "dit",
        "base",
        {
            f"{target}.lora_A.weight": torch.ones(2, original.in_features, dtype=torch.bfloat16),
            f"{target}.lora_B.weight": torch.ones(original.out_features, 2, dtype=torch.bfloat16),
        },
    )
    with binding.request(
        [adapter], device="cpu", max_device_bytes=10000, check_cancelled=lambda: None
    ):
        assert not _rowwise_leaf(layer)
        assert _rowwise_leaf(transformer.transformer_blocks[0].attn.to_k)
    assert _rowwise_leaf(layer)
    binding.close()
    assert transformer.get_submodule(target) is original
