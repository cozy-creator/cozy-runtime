"""CPU state/ownership contracts, without model forwards or accelerator qualification."""

from functools import wraps
from typing import Any, cast

import pytest


def test_runtime_wrappers_do_not_launder_foreign_forward_or_hook_metadata() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.author import is_declared_pure, original_forward, pure, retry_state
    from cozy_runtime.internal import tiers
    from cozy_runtime.internal.memory import Planner

    root = pure(torch.nn.Linear(2, 2))
    planner = Planner(torch, torch.device("cpu"))
    held = planner.adopt("c", "root", root, None, tiers.DEVICE)
    assert planner._state_contract(held, [])
    original = root.forward
    root.forward = tiers.op("forward", original, lambda *_: None)
    assert original_forward(root) == original
    assert planner._state_contract(held, [])

    @wraps(root.forward)
    def foreign(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)

    root.forward = foreign
    assert original_forward(root) is foreign
    assert not tiers.installed(foreign)
    retry_state(root, lambda: None, lambda _: None)
    assert not planner._state_contract(held, [])
    root.forward = original

    @pure
    def hook(*_: Any) -> None:
        pass

    @wraps(hook)
    def foreign_hook(*_: Any) -> None:
        pass

    assert not is_declared_pure(foreign_hook)
    handle = root.register_forward_hook(foreign_hook)
    assert not planner._state_contract(held, [])
    handle.remove()
    handle = root.register_forward_hook(hook)
    assert planner._state_contract(held, [])
    handle.remove()


def test_qwen_hidden_state_root_restores_rope_bindings_and_buffer_provenance() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_h3_conditioner_lifecycle import _conditioner

    with torch.device("meta"):
        model = _conditioner(torch, "meta")
    rotary = model.model.language_model.rotary_emb
    save, restore = model._cozy_retry_state
    before = save()
    frequency = rotary.inv_freq
    census = tuple(model.state_dict())
    model.model.rope_deltas = torch.empty(1, device="meta")
    rotary.register_buffer("inv_freq", torch.empty(2, device="meta"), persistent=True)
    rotary.long_inv_freq = torch.empty(2, device="meta")
    rotary.max_seq_len_cached += 100
    restore(before)
    assert model.model.rope_deltas is None
    assert rotary.inv_freq is frequency
    assert "inv_freq" in rotary._non_persistent_buffers_set
    assert "long_inv_freq" not in vars(rotary)
    assert tuple(model.state_dict()) == census
    after = save()
    assert before[1].keys() == after[1].keys()
    assert all(after[1][key] is value for key, value in before[1].items())


def test_qwen_request_restores_both_kv_contexts_and_removes_its_contract() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.hooks.hooks import HookRegistry

    from cozy_runtime.author import retry_state
    from cozy_runtime.models.qwen_image21.model import _StagedPipeline

    root = torch.nn.Module()
    root.pos_embed = torch.nn.Module()
    freqs = [torch.ones(2)]
    root.pos_embed.freqs = freqs
    retry_state(root, lambda: None, lambda _: None)
    previous = root._cozy_retry_state
    pipeline = _StagedPipeline.__new__(_StagedPipeline)
    object.__setattr__(pipeline, "transformer", root)
    registry = HookRegistry.check_if_exists_or_initialize(root)
    with pytest.raises(ValueError, match="request failed"), pipeline.transformer_retry_state():
        cond, uncond = pipeline._make_kv_cache(2), pipeline._make_kv_cache(2)
        old = torch.ones(2)
        cond.get_layer(0).store(old, old)
        save, restore = root._cozy_retry_state
        state = save()
        root.pos_embed.freqs = [torch.zeros(2)]
        for cache in (cond, uncond):
            for layer in cache.layer_caches:
                layer.store(torch.zeros(2), torch.zeros(2))
        restore(state)
        assert root.pos_embed.freqs is freqs
        assert cond.get_layer(0).k is cond.get_layer(0).v is old
        assert cond.get_layer(1).k is uncond.get_layer(0).k is None
        registry._set_context("cond")
        raise ValueError("request failed")
    assert root._cozy_retry_state is previous
    assert pipeline._kv_caches == []
    assert registry.hooks == {}


def test_qwen_vae_restores_cache_list_aliases_and_first_call_absence() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.models.qwen_image21.model import _declare_vae_retry_state

    vae = torch.nn.Module()
    tensor = torch.ones(2)
    cache, index = [tensor, None], [0]
    vae._feat_map, vae._conv_idx = cache, index
    _declare_vae_retry_state(vae)
    save, restore = vae._cozy_retry_state
    state = save()
    cache[0], index[0] = torch.zeros(2), 5
    vae._feat_map = []
    vae._enc_feat_map = [torch.zeros(2)]
    restore(state)
    assert vae._feat_map is cache and cache[0] is tensor
    assert vae._conv_idx is index and index == [0]
    assert not hasattr(vae, "_enc_feat_map")
    assert torch.equal(tensor, torch.ones(2))


def test_h3_weightnorm_contract_preserves_binding_and_does_not_claim_foreign_hooks() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.internal import tiers
    from cozy_runtime.internal.memory import Planner
    from cozy_runtime.models.minimax_h3.official import _declare_audio_retry_state

    root = torch.nn.Sequential(torch.nn.utils.weight_norm(torch.nn.Linear(2, 2)))
    _declare_audio_retry_state(root)
    save, restore = root._cozy_retry_state
    weight = root[0].weight
    state = save()
    root[0].weight = torch.zeros_like(weight)
    restore(state)
    assert root[0].weight is weight
    del root[0].weight
    state = save()
    root[0].weight = weight
    restore(state)
    assert "weight" not in vars(root[0])
    planner = Planner(torch, torch.device("cpu"))
    held = planner.adopt("c", "audio", root, None, tiers.DEVICE)
    assert planner._state_contract(held, [])
    handle = root[0].register_forward_pre_hook(lambda *_: None)
    assert not planner._state_contract(held, [])
    handle.remove()


def test_h3_pruned_arming_restores_aliases_and_keeps_coverage_with_the_root() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.internal import tiers
    from cozy_runtime.models.minimax_h3.adaln_pruned import (
        AdaLNPrunedMiniMaxH3Transformer,
        _Armed,
    )

    with torch.device("meta"):
        root = AdaLNPrunedMiniMaxH3Transformer(
            table_timesteps=(1000.0,), table_block_keys=((0, 0),),
            num_attention_heads=1, attention_head_dim=8, hidden_size=8, num_layers=2,
            num_refiner_layers=1, ffn_dim=16, in_channels=1, audio_in_channels=2,
            text_dim=8, freq_dim=8, time_embed_hidden_dim=8, time_embed_dim=8,
            rope_freq_dim=1,
        )
    part = tiers.partition("dit", root, tiers.DEVICE)
    assert part.blocks
    assert root.table_coverage is dict(root.named_buffers())["table_coverage"]
    assert any(name == "table_coverage" for _, name, _, _ in part.common.slots())
    source = torch.nn.Module()
    armed = _Armed(source)
    root.time_proj._armed = root.norm_out._armed = armed
    save, restore = root._cozy_retry_state
    state = save()
    root._release()
    assert vars(root.time_proj)["_armed"] is vars(root.norm_out)["_armed"] is None
    restore(state)
    assert vars(root.time_proj)["_armed"] is vars(root.norm_out)["_armed"] is armed
    assert armed.source is source
    assert all(block.adaln_proj._armed is None for block in root.transformer_blocks)


def test_h3_observation_contract_composes_rollback_and_restores_hook_ownership() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.author import retry_state
    from cozy_runtime.models.minimax_h3.official import NumericalChecks

    root = torch.nn.Linear(2, 2)
    root.counter = 4
    retry_state(root, lambda: root.counter, lambda value: setattr(root, "counter", value))
    previous = root._cozy_retry_state
    checks = NumericalChecks(cast(Any, None))  # Rollback empties observations before settlement.
    with checks.forwards(root, "dit"):
        save, restore = root._cozy_retry_state
        state = save()
        before = next(iter(root._forward_pre_hooks.values()))
        after = next(iter(root._forward_hooks.values()))
        tensor = torch.ones(2)
        before(root, (tensor,), {})
        after(root, (), {}, tensor)
        root.counter = 5
        assert save()[1:] == (1, 2)
        restore(state)
        assert root.counter == 4 and save()[1:] == (0, 0)
    assert root._cozy_retry_state is previous
    assert not root._forward_pre_hooks and not root._forward_hooks
