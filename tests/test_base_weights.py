"""Real PEFT/base ownership across eviction and logical checkpoint restaging."""

from types import SimpleNamespace

import pytest
from tensorfs import from_parts

from cozy_runtime.author._loader import census
from cozy_runtime.internal.base_weights import BaseWeights
from rowwise_fixture import rowwise_spec

torch = pytest.importorskip("torch")


def test_peft_parameters_survive_base_eviction_and_logical_slots_survive_nesting() -> None:
    pytest.importorskip("torchao")
    peft = pytest.importorskip("peft")
    torch.manual_seed(623)
    base = torch.nn.Linear(16, 32, dtype=torch.bfloat16)
    root = torch.nn.Sequential(base)
    root.register_buffer("derived", torch.arange(8), persistent=False)
    walked = census(SimpleNamespace(components={"dit": root}))
    owned = BaseWeights("dit", root, walked)
    source = {key: value.clone() for key, value in owned.live().items()}
    payload = torch.randn(32, 16).to(torch.float8_e4m3fn)
    scale = torch.rand(32, 1, dtype=torch.float32)
    base.weight = torch.nn.Parameter(
        from_parts(rowwise_spec(payload.shape), {"data": payload, "scale": scale}),
        requires_grad=False,
    )
    model = peft.get_peft_model(
        root,
        peft.LoraConfig(r=2, lora_alpha=2, target_modules=["0"], init_lora_weights=False),
        autocast_adapter_dtype=False,
    ).eval()
    value = torch.randn(3, 16, dtype=torch.bfloat16)
    with torch.no_grad():
        # Explicit upstream CPU weight-only forward, not emulated native W8A8.
        expected = model(value).clone()
        factors = {
            name: (parameter, parameter.clone())
            for name, parameter in model.named_parameters()
            if ".lora_" in name
        }
        assert list(owned.live()) == ["dit.0.weight", "dit.0.bias"]
        encoded_alias = base.weight.qdata
        codes = encoded_alias.view(torch.uint8).clone()
        owned.evict()
        assert base.weight.device.type == base.weight.qdata.device.type == "meta"
        assert root.derived.device.type == "meta"
        # Another owner's alias remains valid: eviction never resizes its storage.
        assert torch.equal(encoded_alias.view(torch.uint8), codes)
        assert all(
            parameter is factors[name][0] and torch.equal(parameter, factors[name][1])
            for name, parameter in model.named_parameters()
            if name in factors
        )
        owned.reserve("cpu")
        restored = owned.live()
        assert list(restored) == list(source)
        for key, tensor in restored.items():
            tensor.copy_(source[key])
        # The ordinary fill transaction adopts the same encoded parts again.
        base.weight = torch.nn.Parameter(
            from_parts(rowwise_spec(payload.shape), {"data": payload, "scale": scale}),
            requires_grad=False,
        )
        assert torch.equal(model(value), expected)
        assert all(
            parameter is factors[name][0] and torch.equal(parameter, factors[name][1])
            for name, parameter in model.named_parameters()
            if name in factors
        )


def test_tied_parameter_objects_remain_tied_and_unrelated_late_parameters_stay_live() -> None:
    root = torch.nn.Module()
    root.left = torch.nn.Linear(4, 4, bias=False)
    root.right = torch.nn.Linear(4, 4, bias=False)
    root.right.weight = root.left.weight
    walked = census(SimpleNamespace(components={"model": root}))
    owned = BaseWeights("model", root, walked)
    root.register_parameter("request_parameter", torch.nn.Parameter(torch.ones(2)))
    adapter = root.request_parameter
    owned.evict()
    assert root.left.weight is root.right.weight
    assert root.request_parameter is adapter and adapter.device.type == "cpu"
    owned.reserve("cpu")
    assert root.left.weight is root.right.weight
    assert list(owned.live()) == ["model.left.weight", "model.right.weight"]
    with pytest.raises(ValueError, match="evicted"):
        owned.reserve("cpu")


def test_shared_views_refuse_before_moving_any_tensor() -> None:
    root = torch.nn.Module()
    value = torch.arange(16, dtype=torch.float32)
    root.register_parameter("left", torch.nn.Parameter(value[:8]))
    root.register_parameter("right", torch.nn.Parameter(value[8:]))
    walked = census(SimpleNamespace(components={"model": root}))
    with pytest.raises(ValueError, match="shared views"):
        BaseWeights("model", root, walked)
    assert torch.equal(value, torch.arange(16, dtype=torch.float32))


def test_actual_h3_meta_construction_has_stable_supported_base_slots() -> None:
    diffusers = pytest.importorskip("diffusers")
    with torch.device("meta"):
        transformer = diffusers.MiniMaxH3Transformer3DModel()
    walked = census(SimpleNamespace(components={"dit": transformer}))
    owned = BaseWeights("dit", transformer, walked)
    assert list(owned.live()) == [destination.key for destination in walked.destinations]
    assert len(transformer.transformer_blocks) == 50
    assert all(value.device.type == "meta" for value in owned.tensors())
