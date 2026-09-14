"""Real CPU Linear arithmetic and adapter lifetime; no GPU/group quality claim."""

from __future__ import annotations

from contextvars import copy_context
from dataclasses import replace
from typing import Any

import pytest

from cozy_runtime.internal.lora import AdapterInput, AdapterRefusal, LinearAdapters

torch = pytest.importorskip("torch")


def _fixture(dtype: Any = None) -> tuple[Any, LinearAdapters, AdapterInput, Any]:
    dtype = torch.float32 if dtype is None else dtype
    torch.manual_seed(1309)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3, dtype=dtype))
    binding = LinearAdapters({"dit": model}, {"dit": "base-A"}, {"dit": ["0"]})
    source = AdapterInput(
        "adapter-A",
        "dit",
        "base-A",
        {
            "0.lora_A.weight": torch.randn(2, 4),
            "0.lora_B.weight": torch.randn(3, 2),
            "0.alpha": torch.tensor(3.0),
        },
        scale=0.5,
    )
    return model, binding, source, torch.randn(2, 5, 4, dtype=dtype)


def _request(binding: LinearAdapters, *sources: AdapterInput, **kwargs: Any) -> Any:
    return binding.request(
        sources, device="cpu", max_device_bytes=4096, check_cancelled=lambda: None, **kwargs
    )


def test_ordered_additive_strengths_alpha_and_request_isolation() -> None:
    model, binding, source, x = _fixture()
    before = {k: value.detach().clone() for k, value in model.state_dict().items()}
    hook_ids = tuple(model[0]._forward_hooks)
    base = model(x).detach()
    other = replace(source, ref="adapter-B", scale=-0.25)
    linear = torch.nn.functional.linear
    delta = linear(linear(x, source.tensors["0.lora_A.weight"]), source.tensors["0.lora_B.weight"])
    expected = base + delta * (0.5 * 3 / 2)
    expected = expected + delta * (-0.25 * 3 / 2)
    for sources in ((source, other), (), (other,), (source, other)):
        with _request(binding, *sources) as refs:
            assert [(r.ref, r.scale) for r in refs] == [(s.ref, s.scale) for s in sources]
            actual = model(x)
            wanted = (
                expected
                if len(sources) == 2
                else base + delta * (-0.25 * 3 / 2)
                if sources
                else base
            )
            assert torch.equal(actual, wanted)
        assert torch.equal(model(x), base)
        assert tuple(model[0]._forward_hooks) == hook_ids
        assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    binding.close()
    assert not model[0]._forward_hooks


def test_zero_scale_is_exact_and_validates_bad_factors() -> None:
    model, binding, source, x = _fixture(torch.bfloat16)
    base = model(x)
    zero = replace(source, scale=0.0)
    with binding.request([zero], device="cpu", max_device_bytes=0, check_cancelled=lambda: None):
        assert torch.equal(model(x), base)
    invalid = dict(zero.tensors)
    invalid["0.lora_A.weight"] = torch.full((2, 4), float("nan"))
    with (
        pytest.raises(AdapterRefusal, match="nonfinite"),
        _request(binding, replace(zero, tensors=invalid)),
    ):
        pytest.fail("invalid zero-scale tensors admitted")


def test_compute_dtype_and_factor_source_do_not_alias() -> None:
    model, binding, source, x = _fixture(torch.bfloat16)
    base = model(x)
    a = source.tensors["0.lora_A.weight"].to(torch.bfloat16)
    b = source.tensors["0.lora_B.weight"].to(torch.bfloat16)
    expected = base + torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * 0.75
    with _request(binding, source):
        source.tensors["0.lora_A.weight"].zero_()
        assert torch.equal(model(x), expected)
        assert model(x).dtype == torch.bfloat16
    assert torch.equal(model(x), base)


def test_cancellation_exception_stale_context_and_concurrent_reuse() -> None:
    model, binding, source, x = _fixture()
    base = model(x)
    cancel = False

    def check() -> None:
        if cancel:
            raise InterruptedError("cancelled")

    with (
        pytest.raises(InterruptedError),
        binding.request([source], device="cpu", max_device_bytes=4096, check_cancelled=check),
    ):
        stale = copy_context()
        assert not torch.equal(model(x), base)
        with pytest.raises(AdapterRefusal, match="active attempt"), _request(binding):
            pytest.fail("concurrent attempt admitted")
        with pytest.raises(AdapterRefusal, match="active adapter binding"):
            binding.close()
        cancel = True
        model(x)
    assert torch.equal(model(x), base)
    with pytest.raises(AdapterRefusal, match="outlived"):
        stale.run(model, x)
    with pytest.raises(ValueError, match="body failed"), _request(binding, source):
        raise ValueError("body failed")
    with _request(binding, source):
        assert not torch.equal(model(x), base)
    assert torch.equal(model(x), base)


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"base_snapshot": "another-base"}, "adapter_base_mismatch"),
        ({"scale": float("nan")}, "adapter_identity"),
        ({"tensors": {}}, "adapter_empty"),
        ({"tensors": {"unrecognized": torch.ones(1)}}, "adapter_key"),
        ({"tensors": {"0.lora_A.weight": torch.ones(2, 4)}}, "adapter_pair"),
        ({"tensors": {"missing.lora_A.weight": torch.ones(2, 4)}}, "adapter_target_missing"),
        (
            {"tensors": {"0.lora_A.weight": torch.ones(2, 4), "0.lora_B.weight": torch.ones(4, 2)}},
            "adapter_shape",
        ),
        (
            {
                "tensors": {
                    "0.lora_A.weight": torch.ones(2, 4),
                    "0.lora_B.weight": torch.ones(3, 2),
                    "0.alpha": torch.ones(1),
                }
            },
            "adapter_alpha",
        ),
        (
            {
                "tensors": {
                    "0.lora_A.weight": torch.ones(2, 4, dtype=torch.int8),
                    "0.lora_B.weight": torch.ones(3, 2),
                }
            },
            "adapter_tensor",
        ),
    ],
)
def test_refusal_consumes_no_partial_stack(change: dict[str, Any], code: str) -> None:
    model, binding, source, x = _fixture()
    base = model(x)
    with (
        pytest.raises(AdapterRefusal) as error,
        _request(binding, source, replace(source, **change)),
    ):
        pytest.fail("bad adapter admitted")
    assert error.value.code == code
    assert torch.equal(model(x), base)
    with _request(binding, source):
        assert not torch.equal(model(x), base)


def test_budget_overflow_and_unsupported_forward_refuse_before_hooks() -> None:
    model, binding, source, x = _fixture()
    base = model(x)
    with (
        pytest.raises(AdapterRefusal) as error,
        binding.request([source], device="cpu", max_device_bytes=55, check_cancelled=lambda: None),
    ):
        pytest.fail("56-byte adapter exceeded budget")
    assert error.value.code == "adapter_budget"
    assert torch.equal(model(x), base)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    with pytest.raises(AdapterRefusal) as error:
        LinearAdapters({"dit": model}, {"dit": "base"}, {"dit": ["0", "1"]})
    assert error.value.code == "adapter_target_unsupported"
    assert not model[0]._forward_hooks


def test_alias_targets_refuse_and_active_context_cannot_be_bypassed() -> None:
    from contextvars import Context

    model, binding, source, x = _fixture()
    with _request(binding, source), pytest.raises(AdapterRefusal) as error:
        Context().run(model, x)
    assert error.value.code == "adapter_context_missing"
    leaf = torch.nn.Linear(4, 3)
    aliased = torch.nn.ModuleDict({"first": leaf, "second": leaf})
    with pytest.raises(AdapterRefusal) as error:
        LinearAdapters({"dit": aliased}, {"dit": "base"}, {"dit": ["first", "second"]})
    assert error.value.code == "adapter_target_alias"
    assert not leaf._forward_hooks


def test_defaults_alpha_keyword_input_and_fp16_overflow() -> None:
    model, binding, source, x = _fixture()
    without_alpha = replace(
        source, tensors={k: v for k, v in source.tensors.items() if not k.endswith(".alpha")}
    )
    base = model[0](input=x)
    delta = torch.nn.functional.linear(
        torch.nn.functional.linear(x, source.tensors["0.lora_A.weight"]),
        source.tensors["0.lora_B.weight"],
    )
    with _request(binding, without_alpha):
        actual = model[0](input=x)
        assert torch.equal(actual, base + delta * source.scale)
        assert not actual.requires_grad
    model, binding, source, x = _fixture(torch.float16)
    values = dict(source.tensors)
    values["0.lora_A.weight"] = torch.full((2, 4), 1e10)
    with pytest.raises(AdapterRefusal) as error, _request(binding, replace(source, tensors=values)):
        pytest.fail("finite FP32 values overflowed compute dtype")
    assert error.value.code == "adapter_cast_nonfinite"


def test_encoded_leaf_keeps_payload_and_disables_prequantized_carrier() -> None:
    from cozy_runtime.internal.encoding.leaves import RowwiseNativeLeaf
    from cozy_runtime.internal.fusion_install import _rowwise_leaf

    original = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    payload = torch.ones(3, 4).to(torch.float8_e4m3fn)
    scale = torch.ones(3, dtype=torch.float32)
    leaf = RowwiseNativeLeaf("test-encoding").leaf(
        torch, {"data": payload, "scale": scale}, original, torch.bfloat16
    )
    model = torch.nn.Sequential(leaf)
    assert _rowwise_leaf(leaf)
    before = [
        (key, value.data_ptr(), value.view(torch.uint8).clone())
        for key, value in leaf.roles().items()
    ]
    binding = LinearAdapters({"dit": model}, {"dit": "base"}, {"dit": ["0"]})
    assert not _rowwise_leaf(leaf)
    with _request(binding):
        assert not _rowwise_leaf(leaf)
    for key, address, raw in before:
        assert leaf.roles()[key].data_ptr() == address
        assert torch.equal(leaf.roles()[key].view(torch.uint8), raw)
    binding.close()
    assert _rowwise_leaf(leaf)
    # Construction/guard proof only: the encoded GEMM itself requires a CUDA worker.
