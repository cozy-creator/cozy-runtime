"""Actual PEFT arithmetic, ownership, hook semantics and admission failures."""

import contextvars
import gc
import weakref
from types import MethodType
from typing import Any

import pytest

from cozy_runtime.internal.encoding.leaves import PreQuantized
from cozy_runtime.internal.encoding.torchao_rowwise import bind_rowwise_weight
from cozy_runtime.internal.lora import (
    AdapterInput,
    AdapterRefusal,
    LinearAdapters,
    prequantized_input_allowed,
)

torch = pytest.importorskip("torch")
pytest.importorskip("peft")


def fixture(dtype: Any = None) -> tuple[Any, Any, LinearAdapters, AdapterInput]:
    dtype = torch.bfloat16 if dtype is None else dtype
    base = torch.nn.Linear(4, 6, bias=False, dtype=dtype)
    with torch.no_grad():
        base.weight.fill_(0.5)
    model = torch.nn.Sequential(base)
    binding = LinearAdapters({"dit": model}, {"dit": "base"}, {"dit": ["0"]})
    adapter = AdapterInput(
        "adapter",
        "dit",
        "base",
        {"0.lora_A.weight": torch.full((2, 4), 0.25), "0.lora_B.weight": torch.full((6, 2), 0.5)},
        scale=0.5,
    )
    return model, base, binding, adapter


def scope(binding: LinearAdapters, *adapters: AdapterInput, **kwargs: Any) -> Any:
    return binding.request(
        adapters,
        device="cpu",
        max_device_bytes=10000,
        check_cancelled=kwargs.pop("check_cancelled", lambda: None),
        **kwargs,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_peft_computes_full_sequence_with_request_dtype_and_returns_to_base(
    dtype: Any,
) -> None:
    model, base, binding, adapter = fixture(dtype)
    x = torch.ones(8209, 4, dtype=dtype)
    baseline = base(x).detach().clone()
    capacity: list[int] = []
    assert model[0].base_layer is base
    assert all(p.device.type == "meta" for p in model[0].lora_A.parameters())
    with scope(binding, adapter, check_capacity=capacity.append):
        assert binding.preparation_facts() == {
            "factor_device_bytes": 20 * x.element_size(),
            "factor_tensors_converted": 0 if dtype == torch.float32 else 2,
            "degraded_mode": dtype != torch.float32,
        }
        assert model[0].lora_A["request_0"].weight.dtype == dtype
        assert torch.equal(model(x), baseline + 0.5)
        assert torch.equal(model(input=x), baseline + 0.5)
        assert torch.equal(model[0](input=x), baseline + 0.5)
        assert binding.observed_calls() == 3
    width = x.element_size()
    assert capacity == [20 * width, *([8209 * 20 * width] * 3)]
    assert list(model[0].lora_A) == ["_inactive"]
    assert torch.equal(model(x), baseline)
    binding.close()
    binding.close()
    assert model[0] is base


def test_retained_context_cancellation_and_source_mutation_cannot_retain_active_factors() -> None:
    model, base, binding, adapter = fixture()
    x = torch.ones(3, 4, dtype=torch.bfloat16)
    before = base.weight.detach().clone()
    with scope(binding, adapter):
        factor = weakref.ref(model[0].lora_A["request_0"].weight)
        saved = contextvars.copy_context()
        adapter.tensors["0.lora_A.weight"].add_(10)
        assert torch.equal(model(x), torch.full((3, 6), 2.5, dtype=x.dtype))
        with pytest.raises(AdapterRefusal, match="escaped"):
            contextvars.Context().run(model, x)
        with pytest.raises(AdapterRefusal, match="active request"), scope(binding, adapter):
            pass
    gc.collect()
    assert factor() is None
    with pytest.raises(AdapterRefusal, match="outlived"):
        saved.run(model, x)
    assert torch.equal(base.weight, before)
    with (
        pytest.raises(RuntimeError, match="cancelled"),
        scope(
            binding,
            adapter,
            check_cancelled=lambda: (_ for _ in ()).throw(RuntimeError("cancelled")),
        ),
    ):
        pass
    with scope(binding):
        assert torch.equal(model(x), base(x))
    binding.close()


def test_capacity_refuses_before_base_forward_and_active_lora_refuses_prequantized_input() -> None:
    model, _base, binding, adapter = fixture()
    completed = []
    handle = model[0].register_forward_hook(lambda _module, _args, _out: completed.append(True))

    def capacity(nbytes: int) -> None:
        if nbytes > 1000:
            raise RuntimeError("capacity")

    with scope(binding, adapter, check_capacity=capacity):
        with pytest.raises(RuntimeError, match="capacity"):
            model(torch.ones(8209, 4, dtype=torch.bfloat16))
        assert not completed
        assert not prequantized_input_allowed(model[0])
        with pytest.raises(AdapterRefusal, match="original floating"):
            model(
                PreQuantized(torch.zeros(2, 4, dtype=torch.float8_e4m3fn), torch.ones(2, 1), (2, 4))
            )
    handle.remove()
    assert prequantized_input_allowed(model[0])
    binding.close()


def test_existing_hook_order_and_handles_survive_native_peft_and_restoration() -> None:
    base = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        base.weight.fill_(0.5)
    pre = base.register_forward_pre_hook(lambda _module, args: (args[0] * 2,))
    post = base.register_forward_hook(lambda _module, _args, output: output + 7)
    model = torch.nn.Sequential(base)
    binding = LinearAdapters({"dit": model}, {"dit": "base"}, {"dit": ["0"]})
    adapter = AdapterInput(
        "one",
        "dit",
        "base",
        {
            "0.lora_A.weight": torch.full((2, 4), 0.25),
            "0.lora_B.weight": torch.full((6, 2), 0.5),
        },
        scale=0.5,
    )
    x = torch.ones(3, 4, dtype=torch.bfloat16)
    with scope(binding, adapter):
        assert torch.equal(model(x), torch.full((3, 6), 12.0, dtype=x.dtype))
        pre.remove()
        assert torch.equal(model(x), torch.full((3, 6), 9.5, dtype=x.dtype))
        post.remove()
        assert torch.equal(model(x), torch.full((3, 6), 2.5, dtype=x.dtype))
    binding.close()
    assert model[0] is base and not base._forward_hooks and not base._forward_pre_hooks


def test_factor_cast_overflow_refuses_preparation_and_leaves_binding_reusable() -> None:
    model, _base, binding, adapter = fixture(torch.float16)
    adapter.tensors["0.lora_A.weight"].fill_(100000)
    with pytest.raises(AdapterRefusal, match="overflow"), scope(binding, adapter):
        pass
    assert list(model[0].lora_A) == ["_inactive"]
    adapter.tensors["0.lora_A.weight"].fill_(0.25)
    with scope(binding, adapter):
        assert torch.isfinite(model(torch.ones(2, 4, dtype=torch.float16))).all()
    binding.close()


def test_failure_after_execution_releases_factors_even_with_retained_traceback() -> None:
    model, _base, binding, adapter = fixture()
    held_error = None
    before_rng = torch.random.get_rng_state().clone()
    try:
        with scope(binding, adapter):
            factor = weakref.ref(model[0].lora_B["request_0"].weight)
            model(torch.ones(2, 4, dtype=torch.bfloat16))
            raise RuntimeError("request failed")
    except RuntimeError as exc:
        held_error = exc
    assert held_error is not None and held_error.__traceback__ is not None
    gc.collect()
    assert factor() is None
    assert list(model[0].lora_B) == ["_inactive"]
    assert torch.equal(before_rng, torch.random.get_rng_state())
    with scope(binding, adapter):
        assert torch.isfinite(model(torch.ones(2, 4, dtype=torch.bfloat16))).all()
    binding.close()


def test_two_adapters_keep_order_strengths_and_repeat_exactly() -> None:
    model, _base, binding, first = fixture()
    second = AdapterInput("second", "dit", "base", first.tensors, scale=0.25)
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    results = []
    for adapters in ((first,), (second,), (first, second), (first,)):
        with scope(binding, *adapters):
            results.append(model(x).clone())
    assert torch.equal(results[0], torch.full((2, 6), 2.5, dtype=x.dtype))
    assert torch.equal(results[1], torch.full((2, 6), 2.25, dtype=x.dtype))
    assert torch.equal(results[2], torch.full((2, 6), 2.75, dtype=x.dtype))
    assert torch.equal(results[0], results[3])
    binding.close()


def test_cuda_request_without_temporary_admission_refuses_before_device_access() -> None:
    _model, _base, binding, adapter = fixture()
    with (
        pytest.raises(AdapterRefusal, match="temporary memory admission"),
        binding.request(
            [adapter], device="cuda", max_device_bytes=10000, check_cancelled=lambda: None
        ),
    ):
        pass
    binding.close()


def test_standard_weight_admission_includes_compatible_base_quantizer_workspace() -> None:
    pytest.importorskip("torchao")
    base = torch.nn.Linear(4, 6, bias=False, dtype=torch.bfloat16)
    bind_rowwise_weight(base, torch.ones(6, 4, dtype=torch.float8_e4m3fn), torch.ones(6, 1))
    model = torch.nn.Sequential(base)
    binding = LinearAdapters({"dit": model}, {"dit": "base"}, {"dit": ["0"]})
    adapter = AdapterInput(
        "one",
        "dit",
        "base",
        {"0.lora_A.weight": torch.ones(2, 4), "0.lora_B.weight": torch.ones(6, 2)},
    )
    capacity: list[int] = []
    with (
        scope(binding, adapter, check_capacity=capacity.append),
        pytest.raises(RuntimeError, match="no weight-only fallback"),
    ):
        model(torch.ones(8209, 4, dtype=torch.bfloat16))
    assert capacity == [40, 8209 * (4 * 10 + 6 * 2 + 12)]
    binding.close()


def test_late_global_hooks_disable_prequantization_and_refuse_adapter_execution() -> None:
    model, _base, binding, adapter = fixture()
    handle = torch.nn.modules.module.register_module_forward_hook(lambda _module, _args, _out: None)
    try:
        assert not prequantized_input_allowed(model[0])
        with scope(binding, adapter), pytest.raises(AdapterRefusal, match="global hooks"):
            model(torch.ones(2, 4, dtype=torch.bfloat16))
    finally:
        handle.remove()
        binding.close()


def test_prepared_bf16_factors_do_not_report_a_compatibility_cast() -> None:
    _model, _base, binding, original = fixture()
    adapter = AdapterInput(
        "prepared",
        "dit",
        "base",
        {key: value.to(torch.bfloat16) for key, value in original.tensors.items()},
    )
    with scope(binding, adapter):
        assert binding.preparation_facts() == {
            "factor_device_bytes": 40,
            "factor_tensors_converted": 0,
            "degraded_mode": False,
        }
    binding.close()


def test_actual_diffusers_scale_decorator_cannot_silently_reset_request_strength() -> None:
    pytest.importorskip("diffusers")
    from diffusers.utils import apply_lora_scale

    model, _base, binding, adapter = fixture()

    def forward(self: Any, x: Any, attention_kwargs: Any = None) -> Any:
        return self.dit(x)

    pipeline = torch.nn.Module()
    pipeline.dit = model
    pipeline.forward = MethodType(apply_lora_scale("attention_kwargs")(forward), pipeline)
    x = torch.ones(2, 4, dtype=torch.bfloat16)
    with scope(binding, adapter):
        expected = pipeline(x)
        assert torch.equal(pipeline(x, attention_kwargs={"scale": 1.0}), expected)
        with pytest.raises(AdapterRefusal, match="strengths changed"):
            pipeline(x, attention_kwargs={"scale": 0.0})
    with scope(binding, adapter):
        assert torch.equal(pipeline(x), expected)
    binding.close()


@pytest.mark.parametrize("changed", [0, 1])
def test_failed_close_keeps_other_targets_gated_and_prevents_request_reuse(changed: int) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    binding = LinearAdapters({"dit": model}, {"dit": "base"}, {"dit": ["0", "1"]})
    other = model[1 - changed]
    model[changed] = torch.nn.Identity()
    with pytest.raises(AdapterRefusal, match="target was replaced"):
        binding.close()
    assert model[1 - changed] is other
    assert len(other._forward_pre_hooks) == 1
    with pytest.raises(AdapterRefusal, match="closed or poisoned"):
        binding.check_ready()
    with pytest.raises(AdapterRefusal, match="closed or poisoned"):
        other(torch.ones(1000, 4))
    with pytest.raises(AdapterRefusal, match="closed or poisoned"), scope(binding):
        pass


@pytest.mark.parametrize(
    "kwargs",
    [
        {"adapter_names": ["__base__", "__base__"]},
        {"adapter_names": ["request_0", "request_0"]},
        {"alora_offsets": [1, 1]},
    ],
)
def test_peft_selection_and_variant_kwargs_cannot_override_request(kwargs: dict[str, Any]) -> None:
    model, _base, binding, adapter = fixture()
    with scope(binding, adapter), pytest.raises(AdapterRefusal, match="single Linear input"):
        model[0](torch.ones(2, 4, dtype=torch.bfloat16), **kwargs)
    binding.close()


def test_inner_adapter_failure_retaining_an_operand_requires_generation_retirement() -> None:
    from torch.utils._python_dispatch import TorchDispatchMode

    model, _base, binding, adapter = fixture(torch.float32)
    dispatch_base: Any = TorchDispatchMode

    class FailAdapterMM(dispatch_base):  # type: ignore[misc]
        def __init__(self, pointer: int) -> None:
            super().__init__()
            self.pointer = pointer

        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            if func == torch.ops.aten.mm.default and any(
                isinstance(value, torch.Tensor)
                and value.untyped_storage().data_ptr() == self.pointer
                for value in args
            ):
                raise RuntimeError("injected native adapter dispatch refusal")
            return func(*args, **(kwargs or {}))

    held_error = None
    try:
        with scope(binding, adapter):
            pointer = model[0].lora_A["request_0"].weight.untyped_storage().data_ptr()
            with FailAdapterMM(pointer):
                model(torch.ones(2, 4))
    except RuntimeError as exc:
        held_error = exc
    assert held_error is not None
    tb = held_error.__traceback__
    retained: list[Any] = []
    while tb is not None:
        if tb.tb_frame.f_code.co_name == "__torch_dispatch__":
            retained.extend(
                value
                for value in tb.tb_frame.f_locals["args"]
                if isinstance(value, torch.Tensor) and value.untyped_storage().data_ptr() == pointer
            )
        tb = tb.tb_next
    assert retained  # Actual operands remain valid; cleanup did not resize aliases.
    assert all(value.untyped_storage().nbytes() > 0 for value in retained)
    with pytest.raises(AdapterRefusal, match="closed or poisoned"):
        binding.check_ready()
    with pytest.raises(AdapterRefusal, match="closed or poisoned"), scope(binding, adapter):
        pass
    binding.close()
