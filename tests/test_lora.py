"""Real CPU Linear arithmetic and adapter lifetime; no GPU/group quality claim."""

from __future__ import annotations

from contextvars import copy_context
from dataclasses import replace
from typing import Any

import pytest

from cozy_runtime.internal.lora import TILE_ROWS, AdapterInput, AdapterRefusal, LinearAdapters

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
            torch.testing.assert_close(actual, wanted, rtol=5e-6, atol=5e-7)
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
    a = source.tensors["0.lora_A.weight"].clone()
    b = source.tensors["0.lora_B.weight"].clone()
    expected = (
        base.float()
        + torch.nn.functional.linear(torch.nn.functional.linear(x.float(), a), b) * 0.75
    ).to(base.dtype)
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


def test_defaults_alpha_keyword_input_and_preserved_fp32_factor_range() -> None:
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
        torch.testing.assert_close(actual, base + delta * source.scale, rtol=5e-6, atol=5e-7)
        assert not actual.requires_grad
    model, binding, source, x = _fixture(torch.float16)
    values = dict(source.tensors)
    values["0.lora_A.weight"] = torch.full((2, 4), 1e10)
    with _request(binding, replace(source, tensors=values)):
        attempt = binding._active.get()
        assert attempt is not None
        factor = attempt.terms[("dit", "0")][0].a
        assert factor.dtype == torch.float32
        assert torch.equal(factor, values["0.lora_A.weight"])


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
    assert _rowwise_leaf(leaf)
    source = AdapterInput(
        "adapter",
        "dit",
        "base",
        {
            "0.lora_A.weight": torch.ones(2, 4),
            "0.lora_B.weight": torch.ones(3, 2),
        },
    )
    with _request(binding, source):
        assert not _rowwise_leaf(leaf)
    with _request(binding, replace(source, scale=0)):
        assert _rowwise_leaf(leaf)
    with _request(binding):
        assert _rowwise_leaf(leaf)
    for key, address, raw in before:
        assert leaf.roles()[key].data_ptr() == address
        assert torch.equal(leaf.roles()[key].view(torch.uint8), raw)
    binding.close()
    assert _rowwise_leaf(leaf)
    # Construction/guard proof only: the encoded GEMM itself requires a CUDA worker.


def test_retained_exception_and_context_release_attempt_factor_storage() -> None:
    import gc
    import weakref

    model, binding, source, x = _fixture()
    refs: list[Any] = []
    cancel = False

    def check() -> None:
        if cancel:
            raise InterruptedError("cancelled")

    with (
        pytest.raises(InterruptedError) as retained,
        binding.request([source], device="cpu", max_device_bytes=4096, check_cancelled=check),
    ):
        copied = copy_context()
        attempt = binding._active.get()
        assert attempt is not None
        for terms in attempt.terms.values():
            for term in terms:
                refs.extend((weakref.ref(term.a), weakref.ref(term.b)))
        # Retain the term, context and exception for diagnostics, as worker failure
        # reporting does. None may keep the attempt's factor allocations alive.
        cancel = True
        model(x)
    gc.collect()
    assert retained.value is not None and copied is not None
    assert all(ref() is None for ref in refs)


def test_a_coalesced_leaf_cannot_acquire_two_independent_adapter_owners() -> None:
    model, binding, source, x = _fixture()
    with pytest.raises(AdapterRefusal) as error:
        LinearAdapters({"dit": model}, {"dit": "base-A"}, {"dit": ["0"]})
    assert error.value.code == "adapter_target_owned"
    assert len(model[0]._forward_hooks) == 1
    with _request(binding, source):
        assert model(x).shape == (2, 5, 3)


def test_each_stacked_term_budgets_above_current_live_outputs() -> None:
    import weakref

    from torch.utils._python_dispatch import TorchDispatchMode

    model, binding, source, x = _fixture()
    held: dict[int, list[Any]] = {}
    existing = {value.untyped_storage().data_ptr() for value in [x, *model.parameters()]}
    samples: list[tuple[int, int]] = []
    peak = 0
    measuring = False

    def live_bytes() -> int:
        total = 0
        for references in held.values():
            tensor = next((ref() for ref in references if ref() is not None), None)
            if tensor is not None:
                total += tensor.untyped_storage().nbytes()
        return total

    dispatch_base: Any = TorchDispatchMode

    class Track(dispatch_base):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()

        def __torch_dispatch__(
            self, func: Any, types: Any, args: Any = (), kwargs: Any = None
        ) -> Any:
            nonlocal peak
            result = func(*args, **(kwargs or {}))
            if isinstance(result, torch.Tensor):
                address = result.untyped_storage().data_ptr()
                if address not in existing:
                    held.setdefault(address, []).append(weakref.ref(result))
            peak = max(peak, live_bytes())
            return result

    def capacity(needed: int) -> None:
        if measuring:
            samples.append((live_bytes(), needed))

    with binding.request(
        [source, replace(source, scale=-0.25)],
        device="cpu",
        max_device_bytes=4096,
        check_cancelled=lambda: None,
        check_capacity=capacity,
    ):
        attempt = binding._active.get()
        assert attempt is not None
        for terms in attempt.terms.values():
            for term in terms:
                existing.update(
                    (term.a.untyped_storage().data_ptr(), term.b.untyped_storage().data_ptr())
                )
        measuring = True
        with Track():
            result = model(x)
        assert result.shape == (2, 5, 3)
    output_bytes = 2 * 5 * 3 * 4
    static_scratch = TILE_ROWS * (4 + 2 + 2 * 3) * 4
    assert samples == [(output_bytes, static_scratch)]
    assert peak <= max(live + additional for live, additional in samples)


def test_fixed_tiles_preserve_cpu_residual_bits_across_row_partitions() -> None:
    model, binding, source, _ = _fixture(torch.bfloat16)
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    x = torch.randn(TILE_ROWS + 9, 4, dtype=torch.bfloat16)
    with _request(binding, source):
        whole = model(x)
    for count in (2, 4):
        with _request(binding, source):
            pieces = torch.cat([model(part) for part in x.tensor_split(count)])
        assert torch.equal(whole, pieces)


def test_unsafe_layout_and_preceding_output_hook_refuse() -> None:
    model, binding, source, _ = _fixture()
    x = torch.ones(4, 8).t()  # [8,4], noncontiguous: must not copy a whole sequence
    with pytest.raises(AdapterRefusal) as error, _request(binding, source):
        model(x)
    assert error.value.code == "adapter_input_layout"
    x = torch.ones(8, 4)
    hook = model[0].register_forward_hook(lambda *_: torch.ones(8, 3), prepend=True)
    try:
        with pytest.raises(AdapterRefusal) as error, _request(binding, source):
            model(x)
        assert error.value.code == "adapter_forward_unsupported"
    finally:
        hook.remove()


def test_float32_partition_error_is_bounded_without_claiming_bit_identity() -> None:
    model, binding, source, _ = _fixture()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    x = torch.randn(TILE_ROWS + 9, 4)
    with _request(binding, source):
        whole = model(x)
    with _request(binding, source):
        split = torch.cat([model(part) for part in x.tensor_split(4)])
    # CPU BLAS may change reduction details with a row's position inside a tile.
    # This is mathematical agreement only; CUDA/Ulysses needs its own byte proof.
    torch.testing.assert_close(whole, split, rtol=5e-6, atol=5e-7)


def test_fp32_stack_rounds_back_to_bf16_only_after_all_adapters() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, dtype=torch.bfloat16))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(1.0)
    binding = LinearAdapters({"dit": model}, {"dit": "base"}, {"dit": ["0"]})
    source = AdapterInput(
        "style",
        "dit",
        "base",
        {
            "0.lora_A.weight": torch.ones(1, 1),
            "0.lora_B.weight": torch.full((1, 1), 0.003),
        },
    )
    x = torch.ones(1, 1, dtype=torch.bfloat16)
    with _request(binding, source, source):
        actual = model(x)
    expected = torch.tensor([[1.0 + 0.003 + 0.003]], dtype=torch.bfloat16)
    assert torch.equal(actual, expected)
    assert actual.item() == 1.0078125


@pytest.mark.parametrize("rows", [1, TILE_ROWS, TILE_ROWS + 17, 2 * TILE_ROWS + 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_reused_scratch_matches_old_tiles_with_poisoned_empty_storage(
    rows: int, dtype: Any, monkeypatch: Any
) -> None:
    model, binding, source, _ = _fixture(dtype)
    x = torch.randn(rows, 4, dtype=dtype)
    expected = model(x).clone()
    linear = torch.nn.functional.linear
    a, b = source.tensors["0.lora_A.weight"], source.tensors["0.lora_B.weight"]
    # Freeze the previous zero-initialized per-tile arithmetic as the parity oracle.
    for start in range(0, rows, TILE_ROWS):
        count = min(TILE_ROWS, rows - start)
        tile = torch.zeros(TILE_ROWS, 4)
        accumulator = torch.zeros(TILE_ROWS, 3)
        tile[:count].copy_(x[start : start + count])
        accumulator[:count].copy_(expected[start : start + count])
        for coefficient in (0.75, -0.375):
            delta = linear(linear(tile, a), b)
            delta.mul_(coefficient)
            accumulator.add_(delta)
        expected[start : start + count].copy_(accumulator[:count])
    original_empty = torch.empty

    def poisoned(*args: Any, **kwargs: Any) -> Any:
        tensor = original_empty(*args, **kwargs)
        if tensor.dtype == torch.float32:
            tensor.fill_(float("nan"))
        return tensor

    monkeypatch.setattr(torch, "empty", poisoned)
    with _request(binding, source, replace(source, scale=-0.25)):
        actual = model(x)
    assert torch.equal(actual, expected)
    assert bool(torch.isfinite(actual).all())
