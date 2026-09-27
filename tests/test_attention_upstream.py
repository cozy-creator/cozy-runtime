"""Contract refusals on CPU; real upstream arithmetic only on explicitly opted-in GPUs.

Adapters are LOCAL kernels: LSE and context parallelism belong to `attention_ulysses`, which
wraps every one of them and is tested with the registered backend below.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from typing import Any

import pytest

from cozy_runtime.internal import attention, attention_ulysses
from cozy_runtime.internal import attention_upstream as upstream

torch = pytest.importorskip("torch")

ADAPTERS = [upstream.kitchen_int8, upstream.flashinfer_mixed]


@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"dropout_p": 0.1}, "dropout"),
        ({"is_causal": True}, "noncausal"),
        ({"scale": float("nan")}, "finite"),
        ({"scale": float("inf")}, "finite"),
    ],
)
def test_nondefault_unsupported_semantics_refuse(
    adapter: Callable[..., Any], options: dict[str, Any], reason: str
) -> None:
    q = torch.zeros((1, 2, 2, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=reason):
        adapter(q, q, q, **options)


@pytest.mark.parametrize(
    "name", ["kitchen-int8", "sageattention", "sageattention-fp16pv", "flash-attn3"]
)
@pytest.mark.parametrize(
    ("options", "reason"),
    [({"return_lse": True}, "log-sum-exp"), ({"_parallel_config": object()}, "Ulysses")],
)
def test_the_wrapper_refuses_lse_and_anything_but_initialized_ulysses(
    name: str, options: dict[str, Any], reason: str
) -> None:
    """Before any kernel or collective is reached: LSE is never returned, and a parallel
    config that is not Runtime's initialized Ulysses group refuses."""
    pytest.importorskip("diffusers")
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    member = attention._member(attention.BY_NAME[name])
    backend = _AttentionBackendRegistry._backends[member]
    q = torch.zeros((1, 2, 2, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=reason):
        backend(query=q, key=q, value=q, **options)


def test_a_cross_head_kernel_refuses_ulysses_before_any_collective() -> None:
    pytest.importorskip("diffusers")
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    member = attention._member(attention.BY_NAME["flashinfer-bf16-fp8"])
    assert not _AttentionBackendRegistry._is_context_parallel_available(member)
    q = torch.zeros((1, 2, 2, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="across heads"):
        _AttentionBackendRegistry._backends[member](
            query=q, key=q, value=q, _parallel_config=object()
        )


def test_sdpa_names_the_backend_torch_dispatches_and_matches_it() -> None:
    q = torch.randn((1, 16, 2, 64))
    options = {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": False,
        "scale": None,
        "enable_gqa": False,
    }
    expected = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in (q, q, q))
    ).transpose(1, 2)
    assert torch.equal(upstream.sdpa(q, q, q, **options), expected)
    assert upstream.sdpa_impl(q, q, q, options) in {"flash", "efficient", "math", "cudnn"}
    with attention_ulysses.observing() as seen:
        member = attention._member(attention.BY_NAME["sdpa"])
        pytest.importorskip("diffusers")
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        dispatch_attention_fn(q, q, q, backend=member)
    assert attention.implementations(seen) == upstream.sdpa_impl(q, q, q, options)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_cpu_is_not_a_silent_attention_fallback(adapter: Callable[..., Any]) -> None:
    q = torch.zeros((1, 2, 2, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="same CUDA device"):
        adapter(q, q, q)


@pytest.mark.parametrize("adapter", ADAPTERS)
def test_grouped_heads_require_explicit_permission_and_inputs_are_tensors(
    adapter: Callable[..., Any],
) -> None:
    q = torch.zeros((1, 2, 2, 128), dtype=torch.bfloat16)
    kv = q[:, :, :1]
    with pytest.raises(ValueError, match="enable_gqa=True"):
        adapter(q, kv, kv)
    with pytest.raises(ValueError, match="same CUDA device"):
        adapter(q, kv, kv, enable_gqa=True)
    with pytest.raises(ValueError, match="must be tensors"):
        adapter(q, None, q)


def test_upstream_output_contract_rejects_wrong_shape_dtype_or_non_tensor() -> None:
    q = torch.zeros((1, 2, 2, 128), dtype=torch.bfloat16)
    assert upstream._output(q.clone(), q).shape == q.shape
    for output in (None, (q,), q.float(), q[:, :1]):
        with pytest.raises(ValueError, match="query shape, dtype and device"):
            upstream._output(output, q)


def test_mixed_backend_does_not_cast_qk_or_discard_padding() -> None:
    q = torch.zeros((1, 2, 2, 128), dtype=torch.float16)
    with pytest.raises(ValueError, match="requires BF16 inputs"):
        upstream.flashinfer_mixed(q, q, q)
    with pytest.raises(ValueError, match="attention mask"):
        upstream.flashinfer_mixed(q, q, q, attn_mask=torch.ones((2, 2)))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_current_value_scale_preserves_inputs_and_zero_values(dtype: Any) -> None:
    value = torch.tensor([-100, -1, 0, 1, 100], dtype=dtype)
    original = value.clone()
    quantized, scale = upstream._quantize_value(value)
    assert torch.equal(value, original)
    assert quantized.dtype == torch.float8_e4m3fn
    assert torch.isfinite(quantized.float()).all()
    assert scale == 100 / 448
    assert float((quantized.float() * scale - value.float()).abs().max()) < 0.02
    zero, scale = upstream._quantize_value(torch.zeros_like(value))
    assert scale > 0
    assert torch.equal(zero.float(), torch.zeros_like(value).float())
    with pytest.raises(ValueError, match="non-finite"):
        upstream._quantize_value(torch.tensor([float("inf")]))


@pytest.mark.skipif(
    os.environ.get("COZY_TEST_ATTENTION_CUDA") != "1",
    reason="requires an explicitly owned CUDA device and upstream kernel installation",
)
@pytest.mark.parametrize("adapter", ADAPTERS)
@pytest.mark.parametrize("grouped", [False, True])
def test_real_upstream_matches_sdpa_and_preserves_inputs(
    adapter: Callable[..., Any], grouped: bool
) -> None:
    mixed = adapter is upstream.flashinfer_mixed
    dependency = "flashinfer" if mixed else "comfy_kitchen"
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    capability = torch.cuda.get_device_capability()
    if capability not in (((10, 0), (10, 3)) if mixed else ((9, 0), (10, 0), (10, 3))):
        pytest.skip(f"adapter does not serve {capability}")
    generator = torch.Generator(device="cuda").manual_seed(6057)
    kv_heads = 2 if grouped else 4
    # Packed projections leave gaps between heads/tokens, as actual H3 QKV does.
    q = torch.randn((2, 137, 4, 3, 128), device="cuda", dtype=torch.bfloat16, generator=generator)[
        :, :, :, 0, :
    ]
    kv = torch.randn((2, 259, kv_heads, 3, 128), device="cuda", dtype=q.dtype, generator=generator)
    k, v = kv[:, :, :, 1, :], kv[:, :, :, 2, :]
    originals = [tensor.clone() for tensor in (q, k, v)]
    mask = None
    if not mixed:
        mask = torch.ones((2, 1, 137, 259), device="cuda", dtype=torch.bool)
        mask[0, :, :, 129:] = False
    # A nondefault scale and batch-specific mask expose wrappers that drop options
    # or flatten batch boundaries. Actual upstream kernels run; there are no doubles.
    scale = 0.075
    with torch.inference_mode():
        got = adapter(q, k, v, attn_mask=mask, scale=scale, enable_gqa=grouped)
        repeat = adapter(q, k, v, attn_mask=mask, scale=scale, enable_gqa=grouped)
        exact = torch.nn.functional.scaled_dot_product_attention(
            *(tensor.transpose(1, 2).float() for tensor in (q, k, v)),
            attn_mask=mask,
            scale=scale,
            enable_gqa=grouped,
        ).transpose(1, 2)
    assert got.shape == q.shape
    assert got.dtype == q.dtype
    assert torch.isfinite(got).all()
    assert torch.equal(got, repeat)
    assert all(
        torch.equal(tensor, saved) for tensor, saved in zip((q, k, v), originals, strict=True)
    )
    assert float((got.float() - exact).norm() / exact.norm()) < 0.05
