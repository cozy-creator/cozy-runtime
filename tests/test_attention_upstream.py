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

from cozy_runtime.internal import accel, attention, attention_ulysses
from cozy_runtime.internal import attention_upstream as upstream

torch = pytest.importorskip("torch")

NONCAUSAL = [upstream.kitchen_int8, upstream.flashinfer_mixed]
ADAPTERS = [*NONCAUSAL, upstream.sage3_fp4]


@pytest.mark.parametrize("adapter", NONCAUSAL)
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


@pytest.mark.parametrize("name", sorted(set(attention.BY_NAME) - {"sol-attn"}))
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


def test_every_registered_kernel_is_head_local_and_a_cross_head_one_refuses_ulysses() -> None:
    """Every Runtime kernel reduces within one head (FlashInfer's scalar V scale is applied per
    head outside the kernel), so every one is context-parallel. The wrapper still refuses a
    kernel that declares a cross-head reduction, before any collective."""
    pytest.importorskip("diffusers")
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    for candidate in attention.BY_NAME.values():
        assert candidate.local.head_local, candidate.name
        member = attention._member(candidate)
        assert _AttentionBackendRegistry._is_context_parallel_available(member), candidate.name
    local = attention_ulysses.Local(lambda *_, **__: None, lambda *_: "", head_local=False)
    member = attention_ulysses.member("_cozy_test_cross_head", local)
    assert not _AttentionBackendRegistry._is_context_parallel_available(member)
    q = torch.zeros((1, 2, 2, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="across heads"):
        _AttentionBackendRegistry._backends[member](
            query=q, key=q, value=q, _parallel_config=object()
        )


def test_sdpa_names_the_backend_torch_dispatches_and_matches_it() -> None:
    q = torch.randn((1, 16, 2, 64))
    options = attention_ulysses.Options(
        attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in (q, q, q))
    ).transpose(1, 2)
    assert torch.equal(upstream.sdpa(q, q, q, **options), expected)
    assert upstream.sdpa_impl(q, q, q, options) in {"flash", "efficient", "math", "cudnn"}
    pytest.importorskip("diffusers")
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    with attention_ulysses.observing() as seen:
        member = attention._member(attention.BY_NAME["sdpa"])

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


def test_fp8_flash_attention_needs_bf16_and_equal_heads() -> None:
    q = torch.zeros((1, 2, 2, 128), dtype=torch.float16)
    with pytest.raises(ValueError, match="requires BF16 inputs"):
        upstream.fa4_fp8(q, q, q)
    q = q.bfloat16()
    for adapter in (upstream.fa3_fp8, upstream.fa4_fp8):
        with pytest.raises(ValueError, match="equal Q/KV head counts"):
            adapter(q, q[:, :, :1], q[:, :, :1], enable_gqa=True)


def test_sage3_refuses_what_upstream_would_silently_change() -> None:
    """Upstream sends D>=256 to SDPA with a print and fixes the scale at D^-1/2."""
    wide = torch.zeros((1, 2, 2, 256), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="does not serve head dimension 256"):
        upstream.sage3_fp4(wide, wide, wide)
    q = torch.zeros((1, 2, 2, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="same CUDA device"):
        upstream.sage3_fp4(q, q, q, scale=128**-0.5)


def test_sage2_selects_upstreams_mode_by_the_querys_card(monkeypatch: pytest.MonkeyPatch) -> None:
    class Query:
        device = torch.device("cuda", 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    assert upstream.SAGE.impl(Query()) == (
        "sageattn_qk_int8_pv_fp8_cuda_sm90(per_thread, fp32+fp32, smooth_k)"
    )
    assert upstream.SAGE_FP16PV.impl(Query()) == (
        "sageattn_qk_int8_pv_fp16_cuda(per_thread, fp32, smooth_k)"
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))
    assert upstream.SAGE.impl(Query()) == (
        "sageattn_qk_int8_pv_fp8_cuda(per_thread, fp32+fp16, smooth_k)"
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (12, 0))
    assert upstream.SAGE.impl(Query()) == (
        "sageattn_qk_int8_pv_fp8_cuda(per_warp, fp32+fp16, smooth_k)"
    )
    with pytest.raises(ValueError, match="no mode for sm120"):
        upstream.SAGE_FP16PV.impl(Query())
    with pytest.raises(ValueError, match="CUDA devices only"):
        upstream.SAGE.impl(torch.zeros(1))


#: adapter -> (module, cards, masks, grouped heads, scale, rel L2 bound)
REAL: dict[Any, tuple[str, tuple[tuple[int, int], ...], bool, bool, float | None, float]] = {
    upstream.kitchen_int8: ("comfy_kitchen", upstream.KITCHEN_SMS, True, True, 0.075, 0.05),
    upstream.flashinfer_mixed: ("flashinfer", ((10, 0), (10, 3)), False, True, 0.075, 0.05),
    upstream.fa4_fp8: ("flash_attn", ((10, 0), (10, 3)), False, False, 0.075, 0.06),
    upstream.sage3_fp4: ("sageattn3", ((12, 0),), False, False, None, 0.2),
}


@pytest.mark.skipif(
    os.environ.get("COZY_TEST_ATTENTION_CUDA") != "1",
    reason="requires an explicitly owned CUDA device and upstream kernel installation",
)
@pytest.mark.parametrize("adapter", list(REAL))
@pytest.mark.parametrize("grouped", [False, True])
def test_real_upstream_matches_sdpa_and_preserves_inputs(
    adapter: Callable[..., Any], grouped: bool
) -> None:
    dependency, cards, masks, groups, scale, bound = REAL[adapter]
    if grouped and not groups:
        pytest.skip("adapter serves equal Q/KV heads only")
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"{dependency} is not installed")
    if not accel.present(torch, "cuda"):
        pytest.skip("CUDA is unavailable")
    capability = torch.cuda.get_device_capability()
    if capability not in cards:
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
    if masks:
        mask = torch.ones((2, 1, 137, 259), device="cuda", dtype=torch.bool)
        mask[0, :, :, 129:] = False
    # A nondefault scale and batch-specific mask expose wrappers that drop options
    # or flatten batch boundaries. Actual upstream kernels run; there are no doubles.
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
    assert float((got.float() - exact).norm() / exact.norm()) < bound
