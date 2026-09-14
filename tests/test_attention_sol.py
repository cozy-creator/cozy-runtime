"""Model layout and hook tests; native Sol execution is a separate CUDA test."""

from __future__ import annotations

import importlib.util
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from cozy_runtime.author._attention_scope import _ACTIVE_LAYOUT, AttentionLayout, attention_scope
from cozy_runtime.internal import attention_sol


def test_layout_is_explicit_immutable_and_nested_scopes_restore() -> None:
    outer = AttentionLayout(32, 4, 0)
    inner = AttentionLayout(16, 2, 1)
    assert _ACTIVE_LAYOUT.get() is None
    with attention_scope(outer):
        with pytest.raises(FrozenInstanceError):
            outer.step = 1  # type: ignore[misc]
        with pytest.raises(RuntimeError), attention_scope(inner):
            assert _ACTIVE_LAYOUT.get() is inner
            raise RuntimeError("inference failure")
        assert _ACTIVE_LAYOUT.get() is outer
    assert _ACTIVE_LAYOUT.get() is None


@pytest.mark.parametrize("fields", [(0, 0, 0), (3, 4, 0), (3, 0, -1), (True, 0, 0)])
def test_invalid_layout_refuses(fields: tuple[Any, Any, Any]) -> None:
    with pytest.raises(ValueError):
        AttentionLayout(*fields)


def test_real_module_forward_scope_restores_site_after_failure() -> None:
    torch = pytest.importorskip("torch")

    class Failing(torch.nn.Module):  # type: ignore[name-defined, misc]
        def forward(self, value: Any) -> Any:
            assert attention_sol._SITE.get() == attention_sol.Site("dit", "blocks.2.attn")
            raise RuntimeError("forward failure")

    module = Failing()
    attention_sol.install_site(module, "dit", "blocks.2.attn")
    attention_sol.install_site(module, "dit", "blocks.2.attn")
    assert len(module._forward_pre_hooks) == len(module._forward_hooks) == 0
    with pytest.raises(RuntimeError, match="forward failure"):
        module(torch.ones(1))
    assert attention_sol._SITE.get() is None
    with pytest.raises(ValueError, match="another Sol site"):
        attention_sol.install_site(module, "dit", "blocks.3.attn")
    with pytest.raises(ValueError, match="another Sol site"):
        attention_sol.remove_site(module, "dit", "blocks.3.attn")
    assert len(module._forward_pre_hooks) == len(module._forward_hooks) == 0
    attention_sol.remove_site(module, "dit", "blocks.2.attn")
    assert len(module._forward_pre_hooks) == len(module._forward_hooks) == 0
    assert not hasattr(module, "_cozy_sol_site")
    attention_sol.remove_site(module, "dit", "blocks.2.attn")


def _dense_reference(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    import torch

    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale
    ).transpose(1, 2)


@pytest.mark.parametrize("dense_path", [False, True])
def test_dense_warmup_excludes_padding_using_real_attention(dense_path: bool) -> None:
    torch = pytest.importorskip("torch")
    attention_sol.bind_dense(_dense_reference, "test-native-dense")
    generator = torch.Generator().manual_seed(72)
    q, k, v = (torch.randn(1, 16, 2, 128, generator=generator) for _ in range(3))
    layout = AttentionLayout(
        11, 3, 0, dense_until_step=2, dense_paths=("blocks.2",) if dense_path else ()
    )
    site = attention_sol.Site("dit", "blocks.2.attn")
    expected = _dense_reference(query=q[:, :11], key=k[:, :11], value=v[:, :11], scale=None)
    out = attention_sol._execute(q, k, v, None, layout, site)
    torch.testing.assert_close(out[:, :11], expected, rtol=0, atol=0)
    assert torch.count_nonzero(out[:, 11:]) == 0
    # Poisoned padding must not participate in the normalization or output.
    k[:, 11:] = 100
    v[:, 11:] = -100
    again = attention_sol._execute(q, k, v, None, layout, site)
    assert torch.equal(out.view(torch.uint8), again.view(torch.uint8))


def test_observations_are_request_local_and_restore_after_failure() -> None:
    torch = pytest.importorskip("torch")
    attention_sol.bind_dense(_dense_reference, "test-native-dense")
    q = torch.ones(1, 4, 2, 128)
    site = attention_sol.Site("dit", "blocks.2.attn")
    with attention_sol.observing() as outer:
        attention_sol._execute(q, q, q, None, AttentionLayout(4, 0, 0, 10), site)
        with pytest.raises(RuntimeError), attention_sol.observing() as inner:
            attention_sol._execute(
                q, q, q, None, AttentionLayout(4, 0, 0, dense_paths=("blocks.2",)), site
            )
            raise RuntimeError("request failed")
        assert inner == dict(sparse=0, dense_step=0, dense_path=1, dense_prefix=0)
        assert outer == dict(sparse=0, dense_step=1, dense_path=0, dense_prefix=0)
        attention_sol._execute(q, q, q, None, AttentionLayout(4, 0, 1, 10), site)
    assert outer == dict(sparse=0, dense_step=2, dense_path=0, dense_prefix=0)
    assert attention_sol._COUNTS.get() is None
    with attention_sol.observing() as following:
        assert sum(following.values()) == 0


def test_dense_refiner_uses_its_own_short_document() -> None:
    torch = pytest.importorskip("torch")
    attention_sol.bind_dense(_dense_reference, "test-native-dense")
    q = torch.ones(1, 3, 2, 128)
    layout = AttentionLayout(100, 20, 5, dense_paths=("token_refiner",))
    actual = attention_sol._execute(
        q, q, q, None, layout, attention_sol.Site("dit", "token_refiner.attn")
    )
    torch.testing.assert_close(actual, q, rtol=0, atol=0)


def test_missing_semantics_and_cpu_execution_refuse() -> None:
    torch = pytest.importorskip("torch")
    q = torch.ones(1, 4, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="model-owned"):
        attention_sol.sol_attention(q, q, q)
    token = attention_sol._SITE.set(attention_sol.Site("dit", "blocks.2.attn"))
    try:
        with attention_scope(AttentionLayout(4, 0, 0)):
            with pytest.raises(ValueError, match="SM90"):
                attention_sol.sol_attention(q, q, q)
            with pytest.raises(ValueError, match="unmasked"):
                attention_sol.sol_attention(q, q, q, attn_mask=torch.ones(4, 4))
            with pytest.raises(ValueError, match="parallelism"):
                attention_sol.sol_attention(q, q, q, _parallel_config=object())
    finally:
        attention_sol._SITE.reset(token)


def test_native_sol_preserves_protected_queries_and_input_bytes() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires an owned SM90 CUDA qualification worker")
    if importlib.util.find_spec("sol_attn") is None:
        pytest.skip("requires the declared upstream sol-attn package")
    attention_sol.bind_dense(_dense_reference, "test-native-dense")
    generator = torch.Generator(device="cuda").manual_seed(88)
    q, k, v = (
        torch.randn(1, 512, 2, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    original = [t.clone() for t in (q, k, v)]
    token = attention_sol._SITE.set(attention_sol.Site("dit", "blocks.2.attn"))
    try:
        with attention_sol.observing() as counts, attention_scope(AttentionLayout(501, 33, 10)):
            output = attention_sol.sol_attention(q, k, v)
    finally:
        attention_sol._SITE.reset(token)
    expected = _dense_reference(query=q[:, :33], key=k[:, :501], value=v[:, :501], scale=None)
    torch.testing.assert_close(output[:, :33], expected, rtol=0, atol=0)
    assert torch.count_nonzero(output[:, 501:]) == 0
    assert torch.isfinite(output).all()
    assert counts == dict(sparse=1, dense_step=0, dense_path=0, dense_prefix=1)
    for tensor, before in zip((q, k, v), original, strict=True):
        assert torch.equal(tensor.view(torch.uint8), before.view(torch.uint8))
