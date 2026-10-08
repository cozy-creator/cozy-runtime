"""CPU policy/registration contracts; this is not INT8 GPU accuracy or speed proof."""

from __future__ import annotations

from typing import Any

import pytest

from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
from cozy_runtime.internal import attention, attention_kitchen_sol, attention_sol
from cozy_runtime.internal.encoding import DeviceFacts

torch = pytest.importorskip("torch")


def test_real_kitchen_capability_api_refuses_without_a_cuda_runtime() -> None:
    pytest.importorskip("comfy_kitchen")
    if torch.version.cuda is not None:
        pytest.skip("requires the CPU-only Torch validation environment")
    device = DeviceFacts(
        kind="cuda", name="unavailable", sm=120, driver="", configuration="", index=0
    )
    with pytest.raises(attention.AttentionRefusal, match="requires compiled Sol"):
        attention._resolve(attention.BY_NAME["kitchen-sol"], device)


def dense(*, query: Any, key: Any, value: Any, scale: float | None) -> Any:
    return torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2)


def test_kitchen_sol_remains_explicit_and_keeps_ulysses_registration() -> None:
    pytest.importorskip("diffusers")
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    candidate = attention.BY_NAME["kitchen-sol"]
    assert candidate in attention.PINNABLE and candidate not in attention.CANDIDATES
    member = attention._member(candidate)
    assert _AttentionBackendRegistry._is_context_parallel_available(member)
    module = WanAttention(dim=256, heads=2, dim_head=128, processor=WanAttnProcessor())
    roots = {"dit": torch.nn.ModuleList([module]).to(dtype=torch.bfloat16)}
    applied = attention.install(
        [attention.Ready(candidate, member)], roots, "dit=kitchen-sol", degree=4
    )
    assert applied.totals() == {"kitchen-sol": 1}
    assert module._cozy_sol_site.site == attention_sol.Site("dit", "0")
    assert attention_sol.dense_identity("kitchen-sol") == "kitchen-int8"
    attention_sol.remove_site(module, "dit", "0")


@pytest.mark.parametrize("options", [{"is_causal": True}, {"dropout_p": 0.1}, {"enable_gqa": True}])
def test_kitchen_sol_refuses_unprovided_model_semantics_and_cpu(options: dict[str, Any]) -> None:
    q = torch.ones(1, 4, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="model-owned"):
        attention_kitchen_sol.forward(q, q, q)
    token = attention_sol._SITE.set(attention_sol.Site("dit", "blocks.2.attn"))
    try:
        with attention_scope(AttentionLayout(4, 0, 10)):
            with pytest.raises(ValueError, match="unmasked"):
                attention_kitchen_sol.forward(q, q, q, **options)
            with pytest.raises(ValueError, match="CuTe routes only"):
                attention_kitchen_sol.forward(q, q, q)
    finally:
        attention_sol._SITE.reset(token)


def test_actual_kitchen_reference_keeps_runtime_prefix_padding_and_step_policy() -> None:
    pytest.importorskip("comfy_kitchen")
    generator = torch.Generator().manual_seed(73)
    q, k, v = (torch.randn(1, 261, 4, 128, generator=generator).bfloat16() for _ in range(3))
    before = [x.clone() for x in (q, k, v)]
    site = attention_sol.Site("dit", "transformer_blocks.2.attn")
    layout = AttentionLayout(257, 65, 10, dense_until_step=10)
    reference = dense(query=q[:, :257], key=k[:, :257], value=v[:, :257], scale=None)
    with attention_sol.observing() as counts:
        result = attention_sol._execute(
            q,
            k,
            v,
            None,
            layout,
            site,
            sparse=attention_kitchen_sol.sparse,
            dense=dense,
        )
    assert counts == dict(sparse=1, dense_step=0, dense_path=0, dense_prefix=1)
    torch.testing.assert_close(result[:, :65], reference[:, :65], rtol=0, atol=0)
    assert torch.count_nonzero(result[:, 257:]) == 0
    for original, retained in zip((q, k, v), before, strict=True):
        assert torch.equal(original, retained)
    # The real upstream sparse reference is used outside the protected prefix.
    assert not torch.equal(result[:, 65:257], reference[:, 65:257])
    k[:, 257:], v[:, 257:] = 100, -100
    repeated = attention_sol._execute(
        q,
        k,
        v,
        None,
        layout,
        site,
        sparse=attention_kitchen_sol.sparse,
        dense=dense,
    )
    assert torch.equal(result, repeated)
    # Initial dense steps and the first two blocks keep their existing policy.
    for step, paths, expected_kind in [
        (9, (), "dense_step"),
        (10, ("transformer_blocks.2",), "dense_path"),
    ]:
        with attention_sol.observing() as counts:
            result = attention_sol._execute(
                q,
                k,
                v,
                None,
                AttentionLayout(257, 65, step, 10, paths),
                site,
                sparse=attention_kitchen_sol.sparse,
                dense=dense,
            )
        torch.testing.assert_close(result[:, :257], reference, rtol=0, atol=0)
        assert counts[expected_kind] == 1 and counts["sparse"] == 0


def test_actual_kitchen_reference_is_head_local_for_four_and_eight_partitions() -> None:
    pytest.importorskip("comfy_kitchen")
    generator = torch.Generator().manual_seed(74)
    q, k, v = (torch.randn(1, 129, 56, 128, generator=generator).bfloat16() for _ in range(3))
    expected = attention_kitchen_sol.sparse(q, k, v, None, 13)
    for degree in (4, 8):
        parts = [
            attention_kitchen_sol.sparse(qs, ks, vs, None, 13)
            for qs, ks, vs in zip(
                q.chunk(degree, dim=2),
                k.chunk(degree, dim=2),
                v.chunk(degree, dim=2),
                strict=True,
            )
        ]
        torch.testing.assert_close(torch.cat(parts, dim=2), expected, rtol=0, atol=0)
