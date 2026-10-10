"""CPU registry/shape/policy controls; native SM90 calls are separately GPU-qualified."""

from __future__ import annotations

from typing import Any

import pytest

from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
from cozy_runtime.internal import attention
from cozy_runtime.internal import attention_sage2_fixed as fixed
from cozy_runtime.internal import attention_sol as sol
from cozy_runtime.internal.attention_ulysses import Local
from cozy_runtime.internal.encoding import DeviceFacts


def test_fixed_recipes_are_explicit_sm90_bf16_d128_and_head_local() -> None:
    for name in (fixed.DENSE_NAME, fixed.SOL_NAME):
        candidate = attention.BY_NAME[name]
        assert candidate not in attention.CANDIDATES
        assert candidate.local.head_local
        for sm in (80, 89, 100, 120, 121):
            assert attention._device_admits(
                candidate, DeviceFacts("cuda", "other", sm, "driver", "test")
            )
        assert not attention._device_admits(
            candidate, DeviceFacts("cuda", "H100", 90, "driver", "test")
        )
        assert attention._admits(candidate, "bfloat16", 128)
        assert not attention._admits(candidate, "float16", 128)
        assert not attention._admits(candidate, "bfloat16", 64)
    assert fixed.SOL_NAME in sol.NAMES
    assert sol.dense_identity(fixed.SOL_NAME) == fixed.DENSE_NAME


def test_rectangular_prefix_geometry_is_admitted_without_allocating_full_tensors() -> None:
    torch = pytest.importorskip("torch")
    for qn in (257, 109225):
        q = torch.empty(1, qn, 56, 128, device="meta")
        k = torch.empty(1, 109225, 56, 128, device="meta")
        fixed._shape(q, k, k)
    with pytest.raises(ValueError, match="equal Q/KV heads"):
        fixed._shape(torch.empty(1, 257, 28, 128, device="meta"), k, k)
    with pytest.raises(ValueError, match="equal Q/KV heads"):
        fixed._shape(q, k, torch.empty(1, 109224, 56, 128, device="meta"))


def test_hybrid_dense_steps_and_prefix_use_the_same_fixed_local(monkeypatch: Any) -> None:
    torch = pytest.importorskip("torch")
    calls = []

    def dense(query: Any, key: Any, value: Any, **_options: Any) -> Any:
        calls.append((query.shape[1], key.shape[1]))
        return query + 1

    def wrong(**_options: Any) -> Any:
        raise AssertionError("Unfixed global dense reference was used")

    # The policy is real; CPU stand-ins replace only the two unavailable CUDA kernels.
    monkeypatch.setattr(fixed, "DENSE", Local(dense, lambda *_: "fixed CPU policy stand-in"))
    monkeypatch.setattr(fixed, "validate_device", lambda _device: None)
    monkeypatch.setattr(sol, "_DENSE", wrong)
    monkeypatch.setattr(sol, "specialize", lambda *_args: None)
    monkeypatch.setattr(sol, "_ready", lambda *_args: object())
    monkeypatch.setattr(sol, "_native", lambda _q, _k, v, *_args: v.clone() + 2)
    q = torch.zeros(1, 19, 2, 128, dtype=torch.bfloat16)
    k = q.clone()
    v = q.clone() + 4
    site = sol._SITE.set(sol.Site("dit", "transformer_blocks.2.attn"))
    try:
        for step in (0, 4):
            with attention_scope(AttentionLayout(19, 3, step, 4)), sol.observing() as counts:
                output = fixed.SOL.forward(q, k, v, scale=128**-0.5)
            if step == 0:
                assert torch.equal(output, q + 1) and counts["dense_step"] == 1
            else:
                assert torch.equal(output[:, :3], q[:, :3] + 1)
                assert torch.equal(output[:, 3:], v[:, 3:] + 2)
                assert counts == {"sparse": 1, "dense_step": 0, "dense_path": 0, "dense_prefix": 1}
    finally:
        sol._SITE.reset(site)
    assert calls == [(19, 19), (3, 19)]


def test_hopper_prefers_fixed_hybrid_without_changing_generic_or_blackwell_policy() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from cozy_runtime.models.minimax_h3.model import (
        BLACKWELL_DIT_ATTENTION,
        DIT_ATTENTION,
        HOPPER_DIT_ATTENTION,
    )

    assert DIT_ATTENTION == ("sol-attn", "sageattention", "flash-attn3", "sdpa")
    assert fixed.DENSE_NAME not in BLACKWELL_DIT_ATTENTION
    assert fixed.SOL_NAME not in BLACKWELL_DIT_ATTENTION
    assert (fixed.SOL_NAME, *DIT_ATTENTION) == HOPPER_DIT_ATTENTION
    assert fixed.DENSE_NAME not in HOPPER_DIT_ATTENTION
