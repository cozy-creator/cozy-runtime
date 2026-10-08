"""Real CPU encoded GEMMs and Turbo hooks; GPU agreement is a separate gate."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from cozy_runtime.internal import attention, attention_sol  # noqa: E402
from cozy_runtime.internal import attention_producer as producer  # noqa: E402
from cozy_runtime.internal.encoding.leaves import (  # noqa: E402
    PreQuantized,
    RowwiseNativeLeaf,
    quantize_activation_rowwise,
)
from cozy_runtime.models.minimax_h3.turbo import LoRAFactors, _LoRAHook  # noqa: E402


def projections(state: str) -> tuple[Any, list[Any], Any]:
    torch.manual_seed(931)
    factors = LoRAFactors(64, 128, 4, 0.25)
    with torch.no_grad():
        factors.lora_down.normal_(std=0.05)
        factors.lora_up.normal_(std=0.05)
    owner = SimpleNamespace(_arming=None)
    if state != "inactive":
        owner._arming = SimpleNamespace(overlay=SimpleNamespace(get_submodule=lambda _: factors))
    sites, hooks = [], []
    for _ in range(3):
        weight = torch.randn(128, 64, dtype=torch.bfloat16) * 0.1
        if state == "baked":
            weight += (factors.lora_up @ factors.lora_down) * factors.scale
        data, scale = quantize_activation_rowwise(torch, weight)
        original = torch.nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
        leaf = RowwiseNativeLeaf("fp8-rowwise/1").leaf(
            torch, {"data": data, "scale": scale}, original, torch.bfloat16
        )
        hook = _LoRAHook(owner, "projection")
        hook._cozy_baked = state == "baked"
        leaf.register_forward_hook(hook)
        sites.append(leaf)
        hooks.append(hook)
    return SimpleNamespace(to_q=sites[0], to_k=sites[1], to_v=sites[2]), hooks, owner


def run(attn: Any, hidden: Any, shared: bool) -> tuple[Any, dict[str, int], int]:
    counts: dict[str, int] = {}
    with (
        torch.inference_mode(),
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile,
    ):
        chunks = list(
            producer.qkv_chunks(attn, hidden, hidden.shape[1], shared=shared, counts=counts)
        )
    quantizations = sum(
        event.count for event in profile.key_averages() if event.key == "aten::amax"
    )
    return torch.cat(chunks), counts, quantizations


@pytest.mark.parametrize("state", ["inactive", "baked", "unbaked"])
def test_shared_quantization_preserves_real_encoded_gemm_and_lora_bits(state: str) -> None:
    attn, hooks, _ = projections(state)
    hidden = torch.randn(1, producer.CHUNK + 65, 64, dtype=torch.bfloat16)
    expected, _, before = run(attn, hidden, False)
    actual, counts, after = run(attn, hidden, True)
    assert torch.equal(actual, expected)
    assert counts == {"shared": 2}
    assert (before, after) == (6, 2)
    assert all(hook._cozy_baked == (state == "baked") for hook in hooks)


def test_shared_operand_keeps_forward_custody_wrapper_and_rebound_roles() -> None:
    attn, _, _ = projections("unbaked")
    hidden = torch.randn(1, 65, 64, dtype=torch.bfloat16)
    visits = []
    for name in ("to_q", "to_k", "to_v"):
        leaf = getattr(attn, name)
        forward = leaf.forward

        def wrapped(*args: Any, saved: Any = forward, site: str = name, **kwargs: Any) -> Any:
            visits.append(site)
            return saved(*args, **kwargs)

        leaf.forward = wrapped
    expected, _, _ = run(attn, hidden, False)
    actual, counts, _ = run(attn, hidden, True)
    assert torch.equal(actual, expected)
    assert visits == ["to_q", "to_k", "to_v"] * 2 and counts == {"shared": 1}
    # A new resident weight version is read from the same original modules.
    attn.to_k.data = torch.zeros_like(attn.to_k.data)
    replaced, _, _ = run(attn, hidden, True)
    reference, _, _ = run(attn, hidden, False)
    assert torch.equal(replaced, reference) and not torch.equal(actual, replaced)


@pytest.mark.parametrize("kind", ["pre", "post", "float"])
def test_semantic_hooks_and_nonencoded_sites_decline_without_changing_output(kind: str) -> None:
    attn, _, _ = projections("unbaked")
    if kind == "pre":
        attn.to_q.register_forward_pre_hook(lambda module, args: (args[0] * 0.5,))
    elif kind == "post":
        attn.to_q.register_forward_hook(lambda module, args, out: out + 1)
    else:
        attn.to_q = torch.nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
    hidden = torch.randn(1, 65, 64, dtype=torch.bfloat16)
    expected, _, before = run(attn, hidden, False)
    actual, counts, after = run(attn, hidden, True)
    assert torch.equal(actual, expected) and after == before
    reason = {
        "pre": "projection-pre-hook",
        "post": "unknown-projection-hook",
        "float": "non-rowwise-leaf",
    }[kind]
    assert counts == {reason: 1}


def test_existing_quantized_carrier_is_reused_without_another_quantization() -> None:
    attn, _, _ = projections("inactive")
    hidden = torch.randn(1, 65, 64, dtype=torch.bfloat16)
    data, scale = quantize_activation_rowwise(torch, hidden)
    operand = PreQuantized(data, scale, tuple(hidden.shape))
    expected, _, _ = run(attn, operand, False)
    actual, counts, calls = run(attn, operand, True)
    assert torch.equal(actual, expected) and calls == 0
    assert counts == {"already-quantized": 1}


def test_global_input_hook_declines_shared_operand_and_keeps_its_effect() -> None:
    attn, _, _ = projections("inactive")
    hidden = torch.randn(1, 65, 64, dtype=torch.bfloat16)
    handle = torch.nn.modules.module.register_module_forward_pre_hook(
        lambda module, args: (args[0] * 0.5,)
    )
    try:
        expected, _, before = run(attn, hidden, False)
        actual, counts, after = run(attn, hidden, True)
    finally:
        handle.remove()
    assert torch.equal(actual, expected) and after == before
    assert counts == {"global-projection-hook": 1}


@pytest.mark.parametrize(
    "name,base",
    [
        ("kitchen-sol-producer-fast-shared-qkv", "kitchen-sol-producer-fast"),
        ("kitchen-sol-producer-fp4-shared-qkv", "kitchen-sol-producer-fp4"),
    ],
)
def test_shared_variants_remain_explicit_and_keep_the_base_attention_policy(
    name: str, base: str
) -> None:
    assert name in producer.REUSE and name in producer.SHARED_QKV and name in attention_sol.NAMES
    assert attention.BY_NAME[name] not in attention.CANDIDATES
    assert producer.LOCALS[name] is producer.LOCALS[base]
    assert attention_sol.dense_identity(name) == attention_sol.dense_identity(base)
    with pytest.raises(attention.AttentionRefusal, match="context parallelism"):
        attention.admit(name, 4)
