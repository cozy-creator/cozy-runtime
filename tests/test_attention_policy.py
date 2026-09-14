"""Real Diffusers processor selection; CPU execution where attention arithmetic is used."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from cozy_runtime.author import AttentionContext, Model
from cozy_runtime.internal import attention
from cozy_runtime.internal.attention_registry import register
from cozy_runtime.internal.encoding import DeviceFacts
from cozy_runtime.internal.executor import Executor


def _device() -> DeviceFacts:
    return DeviceFacts(kind="cpu", name="CPU", sm=0, driver="", configuration="", index=0)


def _roots() -> dict[str, Any]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    return {
        name: torch.nn.ModuleList(
            [WanAttention(dim=8, heads=1, dim_head=8, processor=WanAttnProcessor())]
        )
        for name in ("dit", "vae")
    }


def _ready(*names: str) -> list[attention.Ready]:
    return [
        attention.Ready(attention.BY_NAME[name], attention._member(attention.BY_NAME[name].backend))
        for name in names
    ]


def _choices(roots: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        getattr(processor, "_attention_backend", None)
        for _, _, processor in attention._sites(roots)
    )


def test_author_context_is_immutable_and_model_policy_executes() -> None:
    class Policy(Model[object]):
        def choose_attention(self, context: AttentionContext) -> str | None:
            assert context.component in ("dit", "vae") and context.module_path == "0"
            assert context.device_kind == "cpu" and context.sm == 0
            assert context.activation_dtype == "float32" and context.head_dim == 8
            assert context.eligible == ("sdpa",) and context.recommended == "sdpa"
            with pytest.raises(FrozenInstanceError):
                context.degree = 99  # type: ignore[misc]
            return "sdpa"

    policy = object.__new__(Policy)
    roots = _roots()
    applied = attention.select(_device(), roots, choose=policy.choose_attention)
    assert applied.totals() == {"sdpa": 2}
    import torch

    with torch.no_grad():
        assert roots["dit"][0](torch.ones(1, 2, 8)).isfinite().all()


def test_invalid_author_choice_leaves_every_processor_unchanged() -> None:
    roots = _roots()
    before = _choices(roots)

    def choose(context: AttentionContext) -> str:
        return "sdpa" if context.component == "dit" else "kitchen-int8"

    with pytest.raises(attention.AttentionRefusal, match="eligible"):
        attention.select(_device(), roots, choose=choose)
    assert _choices(roots) == before


def test_scoped_pin_and_snapshot_restore_different_prepared_policy() -> None:
    roots = _roots()
    # Installation-only control: real members, no claim that FA3 executes on CPU.
    for root in roots.values():
        root.to(dtype=__import__("torch").bfloat16)
    attention.install(_ready("flash-attn3"), roots)
    prepared = attention.snapshot(roots)
    before = _choices(roots)
    applied = attention.plan(_device(), roots, "dit=sdpa").apply()
    assert applied.hosts == {"dit": {"sdpa": 1}} and applied.pin == "dit=sdpa"
    assert _choices(roots)[0] == "native" and _choices(roots)[1] == before[1]
    prepared.restore()
    assert _choices(roots) == before


def test_scoped_prepare_pin_preserves_other_components_and_rejects_typos() -> None:
    roots = _roots()
    seen: list[str] = []

    def choose(context: AttentionContext) -> None:
        seen.append(context.component)

    applied = attention.select(_device(), roots, "dit=sdpa", choose=choose)
    assert applied.hosts == {"dit": {"sdpa": 1}, "vae": {"sdpa": 1}}
    assert seen == ["vae"]
    for pin in ("missing=sdpa", "=sdpa", "dit=", "dit=sdpa=bad"):
        with pytest.raises(attention.AttentionRefusal):
            attention.plan(_device(), roots, pin)


def test_multi_model_scope_and_restore_cover_all_children(tmp_path: Path) -> None:
    children = [Executor(None, tmp_path / str(i)) for i in range(2)]  # type: ignore[arg-type]
    all_roots = {}
    for i, child in enumerate(children):
        roots = _roots()
        attention.install(_ready("sdpa"), roots)
        child.attention_defaults = attention.snapshot(roots)
        all_roots.update({f"model{i}/{name}": root for name, root in roots.items()})
    assert len(attention.plan(_device(), all_roots, "model0/dit=sdpa").sites) == 1
    assert len(attention.plan(_device(), all_roots, "dit=sdpa").sites) == 2
    assert len(attention.plan(_device(), all_roots, "sdpa").sites) == 4
    parent = Executor(None, tmp_path / "parent")  # type: ignore[arg-type]
    parent.model_executors = children
    for _, _, processor in attention._sites(all_roots):
        processor._attention_backend = "stale-request"
    parent._restore_attention()
    assert _choices(all_roots) == ("native",) * 4


def test_scoped_pin_preserves_compiled_and_parallel_fences() -> None:
    roots = _roots()
    roots["dit"][0]._compiled_call_impl = lambda: None
    with pytest.raises(attention.AttentionRefusal) as exc:
        attention.plan(_device(), roots, "dit=sdpa")
    assert exc.value.code == "attention_kernel_frozen"
    attention.plan(_device(), roots, "vae=sdpa")
    with pytest.raises(attention.AttentionRefusal, match="context-parallel"):
        attention.plan(_device(), roots, "vae=sdpa", "a context-parallel group is installed")


def test_registration_is_idempotent_local_and_dispatches_real_cpu_attention() -> None:
    torch = pytest.importorskip("torch")
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry, dispatch_attention_fn

    def cpu_attention(
        query: Any,
        key: Any,
        value: Any,
        attn_mask: Any = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        _parallel_config: Any = None,
    ) -> Any:
        return torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        ).transpose(1, 2)

    active = _AttentionBackendRegistry.get_active_backend()
    member = register("_cozy_cpu_policy_test", cpu_attention)
    assert register("_cozy_cpu_policy_test", cpu_attention) is member
    with pytest.raises(ValueError, match="another implementation"):
        register("_cozy_cpu_policy_test", lambda: None)
    q = torch.randn(1, 4, 2, 8)
    mask = torch.tensor([[True, True, False, False]])
    got = dispatch_attention_fn(q, q, q, backend=member, attn_mask=mask, scale=0.3)
    assert torch.equal(got, cpu_attention(q, q, q, attn_mask=mask, scale=0.3))
    assert not attention._context_parallel(member)
    assert _AttentionBackendRegistry.get_active_backend() == active


def test_new_approximate_backends_remain_explicit_and_architecture_exact() -> None:
    for name, sms in [("kitchen-int8", {90, 100, 103}), ("flashinfer-bf16-fp8", {100, 103})]:
        candidate = attention.BY_NAME[name]
        assert candidate in attention.PINNABLE and candidate not in attention.CANDIDATES
        assert candidate.sms == sms
    assert not attention._admits(attention.BY_NAME["flashinfer-bf16-fp8"], "bfloat16", 64)
    assert attention._admits(attention.BY_NAME["flashinfer-bf16-fp8"], "bfloat16", 128)


def test_model_qualified_preparation_pin_routes_to_one_construction() -> None:
    assert (
        attention.for_model("base_model/dit=sageattention", ["base_model"]) == "dit=sageattention"
    )
    assert attention.for_model("base_model/dit=sageattention", ["adapter"]) == ""
    assert attention.for_model("dit=sageattention", ["adapter"]) == "dit=sageattention"
    assert attention.for_model("sdpa", ["base_model"]) == "sdpa"


def test_explicit_install_rejects_all_sites_before_any_change() -> None:
    roots = _roots()
    torch = pytest.importorskip("torch")
    roots["dit"].to(dtype=torch.bfloat16)
    original = _choices(roots)
    with pytest.raises(attention.AttentionRefusal):
        attention.install(_ready("flash-attn3-fp8"), roots, "flash-attn3-fp8")
    assert _choices(roots) == original
