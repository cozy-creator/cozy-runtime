"""A processor that bypasses backend dispatch is never advertised as a controllable site.

Diffusers' `AttnProcessor2_0` is the exception Runtime removes rather than tolerates: `adopt`
replaces it with `ClassicSdpa`, which really dispatches. Any other legacy processor (here
`AttnProcessor`) stays untouched and refused."""

from __future__ import annotations

from typing import Any

import pytest

from cozy_runtime.author import AttentionContext
from cozy_runtime.internal import attention
from cozy_runtime.internal.attention_registry import register
from cozy_runtime.internal.encoding import DeviceFacts


def _device() -> DeviceFacts:
    return DeviceFacts(kind="cpu", name="CPU", sm=0, driver="", configuration="", index=0)


def _modules() -> tuple[Any, Any]:
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("diffusers.models.attention_processor")
    wan = pytest.importorskip("diffusers.models.transformers.transformer_wan")
    legacy = module.Attention(query_dim=8, heads=1, dim_head=8, processor=module.AttnProcessor())
    managed = wan.WanAttention(dim=8, heads=1, dim_head=8, processor=wan.WanAttnProcessor())
    assert not hasattr(legacy.processor, "_attention_backend")
    assert hasattr(managed.processor, "_attention_backend")
    return legacy, torch.nn.ModuleDict({"managed": managed, "legacy": legacy})


def test_legacy_processor_bypasses_registry_but_managed_control_does_not() -> None:
    torch = pytest.importorskip("torch")
    dispatch = pytest.importorskip("diffusers.models.attention_dispatch")
    legacy, mixed = _modules()
    query = torch.randn(1, 4, 8)
    with torch.no_grad():
        baseline = legacy(query)

    def reject_registry_call(query: Any, key: Any, value: Any) -> Any:
        raise RuntimeError("registry backend was called")

    member = register("_cozy_legacy_coverage_test", reject_registry_call)
    with torch.no_grad(), dispatch.attention_backend(member):
        assert torch.equal(legacy(query), baseline)
        with pytest.raises(RuntimeError, match="registry backend was called"):
            mixed["managed"](query)


def test_an_adopted_classic_sdpa_site_really_dispatches_through_the_registry() -> None:
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("diffusers.models.attention_processor")
    dispatch = pytest.importorskip("diffusers.models.attention_dispatch")
    classic = module.Attention(
        query_dim=8, heads=1, dim_head=8, processor=module.AttnProcessor2_0()
    )
    query = torch.randn(1, 4, 8)
    with torch.no_grad():
        baseline = classic(query)
    attention.adopt({"classic": classic})
    assert type(classic.processor) is attention.ClassicSdpa

    def reject_registry_call(query: Any, key: Any, value: Any, **_: Any) -> Any:
        raise RuntimeError("registry backend was called")

    member = register("_cozy_classic_coverage_test", reject_registry_call)
    with (
        torch.no_grad(),
        dispatch.attention_backend(member),
        pytest.raises(RuntimeError, match="registry backend was called"),
    ):
        classic(query)
    applied = attention.select(_device(), {"classic": classic})
    assert applied.totals() == {"sdpa": 1}
    with torch.no_grad():
        assert torch.equal(classic(query), baseline)


def test_default_keeps_legacy_untouched_and_does_not_report_a_selection() -> None:
    legacy, mixed = _modules()
    applied = attention.select(_device(), {"mixed": mixed})
    assert applied.totals() == {"sdpa": 1}
    assert not hasattr(legacy.processor, "_attention_backend")
    assert len(tuple(attention._sites({"mixed": mixed}))) == 1
    assert len(attention.snapshot({"mixed": mixed}).choices) == 1
    assert attention.observed({"legacy": legacy}).totals() == {}


@pytest.mark.parametrize("operation", ["select", "install", "check", "plan"])
def test_mixed_scope_explicit_pin_refuses_partial_coverage(operation: str) -> None:
    legacy, mixed = _modules()
    before = mixed["managed"].processor._attention_backend
    roots = {"mixed": mixed}
    with pytest.raises(attention.AttentionRefusal, match="per-site"):
        if operation == "install":
            attention.install(attention.pinned("sdpa", _device()), roots, "mixed=sdpa")
        else:
            getattr(attention, operation)(_device(), roots, "mixed=sdpa")
    assert not hasattr(legacy.processor, "_attention_backend")
    assert mixed["managed"].processor._attention_backend is before


def test_author_sees_empty_choices_for_legacy_and_cannot_force_it() -> None:
    legacy, mixed = _modules()
    contexts: list[AttentionContext] = []

    def abstain(context: AttentionContext) -> None:
        contexts.append(context)

    attention.select(_device(), {"mixed": mixed}, choose=abstain)
    context = next(c for c in contexts if c.module_path == "legacy")
    assert context.eligible == () and context.recommended == ""
    assert "per-site" in context.recommendation_reason
    assert not hasattr(legacy.processor, "_attention_backend")

    def force(context: AttentionContext) -> str | None:
        return "sdpa" if context.module_path == "legacy" else None

    with pytest.raises(attention.AttentionRefusal, match="per-site"):
        attention.select(_device(), {"mixed": mixed}, choose=force)
    assert not hasattr(legacy.processor, "_attention_backend")


def test_scoped_pin_does_not_require_control_of_an_unselected_legacy_component() -> None:
    legacy, mixed = _modules()
    roots = {"dit": mixed["managed"], "vae": legacy}
    applied = attention.select(_device(), roots, "dit=sdpa")
    assert applied.hosts == {"dit": {"sdpa": 1}}
    assert not hasattr(legacy.processor, "_attention_backend")
    with pytest.raises(attention.AttentionRefusal, match="per-site"):
        attention.plan(_device(), roots, "vae=sdpa")
