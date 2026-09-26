"""Capability/refusal contract for H3 Turbo bake-once preparation."""

from types import SimpleNamespace
from typing import Any, cast

import pytest


def _linear_root(torch: Any) -> Any:
    root = torch.nn.Module()
    block = torch.nn.Module()
    block.attn = torch.nn.Module()
    block.attn.to_q = torch.nn.Linear(4, 4, bias=False)
    root.transformer_blocks = torch.nn.ModuleList([block])
    return root


def test_filled_encoded_leaf_reports_bake_unsupported() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.models.minimax_h3.turbo import inspect_turbo_composer

    class Encoded(torch.nn.Module):  # type: ignore[name-defined,misc]
        provider = "cozy.fp8-rowwise.native-leaf/1"
        encoding = "fp8-rowwise/1"

    root = _linear_root(torch)
    root.transformer_blocks[0].attn.to_q = Encoded()
    capability = inspect_turbo_composer(root)
    assert not capability.supports_bake
    assert capability.reason_code == "turbo_bake_unsupported"
    assert "cozy.fp8-rowwise.native-leaf/1" in capability.detail
    assert capability.encoded_sites == (
        "transformer_blocks.0.attn.to_q (Encoded cozy.fp8-rowwise.native-leaf/1)",
    )


def test_floating_root_still_refuses_until_full_graph_is_composed() -> None:
    torch = pytest.importorskip("torch")
    from cozy_runtime.models.minimax_h3.turbo import inspect_turbo_composer

    capability = inspect_turbo_composer(_linear_root(torch))
    assert not capability.supports_bake
    assert capability.reason_code == "turbo_bake_unsupported"
    assert capability.linear_sites == ("transformer_blocks.0.attn.to_q",)
    assert "AdaLN tables" in capability.detail


def test_bake_once_refuses_before_overlay_or_request_state_mutation() -> None:
    pytest.importorskip("torch")
    try:
        from cozy_runtime.models.minimax_h3.model import H3TurboLoRA
    except ImportError as exc:
        pytest.skip(f"installed Diffusers does not expose the H3 API: {exc}")
    from cozy_runtime.author import ConformanceError
    from cozy_runtime.models.minimax_h3.turbo import TurboComposerCapability

    calls: list[str] = []
    capability = TurboComposerCapability(
        supports_bake=False,
        reason_code="turbo_bake_unsupported",
        detail="encoded H3 test base cannot be composed",
    )
    adapter = cast(Any, H3TurboLoRA.__new__(H3TurboLoRA))
    adapter.pipe = SimpleNamespace(
        overlay=lambda *_args: calls.append("overlay"),
    )
    base = SimpleNamespace(
        pipe=SimpleNamespace(
            turbo_composer_capability=lambda _trunk: capability,
        )
    )
    state_calls: list[str] = []
    state = SimpleNamespace(
        get=lambda _name: None,
        set=lambda *_args: state_calls.append("set"),
    )
    with pytest.raises(ConformanceError) as raised:
        adapter._sample(
            base,
            "fl2va",
            state,
            on_step=None,
            cancel=None,
            checks=None,
            sol_dense_steps=4,
            bake_once=True,
        )
    assert raised.value.code == "turbo_bake_unsupported"
    assert calls == []
    assert state_calls == []
