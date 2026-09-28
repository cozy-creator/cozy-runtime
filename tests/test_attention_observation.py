"""Request attention observation and restore through actual Diffusers processors."""

from __future__ import annotations

import pytest

from cozy_runtime.internal import attention
from cozy_runtime.internal.encoding import DeviceFacts


def test_request_artifact_observation_and_restore_use_actual_diffusers_processors() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    module = WanAttention(dim=128, heads=1, dim_head=128, processor=WanAttnProcessor())
    roots = {"dit": module}
    device = DeviceFacts("cpu", "CPU", 0, "", "", 0)
    attention.select(device, roots, "sdpa")
    defaults = attention.snapshot(roots)
    before = module.processor._attention_backend
    entry = attention.Ready(
        attention.BY_NAME["sdpa"],
        attention._member(attention.BY_NAME["sdpa"]),
        artifact={"artifact": "fixture", "key": "abc", "compile_ms": "1.0"},
    )
    swap = attention.Swap(entry, (("dit", module.processor),), "dit=sdpa")
    value = torch.ones(1, 4, 128)
    with torch.no_grad():
        baseline = module(value)
        with (
            pytest.raises(RuntimeError, match="handler failed"),
            attention.override(defaults, roots, swap, "dit=sdpa") as applied,
        ):
            assert applied.document()["artifacts"]["sdpa"] == entry.artifact
            assert torch.equal(module(value), baseline)
            raise RuntimeError("handler failed")
        assert module.processor._attention_backend == before
        with attention.override(defaults, roots, None) as restored:
            assert restored.artifacts is None
            assert torch.equal(module(value), baseline)
