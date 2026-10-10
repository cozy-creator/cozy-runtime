"""CPU ownership/delegation control; actual fused CUDA arithmetic has a separate gate."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from cozy_runtime.internal import fusion, fusion_install  # noqa: E402


def test_attention_result_is_released_before_ffn_without_changing_block_value(monkeypatch):
    references = []

    def attention(value, rotary, mask):
        result = value + 1
        references.append(weakref.ref(result))
        return result

    def next_norm(residual, gate, branch, *args, **kwargs):
        # Real CPU tensor arithmetic with the production helper's owned-output
        # contract. The CUDA helper allocates hidden/out separately in _launch_norm_mod.
        hidden = residual + gate * branch
        normed = hidden * 3
        assert not torch._C._is_alias_of(hidden, branch)
        assert not torch._C._is_alias_of(normed, branch)
        return hidden, normed

    def first_ffn(value):
        assert references[-1]() is None, "completed attention result retained into FFN"
        return value + 4

    monkeypatch.setattr(fusion, "rmsnorm_modulate", lambda value, *args, **kwargs: value * 2)
    monkeypatch.setattr(fusion, "gate_add_rmsnorm_modulate", next_norm)
    monkeypatch.setattr(
        fusion, "gate_add", lambda hidden, gate, value, index: hidden + gate * value
    )
    norm = SimpleNamespace(weight=torch.ones(8), eps=1e-5)
    block = SimpleNamespace(
        norm1=norm,
        norm2=norm,
        attn=attention,
        adaln_proj=lambda _: (0, 0, 2, 0, 0, 3),
        ff=SimpleNamespace(net=[first_ffn, None, lambda value: value * 5]),
        _fusion_attn_fp8=False,
        _fusion_ff_fp8=False,
    )
    source = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    original = source.clone()
    with torch.inference_mode():
        result = fusion_install.FusedMiniMaxH3TransformerBlock.forward(
            block, source, None, None, None
        )
    assert torch.equal(result, source * 230 + 152)
    assert torch.equal(source, original)
    assert references[0]() is None
