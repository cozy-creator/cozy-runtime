"""The Runtime's fused Anima self-attention makes no K/V copy: Diffusers' grouped-query step
repeats by a head-width ratio of 1 in every Cosmos model, and `repeat_interleave(1)` copies
both whole (huaisang, rtx-4070-super: 0.35 s of a 30-step render). CPU, with the rope kernel
replaced by an identity: the attention math around it is the Runtime's own."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cozy_runtime.internal.anima_optimization import _FusedSelfAttention, _RopeMatrix  # noqa: E402


class Attention(torch.nn.Module):  # type: ignore[misc,name-defined]
    heads = 2

    def __init__(self) -> None:
        super().__init__()
        for name in ("to_q", "to_k", "to_v"):
            setattr(self, name, torch.nn.Linear(32, 32, bias=False))
        self.to_out = torch.nn.ModuleList(
            [torch.nn.Linear(32, 32, bias=False), torch.nn.Identity()]
        )
        self.norm_q = SimpleNamespace(weight=torch.ones(16), eps=1e-5)
        self.norm_k = SimpleNamespace(weight=torch.ones(16), eps=1e-5)


class Count(torch.overrides.TorchFunctionMode):  # type: ignore[misc,name-defined]
    def __init__(self) -> None:
        super().__init__()
        self.repeats = 0

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if getattr(func, "__name__", "") == "repeat_interleave":
            self.repeats += 1
        return func(*args, **(kwargs or {}))


def test_the_fused_self_attention_copies_neither_k_nor_v() -> None:
    torch.manual_seed(3)
    attention = Attention().eval()
    kernels = SimpleNamespace(rms_rope_split_half=lambda q, k, *_: (q, k))
    fused = _FusedSelfAttention(torch, kernels, _RopeMatrix(torch))
    x = torch.randn(1, 6, 32)
    rope = torch.zeros(1, 1, 6, 64, 2, 2)
    with torch.no_grad():
        q, k, v = (
            getattr(attention, n)(x).unflatten(2, (2, -1)).transpose(1, 2)
            for n in ("to_q", "to_k", "to_v")
        )
        expected = attention.to_out[0](
            torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2, 3)
        )
        with Count() as count:
            got = fused(attention, x, image_rotary_emb=rope)
    assert count.repeats == 0
    torch.testing.assert_close(got, expected)
