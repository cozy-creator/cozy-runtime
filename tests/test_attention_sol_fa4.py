"""Sol's route on FA4 (SM100/SM103): the composed output against the same semantics in FP32."""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")


def _blackwell() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires a B200/B300")
    for name in ("flash_attn", "sol_attn", "triton"):
        if importlib.util.find_spec(name) is None:
            pytest.skip(f"requires {name}")


def _reference(q: Any, k: Any, v: Any, scale: float, pooled: Any, kc: Any, vc: Any) -> Any:
    """Exact attention over each row's pooled tiles plus Sol's approximation of every other
    64-token block (block-mean key, block-sum value over its length, mass times length)."""
    _, t, h, _ = q.shape
    n = kc.shape[1]
    length = torch.clamp(t - torch.arange(n, device=q.device) * 64, max=64).float()
    out = torch.empty(t, h, q.shape[-1], device=q.device)
    for head in range(h):
        s = (q[0, :, head].float() @ k[0, :, head].float().T) * scale
        tile_of_row = torch.arange(t, device=q.device) // 256
        exact = pooled[head][tile_of_row][:, torch.arange(t, device=q.device) // 128]
        s = s.masked_fill(~exact, float("-inf"))
        a = (q[0, :, head].float() @ kc[0, :, head].float().T) * scale + length.log()
        a = a.masked_fill(
            pooled[head][tile_of_row][:, torch.arange(n, device=q.device) // 2], float("-inf")
        )
        values = torch.cat([v[0, :, head].float(), vc[0, :, head].float() / length[:, None]])
        p = torch.softmax(torch.cat([s, a], 1), 1)
        out[:, head] = p @ values
    return out[None]


@pytest.mark.parametrize("fp8", [False, True])
def test_sol_on_fa4_matches_its_semantics(fp8: bool) -> None:
    _blackwell()
    from cozy_runtime.internal import attention_sol_fa4

    g = torch.Generator(device="cuda").manual_seed(7)
    t, h, d, prefix = 4000, 2, 128, 300  # tails at 64, 128 and 256
    q, k, v = (
        torch.randn(1, t, h, d, generator=g, device="cuda").to(torch.bfloat16) for _ in range(3)
    )
    scale = d**-0.5
    out = attention_sol_fa4.sparse(q, k, v, scale, prefix, fp8=fp8)
    kc, vc, threshold = __import__("sol_attn.preprocess", fromlist=["_"]).prepare(
        q, k, v, scale=scale, tau=1.0, thresh_type="exact"
    )
    pooled = attention_sol_fa4.route(q, kc, threshold, scale, prefix)
    assert pooled[:, : math.ceil(prefix / 256)].all()
    assert 0 < pooled.float().mean() < 1
    ref = _reference(q, k, v, scale, pooled, kc, vc)
    err = float((out.float() - ref).norm() / ref.norm())
    assert err < (0.05 if fp8 else 0.01), err
